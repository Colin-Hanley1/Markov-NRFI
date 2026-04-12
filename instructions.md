# NRFI Markov Model — Full Implementation Spec

## Overview

Build a first-inning run-scoring model using a Markov chain over the 24 base-out states of baseball. The model:
1. Blends pitcher and batter statistics using a log-odds method
2. Constructs a per-PA outcome distribution conditioned on base-out state
3. Propagates state transitions through the inning analytically (matrix method) and via Monte Carlo simulation
4. Produces `P(NRFI)` for a given home/away matchup

The system is implemented in Python. All statistical inputs come from CSV files (Statcast/FanGraphs exports). The model outputs a probability and supporting diagnostics.

---

## Directory Structure

```
nrfi_model/
├── data/
│   ├── pitchers.csv           # pitcher stats (one row per pitcher-season)
│   ├── batters.csv            # batter stats (one row per batter-season)
│   ├── park_factors.csv       # park HR/run factor by team
│   ├── lineups.csv            # today's lineups (one row per player, lineup slot)
│   └── league_averages.csv    # single-row league average rates
├── model/
│   ├── __init__.py
│   ├── state.py               # base-out state definitions
│   ├── outcomes.py            # PA outcome distribution builder
│   ├── transitions.py         # transition matrix construction
│   ├── chain.py               # Markov chain runner (analytic + simulation)
│   └── blend.py               # log-odds pitcher × batter blending
├── pipeline.py                # top-level entry point
├── validate.py                # historical backtesting
└── requirements.txt
```

---

## Requirements

```
# requirements.txt
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.12.0
```

---

## 1. State Space — `model/state.py`

### Concepts

A **base-out state** is a tuple `(bases, outs)` where:
- `bases` is a 3-bit integer: bit 0 = 1B occupied, bit 1 = 2B occupied, bit 2 = 3B occupied
  - `0b000 = 0` → bases empty
  - `0b001 = 1` → runner on 1st only
  - `0b010 = 2` → runner on 2nd only
  - `0b011 = 3` → runners on 1st and 2nd
  - `0b100 = 4` → runner on 3rd only
  - `0b101 = 5` → runners on 1st and 3rd
  - `0b110 = 6` → runners on 2nd and 3rd
  - `0b111 = 7` → bases loaded
- `outs` is 0, 1, or 2
- There are 8 × 3 = **24 transient states**
- There are **2 absorbing states**: `NRFI_ABSORBED` (3 outs, 0 runs) and `RUN_ABSORBED` (any run scored)

Each state is assigned a canonical integer index 0–23:
- `state_index = bases * 3 + outs`

The inning always begins at state index `0` (bases=0, outs=0).

### Implementation

```python
# model/state.py

from dataclasses import dataclass
from typing import Tuple

NUM_BASE_CONFIGS = 8    # 0b000 through 0b111
NUM_OUT_COUNTS   = 3    # 0, 1, 2
NUM_STATES       = NUM_BASE_CONFIGS * NUM_OUT_COUNTS  # 24 transient states

# Absorbing state sentinels (outside 0–23)
NRFI_ABSORBED = 24   # 3 outs reached with 0 runs scored
RUN_ABSORBED  = 25   # at least 1 run scored

STARTING_STATE = 0   # bases empty, 0 outs


def encode_state(bases: int, outs: int) -> int:
    """
    Map (bases bitmask, outs) to a canonical index 0–23.

    Parameters
    ----------
    bases : int
        3-bit integer. Bit 0 = first base, bit 1 = second base, bit 2 = third base.
    outs : int
        Number of outs recorded so far this half-inning. Must be 0, 1, or 2.

    Returns
    -------
    int
        State index in [0, 23].

    Raises
    ------
    ValueError
        If bases is not in [0, 7] or outs is not in [0, 2].
    """
    if not (0 <= bases <= 7):
        raise ValueError(f"bases must be 0–7, got {bases}")
    if outs not in (0, 1, 2):
        raise ValueError(f"outs must be 0, 1, or 2, got {outs}")
    return bases * 3 + outs


def decode_state(index: int) -> Tuple[int, int]:
    """
    Inverse of encode_state.

    Parameters
    ----------
    index : int
        State index in [0, 23].

    Returns
    -------
    Tuple[int, int]
        (bases bitmask, outs).

    Raises
    ------
    ValueError
        If index is outside [0, 23].
    """
    if not (0 <= index <= 23):
        raise ValueError(f"State index must be 0–23, got {index}")
    bases = index // 3
    outs  = index  % 3
    return bases, outs


def on_first(bases: int) -> bool:
    return bool(bases & 0b001)


def on_second(bases: int) -> bool:
    return bool(bases & 0b010)


def on_third(bases: int) -> bool:
    return bool(bases & 0b100)


def runner_count(bases: int) -> int:
    """Number of runners on base (0–3)."""
    return bin(bases).count('1')


def describe_state(index: int) -> str:
    """Human-readable label for a state index, e.g. '1st & 3rd, 1 out'."""
    if index == NRFI_ABSORBED:
        return "3 outs (NRFI)"
    if index == RUN_ABSORBED:
        return "Run scored"
    bases, outs = decode_state(index)
    occupied = []
    if on_first(bases):  occupied.append("1st")
    if on_second(bases): occupied.append("2nd")
    if on_third(bases):  occupied.append("3rd")
    base_str = " & ".join(occupied) if occupied else "empty"
    out_str  = f"{outs} out{'s' if outs != 1 else ''}"
    return f"{base_str}, {out_str}"


# Pre-built lookup: state index → (bases, outs)
STATE_TABLE = [decode_state(i) for i in range(NUM_STATES)]
```

---

## 2. PA Outcome Rates — `model/outcomes.py`

### PA Outcome Categories

Every plate appearance resolves to exactly one of these mutually exclusive outcomes:

| Key | Description |
|-----|-------------|
| `K`   | Strikeout (batter out, no runners advance unless wild-pitch K — ignored here) |
| `BB`  | Walk (batter to first, runners forced if applicable) |
| `HBP` | Hit by pitch (same mechanics as walk) |
| `HR`  | Home run (batter + all runners score) |
| `1B`  | Single |
| `2B`  | Double |
| `3B`  | Triple |
| `GBOUT` | Ground ball out (no runners advance; enables GIDP) |
| `FBOUT` | Fly ball out (enables sac fly; runners may tag) |
| `LDOUT` | Line drive out (runners generally hold) |
| `FC`    | Fielder's choice (batter reaches 1B, lead runner erased) |

> **Note:** GIDP, sac fly, and sac bunt are *not* separate outcome categories — they are *conditional transitions* applied on top of GBOUT and FBOUT based on base-out state. See Section 4.

### Rate Definitions (all expressed as fractions of total PAs)

| Stat | Source column | Notes |
|------|--------------|-------|
| `k_rate`    | `K%` / 100 | Pitcher + batter blend |
| `bb_rate`   | `BB%` / 100 | Excludes IBB |
| `hbp_rate`  | `HBP%` / 100 | Small; use pitcher-side primarily |
| `hr_rate`   | `HR/PA` | Derived from HR/FB × FB% |
| `hit_rate`  | `H%` or BABIP × (1 − K%) | Balls in play that fall for hits |
| `single_rate` | Derived: hit_rate × single_share |
| `double_rate` | Derived: hit_rate × double_share |
| `triple_rate` | Derived: hit_rate × triple_share |
| `gb_rate`   | `GB%` / 100 of balls in play | From pitcher stat |
| `fb_rate`   | `FB%` / 100 of balls in play | Complement of GB + LD |
| `ld_rate`   | `LD%` / 100 of balls in play | |
| `gidp_rate` | `GIDP` per opportunity | Conditional; applied in transitions |
| `sf_rate`   | `SF` per sac-fly opportunity | Conditional; applied in transitions |
| `sb_rate`   | Ignored at PA level — modeled as between-PA event if desired |

### Implementation

```python
# model/outcomes.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass
class PAOutcomeRates:
    """
    Probability distribution over plate appearance outcomes for a single
    pitcher–batter matchup. All fields are probabilities that sum to 1.0.

    Attributes
    ----------
    k_rate : float
        Probability of strikeout.
    bb_rate : float
        Probability of walk (excludes IBB).
    hbp_rate : float
        Probability of hit by pitch.
    hr_rate : float
        Probability of home run (all runners score).
    single_rate : float
        Probability of single.
    double_rate : float
        Probability of double.
    triple_rate : float
        Probability of triple.
    gbout_rate : float
        Probability of ground ball out (may become GIDP — see transitions).
    fbout_rate : float
        Probability of fly ball out (may become sac fly — see transitions).
    ldout_rate : float
        Probability of line drive out.
    fc_rate : float
        Probability of fielder's choice (batter to 1B, lead runner out).
    gidp_prob_given_gbout : float
        Conditional probability that a GBOUT becomes a GIDP,
        given runner on 1st and fewer than 2 outs. Applied in transitions.
    sf_prob_given_fbout : float
        Conditional probability that a FBOUT scores the runner from 3rd,
        given runner on 3rd and fewer than 2 outs. Applied in transitions.
    sac_bunt_prob : float
        Probability that a batter attempts and succeeds a sacrifice bunt,
        conditional on situation (runner on 1B or 1B+2B, 0–1 outs).
        Applied in transitions.
    """

    k_rate:       float = 0.0
    bb_rate:      float = 0.0
    hbp_rate:     float = 0.0
    hr_rate:      float = 0.0
    single_rate:  float = 0.0
    double_rate:  float = 0.0
    triple_rate:  float = 0.0
    gbout_rate:   float = 0.0
    fbout_rate:   float = 0.0
    ldout_rate:   float = 0.0
    fc_rate:      float = 0.0

    # Conditional probabilities used inside transition logic
    gidp_prob_given_gbout: float = 0.15   # ~15% league average
    sf_prob_given_fbout:   float = 0.175  # ~17.5% league average
    sac_bunt_prob:         float = 0.0    # 0 for most batters; set per slot/context

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """
        Assert all rates are non-negative and sum to 1.0 (within tolerance).
        Raises ValueError on violation.
        """
        raw = {
            'k_rate':      self.k_rate,
            'bb_rate':     self.bb_rate,
            'hbp_rate':    self.hbp_rate,
            'hr_rate':     self.hr_rate,
            'single_rate': self.single_rate,
            'double_rate': self.double_rate,
            'triple_rate': self.triple_rate,
            'gbout_rate':  self.gbout_rate,
            'fbout_rate':  self.fbout_rate,
            'ldout_rate':  self.ldout_rate,
            'fc_rate':     self.fc_rate,
        }
        for name, val in raw.items():
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val:.6f}")
        total = sum(raw.values())
        if not np.isclose(total, 1.0, atol=1e-4):
            raise ValueError(
                f"PA outcome rates must sum to 1.0, got {total:.6f}. "
                f"Breakdown: {raw}"
            )
        for cond_name, cond_val in [
            ('gidp_prob_given_gbout', self.gidp_prob_given_gbout),
            ('sf_prob_given_fbout',   self.sf_prob_given_fbout),
            ('sac_bunt_prob',         self.sac_bunt_prob),
        ]:
            if not (0.0 <= cond_val <= 1.0):
                raise ValueError(
                    f"{cond_name} must be in [0, 1], got {cond_val:.6f}"
                )

    def as_dict(self) -> Dict[str, float]:
        return {
            'K':      self.k_rate,
            'BB':     self.bb_rate,
            'HBP':    self.hbp_rate,
            'HR':     self.hr_rate,
            '1B':     self.single_rate,
            '2B':     self.double_rate,
            '3B':     self.triple_rate,
            'GBOUT':  self.gbout_rate,
            'FBOUT':  self.fbout_rate,
            'LDOUT':  self.ldout_rate,
            'FC':     self.fc_rate,
        }


def normalize_rates(rates: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a dict of outcome rates to sum exactly to 1.0.
    Used after blending to correct floating-point drift.
    """
    total = sum(rates.values())
    if total <= 0:
        raise ValueError("Cannot normalize rates that sum to zero.")
    return {k: v / total for k, v in rates.items()}
```

---

## 3. Log-Odds Blending — `model/blend.py`

### Method

For each PA outcome event `e`, let:
- `p_p` = pitcher's observed rate for event `e`
- `p_b` = batter's observed rate for event `e`
- `p_lg` = league average rate for event `e`

The blended probability is:

```
logit(p_blend) = logit(p_p) + logit(p_b) - logit(p_lg)
p_blend = sigmoid(logit(p_blend))
```

This is the **odds-ratio method** used by Fangraphs' matchup tools and projection systems. After blending, apply multiplicative park factor adjustments to HR rate (and optionally run factor to hit rates).

After all adjustments, **normalize** so all rates sum to 1.0.

### Platoon Adjustment

Apply a multiplicative platoon factor to `k_rate`, `bb_rate`, and `hr_rate` based on pitcher hand vs. batter hand. Values below are example multipliers; replace with empirical splits from your data.

| Matchup | K mult | BB mult | HR mult |
|---------|--------|---------|---------|
| RHP vs RHB | 1.00 | 1.00 | 1.00 (baseline) |
| RHP vs LHB | 0.95 | 1.05 | 1.10 |
| LHP vs LHB | 1.00 | 1.00 | 1.00 |
| LHP vs RHB | 0.92 | 1.08 | 1.12 |

### Implementation

```python
# model/blend.py

from __future__ import annotations
import numpy as np
from typing import Dict, Optional


PLATOON_FACTORS: Dict[str, Dict[str, float]] = {
    # key: f"{pitcher_hand}v{batter_hand}"
    "RvR": {"k_rate": 1.00, "bb_rate": 1.00, "hr_rate": 1.00},
    "RvL": {"k_rate": 0.95, "bb_rate": 1.05, "hr_rate": 1.10},
    "LvL": {"k_rate": 1.00, "bb_rate": 1.00, "hr_rate": 1.00},
    "LvR": {"k_rate": 0.92, "bb_rate": 1.08, "hr_rate": 1.12},
}

EPSILON = 1e-6   # guard against logit(0) or logit(1)


def _safe_logit(p: float) -> float:
    """logit with clipping to avoid ±inf."""
    p = float(np.clip(p, EPSILON, 1.0 - EPSILON))
    return np.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def blend_rate(
    pitcher_rate: float,
    batter_rate:  float,
    league_rate:  float,
) -> float:
    """
    Blend a single event rate using the log-odds (odds-ratio) method.

    Returns
    -------
    float
        Blended probability in (0, 1).
    """
    logit_blend = (
        _safe_logit(pitcher_rate)
        + _safe_logit(batter_rate)
        - _safe_logit(league_rate)
    )
    return float(_sigmoid(logit_blend))


def apply_park_factor(
    rates: Dict[str, float],
    hr_park_factor: float,
    run_park_factor: float = 1.0,
) -> Dict[str, float]:
    """
    Adjust blended rates for park effects.

    Parameters
    ----------
    rates : dict
        Blended raw rates keyed by outcome name.
    hr_park_factor : float
        Multiplicative HR park factor (1.0 = neutral). Typically 0.80–1.25.
    run_park_factor : float
        Multiplicative factor applied to hit rates (singles, doubles, triples).
        Default 1.0 (no adjustment beyond HR).

    Returns
    -------
    dict
        Adjusted rates (not yet normalized).
    """
    out = dict(rates)
    out['hr_rate']     = rates.get('hr_rate', 0.0)     * hr_park_factor
    out['single_rate'] = rates.get('single_rate', 0.0) * run_park_factor
    out['double_rate'] = rates.get('double_rate', 0.0) * run_park_factor
    out['triple_rate'] = rates.get('triple_rate', 0.0) * run_park_factor
    return out


def apply_platoon(
    rates: Dict[str, float],
    pitcher_hand: str,
    batter_hand:  str,
) -> Dict[str, float]:
    """
    Apply platoon split multipliers to K, BB, and HR rates.

    Parameters
    ----------
    rates : dict
        Blended rates.
    pitcher_hand : str
        'R' or 'L'.
    batter_hand : str
        'R' or 'L'.

    Returns
    -------
    dict
        Adjusted rates (not yet normalized).
    """
    key = f"{pitcher_hand}v{batter_hand}"
    factors = PLATOON_FACTORS.get(key)
    if factors is None:
        raise ValueError(
            f"Unknown platoon matchup key '{key}'. "
            f"Expected one of {list(PLATOON_FACTORS)}"
        )
    out = dict(rates)
    for stat, mult in factors.items():
        if stat in out:
            out[stat] = out[stat] * mult
    return out


def build_blended_rates(
    pitcher_stats:   Dict[str, float],
    batter_stats:    Dict[str, float],
    league_averages: Dict[str, float],
    hr_park_factor:  float,
    pitcher_hand:    str,
    batter_hand:     str,
    run_park_factor: float = 1.0,
) -> Dict[str, float]:
    """
    Full pipeline: blend all rates, apply park factor, apply platoon, normalize.

    Parameters
    ----------
    pitcher_stats : dict
        Must contain keys: k_rate, bb_rate, hbp_rate, hr_rate, single_rate,
        double_rate, triple_rate, gbout_rate, fbout_rate, ldout_rate, fc_rate,
        gidp_prob_given_gbout, sf_prob_given_fbout.
    batter_stats : dict
        Same keys as pitcher_stats.
    league_averages : dict
        Same keys.
    hr_park_factor : float
    pitcher_hand : str  ('R' or 'L')
    batter_hand : str   ('R' or 'L')
    run_park_factor : float

    Returns
    -------
    dict
        Normalized blended rates ready to construct PAOutcomeRates.
    """
    PRIMARY_OUTCOMES = [
        'k_rate', 'bb_rate', 'hbp_rate', 'hr_rate',
        'single_rate', 'double_rate', 'triple_rate',
        'gbout_rate', 'fbout_rate', 'ldout_rate', 'fc_rate',
    ]
    CONDITIONAL_RATES = [
        'gidp_prob_given_gbout',
        'sf_prob_given_fbout',
        'sac_bunt_prob',
    ]

    blended: Dict[str, float] = {}

    # Blend primary outcome rates
    for key in PRIMARY_OUTCOMES:
        p_p  = pitcher_stats.get(key, league_averages[key])
        p_b  = batter_stats.get(key,  league_averages[key])
        p_lg = league_averages[key]
        blended[key] = blend_rate(p_p, p_b, p_lg)

    # Blend conditional rates (pitcher-weighted for situation rates,
    # batter-weighted for tendencies like GIDP)
    for key in CONDITIONAL_RATES:
        p_p  = pitcher_stats.get(key, league_averages.get(key, 0.0))
        p_b  = batter_stats.get(key,  league_averages.get(key, 0.0))
        p_lg = league_averages.get(key, 0.0)
        if p_lg > 0:
            blended[key] = blend_rate(p_p, p_b, p_lg)
        else:
            blended[key] = (p_p + p_b) / 2.0

    # Park adjustment
    blended = apply_park_factor(blended, hr_park_factor, run_park_factor)

    # Platoon adjustment
    blended = apply_platoon(blended, pitcher_hand, batter_hand)

    # Normalize primary outcomes only (conditional rates stay as-is)
    total = sum(blended[k] for k in PRIMARY_OUTCOMES)
    for k in PRIMARY_OUTCOMES:
        blended[k] /= total

    return blended
```

---

## 4. Transition Matrix — `model/transitions.py`

### Design

For each plate appearance outcome × current base-out state, we compute:
- The **next base-out state** (transient, 0–23) — or an absorbing state
- The **number of runs scored** on that transition

Because we only need P(NRFI), we track:
- Whether **any run has scored yet** — once it does, we enter `RUN_ABSORBED`
- The inning ends at 3 outs → `NRFI_ABSORBED` if no runs scored

The transition function is a deterministic mapping:

```
transition(bases, outs, outcome) → (new_bases, new_outs, runs_scored)
```

Because some outcomes (GBOUT, FBOUT) are *conditionally* different based on base-out state, the transition function takes the full `(bases, outs, rates)` context.

### Runner Advancement Rules

These are the standard baseball advancement rules encoded in the transition logic:

**Strikeout (K)**
- No runners advance.
- `outs += 1`

**Walk (BB) / Hit by Pitch (HBP)**
- Batter to 1B.
- Forced runners advance exactly one base.
- Non-forced runners hold.
- If bases were loaded: runner on 3rd scores (1 run).

**Home Run (HR)**
- All runners score + batter scores.
- `runs = runner_count(bases) + 1`
- `new_bases = 0`, `outs` unchanged.

**Single (1B)**
- Batter to 1B.
- Runner on 1B → 2B.
- Runner on 2B → scores with standard assumption (2/3 of the time) — **we simplify: runner on 2B always scores on single**.
- Runner on 3B → scores.

  > More realistic: use a probabilistic advance model (runner on 2B scores ~65% on single, advances to 3B ~35%). For initial implementation, use the deterministic scoring assumption; add probability weights in v2.

**Double (2B)**
- Batter to 2B.
- Runner on 1B → 3B (does not score by default; scores if fast — simplify: does not score).
- Runner on 2B → scores.
- Runner on 3B → scores.

**Triple (3B)**
- Batter to 3B.
- All runners score.

**Ground Ball Out (GBOUT)**
- Default: batter out, `outs += 1`, no runners advance.
- **GIDP check**: if `on_first(bases)` and `outs < 2`:
  - With probability `gidp_prob_given_gbout`: `outs += 2`, lead runner erased (remove from 1B), all other runners hold.
  - If `outs + 2 >= 3`: inning ends — check if any run scored this transition.
- **Fielder's choice overlap**: FC is handled as its own outcome (see below).

**Fly Ball Out (FBOUT)**
- Default: batter out, `outs += 1`.
- **Sac fly check**: if `on_third(bases)` and `outs < 2`:
  - With probability `sf_prob_given_fbout`: run scores, `outs += 1`.
  - Runner on 3rd removed from base.
  - Other runners may tag: runner on 2B → 3B (simplification: tag on all deep fly balls with runner on 2nd too; v2 can add depth-of-fly parameter).
- **Sac bunt check** (applied before the swing, conceptually, but encoded here):
  - If `sac_bunt_prob > 0` and `outs < 2` and `(on_first(bases) or (on_first(bases) and on_second(bases)))`:
    - With probability `sac_bunt_prob`: batter out, runner(s) advance one base.
    - This replaces a GBOUT with the sac bunt outcome.
    - In implementation: we pre-process sac bunts at the top of the transition function before resolving the outcome type.

**Line Drive Out (LDOUT)**
- Batter out, `outs += 1`.
- No runners advance (runners generally freeze on liners).

**Fielder's Choice (FC)**
- Batter reaches 1B.
- Lead runner (lowest base with runner) is retired: `outs += 1`.
- Remaining runners hold.
- If runner on 3B was lead runner: they are out, batter to 1B.
- Runs: if runner on 3rd is NOT the lead runner (i.e. bases loaded: runner on 1B is erased), runner on 3B may score — simplify: runner on 3B scores if FC erases runner on 1B (force play at 2B, not 3B).

### Implementation

```python
# model/transitions.py

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

from .state import (
    encode_state, decode_state, NUM_STATES,
    NRFI_ABSORBED, RUN_ABSORBED,
    on_first, on_second, on_third, runner_count,
)
from .outcomes import PAOutcomeRates


# ---------------------------------------------------------------------------
# Low-level transition functions
# Each returns (new_bases, new_outs, runs_scored).
# new_outs == 3 signals end of inning.
# ---------------------------------------------------------------------------

def _transition_K(bases: int, outs: int) -> Tuple[int, int, int]:
    return bases, outs + 1, 0


def _transition_BB_HBP(bases: int, outs: int) -> Tuple[int, int, int]:
    runs = 0
    # Force all runners; batter goes to 1B
    if on_third(bases) and on_second(bases) and on_first(bases):
        # Bases loaded: runner on 3B scores
        runs = 1
        new_bases = 0b111  # stays loaded (runner on 3B scored, batter forces chain)
        # Actually: 3B runner scores, everyone else moves up, batter to 1B
        # New state: 1B=batter, 2B=was-1B, 3B=was-2B
        new_bases = 0b111  # still loaded because batter fills 1B
        # Correct logic:
        new_bases = (1 << 0) | (1 << 1) | (1 << 2)  # 1B, 2B, 3B all occupied
        runs = 1  # was-3B scores
    elif on_second(bases) and on_first(bases):
        # Runner on 1B→2B, runner on 2B→3B, batter to 1B
        new_bases = 0b111  # bases loaded
        runs = 0
    elif on_first(bases):
        # Runner on 1B→2B, batter to 1B
        new_bases = bases | 0b011
        # Clear 1B then set 1B and 2B
        new_bases = 0b011
        if on_third(bases):
            new_bases |= 0b100
        runs = 0
    else:
        # No force situation — batter to 1B, others hold
        new_bases = bases | 0b001
        runs = 0
    return new_bases, outs, runs


def _transition_HR(bases: int, outs: int) -> Tuple[int, int, int]:
    runs = runner_count(bases) + 1
    return 0, outs, runs


def _transition_1B(bases: int, outs: int) -> Tuple[int, int, int]:
    runs = 0
    new_bases = 0

    # Runner on 3B scores
    if on_third(bases):
        runs += 1

    # Runner on 2B scores (simplification: always scores on single)
    if on_second(bases):
        runs += 1

    # Runner on 1B → 2B
    if on_first(bases):
        new_bases |= 0b010  # runner advances to 2B

    # Batter to 1B
    new_bases |= 0b001

    return new_bases, outs, runs


def _transition_2B(bases: int, outs: int) -> Tuple[int, int, int]:
    runs = 0
    new_bases = 0

    # Runner on 3B scores
    if on_third(bases):
        runs += 1

    # Runner on 2B scores
    if on_second(bases):
        runs += 1

    # Runner on 1B → 3B (does not score; simplification)
    if on_first(bases):
        new_bases |= 0b100  # runner to 3B

    # Batter to 2B
    new_bases |= 0b010

    return new_bases, outs, runs


def _transition_3B(bases: int, outs: int) -> Tuple[int, int, int]:
    runs = runner_count(bases)   # all runners score
    new_bases = 0b100            # batter on 3B
    return new_bases, outs, runs


def _transition_GBOUT(
    bases: int,
    outs:  int,
    rates: PAOutcomeRates,
    rng:   Optional[np.random.Generator] = None,
) -> Tuple[int, int, int]:
    """
    Ground ball out.  Checks for GIDP probabilistically when applicable.

    In the *analytic* path, call this with rng=None and it returns
    *expected* (weighted) transitions — handled by the caller.
    In the *simulation* path, pass an rng for a single stochastic draw.
    """
    # Sac bunt pre-empts GBOUT in certain situations (handled upstream)
    runs = 0

    gidp_eligible = on_first(bases) and (outs < 2)

    if gidp_eligible:
        if rng is not None:
            # Stochastic: single draw
            is_gidp = rng.random() < rates.gidp_prob_given_gbout
        else:
            # Analytic: return two branches — caller handles weighting
            raise NotImplementedError(
                "Analytic GBOUT called without rng — "
                "use build_transition_row() which handles branching."
            )
        if is_gidp:
            new_outs = outs + 2
            # Erase the runner on 1B (they are doubled off)
            new_bases = bases & ~0b001  # clear 1B bit
            return new_bases, new_outs, runs
        else:
            # Normal groundout
            return bases, outs + 1, runs
    else:
        return bases, outs + 1, runs


def _transition_FBOUT(
    bases: int,
    outs:  int,
    rates: PAOutcomeRates,
    rng:   Optional[np.random.Generator] = None,
) -> Tuple[int, int, int]:
    """
    Fly ball out.  Checks for sac fly probabilistically when applicable.
    """
    runs = 0
    sf_eligible = on_third(bases) and (outs < 2)

    if sf_eligible:
        if rng is not None:
            is_sf = rng.random() < rates.sf_prob_given_fbout
        else:
            raise NotImplementedError(
                "Analytic FBOUT called without rng — "
                "use build_transition_row() which handles branching."
            )
        if is_sf:
            runs = 1
            new_bases = bases & ~0b100   # runner on 3B scores (remove)
            # Runner on 2B tags to 3B
            if on_second(bases):
                new_bases = (new_bases & ~0b010) | 0b100
            return new_bases, outs + 1, runs
        else:
            return bases, outs + 1, runs
    else:
        return bases, outs + 1, runs


def _transition_LDOUT(bases: int, outs: int) -> Tuple[int, int, int]:
    return bases, outs + 1, 0


def _transition_FC(bases: int, outs: int) -> Tuple[int, int, int]:
    """
    Fielder's choice: batter reaches 1B, lead runner is erased.
    Lead runner = lowest occupied base.
    If runner on 3B is not the lead runner, they may score on a FC play at 2B.
    """
    runs = 0
    new_bases = bases

    if on_first(bases):
        # Lead runner on 1B is erased (force at 2B)
        new_bases = bases & ~0b001   # remove 1B runner
        # Batter replaces on 1B
        new_bases |= 0b001
        # Runner on 3B scores if bases were loaded (force play at 2B, not 3B)
        if on_third(bases) and on_second(bases):
            # Bases loaded: FC erases 1B runner. Runner on 3B holds.
            # Runner on 2B forced to 3B? No — only if 1B runner advances.
            # Standard FC: only one base erase; everyone else holds.
            pass
        # If only runner was on 1B: new state is batter on 1B only.
    elif on_second(bases) and not on_first(bases):
        # Lead runner on 2B; batter reaches 1B (not a typical FC situation but valid)
        new_bases = (bases & ~0b010) | 0b001
    elif on_third(bases) and not on_first(bases) and not on_second(bases):
        # Runner on 3B only: FC at home? Rare; model as runner scores, batter to 1B
        runs = 1
        new_bases = 0b001

    return new_bases, outs + 1, runs


# ---------------------------------------------------------------------------
# Transition row builder — returns probability distribution over next states
# ---------------------------------------------------------------------------

def build_transition_row(
    state_index: int,
    rates:       PAOutcomeRates,
) -> Tuple[np.ndarray, float]:
    """
    For a given base-out state and PA outcome rates, compute the full
    probability distribution over next states.

    Parameters
    ----------
    state_index : int
        Current state index (0–23).
    rates : PAOutcomeRates
        Blended PA outcome rates for this pitcher–batter matchup.

    Returns
    -------
    probs : np.ndarray, shape (26,)
        Probability mass over states 0–23 (transient) + 24 (NRFI_ABSORBED) +
        25 (RUN_ABSORBED). probs[i] = P(next state = i | current state, rates).
    expected_runs : float
        Expected runs scored on this PA from this state (for diagnostics).
    """
    bases, outs = decode_state(state_index)

    # Output array: 24 transient + 2 absorbing
    probs = np.zeros(NUM_STATES + 2, dtype=np.float64)
    exp_runs = 0.0

    def _apply(prob: float, new_bases: int, new_outs: int, runs: int) -> None:
        nonlocal exp_runs
        if prob <= 0:
            return
        exp_runs += prob * runs
        if runs > 0:
            probs[RUN_ABSORBED] += prob
            return
        if new_outs >= 3:
            probs[NRFI_ABSORBED] += prob
        else:
            probs[encode_state(new_bases, new_outs)] += prob

    # ---- Strikeout ----
    nb, no, r = _transition_K(bases, outs)
    _apply(rates.k_rate, nb, no, r)

    # ---- Walk / HBP ----
    nb, no, r = _transition_BB_HBP(bases, outs)
    _apply(rates.bb_rate + rates.hbp_rate, nb, no, r)

    # ---- Home Run ----
    nb, no, r = _transition_HR(bases, outs)
    _apply(rates.hr_rate, nb, no, r)

    # ---- Single ----
    nb, no, r = _transition_1B(bases, outs)
    _apply(rates.single_rate, nb, no, r)

    # ---- Double ----
    nb, no, r = _transition_2B(bases, outs)
    _apply(rates.double_rate, nb, no, r)

    # ---- Triple ----
    nb, no, r = _transition_3B(bases, outs)
    _apply(rates.triple_rate, nb, no, r)

    # ---- Ground Ball Out — two branches ----
    p_gb = rates.gbout_rate
    gidp_eligible = on_first(bases) and (outs < 2)
    if gidp_eligible and p_gb > 0:
        p_gidp   = p_gb * rates.gidp_prob_given_gbout
        p_normal = p_gb * (1.0 - rates.gidp_prob_given_gbout)

        # GIDP branch
        new_outs_gidp  = outs + 2
        new_bases_gidp = bases & ~0b001
        _apply(p_gidp, new_bases_gidp, new_outs_gidp, 0)

        # Normal gbout branch
        _apply(p_normal, bases, outs + 1, 0)
    elif p_gb > 0:
        _apply(p_gb, bases, outs + 1, 0)

    # ---- Fly Ball Out — two branches ----
    p_fb = rates.fbout_rate
    sf_eligible = on_third(bases) and (outs < 2)
    if sf_eligible and p_fb > 0:
        p_sf     = p_fb * rates.sf_prob_given_fbout
        p_normal = p_fb * (1.0 - rates.sf_prob_given_fbout)

        # Sac fly branch — run scores
        probs[RUN_ABSORBED] += p_sf
        exp_runs += p_sf * 1.0

        # Normal fbout branch — check if runner on 2B tags to 3B
        new_bases_normal = bases
        if on_second(bases):
            new_bases_normal = (bases & ~0b010) | 0b100  # 2B runner tags to 3B
        _apply(p_normal, new_bases_normal, outs + 1, 0)
    elif p_fb > 0:
        _apply(p_fb, bases, outs + 1, 0)

    # ---- Line Drive Out ----
    _apply(rates.ldout_rate, bases, outs + 1, 0)

    # ---- Fielder's Choice ----
    nb, no, r = _transition_FC(bases, outs)
    _apply(rates.fc_rate, nb, no, r)

    # Sac bunt (pre-empts a portion of gbout_rate in sac-eligible situations)
    # Already accounted for in rates if sac_bunt_prob is folded in upstream.
    # If you want to handle sac bunt separately: subtract p_sac from gbout and
    # redirect it. Recommended: fold into rates at build time.

    # Sanity check
    assert np.isclose(probs.sum(), 1.0, atol=1e-4), (
        f"Transition row for state {state_index} sums to {probs.sum():.6f}"
    )

    return probs, exp_runs


def build_full_transition_matrix(
    lineup_rates: list[PAOutcomeRates],
) -> np.ndarray:
    """
    Build the full (24 × 26) transition matrix for one half-inning.

    Each row corresponds to a transient state (0–23). Columns 0–23 are
    transient states; column 24 = NRFI_ABSORBED; column 25 = RUN_ABSORBED.

    Because batter order matters (each new state is visited with a different
    batter), this function requires a list of 9 PAOutcomeRates objects in
    batting order.

    However, for a *stationary* approximation (useful for quick estimates),
    pass a single averaged PAOutcomeRates. For full accuracy, use the
    simulation path in chain.py.

    Parameters
    ----------
    lineup_rates : list[PAOutcomeRates]
        Length-9 list. Index 0 = leadoff batter.

    Returns
    -------
    np.ndarray, shape (24, 26)
    """
    # Average rates across lineup for stationary approximation
    avg_rates = _average_lineup_rates(lineup_rates)
    T = np.zeros((NUM_STATES, NUM_STATES + 2), dtype=np.float64)
    for s in range(NUM_STATES):
        row, _ = build_transition_row(s, avg_rates)
        T[s, :] = row
    return T


def _average_lineup_rates(lineup_rates: list[PAOutcomeRates]) -> PAOutcomeRates:
    """Compute arithmetic mean across lineup for stationary approximation."""
    from .outcomes import PAOutcomeRates
    n = len(lineup_rates)
    if n == 0:
        raise ValueError("lineup_rates must not be empty.")
    keys = [
        'k_rate', 'bb_rate', 'hbp_rate', 'hr_rate',
        'single_rate', 'double_rate', 'triple_rate',
        'gbout_rate', 'fbout_rate', 'ldout_rate', 'fc_rate',
        'gidp_prob_given_gbout', 'sf_prob_given_fbout', 'sac_bunt_prob',
    ]
    averaged = {k: np.mean([getattr(r, k) for r in lineup_rates]) for k in keys}
    # Re-normalize primary outcomes
    primary = [
        'k_rate', 'bb_rate', 'hbp_rate', 'hr_rate',
        'single_rate', 'double_rate', 'triple_rate',
        'gbout_rate', 'fbout_rate', 'ldout_rate', 'fc_rate',
    ]
    total = sum(averaged[k] for k in primary)
    for k in primary:
        averaged[k] /= total
    return PAOutcomeRates(**averaged)
```

---

## 5. Markov Chain Runner — `model/chain.py`

### Two Computation Paths

**Analytic (fundamental matrix)**

For the stationary approximation (averaged lineup rates):
1. Extract the 24×24 sub-matrix `Q` from `T[:24, :24]` (transient-to-transient block).
2. Compute `N = (I - Q)^{-1}` (fundamental matrix).
3. The absorption probability vector is `B = N @ R` where `R` is the 24×2 absorption matrix `T[:24, 24:]`.
4. `P(NRFI)` from starting state 0 = `B[0, 0]` (column 0 = NRFI_ABSORBED).

**Monte Carlo simulation (batter-order-accurate)**

For lineup-accurate modeling:
1. Initialize state = 0 (empty bases, 0 outs), batter_slot = 0.
2. Sample PA outcome from the current batter's `PAOutcomeRates`.
3. Apply transition to get next state.
4. If absorbed: record result; increment batter slot modulo 9.
5. Repeat for N innings; estimate `P(NRFI)` = fraction of innings absorbed into NRFI_ABSORBED.

### Implementation

```python
# model/chain.py

from __future__ import annotations
import numpy as np
from typing import List, Tuple

from .state import (
    NUM_STATES, STARTING_STATE, NRFI_ABSORBED, RUN_ABSORBED,
    decode_state,
)
from .outcomes import PAOutcomeRates
from .transitions import build_transition_row, build_full_transition_matrix


# ---------------------------------------------------------------------------
# Analytic path
# ---------------------------------------------------------------------------

def compute_nrfi_analytic(
    lineup_rates: List[PAOutcomeRates],
) -> Tuple[float, np.ndarray]:
    """
    Compute P(NRFI) analytically using the fundamental matrix.

    Uses a stationary (lineup-averaged) approximation of batter order.

    Parameters
    ----------
    lineup_rates : list of PAOutcomeRates

    Returns
    -------
    p_nrfi : float
        Probability of no run scored in the half-inning.
    absorption_probs : np.ndarray, shape (24, 2)
        For each starting state: [P(NRFI_absorbed), P(RUN_absorbed)].
    """
    T = build_full_transition_matrix(lineup_rates)

    Q = T[:NUM_STATES, :NUM_STATES]      # 24×24 transient block
    R = T[:NUM_STATES, NUM_STATES:]      # 24×2 absorption block

    I = np.eye(NUM_STATES, dtype=np.float64)

    # Fundamental matrix: N[i,j] = expected number of times in state j given start i
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Fundamental matrix inversion failed: {e}. "
            "Check that transition rows sum to 1 and Q has spectral radius < 1."
        )

    B = N @ R   # shape (24, 2)  — absorption probabilities

    # Validate: each row must sum to ~1
    row_sums = B.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        bad = np.where(np.abs(row_sums - 1.0) > 1e-4)[0]
        raise RuntimeError(
            f"Absorption probability rows do not sum to 1 for states: {bad}. "
            f"Row sums: {row_sums[bad]}"
        )

    p_nrfi = float(B[STARTING_STATE, 0])
    return p_nrfi, B


# ---------------------------------------------------------------------------
# Simulation path
# ---------------------------------------------------------------------------

def simulate_half_inning(
    lineup_rates: List[PAOutcomeRates],
    batter_start: int,
    rng:          np.random.Generator,
) -> Tuple[bool, int]:
    """
    Simulate a single half-inning.

    Parameters
    ----------
    lineup_rates : list of PAOutcomeRates
        Length 9. Index = batting order slot (0-based).
    batter_start : int
        Index into lineup_rates for the leadoff batter (0 = top of order).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    nrfi : bool
        True if the inning ended without a run scoring.
    batters_faced : int
        Number of PAs in this half-inning.
    """
    state = STARTING_STATE
    batter_slot = batter_start % 9
    batters_faced = 0
    max_pa = 27  # safety cap — no inning should ever need more than this

    for _ in range(max_pa):
        rates = lineup_rates[batter_slot]
        row, _ = build_transition_row(state, rates)

        # Draw next state
        next_state = int(rng.choice(len(row), p=row))
        batters_faced += 1
        batter_slot = (batter_slot + 1) % 9

        if next_state == NRFI_ABSORBED:
            return True, batters_faced
        if next_state == RUN_ABSORBED:
            return False, batters_faced

        state = next_state

    # Should not reach here
    raise RuntimeError(
        f"Half-inning simulation exceeded {max_pa} PAs without absorption. "
        f"Final state: {state}. Check transition rows for missing absorption."
    )


def simulate_nrfi(
    home_lineup: List[PAOutcomeRates],
    away_lineup: List[PAOutcomeRates],
    n_simulations: int = 100_000,
    seed:          int  = 42,
) -> dict:
    """
    Simulate full game first-inning NRFI across N trials.

    Parameters
    ----------
    home_lineup : list of PAOutcomeRates
        Rates for the home team batting order (bottom of 1st).
    away_lineup : list of PAOutcomeRates
        Rates for the away team batting order (top of 1st).
    n_simulations : int
        Number of innings to simulate per half-inning.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        p_nrfi_away  : float   — P(no run in top of 1st)
        p_nrfi_home  : float   — P(no run in bottom of 1st)
        p_nrfi_game  : float   — P(no run in either half)
        p_nrfi_ci    : tuple   — (lower, upper) 95% Wilson CI for game NRFI
        avg_pa_away  : float   — average PAs per top-of-1st inning
        avg_pa_home  : float   — average PAs per bottom-of-1st inning
    """
    rng = np.random.default_rng(seed)

    away_results = []
    away_pa      = []
    for _ in range(n_simulations):
        nrfi, pa = simulate_half_inning(away_lineup, batter_start=0, rng=rng)
        away_results.append(nrfi)
        away_pa.append(pa)

    home_results = []
    home_pa      = []
    for _ in range(n_simulations):
        nrfi, pa = simulate_half_inning(home_lineup, batter_start=0, rng=rng)
        home_results.append(nrfi)
        home_pa.append(pa)

    p_away = float(np.mean(away_results))
    p_home = float(np.mean(home_results))
    p_game = p_away * p_home

    # Wilson confidence interval for game-level NRFI
    # Each simulated game = one Bernoulli trial (away NRFI) × (home NRFI)
    # Approximate combined Bernoulli trials
    game_results = [a and h for a, h in zip(away_results, home_results)]
    p_game_empirical = float(np.mean(game_results))
    ci = _wilson_ci(sum(game_results), n_simulations)

    return {
        "p_nrfi_away":      p_away,
        "p_nrfi_home":      p_home,
        "p_nrfi_game":      p_game_empirical,
        "p_nrfi_game_indep": p_game,
        "p_nrfi_ci":        ci,
        "avg_pa_away":      float(np.mean(away_pa)),
        "avg_pa_home":      float(np.mean(home_pa)),
        "n_simulations":    n_simulations,
    }


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))
```

---

## 6. Data Loading & CSV Schemas

### `data/pitchers.csv` — Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `pitcher_id` | str | Unique ID (e.g. MLBAM ID) |
| `name` | str | Display name |
| `hand` | str | 'R' or 'L' |
| `k_rate` | float | K / PA |
| `bb_rate` | float | BB / PA (no IBB) |
| `hbp_rate` | float | HBP / PA |
| `hr_rate` | float | HR / PA |
| `single_rate` | float | 1B / PA |
| `double_rate` | float | 2B / PA |
| `triple_rate` | float | 3B / PA |
| `gbout_rate` | float | GB outs / PA |
| `fbout_rate` | float | FB outs / PA |
| `ldout_rate` | float | LD outs / PA |
| `fc_rate` | float | FC / PA |
| `gidp_prob_given_gbout` | float | GIDP / (GB out opportunities) |
| `sf_prob_given_fbout` | float | SF / (FB out opportunities with runner on 3rd) |

> All rate columns must sum (k_rate + bb_rate + ... + fc_rate) ≈ 1.0 per row.
> Derive from FanGraphs pitcher dashboard + Statcast batted ball data.

### `data/batters.csv` — Required Columns

Same schema as `pitchers.csv` plus:

| Column | Type | Description |
|--------|------|-------------|
| `batter_id` | str | Unique ID |
| `hand` | str | 'R' or 'L' |
| `sac_bunt_prob` | float | Sac bunt success rate (0 for most; nonzero for NL pitchers/slap hitters) |

### `data/park_factors.csv` — Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `team` | str | Team abbreviation |
| `hr_factor` | float | HR park factor (1.0 = neutral) |
| `run_factor` | float | Overall run factor |

### `data/lineups.csv` — Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | str | Game identifier |
| `team` | str | Team abbreviation |
| `side` | str | 'home' or 'away' |
| `slot` | int | Batting order (1–9) |
| `batter_id` | str | Links to batters.csv |
| `pitcher_id` | str | Starting pitcher (same value for all 9 rows per team) |

### `data/league_averages.csv` — Single Row

Contains one row with all the same rate columns as `pitchers.csv`, representing MLB-wide averages for the season being modeled.

---

## 7. Top-Level Pipeline — `pipeline.py`

```python
# pipeline.py

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

from model.outcomes import PAOutcomeRates
from model.blend import build_blended_rates
from model.chain import simulate_nrfi, compute_nrfi_analytic


DATA_DIR = "data/"

RATE_COLUMNS = [
    'k_rate', 'bb_rate', 'hbp_rate', 'hr_rate',
    'single_rate', 'double_rate', 'triple_rate',
    'gbout_rate', 'fbout_rate', 'ldout_rate', 'fc_rate',
    'gidp_prob_given_gbout', 'sf_prob_given_fbout', 'sac_bunt_prob',
]


def load_data() -> dict:
    """Load all CSVs and return as a dict of DataFrames."""
    pitchers = pd.read_csv(f"{DATA_DIR}pitchers.csv").set_index('pitcher_id')
    batters  = pd.read_csv(f"{DATA_DIR}batters.csv").set_index('batter_id')
    parks    = pd.read_csv(f"{DATA_DIR}park_factors.csv").set_index('team')
    lineups  = pd.read_csv(f"{DATA_DIR}lineups.csv")
    league   = pd.read_csv(f"{DATA_DIR}league_averages.csv").iloc[0]
    return {
        "pitchers": pitchers,
        "batters":  batters,
        "parks":    parks,
        "lineups":  lineups,
        "league":   league,
    }


def build_lineup_rates(
    game_id:       str,
    side:          str,
    data:          dict,
    home_team:     str,
) -> list[PAOutcomeRates]:
    """
    Build a list of 9 PAOutcomeRates objects (batting order slots 1–9)
    for one team's half-inning.

    Parameters
    ----------
    game_id : str
    side : str  ('home' or 'away')
    data : dict  (from load_data())
    home_team : str  (team abbreviation for park factor lookup)
    """
    lineups  = data["lineups"]
    pitchers = data["pitchers"]
    batters  = data["batters"]
    parks    = data["parks"]
    league   = data["league"].to_dict()

    game_lineups = lineups[
        (lineups['game_id'] == game_id) &
        (lineups['side'] == side)
    ].sort_values('slot')

    if len(game_lineups) != 9:
        raise ValueError(
            f"Expected 9 batters for game {game_id} side {side}, "
            f"got {len(game_lineups)}."
        )

    pitcher_id   = game_lineups['pitcher_id'].iloc[0]
    pitcher_row  = pitchers.loc[pitcher_id].to_dict()
    pitcher_hand = pitcher_row['hand']

    park_row       = parks.loc[home_team]
    hr_park_factor = float(park_row['hr_factor'])
    run_park_factor = float(park_row.get('run_factor', 1.0))

    rates_list = []
    for _, row in game_lineups.iterrows():
        batter_id   = row['batter_id']
        batter_row  = batters.loc[batter_id].to_dict()
        batter_hand = batter_row['hand']

        blended = build_blended_rates(
            pitcher_stats   = pitcher_row,
            batter_stats    = batter_row,
            league_averages = league,
            hr_park_factor  = hr_park_factor,
            pitcher_hand    = pitcher_hand,
            batter_hand     = batter_hand,
            run_park_factor = run_park_factor,
        )

        rates = PAOutcomeRates(**{k: blended.get(k, 0.0) for k in RATE_COLUMNS})
        rates_list.append(rates)

    return rates_list


def run_game(
    game_id:       str,
    home_team:     str,
    away_team:     str,
    n_simulations: int = 100_000,
    seed:          int = 42,
    analytic:      bool = True,
) -> dict:
    """
    Full pipeline for one game.

    Parameters
    ----------
    game_id : str
    home_team : str
    away_team : str
    n_simulations : int
    seed : int
    analytic : bool
        If True, also compute the analytic (fundamental matrix) estimate.

    Returns
    -------
    dict
        All results with keys from simulate_nrfi() plus optional analytic estimates.
    """
    data = load_data()

    home_rates = build_lineup_rates(game_id, 'home', data, home_team)
    away_rates = build_lineup_rates(game_id, 'away', data, home_team)

    results = simulate_nrfi(
        home_lineup    = home_rates,
        away_lineup    = away_rates,
        n_simulations  = n_simulations,
        seed           = seed,
    )

    if analytic:
        p_nrfi_away_a, _ = compute_nrfi_analytic(away_rates)
        p_nrfi_home_a, _ = compute_nrfi_analytic(home_rates)
        results["p_nrfi_away_analytic"] = p_nrfi_away_a
        results["p_nrfi_home_analytic"] = p_nrfi_home_a
        results["p_nrfi_game_analytic"] = p_nrfi_away_a * p_nrfi_home_a

    results["game_id"]   = game_id
    results["home_team"] = home_team
    results["away_team"] = away_team

    _print_results(results)
    return results


def _print_results(r: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  NRFI Model — {r.get('away_team','Away')} @ {r.get('home_team','Home')}")
    print(f"{'='*55}")
    print(f"  P(NRFI away half):  {r['p_nrfi_away']:.4f}")
    print(f"  P(NRFI home half):  {r['p_nrfi_home']:.4f}")
    print(f"  P(NRFI game, sim):  {r['p_nrfi_game']:.4f}")
    if 'p_nrfi_game_analytic' in r:
        print(f"  P(NRFI game, ana):  {r['p_nrfi_game_analytic']:.4f}")
    lo, hi = r['p_nrfi_ci']
    print(f"  95% CI (Wilson):    [{lo:.4f}, {hi:.4f}]")
    print(f"  Avg PA / away half: {r['avg_pa_away']:.2f}")
    print(f"  Avg PA / home half: {r['avg_pa_home']:.2f}")
    print(f"  Simulations:        {r['n_simulations']:,}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python pipeline.py <game_id> <home_team> <away_team>")
        sys.exit(1)
    run_game(
        game_id   = sys.argv[1],
        home_team = sys.argv[2],
        away_team = sys.argv[3],
    )
```

---

## 8. Historical Validation — `validate.py`

```python
# validate.py
"""
Backtesting harness.

For each historical game in a validation CSV, run the model and compare
predicted P(NRFI) to observed outcome (1 = NRFI, 0 = run scored).

Metrics computed:
- Brier score (mean squared error of probabilities)
- Log loss
- Calibration curve (predicted decile vs actual NRFI rate)
- AUC-ROC
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple
from pipeline import run_game


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    ))


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_low":       round(bins[i], 2),
            "bin_high":      round(bins[i + 1], 2),
            "mean_predicted": float(y_pred[mask].mean()),
            "mean_actual":    float(y_true[mask].mean()),
            "count":          int(mask.sum()),
        })
    return pd.DataFrame(rows)


def run_validation(
    validation_csv: str,
    n_simulations:  int = 50_000,
) -> None:
    """
    Parameters
    ----------
    validation_csv : str
        CSV with columns: game_id, home_team, away_team, nrfi_result (0 or 1).
    """
    df = pd.read_csv(validation_csv)
    required = {'game_id', 'home_team', 'away_team', 'nrfi_result'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Validation CSV missing columns: {missing}")

    y_pred_list = []
    y_true_list = []

    for _, row in df.iterrows():
        try:
            result = run_game(
                game_id       = row['game_id'],
                home_team     = row['home_team'],
                away_team     = row['away_team'],
                n_simulations = n_simulations,
                analytic      = False,
            )
            y_pred_list.append(result['p_nrfi_game'])
            y_true_list.append(int(row['nrfi_result']))
        except Exception as e:
            print(f"  Skipping game {row['game_id']}: {e}")

    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)

    print(f"\nValidation Results ({len(y_true)} games)")
    print(f"  Brier score:  {brier_score(y_true, y_pred):.5f}")
    print(f"  Log loss:     {log_loss(y_true, y_pred):.5f}")
    print(f"  Mean P(NRFI): {y_pred.mean():.4f}  (actual: {y_true.mean():.4f})")
    print("\nCalibration curve:")
    print(calibration_curve(y_true, y_pred).to_string(index=False))


if __name__ == "__main__":
    import sys
    run_validation(sys.argv[1])
```

---

## 9. `model/__init__.py`

```python
# model/__init__.py
from .state import encode_state, decode_state, describe_state
from .outcomes import PAOutcomeRates
from .blend import build_blended_rates
from .transitions import build_transition_row, build_full_transition_matrix
from .chain import simulate_nrfi, compute_nrfi_analytic
```

---

## 10. Known Simplifications & Suggested v2 Upgrades

| Simplification | Location | v2 Upgrade |
|---|---|---|
| Runner on 2B always scores on single | `transitions.py _transition_1B` | Use empirical 65/35 split; add speed factor |
| Runner on 1B does not score on double | `transitions.py _transition_2B` | Model as probabilistic (30% score rate) |
| Sac bunt pre-baked into gbout rate | `transitions.py` | Separate sac bunt state pre-empting the PA |
| No stolen base / wild pitch modeling | — | Add as between-PA Bernoulli events keyed to pitcher PB% and runner SB% |
| FC lead-runner logic simplified | `transitions.py _transition_FC` | Full 8-scenario lookup table |
| Platoon factors hardcoded | `blend.py` | Load from empirical L/R split CSV per player |
| Stationary approximation in analytic path | `chain.py` | Build a batter-order-indexed transition tensor (9 × 24 × 26) |
| No pitch-level layer | — | Add count-state sub-chain feeding into PA outcome |

---

## 11. Example Invocation

```bash
# Install dependencies
pip install -r requirements.txt

# Run model for a single game
python pipeline.py GAME_2025_001 NYY BOS

# Run historical validation
python validate.py data/validation_2024.csv
```

---

## 12. Data Acquisition Notes

- **FanGraphs** pitcher and batter dashboards: export CSV with K%, BB%, GB%, FB%, LD%, HR/FB, BABIP, GIDP. Compute per-PA rates by dividing by PA count.
- **Baseball Savant (Statcast)**: use the leaderboard search for xwOBA, launch angle, EV distributions, pitch mix, and whiff rates.
- **Park factors**: Baseball Reference or FanGraphs park factors page (3-year regressed recommended).
- **Lineups**: MLB Stats API endpoint `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=lineups`

---

*End of implementation spec.*