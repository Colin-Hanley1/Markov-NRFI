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
# Probabilistic runner-advancement parameters.
#
# These replace the old deterministic "runner on 2B always scores on single,
# runner on 1B never scores on double" assumptions with league-average rates.
# Values are module-level so the parameter sweep can patch them before
# running the backtest.
# ---------------------------------------------------------------------------

ADVANCEMENT = {
    "p_2b_scores_single":    0.70,  # runner on 2B, single hit → scores
    "p_1b_scores_double":    0.50,  # runner on 1B, double hit → scores
    "p_1b_to_3b_single":     0.30,  # runner on 1B, single hit → goes to 3B (vs 2B)
}


def set_advancement(**params):
    """Update runner-advancement parameters in place (used by param sweep)."""
    for k, v in params.items():
        if k in ADVANCEMENT:
            ADVANCEMENT[k] = v


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
        new_bases = 0b111  # still loaded because batter fills 1B
        runs = 1
    elif on_second(bases) and on_first(bases):
        # Runner on 1B→2B, runner on 2B→3B, batter to 1B
        new_bases = 0b111  # bases loaded
        runs = 0
    elif on_first(bases):
        # Runner on 1B→2B, batter to 1B
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
            pass
    elif on_second(bases) and not on_first(bases):
        # Lead runner on 2B; batter reaches 1B
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

    # ---- Single — probabilistic branching on runner advancement ----
    p_1b = rates.single_rate
    if p_1b > 0:
        p_2b_scores = ADVANCEMENT["p_2b_scores_single"]
        p_1b_to_3b  = ADVANCEMENT["p_1b_to_3b_single"]

        # Base: batter reaches 1B. Runner on 3B always scores.
        # Compute each branch explicitly.
        r3b_scores = on_third(bases)
        base_runs = 1 if r3b_scores else 0

        # Enumerate branches: (prob, runs, bases_after)
        # Keep 2B-runner and 1B-runner fates orthogonal (independent).
        # 2B-runner fate (if present)
        branches_2b = [(1.0, 0, 0)]  # (prob, runs_added, bases_bit_set)
        if on_second(bases):
            branches_2b = [
                (p_2b_scores,    1, 0),          # scores
                (1 - p_2b_scores, 0, 0b100),     # ends at 3B
            ]

        # 1B-runner fate (if present)
        branches_1b = [(1.0, 0, 0)]
        if on_first(bases):
            branches_1b = [
                (1 - p_1b_to_3b, 0, 0b010),  # 1B → 2B
                (p_1b_to_3b,     0, 0b100),  # 1B → 3B (first-to-third)
            ]

        # Batter to 1B always
        batter_bit = 0b001

        for p2, r2, nb2 in branches_2b:
            for p1, r1, nb1 in branches_1b:
                prob_branch = p_1b * p2 * p1
                if prob_branch <= 0: continue
                runs = base_runs + r2 + r1
                new_bases = batter_bit | nb1 | nb2
                # If two runners both end up at 3B, cap at one (shouldn't happen often)
                _apply(prob_branch, new_bases, outs, runs)

    # ---- Double — probabilistic branching on runner-from-1B scoring ----
    p_2b = rates.double_rate
    if p_2b > 0:
        p_1b_scores = ADVANCEMENT["p_1b_scores_double"]

        # Runner on 2B always scores, runner on 3B always scores.
        # Batter always ends up at 2B.
        base_runs = (1 if on_second(bases) else 0) + (1 if on_third(bases) else 0)

        branches_1b = [(1.0, 0, 0)]
        if on_first(bases):
            branches_1b = [
                (p_1b_scores,    1, 0),         # scores
                (1 - p_1b_scores, 0, 0b100),    # ends at 3B
            ]

        for p1, r1, nb1 in branches_1b:
            prob_branch = p_2b * p1
            if prob_branch <= 0: continue
            runs = base_runs + r1
            new_bases = 0b010 | nb1  # batter on 2B
            _apply(prob_branch, new_bases, outs, runs)

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

    Uses a stationary (lineup-averaged) approximation.

    Parameters
    ----------
    lineup_rates : list[PAOutcomeRates]
        Length-9 list. Index 0 = leadoff batter.

    Returns
    -------
    np.ndarray, shape (24, 26)
    """
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
