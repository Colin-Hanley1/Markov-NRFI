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


# Weight given to first-inning split when blending with the global pitcher
# profile. Set to 0 to disable, 1 to use only 1st-inning rates.
FIRST_INNING_BLEND = 0.40

# Minimum PA in a split for it to be trusted (otherwise fall back to global).
MIN_SPLIT_PA = 50


def _resolve_split_rates(
    base_stats: Dict[str, float],
    splits: Optional[Dict[str, float]],
    prefix: str,
    primary_keys: list,
) -> Dict[str, float]:
    """
    If `splits` contains rates with the given prefix and meets MIN_SPLIT_PA,
    return a copy of base_stats with those rates substituted in. Otherwise
    return base_stats unchanged.
    """
    if not splits:
        return base_stats
    pa = splits.get(f"{prefix}pa", 0)
    if pa < MIN_SPLIT_PA:
        return base_stats
    out = dict(base_stats)
    for k in primary_keys:
        sk = f"{prefix}{k}"
        if sk in splits and splits[sk] is not None:
            try:
                v = float(splits[sk])
                if v >= 0:
                    out[k] = v
            except (TypeError, ValueError):
                pass
    return out


def _blend_first_inning(
    pitcher_stats: Dict[str, float],
    pitcher_splits: Optional[Dict[str, float]],
    primary_keys: list,
) -> Dict[str, float]:
    """
    Blend the pitcher's 1st-inning rates with their global rates.
    Uses FIRST_INNING_BLEND as the weight on the 1st-inning sample.
    """
    if not pitcher_splits or FIRST_INNING_BLEND <= 0:
        return pitcher_stats
    pa = pitcher_splits.get("i1_pa", 0)
    if pa < MIN_SPLIT_PA:
        return pitcher_stats
    # Effective weight scales with sample reliability
    w = FIRST_INNING_BLEND * min(1.0, pa / 200.0)
    out = dict(pitcher_stats)
    for k in primary_keys:
        sk = f"i1_{k}"
        if sk in pitcher_splits and pitcher_splits[sk] is not None:
            try:
                v = float(pitcher_splits[sk])
                if v >= 0:
                    out[k] = (1 - w) * out.get(k, 0) + w * v
            except (TypeError, ValueError):
                pass
    return out


def build_blended_rates(
    pitcher_stats:   Dict[str, float],
    batter_stats:    Dict[str, float],
    league_averages: Dict[str, float],
    hr_park_factor:  float,
    pitcher_hand:    str,
    batter_hand:     str,
    run_park_factor: float = 1.0,
    pitcher_splits:  Optional[Dict[str, float]] = None,
    batter_splits:   Optional[Dict[str, float]] = None,
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

    # Substitute platoon-specific rates if available.
    # Pitcher uses split corresponding to BATTER's hand (vr_ when facing R).
    # Batter uses split corresponding to PITCHER's hand (vl_ when facing L).
    pitcher_prefix = "vr_" if batter_hand == "R" else "vl_"
    batter_prefix  = "vl_" if pitcher_hand == "L" else "vr_"
    pitcher_stats = _resolve_split_rates(pitcher_stats, pitcher_splits, pitcher_prefix, PRIMARY_OUTCOMES)
    batter_stats  = _resolve_split_rates(batter_stats,  batter_splits,  batter_prefix,  PRIMARY_OUTCOMES)

    # Blend in the pitcher's 1st-inning tendencies (sample-weighted)
    pitcher_stats = _blend_first_inning(pitcher_stats, pitcher_splits, PRIMARY_OUTCOMES)

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

    # Platoon adjustment — only apply hardcoded multipliers as a fallback.
    # If real per-player splits were already substituted in above, skip this.
    real_pitcher_split = pitcher_splits and pitcher_splits.get(f"{pitcher_prefix}pa", 0) >= MIN_SPLIT_PA
    real_batter_split  = batter_splits  and batter_splits.get(f"{batter_prefix}pa", 0) >= MIN_SPLIT_PA
    if not (real_pitcher_split or real_batter_split):
        blended = apply_platoon(blended, pitcher_hand, batter_hand)

    # Normalize primary outcomes only (conditional rates stay as-is)
    total = sum(blended[k] for k in PRIMARY_OUTCOMES)
    for k in PRIMARY_OUTCOMES:
        blended[k] /= total

    return blended
