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
