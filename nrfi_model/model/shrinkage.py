"""Empirical-Bayes shrinkage for player rate profiles."""

from __future__ import annotations


# PA at which a rate is trusted 50/50 against league average. These are
# starting heuristics and should be tuned against backtest results over time.
SHRINKAGE_K = {
    "k_rate": 60,
    "bb_rate": 120,
    "hbp_rate": 240,
    "hr_rate": 150,
    "single_rate": 200,
    "double_rate": 200,
    "triple_rate": 200,
    "gbout_rate": 100,
    "fbout_rate": 100,
    "ldout_rate": 100,
    "fc_rate": 100,
    "gidp_prob_given_gbout": 100,
    "sf_prob_given_fbout": 100,
    "sac_bunt_prob": 100,
}


def shrink_rates(rates: dict, pa: float, league: dict) -> dict:
    """Shrink each rate toward league average, weighted by plate appearances."""
    out = dict(rates)
    for col, k in SHRINKAGE_K.items():
        if col not in rates or col not in league:
            continue
        w = pa / (pa + k) if pa > 0 else 0.0
        out[col] = w * float(rates[col]) + (1 - w) * float(league[col])
    return out
