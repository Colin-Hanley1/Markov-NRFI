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
