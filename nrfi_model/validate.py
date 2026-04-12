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
