#!/usr/bin/env python3
"""Fit Platt-scaling calibration for NRFI probabilities."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


BASE_DIR = Path(__file__).resolve().parent
RESULTS_LOG = BASE_DIR / "docs/data/results_log.json"
OUT_PATH = BASE_DIR / "data/calibration_params.json"
EPS = 1e-6


def _load_entries() -> list[dict]:
    payload = json.loads(RESULTS_LOG.read_text())
    entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    usable = [
        e for e in entries
        if e.get("p_nrfi_predicted") is not None and e.get("nrfi_actual") is not None and e.get("date")
    ]
    return sorted(usable, key=lambda e: (e["date"], str(e.get("game_id", ""))))


def _arrays(entries: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    p = np.array([float(e["p_nrfi_predicted"]) for e in entries], dtype=float)
    y = np.array([1.0 if bool(e["nrfi_actual"]) else 0.0 for e in entries], dtype=float)
    return np.clip(p, EPS, 1 - EPS), y


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def apply_calibration(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return _sigmoid(a * _logit(np.clip(p, EPS, 1 - EPS)) + b)


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    pc = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def fit_platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = _logit(p)

    def objective(params: np.ndarray) -> float:
        a, b = params
        pred = _sigmoid(a * x + b)
        return _log_loss(pred, y)

    result = minimize(objective, np.array([1.0, 0.0]), method="BFGS")
    if not result.success:
        raise RuntimeError(f"Calibration fit failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def calibration_bins(p_raw: np.ndarray, p_cal: np.ndarray, y: np.ndarray) -> list[dict]:
    edges = np.linspace(0.30, 0.70, 9)
    rows = []
    for i in range(len(edges) - 1):
        m = (p_raw >= edges[i]) & (p_raw < edges[i + 1])
        if int(m.sum()) < 3:
            continue
        rows.append({
            "bin": f"{edges[i]:.0%}-{edges[i + 1]:.0%}",
            "n": int(m.sum()),
            "raw": float(p_raw[m].mean()),
            "calibrated": float(p_cal[m].mean()),
            "actual": float(y[m].mean()),
        })
    return rows


def _walk_forward_split(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    dates = sorted({e["date"] for e in entries})
    split_idx = max(1, min(len(dates) - 1, int(len(dates) * 0.8)))
    cutoff = dates[split_idx]
    train = [e for e in entries if e["date"] < cutoff]
    holdout = [e for e in entries if e["date"] >= cutoff]
    return train, holdout


def _print_metrics(label: str, p: np.ndarray, y: np.ndarray, a: float, b: float) -> dict:
    p_cal = apply_calibration(p, a, b)
    metrics = {
        "brier_before": _brier(p, y),
        "brier_after": _brier(p_cal, y),
        "log_loss_before": _log_loss(p, y),
        "log_loss_after": _log_loss(p_cal, y),
    }
    print(
        f"{label}: Brier {metrics['brier_before']:.5f} -> {metrics['brier_after']:.5f}; "
        f"log-loss {metrics['log_loss_before']:.5f} -> {metrics['log_loss_after']:.5f}"
    )
    return metrics


def main() -> None:
    entries = _load_entries()
    if len(entries) < 20:
        raise SystemExit(f"Need more calibration data; found {len(entries)} usable games.")

    train_entries, holdout_entries = _walk_forward_split(entries)
    p_train, y_train = _arrays(train_entries)
    p_holdout, y_holdout = _arrays(holdout_entries)

    a_wf, b_wf = fit_platt(p_train, y_train)
    print(f"Walk-forward fit: a={a_wf:.6f}, b={b_wf:.6f}")
    train_metrics = _print_metrics("  Train", p_train, y_train, a_wf, b_wf)
    holdout_metrics = _print_metrics("  Holdout", p_holdout, y_holdout, a_wf, b_wf)

    p_all, y_all = _arrays(entries)
    a_full, b_full = fit_platt(p_all, y_all)
    p_all_cal = apply_calibration(p_all, a_full, b_full)
    full_metrics = _print_metrics("  Full refit", p_all, y_all, a_full, b_full)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "a": a_full,
        "b": b_full,
        "fit_date": date.today().isoformat(),
        "n_games": len(entries),
        "insample_brier_before": train_metrics["brier_before"],
        "insample_brier_after": train_metrics["brier_after"],
        "holdout_brier_before": holdout_metrics["brier_before"],
        "holdout_brier_after": holdout_metrics["brier_after"],
        "full_brier_before": full_metrics["brier_before"],
        "full_brier_after": full_metrics["brier_after"],
        "manual_maintenance": "Re-run this script periodically, e.g. weekly; not wired into the hourly pipeline.",
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    print("\nCalibration bins by raw prediction bucket:")
    print("Bin       N    Raw    Calibrated  Actual")
    for row in calibration_bins(p_all, p_all_cal, y_all):
        print(
            f"{row['bin']:9} {row['n']:4d}  "
            f"{row['raw']:.3f}     {row['calibrated']:.3f}     {row['actual']:.3f}"
        )
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
