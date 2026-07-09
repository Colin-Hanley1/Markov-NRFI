#!/usr/bin/env python3
"""
Analyze flat-stake Kalshi NRFI/YRFI betting strategies from joined history.

Reads:
    data/kalshi_backtest_joined.csv

Outputs:
    docs/data/kalshi_strategy_report.json
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from compute_daily import kalshi_taker_fee


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data/kalshi_backtest_joined.csv"
REPORT_PATH = BASE_DIR / "docs/data/kalshi_strategy_report.json"
THRESHOLDS_PP = [0, 3, 5, 8, 12, 15, 20]


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _to_float(value: object) -> float:
    return float(str(value).strip())


def _load_rows() -> list[dict]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} does not exist. Run backtest_kalshi.py first."
        )
    with INPUT_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _brier_score(probs: list[float], outcomes: list[bool]) -> float:
    return _mean([(p - (1.0 if actual else 0.0)) ** 2 for p, actual in zip(probs, outcomes)])


def _ci95(values: list[float]) -> tuple[float | None, float | None]:
    n = len(values)
    if n < 2:
        return None, None
    avg = _mean(values)
    variance = sum((v - avg) ** 2 for v in values) / (n - 1)
    half_width = 1.96 * math.sqrt(variance) / math.sqrt(n)
    return avg - half_width, avg + half_width


def _strategy_result(rows: list[dict], threshold: int, side: str) -> dict:
    returns = []
    net_returns = []
    wins = 0
    for row in rows:
        edge = _to_float(row["edge_pp"])
        actual_nrfi = _to_bool(row["actual_nrfi"])
        if side == "NRFI":
            if edge < threshold:
                continue
            cost = _to_float(row["kalshi_nrfi_ask"])
            won = actual_nrfi
        else:
            if edge > -threshold:
                continue
            cost = _to_float(row["kalshi_yrfi_ask"])
            won = not actual_nrfi

        payoff = 1.0 if won else 0.0
        trade_return = payoff - cost
        returns.append(trade_return)
        net_returns.append(trade_return - kalshi_taker_fee(cost))
        wins += int(won)

    n_bets = len(returns)
    total_profit = sum(returns)
    net_total_profit = sum(net_returns)
    ci_low, ci_high = _ci95(returns)
    return {
        "threshold_pp": threshold,
        "side": side,
        "n_bets": n_bets,
        "win_rate": round(wins / n_bets, 4) if n_bets else None,
        "total_profit": round(total_profit, 4),
        "net_total_profit": round(net_total_profit, 4),
        "roi_pct": round(_mean(returns) * 100, 2) if n_bets else None,
        "net_roi_pct": round(_mean(net_returns) * 100, 2) if n_bets else None,
        "mean_return_ci95": [
            round(ci_low, 4) if ci_low is not None else None,
            round(ci_high, 4) if ci_high is not None else None,
        ],
    }


def _print_table(results: list[dict]) -> None:
    headers = [
        "threshold",
        "side",
        "bets",
        "win_rate",
        "profit",
        "roi_pct",
        "net_roi_pct",
        "mean_return_95ci",
    ]
    print(" | ".join(headers))
    print(" | ".join("-" * len(h) for h in headers))
    for result in results:
        ci_low, ci_high = result["mean_return_ci95"]
        ci = "n/a" if ci_low is None else f"[{ci_low:.4f}, {ci_high:.4f}]"
        win_rate = "n/a"
        if result["win_rate"] is not None:
            win_rate = f"{result['win_rate']:.1%}"
        roi = "n/a" if result["roi_pct"] is None else f"{result['roi_pct']:.2f}"
        net_roi = "n/a" if result["net_roi_pct"] is None else f"{result['net_roi_pct']:.2f}"
        print(
            " | ".join(
                [
                    str(result["threshold_pp"]),
                    result["side"],
                    str(result["n_bets"]),
                    win_rate,
                    f"{result['total_profit']:.4f}",
                    roi,
                    net_roi,
                    ci,
                ]
            )
        )


def main() -> None:
    rows = _load_rows()
    results = []
    for threshold in THRESHOLDS_PP:
        results.append(_strategy_result(rows, threshold, "NRFI"))
        results.append(_strategy_result(rows, threshold, "YRFI"))

    print(f"Kalshi strategy report from {len(rows)} joined games\n")
    _print_table(results)
    print("\nCaveats")
    print(
        "- ~850 games over ~9 weeks is a modest sample; high-threshold buckets "
        "will have few bets and noisy ROI. Treat single-threshold wins skeptically "
        "without more data or an out-of-sample holdout."
    )
    print(
        "- Uses realistic ask-side fill prices by crossing the spread. It does "
        "not model slippage from thin liquidity. Net ROI subtracts "
        "Kalshi's standard taker fee per contract."
    )

    model_probs = [_to_float(row["model_p_nrfi"]) for row in rows]
    market_probs = [_to_float(row["kalshi_nrfi_ask"]) for row in rows]
    outcomes = [_to_bool(row["actual_nrfi"]) for row in rows]
    dates = sorted({row["date"] for row in rows})

    report = {
        "source": str(INPUT_PATH),
        "n_games": len(rows),
        "date_range": [dates[0], dates[-1]] if dates else [],
        "brier": {
            "model": round(_brier_score(model_probs, outcomes), 5) if rows else None,
            "market": round(_brier_score(market_probs, outcomes), 5) if rows else None,
        },
        "thresholds_pp": THRESHOLDS_PP,
        "results": results,
        "caveats": [
            "~850 games over ~9 weeks is a modest sample; high-threshold buckets "
            "will have few bets and noisy ROI. Treat single-threshold wins "
            "skeptically without more data or an out-of-sample holdout.",
            "Uses realistic ask-side fill prices by crossing the spread. It does "
            "not model slippage from thin liquidity. Net ROI subtracts Kalshi's "
            "standard taker fee per contract.",
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
