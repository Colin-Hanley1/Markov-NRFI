#!/usr/bin/env python3
"""
check_results.py — Check actual 1st-inning results against predictions.

Reads docs/data/latest.json, checks which games have completed,
fetches actual 1st-inning scores, and updates docs/data/results_log.json
with a running accuracy tracker.
"""

from __future__ import annotations
import json
import requests
import numpy as np
from datetime import date
from pathlib import Path

MLB_API = "https://statsapi.mlb.com/api/v1"
DOCS_DATA = Path("docs/data")


def main():
    latest_path = DOCS_DATA / "latest.json"
    log_path = DOCS_DATA / "results_log.json"

    if not latest_path.exists():
        print("No latest.json found, skipping.", flush=True)
        return

    predictions = json.load(open(latest_path))
    pred_date = predictions.get("date", "")
    games = predictions.get("games", [])
    modeled = [g for g in games if g.get("modeled")]

    if not modeled:
        print("No modeled games to check.", flush=True)
        return

    # Load existing log
    if log_path.exists():
        log = json.load(open(log_path))
    else:
        log = {"entries": [], "cumulative": {}}

    # Check if we already logged this date
    logged_ids = {e["game_id"] for e in log["entries"]}

    new_entries = 0
    for g in modeled:
        gid = g["game_id"]
        if gid in logged_ids:
            continue

        # Fetch linescore
        try:
            r = requests.get(f"{MLB_API}/game/{gid}/linescore", timeout=10)
            r.raise_for_status()
            ls = r.json()

            status = ls.get("currentInningOrdinal", "")
            innings = ls.get("innings", [])

            # Only log if game has completed at least the 1st inning
            if not innings:
                continue

            inn1 = innings[0]
            away_r = inn1.get("away", {}).get("runs")
            home_r = inn1.get("home", {}).get("runs")

            if away_r is None or home_r is None:
                continue

            nrfi_actual = away_r == 0 and home_r == 0
            p_pred = g["results"]["p_nrfi_game"]

            entry = {
                "game_id": gid,
                "date": pred_date,
                "away_team": g["away_team"],
                "home_team": g["home_team"],
                "p_nrfi_predicted": round(p_pred, 4),
                "nrfi_actual": nrfi_actual,
                "away_runs_1st": away_r,
                "home_runs_1st": home_r,
                "correct": (p_pred >= 0.5) == nrfi_actual,
                "lineup_source": g.get("lineup_source", "unknown"),
            }
            log["entries"].append(entry)
            logged_ids.add(gid)
            new_entries += 1

            result_str = "NRFI" if nrfi_actual else f"{away_r}-{home_r}"
            correct_str = "correct" if entry["correct"] else "MISS"
            print(f"  {g['away_team']}@{g['home_team']}: pred={p_pred:.1%} actual={result_str} [{correct_str}]", flush=True)

        except Exception as e:
            continue

    if new_entries == 0:
        print("No new results to log (games not started or already logged).", flush=True)
        return

    # Compute cumulative stats
    entries = log["entries"]
    n = len(entries)
    if n > 0:
        y_true = np.array([int(e["nrfi_actual"]) for e in entries], dtype=float)
        y_pred = np.array([e["p_nrfi_predicted"] for e in entries])

        brier = float(np.mean((y_pred - y_true) ** 2))
        actual_rate = float(y_true.mean())
        brier_naive = float(np.mean((np.full_like(y_true, actual_rate) - y_true) ** 2))
        y_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
        logloss = float(-np.mean(y_true * np.log(y_clip) + (1 - y_true) * np.log(1 - y_clip)))
        accuracy = float(np.mean(np.array([e["correct"] for e in entries])))

        # Per-date breakdown
        dates = sorted(set(e["date"] for e in entries))
        daily = []
        for d in dates:
            day_entries = [e for e in entries if e["date"] == d]
            day_correct = sum(1 for e in day_entries if e["correct"])
            day_nrfi = sum(1 for e in day_entries if e["nrfi_actual"])
            daily.append({
                "date": d,
                "n_games": len(day_entries),
                "correct": day_correct,
                "accuracy": round(day_correct / len(day_entries), 4),
                "nrfi_count": day_nrfi,
            })

        log["cumulative"] = {
            "n_games": n,
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier, 5),
            "brier_skill_score": round(1 - brier / max(brier_naive, 1e-10), 4),
            "log_loss": round(logloss, 5),
            "actual_nrfi_rate": round(actual_rate, 4),
            "mean_predicted": round(float(y_pred.mean()), 4),
            "last_updated": date.today().isoformat(),
        }
        log["daily"] = daily

    # Write
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nLogged {new_entries} new results. Total: {n} games tracked.", flush=True)
    c = log["cumulative"]
    print(f"  Accuracy: {c['accuracy']:.1%} | Brier: {c['brier_score']:.4f} | BSS: {c['brier_skill_score']:.1%}", flush=True)


if __name__ == "__main__":
    main()
