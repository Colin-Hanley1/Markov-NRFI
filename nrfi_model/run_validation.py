#!/usr/bin/env python3
"""
run_validation.py — Backtest the NRFI model against historical games.

Fetches completed games from the MLB API, extracts actual lineups and
1st-inning results, runs the model, and computes accuracy metrics.

Usage:
    python3 run_validation.py                         # last 7 days
    python3 run_validation.py 2026-03-27 2026-04-09   # date range
    python3 run_validation.py 2026-03-27 2026-04-09 --sims 20000
"""

from __future__ import annotations
import sys
import time
import requests
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional

from model.outcomes import PAOutcomeRates
from model.blend import build_blended_rates
from model.chain import simulate_nrfi, compute_nrfi_analytic

DATA_DIR = "data/"
MLB_API = "https://statsapi.mlb.com/api/v1"

TEAM_ABBREVS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

RATE_COLUMNS = [
    'k_rate', 'bb_rate', 'hbp_rate', 'hr_rate',
    'single_rate', 'double_rate', 'triple_rate',
    'gbout_rate', 'fbout_rate', 'ldout_rate', 'fc_rate',
    'gidp_prob_given_gbout', 'sf_prob_given_fbout', 'sac_bunt_prob',
]


def load_model_data():
    """Load pitcher/batter profiles, park factors, and league averages."""
    pitchers = pd.read_csv(f"{DATA_DIR}pitchers.csv", dtype={"pitcher_id": str}).set_index("pitcher_id")
    batters = pd.read_csv(f"{DATA_DIR}batters.csv", dtype={"batter_id": str}).set_index("batter_id")
    parks = pd.read_csv(f"{DATA_DIR}park_factors.csv").set_index("team")
    league = pd.read_csv(f"{DATA_DIR}league_averages.csv").iloc[0]
    return {"pitchers": pitchers, "batters": batters, "parks": parks, "league": league}


def get_player_rates(player_id: str, player_type: str, data: dict) -> dict:
    league = data["league"].to_dict()
    if player_type == "pitcher":
        df = data["pitchers"]
        if player_id in df.index:
            return df.loc[player_id].to_dict()
        fallback = dict(league)
        fallback["hand"] = "R"
        fallback["name"] = f"Unknown ({player_id})"
        return fallback
    else:
        df = data["batters"]
        if player_id in df.index:
            return df.loc[player_id].to_dict()
        fallback = dict(league)
        fallback["hand"] = "R"
        fallback["name"] = f"Unknown ({player_id})"
        fallback["sac_bunt_prob"] = 0.0
        return fallback


def build_matchup_rates(pitcher_row, batter_row, league, park_row):
    pitcher_hand = pitcher_row.get("hand", "R")
    batter_hand = batter_row.get("hand", "R")
    hr_pf = float(park_row.get("hr_factor", 1.0))
    run_pf = float(park_row.get("run_factor", 1.0))
    blended = build_blended_rates(
        pitcher_stats=pitcher_row, batter_stats=batter_row,
        league_averages=league, hr_park_factor=hr_pf,
        pitcher_hand=pitcher_hand, batter_hand=batter_hand,
        run_park_factor=run_pf,
    )
    return PAOutcomeRates(**{k: blended.get(k, 0.0) for k in RATE_COLUMNS})


def fetch_completed_games(start_date: str, end_date: str) -> List[dict]:
    """Fetch all completed regular-season games in a date range."""
    url = (
        f"{MLB_API}/schedule?sportId=1&gameType=R"
        f"&startDate={start_date}&endDate={end_date}"
    )
    data = requests.get(url, timeout=15).json()
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                games.append({
                    "game_pk": g["gamePk"],
                    "date": d["date"],
                    "away_id": g["teams"]["away"]["team"]["id"],
                    "home_id": g["teams"]["home"]["team"]["id"],
                })
    return games


def fetch_game_data(game_pk: int) -> Optional[dict]:
    """Fetch boxscore + linescore for a single game."""
    try:
        url = f"{MLB_API}.1/game/{game_pk}/feed/live"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()

        linescore = data["liveData"]["linescore"]
        boxscore = data["liveData"]["boxscore"]

        innings = linescore.get("innings", [])
        if not innings:
            return None

        inn1 = innings[0]
        away_runs_1 = inn1.get("away", {}).get("runs", 0)
        home_runs_1 = inn1.get("home", {}).get("runs", 0)

        away_box = boxscore["teams"]["away"]
        home_box = boxscore["teams"]["home"]

        away_bo = away_box.get("battingOrder", [])[:9]
        home_bo = home_box.get("battingOrder", [])[:9]
        away_sp = away_box.get("pitchers", [None])[0]
        home_sp = home_box.get("pitchers", [None])[0]

        if len(away_bo) < 9 or len(home_bo) < 9 or not away_sp or not home_sp:
            return None

        return {
            "away_runs_1": away_runs_1,
            "home_runs_1": home_runs_1,
            "nrfi": away_runs_1 == 0 and home_runs_1 == 0,
            "away_lineup": [str(pid) for pid in away_bo],
            "home_lineup": [str(pid) for pid in home_bo],
            "away_sp": str(away_sp),
            "home_sp": str(home_sp),
        }
    except Exception as e:
        return None


def run_model_for_game(game_data: dict, home_team: str, data: dict, n_sims: int) -> Optional[dict]:
    """Run the NRFI model for a single historical game."""
    league = data["league"].to_dict()

    park_row = data["parks"].loc[home_team] if home_team in data["parks"].index else pd.Series({"hr_factor": 1.0, "run_factor": 1.0})

    # Away batters vs home SP
    home_sp_row = get_player_rates(game_data["home_sp"], "pitcher", data)
    away_rates = []
    for bid in game_data["away_lineup"]:
        batter_row = get_player_rates(bid, "batter", data)
        away_rates.append(build_matchup_rates(home_sp_row, batter_row, league, park_row))

    # Home batters vs away SP
    away_sp_row = get_player_rates(game_data["away_sp"], "pitcher", data)
    home_rates = []
    for bid in game_data["home_lineup"]:
        batter_row = get_player_rates(bid, "batter", data)
        home_rates.append(build_matchup_rates(away_sp_row, batter_row, league, park_row))

    # Simulate
    sim = simulate_nrfi(home_rates, away_rates, n_simulations=n_sims, seed=42)

    # Analytic
    p_away_a, _ = compute_nrfi_analytic(away_rates)
    p_home_a, _ = compute_nrfi_analytic(home_rates)

    return {
        "p_nrfi_sim": sim["p_nrfi_game"],
        "p_nrfi_analytic": p_away_a * p_home_a,
        "p_nrfi_away": sim["p_nrfi_away"],
        "p_nrfi_home": sim["p_nrfi_home"],
    }


def brier_score(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))


def log_loss(y_true, y_pred, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def calibration_table(y_true, y_pred, n_bins=5):
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[i]:.0%}-{bins[i+1]:.0%}",
            "count": int(mask.sum()),
            "mean_pred": round(float(y_pred[mask].mean()), 4),
            "actual_rate": round(float(y_true[mask].mean()), 4),
            "diff": round(float(y_true[mask].mean()) - float(y_pred[mask].mean()), 4),
        })
    return rows


def main():
    # Parse args
    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    else:
        end_date = (date.today() - timedelta(days=1)).isoformat()
        start_date = (date.today() - timedelta(days=7)).isoformat()

    n_sims = 20000
    for arg in sys.argv:
        if arg.startswith("--sims="):
            n_sims = int(arg.split("=")[1])
        elif arg == "--sims" and sys.argv.index(arg) + 1 < len(sys.argv):
            n_sims = int(sys.argv[sys.argv.index(arg) + 1])

    print("=" * 62)
    print("  NRFI Model — Historical Validation")
    print("=" * 62)
    print(f"  Date range:   {start_date} to {end_date}")
    print(f"  Simulations:  {n_sims:,} per game")
    print()

    # Load model data
    print("Loading player profiles...", flush=True)
    data = load_model_data()
    print(f"  {len(data['pitchers'])} pitchers, {len(data['batters'])} batters loaded")

    # Fetch games
    print(f"\nFetching completed games...", flush=True)
    games = fetch_completed_games(start_date, end_date)
    print(f"  {len(games)} completed games found")

    # Process each game
    print(f"\nRunning model on each game...", flush=True)
    results = []
    skipped = 0

    for i, game in enumerate(games):
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(games)} games processed", flush=True)

        away_abbr = TEAM_ABBREVS.get(game["away_id"], "???")
        home_abbr = TEAM_ABBREVS.get(game["home_id"], "???")

        gd = fetch_game_data(game["game_pk"])
        if gd is None:
            skipped += 1
            continue

        try:
            model_result = run_model_for_game(gd, home_abbr, data, n_sims)
            if model_result is None:
                skipped += 1
                continue

            results.append({
                "game_pk": game["game_pk"],
                "date": game["date"],
                "away": away_abbr,
                "home": home_abbr,
                "away_runs_1": gd["away_runs_1"],
                "home_runs_1": gd["home_runs_1"],
                "nrfi_actual": int(gd["nrfi"]),
                "p_nrfi_sim": model_result["p_nrfi_sim"],
                "p_nrfi_analytic": model_result["p_nrfi_analytic"],
                "p_nrfi_away": model_result["p_nrfi_away"],
                "p_nrfi_home": model_result["p_nrfi_home"],
            })
        except Exception as e:
            print(f"  ERROR game {game['game_pk']} ({away_abbr}@{home_abbr}): {e}", flush=True)
            skipped += 1

        time.sleep(0.1)  # rate limit

    print(f"\n  Processed: {len(results)} games, Skipped: {skipped}")

    if len(results) < 5:
        print("\nToo few games to compute meaningful metrics.")
        return

    # Compute metrics
    df = pd.DataFrame(results)
    y_true = df["nrfi_actual"].values.astype(float)
    y_sim = df["p_nrfi_sim"].values
    y_ana = df["p_nrfi_analytic"].values

    actual_rate = y_true.mean()
    mean_pred_sim = y_sim.mean()
    mean_pred_ana = y_ana.mean()

    bs_sim = brier_score(y_true, y_sim)
    bs_ana = brier_score(y_true, y_ana)
    bs_naive = brier_score(y_true, np.full_like(y_true, actual_rate))

    ll_sim = log_loss(y_true, y_sim)
    ll_ana = log_loss(y_true, y_ana)

    # Accuracy at 50% threshold
    acc_sim = np.mean((y_sim >= 0.5) == y_true)
    acc_ana = np.mean((y_ana >= 0.5) == y_true)

    print(f"\n{'=' * 62}")
    print(f"  Validation Results — {len(results)} games")
    print(f"{'=' * 62}")
    print(f"  Actual NRFI rate:          {actual_rate:.4f} ({int(y_true.sum())}/{len(y_true)})")
    print()
    print(f"  {'Metric':<28} {'Simulation':>12} {'Analytic':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Mean predicted P(NRFI)':<28} {mean_pred_sim:>12.4f} {mean_pred_ana:>12.4f}")
    print(f"  {'Brier score':<28} {bs_sim:>12.5f} {bs_ana:>12.5f}")
    print(f"  {'Brier score (naive)':<28} {bs_naive:>12.5f} {bs_naive:>12.5f}")
    print(f"  {'Brier skill score':<28} {1 - bs_sim/bs_naive:>12.4f} {1 - bs_ana/bs_naive:>12.4f}")
    print(f"  {'Log loss':<28} {ll_sim:>12.5f} {ll_ana:>12.5f}")
    print(f"  {'Accuracy (≥50% threshold)':<28} {acc_sim:>12.4f} {acc_ana:>12.4f}")

    print(f"\n  Calibration — Simulation:")
    print(f"  {'Bin':<12} {'Count':>6} {'Predicted':>10} {'Actual':>10} {'Diff':>8}")
    print(f"  {'-'*48}")
    for row in calibration_table(y_true, y_sim):
        print(f"  {row['bin']:<12} {row['count']:>6} {row['mean_pred']:>10.4f} {row['actual_rate']:>10.4f} {row['diff']:>+8.4f}")

    print(f"\n  Calibration — Analytic:")
    print(f"  {'Bin':<12} {'Count':>6} {'Predicted':>10} {'Actual':>10} {'Diff':>8}")
    print(f"  {'-'*48}")
    for row in calibration_table(y_true, y_ana):
        print(f"  {row['bin']:<12} {row['count']:>6} {row['mean_pred']:>10.4f} {row['actual_rate']:>10.4f} {row['diff']:>+8.4f}")

    # Biggest misses
    df["miss_sim"] = (df["p_nrfi_sim"] - df["nrfi_actual"]).abs()
    worst = df.nlargest(5, "miss_sim")
    print(f"\n  Top 5 biggest misses (simulation):")
    print(f"  {'Game':<18} {'Pred':>6} {'Actual':>7} {'1st Inn':>10}")
    print(f"  {'-'*43}")
    for _, r in worst.iterrows():
        outcome = "NRFI" if r["nrfi_actual"] else f"{int(r['away_runs_1'])}-{int(r['home_runs_1'])}"
        print(f"  {r['away']+'@'+r['home']:<10} {r['date']:<8} {r['p_nrfi_sim']:>5.1%} {'NRFI' if r['nrfi_actual'] else 'RUN':>7} {outcome:>10}")

    print(f"\n{'=' * 62}")

    # Save results CSV
    df.to_csv("data/validation_results.csv", index=False, float_format="%.4f")
    print(f"  Detailed results saved to data/validation_results.csv")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
