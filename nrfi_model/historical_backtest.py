#!/usr/bin/env python3
"""
historical_backtest.py — Run validation across a wider date range and
output summary statistics as JSON for the frontend Results Tracker.

Usage:
    python3 historical_backtest.py                 # last 60 days
    python3 historical_backtest.py 2026-02-15 2026-04-13
"""

from __future__ import annotations
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

from model.outcomes import PAOutcomeRates
from model.blend import build_blended_rates
from model.chain import simulate_nrfi, compute_nrfi_analytic

DATA_DIR = "data/"
OUT_PATH = Path("docs/data/historical_stats.json")
MLB_API = "https://statsapi.mlb.com/api/v1"
N_SIMS = 10000  # reduced for speed across many games

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


def load_data():
    return {
        "pitchers": pd.read_csv(f"{DATA_DIR}pitchers.csv", dtype={"pitcher_id": str}).set_index("pitcher_id"),
        "batters":  pd.read_csv(f"{DATA_DIR}batters.csv",  dtype={"batter_id":  str}).set_index("batter_id"),
        "parks":    pd.read_csv(f"{DATA_DIR}park_factors.csv").set_index("team"),
        "league":   pd.read_csv(f"{DATA_DIR}league_averages.csv").iloc[0],
    }


def get_player_rates(pid, ptype, data):
    league = data["league"].to_dict()
    df = data["pitchers"] if ptype == "pitcher" else data["batters"]
    if pid in df.index:
        return df.loc[pid].to_dict()
    fb = dict(league)
    fb["hand"] = "R"
    if ptype != "pitcher":
        fb["sac_bunt_prob"] = 0.0
    return fb


def build_rates(pitcher, batter, league, park):
    blended = build_blended_rates(
        pitcher_stats=pitcher, batter_stats=batter,
        league_averages=league,
        hr_park_factor=float(park.get("hr_factor", 1.0)),
        pitcher_hand=pitcher.get("hand", "R"),
        batter_hand=batter.get("hand", "R"),
        run_park_factor=float(park.get("run_factor", 1.0)),
    )
    return PAOutcomeRates(**{k: blended.get(k, 0.0) for k in RATE_COLUMNS})


def fetch_completed_games(start, end):
    url = f"{MLB_API}/schedule?sportId=1&gameType=R&startDate={start}&endDate={end}"
    data = requests.get(url, timeout=15).json()
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                games.append({
                    "pk": g["gamePk"],
                    "date": d["date"],
                    "away_id": g["teams"]["away"]["team"]["id"],
                    "home_id": g["teams"]["home"]["team"]["id"],
                })
    return games


def fetch_game_data(pk):
    try:
        url = f"{MLB_API}.1/game/{pk}/feed/live"
        d = requests.get(url, timeout=15).json()
        ls = d["liveData"]["linescore"]
        bs = d["liveData"]["boxscore"]
        innings = ls.get("innings", [])
        if not innings:
            return None
        inn1 = innings[0]
        ar = inn1.get("away", {}).get("runs", 0)
        hr = inn1.get("home", {}).get("runs", 0)
        away_box = bs["teams"]["away"]
        home_box = bs["teams"]["home"]
        a_bo = away_box.get("battingOrder", [])[:9]
        h_bo = home_box.get("battingOrder", [])[:9]
        a_sp = (away_box.get("pitchers") or [None])[0]
        h_sp = (home_box.get("pitchers") or [None])[0]
        if len(a_bo) < 9 or len(h_bo) < 9 or not a_sp or not h_sp:
            return None
        return {
            "ar": ar, "hr": hr,
            "nrfi": ar == 0 and hr == 0,
            "away_lineup": [str(x) for x in a_bo],
            "home_lineup": [str(x) for x in h_bo],
            "away_sp": str(a_sp),
            "home_sp": str(h_sp),
        }
    except Exception:
        return None


def run_model(gd, home_team, data):
    league = data["league"].to_dict()
    park = data["parks"].loc[home_team] if home_team in data["parks"].index else pd.Series({"hr_factor": 1.0, "run_factor": 1.0})

    home_sp = get_player_rates(gd["home_sp"], "pitcher", data)
    away_sp = get_player_rates(gd["away_sp"], "pitcher", data)
    away_rates = [build_rates(home_sp, get_player_rates(b, "batter", data), league, park) for b in gd["away_lineup"]]
    home_rates = [build_rates(away_sp, get_player_rates(b, "batter", data), league, park) for b in gd["home_lineup"]]

    sim = simulate_nrfi(home_rates, away_rates, n_simulations=N_SIMS, seed=42)
    p_a, _ = compute_nrfi_analytic(away_rates)
    p_h, _ = compute_nrfi_analytic(home_rates)
    return {
        "p_sim": sim["p_nrfi_game"],
        "p_analytic": p_a * p_h,
    }


def main():
    if len(sys.argv) >= 3:
        start, end = sys.argv[1], sys.argv[2]
    else:
        end = (date.today() - timedelta(days=1)).isoformat()
        start = (date.today() - timedelta(days=60)).isoformat()

    print(f"Historical backtest {start} → {end}", flush=True)
    data = load_data()
    print(f"  {len(data['pitchers'])} pitchers, {len(data['batters'])} batters", flush=True)

    games = fetch_completed_games(start, end)
    print(f"  {len(games)} completed games", flush=True)

    results = []
    entries = []
    for i, g in enumerate(games):
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(games)} processed", flush=True)
        gd = fetch_game_data(g["pk"])
        if not gd:
            continue
        try:
            home_abbr = TEAM_ABBREVS.get(g["home_id"], "???")
            away_abbr = TEAM_ABBREVS.get(g["away_id"], "???")
            mr = run_model(gd, home_abbr, data)
            results.append({
                "date": g["date"],
                "p_pred": mr["p_sim"],
                "p_analytic": mr["p_analytic"],
                "nrfi": gd["nrfi"],
            })
            entries.append({
                "game_id": str(g["pk"]),
                "date": g["date"],
                "away_team": away_abbr,
                "home_team": home_abbr,
                "p_nrfi_predicted": round(mr["p_sim"], 4),
                "nrfi_actual": gd["nrfi"],
                "away_runs_1st": gd["ar"],
                "home_runs_1st": gd["hr"],
                "correct": (mr["p_sim"] >= 0.5) == gd["nrfi"],
                "lineup_source": "historical",
            })
        except Exception:
            pass
        time.sleep(0.05)

    if not results:
        print("No results")
        return

    df = pd.DataFrame(results)
    y = df["nrfi"].astype(float).values
    p = df["p_pred"].values

    brier = float(np.mean((p - y) ** 2))
    actual_rate = float(y.mean())
    brier_naive = float(np.mean((np.full_like(y, actual_rate) - y) ** 2))
    bss = 1 - brier / max(brier_naive, 1e-10)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    ll = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    accuracy = float(np.mean((p >= 0.5) == y))

    # Calibration bins
    bin_edges = np.linspace(0.30, 0.70, 9)  # 5pp bins
    cal = []
    for i in range(len(bin_edges) - 1):
        m = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        if m.sum() >= 3:
            cal.append({
                "bin": f"{bin_edges[i]:.0%}-{bin_edges[i+1]:.0%}",
                "lo": float(bin_edges[i]),
                "hi": float(bin_edges[i + 1]),
                "n": int(m.sum()),
                "predicted": round(float(p[m].mean()), 4),
                "actual": round(float(y[m].mean()), 4),
            })

    # Confidence tiers
    tiers = []
    for tier_def in [
        ("Strong NRFI (>58%)", p >= 0.58),
        ("Lean NRFI (52-58%)", (p >= 0.52) & (p < 0.58)),
        ("Coin Flip (48-52%)", (p >= 0.48) & (p < 0.52)),
        ("Lean RFI (42-48%)", (p >= 0.42) & (p < 0.48)),
        ("Strong RFI (<42%)", p < 0.42),
    ]:
        name, mask = tier_def
        if mask.sum() == 0:
            continue
        tiers.append({
            "tier": name,
            "n": int(mask.sum()),
            "predicted": round(float(p[mask].mean()), 4),
            "actual_nrfi_rate": round(float(y[mask].mean()), 4),
            "accuracy": round(float(np.mean((p[mask] >= 0.5) == y[mask])), 4),
        })

    out = {
        "date_range": [start, end],
        "n_games": len(results),
        "computed_at": pd.Timestamp.now().isoformat(),
        "actual_nrfi_rate": round(actual_rate, 4),
        "mean_predicted": round(float(p.mean()), 4),
        "brier_score": round(brier, 5),
        "brier_skill_score": round(bss, 4),
        "log_loss": round(ll, 5),
        "accuracy": round(accuracy, 4),
        "calibration": cal,
        "tiers": tiers,
        "entries": entries,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nDone: {len(results)} games")
    print(f"  Brier: {brier:.4f}  BSS: {bss:.1%}  Accuracy: {accuracy:.1%}")
    print(f"  Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
