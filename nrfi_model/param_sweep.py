#!/usr/bin/env python3
"""
param_sweep.py — Grid search over runner-advancement probabilities and
recency weights, using cached backtest data for speed.

Phase 1 (cache): fetches all game lineups and per-season player stats for
the backtest date range once, pickles them.
Phase 2 (sweep): for each parameter combo, reconstructs player profiles in
memory from cached raw stats, runs the analytic Markov solution on each
cached game, and records Brier / log-loss / accuracy.

Usage:
    python3 param_sweep.py                          # last 60 days backtest
    python3 param_sweep.py 2026-03-27 2026-04-12    # explicit range
"""

from __future__ import annotations
import sys
import json
import pickle
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from itertools import product

from model.outcomes import PAOutcomeRates
from model.blend import build_blended_rates
from model.chain import compute_nrfi_analytic
from model import transitions as T

MLB_API = "https://statsapi.mlb.com/api/v1"
DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

SEASONS = [2024, 2025, 2026]
AIR_OUT_FB_FRAC = 0.80
FC_FRAC_OF_GBOUT = 0.06

PITCHER_COUNT_KEYS = [
    "strikeOuts", "baseOnBalls", "intentionalWalks", "hitByPitch",
    "homeRuns", "hits", "doubles", "triples",
    "groundOuts", "airOuts", "groundIntoDoublePlay", "sacFlies",
    "battersFaced",
]
BATTER_COUNT_KEYS = [
    "strikeOuts", "baseOnBalls", "intentionalWalks", "hitByPitch",
    "homeRuns", "hits", "doubles", "triples",
    "groundOuts", "airOuts", "groundIntoDoublePlay", "sacFlies",
    "sacBunts", "plateAppearances",
]

RATE_COLUMNS = [
    "k_rate", "bb_rate", "hbp_rate", "hr_rate",
    "single_rate", "double_rate", "triple_rate",
    "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
    "gidp_prob_given_gbout", "sf_prob_given_fbout", "sac_bunt_prob",
]

TEAM_ABBREVS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


# ── Cache: raw per-season stats per player ───────────────────────────────

def fetch_season_stats(player_id: int, group: str) -> dict:
    """Returns dict {season_int: counting_stats_dict} for all seasons found."""
    out = {}
    for season in SEASONS:
        try:
            url = f"{MLB_API}/people/{player_id}/stats?stats=season&season={season}&group={group}"
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            stat_groups = data.get("stats", [])
            if not stat_groups:
                continue
            splits = stat_groups[0].get("splits", [])
            if splits:
                out[season] = splits[0]["stat"]
        except Exception:
            continue
    return out


def fetch_completed_games(start: str, end: str) -> list:
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


def fetch_game_data(pk: int):
    try:
        r = requests.get(f"{MLB_API}.1/game/{pk}/feed/live", timeout=15)
        d = r.json()
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
            "nrfi": ar == 0 and hr == 0,
            "away_lineup": [str(x) for x in a_bo],
            "home_lineup": [str(x) for x in h_bo],
            "away_sp": str(a_sp),
            "home_sp": str(h_sp),
        }
    except Exception:
        return None


def build_cache(start: str, end: str, force: bool = False) -> dict:
    cache_path = CACHE_DIR / f"backtest_{start}_to_{end}.pkl"
    if cache_path.exists() and not force:
        print(f"Loading cache from {cache_path}...", flush=True)
        return pickle.load(open(cache_path, "rb"))

    print(f"Building cache for {start} → {end}...", flush=True)

    # Fetch completed games
    games_meta = fetch_completed_games(start, end)
    print(f"  {len(games_meta)} completed games, fetching boxscores...", flush=True)

    valid_games = []
    all_pitcher_ids = set()
    all_batter_ids = set()

    for i, g in enumerate(games_meta):
        if (i + 1) % 25 == 0:
            print(f"  boxscore {i+1}/{len(games_meta)}", flush=True)
        gd = fetch_game_data(g["pk"])
        if not gd:
            continue
        g.update(gd)
        valid_games.append(g)
        all_pitcher_ids.add(gd["away_sp"])
        all_pitcher_ids.add(gd["home_sp"])
        all_batter_ids.update(gd["away_lineup"])
        all_batter_ids.update(gd["home_lineup"])
        time.sleep(0.05)

    print(f"  {len(valid_games)} valid games, {len(all_pitcher_ids)} pitchers, {len(all_batter_ids)} batters", flush=True)

    # Fetch raw per-season stats for each player
    pitcher_stats = {}
    for i, pid in enumerate(sorted(all_pitcher_ids)):
        if (i + 1) % 20 == 0:
            print(f"  pitcher stats {i+1}/{len(all_pitcher_ids)}", flush=True)
        pitcher_stats[pid] = fetch_season_stats(int(pid), "pitching")
        time.sleep(0.05)

    batter_stats = {}
    for i, bid in enumerate(sorted(all_batter_ids)):
        if (i + 1) % 50 == 0:
            print(f"  batter stats {i+1}/{len(all_batter_ids)}", flush=True)
        batter_stats[bid] = fetch_season_stats(int(bid), "hitting")
        time.sleep(0.05)

    # Player bios (for handedness)
    bios = {}
    print(f"  fetching bios...", flush=True)
    for pid in list(all_pitcher_ids) + list(all_batter_ids):
        try:
            r = requests.get(f"{MLB_API}/people/{pid}", timeout=15)
            people = r.json().get("people", [])
            if people:
                bios[pid] = {
                    "pitchHand": people[0].get("pitchHand", {}).get("code", "R"),
                    "batSide": people[0].get("batSide", {}).get("code", "R"),
                }
        except Exception:
            pass
        time.sleep(0.03)

    cache = {
        "games": valid_games,
        "pitcher_stats": pitcher_stats,
        "batter_stats": batter_stats,
        "bios": bios,
    }
    pickle.dump(cache, open(cache_path, "wb"))
    print(f"  saved to {cache_path}", flush=True)
    return cache


# ── Profile construction from cached raw stats ───────────────────────────

def _apply_recency(season_dict: dict, recency: dict, keys: list) -> dict:
    """Sum counting stats across seasons, scaled by recency weights."""
    totals = {k: 0.0 for k in keys}
    for season, stats in season_dict.items():
        w = recency.get(season, 1.0)
        for k in keys:
            totals[k] += float(stats.get(k, 0)) * w
    return totals


def _rates_from_totals(totals: dict, pa_key: str, include_sac_bunt: bool = False) -> dict:
    pa = totals.get(pa_key, 0)
    if pa < 1:
        return None
    singles = totals["hits"] - totals["doubles"] - totals["triples"] - totals["homeRuns"]
    singles = max(0, singles)
    go = totals["groundOuts"]
    ao = totals["airOuts"]
    fc = round(go * FC_FRAC_OF_GBOUT)
    gb = go - fc
    fb = round(ao * AIR_OUT_FB_FRAC)
    ld = ao - fb
    rates = {
        "k_rate": totals["strikeOuts"] / pa,
        "bb_rate": (totals["baseOnBalls"] - totals["intentionalWalks"]) / pa,
        "hbp_rate": totals["hitByPitch"] / pa,
        "hr_rate": totals["homeRuns"] / pa,
        "single_rate": singles / pa,
        "double_rate": totals["doubles"] / pa,
        "triple_rate": totals["triples"] / pa,
        "gbout_rate": gb / pa,
        "fbout_rate": fb / pa,
        "ldout_rate": ld / pa,
        "fc_rate": fc / pa,
        "gidp_prob_given_gbout": min(0.40, totals["groundIntoDoublePlay"] / max(1, go)),
        "sf_prob_given_fbout": min(0.40, totals["sacFlies"] / max(1, ao)),
    }
    if include_sac_bunt:
        rates["sac_bunt_prob"] = min(0.20, totals["sacBunts"] / max(1, pa))
    primary = [k for k in rates if k not in ("gidp_prob_given_gbout", "sf_prob_given_fbout", "sac_bunt_prob")]
    tot = sum(rates[k] for k in primary)
    if tot > 0:
        for k in primary:
            rates[k] /= tot
    return rates


def build_profiles_from_cache(cache: dict, recency: dict, league_avg: dict) -> tuple:
    pitchers = {}
    for pid, season_stats in cache["pitcher_stats"].items():
        if not season_stats:
            continue
        totals = _apply_recency(season_stats, recency, PITCHER_COUNT_KEYS)
        rates = _rates_from_totals(totals, "battersFaced")
        if rates:
            rates["hand"] = cache["bios"].get(pid, {}).get("pitchHand", "R")
            pitchers[pid] = rates
    batters = {}
    for bid, season_stats in cache["batter_stats"].items():
        if not season_stats:
            continue
        totals = _apply_recency(season_stats, recency, BATTER_COUNT_KEYS)
        rates = _rates_from_totals(totals, "plateAppearances", include_sac_bunt=True)
        if rates:
            bs = cache["bios"].get(bid, {}).get("batSide", "R")
            rates["hand"] = "R" if bs == "S" else bs
            batters[bid] = rates
    return pitchers, batters


# ── Sweep ────────────────────────────────────────────────────────────────

def _lookup(dct, pid, league_avg, is_pitcher):
    if pid in dct:
        return dct[pid]
    fb = dict(league_avg)
    fb["hand"] = "R"
    if not is_pitcher:
        fb["sac_bunt_prob"] = 0.0
    return fb


def _build_rates(pitcher, batter, league, park):
    blended = build_blended_rates(
        pitcher_stats=pitcher, batter_stats=batter,
        league_averages=league,
        hr_park_factor=float(park.get("hr_factor", 1.0)),
        pitcher_hand=pitcher.get("hand", "R"),
        batter_hand=batter.get("hand", "R"),
        run_park_factor=float(park.get("run_factor", 1.0)),
    )
    return PAOutcomeRates(**{k: blended.get(k, 0.0) for k in RATE_COLUMNS})


def score_params(cache: dict, advancement: dict, recency: dict,
                 league_avg: dict, parks: pd.DataFrame) -> dict:
    # Patch advancement params
    T.set_advancement(**advancement)

    pitchers, batters = build_profiles_from_cache(cache, recency, league_avg)

    preds, actuals = [], []
    for g in cache["games"]:
        home_abbr = TEAM_ABBREVS.get(g["home_id"], "???")
        park = parks.loc[home_abbr] if home_abbr in parks.index else pd.Series({"hr_factor": 1.0, "run_factor": 1.0})

        home_sp = _lookup(pitchers, g["home_sp"], league_avg, True)
        away_sp = _lookup(pitchers, g["away_sp"], league_avg, True)

        try:
            away_rates = [_build_rates(home_sp, _lookup(batters, b, league_avg, False), league_avg, park)
                          for b in g["away_lineup"]]
            home_rates = [_build_rates(away_sp, _lookup(batters, b, league_avg, False), league_avg, park)
                          for b in g["home_lineup"]]
            p_a, _ = compute_nrfi_analytic(away_rates)
            p_h, _ = compute_nrfi_analytic(home_rates)
            preds.append(p_a * p_h)
            actuals.append(1 if g["nrfi"] else 0)
        except Exception:
            continue

    if not preds:
        return None

    y = np.array(actuals, dtype=float)
    p = np.array(preds)
    brier = float(np.mean((p - y) ** 2))
    actual_rate = float(y.mean())
    brier_naive = float(np.mean((np.full_like(y, actual_rate) - y) ** 2))
    bss = 1 - brier / max(brier_naive, 1e-10)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    ll = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    acc = float(np.mean((p >= 0.5) == y))

    return {"n": len(preds), "brier": brier, "bss": bss, "log_loss": ll, "accuracy": acc}


def main():
    if len(sys.argv) >= 3:
        start, end = sys.argv[1], sys.argv[2]
    else:
        end = (date.today() - timedelta(days=1)).isoformat()
        start = (date.today() - timedelta(days=60)).isoformat()

    print(f"=== Parameter sweep: {start} → {end} ===\n", flush=True)

    cache = build_cache(start, end)
    league_avg = pd.read_csv(DATA_DIR / "league_averages.csv").iloc[0].to_dict()
    parks = pd.read_csv(DATA_DIR / "park_factors.csv").set_index("team")

    # Grid
    p_2b_scores_vals = [0.50, 0.60, 0.70]
    p_1b_scores_vals = [0.30, 0.40, 0.50]
    p_1b_to_3b_vals  = [0.20, 0.30]
    recency_schemes = {
        "flat (1,1,1)":       {2024: 1.0, 2025: 1.0, 2026: 1.0},
        "linear (1,2,3)":     {2024: 1.0, 2025: 2.0, 2026: 3.0},
        "heavy (1,2,5)":      {2024: 1.0, 2025: 2.0, 2026: 5.0},
        "very-recent (1,3,5)":{2024: 1.0, 2025: 3.0, 2026: 5.0},
    }

    results = []
    combos = list(product(p_2b_scores_vals, p_1b_scores_vals, p_1b_to_3b_vals, recency_schemes.items()))
    print(f"Evaluating {len(combos)} parameter combos...\n", flush=True)

    for i, (p2, p1, p13, (rname, rdict)) in enumerate(combos):
        advancement = {
            "p_2b_scores_single": p2,
            "p_1b_scores_double": p1,
            "p_1b_to_3b_single":  p13,
        }
        r = score_params(cache, advancement, rdict, league_avg, parks)
        if r is None:
            continue
        r["p_2b_scores_single"] = p2
        r["p_1b_scores_double"] = p1
        r["p_1b_to_3b_single"]  = p13
        r["recency_scheme"] = rname
        r["recency"] = rdict
        results.append(r)
        if (i + 1) % 10 == 0 or i == len(combos) - 1:
            print(f"  [{i+1}/{len(combos)}] 2B→s={p2} 1B→d={p1} 1B→3B={p13} recency={rname:20s}   Brier={r['brier']:.4f}  BSS={r['bss']*100:+5.1f}%  Acc={r['accuracy']*100:4.1f}%", flush=True)

    results.sort(key=lambda r: r["brier"])

    print("\n\n=== Top 10 by Brier score ===\n", flush=True)
    for r in results[:10]:
        print(f"  Brier={r['brier']:.4f}  BSS={r['bss']*100:+5.1f}%  Acc={r['accuracy']*100:4.1f}%  "
              f"|  2B→s={r['p_2b_scores_single']} 1B→d={r['p_1b_scores_double']} 1B→3B={r['p_1b_to_3b_single']} recency={r['recency_scheme']}",
              flush=True)

    print(f"\nBEST: {results[0]}", flush=True)

    Path("cache").mkdir(exist_ok=True)
    with open(CACHE_DIR / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {CACHE_DIR}/sweep_results.json", flush=True)


if __name__ == "__main__":
    main()
