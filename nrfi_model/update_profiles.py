#!/usr/bin/env python3
"""
update_profiles.py — Fast daily update of player profiles.

Only fetches the current season's stats and merges with existing
multi-year profiles. Much faster than a full rebuild.

Usage:
    python3 update_profiles.py          # update 2026 stats
    python3 update_profiles.py 2025     # update a specific season
"""

from __future__ import annotations
import sys
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

MLB_API = "https://statsapi.mlb.com/api/v1"
DATA_DIR = "data/"
CURRENT_SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026

AIR_OUT_FB_FRAC = 0.80
FC_FRAC_OF_GBOUT = 0.06

# Recency weights (must match build_profiles.RECENCY_WEIGHTS). When updating
# an existing profile we scale the new season's contribution so the recency
# scheme established during the full rebuild is preserved.
RECENCY_WEIGHTS = {2024: 1.0, 2025: 2.0, 2026: 5.0}
_NEW_SEASON_WEIGHT = RECENCY_WEIGHTS.get(CURRENT_SEASON, 1.0)

PITCHER_COLS = [
    "pitcher_id", "name", "team", "hand", "pa",
    "k_rate", "bb_rate", "hbp_rate", "hr_rate",
    "single_rate", "double_rate", "triple_rate",
    "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
    "gidp_prob_given_gbout", "sf_prob_given_fbout",
]

BATTER_COLS = [
    "batter_id", "name", "team", "hand", "pa",
    "k_rate", "bb_rate", "hbp_rate", "hr_rate",
    "single_rate", "double_rate", "triple_rate",
    "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
    "gidp_prob_given_gbout", "sf_prob_given_fbout", "sac_bunt_prob",
]

LEAGUE_COLS = [
    "k_rate", "bb_rate", "hbp_rate", "hr_rate",
    "single_rate", "double_rate", "triple_rate",
    "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
    "gidp_prob_given_gbout", "sf_prob_given_fbout", "sac_bunt_prob",
]


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1)


def get_all_rosters():
    teams = fetch_json(f"{MLB_API}/teams?sportId=1")["teams"]
    players = {}
    for t in teams:
        roster = fetch_json(f"{MLB_API}/teams/{t['id']}/roster?rosterType=active")
        for p in roster.get("roster", []):
            pid = p["person"]["id"]
            players[pid] = {
                "id": pid,
                "name": p["person"]["fullName"],
                "pos": p["position"]["abbreviation"],
                "team": t["abbreviation"],
                "is_pitcher": p["position"]["abbreviation"] == "P",
            }
    return players


def get_bio(pid):
    data = fetch_json(f"{MLB_API}/people/{pid}")
    people = data.get("people", [])
    if not people:
        return {"fullName": f"Unknown ({pid})", "batSide": {"code": "R"}, "pitchHand": {"code": "R"}}
    return people[0]


def get_season_stats(pid, season, group):
    data = fetch_json(f"{MLB_API}/people/{pid}/stats?stats=season&season={season}&group={group}")
    stat_groups = data.get("stats", [])
    if not stat_groups:
        return None
    splits = stat_groups[0].get("splits", [])
    return splits[0]["stat"] if splits else None


def raw_to_rates(raw, pa_key, include_sac_bunt=False):
    pa = int(raw.get(pa_key, 0))
    if pa < 1:
        return None

    singles = raw.get("hits", 0) - raw.get("doubles", 0) - raw.get("triples", 0) - raw.get("homeRuns", 0)
    singles = max(0, singles)
    go = raw.get("groundOuts", 0)
    ao = raw.get("airOuts", 0)
    fc = round(go * FC_FRAC_OF_GBOUT)
    gb = go - fc
    fb = round(ao * AIR_OUT_FB_FRAC)
    ld = ao - fb

    rates = {
        "k_rate": raw.get("strikeOuts", 0) / pa,
        "bb_rate": (raw.get("baseOnBalls", 0) - raw.get("intentionalWalks", 0)) / pa,
        "hbp_rate": raw.get("hitByPitch", 0) / pa,
        "hr_rate": raw.get("homeRuns", 0) / pa,
        "single_rate": singles / pa,
        "double_rate": raw.get("doubles", 0) / pa,
        "triple_rate": raw.get("triples", 0) / pa,
        "gbout_rate": gb / pa,
        "fbout_rate": fb / pa,
        "ldout_rate": ld / pa,
        "fc_rate": fc / pa,
        "gidp_prob_given_gbout": min(0.40, raw.get("groundIntoDoublePlay", 0) / max(1, go)),
        "sf_prob_given_fbout": min(0.40, raw.get("sacFlies", 0) / max(1, ao)),
    }
    if include_sac_bunt:
        rates["sac_bunt_prob"] = min(0.20, raw.get("sacBunts", 0) / max(1, pa))

    primary = [k for k in rates if k not in ("gidp_prob_given_gbout", "sf_prob_given_fbout", "sac_bunt_prob")]
    total = sum(rates[k] for k in primary)
    if total > 0:
        for k in primary:
            rates[k] /= total
    rates["pa"] = pa
    return rates


def main():
    print(f"Updating profiles with {CURRENT_SEASON} season stats...", flush=True)

    # Load existing profiles
    try:
        old_pitchers = pd.read_csv(f"{DATA_DIR}pitchers.csv", dtype={"pitcher_id": str}).set_index("pitcher_id")
        old_batters = pd.read_csv(f"{DATA_DIR}batters.csv", dtype={"batter_id": str}).set_index("batter_id")
        print(f"  Existing: {len(old_pitchers)} pitchers, {len(old_batters)} batters", flush=True)
    except FileNotFoundError:
        print("  No existing profiles found. Run build_profiles.py first for a full build.")
        return

    # Get active rosters
    print("  Fetching active rosters...", flush=True)
    players = get_all_rosters()
    pitchers = {pid: p for pid, p in players.items() if p["is_pitcher"]}
    batters = {pid: p for pid, p in players.items() if not p["is_pitcher"]}
    print(f"  Active roster: {len(pitchers)} pitchers, {len(batters)} batters", flush=True)

    # Update pitchers
    print(f"  Updating pitcher stats...", flush=True)
    updated_p = 0
    new_p = 0
    for i, (pid, info) in enumerate(sorted(pitchers.items())):
        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{len(pitchers)}", flush=True)
        try:
            raw = get_season_stats(pid, CURRENT_SEASON, "pitching")
            if not raw:
                continue
            rates = raw_to_rates(raw, "battersFaced")
            if not rates:
                continue
            bio = get_bio(pid)
            pid_str = str(pid)

            if pid_str in old_pitchers.index:
                # Weighted merge with recency scale on new season
                old_pa = float(old_pitchers.loc[pid_str].get("pa", 0))
                new_pa = rates["pa"] * _NEW_SEASON_WEIGHT
                total_pa = old_pa + new_pa
                if total_pa > 0:
                    w_old = old_pa / total_pa
                    w_new = new_pa / total_pa
                    for k in ["k_rate", "bb_rate", "hbp_rate", "hr_rate", "single_rate",
                              "double_rate", "triple_rate", "gbout_rate", "fbout_rate",
                              "ldout_rate", "fc_rate", "gidp_prob_given_gbout", "sf_prob_given_fbout"]:
                        old_val = float(old_pitchers.loc[pid_str].get(k, 0))
                        old_pitchers.loc[pid_str, k] = w_old * old_val + w_new * rates[k]
                    old_pitchers.loc[pid_str, "pa"] = total_pa
                    old_pitchers.loc[pid_str, "team"] = info["team"]
                    # Re-normalize
                    primary = ["k_rate", "bb_rate", "hbp_rate", "hr_rate", "single_rate",
                               "double_rate", "triple_rate", "gbout_rate", "fbout_rate",
                               "ldout_rate", "fc_rate"]
                    psum = sum(float(old_pitchers.loc[pid_str, k]) for k in primary)
                    if psum > 0:
                        for k in primary:
                            old_pitchers.loc[pid_str, k] = float(old_pitchers.loc[pid_str, k]) / psum
                updated_p += 1
            else:
                # New player
                row = rates.copy()
                row["name"] = bio.get("fullName", info["name"])
                row["hand"] = bio.get("pitchHand", {}).get("code", "R")
                row["team"] = info["team"]
                old_pitchers.loc[pid_str] = row
                new_p += 1
        except Exception:
            pass
        time.sleep(0.05)

    print(f"    Updated {updated_p}, added {new_p} new pitchers", flush=True)

    # Update batters
    print(f"  Updating batter stats...", flush=True)
    updated_b = 0
    new_b = 0
    for i, (pid, info) in enumerate(sorted(batters.items())):
        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{len(batters)}", flush=True)
        try:
            raw = get_season_stats(pid, CURRENT_SEASON, "hitting")
            if not raw:
                continue
            rates = raw_to_rates(raw, "plateAppearances", include_sac_bunt=True)
            if not rates:
                continue
            bio = get_bio(pid)
            pid_str = str(pid)

            if pid_str in old_batters.index:
                old_pa = float(old_batters.loc[pid_str].get("pa", 0))
                new_pa = rates["pa"] * _NEW_SEASON_WEIGHT
                total_pa = old_pa + new_pa
                if total_pa > 0:
                    w_old = old_pa / total_pa
                    w_new = new_pa / total_pa
                    for k in ["k_rate", "bb_rate", "hbp_rate", "hr_rate", "single_rate",
                              "double_rate", "triple_rate", "gbout_rate", "fbout_rate",
                              "ldout_rate", "fc_rate", "gidp_prob_given_gbout",
                              "sf_prob_given_fbout", "sac_bunt_prob"]:
                        old_val = float(old_batters.loc[pid_str].get(k, 0))
                        old_batters.loc[pid_str, k] = w_old * old_val + w_new * rates[k]
                    old_batters.loc[pid_str, "pa"] = total_pa
                    old_batters.loc[pid_str, "team"] = info["team"]
                    primary = ["k_rate", "bb_rate", "hbp_rate", "hr_rate", "single_rate",
                               "double_rate", "triple_rate", "gbout_rate", "fbout_rate",
                               "ldout_rate", "fc_rate"]
                    psum = sum(float(old_batters.loc[pid_str, k]) for k in primary)
                    if psum > 0:
                        for k in primary:
                            old_batters.loc[pid_str, k] = float(old_batters.loc[pid_str, k]) / psum
                updated_b += 1
            else:
                row = rates.copy()
                row["name"] = bio.get("fullName", info["name"])
                bat_side = bio.get("batSide", {}).get("code", "R")
                if bat_side == "S":
                    bat_side = "R"
                row["hand"] = bat_side
                row["team"] = info["team"]
                old_batters.loc[pid_str] = row
                new_b += 1
        except Exception:
            pass
        time.sleep(0.05)

    print(f"    Updated {updated_b}, added {new_b} new batters", flush=True)

    # Update league averages
    print("  Updating league averages...", flush=True)
    try:
        url = f"{MLB_API}/teams/stats?stats=season&season={CURRENT_SEASON}&group=pitching&sportId=1&gameType=R"
        data = fetch_json(url)
        totals = {}
        keys = ["strikeOuts", "baseOnBalls", "intentionalWalks", "hitByPitch",
                "homeRuns", "hits", "doubles", "triples", "groundOuts", "airOuts",
                "groundIntoDoublePlay", "sacFlies", "battersFaced"]
        for k in keys:
            totals[k] = 0
        for split in data.get("stats", [{}])[0].get("splits", []):
            for k in keys:
                totals[k] += int(split["stat"].get(k, 0))
        lg = raw_to_rates(totals, "battersFaced")
        if lg:
            lg["sac_bunt_prob"] = 0.003
            league_df = pd.DataFrame([lg])
            league_df[LEAGUE_COLS].to_csv(f"{DATA_DIR}league_averages.csv", index=False, float_format="%.6f")
            print(f"    League averages updated ({CURRENT_SEASON})", flush=True)
    except Exception as e:
        print(f"    WARNING: Could not update league averages: {e}", flush=True)

    # Save
    print("  Writing CSVs...", flush=True)
    old_pitchers.reset_index().rename(columns={"index": "pitcher_id"})[PITCHER_COLS].to_csv(
        f"{DATA_DIR}pitchers.csv", index=False, float_format="%.6f")
    old_batters.reset_index().rename(columns={"index": "batter_id"})[BATTER_COLS].to_csv(
        f"{DATA_DIR}batters.csv", index=False, float_format="%.6f")

    print(f"\nDone. Profiles: {len(old_pitchers)} pitchers, {len(old_batters)} batters", flush=True)


if __name__ == "__main__":
    main()
