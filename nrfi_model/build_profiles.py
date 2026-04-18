#!/usr/bin/env python3
"""
build_profiles.py — Build 3-year blended player profiles for every player
on an active MLB 40-man roster.

Aggregates raw counting stats from 2024, 2025, and 2026 (where available)
then converts to per-PA rates for the NRFI model.

Usage:
    python3 build_profiles.py
"""

from __future__ import annotations
import requests
import pandas as pd
import numpy as np
import time
import sys
import json
from typing import Dict, List, Optional, Tuple

MLB_API = "https://statsapi.mlb.com/api/v1"

SEASONS = [2024, 2025, 2026]

# Recency weights applied to each season's counting stats before aggregation.
# Higher weights for more recent seasons let profiles track current trends.
# Default: linear (2024: 1x, 2025: 2x, 2026: 3x).
RECENCY_WEIGHTS = {2024: 1.0, 2025: 2.0, 2026: 5.0}

# Fraction of air outs that are fly ball outs vs line drive outs
AIR_OUT_FB_FRAC = 0.80
AIR_OUT_LD_FRAC = 0.20

# Fielder's choice estimated as fraction of ground ball plays
FC_FRAC_OF_GBOUT = 0.06


def fetch_json(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1)


def get_all_rosters() -> Dict[int, dict]:
    """Get all players on 40-man rosters across MLB."""
    teams = fetch_json(f"{MLB_API}/teams?sportId=1")["teams"]
    players = {}
    for t in teams:
        tid = t["id"]
        abbr = t["abbreviation"]
        roster = fetch_json(f"{MLB_API}/teams/{tid}/roster?rosterType=40Man")
        for p in roster.get("roster", []):
            pid = p["person"]["id"]
            pos = p["position"]["abbreviation"]
            players[pid] = {
                "id": pid,
                "name": p["person"]["fullName"],
                "pos": pos,
                "team": abbr,
                "is_pitcher": pos == "P",
            }
    return players


def get_player_bio(player_id: int) -> dict:
    data = fetch_json(f"{MLB_API}/people/{player_id}")
    people = data.get("people", [])
    if not people:
        return {"fullName": f"Unknown ({player_id})", "batSide": {"code": "R"}, "pitchHand": {"code": "R"}}
    return people[0]


def get_multi_season_pitching(player_id: int) -> List[dict]:
    """Fetch pitching stats for all available seasons in SEASONS."""
    results = []
    for season in SEASONS:
        url = f"{MLB_API}/people/{player_id}/stats?stats=season&season={season}&group=pitching"
        data = fetch_json(url)
        stat_groups = data.get("stats", [])
        if not stat_groups:
            continue
        splits = stat_groups[0].get("splits", [])
        if splits:
            stat = splits[0]["stat"]
            stat["_season"] = season
            results.append(stat)
    return results


def get_multi_season_hitting(player_id: int) -> List[dict]:
    """Fetch hitting stats for all available seasons in SEASONS."""
    results = []
    for season in SEASONS:
        url = f"{MLB_API}/people/{player_id}/stats?stats=season&season={season}&group=hitting"
        data = fetch_json(url)
        stat_groups = data.get("stats", [])
        if not stat_groups:
            continue
        splits = stat_groups[0].get("splits", [])
        if splits:
            stat = splits[0]["stat"]
            stat["_season"] = season
            results.append(stat)
    return results


def aggregate_counting_stats(season_stats: List[dict], count_keys: List[str]) -> dict:
    """
    Sum counting stats across seasons, scaled by RECENCY_WEIGHTS.
    More recent seasons contribute proportionally more to the blended profile.
    """
    totals = {k: 0.0 for k in count_keys}
    seasons_found = []
    for s in season_stats:
        season = s.get("_season", 0)
        w = RECENCY_WEIGHTS.get(season, 1.0)
        seasons_found.append(season)
        for k in count_keys:
            totals[k] += float(s.get(k, 0)) * w
    totals["_seasons"] = seasons_found
    return totals


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


def counting_to_pitcher_rates(totals: dict) -> Optional[dict]:
    """Convert aggregated counting stats to per-PA rates for a pitcher."""
    pa = totals.get("battersFaced", 0)
    if pa < 1:
        return None

    singles = totals["hits"] - totals["doubles"] - totals["triples"] - totals["homeRuns"]
    singles = max(0, singles)

    ground_outs_raw = totals["groundOuts"]
    air_outs_raw = totals["airOuts"]

    fc_count = round(ground_outs_raw * FC_FRAC_OF_GBOUT)
    gb_outs = ground_outs_raw - fc_count
    fb_outs = round(air_outs_raw * AIR_OUT_FB_FRAC)
    ld_outs = air_outs_raw - fb_outs

    rates = {
        "k_rate":      totals["strikeOuts"] / pa,
        "bb_rate":     (totals["baseOnBalls"] - totals["intentionalWalks"]) / pa,
        "hbp_rate":    totals["hitByPitch"] / pa,
        "hr_rate":     totals["homeRuns"] / pa,
        "single_rate": singles / pa,
        "double_rate": totals["doubles"] / pa,
        "triple_rate": totals["triples"] / pa,
        "gbout_rate":  gb_outs / pa,
        "fbout_rate":  fb_outs / pa,
        "ldout_rate":  ld_outs / pa,
        "fc_rate":     fc_count / pa,
    }

    gidp = totals["groundIntoDoublePlay"]
    rates["gidp_prob_given_gbout"] = min(0.40, gidp / max(1, ground_outs_raw))

    sf = totals["sacFlies"]
    rates["sf_prob_given_fbout"] = min(0.40, sf / max(1, air_outs_raw))

    # Normalize primary outcomes
    primary = [
        "k_rate", "bb_rate", "hbp_rate", "hr_rate",
        "single_rate", "double_rate", "triple_rate",
        "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
    ]
    total = sum(rates[k] for k in primary)
    if total > 0:
        for k in primary:
            rates[k] /= total

    rates["pa"] = pa
    rates["_seasons"] = totals["_seasons"]
    return rates


def counting_to_batter_rates(totals: dict) -> Optional[dict]:
    """Convert aggregated counting stats to per-PA rates for a batter."""
    pa = totals.get("plateAppearances", 0)
    if pa < 1:
        return None

    singles = totals["hits"] - totals["doubles"] - totals["triples"] - totals["homeRuns"]
    singles = max(0, singles)

    ground_outs_raw = totals["groundOuts"]
    air_outs_raw = totals["airOuts"]

    fc_count = round(ground_outs_raw * FC_FRAC_OF_GBOUT)
    gb_outs = ground_outs_raw - fc_count
    fb_outs = round(air_outs_raw * AIR_OUT_FB_FRAC)
    ld_outs = air_outs_raw - fb_outs

    rates = {
        "k_rate":      totals["strikeOuts"] / pa,
        "bb_rate":     (totals["baseOnBalls"] - totals["intentionalWalks"]) / pa,
        "hbp_rate":    totals["hitByPitch"] / pa,
        "hr_rate":     totals["homeRuns"] / pa,
        "single_rate": singles / pa,
        "double_rate": totals["doubles"] / pa,
        "triple_rate": totals["triples"] / pa,
        "gbout_rate":  gb_outs / pa,
        "fbout_rate":  fb_outs / pa,
        "ldout_rate":  ld_outs / pa,
        "fc_rate":     fc_count / pa,
    }

    gidp = totals["groundIntoDoublePlay"]
    rates["gidp_prob_given_gbout"] = min(0.40, gidp / max(1, ground_outs_raw))

    sf = totals["sacFlies"]
    rates["sf_prob_given_fbout"] = min(0.40, sf / max(1, air_outs_raw))

    rates["sac_bunt_prob"] = min(0.20, totals["sacBunts"] / max(1, pa))

    primary = [
        "k_rate", "bb_rate", "hbp_rate", "hr_rate",
        "single_rate", "double_rate", "triple_rate",
        "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
    ]
    total = sum(rates[k] for k in primary)
    if total > 0:
        for k in primary:
            rates[k] /= total

    rates["pa"] = pa
    rates["_seasons"] = totals["_seasons"]
    return rates


def build_league_averages() -> dict:
    """Aggregate league-wide pitching stats across 2024-2026."""
    all_totals = {k: 0 for k in PITCHER_COUNT_KEYS}
    seasons_found = []

    for season in SEASONS:
        try:
            url = f"{MLB_API}/teams/stats?stats=season&season={season}&group=pitching&sportId=1&gameType=R"
            data = fetch_json(url)
            for split in data.get("stats", [{}])[0].get("splits", []):
                stat = split["stat"]
                for k in PITCHER_COUNT_KEYS:
                    all_totals[k] += int(stat.get(k, 0))
            seasons_found.append(season)
        except Exception as e:
            print(f"  Warning: could not fetch league stats for {season}: {e}")

    all_totals["_seasons"] = seasons_found
    rates = counting_to_pitcher_rates(all_totals)
    rates["sac_bunt_prob"] = 0.003
    return rates


def main():
    print("=" * 60)
    print("  Building 3-year (2024-2026) MLB player profiles")
    print("=" * 60)

    # 1. Get all rosters
    print("\n[1/4] Fetching 40-man rosters across all 30 teams...")
    players = get_all_rosters()
    pitchers_list = {pid: p for pid, p in players.items() if p["is_pitcher"]}
    batters_list = {pid: p for pid, p in players.items() if not p["is_pitcher"]}
    print(f"  Found {len(pitchers_list)} pitchers, {len(batters_list)} position players")

    # 2. Build pitcher profiles
    print(f"\n[2/4] Fetching pitcher stats (3 seasons × {len(pitchers_list)} pitchers)...")
    pitcher_rows = []
    skipped_pitchers = 0
    for i, (pid, info) in enumerate(sorted(pitchers_list.items())):
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(pitchers_list)} pitchers processed")
        try:
            bio = get_player_bio(pid)
            season_stats = get_multi_season_pitching(pid)
            if not season_stats:
                skipped_pitchers += 1
                continue

            totals = aggregate_counting_stats(season_stats, PITCHER_COUNT_KEYS)
            rates = counting_to_pitcher_rates(totals)
            if rates is None:
                skipped_pitchers += 1
                continue

            rates["pitcher_id"] = str(pid)
            rates["name"] = bio.get("fullName", info["name"])
            rates["hand"] = bio.get("pitchHand", {}).get("code", "R")
            rates["team"] = info["team"]
            pitcher_rows.append(rates)
        except Exception as e:
            print(f"  ERROR pitcher {pid} ({info['name']}): {e}")
            skipped_pitchers += 1
        time.sleep(0.08)

    print(f"  Built {len(pitcher_rows)} pitcher profiles (skipped {skipped_pitchers})")

    # 3. Build batter profiles
    print(f"\n[3/4] Fetching batter stats (3 seasons × {len(batters_list)} batters)...")
    batter_rows = []
    skipped_batters = 0
    for i, (pid, info) in enumerate(sorted(batters_list.items())):
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(batters_list)} batters processed")
        try:
            bio = get_player_bio(pid)
            season_stats = get_multi_season_hitting(pid)
            if not season_stats:
                skipped_batters += 1
                continue

            totals = aggregate_counting_stats(season_stats, BATTER_COUNT_KEYS)
            rates = counting_to_batter_rates(totals)
            if rates is None:
                skipped_batters += 1
                continue

            rates["batter_id"] = str(pid)
            rates["name"] = bio.get("fullName", info["name"])
            bat_side = bio.get("batSide", {}).get("code", "R")
            if bat_side == "S":
                bat_side = "R"  # switch hitters default to R
            rates["hand"] = bat_side
            rates["team"] = info["team"]
            batter_rows.append(rates)
        except Exception as e:
            print(f"  ERROR batter {pid} ({info['name']}): {e}")
            skipped_batters += 1
        time.sleep(0.08)

    print(f"  Built {len(batter_rows)} batter profiles (skipped {skipped_batters})")

    # Also fetch hitting stats for pitchers who bat (NL, interleague)
    print("\n  Checking pitchers for batting stats...")
    pitcher_batting_count = 0
    for i, (pid, info) in enumerate(sorted(pitchers_list.items())):
        # Check if this pitcher already has a batter profile
        if any(r["batter_id"] == str(pid) for r in batter_rows):
            continue
        try:
            season_stats = get_multi_season_hitting(pid)
            if not season_stats:
                continue
            totals = aggregate_counting_stats(season_stats, BATTER_COUNT_KEYS)
            if totals.get("plateAppearances", 0) < 1:
                continue
            rates = counting_to_batter_rates(totals)
            if rates is None:
                continue
            bio = get_player_bio(pid)
            rates["batter_id"] = str(pid)
            rates["name"] = bio.get("fullName", info["name"])
            bat_side = bio.get("batSide", {}).get("code", "R")
            if bat_side == "S":
                bat_side = "R"
            rates["hand"] = bat_side
            rates["team"] = info["team"]
            batter_rows.append(rates)
            pitcher_batting_count += 1
        except Exception:
            pass
        time.sleep(0.08)
    if pitcher_batting_count:
        print(f"  Added {pitcher_batting_count} pitcher batting profiles")

    # 4. League averages
    print("\n[4/4] Computing league averages (2024-2026)...")
    league_avg = build_league_averages()

    # Write CSVs
    print("\nWriting CSVs...")

    pitcher_df = pd.DataFrame(pitcher_rows)
    pitcher_cols = [
        "pitcher_id", "name", "team", "hand", "pa",
        "k_rate", "bb_rate", "hbp_rate", "hr_rate",
        "single_rate", "double_rate", "triple_rate",
        "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
        "gidp_prob_given_gbout", "sf_prob_given_fbout",
    ]
    pitcher_df[pitcher_cols].to_csv("data/pitchers.csv", index=False, float_format="%.6f")
    print(f"  data/pitchers.csv: {len(pitcher_df)} pitchers")

    batter_df = pd.DataFrame(batter_rows)
    batter_cols = [
        "batter_id", "name", "team", "hand", "pa",
        "k_rate", "bb_rate", "hbp_rate", "hr_rate",
        "single_rate", "double_rate", "triple_rate",
        "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
        "gidp_prob_given_gbout", "sf_prob_given_fbout", "sac_bunt_prob",
    ]
    batter_df[batter_cols].to_csv("data/batters.csv", index=False, float_format="%.6f")
    print(f"  data/batters.csv: {len(batter_df)} batters")

    league_df = pd.DataFrame([league_avg])
    league_cols = [
        "k_rate", "bb_rate", "hbp_rate", "hr_rate",
        "single_rate", "double_rate", "triple_rate",
        "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
        "gidp_prob_given_gbout", "sf_prob_given_fbout", "sac_bunt_prob",
    ]
    league_df[league_cols].to_csv("data/league_averages.csv", index=False, float_format="%.6f")
    print(f"  data/league_averages.csv: 1 row (seasons: {league_avg.get('_seasons')})")

    # Summary stats
    print(f"\n{'='*60}")
    print(f"  Profile Build Complete")
    print(f"{'='*60}")
    print(f"  Pitchers: {len(pitcher_df)} profiles")
    print(f"  Batters:  {len(batter_df)} profiles")
    print(f"  Seasons:  {SEASONS}")
    p_pa = pitcher_df['pa']
    b_pa = batter_df['pa']
    print(f"  Pitcher PA range: {int(p_pa.min())}–{int(p_pa.max())} (median {int(p_pa.median())})")
    print(f"  Batter PA range:  {int(b_pa.min())}–{int(b_pa.max())} (median {int(b_pa.median())})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
