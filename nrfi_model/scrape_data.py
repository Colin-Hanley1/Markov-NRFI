#!/usr/bin/env python3
"""
scrape_data.py — Pull real MLB data from the public MLB Stats API
and write the CSV files needed by the NRFI model.

Usage:
    python3 scrape_data.py                  # today's games, 2025 stats
    python3 scrape_data.py 2025-04-10       # specific date
    python3 scrape_data.py 2025-04-10 2024  # specific date, stats from 2024 season
"""

from __future__ import annotations
import requests
import pandas as pd
import numpy as np
import sys
import time
from datetime import date
from typing import Dict, List, Optional, Tuple

MLB_API = "https://statsapi.mlb.com/api/v1"

# Team abbreviation mapping from MLB API team IDs
TEAM_ABBREVS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

# Park factors (HR and run) — sourced from FanGraphs 3-year regressed averages
# These are static; update annually
PARK_FACTORS = {
    "LAA": (0.96, 0.99), "ARI": (1.08, 1.04), "BAL": (1.04, 1.01),
    "BOS": (1.10, 1.03), "CHC": (1.05, 1.02), "CIN": (1.12, 1.05),
    "CLE": (0.93, 0.97), "COL": (1.25, 1.12), "DET": (0.95, 0.98),
    "HOU": (1.00, 1.00), "KC":  (0.92, 0.98), "LAD": (0.95, 0.98),
    "WSH": (1.00, 1.00), "NYM": (0.92, 0.97), "OAK": (0.90, 0.96),
    "PIT": (0.88, 0.96), "SD":  (0.93, 0.97), "SEA": (0.93, 0.97),
    "SF":  (0.85, 0.95), "STL": (0.97, 0.99), "TB":  (0.88, 0.96),
    "TEX": (1.05, 1.02), "TOR": (1.02, 1.01), "MIN": (1.08, 1.03),
    "PHI": (1.03, 1.01), "ATL": (1.05, 1.02), "CWS": (1.06, 1.02),
    "MIA": (0.82, 0.94), "NYY": (1.15, 1.05), "MIL": (0.98, 0.99),
}

# League-average out-type split: of all air outs, ~70% are fly balls, ~20% are
# line-drive outs, ~10% are popups. We merge popup into fbout.
AIR_OUT_FB_FRAC = 0.80   # fraction of air outs classified as fbout
AIR_OUT_LD_FRAC = 0.20   # fraction of air outs classified as ldout

# Fielder's choice is not in the MLB API counting stats — estimate as fraction of groundOuts
FC_FRAC_OF_GBOUT = 0.06  # ~6% of ground ball plays result in FC


def fetch_json(url: str) -> dict:
    """Fetch JSON from MLB Stats API with basic error handling."""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def get_schedule(game_date: str) -> List[dict]:
    """Get today's schedule with lineups and probable pitchers."""
    url = (
        f"{MLB_API}/schedule?sportId=1&date={game_date}"
        f"&hydrate=lineups,probablePitcher"
    )
    data = fetch_json(url)
    if not data.get("dates"):
        return []
    return data["dates"][0].get("games", [])


def get_player_bio(player_id: int) -> dict:
    """Get player bio info (handedness, name, etc.)."""
    data = fetch_json(f"{MLB_API}/people/{player_id}")
    people = data.get("people", [])
    if not people:
        return {"fullName": f"Unknown ({player_id})", "batSide": {"code": "R"}, "pitchHand": {"code": "R"}}
    return people[0]


def get_pitcher_stats(player_id: int, season: int) -> Optional[dict]:
    """Get pitcher season stats from MLB API."""
    url = f"{MLB_API}/people/{player_id}/stats?stats=season&season={season}&group=pitching"
    data = fetch_json(url)
    stat_groups = data.get("stats", [])
    if not stat_groups:
        return None
    splits = stat_groups[0].get("splits", [])
    if not splits:
        return None
    return splits[0]["stat"]


def get_batter_stats(player_id: int, season: int) -> Optional[dict]:
    """Get batter season stats from MLB API."""
    url = f"{MLB_API}/people/{player_id}/stats?stats=season&season={season}&group=hitting"
    data = fetch_json(url)
    stat_groups = data.get("stats", [])
    if not stat_groups:
        return None
    splits = stat_groups[0].get("splits", [])
    if not splits:
        return None
    return splits[0]["stat"]


def pitcher_stats_to_rates(raw: dict, pa: int) -> Dict[str, float]:
    """
    Convert MLB API pitcher counting stats to per-PA rates.

    The MLB API provides: strikeOuts, baseOnBalls, hitByPitch, homeRuns,
    hits (total), doubles, triples, groundOuts, airOuts, groundIntoDoublePlay,
    sacFlies, battersFaced.
    """
    if pa <= 0:
        return None

    singles = raw.get("hits", 0) - raw.get("doubles", 0) - raw.get("triples", 0) - raw.get("homeRuns", 0)
    singles = max(0, singles)

    ground_outs_raw = raw.get("groundOuts", 0)
    air_outs_raw = raw.get("airOuts", 0)

    # Split ground outs into gbout and FC
    fc_count = round(ground_outs_raw * FC_FRAC_OF_GBOUT)
    gb_outs = ground_outs_raw - fc_count

    # Split air outs into fbout and ldout
    fb_outs = round(air_outs_raw * AIR_OUT_FB_FRAC)
    ld_outs = air_outs_raw - fb_outs

    rates = {
        "k_rate":      raw.get("strikeOuts", 0) / pa,
        "bb_rate":     (raw.get("baseOnBalls", 0) - raw.get("intentionalWalks", 0)) / pa,
        "hbp_rate":    raw.get("hitByPitch", 0) / pa,
        "hr_rate":     raw.get("homeRuns", 0) / pa,
        "single_rate": singles / pa,
        "double_rate": raw.get("doubles", 0) / pa,
        "triple_rate": raw.get("triples", 0) / pa,
        "gbout_rate":  gb_outs / pa,
        "fbout_rate":  fb_outs / pa,
        "ldout_rate":  ld_outs / pa,
        "fc_rate":     fc_count / pa,
    }

    # GIDP rate given GB out opportunity (runner on 1st, < 2 outs)
    # Approximate: GIDP / groundOuts is a decent proxy
    gidp = raw.get("groundIntoDoublePlay", 0)
    rates["gidp_prob_given_gbout"] = min(0.40, gidp / max(1, ground_outs_raw))

    # Sac fly rate given FB out with runner on 3rd
    sf = raw.get("sacFlies", 0)
    rates["sf_prob_given_fbout"] = min(0.40, sf / max(1, air_outs_raw))

    # Normalize primary outcomes to sum to 1.0
    primary_keys = [
        "k_rate", "bb_rate", "hbp_rate", "hr_rate",
        "single_rate", "double_rate", "triple_rate",
        "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
    ]
    total = sum(rates[k] for k in primary_keys)
    if total > 0:
        for k in primary_keys:
            rates[k] /= total

    return rates


def batter_stats_to_rates(raw: dict, pa: int) -> Dict[str, float]:
    """Convert MLB API batter counting stats to per-PA rates."""
    if pa <= 0:
        return None

    singles = raw.get("hits", 0) - raw.get("doubles", 0) - raw.get("triples", 0) - raw.get("homeRuns", 0)
    singles = max(0, singles)

    ground_outs_raw = raw.get("groundOuts", 0)
    air_outs_raw = raw.get("airOuts", 0)

    fc_count = round(ground_outs_raw * FC_FRAC_OF_GBOUT)
    gb_outs = ground_outs_raw - fc_count

    fb_outs = round(air_outs_raw * AIR_OUT_FB_FRAC)
    ld_outs = air_outs_raw - fb_outs

    rates = {
        "k_rate":      raw.get("strikeOuts", 0) / pa,
        "bb_rate":     (raw.get("baseOnBalls", 0) - raw.get("intentionalWalks", 0)) / pa,
        "hbp_rate":    raw.get("hitByPitch", 0) / pa,
        "hr_rate":     raw.get("homeRuns", 0) / pa,
        "single_rate": singles / pa,
        "double_rate": raw.get("doubles", 0) / pa,
        "triple_rate": raw.get("triples", 0) / pa,
        "gbout_rate":  gb_outs / pa,
        "fbout_rate":  fb_outs / pa,
        "ldout_rate":  ld_outs / pa,
        "fc_rate":     fc_count / pa,
    }

    gidp = raw.get("groundIntoDoublePlay", 0)
    rates["gidp_prob_given_gbout"] = min(0.40, gidp / max(1, ground_outs_raw))

    sf = raw.get("sacFlies", 0)
    rates["sf_prob_given_fbout"] = min(0.40, sf / max(1, air_outs_raw))

    rates["sac_bunt_prob"] = min(0.20, raw.get("sacBunts", 0) / max(1, pa))

    primary_keys = [
        "k_rate", "bb_rate", "hbp_rate", "hr_rate",
        "single_rate", "double_rate", "triple_rate",
        "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
    ]
    total = sum(rates[k] for k in primary_keys)
    if total > 0:
        for k in primary_keys:
            rates[k] /= total

    return rates


def build_league_averages(season: int) -> Dict[str, float]:
    """
    Compute league-average rates from MLB-wide pitching stats.
    Uses the MLB API league-level stats endpoint.
    """
    url = f"{MLB_API}/teams/stats?stats=season&season={season}&group=pitching&sportId=1&gameType=R"
    data = fetch_json(url)

    # Aggregate across all teams
    totals = {}
    count_keys = [
        "strikeOuts", "baseOnBalls", "intentionalWalks", "hitByPitch",
        "homeRuns", "hits", "doubles", "triples",
        "groundOuts", "airOuts", "groundIntoDoublePlay", "sacFlies",
        "battersFaced",
    ]
    for k in count_keys:
        totals[k] = 0

    for split in data.get("stats", [{}])[0].get("splits", []):
        stat = split["stat"]
        for k in count_keys:
            totals[k] += int(stat.get(k, 0))

    pa = totals["battersFaced"]
    if pa <= 0:
        raise RuntimeError("No league stats found")

    return pitcher_stats_to_rates(totals, pa)


def scrape_all(game_date: str, season: int):
    """Main scraping pipeline."""
    print(f"Scraping data for {game_date} (stats from {season} season)...")

    # 1. Get schedule and lineups
    games = get_schedule(game_date)
    if not games:
        print(f"No games found for {game_date}")
        return

    print(f"Found {len(games)} games")

    # Collect all unique player IDs we need
    all_pitcher_ids = set()
    all_batter_ids = set()
    game_infos = []

    for game in games:
        gid = game["gamePk"]
        away_team_id = game["teams"]["away"]["team"]["id"]
        home_team_id = game["teams"]["home"]["team"]["id"]
        away_abbr = TEAM_ABBREVS.get(away_team_id, f"T{away_team_id}")
        home_abbr = TEAM_ABBREVS.get(home_team_id, f"T{home_team_id}")

        away_pitcher = game["teams"]["away"].get("probablePitcher", {})
        home_pitcher = game["teams"]["home"].get("probablePitcher", {})

        if not away_pitcher.get("id") or not home_pitcher.get("id"):
            print(f"  Skipping {gid} ({away_abbr}@{home_abbr}): no probable pitchers")
            continue

        lineups = game.get("lineups", {})
        away_lineup = lineups.get("awayPlayers", [])
        home_lineup = lineups.get("homePlayers", [])

        if len(away_lineup) < 9 or len(home_lineup) < 9:
            print(f"  Skipping {gid} ({away_abbr}@{home_abbr}): incomplete lineups "
                  f"(away={len(away_lineup)}, home={len(home_lineup)})")
            continue

        ap_id = away_pitcher["id"]
        hp_id = home_pitcher["id"]
        all_pitcher_ids.add(ap_id)
        all_pitcher_ids.add(hp_id)

        away_batter_ids = [p["id"] for p in away_lineup[:9]]
        home_batter_ids = [p["id"] for p in home_lineup[:9]]
        all_batter_ids.update(away_batter_ids)
        all_batter_ids.update(home_batter_ids)

        game_infos.append({
            "game_id": str(gid),
            "away_team": away_abbr,
            "home_team": home_abbr,
            "away_pitcher_id": ap_id,
            "home_pitcher_id": hp_id,
            "away_batters": away_batter_ids,
            "home_batters": home_batter_ids,
            "away_pitcher_name": away_pitcher.get("fullName", ""),
            "home_pitcher_name": home_pitcher.get("fullName", ""),
        })

    if not game_infos:
        print("No games with complete lineups and probable pitchers found.")
        return

    print(f"\nProcessing {len(game_infos)} games with lineups")
    print(f"  Pitchers to fetch: {len(all_pitcher_ids)}")
    print(f"  Batters to fetch:  {len(all_batter_ids)}")

    # 2. Fetch pitcher stats and bios
    print("\nFetching pitcher stats...")
    pitcher_rows = []
    for pid in sorted(all_pitcher_ids):
        try:
            bio = get_player_bio(pid)
            raw = get_pitcher_stats(pid, season)
            if raw is None:
                # Try previous season
                raw = get_pitcher_stats(pid, season - 1)
                if raw is None:
                    print(f"  WARNING: No stats for pitcher {pid} ({bio.get('fullName','?')}), using league avg")
                    continue
            pa = int(raw.get("battersFaced", 0))
            if pa < 20:
                print(f"  WARNING: Pitcher {bio['fullName']} has only {pa} PA, stats may be noisy")
            rates = pitcher_stats_to_rates(raw, pa)
            if rates is None:
                continue
            rates["pitcher_id"] = str(pid)
            rates["name"] = bio.get("fullName", "")
            rates["hand"] = bio.get("pitchHand", {}).get("code", "R")
            pitcher_rows.append(rates)
        except Exception as e:
            print(f"  ERROR fetching pitcher {pid}: {e}")
        time.sleep(0.15)  # rate limiting

    # 3. Fetch batter stats and bios
    print("Fetching batter stats...")
    batter_rows = []
    for bid in sorted(all_batter_ids):
        try:
            bio = get_player_bio(bid)
            raw = get_batter_stats(bid, season)
            if raw is None:
                raw = get_batter_stats(bid, season - 1)
                if raw is None:
                    print(f"  WARNING: No stats for batter {bid} ({bio.get('fullName','?')}), using league avg")
                    continue
            pa = int(raw.get("plateAppearances", 0))
            if pa < 20:
                print(f"  WARNING: Batter {bio['fullName']} has only {pa} PA, stats may be noisy")
            rates = batter_stats_to_rates(raw, pa)
            if rates is None:
                continue
            rates["batter_id"] = str(bid)
            rates["name"] = bio.get("fullName", "")
            bat_side = bio.get("batSide", {}).get("code", "R")
            # Handle switch hitters: default to R (will be refined per-matchup in v2)
            if bat_side == "S":
                bat_side = "R"
            rates["hand"] = bat_side
            batter_rows.append(rates)
        except Exception as e:
            print(f"  ERROR fetching batter {bid}: {e}")
        time.sleep(0.15)

    # 4. Fetch league averages
    print("Fetching league averages...")
    league_avg = build_league_averages(season)
    league_avg["sac_bunt_prob"] = 0.005  # small league-wide default

    # 5. Build lineup CSV
    lineup_rows = []
    for gi in game_infos:
        # Away team bats against home pitcher
        for slot, bid in enumerate(gi["away_batters"], 1):
            lineup_rows.append({
                "game_id": gi["game_id"],
                "team": gi["away_team"],
                "side": "away",
                "slot": slot,
                "batter_id": str(bid),
                "pitcher_id": str(gi["home_pitcher_id"]),
            })
        # Home team bats against away pitcher
        for slot, bid in enumerate(gi["home_batters"], 1):
            lineup_rows.append({
                "game_id": gi["game_id"],
                "team": gi["home_team"],
                "side": "home",
                "slot": slot,
                "batter_id": str(bid),
                "pitcher_id": str(gi["away_pitcher_id"]),
            })

    # 6. Build park factors CSV
    park_rows = [
        {"team": team, "hr_factor": pf[0], "run_factor": pf[1]}
        for team, pf in PARK_FACTORS.items()
    ]

    # 7. Write all CSVs
    print("\nWriting CSVs...")

    pitcher_df = pd.DataFrame(pitcher_rows)
    pitcher_cols = [
        "pitcher_id", "name", "hand",
        "k_rate", "bb_rate", "hbp_rate", "hr_rate",
        "single_rate", "double_rate", "triple_rate",
        "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
        "gidp_prob_given_gbout", "sf_prob_given_fbout",
    ]
    pitcher_df[pitcher_cols].to_csv("data/pitchers.csv", index=False, float_format="%.6f")
    print(f"  data/pitchers.csv: {len(pitcher_df)} pitchers")

    batter_df = pd.DataFrame(batter_rows)
    batter_cols = [
        "batter_id", "name", "hand",
        "k_rate", "bb_rate", "hbp_rate", "hr_rate",
        "single_rate", "double_rate", "triple_rate",
        "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
        "gidp_prob_given_gbout", "sf_prob_given_fbout", "sac_bunt_prob",
    ]
    batter_df[batter_cols].to_csv("data/batters.csv", index=False, float_format="%.6f")
    print(f"  data/batters.csv: {len(batter_df)} batters")

    lineup_df = pd.DataFrame(lineup_rows)
    lineup_df.to_csv("data/lineups.csv", index=False)
    print(f"  data/lineups.csv: {len(lineup_df)} rows ({len(game_infos)} games)")

    park_df = pd.DataFrame(park_rows)
    park_df.to_csv("data/park_factors.csv", index=False)
    print(f"  data/park_factors.csv: {len(park_df)} parks")

    league_df = pd.DataFrame([league_avg])
    league_cols = [
        "k_rate", "bb_rate", "hbp_rate", "hr_rate",
        "single_rate", "double_rate", "triple_rate",
        "gbout_rate", "fbout_rate", "ldout_rate", "fc_rate",
        "gidp_prob_given_gbout", "sf_prob_given_fbout", "sac_bunt_prob",
    ]
    league_df[league_cols].to_csv("data/league_averages.csv", index=False, float_format="%.6f")
    print(f"  data/league_averages.csv: 1 row")

    # 8. Print summary of games ready to model
    print(f"\n{'='*60}")
    print(f"  Games ready for modeling ({game_date})")
    print(f"{'='*60}")
    for gi in game_infos:
        print(f"  {gi['game_id']}: {gi['away_team']} @ {gi['home_team']}")
        print(f"    Away SP: {gi['away_pitcher_name']} ({gi['away_pitcher_id']})")
        print(f"    Home SP: {gi['home_pitcher_name']} ({gi['home_pitcher_id']})")
    print(f"{'='*60}")
    print(f"\nRun the model:")
    for gi in game_infos:
        print(f"  python3 pipeline.py {gi['game_id']} {gi['home_team']} {gi['away_team']}")


if __name__ == "__main__":
    game_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    season = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    scrape_all(game_date, season)
