#!/usr/bin/env python3
"""
fetch_splits.py — Pull 1st-inning splits (i01) and platoon splits (vr, vl)
for every active MLB player and aggregate across 2024-2026 with the same
recency weighting as the main profile builder.

Outputs:
    data/pitcher_splits.csv  — one row per pitcher with i01_*, vr_*, vl_* rates
    data/batter_splits.csv   — one row per batter with vr_*, vl_* rates

These files are consumed by build_blended_rates() to replace the hardcoded
platoon multipliers and to optionally blend in 1st-inning-specific pitcher
behaviour.
"""

from __future__ import annotations
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

MLB_API = "https://statsapi.mlb.com/api/v1"
DATA_DIR = Path("data")
SEASONS = [2024, 2025, 2026]
RECENCY_WEIGHTS = {2024: 1.0, 2025: 2.0, 2026: 5.0}  # match build_profiles

AIR_OUT_FB_FRAC = 0.80
FC_FRAC_OF_GBOUT = 0.06

PITCHER_KEYS = [
    "strikeOuts", "baseOnBalls", "intentionalWalks", "hitByPitch",
    "homeRuns", "hits", "doubles", "triples",
    "groundOuts", "airOuts", "groundIntoDoublePlay", "sacFlies",
    "battersFaced",
]
BATTER_KEYS = [
    "strikeOuts", "baseOnBalls", "intentionalWalks", "hitByPitch",
    "homeRuns", "hits", "doubles", "triples",
    "groundOuts", "airOuts", "groundIntoDoublePlay", "sacFlies",
    "sacBunts", "plateAppearances",
]


def fetch_json(url, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(0.5)


def get_split_stat(player_id, season, group, code):
    """Return raw stat dict for one split, or None if no data."""
    url = f"{MLB_API}/people/{player_id}/stats?stats=statSplits&sitCodes={code}&season={season}&group={group}"
    data = fetch_json(url)
    sg = data.get("stats", [])
    if not sg:
        return None
    splits = sg[0].get("splits", [])
    if not splits:
        return None
    return splits[0].get("stat", {})


def aggregate_weighted(per_season_stats: List[dict], keys: List[str]) -> dict:
    """Sum counting stats across seasons with recency weights."""
    totals = {k: 0.0 for k in keys}
    for s in per_season_stats:
        season = s.get("_season")
        w = RECENCY_WEIGHTS.get(season, 1.0)
        for k in keys:
            totals[k] += float(s.get(k, 0)) * w
    return totals


def to_pitcher_rates(totals: dict, prefix: str = "") -> Optional[dict]:
    pa = totals.get("battersFaced", 0)
    if pa < 5:
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
        f"{prefix}k_rate":      totals["strikeOuts"] / pa,
        f"{prefix}bb_rate":     (totals["baseOnBalls"] - totals["intentionalWalks"]) / pa,
        f"{prefix}hbp_rate":    totals["hitByPitch"] / pa,
        f"{prefix}hr_rate":     totals["homeRuns"] / pa,
        f"{prefix}single_rate": singles / pa,
        f"{prefix}double_rate": totals["doubles"] / pa,
        f"{prefix}triple_rate": totals["triples"] / pa,
        f"{prefix}gbout_rate":  gb / pa,
        f"{prefix}fbout_rate":  fb / pa,
        f"{prefix}ldout_rate":  ld / pa,
        f"{prefix}fc_rate":     fc / pa,
    }
    primary = [k for k in rates if k.endswith("_rate")]
    total = sum(rates[k] for k in primary)
    if total > 0:
        for k in primary:
            rates[k] /= total
    rates[f"{prefix}pa"] = pa
    return rates


def to_batter_rates(totals: dict, prefix: str = "") -> Optional[dict]:
    pa = totals.get("plateAppearances", 0)
    if pa < 5:
        return None
    return _to_rates(totals, pa, prefix)


def _to_rates(totals, pa, prefix):
    singles = totals["hits"] - totals["doubles"] - totals["triples"] - totals["homeRuns"]
    singles = max(0, singles)
    go = totals["groundOuts"]
    ao = totals["airOuts"]
    fc = round(go * FC_FRAC_OF_GBOUT)
    gb = go - fc
    fb = round(ao * AIR_OUT_FB_FRAC)
    ld = ao - fb
    rates = {
        f"{prefix}k_rate":      totals["strikeOuts"] / pa,
        f"{prefix}bb_rate":     (totals["baseOnBalls"] - totals["intentionalWalks"]) / pa,
        f"{prefix}hbp_rate":    totals["hitByPitch"] / pa,
        f"{prefix}hr_rate":     totals["homeRuns"] / pa,
        f"{prefix}single_rate": singles / pa,
        f"{prefix}double_rate": totals["doubles"] / pa,
        f"{prefix}triple_rate": totals["triples"] / pa,
        f"{prefix}gbout_rate":  gb / pa,
        f"{prefix}fbout_rate":  fb / pa,
        f"{prefix}ldout_rate":  ld / pa,
        f"{prefix}fc_rate":     fc / pa,
    }
    primary = [k for k in rates if k.endswith("_rate")]
    total = sum(rates[k] for k in primary)
    if total > 0:
        for k in primary:
            rates[k] /= total
    rates[f"{prefix}pa"] = pa
    return rates


def fetch_player_split(player_id, group, code, keys):
    """Fetch a split across all SEASONS, return weighted rate dict or None."""
    per_season = []
    for season in SEASONS:
        s = get_split_stat(player_id, season, group, code)
        if s:
            s["_season"] = season
            per_season.append(s)
        time.sleep(0.04)
    if not per_season:
        return None, []
    totals = aggregate_weighted(per_season, keys)
    return totals, [s["_season"] for s in per_season]


def main():
    pitchers_csv = pd.read_csv(DATA_DIR / "pitchers.csv", dtype={"pitcher_id": str})
    batters_csv = pd.read_csv(DATA_DIR / "batters.csv", dtype={"batter_id": str})

    print(f"Fetching splits for {len(pitchers_csv)} pitchers and {len(batters_csv)} batters", flush=True)
    print(f"  3 splits × 3 seasons per pitcher, 2 splits × 3 seasons per batter", flush=True)
    print(f"  Estimated runtime: ~13-15 minutes", flush=True)

    # ---- Pitchers ----
    pitcher_rows = []
    for i, row in pitchers_csv.iterrows():
        pid = row["pitcher_id"]
        if (i + 1) % 50 == 0:
            print(f"  pitcher {i+1}/{len(pitchers_csv)}", flush=True)
        out = {"pitcher_id": pid, "name": row.get("name", "")}

        # 1st inning split
        totals_i1, _ = fetch_player_split(pid, "pitching", "i01", PITCHER_KEYS)
        if totals_i1:
            r = to_pitcher_rates(totals_i1, prefix="i1_")
            if r: out.update(r)

        # vs Right
        totals_vr, _ = fetch_player_split(pid, "pitching", "vr", PITCHER_KEYS)
        if totals_vr:
            r = to_pitcher_rates(totals_vr, prefix="vr_")
            if r: out.update(r)

        # vs Left
        totals_vl, _ = fetch_player_split(pid, "pitching", "vl", PITCHER_KEYS)
        if totals_vl:
            r = to_pitcher_rates(totals_vl, prefix="vl_")
            if r: out.update(r)

        if len(out) > 2:  # got at least one split
            pitcher_rows.append(out)

    pdf = pd.DataFrame(pitcher_rows)
    pdf.to_csv(DATA_DIR / "pitcher_splits.csv", index=False, float_format="%.6f")
    print(f"\n  Wrote {len(pdf)} pitcher splits → data/pitcher_splits.csv", flush=True)

    # ---- Batters ----
    batter_rows = []
    for i, row in batters_csv.iterrows():
        bid = row["batter_id"]
        if (i + 1) % 50 == 0:
            print(f"  batter {i+1}/{len(batters_csv)}", flush=True)
        out = {"batter_id": bid, "name": row.get("name", "")}

        # vs Right
        totals_vr, _ = fetch_player_split(bid, "hitting", "vr", BATTER_KEYS)
        if totals_vr:
            r = to_batter_rates(totals_vr, prefix="vr_")
            if r: out.update(r)

        # vs Left
        totals_vl, _ = fetch_player_split(bid, "hitting", "vl", BATTER_KEYS)
        if totals_vl:
            r = to_batter_rates(totals_vl, prefix="vl_")
            if r: out.update(r)

        if len(out) > 2:
            batter_rows.append(out)

    bdf = pd.DataFrame(batter_rows)
    bdf.to_csv(DATA_DIR / "batter_splits.csv", index=False, float_format="%.6f")
    print(f"  Wrote {len(bdf)} batter splits → data/batter_splits.csv", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
