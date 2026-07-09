#!/usr/bin/env python3
"""
compute_daily.py — Pre-compute NRFI predictions for today's games.

Outputs docs/data/games_YYYY-MM-DD.json and docs/data/latest.json
for the static GitHub Pages frontend to consume.

Usage:
    python3 compute_daily.py                # today
    python3 compute_daily.py 2026-04-12     # specific date
"""

from __future__ import annotations
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from datetime import datetime, timezone
from pathlib import Path

from model.outcomes import PAOutcomeRates
from model.blend import build_blended_rates
from model.chain import (
    simulate_nrfi, compute_nrfi_analytic, compute_half_inning_detail,
    simulate_with_details, simulate_full_inning_traced,
)

DATA_DIR = "data/"
OUT_DIR = Path("docs/data")
CALIBRATION_PATH = Path(DATA_DIR) / "calibration_params.json"

MLB_API = "https://statsapi.mlb.com/api/v1"
N_SIMS = 50_000
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES = "KXMLBRFI"
KALSHI_TAKER_FEE_RATE = 0.07

# Kalshi's occurrence_datetime runs a consistent ~3h later than MLB's own
# gameDate (verified live, e.g. 23:10Z vs 02:10Z next day for the same game).
# Title matching (both team hints) is the primary safety check now, so this
# window just needs to comfortably cover that offset, not pin exact time.
KALSHI_MATCH_TOLERANCE_MIN = 240


def kalshi_taker_fee(price: float) -> float:
    return KALSHI_TAKER_FEE_RATE * price * (1 - price)

TEAM_ABBREVS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}
TEAM_IDS = {v: k for k, v in TEAM_ABBREVS.items()}

TEAM_NAMES = {
    "LAA": "Angels", "ARI": "D-backs", "BAL": "Orioles", "BOS": "Red Sox",
    "CHC": "Cubs", "CIN": "Reds", "CLE": "Guardians", "COL": "Rockies",
    "DET": "Tigers", "HOU": "Astros", "KC": "Royals", "LAD": "Dodgers",
    "WSH": "Nationals", "NYM": "Mets", "OAK": "Athletics", "PIT": "Pirates",
    "SD": "Padres", "SEA": "Mariners", "SF": "Giants", "STL": "Cardinals",
    "TB": "Rays", "TEX": "Rangers", "TOR": "Blue Jays", "MIN": "Twins",
    "PHI": "Phillies", "ATL": "Braves", "CWS": "White Sox", "MIA": "Marlins",
    "NYY": "Yankees", "MIL": "Brewers",
}

# Substring each team's abbreviation maps to in Kalshi's market titles
# (e.g. "Toronto vs San Diego First Inning Run?"). Kalshi disambiguates
# same-city teams with a trailing initial ("Los Angeles D" = Dodgers,
# "Chicago C" = Cubs), confirmed against live API samples where noted.
KALSHI_CITY_HINTS = {
    "LAA": "Los Angeles A", "ARI": "Arizona", "BAL": "Baltimore", "BOS": "Boston",  # BOS, ARI confirmed live
    "CHC": "Chicago C", "CIN": "Cincinnati", "CLE": "Cleveland", "COL": "Colorado",  # CIN, COL confirmed live
    "DET": "Detroit", "HOU": "Houston", "KC": "Kansas City", "LAD": "Los Angeles D",  # LAD confirmed live
    "WSH": "Washington", "NYM": "New York M", "OAK": "A's", "PIT": "Pittsburgh",  # NYM, OAK confirmed live
    "SD": "San Diego", "SEA": "Seattle", "SF": "San Francisco", "STL": "St. Louis",  # SD, SF confirmed live
    "TB": "Tampa Bay", "TEX": "Texas", "TOR": "Toronto", "MIN": "Minnesota",  # TOR confirmed live
    "PHI": "Philadelphia", "ATL": "Atlanta", "CWS": "Chicago W", "MIA": "Miami",
    "NYY": "New York Y", "MIL": "Milwaukee",  # MIL confirmed live
}

RATE_COLUMNS = [
    'k_rate', 'bb_rate', 'hbp_rate', 'hr_rate',
    'single_rate', 'double_rate', 'triple_rate',
    'gbout_rate', 'fbout_rate', 'ldout_rate', 'fc_rate',
    'gidp_prob_given_gbout', 'sf_prob_given_fbout', 'sac_bunt_prob',
]


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def load_calibration_params() -> dict:
    try:
        with open(CALIBRATION_PATH) as f:
            params = json.load(f)
        return {"a": float(params.get("a", 1.0)), "b": float(params.get("b", 0.0))}
    except Exception as e:
        print(f"  Warning: using identity calibration ({e})", flush=True)
        return {"a": 1.0, "b": 0.0}


def calibrate_probability(p: float, calibration: dict) -> float:
    return _sigmoid(float(calibration["a"]) * _logit(p) + float(calibration["b"]))


def load_data():
    pitchers = pd.read_csv(f"{DATA_DIR}pitchers.csv", dtype={"pitcher_id": str}).set_index("pitcher_id")
    batters = pd.read_csv(f"{DATA_DIR}batters.csv", dtype={"batter_id": str}).set_index("batter_id")
    parks = pd.read_csv(f"{DATA_DIR}park_factors.csv").set_index("team")
    league = pd.read_csv(f"{DATA_DIR}league_averages.csv").iloc[0]
    # Optional splits files (1st-inning + L/R platoon)
    pitcher_splits = None
    batter_splits = None
    try:
        pitcher_splits = pd.read_csv(f"{DATA_DIR}pitcher_splits.csv", dtype={"pitcher_id": str}).set_index("pitcher_id")
    except FileNotFoundError:
        pass
    try:
        batter_splits = pd.read_csv(f"{DATA_DIR}batter_splits.csv", dtype={"batter_id": str}).set_index("batter_id")
    except FileNotFoundError:
        pass
    return {
        "pitchers": pitchers, "batters": batters,
        "parks": parks, "league": league,
        "pitcher_splits": pitcher_splits, "batter_splits": batter_splits,
        "calibration": load_calibration_params(),
    }


def get_player_rates(pid, ptype, data):
    league = data["league"].to_dict()
    df = data["pitchers"] if ptype == "pitcher" else data["batters"]
    if pid in df.index:
        return df.loc[pid].to_dict()
    fallback = dict(league)
    fallback["hand"] = "R"
    fallback["name"] = f"Unknown ({pid})"
    if ptype != "pitcher":
        fallback["sac_bunt_prob"] = 0.0
    return fallback


def get_player_splits(pid, ptype, data):
    """Look up splits row, or None if missing."""
    df = data.get("pitcher_splits" if ptype == "pitcher" else "batter_splits")
    if df is None or pid not in df.index:
        return None
    return df.loc[pid].to_dict()


def _safe_pa(splits, prefix):
    """Read a split's PA count safely, returning 0 on missing/NaN."""
    if not splits:
        return 0
    v = splits.get(f"{prefix}pa")
    try:
        return int(float(v)) if v is not None else 0
    except (ValueError, TypeError):
        return 0


def split_summary(splits, prefix):
    """Return {pa, k_rate, bb_rate, hr_rate} for a given split prefix, or None."""
    if not splits:
        return None
    pa = _safe_pa(splits, prefix)
    if pa < 1:
        return None
    out = {"pa": pa}
    for k in ("k_rate", "bb_rate", "hr_rate"):
        v = splits.get(f"{prefix}{k}")
        if v is not None:
            try:
                out[k] = round(float(v), 4)
            except (ValueError, TypeError):
                pass
    return out if len(out) > 1 else None


def build_rates(pitcher_row, batter_row, league, park_row,
                pitcher_splits=None, batter_splits=None):
    blended = build_blended_rates(
        pitcher_stats=pitcher_row, batter_stats=batter_row,
        league_averages=league,
        hr_park_factor=float(park_row.get("hr_factor", 1.0)),
        pitcher_hand=pitcher_row.get("hand", "R"),
        batter_hand=batter_row.get("hand", "R"),
        pitcher_splits=pitcher_splits,
        batter_splits=batter_splits,
        run_park_factor=float(park_row.get("run_factor", 1.0)),
    )
    return PAOutcomeRates(**{k: blended.get(k, 0.0) for k in RATE_COLUMNS})


def fetch_last_lineup(team_abbr):
    """
    Get a team's most recent batting order.
    Walks backward through up to 5 candidate games, skipping any whose
    boxscore is incomplete (the MLB API occasionally marks today's game
    as "Final" with an empty boxscore — defensive guard).
    """
    team_id = TEAM_IDS.get(team_abbr)
    if not team_id:
        return []
    today = date.today()
    start = (today - timedelta(days=14)).isoformat()
    end = (today - timedelta(days=1)).isoformat()
    try:
        url = f"{MLB_API}/schedule?sportId=1&teamId={team_id}&startDate={start}&endDate={end}&gameType=R"
        sched = requests.get(url, timeout=10).json()

        # Collect candidate game_pks in reverse-chronological order
        candidates = []
        for d in reversed(sched.get("dates", [])):
            game_date = d.get("date", "")
            for g in reversed(d.get("games", [])):
                # Belt-and-suspenders: skip games dated today or later
                if game_date >= today.isoformat():
                    continue
                if g.get("status", {}).get("abstractGameState") == "Final":
                    candidates.append(g["gamePk"])

        # Try each candidate until we get a real lineup
        for game_pk in candidates[:5]:
            try:
                box = requests.get(f"{MLB_API}/game/{game_pk}/boxscore", timeout=10).json()
                home_id = box["teams"]["home"]["team"]["id"]
                side = "home" if home_id == team_id else "away"
                team_data = box["teams"][side]
                batting_order = team_data.get("battingOrder", [])[:9]
                if len(batting_order) < 9:
                    continue  # incomplete boxscore, walk further back
                players = team_data.get("players", {})
                return [
                    {"id": str(pid), "name": players.get(f"ID{pid}", {}).get("person", {}).get("fullName", f"Player {pid}")}
                    for pid in batting_order
                ]
            except Exception:
                continue
        return []
    except Exception:
        return []


def fetch_kalshi_rfi_markets() -> list[dict]:
    """Fetch open Kalshi first-inning-run markets; return [] on any failure."""
    markets = []
    cursor = None
    try:
        for _ in range(5):
            params = {"series_ticker": KALSHI_SERIES, "status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(f"{KALSHI_API}/markets", params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            markets.extend(payload.get("markets", []))
            cursor = payload.get("cursor")
            if not cursor:
                break
    except Exception as e:
        print(f"  Warning: could not fetch Kalshi markets: {e}", flush=True)
        return []
    return markets


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def fetch_game_start_utc(game_id: str) -> datetime | None:
    """Look up a game's actual UTC start time from the MLB schedule API."""
    try:
        resp = requests.get(f"{MLB_API}/schedule", params={"gamePk": game_id}, timeout=10)
        resp.raise_for_status()
        dates = resp.json().get("dates", [])
        if not dates or not dates[0].get("games"):
            return None
        return _parse_iso_utc(dates[0]["games"][0].get("gameDate"))
    except Exception:
        return None


def match_kalshi_market(markets, game_date_iso_utc: str, away_abbr: str, home_abbr: str) -> dict | None:
    """Match a Kalshi market to an MLB game by scheduled UTC start time.

    Time proximity alone is unreliable: MLB games cluster tightly around
    common start slots (7:05-7:40 PM ET most nights), so the closest-in-time
    open market is often a *different* game. Require both teams' city hints
    to appear in the market title before accepting a match; otherwise a
    game can end up silently showing another game's live odds.
    """
    if not markets:
        return None

    game_dt = _parse_iso_utc(game_date_iso_utc)
    if game_dt is None:
        return None

    away_hint = KALSHI_CITY_HINTS.get(away_abbr, "").lower()
    home_hint = KALSHI_CITY_HINTS.get(home_abbr, "").lower()
    if not away_hint or not home_hint:
        return None

    best_market = None
    best_delta = None
    for market in markets:
        title = (market.get("title") or "").lower()
        if away_hint not in title or home_hint not in title:
            continue
        market_dt = _parse_iso_utc(market.get("occurrence_datetime"))
        if market_dt is None:
            continue
        delta_min = abs((market_dt - game_dt).total_seconds()) / 60
        if best_delta is None or delta_min < best_delta:
            best_delta = delta_min
            best_market = market

    if best_delta is None or best_delta > KALSHI_MATCH_TOLERANCE_MIN:
        return None
    return best_market


def build_kalshi_entry(market: dict | None, p_nrfi_game: float | None) -> dict:
    if market is None:
        return {"available": False}

    try:
        yes_bid = float(market.get("yes_bid_dollars") or 0)
        yes_ask = float(market.get("yes_ask_dollars") or 0)
    except (ValueError, TypeError):
        return {"available": False}

    if yes_bid <= 0 and yes_ask <= 0:
        return {"available": False}

    yrfi_prob = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else (yes_bid or yes_ask)
    nrfi_prob = 1 - yrfi_prob
    yrfi_ask = yes_ask
    yrfi_bid = yes_bid
    nrfi_ask = 1 - yes_bid
    nrfi_bid = 1 - yes_ask
    spread = round(yes_ask - yes_bid, 4) if (yes_bid and yes_ask) else None
    ticker = market.get("ticker")
    if not ticker:
        return {"available": False}

    return {
        "available": True,
        "ticker": ticker,
        "title": market.get("title", ""),
        "yrfi_prob": round(yrfi_prob, 4),
        "nrfi_prob": round(nrfi_prob, 4),
        "yrfi_ask": round(yrfi_ask, 4),
        "yrfi_bid": round(yrfi_bid, 4),
        "nrfi_ask": round(nrfi_ask, 4),
        "nrfi_bid": round(nrfi_bid, 4),
        "spread": spread,
        "low_liquidity": spread is not None and spread > 0.30,
        "edge_pp": round((p_nrfi_game - nrfi_prob) * 100, 1) if p_nrfi_game is not None else None,
        "market_url": f"https://kalshi.com/markets/kxmlbrfi/{ticker.lower()}",
    }


def run_game_model(away_lineup, home_lineup, away_sp_id, home_sp_id, home_team, data):
    """Run full NRFI model for a game, return results dict."""
    league = data["league"].to_dict()
    park_row = data["parks"].loc[home_team] if home_team in data["parks"].index else pd.Series({"hr_factor": 1.0, "run_factor": 1.0})

    home_sp = get_player_rates(home_sp_id, "pitcher", data)
    away_sp = get_player_rates(away_sp_id, "pitcher", data)
    home_sp_splits = get_player_splits(home_sp_id, "pitcher", data)
    away_sp_splits = get_player_splits(away_sp_id, "pitcher", data)

    # Pitcher hands for split lookup direction
    home_sp_hand = home_sp.get("hand", "R")
    away_sp_hand = away_sp.get("hand", "R")

    away_rates, away_details = [], []
    for b in away_lineup:
        br = get_player_rates(b["id"], "batter", data)
        bs = get_player_splits(b["id"], "batter", data)
        away_rates.append(build_rates(home_sp, br, league, park_row,
                                      pitcher_splits=home_sp_splits, batter_splits=bs))
        # Batter facing home pitcher: use vL_ if home_sp is L, else vR_
        prefix = "vl_" if home_sp_hand == "L" else "vr_"
        platoon = split_summary(bs, prefix)
        away_details.append({
            "name": br.get("name", b["name"]), "hand": br.get("hand", "?"),
            "k_rate": round(float(br.get("k_rate", 0)), 3),
            "hr_rate": round(float(br.get("hr_rate", 0)), 3),
            "bb_rate": round(float(br.get("bb_rate", 0)), 3),
            "platoon_split": platoon,  # vs the pitcher's hand
            "platoon_vs": home_sp_hand,
        })

    home_rates, home_details = [], []
    for b in home_lineup:
        br = get_player_rates(b["id"], "batter", data)
        bs = get_player_splits(b["id"], "batter", data)
        home_rates.append(build_rates(away_sp, br, league, park_row,
                                      pitcher_splits=away_sp_splits, batter_splits=bs))
        prefix = "vl_" if away_sp_hand == "L" else "vr_"
        platoon = split_summary(bs, prefix)
        home_details.append({
            "name": br.get("name", b["name"]), "hand": br.get("hand", "?"),
            "k_rate": round(float(br.get("k_rate", 0)), 3),
            "hr_rate": round(float(br.get("hr_rate", 0)), 3),
            "bb_rate": round(float(br.get("bb_rate", 0)), 3),
            "platoon_split": platoon,
            "platoon_vs": away_sp_hand,
        })

    sim = simulate_nrfi(home_rates, away_rates, n_simulations=N_SIMS, seed=42)
    p_away_a, _ = compute_nrfi_analytic(away_rates)
    p_home_a, _ = compute_nrfi_analytic(home_rates)

    # Rich model detail for each half-inning
    away_detail = compute_half_inning_detail(away_rates)
    home_detail = compute_half_inning_detail(home_rates)

    # Per-batter NRFI impact (swap each batter with league avg, measure delta)
    league_rates_obj = build_rates(
        get_player_rates("_none_", "pitcher", data),  # league avg pitcher
        get_player_rates("_none_", "batter", data),    # league avg batter
        league, park_row
    )
    for side_rates, side_details, sp_row, lineup in [
        (away_rates, away_details, home_sp, away_lineup),
        (home_rates, home_details, away_sp, home_lineup),
    ]:
        base_p, _ = compute_nrfi_analytic(side_rates)
        for i in range(len(side_rates)):
            swapped = list(side_rates)
            swapped[i] = league_rates_obj
            swap_p, _ = compute_nrfi_analytic(swapped)
            side_details[i]["nrfi_impact"] = round(base_p - swap_p, 4)

    # Sensitivity analysis: perturb K%, BB%, HR% by +1pp on averaged rates
    def compute_sensitivity(rates_list):
        from model.transitions import _average_lineup_rates
        avg = _average_lineup_rates(rates_list)
        base_p, _ = compute_nrfi_analytic(rates_list)
        sens = {}
        for attr in ["k_rate", "bb_rate", "hr_rate"]:
            perturbed = _average_lineup_rates(rates_list)
            old_val = getattr(perturbed, attr)
            setattr(perturbed, attr, old_val + 0.01)
            # Renormalize: steal from gbout_rate
            setattr(perturbed, "gbout_rate", perturbed.gbout_rate - 0.01)
            try:
                perturbed.validate()
                perturbed_p, _ = compute_nrfi_analytic([perturbed] * 9)
                sens[attr] = round(perturbed_p - base_p, 4)
            except Exception:
                sens[attr] = 0.0
        return sens

    away_sensitivity = compute_sensitivity(away_rates)
    home_sensitivity = compute_sensitivity(home_rates)

    # Keep transition matrix for frontend heatmap visualization
    # (~6KB per half, rounded to 4 decimals)
    def _round_matrix(m):
        return [[round(v, 4) for v in row] for row in m] if m else None
    if "transition_matrix" in away_detail:
        away_detail["transition_matrix"] = _round_matrix(away_detail["transition_matrix"])
    if "transition_matrix" in home_detail:
        home_detail["transition_matrix"] = _round_matrix(home_detail["transition_matrix"])

    # Simulation details: PA distribution, convergence (per-half)
    away_sim_detail = simulate_with_details(away_rates, n_simulations=N_SIMS, seed=42, n_sample_traces=0)
    home_sim_detail = simulate_with_details(home_rates, n_simulations=N_SIMS, seed=43, n_sample_traces=0)
    # Remove per-half sample traces (we use full-inning traces instead)
    away_sim_detail.pop("sample_traces", None)
    home_sim_detail.pop("sample_traces", None)

    # Full first-inning traces with batter names and event details
    trace_rng = np.random.default_rng(44)
    away_names = [d["name"] for d in away_details]
    home_names = [d["name"] for d in home_details]
    full_inning_traces = []
    for _ in range(20):
        full_inning_traces.append(simulate_full_inning_traced(
            away_rates, home_rates, trace_rng,
            away_names=away_names, home_names=home_names,
        ))

    p_nrfi_game = float(sim["p_nrfi_game"])
    p_nrfi_game_calibrated = calibrate_probability(p_nrfi_game, data["calibration"])

    return {
        "results": {
            "p_nrfi_away": round(sim["p_nrfi_away"], 4),
            "p_nrfi_home": round(sim["p_nrfi_home"], 4),
            "p_nrfi_game": round(p_nrfi_game, 4),
            "p_nrfi_game_calibrated": round(p_nrfi_game_calibrated, 4),
            "p_nrfi_away_analytic": round(p_away_a, 4),
            "p_nrfi_home_analytic": round(p_home_a, 4),
            "p_nrfi_game_analytic": round(p_away_a * p_home_a, 4),
            "ci_lower": round(sim["p_nrfi_ci"][0], 4),
            "ci_upper": round(sim["p_nrfi_ci"][1], 4),
            "avg_pa_away": round(sim["avg_pa_away"], 2),
            "avg_pa_home": round(sim["avg_pa_home"], 2),
            "n_simulations": N_SIMS,
        },
        "model_detail": {
            "away_half": {**away_detail, "sensitivity": away_sensitivity, "simulation": away_sim_detail},
            "home_half": {**home_detail, "sensitivity": home_sensitivity, "simulation": home_sim_detail},
            "full_inning_traces": full_inning_traces,
        },
        "pitchers": {
            "away": {
                "id": away_sp_id,
                "name": away_sp.get("name", "Unknown"), "hand": away_sp_hand,
                "k_rate": round(float(away_sp.get("k_rate", 0)), 3),
                "hr_rate": round(float(away_sp.get("hr_rate", 0)), 3),
                "bb_rate": round(float(away_sp.get("bb_rate", 0)), 3),
                "first_inning": split_summary(away_sp_splits, "i1_"),
                "vs_R": split_summary(away_sp_splits, "vr_"),
                "vs_L": split_summary(away_sp_splits, "vl_"),
            },
            "home": {
                "id": home_sp_id,
                "name": home_sp.get("name", "Unknown"), "hand": home_sp_hand,
                "k_rate": round(float(home_sp.get("k_rate", 0)), 3),
                "hr_rate": round(float(home_sp.get("hr_rate", 0)), 3),
                "bb_rate": round(float(home_sp.get("bb_rate", 0)), 3),
                "first_inning": split_summary(home_sp_splits, "i1_"),
                "vs_R": split_summary(home_sp_splits, "vr_"),
                "vs_L": split_summary(home_sp_splits, "vl_"),
            },
        },
        "park": {
            "team": home_team,
            "hr_factor": round(float(park_row.get("hr_factor", 1.0)), 2),
            "run_factor": round(float(park_row.get("run_factor", 1.0)), 2),
        },
        "away_lineup": away_details,
        "home_lineup": home_details,
    }


def main():
    game_date = date.today().isoformat()
    if len(sys.argv) > 1 and len(sys.argv[1]) == 10 and sys.argv[1][4] == '-':
        game_date = sys.argv[1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Computing NRFI predictions for {game_date}...", flush=True)

    data = load_data()
    print(f"  {len(data['pitchers'])} pitchers, {len(data['batters'])} batters loaded", flush=True)

    # Fetch schedule
    url = f"{MLB_API}/schedule?sportId=1&date={game_date}&hydrate=lineups,probablePitcher"
    sched = requests.get(url, timeout=15).json()
    games_raw = sched.get("dates", [{}])[0].get("games", []) if sched.get("dates") else []
    print(f"  {len(games_raw)} games on schedule", flush=True)
    kalshi_markets = fetch_kalshi_rfi_markets()

    output_games = []

    for g in games_raw:
        game_start_utc = g.get("gameDate")
        away = g["teams"]["away"]
        home = g["teams"]["home"]
        away_abbr = TEAM_ABBREVS.get(away["team"]["id"], "???")
        home_abbr = TEAM_ABBREVS.get(home["team"]["id"], "???")

        away_pitcher = away.get("probablePitcher", {})
        home_pitcher = home.get("probablePitcher", {})

        lineups = g.get("lineups", {})
        away_lineup = [{"id": str(p["id"]), "name": p["fullName"]} for p in lineups.get("awayPlayers", [])[:9]]
        home_lineup = [{"id": str(p["id"]), "name": p["fullName"]} for p in lineups.get("homePlayers", [])[:9]]

        has_lineups = len(away_lineup) >= 9 and len(home_lineup) >= 9
        lineup_source = "official"

        # Fall back to last game's lineup if not posted
        if not has_lineups:
            print(f"  {away_abbr}@{home_abbr}: No lineups, fetching last game...", flush=True)
            if len(away_lineup) < 9:
                away_lineup = fetch_last_lineup(away_abbr)
            if len(home_lineup) < 9:
                home_lineup = fetch_last_lineup(home_abbr)
            has_lineups = len(away_lineup) >= 9 and len(home_lineup) >= 9
            lineup_source = "projected"
            time.sleep(0.2)

        has_pitchers = bool(away_pitcher.get("id")) and bool(home_pitcher.get("id"))

        game_entry = {
            "game_id": str(g["gamePk"]),
            "away_team": away_abbr,
            "home_team": home_abbr,
            "away_name": TEAM_NAMES.get(away_abbr, away_abbr),
            "home_name": TEAM_NAMES.get(home_abbr, home_abbr),
            "away_pitcher": {"id": str(away_pitcher.get("id", "")), "name": away_pitcher.get("fullName", "TBD")},
            "home_pitcher": {"id": str(home_pitcher.get("id", "")), "name": home_pitcher.get("fullName", "TBD")},
            "status": g.get("status", {}).get("detailedState", ""),
            "lineup_source": lineup_source,
            "modeled": False,
        }

        if has_lineups and has_pitchers:
            try:
                result = run_game_model(
                    away_lineup, home_lineup,
                    str(away_pitcher["id"]), str(home_pitcher["id"]),
                    home_abbr, data,
                )
                game_entry.update(result)
                game_entry["modeled"] = True
                p = result["results"]["p_nrfi_game"]
                print(f"  {away_abbr}@{home_abbr}: P(NRFI)={p:.1%} [{lineup_source}]", flush=True)
            except Exception as e:
                print(f"  {away_abbr}@{home_abbr}: ERROR {e}", flush=True)
        else:
            reason = "no pitchers" if not has_pitchers else "no lineups"
            print(f"  {away_abbr}@{home_abbr}: skipped ({reason})", flush=True)

        result_probs = game_entry.get("results", {})
        p_nrfi = result_probs.get("p_nrfi_game_calibrated", result_probs.get("p_nrfi_game"))
        matched = match_kalshi_market(kalshi_markets, game_start_utc, away_abbr, home_abbr) if game_start_utc else None
        game_entry["kalshi"] = build_kalshi_entry(matched, p_nrfi)
        if game_entry["kalshi"]["available"]:
            edge = game_entry["kalshi"]["edge_pp"]
            edge_str = f"{edge:+.1f}pp" if edge is not None else "n/a"
            print(
                f"  {away_abbr}@{home_abbr}: Kalshi match -> {game_entry['kalshi']['title']!r} "
                f"(edge {edge_str})",
                flush=True,
            )

        output_games.append(game_entry)

    # Compute daily summary
    modeled_games = [g for g in output_games if g.get("modeled")]
    summary = {}
    if modeled_games:
        probs = [g["results"]["p_nrfi_game"] for g in modeled_games]
        probs_arr = np.array(probs)
        ci_widths = [g["results"]["ci_upper"] - g["results"]["ci_lower"] for g in modeled_games]
        sim_ana_gaps = [abs(g["results"]["p_nrfi_game"] - g["results"]["p_nrfi_game_analytic"]) for g in modeled_games]

        best_g = modeled_games[int(np.argmax(probs_arr))]
        worst_g = modeled_games[int(np.argmin(probs_arr))]

        # Histogram bins
        bins = [0.30, 0.38, 0.46, 0.54, 0.62, 0.70]
        hist_counts = [int(np.sum((probs_arr >= bins[i]) & (probs_arr < bins[i+1]))) for i in range(len(bins)-1)]

        summary = {
            "n_games_modeled": len(modeled_games),
            "n_games_total": len(output_games),
            "mean_p_nrfi": round(float(probs_arr.mean()), 4),
            "median_p_nrfi": round(float(np.median(probs_arr)), 4),
            "std_p_nrfi": round(float(probs_arr.std()), 4),
            "best_nrfi": {
                "matchup": f"{best_g['away_team']}@{best_g['home_team']}",
                "value": round(float(probs_arr.max()), 4),
            },
            "worst_nrfi": {
                "matchup": f"{worst_g['away_team']}@{worst_g['home_team']}",
                "value": round(float(probs_arr.min()), 4),
            },
            "histogram": {"bins": bins, "counts": hist_counts},
            "mean_ci_width": round(float(np.mean(ci_widths)), 4),
            "mean_sim_analytic_gap": round(float(np.mean(sim_ana_gaps)), 4),
        }

    # Load validation results if available
    validation = {}
    val_path = Path(DATA_DIR) / "validation_results.csv"
    if val_path.exists():
        try:
            vdf = pd.read_csv(val_path)
            y_true = vdf["nrfi_actual"].values.astype(float)
            y_pred = vdf["p_nrfi_sim"].values
            y_ana = vdf["p_nrfi_analytic"].values
            bs = float(np.mean((y_pred - y_true) ** 2))
            bs_naive = float(np.mean((np.full_like(y_true, y_true.mean()) - y_true) ** 2))
            y_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
            ll = float(-np.mean(y_true * np.log(y_clip) + (1 - y_true) * np.log(1 - y_clip)))
            # Calibration
            cal_bins = np.linspace(0.3, 0.7, 6)
            calibration = []
            for j in range(len(cal_bins) - 1):
                mask = (y_pred >= cal_bins[j]) & (y_pred < cal_bins[j+1])
                if mask.sum() > 0:
                    calibration.append({
                        "bin": f"{cal_bins[j]:.0%}-{cal_bins[j+1]:.0%}",
                        "predicted": round(float(y_pred[mask].mean()), 4),
                        "actual": round(float(y_true[mask].mean()), 4),
                        "n": int(mask.sum()),
                    })
            validation = {
                "n_games": len(vdf),
                "date_range": [str(vdf["date"].min()), str(vdf["date"].max())],
                "actual_nrfi_rate": round(float(y_true.mean()), 4),
                "brier_score": round(bs, 5),
                "brier_skill_score": round(1 - bs / max(bs_naive, 1e-10), 4),
                "log_loss": round(ll, 5),
                "accuracy": round(float(np.mean((y_pred >= 0.5) == y_true)), 4),
                "calibration": calibration,
            }
        except Exception as e:
            print(f"  Warning: could not load validation data: {e}", flush=True)

    # Build output
    output = {
        "date": game_date,
        "computed_at": pd.Timestamp.now().isoformat(),
        "n_simulations": N_SIMS,
        "summary": summary,
        "validation": validation,
        "games": output_games,
    }

    # Write date-specific and latest files
    date_file = OUT_DIR / f"games_{game_date}.json"
    latest_file = OUT_DIR / "latest.json"

    with open(date_file, "w") as f:
        json.dump(output, f)
    with open(latest_file, "w") as f:
        json.dump(output, f)

    print(f"\nDone: {len(modeled_games)}/{len(output_games)} games modeled", flush=True)
    print(f"  {date_file}", flush=True)
    print(f"  {latest_file}", flush=True)


if __name__ == "__main__":
    main()
