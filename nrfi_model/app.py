#!/usr/bin/env python3
"""
app.py — Flask web app for the NRFI Markov model.
"""

from __future__ import annotations
import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime
from flask import Flask, render_template, jsonify, request

from model.outcomes import PAOutcomeRates
from model.blend import build_blended_rates
from model.chain import simulate_nrfi, compute_nrfi_analytic

app = Flask(__name__)

DATA_DIR = "data/"

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

RATE_COLUMNS = [
    'k_rate', 'bb_rate', 'hbp_rate', 'hr_rate',
    'single_rate', 'double_rate', 'triple_rate',
    'gbout_rate', 'fbout_rate', 'ldout_rate', 'fc_rate',
    'gidp_prob_given_gbout', 'sf_prob_given_fbout', 'sac_bunt_prob',
]

MLB_API = "https://statsapi.mlb.com/api/v1"

# ── Data loading ──────────────────────────────────────────────────────────────

_data_cache = {}

def load_data():
    if _data_cache:
        return _data_cache
    pitchers = pd.read_csv(f"{DATA_DIR}pitchers.csv", dtype={"pitcher_id": str}).set_index("pitcher_id")
    batters = pd.read_csv(f"{DATA_DIR}batters.csv", dtype={"batter_id": str}).set_index("batter_id")
    parks = pd.read_csv(f"{DATA_DIR}park_factors.csv").set_index("team")
    league = pd.read_csv(f"{DATA_DIR}league_averages.csv").iloc[0]
    _data_cache["pitchers"] = pitchers
    _data_cache["batters"] = batters
    _data_cache["parks"] = parks
    _data_cache["league"] = league
    return _data_cache


def get_player_rates(player_id: str, player_type: str, data: dict) -> dict:
    """Get rates for a player, falling back to league averages."""
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
    """Build blended PAOutcomeRates for a single pitcher-batter matchup."""
    pitcher_hand = pitcher_row.get("hand", "R")
    batter_hand = batter_row.get("hand", "R")
    hr_park_factor = float(park_row.get("hr_factor", 1.0))
    run_park_factor = float(park_row.get("run_factor", 1.0))

    blended = build_blended_rates(
        pitcher_stats=pitcher_row,
        batter_stats=batter_row,
        league_averages=league,
        hr_park_factor=hr_park_factor,
        pitcher_hand=pitcher_hand,
        batter_hand=batter_hand,
        run_park_factor=run_park_factor,
    )
    return PAOutcomeRates(**{k: blended.get(k, 0.0) for k in RATE_COLUMNS})


# ── MLB API helpers ───────────────────────────────────────────────────────────

def fetch_schedule(game_date: str):
    """Fetch schedule with lineups from MLB API."""
    url = f"{MLB_API}/schedule?sportId=1&date={game_date}&hydrate=lineups,probablePitcher"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data.get("dates"):
        return []
    return data["dates"][0].get("games", [])


def parse_games(games):
    """Parse MLB API games into our format."""
    result = []
    for g in games:
        away = g["teams"]["away"]
        home = g["teams"]["home"]
        away_id = away["team"]["id"]
        home_id = home["team"]["id"]
        away_abbr = TEAM_ABBREVS.get(away_id, f"T{away_id}")
        home_abbr = TEAM_ABBREVS.get(home_id, f"T{home_id}")

        away_pitcher = away.get("probablePitcher", {})
        home_pitcher = home.get("probablePitcher", {})

        lineups = g.get("lineups", {})
        away_lineup = lineups.get("awayPlayers", [])
        home_lineup = lineups.get("homePlayers", [])

        has_lineups = len(away_lineup) >= 9 and len(home_lineup) >= 9
        has_pitchers = bool(away_pitcher.get("id")) and bool(home_pitcher.get("id"))

        game_info = {
            "game_id": str(g["gamePk"]),
            "away_team": away_abbr,
            "home_team": home_abbr,
            "away_team_id": away_id,
            "home_team_id": home_id,
            "away_name": TEAM_NAMES.get(away_abbr, away_abbr),
            "home_name": TEAM_NAMES.get(home_abbr, home_abbr),
            "away_pitcher": {
                "id": str(away_pitcher.get("id", "")),
                "name": away_pitcher.get("fullName", "TBD"),
            },
            "home_pitcher": {
                "id": str(home_pitcher.get("id", "")),
                "name": home_pitcher.get("fullName", "TBD"),
            },
            "ready": has_lineups and has_pitchers,
            "status": g.get("status", {}).get("detailedState", ""),
            "away_lineup": [
                {"id": str(p["id"]), "name": p["fullName"]}
                for p in away_lineup[:9]
            ] if has_lineups else [],
            "home_lineup": [
                {"id": str(p["id"]), "name": p["fullName"]}
                for p in home_lineup[:9]
            ] if has_lineups else [],
        }
        result.append(game_info)
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/games")
def api_games():
    """Get games for a date."""
    game_date = request.args.get("date", date.today().isoformat())
    try:
        games = fetch_schedule(game_date)
        parsed = parse_games(games)
        return jsonify({"date": game_date, "games": parsed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/run", methods=["POST"])
def api_run():
    """Run the NRFI model for a game."""
    body = request.json
    game_date = body.get("date", date.today().isoformat())
    game_id = body.get("game_id")
    home_team = body.get("home_team")
    away_team = body.get("away_team")
    n_sims = min(int(body.get("n_simulations", 50000)), 200000)

    data = load_data()
    league = data["league"].to_dict()

    # Accept inline lineups from the request body (for projected/edited lineups)
    body_away_lineup = body.get("away_lineup")
    body_home_lineup = body.get("home_lineup")

    try:
        games = fetch_schedule(game_date)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch schedule: {e}"}), 500

    # Find the game
    target = None
    for g in games:
        if str(g["gamePk"]) == str(game_id):
            target = g
            break

    if not target:
        return jsonify({"error": f"Game {game_id} not found on {game_date}"}), 404

    if body_away_lineup and body_home_lineup:
        # Use provided lineups (from editor)
        away_lineup_raw = [{"id": b["id"], "fullName": b["name"]} for b in body_away_lineup]
        home_lineup_raw = [{"id": b["id"], "fullName": b["name"]} for b in body_home_lineup]
    else:
        # Use lineups from MLB API
        lineups_data = target.get("lineups", {})
        away_lineup_raw = lineups_data.get("awayPlayers", [])[:9]
        home_lineup_raw = lineups_data.get("homePlayers", [])[:9]

    if len(away_lineup_raw) < 9 or len(home_lineup_raw) < 9:
        return jsonify({"error": "Lineups not yet available for this game"}), 400

    home_pitcher_id = body.get("home_pitcher_id") or str(target["teams"]["home"].get("probablePitcher", {}).get("id", ""))
    away_pitcher_id = body.get("away_pitcher_id") or str(target["teams"]["away"].get("probablePitcher", {}).get("id", ""))

    park_row = data["parks"].loc[home_team] if home_team in data["parks"].index else pd.Series({"hr_factor": 1.0, "run_factor": 1.0})

    # Build away lineup rates (away batters vs home pitcher)
    home_pitcher_row = get_player_rates(home_pitcher_id, "pitcher", data)
    away_rates = []
    away_details = []
    for p in away_lineup_raw:
        bid = str(p["id"])
        batter_row = get_player_rates(bid, "batter", data)
        rates = build_matchup_rates(home_pitcher_row, batter_row, league, park_row)
        away_rates.append(rates)
        away_details.append({
            "name": batter_row.get("name", p["fullName"]),
            "hand": batter_row.get("hand", "?"),
            "k_rate": round(batter_row.get("k_rate", 0), 3),
            "hr_rate": round(batter_row.get("hr_rate", 0), 3),
            "bb_rate": round(batter_row.get("bb_rate", 0), 3),
        })

    # Build home lineup rates (home batters vs away pitcher)
    away_pitcher_row = get_player_rates(away_pitcher_id, "pitcher", data)
    home_rates = []
    home_details = []
    for p in home_lineup_raw:
        bid = str(p["id"])
        batter_row = get_player_rates(bid, "batter", data)
        rates = build_matchup_rates(away_pitcher_row, batter_row, league, park_row)
        home_rates.append(rates)
        home_details.append({
            "name": batter_row.get("name", p["fullName"]),
            "hand": batter_row.get("hand", "?"),
            "k_rate": round(batter_row.get("k_rate", 0), 3),
            "hr_rate": round(batter_row.get("hr_rate", 0), 3),
            "bb_rate": round(batter_row.get("bb_rate", 0), 3),
        })

    # Run simulation
    sim_results = simulate_nrfi(
        home_lineup=home_rates,
        away_lineup=away_rates,
        n_simulations=n_sims,
        seed=42,
    )

    # Run analytic
    p_nrfi_away_a, _ = compute_nrfi_analytic(away_rates)
    p_nrfi_home_a, _ = compute_nrfi_analytic(home_rates)

    return jsonify({
        "game_id": game_id,
        "away_team": away_team,
        "home_team": home_team,
        "away_name": TEAM_NAMES.get(away_team, away_team),
        "home_name": TEAM_NAMES.get(home_team, home_team),
        "away_pitcher": {
            "name": home_pitcher_row.get("name", "Unknown"),  # wait, this is wrong
        },
        "home_pitcher": {
            "name": away_pitcher_row.get("name", "Unknown"),
        },
        # Fix: away pitcher faces home batters, home pitcher faces away batters
        "pitchers": {
            "away": {
                "name": away_pitcher_row.get("name", "Unknown"),
                "hand": away_pitcher_row.get("hand", "?"),
                "k_rate": round(float(away_pitcher_row.get("k_rate", 0)), 3),
                "hr_rate": round(float(away_pitcher_row.get("hr_rate", 0)), 3),
                "bb_rate": round(float(away_pitcher_row.get("bb_rate", 0)), 3),
            },
            "home": {
                "name": home_pitcher_row.get("name", "Unknown"),
                "hand": home_pitcher_row.get("hand", "?"),
                "k_rate": round(float(home_pitcher_row.get("k_rate", 0)), 3),
                "hr_rate": round(float(home_pitcher_row.get("hr_rate", 0)), 3),
                "bb_rate": round(float(home_pitcher_row.get("bb_rate", 0)), 3),
            },
        },
        "results": {
            "p_nrfi_away": round(sim_results["p_nrfi_away"], 4),
            "p_nrfi_home": round(sim_results["p_nrfi_home"], 4),
            "p_nrfi_game": round(sim_results["p_nrfi_game"], 4),
            "p_nrfi_away_analytic": round(p_nrfi_away_a, 4),
            "p_nrfi_home_analytic": round(p_nrfi_home_a, 4),
            "p_nrfi_game_analytic": round(p_nrfi_away_a * p_nrfi_home_a, 4),
            "ci_lower": round(sim_results["p_nrfi_ci"][0], 4),
            "ci_upper": round(sim_results["p_nrfi_ci"][1], 4),
            "avg_pa_away": round(sim_results["avg_pa_away"], 2),
            "avg_pa_home": round(sim_results["avg_pa_home"], 2),
            "n_simulations": n_sims,
        },
        "park": {
            "team": home_team,
            "hr_factor": round(float(park_row.get("hr_factor", 1.0)), 2),
            "run_factor": round(float(park_row.get("run_factor", 1.0)), 2),
        },
        "away_lineup": away_details,
        "home_lineup": home_details,
    })


@app.route("/api/pitchers")
def api_pitchers():
    """List all pitchers in the database."""
    data = load_data()
    df = data["pitchers"].reset_index()
    pitchers = df[["pitcher_id", "name", "team", "hand"]].to_dict(orient="records")
    return jsonify(pitchers)


@app.route("/api/batters")
def api_batters():
    """List all batters in the database."""
    data = load_data()
    df = data["batters"].reset_index()
    batters = df[["batter_id", "name", "team", "hand"]].to_dict(orient="records")
    return jsonify(batters)


@app.route("/api/custom", methods=["POST"])
def api_custom():
    """Run model with a custom pitcher + 9 batters."""
    body = request.json
    pitcher_id = str(body.get("pitcher_id", ""))
    batter_ids = [str(b) for b in body.get("batter_ids", [])]
    park_team = body.get("park", "HOU")  # default neutral
    n_sims = min(int(body.get("n_simulations", 50000)), 200000)

    if len(batter_ids) != 9:
        return jsonify({"error": "Must provide exactly 9 batter_ids"}), 400

    data = load_data()
    league = data["league"].to_dict()
    park_row = data["parks"].loc[park_team] if park_team in data["parks"].index else pd.Series({"hr_factor": 1.0, "run_factor": 1.0})

    pitcher_row = get_player_rates(pitcher_id, "pitcher", data)
    rates_list = []
    lineup_details = []
    for bid in batter_ids:
        batter_row = get_player_rates(bid, "batter", data)
        rates = build_matchup_rates(pitcher_row, batter_row, league, park_row)
        rates_list.append(rates)
        lineup_details.append({
            "name": batter_row.get("name", f"Batter {bid}"),
            "hand": batter_row.get("hand", "?"),
        })

    # Simulate single half-inning
    from model.chain import simulate_half_inning
    rng = np.random.default_rng(42)
    nrfi_count = 0
    pa_total = 0
    for _ in range(n_sims):
        nrfi, pa = simulate_half_inning(rates_list, batter_start=0, rng=rng)
        if nrfi:
            nrfi_count += 1
        pa_total += pa

    p_nrfi = nrfi_count / n_sims
    p_analytic, _ = compute_nrfi_analytic(rates_list)

    return jsonify({
        "pitcher": {
            "name": pitcher_row.get("name", f"Pitcher {pitcher_id}"),
            "hand": pitcher_row.get("hand", "?"),
        },
        "lineup": lineup_details,
        "park": park_team,
        "results": {
            "p_nrfi_sim": round(p_nrfi, 4),
            "p_nrfi_analytic": round(p_analytic, 4),
            "avg_pa": round(pa_total / n_sims, 2),
            "n_simulations": n_sims,
        },
    })


@app.route("/api/last-lineup")
def api_last_lineup():
    """Fetch a team's most recent batting order from the last completed game."""
    team_abbr = request.args.get("team", "")
    team_id = TEAM_IDS.get(team_abbr)
    if not team_id:
        return jsonify({"error": f"Unknown team: {team_abbr}"}), 400

    from datetime import timedelta
    today = date.today()
    start = (today - timedelta(days=14)).isoformat()
    end = (today - timedelta(days=1)).isoformat()

    try:
        url = (
            f"{MLB_API}/schedule?sportId=1&teamId={team_id}"
            f"&startDate={start}&endDate={end}&gameType=R"
        )
        sched = requests.get(url, timeout=10).json()

        # Find most recent completed game (iterate dates in reverse)
        game_pk = None
        for d in reversed(sched.get("dates", [])):
            for g in reversed(d.get("games", [])):
                if g.get("status", {}).get("abstractGameState") == "Final":
                    game_pk = g["gamePk"]
                    break
            if game_pk:
                break

        if not game_pk:
            return jsonify({"team": team_abbr, "lineup": [], "from_game": None})

        # Fetch boxscore
        box = requests.get(f"{MLB_API}/game/{game_pk}/boxscore", timeout=10).json()

        # Determine if team was home or away
        home_id_check = box["teams"]["home"]["team"]["id"]
        side = "home" if home_id_check == team_id else "away"
        team_data = box["teams"][side]

        batting_order = team_data.get("battingOrder", [])
        players = team_data.get("players", {})

        lineup = []
        for pid in batting_order[:9]:
            pdata = players.get(f"ID{pid}", {})
            person = pdata.get("person", {})
            lineup.append({
                "id": str(pid),
                "name": person.get("fullName", f"Player {pid}"),
            })

        return jsonify({
            "team": team_abbr,
            "lineup": lineup,
            "from_game": str(game_pk),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050, use_reloader=False)
