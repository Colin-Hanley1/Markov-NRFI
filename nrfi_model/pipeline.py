# pipeline.py

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

from model.outcomes import PAOutcomeRates
from model.blend import build_blended_rates
from model.chain import simulate_nrfi, compute_nrfi_analytic


DATA_DIR = "data/"

RATE_COLUMNS = [
    'k_rate', 'bb_rate', 'hbp_rate', 'hr_rate',
    'single_rate', 'double_rate', 'triple_rate',
    'gbout_rate', 'fbout_rate', 'ldout_rate', 'fc_rate',
    'gidp_prob_given_gbout', 'sf_prob_given_fbout', 'sac_bunt_prob',
]


def load_data() -> dict:
    """Load all CSVs and return as a dict of DataFrames."""
    pitchers = pd.read_csv(f"{DATA_DIR}pitchers.csv", dtype={'pitcher_id': str}).set_index('pitcher_id')
    batters  = pd.read_csv(f"{DATA_DIR}batters.csv", dtype={'batter_id': str}).set_index('batter_id')
    parks    = pd.read_csv(f"{DATA_DIR}park_factors.csv").set_index('team')
    lineups  = pd.read_csv(f"{DATA_DIR}lineups.csv", dtype={'game_id': str, 'batter_id': str, 'pitcher_id': str})
    league   = pd.read_csv(f"{DATA_DIR}league_averages.csv").iloc[0]
    return {
        "pitchers": pitchers,
        "batters":  batters,
        "parks":    parks,
        "lineups":  lineups,
        "league":   league,
    }


def build_lineup_rates(
    game_id:       str,
    side:          str,
    data:          dict,
    home_team:     str,
) -> list[PAOutcomeRates]:
    """
    Build a list of 9 PAOutcomeRates objects (batting order slots 1–9)
    for one team's half-inning.

    Parameters
    ----------
    game_id : str
    side : str  ('home' or 'away')
    data : dict  (from load_data())
    home_team : str  (team abbreviation for park factor lookup)
    """
    lineups  = data["lineups"]
    pitchers = data["pitchers"]
    batters  = data["batters"]
    parks    = data["parks"]
    league   = data["league"].to_dict()

    game_lineups = lineups[
        (lineups['game_id'] == game_id) &
        (lineups['side'] == side)
    ].sort_values('slot')

    if len(game_lineups) != 9:
        raise ValueError(
            f"Expected 9 batters for game {game_id} side {side}, "
            f"got {len(game_lineups)}."
        )

    pitcher_id   = game_lineups['pitcher_id'].iloc[0]
    if pitcher_id in pitchers.index:
        pitcher_row  = pitchers.loc[pitcher_id].to_dict()
    else:
        print(f"  WARNING: pitcher {pitcher_id} not found, using league averages")
        pitcher_row = dict(league)
        pitcher_row['hand'] = 'R'
    pitcher_hand = pitcher_row['hand']

    park_row       = parks.loc[home_team]
    hr_park_factor = float(park_row['hr_factor'])
    run_park_factor = float(park_row.get('run_factor', 1.0))

    rates_list = []
    for _, row in game_lineups.iterrows():
        batter_id   = row['batter_id']
        if batter_id in batters.index:
            batter_row  = batters.loc[batter_id].to_dict()
        else:
            print(f"  WARNING: batter {batter_id} not found, using league averages")
            batter_row = dict(league)
            batter_row['hand'] = 'R'
            batter_row['sac_bunt_prob'] = 0.0
        batter_hand = batter_row['hand']

        blended = build_blended_rates(
            pitcher_stats   = pitcher_row,
            batter_stats    = batter_row,
            league_averages = league,
            hr_park_factor  = hr_park_factor,
            pitcher_hand    = pitcher_hand,
            batter_hand     = batter_hand,
            run_park_factor = run_park_factor,
        )

        rates = PAOutcomeRates(**{k: blended.get(k, 0.0) for k in RATE_COLUMNS})
        rates_list.append(rates)

    return rates_list


def run_game(
    game_id:       str,
    home_team:     str,
    away_team:     str,
    n_simulations: int = 100_000,
    seed:          int = 42,
    analytic:      bool = True,
) -> dict:
    """
    Full pipeline for one game.

    Parameters
    ----------
    game_id : str
    home_team : str
    away_team : str
    n_simulations : int
    seed : int
    analytic : bool
        If True, also compute the analytic (fundamental matrix) estimate.

    Returns
    -------
    dict
        All results with keys from simulate_nrfi() plus optional analytic estimates.
    """
    data = load_data()

    home_rates = build_lineup_rates(game_id, 'home', data, home_team)
    away_rates = build_lineup_rates(game_id, 'away', data, home_team)

    results = simulate_nrfi(
        home_lineup    = home_rates,
        away_lineup    = away_rates,
        n_simulations  = n_simulations,
        seed           = seed,
    )

    if analytic:
        p_nrfi_away_a, _ = compute_nrfi_analytic(away_rates)
        p_nrfi_home_a, _ = compute_nrfi_analytic(home_rates)
        results["p_nrfi_away_analytic"] = p_nrfi_away_a
        results["p_nrfi_home_analytic"] = p_nrfi_home_a
        results["p_nrfi_game_analytic"] = p_nrfi_away_a * p_nrfi_home_a

    results["game_id"]   = game_id
    results["home_team"] = home_team
    results["away_team"] = away_team

    _print_results(results)
    return results


def _print_results(r: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  NRFI Model — {r.get('away_team','Away')} @ {r.get('home_team','Home')}")
    print(f"{'='*55}")
    print(f"  P(NRFI away half):  {r['p_nrfi_away']:.4f}")
    print(f"  P(NRFI home half):  {r['p_nrfi_home']:.4f}")
    print(f"  P(NRFI game, sim):  {r['p_nrfi_game']:.4f}")
    if 'p_nrfi_game_analytic' in r:
        print(f"  P(NRFI game, ana):  {r['p_nrfi_game_analytic']:.4f}")
    lo, hi = r['p_nrfi_ci']
    print(f"  95% CI (Wilson):    [{lo:.4f}, {hi:.4f}]")
    print(f"  Avg PA / away half: {r['avg_pa_away']:.2f}")
    print(f"  Avg PA / home half: {r['avg_pa_home']:.2f}")
    print(f"  Simulations:        {r['n_simulations']:,}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python pipeline.py <game_id> <home_team> <away_team>")
        sys.exit(1)
    run_game(
        game_id   = sys.argv[1],
        home_team = sys.argv[2],
        away_team = sys.argv[3],
    )
