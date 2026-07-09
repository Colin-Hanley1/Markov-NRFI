#!/usr/bin/env python3
"""
Refresh Kalshi odds in docs/data/latest.json without rerunning the NRFI model.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from compute_daily import (
    build_kalshi_entry,
    fetch_game_start_utc,
    fetch_kalshi_rfi_markets,
    match_kalshi_market,
)


BASE_DIR = Path(__file__).resolve().parent
DOCS_DATA_DIR = BASE_DIR / "docs/data"
LATEST_PATH = DOCS_DATA_DIR / "latest.json"


def _load_latest() -> dict:
    if not LATEST_PATH.exists():
        raise FileNotFoundError(f"{LATEST_PATH} does not exist. Run compute_daily.py first.")
    with LATEST_PATH.open() as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f)


def refresh_kalshi_odds() -> dict:
    current = _load_latest()
    updated = deepcopy(current)

    markets = fetch_kalshi_rfi_markets()
    if not markets:
        raise RuntimeError("No Kalshi markets returned; leaving latest.json unchanged.")

    refreshed = 0
    for game in updated.get("games", []):
        if not game.get("modeled"):
            continue

        game_id = str(game.get("game_id", ""))
        start_utc = fetch_game_start_utc(game_id)
        market = None
        if start_utc is not None:
            market = match_kalshi_market(
                markets,
                start_utc.isoformat(),
                game.get("away_team", ""),
                game.get("home_team", ""),
            )

        results = game.get("results", {})
        p_nrfi = results.get("p_nrfi_game_calibrated", results.get("p_nrfi_game"))
        game["kalshi"] = build_kalshi_entry(market, p_nrfi)
        if game["kalshi"].get("available"):
            refreshed += 1

    updated["kalshi_refreshed_at"] = datetime.now().isoformat()

    date_file = DOCS_DATA_DIR / f"games_{updated.get('date')}.json"
    _write_json(LATEST_PATH, updated)
    if updated.get("date") and date_file.exists():
        _write_json(date_file, updated)

    print(
        f"Refreshed Kalshi odds for {refreshed}/{len(updated.get('games', []))} games",
        flush=True,
    )
    print(f"  {LATEST_PATH}", flush=True)
    if updated.get("date") and date_file.exists():
        print(f"  {date_file}", flush=True)
    return updated


if __name__ == "__main__":
    refresh_kalshi_odds()
