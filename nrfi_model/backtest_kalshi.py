#!/usr/bin/env python3
"""
Join historical NRFI model results to Kalshi KXMLBRFI pregame prices.

Outputs:
    data/kalshi_backtest_joined.csv
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from compute_daily import (
    KALSHI_API,
    KALSHI_CITY_HINTS,
    KALSHI_MATCH_TOLERANCE_MIN,
    KALSHI_SERIES,
    calibrate_probability,
    fetch_game_start_utc,
    load_calibration_params,
)


BASE_DIR = Path(__file__).resolve().parent
RESULTS_LOG = BASE_DIR / "docs/data/results_log.json"
OUT_PATH = BASE_DIR / "data/kalshi_backtest_joined.csv"
START_DATE = "2026-05-05"
REQUEST_DELAY_SEC = 0.15  # be polite to Kalshi's/MLB's public APIs across ~850 sequential calls

FIELDNAMES = [
    "game_id",
    "date",
    "away_team",
    "home_team",
    "model_p_nrfi",
    "actual_nrfi",
    "kalshi_nrfi_ask",
    "kalshi_nrfi_bid",
    "kalshi_yrfi_ask",
    "kalshi_yrfi_bid",
    "edge_pp",
]


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _date_to_ts(value: str, add_days: int = 0) -> int:
    dt = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return int((dt + timedelta(days=add_days)).timestamp())


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_settled_markets(start_date: str, end_date: str) -> list[dict]:
    """Fetch settled Kalshi first-inning-run markets; return [] on failure."""
    markets: list[dict] = []
    cursor = None
    try:
        while True:
            params = {
                "series_ticker": KALSHI_SERIES,
                "status": "settled",
                "min_close_ts": _date_to_ts(start_date),
                "max_close_ts": _date_to_ts(end_date, add_days=1),
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(f"{KALSHI_API}/markets", params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            markets.extend(payload.get("markets", []))
            cursor = payload.get("cursor")
            if not cursor:
                break
    except Exception as e:
        print(f"Warning: could not fetch settled Kalshi markets: {e}", flush=True)
        return []
    return markets


def match_settled_market(
    markets: list[dict], game_start_utc: datetime, away_abbr: str, home_abbr: str
) -> dict | None:
    """Match by title (both team hints) then closest occurrence_datetime,
    exactly mirroring compute_daily.py::match_kalshi_market for the live
    feature -- team-pair-only + date-window matching is not reliable because
    the same two teams routinely play on consecutive days (a normal series,
    not just doubleheaders), which made ~2/3 of games "ambiguous" in an
    earlier version of this script.
    """
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
        delta_min = abs((market_dt - game_start_utc).total_seconds()) / 60
        if best_delta is None or delta_min < best_delta:
            best_delta = delta_min
            best_market = market

    if best_delta is None or best_delta > KALSHI_MATCH_TOLERANCE_MIN:
        return None
    return best_market


def _get_with_backoff(url: str, params: dict, max_retries: int = 4) -> requests.Response:
    """GET with exponential backoff on 429 (Kalshi rate limit under ~850 sequential calls)."""
    delay = 1.0
    for attempt in range(max_retries + 1):
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        if attempt == max_retries:
            resp.raise_for_status()
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def fetch_pregame_price(
    ticker: str, occurrence_dt: datetime, minutes_before: int = 60
) -> dict | None:
    """Fetch the hourly candle closest to the target pregame timestamp."""
    target = occurrence_dt - timedelta(minutes=minutes_before)
    start_ts = int((occurrence_dt - timedelta(days=7)).timestamp())
    end_ts = int(occurrence_dt.timestamp())
    url = f"{KALSHI_API}/series/{KALSHI_SERIES}/markets/{ticker}/candlesticks"
    try:
        resp = _get_with_backoff(
            url,
            params={
                "period_interval": 60,
                "start_ts": start_ts,
                "end_ts": end_ts,
            },
        )
        candles = resp.json().get("candlesticks", [])
    except Exception as e:
        print(f"Warning: could not fetch candlesticks for {ticker}: {e}", flush=True)
        return None

    best = None
    best_delta = None
    for candle in candles:
        end_period_ts = candle.get("end_period_ts")
        if end_period_ts is None:
            continue
        delta = abs(float(end_period_ts) - target.timestamp())
        if best_delta is None or delta < best_delta:
            best = candle
            best_delta = delta

    if best is None:
        return None

    yes_bid = _float_or_none((best.get("yes_bid") or {}).get("close_dollars"))
    yes_ask = _float_or_none((best.get("yes_ask") or {}).get("close_dollars"))
    if yes_bid is None or yes_ask is None:
        return None
    if not (0 <= yes_bid <= 1 and 0 <= yes_ask <= 1):
        return None

    nrfi_bid = 1 - yes_ask
    nrfi_ask = 1 - yes_bid
    return {
        "yrfi_bid": yes_bid,
        "yrfi_ask": yes_ask,
        "yrfi_mid": (yes_bid + yes_ask) / 2,
        "nrfi_bid": nrfi_bid,
        "nrfi_ask": nrfi_ask,
        "nrfi_mid": (nrfi_bid + nrfi_ask) / 2,
    }


def _load_results() -> list[dict]:
    payload = json.loads(RESULTS_LOG.read_text())
    if isinstance(payload, dict):
        entries = payload.get("entries", [])
    else:
        entries = payload
    return [e for e in entries if e.get("date", "") >= START_DATE]


def _load_cached() -> dict[str, dict]:
    if not OUT_PATH.exists():
        return {}
    with OUT_PATH.open(newline="") as f:
        return {row["game_id"]: row for row in csv.DictReader(f)}


def _write_cache(rows: list[dict]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["date"], r["game_id"]))
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _model_probability(entry: dict, calibration: dict) -> float | None:
    # results_log.json (written by check_results.py) only ever stores the raw
    # p_nrfi_predicted -- p_nrfi_game_calibrated is a compute_daily.py output
    # field, not something check_results.py logs historically. So for every
    # existing entry we must apply the fitted calibration transform ourselves
    # to get a calibrated probability; the p_nrfi_game_calibrated lookup is
    # kept only in case results_log.json ever gains that field directly.
    if entry.get("p_nrfi_game_calibrated") is not None:
        return _float_or_none(entry.get("p_nrfi_game_calibrated"))
    raw = _float_or_none(entry.get("p_nrfi_predicted") or entry.get("model_p_nrfi"))
    if raw is None:
        return None
    return calibrate_probability(raw, calibration)


def main() -> None:
    entries = _load_results()
    if not entries:
        print(f"No results_log entries found with date >= {START_DATE}.", flush=True)
        return

    calibration = load_calibration_params()
    print(f"Calibration in use: a={calibration['a']:.6f}, b={calibration['b']:.6f}", flush=True)

    cached = _load_cached()
    rows = list(cached.values())
    end_date = max(e["date"] for e in entries)
    markets = fetch_settled_markets(START_DATE, end_date)

    matched = len(cached)
    unmatched = 0
    no_game_time = 0
    price_missing = 0
    fetched = 0
    to_fetch = [e for e in entries if str(e["game_id"]) not in cached]
    print(f"Fetching {len(to_fetch)} new games (of {len(entries)} in window)...", flush=True)

    for i, entry in enumerate(to_fetch):
        game_id = str(entry["game_id"])

        game_start_utc = fetch_game_start_utc(game_id)
        time.sleep(REQUEST_DELAY_SEC)
        if game_start_utc is None:
            no_game_time += 1
            print(
                f"No game time: {entry['date']} {entry['away_team']}@{entry['home_team']} ({game_id})",
                flush=True,
            )
            continue

        market = match_settled_market(
            markets, game_start_utc, entry["away_team"], entry["home_team"]
        )
        if market is None:
            unmatched += 1
            print(
                f"Unmatched: {entry['date']} {entry['away_team']}@{entry['home_team']} ({game_id})",
                flush=True,
            )
            continue

        price = fetch_pregame_price(str(market["ticker"]), game_start_utc)
        time.sleep(REQUEST_DELAY_SEC)
        if price is None:
            price_missing += 1
            continue

        p_model = _model_probability(entry, calibration)
        if p_model is None:
            continue

        row = {
            "game_id": game_id,
            "date": entry["date"],
            "away_team": entry["away_team"],
            "home_team": entry["home_team"],
            "model_p_nrfi": round(p_model, 4),
            "actual_nrfi": bool(entry.get("nrfi_actual")),
            "kalshi_nrfi_ask": round(price["nrfi_ask"], 4),
            "kalshi_nrfi_bid": round(price["nrfi_bid"], 4),
            "kalshi_yrfi_ask": round(price["yrfi_ask"], 4),
            "kalshi_yrfi_bid": round(price["yrfi_bid"], 4),
            "edge_pp": round((p_model - price["nrfi_mid"]) * 100, 1),
        }
        rows.append(row)
        matched += 1
        fetched += 1

        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(to_fetch)} processed", flush=True)
            _write_cache(rows)  # periodic checkpoint in case of a long run

    _write_cache(rows)

    print("\nKalshi backtest join summary", flush=True)
    print(f"  entries in window: {len(entries)}", flush=True)
    print(f"  matched rows: {matched} ({fetched} newly fetched)", flush=True)
    print(f"  unmatched (no Kalshi market found): {unmatched}", flush=True)
    print(f"  no game time (MLB API lookup failed): {no_game_time}", flush=True)
    print(f"  missing pregame price: {price_missing}", flush=True)
    print(f"  wrote: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
