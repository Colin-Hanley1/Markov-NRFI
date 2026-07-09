#!/usr/bin/env python3
"""
Maintain the forward paper-trading ledger from local prediction and result files.

Reads:
    docs/data/latest.json
    docs/data/results_log.json

Writes:
    docs/data/paper_trades.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from compute_daily import kalshi_taker_fee


DOCS_DATA = Path("docs/data")
LEDGER_PATH = DOCS_DATA / "paper_trades.json"
LATEST_PATH = DOCS_DATA / "latest.json"
RESULTS_PATH = DOCS_DATA / "results_log.json"


STRATEGIES = [
    {
        "id": f"yrfi_{threshold}pp_{sizing}",
        "label": f"{threshold}pp · {label}",
        "side": "YRFI",
        "threshold_pp": threshold,
        "sizing": sizing,
    }
    for threshold in (0, 3, 5)
    for sizing, label in (
        ("flat", "Flat"),
        ("pct_bankroll", "2% Bankroll"),
        ("kelly", "Kelly"),
    )
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def _new_strategy(strategy: dict, today: str) -> dict:
    return {
        "id": strategy["id"],
        "label": strategy["label"],
        "side": strategy["side"],
        "threshold_pp": strategy["threshold_pp"],
        "sizing": strategy["sizing"],
        "current_bankroll": 1.0,
        "balance_history": [{"date": today, "bankroll": 1.0}],
    }


def _load_ledger(today: str) -> dict:
    ledger = _load_json(LEDGER_PATH, {"strategies": {}, "trades": []})
    ledger.setdefault("strategies", {})
    ledger.setdefault("trades", [])
    for strategy in STRATEGIES:
        ledger["strategies"].setdefault(strategy["id"], _new_strategy(strategy, today))
    return ledger


def _result_index() -> dict[str, dict]:
    payload = _load_json(RESULTS_PATH, {"entries": []})
    return {str(entry["game_id"]): entry for entry in payload.get("entries", [])}


def _settle_open_trades(ledger: dict, results: dict[str, dict], now: str) -> int:
    settled = 0
    for trade in ledger["trades"]:
        if trade.get("status") != "open":
            continue
        result = results.get(str(trade.get("game_id")))
        if not result:
            continue

        side = trade["side"]
        actual_nrfi = bool(result.get("nrfi_actual"))
        won = (side == "YRFI" and not actual_nrfi) or (side == "NRFI" and actual_nrfi)
        stake = float(trade["stake"])
        entry_price = float(trade["entry_price"])
        contracts = stake / entry_price
        payout = contracts if won else 0.0
        fee = kalshi_taker_fee(entry_price) * contracts
        pnl = payout - stake - fee

        trade.update(
            {
                "status": "settled",
                "won": won,
                "contracts": round(contracts, 8),
                "payout": round(payout, 8),
                "fee": round(fee, 8),
                "pnl": round(pnl, 8),
                "settled_at": now,
            }
        )

        strategy = ledger["strategies"][trade["strategy_id"]]
        strategy["current_bankroll"] = round(float(strategy["current_bankroll"]) + pnl, 8)
        strategy.setdefault("balance_history", []).append(
            {
                "date": result.get("date") or now[:10],
                "bankroll": strategy["current_bankroll"],
            }
        )
        settled += 1
    return settled


def _model_side_probability(game: dict, side: str) -> float | None:
    results = game.get("results") or {}
    p_nrfi = results.get("p_nrfi_game_calibrated")
    if p_nrfi is None:
        p_nrfi = results.get("p_nrfi_game")
    if p_nrfi is None:
        return None
    p_nrfi = float(p_nrfi)
    return p_nrfi if side == "NRFI" else 1 - p_nrfi


def _stake(strategy: dict, current_bankroll: float, model_prob: float, entry_price: float) -> float:
    if strategy["sizing"] == "flat":
        return 0.01
    if strategy["sizing"] == "pct_bankroll":
        return 0.02 * current_bankroll
    if strategy["sizing"] == "kelly":
        fraction = 0.25 * max(0.0, (model_prob - entry_price) / (1 - entry_price))
        return min(fraction, 0.15) * current_bankroll
    raise ValueError(f"Unknown sizing rule: {strategy['sizing']}")


def _entry_price(kalshi: dict, side: str) -> float | None:
    price_key = "yrfi_ask" if side == "YRFI" else "nrfi_ask"
    if kalshi.get(price_key) is not None:
        return float(kalshi[price_key])

    prob_key = "yrfi_prob" if side == "YRFI" else "nrfi_prob"
    if kalshi.get(prob_key) is None or kalshi.get("spread") is None:
        return None
    return float(kalshi[prob_key]) + float(kalshi["spread"]) / 2


def _trade_id(strategy_id: str, game_id: str) -> str:
    return f"{strategy_id}:{game_id}"


def _place_new_trades(ledger: dict, latest: dict, now: str) -> int:
    existing = {
        (trade.get("strategy_id"), str(trade.get("game_id")))
        for trade in ledger["trades"]
    }
    placed = 0
    for game in latest.get("games", []):
        if not game.get("modeled"):
            continue
        kalshi = game.get("kalshi") or {}
        if not kalshi.get("available"):
            continue
        edge_pp = kalshi.get("edge_pp")
        if edge_pp is None:
            continue

        game_id = str(game["game_id"])
        for strategy in STRATEGIES:
            strategy_id = strategy["id"]
            if (strategy_id, game_id) in existing:
                continue

            side = strategy["side"]
            threshold = float(strategy["threshold_pp"])
            if side == "YRFI" and float(edge_pp) > -threshold:
                continue
            if side == "NRFI" and float(edge_pp) < threshold:
                continue

            entry_price = _entry_price(kalshi, side)
            if entry_price is None:
                continue
            entry_price = float(entry_price)
            if entry_price <= 0 or entry_price >= 1:
                continue

            model_prob = _model_side_probability(game, side)
            if model_prob is None:
                continue

            strategy_state = ledger["strategies"][strategy_id]
            current_bankroll = float(strategy_state["current_bankroll"])
            stake = _stake(strategy, current_bankroll, model_prob, entry_price)
            if stake <= 0:
                continue

            ledger["trades"].append(
                {
                    "trade_id": _trade_id(strategy_id, game_id),
                    "strategy_id": strategy_id,
                    "game_id": game_id,
                    "date": latest.get("date") or now[:10],
                    "matchup": f"{game.get('away_team', '')}@{game.get('home_team', '')}",
                    "side": side,
                    "edge_pp_at_entry": round(float(edge_pp), 1),
                    "entry_price": round(entry_price, 4),
                    "stake": round(stake, 8),
                    "placed_at": now,
                    "status": "open",
                }
            )
            existing.add((strategy_id, game_id))
            placed += 1
    return placed


def main() -> None:
    now = _now_iso()
    latest = _load_json(LATEST_PATH, {"date": now[:10], "games": []})
    ledger = _load_ledger(latest.get("date") or now[:10])

    settled = _settle_open_trades(ledger, _result_index(), now)
    placed = _place_new_trades(ledger, latest, now)

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("w") as f:
        json.dump(ledger, f, indent=2)
        f.write("\n")

    print(
        f"Paper trading ledger updated: {settled} settled, {placed} placed, "
        f"{len(ledger['trades'])} total trades."
    )


if __name__ == "__main__":
    main()
