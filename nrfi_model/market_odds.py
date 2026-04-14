#!/usr/bin/env python3
"""
market_odds.py — Helpers for market-implied NRFI probabilities.

Reads data/market_odds.csv (manually populated with sportsbook NRFI/YRFI
lines) and computes no-vig fair probabilities and Kelly-optimal stake
fractions vs the model's prediction.

CSV format:
    game_id,nrfi_american,yrfi_american,book
    824859,-115,+105,DK

If the file is empty or missing for a given game_id, returns None.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional, Dict


MARKET_PATH = Path("data/market_odds.csv")


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal (e.g. -110 -> 1.909, +120 -> 2.20)."""
    if odds >= 0:
        return 1 + odds / 100.0
    return 1 + 100.0 / abs(odds)


def implied_from_american(odds: float) -> float:
    """Implied probability from American odds (with vig)."""
    return 1.0 / american_to_decimal(odds)


def no_vig_probability(nrfi_odds: float, yrfi_odds: float) -> dict:
    """
    Compute no-vig fair probabilities from a two-way market.
    Returns dict with nrfi_fair, yrfi_fair, and the book's vig%.
    """
    p_nrfi_raw = implied_from_american(nrfi_odds)
    p_yrfi_raw = implied_from_american(yrfi_odds)
    total = p_nrfi_raw + p_yrfi_raw
    return {
        "nrfi_fair": p_nrfi_raw / total,
        "yrfi_fair": p_yrfi_raw / total,
        "vig": total - 1.0,
        "nrfi_raw": p_nrfi_raw,
    }


def compute_edge(model_p: float, market: dict, nrfi_odds: float) -> dict:
    """
    Given the model's P(NRFI), compute edge over the no-vig market price
    and Kelly optimal stake fraction.
    """
    fair = market["nrfi_fair"]
    edge_pp = model_p - fair  # in probability points

    # Kelly: f = (b·p − q) / b  where b = decimal odds − 1, p = model P, q = 1−p
    decimal = american_to_decimal(nrfi_odds)
    b = decimal - 1
    p = model_p
    q = 1 - p
    kelly = (b * p - q) / b if b > 0 else 0
    kelly = max(0.0, kelly)  # never bet a negative Kelly

    return {
        "model_p": round(model_p, 4),
        "market_fair_p": round(fair, 4),
        "edge_pp": round(edge_pp, 4),
        "kelly": round(kelly, 4),
        "decimal_odds": round(decimal, 4),
    }


def load_market_data() -> Dict[str, dict]:
    """
    Load the manual market odds CSV into a dict keyed by game_id.
    Each value contains parsed odds, no-vig probabilities, and book name.
    """
    if not MARKET_PATH.exists():
        return {}
    try:
        df = pd.read_csv(MARKET_PATH, dtype={"game_id": str})
    except Exception:
        return {}
    if df.empty:
        return {}

    out = {}
    for _, row in df.iterrows():
        gid = str(row.get("game_id", "")).strip()
        if not gid:
            continue
        try:
            nrfi_o = float(row["nrfi_american"])
            yrfi_o = float(row["yrfi_american"])
        except (ValueError, KeyError, TypeError):
            continue
        nv = no_vig_probability(nrfi_o, yrfi_o)
        out[gid] = {
            "nrfi_american": nrfi_o,
            "yrfi_american": yrfi_o,
            "book": str(row.get("book", "")) if pd.notna(row.get("book", "")) else "",
            "nrfi_fair": round(nv["nrfi_fair"], 4),
            "yrfi_fair": round(nv["yrfi_fair"], 4),
            "vig": round(nv["vig"], 4),
        }
    return out


def attach_to_game(game_entry: dict, market: dict) -> dict:
    """Attach market data + edge to a game entry, returning the modified dict."""
    gid = game_entry.get("game_id")
    if not gid or gid not in market:
        return game_entry
    m = market[gid]
    if game_entry.get("modeled") and game_entry.get("results"):
        edge = compute_edge(
            model_p=game_entry["results"]["p_nrfi_game"],
            market={"nrfi_fair": m["nrfi_fair"]},
            nrfi_odds=m["nrfi_american"],
        )
        game_entry["market"] = {
            "nrfi_american": m["nrfi_american"],
            "yrfi_american": m["yrfi_american"],
            "book": m["book"],
            "vig": m["vig"],
            "fair_nrfi": m["nrfi_fair"],
            "edge_pp": edge["edge_pp"],
            "kelly": edge["kelly"],
            "decimal_odds": edge["decimal_odds"],
        }
    return game_entry
