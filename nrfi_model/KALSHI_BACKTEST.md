# NRFI Model vs. Kalshi Market — Backtest Results

**Generated:** 2026-07-09
**Sample:** 851 MLB games, 2026-05-05 → 2026-07-08 (99.4% match rate against Kalshi's `KXMLBRFI` settled-market history; 5 games had no listed Kalshi market and were excluded)

## TL;DR

Betting **NRFI** whenever the model disagrees with the market shows **no edge** (flat to slightly negative ROI at every threshold). Betting **YRFI** whenever the model disagrees with the market shows a **real, consistent edge**: +9.0% ROI with no filter, rising to +13–15% ROI once filtered to a 3–8 percentage-point disagreement, with 95% confidence intervals staying entirely above zero through the 5pp bucket (93 bets). This is a promising lead, not a proven strategy — see [Limitations](#limitations).

## Methodology

- **Model predictions:** pulled directly from `docs/data/results_log.json`, which is populated automatically every cycle by `check_results.py` — these are the model's actual *at-the-time* forecasts (`p_nrfi_predicted`), not resimulated after the fact, so there's no lookahead bias.
- **Market prices:** pulled from Kalshi's public `KXMLBRFI` ("First Inning Run?") market via the `/candlesticks` endpoint, using the hourly candle closest to **60 minutes before the game's actual first-pitch time** (fetched per-game from the MLB Stats API) — this simulates a realistic pregame entry price, not the post-resolution settled price.
- **Game matching:** each `results_log` entry is matched to a Kalshi market by requiring both team names to appear in the market's title *and* picking the closest `occurrence_datetime` to the real game start time (within a 4-hour tolerance). Team-pair-only / date-window matching was tried first and produced a 66% false-ambiguous rate, because the same two teams routinely play on consecutive days (a normal series) — using the actual per-game start time fixed this.
- **Fill price:** all P&L uses the **ask price** (the actual cost of crossing the spread to enter a position), not the midpoint — this is a realistic cost basis, not an optimistic one.
- **Reproduce:** `python3 backtest_kalshi.py` (joins model + market data, caches to `data/kalshi_backtest_joined.csv`) then `python3 analyze_kalshi_strategy.py` (evaluates strategies, writes `data/kalshi_strategy_report.json`). Re-running only fetches new games, so the sample grows automatically as more weeks pass.

## Calibration: model vs. market vs. reality

| | Mean predicted P(NRFI) | Brier score (lower is better) |
|---|---|---|
| **Model** | 53.4% | 0.2515 |
| **Kalshi market** | 50.6% | 0.2483 |
| **Actual NRFI rate** | 47.1% | — |

The model is **systematically overconfident on NRFI** — it predicts NRFI about 6.2 points more often than it actually happens, while Kalshi's market average sits much closer to the true rate. On raw calibration, **the market slightly out-forecasts the model overall** (lower Brier score). This is consistent with the strategy results below: the model has no edge on the side where it's biased (NRFI), but retains real signal on the side where its disagreements with the market are informative (YRFI).

## Strategy backtest: flat $1-notional bets by edge threshold

`edge_pp = model P(NRFI) − Kalshi-implied P(NRFI)`. "Bet NRFI" fires when `edge_pp ≥ threshold`; "Bet YRFI" fires when `edge_pp ≤ −threshold`.

| Threshold | Side | Bets | Win rate | Total profit | ROI | 95% CI on mean return |
|---:|:---|---:|---:|---:|---:|:---|
| 0pp | NRFI | 574 | 50.0% | −$5.39 | −0.94% | [−5.02%, +3.14%] |
| 0pp | **YRFI** | 281 | **58.4%** | **+$25.25** | **+8.99%** | **[+3.27%, +14.70%]** |
| 3pp | NRFI | 406 | 49.3% | −$6.70 | −1.65% | [−6.49%, +3.19%] |
| 3pp | **YRFI** | 154 | **63.0%** | **+$21.27** | **+13.81%** | **[+6.32%, +21.31%]** |
| 5pp | NRFI | 298 | 48.3% | −$6.45 | −2.16% | [−7.80%, +3.47%] |
| 5pp | **YRFI** | 93 | **64.5%** | **+$14.00** | **+15.05%** | **[+5.58%, +24.53%]** |
| 8pp | NRFI | 172 | 45.4% | −$8.21 | −4.77% | [−12.18%, +2.64%] |
| 8pp | YRFI | 34 | 64.7% | +$5.01 | +14.74% | [−0.92%, +30.39%] |
| 12pp | NRFI | 63 | 47.6% | −$1.24 | −1.97% | [−14.10%, +10.17%] |
| 12pp | YRFI | 7 | 71.4% | +$1.45 | +20.71% | [−14.78%, +56.21%] |
| 15pp | NRFI | 24 | 41.7% | −$1.38 | −5.75% | [−25.39%, +13.89%] |
| 15pp | YRFI | 2 | 50.0% | −$0.04 | −2.00% | wide, n too small |
| 20pp | NRFI | 8 | 50.0% | +$0.15 | +1.87% | wide, n too small |
| 20pp | YRFI | 1 | 0.0% | −$0.55 | −55.00% | n=1, ignore |

**Bold** rows are where the 95% CI sits entirely above zero — the only region showing a statistically meaningful edge in this sample.

## Trend over time

| Month | Games | Mean edge (model − market, pp) |
|---|---:|---:|
| 2026-05 | 359 | +1.07 |
| 2026-06 | 392 | +3.85 |
| 2026-07 | 100 | +4.60 |

The average gap between model and market has been widening month over month. Worth watching — it's unclear yet whether this reflects the market getting sharper, the model drifting, or just short-window noise; more data will clarify.

## Limitations

- **Sample size:** 851 games over ~9 weeks is a modest sample for a binary outcome this noisy. The higher-threshold buckets (8pp+) have under 40 bets and correspondingly wide confidence intervals — don't read too much into any single high-threshold number.
- **No out-of-sample holdout:** every result above is in-sample. The right next step before trusting this is to let a few more weeks accumulate and check whether the YRFI edge holds up on data the strategy wasn't "tuned" on (note: no explicit tuning was done — the threshold grid was fixed in advance — but the general risk of overfitting to a 9-week window still applies).
- **Fees and slippage not modeled:** P&L uses ask-price fills (realistic spread cost) but does not subtract Kalshi's per-contract trading fee, and does not account for slippage on the thinner markets (some `KXMLBRFI` markets showed near-zero live liquidity when checked). Real-world returns would run somewhat below these numbers.
- **Single-book:** this only checks Kalshi's own market; no comparison against sportsbook NRFI/YRFI lines.
