# NRFI Model Improvement Roadmap

**Generated:** 2026-07-09
**Scope:** prioritized improvements that require new data plumbing, longer validation windows, or both. The immediate zero-new-data fixes are handled separately by small-sample shrinkage and Platt calibration.

## TL;DR

The best near-term additions are features that are free to collect and plausibly underpriced by the market: weather first, then starter-role uncertainty. Rest days and umpire tendencies are attainable but need careful validation before they should affect betting decisions. Statcast-derived batted-ball quality is the biggest modeling upgrade, but also the largest integration project.

## 1. Weather: temperature and wind

**What:** Add dynamic per-game weather adjustments using a static ballpark latitude/longitude table plus Open-Meteo forecast and historical weather data. Start with temperature, wind speed, and wind direction relative to park orientation.

**Why attainable:** Open-Meteo is free and does not require an API key. The only static data needed is a one-time park location/orientation table, and the existing daily compute path already has game date, time, and home park context.

**Expected impact:** High. Weather changes daily and can affect first-inning run environment in a way a static annual park-factor table cannot. It is the strongest candidate for genuinely new signal that may not be fully priced game by game.

**Rough effort:** Medium. Add park metadata, fetch/cache weather, define first-pass adjustment, then validate in the backtest harness before making it betting-facing.

## 2. Opener / thin-sample-starter detection

**What:** Flag probable pitchers with unusually limited starter-role track record using existing `pa` and games-started style data, then apply extra shrinkage or expose a "thin sample" warning in output.

**Why attainable:** This can begin with data already present in `data/pitchers.csv` and the MLB profile/schedule data already fetched by the project. No paid or scraped source is required.

**Expected impact:** Medium to high. The model currently treats a low-PA pitcher profile too much like an established starter profile. Extra starter-role uncertainty should reduce false confidence around openers, bullpen games, and recent call-ups.

**Rough effort:** Small to medium. A first version can be a deterministic flag plus stronger shrinkage; a better version should collect role history and validate thresholds against `backtest_kalshi.py`.

## 3. Rest days / getaway-day features

**What:** Derive team rest, travel-like schedule compression, day-after-night, and getaway-day indicators from schedule data already fetched by the pipeline.

**Why attainable:** MLB schedule endpoints are already in use. These features can be computed without a new provider.

**Expected impact:** Low to medium. The signal is plausible but easier to overfit than weather, so it should stay out of the betting display until the existing backtest harness shows stable out-of-sample value.

**Rough effort:** Small. The larger work is validation and deciding whether the feature belongs as a model adjustment, confidence flag, or no-op.

## 4. Self-derived umpire tendency tracking

**What:** Collect plate umpire assignments from MLB API schedule data with `hydrate=officials`, then accumulate first-inning K/BB/run-environment tendencies by umpire over time.

**Why attainable:** The data is available from MLB's free API. This avoids depending on external umpire-scorecard scraping and creates a project-owned historical table.

**Expected impact:** Medium long-term, low short-term. Umpire samples are thin, so the first season of collection is more useful for building history than immediate betting changes.

**Rough effort:** Medium. Add collection/storage now, but require at least a season-scale sample and heavy shrinkage before using it in forecasts.

## 5. pybaseball / Statcast integration

**What:** Use Statcast batted-ball quality such as barrel rate, exit velocity, launch angle, and expected stats to improve HR and extra-base-hit projections beyond season-aggregate outcomes.

**Why attainable:** `README.md` already mentions pybaseball/Statcast as a data source, but the codebase does not currently import it. The data is realistic to obtain, though it adds dependency and caching complexity.

**Expected impact:** Medium to high if validated. Better contact-quality inputs could improve the model's run-scoring tail estimates, especially for HR risk in the first inning.

**Rough effort:** Large. Requires dependency work, data caching, feature design, shrinkage, and careful backtesting to avoid adding noisy complexity.

## Maintenance Notes

- Re-run `python3 fit_calibration.py` periodically, such as weekly, until it is worth automating in the daily pipeline.
- Treat every new adjustment as disabled-by-default until it improves held-out Brier/log-loss or Kalshi strategy results.
- Keep raw model probability, calibrated probability, and market probability visible separately so future regressions are easy to diagnose.
