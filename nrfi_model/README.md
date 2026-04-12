# NRFI Markov Model

A first-inning run-scoring model for MLB using absorbing Markov chains. Predicts the probability of **No Run in the First Inning (NRFI)** for every game on the daily slate.

**[Live site](https://colinhanley.github.io/nrfi-model/)** — updated automatically via GitHub Actions.

---

## How It Works

### The Markov Chain

The first inning is modeled as an **absorbing Markov chain** over **24 transient states** (8 base configurations × 3 out counts) and **2 absorbing states** (3 outs with 0 runs = NRFI, any run scored = RFI).

Each plate appearance maps to a probability distribution over 11 mutually exclusive outcomes (K, BB, HBP, HR, 1B, 2B, 3B, ground ball out, fly ball out, line drive out, fielder's choice), which deterministically transition the chain to a new base-out state via standard baseball advancement rules. Conditional branching handles GIDP and sacrifice flies.

### Rate Blending

Pitcher and batter statistics are combined using the **log-odds (odds-ratio) method**:

```
logit(p_blend) = logit(p_pitcher) + logit(p_batter) - logit(p_league)
```

This produces matchup-specific outcome distributions adjusted for park factors (HR/run environment) and platoon splits (pitcher hand vs batter hand).

### Two Computation Paths

| Method | How | What it captures |
|--------|-----|-----------------|
| **Analytic** | Fundamental matrix N = (I - Q)⁻¹, absorption probs B = N·R | Exact solution under lineup-averaged approximation |
| **Monte Carlo** | 50,000 simulated half-innings with proper batting order | Batter sequencing effects; provides confidence intervals |

The gap between the two quantifies the **batting order effect** — typically 1-4 percentage points.

### Player Profiles

Every player's rates are built from **3 years of MLB counting stats (2024–2026)** pulled from the MLB Stats API. The weighted merge uses plate appearance counts, so early-season noise doesn't overwhelm established priors.

---

## Project Structure

```
nrfi_model/
├── model/                     # Core Markov chain engine
│   ├── state.py               # 24 base-out state definitions
│   ├── outcomes.py            # PA outcome distribution (11 categories)
│   ├── blend.py               # Log-odds blending + park/platoon adjustments
│   ├── transitions.py         # State transition matrix construction
│   └── chain.py               # Analytic solver + Monte Carlo simulation
├── data/                      # Player profiles and park factors
│   ├── pitchers.csv           # ~650 pitcher profiles
│   ├── batters.csv            # ~520 batter profiles
│   ├── park_factors.csv       # HR and run park factors (30 parks)
│   └── league_averages.csv    # MLB-wide average rates
├── docs/                      # Static site (GitHub Pages)
│   ├── index.html             # Frontend with Chart.js + KaTeX
│   └── data/latest.json       # Pre-computed daily predictions
├── compute_daily.py           # Generate predictions for a date
├── build_profiles.py          # Full 3-year profile rebuild (~5 min)
├── update_profiles.py         # Fast daily profile update (~2 min)
├── scrape_data.py             # Single-date data scraper
├── run_validation.py          # Historical backtesting harness
├── app.py                     # Flask app (local development)
├── pipeline.py                # CLI entry point for single games
└── .github/workflows/daily.yml # Automated daily predictions
```

---

## Quickstart

### Install

```bash
pip install numpy pandas scipy flask requests
```

### Generate today's predictions

```bash
python3 compute_daily.py
```

### Preview locally

```bash
cd docs && python3 -m http.server 8080
# Open http://localhost:8080
```

### Run for a specific date

```bash
python3 compute_daily.py 2026-04-12
```

### Run a single game from the CLI

```bash
python3 pipeline.py <game_id> <home_team> <away_team>
```

### Update player profiles

```bash
# Fast daily update (current season only, ~2 min)
python3 update_profiles.py

# Full rebuild from scratch (3 seasons, ~5 min)
python3 build_profiles.py
```

### Run historical validation

```bash
python3 run_validation.py 2026-03-27 2026-04-09
```

---

## Visualizations

The static site includes:

- **Daily summary dashboard** — stat cards, full-slate ranked bar chart
- **Per-game detail overlay** with:
  - P(NRFI) with 95% Wilson confidence interval
  - Per-half breakdown with pitcher stats
  - **Batter NRFI impact** — marginal effect of each batter vs league average
  - **Markov state visit probabilities** — expected visits to each base-out state (fundamental matrix visualization)
  - **Sensitivity analysis** — P(NRFI) response to +1pp changes in K%, BB%, HR%
  - Full lineup tables with per-batter K%, HR%, BB%
- **Methodology page** with state space diagram, KaTeX-rendered formulas, and model explanation

---

## Deployment

The site deploys to GitHub Pages from the `/docs` folder. A GitHub Action (`.github/workflows/daily.yml`) runs twice daily:

1. Updates player profiles with latest stats
2. Fetches lineups from the MLB API (falls back to last game's lineup if not posted)
3. Runs 50,000 Monte Carlo simulations per game
4. Commits the results JSON to the repo

To enable: **Settings > Pages > Deploy from branch `main`, folder `/docs`**.

---

## Data Sources

- **MLB Stats API** — schedules, lineups, player counting stats, boxscores
- **Baseball Savant / Statcast** — batted ball profiles (via pybaseball where available)
- **Park factors** — FanGraphs 3-year regressed averages (static, updated annually)

---

## Model Accuracy

Backtested on 179 games (Mar 27 – Apr 9, 2026):

| Metric | Value |
|--------|-------|
| Brier score | 0.2439 |
| Brier skill score (vs naive) | +1.1% |
| Log loss | 0.681 |
| Accuracy (50% threshold) | 58.1% |
| Actual NRFI rate (sample) | 55.9% |

The model is well-calibrated in the 40–60% prediction range where most games fall.
