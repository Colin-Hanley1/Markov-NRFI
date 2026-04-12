# model/chain.py

from __future__ import annotations
import numpy as np
from typing import List, Tuple

from .state import (
    NUM_STATES, STARTING_STATE, NRFI_ABSORBED, RUN_ABSORBED,
    decode_state,
)
from .outcomes import PAOutcomeRates
from .transitions import build_transition_row, build_full_transition_matrix


# ---------------------------------------------------------------------------
# Analytic path
# ---------------------------------------------------------------------------

def compute_nrfi_analytic(
    lineup_rates: List[PAOutcomeRates],
) -> Tuple[float, np.ndarray]:
    """
    Compute P(NRFI) analytically using the fundamental matrix.

    Uses a stationary (lineup-averaged) approximation of batter order.

    Parameters
    ----------
    lineup_rates : list of PAOutcomeRates

    Returns
    -------
    p_nrfi : float
        Probability of no run scored in the half-inning.
    absorption_probs : np.ndarray, shape (24, 2)
        For each starting state: [P(NRFI_absorbed), P(RUN_absorbed)].
    """
    T = build_full_transition_matrix(lineup_rates)

    Q = T[:NUM_STATES, :NUM_STATES]      # 24×24 transient block
    R = T[:NUM_STATES, NUM_STATES:]      # 24×2 absorption block

    I = np.eye(NUM_STATES, dtype=np.float64)

    # Fundamental matrix: N[i,j] = expected number of times in state j given start i
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Fundamental matrix inversion failed: {e}. "
            "Check that transition rows sum to 1 and Q has spectral radius < 1."
        )

    B = N @ R   # shape (24, 2)  — absorption probabilities

    # Validate: each row must sum to ~1
    row_sums = B.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        bad = np.where(np.abs(row_sums - 1.0) > 1e-4)[0]
        raise RuntimeError(
            f"Absorption probability rows do not sum to 1 for states: {bad}. "
            f"Row sums: {row_sums[bad]}"
        )

    p_nrfi = float(B[STARTING_STATE, 0])
    return p_nrfi, B


def compute_half_inning_detail(
    lineup_rates: List[PAOutcomeRates],
) -> dict:
    """
    Compute rich model details for a half-inning: transition matrix,
    fundamental matrix, state visit probabilities, and expected PA.

    Returns a dict with all model artifacts for visualization.
    """
    from .state import describe_state, STARTING_STATE

    T = build_full_transition_matrix(lineup_rates)
    Q = T[:NUM_STATES, :NUM_STATES]
    R = T[:NUM_STATES, NUM_STATES:]
    I = np.eye(NUM_STATES, dtype=np.float64)
    N = np.linalg.inv(I - Q)
    B = N @ R

    # State visit probabilities from starting state
    state_visits = N[STARTING_STATE, :]  # expected visits to each state

    # Expected total PA (analytic)
    expected_pa = float(state_visits.sum())

    # State labels
    state_labels = [describe_state(i) for i in range(NUM_STATES)]

    # Absorption probabilities from starting state
    p_nrfi = float(B[STARTING_STATE, 0])
    p_run = float(B[STARTING_STATE, 1])

    # Transition matrix rounded for JSON (full 24x26)
    T_rounded = np.round(T, 5).tolist()

    # State visits rounded
    visits_rounded = [round(float(v), 5) for v in state_visits]

    # Fundamental matrix diagonal (expected self-visits)
    N_diag = [round(float(N[i, i]), 5) for i in range(NUM_STATES)]

    return {
        "transition_matrix": T_rounded,
        "state_visit_probs": visits_rounded,
        "fundamental_diag": N_diag,
        "expected_pa": round(expected_pa, 3),
        "p_nrfi": round(p_nrfi, 5),
        "p_run": round(p_run, 5),
        "state_labels": state_labels,
    }


# ---------------------------------------------------------------------------
# Simulation path
# ---------------------------------------------------------------------------

def simulate_half_inning(
    lineup_rates: List[PAOutcomeRates],
    batter_start: int,
    rng:          np.random.Generator,
) -> Tuple[bool, int]:
    """
    Simulate a single half-inning.

    Parameters
    ----------
    lineup_rates : list of PAOutcomeRates
        Length 9. Index = batting order slot (0-based).
    batter_start : int
        Index into lineup_rates for the leadoff batter (0 = top of order).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    nrfi : bool
        True if the inning ended without a run scoring.
    batters_faced : int
        Number of PAs in this half-inning.
    """
    state = STARTING_STATE
    batter_slot = batter_start % 9
    batters_faced = 0
    max_pa = 27  # safety cap — no inning should ever need more than this

    for _ in range(max_pa):
        rates = lineup_rates[batter_slot]
        row, _ = build_transition_row(state, rates)

        # Draw next state
        next_state = int(rng.choice(len(row), p=row))
        batters_faced += 1
        batter_slot = (batter_slot + 1) % 9

        if next_state == NRFI_ABSORBED:
            return True, batters_faced
        if next_state == RUN_ABSORBED:
            return False, batters_faced

        state = next_state

    # Should not reach here
    raise RuntimeError(
        f"Half-inning simulation exceeded {max_pa} PAs without absorption. "
        f"Final state: {state}. Check transition rows for missing absorption."
    )


def simulate_half_inning_traced(
    lineup_rates: List[PAOutcomeRates],
    batter_start: int,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate a single half-inning and return the full state trace.
    """
    from .state import describe_state
    state = STARTING_STATE
    batter_slot = batter_start % 9
    trace = [{"state": state, "label": describe_state(state)}]

    for _ in range(27):
        rates = lineup_rates[batter_slot]
        row, _ = build_transition_row(state, rates)
        next_state = int(rng.choice(len(row), p=row))
        batter_slot = (batter_slot + 1) % 9

        if next_state == NRFI_ABSORBED:
            trace.append({"state": NRFI_ABSORBED, "label": "3 outs (NRFI)"})
            return {"nrfi": True, "pa": len(trace) - 1, "trace": trace}
        if next_state == RUN_ABSORBED:
            trace.append({"state": RUN_ABSORBED, "label": "Run scored"})
            return {"nrfi": False, "pa": len(trace) - 1, "trace": trace}

        trace.append({"state": next_state, "label": describe_state(next_state)})
        state = next_state

    trace.append({"state": -1, "label": "Max PA reached"})
    return {"nrfi": False, "pa": 27, "trace": trace}


def simulate_full_inning_traced(
    away_lineup: List[PAOutcomeRates],
    home_lineup: List[PAOutcomeRates],
    rng: np.random.Generator,
) -> dict:
    """
    Simulate a full first inning (top + bottom) and return traces for both halves.
    """
    top = simulate_half_inning_traced(away_lineup, 0, rng)
    # If a run scored in the top half, it's already RFI — but we still
    # play out the bottom half for visualization purposes
    bot = simulate_half_inning_traced(home_lineup, 0, rng)

    nrfi = top["nrfi"] and bot["nrfi"]
    return {
        "nrfi": nrfi,
        "top": {"nrfi": top["nrfi"], "pa": top["pa"], "trace": top["trace"]},
        "bottom": {"nrfi": bot["nrfi"], "pa": bot["pa"], "trace": bot["trace"]},
    }


def simulate_with_details(
    lineup_rates: List[PAOutcomeRates],
    n_simulations: int = 50_000,
    seed: int = 42,
    n_sample_traces: int = 20,
) -> dict:
    """
    Run simulations and collect rich detail: PA distribution,
    convergence curve, and sample inning traces for visualization.
    """
    rng = np.random.default_rng(seed)

    nrfi_count = 0
    pa_counts = []
    convergence = []  # running P(NRFI) at checkpoints

    # Collect sample traces from first N innings
    sample_traces = []

    for i in range(n_simulations):
        if i < n_sample_traces:
            result = simulate_half_inning_traced(lineup_rates, 0, rng)
            sample_traces.append(result)
            nrfi = result["nrfi"]
            pa = result["pa"]
        else:
            nrfi, pa = simulate_half_inning(lineup_rates, 0, rng)

        if nrfi:
            nrfi_count += 1
        pa_counts.append(pa)

        # Log convergence at checkpoints
        if (i + 1) in (100, 500, 1000, 2000, 5000, 10000, 20000, 50000) and (i + 1) <= n_simulations:
            convergence.append({
                "n": i + 1,
                "p_nrfi": round(nrfi_count / (i + 1), 4),
            })

    # PA distribution
    pa_arr = np.array(pa_counts)
    max_pa = int(pa_arr.max())
    pa_dist = [0] * (max_pa + 1)
    for p in pa_counts:
        pa_dist[p] += 1
    pa_dist = [round(c / n_simulations, 5) for c in pa_dist]

    return {
        "p_nrfi": round(nrfi_count / n_simulations, 4),
        "pa_distribution": pa_dist,
        "pa_mean": round(float(pa_arr.mean()), 2),
        "pa_median": int(np.median(pa_arr)),
        "convergence": convergence,
        "sample_traces": [
            {"nrfi": t["nrfi"], "pa": t["pa"], "trace": t["trace"]}
            for t in sample_traces
        ],
    }


def simulate_nrfi(
    home_lineup: List[PAOutcomeRates],
    away_lineup: List[PAOutcomeRates],
    n_simulations: int = 100_000,
    seed:          int  = 42,
) -> dict:
    """
    Simulate full game first-inning NRFI across N trials.

    Parameters
    ----------
    home_lineup : list of PAOutcomeRates
        Rates for the home team batting order (bottom of 1st).
    away_lineup : list of PAOutcomeRates
        Rates for the away team batting order (top of 1st).
    n_simulations : int
        Number of innings to simulate per half-inning.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        p_nrfi_away  : float   — P(no run in top of 1st)
        p_nrfi_home  : float   — P(no run in bottom of 1st)
        p_nrfi_game  : float   — P(no run in either half)
        p_nrfi_ci    : tuple   — (lower, upper) 95% Wilson CI for game NRFI
        avg_pa_away  : float   — average PAs per top-of-1st inning
        avg_pa_home  : float   — average PAs per bottom-of-1st inning
    """
    rng = np.random.default_rng(seed)

    away_results = []
    away_pa      = []
    for _ in range(n_simulations):
        nrfi, pa = simulate_half_inning(away_lineup, batter_start=0, rng=rng)
        away_results.append(nrfi)
        away_pa.append(pa)

    home_results = []
    home_pa      = []
    for _ in range(n_simulations):
        nrfi, pa = simulate_half_inning(home_lineup, batter_start=0, rng=rng)
        home_results.append(nrfi)
        home_pa.append(pa)

    p_away = float(np.mean(away_results))
    p_home = float(np.mean(home_results))
    p_game = p_away * p_home

    # Wilson confidence interval for game-level NRFI
    game_results = [a and h for a, h in zip(away_results, home_results)]
    p_game_empirical = float(np.mean(game_results))
    ci = _wilson_ci(sum(game_results), n_simulations)

    return {
        "p_nrfi_away":      p_away,
        "p_nrfi_home":      p_home,
        "p_nrfi_game":      p_game_empirical,
        "p_nrfi_game_indep": p_game,
        "p_nrfi_ci":        ci,
        "avg_pa_away":      float(np.mean(away_pa)),
        "avg_pa_home":      float(np.mean(home_pa)),
        "n_simulations":    n_simulations,
    }


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))
