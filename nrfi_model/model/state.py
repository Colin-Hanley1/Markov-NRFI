# model/state.py

from dataclasses import dataclass
from typing import Tuple

NUM_BASE_CONFIGS = 8    # 0b000 through 0b111
NUM_OUT_COUNTS   = 3    # 0, 1, 2
NUM_STATES       = NUM_BASE_CONFIGS * NUM_OUT_COUNTS  # 24 transient states

# Absorbing state sentinels (outside 0–23)
NRFI_ABSORBED = 24   # 3 outs reached with 0 runs scored
RUN_ABSORBED  = 25   # at least 1 run scored

STARTING_STATE = 0   # bases empty, 0 outs


def encode_state(bases: int, outs: int) -> int:
    """
    Map (bases bitmask, outs) to a canonical index 0–23.

    Parameters
    ----------
    bases : int
        3-bit integer. Bit 0 = first base, bit 1 = second base, bit 2 = third base.
    outs : int
        Number of outs recorded so far this half-inning. Must be 0, 1, or 2.

    Returns
    -------
    int
        State index in [0, 23].

    Raises
    ------
    ValueError
        If bases is not in [0, 7] or outs is not in [0, 2].
    """
    if not (0 <= bases <= 7):
        raise ValueError(f"bases must be 0–7, got {bases}")
    if outs not in (0, 1, 2):
        raise ValueError(f"outs must be 0, 1, or 2, got {outs}")
    return bases * 3 + outs


def decode_state(index: int) -> Tuple[int, int]:
    """
    Inverse of encode_state.

    Parameters
    ----------
    index : int
        State index in [0, 23].

    Returns
    -------
    Tuple[int, int]
        (bases bitmask, outs).

    Raises
    ------
    ValueError
        If index is outside [0, 23].
    """
    if not (0 <= index <= 23):
        raise ValueError(f"State index must be 0–23, got {index}")
    bases = index // 3
    outs  = index  % 3
    return bases, outs


def on_first(bases: int) -> bool:
    return bool(bases & 0b001)


def on_second(bases: int) -> bool:
    return bool(bases & 0b010)


def on_third(bases: int) -> bool:
    return bool(bases & 0b100)


def runner_count(bases: int) -> int:
    """Number of runners on base (0–3)."""
    return bin(bases).count('1')


def describe_state(index: int) -> str:
    """Human-readable label for a state index, e.g. '1st & 3rd, 1 out'."""
    if index == NRFI_ABSORBED:
        return "3 outs (NRFI)"
    if index == RUN_ABSORBED:
        return "Run scored"
    bases, outs = decode_state(index)
    occupied = []
    if on_first(bases):  occupied.append("1st")
    if on_second(bases): occupied.append("2nd")
    if on_third(bases):  occupied.append("3rd")
    base_str = " & ".join(occupied) if occupied else "empty"
    out_str  = f"{outs} out{'s' if outs != 1 else ''}"
    return f"{base_str}, {out_str}"


# Pre-built lookup: state index → (bases, outs)
STATE_TABLE = [decode_state(i) for i in range(NUM_STATES)]
