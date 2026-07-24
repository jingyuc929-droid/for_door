"""Pure helpers for the cuVAETest performance-driven velocity curriculum.

This module intentionally has no Isaac Lab or torch dependency.  Keeping the
small pieces of schedule/state logic here makes them cheap to unit test without
starting a simulator.
"""

from __future__ import annotations

from collections.abc import Sequence


def level_index(level: float, number_of_ranges: int) -> int:
    """Return a clamped integer schedule index for a (legacy float) level."""
    if number_of_ranges <= 0:
        raise ValueError("A velocity curriculum schedule must not be empty.")
    rounded = round(float(level))
    if abs(float(level) - rounded) > 1.0e-6:
        raise ValueError(f"Velocity curriculum level must be integral, got {level}.")
    return min(max(int(rounded), 0), number_of_ranges - 1)


def range_at_level(
    schedule: Sequence[tuple[float, float]], level: float
) -> tuple[float, float]:
    """Select and validate a two-sided command range from an explicit schedule."""
    index = level_index(level, len(schedule))
    lower, upper = schedule[index]
    lower = float(lower)
    upper = float(upper)
    if lower > upper:
        raise ValueError(
            f"Invalid velocity range at level {index}: lower {lower} > upper {upper}."
        )
    return lower, upper


def threshold_at_level(thresholds: Sequence[float], level: float) -> float:
    """Select a positive MAE threshold using the same level convention."""
    threshold = float(thresholds[level_index(level, len(thresholds))])
    if threshold <= 0.0:
        raise ValueError(f"Velocity MAE threshold must be positive, got {threshold}.")
    return threshold


def dwell_complete(current_step: int, level_enter_step: int, minimum_steps: int) -> bool:
    """Whether an axis has spent the configured number of control steps at its level."""
    if minimum_steps < 0:
        raise ValueError(f"minimum_steps must be non-negative, got {minimum_steps}.")
    return int(current_step) - int(level_enter_step) >= int(minimum_steps)
