"""Diagnostics for goal stitching and temporal-resolution choices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def horizon_histogram(selected_horizons: Sequence[int]) -> dict[int, int]:
    if not selected_horizons:
        raise ValueError("selected_horizons must not be empty.")
    unique, counts = np.unique(np.asarray(selected_horizons, dtype=int), return_counts=True)
    return {int(horizon): int(count) for horizon, count in zip(unique, counts)}


def stitching_success_rate(successes: Sequence[bool], requires_stitching: Sequence[bool]) -> float:
    success_array = np.asarray(successes, dtype=bool)
    stitching_array = np.asarray(requires_stitching, dtype=bool)
    if success_array.shape != stitching_array.shape:
        raise ValueError("successes and requires_stitching must have the same shape.")
    mask = stitching_array
    if not np.any(mask):
        raise ValueError("At least one example must require stitching.")
    return float(np.mean(success_array[mask]))


def success_by_horizon(successes: Sequence[bool], selected_horizons: Sequence[int]) -> dict[int, float]:
    success_array = np.asarray(successes, dtype=bool)
    horizon_array = np.asarray(selected_horizons, dtype=int)
    if success_array.shape != horizon_array.shape:
        raise ValueError("successes and selected_horizons must have the same shape.")
    if success_array.size == 0:
        raise ValueError("successes must not be empty.")
    rates: dict[int, float] = {}
    for horizon in np.unique(horizon_array):
        mask = horizon_array == horizon
        rates[int(horizon)] = float(np.mean(success_array[mask]))
    return rates


def temporal_ablation_gap(adaptive_scores: Sequence[float], fixed_scores: Sequence[float]) -> float:
    adaptive = np.asarray(adaptive_scores, dtype=float)
    fixed = np.asarray(fixed_scores, dtype=float)
    if adaptive.size == 0 or fixed.size == 0:
        raise ValueError("score sequences must not be empty.")
    return float(np.mean(adaptive) - np.mean(fixed))
