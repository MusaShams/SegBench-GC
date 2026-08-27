"""Reliability-oriented aggregate metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def interquartile_mean(scores: Sequence[float]) -> float:
    values = np.sort(np.asarray(scores, dtype=float))
    if values.size == 0:
        raise ValueError("scores must not be empty.")
    lower = int(math.floor(0.25 * values.size))
    upper = int(math.ceil(0.75 * values.size))
    trimmed = values[lower:upper]
    if trimmed.size == 0:
        trimmed = values
    return float(np.mean(trimmed))


def bootstrap_ci(scores: Sequence[float], *, samples: int = 2000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("scores must not be empty.")
    if samples <= 0:
        raise ValueError("samples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for idx in range(samples):
        resample = rng.choice(values, size=values.size, replace=True)
        estimates[idx] = interquartile_mean(resample)

    alpha = 1.0 - confidence
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def aggregate_scores(scores: Sequence[float]) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("scores must not be empty.")
    ci_low, ci_high = bootstrap_ci(values)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "iqm": interquartile_mean(values),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def performance_profile(scores: Sequence[float], thresholds: Sequence[float]) -> dict[float, float]:
    values = np.asarray(scores, dtype=float)
    threshold_array = np.asarray(thresholds, dtype=float)
    if values.size == 0:
        raise ValueError("scores must not be empty.")
    if threshold_array.size == 0:
        raise ValueError("thresholds must not be empty.")
    return {float(threshold): float(np.mean(values >= threshold)) for threshold in threshold_array}
