"""Small numerical losses used by tests and algorithm scaffolds."""

from __future__ import annotations

import numpy as np


def expectile_loss(residuals: np.ndarray, expectile: float) -> float:
    if not 0.0 < expectile < 1.0:
        raise ValueError("expectile must be in (0, 1).")
    residual_array = np.asarray(residuals, dtype=float)
    weights = np.where(residual_array >= 0.0, expectile, 1.0 - expectile)
    return float(np.mean(weights * residual_array**2))


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    prediction_array = np.asarray(predictions, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    if prediction_array.shape != target_array.shape:
        raise ValueError("predictions and targets must have the same shape.")
    return float(np.mean((prediction_array - target_array) ** 2))


def advantage_weights(advantages: np.ndarray, temperature: float, clip: float = 100.0) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if clip <= 0.0:
        raise ValueError("clip must be positive.")
    scaled = np.asarray(advantages, dtype=float) * temperature
    return np.minimum(np.exp(scaled), clip)


def multi_step_returns(rewards: np.ndarray, terminals: np.ndarray, horizons: tuple[int, ...], discount: float) -> np.ndarray:
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1].")
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive.")
    reward_array = np.asarray(rewards, dtype=float).reshape(-1)
    terminal_array = np.asarray(terminals, dtype=bool).reshape(-1)
    if reward_array.shape != terminal_array.shape:
        raise ValueError("rewards and terminals must have the same shape.")

    targets = np.zeros((reward_array.shape[0], len(horizons)), dtype=float)
    for index in range(reward_array.shape[0]):
        for horizon_index, horizon in enumerate(horizons):
            total = 0.0
            scale = 1.0
            for offset in range(horizon):
                step = index + offset
                if step >= reward_array.shape[0]:
                    break
                total += scale * reward_array[step]
                if terminal_array[step]:
                    break
                scale *= discount
            targets[index, horizon_index] = total
    return targets
