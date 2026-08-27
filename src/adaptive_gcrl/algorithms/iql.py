"""Goal-conditioned IQL baseline interface scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_gcrl.training.losses import advantage_weights, expectile_loss, mse_loss


@dataclass(frozen=True)
class IQLConfig:
    discount: float = 0.99
    expectile: float = 0.7
    advantage_temperature: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1].")
        if not 0.0 < self.expectile < 1.0:
            raise ValueError("expectile must be in (0, 1).")
        if self.advantage_temperature <= 0.0:
            raise ValueError("advantage_temperature must be positive.")


@dataclass(frozen=True)
class IQLLossSummary:
    value_loss: float
    critic_loss: float
    actor_weight_mean: float


def compute_iql_losses(q_values: np.ndarray, values: np.ndarray, target_q_values: np.ndarray, config: IQLConfig) -> IQLLossSummary:
    q_array = np.asarray(q_values, dtype=float)
    value_array = np.asarray(values, dtype=float)
    target_array = np.asarray(target_q_values, dtype=float)
    if q_array.shape != value_array.shape or q_array.shape != target_array.shape:
        raise ValueError("q_values, values, and target_q_values must have matching shapes.")

    advantages = q_array - value_array
    return IQLLossSummary(
        value_loss=expectile_loss(advantages, config.expectile),
        critic_loss=mse_loss(q_array, target_array),
        actor_weight_mean=float(np.mean(advantage_weights(advantages, config.advantage_temperature))),
    )
