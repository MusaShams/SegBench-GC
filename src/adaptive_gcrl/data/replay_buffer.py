"""Array-backed replay data containers used by offline algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class HorizonTargets:
    horizons: tuple[int, ...]
    returns: np.ndarray
    next_observations: np.ndarray
    discounts: np.ndarray
    effective_steps: np.ndarray

    def __post_init__(self) -> None:
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("Horizon targets require positive horizons.")
        if self.returns.ndim != 2:
            raise ValueError("Horizon returns must have shape (batch, horizons).")
        expected = (self.returns.shape[0], len(self.horizons))
        if self.returns.shape != expected:
            raise ValueError(f"Horizon returns must have shape {expected}.")
        if self.discounts.shape != expected:
            raise ValueError(f"Horizon discounts must have shape {expected}.")
        if self.next_observations.shape[:2] != expected:
            raise ValueError(f"Horizon next observations must start with shape {expected}.")
        if self.effective_steps.shape != expected:
            raise ValueError(f"Horizon effective steps must have shape {expected}.")
        if np.any(self.effective_steps < 0):
            raise ValueError("Horizon effective steps must be non-negative.")
        horizon_array = np.asarray(self.horizons)[None, :]
        if np.any(self.effective_steps > horizon_array):
            raise ValueError("Horizon effective steps cannot exceed their horizons.")

    def sample(self, indices: np.ndarray) -> "HorizonTargets":
        return HorizonTargets(
            horizons=self.horizons,
            returns=self.returns[indices],
            next_observations=self.next_observations[indices],
            discounts=self.discounts[indices],
            effective_steps=self.effective_steps[indices],
        )


@dataclass(frozen=True)
class TransitionBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    goals: Optional[np.ndarray] = None
    actor_goals: Optional[np.ndarray] = None
    masks: Optional[np.ndarray] = None
    horizon_targets: Optional[HorizonTargets] = None

    def __post_init__(self) -> None:
        size = self.observations.shape[0]
        fields = {
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "terminals": self.terminals,
        }
        if self.goals is not None:
            fields["goals"] = self.goals
        if self.actor_goals is not None:
            fields["actor_goals"] = self.actor_goals
        if self.masks is not None:
            fields["masks"] = self.masks
        if self.horizon_targets is not None and self.horizon_targets.returns.shape[0] != size:
            raise ValueError(
                f"horizon_targets has batch dimension {self.horizon_targets.returns.shape[0]}, expected {size}."
            )
        for name, value in fields.items():
            if value.shape[0] != size:
                raise ValueError(f"{name} has batch dimension {value.shape[0]}, expected {size}.")

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])


class ReplayBuffer:
    def __init__(self, batch: TransitionBatch) -> None:
        if batch.size == 0:
            raise ValueError("ReplayBuffer requires at least one transition.")
        self._batch = batch

    @property
    def size(self) -> int:
        return self._batch.size

    def sample(self, batch_size: int, rng: np.random.Generator) -> TransitionBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        indices = rng.integers(0, self.size, size=batch_size)
        return TransitionBatch(
            observations=self._batch.observations[indices],
            actions=self._batch.actions[indices],
            rewards=self._batch.rewards[indices],
            next_observations=self._batch.next_observations[indices],
            terminals=self._batch.terminals[indices],
            goals=None if self._batch.goals is None else self._batch.goals[indices],
            actor_goals=None
            if self._batch.actor_goals is None
            else self._batch.actor_goals[indices],
            masks=None if self._batch.masks is None else self._batch.masks[indices],
            horizon_targets=None
            if self._batch.horizon_targets is None
            else self._batch.horizon_targets.sample(indices),
        )
