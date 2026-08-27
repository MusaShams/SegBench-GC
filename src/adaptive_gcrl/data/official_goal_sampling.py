"""Dynamic goal sampling aligned with the official OGBench GCDataset protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from adaptive_gcrl.data.replay_buffer import HorizonTargets, TransitionBatch


@dataclass(frozen=True)
class OfficialGoalSamplingConfig:
    discount: float = 0.99
    value_p_curgoal: float = 0.2
    value_p_trajgoal: float = 0.5
    value_p_randomgoal: float = 0.3
    value_geom_sample: bool = True
    actor_p_curgoal: float = 0.0
    actor_p_trajgoal: float = 0.5
    actor_p_randomgoal: float = 0.5
    actor_geom_sample: bool = False
    gc_negative: bool = True
    horizons: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.discount < 1.0:
            raise ValueError("discount must be in (0, 1).")
        value_sum = self.value_p_curgoal + self.value_p_trajgoal + self.value_p_randomgoal
        actor_sum = self.actor_p_curgoal + self.actor_p_trajgoal + self.actor_p_randomgoal
        if not np.isclose(value_sum, 1.0):
            raise ValueError("Value goal probabilities must sum to one.")
        if not np.isclose(actor_sum, 1.0):
            raise ValueError("Actor goal probabilities must sum to one.")
        if self.horizons is not None and (
            not self.horizons or any(horizon <= 0 for horizon in self.horizons)
        ):
            raise ValueError("horizons must be positive when provided.")


class OfficialGoalReplayBuffer:
    def __init__(
        self,
        batch: TransitionBatch,
        config: OfficialGoalSamplingConfig,
        *,
        backup_boundaries: Optional[np.ndarray] = None,
        bootstrap_at_boundaries: bool = True,
        boundary_continuations: Optional[np.ndarray] = None,
    ) -> None:
        if batch.size == 0:
            raise ValueError("Replay buffer requires transitions.")
        terminal_locs = np.flatnonzero(np.asarray(batch.terminals) > 0)
        if terminal_locs.size == 0 or terminal_locs[-1] != batch.size - 1:
            raise ValueError("Ordered source data must end with a terminal transition.")
        self._batch = batch
        self.config = config
        self.terminal_locs = terminal_locs
        if backup_boundaries is None:
            self.backup_boundary_locs = terminal_locs
        else:
            boundary_array = np.asarray(backup_boundaries, dtype=bool)
            if boundary_array.shape != (batch.size,):
                raise ValueError(
                    f"backup_boundaries must have shape ({batch.size},)."
                )
            self.backup_boundary_locs = np.flatnonzero(boundary_array)
            if (
                self.backup_boundary_locs.size == 0
                or self.backup_boundary_locs[-1] != batch.size - 1
            ):
                raise ValueError(
                    "backup_boundaries must include the final transition."
                )
        self.bootstrap_at_boundaries = bootstrap_at_boundaries
        backup_boundary_mask = np.zeros(batch.size, dtype=bool)
        backup_boundary_mask[self.backup_boundary_locs] = True
        if boundary_continuations is None:
            self.boundary_continuations = backup_boundary_mask
        else:
            continuation_array = np.asarray(
                boundary_continuations,
                dtype=bool,
            )
            if continuation_array.shape != (batch.size,):
                raise ValueError(
                    f"boundary_continuations must have shape ({batch.size},)."
                )
            if np.any(continuation_array & ~backup_boundary_mask):
                raise ValueError(
                    "boundary_continuations may only mark backup boundaries."
                )
            self.boundary_continuations = continuation_array

    @property
    def size(self) -> int:
        return self._batch.size

    def _final_state_indices(self, indices: np.ndarray) -> np.ndarray:
        return self.terminal_locs[np.searchsorted(self.terminal_locs, indices)]

    def _backup_final_indices(self, indices: np.ndarray) -> np.ndarray:
        return self.backup_boundary_locs[
            np.searchsorted(self.backup_boundary_locs, indices)
        ]

    def _sample_goals(
        self,
        indices: np.ndarray,
        *,
        p_curgoal: float,
        p_trajgoal: float,
        geom_sample: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        random_goal_indices = rng.integers(0, self.size, size=len(indices))
        final_state_indices = self._final_state_indices(indices)
        if geom_sample:
            offsets = rng.geometric(p=1.0 - self.config.discount, size=len(indices))
            trajectory_goal_indices = np.minimum(indices + offsets, final_state_indices)
        else:
            distances = rng.random(len(indices))
            trajectory_goal_indices = np.round(
                np.minimum(indices + 1, final_state_indices) * distances
                + final_state_indices * (1.0 - distances)
            ).astype(int)
        if p_curgoal == 1.0:
            return indices.copy()
        choose_trajectory = rng.random(len(indices)) < p_trajgoal / (1.0 - p_curgoal)
        goal_indices = np.where(
            choose_trajectory,
            trajectory_goal_indices,
            random_goal_indices,
        )
        choose_current = rng.random(len(indices)) < p_curgoal
        return np.where(choose_current, indices, goal_indices)

    def _horizon_targets(
        self,
        indices: np.ndarray,
        value_goal_indices: np.ndarray,
    ) -> Optional[HorizonTargets]:
        if self.config.horizons is None:
            return None
        horizons = self.config.horizons
        batch_size = len(indices)
        returns = np.zeros((batch_size, len(horizons)), dtype=np.float32)
        discounts = np.zeros((batch_size, len(horizons)), dtype=np.float32)
        effective_steps = np.zeros(
            (batch_size, len(horizons)),
            dtype=np.int32,
        )
        next_observations = np.repeat(
            self._batch.next_observations[indices, None, :],
            len(horizons),
            axis=1,
        ).astype(np.float32)
        final_indices = self._backup_final_indices(indices)
        negative_offset = 1.0 if self.config.gc_negative else 0.0

        for horizon_index, horizon in enumerate(horizons):
            active = np.ones(batch_size, dtype=bool)
            scale = 1.0
            for offset in range(horizon):
                current_indices = indices + offset
                valid = active & (current_indices <= final_indices)
                successes = current_indices == value_goal_indices
                rewards = successes.astype(np.float32) - negative_offset
                returns[:, horizon_index] += scale * rewards * valid
                effective_steps[valid, horizon_index] = offset + 1
                if np.any(valid):
                    next_observations[valid, horizon_index] = self._batch.next_observations[
                        current_indices[valid]
                    ]
                boundaries = current_indices == final_indices
                continuations = self.boundary_continuations[
                    current_indices.clip(max=self.size - 1)
                ]
                boundary_bootstraps = (
                    valid
                    & ~successes
                    & boundaries
                    & continuations
                )
                if self.bootstrap_at_boundaries:
                    discounts[boundary_bootstraps, horizon_index] = (
                        self.config.discount ** (offset + 1)
                    )
                active = valid & ~successes & ~boundaries
                scale *= self.config.discount
            discounts[active, horizon_index] = self.config.discount**horizon

        return HorizonTargets(
            horizons=horizons,
            returns=returns,
            next_observations=next_observations,
            discounts=discounts,
            effective_steps=effective_steps,
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> TransitionBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        indices = rng.integers(0, self.size, size=batch_size)
        value_goal_indices = self._sample_goals(
            indices,
            p_curgoal=self.config.value_p_curgoal,
            p_trajgoal=self.config.value_p_trajgoal,
            geom_sample=self.config.value_geom_sample,
            rng=rng,
        )
        actor_goal_indices = self._sample_goals(
            indices,
            p_curgoal=self.config.actor_p_curgoal,
            p_trajgoal=self.config.actor_p_trajgoal,
            geom_sample=self.config.actor_geom_sample,
            rng=rng,
        )
        successes = indices == value_goal_indices
        rewards = successes.astype(np.float32) - (
            1.0 if self.config.gc_negative else 0.0
        )
        return TransitionBatch(
            observations=self._batch.observations[indices],
            actions=self._batch.actions[indices],
            rewards=rewards,
            next_observations=self._batch.next_observations[indices],
            terminals=self._batch.terminals[indices],
            goals=self._batch.observations[value_goal_indices],
            actor_goals=self._batch.observations[actor_goal_indices],
            masks=1.0 - successes.astype(np.float32),
            horizon_targets=self._horizon_targets(indices, value_goal_indices),
        )
