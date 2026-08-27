"""Trajectory-aware multi-horizon targets for goal-conditioned offline RL."""

from __future__ import annotations

import numpy as np

from adaptive_gcrl.data.offline_dataset import goal_relabel_rewards
from adaptive_gcrl.data.replay_buffer import HorizonTargets, TransitionBatch


def compute_goal_horizon_targets(
    batch: TransitionBatch,
    horizons: tuple[int, ...],
    *,
    discount: float,
    reward_mode: str,
    success_threshold: float,
) -> HorizonTargets:
    if batch.goals is None:
        raise ValueError("Goal-conditioned horizon targets require goals.")
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive.")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1].")

    size = batch.size
    returns = np.zeros((size, len(horizons)), dtype=np.float32)
    discounts = np.zeros((size, len(horizons)), dtype=np.float32)
    effective_steps = np.zeros((size, len(horizons)), dtype=np.int32)
    next_observations = np.repeat(
        np.asarray(batch.next_observations[:, None, :], dtype=np.float32),
        len(horizons),
        axis=1,
    )
    terminals = np.asarray(batch.terminals, dtype=bool)

    for horizon_index, horizon in enumerate(horizons):
        active = np.ones(size, dtype=bool)
        scale = 1.0
        for offset in range(horizon):
            valid_count = size - offset
            if valid_count <= 0:
                active[:] = False
                break
            active[valid_count:] = False
            active_slice = active[:valid_count]
            achieved = np.asarray(batch.next_observations[offset : offset + valid_count])
            start_goals = np.asarray(batch.goals[:valid_count])
            step_rewards = goal_relabel_rewards(
                achieved,
                start_goals,
                reward_mode=reward_mode,
                success_threshold=success_threshold,
            ).astype(np.float32)
            successes = (
                np.linalg.norm(
                    np.asarray(achieved, dtype=float)
                    - np.asarray(start_goals, dtype=float),
                    axis=1,
                )
                <= success_threshold
            )
            returns[:valid_count, horizon_index] += scale * step_rewards * active_slice
            effective_steps[:valid_count, horizon_index] = np.where(
                active_slice,
                offset + 1,
                effective_steps[:valid_count, horizon_index],
            )
            next_observations[:valid_count, horizon_index] = np.where(
                active_slice[:, None],
                achieved,
                next_observations[:valid_count, horizon_index],
            )
            active[:valid_count] = (
                active_slice
                & ~successes
                & ~terminals[offset : offset + valid_count]
            )
            scale *= discount
        discounts[:, horizon_index] = (discount**horizon) * active.astype(np.float32)

    return HorizonTargets(
        horizons=tuple(int(horizon) for horizon in horizons),
        returns=returns,
        next_observations=next_observations,
        discounts=discounts,
        effective_steps=effective_steps,
    )


def attach_horizon_targets(batch: TransitionBatch, targets: HorizonTargets) -> TransitionBatch:
    return TransitionBatch(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        next_observations=batch.next_observations,
        terminals=batch.terminals,
        goals=batch.goals,
        actor_goals=batch.actor_goals,
        masks=batch.masks,
        horizon_targets=targets,
    )
