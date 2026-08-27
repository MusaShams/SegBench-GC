"""Action chunking transforms for offline transition batches."""

from __future__ import annotations

import numpy as np

from adaptive_gcrl.data.replay_buffer import TransitionBatch


def make_action_chunk_batch(batch: TransitionBatch, chunk_size: int, discount: float = 1.0) -> TransitionBatch:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1].")
    if batch.size < chunk_size:
        raise ValueError("Batch is too small to form one complete action chunk.")

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminals: list[bool] = []
    goals: list[np.ndarray] = []
    has_goals = batch.goals is not None

    for start in range(0, batch.size - chunk_size + 1):
        end = start + chunk_size
        terminal_window = np.asarray(batch.terminals[start:end], dtype=bool)
        if np.any(np.asarray(batch.terminals[start : end - 1], dtype=bool)):
            continue
        observations.append(batch.observations[start])
        actions.append(batch.actions[start:end].reshape(-1))
        discounted_reward = 0.0
        scale = 1.0
        for reward in batch.rewards[start:end]:
            discounted_reward += scale * float(reward)
            scale *= discount
        rewards.append(discounted_reward)
        next_observations.append(batch.next_observations[end - 1])
        terminals.append(bool(np.any(terminal_window)))
        if has_goals:
            goals.append(batch.goals[start])

    if not observations:
        raise ValueError("No valid chunks could be formed without crossing terminal boundaries.")

    return TransitionBatch(
        observations=np.asarray(observations),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards, dtype=float),
        next_observations=np.asarray(next_observations),
        terminals=np.asarray(terminals, dtype=bool),
        goals=None if not has_goals else np.asarray(goals),
        actor_goals=None,
        masks=None,
    )
