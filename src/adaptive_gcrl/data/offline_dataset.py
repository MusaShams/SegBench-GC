"""Conversion helpers for offline RL dataset dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

from adaptive_gcrl.data.replay_buffer import TransitionBatch


def _required_array(dataset: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in dataset:
        raise KeyError(f"Dataset is missing required key: {key}")
    return np.asarray(dataset[key])


def _optional_array(dataset: Mapping[str, Any], *keys: str) -> Optional[np.ndarray]:
    for key in keys:
        if key in dataset:
            return np.asarray(dataset[key])
    return None


def transition_batch_from_dataset(dataset: Mapping[str, Any]) -> TransitionBatch:
    """Normalize a D4RL/OGBench-style dataset mapping into a transition batch."""

    observations = _required_array(dataset, "observations")
    actions = _required_array(dataset, "actions")
    rewards = _required_array(dataset, "rewards")
    terminals = _optional_array(dataset, "terminals", "dones")
    if terminals is None:
        raise KeyError("Dataset is missing required key: terminals or dones")

    next_observations = _optional_array(dataset, "next_observations")
    if next_observations is None:
        if observations.shape[0] < 2:
            raise ValueError("Cannot infer next_observations from fewer than two observations.")
        observations = observations[:-1]
        actions = actions[:-1]
        rewards = rewards[:-1]
        terminals = terminals[:-1]
        next_observations = np.asarray(dataset["observations"])[1:]

    goals = _optional_array(dataset, "goals", "desired_goals")
    if goals is not None and goals.shape[0] != observations.shape[0]:
        goals = goals[: observations.shape[0]]

    return TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals.astype(bool),
        goals=goals,
    )


def goal_relabel_rewards(
    achieved_goals: np.ndarray,
    goals: np.ndarray,
    *,
    reward_mode: str = "dense_negative_distance",
    success_threshold: float = 1.0,
) -> np.ndarray:
    if success_threshold <= 0.0:
        raise ValueError("success_threshold must be positive.")
    distances = np.linalg.norm(np.asarray(achieved_goals, dtype=float) - np.asarray(goals, dtype=float), axis=1)
    if reward_mode == "dense_negative_distance":
        return -distances
    if reward_mode == "sparse_success":
        return (distances <= success_threshold).astype(float)
    if reward_mode == "sparse_failure":
        return np.where(distances <= success_threshold, 0.0, -1.0)
    raise ValueError(f"Unsupported goal relabel reward_mode: {reward_mode}")


def goal_relabel_batch_from_dataset(
    dataset: Mapping[str, Any],
    seed: int = 0,
    *,
    reward_mode: str = "dense_negative_distance",
    success_threshold: float = 1.0,
) -> TransitionBatch:
    """Create a goal-conditioned batch from reward-free offline trajectories."""

    observations = _required_array(dataset, "observations")
    actions = _required_array(dataset, "actions")
    next_observations = _optional_array(dataset, "next_observations")
    if next_observations is None:
        if observations.shape[0] < 2:
            raise ValueError("Cannot infer next_observations from fewer than two observations.")
        observations = observations[:-1]
        actions = actions[:-1]
        next_observations = np.asarray(dataset["observations"])[1:]
    terminals = _optional_array(dataset, "terminals", "dones")
    if terminals is None:
        raise KeyError("Dataset is missing required key: terminals or dones")
    terminals = terminals[: observations.shape[0]].astype(bool)

    rng = np.random.default_rng(seed)
    goal_indices = np.empty((observations.shape[0],), dtype=int)
    start = 0
    terminal_indices = np.flatnonzero(terminals)
    if terminal_indices.size == 0 or terminal_indices[-1] != observations.shape[0] - 1:
        terminal_indices = np.concatenate([terminal_indices, np.array([observations.shape[0] - 1])])
    for end in terminal_indices:
        indices = np.arange(start, end + 1)
        goal_indices[start : end + 1] = rng.integers(low=indices, high=end + 1)
        start = end + 1

    goals = np.asarray(next_observations)[goal_indices]
    rewards = goal_relabel_rewards(
        np.asarray(next_observations, dtype=float),
        goals.astype(float),
        reward_mode=reward_mode,
        success_threshold=success_threshold,
    )
    return TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        goals=goals,
    )
