"""Synthetic datasets for deterministic smoke tests and CLI checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_gcrl.data.replay_buffer import TransitionBatch


@dataclass(frozen=True)
class SyntheticGCRLConfig:
    num_transitions: int = 256
    observation_dim: int = 4
    goal_dim: int = 4
    action_dim: int = 2
    noise_std: float = 0.01

    def __post_init__(self) -> None:
        if self.num_transitions <= 1:
            raise ValueError("num_transitions must be greater than one.")
        if self.observation_dim <= 0 or self.goal_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Synthetic dimensions must be positive.")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")


def make_synthetic_gcrl_batch(config: SyntheticGCRLConfig, seed: int) -> TransitionBatch:
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(config.num_transitions, config.observation_dim))
    goals = rng.normal(size=(config.num_transitions, config.goal_dim))
    obs_weights = rng.normal(scale=0.5, size=(config.observation_dim, config.action_dim))
    goal_weights = rng.normal(scale=0.5, size=(config.goal_dim, config.action_dim))
    clean_actions = observations @ obs_weights + goals @ goal_weights
    actions = clean_actions + rng.normal(scale=config.noise_std, size=clean_actions.shape)

    transition_projection = rng.normal(scale=0.1, size=(config.action_dim, config.observation_dim))
    next_observations = observations + actions @ transition_projection
    rewards = -np.linalg.norm(next_observations - goals[:, : config.observation_dim], axis=1)
    terminals = np.zeros((config.num_transitions,), dtype=bool)
    terminals[-1] = True
    return TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        goals=goals,
    )

