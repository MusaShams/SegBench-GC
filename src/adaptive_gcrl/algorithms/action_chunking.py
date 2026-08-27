"""Agent wrapper for action-chunked policies."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from adaptive_gcrl.algorithms.base import EvaluationResult, OfflineAgent, TrainResult
from adaptive_gcrl.data.replay_buffer import TransitionBatch


class ActionChunkingAgent:
    def __init__(self, inner_agent: OfflineAgent, chunk_size: int, primitive_action_dim: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if primitive_action_dim <= 0:
            raise ValueError("primitive_action_dim must be positive.")
        self.inner_agent = inner_agent
        self.chunk_size = chunk_size
        self.primitive_action_dim = primitive_action_dim
        self._cached_actions: list[np.ndarray] = []
        self._cached_horizons: list[float] = []

    def reset_policy(self) -> None:
        self._cached_actions = []
        self._cached_horizons = []

    def train_step(self, batch: TransitionBatch, rng: np.random.Generator) -> TrainResult:
        result = self.inner_agent.train_step(batch, rng)
        metrics = dict(result.metrics)
        metrics["action_chunk_size"] = float(self.chunk_size)
        return TrainResult(step=result.step, metrics=metrics)

    def predict(self, observations: np.ndarray, goals: Optional[np.ndarray] = None) -> np.ndarray:
        if observations.shape[0] != 1:
            chunk_actions = self.inner_agent.predict(observations, goals)
            return chunk_actions.reshape(observations.shape[0], self.chunk_size, self.primitive_action_dim)[:, 0, :]
        if not self._cached_actions:
            chunk = self.inner_agent.predict(observations, goals).reshape(self.chunk_size, self.primitive_action_dim)
            self._cached_actions = [action.copy() for action in chunk]
        return self._cached_actions.pop(0).reshape(1, self.primitive_action_dim)

    def predict_with_info(
        self,
        observations: np.ndarray,
        goals: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if observations.shape[0] != 1:
            raise ValueError("Chunked rollout prediction requires a single observation.")
        if not self._cached_actions:
            if hasattr(self.inner_agent, "predict_with_info"):
                chunk_batch, info = self.inner_agent.predict_with_info(observations, goals)
                selected_horizon = float(np.asarray(info["selected_horizon"]).reshape(-1)[0])
            else:
                chunk_batch = self.inner_agent.predict(observations, goals)
                selected_horizon = float(self.chunk_size)
            chunk = chunk_batch.reshape(self.chunk_size, self.primitive_action_dim)
            self._cached_actions = [action.copy() for action in chunk]
            self._cached_horizons = [selected_horizon] * self.chunk_size
        action = self._cached_actions.pop(0).reshape(1, self.primitive_action_dim)
        horizon = self._cached_horizons.pop(0)
        return action, {"selected_horizon": np.asarray([horizon])}

    def evaluate_batch(self, batch: TransitionBatch) -> EvaluationResult:
        result = self.inner_agent.evaluate_batch(batch)
        metrics = dict(result.metrics)
        metrics["action_chunk_size"] = float(self.chunk_size)
        return EvaluationResult(metrics=metrics)

    def save_checkpoint(self, path: Path) -> None:
        if not hasattr(self.inner_agent, "save_checkpoint"):
            raise TypeError(f"Inner agent of type {type(self.inner_agent).__name__} does not support checkpointing.")
        self.inner_agent.save_checkpoint(path)

    def load_checkpoint(self, path: Path) -> None:
        if not hasattr(self.inner_agent, "load_checkpoint"):
            raise TypeError(f"Inner agent of type {type(self.inner_agent).__name__} does not support checkpoint loading.")
        self.inner_agent.load_checkpoint(path)
