"""Behavioral cloning baseline for smoke-testable offline GCRL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from adaptive_gcrl.algorithms.base import EvaluationResult, TrainResult
from adaptive_gcrl.data.replay_buffer import TransitionBatch


@dataclass(frozen=True)
class BehavioralCloningConfig:
    learning_rate: float = 0.05
    l2_regularization: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.l2_regularization < 0.0:
            raise ValueError("l2_regularization must be non-negative.")


class LinearBehavioralCloningAgent:
    """Linear least-squares policy trained with gradient descent."""

    def __init__(self, observation_dim: int, action_dim: int, goal_dim: int = 0, config: Optional[BehavioralCloningConfig] = None) -> None:
        if observation_dim <= 0 or action_dim <= 0 or goal_dim < 0:
            raise ValueError("Agent dimensions must be positive, with non-negative goal_dim.")
        self.config = config or BehavioralCloningConfig()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.weights = np.zeros((observation_dim + goal_dim, action_dim), dtype=float)
        self.bias = np.zeros((action_dim,), dtype=float)
        self.step = 0

    def _features(self, observations: np.ndarray, goals: Optional[np.ndarray]) -> np.ndarray:
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError(f"observations must have shape (batch, {self.observation_dim}).")
        if self.goal_dim == 0:
            return observations
        if goals is None:
            raise ValueError("goals are required for this behavioral cloning agent.")
        goals = np.asarray(goals, dtype=float)
        if goals.ndim != 2 or goals.shape != (observations.shape[0], self.goal_dim):
            raise ValueError(f"goals must have shape (batch, {self.goal_dim}).")
        return np.concatenate([observations, goals], axis=1)

    def predict(self, observations: np.ndarray, goals: Optional[np.ndarray] = None) -> np.ndarray:
        features = self._features(observations, goals)
        return features @ self.weights + self.bias

    def train_step(self, batch: TransitionBatch, rng: np.random.Generator) -> TrainResult:
        del rng
        features = self._features(batch.observations, batch.goals)
        targets = np.asarray(batch.actions, dtype=float)
        predictions = features @ self.weights + self.bias
        residuals = predictions - targets
        loss = float(np.mean(residuals**2))

        grad_predictions = 2.0 * residuals / residuals.size
        grad_weights = features.T @ grad_predictions + self.config.l2_regularization * self.weights
        grad_bias = np.sum(grad_predictions, axis=0)
        self.weights -= self.config.learning_rate * grad_weights
        self.bias -= self.config.learning_rate * grad_bias
        self.step += 1

        regularized_loss = loss + 0.5 * self.config.l2_regularization * float(np.sum(self.weights**2))
        return TrainResult(step=self.step, metrics={"bc_loss": regularized_loss, "action_mse": loss})

    def evaluate_batch(self, batch: TransitionBatch) -> EvaluationResult:
        predictions = self.predict(batch.observations, batch.goals)
        mse = float(np.mean((predictions - batch.actions) ** 2))
        return EvaluationResult(metrics={"eval_action_mse": mse})


def make_behavioral_cloning_agent(batch: TransitionBatch, config: Optional[BehavioralCloningConfig] = None) -> LinearBehavioralCloningAgent:
    goal_dim = 0 if batch.goals is None else int(batch.goals.shape[1])
    return LinearBehavioralCloningAgent(
        observation_dim=int(batch.observations.shape[1]),
        action_dim=int(batch.actions.shape[1]),
        goal_dim=goal_dim,
        config=config,
    )

