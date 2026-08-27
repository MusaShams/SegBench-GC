"""Critic model placeholders for later PyTorch implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_gcrl.data.replay_buffer import TransitionBatch
from adaptive_gcrl.training.losses import mse_loss, multi_step_returns


@dataclass(frozen=True)
class CriticSpec:
    observation_dim: int
    action_dim: int
    goal_dim: int
    horizons: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 0 or self.goal_dim <= 0:
            raise ValueError("Critic dimensions must be positive.")
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("Critic horizons must be positive.")


class LinearMultiHorizonCritic:
    """Small trainable linear critic for smoke-tested multi-horizon value heads."""

    def __init__(
        self,
        spec: CriticSpec,
        learning_rate: float = 0.005,
        l2_regularization: float = 0.0,
        gradient_clip_norm: float = 10.0,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if l2_regularization < 0.0:
            raise ValueError("l2_regularization must be non-negative.")
        if gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")
        self.spec = spec
        self.learning_rate = float(learning_rate)
        self.l2_regularization = float(l2_regularization)
        self.gradient_clip_norm = float(gradient_clip_norm)
        feature_dim = spec.observation_dim + spec.goal_dim + spec.action_dim
        self.weights = np.zeros((feature_dim, len(spec.horizons)), dtype=float)
        self.bias = np.zeros((len(spec.horizons),), dtype=float)
        self.step = 0

    def _features(self, batch: TransitionBatch) -> np.ndarray:
        if batch.goals is None:
            raise ValueError("Multi-horizon critic requires goal-conditioned batches.")
        arrays = (batch.observations, batch.goals, batch.actions)
        expected_dims = (self.spec.observation_dim, self.spec.goal_dim, self.spec.action_dim)
        for name, value, expected in zip(("observations", "goals", "actions"), arrays, expected_dims):
            if value.ndim != 2 or value.shape[1] != expected:
                raise ValueError(f"{name} must have shape (batch, {expected}).")
        return np.clip(np.concatenate(arrays, axis=1).astype(float), -10.0, 10.0)

    def predict(self, batch: TransitionBatch) -> np.ndarray:
        return np.einsum("bf,fh->bh", self._features(batch), self.weights) + self.bias

    def target_returns(self, batch: TransitionBatch, discount: float) -> np.ndarray:
        return np.clip(multi_step_returns(batch.rewards, batch.terminals, self.spec.horizons, discount), -100.0, 100.0)

    def train_step(self, batch: TransitionBatch, discount: float) -> dict[str, float]:
        features = self._features(batch)
        targets = self.target_returns(batch, discount)
        predictions = np.einsum("bf,fh->bh", features, self.weights) + self.bias
        residuals = predictions - targets
        loss = mse_loss(predictions, targets)

        grad_predictions = 2.0 * residuals / residuals.size
        grad_weights = np.einsum("bf,bh->fh", features, grad_predictions) + self.l2_regularization * self.weights
        grad_bias = np.sum(grad_predictions, axis=0)
        grad_norm = float(np.sqrt(np.sum(grad_weights**2) + np.sum(grad_bias**2)))
        if grad_norm > self.gradient_clip_norm:
            scale = self.gradient_clip_norm / grad_norm
            grad_weights *= scale
            grad_bias *= scale
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias
        self.step += 1
        return {
            "critic_loss": loss + 0.5 * self.l2_regularization * float(np.sum(self.weights**2)),
            "critic_target_mean": float(np.mean(targets)),
            "critic_prediction_mean": float(np.mean(predictions)),
            "critic_grad_norm": min(grad_norm, self.gradient_clip_norm),
        }

    def horizon_values(self, batch: TransitionBatch) -> tuple[list[float], list[float]]:
        predictions = self.predict(batch)
        return (
            [float(value) for value in np.mean(predictions, axis=0)],
            [float(value) for value in np.std(predictions, axis=0)],
        )
