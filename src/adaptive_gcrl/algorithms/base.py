"""Common interfaces for offline RL agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from adaptive_gcrl.data.replay_buffer import TransitionBatch


@dataclass(frozen=True)
class TrainResult:
    step: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]


class OfflineAgent(Protocol):
    def train_step(self, batch: TransitionBatch, rng: np.random.Generator) -> TrainResult:
        """Update the agent from an offline batch."""

    def predict(self, observations: np.ndarray, goals: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict actions for observations and optional goals."""

    def evaluate_batch(self, batch: TransitionBatch) -> EvaluationResult:
        """Evaluate the agent on a held-out transition batch."""
