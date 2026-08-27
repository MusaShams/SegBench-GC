"""Fixed temporal-resolution baselines for adaptive-horizon ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from adaptive_gcrl.algorithms.base import EvaluationResult, OfflineAgent, TrainResult
from adaptive_gcrl.algorithms.bc import BehavioralCloningConfig, make_behavioral_cloning_agent
from adaptive_gcrl.data.replay_buffer import TransitionBatch


@dataclass(frozen=True)
class FixedTemporalConfig:
    horizon: int = 8
    chunk_size: int = 1
    learning_rate: float = 0.05
    l2_regularization: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")


class FixedTemporalBaseline:
    """Executable matched-surface baseline with fixed horizon and chunk size."""

    def __init__(self, base_agent: OfflineAgent, config: FixedTemporalConfig) -> None:
        self.base_agent = base_agent
        self.config = config

    def train_step(self, batch: TransitionBatch, rng: np.random.Generator) -> TrainResult:
        result = self.base_agent.train_step(batch, rng)
        metrics = dict(result.metrics)
        metrics["horizon"] = float(self.config.horizon)
        metrics["chunk_size"] = float(self.config.chunk_size)
        return TrainResult(step=result.step, metrics=metrics)

    def predict(self, observations: np.ndarray, goals: Optional[np.ndarray] = None) -> np.ndarray:
        return self.base_agent.predict(observations, goals)

    def evaluate_batch(self, batch: TransitionBatch) -> EvaluationResult:
        result = self.base_agent.evaluate_batch(batch)
        metrics = dict(result.metrics)
        metrics["horizon"] = float(self.config.horizon)
        metrics["chunk_size"] = float(self.config.chunk_size)
        return EvaluationResult(metrics=metrics)


def make_fixed_temporal_baseline(batch: TransitionBatch, config: FixedTemporalConfig) -> FixedTemporalBaseline:
    base_agent = make_behavioral_cloning_agent(
        batch,
        BehavioralCloningConfig(
            learning_rate=config.learning_rate,
            l2_regularization=config.l2_regularization,
        ),
    )
    return FixedTemporalBaseline(base_agent=base_agent, config=config)

