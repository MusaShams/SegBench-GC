"""Adaptive-horizon action-chunking agent scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from adaptive_gcrl.algorithms.base import EvaluationResult, OfflineAgent, TrainResult
from adaptive_gcrl.algorithms.bc import BehavioralCloningConfig, make_behavioral_cloning_agent
from adaptive_gcrl.data.replay_buffer import TransitionBatch
from adaptive_gcrl.models.chunk_policy import ChunkPolicy
from adaptive_gcrl.models.critics import CriticSpec, LinearMultiHorizonCritic
from adaptive_gcrl.models.horizon_gate import HorizonGate, HorizonScore


@dataclass(frozen=True)
class AdaptiveHorizonConfig:
    horizons: tuple[int, ...] = (1, 2, 4, 8, 16)
    chunk_size: int = 4
    temperature: float = 1.0
    uncertainty_penalty: float = 0.25

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("At least one candidate horizon is required.")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("Candidate horizons must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")


class AdaptiveHorizonAgent:
    """Minimal decision scaffold for adaptive horizon/chunk ablations."""

    def __init__(self, config: AdaptiveHorizonConfig) -> None:
        self.config = config
        self.gate = HorizonGate(
            horizons=config.horizons,
            temperature=config.temperature,
            uncertainty_penalty=config.uncertainty_penalty,
        )
        self.chunk_policy = ChunkPolicy(chunk_size=config.chunk_size)

    def select_horizon(self, values: Sequence[float], uncertainties: Optional[Sequence[float]] = None) -> HorizonScore:
        return self.gate.select(values=values, uncertainties=uncertainties)

    def chunk_actions(self, primitive_actions: np.ndarray) -> np.ndarray:
        return self.chunk_policy.to_chunks(primitive_actions)


@dataclass(frozen=True)
class AdaptiveHorizonBaselineConfig(AdaptiveHorizonConfig):
    learning_rate: float = 0.05
    critic_learning_rate: float = 0.005
    l2_regularization: float = 0.0
    discount: float = 0.99
    learned_gate: bool = False
    gate_learning_rate: float = 1e-3
    gate_hidden_dim: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive.")
        if self.l2_regularization < 0.0:
            raise ValueError("l2_regularization must be non-negative.")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1].")
        if self.gate_learning_rate <= 0.0:
            raise ValueError("gate_learning_rate must be positive.")
        if self.gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive.")


class AdaptiveHorizonTrainingAgent:
    """Executable adaptive-horizon scaffold with a BC low-level policy."""

    def __init__(self, base_agent: OfflineAgent, critic: LinearMultiHorizonCritic, config: AdaptiveHorizonBaselineConfig) -> None:
        self.base_agent = base_agent
        self.critic = critic
        self.config = config
        self.learned_gate: Optional[Any] = None
        self.temporal_agent = AdaptiveHorizonAgent(
            AdaptiveHorizonConfig(
                horizons=config.horizons,
                chunk_size=config.chunk_size,
                temperature=config.temperature,
                uncertainty_penalty=config.uncertainty_penalty,
            )
        )
        if config.learned_gate:
            from adaptive_gcrl.models.torch_gate import TorchHorizonGate, TorchHorizonGateConfig

            self.learned_gate = TorchHorizonGate(
                TorchHorizonGateConfig(
                    horizons=config.horizons,
                    hidden_dim=config.gate_hidden_dim,
                    learning_rate=config.gate_learning_rate,
                    uncertainty_penalty=config.uncertainty_penalty,
                )
            )

    def _score_horizons(self, batch: TransitionBatch) -> tuple[list[float], list[float]]:
        return self.critic.horizon_values(batch)

    def select_horizon_for_batch(self, batch: TransitionBatch) -> HorizonScore:
        values, uncertainties = self._score_horizons(batch)
        return self._select_from_scores(values, uncertainties)

    def _select_from_scores(self, values: Sequence[float], uncertainties: Sequence[float]) -> HorizonScore:
        if self.learned_gate is not None:
            return self.learned_gate.select(values, uncertainties)
        return self.temporal_agent.select_horizon(values, uncertainties)

    def train_step(self, batch: TransitionBatch, rng: np.random.Generator) -> TrainResult:
        critic_metrics = self.critic.train_step(batch, self.config.discount)
        values, uncertainties = self._score_horizons(batch)
        target_score = self.temporal_agent.select_horizon(values, uncertainties)
        gate_metrics: dict[str, float] = {}
        if self.learned_gate is not None:
            gate_metrics = self.learned_gate.fit_step(values, uncertainties, target_score.index)
        horizon_score = self._select_from_scores(values, uncertainties)
        result = self.base_agent.train_step(batch, rng)
        metrics = dict(result.metrics)
        metrics.update(critic_metrics)
        metrics.update(gate_metrics)
        metrics.update(
            {
                "selected_horizon": float(horizon_score.horizon),
                "selected_horizon_index": float(horizon_score.index),
                "target_horizon": float(target_score.horizon),
                "chunk_size": float(self.config.chunk_size),
            }
        )
        return TrainResult(step=result.step, metrics=metrics)

    def predict(self, observations: np.ndarray, goals: Optional[np.ndarray] = None) -> np.ndarray:
        return self.base_agent.predict(observations, goals)

    def evaluate_batch(self, batch: TransitionBatch) -> EvaluationResult:
        horizon_score = self.select_horizon_for_batch(batch)
        result = self.base_agent.evaluate_batch(batch)
        metrics = dict(result.metrics)
        metrics.update(
            {
                "selected_horizon": float(horizon_score.horizon),
                "selected_horizon_index": float(horizon_score.index),
                "learned_gate": float(self.learned_gate is not None),
                "chunk_size": float(self.config.chunk_size),
            }
        )
        return EvaluationResult(metrics=metrics)


def make_adaptive_horizon_training_agent(batch: TransitionBatch, config: AdaptiveHorizonBaselineConfig) -> AdaptiveHorizonTrainingAgent:
    base_agent = make_behavioral_cloning_agent(
        batch,
        BehavioralCloningConfig(
            learning_rate=config.learning_rate,
            l2_regularization=config.l2_regularization,
        ),
    )
    if batch.goals is None:
        raise ValueError("Adaptive-horizon training requires goal-conditioned batches.")
    critic = LinearMultiHorizonCritic(
        CriticSpec(
            observation_dim=int(batch.observations.shape[1]),
            action_dim=int(batch.actions.shape[1]),
            goal_dim=int(batch.goals.shape[1]),
            horizons=config.horizons,
        ),
        learning_rate=config.critic_learning_rate,
        l2_regularization=config.l2_regularization,
    )
    return AdaptiveHorizonTrainingAgent(base_agent=base_agent, critic=critic, config=config)
