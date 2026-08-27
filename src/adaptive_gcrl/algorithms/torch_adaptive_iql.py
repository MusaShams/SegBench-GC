"""Adaptive-horizon PyTorch IQL agent."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from adaptive_gcrl.algorithms.base import EvaluationResult, TrainResult
from adaptive_gcrl.algorithms.torch_iql import MLP, _as_tensor, _expectile_loss, _normalization_stats
from adaptive_gcrl.data.replay_buffer import TransitionBatch
from adaptive_gcrl.models.torch_gate import TorchHorizonGate, TorchHorizonGateConfig


@dataclass(frozen=True)
class TorchAdaptiveIQLConfig:
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    chunk_size: int = 4
    learning_rate: float = 1e-3
    gate_learning_rate: float = 1e-3
    hidden_dim: int = 256
    gate_hidden_dim: int = 32
    discount: float = 0.99
    expectile: float = 0.7
    advantage_temperature: float = 3.0
    advantage_clip: float = 100.0
    target_tau: float = 0.005
    hidden_layers: int = 2
    activation: str = "relu"
    layer_norm: bool = True
    actor_loss_mode: str = "awr"
    actor_alpha: float = 3.0
    uncertainty_penalty: float = 0.25
    horizon_value_weight: float = 1.0
    horizon_value_mode: str = "cumulative"
    horizon_penalty: float = 0.0
    horizon_prior_center: Optional[float] = None
    horizon_prior_penalty: float = 0.0
    gate_target_smoothing: float = 0.0
    gate_entropy_regularization: float = 0.0
    gate_selection_strategy: str = "argmax"
    actor_horizon_weighting: str = "selected"
    gate_execution_strategy: str = "argmax"
    static_horizon_weights: Optional[tuple[float, ...]] = None
    support_temperature: Optional[float] = None
    cross_horizon_consistency_weight: float = 0.0
    normalize_inputs: bool = False
    actor_output_activation: str = "identity"
    goal_direction_loss_weight: float = 0.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must be positive.")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be strictly increasing and unique.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.learning_rate <= 0.0 or self.gate_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive.")
        if self.hidden_dim <= 0 or self.gate_hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive.")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1].")
        if not 0.0 < self.expectile < 1.0:
            raise ValueError("expectile must be in (0, 1).")
        if self.advantage_temperature <= 0.0 or self.advantage_clip <= 0.0:
            raise ValueError("advantage parameters must be positive.")
        if not 0.0 < self.target_tau <= 1.0:
            raise ValueError("target_tau must be in (0, 1].")
        if self.hidden_layers <= 0:
            raise ValueError("hidden_layers must be positive.")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'.")
        if self.actor_loss_mode not in {"awr", "ddpgbc"}:
            raise ValueError("actor_loss_mode must be 'awr' or 'ddpgbc'.")
        if self.actor_alpha <= 0.0:
            raise ValueError("actor_alpha must be positive.")
        if self.uncertainty_penalty < 0.0:
            raise ValueError("uncertainty_penalty must be non-negative.")
        if self.horizon_value_weight < 0.0:
            raise ValueError("horizon_value_weight must be non-negative.")
        if self.horizon_value_mode not in {"cumulative", "per_step", "sqrt_horizon"}:
            raise ValueError("horizon_value_mode must be 'cumulative', 'per_step', or 'sqrt_horizon'.")
        if self.horizon_penalty < 0.0:
            raise ValueError("horizon_penalty must be non-negative.")
        if self.horizon_prior_center is not None and self.horizon_prior_center <= 0.0:
            raise ValueError("horizon_prior_center must be positive when provided.")
        if self.horizon_prior_penalty < 0.0:
            raise ValueError("horizon_prior_penalty must be non-negative.")
        if not 0.0 <= self.gate_target_smoothing < 1.0:
            raise ValueError("gate_target_smoothing must be in [0, 1).")
        if self.gate_entropy_regularization < 0.0:
            raise ValueError("gate_entropy_regularization must be non-negative.")
        if self.gate_selection_strategy not in {"argmax", "sample"}:
            raise ValueError("gate_selection_strategy must be 'argmax' or 'sample'.")
        if self.actor_horizon_weighting not in {"selected", "gate"}:
            raise ValueError("actor_horizon_weighting must be 'selected' or 'gate'.")
        if self.gate_execution_strategy not in {"argmax", "mixture"}:
            raise ValueError("gate_execution_strategy must be 'argmax' or 'mixture'.")
        if self.static_horizon_weights is not None:
            if len(self.static_horizon_weights) != len(self.horizons):
                raise ValueError(
                    "static_horizon_weights must match the number of horizons."
                )
            if any(weight < 0.0 for weight in self.static_horizon_weights):
                raise ValueError("static_horizon_weights must be non-negative.")
            if not np.isclose(sum(self.static_horizon_weights), 1.0):
                raise ValueError("static_horizon_weights must sum to one.")
        if self.support_temperature is not None and self.support_temperature <= 0.0:
            raise ValueError("support_temperature must be positive when provided.")
        if self.cross_horizon_consistency_weight < 0.0:
            raise ValueError(
                "cross_horizon_consistency_weight must be non-negative."
            )
        if self.actor_output_activation not in {"identity", "tanh"}:
            raise ValueError("actor_output_activation must be 'identity' or 'tanh'.")
        if self.goal_direction_loss_weight < 0.0:
            raise ValueError("goal_direction_loss_weight must be non-negative.")


class TorchAdaptiveIQLAgent:
    """Multi-horizon IQL agent with learned horizon selection."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        goal_dim: int,
        config: Optional[TorchAdaptiveIQLConfig] = None,
        observation_mean: Optional[np.ndarray] = None,
        observation_std: Optional[np.ndarray] = None,
        goal_mean: Optional[np.ndarray] = None,
        goal_std: Optional[np.ndarray] = None,
    ) -> None:
        if observation_dim <= 0 or action_dim <= 0 or goal_dim <= 0:
            raise ValueError("observation_dim, action_dim, and goal_dim must be positive.")
        self.config = config or TorchAdaptiveIQLConfig()
        self.device = torch.device(self.config.device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.num_horizons = len(self.config.horizons)
        self.observation_mean = _as_tensor(np.zeros(observation_dim) if observation_mean is None else observation_mean, self.device)
        self.observation_std = _as_tensor(np.ones(observation_dim) if observation_std is None else observation_std, self.device)
        self.goal_mean = _as_tensor(np.zeros(goal_dim) if goal_mean is None else goal_mean, self.device)
        self.goal_std = _as_tensor(np.ones(goal_dim) if goal_std is None else goal_std, self.device)
        state_goal_dim = observation_dim + goal_dim
        q_input_dim = state_goal_dim + action_dim

        network_kwargs = {
            "hidden_layers": self.config.hidden_layers,
            "activation": self.config.activation,
            "layer_norm": self.config.layer_norm,
        }
        self.actor = MLP(
            state_goal_dim,
            action_dim * self.num_horizons,
            self.config.hidden_dim,
            **network_kwargs,
        ).to(self.device)
        self.critic1 = MLP(q_input_dim, self.num_horizons, self.config.hidden_dim, **network_kwargs).to(self.device)
        self.critic2 = MLP(q_input_dim, self.num_horizons, self.config.hidden_dim, **network_kwargs).to(self.device)
        self.target_critic1 = MLP(q_input_dim, self.num_horizons, self.config.hidden_dim, **network_kwargs).to(self.device)
        self.target_critic2 = MLP(q_input_dim, self.num_horizons, self.config.hidden_dim, **network_kwargs).to(self.device)
        self.value = MLP(state_goal_dim, self.num_horizons, self.config.hidden_dim, **network_kwargs).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.gate = TorchHorizonGate(
            TorchHorizonGateConfig(
                horizons=self.config.horizons,
                hidden_dim=self.config.gate_hidden_dim,
                learning_rate=self.config.gate_learning_rate,
                uncertainty_penalty=self.config.uncertainty_penalty,
                target_smoothing=self.config.gate_target_smoothing,
                entropy_regularization=self.config.gate_entropy_regularization,
                device=self.config.device,
            )
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.config.learning_rate,
        )
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.config.learning_rate)
        self.step = 0

    def _state_goal(self, observations: np.ndarray, goals: np.ndarray) -> torch.Tensor:
        obs = _as_tensor(observations, self.device)
        goal = _as_tensor(goals, self.device)
        if obs.ndim != 2 or obs.shape[1] != self.observation_dim:
            raise ValueError(f"observations must have shape (batch, {self.observation_dim}).")
        if goal.ndim != 2 or goal.shape != (obs.shape[0], self.goal_dim):
            raise ValueError(f"goals must have shape (batch, {self.goal_dim}).")
        if self.config.normalize_inputs:
            obs = (obs - self.observation_mean) / self.observation_std
            goal = (goal - self.goal_mean) / self.goal_std
        return torch.cat([obs, goal], dim=-1)

    def _batch_tensors(
        self,
        batch: TransitionBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch.goals is None:
            raise ValueError("TorchAdaptiveIQLAgent requires goal-conditioned batches.")
        state_goal = self._state_goal(batch.observations, batch.goals)
        actor_goals = batch.goals if batch.actor_goals is None else batch.actor_goals
        actor_state_goal = self._state_goal(batch.observations, actor_goals)
        actions = _as_tensor(batch.actions, self.device)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"actions must have shape (batch, {self.action_dim}).")
        return state_goal, actor_state_goal, actions

    def _qs(
        self,
        state_goal: torch.Tensor,
        actions: torch.Tensor,
        *,
        target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([state_goal, actions], dim=-1)
        critic1 = self.target_critic1 if target else self.critic1
        critic2 = self.target_critic2 if target else self.critic2
        return critic1(inputs), critic2(inputs)

    def _min_q(self, state_goal: torch.Tensor, actions: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        q1, q2 = self._qs(state_goal, actions, target=target)
        return torch.minimum(q1, q2)

    def _actor_heads(self, state_goal: torch.Tensor) -> torch.Tensor:
        actions = self.actor(state_goal).reshape(-1, self.num_horizons, self.action_dim)
        if self.config.actor_output_activation == "tanh":
            return torch.tanh(actions)
        return actions

    def _goal_direction_targets(self, batch: TransitionBatch) -> torch.Tensor:
        actor_goals = batch.goals if batch.actor_goals is None else batch.actor_goals
        if actor_goals is None:
            raise ValueError("Goal direction loss requires goal-conditioned batches.")
        if self.observation_dim != self.goal_dim or self.action_dim != self.observation_dim:
            raise ValueError("Goal direction loss requires observation_dim == goal_dim == action_dim.")
        observations = _as_tensor(batch.observations, self.device)
        goals = _as_tensor(actor_goals, self.device)
        deltas = goals - observations
        norms = torch.linalg.norm(deltas, dim=-1, keepdim=True).clamp_min(1e-6)
        return deltas / norms

    def _critic_target(self, batch: TransitionBatch) -> torch.Tensor:
        if batch.goals is None or batch.horizon_targets is None:
            raise ValueError("Adaptive IQL requires precomputed trajectory-aware horizon targets.")
        if batch.horizon_targets.horizons != self.config.horizons:
            raise ValueError(
                f"Batch horizon targets {batch.horizon_targets.horizons} do not match config {self.config.horizons}."
            )
        returns = _as_tensor(batch.horizon_targets.returns, self.device)
        discounts = _as_tensor(batch.horizon_targets.discounts, self.device)
        batch_size = batch.size
        next_observations = np.asarray(batch.horizon_targets.next_observations)
        repeated_goals = np.repeat(np.asarray(batch.goals[:, None, :]), self.num_horizons, axis=1)
        flat_next_state_goal = self._state_goal(
            next_observations.reshape(batch_size * self.num_horizons, self.observation_dim),
            repeated_goals.reshape(batch_size * self.num_horizons, self.goal_dim),
        )
        with torch.no_grad():
            all_next_values = self.value(flat_next_state_goal).reshape(
                batch_size,
                self.num_horizons,
                self.num_horizons,
            )
            matching_next_values = torch.diagonal(all_next_values, dim1=1, dim2=2)
            return returns + discounts * matching_next_values

    def _horizon_selection_inputs(
        self,
        q_values: torch.Tensor,
        second_q_values: Optional[torch.Tensor] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if second_q_values is None:
            values_tensor = q_values
            uncertainty_tensor = torch.zeros_like(q_values)
        else:
            values_tensor = torch.minimum(q_values, second_q_values)
            # Half-disagreement keeps compatibility with the existing
            # uncertainty penalty; support_temperature uses this same scale.
            uncertainty_tensor = 0.5 * torch.abs(q_values - second_q_values)
        values = values_tensor.detach().cpu().numpy()
        uncertainties = uncertainty_tensor.detach().cpu().numpy()
        if self.config.horizon_value_mode == "per_step":
            horizons = np.asarray(self.config.horizons, dtype=float)
            values = values / horizons
            uncertainties = uncertainties / horizons
        elif self.config.horizon_value_mode == "sqrt_horizon":
            horizons = np.sqrt(np.asarray(self.config.horizons, dtype=float))
            values = values / horizons
            uncertainties = uncertainties / horizons
        return values, uncertainties

    def _support_probabilities(
        self,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        if self.config.support_temperature is None:
            raise RuntimeError(
                "support probabilities require support_temperature."
            )
        if self.config.static_horizon_weights is None:
            prior = np.full(
                self.num_horizons,
                1.0 / self.num_horizons,
                dtype=np.float32,
            )
        else:
            prior = np.asarray(
                self.config.static_horizon_weights,
                dtype=np.float32,
            )
        logits = (
            np.log(np.maximum(prior, 1e-8))[None, :]
            - uncertainties / self.config.support_temperature
        )
        logits -= np.max(logits, axis=-1, keepdims=True)
        unnormalized = np.exp(logits)
        return (
            unnormalized / np.sum(unnormalized, axis=-1, keepdims=True)
        ).astype(np.float32)

    def _cross_horizon_consistency_loss(
        self,
        q_values: torch.Tensor,
        targets: torch.Tensor,
        batch: TransitionBatch,
    ) -> tuple[torch.Tensor, float]:
        if self.num_horizons < 2:
            return q_values.new_zeros(()), 0.0
        if batch.horizon_targets is None:
            raise ValueError(
                "Cross-horizon consistency requires horizon targets."
            )
        effective_steps = _as_tensor(
            batch.horizon_targets.effective_steps,
            self.device,
        )
        horizon_scale = torch.as_tensor(
            self.config.horizons,
            dtype=torch.float32,
            device=self.device,
        )
        normalized_q = q_values / horizon_scale
        normalized_targets = targets.detach() / horizon_scale
        losses: list[torch.Tensor] = []
        valid_count = 0
        possible_count = 0
        for lower_index in range(self.num_horizons - 1):
            upper_index = lower_index + 1
            valid = (
                effective_steps[:, upper_index]
                >= float(self.config.horizons[upper_index])
            )
            possible_count += batch.size
            valid_count += int(valid.sum().item())
            if torch.any(valid):
                predicted_difference = (
                    normalized_q[valid, upper_index]
                    - normalized_q[valid, lower_index]
                )
                target_difference = (
                    normalized_targets[valid, upper_index]
                    - normalized_targets[valid, lower_index]
                )
                losses.append(
                    F.smooth_l1_loss(
                        predicted_difference,
                        target_difference,
                    )
                )
        if not losses:
            return q_values.new_zeros(()), 0.0
        return torch.stack(losses).mean(), valid_count / possible_count

    def _select_horizon(
        self,
        q_values: torch.Tensor,
        second_q_values: Optional[torch.Tensor] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        values, uncertainties = self._horizon_selection_inputs(q_values, second_q_values)
        horizon_array = np.asarray(self.config.horizons, dtype=float)
        horizon_penalties = self.config.horizon_penalty * np.log1p(horizon_array)
        if self.config.horizon_prior_center is not None and self.config.horizon_prior_penalty > 0.0:
            horizon_penalties = horizon_penalties + self.config.horizon_prior_penalty * np.abs(
                np.log(horizon_array) - np.log(float(self.config.horizon_prior_center))
            )
        adjusted = (
            self.config.horizon_value_weight * values
            - self.config.uncertainty_penalty * uncertainties
            - horizon_penalties
        )
        target_indices = np.argmax(adjusted, axis=-1).astype(int)
        if self.config.support_temperature is not None:
            gate_probabilities = self._support_probabilities(uncertainties)
            selected_indices = np.argmax(
                gate_probabilities,
                axis=-1,
            ).astype(int)
            gate_metrics = {
                "gate_loss": 0.0,
                "gate_entropy": float(
                    np.mean(
                        -np.sum(
                            gate_probabilities
                            * np.log(
                                np.maximum(gate_probabilities, 1e-8)
                            ),
                            axis=-1,
                        )
                    )
                ),
                "gate_target_probability": float(
                    np.mean(
                        gate_probabilities[
                            np.arange(values.shape[0]),
                            target_indices,
                        ]
                    )
                ),
            }
        elif self.config.static_horizon_weights is None:
            gate_metrics = self.gate.fit_step(
                values,
                uncertainties,
                target_indices,
            )
            selected_indices, gate_probabilities = self.gate.select_batch(
                values,
                uncertainties,
                strategy=self.config.gate_selection_strategy,
                rng=rng,
            )
        else:
            static_probabilities = np.asarray(
                self.config.static_horizon_weights,
                dtype=np.float32,
            )
            gate_probabilities = np.repeat(
                static_probabilities[None, :],
                values.shape[0],
                axis=0,
            )
            selected_indices = np.full(
                values.shape[0],
                int(np.argmax(static_probabilities)),
                dtype=int,
            )
            gate_metrics = {
                "gate_loss": 0.0,
                "gate_entropy": float(
                    -np.sum(
                        static_probabilities
                        * np.log(np.maximum(static_probabilities, 1e-8))
                    )
                ),
                "gate_target_probability": float(
                    np.mean(
                        gate_probabilities[
                            np.arange(values.shape[0]),
                            target_indices,
                        ]
                    )
                ),
            }
        selected_horizons = horizon_array[selected_indices]
        target_horizons = horizon_array[target_indices]
        gate_metrics.update(
            {
                "selected_horizon": float(np.mean(selected_horizons)),
                "selected_horizon_index": float(np.mean(selected_indices)),
                "target_horizon": float(np.mean(target_horizons)),
                "target_adjusted_value": float(np.mean(adjusted[np.arange(adjusted.shape[0]), target_indices])),
                "horizon_penalty": float(self.config.horizon_penalty),
                "horizon_value_weight": float(self.config.horizon_value_weight),
                "horizon_prior_center": 0.0 if self.config.horizon_prior_center is None else float(self.config.horizon_prior_center),
                "horizon_prior_penalty": float(self.config.horizon_prior_penalty),
                "gate_target_smoothing": float(self.config.gate_target_smoothing),
                "gate_entropy_regularization": float(self.config.gate_entropy_regularization),
                "actor_horizon_weighting": float(self.config.actor_horizon_weighting == "gate"),
                "chunk_size": float(self.config.chunk_size),
                "static_horizon_mixture": float(
                    self.config.static_horizon_weights is not None
                    and self.config.support_temperature is None
                ),
                "support_aware_fusion": float(
                    self.config.support_temperature is not None
                ),
            }
        )
        for horizon_index, horizon in enumerate(self.config.horizons):
            gate_metrics[f"gate_probability_{horizon}"] = float(np.mean(gate_probabilities[:, horizon_index]))
            gate_metrics[f"gate_probability_std_{horizon}"] = float(
                np.std(gate_probabilities[:, horizon_index])
            )
            gate_metrics[f"selected_horizon_fraction_{horizon}"] = float(np.mean(selected_indices == horizon_index))
            gate_metrics[f"target_horizon_fraction_{horizon}"] = float(np.mean(target_indices == horizon_index))
            gate_metrics[f"horizon_selection_value_{horizon}"] = float(np.mean(values[:, horizon_index]))
            gate_metrics[f"horizon_uncertainty_{horizon}"] = float(np.mean(uncertainties[:, horizon_index]))
            gate_metrics[f"horizon_adjusted_value_{horizon}"] = float(np.mean(adjusted[:, horizon_index]))
        return selected_indices, gate_probabilities.astype(np.float32), gate_metrics

    def _policy_q_heads(
        self,
        state_goal: torch.Tensor,
        action_heads: torch.Tensor,
        *,
        target: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1_heads: list[torch.Tensor] = []
        q2_heads: list[torch.Tensor] = []
        for horizon_index in range(self.num_horizons):
            q1, q2 = self._qs(
                state_goal,
                action_heads[:, horizon_index],
                target=target,
            )
            q1_heads.append(q1[:, horizon_index])
            q2_heads.append(q2[:, horizon_index])
        return torch.stack(q1_heads, dim=-1), torch.stack(q2_heads, dim=-1)

    def _select_policy_actions(
        self,
        state_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        action_heads = self._actor_heads(state_goal)
        with torch.no_grad():
            q1_heads, q2_heads = self._gate_q_heads(state_goal, action_heads)
            values, uncertainties = self._horizon_selection_inputs(q1_heads, q2_heads)
            if self.config.support_temperature is not None:
                probabilities = self._support_probabilities(uncertainties)
                selected_indices = np.argmax(
                    probabilities,
                    axis=-1,
                ).astype(int)
            elif self.config.static_horizon_weights is None:
                selected_indices, probabilities = self.gate.select_batch(
                    values,
                    uncertainties,
                    strategy="argmax",
                )
            else:
                static_probabilities = np.asarray(
                    self.config.static_horizon_weights,
                    dtype=np.float32,
                )
                probabilities = np.repeat(
                    static_probabilities[None, :],
                    values.shape[0],
                    axis=0,
                )
                selected_indices = np.full(
                    values.shape[0],
                    int(np.argmax(static_probabilities)),
                    dtype=int,
                )
        if self.config.gate_execution_strategy == "argmax":
            index_tensor = torch.as_tensor(
                selected_indices,
                dtype=torch.long,
                device=self.device,
            )
            selected_actions = action_heads[
                torch.arange(action_heads.shape[0], device=self.device),
                index_tensor,
            ]
            reported_horizons = np.asarray(self.config.horizons, dtype=float)[
                selected_indices
            ]
        else:
            probability_tensor = torch.as_tensor(
                probabilities,
                dtype=torch.float32,
                device=self.device,
            )
            selected_actions = torch.sum(
                action_heads * probability_tensor[:, :, None],
                dim=1,
            )
            reported_horizons = np.sum(
                probabilities
                * np.asarray(self.config.horizons, dtype=float)[None, :],
                axis=-1,
            )
        return selected_actions, reported_horizons, probabilities

    def _gate_q_heads(
        self,
        state_goal: torch.Tensor,
        action_heads: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action_heads is None:
            action_heads = self._actor_heads(state_goal)
        return self._policy_q_heads(
            state_goal,
            action_heads.detach(),
            target=True,
        )

    def train_step(self, batch: TransitionBatch, rng: np.random.Generator) -> TrainResult:
        state_goal, actor_state_goal, actions = self._batch_tensors(batch)
        with torch.no_grad():
            target_q = self._min_q(state_goal, actions, target=True)
        values = self.value(state_goal)
        value_loss = _expectile_loss(target_q - values, self.config.expectile)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()

        critic_target = self._critic_target(batch)
        q1, q2 = self._qs(state_goal, actions)
        critic1_loss = F.mse_loss(q1, critic_target)
        critic2_loss = F.mse_loss(q2, critic_target)
        if self.config.cross_horizon_consistency_weight > 0.0:
            consistency1_loss, consistency_coverage = (
                self._cross_horizon_consistency_loss(
                    q1,
                    critic_target,
                    batch,
                )
            )
            consistency2_loss, _ = self._cross_horizon_consistency_loss(
                q2,
                critic_target,
                batch,
            )
        else:
            consistency1_loss = q1.new_zeros(())
            consistency2_loss = q2.new_zeros(())
            consistency_coverage = 0.0
        consistency_loss = consistency1_loss + consistency2_loss
        critic_loss = (
            critic1_loss
            + critic2_loss
            + self.config.cross_horizon_consistency_weight
            * consistency_loss
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        gate_action_heads = self._actor_heads(actor_state_goal)
        target_q1, target_q2 = self._gate_q_heads(
            actor_state_goal,
            gate_action_heads,
        )
        selected_indices, gate_probabilities, gate_metrics = self._select_horizon(
            target_q1,
            target_q2,
            rng=rng,
        )
        predicted_action_heads = self._actor_heads(actor_state_goal)
        behavior_loss_by_horizon = torch.mean(
            (predicted_action_heads - actions[:, None, :]).pow(2),
            dim=-1,
        )
        selected_index_tensor = torch.as_tensor(selected_indices, dtype=torch.long, device=self.device)
        horizon_probabilities = torch.as_tensor(
            gate_probabilities,
            dtype=torch.float32,
            device=self.device,
        )
        row_indices = torch.arange(batch.size, device=self.device)
        q_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        bc_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        if self.config.actor_loss_mode == "awr":
            with torch.no_grad():
                all_advantages = (
                    self._min_q(actor_state_goal, actions, target=True)
                    - self.value(actor_state_goal)
                )
                all_weights = torch.clamp(
                    torch.exp(self.config.advantage_temperature * all_advantages),
                    max=self.config.advantage_clip,
                )
            if self.config.actor_horizon_weighting == "selected":
                weights = all_weights[row_indices, selected_index_tensor]
                behavior_loss = behavior_loss_by_horizon[row_indices, selected_index_tensor]
            else:
                weights = torch.sum(all_weights * horizon_probabilities, dim=-1)
                behavior_loss = torch.sum(
                    behavior_loss_by_horizon * horizon_probabilities,
                    dim=-1,
                )
            actor_loss = torch.mean(weights * behavior_loss)
        else:
            critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            try:
                q_action_heads = torch.clamp(predicted_action_heads, -1.0, 1.0)
                actor_q1, actor_q2 = self._policy_q_heads(
                    actor_state_goal,
                    q_action_heads,
                    target=False,
                )
                actor_q = torch.minimum(actor_q1, actor_q2)
                bc_by_horizon = 0.5 * torch.sum(
                    (predicted_action_heads - actions[:, None, :]).pow(2),
                    dim=-1,
                )
                if self.config.actor_horizon_weighting == "selected":
                    weighted_q = actor_q[row_indices, selected_index_tensor]
                    weighted_bc = bc_by_horizon[row_indices, selected_index_tensor]
                else:
                    weighted_q = torch.sum(
                        actor_q * horizon_probabilities,
                        dim=-1,
                    )
                    weighted_bc = torch.sum(
                        bc_by_horizon * horizon_probabilities,
                        dim=-1,
                    )
                q_loss = -weighted_q.mean() / (
                    weighted_q.abs().mean().detach() + 1e-6
                )
                bc_loss = weighted_bc.mean()
                actor_loss = q_loss + self.config.actor_alpha * bc_loss
                weights = torch.ones_like(weighted_q)
            finally:
                for parameter in critic_parameters:
                    parameter.requires_grad_(True)
        goal_direction_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        if self.config.goal_direction_loss_weight > 0.0:
            direction_targets = self._goal_direction_targets(batch)
            goal_losses = torch.mean(
                (predicted_action_heads - direction_targets[:, None, :]).pow(2),
                dim=-1,
            )
            if self.config.actor_horizon_weighting == "selected":
                row_indices = torch.arange(batch.size, device=self.device)
                goal_direction_loss = goal_losses[row_indices, selected_index_tensor].mean()
            else:
                goal_direction_loss = torch.sum(goal_losses * horizon_probabilities, dim=-1).mean()
        actor_loss = actor_loss + self.config.goal_direction_loss_weight * goal_direction_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update_target()
        self.step += 1
        metrics = {
            "adaptive_iql_actor_loss": float(actor_loss.detach().cpu()),
            "adaptive_iql_critic_loss": float(critic_loss.detach().cpu()),
            "adaptive_iql_critic1_loss": float(critic1_loss.detach().cpu()),
            "adaptive_iql_critic2_loss": float(critic2_loss.detach().cpu()),
            "adaptive_iql_cross_horizon_consistency_loss": float(
                consistency_loss.detach().cpu()
            ),
            "adaptive_iql_cross_horizon_consistency_coverage": float(
                consistency_coverage
            ),
            "adaptive_iql_value_loss": float(value_loss.detach().cpu()),
            "adaptive_iql_weight_mean": float(weights.mean().detach().cpu()),
            "adaptive_iql_weight_max": float(weights.max().detach().cpu()),
            "adaptive_iql_goal_direction_loss": float(goal_direction_loss.detach().cpu()),
            "adaptive_iql_actor_q_loss": float(q_loss.detach().cpu()),
            "adaptive_iql_actor_bc_loss": float(bc_loss.detach().cpu()),
        }
        metrics.update(gate_metrics)
        return TrainResult(step=self.step, metrics=metrics)

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.mul_(1.0 - self.config.target_tau)
                target_param.add_(self.config.target_tau * source_param)
            for target_param, source_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.mul_(1.0 - self.config.target_tau)
                target_param.add_(self.config.target_tau * source_param)

    def predict(self, observations: np.ndarray, goals: Optional[np.ndarray] = None) -> np.ndarray:
        if goals is None:
            raise ValueError("goals are required for TorchAdaptiveIQLAgent predictions.")
        state_goal = self._state_goal(observations, goals)
        self.actor.eval()
        with torch.no_grad():
            actions, _, _ = self._select_policy_actions(state_goal)
            actions = actions.cpu().numpy()
        self.actor.train()
        return actions

    def predict_with_info(
        self,
        observations: np.ndarray,
        goals: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if goals is None:
            raise ValueError("goals are required for TorchAdaptiveIQLAgent predictions.")
        state_goal = self._state_goal(observations, goals)
        self.actor.eval()
        with torch.no_grad():
            actions, horizons, probabilities = self._select_policy_actions(state_goal)
        self.actor.train()
        return actions.cpu().numpy(), {
            "selected_horizon": horizons,
            "horizon_probabilities": probabilities,
            "horizon_candidates": np.asarray(self.config.horizons, dtype=float),
        }

    def evaluate_batch(self, batch: TransitionBatch) -> EvaluationResult:
        if batch.goals is None:
            raise ValueError("TorchAdaptiveIQLAgent requires goal-conditioned batches.")
        state_goal, _, actions = self._batch_tensors(batch)
        predictions = self.predict(batch.observations, batch.goals)
        action_mse = float(np.mean((predictions - batch.actions) ** 2))
        with torch.no_grad():
            q1, q2 = self._qs(state_goal, actions)
            values, uncertainties = self._horizon_selection_inputs(q1, q2)
            if self.config.support_temperature is not None:
                evaluation_probabilities = self._support_probabilities(
                    uncertainties
                )
                selected_indices = np.argmax(
                    evaluation_probabilities,
                    axis=-1,
                ).astype(int)
            elif self.config.static_horizon_weights is None:
                selected_indices, evaluation_probabilities = self.gate.select_batch(
                    values,
                    uncertainties,
                )
            else:
                evaluation_probabilities = np.repeat(
                    np.asarray(
                        self.config.static_horizon_weights,
                        dtype=np.float32,
                    )[None, :],
                    values.shape[0],
                    axis=0,
                )
                selected_indices = np.full(
                    values.shape[0],
                    int(np.argmax(self.config.static_horizon_weights)),
                    dtype=int,
                )
            selected_horizons = np.asarray(self.config.horizons)[selected_indices]
            q_values = torch.minimum(q1, q2)
        metrics = {
            "eval_action_mse": action_mse,
            "eval_q_mean": float(q_values.mean().cpu()),
            "eval_q1_mean": float(q1.mean().cpu()),
            "eval_q2_mean": float(q2.mean().cpu()),
            "selected_horizon": float(np.mean(selected_horizons)),
            "selected_horizon_index": float(np.mean(selected_indices)),
            "chunk_size": float(self.config.chunk_size),
        }
        for horizon_index, horizon in enumerate(self.config.horizons):
            metrics[f"selected_horizon_fraction_{horizon}"] = float(
                np.mean(selected_indices == horizon_index)
            )
            metrics[f"horizon_probability_{horizon}"] = float(
                np.mean(evaluation_probabilities[:, horizon_index])
            )
            metrics[f"horizon_probability_std_{horizon}"] = float(
                np.std(evaluation_probabilities[:, horizon_index])
            )
        return EvaluationResult(
            metrics=metrics
        )

    def state_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "goal_dim": self.goal_dim,
            "step": self.step,
            "observation_mean": self.observation_mean.detach().cpu(),
            "observation_std": self.observation_std.detach().cpu(),
            "goal_mean": self.goal_mean.detach().cpu(),
            "goal_std": self.goal_std.detach().cpu(),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "value": self.value.state_dict(),
            "gate": self.gate.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["observation_dim"]) != self.observation_dim:
            raise ValueError("Checkpoint observation_dim does not match this agent.")
        if int(state["action_dim"]) != self.action_dim:
            raise ValueError("Checkpoint action_dim does not match this agent.")
        if int(state["goal_dim"]) != self.goal_dim:
            raise ValueError("Checkpoint goal_dim does not match this agent.")
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.target_critic1.load_state_dict(state["target_critic1"])
        self.target_critic2.load_state_dict(state["target_critic2"])
        self.value.load_state_dict(state["value"])
        self.gate.load_state_dict(state["gate"])
        self.observation_mean = state["observation_mean"].to(self.device)
        self.observation_std = state["observation_std"].to(self.device)
        self.goal_mean = state["goal_mean"].to(self.device)
        self.goal_std = state["goal_std"].to(self.device)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.value_optimizer.load_state_dict(state["value_optimizer"])
        self.step = int(state["step"])

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: Path) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))


def make_torch_adaptive_iql_agent(batch: TransitionBatch, config: Optional[TorchAdaptiveIQLConfig] = None) -> TorchAdaptiveIQLAgent:
    if batch.goals is None:
        raise ValueError("Adaptive IQL requires goal-conditioned batches.")
    config = config or TorchAdaptiveIQLConfig()
    observation_mean, observation_std = _normalization_stats(batch.observations)
    goal_mean, goal_std = _normalization_stats(batch.goals)
    return TorchAdaptiveIQLAgent(
        observation_dim=int(batch.observations.shape[1]),
        action_dim=int(batch.actions.shape[1]),
        goal_dim=int(batch.goals.shape[1]),
        config=config,
        observation_mean=observation_mean if config.normalize_inputs else None,
        observation_std=observation_std if config.normalize_inputs else None,
        goal_mean=goal_mean if config.normalize_inputs else None,
        goal_std=goal_std if config.normalize_inputs else None,
    )
