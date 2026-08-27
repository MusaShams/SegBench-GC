"""PyTorch goal-conditioned IQL implementation."""

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
from adaptive_gcrl.data.replay_buffer import TransitionBatch


@dataclass(frozen=True)
class TorchIQLConfig:
    learning_rate: float = 1e-3
    hidden_dim: int = 256
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
    normalize_inputs: bool = False
    actor_output_activation: str = "identity"
    goal_direction_loss_weight: float = 0.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1].")
        if not 0.0 < self.expectile < 1.0:
            raise ValueError("expectile must be in (0, 1).")
        if self.advantage_temperature <= 0.0:
            raise ValueError("advantage_temperature must be positive.")
        if self.advantage_clip <= 0.0:
            raise ValueError("advantage_clip must be positive.")
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
        if self.actor_output_activation not in {"identity", "tanh"}:
            raise ValueError("actor_output_activation must be 'identity' or 'tanh'.")
        if self.goal_direction_loss_weight < 0.0:
            raise ValueError("goal_direction_loss_weight must be non-negative.")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        *,
        hidden_layers: int = 2,
        activation: str = "relu",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if hidden_layers <= 0:
            raise ValueError("hidden_layers must be positive.")
        activation_type: type[nn.Module] = nn.ReLU if activation == "relu" else nn.GELU
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_type())
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _expectile_loss(residuals: torch.Tensor, expectile: float) -> torch.Tensor:
    weights = torch.where(residuals >= 0.0, expectile, 1.0 - expectile)
    return torch.mean(weights * residuals.pow(2))


def _as_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


class TorchIQLAgent:
    """Deterministic actor-critic IQL agent for continuous goal-conditioned batches."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        goal_dim: int,
        config: Optional[TorchIQLConfig] = None,
        observation_mean: Optional[np.ndarray] = None,
        observation_std: Optional[np.ndarray] = None,
        goal_mean: Optional[np.ndarray] = None,
        goal_std: Optional[np.ndarray] = None,
    ) -> None:
        if observation_dim <= 0 or action_dim <= 0 or goal_dim <= 0:
            raise ValueError("observation_dim, action_dim, and goal_dim must be positive.")
        self.config = config or TorchIQLConfig()
        self.device = torch.device(self.config.device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
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
        self.actor = MLP(state_goal_dim, action_dim, self.config.hidden_dim, **network_kwargs).to(self.device)
        self.critic1 = MLP(q_input_dim, 1, self.config.hidden_dim, **network_kwargs).to(self.device)
        self.critic2 = MLP(q_input_dim, 1, self.config.hidden_dim, **network_kwargs).to(self.device)
        self.target_critic1 = MLP(q_input_dim, 1, self.config.hidden_dim, **network_kwargs).to(self.device)
        self.target_critic2 = MLP(q_input_dim, 1, self.config.hidden_dim, **network_kwargs).to(self.device)
        self.value = MLP(state_goal_dim, 1, self.config.hidden_dim, **network_kwargs).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if batch.goals is None:
            raise ValueError("TorchIQLAgent requires goal-conditioned batches.")
        state_goal = self._state_goal(batch.observations, batch.goals)
        actor_goals = batch.goals if batch.actor_goals is None else batch.actor_goals
        actor_state_goal = self._state_goal(batch.observations, actor_goals)
        next_state_goal = self._state_goal(batch.next_observations, batch.goals)
        actions = _as_tensor(batch.actions, self.device)
        rewards = _as_tensor(batch.rewards.reshape(-1, 1), self.device)
        masks = (
            1.0 - np.asarray(batch.terminals, dtype=np.float32)
            if batch.masks is None
            else np.asarray(batch.masks, dtype=np.float32)
        )
        mask_tensor = _as_tensor(masks.reshape(-1, 1), self.device)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"actions must have shape (batch, {self.action_dim}).")
        return state_goal, actor_state_goal, actions, rewards, next_state_goal, mask_tensor

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

    def _actor(self, state_goal: torch.Tensor) -> torch.Tensor:
        actions = self.actor(state_goal)
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
        _, _, _, rewards, next_state_goal, masks = self._batch_tensors(batch)
        with torch.no_grad():
            return rewards + self.config.discount * masks * self.value(next_state_goal)

    def train_step(self, batch: TransitionBatch, rng: np.random.Generator) -> TrainResult:
        del rng
        state_goal, actor_state_goal, actions, _, _, _ = self._batch_tensors(batch)

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
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        predicted_actions = self._actor(actor_state_goal)
        goal_direction_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        if self.config.goal_direction_loss_weight > 0.0:
            goal_direction_loss = F.mse_loss(predicted_actions, self._goal_direction_targets(batch))
        q_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        bc_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        if self.config.actor_loss_mode == "awr":
            with torch.no_grad():
                advantages = (
                    self._min_q(actor_state_goal, actions, target=True)
                    - self.value(actor_state_goal)
                )
                weights = torch.clamp(
                    torch.exp(self.config.advantage_temperature * advantages),
                    max=self.config.advantage_clip,
                )
            behavior_loss = torch.mean(
                (predicted_actions - actions).pow(2),
                dim=-1,
                keepdim=True,
            )
            actor_loss = torch.mean(weights * behavior_loss)
        else:
            critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            try:
                q_actions = torch.clamp(predicted_actions, -1.0, 1.0)
                actor_q = self._min_q(actor_state_goal, q_actions)
                q_loss = -actor_q.mean() / (actor_q.abs().mean().detach() + 1e-6)
                bc_loss = 0.5 * torch.sum(
                    (predicted_actions - actions).pow(2),
                    dim=-1,
                ).mean()
                actor_loss = q_loss + self.config.actor_alpha * bc_loss
                weights = torch.ones_like(actor_q)
            finally:
                for parameter in critic_parameters:
                    parameter.requires_grad_(True)
        actor_loss = actor_loss + self.config.goal_direction_loss_weight * goal_direction_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update_target()
        self.step += 1
        return TrainResult(
            step=self.step,
            metrics={
                "iql_actor_loss": float(actor_loss.detach().cpu()),
                "iql_critic_loss": float(critic_loss.detach().cpu()),
                "iql_critic1_loss": float(critic1_loss.detach().cpu()),
                "iql_critic2_loss": float(critic2_loss.detach().cpu()),
                "iql_value_loss": float(value_loss.detach().cpu()),
                "iql_advantage_weight_mean": float(weights.mean().detach().cpu()),
                "iql_goal_direction_loss": float(goal_direction_loss.detach().cpu()),
                "iql_actor_q_loss": float(q_loss.detach().cpu()),
                "iql_actor_bc_loss": float(bc_loss.detach().cpu()),
            },
        )

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
            raise ValueError("goals are required for TorchIQLAgent predictions.")
        self.actor.eval()
        with torch.no_grad():
            actions = self._actor(self._state_goal(observations, goals)).cpu().numpy()
        self.actor.train()
        return actions

    def evaluate_batch(self, batch: TransitionBatch) -> EvaluationResult:
        if batch.goals is None:
            raise ValueError("TorchIQLAgent requires goal-conditioned batches.")
        predictions = self.predict(batch.observations, batch.goals)
        action_mse = float(np.mean((predictions - batch.actions) ** 2))
        state_goal = self._state_goal(batch.observations, batch.goals)
        actions = _as_tensor(batch.actions, self.device)
        with torch.no_grad():
            q_mean = float(self._min_q(state_goal, actions).mean().cpu())
            q1_mean = float(self._qs(state_goal, actions)[0].mean().cpu())
            q2_mean = float(self._qs(state_goal, actions)[1].mean().cpu())
            value_mean = float(self.value(state_goal).mean().cpu())
        return EvaluationResult(
            metrics={
                "eval_action_mse": action_mse,
                "eval_q_mean": q_mean,
                "eval_q1_mean": q1_mean,
                "eval_q2_mean": q2_mean,
                "eval_value_mean": value_mean,
            }
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


def make_torch_iql_agent(batch: TransitionBatch, config: Optional[TorchIQLConfig] = None) -> TorchIQLAgent:
    if batch.goals is None:
        raise ValueError("Torch IQL requires goal-conditioned batches.")
    config = config or TorchIQLConfig()
    observation_mean, observation_std = _normalization_stats(batch.observations)
    goal_mean, goal_std = _normalization_stats(batch.goals)
    return TorchIQLAgent(
        observation_dim=int(batch.observations.shape[1]),
        action_dim=int(batch.actions.shape[1]),
        goal_dim=int(batch.goals.shape[1]),
        config=config,
        observation_mean=observation_mean if config.normalize_inputs else None,
        observation_std=observation_std if config.normalize_inputs else None,
        goal_mean=goal_mean if config.normalize_inputs else None,
        goal_std=goal_std if config.normalize_inputs else None,
    )


def _normalization_stats(array: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(array, dtype=np.float32)
    return np.mean(values, axis=0), np.maximum(np.std(values, axis=0), eps)
