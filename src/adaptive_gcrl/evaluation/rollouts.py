"""Environment rollout evaluation for goal-conditioned policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import numpy as np


class GoalConditionedPolicy(Protocol):
    def predict(self, observations: np.ndarray, goals: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict actions for a batch of observations and goals."""


@dataclass(frozen=True)
class RolloutSummary:
    episodes: int
    return_mean: float
    return_std: float
    success_rate: float
    length_mean: float
    selected_horizon_mean: Optional[float] = None
    task_success_rates: Optional[dict[int, float]] = None
    task_selected_horizon_means: Optional[dict[int, float]] = None
    task_horizon_probability_means: Optional[dict[int, dict[int, float]]] = None

    def as_metrics(self, prefix: str = "rollout") -> dict[str, float]:
        metrics = {
            f"{prefix}_episodes": float(self.episodes),
            f"{prefix}_return_mean": self.return_mean,
            f"{prefix}_return_std": self.return_std,
            f"{prefix}_success_rate": self.success_rate,
            f"{prefix}_length_mean": self.length_mean,
        }
        if self.selected_horizon_mean is not None:
            metrics[f"{prefix}_selected_horizon_mean"] = self.selected_horizon_mean
        if self.task_success_rates is not None:
            for task_id, success_rate in sorted(self.task_success_rates.items()):
                metrics[f"{prefix}_task_{task_id}_success_rate"] = success_rate
        if self.task_selected_horizon_means is not None:
            for task_id, horizon_mean in sorted(self.task_selected_horizon_means.items()):
                metrics[f"{prefix}_task_{task_id}_selected_horizon_mean"] = horizon_mean
        if self.task_horizon_probability_means is not None:
            for task_id, probabilities in sorted(
                self.task_horizon_probability_means.items()
            ):
                for horizon, probability in sorted(probabilities.items()):
                    metrics[
                        f"{prefix}_task_{task_id}_horizon_{horizon}_probability"
                    ] = probability
        return metrics


def _reset_env(
    env: Any,
    seed: int,
    task_id: Optional[int] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    reset_kwargs: dict[str, Any] = {"seed": seed}
    if task_id is not None:
        reset_kwargs["options"] = {"task_id": task_id}
    reset_result = env.reset(**reset_kwargs)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        observation, info = reset_result
    else:
        observation, info = reset_result, {}
    if not isinstance(info, dict):
        raise TypeError("Environment reset info must be a dictionary.")
    return np.asarray(observation, dtype=float), info


def _step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    step_result = env.step(action)
    if not isinstance(step_result, tuple):
        raise TypeError("Environment step result must be a tuple.")
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
    elif len(step_result) == 4:
        observation, reward, done, info = step_result
    else:
        raise ValueError("Environment step result must have length 4 or 5.")
    if not isinstance(info, dict):
        raise TypeError("Environment step info must be a dictionary.")
    return np.asarray(observation, dtype=float), float(reward), bool(done), info


def evaluate_goal_conditioned_policy(
    env: Any,
    policy: GoalConditionedPolicy,
    *,
    episodes: int,
    seed: int,
    max_steps: Optional[int] = None,
    task_ids: Optional[Sequence[int]] = None,
) -> RolloutSummary:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")

    returns: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []
    selected_horizons: list[float] = []
    task_successes: dict[int, list[float]] = {}
    task_selected_horizons: dict[int, list[float]] = {}
    task_horizon_probabilities: dict[int, list[np.ndarray]] = {}
    horizon_candidates: Optional[np.ndarray] = None
    evaluation_tasks: list[Optional[int]] = [None] if task_ids is None else list(task_ids)
    for task_index, task_id in enumerate(evaluation_tasks):
        for episode in range(episodes):
            if hasattr(policy, "reset_policy"):
                policy.reset_policy()
            episode_seed = seed + task_index * episodes + episode
            observation, reset_info = _reset_env(env, episode_seed, task_id=task_id)
            if "goal" not in reset_info:
                raise KeyError("Goal-conditioned rollout requires reset info to contain 'goal'.")
            goal = np.asarray(reset_info["goal"], dtype=float)
            total_return = 0.0
            success = 0.0
            steps = 0
            while True:
                if hasattr(policy, "predict_with_info"):
                    action_batch, policy_info = policy.predict_with_info(
                        observation.reshape(1, -1),
                        goal.reshape(1, -1),
                    )
                    action = action_batch[0]
                    if "selected_horizon" in policy_info:
                        current_horizons = [
                            float(horizon)
                            for horizon in np.asarray(
                                policy_info["selected_horizon"]
                            ).reshape(-1)
                        ]
                        selected_horizons.extend(current_horizons)
                        if task_id is not None:
                            task_selected_horizons.setdefault(task_id, []).extend(
                                current_horizons
                            )
                    if "horizon_probabilities" in policy_info:
                        if "horizon_candidates" not in policy_info:
                            raise KeyError(
                                "horizon_candidates are required with horizon_probabilities."
                            )
                        candidates = np.asarray(
                            policy_info["horizon_candidates"], dtype=float
                        ).reshape(-1)
                        probabilities = np.asarray(
                            policy_info["horizon_probabilities"], dtype=float
                        ).reshape(-1, candidates.size)
                        if horizon_candidates is None:
                            horizon_candidates = candidates
                        elif not np.array_equal(horizon_candidates, candidates):
                            raise ValueError(
                                "Horizon candidates changed during rollout evaluation."
                            )
                        if task_id is not None:
                            task_horizon_probabilities.setdefault(task_id, []).extend(
                                probabilities
                            )
                else:
                    action = policy.predict(observation.reshape(1, -1), goal.reshape(1, -1))[0]
                if hasattr(env, "action_space"):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                observation, reward, done, info = _step_env(env, action)
                total_return += reward
                success = max(success, float(info.get("success", 0.0)))
                steps += 1
                if done or (max_steps is not None and steps >= max_steps):
                    break
            returns.append(total_return)
            successes.append(success)
            lengths.append(steps)
            if task_id is not None:
                task_successes.setdefault(task_id, []).append(success)

    return RolloutSummary(
        episodes=len(returns),
        return_mean=float(np.mean(returns)),
        return_std=float(np.std(returns)),
        success_rate=float(np.mean(successes)),
        length_mean=float(np.mean(lengths)),
        selected_horizon_mean=None
        if not selected_horizons
        else float(np.mean(selected_horizons)),
        task_success_rates=None
        if not task_successes
        else {
            task_id: float(np.mean(task_values))
            for task_id, task_values in task_successes.items()
        },
        task_selected_horizon_means=None
        if not task_selected_horizons
        else {
            task_id: float(np.mean(task_values))
            for task_id, task_values in task_selected_horizons.items()
        },
        task_horizon_probability_means=None
        if not task_horizon_probabilities or horizon_candidates is None
        else {
            task_id: {
                int(horizon): float(probability)
                for horizon, probability in zip(
                    horizon_candidates,
                    np.mean(np.asarray(task_values), axis=0),
                )
            }
            for task_id, task_values in task_horizon_probabilities.items()
        },
    )
