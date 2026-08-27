"""Evaluate learned, static-mixture, and forced-H8 checkpoint policies."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from adaptive_gcrl.algorithms.factory import create_agent
from adaptive_gcrl.data.ogbench import OGBenchSpec, load_ogbench_env
from adaptive_gcrl.evaluation.rollouts import evaluate_goal_conditioned_policy
from adaptive_gcrl.training.checkpoints import load_agent_checkpoint
from adaptive_gcrl.utils.config import load_config_files

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_training_batch


STATIC_WEIGHTS = np.asarray([0.05, 0.05, 0.05, 0.85], dtype=np.float32)
TASK_IDS = (1, 2, 3, 4, 5)


class CounterfactualPolicy:
    def __init__(
        self,
        agent,
        *,
        horizon: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        if (horizon is None) == (weights is None):
            raise ValueError("Specify exactly one of horizon or weights.")
        self.agent = agent
        self.horizon = horizon
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)
        if self.weights is not None:
            if self.weights.shape != (agent.num_horizons,):
                raise ValueError("weights must match the number of horizon heads.")
            if not np.isclose(self.weights.sum(), 1.0):
                raise ValueError("weights must sum to one.")

    def predict(
        self,
        observations: np.ndarray,
        goals: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if goals is None:
            raise ValueError("goals are required.")
        with torch.no_grad():
            heads = self.agent._actor_heads(
                self.agent._state_goal(observations, goals)
            )
            if self.horizon is not None:
                index = self.agent.config.horizons.index(self.horizon)
                actions = heads[:, index]
            else:
                weights = torch.as_tensor(
                    self.weights,
                    dtype=torch.float32,
                    device=self.agent.device,
                )
                actions = torch.sum(heads * weights[None, :, None], dim=1)
        return actions.cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-zero-root",
        type=Path,
        default=Path("corrected-seed0-results"),
    )
    parser.add_argument(
        "--multiseed-root",
        type=Path,
        default=Path("corrected-multiseed-results"),
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/tables/tempostitch_counterfactual.csv"),
    )
    return parser.parse_args()


def checkpoint_path(seed: int, seed_zero_root: Path, multiseed_root: Path) -> Path:
    root = seed_zero_root if seed == 0 else multiseed_root
    return (
        root
        / "runs"
        / "full"
        / "adaptive-corrected"
        / f"seed_{seed}"
        / "agent.pt"
    )


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive.")
    config = load_config_files(
        [
            Path("configs/experiment/ogbench_full.yaml"),
            Path("configs/env/ogbench_official_goals.yaml"),
            Path("configs/algo/adaptive_iql_official_matched.yaml"),
        ]
    )
    config["device"] = "cpu"
    batch = load_training_batch(config, seed=0)
    env = load_ogbench_env(
        OGBenchSpec(
            task=str(config["task"]),
            dataset=str(config["dataset"]),
            observation_mode=str(config["observation_mode"]),
        )
    )
    rows: list[dict[str, float | int | str]] = []
    for seed in range(5):
        agent = create_agent(config, batch)
        load_agent_checkpoint(
            agent,
            checkpoint_path(seed, args.seed_zero_root, args.multiseed_root),
        )
        policies = {
            "learned_gate": agent,
            "static_mixture": CounterfactualPolicy(
                agent,
                weights=STATIC_WEIGHTS,
            ),
            "forced_h8": CounterfactualPolicy(agent, horizon=8),
        }
        for mode, policy in policies.items():
            summary = evaluate_goal_conditioned_policy(
                env,
                policy,
                episodes=args.episodes,
                seed=20000 + seed * len(TASK_IDS) * args.episodes,
                max_steps=int(config["rollout_max_steps"]),
                task_ids=TASK_IDS,
            )
            metrics = summary.as_metrics()
            row: dict[str, float | int | str] = {
                "seed": seed,
                "mode": mode,
                "success_rate": metrics["rollout_success_rate"],
            }
            for task_id in TASK_IDS:
                row[f"task_{task_id}_success_rate"] = metrics[
                    f"rollout_task_{task_id}_success_rate"
                ]
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "mode",
        "success_rate",
        *(f"task_{task_id}_success_rate" for task_id in TASK_IDS),
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
