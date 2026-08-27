"""Analyze paired fixed-H8 and TempoStitch results across five seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_gcrl.algorithms.factory import create_agent
from adaptive_gcrl.data.ogbench import OGBenchSpec, load_ogbench_env
from adaptive_gcrl.evaluation.rollouts import evaluate_goal_conditioned_policy
from adaptive_gcrl.training.checkpoints import load_agent_checkpoint
from adaptive_gcrl.utils.config import load_config_files

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_training_batch


METHODS = ("fixed-h8", "adaptive-corrected")
TASK_IDS = (1, 2, 3, 4, 5)


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
    parser.add_argument("--diagnostic-episodes", type=int, default=3)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("paper/tables/tempostitch_pointmaze_five_seed.csv"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("paper/tables/tempostitch_pointmaze_five_seed.json"),
    )
    return parser.parse_args()


def result_path(
    method: str,
    seed: int,
    *,
    seed_zero_root: Path,
    multiseed_root: Path,
    filename: str,
) -> Path:
    root = seed_zero_root if seed == 0 else multiseed_root
    return root / "runs" / "full" / method / f"seed_{seed}" / filename


def final_rollout_metrics(path: Path) -> dict[str, float]:
    final: dict[str, float] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("event") == "rollout_eval":
                final = {
                    key: float(value)
                    for key, value in payload["metrics"].items()
                }
    if final is None:
        raise ValueError(f"No rollout_eval event found in {path}.")
    return final


def bootstrap_paired_mean_ci(
    differences: np.ndarray,
    *,
    samples: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("differences must be a non-empty vector.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        differences.size,
        size=(samples, differences.size),
    )
    estimates = differences[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(
        np.quantile(estimates, 0.975)
    )


def summarize_results(
    results: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    scores = {
        method: np.asarray(
            [metrics["rollout_success_rate"] for metrics in results[method]],
            dtype=float,
        )
        for method in METHODS
    }
    differences = scores["adaptive-corrected"] - scores["fixed-h8"]
    ci_low, ci_high = bootstrap_paired_mean_ci(differences)
    summary: dict[str, Any] = {
        "methods": {
            method: {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "scores": values.tolist(),
            }
            for method, values in scores.items()
        },
        "paired_difference": {
            "mean": float(differences.mean()),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "differences": differences.tolist(),
            "wins": int(np.sum(differences > 0.0)),
            "ties": int(np.sum(differences == 0.0)),
        },
        "tasks": {},
    }
    for task_id in TASK_IDS:
        key = f"rollout_task_{task_id}_success_rate"
        fixed = np.asarray(
            [metrics[key] for metrics in results["fixed-h8"]],
            dtype=float,
        )
        adaptive = np.asarray(
            [metrics[key] for metrics in results["adaptive-corrected"]],
            dtype=float,
        )
        summary["tasks"][str(task_id)] = {
            "fixed_h8_mean": float(fixed.mean()),
            "tempostitch_mean": float(adaptive.mean()),
            "paired_difference_mean": float((adaptive - fixed).mean()),
        }
    return summary


def evaluate_gate_diagnostics(
    checkpoints: list[Path],
    *,
    episodes: int,
) -> list[dict[str, float]]:
    if episodes <= 0:
        raise ValueError("diagnostic episodes must be positive.")
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
    diagnostics: list[dict[str, float]] = []
    for seed, checkpoint in enumerate(checkpoints):
        agent = create_agent(config, batch)
        load_agent_checkpoint(agent, checkpoint)
        summary = evaluate_goal_conditioned_policy(
            env,
            agent,
            episodes=episodes,
            seed=10000 + seed * len(TASK_IDS) * episodes,
            max_steps=int(config["rollout_max_steps"]),
            task_ids=TASK_IDS,
        )
        diagnostics.append(summary.as_metrics())
    return diagnostics


def write_csv(
    path: Path,
    results: dict[str, list[dict[str, float]]],
    diagnostics: list[dict[str, float]],
) -> None:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed, metrics in enumerate(results[method]):
            for task_id in TASK_IDS:
                row: dict[str, Any] = {
                    "method": method,
                    "seed": seed,
                    "task_id": task_id,
                    "success_rate": metrics[
                        f"rollout_task_{task_id}_success_rate"
                    ],
                }
                if method == "adaptive-corrected":
                    diagnostic = diagnostics[seed]
                    row["selected_horizon_mean"] = diagnostic[
                        f"rollout_task_{task_id}_selected_horizon_mean"
                    ]
                    for horizon in (1, 2, 4, 8):
                        row[f"horizon_{horizon}_probability"] = diagnostic[
                            f"rollout_task_{task_id}_horizon_{horizon}_probability"
                        ]
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "seed",
        "task_id",
        "success_rate",
        "selected_horizon_mean",
        "horizon_1_probability",
        "horizon_2_probability",
        "horizon_4_probability",
        "horizon_8_probability",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    results = {
        method: [
            final_rollout_metrics(
                result_path(
                    method,
                    seed,
                    seed_zero_root=args.seed_zero_root,
                    multiseed_root=args.multiseed_root,
                    filename="metrics.jsonl",
                )
            )
            for seed in range(5)
        ]
        for method in METHODS
    }
    checkpoints = [
        result_path(
            "adaptive-corrected",
            seed,
            seed_zero_root=args.seed_zero_root,
            multiseed_root=args.multiseed_root,
            filename="agent.pt",
        )
        for seed in range(5)
    ]
    diagnostics = evaluate_gate_diagnostics(
        checkpoints,
        episodes=args.diagnostic_episodes,
    )
    payload = summarize_results(results)
    payload["gate_diagnostics"] = diagnostics
    payload["diagnostic_episodes_per_task"] = args.diagnostic_episodes
    write_csv(args.csv_output, results, diagnostics)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
