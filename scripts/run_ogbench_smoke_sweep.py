"""Run a small multi-seed OGBench smoke comparison and export summary tables."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ALGORITHMS = {
    "iql": Path("configs/algo/iql.yaml"),
    "gciql_official_matched": Path("configs/algo/gciql_official_matched.yaml"),
    "adaptive_iql": Path("configs/algo/adaptive_iql.yaml"),
    "adaptive_iql_sparse_tuned": Path("configs/algo/adaptive_iql_sparse_tuned.yaml"),
    "adaptive_iql_sparse_smoothed": Path("configs/algo/adaptive_iql_sparse_smoothed.yaml"),
    "adaptive_iql_sparse_centered": Path("configs/algo/adaptive_iql_sparse_centered.yaml"),
    "adaptive_iql_sparse_centered_weighted": Path("configs/algo/adaptive_iql_sparse_centered_weighted.yaml"),
    "adaptive_iql_official_matched": Path("configs/algo/adaptive_iql_official_matched.yaml"),
    "static_mixture_iql_official_matched": Path(
        "configs/algo/static_mixture_iql_official_matched.yaml"
    ),
    "hf_gciql_official_matched": Path(
        "configs/algo/hf_gciql_official_matched.yaml"
    ),
    "consistent_mixture_iql_official_matched": Path(
        "configs/algo/consistent_mixture_iql_official_matched.yaml"
    ),
    "adaptive_iql_sparse_centered_weighted_normalized": Path("configs/algo/adaptive_iql_sparse_centered_weighted_normalized.yaml"),
    "adaptive_iql_sparse_centered_weighted_squashed": Path("configs/algo/adaptive_iql_sparse_centered_weighted_squashed.yaml"),
    "adaptive_iql_sparse_centered_weighted_goal_directed": Path("configs/algo/adaptive_iql_sparse_centered_weighted_goal_directed.yaml"),
    "adaptive_iql_sparse_centered_weighted_chunked": Path("configs/algo/adaptive_iql_sparse_centered_weighted_chunked.yaml"),
    "fixed_horizon_iql_h1": Path("configs/algo/fixed_horizon_iql_h1.yaml"),
    "fixed_horizon_iql_h1_official_matched": Path(
        "configs/algo/fixed_horizon_iql_h1_official_matched.yaml"
    ),
    "fixed_horizon_iql_h4": Path("configs/algo/fixed_horizon_iql_h4.yaml"),
    "fixed_horizon_iql_h4_normalized": Path("configs/algo/fixed_horizon_iql_h4_normalized.yaml"),
    "fixed_horizon_iql_h4_squashed": Path("configs/algo/fixed_horizon_iql_h4_squashed.yaml"),
    "fixed_horizon_iql_h4_goal_directed": Path("configs/algo/fixed_horizon_iql_h4_goal_directed.yaml"),
    "fixed_horizon_iql_h4_chunked": Path("configs/algo/fixed_horizon_iql_h4_chunked.yaml"),
    "fixed_horizon_iql_h8": Path("configs/algo/fixed_horizon_iql_h8.yaml"),
    "fixed_horizon_iql_h8_official_matched": Path(
        "configs/algo/fixed_horizon_iql_h8_official_matched.yaml"
    ),
}


def build_train_command(
    algorithm: str,
    seed: int,
    output_root: Path,
    *,
    env_config: Path = Path("configs/env/ogbench_smoke.yaml"),
    steps: int | None = None,
    rollout_episodes: int | None = None,
    rollout_max_steps: int | None = None,
    eval_batch_size: int | None = None,
) -> list[str]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    output_dir = output_root / algorithm / f"seed_{seed}"
    command = [
        sys.executable,
        "scripts/train.py",
        "--config",
        "configs/experiment/ogbench_pointmaze_smoke.yaml",
        "--config",
        str(env_config),
        "--config",
        str(ALGORITHMS[algorithm]),
        "--set",
        f"seed={seed}",
        "--set",
        f"output_dir={output_dir}",
        "--output-dir",
        str(output_dir),
    ]
    overrides = {
        "steps": steps,
        "rollout_episodes": rollout_episodes,
        "rollout_max_steps": rollout_max_steps,
        "eval_batch_size": eval_batch_size,
    }
    for key, value in overrides.items():
        if value is not None:
            command.extend(["--set", f"{key}={value}"])
    return command


def build_export_command(
    algorithms: list[str],
    seeds: list[int],
    output_root: Path,
    csv_output: Path,
    json_output: Path,
    *,
    metric: str,
    event: str,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/export_paper_tables.py",
        "--event",
        event,
        "--metric",
        metric,
        "--csv-output",
        str(csv_output),
        "--json-output",
        str(json_output),
    ]
    for algorithm in algorithms:
        for seed in seeds:
            label = f"{algorithm}_seed_{seed}"
            path = output_root / algorithm / f"seed_{seed}" / "metrics.jsonl"
            command.extend(["--run", f"{label}={path}"])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--algorithms", choices=sorted(ALGORITHMS), nargs="+", default=["iql", "adaptive_iql"])
    parser.add_argument("--env-config", type=Path, default=Path("configs/env/ogbench_smoke.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/ogbench-pointmaze-multiseed"))
    parser.add_argument("--csv-output", type=Path, default=Path("paper/tables/ogbench_pointmaze_smoke.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("paper/tables/ogbench_pointmaze_smoke.json"))
    parser.add_argument("--rollout-csv-output", type=Path, default=Path("paper/tables/ogbench_pointmaze_rollout.csv"))
    parser.add_argument("--rollout-json-output", type=Path, default=Path("paper/tables/ogbench_pointmaze_rollout.json"))
    parser.add_argument("--steps", type=int, default=None, help="Override training steps for each run.")
    parser.add_argument("--rollout-episodes", type=int, default=None, help="Override rollout evaluation episodes.")
    parser.add_argument("--rollout-max-steps", type=int, default=None, help="Override rollout max steps per episode.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override held-out batch evaluation size.")
    parser.add_argument("--skip-training", action="store_true", help="Only export tables from existing metrics files.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not args.skip_training:
        for algorithm in args.algorithms:
            for seed in args.seeds:
                run_command(
                    build_train_command(
                        algorithm,
                        seed,
                        args.output_root,
                        env_config=args.env_config,
                        steps=args.steps,
                        rollout_episodes=args.rollout_episodes,
                        rollout_max_steps=args.rollout_max_steps,
                        eval_batch_size=args.eval_batch_size,
                    )
                )
    run_command(
        build_export_command(
            args.algorithms,
            args.seeds,
            args.output_root,
            args.csv_output,
            args.json_output,
            metric="eval_action_mse",
            event="eval",
        )
    )
    run_command(
        build_export_command(
            args.algorithms,
            args.seeds,
            args.output_root,
            args.rollout_csv_output,
            args.rollout_json_output,
            metric="rollout_success_rate",
            event="rollout_eval",
        )
    )


if __name__ == "__main__":
    main()
