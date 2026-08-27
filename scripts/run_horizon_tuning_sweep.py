"""Run sparse OGBench adaptive-horizon tuning sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from adaptive_gcrl.evaluation.run_logs import final_metric, metric_series


def run_label(mode: str, penalty: float, seed: int) -> str:
    penalty_text = str(penalty).replace(".", "p")
    return f"{mode}_penalty_{penalty_text}_seed_{seed}"


def build_train_command(
    *,
    mode: str,
    penalty: float,
    seed: int,
    output_root: Path,
    steps: int,
    rollout_episodes: int,
    rollout_max_steps: int,
) -> tuple[str, Path, list[str]]:
    label = run_label(mode, penalty, seed)
    output_dir = output_root / label
    command = [
        sys.executable,
        "scripts/train.py",
        "--config",
        "configs/experiment/ogbench_pointmaze_smoke.yaml",
        "--config",
        "configs/env/ogbench_sparse_smoke.yaml",
        "--config",
        "configs/algo/adaptive_iql_sparse_tuned.yaml",
        "--output-dir",
        str(output_dir),
        "--set",
        f"seed={seed}",
        "--set",
        f"output_dir={output_dir}",
        "--set",
        f"steps={steps}",
        "--set",
        f"rollout_episodes={rollout_episodes}",
        "--set",
        f"rollout_max_steps={rollout_max_steps}",
        "--set",
        f"horizon_value_mode={mode}",
        "--set",
        f"horizon_penalty={penalty}",
        "--set",
        "save_checkpoint=false",
    ]
    return label, output_dir, command


def summarize_run(label: str, mode: str, penalty: float, seed: int, metrics_path: Path) -> dict[str, Any]:
    horizons = metric_series(metrics_path, "selected_horizon")
    target_horizons = metric_series(metrics_path, "target_horizon")
    histogram: dict[str, int] = {}
    for horizon in horizons:
        key = str(int(horizon))
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "label": label,
        "mode": mode,
        "penalty": penalty,
        "seed": seed,
        "eval_action_mse": final_metric(metrics_path, "eval_action_mse", event="eval"),
        "rollout_success_rate": final_metric(metrics_path, "rollout_success_rate", event="rollout_eval"),
        "selected_horizon_mean": None if not horizons else sum(horizons) / len(horizons),
        "target_horizon_mean": None if not target_horizons else sum(target_horizons) / len(target_horizons),
        "selected_horizon_histogram": histogram,
        "metrics_path": str(metrics_path),
    }


def aggregate_by_setting(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["mode"]), float(row["penalty"])), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (mode, penalty), group in sorted(grouped.items()):
        aggregates.append(
            {
                "mode": mode,
                "penalty": penalty,
                "runs": len(group),
                "eval_action_mse_mean": sum(float(row["eval_action_mse"]) for row in group) / len(group),
                "rollout_success_rate_mean": sum(float(row["rollout_success_rate"]) for row in group) / len(group),
                "selected_horizon_mean": sum(float(row["selected_horizon_mean"]) for row in group) / len(group),
                "target_horizon_mean": sum(float(row["target_horizon_mean"]) for row in group) / len(group),
            }
        )
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "mode",
        "penalty",
        "seed",
        "eval_action_mse",
        "rollout_success_rate",
        "selected_horizon_mean",
        "target_horizon_mean",
        "metrics_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=["sqrt_horizon"])
    parser.add_argument("--penalties", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.65, 0.85, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--rollout-episodes", type=int, default=2)
    parser.add_argument("--rollout-max-steps", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=Path("runs/ogbench-horizon-tuning"))
    parser.add_argument("--csv-output", type=Path, default=Path("paper/tables/ogbench_horizon_tuning.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("paper/tables/ogbench_horizon_tuning.json"))
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_specs: list[tuple[str, str, float, int, Path]] = []
    for mode in args.modes:
        for penalty in args.penalties:
            for seed in args.seeds:
                label, output_dir, command = build_train_command(
                    mode=mode,
                    penalty=penalty,
                    seed=seed,
                    output_root=args.output_root,
                    steps=args.steps,
                    rollout_episodes=args.rollout_episodes,
                    rollout_max_steps=args.rollout_max_steps,
                )
                if not args.skip_training:
                    subprocess.run(command, check=True)
                run_specs.append((label, mode, penalty, seed, output_dir / "metrics.jsonl"))

    rows = [summarize_run(label, mode, penalty, seed, path) for label, mode, penalty, seed, path in run_specs]
    summary = {"runs": rows, "by_setting": aggregate_by_setting(rows)}
    write_csv(args.csv_output, rows)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

