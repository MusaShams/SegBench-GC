"""Run a causal trajectory-resegmentation pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONDITIONS = {
    "original": [],
    "robust_cut_25_offset_4": [
        "backup_segmentation_interval=25",
        "backup_segmentation_offset=4",
        "bootstrap_at_backup_boundaries=true",
    ],
    "naive_cut_25_offset_4": [
        "backup_segmentation_interval=25",
        "backup_segmentation_offset=4",
        "bootstrap_at_backup_boundaries=false",
    ],
    "robust_cut_25_offset_14": [
        "backup_segmentation_interval=25",
        "backup_segmentation_offset=14",
        "bootstrap_at_backup_boundaries=true",
    ],
    "naive_cut_25_offset_14": [
        "backup_segmentation_interval=25",
        "backup_segmentation_offset=14",
        "bootstrap_at_backup_boundaries=false",
    ],
    "robust_cut_25": [
        "backup_segmentation_interval=25",
        "backup_segmentation_offset=24",
        "bootstrap_at_backup_boundaries=true",
    ],
    "naive_cut_25": [
        "backup_segmentation_interval=25",
        "backup_segmentation_offset=24",
        "bootstrap_at_backup_boundaries=false",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/segmentation-robustness-pilot"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "paper/tables/segmentation_robustness_pilot.json"
        ),
    )
    return parser.parse_args()


def final_eval(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return next(
        row["metrics"]
        for row in reversed(rows)
        if row.get("event") == "eval"
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    results: dict[str, list[dict[str, float]]] = {
        condition: [] for condition in CONDITIONS
    }
    for seed in args.seeds:
        for condition, overrides in CONDITIONS.items():
            output_dir = args.output_root / condition / f"seed_{seed}"
            metrics_path = output_dir / "metrics.jsonl"
            if metrics_path.exists():
                try:
                    results[condition].append(
                        {
                            "seed": seed,
                            **final_eval(metrics_path),
                        }
                    )
                    continue
                except (StopIteration, ValueError):
                    pass
            command = [
                sys.executable,
                "scripts/train.py",
                "--config",
                "configs/experiment/segmentation_robustness_pilot.yaml",
                "--config",
                "configs/env/ogbench_official_goals.yaml",
                "--config",
                "configs/algo/static_mixture_iql_official_matched.yaml",
                "--output-dir",
                str(output_dir),
                "--set",
                f"seed={seed}",
                "--set",
                f"steps={args.steps}",
            ]
            for override in overrides:
                command.extend(["--set", override])
            subprocess.run(command, check=True)
            results[condition].append(
                {
                    "seed": seed,
                    **final_eval(metrics_path),
                }
            )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
