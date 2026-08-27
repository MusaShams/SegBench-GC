"""Aggregate SegBench-GC experiments into paper-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXPERIMENTS = {
    "medium_10k": {
        "root": "segbench",
        "directories": ("segmentation-10k-gate",),
        "label": "PointMaze Medium, 10k",
    },
    "medium_100k": {
        "root": "segbench",
        "directories": ("segmentation-100k-gate",),
        "label": "PointMaze Medium, 100k",
    },
    "large_10k": {
        "root": "segbench",
        "directories": ("segmentation-large-10k-gate",),
        "label": "PointMaze Large, 10k",
    },
    "large_100k": {
        "root": "segbench",
        "directories": ("segmentation-large-100k-gate",),
        "label": "PointMaze Large, 100k",
    },
    "medium_1m": {
        "root": "segbench",
        "directories": ("segmentation-medium-full-gate",),
        "label": "PointMaze Medium, 1M",
    },
    "antmaze_10k": {
        "root": "final",
        "directories": ("segmentation-antmaze-10k-gate",),
        "label": "AntMaze Medium, 10k",
    },
    "antmaze_100k": {
        "root": "final",
        "directories": ("segmentation-antmaze-100k-gate",),
        "label": "AntMaze Medium, 100k",
    },
    "antmaze_1m": {
        "root": "final",
        "directories": (
            "segmentation-antmaze-full-seed0-gate",
            "segmentation-antmaze-full-gate",
        ),
        "label": "AntMaze Medium, 1M",
    },
}
MODES = ("original", "robust", "naive")
TASK_IDS = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segbench-root",
        type=Path,
        default=Path("segbench-gc-results"),
    )
    parser.add_argument(
        "--uncut-root",
        type=Path,
        default=Path("static-mixture-five-seed-results"),
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=Path("segbench-gc-final-results"),
    )
    parser.add_argument(
        "--broader-root",
        type=Path,
        default=Path("segbench-gc-broader-results"),
    )
    parser.add_argument(
        "--random-full-root",
        type=Path,
        default=Path("segbench-random-full-results"),
    )
    parser.add_argument(
        "--fixed-h8-full-root",
        type=Path,
        default=Path("segbench-fixed-h8-full-results"),
    )
    parser.add_argument(
        "--cube-double-root",
        type=Path,
        default=Path("segbench-cube-double-results"),
    )
    parser.add_argument(
        "--corrected-seed-zero-root",
        type=Path,
        default=Path("corrected-seed0-results"),
    )
    parser.add_argument(
        "--corrected-multiseed-root",
        type=Path,
        default=Path("corrected-multiseed-results"),
    )
    parser.add_argument(
        "--final-pointmaze-root",
        type=Path,
        default=Path("segbench-final-pointmaze-controls"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/tables"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("paper/figures"),
    )
    return parser.parse_args()


def read_final_events(path: Path) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    start: dict[str, Any] | None = None
    evaluation: dict[str, float] | None = None
    rollout: dict[str, float] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            event = payload.get("event")
            if event == "train_start":
                start = payload
            elif event == "eval":
                evaluation = {
                    key: float(value)
                    for key, value in payload["metrics"].items()
                }
            elif event == "rollout_eval":
                rollout = {
                    key: float(value)
                    for key, value in payload["metrics"].items()
                }
                rollout["_event_step"] = float(payload["step"])
    if start is None or evaluation is None or rollout is None:
        raise ValueError(f"Incomplete run log: {path}")
    return start, evaluation, rollout


def condition_mode(condition: str) -> str:
    for mode in MODES:
        if condition == mode or condition.startswith(f"{mode}-"):
            return mode
    raise ValueError(f"Unknown SegBench-GC condition: {condition}")


def update_step_provenance(
    start: dict[str, Any],
    rollout: dict[str, float],
) -> dict[str, int | bool]:
    invocation_steps = int(start["config"]["steps"])
    total_steps = int(rollout["_event_step"])
    initial_steps = total_steps - invocation_steps
    if initial_steps < 0:
        raise ValueError(
            "Total update step cannot be smaller than invocation steps."
        )
    resumed = initial_steps > 0
    return {
        "initial_update_steps": initial_steps,
        "invocation_update_steps": invocation_steps,
        "resumed_update_steps": invocation_steps if resumed else 0,
        "total_update_steps": total_steps,
        "resumed_from_checkpoint": resumed,
    }


def collect_runs(
    segbench_root: Path,
    uncut_root: Path,
    final_root: Path,
    final_pointmaze_root: Path,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for experiment, spec in EXPERIMENTS.items():
        root = segbench_root if spec["root"] == "segbench" else final_root
        for directory in spec["directories"]:
            experiment_root = root / "runs" / directory
            for path in sorted(
                experiment_root.glob("*/seed_*/metrics.jsonl")
            ):
                condition = path.parent.parent.name
                seed = int(path.parent.name.removeprefix("seed_"))
                start, evaluation, rollout = read_final_events(path)
                runs.append(
                    {
                        "experiment": experiment,
                        "experiment_label": spec["label"],
                        "condition": condition,
                        "mode": condition_mode(condition),
                        "seed": seed,
                        "task": start["config"]["task"],
                        **update_step_provenance(start, rollout),
                        "backup_boundary_count": int(
                            start.get("backup_boundary_count", 0)
                        ),
                        "success_rate": rollout["rollout_success_rate"],
                        "action_mse": evaluation["eval_action_mse"],
                        **{
                            f"task_{task_id}_success_rate": rollout[
                                f"rollout_task_{task_id}_success_rate"
                            ]
                            for task_id in TASK_IDS
                        },
                        "path": str(path),
                    }
                )

    runs = [
        run for run in runs if run["experiment"] != "medium_1m"
    ]
    pointmaze_root = (
        final_pointmaze_root
        / "runs"
        / "segbench-final-pointmaze-controls"
    )
    for seed in range(5):
        path = (
            pointmaze_root
            / "original"
            / f"seed_{seed}"
            / "metrics.jsonl"
        )
        start, evaluation, rollout = read_final_events(path)
        runs.append(
            {
                "experiment": "medium_1m",
                "experiment_label": (
                    EXPERIMENTS["medium_1m"]["label"]
                ),
                "condition": "original",
                "mode": "original",
                "seed": seed,
                "task": start["config"]["task"],
                **update_step_provenance(start, rollout),
                "backup_boundary_count": int(
                    start["backup_boundary_count"]
                ),
                "success_rate": rollout["rollout_success_rate"],
                "action_mse": evaluation["eval_action_mse"],
                **{
                    f"task_{task_id}_success_rate": rollout[
                        f"rollout_task_{task_id}_success_rate"
                    ]
                    for task_id in TASK_IDS
                },
                "path": str(path),
            }
        )
    for mode, directory in (
        ("robust", "robust-offset24"),
        ("naive", "naive-offset24"),
    ):
        for seed in range(5):
            if seed < 3:
                path = (
                    segbench_root
                    / "runs"
                    / "segmentation-medium-full-gate"
                    / f"{mode}-offset24"
                    / f"seed_{seed}"
                    / "metrics.jsonl"
                )
            else:
                path = (
                    pointmaze_root
                    / directory
                    / f"seed_{seed}"
                    / "metrics.jsonl"
                )
            start, evaluation, rollout = read_final_events(path)
            runs.append(
                {
                    "experiment": "medium_1m",
                    "experiment_label": (
                        EXPERIMENTS["medium_1m"]["label"]
                    ),
                    "condition": directory,
                    "mode": mode,
                    "seed": seed,
                    "task": start["config"]["task"],
                    **update_step_provenance(start, rollout),
                    "backup_boundary_count": int(
                        start["backup_boundary_count"]
                    ),
                    "success_rate": rollout[
                        "rollout_success_rate"
                    ],
                    "action_mse": evaluation["eval_action_mse"],
                    **{
                        f"task_{task_id}_success_rate": rollout[
                            f"rollout_task_{task_id}_success_rate"
                        ]
                        for task_id in TASK_IDS
                    },
                    "path": str(path),
                }
            )
    return runs


def collect_interval_runs(final_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    root = final_root / "runs" / "segmentation-interval-10k-gate"
    for path in sorted(root.glob("*/seed_*/metrics.jsonl")):
        condition = path.parent.parent.name
        match = re.fullmatch(r"(robust|naive)-interval(\d+)", condition)
        if match is None:
            raise ValueError(f"Unexpected interval condition: {condition}")
        mode, interval = match.groups()
        seed = int(path.parent.name.removeprefix("seed_"))
        _, evaluation, rollout = read_final_events(path)
        runs.append(
            {
                "mode": mode,
                "interval": int(interval),
                "seed": seed,
                "success_rate": rollout["rollout_success_rate"],
                "action_mse": evaluation["eval_action_mse"],
                "path": str(path),
            }
        )
    return runs


def summarize_intervals(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intervals = sorted({int(run["interval"]) for run in runs})
    for interval in intervals:
        for mode in ("robust", "naive"):
            selected = [
                run
                for run in runs
                if run["interval"] == interval and run["mode"] == mode
            ]
            success = np.asarray(
                [run["success_rate"] for run in selected],
                dtype=float,
            )
            mse = np.asarray(
                [run["action_mse"] for run in selected],
                dtype=float,
            )
            rows.append(
                {
                    "interval": interval,
                    "mode": mode,
                    "seeds": len(selected),
                    "success_mean": float(success.mean()),
                    "success_sample_std": float(success.std(ddof=1)),
                    "action_mse_mean": float(mse.mean()),
                    "action_mse_sample_std": float(mse.std(ddof=1)),
                }
            )
    return rows


def collect_broader_runs(
    root: Path,
    random_full_root: Path,
    fixed_h8_full_root: Path,
    corrected_seed_zero_root: Path,
    corrected_multiseed_root: Path,
) -> list[dict[str, Any]]:
    specifications = {
        "fixed_h8": "segmentation-fixed-h8-100k-gate",
        "random_cuts": "segmentation-random-100k-gate",
    }
    runs: list[dict[str, Any]] = []
    for study, directory in specifications.items():
        for path in sorted(
            (root / "runs" / directory).glob(
                "*/seed_*/metrics.jsonl"
            )
        ):
            condition = path.parent.parent.name
            seed = int(path.parent.name.removeprefix("seed_"))
            start, evaluation, rollout = read_final_events(path)
            runs.append(
                {
                    "study": study,
                    "condition": condition,
                    "mode": condition_mode(condition),
                    "seed": seed,
                    "backup_boundary_count": int(
                        start["backup_boundary_count"]
                    ),
                    "success_rate": rollout["rollout_success_rate"],
                    "action_mse": evaluation["eval_action_mse"],
                    "path": str(path),
                }
            )
    for path in sorted(
        (
            random_full_root
            / "runs"
            / "segmentation-random-full-gate"
        ).glob("*/seed_*/metrics.jsonl")
    ):
        condition = path.parent.parent.name
        seed = int(path.parent.name.removeprefix("seed_"))
        start, evaluation, rollout = read_final_events(path)
        runs.append(
            {
                "study": "random_cuts_1m",
                "condition": condition,
                "mode": condition_mode(condition),
                "seed": seed,
                "backup_boundary_count": int(
                    start["backup_boundary_count"]
                ),
                "success_rate": rollout["rollout_success_rate"],
                "action_mse": evaluation["eval_action_mse"],
                "path": str(path),
            }
        )
    for seed in range(3):
        original_root = (
            corrected_seed_zero_root
            if seed == 0
            else corrected_multiseed_root
        )
        path = (
            original_root
            / "runs"
            / "full"
            / "fixed-h8"
            / f"seed_{seed}"
            / "metrics.jsonl"
        )
        start, evaluation, rollout = read_final_events(path)
        runs.append(
            {
                "study": "fixed_h8_1m",
                "condition": "original",
                "mode": "original",
                "seed": seed,
                "backup_boundary_count": int(
                    start.get("backup_boundary_count", 0)
                ),
                "success_rate": rollout["rollout_success_rate"],
                "action_mse": evaluation["eval_action_mse"],
                "path": str(path),
            }
        )
    for path in sorted(
        (
            fixed_h8_full_root
            / "runs"
            / "segmentation-fixed-h8-full-gate"
        ).glob("*/seed_*/metrics.jsonl")
    ):
        condition = path.parent.parent.name
        seed = int(path.parent.name.removeprefix("seed_"))
        start, evaluation, rollout = read_final_events(path)
        runs.append(
            {
                "study": "fixed_h8_1m",
                "condition": condition,
                "mode": condition_mode(condition),
                "seed": seed,
                "backup_boundary_count": int(
                    start["backup_boundary_count"]
                ),
                "success_rate": rollout["rollout_success_rate"],
                "action_mse": evaluation["eval_action_mse"],
                "path": str(path),
            }
        )
    return runs


def summarize_broader_runs(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study in (
        "fixed_h8",
        "random_cuts",
        "random_cuts_1m",
        "fixed_h8_1m",
    ):
        modes = (
            ("original", "robust", "naive")
            if study in {"fixed_h8", "fixed_h8_1m"}
            else ("robust", "naive")
        )
        for mode in modes:
            selected = [
                run
                for run in runs
                if run["study"] == study and run["mode"] == mode
            ]
            success = np.asarray(
                [run["success_rate"] for run in selected],
                dtype=float,
            )
            mse = np.asarray(
                [run["action_mse"] for run in selected],
                dtype=float,
            )
            rows.append(
                {
                    "study": study,
                    "mode": mode,
                    "seeds": len(selected),
                    "boundary_count_mean": float(
                        np.mean(
                            [
                                run["backup_boundary_count"]
                                for run in selected
                            ]
                        )
                    ),
                    "success_mean": float(success.mean()),
                    "success_sample_std": float(success.std(ddof=1)),
                    "action_mse_mean": float(mse.mean()),
                    "action_mse_sample_std": float(mse.std(ddof=1)),
                }
            )
    return rows


def collect_manipulation_runs(root: Path) -> list[dict[str, Any]]:
    specifications = {
        "cube_double_10k": "segmentation-cube-double-10k-gate",
        "cube_double_100k": "segmentation-cube-double-100k-gate",
        "cube_double_1m_seed0": (
            "segmentation-cube-double-full-seed0-gate"
        ),
    }
    runs: list[dict[str, Any]] = []
    for study, directory in specifications.items():
        for path in sorted(
            (root / "runs" / directory).glob(
                "*/seed_*/metrics.jsonl"
            )
        ):
            condition = path.parent.parent.name
            seed = int(path.parent.name.removeprefix("seed_"))
            start, evaluation, rollout = read_final_events(path)
            runs.append(
                {
                    "study": study,
                    "condition": condition,
                    "mode": condition_mode(condition),
                    "seed": seed,
                    "backup_boundary_count": int(
                        start["backup_boundary_count"]
                    ),
                    "success_rate": rollout["rollout_success_rate"],
                    "action_mse": evaluation["eval_action_mse"],
                    "path": str(path),
                }
            )
    return runs


def summarize_manipulation_runs(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study in (
        "cube_double_10k",
        "cube_double_100k",
        "cube_double_1m_seed0",
    ):
        for mode in MODES:
            selected = [
                run
                for run in runs
                if run["study"] == study and run["mode"] == mode
            ]
            success = np.asarray(
                [run["success_rate"] for run in selected],
                dtype=float,
            )
            mse = np.asarray(
                [run["action_mse"] for run in selected],
                dtype=float,
            )
            rows.append(
                {
                    "study": study,
                    "mode": mode,
                    "seeds": len(selected),
                    "success_mean": float(success.mean()),
                    "success_sample_std": ""
                    if success.size < 2
                    else float(success.std(ddof=1)),
                    "action_mse_mean": float(mse.mean()),
                    "action_mse_sample_std": ""
                    if mse.size < 2
                    else float(mse.std(ddof=1)),
                }
            )
    return rows


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    samples: int = 20_000,
    seed: int = 0,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty vector.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    estimates = array[indices].mean(axis=1)
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def seed_level_values(
    runs: list[dict[str, Any]],
    *,
    experiment: str,
    mode: str,
    metric: str,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        if run["experiment"] == experiment and run["mode"] == mode:
            grouped[int(run["seed"])].append(float(run[metric]))
    if not grouped:
        raise ValueError(f"No runs for {experiment}/{mode}/{metric}.")
    return {
        seed: float(np.mean(values))
        for seed, values in sorted(grouped.items())
    }


def summarize_modes(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment, spec in EXPERIMENTS.items():
        for mode in MODES:
            success = seed_level_values(
                runs,
                experiment=experiment,
                mode=mode,
                metric="success_rate",
            )
            mse = seed_level_values(
                runs,
                experiment=experiment,
                mode=mode,
                metric="action_mse",
            )
            success_values = np.asarray(list(success.values()))
            mse_values = np.asarray(list(mse.values()))
            ci_low, ci_high = bootstrap_mean_ci(success_values)
            condition_count = len(
                {
                    run["condition"]
                    for run in runs
                    if run["experiment"] == experiment
                    and run["mode"] == mode
                }
            )
            rows.append(
                {
                    "experiment": experiment,
                    "experiment_label": spec["label"],
                    "mode": mode,
                    "seeds": len(success),
                    "conditions_per_seed": condition_count,
                    "success_mean": float(success_values.mean()),
                    "success_sample_std": float(
                        success_values.std(ddof=1)
                    ),
                    "success_seed_bootstrap_low": ci_low,
                    "success_seed_bootstrap_high": ci_high,
                    "action_mse_mean": float(mse_values.mean()),
                    "action_mse_sample_std": float(
                        mse_values.std(ddof=1)
                    ),
                }
            )
    return rows


def summarize_contrasts(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = (
        ("robust", "original"),
        ("naive", "original"),
        ("robust", "naive"),
    )
    for experiment, spec in EXPERIMENTS.items():
        for left, right in comparisons:
            left_values = seed_level_values(
                runs,
                experiment=experiment,
                mode=left,
                metric="success_rate",
            )
            right_values = seed_level_values(
                runs,
                experiment=experiment,
                mode=right,
                metric="success_rate",
            )
            seeds = sorted(set(left_values) & set(right_values))
            differences = np.asarray(
                [
                    left_values[seed] - right_values[seed]
                    for seed in seeds
                ]
            )
            ci_low, ci_high = bootstrap_mean_ci(differences)
            rows.append(
                {
                    "experiment": experiment,
                    "experiment_label": spec["label"],
                    "contrast": f"{left}_minus_{right}",
                    "seeds": len(seeds),
                    "mean_difference": float(differences.mean()),
                    "sample_std": float(differences.std(ddof=1)),
                    "seed_bootstrap_low": ci_low,
                    "seed_bootstrap_high": ci_high,
                    "paired_differences": json.dumps(
                        differences.tolist()
                    ),
                }
            )
        original = seed_level_values(
            runs,
            experiment=experiment,
            mode="original",
            metric="success_rate",
        )
        robust = seed_level_values(
            runs,
            experiment=experiment,
            mode="robust",
            metric="success_rate",
        )
        original_mean = float(np.mean(list(original.values())))
        robust_mean = float(np.mean(list(robust.values())))
        rows.append(
            {
                "experiment": experiment,
                "experiment_label": spec["label"],
                "contrast": "robust_retention",
                "seeds": len(original),
                "mean_difference": ""
                if original_mean == 0.0
                else robust_mean / original_mean,
                "sample_std": "",
                "seed_bootstrap_low": "",
                "seed_bootstrap_high": "",
                "paired_differences": "",
            }
        )
    return rows


def summarize_tasks(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment, spec in EXPERIMENTS.items():
        for mode in MODES:
            for task_id in TASK_IDS:
                values = seed_level_values(
                    runs,
                    experiment=experiment,
                    mode=mode,
                    metric=f"task_{task_id}_success_rate",
                )
                array = np.asarray(list(values.values()))
                rows.append(
                    {
                        "experiment": experiment,
                        "experiment_label": spec["label"],
                        "mode": mode,
                        "task_id": task_id,
                        "success_mean": float(array.mean()),
                        "success_sample_std": float(
                            array.std(ddof=1)
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_main_table(path: Path, summaries: list[dict[str, Any]]) -> None:
    by_key = {
        (row["experiment"], row["mode"]): row
        for row in summaries
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Setting & Original & CVT & Naive & CVT/original \\",
        r"\midrule",
    ]
    for experiment, spec in EXPERIMENTS.items():
        original = by_key[(experiment, "original")]["success_mean"]
        robust = by_key[(experiment, "robust")]["success_mean"]
        naive = by_key[(experiment, "naive")]["success_mean"]
        retention = None if original == 0.0 else robust / original
        retention_text = (
            "--" if retention is None else f"{100 * retention:.1f}\\%"
        )
        lines.append(
            f"{spec['label']} & {100 * original:.1f} & "
            f"{100 * robust:.1f} & {100 * naive:.1f} & "
            f"{retention_text} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_key_seed_table(
    path: Path,
    runs: list[dict[str, Any]],
    broader_runs: list[dict[str, Any]],
) -> None:
    main_lookup = {
        (
            run["experiment"],
            run["mode"],
            int(run["seed"]),
        ): float(run["success_rate"])
        for run in runs
    }
    broader_lookup = {
        (
            run["study"],
            run["mode"],
            int(run["seed"]),
        ): float(run["success_rate"])
        for run in broader_runs
    }
    studies = [
        (
            "PointMaze multi-head periodic",
            range(5),
            lambda mode, seed: main_lookup[
                ("medium_1m", mode, seed)
            ],
        ),
        (
            "PointMaze multi-head random",
            range(3),
            lambda mode, seed: (
                main_lookup[("medium_1m", "original", seed)]
                if mode == "original"
                else broader_lookup[
                    ("random_cuts_1m", mode, seed)
                ]
            ),
        ),
        (
            "PointMaze fixed-H8 periodic",
            range(3),
            lambda mode, seed: broader_lookup[
                ("fixed_h8_1m", mode, seed)
            ],
        ),
        (
            "AntMaze multi-head periodic",
            range(3),
            lambda mode, seed: main_lookup[
                ("antmaze_1m", mode, seed)
            ],
        ),
    ]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Study & Seed & Original & CVT & Naive \\",
        r"\midrule",
    ]
    for study, seeds, value in studies:
        for seed in seeds:
            lines.append(
                f"{study} & {seed} & "
                f"{100 * value('original', seed):.1f} & "
                f"{100 * value('robust', seed):.1f} & "
                f"{100 * value('naive', seed):.1f} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Seed-level five-task success (\%) for the key full-budget comparisons.}",
            r"\label{tab:seed-results}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(
    figure_dir: Path,
    summaries: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
) -> None:
    import matplotlib as mpl

    mpl.rcParams["svg.hashsalt"] = "segbench-gc"
    import matplotlib.pyplot as plt

    svg_metadata = {"Date": None}
    pdf_metadata = {
        "CreationDate": None,
        "ModDate": None,
        "Creator": "SegBench-GC",
    }
    figure_dir.mkdir(parents=True, exist_ok=True)
    by_key = {
        (row["experiment"], row["mode"]): row
        for row in summaries
    }
    labels = [spec["label"] for spec in EXPERIMENTS.values()]
    x = np.arange(len(labels))
    width = 0.24
    colors = {
        "original": "#4C78A8",
        "robust": "#59A14F",
        "naive": "#E15759",
    }
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for index, mode in enumerate(MODES):
        values = [
            by_key[(experiment, mode)]["success_mean"]
            for experiment in EXPERIMENTS
        ]
        errors = [
            by_key[(experiment, mode)]["success_sample_std"]
            for experiment in EXPERIMENTS
        ]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            label="CVT" if mode == "robust" else mode.title(),
            color=colors[mode],
        )
    ax.set_ylabel("Success rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.0, 0.8)
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "segbench_gc_success.svg",
        metadata=svg_metadata,
    )
    fig.savefig(
        figure_dir / "segbench_gc_success.pdf",
        metadata=pdf_metadata,
    )
    plt.close(fig)

    task_lookup = {
        (row["mode"], int(row["task_id"])): row["success_mean"]
        for row in task_rows
        if row["experiment"] == "medium_1m"
    }
    task_x = np.arange(len(TASK_IDS))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, mode in enumerate(MODES):
        ax.bar(
            task_x + (index - 1) * width,
            [task_lookup[(mode, task_id)] for task_id in TASK_IDS],
            width,
            label="CVT" if mode == "robust" else mode.title(),
            color=colors[mode],
        )
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Success rate")
    ax.set_xticks(task_x)
    ax.set_xticklabels([str(task_id) for task_id in TASK_IDS])
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "segbench_gc_medium_full_tasks.svg",
        metadata=svg_metadata,
    )
    fig.savefig(
        figure_dir / "segbench_gc_medium_full_tasks.pdf",
        metadata=pdf_metadata,
    )
    plt.close(fig)

    interval_lookup = {
        (int(row["interval"]), row["mode"]): row
        for row in interval_rows
    }
    intervals = sorted({key[0] for key in interval_lookup})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for mode in ("robust", "naive"):
        ax.errorbar(
            intervals,
            [
                interval_lookup[(interval, mode)]["success_mean"]
                for interval in intervals
            ],
            yerr=[
                interval_lookup[(interval, mode)][
                    "success_sample_std"
                ]
                for interval in intervals
            ],
            marker="o",
            linewidth=2,
            capsize=3,
            label="CVT" if mode == "robust" else "Naive",
            color=colors[mode],
        )
    ax.set_xlabel("Artificial cut interval")
    ax.set_ylabel("Success rate")
    ax.set_xticks(intervals)
    ax.set_ylim(0.0, 0.5)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "segbench_gc_cut_density.svg",
        metadata=svg_metadata,
    )
    fig.savefig(
        figure_dir / "segbench_gc_cut_density.pdf",
        metadata=pdf_metadata,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = collect_runs(
        args.segbench_root,
        args.uncut_root,
        args.final_root,
        args.final_pointmaze_root,
    )
    interval_runs = collect_interval_runs(args.final_root)
    interval_rows = summarize_intervals(interval_runs)
    broader_runs = collect_broader_runs(
        args.broader_root,
        args.random_full_root,
        args.fixed_h8_full_root,
        args.corrected_seed_zero_root,
        args.corrected_multiseed_root,
    )
    broader_rows = summarize_broader_runs(broader_runs)
    manipulation_runs = collect_manipulation_runs(
        args.cube_double_root
    )
    manipulation_rows = summarize_manipulation_runs(
        manipulation_runs
    )
    summaries = summarize_modes(runs)
    contrasts = summarize_contrasts(runs)
    task_rows = summarize_tasks(runs)
    write_csv(args.output_dir / "segbench_gc_runs.csv", runs)
    write_csv(args.output_dir / "segbench_gc_summary.csv", summaries)
    write_csv(args.output_dir / "segbench_gc_contrasts.csv", contrasts)
    write_csv(args.output_dir / "segbench_gc_tasks.csv", task_rows)
    write_csv(
        args.output_dir / "segbench_gc_intervals.csv",
        interval_rows,
    )
    write_csv(
        args.output_dir / "segbench_gc_broader_validation.csv",
        broader_rows,
    )
    write_csv(
        args.output_dir / "segbench_gc_manipulation_diagnostic.csv",
        manipulation_rows,
    )
    payload = {
        "runs": runs,
        "summary": summaries,
        "contrasts": contrasts,
        "tasks": task_rows,
        "interval_runs": interval_runs,
        "intervals": interval_rows,
        "broader_runs": broader_runs,
        "broader_validation": broader_rows,
        "manipulation_runs": manipulation_runs,
        "manipulation_diagnostic": manipulation_rows,
    }
    (args.output_dir / "segbench_gc_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_main_table(
        args.output_dir / "segbench_gc_main.tex",
        summaries,
    )
    write_key_seed_table(
        args.output_dir / "segbench_gc_key_seed_results.tex",
        runs,
        broader_runs,
    )
    write_figures(
        args.figure_dir,
        summaries,
        task_rows,
        interval_rows,
    )


if __name__ == "__main__":
    main()
