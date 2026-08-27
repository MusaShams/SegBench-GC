"""Analyze the matched-count SegBench-GC intervention."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODES = ("original", "robust", "naive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs/segmentation-matched-count-10000-gate"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/tables"),
    )
    parser.add_argument(
        "--prefix",
        default="segbench_gc_matched_count_10k",
    )
    return parser.parse_args()


def read_final_events(path: Path) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    start: dict[str, Any] | None = None
    evaluation: dict[str, float] | None = None
    rollout: dict[str, float] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            event = record.get("event")
            if event == "train_start":
                start = record
            elif event == "eval":
                evaluation = {
                    key: float(value)
                    for key, value in record["metrics"].items()
                }
            elif event == "rollout_eval":
                rollout = {
                    key: float(value)
                    for key, value in record["metrics"].items()
                }
    if start is None or evaluation is None or rollout is None:
        raise ValueError(f"Incomplete run log: {path}")
    return start, evaluation, rollout


def _parse_run_identity(path: Path, root: Path) -> tuple[str, int, int | None]:
    relative = path.relative_to(root)
    mode = relative.parts[0]
    run_name = relative.parts[1]
    if mode not in MODES:
        raise ValueError(f"Unknown mode in {path}: {mode}")
    if not run_name.startswith("opt"):
        raise ValueError(f"Unexpected run directory: {run_name}")

    pieces = run_name.split("-seg")
    opt_seed = int(pieces[0].removeprefix("opt"))
    segmentation_seed = None if len(pieces) == 1 else int(pieces[1])
    if mode == "original" and segmentation_seed is not None:
        raise ValueError("Original controls must not have segmentation seeds.")
    if mode != "original" and segmentation_seed is None:
        raise ValueError(f"{mode} run is missing segmentation seed: {path}")
    return mode, opt_seed, segmentation_seed


def collect_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/metrics.jsonl")):
        mode, opt_seed, segmentation_seed = _parse_run_identity(path, root)
        start, evaluation, rollout = read_final_events(path)
        config = start.get("config", {})
        rows.append(
            {
                "mode": mode,
                "opt_seed": opt_seed,
                "segmentation_seed": "" if segmentation_seed is None else segmentation_seed,
                "success_rate": rollout["rollout_success_rate"],
                "action_mse": evaluation["eval_action_mse"],
                "git_commit": start.get("git_commit"),
                "git_dirty": start.get("git_dirty"),
                "config_sha256": start.get("config_sha256"),
                "source_boundary_sha256": start.get("source_boundary_sha256"),
                "backup_boundary_sha256": start.get("backup_boundary_sha256"),
                "source_boundary_count": int(start.get("source_boundary_count", 0)),
                "backup_boundary_count": int(start.get("backup_boundary_count", 0)),
                "requested_artificial_backup_boundary_count": int(
                    config.get("backup_segmentation_count", 0)
                ),
                "artificial_backup_boundary_count": int(
                    start.get("artificial_backup_boundary_count", 0)
                ),
                "path": str(path),
            }
        )
    if not rows:
        raise ValueError(f"No matched-count runs found under {root}")
    return rows


def validate_pairs(rows: list[dict[str, Any]]) -> None:
    originals = {
        int(row["opt_seed"]): row
        for row in rows
        if row["mode"] == "original"
    }
    paired: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["mode"] == "original":
            continue
        key = (int(row["opt_seed"]), int(row["segmentation_seed"]))
        paired[key][row["mode"]] = row

    for opt_seed, original in originals.items():
        if original["requested_artificial_backup_boundary_count"] != 0:
            raise ValueError(
                f"Original optimization seed {opt_seed} unexpectedly requests artificial cuts."
            )
        if original["artificial_backup_boundary_count"] != 0:
            raise ValueError(
                f"Original optimization seed {opt_seed} unexpectedly contains artificial cuts."
            )

    for key, modes in sorted(paired.items()):
        if set(modes) != {"robust", "naive"}:
            raise ValueError(f"Incomplete robust/naive pair for {key}: {sorted(modes)}")
        robust = modes["robust"]
        naive = modes["naive"]
        for field in (
            "git_commit",
            "source_boundary_sha256",
            "backup_boundary_sha256",
            "source_boundary_count",
            "backup_boundary_count",
            "requested_artificial_backup_boundary_count",
            "artificial_backup_boundary_count",
        ):
            if robust[field] != naive[field]:
                raise ValueError(
                    f"Pair {key} differs in {field}: "
                    f"{robust[field]!r} != {naive[field]!r}"
                )
        requested = robust["requested_artificial_backup_boundary_count"]
        actual = robust["artificial_backup_boundary_count"]
        if requested <= 0:
            raise ValueError(f"Pair {key} does not request artificial cuts.")
        if actual != requested:
            raise ValueError(
                f"Pair {key} requested {requested} artificial cuts but logged {actual}."
            )
        if key[0] not in originals:
            raise ValueError(f"Missing original control for optimization seed {key[0]}.")
        original = originals[key[0]]
        if original["source_boundary_sha256"] != robust["source_boundary_sha256"]:
            raise ValueError(f"Source boundaries differ from original for pair {key}.")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_level_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    originals = {
        int(row["opt_seed"]): float(row["success_rate"])
        for row in rows
        if row["mode"] == "original"
    }
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["mode"] == "original":
            continue
        grouped[(row["mode"], int(row["opt_seed"]))].append(
            float(row["success_rate"])
        )

    output: list[dict[str, Any]] = []
    for opt_seed, original in sorted(originals.items()):
        output.append(
            {
                "mode": "original",
                "opt_seed": opt_seed,
                "original_success": original,
                "segmentation_mean_success": original,
                "segmentation_min_success": original,
                "segmentation_std": 0.0,
                "segmentation_gap": 0.0,
                "worst_drop": 0.0,
            }
        )
        for mode in ("robust", "naive"):
            values = np.asarray(grouped[(mode, opt_seed)], dtype=float)
            if values.size == 0:
                raise ValueError(f"Missing {mode} runs for optimization seed {opt_seed}.")
            mean = float(values.mean())
            minimum = float(values.min())
            output.append(
                {
                    "mode": mode,
                    "opt_seed": opt_seed,
                    "original_success": original,
                    "segmentation_mean_success": mean,
                    "segmentation_min_success": minimum,
                    "segmentation_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "segmentation_gap": original - mean,
                    "worst_drop": original - minimum,
                }
            )
    return output


def aggregate_summary(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for mode in MODES:
        selected = [row for row in seed_rows if row["mode"] == mode]
        if not selected:
            continue
        output.append(
            {
                "mode": mode,
                "optimization_seeds": len(selected),
                "success_mean": float(
                    np.mean([row["segmentation_mean_success"] for row in selected])
                ),
                "success_sample_std": float(
                    np.std(
                        [row["segmentation_mean_success"] for row in selected],
                        ddof=1,
                    )
                )
                if len(selected) > 1
                else 0.0,
                "segmentation_gap_mean": float(
                    np.mean([row["segmentation_gap"] for row in selected])
                ),
                "worst_drop_mean": float(
                    np.mean([row["worst_drop"] for row in selected])
                ),
                "segmentation_dispersion_mean": float(
                    np.mean([row["segmentation_std"] for row in selected])
                ),
            }
        )
    return output


def paired_semantics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["mode"], int(row["opt_seed"]), int(row["segmentation_seed"])): row
        for row in rows
        if row["mode"] != "original"
    }
    keys = sorted(
        {
            (int(row["opt_seed"]), int(row["segmentation_seed"]))
            for row in rows
            if row["mode"] == "robust"
        }
    )
    output: list[dict[str, Any]] = []
    for opt_seed, segmentation_seed in keys:
        robust = lookup[("robust", opt_seed, segmentation_seed)]
        naive = lookup[("naive", opt_seed, segmentation_seed)]
        output.append(
            {
                "opt_seed": opt_seed,
                "segmentation_seed": segmentation_seed,
                "robust_success": float(robust["success_rate"]),
                "naive_success": float(naive["success_rate"]),
                "robust_minus_naive": float(robust["success_rate"])
                - float(naive["success_rate"]),
                "backup_boundary_sha256": robust["backup_boundary_sha256"],
                "requested_artificial_backup_boundary_count": robust[
                    "requested_artificial_backup_boundary_count"
                ],
                "artificial_backup_boundary_count": robust[
                    "artificial_backup_boundary_count"
                ],
            }
        )
    return output


def main() -> None:
    args = parse_args()
    rows = collect_runs(args.root)
    validate_pairs(rows)
    seed_rows = seed_level_summary(rows)
    summary = aggregate_summary(seed_rows)
    pairs = paired_semantics(rows)

    write_csv(args.output_dir / f"{args.prefix}_runs.csv", rows)
    write_csv(args.output_dir / f"{args.prefix}_seed_summary.csv", seed_rows)
    write_csv(args.output_dir / f"{args.prefix}_summary.csv", summary)
    write_csv(args.output_dir / f"{args.prefix}_paired.csv", pairs)

    print(json.dumps(summary, indent=2))
    requested_counts = sorted(
        {pair["requested_artificial_backup_boundary_count"] for pair in pairs}
    )
    print(
        f"Validated {len(pairs)} robust/naive pairs with identical "
        f"backup-boundary hashes and exact requested cut counts {requested_counts}."
    )


if __name__ == "__main__":
    main()
