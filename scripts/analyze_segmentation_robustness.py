"""Summarize sensitivity to artificial trajectory segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "paper/tables/segmentation_robustness_pilot.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "paper/tables/segmentation_robustness_summary.json"
        ),
    )
    return parser.parse_args()


def summarize(payload: dict) -> dict:
    offsets = ("offset_4", "offset_14", "")
    condition_values: dict[str, list[float]] = {}
    for mode in ("robust", "naive"):
        values = []
        for offset in offsets:
            suffix = f"_{offset}" if offset else ""
            key = f"{mode}_cut_25{suffix}"
            values.append(
                [
                    float(row["eval_action_mse"])
                    for row in payload[key]
                ]
            )
        condition_values[mode] = values

    robust = np.asarray(condition_values["robust"], dtype=float)
    naive = np.asarray(condition_values["naive"], dtype=float)
    original = np.asarray(
        [
            float(row["eval_action_mse"])
            for row in payload["original"]
        ],
        dtype=float,
    )
    return {
        "original": {
            "mean": float(original.mean()),
            "sample_std": float(original.std(ddof=1)),
        },
        "robust": {
            "mean": float(robust.mean()),
            "variance": float(robust.var()),
            "mean_offset_range_per_seed": float(
                np.ptp(robust, axis=0).mean()
            ),
        },
        "naive": {
            "mean": float(naive.mean()),
            "variance": float(naive.var()),
            "mean_offset_range_per_seed": float(
                np.ptp(naive, axis=0).mean()
            ),
        },
        "variance_ratio_naive_over_robust": float(
            naive.var() / robust.var()
        ),
        "offset_range_ratio_naive_over_robust": float(
            np.ptp(naive, axis=0).mean()
            / np.ptp(robust, axis=0).mean()
        ),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
