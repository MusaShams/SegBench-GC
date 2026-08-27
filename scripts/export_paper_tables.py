"""Export paper-ready CSV and JSON summaries from JSONL run logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from adaptive_gcrl.evaluation.tables import summarize_runs


def parse_run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run specs must use label=/path/to/metrics.jsonl.")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("Run label must not be empty.")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run_spec, required=True)
    parser.add_argument("--event", default="eval")
    parser.add_argument("--metric", default="eval_action_mse")
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "algorithm",
        "suite",
        "task",
        "seed",
        "action_chunk_size",
        "event",
        "metric",
        "value",
        "selected_horizon_mean",
        "selected_horizon_count",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    args = parse_args()
    summary = summarize_runs(args.run, args.metric, event=args.event)
    write_csv(args.csv_output, summary["runs"])
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
