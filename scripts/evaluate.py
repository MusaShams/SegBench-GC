"""Evaluation entry point for saved adaptive GCRL runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_gcrl.evaluation.metrics import aggregate_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate_scores(args.scores)
    text = "\n".join(f"{key}: {value}" for key, value in summary.items())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

