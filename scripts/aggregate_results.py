"""Aggregate experiment score files into reliability-oriented summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_gcrl.evaluation.metrics import aggregate_scores, performance_profile
from adaptive_gcrl.evaluation.run_logs import final_metric, metric_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_files", type=Path, nargs="+")
    parser.add_argument("--metric", default="eval_action_mse")
    parser.add_argument("--mode", choices=["score-json", "metrics-jsonl"], default="score-json")
    parser.add_argument("--profile-thresholds", default=None, help="Comma-separated thresholds, for example: 0.0,0.5,1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores: list[float] = []
    horizon_choices: list[float] = []
    if args.mode == "score-json":
        for path in args.score_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            scores.extend(float(score) for score in payload["scores"])
    else:
        for path in args.score_files:
            scores.append(final_metric(path, args.metric))
            horizon_choices.extend(metric_series(path, "selected_horizon"))
    summary = {"scores": aggregate_scores(scores)}
    if args.profile_thresholds:
        thresholds = [float(value) for value in args.profile_thresholds.split(",")]
        summary["performance_profile"] = {
            str(key): value for key, value in performance_profile(scores, thresholds).items()
        }
    if horizon_choices:
        summary["horizon_choices"] = {
            "mean": sum(horizon_choices) / len(horizon_choices),
            "count": len(horizon_choices),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
