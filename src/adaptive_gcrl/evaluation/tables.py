"""Paper table exports derived from JSONL run logs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from adaptive_gcrl.evaluation.metrics import aggregate_scores
from adaptive_gcrl.evaluation.run_logs import final_metric, metric_series, read_jsonl


@dataclass(frozen=True)
class RunSummary:
    label: str
    path: str
    event: str
    metric: str
    value: float
    algorithm: str
    suite: str
    task: str
    seed: int
    action_chunk_size: int
    selected_horizon_mean: Optional[float]
    selected_horizon_count: int
    selected_horizon_histogram: dict[str, int]


def summarize_run(label: str, path: Path, metric: str, event: str = "eval") -> RunSummary:
    records = read_jsonl(path)
    start_records = [record for record in records if record.get("event") == "train_start"]
    config: dict[str, Any] = {}
    seed = 0
    if start_records:
        config_payload = start_records[0].get("config", {})
        if isinstance(config_payload, dict):
            config = config_payload
        seed = int(start_records[0].get("seed", config.get("seed", 0)))
    horizon_choices = metric_series(path, "selected_horizon")
    action_chunk_size = int(config.get("action_chunk_size", 1))
    try:
        action_chunk_size = int(final_metric(path, "action_chunk_size", event=event))
    except KeyError:
        pass
    selected_horizon_mean = None if not horizon_choices else float(sum(horizon_choices) / len(horizon_choices))
    selected_horizon_histogram: dict[str, int] = {}
    for horizon in horizon_choices:
        key = str(int(horizon))
        selected_horizon_histogram[key] = selected_horizon_histogram.get(key, 0) + 1
    return RunSummary(
        label=label,
        path=str(path),
        event=event,
        metric=metric,
        value=final_metric(path, metric, event=event),
        algorithm=str(config.get("algorithm", "unknown")),
        suite=str(config.get("suite", "unknown")),
        task=str(config.get("task", "unknown")),
        seed=seed,
        action_chunk_size=action_chunk_size,
        selected_horizon_mean=selected_horizon_mean,
        selected_horizon_count=len(horizon_choices),
        selected_horizon_histogram=selected_horizon_histogram,
    )


def summarize_runs(run_specs: list[tuple[str, Path]], metric: str, event: str = "eval") -> dict[str, Any]:
    runs = [summarize_run(label, path, metric, event=event) for label, path in run_specs]
    by_algorithm: dict[str, list[float]] = {}
    by_algorithm_chunk: dict[str, list[float]] = {}
    by_label_group: dict[str, list[float]] = {}
    for run in runs:
        by_algorithm.setdefault(run.algorithm, []).append(run.value)
        by_algorithm_chunk.setdefault(f"{run.algorithm}_chunk_{run.action_chunk_size}", []).append(run.value)
        by_label_group.setdefault(run.label.rsplit("_seed_", 1)[0], []).append(run.value)
    return {
        "runs": [asdict(run) for run in runs],
        "aggregate": aggregate_scores([run.value for run in runs]),
        "by_algorithm": {algorithm: aggregate_scores(scores) for algorithm, scores in by_algorithm.items()},
        "by_algorithm_chunk": {algorithm: aggregate_scores(scores) for algorithm, scores in by_algorithm_chunk.items()},
        "by_label_group": {label: aggregate_scores(scores) for label, scores in by_label_group.items()},
    }
