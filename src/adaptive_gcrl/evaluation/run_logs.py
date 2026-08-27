"""Utilities for reading JSONL metric logs produced by training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object on line {line_number} of {path}.")
        records.append(payload)
    return records


def final_metric(path: Path, metric_name: str, event: str = "eval") -> float:
    records = read_jsonl(path)
    for record in reversed(records):
        if record.get("event") != event:
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Event {event} in {path} does not contain a metrics object.")
        if metric_name in metrics:
            return float(metrics[metric_name])
    raise KeyError(f"Metric {metric_name} for event {event} was not found in {path}.")


def metric_series(path: Path, metric_name: str, event: str = "train_step") -> list[float]:
    values: list[float] = []
    for record in read_jsonl(path):
        if record.get("event") != event:
            continue
        metrics = record.get("metrics")
        if isinstance(metrics, dict) and metric_name in metrics:
            values.append(float(metrics[metric_name]))
    return values

