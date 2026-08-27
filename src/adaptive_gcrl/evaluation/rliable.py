"""Compatibility layer for rliable-style summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from adaptive_gcrl.evaluation.metrics import aggregate_scores, performance_profile


def summarize_by_task(task_scores: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    if not task_scores:
        raise ValueError("task_scores must not be empty.")
    return {task: aggregate_scores(scores) for task, scores in task_scores.items()}


def summarize_experiment(scores_by_task: Mapping[str, Sequence[float]], profile_points: int = 21) -> dict:
    if profile_points <= 1:
        raise ValueError("profile_points must be greater than one.")
    task_summary = summarize_by_task(scores_by_task)
    all_scores = [float(score) for scores in scores_by_task.values() for score in scores]
    low = min(all_scores)
    high = max(all_scores)
    thresholds = np.linspace(low, high, profile_points)
    return {
        "tasks": task_summary,
        "aggregate": aggregate_scores(all_scores),
        "performance_profile": {str(key): value for key, value in performance_profile(all_scores, thresholds).items()},
    }
