from adaptive_gcrl.evaluation.metrics import performance_profile
from adaptive_gcrl.evaluation.rliable import summarize_experiment
from adaptive_gcrl.evaluation.stitching import success_by_horizon, temporal_ablation_gap


def test_performance_profile_counts_threshold_survival() -> None:
    profile = performance_profile([0.1, 0.5, 0.9], [0.0, 0.5, 1.0])

    assert profile[0.0] == 1.0
    assert profile[0.5] == 2 / 3
    assert profile[1.0] == 0.0


def test_summarize_experiment_includes_aggregate_and_tasks() -> None:
    summary = summarize_experiment({"task-a": [1.0, 2.0], "task-b": [3.0, 4.0]}, profile_points=3)

    assert "aggregate" in summary
    assert "task-a" in summary["tasks"]
    assert len(summary["performance_profile"]) == 3


def test_temporal_diagnostics() -> None:
    assert success_by_horizon([True, False, True], [1, 1, 4]) == {1: 0.5, 4: 1.0}
    assert temporal_ablation_gap([2.0, 3.0], [1.0, 1.5]) == 1.25
