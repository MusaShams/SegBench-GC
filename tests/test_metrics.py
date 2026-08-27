import pytest

from adaptive_gcrl.evaluation.metrics import aggregate_scores, bootstrap_ci, interquartile_mean


def test_interquartile_mean_uses_middle_half() -> None:
    assert interquartile_mean([0.0, 1.0, 2.0, 100.0]) == pytest.approx(1.5)


def test_bootstrap_ci_is_ordered() -> None:
    low, high = bootstrap_ci([1.0, 2.0, 3.0, 4.0], samples=100, seed=0)

    assert low <= high


def test_aggregate_scores_contains_reliability_fields() -> None:
    summary = aggregate_scores([1.0, 2.0, 3.0, 4.0])

    assert {"mean", "median", "iqm", "ci_low", "ci_high"} <= summary.keys()

