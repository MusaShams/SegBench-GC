import json

import numpy as np

from scripts.analyze_tempostitch_multiseed import (
    bootstrap_paired_mean_ci,
    final_rollout_metrics,
    summarize_results,
)


def test_final_rollout_metrics_reads_last_rollout_event(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "rollout_eval",
                        "metrics": {"rollout_success_rate": 0.25},
                    }
                ),
                json.dumps(
                    {
                        "event": "rollout_eval",
                        "metrics": {"rollout_success_rate": 0.75},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert final_rollout_metrics(path)["rollout_success_rate"] == 0.75


def test_bootstrap_paired_mean_ci_is_deterministic() -> None:
    differences = np.array([0.1, 0.0, -0.1])

    first = bootstrap_paired_mean_ci(differences, samples=100, seed=4)
    second = bootstrap_paired_mean_ci(differences, samples=100, seed=4)

    assert first == second


def test_summarize_results_preserves_paired_seed_order() -> None:
    def metrics(success: float) -> dict[str, float]:
        payload = {"rollout_success_rate": success}
        for task_id in range(1, 6):
            payload[f"rollout_task_{task_id}_success_rate"] = success
        return payload

    summary = summarize_results(
        {
            "fixed-h8": [metrics(0.2), metrics(0.4)],
            "adaptive-corrected": [metrics(0.3), metrics(0.35)],
        }
    )

    assert summary["paired_difference"]["differences"] == [
        0.09999999999999998,
        -0.050000000000000044,
    ]
    assert summary["paired_difference"]["wins"] == 1
