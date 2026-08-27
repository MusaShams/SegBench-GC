import json

from adaptive_gcrl.evaluation.tables import summarize_run, summarize_runs


def write_metrics(path):
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "train_start",
                        "config": {"algorithm": "adaptive_horizon_chunking", "suite": "synthetic", "task": "linear"},
                        "seed": 3,
                    }
                ),
                json.dumps({"event": "train_step", "metrics": {"selected_horizon": 2.0}}),
                json.dumps({"event": "eval", "metrics": {"eval_action_mse": 0.25}}),
                json.dumps({"event": "rollout_eval", "metrics": {"rollout_success_rate": 1.0}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_summarize_run_extracts_paper_fields(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path)

    summary = summarize_run("adaptive", path, "eval_action_mse")

    assert summary.label == "adaptive"
    assert summary.event == "eval"
    assert summary.algorithm == "adaptive_horizon_chunking"
    assert summary.seed == 3
    assert summary.action_chunk_size == 1
    assert summary.selected_horizon_mean == 2.0
    assert summary.selected_horizon_histogram == {"2": 1}


def test_summarize_runs_includes_aggregate(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path)

    summary = summarize_runs([("adaptive", path)], "eval_action_mse")

    assert summary["aggregate"]["mean"] == 0.25
    assert summary["by_algorithm"]["adaptive_horizon_chunking"]["mean"] == 0.25
    assert summary["by_algorithm_chunk"]["adaptive_horizon_chunking_chunk_1"]["mean"] == 0.25
    assert summary["by_label_group"]["adaptive"]["mean"] == 0.25
    assert summary["runs"][0]["selected_horizon_histogram"] == {"2": 1}
    assert summary["runs"][0]["action_chunk_size"] == 1
    assert summary["runs"][0]["task"] == "linear"


def test_summarize_run_supports_rollout_event(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path)

    summary = summarize_run("adaptive", path, "rollout_success_rate", event="rollout_eval")

    assert summary.event == "rollout_eval"
    assert summary.value == 1.0
