import json

from adaptive_gcrl.evaluation.run_logs import final_metric, metric_series, read_jsonl


def test_run_log_helpers_extract_final_eval_metric(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"event": "train_step", "metrics": {"loss": 2.0, "selected_horizon": 4}}),
                json.dumps({"event": "eval", "metrics": {"score": 1.25}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert len(read_jsonl(path)) == 2
    assert final_metric(path, "score") == 1.25
    assert metric_series(path, "selected_horizon") == [4.0]

