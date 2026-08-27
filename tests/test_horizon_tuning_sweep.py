import json
from pathlib import Path

from scripts.run_horizon_tuning_sweep import aggregate_by_setting, build_train_command, run_label, summarize_run


def test_run_label_is_path_safe() -> None:
    assert run_label("sqrt_horizon", 0.65, 2) == "sqrt_horizon_penalty_0p65_seed_2"


def test_build_train_command_includes_tuning_overrides() -> None:
    label, output_dir, command = build_train_command(
        mode="sqrt_horizon",
        penalty=0.65,
        seed=0,
        output_root=Path("runs/tune"),
        steps=50,
        rollout_episodes=1,
        rollout_max_steps=25,
    )

    assert label == "sqrt_horizon_penalty_0p65_seed_0"
    assert output_dir == Path("runs/tune/sqrt_horizon_penalty_0p65_seed_0")
    assert "horizon_value_mode=sqrt_horizon" in command
    assert "horizon_penalty=0.65" in command
    assert "steps=50" in command


def test_summarize_run_extracts_horizon_metrics(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"event": "train_step", "metrics": {"selected_horizon": 1.0, "target_horizon": 1.0}}),
                json.dumps({"event": "train_step", "metrics": {"selected_horizon": 2.0, "target_horizon": 4.0}}),
                json.dumps({"event": "eval", "metrics": {"eval_action_mse": 0.5}}),
                json.dumps({"event": "rollout_eval", "metrics": {"rollout_success_rate": 0.25}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row = summarize_run("run", "sqrt_horizon", 0.65, 0, path)

    assert row["selected_horizon_mean"] == 1.5
    assert row["target_horizon_mean"] == 2.5
    assert row["selected_horizon_histogram"] == {"1": 1, "2": 1}


def test_aggregate_by_setting_groups_rows() -> None:
    rows = [
        {"mode": "sqrt_horizon", "penalty": 0.1, "eval_action_mse": 0.4, "rollout_success_rate": 0.0, "selected_horizon_mean": 1.0, "target_horizon_mean": 1.0},
        {"mode": "sqrt_horizon", "penalty": 0.1, "eval_action_mse": 0.6, "rollout_success_rate": 0.5, "selected_horizon_mean": 3.0, "target_horizon_mean": 5.0},
    ]

    summary = aggregate_by_setting(rows)

    assert summary[0]["eval_action_mse_mean"] == 0.5
    assert summary[0]["selected_horizon_mean"] == 2.0
