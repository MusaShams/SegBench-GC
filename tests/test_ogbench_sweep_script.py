from pathlib import Path

from scripts.run_ogbench_smoke_sweep import build_export_command, build_train_command
from adaptive_gcrl.utils.config import load_config_files


def test_build_train_command_contains_seed_and_output_override() -> None:
    command = build_train_command(
        "adaptive_iql_sparse_tuned",
        2,
        Path("runs/sweep"),
        env_config=Path("configs/env/ogbench_sparse_smoke.yaml"),
        steps=500,
        rollout_episodes=4,
        rollout_max_steps=200,
        eval_batch_size=1024,
    )

    assert "--set" in command
    assert "seed=2" in command
    assert "output_dir=runs/sweep/adaptive_iql_sparse_tuned/seed_2" in command
    assert "steps=500" in command
    assert "rollout_episodes=4" in command
    assert "rollout_max_steps=200" in command
    assert "eval_batch_size=1024" in command
    assert "configs/env/ogbench_sparse_smoke.yaml" in command
    assert "configs/algo/adaptive_iql_sparse_tuned.yaml" in command


def test_build_export_command_includes_all_runs() -> None:
    command = build_export_command(
        ["iql", "adaptive_iql"],
        [0, 1],
        Path("runs/sweep"),
        Path("table.csv"),
        Path("table.json"),
        metric="rollout_success_rate",
        event="rollout_eval",
    )

    run_specs = [command[index + 1] for index, value in enumerate(command) if value == "--run"]

    assert "rollout_success_rate" in command
    assert "rollout_eval" in command
    assert len(run_specs) == 4
    assert "adaptive_iql_seed_1=runs/sweep/adaptive_iql/seed_1/metrics.jsonl" in run_specs


def test_fixed_horizon_ablation_is_available() -> None:
    command = build_train_command("fixed_horizon_iql_h4", 0, Path("runs/ablation"))

    assert "configs/algo/fixed_horizon_iql_h4.yaml" in command
    assert "output_dir=runs/ablation/fixed_horizon_iql_h4/seed_0" in command


def test_centered_adaptive_ablation_is_available() -> None:
    command = build_train_command("adaptive_iql_sparse_centered", 0, Path("runs/ablation"))

    assert "configs/algo/adaptive_iql_sparse_centered.yaml" in command
    assert "output_dir=runs/ablation/adaptive_iql_sparse_centered/seed_0" in command


def test_centered_weighted_adaptive_ablation_is_available() -> None:
    command = build_train_command("adaptive_iql_sparse_centered_weighted", 0, Path("runs/ablation"))

    assert "configs/algo/adaptive_iql_sparse_centered_weighted.yaml" in command
    assert "output_dir=runs/ablation/adaptive_iql_sparse_centered_weighted/seed_0" in command


def test_chunked_adaptive_ablation_is_available() -> None:
    command = build_train_command("adaptive_iql_sparse_centered_weighted_chunked", 0, Path("runs/ablation"))

    assert "configs/algo/adaptive_iql_sparse_centered_weighted_chunked.yaml" in command
    assert "output_dir=runs/ablation/adaptive_iql_sparse_centered_weighted_chunked/seed_0" in command


def test_normalized_ablation_configs_are_available() -> None:
    fixed_command = build_train_command("fixed_horizon_iql_h4_normalized", 0, Path("runs/ablation"))
    adaptive_command = build_train_command("adaptive_iql_sparse_centered_weighted_normalized", 0, Path("runs/ablation"))

    assert "configs/algo/fixed_horizon_iql_h4_normalized.yaml" in fixed_command
    assert "configs/algo/adaptive_iql_sparse_centered_weighted_normalized.yaml" in adaptive_command


def test_squashed_ablation_configs_are_available() -> None:
    fixed_command = build_train_command("fixed_horizon_iql_h4_squashed", 0, Path("runs/ablation"))
    adaptive_command = build_train_command("adaptive_iql_sparse_centered_weighted_squashed", 0, Path("runs/ablation"))

    assert "configs/algo/fixed_horizon_iql_h4_squashed.yaml" in fixed_command
    assert "configs/algo/adaptive_iql_sparse_centered_weighted_squashed.yaml" in adaptive_command


def test_goal_directed_ablation_configs_are_available() -> None:
    fixed_command = build_train_command("fixed_horizon_iql_h4_goal_directed", 0, Path("runs/ablation"))
    adaptive_command = build_train_command("adaptive_iql_sparse_centered_weighted_goal_directed", 0, Path("runs/ablation"))

    assert "configs/algo/fixed_horizon_iql_h4_goal_directed.yaml" in fixed_command
    assert "configs/algo/adaptive_iql_sparse_centered_weighted_goal_directed.yaml" in adaptive_command


def test_official_matched_pytorch_configs_are_available() -> None:
    baseline_command = build_train_command("gciql_official_matched", 0, Path("runs/official"))
    adaptive_command = build_train_command("adaptive_iql_official_matched", 0, Path("runs/official"))

    assert "configs/algo/gciql_official_matched.yaml" in baseline_command
    assert "configs/algo/adaptive_iql_official_matched.yaml" in adaptive_command


def test_official_matched_fixed_horizon_configs_are_available() -> None:
    h1_command = build_train_command(
        "fixed_horizon_iql_h1_official_matched", 0, Path("runs/official")
    )
    h8_command = build_train_command(
        "fixed_horizon_iql_h8_official_matched", 0, Path("runs/official")
    )

    assert "configs/algo/fixed_horizon_iql_h1_official_matched.yaml" in h1_command
    assert "configs/algo/fixed_horizon_iql_h8_official_matched.yaml" in h8_command


def test_official_matched_static_mixture_config_is_available() -> None:
    command = build_train_command(
        "static_mixture_iql_official_matched",
        0,
        Path("runs/official"),
    )

    assert "configs/algo/static_mixture_iql_official_matched.yaml" in command


def test_hf_gciql_config_is_available() -> None:
    command = build_train_command(
        "hf_gciql_official_matched",
        0,
        Path("runs/official"),
    )

    assert "configs/algo/hf_gciql_official_matched.yaml" in command

    config_paths = [
        Path(command[index + 1])
        for index, value in enumerate(command)
        if value == "--config"
    ]
    resolved = load_config_files(config_paths)

    assert resolved["goal_sampling_mode"] == "official"
    assert resolved["actor_p_trajgoal"] == 0.5
    assert resolved["actor_p_randomgoal"] == 0.5
