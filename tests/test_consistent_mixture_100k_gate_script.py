from pathlib import Path

from adaptive_gcrl.utils.config import load_config_files


def test_consistent_mixture_config_is_static_and_official() -> None:
    config = load_config_files(
        [Path("configs/algo/consistent_mixture_iql_official_matched.yaml")]
    )

    assert config["goal_sampling_mode"] == "official"
    assert config["static_horizon_weights"] == [0.05, 0.05, 0.05, 0.85]
    assert config["cross_horizon_consistency_weight"] == 0.03
    assert "support_temperature" not in config


def test_consistent_mixture_100k_runner_is_resumable() -> None:
    script = Path("scripts/run_consistent_mixture_100k_gate.sh").read_text(
        encoding="utf-8"
    )

    assert "for seed in 1 2" in script
    assert "consistent_mixture_iql_official_matched.yaml" in script
    assert "remaining_steps=$((100000 - completed_steps))" in script
    assert 'resume_checkpoint_path=$checkpoint' in script
