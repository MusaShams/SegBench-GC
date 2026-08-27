from pathlib import Path

from adaptive_gcrl.utils.config import load_config_files


def test_interval_gate_varies_cut_density() -> None:
    script = Path(
        "scripts/run_segmentation_interval_10k_gate.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2" in script
    for interval in (10, 50, 100):
        offset = interval - 1
        assert (
            f"robust-interval{interval}:{interval}:{offset}:true"
            in script
        )
        assert (
            f"naive-interval{interval}:{interval}:{offset}:false"
            in script
        )


def test_antmaze_gate_uses_official_dynamic_goals() -> None:
    config = load_config_files(
        [
            Path(
                "configs/env/"
                "ogbench_antmaze_medium_official_goals.yaml"
            )
        ]
    )
    script = Path(
        "scripts/run_segmentation_antmaze_10k_gate.sh"
    ).read_text(encoding="utf-8")

    assert config["task"] == "antmaze-medium-stitch-v0"
    assert config["goal_sampling_mode"] == "official"
    assert "for seed in 0 1 2" in script
    assert "original:0:0:true" in script
    assert "robust-offset24:25:24:true" in script
    assert "naive-offset24:25:24:false" in script


def test_antmaze_100k_gate_is_resumable() -> None:
    script = Path(
        "scripts/run_segmentation_antmaze_100k_gate.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2" in script
    assert "ogbench_antmaze_medium_official_goals.yaml" in script
    assert "remaining_steps=$((100000 - completed_steps))" in script
    assert "--set rollout_episodes=10" in script


def test_antmaze_full_gate_resumes_seed_zero() -> None:
    script = Path(
        "scripts/run_segmentation_antmaze_full_seed0_gate.sh"
    ).read_text(encoding="utf-8")

    assert "segmentation-antmaze-100k-gate" in script
    assert "checkpoints/latest.pt" in script
    assert "--set resume_checkpoint_path=" in script
    assert "--set steps=900000" in script
    assert "--set rollout_episodes=50" in script
    assert "seed_0" in script


def test_antmaze_full_remaining_gate_resumes_paired_seeds() -> None:
    script = Path(
        "scripts/run_segmentation_antmaze_full_remaining.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 1 2" in script
    assert "segmentation-antmaze-100k-gate" in script
    assert "--set resume_checkpoint_path=" in script
    assert "--set steps=900000" in script
    assert "--set rollout_episodes=50" in script
