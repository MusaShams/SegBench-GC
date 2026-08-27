from pathlib import Path

from adaptive_gcrl.utils.config import load_config_files


def test_cube_double_gate_uses_official_manipulation_protocol() -> None:
    config = load_config_files(
        [Path("configs/env/ogbench_cube_double_official_goals.yaml")]
    )

    assert config["task"] == "cube-double-play-v0"
    assert config["goal_sampling_mode"] == "official"
    assert config["actor_p_trajgoal"] == 1.0
    assert config["actor_p_randomgoal"] == 0.0


def test_cube_double_10k_gate_is_matched_and_bounded() -> None:
    script = Path(
        "scripts/run_segmentation_cube_double_10k_gate.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2" in script
    assert "ogbench_cube_double_official_goals.yaml" in script
    assert "original:0:0:true" in script
    assert "robust-offset24:25:24:true" in script
    assert "naive-offset24:25:24:false" in script
    assert "--set steps=10000" in script


def test_cube_double_100k_gate_resumes_10k_checkpoints() -> None:
    script = Path(
        "scripts/run_segmentation_cube_double_100k_gate.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2" in script
    assert "segmentation-cube-double-10k-gate" in script
    assert 'resume_checkpoint="$source_dir/agent.pt"' in script
    assert "--set steps=90000" in script
    assert "--set rollout_episodes=10" in script


def test_cube_double_full_gate_resumes_seed_zero() -> None:
    script = Path(
        "scripts/run_segmentation_cube_double_full_seed0_gate.sh"
    ).read_text(encoding="utf-8")

    assert "segmentation-cube-double-100k-gate" in script
    assert 'resume_checkpoint="$source_dir/agent.pt"' in script
    assert "--set seed=0" in script
    assert "--set steps=900000" in script
    assert "--set rollout_episodes=50" in script
