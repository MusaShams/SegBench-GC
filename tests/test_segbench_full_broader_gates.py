from pathlib import Path


def test_random_full_gate_resumes_paired_100k_checkpoints() -> None:
    script = Path(
        "scripts/run_segmentation_random_full_gate.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2" in script
    assert "for mode in robust naive" in script
    assert "segmentation-random-100k-gate" in script
    assert 'resume_checkpoint="$source_dir/agent.pt"' in script
    assert "--set steps=900000" in script
    assert "--set backup_segmentation_probability=0.04" in script
    assert "--set rollout_episodes=50" in script


def test_fixed_h8_full_gate_reuses_existing_uncut_controls() -> None:
    script = Path(
        "scripts/run_segmentation_fixed_h8_full_gate.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2" in script
    assert "robust-offset24:true" in script
    assert "naive-offset24:false" in script
    assert "original" not in script
    assert "segmentation-fixed-h8-100k-gate" in script
    assert "fixed_horizon_iql_h8_official_matched.yaml" in script
    assert "--set steps=900000" in script
