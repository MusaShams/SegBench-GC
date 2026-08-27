from pathlib import Path


def test_medium_full_gate_reuses_original_controls() -> None:
    script = Path(
        "scripts/run_segmentation_medium_full_gate.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2" in script
    assert "robust-offset24:true" in script
    assert "naive-offset24:false" in script
    assert "original" not in script
    assert "remaining_steps=$((1000000 - completed_steps))" in script
    assert "--set rollout_episodes=50" in script
