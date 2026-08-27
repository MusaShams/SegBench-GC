from pathlib import Path


def test_segmentation_100k_gate_targets_collapsed_offsets() -> None:
    script = Path("scripts/run_segmentation_100k_gate.sh").read_text(
        encoding="utf-8"
    )

    assert "for seed in 0 1 2" in script
    assert "original:0:0:true" in script
    for offset in (14, 24):
        assert f"robust-offset{offset}:25:{offset}:true" in script
        assert f"naive-offset{offset}:25:{offset}:false" in script
    assert "--set rollout_episodes=10" in script
    assert "remaining_steps=$((100000 - completed_steps))" in script
