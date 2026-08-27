from pathlib import Path


def test_segmentation_10k_gate_is_paired_across_offsets() -> None:
    script = Path("scripts/run_segmentation_10k_gate.sh").read_text(
        encoding="utf-8"
    )

    assert "for seed in 0 1 2" in script
    assert "original:0:0:true" in script
    for offset in (4, 14, 24):
        assert f"robust-offset{offset}:25:{offset}:true" in script
        assert f"naive-offset{offset}:25:{offset}:false" in script
    assert "--set steps=10000" in script
    assert "--set rollout_episodes=2" in script
    assert 'if [[ -f "$log_dir/SUCCESS" ]]' in script
