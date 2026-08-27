from pathlib import Path


def test_large_100k_gate_is_resumable_and_matched() -> None:
    script = Path(
        "scripts/run_segmentation_large_100k_gate.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2" in script
    assert "ogbench_large_official_goals.yaml" in script
    assert "original:0:0:true" in script
    assert "robust-offset24:25:24:true" in script
    assert "naive-offset24:25:24:false" in script
    assert "remaining_steps=$((100000 - completed_steps))" in script
    assert "--set rollout_episodes=10" in script
