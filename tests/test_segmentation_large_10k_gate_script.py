from pathlib import Path

from adaptive_gcrl.utils.config import load_config_files


def test_large_segmentation_gate_uses_official_goals() -> None:
    config = load_config_files(
        [Path("configs/env/ogbench_large_official_goals.yaml")]
    )

    assert config["task"] == "pointmaze-large-stitch-v0"
    assert config["goal_sampling_mode"] == "official"


def test_large_segmentation_gate_is_matched() -> None:
    script = Path(
        "scripts/run_segmentation_large_10k_gate.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2" in script
    assert "ogbench_large_official_goals.yaml" in script
    assert "original:0:0:true" in script
    assert "robust-offset24:25:24:true" in script
    assert "naive-offset24:25:24:false" in script
    assert "--set steps=10000" in script
