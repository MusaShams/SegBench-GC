from pathlib import Path


def test_fixed_h8_gate_uses_second_multistep_learner() -> None:
    script = Path(
        "scripts/run_segmentation_fixed_h8_100k_gate.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2" in script
    assert "fixed_horizon_iql_h8_official_matched.yaml" in script
    assert "original:0:0:true" in script
    assert "robust-offset24:25:24:true" in script
    assert "naive-offset24:25:24:false" in script
    assert "--set steps=100000" in script


def test_random_gate_pairs_boundary_seed_and_training_seed() -> None:
    script = Path(
        "scripts/run_segmentation_random_100k_gate.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2" in script
    assert "segmentation_seed=$((10000 + seed))" in script
    assert "--set backup_segmentation_probability=0.04" in script
    assert '--set backup_segmentation_seed="$segmentation_seed"' in script
    assert "for mode in robust naive" in script
