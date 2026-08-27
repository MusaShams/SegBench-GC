from pathlib import Path

from scripts.run_segmentation_robustness_pilot import CONDITIONS


def test_segmentation_pilot_has_causal_conditions() -> None:
    assert CONDITIONS["original"] == []
    assert "bootstrap_at_backup_boundaries=true" in CONDITIONS[
        "robust_cut_25"
    ]
    assert "bootstrap_at_backup_boundaries=false" in CONDITIONS[
        "naive_cut_25"
    ]
    assert "backup_segmentation_offset=4" in CONDITIONS[
        "robust_cut_25_offset_4"
    ]
    assert "backup_segmentation_offset=14" in CONDITIONS[
        "naive_cut_25_offset_14"
    ]


def test_segmentation_pilot_uses_official_goal_config() -> None:
    script = Path(
        "scripts/run_segmentation_robustness_pilot.py"
    ).read_text(encoding="utf-8")

    assert "configs/env/ogbench_official_goals.yaml" in script
    assert "static_mixture_iql_official_matched.yaml" in script
