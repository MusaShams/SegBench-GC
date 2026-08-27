from pathlib import Path


def test_matched_count_launcher_only_terminalizes_artificial_cuts() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_segmentation_matched_count_gate.sh"
    ).read_text(encoding="utf-8")

    # Source-trajectory boundaries remain continuation-valid in both segmented
    # conditions; only the artificial-cut continuation flag changes.
    assert '--set "bootstrap_at_backup_boundaries=true"' in script
    assert '--set "artificial_boundaries_are_continuing=$artificial_continuing"' in script
    assert 'run_job "robust" "$opt_seed" "$segmentation_seed" true' in script
    assert 'run_job "naive" "$opt_seed" "$segmentation_seed" false' in script
    assert 'bootstrap_at_backup_boundaries=$bootstrap' not in script
