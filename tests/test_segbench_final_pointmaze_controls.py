from pathlib import Path


def test_final_pointmaze_controls_match_provenance_and_expand_seeds() -> None:
    script = Path(
        "scripts/run_segbench_final_pointmaze_controls.sh"
    ).read_text(encoding="utf-8")

    assert "for seed in 0 1 2 3 4" in script
    assert 'run_job "original" "$seed" 0 0 true' in script
    assert "for seed in 3 4" in script
    assert 'run_job "robust-offset24" "$seed" 25 24 true' in script
    assert 'run_job "naive-offset24" "$seed" 25 24 false' in script
    assert "--set steps=" in script
    assert "--set rollout_episodes=50" in script
    assert 'if [[ -f "$log_dir/SUCCESS" ]]' in script
