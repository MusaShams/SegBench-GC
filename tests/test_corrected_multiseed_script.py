from pathlib import Path


def test_corrected_multiseed_runner_is_resumable_and_paired() -> None:
    script = Path("scripts/run_corrected_multiseed.sh").read_text(encoding="utf-8")

    assert 'seeds=(1 2)' in script
    assert 'if [[ -f "$log_dir/SUCCESS" ]]' in script
    assert 'remaining_steps=$((1000000 - completed_steps))' in script
    assert 'resume_checkpoint_path=$checkpoint' in script
    assert "fixed_horizon_iql_h8_official_matched.yaml" in script
    assert "adaptive_iql_official_matched.yaml" in script
    assert '--set seed="$seed"' in script
    assert "--set action_chunk_size=1" in script
