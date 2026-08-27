from pathlib import Path


def test_hf_gciql_100k_gate_is_paired_and_resumable() -> None:
    script = Path("scripts/run_hf_gciql_100k_gate.sh").read_text(
        encoding="utf-8"
    )

    assert "for seed in 0 1 2" in script
    assert "static_mixture_iql_official_matched.yaml" in script
    assert "hf_gciql_official_matched.yaml" in script
    assert "remaining_steps=$((100000 - completed_steps))" in script
    assert 'resume_checkpoint_path=$checkpoint' in script
    assert "--set rollout_episodes=10" in script
    assert 'if [[ -f "$log_dir/SUCCESS" ]]' in script
