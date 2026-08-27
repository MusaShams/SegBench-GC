from pathlib import Path


def test_static_mixture_runner_is_resumable() -> None:
    script = Path("scripts/run_static_mixture_multiseed.sh").read_text(
        encoding="utf-8"
    )

    assert "seeds=(1 2 3 4)" in script
    assert 'if [[ -f "$log_dir/SUCCESS" ]]' in script
    assert "static_mixture_iql_official_matched.yaml" in script
    assert 'remaining_steps=$((1000000 - completed_steps))' in script
    assert 'resume_checkpoint_path=$checkpoint' in script
    assert '--set seed="$seed"' in script
