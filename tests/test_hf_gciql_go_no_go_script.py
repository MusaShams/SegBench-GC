from pathlib import Path


def test_hf_gciql_go_no_go_is_matched_and_bounded() -> None:
    script = Path("scripts/run_hf_gciql_go_no_go.sh").read_text(
        encoding="utf-8"
    )

    assert "static_mixture_iql_official_matched.yaml" in script
    assert "hf_gciql_official_matched.yaml" in script
    assert "--set steps=10000" in script
    assert "--set rollout_episodes=5" in script
    assert "--set seed=0" in script
    assert 'if [[ -f "$log_dir/SUCCESS" ]]' in script
