from pathlib import Path

from adaptive_gcrl.utils.config import deep_merge, load_config_files


def test_deep_merge_overrides_nested_values() -> None:
    base = {"algo": {"name": "iql", "discount": 0.99}, "seed": 0}
    override = {"algo": {"discount": 0.95}}

    assert deep_merge(base, override) == {"algo": {"name": "iql", "discount": 0.95}, "seed": 0}


def test_load_config_files_merges_in_order(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("seed: 1\nalgo:\n  name: fixed\n", encoding="utf-8")
    second.write_text("algo:\n  horizon: 8\n", encoding="utf-8")

    assert load_config_files([first, second]) == {"seed": 1, "algo": {"name": "fixed", "horizon": 8}}


def test_official_matched_adaptive_config_does_not_claim_action_chunking() -> None:
    config = load_config_files([Path("configs/algo/adaptive_iql_official_matched.yaml")])

    assert config["chunk_size"] == 1
    assert config["action_chunk_size"] == 1


def test_hf_gciql_config_enforces_official_goal_sampling() -> None:
    config = load_config_files(
        [Path("configs/algo/hf_gciql_official_matched.yaml")]
    )

    assert config["goal_sampling_mode"] == "official"
    assert config["value_geom_sample"] is True
    assert config["actor_p_trajgoal"] == 0.5
    assert config["actor_p_randomgoal"] == 0.5


def test_official_goal_config_marks_source_boundaries_continuing() -> None:
    config = load_config_files(
        [Path("configs/env/ogbench_official_goals.yaml")]
    )

    assert config["source_boundaries_are_continuing"] is True
