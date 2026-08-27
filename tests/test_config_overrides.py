import pytest

from adaptive_gcrl.utils.config import apply_overrides, parse_override


def test_parse_override_infers_yaml_values() -> None:
    assert parse_override("seed=3") == {"seed": 3}
    assert parse_override("learned_gate=true") == {"learned_gate": True}


def test_parse_override_supports_nested_keys() -> None:
    assert parse_override("algo.hidden_dim=64") == {"algo": {"hidden_dim": 64}}


def test_apply_overrides_merges_in_order() -> None:
    config = apply_overrides({"seed": 0, "algo": {"name": "iql"}}, ["seed=2", "algo.hidden_dim=32"])

    assert config == {"seed": 2, "algo": {"name": "iql", "hidden_dim": 32}}


def test_parse_override_rejects_invalid_syntax() -> None:
    with pytest.raises(ValueError):
        parse_override("seed")

