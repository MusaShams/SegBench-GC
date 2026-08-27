from adaptive_gcrl.data.synthetic import SyntheticGCRLConfig, make_synthetic_gcrl_batch


def test_synthetic_gcrl_batch_is_deterministic() -> None:
    config = SyntheticGCRLConfig(num_transitions=16, observation_dim=3, goal_dim=3, action_dim=2)

    first = make_synthetic_gcrl_batch(config, seed=123)
    second = make_synthetic_gcrl_batch(config, seed=123)

    assert first.size == 16
    assert first.goals is not None
    assert (first.actions == second.actions).all()
    assert first.next_observations.shape == (16, 3)
