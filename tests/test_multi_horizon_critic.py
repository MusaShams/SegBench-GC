from adaptive_gcrl.data.synthetic import SyntheticGCRLConfig, make_synthetic_gcrl_batch
from adaptive_gcrl.models.critics import CriticSpec, LinearMultiHorizonCritic


def test_linear_multi_horizon_critic_trains_and_predicts_heads() -> None:
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=32, observation_dim=3, goal_dim=3), seed=3)
    critic = LinearMultiHorizonCritic(
        CriticSpec(observation_dim=3, action_dim=2, goal_dim=3, horizons=(1, 2, 4)),
        learning_rate=0.01,
    )

    metrics = critic.train_step(batch, discount=0.99)
    values, uncertainties = critic.horizon_values(batch)

    assert "critic_loss" in metrics
    assert len(values) == 3
    assert len(uncertainties) == 3

