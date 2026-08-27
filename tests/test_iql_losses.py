import numpy as np

from adaptive_gcrl.algorithms.iql import IQLConfig, compute_iql_losses
from adaptive_gcrl.training.losses import advantage_weights, multi_step_returns


def test_multi_step_returns_respect_horizons_and_terminals() -> None:
    returns = multi_step_returns(
        rewards=np.array([1.0, 2.0, 4.0]),
        terminals=np.array([False, True, False]),
        horizons=(1, 3),
        discount=0.5,
    )

    assert returns.shape == (3, 2)
    assert returns[0, 0] == 1.0
    assert returns[0, 1] == 2.0


def test_advantage_weights_are_positive_and_clipped() -> None:
    weights = advantage_weights(np.array([-1.0, 0.0, 10.0]), temperature=2.0, clip=5.0)

    assert (weights > 0.0).all()
    assert weights[-1] == 5.0


def test_compute_iql_losses_returns_scalar_summary() -> None:
    summary = compute_iql_losses(
        q_values=np.array([1.0, 2.0]),
        values=np.array([0.5, 2.5]),
        target_q_values=np.array([1.5, 1.5]),
        config=IQLConfig(),
    )

    assert summary.value_loss > 0.0
    assert summary.critic_loss > 0.0
    assert summary.actor_weight_mean > 0.0

