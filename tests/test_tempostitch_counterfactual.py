import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.analyze_tempostitch_counterfactual import CounterfactualPolicy


class FakeAgent:
    num_horizons = 2
    device = torch.device("cpu")

    class config:
        horizons = (1, 8)

    def _state_goal(self, observations, goals):
        return torch.as_tensor(observations + goals, dtype=torch.float32)

    def _actor_heads(self, state_goal):
        return torch.stack([state_goal, 2.0 * state_goal], dim=1)


def test_counterfactual_policy_can_force_horizon() -> None:
    policy = CounterfactualPolicy(FakeAgent(), horizon=8)

    actions = policy.predict(np.array([[1.0]]), np.array([[2.0]]))

    np.testing.assert_allclose(actions, [[6.0]])


def test_counterfactual_policy_can_use_static_mixture() -> None:
    policy = CounterfactualPolicy(
        FakeAgent(),
        weights=np.array([0.25, 0.75]),
    )

    actions = policy.predict(np.array([[1.0]]), np.array([[2.0]]))

    np.testing.assert_allclose(actions, [[5.25]])
