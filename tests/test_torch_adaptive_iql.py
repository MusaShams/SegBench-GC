import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adaptive_gcrl.algorithms.torch_adaptive_iql import TorchAdaptiveIQLConfig, make_torch_adaptive_iql_agent
from adaptive_gcrl.data.horizon_targets import attach_horizon_targets, compute_goal_horizon_targets
from adaptive_gcrl.data.replay_buffer import ReplayBuffer
from adaptive_gcrl.data.synthetic import SyntheticGCRLConfig, make_synthetic_gcrl_batch
from adaptive_gcrl.training.trainer import OfflineTrainer


def make_targeted_batch(config: SyntheticGCRLConfig, seed: int, horizons: tuple[int, ...] = (1, 2, 4)):
    batch = make_synthetic_gcrl_batch(config, seed=seed)
    return attach_horizon_targets(
        batch,
        compute_goal_horizon_targets(
            batch,
            horizons,
            discount=0.99,
            reward_mode="dense_negative_distance",
            success_threshold=1.0,
        ),
    )


def test_torch_adaptive_iql_runs_train_and_eval_step() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=64), seed=6)
    agent = make_torch_adaptive_iql_agent(batch, TorchAdaptiveIQLConfig(horizons=(1, 2, 4), hidden_dim=16))

    result = agent.train_step(batch, np.random.default_rng(0))
    evaluation = agent.evaluate_batch(batch)

    assert result.metrics["adaptive_iql_actor_loss"] >= 0.0
    assert 1.0 <= result.metrics["selected_horizon"] <= 4.0
    assert sum(result.metrics[f"selected_horizon_fraction_{horizon}"] for horizon in (1, 2, 4)) == pytest.approx(1.0)
    assert 1.0 <= evaluation.metrics["selected_horizon"] <= 4.0


def test_torch_adaptive_iql_trainer_reduces_behavior_mse() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=128, noise_std=0.0), seed=7)
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(horizons=(1, 2, 4), hidden_dim=32, learning_rate=3e-3, gate_hidden_dim=8),
    )
    trainer = OfflineTrainer(agent=agent, replay_buffer=ReplayBuffer(batch), batch_size=32, rng=np.random.default_rng(0))

    initial = agent.evaluate_batch(batch).metrics["eval_action_mse"]
    for _ in range(20):
        trainer.train_step()
    final = agent.evaluate_batch(batch).metrics["eval_action_mse"]

    assert final < initial


def test_torch_adaptive_iql_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=64), seed=8)
    config = TorchAdaptiveIQLConfig(horizons=(1, 2, 4), hidden_dim=16, gate_hidden_dim=8)
    agent = make_torch_adaptive_iql_agent(batch, config)
    agent.train_step(batch, np.random.default_rng(0))
    expected_actions = agent.predict(batch.observations[:4], batch.goals[:4])
    checkpoint_path = tmp_path / "adaptive_iql.pt"

    agent.save_checkpoint(checkpoint_path)

    restored = make_torch_adaptive_iql_agent(batch, config)
    restored.load_checkpoint(checkpoint_path)
    np.testing.assert_allclose(restored.predict(batch.observations[:4], batch.goals[:4]), expected_actions)
    assert restored.step == agent.step


def test_per_step_horizon_value_mode_changes_target_horizon() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=16), seed=9)
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(horizons=(1, 2, 4), hidden_dim=8, gate_hidden_dim=8, horizon_value_mode="per_step"),
    )
    q_values = torch.tensor([[1.0, 1.5, 2.0]], dtype=torch.float32)

    _, _, metrics = agent._select_horizon(q_values)

    assert metrics["target_horizon"] == 1.0
    assert "horizon_adjusted_value_1" in metrics


def test_sqrt_horizon_value_mode_can_select_middle_horizon() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=16), seed=10)
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(horizons=(1, 2, 4), hidden_dim=8, gate_hidden_dim=8, horizon_value_mode="sqrt_horizon"),
    )
    q_values = torch.tensor([[1.0, 1.5, 2.0]], dtype=torch.float32)

    _, _, metrics = agent._select_horizon(q_values)

    assert metrics["target_horizon"] == 2.0
    assert "horizon_selection_value_2" in metrics


def test_adaptive_iql_supports_smoothed_sampled_gate() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=16), seed=11)
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(
            horizons=(1, 2, 4),
            hidden_dim=8,
            gate_hidden_dim=8,
            gate_target_smoothing=0.2,
            gate_entropy_regularization=0.01,
            gate_selection_strategy="sample",
        ),
    )

    result = agent.train_step(batch, np.random.default_rng(0))

    assert result.metrics["gate_target_smoothing"] == 0.2
    assert result.metrics["gate_entropy_regularization"] == 0.01
    assert 1.0 <= result.metrics["selected_horizon"] <= 4.0
    assert sum(result.metrics[f"selected_horizon_fraction_{horizon}"] for horizon in (1, 2, 4)) == pytest.approx(1.0)


def test_centered_horizon_prior_can_target_middle_horizon() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=16), seed=12)
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(
            horizons=(1, 2, 4),
            hidden_dim=8,
            gate_hidden_dim=8,
            horizon_value_mode="sqrt_horizon",
            horizon_prior_center=2.0,
            horizon_prior_penalty=0.5,
        ),
    )
    q_values = torch.tensor([[1.0, 1.5, 2.0]], dtype=torch.float32)

    _, _, metrics = agent._select_horizon(q_values)

    assert metrics["target_horizon"] == 2.0
    assert metrics["horizon_prior_center"] == 2.0
    assert metrics["horizon_prior_penalty"] == 0.5


def test_horizon_value_weight_controls_cross_horizon_value_scale() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=16), seed=21)
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(
            horizons=(1, 2, 4),
            hidden_dim=8,
            gate_hidden_dim=8,
            horizon_value_weight=0.0,
            horizon_prior_center=2.0,
            horizon_prior_penalty=0.5,
        ),
    )
    q_values = torch.tensor([[100.0, 1.0, 100.0]], dtype=torch.float32)

    _, _, metrics = agent._select_horizon(q_values, q_values)

    assert metrics["target_horizon"] == 2.0
    assert metrics["horizon_value_weight"] == 0.0


def test_adaptive_iql_supports_gate_weighted_actor_loss() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=16), seed=13)
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(
            horizons=(1, 2, 4),
            hidden_dim=8,
            gate_hidden_dim=8,
            gate_target_smoothing=0.2,
            gate_entropy_regularization=0.01,
            gate_selection_strategy="sample",
            actor_horizon_weighting="gate",
        ),
    )

    result = agent.train_step(batch, np.random.default_rng(0))

    assert result.metrics["actor_horizon_weighting"] == 1.0
    assert result.metrics["adaptive_iql_weight_max"] >= result.metrics["adaptive_iql_weight_mean"]


def test_adaptive_iql_supports_static_horizon_mixture() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(
        SyntheticGCRLConfig(num_transitions=16),
        seed=22,
    )
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(
            horizons=(1, 2, 4),
            hidden_dim=8,
            gate_hidden_dim=8,
            actor_horizon_weighting="gate",
            gate_execution_strategy="mixture",
            static_horizon_weights=(0.1, 0.2, 0.7),
        ),
    )
    gate_parameters_before = {
        key: value.clone()
        for key, value in agent.gate.model.state_dict().items()
    }
    gate_step_before = agent.gate.step

    result = agent.train_step(batch, np.random.default_rng(0))
    _, info = agent.predict_with_info(
        batch.observations[:2],
        batch.goals[:2],
    )

    assert result.metrics["static_horizon_mixture"] == 1.0
    np.testing.assert_allclose(
        info["horizon_probabilities"],
        np.tile([0.1, 0.2, 0.7], (2, 1)),
    )
    for key, value in agent.gate.model.state_dict().items():
        assert torch.equal(value, gate_parameters_before[key])
    assert agent.gate.step == gate_step_before


def test_support_aware_fusion_prefers_lower_uncertainty() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(
        SyntheticGCRLConfig(num_transitions=16),
        seed=23,
    )
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(
            horizons=(1, 2, 4),
            hidden_dim=8,
            gate_hidden_dim=8,
            static_horizon_weights=(0.2, 0.2, 0.6),
            support_temperature=0.1,
        ),
    )

    probabilities = agent._support_probabilities(
        np.array([[0.4, 0.0, 0.4]], dtype=np.float32)
    )

    assert probabilities[0, 1] > probabilities[0, 0]
    assert probabilities[0, 1] > 0.2
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    evaluation = agent.evaluate_batch(batch)
    assert "horizon_probability_std_1" in evaluation.metrics


def test_support_probabilities_match_half_disagreement_convention() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(
        SyntheticGCRLConfig(num_transitions=16),
        seed=25,
    )
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(
            horizons=(1, 2, 4),
            hidden_dim=8,
            gate_hidden_dim=8,
            support_temperature=0.1,
        ),
    )
    q1 = torch.zeros((1, 3), dtype=torch.float32)
    q2 = torch.tensor([[0.0, 0.2, 0.4]], dtype=torch.float32)

    _, uncertainties = agent._horizon_selection_inputs(q1, q2)
    probabilities = agent._support_probabilities(uncertainties)
    expected = np.exp(np.array([0.0, -1.0, -2.0]))
    expected /= expected.sum()

    np.testing.assert_allclose(uncertainties, [[0.0, 0.1, 0.2]])
    np.testing.assert_allclose(probabilities[0], expected, rtol=1e-6)


@pytest.mark.parametrize("horizons", [(1, 4, 2), (1, 2, 2)])
def test_adaptive_iql_requires_sorted_unique_horizons(
    horizons: tuple[int, ...],
) -> None:
    with pytest.raises(
        ValueError,
        match="strictly increasing and unique",
    ):
        TorchAdaptiveIQLConfig(horizons=horizons)


def test_cross_horizon_consistency_uses_full_valid_prefixes() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(
        SyntheticGCRLConfig(num_transitions=16),
        seed=24,
    )
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(
            horizons=(1, 2, 4),
            hidden_dim=8,
            gate_hidden_dim=8,
            cross_horizon_consistency_weight=0.1,
        ),
    )
    q_values = torch.zeros((batch.size, 3), dtype=torch.float32)
    targets = torch.as_tensor(
        batch.horizon_targets.returns,
        dtype=torch.float32,
    )

    loss, coverage = agent._cross_horizon_consistency_loss(
        q_values,
        targets,
        batch,
    )

    assert loss >= 0.0
    assert 0.0 < coverage <= 1.0


def test_adaptive_iql_normalizes_state_goal_inputs() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=64), seed=14)
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(horizons=(1, 2, 4), hidden_dim=8, gate_hidden_dim=8, normalize_inputs=True),
    )

    state_goal = agent._state_goal(batch.observations, batch.goals)

    assert torch.allclose(state_goal.mean(dim=0), torch.zeros(state_goal.shape[1]), atol=1e-5)


def test_adaptive_iql_tanh_actor_bounds_predictions() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=64), seed=15)
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(horizons=(1, 2, 4), hidden_dim=8, gate_hidden_dim=8, actor_output_activation="tanh"),
    )

    actions = agent.predict(batch.observations[:8], batch.goals[:8])

    assert np.max(actions) <= 1.0
    assert np.min(actions) >= -1.0


def test_adaptive_iql_goal_direction_loss_runs_for_pointmaze_shapes() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(
        SyntheticGCRLConfig(num_transitions=64, observation_dim=2, goal_dim=2, action_dim=2),
        seed=16,
    )
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(
            horizons=(1, 2, 4),
            hidden_dim=8,
            gate_hidden_dim=8,
            goal_direction_loss_weight=0.1,
        ),
    )

    result = agent.train_step(batch, np.random.default_rng(0))

    assert result.metrics["adaptive_iql_goal_direction_loss"] >= 0.0


def test_adaptive_critic_target_uses_matching_horizon_next_values() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=8), seed=17)
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(horizons=(1, 2, 4), hidden_dim=8, gate_hidden_dim=8),
    )
    for parameter in agent.value.parameters():
        parameter.data.zero_()
    agent.value.net[-1].bias.data.copy_(torch.tensor([1.0, 2.0, 3.0]))

    target = agent._critic_target(batch)
    expected = torch.as_tensor(batch.horizon_targets.returns, dtype=torch.float32)
    expected = expected + torch.as_tensor(batch.horizon_targets.discounts) * torch.tensor([1.0, 2.0, 3.0])

    assert torch.allclose(target.cpu(), expected)


def test_state_dependent_gate_executes_different_actor_heads() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(
        SyntheticGCRLConfig(num_transitions=8, observation_dim=1, goal_dim=1, action_dim=1),
        seed=18,
        horizons=(1, 2),
    )
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(horizons=(1, 2), hidden_dim=4, gate_hidden_dim=4),
    )
    for parameter in agent.actor.parameters():
        parameter.data.zero_()
    agent.actor.net[-1].bias.data.copy_(torch.tensor([0.1, 0.9]))
    for parameter in agent.gate.model.parameters():
        parameter.data.zero_()
    agent.gate.model[0].weight.data[0, 0] = 1.0
    agent.gate.model[0].weight.data[1, 1] = 1.0
    agent.gate.model[2].weight.data[0, 0] = 1.0
    agent.gate.model[2].weight.data[1, 1] = 1.0

    agent._policy_q_heads = lambda state_goal, action_heads, target=True: (
        torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
        torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
    )
    actions, info = agent.predict_with_info(
        observations=np.array([[0.0], [1.0]]),
        goals=np.array([[1.0], [0.0]]),
    )

    np.testing.assert_allclose(actions[:, 0], np.array([0.1, 0.9]), atol=1e-6)
    np.testing.assert_array_equal(info["selected_horizon"], np.array([1, 2]))


def test_soft_gate_execution_blends_actor_heads_and_reports_expected_horizon() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(
        SyntheticGCRLConfig(num_transitions=8, observation_dim=1, goal_dim=1, action_dim=1),
        seed=22,
        horizons=(1, 4),
    )
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(
            horizons=(1, 4),
            hidden_dim=4,
            gate_hidden_dim=4,
            gate_execution_strategy="mixture",
        ),
    )
    for parameter in agent.actor.parameters():
        parameter.data.zero_()
    agent.actor.net[-1].bias.data.copy_(torch.tensor([0.0, 1.0]))
    agent.gate.select_batch = lambda values, uncertainties, strategy="argmax": (
        np.array([1]),
        np.array([[0.25, 0.75]], dtype=np.float32),
    )
    agent._gate_q_heads = lambda state_goal, action_heads=None: (
        torch.ones((1, 2)),
        torch.ones((1, 2)),
    )

    actions, info = agent.predict_with_info(
        observations=np.array([[0.0]]),
        goals=np.array([[1.0]]),
    )

    np.testing.assert_allclose(actions, np.array([[0.75]]), atol=1e-6)
    np.testing.assert_allclose(info["selected_horizon"], np.array([3.25]))


def test_gate_training_uses_policy_proposal_q_heads() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(
        SyntheticGCRLConfig(num_transitions=8, observation_dim=1, goal_dim=1, action_dim=1),
        seed=20,
        horizons=(1, 2),
    )
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(horizons=(1, 2), hidden_dim=4, gate_hidden_dim=4),
    )
    observed: dict[str, torch.Tensor] = {}

    def fake_gate_q_heads(state_goal, action_heads=None):
        observed["actions"] = action_heads.detach().clone()
        return torch.ones((batch.size, 2)), torch.ones((batch.size, 2))

    agent._gate_q_heads = fake_gate_q_heads
    actor_goals = batch.goals if batch.actor_goals is None else batch.actor_goals
    expected_heads = agent._actor_heads(
        agent._state_goal(batch.observations, actor_goals)
    )

    agent.train_step(batch, np.random.default_rng(0))

    assert torch.allclose(observed["actions"], expected_heads.detach())


def test_adaptive_iql_ddpgbc_actor_loss_reports_q_and_bc_terms() -> None:
    torch.manual_seed(0)
    batch = make_targeted_batch(SyntheticGCRLConfig(num_transitions=32), seed=19)
    agent = make_torch_adaptive_iql_agent(
        batch,
        TorchAdaptiveIQLConfig(
            horizons=(1, 2, 4),
            hidden_dim=16,
            hidden_layers=3,
            activation="gelu",
            gate_hidden_dim=8,
            actor_loss_mode="ddpgbc",
            actor_alpha=0.003,
            actor_horizon_weighting="gate",
        ),
    )

    result = agent.train_step(batch, np.random.default_rng(0))

    assert result.metrics["adaptive_iql_actor_q_loss"] != 0.0
    assert result.metrics["adaptive_iql_actor_bc_loss"] > 0.0
