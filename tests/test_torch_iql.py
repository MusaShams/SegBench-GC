import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adaptive_gcrl.algorithms.torch_iql import TorchIQLConfig, make_torch_iql_agent
from adaptive_gcrl.data.synthetic import SyntheticGCRLConfig, make_synthetic_gcrl_batch
from adaptive_gcrl.training.trainer import OfflineTrainer, TrainState
from adaptive_gcrl.data.replay_buffer import ReplayBuffer
from adaptive_gcrl.training.checkpoints import CheckpointMetadata, checkpoint_metadata_path, load_agent_checkpoint, read_metadata, save_agent_checkpoint


def test_torch_iql_agent_runs_train_and_eval_step() -> None:
    torch.manual_seed(0)
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=64), seed=0)
    agent = make_torch_iql_agent(batch, TorchIQLConfig(hidden_dim=16, learning_rate=1e-3))

    result = agent.train_step(batch, np.random.default_rng(0))
    evaluation = agent.evaluate_batch(batch)

    assert result.step == 1
    assert result.metrics["iql_actor_loss"] >= 0.0
    assert result.metrics["iql_critic_loss"] == pytest.approx(
        result.metrics["iql_critic1_loss"] + result.metrics["iql_critic2_loss"]
    )
    assert evaluation.metrics["eval_action_mse"] >= 0.0


def test_torch_iql_trainer_reduces_behavior_mse_on_synthetic_data() -> None:
    torch.manual_seed(0)
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=128, noise_std=0.0), seed=1)
    agent = make_torch_iql_agent(batch, TorchIQLConfig(hidden_dim=32, learning_rate=3e-3))
    trainer = OfflineTrainer(agent=agent, replay_buffer=ReplayBuffer(batch), batch_size=32, rng=np.random.default_rng(0))

    initial = agent.evaluate_batch(batch).metrics["eval_action_mse"]
    for _ in range(20):
        trainer.train_step()
    final = agent.evaluate_batch(batch).metrics["eval_action_mse"]

    assert final < initial


def test_torch_iql_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=64), seed=3)
    config = TorchIQLConfig(hidden_dim=16, learning_rate=1e-3)
    agent = make_torch_iql_agent(batch, config)
    agent.train_step(batch, np.random.default_rng(0))
    expected_actions = agent.predict(batch.observations[:4], batch.goals[:4])
    checkpoint_path = tmp_path / "iql.pt"

    rng = np.random.default_rng(3)
    save_agent_checkpoint(
        agent,
        checkpoint_path,
        CheckpointMetadata(
            step=agent.step,
            seed=3,
            tfvc_changeset="123",
            rng_state=rng.bit_generator.state,
        ),
    )

    restored = make_torch_iql_agent(batch, config)
    load_agent_checkpoint(restored, checkpoint_path)
    np.testing.assert_allclose(restored.predict(batch.observations[:4], batch.goals[:4]), expected_actions)
    assert restored.step == agent.step
    metadata = read_metadata(checkpoint_metadata_path(checkpoint_path))
    assert metadata.tfvc_changeset == "123"
    assert metadata.rng_state == rng.bit_generator.state


def test_offline_trainer_can_continue_from_checkpoint_step(tmp_path) -> None:
    torch.manual_seed(0)
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=64), seed=4)
    config = TorchIQLConfig(hidden_dim=16, learning_rate=1e-3)
    agent = make_torch_iql_agent(batch, config)
    agent.train_step(batch, np.random.default_rng(0))
    checkpoint_path = tmp_path / "resume.pt"
    save_agent_checkpoint(agent, checkpoint_path, CheckpointMetadata(step=agent.step, seed=4))

    restored = make_torch_iql_agent(batch, config)
    load_agent_checkpoint(restored, checkpoint_path)
    trainer = OfflineTrainer(
        agent=restored,
        replay_buffer=ReplayBuffer(batch),
        batch_size=16,
        rng=np.random.default_rng(0),
        state=TrainState(step=read_metadata(checkpoint_metadata_path(checkpoint_path)).step),
    )
    result = trainer.train_step()

    assert result.step == 2
    assert trainer.state.step == 2


def test_torch_iql_normalizes_state_goal_inputs() -> None:
    torch.manual_seed(0)
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=64), seed=5)
    agent = make_torch_iql_agent(batch, TorchIQLConfig(hidden_dim=16, normalize_inputs=True))

    state_goal = agent._state_goal(batch.observations, batch.goals)

    assert torch.allclose(state_goal.mean(dim=0), torch.zeros(state_goal.shape[1]), atol=1e-5)


def test_torch_iql_tanh_actor_bounds_predictions() -> None:
    torch.manual_seed(0)
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=64), seed=6)
    agent = make_torch_iql_agent(batch, TorchIQLConfig(hidden_dim=16, actor_output_activation="tanh"))

    actions = agent.predict(batch.observations[:8], batch.goals[:8])

    assert np.max(actions) <= 1.0
    assert np.min(actions) >= -1.0


def test_torch_iql_goal_direction_loss_runs_for_pointmaze_shapes() -> None:
    torch.manual_seed(0)
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=64, observation_dim=2, goal_dim=2, action_dim=2), seed=7)
    agent = make_torch_iql_agent(batch, TorchIQLConfig(hidden_dim=16, goal_direction_loss_weight=0.1))

    result = agent.train_step(batch, np.random.default_rng(0))

    assert result.metrics["iql_goal_direction_loss"] >= 0.0


def test_iql_critic_target_bootstraps_from_next_value_not_actor() -> None:
    torch.manual_seed(0)
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=8), seed=8)
    agent = make_torch_iql_agent(batch, TorchIQLConfig(hidden_dim=8, discount=0.9))
    for parameter in agent.value.parameters():
        parameter.data.zero_()
    agent.value.net[-1].bias.data.fill_(2.0)
    target_before = agent._critic_target(batch)
    for parameter in agent.actor.parameters():
        parameter.data.fill_(100.0)
    target_after = agent._critic_target(batch)
    expected = torch.as_tensor(batch.rewards.reshape(-1, 1), dtype=torch.float32)
    expected = expected + 0.9 * (1.0 - torch.as_tensor(batch.terminals.reshape(-1, 1), dtype=torch.float32)) * 2.0

    assert torch.allclose(target_before.cpu(), expected)
    assert torch.allclose(target_after, target_before)


def test_iql_value_loss_uses_minimum_target_q() -> None:
    torch.manual_seed(0)
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=8), seed=9)
    agent = make_torch_iql_agent(batch, TorchIQLConfig(hidden_dim=8))
    state_goal = agent._state_goal(batch.observations, batch.goals)
    actions = torch.as_tensor(batch.actions, dtype=torch.float32)
    for parameter in agent.target_critic1.parameters():
        parameter.data.zero_()
    for parameter in agent.target_critic2.parameters():
        parameter.data.zero_()
    agent.target_critic1.net[-1].bias.data.fill_(3.0)
    agent.target_critic2.net[-1].bias.data.fill_(1.0)

    minimum = agent._min_q(state_goal, actions, target=True)

    assert torch.allclose(minimum, torch.ones_like(minimum))


def test_iql_ddpgbc_actor_loss_reports_q_and_bc_terms() -> None:
    torch.manual_seed(0)
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=32), seed=10)
    agent = make_torch_iql_agent(
        batch,
        TorchIQLConfig(
            hidden_dim=16,
            hidden_layers=3,
            activation="gelu",
            actor_loss_mode="ddpgbc",
            actor_alpha=0.003,
        ),
    )

    result = agent.train_step(batch, np.random.default_rng(0))

    assert result.metrics["iql_actor_q_loss"] != 0.0
    assert result.metrics["iql_actor_bc_loss"] > 0.0
