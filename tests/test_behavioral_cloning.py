import numpy as np

from adaptive_gcrl.algorithms.bc import BehavioralCloningConfig, make_behavioral_cloning_agent
from adaptive_gcrl.data.synthetic import SyntheticGCRLConfig, make_synthetic_gcrl_batch
from adaptive_gcrl.training.trainer import OfflineTrainer
from adaptive_gcrl.data.replay_buffer import ReplayBuffer


def test_behavioral_cloning_reduces_action_mse() -> None:
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=128, noise_std=0.0), seed=7)
    agent = make_behavioral_cloning_agent(batch, BehavioralCloningConfig(learning_rate=0.1))
    rng = np.random.default_rng(0)

    initial = agent.evaluate_batch(batch).metrics["eval_action_mse"]
    for _ in range(50):
        agent.train_step(batch, rng)
    final = agent.evaluate_batch(batch).metrics["eval_action_mse"]

    assert final < initial * 0.25


def test_offline_trainer_advances_state() -> None:
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=64), seed=0)
    agent = make_behavioral_cloning_agent(batch)
    trainer = OfflineTrainer(agent=agent, replay_buffer=ReplayBuffer(batch), batch_size=16, rng=np.random.default_rng(0))

    result = trainer.train_step()

    assert result.step == 1
    assert trainer.state.step == 1
    assert "bc_loss" in result.metrics

