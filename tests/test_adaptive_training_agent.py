import numpy as np

from adaptive_gcrl.algorithms.adaptive_horizon import AdaptiveHorizonBaselineConfig, make_adaptive_horizon_training_agent
from adaptive_gcrl.data.synthetic import SyntheticGCRLConfig, make_synthetic_gcrl_batch


def test_adaptive_horizon_training_agent_reports_temporal_choice() -> None:
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=64), seed=4)
    agent = make_adaptive_horizon_training_agent(
        batch,
        AdaptiveHorizonBaselineConfig(horizons=(1, 2, 4), chunk_size=2, uncertainty_penalty=0.1),
    )

    result = agent.train_step(batch, np.random.default_rng(0))
    evaluation = agent.evaluate_batch(batch)

    assert result.metrics["selected_horizon"] in {1.0, 2.0, 4.0}
    assert result.metrics["chunk_size"] == 2.0
    assert evaluation.metrics["selected_horizon"] in {1.0, 2.0, 4.0}


def test_adaptive_horizon_training_agent_can_use_learned_gate() -> None:
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=64), seed=5)
    agent = make_adaptive_horizon_training_agent(
        batch,
        AdaptiveHorizonBaselineConfig(horizons=(1, 2, 4), chunk_size=2, learned_gate=True, gate_hidden_dim=8),
    )

    result = agent.train_step(batch, np.random.default_rng(0))
    evaluation = agent.evaluate_batch(batch)

    assert "gate_loss" in result.metrics
    assert result.metrics["target_horizon"] in {1.0, 2.0, 4.0}
    assert evaluation.metrics["learned_gate"] == 1.0
