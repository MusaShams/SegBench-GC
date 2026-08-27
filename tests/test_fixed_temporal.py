import numpy as np

from adaptive_gcrl.algorithms.fixed_temporal import FixedTemporalConfig, make_fixed_temporal_baseline
from adaptive_gcrl.data.synthetic import SyntheticGCRLConfig, make_synthetic_gcrl_batch


def test_fixed_temporal_baseline_reports_ablation_metadata() -> None:
    batch = make_synthetic_gcrl_batch(SyntheticGCRLConfig(num_transitions=32), seed=2)
    agent = make_fixed_temporal_baseline(batch, FixedTemporalConfig(horizon=4, chunk_size=2))

    result = agent.train_step(batch, np.random.default_rng(0))
    evaluation = agent.evaluate_batch(batch)

    assert result.metrics["horizon"] == 4.0
    assert result.metrics["chunk_size"] == 2.0
    assert evaluation.metrics["horizon"] == 4.0
    assert evaluation.metrics["chunk_size"] == 2.0
