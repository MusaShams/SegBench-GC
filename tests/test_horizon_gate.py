import numpy as np
import pytest

from adaptive_gcrl.models.chunk_policy import ChunkPolicy
from adaptive_gcrl.models.horizon_gate import HorizonGate


def test_horizon_gate_penalizes_uncertainty() -> None:
    gate = HorizonGate([1, 4], uncertainty_penalty=1.0)

    selected = gate.select(values=[2.0, 3.0], uncertainties=[0.0, 2.0])

    assert selected.horizon == 1
    assert sum(selected.probabilities) == pytest.approx(1.0)


def test_chunk_policy_forms_complete_chunks() -> None:
    actions = np.arange(12).reshape(6, 2)

    chunks = ChunkPolicy(chunk_size=3).to_chunks(actions)

    assert chunks.shape == (2, 3, 2)
    assert chunks[0, 0, 0] == 0

