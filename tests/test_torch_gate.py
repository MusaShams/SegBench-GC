import pytest
import numpy as np

torch = pytest.importorskip("torch")

from adaptive_gcrl.models.torch_gate import TorchHorizonGate, TorchHorizonGateConfig


def test_torch_horizon_gate_trains_and_selects_valid_horizon() -> None:
    torch.manual_seed(0)
    gate = TorchHorizonGate(TorchHorizonGateConfig(horizons=(1, 2, 4), hidden_dim=8, uncertainty_penalty=0.1))

    metrics = gate.fit_step(values=[0.1, 0.5, 0.2], uncertainties=[0.0, 0.1, 0.0], target_index=1)
    selected = gate.select(values=[0.1, 0.5, 0.2], uncertainties=[0.0, 0.1, 0.0])

    assert metrics["gate_loss"] >= 0.0
    assert metrics["gate_entropy"] >= 0.0
    assert metrics["gate_target_probability"] == 1.0
    assert selected.horizon in {1, 2, 4}


def test_torch_horizon_gate_supports_smoothed_targets_and_sampling() -> None:
    torch.manual_seed(0)
    gate = TorchHorizonGate(TorchHorizonGateConfig(horizons=(1, 2, 4), hidden_dim=8, target_smoothing=0.3))

    metrics = gate.fit_step(values=[0.1, 0.5, 0.2], uncertainties=[0.0, 0.1, 0.0], target_index=1)
    selected = gate.select(
        values=[0.1, 0.5, 0.2],
        uncertainties=[0.0, 0.1, 0.0],
        strategy="sample",
        rng=np.random.default_rng(0),
    )

    assert metrics["gate_target_probability"] == pytest.approx(0.8)
    assert selected.horizon in {1, 2, 4}


def test_torch_horizon_gate_supports_per_transition_targets() -> None:
    torch.manual_seed(0)
    gate = TorchHorizonGate(TorchHorizonGateConfig(horizons=(1, 2, 4), hidden_dim=8))

    metrics = gate.fit_step(
        values=[[0.9, 0.1, 0.0], [0.0, 0.1, 0.9]],
        uncertainties=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        target_index=[0, 2],
    )
    indices, probabilities = gate.select_batch(
        values=[[0.9, 0.1, 0.0], [0.0, 0.1, 0.9]],
        uncertainties=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )

    assert metrics["gate_loss"] >= 0.0
    assert indices.shape == (2,)
    assert probabilities.shape == (2, 3)
