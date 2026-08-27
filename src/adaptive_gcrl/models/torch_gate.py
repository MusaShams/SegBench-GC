"""Trainable PyTorch horizon gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from adaptive_gcrl.models.horizon_gate import HorizonScore


@dataclass(frozen=True)
class TorchHorizonGateConfig:
    horizons: tuple[int, ...]
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    uncertainty_penalty: float = 0.0
    target_smoothing: float = 0.0
    entropy_regularization: float = 0.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.uncertainty_penalty < 0.0:
            raise ValueError("uncertainty_penalty must be non-negative.")
        if not 0.0 <= self.target_smoothing < 1.0:
            raise ValueError("target_smoothing must be in [0, 1).")
        if self.entropy_regularization < 0.0:
            raise ValueError("entropy_regularization must be non-negative.")


class TorchHorizonGate:
    """Small classifier that learns to imitate or refine horizon choices."""

    def __init__(self, config: TorchHorizonGateConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        input_dim = 2 * len(config.horizons)
        self.model = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, len(config.horizons)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.step = 0

    def _features(self, values: Sequence[float], uncertainties: Sequence[float]) -> torch.Tensor:
        value_array = np.asarray(values, dtype=np.float32)
        uncertainty_array = np.asarray(uncertainties, dtype=np.float32)
        if value_array.ndim == 1:
            value_array = value_array.reshape(1, -1)
        if uncertainty_array.ndim == 1:
            uncertainty_array = uncertainty_array.reshape(1, -1)
        expected_width = len(self.config.horizons)
        if value_array.ndim != 2 or value_array.shape[1] != expected_width:
            raise ValueError(f"values must have shape (batch, {expected_width}).")
        if uncertainty_array.shape != value_array.shape:
            raise ValueError("uncertainties must match values shape.")
        features = np.concatenate([value_array, uncertainty_array], axis=-1)
        return torch.as_tensor(features, dtype=torch.float32, device=self.device)

    def fit_step(
        self,
        values: Sequence[float],
        uncertainties: Sequence[float],
        target_index: Union[int, Sequence[int]],
    ) -> dict[str, float]:
        features = self._features(values, uncertainties)
        target_indices = np.asarray(target_index, dtype=np.int64)
        if target_indices.ndim == 0:
            target_indices = np.full(features.shape[0], int(target_indices), dtype=np.int64)
        if target_indices.shape != (features.shape[0],):
            raise ValueError(f"target_index must have shape ({features.shape[0]},).")
        if np.any(target_indices < 0) or np.any(target_indices >= len(self.config.horizons)):
            raise ValueError("target_index is out of range.")
        logits = self.model(features)
        target = torch.full(
            (features.shape[0], len(self.config.horizons)),
            self.config.target_smoothing / len(self.config.horizons),
            dtype=torch.float32,
            device=self.device,
        )
        target_rows = torch.arange(features.shape[0], device=self.device)
        target_columns = torch.as_tensor(target_indices, dtype=torch.long, device=self.device)
        target[target_rows, target_columns] += 1.0 - self.config.target_smoothing
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities_for_loss = torch.softmax(logits, dim=-1)
        entropy_for_loss = -torch.sum(
            probabilities_for_loss * torch.log(torch.clamp(probabilities_for_loss, min=1e-8)),
            dim=-1,
        )
        loss = -torch.sum(target * log_probabilities, dim=-1).mean()
        loss = loss - self.config.entropy_regularization * entropy_for_loss.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.step += 1
        probabilities = torch.softmax(logits.detach(), dim=-1)
        entropy = -torch.sum(probabilities * torch.log(torch.clamp(probabilities, min=1e-8)), dim=-1)
        return {
            "gate_loss": float(loss.detach().cpu()),
            "gate_entropy": float(entropy.mean().cpu()),
            "gate_target_probability": float(target[target_rows, target_columns].mean().detach().cpu()),
        }

    def probabilities(self, values: Sequence[float], uncertainties: Sequence[float]) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(self._features(values, uncertainties))
            return torch.softmax(logits, dim=-1).cpu().numpy()

    def select_batch(
        self,
        values: Sequence[float],
        uncertainties: Sequence[float],
        *,
        strategy: str = "argmax",
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self.probabilities(values, uncertainties)
        if strategy == "argmax":
            indices = np.argmax(probabilities, axis=-1).astype(int)
        elif strategy == "sample":
            if rng is None:
                raise ValueError("rng is required when strategy='sample'.")
            indices = np.asarray(
                [rng.choice(len(self.config.horizons), p=row) for row in probabilities],
                dtype=int,
            )
        else:
            raise ValueError("strategy must be 'argmax' or 'sample'.")
        return indices, probabilities

    def select(
        self,
        values: Sequence[float],
        uncertainties: Sequence[float],
        *,
        strategy: str = "argmax",
        rng: Optional[np.random.Generator] = None,
    ) -> HorizonScore:
        indices, batch_probabilities = self.select_batch(
            values,
            uncertainties,
            strategy=strategy,
            rng=rng,
        )
        if indices.shape != (1,):
            raise ValueError("select requires one set of horizon values; use select_batch for batched values.")
        index = int(indices[0])
        probabilities = batch_probabilities[0]
        value_array = np.asarray(values, dtype=float)
        uncertainty_array = np.asarray(uncertainties, dtype=float)
        if value_array.ndim == 2:
            value_array = value_array[0]
            uncertainty_array = uncertainty_array[0]
        adjusted = value_array - self.config.uncertainty_penalty * uncertainty_array
        return HorizonScore(
            horizon=self.config.horizons[index],
            index=index,
            probabilities=tuple(float(probability) for probability in probabilities),
            adjusted_values=tuple(float(value) for value in adjusted),
        )

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = int(state["step"])

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: Path) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))
