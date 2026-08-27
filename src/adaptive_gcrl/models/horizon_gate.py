"""Horizon selection utilities for adaptive temporal abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class HorizonScore:
    horizon: int
    index: int
    probabilities: tuple[float, ...]
    adjusted_values: tuple[float, ...]


class HorizonGate:
    def __init__(self, horizons: Sequence[int], *, temperature: float = 1.0, uncertainty_penalty: float = 0.0) -> None:
        if not horizons:
            raise ValueError("horizons must not be empty.")
        if any(horizon <= 0 for horizon in horizons):
            raise ValueError("horizons must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if uncertainty_penalty < 0.0:
            raise ValueError("uncertainty_penalty must be non-negative.")
        self.horizons = tuple(int(horizon) for horizon in horizons)
        self.temperature = float(temperature)
        self.uncertainty_penalty = float(uncertainty_penalty)

    def select(self, values: Sequence[float], uncertainties: Optional[Sequence[float]] = None) -> HorizonScore:
        value_array = np.asarray(values, dtype=float)
        if value_array.shape != (len(self.horizons),):
            raise ValueError(f"values must have shape ({len(self.horizons)},).")
        if uncertainties is None:
            uncertainty_array = np.zeros_like(value_array)
        else:
            uncertainty_array = np.asarray(uncertainties, dtype=float)
            if uncertainty_array.shape != value_array.shape:
                raise ValueError("uncertainties must match values shape.")
            if np.any(uncertainty_array < 0.0):
                raise ValueError("uncertainties must be non-negative.")

        adjusted = value_array - self.uncertainty_penalty * uncertainty_array
        logits = adjusted / self.temperature
        logits = logits - np.max(logits)
        probabilities = np.exp(logits)
        probabilities = probabilities / np.sum(probabilities)
        index = int(np.argmax(probabilities))
        return HorizonScore(
            horizon=self.horizons[index],
            index=index,
            probabilities=tuple(float(prob) for prob in probabilities),
            adjusted_values=tuple(float(value) for value in adjusted),
        )
