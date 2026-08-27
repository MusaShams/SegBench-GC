"""Trainer interfaces shared by algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from adaptive_gcrl.algorithms.base import OfflineAgent, TrainResult
from adaptive_gcrl.data.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class TrainState:
    step: int = 0

    def advance(self, amount: int = 1) -> "TrainState":
        if amount <= 0:
            raise ValueError("amount must be positive.")
        return TrainState(step=self.step + amount)


class Trainer(Protocol):
    def train_step(self) -> TrainState:
        """Run one optimizer update and return the updated train state."""


@dataclass
class OfflineTrainer:
    agent: OfflineAgent
    replay_buffer: ReplayBuffer
    batch_size: int
    rng: np.random.Generator
    state: TrainState = field(default_factory=TrainState)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def train_step(self) -> TrainResult:
        batch = self.replay_buffer.sample(self.batch_size, self.rng)
        result = self.agent.train_step(batch, self.rng)
        self.state = self.state.advance()
        return result

