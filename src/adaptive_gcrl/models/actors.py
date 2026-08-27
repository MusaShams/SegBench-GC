"""Actor model placeholders for later PyTorch implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActorSpec:
    observation_dim: int
    action_dim: int
    hidden_dim: int = 256

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("Actor dimensions must be positive.")

