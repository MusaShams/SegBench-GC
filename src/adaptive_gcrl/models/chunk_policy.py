"""Action chunking helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChunkPolicy:
    chunk_size: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

    def to_chunks(self, primitive_actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(primitive_actions)
        if actions.ndim < 2:
            raise ValueError("primitive_actions must have shape (time, action_dim, ...).")
        usable = actions.shape[0] - (actions.shape[0] % self.chunk_size)
        if usable == 0:
            raise ValueError("Not enough actions to form one complete chunk.")
        chunk_shape = (usable // self.chunk_size, self.chunk_size) + actions.shape[1:]
        return actions[:usable].reshape(chunk_shape)

