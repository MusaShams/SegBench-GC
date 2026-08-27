"""Fixed action-chunking baseline scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkedIQLConfig:
    chunk_size: int = 4
    horizon: int = 8

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")

