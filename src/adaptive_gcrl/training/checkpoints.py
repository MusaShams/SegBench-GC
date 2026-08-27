"""Checkpoint metadata helpers that do not assume Git."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class CheckpointMetadata:
    step: int
    seed: int
    tfvc_changeset: Optional[str] = None
    rng_state: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative.")


def write_metadata(path: Path, metadata: CheckpointMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_metadata(path: Path) -> CheckpointMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CheckpointMetadata(**payload)


@runtime_checkable
class CheckpointableAgent(Protocol):
    def save_checkpoint(self, path: Path) -> None:
        """Persist agent state to a checkpoint path."""


@runtime_checkable
class LoadableCheckpointAgent(Protocol):
    def load_checkpoint(self, path: Path) -> None:
        """Load agent state from a checkpoint path."""


def checkpoint_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def save_agent_checkpoint(agent: object, path: Path, metadata: CheckpointMetadata) -> None:
    if not isinstance(agent, CheckpointableAgent):
        raise TypeError(f"Agent of type {type(agent).__name__} does not support checkpointing.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    agent.save_checkpoint(temporary_path)
    temporary_path.replace(path)
    write_metadata(checkpoint_metadata_path(path), metadata)


def load_agent_checkpoint(agent: object, path: Path) -> None:
    if not isinstance(agent, LoadableCheckpointAgent):
        raise TypeError(f"Agent of type {type(agent).__name__} does not support checkpoint loading.")
    agent.load_checkpoint(path)
