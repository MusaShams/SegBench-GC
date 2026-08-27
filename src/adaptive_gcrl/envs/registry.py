"""Small registry for benchmark environment specs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentSpec:
    suite: str
    task: str
    observation_mode: str = "state"


class EnvironmentRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, EnvironmentSpec] = {}

    def register(self, name: str, spec: EnvironmentSpec) -> None:
        if not name:
            raise ValueError("Environment name must not be empty.")
        self._specs[name] = spec

    def get(self, name: str) -> EnvironmentSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown environment spec: {name}") from exc

