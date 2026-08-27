"""Benchmark dependency diagnostics."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    installed: bool
    detail: Optional[str] = None


def module_status(module_name: str) -> DependencyStatus:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return DependencyStatus(name=module_name, installed=False, detail="module not importable")
    return DependencyStatus(name=module_name, installed=True, detail=spec.origin)


def ogbench_setup_status() -> dict:
    statuses = [module_status("ogbench"), module_status("mujoco"), module_status("gymnasium")]
    mujoco_path = os.environ.get("MUJOCO_PATH")
    ready = all(status.installed for status in statuses)
    return {
        "dependencies": [asdict(status) for status in statuses],
        "mujoco_path": mujoco_path,
        "ready": ready,
        "notes": (
            "Install benchmark extras with the pinned MuJoCo wheel before OGBench runs. "
            "MUJOCO_PATH is optional when the wheel-installed mujoco package imports successfully."
        ),
    }
