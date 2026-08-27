"""Lightweight provenance helpers for reproducible benchmark runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _run_git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def current_git_state(root: Path) -> dict[str, Any]:
    """Return commit and dirty-state metadata when Git is available."""
    commit = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def canonical_config_sha256(config: dict[str, Any]) -> str:
    """Hash a resolved configuration using stable JSON serialization."""
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash array shape, dtype, and contiguous byte representation."""
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(value.shape).encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_versions() -> dict[str, str | None]:
    """Return runtime versions relevant to SegBench-GC reproduction."""
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": package_version("numpy"),
        "torch_version": package_version("torch"),
        "ogbench_version": package_version("ogbench"),
        "mujoco_version": package_version("mujoco"),
    }
