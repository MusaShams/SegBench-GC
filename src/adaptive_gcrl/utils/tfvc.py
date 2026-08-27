"""TFVC metadata helpers for experiment provenance."""

from __future__ import annotations

import subprocess
import os
from shutil import which
from typing import Optional


def tf_executable() -> Optional[str]:
    configured = os.environ.get("TFVC_CLI")
    if configured:
        return configured
    resolved = which("tf")
    return resolved


def current_tfvc_changeset(collection: str, workspace: str) -> Optional[str]:
    if not collection or not workspace:
        return None
    executable = tf_executable()
    if executable is None:
        return None
    command = [executable, "history", ".", "-recursive", "-stopafter:1", f"-collection:{collection}", f"-workspace:{workspace}"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            return parts[0]
    return None
