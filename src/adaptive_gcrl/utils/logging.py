"""JSONL metric logging."""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, Type


class MetricLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "MetricLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, metrics: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("MetricLogger must be used as a context manager.")
        self._handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        self._handle.flush()
