"""Baseline reproduction helper for small benchmark slices."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_gcrl.utils.config import load_config_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    config = load_config_files(parse_args().config)
    algorithm = config.get("algorithm", "unknown")
    task = config.get("task", "unknown")
    print(f"Prepared baseline reproduction for algorithm={algorithm}, task={task}.")


if __name__ == "__main__":
    main()

