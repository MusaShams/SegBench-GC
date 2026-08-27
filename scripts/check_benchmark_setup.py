"""Print benchmark dependency diagnostics as JSON."""

from __future__ import annotations

import json

from adaptive_gcrl.envs.benchmark_setup import ogbench_setup_status


def main() -> None:
    print(json.dumps(ogbench_setup_status(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

