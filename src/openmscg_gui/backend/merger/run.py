from __future__ import annotations

import sys

from . import config
from .merge import deterministic_merge
from .optimizer import optimize
from .system_io import discover_systems


def main() -> None:
    systems = discover_systems()
    print(f"Systems: {', '.join(system.name for system in systems)}")
    print(f"Output: {config.OUTPUT_DIR}")

    merge_result = deterministic_merge(systems)
    print(f"Deterministic tables: {merge_result.tables_dir}")

    if config.RUN_OPTIMIZER:
        print(f"Optimizer trials: {config.RUNS_DIR}")
        optimize(systems, merge_result.tables_dir)
    else:
        print("Optimizer disabled by config.RUN_OPTIMIZER")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
