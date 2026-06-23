"""Deprecated shim — league ingest is now part of the source-aware ingest stage.

Prefer:
    python -m pipelines.ingest --source league --division E0 [--since 2005-08-01]

This wrapper forwards to that for back-compatibility.
"""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    if "--source" not in sys.argv:
        sys.argv += ["--source", "league"]
    if "--division" not in sys.argv:
        sys.argv += ["--division", "E0"]
    sys.argv[0] = "pipelines.ingest"
    runpy.run_module("pipelines.ingest", run_name="__main__")


if __name__ == "__main__":
    main()
