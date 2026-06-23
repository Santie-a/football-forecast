"""Orchestrate the full pipeline from one config: ingest → train → backtest → forecast.

    python -m pipelines.run --config configs/intl.toml
    python -m pipelines.run --source league --division E0 --model dixon_coles
    python -m pipelines.run --config configs/intl.toml --stages train,forecast --no-fetch

Each stage runs as `python -m pipelines.<stage>`, forwarding the same config (so the
stages stay the single source of truth and each writes its own manifest).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from pipelines import _common

ALL_STAGES = ["ingest", "train", "backtest", "forecast"]


def _write_config(cfg) -> str:
    """Serialize the resolved config to a temp TOML the stages can consume."""
    lines = []
    for k, v in cfg.to_dict().items():
        if v is None or (isinstance(v, dict) and not v):
            continue
        if isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, bool):
            lines.append(f"{k} = {str(v).lower()}")
        elif isinstance(v, dict):
            inner = ", ".join(f'{ik} = {iv!r}' for ik, iv in v.items())
            lines.append(f"{k} = {{ {inner} }}")
        else:
            lines.append(f"{k} = {v}")
    fd = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    fd.write("\n".join(lines) + "\n")
    fd.close()
    return fd.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    _common.add_common_args(ap, model_choices=None)
    ap.add_argument("--model")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    cfg = _common.base_config(args)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    cfg_path = _write_config(cfg)
    print(f"resolved config -> {cfg_path}")
    try:
        for stage in stages:
            cmd = [sys.executable, "-m", f"pipelines.{stage}", "--config", cfg_path]
            if stage == "ingest" and args.no_fetch:
                cmd.append("--no-fetch")
            print(f"\n=== {stage} ===\n$ {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
    finally:
        Path(cfg_path).unlink(missing_ok=True)
    print("\npipeline complete.")


if __name__ == "__main__":
    main()
