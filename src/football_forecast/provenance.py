"""Run manifests — a JSON sidecar recording how each artifact was produced.

`write_manifest(artifact)` drops `<artifact>.manifest.json` next to the output
with: stage, UTC timestamp, git commit, Python version, seed, the full run config,
fingerprints (size + sha256) of input files, output paths, and resulting metrics.
Any forecast/model/metric in `artifacts/` is then traceable to its exact inputs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def file_fingerprint(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest()[:16]}


def write_manifest(
    artifact: str | Path,
    stage: str,
    *,
    config: dict | None = None,
    seed: int | None = None,
    inputs: list[str | Path] | None = None,
    outputs: list[str | Path] | None = None,
    metrics: dict | None = None,
) -> Path:
    """Write `<artifact>.manifest.json` and return its path."""
    manifest = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "seed": seed,
        "config": config,
        "inputs": [fp for p in (inputs or []) if (fp := file_fingerprint(p))],
        "outputs": [str(o) for o in (outputs or [artifact])],
        "metrics": metrics or {},
    }
    dest = Path(f"{artifact}.manifest.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2, default=str))
    return dest
