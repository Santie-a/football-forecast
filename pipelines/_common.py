"""Shared pipeline plumbing: common CLI args and RunConfig resolution.

Every stage accepts either a `--config <toml>` (used wholesale) or individual
flags (which fall back to RunConfig defaults). Keeps stages thin and their flags
consistent (docs/workflow.md).
"""

from __future__ import annotations

import argparse

from football_forecast.config import RunConfig

# RunConfig fields that common/stage flags may populate.
_FIELDS = (
    "source", "division", "data", "raw", "since", "model", "seed",
    "start_year", "min_train", "competition", "forecast_mode", "forecast_n",
)


def add_common_args(ap: argparse.ArgumentParser, *, model_choices=None) -> None:
    ap.add_argument("--config", help="TOML run config (used wholesale; ignores other flags)")
    ap.add_argument("--source", choices=["intl", "league"])
    ap.add_argument("--division")
    ap.add_argument("--data", help="processed parquet (derived from source if omitted)")
    ap.add_argument("--raw", help="raw input path (derived from source if omitted)")
    ap.add_argument("--since")
    ap.add_argument("--seed", type=int)
    if model_choices is not None:
        ap.add_argument("--model", choices=model_choices)


def base_config(args: argparse.Namespace) -> RunConfig:
    """Build a RunConfig from --config (wholesale) or the provided flags."""
    if getattr(args, "config", None):
        return RunConfig.from_toml(args.config)
    kw = {f: getattr(args, f) for f in _FIELDS if getattr(args, f, None) is not None}
    return RunConfig(**kw)
