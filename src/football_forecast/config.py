"""Run configuration — one object that parameterizes a pipeline run.

Loadable from a TOML file (committed under `configs/`) or built from CLI args, so
every run is described by an explicit, recordable config rather than scattered
flags. Paired with `provenance.write_manifest`, this makes any artifact traceable
to exactly how it was produced (the project's reproducibility goal).
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    # --- data selection ---
    source: str = "intl"               # "intl" (national teams) | "league"
    division: str | None = None        # league division code, e.g. "E0"
    since: str | None = None           # earliest date to load (league)
    raw: str | None = None             # raw input path (derived if None)
    data: str | None = None            # processed parquet path (derived if None)

    # --- model ---
    model: str = "dixon_coles"
    seed: int = 0
    model_kwargs: dict = field(default_factory=dict)

    # --- backtest ---
    start_year: int = 1990
    min_train: int = 500

    # --- forecast ---
    forecast_mode: str = "upcoming"    # "upcoming" (fixtures queue) | "backfill"
    forecast_n: int = 200              # backfill: most-recent-N matches
    competition: str | None = None     # restrict upcoming forecast to one competition

    # --- output locations ---
    models_dir: str = "artifacts/models"
    metrics_path: str = "artifacts/metrics/backtests.parquet"
    store_path: str = "artifacts/forecasts/forecasts.sqlite"
    fixtures_path: str = "artifacts/fixtures/fixtures.sqlite"

    def __post_init__(self) -> None:
        if self.source not in ("intl", "league"):
            raise ValueError(f"source must be 'intl' or 'league', got {self.source!r}")
        if self.data is None:
            self.data = (
                "data/processed/intl_results.parquet" if self.source == "intl"
                else f"data/processed/league_{self.division or 'E0'}.parquet"
            )
        if self.raw is None:
            self.raw = (
                "data/raw/intl_results.csv" if self.source == "intl"
                else "data/raw/club_matches.csv"
            )

    @classmethod
    def from_toml(cls, path: str | Path) -> "RunConfig":
        with open(path, "rb") as fh:
            return cls(**tomllib.load(fh))

    def to_dict(self) -> dict:
        return asdict(self)
