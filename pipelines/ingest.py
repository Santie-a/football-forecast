"""Ingest stage: fetch + normalize source data into a processed parquet (+ manifest).

Source-aware (config/flags decide):
    python -m pipelines.ingest                          # national-team results (intl)
    python -m pipelines.ingest --source league --division E0
    python -m pipelines.ingest --config configs/league_e0.toml

Writes a validated canonical-schema parquet to cfg.data with a `.manifest.json`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from football_forecast.data.sources import club_matches, intl_results
from football_forecast.provenance import write_manifest
from pipelines import _common


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    _common.add_common_args(ap)
    ap.add_argument("--out", help="processed parquet (default from config)")
    ap.add_argument("--no-fetch", action="store_true", help="use existing raw file")
    args = ap.parse_args()
    cfg = _common.base_config(args)
    out = args.out or cfg.data

    if cfg.source == "intl":
        if not args.no_fetch and not Path(cfg.raw).exists():
            print(f"fetching {intl_results.RAW_URL} -> {cfg.raw}")
            intl_results.fetch_raw(cfg.raw)
        df = intl_results.load(cfg.raw)
        man_cfg = {"source": "intl", "raw": cfg.raw}
    else:
        if not args.no_fetch and not Path(cfg.raw).exists():
            print(f"fetching {club_matches.RAW_URL} -> {cfg.raw}")
            club_matches.fetch_raw(cfg.raw)
        df = club_matches.load(cfg.raw, division=cfg.division or "E0", since=cfg.since)
        man_cfg = {"source": "league", "division": cfg.division, "since": cfg.since, "raw": cfg.raw}

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    write_manifest(
        out, "ingest", config=man_cfg, inputs=[cfg.raw],
        metrics={"n_matches": len(df), "n_teams": int(df["home"].nunique()),
                 "date_min": str(df["date"].min().date()), "date_max": str(df["date"].max().date())},
    )
    print(f"wrote {len(df):,} matches to {out} "
          f"({df['date'].min().date()} … {df['date'].max().date()}, {df['home'].nunique()} teams)")


if __name__ == "__main__":
    main()
