"""Ingest stage: fetch + normalize national-team results into processed data.

    python -m pipelines.ingest [--no-fetch] [--raw PATH] [--out PATH]

Writes a validated, canonical-schema parquet to data/processed/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from football_forecast.data.sources import intl_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/intl_results.csv")
    ap.add_argument("--out", default="data/processed/intl_results.parquet")
    ap.add_argument("--no-fetch", action="store_true", help="use existing raw file")
    args = ap.parse_args()

    if not args.no_fetch:
        print(f"fetching {intl_results.RAW_URL} -> {args.raw}")
        intl_results.fetch_raw(args.raw)

    df = intl_results.load(args.raw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(
        f"wrote {len(df):,} matches to {args.out} "
        f"({df['date'].min().date()} … {df['date'].max().date()}, "
        f"{df['home'].nunique()} teams)"
    )
    print(df["comp_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
