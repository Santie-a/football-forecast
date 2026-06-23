"""Ingest stage for league data (goals + fouls/cards/corners + odds).

    python -m pipelines.ingest_league [--division E0] [--since 2005-08-01] [--no-fetch]

Fetches the consolidated club-matches CSV (once) and writes a validated,
canonical-schema parquet for one division to data/processed/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from football_forecast.data.sources import club_matches as cm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--division", default="E0")
    ap.add_argument("--since", default="2005-08-01")
    ap.add_argument("--raw", default="data/raw/club_matches.csv")
    ap.add_argument("--out", default=None, help="default data/processed/league_<division>.parquet")
    ap.add_argument("--no-fetch", action="store_true", help="use existing raw file")
    args = ap.parse_args()

    if not args.no_fetch and not Path(args.raw).exists():
        print(f"fetching {cm.RAW_URL} -> {args.raw}")
        cm.fetch_raw(args.raw)

    df = cm.load(args.raw, division=args.division, since=args.since)
    out = args.out or f"data/processed/league_{args.division}.parquet"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"wrote {len(df):,} {args.division} matches to {out} "
        f"({df['date'].min().date()} … {df['date'].max().date()}, "
        f"{df['home'].nunique()} teams)"
    )


if __name__ == "__main__":
    main()
