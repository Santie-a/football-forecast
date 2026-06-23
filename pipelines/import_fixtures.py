"""Import fixtures into the fixtures store, from any competition.

From the raw results source (carries unplayed future fixtures with blank scores):

    python -m pipelines.import_fixtures --source raw --competition "FIFA World Cup" \
        --since 2026-06-01 [--forecast-asof 2026-06-11]

From a plain CSV (columns: date, home, away[, competition, comp_type, neutral,
home_goals, away_goals]):

    python -m pipelines.import_fixtures --source csv --csv data/fixtures/my.csv

Played matches are registered with their result; unplayed ones are queued (drain
them with `pipelines.process_queue`). `--forecast-asof` fits a Dixon–Coles model
strictly before that date and attaches its forecast to the played matches (a
leakage-free pre-event forecast, e.g. for a tournament).
"""

from __future__ import annotations

import argparse

import pandas as pd

from football_forecast import fixtures_import as fi
from football_forecast.models.dixon_coles import DixonColesModel
from football_forecast.store import fixtures as fxstore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["raw", "csv"], default="raw")
    ap.add_argument("--raw", default="data/raw/intl_results.csv")
    ap.add_argument("--csv", help="path to a generic fixtures CSV (for --source csv)")
    ap.add_argument("--competition", help="filter (raw source)")
    ap.add_argument("--since", help="earliest date YYYY-MM-DD (raw source)")
    ap.add_argument("--until", help="latest date YYYY-MM-DD (raw source)")
    ap.add_argument("--forecast-asof", help="fit a pre-event model before this date for played matches")
    ap.add_argument("--data", default="data/processed/intl_results.parquet")
    ap.add_argument("--fixtures", default=fxstore.DEFAULT_PATH)
    args = ap.parse_args()

    if args.source == "csv":
        if not args.csv:
            ap.error("--csv is required when --source csv")
        df = fi.from_csv(args.csv)
    else:
        df = fi.from_raw_results(args.raw, args.competition, args.since, args.until)
    if df.empty:
        print("no fixtures matched the filters — nothing imported")
        return

    model = None
    if args.forecast_asof:
        matches = pd.read_parquet(args.data)
        model = DixonColesModel().fit(matches, asof=pd.Timestamp(args.forecast_asof).date())
        print(f"fit pre-event Dixon–Coles asof {args.forecast_asof} for played-match forecasts")

    counts = fi.register(df, path=args.fixtures, model=model)
    print(
        f"imported {counts['total']} fixtures -> {args.fixtures}: "
        f"{counts['played']} played, {counts['queued']} queued"
    )
    if counts["queued"]:
        print("run 'python -m pipelines.process_queue' to forecast the queued fixtures")


if __name__ == "__main__":
    main()
