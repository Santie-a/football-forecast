"""Seed the fixtures store with World Cup 2026 group-stage matches.

A thin convenience wrapper over the generic importer (`pipelines.import_fixtures`
/ `football_forecast.fixtures_import`) preset for WC2026: pulls the played and
upcoming WC2026 matches from the raw source, and attaches a **pre-tournament**
Dixon–Coles forecast (fit before `--asof`, the tournament start) to the played
ones. Equivalent to:

    python -m pipelines.import_fixtures --source raw --competition "FIFA World Cup" \
        --since 2026-06-01 --forecast-asof 2026-06-11
"""

from __future__ import annotations

import argparse

import pandas as pd

from football_forecast import fixtures_import as fi
from football_forecast.models.dixon_coles import DixonColesModel
from football_forecast.store import fixtures as fxstore

COMPETITION = "FIFA World Cup"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/intl_results.parquet")
    ap.add_argument("--raw", default="data/raw/intl_results.csv")
    ap.add_argument("--fixtures", default=fxstore.DEFAULT_PATH)
    ap.add_argument("--asof", default="2026-06-11", help="tournament start (forecast cutoff)")
    ap.add_argument("--season-start", default="2026-06-01")
    args = ap.parse_args()

    df = fi.from_raw_results(args.raw, COMPETITION, since=args.season_start)
    if df.empty:
        print("no WC2026 matches found in the raw data — nothing to seed")
        return

    matches = pd.read_parquet(args.data)
    model = DixonColesModel().fit(matches, asof=pd.Timestamp(args.asof).date())
    counts = fi.register(df, path=args.fixtures, model=model, model_name="dixon_coles")
    print(
        f"seeded WC2026 -> {args.fixtures}: {counts['played']} played "
        f"(pre-tournament forecasts), {counts['queued']} upcoming queued. "
        f"Run 'python -m pipelines.process_queue' to forecast the queue."
    )


if __name__ == "__main__":
    main()
