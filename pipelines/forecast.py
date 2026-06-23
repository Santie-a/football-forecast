"""Forecast stage: write forecasts to the store for the dashboard to read.

    python -m pipelines.forecast [--model elo|maher|dixon_coles] [--n 200]
                                 [--data PATH] [--store PATH]

Goal models (those exposing `rates`) publish the full bundle — 1X2, the scoreline
matrix, over/under and correct-score — all summed from one matrix. Result-only
models (Elo) publish 1X2 only. Phase 1 stand-in for "upcoming fixtures": forecasts
the most recent N matches with a model fit only on data *before* them (no leakage).
"""

from __future__ import annotations

import argparse

import pandas as pd

from football_forecast.data.schema import Fixture
from football_forecast.forecast.bundle import markets_for_model
from football_forecast.models.bayesian import BayesianModel
from football_forecast.models.dixon_coles import DixonColesModel, MaherModel
from football_forecast.models.elo import EloModel
from football_forecast.store.forecasts import DEFAULT_PATH, write_forecasts

FACTORIES = {
    "elo": EloModel,
    "maher": MaherModel,
    "dixon_coles": DixonColesModel,
    "bayesian": BayesianModel,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="dixon_coles", choices=list(FACTORIES))
    ap.add_argument("--n", type=int, default=200, help="most recent N matches to forecast")
    ap.add_argument("--data", default="data/processed/intl_results.parquet")
    ap.add_argument("--store", default=DEFAULT_PATH)
    args = ap.parse_args()

    matches = pd.read_parquet(args.data).sort_values("date", kind="stable")
    recent = matches.tail(args.n)
    cutoff = pd.to_datetime(recent["date"]).min().date()
    model = FACTORIES[args.model]().fit(matches, asof=cutoff)

    rows = []
    for r in recent.itertuples(index=False):
        fixture = Fixture(
            r.home, r.away, pd.Timestamp(r.date).date(), r.competition, r.comp_type, bool(r.neutral)
        )
        for market, payload in markets_for_model(model, fixture).items():
            rows.append(
                {
                    "match_id": r.match_id,
                    "date": r.date,
                    "home": r.home,
                    "away": r.away,
                    "competition": r.competition,
                    "model": args.model,
                    "market": market,
                    "payload": payload,
                }
            )

    n = write_forecasts(pd.DataFrame(rows), args.store)
    markets = sorted({row["market"] for row in rows})
    print(
        f"wrote {n} rows ({', '.join(markets)}) for {args.model} "
        f"(fit asof {cutoff}) -> {args.store}"
    )


if __name__ == "__main__":
    main()
