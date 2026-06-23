"""Drain the fixtures forecast queue.

    python -m pipelines.process_queue [--model-file PKL] [--data PATH]
                                      [--fixtures PATH] [--model dixon_coles]

Computes forecasts for every pending fixture. Loads a fitted model from a pickle
(`--model-file`, the path the Pi would use after a sync) or, on the PC, fits one
fresh from processed data. Cheap inference only — no training happens here.
"""

from __future__ import annotations

import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd

from football_forecast.fixtures_queue import drain
from football_forecast.models.bayesian import BayesianModel
from football_forecast.models.boosting import BoostingModel
from football_forecast.models.dixon_coles import DixonColesModel, MaherModel
from football_forecast.models.elo import EloModel
from football_forecast.store import fixtures as fxstore

FACTORIES = {
    "elo": EloModel,
    "maher": MaherModel,
    "dixon_coles": DixonColesModel,
    "bayesian": BayesianModel,
    "boosting": BoostingModel,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="dixon_coles", choices=list(FACTORIES))
    ap.add_argument("--model-file", default=None, help="pickled fitted model (Pi path)")
    ap.add_argument("--data", default="data/processed/intl_results.parquet")
    ap.add_argument("--fixtures", default=fxstore.DEFAULT_PATH)
    args = ap.parse_args()

    if args.model_file and Path(args.model_file).exists():
        with open(args.model_file, "rb") as fh:
            model = pickle.load(fh)
        print(f"loaded fitted model from {args.model_file}")
    else:
        matches = pd.read_parquet(args.data)
        asof = (pd.to_datetime(matches["date"]).max() + timedelta(days=1)).date()
        model = FACTORIES[args.model]().fit(matches, asof=asof)
        print(f"fit {args.model} on {len(matches):,} matches (asof {asof})")

    n = drain(model, args.model, args.fixtures)
    pending = len(fxstore.list_pending(args.fixtures))
    print(f"processed {n} pending fixture(s); {pending} still pending -> {args.fixtures}")


if __name__ == "__main__":
    main()
