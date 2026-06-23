"""Train stage: fit a model on all available history and persist it.

    python -m pipelines.train [--model elo] [--data PATH] [--out DIR]

For Elo there is no randomness (no seed needed). The fitted model is pickled to
artifacts/models/ with its config, so any forecast is traceable to how it was
produced (docs/workflow.md → reproducing results).
"""

from __future__ import annotations

import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd

from football_forecast.models.bayesian import BayesianModel
from football_forecast.models.dixon_coles import DixonColesModel, MaherModel
from football_forecast.models.elo import EloModel

FACTORIES = {
    "elo": EloModel,
    "maher": MaherModel,
    "dixon_coles": DixonColesModel,
    "bayesian": BayesianModel,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="elo", choices=list(FACTORIES))
    ap.add_argument("--data", default="data/processed/intl_results.parquet")
    ap.add_argument("--out", default="artifacts/models")
    args = ap.parse_args()

    matches = pd.read_parquet(args.data)
    asof = (pd.to_datetime(matches["date"]).max() + timedelta(days=1)).date()
    model = FACTORIES[args.model]().fit(matches, asof=asof)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    dest = Path(args.out) / f"{args.model}-{asof}.pkl"
    with open(dest, "wb") as fh:
        pickle.dump(model, fh)
    print(f"fitted {args.model} on {len(matches):,} matches (asof {asof}) -> {dest}")


if __name__ == "__main__":
    main()
