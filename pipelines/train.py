"""Train stage: fit a model on all available history and persist it (+ manifest).

    python -m pipelines.train [--model dixon_coles] [--data PATH] [--seed N]
                              [--config run.toml]

The fitted model is pickled to artifacts/models/ alongside a `.manifest.json`
recording the config, seed, git commit, and input data fingerprint — so any
forecast is traceable to how it was produced (docs/workflow.md).
"""

from __future__ import annotations

import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd

from football_forecast.models.bayesian import BayesianModel
from football_forecast.models.boosting import BoostingModel
from football_forecast.models.dixon_coles import DixonColesModel, MaherModel
from football_forecast.models.elo import EloModel
from football_forecast.provenance import write_manifest
from pipelines import _common

FACTORIES = {
    "elo": EloModel,
    "maher": MaherModel,
    "dixon_coles": DixonColesModel,
    "bayesian": BayesianModel,
    "boosting": BoostingModel,
}


def _build(model: str, kwargs: dict, seed: int):
    cls = FACTORIES[model]
    kw = dict(kwargs)
    if model in ("bayesian", "boosting") and "seed" not in kw:
        kw["seed"] = seed
    return cls(**kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    _common.add_common_args(ap, model_choices=list(FACTORIES))
    ap.add_argument("--out", help="models dir (default from config)")
    args = ap.parse_args()
    cfg = _common.base_config(args)

    matches = pd.read_parquet(cfg.data)
    asof = (pd.to_datetime(matches["date"]).max() + timedelta(days=1)).date()
    model = _build(cfg.model, cfg.model_kwargs, cfg.seed).fit(matches, asof=asof)

    out_dir = Path(args.out or cfg.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{cfg.model}-{asof}.pkl"
    with open(dest, "wb") as fh:
        pickle.dump(model, fh)
    write_manifest(
        dest, "train", config=cfg.to_dict(), seed=cfg.seed, inputs=[cfg.data],
        metrics={"asof": str(asof), "n_train": len(matches)},
    )
    print(f"fitted {cfg.model} on {len(matches):,} matches (asof {asof}) -> {dest}")


if __name__ == "__main__":
    main()
