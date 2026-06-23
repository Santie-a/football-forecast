"""Forecast stage: produce forecasts for matches that haven't happened.

    python -m pipelines.forecast [--mode upcoming|backfill] [--model dixon_coles]
                                 [--data PATH | --source league --division E0]

- **upcoming** (default): forecast the unplayed fixtures in the fixtures store
  (the queue) with a freshly-fit model — the real "predict the next matches" path,
  unified with the fixtures store/queue. Writes the market bundle per fixture.
- **backfill**: forecast the most-recent-N played matches into the read-only
  forecasts store, so the dashboard's main page has history. Fit strictly before
  those matches (no leakage).

Both write a `.manifest.json` (config, seed, git commit, data fingerprint).
"""

from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from football_forecast.data.schema import Fixture
from football_forecast.fixtures_queue import drain
from football_forecast.forecast.bundle import markets_for_model
from football_forecast.models.bayesian import BayesianModel
from football_forecast.models.boosting import BoostingModel
from football_forecast.models.dixon_coles import DixonColesModel, MaherModel
from football_forecast.models.elo import EloModel
from football_forecast.provenance import write_manifest
from football_forecast.store import fixtures as fxstore
from football_forecast.store.forecasts import write_forecasts
from pipelines import _common

FACTORIES = {
    "elo": EloModel, "maher": MaherModel, "dixon_coles": DixonColesModel,
    "bayesian": BayesianModel, "boosting": BoostingModel,
}


def _build(model: str, kwargs: dict, seed: int):
    kw = dict(kwargs)
    if model in ("bayesian", "boosting") and "seed" not in kw:
        kw["seed"] = seed
    return FACTORIES[model](**kw)


def _upcoming(cfg) -> None:
    matches = pd.read_parquet(cfg.data)
    asof = (pd.to_datetime(matches["date"]).max() + timedelta(days=1)).date()
    model = _build(cfg.model, cfg.model_kwargs, cfg.seed).fit(matches, asof=asof)
    n = drain(model, cfg.model, cfg.fixtures_path)
    write_manifest(
        cfg.fixtures_path, "forecast-upcoming", config=cfg.to_dict(), seed=cfg.seed,
        inputs=[cfg.data], metrics={"fit_asof": str(asof), "n_forecast": n},
    )
    pending = len(fxstore.list_pending(cfg.fixtures_path))
    print(f"forecast {n} upcoming fixture(s) with {cfg.model} (fit asof {asof}); "
          f"{pending} still pending -> {cfg.fixtures_path}")


def _backfill(cfg) -> None:
    matches = pd.read_parquet(cfg.data).sort_values("date", kind="stable")
    recent = matches.tail(cfg.forecast_n)
    cutoff = pd.to_datetime(recent["date"]).min().date()
    model = _build(cfg.model, cfg.model_kwargs, cfg.seed).fit(matches, asof=cutoff)
    rows = []
    for r in recent.itertuples(index=False):
        fx = Fixture(r.home, r.away, pd.Timestamp(r.date).date(), r.competition, r.comp_type, bool(r.neutral))
        for market, payload in markets_for_model(model, fx).items():
            rows.append({"match_id": r.match_id, "date": r.date, "home": r.home,
                         "away": r.away, "competition": r.competition,
                         "model": cfg.model, "market": market, "payload": payload})
    n = write_forecasts(pd.DataFrame(rows), cfg.store_path)
    write_manifest(
        cfg.store_path, "forecast-backfill", config=cfg.to_dict(), seed=cfg.seed,
        inputs=[cfg.data], metrics={"fit_asof": str(cutoff), "n_rows": n},
    )
    print(f"backfilled {n} rows for {cfg.model} (fit asof {cutoff}) -> {cfg.store_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    _common.add_common_args(ap, model_choices=list(FACTORIES))
    ap.add_argument("--mode", choices=["upcoming", "backfill"], default=None)
    ap.add_argument("--n", dest="forecast_n", type=int, help="backfill: recent N matches")
    args = ap.parse_args()
    cfg = _common.base_config(args)
    mode = args.mode or cfg.forecast_mode
    (_upcoming if mode == "upcoming" else _backfill)(cfg)


if __name__ == "__main__":
    main()
