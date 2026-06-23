"""Backtest stage: walk-forward evaluation of models on identical splits.

    python -m pipelines.backtest [--model all|elo|...] [--start-year YEAR]
                                 [--data PATH | --source league --division E0]

Prints the RPS / log-loss / Brier table and writes per-origin metrics to parquet
with a `.manifest.json`. Acceptance checks: Elo > base rate (Phase 1); Dixon–Coles
> Elo on RPS and log loss (Phase 2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from football_forecast.eval.backtest import walk_forward, yearly_origins
from football_forecast.models.baseline import BaseRateModel
from football_forecast.models.boosting import BoostingModel
from football_forecast.models.dixon_coles import DixonColesModel, MaherModel
from football_forecast.models.elo import EloModel
from football_forecast.provenance import write_manifest
from pipelines import _common

FACTORIES = {
    "baseline": BaseRateModel,
    "elo": EloModel,
    "maher": MaherModel,
    "dixon_coles": DixonColesModel,
    "boosting": BoostingModel,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    _common.add_common_args(ap)  # --config/--source/--division/--data/--since/--seed
    ap.add_argument("--model", default="all", choices=["all", *FACTORIES])
    ap.add_argument("--start", "--start-year", dest="start_year", type=int)
    ap.add_argument("--min-train", dest="min_train", type=int)
    ap.add_argument("--out", help="metrics parquet (default from config)")
    args = ap.parse_args()
    cfg = _common.base_config(args)

    matches = pd.read_parquet(cfg.data)
    origins = yearly_origins(matches, start=cfg.start_year)
    names = list(FACTORIES) if args.model == "all" else [args.model]

    frames, overall = [], {}
    for name in names:
        res = walk_forward(FACTORIES[name], matches, origins, min_train_matches=cfg.min_train)
        overall[name] = res.overall
        frames.append(res.per_origin.assign(model=name))
        print(f"{name:12s} {res}")

    if "elo" in overall and "baseline" in overall:
        d = overall["baseline"]["rps"] - overall["elo"]["rps"]
        print(f"\nPhase 1 — Elo beats base rate on RPS: {'PASS' if d > 0 else 'FAIL'} ({d:+.4f})")
    if "dixon_coles" in overall and "elo" in overall:
        d_rps = overall["elo"]["rps"] - overall["dixon_coles"]["rps"]
        d_ll = overall["elo"]["log_loss"] - overall["dixon_coles"]["log_loss"]
        ok = d_rps > 0 and d_ll > 0
        print(f"Phase 2 — Dixon–Coles beats Elo on RPS AND log loss: "
              f"{'PASS' if ok else 'FAIL'} (dRPS {d_rps:+.4f}, dLogLoss {d_ll:+.4f})")

    out = args.out or cfg.metrics_path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    write_manifest(out, "backtest", config=cfg.to_dict(), seed=cfg.seed,
                   inputs=[cfg.data], metrics=overall)
    print(f"wrote per-origin metrics to {out}")


if __name__ == "__main__":
    main()
