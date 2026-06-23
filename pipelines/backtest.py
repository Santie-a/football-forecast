"""Backtest stage: walk-forward evaluation of models on identical splits.

    python -m pipelines.backtest [--model elo|baseline|both] [--start YEAR]
                                 [--data PATH] [--out PATH]

Prints the RPS / log-loss / Brier table and writes per-origin metrics to parquet.
The Phase 1 acceptance check: Elo must beat the base-rate baseline on RPS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from football_forecast.eval.backtest import walk_forward, yearly_origins
from football_forecast.models.baseline import BaseRateModel
from football_forecast.models.dixon_coles import DixonColesModel, MaherModel
from football_forecast.models.elo import EloModel

FACTORIES = {
    "baseline": BaseRateModel,
    "elo": EloModel,
    "maher": MaherModel,
    "dixon_coles": DixonColesModel,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", default="all",
        choices=["all", "baseline", "elo", "maher", "dixon_coles"],
    )
    ap.add_argument("--start", type=int, default=1990, help="first backtest origin year")
    ap.add_argument("--data", default="data/processed/intl_results.parquet")
    ap.add_argument("--out", default="artifacts/metrics/backtests.parquet")
    ap.add_argument("--min-train", type=int, default=500)
    args = ap.parse_args()

    matches = pd.read_parquet(args.data)
    origins = yearly_origins(matches, start=args.start)
    names = list(FACTORIES) if args.model == "all" else [args.model]

    frames = []
    overall = {}
    for name in names:
        res = walk_forward(
            FACTORIES[name], matches, origins, min_train_matches=args.min_train
        )
        overall[name] = res.overall
        frames.append(res.per_origin.assign(model=name))
        print(f"{name:12s} {res}")

    if "elo" in overall and "baseline" in overall:
        d = overall["baseline"]["rps"] - overall["elo"]["rps"]
        print(
            f"\nPhase 1 — Elo beats base rate on RPS: {'PASS' if d > 0 else 'FAIL'} "
            f"({d:+.4f})"
        )
    if "dixon_coles" in overall and "elo" in overall:
        dc, elo = overall["dixon_coles"], overall["elo"]
        d_rps = elo["rps"] - dc["rps"]
        d_ll = elo["log_loss"] - dc["log_loss"]
        ok = d_rps > 0 and d_ll > 0
        print(
            f"Phase 2 — Dixon–Coles beats Elo on RPS AND log loss: "
            f"{'PASS' if ok else 'FAIL'} (dRPS {d_rps:+.4f}, dLogLoss {d_ll:+.4f})"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(args.out, index=False)
    print(f"wrote per-origin metrics to {args.out}")


if __name__ == "__main__":
    main()
