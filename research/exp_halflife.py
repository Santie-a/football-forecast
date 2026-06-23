"""Experiment: select the Dixon–Coles time-decay half-life by backtest RPS (M02).

    python -m research.exp_halflife [--start YEAR]

Sweeps the half-life over a few values on identical walk-forward splits and
reports RPS / log loss, so the chosen default is justified, not guessed.
"""

from __future__ import annotations

import argparse

import pandas as pd

from football_forecast.eval.backtest import walk_forward, yearly_origins
from football_forecast.models.dixon_coles import DixonColesModel

HALF_LIVES = [365, 540, 730, 1460, None]  # days; None = no decay (all history equal)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=2005)
    ap.add_argument("--data", default="data/processed/intl_results.parquet")
    args = ap.parse_args()

    matches = pd.read_parquet(args.data)
    origins = yearly_origins(matches, start=args.start)

    print(f"half-life sweep, origins {args.start}–{int(matches['date'].max().year)}")
    print(f"{'half_life':>10}  {'RPS':>8}  {'log_loss':>9}")
    best = None
    for hl in HALF_LIVES:
        res = walk_forward(
            lambda hl=hl: DixonColesModel(half_life_days=hl),
            matches, origins, min_train_matches=500,
        )
        label = "none" if hl is None else str(hl)
        print(f"{label:>10}  {res.overall['rps']:.4f}  {res.overall['log_loss']:.4f}")
        if best is None or res.overall["rps"] < best[1]:
            best = (label, res.overall["rps"])
    print(f"\nbest half-life by RPS: {best[0]} days (RPS {best[1]:.4f})")


if __name__ == "__main__":
    main()
