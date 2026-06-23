"""Phase 5 synthesis: all models on one comparison, on real league data (E0).

    python -m research.exp_phase5 [--division E0] [--since 2005-08-01]

A. 1X2: base rate, Elo, Dixon-Coles, gradient boosting, vs the de-vigged market —
   walk-forward, identical splits, RPS / log loss / calibration (ECE).
B. Counts (corners): structured GLM count model vs LightGBM Poisson — MAE and
   Poisson deviance on the same holdout.

The point is an honest answer to "which model wins which target, and why".
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from football_forecast.data.schema import Fixture, match_outcome
from football_forecast.data.sources import club_matches as cm
from football_forecast.eval import metrics as M
from football_forecast.eval.calibration import expected_calibration_error
from football_forecast.eval.market import devig
from football_forecast.models.boosting import BoostingCountModel, BoostingModel
from football_forecast.models.counts import CountModel
from football_forecast.models.dixon_coles import DixonColesModel
from football_forecast.models.elo import EloModel

GOAL_MODELS = {"elo": EloModel, "dixon_coles": DixonColesModel, "boosting": BoostingModel}


def _origins(df, start_year):
    yr_max = int(pd.to_datetime(df["date"]).max().year)
    return [pd.Timestamp(year=y, month=8, day=1) for y in range(start_year, yr_max + 1)]


def part_a(df, start_year):
    print("== A. 1X2 comparison (walk-forward, vs market) ==")
    df = df.dropna(subset=["odds_h", "odds_d", "odds_a"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    origins = _origins(df, start_year)
    probs = {k: [] for k in ["base_rate", "market", *GOAL_MODELS]}
    outs: list[str] = []

    for i, origin in enumerate(origins):
        nxt = origins[i + 1] if i + 1 < len(origins) else pd.Timestamp.max
        train = df[df["date"] < origin]
        test = df[(df["date"] >= origin) & (df["date"] < nxt)]
        if len(train) < 800 or test.empty:
            continue
        fitted = {name: f().fit(df, asof=origin.date()) for name, f in GOAL_MODELS.items()}
        base = train.assign(o=[match_outcome(h, a) for h, a in zip(train.home_goals, train.away_goals)])
        rates = base["o"].value_counts(normalize=True).reindex(["H", "D", "A"]).fillna(0).to_dict()
        for r in test.itertuples(index=False):
            outs.append(match_outcome(r.home_goals, r.away_goals))
            fx = Fixture(r.home, r.away, pd.Timestamp(r.date).date(), r.competition, r.comp_type, False)
            probs["base_rate"].append(rates)
            probs["market"].append(devig(r.odds_h, r.odds_d, r.odds_a))
            for name, m in fitted.items():
                probs[name].append(m.predict_1x2(fx))

    print(f"  test matches: {len(outs)}")
    print(f"  {'model':12s} {'RPS':>7} {'logloss':>8} {'ECE':>6}")
    rows = []
    for name, ps in probs.items():
        rps = float(np.mean([M.rps(p, o) for p, o in zip(ps, outs)]))
        ll = float(np.mean([M.log_loss(p, o) for p, o in zip(ps, outs)]))
        ece = expected_calibration_error(ps, outs)
        rows.append((name, rps, ll, ece))
    for name, rps, ll, ece in sorted(rows, key=lambda x: x[1]):
        print(f"  {name:12s} {rps:7.4f} {ll:8.4f} {ece:6.3f}")


def _poisson_deviance(k, mu):
    k, mu = np.asarray(k, float), np.maximum(np.asarray(mu, float), 1e-9)
    term = np.where(k > 0, k * np.log(k / mu), 0.0)
    return float(2.0 * np.mean(term - (k - mu)))


def part_b(df, start_year, target="corners"):
    print(f"\n== B. Count target '{target}': GLM vs boosting (Poisson) ==")
    df = df.dropna(subset=[f"home_{target}", f"away_{target}"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    origins = _origins(df, start_year)
    err = {"glm": [], "boosting": []}
    dev_k, dev_mu = {"glm": [], "boosting": []}, {"glm": [], "boosting": []}

    for i, origin in enumerate(origins):
        nxt = origins[i + 1] if i + 1 < len(origins) else pd.Timestamp.max
        train = df[df["date"] < origin]
        test = df[(df["date"] >= origin) & (df["date"] < nxt)]
        if len(train) < 800 or test.empty:
            continue
        glm = CountModel(target, family="auto").fit(df, asof=origin.date())
        boost = BoostingCountModel(target, objective="poisson").fit(df, asof=origin.date())
        for r in test.itertuples(index=False):
            fx = Fixture(r.home, r.away, pd.Timestamp(r.date).date(), r.competition, r.comp_type, False)
            gh, ga = glm.expected_counts(r.home, r.away)
            bh, ba = boost.expected_counts(fx)
            actual = [getattr(r, f"home_{target}"), getattr(r, f"away_{target}")]
            err["glm"] += [abs(gh - actual[0]), abs(ga - actual[1])]
            err["boosting"] += [abs(bh - actual[0]), abs(ba - actual[1])]
            dev_k["glm"] += actual; dev_mu["glm"] += [gh, ga]
            dev_k["boosting"] += actual; dev_mu["boosting"] += [bh, ba]

    print(f"  {'model':10s} {'MAE':>7} {'PoissonDev':>11}")
    for name in ("glm", "boosting"):
        print(f"  {name:10s} {np.mean(err[name]):7.3f} {_poisson_deviance(dev_k[name], dev_mu[name]):11.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--division", default="E0")
    ap.add_argument("--since", default="2005-08-01")
    ap.add_argument("--raw", default="data/raw/club_matches.csv")
    ap.add_argument("--start-year", type=int, default=2012)
    args = ap.parse_args()

    df = cm.load(args.raw, division=args.division, since=args.since)
    print(f"{args.division}: {len(df)} matches\n")
    part_a(df, args.start_year)
    part_b(df, args.start_year)


if __name__ == "__main__":
    main()
