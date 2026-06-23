"""Phase 4 acceptance on real league data (English Premier League, E0).

    python -m research.exp_phase4 [--division E0] [--since 2005-08-01]

Three parts:
  A. Dispersion per target -> Poisson vs NegBin (resolves M04).
  B. Walk-forward Dixon-Coles 1X2 vs the de-vigged closing-odds market and the
     base rate (the headline benchmark).
  C. Count-model predictive coverage (calibration) on a holdout season.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from football_forecast.data.sources import club_matches as cm
from football_forecast.eval import metrics as M
from football_forecast.eval.market import devig
from football_forecast.models.counts import CountModel
from football_forecast.models.dixon_coles import DixonColesModel
from football_forecast.data.schema import Fixture, match_outcome


def part_a_dispersion(df):
    print("== A. Dispersion per target (residual var/mean -> family) ==")
    asof = pd.to_datetime(df["date"]).max() + pd.Timedelta(days=1)
    for target in ("fouls", "corners", "yellow"):
        if f"home_{target}" not in df.columns:
            continue
        m = CountModel(target, family="auto").fit(df, asof=asof.date())
        print(f"  {target:8s} dispersion {m.dispersion_:.2f}  alpha {m.alpha_:.3f}  -> {m.family_}")
    # goals dispersion (marginal, for reference)
    g = pd.concat([df["home_goals"], df["away_goals"]]).astype(float)
    print(f"  goals    marginal var/mean {g.var() / g.mean():.2f}  (modelled as Poisson)")


def part_b_market(df, start_year):
    print("\n== B. Dixon-Coles vs market (de-vigged closing odds) vs base rate ==")
    df = df.dropna(subset=["odds_h", "odds_d", "odds_a"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    origins = [pd.Timestamp(year=y, month=8, day=1) for y in range(start_year, int(df["date"].max().year) + 1)]
    pooled = {"dixon_coles": [], "market": [], "base_rate": []}
    for i, origin in enumerate(origins):
        nxt = origins[i + 1] if i + 1 < len(origins) else pd.Timestamp.max
        train = df[df["date"] < origin]
        test = df[(df["date"] >= origin) & (df["date"] < nxt)]
        if len(train) < 500 or test.empty:
            continue
        dc = DixonColesModel().fit(df, asof=origin.date())
        base = train.assign(o=[match_outcome(h, a) for h, a in zip(train["home_goals"], train["away_goals"])])
        base_rates = base["o"].value_counts(normalize=True).reindex(["H", "D", "A"]).fillna(0).to_dict()
        for r in test.itertuples(index=False):
            o = match_outcome(r.home_goals, r.away_goals)
            fx = Fixture(r.home, r.away, pd.Timestamp(r.date).date(), r.competition, r.comp_type, False)
            pooled["dixon_coles"].append(M.rps(dc.predict_1x2(fx), o))
            pooled["market"].append(M.rps(devig(r.odds_h, r.odds_d, r.odds_a), o))
            pooled["base_rate"].append(M.rps(base_rates, o))
    n = len(pooled["market"])
    print(f"  test matches: {n}")
    for k in ("base_rate", "dixon_coles", "market"):
        print(f"  {k:12s} RPS {np.mean(pooled[k]):.4f}")
    gap = np.mean(pooled["dixon_coles"]) - np.mean(pooled["market"])
    print(f"  DC - market = {gap:+.4f} (market is the gold standard; close = strong)")


def part_c_coverage(df, holdout_year):
    print("\n== C. Count-model predictive coverage (calibration) ==")
    df["date"] = pd.to_datetime(df["date"])
    origin = pd.Timestamp(year=holdout_year, month=8, day=1)
    end = pd.Timestamp(year=holdout_year + 1, month=7, day=1)
    test = df[(df["date"] >= origin) & (df["date"] < end)]
    for target in ("fouls", "corners", "yellow"):
        if f"home_{target}" not in df.columns:
            continue
        m = CountModel(target, family="auto").fit(df, asof=origin.date())
        for level in (0.5, 0.8):
            cov = tot = 0
            for r in test.itertuples(index=False):
                for actor, opp, ih, col in [(r.home, r.away, 1.0, f"home_{target}"),
                                            (r.away, r.home, 0.0, f"away_{target}")]:
                    val = getattr(r, col)
                    if pd.isna(val):
                        continue
                    lo, hi = m.interval(m._expected(actor, opp, ih), level)
                    cov += lo <= val <= hi
                    tot += 1
            print(f"  {target:8s} ({m.family_}) {int(level*100)}% interval covers {cov/tot:.2f} (target {level:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--division", default="E0")
    ap.add_argument("--since", default="2005-08-01")
    ap.add_argument("--raw", default="data/raw/club_matches.csv")
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--holdout-year", type=int, default=2023)
    args = ap.parse_args()

    df = cm.load(args.raw, division=args.division, since=args.since)
    print(f"{args.division}: {len(df)} matches, {df['date'].min().date()}–{df['date'].max().date()}\n")
    part_a_dispersion(df)
    part_b_market(df, args.start_year)
    part_c_coverage(df, args.holdout_year)


if __name__ == "__main__":
    main()
