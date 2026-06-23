"""Phase 3 acceptance: Bayesian hierarchical vs Dixon–Coles on a recent holdout.

    python -m research.exp_phase3 [--origin 2025-06-01] [--end 2026-06-22]

Both models train on data before `--origin` and are scored on matches in
[origin, end) — identical, time-ordered split. Reports RPS / log loss for each,
the Bayesian fit's max R-hat (convergence gate), and the key partial-pooling
check: do low-data teams get wider posterior attack intervals?

MCMC is expensive, so this is a single holdout (not the full walk-forward).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from football_forecast.data.schema import Fixture, match_outcome
from football_forecast.eval import metrics as M
from football_forecast.models.bayesian import BayesianModel
from football_forecast.models.dixon_coles import DixonColesModel


def _score(model, test) -> dict:
    rps, ll = [], []
    for r in test.itertuples(index=False):
        p = model.predict_1x2(
            Fixture(r.home, r.away, pd.Timestamp(r.date).date(), r.competition, r.comp_type, bool(r.neutral))
        )
        o = match_outcome(r.home_goals, r.away_goals)
        rps.append(M.rps(p, o))
        ll.append(M.log_loss(p, o))
    return {"rps": float(np.mean(rps)), "log_loss": float(np.mean(ll)), "n": len(rps)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origin", default="2025-06-01")
    ap.add_argument("--end", default="2026-06-22")
    ap.add_argument("--data", default="data/processed/intl_results.parquet")
    args = ap.parse_args()

    matches = pd.read_parquet(args.data)
    matches["date"] = pd.to_datetime(matches["date"])
    origin, end = pd.Timestamp(args.origin), pd.Timestamp(args.end)
    test = matches[(matches["date"] >= origin) & (matches["date"] < end)]
    print(f"holdout [{args.origin}, {args.end}): {len(test)} matches\n")

    t0 = time.time()
    dc = DixonColesModel().fit(matches, asof=origin.date())
    dc_s = _score(dc, test)
    print(f"dixon_coles  RPS {dc_s['rps']:.4f}  log_loss {dc_s['log_loss']:.4f}  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    bm = BayesianModel(draws=1000, tune=2500, chains=4, window_days=2000, seed=1).fit(
        matches, asof=origin.date()
    )
    bm_s = _score(bm, test)
    print(f"bayesian     RPS {bm_s['rps']:.4f}  log_loss {bm_s['log_loss']:.4f}  ({time.time()-t0:.0f}s)")
    print(f"             max R-hat {bm.max_rhat_:.4f}  (converged: {bm.converged})\n")

    d_rps = dc_s["rps"] - bm_s["rps"]
    d_ll = dc_s["log_loss"] - bm_s["log_loss"]
    print(f"Phase 3 — Bayesian vs Dixon–Coles: dRPS {d_rps:+.4f}, dLogLoss {d_ll:+.4f}")
    print("  (acceptance = converged AND matches/beats DC; close is fine — DC is strong)\n")

    # Partial-pooling check: posterior attack sd vs number of matches per team.
    teams = list(bm.idx_)
    n = np.array([bm.team_matches_[t] for t in teams])
    sd = np.array([bm.atk_std_[t] for t in teams])
    corr = np.corrcoef(np.log(n), sd)[0, 1]
    print(f"partial pooling: corr(log #matches, posterior attack sd) = {corr:+.3f} (expect < 0)")
    order = np.argsort(n)
    for label, i in [("sparsest", order[0]), ("densest", order[-1])]:
        print(f"  {label:8s} {teams[i]:<18} {n[i]:4d} matches  attack sd {sd[i]:.3f}")


if __name__ == "__main__":
    main()
