"""Count models for the league targets: fouls, corners, cards.

Each target is its own count model (docs/models-explained.md §10). The mean is a
log-linear team model fit by penalized Poisson MLE (analytic gradient, L2
shrinkage — same machinery as the goal model): for an actor team committing/earning
the count against an opponent,

    log E[count] = mu + home*[actor is home] + for_[actor] + against_[opponent]

The predictive distribution is **Poisson or negative binomial**, chosen from the
residual dispersion (resolves M04 per target empirically). NegBin dispersion is
estimated by method of moments. (No referee term — not in the reachable source; M06
deferred.)
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

NB_DISPERSION_TRIGGER = 1.15  # residual var/mean above this -> negative binomial


class CountModel:
    def __init__(self, target: str, family: str = "auto", reg: float = 0.05) -> None:
        self.target = target
        self.family = family
        self.reg = reg
        self.home_col = f"home_{target}"
        self.away_col = f"away_{target}"
        # fitted state
        self.mu_ = 0.0
        self.home_ = 0.0
        self.for_: dict[str, float] = {}
        self.against_: dict[str, float] = {}
        self.dispersion_: float = 1.0   # residual var/mean
        self.alpha_: float = 0.0        # NegBin dispersion (Var = mu + alpha*mu^2)
        self.family_: str = "poisson"   # resolved family

    def fit(self, matches: pd.DataFrame, asof: date) -> "CountModel":
        train = matches[pd.to_datetime(matches["date"]) < pd.Timestamp(asof)].copy()
        train = train.dropna(subset=[self.home_col, self.away_col])
        if train.empty:
            raise ValueError(f"no training rows with {self.home_col}/{self.away_col}")

        teams = sorted(set(train["home"]) | set(train["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        t = len(teams)

        # Stack two count observations per match (home actor, then away actor).
        actor = np.r_[train["home"].map(idx).to_numpy(), train["away"].map(idx).to_numpy()]
        opp = np.r_[train["away"].map(idx).to_numpy(), train["home"].map(idx).to_numpy()]
        is_home = np.r_[np.ones(len(train)), np.zeros(len(train))]
        k = np.r_[train[self.home_col].to_numpy(float), train[self.away_col].to_numpy(float)]

        p0 = np.zeros(2 * t + 2)
        p0[0] = math.log(max(k.mean(), 0.1))
        res = minimize(
            self._nll_grad, p0, args=(actor, opp, is_home, k, t), jac=True,
            method="L-BFGS-B", options={"maxiter": 200},
        )
        p = res.x
        self.mu_, self.home_ = float(p[0]), float(p[1])
        self.for_ = {tm: float(p[2 + i]) for tm, i in idx.items()}
        self.against_ = {tm: float(p[2 + t + i]) for tm, i in idx.items()}

        mu = np.exp(p[0] + p[1] * is_home + p[2 : 2 + t][actor] + p[2 + t :][opp])
        self.dispersion_ = float(np.mean((k - mu) ** 2 / np.maximum(mu, 1e-9)))
        # Method-of-moments NegBin dispersion: Var = mu + alpha*mu^2.
        self.alpha_ = float(max(np.mean(((k - mu) ** 2 - mu) / np.maximum(mu**2, 1e-9)), 0.0))
        if self.family == "auto":
            self.family_ = "nbinom" if self.dispersion_ > NB_DISPERSION_TRIGGER else "poisson"
        else:
            self.family_ = self.family
        return self

    def _nll_grad(self, p, actor, opp, is_home, k, t):
        log_mu = p[0] + p[1] * is_home + p[2 : 2 + t][actor] + p[2 + t :][opp]
        mu = np.exp(log_mu)
        nll = float(np.sum(mu - k * log_mu)) + 0.5 * self.reg * (
            p[2 : 2 + t] @ p[2 : 2 + t] + p[2 + t :] @ p[2 + t :]
        )
        r = mu - k
        g = np.zeros_like(p)
        g[0] = r.sum()
        g[1] = (r * is_home).sum()
        g[2 : 2 + t] = np.bincount(actor, r, minlength=t) + self.reg * p[2 : 2 + t]
        g[2 + t :] = np.bincount(opp, r, minlength=t) + self.reg * p[2 + t :]
        return nll, g

    # -- prediction --------------------------------------------------------
    def _expected(self, actor: str, opp: str, is_home: float) -> float:
        return math.exp(
            self.mu_ + self.home_ * is_home
            + self.for_.get(actor, 0.0) + self.against_.get(opp, 0.0)
        )

    def expected_counts(self, home: str, away: str) -> tuple[float, float]:
        """Expected (home count, away count) for the target."""
        return self._expected(home, away, 1.0), self._expected(away, home, 0.0)

    def predictive(self, mu: float):
        """A frozen scipy distribution for a single team-count with mean `mu`."""
        if self.family_ == "nbinom" and self.alpha_ > 0:
            n = 1.0 / self.alpha_
            return nbinom(n, n / (n + mu))
        return poisson(mu)

    def interval(self, mu: float, level: float = 0.8) -> tuple[int, int]:
        """Central predictive interval at `level` (for coverage calibration)."""
        d = self.predictive(mu)
        lo = (1.0 - level) / 2.0
        return int(d.ppf(lo)), int(d.ppf(1.0 - lo))
