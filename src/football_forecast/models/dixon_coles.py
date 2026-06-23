"""Poisson goal models: Maher (independent) and Dixon–Coles.

Each team has an attack and a defence strength; a global home-advantage term
applies only on non-neutral venues (M03). For a match (home h, away a):

    log lambda_home = mu + home*[not neutral] + atk[h] - def[a]
    log lambda_away = mu                      + atk[a] - def[h]

**Fitting.** Penalized maximum likelihood. The negative log-likelihood and its
gradient are computed in O(N) (scatter-adds over team indices), so L-BFGS-B
handles the 2T+2 parameters fast. An L2 penalty on attack/defence both removes
the additive degeneracy and shrinks sparse teams toward average (the
regularization the roadmap calls for). Matches are weighted by an exponential
time-decay (half-life is a hyperparameter, tuned by backtest — M02).

**Dixon–Coles.** A second stage fits the single low-score parameter rho by
profile likelihood, holding the Poisson means fixed (a standard, fast two-stage
approximation; logged in docs/decisions.md). Maher is this model with the
correction and time-decay switched off.

Forecasts come from the keystone: rates -> scoreline matrix -> 1X2 (markets.py).
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from football_forecast.data.schema import Fixture
from football_forecast.forecast.markets import one_x_two
from football_forecast.forecast.scoreline import scoreline_matrix


class DixonColesModel:
    def __init__(
        self,
        use_dc: bool = True,
        half_life_days: float | None = 1460.0,  # ~4y; selected by backtest (M02)
        reg: float = 0.05,
        max_goals: int = 10,
    ) -> None:
        self.use_dc = use_dc
        self.half_life_days = half_life_days
        self.reg = reg
        self.max_goals = max_goals
        # fitted state
        self.mu_: float = 0.0
        self.home_: float = 0.0
        self.atk_: dict[str, float] = {}
        self.def_: dict[str, float] = {}
        self.rho_: float = 0.0

    # -- fitting -----------------------------------------------------------
    def fit(self, matches: pd.DataFrame, asof: date) -> "DixonColesModel":
        train = matches[pd.to_datetime(matches["date"]) < pd.Timestamp(asof)].copy()
        if len(train) == 0:
            raise ValueError("no training matches before asof")

        teams = sorted(set(train["home"]) | set(train["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        t = len(teams)

        hi = train["home"].map(idx).to_numpy()
        ai = train["away"].map(idx).to_numpy()
        x = train["home_goals"].to_numpy(float)
        y = train["away_goals"].to_numpy(float)
        nn = (~train["neutral"].to_numpy(bool)).astype(float)
        w = self._weights(train["date"], asof)

        p0 = np.zeros(2 * t + 2)
        p0[0] = math.log(max((x + y).mean() / 2.0, 0.1))  # mu ~ log mean goals
        p0[1] = 0.1  # home advantage

        res = minimize(
            self._nll_grad,
            p0,
            args=(hi, ai, x, y, nn, w, t),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        p = res.x
        self.mu_ = float(p[0])
        self.home_ = float(p[1])
        self.atk_ = {team: float(p[2 + i]) for team, i in idx.items()}
        self.def_ = {team: float(p[2 + t + i]) for team, i in idx.items()}

        self.rho_ = self._fit_rho(hi, ai, x, y, nn, w) if self.use_dc else 0.0
        return self

    def _weights(self, dates: pd.Series, asof: date) -> np.ndarray:
        if not self.half_life_days:
            return np.ones(len(dates))
        age = (pd.Timestamp(asof) - pd.to_datetime(dates)).dt.days.to_numpy(float)
        xi = math.log(2.0) / self.half_life_days
        return np.exp(-xi * age)

    def _nll_grad(self, p, hi, ai, x, y, nn, w, t):
        mu, home = p[0], p[1]
        atk, dfn = p[2 : 2 + t], p[2 + t : 2 + 2 * t]

        log_lh = mu + home * nn + atk[hi] - dfn[ai]
        log_la = mu + atk[ai] - dfn[hi]
        lh, la = np.exp(log_lh), np.exp(log_la)

        nll = float(
            np.sum(w * (lh - x * log_lh)) + np.sum(w * (la - y * log_la))
        ) + 0.5 * self.reg * (atk @ atk + dfn @ dfn)

        rh = w * (lh - x)
        ra = w * (la - y)
        g = np.zeros_like(p)
        g[0] = rh.sum() + ra.sum()
        g[1] = (rh * nn).sum()
        g[2 : 2 + t] = (
            np.bincount(hi, rh, minlength=t)
            + np.bincount(ai, ra, minlength=t)
            + self.reg * atk
        )
        g[2 + t : 2 + 2 * t] = (
            -np.bincount(ai, rh, minlength=t)
            - np.bincount(hi, ra, minlength=t)
            + self.reg * dfn
        )
        return nll, g

    def _fit_rho(self, hi, ai, x, y, nn, w) -> float:
        # atk_/def_ were inserted in sorted-team order, matching the hi/ai indices.
        a = np.array(list(self.atk_.values()))
        d = np.array(list(self.def_.values()))
        lh = np.exp(self.mu_ + self.home_ * nn + a[hi] - d[ai])
        la = np.exp(self.mu_ + a[ai] - d[hi])

        low = (x <= 1) & (y <= 1)
        xl, yl, lhl, lal, wl = x[low], y[low], lh[low], la[low], w[low]

        def neg_ll(rho: float) -> float:
            tau = np.ones_like(xl)
            m00 = (xl == 0) & (yl == 0)
            m01 = (xl == 0) & (yl == 1)
            m10 = (xl == 1) & (yl == 0)
            m11 = (xl == 1) & (yl == 1)
            tau[m00] = 1.0 - lhl[m00] * lal[m00] * rho
            tau[m01] = 1.0 + lhl[m01] * rho
            tau[m10] = 1.0 + lal[m10] * rho
            tau[m11] = 1.0 - rho
            if np.any(tau <= 0):
                return 1e12
            return float(-np.sum(wl * np.log(tau)))

        res = minimize_scalar(neg_ll, bounds=(-0.25, 0.25), method="bounded")
        return float(res.x)

    # -- prediction --------------------------------------------------------
    def rates(self, fixture: Fixture) -> tuple[float, float]:
        ah = self.atk_.get(fixture.home, 0.0)
        aa = self.atk_.get(fixture.away, 0.0)
        dh = self.def_.get(fixture.home, 0.0)
        da = self.def_.get(fixture.away, 0.0)
        home = 0.0 if fixture.neutral else self.home_
        lh = math.exp(self.mu_ + home + ah - da)
        la = math.exp(self.mu_ + aa - dh)
        return lh, la

    def predict_1x2(self, fixture: Fixture) -> dict[str, float]:
        lh, la = self.rates(fixture)
        rho = self.rho_ if self.use_dc else None
        return one_x_two(scoreline_matrix(lh, la, self.max_goals, rho))


class MaherModel(DixonColesModel):
    """Independent Poisson, no low-score correction, no time-decay — the Phase 2
    baseline that Dixon–Coles must beat."""

    def __init__(self, reg: float = 0.05, max_goals: int = 10) -> None:
        super().__init__(use_dc=False, half_life_days=None, reg=reg, max_goals=max_goals)
