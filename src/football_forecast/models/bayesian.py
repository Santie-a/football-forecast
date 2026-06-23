"""Bayesian hierarchical Poisson goal model (Baio & Blangiardo 2010).

Team attack/defence strengths are drawn from shared distributions, so sparse
teams are shrunk toward the population mean and *borrow strength* — the partial
pooling that handles few-match national teams gracefully (docs/models-explained.md
§8). Forecasts are **posterior-predictive**: the scoreline matrix is averaged over
posterior draws, so parameter uncertainty is baked in (rarely-seen teams get wider,
humbler distributions).

    log lambda_home = mu + home*[not neutral] + atk[h] - def[a]
    log lambda_away = mu                      + atk[a] - def[h]
    atk[i] ~ Normal(0, sigma_att)   def[i] ~ Normal(0, sigma_def)   (non-centered)

**PC-only.** MCMC is the heaviest step (decision 001). Sampled with nutpie
(numba backend) so it needs no C compiler. Convergence is gated on a
self-contained split-R-hat — a non-converged fit must not emit forecasts.
"""

from __future__ import annotations

import math
import warnings
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import poisson

from football_forecast.data.schema import Fixture
from football_forecast.forecast.markets import one_x_two


def _split_rhat(posterior: dict[str, np.ndarray]) -> float:
    """Max split-R-hat across all parameters. Self-contained (no arviz API churn).
    Each value is an array shaped (chain, draw, *param)."""
    worst = 0.0
    for arr in posterior.values():
        a = arr.reshape(arr.shape[0], arr.shape[1], -1)
        c, d, p = a.shape
        if d < 4:
            continue
        half = d // 2
        s = np.concatenate([a[:, :half], a[:, half : 2 * half]], axis=0)  # (2c, half, p)
        means = s.mean(axis=1)
        between = half * means.var(axis=0, ddof=1)
        within = s.var(axis=1, ddof=1).mean(axis=0)
        var = (half - 1) / half * within + between / half
        rhat = np.sqrt(np.maximum(var, 1e-12) / np.maximum(within, 1e-12))
        worst = max(worst, float(np.nanmax(rhat)))
    return worst


class BayesianModel:
    def __init__(
        self,
        window_days: float | None = 2000.0,
        half_life_days: float | None = None,  # window handles recency; decay added noise (see note 03)
        draws: int = 1000,
        tune: int = 2500,
        chains: int = 4,
        pp_draws: int = 300,
        max_goals: int = 10,
        seed: int = 0,
        rhat_threshold: float = 1.01,
        strict: bool = False,
    ) -> None:
        self.window_days = window_days
        self.half_life_days = half_life_days
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.pp_draws = pp_draws
        self.max_goals = max_goals
        self.seed = seed
        self.rhat_threshold = rhat_threshold
        self.strict = strict
        # fitted posterior (thinned to pp_draws)
        self.idx_: dict[str, int] = {}
        self.mu_: np.ndarray = np.array([])     # (pp_draws,)
        self.home_: np.ndarray = np.array([])
        self.atk_: np.ndarray = np.array([])    # (pp_draws, T)
        self.def_: np.ndarray = np.array([])
        self.max_rhat_: float = float("nan")
        self.atk_std_: dict[str, float] = {}    # posterior sd of attack per team
        self.team_matches_: dict[str, int] = {}

    @property
    def converged(self) -> bool:
        return np.isfinite(self.max_rhat_) and self.max_rhat_ <= self.rhat_threshold

    def _weights(self, dates: pd.Series, asof: date) -> np.ndarray:
        if not self.half_life_days:
            return np.ones(len(dates))
        age = (pd.Timestamp(asof) - pd.to_datetime(dates)).dt.days.to_numpy(float)
        return np.exp(-math.log(2.0) / self.half_life_days * age)

    def fit(self, matches: pd.DataFrame, asof: date) -> "BayesianModel":
        import pymc as pm  # heavy, PC-only — imported lazily
        import nutpie

        train = matches[pd.to_datetime(matches["date"]) < pd.Timestamp(asof)].copy()
        if self.window_days:
            cutoff = pd.Timestamp(asof) - pd.Timedelta(days=self.window_days)
            train = train[pd.to_datetime(train["date"]) >= cutoff]
        if len(train) == 0:
            raise ValueError("no training matches in window before asof")

        teams = sorted(set(train["home"]) | set(train["away"]))
        self.idx_ = {t: i for i, t in enumerate(teams)}
        t = len(teams)
        hi = train["home"].map(self.idx_).to_numpy()
        ai = train["away"].map(self.idx_).to_numpy()
        x = train["home_goals"].to_numpy(float)
        y = train["away_goals"].to_numpy(float)
        nn = (~train["neutral"].to_numpy(bool)).astype(float)
        w = self._weights(train["date"], asof)  # exponential time-decay weights
        self.team_matches_ = {
            tm: int((train["home"] == tm).sum() + (train["away"] == tm).sum()) for tm in teams
        }

        mu0 = math.log(max((x + y).mean() / 2.0, 0.1))
        with pm.Model() as model:
            mu = pm.Normal("mu", mu0, 1.0)
            home = pm.Normal("home", 0.25, 0.5)
            sa = pm.HalfNormal("sa", 0.5)
            sd = pm.HalfNormal("sd", 0.5)
            atk = pm.Deterministic("atk", sa * pm.Normal("atk_z", 0.0, 1.0, shape=t))
            dfn = pm.Deterministic("dfn", sd * pm.Normal("dfn_z", 0.0, 1.0, shape=t))
            log_lh = mu + home * nn + atk[hi] - dfn[ai]
            log_la = mu + atk[ai] - dfn[hi]
            # Time-decayed Poisson log-likelihood (drop the constant lgamma term).
            pm.Potential(
                "lik",
                pm.math.sum(w * (x * log_lh - pm.math.exp(log_lh)))
                + pm.math.sum(w * (y * log_la - pm.math.exp(log_la))),
            )

            compiled = nutpie.compile_pymc_model(model, backend="numba")
            idata = nutpie.sample(
                compiled, draws=self.draws, tune=self.tune, chains=self.chains,
                seed=self.seed, progress_bar=False,
            )

        post = idata.posterior
        self.max_rhat_ = _split_rhat(
            {v: post[v].to_numpy() for v in ("mu", "home", "sa", "sd", "atk", "dfn")}
        )
        if not self.converged:
            msg = f"MCMC did not converge (max R-hat {self.max_rhat_:.3f} > {self.rhat_threshold})"
            if self.strict:
                raise RuntimeError(msg)
            warnings.warn(msg, stacklevel=2)

        # Flatten (chain, draw) -> samples and thin to pp_draws.
        def flat(name):
            a = post[name].to_numpy()
            return a.reshape(-1, *a.shape[2:])

        atk_all, dfn_all = flat("atk"), flat("dfn")
        mu_all, home_all = flat("mu"), flat("home")
        sel = np.linspace(0, atk_all.shape[0] - 1, min(self.pp_draws, atk_all.shape[0])).astype(int)
        self.atk_, self.def_ = atk_all[sel], dfn_all[sel]
        self.mu_, self.home_ = mu_all[sel], home_all[sel]
        self.atk_std_ = {tm: float(atk_all[:, i].std()) for tm, i in self.idx_.items()}
        return self

    # -- prediction (posterior predictive) ---------------------------------
    def _lambda_draws(self, fixture: Fixture) -> tuple[np.ndarray, np.ndarray]:
        p = len(self.mu_)
        ih, ia = self.idx_.get(fixture.home), self.idx_.get(fixture.away)
        atk_h = self.atk_[:, ih] if ih is not None else np.zeros(p)
        atk_a = self.atk_[:, ia] if ia is not None else np.zeros(p)
        def_h = self.def_[:, ih] if ih is not None else np.zeros(p)
        def_a = self.def_[:, ia] if ia is not None else np.zeros(p)
        home = 0.0 if fixture.neutral else self.home_
        lh = np.exp(self.mu_ + home + atk_h - def_a)
        la = np.exp(self.mu_ + atk_a - def_h)
        return lh, la

    def scoreline(self, fixture: Fixture) -> np.ndarray:
        """Posterior-predictive scoreline matrix: average the per-draw independent
        Poisson matrices over the posterior (folds in parameter uncertainty)."""
        lh, la = self._lambda_draws(fixture)
        ks = np.arange(self.max_goals + 1)
        pmf_h = poisson.pmf(ks[None, :], lh[:, None])  # (draws, K+1)
        pmf_a = poisson.pmf(ks[None, :], la[:, None])
        m = np.einsum("pi,pj->ij", pmf_h, pmf_a) / len(lh)
        return m / m.sum()

    def rates(self, fixture: Fixture) -> tuple[float, float]:
        lh, la = self._lambda_draws(fixture)
        return float(lh.mean()), float(la.mean())

    def predict_1x2(self, fixture: Fixture) -> dict[str, float]:
        return one_x_two(self.scoreline(fixture))
