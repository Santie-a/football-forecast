"""Elo ratings for national teams, with a draw-aware 1X2 head.

Two pieces (docs/models-explained.md §2):

1. **Ratings.** A single strength per team, updated after each match by the
   surprise `(S - E)`, scaled by a K-factor that accounts for margin of victory
   and competition importance (the M01 comp_type feature). Home advantage is a
   fixed rating bump, zeroed on neutral venues (decision M03).

2. **1X2 head.** Standard Elo gives only a win/not split. We add draws with a
   Davidson tie model: with team strengths s = 10^(R/400),

       P(H) ∝ s_home,   P(A) ∝ s_away,   P(D) ∝ nu · sqrt(s_home · s_away)

   The home/away odds ratio stays exactly 10^(diff/400) (standard Elo), and the
   draw share peaks when teams are evenly matched — the observed behaviour. The
   single draw parameter `nu` is fit by minimizing log loss on the pre-match
   rating gaps seen during training (no leakage). A refinement to a full ordered
   logit on comp_type is logged for later.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from football_forecast.data.schema import OUTCOMES, Fixture, match_outcome

# K-factor multipliers by competition importance (the M01 comp_type feature).
DEFAULT_IMPORTANCE: dict[str, float] = {
    "friendly": 0.7,
    "qualifier": 1.0,
    "league": 1.0,
    "continental": 1.1,
    "final_tournament": 1.2,
}


def _mov_multiplier(goal_diff: int) -> float:
    """World-Football-Elo margin-of-victory scaling: a rout moves ratings more."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def _expected_home(diff: float) -> float:
    """Standard Elo expected score in [0, 1] from a rating difference."""
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


class EloModel:
    def __init__(
        self,
        k: float = 32.0,
        home_advantage: float = 65.0,
        base_rating: float = 1500.0,
        use_mov: bool = True,
        importance: dict[str, float] | None = None,
    ) -> None:
        self.k = k
        self.home_advantage = home_advantage
        self.base_rating = base_rating
        self.use_mov = use_mov
        self.importance = importance or dict(DEFAULT_IMPORTANCE)
        self.ratings_: dict[str, float] = {}
        self.nu_: float = 1.0  # draw propensity, fit in `fit`

    # -- ratings -----------------------------------------------------------
    def _rating(self, team: str) -> float:
        return self.ratings_.get(team, self.base_rating)

    def _diff(self, home: str, away: str, neutral: bool) -> float:
        h = 0.0 if neutral else self.home_advantage
        return self._rating(home) + h - self._rating(away)

    def fit(self, matches: pd.DataFrame, asof: date) -> "EloModel":
        train = matches[pd.to_datetime(matches["date"]) < pd.Timestamp(asof)]
        train = train.sort_values("date", kind="stable")

        self.ratings_ = {}
        gaps: list[float] = []  # pre-match rating gaps
        codes: list[int] = []   # 0=H, 1=D, 2=A (index into OUTCOMES)

        for row in train.itertuples(index=False):
            diff = self._diff(row.home, row.away, bool(row.neutral))
            gaps.append(diff)
            codes.append(OUTCOMES.index(match_outcome(row.home_goals, row.away_goals)))

            exp_home = _expected_home(diff)
            s = 1.0 if row.home_goals > row.away_goals else (
                0.5 if row.home_goals == row.away_goals else 0.0
            )
            k_eff = self.k * self.importance.get(row.comp_type, 1.0)
            if self.use_mov:
                k_eff *= _mov_multiplier(int(row.home_goals) - int(row.away_goals))
            delta = k_eff * (s - exp_home)
            self.ratings_[row.home] = self._rating(row.home) + delta
            self.ratings_[row.away] = self._rating(row.away) - delta

        self.nu_ = self._fit_nu(np.asarray(gaps), np.asarray(codes))
        return self

    # -- 1X2 head ----------------------------------------------------------
    @staticmethod
    def _probs_from_gap(diff: float | np.ndarray, nu: float):
        """Davidson tie model in log space (stable for large ratings)."""
        a = (np.asarray(diff, float) / 400.0) * np.log(10.0)  # log s_home (rel.)
        log_h = a / 2.0
        log_a = -a / 2.0
        log_d = np.log(max(nu, 1e-12)) + 0.0  # sqrt(s_h*s_a) is constant -> 0 in rel.
        stack = np.stack([log_h, log_d * np.ones_like(log_h), log_a], axis=-1)
        stack -= stack.max(axis=-1, keepdims=True)
        e = np.exp(stack)
        return e / e.sum(axis=-1, keepdims=True)

    def _fit_nu(self, gaps: np.ndarray, codes: np.ndarray) -> float:
        if len(gaps) == 0:
            return 1.0

        def neg_log_loss(log_nu: float) -> float:
            p = self._probs_from_gap(gaps, float(np.exp(log_nu)))
            chosen = p[np.arange(len(codes)), codes]
            return float(-np.mean(np.log(np.clip(chosen, 1e-15, 1.0))))

        res = minimize_scalar(neg_log_loss, bounds=(-6.0, 6.0), method="bounded")
        return float(np.exp(res.x))

    def predict_1x2(self, fixture: Fixture) -> dict[str, float]:
        diff = self._diff(fixture.home, fixture.away, fixture.neutral)
        p = self._probs_from_gap(np.array([diff]), self.nu_)[0]
        return {o: float(pi) for o, pi in zip(OUTCOMES, p)}
