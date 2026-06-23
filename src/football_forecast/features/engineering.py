"""Leakage-free pre-match features for the ML models (roadmap Part 5.4).

A single chronological pass maintains per-team state (Elo, rolling form, last
match date). For each match we snapshot the state **before** the match (so every
feature uses only the past), then update with the result. The same state, after
processing the training matches, produces features for a future fixture — exactly
how a model fit at the origin would predict the next period (consistent with how
Elo/Dixon–Coles behave in the backtester).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from football_forecast.data.schema import match_outcome

# Feature columns the models consume (order fixed).
FEATURES = [
    "elo_diff", "neutral",
    "home_ppg", "away_ppg", "ppg_diff",
    "home_gf", "home_ga", "away_gf", "away_ga", "gf_diff", "ga_diff",
    "rest_home", "rest_away",
]


@dataclass
class _Team:
    elo: float
    form: deque = field(default_factory=deque)  # (gf, ga, pts) most-recent-last
    last_date: pd.Timestamp | None = None


class FeatureState:
    def __init__(self, window: int = 10, elo_k: float = 20.0, home_adv: float = 60.0, base: float = 1500.0):
        self.window = window
        self.elo_k = elo_k
        self.home_adv = home_adv
        self.base = base
        self.teams: dict[str, _Team] = {}

    def _t(self, name: str) -> _Team:
        if name not in self.teams:
            self.teams[name] = _Team(elo=self.base, form=deque(maxlen=self.window))
        return self.teams[name]

    @staticmethod
    def _avg(form: deque, i: int) -> float:
        return float(np.mean([f[i] for f in form])) if form else np.nan

    def snapshot(self, home: str, away: str, date, neutral: bool) -> dict:
        h, a = self._t(home), self._t(away)
        date = pd.Timestamp(date)
        ha = 0.0 if neutral else self.home_adv
        home_ppg, away_ppg = self._avg(h.form, 2), self._avg(a.form, 2)
        home_gf, home_ga = self._avg(h.form, 0), self._avg(h.form, 1)
        away_gf, away_ga = self._avg(a.form, 0), self._avg(a.form, 1)
        rest_home = (date - h.last_date).days if h.last_date is not None else np.nan
        rest_away = (date - a.last_date).days if a.last_date is not None else np.nan
        return {
            "elo_diff": h.elo + ha - a.elo,
            "neutral": float(neutral),
            "home_ppg": home_ppg, "away_ppg": away_ppg,
            "ppg_diff": (home_ppg - away_ppg) if not (np.isnan(home_ppg) or np.isnan(away_ppg)) else np.nan,
            "home_gf": home_gf, "home_ga": home_ga, "away_gf": away_gf, "away_ga": away_ga,
            "gf_diff": (home_gf - away_gf) if not (np.isnan(home_gf) or np.isnan(away_gf)) else np.nan,
            "ga_diff": (home_ga - away_ga) if not (np.isnan(home_ga) or np.isnan(away_ga)) else np.nan,
            "rest_home": min(rest_home, 30) if not np.isnan(rest_home) else np.nan,
            "rest_away": min(rest_away, 30) if not np.isnan(rest_away) else np.nan,
        }

    def update(self, home: str, away: str, hg: int, ag: int, date, neutral: bool) -> None:
        h, a = self._t(home), self._t(away)
        date = pd.Timestamp(date)
        # Elo update.
        exp_h = 1.0 / (1.0 + 10.0 ** (-((h.elo + (0.0 if neutral else self.home_adv)) - a.elo) / 400.0))
        s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = self.elo_k * (s - exp_h)
        h.elo += delta
        a.elo -= delta
        # Rolling form (points + goals).
        hp, ap = (3, 0) if hg > ag else ((1, 1) if hg == ag else (0, 3))
        h.form.append((hg, ag, hp))
        a.form.append((ag, hg, ap))
        h.last_date = a.last_date = date


def build_training(matches: pd.DataFrame, **kw) -> tuple[pd.DataFrame, list[str], FeatureState]:
    """Replay matches chronologically → (feature matrix, outcomes, final state)."""
    st = FeatureState(**kw)
    df = matches.sort_values("date", kind="stable")
    rows, ys = [], []
    for r in df.itertuples(index=False):
        rows.append(st.snapshot(r.home, r.away, r.date, bool(r.neutral)))
        ys.append(match_outcome(r.home_goals, r.away_goals))
        st.update(r.home, r.away, int(r.home_goals), int(r.away_goals), r.date, bool(r.neutral))
    return pd.DataFrame(rows, columns=FEATURES), ys, st
