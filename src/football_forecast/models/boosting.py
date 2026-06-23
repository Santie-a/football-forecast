"""Gradient-boosting models — the ML challenger (roadmap Part 3.5 / Phase 5).

A different philosophy from the structured goal models: engineer pre-match
features (features/engineering.py) and let LightGBM learn the mapping. 1X2 uses
the **multiclass** objective; counts use the **Poisson** objective (Tweedie is a
drop-in alternative). The honest expectation (roadmap 3.5) is that a well-tuned
Dixon–Coles is hard to beat on 1X2 from match data alone; boosting tends to shine
where many features help (the count targets).

PC-only (training is CPU-heavy). `BoostingModel` implements the OutcomeModel
protocol so it drops into the same backtester.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from football_forecast.data.schema import OUTCOMES, Fixture
from football_forecast.features.engineering import FEATURES, build_training

_OUT_CODE = {o: i for i, o in enumerate(OUTCOMES)}  # H->0, D->1, A->2


class BoostingModel:
    """LightGBM multiclass 1X2 on engineered pre-match features."""

    def __init__(
        self,
        window: int = 10,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 50,
        seed: int = 0,
    ) -> None:
        self.window = window
        self.params = dict(
            objective="multiclass", num_class=len(OUTCOMES), n_estimators=n_estimators,
            learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, random_state=seed, verbose=-1,
        )
        self.clf_: LGBMClassifier | None = None
        self.state_ = None

    def fit(self, matches: pd.DataFrame, asof: date) -> "BoostingModel":
        train = matches[pd.to_datetime(matches["date"]) < pd.Timestamp(asof)]
        x, y, state = build_training(train, window=self.window)
        self.state_ = state
        codes = np.array([_OUT_CODE[o] for o in y])
        self.clf_ = LGBMClassifier(**self.params).fit(x[FEATURES], codes)
        return self

    def predict_1x2(self, fixture: Fixture) -> dict[str, float]:
        feat = self.state_.snapshot(fixture.home, fixture.away, fixture.date, fixture.neutral)
        x = pd.DataFrame([feat], columns=FEATURES)
        proba = self.clf_.predict_proba(x)[0]  # columns ordered by class code = OUTCOMES
        return {o: float(p) for o, p in zip(OUTCOMES, proba)}


class BoostingCountModel:
    """LightGBM Poisson/Tweedie regressor for a count target (fouls/corners/cards).

    Predicts the home-side and away-side counts from the same engineered features
    plus an `is_home` flag (two stacked observations per match, like CountModel)."""

    def __init__(
        self,
        target: str,
        objective: str = "poisson",
        window: int = 10,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 50,
        seed: int = 0,
    ) -> None:
        self.target = target
        self.home_col, self.away_col = f"home_{target}", f"away_{target}"
        self.window = window
        self.params = dict(
            objective=objective, n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=min_child_samples,
            random_state=seed, verbose=-1,
        )
        self.reg_: LGBMRegressor | None = None
        self.state_ = None

    def fit(self, matches: pd.DataFrame, asof: date) -> "BoostingCountModel":
        train = matches[pd.to_datetime(matches["date"]) < pd.Timestamp(asof)]
        train = train.dropna(subset=[self.home_col, self.away_col])
        x, _, state = build_training(train, window=self.window)
        self.state_ = state
        # Two rows per match: home-actor (is_home=1) and away-actor (is_home=0).
        home_x = x.assign(is_home=1.0)
        away_x = x.assign(is_home=0.0)
        # Away-actor view mirrors home/away feature pairs.
        swap = {
            "home_ppg": "away_ppg", "away_ppg": "home_ppg",
            "home_gf": "away_gf", "away_gf": "home_gf",
            "home_ga": "away_ga", "away_ga": "home_ga",
            "rest_home": "rest_away", "rest_away": "rest_home",
        }
        away_x = away_x.rename(columns=swap)[home_x.columns]
        for col in ("elo_diff", "ppg_diff", "gf_diff", "ga_diff"):
            away_x[col] = -away_x[col]
        big_x = pd.concat([home_x, away_x], ignore_index=True)
        big_y = np.r_[train[self.home_col].to_numpy(float), train[self.away_col].to_numpy(float)]
        self.reg_ = LGBMRegressor(**self.params).fit(big_x, big_y)
        return self

    def expected_counts(self, fixture: Fixture) -> tuple[float, float]:
        feat = self.state_.snapshot(fixture.home, fixture.away, fixture.date, fixture.neutral)
        cols = FEATURES + ["is_home"]
        home_row = pd.DataFrame([{**feat, "is_home": 1.0}])[cols]
        mh = float(self.reg_.predict(home_row)[0])
        swap = {"home_ppg": "away_ppg", "away_ppg": "home_ppg", "home_gf": "away_gf",
                "away_gf": "home_gf", "home_ga": "away_ga", "away_ga": "home_ga",
                "rest_home": "rest_away", "rest_away": "rest_home"}
        afeat = {swap.get(k, k): v for k, v in feat.items()}
        for c in ("elo_diff", "ppg_diff", "gf_diff", "ga_diff"):
            afeat[c] = -afeat[c]
        away_row = pd.DataFrame([{**afeat, "is_home": 0.0}])[cols]
        ma = float(self.reg_.predict(away_row)[0])
        return mh, ma
