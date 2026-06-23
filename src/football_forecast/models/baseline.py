"""Base-rate baseline: predict the training-set H/D/A frequencies, ignoring who
is playing. This is the floor every real model must beat on RPS (Phase 1
acceptance criterion). A model that can't beat it has learned nothing."""

from __future__ import annotations

from datetime import date

import pandas as pd

from football_forecast.data.schema import OUTCOMES, Fixture, add_outcome


class BaseRateModel:
    """Constant forecast equal to the empirical outcome frequencies in training."""

    def __init__(self) -> None:
        self.rates_: dict[str, float] = {o: 1 / 3 for o in OUTCOMES}

    def fit(self, matches: pd.DataFrame, asof: date) -> "BaseRateModel":
        train = matches[pd.to_datetime(matches["date"]) < pd.Timestamp(asof)]
        if len(train) == 0:
            self.rates_ = {o: 1 / 3 for o in OUTCOMES}
            return self
        freqs = add_outcome(train)["outcome"].value_counts(normalize=True)
        self.rates_ = {o: float(freqs.get(o, 0.0)) for o in OUTCOMES}
        return self

    def predict_1x2(self, fixture: Fixture) -> dict[str, float]:
        return dict(self.rates_)
