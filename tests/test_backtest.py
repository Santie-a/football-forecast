"""Tests for the walk-forward harness — most importantly, that it never leaks
future matches into a fit."""

from datetime import date

import pandas as pd
import pytest

from football_forecast.data.schema import OUTCOMES, Fixture, validate
from football_forecast.eval.backtest import walk_forward, yearly_origins
from football_forecast.models.baseline import BaseRateModel


def _matches(n: int = 600) -> pd.DataFrame:
    rng = pd.date_range("2000-01-01", periods=n, freq="3D")
    rows = []
    for i, d in enumerate(rng):
        rows.append(
            {
                "match_id": f"m{i}",
                "date": d,
                "competition": "Friendly",
                "comp_type": "friendly",
                "home": f"T{i % 8}",
                "away": f"T{(i + 3) % 8}",
                "home_goals": i % 3,
                "away_goals": (i + 1) % 2,
                "neutral": False,
            }
        )
    return validate(pd.DataFrame(rows))


class _SpyModel:
    """Records the latest match date it sees during fit, to catch leakage."""

    def __init__(self) -> None:
        self.max_seen: pd.Timestamp | None = None

    def fit(self, matches: pd.DataFrame, asof: date) -> "_SpyModel":
        train = matches[pd.to_datetime(matches["date"]) < pd.Timestamp(asof)]
        self.max_seen = pd.to_datetime(train["date"]).max() if len(train) else None
        self._asof = asof
        return self

    def predict_1x2(self, fixture: Fixture) -> dict:
        # Assert at predict time too: nothing at/after asof should have been seen.
        assert self.max_seen is None or self.max_seen < pd.Timestamp(self._asof)
        return {o: 1 / 3 for o in OUTCOMES}


def test_walk_forward_runs_and_pools():
    m = _matches()
    res = walk_forward(
        BaseRateModel, m, yearly_origins(m, start=2002), min_train_matches=50
    )
    assert res.n_test > 0
    assert set(res.overall) == {"rps", "log_loss", "brier"}
    assert not res.per_origin.empty


def test_walk_forward_no_leakage():
    m = _matches()
    # If any fold leaked a future match into fit, the SpyModel assertions fire.
    res = walk_forward(
        _SpyModel, m, yearly_origins(m, start=2002), min_train_matches=50
    )
    assert res.n_test > 0


def test_walk_forward_raises_without_folds():
    m = _matches(n=50)
    with pytest.raises(ValueError, match="no usable folds"):
        walk_forward(BaseRateModel, m, [pd.Timestamp("1990-01-01")])


def test_base_rate_recovers_training_frequencies():
    m = _matches()
    model = BaseRateModel().fit(m, asof=date(2010, 1, 1))
    assert sum(model.rates_.values()) == pytest.approx(1.0)
