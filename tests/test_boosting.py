"""Tests for the feature builder and the gradient-boosting models."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from football_forecast.data.schema import OUTCOMES, Fixture, validate
from football_forecast.features.engineering import FEATURES, FeatureState, build_training
from football_forecast.models.boosting import BoostingCountModel, BoostingModel


def _league(seed=0, n_seasons=8):
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D", "E", "F"]
    stren = {t: v for t, v in zip(teams, np.linspace(0.6, -0.6, len(teams)))}
    rows, d = [], pd.Timestamp("2016-08-01")
    for _ in range(n_seasons):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lh = np.exp(0.2 + 0.25 + stren[h] - stren[a])
                la = np.exp(0.2 + stren[a] - stren[h])
                rows.append({
                    "match_id": f"m{len(rows)}", "date": d, "competition": "L",
                    "comp_type": "league", "home": h, "away": a,
                    "home_goals": int(rng.poisson(lh)), "away_goals": int(rng.poisson(la)),
                    "neutral": False,
                    "home_corners": int(rng.poisson(6)), "away_corners": int(rng.poisson(5)),
                })
                d += pd.Timedelta(days=3)
    return validate(pd.DataFrame(rows))


# --- features ------------------------------------------------------------

def test_build_training_shapes_and_no_leakage():
    df = _league()
    x, y, state = build_training(df)
    assert list(x.columns) == FEATURES
    assert len(x) == len(df) == len(y)
    # First-ever match: both teams at base Elo -> elo_diff is just the home edge,
    # and form features are NaN (no history).
    assert x.iloc[0]["elo_diff"] == pytest.approx(60.0)  # default home_adv
    assert np.isnan(x.iloc[0]["home_ppg"])


def test_snapshot_is_pre_match():
    # A strong team that has won should carry a positive elo_diff at home later.
    df = _league()
    _, _, state = build_training(df)
    feat = state.snapshot("A", "F", pd.Timestamp("2030-01-01"), False)
    assert feat["elo_diff"] > 0  # A (strong) vs F (weak) at home


# --- boosting 1X2 --------------------------------------------------------

@pytest.fixture(scope="module")
def boost():
    return BoostingModel(n_estimators=120, seed=1).fit(_league(), asof=date(2030, 1, 1))


def test_predict_valid_distribution(boost):
    p = boost.predict_1x2(Fixture("A", "F", date(2030, 1, 2), "L", "league"))
    assert set(p) == set(OUTCOMES)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)


def test_favors_stronger_team(boost):
    p = boost.predict_1x2(Fixture("A", "F", date(2030, 1, 2), "L", "league"))
    assert p["H"] > p["A"]


def test_beats_base_rate_on_holdout():
    from football_forecast.eval import metrics as M
    from football_forecast.data.schema import match_outcome
    df = _league()
    origin = date(2017, 8, 1)  # within the synthetic data's span
    model = BoostingModel(n_estimators=150, seed=1).fit(df, asof=origin)
    test = df[pd.to_datetime(df["date"]) >= pd.Timestamp(origin)]
    train = df[pd.to_datetime(df["date"]) < pd.Timestamp(origin)]
    base = train.assign(o=[match_outcome(h, a) for h, a in zip(train.home_goals, train.away_goals)])
    rates = base["o"].value_counts(normalize=True).reindex(OUTCOMES).fillna(0).to_dict()
    rps_model = np.mean([M.rps(model.predict_1x2(Fixture(r.home, r.away, pd.Timestamp(r.date).date(), "L", "league")),
                               match_outcome(r.home_goals, r.away_goals)) for r in test.itertuples(index=False)])
    rps_base = np.mean([M.rps(rates, match_outcome(r.home_goals, r.away_goals)) for r in test.itertuples(index=False)])
    assert rps_model < rps_base


# --- boosting counts -----------------------------------------------------

def test_count_regressor_predicts_positive():
    m = BoostingCountModel("corners", objective="poisson", n_estimators=120, seed=1).fit(
        _league(), asof=date(2030, 1, 1)
    )
    mh, ma = m.expected_counts(Fixture("A", "F", date(2030, 1, 2), "L", "league"))
    assert mh > 0 and ma > 0
