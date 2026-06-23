"""Tests for the Elo model: rating dynamics, the M03 neutral rule, and the
draw-aware 1X2 head."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from football_forecast.data.schema import OUTCOMES, validate
from football_forecast.models.elo import EloModel, _expected_home, _mov_multiplier


def _matches() -> pd.DataFrame:
    # A always beats B; ratings should diverge A up, B down.
    rows = []
    for i in range(20):
        rows.append(
            {
                "match_id": f"m{i}",
                "date": pd.Timestamp("2000-01-01") + pd.Timedelta(days=i),
                "competition": "Friendly",
                "comp_type": "friendly",
                "home": "A",
                "away": "B",
                "home_goals": 2,
                "away_goals": 0,
                "neutral": False,
            }
        )
    return validate(pd.DataFrame(rows))


def test_probs_sum_to_one():
    m = _matches()
    model = EloModel().fit(m, asof=date(2001, 1, 1))
    from football_forecast.data.schema import Fixture

    p = model.predict_1x2(Fixture("A", "B", date(2001, 1, 2), "Friendly", "friendly"))
    assert set(p) == set(OUTCOMES)
    assert sum(p.values()) == pytest.approx(1.0)


def test_winner_gains_rating():
    m = _matches()
    model = EloModel().fit(m, asof=date(2001, 1, 1))
    assert model.ratings_["A"] > model.ratings_["B"]


def test_stronger_team_favored():
    m = _matches()
    model = EloModel().fit(m, asof=date(2001, 1, 1))
    from football_forecast.data.schema import Fixture

    p = model.predict_1x2(Fixture("A", "B", date(2001, 1, 2), "Friendly", "friendly"))
    assert p["H"] > p["A"]


def test_neutral_removes_home_advantage():
    # Same matchup, equal ratings: on neutral ground H and A must be symmetric.
    model = EloModel()
    model.ratings_ = {"A": 1500.0, "B": 1500.0}
    model.nu_ = 1.0
    from football_forecast.data.schema import Fixture

    home = model.predict_1x2(Fixture("A", "B", date(2001, 1, 2), "Cup", "final_tournament", neutral=False))
    neut = model.predict_1x2(Fixture("A", "B", date(2001, 1, 2), "Cup", "final_tournament", neutral=True))
    assert home["H"] > home["A"]            # home edge when not neutral
    assert neut["H"] == pytest.approx(neut["A"])  # symmetric when neutral


def test_expected_home_monotone_and_centered():
    assert _expected_home(0.0) == pytest.approx(0.5)
    assert _expected_home(400.0) > _expected_home(0.0) > _expected_home(-400.0)


@pytest.mark.parametrize("gd,expected", [(0, 1.0), (1, 1.0), (2, 1.5), (3, 1.75)])
def test_mov_multiplier(gd, expected):
    assert _mov_multiplier(gd) == pytest.approx(expected)


def test_draw_share_peaks_for_even_teams():
    # With equal strengths the draw probability should exceed that of a lopsided tie.
    model = EloModel()
    model.nu_ = 1.5
    even = model._probs_from_gap(np.array([0.0]), model.nu_)[0]
    lop = model._probs_from_gap(np.array([400.0]), model.nu_)[0]
    assert even[1] > lop[1]  # index 1 = draw
