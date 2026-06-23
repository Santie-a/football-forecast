"""Tests for the Poisson goal models. We generate matches from known team
strengths and check the fit recovers the ordering and produces sane forecasts."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from football_forecast.data.schema import Fixture, validate
from football_forecast.models.dixon_coles import DixonColesModel, MaherModel


def _synthetic(seed: int = 0, n_per_pair: int = 40) -> pd.DataFrame:
    """Three teams: Strong > Mid > Weak, generated from independent Poisson."""
    rng = np.random.default_rng(seed)
    atk = {"Strong": 0.5, "Mid": 0.0, "Weak": -0.5}
    dfn = {"Strong": 0.4, "Mid": 0.0, "Weak": -0.4}
    mu, home = 0.1, 0.25
    teams = list(atk)
    rows, d = [], pd.Timestamp("2005-01-01")
    for _ in range(n_per_pair):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lh = np.exp(mu + home + atk[h] - dfn[a])
                la = np.exp(mu + atk[a] - dfn[h])
                rows.append(
                    {
                        "match_id": f"m{len(rows)}",
                        "date": d,
                        "competition": "Test",
                        "comp_type": "friendly",
                        "home": h,
                        "away": a,
                        "home_goals": int(rng.poisson(lh)),
                        "away_goals": int(rng.poisson(la)),
                        "neutral": False,
                    }
                )
                d += pd.Timedelta(days=1)
    return validate(pd.DataFrame(rows))


def test_recovers_strength_ordering():
    model = MaherModel().fit(_synthetic(), asof=date(2030, 1, 1))
    assert model.atk_["Strong"] > model.atk_["Mid"] > model.atk_["Weak"]
    assert model.def_["Strong"] > model.def_["Mid"] > model.def_["Weak"]


def test_home_advantage_positive():
    model = MaherModel().fit(_synthetic(), asof=date(2030, 1, 1))
    assert model.home_ > 0


def test_predict_1x2_valid_and_favors_strong():
    model = DixonColesModel().fit(_synthetic(), asof=date(2030, 1, 1))
    p = model.predict_1x2(Fixture("Strong", "Weak", date(2030, 1, 2), "Test", "friendly"))
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["H"] > p["A"]


def test_neutral_reduces_home_rate():
    model = DixonColesModel().fit(_synthetic(), asof=date(2030, 1, 1))
    lh_home, _ = model.rates(Fixture("Strong", "Mid", date(2030, 1, 2), "T", "friendly"))
    lh_neut, _ = model.rates(Fixture("Strong", "Mid", date(2030, 1, 2), "T", "friendly", neutral=True))
    assert lh_home > lh_neut  # home advantage dropped on neutral ground


def test_dixon_coles_fits_low_score_rho():
    model = DixonColesModel().fit(_synthetic(), asof=date(2030, 1, 1))
    assert -0.25 <= model.rho_ <= 0.25


def test_unknown_team_uses_average():
    model = MaherModel().fit(_synthetic(), asof=date(2030, 1, 1))
    # An unseen team gets attack/defence 0 -> a valid, finite forecast.
    p = model.predict_1x2(Fixture("Strong", "Newland", date(2030, 1, 2), "T", "friendly"))
    assert sum(p.values()) == pytest.approx(1.0)
    assert all(np.isfinite(list(p.values())))
