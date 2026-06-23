"""Tests for the Bayesian hierarchical model. MCMC is kept tiny for speed; these
check the model's qualitative properties, not exact numbers."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from football_forecast.data.schema import Fixture, validate
from football_forecast.models.bayesian import BayesianModel, _split_rhat


def _synthetic(seed: int = 0) -> pd.DataFrame:
    """Strong > Mid > Weak with many matches each; 'Rare' plays very few."""
    rng = np.random.default_rng(seed)
    atk = {"Strong": 0.6, "Mid": 0.0, "Weak": -0.6, "Rare": 0.0}
    dfn = {"Strong": 0.5, "Mid": 0.0, "Weak": -0.5, "Rare": 0.0}
    mu, home = 0.2, 0.25
    rows, d = [], pd.Timestamp("2018-01-01")

    def add(h, a):
        nonlocal d
        lh = np.exp(mu + home + atk[h] - dfn[a])
        la = np.exp(mu + atk[a] - dfn[h])
        rows.append({
            "match_id": f"m{len(rows)}", "date": d, "competition": "T", "comp_type": "friendly",
            "home": h, "away": a, "home_goals": int(rng.poisson(lh)),
            "away_goals": int(rng.poisson(la)), "neutral": False,
        })
        d += pd.Timedelta(days=2)

    core = ["Strong", "Mid", "Weak"]
    for _ in range(30):
        for h in core:
            for a in core:
                if h != a:
                    add(h, a)
    for _ in range(3):  # Rare plays only a handful
        add("Rare", "Mid")
    return validate(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def fitted():
    return BayesianModel(window_days=None, draws=300, tune=300, chains=2, seed=1).fit(
        _synthetic(), asof=date(2030, 1, 1)
    )


def test_converges(fitted):
    assert np.isfinite(fitted.max_rhat_)
    assert fitted.max_rhat_ < 1.1  # loose for a tiny chain


def test_recovers_strength_ordering(fitted):
    a = {t: fitted.atk_[:, i].mean() for t, i in fitted.idx_.items()}
    assert a["Strong"] > a["Mid"] > a["Weak"]


def test_predict_is_valid_and_favors_strong(fitted):
    p = fitted.predict_1x2(Fixture("Strong", "Weak", date(2030, 1, 2), "T", "friendly"))
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)
    assert p["H"] > p["A"]


def test_sparse_team_has_wider_posterior(fitted):
    # The whole point of partial pooling: a rarely-seen team is more uncertain.
    assert fitted.atk_std_["Rare"] > fitted.atk_std_["Strong"]


def test_scoreline_matrix_normalized(fitted):
    m = fitted.scoreline(Fixture("Strong", "Mid", date(2030, 1, 2), "T", "friendly"))
    assert m.sum() == pytest.approx(1.0)


def test_split_rhat_detects_disagreement():
    # Two chains stuck at different values (shape: chain, draw, param) -> R-hat >> 1.
    bad = {"p": np.concatenate([np.zeros((1, 100, 1)), np.ones((1, 100, 1))], axis=0)}
    assert _split_rhat(bad) > 1.5
