"""Tests for Phase 4: league source, market de-vig, and count models."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from football_forecast.data.sources import club_matches as cm
from football_forecast.eval.market import devig, market_probs, overround
from football_forecast.models.counts import CountModel


# --- league source -------------------------------------------------------

def _raw_csv(tmp_path):
    p = tmp_path / "matches.csv"
    pd.DataFrame(
        {
            "Division": ["E0", "E0", "F1"],
            "MatchDate": ["2023-08-12", "2023-08-19", "2023-08-12"],
            "MatchTime": ["15:00", "15:00", "20:00"],
            "HomeTeam": ["Arsenal", "Chelsea", "Lyon"],
            "AwayTeam": ["Forest", "Arsenal", "Lille"],
            "FTHome": [2, 1, 0], "FTAway": [1, 1, 0], "FTResult": ["H", "D", "D"],
            "HomeFouls": [10, 12, 9], "AwayFouls": [11, 9, 14],
            "HomeCorners": [6, 4, 5], "AwayCorners": [3, 7, 5],
            "HomeYellow": [1, 2, 3], "AwayYellow": [2, 1, 1],
            "HomeRed": [0, 0, 0], "AwayRed": [0, 1, 0],
            "OddHome": [1.5, 2.1, 2.5], "OddDraw": [4.0, 3.3, 3.1], "OddAway": [6.0, 3.6, 2.9],
        }
    ).to_csv(p, index=False)
    return p


def test_load_filters_division_and_maps_columns(tmp_path):
    df = cm.load(_raw_csv(tmp_path), division="E0")
    assert len(df) == 2
    assert set(df["competition"]) == {"Premier League"}
    assert (df["comp_type"] == "league").all()
    assert "home_fouls" in df and "odds_h" in df
    assert df.iloc[0]["season"] == "2023/2024"


def test_load_unknown_division_raises(tmp_path):
    with pytest.raises(ValueError, match="no rows for division"):
        cm.load(_raw_csv(tmp_path), division="ZZ")


# --- market de-vig -------------------------------------------------------

def test_devig_sums_to_one_and_favours_short_odds():
    p = devig(1.5, 4.0, 6.0)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["H"] > p["A"]  # shorter home odds -> higher prob


def test_overround_positive_for_real_odds():
    assert overround(1.5, 4.0, 6.0) > 0


def test_market_probs_handles_missing(tmp_path):
    df = cm.load(_raw_csv(tmp_path), division="E0")
    probs = market_probs(df)
    assert len(probs) == 2 and all(abs(sum(p.values()) - 1) < 1e-9 for p in probs)


# --- count model ---------------------------------------------------------

def _league(seed=0, n_seasons=6):
    """Synthetic league: teams differ in corner tendency; corners overdispersed."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D", "E", "F"]
    cfor = {t: v for t, v in zip(teams, np.linspace(0.4, -0.4, len(teams)))}
    rows, d = [], pd.Timestamp("2017-08-01")
    for _ in range(n_seasons):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                mu_h = np.exp(1.6 + 0.1 + cfor[h] - cfor[a])
                mu_a = np.exp(1.6 + cfor[a] - cfor[h])
                # NegBin-ish overdispersion via gamma-poisson mixture
                hc = rng.poisson(mu_h * rng.gamma(4, 1 / 4))
                ac = rng.poisson(mu_a * rng.gamma(4, 1 / 4))
                rows.append({
                    "match_id": f"m{len(rows)}", "date": d, "competition": "L",
                    "comp_type": "league", "home": h, "away": a,
                    "home_goals": rng.poisson(1.4), "away_goals": rng.poisson(1.1),
                    "neutral": False, "home_corners": hc, "away_corners": ac,
                })
                d += pd.Timedelta(days=1)
    from football_forecast.data.schema import validate
    return validate(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def corner_model():
    return CountModel("corners", family="auto").fit(_league(), asof=date(2030, 1, 1))


def test_count_model_recovers_for_ordering(corner_model):
    # Team A tends to win more corners than F.
    assert corner_model.for_["A"] > corner_model.for_["F"]


def test_count_model_detects_overdispersion(corner_model):
    assert corner_model.dispersion_ > 1.15
    assert corner_model.family_ == "nbinom"


def test_expected_counts_positive(corner_model):
    mh, ma = corner_model.expected_counts("A", "F")
    assert mh > 0 and ma > 0 and mh > ma  # strong home vs weak away -> more home corners


def test_predictive_interval_covers(corner_model):
    lo, hi = corner_model.interval(6.0, level=0.8)
    assert 0 <= lo < hi


def test_count_coverage_is_calibrated(corner_model):
    # On fresh data the 80% predictive interval should cover ~80% of outcomes.
    test = _league(seed=99, n_seasons=2)
    covered = tot = 0
    for r in test.itertuples(index=False):
        for actor, opp, ih, col in [(r.home, r.away, 1.0, "home_corners"),
                                    (r.away, r.home, 0.0, "away_corners")]:
            mu = corner_model._expected(actor, opp, ih)
            lo, hi = corner_model.interval(mu, 0.8)
            covered += lo <= getattr(r, col) <= hi
            tot += 1
    assert 0.7 <= covered / tot <= 0.9  # ~80%
