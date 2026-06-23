"""Tests for the generic fixtures importer."""

import pandas as pd
import pytest

from football_forecast import fixtures_import as fi
from football_forecast.store import fixtures as fx


def _raw_csv(tmp_path):
    """A martj42-format raw CSV with one played and one unplayed match."""
    p = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "date": ["2026-06-11", "2026-06-25", "2025-01-01"],
            "home_team": ["Mexico", "Brazil", "Spain"],
            "away_team": ["South Africa", "Serbia", "France"],
            "home_score": [2, None, 1],          # match 2 unplayed
            "away_score": [0, None, 1],
            "tournament": ["FIFA World Cup", "FIFA World Cup", "Friendly"],
            "neutral": [False, True, True],
        }
    ).to_csv(p, index=False)
    return p


def test_from_raw_filters_competition_and_date(tmp_path):
    df = fi.from_raw_results(_raw_csv(tmp_path), competition="FIFA World Cup", since="2026-06-01")
    assert len(df) == 2  # the Friendly and pre-2026 rows excluded
    assert set(df["comp_type"]) == {"final_tournament"}


def test_from_raw_keeps_unplayed(tmp_path):
    df = fi.from_raw_results(_raw_csv(tmp_path), competition="FIFA World Cup")
    unplayed = df[df["home_goals"].isna()]
    assert len(unplayed) == 1 and unplayed.iloc[0]["home"] == "Brazil"


def test_from_csv_defaults_and_validation(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"date": ["2026-07-01"], "home": ["A"], "away": ["B"]}).to_csv(p, index=False)
    df = fi.from_csv(p)
    assert df.iloc[0]["competition"] == "Unknown"
    assert df.iloc[0]["comp_type"] == "friendly"
    assert not bool(df.iloc[0]["neutral"])


def test_from_csv_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2026-07-01"], "home": ["A"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        fi.from_csv(p)


def test_from_csv_rejects_bad_comp_type(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame(
        {"date": ["2026-07-01"], "home": ["A"], "away": ["B"], "comp_type": ["scrimmage"]}
    ).to_csv(p, index=False)
    with pytest.raises(ValueError, match="unknown comp_type"):
        fi.from_csv(p)


class _StubModel:
    use_dc = False
    max_goals = 10

    def rates(self, fixture):
        return 1.5, 1.0


def test_register_splits_played_and_queued(tmp_path):
    db = tmp_path / "fx.sqlite"
    df = fi.from_raw_results(_raw_csv(tmp_path), competition="FIFA World Cup")
    counts = fi.register(df, path=db, model=_StubModel(), model_name="dixon_coles")
    assert counts == {"played": 1, "queued": 1, "total": 2}
    # Played match got a forecast; unplayed is still pending in the queue.
    played = fx.get(fx.make_id("2026-06-11", "Mexico", "South Africa"), path=db)
    assert played["status"] == "played" and played["forecast"] is not None
    assert len(fx.list_pending(db)) == 1


def test_register_without_model_leaves_played_unforecast(tmp_path):
    db = tmp_path / "fx.sqlite"
    df = fi.from_raw_results(_raw_csv(tmp_path), competition="FIFA World Cup")
    fi.register(df, path=db)  # no model
    played = fx.get(fx.make_id("2026-06-11", "Mexico", "South Africa"), path=db)
    assert played["status"] == "played" and played["forecast"] is None
