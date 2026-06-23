"""Tests for the canonical match record: validation rules and the Fixture type."""

import pandas as pd
import pytest

from football_forecast.data.schema import (
    Fixture,
    add_outcome,
    match_outcome,
    validate,
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m2", "m1"],
            "date": ["2022-11-21", "2022-11-20"],
            "competition": ["FIFA World Cup", "FIFA World Cup"],
            "comp_type": ["final_tournament", "final_tournament"],
            "home": ["Argentina", "Qatar"],
            "away": ["Saudi Arabia", "Ecuador"],
            "home_goals": [2, 0],
            "away_goals": [1, 2],
            "neutral": [True, False],
        }
    )


def test_validate_passes_and_sorts_by_date():
    out = validate(_valid_frame())
    assert list(out["match_id"]) == ["m1", "m2"]  # time-ordered
    assert out["neutral"].dtype == bool
    assert str(out["date"].dtype).startswith("datetime64")


def test_validate_rejects_missing_column():
    df = _valid_frame().drop(columns=["neutral"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate(df)


def test_validate_rejects_self_match():
    df = _valid_frame()
    df.loc[0, "away"] = df.loc[0, "home"]
    with pytest.raises(ValueError, match="home == away"):
        validate(df)


def test_validate_rejects_unknown_comp_type():
    df = _valid_frame()
    df.loc[0, "comp_type"] = "exhibition"
    with pytest.raises(ValueError, match="unknown comp_type"):
        validate(df)


def test_validate_rejects_null_in_required():
    df = _valid_frame()
    df.loc[0, "home_goals"] = None
    with pytest.raises(ValueError, match="null values"):
        validate(df)


def test_validate_rejects_negative_goals():
    df = _valid_frame()
    df.loc[0, "home_goals"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate(df)


@pytest.mark.parametrize(
    "h,a,expected", [(2, 0, "H"), (1, 1, "D"), (0, 3, "A")]
)
def test_match_outcome(h, a, expected):
    assert match_outcome(h, a) == expected


def test_add_outcome_column():
    out = add_outcome(validate(_valid_frame()))
    assert list(out["outcome"]) == ["A", "H"]  # m1: 0-2 away, m2: 1-2 away


def test_fixture_rejects_self_match():
    with pytest.raises(ValueError, match="must differ"):
        Fixture("Brazil", "Brazil", pd.Timestamp("2026-06-01"), "Friendly", "friendly")


def test_fixture_rejects_bad_comp_type():
    with pytest.raises(ValueError, match="comp_type"):
        Fixture("Brazil", "Peru", pd.Timestamp("2026-06-01"), "X", "scrimmage")
