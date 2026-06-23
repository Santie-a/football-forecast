"""Tests for the forecast store: round-trip and idempotent writes."""

import pandas as pd

from football_forecast.store.forecasts import read_forecasts, write_forecasts


def _frame(prob_h: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2026-06-01",
                "home": "Brazil",
                "away": "Peru",
                "competition": "Friendly",
                "model": "elo",
                "market": "1x2",
                "payload": {"H": prob_h, "D": 0.2, "A": round(1 - prob_h - 0.2, 3)},
            }
        ]
    )


def test_round_trip(tmp_path):
    path = tmp_path / "f.sqlite"
    n = write_forecasts(_frame(0.5), path)
    assert n == 1
    df = read_forecasts(path, model="elo", market="1x2")
    assert len(df) == 1
    assert df.iloc[0]["payload"]["H"] == 0.5
    assert df.iloc[0]["home"] == "Brazil"


def test_write_is_idempotent_per_model_market(tmp_path):
    path = tmp_path / "f.sqlite"
    write_forecasts(_frame(0.5), path)
    write_forecasts(_frame(0.6), path)  # same model+market -> replace, not append
    df = read_forecasts(path)
    assert len(df) == 1
    assert df.iloc[0]["payload"]["H"] == 0.6
