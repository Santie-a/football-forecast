"""Drain the fixtures queue: compute forecasts for pending fixtures.

This is the cheap-inference step — it needs an already-fitted model (never trains
here), so it can run on the Pi with a synced model pickle as well as on the PC.
A pending fixture is one with no forecast yet (store/fixtures.py).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from football_forecast.data.schema import Fixture
from football_forecast.forecast.bundle import markets_for_model
from football_forecast.store import fixtures as fxstore


def drain(model, model_name: str, path: str | Path = fxstore.DEFAULT_PATH) -> int:
    """Compute and store a forecast bundle for every pending fixture. Returns the
    number processed. The model must already be fitted (has rates/predict_1x2)."""
    pending = fxstore.list_pending(path)
    for f in pending:
        fixture = Fixture(
            home=f["home"],
            away=f["away"],
            date=date.fromisoformat(f["date"]),
            competition=f["competition"],
            comp_type=f["comp_type"],
            neutral=f["neutral"],
        )
        bundle = markets_for_model(model, fixture)
        fxstore.mark_forecast(f["fixture_id"], model_name, bundle, path)
    return len(pending)
