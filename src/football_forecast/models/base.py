"""Model protocols — the shared shape every forecaster implements.

Result-only models (Elo) implement OutcomeModel; goal models (Dixon–Coles,
Bayesian) additionally expose `rates()` and get `predict_1x2` for free via the
keystone. The backtester programs against these protocols, so a new model is one
new file and zero harness changes. See docs/implementation-plan.md §1.3.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from football_forecast.data.schema import Fixture


@runtime_checkable
class OutcomeModel(Protocol):
    """A forecaster that produces 1X2 probabilities."""

    def fit(self, matches: pd.DataFrame, asof: date) -> "OutcomeModel":
        """Fit on matches strictly before `asof`. Must not use any match with
        date >= asof — this is the no-leakage contract the backtester enforces."""
        ...

    def predict_1x2(self, fixture: Fixture) -> dict[str, float]:
        """Return {"H":.., "D":.., "A":..} summing to 1."""
        ...


@runtime_checkable
class GoalModel(OutcomeModel, Protocol):
    """A forecaster built on a goal-rate model; 1X2 is derived from its rates."""

    def rates(self, fixture: Fixture) -> tuple[float, float]:
        """Return (lambda_home, lambda_away) expected goals for the fixture."""
        ...
