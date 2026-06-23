"""Walk-forward backtesting — the one harness every model is judged through.

Strictly time-ordered: at each origin date the model is fit on matches *before*
that date and scored on matches in the following window. Never a random split
(docs/modeling-guidelines.md). The model receives the full match frame plus the
`asof` date and is responsible for filtering to the past; the leakage test
verifies it does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_forecast.data.schema import Fixture, match_outcome
from football_forecast.eval import metrics as M

# name -> per-forecast scoring function (probs, outcome) -> float
DEFAULT_METRICS: dict[str, Callable] = {
    "rps": M.rps,
    "log_loss": M.log_loss,
    "brier": M.brier,
}


@dataclass
class BacktestResult:
    per_origin: pd.DataFrame  # one row per origin: n_train, n_test, <metric means>
    overall: dict[str, float]  # metrics pooled over every scored match
    n_test: int

    def __str__(self) -> str:
        bits = ", ".join(f"{k}={v:.4f}" for k, v in self.overall.items())
        return f"BacktestResult(n_test={self.n_test}, {bits})"


def yearly_origins(matches: pd.DataFrame, start: int, end: int | None = None) -> list[pd.Timestamp]:
    """Jan-1 origins from `start` to the last match year (inclusive)."""
    dates = pd.to_datetime(matches["date"])
    last = end if end is not None else int(dates.max().year)
    return [pd.Timestamp(year=y, month=1, day=1) for y in range(start, last + 1)]


def _fixture(row) -> Fixture:
    return Fixture(
        home=row.home,
        away=row.away,
        date=pd.Timestamp(row.date).date(),
        competition=row.competition,
        comp_type=row.comp_type,
        neutral=bool(row.neutral),
    )


def walk_forward(
    model_factory: Callable[[], object],
    matches: pd.DataFrame,
    origins: Sequence[pd.Timestamp],
    metrics: Mapping[str, Callable] | None = None,
    min_train_matches: int = 100,
) -> BacktestResult:
    metrics = dict(metrics or DEFAULT_METRICS)
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    origins = sorted(pd.Timestamp(o) for o in origins)
    rows: list[dict] = []
    pooled: dict[str, list[float]] = {name: [] for name in metrics}

    for i, origin in enumerate(origins):
        nxt = origins[i + 1] if i + 1 < len(origins) else pd.Timestamp.max
        train_mask = df["date"] < origin
        test = df[(df["date"] >= origin) & (df["date"] < nxt)]
        if int(train_mask.sum()) < min_train_matches or len(test) == 0:
            continue

        model = model_factory().fit(df, asof=origin.date())

        fold: dict[str, list[float]] = {name: [] for name in metrics}
        for row in test.itertuples(index=False):
            probs = model.predict_1x2(_fixture(row))
            outcome = match_outcome(row.home_goals, row.away_goals)
            for name, fn in metrics.items():
                val = fn(probs, outcome)
                fold[name].append(val)
                pooled[name].append(val)

        rows.append(
            {
                "origin": origin,
                "n_train": int(train_mask.sum()),
                "n_test": len(test),
                **{name: float(np.mean(vals)) for name, vals in fold.items()},
            }
        )

    if not rows:
        raise ValueError(
            "no usable folds — check origins, min_train_matches, and data span"
        )

    per_origin = pd.DataFrame(rows)
    overall = {name: float(np.mean(vals)) for name, vals in pooled.items() if vals}
    n_test = int(per_origin["n_test"].sum())
    return BacktestResult(per_origin=per_origin, overall=overall, n_test=n_test)
