"""Bundle every market derived from one scoreline matrix into store-ready
payloads. This is the single place that decides *what* a goal model publishes —
all of it summed from the one matrix (docs/models-explained.md §3).
"""

from __future__ import annotations

import numpy as np

from football_forecast.data.schema import Fixture
from football_forecast.forecast.markets import (
    correct_score,
    one_x_two,
    over_under,
)
from football_forecast.forecast.scoreline import scoreline_matrix

DEFAULT_LINES = (1.5, 2.5, 3.5)


def markets_for_model(model, fixture: Fixture) -> dict[str, dict]:
    """Full market bundle for a goal model (exposes `rates`); 1X2 only for a
    result-only model (e.g. Elo). The single place that decides what each model
    type publishes — reused by the forecast pipeline and the queue drainer."""
    if hasattr(model, "rates"):
        lh, la = model.rates(fixture)
        rho = getattr(model, "rho_", None) if getattr(model, "use_dc", False) else None
        max_goals = getattr(model, "max_goals", 10)
        return build_markets(scoreline_matrix(lh, la, max_goals, rho))
    return {"1x2": model.predict_1x2(fixture)}


def build_markets(
    matrix: np.ndarray,
    display_goals: int = 6,
    lines: tuple[float, ...] = DEFAULT_LINES,
    top_scores: int = 8,
    decimals: int = 5,
) -> dict[str, dict]:
    """Return {market: payload} for the store.

    - 1x2:           {"H","D","A"}
    - scoreline:     {"goals": G, "matrix": GxG nested list of P(i,j)} (display slice)
    - over_under:    {"2.5": {"over","under"}, ...}
    - correct_score: {"i-j": p} for the top `top_scores` scorelines
    """
    g = min(display_goals, matrix.shape[0] - 1)
    sub = np.round(matrix[: g + 1, : g + 1], decimals)

    def r(d: dict) -> dict:
        return {k: round(float(v), decimals) for k, v in d.items()}

    return {
        "1x2": r(one_x_two(matrix)),
        "scoreline": {"goals": g, "matrix": sub.tolist()},
        "over_under": {str(line): r(over_under(matrix, line)) for line in lines},
        "correct_score": {
            f"{i}-{j}": round(p, decimals)
            for (i, j), p in correct_score(matrix, top=top_scores).items()
        },
    }
