"""Markets derived from the scoreline matrix — all just sums over its cells
(docs/models-explained.md §3). One goal model feeds every market; none are
modelled independently.

Convention: rows = home goals i, columns = away goals j.
"""

from __future__ import annotations

import numpy as np

from football_forecast.data.schema import OUTCOMES


def one_x_two(m: np.ndarray) -> dict[str, float]:
    """{"H","D","A"}: home win (i>j), draw (i=j), away win (i<j)."""
    home = float(np.tril(m, -1).sum())  # below diagonal
    draw = float(np.trace(m))
    away = float(np.triu(m, 1).sum())   # above diagonal
    return dict(zip(OUTCOMES, (home, draw, away)))


def over_under(m: np.ndarray, line: float = 2.5) -> dict[str, float]:
    """{"over","under"} total goals vs a (typically .5) line."""
    n = m.shape[0]
    totals = np.add.outer(np.arange(n), np.arange(n))
    over = float(m[totals > line].sum())
    return {"over": over, "under": 1.0 - over}


def btts(m: np.ndarray) -> dict[str, float]:
    """Both teams to score: both i>=1 and j>=1."""
    yes = float(m[1:, 1:].sum())
    return {"yes": yes, "no": 1.0 - yes}


def correct_score(m: np.ndarray, top: int | None = None) -> dict[tuple[int, int], float]:
    """Per-scoreline probabilities {(i,j): p}, optionally only the `top` most likely."""
    cells = {(int(i), int(j)): float(m[i, j]) for i in range(m.shape[0]) for j in range(m.shape[1])}
    if top is None:
        return cells
    return dict(sorted(cells.items(), key=lambda kv: kv[1], reverse=True)[:top])
