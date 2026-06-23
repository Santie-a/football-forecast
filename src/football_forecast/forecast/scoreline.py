"""The keystone: turn two goal rates into a scoreline probability matrix.

Everything else (1X2, over/under, correct score) is a sum over this matrix
(see markets.py and docs/models-explained.md §3). Independent Poisson by
default; pass `dc_rho` for the Dixon–Coles low-score correction (§5).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def scoreline_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
    dc_rho: float | None = None,
) -> np.ndarray:
    """P(home=i, away=j) as an (max_goals+1) x (max_goals+1) array.

    Rows = home goals i, columns = away goals j. Truncated at `max_goals` and
    renormalized to sum to 1 (the lost tail mass and the Dixon–Coles correction
    both make raw cells not quite sum to 1).
    """
    ks = np.arange(max_goals + 1)
    ph = poisson.pmf(ks, lambda_home)
    pa = poisson.pmf(ks, lambda_away)
    m = np.outer(ph, pa)

    if dc_rho:
        m = m.copy()
        m[0, 0] *= 1.0 - lambda_home * lambda_away * dc_rho
        m[0, 1] *= 1.0 + lambda_home * dc_rho
        m[1, 0] *= 1.0 + lambda_away * dc_rho
        m[1, 1] *= 1.0 - dc_rho
        m = np.clip(m, 0.0, None)  # guard against a pathological rho

    total = m.sum()
    if total <= 0:
        raise ValueError("degenerate scoreline matrix (sum <= 0)")
    return m / total


def dc_tau(i: int, j: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """The Dixon–Coles low-score correction factor for a single cell (=1 outside
    the four lowest cells). Exposed for the model's rho-fitting step and tests."""
    if i == 0 and j == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if i == 0 and j == 1:
        return 1.0 + lambda_home * rho
    if i == 1 and j == 0:
        return 1.0 + lambda_away * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0
