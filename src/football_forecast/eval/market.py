"""The market benchmark: bookmaker odds → de-vigged implied 1X2 probabilities.

Decimal odds imply probabilities 1/odds, but they sum to >1 by the bookmaker's
margin ("vig" / overround). De-vigging removes it so the three probabilities sum
to 1. We use the basic proportional (normalization) method — the standard
baseline; fancier methods (Shin, power) are a possible refinement.

The de-vigged closing line is the gold-standard baseline (docs/models-explained.md
§12): beating it is extremely hard; getting close is a strong result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_forecast.data.schema import OUTCOMES


def devig(odds_h: float, odds_d: float, odds_a: float) -> dict[str, float]:
    """One match: decimal odds → de-vigged {H, D, A} (proportional method)."""
    inv = np.array([1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a], dtype=float)
    return dict(zip(OUTCOMES, inv / inv.sum()))


def overround(odds_h: float, odds_d: float, odds_a: float) -> float:
    """The bookmaker margin: (sum of inverse odds) - 1."""
    return float(1.0 / odds_h + 1.0 / odds_d + 1.0 / odds_a - 1.0)


def market_probs(df: pd.DataFrame) -> list[dict[str, float] | None]:
    """De-vigged {H,D,A} per row from odds_h/odds_d/odds_a; None where odds missing."""
    out: list[dict[str, float] | None] = []
    for r in df.itertuples(index=False):
        oh, od, oa = getattr(r, "odds_h", None), getattr(r, "odds_d", None), getattr(r, "odds_a", None)
        if oh and od and oa and np.isfinite([oh, od, oa]).all():
            out.append(devig(oh, od, oa))
        else:
            out.append(None)
    return out
