"""Calibration analysis for 1X2 forecasts (docs/models-explained.md §11).

Are events called "20%" happening ~20% of the time? We pool every class
probability (H, D, A) across all matches into a reliability table: predicted
probability vs observed frequency per bin. The expected calibration error (ECE)
is the count-weighted mean gap — 0 is perfect calibration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from football_forecast.data.schema import OUTCOMES


def _stack(probs_rows, outcomes) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to (predicted_prob, hit) over every (match, class) pair."""
    if len(probs_rows) != len(outcomes):
        raise ValueError("probs_rows and outcomes must have equal length")
    preds, hits = [], []
    for p, o in zip(probs_rows, outcomes):
        vec = [float(p[k]) for k in OUTCOMES] if isinstance(p, Mapping) else list(p)
        for k, label in enumerate(OUTCOMES):
            preds.append(vec[k])
            hits.append(1.0 if label == o else 0.0)
    return np.asarray(preds), np.asarray(hits)


def reliability_table(
    probs_rows: Sequence,
    outcomes: Sequence[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-bin predicted vs observed frequency (the reliability diagram, as data)."""
    preds, hits = _stack(probs_rows, outcomes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(preds, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "count": n,
                "mean_pred": float(preds[sel].mean()) if n else np.nan,
                "obs_freq": float(hits[sel].mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    probs_rows: Sequence, outcomes: Sequence[str], n_bins: int = 10
) -> float:
    """Count-weighted mean |predicted - observed| over the bins (0 = perfect)."""
    tbl = reliability_table(probs_rows, outcomes, n_bins).dropna()
    if tbl["count"].sum() == 0:
        return float("nan")
    gap = (tbl["mean_pred"] - tbl["obs_freq"]).abs()
    return float(np.average(gap, weights=tbl["count"]))
