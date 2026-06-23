"""Proper scoring rules for probabilistic 1X2 forecasts.

Accuracy is the wrong metric (docs/models-explained.md §11). We score with RPS
(primary, because H/D/A is ordinal), log loss, and Brier. Every model is judged
through these on identical time-ordered splits.

Probabilities are taken in `schema.OUTCOMES` order — ("H", "D", "A") — passed
either as a 3-sequence or a mapping keyed by those labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from football_forecast.data.schema import OUTCOMES

_N = len(OUTCOMES)


def _as_vector(probs: Mapping[str, float] | Sequence[float]) -> np.ndarray:
    """Coerce probs to a float array in OUTCOMES order; do not renormalize."""
    if isinstance(probs, Mapping):
        return np.array([float(probs[o]) for o in OUTCOMES], dtype=float)
    vec = np.asarray(probs, dtype=float)
    if vec.shape != (_N,):
        raise ValueError(f"expected {_N} probabilities, got shape {vec.shape}")
    return vec


def _onehot(outcome: str) -> np.ndarray:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome {outcome!r} not in {OUTCOMES}")
    return np.array([1.0 if o == outcome else 0.0 for o in OUTCOMES], dtype=float)


def rps(probs: Mapping[str, float] | Sequence[float], outcome: str) -> float:
    """Ranked Probability Score for one forecast (lower is better; 0..1).

    RPS = 1/(C-1) * sum_{k=1}^{C-1} ( sum_{i<=k} (p_i - o_i) )^2 over the ordinal
    categories. A certain, correct forecast scores 0; a certain, maximally-wrong
    one (predicting the opposite extreme) scores 1.
    """
    p = _as_vector(probs)
    o = _onehot(outcome)
    cum = np.cumsum(p - o)[:-1]  # k = 1 .. C-1
    return float(np.sum(cum**2) / (_N - 1))


def log_loss(
    probs: Mapping[str, float] | Sequence[float], outcome: str, eps: float = 1e-15
) -> float:
    """Negative log-likelihood of the realized outcome (cross-entropy)."""
    p = _as_vector(probs)
    return float(-np.log(np.clip(p[OUTCOMES.index(outcome)], eps, 1.0)))


def brier(probs: Mapping[str, float] | Sequence[float], outcome: str) -> float:
    """Multiclass Brier score: squared error of the probability vector."""
    p = _as_vector(probs)
    return float(np.sum((p - _onehot(outcome)) ** 2))


def mean_rps(
    probs_rows: Sequence[Mapping[str, float] | Sequence[float]],
    outcomes: Sequence[str],
) -> float:
    """Mean RPS over many forecasts (the headline backtest number)."""
    if len(probs_rows) != len(outcomes):
        raise ValueError("probs_rows and outcomes must have equal length")
    if not probs_rows:
        raise ValueError("cannot average over zero forecasts")
    return float(np.mean([rps(p, o) for p, o in zip(probs_rows, outcomes)]))
