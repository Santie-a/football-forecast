"""The canonical match record — the data contract every model consumes.

One row per match, one schema across all sources (national + league). Optional
columns stay absent/null until a source provides them. No model reads a raw CSV;
sources normalize into this schema first. See docs/implementation-plan.md §1.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date

import pandas as pd

# Columns required from Phase 1 on. Optional columns (season, referee, counts,
# odds) are added by later-phase sources and are not enforced here.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "match_id",
    "date",
    "competition",
    "comp_type",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "neutral",
)

# The M01 competition-type feature. Friendlies are kept, not filtered (decision M01).
COMP_TYPES: frozenset[str] = frozenset(
    {"friendly", "qualifier", "final_tournament", "league", "continental"}
)

# Outcome categories in ordinal order (home win — draw — away win). RPS relies on
# this ordering (docs/models-explained.md §11), so it lives here as the single source.
OUTCOMES: tuple[str, str, str] = ("H", "D", "A")


@dataclass(frozen=True)
class Fixture:
    """A not-yet-played match: what a model is asked to forecast.

    Carries only pre-kickoff information (no goals). `neutral` drives the
    home-advantage rule (decision M03): on a neutral venue there is no host.
    """

    home: str
    away: str
    date: _date
    competition: str
    comp_type: str
    neutral: bool = False

    def __post_init__(self) -> None:
        if self.home == self.away:
            raise ValueError(f"home and away must differ (both {self.home!r})")
        if self.comp_type not in COMP_TYPES:
            raise ValueError(
                f"comp_type {self.comp_type!r} not in {sorted(COMP_TYPES)}"
            )


def match_outcome(home_goals: int, away_goals: int) -> str:
    """Map a final score to an ordinal outcome label in OUTCOMES."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an `outcome` column derived from the goals."""
    out = df.copy()
    out["outcome"] = [
        match_outcome(h, a) for h, a in zip(out["home_goals"], out["away_goals"])
    ]
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a match frame against the canonical schema.

    Coerces dtypes, enforces the required columns / no-null / no self-match /
    non-negative-goals / known-comp_type rules, and returns a time-sorted copy.
    Raises ValueError on any violation — fail loud, never silently drop rows.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    nulls = [c for c in REQUIRED_COLUMNS if out[c].isna().any()]
    if nulls:
        raise ValueError(f"null values in required columns: {nulls}")

    unknown = sorted(set(out["comp_type"].astype(str).unique()) - COMP_TYPES)
    if unknown:
        raise ValueError(f"unknown comp_type values: {unknown}")

    if (out["home"].astype(str) == out["away"].astype(str)).any():
        raise ValueError("found rows where home == away")

    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    if (out["home_goals"] < 0).any() or (out["away_goals"] < 0).any():
        raise ValueError("goal counts must be non-negative")

    out["neutral"] = out["neutral"].astype(bool)

    return out.sort_values("date", kind="stable").reset_index(drop=True)
