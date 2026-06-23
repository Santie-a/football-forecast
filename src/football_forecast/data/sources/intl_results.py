"""National-team results source (scorelines only; no fouls/cards/corners).

Uses the public, auth-free GitHub mirror of the "International football results
from 1872 to present" dataset (martj42), so ingestion is reproducible without a
Kaggle account. Raw columns:

    date, home_team, away_team, home_score, away_score, tournament, city,
    country, neutral

We normalize these into the canonical schema (docs/implementation-plan.md §1.1)
and derive the M01 `comp_type` feature from the tournament name.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from football_forecast.data.schema import validate

RAW_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def classify_comp_type(tournament: str) -> str:
    """Map a tournament name to a canonical comp_type bucket.

    Coarse but defensible for Phase 1 (the buckets carry different K-factor
    importance in Elo). Refinement to a fuller competition taxonomy is logged as
    a future step — see docs/decisions.md (M01).
    """
    t = tournament.strip().lower()
    if t == "friendly":
        return "friendly"
    if "qualification" in t or "qualifier" in t:
        return "qualifier"
    if "fifa world cup" in t:
        return "final_tournament"
    # All other tournaments (continental cups, Nations League, regional cups) →
    # treated as continental-tier competitive matches for now.
    return "continental"


def fetch_raw(dest: str | Path = "data/raw/intl_results.csv", url: str = RAW_URL) -> Path:
    """Download the raw CSV to `dest` (data/raw is immutable once fetched)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def load(path: str | Path = "data/raw/intl_results.csv") -> pd.DataFrame:
    """Load raw results and normalize into the validated canonical schema."""
    raw = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"]),
            "competition": raw["tournament"].astype(str),
            "comp_type": raw["tournament"].astype(str).map(classify_comp_type),
            "home": raw["home_team"].astype(str),
            "away": raw["away_team"].astype(str),
            "home_goals": raw["home_score"],
            "away_goals": raw["away_score"],
            "neutral": raw["neutral"].astype(bool),
        }
    )
    # Drop rows with missing scores (a handful of future/abandoned fixtures).
    out = out.dropna(subset=["home_goals", "away_goals"]).reset_index(drop=True)

    # Stable, source-independent match_id: date + teams, de-duplicated.
    base = (
        out["date"].dt.strftime("%Y%m%d")
        + "_" + out["home"].str.replace(r"\s+", "", regex=True)
        + "_" + out["away"].str.replace(r"\s+", "", regex=True)
    )
    dup = base.groupby(base).cumcount()
    out["match_id"] = base.where(dup == 0, base + "_" + dup.astype(str))

    return validate(out)
