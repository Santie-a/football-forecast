"""Import fixtures into the fixtures store from any competition.

Two sources, one registration path:

- `from_raw_results`: a martj42-format results CSV, which carries both played
  matches (with scores) and **upcoming** ones (blank scores) that the normal
  ingest drops. Filter by competition and date.
- `from_csv`: a plain user fixtures CSV (columns: date, home, away, and optional
  competition, comp_type, neutral, home_goals, away_goals).

`register` then writes them to the store: played → with results, unplayed →
queued. If a fitted model is supplied, **played** matches also get a forecast from
it (pass a model fit *before* the matches to avoid leakage — e.g. pre-tournament).
Unplayed fixtures are left for `pipelines.process_queue` to forecast.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_forecast.data.schema import COMP_TYPES, Fixture
from football_forecast.data.sources.intl_results import classify_comp_type
from football_forecast.forecast.bundle import markets_for_model
from football_forecast.store import fixtures as fxstore

CANON_COLS = ("date", "competition", "comp_type", "home", "away", "neutral", "home_goals", "away_goals")


def from_raw_results(
    raw_path: str | Path,
    competition: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> pd.DataFrame:
    """Load fixtures (played + unplayed) from a raw results CSV, filtered."""
    raw = pd.read_csv(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    df = raw
    if competition:
        df = df[df["tournament"] == competition]
    if since:
        df = df[df["date"] >= pd.Timestamp(since)]
    if until:
        df = df[df["date"] <= pd.Timestamp(until)]
    return pd.DataFrame(
        {
            "date": df["date"],
            "competition": df["tournament"].astype(str),
            "comp_type": df["tournament"].astype(str).map(classify_comp_type),
            "home": df["home_team"].astype(str),
            "away": df["away_team"].astype(str),
            "neutral": df["neutral"].astype(bool),
            "home_goals": df["home_score"],
            "away_goals": df["away_score"],
        }
    ).sort_values("date")


def from_csv(csv_path: str | Path, default_comp_type: str = "friendly") -> pd.DataFrame:
    """Load fixtures from a generic CSV. Requires date/home/away; the rest default."""
    df = pd.read_csv(csv_path)
    missing = [c for c in ("date", "home", "away") if c not in df.columns]
    if missing:
        raise ValueError(f"fixtures CSV missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    out = pd.DataFrame(
        {
            "date": df["date"],
            "competition": df["competition"] if "competition" in df else "Unknown",
            "comp_type": df["comp_type"] if "comp_type" in df else default_comp_type,
            "home": df["home"].astype(str),
            "away": df["away"].astype(str),
            "neutral": df["neutral"].astype(bool) if "neutral" in df else False,
            "home_goals": df["home_goals"] if "home_goals" in df else pd.NA,
            "away_goals": df["away_goals"] if "away_goals" in df else pd.NA,
        }
    )
    bad = sorted(set(out["comp_type"].astype(str)) - COMP_TYPES)
    if bad:
        raise ValueError(f"unknown comp_type values in CSV: {bad} (allowed: {sorted(COMP_TYPES)})")
    return out.sort_values("date")


def register(
    fixtures_df: pd.DataFrame,
    path: str | Path = fxstore.DEFAULT_PATH,
    model=None,
    model_name: str = "dixon_coles",
) -> dict[str, int]:
    """Write fixtures to the store. Played→results, unplayed→queued. If `model`
    (a fitted forecaster) is given, played matches also get its forecast."""
    played = queued = 0
    for r in fixtures_df.itertuples(index=False):
        has_result = pd.notna(r.home_goals) and pd.notna(r.away_goals)
        fid = fxstore.add_fixture(
            date=str(pd.Timestamp(r.date).date()),
            competition=r.competition,
            comp_type=r.comp_type,
            home=r.home,
            away=r.away,
            neutral=bool(r.neutral),
            home_goals=int(r.home_goals) if has_result else None,
            away_goals=int(r.away_goals) if has_result else None,
            path=path,
        )
        if has_result:
            played += 1
            if model is not None:
                fixture = Fixture(r.home, r.away, pd.Timestamp(r.date).date(), r.competition, r.comp_type, bool(r.neutral))
                fxstore.mark_forecast(fid, model_name, markets_for_model(model, fixture), path)
        else:
            queued += 1
    return {"played": played, "queued": queued, "total": played + queued}
