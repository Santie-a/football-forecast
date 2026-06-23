"""League/club match source (goals + fouls/cards/corners + bookmaker odds).

football-data.co.uk is the canonical source for these per-match stats, but it is
not reachable from this environment; we use a maintained GitHub mirror that
consolidates 27 countries / 42 leagues 2000–2025 into one CSV with the same
fields (xgabora/Club-Football-Match-Data-2000-2025). It carries goals, shots,
fouls, corners, yellow/red cards and 1X2 + over/under odds — but **no referee
column** (so M06 referee effects stay deferred until a referee-bearing source is
reachable).

Normalized into the canonical schema (`comp_type="league"`, `neutral=False`) plus
the optional count and odds columns documented in docs/data.md.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from football_forecast.data.schema import validate

RAW_URL = (
    "https://raw.githubusercontent.com/xgabora/"
    "Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
)

# football-data.co.uk division codes → readable competition names (common ones).
DIVISION_NAMES = {
    "E0": "Premier League", "E1": "Championship", "E2": "League One", "E3": "League Two",
    "SC0": "Scottish Premiership", "D1": "Bundesliga", "D2": "Bundesliga 2",
    "I1": "Serie A", "I2": "Serie B", "SP1": "La Liga", "SP2": "La Liga 2",
    "F1": "Ligue 1", "F2": "Ligue 2", "N1": "Eredivisie", "P1": "Primeira Liga",
    "B1": "Belgian Pro League", "T1": "Super Lig", "G1": "Super League Greece",
}

# Raw → canonical extra columns (optional count + odds fields).
_EXTRA = {
    "HomeFouls": "home_fouls", "AwayFouls": "away_fouls",
    "HomeCorners": "home_corners", "AwayCorners": "away_corners",
    "HomeYellow": "home_yellow", "AwayYellow": "away_yellow",
    "HomeRed": "home_red", "AwayRed": "away_red",
    "OddHome": "odds_h", "OddDraw": "odds_d", "OddAway": "odds_a",
}


def fetch_raw(dest: str | Path = "data/raw/club_matches.csv", url: str = RAW_URL) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def _season(d: pd.Timestamp) -> str:
    y = d.year if d.month >= 7 else d.year - 1
    return f"{y}/{y + 1}"


def load(
    path: str | Path = "data/raw/club_matches.csv",
    division: str = "E0",
    since: str | None = None,
) -> pd.DataFrame:
    """Load one division, normalized into the canonical schema + extra columns."""
    raw = pd.read_csv(path, low_memory=False)
    d = raw[raw["Division"] == division].copy()
    if d.empty:
        raise ValueError(f"no rows for division {division!r}")
    d["MatchDate"] = pd.to_datetime(d["MatchDate"])
    if since:
        d = d[d["MatchDate"] >= pd.Timestamp(since)]
    d = d.dropna(subset=["FTHome", "FTAway"])

    comp = DIVISION_NAMES.get(division, division)
    out = pd.DataFrame(
        {
            "date": d["MatchDate"],
            "competition": comp,
            "comp_type": "league",
            "home": d["HomeTeam"].astype(str),
            "away": d["AwayTeam"].astype(str),
            "home_goals": d["FTHome"],
            "away_goals": d["FTAway"],
            "neutral": False,
            "season": d["MatchDate"].map(_season),
        }
    )
    for raw_col, canon in _EXTRA.items():
        if raw_col in d.columns:
            out[canon] = d[raw_col].to_numpy()

    base = (
        division + "_" + out["date"].dt.strftime("%Y%m%d")
        + "_" + out["home"].str.replace(r"\s+", "", regex=True)
        + "_" + out["away"].str.replace(r"\s+", "", regex=True)
    )
    dup = base.groupby(base).cumcount()
    out["match_id"] = base.where(dup == 0, base + "_" + dup.astype(str))

    return validate(out)
