"""The fixtures store + forecast queue (standard library only).

A *writable* SQLite DB, separate from the read-only forecasts store, so the Pi
can record fixtures/results without colliding with the PC→Pi sync. Each row is a
fixture and carries its own computed forecast bundle.

**Queue semantics.** A row whose `forecast` is NULL is *pending* — it has been
requested but not yet computed. Draining the queue (cheap Dixon–Coles inference
on the Pi, or a full run on the PC) fills `forecast` and is what `mark_forecast`
records. Adding a *result* never needs heavy compute here; folding results back
into model training is a separate PC job.

Stdlib-only on purpose: both `pipelines/` (PC) and `app/` (Pi) import this without
pulling pandas or any modelling dependency.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = "artifacts/fixtures/fixtures.sqlite"

_CREATE = """
CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id     TEXT PRIMARY KEY,
    date           TEXT,
    competition    TEXT,
    comp_type      TEXT,
    home           TEXT,
    away           TEXT,
    neutral        INTEGER,
    home_goals     INTEGER,
    away_goals     INTEGER,
    forecast_model TEXT,
    forecast       TEXT,          -- JSON bundle {market: payload}, NULL = queued
    requested_at   TEXT,
    forecast_at    TEXT,
    result_at      TEXT
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute(_CREATE)
    return con


def make_id(date: str, home: str, away: str) -> str:
    d = str(date)[:10].replace("-", "")
    return f"{d}_{home.replace(' ', '')}_{away.replace(' ', '')}"


def _to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["neutral"] = bool(d["neutral"])
    d["forecast"] = json.loads(d["forecast"]) if d["forecast"] else None
    if d["home_goals"] is not None and d["away_goals"] is not None:
        d["status"] = "played"
    elif d["forecast"] is not None:
        d["status"] = "forecast"
    else:
        d["status"] = "pending"
    return d


def add_fixture(
    date: str,
    competition: str,
    comp_type: str,
    home: str,
    away: str,
    neutral: bool = False,
    *,
    home_goals: int | None = None,
    away_goals: int | None = None,
    fixture_id: str | None = None,
    path: str | Path = DEFAULT_PATH,
) -> str:
    """Insert (or update the descriptive fields of) a fixture. Leaves any existing
    forecast/result intact. Returns the fixture_id. New fixtures are queued."""
    fid = fixture_id or make_id(date, home, away)
    has_result = home_goals is not None and away_goals is not None
    con = _connect(path)
    try:
        con.execute(
            """
            INSERT INTO fixtures
                (fixture_id, date, competition, comp_type, home, away, neutral,
                 home_goals, away_goals, requested_at, result_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(fixture_id) DO UPDATE SET
                date=excluded.date, competition=excluded.competition,
                comp_type=excluded.comp_type, home=excluded.home,
                away=excluded.away, neutral=excluded.neutral
            """,
            (
                fid, str(date)[:10], competition, comp_type, home, away,
                int(bool(neutral)), home_goals, away_goals, _now(),
                _now() if has_result else None,
            ),
        )
        con.commit()
    finally:
        con.close()
    return fid


def set_result(
    fixture_id: str, home_goals: int, away_goals: int, path: str | Path = DEFAULT_PATH
) -> bool:
    con = _connect(path)
    try:
        cur = con.execute(
            "UPDATE fixtures SET home_goals=?, away_goals=?, result_at=? WHERE fixture_id=?",
            (int(home_goals), int(away_goals), _now(), fixture_id),
        )
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()


def mark_forecast(
    fixture_id: str, model: str, bundle: dict, path: str | Path = DEFAULT_PATH
) -> None:
    con = _connect(path)
    try:
        con.execute(
            "UPDATE fixtures SET forecast_model=?, forecast=?, forecast_at=? WHERE fixture_id=?",
            (model, json.dumps(bundle), _now(), fixture_id),
        )
        con.commit()
    finally:
        con.close()


def list_pending(path: str | Path = DEFAULT_PATH) -> list[dict]:
    """The queue: fixtures with no forecast yet."""
    if not Path(path).exists():
        return []
    con = _connect(path)
    try:
        rows = con.execute(
            "SELECT * FROM fixtures WHERE forecast IS NULL ORDER BY date, home"
        )
        return [_to_dict(r) for r in rows]
    finally:
        con.close()


def list_fixtures(
    competition: str | None = None, path: str | Path = DEFAULT_PATH
) -> list[dict]:
    if not Path(path).exists():
        return []
    con = _connect(path)
    try:
        if competition:
            rows = con.execute(
                "SELECT * FROM fixtures WHERE competition=? ORDER BY date, home",
                (competition,),
            )
        else:
            rows = con.execute("SELECT * FROM fixtures ORDER BY date, home")
        return [_to_dict(r) for r in rows]
    finally:
        con.close()


def delete(fixture_id: str, path: str | Path = DEFAULT_PATH) -> bool:
    if not Path(path).exists():
        return False
    con = _connect(path)
    try:
        cur = con.execute("DELETE FROM fixtures WHERE fixture_id=?", (fixture_id,))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()


def delete_after(date: str, path: str | Path = DEFAULT_PATH, inclusive: bool = False) -> int:
    """Delete fixtures dated strictly after `date` (or on/after it if inclusive).
    Returns the number deleted. Useful for clearing a manually-loaded set."""
    if not Path(path).exists():
        return 0
    op = ">=" if inclusive else ">"
    con = _connect(path)
    try:
        cur = con.execute(f"DELETE FROM fixtures WHERE date {op} ?", (str(date)[:10],))
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def get(fixture_id: str, path: str | Path = DEFAULT_PATH) -> dict | None:
    if not Path(path).exists():
        return None
    con = _connect(path)
    try:
        row = con.execute(
            "SELECT * FROM fixtures WHERE fixture_id=?", (fixture_id,)
        ).fetchone()
        return _to_dict(row) if row else None
    finally:
        con.close()
