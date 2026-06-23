"""The forecast store — the batch contract the PC writes and the Pi app reads.

A single SQLite file (docs/data.md → artifact-store schema). Writes are
idempotent per model+market: re-running a forecast replaces that model's rows
rather than duplicating them. The app opens this read-only.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_PATH = "artifacts/forecasts/forecasts.sqlite"

_COLUMNS = [
    "match_id",
    "date",
    "home",
    "away",
    "competition",
    "model",
    "market",
    "payload",
    "generated_at",
]

_CREATE = """
CREATE TABLE IF NOT EXISTS forecasts (
    match_id     TEXT,
    date         TEXT,
    home         TEXT,
    away         TEXT,
    competition  TEXT,
    model        TEXT,
    market       TEXT,
    payload      TEXT,
    generated_at TEXT
)
"""


def _connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute(_CREATE)
    return con


def write_forecasts(df: pd.DataFrame, path: str | Path = DEFAULT_PATH) -> int:
    """Write forecast rows, replacing any existing rows for the same (model,
    market) pairs. `payload` may be a dict (JSON-encoded here) or a JSON string.
    Returns the number of rows written."""
    if df.empty:
        return 0
    missing = [c for c in _COLUMNS if c not in df.columns and c != "generated_at"]
    if missing:
        raise ValueError(f"forecast frame missing columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["payload"] = out["payload"].map(
        lambda p: p if isinstance(p, str) else json.dumps(p)
    )
    if "generated_at" not in out.columns:
        out["generated_at"] = datetime.now(timezone.utc).isoformat()
    out = out[_COLUMNS]

    con = _connect(path)
    try:
        for model, market in out[["model", "market"]].drop_duplicates().itertuples(index=False):
            con.execute(
                "DELETE FROM forecasts WHERE model = ? AND market = ?", (model, market)
            )
        out.to_sql("forecasts", con, if_exists="append", index=False)
        con.commit()
    finally:
        con.close()
    return len(out)


def read_forecasts(
    path: str | Path = DEFAULT_PATH,
    model: str | None = None,
    market: str | None = None,
    parse_payload: bool = True,
) -> pd.DataFrame:
    """Read forecasts, optionally filtered by model/market. Read-only."""
    con = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        query = "SELECT * FROM forecasts"
        clauses, params = [], []
        if model is not None:
            clauses.append("model = ?")
            params.append(model)
        if market is not None:
            clauses.append("market = ?")
            params.append(market)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        df = pd.read_sql_query(query, con, params=params)
    finally:
        con.close()
    if parse_payload and not df.empty:
        df["payload"] = df["payload"].map(json.loads)
    return df
