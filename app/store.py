"""Read-only access to the forecast store, using only the standard library.

The app deliberately does NOT import football_forecast (which pulls pandas/numpy)
— it keeps the Pi image small. It speaks the same SQLite schema the PC writes
(docs/data.md). Opens the DB read-only; never writes.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

DEFAULT_PATH = "artifacts/forecasts/forecasts.sqlite"


def store_path() -> Path:
    return Path(os.environ.get("FORECAST_STORE", DEFAULT_PATH))


def available() -> bool:
    return store_path().exists()


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{store_path()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["payload"] = json.loads(d["payload"])
    except (TypeError, json.JSONDecodeError):
        d["payload"] = {}
    # `top` is the favoured 1X2 outcome — only meaningful for that market, whose
    # payload is a flat {label: prob}. Other markets nest dicts/lists.
    if d.get("market") == "1x2" and isinstance(d["payload"], dict) and d["payload"]:
        d["top"] = max(d["payload"], key=d["payload"].get)
    else:
        d["top"] = None
    return d


def list_models() -> list[str]:
    if not available():
        return []
    with _connect() as con:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT model FROM forecasts ORDER BY model"
        )]


def list_competitions() -> list[str]:
    if not available():
        return []
    with _connect() as con:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT competition FROM forecasts ORDER BY competition"
        )]


def list_forecasts(
    model: str | None = None,
    competition: str | None = None,
    q: str | None = None,
    market: str = "1x2",
    limit: int = 200,
) -> list[dict]:
    if not available():
        return []
    clauses = ["market = ?"]
    params: list = [market]
    if model:
        clauses.append("model = ?")
        params.append(model)
    if competition:
        clauses.append("competition = ?")
        params.append(competition)
    if q:
        clauses.append("(home LIKE ? OR away LIKE ?)")
        params += [f"%{q}%", f"%{q}%"]
    sql = (
        "SELECT * FROM forecasts WHERE "
        + " AND ".join(clauses)
        + " ORDER BY date DESC, home LIMIT ?"
    )
    params.append(limit)
    with _connect() as con:
        return [_row_to_dict(r) for r in con.execute(sql, params)]


# Preference order when one match has several model forecasts (for the headline bar).
MODEL_PREF = ("dixon_coles", "maher", "elo")


def _primary_model(models: dict) -> str | None:
    for m in MODEL_PREF:
        if m in models:
            return m
    return next(iter(models), None)


def list_matches(
    fixture_rows: list[dict] | None = None,
    model: str | None = None,
    competition: str | None = None,
    q: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """One row per match, compacting all model forecasts together and merging the
    fixtures store (so user-added fixtures appear on the main page). `fixture_rows`
    comes from the fixtures DAL (passed in to keep this module path-agnostic)."""
    cards: dict[str, dict] = {}

    if available():
        with _connect() as con:
            for r in con.execute(
                "SELECT match_id,date,home,away,competition,model,payload "
                "FROM forecasts WHERE market='1x2'"
            ):
                mid = r["match_id"]
                c = cards.get(mid)
                if c is None:
                    c = cards[mid] = {
                        "match_id": mid, "date": r["date"], "home": r["home"],
                        "away": r["away"], "competition": r["competition"],
                        "models": {}, "url": f"/match/{mid}", "result": None, "source": "store",
                    }
                try:
                    c["models"][r["model"]] = json.loads(r["payload"])
                except (TypeError, json.JSONDecodeError):
                    pass

    for f in fixture_rows or []:
        mid = f["fixture_id"]
        models: dict = {}
        fc = f.get("forecast")
        if isinstance(fc, dict) and "1x2" in fc:
            models[f.get("forecast_model") or "model"] = fc["1x2"]
        # Fixtures take precedence on id collision (they carry status/result).
        cards[mid] = {
            "match_id": mid, "date": f["date"], "home": f["home"], "away": f["away"],
            "competition": f["competition"], "models": models, "url": f"/fixture/{mid}",
            "result": (f["home_goals"], f["away_goals"]) if f["status"] == "played" else None,
            "source": "fixture",
        }

    ql = (q or "").lower()
    out = []
    for c in cards.values():
        if model and model not in c["models"]:
            continue
        if competition and c["competition"] != competition:
            continue
        if ql and ql not in c["home"].lower() and ql not in c["away"].lower():
            continue
        pm = _primary_model(c["models"])
        c["primary_model"] = pm
        c["primary"] = c["models"].get(pm) if pm else None
        c["top"] = max(c["primary"], key=c["primary"].get) if c["primary"] else None
        c["model_names"] = sorted(c["models"])
        out.append(c)
    out.sort(key=lambda c: c["date"], reverse=True)
    return out[:limit]


def match_options(fixture_rows: list[dict] | None = None) -> tuple[list[str], list[str]]:
    """Distinct models and competitions across both stores, for the filter UI."""
    models, comps = set(), set()
    if available():
        with _connect() as con:
            models.update(r[0] for r in con.execute("SELECT DISTINCT model FROM forecasts"))
            comps.update(r[0] for r in con.execute("SELECT DISTINCT competition FROM forecasts"))
    for f in fixture_rows or []:
        if f.get("forecast_model"):
            models.add(f["forecast_model"])
        comps.add(f["competition"])
    return sorted(models), sorted(comps)


def get_match(match_id: str) -> list[dict]:
    """All stored forecasts (any model/market) for one match."""
    if not available():
        return []
    with _connect() as con:
        rows = con.execute(
            "SELECT * FROM forecasts WHERE match_id = ? ORDER BY model, market",
            (match_id,),
        )
        return [_row_to_dict(r) for r in rows]


def match_models(match_id: str) -> dict | None:
    """Grouped view for the match page: fixture meta + one entry per model, each
    holding its markets {market: payload}."""
    rows = get_match(match_id)
    if not rows:
        return None
    first = rows[0]
    models: dict[str, dict] = {}
    for r in rows:
        m = models.setdefault(
            r["model"], {"model": r["model"], "generated_at": r["generated_at"], "markets": {}}
        )
        m["markets"][r["market"]] = r["payload"]
    return {
        "match_id": match_id,
        "home": first["home"],
        "away": first["away"],
        "competition": first["competition"],
        "date": first["date"],
        "models": list(models.values()),
    }
