"""FastAPI app: fixtures list + match detail, server-rendered with HTMX for the
live filter. Read-only over the forecast store."""

from __future__ import annotations

from pathlib import Path

import os

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import store
from football_forecast.store import fixtures as fxstore  # stdlib-only module

FIXTURES_PATH = os.environ.get("FIXTURES_STORE", fxstore.DEFAULT_PATH)
# Mirrors football_forecast.data.schema.COMP_TYPES; inlined so the app's import
# graph stays free of pandas (architecture decision 003).
COMP_TYPES = ("friendly", "qualifier", "final_tournament", "league", "continental")

BASE = Path(__file__).parent
app = FastAPI(title="Football Forecast")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))


def _pct(x: float) -> float:
    return round(float(x) * 100, 1)


def _flat_max(nested) -> float:
    return max((v for row in nested for v in row), default=0.0)


templates.env.filters["pct"] = _pct
templates.env.filters["flat_max"] = _flat_max


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "store": str(store.store_path()), "available": store.available()}


def _matches(model: str, competition: str, q: str) -> list:
    rows = fxstore.list_fixtures(path=FIXTURES_PATH)
    return store.list_matches(rows, model or None, competition or None, q or None)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, model: str = "", competition: str = "", q: str = ""):
    fixture_rows = fxstore.list_fixtures(path=FIXTURES_PATH)
    models, comps = store.match_options(fixture_rows)
    ctx = {
        "models": models,
        "competitions": comps,
        "fixtures": store.list_matches(fixture_rows, model or None, competition or None, q or None),
        "sel_model": model,
        "sel_comp": competition,
        "q": q,
        "available": store.available() or bool(fixture_rows),
    }
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/fragments/fixtures", response_class=HTMLResponse)
def fixtures_fragment(
    request: Request, model: str = "", competition: str = "", q: str = ""
):
    return templates.TemplateResponse(
        request, "_fixtures.html", {"fixtures": _matches(model, competition, q)}
    )


@app.get("/match/{match_id}", response_class=HTMLResponse)
def match_detail(request: Request, match_id: str):
    ctx = {"match_id": match_id, "match": store.match_models(match_id)}
    return templates.TemplateResponse(request, "match.html", ctx)


# --- fixtures + queue ----------------------------------------------------

@app.get("/fixtures", response_class=HTMLResponse)
def fixtures_page(request: Request):
    rows = fxstore.list_fixtures(path=FIXTURES_PATH)
    pending = sum(1 for f in rows if f["status"] == "pending")
    ctx = {
        "fixtures": rows,
        "pending": pending,
        "comp_types": sorted(COMP_TYPES),  # noqa: keep local list in sync with schema
    }
    return templates.TemplateResponse(request, "fixtures.html", ctx)


@app.post("/fixtures/add")
def fixtures_add(
    date: str = Form(...),
    home: str = Form(...),
    away: str = Form(...),
    competition: str = Form("FIFA World Cup"),
    comp_type: str = Form("final_tournament"),
    neutral: bool = Form(False),
):
    fxstore.add_fixture(
        date=date, competition=competition, comp_type=comp_type,
        home=home.strip(), away=away.strip(), neutral=neutral, path=FIXTURES_PATH,
    )
    return RedirectResponse("/fixtures", status_code=303)


@app.post("/fixtures/{fixture_id}/result")
def fixtures_result(
    fixture_id: str, home_goals: int = Form(...), away_goals: int = Form(...)
):
    fxstore.set_result(fixture_id, home_goals, away_goals, path=FIXTURES_PATH)
    return RedirectResponse("/fixtures", status_code=303)


@app.get("/fixture/{fixture_id}", response_class=HTMLResponse)
def fixture_detail(request: Request, fixture_id: str):
    f = fxstore.get(fixture_id, path=FIXTURES_PATH)
    match = None
    if f:
        models = []
        if f["forecast"]:
            models = [{
                "model": f["forecast_model"] or "model",
                "markets": f["forecast"],
                "generated_at": f["forecast_at"],
            }]
        match = {
            "match_id": fixture_id, "home": f["home"], "away": f["away"],
            "competition": f["competition"], "date": f["date"],
            "result": (f["home_goals"], f["away_goals"]) if f["status"] == "played" else None,
            "models": models,
        }
    return templates.TemplateResponse(request, "match.html", {"match_id": fixture_id, "match": match})
