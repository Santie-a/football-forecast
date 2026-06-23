"""FastAPI app: fixtures list + match detail, server-rendered with HTMX for the
live filter. Read-only over the forecast store."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import store

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


@app.get("/", response_class=HTMLResponse)
def index(request: Request, model: str = "", competition: str = "", q: str = ""):
    ctx = {
        "models": store.list_models(),
        "competitions": store.list_competitions(),
        "fixtures": store.list_forecasts(model or None, competition or None, q or None),
        "sel_model": model,
        "sel_comp": competition,
        "q": q,
        "available": store.available(),
    }
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/fragments/fixtures", response_class=HTMLResponse)
def fixtures_fragment(
    request: Request, model: str = "", competition: str = "", q: str = ""
):
    ctx = {
        "fixtures": store.list_forecasts(model or None, competition or None, q or None),
    }
    return templates.TemplateResponse(request, "_fixtures.html", ctx)


@app.get("/match/{match_id}", response_class=HTMLResponse)
def match_detail(request: Request, match_id: str):
    ctx = {"match_id": match_id, "match": store.match_models(match_id)}
    return templates.TemplateResponse(request, "match.html", ctx)
