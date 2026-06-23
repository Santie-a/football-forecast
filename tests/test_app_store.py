"""Smoke tests for the app's read-only data layer (stdlib sqlite, no library import)."""

import json
import sqlite3

import pytest

from app import store


@pytest.fixture
def populated_store(tmp_path, monkeypatch):
    path = tmp_path / "forecasts.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE forecasts (match_id TEXT, date TEXT, home TEXT, away TEXT,"
        " competition TEXT, model TEXT, market TEXT, payload TEXT, generated_at TEXT)"
    )
    con.executemany(
        "INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?)",
        [
            ("m1", "2026-06-21", "Uruguay", "Cape Verde", "Friendly", "elo", "1x2",
             json.dumps({"H": 0.68, "D": 0.20, "A": 0.12}), "2026-06-22T00:00:00Z"),
            ("m2", "2026-06-20", "Brazil", "Peru", "Copa", "elo", "1x2",
             json.dumps({"H": 0.55, "D": 0.25, "A": 0.20}), "2026-06-22T00:00:00Z"),
        ],
    )
    con.commit()
    con.close()
    monkeypatch.setenv("FORECAST_STORE", str(path))
    return path


def test_available_and_models(populated_store):
    assert store.available()
    assert store.list_models() == ["elo"]
    assert "Copa" in store.list_competitions()


def test_list_and_filter(populated_store):
    assert len(store.list_forecasts()) == 2
    only_brazil = store.list_forecasts(q="Brazil")
    assert len(only_brazil) == 1 and only_brazil[0]["home"] == "Brazil"


def test_payload_parsed_and_top(populated_store):
    f = store.list_forecasts(q="Uruguay")[0]
    assert f["payload"]["H"] == 0.68
    assert f["top"] == "H"


def test_get_match(populated_store):
    rows = store.get_match("m1")
    assert len(rows) == 1 and rows[0]["away"] == "Cape Verde"


def test_missing_store_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setenv("FORECAST_STORE", str(tmp_path / "nope.sqlite"))
    assert not store.available()
    assert store.list_forecasts() == []


def test_match_models_groups_markets(tmp_path, monkeypatch):
    path = tmp_path / "f.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE forecasts (match_id TEXT, date TEXT, home TEXT, away TEXT,"
        " competition TEXT, model TEXT, market TEXT, payload TEXT, generated_at TEXT)"
    )
    common = ("m1", "2026-06-21", "Uruguay", "Cape Verde", "Friendly")
    con.executemany(
        "INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (*common, "dixon_coles", "1x2", json.dumps({"H": 0.68, "D": 0.2, "A": 0.12}), "t"),
            (*common, "dixon_coles", "scoreline", json.dumps({"goals": 1, "matrix": [[0.1, 0.1], [0.1, 0.1]]}), "t"),
            (*common, "elo", "1x2", json.dumps({"H": 0.6, "D": 0.25, "A": 0.15}), "t"),
        ],
    )
    con.commit()
    con.close()
    monkeypatch.setenv("FORECAST_STORE", str(path))

    grouped = store.match_models("m1")
    assert grouped["home"] == "Uruguay"
    models = {m["model"]: m for m in grouped["models"]}
    assert set(models) == {"dixon_coles", "elo"}
    assert set(models["dixon_coles"]["markets"]) == {"1x2", "scoreline"}
    assert models["elo"]["markets"].keys() == {"1x2"}
    assert store.match_models("nope") is None
