"""Tests for the fixtures store + queue drain."""


import pytest

from football_forecast.data.schema import OUTCOMES
from football_forecast.fixtures_queue import drain
from football_forecast.store import fixtures as fx


@pytest.fixture
def db(tmp_path):
    return tmp_path / "fixtures.sqlite"


def test_add_fixture_is_pending(db):
    fid = fx.add_fixture("2026-06-25", "FIFA World Cup", "final_tournament",
                         "Brazil", "Serbia", neutral=True, path=db)
    row = fx.get(fid, path=db)
    assert row["status"] == "pending"
    assert row["forecast"] is None
    assert fid in {f["fixture_id"] for f in fx.list_pending(db)}


def test_make_id_stable():
    assert fx.make_id("2026-06-25", "South Korea", "Cape Verde") == "20260625_SouthKorea_CapeVerde"


def test_add_with_result_marks_played(db):
    fid = fx.add_fixture("2026-06-20", "FIFA World Cup", "final_tournament",
                         "Spain", "Cape Verde", home_goals=0, away_goals=0, path=db)
    assert fx.get(fid, path=db)["status"] == "played"


def test_set_result_transitions(db):
    fid = fx.add_fixture("2026-06-25", "Friendly", "friendly", "A", "B", path=db)
    assert fx.set_result(fid, 2, 1, path=db)
    row = fx.get(fid, path=db)
    assert row["status"] == "played"
    assert row["home_goals"] == 2 and row["away_goals"] == 1


def test_add_fixture_upsert_keeps_forecast(db):
    fid = fx.add_fixture("2026-06-25", "Friendly", "friendly", "A", "B", path=db)
    fx.mark_forecast(fid, "dixon_coles", {"1x2": {"H": 0.5, "D": 0.3, "A": 0.2}}, path=db)
    # Re-adding the same fixture (e.g. fixing a typo in competition) must not wipe it.
    fx.add_fixture("2026-06-25", "Friendly Intl", "friendly", "A", "B", path=db)
    row = fx.get(fid, path=db)
    assert row["forecast"]["1x2"]["H"] == 0.5
    assert row["competition"] == "Friendly Intl"


class _StubGoalModel:
    """Minimal goal model: constant rates, exposes `rates` like a real one."""

    use_dc = False
    max_goals = 10

    def rates(self, fixture):
        return 1.6, 1.0


def test_drain_computes_pending(db):
    fx.add_fixture("2026-06-25", "FIFA World Cup", "final_tournament", "A", "B", path=db)
    fx.add_fixture("2026-06-26", "FIFA World Cup", "final_tournament", "C", "D", path=db)
    n = drain(_StubGoalModel(), "dixon_coles", path=db)
    assert n == 2
    assert fx.list_pending(db) == []  # queue drained
    row = fx.list_fixtures(path=db)[0]
    assert row["status"] == "forecast"
    assert set(row["forecast"]) == {"1x2", "scoreline", "over_under", "correct_score"}
    assert sum(row["forecast"]["1x2"][o] for o in OUTCOMES) == pytest.approx(1.0, abs=1e-4)


def test_drain_is_idempotent(db):
    fx.add_fixture("2026-06-25", "FIFA World Cup", "final_tournament", "A", "B", path=db)
    drain(_StubGoalModel(), "dixon_coles", path=db)
    assert drain(_StubGoalModel(), "dixon_coles", path=db) == 0  # nothing left pending


def test_list_pending_missing_db(tmp_path):
    assert fx.list_pending(tmp_path / "nope.sqlite") == []
