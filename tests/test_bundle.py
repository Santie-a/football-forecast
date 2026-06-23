"""Tests for the market bundle built from a scoreline matrix."""

import pytest

from football_forecast.forecast.bundle import build_markets
from football_forecast.forecast.scoreline import scoreline_matrix


def test_bundle_has_all_markets():
    b = build_markets(scoreline_matrix(1.6, 1.1))
    assert set(b) == {"1x2", "scoreline", "over_under", "correct_score"}


def test_1x2_consistent_with_markets():
    m = scoreline_matrix(1.8, 0.9)
    b = build_markets(m)
    assert sum(b["1x2"].values()) == pytest.approx(1.0, abs=1e-4)
    assert b["1x2"]["H"] > b["1x2"]["A"]


def test_scoreline_slice_shape():
    b = build_markets(scoreline_matrix(1.4, 1.2), display_goals=5)
    mat = b["scoreline"]["matrix"]
    assert b["scoreline"]["goals"] == 5
    assert len(mat) == 6 and len(mat[0]) == 6  # 0..5 inclusive


def test_over_under_lines_present_and_normalized():
    b = build_markets(scoreline_matrix(1.5, 1.5))
    for line in ("1.5", "2.5", "3.5"):
        ou = b["over_under"][line]
        assert ou["over"] + ou["under"] == pytest.approx(1.0, abs=1e-4)


def test_correct_score_top_n_sorted():
    b = build_markets(scoreline_matrix(1.4, 1.1), top_scores=5)
    vals = list(b["correct_score"].values())
    assert len(vals) == 5
    assert vals == sorted(vals, reverse=True)
