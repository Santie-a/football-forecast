"""Golden-number and structural tests for the keystone (scoreline -> markets).
Hand-computed values are in the comments so a regression is unambiguous."""

import math

import numpy as np
import pytest

from football_forecast.forecast.markets import (
    btts,
    correct_score,
    one_x_two,
    over_under,
)
from football_forecast.forecast.scoreline import dc_tau, scoreline_matrix


def test_zero_rates_put_all_mass_on_nil_nil():
    m = scoreline_matrix(0.0, 0.0)
    assert m[0, 0] == pytest.approx(1.0)
    assert one_x_two(m)["D"] == pytest.approx(1.0)


def test_independent_cell_matches_hand_computation():
    # lh=1.5, la=1.1. P(2-0) = pmf(2;1.5) * pmf(0;1.1)
    #   = (e^-1.5 * 1.5^2 / 2) * e^-1.1 = 0.250902 * 0.332871 = 0.083528
    m = scoreline_matrix(1.5, 1.1, max_goals=15)
    expected = (math.exp(-1.5) * 1.5**2 / 2) * math.exp(-1.1)
    assert m[2, 0] == pytest.approx(expected, abs=1e-4)
    assert correct_score(m)[(2, 0)] == pytest.approx(m[2, 0])


def test_matrix_normalized():
    assert scoreline_matrix(1.7, 1.2).sum() == pytest.approx(1.0)


def test_one_x_two_sums_to_one_and_symmetric():
    m = scoreline_matrix(1.3, 1.3)
    p = one_x_two(m)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["H"] == pytest.approx(p["A"])  # equal rates -> symmetric


def test_stronger_home_rate_favors_home():
    p = one_x_two(scoreline_matrix(2.2, 0.8))
    assert p["H"] > p["A"] and p["H"] > p["D"]


def test_dixon_coles_inflates_draws():
    # Negative rho should lift draw probability vs independent Poisson.
    lh = la = 1.2
    indep = one_x_two(scoreline_matrix(lh, la))["D"]
    dc = one_x_two(scoreline_matrix(lh, la, dc_rho=-0.1))["D"]
    assert dc > indep


def test_dc_tau_is_identity_outside_low_cells():
    assert dc_tau(3, 2, 1.4, 1.1, -0.1) == 1.0
    assert dc_tau(1, 1, 1.4, 1.1, -0.1) == pytest.approx(1.1)  # 1 - (-0.1)


def test_over_under_monotone_in_rates():
    low = over_under(scoreline_matrix(0.7, 0.6))["over"]
    high = over_under(scoreline_matrix(2.6, 2.4))["over"]
    assert high > low
    assert sum(over_under(scoreline_matrix(1.5, 1.5)).values()) == pytest.approx(1.0)


def test_btts_increases_with_rates():
    assert btts(scoreline_matrix(2.0, 2.0))["yes"] > btts(scoreline_matrix(0.5, 0.5))["yes"]


def test_correct_score_top_n():
    top3 = correct_score(scoreline_matrix(1.4, 1.1), top=3)
    assert len(top3) == 3
    vals = list(top3.values())
    assert vals == sorted(vals, reverse=True)
