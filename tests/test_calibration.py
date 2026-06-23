"""Tests for calibration analysis."""

import pytest

from football_forecast.eval.calibration import (
    expected_calibration_error,
    reliability_table,
)


def test_perfectly_confident_correct_is_calibrated():
    # Always 100% on the true outcome -> zero calibration error.
    rows = [{"H": 1.0, "D": 0.0, "A": 0.0}, {"H": 0.0, "D": 0.0, "A": 1.0}]
    outcomes = ["H", "A"]
    assert expected_calibration_error(rows, outcomes) == pytest.approx(0.0)


def test_overconfident_is_miscalibrated():
    # Claim H=0.9 every time, but H happens only ~30% -> large ECE.
    rows = [{"H": 0.9, "D": 0.05, "A": 0.05}] * 100
    outcomes = (["H"] * 30) + (["D"] * 40) + (["A"] * 30)
    assert expected_calibration_error(rows, outcomes) > 0.1


def test_reliability_table_shape_and_counts():
    rows = [{"H": 0.5, "D": 0.3, "A": 0.2}] * 10
    outcomes = ["H"] * 10
    tbl = reliability_table(rows, outcomes, n_bins=10)
    assert len(tbl) == 10
    # 10 matches x 3 classes = 30 pooled entries.
    assert tbl["count"].sum() == 30
