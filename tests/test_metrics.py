"""Golden-number tests for the scoring rules. Values are hand-computed in the
docstrings so a regression is unambiguous."""

import math

import numpy as np
import pytest

from football_forecast.eval.metrics import brier, log_loss, mean_rps, rps


def test_rps_certain_correct_is_zero():
    assert rps([1.0, 0.0, 0.0], "H") == 0.0


def test_rps_certain_maximally_wrong_is_one():
    # Certain away win, true outcome home: cumulative diffs (0-1), (0-1) -> 2/2 = 1.
    assert rps([0.0, 0.0, 1.0], "H") == pytest.approx(1.0)


def test_rps_worked_example():
    # p=[.5,.3,.2], outcome H: cum p=[.5,.8], cum o=[1,1];
    # (.5-1)^2 + (.8-1)^2 = .25 + .04 = .29; /2 = .145
    assert rps([0.5, 0.3, 0.2], "H") == pytest.approx(0.145)


def test_rps_is_symmetric_under_mirror():
    # Mirroring H<->A on both probs and outcome must give the same RPS.
    assert rps([0.2, 0.3, 0.5], "A") == pytest.approx(rps([0.5, 0.3, 0.2], "H"))


def test_rps_accepts_mapping_and_sequence_equally():
    seq = [0.5, 0.3, 0.2]
    mapping = {"H": 0.5, "D": 0.3, "A": 0.2}
    assert rps(mapping, "H") == pytest.approx(rps(seq, "H"))


def test_log_loss_worked_example():
    assert log_loss([0.5, 0.3, 0.2], "H") == pytest.approx(-math.log(0.5))


def test_log_loss_clips_zero_probability():
    # No infinity: a 0 on the true outcome is clipped, not -inf.
    assert math.isfinite(log_loss([0.0, 0.5, 0.5], "H"))


def test_brier_worked_example():
    # (.5-1)^2 + .3^2 + .2^2 = .25 + .09 + .04 = .38
    assert brier([0.5, 0.3, 0.2], "H") == pytest.approx(0.38)


def test_mean_rps_averages():
    rows = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    # rps = 0 and 1 -> mean 0.5
    assert mean_rps(rows, ["H", "H"]) == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [[0.5, 0.5], [0.3, 0.3, 0.3, 0.1]])
def test_rps_rejects_wrong_length(bad):
    with pytest.raises(ValueError):
        rps(bad, "H")


def test_rps_rejects_unknown_outcome():
    with pytest.raises(ValueError):
        rps([0.4, 0.3, 0.3], "X")


def test_uniform_forecast_rps_is_between_extremes():
    # Sanity: the no-information forecast scores strictly between 0 and 1.
    val = rps([1 / 3, 1 / 3, 1 / 3], "H")
    assert 0.0 < val < 1.0
    assert val == pytest.approx(np.sum(np.array([1 / 3 - 1, 2 / 3 - 1]) ** 2) / 2)
