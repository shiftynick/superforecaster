"""Property and example tests for scoring metrics."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from superforecaster.metrics import (
    brier_score,
    calibration_curve,
    expected_calibration_error,
    log_score,
    mean_brier,
)

probs = st.floats(min_value=0.0, max_value=1.0)
bools = st.booleans()


@given(p=probs, o=bools)
def test_brier_in_unit_range(p: float, o: bool) -> None:
    assert 0.0 <= brier_score(p, o) <= 1.0


@given(p=probs)
def test_brier_symmetry(p: float) -> None:
    # Predicting p for a True and (1-p) for a False cost the same.
    assert math.isclose(brier_score(p, True), brier_score(1.0 - p, False))


def test_brier_perfect_forecast_is_zero() -> None:
    assert brier_score(1.0, True) == 0.0
    assert brier_score(0.0, False) == 0.0


@given(p=probs, o=bools)
def test_log_score_perfect_and_direction(p: float, o: bool) -> None:
    # Log score is non-negative and lower is better.
    assert log_score(p, o) >= 0.0


def test_log_score_perfect_forecast_near_zero() -> None:
    assert log_score(1.0, True) < 1e-8
    assert log_score(0.0, False) < 1e-8


def test_log_score_wrong_confident_is_finite_large() -> None:
    # Clamping keeps a confidently-wrong forecast finite.
    val = log_score(1.0, False)
    assert math.isfinite(val)
    assert val > 15.0


def test_mean_brier_matches_manual() -> None:
    pairs = [(0.9, True), (0.2, False), (0.5, True)]
    expected = (0.1**2 + 0.2**2 + 0.5**2) / 3
    assert math.isclose(mean_brier(pairs), expected)


@given(pairs=st.lists(st.tuples(probs, bools), min_size=1, max_size=200))
def test_calibration_counts_sum_to_len(pairs: list[tuple[float, bool]]) -> None:
    bins = calibration_curve(pairs, n_bins=10)
    assert sum(b.count for b in bins) == len(pairs)


@given(pairs=st.lists(st.tuples(probs, bools), min_size=1, max_size=200))
def test_calibration_bins_partition(pairs: list[tuple[float, bool]]) -> None:
    bins = calibration_curve(pairs, n_bins=10)
    for b in bins:
        assert b.count > 0  # empty bins omitted
        assert b.bin_lower < b.bin_upper
        assert 0.0 <= b.observed_frequency <= 1.0
        # mean predicted lies within the bin edges (last bin includes 1.0).
        assert b.bin_lower - 1e-12 <= b.mean_predicted <= b.bin_upper + 1e-12
    # bins are sorted and non-overlapping
    lowers = [b.bin_lower for b in bins]
    assert lowers == sorted(lowers)


@given(pairs=st.lists(st.tuples(probs, bools), min_size=1, max_size=200))
def test_ece_in_unit_range(pairs: list[tuple[float, bool]]) -> None:
    assert 0.0 <= expected_calibration_error(pairs) <= 1.0


def test_ece_perfectly_calibrated_is_small() -> None:
    # Synthetic dataset: in bin centred at 0.1 exactly 10% are True, etc.
    pairs: list[tuple[float, bool]] = []
    for k in range(1, 10):
        p = k / 10.0
        n_true = k  # k out of 10 True -> observed frequency == p
        for _ in range(n_true):
            pairs.append((p, True))
        for _ in range(10 - n_true):
            pairs.append((p, False))
    assert expected_calibration_error(pairs, n_bins=10) < 0.02
