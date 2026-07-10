"""Property and example tests for aggregation and recalibration."""

from __future__ import annotations

import math
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from superforecaster.aggregation import (
    PlattParams,
    agreement_scaled_coefficient,
    extremize,
    from_log_odds,
    platt_apply,
    platt_fit,
    pool_log_odds,
    population_std,
    to_log_odds,
)
from superforecaster.metrics import mean_brier

# Probabilities kept away from the exact edges where clamping loses precision.
safe_probs = st.floats(min_value=1e-6, max_value=1.0 - 1e-6)


@given(p=safe_probs)
def test_log_odds_round_trip(p: float) -> None:
    assert math.isclose(from_log_odds(to_log_odds(p)), p, rel_tol=1e-6, abs_tol=1e-9)


@given(p=safe_probs)
def test_pool_identical_returns_same(p: float) -> None:
    assert math.isclose(pool_log_odds([p, p, p]), p, rel_tol=1e-6, abs_tol=1e-9)


@given(ps=st.lists(safe_probs, min_size=1, max_size=8))
def test_pool_permutation_invariant(ps: list[float]) -> None:
    shuffled = list(reversed(ps))
    assert math.isclose(pool_log_odds(ps), pool_log_odds(shuffled), rel_tol=1e-9, abs_tol=1e-12)


@given(ps=st.lists(safe_probs, min_size=1, max_size=8))
def test_pool_within_min_max(ps: list[float]) -> None:
    pooled = pool_log_odds(ps)
    assert min(ps) - 1e-9 <= pooled <= max(ps) + 1e-9


def test_pool_weight_normalisation() -> None:
    # Scaling all weights leaves the result unchanged.
    ps = [0.2, 0.8, 0.6]
    a = pool_log_odds(ps, [1.0, 2.0, 3.0])
    b = pool_log_odds(ps, [10.0, 20.0, 30.0])
    assert math.isclose(a, b, rel_tol=1e-12)


def test_pool_errors() -> None:
    with pytest.raises(ValueError):
        pool_log_odds([])
    with pytest.raises(ValueError):
        pool_log_odds([0.5, 0.6], [1.0])
    with pytest.raises(ValueError):
        pool_log_odds([0.5, 0.6], [0.0, 0.0])


@given(p=safe_probs)
def test_extremize_identity(p: float) -> None:
    assert math.isclose(extremize(p, 1.0), p, rel_tol=1e-9, abs_tol=1e-12)


def test_extremize_half_is_fixed() -> None:
    for c in (0.1, 0.5, 1.0, 2.0, 5.0):
        assert math.isclose(extremize(0.5, c), 0.5, abs_tol=1e-12)


@given(p=st.floats(min_value=1e-6, max_value=0.5 - 1e-3))
def test_extremize_direction_below_half(p: float) -> None:
    # p < 0.5: coefficient > 1 pushes further down, < 1 pulls up toward 0.5.
    assert extremize(p, 2.0) < p
    assert extremize(p, 0.5) > p


@given(p=st.floats(min_value=0.5 + 1e-3, max_value=1.0 - 1e-6))
def test_extremize_direction_above_half(p: float) -> None:
    # p > 0.5: coefficient > 1 pushes further up, < 1 pulls down toward 0.5.
    assert extremize(p, 2.0) > p
    assert extremize(p, 0.5) < p


# --- agreement_scaled_coefficient ---------------------------------------------

log_odds_lists = st.lists(st.floats(min_value=-6.0, max_value=6.0), min_size=1, max_size=8)


@given(
    xs=log_odds_lists,
    lo=st.floats(min_value=0.5, max_value=1.0),
    hi=st.floats(min_value=1.0, max_value=2.0),
)
def test_agreement_within_bounds(xs: list[float], lo: float, hi: float) -> None:
    # lo <= 1.0 <= hi by construction, so min <= max always holds.
    c = agreement_scaled_coefficient(xs, min_coefficient=lo, max_coefficient=hi)
    assert lo - 1e-9 <= c <= hi + 1e-9


@given(v=st.floats(min_value=-6.0, max_value=6.0), n=st.integers(min_value=1, max_value=8))
def test_agreement_unanimous_gives_max(v: float, n: int) -> None:
    # Zero spread (identical members) must map to the max coefficient.
    c = agreement_scaled_coefficient([v] * n, min_coefficient=1.0, max_coefficient=1.3)
    assert math.isclose(c, 1.3, rel_tol=1e-12, abs_tol=1e-12)


@given(xs=log_odds_lists, k=st.floats(min_value=1.0, max_value=4.0))
def test_agreement_monotone_non_increasing_in_spread(xs: list[float], k: float) -> None:
    # Scaling every log-odds by k >= 1 multiplies the spread by k (never shrinks
    # it), so the coefficient must not increase.
    base = agreement_scaled_coefficient(xs, min_coefficient=1.0, max_coefficient=1.3)
    wider = agreement_scaled_coefficient(
        [x * k for x in xs], min_coefficient=1.0, max_coefficient=1.3
    )
    assert wider <= base + 1e-9


@given(xs=log_odds_lists)
def test_agreement_identity_when_bounds_equal(xs: list[float]) -> None:
    # min == max == 1 -> always exactly 1, whatever the spread.
    assert agreement_scaled_coefficient(xs, min_coefficient=1.0, max_coefficient=1.0) == 1.0


def test_agreement_reaches_min_at_ceiling() -> None:
    # A spread at/above the ceiling clamps to the min coefficient.
    xs = [-3.0, 3.0]  # population std = 3.0, above the default ceiling of 1.5
    c = agreement_scaled_coefficient(xs, min_coefficient=1.0, max_coefficient=1.3)
    assert math.isclose(c, 1.0, abs_tol=1e-12)


def test_agreement_errors() -> None:
    with pytest.raises(ValueError):
        agreement_scaled_coefficient([], min_coefficient=1.0, max_coefficient=1.3)
    with pytest.raises(ValueError):
        agreement_scaled_coefficient([0.0, 1.0], min_coefficient=1.5, max_coefficient=1.0)


def test_population_std_examples() -> None:
    assert population_std([2.0, 2.0, 2.0]) == 0.0
    assert math.isclose(population_std([-1.0, 1.0]), 1.0, abs_tol=1e-12)
    with pytest.raises(ValueError):
        population_std([])


def test_platt_errors() -> None:
    with pytest.raises(ValueError):
        platt_fit([(0.5, True)])
    with pytest.raises(ValueError):
        platt_fit([(0.3, True), (0.7, True)])  # all identical outcomes


def test_platt_apply_matches_formula() -> None:
    params = PlattParams(a=1.5, b=-0.3)
    p = 0.7
    expected_logit = 1.5 * to_log_odds(0.7) - 0.3
    assert math.isclose(platt_apply(p, params), from_log_odds(expected_logit), rel_tol=1e-12)


def test_platt_improves_underconfident_dataset() -> None:
    # Build a systematically UNDERconfident dataset deterministically:
    # the true probability is a sharpened version of the stated forecast, so
    # recalibration should sharpen and reduce log loss.
    rng = random.Random(20260709)
    pairs: list[tuple[float, bool]] = []
    for _ in range(2000):
        stated = rng.uniform(0.05, 0.95)
        true_logit = 1.8 * to_log_odds(stated)  # reality is more extreme
        true_p = from_log_odds(true_logit)
        outcome = rng.random() < true_p
        pairs.append((stated, outcome))

    def mean_log_loss(ps: list[tuple[float, bool]]) -> float:
        total = 0.0
        for p, o in ps:
            p_c = min(max(p, 1e-9), 1.0 - 1e-9)
            total += -math.log(p_c) if o else -math.log(1.0 - p_c)
        return total / len(ps)

    params = platt_fit(pairs)
    calibrated = [(platt_apply(p, params), o) for p, o in pairs]

    assert mean_log_loss(calibrated) < mean_log_loss(pairs)
    # Sanity: sharpening slope should be > 1 for an underconfident source.
    assert params.a > 1.0
    # Brier should also improve.
    assert mean_brier(calibrated) < mean_brier(pairs)
