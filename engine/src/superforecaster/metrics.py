"""Scoring metrics for probability forecasts.

These functions grade how good a set of probability forecasts was, after the
real-world outcomes are known. Two scoring rules are provided:

- **Brier score** — squared error between the forecast and the outcome.
  Ranges 0..1 for a single forecast. **Lower is better** (0 = perfect).
- **Log score** — the negative log-likelihood of the outcome under the
  forecast. **Lower is better** (0 = perfect). It punishes confident mistakes
  far more harshly than the Brier score does.

There are also **calibration** helpers. "Calibration" asks: when you say 70%,
does the thing actually happen about 70% of the time? A perfectly calibrated
forecaster's stated probabilities match the observed frequencies.

Throughout, a "pair" is ``(probability, outcome)`` where ``probability`` is a
float in [0, 1] and ``outcome`` is ``True`` if the event happened.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_LOG_EPS = 1e-9


def brier_score(probability: float, outcome: bool) -> float:
    """Squared error of one forecast: ``(p - o)^2``.

    ``o`` is 1 if the event happened, else 0. Result is in [0, 1] for a valid
    probability. **Lower is better**; 0 means the forecast was perfect.
    """
    o = 1.0 if outcome else 0.0
    diff = probability - o
    return diff * diff


def mean_brier(pairs: Sequence[tuple[float, bool]]) -> float:
    """Average Brier score over many forecasts. **Lower is better**.

    Raises ``ValueError`` if ``pairs`` is empty.
    """
    if not pairs:
        raise ValueError("mean_brier requires at least one (probability, outcome) pair")
    total = sum(brier_score(p, o) for p, o in pairs)
    return total / len(pairs)


def log_score(probability: float, outcome: bool) -> float:
    """Negative log-likelihood of the outcome under the forecast.

    Returns ``-ln(p)`` if the event happened, else ``-ln(1 - p)``. Measured in
    nats (natural-log units). **Lower is better**; 0 means perfect confidence in
    the correct answer.

    The probability is clamped to ``[eps, 1 - eps]`` with ``eps = 1e-9`` so that
    a forecast of exactly 0 or 1 that turns out wrong yields a large finite
    penalty instead of infinity.
    """
    p = min(max(probability, _LOG_EPS), 1.0 - _LOG_EPS)
    return -math.log(p) if outcome else -math.log(1.0 - p)


@dataclass(frozen=True)
class CalibrationBin:
    """One bucket of a calibration curve.

    Attributes:
        bin_lower: Lower edge of the probability bucket (inclusive).
        bin_upper: Upper edge of the probability bucket (inclusive on the last
            bin, otherwise exclusive).
        mean_predicted: Average forecast probability of the pairs in this bucket.
        observed_frequency: Fraction of those pairs whose event actually
            happened. For a well-calibrated forecaster this is close to
            ``mean_predicted``.
        count: How many pairs fell into this bucket.
    """

    bin_lower: float
    bin_upper: float
    mean_predicted: float
    observed_frequency: float
    count: int


def _bin_index(probability: float, n_bins: int) -> int:
    """Return the bucket index in ``[0, n_bins - 1]`` for a probability."""
    idx = int(probability * n_bins)
    if idx >= n_bins:  # probability == 1.0 lands in the last bin
        idx = n_bins - 1
    return idx


def calibration_curve(
    pairs: Sequence[tuple[float, bool]], n_bins: int = 10
) -> list[CalibrationBin]:
    """Group forecasts into probability buckets and compare stated vs observed.

    The [0, 1] range is split into ``n_bins`` equal-width buckets. Each forecast
    is placed in the bucket matching its probability. For every non-empty bucket
    we report the average forecast and the fraction of events that actually
    happened. **Empty buckets are omitted.** Buckets are returned in ascending
    order of probability.

    Raises ``ValueError`` if ``n_bins < 1`` or ``pairs`` is empty.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if not pairs:
        raise ValueError("calibration_curve requires at least one pair")

    sums: list[float] = [0.0] * n_bins
    hits: list[int] = [0] * n_bins
    counts: list[int] = [0] * n_bins

    for p, o in pairs:
        i = _bin_index(p, n_bins)
        sums[i] += p
        hits[i] += 1 if o else 0
        counts[i] += 1

    width = 1.0 / n_bins
    out: list[CalibrationBin] = []
    for i in range(n_bins):
        if counts[i] == 0:
            continue
        out.append(
            CalibrationBin(
                bin_lower=i * width,
                bin_upper=(i + 1) * width,
                mean_predicted=sums[i] / counts[i],
                observed_frequency=hits[i] / counts[i],
                count=counts[i],
            )
        )
    return out


def expected_calibration_error(pairs: Sequence[tuple[float, bool]], n_bins: int = 10) -> float:
    """Count-weighted average gap between stated and observed probabilities.

    For each non-empty bucket, take ``|mean_predicted - observed_frequency|``,
    weight it by how many forecasts fell in the bucket, and average. Result is
    in [0, 1]. **Lower is better**; 0 means perfectly calibrated on this data.
    """
    bins = calibration_curve(pairs, n_bins)
    total = sum(b.count for b in bins)
    if total == 0:  # unreachable given calibration_curve rejects empty pairs
        raise ValueError("expected_calibration_error requires at least one pair")
    weighted = sum(abs(b.mean_predicted - b.observed_frequency) * b.count for b in bins)
    return weighted / total
