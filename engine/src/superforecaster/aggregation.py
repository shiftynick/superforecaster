"""Combining and recalibrating probability forecasts.

When several forecasters (or several model runs) each give a probability for the
same question, we need one number that summarises them. The tools here work in
**log-odds** space, which is the natural scale for combining probabilities.

- **Log-odds** (a.k.a. the logit) of a probability ``p`` is ``ln(p / (1 - p))``.
  It stretches the [0, 1] probability scale out to the whole number line, so
  that averaging behaves sensibly near the extremes.
- **Log-odds pooling** averages forecasts on the log-odds scale, then converts
  back. This is the standard way to aggregate expert probabilities.
- **Extremizing** pushes a pooled probability further from 50%. A group of
  independent forecasters who each hedge tend, collectively, to be
  underconfident; multiplying the pooled log-odds by a coefficient > 1 corrects
  for that.
- **Platt scaling** learns a simple correction ``p' = sigmoid(a * logit(p) + b)``
  from past (forecast, outcome) data, to fix systematic over/under-confidence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_EPS = 1e-9


def to_log_odds(probability: float) -> float:
    """Convert a probability to log-odds ``ln(p / (1 - p))``.

    The probability is clamped to ``[1e-9, 1 - 1e-9]`` first, so 0 and 1 map to
    large finite values instead of infinities.
    """
    p = min(max(probability, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def from_log_odds(log_odds: float) -> float:
    """Convert log-odds back to a probability via the logistic sigmoid.

    Inverse of :func:`to_log_odds` (up to the clamping near 0 and 1). Always
    returns a value strictly inside (0, 1).
    """
    # Numerically stable logistic for both signs of the input.
    if log_odds >= 0.0:
        z = math.exp(-log_odds)
        return 1.0 / (1.0 + z)
    z = math.exp(log_odds)
    return z / (1.0 + z)


def pool_log_odds(probabilities: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Combine several probabilities into one by averaging in log-odds space.

    Each probability is converted to log-odds, a (optionally weighted) mean is
    taken, and the result is converted back to a probability. With no weights,
    all forecasts count equally. Weights are normalised internally, so only
    their relative sizes matter.

    Raises ``ValueError`` if ``probabilities`` is empty, if ``weights`` has a
    different length, or if the weights sum to a non-positive number.
    """
    n = len(probabilities)
    if n == 0:
        raise ValueError("pool_log_odds requires at least one probability")

    if weights is None:
        weights = [1.0] * n
    elif len(weights) != n:
        raise ValueError("weights must have the same length as probabilities")

    weight_sum = math.fsum(weights)
    if weight_sum <= 0.0:
        raise ValueError("weights must sum to a positive number")

    terms = (w * to_log_odds(p) for p, w in zip(probabilities, weights, strict=True))
    pooled = math.fsum(terms) / weight_sum
    return from_log_odds(pooled)


def population_std(values: Sequence[float]) -> float:
    """Population standard deviation of ``values`` (divides by ``n``, not ``n-1``).

    Used as the *spread* signal for an ensemble: how far apart the members'
    log-odds are. Zero when every value is identical (unanimity). Raises
    ``ValueError`` on empty input.
    """
    n = len(values)
    if n == 0:
        raise ValueError("population_std requires at least one value")
    mean = math.fsum(values) / n
    variance = math.fsum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def agreement_scaled_coefficient(
    log_odds_values: Sequence[float],
    *,
    min_coefficient: float,
    max_coefficient: float,
    spread_floor: float = 0.0,
    spread_ceiling: float = 1.5,
) -> float:
    """Pick an extremization coefficient from how much the members agree.

    The idea (a field-survey graft): a *tight* cluster of independent forecasts is
    strong collective evidence and warrants pushing the pooled number further from
    50%; a *scattered* cluster is weak evidence and warrants little or no push. So
    we map the members' **spread** (the population std dev of their log-odds) onto
    a coefficient that runs the opposite way — high agreement (low spread) gives a
    high coefficient (extremize more), high disagreement gives ``min_coefficient``.

    The spread is linearly mapped from ``[spread_floor, spread_ceiling]`` onto
    ``[max_coefficient, min_coefficient]`` and clamped outside that range: a spread
    at or below ``spread_floor`` yields ``max_coefficient``, one at or above
    ``spread_ceiling`` yields ``min_coefficient``.

    Args:
        log_odds_values: The members' forecasts, already in log-odds space.
        min_coefficient: Coefficient at maximum disagreement (spread ceiling).
        max_coefficient: Coefficient at full agreement (spread floor).
        spread_floor: Spread at/below which agreement counts as total.
        spread_ceiling: Spread at/above which disagreement counts as total.

    Returns:
        A coefficient in ``[min_coefficient, max_coefficient]``.

    Raises:
        ValueError: If ``log_odds_values`` is empty or ``min_coefficient`` exceeds
            ``max_coefficient``.
    """
    if not log_odds_values:
        raise ValueError("agreement_scaled_coefficient requires at least one value")
    if min_coefficient > max_coefficient:
        raise ValueError("min_coefficient must not exceed max_coefficient")

    spread = population_std(log_odds_values)
    span = spread_ceiling - spread_floor
    if span <= 0.0:
        # Degenerate range: everything at/below the floor is full agreement.
        fraction = 0.0 if spread <= spread_floor else 1.0
    else:
        fraction = (spread - spread_floor) / span
        fraction = min(1.0, max(0.0, fraction))
    # fraction 0 -> max_coefficient (agree), fraction 1 -> min_coefficient (differ).
    return max_coefficient + fraction * (min_coefficient - max_coefficient)


def extremize(probability: float, coefficient: float) -> float:
    """Push a probability away from (or toward) 50% by scaling its log-odds.

    Multiplies the log-odds by ``coefficient`` and converts back:

    - ``coefficient == 1.0`` returns the probability unchanged (identity).
    - ``coefficient > 1`` makes the forecast more confident (further from 0.5).
    - ``0 <= coefficient < 1`` makes it less confident (closer to 0.5).

    A probability of exactly 0.5 stays 0.5 for any coefficient.
    """
    return from_log_odds(coefficient * to_log_odds(probability))


@dataclass(frozen=True)
class PlattParams:
    """Learned recalibration parameters for :func:`platt_apply`.

    The correction is ``p' = sigmoid(a * logit(p) + b)``:

    - ``a`` is a slope (sharpness). ``a < 1`` softens overconfident forecasts;
      ``a > 1`` sharpens underconfident ones.
    - ``b`` is a bias (shift) that corrects a systematic lean toward yes or no.
    """

    a: float
    b: float


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def platt_fit(pairs: Sequence[tuple[float, bool]]) -> PlattParams:
    """Learn a logistic recalibration from past (forecast, outcome) data.

    Fits ``p' = sigmoid(a * logit(p) + b)`` by minimising the average log loss
    with plain gradient descent. The routine is **deterministic**: it always
    starts from ``a = 1, b = 0`` and runs a fixed number of iterations with a
    fixed learning rate, so the same input always yields the same parameters.

    Raises ``ValueError`` if there are fewer than 2 pairs, or if every outcome is
    the same (in that degenerate case the ideal fit runs off to infinity).
    """
    n = len(pairs)
    if n < 2:
        raise ValueError("platt_fit requires at least 2 pairs")

    outcomes = [1.0 if o else 0.0 for _, o in pairs]
    if all(o == outcomes[0] for o in outcomes):
        raise ValueError("platt_fit requires at least one True and one False outcome")

    # Pre-compute the log-odds of each forecast once.
    features = [to_log_odds(p) for p, _ in pairs]

    a = 1.0
    b = 0.0
    lr = 0.1
    iterations = 2000

    for _ in range(iterations):
        grad_a = 0.0
        grad_b = 0.0
        for x, y in zip(features, outcomes, strict=True):
            pred = _sigmoid(a * x + b)
            err = pred - y  # gradient of log loss wrt the linear input
            grad_a += err * x
            grad_b += err
        grad_a /= n
        grad_b /= n
        a -= lr * grad_a
        b -= lr * grad_b

    return PlattParams(a=a, b=b)


def platt_apply(probability: float, params: PlattParams) -> float:
    """Apply learned recalibration: ``sigmoid(a * logit(p) + b)``.

    Returns a corrected probability strictly inside (0, 1).
    """
    return _sigmoid(params.a * to_log_odds(probability) + params.b)
