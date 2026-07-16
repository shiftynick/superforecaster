"""Tests for the diverse ensemble (candidate B), driven by fake LLM clients.

Each member forecast runs through the real ``run_forecast_pass`` machinery; the
only seam is ``client_factory``, which hands each member a canned client. A fake
either returns a fixed probability as fenced JSON or raises ``LlmError`` to
simulate a member that fails and must be skipped.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from superforecaster.aggregation import extremize, pool_log_odds, to_log_odds
from superforecaster.config import EnsembleMemberConfig, EnsembleSettings, TaskModelConfig
from superforecaster.ensemble import FRAMES, EnsembleResult, run_ensemble
from superforecaster.llm import LlmError, LlmResult
from superforecaster.models import QuestionSnapshot

NOW = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)


def _fenced(probability: float) -> str:
    return (
        "Reasoning about the base rate.\n\n"
        "```json\n"
        f'{{"probability": {probability}, "confidence": "medium", '
        '"rationale": "base rate then adjust, briefly", "reference_class": "c"}\n'
        "```"
    )


class _FakeLlm:
    """A one-shot fake: returns a fixed probability, or raises to simulate failure."""

    def __init__(self, probability: float | None = None, *, fail: bool = False) -> None:
        self._probability = probability
        self._fail = fail

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        if self._fail:
            raise LlmError("simulated member failure")
        assert self._probability is not None
        return LlmResult(
            text=_fenced(self._probability),
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            stop_reason="end_turn",
            model=model,
        )


def _factory(*fakes: _FakeLlm) -> Callable[[TaskModelConfig], _FakeLlm]:
    """A client_factory that hands out the given fakes in member order."""
    it = iter(fakes)

    def factory(config: TaskModelConfig) -> _FakeLlm:
        return next(it)

    return factory


def _snapshot() -> QuestionSnapshot:
    return QuestionSnapshot(
        scraped_at=NOW,
        post_id=201,
        question_id=301,
        title="Will it rain tomorrow?",
        description="A weather question.",
        resolution_criteria="Any measurable precipitation counts.",
        fine_print="",
        scheduled_close_time=datetime(2026, 8, 1, tzinfo=UTC),
        status="open",
    )


def _member(frame: str, model: str) -> EnsembleMemberConfig:
    return EnsembleMemberConfig(frame=frame, model=model, max_tokens=100, provider="claude-cli")


def _settings(members: list[EnsembleMemberConfig], **overrides: object) -> EnsembleSettings:
    base: dict[str, object] = {"members": tuple(members)}
    base.update(overrides)
    return EnsembleSettings(**base)  # type: ignore[arg-type]


def test_happy_path_pools_correctly() -> None:
    # Symmetric members: log-odds of 0.4, 0.5, 0.6 average to 0 -> pooled 0.5.
    members = [
        _member("outside-first", "m0"),
        _member("inside-first", "m1"),
        _member("status-quo", "m2"),
    ]
    result = run_ensemble(
        _snapshot(),
        settings=_settings(members),
        client_factory=_factory(_FakeLlm(0.4), _FakeLlm(0.5), _FakeLlm(0.6)),
        now=NOW,
    )

    assert isinstance(result, EnsembleResult)
    assert result.pooled_probability == pytest.approx(0.5)
    assert result.member_probabilities == [0.4, 0.5, 0.6]
    assert len(result.member_records) == 3
    assert result.failures == []
    # Fixed mode with the default coefficient (1.0) leaves the pool unchanged.
    assert result.coefficient_applied == 1.0
    assert result.extremization_mode == "fixed"
    # Three members at $0.01 each.
    assert result.total_cost_usd == pytest.approx(0.03)
    # Tight spread (~0.33 < 0.4 default high threshold) -> high confidence.
    assert result.confidence == "high"


def test_each_member_gets_its_frame_addendum() -> None:
    members = [_member("outside-first", "m0"), _member("status-quo", "m1")]
    result = run_ensemble(
        _snapshot(),
        settings=_settings(members),
        client_factory=_factory(_FakeLlm(0.5), _FakeLlm(0.5)),
        now=NOW,
    )
    # The frame text is a system-prompt addendum; each member's record still parses.
    assert len(result.member_records) == 2
    # Frames are genuinely distinct prompts (not synonyms of one another).
    assert len({FRAMES[k] for k in FRAMES}) == 3


def test_one_member_fails_pools_from_survivors() -> None:
    members = [
        _member("outside-first", "m0"),
        _member("inside-first", "m1"),
        _member("status-quo", "m2"),
    ]
    result = run_ensemble(
        _snapshot(),
        settings=_settings(members),
        client_factory=_factory(_FakeLlm(0.4), _FakeLlm(fail=True), _FakeLlm(0.6)),
        now=NOW,
    )
    assert result.member_probabilities == [0.4, 0.6]
    assert result.pooled_probability == pytest.approx(0.5)
    assert len(result.member_records) == 2
    assert len(result.failures) == 1
    assert "inside-first/m1" in result.failures[0]
    assert "simulated member failure" in result.failures[0]
    # Cost sums only the survivors.
    assert result.total_cost_usd == pytest.approx(0.02)


def test_all_but_one_fail_raises() -> None:
    members = [
        _member("outside-first", "m0"),
        _member("inside-first", "m1"),
        _member("status-quo", "m2"),
    ]
    with pytest.raises(LlmError, match="at least 2 surviving members"):
        run_ensemble(
            _snapshot(),
            settings=_settings(members),
            client_factory=_factory(_FakeLlm(0.4), _FakeLlm(fail=True), _FakeLlm(fail=True)),
            now=NOW,
        )


def test_agreement_mode_extremizes_unanimous_more_than_divergent() -> None:
    members = [
        _member("outside-first", "m0"),
        _member("inside-first", "m1"),
        _member("status-quo", "m2"),
    ]
    agreement = {
        "extremization_mode": "agreement",
        "agreement_min_coefficient": 1.0,
        "agreement_max_coefficient": 1.5,
    }

    unanimous = run_ensemble(
        _snapshot(),
        settings=_settings(members, **agreement),
        client_factory=_factory(_FakeLlm(0.7), _FakeLlm(0.7), _FakeLlm(0.7)),
        now=NOW,
    )
    divergent = run_ensemble(
        _snapshot(),
        settings=_settings(members, **agreement),
        client_factory=_factory(_FakeLlm(0.1), _FakeLlm(0.5), _FakeLlm(0.9)),
        now=NOW,
    )

    # Unanimity -> max coefficient; disagreement -> a smaller one.
    assert unanimous.coefficient_applied == pytest.approx(1.5)
    assert divergent.coefficient_applied < unanimous.coefficient_applied
    # And the unanimous pool is actually pushed further from 0.5 (0.7 -> ~0.78).
    assert unanimous.pooled_probability > 0.7
    expected = extremize(pool_log_odds([0.7, 0.7, 0.7]), 1.5)
    assert unanimous.pooled_probability == pytest.approx(expected)


@pytest.mark.parametrize(
    ("probs", "expected"),
    [
        ([0.5, 0.5], "high"),  # spread 0
        ([0.3, 0.7], "medium"),  # spread ~0.85, between the thresholds
        ([0.02, 0.98], "low"),  # spread ~3.9, above the low threshold
    ],
)
def test_confidence_bands_from_spread(probs: list[float], expected: str) -> None:
    members = [_member("outside-first", "m0"), _member("inside-first", "m1")]
    result = run_ensemble(
        _snapshot(),
        settings=_settings(members),  # default thresholds: high<=0.4, low>=1.2
        client_factory=_factory(*(_FakeLlm(p) for p in probs)),
        now=NOW,
    )
    assert result.confidence == expected
    # Reported spread matches the population std of the member log-odds.
    logits = [to_log_odds(p) for p in probs]
    mean = sum(logits) / len(logits)
    expected_spread = math.sqrt(sum((x - mean) ** 2 for x in logits) / len(logits))
    assert result.spread == pytest.approx(expected_spread)


def test_unknown_frame_raises_value_error() -> None:
    members = [_member("outside-first", "m0"), _member("nonsense-frame", "m1")]
    with pytest.raises(ValueError, match="Unknown ensemble frame"):
        run_ensemble(
            _snapshot(),
            settings=_settings(members),
            client_factory=_factory(_FakeLlm(0.5), _FakeLlm(0.5)),
            now=NOW,
        )
