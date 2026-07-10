"""Tests for the baseline forecasting pass, using a fake LLM client.

The client seam is now ``.run(system, prompt, model, max_tokens) -> LlmResult``;
structured output is parsed out of the model's text (a JSON object, optionally
fenced) rather than via a server-side schema. The fake below returns canned
:class:`LlmResult` values, one per call, so we can drive the happy path, the
one-retry path, and the give-up path deterministically.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from superforecaster.config import TaskModelConfig
from superforecaster.extract import TIER_HTTPX_TRAFILATURA, ExtractedDoc
from superforecaster.forecaster import (
    CHECKLIST_PROMPT,
    OUTPUT_FORMAT,
    ForecastOutput,
    run_forecast_pass,
)
from superforecaster.llm import LlmError, LlmResult
from superforecaster.models import QuestionSnapshot
from superforecaster.research import ResearchBundle
from superforecaster.search import SearchResult

NOW = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)
CONFIG = TaskModelConfig(model="claude-sonnet-5", max_tokens=4096, effort=None)


def _result(
    text: str,
    *,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cost_usd: float | None = 0.02,
) -> LlmResult:
    return LlmResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        stop_reason="end_turn",
        model="claude-sonnet-5",
    )


class _FakeLlm:
    """Returns queued :class:`LlmResult` values and records each call's args."""

    def __init__(self, *results: LlmResult) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        self.calls.append(
            {
                "system": system,
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "json_schema": json_schema,
                "effort": effort,
            }
        )
        return self._results.pop(0)


def _fenced(probability: float = 0.63) -> str:
    return (
        "Here is my reasoning about the base rate and adjustments.\n\n"
        "```json\n"
        f'{{"probability": {probability}, "confidence": "medium", '
        '"rationale": "Base rate then adjust.", '
        '"reference_class": "rain days in this region"}\n'
        "```"
    )


def _bare(probability: float = 0.63) -> str:
    return (
        "Reasoning first, no code fence this time. Final answer:\n"
        f'{{"probability": {probability}, "confidence": "medium", '
        '"rationale": "Base rate then adjust.", '
        '"reference_class": "rain days in this region"}'
    )


def _snapshot() -> QuestionSnapshot:
    return QuestionSnapshot(
        scraped_at=NOW,
        post_id=201,
        question_id=301,
        title="Will it rain tomorrow?",
        description="A weather question.",
        resolution_criteria="Any measurable precipitation counts.",
        fine_print="Trace amounts do not count.",
        scheduled_close_time=datetime(2026, 8, 1, tzinfo=UTC),
        status="open",
    )


def test_happy_path_fenced_json() -> None:
    client = _FakeLlm(_result(_fenced(0.63)))

    record = run_forecast_pass(_snapshot(), client=client, config=CONFIG, now=NOW)

    assert record.probability == 0.63
    assert record.raw_probability == 0.63
    assert record.confidence == "medium"
    assert record.reference_class == "rain days in this region"
    assert record.model == "claude-sonnet-5"
    assert record.input_tokens == 1000
    assert record.output_tokens == 500
    assert record.dry_run is True
    assert record.submitted_at is None
    # Cost is the CLI-reported figure, not the local pricing table.
    assert record.cost_usd == pytest.approx(0.02)


def test_happy_path_trailing_bare_json() -> None:
    client = _FakeLlm(_result(_bare(0.42)))
    record = run_forecast_pass(_snapshot(), client=client, config=CONFIG, now=NOW)
    assert record.probability == 0.42
    assert record.reference_class == "rain days in this region"


def test_invalid_then_valid_retry_sums_tokens_and_cost() -> None:
    bad = _result("I refuse to answer.", input_tokens=100, output_tokens=50, cost_usd=0.01)
    good = _result(_fenced(0.30), input_tokens=200, output_tokens=75, cost_usd=0.03)
    client = _FakeLlm(bad, good)

    record = run_forecast_pass(_snapshot(), client=client, config=CONFIG, now=NOW)

    assert record.probability == 0.30
    assert len(client.calls) == 2
    # Tokens summed across both calls; cost summed too.
    assert record.input_tokens == 300
    assert record.output_tokens == 125
    assert record.cost_usd == pytest.approx(0.04)
    # The retry prompt quotes the parse failure and demands the JSON object.
    assert "could not be parsed" in client.calls[1]["prompt"]


def test_invalid_twice_raises_llm_error() -> None:
    client = _FakeLlm(_result("no json here"), _result("still no json"))
    with pytest.raises(LlmError, match="after one retry"):
        run_forecast_pass(_snapshot(), client=client, config=CONFIG, now=NOW)


def test_validation_error_triggers_retry() -> None:
    # Well-formed JSON but a probability outside [0, 1] -> ValidationError -> retry.
    bad_text = '{"probability": 5, "confidence": "high", "rationale": "r", "reference_class": "c"}'
    client = _FakeLlm(_result(bad_text), _result(_fenced(0.51)))
    record = run_forecast_pass(_snapshot(), client=client, config=CONFIG, now=NOW)
    assert record.probability == 0.51
    assert len(client.calls) == 2


def test_probability_is_clamped() -> None:
    client = _FakeLlm(_result(_fenced(0.999)))
    record = run_forecast_pass(_snapshot(), client=client, config=CONFIG, now=NOW)
    assert record.probability == 0.99  # clamped for Metaculus
    assert record.raw_probability == 0.999  # raw preserved

    record2 = run_forecast_pass(
        _snapshot(), client=_FakeLlm(_result(_fenced(0.0))), config=CONFIG, now=NOW
    )
    assert record2.probability == 0.01
    assert record2.raw_probability == 0.0


def test_prompt_includes_key_fields_and_output_format() -> None:
    client = _FakeLlm(_result(_fenced()))
    run_forecast_pass(_snapshot(), client=client, config=CONFIG, now=NOW)

    call = client.calls[0]
    assert "Will it rain tomorrow?" in call["prompt"]
    assert "Any measurable precipitation counts." in call["prompt"]
    assert "2026-08-01" in call["prompt"]  # scheduled close date
    # Today's date and the OUTPUT FORMAT instruction are in the system prompt.
    assert "2026-07-09" in call["system"]
    assert "OUTPUT FORMAT" in call["system"]
    assert call["model"] == "claude-sonnet-5"
    assert call["max_tokens"] == 4096
    # No effort on the default CONFIG -> passed through as None.
    assert call["effort"] is None


def test_effort_passed_through_to_client() -> None:
    cfg = TaskModelConfig(model="claude-sonnet-5", max_tokens=4096, effort="low")
    client = _FakeLlm(_result(_fenced()))
    run_forecast_pass(_snapshot(), client=client, config=cfg, now=NOW)
    assert client.calls[0]["effort"] == "low"


def test_cost_falls_back_to_pricing_when_cli_gives_none() -> None:
    # CLI reported no cost -> fall back to the committed pricing table.
    # 1000*3/1e6 + 500*15/1e6 = 0.0105.
    client = _FakeLlm(_result(_fenced(0.4), cost_usd=None))
    record = run_forecast_pass(_snapshot(), client=client, config=CONFIG, now=NOW)
    assert record.cost_usd == pytest.approx(0.0105)


def test_cost_none_for_unpriced_model(tmp_path: Path) -> None:
    toml = tmp_path / "settings.toml"
    toml.write_text(
        '[tasks.forecast_pass]\nmodel = "mystery"\nmax_tokens = 100\n', encoding="utf-8"
    )
    client = _FakeLlm(_result(_fenced(0.4), cost_usd=None))
    cfg = TaskModelConfig(model="mystery", max_tokens=100, effort=None)
    import os

    os.environ["SUPERFORECASTER_CONFIG"] = str(toml)
    try:
        record = run_forecast_pass(_snapshot(), client=client, config=cfg, now=NOW)
    finally:
        del os.environ["SUPERFORECASTER_CONFIG"]
    assert record.cost_usd is None


def _bundle() -> ResearchBundle:
    result = SearchResult(
        url="https://fed.gov/statement",
        title="FOMC leaves rates unchanged",
        snippet="The committee held the target range steady in July.",
        published_at=datetime(2026, 7, 5, tzinfo=UTC),
        source_domain="fed.gov",
        provider="asknews",
        raw={},
    )
    no_date = SearchResult(
        url="https://blog.example/take",
        title="One analyst's take",
        snippet="A cut looks likely by autumn.",
        published_at=None,
        source_domain="blog.example",
        provider="tavily",
        raw={},
    )
    doc = ExtractedDoc(
        url="https://fed.gov/statement",
        text="Full extracted statement body: the FOMC maintained the range and signalled patience.",
        title="FOMC statement",
        published_at=datetime(2026, 7, 5, tzinfo=UTC),
        tier=TIER_HTTPX_TRAFILATURA,
        ok=True,
    )
    return ResearchBundle(
        queries=["fomc july statement"],
        results=[result, no_date],
        docs=[doc],
        tier_stats={TIER_HTTPX_TRAFILATURA: 1},
        search_credits_used=3,
        news_quota_exhausted=False,
        retrieved_at=NOW,
        input_tokens=100,
        output_tokens=40,
        cost_usd=0.001,
    )


def test_research_renders_sources_into_prompt() -> None:
    client = _FakeLlm(_result(_fenced()))
    run_forecast_pass(_snapshot(), client=client, config=CONFIG, research=_bundle(), now=NOW)

    call = client.calls[0]
    # Sources block, with titles, domains, dates (and a "date unknown" fallback).
    assert "SOURCES" in call["prompt"]
    assert "[S1]" in call["prompt"] and "[S2]" in call["prompt"]
    assert "FOMC leaves rates unchanged" in call["prompt"]
    assert "fed.gov" in call["prompt"]
    assert "2026-07-05" in call["prompt"]
    assert "date unknown" in call["prompt"]
    # Fetched source shows extracted body; unfetched falls back to its snippet.
    assert "Full extracted statement body" in call["prompt"]
    assert "A cut looks likely by autumn." in call["prompt"]
    # The system prompt gains the research instruction (cite sources, weigh quality).
    assert "RESEARCH PROVIDED" in call["system"]
    assert "[S3]" in call["system"] or "cite" in call["system"].lower()


def test_research_bundle_and_credits_stored_on_record() -> None:
    client = _FakeLlm(_result(_fenced()))
    record = run_forecast_pass(
        _snapshot(), client=client, config=CONFIG, research=_bundle(), now=NOW
    )
    assert record.research is not None
    assert record.research.queries == ["fomc july statement"]
    assert record.research_search_credits == 3
    # cost_usd stays LLM-only (the forecast pass), not folded with search credits.
    assert record.cost_usd == pytest.approx(0.02)


def test_no_research_omits_sources_and_instruction() -> None:
    client = _FakeLlm(_result(_fenced()))
    record = run_forecast_pass(_snapshot(), client=client, config=CONFIG, now=NOW)
    assert record.research is None
    assert record.research_search_credits == 0
    assert "SOURCES" not in client.calls[0]["prompt"]
    assert "RESEARCH PROVIDED" not in client.calls[0]["system"]
    # The blind baseline keeps its "no web access" framing.
    assert "no web access" in client.calls[0]["system"].lower()


def test_checklist_prompt_is_documented() -> None:
    # The prompt is our spec for reasoning; guard its load-bearing pieces.
    assert "{today}" in CHECKLIST_PROMPT
    assert "reference class" in CHECKLIST_PROMPT.lower()
    assert "base rate" in CHECKLIST_PROMPT.lower()
    assert "no web access" in CHECKLIST_PROMPT.lower()
    # OUTPUT FORMAT documents the JSON contract each field by field.
    for field in ForecastOutput.model_fields:
        assert field in OUTPUT_FORMAT
