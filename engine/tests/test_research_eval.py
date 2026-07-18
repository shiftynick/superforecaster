"""Tests for the research-quality eval harness — no live network, no live LLM.

The judge LLM is a fake returning queued replies; metrics, blinding, parsing,
retry, randomization, and the store are all exercised deterministically.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import Any

import pytest

from superforecaster.config import RetrievalSettings, TaskModelConfig
from superforecaster.extract import TIER_HTTPX_TRAFILATURA, TIER_NONE, ExtractedDoc
from superforecaster.llm import LlmError, LlmResult
from superforecaster.models import QuestionSnapshot
from superforecaster.research import ResearchBundle
from superforecaster.research_eval import (
    BundleEvalRecord,
    BundleEvalStore,
    PairwiseRecord,
    RubricJudgment,
    bundle_fingerprint,
    compute_metrics,
    render_blinded_bundle,
    run_pairwise_judge,
    run_rubric_judge,
    summarize_retrieval,
)
from superforecaster.search import SearchResult

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
JUDGE_CONFIG = TaskModelConfig(model="z-ai/glm-5.2", max_tokens=2048, effort="low")


def _snapshot() -> QuestionSnapshot:
    return QuestionSnapshot(
        scraped_at=NOW,
        post_id=201,
        question_id=301,
        title="Will the Fed cut interest rates in September 2026?",
        description="A macroeconomics question.",
        resolution_criteria=(
            "Resolves YES if the FOMC lowers the target range, per federalreserve.gov."
        ),
        fine_print="Only scheduled meetings count.",
        scheduled_close_time=datetime(2026, 9, 30, tzinfo=UTC),
        status="open",
    )


def _result(
    url: str,
    *,
    title: str = "A title",
    snippet: str = "",
    provider: str = "asknews",
    domain: str = "example.com",
    published_at: datetime | None = NOW,
) -> SearchResult:
    return SearchResult(
        url=url,
        title=title,
        snippet=snippet,
        published_at=published_at,
        source_domain=domain,
        provider=provider,
        raw={"provider_note": provider, "secret": "asknews-internal"},
    )


def _doc(url: str, *, text: str = "body text", ok: bool = True) -> ExtractedDoc:
    return ExtractedDoc(
        url=url,
        text=text,
        title="Doc",
        published_at=NOW,
        tier=TIER_HTTPX_TRAFILATURA if ok else TIER_NONE,
        ok=ok,
    )


def _bundle(
    *,
    results: list[SearchResult] | None = None,
    docs: list[ExtractedDoc] | None = None,
    tier_stats: dict[str, int] | None = None,
    credits: int = 0,
) -> ResearchBundle:
    return ResearchBundle(
        queries=["q1"],
        results=results if results is not None else [],
        docs=docs if docs is not None else [],
        tier_stats=tier_stats if tier_stats is not None else {},
        search_credits_used=credits,
        news_quota_exhausted=False,
        retrieved_at=NOW,
    )


class _FakeLlm:
    """Returns queued LlmResults and records each call's system+prompt."""

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
        self.calls.append({"system": system, "prompt": prompt})
        return self._results.pop(0)


def _llm_result(text: str, *, cost_usd: float | None = 0.002) -> LlmResult:
    return LlmResult(
        text=text,
        input_tokens=200,
        output_tokens=80,
        cost_usd=cost_usd,
        stop_reason="end_turn",
        model="z-ai/glm-5.2",
    )


def _rubric_json(
    *,
    coverage: int = 4,
    overall: int = 4,
    missing: str = "The official Fed rate-decision page is not present.",
    justification: str = "Sources address the rate decision directly.",
) -> str:
    return (
        "Here is my grade.\n\n"
        "{"
        f'"coverage": {coverage}, "coverage_justification": "{justification}", '
        f'"source_quality": 3, "source_quality_justification": "{justification}", '
        f'"recency": 5, "recency_justification": "{justification}", '
        f'"relevance": 4, "relevance_justification": "{justification}", '
        f'"resolution_grounding": 2, "resolution_grounding_justification": "{justification}", '
        f'"overall": {overall}, "missing": "{missing}"'
        "}"
    )


def _pairwise_json(*, overall: str = "X") -> str:
    return (
        "Comparing.\n\n"
        "{"
        '"coverage": "X", "source_quality": "Y", "recency": "tie", '
        '"relevance": "X", "resolution_grounding": "Y", '
        f'"overall": "{overall}", "reason": "X has the resolution source."'
        "}"
    )


# --- compute_metrics ----------------------------------------------------------


def test_compute_metrics_counts_and_coverage() -> None:
    # One result whose domain is in the resolution criteria (federalreserve.gov),
    # one dated result, one undated result; a fetched doc and a failed doc.
    results = [
        _result(
            "https://federalreserve.gov/x",
            title="FOMC rate decision September",
            domain="federalreserve.gov",
            published_at=datetime(2026, 7, 8, tzinfo=UTC),  # 10 days old
        ),
        _result(
            "https://news.example/a",
            title="Analysts expect a rate cut",
            snippet="The FOMC target range discussion",
            domain="news.example",
            published_at=datetime(2026, 7, 16, tzinfo=UTC),  # 2 days old
        ),
        _result(
            "https://blog.example/b",
            title="Opinion piece",
            domain="blog.example",
            published_at=None,  # undated
        ),
    ]
    docs = [
        _doc("https://federalreserve.gov/x", text="rate cut target range"),
        _doc("https://news.example/a", ok=False),
    ]
    bundle = _bundle(
        results=results,
        docs=docs,
        tier_stats={TIER_HTTPX_TRAFILATURA: 1, TIER_NONE: 1},
        credits=4,
    )

    m = compute_metrics(_snapshot(), bundle, now=NOW)

    assert m.result_count == 3
    assert m.doc_count == 2
    assert m.doc_ok_count == 1
    assert m.unique_domains == 3
    assert m.tier_stats == {TIER_HTTPX_TRAFILATURA: 1, TIER_NONE: 1}
    # recency: NOW is 12:00, publishes at 00:00 -> ages 10.5 and 2.5 days; median
    # 6.5, newest 2.5; one of three undated.
    assert m.median_age_days == pytest.approx(6.5)
    assert m.newest_age_days == pytest.approx(2.5)
    assert m.undated_fraction == pytest.approx(1 / 3)
    # resolution-source HIT: federalreserve.gov appears in the criteria text.
    assert m.resolution_source_present is True
    # key-term coverage: question terms found somewhere in the bundle text.
    assert m.key_terms_total > 0
    assert 0.0 < m.key_term_coverage <= 1.0
    assert m.key_terms_covered >= 1
    assert m.search_credits_used == 4
    assert m.approx_bundle_chars > 0


def test_compute_metrics_resolution_source_miss_and_all_undated() -> None:
    results = [
        _result("https://off.example/a", domain="off.example", published_at=None),
        _result("https://off.example/b", domain="other.example", published_at=None),
    ]
    bundle = _bundle(results=results)
    m = compute_metrics(_snapshot(), bundle, now=NOW)

    assert m.resolution_source_present is False  # no result domain in the criteria
    assert m.undated_fraction == pytest.approx(1.0)
    assert m.median_age_days is None
    assert m.newest_age_days is None


def test_compute_metrics_empty_bundle() -> None:
    m = compute_metrics(_snapshot(), _bundle(), now=NOW)
    assert m.result_count == 0
    assert m.undated_fraction == 0.0
    assert m.key_term_coverage == pytest.approx(m.key_terms_covered / max(m.key_terms_total, 1))


# --- blinding -----------------------------------------------------------------


def test_render_blinded_bundle_omits_provider_strings() -> None:
    results = [
        _result("https://a.example/1", provider="asknews", domain="a.example"),
        _result("https://b.example/2", provider="tavily", domain="b.example"),
    ]
    docs = [_doc("https://a.example/1", text="the extracted body")]
    rendered = render_blinded_bundle(_bundle(results=results, docs=docs))

    # Provider identities (and their raw-payload secret) must not leak.
    assert "asknews" not in rendered
    assert "tavily" not in rendered
    assert "asknews-internal" not in rendered
    # But URLs/domains and bodies ARE kept — they are evidence of quality.
    assert "a.example" in rendered
    assert "https://b.example/2" in rendered
    assert "the extracted body" in rendered


# --- rubric judge -------------------------------------------------------------


def test_run_rubric_judge_happy_path() -> None:
    llm = _FakeLlm(_llm_result(_rubric_json(coverage=4, overall=5)))
    result = run_rubric_judge(_snapshot(), _bundle(), client=llm, config=JUDGE_CONFIG)

    assert isinstance(result.judgment, RubricJudgment)
    assert result.judgment.coverage == 4
    assert result.judgment.overall == 5
    assert result.judgment.resolution_grounding == 2
    assert result.input_tokens == 200
    assert result.output_tokens == 80
    assert result.cost_usd == pytest.approx(0.002)
    assert len(llm.calls) == 1
    # The judge prompt carries the question and blinded sources, never a provider.
    assert "RESOLUTION CRITERIA" in llm.calls[0]["prompt"]


def test_run_rubric_judge_degenerate_retries_then_succeeds() -> None:
    degenerate = _rubric_json(justification="n/a", missing="n/a")
    llm = _FakeLlm(_llm_result(degenerate), _llm_result(_rubric_json()))
    result = run_rubric_judge(_snapshot(), _bundle(), client=llm, config=JUDGE_CONFIG)

    assert result.judgment.overall == 4
    assert len(llm.calls) == 2
    assert "could not be used" in llm.calls[1]["prompt"]
    # Tokens summed across both calls.
    assert result.input_tokens == 400


def test_run_rubric_judge_double_failure_raises() -> None:
    llm = _FakeLlm(_llm_result("no json here"), _llm_result("still no json"))
    with pytest.raises(LlmError, match="after one retry"):
        run_rubric_judge(_snapshot(), _bundle(), client=llm, config=JUDGE_CONFIG)


# --- pairwise judge -----------------------------------------------------------


def test_run_pairwise_judge_records_xy_mapping_and_verdict() -> None:
    llm = _FakeLlm(_llm_result(_pairwise_json(overall="X")))
    result = run_pairwise_judge(
        _snapshot(),
        _bundle(results=[_result("https://a.example/1")]),
        _bundle(results=[_result("https://b.example/2")]),
        client=llm,
        config=JUDGE_CONFIG,
        rng=random.Random(0),
    )

    assert result.x_source in ("a", "b")
    assert result.verdict.overall == "X"
    assert result.verdict.coverage == "X"
    assert result.verdict.source_quality == "Y"
    assert result.verdict.recency == "tie"
    assert result.cost_usd == pytest.approx(0.002)
    # Both blinded sets are shown, labeled X and Y.
    prompt = llm.calls[0]["prompt"]
    assert "RESEARCH SET X" in prompt
    assert "RESEARCH SET Y" in prompt


def test_run_pairwise_judge_assignment_flips_with_seed() -> None:
    # Two seeds giving opposite first-draw sides prove the mapping is recorded,
    # not hard-coded. Find one seed per side.
    def x_source_for(seed: int) -> str:
        llm = _FakeLlm(_llm_result(_pairwise_json()))
        res = run_pairwise_judge(
            _snapshot(),
            _bundle(results=[_result("https://a.example/1")]),
            _bundle(results=[_result("https://b.example/2")]),
            client=llm,
            config=JUDGE_CONFIG,
            rng=random.Random(seed),
        )
        return res.x_source

    sources = {x_source_for(seed) for seed in range(20)}
    assert sources == {"a", "b"}  # both assignments occur across seeds


def test_run_pairwise_judge_bad_verdict_retries() -> None:
    bad = (
        '{"coverage": "maybe", "source_quality": "X", "recency": "X", '
        '"relevance": "X", "resolution_grounding": "X", "overall": "X", "reason": "r"}'
    )
    llm = _FakeLlm(_llm_result(bad), _llm_result(_pairwise_json()))
    result = run_pairwise_judge(
        _snapshot(),
        _bundle(),
        _bundle(),
        client=llm,
        config=JUDGE_CONFIG,
        rng=random.Random(1),
    )
    assert result.verdict.overall == "X"
    assert len(llm.calls) == 2


# --- fingerprint + store ------------------------------------------------------


def test_bundle_fingerprint_is_order_independent() -> None:
    a = _bundle(results=[_result("https://x/1"), _result("https://y/2")])
    b = _bundle(results=[_result("https://y/2"), _result("https://x/1")])
    c = _bundle(results=[_result("https://x/1")])
    assert bundle_fingerprint(a) == bundle_fingerprint(b)
    assert bundle_fingerprint(a) != bundle_fingerprint(c)


def _rubric_record() -> BundleEvalRecord:
    return BundleEvalRecord(
        created_at=NOW,
        question_id=301,
        post_id=201,
        title="Q?",
        kind="rubric",
        bundle_fingerprint="fp-a",
        retrieval=summarize_retrieval(RetrievalSettings()),
        metrics=compute_metrics(_snapshot(), _bundle(), now=NOW),
        rubric=RubricJudgment.model_validate_json(_rubric_json().split("\n\n", 1)[1]),
        judge_model="z-ai/glm-5.2",
        input_tokens=200,
        output_tokens=80,
        cost_usd=0.002,
    )


def test_store_round_trip(tmp_path: Any) -> None:
    store = BundleEvalStore(tmp_path)
    older = _rubric_record()
    newer = older.model_copy(update={"created_at": datetime(2026, 7, 19, tzinfo=UTC)})
    assert store.append([older]) == 1
    assert store.append([newer]) == 1

    all_records = list(store.iter_all())
    assert len(all_records) == 2
    assert all_records[0].bundle_fingerprint == "fp-a"

    latest = store.latest_by_question()
    assert latest[301].created_at == datetime(2026, 7, 19, tzinfo=UTC)


def test_store_round_trip_pairwise_record(tmp_path: Any) -> None:
    from superforecaster.research_eval import PairwiseVerdict

    pairwise = PairwiseRecord(
        verdict=PairwiseVerdict(
            coverage="X",
            source_quality="Y",
            recency="tie",
            relevance="X",
            resolution_grounding="Y",
            overall="X",
            reason="r",
        ),
        x_source="self",
        other_fingerprint="fp-b",
        other_retrieval=summarize_retrieval(RetrievalSettings(news_provider="perplexity")),
    )
    record = _rubric_record().model_copy(
        update={"kind": "pairwise", "rubric": None, "pairwise": pairwise}
    )
    store = BundleEvalStore(tmp_path)
    store.append([record])
    (loaded,) = list(store.iter_all())
    assert loaded.pairwise is not None
    assert loaded.pairwise.x_source == "self"
    assert loaded.pairwise.other_retrieval.news_provider == "perplexity"
