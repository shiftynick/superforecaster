"""Tests for the research pipeline — no live network, no live LLM.

Search clients, the extractor, and the query-gen LLM are all fakes: the pipeline
logic (query parsing/retry, dedupe, fetch preference, trimming, truncation, and
graceful quota degradation) is exercised deterministically.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from superforecaster.config import RetrievalSettings, TaskModelConfig
from superforecaster.extract import TIER_HTTPX_TRAFILATURA, TIER_NONE, ExtractedDoc, Extractor
from superforecaster.llm import LlmError, LlmResult
from superforecaster.models import QuestionSnapshot
from superforecaster.research import ResearchBundle, generate_queries, run_research
from superforecaster.search import SearchQuotaError, SearchResult

NOW = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)
QUERY_CONFIG = TaskModelConfig(model="claude-sonnet-5", max_tokens=1024, effort="low")


def _snapshot() -> QuestionSnapshot:
    return QuestionSnapshot(
        scraped_at=NOW,
        post_id=201,
        question_id=301,
        title="Will the Fed cut rates in September 2026?",
        description="A macroeconomics question.",
        resolution_criteria="Resolves YES if the FOMC lowers the target range.",
        status="open",
    )


def _llm_result(text: str, *, cost_usd: float | None = 0.001) -> LlmResult:
    return LlmResult(
        text=text,
        input_tokens=100,
        output_tokens=40,
        cost_usd=cost_usd,
        stop_reason="end_turn",
        model="claude-sonnet-5",
    )


class _FakeLlm:
    """Returns queued LlmResults and records each call's prompt."""

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
        self.calls.append({"system": system, "prompt": prompt, "effort": effort})
        return self._results.pop(0)


def _queries_json(*queries: str) -> str:
    inner = ", ".join(f'"{q}"' for q in queries)
    return f'Here are the queries.\n\n{{"queries": [{inner}]}}'


def _result(
    url: str,
    *,
    snippet: str = "",
    provider: str = "tavily",
    raw: dict[str, Any] | None = None,
    title: str = "A title",
) -> SearchResult:
    return SearchResult(
        url=url,
        title=title,
        snippet=snippet,
        published_at=NOW,
        source_domain="example.com",
        provider=provider,
        raw=raw if raw is not None else {},
    )


class _FakeSearch:
    """A search client returning fixed results; optionally meters credits."""

    def __init__(
        self,
        results: list[SearchResult],
        *,
        credits_per_call: int | None = None,
        raise_quota: bool = False,
    ) -> None:
        self._results = results
        self._credits_per_call = credits_per_call
        self._raise_quota = raise_quota
        self._used = 0
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        max_results: int,
        start_date: Any = None,
        end_date: Any = None,
    ) -> list[SearchResult]:
        if self._raise_quota:
            raise SearchQuotaError("cap reached")
        self.queries.append(query)
        if self._credits_per_call is not None:
            self._used += self._credits_per_call
        return list(self._results[:max_results])

    def credits_used_this_month(self) -> int:
        return self._used


class _FakeExtractor:
    """Records extract() calls and returns a doc from a per-URL map (or a default)."""

    def __init__(self, docs: dict[str, ExtractedDoc] | None = None) -> None:
        self._docs = docs or {}
        self.calls: list[str] = []

    def extract(self, url: str) -> ExtractedDoc:
        self.calls.append(url)
        if url in self._docs:
            return self._docs[url]
        return ExtractedDoc(
            url=url,
            text="Extracted body text about interest rates and the FOMC decision.",
            title="Extracted",
            published_at=NOW,
            tier=TIER_HTTPX_TRAFILATURA,
            ok=True,
        )


def _run(
    llm: _FakeLlm,
    news: _FakeSearch,
    general: _FakeSearch,
    extractor: _FakeExtractor,
    settings: RetrievalSettings,
) -> ResearchBundle:
    return run_research(
        _snapshot(),
        llm,
        news,
        general,
        cast("Extractor", extractor),
        settings,
        query_gen_config=QUERY_CONFIG,
        now=NOW,
    )


# --- query generation -------------------------------------------------------


def test_generate_queries_parses() -> None:
    llm = _FakeLlm(_llm_result(_queries_json("fed september rate cut", "fomc statement")))
    queries, calls = generate_queries(_snapshot(), llm, QUERY_CONFIG)
    assert queries == ["fed september rate cut", "fomc statement"]
    assert len(calls) == 1
    # Question fields reach the prompt; the system prompt names the resolution source.
    assert "Fed cut rates" in llm.calls[0]["prompt"]
    assert "RESOLUTION SOURCE" in llm.calls[0]["system"]


def test_generate_queries_retries_then_succeeds() -> None:
    llm = _FakeLlm(
        _llm_result("no json at all here"),
        _llm_result(_queries_json("valid query one", "valid query two")),
    )
    queries, calls = generate_queries(_snapshot(), llm, QUERY_CONFIG)
    assert queries == ["valid query one", "valid query two"]
    assert len(calls) == 2
    assert "could not be parsed" in llm.calls[1]["prompt"]


def test_generate_queries_double_failure_raises() -> None:
    llm = _FakeLlm(_llm_result("garbage"), _llm_result("still garbage"))
    with pytest.raises(LlmError, match="after one retry"):
        generate_queries(_snapshot(), llm, QUERY_CONFIG)


def test_generate_queries_dedupes_and_caps_at_four() -> None:
    llm = _FakeLlm(_llm_result(_queries_json("a", "a", "b", "c", "d", "e")))
    queries, _ = generate_queries(_snapshot(), llm, QUERY_CONFIG)
    assert queries == ["a", "b", "c", "d"]


# --- run_research happy path ------------------------------------------------


def test_run_research_happy_path() -> None:
    llm = _FakeLlm(_llm_result(_queries_json("q1")))
    news = _FakeSearch([_result("https://news.example/a", provider="asknews")], credits_per_call=2)
    general = _FakeSearch([_result("https://web.example/b")])
    extractor = _FakeExtractor()
    settings = RetrievalSettings(max_results=5, max_fetches_per_question=8)

    bundle = _run(llm, news, general, extractor, settings)

    assert bundle.queries == ["q1"]
    assert {r.url for r in bundle.results} == {"https://news.example/a", "https://web.example/b"}
    assert len(bundle.docs) == 2
    assert bundle.tier_stats == {TIER_HTTPX_TRAFILATURA: 2}
    assert bundle.search_credits_used == 2  # one news call, 2 credits
    assert bundle.news_quota_exhausted is False
    assert bundle.retrieved_at == NOW
    # Query-gen accounting captured.
    assert bundle.input_tokens == 100
    assert bundle.output_tokens == 40
    assert bundle.cost_usd == pytest.approx(0.001)


def test_run_research_dedupes_by_url() -> None:
    llm = _FakeLlm(_llm_result(_queries_json("q1")))
    shared = "https://dup.example/x"
    news = _FakeSearch([_result(shared, provider="asknews")])
    general = _FakeSearch([_result(shared), _result("https://web.example/other")])
    extractor = _FakeExtractor()
    settings = RetrievalSettings()

    bundle = _run(llm, news, general, extractor, settings)

    urls = [r.url for r in bundle.results]
    assert urls.count(shared) == 1
    assert len(bundle.results) == 2


def test_run_research_prefers_thin_snippet_results_for_fetch() -> None:
    llm = _FakeLlm(_llm_result(_queries_json("q1")))
    fat = _result("https://has-summary.example", snippet="x" * 300)  # usable snippet
    thin = _result("https://needs-body.example", snippet="short")  # not usable
    # News returns the fat-snippet one first; general the thin one.
    news = _FakeSearch([fat])
    general = _FakeSearch([thin])
    extractor = _FakeExtractor()
    settings = RetrievalSettings(max_fetches_per_question=1)  # budget for exactly one fetch

    bundle = _run(llm, news, general, extractor, settings)

    # Only the thin-snippet URL is fetched despite appearing second in results.
    assert extractor.calls == ["https://needs-body.example"]
    assert len(bundle.docs) == 1


def test_run_research_quota_exhausted_degrades() -> None:
    llm = _FakeLlm(_llm_result(_queries_json("q1", "q2")))
    news = _FakeSearch([], raise_quota=True)
    general = _FakeSearch([_result("https://web.example/b")])
    extractor = _FakeExtractor()
    settings = RetrievalSettings()

    bundle = _run(llm, news, general, extractor, settings)

    assert bundle.news_quota_exhausted is True
    assert bundle.search_credits_used == 0
    # General client still ran for every query; results come from it.
    assert general.queries == ["q1", "q2"]
    assert {r.url for r in bundle.results} == {"https://web.example/b"}


def test_run_research_trims_oversized_raw_content() -> None:
    llm = _FakeLlm(_llm_result(_queries_json("q1")))
    big = "y" * 5000
    news = _FakeSearch([])
    general = _FakeSearch([_result("https://web.example/big", raw={"raw_content": big})])
    extractor = _FakeExtractor()
    settings = RetrievalSettings()

    bundle = _run(llm, news, general, extractor, settings)

    stored = bundle.results[0].raw["raw_content"]
    assert len(stored) == 2000  # trimmed to the cap


def test_run_research_truncates_doc_text() -> None:
    llm = _FakeLlm(_llm_result(_queries_json("q1")))
    long_doc = ExtractedDoc(
        url="https://web.example/long",
        text="z" * 12000,
        title="Long",
        published_at=NOW,
        tier=TIER_HTTPX_TRAFILATURA,
        ok=True,
    )
    news = _FakeSearch([])
    general = _FakeSearch([_result("https://web.example/long")])
    extractor = _FakeExtractor({"https://web.example/long": long_doc})
    settings = RetrievalSettings(max_doc_chars=8000)

    bundle = _run(llm, news, general, extractor, settings)

    assert len(bundle.docs[0].text) == 8000


def test_run_research_counts_failed_extraction_as_none_tier() -> None:
    llm = _FakeLlm(_llm_result(_queries_json("q1")))
    failed = ExtractedDoc(
        url="https://web.example/fail",
        text="",
        title=None,
        published_at=None,
        tier=TIER_NONE,
        ok=False,
    )
    news = _FakeSearch([])
    general = _FakeSearch([_result("https://web.example/fail")])
    extractor = _FakeExtractor({"https://web.example/fail": failed})
    settings = RetrievalSettings()

    bundle = _run(llm, news, general, extractor, settings)

    assert bundle.tier_stats == {TIER_NONE: 1}
