"""The research pipeline — turn one question into a frozen *research bundle*.

This is slice 2 of the retrieval stack. Given a question it: asks a cheap LLM
for a handful of search queries (one aimed at the page that will actually decide
the question's resolution), runs those queries against both a *news* client and a
*general web* client, dedupes and content-extracts the most useful hits, and
packs the whole thing into one :class:`ResearchBundle`.

That bundle is stored verbatim inside each forecast's trace (ADR-0004's
"frozen context"): a forecast made from it is leakage-proof by construction, and
a reviewer can see exactly which sources the model was shown and what they cost.

Design notes:

- **Config over code.** Search budgets, fetch counts, and text caps all come
  from :class:`~superforecaster.config.RetrievalSettings`; nothing is a literal.
- **Graceful degradation.** The metered news client can run out of monthly
  credits mid-run; a :class:`~superforecaster.search.SearchQuotaError` is caught,
  flagged on the bundle (``news_quota_exhausted``), and research continues on the
  general client alone rather than failing the forecast.
- **Reviewable records.** Provider responses can carry huge raw page dumps and
  extracted bodies can be long; both are trimmed before storage so a forecast
  JSONL line stays something a human can actually read.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime

from pydantic import BaseModel

from .config import RetrievalSettings, TaskModelConfig, load_task_config, price_for
from .extract import ExtractedDoc, Extractor
from .llm import LlmClient, LlmError, LlmResult
from .models import QuestionSnapshot
from .search import SearchClient, SearchQuotaError, SearchResult

# Tavily can return a whole page dump in `raw_content`; keep at most this many
# characters of it in the stored result so records stay reviewable.
_MAX_RAW_CONTENT_CHARS = 2000

# A search hit's snippet counts as "usable" (worth *not* fetching the body for)
# once it is at least this long. AskNews summaries clear this comfortably; Tavily
# excerpts usually do not — so the fetch budget is spent where it helps most.
_SNIPPET_USABLE_MIN_CHARS = 200


class ResearchBundle(BaseModel):
    """Everything gathered for one question, frozen for the forecast trace.

    Attributes:
        queries: The search queries the LLM generated for this question.
        results: Deduplicated, normalized search hits (news + general), with any
            oversized ``raw`` page dump trimmed.
        docs: Content-extracted bodies for the fetched subset of ``results``
            (text truncated to ``max_doc_chars``).
        tier_stats: Count of extracted docs per extraction tier, including
            ``"none"`` for fetches that failed every tier.
        search_credits_used: Metered search credits (AskNews) drawn this run.
        news_quota_exhausted: True when the news client hit its monthly cap and
            research fell back to the general client only.
        retrieved_at: When this bundle was assembled (UTC).
        input_tokens: Prompt tokens spent on query generation (summed over any
            retry).
        output_tokens: Completion tokens spent on query generation.
        cost_usd: Query-generation LLM cost, or ``None`` if the provider gave no
            cost and the model is unpriced. This is the *query-gen* cost only;
            the forecast pass's own cost is recorded separately on the record.
    """

    queries: list[str]
    results: list[SearchResult]
    docs: list[ExtractedDoc]
    tier_stats: dict[str, int]
    search_credits_used: int = 0
    news_quota_exhausted: bool = False
    retrieved_at: datetime
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


# The query-gen system prompt. Read it as the spec for what a good query set is:
# specific, multi-angle, and grounded against the actual resolution source.
QUERY_GEN_SYSTEM = """\
You generate web-search queries that will surface the evidence needed to \
forecast a specific yes/no question. Given the question and its resolution \
criteria, produce 2 to 4 focused search queries.

Rules:
- Make each query specific and keyword-rich — names, places, dates, numbers, \
institutions — not a vague restatement of the question.
- Cover different angles across the set: recent news and developments, plus \
relevant background or base-rate context.
- EXACTLY ONE query must target the question's RESOLUTION SOURCE: the specific \
page, institution, dataset, official body, or website that will actually decide \
how this question resolves (for example the agency that publishes the figure, \
or the site that posts the official ruling). Aim that query at finding that \
source's own latest statement on the matter.

Return ONLY a single JSON object and nothing else, of exactly this form:
{"queries": ["first query", "second query", "third query"]}"""

# Advisory schema for providers that support server-side structured output
# (Codex CLI). The Claude CLI ignores it and we fall back to prompt-JSON + retry.
_QUERY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}


def _build_query_gen_prompt(question: QuestionSnapshot) -> str:
    """Assemble the query-gen user message from the question's key fields."""

    def _section(label: str, value: str) -> str:
        text = value.strip() if value else ""
        return f"{label}:\n{text if text else '(none provided)'}"

    parts = [
        _section("QUESTION", question.title),
        _section("DESCRIPTION", question.description),
        _section("RESOLUTION CRITERIA", question.resolution_criteria),
        "Return the JSON object of search queries now.",
    ]
    return "\n\n".join(parts)


def _parse_queries(text: str) -> list[str]:
    """Extract and validate the ``queries`` list from the model's reply.

    Reuses the forecaster's JSON extractor (imported lazily to avoid an import
    cycle: the forecaster imports this module for the bundle type). Deduplicates,
    drops blanks, and caps the set at four.

    Raises:
        json.JSONDecodeError: No JSON object could be located in the reply.
        ValueError: The JSON was the wrong shape or held no usable query.
    """
    from .forecaster import _extract_json_object

    raw = _extract_json_object(text)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("query-gen JSON was not an object")
    queries = data.get("queries")
    if not isinstance(queries, list):
        raise ValueError("query-gen JSON had no 'queries' list")
    cleaned: list[str] = []
    for item in queries:
        if isinstance(item, str) and item.strip():
            candidate = item.strip()
            if candidate not in cleaned:
                cleaned.append(candidate)
    if not cleaned:
        raise ValueError("query-gen returned no usable queries")
    return cleaned[:4]


def _query_retry_note(error: Exception) -> str:
    """The addendum demanding a clean queries object after a parse failure."""
    return (
        "\n\nYour previous reply could not be parsed. The error was: "
        f"{error}\nReply again with ONLY a single JSON object of the form "
        '{"queries": ["...", "..."]} — 2 to 4 queries, nothing before or after '
        "the object."
    )


def generate_queries(
    question: QuestionSnapshot,
    llm: LlmClient,
    config: TaskModelConfig,
) -> tuple[list[str], list[LlmResult]]:
    """Ask the LLM for this question's search queries, retrying parse once.

    Mirrors the forecaster's retry-once pattern: on a parse/validation failure we
    re-prompt with the error and demand a clean object; a second failure raises.

    Returns:
        The query list, plus the raw :class:`LlmResult` calls made (one, or two
        on a retry) so the caller can sum tokens and cost for the bundle.

    Raises:
        LlmError: The model did not return a valid queries object after one retry.
    """
    user = _build_query_gen_prompt(question)
    calls: list[LlmResult] = []

    first = llm.run(
        QUERY_GEN_SYSTEM,
        user,
        config.model,
        config.max_tokens,
        json_schema=_QUERY_SCHEMA,
        effort=config.effort,
    )
    calls.append(first)
    try:
        return _parse_queries(first.text), calls
    except (json.JSONDecodeError, ValueError) as first_error:
        retry = user + _query_retry_note(first_error)
        second = llm.run(
            QUERY_GEN_SYSTEM,
            retry,
            config.model,
            config.max_tokens,
            json_schema=_QUERY_SCHEMA,
            effort=config.effort,
        )
        calls.append(second)
        try:
            return _parse_queries(second.text), calls
        except (json.JSONDecodeError, ValueError) as second_error:
            raise LlmError(
                f"Query generation did not return a valid queries object after one "
                f"retry: {second_error}"
            ) from second_error


def _dedupe_by_url(results: list[SearchResult]) -> list[SearchResult]:
    """Return results with duplicate URLs removed, first occurrence kept."""
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for result in results:
        if result.url and result.url not in seen:
            seen.add(result.url)
            deduped.append(result)
    return deduped


def _trim_raw_content(result: SearchResult) -> SearchResult:
    """Truncate an oversized ``raw_content`` page dump in a result's ``raw``."""
    raw_content = result.raw.get("raw_content")
    if isinstance(raw_content, str) and len(raw_content) > _MAX_RAW_CONTENT_CHARS:
        new_raw = dict(result.raw)
        new_raw["raw_content"] = raw_content[:_MAX_RAW_CONTENT_CHARS]
        return result.model_copy(update={"raw": new_raw})
    return result


def _has_usable_snippet(result: SearchResult) -> bool:
    """True when a result's snippet is substantial enough to skip fetching."""
    return len(result.snippet.strip()) >= _SNIPPET_USABLE_MIN_CHARS


def _select_fetch_urls(results: list[SearchResult], max_fetches: int) -> list[str]:
    """Pick which URLs to content-extract, thin-snippet results first.

    A stable sort puts results lacking a usable snippet ahead of ones that
    already carry a full summary, then we take the first ``max_fetches`` — so the
    limited fetch budget lands on the hits that most need their body pulled.
    """
    ordered = sorted(results, key=_has_usable_snippet)
    return [result.url for result in ordered[:max_fetches]]


def _truncate_doc(doc: ExtractedDoc, max_chars: int) -> ExtractedDoc:
    """Return the doc with its text truncated to ``max_chars`` if longer."""
    if len(doc.text) > max_chars:
        return doc.model_copy(update={"text": doc.text[:max_chars]})
    return doc


def run_research(
    question: QuestionSnapshot,
    llm: LlmClient,
    news_client: SearchClient,
    general_client: SearchClient,
    extractor: Extractor,
    settings: RetrievalSettings,
    *,
    query_gen_config: TaskModelConfig | None = None,
    now: datetime | None = None,
) -> ResearchBundle:
    """Run the full research pipeline for one question and return its bundle.

    Steps: generate queries -> run each against the news and general clients
    (respecting ``settings.max_results``) -> dedupe by URL -> trim oversized raw
    page dumps -> fetch and extract the top ``settings.max_fetches_per_question``
    URLs (thin-snippet hits first) -> assemble the bundle. A news-client quota
    exhaustion is caught and flagged, not fatal.

    Args:
        question: The question to research.
        llm: Client for the one query-generation call.
        news_client: The metered *news* search client (e.g. AskNews).
        general_client: The *general web* search client (e.g. Tavily).
        extractor: The tiered content extractor.
        settings: Retrieval budgets and text caps.
        query_gen_config: Model config for query generation; defaults to the
            ``[tasks.query_gen]`` task config. Injectable for tests.
        now: Bundle timestamp (UTC); defaults to the current time.

    Returns:
        The assembled :class:`ResearchBundle`.

    Raises:
        LlmError: Query generation failed to parse after one retry.
    """
    moment = now if now is not None else datetime.now(UTC)
    config = query_gen_config if query_gen_config is not None else load_task_config("query_gen")

    queries, calls = generate_queries(question, llm, config)

    credits_before = _credits_used(news_client)
    news_quota_exhausted = False
    raw_results: list[SearchResult] = []
    for query in queries:
        if not news_quota_exhausted:
            try:
                raw_results.extend(news_client.search(query, max_results=settings.max_results))
            except SearchQuotaError:
                # Out of monthly credits: flag it and carry on with general only.
                news_quota_exhausted = True
        raw_results.extend(general_client.search(query, max_results=settings.max_results))
    credits_after = _credits_used(news_client)

    results = [_trim_raw_content(r) for r in _dedupe_by_url(raw_results)]

    fetch_urls = _select_fetch_urls(results, settings.max_fetches_per_question)
    docs = [_truncate_doc(extractor.extract(url), settings.max_doc_chars) for url in fetch_urls]
    tier_stats = dict(Counter(doc.tier for doc in docs))

    search_credits_used = (
        credits_after - credits_before
        if credits_before is not None and credits_after is not None
        else 0
    )

    return ResearchBundle(
        queries=queries,
        results=results,
        docs=docs,
        tier_stats=tier_stats,
        search_credits_used=max(0, search_credits_used),
        news_quota_exhausted=news_quota_exhausted,
        retrieved_at=moment,
        input_tokens=sum(c.input_tokens for c in calls),
        output_tokens=sum(c.output_tokens for c in calls),
        cost_usd=_sum_cost(calls, config.model),
    )


def _credits_used(client: SearchClient) -> int | None:
    """Read a metered client's month-to-date credits, or ``None`` if unmetered.

    Only the news client (AskNews) tracks credits; the general client has no such
    method, so ``getattr`` returns ``None`` and the run records zero draw for it.
    """
    getter = getattr(client, "credits_used_this_month", None)
    if callable(getter):
        try:
            return int(getter())
        except (TypeError, ValueError):
            return None
    return None


def _sum_cost(calls: list[LlmResult], model: str) -> float | None:
    """Sum per-call costs, falling back to priced tokens when the CLI gave none.

    Mirrors the forecaster's cost accounting: use the provider-reported cost when
    present, else price the call's tokens from the config table. ``None`` only
    when *every* call is both cost-free and unpriced.
    """
    per_call = [
        c.cost_usd if c.cost_usd is not None else price_for(model, c.input_tokens, c.output_tokens)
        for c in calls
    ]
    known = [c for c in per_call if c is not None]
    return sum(known) if known else None
