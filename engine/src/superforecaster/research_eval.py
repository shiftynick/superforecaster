"""A yardstick for research bundles — mechanical metrics plus LLM judgment.

ADR-0004's retrieval stack can swap providers and tiers (Perplexity Search API
first); this module is how such a change is adopted *on evidence* rather than by
feel. It offers three graduated measures of one :class:`ResearchBundle`:

1. **Mechanical metrics** (:func:`compute_metrics`) — pure, free, no LLM: counts,
   unique domains, extraction-tier stats, recency, key-term coverage,
   resolution-source presence, credits, and approximate size.
2. **A rubric judgment** (:func:`run_rubric_judge`) — one LLM call scoring five
   dimensions 1-5 plus an overall and the single most important gap, reading a
   **provider-blinded** rendering of the bundle so the judge grades content, not
   brand.
3. **A blinded pairwise verdict** (:func:`run_pairwise_judge`) — one LLM call
   comparing two bundles for the same question, with the X/Y assignment
   randomized per call and recorded so a later analysis can undo it.

Design notes (all reused house patterns, not reinvented):

- **Blinding.** The judge never sees a provider name. :func:`render_blinded_bundle`
  renders titles, domains, publish dates, and bodies (the same content the
  forecaster's ``_render_sources`` shows) but omits every result's ``provider``
  field and its raw provider payload. URLs/domains are kept — they are evidence of
  source quality and do not, in general, reveal which client fetched them. The
  retrieval config (providers/max_results) is recorded on the eval record for
  analysis but is *never* rendered into a judge prompt.
- **Structured output + degenerate guard + retry-once.** The judge calls carry a
  ``json_schema`` (advisory; the Claude CLI falls back to prompt-JSON), and parse
  failures — including a *degenerate* judgment whose justifications are stubs —
  are retried exactly once before raising, mirroring
  :func:`~superforecaster.forecaster.run_forecast_pass`.
- **Config-over-code.** The judge model/effort come from ``[tasks.bundle_judge]``,
  falling back to ``[tasks.forecast_pass]`` (a judgment call inherits the forecast
  model, not the cheap query-gen one).
- **Persistence.** :class:`BundleEvalRecord` is appended to a JSONL store with
  latest-wins accessors, exactly like every other store in the engine.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .config import RetrievalSettings, TaskModelConfig, load_task_config
from .forecaster import _DEGENERATE_RATIONALES, _call_cost, _extract_json_object
from .llm import LlmClient, LlmError, LlmResult, build_llm_client
from .models import QuestionSnapshot
from .pyramid_runner import _significant_terms
from .research import ResearchBundle

# Minimum characters a rubric justification (or the ``missing`` line) needs before
# it counts as real reasoning rather than a stub — mirrors the forecaster's
# ``_MIN_RATIONALE_CHARS`` guard, sized down for one-line justifications.
_MIN_JUSTIFICATION_CHARS = 10

# --- Mechanical metrics -------------------------------------------------------


class BundleMetrics(BaseModel):
    """Pure, LLM-free measurements of one research bundle.

    Attributes:
        result_count: Number of (deduplicated) search results in the bundle.
        doc_count: Number of content-extraction attempts (fetched URLs).
        doc_ok_count: How many of those extractions produced usable text.
        unique_domains: Distinct source domains across the results.
        tier_stats: Extraction outcomes per tier (copied from the bundle),
            including ``"none"`` for fetches that failed every tier.
        median_age_days: Median age (days, relative to ``now``) of the dated
            results' publish times, or ``None`` when none are dated.
        newest_age_days: Age (days) of the most recent dated result, or ``None``.
        undated_fraction: Fraction of results carrying no publish date.
        key_terms_total: Count of the question's significant terms considered.
        key_terms_covered: How many of those terms appear somewhere in the
            bundle's titles/snippets/doc text.
        key_term_coverage: ``key_terms_covered / key_terms_total`` (``1.0`` when
            the question has no significant terms — nothing to miss).
        resolution_source_present: True when at least one result's domain appears
            verbatim in the question's resolution criteria or fine print — a cheap
            proxy for "the source this will resolve by is in the bundle".
        search_credits_used: Metered search credits the bundle drew (from AskNews).
        approx_bundle_chars: Approximate rendered size (titles + snippets + doc
            bodies), a rough token/cost proxy.
    """

    result_count: int
    doc_count: int
    doc_ok_count: int
    unique_domains: int
    tier_stats: dict[str, int]
    median_age_days: float | None
    newest_age_days: float | None
    undated_fraction: float
    key_terms_total: int
    key_terms_covered: int
    key_term_coverage: float
    resolution_source_present: bool
    search_credits_used: int
    approx_bundle_chars: int


def _age_days(published_at: datetime, now: datetime) -> float:
    """Age of a publish time in days relative to ``now`` (naive times = UTC)."""
    when = published_at if published_at.tzinfo is not None else published_at.replace(tzinfo=UTC)
    return (now - when).total_seconds() / 86_400.0


def _question_key_terms(question: QuestionSnapshot) -> set[str]:
    """The question's significant terms, from its title + resolution criteria.

    Reuses the pyramid leaf-top-up heuristic's term logic (``_significant_terms``)
    so "what the question is about" is defined identically to how a leaf decides a
    bundle covers it — one notion of key terms across the engine.
    """
    return _significant_terms(f"{question.title} {question.resolution_criteria}")


def _bundle_haystack(bundle: ResearchBundle) -> str:
    """Lowercased concatenation of every result title/snippet and doc body."""
    parts: list[str] = []
    for result in bundle.results:
        parts.append(result.title)
        parts.append(result.snippet)
    for doc in bundle.docs:
        parts.append(doc.text)
    return " ".join(parts).lower()


def compute_metrics(
    question: QuestionSnapshot,
    bundle: ResearchBundle,
    *,
    now: datetime | None = None,
) -> BundleMetrics:
    """Compute the mechanical metrics for one bundle. Pure; makes no LLM call.

    Args:
        question: The question the bundle was gathered for (its text supplies the
            key terms and the resolution-source text to match domains against).
        bundle: The research bundle to measure.
        now: Reference time for recency ages (UTC); defaults to the current time.
    """
    moment = now if now is not None else datetime.now(UTC)

    results = bundle.results
    result_count = len(results)

    ages = [_age_days(r.published_at, moment) for r in results if r.published_at is not None]
    dated_count = len(ages)
    undated_fraction = (result_count - dated_count) / result_count if result_count else 0.0
    median_age_days = statistics.median(ages) if ages else None
    newest_age_days = min(ages) if ages else None

    domains = {r.source_domain for r in results if r.source_domain}

    terms = _question_key_terms(question)
    haystack = _bundle_haystack(bundle)
    covered = {term for term in terms if term in haystack}
    key_terms_total = len(terms)
    key_terms_covered = len(covered)
    key_term_coverage = key_terms_covered / key_terms_total if key_terms_total else 1.0

    criteria_text = f"{question.resolution_criteria}\n{question.fine_print}".lower()
    resolution_source_present = any(
        r.source_domain and r.source_domain.lower() in criteria_text for r in results
    )

    approx_bundle_chars = sum(len(r.title) + len(r.snippet) for r in results) + sum(
        len(doc.text) for doc in bundle.docs
    )

    return BundleMetrics(
        result_count=result_count,
        doc_count=len(bundle.docs),
        doc_ok_count=sum(1 for doc in bundle.docs if doc.ok),
        unique_domains=len(domains),
        tier_stats=dict(bundle.tier_stats),
        median_age_days=median_age_days,
        newest_age_days=newest_age_days,
        undated_fraction=undated_fraction,
        key_terms_total=key_terms_total,
        key_terms_covered=key_terms_covered,
        key_term_coverage=key_term_coverage,
        resolution_source_present=resolution_source_present,
        search_credits_used=bundle.search_credits_used,
        approx_bundle_chars=approx_bundle_chars,
    )


# --- Blinded rendering --------------------------------------------------------


def render_blinded_bundle(bundle: ResearchBundle) -> str:
    """Render a bundle for the judge with every provider identity stripped.

    Mirrors the forecaster's ``_render_sources`` — one ``[S<n>]`` entry per result
    with title, domain, publish date, and body (extracted doc text when we fetched
    it, else the snippet) — but deliberately omits each result's ``provider`` field
    and its raw provider payload. URLs and domains are kept (they are evidence of
    source quality); the provider that returned them is not, so a rubric or
    pairwise judge cannot grade on brand.
    """
    docs_by_url = {doc.url: doc for doc in bundle.docs if doc.ok and doc.text}
    lines = ["SOURCES"]
    for index, result in enumerate(bundle.results, start=1):
        title = result.title.strip() or "(untitled)"
        domain = result.source_domain or "unknown source"
        when = result.published_at.date().isoformat() if result.published_at else "date unknown"
        url = result.url or "(no url)"
        doc = docs_by_url.get(result.url)
        body = doc.text.strip() if doc is not None else result.snippet.strip()
        if not body:
            body = "(no content extracted)"
        lines.append(
            f"[S{index}] {title}\n    source: {domain} | url: {url} | published: {when}\n    {body}"
        )
    return "\n\n".join(lines)


def _question_block(question: QuestionSnapshot) -> str:
    """Render the question's decisive fields for a judge prompt."""

    def _section(label: str, value: str) -> str:
        text = value.strip() if value else ""
        return f"{label}:\n{text if text else '(none provided)'}"

    close = (
        question.scheduled_close_time.isoformat()
        if question.scheduled_close_time is not None
        else "(unknown)"
    )
    return "\n\n".join(
        [
            _section("QUESTION", question.title),
            _section("RESOLUTION CRITERIA", question.resolution_criteria),
            _section("FINE PRINT", question.fine_print),
            f"SCHEDULED CLOSE TIME: {close}",
        ]
    )


# --- Rubric judge -------------------------------------------------------------

DimensionScore = int  # 1-5, validated on the model below.


class RubricJudgment(BaseModel):
    """The rubric judge's structured verdict on one bundle.

    Each dimension carries a 1-5 score and a one-line justification; ``overall`` is
    a 1-5 summary and ``missing`` names the single most important thing the bundle
    lacks. See :data:`RUBRIC_SYSTEM` for what each dimension means.
    """

    coverage: DimensionScore = Field(ge=1, le=5)
    coverage_justification: str
    source_quality: DimensionScore = Field(ge=1, le=5)
    source_quality_justification: str
    recency: DimensionScore = Field(ge=1, le=5)
    recency_justification: str
    relevance: DimensionScore = Field(ge=1, le=5)
    relevance_justification: str
    resolution_grounding: DimensionScore = Field(ge=1, le=5)
    resolution_grounding_justification: str
    overall: DimensionScore = Field(ge=1, le=5)
    missing: str

    def justifications(self) -> list[str]:
        """The five per-dimension justification strings, in dimension order."""
        return [
            self.coverage_justification,
            self.source_quality_justification,
            self.recency_justification,
            self.relevance_justification,
            self.resolution_grounding_justification,
        ]


RUBRIC_SYSTEM = """\
You are grading the RESEARCH gathered to forecast a specific yes/no question — not
answering the question. You are shown the question and a numbered SOURCES block
(titles, domains, publish dates, and bodies). Judge ONLY the research's quality.

Score each dimension from 1 (poor) to 5 (excellent), with a one-line justification:

- coverage: are the question's core factors AND its resolution mechanics addressed
  by the sources?
- source_quality: are the sources authoritative/primary (official bodies, primary
  reporting) rather than content-farm or low-quality?
- recency: is the latest relevant development present, given the question's window?
- relevance: how much of the bundle is actually about THIS question versus noise?
- resolution_grounding: is the specific source the question will RESOLVE BY (the
  agency, dataset, court, or page that decides it) — or its data — present?

Then give an "overall" score (1-5) and "missing": one line naming the single most
important thing the bundle lacks.

Return ONLY a single JSON object and nothing else, of exactly this form:
{"coverage": 1-5, "coverage_justification": "...",
 "source_quality": 1-5, "source_quality_justification": "...",
 "recency": 1-5, "recency_justification": "...",
 "relevance": 1-5, "relevance_justification": "...",
 "resolution_grounding": 1-5, "resolution_grounding_justification": "...",
 "overall": 1-5, "missing": "..."}"""


class DegenerateBundleJudgment(ValueError):
    """A rubric judgment parsed but its justifications/missing are stubs, not reasoning."""


def _is_stub(text: str) -> bool:
    """True when a one-line judgment field is empty, too short, or placeholder."""
    stripped = text.strip().lower()
    return len(stripped) < _MIN_JUSTIFICATION_CHARS or stripped in _DEGENERATE_RATIONALES


def _parse_rubric(text: str) -> RubricJudgment:
    """Extract, decode, validate, and guard the rubric JSON from a judge reply.

    Raises:
        json.JSONDecodeError: No JSON object is present or it is malformed.
        pydantic.ValidationError: The JSON is the wrong shape or a score is out of
            the 1-5 range.
        DegenerateBundleJudgment: The shape is right but every dimension
            justification is a stub, or the ``missing`` line is a stub — retried
            like a parse failure, because scores without reasons are unauditable.
    """
    raw = _extract_json_object(text)
    data = json.loads(raw)
    judgment = RubricJudgment.model_validate(data)
    if all(_is_stub(j) for j in judgment.justifications()) or _is_stub(judgment.missing):
        raise DegenerateBundleJudgment(
            "rubric justifications/missing are stubs; explain each score and the gap"
        )
    return judgment


@dataclass(frozen=True)
class RubricResult:
    """A parsed rubric judgment plus the LLM bookkeeping the call incurred."""

    judgment: RubricJudgment
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


def _rubric_prompt(question: QuestionSnapshot, bundle: ResearchBundle) -> str:
    """Assemble the rubric judge's user message: the question + blinded bundle."""
    return "\n\n".join(
        [
            _question_block(question),
            render_blinded_bundle(bundle),
            "Grade the research now and return the JSON object.",
        ]
    )


def _judge_retry_note(error: Exception) -> str:
    """The addendum demanding a clean judgment object after a parse failure."""
    return (
        "\n\nYour previous reply could not be used. The error was: "
        f"{error}\nReply again with ONLY the single JSON object described above — "
        "every field present, each score an integer 1-5, and a real one-line "
        "justification for each (no placeholders), nothing after the closing brace."
    )


def _tokens_and_cost(results: list[LlmResult], model: str) -> tuple[int, int, float | None]:
    """Sum tokens across calls and total their cost (``None`` only if all unpriced)."""
    input_tokens = sum(r.input_tokens for r in results)
    output_tokens = sum(r.output_tokens for r in results)
    per_call = [_call_cost(r, model) for r in results]
    cost = (
        sum(c for c in per_call if c is not None) if any(c is not None for c in per_call) else None
    )
    return input_tokens, output_tokens, cost


def run_rubric_judge(
    question: QuestionSnapshot,
    bundle: ResearchBundle,
    *,
    client: LlmClient | None = None,
    config: TaskModelConfig | None = None,
) -> RubricResult:
    """Score one bundle with the rubric judge, retrying a bad parse exactly once.

    Structured output is prompted (schema is advisory): the judge ends with a JSON
    object we extract, validate, and guard against degeneracy. On a parse,
    validation, or degeneracy failure we retry once with the error appended; a
    second failure raises :class:`LlmError` — the same shape as the forecaster.

    Args:
        question: The question the bundle serves.
        bundle: The research bundle to grade (rendered blinded for the judge).
        client: Injection seam — any object with a matching ``.run(...)``.
            Defaults to the client for the judge task's provider. Tests pass a fake.
        config: Judge model settings; defaults to :func:`load_bundle_judge_config`.

    Raises:
        LlmError: The judge did not return a valid rubric object after one retry.
    """
    cfg = config if config is not None else load_bundle_judge_config()
    llm = client if client is not None else build_llm_client(cfg)
    user = _rubric_prompt(question, bundle)
    schema = RubricJudgment.model_json_schema()

    results: list[LlmResult] = []
    first = llm.run(
        RUBRIC_SYSTEM, user, cfg.model, cfg.max_tokens, json_schema=schema, effort=cfg.effort
    )
    results.append(first)
    try:
        judgment = _parse_rubric(first.text)
    except (json.JSONDecodeError, ValidationError, DegenerateBundleJudgment) as first_error:
        retry = user + _judge_retry_note(first_error)
        second = llm.run(
            RUBRIC_SYSTEM, retry, cfg.model, cfg.max_tokens, json_schema=schema, effort=cfg.effort
        )
        results.append(second)
        try:
            judgment = _parse_rubric(second.text)
        except (json.JSONDecodeError, ValidationError, DegenerateBundleJudgment) as second_error:
            raise LlmError(
                f"Rubric judge did not return a valid judgment after one retry: {second_error}"
            ) from second_error

    input_tokens, output_tokens, cost = _tokens_and_cost(results, cfg.model)
    return RubricResult(
        judgment=judgment,
        model=cfg.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )


# --- Pairwise judge -----------------------------------------------------------

PairwiseWinner = Literal["X", "Y", "tie"]
_VALID_WINNERS: frozenset[str] = frozenset({"X", "Y", "tie"})


class PairwiseVerdict(BaseModel):
    """The pairwise judge's per-dimension and overall winner, in X/Y terms.

    Every field is one of ``"X"``, ``"Y"``, or ``"tie"``. The X/Y-to-bundle
    mapping is randomized per call and recorded separately (see
    :class:`PairwiseRecord`), so these labels are meaningless without it — which is
    the point: the judge cannot know which bundle is the "new" one.
    """

    coverage: PairwiseWinner
    source_quality: PairwiseWinner
    recency: PairwiseWinner
    relevance: PairwiseWinner
    resolution_grounding: PairwiseWinner
    overall: PairwiseWinner
    reason: str


PAIRWISE_SYSTEM = """\
You are comparing TWO sets of research (labeled X and Y) gathered to forecast the
SAME yes/no question. Judge the research quality only — do not answer the question.

For each dimension, say which set's research is better — answer "X", "Y", or "tie":

- coverage: the question's core factors and resolution mechanics addressed.
- source_quality: authoritative/primary sources over content-farm/low-quality.
- recency: the latest relevant development present, given the question's window.
- relevance: how much of the set is actually about THIS question.
- resolution_grounding: the specific source the question RESOLVES BY (or its data)
  present.

Then give the "overall" winner ("X", "Y", or "tie") and a one-line "reason".

Return ONLY a single JSON object and nothing else, of exactly this form:
{"coverage": "X"|"Y"|"tie", "source_quality": "X"|"Y"|"tie",
 "recency": "X"|"Y"|"tie", "relevance": "X"|"Y"|"tie",
 "resolution_grounding": "X"|"Y"|"tie", "overall": "X"|"Y"|"tie",
 "reason": "..."}"""


def _norm_winner(value: object) -> PairwiseWinner:
    """Normalize a model's winner token to ``"X"``/``"Y"``/``"tie"`` or raise."""
    text = str(value).strip().lower()
    if text == "x":
        return "X"
    if text == "y":
        return "Y"
    if text in ("tie", "draw", "equal", "same"):
        return "tie"
    raise ValueError(f"pairwise winner must be X/Y/tie, got {value!r}")


def _parse_pairwise(text: str) -> PairwiseVerdict:
    """Extract, decode, and validate the pairwise JSON from a judge reply.

    Raises:
        json.JSONDecodeError: No JSON object is present or it is malformed.
        ValueError: A required winner field is missing or not X/Y/tie.
    """
    data = json.loads(_extract_json_object(text))
    if not isinstance(data, dict):
        raise ValueError("pairwise JSON was not an object")
    dims = ("coverage", "source_quality", "recency", "relevance", "resolution_grounding", "overall")
    winners = {dim: _norm_winner(data.get(dim)) for dim in dims}
    reason_raw = data.get("reason", "")
    reason = reason_raw.strip() if isinstance(reason_raw, str) else ""
    return PairwiseVerdict(reason=reason, **winners)


@dataclass(frozen=True)
class PairwiseResult:
    """A parsed pairwise verdict, the X/Y mapping, and the call's bookkeeping.

    Attributes:
        verdict: The winners in X/Y terms as the judge returned them.
        x_source: Which input bundle was shown as X — ``"a"`` or ``"b"`` (the other
            was Y). This is the recorded randomization mapping; without it the
            verdict's X/Y labels cannot be tied back to a bundle.
        model, input_tokens, output_tokens, cost_usd: LLM bookkeeping.
    """

    verdict: PairwiseVerdict
    x_source: Literal["a", "b"]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


def _pairwise_prompt(
    question: QuestionSnapshot, x_bundle: ResearchBundle, y_bundle: ResearchBundle
) -> str:
    """Assemble the pairwise user message: the question + both blinded bundles."""
    return "\n\n".join(
        [
            _question_block(question),
            "RESEARCH SET X\n\n" + render_blinded_bundle(x_bundle),
            "RESEARCH SET Y\n\n" + render_blinded_bundle(y_bundle),
            "Compare the two research sets now and return the JSON object.",
        ]
    )


def run_pairwise_judge(
    question: QuestionSnapshot,
    bundle_a: ResearchBundle,
    bundle_b: ResearchBundle,
    *,
    client: LlmClient | None = None,
    config: TaskModelConfig | None = None,
    rng: random.Random | None = None,
    now: datetime | None = None,
) -> PairwiseResult:
    """Compare two bundles blinded, with X/Y randomized and recorded.

    Bundle A and B are randomly assigned to labels X and Y (so the judge cannot
    tell which is the incumbent), rendered blinded, and compared in one LLM call.
    The X/Y-to-bundle mapping is returned on the result. A parse/validation failure
    is retried exactly once before raising, matching the rubric judge.

    Args:
        question: The question both bundles serve.
        bundle_a: The first bundle (e.g. current-settings research).
        bundle_b: The second bundle (e.g. alt-config research).
        client: Injection seam; defaults to the judge task's client.
        config: Judge model settings; defaults to :func:`load_bundle_judge_config`.
        rng: Randomness source for the X/Y assignment; defaults to a
            :class:`random.Random` seeded from ``now`` (engine code, not a
            reproducibility-critical script — a timestamp seed is fine per spec).
        now: Timestamp used to seed the default RNG (UTC); defaults to now.

    Raises:
        LlmError: The judge did not return a valid verdict after one retry.
    """
    cfg = config if config is not None else load_bundle_judge_config()
    llm = client if client is not None else build_llm_client(cfg)
    moment = now if now is not None else datetime.now(UTC)
    generator = rng if rng is not None else random.Random(int(moment.timestamp()))

    x_is_a = generator.random() < 0.5
    x_source: Literal["a", "b"] = "a" if x_is_a else "b"
    x_bundle, y_bundle = (bundle_a, bundle_b) if x_is_a else (bundle_b, bundle_a)

    user = _pairwise_prompt(question, x_bundle, y_bundle)
    schema = PairwiseVerdict.model_json_schema()

    results: list[LlmResult] = []
    first = llm.run(
        PAIRWISE_SYSTEM, user, cfg.model, cfg.max_tokens, json_schema=schema, effort=cfg.effort
    )
    results.append(first)
    try:
        verdict = _parse_pairwise(first.text)
    except (json.JSONDecodeError, ValueError, ValidationError) as first_error:
        retry = user + _judge_retry_note(first_error)
        second = llm.run(
            PAIRWISE_SYSTEM, retry, cfg.model, cfg.max_tokens, json_schema=schema, effort=cfg.effort
        )
        results.append(second)
        try:
            verdict = _parse_pairwise(second.text)
        except (json.JSONDecodeError, ValueError, ValidationError) as second_error:
            raise LlmError(
                f"Pairwise judge did not return a valid verdict after one retry: {second_error}"
            ) from second_error

    input_tokens, output_tokens, cost = _tokens_and_cost(results, cfg.model)
    return PairwiseResult(
        verdict=verdict,
        x_source=x_source,
        model=cfg.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    )


# --- Fingerprint + retrieval summary ------------------------------------------


def bundle_fingerprint(bundle: ResearchBundle) -> str:
    """A stable content hash of a bundle: sorted result URLs + sorted doc URLs.

    Two bundles with the same set of source and fetched-document URLs fingerprint
    identically regardless of ordering, so an eval can be tied to exactly the
    research it graded. Returns a hex SHA-256 digest.
    """
    result_urls = sorted(r.url for r in bundle.results)
    doc_urls = sorted(doc.url for doc in bundle.docs)
    payload = "\n".join(["results", *result_urls, "docs", *doc_urls])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RetrievalSummary(BaseModel):
    """The retrieval config a bundle was built under — for analysis, never judged.

    Recorded on the eval record so a provider comparison can be sliced by config,
    but deliberately kept out of every judge prompt (the judge grades blinded).
    """

    news_provider: str
    general_provider: str
    max_results: int


def summarize_retrieval(settings: RetrievalSettings) -> RetrievalSummary:
    """Extract the analysis-relevant retrieval fields from full settings."""
    return RetrievalSummary(
        news_provider=settings.news_provider,
        general_provider=settings.general_provider,
        max_results=settings.max_results,
    )


# --- Persistence --------------------------------------------------------------


class PairwiseRecord(BaseModel):
    """A pairwise verdict plus everything needed to interpret its X/Y labels.

    Attributes:
        verdict: The winners in X/Y terms.
        x_source: Whether THIS record's own bundle was shown as X (``"self"``) or
            the other bundle was (``"other"``) — the randomization mapping.
        other_fingerprint: Fingerprint of the bundle compared against.
        other_retrieval: The other bundle's retrieval config (for analysis).
    """

    verdict: PairwiseVerdict
    x_source: Literal["self", "other"]
    other_fingerprint: str
    other_retrieval: RetrievalSummary


class BundleEvalRecord(BaseModel):
    """One stored evaluation of a research bundle — metrics, and optional judgments.

    A ``kind="rubric"`` record carries the mechanical metrics plus the rubric
    judgment; a ``kind="pairwise"`` record carries the metrics plus a
    :class:`PairwiseRecord` (its verdict attaches to the current-settings bundle,
    referencing the other). The retrieval summary is recorded for analysis and is
    never shown to any judge. Token/cost fields satisfy the cost-visibility rule.
    """

    created_at: datetime
    question_id: int
    post_id: int
    title: str
    kind: Literal["rubric", "pairwise"]
    bundle_fingerprint: str
    retrieval: RetrievalSummary
    metrics: BundleMetrics
    rubric: RubricJudgment | None = None
    pairwise: PairwiseRecord | None = None
    judge_model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class BundleEvalStore:
    """Append-only JSON Lines store of :class:`BundleEvalRecord` records.

    All records live in one ``bundle-evals.jsonl`` file (evals are low-volume,
    like web resolutions). Appending never rewrites history;
    :meth:`latest_by_question` gives the newest eval per question.

    Args:
        dir: Directory holding ``bundle-evals.jsonl``. Created on first write.
    """

    _FILENAME = "bundle-evals.jsonl"

    def __init__(self, dir: Path) -> None:
        self._dir = Path(dir)

    @property
    def dir(self) -> Path:
        """The directory this store reads from and writes to."""
        return self._dir

    @property
    def path(self) -> Path:
        """The single JSONL file backing this store."""
        return self._dir / self._FILENAME

    def append(self, records: Iterable[BundleEvalRecord]) -> int:
        """Append records to ``bundle-evals.jsonl``; return how many were written."""
        lines = [r.model_dump_json() for r in records]
        if not lines:
            return 0
        self._dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line)
                fh.write("\n")
        return len(lines)

    def iter_all(self) -> Iterator[BundleEvalRecord]:
        """Yield every stored eval, in file (append) order."""
        path = self.path
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                yield BundleEvalRecord.model_validate_json(line)

    def latest_by_question(self) -> dict[int, BundleEvalRecord]:
        """Return the newest eval (by ``created_at``) per question id."""
        latest: dict[int, BundleEvalRecord] = {}
        for record in self.iter_all():
            current = latest.get(record.question_id)
            if current is None or record.created_at >= current.created_at:
                latest[record.question_id] = record
        return latest


# --- Config plumbing ----------------------------------------------------------


def load_bundle_judge_config() -> TaskModelConfig:
    """The bundle-judge config: ``[tasks.bundle_judge]``, else ``forecast_pass``.

    Grading a bundle is a judgment call, so when ``[tasks.bundle_judge]`` is absent
    the judge inherits the FORECAST model (``[tasks.forecast_pass]``), not the
    cheap query-gen one — same fallback shape as the pyramid's holistic/verdict
    loaders (``load_task_config`` raises ``KeyError`` when a table is missing).
    """
    try:
        return load_task_config("bundle_judge")
    except KeyError:
        return load_task_config("forecast_pass")
