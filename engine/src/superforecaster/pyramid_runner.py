"""Candidate C — the decomposition-pyramid *runner*: LLM calls + orchestration.

Where :mod:`superforecaster.pyramid` is the pure, LLM-free half (schema, parser,
combination math), this module is *stage 2*: it drives the actual forecast.
:func:`run_pyramid_forecast` orchestrates one question end to end —

1. **Decompose** (one LLM call) the question into a typed tree, retrying once on a
   parse failure and, if that also fails, degrading to a single careful leaf.
2. **Estimate** every bedrock leaf with the existing :func:`run_forecast_pass`
   machinery — ``evidence`` leaves see the shared research bundle, ``model-
   knowledge`` leaves do not (spec §2). A leaf whose own pass fails degrades to
   ``0.5`` / ``"low"`` rather than crashing the whole forecast.
3. **Combine** the leaves upward (pure math from ``pyramid.py``) into a
   *structural* probability plus a per-node trace.
4. **Holistic** — one direct forecast of the root question, alongside the tree.
5. **Blend** structural and holistic in log-odds space into the final probability.
6. **Confidence** from three raw signals (weakest leaf confidence on a multiplied
   path; the structural-vs-holistic log-odds gap; the count of unverified
   load-bearing assumptions), all stored, the label being their conservative
   minimum.

The whole legible argument graph is persisted on one :class:`ForecastRecord`
(``method="pyramid-c"``) via its :class:`PyramidDetail` — the same "rich detail on
the single stored record" pattern the ensemble uses for :class:`EnsembleDetail`.

Guiding principle throughout: **degrade, never crash.** A failed decomposition, a
failed leaf, or a failed holistic pass each fall back to a documented default and
leave a note in the trace; the pyramid always returns a record.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import ValidationError

from .aggregation import to_log_odds
from .config import (
    EnsembleSettings,
    PyramidSettings,
    RetrievalSettings,
    TaskModelConfig,
    load_ensemble_settings,
    load_pyramid_settings,
    load_retrieval_settings,
    load_task_config,
)
from .forecaster import (
    Confidence,
    ForecastRecord,
    LeafEstimate,
    PyramidDetail,
    _build_user_prompt,
    _extract_json_object,
    run_forecast_pass,
)
from .llm import LlmClient, LlmError, LlmResult, build_llm_client
from .models import QuestionSnapshot
from .pyramid import (
    Assumption,
    AssumptionStatus,
    PruneNote,
    PyramidNode,
    PyramidParseError,
    blend,
    combine_tree,
    decomposition_json_schema,
    parse_decomposition,
)
from .research import ResearchBundle
from .search import SearchClient, SearchError, SearchResult

# Same Metaculus-driven clamp the single pass and the ensemble apply: a stored
# probability of exactly 0 or 1 is rejected, so the blended final is squeezed
# into this closed interval (``raw_probability`` keeps the unclamped value).
_MIN_PROB = 0.01
_MAX_PROB = 0.99

# The multiplying node types — the shapes exposed to conjunctive error
# compounding. Confidence signal (a) looks at the leaves under one of these.
_MULTIPLYING_TYPES: frozenset[str] = frozenset({"conjunction", "conditional"})

# Confidence ordering, weakest first, so ``min`` picks the more cautious label.
_CONFIDENCE_ORDER: dict[Confidence, int] = {"low": 0, "medium": 1, "high": 2}


# --- Decomposition prompt -----------------------------------------------------

# The decomposer's system prompt. It states the structural caps (filled from
# config, never hard-coded) and the two labels every node must carry, then asks
# for the tree as one JSON object. Read it as the spec for *how* to decompose.
DECOMPOSE_PROMPT = """\
You are a careful superforecaster building a DECOMPOSITION TREE for a question.
Today's date is {today}. Your job is NOT to answer the question yet — it is to
break it into the sub-questions that must be answered first, down to a bedrock of
researchable facts and base rates.

Return a tree of nodes. Each node has:
- "id": a short unique string within the tree (e.g. "root", "a", "b1").
- "question": the sub-question in plain words.
- "node_type": one of
    - "conjunction": resolves YES only if ALL its children do (probabilities
      multiply — use this sparingly; multiplying several uncertain sub-events
      drives the product too low).
    - "disjunction": resolves YES if ANY child does.
    - "conditional": an ordered chain — each child is phrased "given the previous
      steps happened".
    - "leaf": bedrock — a base rate, a researchable fact, or a direct estimate.
      A leaf has NO children.
- "reasoning_type": leave as "checklist" unless you have a specific reason.
- "grounding": REQUIRED on every leaf, one of:
    - "model-knowledge": a stable fact or base rate you can answer confidently
      WITHOUT looking anything up (it will be answered without the research).
    - "evidence": concerns current events and MUST be grounded in the provided
      research.
  Internal (non-leaf) nodes MUST NOT carry a grounding.
- "assumptions": OPTIONAL list of the implicit, near-certain beliefs THIS node's
  estimate silently rests on — the things you are quietly treating as true (for
  example "the vote happens on schedule", "the agency measures what we think it
  measures"). Each assumption is an object:
    - "text": the assumption in plain, falsifiable words.
    - "load_bearing": true if the node's estimate would change materially were
      this assumption false, otherwise false.
    - "verifiability": "researchable" if a web search could settle it now, or
      "unverifiable" if it cannot be checked yet.
  Surface EVERY assumption the estimate genuinely rests on — typically 1 to 4 —
  but do NOT manufacture filler. Zero assumptions is a legitimate answer.
- "children": the list of sub-nodes (empty for a leaf).

STRUCTURAL LIMITS (a tree exceeding them will be pruned):
- Maximum depth: {max_depth} (the root is depth 0).
- At most {max_children} children per node.
- At most {max_conjuncts} multiplied children on any conjunction or conditional.

Keep the tree SHALLOW and HONEST. If the question does not usefully decompose,
return a SINGLE leaf node — that is a legitimate answer. Prefer fewer, sturdier
sub-questions over a sprawling tree of guesses.

CRITICAL: every leaf question's wording must stay consistent with the ROOT
question's resolution criteria and fine print — carry the exact time windows,
resolution sources, and edge-case rules into the leaf wording. A leaf that
silently widens a time window or swaps the resolution source answers a different
question and poisons the whole tree."""

# Appended so the decomposer ends with a machine-readable tree. Mirrors the
# forecaster's OUTPUT_FORMAT convention: prose first, then one JSON object.
DECOMPOSE_OUTPUT_FORMAT = """\
OUTPUT FORMAT

Think first, then END your response with ONLY a single JSON object — the ROOT
node — and nothing after it. You may wrap it in a ```json code fence. Every node
must have "id", "question", "node_type", and "children"; every leaf must also
have "grounding". A node may include an "assumptions" list (objects with "text",
"load_bearing", "verifiability"); omit it when the node rests on none. Output
nothing after the closing brace."""


class _CallCost:
    """A tiny running total of tokens and cost across every LLM call made.

    Cost is ``None`` only if *no* call reported one (matching the single pass and
    ensemble convention); otherwise it is the sum of the calls that did.
    """

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self._costs: list[float] = []
        self._any = False

    def add_call(self, input_tokens: int, output_tokens: int, cost_usd: float | None) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if cost_usd is not None:
            self._any = True
            self._costs.append(cost_usd)

    def add_record(self, record: ForecastRecord) -> None:
        """Fold a full :func:`run_forecast_pass` record's tokens/cost into the total."""
        self.add_call(record.input_tokens, record.output_tokens, record.cost_usd)

    @property
    def cost_usd(self) -> float | None:
        return sum(self._costs) if self._any else None


def run_pyramid_forecast(
    snapshot: QuestionSnapshot,
    *,
    settings: PyramidSettings | None = None,
    ensemble_settings: EnsembleSettings | None = None,
    research: ResearchBundle | None = None,
    search_client: SearchClient | None = None,
    client_factory: Callable[[TaskModelConfig], LlmClient] = build_llm_client,
    now: datetime | None = None,
) -> ForecastRecord:
    """Forecast one question with the decomposition pyramid; return one record.

    Runs the full pipeline (decompose → leaves → combine → holistic → blend →
    confidence) and stores the whole argument graph on a single
    ``method="pyramid-c"`` :class:`ForecastRecord`. Every model failure degrades
    to a documented fallback with a trace note rather than raising, so this
    function returns a record even on a bad decomposition or a flaky leaf.

    Args:
        snapshot: The question to forecast.
        settings: Pyramid tunables; defaults to :func:`load_pyramid_settings`.
        ensemble_settings: Source of the spread thresholds reused to map the
            structural-vs-holistic gap onto a confidence band (spec §4); defaults
            to :func:`load_ensemble_settings`.
        research: The shared research bundle. Evidence leaves, the decomposer, and
            the holistic pass see it; model-knowledge leaves do not. ``None``
            forecasts blind throughout.
        search_client: The search client the assumption-verification pass issues
            its targeted retrievals through (one search per researchable
            assumption). ``None`` disables verification: assumptions are still
            surfaced and stored, but none are checked (they stay ``unverified``).
        client_factory: Test seam — builds an :class:`LlmClient` from a task's
            :class:`TaskModelConfig`. Defaults to :func:`build_llm_client` so each
            task (decompose / leaf / holistic) uses its own configured provider.
        now: Current time (UTC), overridable for deterministic tests.

    Returns:
        A dry-run :class:`ForecastRecord` (``method="pyramid-c"``) whose
        ``probability`` is the blended final and whose ``pyramid`` field carries
        the full :class:`PyramidDetail` trace.
    """
    pyr = settings if settings is not None else load_pyramid_settings()
    ens = ensemble_settings if ensemble_settings is not None else load_ensemble_settings()
    moment = now if now is not None else datetime.now(UTC)
    notes: list[str] = []
    cost = _CallCost()

    decompose_config = load_task_config("decompose")
    leaf_config = load_task_config("forecast_pass")
    holistic_config = _load_holistic_config()

    # 1. Decompose (with retry + single-leaf fallback).
    tree, prune_notes, decompose_failed = _decompose(
        snapshot, pyr, research, client_factory(decompose_config), decompose_config, cost, notes
    )

    # 1.5. Verify the researchable assumptions the tree rests on. Verified-false
    # ones are injected into their owning leaf's estimation prompt below.
    assumptions, verifications_run, verification_credits = _verify_assumptions(
        snapshot, tree, pyr, search_client, client_factory, cost, notes
    )
    falsified_by_node = _falsified_by_node(assumptions)
    unverified_lb_by_node = _unverified_load_bearing_by_node(assumptions)
    unverified_lb_count = sum(len(v) for v in unverified_lb_by_node.values())

    # 2. Leaves. Each evidence leaf whose key terms are uncovered by the shared
    # bundle may draw ONE top-up search (spec §4); a leaf resting on an unverified
    # load-bearing assumption then has its estimate pulled toward 0.5 (spec §3).
    leaf_nodes = _collect_leaves(tree)
    leaf_estimates: list[LeafEstimate] = []
    leaf_probabilities: dict[str, float] = {}
    leaf_client = client_factory(leaf_config)
    topup_budget = _TopupBudget(pyr.leaf_topup_max_searches)
    topup_credits_before = _search_credits(search_client) if search_client is not None else None
    for leaf in leaf_nodes:
        extra_results, topup_used = _leaf_topup(leaf, research, search_client, topup_budget, notes)
        estimate = _estimate_leaf(
            snapshot,
            leaf,
            research,
            leaf_client,
            leaf_config,
            moment,
            cost,
            notes,
            falsified=falsified_by_node.get(leaf.id, []),
            extra_results=extra_results,
            topup_used=topup_used,
        )
        estimate = _apply_fragility_pull(estimate, leaf.id in unverified_lb_by_node, pyr, notes)
        leaf_estimates.append(estimate)
        leaf_probabilities[leaf.id] = estimate.probability
    topup_credits = (
        _credit_delta(topup_credits_before, _search_credits(search_client))
        if search_client is not None
        else 0
    )

    # 3. Combine upward into a structural probability + per-node trace.
    structural_probability, trace = combine_tree(
        tree,
        leaf_probabilities,
        clip_low=pyr.leaf_clip_low,
        clip_high=pyr.leaf_clip_high,
        mode=pyr.combination_mode,
    )

    # 4. Holistic sanity check (one direct forecast of the root question).
    holistic_probability = _holistic(
        snapshot, research, client_factory(holistic_config), holistic_config, moment, cost, notes
    )
    if holistic_probability is None:
        # A failed holistic degrades to "no independent check": blend with the
        # structural value itself so the final equals structural, and note it.
        holistic_probability = structural_probability
        notes.append("holistic pass failed; blended structural against itself (no holistic check)")

    # 5. Blend.
    blended = blend(structural_probability, holistic_probability, pyr.holistic_blend_weight)

    # 6. Confidence — three raw signals, label = their conservative minimum. The
    # third (spec §3) caps confidence by the count of unverified load-bearing
    # assumptions the tree still rests on.
    weakest_leaf = _weakest_leaf_confidence(tree, leaf_estimates)
    gap = abs(to_log_odds(structural_probability) - to_log_odds(holistic_probability))
    gap_confidence = _confidence_from_gap(gap, ens)
    unverified_lb_cap = _confidence_from_unverified_load_bearing(unverified_lb_count, pyr)
    final_confidence = _min_confidence(
        _min_confidence(weakest_leaf, gap_confidence), unverified_lb_cap
    )

    detail = PyramidDetail(
        tree=tree,
        prune_notes=prune_notes,
        leaf_estimates=leaf_estimates,
        combination_steps=[trace[node_id] for node_id in _node_order(tree)],
        structural_probability=structural_probability,
        holistic_probability=holistic_probability,
        holistic_blend_weight=pyr.holistic_blend_weight,
        weakest_leaf_confidence=weakest_leaf,
        structural_holistic_gap=gap,
        decompose_failed=decompose_failed,
        assumptions=assumptions,
        assumption_verifications_run=verifications_run,
        assumption_search_credits=verification_credits,
        unverified_load_bearing_count=unverified_lb_count,
        leaf_topup_searches_run=topup_budget.used,
        leaf_topup_search_credits=topup_credits,
        notes=notes,
    )

    return ForecastRecord(
        created_at=moment,
        question_id=snapshot.question_id,
        post_id=snapshot.post_id,
        title=snapshot.title,
        probability=_clamp(blended),
        raw_probability=blended,
        confidence=final_confidence,
        rationale=_synthesize_rationale(detail, blended),
        reference_class=_join_reference_classes(leaf_estimates),
        model="pyramid-c",
        input_tokens=cost.input_tokens,
        output_tokens=cost.output_tokens,
        cost_usd=cost.cost_usd,
        dry_run=True,
        submitted_at=None,
        research=research,
        research_search_credits=(
            (research.search_credits_used if research is not None else 0)
            + verification_credits
            + topup_credits
        ),
        method="pyramid-c",
        frame=None,
        ensemble=None,
        pyramid=detail,
    )


# --- Decomposition ------------------------------------------------------------


def _decompose(
    snapshot: QuestionSnapshot,
    settings: PyramidSettings,
    research: ResearchBundle | None,
    client: LlmClient,
    config: TaskModelConfig,
    cost: _CallCost,
    notes: list[str],
) -> tuple[PyramidNode, list[PruneNote], bool]:
    """Run the decomposition call and return ``(tree, prune_notes, failed)``.

    One call, then one retry quoting the parse error (mirroring
    :func:`run_forecast_pass`). If the retry also fails to yield a valid tree, the
    pyramid degrades to a single-leaf tree of the whole question (``failed=True``)
    and records the reason in ``notes`` — a careful single pass, never a crash.
    """
    system = _decompose_system(settings)
    user_prompt = _decompose_user_prompt(snapshot, research)
    schema = decomposition_json_schema()

    try:
        result = _run(client, system, user_prompt, config, schema)
    except LlmError as exc:
        notes.append(f"decomposition LLM call failed ({exc}); degraded to a single-leaf tree")
        return _single_leaf(snapshot), [], True
    cost.add_call(result.input_tokens, result.output_tokens, result.cost_usd)

    try:
        return (*_parse_tree(result.text, settings), False)
    except (json.JSONDecodeError, PyramidParseError, ValidationError) as first_error:
        retry_prompt = user_prompt + _retry_note(first_error)
        try:
            retry = _run(client, system, retry_prompt, config, schema)
        except LlmError as exc:
            notes.append(f"decomposition retry LLM call failed ({exc}); degraded to a single leaf")
            return _single_leaf(snapshot), [], True
        cost.add_call(retry.input_tokens, retry.output_tokens, retry.cost_usd)
        try:
            return (*_parse_tree(retry.text, settings), False)
        except (json.JSONDecodeError, PyramidParseError, ValidationError) as second_error:
            notes.append(
                f"decomposition did not parse after one retry ({second_error}); "
                "degraded to a single-leaf tree"
            )
            return _single_leaf(snapshot), [], True


def _parse_tree(text: str, settings: PyramidSettings) -> tuple[PyramidNode, list[PruneNote]]:
    """Extract the JSON object from the reply and parse it into a capped tree."""
    raw = json.loads(_extract_json_object(text))
    return parse_decomposition(raw, settings)


def _decompose_system(settings: PyramidSettings) -> str:
    """Build the decomposer system prompt with the caps filled from config."""
    body = (
        DECOMPOSE_PROMPT.replace("{today}", datetime.now(UTC).date().isoformat())
        .replace("{max_depth}", str(settings.max_depth))
        .replace("{max_children}", str(settings.max_children))
        .replace("{max_conjuncts}", str(settings.max_conjuncts))
    )
    return f"{body}\n\n{DECOMPOSE_OUTPUT_FORMAT}"


def _decompose_user_prompt(snapshot: QuestionSnapshot, research: ResearchBundle | None) -> str:
    """Reuse the forecaster's per-question prompt (sources included when present).

    The trailing "give your probability" line is harmless here — the decomposer's
    system prompt overrides the task — and reusing the builder keeps the question,
    resolution criteria, dates, and SOURCES block rendered identically to a leaf's.
    """
    return _build_user_prompt(snapshot, research)


def _single_leaf(snapshot: QuestionSnapshot) -> PyramidNode:
    """The fallback tree: the whole question as one evidence leaf."""
    return PyramidNode(
        id="root",
        question=snapshot.title,
        node_type="leaf",
        reasoning_type="checklist",
        grounding="evidence",
    )


# --- Leaves -------------------------------------------------------------------


def _estimate_leaf(
    snapshot: QuestionSnapshot,
    leaf: PyramidNode,
    research: ResearchBundle | None,
    client: LlmClient,
    config: TaskModelConfig,
    moment: datetime,
    cost: _CallCost,
    notes: list[str],
    *,
    falsified: list[Assumption],
    extra_results: list[SearchResult],
    topup_used: bool,
) -> LeafEstimate:
    """Estimate one leaf via :func:`run_forecast_pass`, degrading on failure.

    ``evidence`` leaves get the shared research bundle in context; ``model-
    knowledge`` leaves get NONE (spec §2 — cheaper, and it keeps stable-fact
    estimates from being diluted by current-events sources). A leaf whose pass
    fails after its own retry becomes a ``0.5`` / ``"low"`` fallback with a note.

    ``falsified`` are this leaf's own assumptions that verification found FALSE;
    when present, they are injected prominently into the estimation prompt (via
    ``run_forecast_pass``'s ``system_addendum`` seam) so the leaf reasons with the
    corrected facts rather than the wrong hidden guess (spec §4).

    ``extra_results`` are top-up search hits (spec §4) for an evidence leaf whose
    key terms the shared bundle did not cover; they are appended to the leaf's
    SOURCES block. ``topup_used`` records whether that draw happened (stored on
    the returned estimate). The fragility pull is applied by the caller, not here.
    """
    grounding = leaf.grounding if leaf.grounding is not None else "evidence"
    leaf_research = research if grounding == "evidence" else None
    if extra_results and leaf_research is not None:
        leaf_research = leaf_research.model_copy(
            update={"results": [*leaf_research.results, *extra_results]}
        )
    leaf_snapshot = _leaf_snapshot(snapshot, leaf.question)
    addendum = _falsified_addendum(falsified) if falsified else None
    try:
        record = run_forecast_pass(
            leaf_snapshot,
            client=client,
            config=config,
            research=leaf_research,
            now=moment,
            system_addendum=addendum,
        )
    except LlmError as exc:
        notes.append(f"leaf {leaf.id!r} failed ({exc}); degraded to p=0.5 confidence=low")
        return LeafEstimate(
            node_id=leaf.id,
            question=leaf.question,
            grounding=grounding,
            probability=0.5,
            confidence="low",
            rationale=f"Leaf estimation failed and was defaulted: {exc}",
            reference_class="",
            failed=True,
            topup_search_used=topup_used,
        )
    cost.add_record(record)
    return LeafEstimate(
        node_id=leaf.id,
        question=leaf.question,
        grounding=grounding,
        probability=record.probability,
        confidence=record.confidence,
        rationale=record.rationale,
        reference_class=record.reference_class,
        failed=False,
        topup_search_used=topup_used,
    )


def _leaf_snapshot(snapshot: QuestionSnapshot, question: str) -> QuestionSnapshot:
    """A derived snapshot posing the leaf sub-question, not the original question.

    The leaf's ``question`` is self-contained plain wording, but the ROOT
    question's resolution criteria and fine print stay attached as labeled
    context: they define the time windows, sources, and edge cases the leaf's
    wording must be interpreted under. P1 falsified the earlier "blank them"
    approach — a Bitcoin leaf answered "did BTC *ever* trade above $120k" at 0.99
    because it never saw the root's window definition (docs/reviews/
    forecast-43507.html). Ids and dates are kept so the trace stays attributable.
    """
    root_context = (
        f"This sub-question is one node in a decomposition of the root question: "
        f"{snapshot.title!r}. Interpret time windows, sources, and edge cases in "
        f"your sub-question consistently with the ROOT question's resolution "
        f"criteria below.\n\nROOT resolution criteria: {snapshot.resolution_criteria}"
        + (f"\n\nROOT fine print: {snapshot.fine_print}" if snapshot.fine_print else "")
    )
    return snapshot.model_copy(
        update={
            "title": question,
            "description": root_context,
            "resolution_criteria": "",
            "fine_print": "",
        }
    )


# --- Holistic -----------------------------------------------------------------


def _holistic(
    snapshot: QuestionSnapshot,
    research: ResearchBundle | None,
    client: LlmClient,
    config: TaskModelConfig,
    moment: datetime,
    cost: _CallCost,
    notes: list[str],
) -> float | None:
    """One direct forecast of the ROOT question; ``None`` if the pass failed.

    This is the holistic sanity check that never lets the structural estimate
    stand alone (docs/03 guard). It sees the full research bundle. A failure is
    caught and reported as ``None`` so the caller can degrade the blend.
    """
    try:
        record = run_forecast_pass(
            snapshot, client=client, config=config, research=research, now=moment
        )
    except LlmError as exc:
        notes.append(f"holistic pass raised ({exc})")
        return None
    cost.add_record(record)
    return record.probability


# --- Assumption verification --------------------------------------------------

# The verification query-gen system prompt: turn ONE assumption (plus the root
# question, for context) into 1-2 targeted web searches. Mirrors research.py's
# QUERY_GEN_SYSTEM style — short, plain, JSON out — because the failure it fixes is
# exactly QUERY_GEN's absence here: searching an assumption's prose verbatim ("the
# resolution hinges on the specified accusation...") retrieves meta-noise. This
# step forces the search back onto the concrete FACT the assumption asserts.
VERIFICATION_QUERY_GEN_SYSTEM = """\
You turn ONE assumption that a forecast quietly relies on into web-search queries \
that would confirm or refute it. You are given the assumption and the root \
forecasting question it sits under, for context only.

Rules:
- Produce 1 or 2 focused search queries — concrete and keyword-rich: the specific \
people, organizations, places, dates, numbers, laws, or documents the assumption \
is about.
- Search for the underlying FACT itself, the way a news or reference lookup would \
phrase it. NEVER write meta-language about the forecast — no "resolution criteria", \
"resolution hinges", "assumption", "load-bearing", and never just restate the \
question. Those words retrieve commentary, not the fact.
- If the assumption turns on an official source (an agency, court, calendar, \
dataset, filing), aim one query at that source's own latest statement.

Return ONLY a single JSON object and nothing else, of exactly this form:
{"queries": ["first query", "second query"]}"""

# Advisory schema for providers with server-side structured output; the Claude CLI
# ignores it and we fall back to prompt-JSON extraction + validation.
_VQUERY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}

# The verdict system prompt: read one search result set and vote on whether it
# verifies, falsifies, or fails to settle a single factual assumption. Cheap by
# design — it reads evidence and returns a small JSON object, nothing more.
VERDICT_SYSTEM = """\
You are fact-checking a single ASSUMPTION that a forecast silently relies on.
You are given the assumption and a few search results gathered for it. Decide,
using ONLY the results, which verdict fits:
- "verified-true": the results clearly SUPPORT the assumption.
- "verified-false": the results clearly CONTRADICT the assumption.
- "unverified": the results do not settle it (absent, off-topic, or mixed).

Be conservative: only claim verified-true / verified-false when the evidence is
clear. When in doubt, answer "unverified".

Return ONLY a single JSON object of exactly this form, nothing else:
{"verdict": "verified-true" | "verified-false" | "unverified",
 "evidence": "one short line citing what settled it, or why it could not be"}"""

# Advisory schema for providers with server-side structured output; the Claude
# CLI ignores it and we fall back to prompt-JSON extraction + validation.
_VERDICT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["verified-true", "verified-false", "unverified"],
        },
        "evidence": {"type": "string"},
    },
    "required": ["verdict", "evidence"],
}

_VALID_VERDICTS: frozenset[str] = frozenset({"verified-true", "verified-false", "unverified"})


class _SearchBudget:
    """A bounded per-question counter for the assumption-verification search draw.

    See :func:`_verify_assumptions` for the budget semantics this enforces: a total
    ceiling on the searches the verification pass may issue across all assumptions.
    ``take`` grants one search if room remains; ``remaining`` is what is left.
    """

    def __init__(self, cap: int) -> None:
        self._cap = max(0, cap)
        self.used = 0

    def remaining(self) -> int:
        return max(0, self._cap - self.used)

    def take(self) -> bool:
        """Grant one search if budget remains; return whether it was granted."""
        if self.used >= self._cap:
            return False
        self.used += 1
        return True


def _verify_assumptions(
    snapshot: QuestionSnapshot,
    tree: PyramidNode,
    settings: PyramidSettings,
    search_client: SearchClient | None,
    client_factory: Callable[[TaskModelConfig], LlmClient],
    cost: _CallCost,
    notes: list[str],
) -> tuple[list[Assumption], int, int]:
    """Verify the tree's researchable assumptions; return ``(all, run, credits)``.

    Collects every assumption across the tree (filling each ``node_id``), selects
    the researchable ones — deduped by normalized text, load-bearing first then
    tree order, capped at ``max_assumption_verifications`` — and, for each, first
    runs one cheap query-gen call turning the assumption (plus the root question
    for context) into 1-2 targeted searches, then issues those searches and one
    cheap verdict call. Searching the assumption *prose* verbatim retrieves
    meta-noise; the query-gen step forces the search onto the concrete fact.

    **Budget semantics (documented deliberately).** ``max_assumption_verifications``
    keeps its spec meaning — it caps how many ASSUMPTIONS are put through
    verification (the selection cap above). A SEPARATE total-search ceiling of
    ``max_assumption_verifications * 2`` bounds the searches the pass may draw,
    since each assumption fans out to at most 2 queries (1 search each). The two
    caps coincide in the normal case (N assumptions x 2 queries = 2N searches), so
    every selected assumption is verified; the search ceiling only ever binds if
    query fan-out would push the draw past 2N, in which case later assumptions are
    left unverified rather than over-spending. Order stays load-bearing-first, so a
    binding ceiling drops the least-important assumptions.

    The verdict flips the assumption's ``status``/``evidence`` in place and its
    ``queries_used`` records the searches issued; a duplicate assumption elsewhere
    in the tree inherits all three. Every failure (no search client, a query-gen
    failure — which degrades to a verbatim search — a search error, a bad verdict
    call) degrades with a trace note and never raises.

    Returns the full flat assumption list, how many were actually put through
    verification, and the metered search credits the pass drew.
    """
    all_assumptions = _collect_assumptions(tree)
    if not all_assumptions:
        return [], 0, 0

    researchable = [a for a in all_assumptions if a.verifiability == "researchable"]
    to_verify = _select_assumptions_to_verify(researchable, settings.max_assumption_verifications)
    if not to_verify:
        return all_assumptions, 0, 0

    if search_client is None:
        notes.append(
            f"assumption verification skipped: no search client available "
            f"({len(to_verify)} researchable assumption(s) left unverified)"
        )
        return all_assumptions, 0, 0

    retrieval = load_retrieval_settings()
    verdict_config = _load_verdict_config()
    verdict_client = client_factory(verdict_config)
    query_config = _load_verification_query_config()
    query_client = client_factory(query_config)
    budget = _SearchBudget(settings.max_assumption_verifications * 2)

    credits_before = _search_credits(search_client)
    verifications_run = 0
    for assumption in to_verify:
        if budget.remaining() <= 0:
            notes.append(
                f"assumption verification search budget exhausted; {assumption.text!r} "
                f"left unverified"
            )
            continue
        if _verify_one(
            assumption,
            snapshot.title,
            search_client,
            query_client,
            query_config,
            verdict_client,
            verdict_config,
            retrieval,
            budget,
            cost,
            notes,
        ):
            verifications_run += 1
    credits_after = _search_credits(search_client)
    credits = _credit_delta(credits_before, credits_after)

    _propagate_verdicts(all_assumptions, to_verify)
    return all_assumptions, verifications_run, credits


def _verify_one(
    assumption: Assumption,
    root_title: str,
    search_client: SearchClient,
    query_client: LlmClient,
    query_config: TaskModelConfig,
    verdict_client: LlmClient,
    verdict_config: TaskModelConfig,
    retrieval: RetrievalSettings,
    budget: _SearchBudget,
    cost: _CallCost,
    notes: list[str],
) -> bool:
    """Run one assumption's query-gen, searches, and verdict; mutate it in place.

    Returns whether the assumption was actually verified (at least one search plus
    a verdict call attempted). Degrades to ``unverified`` (with a note, returning
    ``False``) when the search budget leaves no room, every search errors, the
    verdict call fails, or the verdict is unparseable. ``queries_used`` always
    records the queries issued (the generated targeted ones, or the verbatim
    assumption text when query-gen fell back). The query-gen and verdict calls'
    tokens/cost are folded into the running total whenever the call was made.
    """
    queries = _generate_verification_queries(
        assumption, root_title, query_client, query_config, cost, notes
    )

    results: list[SearchResult] = []
    searched: list[str] = []
    for query in queries:
        if not budget.take():
            notes.append(
                f"assumption verification search budget exhausted mid-assumption for "
                f"{assumption.text!r}; searched {len(searched)} of {len(queries)} query(ies)"
            )
            break
        try:
            results.extend(search_client.search(query, max_results=retrieval.max_results))
        except SearchError as exc:
            notes.append(
                f"assumption verification search failed for query {query!r} ({exc}); skipped"
            )
            continue
        searched.append(query)

    assumption.queries_used = searched
    if not searched:
        notes.append(
            f"assumption verification made no successful search for {assumption.text!r}; "
            f"left unverified"
        )
        return False

    try:
        result = _run(
            verdict_client,
            VERDICT_SYSTEM,
            _verdict_prompt(assumption, results, searched),
            verdict_config,
            _VERDICT_SCHEMA,
        )
    except LlmError as exc:
        notes.append(
            f"assumption verdict call failed for {assumption.text!r} ({exc}); left unverified"
        )
        return False
    cost.add_call(result.input_tokens, result.output_tokens, result.cost_usd)

    try:
        status, evidence = _parse_verdict(result.text)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        notes.append(
            f"assumption verdict did not parse for {assumption.text!r} ({exc}); left unverified"
        )
        return False

    assumption.status = status
    assumption.evidence = evidence
    if status == "verified-false":
        notes.append(f"assumption verified FALSE: {assumption.text!r} — {evidence}")
    return True


def _generate_verification_queries(
    assumption: Assumption,
    root_title: str,
    query_client: LlmClient,
    config: TaskModelConfig,
    cost: _CallCost,
    notes: list[str],
) -> list[str]:
    """Turn one assumption into 1-2 targeted searches; degrade to verbatim on failure.

    ONE cheap LLM call (no retry — this is a routing step). On an ``LlmError`` or an
    unparseable reply, falls back to today's behavior — a single verbatim search of
    the assumption text — leaving a trace note rather than crashing. The call's
    tokens/cost are folded into the running total whenever the call was made.
    """
    prompt = _verification_query_prompt(assumption, root_title)
    try:
        result = _run(query_client, VERIFICATION_QUERY_GEN_SYSTEM, prompt, config, _VQUERY_SCHEMA)
    except LlmError as exc:
        notes.append(
            f"verification query-gen call failed for {assumption.text!r} ({exc}); "
            f"searched the assumption text verbatim"
        )
        return [assumption.text]
    cost.add_call(result.input_tokens, result.output_tokens, result.cost_usd)

    try:
        return _parse_verification_queries(result.text)
    except (json.JSONDecodeError, ValueError) as exc:
        notes.append(
            f"verification query-gen did not parse for {assumption.text!r} ({exc}); "
            f"searched the assumption text verbatim"
        )
        return [assumption.text]


def _parse_verification_queries(text: str) -> list[str]:
    """Extract and validate the ``queries`` list from a verification query-gen reply.

    Dedupes, drops blanks, and caps at TWO (the per-assumption fan-out limit the
    search budget is sized against).

    Raises:
        json.JSONDecodeError: No JSON object could be located in the reply.
        ValueError: The JSON was the wrong shape or held no usable query.
    """
    data = json.loads(_extract_json_object(text))
    if not isinstance(data, dict):
        raise ValueError("verification query-gen JSON was not an object")
    queries = data.get("queries")
    if not isinstance(queries, list):
        raise ValueError("verification query-gen JSON had no 'queries' list")
    cleaned: list[str] = []
    for item in queries:
        if isinstance(item, str) and item.strip():
            candidate = item.strip()
            if candidate not in cleaned:
                cleaned.append(candidate)
    if not cleaned:
        raise ValueError("verification query-gen returned no usable queries")
    return cleaned[:2]


def _verification_query_prompt(assumption: Assumption, root_title: str) -> str:
    """Build the verification query-gen user message: the assumption + root context."""
    return "\n\n".join(
        [
            f"ROOT QUESTION (context only, do NOT search it verbatim):\n{root_title}",
            f"ASSUMPTION TO CHECK:\n{assumption.text}",
            "Return the JSON object of search queries now.",
        ]
    )


def _collect_assumptions(tree: PyramidNode) -> list[Assumption]:
    """Every assumption across the tree, in pre-order, each with ``node_id`` filled.

    Mutates each assumption's ``node_id`` to its owning node (they are the tree's
    own assumption objects), so the flat list and the stored tree agree.
    """
    collected: list[Assumption] = []

    def walk(node: PyramidNode) -> None:
        for assumption in node.assumptions:
            assumption.node_id = node.id
            collected.append(assumption)
        for child in node.children:
            walk(child)

    walk(tree)
    return collected


def _select_assumptions_to_verify(researchable: list[Assumption], cap: int) -> list[Assumption]:
    """Dedupe by normalized text, order load-bearing first (then tree order), cap.

    Surfacing assumptions is cheap but verifying them is not, so when the budget
    binds the load-bearing researchable ones are checked first. Dedup keeps the
    first occurrence (tree order), and the load-bearing-first sort is stable, so
    ties preserve tree order.
    """
    deduped: list[Assumption] = []
    seen: set[str] = set()
    for assumption in researchable:
        key = _normalize(assumption.text)
        if key not in seen:
            seen.add(key)
            deduped.append(assumption)
    ordered = sorted(deduped, key=lambda a: not a.load_bearing)
    return ordered[:cap]


def _propagate_verdicts(all_assumptions: list[Assumption], verified: list[Assumption]) -> None:
    """Copy each verified assumption's verdict onto its near-identical duplicates.

    Verification checks one representative per normalized text; any other node
    that rested on the same assumption inherits the verdict, so the flat trace is
    coherent (no duplicate left spuriously ``unverified``).
    """
    by_norm = {_normalize(a.text): a for a in verified}
    for assumption in all_assumptions:
        representative = by_norm.get(_normalize(assumption.text))
        if representative is not None and assumption is not representative:
            assumption.status = representative.status
            assumption.evidence = representative.evidence
            assumption.queries_used = list(representative.queries_used)


def _falsified_by_node(assumptions: list[Assumption]) -> dict[str, list[Assumption]]:
    """Group the ``verified-false`` assumptions by their owning node id."""
    grouped: dict[str, list[Assumption]] = {}
    for assumption in assumptions:
        if assumption.status == "verified-false":
            grouped.setdefault(assumption.node_id, []).append(assumption)
    return grouped


def _unverified_load_bearing_by_node(assumptions: list[Assumption]) -> dict[str, list[Assumption]]:
    """Group the still-``unverified`` load-bearing assumptions by owning node id.

    These are the assumptions that drive both the fragility pull (spec §3, applied
    per leaf) and the third confidence signal (their total count). An assumption
    counts only when it is load-bearing AND ``unverified`` after the verification
    pass — a verified-true/false one is no longer a hidden risk.
    """
    grouped: dict[str, list[Assumption]] = {}
    for assumption in assumptions:
        if assumption.load_bearing and assumption.status == "unverified":
            grouped.setdefault(assumption.node_id, []).append(assumption)
    return grouped


def _falsified_addendum(falsified: list[Assumption]) -> str:
    """The system-prompt section injecting a leaf's falsified assumptions (spec §4)."""
    lines = [
        "CORRECTED ASSUMPTIONS",
        "",
        "One or more assumptions this estimate might rest on have been checked "
        "against evidence and found FALSE. Do NOT treat them as true — reason with "
        "the corrected facts below:",
    ]
    for assumption in falsified:
        evidence = assumption.evidence.strip() or "(no evidence detail)"
        lines.append(f'- The assumption "{assumption.text}" is FALSE: {evidence}')
    return "\n".join(lines)


def _verdict_prompt(assumption: Assumption, results: list[SearchResult], queries: list[str]) -> str:
    """Build the verdict user message: the assumption, the queries, and the results.

    The queries that produced ``results`` are shown so the verdict knows what was
    (and was not) searched — an off-target query set is itself a reason a result
    set fails to settle the assumption.
    """
    return "\n\n".join(
        [
            f"ASSUMPTION TO CHECK:\n{assumption.text}",
            _render_verdict_queries(queries),
            _render_verdict_sources(results),
            "Return the JSON verdict object now.",
        ]
    )


def _render_verdict_queries(queries: list[str]) -> str:
    """Render the compact list of search queries the verdict's results came from."""
    if not queries:
        return "SEARCH QUERIES USED:\n(none)"
    lines = ["SEARCH QUERIES USED:"]
    lines.extend(f"- {query}" for query in queries)
    return "\n".join(lines)


def _render_verdict_sources(results: list[SearchResult]) -> str:
    """Render a compact numbered source block for the verdict prompt."""
    if not results:
        return "SEARCH RESULTS:\n(no results returned)"
    lines = ["SEARCH RESULTS:"]
    for index, result in enumerate(results, start=1):
        title = result.title.strip() or "(untitled)"
        domain = result.source_domain or "unknown source"
        when = result.published_at.date().isoformat() if result.published_at else "date unknown"
        snippet = result.snippet.strip() or "(no snippet)"
        lines.append(f"[S{index}] {title} ({domain}, {when})\n    {snippet}")
    return "\n".join(lines)


def _parse_verdict(text: str) -> tuple[AssumptionStatus, str]:
    """Extract and validate the ``{verdict, evidence}`` object from a verdict reply.

    Raises:
        json.JSONDecodeError: No JSON object could be located in the reply.
        ValueError: The JSON was the wrong shape or the verdict was not one of the
            three allowed labels.
    """
    raw = json.loads(_extract_json_object(text))
    if not isinstance(raw, dict):
        raise ValueError("verdict JSON was not an object")
    verdict = raw.get("verdict")
    if not isinstance(verdict, str) or verdict not in _VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {sorted(_VALID_VERDICTS)}, got {verdict!r}")
    evidence_raw = raw.get("evidence", "")
    evidence = evidence_raw.strip() if isinstance(evidence_raw, str) else ""
    # verdict is narrowed to the AssumptionStatus literal set by the check above.
    return verdict, evidence  # type: ignore[return-value]


def _search_credits(client: SearchClient) -> int | None:
    """Read a metered search client's month-to-date credits, or ``None``.

    Mirrors ``research._credits_used``: only a metered client (AskNews) exposes
    ``credits_used_this_month``; an unmetered one (Tavily) returns ``None`` and the
    pass records a zero draw for it.
    """
    getter = getattr(client, "credits_used_this_month", None)
    if callable(getter):
        try:
            return int(getter())
        except (TypeError, ValueError):
            return None
    return None


def _credit_delta(before: int | None, after: int | None) -> int:
    """The non-negative credit draw between two reads, or 0 if either is unknown."""
    if before is None or after is None:
        return 0
    return max(0, after - before)


def _normalize(text: str) -> str:
    """Case- and whitespace-insensitive key for near-identical-assumption dedup."""
    return " ".join(text.lower().split())


# --- Fragility pull -----------------------------------------------------------

# The anchor the fragility pull drags a fragile leaf toward. Spec §3 calls for the
# leaf's *outside-view base rate*, but leaves do not yet emit a machine-readable
# base rate (ForecastOutput carries no numeric anchor — extracting one is a queued
# improvement). In stage 2 the anchor is therefore 0.5, "maximum uncertainty":
# discount a fragile leaf toward not-knowing rather than toward a specific number
# we cannot read. Documented as a deliberate deviation in the v2 spec.
_FRAGILITY_ANCHOR = 0.5


def _apply_fragility_pull(
    estimate: LeafEstimate,
    has_unverified_load_bearing: bool,
    settings: PyramidSettings,
    notes: list[str],
) -> LeafEstimate:
    """Pull a fragile leaf's probability toward the 0.5 anchor (spec §3).

    A leaf resting on ≥1 unverified load-bearing assumption is "fragile": its
    estimate silently rides on a belief we could not confirm. We discount it by
    pulling ``probability`` toward :data:`_FRAGILITY_ANCHOR` (0.5) in log-odds
    space by ``fragility_pull_weight`` — reusing :func:`blend`, the project's
    log-odds pool, so no new math is introduced (``blend(p, 0.5, w)`` weights the
    anchor ``w``). The pre-pull probability is stored on the returned estimate so
    an experiment can re-score with the pull disabled (recomputability).

    A weight of ``0.0`` or a non-fragile leaf returns the estimate unchanged.
    """
    if not has_unverified_load_bearing or settings.fragility_pull_weight <= 0.0:
        return estimate
    pulled = blend(estimate.probability, _FRAGILITY_ANCHOR, settings.fragility_pull_weight)
    notes.append(
        f"fragility pull: leaf {estimate.node_id!r} rests on an unverified load-bearing "
        f"assumption; pulled {estimate.probability:.3f} -> {pulled:.3f} toward 0.5 "
        f"(weight {settings.fragility_pull_weight:.2f})"
    )
    return estimate.model_copy(
        update={
            "pre_pull_probability": estimate.probability,
            "probability": pulled,
            "fragility_pull_applied": True,
        }
    )


# --- Leaf retrieval top-up ----------------------------------------------------

# Short function words stripped before checking bundle coverage: they carry no
# topic signal, so a leaf sharing only "will"/"that" with the bundle is still
# uncovered. Kept deliberately small — the heuristic only needs to catch leaves
# the bundle plainly never mentions.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "will",
        "would",
        "that",
        "this",
        "with",
        "from",
        "have",
        "has",
        "had",
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "been",
        "being",
        "into",
        "than",
        "then",
        "them",
        "they",
        "their",
        "there",
        "here",
        "what",
        "when",
        "which",
        "whether",
        "does",
        "did",
        "any",
        "all",
        "not",
        "but",
        "its",
        "his",
        "her",
        "our",
        "your",
        "over",
        "under",
        "more",
        "most",
        "some",
        "such",
        "only",
        "also",
        "about",
        "before",
        "after",
        "during",
        "between",
    }
)


class _TopupBudget:
    """A tiny per-question counter for the leaf top-up search budget (spec §4).

    Top-up shares the search stack with verification, which has priority (it runs
    first, before this counter is touched). ``cap`` is ``leaf_topup_max_searches``;
    each leaf that draws a top-up search calls :meth:`take` once. ``used`` is the
    number granted, for the trace.
    """

    def __init__(self, cap: int) -> None:
        self._cap = max(0, cap)
        self.used = 0

    def take(self) -> bool:
        """Grant one top-up search if budget remains; return whether it was granted."""
        if self.used >= self._cap:
            return False
        self.used += 1
        return True


def _significant_terms(text: str) -> set[str]:
    """Lowercased, stopword-stripped word tokens longer than three characters."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 3 and t not in _STOPWORDS}


def _bundle_covers(terms: set[str], research: ResearchBundle) -> bool:
    """Whether any of ``terms`` appears in the bundle's titles/snippets/doc text.

    The cheap coverage heuristic from spec §4: a leaf is "covered" if even one of
    its significant terms shows up somewhere in the shared bundle. Empty ``terms``
    (a leaf with no significant words) counts as covered — there is nothing to
    search for.
    """
    if not terms:
        return True
    parts: list[str] = []
    for result in research.results:
        parts.append(result.title)
        parts.append(result.snippet)
    for doc in research.docs:
        parts.append(doc.text)
    haystack = " ".join(parts).lower()
    return any(term in haystack for term in terms)


def _leaf_topup(
    leaf: PyramidNode,
    research: ResearchBundle | None,
    search_client: SearchClient | None,
    budget: _TopupBudget,
    notes: list[str],
) -> tuple[list[SearchResult], bool]:
    """Maybe draw one top-up search for an evidence leaf the bundle does not cover.

    Returns ``(extra_results, used)``. Top-up is attempted only when: there is a
    search client and a shared bundle, the leaf is an ``evidence`` leaf, the
    bundle gives its key terms zero coverage, and the top-up budget has room. A
    search error degrades to no extra results (``used=False``) with a note; the
    budget slot is still consumed (the attempt is recorded), matching how
    verification meters an attempt even when its verdict fails.
    """
    if search_client is None or research is None:
        return [], False
    if leaf.grounding != "evidence":
        return [], False
    terms = _significant_terms(leaf.question)
    if _bundle_covers(terms, research):
        return [], False
    if not budget.take():
        notes.append(
            f"leaf top-up skipped for {leaf.id!r}: zero bundle coverage but the "
            f"top-up search budget is exhausted"
        )
        return [], False

    retrieval = load_retrieval_settings()
    try:
        results = search_client.search(leaf.question, max_results=retrieval.max_results)
    except SearchError as exc:
        notes.append(f"leaf top-up search failed for {leaf.id!r} ({exc}); leaf left as-is")
        return [], False
    notes.append(
        f"leaf top-up: {leaf.id!r} had zero bundle coverage; appended {len(results)} "
        f"targeted result(s) to its context"
    )
    return results, True


# --- Confidence ---------------------------------------------------------------


def _confidence_from_unverified_load_bearing(count: int, settings: PyramidSettings) -> Confidence:
    """Map the unverified-load-bearing count onto a confidence cap (spec §3).

    ``≥ unverified_lb_low_cap`` caps at ``"low"``; ``≥ unverified_lb_medium_cap``
    caps at ``"medium"``; otherwise no cap (``"high"``). This is the third raw
    confidence signal, combined with the other two by the conservative minimum.
    """
    if count >= settings.unverified_lb_low_cap:
        return "low"
    if count >= settings.unverified_lb_medium_cap:
        return "medium"
    return "high"


def _weakest_leaf_confidence(tree: PyramidNode, estimates: list[LeafEstimate]) -> Confidence:
    """Confidence signal (a): the weakest leaf confidence on any multiplied path.

    "Multiplied path" = a path from the root that passes through at least one
    conjunction or conditional node (the shapes exposed to conjunctive error
    compounding). We take the minimum confidence over the leaves on such paths.
    When the tree multiplies nothing (a pure disjunction tree, or a single leaf),
    we fall back to the minimum over ALL leaves so the signal is always defined.
    """
    by_id = {e.node_id: e.confidence for e in estimates}
    on_multiplied: list[Confidence] = []

    def walk(node: PyramidNode, under_multiply: bool) -> None:
        if node.node_type == "leaf":
            if under_multiply and node.id in by_id:
                on_multiplied.append(by_id[node.id])
            return
        deeper = under_multiply or node.node_type in _MULTIPLYING_TYPES
        for child in node.children:
            walk(child, deeper)

    walk(tree, under_multiply=False)
    pool = on_multiplied if on_multiplied else [e.confidence for e in estimates]
    if not pool:
        return "low"
    return min(pool, key=lambda c: _CONFIDENCE_ORDER[c])


def _confidence_from_gap(gap: float, settings: EnsembleSettings) -> Confidence:
    """Map the structural-vs-holistic log-odds gap onto a confidence band.

    Reuses the ensemble's spread thresholds (spec §4): a small gap (the two
    estimates agree) reads as high confidence, a large gap as low. Identical shape
    to ``ensemble._confidence_from_spread`` — deliberately, so the two candidates'
    confidence scales stay comparable.
    """
    if gap <= settings.confidence_high_max_spread:
        return "high"
    if gap >= settings.confidence_low_min_spread:
        return "low"
    return "medium"


def _min_confidence(a: Confidence, b: Confidence) -> Confidence:
    """The more cautious of two confidence labels (``low`` < ``medium`` < ``high``)."""
    return a if _CONFIDENCE_ORDER[a] <= _CONFIDENCE_ORDER[b] else b


# --- Tree helpers -------------------------------------------------------------


def _collect_leaves(tree: PyramidNode) -> list[PyramidNode]:
    """Every leaf node, in a stable pre-order walk."""
    leaves: list[PyramidNode] = []

    def walk(node: PyramidNode) -> None:
        if node.node_type == "leaf":
            leaves.append(node)
            return
        for child in node.children:
            walk(child)

    walk(tree)
    return leaves


def _node_order(tree: PyramidNode) -> list[str]:
    """Every node id in pre-order, so the stored combination steps read top-down."""
    order: list[str] = []

    def walk(node: PyramidNode) -> None:
        order.append(node.id)
        for child in node.children:
            walk(child)

    walk(tree)
    return order


# --- Record assembly ----------------------------------------------------------


def _synthesize_rationale(detail: PyramidDetail, blended: float) -> str:
    """A short, legible summary of the pyramid's argument for the record."""
    lines = [
        f"Decomposition pyramid: structural {detail.structural_probability:.2f}, "
        f"holistic {detail.holistic_probability:.2f}, "
        f"blended {blended:.2f} (holistic weight {detail.holistic_blend_weight:.2f})."
    ]
    if detail.decompose_failed:
        lines.append("Decomposition failed; ran as a single careful leaf.")
    for estimate in detail.leaf_estimates:
        flag = " [FAILED]" if estimate.failed else ""
        lines.append(
            f"  leaf {estimate.node_id} ({estimate.grounding}) -> "
            f"{estimate.probability:.2f}{flag}: {estimate.question}"
        )
    if detail.prune_notes:
        lines.append(f"Pruned {len(detail.prune_notes)} subtree(s) to satisfy structural caps.")
    for note in detail.notes:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def _join_reference_classes(estimates: list[LeafEstimate]) -> str:
    """A deduped, order-preserving join of the leaves' outside-view anchors."""
    return "; ".join(dict.fromkeys(e.reference_class for e in estimates if e.reference_class))


# --- Config + LLM plumbing ----------------------------------------------------


def _load_holistic_config() -> TaskModelConfig:
    """The holistic pass config: ``[tasks.holistic]``, falling back to forecast_pass.

    ``load_task_config`` raises ``KeyError`` when a task table is absent, so we
    catch that and fall back to ``[tasks.forecast_pass]`` — the spec's declared
    default when ``[tasks.holistic]`` is not configured.
    """
    try:
        return load_task_config("holistic")
    except KeyError:
        return load_task_config("forecast_pass")


def _load_verdict_config() -> TaskModelConfig:
    """The assumption-verdict config: ``[tasks.assumption_verdict]``, else forecast_pass.

    Same fallback shape as :func:`_load_holistic_config`: ``load_task_config``
    raises ``KeyError`` when the task table is absent, so we fall back to
    ``[tasks.forecast_pass]`` — the spec's declared default for the verdict call.
    """
    try:
        return load_task_config("assumption_verdict")
    except KeyError:
        return load_task_config("forecast_pass")


def _load_verification_query_config() -> TaskModelConfig:
    """The verification query-gen config: ``[tasks.verification_query_gen]``, else query_gen.

    Unlike the verdict/holistic fallbacks (which fall back to ``forecast_pass``),
    this routing step falls back to ``[tasks.query_gen]`` — turning an assumption
    into targeted searches is the same cheap routing job the research stack's
    query generation already does, so it should inherit that model, not the
    heavier forecast model.
    """
    try:
        return load_task_config("verification_query_gen")
    except KeyError:
        return load_task_config("query_gen")


def _run(
    client: LlmClient,
    system: str,
    prompt: str,
    config: TaskModelConfig,
    schema: dict[str, object],
) -> LlmResult:
    """One LLM call through a client, carrying the task's model/effort/token budget."""
    return client.run(
        system,
        prompt,
        config.model,
        config.max_tokens,
        json_schema=schema,
        effort=config.effort,
    )


def _retry_note(error: Exception) -> str:
    """The decompose retry addendum that quotes the parse error (like the pass)."""
    return (
        "\n\nYour previous reply could not be parsed as the required decomposition "
        f"tree. The error was: {error}\n"
        "Reply again with your reasoning and end with ONLY the single JSON object "
        "described in the OUTPUT FORMAT section — the root node, every leaf carrying "
        "a 'grounding', and no text after the closing brace."
    )


def _clamp(probability: float) -> float:
    """Squeeze a probability into [0.01, 0.99] so Metaculus will accept it."""
    return max(_MIN_PROB, min(_MAX_PROB, probability))
