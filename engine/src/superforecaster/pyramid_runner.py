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
6. **Confidence** from two raw signals (weakest leaf confidence on a multiplied
   path; the structural-vs-holistic log-odds gap), both stored, the label being
   their conservative minimum.

The whole legible argument graph is persisted on one :class:`ForecastRecord`
(``method="pyramid-c"``) via its :class:`PyramidDetail` — the same "rich detail on
the single stored record" pattern the ensemble uses for :class:`EnsembleDetail`.

Guiding principle throughout: **degrade, never crash.** A failed decomposition, a
failed leaf, or a failed holistic pass each fall back to a documented default and
leave a note in the trace; the pyramid always returns a record.
"""

from __future__ import annotations

import json
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
sub-questions over a sprawling tree of guesses."""

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
        tree, pyr, search_client, client_factory, cost, notes
    )
    falsified_by_node = _falsified_by_node(assumptions)

    # 2. Leaves.
    leaf_nodes = _collect_leaves(tree)
    leaf_estimates: list[LeafEstimate] = []
    leaf_probabilities: dict[str, float] = {}
    leaf_client = client_factory(leaf_config)
    for leaf in leaf_nodes:
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
        )
        leaf_estimates.append(estimate)
        leaf_probabilities[leaf.id] = estimate.probability

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

    # 6. Confidence — two raw signals, label = their conservative minimum.
    weakest_leaf = _weakest_leaf_confidence(tree, leaf_estimates)
    gap = abs(to_log_odds(structural_probability) - to_log_odds(holistic_probability))
    gap_confidence = _confidence_from_gap(gap, ens)
    final_confidence = _min_confidence(weakest_leaf, gap_confidence)

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
            (research.search_credits_used if research is not None else 0) + verification_credits
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
    """
    grounding = leaf.grounding if leaf.grounding is not None else "evidence"
    leaf_research = research if grounding == "evidence" else None
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
    )


def _leaf_snapshot(snapshot: QuestionSnapshot, question: str) -> QuestionSnapshot:
    """A derived snapshot posing the leaf sub-question, not the original question.

    The leaf's ``question`` is self-contained plain wording, so the original
    resolution criteria / description / fine print (which are about the *root*
    question) are cleared to avoid answering the wrong thing. Ids and dates are
    kept so the trace stays attributable to the same question.
    """
    return snapshot.model_copy(
        update={
            "title": question,
            "description": "",
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


def _verify_assumptions(
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
    tree order, capped at ``max_assumption_verifications`` — and, for each, issues
    ONE search plus one cheap verdict call. The verdict flips the assumption's
    ``status``/``evidence`` in place; a duplicate assumption elsewhere in the tree
    inherits the verdict. Every failure (no search client, a search error, a bad
    verdict call) degrades that assumption to ``unverified`` with a trace note and
    never raises.

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

    credits_before = _search_credits(search_client)
    for assumption in to_verify:
        _verify_one(
            assumption, search_client, verdict_client, verdict_config, retrieval, cost, notes
        )
    credits_after = _search_credits(search_client)
    credits = _credit_delta(credits_before, credits_after)

    _propagate_verdicts(all_assumptions, to_verify)
    return all_assumptions, len(to_verify), credits


def _verify_one(
    assumption: Assumption,
    search_client: SearchClient,
    verdict_client: LlmClient,
    config: TaskModelConfig,
    retrieval: RetrievalSettings,
    cost: _CallCost,
    notes: list[str],
) -> None:
    """Run one assumption's search + verdict call, mutating it in place on success.

    Degrades to ``unverified`` (with a note) on a search error, a failed verdict
    call, or an unparseable verdict — the assumption is left as it was. The verdict
    call's tokens/cost are always folded into the running total when the call was
    made (even if its output could not be parsed).
    """
    try:
        results = search_client.search(assumption.text, max_results=retrieval.max_results)
    except SearchError as exc:
        notes.append(
            f"assumption verification search failed for {assumption.text!r} "
            f"({exc}); left unverified"
        )
        return

    try:
        result = _run(
            verdict_client,
            VERDICT_SYSTEM,
            _verdict_prompt(assumption, results),
            config,
            _VERDICT_SCHEMA,
        )
    except LlmError as exc:
        notes.append(
            f"assumption verdict call failed for {assumption.text!r} ({exc}); left unverified"
        )
        return
    cost.add_call(result.input_tokens, result.output_tokens, result.cost_usd)

    try:
        status, evidence = _parse_verdict(result.text)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        notes.append(
            f"assumption verdict did not parse for {assumption.text!r} ({exc}); left unverified"
        )
        return

    assumption.status = status
    assumption.evidence = evidence
    if status == "verified-false":
        notes.append(f"assumption verified FALSE: {assumption.text!r} — {evidence}")


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


def _falsified_by_node(assumptions: list[Assumption]) -> dict[str, list[Assumption]]:
    """Group the ``verified-false`` assumptions by their owning node id."""
    grouped: dict[str, list[Assumption]] = {}
    for assumption in assumptions:
        if assumption.status == "verified-false":
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


def _verdict_prompt(assumption: Assumption, results: list[SearchResult]) -> str:
    """Build the verdict user message: the assumption plus the search results."""
    return "\n\n".join(
        [
            f"ASSUMPTION TO CHECK:\n{assumption.text}",
            _render_verdict_sources(results),
            "Return the JSON verdict object now.",
        ]
    )


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


# --- Confidence ---------------------------------------------------------------


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
