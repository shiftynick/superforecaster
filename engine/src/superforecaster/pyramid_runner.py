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
    TaskModelConfig,
    load_ensemble_settings,
    load_pyramid_settings,
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
    PruneNote,
    PyramidNode,
    PyramidParseError,
    blend,
    combine_tree,
    decomposition_json_schema,
    parse_decomposition,
)
from .research import ResearchBundle

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
have "grounding". Output nothing after the closing brace."""


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

    # 2. Leaves.
    leaf_nodes = _collect_leaves(tree)
    leaf_estimates: list[LeafEstimate] = []
    leaf_probabilities: dict[str, float] = {}
    leaf_client = client_factory(leaf_config)
    for leaf in leaf_nodes:
        estimate = _estimate_leaf(
            snapshot, leaf, research, leaf_client, leaf_config, moment, cost, notes
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
        research_search_credits=research.search_credits_used if research is not None else 0,
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
) -> LeafEstimate:
    """Estimate one leaf via :func:`run_forecast_pass`, degrading on failure.

    ``evidence`` leaves get the shared research bundle in context; ``model-
    knowledge`` leaves get NONE (spec §2 — cheaper, and it keeps stable-fact
    estimates from being diluted by current-events sources). A leaf whose pass
    fails after its own retry becomes a ``0.5`` / ``"low"`` fallback with a note.
    """
    grounding = leaf.grounding if leaf.grounding is not None else "evidence"
    leaf_research = research if grounding == "evidence" else None
    leaf_snapshot = _leaf_snapshot(snapshot, leaf.question)
    try:
        record = run_forecast_pass(
            leaf_snapshot,
            client=client,
            config=config,
            research=leaf_research,
            now=moment,
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
