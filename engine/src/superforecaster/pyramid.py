"""Candidate C — the recursive decomposition pyramid: schema + combination math.

The founding idea, built honestly (spec: ``docs/specs/decomposition-pyramid.md``):
break a question into the sub-questions that must be answered first, down to a
**bedrock** of researchable facts and base rates; estimate the bedrock; combine
upward with explicit structure into one probability plus a legible argument graph
a human can audit.

This module is *stage 1* — the pure, LLM-free half:

- :class:`PyramidNode` — the typed decomposition tree the decomposer returns.
- :func:`decomposition_json_schema` — the structured-output schema for that tree,
  built from the pydantic model (mirrors ``forecaster.ForecastOutput`` usage).
- :func:`parse_decomposition` — validate raw JSON into a tree and **enforce the
  structural caps** from config, pruning any over-cap subtree down to a leaf and
  recording a :class:`PruneNote` for each cut (the notes travel into the trace).
- The **combination math** (:func:`combine_conjunction`, :func:`combine_disjunction`,
  :func:`combine_conditional`, :func:`combine_tree`) plus :func:`blend` — pure,
  property-tested functions that turn leaf estimates into a structural probability.

Some terms, glossed once:

- **Conjunction / disjunction / conditional**: a conjunction node resolves YES
  only if *all* its children do (probabilities multiply); a disjunction resolves
  YES if *any* child does; a conditional is an ordered chain where each child is
  phrased "given the previous steps happened".
- **Conjunctive error compounding**: multiply several slightly-off probabilities
  and the product is badly off — almost always too low. The guards below (per-node
  clipping, the conjunct cap, and the holistic blend) exist to fight exactly this.
- **Leaf clipping**: before any child probability is multiplied it is squeezed into
  ``[clip_low, clip_high]`` (default ``[0.03, 0.97]``). Real sub-events correlate,
  so naive products overshoot toward 0; refusing to multiply by near-certainties
  blunts that.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .aggregation import pool_log_odds
from .config import PyramidSettings

# --- Typed vocabulary ---------------------------------------------------------

NodeType = Literal["conjunction", "disjunction", "conditional", "leaf"]
"""What a node is: an internal combinator or the bedrock ``leaf``."""

Grounding = Literal["model-knowledge", "evidence"]
"""How a leaf is answered: a stable fact the model knows, or current evidence.

``model-knowledge`` leaves are base rates and stable facts the model can answer
without search (they skip the research bundle); ``evidence`` leaves concern
current events and MUST ground in the shared bundle. Stored per leaf so a
postmortem can audit whether ``model-knowledge`` was ever claimed on something
that actually needed evidence.
"""

Verifiability = Literal["researchable", "unverifiable"]
"""Whether a targeted search could settle an assumption now.

``researchable`` assumptions enter the verification pass (one search + one cheap
verdict call); ``unverifiable`` ones cannot be checked yet and are surfaced but
never searched.
"""

AssumptionStatus = Literal["verified-true", "verified-false", "unverified"]
"""The runtime outcome of verifying an assumption.

``unverified`` is the default (not yet checked, or checking failed / did not
settle it); ``verified-true`` / ``verified-false`` are set by the verdict call.
"""

CombinationMode = Literal["guarded-product", "llm-synthesis"]
"""How internal nodes combine their children.

v1 ships ``guarded-product`` (transparent, property-testable). ``llm-synthesis``
(an LLM reasoning over each node's children) is a registered v2 hypothesis; the
mode dispatch in :func:`combine_tree` is the seam that lets it slot in without
touching callers.
"""

_INTERNAL_TYPES: frozenset[str] = frozenset({"conjunction", "disjunction", "conditional"})

# The grounding assigned to a subtree that is pruned down to a leaf. It was an
# internal node (so it carried no grounding of its own) and was complex enough
# that the decomposer tried to break it up, so it is treated as needing current
# evidence rather than pure model knowledge — the safer default.
_PRUNED_LEAF_GROUNDING: Grounding = "evidence"


class PyramidParseError(Exception):
    """Raised when raw decomposition JSON cannot be made into a valid tree.

    Covers both shape failures (wrong types, unknown ``node_type``, a leaf with
    children) and the grounding contract (every leaf must carry a ``grounding``
    label). Structural-cap violations are *not* errors — they are pruned and
    recorded as :class:`PruneNote`\\ s instead.
    """


# --- The tree -----------------------------------------------------------------


class Assumption(BaseModel):
    """One implicit near-certain belief a node's estimate silently rests on.

    Every estimate rides on assumptions — probability-≈1 beliefs ("the referendum
    happens as scheduled", "the source measures what we think it measures"). v2
    makes them explicit so the checkable ones can be verified with targeted
    retrieval before the estimate is made.

    Attributes:
        text: The assumption in plain, falsifiable words.
        load_bearing: Would the owning node's estimate change materially if this
            were false? Load-bearing researchable assumptions are verified first
            when the budget binds.
        verifiability: ``"researchable"`` (a search could settle it now) or
            ``"unverifiable"`` (it cannot be checked yet). Only researchable
            assumptions enter the verification pass.
        status: Runtime outcome — ``"unverified"`` until the verdict call sets it.
        evidence: One-line citation or reason produced by verification (empty
            until then).
        queries_used: The web-search queries the verification pass actually issued
            for this assumption — the targeted queries a query-gen step derived
            from the assumption text (or the assumption text itself when that step
            failed and it fell back to a verbatim search). Empty until verified (and
            on old records that predate the field, which parse with this default).
        node_id: The id of the node this assumption belongs to. Filled by the
            runner's tree-wide collector, so a flat list stays attributable to its
            node; empty on a node's own ``assumptions`` before collection.
    """

    text: str
    load_bearing: bool
    verifiability: Verifiability
    status: AssumptionStatus = "unverified"
    evidence: str = ""
    queries_used: list[str] = Field(default_factory=list)
    node_id: str = ""


class PyramidNode(BaseModel):
    """One node of the decomposition tree (v1: a tree, not a DAG).

    Attributes:
        id: A stable identifier, unique within the tree — the key the leaf
            estimates and the combination trace are addressed by.
        question: The sub-question in plain words.
        node_type: ``conjunction`` / ``disjunction`` / ``conditional`` (internal
            combinators) or ``leaf`` (bedrock — a base rate, fact, or estimate).
        reasoning_type: The reasoning mode for this node's estimate. Defaults to
            ``"checklist"``; extensible (a proven refiner can become a node-level
            mode later, per docs/03).
        grounding: REQUIRED on leaves, absent (``None``) on internal nodes — see
            :data:`Grounding`. The parser enforces both halves of that contract.
        assumptions: The implicit beliefs this node's estimate rests on (see
            :class:`Assumption`). Default empty — zero assumptions is legitimate.
            The parser truncates any list over ``max_assumptions_per_node`` with a
            :class:`PruneNote`.
        children: Sub-nodes. Empty on a leaf; one or more on an internal node.
    """

    id: str
    question: str
    node_type: NodeType
    reasoning_type: str = "checklist"
    grounding: Grounding | None = None
    assumptions: list[Assumption] = Field(default_factory=list)
    children: list[PyramidNode] = Field(default_factory=list)


# Resolve the ``list[PyramidNode]`` self-reference so ``model_json_schema`` and
# ``model_validate`` work on the recursive model.
PyramidNode.model_rebuild()


class PruneNote(BaseModel):
    """A record that one subtree was cut back to satisfy a structural cap.

    Returned alongside the parsed tree and carried into the forecast trace, so a
    reviewer can always see *what the decomposer proposed* versus *what the caps
    allowed* — never a silent truncation.

    Attributes:
        node_id: The node the cap acted on.
        cap: Which cap fired — ``"depth"`` (subtree collapsed to a leaf),
            ``"children"`` (extra children dropped), ``"conjunct"`` (a conjunction
            trimmed to the multiplied-conjunct limit), or ``"assumptions"`` (a
            node's assumption list truncated to the per-node cap).
        message: A human-readable explanation for the trace / render view.
    """

    node_id: str
    cap: Literal["depth", "children", "conjunct", "assumptions"]
    message: str


def decomposition_json_schema() -> dict[str, Any]:
    """The JSON Schema for the decomposer's structured output (the root node).

    Built from :class:`PyramidNode` via pydantic — the same approach
    ``forecaster.run_forecast_pass`` uses for ``ForecastOutput`` — so the schema
    can never drift from the model. Compatible with the ``json_schema`` parameter
    of ``llm.LlmClient.run``: providers that support server-side structured
    output constrain to it; the Claude CLI ignores it and relies on prompt-JSON
    plus :func:`parse_decomposition` downstream.
    """
    return PyramidNode.model_json_schema()


# --- Parsing + cap enforcement ------------------------------------------------


def parse_decomposition(
    raw: object,
    settings: PyramidSettings,
) -> tuple[PyramidNode, list[PruneNote]]:
    """Validate raw decomposition JSON into a tree and enforce the structural caps.

    Two stages: (1) shape validation via pydantic — a failure (wrong types,
    unknown ``node_type``) raises :class:`PyramidParseError`; (2) a top-down walk
    that enforces the grounding contract and the three caps from ``settings``:

    - ``max_depth``: an internal node at or beyond the depth limit is **pruned to
      a leaf** (its subtree dropped), grounding set to ``"evidence"``.
    - ``max_children``: an internal node with too many children keeps the first
      ``max_children`` and drops the rest.
    - ``max_conjuncts``: a ``conjunction`` node — the shape most exposed to
      conjunctive error compounding — is additionally trimmed to at most
      ``max_conjuncts`` multiplied children.

    The root is at depth 0. A lone leaf ("this question does not decompose") is a
    valid tree and returns with no prune notes.

    Args:
        raw: The decoded JSON object (a mapping) the decomposer produced.
        settings: The active :class:`PyramidSettings` supplying the caps.

    Returns:
        ``(tree, prune_notes)`` — the validated, cap-compliant tree and one
        :class:`PruneNote` per cut, in the order the cuts were made.

    Raises:
        PyramidParseError: On a malformed tree or a leaf missing its grounding.
    """
    try:
        root = PyramidNode.model_validate(raw)
    except ValidationError as exc:
        raise PyramidParseError(f"decomposition JSON is not a valid tree: {exc}") from exc

    notes: list[PruneNote] = []
    tree = _enforce(root, depth=0, settings=settings, notes=notes)
    return tree, notes


def _enforce(
    node: PyramidNode,
    *,
    depth: int,
    settings: PyramidSettings,
    notes: list[PruneNote],
) -> PyramidNode:
    """Recursively enforce the grounding contract and structural caps on ``node``.

    Appends a :class:`PruneNote` to ``notes`` for every cut and returns a new,
    cap-compliant node (the input tree is not mutated).
    """
    if node.node_type == "leaf":
        if node.grounding is None:
            raise PyramidParseError(f"leaf node {node.id!r} is missing its 'grounding' label")
        if node.children:
            raise PyramidParseError(f"leaf node {node.id!r} must not have children")
        return node.model_copy(
            deep=True, update={"assumptions": _cap_assumptions(node, settings, notes)}
        )

    # Internal node from here on.
    if not node.children:
        raise PyramidParseError(f"{node.node_type} node {node.id!r} has no children")

    # Depth cap: collapse an over-deep internal node into a single leaf.
    if depth >= settings.max_depth:
        notes.append(
            PruneNote(
                node_id=node.id,
                cap="depth",
                message=(
                    f"node {node.id!r} at depth {depth} reached the max_depth "
                    f"limit ({settings.max_depth}); its {len(node.children)}-child "
                    f"subtree was pruned to an evidence leaf"
                ),
            )
        )
        return PyramidNode(
            id=node.id,
            question=node.question,
            node_type="leaf",
            reasoning_type=node.reasoning_type,
            grounding=_PRUNED_LEAF_GROUNDING,
            assumptions=_cap_assumptions(node, settings, notes),
            children=[],
        )

    children = list(node.children)

    # Children cap: keep the first max_children, drop the overflow.
    if len(children) > settings.max_children:
        dropped = len(children) - settings.max_children
        notes.append(
            PruneNote(
                node_id=node.id,
                cap="children",
                message=(
                    f"node {node.id!r} had {len(children)} children; dropped "
                    f"{dropped} beyond the max_children limit ({settings.max_children})"
                ),
            )
        )
        children = children[: settings.max_children]

    # Conjunct cap: conjunctions AND conditional chains multiply their children
    # (combine_conditional is numerically the same product), so both are held to
    # the tighter multiplied-conjunct limit on top of the general children cap.
    if node.node_type in ("conjunction", "conditional") and len(children) > settings.max_conjuncts:
        dropped = len(children) - settings.max_conjuncts
        notes.append(
            PruneNote(
                node_id=node.id,
                cap="conjunct",
                message=(
                    f"{node.node_type} {node.id!r} had {len(children)} multiplied "
                    f"children; dropped {dropped} beyond the max_conjuncts limit "
                    f"({settings.max_conjuncts})"
                ),
            )
        )
        children = children[: settings.max_conjuncts]

    capped_assumptions = _cap_assumptions(node, settings, notes)
    new_children = [
        _enforce(child, depth=depth + 1, settings=settings, notes=notes) for child in children
    ]
    return PyramidNode(
        id=node.id,
        question=node.question,
        node_type=node.node_type,
        reasoning_type=node.reasoning_type,
        grounding=None,
        assumptions=capped_assumptions,
        children=new_children,
    )


def _cap_assumptions(
    node: PyramidNode,
    settings: PyramidSettings,
    notes: list[PruneNote],
) -> list[Assumption]:
    """Return ``node``'s assumptions, truncated to the per-node cap.

    A list over ``settings.max_assumptions_per_node`` keeps the first N and drops
    the overflow, recording one :class:`PruneNote` (``cap="assumptions"``) — the
    runaway-output guard from the spec, never a silent truncation. Each returned
    assumption is a deep copy so the enforced tree never shares mutable state with
    the input.
    """
    assumptions = list(node.assumptions)
    cap = settings.max_assumptions_per_node
    if len(assumptions) > cap:
        dropped = len(assumptions) - cap
        notes.append(
            PruneNote(
                node_id=node.id,
                cap="assumptions",
                message=(
                    f"node {node.id!r} listed {len(assumptions)} assumptions; dropped "
                    f"{dropped} beyond the max_assumptions_per_node limit ({cap})"
                ),
            )
        )
        assumptions = assumptions[:cap]
    return [a.model_copy(deep=True) for a in assumptions]


# --- Combination math ---------------------------------------------------------


def _clip(probability: float, low: float, high: float) -> float:
    """Squeeze a probability into ``[low, high]`` — the independence guard."""
    return min(max(probability, low), high)


def combine_conjunction(child_ps: list[float], clip_low: float, clip_high: float) -> float:
    """Probability a conjunction resolves YES: the product of clipped children.

    Each child is clipped to ``[clip_low, clip_high]`` *before* multiplying, so no
    single near-certain child can drag the product to an unwarranted extreme.
    """
    return math.prod(_clip(p, clip_low, clip_high) for p in child_ps)


def combine_disjunction(child_ps: list[float], clip_low: float, clip_high: float) -> float:
    """Probability a disjunction resolves YES: ``1 - ∏(1 - clipped child)``.

    The dual of the conjunction — YES if *any* child resolves YES — with the same
    per-child clipping applied before combining.
    """
    product = math.prod(1.0 - _clip(p, clip_low, clip_high) for p in child_ps)
    return 1.0 - product


def combine_conditional(child_ps: list[float], clip_low: float, clip_high: float) -> float:
    """Probability of an ordered conditional chain: ``∏`` of clipped children.

    Numerically identical to :func:`combine_conjunction`, but semantically the
    children are ``P(c₁) · P(c₂ | c₁) · …`` — each already phrased "given the
    previous steps happened" — so it is kept a distinct operation for a legible
    trace and so an alternative chain rule can replace it independently later.
    """
    return math.prod(_clip(p, clip_low, clip_high) for p in child_ps)


class CombinationStep(BaseModel):
    """One node's entry in the combination trace: the math, made inspectable.

    Attributes:
        node_id: The node this step combined.
        operation: The node type applied (``leaf`` for a bedrock estimate).
        inputs: The child node outputs fed in (a single-element ``[estimate]``
            for a leaf) — the raw values *before* clipping.
        clipped_inputs: ``inputs`` after per-node clipping (identical to
            ``inputs`` for a leaf, which is not clipped).
        output: The probability this node produced.
    """

    node_id: str
    operation: NodeType
    inputs: list[float]
    clipped_inputs: list[float]
    output: float


def _combine_node(
    node_type: NodeType,
    child_ps: list[float],
    clip_low: float,
    clip_high: float,
    mode: CombinationMode,
) -> float:
    """Dispatch one internal node's combination on ``mode`` then ``node_type``.

    This is the ``combination_mode`` seam: v1 implements ``"guarded-product"``;
    ``"llm-synthesis"`` is reserved and raises until stage 2 wires an LLM in.
    Adding it is a new branch here — :func:`combine_tree` and its callers are
    untouched.
    """
    if mode == "guarded-product":
        if node_type == "conjunction":
            return combine_conjunction(child_ps, clip_low, clip_high)
        if node_type == "disjunction":
            return combine_disjunction(child_ps, clip_low, clip_high)
        if node_type == "conditional":
            return combine_conditional(child_ps, clip_low, clip_high)
        raise ValueError(f"cannot combine a {node_type!r} node as an internal node")
    if mode == "llm-synthesis":
        raise NotImplementedError("combination_mode 'llm-synthesis' is a stage-2 hypothesis")
    raise ValueError(f"unknown combination_mode {mode!r}")


def combine_tree(
    tree: PyramidNode,
    leaf_estimates: dict[str, float],
    *,
    clip_low: float,
    clip_high: float,
    mode: CombinationMode = "guarded-product",
) -> tuple[float, dict[str, CombinationStep]]:
    """Walk the tree bottom-up into a structural probability plus a per-node trace.

    Every leaf's probability comes from ``leaf_estimates`` (keyed by node id);
    each internal node combines its children's outputs via :func:`_combine_node`,
    clipping them first. The returned trace maps every node id to the
    :class:`CombinationStep` that produced its value — this IS the legible
    argument graph the record stores.

    Args:
        tree: A cap-compliant tree from :func:`parse_decomposition`.
        leaf_estimates: Probability per leaf node id (in ``[0, 1]``).
        clip_low: Lower clip bound applied to each child before combining.
        clip_high: Upper clip bound applied to each child before combining.
        mode: The combination mode (v1: ``"guarded-product"``).

    Returns:
        ``(structural_probability, trace)``.

    Raises:
        ValueError: If a leaf has no estimate, or the mode is unknown.
        NotImplementedError: If ``mode`` is a reserved-but-unimplemented mode.
    """
    trace: dict[str, CombinationStep] = {}

    def walk(node: PyramidNode) -> float:
        if node.node_type == "leaf":
            if node.id not in leaf_estimates:
                raise ValueError(f"no leaf estimate supplied for leaf node {node.id!r}")
            estimate = leaf_estimates[node.id]
            trace[node.id] = CombinationStep(
                node_id=node.id,
                operation="leaf",
                inputs=[estimate],
                clipped_inputs=[estimate],
                output=estimate,
            )
            return estimate

        child_outputs = [walk(child) for child in node.children]
        clipped = [_clip(p, clip_low, clip_high) for p in child_outputs]
        output = _combine_node(node.node_type, child_outputs, clip_low, clip_high, mode)
        trace[node.id] = CombinationStep(
            node_id=node.id,
            operation=node.node_type,
            inputs=child_outputs,
            clipped_inputs=clipped,
            output=output,
        )
        return output

    structural_probability = walk(tree)
    return structural_probability, trace


def blend(structural_p: float, holistic_p: float, weight: float) -> float:
    """Blend the structural and holistic estimates in log-odds space.

    The pyramid's structural estimate never stands alone: one direct forecast of
    the root question (the *holistic* sanity check) is pooled with it. ``weight``
    is the weight placed on the holistic estimate — ``0`` trusts the structure
    fully, ``1`` trusts the holistic pass fully, ``0.5`` (the v1 default) splits
    the difference. Reuses :func:`aggregation.pool_log_odds`, the project's
    log-odds pooling primitive, rather than re-deriving the mixing math.

    Args:
        structural_p: The pyramid's combined probability.
        holistic_p: The direct one-shot probability for the root question.
        weight: Weight on ``holistic_p`` in ``[0, 1]``.

    Returns:
        The blended probability.

    Raises:
        ValueError: If ``weight`` is outside ``[0, 1]``.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"blend weight must be in [0, 1], got {weight}")
    return pool_log_odds([structural_p, holistic_p], [1.0 - weight, weight])
