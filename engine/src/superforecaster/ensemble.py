"""Candidate B: a diverse ensemble of forecasting passes over one research bundle.

Where candidate A (``forecaster.py``) makes a single checklist pass, this module
runs *several* passes over the **same** shared research and pools them in log-odds
space. The whole point is disagreement made productive: if independent, genuinely
different lines of reasoning still land near the same number, that agreement is
real evidence and the pooled forecast can be pushed a little more confident; if
they scatter, that disagreement is the honest signal that the question is hard.

Two axes of diversity, both config-driven (see ``config.EnsembleSettings``):

- **Reasoning frames** — named system-prompt addenda layered onto the shared
  :data:`~superforecaster.forecaster.CHECKLIST_PROMPT`. They reorder and reweight
  the *same* checklist so members reason in a different order, not merely at a
  different temperature. The three frames live in :data:`FRAMES` below.
- **Models** — each member names its own provider/model/effort, so the roster can
  span cheap and strong models and multiple providers.

Key objects:

- :data:`FRAMES` — the reasoning-frame prompt addenda (the diversity engine).
- :class:`EnsembleResult` — the pooled answer plus every member record, the
  spread, the extremization actually applied, cost, and any member failures.
- :func:`run_ensemble` — the runner: fan out members, pool survivors, extremize,
  derive confidence from spread.

Some terms, glossed once:

- **Spread**: the population standard deviation of the members' forecasts measured
  in log-odds. Small spread = the members agree; large spread = they disagree.
- **Extremization**: nudging the pooled probability away from 50%. Independent
  forecasters who each hedge tend, collectively, to be under-confident; a
  coefficient > 1 corrects for that. Here it can be a fixed value or scaled by how
  much the members agreed (:func:`~superforecaster.aggregation.agreement_scaled_coefficient`).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel

from .aggregation import (
    agreement_scaled_coefficient,
    extremize,
    pool_log_odds,
    population_std,
    to_log_odds,
)
from .config import EnsembleSettings, TaskModelConfig, load_ensemble_settings
from .forecaster import Confidence, EnsembleDetail, ForecastRecord, run_forecast_pass
from .llm import LlmClient, LlmError, build_llm_client
from .models import QuestionSnapshot
from .research import ResearchBundle

# Same Metaculus-driven clamp the single pass applies (a probability of exactly 0
# or 1 is rejected). Extremizing the pooled number can push it past these bounds,
# so we re-clamp the final ensemble probability just as ``forecaster`` does.
_MIN_PROB = 0.01
_MAX_PROB = 0.99


# The reasoning frames: each is a titled system-prompt section appended after the
# shared checklist. They deliberately impose *different reasoning orders* so the
# members are not near-duplicates — this is the ensemble's diversity engine.
FRAMES: dict[str, str] = {
    "outside-first": """\
REASONING FRAME — OUTSIDE VIEW FIRST

Work the checklist in strict outside-view order. BEFORE you consider any specific
of this particular case:

1. Name the reference class explicitly — a concrete group of comparable past
   situations — and state its base rate as a number (or an honest range with a
   chosen point). Write this down first; it is your anchor.
2. Only then read the case specifics, and move OFF that anchor in small, clearly
   justified steps. Each adjustment must cite the base rate it is moving away
   from and say why this case differs from the class.

Discipline: if you find yourself reasoning from the vivid particulars of this case
before you have written down a base rate, stop and name the reference class first.
The base rate leads; the specifics adjust it.""",
    "inside-first": """\
REASONING FRAME — INSIDE VIEW FIRST, THEN CROSS-CHECK

Reason the question-specific mechanics through FIRST, then use a base rate as an
independent cross-check:

1. Build the causal story of THIS question. What are the concrete mechanisms,
   actors, incentives, deadlines, and decision points that would drive it to YES
   or to NO? Reason it through on its own terms and reach a provisional estimate
   from the mechanics alone.
2. NOW, as a separate step, fetch a reference class and its base rate — chosen
   without peeking back at your provisional number — and compare. If your
   mechanism-driven estimate and the base rate disagree, say so explicitly and
   RECONCILE them: decide which deserves more weight for this question and why,
   and let the base rate pull your number when your causal story is thin.

Discipline: do not let the base rate silently pre-empt the causal analysis, and do
not ignore it when it contradicts you. Two independent estimates, then a reasoned
reconciliation.""",
    "status-quo": """\
REASONING FRAME — STATUS QUO, BURDEN OF PROOF ON CHANGE

Start from inertia and make change earn its probability:

1. State what happens if NOTHING changes — the default trajectory the world is
   already on for this question. Treat that as the baseline outcome.
2. Put the burden of proof on change. A YES (or a NO) that requires the status quo
   to break needs a specific, credible mechanism and enough time for it to play
   out. Absent that, weight stays on the default.
3. Run an explicit FRAGILITY TEST on your emerging forecast: "what would have to go
   wrong for this forecast to be wrong?" If your number rests on a SINGLE fragile
   assumption — one event, one actor, one deadline holding — pull the forecast back
   toward uncertainty (toward 50%), because a single point of failure is rarely as
   reliable as it feels.

Discipline: change is usually slower and rarer than the latest news makes it feel.
Do not talk yourself out of the status quo without a concrete mechanism.""",
}


class EnsembleResult(BaseModel):
    """The pooled ensemble forecast plus everything a postmortem needs.

    Attributes:
        pooled_probability: The final probability to submit — log-odds pool of the
            surviving members, extremized, then clamped to [0.01, 0.99].
        confidence: ``"high"``/``"medium"``/``"low"`` derived from member spread
            (agreement is the honest confidence signal, not the probability).
        member_records: The full :class:`ForecastRecord` of each surviving member.
        member_probabilities: The clamped probability each survivor contributed to
            the pool, in member order.
        spread: Population std dev of the survivors' log-odds (the disagreement).
        coefficient_applied: The extremization coefficient actually used.
        extremization_mode: ``"fixed"`` or ``"agreement"`` — how that coefficient
            was chosen.
        total_cost_usd: Sum of the surviving members' costs, or ``None`` when no
            member reported a cost.
        failures: One ``"label: error"`` string per member that raised and was
            skipped, so a dropped member is never silent.
    """

    pooled_probability: float
    confidence: Confidence
    member_records: list[ForecastRecord]
    member_probabilities: list[float]
    spread: float
    coefficient_applied: float
    extremization_mode: str
    total_cost_usd: float | None
    failures: list[str]


def _clamp(probability: float) -> float:
    """Squeeze a probability into [0.01, 0.99] (same clamp as the single pass)."""
    return max(_MIN_PROB, min(_MAX_PROB, probability))


def _confidence_from_spread(spread: float, settings: EnsembleSettings) -> Confidence:
    """Map member spread onto a confidence band using the config thresholds."""
    if spread <= settings.confidence_high_max_spread:
        return "high"
    if spread >= settings.confidence_low_min_spread:
        return "low"
    return "medium"


def run_ensemble(
    snapshot: QuestionSnapshot,
    *,
    settings: EnsembleSettings | None = None,
    research: ResearchBundle | None = None,
    client_factory: Callable[[TaskModelConfig], LlmClient] = build_llm_client,
    now: datetime | None = None,
) -> EnsembleResult:
    """Run the diverse ensemble for one question and return the pooled result.

    Each configured member forecasts the SAME question over the SAME ``research``
    bundle, using its own model and a reasoning frame from :data:`FRAMES`. Member
    forecasts run through the existing :func:`run_forecast_pass` (same JSON
    contract, retry, and cost logging); the frame is injected via that function's
    ``system_addendum`` seam.

    A member that raises :class:`LlmError` is recorded in ``failures`` and skipped;
    the pool forms from the survivors. Fewer than two survivors is a failed
    ensemble for this question and raises :class:`LlmError`.

    The survivors are pooled in log-odds space (equal weights), the pooled number
    is extremized (fixed coefficient, or one scaled by member agreement), and the
    result is clamped. Ensemble confidence comes from the member spread.

    Args:
        snapshot: The question to forecast.
        settings: Ensemble tunables; defaults to :func:`load_ensemble_settings`.
        research: Optional shared research bundle handed to every member.
        client_factory: Test seam — builds an :class:`LlmClient` from a member's
            :class:`TaskModelConfig`. Defaults to :func:`build_llm_client`.
        now: Current time (UTC), overridable for deterministic tests.

    Returns:
        An :class:`EnsembleResult`.

    Raises:
        ValueError: If a member names a frame not in :data:`FRAMES`.
        LlmError: If fewer than two members survive.
    """
    cfg = settings if settings is not None else load_ensemble_settings()

    # A frame typo is a config bug, not a transient member failure — fail loudly
    # and early rather than silently dropping the member into ``failures``.
    for member in cfg.members:
        if member.frame not in FRAMES:
            raise ValueError(
                f"Unknown ensemble frame {member.frame!r}; "
                f"known frames: {', '.join(sorted(FRAMES))}."
            )

    records: list[ForecastRecord] = []
    failures: list[str] = []
    for member in cfg.members:
        label = f"{member.frame}/{member.model or '(default model)'}"
        task_config = member.task_config()
        try:
            client = client_factory(task_config)
            record = run_forecast_pass(
                snapshot,
                client=client,
                config=task_config,
                research=research,
                now=now,
                system_addendum=FRAMES[member.frame],
            )
        except LlmError as exc:
            failures.append(f"{label}: {exc}")
            continue
        # Tag the member record: it is one member of an ensemble, and it carries
        # its reasoning frame. The pooled record (built by ``pooled_forecast_record``)
        # is the one that carries the ``ensemble`` detail; members carry only these.
        record = record.model_copy(update={"method": "ensemble-b-member", "frame": member.frame})
        records.append(record)

    if len(records) < 2:
        raise LlmError(
            f"Ensemble needs at least 2 surviving members but got {len(records)} "
            f"(of {len(cfg.members)}); failures: {failures or 'none'}."
        )

    member_probabilities = [r.probability for r in records]
    log_odds = [to_log_odds(p) for p in member_probabilities]
    spread = population_std(log_odds)

    if cfg.extremization_mode == "fixed":
        coefficient = cfg.extremization_coefficient
    else:
        coefficient = agreement_scaled_coefficient(
            log_odds,
            min_coefficient=cfg.agreement_min_coefficient,
            max_coefficient=cfg.agreement_max_coefficient,
        )

    pooled = _clamp(extremize(pool_log_odds(member_probabilities), coefficient))

    costs = [r.cost_usd for r in records]
    total_cost_usd = (
        sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None
    )

    return EnsembleResult(
        pooled_probability=pooled,
        confidence=_confidence_from_spread(spread, cfg),
        member_records=records,
        member_probabilities=member_probabilities,
        spread=spread,
        coefficient_applied=coefficient,
        extremization_mode=cfg.extremization_mode,
        total_cost_usd=total_cost_usd,
        failures=failures,
    )


def pooled_forecast_record(
    snapshot: QuestionSnapshot,
    result: EnsembleResult,
    *,
    research: ResearchBundle | None = None,
    now: datetime | None = None,
) -> ForecastRecord:
    """Build the stored pooled :class:`ForecastRecord` from an ensemble result.

    This is the single record that represents the ensemble's answer for a
    question (stored alongside the individual member records). It carries:

    - ``probability`` = the pooled, extremized, clamped number
      (``result.pooled_probability``);
    - ``raw_probability`` = the pre-extremization pooled value, recomputed as the
      plain log-odds pool of the members so a clamp/extremization is never hidden;
    - ``confidence`` = the ensemble confidence (derived from member spread);
    - a synthesized ``rationale`` — one header line plus one line per member;
    - token/cost sums over the surviving members;
    - the :class:`EnsembleDetail` postmortem object.

    ``research_search_credits`` mirrors the single-pass convention: it is the
    bundle's ``search_credits_used`` (``0`` without a bundle). The CLI zeroes it
    for a *cached* bundle exactly as it does for a single pass, so a reused bundle
    is never double-counted; members' credits are attributed to this pooled record
    once, not per member.

    Args:
        snapshot: The question that was forecast.
        result: The :class:`EnsembleResult` from :func:`run_ensemble`.
        research: The shared research bundle the members forecast over (or ``None``).
        now: Current time (UTC), overridable for deterministic tests.

    Returns:
        A pooled :class:`ForecastRecord` (``method``/``model`` = ``"ensemble-b"``).
    """
    moment = now if now is not None else datetime.now(UTC)
    records = result.member_records

    # Pre-extremization pooled value: the plain log-odds pool of the members. The
    # final ``pooled_probability`` is this value extremized and clamped.
    raw_pooled = pool_log_odds(result.member_probabilities)

    frames = [r.frame or "" for r in records]
    models = [r.model for r in records]

    detail = EnsembleDetail(
        member_frames=frames,
        member_models=models,
        member_probabilities=list(result.member_probabilities),
        spread=result.spread,
        coefficient_applied=result.coefficient_applied,
        extremization_mode=result.extremization_mode,
        member_failures=list(result.failures),
    )

    lines = [
        f"Pooled from {len(records)} members "
        f"(frames: {', '.join(frames)}); spread {result.spread:.3f}; "
        f"coefficient {result.coefficient_applied:.3f} ({result.extremization_mode})."
    ]
    for record in records:
        lines.append(
            f"  {record.frame}/{record.model or '(default model)'} -> {record.probability:.2f}"
        )
    if result.failures:
        lines.append(f"Skipped members: {'; '.join(result.failures)}")
    rationale = "\n".join(lines)

    # A joined summary of the reference classes the members anchored on (deduped,
    # order-preserving), so the pooled record still names its base-rate anchors.
    reference_class = "; ".join(
        dict.fromkeys(r.reference_class for r in records if r.reference_class)
    )

    input_tokens = sum(r.input_tokens for r in records)
    output_tokens = sum(r.output_tokens for r in records)
    costs = [r.cost_usd for r in records]
    cost_usd = sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None

    return ForecastRecord(
        created_at=moment,
        question_id=snapshot.question_id,
        post_id=snapshot.post_id,
        title=snapshot.title,
        probability=result.pooled_probability,
        raw_probability=raw_pooled,
        confidence=result.confidence,
        rationale=rationale,
        reference_class=reference_class,
        model="ensemble-b",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        dry_run=True,
        submitted_at=None,
        research=research,
        research_search_credits=research.search_credits_used if research is not None else 0,
        method="ensemble-b",
        frame=None,
        ensemble=detail,
    )
