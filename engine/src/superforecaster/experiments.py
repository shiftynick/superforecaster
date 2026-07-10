"""Status tracking for pre-registered forecasting experiments.

A pre-registered experiment (see ``docs/experiments/``) freezes a question set, a
metric, and two or more *arms* — competing configurations scored side by side —
before any forecast is graded. This module reads a machine-readable manifest
(``NNN-slug.toml``, the mirror of the frozen markdown contract), matches stored
forecast records to each arm, and reports interim progress: how much of the
question set each arm has covered, how many pinned questions have resolved, the
interim paired Brier delta between the arms, and cost against the budget.

The math is deliberately pure. :func:`compute_status` takes plain data — the
manifest, an iterable of :class:`ForecastRecord`, and a resolution map — and
returns a :class:`ExperimentStatus`; it does no IO. The IO seams
(:func:`load_manifests`, :func:`resolutions_from_store`) are thin wrappers so the
scoring can be property-tested without touching disk or the network.

Terms, glossed once:

- **Arm**: one configuration under comparison (here arm A = a single baseline
  pass, arm B = the pooled ensemble). An arm "matches" a record when the record's
  ``method`` (and, optionally, ``model``) equals the arm's.
- **Paired Brier delta**: for each question that BOTH arms forecast and that then
  resolved yes/no, score each arm's forecast with the Brier rule (squared error,
  lower is better) and take ``B - A``. The mean over those paired questions is the
  headline number — negative means the ensemble beat the baseline. It is *paired*
  because both arms saw the same question, which cancels question-difficulty
  noise.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date
from pathlib import Path

from .forecaster import ForecastRecord
from .metrics import brier_score

# Default location of experiment manifests, relative to the repo root (CLI
# commands run from there). Kept as a string so callers can override cheaply.
DEFAULT_EXPERIMENTS_DIR = "./docs/experiments"


@dataclass(frozen=True)
class ArmSpec:
    """How to recognise a forecast record as belonging to one arm.

    Attributes:
        name: The arm key from the manifest (e.g. ``"A"``).
        label: A human-readable name for reports (e.g. ``"ensemble"``).
        kind: A descriptive tag (``"single"`` / ``"ensemble-pooled"``) carried
            through for reports; not used for matching.
        method: The ``ForecastRecord.method`` value this arm's records carry.
        model: An optional ``ForecastRecord.model`` value to additionally require;
            ``None`` matches any model.
    """

    name: str
    label: str
    kind: str
    method: str
    model: str | None = None

    def matches(self, record: ForecastRecord) -> bool:
        """Return whether ``record`` belongs to this arm (method, then model)."""
        if record.method != self.method:
            return False
        return self.model is None or record.model == self.model


@dataclass(frozen=True)
class ExperimentManifest:
    """The frozen, machine-readable description of one experiment.

    Attributes:
        id: The experiment number as a string (e.g. ``"001"``).
        title: The experiment title.
        status: Lifecycle status (``"draft"`` / ``"frozen"`` / ``"complete"`` /
            ``"aborted"``).
        frozen: The freeze date. Forecasts made on or after this date count.
        budget_usd: Maximum LLM spend registered for the experiment.
        asknews_credit_budget: Maximum AskNews search credits registered.
        question_ids: The pinned question ids, in the frozen order.
        arms: The arms keyed by name, in manifest order.
        source_path: Where this manifest was loaded from (``None`` if built
            in-memory, e.g. in tests).
    """

    id: str
    title: str
    status: str
    frozen: date
    budget_usd: float
    asknews_credit_budget: int
    question_ids: tuple[int, ...]
    arms: tuple[ArmSpec, ...]
    source_path: Path | None = None

    @property
    def pinned(self) -> frozenset[int]:
        """The pinned question ids as a set (order-independent membership tests)."""
        return frozenset(self.question_ids)


@dataclass(frozen=True)
class ArmStatus:
    """Per-arm progress within an experiment.

    Attributes:
        spec: The arm this status is for.
        covered_qids: Pinned question ids this arm has at least one in-window
            record for.
        matched_records: How many in-window records matched this arm (all of
            them, including re-runs — this is the true spend basis).
        total_cost_usd: Sum of known ``cost_usd`` over the matched records.
        unknown_cost_records: How many matched records had ``cost_usd`` unknown
            (``None``); their cost is not in ``total_cost_usd``.
    """

    spec: ArmSpec
    covered_qids: frozenset[int]
    matched_records: int
    total_cost_usd: float
    unknown_cost_records: int

    @property
    def coverage(self) -> int:
        """How many pinned questions this arm has covered."""
        return len(self.covered_qids)


@dataclass(frozen=True)
class PairedScore:
    """One question scored under both arms (used for the paired Brier delta).

    Attributes:
        question_id: The pinned question.
        outcome: The resolved outcome (``True`` = yes).
        brier_a: Brier score of the first arm's latest forecast.
        brier_b: Brier score of the second arm's latest forecast.
    """

    question_id: int
    outcome: bool
    brier_a: float
    brier_b: float

    @property
    def delta(self) -> float:
        """The paired Brier delta for this question (B - A)."""
        return self.brier_b - self.brier_a


@dataclass(frozen=True)
class ExperimentStatus:
    """The full interim status of one experiment.

    Attributes:
        manifest: The experiment this status is for.
        arms: Per-arm status, in manifest order.
        resolved_qids: Pinned questions that have resolved yes/no.
        resolutions: The resolution string per resolved pinned question.
        paired: Per-question paired scores (both arms have a record, resolved
            yes/no), in pinned order.
        budget_remaining_usd: ``budget_usd`` minus total known cost across arms.
    """

    manifest: ExperimentManifest
    arms: tuple[ArmStatus, ...]
    resolved_qids: frozenset[int]
    resolutions: Mapping[int, str]
    paired: tuple[PairedScore, ...]
    budget_remaining_usd: float

    @property
    def paired_brier_delta(self) -> float | None:
        """Mean paired Brier delta (B - A), or ``None`` when no paired questions."""
        if not self.paired:
            return None
        return sum(p.delta for p in self.paired) / len(self.paired)

    @property
    def total_known_cost_usd(self) -> float:
        """Sum of known cost across all arms."""
        return sum(a.total_cost_usd for a in self.arms)


def binary_outcome(resolution: str | None) -> bool | None:
    """Map a raw resolution string to a yes/no outcome, or ``None`` if neither.

    Only ``"yes"`` and ``"no"`` (case-insensitive) count; ``"ambiguous"``,
    ``"annulled"``, and anything else — including ``None`` for an open question —
    return ``None`` so they are excluded from scoring.
    """
    if resolution is None:
        return None
    value = resolution.strip().lower()
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


def _in_window(record: ForecastRecord, frozen: date) -> bool:
    """Whether a record was created on or after the freeze date (UTC calendar)."""
    created = record.created_at
    day = created.astimezone(UTC).date() if created.tzinfo is not None else created.date()
    return day >= frozen


def compute_status(
    manifest: ExperimentManifest,
    records: Iterable[ForecastRecord],
    resolutions: Mapping[int, str | None],
) -> ExperimentStatus:
    """Score an experiment's interim progress from records and resolutions.

    Pure: no IO. Records outside the pinned set or made before the freeze date are
    ignored. For each arm we track coverage, the true matched-record count, and
    cost (known sum plus an unknown-cost tally). The paired Brier delta uses the
    *latest* in-window record per arm per question (a re-run supersedes an earlier
    forecast) over questions that both arms forecast and that resolved yes/no.

    Args:
        manifest: The experiment to score.
        records: Any iterable of forecast records (the whole store is fine —
            non-pinned and pre-freeze records are filtered out here).
        resolutions: Raw resolution string per question id (``None`` = open).
            Only pinned questions are consulted.

    Returns:
        The :class:`ExperimentStatus`.
    """
    pinned = manifest.pinned

    # Per arm: covered qids, record count, known-cost sum, unknown-cost tally, and
    # the latest in-window record per pinned question (for scoring).
    covered: dict[str, set[int]] = {arm.name: set() for arm in manifest.arms}
    counts: dict[str, int] = {arm.name: 0 for arm in manifest.arms}
    cost_sum: dict[str, float] = {arm.name: 0.0 for arm in manifest.arms}
    unknown: dict[str, int] = {arm.name: 0 for arm in manifest.arms}
    latest: dict[str, dict[int, ForecastRecord]] = {arm.name: {} for arm in manifest.arms}

    for record in records:
        if record.question_id not in pinned:
            continue
        if not _in_window(record, manifest.frozen):
            continue
        for arm in manifest.arms:
            if not arm.matches(record):
                continue
            covered[arm.name].add(record.question_id)
            counts[arm.name] += 1
            if record.cost_usd is None:
                unknown[arm.name] += 1
            else:
                cost_sum[arm.name] += record.cost_usd
            current = latest[arm.name].get(record.question_id)
            if current is None or record.created_at >= current.created_at:
                latest[arm.name][record.question_id] = record
            break  # a record belongs to at most one arm

    arm_statuses = tuple(
        ArmStatus(
            spec=arm,
            covered_qids=frozenset(covered[arm.name]),
            matched_records=counts[arm.name],
            total_cost_usd=cost_sum[arm.name],
            unknown_cost_records=unknown[arm.name],
        )
        for arm in manifest.arms
    )

    # Resolutions restricted to pinned questions that resolved yes/no.
    resolved_qids: set[int] = set()
    resolved_str: dict[int, str] = {}
    outcomes: dict[int, bool] = {}
    for qid in manifest.question_ids:
        raw = resolutions.get(qid)
        outcome = binary_outcome(raw)
        if outcome is None:
            continue
        resolved_qids.add(qid)
        assert raw is not None  # binary_outcome rejects None
        resolved_str[qid] = raw
        outcomes[qid] = outcome

    # Paired Brier delta: needs exactly the two arms both present per question.
    paired: list[PairedScore] = []
    if len(manifest.arms) >= 2:
        arm_a, arm_b = manifest.arms[0].name, manifest.arms[1].name
        for qid in manifest.question_ids:
            if qid not in outcomes:
                continue
            rec_a = latest[arm_a].get(qid)
            rec_b = latest[arm_b].get(qid)
            if rec_a is None or rec_b is None:
                continue
            outcome = outcomes[qid]
            paired.append(
                PairedScore(
                    question_id=qid,
                    outcome=outcome,
                    brier_a=brier_score(rec_a.probability, outcome),
                    brier_b=brier_score(rec_b.probability, outcome),
                )
            )

    total_known = sum(cost_sum.values())
    return ExperimentStatus(
        manifest=manifest,
        arms=arm_statuses,
        resolved_qids=frozenset(resolved_qids),
        resolutions=resolved_str,
        paired=tuple(paired),
        budget_remaining_usd=manifest.budget_usd - total_known,
    )


def _require(table: Mapping[str, object], key: str, source: Path) -> object:
    """Return ``table[key]`` or raise a clear error naming the manifest file."""
    if key not in table:
        raise ValueError(f"{source}: manifest is missing required key {key!r}")
    return table[key]


def _parse_arm(name: str, raw: object, source: Path) -> ArmSpec:
    """Build one :class:`ArmSpec` from a ``[arms.<name>]`` table."""
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: [arms.{name}] must be a table")
    method = raw.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError(f"{source}: [arms.{name}] needs a non-empty 'method' string")
    label = raw.get("label")
    kind = raw.get("kind")
    model = raw.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError(f"{source}: [arms.{name}] 'model' must be a string if set")
    return ArmSpec(
        name=name,
        label=label if isinstance(label, str) and label else name,
        kind=kind if isinstance(kind, str) else "",
        method=method,
        model=model if isinstance(model, str) and model else None,
    )


def parse_manifest(data: Mapping[str, object], source: Path) -> ExperimentManifest:
    """Build an :class:`ExperimentManifest` from parsed TOML data.

    Args:
        data: The already-parsed TOML mapping.
        source: The file it came from (used only for error messages and the
            manifest's ``source_path``).

    Raises:
        ValueError: If a required key is missing or a field is the wrong type.
    """
    id_raw = _require(data, "id", source)
    if not isinstance(id_raw, str):
        raise ValueError(f"{source}: 'id' must be a string")
    title = _require(data, "title", source)
    if not isinstance(title, str):
        raise ValueError(f"{source}: 'title' must be a string")
    status = _require(data, "status", source)
    if not isinstance(status, str):
        raise ValueError(f"{source}: 'status' must be a string")
    frozen = _require(data, "frozen", source)
    if not isinstance(frozen, date):
        raise ValueError(f"{source}: 'frozen' must be a TOML date (e.g. 2026-07-10)")

    budget = _require(data, "budget_usd", source)
    if isinstance(budget, bool) or not isinstance(budget, int | float):
        raise ValueError(f"{source}: 'budget_usd' must be a number")
    credits = _require(data, "asknews_credit_budget", source)
    if isinstance(credits, bool) or not isinstance(credits, int):
        raise ValueError(f"{source}: 'asknews_credit_budget' must be an integer")

    qids_raw = _require(data, "question_ids", source)
    if not isinstance(qids_raw, list) or not all(
        isinstance(q, int) and not isinstance(q, bool) for q in qids_raw
    ):
        raise ValueError(f"{source}: 'question_ids' must be a list of integers")

    arms_raw = _require(data, "arms", source)
    if not isinstance(arms_raw, dict) or not arms_raw:
        raise ValueError(f"{source}: 'arms' must be a non-empty table of arm tables")
    arms = tuple(_parse_arm(name, raw, source) for name, raw in arms_raw.items())

    return ExperimentManifest(
        id=id_raw,
        title=title,
        status=status,
        frozen=frozen,
        budget_usd=float(budget),
        asknews_credit_budget=credits,
        question_ids=tuple(qids_raw),
        arms=arms,
        source_path=source,
    )


def load_manifest(path: Path) -> ExperimentManifest:
    """Load and parse a single experiment manifest TOML file."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return parse_manifest(data, path)


def load_manifests(directory: Path | None = None) -> list[ExperimentManifest]:
    """Load every ``*.toml`` manifest in ``directory``, sorted by id.

    Args:
        directory: Where the manifests live. Defaults to
            :data:`DEFAULT_EXPERIMENTS_DIR` (``./docs/experiments``, relative to
            the repo root).

    Returns:
        The manifests, ordered by their ``id``. An absent directory yields an
        empty list rather than raising.
    """
    root = Path(directory) if directory is not None else Path(DEFAULT_EXPERIMENTS_DIR)
    if not root.exists():
        return []
    manifests = [load_manifest(path) for path in sorted(root.glob("*.toml"))]
    manifests.sort(key=lambda m: m.id)
    return manifests


def resolutions_from_store(
    latest_by_question: Mapping[int, object],
    question_ids: Iterable[int],
) -> dict[int, str | None]:
    """Extract a ``{question_id: resolution}`` map for the given questions.

    ``latest_by_question`` is a :meth:`SnapshotStore.latest_by_question` result
    (each value a :class:`QuestionSnapshot`); only its ``resolution`` attribute is
    read, so any object exposing that works. Questions absent from the store map
    to ``None`` (unknown / not snapshotted).
    """
    out: dict[int, str | None] = {}
    for qid in question_ids:
        snapshot = latest_by_question.get(qid)
        out[qid] = getattr(snapshot, "resolution", None) if snapshot is not None else None
    return out
