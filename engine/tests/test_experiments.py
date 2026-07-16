"""Tests for experiment manifest parsing, arm matching, and status scoring."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from superforecaster.experiments import (
    ArmSpec,
    ExperimentManifest,
    binary_outcome,
    compute_status,
    load_manifest,
    load_manifests,
    parse_manifest,
    resolutions_from_store,
)
from superforecaster.forecaster import ForecastRecord

FROZEN = date(2026, 7, 10)
# A record made on the freeze date (counts) and one made the day before (excluded).
ON = datetime(2026, 7, 10, 9, 0, tzinfo=UTC)
LATER = datetime(2026, 7, 10, 18, 0, tzinfo=UTC)
BEFORE = datetime(2026, 7, 9, 23, 0, tzinfo=UTC)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_MANIFEST = REPO_ROOT / "docs" / "experiments" / "001-ensemble-vs-baseline.toml"


def _record(
    question_id: int,
    *,
    method: str,
    model: str,
    probability: float = 0.5,
    created_at: datetime = ON,
    cost_usd: float | None = 0.01,
) -> ForecastRecord:
    """Build a ForecastRecord matching the real stored shape."""
    return ForecastRecord(
        created_at=created_at,
        question_id=question_id,
        post_id=question_id + 100,
        title=f"Q{question_id}",
        probability=probability,
        raw_probability=probability,
        confidence="medium",
        rationale="because",
        reference_class="similar cases",
        model=model,
        input_tokens=1000,
        output_tokens=500,
        cost_usd=cost_usd,
        method=method,
    )


def _arm_a(qid: int, **kw: object) -> ForecastRecord:
    return _record(qid, method="checklist-a", model="z-ai/glm-5.2", **kw)  # type: ignore[arg-type]


def _arm_b(qid: int, **kw: object) -> ForecastRecord:
    return _record(qid, method="ensemble-b", model="ensemble-b", **kw)  # type: ignore[arg-type]


def _manifest(question_ids: tuple[int, ...] = (1, 2, 3)) -> ExperimentManifest:
    return ExperimentManifest(
        id="001",
        title="Test",
        status="frozen",
        frozen=FROZEN,
        budget_usd=40.0,
        asknews_credit_budget=250,
        question_ids=question_ids,
        arms=(
            ArmSpec("A", "single", "single", "checklist-a", "z-ai/glm-5.2"),
            ArmSpec("B", "ensemble", "ensemble-pooled", "ensemble-b"),
        ),
    )


# --- manifest parsing ------------------------------------------------------


def test_parse_manifest_reads_all_fields(tmp_path: Path) -> None:
    text = """
    id = "007"
    title = "Sample"
    status = "frozen"
    frozen = 2026-07-10
    budget_usd = 40.0
    asknews_credit_budget = 250
    question_ids = [11, 22, 33]

    [arms.A]
    label = "single"
    kind = "single"
    method = "checklist-a"
    model = "z-ai/glm-5.2"

    [arms.B]
    label = "ensemble"
    kind = "ensemble-pooled"
    method = "ensemble-b"
    """
    path = tmp_path / "007-x.toml"
    path.write_text(text, encoding="utf-8")

    m = load_manifest(path)
    assert m.id == "007"
    assert m.frozen == date(2026, 7, 10)
    assert m.budget_usd == 40.0
    assert m.asknews_credit_budget == 250
    assert m.question_ids == (11, 22, 33)
    assert [a.name for a in m.arms] == ["A", "B"]
    assert m.arms[0].method == "checklist-a"
    assert m.arms[0].model == "z-ai/glm-5.2"
    assert m.arms[1].model is None  # no model pinned => matches any


def test_parse_manifest_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="missing required key 'title'"):
        parse_manifest({"id": "1"}, Path("x.toml"))


def test_parse_manifest_bad_arm_raises() -> None:
    data = {
        "id": "1",
        "title": "t",
        "status": "frozen",
        "frozen": date(2026, 7, 10),
        "budget_usd": 1.0,
        "asknews_credit_budget": 1,
        "question_ids": [1],
        "arms": {"A": {"label": "x"}},  # no method
    }
    with pytest.raises(ValueError, match="needs a non-empty 'method'"):
        parse_manifest(data, Path("x.toml"))


def test_real_manifest_loads_and_matches_doc() -> None:
    """The committed 001 manifest parses and has the 40 pinned ids and two arms."""
    m = load_manifest(REAL_MANIFEST)
    assert m.id == "001"
    assert m.status == "frozen"
    assert m.frozen == date(2026, 7, 10)
    assert m.budget_usd == 40.0
    assert m.asknews_credit_budget == 250
    assert len(m.question_ids) == 40
    assert 43937 in m.pinned and 44241 in m.pinned
    names = {a.name for a in m.arms}
    assert names == {"A", "B"}
    arm_a = next(a for a in m.arms if a.name == "A")
    arm_b = next(a for a in m.arms if a.name == "B")
    assert arm_a.method == "checklist-a" and arm_a.model == "z-ai/glm-5.2"
    assert arm_b.method == "ensemble-b"


def test_load_manifests_dir(tmp_path: Path) -> None:
    (tmp_path / "001-a.toml").write_text(
        'id="001"\ntitle="A"\nstatus="frozen"\nfrozen=2026-07-10\n'
        "budget_usd=1.0\nasknews_credit_budget=1\nquestion_ids=[1]\n"
        '[arms.A]\nmethod="checklist-a"\n',
        encoding="utf-8",
    )
    manifests = load_manifests(tmp_path)
    assert [m.id for m in manifests] == ["001"]


def test_load_manifests_absent_dir_is_empty(tmp_path: Path) -> None:
    assert load_manifests(tmp_path / "nope") == []


# --- arm matching ----------------------------------------------------------


def test_arm_matches_method_and_model() -> None:
    arm = ArmSpec("A", "single", "single", "checklist-a", "z-ai/glm-5.2")
    assert arm.matches(_arm_a(1))
    assert not arm.matches(_arm_b(1))  # wrong method
    # right method, wrong model:
    assert not arm.matches(_record(1, method="checklist-a", model="other"))


def test_arm_without_model_matches_any_model() -> None:
    arm = ArmSpec("B", "ensemble", "ensemble-pooled", "ensemble-b")
    assert arm.matches(_arm_b(1))
    assert arm.matches(_record(1, method="ensemble-b", model="something-else"))


def test_ensemble_member_matches_neither_arm() -> None:
    m = _manifest()
    member = _record(1, method="ensemble-b-member", model="z-ai/glm-5.2")
    status = compute_status(m, [member], {1: None})
    assert all(a.coverage == 0 for a in status.arms)


def test_record_stamped_with_other_experiment_is_excluded() -> None:
    """A record stamped for another experiment never matches this manifest's arms."""
    m = _manifest()  # id "001"
    other = _arm_a(1).model_copy(update={"experiment_id": "002"})
    status = compute_status(m, [other], {1: None})
    assert all(a.coverage == 0 for a in status.arms)
    assert all(a.matched_records == 0 for a in status.arms)


def test_records_stamped_same_or_unstamped_are_included() -> None:
    """Stamped-with-this-id and unstamped (legacy) records both stay eligible."""
    m = _manifest()  # id "001"
    same = _arm_a(1).model_copy(update={"experiment_id": "001"})
    legacy = _arm_a(2)  # experiment_id None
    status = compute_status(m, [same, legacy], {})
    assert status.arms[0].covered_qids == frozenset({1, 2})
    assert status.arms[0].matched_records == 2


def test_stamped_other_record_does_not_displace_own_latest() -> None:
    """The scoring-integrity bug: a later, other-experiment record with the same
    method/model must not supersede this experiment's own forecast."""
    m = _manifest((1,))
    own = _arm_a(1, probability=0.9, created_at=ON).model_copy(update={"experiment_id": "001"})
    intruder = _arm_a(1, probability=0.2, created_at=LATER).model_copy(
        update={"experiment_id": "002"}
    )
    own_b = _arm_b(1, probability=0.5).model_copy(update={"experiment_id": "001"})
    status = compute_status(m, [own, intruder, own_b], {1: "yes"})
    (paired,) = status.paired
    # Arm A must be scored on ITS 0.9, not the intruder's later 0.2.
    assert paired.brier_a == pytest.approx((0.9 - 1.0) ** 2)


# --- coverage & window -----------------------------------------------------


def test_coverage_counts_distinct_pinned_questions() -> None:
    m = _manifest((1, 2, 3))
    records = [_arm_a(1), _arm_a(2), _arm_b(1)]
    status = compute_status(m, records, {})
    arm_a = status.arms[0]
    arm_b = status.arms[1]
    assert arm_a.coverage == 2
    assert arm_a.covered_qids == frozenset({1, 2})
    assert arm_b.coverage == 1


def test_non_pinned_and_prefreeze_records_excluded() -> None:
    m = _manifest((1, 2))
    records = [
        _arm_a(1),  # counts
        _arm_a(99),  # not pinned
        _arm_a(2, created_at=BEFORE),  # before freeze
    ]
    status = compute_status(m, records, {})
    assert status.arms[0].coverage == 1
    assert status.arms[0].covered_qids == frozenset({1})


# --- cost & budget ---------------------------------------------------------


def test_cost_sum_and_unknown_tally_and_budget() -> None:
    m = _manifest((1, 2, 3))
    records = [
        _arm_a(1, cost_usd=0.10),
        _arm_a(2, cost_usd=None),  # unknown
        _arm_b(1, cost_usd=1.00),
        _arm_b(2, cost_usd=None),  # unknown
    ]
    status = compute_status(m, records, {})
    arm_a, arm_b = status.arms
    assert arm_a.total_cost_usd == pytest.approx(0.10)
    assert arm_a.unknown_cost_records == 1
    assert arm_b.total_cost_usd == pytest.approx(1.00)
    assert arm_b.unknown_cost_records == 1
    assert status.total_known_cost_usd == pytest.approx(1.10)
    assert status.budget_remaining_usd == pytest.approx(40.0 - 1.10)


def test_reruns_all_count_toward_records_and_cost() -> None:
    m = _manifest((1,))
    records = [_arm_a(1, cost_usd=0.10), _arm_a(1, cost_usd=0.20, created_at=LATER)]
    status = compute_status(m, records, {})
    arm_a = status.arms[0]
    assert arm_a.coverage == 1
    assert arm_a.matched_records == 2
    assert arm_a.total_cost_usd == pytest.approx(0.30)


# --- paired Brier delta ----------------------------------------------------


def test_paired_brier_delta_math() -> None:
    m = _manifest((1, 2))
    # q1 resolves YES; A said 0.8 (brier 0.04), B said 0.6 (brier 0.16) -> B-A=+0.12
    # q2 resolves NO;  A said 0.5 (brier 0.25), B said 0.1 (brier 0.01) -> B-A=-0.24
    records = [
        _arm_a(1, probability=0.8),
        _arm_b(1, probability=0.6),
        _arm_a(2, probability=0.5),
        _arm_b(2, probability=0.1),
    ]
    status = compute_status(m, records, {1: "yes", 2: "no"})
    assert len(status.paired) == 2
    by_q = {p.question_id: p for p in status.paired}
    assert by_q[1].delta == pytest.approx(0.16 - 0.04)
    assert by_q[2].delta == pytest.approx(0.01 - 0.25)
    assert status.paired_brier_delta == pytest.approx(((0.12) + (-0.24)) / 2)


def test_paired_requires_both_arms_and_resolution() -> None:
    m = _manifest((1, 2, 3))
    records = [
        _arm_a(1, probability=0.8),  # q1: only arm A -> not paired
        _arm_a(2, probability=0.8),
        _arm_b(2, probability=0.6),  # q2: both arms, resolved -> paired
        _arm_a(3, probability=0.8),
        _arm_b(3, probability=0.6),  # q3: both arms but ambiguous -> not paired
    ]
    status = compute_status(m, records, {1: "yes", 2: "yes", 3: "ambiguous"})
    assert {p.question_id for p in status.paired} == {2}
    assert len(status.resolved_qids) == 2  # q1 and q2 resolved yes/no


def test_paired_uses_latest_record_per_arm() -> None:
    m = _manifest((1,))
    records = [
        _arm_a(1, probability=0.9, created_at=ON),  # earlier
        _arm_a(1, probability=0.2, created_at=LATER),  # latest wins
        _arm_b(1, probability=0.5),
    ]
    status = compute_status(m, records, {1: "yes"})
    (paired,) = status.paired
    # latest A prob 0.2 on YES -> brier 0.64; B 0.5 -> 0.25; delta = 0.25-0.64
    assert paired.brier_a == pytest.approx((0.2 - 1.0) ** 2)
    assert paired.delta == pytest.approx(0.25 - 0.64)


def test_no_paired_questions_gives_none_delta() -> None:
    m = _manifest((1,))
    status = compute_status(m, [_arm_a(1)], {1: None})
    assert status.paired_brier_delta is None


# --- helpers ---------------------------------------------------------------


def test_binary_outcome() -> None:
    assert binary_outcome("yes") is True
    assert binary_outcome("NO") is False
    assert binary_outcome("ambiguous") is None
    assert binary_outcome("annulled") is None
    assert binary_outcome(None) is None


def test_resolutions_from_store() -> None:
    class _Snap:
        def __init__(self, resolution: str | None) -> None:
            self.resolution = resolution

    latest = {1: _Snap("yes"), 2: _Snap(None)}
    out = resolutions_from_store(latest, [1, 2, 3])
    assert out == {1: "yes", 2: None, 3: None}


def test_resolutions_from_store_web_overlay_precedence() -> None:
    class _Snap:
        def __init__(self, resolution: str | None) -> None:
            self.resolution = resolution

    # q1: API resolved -> API wins over any web value.
    # q2: API blank but present in store -> web fills the gap.
    # q3: absent from store -> web fills the gap.
    # q4: no API, no web -> stays None.
    latest = {1: _Snap("yes"), 2: _Snap(None)}
    web_overlay = {1: "no", 2: "no", 3: "yes"}
    out = resolutions_from_store(latest, [1, 2, 3, 4], web_overlay)
    assert out == {1: "yes", 2: "no", 3: "yes", 4: None}
