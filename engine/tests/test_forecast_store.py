"""Tests for the append-only ForecastStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from superforecaster.extract import TIER_HTTPX_TRAFILATURA, ExtractedDoc
from superforecaster.forecast_store import ForecastStore
from superforecaster.forecaster import EnsembleDetail, ForecastRecord
from superforecaster.research import ResearchBundle
from superforecaster.search import SearchResult

BASE = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)


def _record(
    question_id: int,
    *,
    created_at: datetime = BASE,
    cost_usd: float | None = 0.01,
    dry_run: bool = True,
    submitted_at: datetime | None = None,
) -> ForecastRecord:
    return ForecastRecord(
        created_at=created_at,
        question_id=question_id,
        post_id=question_id + 100,
        title=f"Q{question_id}",
        probability=0.5,
        raw_probability=0.5,
        confidence="medium",
        rationale="because",
        reference_class="similar cases",
        model="claude-sonnet-5",
        input_tokens=1000,
        output_tokens=500,
        cost_usd=cost_usd,
        dry_run=dry_run,
        submitted_at=submitted_at,
    )


def test_append_and_iter_roundtrip(tmp_path: Path) -> None:
    store = ForecastStore(tmp_path)
    written = store.append([_record(1), _record(2)])
    assert written == 2
    got = list(store.iter_all())
    assert {r.question_id for r in got} == {1, 2}
    assert got[0].reference_class == "similar cases"


def test_files_named_by_utc_date(tmp_path: Path) -> None:
    store = ForecastStore(tmp_path)
    store.append([_record(1, created_at=BASE)])
    assert (tmp_path / "2026-07-09.jsonl").exists()


def test_stats(tmp_path: Path) -> None:
    store = ForecastStore(tmp_path)
    store.append(
        [
            _record(1, cost_usd=0.01, dry_run=True),
            _record(
                2,
                created_at=BASE + timedelta(days=1),
                cost_usd=0.02,
                dry_run=False,
                submitted_at=BASE + timedelta(days=1),
            ),
            _record(3, cost_usd=None, dry_run=True),
        ]
    )
    s = store.stats()
    assert s.total == 3
    assert s.dry_run == 2
    assert s.submitted == 1
    assert s.total_cost_usd == 0.03  # unpriced record contributes nothing


def test_record_with_research_bundle_roundtrips(tmp_path: Path) -> None:
    bundle = ResearchBundle(
        queries=["q1", "q2"],
        results=[
            SearchResult(
                url="https://ex.com/a",
                title="A",
                snippet="snip",
                published_at=BASE,
                source_domain="ex.com",
                provider="asknews",
                raw={"raw_content": "kept short"},
            )
        ],
        docs=[
            ExtractedDoc(
                url="https://ex.com/a",
                text="body",
                title="A",
                published_at=BASE,
                tier=TIER_HTTPX_TRAFILATURA,
                ok=True,
            )
        ],
        tier_stats={TIER_HTTPX_TRAFILATURA: 1},
        search_credits_used=4,
        news_quota_exhausted=True,
        retrieved_at=BASE,
        input_tokens=100,
        output_tokens=40,
        cost_usd=0.001,
    )
    record = _record(1).model_copy(update={"research": bundle, "research_search_credits": 4})

    store = ForecastStore(tmp_path)
    store.append([record])
    (loaded,) = list(store.iter_all())

    assert loaded.research_search_credits == 4
    assert loaded.research is not None
    assert loaded.research.queries == ["q1", "q2"]
    assert loaded.research.results[0].url == "https://ex.com/a"
    assert loaded.research.results[0].published_at == BASE
    assert loaded.research.docs[0].tier == TIER_HTTPX_TRAFILATURA
    assert loaded.research.tier_stats == {TIER_HTTPX_TRAFILATURA: 1}
    assert loaded.research.news_quota_exhausted is True


def test_defaults_for_new_method_fields(tmp_path: Path) -> None:
    """A plain record round-trips with the ensemble fields at their defaults."""
    store = ForecastStore(tmp_path)
    store.append([_record(1)])
    (loaded,) = list(store.iter_all())
    assert loaded.method == "checklist-a"
    assert loaded.frame is None
    assert loaded.ensemble is None


def test_ensemble_fields_roundtrip(tmp_path: Path) -> None:
    """A pooled record's method/frame/ensemble detail survive a store round-trip."""
    detail = EnsembleDetail(
        member_frames=["outside-first", "inside-first"],
        member_models=["m0", "m1"],
        member_probabilities=[0.4, 0.6],
        spread=0.81,
        coefficient_applied=1.2,
        extremization_mode="agreement",
        member_failures=["status-quo/m2: boom"],
    )
    member = _record(1).model_copy(update={"method": "ensemble-b-member", "frame": "outside-first"})
    pooled = _record(1).model_copy(
        update={"method": "ensemble-b", "model": "ensemble-b", "ensemble": detail}
    )

    store = ForecastStore(tmp_path)
    store.append([member, pooled])
    loaded = list(store.iter_all())

    member_out = next(r for r in loaded if r.method == "ensemble-b-member")
    pooled_out = next(r for r in loaded if r.method == "ensemble-b")
    assert member_out.frame == "outside-first"
    assert member_out.ensemble is None
    assert pooled_out.frame is None
    assert pooled_out.ensemble is not None
    assert pooled_out.ensemble.member_frames == ["outside-first", "inside-first"]
    assert pooled_out.ensemble.member_probabilities == [0.4, 0.6]
    assert pooled_out.ensemble.extremization_mode == "agreement"
    assert pooled_out.ensemble.member_failures == ["status-quo/m2: boom"]


def test_old_jsonl_line_without_new_fields_parses(tmp_path: Path) -> None:
    """A pre-ensemble JSONL line (no method/frame/ensemble keys) still loads."""
    # A minimal record as written before the ensemble fields existed.
    line = (
        '{"created_at": "2026-07-09T12:00:00+00:00", "question_id": 7, '
        '"post_id": 107, "title": "Legacy", "probability": 0.5, '
        '"raw_probability": 0.5, "confidence": "medium", "rationale": "r", '
        '"reference_class": "c", "model": "claude-sonnet-5", "input_tokens": 10, '
        '"output_tokens": 5, "cost_usd": 0.01, "dry_run": true}'
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "2026-07-09.jsonl").write_text(line + "\n", encoding="utf-8")

    (loaded,) = list(ForecastStore(tmp_path).iter_all())
    assert loaded.question_id == 7
    assert loaded.method == "checklist-a"  # default fills in
    assert loaded.frame is None
    assert loaded.ensemble is None


def test_empty_store(tmp_path: Path) -> None:
    store = ForecastStore(tmp_path / "nope")
    assert list(store.iter_all()) == []
    s = store.stats()
    assert s.total == 0
    assert s.total_cost_usd == 0.0
