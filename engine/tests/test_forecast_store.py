"""Tests for the append-only ForecastStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from superforecaster.extract import TIER_HTTPX_TRAFILATURA, ExtractedDoc
from superforecaster.forecast_store import ForecastStore
from superforecaster.forecaster import ForecastRecord
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


def test_empty_store(tmp_path: Path) -> None:
    store = ForecastStore(tmp_path / "nope")
    assert list(store.iter_all()) == []
    s = store.stats()
    assert s.total == 0
    assert s.total_cost_usd == 0.0
