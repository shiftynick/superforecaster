"""Tests for the append-only SnapshotStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from superforecaster.models import QuestionSnapshot
from superforecaster.store import SnapshotStore

BASE = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)


def _snap(
    question_id: int,
    post_id: int,
    *,
    scraped_at: datetime = BASE,
    status: str = "open",
    resolution: str | None = None,
) -> QuestionSnapshot:
    return QuestionSnapshot(
        scraped_at=scraped_at,
        post_id=post_id,
        question_id=question_id,
        title=f"Q{question_id}",
        status=status,
        resolution=resolution,
    )


def test_append_and_iter_roundtrip(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    written = store.append([_snap(1, 11), _snap(2, 12)])
    assert written == 2
    got = list(store.iter_all())
    assert {s.question_id for s in got} == {1, 2}


def test_latest_by_question_prefers_newest(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    older = _snap(1, 11, scraped_at=BASE, status="open")
    newer = _snap(1, 11, scraped_at=BASE + timedelta(days=1), status="resolved", resolution="no")
    store.append([older])
    store.append([newer])

    latest = store.latest_by_question()
    assert set(latest) == {1}
    assert latest[1].status == "resolved"
    assert latest[1].resolution == "no"


def test_unresolved_post_ids(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.append(
        [
            _snap(1, 11, status="open"),
            _snap(2, 12, status="resolved", resolution="yes"),
            _snap(3, 13, status="closed"),
        ]
    )
    assert store.unresolved_question_post_ids() == [11, 13]


def test_stats(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.append(
        [
            _snap(1, 11, scraped_at=BASE, status="open"),
            _snap(1, 11, scraped_at=BASE + timedelta(days=2), status="resolved", resolution="yes"),
            _snap(2, 12, scraped_at=BASE + timedelta(days=1), status="open"),
        ]
    )
    s = store.stats()
    assert s.total_records == 3
    assert s.unique_questions == 2
    assert s.resolved_questions == 1
    assert s.first_scrape == BASE
    assert s.last_scrape == BASE + timedelta(days=2)


def test_empty_store_stats(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "does-not-exist")
    s = store.stats()
    assert s.total_records == 0
    assert s.first_scrape is None
    assert list(store.iter_all()) == []


def test_files_named_by_utc_date(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.append([_snap(1, 11, scraped_at=BASE)])
    assert (tmp_path / "2026-07-09.jsonl").exists()
