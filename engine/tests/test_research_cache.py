"""Tests for the on-disk research bundle cache — no network, pure filesystem."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from superforecaster.extract import TIER_HTTPX_TRAFILATURA, ExtractedDoc
from superforecaster.research import ResearchBundle
from superforecaster.research_cache import ResearchCache
from superforecaster.search import SearchResult

NOW = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)


def _bundle(retrieved_at: datetime = NOW) -> ResearchBundle:
    return ResearchBundle(
        queries=["rate cut news"],
        results=[
            SearchResult(
                url="https://news.example/a",
                title="A title",
                snippet="short",
                published_at=NOW,
                source_domain="news.example",
                provider="asknews",
                raw={},
            )
        ],
        docs=[
            ExtractedDoc(
                url="https://news.example/a",
                text="body",
                title="A title",
                published_at=NOW,
                tier=TIER_HTTPX_TRAFILATURA,
                ok=True,
            )
        ],
        tier_stats={TIER_HTTPX_TRAFILATURA: 1},
        search_credits_used=2,
        retrieved_at=retrieved_at,
        input_tokens=80,
        output_tokens=20,
        cost_usd=0.0005,
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research")
    cache.save(301, _bundle())

    loaded = cache.load(301, max_age=timedelta(hours=6), now=NOW)
    assert loaded is not None
    assert loaded.queries == ["rate cut news"]
    assert loaded.retrieved_at == NOW  # datetime round-trips
    assert loaded.search_credits_used == 2
    assert loaded.results[0].url == "https://news.example/a"
    assert loaded.docs[0].published_at == NOW


def test_load_missing_returns_none(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research")
    assert cache.load(999, max_age=timedelta(hours=6), now=NOW) is None


def test_load_stale_by_age_returns_none(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research")
    cache.save(301, _bundle(retrieved_at=NOW - timedelta(hours=8)))
    assert cache.load(301, max_age=timedelta(hours=6), now=NOW) is None
    # But still fresh within a wider window.
    assert cache.load(301, max_age=timedelta(hours=12), now=NOW) is not None


def test_load_corrupt_file_returns_none(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research")
    cache.save(301, _bundle())
    cache._path_for(301).write_text("{not valid json", encoding="utf-8")
    assert cache.load(301, max_age=timedelta(hours=6), now=NOW) is None


def test_save_overwrites_existing(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research")
    cache.save(301, _bundle())
    cache.save(301, _bundle().model_copy(update={"queries": ["new query"]}))
    loaded = cache.load(301, max_age=timedelta(hours=6), now=NOW)
    assert loaded is not None
    assert loaded.queries == ["new query"]
