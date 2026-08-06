"""CLI test for build-market-corpus, using a fake market client (no network)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from superforecaster import cli
from superforecaster.config import MarketCorpusSettings
from superforecaster.market_corpus import MarketCorpusStore, MarketQuestion

from .test_market_corpus import manifold_detail, manifold_lite, poly_market

runner = CliRunner()


class FakeMarketClient:
    """Serves pre-parsed MarketQuestion records; no network, no filtering."""

    def __init__(self, records: list[MarketQuestion], *, pulled: int, skipped: int) -> None:
        self._records = records
        self.pulled_count = pulled
        self.skipped_count = skipped
        self.closed = False

    def iter_resolved_binaries(self, max_markets: int) -> Iterator[MarketQuestion]:
        yield from self._records[:max_markets]

    def prices(
        self, market_id: str, start: datetime, end: datetime
    ) -> list[tuple[datetime, float]]:
        return []

    def close(self) -> None:
        self.closed = True


def test_build_market_corpus_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    poly = [
        MarketQuestion.from_polymarket(poly_market("p1", volume=50_000.0)),
        MarketQuestion.from_polymarket(poly_market("p2", description="tiny", volume=50_000.0)),
    ]
    mani = [MarketQuestion.from_manifold(manifold_lite("q1"), manifold_detail("q1"))]

    def fake_build(platform: str, settings: MarketCorpusSettings) -> FakeMarketClient:
        if platform == "polymarket":
            return FakeMarketClient(poly, pulled=3, skipped=1)
        return FakeMarketClient(mani, pulled=1, skipped=0)

    monkeypatch.setattr(cli, "_build_market_client", fake_build)

    data_dir = tmp_path / "snapshots"
    result = runner.invoke(
        cli.app,
        ["build-market-corpus", "--platform", "both", "--max", "50", "--data-dir", str(data_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "build-market-corpus [polymarket]" in result.output
    assert "build-market-corpus [manifold]" in result.output
    # p2 has a short description -> dropped by min_criteria_chars (default 80).
    assert "min_criteria_chars=1" in result.output

    # Records landed in the sibling market-corpus dir, kept + dropped alike.
    corpus_dir = data_dir.parent / "market-corpus"
    store = MarketCorpusStore(corpus_dir)
    poly_ids = {r.id for r in store.iter_platform("polymarket")}
    assert poly_ids == {"p1", "p2"}
    assert {r.id for r in store.iter_platform("manifold")} == {"q1"}


def test_build_market_corpus_rejects_bad_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(cli.app, ["build-market-corpus", "--platform", "kalshi"])
    assert result.exit_code == 1
    assert "polymarket|manifold|both" in result.output
