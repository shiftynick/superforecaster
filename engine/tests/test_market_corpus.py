"""Tests for the resolved-market corpus, using httpx.MockTransport (no network).

Response JSON mirrors the live shapes verified 2026-08-06:

- Polymarket Gamma ``/markets`` — ``outcomes``/``outcomePrices`` are JSON-encoded
  STRINGS; the outcome is inferred from the settled prices.
- Manifold ``/v0/search-markets`` (resolved binaries) + ``/v0/market/{id}`` detail
  for ``textDescription``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from superforecaster.config import MarketCorpusSettings
from superforecaster.market_corpus import (
    ManifoldClient,
    MarketCorpusError,
    MarketCorpusStore,
    MarketQuestion,
    PolymarketClient,
    SkipMarket,
    apply_filters,
    build_market_client,
    filtered,
)


def _noop_sleep(_seconds: float) -> None:
    """Stand-in for time.sleep so tests never actually wait."""


# A comfortably-long (>= 80 char) criteria default so the min_criteria_chars
# filter passes unless a test deliberately overrides it with a short string.
_LONG_DESC = "Resolves YES if the launch occurs before the stated deadline, per the official site."


# --- Realistic fixtures -----------------------------------------------------


def poly_market(
    market_id: str = "0x01",
    *,
    outcomes: str = '["Yes", "No"]',
    outcome_prices: str = '["1", "0"]',
    description: str = _LONG_DESC,
    volume: float = 50_000.0,
) -> dict[str, Any]:
    """A resolved binary Polymarket market (Gamma shape: prices are string-encoded)."""
    return {
        "id": market_id,
        "conditionId": f"cond-{market_id}",
        "slug": f"market-{market_id}",
        "question": "Will the launch happen before the deadline?",
        "description": description,
        "outcomes": outcomes,
        "outcomePrices": outcome_prices,
        "clobTokenIds": '["yes-token-1", "no-token-1"]',
        "volume": str(volume),
        "volumeNum": volume,
        "liquidity": "0",
        "liquidityNum": 0.0,
        "createdAt": "2024-01-02T16:10:01.467Z",
        "endDate": "2024-11-04T00:00:00Z",
        "endDateIso": "2024-11-04",
        "closedTime": "2024-11-06 15:17:41+00",
        "closed": True,
        "active": True,
    }


def manifold_lite(
    market_id: str = "abc123",
    *,
    resolution: str = "YES",
    outcome_type: str = "BINARY",
    creator_id: str = "creator-1",
    resolver_id: str = "resolver-1",
    bettors: int = 40,
) -> dict[str, Any]:
    """A resolved-binary Manifold search-markets row (no description field)."""
    return {
        "id": market_id,
        "creatorId": creator_id,
        "resolverId": resolver_id,
        "question": "Will the milestone be reached this year?",
        "outcomeType": outcome_type,
        "mechanism": "cpmm-1",
        "isResolved": True,
        "resolution": resolution,
        "resolutionTime": 1735780870155,
        "resolutionProbability": 0.99,
        "createdTime": 1690318393221,
        "closeTime": 1735737881975,
        "volume": 55_000.0,
        "uniqueBettorCount": bettors,
        "url": f"https://manifold.markets/creator/{market_id}",
    }


def manifold_detail(market_id: str = "abc123") -> dict[str, Any]:
    """A Manifold FullMarket detail carrying the textDescription the listing omits."""
    return {
        "id": market_id,
        "textDescription": "Resolves YES if the milestone is publicly confirmed before year end.",
        "outcomeType": "BINARY",
        "isResolved": True,
        "resolution": "YES",
    }


# --- Parse: Polymarket ------------------------------------------------------


def test_polymarket_parses_yes_from_settled_prices() -> None:
    record = MarketQuestion.from_polymarket(poly_market(outcome_prices='["1", "0"]'))
    assert record.outcome == "yes"
    assert record.platform == "polymarket"
    assert record.id == "0x01"
    assert record.volume == 50_000.0
    assert record.url == "https://polymarket.com/market/market-0x01"
    # closedTime "... +00" is normalized to tz-aware UTC.
    assert record.resolution_time == datetime(2024, 11, 6, 15, 17, 41, tzinfo=UTC)
    assert record.created_time == datetime(2024, 1, 2, 16, 10, 1, 467000, tzinfo=UTC)


def test_polymarket_parses_no_from_reversed_prices() -> None:
    record = MarketQuestion.from_polymarket(poly_market(outcome_prices='["0", "1"]'))
    assert record.outcome == "no"


def test_polymarket_non_binary_outcomes_skipped() -> None:
    with pytest.raises(SkipMarket):
        MarketQuestion.from_polymarket(poly_market(outcomes='["A", "B", "C"]'))


def test_polymarket_ambiguous_prices_skipped() -> None:
    # A void / still-liquid market never settled to 1/0.
    with pytest.raises(SkipMarket):
        MarketQuestion.from_polymarket(poly_market(outcome_prices='["0.5", "0.5"]'))
    with pytest.raises(SkipMarket):
        MarketQuestion.from_polymarket(poly_market(outcome_prices='["0", "0"]'))


# --- Parse: Manifold --------------------------------------------------------


def test_manifold_parses_yes_with_detail_criteria() -> None:
    record = MarketQuestion.from_manifold(manifold_lite(resolution="YES"), manifold_detail())
    assert record.outcome == "yes"
    assert record.platform == "manifold"
    assert record.criteria_text.startswith("Resolves YES if the milestone")
    assert record.liquidity_or_bettors == 40
    assert record.self_resolved is False
    assert record.resolution_time == datetime.fromtimestamp(1735780870155 / 1000, tz=UTC)


def test_manifold_no_resolution() -> None:
    record = MarketQuestion.from_manifold(manifold_lite(resolution="NO"), manifold_detail())
    assert record.outcome == "no"


def test_manifold_mkt_and_cancel_skipped() -> None:
    for res in ("MKT", "CANCEL", "PROB"):
        with pytest.raises(SkipMarket):
            MarketQuestion.from_manifold(manifold_lite(resolution=res), manifold_detail())


def test_manifold_non_binary_skipped() -> None:
    with pytest.raises(SkipMarket):
        MarketQuestion.from_manifold(manifold_lite(outcome_type="MULTIPLE_CHOICE"), None)


def test_manifold_self_resolution_detected() -> None:
    lite = manifold_lite(creator_id="same", resolver_id="same")
    record = MarketQuestion.from_manifold(lite, manifold_detail())
    assert record.self_resolved is True


# --- Filters ----------------------------------------------------------------


def test_filter_min_criteria_chars_fails_and_records_reason() -> None:
    settings = MarketCorpusSettings(min_criteria_chars=80, min_volume_usd=0.0)
    record = MarketQuestion.from_polymarket(poly_market(description="too short"))
    kept, reasons = apply_filters(record, settings)
    assert kept is False
    assert any(r.startswith("min_criteria_chars") for r in reasons)


def test_filter_min_volume_polymarket() -> None:
    settings = MarketCorpusSettings(min_criteria_chars=0, min_volume_usd=10_000.0)
    low = MarketQuestion.from_polymarket(poly_market(volume=500.0))
    high = MarketQuestion.from_polymarket(poly_market(volume=50_000.0))
    assert apply_filters(low, settings)[0] is False
    assert apply_filters(high, settings)[0] is True


def test_filter_min_unique_bettors_manifold() -> None:
    settings = MarketCorpusSettings(min_criteria_chars=0, min_unique_bettors=10)
    thin = MarketQuestion.from_manifold(manifold_lite(bettors=3), manifold_detail())
    thick = MarketQuestion.from_manifold(manifold_lite(bettors=40), manifold_detail())
    assert apply_filters(thin, settings)[0] is False
    assert apply_filters(thick, settings)[0] is True


def test_filter_excludes_manifold_self_resolved() -> None:
    settings = MarketCorpusSettings(min_criteria_chars=0, exclude_self_resolved=True)
    selfres = MarketQuestion.from_manifold(
        manifold_lite(creator_id="x", resolver_id="x"), manifold_detail()
    )
    kept, reasons = apply_filters(selfres, settings)
    assert kept is False
    assert any(r.startswith("not_self_resolved") for r in reasons)


def test_filter_self_resolved_allowed_when_flag_off() -> None:
    settings = MarketCorpusSettings(min_criteria_chars=0, exclude_self_resolved=False)
    selfres = MarketQuestion.from_manifold(
        manifold_lite(creator_id="x", resolver_id="x"), manifold_detail()
    )
    assert apply_filters(selfres, settings)[0] is True


def test_filtered_records_pass_fail_provenance() -> None:
    settings = MarketCorpusSettings(min_criteria_chars=80, min_volume_usd=10_000.0)
    dropped = filtered(MarketQuestion.from_polymarket(poly_market(volume=1.0)), settings)
    assert dropped.kept is False
    assert "min_volume" in {r.split(":", 1)[0] for r in dropped.filters_failed}
    # min_criteria_chars passed (the default description is long), so it is recorded.
    assert "min_criteria_chars" in dropped.filters_passed

    kept = filtered(MarketQuestion.from_polymarket(poly_market(volume=50_000.0)), settings)
    assert kept.kept is True
    assert kept.filters_failed == []
    assert set(kept.filters_passed) == {"min_criteria_chars", "min_volume"}


def test_llm_rubric_stub_is_inert_when_enabled() -> None:
    # DEFERRED (M3): enabling the flag must not crash or change the mechanical
    # verdict — it is a wired-but-inert no-op for now.
    settings = MarketCorpusSettings(
        min_criteria_chars=0, min_volume_usd=0.0, llm_rubric_enabled=True
    )
    record = MarketQuestion.from_polymarket(poly_market())
    kept, reasons = apply_filters(record, settings)
    assert kept is True
    assert reasons == []
    assert "llm_rubric" in filtered(record, settings).filters_passed


# --- Client pagination (Polymarket) -----------------------------------------


def test_polymarket_pagination_and_skip_counts() -> None:
    # Page 1: one good binary + one non-binary (skipped). Page 2: one good binary.
    page1 = [poly_market("m1"), poly_market("m2", outcomes='["A", "B", "C"]')]
    page2 = [poly_market("m3")]
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        offset = int(params.get("offset", "0"))
        return httpx.Response(200, json=page1 if offset == 0 else page2)

    client = PolymarketClient(
        transport=httpx.MockTransport(handler),
        page_size=2,
        min_request_interval_s=0.0,
        sleep=_noop_sleep,
    )
    records = list(client.iter_resolved_binaries(max_markets=3))
    client.close()

    assert [r.id for r in records] == ["m1", "m3"]
    assert client.pulled_count == 3  # three raw markets seen
    assert client.skipped_count == 1  # the non-binary one
    assert calls[0]["closed"] == "true"
    assert calls[0]["order"] == "volumeNum"


def test_polymarket_stops_at_max() -> None:
    page = [poly_market(f"m{i}") for i in range(10)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=page)

    client = PolymarketClient(
        transport=httpx.MockTransport(handler),
        page_size=10,
        min_request_interval_s=0.0,
        sleep=_noop_sleep,
    )
    records = list(client.iter_resolved_binaries(max_markets=3))
    client.close()
    assert len(records) == 3


def test_polymarket_prices_history() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "prices-history" in str(request.url)
        assert request.url.params.get("market") == "yes-token-1"
        return httpx.Response(
            200,
            json={"history": [{"t": 1730000000, "p": 0.4}, {"t": 1730003600, "p": 0.7}]},
        )

    client = PolymarketClient(transport=httpx.MockTransport(handler), min_request_interval_s=0.0)
    series = client.prices(
        "yes-token-1",
        datetime(2024, 10, 1, tzinfo=UTC),
        datetime(2024, 12, 1, tzinfo=UTC),
    )
    client.close()
    assert len(series) == 2
    assert series[0][1] == 0.4
    assert series[0][0].tzinfo is not None


# --- Client (Manifold): listing + detail fetch ------------------------------


def test_manifold_iter_fetches_detail_and_skips_precheck_failures() -> None:
    # Listing has one good binary, one MKT (pre-check skip, NO detail fetch), one
    # non-binary (pre-check skip).
    listing = [
        manifold_lite("g1", resolution="YES"),
        manifold_lite("m2", resolution="MKT"),
        manifold_lite("m3", outcome_type="MULTIPLE_CHOICE"),
    ]
    detail_fetches: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/v0/search-markets"):
            offset = int(request.url.params.get("offset", "0"))
            return httpx.Response(200, json=listing if offset == 0 else [])
        if "/v0/market/" in path:
            market_id = path.rsplit("/", 1)[-1]
            detail_fetches.append(market_id)
            return httpx.Response(200, json=manifold_detail(market_id))
        return httpx.Response(404, json={})

    client = ManifoldClient(
        transport=httpx.MockTransport(handler),
        page_size=100,
        min_request_interval_s=0.0,
        sleep=_noop_sleep,
    )
    records = list(client.iter_resolved_binaries(max_markets=50))
    client.close()

    assert [r.id for r in records] == ["g1"]
    assert detail_fetches == ["g1"]  # only the pre-check pass triggered a detail call
    assert client.skipped_count == 2
    assert records[0].criteria_text.startswith("Resolves YES")


def test_manifold_prices_reconstructed_from_bets() -> None:
    bets = [
        {"id": "b1", "createdTime": 1730000000000, "probAfter": 0.55},
        {"id": "b2", "createdTime": 1730003600000, "probAfter": 0.62},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v0/bets")
        # Single page: return the bets once, then empty on the paginated follow-up.
        if request.url.params.get("before"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=bets)

    client = ManifoldClient(transport=httpx.MockTransport(handler), min_request_interval_s=0.0)
    series = client.prices(
        "abc123",
        datetime(2024, 10, 1, tzinfo=UTC),
        datetime(2024, 12, 1, tzinfo=UTC),
    )
    client.close()
    assert [p for _, p in series] == [0.55, 0.62]  # sorted ascending by time


# --- Factory ----------------------------------------------------------------


def test_build_market_client_dispatch() -> None:
    assert isinstance(build_market_client("polymarket"), PolymarketClient)
    assert isinstance(build_market_client("manifold"), ManifoldClient)


def test_build_market_client_unknown() -> None:
    with pytest.raises(MarketCorpusError):
        build_market_client("kalshi")


# --- Store ------------------------------------------------------------------


def test_store_roundtrip_and_dedupe(tmp_path: Path) -> None:
    store = MarketCorpusStore(tmp_path / "market-corpus")
    r1 = MarketQuestion.from_polymarket(poly_market("m1"))
    r2 = MarketQuestion.from_polymarket(poly_market("m2"))

    assert store.append([r1, r2]) == 2
    # Re-appending the same ids writes nothing (dedupe by (platform, id)).
    assert store.append([r1]) == 0
    # A duplicate WITHIN one batch is also collapsed.
    assert store.append([MarketQuestion.from_polymarket(poly_market("m3")), r1]) == 1

    ids = {r.id for r in store.iter_platform("polymarket")}
    assert ids == {"m1", "m2", "m3"}


def test_store_separates_platforms(tmp_path: Path) -> None:
    store = MarketCorpusStore(tmp_path / "mc")
    poly = MarketQuestion.from_polymarket(poly_market("p1"))
    mani = MarketQuestion.from_manifold(manifold_lite("q1"), manifold_detail("q1"))
    store.append([poly, mani])

    assert store.path_for("polymarket").exists()
    assert store.path_for("manifold").exists()
    assert {r.id for r in store.iter_all()} == {"p1", "q1"}
    assert list(store.latest_by_id("manifold")) == ["q1"]
