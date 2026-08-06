"""Resolved prediction-market corpus — ground truth at scale, plus a price hook.

This is the M2 slice of ``docs/specs/market-corpus.md``: pull *resolved binary*
questions from two open, no-auth prediction markets, normalize them into one
:class:`MarketQuestion` shape, run cheap **mechanical** quality filters, and
append the result to a per-platform JSONL store. It complements Metaculus as a
backtest ground-truth source (ADR-0005 diversification).

Mirrors the house patterns deliberately:

- A structural :class:`MarketClient` protocol with one concrete client per
  platform and a single :func:`build_market_client` factory — the same shape as
  :mod:`superforecaster.search`'s ``build_search_client``.
- Injectable ``httpx`` transport + a politeness delay between requests, so tests
  answer from fixtures with no network — the seam :mod:`superforecaster.metaculus`
  uses.
- Defensive, ``None``-tolerant field extraction with a typed :class:`SkipMarket`
  raised at parse for anything that is not a clean resolved binary — the same
  contract as :class:`superforecaster.models.SkipPost`.
- An append-only JSONL store with dedupe + latest accessors, like
  :class:`superforecaster.store.SnapshotStore`.

Two live shapes, verified 2026-08-06:

**Polymarket Gamma** (``GET https://gamma-api.polymarket.com/markets?closed=true``)
returns markets whose ``outcomes`` and ``outcomePrices`` are *JSON-encoded
strings* (e.g. ``"[\\"Yes\\", \\"No\\"]"`` and ``"[\\"1\\", \\"0\\"]"``), not
arrays — they must be ``json.loads``-ed. A binary market is one whose outcomes are
``["Yes", "No"]``; there is no resolution enum, so the outcome is *inferred* from
the settled prices (``["1","0"]`` -> yes, ``["0","1"]`` -> no; anything else is
ambiguous and skipped). No auth.

**Manifold** (``GET https://api.manifold.markets/v0/search-markets`` with
``filter=resolved&contractType=BINARY``) filters to resolved binaries server-side
and returns a ``resolution`` enum (``YES``/``NO``/``MKT``/``CANCEL``). Its listing
rows carry ``resolverId`` and ``creatorId`` (so creator self-resolution is
detectable) but *not* the description text, so the criteria text is pulled from a
per-market detail fetch (``GET /v0/market/{id}``). No auth; 500 req/min/IP.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from .config import MarketCorpusSettings

# -- API endpoints (verified live 2026-08-06) --------------------------------

_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
_POLY_CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"
_POLY_MARKET_URL = "https://polymarket.com/market/{slug}"

_MANIFOLD_BASE_URL = "https://api.manifold.markets"

# A normal-looking desktop browser UA — some hosts serve empty bodies to obvious
# bots; keeps these read-only requests unremarkable (mirrors search.py).
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

Platform = Literal["polymarket", "manifold"]
Outcome = Literal["yes", "no"]

POLYMARKET: Platform = "polymarket"
MANIFOLD: Platform = "manifold"

# Prices whose |value| is within this of 1.0 / 0.0 count as a settled yes/no leg.
# Resolved markets settle their legs to exactly 1/0, but a small tolerance guards
# against float noise in the string prices.
_SETTLED_EPS = 0.02


class MarketCorpusError(Exception):
    """A corpus pull failed: a bad HTTP call or an unexpected response shape."""


class SkipMarket(Exception):
    """Raised when a raw market is not a clean resolved binary we can keep.

    Non-binary markets, unresolved markets, MKT/CANCEL resolutions, and ambiguous
    settled prices all trigger this. The builder catches it and counts the market
    as skipped rather than crashing — the same contract as
    :class:`superforecaster.models.SkipPost`.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class MarketQuestion(BaseModel):
    """One resolved binary market, normalized across platforms.

    Timestamps are tz-aware UTC (or ``None`` when the platform omitted them).
    ``criteria_text`` is the platform's resolution/description prose (Polymarket
    ``description`` / Manifold ``textDescription``) — the substrate the criteria
    filter and (later, M3) the LLM rubric read.

    The filter-provenance fields (``kept`` / ``filters_passed`` / ``filters_failed``)
    record the outcome of :func:`apply_filters` on this record, so a stored corpus
    can be re-filtered with new thresholds *without re-pulling* — every filter input
    (criteria length, volume, bettors, self-resolution) is persisted on the record.

    Attributes:
        id: The platform's market id (Polymarket Gamma ``id`` / Manifold ``id``).
        platform: ``"polymarket"`` or ``"manifold"``.
        question: The market question text.
        criteria_text: The resolution/description prose.
        outcome: The resolved outcome, normalized to ``"yes"`` / ``"no"``.
        created_time: When the market was created (UTC), or ``None``.
        close_time: The market's scheduled/actual close (UTC), or ``None``.
        resolution_time: When the market resolved (UTC), or ``None``. For Polymarket
            this is ``closedTime`` (its closest public resolution timestamp).
        volume: Traded volume (Polymarket dollars / Manifold mana), or ``None``.
        liquidity_or_bettors: Polymarket liquidity (dollars) or Manifold
            ``uniqueBettorCount`` — the platform's participation proxy, or ``None``.
        self_resolved: Manifold only — ``True`` when the market's creator resolved it
            themselves (``resolverId == creatorId``). ``None`` for Polymarket (UMA
            oracle adjudicated, so the concept does not apply).
        url: The public market page URL.
        raw: A *minimal* provenance dict (identifiers + the fields the outcome was
            inferred from); giant payloads are dropped.
        kept: Whether this record passed every applicable filter.
        filters_passed: Names of the filters it passed.
        filters_failed: Human-readable failure reasons (``"name: detail"``).
    """

    id: str
    platform: Platform
    question: str
    criteria_text: str
    outcome: Outcome
    created_time: datetime | None = None
    close_time: datetime | None = None
    resolution_time: datetime | None = None
    volume: float | None = None
    liquidity_or_bettors: float | None = None
    self_resolved: bool | None = None
    url: str
    raw: dict[str, Any] = Field(default_factory=dict)
    kept: bool = True
    filters_passed: list[str] = Field(default_factory=list)
    filters_failed: list[str] = Field(default_factory=list)

    @classmethod
    def from_polymarket(cls, market: dict[str, Any]) -> MarketQuestion:
        """Build a record from one raw Polymarket Gamma market dict.

        Raises :class:`SkipMarket` when the market is not a binary ``["Yes","No"]``
        market or its settled prices do not resolve cleanly to yes/no.
        """
        outcomes = _parse_json_list(market.get("outcomes"))
        if [str(o).strip().lower() for o in outcomes] != ["yes", "no"]:
            raise SkipMarket(f"non-binary outcomes: {outcomes!r}")

        prices = _parse_json_list(market.get("outcomePrices"))
        outcome = _infer_polymarket_outcome(prices)

        slug = str(market.get("slug") or "")
        market_id = str(market.get("id") or market.get("conditionId") or "")
        if not market_id:
            raise SkipMarket("missing market id")

        return cls(
            id=market_id,
            platform=POLYMARKET,
            question=str(market.get("question") or ""),
            criteria_text=str(market.get("description") or ""),
            outcome=outcome,
            created_time=_parse_iso8601(market.get("createdAt")),
            close_time=_parse_iso8601(market.get("endDate") or market.get("endDateIso")),
            resolution_time=_parse_iso8601(market.get("closedTime")),
            volume=_parse_float(market.get("volumeNum") or market.get("volume")),
            liquidity_or_bettors=_parse_float(
                market.get("liquidityNum") or market.get("liquidity")
            ),
            self_resolved=None,
            url=_POLY_MARKET_URL.format(slug=slug) if slug else "",
            raw={
                "slug": slug,
                "conditionId": market.get("conditionId"),
                "clobTokenIds": market.get("clobTokenIds"),
                "outcomePrices": market.get("outcomePrices"),
            },
        )

    @classmethod
    def from_manifold(
        cls, lite: dict[str, Any], detail: dict[str, Any] | None = None
    ) -> MarketQuestion:
        """Build a record from a Manifold listing row plus its detail fetch.

        ``lite`` is a ``search-markets`` row (carries ``resolution``/``resolverId``/
        ``creatorId``/metrics); ``detail`` is the ``/v0/market/{id}`` FullMarket,
        used only for ``textDescription`` (absent from the listing). Raises
        :class:`SkipMarket` for non-binary markets or non-yes/no resolutions
        (MKT/CANCEL/probabilistic).
        """
        if str(lite.get("outcomeType") or "").upper() != "BINARY":
            raise SkipMarket(f"non-binary outcomeType: {lite.get('outcomeType')!r}")
        if not lite.get("isResolved"):
            raise SkipMarket("not resolved")

        outcome = _infer_manifold_outcome(lite.get("resolution"))

        market_id = str(lite.get("id") or "")
        if not market_id:
            raise SkipMarket("missing market id")

        creator_id = lite.get("creatorId")
        resolver_id = lite.get("resolverId")
        # Self-resolution is detectable from the listing: creator == resolver.
        # ``None`` resolver (rare) reads as not-self-resolved rather than a guess.
        self_resolved = resolver_id is not None and resolver_id == creator_id

        source = detail if detail is not None else lite
        criteria_text = str(source.get("textDescription") or "")

        return cls(
            id=market_id,
            platform=MANIFOLD,
            question=str(lite.get("question") or ""),
            criteria_text=criteria_text,
            outcome=outcome,
            created_time=_parse_epoch_ms(lite.get("createdTime")),
            close_time=_parse_epoch_ms(lite.get("closeTime")),
            resolution_time=_parse_epoch_ms(lite.get("resolutionTime")),
            volume=_parse_float(lite.get("volume")),
            liquidity_or_bettors=_parse_float(lite.get("uniqueBettorCount")),
            self_resolved=self_resolved,
            url=str(lite.get("url") or ""),
            raw={
                "resolution": lite.get("resolution"),
                "creatorId": creator_id,
                "resolverId": resolver_id,
                "resolutionProbability": lite.get("resolutionProbability"),
            },
        )


# -- Field-extraction helpers (defensive, None-tolerant) ---------------------


def _parse_json_list(value: object) -> list[Any]:
    """Parse a JSON-encoded list string (Polymarket ``outcomes``/``outcomePrices``).

    Polymarket returns these as strings like ``"[\\"Yes\\", \\"No\\"]"``. Tolerates
    an already-decoded list (some proxies pre-parse) and returns ``[]`` for anything
    unparseable so one odd field never crashes a whole pull.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return parsed
    return []


def _infer_polymarket_outcome(prices: list[Any]) -> Outcome:
    """Infer yes/no from settled ``outcomePrices``; raise :class:`SkipMarket` else.

    A resolved binary settles to ``["1","0"]`` (yes) or ``["0","1"]`` (no). Anything
    else — a still-liquid ``["0.5","0.5"]``, a void ``["0","0"]``, or a malformed
    pair — is ambiguous and skipped rather than guessed.
    """
    if len(prices) != 2:
        raise SkipMarket(f"expected 2 outcome prices, got {prices!r}")
    yes = _parse_float(prices[0])
    no = _parse_float(prices[1])
    if yes is None or no is None:
        raise SkipMarket(f"unparseable outcome prices: {prices!r}")
    if abs(yes - 1.0) <= _SETTLED_EPS and abs(no - 0.0) <= _SETTLED_EPS:
        return "yes"
    if abs(yes - 0.0) <= _SETTLED_EPS and abs(no - 1.0) <= _SETTLED_EPS:
        return "no"
    raise SkipMarket(f"ambiguous settled prices: {prices!r}")


def _infer_manifold_outcome(resolution: object) -> Outcome:
    """Map a Manifold ``resolution`` to yes/no; raise :class:`SkipMarket` otherwise.

    ``MKT`` (partial/probabilistic), ``CANCEL`` (annulled), and any other value are
    excluded — only clean ``YES``/``NO`` outcomes enter the corpus.
    """
    value = str(resolution or "").strip().upper()
    if value == "YES":
        return "yes"
    if value == "NO":
        return "no"
    raise SkipMarket(f"non-yes/no resolution: {resolution!r}")


def _parse_iso8601(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp to a tz-aware UTC ``datetime``, else ``None``.

    Tolerates the several shapes Polymarket emits: a trailing ``Z`` (``createdAt``,
    ``endDate``), a space separator with a short ``+00`` offset (``closedTime``,
    e.g. ``"2020-11-02 16:31:01+00"``), and a bare date (``endDateIso``). A naive
    result is assumed UTC; everything is normalized to UTC.
    """
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    # Normalize a short "+00" / "-00" offset to the "+00:00" Python wants.
    if len(text) >= 3 and text[-3] in "+-":
        text = text + ":00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_epoch_ms(value: object) -> datetime | None:
    """Parse a Unix-milliseconds timestamp (Manifold) to tz-aware UTC, else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, int | float):
        return None
    try:
        return datetime.fromtimestamp(value / 1000.0, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _parse_float(value: object) -> float | None:
    """Best-effort conversion to ``float`` (Polymarket sends numeric strings)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return None
    return None


# -- Quality filters ---------------------------------------------------------

# The LLM rubric filter (M3) is deferred; this name is recorded on records that
# passed the (currently inert) gate, so a later re-filter can tell it was wired.
_LLM_RUBRIC_FILTER = "llm_rubric"


def apply_filters(record: MarketQuestion, settings: MarketCorpusSettings) -> tuple[bool, list[str]]:
    """Run the mechanical quality filters over one record.

    Returns ``(kept, reasons)`` where ``reasons`` is the list of human-readable
    failure strings (empty when kept). Pure and re-runnable: it reads only fields
    persisted on ``record``, so a stored corpus can be re-filtered with new
    thresholds without re-pulling. The caller is expected to write the outcome back
    onto the record (see :func:`filtered`).

    Filters (each applied only where it makes sense for the platform):

    - ``min_criteria_chars`` — both platforms.
    - ``min_volume`` — Polymarket only.
    - ``min_unique_bettors`` — Manifold only.
    - ``not_self_resolved`` — Manifold only, when
      ``settings.exclude_self_resolved``.
    - ``llm_rubric`` — DEFERRED (M3): a config-gated no-op. When
      ``settings.llm_rubric_enabled`` it records a wired-but-inert pass; it does
      not call an LLM.

    MKT/CANCEL/non-binary/ambiguous markets never reach here — they are excluded at
    parse via :class:`SkipMarket`.
    """
    reasons: list[str] = []

    n = len(record.criteria_text)
    if n < settings.min_criteria_chars:
        reasons.append(f"min_criteria_chars: {n} < {settings.min_criteria_chars}")

    if record.platform == POLYMARKET:
        vol = record.volume if record.volume is not None else 0.0
        if vol < settings.min_volume_usd:
            reasons.append(f"min_volume: {vol:.0f} < {settings.min_volume_usd:.0f}")

    if record.platform == MANIFOLD:
        bettors = record.liquidity_or_bettors if record.liquidity_or_bettors is not None else 0.0
        if bettors < settings.min_unique_bettors:
            reasons.append(f"min_unique_bettors: {bettors:.0f} < {settings.min_unique_bettors}")
        if settings.exclude_self_resolved and record.self_resolved:
            reasons.append("not_self_resolved: creator resolved their own market")

    # DEFERRED (M3): the LLM rubric is wired but inert. Gated OFF by default; even
    # when the flag is on it does not spend an LLM call yet.
    if settings.llm_rubric_enabled:
        _llm_rubric_stub(record, settings)

    return (not reasons, reasons)


def _llm_rubric_stub(record: MarketQuestion, settings: MarketCorpusSettings) -> None:
    """No-op placeholder for the M3 LLM "is this resolvable as written?" rubric.

    Wired into :func:`apply_filters` so the call site exists, but intentionally does
    nothing in M2 — the real rubric (a cheap scored LLM pass with a config threshold)
    lands in M3. Kept as a named seam so the deferral is legible in the code.
    """
    del record, settings  # deferred: no LLM call in M2.


def _all_filter_names(record: MarketQuestion, settings: MarketCorpusSettings) -> list[str]:
    """The ordered names of every filter that *applies* to a record's platform."""
    names = ["min_criteria_chars"]
    if record.platform == POLYMARKET:
        names.append("min_volume")
    if record.platform == MANIFOLD:
        names.append("min_unique_bettors")
        if settings.exclude_self_resolved:
            names.append("not_self_resolved")
    if settings.llm_rubric_enabled:
        names.append(_LLM_RUBRIC_FILTER)
    return names


def filtered(record: MarketQuestion, settings: MarketCorpusSettings) -> MarketQuestion:
    """Return a copy of ``record`` with its filter provenance filled in.

    Records the pass/fail outcome of :func:`apply_filters` onto the record (``kept``,
    ``filters_passed`` names, ``filters_failed`` reason strings) so every stored
    market — kept or dropped — carries why, and the whole corpus is re-filterable
    without a re-pull.
    """
    kept, reasons = apply_filters(record, settings)
    # A failure reason is "name: detail"; map back to the filter names that failed.
    failed_names = {reason.split(":", 1)[0] for reason in reasons}
    passed = [name for name in _all_filter_names(record, settings) if name not in failed_names]
    return record.model_copy(
        update={"kept": kept, "filters_passed": passed, "filters_failed": reasons}
    )


# -- Clients -----------------------------------------------------------------


class MarketClient(Protocol):
    """Structural type for a per-platform resolved-market client."""

    def iter_resolved_binaries(self, max_markets: int) -> Iterator[MarketQuestion]:
        """Yield up to ``max_markets`` parsed resolved-binary records (pre-filter)."""
        ...

    def prices(
        self, market_id: str, start: datetime, end: datetime
    ) -> list[tuple[datetime, float]]:
        """Return the ``(time, yes-probability)`` series in ``[start, end]``.

        The candidate-D hook: the ``start``/``end`` bounds enforce the forecast-time
        cutoff (a price series trivially leaks the outcome near resolution).
        """
        ...

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        ...


class _ThrottledClient:
    """Shared base: an httpx client with a politeness delay + one 429 retry.

    Injectable ``transport``/``monotonic``/``sleep`` are the same test seams the
    rest of the engine uses; ``pulled_count`` / ``skipped_count`` are running
    tallies the builder reads after iteration for its stats summary.
    """

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None,
        timeout_s: float,
        min_request_interval_s: float,
        monotonic: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self._min_interval = min_request_interval_s
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_at: float | None = None
        self.pulled_count = 0
        self.skipped_count = 0
        self._client = httpx.Client(
            base_url=base_url,
            transport=transport,
            headers={"User-Agent": _BROWSER_UA, "Accept": "application/json"},
            timeout=timeout_s,
        )

    def __enter__(self) -> _ThrottledClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def _throttled_get(self, url: str, params: dict[str, Any]) -> httpx.Response:
        """GET with a request-interval floor and one retry on 429."""
        now = self._monotonic()
        if self._last_request_at is not None:
            wait = self._min_interval - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        response = self._client.get(url, params=params)
        self._last_request_at = self._monotonic()
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = min(float(retry_after), 30.0) if retry_after else 5.0
            except ValueError:
                delay = 5.0
            self._sleep(max(delay, self._min_interval))
            response = self._client.get(url, params=params)
            self._last_request_at = self._monotonic()
        return response


class PolymarketClient(_ThrottledClient):
    """Resolved-binary client for the Polymarket Gamma + CLOB APIs (no auth)."""

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 20.0,
        page_size: int = 100,
        min_request_interval_s: float = 0.5,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            _GAMMA_BASE_URL,
            transport=transport,
            timeout_s=timeout_s,
            min_request_interval_s=min_request_interval_s,
            monotonic=monotonic,
            sleep=sleep,
        )
        self._page_size = max(1, min(page_size, 100))

    def iter_resolved_binaries(self, max_markets: int) -> Iterator[MarketQuestion]:
        """Page ``/markets?closed=true`` (high-volume first), parse, yield binaries.

        Non-binary / ambiguous markets raise :class:`SkipMarket` at parse and are
        counted (``skipped_count``) rather than yielded. Pagination stops at
        ``max_markets`` kept-or-skipped rows, an empty page, or a short final page.
        """
        offset = 0
        yielded = 0
        while yielded < max_markets:
            limit = min(self._page_size, max_markets - yielded)
            params: dict[str, Any] = {
                "closed": "true",
                "limit": limit,
                "offset": offset,
                "order": "volumeNum",
                "ascending": "false",
            }
            try:
                response = self._throttled_get("/markets", params)
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                raise MarketCorpusError(f"Polymarket request failed: {exc}") from exc
            markets = payload if isinstance(payload, list) else []
            if not markets:
                return
            for market in markets:
                if not isinstance(market, dict):
                    continue
                self.pulled_count += 1
                try:
                    record = MarketQuestion.from_polymarket(market)
                except SkipMarket:
                    self.skipped_count += 1
                    continue
                yield record
                yielded += 1
                if yielded >= max_markets:
                    return
            if len(markets) < limit:
                return
            offset += len(markets)

    def prices(
        self, market_id: str, start: datetime, end: datetime
    ) -> list[tuple[datetime, float]]:
        """Return the YES-token price series in ``[start, end]`` (candidate-D hook).

        ``market_id`` here is the **CLOB token id** of the YES leg (from a record's
        ``raw["clobTokenIds"]`` — a JSON-encoded ``[yesTokenId, noTokenId]`` pair),
        not the Gamma market id, because ``clob.polymarket.com/prices-history`` keys
        the series by token. The bounds are sent as ``startTs``/``endTs`` so the
        cutoff is enforced server-side.
        """
        params: dict[str, Any] = {
            "market": market_id,
            "startTs": int(start.timestamp()),
            "endTs": int(end.timestamp()),
            "fidelity": 60,
        }
        try:
            response = self._throttled_get(_POLY_CLOB_PRICES_URL, params)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise MarketCorpusError(f"Polymarket prices request failed: {exc}") from exc
        history = payload.get("history") if isinstance(payload, dict) else None
        if not isinstance(history, list):
            return []
        series: list[tuple[datetime, float]] = []
        for point in history:
            if not isinstance(point, dict):
                continue
            ts = _parse_float(point.get("t"))
            price = _parse_float(point.get("p"))
            if ts is None or price is None:
                continue
            series.append((datetime.fromtimestamp(ts, tz=UTC), price))
        return series


class ManifoldClient(_ThrottledClient):
    """Resolved-binary client for the Manifold v0 API (no auth, 500 req/min/IP).

    Lists via ``search-markets`` (``filter=resolved&contractType=BINARY``) — which
    filters to resolved binaries server-side — then fetches each candidate's
    FullMarket detail (``/v0/market/{id}``) for the ``textDescription`` the listing
    omits. The politeness delay spaces every request (listing pages and detail
    fetches alike).
    """

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 20.0,
        page_size: int = 100,
        min_request_interval_s: float = 0.5,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            _MANIFOLD_BASE_URL,
            transport=transport,
            timeout_s=timeout_s,
            min_request_interval_s=min_request_interval_s,
            monotonic=monotonic,
            sleep=sleep,
        )
        self._page_size = max(1, min(page_size, 1000))

    def iter_resolved_binaries(self, max_markets: int) -> Iterator[MarketQuestion]:
        """Page resolved binaries, fetch each one's detail, parse, yield.

        A listing row that fails the cheap pre-check (non-binary, unresolved, or a
        non-yes/no resolution) is counted as skipped without a detail fetch. Only a
        pre-check pass triggers the ``/v0/market/{id}`` call. Pagination stops at
        ``max_markets`` yielded, an empty page, or a short final page.
        """
        offset = 0
        yielded = 0
        while yielded < max_markets:
            limit = min(self._page_size, max_markets - yielded)
            params: dict[str, Any] = {
                "term": "",
                "filter": "resolved",
                "contractType": "BINARY",
                "sort": "most-popular",
                "limit": limit,
                "offset": offset,
            }
            try:
                response = self._throttled_get("/v0/search-markets", params)
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                raise MarketCorpusError(f"Manifold request failed: {exc}") from exc
            rows = payload if isinstance(payload, list) else []
            if not rows:
                return
            for lite in rows:
                if not isinstance(lite, dict):
                    continue
                self.pulled_count += 1
                # Cheap pre-check: only fetch detail for markets that will parse.
                resolution = str(lite.get("resolution") or "").strip().upper()
                is_binary = str(lite.get("outcomeType") or "").upper() == "BINARY"
                if not (lite.get("isResolved") and is_binary and resolution in ("YES", "NO")):
                    self.skipped_count += 1
                    continue
                detail = self._fetch_detail(str(lite.get("id") or ""))
                try:
                    record = MarketQuestion.from_manifold(lite, detail)
                except SkipMarket:
                    self.skipped_count += 1
                    continue
                yield record
                yielded += 1
                if yielded >= max_markets:
                    return
            if len(rows) < limit:
                return
            offset += len(rows)

    def _fetch_detail(self, market_id: str) -> dict[str, Any] | None:
        """Fetch one market's FullMarket detail, or ``None`` on any failure.

        The detail is used only for ``textDescription``; a failed fetch degrades to
        ``None`` (the record's criteria text falls back to the listing, i.e. empty)
        rather than sinking the whole pull.
        """
        if not market_id:
            return None
        try:
            response = self._throttled_get(f"/v0/market/{market_id}", {})
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError:
            return None
        return payload if isinstance(payload, dict) else None

    def prices(
        self, market_id: str, start: datetime, end: datetime
    ) -> list[tuple[datetime, float]]:
        """Reconstruct the yes-probability series from ``/v0/bets`` (candidate-D hook).

        Manifold has no price-history endpoint, so the series is rebuilt from the
        market's bets: each bet's ``probAfter`` is the market probability just after
        it. Paginated backwards with ``before=<betId>``; the result is sorted
        ascending and clipped to ``[start, end]`` (the forecast-time cutoff).
        """
        collected: list[tuple[datetime, float]] = []
        before: str | None = None
        while True:
            params: dict[str, Any] = {"contractId": market_id, "limit": 1000}
            if before is not None:
                params["before"] = before
            try:
                response = self._throttled_get("/v0/bets", params)
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                raise MarketCorpusError(f"Manifold bets request failed: {exc}") from exc
            bets = payload if isinstance(payload, list) else []
            if not bets:
                break
            for bet in bets:
                if not isinstance(bet, dict):
                    continue
                when = _parse_epoch_ms(bet.get("createdTime"))
                prob = _parse_float(bet.get("probAfter"))
                if when is None or prob is None:
                    continue
                if start <= when <= end:
                    collected.append((when, prob))
            last = bets[-1]
            before = str(last.get("id") or "") if isinstance(last, dict) else ""
            if not before or len(bets) < 1000:
                break
        collected.sort(key=lambda pair: pair[0])
        return collected


def build_market_client(
    platform: str,
    *,
    settings: MarketCorpusSettings | None = None,
    transport: httpx.BaseTransport | None = None,
) -> MarketClient:
    """Return the market client for a platform name (the single dispatch point).

    Mirrors :func:`superforecaster.search.build_search_client`. ``settings`` supplies
    the page size, politeness delay, and timeout; ``transport`` is the test seam.

    Raises:
        MarketCorpusError: The platform name is unknown.
    """
    cfg = settings if settings is not None else MarketCorpusSettings()
    if platform == POLYMARKET:
        return PolymarketClient(
            transport=transport,
            timeout_s=cfg.timeout_seconds,
            page_size=cfg.page_size,
            min_request_interval_s=cfg.page_delay_seconds,
        )
    if platform == MANIFOLD:
        return ManifoldClient(
            transport=transport,
            timeout_s=cfg.timeout_seconds,
            page_size=cfg.page_size,
            min_request_interval_s=cfg.page_delay_seconds,
        )
    raise MarketCorpusError(
        f"Unknown market platform {platform!r}; supported: {POLYMARKET}, {MANIFOLD}."
    )


# -- Store -------------------------------------------------------------------


class MarketCorpusStore:
    """Append-only JSON Lines store of :class:`MarketQuestion` records.

    One file per platform (``<dir>/<platform>.jsonl``), appended never rewritten —
    the same crash-safe shape as :class:`superforecaster.store.SnapshotStore`.
    Records are deduped by ``(platform, id)`` at write time: an id already on disk
    is not appended again.

    Args:
        dir: Directory holding the per-platform files. Created on first write.
    """

    def __init__(self, dir: Path) -> None:
        self._dir = Path(dir)

    @property
    def dir(self) -> Path:
        """The directory this store reads from and writes to."""
        return self._dir

    def path_for(self, platform: str) -> Path:
        """Return the JSONL file backing one platform."""
        return self._dir / f"{platform}.jsonl"

    def existing_ids(self, platform: str) -> set[str]:
        """Return the set of market ids already stored for a platform."""
        return {record.id for record in self.iter_platform(platform)}

    def append(self, records: list[MarketQuestion]) -> int:
        """Append records to their platform files, skipping ids already stored.

        Returns the number of new records actually written (duplicates by
        ``(platform, id)`` — on disk or within this batch — are skipped).
        """
        if not records:
            return 0
        # Group by platform so each file is opened once, and dedupe against both
        # what is already on disk and earlier records in this same batch.
        by_platform: dict[str, list[MarketQuestion]] = {}
        for record in records:
            by_platform.setdefault(record.platform, []).append(record)

        written = 0
        self._dir.mkdir(parents=True, exist_ok=True)
        for platform, group in by_platform.items():
            seen = self.existing_ids(platform)
            new_lines: list[str] = []
            for record in group:
                if record.id in seen:
                    continue
                seen.add(record.id)
                new_lines.append(record.model_dump_json())
            if not new_lines:
                continue
            with self.path_for(platform).open("a", encoding="utf-8") as fh:
                for line in new_lines:
                    fh.write(line)
                    fh.write("\n")
            written += len(new_lines)
        return written

    def iter_platform(self, platform: str) -> Iterator[MarketQuestion]:
        """Yield every stored record for one platform, in append order."""
        path = self.path_for(platform)
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                yield MarketQuestion.model_validate_json(line)

    def iter_all(self) -> Iterator[MarketQuestion]:
        """Yield every stored record across all platform files."""
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    yield MarketQuestion.model_validate_json(line)

    def latest_by_id(self, platform: str) -> dict[str, MarketQuestion]:
        """Return the last-written record per market id for one platform."""
        latest: dict[str, MarketQuestion] = {}
        for record in self.iter_platform(platform):
            latest[record.id] = record
        return latest
