"""Provider-neutral web/news search — the retrieval stack's search layer.

Mirrors :mod:`superforecaster.llm`'s provider pattern: a structural
:class:`SearchClient` protocol, one concrete client per provider, and a single
:func:`build_search_client` factory that maps a provider name to a client. The
forecaster never names a provider or an endpoint itself; it asks the factory.

Three providers today (ADR-0004):

- :class:`AskNewsClient` — the *news* role. AskNews returns AI summaries of
  recent articles plus a per-call credit cost, which we meter against a
  persisted monthly cap so we never blow past the free tier unknowingly.
- :class:`TavilyClient` — the *general web* role. Tavily returns page excerpts;
  a publish date is only surfaced when the request sets ``topic="news"``.
- :class:`PerplexitySearchClient` — an alternative *general web* role, added for
  experiment 006. Perplexity's standalone **Search API** (``POST /search`` — raw
  ranked results, NOT the Sonar chat/answers API) returns page excerpts with an
  ISO-8601 ``date`` field on each hit.

Both normalize into one :class:`SearchResult` shape (publish dates parsed from
each provider's differing format into a common ``datetime``), and both take an
injectable ``httpx`` transport so tests answer requests from fixtures with no
real network — the same seam :mod:`superforecaster.metaculus` uses.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from .config import RetrievalSettings

# Env var names holding each provider's key (loaded from engine/.env by the CLI).
ASK_NEWS_API_KEY_ENV = "ASK_NEWS_API_KEY"
TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
PERPLEXITY_API_KEY_ENV = "PERPLEXITY_API_KEY"

# Provider role names (as they appear in [retrieval] config).
ASKNEWS_PROVIDER = "asknews"
TAVILY_PROVIDER = "tavily"
PERPLEXITY_PROVIDER = "perplexity"

_ASKNEWS_BASE_URL = "https://api.asknews.app"
_TAVILY_BASE_URL = "https://api.tavily.com"
_PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

# A normal-looking desktop browser UA. Some hosts serve empty bodies to obvious
# bots; this keeps search-provider requests unremarkable.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class SearchError(Exception):
    """A search call failed: bad config, missing key, or a provider error."""


class SearchQuotaError(SearchError):
    """The configured monthly credit cap for a metered provider is exhausted.

    Raised *before* a call is made, so no further quota is drawn once the cap is
    reached. Distinct from :class:`SearchError` so callers can degrade politely
    (skip the news provider) rather than treat it as a hard failure.
    """


class SearchResult(BaseModel):
    """One normalized search hit, provider-agnostic.

    Attributes:
        url: The article/page URL.
        title: The result title.
        snippet: A short excerpt or AI summary of the content.
        published_at: Publish time as a ``datetime`` (tz-aware when the provider
            gave one), or ``None`` when absent or unparseable.
        source_domain: Bare domain (e.g. ``"example.com"``), for source-quality
            weighting downstream.
        provider: Which client produced this result (``"asknews"``/``"tavily"``).
        raw: The provider's untouched result object, kept verbatim for the trace.
    """

    url: str
    title: str
    snippet: str
    published_at: datetime | None
    source_domain: str
    provider: str
    raw: dict[str, Any]


class SearchClient(Protocol):
    """Structural type for any search client the research pipeline can drive."""

    def search(
        self,
        query: str,
        *,
        max_results: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[SearchResult]: ...


def _parse_iso8601(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp (AskNews ``pub_date``), tolerating ``Z``.

    Anything missing, non-string, or unparseable becomes ``None`` so one odd
    field never sinks a whole result.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_rfc1123(value: object) -> datetime | None:
    """Parse an RFC 1123 date (Tavily ``published_date``), else ``None``.

    Example input: ``"Tue, 16 Jun 2026 09:12:25 GMT"``. Tavily only includes
    this field when the request sets ``topic="news"``; every other case (absent,
    non-string, malformed) returns ``None``.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


def _domain_of(url: str) -> str:
    """Return the bare host of a URL (Tavily gives no domain field of its own)."""
    try:
        return urlparse(url).netloc or ""
    except ValueError:
        return ""


# Basename of the AskNews credit-counter file inside the data directory.
ASKNEWS_QUOTA_FILENAME = "asknews_quota.json"


def _read_quota_file(quota_path: Path) -> tuple[str, int]:
    """Return ``(month, credits_used)`` from a quota file, else ``("", 0)``.

    Tolerant by design: a missing file, unreadable file, or malformed contents
    all read as "nothing spent" rather than raising — the counter is a budget
    guard, not a source of truth we would ever want to crash on.
    """
    if not quota_path.exists():
        return "", 0
    try:
        data = json.loads(quota_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "", 0
    if not isinstance(data, dict):
        return "", 0
    month = data.get("month")
    used = data.get("credits_used")
    month_str = month if isinstance(month, str) else ""
    used_int = used if isinstance(used, int) and not isinstance(used, bool) else 0
    return month_str, used_int


def read_asknews_credits_used(
    data_dir: Path,
    *,
    today: Callable[[], date] = date.today,
) -> int:
    """Return AskNews credits spent *this month*, read straight from the counter.

    A read-only view of the same persisted counter :class:`AskNewsClient` writes
    (``<data_dir>/asknews_quota.json``), so the ``stats`` command can report usage
    without constructing a client or needing the API key. Returns 0 when the
    stored month is not the current one (a rollover) or the file is absent.
    """
    month, used = _read_quota_file(data_dir / ASKNEWS_QUOTA_FILENAME)
    return used if month == today().strftime("%Y-%m") else 0


class AskNewsClient:
    """Search recent news via AskNews, metering credits against a monthly cap.

    Talks to ``GET /v1/news/search`` and normalizes ``as_dicts`` articles into
    :class:`SearchResult`\\ s. Each response reports ``usage.credits``; we persist
    a running monthly total in a small JSON file and refuse (raise
    :class:`SearchQuotaError`) once the configured cap would be exceeded. The
    counter resets automatically when the calendar month rolls over.
    """

    def __init__(
        self,
        api_key: str,
        *,
        quota_path: Path,
        monthly_credit_cap: int,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 20.0,
        method: str = "nl",
        today: Callable[[], date] = date.today,
        min_request_interval_s: float = 2.1,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Configure the client.

        Args:
            api_key: The AskNews key. Sent as ``Authorization: Bearer <key>``.
            quota_path: JSON file where the monthly credit counter is persisted.
            monthly_credit_cap: Hard cap; calls are refused once reached.
            transport: Optional ``httpx`` transport (tests inject a
                ``MockTransport``).
            timeout_s: Per-request HTTP timeout.
            method: AskNews search method; ``"nl"`` (natural-language) per recon.
            today: Returns the current date; injectable so tests can drive
                month-rollover behaviour. Defaults to ``date.today``.
            min_request_interval_s: Floor between consecutive requests from this
                client. The AskNews free tier allows 1 request per 2 seconds;
                bursting past that returns 429 (seen live 2026-07-10).
            monotonic: Clock for the throttle; injectable for tests.
            sleep: Sleeper for throttle/backoff waits; injectable for tests.
        """
        self._api_key = api_key
        self._quota_path = quota_path
        self._cap = monthly_credit_cap
        self._method = method
        self._today = today
        self._min_interval = min_request_interval_s
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_at: float | None = None
        # NOTE: AskNews' OpenAPI spec documents a custom ``APIKey`` header, but
        # that returns 401 in practice. A plain Bearer token is what actually
        # works — verified live against the real key on 2026-07-09 (see the
        # retrieval recon). Do not "fix" this to match the published spec.
        self._client = httpx.Client(
            base_url=_ASKNEWS_BASE_URL,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": _BROWSER_UA,
            },
            timeout=timeout_s,
        )

    def __enter__(self) -> AskNewsClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    # -- quota bookkeeping --------------------------------------------------

    def _current_month(self) -> str:
        return self._today().strftime("%Y-%m")

    def _read_quota(self) -> tuple[str, int]:
        """Return ``(month, credits_used)`` from the quota file, else ``("", 0)``."""
        return _read_quota_file(self._quota_path)

    def _write_quota(self, month: str, credits_used: int) -> None:
        self._quota_path.parent.mkdir(parents=True, exist_ok=True)
        self._quota_path.write_text(
            json.dumps({"month": month, "credits_used": credits_used}),
            encoding="utf-8",
        )

    def credits_used_this_month(self) -> int:
        """Return credits already spent this month (0 after a month rollover)."""
        month, used = self._read_quota()
        return used if month == self._current_month() else 0

    # -- search -------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        max_results: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[SearchResult]:
        """Search news via AskNews, drawing down the monthly credit budget.

        ``start_date``/``end_date`` are accepted for interface parity but not
        sent: the recon captures showed no date-window params for the
        ``/news/search`` endpoint, so filtering by date is done downstream. When
        AskNews' documented date support is confirmed, map them here.

        Raises:
            SearchQuotaError: The monthly credit cap is already reached.
            SearchError: The HTTP call failed or returned an unexpected shape.
        """
        del start_date, end_date  # documented no-op: no date-window params here.

        current_month = self._current_month()
        month, used = self._read_quota()
        if month != current_month:
            used = 0  # calendar-month rollover resets the counter
        if used >= self._cap:
            raise SearchQuotaError(
                f"AskNews monthly credit cap reached ({used}/{self._cap} for "
                f"{current_month}); refusing to call."
            )

        params: dict[str, str | int] = {
            "query": query,
            "n_articles": max_results,
            "method": self._method,
        }
        try:
            response = self._throttled_get("/v1/news/search", params)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise SearchError(f"AskNews request failed: {exc}") from exc

        if not isinstance(payload, dict):
            raise SearchError("AskNews returned a non-object response.")

        # Record credit draw before returning, so a mid-batch crash still bills.
        credits = 0
        usage = payload.get("usage")
        if isinstance(usage, dict):
            raw_credits = usage.get("credits")
            if isinstance(raw_credits, int) and not isinstance(raw_credits, bool):
                credits = raw_credits
        self._write_quota(current_month, used + credits)

        articles = payload.get("as_dicts")
        if not isinstance(articles, list):
            return []
        results: list[SearchResult] = []
        for article in articles:
            if isinstance(article, dict):
                results.append(self._to_result(article))
        return results

    def _throttled_get(self, path: str, params: dict[str, str | int]) -> httpx.Response:
        """GET with a request-interval floor and one retry on 429.

        The free tier allows 1 request per 2 seconds; we space requests to
        respect that and, if a 429 still arrives (shared limiter state on the
        server side), wait ``Retry-After`` (default 5s, capped 30s) and retry
        once. A second 429 propagates to the normal error path.
        """
        now = self._monotonic()
        if self._last_request_at is not None:
            wait = self._min_interval - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        response = self._client.get(path, params=params)
        self._last_request_at = self._monotonic()
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = min(float(retry_after), 30.0) if retry_after else 5.0
            except ValueError:
                delay = 5.0
            self._sleep(max(delay, self._min_interval))
            response = self._client.get(path, params=params)
            self._last_request_at = self._monotonic()
        return response

    @staticmethod
    def _to_result(article: dict[str, Any]) -> SearchResult:
        """Normalize one AskNews article dict into a :class:`SearchResult`."""
        url = str(article.get("article_url") or "")
        # eng_title is the English title; fall back to the alt `title` field.
        title = str(article.get("eng_title") or article.get("title") or "")
        snippet = str(article.get("summary") or "")
        domain = str(article.get("domain_url") or _domain_of(url))
        return SearchResult(
            url=url,
            title=title,
            snippet=snippet,
            published_at=_parse_iso8601(article.get("pub_date")),
            source_domain=domain,
            provider=ASKNEWS_PROVIDER,
            raw=article,
        )


class TavilyClient:
    """Search the general web via Tavily's ``POST /search`` endpoint.

    Normalizes ``results`` into :class:`SearchResult`\\ s. Tavily surfaces no
    domain field, so we derive it from each URL, and it only returns a
    ``published_date`` when the request sets ``topic="news"`` — set ``topic`` at
    construction (the factory does so for the news role) to get dates back.
    Tavily reports no per-request credit usage, so there is no quota metering
    here; billing is tracked provider-side.
    """

    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 20.0,
        topic: str | None = None,
    ) -> None:
        """Configure the client.

        Args:
            api_key: The Tavily key. Sent as ``Authorization: Bearer <key>``.
            transport: Optional ``httpx`` transport (tests inject a
                ``MockTransport``).
            timeout_s: Per-request HTTP timeout.
            topic: When set to ``"news"``, sent in the payload so each result
                carries a ``published_date``. ``None`` uses Tavily's default
                (general) topic, which omits dates.
        """
        self._api_key = api_key
        self._topic = topic
        self._client = httpx.Client(
            base_url=_TAVILY_BASE_URL,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": _BROWSER_UA,
            },
            timeout=timeout_s,
        )

    def __enter__(self) -> TavilyClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def search(
        self,
        query: str,
        *,
        max_results: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[SearchResult]:
        """Search via Tavily; ``start_date``/``end_date`` filter server-side.

        Dates are sent as ISO ``YYYY-MM-DD`` strings when supplied.

        Raises:
            SearchError: The HTTP call failed or returned an unexpected shape.
        """
        payload: dict[str, Any] = {"query": query, "max_results": max_results}
        if self._topic is not None:
            payload["topic"] = self._topic
        if start_date is not None:
            payload["start_date"] = start_date.isoformat()
        if end_date is not None:
            payload["end_date"] = end_date.isoformat()

        try:
            response = self._client.post("/search", json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise SearchError(f"Tavily request failed: {exc}") from exc

        if not isinstance(body, dict):
            raise SearchError("Tavily returned a non-object response.")
        results_raw = body.get("results")
        if not isinstance(results_raw, list):
            return []
        results: list[SearchResult] = []
        for item in results_raw:
            if isinstance(item, dict):
                results.append(self._to_result(item))
        return results

    @staticmethod
    def _to_result(item: dict[str, Any]) -> SearchResult:
        """Normalize one Tavily result dict into a :class:`SearchResult`."""
        url = str(item.get("url") or "")
        return SearchResult(
            url=url,
            title=str(item.get("title") or ""),
            snippet=str(item.get("content") or ""),
            # published_date only present with topic="news"; None otherwise.
            published_at=_parse_rfc1123(item.get("published_date")),
            source_domain=_domain_of(url),
            provider=TAVILY_PROVIDER,
            raw=item,
        )


class PerplexitySearchClient:
    """Search the general web via Perplexity's ``POST /search`` endpoint.

    This is Perplexity's standalone **Search API** — raw ranked web results, NOT
    the Sonar chat/answers API (ADR-0004 rejected LLM-synthesized answers for the
    retrieval layer; we want excerpts we can weight and cite, not a paragraph).
    Normalizes the top-level ``results`` array into :class:`SearchResult`\\ s.
    Perplexity surfaces no domain field, so we derive it from each URL, and each
    hit carries an optional ISO-8601 (``YYYY-MM-DD``) ``date`` field which maps to
    ``published_at`` (``None`` when absent).

    The Search API bills a flat rate per request (docs: $5 / 1,000 requests) with
    no token charges, and rate limits are tiered by lifetime spend (the docs'
    ladder runs 20-100 RPM; Tier 0 with no purchase is the low end, ~20 RPM = one
    request per ~3 s). We space consecutive requests by ``min_request_interval_s``
    and retry once on a 429, mirroring :class:`AskNewsClient`.
    """

    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 20.0,
        min_request_interval_s: float = 3.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Configure the client.

        Args:
            api_key: The Perplexity key. Sent as ``Authorization: Bearer <key>``.
            transport: Optional ``httpx`` transport (tests inject a
                ``MockTransport``).
            timeout_s: Per-request HTTP timeout.
            min_request_interval_s: Floor between consecutive requests from this
                client. Tier 0 (no purchase) is ~20 RPM per the docs' rate-limit
                ladder, i.e. one request per ~3 s; spacing requests keeps us under
                it. Set to 0 to disable throttling.
            monotonic: Clock for the throttle; injectable for tests.
            sleep: Sleeper for throttle/backoff waits; injectable for tests.
        """
        self._api_key = api_key
        self._min_interval = min_request_interval_s
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_at: float | None = None
        self._client = httpx.Client(
            base_url=_PERPLEXITY_BASE_URL,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": _BROWSER_UA,
            },
            timeout=timeout_s,
        )

    def __enter__(self) -> PerplexitySearchClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def search(
        self,
        query: str,
        *,
        max_results: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[SearchResult]:
        """Search via Perplexity's Search API; date bounds filter server-side.

        ``start_date``/``end_date`` map to the API's ``search_after_date_filter``/
        ``search_before_date_filter`` params, which the Search API expects in
        ``MM/DD/YYYY`` format (its response ``date`` is ISO ``YYYY-MM-DD``).

        Raises:
            SearchError: The HTTP call failed or returned an unexpected shape.
        """
        payload: dict[str, Any] = {"query": query, "max_results": max_results}
        if start_date is not None:
            payload["search_after_date_filter"] = start_date.strftime("%m/%d/%Y")
        if end_date is not None:
            payload["search_before_date_filter"] = end_date.strftime("%m/%d/%Y")

        try:
            response = self._throttled_post("/search", payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise SearchError(f"Perplexity request failed: {exc}") from exc

        if not isinstance(body, dict):
            raise SearchError("Perplexity returned a non-object response.")
        results_raw = body.get("results")
        if not isinstance(results_raw, list):
            return []
        results: list[SearchResult] = []
        for item in results_raw:
            if isinstance(item, dict):
                results.append(self._to_result(item))
        return results

    def _throttled_post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        """POST with a request-interval floor and one retry on 429.

        Spaces consecutive requests by ``min_request_interval_s`` to respect the
        tier rate limit and, if a 429 still arrives, waits ``Retry-After``
        (default 5s, capped 30s) and retries once. A second 429 propagates to the
        normal error path. Mirrors :meth:`AskNewsClient._throttled_get`.
        """
        now = self._monotonic()
        if self._last_request_at is not None:
            wait = self._min_interval - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        response = self._client.post(path, json=payload)
        self._last_request_at = self._monotonic()
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = min(float(retry_after), 30.0) if retry_after else 5.0
            except ValueError:
                delay = 5.0
            self._sleep(max(delay, self._min_interval))
            response = self._client.post(path, json=payload)
            self._last_request_at = self._monotonic()
        return response

    @staticmethod
    def _to_result(item: dict[str, Any]) -> SearchResult:
        """Normalize one Perplexity result dict into a :class:`SearchResult`."""
        url = str(item.get("url") or "")
        return SearchResult(
            url=url,
            title=str(item.get("title") or ""),
            snippet=str(item.get("snippet") or ""),
            # `date` is ISO YYYY-MM-DD when present; None otherwise.
            published_at=_parse_iso8601(item.get("date")),
            source_domain=_domain_of(url),
            provider=PERPLEXITY_PROVIDER,
            raw=item,
        )


def _require_env(env_var: str) -> str:
    """Return an env var's value or raise :class:`SearchError` if it is unset."""
    value = os.environ.get(env_var)
    if not value:
        raise SearchError(f"Missing {env_var} in the environment; set it in engine/.env.")
    return value


def build_search_client(
    provider: str,
    *,
    settings: RetrievalSettings,
    data_dir: Path,
    api_key: str | None = None,
    transport: httpx.BaseTransport | None = None,
    news_role: bool = False,
) -> SearchClient:
    """Return the search client for a provider name (the single dispatch point).

    Args:
        provider: ``"asknews"``, ``"tavily"``, or ``"perplexity"``.
        settings: The ``[retrieval]`` settings (timeouts, quota cap).
        data_dir: Directory under ``[data]`` where the AskNews quota counter
            file (``asknews_quota.json``) is persisted.
        api_key: The provider key; read from the environment when ``None``.
        transport: Optional ``httpx`` transport for tests.
        news_role: When this client fills the *news* role, Tavily is told to use
            ``topic="news"`` (AskNews ignores this; it is always news).

    Raises:
        SearchError: The provider name is unknown, or its key is missing.
    """
    if provider == ASKNEWS_PROVIDER:
        key = api_key if api_key is not None else _require_env(ASK_NEWS_API_KEY_ENV)
        return AskNewsClient(
            key,
            quota_path=data_dir / ASKNEWS_QUOTA_FILENAME,
            monthly_credit_cap=settings.asknews_monthly_credit_cap,
            transport=transport,
            timeout_s=settings.timeout_seconds,
        )
    if provider == TAVILY_PROVIDER:
        key = api_key if api_key is not None else _require_env(TAVILY_API_KEY_ENV)
        return TavilyClient(
            key,
            transport=transport,
            timeout_s=settings.timeout_seconds,
            topic="news" if news_role else None,
        )
    if provider == PERPLEXITY_PROVIDER:
        key = api_key if api_key is not None else _require_env(PERPLEXITY_API_KEY_ENV)
        return PerplexitySearchClient(
            key,
            transport=transport,
            timeout_s=settings.timeout_seconds,
        )
    raise SearchError(
        f"Unknown search provider {provider!r}; supported: "
        f"{ASKNEWS_PROVIDER}, {TAVILY_PROVIDER}, {PERPLEXITY_PROVIDER}."
    )
