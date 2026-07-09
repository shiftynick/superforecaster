"""A thin, testable client for the Metaculus REST API.

Metaculus is a forecasting website; its API lets us list "posts" (each post
wraps one forecasting question) and fetch a single post by id. This module only
knows how to *talk to the API and hand back raw dicts* — turning those dicts
into clean snapshots is :mod:`superforecaster.models`'s job.

Two design choices make this easy to test without touching the network:

- The HTTP transport is injectable, so tests pass an ``httpx.MockTransport``.
- The pacing ``sleep`` is injectable, so tests run instantly instead of waiting.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx

# Be polite: at most one request per second against the public API.
_MIN_INTERVAL_SECONDS = 1.0


class MetaculusClient:
    """Talks to the Metaculus API and returns raw post dicts.

    Args:
        token: Metaculus API token, sent as ``Authorization: Token <token>``.
        base_url: API root; override in tests.
        transport: Optional ``httpx`` transport. Injecting an
            ``httpx.MockTransport`` lets tests answer requests from fixtures.
        sleep: The function used to pause between requests (defaults to
            ``time.sleep``); tests pass a no-op to skip real waiting.
    """

    def __init__(
        self,
        token: str,
        base_url: str = "https://www.metaculus.com/api",
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._sleep = sleep
        self._last_request_at: float | None = None
        self._client = httpx.Client(
            base_url=self._base_url,
            transport=transport,
            headers={
                "Authorization": f"Token {token}",
                "Accept-Language": "en",
            },
            timeout=30.0,
        )

    def __enter__(self) -> MetaculusClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def _pace(self) -> None:
        """Sleep just long enough to keep under one request per second."""
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait = _MIN_INTERVAL_SECONDS - elapsed
        if wait > 0:
            self._sleep(wait)

    def _get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET with pacing plus a single retry that honours ``Retry-After`` on 429."""
        self._pace()
        response = self._client.get(url, params=params)
        self._last_request_at = time.monotonic()

        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            self._sleep(retry_after)
            response = self._client.get(url, params=params)
            self._last_request_at = time.monotonic()

        response.raise_for_status()
        return response

    def iter_open_binary_posts(
        self, page_size: int = 100, max_posts: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield raw open binary post dicts, newest first, paging as needed.

        Walks the list endpoint using ``limit``/``offset``. Stops when a page
        comes back empty, when the envelope reports no ``next`` page, or when
        ``max_posts`` posts have been yielded. ``page_size`` is capped at the
        API maximum of 100.
        """
        limit = max(1, min(page_size, 100))
        offset = 0
        yielded = 0

        while True:
            if max_posts is not None:
                remaining = max_posts - yielded
                if remaining <= 0:
                    return
                limit = min(limit, remaining)

            params: dict[str, Any] = {
                "statuses": "open",
                "forecast_type": "binary",
                "limit": limit,
                "offset": offset,
                "order_by": "-published_at",
                "with_cp": "true",
            }
            response = self._get("/posts/", params=params)
            results, has_next = _unpack_list(response.json())
            if not results:
                return

            for post in results:
                yield post
                yielded += 1
                if max_posts is not None and yielded >= max_posts:
                    return

            if not has_next:
                return
            offset += len(results)

    def iter_tournament_binary_posts(
        self, tournament: str, page_size: int = 50, max_posts: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield raw open binary posts belonging to one tournament, paging as needed.

        Same list endpoint and pagination as :meth:`iter_open_binary_posts`, but
        scoped to a tournament via the ``tournaments=<slug>`` parameter. Stops when
        a page comes back empty, when the envelope reports no ``next`` page, or when
        ``max_posts`` posts have been yielded. ``page_size`` is capped at the API
        maximum of 100.
        """
        limit = max(1, min(page_size, 100))
        offset = 0
        yielded = 0

        while True:
            if max_posts is not None:
                remaining = max_posts - yielded
                if remaining <= 0:
                    return
                limit = min(limit, remaining)

            params: dict[str, Any] = {
                "tournaments": tournament,
                "statuses": "open",
                "forecast_type": "binary",
                "limit": limit,
                "offset": offset,
                "order_by": "-published_at",
                "with_cp": "true",
            }
            response = self._get("/posts/", params=params)
            results, has_next = _unpack_list(response.json())
            if not results:
                return

            for post in results:
                yield post
                yielded += 1
                if max_posts is not None and yielded >= max_posts:
                    return

            if not has_next:
                return
            offset += len(results)

    def get_post(self, post_id: int) -> dict[str, Any]:
        """Fetch a single post by id and return its raw dict."""
        response = self._get(f"/posts/{post_id}/")
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"unexpected post payload for {post_id}: {type(data)!r}")
        return data


def _unpack_list(payload: Any) -> tuple[list[dict[str, Any]], bool]:
    """Return ``(results, has_next)`` from either a DRF envelope or a bare list.

    Metaculus *probably* wraps results in ``{"count", "next", "results"}`` but we
    tolerate a plain list too. ``has_next`` is ``True`` only when an envelope's
    ``next`` field is a non-empty value.
    """
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return results, bool(payload.get("next"))
        return [], False
    if isinstance(payload, list):
        return payload, False
    return [], False


def _parse_retry_after(value: str | None) -> float:
    """Read a ``Retry-After`` header (seconds); default to one second."""
    if not value:
        return _MIN_INTERVAL_SECONDS
    try:
        return max(0.0, float(value))
    except ValueError:
        return _MIN_INTERVAL_SECONDS
