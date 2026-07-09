"""A thin, testable client for *writing* to the Metaculus API.

Where :mod:`superforecaster.metaculus` only reads (listing and fetching posts),
this module posts a bot's forecasts back to Metaculus. Two writes are needed for
one tournament forecast, and the tournament rules require both:

- the numeric forecast itself (``POST /questions/forecast/``), and
- a mandatory rationale comment (``POST /comments/create/``).

The design mirrors :class:`~superforecaster.metaculus.MetaculusClient` exactly so
it is just as easy to test without a network:

- The HTTP transport is injectable, so tests pass an ``httpx.MockTransport``.
- The pacing ``sleep`` is injectable, so tests run instantly.

Submissions authenticate with the *bot* token (``METACULUS_BOT_TOKEN``), which is
distinct from the human read token used for scraping.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

# Be polite: at most one request per second against the public API.
_MIN_INTERVAL_SECONDS = 1.0

# Metaculus rejects a forecast of exactly 0 or 1, so every probability is squeezed
# into this closed interval before it is sent.
_MIN_PROB = 0.001
_MAX_PROB = 0.999


class SubmissionError(Exception):
    """Raised when Metaculus rejects a forecast or comment write.

    The message includes the HTTP status and the response body text so a failed
    submission is diagnosable from logs alone, without replaying the request.
    """

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Metaculus write failed with HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class SubmissionClient:
    """Posts a bot's forecasts and rationale comments to the Metaculus API.

    Args:
        bot_token: The bot's API token, sent as ``Authorization: Token <token>``.
            This is the bot token, never the human read token.
        base_url: API root; override in tests.
        transport: Optional ``httpx`` transport. Injecting an
            ``httpx.MockTransport`` lets tests answer requests from fixtures.
        sleep: The function used to pause between requests (defaults to
            ``time.sleep``); tests pass a no-op to skip real waiting.
    """

    def __init__(
        self,
        bot_token: str,
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
                "Authorization": f"Token {bot_token}",
                "Accept-Language": "en",
            },
            timeout=30.0,
        )

    def __enter__(self) -> SubmissionClient:
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

    def _post(self, url: str, payload: Any) -> httpx.Response:
        """POST JSON with pacing plus a single retry that honours 429."""
        self._pace()
        response = self._client.post(url, json=payload)
        self._last_request_at = time.monotonic()

        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            self._sleep(retry_after)
            response = self._client.post(url, json=payload)
            self._last_request_at = time.monotonic()

        return response

    def submit_binary_forecast(self, question_id: int, probability: float) -> None:
        """Submit one binary forecast (probability of YES) for a question.

        The probability is clamped to [0.001, 0.999] first. The body is a JSON
        *list* of one entry, as the endpoint requires. Raises
        :class:`SubmissionError` on any non-2xx response.
        """
        clamped = _clamp(probability)
        payload = [{"question": question_id, "probability_yes": clamped}]
        response = self._post("/questions/forecast/", payload)
        if not response.is_success:
            raise SubmissionError(response.status_code, response.text)

    def post_rationale_comment(self, post_id: int, text: str) -> None:
        """Post the mandatory private rationale comment for a forecast.

        The comment is marked ``is_private`` and ``included_forecast`` so Metaculus
        attaches it to the bot's forecast and auto-publishes it after the question
        closes. Raises :class:`SubmissionError` on any non-2xx response.
        """
        payload = {
            "on_post": post_id,
            "text": text,
            "is_private": True,
            "included_forecast": True,
        }
        response = self._post("/comments/create/", payload)
        if not response.is_success:
            raise SubmissionError(response.status_code, response.text)


def _clamp(probability: float) -> float:
    """Squeeze a probability into [0.001, 0.999] so Metaculus will accept it."""
    return max(_MIN_PROB, min(_MAX_PROB, probability))


def _parse_retry_after(value: str | None) -> float:
    """Read a ``Retry-After`` header (seconds); default to one second."""
    if not value:
        return _MIN_INTERVAL_SECONDS
    try:
        return max(0.0, float(value))
    except ValueError:
        return _MIN_INTERVAL_SECONDS
