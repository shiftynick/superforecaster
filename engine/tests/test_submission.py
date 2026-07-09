"""Tests for SubmissionClient using httpx.MockTransport (no real network)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from superforecaster.submission import SubmissionClient, SubmissionError


def _noop_sleep(_seconds: float) -> None:
    """Stand-in for time.sleep so tests never actually wait."""


def test_submit_binary_forecast_posts_expected_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("Authorization", "")
        seen["body"] = request.read().decode()
        return httpx.Response(201, json={"ok": True})

    client = SubmissionClient(
        bot_token="botsecret", transport=httpx.MockTransport(handler), sleep=_noop_sleep
    )
    client.submit_binary_forecast(201, 0.42)
    client.close()

    assert seen["path"].endswith("/questions/forecast/")
    assert seen["auth"] == "Token botsecret"
    assert json.loads(seen["body"]) == [{"question": 201, "probability_yes": 0.42}]


def test_submit_binary_forecast_clamps_high() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={})

    client = SubmissionClient(
        bot_token="t", transport=httpx.MockTransport(handler), sleep=_noop_sleep
    )
    client.submit_binary_forecast(7, 0.9999)
    client.close()

    assert json.loads(seen["body"]) == [{"question": 7, "probability_yes": 0.999}]


def test_submit_binary_forecast_clamps_low() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={})

    client = SubmissionClient(
        bot_token="t", transport=httpx.MockTransport(handler), sleep=_noop_sleep
    )
    client.submit_binary_forecast(7, 0.0)
    client.close()

    assert json.loads(seen["body"]) == [{"question": 7, "probability_yes": 0.001}]


def test_post_rationale_comment_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["json"] = request.read().decode()
        return httpx.Response(201, json={"id": 1})

    client = SubmissionClient(
        bot_token="t", transport=httpx.MockTransport(handler), sleep=_noop_sleep
    )
    client.post_rationale_comment(999, "here is why")
    client.close()

    assert seen["path"].endswith("/comments/create/")
    assert json.loads(seen["json"]) == {
        "on_post": 999,
        "text": "here is why",
        "is_private": True,
        "included_forecast": True,
    }


def test_error_on_400_includes_body_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="already forecast on this question")

    client = SubmissionClient(
        bot_token="t", transport=httpx.MockTransport(handler), sleep=_noop_sleep
    )
    with pytest.raises(SubmissionError) as excinfo:
        client.submit_binary_forecast(1, 0.5)
    client.close()

    assert excinfo.value.status_code == 400
    assert "already forecast on this question" in str(excinfo.value)


def test_comment_error_on_400_includes_body_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad comment")

    client = SubmissionClient(
        bot_token="t", transport=httpx.MockTransport(handler), sleep=_noop_sleep
    )
    with pytest.raises(SubmissionError) as excinfo:
        client.post_rationale_comment(1, "x")
    client.close()

    assert excinfo.value.status_code == 400
    assert "bad comment" in str(excinfo.value)


def test_429_then_success_retry() -> None:
    slept: list[float] = []
    state = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["count"] += 1
        if state["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"}, json={})
        return httpx.Response(200, json={"ok": True})

    client = SubmissionClient(
        bot_token="t", transport=httpx.MockTransport(handler), sleep=slept.append
    )
    client.submit_binary_forecast(1, 0.5)
    client.close()

    assert state["count"] == 2
    assert 3.0 in slept
