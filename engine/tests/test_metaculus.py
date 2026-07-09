"""Tests for MetaculusClient using httpx.MockTransport (no real network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx

from superforecaster.metaculus import MetaculusClient
from superforecaster.models import QuestionSnapshot

from . import fixtures


def _noop_sleep(_seconds: float) -> None:
    """Stand-in for time.sleep so tests never actually wait."""


def test_pagination_across_two_pages() -> None:
    """A DRF envelope with a `next` link should be followed to page two."""
    page_one = [fixtures.open_binary_post(101, 201), fixtures.group_post(103)]
    page_two = [fixtures.resolved_yes_post(104, 204)]
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset", "0"))
        calls.append(dict(request.url.params))
        if offset == 0:
            body = {"count": 3, "next": "?offset=2", "results": page_one}
        else:
            body = {"count": 3, "next": None, "results": page_two}
        return httpx.Response(200, json=body)

    client = MetaculusClient(token="t", transport=httpx.MockTransport(handler), sleep=_noop_sleep)
    posts = list(client.iter_open_binary_posts(page_size=2))
    client.close()

    assert [p["id"] for p in posts] == [101, 103, 104]
    # First page requested statuses=open & forecast_type=binary.
    assert calls[0]["statuses"] == "open"
    assert calls[0]["forecast_type"] == "binary"
    assert len(calls) == 2


def test_max_posts_stops_early() -> None:
    page = [fixtures.open_binary_post(i, i + 1000) for i in range(5)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"next": "?offset=99", "results": page})

    client = MetaculusClient(token="t", transport=httpx.MockTransport(handler), sleep=_noop_sleep)
    posts = list(client.iter_open_binary_posts(page_size=100, max_posts=3))
    client.close()
    assert len(posts) == 3


def test_bare_list_response_is_tolerated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[fixtures.open_binary_post()])

    client = MetaculusClient(token="t", transport=httpx.MockTransport(handler), sleep=_noop_sleep)
    posts = list(client.iter_open_binary_posts())
    client.close()
    assert len(posts) == 1


def test_get_post_and_field_extraction() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/posts/104/")
        return httpx.Response(200, json=fixtures.resolved_yes_post())

    client = MetaculusClient(
        token="secret", transport=httpx.MockTransport(handler), sleep=_noop_sleep
    )
    post = client.get_post(104)
    client.close()

    snap = QuestionSnapshot.from_api_post(post, datetime(2026, 7, 9, tzinfo=UTC))
    assert snap.resolution == "yes"
    assert snap.community_prediction == 0.9


def test_429_is_retried_once() -> None:
    slept: list[float] = []
    state = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["count"] += 1
        if state["count"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, json={})
        return httpx.Response(200, json={"results": [fixtures.open_binary_post()]})

    client = MetaculusClient(token="t", transport=httpx.MockTransport(handler), sleep=slept.append)
    posts = list(client.iter_open_binary_posts())
    client.close()

    assert len(posts) == 1
    assert 2.0 in slept  # honoured Retry-After


def test_auth_header_is_sent() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        seen["lang"] = request.headers.get("Accept-Language", "")
        return httpx.Response(200, json={"results": []})

    client = MetaculusClient(
        token="abc123", transport=httpx.MockTransport(handler), sleep=_noop_sleep
    )
    list(client.iter_open_binary_posts())
    client.close()
    assert seen["auth"] == "Token abc123"
    assert seen["lang"] == "en"


def test_json_roundtrip_shape() -> None:
    """Sanity: fixtures serialize cleanly (guards against accidental non-JSON)."""
    json.dumps(fixtures.open_binary_post())
