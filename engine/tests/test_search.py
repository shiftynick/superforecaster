"""Tests for the search layer, using httpx.MockTransport (no real network).

Response JSON mirrors the shapes captured in the live retrieval recon
(2026-07-09): AskNews ``/v1/news/search`` and Tavily ``/search``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from superforecaster.config import RetrievalSettings
from superforecaster.search import (
    AskNewsClient,
    PerplexitySearchClient,
    SearchError,
    SearchQuotaError,
    TavilyClient,
    build_search_client,
    read_asknews_credits_used,
)

# --- Realistic response fixtures (trimmed from the recon captures) ----------

_ASKNEWS_ARTICLE: dict[str, Any] = {
    "article_url": "https://talkmarkets.com/article/fed-minutes-1783597034",
    "article_id": "3a9ef717-6763-4be0-92b9-b5a89ad2f466",
    "eng_title": "Fed Minutes Reveal Deep Divide Over Future Interest Rate Hikes",
    "title": "Fed Minutes: Deep Divide",
    "summary": "According to the minutes of the June FOMC meeting, officials were split.",
    "domain_url": "talkmarkets.com",
    "source_id": "TalkMarkets",
    "pub_date": "2026-07-09T16:00:00Z",
    "language": "en",
}


def _asknews_body(articles: list[dict[str, Any]], credits: int = 1) -> dict[str, Any]:
    return {
        "as_dicts": articles,
        "as_string": None,
        "offset": len(articles),
        "usage": {"credits": credits},
    }


_TAVILY_NEWS_RESULT: dict[str, Any] = {
    "url": "https://en.wikipedia.org/wiki/2026_Iran_war_ceasefire",
    "title": "2026 Iran war ceasefire",
    "content": "A ceasefire between the parties took effect in June 2026.",
    "score": 0.8396,
    "raw_content": None,
    "published_date": "Tue, 16 Jun 2026 09:12:25 GMT",
}

_TAVILY_PLAIN_RESULT: dict[str, Any] = {
    "url": "https://example.com/general-page",
    "title": "A general web page",
    "content": "Some general web content with no date attached.",
    "score": 0.5,
    "raw_content": None,
}


_PERPLEXITY_RESULT: dict[str, Any] = {
    "title": "Federal Reserve holds rates steady in July 2026",
    "url": "https://www.reuters.com/markets/us/fed-july-2026",
    "snippet": "The Federal Reserve left its benchmark rate unchanged at the July meeting.",
    "date": "2026-07-15",
    "last_updated": "2026-07-16",
}

_PERPLEXITY_RESULT_NO_DATE: dict[str, Any] = {
    "title": "An undated explainer page",
    "url": "https://example.org/explainer",
    "snippet": "Background with no publication date attached.",
}


def _settings() -> RetrievalSettings:
    return RetrievalSettings()


# --- AskNews ----------------------------------------------------------------


def test_asknews_parses_and_normalizes(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=_asknews_body([_ASKNEWS_ARTICLE]))

    client = AskNewsClient(
        "ank_test",
        quota_path=tmp_path / "q.json",
        monthly_credit_cap=900,
        transport=httpx.MockTransport(handler),
        today=lambda: date(2026, 7, 9),
    )
    results = client.search("Fed rates", max_results=3)
    client.close()

    assert seen["auth"] == "Bearer ank_test"  # Bearer, not the documented APIKey
    assert seen["path"] == "/v1/news/search"
    assert seen["params"] == {"query": "Fed rates", "n_articles": "3", "method": "nl"}

    assert len(results) == 1
    r = results[0]
    assert r.url == _ASKNEWS_ARTICLE["article_url"]
    assert r.title == _ASKNEWS_ARTICLE["eng_title"]
    assert r.snippet == _ASKNEWS_ARTICLE["summary"]
    assert r.source_domain == "talkmarkets.com"
    assert r.provider == "asknews"
    # ISO8601 with Z parsed to a tz-aware datetime.
    assert r.published_at is not None
    assert r.published_at.year == 2026 and r.published_at.month == 7
    assert r.published_at.tzinfo is not None


def test_asknews_missing_pub_date_yields_none(tmp_path: Path) -> None:
    article = {k: v for k, v in _ASKNEWS_ARTICLE.items() if k != "pub_date"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_asknews_body([article]))

    client = AskNewsClient(
        "k",
        quota_path=tmp_path / "q.json",
        monthly_credit_cap=900,
        transport=httpx.MockTransport(handler),
        today=lambda: date(2026, 7, 9),
    )
    results = client.search("q", max_results=1)
    client.close()
    assert results[0].published_at is None


def test_asknews_quota_increments_and_persists(tmp_path: Path) -> None:
    quota = tmp_path / "asknews_quota.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_asknews_body([_ASKNEWS_ARTICLE], credits=2))

    client = AskNewsClient(
        "k",
        quota_path=quota,
        monthly_credit_cap=900,
        transport=httpx.MockTransport(handler),
        today=lambda: date(2026, 7, 9),
    )
    client.search("q", max_results=1)
    client.search("q", max_results=1)
    client.close()

    saved = json.loads(quota.read_text(encoding="utf-8"))
    assert saved == {"month": "2026-07", "credits_used": 4}


def test_asknews_blocks_when_cap_reached(tmp_path: Path) -> None:
    quota = tmp_path / "asknews_quota.json"
    quota.write_text(json.dumps({"month": "2026-07", "credits_used": 900}), encoding="utf-8")
    called = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        called["n"] += 1
        return httpx.Response(200, json=_asknews_body([_ASKNEWS_ARTICLE]))

    client = AskNewsClient(
        "k",
        quota_path=quota,
        monthly_credit_cap=900,
        transport=httpx.MockTransport(handler),
        today=lambda: date(2026, 7, 9),
    )
    with pytest.raises(SearchQuotaError):
        client.search("q", max_results=1)
    client.close()
    assert called["n"] == 0  # refused before any HTTP call


def test_asknews_resets_counter_on_month_rollover(tmp_path: Path) -> None:
    quota = tmp_path / "asknews_quota.json"
    # Previous month is maxed out, but "today" is a new month -> reset.
    quota.write_text(json.dumps({"month": "2026-06", "credits_used": 900}), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_asknews_body([_ASKNEWS_ARTICLE], credits=1))

    client = AskNewsClient(
        "k",
        quota_path=quota,
        monthly_credit_cap=900,
        transport=httpx.MockTransport(handler),
        today=lambda: date(2026, 7, 1),
    )
    results = client.search("q", max_results=1)
    client.close()

    assert len(results) == 1
    saved = json.loads(quota.read_text(encoding="utf-8"))
    assert saved == {"month": "2026-07", "credits_used": 1}  # reset, then +1


def test_asknews_http_error_becomes_search_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = AskNewsClient(
        "k",
        quota_path=tmp_path / "q.json",
        monthly_credit_cap=900,
        transport=httpx.MockTransport(handler),
        today=lambda: date(2026, 7, 9),
    )
    with pytest.raises(SearchError):
        client.search("q", max_results=1)
    client.close()


# --- Tavily -----------------------------------------------------------------


def test_tavily_news_topic_parses_rfc1123_date() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [_TAVILY_NEWS_RESULT]})

    client = TavilyClient("tvly_k", transport=httpx.MockTransport(handler), topic="news")
    results = client.search(
        "Iran ceasefire",
        max_results=3,
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 30),
    )
    client.close()

    assert seen["auth"] == "Bearer tvly_k"
    assert seen["body"] == {
        "query": "Iran ceasefire",
        "max_results": 3,
        "topic": "news",
        "start_date": "2026-06-01",
        "end_date": "2026-06-30",
    }

    r = results[0]
    assert r.provider == "tavily"
    assert r.snippet == _TAVILY_NEWS_RESULT["content"]
    assert r.source_domain == "en.wikipedia.org"  # derived from url
    # RFC 1123 date parsed to a tz-aware datetime.
    assert r.published_at is not None
    assert r.published_at.year == 2026 and r.published_at.month == 6
    assert r.published_at.tzinfo is not None


def test_tavily_plain_result_has_no_date() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "topic" not in body  # general role omits topic
        return httpx.Response(200, json={"results": [_TAVILY_PLAIN_RESULT]})

    client = TavilyClient("k", transport=httpx.MockTransport(handler))
    results = client.search("general query", max_results=2)
    client.close()

    assert results[0].published_at is None
    assert results[0].source_domain == "example.com"


def test_tavily_http_error_becomes_search_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad key"})

    client = TavilyClient("k", transport=httpx.MockTransport(handler))
    with pytest.raises(SearchError):
        client.search("q", max_results=1)
    client.close()


# --- Perplexity Search API --------------------------------------------------


def test_perplexity_parses_and_normalizes() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [_PERPLEXITY_RESULT]})

    client = PerplexitySearchClient("pplx_k", transport=httpx.MockTransport(handler))
    results = client.search(
        "Fed rate decision",
        max_results=3,
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 18),
    )
    client.close()

    assert seen["auth"] == "Bearer pplx_k"
    assert seen["path"] == "/search"
    # Date bounds map to the MM/DD/YYYY filter params the Search API expects.
    assert seen["body"] == {
        "query": "Fed rate decision",
        "max_results": 3,
        "search_after_date_filter": "07/01/2026",
        "search_before_date_filter": "07/18/2026",
    }

    assert len(results) == 1
    r = results[0]
    assert r.url == _PERPLEXITY_RESULT["url"]
    assert r.title == _PERPLEXITY_RESULT["title"]
    assert r.snippet == _PERPLEXITY_RESULT["snippet"]
    assert r.source_domain == "www.reuters.com"  # derived from url
    assert r.provider == "perplexity"
    # ISO YYYY-MM-DD `date` parsed to a datetime.
    assert r.published_at is not None
    assert r.published_at.year == 2026 and r.published_at.month == 7
    assert r.published_at.day == 15


def test_perplexity_missing_date_yields_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # No date bounds -> no filter params sent.
        assert "search_after_date_filter" not in json.loads(request.content)
        return httpx.Response(200, json={"results": [_PERPLEXITY_RESULT_NO_DATE]})

    client = PerplexitySearchClient("k", transport=httpx.MockTransport(handler))
    results = client.search("no dates", max_results=2)
    client.close()

    assert results[0].published_at is None
    assert results[0].source_domain == "example.org"


def test_perplexity_empty_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client = PerplexitySearchClient("k", transport=httpx.MockTransport(handler))
    results = client.search("nothing", max_results=1)
    client.close()
    assert results == []


def test_perplexity_http_error_becomes_search_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    client = PerplexitySearchClient("k", transport=httpx.MockTransport(handler))
    with pytest.raises(SearchError):
        client.search("q", max_results=1)
    client.close()


# --- Factory ----------------------------------------------------------------


def test_build_search_client_asknews_uses_env_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASK_NEWS_API_KEY", "env_ank")
    client = build_search_client("asknews", settings=_settings(), data_dir=tmp_path)
    assert isinstance(client, AskNewsClient)


def test_build_search_client_tavily_news_role_sets_topic(tmp_path: Path) -> None:
    client = build_search_client(
        "tavily",
        settings=_settings(),
        data_dir=tmp_path,
        api_key="k",
        news_role=True,
    )
    assert isinstance(client, TavilyClient)
    assert client._topic == "news"


def test_build_search_client_missing_key_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(SearchError, match="TAVILY_API_KEY"):
        build_search_client("tavily", settings=_settings(), data_dir=tmp_path)


def test_build_search_client_perplexity_uses_env_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "env_pplx")
    client = build_search_client("perplexity", settings=_settings(), data_dir=tmp_path)
    assert isinstance(client, PerplexitySearchClient)


def test_build_search_client_perplexity_missing_key_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(SearchError, match="PERPLEXITY_API_KEY"):
        build_search_client("perplexity", settings=_settings(), data_dir=tmp_path)


def test_build_search_client_unknown_provider_raises(tmp_path: Path) -> None:
    with pytest.raises(SearchError, match="Unknown search provider"):
        build_search_client("bing", settings=_settings(), data_dir=tmp_path, api_key="k")


# --- Quota reader (for the stats command) -----------------------------------


def test_read_asknews_credits_used_current_month(tmp_path: Path) -> None:
    (tmp_path / "asknews_quota.json").write_text(
        json.dumps({"month": "2026-07", "credits_used": 42}), encoding="utf-8"
    )
    used = read_asknews_credits_used(tmp_path, today=lambda: date(2026, 7, 15))
    assert used == 42


def test_read_asknews_credits_used_resets_on_rollover(tmp_path: Path) -> None:
    (tmp_path / "asknews_quota.json").write_text(
        json.dumps({"month": "2026-06", "credits_used": 42}), encoding="utf-8"
    )
    used = read_asknews_credits_used(tmp_path, today=lambda: date(2026, 7, 1))
    assert used == 0  # a new month shows zero draw


def test_read_asknews_credits_used_missing_file(tmp_path: Path) -> None:
    assert read_asknews_credits_used(tmp_path, today=lambda: date(2026, 7, 1)) == 0


# --- AskNews throttle + 429 retry (free tier: 1 request / 2 seconds) ---------


def test_asknews_spaces_consecutive_requests(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_asknews_body([_ASKNEWS_ARTICLE]))

    clock = {"now": 100.0}
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        clock["now"] += s

    client = AskNewsClient(
        "k",
        quota_path=tmp_path / "q.json",
        monthly_credit_cap=900,
        transport=httpx.MockTransport(handler),
        today=lambda: date(2026, 7, 10),
        min_request_interval_s=2.1,
        monotonic=lambda: clock["now"],
        sleep=fake_sleep,
    )
    client.search("first", max_results=1)
    assert sleeps == []  # first request goes straight through
    client.search("second", max_results=1)  # zero time elapsed since the first
    client.close()

    assert len(sleeps) == 1 and sleeps[0] == pytest.approx(2.1)


def test_asknews_retries_once_on_429(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json=_asknews_body([_ASKNEWS_ARTICLE]))

    sleeps: list[float] = []
    client = AskNewsClient(
        "k",
        quota_path=tmp_path / "q.json",
        monthly_credit_cap=900,
        transport=httpx.MockTransport(handler),
        today=lambda: date(2026, 7, 10),
        monotonic=lambda: 0.0,
        sleep=sleeps.append,
    )
    results = client.search("q", max_results=1)
    client.close()

    assert calls["n"] == 2 and len(results) == 1
    assert 3.0 in sleeps  # honored Retry-After


def test_asknews_second_429_is_search_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    client = AskNewsClient(
        "k",
        quota_path=tmp_path / "q.json",
        monthly_credit_cap=900,
        transport=httpx.MockTransport(handler),
        today=lambda: date(2026, 7, 10),
        monotonic=lambda: 0.0,
        sleep=lambda s: None,
    )
    with pytest.raises(SearchError):
        client.search("q", max_results=1)
    client.close()
