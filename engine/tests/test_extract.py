"""Tests for the tiered extraction layer.

The HTTP layer is mocked with httpx.MockTransport (feeding static HTML/text),
but trafilatura runs for real on that HTML — so these exercise the genuine
Tier-0 extraction path and the quality-gate fallback logic.
"""

from __future__ import annotations

import httpx
import pytest

from superforecaster.config import RetrievalSettings
from superforecaster.extract import (
    TIER_HTTPX_TRAFILATURA,
    TIER_JINA,
    TIER_NONE,
    Extractor,
    build_extractor,
)

# A substantial article body: comfortably over the default 500-char / 80-word
# gate once trafilatura strips boilerplate.
_PARAGRAPH = (
    "The Federal Reserve released the minutes of its June meeting on Wednesday, "
    "revealing a committee more divided over the path of interest rates than the "
    "headline statement suggested. Several participants argued that inflation had "
    "cooled enough to justify a pause, while others warned that services prices "
    "remained sticky and that cutting too soon risked reigniting demand. The "
    "debate over the neutral rate featured prominently, with staff economists "
    "presenting a range of estimates that underscored how much uncertainty still "
    "surrounds the terminal level of policy this cycle."
)

_ARTICLE_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <title>Fed Minutes Reveal Deep Divide</title>
  <meta property="article:published_time" content="2026-07-09T16:00:00Z" />
</head>
<body>
  <nav>Home | Markets | Login</nav>
  <article>
    <h1>Fed Minutes Reveal Deep Divide Over Interest Rates</h1>
    <p>{_PARAGRAPH}</p>
    <p>{_PARAGRAPH}</p>
  </article>
  <footer>Copyright 2026</footer>
</body>
</html>
"""

# A JS-only SPA shell: trafilatura extracts little or nothing -> Tier-0 fails.
_SPA_SHELL_HTML = (
    "<!DOCTYPE html><html><head><title>App</title></head>"
    "<body><div id='root'></div><script src='/app.js'></script></body></html>"
)

_JINA_MARKDOWN = "Title: Fed Minutes\n\n" + _PARAGRAPH + "\n\n" + _PARAGRAPH

_SHORT_TEXT = "Too short."


def _settings() -> RetrievalSettings:
    return RetrievalSettings()


def _is_jina(request: httpx.Request) -> bool:
    return request.url.host == "r.jina.ai"


def test_tier0_success_extracts_and_gates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert not _is_jina(request)  # never reaches Tier 1
        return httpx.Response(200, text=_ARTICLE_HTML)

    extractor = Extractor(
        min_chars=500,
        min_words=80,
        transport=httpx.MockTransport(handler),
    )
    doc = extractor.extract("https://example.com/fed")
    extractor.close()

    assert doc.ok is True
    assert doc.tier == TIER_HTTPX_TRAFILATURA
    assert len(doc.text) >= 500
    assert "Federal Reserve" in doc.text
    assert doc.title is not None and "Fed Minutes" in doc.title
    assert doc.published_at is not None and doc.published_at.year == 2026


def test_tier0_fails_falls_back_to_tier1() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if _is_jina(request):
            # Jina URL is r.jina.ai/<original-url>.
            assert request.url.path.endswith("https://example.com/spa")
            return httpx.Response(200, text=_JINA_MARKDOWN)
        return httpx.Response(200, text=_SPA_SHELL_HTML)

    extractor = Extractor(
        min_chars=500,
        min_words=80,
        transport=httpx.MockTransport(handler),
    )
    doc = extractor.extract("https://example.com/spa")
    extractor.close()

    assert doc.ok is True
    assert doc.tier == TIER_JINA
    assert "Federal Reserve" in doc.text


def test_both_tiers_fail_yields_none_tier() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if _is_jina(request):
            return httpx.Response(200, text=_SHORT_TEXT)
        return httpx.Response(200, text=_SPA_SHELL_HTML)

    extractor = Extractor(
        min_chars=500,
        min_words=80,
        transport=httpx.MockTransport(handler),
    )
    doc = extractor.extract("https://example.com/dead")
    extractor.close()

    assert doc.ok is False
    assert doc.tier == TIER_NONE
    assert doc.text == ""


def test_tier0_http_error_falls_through_to_tier1() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if _is_jina(request):
            return httpx.Response(200, text=_JINA_MARKDOWN)
        return httpx.Response(503, text="unavailable")

    extractor = Extractor(
        min_chars=500,
        min_words=80,
        transport=httpx.MockTransport(handler),
    )
    doc = extractor.extract("https://example.com/err")
    extractor.close()
    assert doc.tier == TIER_JINA and doc.ok is True


def test_tier1_sends_jina_bearer_when_key_present() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if _is_jina(request):
            seen["auth"] = request.headers.get("Authorization", "")
            return httpx.Response(200, text=_JINA_MARKDOWN)
        return httpx.Response(200, text=_SPA_SHELL_HTML)

    extractor = Extractor(
        min_chars=500,
        min_words=80,
        transport=httpx.MockTransport(handler),
        jina_api_key="jina_secret",
    )
    extractor.extract("https://example.com/spa")
    extractor.close()
    assert seen["auth"] == "Bearer jina_secret"


def test_build_extractor_reads_settings_and_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JINA_API_KEY", "env_jina")
    extractor = build_extractor(_settings())
    assert extractor._min_chars == 500
    assert extractor._min_words == 80
    assert extractor._jina_api_key == "env_jina"
    extractor.close()
