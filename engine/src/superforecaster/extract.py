"""Tiered content extraction — the retrieval stack's extraction layer.

A search hit gives us a URL and a short snippet; to reason well the forecaster
often needs the article body. :func:`Extractor.extract` fetches and cleans one
URL, trying cheaper methods first and falling back on failure:

- **Tier 0** — fetch the raw HTML with ``httpx`` and run ``trafilatura`` locally
  (free, no external service). A *quality gate* (minimum characters and words)
  rejects SPA shells and paywall teasers so they register as failures rather
  than polluting the context with junk.
- **Tier 1** — on Tier-0 failure, fetch through the Jina Reader proxy
  (``https://r.jina.ai/<url>``), which renders JS and returns clean
  text/markdown. Works keyless at a lower rate limit; an optional
  ``JINA_API_KEY`` raises it.

If both tiers fail the result is ``ok=False`` with ``tier="none"``. The tier
that succeeded is recorded on every :class:`ExtractedDoc` so hit-rates can be
logged per run (ADR-0004's "instrument before buying a paid Tier 2").

The HTTP transport is injectable, so tests feed static HTML through an
``httpx.MockTransport`` and let ``trafilatura`` run for real on it.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx
import trafilatura
from pydantic import BaseModel

from .config import RetrievalSettings

# Env var holding the optional Jina Reader key (higher rate limit when present).
JINA_API_KEY_ENV = "JINA_API_KEY"

_JINA_BASE_URL = "https://r.jina.ai/"

# Tier labels recorded on the result.
TIER_HTTPX_TRAFILATURA = "tier0"
TIER_JINA = "tier1"
TIER_NONE = "none"

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class ExtractedDoc(BaseModel):
    """The outcome of extracting one URL.

    Attributes:
        url: The URL that was fetched.
        text: The extracted main text (empty string when extraction failed).
        title: The article title, when the extractor recovered one.
        published_at: The publish datetime, when recovered and parseable.
        tier: Which tier produced the text (``"tier0"``/``"tier1"``), or
            ``"none"`` when every tier failed.
        ok: Whether usable text cleared the quality gate.
    """

    url: str
    text: str
    title: str | None
    published_at: datetime | None
    tier: str
    ok: bool


def _word_count(text: str) -> int:
    return len(text.split())


def _parse_trafilatura_date(value: object) -> datetime | None:
    """Parse trafilatura's ``date`` metadata (``YYYY-MM-DD`` or ISO), else None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class Extractor:
    """Fetch-and-clean one URL at a time through the tier ladder."""

    def __init__(
        self,
        *,
        min_chars: int,
        min_words: int,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 20.0,
        jina_api_key: str | None = None,
    ) -> None:
        """Configure the extractor.

        Args:
            min_chars: Quality gate — text shorter than this fails a tier.
            min_words: Quality gate — text with fewer words than this fails.
            transport: Optional ``httpx`` transport (tests inject a
                ``MockTransport``).
            timeout_s: Per-request HTTP timeout.
            jina_api_key: Optional Jina Reader key for the Tier-1 fallback; when
                set it is sent as a Bearer token to lift the rate limit.
        """
        self._min_chars = min_chars
        self._min_words = min_words
        self._jina_api_key = jina_api_key
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": _BROWSER_UA},
        )

    def __enter__(self) -> Extractor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def _passes_gate(self, text: str) -> bool:
        """True when text is long enough (chars and words) to count as usable."""
        return len(text) >= self._min_chars and _word_count(text) >= self._min_words

    def extract(self, url: str) -> ExtractedDoc:
        """Extract one URL, trying Tier 0 then Tier 1; both fail -> ``ok=False``."""
        tier0 = self._tier0(url)
        if tier0 is not None:
            return tier0
        tier1 = self._tier1(url)
        if tier1 is not None:
            return tier1
        return ExtractedDoc(
            url=url,
            text="",
            title=None,
            published_at=None,
            tier=TIER_NONE,
            ok=False,
        )

    def _tier0(self, url: str) -> ExtractedDoc | None:
        """Fetch raw HTML and run trafilatura; ``None`` if the gate is not met."""
        try:
            response = self._client.get(url)
            response.raise_for_status()
            html = response.text
        except httpx.HTTPError:
            return None

        text = trafilatura.extract(html) or ""
        if not self._passes_gate(text):
            return None

        title: str | None = None
        published_at: datetime | None = None
        try:
            metadata: Any = trafilatura.extract_metadata(html)
        except Exception:  # metadata is best-effort, never fatal
            metadata = None
        if metadata is not None:
            raw_title = getattr(metadata, "title", None)
            if isinstance(raw_title, str) and raw_title:
                title = raw_title
            published_at = _parse_trafilatura_date(getattr(metadata, "date", None))

        return ExtractedDoc(
            url=url,
            text=text,
            title=title,
            published_at=published_at,
            tier=TIER_HTTPX_TRAFILATURA,
            ok=True,
        )

    def _tier1(self, url: str) -> ExtractedDoc | None:
        """Fetch via the Jina Reader proxy; ``None`` if the gate is not met.

        Jina returns clean text/markdown directly, so no trafilatura pass is
        needed. Title/date are not parsed out of the markdown here.
        """
        headers = {}
        if self._jina_api_key:
            headers["Authorization"] = f"Bearer {self._jina_api_key}"
        try:
            response = self._client.get(_JINA_BASE_URL + url, headers=headers)
            response.raise_for_status()
            text = response.text or ""
        except httpx.HTTPError:
            return None

        if not self._passes_gate(text):
            return None
        return ExtractedDoc(
            url=url,
            text=text,
            title=None,
            published_at=None,
            tier=TIER_JINA,
            ok=True,
        )


def build_extractor(
    settings: RetrievalSettings,
    *,
    transport: httpx.BaseTransport | None = None,
    jina_api_key: str | None = None,
) -> Extractor:
    """Return an :class:`Extractor` configured from ``[retrieval]`` settings.

    The optional Jina key is read from ``JINA_API_KEY`` when not passed
    explicitly; its absence is fine (Jina works keyless at a lower rate limit).
    """
    key = jina_api_key if jina_api_key is not None else os.environ.get(JINA_API_KEY_ENV)
    return Extractor(
        min_chars=settings.extract_min_chars,
        min_words=settings.extract_min_words,
        transport=transport,
        timeout_s=settings.timeout_seconds,
        jina_api_key=key or None,
    )
