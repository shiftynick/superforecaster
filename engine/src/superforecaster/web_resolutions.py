"""Fill resolution gaps by reading Metaculus's public question pages.

Metaculus's REST API hides a question's ``resolution`` field unless the account
that asks forecast on that question — a documented access tier. The outcome is
nonetheless *public on the website*: a resolved question page shows a banner like
``RESOLVED · No``. This module reads that banner and records it honestly:

- Every value carries provenance — the source URL, the fetch time (UTC), and the
  page title — so a reader can always trace where a resolution came from.
- We NEVER guess. A page with no recognisable ``RESOLVED <outcome>`` banner
  yields ``None`` (with a logged reason), not a fabricated outcome.

The browser is quarantined behind a ``page_loader`` seam. :func:`fetch_resolution`
takes a callable that maps a URL to the page's visible text, so tests drive the
parser with fixtures and never launch a browser. :func:`default_page_loader` is
the real seam — a headless Chromium via Playwright — used only by the CLI's
web-resolution pass and the one-off live check.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

_log = logging.getLogger(__name__)

# The public per-post URL. Metaculus routes ``/questions/<post_id>/`` to the
# question page regardless of the (separate) internal question id.
_QUESTION_URL = "https://www.metaculus.com/questions/{post_id}/"

# A realistic desktop-Chrome User-Agent. Metaculus is fronted by Cloudflare,
# which 403s a bare httpx client; a real browser engine plus a normal UA passes.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# Default page-load timeout for the live loader (milliseconds).
DEFAULT_TIMEOUT_MS = 30_000

# The resolution banner: the literal word ``RESOLVED`` immediately followed —
# across only whitespace and separator glyphs, never letters — by the outcome
# word. Because the separator class excludes letters/digits, the match is pinned
# to the word right after ``RESOLVED`` (the banner), never a distant outcome word
# elsewhere on the page. Case-insensitive. The separator class allows whitespace
# plus the glyphs Metaculus renders in the status chip: the middle dot U+00B7 and
# the en/em dashes U+2013/U+2014 (written as \u escapes so the source stays ASCII).
_RESOLVED_RE = re.compile(
    r"RESOLVED\b[\s\u00b7:>|.\u2013\u2014\-]*\b(yes|no|ambiguous|annulled)\b",
    re.IGNORECASE,
)


class WebResolution(BaseModel):
    """One question outcome read off the public Metaculus page, with provenance.

    Attributes:
        question_id: The Metaculus question id (the id the snapshot store and
            experiment manifests key on).
        post_id: The post id used in the page URL.
        resolution: The outcome, normalised lowercase — one of ``"yes"``,
            ``"no"``, ``"ambiguous"``, ``"annulled"``.
        source_url: The exact page the outcome was read from.
        fetched_at: When the page was read (UTC).
        page_title: The page's leading title line, kept for a human sanity check.
    """

    question_id: int
    post_id: int
    resolution: str
    source_url: str
    fetched_at: datetime
    page_title: str


def question_url(post_id: int) -> str:
    """Return the public Metaculus question-page URL for a post id."""
    return _QUESTION_URL.format(post_id=post_id)


def parse_resolution(text: str) -> str | None:
    """Return the normalised outcome from a page's visible text, or ``None``.

    Searches for the first ``RESOLVED <outcome>`` banner (see :data:`_RESOLVED_RE`)
    and returns the outcome lower-cased. Returns ``None`` when no banner is found —
    the caller must treat that as "unknown", never as a guess.
    """
    match = _RESOLVED_RE.search(text)
    if match is None:
        return None
    return match.group(1).lower()


def _first_line(text: str) -> str:
    """Return the first non-empty, stripped line of ``text`` (the page title)."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def fetch_resolution(
    post_id: int,
    *,
    page_loader: Callable[[str], str],
    question_id: int | None = None,
) -> WebResolution | None:
    """Read one question's outcome from its public page, or ``None`` if unresolved.

    ``page_loader`` maps the page URL to its visible text — the injectable seam so
    tests never touch a browser (the CLI and the live check pass
    :func:`default_page_loader`). ``question_id`` is stored on the result; when
    omitted it mirrors ``post_id`` (only the one-off live check omits it — the CLI
    always supplies the real question id from the snapshot).

    Returns a provenance-stamped :class:`WebResolution` on success, or ``None``
    (logged) when the page shows no recognisable ``RESOLVED <outcome>`` banner.
    """
    url = question_url(post_id)
    text = page_loader(url)
    resolution = parse_resolution(text)
    if resolution is None:
        _log.info(
            "web-resolution: no RESOLVED banner found on %s (post %d); recording nothing",
            url,
            post_id,
        )
        return None
    return WebResolution(
        question_id=question_id if question_id is not None else post_id,
        post_id=post_id,
        resolution=resolution,
        source_url=url,
        fetched_at=datetime.now(UTC),
        page_title=_first_line(text),
    )


def default_page_loader(url: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str:
    """Load ``url`` in headless Chromium and return the visible main-content text.

    Uses a fresh sync Playwright browser per call (fine at this volume — a handful
    of pages a week), a realistic User-Agent to clear Cloudflare, and a
    network-idle-ish settle. Returns ``page.inner_text("main")`` when a ``<main>``
    element exists, else the ``<body>`` text. Requires the Chromium browser to be
    installed (``uv run playwright install chromium``).
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            context = browser.new_context(user_agent=_USER_AGENT)
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            # Best-effort settle: let late XHRs finish, but never fail the read
            # just because a long-poll keeps the network from going fully idle.
            try:
                page.wait_for_load_state("networkidle", timeout=timeout_ms)
            except Exception:
                # A settle timeout is not fatal — read whatever has rendered.
                _log.debug("web-resolution: networkidle wait timed out for %s", url)
            main = page.query_selector("main")
            if main is not None:
                return main.inner_text()
            body = page.query_selector("body")
            return body.inner_text() if body is not None else ""
        finally:
            browser.close()


class WebResolutionStore:
    """Append-only JSON Lines store of :class:`WebResolution` records.

    All records live in a single ``resolutions.jsonl`` file (unlike the dated
    snapshot files — web resolutions are low-volume). Appending never rewrites
    history; :meth:`latest_by_question` gives the newest reading per question.

    Args:
        dir: Directory holding ``resolutions.jsonl``. Created on first write.
    """

    _FILENAME = "resolutions.jsonl"

    def __init__(self, dir: Path) -> None:
        self._dir = Path(dir)

    @property
    def dir(self) -> Path:
        """The directory this store reads from and writes to."""
        return self._dir

    @property
    def path(self) -> Path:
        """The single JSONL file backing this store."""
        return self._dir / self._FILENAME

    def append(self, resolutions: Iterable[WebResolution]) -> int:
        """Append resolutions to ``resolutions.jsonl`` and return how many were written."""
        lines = [r.model_dump_json() for r in resolutions]
        if not lines:
            return 0
        self._dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line)
                fh.write("\n")
        return len(lines)

    def iter_all(self) -> Iterator[WebResolution]:
        """Yield every stored resolution, in file (append) order."""
        path = self.path
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                yield WebResolution.model_validate_json(line)

    def latest_by_question(self) -> dict[int, WebResolution]:
        """Return the newest resolution (by ``fetched_at``) per question id."""
        latest: dict[int, WebResolution] = {}
        for resolution in self.iter_all():
            current = latest.get(resolution.question_id)
            if current is None or resolution.fetched_at >= current.fetched_at:
                latest[resolution.question_id] = resolution
        return latest
