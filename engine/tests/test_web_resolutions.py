"""Tests for the web-resolution reader: parser, fetch seam, and store.

No browser and no network are touched: the parser runs on realistic page-text
fixtures (shaped like ``page.inner_text("main")`` on a resolved Metaculus page),
and :func:`fetch_resolution` is driven through an injected fake page loader.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from superforecaster.web_resolutions import (
    WebResolution,
    WebResolutionStore,
    fetch_resolution,
    parse_resolution,
    question_url,
)

# A resolved-NO page, shaped like the real World Cup question (post 41090): the
# title first, a category chip, the RESOLVED banner with the outcome on the next
# line, then community/comment noise — including the words "yes" and "no" inside
# comments, which must NOT be mistaken for the banner.
NO_PAGE = """\
Will Spain win the 2026 FIFA World Cup?
Sports & Entertainment
RESOLVED
No
Community Prediction: 22%
Comments (14)
alice: I really thought the answer would be yes.
bob: No surprise here, honestly.
"""

# A resolved-YES page with the banner rendered inline with a middle-dot separator.
YES_PAGE = """\
Will NASA launch Artemis III before 2027?
Space
RESOLVED · Yes
Community Prediction: 61%
Comments (3)
carol: called it, no doubt about it.
"""

# An annulled question (banner uses an em dash separator).
ANNULLED_PAGE = """\
Will the treaty be ratified by June?
Politics
RESOLVED — Annulled
This question was annulled because the underlying event was cancelled.
"""

# An ambiguous resolution.
AMBIGUOUS_PAGE = """\
Did the company ship the product on time?
Business
Resolved: Ambiguous
The resolution source became unavailable.
"""

# A still-open page: no RESOLVED banner anywhere -> parser must return None.
OPEN_PAGE = """\
Will it rain in Seattle tomorrow?
Weather
OPEN
Community Prediction: 47%
Comments (2)
dave: my guess is yes.
erin: probably not.
"""


def test_parse_resolution_no_from_realistic_page() -> None:
    assert parse_resolution(NO_PAGE) == "no"


def test_parse_resolution_yes_inline_middot() -> None:
    assert parse_resolution(YES_PAGE) == "yes"


def test_parse_resolution_annulled_emdash() -> None:
    assert parse_resolution(ANNULLED_PAGE) == "annulled"


def test_parse_resolution_ambiguous_colon() -> None:
    assert parse_resolution(AMBIGUOUS_PAGE) == "ambiguous"


def test_parse_resolution_open_page_returns_none() -> None:
    assert parse_resolution(OPEN_PAGE) is None


def test_parse_resolution_ignores_comment_yes_no() -> None:
    # The banner outcome ("no") wins even though "yes" appears earlier in a
    # comment lower in the text — the marker anchors the match.
    assert parse_resolution(NO_PAGE) == "no"


def test_fetch_resolution_success_records_provenance() -> None:
    seen: list[str] = []

    def loader(url: str) -> str:
        seen.append(url)
        return NO_PAGE

    before = datetime.now(UTC)
    result = fetch_resolution(41090, page_loader=loader, question_id=555)
    after = datetime.now(UTC)

    assert result is not None
    assert result.resolution == "no"
    assert result.post_id == 41090
    assert result.question_id == 555
    assert result.source_url == question_url(41090)
    assert result.page_title == "Will Spain win the 2026 FIFA World Cup?"
    assert before <= result.fetched_at <= after
    # The loader was called exactly once, with the post's page URL.
    assert seen == [question_url(41090)]


def test_fetch_resolution_defaults_question_id_to_post_id() -> None:
    result = fetch_resolution(41090, page_loader=lambda _url: YES_PAGE)
    assert result is not None
    assert result.question_id == 41090  # no question_id supplied -> mirrors post_id


def test_fetch_resolution_returns_none_on_open_page() -> None:
    result = fetch_resolution(999, page_loader=lambda _url: OPEN_PAGE, question_id=1)
    assert result is None


def _res(question_id: int, *, fetched_at: datetime, resolution: str = "no") -> WebResolution:
    return WebResolution(
        question_id=question_id,
        post_id=question_id + 10_000,
        resolution=resolution,
        source_url=question_url(question_id + 10_000),
        fetched_at=fetched_at,
        page_title=f"Q{question_id}",
    )


def test_store_roundtrip_and_latest_by_question(tmp_path: Path) -> None:
    base = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    store = WebResolutionStore(tmp_path)

    older = _res(1, fetched_at=base, resolution="yes")
    newer = _res(1, fetched_at=base + timedelta(hours=1), resolution="no")
    other = _res(2, fetched_at=base, resolution="annulled")

    written = store.append([older, newer, other])
    assert written == 3
    # Single-file JSONL store.
    assert (tmp_path / "resolutions.jsonl").exists()

    got = list(store.iter_all())
    assert len(got) == 3

    latest = store.latest_by_question()
    assert set(latest) == {1, 2}
    assert latest[1].resolution == "no"  # newest fetched_at wins for q1
    assert latest[2].resolution == "annulled"


def test_store_append_empty_is_noop(tmp_path: Path) -> None:
    store = WebResolutionStore(tmp_path)
    assert store.append([]) == 0
    assert list(store.iter_all()) == []


def test_store_missing_file_iterates_empty(tmp_path: Path) -> None:
    store = WebResolutionStore(tmp_path / "nope")
    assert list(store.iter_all()) == []
    assert store.latest_by_question() == {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("RESOLVED\n\n\nYes", "yes"),
        ("resolved no", "no"),
        ("RESOLVED:   Ambiguous", "ambiguous"),
        ("RESOLVED | Annulled", "annulled"),
        ("nothing here", None),
        ("RESOLVED maybe someday", None),  # 'maybe' is not an outcome word
    ],
)
def test_parse_resolution_separator_variants(text: str, expected: str | None) -> None:
    assert parse_resolution(text) == expected
