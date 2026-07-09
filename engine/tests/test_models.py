"""Tests for QuestionSnapshot.from_api_post extraction and skipping."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from superforecaster.models import QuestionSnapshot, SkipPost

from . import fixtures

SCRAPED_AT = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)


def test_extracts_full_open_post() -> None:
    snap = QuestionSnapshot.from_api_post(fixtures.open_binary_post(), SCRAPED_AT)
    assert snap.post_id == 101
    assert snap.question_id == 201
    assert snap.title == "Will it rain tomorrow?"
    assert snap.community_prediction == 0.63
    assert snap.forecaster_count == 37
    assert snap.nr_forecasters == 42
    assert snap.status == "open"
    assert snap.resolution is None
    assert snap.open_time == datetime(2026, 7, 1, tzinfo=UTC)
    assert snap.actual_resolve_time is None


def test_tolerates_missing_cp_and_null_times() -> None:
    snap = QuestionSnapshot.from_api_post(fixtures.open_binary_post_no_cp(), SCRAPED_AT)
    assert snap.community_prediction is None
    assert snap.forecaster_count is None
    assert snap.nr_forecasters is None
    assert snap.open_time is None
    assert snap.description == ""
    assert snap.fine_print == ""


def test_resolved_post_preserves_raw_resolution() -> None:
    snap = QuestionSnapshot.from_api_post(fixtures.resolved_yes_post(), SCRAPED_AT)
    assert snap.status == "resolved"
    assert snap.resolution == "yes"
    assert snap.actual_resolve_time == datetime(2026, 6, 15, tzinfo=UTC)


def test_group_post_is_skipped() -> None:
    with pytest.raises(SkipPost):
        QuestionSnapshot.from_api_post(fixtures.group_post(), SCRAPED_AT)


def test_non_binary_is_skipped() -> None:
    post = fixtures.open_binary_post()
    post["question"]["type"] = "numeric"
    with pytest.raises(SkipPost):
        QuestionSnapshot.from_api_post(post, SCRAPED_AT)
