"""Realistic Metaculus post fixtures shared across tests.

These dicts mimic the shapes documented in the API recon: a normal open binary
post with a community prediction, one missing its aggregations, a question
group, and a resolved post. They are deliberately partial — the extraction code
must tolerate missing keys.
"""

from __future__ import annotations

from typing import Any


def open_binary_post(post_id: int = 101, question_id: int = 201) -> dict[str, Any]:
    """A normal open binary question with a crowd prediction."""
    return {
        "id": post_id,
        "title": "Will it rain tomorrow?",
        "status": "open",
        "nr_forecasters": 42,
        "group_of_questions": None,
        "question": {
            "id": question_id,
            "type": "binary",
            "title": "Will it rain tomorrow?",
            "description": "Resolves YES if measurable rain falls.",
            "resolution_criteria": "Any measurable precipitation.",
            "fine_print": "Trace amounts do not count.",
            "open_time": "2026-07-01T00:00:00Z",
            "scheduled_close_time": "2026-08-01T00:00:00Z",
            "scheduled_resolve_time": "2026-08-02T00:00:00Z",
            "actual_resolve_time": None,
            "status": "open",
            "resolution": None,
            "aggregations": {
                "recency_weighted": {
                    "latest": {
                        "centers": [0.63],
                        "forecaster_count": 37,
                    }
                }
            },
        },
    }


def open_binary_post_no_cp(post_id: int = 102, question_id: int = 202) -> dict[str, Any]:
    """An open binary question with no aggregations block at all."""
    return {
        "id": post_id,
        "title": "Will the bridge open on time?",
        "status": "open",
        "nr_forecasters": None,
        "question": {
            "id": question_id,
            "type": "binary",
            "title": "Will the bridge open on time?",
            "description": "",
            "open_time": None,
            "scheduled_close_time": None,
            "scheduled_resolve_time": None,
            "actual_resolve_time": None,
            "status": "open",
            "resolution": None,
        },
    }


def group_post(post_id: int = 103) -> dict[str, Any]:
    """A question group, which the scraper must skip."""
    return {
        "id": post_id,
        "title": "Grouped questions",
        "status": "open",
        "group_of_questions": {"questions": [{"id": 1}, {"id": 2}]},
        "question": None,
    }


def resolved_yes_post(post_id: int = 104, question_id: int = 204) -> dict[str, Any]:
    """A binary question that has resolved YES."""
    return {
        "id": post_id,
        "title": "Did the launch happen in June?",
        "status": "resolved",
        "nr_forecasters": 88,
        "question": {
            "id": question_id,
            "type": "binary",
            "title": "Did the launch happen in June?",
            "description": "Resolves YES if launched by end of June.",
            "open_time": "2026-05-01T00:00:00Z",
            "scheduled_close_time": "2026-06-01T00:00:00Z",
            "scheduled_resolve_time": "2026-07-01T00:00:00Z",
            "actual_resolve_time": "2026-06-15T00:00:00Z",
            "status": "resolved",
            "resolution": "yes",
            "aggregations": {
                "recency_weighted": {"latest": {"centers": [0.9], "forecaster_count": 60}}
            },
        },
    }
