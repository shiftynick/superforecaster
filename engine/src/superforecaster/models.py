"""Data shapes for one scraped Metaculus question.

A *snapshot* is a photo of a single forecasting question at one moment in time:
its text, when it opens and resolves, its current status, and — if available —
the crowd's current probability estimate. We store many snapshots of the same
question over time so we can later see how the crowd moved and what the question
finally resolved to.

The tricky part is that Metaculus's API returns big, loosely-shaped JSON where
fields we want may be missing or null. ``QuestionSnapshot.from_api_post`` does
all the defensive digging so the rest of the code can work with a clean object.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class SkipPost(Exception):
    """Raised when a Metaculus post is not a plain binary question we can snapshot.

    Two cases trigger this: the post bundles several sub-questions (a "question
    group"), or its question type is not ``binary``. The scraper catches this and
    counts the post as skipped rather than crashing.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _parse_dt(value: Any) -> datetime | None:
    """Turn a Metaculus timestamp string into a ``datetime``, or ``None``.

    Metaculus sends ISO-8601 strings (sometimes ending in ``Z``). Anything that
    is missing, null, or unparseable becomes ``None`` so a single odd field can
    never sink a whole snapshot.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


class QuestionSnapshot(BaseModel):
    """One binary Metaculus question captured at a single moment.

    A "binary" question is a yes/no question. ``community_prediction`` is the
    crowd's current probability that the answer is *yes*, as a number in [0, 1];
    it may be ``None`` before anyone has forecast. ``resolution`` is ``None``
    while the question is still open and becomes a string such as ``"yes"`` /
    ``"no"`` / ``"ambiguous"`` / ``"annulled"`` once it settles.
    """

    scraped_at: datetime
    post_id: int
    question_id: int
    title: str
    description: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    open_time: datetime | None = None
    scheduled_close_time: datetime | None = None
    scheduled_resolve_time: datetime | None = None
    actual_resolve_time: datetime | None = None
    status: str
    resolution: str | None = None
    community_prediction: float | None = None
    forecaster_count: int | None = None
    nr_forecasters: int | None = None

    @classmethod
    def from_api_post(cls, post: dict[str, Any], scraped_at: datetime) -> QuestionSnapshot:
        """Build a snapshot from one raw Metaculus post dict.

        Raises :class:`SkipPost` when the post is a question group or is not a
        binary question. Every other field is extracted defensively: missing text
        becomes ``""``, missing numbers/times become ``None``.
        """
        if post.get("group_of_questions"):
            raise SkipPost("question group")

        question = post.get("question") or {}
        if question.get("type") != "binary":
            raise SkipPost(f"non-binary question type: {question.get('type')!r}")

        community_prediction = _extract_community_prediction(question)
        forecaster_count = _extract_forecaster_count(question)

        return cls(
            scraped_at=scraped_at,
            post_id=int(post["id"]),
            question_id=int(question["id"]),
            title=str(post.get("title") or question.get("title") or ""),
            description=str(question.get("description") or ""),
            resolution_criteria=str(question.get("resolution_criteria") or ""),
            fine_print=str(question.get("fine_print") or ""),
            open_time=_parse_dt(question.get("open_time")),
            scheduled_close_time=_parse_dt(question.get("scheduled_close_time")),
            scheduled_resolve_time=_parse_dt(question.get("scheduled_resolve_time")),
            actual_resolve_time=_parse_dt(question.get("actual_resolve_time")),
            # question.status is the verified open/closed/resolved lifecycle field;
            # post-level status is only a fallback (its semantics are unverified).
            status=str(question.get("status") or post.get("status") or "unknown"),
            resolution=_extract_resolution(question),
            community_prediction=community_prediction,
            forecaster_count=forecaster_count,
            nr_forecasters=_coerce_int(post.get("nr_forecasters")),
        )


def _extract_community_prediction(question: dict[str, Any]) -> float | None:
    """Dig out the crowd's latest probability, tolerating missing structure.

    Path: ``aggregations.recency_weighted.latest.centers[0]``. Any gap along the
    way yields ``None``.
    """
    latest = _latest_aggregation(question)
    if latest is None:
        return None
    centers = latest.get("centers")
    if not isinstance(centers, list) or not centers:
        return None
    try:
        return float(centers[0])
    except (TypeError, ValueError):
        return None


def _extract_forecaster_count(question: dict[str, Any]) -> int | None:
    """Read ``forecaster_count`` off the latest aggregation, if present."""
    latest = _latest_aggregation(question)
    if latest is None:
        return None
    return _coerce_int(latest.get("forecaster_count"))


def _latest_aggregation(question: dict[str, Any]) -> dict[str, Any] | None:
    """Return ``aggregations.recency_weighted.latest`` or ``None`` if absent."""
    aggregations = question.get("aggregations")
    if not isinstance(aggregations, dict):
        return None
    recency = aggregations.get("recency_weighted")
    if not isinstance(recency, dict):
        return None
    latest = recency.get("latest")
    if not isinstance(latest, dict):
        return None
    return latest


def _extract_resolution(question: dict[str, Any]) -> str | None:
    """Return the raw resolution string, preserved as-is, or ``None`` if open."""
    resolution = question.get("resolution")
    if resolution is None:
        return None
    return str(resolution)


def _coerce_int(value: Any) -> int | None:
    """Best-effort conversion to ``int``; ``None`` if not a whole number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
