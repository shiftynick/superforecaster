"""On-disk storage for question snapshots, one JSON object per line.

Snapshots pile up over time, so we keep them in append-only JSON Lines files
named by UTC date (``2026-07-09.jsonl``). Appending never rewrites old data,
which makes the store cheap and crash-safe. Reading back is a simple scan.

Two convenience views sit on top of the raw records:

- ``latest_by_question`` — the most recent snapshot we hold for each question.
- ``unresolved_question_post_ids`` — post ids whose latest snapshot is not yet
  resolved, i.e. the ones worth re-checking for an outcome.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .models import QuestionSnapshot


@dataclass(frozen=True)
class StoreStats:
    """A quick numeric summary of everything in the store.

    Attributes:
        total_records: Every snapshot line ever written (duplicates included).
        unique_questions: Distinct question ids seen.
        resolved_questions: Questions whose latest snapshot has a resolution.
        first_scrape: Earliest ``scraped_at`` on record, or ``None`` if empty.
        last_scrape: Latest ``scraped_at`` on record, or ``None`` if empty.
    """

    total_records: int
    unique_questions: int
    resolved_questions: int
    first_scrape: datetime | None
    last_scrape: datetime | None


class SnapshotStore:
    """Append-only JSON Lines store of :class:`QuestionSnapshot` records.

    Args:
        dir: Directory holding the dated ``.jsonl`` files. Created on first write.
    """

    def __init__(self, dir: Path) -> None:
        self._dir = Path(dir)

    @property
    def dir(self) -> Path:
        """The directory this store reads from and writes to."""
        return self._dir

    def append(self, snapshots: Iterable[QuestionSnapshot]) -> int:
        """Append snapshots to their per-day files and return how many were written.

        Each snapshot lands in ``<dir>/<UTC-date-of-scraped_at>.jsonl``. Missing
        directories are created. Returns the number of records written.
        """
        written = 0
        # Group by file so we open each dated file at most once.
        handles: dict[Path, list[str]] = {}
        for snapshot in snapshots:
            path = self._path_for(snapshot.scraped_at)
            handles.setdefault(path, []).append(snapshot.model_dump_json())
            written += 1

        if written == 0:
            return 0

        self._dir.mkdir(parents=True, exist_ok=True)
        for path, lines in handles.items():
            with path.open("a", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(line)
                    fh.write("\n")
        return written

    def _path_for(self, scraped_at: datetime) -> Path:
        """Return the dated file path for a snapshot's UTC scrape date.

        Timezone-aware times are converted to UTC first so files line up with the
        UTC calendar day; naive times are assumed to already be UTC.
        """
        if scraped_at.tzinfo is not None:
            day = scraped_at.astimezone(UTC).date()
        else:
            day = scraped_at.date()
        return self._dir / f"{day.isoformat()}.jsonl"

    def iter_all(self) -> Iterator[QuestionSnapshot]:
        """Yield every stored snapshot, oldest file first, line by line."""
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    yield QuestionSnapshot.model_validate_json(line)

    def latest_by_question(self) -> dict[int, QuestionSnapshot]:
        """Return the newest snapshot (by ``scraped_at``) per question id."""
        latest: dict[int, QuestionSnapshot] = {}
        for snapshot in self.iter_all():
            current = latest.get(snapshot.question_id)
            if current is None or snapshot.scraped_at >= current.scraped_at:
                latest[snapshot.question_id] = snapshot
        return latest

    def unresolved_question_post_ids(self) -> list[int]:
        """Post ids whose latest snapshot is not yet resolved.

        These are the questions still worth polling for an outcome. Ordered by
        post id for stable output.
        """
        ids = {
            snapshot.post_id
            for snapshot in self.latest_by_question().values()
            if snapshot.status.lower() != "resolved"
        }
        return sorted(ids)

    def stats(self) -> StoreStats:
        """Compute a :class:`StoreStats` summary in a single pass over the data."""
        total = 0
        first: datetime | None = None
        last: datetime | None = None
        latest: dict[int, QuestionSnapshot] = {}

        for snapshot in self.iter_all():
            total += 1
            if first is None or snapshot.scraped_at < first:
                first = snapshot.scraped_at
            if last is None or snapshot.scraped_at > last:
                last = snapshot.scraped_at
            current = latest.get(snapshot.question_id)
            if current is None or snapshot.scraped_at >= current.scraped_at:
                latest[snapshot.question_id] = snapshot

        resolved = sum(1 for s in latest.values() if s.status.lower() == "resolved")
        return StoreStats(
            total_records=total,
            unique_questions=len(latest),
            resolved_questions=resolved,
            first_scrape=first,
            last_scrape=last,
        )
