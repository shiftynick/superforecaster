"""On-disk storage for forecasts, one JSON object per line.

A direct sibling of :class:`~superforecaster.store.SnapshotStore`: forecasts are
appended to date-named JSON Lines files (``2026-07-09.jsonl``) so the history is
never rewritten and is trivial to scan. Every forecast — dry run or live — is
kept with its full trace (rationale, model, tokens, cost) so later postmortems
and backtests can replay exactly what was thought and what it cost.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .forecaster import ForecastRecord

# Default location: a sibling of data/snapshots, so the two stores sit together.
DEFAULT_FORECAST_DIR = "./data/forecasts"


@dataclass(frozen=True)
class ForecastStats:
    """A quick numeric summary of the stored forecasts.

    Attributes:
        total: Every forecast record written.
        dry_run: How many were dry runs (never submitted anywhere).
        submitted: How many carry a ``submitted_at`` timestamp.
        total_cost_usd: Sum of known per-call costs; forecasts whose model was
            unpriced contribute nothing (their cost is recorded as unknown).
    """

    total: int
    dry_run: int
    submitted: int
    total_cost_usd: float


class ForecastStore:
    """Append-only JSON Lines store of :class:`ForecastRecord` records.

    Args:
        dir: Directory holding the dated ``.jsonl`` files. Created on first write.
    """

    def __init__(self, dir: Path) -> None:
        self._dir = Path(dir)

    @property
    def dir(self) -> Path:
        """The directory this store reads from and writes to."""
        return self._dir

    def append(self, records: Iterable[ForecastRecord]) -> int:
        """Append forecasts to their per-day files; return how many were written.

        Each record lands in ``<dir>/<UTC-date-of-created_at>.jsonl``. Missing
        directories are created. Returns the number of records written.
        """
        written = 0
        handles: dict[Path, list[str]] = {}
        for record in records:
            path = self._path_for(record.created_at)
            handles.setdefault(path, []).append(record.model_dump_json())
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

    def _path_for(self, created_at: datetime) -> Path:
        """Return the dated file path for a record's UTC creation date."""
        if created_at.tzinfo is not None:
            day = created_at.astimezone(UTC).date()
        else:
            day = created_at.date()
        return self._dir / f"{day.isoformat()}.jsonl"

    def iter_all(self) -> Iterator[ForecastRecord]:
        """Yield every stored forecast, oldest file first, line by line."""
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    yield ForecastRecord.model_validate_json(line)

    def stats(self) -> ForecastStats:
        """Compute a :class:`ForecastStats` summary in a single pass."""
        total = 0
        dry_run = 0
        submitted = 0
        total_cost = 0.0
        for record in self.iter_all():
            total += 1
            if record.dry_run:
                dry_run += 1
            if record.submitted_at is not None:
                submitted += 1
            if record.cost_usd is not None:
                total_cost += record.cost_usd
        return ForecastStats(
            total=total,
            dry_run=dry_run,
            submitted=submitted,
            total_cost_usd=total_cost,
        )
