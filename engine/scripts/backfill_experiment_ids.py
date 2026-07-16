"""One-time backfill: stamp stored forecast records with their experiment id.

Context (scoring integrity, 2026-07-16): experiments 001 and 002 share pinned
questions, and arm matching used to be method/model + created-on-or-after-freeze
only — so 002's later records silently displaced 001's frozen forecasts in 001's
scoring (compute_status takes the latest matching record per arm per question).
Records now carry ``experiment_id``; this script stamps the historical records
so each experiment's scoring sees only its own runs.

Rules, applied in order to each record (UTC calendar date of ``created_at``):

1. method in {checklist-a, ensemble-b, ensemble-b-member} AND question pinned by
   experiment 001 AND 2026-07-10 <= date < 2026-07-12  ->  experiment_id "001"
2. question pinned by experiment 002 AND date >= 2026-07-12 AND method in
   {checklist-a, ensemble-b, ensemble-b-member, pyramid-c}  ->  experiment_id "002"
3. otherwise  ->  left unstamped (pre-freeze B2/E1/P1-preview records are
   intentionally not attributed to any experiment)

A record that is already stamped is never touched, so re-running is a no-op
(idempotent). Files are rewritten in place: a ``.bak`` copy of the original is
made first (never overwritten on later runs), the new content goes to a temp
file, and an atomic replace swaps it in. Unchanged lines are copied through
byte-for-byte — only stamped lines are re-serialized.

Run from ``engine/``:  uv run python scripts/backfill_experiment_ids.py
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from superforecaster.experiments import ExperimentManifest, load_manifests

# The repo root, derived from this file (engine/scripts/ -> engine/ -> root).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORECAST_DIR = REPO_ROOT / "data" / "forecasts"
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "docs" / "experiments"

# Rule windows (UTC calendar dates) and eligible methods, frozen per the docs.
_RULE_001_START = date(2026, 7, 10)
_RULE_001_END = date(2026, 7, 12)  # exclusive
_RULE_002_START = date(2026, 7, 12)  # inclusive
_RULE_001_METHODS = frozenset({"checklist-a", "ensemble-b", "ensemble-b-member"})
_RULE_002_METHODS = _RULE_001_METHODS | {"pyramid-c"}


@dataclass
class BackfillCounts:
    """Per-rule tallies for the summary line (and the idempotency check)."""

    stamped_001: int = 0
    stamped_002: int = 0
    left_unstamped: int = 0
    already_stamped: int = 0
    files_rewritten: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        """How many records this run actually modified."""
        return self.stamped_001 + self.stamped_002


def _utc_date(created_at_raw: str) -> date:
    """The UTC calendar date of a record's ISO-8601 ``created_at`` string."""
    moment = datetime.fromisoformat(created_at_raw)
    return moment.astimezone(UTC).date() if moment.tzinfo is not None else moment.date()


def target_experiment_id(
    method: str,
    question_id: int,
    created_day: date,
    pinned_001: frozenset[int],
    pinned_002: frozenset[int],
) -> str | None:
    """Apply the backfill rules, in order; ``None`` means leave unstamped."""
    if (
        method in _RULE_001_METHODS
        and question_id in pinned_001
        and _RULE_001_START <= created_day < _RULE_001_END
    ):
        return "001"
    if question_id in pinned_002 and created_day >= _RULE_002_START and method in _RULE_002_METHODS:
        return "002"
    return None


def _process_line(
    line: str,
    pinned_001: frozenset[int],
    pinned_002: frozenset[int],
    counts: BackfillCounts,
) -> str:
    """Return the (possibly re-stamped) JSONL line; count what happened."""
    record = json.loads(line)
    if record.get("experiment_id") is not None:
        counts.already_stamped += 1
        return line
    target = target_experiment_id(
        str(record.get("method", "checklist-a")),
        int(record["question_id"]),
        _utc_date(str(record["created_at"])),
        pinned_001,
        pinned_002,
    )
    if target is None:
        counts.left_unstamped += 1
        return line
    record["experiment_id"] = target
    if target == "001":
        counts.stamped_001 += 1
    else:
        counts.stamped_002 += 1
    # Compact separators match pydantic's model_dump_json output style.
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False)


def _rewrite_file(
    path: Path,
    pinned_001: frozenset[int],
    pinned_002: frozenset[int],
    counts: BackfillCounts,
) -> None:
    """Backfill one JSONL file in place (bak copy, temp file, atomic replace)."""
    original_lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = [
        _process_line(line, pinned_001, pinned_002, counts) if line.strip() else line
        for line in original_lines
    ]
    if new_lines == original_lines:
        return  # nothing to change in this file; leave it (and any .bak) alone

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():  # never clobber the true pre-backfill original
        shutil.copy2(path, backup)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    counts.files_rewritten.append(path.name)


def _pinned(manifests: list[ExperimentManifest], experiment_id: str) -> frozenset[int]:
    """The pinned question ids of the manifest with the given id."""
    for manifest in manifests:
        if manifest.id == experiment_id:
            return manifest.pinned
    raise SystemExit(f"backfill: no manifest with id {experiment_id!r} found")


def backfill(
    forecast_dir: Path = DEFAULT_FORECAST_DIR,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
) -> BackfillCounts:
    """Run the backfill over every ``*.jsonl`` file; return the tallies."""
    manifests = load_manifests(experiments_dir)
    pinned_001 = _pinned(manifests, "001")
    pinned_002 = _pinned(manifests, "002")

    counts = BackfillCounts()
    for path in sorted(forecast_dir.glob("*.jsonl")):
        _rewrite_file(path, pinned_001, pinned_002, counts)
    return counts


def main() -> None:
    """Entry point: backfill the real store and print the per-rule summary."""
    counts = backfill()
    print(f"backfill: forecast dir {DEFAULT_FORECAST_DIR}")
    print(f"  stamped experiment_id=001: {counts.stamped_001}")
    print(f"  stamped experiment_id=002: {counts.stamped_002}")
    print(f"  left unstamped (no rule matched): {counts.left_unstamped}")
    print(f"  already stamped (untouched): {counts.already_stamped}")
    rewritten = ", ".join(counts.files_rewritten) if counts.files_rewritten else "(none)"
    print(f"  files rewritten: {rewritten}")


if __name__ == "__main__":
    main()
