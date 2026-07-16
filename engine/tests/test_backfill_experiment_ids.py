"""Tests for the experiment-id backfill script (scripts/backfill_experiment_ids.py)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from superforecaster.forecast_store import ForecastStore
from superforecaster.forecaster import ForecastRecord

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill_experiment_ids.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("backfill_experiment_ids", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registering the module lets dataclasses resolve its (PEP 563) string
    # annotations, which look the module up in sys.modules by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backfill_mod = _load_script()

# Synthetic pinned sets: q1/q2 pinned by 001, q2/q3 pinned by 002 (q2 overlaps).
_MANIFEST_001 = (
    'id = "001"\ntitle = "One"\nstatus = "frozen"\nfrozen = 2026-07-10\n'
    "budget_usd = 1.0\nasknews_credit_budget = 1\nquestion_ids = [1, 2]\n"
    '[arms.A]\nmethod = "checklist-a"\n'
)
_MANIFEST_002 = (
    'id = "002"\ntitle = "Two"\nstatus = "frozen"\nfrozen = 2026-07-12\n'
    "budget_usd = 1.0\nasknews_credit_budget = 1\nquestion_ids = [2, 3]\n"
    '[arms.C]\nmethod = "pyramid-c"\n'
)


def _write_manifests(tmp_path: Path) -> Path:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "001-one.toml").write_text(_MANIFEST_001, encoding="utf-8")
    (experiments / "002-two.toml").write_text(_MANIFEST_002, encoding="utf-8")
    return experiments


def _record(
    question_id: int,
    *,
    method: str,
    created_at: datetime,
    experiment_id: str | None = None,
) -> ForecastRecord:
    return ForecastRecord(
        created_at=created_at,
        question_id=question_id,
        post_id=question_id + 100,
        title=f"Q{question_id}",
        probability=0.5,
        raw_probability=0.5,
        confidence="medium",
        rationale="because",
        reference_class="similar cases",
        model="z-ai/glm-5.2",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.01,
        method=method,
        experiment_id=experiment_id,
    )


JULY_9 = datetime(2026, 7, 9, 12, 0, tzinfo=UTC)
JULY_10 = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
JULY_11 = datetime(2026, 7, 11, 23, 59, tzinfo=UTC)
JULY_12 = datetime(2026, 7, 12, 0, 30, tzinfo=UTC)
JULY_16 = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)
JULY_16_LATER = datetime(2026, 7, 16, 9, 0, tzinfo=UTC)


def test_backfill_rules(tmp_path: Path) -> None:
    experiments = _write_manifests(tmp_path)
    fc_dir = tmp_path / "forecasts"
    store = ForecastStore(fc_dir)
    store.append(
        [
            # Rule 1 hits: 001-pinned qids, eligible methods, in [07-10, 07-12).
            _record(1, method="checklist-a", created_at=JULY_10),
            _record(1, method="ensemble-b", created_at=JULY_11),
            _record(2, method="ensemble-b-member", created_at=JULY_10),
            # Rule 2 hits: 002-pinned qids, eligible methods, on/after 07-12.
            _record(3, method="pyramid-c", created_at=JULY_16),
            _record(2, method="checklist-a", created_at=JULY_12),  # overlap qid -> 002
            # No rule: pre-freeze; 001 method after its window on a non-002 qid;
            # ineligible method; non-pinned question.
            _record(1, method="checklist-a", created_at=JULY_9),
            _record(1, method="checklist-a", created_at=JULY_16),  # q1 not in 002
            _record(3, method="something-else", created_at=JULY_16),
            _record(99, method="pyramid-c", created_at=JULY_16),
            # Already stamped: untouched even though rule 2 would say "002".
            _record(3, method="pyramid-c", created_at=JULY_16_LATER, experiment_id="junk"),
        ]
    )

    counts = backfill_mod.backfill(fc_dir, experiments)
    assert counts.stamped_001 == 3
    assert counts.stamped_002 == 2
    assert counts.left_unstamped == 4
    assert counts.already_stamped == 1

    loaded = list(ForecastStore(fc_dir).iter_all())  # still parses after rewrite
    by_key = {(r.question_id, r.method, r.created_at): r.experiment_id for r in loaded}
    assert by_key[(1, "checklist-a", JULY_10)] == "001"
    assert by_key[(1, "ensemble-b", JULY_11)] == "001"
    assert by_key[(2, "ensemble-b-member", JULY_10)] == "001"
    assert by_key[(3, "pyramid-c", JULY_16)] == "002"
    assert by_key[(2, "checklist-a", JULY_12)] == "002"
    assert by_key[(1, "checklist-a", JULY_9)] is None
    assert by_key[(1, "checklist-a", JULY_16)] is None
    assert by_key[(3, "something-else", JULY_16)] is None
    assert by_key[(99, "pyramid-c", JULY_16)] is None
    stamped_junk = [r for r in loaded if r.experiment_id == "junk"]
    assert len(stamped_junk) == 1  # pre-existing stamp preserved


def test_backfill_is_idempotent_and_makes_bak(tmp_path: Path) -> None:
    experiments = _write_manifests(tmp_path)
    fc_dir = tmp_path / "forecasts"
    ForecastStore(fc_dir).append(
        [
            _record(1, method="checklist-a", created_at=JULY_10),
            _record(1, method="checklist-a", created_at=JULY_9),  # unchanged
        ]
    )
    original = (fc_dir / "2026-07-10.jsonl").read_text(encoding="utf-8")

    first = backfill_mod.backfill(fc_dir, experiments)
    assert first.stamped_001 == 1
    assert first.files_rewritten == ["2026-07-10.jsonl"]
    # A .bak of the pre-backfill original sits next to the rewritten file; the
    # untouched file got neither a rewrite nor a backup.
    assert (fc_dir / "2026-07-10.jsonl.bak").read_text(encoding="utf-8") == original
    assert not (fc_dir / "2026-07-09.jsonl.bak").exists()

    after_first = (fc_dir / "2026-07-10.jsonl").read_text(encoding="utf-8")
    second = backfill_mod.backfill(fc_dir, experiments)
    assert second.changed == 0
    assert second.already_stamped == 1
    assert second.files_rewritten == []
    assert (fc_dir / "2026-07-10.jsonl").read_text(encoding="utf-8") == after_first
    # The .bak still holds the ORIGINAL (a re-run never clobbers it).
    assert (fc_dir / "2026-07-10.jsonl.bak").read_text(encoding="utf-8") == original


def test_target_experiment_id_boundaries() -> None:
    """The pure rule function honours the window edges exactly."""
    from datetime import date

    pinned_001 = frozenset({1})
    pinned_002 = frozenset({1})
    f = backfill_mod.target_experiment_id
    # 001 window is [2026-07-10, 2026-07-12); 002 starts 2026-07-12.
    assert f("checklist-a", 1, date(2026, 7, 9), pinned_001, pinned_002) is None
    assert f("checklist-a", 1, date(2026, 7, 10), pinned_001, pinned_002) == "001"
    assert f("checklist-a", 1, date(2026, 7, 11), pinned_001, pinned_002) == "001"
    assert f("checklist-a", 1, date(2026, 7, 12), pinned_001, pinned_002) == "002"
    # pyramid-c is 002-only: never stamped 001 even inside 001's window.
    assert f("pyramid-c", 1, date(2026, 7, 11), pinned_001, pinned_002) is None
    assert f("pyramid-c", 1, date(2026, 7, 12), pinned_001, pinned_002) == "002"
