"""CLI tests for `sweep-resolutions` (no browser, no network).

Every pinned experiment question is read straight from an injected fake page
loader; the Metaculus API is never touched by this command at all (unlike
`update-resolutions --from-web`, which mops up gaps left by an API pass).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from superforecaster import cli
from superforecaster.models import QuestionSnapshot
from superforecaster.store import SnapshotStore
from superforecaster.web_resolutions import WebResolution, WebResolutionStore

runner = CliRunner()

ON = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)

NO_PAGE = "Q title\nCategory\nRESOLVED\nNo\nComments\n"
YES_PAGE = "Q title\nCategory\nRESOLVED\nYes\nComments\n"
OPEN_PAGE = "Q title\nCategory\nOPEN\nComments\n"


def _snapshot(question_id: int, post_id: int, *, status: str = "open") -> QuestionSnapshot:
    return QuestionSnapshot(
        scraped_at=ON,
        post_id=post_id,
        question_id=question_id,
        title=f"Q{question_id}",
        status=status,
        resolution=None,
    )


def _write_manifest(experiments_dir: Path, question_ids: list[int]) -> None:
    experiments_dir.mkdir(parents=True, exist_ok=True)
    ids = ", ".join(str(q) for q in question_ids)
    (experiments_dir / "001-test.toml").write_text(
        f"""
id = "001"
title = "Test experiment"
status = "frozen"
frozen = 2026-07-01
budget_usd = 10.0
asknews_credit_budget = 100
question_ids = [{ids}]

[arms.A]
method = "checklist-a"
""",
        encoding="utf-8",
    )


def _write_doc_only_ids(experiments_dir: Path, question_ids: list[int]) -> None:
    experiments_dir.mkdir(parents=True, exist_ok=True)
    lines = ", ".join(str(q) for q in question_ids)
    (experiments_dir / "005-test.md").write_text(
        f"""# 005 — doc-only test experiment

## Question set

- Question ids ({len(question_ids)}, pinned 2026-07-17, soonest-resolving first):
  {lines}.
""",
        encoding="utf-8",
    )


def _run(monkeypatch: pytest.MonkeyPatch, *, extra: list[str]) -> Any:
    return runner.invoke(cli.app, ["sweep-resolutions", *extra])


def test_finds_and_stores_resolved_pinned_questions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "snapshots"
    experiments_dir = tmp_path / "experiments"
    _write_manifest(experiments_dir, [40201, 40202])
    _write_doc_only_ids(experiments_dir, [40203])

    SnapshotStore(data_dir).append(
        [
            _snapshot(40201, 101, status="resolved"),
            _snapshot(40202, 102, status="resolved"),
            _snapshot(40203, 103, status="resolved"),
        ]
    )

    def fake_loader(url: str) -> str:
        if url.endswith("/101/"):
            return NO_PAGE
        if url.endswith("/102/"):
            return YES_PAGE
        return OPEN_PAGE  # post 103 not yet resolved on the page

    sleeps: list[float] = []
    monkeypatch.setattr(cli, "_build_web_page_loader", lambda: fake_loader)
    monkeypatch.setattr(cli, "_web_sleep", lambda seconds: sleeps.append(seconds))

    result = _run(
        monkeypatch,
        extra=["--data-dir", str(data_dir), "--experiments-dir", str(experiments_dir)],
    )
    assert result.exit_code == 0, result.output

    assert "pinned=3 mapped=3 already=0 checked=3 found=2" in result.output
    assert "q40201: RESOLVED 'no'" in result.output
    assert "q40202: RESOLVED 'yes'" in result.output
    assert len(sleeps) == 2  # throttle between each of the 3 checked pages

    web_store = WebResolutionStore(tmp_path / "web-resolutions")
    recorded = {r.question_id: r for r in web_store.iter_all()}
    assert recorded[40201].resolution == "no"
    assert recorded[40202].resolution == "yes"
    assert 40203 not in recorded


def test_skips_already_resolved_questions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "snapshots"
    experiments_dir = tmp_path / "experiments"
    _write_manifest(experiments_dir, [301, 302])

    SnapshotStore(data_dir).append(
        [_snapshot(301, 111, status="resolved"), _snapshot(302, 112, status="resolved")]
    )
    WebResolutionStore(tmp_path / "web-resolutions").append(
        [
            WebResolution(
                question_id=301,
                post_id=111,
                resolution="no",
                source_url="https://www.metaculus.com/questions/111/",
                fetched_at=ON,
                page_title="Q301",
            )
        ]
    )

    load_calls: list[str] = []

    def fake_loader(url: str) -> str:
        load_calls.append(url)
        return YES_PAGE

    monkeypatch.setattr(cli, "_build_web_page_loader", lambda: fake_loader)
    monkeypatch.setattr(cli, "_web_sleep", lambda seconds: None)

    result = _run(
        monkeypatch,
        extra=["--data-dir", str(data_dir), "--experiments-dir", str(experiments_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "pinned=2 mapped=2 already=1 checked=1 found=1" in result.output
    assert load_calls == ["https://www.metaculus.com/questions/112/"]


def test_respects_max_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "snapshots"
    experiments_dir = tmp_path / "experiments"
    qids = list(range(400, 405))  # 5 pinned questions
    _write_manifest(experiments_dir, qids)

    SnapshotStore(data_dir).append(
        [_snapshot(qid, 1000 + i, status="resolved") for i, qid in enumerate(qids)]
    )

    load_calls: list[str] = []

    def fake_loader(url: str) -> str:
        load_calls.append(url)
        return YES_PAGE

    monkeypatch.setattr(cli, "_build_web_page_loader", lambda: fake_loader)
    monkeypatch.setattr(cli, "_web_sleep", lambda seconds: None)

    result = _run(
        monkeypatch,
        extra=[
            "--data-dir",
            str(data_dir),
            "--experiments-dir",
            str(experiments_dir),
            "--max",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "pinned=5 mapped=5 already=0 checked=2 found=2" in result.output
    assert len(load_calls) == 2


def test_reader_error_on_one_question_does_not_abort_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "snapshots"
    experiments_dir = tmp_path / "experiments"
    _write_manifest(experiments_dir, [501, 502])

    SnapshotStore(data_dir).append(
        [_snapshot(501, 121, status="resolved"), _snapshot(502, 122, status="resolved")]
    )

    def flaky_loader(url: str) -> str:
        if url.endswith("/121/"):
            raise RuntimeError("boom")
        return NO_PAGE

    monkeypatch.setattr(cli, "_build_web_page_loader", lambda: flaky_loader)
    monkeypatch.setattr(cli, "_web_sleep", lambda seconds: None)

    result = _run(
        monkeypatch,
        extra=["--data-dir", str(data_dir), "--experiments-dir", str(experiments_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "q501 (post 121): web-resolution FAILED" in result.output
    assert "pinned=2 mapped=2 already=0 checked=2 found=1" in result.output

    recorded = list(WebResolutionStore(tmp_path / "web-resolutions").iter_all())
    assert len(recorded) == 1
    assert recorded[0].question_id == 502


def test_pinned_ids_from_docs_parses_id_list_lines(tmp_path: Path) -> None:
    experiments_dir = tmp_path / "experiments"
    _write_doc_only_ids(experiments_dir, [43716, 43507, 44373])
    (experiments_dir / "prose.md").write_text(
        "This doc mentions 2026 and 60 days and 40 questions but has no id list.\n",
        encoding="utf-8",
    )

    ids = cli._pinned_ids_from_docs(experiments_dir)
    assert ids == {43716, 43507, 44373}


def test_pinned_ids_from_docs_missing_directory(tmp_path: Path) -> None:
    assert cli._pinned_ids_from_docs(tmp_path / "does-not-exist") == set()
