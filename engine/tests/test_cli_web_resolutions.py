"""CLI tests for `update-resolutions --from-web` (no browser, no network).

The API pass is short-circuited (all seeded questions are already resolved, so
nothing is re-fetched) and the web pass runs against an injected fake page loader
plus a spy sleep, so we can assert the throttle-and-cap behaviour precisely.
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
from superforecaster.web_resolutions import WebResolutionStore

runner = CliRunner()

ON = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)

NO_PAGE = "Q title\nCategory\nRESOLVED\nNo\nComments\n"
OPEN_PAGE = "Q title\nCategory\nOPEN\nComments\n"


class _FakeClient:
    """A Metaculus client whose get_post is never expected to be called here."""

    def __init__(self) -> None:
        self.closed = False

    def get_post(self, post_id: int) -> dict[str, Any]:  # pragma: no cover - not reached
        raise AssertionError("API pass should not re-fetch already-resolved questions")

    def close(self) -> None:
        self.closed = True


def _resolved_blank(question_id: int, post_id: int) -> QuestionSnapshot:
    """A snapshot that is resolved/closed on the API but has no resolution field."""
    return QuestionSnapshot(
        scraped_at=ON,
        post_id=post_id,
        question_id=question_id,
        title=f"Q{question_id}",
        status="resolved",
        resolution=None,
    )


def _seed(data_dir: Path, n: int) -> None:
    SnapshotStore(data_dir).append(
        [_resolved_blank(question_id=200 + i, post_id=100 + i) for i in range(n)]
    )


def _run(monkeypatch: pytest.MonkeyPatch, data_dir: Path, *, extra: list[str]) -> Any:
    fake_client = _FakeClient()
    monkeypatch.setattr(cli, "_build_client", lambda token: fake_client)
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    return runner.invoke(
        cli.app,
        ["update-resolutions", "--data-dir", str(data_dir), *extra],
    )


def test_from_web_respects_throttle_and_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "snapshots"
    # 22 resolved-blank questions; the hard cap is 20.
    _seed(data_dir, 22)

    load_calls: list[str] = []
    sleep_calls: list[float] = []

    def fake_loader(url: str) -> str:
        load_calls.append(url)
        return NO_PAGE

    monkeypatch.setattr(cli, "_build_web_page_loader", lambda: fake_loader)
    monkeypatch.setattr(cli, "_web_sleep", lambda seconds: sleep_calls.append(seconds))

    result = _run(monkeypatch, data_dir, extra=[])
    assert result.exit_code == 0, result.output

    # Cap: at most _WEB_PAGE_CAP pages loaded. Throttle: one sleep between pages.
    assert len(load_calls) == cli._WEB_PAGE_CAP
    assert len(sleep_calls) == cli._WEB_PAGE_CAP - 1
    assert all(s == cli._WEB_THROTTLE_SECONDS for s in sleep_calls)
    assert f"update-resolutions (web): checked={cli._WEB_PAGE_CAP} found={cli._WEB_PAGE_CAP}" in (
        result.output
    )

    # The web store recorded exactly the capped number, with provenance.
    web_store = WebResolutionStore(tmp_path / "web-resolutions")
    recorded = list(web_store.iter_all())
    assert len(recorded) == cli._WEB_PAGE_CAP
    assert all(r.resolution == "no" for r in recorded)
    assert all(r.source_url.endswith(f"/{r.post_id}/") for r in recorded)


def test_from_web_counts_misses_without_guessing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "snapshots"
    _seed(data_dir, 3)

    # Page 100 (first by post_id) has a banner; the other two are still open.
    def fake_loader(url: str) -> str:
        return NO_PAGE if url.endswith("/100/") else OPEN_PAGE

    monkeypatch.setattr(cli, "_build_web_page_loader", lambda: fake_loader)
    monkeypatch.setattr(cli, "_web_sleep", lambda seconds: None)

    result = _run(monkeypatch, data_dir, extra=[])
    assert result.exit_code == 0, result.output
    assert "update-resolutions (web): checked=3 found=1 failed=2" in result.output

    recorded = list(WebResolutionStore(tmp_path / "web-resolutions").iter_all())
    assert len(recorded) == 1  # only the real banner was recorded; misses never guessed


def test_no_from_web_skips_web_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "snapshots"
    _seed(data_dir, 2)

    built: list[bool] = []

    def factory() -> Any:
        built.append(True)
        return lambda _url: ""

    monkeypatch.setattr(cli, "_build_web_page_loader", factory)
    monkeypatch.setattr(cli, "_web_sleep", lambda seconds: None)

    result = _run(monkeypatch, data_dir, extra=["--no-from-web"])
    assert result.exit_code == 0, result.output
    assert "update-resolutions (web)" not in result.output
    assert built == []  # the web page loader was never even built
    assert not (tmp_path / "web-resolutions").exists()
