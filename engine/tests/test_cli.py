"""CLI smoke tests using a fake client injected through the _build_client seam."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from superforecaster import cli
from superforecaster.metaculus import MetaculusClient

from . import fixtures

runner = CliRunner()


class FakeClient:
    """Stands in for MetaculusClient; serves fixture posts, no network."""

    def __init__(self, posts: list[dict[str, Any]]) -> None:
        self._posts = posts
        self.closed = False

    def iter_open_binary_posts(
        self, page_size: int = 100, max_posts: int | None = None
    ) -> list[dict[str, Any]]:
        return self._posts if max_posts is None else self._posts[:max_posts]

    def get_post(self, post_id: int) -> dict[str, Any]:
        for post in self._posts:
            if post["id"] == post_id:
                return post
        raise KeyError(post_id)

    def close(self) -> None:
        self.closed = True


def _install_fake(monkeypatch: pytest.MonkeyPatch, posts: list[dict[str, Any]]) -> FakeClient:
    fake = FakeClient(posts)
    monkeypatch.setattr(cli, "_build_client", lambda token: fake)
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    return fake


def test_scrape_writes_and_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    posts = [
        fixtures.open_binary_post(101, 201),
        fixtures.open_binary_post_no_cp(102, 202),
        fixtures.group_post(103),  # skipped
    ]
    fake = _install_fake(monkeypatch, posts)

    result = runner.invoke(cli.app, ["scrape", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "fetched=3" in result.output
    assert "written=2" in result.output
    assert "skipped=1" in result.output
    assert fake.closed


def test_scrape_without_token_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)  # ensure no stray .env is picked up
    result = runner.invoke(cli.app, ["scrape", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "METACULUS_TOKEN" in result.output


def test_update_resolutions_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Seed an open snapshot, then have the fake return the resolved version.
    open_post = fixtures.open_binary_post(104, 204)
    _install_fake(monkeypatch, [open_post])
    runner.invoke(cli.app, ["scrape", "--data-dir", str(tmp_path)])

    resolved = fixtures.resolved_yes_post(104, 204)
    _install_fake(monkeypatch, [resolved])
    result = runner.invoke(cli.app, ["update-resolutions", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "newly_resolved=1" in result.output


def test_stats_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake(monkeypatch, [fixtures.open_binary_post(101, 201)])
    runner.invoke(cli.app, ["scrape", "--data-dir", str(tmp_path)])
    result = runner.invoke(cli.app, ["stats", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "total records:      1" in result.output
    assert "unique questions:   1" in result.output


def test_build_client_returns_real_client() -> None:
    """The seam builds a real client when not monkeypatched (constructor smoke)."""
    client = cli._build_client("tok")
    assert isinstance(client, MetaculusClient)
    client.close()
