"""CLI smoke tests using a fake client injected through the _build_client seam."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from superforecaster import cli
from superforecaster.llm import LlmResult
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


_FORECAST_JSON = (
    "Reasoning about the reference class and adjustments.\n"
    '```json\n{"probability": 0.42, "confidence": "medium", '
    '"rationale": "reference class then adjust", '
    '"reference_class": "comparable questions"}\n```'
)


class _FakeLlm:
    """Stands in for the LLM client; returns one fixed forecast envelope."""

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        return LlmResult(
            text=_FORECAST_JSON,
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.02,
            stop_reason="end_turn",
            model=model,
        )


def _pass_cli_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the provider CLI check succeed without a real 'claude' binary."""
    monkeypatch.setattr("superforecaster.llm.check_claude_cli", lambda path="claude": True)


def _seed_open_question(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Scrape one open, soon-closing question into the snapshot store."""
    _install_fake(monkeypatch, [fixtures.open_binary_post(101, 201)])
    runner.invoke(cli.app, ["scrape", "--data-dir", str(tmp_path)])


def test_forecast_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _seed_open_question(monkeypatch, snap_dir)

    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FakeLlm())
    _pass_cli_check(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "p=0.42" in result.output
    assert "confidence=medium" in result.output
    assert "n=1" in result.output

    from superforecaster.forecast_store import ForecastStore

    stored = list(ForecastStore(fc_dir).iter_all())
    assert len(stored) == 1
    assert stored[0].dry_run is True
    assert stored[0].reference_class == "comparable questions"


def test_forecast_without_cli_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("superforecaster.llm.check_claude_cli", lambda path="claude": False)
    monkeypatch.chdir(tmp_path)  # avoid picking up a stray .env
    result = runner.invoke(cli.app, ["forecast", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "Claude Code" in result.output


def test_forecast_no_matching_questions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FakeLlm())
    _pass_cli_check(monkeypatch)
    result = runner.invoke(cli.app, ["forecast", "--data-dir", str(tmp_path / "empty")])
    assert result.exit_code == 0, result.output
    assert "no matching open questions" in result.output


def test_data_dir_comes_from_settings_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no --data-dir flag or env, the snapshot dir comes from settings.toml."""
    snap_dir = tmp_path / "from-settings"
    settings = tmp_path / "settings.toml"
    settings.write_text(f'[data]\nsnapshot_dir = "{snap_dir.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("SUPERFORECASTER_CONFIG", str(settings))
    monkeypatch.delenv("SUPERFORECASTER_DATA_DIR", raising=False)
    _install_fake(monkeypatch, [fixtures.open_binary_post(101, 201)])

    result = runner.invoke(cli.app, ["scrape"])  # no --data-dir -> settings dir
    assert result.exit_code == 0, result.output
    assert "written=1" in result.output

    stats = runner.invoke(cli.app, ["stats"])
    assert str(snap_dir) in stats.output
