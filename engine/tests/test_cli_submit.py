"""Tests for the ``submit-tournament`` CLI command (no real network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from superforecaster import cli
from superforecaster.llm import LlmResult
from superforecaster.submission import SubmissionError

from . import fixtures

runner = CliRunner()

_FORECAST_JSON = (
    "Reasoning about the reference class and adjustments.\n"
    '```json\n{"probability": 0.42, "confidence": "medium", '
    '"rationale": "reference class then adjust", '
    '"reference_class": "comparable questions"}\n```'
)


class _FakeLlm:
    """Stands in for the LLM client; returns one fixed forecast envelope."""

    def run(self, system: str, prompt: str, model: str, max_tokens: int | None = None) -> LlmResult:
        return LlmResult(
            text=_FORECAST_JSON,
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.02,
            stop_reason="end_turn",
            model=model,
        )


class _FakeListClient:
    """Serves tournament posts; records the tournament asked for."""

    def __init__(self, posts: list[dict[str, Any]]) -> None:
        self._posts = posts
        self.closed = False
        self.tournament: str | None = None

    def iter_tournament_binary_posts(
        self, tournament: str, page_size: int = 50, max_posts: int | None = None
    ) -> list[dict[str, Any]]:
        self.tournament = tournament
        return self._posts if max_posts is None else self._posts[:max_posts]

    def close(self) -> None:
        self.closed = True


class _FakeSubmissionClient:
    """Records forecast/comment calls in order; can be told to fail per-question."""

    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.closed = False
        self._fail_on = fail_on or set()

    def submit_binary_forecast(self, question_id: int, probability: float) -> None:
        self.calls.append(("submit", (question_id, probability)))
        if question_id in self._fail_on:
            raise SubmissionError(400, "already forecast on this question")

    def post_rationale_comment(self, post_id: int, text: str) -> None:
        self.calls.append(("comment", (post_id, text)))

    def close(self) -> None:
        self.closed = True


def _install(
    monkeypatch: pytest.MonkeyPatch,
    posts: list[dict[str, Any]],
    sub: _FakeSubmissionClient | None = None,
) -> _FakeListClient:
    list_client = _FakeListClient(posts)
    monkeypatch.setattr(cli, "_build_client", lambda token: list_client)
    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FakeLlm())
    monkeypatch.setattr("superforecaster.llm.check_claude_cli", lambda path="claude": True)
    if sub is not None:
        monkeypatch.setattr(cli, "_build_submission_client", lambda token: sub)
    monkeypatch.setenv("METACULUS_TOKEN", "human-token")
    return list_client


def test_double_gate_live_tournament_without_live_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # avoid picking up a stray .env
    result = runner.invoke(
        cli.app,
        ["submit-tournament", "--tournament", cli.LIVE_TOURNAMENT, "--forecast-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "--live" in result.output


def test_dry_run_posts_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install(monkeypatch, [fixtures.open_binary_post(101, 201)])

    # If the submission client is built at all in dry mode, fail loudly.
    def _boom(token: str) -> Any:
        raise AssertionError("submission client must not be built in dry-run")

    monkeypatch.setattr(cli, "_build_submission_client", _boom)

    result = runner.invoke(
        cli.app,
        ["submit-tournament", "--forecast-dir", str(tmp_path), "--max-questions", "5"],
    )
    assert result.exit_code == 0, result.output
    assert "[dry]" in result.output
    assert "submitted=0" in result.output

    from superforecaster.forecast_store import ForecastStore

    stored = list(ForecastStore(tmp_path).iter_all())
    assert len(stored) == 1
    assert stored[0].dry_run is True


def test_live_submits_then_comments_and_marks_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sub = _FakeSubmissionClient()
    _install(monkeypatch, [fixtures.open_binary_post(101, 201)], sub=sub)
    monkeypatch.setenv("METACULUS_BOT_TOKEN", "bot-token")

    result = runner.invoke(
        cli.app,
        ["submit-tournament", "--forecast-dir", str(tmp_path), "--live"],
    )
    assert result.exit_code == 0, result.output
    assert "SUBMITTED" in result.output
    assert "submitted=1" in result.output

    # Forecast first, then comment, in that order.
    assert [c[0] for c in sub.calls] == ["submit", "comment"]
    assert sub.calls[0][1] == (201, pytest.approx(0.42))
    # Comment carries the disclosure prefix and the rationale.
    assert sub.calls[1][1][0] == 101
    assert cli._DISCLOSURE in sub.calls[1][1][1]
    assert "reference class then adjust" in sub.calls[1][1][1]
    assert sub.closed

    from superforecaster.forecast_store import ForecastStore

    stored = list(ForecastStore(tmp_path).iter_all())
    assert len(stored) == 1
    assert stored[0].dry_run is False
    assert stored[0].submitted_at is not None


def test_already_submitted_question_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # First run submits q201.
    sub = _FakeSubmissionClient()
    _install(monkeypatch, [fixtures.open_binary_post(101, 201)], sub=sub)
    monkeypatch.setenv("METACULUS_BOT_TOKEN", "bot-token")
    runner.invoke(cli.app, ["submit-tournament", "--forecast-dir", str(tmp_path), "--live"])

    # Second run sees the same question and must skip it (one-forecast rule).
    sub2 = _FakeSubmissionClient()
    _install(monkeypatch, [fixtures.open_binary_post(101, 201)], sub=sub2)
    monkeypatch.setenv("METACULUS_BOT_TOKEN", "bot-token")
    result = runner.invoke(
        cli.app, ["submit-tournament", "--forecast-dir", str(tmp_path), "--live"]
    )
    assert result.exit_code == 0, result.output
    assert "already submitted" in result.output
    assert "submitted=0" in result.output
    assert sub2.calls == []


def test_midbatch_failure_continues_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    posts = [
        fixtures.open_binary_post(101, 201),  # this one fails to submit
        fixtures.open_binary_post(102, 202),  # this one should still succeed
    ]
    sub = _FakeSubmissionClient(fail_on={201})
    _install(monkeypatch, posts, sub=sub)
    monkeypatch.setenv("METACULUS_BOT_TOKEN", "bot-token")

    result = runner.invoke(
        cli.app,
        ["submit-tournament", "--forecast-dir", str(tmp_path), "--live", "--max-questions", "5"],
    )
    assert result.exit_code == 1, result.output
    assert "1 submission(s) failed" in result.output
    assert "submitted=1" in result.output

    from superforecaster.forecast_store import ForecastStore

    stored = {r.question_id: r for r in ForecastStore(tmp_path).iter_all()}
    assert stored[201].dry_run is True  # failed submission stored as dry-run
    assert stored[202].dry_run is False  # the later one submitted fine


def test_live_falls_back_to_metaculus_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no bot token, a live run uses METACULUS_TOKEN to submit."""
    sub = _FakeSubmissionClient()
    seen: dict[str, str] = {}

    def _capture(token: str) -> _FakeSubmissionClient:
        seen["token"] = token
        return sub

    monkeypatch.setattr(cli, "_build_submission_client", _capture)
    _install(monkeypatch, [fixtures.open_binary_post(101, 201)])
    monkeypatch.delenv("METACULUS_BOT_TOKEN", raising=False)
    # _install set METACULUS_TOKEN = "human-token"; that is the fallback.

    result = runner.invoke(
        cli.app, ["submit-tournament", "--forecast-dir", str(tmp_path), "--live"]
    )
    assert result.exit_code == 0, result.output
    assert "submitted=1" in result.output
    assert seen["token"] == "human-token"
    assert [c[0] for c in sub.calls] == ["submit", "comment"]


def test_live_without_any_token_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install(monkeypatch, [fixtures.open_binary_post(101, 201)])
    monkeypatch.delenv("METACULUS_BOT_TOKEN", raising=False)
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)  # avoid a stray .env providing a token
    result = runner.invoke(
        cli.app, ["submit-tournament", "--forecast-dir", str(tmp_path), "--live"]
    )
    assert result.exit_code == 1
    assert "METACULUS_BOT_TOKEN" in result.output
