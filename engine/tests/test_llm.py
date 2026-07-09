"""Tests for the Claude CLI wrapper, using a fake subprocess runner.

None of these tests shell out to the real ``claude`` binary; a fake ``runner``
callable returns canned ``subprocess.CompletedProcess`` objects and records the
argv/env it was handed, so we can assert on both the parsing and the invocation.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from superforecaster.config import TaskModelConfig
from superforecaster.llm import (
    ClaudeCliClient,
    LlmError,
    build_llm_client,
    check_provider_cli,
)


def _envelope(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": 'reasoning...\n```json\n{"probability": 0.4}\n```',
        "stop_reason": "end_turn",
        "total_cost_usd": 0.1234,
        "usage": {"input_tokens": 4590, "output_tokens": 42},
    }
    base.update(overrides)
    return base


class _FakeRunner:
    """Records the last call and returns a fixed CompletedProcess."""

    def __init__(self, completed: subprocess.CompletedProcess[str]) -> None:
        self._completed = completed
        self.calls: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"argv": argv, "kwargs": kwargs})
        return self._completed


def _completed(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_success_envelope_parses() -> None:
    runner = _FakeRunner(_completed(json.dumps(_envelope())))
    client = ClaudeCliClient(runner=runner)

    result = client.run(system="sys", prompt="hi", model="claude-sonnet-5")

    assert result.text.endswith("```")
    assert result.input_tokens == 4590
    assert result.output_tokens == 42
    assert result.cost_usd == pytest.approx(0.1234)
    assert result.stop_reason == "end_turn"
    assert result.model == "claude-sonnet-5"


def test_argv_contains_expected_flags() -> None:
    runner = _FakeRunner(_completed(json.dumps(_envelope())))
    client = ClaudeCliClient(claude_path="claude", runner=runner)

    client.run(system="the system", prompt="the user", model="claude-sonnet-5")

    argv = runner.calls[0]["argv"]
    assert argv[0] == "claude"
    assert "-p" in argv and argv[argv.index("-p") + 1] == "the user"
    assert "--system-prompt" in argv
    assert argv[argv.index("--system-prompt") + 1] == "the system"
    assert "--model" in argv and argv[argv.index("--model") + 1] == "claude-sonnet-5"
    assert argv[argv.index("--output-format") + 1] == "json"


def test_env_passed_to_runner_lacks_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")
    monkeypatch.setenv("KEEP_ME", "yes")
    runner = _FakeRunner(_completed(json.dumps(_envelope())))
    client = ClaudeCliClient(runner=runner)

    client.run(system="s", prompt="p", model="m")

    env = runner.calls[0]["kwargs"]["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert env["KEEP_ME"] == "yes"
    # stdin is closed so the CLI does not warn about an open stdin.
    assert runner.calls[0]["kwargs"]["stdin"] == subprocess.DEVNULL


def test_is_error_envelope_raises() -> None:
    env = _envelope(is_error=True, subtype="error_during_execution")
    runner = _FakeRunner(_completed(json.dumps(env)))
    client = ClaudeCliClient(runner=runner)

    with pytest.raises(LlmError) as excinfo:
        client.run(system="s", prompt="p", model="m")
    assert excinfo.value.envelope is not None


def test_non_success_subtype_raises() -> None:
    env = _envelope(subtype="something_else")
    runner = _FakeRunner(_completed(json.dumps(env)))
    client = ClaudeCliClient(runner=runner)

    with pytest.raises(LlmError):
        client.run(system="s", prompt="p", model="m")


def test_nonzero_returncode_raises_with_stderr() -> None:
    runner = _FakeRunner(_completed("", returncode=2, stderr="boom line 1\nboom line 2"))
    client = ClaudeCliClient(runner=runner)

    with pytest.raises(LlmError) as excinfo:
        client.run(system="s", prompt="p", model="m")
    assert excinfo.value.stderr is not None
    assert "boom line 2" in excinfo.value.stderr


def test_unparseable_stdout_raises() -> None:
    runner = _FakeRunner(_completed("not json at all", stderr="warn"))
    client = ClaudeCliClient(runner=runner)

    with pytest.raises(LlmError):
        client.run(system="s", prompt="p", model="m")


def test_missing_result_text_raises() -> None:
    env = _envelope()
    del env["result"]
    runner = _FakeRunner(_completed(json.dumps(env)))
    client = ClaudeCliClient(runner=runner)

    with pytest.raises(LlmError):
        client.run(system="s", prompt="p", model="m")


def test_cli_not_found_raises_clear_error() -> None:
    def _boom(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("claude")

    client = ClaudeCliClient(runner=_boom)
    with pytest.raises(LlmError, match="Claude CLI not found"):
        client.run(system="s", prompt="p", model="m")


def test_build_llm_client_claude_cli() -> None:
    cfg = TaskModelConfig(model="m", max_tokens=100, provider="claude-cli")
    assert isinstance(build_llm_client(cfg), ClaudeCliClient)


def test_build_llm_client_unknown_provider() -> None:
    cfg = TaskModelConfig(model="m", max_tokens=100, provider="codex-cli")
    with pytest.raises(LlmError, match="codex-cli"):
        build_llm_client(cfg)


def test_check_provider_cli_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    import superforecaster.llm as llm

    monkeypatch.setattr(llm, "check_claude_cli", lambda path="claude": True)
    cfg = TaskModelConfig(model="m", max_tokens=100, provider="claude-cli")
    assert check_provider_cli(cfg) is True


def test_check_provider_cli_unknown_raises() -> None:
    cfg = TaskModelConfig(model="m", max_tokens=100, provider="nope")
    with pytest.raises(LlmError, match="nope"):
        check_provider_cli(cfg)
