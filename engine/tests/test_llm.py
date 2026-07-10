"""Tests for the Claude CLI wrapper, using a fake subprocess runner.

None of these tests shell out to the real ``claude`` binary; a fake ``runner``
callable returns canned ``subprocess.CompletedProcess`` objects and records the
argv/env it was handed, so we can assert on both the parsing and the invocation.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

import pytest

from superforecaster.config import TaskModelConfig
from superforecaster.llm import (
    ClaudeCliClient,
    CodexCliClient,
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
    # The user prompt is piped via stdin, never argv (Windows ~32k argv limit).
    assert "-p" in argv and "the user" not in argv
    assert runner.calls[0]["kwargs"]["input"] == "the user"
    assert "--system-prompt" in argv
    assert argv[argv.index("--system-prompt") + 1] == "the system"
    assert "--model" in argv and argv[argv.index("--model") + 1] == "claude-sonnet-5"
    assert argv[argv.index("--output-format") + 1] == "json"


def test_effort_flag_present_when_set() -> None:
    runner = _FakeRunner(_completed(json.dumps(_envelope())))
    client = ClaudeCliClient(runner=runner)

    client.run(system="s", prompt="p", model="m", effort="low")

    argv = runner.calls[0]["argv"]
    assert "--effort" in argv
    assert argv[argv.index("--effort") + 1] == "low"


def test_effort_flag_absent_when_none() -> None:
    runner = _FakeRunner(_completed(json.dumps(_envelope())))
    client = ClaudeCliClient(runner=runner)

    client.run(system="s", prompt="p", model="m")  # effort defaults to None

    assert "--effort" not in runner.calls[0]["argv"]


def test_env_passed_to_runner_lacks_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")
    monkeypatch.setenv("KEEP_ME", "yes")
    runner = _FakeRunner(_completed(json.dumps(_envelope())))
    client = ClaudeCliClient(runner=runner)

    client.run(system="s", prompt="p", model="m")

    env = runner.calls[0]["kwargs"]["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert env["KEEP_ME"] == "yes"
    # The prompt rides stdin; subprocess closes the pipe after writing it.
    assert runner.calls[0]["kwargs"]["input"] == "p"


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
    cfg = TaskModelConfig(model="m", max_tokens=100, provider="openrouter")
    with pytest.raises(LlmError, match="openrouter"):
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


# --- Codex CLI client -------------------------------------------------------

# A realistic JSONL stream: thread/turn bookkeeping, a reasoning item, then the
# agent_message, then the usage-bearing turn.completed event.
_CODEX_STDOUT = "\n".join(
    [
        json.dumps({"type": "thread.started", "thread_id": "abc"}),
        json.dumps({"type": "turn.started"}),
        json.dumps(
            {"type": "item.completed", "item": {"id": "r0", "type": "reasoning", "text": "think"}}
        ),
        json.dumps(
            {
                "type": "item.completed",
                "item": {"id": "i0", "type": "agent_message", "text": '{"probability": 0.4}'},
            }
        ),
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 17881, "cached_input_tokens": 4992, "output_tokens": 5},
            }
        ),
    ]
)


class _CodexFakeRunner:
    """Records argv/kwargs and, at call time, reads any --output-schema file.

    Reading the schema file *during* the call is deliberate: the client deletes
    it in a finally block, so a test cannot open it afterwards.
    """

    def __init__(self, completed: subprocess.CompletedProcess[str]) -> None:
        self._completed = completed
        self.calls: list[dict[str, Any]] = []
        self.schema_file_contents: list[str | None] = []
        # cwd/schema temp files exist only during the call (the client removes
        # them in a finally block), so we snapshot what we need here.
        self.cwd_is_empty_dir: list[bool] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"argv": argv, "kwargs": kwargs})
        cwd = kwargs.get("cwd")
        self.cwd_is_empty_dir.append(
            isinstance(cwd, str) and os.path.isdir(cwd) and os.listdir(cwd) == []
        )
        if "--output-schema" in argv:
            path = argv[argv.index("--output-schema") + 1]
            with open(path, encoding="utf-8") as fh:
                self.schema_file_contents.append(fh.read())
        else:
            self.schema_file_contents.append(None)
        return self._completed


def _codex_completed(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["codex"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_codex_jsonl_happy_path() -> None:
    runner = _CodexFakeRunner(_codex_completed(_CODEX_STDOUT))
    client = CodexCliClient(runner=runner)

    result = client.run(system="sys", prompt="hi", model="gpt-5.1-codex")

    assert result.text == '{"probability": 0.4}'
    assert result.input_tokens == 17881
    assert result.output_tokens == 5
    assert result.stop_reason is None
    assert result.model == "gpt-5.1-codex"
    # Model is unpriced in the config, so cost is unknown rather than guessed.
    assert result.cost_usd is None


def test_codex_last_agent_message_wins() -> None:
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "i0", "type": "agent_message", "text": "first"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "i1", "type": "agent_message", "text": "second"},
                }
            ),
            json.dumps(
                {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}}
            ),
        ]
    )
    runner = _CodexFakeRunner(_codex_completed(stdout))
    client = CodexCliClient(runner=runner)

    result = client.run(system="s", prompt="p", model="m")

    assert result.text == "second"


def test_codex_skips_unparseable_lines() -> None:
    stdout = "\n".join(
        [
            "Reading additional input from stdin...",
            "not json at all {",
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "i0", "type": "agent_message", "text": "ok"},
                }
            ),
            json.dumps(
                {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 4}}
            ),
        ]
    )
    runner = _CodexFakeRunner(_codex_completed(stdout))
    client = CodexCliClient(runner=runner)

    result = client.run(system="s", prompt="p", model="m")

    assert result.text == "ok"
    assert result.input_tokens == 3
    assert result.output_tokens == 4


def test_codex_missing_agent_message_raises() -> None:
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "r0", "type": "reasoning", "text": "only thinking"},
                }
            ),
            json.dumps(
                {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}
            ),
        ]
    )
    runner = _CodexFakeRunner(_codex_completed(stdout))
    client = CodexCliClient(runner=runner)

    with pytest.raises(LlmError, match="no agent_message"):
        client.run(system="s", prompt="p", model="m")


def test_codex_nonzero_exit_raises_with_stderr() -> None:
    runner = _CodexFakeRunner(_codex_completed("", returncode=1, stderr="oops line 1\noops line 2"))
    client = CodexCliClient(runner=runner)

    with pytest.raises(LlmError) as excinfo:
        client.run(system="s", prompt="p", model="m")
    assert excinfo.value.stderr is not None
    assert "oops line 2" in excinfo.value.stderr


def test_codex_argv_and_schema_file() -> None:
    runner = _CodexFakeRunner(_codex_completed(_CODEX_STDOUT))
    client = CodexCliClient(codex_path="codex", runner=runner)
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"probability": {"type": "number"}},
    }

    client.run(system="the system", prompt="the user", model="gpt-5.1-codex", json_schema=schema)

    argv = runner.calls[0]["argv"]
    assert argv[0] == "codex"
    assert argv[1] == "exec"
    # "-" makes codex read the prompt from stdin (Windows ~32k argv limit);
    # system + user prompt are combined into that piped input.
    assert argv[2] == "-"
    piped = runner.calls[0]["kwargs"]["input"]
    assert "the system" in piped and "the user" in piped
    assert "--skip-git-repo-check" in argv
    assert argv[argv.index("-s") + 1] == "read-only"
    assert "--json" in argv
    # Model passed via -m because it is non-empty.
    assert argv[argv.index("-m") + 1] == "gpt-5.1-codex"
    # Schema written to the temp file named on the command line.
    assert "--output-schema" in argv
    written = runner.schema_file_contents[0]
    assert written is not None
    # The written schema is the input plus OpenAI-strict normalization: every
    # object node gains "additionalProperties": false (see _strictify_schema).
    assert json.loads(written) == {**schema, "additionalProperties": False}


def test_codex_effort_override_present_when_set() -> None:
    runner = _CodexFakeRunner(_codex_completed(_CODEX_STDOUT))
    client = CodexCliClient(runner=runner)

    client.run(system="s", prompt="p", model="m", effort="low")

    argv = runner.calls[0]["argv"]
    assert "-c" in argv
    # The value carries TOML quotes, so the CLI parses the override as the
    # string "low" (verified accepted in a live smoke call).
    assert argv[argv.index("-c") + 1] == 'model_reasoning_effort="low"'


def test_codex_effort_override_absent_when_none() -> None:
    runner = _CodexFakeRunner(_codex_completed(_CODEX_STDOUT))
    client = CodexCliClient(runner=runner)

    client.run(system="s", prompt="p", model="m")  # effort defaults to None

    assert "model_reasoning_effort" not in " ".join(runner.calls[0]["argv"])


def test_codex_empty_model_omits_flag_and_schema_optional() -> None:
    runner = _CodexFakeRunner(_codex_completed(_CODEX_STDOUT))
    client = CodexCliClient(runner=runner)

    client.run(system="s", prompt="p", model="")

    argv = runner.calls[0]["argv"]
    assert "-m" not in argv
    assert "--output-schema" not in argv


def test_codex_stdin_prompt_and_empty_cwd() -> None:
    runner = _CodexFakeRunner(_codex_completed(_CODEX_STDOUT))
    client = CodexCliClient(runner=runner)

    client.run(system="s", prompt="p", model="m")

    kwargs = runner.calls[0]["kwargs"]
    # The combined prompt is piped via stdin; the pipe closes after writing, so
    # codex's block-on-open-stdin behavior is not triggered.
    assert kwargs["input"] == "s\n\n---\n\np"
    cwd = kwargs["cwd"]
    # The working directory was a real, empty temp dir at call time (snapshotted
    # by the fake runner) — not the repo root, so Codex cannot ingest project
    # context (AGENTS.md, etc.).
    assert runner.cwd_is_empty_dir[0] is True
    assert os.path.abspath(cwd) != os.getcwd()


def test_codex_not_found_raises_clear_error() -> None:
    def _boom(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("codex")

    client = CodexCliClient(runner=_boom)
    with pytest.raises(LlmError, match="Codex CLI not found"):
        client.run(system="s", prompt="p", model="m")


def test_build_llm_client_codex_cli() -> None:
    cfg = TaskModelConfig(model="m", max_tokens=100, provider="codex-cli")
    assert isinstance(build_llm_client(cfg), CodexCliClient)


def test_check_provider_cli_codex_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    import superforecaster.llm as llm

    monkeypatch.setattr(llm, "check_codex_cli", lambda path="codex": True)
    cfg = TaskModelConfig(model="m", max_tokens=100, provider="codex-cli")
    assert check_provider_cli(cfg) is True
