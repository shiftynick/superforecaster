"""Call an LLM through a local CLI, billed to the owner's subscription.

The engine used to call the Anthropic API SDK, which bills a separately-metered
API key. This module replaces that with a subprocess wrapper around the **Claude
Code CLI** (``claude -p ... --output-format json``), so forecast passes draw on
the owner's Claude subscription instead. See ``docs/adr/0002-llm-via-claude-cli``.

The design is deliberately provider-neutral so a second provider (a Codex CLI,
OpenRouter, ...) can be added later as a purely additive change:

- :class:`LlmResult` and :class:`LlmError` carry no provider-specific detail.
- :class:`LlmClient` is the structural type every client satisfies: a single
  ``run(system, prompt, model, max_tokens)`` method returning an
  :class:`LlmResult`.
- :func:`build_llm_client` is the one factory the rest of the engine calls; it
  dispatches on the task's ``provider`` field and is the only place that knows
  which concrete client class a provider name maps to.

Two subscription-billing safeguards live in :class:`ClaudeCliClient`:

- ``ANTHROPIC_API_KEY`` is stripped from the subprocess environment, so a stray
  key can never silently redirect billing to the metered API.
- ``stdin`` is closed (``DEVNULL``); otherwise the CLI prints a warning to
  stderr about an open stdin.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .config import TaskModelConfig

# The env var that, if present, would bill the metered API instead of the
# subscription. We remove it from the child process's environment.
_API_KEY_ENV = "ANTHROPIC_API_KEY"

# Provider name -> the CLI binary it shells out to. Adding a provider is a new
# entry here plus a branch in build_llm_client; nothing else needs to change.
_PROVIDER_BINARIES: dict[str, str] = {"claude-cli": "claude"}

# Human-facing tuple of what we support, for error messages.
SUPPORTED_PROVIDERS: tuple[str, ...] = tuple(sorted(_PROVIDER_BINARIES))


@dataclass(frozen=True)
class LlmResult:
    """One LLM reply plus the bookkeeping every call must record.

    Attributes:
        text: The model's full text answer (we parse JSON out of it upstream).
        input_tokens: Prompt tokens the provider reported for this call.
        output_tokens: Completion tokens the provider reported for this call.
        cost_usd: The provider-reported cost, or ``None`` if it gave none. Under
            a subscription this is *notional* — it measures quota draw, not
            dollars actually billed — but it satisfies the cost-visibility rule.
        stop_reason: Why generation stopped (``"end_turn"``, a refusal, ...), or
            ``None`` if the provider did not say.
        model: The model id that was requested for this call.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    stop_reason: str | None
    model: str


class LlmError(Exception):
    """An LLM call failed: the CLI was missing, errored, or gave junk output.

    Carries optional diagnostic detail so callers can log *why* without the raw
    (possibly large) payloads leaking into normal control flow.

    Attributes:
        stderr: A short tail of the CLI's stderr, when a subprocess failed.
        envelope: The parsed JSON envelope, when the CLI ran but reported an
            error inside a well-formed result object.
    """

    def __init__(
        self,
        message: str,
        *,
        stderr: str | None = None,
        envelope: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.stderr = stderr
        self.envelope = envelope


class LlmClient(Protocol):
    """Structural type for any LLM client the forecaster can drive.

    A single method: build the request, run it, and return an
    :class:`LlmResult`. ``ClaudeCliClient`` satisfies this, and so does any fake
    used in tests, so the forecaster never depends on a concrete class.
    """

    def run(
        self, system: str, prompt: str, model: str, max_tokens: int | None = None
    ) -> LlmResult: ...


# The subprocess.run signature, narrowed to what we use. Tests pass a fake here.
Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _stderr_tail(stderr: str | None, max_lines: int = 8) -> str | None:
    """Return the last few non-blank lines of stderr, for a compact error."""
    if not stderr:
        return None
    lines = [line for line in stderr.splitlines() if line.strip()]
    if not lines:
        return None
    return "\n".join(lines[-max_lines:])


class ClaudeCliClient:
    """Run one-shot prompts through the Claude Code CLI (subscription billing).

    Each :meth:`run` shells out to ``claude -p <prompt> --system-prompt <system>
    --model <model> --output-format json`` and parses the single JSON result
    envelope the CLI prints. The subprocess gets ``stdin=DEVNULL`` and an
    environment with ``ANTHROPIC_API_KEY`` removed (see module docstring).
    """

    def __init__(
        self,
        claude_path: str = "claude",
        timeout_s: float = 300.0,
        runner: Runner | None = None,
    ) -> None:
        """Configure the client.

        Args:
            claude_path: Name or path of the CLI binary (found on ``PATH``).
            timeout_s: Hard wall-clock cap on a single call.
            runner: Test seam — a callable with ``subprocess.run``'s signature.
                Defaults to ``subprocess.run`` itself.
        """
        self._claude_path = claude_path
        self._timeout_s = timeout_s
        self._runner: Runner = runner if runner is not None else subprocess.run

    def run(self, system: str, prompt: str, model: str, max_tokens: int | None = None) -> LlmResult:
        """Prompt the model once and return the parsed result.

        Args:
            system: The system prompt (replaces Claude Code's default one).
            prompt: The user prompt.
            model: Model id or alias for ``--model`` (e.g. ``"claude-sonnet-5"``).
            max_tokens: Ignored — the print-mode CLI has no output-cap flag, and
                the forecast JSON we ask for is small. Kept in the signature so
                every provider shares one call shape.

        Returns:
            An :class:`LlmResult` with the text, token counts, and cost.

        Raises:
            LlmError: The CLI is not installed, exited non-zero, printed
                unparseable output, or reported an error in its envelope.
        """
        del max_tokens  # documented no-op: the print-mode CLI has no such flag.

        argv = [
            self._claude_path,
            "-p",
            prompt,
            "--system-prompt",
            system,
            "--model",
            model,
            "--output-format",
            "json",
        ]
        # Copy the environment minus the API key so billing stays on the
        # subscription no matter what the caller's shell has exported.
        env = {k: v for k, v in os.environ.items() if k != _API_KEY_ENV}

        try:
            completed = self._runner(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                env=env,
            )
        except FileNotFoundError as exc:
            raise LlmError(
                "Claude CLI not found — install Claude Code and log in "
                "(https://claude.com/claude-code), then re-run."
            ) from exc

        if completed.returncode != 0:
            raise LlmError(
                f"Claude CLI exited with code {completed.returncode}.",
                stderr=_stderr_tail(completed.stderr),
            )

        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise LlmError(
                "Claude CLI returned output that was not valid JSON.",
                stderr=_stderr_tail(completed.stderr),
            ) from exc
        if not isinstance(envelope, dict):
            raise LlmError(
                "Claude CLI returned JSON that was not a result object.",
                stderr=_stderr_tail(completed.stderr),
            )

        if envelope.get("is_error") or envelope.get("subtype") != "success":
            raise LlmError(
                "Claude CLI reported an error result.",
                envelope=envelope,
            )

        return _result_from_envelope(envelope, model)


def _result_from_envelope(envelope: dict[str, object], model: str) -> LlmResult:
    """Turn a successful CLI result envelope into an :class:`LlmResult`."""
    text = envelope.get("result")
    if not isinstance(text, str):
        raise LlmError(
            "Claude CLI success envelope had no 'result' text.",
            envelope=envelope,
        )

    usage = envelope.get("usage")
    input_tokens = 0
    output_tokens = 0
    if isinstance(usage, dict):
        input_tokens = _as_int(usage.get("input_tokens"))
        output_tokens = _as_int(usage.get("output_tokens"))

    cost_raw = envelope.get("total_cost_usd")
    cost_usd = float(cost_raw) if isinstance(cost_raw, int | float) else None

    stop_raw = envelope.get("stop_reason")
    stop_reason = stop_raw if isinstance(stop_raw, str) else None

    return LlmResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        stop_reason=stop_reason,
        model=model,
    )


def _as_int(value: object) -> int:
    """Coerce a JSON number to ``int``, defaulting to 0 for anything else."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    return 0


def build_llm_client(config: TaskModelConfig) -> LlmClient:
    """Return the client for a task's configured provider.

    This is the single dispatch point on ``provider``. Today only ``claude-cli``
    is wired up; a future Codex/OpenRouter client is a new branch here.

    Raises:
        LlmError: If the provider name is not one we support.
    """
    if config.provider == "claude-cli":
        return ClaudeCliClient()
    raise LlmError(
        f"Unknown LLM provider {config.provider!r}; "
        f"supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def check_claude_cli(claude_path: str = "claude") -> bool:
    """Return whether the Claude CLI binary is findable on ``PATH``."""
    return shutil.which(claude_path) is not None


def check_provider_cli(config: TaskModelConfig) -> bool:
    """Return whether the CLI binary the task's provider needs is on ``PATH``.

    Raises:
        LlmError: If the provider name is not one we support.
    """
    if config.provider == "claude-cli":
        # Delegate so tests can monkeypatch a single seam, ``check_claude_cli``.
        return check_claude_cli(_PROVIDER_BINARIES["claude-cli"])
    if config.provider not in _PROVIDER_BINARIES:
        raise LlmError(
            f"Unknown LLM provider {config.provider!r}; "
            f"supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return shutil.which(_PROVIDER_BINARIES[config.provider]) is not None
