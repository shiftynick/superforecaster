"""Call an LLM through a local CLI, billed to the owner's subscription.

The engine used to call the Anthropic API SDK, which bills a separately-metered
API key. This module replaces that with a subprocess wrapper around the **Claude
Code CLI** (``claude -p ... --output-format json``), so forecast passes draw on
the owner's Claude subscription instead. See ``docs/adr/0002-llm-via-claude-cli``.

The design is deliberately provider-neutral, so providers of different shapes
coexist behind one interface: two are local CLIs (:class:`ClaudeCliClient`,
:class:`CodexCliClient`, subprocess-billed to the owner's subscriptions) and one
is a plain HTTPS API (:class:`OpenRouterClient`, using ``httpx`` with an
injectable transport like the search/metaculus clients — no CLI involved).

- :class:`LlmResult` and :class:`LlmError` carry no provider-specific detail.
- :class:`LlmClient` is the structural type every client satisfies: a single
  ``run(system, prompt, model, max_tokens, json_schema)`` method returning an
  :class:`LlmResult`. ``json_schema`` is optional *capability*, not contract:
  providers that support schema-constrained output (Codex CLI, OpenRouter) use
  it; the Claude CLI ignores it and relies on prompt-JSON + validation retry
  upstream. ``max_tokens`` is likewise capability: honored by OpenRouter, a
  documented no-op for both CLIs.
- :func:`build_llm_client` is the one factory the rest of the engine calls; it
  dispatches on the task's ``provider`` field and is the only place that knows
  which concrete client class a provider name maps to.

Two subscription-billing safeguards live in :class:`ClaudeCliClient`:

- ``ANTHROPIC_API_KEY`` is stripped from the subprocess environment, so a stray
  key can never silently redirect billing to the metered API.
- The user prompt is piped via ``stdin`` (never argv): researched prompts carry
  multi-source context and overflow Windows' ~32k command-line limit as an
  argument. The pipe is closed after writing, so neither CLI hangs on it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from .config import TaskModelConfig, price_for

# The env var that, if present, would bill the metered API instead of the
# subscription. We remove it from the child process's environment.
_API_KEY_ENV = "ANTHROPIC_API_KEY"

# Provider name -> the CLI binary it shells out to. Adding a *CLI* provider is a
# new entry here plus a branch in build_llm_client. OpenRouter is not a CLI (it
# is a plain HTTPS API), so it is deliberately absent from this map and handled
# as its own branch throughout.
_PROVIDER_BINARIES: dict[str, str] = {"claude-cli": "claude", "codex-cli": "codex"}

# The HTTP (non-CLI) provider name.
OPENROUTER_PROVIDER = "openrouter"

# Env var holding the OpenRouter key (loaded from engine/.env by the CLI layer).
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

# OpenRouter's OpenAI-compatible chat-completions endpoint.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Human-facing tuple of what we support, for error messages.
SUPPORTED_PROVIDERS: tuple[str, ...] = tuple(sorted([*_PROVIDER_BINARIES, OPENROUTER_PROVIDER]))


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
    :class:`LlmResult`. ``ClaudeCliClient`` and ``CodexCliClient`` satisfy this,
    and so does any fake used in tests, so the forecaster never depends on a
    concrete class.
    """

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
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

    Each :meth:`run` shells out to ``claude -p --system-prompt <system>
    --model <model> --output-format json`` with the user prompt piped to stdin,
    and parses the single JSON result envelope the CLI prints. The subprocess
    environment has ``ANTHROPIC_API_KEY`` removed (see module docstring).
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

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        """Prompt the model once and return the parsed result.

        Args:
            system: The system prompt (replaces Claude Code's default one).
            prompt: The user prompt.
            model: Model id or alias for ``--model`` (e.g. ``"claude-sonnet-5"``).
            max_tokens: Ignored — the print-mode CLI has no output-cap flag, and
                the forecast JSON we ask for is small. Kept in the signature so
                every provider shares one call shape.
            json_schema: Ignored — the Claude CLI path uses prompt-JSON plus a
                validation retry, not server-side schema enforcement. Accepted so
                every provider shares one call shape; schema is advisory here.
            effort: Optional reasoning-effort level. When set, passed to the CLI
                as ``--effort <value>`` (verified accepted with "low"). ``None``
                omits the flag, leaving the CLI's own default.

        Returns:
            An :class:`LlmResult` with the text, token counts, and cost.

        Raises:
            LlmError: The CLI is not installed, exited non-zero, printed
                unparseable output, or reported an error in its envelope.
        """
        del max_tokens  # documented no-op: the print-mode CLI has no such flag.
        del json_schema  # documented no-op: schema is advisory for this provider.

        # The user prompt goes in via STDIN, not argv: researched prompts carry
        # multi-source context and blow past Windows' ~32k CreateProcess
        # command-line limit as an argument (seen live: WinError 206). `claude -p`
        # with no positional prompt reads the prompt from stdin; subprocess closes
        # the pipe after writing, so the CLI still terminates normally.
        argv = [
            self._claude_path,
            "-p",
            "--system-prompt",
            system,
            "--output-format",
            "json",
        ]
        # Empty model means "use the CLI's own default model" — same convention
        # as the Codex client, so config treats "" uniformly across providers.
        if model:
            argv += ["--model", model]
        if effort:
            argv += ["--effort", effort]
        # Copy the environment minus the API key so billing stays on the
        # subscription no matter what the caller's shell has exported.
        env = {k: v for k, v in os.environ.items() if k != _API_KEY_ENV}

        try:
            completed = self._runner(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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


class CodexCliClient:
    """Run one-shot prompts through the Codex CLI (owner's Codex subscription).

    Each :meth:`run` shells out to ``codex exec - --skip-git-repo-check
    -s read-only --json`` with the prompt piped to stdin, and parses the JSONL
    event stream it prints. Codex has no separate system-prompt flag in ``exec``
    mode, so the system and user prompts are concatenated with a delimiter.

    Three facts about ``codex exec`` shape this client (verified live on
    codex-cli 0.140.0):

    - It **blocks forever if stdin stays open** — fine here: subprocess writes
      the piped prompt and closes the pipe (``exec -`` reads stdin to EOF).
    - It **ingests the working directory's context** (``AGENTS.md`` and friends —
      ~18k input tokens on a trivial prompt when run inside a repo), so each call
      runs in a fresh empty temp directory that is removed afterwards.
    - ``--output-schema <file>`` constrains the *final* agent message to JSON
      matching a JSON Schema; we write ``json_schema`` to a temp file and pass it
      when supplied. This is Codex's server-side answer to structured output.

    We deliberately do **not** touch ``OPENAI_API_KEY``: Codex authenticates from
    its own stored subscription login (already configured on this machine), and
    there is no evidence a stray env key silently redirects billing the way
    ``ANTHROPIC_API_KEY`` does for the Claude API. The child inherits the
    environment unchanged.

    Codex reports token usage but no cost, so ``cost_usd`` is the priced-token
    fallback from ``config.price_for`` when the model is priced, else ``None``.
    """

    def __init__(
        self,
        codex_path: str = "codex",
        timeout_s: float = 300.0,
        runner: Runner | None = None,
    ) -> None:
        """Configure the client.

        Args:
            codex_path: Name or path of the CLI binary (found on ``PATH``).
            timeout_s: Hard wall-clock cap on a single call.
            runner: Test seam — a callable with ``subprocess.run``'s signature.
                Defaults to ``subprocess.run`` itself.
        """
        self._codex_path = codex_path
        self._timeout_s = timeout_s
        self._runner: Runner = runner if runner is not None else subprocess.run

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        """Prompt the model once and return the parsed result.

        Args:
            system: The system prompt (prepended to ``prompt`` with a delimiter,
                since ``codex exec`` has no separate system-prompt flag).
            prompt: The user prompt.
            model: Model id for ``-m``. Passed only when non-empty; an empty
                string means "use the Codex CLI's own default model".
            max_tokens: Ignored — ``codex exec`` has no output-cap flag. Kept in
                the signature so every provider shares one call shape.
            json_schema: If given, written to a temp file and passed via
                ``--output-schema`` to constrain the final message to JSON.
            effort: Optional reasoning-effort level. When set, passed as the Codex
                config override ``-c model_reasoning_effort="<value>"`` (verified
                accepted with "low"). The quotes are part of the value — the CLI
                TOML-parses the override, so ``"low"`` is the string ``low``.
                ``None`` omits the override, leaving the CLI's own default.

        Returns:
            An :class:`LlmResult` with the text, token counts, and cost.

        Raises:
            LlmError: The CLI is not installed, exited non-zero, or emitted no
                ``agent_message`` event.
        """
        del max_tokens  # documented no-op: codex exec has no output-cap flag.

        combined_prompt = f"{system}\n\n---\n\n{prompt}"
        # Prompt goes in via STDIN (`codex exec -` reads it there): as an argv
        # element, researched prompts exceed Windows' ~32k command-line limit
        # (WinError 206). subprocess writes the input then closes the pipe, so
        # the "blocks forever if stdin stays open" hazard does not apply.
        argv = [
            self._codex_path,
            "exec",
            "-",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "--json",
        ]
        if model:
            argv += ["-m", model]
        if effort:
            argv += ["-c", f'model_reasoning_effort="{effort}"']

        # Run in a throwaway empty directory so Codex does not ingest repo
        # context, and (optionally) drop the schema in a temp file. Both are
        # cleaned up in the finally, whatever happens.
        workdir = tempfile.mkdtemp(prefix="sf-codex-")
        schema_file: str | None = None
        try:
            if json_schema is not None:
                json_schema = _strictify_schema(json_schema)
                fd, schema_file = tempfile.mkstemp(prefix="sf-codex-schema-", suffix=".json")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(json_schema, fh)
                argv += ["--output-schema", schema_file]

            try:
                completed = self._runner(
                    argv,
                    input=combined_prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self._timeout_s,
                    cwd=workdir,
                )
            except FileNotFoundError as exc:
                raise LlmError(
                    "Codex CLI not found — install the Codex CLI and log in with "
                    "your Codex subscription, then re-run."
                ) from exc
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            if schema_file is not None:
                Path(schema_file).unlink(missing_ok=True)

        if completed.returncode != 0:
            raise LlmError(
                f"Codex CLI exited with code {completed.returncode}.",
                stderr=_stderr_tail(completed.stderr),
            )

        return _result_from_jsonl(completed.stdout, model, completed.stderr)


def _result_from_jsonl(stdout: str, model: str, stderr: str) -> LlmResult:
    """Parse Codex's JSONL event stream into an :class:`LlmResult`.

    The answer is the text of the **last** ``item.completed`` event whose item
    type is ``agent_message`` (earlier items may be reasoning or tool calls).
    Usage comes from the ``turn.completed`` event. Lines that are not valid JSON
    (stray CLI notices printed on stdout) are skipped rather than fatal.
    """
    agent_text: str | None = None
    usage: dict[str, object] | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue  # tolerate non-JSON noise interleaved on stdout
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    agent_text = text  # last agent_message wins
        elif etype == "turn.completed":
            turn_usage = event.get("usage")
            if isinstance(turn_usage, dict):
                usage = turn_usage

    if agent_text is None:
        raise LlmError(
            "Codex CLI produced no agent_message in its JSONL output.",
            stderr=_stderr_tail(stderr),
        )

    input_tokens = _as_int(usage.get("input_tokens")) if usage is not None else 0
    output_tokens = _as_int(usage.get("output_tokens")) if usage is not None else 0

    # Codex reports no cost; fall back to the pricing table (None if unpriced).
    cost_usd = price_for(model, input_tokens, output_tokens) if model else None

    return LlmResult(
        text=agent_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        stop_reason=None,
        model=model,
    )


def _strictify_schema(schema: dict[str, object]) -> dict[str, object]:
    """Return a copy of a JSON Schema adjusted for OpenAI strict mode.

    Codex validates ``--output-schema`` against OpenAI's strict rules, which
    require every ``object`` node to carry ``"additionalProperties": false``.
    Pydantic's ``model_json_schema()`` omits that, so we add it recursively.
    The input is not mutated.
    """

    def walk(node: object) -> object:
        if isinstance(node, dict):
            out = {key: walk(value) for key, value in node.items()}
            if out.get("type") == "object":
                out.setdefault("additionalProperties", False)
            return out
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    result = walk(schema)
    assert isinstance(result, dict)  # a dict in is a dict out; narrows for mypy
    return result


def _http_body_tail(text: str, max_chars: int = 500) -> str | None:
    """Return a compact tail of an HTTP error body for diagnostics, or ``None``."""
    stripped = text.strip()
    if not stripped:
        return None
    if len(stripped) <= max_chars:
        return stripped
    return "..." + stripped[-max_chars:]


# Substrings in an OpenRouter 4xx error body that mean "this model does not
# support server-side structured output" — the signal to retry without a schema.
_SCHEMA_UNSUPPORTED_MARKERS: tuple[str, ...] = (
    "response_format",
    "json_schema",
    "structured output",
    "structured outputs",
    "schema",
)


def _is_schema_unsupported(body_text: str) -> bool:
    """Heuristic: does a 4xx body say response_format/structured output is unsupported?"""
    lowered = body_text.lower()
    return any(marker in lowered for marker in _SCHEMA_UNSUPPORTED_MARKERS)


class OpenRouterClient:
    """Call a model through OpenRouter's OpenAI-compatible chat-completions API.

    Unlike :class:`ClaudeCliClient` and :class:`CodexCliClient`, this provider is
    **not** a local CLI: it is a plain HTTPS endpoint. So this client uses
    ``httpx`` directly with an injectable transport (the same ``MockTransport``
    test seam :mod:`superforecaster.search` and :mod:`superforecaster.metaculus`
    use), not a subprocess ``runner``. Each :meth:`run` POSTs one request to
    ``/chat/completions`` and parses the single JSON reply.

    Behaviours that set this client apart from the CLI ones:

    - **``max_tokens`` is honored.** This is the first provider that actually
      caps output length — it is sent as the request's ``max_tokens`` field.
      (The two CLIs have no output-cap flag and document it as a no-op.)
    - **No default model.** OpenRouter routes by explicit model id and has no
      "use my default" notion, so an empty ``model`` is an error here — the
      opposite of the CLIs, where ``""`` means "the CLI's own default model".
    - **Real-dollar cost.** OpenRouter always returns ``usage.cost`` (real
      account charge in USD), so ``cost_usd`` comes straight from the API with no
      ``price_for`` guess needed; the pricing table is only a fallback for the
      rare response that omits it. Tournament credits donated by Metaculus land
      on the OpenRouter key, so this cost is the true draw against them.
    - **``effort`` maps to OpenRouter's ``reasoning`` param** (``{"effort": ...}``),
      omitted entirely when ``None``.
    - **``json_schema`` maps to ``response_format`` structured output**, but not
      every model supports it. On a 4xx whose body names response_format /
      structured output as the problem, we retry the call **once** without the
      schema (callers still have prompt-JSON + validation retry upstream as the
      final backstop) before surfacing an :class:`LlmError`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        """Configure the client.

        Args:
            api_key: The OpenRouter key. When ``None``, read from
                ``OPENROUTER_API_KEY`` in the environment (loading ``engine/.env``
                is the CLI layer's job — this only reads ``os.environ``). A
                constructor arg overrides the env, mirroring the search clients.
            transport: Optional ``httpx`` transport (tests inject a
                ``MockTransport`` so no real network is touched).
            timeout_s: Per-request HTTP timeout, matching the CLI clients' cap.

        Raises:
            LlmError: No key was passed and none is set in the environment.
        """
        key = api_key if api_key is not None else os.environ.get(OPENROUTER_API_KEY_ENV)
        if not key:
            raise LlmError(
                f"Missing {OPENROUTER_API_KEY_ENV} in the environment; "
                f"set it in engine/.env to use the OpenRouter provider."
            )
        self._client = httpx.Client(
            base_url=_OPENROUTER_BASE_URL,
            transport=transport,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=timeout_s,
        )

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        """Prompt the model once via OpenRouter and return the parsed result.

        Args:
            system: The system prompt (sent as a ``role: "system"`` message).
            prompt: The user prompt (sent as a ``role: "user"`` message).
            model: OpenRouter model id (e.g. ``"openai/gpt-4o-mini"``). Required —
                OpenRouter has no default model, so an empty string is an error.
            max_tokens: Honored here (unlike the CLIs) — sent as ``max_tokens``.
                ``None`` leaves output length uncapped.
            json_schema: If given, sent as a ``response_format`` json_schema
                (strict) so the model returns schema-constrained JSON. Not every
                model supports this; an unsupported-schema 4xx triggers one retry
                without the schema before failing.
            effort: Optional reasoning-effort level, mapped to OpenRouter's
                ``reasoning: {"effort": <value>}``. ``None`` omits it.

        Returns:
            An :class:`LlmResult` with the text, token counts, and real cost.

        Raises:
            LlmError: Empty model, the HTTP call failed, a non-2xx response, or a
                malformed body.
        """
        if not model:
            raise LlmError(
                "OpenRouter requires an explicit model id (it has no default "
                "model); set 'model' in the task config."
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            # Usage accounting: usage.cost is always returned now, so this flag is
            # a documented no-op, but it is harmless and states intent explicitly.
            "usage": {"include": True},
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if effort:
            body["reasoning"] = {"effort": effort}
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "strict": True,
                    "schema": _strictify_schema(json_schema),
                },
            }

        response = self._post(body)

        # One cheap resilience step: if the model rejected the schema, drop it and
        # try once more. Upstream prompt-JSON + validation retry is the backstop.
        if (
            json_schema is not None
            and 400 <= response.status_code < 500
            and _is_schema_unsupported(response.text)
        ):
            body.pop("response_format", None)
            response = self._post(body)

        if response.status_code >= 300:
            raise LlmError(
                f"OpenRouter returned HTTP {response.status_code}.",
                stderr=_http_body_tail(response.text),
            )

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LlmError(
                "OpenRouter returned a body that was not valid JSON.",
                stderr=_http_body_tail(response.text),
            ) from exc
        if not isinstance(payload, dict):
            raise LlmError(
                "OpenRouter returned JSON that was not a result object.",
                stderr=_http_body_tail(response.text),
            )

        return _result_from_openrouter(payload, model)

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        """POST one request body to ``/chat/completions``, mapping transport errors."""
        try:
            return self._client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise LlmError(f"OpenRouter request failed: {exc}") from exc


def _result_from_openrouter(payload: dict[str, Any], model: str) -> LlmResult:
    """Turn an OpenRouter chat-completions reply into an :class:`LlmResult`.

    Text comes from ``choices[0].message.content``; token counts from
    ``usage.prompt_tokens``/``completion_tokens``; ``stop_reason`` from
    ``choices[0].finish_reason``. Cost is the real-dollar ``usage.cost`` when the
    API reports it, else the priced-token fallback from ``config.price_for``.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LlmError("OpenRouter response had no choices.", envelope=payload)
    choice = choices[0]

    message = choice.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str):
        raise LlmError("OpenRouter choice had no message content text.", envelope=payload)

    stop_raw = choice.get("finish_reason")
    stop_reason = stop_raw if isinstance(stop_raw, str) else None

    usage = payload.get("usage")
    input_tokens = 0
    output_tokens = 0
    cost_usd: float | None = None
    if isinstance(usage, dict):
        input_tokens = _as_int(usage.get("prompt_tokens"))
        output_tokens = _as_int(usage.get("completion_tokens"))
        cost_raw = usage.get("cost")
        if isinstance(cost_raw, int | float) and not isinstance(cost_raw, bool):
            cost_usd = float(cost_raw)

    # Real-dollar cost from the API needs no guess; only fall back to the pricing
    # table when the API omitted cost (rare).
    if cost_usd is None:
        cost_usd = price_for(model, input_tokens, output_tokens)

    return LlmResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        stop_reason=stop_reason,
        model=model,
    )


def build_llm_client(config: TaskModelConfig) -> LlmClient:
    """Return the client for a task's configured provider.

    This is the single dispatch point on ``provider``. Supported today:
    ``claude-cli``, ``codex-cli`` (local CLIs), and ``openrouter`` (HTTPS API).

    Raises:
        LlmError: If the provider name is not one we support.
    """
    if config.provider == "claude-cli":
        return ClaudeCliClient()
    if config.provider == "codex-cli":
        return CodexCliClient()
    if config.provider == OPENROUTER_PROVIDER:
        return OpenRouterClient()
    raise LlmError(
        f"Unknown LLM provider {config.provider!r}; "
        f"supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def check_claude_cli(claude_path: str = "claude") -> bool:
    """Return whether the Claude CLI binary is findable on ``PATH``."""
    return shutil.which(claude_path) is not None


def check_codex_cli(codex_path: str = "codex") -> bool:
    """Return whether the Codex CLI binary is findable on ``PATH``."""
    return shutil.which(codex_path) is not None


def check_provider_cli(config: TaskModelConfig) -> bool:
    """Return whether the task's provider is ready to use on this machine.

    For the CLI providers this means "is the CLI binary on ``PATH``". OpenRouter
    is not a CLI — it has no binary to find — so "ready" instead means "is
    ``OPENROUTER_API_KEY`` set in the environment" (loading ``engine/.env`` is the
    CLI layer's job; this only inspects ``os.environ``).

    Raises:
        LlmError: If the provider name is not one we support.
    """
    if config.provider == "claude-cli":
        # Delegate so tests can monkeypatch a single seam, ``check_claude_cli``.
        return check_claude_cli(_PROVIDER_BINARIES["claude-cli"])
    if config.provider == "codex-cli":
        # Delegate so tests can monkeypatch a single seam, ``check_codex_cli``.
        return check_codex_cli(_PROVIDER_BINARIES["codex-cli"])
    if config.provider == OPENROUTER_PROVIDER:
        # No CLI to locate: readiness is "the API key is present".
        return bool(os.environ.get(OPENROUTER_API_KEY_ENV))
    if config.provider not in _PROVIDER_BINARIES:
        raise LlmError(
            f"Unknown LLM provider {config.provider!r}; "
            f"supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return shutil.which(_PROVIDER_BINARIES[config.provider]) is not None
