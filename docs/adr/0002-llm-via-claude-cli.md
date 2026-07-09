# 0002 — LLM calls via Claude Code CLI wrapper (subscription), not the API SDK

- Status: accepted
- Date: 2026-07-09
- Decided by: human

## Context

The forecasting passes need LLM calls. The default build used the Anthropic Python SDK, which requires a separately-billed API key. The owner has a Claude Code subscription and wants engine LLM usage to draw on it instead. Claude Code's documented headless mode (`claude -p "<prompt>" --output-format json`) supports exactly this: one-shot prompting, model selection (`--model sonnet|opus`), system-prompt override, and a JSON result envelope carrying the text plus token usage.

## Decision

Engine LLM calls go through a **subprocess wrapper around the Claude CLI** (`llm.py`), which implements the same client seam `run_forecast_pass` already uses. Structured output is achieved by prompting for JSON and validating with pydantic (one retry with the validation error appended). Token counts and the CLI-reported cost are recorded per the cost-visibility rule (cost is notional under subscription — it measures quota draw, not dollars billed).

## Alternatives considered

- **Anthropic API SDK** (the initial build): cleaner structured outputs (`messages.parse`), lower latency, but separately metered billing the owner doesn't want. The client seam means switching back later is a one-module swap — this stays the designated escape hatch if volume or reliability demands it.
- **AIB free LLM credits via OpenRouter** (offered to tournament participants): viable fallback for tournament traffic specifically; adds a third-party dependency and another key. Not needed now.

## Consequences

- Engine requires the `claude` CLI installed and logged in on the host — an environment dependency, checked at startup with a clear error.
- No server-side schema enforcement: JSON parsing failures are handled with a single retry, then surfaced as errors (never silently defaulted).
- Bot cadence shares the subscription's usage quota with interactive Claude Code work — forecast passes default to Sonnet, and per-run caps matter more.
- `ANTHROPIC_API_KEY` is no longer required for forecasting (and must NOT be set in the CLI's environment, or it would bill the API instead of the subscription).
