# 0003 — Multi-provider model layer: Claude CLI + Codex CLI, OpenRouter planned

- Status: accepted
- Date: 2026-07-09
- Decided by: human

## Context

The vision requires trialing multiple models per task type, and the architecture portfolio benefits from cross-provider ensemble diversity (passes from different model families share less training data → better pooling). The owner holds both a Claude Code subscription (ADR-0002) and a Codex subscription, and anticipates needing broader model access later.

## Decision

The LLM layer (`llm.py`) is a **provider registry behind one protocol**: `LlmClient.run(system, prompt, model, max_tokens, json_schema) -> LlmResult`, dispatched by the per-task `provider` field in `config/models.toml`.

Providers, in order of adoption:
1. **`claude-cli`** — Claude subscription (ADR-0002). Ignores `json_schema` (prompt-JSON + parse).
2. **`codex-cli`** — Codex subscription via `codex exec` headless mode (verified live, codex-cli 0.140.0): JSONL events (`item.completed` → `agent_message` text; `turn.completed` → usage), `--output-schema <file>` for schema-constrained final output, `-m` for model, read-only sandbox, `stdin=DEVNULL` mandatory (blocks otherwise), run in a neutral empty cwd (else it ingests repo context — ~18k tokens observed on a trivial prompt).
3. **`openrouter`** (planned, not yet built) — metered API for model breadth beyond the two subscriptions (Gemini, Llama, etc.), wanted for maximum ensemble diversity. Will be an HTTP client behind the same protocol; requires its own key and per-model pricing entries.

## Alternatives considered

- **Single provider (Claude only):** simpler, but forfeits cross-family ensemble diversity — one of the strongest levers the evidence offers — and concentrates quota risk on one subscription.
- **A framework (LiteLLM/LangChain) as the abstraction:** heavy dependency for what is a ~300-line protocol + registry; our seam is already proven by three implementations (SDK, Claude CLI, fakes).

## Consequences

- The `json_schema` parameter is optional capability, not contract: providers that support schema-constrained output use it; others fall back to prompt-JSON + validation retry. The forecaster always validates regardless.
- Codex usage reports tokens but no cost — `cost_usd` is computed from the pricing table when configured, else `None` (both subscriptions make cost notional anyway).
- Ensemble configs can now mix providers per pass — the "different model" diversity axis in candidate B becomes real.
- Each CLI provider adds a host dependency (binary installed + logged in), checked per-provider at startup.
