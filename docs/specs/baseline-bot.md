# Spec: Baseline Tournament Bot (Candidate A, v0)

- Status: in progress
- Date: 2026-07-09
- Parent docs: [VISION.md](../VISION.md) (validation rung 1), [03-architecture-candidates.md](../03-architecture-candidates.md) (candidate A)

## Goal

A live-scored entry in the Metaculus AI Benchmark: the simplest honest forecaster — one Claude call following the superforecaster checklist — registered, submitting, and accumulating a real track record. Rung 1 of the validation ladder is "infrastructure works," not "forecasts well."

## Scope — v0

- **Forecasting pass (candidate A, minus retrieval):** one LLM call per question — checklist prompt (reference class → base rate anchor → adjust → argue against yourself) over the question's text/criteria, returning probability + confidence + rationale via structured output. **No web retrieval yet** — that lands with the search-provider decision (harness spec checkpoint C2). This is explicitly a floor, not the plan.
- **Model config:** per-task-type model + effort read from `engine/config/models.toml` — the first instance of the config-over-code requirement. Baseline pass defaults to Sonnet.
- **Forecast store:** every forecast (incl. dry runs) persisted with its full trace (prompt, rationale, model config, tokens/cost) — feeds postmortems and backtests later.
- **Submission client:** POST forecasts to Metaculus (endpoint per recon), gated by an explicit `--live` flag; default is dry-run.
- **CLI:** `forecast` (run passes over open tournament questions, dry-run default), `submit` (only after checkpoint B2).

## Out of scope (v0)

Retrieval, ensembles, market anchoring (CP is account-gated anyway), any other candidate, automation/scheduling (manual kickoff per vision).

## Declared checkpoints

| # | Gate | Artifact / user action |
|---|---|---|
| **B1** | After recon | Registration instructions — likely requires the owner to create a dedicated Metaculus **bot account** and register it for the AIB tournament; bot token goes in `engine/.env` |
| **B2** | Before any live submission | Dry-run forecasts on real tournament questions, with rationales, reviewed by the owner |
| **B3** | Live | First submitted forecasts confirmed on the tournament page |

## New external dependency

Forecasting passes call the Anthropic API — needs `ANTHROPIC_API_KEY` in `engine/.env` and incurs per-call cost (~$0.17/question at Sonnet, logged per the cost-visibility requirement).

## Agent execution plan

1. **Sonnet** — recon: AIB current season, bot registration flow, tournament question listing, forecast-submission endpoint + payload, bot-account rules.
2. **Opus** — forecasting module: model config loader, checklist prompt, Anthropic client wrapper (cost logging), forecast store, `forecast` CLI (dry-run).
3. **Opus** — submission client + `submit` CLI, from recon findings (after 1).
4. **Fable** — review, verify, checkpoint B1/B2 presentations.
