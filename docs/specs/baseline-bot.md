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

## Recon findings (verified 2026-07-09)

- **Season:** Summer 2026 FutureEval Bot Tournament — id `33022`, slug `summer-futureeval-2026`, $50k pool, forecasting ends 2026-09-06, ongoing now. Test sandbox: `bot-testing-area` (id `32977`, unscored).
- **Registration:** owner creates a bot from account settings → "My Forecasting Bots" → "Create a Bot"; the bot gets its own API token (that token, not the human one, is what submits). No approval step — first API forecast *is* entry.
- **Submission:** `POST /api/questions/forecast/` with `[{"question": <qid>, "probability_yes": p}]` (clamp 0.001–0.999); a rationale comment is **mandatory** per forecast: `POST /api/comments/create/` `{"on_post", "text", "is_private": true, "included_forecast": true}` (auto-published after close).
- **Rule — no human in the loop:** the owner may not preview/approve/tweak the bot's forecasts on live tournament questions. Human-triggered *runs* are fine; human-filtered *forecasts* are not (prize eligibility requires passing an inspection of exactly this).
- **Rule — one forecast per question**, no updating; scoring is spot peer score (log-based) on the last pre-close forecast.
- **Rule — brutal timing:** questions open at random hours in batches and stay open **~1.5 hours**. The reference bot polls every 20 minutes. Manual-kickoff-only participation would miss nearly everything.

## Declared checkpoints (revised per recon — the no-human-in-the-loop rule reshapes B2/B3)

| # | Gate | Artifact / user action |
|---|---|---|
| **B1** | After recon | Registration instructions (above) — owner creates the bot, puts its token in `engine/.env` as `METACULUS_BOT_TOKEN` |
| **B2** | Pipeline validation | Dry-run forecasts + rationales reviewed by owner on **non-tournament questions** (our scraped store and/or the unscored `bot-testing-area`) — human review is legal there and validates prompt quality without touching live questions |
| **B3** | Autonomy decision | Owner decides the automation question (below) and green-lights live entry; from then on the bot runs **fully autonomously** on tournament questions — no per-forecast human review, by rule |

## Open decision for the owner (at B3)

The 1.5-hour question windows mean real participation needs a **scheduled poller** (~every 20 min), which collides with the "manual kickoffs only" cost-control decision. Options: (a) scheduled task with hard cost caps (max forecasts/day, Sonnet-only — worst case a few dollars/day at ~5 questions/batch); (b) manual-only participation accepting we miss most questions (rung 1 becomes "infrastructure proven" rather than "meaningful score"); (c) defer live entry, validate on bot-testing-area only. Recommendation: (a) with caps.

## New external dependency (revised 2026-07-09, ADR-0002)

Forecasting passes call Claude through a **Claude Code CLI wrapper** using the owner's subscription — no `ANTHROPIC_API_KEY` needed. Requires the `claude` CLI installed and logged in. Token usage still logged per call (cost is notional quota draw, not billed dollars).

**Token note (owner, 2026-07-09):** the `METACULUS_TOKEN` already in `engine/.env` is the **bot account's** token (Metaculus exposes no separate human token). It serves both reading and submitting; `METACULUS_BOT_TOKEN` remains an optional override. This also explains the CP gating — bot accounts are walled off from community predictions.

## Agent execution plan

1. **Sonnet** — recon: AIB current season, bot registration flow, tournament question listing, forecast-submission endpoint + payload, bot-account rules.
2. **Opus** — forecasting module: model config loader, checklist prompt, Anthropic client wrapper (cost logging), forecast store, `forecast` CLI (dry-run).
3. **Opus** — submission client + `submit` CLI, from recon findings (after 1).
4. **Fable** — review, verify, checkpoint B1/B2 presentations.
