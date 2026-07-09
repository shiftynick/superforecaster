# What We're Building

*One-page summary of the target system. Evidence citations live in [01-prior-art.md](01-prior-art.md); undecided parts are marked as such. Revised 2026-07-09 after decision review ([VISION-review.html](VISION-review.html)).*

## In one sentence

A personal forecasting system: give it a question about the future and it researches, then returns a **calibrated probability, a confidence level, and a legible argument** — flexible enough to serve as a decision-support tool, a research assistant, and a tournament competitor.

## Three use modes, one engine

1. **Decision support** — forecast questions the user personally cares about ("probability of outcome X"), on demand.
2. **Forecast reports** — research-assistant mode: the full evidence trail, reasoning, and probability packaged as a readable report. The explanation is a first-class deliverable (for the user's own consumption).
3. **Tournament competitor** — entered in the Metaculus AI Benchmark. This mode exists to keep us honest: it is the only objective, external scorecard for whether the engine actually forecasts well. The public ladder is validation, not the mission.

## What it does, end to end

1. **Ingest** a question (Metaculus/Kalshi API or user-posed) with resolution criteria and deadline.
2. **Research**: agentic news/web retrieval, date-filtered so the agent only sees information from before its forecast time. Sources carry **quality/believability weights** that are tunable configuration, not hard-coded logic.
3. **Forecast**: multiple diverse passes (different reasoning orders, personas, retrieval seeds), each anchored outside-view-first (reference class → base rate → adjust).
4. **Aggregate**: pool the passes in log-odds space with a tuned (mild) extremization coefficient, weight members by tracked performance; apply calibration (Platt scaling) as a final layer.
5. **Integrate market data**: where a market/community price exists, it enters as **another data point — potentially heavily weighted, but never the sole reliance**. The weight is learned from backtests, not assumed. (Evidence note: the best published systems win as roughly ⅓-model/⅔-market blends, so a heavy learned weight is the likely outcome — but it stays a weight, not an anchor we're chained to.)
6. **Report confidence**: every forecast carries an explicit confidence level. Below a tunable threshold the system says so — in personal modes it flags "low confidence" rather than refusing; in tournament mode it submits the community/market prior instead of a low-signal guess.
7. **Update on demand**: re-forecasting is core to accuracy (strongest human predictor), but runs are **manually kicked off for now** — no automations — so costs stay visible and controlled.
8. **Explain**: return the reasoning behind the number, written for the user.
9. **Learn from misses (postmortems)**: after a question resolves badly — a large scoring miss, or confidence that didn't match reality — a deep post-hoc analysis reconstructs *why*: was evidence missed, misread, badly combined, or was the outcome genuinely unforeseeable? Each postmortem produces an **addressable lesson** routed to the specific tunable it implicates (source weights, prompts, aggregation coefficients, calibration curve), and lessons are treated as hypotheses to verify on backtest before adoption. Protocol: [02-evaluation-harness.md](02-evaluation-harness.md) § Postmortems.

## How it gets built

The system will be built with **heavy use of agentic AI** — most code agent-written, human-steered — under a disciplined SDLC: spec → plan → implement → agent review → verify → decision record, with project-specific Claude Code skills encoding the recurring workflows. Full approach: [04-development-approach.md](04-development-approach.md).

## System properties (cross-cutting requirements)

- **Model configurability**: every task type (retrieval, forecasting pass, aggregation/supervision, verification) has its own model + reasoning-effort config, swappable without code changes — so we can trial models and thinking levels cheaply and pin what wins.
- **Tunable everything**: source weights, ensemble size, extremization coefficient, confidence thresholds, prompts — all live in configuration intended to be adjusted as the system improves, not buried in code.
- **Cost visibility**: per-forecast token/cost tracking; manual kickoffs until the cost profile is understood. Rough per-question costs at current pricing (verify before relying): a single retrieval-augmented pass ≈ $0.05 (Haiku) / $0.17 (Sonnet) / $0.70 (Opus); an 8-pass Sonnet ensemble ≈ $1.40/question; Batch API halves non-urgent runs.
- **Web interface**: a simple web application for working with the system — browsing questions, reading research traces and forecast reports, kicking off runs, adjusting configs. Interface development stays flexible; the engine is API-first so the UI is a client, not the core.

## What it is NOT (current priorities, not permanent principles)

- **No model training/fine-tuning** — not a permanent principle, but not a priority now; the frontier is scaffolds.
- **No trading** — the score is calibration against resolved questions, not P&L.
- **Long-horizon questions** ("AGI by 2035?") — not a priority now, but the design must not preclude supporting them later; we start where feedback and reference classes exist.

## How we'll know it works

Everything is adjudicated by the evaluation harness ([02-evaluation-harness.md](02-evaluation-harness.md)): leakage-hardened backtests for iteration speed, live Metaculus scoring as ground truth. The validation ladder:

1. Baseline bot live and scored (infrastructure works)
2. Beat baseline A with the ensemble, statistically significant on backtest
3. Beat the median AIB bot on live scoring
4. Beat community prediction at forecast time on a question subset (selective)
5. Approach Pro-human parity (current frontier gap: Pros beat all bots, −20 head-to-head as of Q2 2025)

Personal-tool success is harder to score but rides on the same engine: if the engine is calibrated on the ladder, its numbers are trustworthy for decisions.

## The architecture portfolio (decided 2026-07-09)

All five candidate architectures (A–E, see [architecture-explainer.html](architecture-explainer.html)) are pursued **in parallel and ensembled against each other**: each produces its own forecast behind a common interface, and an architecture-level pool combines them with weights learned on backtest — possibly per question category. No candidate is "eliminated"; underperformers converge to near-zero weight while remaining available as explanation layers or diversity sources. Build order (A first as baseline and live tournament entry) is a cost/sequencing choice, not a verdict.

## Undecided (tuned empirically on backtest)

- Per-candidate weights in the architecture-level pool, and whether they vary by question category
- Ensemble size within candidates, extremization coefficient, market weight, confidence thresholds

## Current status

Discovery phase — vision under active discussion; **no system code until the vision is settled**. Prior-art dossier complete (2026-07-09), decision review round 1 complete (2026-07-09).
