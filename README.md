# Superforecaster Agent

An LLM-based forecasting agent: takes a resolvable question (Metaculus/Kalshi-style), researches it, and produces a calibrated probability with legible reasoning.

## Status

**Phase: Discovery** — figuring out how to build it before building it. See [docs/00-discovery-plan.md](docs/00-discovery-plan.md).

## Guiding constraints

1. **Feedback is slow.** Real questions take months to resolve. Every design decision must be testable through the evaluation harness (backtests + live tournament), not intuition.
2. **The frontier is scaffolds, not models.** The best current systems (FutureSearch, Preseen, per ACX July 2026) are orchestration layers around frontier models — this is buildable solo.
3. **Measure before architecting.** No architecture gets adopted without beating the baseline on a leakage-hardened backtest with enough questions for statistical power.

## Documents

| Doc | Purpose |
|---|---|
| [VISION.md](docs/VISION.md) | **What we're building** — one-page target-system summary |
| [architecture-explainer.html](docs/architecture-explainer.html) | Visual explainer: concept primers + diagrams of candidates A–E |
| [VISION-review.html](docs/VISION-review.html) | Annotated decision-review page (round 1 feedback processed 2026-07-09) |
| [00-discovery-plan.md](docs/00-discovery-plan.md) | The plan for figuring out the plan |
| [01-prior-art.md](docs/01-prior-art.md) | Research dossier: human craft, AI systems, aggregation, decomposition evidence |
| [02-evaluation-harness.md](docs/02-evaluation-harness.md) | How we'll know if any design works |
| [03-architecture-candidates.md](docs/03-architecture-candidates.md) | The designs to race, and results |
| [04-development-approach.md](docs/04-development-approach.md) | How we build: agentic SDLC, standards, ADRs, project skills |

## Origin

Prompted by [The AI Superforecasters Are Here](https://www.astralcodexten.com/p/the-ai-superforecasters-are-here) (Astral Codex Ten). Original intuition: recursively decompose a question into knowable sub-facts, source reliable data for each, weight by competence, and let probabilities bubble up a graph. That idea is **hypothesis C** in the architecture race — the literature suggests boring ensembles win on Brier score, but the graph may earn its keep as an explainability layer.
