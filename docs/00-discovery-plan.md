# Discovery Plan

The plan for figuring out how to build the agent — not the build plan itself. The build plan (docs/04) gets written only after the evaluation harness exists and the architecture race has a winner.

## Why discovery-first

Forecasting has the slowest feedback loop of any ML-adjacent domain: a question resolving in six months yields one bit of ground truth in six months. Score differences between decent and great forecasters are small (hundredths of Brier points), so distinguishing two designs needs hundreds of resolved questions. Intuition-driven architecture choices are unfalsifiable at that feedback rate. Everything below exists to buy faster, trustworthy feedback.

## Phases

### Phase 1 — Extract the human craft as a checklist ✅ (folded into research dossier)

Tetlock / Good Judgment Project literature, mined for *mechanizable* techniques, each a candidate scaffold component:

- Outside view first: find a reference class, anchor on its base rate
- Inside view second: adjust the anchor with case-specific evidence
- Fermi decomposition of the question
- Frequent, small, incremental updates
- Premortem / actively generate disconfirming hypotheses
- Aggregate many independent estimates; extremize the aggregate

### Phase 2 — Survey prior art → [01-prior-art.md](01-prior-art.md) ✅

Completed 2026-07-09 via adversarially-verified deep-research pass (23 sources, 18 confirmed claims). Headlines: log-odds aggregation confirmed, extremization is conditional (mild for correlated LLM ensembles), selective forecasting is a confirmed edge, AIA-style supervisor+calibration is SOTA, decomposition evidence is a genuine gap (hypothesis C stays open), Pros still beat all bots on the live benchmark.

### Phase 3 — Build the evaluation harness → [02-evaluation-harness.md](02-evaluation-harness.md)

Three tiers, fast to slow:

1. **Backtests** on resolved Metaculus/Kalshi questions with information cutoffs. High volume, but leakage is the trap — needs explicit leakage detection.
2. **Live tournament entry** (Metaculus AI Benchmark) with a dumb baseline, entered early. Zero leakage, real competition, slow.
3. **Calibration diagnostics** — reliability curves, not just Brier, so we can tell *why* a design over/under-performs.

Harness ships before any architecture work. The baseline bot is part of the harness.

### Phase 4 — Race candidate architectures → [03-architecture-candidates.md](03-architecture-candidates.md)

| ID | Architecture | Prior from literature |
|---|---|---|
| A | Single pass: retrieval + superforecaster-checklist prompt | Baseline; already near-human in Halawi et al. |
| B | Ensemble: N diverse passes → log-odds pool → mild extremization | 12-LLM ensemble already matches human crowd |
| C | Recursive decomposition graph with per-node confidence (original idea) | Open — literature has no direct evidence either way |
| D | Market-anchored: start from community/market price, model the adjustment | Confirmed additive: ⅓-model/⅔-market beats market alone |
| E | Supervisor-reconciled agentic scaffold + Platt calibration (AIA-style) | Confirmed 2025 SOTA |

Adjudicated by the Phase 3 harness. Adoption rule: beats baseline A with statistical significance on the backtest set, confirmed directionally on live scoring.

### Phase 5 — Write the build plan

Mostly writes itself from Phase 4's winner.

## Non-goals (for now)

- Fine-tuning or training models — the frontier is scaffolds
- Trading real money — evaluation is tournament/backtest scoring
- Long-horizon "will AGI arrive by 2035" questions — start where feedback exists
