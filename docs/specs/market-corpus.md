# Spec: Market Corpus — resolved-question ground truth + market prices

- Status: draft (M1 recon complete 2026-07-16; build after reader/v2 worktrees merge)
- Date: 2026-07-16
- Parent docs: [ADR-0005](../adr/0005-resolution-access.md) (diversification decision),
  [02-evaluation-harness.md](../02-evaluation-harness.md) (leakage tiers),
  [03-architecture-candidates.md](../03-architecture-candidates.md) § D (market-anchored)

## Goal

Reduce single-source dependence on Metaculus for ground truth, at scale: a corpus of
**resolved binary questions** (text + criteria + outcome + timestamps) for backtesting,
plus **historical price series** as candidate D's market-anchor input. All read-only,
internal research use.

## Sources (recon 2026-07-16, three-platform survey)

| Source | Role | Access | Criteria rigor | Notes |
|---|---|---|---|---|
| **Polymarket** (Gamma API) | Primary corpus + prices | Open, no auth | ~2/5 | `gamma-api.polymarket.com/markets?closed=true`; outcome inferred from `outcomePrices` after close (no enum field); UMA-oracle adjudication; `clob.polymarket.com/prices-history` for series. Internal use fine; NO public redistribution. |
| **Manifold** (v0 API) | Secondary corpus (volume) | Open, 500 req/min/IP | ~2/5 | Clean fields (`resolution` YES/NO/MKT/CANCEL, `resolutionTime`); creator SELF-RESOLUTION noise is real — hard filters required. Non-commercial research OK; commercial ML use needs their data license (flag if we commercialize). |
| **Kalshi** | DEFERRED | Open reads (likely) | ~4/5 structurally | Data ToS reportedly bars ML-training/redistribution without written consent (secondary-source claim; PDF unread). Do NOT ingest until the ToS is read directly and/or consent obtained. |
| **Autocast** (GitHub) | Static historical backtest set | Open | high (GJOpen/Metaculus-derived) | Frozen ~2022; leakage-safe for old-model baselines, stale otherwise. |
| **ForecastBench** | Methodology template, not a source | — | — | Their pipeline (LLM quality-filter, multi-platform pull, anti-memorization combos) is the blueprint to imitate. |

Metaculus remains the forward-eval backbone (question quality + independent adjudication);
this corpus complements it.

## Design

1. **Provider-neutral `MarketClient` protocol** (the llm.py/search.py pattern): one client
   per platform, normalized `MarketQuestion` record: id, platform, question, criteria text,
   outcome ("yes"|"no"; MKT/CANCEL/ambiguous excluded), created/close/resolution
   timestamps, liquidity/volume metrics, provenance URL. `[market_corpus]` settings table.
2. **Corpus builder CLI** (`build-market-corpus`): paginated pull of resolved binaries →
   `data/market-corpus/<platform>.jsonl` (append-only, dedupe by platform id). Caps and
   politeness delays in config.
3. **Quality filters** (config, all recorded per-record so filters are re-runnable):
   min criteria length; min liquidity/volume (Polymarket) or min unique bettors (Manifold);
   Manifold: exclude self-resolved-by-creator where API exposes it, exclude MKT/CANCEL;
   optional LLM rubric pass (ForecastBench-style "is this resolvable as written?" —
   cheap model, scored 1–5, threshold in config).
4. **Leakage labels**: every record carries resolution_time; backtest selection uses
   created→resolved windows per docs/02's tiers. Price series are fetched separately and
   ONLY for candidate-D experiments (a price series trivially leaks the outcome near
   resolution — D's experiments must cut the series at forecast-time, enforced in code).
5. **Candidate D hook**: `prices(market_id, start, end)` on the protocol (Polymarket
   prices-history; Manifold reconstructed from `/v0/bets`), cut-off enforcement built in.
6. **Terms compliance**: corpus is gitignored (like all data/), never redistributed;
   Kalshi excluded until its ToS is read directly; Manifold commercial-license flag
   recorded in ADR-0005's consequences.

## Declared checkpoints

| # | Gate | Artifact |
|---|---|---|
| M1 | Recon (DONE 2026-07-16) | Source table above |
| M2 | Builder + first pull | ~500-question filtered sample + quality-filter stats for owner skim |
| M3 | First market-backtest registered | Pre-registered experiment using the corpus (and/or candidate D price-anchor experiment) |

## Open questions

- Kalshi ToS direct read (owner or agent w/ PDF) before any Kalshi ingestion.
- Whether Halawi et al.'s released 5k-question set is directly reusable (license check) —
  would shortcut M2 curation.
- LLM rubric filter: which cheap model + threshold (tune at M2 on a hand-labeled sample).
