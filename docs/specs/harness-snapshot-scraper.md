# Spec: Evaluation Harness v1 + Snapshot Scraper

- Status: in progress (vertical slice)
- Date: 2026-07-09
- Parent docs: [02-evaluation-harness.md](../02-evaluation-harness.md), [VISION.md](../VISION.md)

## Goal

Start the two clocks that matter: (1) a **snapshot scraper** that begins archiving unresolved forecasting questions *today* — every scraped question that later resolves becomes leakage-proof backtest data we cannot obtain retroactively; (2) the **harness math** (scoring + aggregation) as tested pure functions everything else will build on.

## Scope — v1

- **Source:** Metaculus public API, binary questions only. (Kalshi/Polymarket: later.)
- **Scraper:** manual-kickoff CLI (`scrape`) that fetches open binary questions and appends timestamped snapshots — question text, resolution criteria, deadline, community prediction at scrape time — to an append-only store. Re-running later captures resolutions (`scrape` also re-checks previously seen questions).
- **Store:** JSONL under `data/snapshots/` (gitignored; local data, not code). One record per (question, scrape-time). Append-only — history is the point.
- **Metrics:** Brier score, log score, calibration curve + ECE. Pure functions, property-tested.
- **Aggregation:** probability↔log-odds conversion, log-odds pooling (weighted), extremization (coefficient param), Platt scaling (fit + apply). Pure functions, property-tested.
- **CLI:** `uv run superforecaster scrape` / `... stats` (snapshot store summary).

## Explicitly out of scope (v1)

- **Search-context capture** (freezing contemporaneous web-search results alongside each snapshot — the full "frozen context" design). Needs a search-provider decision (ADR + possibly an API key from the owner). Phase 2, checkpoint C2.
- Forecasting passes, LLM calls, tournament bot, backtest runner UI.

## Module layout

```
engine/src/superforecaster/
  models.py        # pydantic: Question, QuestionSnapshot, Resolution
  metrics.py       # brier, log_score, calibration_curve, ece
  aggregation.py   # to_log_odds, from_log_odds, pool, extremize, platt_fit/apply
  metaculus.py     # typed API client (httpx)
  store.py         # append-only JSONL snapshot store
  cli.py           # typer entry point
```

## Declared checkpoints

| # | Gate | Artifact for human review |
|---|---|---|
| **C1** | Vertical slice done (this run) | Real scraped snapshots on disk, math module test results, CLI demo output, review summary |
| **C2** | Before search-context capture | ADR proposing the search provider (cost, ToS, coverage), any API-key ask |
| **C3** | Before first real backtest | The pre-registration format + a dry-run experiment record |

## Agent execution plan (this run)

Fable orchestrates and reviews; subagents implement:

1. **Sonnet** — hands-on Metaculus API recon (endpoints, fields, pagination, rate limits) → implementation-ready report.
2. **Opus** — metrics + aggregation module with hypothesis property tests (independent of 1).
3. **Opus** — models + client + store + CLI, built from 1's report (blocked by 1).
4. **Fable** — review all diffs, run full verification, live-run the scraper once, commit, present C1.
