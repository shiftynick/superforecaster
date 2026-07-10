# Spec: Retrieval Module (ADR-0004 implementation)

- Status: proposed (awaiting owner approval of plan + checkpoints)
- Date: 2026-07-09
- Parent docs: [ADR-0004](../adr/0004-retrieval-stack.md), [02-evaluation-harness.md](../02-evaluation-harness.md) (frozen context, checkpoint C2), [03-architecture-candidates.md](../03-architecture-candidates.md) (retrieval is cross-cutting infrastructure for all candidates)

## Goal

Give every architecture candidate real, date-aware research: a provider-neutral search layer
(AskNews for news, Tavily for general web), tiered content extraction, and a frozen
**research bundle** stored in each forecast's trace. Candidate A stops forecasting blind.

Keys already in `engine/.env`: `ASK_NEWS_API_KEY`, `TAVILY_API_KEY` (also `OPENROUTER_API_KEY` —
out of scope here, unlocks the planned third LLM provider separately).

## Design

Mirrors `llm.py`'s provider pattern throughout — protocol + swappable clients + config over code.

### Search layer (`search.py`)

- `SearchResult` (pydantic): `url`, `title`, `snippet` (or AI summary), `published_at`,
  `source_domain`, `provider`, `raw` (provider response for the trace).
- `SearchClient` Protocol: `search(query, *, max_results, start_date=None, end_date=None) -> list[SearchResult]`.
- `AskNewsClient` — `/news` endpoint; 1 call each; monthly-quota counter persisted (free tier =
  1,000 calls/mo, 4,000/season — hard budget guard, refuse politely when exhausted).
- `TavilyClient` — basic search (1 credit); `start_date`/`end_date` passthrough.
- Factory + `[retrieval]` section in `settings.toml`: provider roles (news / general),
  `max_results`, `queries_per_question`, `max_fetches_per_question`, timeouts. All tunable.

### Extraction layer (`extract.py`)

- `ExtractedDoc`: `url`, `text`, `title`, `published_at`, `tier` (how it was extracted), `ok`.
- Tier 0: `httpx` fetch + `trafilatura` — with a quality gate (min chars, min word count) so
  SPA shells and paywall teasers register as failures rather than junk context.
- Tier 1: Jina Reader (`https://r.jina.ai/<url>`, free key tier) on Tier-0 failure.
- Tier hit-rates logged per run (the ADR's "instrument before buying Tier 2" requirement).
- New dep: `trafilatura` (plus its stack). No paid Tier 2 in v1.

### Research bundle (`research.py`)

- Query generation: one LLM call (`[tasks.query_gen]` in settings — cheap model/effort)
  turns the question title + resolution criteria into 2–4 search queries, **including one
  aimed at the question's resolution source** (field-survey graft: ground against the page
  that will actually decide resolution).
- Pipeline: queries → search (news + general) → dedupe by URL/domain → top-K fetch+extract
  (budgets from config) → `ResearchBundle` (queries, results, docs, tier stats, costs,
  `retrieved_at`).
- **Frozen context:** the bundle is stored verbatim inside the `ForecastRecord` trace —
  leakage-proof by construction for all forward-looking forecasts. (Backtest-time historical
  retrieval via AskNews archive `cutoff_date` is C3/backtest-runner scope, not this spec.)

### Forecaster integration

- `run_forecast_pass` gains an optional research step: bundle rendered into the checklist
  prompt as a SOURCES block (title, date, domain, extract) with an instruction to weigh
  source quality and dates. `--no-research` CLI flag preserves the blind baseline —
  we keep it runnable forever as the control arm.
- Cost accounting extended: search calls + extraction fetches counted per forecast alongside
  LLM tokens; AskNews quota drawn down visibly (`stats` shows remaining).

## Out of scope

OpenRouter LLM provider (separate small change under ADR-0003); AskNews `/deepnews` and
archive endpoints; paid extraction tier; snapshot-scraper context capture for *unforecast*
questions (revisit when the backtest runner defines what it needs); candidates B–E.

## Declared checkpoints

| # | Gate | Artifact for human review |
|---|---|---|
| **R1** | Module built, tests green | Live smoke test transcript: 2 real questions → queries, sources found, extraction tier stats, quota/credit draw. Minutes-reviewable |
| **R2** | Before this becomes the default path | Side-by-side dry-run doc (like B2): same 3 questions, retrieval-on vs retrieval-off, both providers — owner judges whether sourced rationales actually improved |

## Verification

- All four gates green (`pytest`, `ruff check`, `ruff format`, `mypy`) — network mocked in tests
  (httpx MockTransport seam, as in the Metaculus client).
- One live probe per provider during recon (auth shape, response shape) before the build —
  note: AskNews auth needs verification (their SDK historically used client-id/secret OAuth;
  the owner's key is a single `ASK_NEWS_API_KEY` — confirm which scheme the current API takes).

## Agent execution plan

1. **Sonnet** — live recon probes: AskNews auth + `/news` response shape against our real key,
   Tavily search + date-filter response shape; report exact JSON.
2. **Opus** — `search.py` + `extract.py` + settings/config plumbing + tests (from recon findings).
3. **Opus** — `research.py` + forecaster/CLI integration + quota accounting + tests.
4. **Fable** — review both builds, run R1 live smoke test, present R1/R2 artifacts.
