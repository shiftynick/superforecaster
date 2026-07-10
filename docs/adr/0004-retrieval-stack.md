# 0004 — Retrieval stack: AskNews + Tavily search, tiered local-first extraction

- Status: accepted
- Date: 2026-07-09
- Decided by: human (owner approved at checkpoint C2, 2026-07-09)

## Context

Every architecture candidate needs retrieval — it is the single most evidence-backed
performance component in the prior-art dossier (docs/01), and candidate A currently runs
blind. Harness checkpoint C2 requires this decision before frozen-context capture. Owner
constraints: as cheap as possible, high-quality real data from good sources; browser-use
and exa.ai explicitly acceptable if needed.

Retrieval decomposes into two problems: **search** (query → relevant URLs/snippets) and
**extraction** (URL → clean article text). A third, forecasting-specific requirement is
**leakage control**: backtests must not see post-question-date information.

Research findings (2026-07-09, three-agent web survey + verified in-browser on the
Metaculus resources page, notebook 38928):

- **Metaculus gives registered AIB bot builders free AskNews access**: 1,000 calls/month,
  4,000/season, 5M tokens, `/news` (1 call) and `/deepnews`; archive queries cost 5 calls.
  Signup: AskNews account on the bot's email + a Discord/email request. Restriction:
  competition use only (broader use needs an email to Metaculus, or a paid plan —
  pay-as-you-go is $0.025/call). AskNews was the #1 Q4 bot's primary news source, and
  Metaculus reports bots improved when switching to it from Perplexity.
- AskNews's forecast endpoint supports an explicit `cutoff_date` — the only surveyed
  provider purpose-built for backtesting.
- **Tavily**: 1,000 free credits/month (basic search = 1 credit), `start_date`/`end_date`
  published-date filters, agent-native output, official Python SDK. At our volume the free
  tier likely covers everything.
- **Exa**: strong date filters and semantic search; $5/1k searches + $1/1k content items —
  fine but not free; kept as an optional provider, not the default.
- **Extraction quality**: local `trafilatura` scores F1 ≈ 0.93–0.97 on news articles (free);
  Jina Reader (`r.jina.ai`) renders JS server-side with a large free tier; Firecrawl/Tavily
  extract cost fractions of a cent per page. Browser-use is agentic (LLM per page) —
  wrong tool for batch article extraction.
- **Leakage caveat (all providers)**: live search APIs filter today's index by published-date
  metadata; none replay a historical index. Date filters *reduce* leakage, they cannot
  eliminate it. True leakage-proofing requires freezing retrieved context at forecast time,
  or archive corpora (AskNews archive, GDELT via BigQuery — free).

## Decision

1. **Provider-neutral `SearchClient` protocol** in the engine, mirroring `llm.py`'s
   `LlmClient` pattern: providers are swappable via `engine/config/settings.toml`
   (`[retrieval]` section — provider, per-question search/fetch budgets, date-filter
   behavior). Every call logs cost.
2. **News search: AskNews** (free AIB tier) as primary news provider for
   tournament-related development and backtesting.
3. **General web search: Tavily** (free tier) as the default general provider — also the
   fallback if AskNews signup stalls or its competition-only scope binds.
4. **Extraction, tiered local-first**: `httpx` fetch + `trafilatura` (free, with a
   min-length/quality heuristic to catch SPA shells and paywall teasers) → Jina Reader
   fallback for JS-heavy pages (free tier) → paid extractor (Firecrawl or Tavily extract)
   only if evidence shows the first two tiers failing often. Instrument tier hit-rates.
5. **Leakage control**: all retrieved documents are stored in the forecast trace (frozen
   context — already the store's pattern); backtests over historical questions use
   AskNews archive `cutoff_date` (or GDELT free fallback) and are labeled with their
   leakage-risk level per docs/02.

Expected steady-state cost: **≈ $0/month** at current volume (free tiers + local
extraction), with cents-per-day worst case if paid fallbacks engage.

## Alternatives considered

- **Exa as primary** — excellent date filtering and bundled content, but costs real money
  per search while Tavily's free tier covers our volume; kept as an optional provider.
- **Perplexity Sonar** — returns LLM-synthesized answers rather than raw sources; conflates
  the search and reasoning steps (hallucination-laundering risk for a forecasting pipeline).
- **Serper/SerpAPI (Google SERP)** — cheapest raw SERP access, but date filtering is
  crawl-recency-based, snippets only, and Tavily's free tier beats "cheap".
- **SearXNG self-hosted** — free but relative-recency filters only, upstream-engine
  reliability risk; not worth the ops burden at this volume.
- **browser-use / hosted headless browsers** — LLM-in-the-loop or priced for high-volume
  anti-bot scraping; overkill for news articles. Playwright+trafilatura remains a free
  local option if JS-heavy sites become a recurring problem.
- **Google Custom Search JSON API** — closed to new signups (migration deadline Jan 2027).

## Consequences

- Candidate A gets real retrieval; the checklist prompt gains a research step; frozen-context
  capture (C2) can proceed on the snapshot store.
- Owner setup needed: AskNews account + Discord/email request (bot email), Tavily API key;
  optionally the Metaculus OpenRouter LLM-credit form (free OpenAI/Anthropic/Google credits,
  reapply each season).
- AskNews free tier is competition-scoped: when the system pivots to personal-platform use,
  either pay AskNews (~$0.025/call) or lean on Tavily+extraction; the provider-neutral
  protocol makes that a config change.
- Date filters are trusted as leakage *reducers* only; the harness's leakage labels
  (docs/02) stay load-bearing, and postmortems should spot-check retrieved sources' dates.
