# 0005 — Resolution access: public-page reader; no forecast submission yet

- Status: accepted
- Date: 2026-07-16
- Decided by: human (owner, chat, 2026-07-16)

## Context

Scoring any experiment requires question outcomes. Discovered 2026-07-16: the Metaculus
API hides `question.resolution` (and community prediction) from token access unless the
account **predicted on the question**, or holds the separately-granted "Bot Benchmarking
Access Tier" (~250 open + ~250 resolved curated questions; free Google-Form application) —
documented in their OpenAPI spec; verified live with both our bot and human tokens (null
back to 2016 questions; no query param unlocks it; unauthenticated API 403s).
Resolutions are, however, fully public on the website.

Options: (a) submit real forecasts on pinned questions from the bot account (unlocks API
resolution per-question, starts a public track record); (b) apply for the benchmarking
tier (solves backtest corpus, not our own pinned questions); (c) read the public question
pages with a real browser engine (plain HTTP is bot-filtered).

## Decision

1. **No forecast submission yet.** The owner is not ready to publish anything under the
   bot's name until the system and processes are complete ("I'd rather keep working until
   we have a complete set of code and processes that we're happy with"). Submission
   remains an owner-gated future step, consistent with the tournament deferral (B3,
   baseline-bot spec).
2. **Build the public-page resolution reader**: Playwright/Chromium reading the public
   question pages for our pinned experiment questions (~a handful of newly-resolved pages
   per week), parsing the RESOLVED marker, storing to a separate provenance-stamped
   store (`data/web-resolutions/`, source URL + fetch time) that overlays experiment
   scoring — API-derived values win when present; web values fill gaps. Never guesses:
   unparseable pages record nothing.
3. **Owner applies for the Bot Benchmarking Access Tier** (form) for the curated resolved
   corpus — read access for eval purposes, publishes nothing.
4. **Diversify ground truth beyond Metaculus**: recon + spec a prediction-market corpus
   (Polymarket / Kalshi / Manifold public APIs — machine-readable resolutions at scale,
   no gating), which also supplies real market prices for candidate D. Metaculus stays
   the forward-eval backbone (question quality + independent adjudication).

## Alternatives considered

- **Submit forecasts now** — cleanest API unlock and the eventual mission, but publishes
  a permanent public track record before the owner is ready; declined for now.
- **Session-cookie scraping of the API** — would require handling the owner's login
  credentials; prohibited.
- **Manual resolution entry** — invites transcription errors and self-serving ambiguity;
  the reader keeps provenance mechanical.

## Consequences

- Experiments 001/002 become scorable this week without publishing anything.
- A new heavyweight dev dependency (Playwright + Chromium) enters the engine for a
  low-volume, fallback-only path; if Metaculus later grants API access (tier or
  submission), the reader demotes to backup.
- Web-parsed resolutions are second-class evidence by design (overlay, provenance
  stamped) — postmortems can always distinguish them from API-derived values.
- The market-corpus track (own spec) reduces single-source dependence and unblocks
  candidate D's price data.
