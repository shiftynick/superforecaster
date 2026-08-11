# HANDOFF — paused 2026-08-06

Written for a cold restart after a long break. Read this first, then `docs/VISION.md`.
Everything below was true at commit `a9eb290`; working tree clean, 411 tests green.

---

## 1. What this is

A personal forecasting system: give it a resolvable question, it researches the web and
returns a calibrated probability + confidence + a legible argument. Built agentically
under a strict SDLC — **spec → plan → build → review → verify → record**, with
pre-registered experiments deciding every design question. Nothing is adopted on vibes;
a config only changes when a frozen experiment's pre-committed rule says so.

The engine (`engine/`, Python) is real and working. The web app (`web/`) does not exist
yet. Ground truth comes from Metaculus (forward) and a market corpus (historical).

---

## 2. Do these two things when you come back

Live experiments have been accruing resolutions the whole time you were away. From the
repo root:

```bash
uv run --project engine superforecaster sweep-resolutions
```
```bash
uv run --project engine superforecaster experiment-status
```

The first reads each pinned question's **public Metaculus page** for its outcome (the
API hides resolutions — see §6) and appends what it finds; `--max` caps pages per run
(default 25), and re-runs resume, so run it 2–3 times if many resolved. The second prints
the interim scoreboard for experiments 001/002.

For the full board including the pooled arms (004) and effort arms (005), those are
script-scored — see §5.

---

## 3. State of play

### Architecture portfolio (docs/03)
| Candidate | Status |
|---|---|
| **A** checklist single pass | ✅ built, with retrieval |
| **B** diverse ensemble (6 members, log-odds pool) | ✅ built |
| **C** decomposition pyramid | ✅ v1 **and** v2 (assumption surfacing + verification) |
| **D** market-anchored | ❌ not built — needs the backtest runner + `prices()` hook (both now exist/specced) |
| **E** supervisor-reconciled | ❌ not built |

### Experiments
| # | What | Status @ 2026-08-06 |
|---|---|---|
| 001 | B vs A | frozen, forecasting complete, **9/40 resolved** |
| 002 | C vs A vs B | frozen, forecasting complete, **7/40 resolved** |
| 003 | *(reserved: C v2 vs v1)* | **never frozen** — waiting on your V1 verdict (§4) |
| 004 | architecture pool (derived from 002, $0) | frozen, script-scored |
| 005 | effort low vs medium | frozen, forecasting complete, **6/40 resolved** |
| 006 | Perplexity vs Tavily | ✅ **COMPLETE** → Perplexity adopted as general provider |

**First interim scoreboard (N=6–9 — DIRECTIONAL ONLY, no adoptions triggered):**
- 002 (N=7): B ensemble **0.091** < P-BC pool 0.098 < P-ABC 0.103 < A 0.120 < C-blended 0.150 < C-raw 0.288
- 001 (N=9): A 0.207 < B 0.223 *(contradicts 002 — that's what noise looks like at low N)*
- 005 (N=6): E-med 0.137 < E-low 0.143

Two findings worth remembering: **(1) C's holistic blend is strongly validated** — raw
structural 0.288 vs blended 0.150, the blend nearly halved the pyramid's error (a
within-C comparison, the most trustworthy signal on the board). **(2) The naive
equal-weight architecture pool did NOT beat its best member** — pooling averaged strong B
with weaker A/C. Learned/per-category weighting is the plausible rescue, and belongs to a
future experiment.

Caveat that matters: the resolved questions are the **early-resolving subset**, plausibly
easier than the full set. The real read comes from the remaining ~30 per experiment.

### Data on disk (all gitignored)
`data/snapshots/` ~1,260 Metaculus questions · `data/forecasts/` all forecast records
(stamped with `experiment_id`) · `data/market-corpus/` **413 resolved market questions**
(300 Polymarket + 113 Manifold) · `data/web-resolutions/` web-read outcomes ·
`data/research/` cached research bundles · `data/bundle-evals/` research-quality judgments

---

## 4. Decisions waiting on you

1. **C v2 V1 verdict** — three assumption-verified argument graphs are rendered at
   `docs/reviews/forecast-{44013,43736,40625}.html`. The question: *are these the
   assumptions a careful human would have listed?* Your verdict unlocks **freezing
   experiment 003** (C v2 vs v1).
2. **BT0 sign-off** on `docs/specs/backtest-runner.md` — the M3 backtest runner. The
   load-bearing decision is the **leakage methodology**: market-open date as the
   forecast-time anchor, cutoff-bounded research, and the commitment that backtest Brier
   is *relative-only* (valid for ranking candidates, never quoted as "how good we are").
   This was the last thing in flight when we paused.
3. **Metaculus Bot Benchmarking Tier form** (optional, free) — grants API access to ~250
   open + ~250 resolved curated questions. Would reduce reliance on the web reader and
   partially unblock candidate D. Link is in their OpenAPI spec; publishing nothing.

---

## 5. Non-obvious operational facts

- **Experiment stamping is mandatory.** Every forecast run for an experiment MUST pass
  `--experiment-id <id>`. Experiments share pinned questions and reuse method/model
  values, so without the stamp one experiment's records silently displace another's under
  latest-record-wins scoring. Backfill script: `engine/scripts/backfill_experiment_ids.py`.
- **Script-scored experiments.** 004 (pooled arms) and 005 (same method+model, differing
  only in `effort`, which the record doesn't store) have **no `.toml` manifest** — they're
  scored by ad-hoc scripts against their markdown contract. Queued machinery fix: add
  `effort` to `ForecastRecord` + optional `ArmSpec.effort` matching, and derived-arm
  support in `experiment-status`.
- **Frozen configs** live beside the main one: `engine/config/experiment-00{1,5,6}*.toml`.
  Run with `SUPERFORECASTER_CONFIG=<path>`.
- **Never edit engine source while a driver is running.** Each per-question CLI invocation
  re-imports source; a half-saved edit crashed ~16 questions once.
- **Background drivers survive session end and machine sleep** (Windows). A driver once
  woke 4 days later and collided with its replacement (`os error 32` on the CLI exe).
  **Hunt zombies before launching any resume driver** (`Get-CimInstance Win32_Process`,
  match on the script name).
- **`uv run` from the repo root**, always — paths in `settings.toml` resolve from there.
  Running from `engine/` silently creates `engine/data/` and finds no questions.

---

## 6. External services — what works and what's walled off

| Service | State |
|---|---|
| **Metaculus API** | Reads questions fine. **Hides `resolution` AND community prediction** unless the account forecasted on that question (or holds the Benchmarking Tier). Also 429s under volume. |
| **Metaculus public pages** | The working path for outcomes — `sweep-resolutions` reads them with Playwright/Chromium, provenance-stamped. |
| **AskNews** (news search) | Free via the AIB partnership, **900 credits/month cap** (resets monthly; was 591 used in July). Capped at **2 queries/question** (`news_max_queries`) to halve spend. |
| **Perplexity Search API** | Adopted general-web provider (experiment 006). Pay-as-you-go, $5/1k searches — real money, tiny amounts. |
| **OpenRouter** | Real dollars, and how the Metaculus-donated LLM credits arrive. Default model GLM-5.2. |
| **Claude / Codex CLIs** | Subscription-based (notional cost). Claude CLI runs `query_gen` — **if the subscription hits its usage limit, research fails and every forecast arm fails with it** (this happened once mid-experiment). |
| **Kalshi** | Deliberately NOT ingested — its data ToS reportedly bars ML use/redistribution without written consent, and the PDF is unread. |

**Nothing has ever been submitted to Metaculus.** Live tournament entry is deferred by
decision (ADR-0005 / baseline-bot B3) until the frameworks are complete — the bot account
has no public track record, and that stays your call. The Summer 2026 AIB season's
forecasting window closes 2026-09-06; re-entry was always targeted at a later season.

---

## 7. Where to pick up

**If you want evidence:** run the two commands in §2. As the remaining ~30 questions per
experiment resolve, 001/002/004/005 finally produce non-trivial reads. That's the cheapest
high-value action available and needs no new code.

**If you want to build:** the backtest runner (`docs/specs/backtest-runner.md`, awaiting
BT0) is the highest-leverage next build — it turns the eval loop from *weeks* into
*minutes* using the 413 resolved corpus questions, and it's the prerequisite for candidate
D. Its three gates are BT0 (spec) → BT1 (you inspect the **leakage audit**, not the Brier)
→ BT2 (first instant A/B/C scoreboard).

**Smaller queued items:** both-orders pairwise judging (fixes experiment 006's disclosed
position bias); deep-research-tier survey (task #17's remainder); C v3 adaptive
decomposition (your recorded direction — structure as an *output* of analysis, not a
pre-analysis guess); the postmortem pipeline; the web app design round.

---

## 8. Map

| Path | What |
|---|---|
| `docs/VISION.md` | What we're building — read first |
| `docs/00`–`04` | Discovery plan, prior art, evaluation harness, architecture candidates, dev approach |
| `docs/adr/` | 5 ADRs — the decisions that would puzzle you later |
| `docs/specs/` | Per-build contracts with declared checkpoints |
| `docs/experiments/` | Frozen pre-registrations + results |
| `docs/reviews/` | Rendered argument graphs + checkpoint artifacts |
| `docs/architecture-explainer.html` | Concept primers + diagrams (jargon glossed) |
| `engine/config/settings.toml` | The control panel — every tunable lives here |
| `.claude/skills/` ≡ `.agents/skills/` | Project skills; **must stay byte-identical** |

Engine commands (from repo root): `scrape`, `sweep-resolutions`, `update-resolutions`,
`stats`, `forecast` (`--ensemble` / `--pyramid` / `--experiment-id`), `experiment-status`,
`render-forecast`, `build-market-corpus`, `eval-research`, `compare-research`,
`submit-tournament` (double-gated, never used live).

Verification, all four must be green before any commit (from `engine/`):
`uv run pytest` · `uv run ruff check .` · `uv run ruff format .` · `uv run mypy .`
