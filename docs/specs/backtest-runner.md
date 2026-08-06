# Spec: Backtest Runner (M3) — instant relative eval over the market corpus

- Status: draft — at checkpoint **BT0 (spec review)**, awaiting owner sign-off on the
  leakage methodology before any code
- Date: 2026-08-06
- Parent docs: [market-corpus.md](market-corpus.md) (M3), [02-evaluation-harness.md](../02-evaluation-harness.md)
  (leakage tiers), [03-architecture-candidates.md](../03-architecture-candidates.md)

## Goal

Turn the evaluation loop from *forecast → wait weeks → sweep resolutions* into
*forecast → score now*. We hold 413 already-resolved market questions
(`data/market-corpus/`). A backtest runner forecasts those with any candidate (A/B/C/…),
scores against the known outcome immediately, and reports Brier + calibration. This makes
every future candidate/config an afternoon's experiment instead of a month's wait.

## The load-bearing problem: leakage (read before trusting any number)

A resolved question's outcome is public *now*. If the research step retrieves post-event
coverage, the model reads the answer and the score is meaningless. Our providers offer
date filters (AskNews `cutoff_date`, Perplexity `search_before_date_filter`) — but docs/02
already established these are **leakage reducers, not eliminators**: published-date
metadata can be wrong/missing, and the live index is today's, not a historical snapshot.

**The honest position this spec commits to:**

- **Absolute Brier from a backtest is OPTIMISTIC** (some leakage always survives) and is
  NEVER reported as "how good our system is". Every backtest number carries a
  `leakage_tier` label per docs/02 and the word "backtest" in its provenance.
- **Relative comparison stays valid**: every candidate in a backtest sees the SAME
  cutoff-bounded (equally-leaky) research for the same question, so "which candidate is
  better" is a fair question even when "how good are we absolutely" is not. Backtests
  answer *ranking and direction*, not *calibration truth* — calibration truth comes only
  from forward Metaculus experiments (001/002/…).
- A **leakage audit** is a first-class output, not an afterthought (below).

## Design

### 1. Forecast-time anchor + cutoff

Each corpus question gets a **forecast-time anchor** = a date strictly before its
resolution: default = its `created_time` (when the market opened — the most defensible
"what was knowable then"). All research for that question is cutoff-bounded to the anchor:
AskNews `cutoff_date=anchor`, Perplexity `search_before_date_filter=anchor`, and any
result whose `published_at > anchor` is dropped in code as a belt-and-suspenders filter
(the one guard that doesn't depend on the provider honoring the date param).

### 2. Runner

`backtest --candidate {A|B|C} --corpus {polymarket|manifold|both} --max N [--min-anchor-gap-days D]`:
for each kept corpus question (filters from M2 applied), build cutoff-bounded research at
the anchor, run the candidate (reusing `run_forecast_pass` / ensemble / pyramid unchanged),
score `brier(prob, outcome)`, and persist a `BacktestRecord` (candidate, question id +
platform, anchor, prob, outcome, brier, leakage_tier, dropped-by-date count, cost).
`--min-anchor-gap-days` skips questions that resolve too soon after opening (little time
for the forecast to be non-trivial).

### 3. Leakage audit (first-class)

Every run emits an audit alongside the scores: how many results were date-dropped in code
(high drop rate = the provider ignored the cutoff = trust the number less); a sampled
handful of (question, top source, source date, anchor) rows for manual spot-check; and the
fraction of questions where the model's rationale cites a date after the anchor (a cheap
regex/LLM check — a smoking gun for leakage). A run whose audit looks dirty is reported as
such, loudly.

### 4. Scoring output

Mean Brier + a calibration table (predicted-bin vs realized frequency) per candidate,
**always** shown next to the **base-rate baseline** (predict the corpus base rate every
time) — because the Polymarket slice is 82% "no", a lazy "always 0.15" scores deceptively
well and must be visible as the floor to beat. Paired deltas when two candidates run the
same corpus slice.

### 5. Persistence & config

`data/backtests/` (gitignored), append-only, keyed by (candidate, platform, question,
anchor). `[backtest]` config: default anchor policy, min-anchor-gap, per-run caps,
leakage-audit sample size.

## Out of scope (M3 v1)

Learned/optimal anchor selection; historical-index reconstruction (GDELT/Common Crawl —
the only true leakage fix, a much larger build); candidate D itself (this runner is its
prerequisite; D is the next spec); auto-promotion of any config from a backtest number
(backtests inform, forward experiments decide).

## Declared checkpoints

| # | Gate | Artifact / owner action |
|---|---|---|
| **BT0** | Spec review (NOW) | This doc — approve the leakage methodology + anchor policy before code |
| **BT1** | Vertical slice + leakage audit | Candidate A backtested on ~30 corpus questions; owner reviews the LEAKAGE AUDIT (not the Brier) — is the cutoff actually holding? A dirty audit here means the whole approach needs rethinking before we trust any backtest |
| **BT2** | First multi-candidate backtest | A vs B vs C on a corpus slice — the first instant relative scoreboard, reported with its leakage caveat |

## Agent execution plan (after BT0)

1. **Opus** — cutoff-bounded research seam (thread anchor → AskNews cutoff_date / Perplexity
   date bound / in-code post-anchor drop) + `BacktestRecord` + store + property tests on the
   date filtering.
2. **Opus** — `backtest` CLI + leakage audit + base-rate baseline + calibration table.
3. **Fable** — review, run the BT1 slice, present the leakage audit for owner sign-off.
