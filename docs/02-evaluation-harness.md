# Evaluation Harness

How we'll know whether any design works. Updated 2026-07-09 with findings from the prior-art dossier ([01-prior-art.md](01-prior-art.md)).

## Principles

- **The harness ships before the agent.** No architecture decision without a score.
- **Every experiment pre-registers its question set and metric** before running, to keep us from cherry-picking.
- **Power first**: with Brier differences of ~0.01–0.03 between designs, we need hundreds of questions per comparison. Compute the needed N before believing any result.
- **Model configs are part of the experiment.** Every run records which model + reasoning-effort config each task type used; the harness must make it trivial to re-run the same experiment with a different model config (e.g. Sonnet vs Opus forecasting passes) and compare.

## Cost model (rough, 2026-07 pricing — refine by measuring the baseline)

Assumes a retrieval-augmented forecasting pass ≈ 40k input + 3k output tokens. Current per-MTok pricing: Haiku 4.5 $1/$5, Sonnet $3/$15, Opus 4.8 $5/$25; Batch API −50% for non-urgent runs.

| Unit | Haiku | Sonnet | Opus |
|---|---|---|---|
| Single pass | ~$0.06 | ~$0.17 | ~$0.68 |
| 8-pass ensemble / question | ~$0.45 | ~$1.40 | ~$5.40 |
| 300-question backtest arm (ensemble) | ~$135 | ~$410 | ~$1,630 |
| Same, via Batch API | ~$70 | ~$205 | ~$815 |

Baseline A (single pass) is ~8× cheaper per question than ensemble B. Practical implication: run broad comparisons at Sonnet via Batch API, reserve Opus for the supervisor/judge roles, and measure real token counts from the first baseline runs to replace these assumptions.

## Tier 1 — Leakage-hardened backtests (fast, high volume)

Resolved binary questions from Metaculus (and possibly Kalshi/Polymarket archives), forecast by the agent with information restricted to a cutoff date before resolution.

**The leakage problem:** the base model's training data may contain the outcome. The literature ([arXiv:2507.19477](https://arxiv.org/pdf/2507.19477)) names exactly two failure modes that inflated early LLM-forecasting results — both must be explicitly guarded:

1. Evaluating on events resolved *before* the model's knowledge cutoff
2. Retrieval pipelines mixing *post-resolution* documents into search results

Mitigations, strongest first:

- **Frozen context snapshots** (forward-only backtesting, [OpenReview ICLR'26 sub](https://openreview.net/forum?id=Q5o249Z3Je)): continuously scrape *unresolved* questions now and store contemporaneous search-result snapshots; evaluate later against the frozen snapshot. Leakage-impossible by construction; start the scraper early because the dataset accrues in real time.
- **[Autocast](https://arxiv.org/abs/2206.15474)**: tournament questions paired with a date-organized news corpus — simulate information-at-forecast-time for historical questions.
- Prefer questions that resolved *after* the model's training cutoff (clean by construction; smaller pool)
- Retrieval restricted to documents dated before the question's cutoff (guards failure mode 2)
- **Leakage detector**: compare agent accuracy on pre-cutoff-resolved vs. post-cutoff-resolved question sets; a significant gap on otherwise-matched questions = leakage, backtest untrusted
- Ask-the-model probes ("do you know how X resolved?") as a cheap screen, not a guarantee

⚠️ Do **not** treat ForecastBench as leakage-hardened by construction — that claim was refuted (0-3) in verification; re-derive its design properties from the primary paper before using it.

**Metrics:** Brier score (primary), log score (tail sensitivity), calibration curve + ECE (diagnosis), and score *relative to community prediction at the same timestamp* (the real bar).

## Tier 2 — Live scoring (slow, zero leakage)

Enter the [Metaculus AI Benchmark](https://www.metaculus.com/aib/) with the baseline bot as early as possible, via the official [metac-bot-template](https://github.com/Metaculus/metac-bot-template) (two paths: `main.py` on the `forecasting-tools` package, or `main_with_no_framework.py` with minimal deps; auth via `METACULUS_TOKEN`). Live scores are the ground truth that backtests approximate; any backtest-vs-live divergence is itself a signal that the backtest is broken.

**Current bar (to beat):** Metaculus Pro humans have beaten bots every quarter through at least Dec 2025; Q2 2025 bot-vs-pro head-to-head was −20.03 (95% CI [−28.63, −11.41]). The gap is real; parity with Pros is the long-term target, beating the median bot is the near-term one.

## Tier 3 — Diagnostics

- Reliability diagrams per architecture per question category
- Over/under-confidence decomposition (calibration vs. resolution components of Brier)
- Per-question postmortems (protocol below) — feeds the next design iteration

## Postmortems — learning from misses

*Added 2026-07-09 per user feedback: a first-class capability, not just a dev chore.*

**Trigger.** After resolution, flag (a) the largest scoring misses and (b) confidence miscalibrations — especially high-confidence forecasts that went wrong, which are worse than their Brier alone suggests.

**Why this system can do it well.** Every forecast keeps its full trace: the dated research trail and sources, each ensemble pass's argument, the aggregation record, market data at forecast time, and (if candidate C is in play) the decomposition graph. A postmortem agent re-examines that trace *with the benefit of knowing what actually happened* — including retrieving post-resolution reporting that explains the outcome — and pinpoints where the process diverged from reality. (Note: a legible decomposition graph makes attribution sharper — you can see *which node* was wrong — which is a further point in hypothesis C's favor as an explanation layer.)

**Root-cause taxonomy** (each failure classified into one or more):

| # | Failure mode | Example | Addressable via |
|---|---|---|---|
| 1 | **Information miss** | Evidence existed before the deadline; retrieval never surfaced it | Source list, source weights, retrieval queries |
| 2 | **Reasoning error** | Evidence was found but misread; wrong reference class; conjunction miscalculated | Prompts, checklist, decomposition guards |
| 3 | **Aggregation error** | Individual passes were near-right; the combination wasn't | Pooling weights, extremization coefficient |
| 4 | **Calibration error** | Directionally right, magnitude systematically off in this category | Platt curve, per-category confidence thresholds |
| 5 | **Question misread** | Forecast the wrong thing — resolution criteria misunderstood | Ingestion/clarification step |
| 6 | **Genuinely unforeseeable** | No pre-deadline evidence pointed to the outcome | Nothing — expected losses; log and move on |

**The guard against overfitting.** One core forecasting truth shapes this whole protocol: *a 20% event happening does not mean the 20% was wrong* — one-in-five shots land all the time, and a perfectly calibrated forecaster is "wrong" often. So postmortems separate **process errors** (modes 1–5: always legitimate to fix — the process was faulty regardless of luck) from **outcome regret** (mode 6: fixing it would damage calibration). And every extracted lesson is a **hypothesis, not a patch**: it gets verified on the backtest set (does applying it improve scores across *many* questions?) before it changes any config. Lessons that only explain one miss get logged, not adopted.

**Outputs.**
- A versioned postmortem record per analyzed miss (trace → root cause → lesson → verification status)
- Config-change proposals routed to the specific tunable implicated, queued for backtest verification
- A **lessons library** the forecasting agents consult at run time (the mechanized version of the superforecaster habit that drove much of their measured edge: frequent, honest self-correction)
- Periodic aggregate review: recurring root causes across postmortems are stronger evidence than any single one

## Experiment protocol (per architecture comparison)

1. Freeze the question set (stratified by category, resolution date, community-prediction entropy)
2. Run both architectures with identical retrieval budget and model version
3. Paired-difference test on per-question Brier scores
4. Report: score delta, CI, calibration curves, cost per question

## Remaining open items (hands-on verification needed)

The dossier's verification pass was weakest on infrastructure claims — confirm these by direct inspection, not further reading:

- Clone metac-bot-template; confirm current AIB season, entry mechanics, scoring rules, cadence
- Metaculus API: confirm resolved-question export with community-prediction time series (needed for the "vs. crowd at same timestamp" metric)
- Stand up the frozen-snapshot scraper early — its value compounds with time
- Re-derive ForecastBench's actual leakage properties from the primary paper
