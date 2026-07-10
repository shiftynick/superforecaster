# 001 — Candidate B ensemble vs. candidate A baseline (forward, frozen-context)

- Status: draft (awaiting owner sign-off to freeze — checkpoint E1 / C3)
- Registered: 2026-07-09
- Frozen: —
- Owner sign-off: —

## Hypothesis

The candidate-B ensemble (5 members: 3 reasoning frames on GLM-5.2 + outside-first on
Grok-4.5 and claude-sonnet-5; log-odds pool; agreement-scaled extremization 1.0–1.3)
achieves a lower mean Brier score than candidate A (single outside-first GLM-5.2 pass)
on the same questions with the same research bundles.

## Arms

| Arm | Config (exact) |
|---|---|
| A | `[tasks.forecast_pass]` provider=openrouter, model=z-ai/glm-5.2, effort=low, max_tokens=16384; research on (settings.toml `[retrieval]` as at freeze commit) |
| B | `[ensemble]` default roster and coefficients as at freeze commit, extremization_mode="agreement" (note: shipped default is "fixed"/1.0 — the freeze pins the experimental value); same research bundles as arm A (shared via the research cache) |

Both arms forecast each question within the same run window so they share one cached
research bundle per question (identical evidence; the architecture is the only variable).

## Question set

- Selection rule (to freeze): all open binary questions in the snapshot store with
  `scheduled_resolve_time` within 30 days of the freeze date, capped at the first N by
  soonest resolve time, where N is set by the budget below (target N ≥ 40 for a first
  directional read; a confirmatory run needs hundreds — see power check).
- Leakage tier: **frozen-context forward** (docs/02's strongest tier) — forecasts are made
  before resolution with the research bundle frozen into each record; leakage-impossible
  by construction.
- Question ids: *(appended at freeze time)*.

## Metric & decision rule

- Primary: mean paired Brier delta (B − A) over resolved questions.
- Secondaries (diagnostic only): log score, spread-vs-error relationship (does member
  disagreement predict misses?), cost per question.
- Decision rule, pre-committed: with N < 100, this experiment CANNOT adopt a new default —
  it reports directional evidence only. If the paired delta favors B and the 90% bootstrap
  CI excludes 0, candidate B is promoted to "presumptive default pending confirmatory
  experiment"; otherwise both candidates continue accruing forward records.

## Power check

Published architecture gaps are ~0.01–0.03 Brier; detecting that reliably needs hundreds
of paired questions. N ≈ 40–60 (30-day resolution window) is underpowered for adoption
and is registered as **directional evidence only** — its real product is the forward-record
pipeline itself, which keeps accruing for a later confirmatory experiment at N ≥ 300.

## Budget

Max LLM spend: $40. Max AskNews credits: 250 (fresh bundles for ~N questions at ~4/question,
within the 900/month cap alongside other work).

## Results

*(at completion)*
