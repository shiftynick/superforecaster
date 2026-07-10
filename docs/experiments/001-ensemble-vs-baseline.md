# 001 — Candidate B ensemble vs. candidate A baseline (forward, frozen-context)

- Status: **frozen** (running)
- Registered: 2026-07-09
- Frozen: 2026-07-10
- Owner sign-off: 2026-07-10 ("The rest is good" — with one pre-freeze amendment: add a
  frontier member from a lab not already on the roster). Second pre-freeze amendment by
  agent: selection window widened 30 → 60 days — the snapshot store held only 8 open
  binary questions resolving within 30 days (N far below target); 60 days yields 51
  eligible. Both amendments made before any arm ran.

## Hypothesis

The candidate-B ensemble (6 members: 3 reasoning frames on GLM-5.2 + outside-first on
Grok-4.5, claude-sonnet-5, and openai/gpt-5.6-terra; log-odds pool; agreement-scaled
extremization 1.0–1.3) achieves a lower mean Brier score than candidate A (single
outside-first GLM-5.2 pass) on the same questions with the same research bundles.

Owner FYI at freeze: candidates C, D, E are expected to be built during this experiment's
resolution window; each gets its own pre-registered experiment (they may reuse this
question set and cached research bundles where windows overlap) — this experiment stays
A-vs-B only.

## Arms

| Arm | Config (exact) |
|---|---|
| A | `[tasks.forecast_pass]` provider=openrouter, model=z-ai/glm-5.2, effort=low, max_tokens=16384; research on (settings.toml `[retrieval]` as at freeze commit) |
| B | `[ensemble]` default roster and coefficients as at freeze commit, extremization_mode="agreement" (note: shipped default is "fixed"/1.0 — the freeze pins the experimental value); same research bundles as arm A (shared via the research cache) |

Both arms forecast each question within the same run window so they share one cached
research bundle per question (identical evidence; the architecture is the only variable).

## Question set

- Selection rule (frozen 2026-07-10): all open binary questions in the snapshot store with
  `scheduled_resolve_time` within **60 days** of the freeze date (amended from 30 — see
  status note), capped at the first N = 40 by soonest resolve time.
- Leakage tier: **frozen-context forward** (docs/02's strongest tier) — forecasts are made
  before resolution with the research bundle frozen into each record; leakage-impossible
  by construction.
- Question ids (40, pinned 2026-07-10, soonest-resolving first):
  43937, 40795, 43716, 43612, 43507, 44373, 43892, 44422, 43873, 44011,
  44017, 44256, 35710, 44139, 43852, 44248, 43854, 39432, 43522, 43506,
  44250, 43737, 43432, 44014, 43738, 44634, 44423, 44406, 43908, 43882,
  43870, 43732, 43503, 43430, 43510, 43360, 44547, 44415, 44253, 44241.

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
