# 002 — Candidate C pyramid vs. A and B (forward, frozen-context, three-arm)

- Status: **frozen** (running)
- Registered: 2026-07-12
- Frozen: 2026-07-12 (fresh scrape immediately prior; 54 eligible, first 40 pinned)
- Owner sign-off: 2026-07-12 ("proceed - freeze 002 and spec C v2")

## Hypothesis

The decomposition pyramid (candidate C, v1 as at freeze commit) achieves a lower mean
Brier score than candidate A (single pass) on the same questions with the same research
bundles. Secondary hypothesis (the P0 "raw visibility" question): the holistic blend
improves on the raw structural estimate — i.e. blended-C beats raw-C.

## Arms

| Arm | Config (exact, at freeze commit) |
|---|---|
| A | `[tasks.forecast_pass]` single pass (settings.toml as at freeze) |
| B | `[ensemble]` default roster (6 members), extremization "fixed"/1.0 (shipped default — NOTE: differs from experiment 001's arm B, which pinned "agreement") |
| C | `forecast --pyramid`, `[pyramid]` defaults (depth 2, blend 0.5, guarded-product) |
| C-raw | NOT a separate run: scored from arm C's records using `structural_probability` (zero extra cost) |

All arms run per question in one window (C → B → A per question) sharing one cached
research bundle: identical evidence, architecture is the only variable. Fresh freeze and
fresh runs — experiment 001's arm A/B records are NOT reused (they were forecast on
2026-07-10 with different information and older prompt machinery).

## Question set

- Selection rule (to freeze): re-scrape first, then all open binary questions in the
  snapshot store with `scheduled_resolve_time` within 60 days of the freeze date,
  **excluding nothing** — including questions pinned by experiment 001 is fine (001's
  arms match only records made on its own frozen configs/dates; 002's records are new).
  Capped at the first N = 40 by soonest resolve time.
- Decomposition-rate stratification (pre-registered): report results separately for
  questions where C produced a multi-node tree vs. single-leaf trees. On single-leaf
  questions C ≈ A + holistic double-check, so the pooled delta understates whatever
  decomposition itself does. The multi-node stratum is the real test of the founding
  idea; the single-leaf rate is itself a registered result.
- Leakage tier: frozen-context forward (docs/02's strongest) — leakage-impossible.
- Question ids (40, pinned 2026-07-12, soonest-resolving first; 35 overlap experiment
  001's set — permitted by the selection rule, records distinguished by method+date):
  40795, 43716, 43507, 44373, 43892, 44422, 43873, 44011, 44017, 44256,
  35710, 44139, 44698, 43852, 44248, 43854, 39432, 44562, 43522, 43506,
  44250, 43737, 43432, 44832, 44014, 43738, 44634, 44423, 44406, 43908,
  43882, 43870, 43732, 43503, 44702, 44701, 43510, 43360, 44547, 44415.
- Note (freeze): C-raw scoring reads `structural_probability` from arm C's records at
  analysis time; the `experiment-status` tool gains that column as ordinary machinery
  work (does not alter this design).

## Metric & decision rule

- Primary: mean paired Brier delta (C − A) over resolved questions.
- Secondaries (diagnostic): C-raw − C-blended (does the blend help?); C − B; spread of
  structural-vs-holistic gap vs. error (does disagreement predict misses?); cost per
  question; decomposition rate; per-stratum deltas.
- Decision rule, pre-committed: N < 100 ⇒ directional evidence only; no default changes.
  If blended-C beats raw-C on ≥ 60% of resolved multi-node questions, the 0.5 blend
  weight survives; a raw-C win moves the P3 discussion toward higher structural weight —
  discussed, not auto-adopted.

## Power check

Same as 001: N ≈ 40 detects only large effects; this is registered as directional. The
forward records keep accruing for a later confirmatory pass at N ≥ 300.

## Budget

Max LLM spend: $45 (~$0.85/question across the three arms). Max AskNews credits: 200
(bundles are shared across arms — ~4/question fresh). Hard stop at either cap.

## Results

*(at completion)*
