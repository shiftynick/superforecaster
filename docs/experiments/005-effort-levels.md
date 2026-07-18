# 005 — Reasoning-effort levels (low vs medium), candidate A

- Status: **frozen** (running)
- Registered: 2026-07-17
- Frozen: 2026-07-17 (fresh scrape immediately prior; 55 eligible, first 40 pinned)
- Owner sign-off: 2026-07-17 ("freeze 005")

## Hypothesis

Raising per-call reasoning effort from `low` (the day-one testing default, never
revisited) to `medium` measurably improves mean Brier. Effort is plausibly the largest
quality dial we own, and we have zero evidence about it.

## Arms

| Arm | Config (exact at freeze) |
|---|---|
| E-low | candidate A single pass, `[tasks.forecast_pass]` with effort="low" (current default) |
| E-med | identical, effort="medium" |

Candidate A is deliberately the vehicle: cheapest arm, single call, isolates the dial
(an ensemble would confound effort with pooling). Same research bundle shared per
question (E-med runs first, E-low reuses the cache). Retrieval is held constant across
arms at the shipped default (news=asknews, general=perplexity as of experiment 006);
only `[tasks.forecast_pass] effort` differs, via two frozen configs
(`engine/config/experiment-005-{med,low}.toml`).

**Arm distinguishing (scoring-integrity note):** both arms are the SAME method
(`checklist-a`) and SAME model (`z-ai/glm-5.2`) — the manifest matcher (method+model)
cannot tell them apart, and `ForecastRecord` does not store `effort`. So the arms are
stamped with distinct ids — E-med `--experiment-id 005-med`, E-low `--experiment-id
005-low` — and scored by a dedicated pairing script (pair by question_id across the two
stamps), the same script-scored pattern experiment 004 uses. No `.toml` manifest is
registered. Queued machinery fix: add `effort` to `ForecastRecord` + optional
`ArmSpec.effort` matching, which would let same-method/model effort arms use the standard
manifest path.

## Question set

- Selection rule (to freeze): fresh scrape, open binary questions resolving within 60
  days, first N = 40 by soonest resolve; overlap with 001/002 pinned sets permitted
  (stamping isolates scoring).
- Question ids (40, pinned 2026-07-17, soonest-resolving first):
  43716, 43507, 44373, 43892, 44422, 43873, 44011, 44017, 44256, 35710,
  44139, 44698, 43852, 44248, 43854, 39432, 43522, 43506, 44250, 43737,
  43432, 44832, 44014, 43738, 44634, 44423, 44406, 43908, 43882, 43870,
  43732, 43503, 44702, 44701, 44711, 43510, 43360, 44547, 44415, 44253.

## Metric & decision rule

- Primary: mean paired Brier delta (E-med − E-low). Secondaries: cost ratio,
  latency, confidence-label distribution shift.
- Decision rule: directional at N ≈ 40. If E-med wins the delta AND costs < 3× E-low,
  the default effort for real (non-testing) runs moves to medium — a settings.toml
  change, recorded as an amendment to the testing-defaults note.

## Budget

Max LLM spend: $15 (arm A is cheap even at medium effort; effort mostly moves reasoning
tokens). Max AskNews credits: 170 (~4/question fresh).

## Owner note

This costs real money (~$5–15) and changes nothing until its read is in. Freeze on your
word — reply "freeze 005" (or amend first).
