# 005 — Reasoning-effort levels (low vs medium), candidate A

- Status: draft (awaiting owner sign-off to freeze; 003 remains reserved for C v2-vs-v1)
- Registered: 2026-07-17
- Frozen: —
- Owner sign-off: —

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
question (E-med runs first, E-low reuses the cache). Both stamped `--experiment-id 005`.

## Question set

- Selection rule (to freeze): fresh scrape, open binary questions resolving within 60
  days, first N = 40 by soonest resolve; overlap with 001/002 pinned sets permitted
  (stamping isolates scoring).
- Question ids: *(pinned at freeze)*.

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
