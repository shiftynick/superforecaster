# E1 — Candidate B ensemble: first live runs (2026-07-10, overnight)

Five members per question — three reasoning frames (outside-first / inside-first /
status-quo) on GLM-5.2, plus outside-first on Grok-4.5 and claude-sonnet-5 — all
reading the SAME frozen research bundle, pooled in log-odds space (no extremization:
fixed mode, coefficient 1.0). Confidence derives from member spread (agreement =
confidence). Dry-run only. Spec: docs/specs/ensemble.md.

Reading the table: 'spread' is the disagreement between members measured in log-odds
(0 = unanimous; 1+ = strongly split). Earlier single-pass forecasts for the same
question are shown for comparison.

## Observations (Fable review)

1. **Spread-derived confidence behaved exactly as designed.** The World Cup question got
   tight agreement (spread 0.33 → high confidence); the Shakira chart question split the
   members hard (0.05 vs 0.58, spread 1.27 → low confidence). That question genuinely is
   contested — the low-confidence label is the honest output, and under the selective-
   forecasting principle (docs/03) it is exactly the kind of question a live bot might
   abstain on.
2. **The Pope question ran twice** (the second CLI invocation's default selection re-picked
   the soonest-closing question — selection does not yet exclude already-forecast
   questions; noted below). Silver lining: it is a free repeatability probe. Pooled 0.08 vs
   0.04 across runs; individual members moved more (GLM outside-first 0.08 → 0.01). The
   pool damps run-to-run noise but does not eliminate it — worth keeping in mind when
   reading any single number.
3. **Sonnet is the lone dissenter on Shakira** (0.05 vs 0.38–0.58 for GLM/Grok) — same
   evidence bundle, opposite conclusion. This is the crux-resolution use case (docs/03
   grafts, candidate E): a supervisor would ask *why* they disagree instead of averaging
   past it.
4. **One member failure, tolerated by design**: GLM status-quo returned empty output once
   (the pool formed from 4 survivors and the failure is recorded on the record). Second
   sighting of GLM returning empty; if it recurs, investigate (suspect: reasoning-token
   exhaustion despite the 16384 cap, or a provider-side hiccup).
5. **Cost**: $0.17–0.77 per 5-member ensemble question (cache state is the swing factor).
   The overnight demo's total live spend was $2.00.

**Known gaps queued (not fixed tonight, out of declared scope):** question selection
should optionally skip already-forecast questions; members run sequentially (parallelizing
would cut wall-clock ~5×); GLM empty-output recurrence worth a look.

**Decisions waiting on the owner:** (a) freeze [experiment 001](../experiments/001-ensemble-vs-baseline.md)
(B vs A, forward frozen-context, directional-only at this N) — the freeze pins
extremization_mode="agreement" for the experiment while the shipped default stays
fixed/1.0; (b) any roster/coefficient changes before the forward record starts accruing.

## [40795] Will the winner of the 2026 FIFA World Cup be a country that has never won before?

**Pooled: 0.13** (raw pool 0.13, coefficient 1.00 fixed) | confidence high | spread 0.33 | cost $0.5915

| member | frame | model | p |
|---|---|---|---|
| 1 | outside-first | z-ai/glm-5.2 | 0.13 |
| 2 | inside-first | z-ai/glm-5.2 | 0.12 |
| 3 | status-quo | z-ai/glm-5.2 | 0.08 |
| 4 | outside-first | x-ai/grok-4.5 | 0.19 |
| 5 | outside-first | claude-sonnet-5 | 0.15 |

_Earlier single-pass forecasts: claude-sonnet-5(blind)=0.18, codex-default(blind)=0.17, claude-sonnet-5=0.12_

## [43430] Will Pope Leo XIV visit another country other than Spain before September 2026?

**Pooled: 0.08** (raw pool 0.08, coefficient 1.00 fixed) | confidence medium | spread 0.54 | cost $0.1747

| member | frame | model | p |
|---|---|---|---|
| 1 | outside-first | z-ai/glm-5.2 | 0.08 |
| 2 | inside-first | z-ai/glm-5.2 | 0.10 |
| 3 | status-quo | z-ai/glm-5.2 | 0.07 |
| 4 | outside-first | x-ai/grok-4.5 | 0.18 |
| 5 | outside-first | claude-sonnet-5 | 0.04 |

**Pooled: 0.04** (raw pool 0.04, coefficient 1.00 fixed) | confidence medium | spread 1.12 | cost $0.4721

| member | frame | model | p |
|---|---|---|---|
| 1 | outside-first | z-ai/glm-5.2 | 0.01 |
| 2 | inside-first | z-ai/glm-5.2 | 0.07 |
| 3 | status-quo | z-ai/glm-5.2 | 0.01 |
| 4 | outside-first | x-ai/grok-4.5 | 0.12 |
| 5 | outside-first | claude-sonnet-5 | 0.08 |

_Earlier single-pass forecasts: claude-sonnet-5(blind)=0.17, codex-default(blind)=0.28, claude-sonnet-5=0.04, claude-sonnet-5=0.04, claude-sonnet-5=0.06, claude-sonnet-5=0.05_

## [43612] Will Shakira's "Dai Dai" outpeak Shakira's "Waka Waka" on the Billboard Hot 100?

**Pooled: 0.31** (raw pool 0.31, coefficient 1.00 fixed) | confidence low | spread 1.27 | cost $0.7650

| member | frame | model | p |
|---|---|---|---|
| 1 | outside-first | z-ai/glm-5.2 | 0.38 |
| 2 | inside-first | z-ai/glm-5.2 | 0.58 |
| 3 | outside-first | x-ai/grok-4.5 | 0.48 |
| 4 | outside-first | claude-sonnet-5 | 0.05 |

Member failures: status-quo/z-ai/glm-5.2: Forecast model did not return a valid JSON object after one retry: no JSON object found in model output: line 1 column 1 (char 0)

_Earlier single-pass forecasts: claude-sonnet-5(blind)=0.17, codex-default(blind)=0.19, claude-sonnet-5=0.06, z-ai/glm-5.2=0.38, x-ai/grok-4.5=0.28_
