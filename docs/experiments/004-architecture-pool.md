# 004 — Architecture-level pool (scoring-time, derived from experiment 002)

- Status: **frozen**
- Registered: 2026-07-17
- Frozen: 2026-07-17 (before any 002-pinned question beyond q40795 resolved)
- Owner sign-off: 2026-07-17 ("i concur with your ideas proceed")

## Hypothesis

The portfolio decision's core bet (docs/03 § Portfolio): an equal-weight log-odds pool of
the three architecture candidates' forecasts beats the best individual candidate on mean
Brier. Diverse systems pooled should out-score their members even when no member
dominates.

## Design — no new forecasts; a frozen scoring rule over frozen inputs

**Inputs (already frozen):** experiment 002's stamped records (`experiment_id="002"`) —
for each pinned question, the LATEST record per arm: A (`checklist-a`), B (`ensemble-b`
pooled), C (`pyramid-c` blended). Same 40 questions, same research bundles, forecasts all
made 2026-07-12..16, before any resolution.

**Derived arms (computed at analysis time; the rule is frozen HERE, the computation may be
implemented any time):**

| Arm | Rule |
|---|---|
| P-ABC | `pool_log_odds([pA, pB, pC], [1/3, 1/3, 1/3])` per question |
| P-BC | `pool_log_odds([pB, pC], [1/2, 1/2])` (the two "real" architectures, excluding the floor) |
| C-raw-pool variant | NOT registered — one pool rule, no shopping |

`pool_log_odds` is the existing property-tested function in `engine/src/superforecaster/
aggregation.py`, applied to the records' clamped `probability` values. Equal weights are
pre-committed: learned weights require resolved outcomes and belong to a later
experiment, not this one.

## Question set

Exactly experiment 002's 40 pinned questions. A question enters scoring when it resolves
yes/no AND all pooled members have a stamped 002 record for it (all 40 currently do).
q40795 (World Cup) resolved BEFORE this registration; it is **excluded from 004's primary
metric** (its outcome was knowable at freeze time) and may be reported separately,
clearly labeled.

## Metric & decision rule

- Primary: mean paired Brier delta, P-ABC − best-single-arm, where "best single arm" is
  pre-committed as **arm B** (the dossier's predicted strongest built candidate) — NOT
  chosen after seeing results.
- Secondaries (diagnostic): P-ABC vs A, vs C; P-BC vs B; per-stratum (multi-node vs
  single-leaf C questions, reusing 002's stratification).
- Decision rule: N < 100 ⇒ directional only. If P-ABC beats B on ≥ 60% of resolved
  questions with a negative mean delta, the architecture pool is promoted to "presumptive
  reporting default" for future runs (still dry-run; no live implications), pending
  confirmatory N.

## Power check

Bound by 002's N ≈ 39 scorable questions — directional only, same as 001/002.

## Budget

$0 LLM. Analysis compute only. (This is the experiment's point: the portfolio's first
test is free because the members already ran.)

## Machinery note (not part of the frozen design)

`experiment-status` support for derived/pooled arms is ordinary scoring machinery and may
be built whenever; until then the doc's rule is computed by script. No `.toml` manifest is
registered (manifests describe record-matching arms; 004 has none of its own).
