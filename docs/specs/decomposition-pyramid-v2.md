# Spec: Candidate C v2 — Assumption Surfacing & Verification

- Status: draft — awaiting owner review (V0 gate below)
- Date: 2026-07-12
- Parent docs: [decomposition-pyramid.md](decomposition-pyramid.md) (v1, built),
  [03-architecture-candidates.md](../03-architecture-candidates.md) (§ C guards, § grafts
  "fragility test"), owner direction 2026-07-12 ("break down the nodes based on
  assumptions as well and verify assumptions")

## Goal

v1 decomposes a question into sub-questions. v2 adds the orthogonal axis: every estimate
silently rides on **assumptions** — implicit probability-≈1 beliefs ("the referendum
happens as scheduled", "the source measures what we think it measures"). v2 makes them
explicit, **verifies the checkable ones with targeted retrieval**, and lets unverified
load-bearing assumptions discount confidence. Verification is the one form of added
granularity that *reduces* error instead of compounding it: a verified assumption
replaces a hidden guess with a fact; a verified-FALSE assumption catches a wrong forecast
before it is made (the P1 Bitcoin miss was exactly a false hidden assumption about the
resolution window).

## Design

### 1. Schema extension (pyramid.py)

Every node gains `assumptions: list[Assumption]` (default empty). `Assumption`:

- `text` — the assumption in plain words, falsifiable phrasing.
- `load_bearing: bool` — would the node's estimate change materially if this were false?
- `verifiability: "researchable" | "unverifiable"` — can a search settle it now?
- (filled at runtime) `status: "verified-true" | "verified-false" | "unverified"`,
  `evidence: str` (one-line citation or reason), `verification_cost_usd`.

The decomposer prompt is extended: for each node, surface 0–3 assumptions the estimate
rests on, marking load-bearing and verifiability. Zero assumptions is a legitimate
answer — do not manufacture filler.

### 2. Verification pass (new, pyramid_runner.py)

After decomposition, before leaf estimation:

- Collect all `researchable` assumptions across the tree (deduplicate near-identical).
- For each, run a **targeted retrieval**: 1 search (existing SearchClient stack) + a
  cheap LLM verdict call ("does this evidence verify, falsify, or fail to settle the
  assumption?") → status + one-line evidence. Budget caps in config:
  `max_assumption_verifications` per question (default 6), reusing the per-question
  fetch caps. Verification failures degrade to `unverified` with a note, never crash.
- **Verified-false handling**: the owning node's estimation prompt receives the
  falsified assumption prominently ("NOTE: the assumption X you might make is FALSE:
  <evidence>"). This is context injection, not arithmetic — the leaf reasons with
  corrected facts.

### 3. Effect on estimates (deliberately conservative in v2)

- Assumptions do NOT multiply into the combination math (that would reinvent error
  compounding through the back door).
- Confidence: unverified **load-bearing** assumptions act as a third confidence signal —
  `n_unverified_load_bearing ≥ 2` caps the record's confidence at "low",
  `= 1` caps at "medium" (thresholds in config). Raw counts stored on the record.
- Fragility pull (config, default ON, `fragility_pull_weight = 0.15`): a node with ≥1
  unverified load-bearing assumption has its estimate pulled toward its outside-view
  anchor's base rate by the configured weight in log-odds space — the docs/03 fragility
  graft, applied at node level. Set weight 0.0 to disable (the experiment can score both:
  the pull is recomputable from stored raw values).

### 4. Per-leaf targeted retrieval (v2 unlock, was queued from v1)

Evidence leaves whose sub-question is poorly covered by the shared bundle (heuristic:
zero bundle sources mention the leaf's key terms — cheap string/LLM check) get ONE
targeted search appended to their context, under the same per-question budget caps.
Assumption verification and leaf top-up share the budget; verification has priority
(config: `leaf_topup_max_searches`, default 2).

### 5. Trace, render, postmortem

`PyramidDetail` gains `assumptions` (flat list with node ids + statuses + evidence) and
the budget accounting. `render-forecast` shows assumptions per node as
verified✓ / falsified✗ / unverified○ badges with evidence lines — the "which assumption
broke?" postmortem view. Costs logged per verification call.

## Config (`[pyramid]` additions)

`max_assumption_verifications`, `leaf_topup_max_searches`, `fragility_pull_weight`,
`unverified_lb_low_cap` (=2), `unverified_lb_medium_cap` (=1), plus
`[tasks.assumption_verdict]` (cheap model — it reads one search result and votes).

## Cost estimate

v1 (~$0.20/q) + up to 6 verdict calls (~$0.01 each cheap model) + extra search credits
(≤8/question, shared budget) ≈ **$0.25–0.45/question**. Still below B.

## Out of scope (v2)

Assumption graphs across questions (shared world-model); recursive assumption
decomposition; assumptions as tradable conjuncts in the math; multi-round verification.

## Evidence plan

Experiment **003** (register after 002 freezes results are underway): C-v2 vs C-v1,
paired, same bundles, same question set shape. Primary: paired Brier delta. Secondaries:
verified-false catch rate (how often verification flips/repairs a node), fragility-pull
on/off recomputation, cost delta. v2 changes NOTHING in v1's shipped defaults until 003
reports.

## Declared checkpoints

| # | Gate | Artifact / owner action |
|---|---|---|
| **V0** | Spec review (NOW) | This document — approve before any code |
| **V1** | Vertical slice | 2–3 questions with assumption badges in rendered graphs; owner judges "are these the assumptions a careful human would have listed?" |
| **V2** | Experiment 003 freeze | Owner signs; runs alongside 001/002 |

## Agent execution plan (after V0)

1. **Opus** — schema + verification pass + budget accounting + tests (fake search/LLM).
2. **Opus** — fragility pull + confidence integration + render badges + CLI surfacing.
3. **Fable** — review, V1 slice live, present.
