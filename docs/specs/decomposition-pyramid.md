# Spec: Candidate C — Recursive Decomposition Pyramid (v1)

- Status: **P0 approved with revisions** (owner review 2026-07-12, via annotated review page;
  revisions folded in below — grounding triage, combination_mode, raw-vs-blended recording,
  provisional confidence, render-forecast deliverable)
- Date: 2026-07-12
- Parent docs: [VISION.md](../VISION.md), [03-architecture-candidates.md](../03-architecture-candidates.md) § C
  (incl. design guards + refiner compatibility), [02-evaluation-harness.md](../02-evaluation-harness.md)

## Goal

The owner's founding idea, built honestly: decompose a question into the sub-questions
that must be answered first, down to a **bedrock** of researchable facts and base rates;
estimate the bedrock; combine upward with explicit structure into a final probability +
confidence + a **legible argument graph** — the trace a human can actually audit.

The dossier found no evidence either way on decomposition for calibration (docs/03 § C):
this build is one of the first direct tests. The known theoretical risk is **conjunctive
error compounding** — multiply four slightly-off probabilities and the product is badly
off, almost always too low. The design below treats that risk as the central enemy.

## Design

### 1. Decomposition pass (one LLM call)

Question + research bundle → a typed decomposition **tree** (v1: tree, not DAG), JSON
via structured output:

- Node fields: `id`, `question` (the sub-question, plainly worded), `node_type`,
  `reasoning_type` (default `"checklist"`; extensible — a proven refiner can become a
  node-level reasoning mode later, per docs/03), `children`.
- `node_type` one of:
  - `conjunction` — resolves YES only if all children do (P = product, guarded below)
  - `disjunction` — resolves YES if any child does (P = 1 − ∏(1−p), same guard)
  - `conditional` — ordered chain: P = P(c₁) · P(c₂ | c₁) · … (children carry
    "given the previous steps happened" in their wording)
  - `leaf` — bedrock: a base rate, a researchable fact, or a direct estimate; no children
- Structural caps (config): `max_depth` (default **2**), `max_children` (default **4**),
  and ≤ **3** multiplied conjuncts per node (docs/03 guard) — the decomposer prompt states
  the caps and the parser enforces them (over-cap trees are pruned to a leaf with a note
  in the trace). Owner note (P0): depth 2 is a testing default; expect to raise it for
  real runs once the vertical slice shows the failure modes — it is config, never code.
- The decomposer may return a single leaf ("this question does not decompose") — that is
  a legitimate answer, recorded as such, and the pyramid then equals a single careful pass.

### 2. Leaf estimation (one cheap LLM call per leaf)

**Grounding triage (owner revision at P0):** the decomposer labels every leaf
`grounding = "model-knowledge" | "evidence"`. Model-knowledge leaves are stable facts and
base rates the model can answer confidently without search (they skip the research-bundle
context entirely — cheaper, and avoids diluting the prompt); evidence leaves concern
current events and MUST ground in the shared research bundle. The label is stored per
node so postmortems can audit whether "model-knowledge" was ever claimed on something
that actually needed evidence.

Each leaf runs the existing `run_forecast_pass` machinery (checklist prompt + the leaf's
sub-question); evidence leaves see the **shared research bundle** — same evidence for
every node, no per-leaf retrieval in v1 (cost + leakage control). Every leaf must carry
an **outside-view anchor**: the prompt requires a reference class + base rate before any
adjustment (docs/03 guard). Leaf output: probability, confidence, rationale, anchor.

### 3. Upward combination (pure math, property-tested)

**`combination_mode` (owner note at P0: "may not be a straight probability
multiplication kind of thing"):** v1 ships `"guarded-product"` (below) because it is
transparent and property-testable, but the mode is config and the module's interface
takes (node, children estimates) → estimate, so alternatives slot in. Registered v2
hypothesis: `"llm-synthesis"` — an LLM at each internal node reasons over its children's
answers to produce the parent estimate (handles correlation and partial evidence the way
multiplication can't; costs one call per internal node). It enters via a pre-registered
experiment against guarded-product, like everything else.

- Conjunction/disjunction/conditional math in `aggregation.py`-style pure functions.
- **Independence guard**: naive products assume independent children — real sub-events
  correlate, so products systematically overshoot toward 0. Mitigation, in order:
  (a) per-node probability clipping to [0.03, 0.97] before multiplying (config);
  (b) the conjunct cap above;
  (c) the holistic blend below — the structural estimate never stands alone.
- **Holistic sanity check** (docs/03 guard): one direct forecast of the root question
  (same machinery as candidate A) runs alongside the pyramid. Final probability =
  log-odds blend of structural and holistic estimates, weight in config
  (`holistic_blend_weight`, default **0.5** — the honest prior that we don't yet know
  which side to trust; the experiment will tell us).
- **Raw visibility (owner revision at P0):** the record stores `structural_probability`,
  `holistic_probability`, and the blended final as three separate fields — so unblended C
  is always inspectable, and the P2 experiment scores BOTH raw-C and blended-C from the
  same records at zero extra cost.

### 4. Confidence (provisional — owner flagged uncertainty at P0)

Two raw signals are ALWAYS stored on the record: (a) the weakest leaf confidence on any
multiplied path; (b) the structural–holistic log-odds gap. The v1 *label* maps them
conservatively (min of the two, gap thresholds reused from the ensemble config) — but
this mapping is explicitly provisional. Neither we nor the literature knows which signal
predicts misses; resolved forecasts will (the experiments' spread-vs-error secondary
metric is exactly this analysis). The mapping lives in config and gets revisited with
data, not argued about now.

### 5. Record + trace

`ForecastRecord` with `method="pyramid-c"`; the full tree (nodes, estimates, anchors,
grounding labels, rationales, combination steps, structural/holistic/blended estimates)
stored in the record — this IS the legible argument, and the web app's future rendering
target. Cost = sum of all LLM calls (decompose + leaves + holistic), logged per node.

### 6. `render-forecast` — the argument view without web dev (owner revision at P0)

A CLI command: `superforecaster render-forecast <question_id> [--out path.html]` renders
any stored pyramid record (and, degraded gracefully, any other method's record) into a
**self-contained HTML page** — the tree as nested cards: each node's question, estimate,
anchor, grounding label, rationale, and the combination math, with
structural/holistic/blend shown side by side. Deterministic template (no LLM call), no
server, no external assets — the same pattern as our review pages. This is the interim
argument UX until the web app exists, and its template becomes the web app's first
component spec.

## Config (`[pyramid]` in settings.toml — all tunable, none hard-coded)

`max_depth`, `max_children`, `max_conjuncts`, `leaf_clip_low/high`,
`holistic_blend_weight`, plus `[tasks.decompose]` and leaf/holistic model configs
(decomposer wants a stronger model; leaves can run cheap).

## Cost estimate

Decompose (1 call, stronger model ~$0.05) + ≤8 leaves (cheap, ~$0.02 each) + holistic
(~$0.02) ≈ **$0.15–0.35/question** — between A (~$0.02) and B (~$0.50).

## Out of scope (v1)

Per-leaf targeted retrieval; DAG-shaped graphs (shared sub-questions); cross-node
correlation modeling beyond the guards; multiple-choice/numeric questions; refiners
(separate module per docs/03); any live-tournament use.

## Declared checkpoints

| # | Gate | Artifact / owner action |
|---|---|---|
| **P0** | Spec review (NOW) | This document — owner approves design + checkpoint plan before any code |
| **P1** | Vertical slice | 2–3 real questions end-to-end; owner reviews the printed argument graphs (minutes-reviewable) — the test is "is the argument legible and sane?", not the Brier |
| **P2** | Experiment | Pre-registered experiment (C vs A vs B, shared bundles, frozen question set) — owner signs the freeze, then it accrues alongside 001 |

## Agent execution plan (after P0)

1. **Opus** — decomposition schema (incl. grounding + reasoning_type) + parser +
   structural caps + combination math with property tests (guards: clipping, conjunct
   cap, blend; combination_mode seam).
2. **Opus** — pyramid runner (decompose → leaves → holistic → blend → record) + CLI
   `forecast --pyramid` + `[pyramid]` config plumbing + `render-forecast` HTML view.
3. **Fable** — review both, run the P1 vertical slice live, present rendered argument
   graphs (the render-forecast output IS the P1 artifact).
