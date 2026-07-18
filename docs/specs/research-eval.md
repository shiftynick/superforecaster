# Spec: Research-Quality Eval Harness (task #17, piece 3)

- Status: approved-in-principle (owner: "build the eval harness — then we will compare
  perplexity", 2026-07-17)
- Date: 2026-07-17
- Parent docs: [ADR-0004](../adr/0004-retrieval-stack.md), task #17, docs/02 (evaluation
  discipline)

## Goal

A yardstick for research bundles, so provider/tier changes (Perplexity Search API first)
are adopted on evidence. Given a question + bundle: mechanical metrics (free, pure code)
plus an LLM rubric judgment; given a question + TWO bundles: a blinded pairwise verdict.

## Design

### 1. Mechanical metrics (pure, no LLM)

Per bundle: result/doc counts; unique source domains; extraction tier stats (already on
the bundle); recency (median + newest `published_at` age in days, fraction undated);
key-term coverage (question's significant terms found in titles/snippets/docs — reuse the
leaf top-up heuristic's term logic); resolution-source presence (any bundle URL's domain
appears in the question's resolution criteria / fine print text); search credits used;
approximate bundle chars. Typed `BundleMetrics`, computed by a pure function.

### 2. Rubric judge (one LLM call, `[tasks.bundle_judge]` config)

Input: the question (title/criteria/fine print/window) + a **provider-blinded** bundle
rendering (provider fields stripped; URLs kept — domains are evidence of quality, and
provider identity is not inferable from them in general). Output (structured):
1–5 score + one-line justification per dimension:

- coverage — are the question's core factors and resolution mechanics addressed?
- source_quality — authoritative/primary vs content-farm.
- recency — is the latest relevant development present?
- relevance — how much of the bundle is actually about the question?
- resolution_grounding — is the source the question will resolve by (or its data)
  present?

Plus `overall` (1–5) and `missing` (one line: the most important thing the bundle lacks).
Degenerate-output guard reused. Retry-once pattern reused.

### 3. Pairwise compare (one LLM call)

Input: question + two blinded bundles labeled X and Y — assignment randomized per call,
mapping recorded on the eval record. Output: per-dimension winner (X/Y/tie) + overall
winner + one-line reason. Never run with the absolute judge in the same call.

### 4. Persistence

`BundleEvalRecord` JSONL at `data/bundle-evals/` (config `[data] bundle_eval_dir`):
question_id, bundle fingerprint (stable hash of sorted result URLs + doc URLs), retrieval
config summary (providers/max_results — recorded for analysis, never shown to the judge),
metrics, rubric scores, pairwise verdicts (with X/Y mapping), judge task config, cost,
timestamps. Append-only, latest-wins accessors, same store pattern as everything else.

### 5. CLI

- `eval-research --post-id N [--fresh|--cached]` — build/load a bundle under current
  settings, compute metrics, run the rubric judge, store + echo a summary line.
- `compare-research --post-id N --alt-config path/to/settings.toml` — build bundle A
  under current settings and bundle B under the alt config (fresh both, same moment),
  run metrics on both + blinded pairwise, store both evals + the verdict, echo summary.
  This is the Perplexity test's entry point: the alt config differs only in retrieval
  providers.

### 6. Out of scope (this build)

The Perplexity client itself (next step); the Brier-link experiment (registered when the
first provider comparison is designed — pairwise verdicts + stamped forecasts make it
possible); rubric-weight tuning (weights are reporting-only for now, no composite scalar
beyond `overall`).

## Cost

Mechanical metrics free; rubric ≈ 1 cheap call (~$0.01–0.03) per bundle; pairwise ≈ 1
call per comparison. A 20-question provider comparison ≈ $1–2.

## Checkpoints

| Gate | Artifact |
|---|---|
| R1 | Harness built + demo: eval-research on 3 real cached bundles, owner skims the rubric outputs for face validity |
| R2 | Perplexity comparison designed (own experiment doc) — owner freezes |
