# Experiments — pre-registration protocol (harness checkpoint C3)

Every claim of the form "X forecasts better than Y" or "changing Z improves scores" must
come from a **pre-registered experiment**: the question set, metric, and decision rule are
frozen in a file here *before* any forecast in the experiment is scored. This is the guard
against cherry-picking and against the single-miss-patching failure mode (docs/02
§ Postmortems: a lesson is a hypothesis, not a patch).

## Lifecycle

1. **Draft** — copy `TEMPLATE.md` to `NNN-<slug>.md`, fill it in, status `draft`.
2. **Frozen** — the owner (or a checkpoint review) approves it; status `frozen`, and the
   freeze is committed *before* the first scored run. From this moment the question set,
   metric, and decision rule may not change. If a flaw is found mid-run, the experiment is
   marked `aborted` (with the reason) and a new number is registered — never edited in place.
3. **Complete** — results and the decision go into the same file's Results section;
   status `complete`. Negative and null results are recorded with the same care as wins.

Numbering is global and sequential (`001`, `002`, ...). Aborted numbers are not reused.

## What must be frozen (the template enforces this)

- **Hypothesis** — one falsifiable sentence.
- **Arms** — exact configs under comparison (settings.toml snapshots or diffs; model ids,
  effort, member rosters, coefficients). "Config is part of the experiment" (docs/02).
- **Question set** — the exact selection rule (and, once selected, the exact question ids),
  including the leakage tier per docs/02 (frozen-context / post-cutoff-resolved / live) and
  a leakage-risk label.
- **Metric + decision rule** — primary metric (usually paired Brier delta), any secondaries,
  and the pre-committed threshold for adopting/rejecting (e.g. "adopt B as default if mean
  paired Brier delta < −0.01 with 95% bootstrap CI excluding 0").
- **Power check** — expected effect size vs. N; if N is too small to detect the effect the
  experiment claims to test, say so explicitly and downgrade the conclusion language in
  advance ("directional evidence only").
- **Budget** — max LLM spend and search-credit draw.

## What is NOT allowed after freezing

Swapping questions, adding arms, changing the metric, re-running "just once more" with a
tweak, or promoting a secondary metric to primary because it looks better. Any of these =
abort and re-register.

## Index

| # | Title | Status | Decision |
|---|---|---|---|
| [001](001-ensemble-vs-baseline.md) | Ensemble (B) vs baseline (A), forward frozen-context | draft | — |
