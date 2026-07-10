# Spec: Candidate B — Diverse Ensemble (v1)

- Status: in progress (owner pre-approved the chunk 2026-07-09 before stepping away;
  checkpoint E1 artifact awaits their review)
- Date: 2026-07-09
- Parent docs: [03-architecture-candidates.md](../03-architecture-candidates.md) (candidate B
  + field-survey grafts), [01-prior-art.md](../01-prior-art.md) (§3 aggregation evidence),
  [02-evaluation-harness.md](../02-evaluation-harness.md) (pre-registration requirement)

## Goal

N diverse forecasting passes over **one shared research bundle**, pooled in log-odds space
by the existing `aggregation.py` math. Diversity comes from two axes, both config-driven:

- **Reasoning frames** (field-survey graft — heterogeneous lenses, not temperature):
  - `outside-first` — reference class and base rate before any case-specific reasoning
    (the current checklist order).
  - `inside-first` — causal, question-specific mechanics first; base rate as a cross-check.
  - `status-quo` — "what happens if nothing changes"; burden of proof on change; explicit
    fragility test ("what must go wrong for this to be wrong").
- **Models** — any provider/model the LLM layer supports, per member.

## Design

### `ensemble.py`

- Frame definitions: named system-prompt addenda layered onto `CHECKLIST_PROMPT` (the
  checklist stays the spine; the frame reorders/reweights the steps). Frames live in code
  (they are prompts, i.e. logic); *which* frames run is config.
- `run_ensemble(question, research, members, settings) -> EnsembleResult`: run each member
  via the existing `run_forecast_pass` machinery (same JSON contract, retry, cost logging),
  collect per-member `ForecastRecord`s, pool.
- **Pooling**: `pool_log_odds` over member probabilities (equal weights v1; per-member
  learned weights are a later, backtest-earned feature).
- **Extremization** (config): `mode = "fixed"` applies `extremize(p, coefficient)`
  (default coefficient **1.0** = none — dossier: shared-information LLM ensembles warrant
  mild/none); `mode = "agreement"` applies the agreement-scaled variant (graft): coefficient
  interpolates from `agreement_min_coefficient` (at max disagreement) to
  `agreement_max_coefficient` (at unanimity), driven by the std-dev of member log-odds.
  Pure function in `aggregation.py` (`agreement_scaled_coefficient`), property-tested.
- **Ensemble confidence**: derived from member spread (config thresholds on log-odds
  std-dev → low/medium/high) — disagreement is the honest confidence signal.
- One member failing (LlmError) is logged and skipped; the pool forms from survivors
  (minimum 2, else the ensemble fails for that question).

### Config (`settings.toml`)

```toml
[ensemble]
extremization_mode = "fixed"        # "fixed" | "agreement"
extremization_coefficient = 1.0
agreement_min_coefficient = 1.0
agreement_max_coefficient = 1.3
# spread thresholds (std dev of member log-odds) for ensemble confidence
confidence_high_max_spread = 0.4
confidence_low_min_spread = 1.2

[[ensemble.members]]
frame = "outside-first"
provider = "openrouter"
model = "z-ai/glm-5.2"
effort = "low"
max_tokens = 16384
# ... more members: frames × {glm-5.2, grok-4.5, claude sonnet}
```

Default member set (v1 demo): 3 frames × GLM-5.2, plus outside-first on Grok-4.5 and
claude-sonnet-5 → **5 members**, dominated by the cheap model, cross-provider on the
default frame.

### Storage

Member records are ordinary `ForecastRecord`s tagged with new optional fields
`method` (`"checklist-a"` default / `"ensemble-b-member"`) and `frame`. The pooled result
is stored as its own `ForecastRecord` (`method = "ensemble-b"`) carrying an
`ensemble` detail object (member question-ids/probabilities/frames/models, spread,
coefficient applied, total cost) — everything a postmortem needs, one store.

### CLI

`superforecaster forecast --ensemble` — same question selection, research/cache, and store
as candidate A; the flag swaps the single pass for the ensemble. `--no-research` composes.

## Out of scope (v1)

Per-member learned weights (needs resolutions), Platt layer wiring (needs resolutions),
per-category anything, candidates C/D/E, tournament submission.

## Declared checkpoints

| # | Gate | Artifact |
|---|---|---|
| **E1** | Build + live dry-run demo done | Comparison doc in `docs/reviews/`: per-member forecasts, pooled number, spread, extremization effect, costs — owner reviews in the morning |

Hard caps for the autonomous run: dry-run only, ≤6 live questions, ≤$2 LLM spend.

## Verification

Four gates green; pooling/extremization math property-tested; ensemble runner tested with
fake LLM clients (member failure, minimum-survivor rule, config parsing); live demo within caps.

## Agent execution plan

1. **Opus** — `agreement_scaled_coefficient` in aggregation.py (+ property tests); ensemble.py
   (frames, runner, pooling, confidence-from-spread); config section + loader.
2. **Opus** — storage fields, CLI `--ensemble`, tests end-to-end (fake clients).
3. **Fable** — review both, run live demo, write E1 checkpoint doc, C3 pre-registration
   format draft alongside.
