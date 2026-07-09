# Superforecaster — agent workspace guide

Personal forecasting system: give it a resolvable question, it researches and returns a calibrated probability + confidence + legible argument. Currently: **discovery/foundation phase — engine code is just beginning; read `docs/VISION.md` before designing anything.**

## Repo map

| Path | What |
|---|---|
| `docs/VISION.md` | What we're building — read first |
| `docs/00`–`04` | Discovery plan, prior-art dossier, evaluation harness (incl. postmortem protocol), architecture candidates A–E (portfolio: build all, ensemble across them), development approach |
| `docs/architecture-explainer.html` | Concept primers + architecture diagrams (keep in sync via `update-explainer` intent) |
| `docs/adr/` | Architecture Decision Records — see its README for format; scaffold with the `new-adr` skill |
| `engine/` | Python forecasting engine (uv project) |
| `web/` | TypeScript web app (does not exist yet — see ADR-0001) |
| `.claude/skills/` | Project skills — versioned process; prefer invoking/extending a skill over ad-hoc process |

## Commands (run from `engine/`)

```
uv sync              # install/refresh env (incl. dev deps)
uv run pytest        # tests
uv run ruff check .  # lint
uv run ruff format . # format
uv run mypy .        # type-check
```

All four must be green before any commit. Pre-commit hooks enforce ruff (`uv run pre-commit install` once per clone).

## Conventions

- **Python ≥3.11, fully typed** — `mypy` strict-leaning; no untyped public functions.
- **Config over code**: tunables (source weights, ensemble size, extremization coefficient, confidence thresholds, model/effort per task type) live in config files, never as literals in logic. This is a vision-level requirement.
- **Small typed modules**; pure functions for the aggregation math (log-odds pooling, extremization, Platt) — they get property tests.
- **Cost visibility**: every code path that calls an LLM must record tokens/cost.
- **Model configurability**: any LLM call site takes its model + effort from per-task-type config, never hard-coded.

## SDLC rules (non-negotiable; full text in docs/04-development-approach.md)

- **Spec → plan → implement → review → verify → record.** Plans are approved before code and **pre-declare human checkpoints**.
- **Bounded autonomy**: vertical slice before broad build-out; stop and surface if the plan turns out wrong; never run past a declared checkpoint; checkpoints deliver minutes-reviewable artifacts (demo / annotated review page / decisions-called-out summary).
- **Merges**: routine in-plan changes are gated by `/code-review` + green verification — the human does not approve every merge; decision-weight items wait for checkpoint sign-off.
- **Web-app verification**: changes to `web/` require actually running the app and manually testing the relevant flows in a real browser (click through, inspect console/network) — unit tests alone are insufficient.
- **Significant decisions get an ADR** (docs/adr/); commits reference their ADR.
- **Forecast-science guardrails**: experiments are pre-registered (question set + metric frozen first); a lesson from a single missed forecast is a hypothesis, not a patch — verify on backtest before changing config (docs/02 § Postmortems).

## The user

Owner is new to forecasting/statistics jargon — gloss terms plainly on first use; extend `docs/architecture-explainer.html` when a new concept becomes load-bearing.
