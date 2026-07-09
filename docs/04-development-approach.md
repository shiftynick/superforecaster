# How We Build — Development Approach

*Added 2026-07-09 from decision-review follow-up. Status: proposal for discussion, like the rest of the vision.*

## The core decision

This system will be built with **heavy use of agentic AI** — most code will be written by AI agents ("vibe coded"), with the human steering through specs, reviews, and decisions. That is not a shortcut; it's a discipline with its own requirements. Everything below exists to make agent-built code **high-quality, maintainable, standards-conforming, and transparent about its decisions**.

## Principle: the repo is the agent's workspace

An agent-built codebase must be optimized for agents to work in — which happens to be the same shape that keeps it maintainable for humans:

- **`CLAUDE.md` at the root**, kept religiously current: conventions, commands, architecture map, gotchas. Agents read it every session; it is the single highest-leverage quality document in the repo.
- **Small, well-named modules** with typed interfaces — agents (and reviewers) reason about bounded contexts far better than sprawling files.
- **Config over code** (already a system requirement from the tunability decision): parameters live in config files agents can adjust without touching logic.
- **Deterministic tooling**: formatter, linter, and type-checker enforced via pre-commit hooks and CI — style debates never reach review, and agents get instant mechanical feedback.
- **Tests as guardrails**: the aggregation math (log-odds pooling, extremization, Platt) is pure-function territory — ideal for unit and property tests. Pipelines get integration tests. The evaluation harness itself is the end-to-end test.
- **Cost/token logging built in from day one** (already a system requirement — it doubles as SDLC observability).

## The SDLC loop

Every non-trivial change moves through six stages:

| Stage | What happens | Agent/human split |
|---|---|---|
| 1. **Spec** | Short doc: goal, constraints, done-when | Human intent, agent drafts |
| 2. **Plan** | Implementation plan reviewed *before* code | Agent plans (plan mode), human approves |
| 3. **Implement** | Feature branch, small focused changes | Agent codes |
| 4. **Review** | `/code-review` on every change; deep multi-agent review (`/code-review ultra`) at milestones; annotated review pages (like [VISION-review.html](VISION-review.html)) for decision-heavy changes | Agent reviews agent; human reads and adjudicates |
| 5. **Verify** | Tests + lint + type-check green; behavior confirmed against the harness where applicable. **Web-app changes additionally require live browser verification**: the agent runs the app and manually exercises the relevant flows in a real browser — clicking through, inspecting rendered state, checking the console and network — not just passing unit tests | Agent verifies, reports faithfully |
| 6. **Record** | Decision record written; merged with a descriptive commit | Agent records, human owns the history |

## Visibility into decisions: ADRs

Every significant decision gets an **Architecture Decision Record** — a short numbered file in `docs/adr/` stating context, the decision, alternatives considered, and consequences. The vision docs already work this way informally; ADRs make it systematic once code exists. Commits reference their ADR. Six months from now, "why does aggregation use a fixed coefficient?" has a findable, dated answer.

## Skills as the unit of process

Recurring development workflows get encoded as **project-specific Claude Code skills** (in `.claude/skills/`, versioned with the repo, built with the skill-creator tooling). Skills are how process discipline survives across sessions — an agent doesn't need to remember the backtest protocol if `/run-backtest` *is* the protocol. Initial roster to develop as needs emerge:

| Skill | Encodes |
|---|---|
| `new-adr` | Scaffold + register a decision record correctly |
| `run-backtest` | The pre-registered experiment protocol (question set freezing, paired scoring, leakage checks) — enforced, not remembered |
| `cost-report` | Summarize per-run and cumulative token/dollar spend |
| `add-candidate` | Scaffold a new architecture candidate behind the common engine interface |
| `forecast-postmortem` | Run the full postmortem protocol ([02 § Postmortems](02-evaluation-harness.md)): root-cause a miss from its trace, classify it, draft the lesson + verification plan |
| `review-round` | Generate an annotated review page for any decision set (the VISION-review pattern, reusable) |
| `update-explainer` | Keep [architecture-explainer.html](architecture-explainer.html) in sync as concepts/designs evolve |

The skill list itself is living — building skills for the project's own development is an explicit, ongoing workstream, not a one-time setup.

## Version control

Git from day zero — including the docs, so vision decisions get history too. Feature branches; `main` stays green; small reviewable diffs over big drops. (The project folder is not yet a git repo — `git init` + an initial commit of the existing docs is step zero, ready to run on your word.)

## Open questions (your call)

1. **Stack**: Python engine (dominant ecosystem for the forecasting/stats libraries and the metac-bot template) + TypeScript web app is the conventional split; a single TypeScript codebase is the alternative. Preference?
2. **Human gate**: do you want to approve every merge, or only milestone/decision-heavy reviews (with `/code-review` gating the rest)?
3. **ADR granularity**: every notable choice, or only choices that would surprise a future reader?
