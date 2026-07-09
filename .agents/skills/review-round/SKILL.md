---
name: review-round
description: Run a human decision-review round for this project - build an annotated HTML review page for a set of decisions/designs/mockups, then (on a later invocation) fold the exported feedback back into the docs and commit. Use at pre-declared checkpoints, for decision-heavy changes, or whenever the user wants to give structured feedback. Triggers - "review round", "let me review this", "I want to give feedback on these decisions".
---

# Review Round

Two phases; detect which from context.

## Phase 1 — build the review page

1. Synthesize the decision set into review units (decisions/risks/workflows, never raw doc headings). Tag each unit by movability: `evidence-settled` / `default` / `open — steer this`.
2. Build the page with the `annotated-review-html` skill (it holds the design contract). Project specifics:
   - Output beside the source docs: `docs/<topic>-review.html`.
   - Statuses: Agree / Agree with notes / Revise / Reject / Discuss.
   - localStorage key: `arh:superforecaster-<topic>:v1`.
   - Every unit gets 1–3 review prompts aimed at what the user's answer would actually change.
   - Gloss forecasting/statistics jargon inline (owner is new to the field) — or link the relevant primer in `docs/architecture-explainer.html`.
3. Validate (script syntax via node, placeholder scan), commit the page, tell the user how to export feedback (Copy feedback markdown → paste into chat).

## Phase 2 — fold feedback in

When the user pastes the exported markdown:

1. Process every section with a status or note; restate any Discuss items back in plain language and answer them in chat.
2. Apply Agree/Revise/Reject outcomes to the actual docs (VISION, 00–04, CLAUDE.md, explainer) — surgical edits, and report exactly what changed where.
3. New concepts the user flagged as unfamiliar → extend `docs/architecture-explainer.html`.
4. Decisions meeting the ADR bar → `new-adr` skill.
5. Update project memory with durable decisions.
6. Commit as one feedback-round commit; mention unresolved Discuss items as the open thread.
