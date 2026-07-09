---
name: new-adr
description: Scaffold and register an Architecture Decision Record in docs/adr/. Use when a significant decision has been made (or needs proposing) — a choice that would surprise or puzzle a future reader if undocumented. Triggers - "write an ADR", "record this decision", "why did we choose X should be documented".
---

# New ADR

1. **Number**: read `docs/adr/README.md`'s index; next zero-padded 4-digit number.
2. **Bar check**: confirm the decision meets the bar (would its absence surprise a future reader?). If it's routine, say so and stop — do not create noise ADRs.
3. **Write** `docs/adr/NNNN-<kebab-slug>.md` using exactly the format in `docs/adr/README.md` (Status / Date / Decided by / Context / Decision / Alternatives considered / Consequences).
   - Status is `proposed` unless the human already decided (checkpoint answer, explicit instruction) — then `accepted`.
   - "Decided by" must be honest: human, checkpoint review, or agent-within-plan.
   - Alternatives must name what actually lost and why — not strawmen.
4. **Register**: add one line to the Index in `docs/adr/README.md`.
5. **Cross-link**: if the decision changes VISION.md, docs/02–04, or CLAUDE.md, update those in the same change.
6. **Commit** (when part of a committing change): reference the ADR number in the commit message, e.g. `feat: log-odds pooling (ADR-0007)`.
