# Architecture Decision Records

One numbered file per significant decision. "Significant" = a choice that would surprise or puzzle a future reader if undocumented (per decision review, we do **not** record every notable choice — only surprising/shaping ones... pending final ADR-granularity decision in [04-development-approach.md](../04-development-approach.md)).

## Format

```markdown
# NNNN — Title (imperative or noun phrase)

- Status: proposed | accepted | superseded by NNNN
- Date: YYYY-MM-DD
- Decided by: (human | checkpoint review | agent-within-plan)

## Context
Why a decision was needed; forces at play.

## Decision
What we chose, stated plainly.

## Alternatives considered
What else was on the table and why it lost.

## Consequences
What this makes easier, harder, or irreversible.
```

Commits implementing a decision reference its ADR number. Use the `new-adr` skill to scaffold.

## Index

- [0001 — Python engine, TypeScript web](0001-python-engine-typescript-web.md) — accepted
- [0002 — LLM calls via Claude Code CLI wrapper (subscription)](0002-llm-via-claude-cli.md) — accepted
