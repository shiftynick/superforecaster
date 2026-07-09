# Agent workspace guide (Codex and other non-Claude agents)

**The canonical agent guide is [CLAUDE.md](CLAUDE.md) — read it first.** Everything there (repo map, commands, conventions, SDLC rules, checkpoints) applies to every agent working in this repo regardless of vendor; only the filename is Claude-flavored.

One rule that exists *because* both Claude and Codex agents work here:

## Skills live in two mirrored directories

- `.claude/skills/` — read by Claude agents
- `.agents/skills/` — read by Codex/other agents

These must stay **byte-identical**. Any skill you create, edit, or delete must be applied to **both** directories in the same change. Before committing skill changes, verify the trees match (e.g. compare recursive file hashes). If you find them diverged, treat the union of newest versions as truth, sync both, and note it in the commit message.
