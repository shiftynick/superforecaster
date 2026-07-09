# 0001 — Python engine, TypeScript web

- Status: accepted
- Date: 2026-07-09
- Decided by: human (option selection during foundation planning)

## Context

The system has two major components with different ecosystem gravity: the forecasting engine (retrieval, ensemble passes, aggregation math, harness, tournament bot) and a web application for working with it. A single-language codebase simplifies tooling; a split follows each component's ecosystem.

## Decision

**Python for the engine** (everything under `engine/`), **TypeScript for the web app** (future `web/`). The engine is API-first; the web app is a client of it.

## Alternatives considered

- **TypeScript end-to-end** — one language, shared types, one toolchain. Rejected because the forecasting ecosystem is Python-native: the official Metaculus bot template and `forecasting-tools` package, the stats stack for calibration/aggregation work (numpy/scipy/sklearn for Platt scaling, property-testing the pooling math), and most prior-art reference code. Porting all of that costs more than maintaining two toolchains.

## Consequences

- Engine work starts immediately on `uv` + ruff + mypy + pytest; web toolchain decisions are deferred until the web app design round.
- The engine must expose a clean API boundary (the web app — and any future client — talks to it over HTTP/JSON), which was already a vision requirement ("interface is a client, not the core").
- Cross-boundary types get defined once in the API schema, not shared source.
