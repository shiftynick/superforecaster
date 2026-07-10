---
name: experiment-checkin
description: Check in on the status of all pre-registered forecasting experiments - coverage, resolutions, interim scores, budget burn - and surface anything needing a decision. Use when the user asks "how are the experiments doing", "experiment status", "any results yet", or at the start of a session touching experiment work.
---

# Experiment check-in

Experiments run asynchronously over weeks (forecasts now, resolutions trickle in later).
This skill is the periodic pulse-check.

## Steps

1. **Refresh + report**: from the repo root run
   `uv run --project engine superforecaster experiment-status --refresh`
   (`--refresh` re-checks resolutions for pinned unresolved questions first; drop it
   only if offline). Manifests live in `docs/experiments/*.toml`.
2. **Translate for the owner** (new to forecasting jargon — gloss terms):
   - Coverage: are both arms fully forecast? If not, name the gap and its cause.
   - Resolutions: how many pinned questions have resolved, how many remain, when the
     next ones are due.
   - Interim score: report the paired Brier delta ONLY with its resolved-N and the
     registered caveat (underpowered until the doc's power threshold — do not let a
     small-N delta read as a verdict).
   - Budget: spend vs. registered cap per experiment; flag any unknown-cost records.
3. **Cross-check the contract**: skim each experiment's markdown doc (same directory)
   for its decision rule; if a pre-registered threshold has been reached, say so
   explicitly — that is a decision moment for the owner, not something to act on
   unilaterally.
4. **Surface, don't fix**: any anomaly (missing arm records, stalled runs, resolution
   disputes on Metaculus, budget overrun) gets reported with evidence. Fixes to
   experiment machinery are fine; changes to a FROZEN experiment's design are never
   made — a flawed frozen experiment is abandoned and re-registered, not edited.
5. **Close the loop**: end with a one-line "next check-in worth doing around <date>"
   based on the soonest pending resolution.

## Notes

- Resolution refreshes are manual by design (owner's cost-control decision) — this
  skill IS the manual kickoff; do not create schedulers from here.
- When a new experiment freezes, it must ship BOTH the markdown contract and the
  `.toml` manifest, or this skill will not see it.
