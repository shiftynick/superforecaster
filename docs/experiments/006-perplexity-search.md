# 006 — Perplexity Search API vs. Tavily as general-web provider (bundle quality)

- Status: **frozen** (design owner-approved in chat 2026-07-17: "key added as
  PERPLEXITY_API_KEY - proceed with R2"; matches the design presented at R1)
- Registered: 2026-07-17
- Frozen: 2026-07-17
- Parent: docs/specs/research-eval.md (checkpoint R2), task #17, ADR-0004

## Hypothesis

Replacing Tavily with the Perplexity **Search API** (raw ranked results — NOT Sonar
synthesized answers, per ADR-0004's rejection rationale) as the general-web provider
produces measurably better research bundles.

## Design — one isolated change, blinded pairwise judging

- **Baseline config:** current settings.toml retrieval (news=asknews, general=tavily).
- **Alt config:** identical except general_provider="perplexity". AskNews unchanged in
  both (isolates the general-web slot; changing two providers at once would confound).
- Per question: `compare-research` builds BOTH bundles fresh in the same run (same
  moment, same query-gen output where the pipeline allows), computes mechanical metrics
  on each, and runs ONE blinded pairwise judgment (randomized X/Y, mapping recorded).
  The judge never sees provider identities.
- Judge: `[tasks.bundle_judge]` pinned for this experiment to a NEUTRAL third-party
  model — claude-sonnet-5 via claude-cli (neither provider under test is an LLM, and
  the judge model vendor has no stake in the search comparison). Effort low.

## Question set (20, pinned 2026-07-17, soonest-resolving first)

43716, 43507, 44373, 43892, 44422, 43873, 44011, 44017, 44256, 35710,
44139, 44698, 43852, 44248, 43854, 39432, 43522, 43506, 44250, 43737.

Overlap with 001/002 pinned sets is irrelevant: this experiment creates bundle-eval
records only, never ForecastRecords — no experiment-scoring contamination is possible.

## Metric & decision rule (pre-committed)

- Primary: pairwise **overall** win rate over the 20 questions (ties count half).
- Decision rule: Perplexity config wins ≥ **60%** → adopted as the default
  general_provider in settings.toml (a config change, recorded as an ADR-0004
  amendment note). Win rate in (40%, 60%) → no change (Tavily stays, it's free).
  ≤ 40% → Perplexity rejected for this slot.
- Secondaries (diagnostic only): per-dimension win rates; mechanical metric deltas
  (recency, domain diversity, resolution-source hits); cost per bundle.

## Budget

≤ $5 Perplexity spend; ~80 AskNews credits (2 bundles × ~2 news calls × 20 questions);
~40 judge calls ≈ $1–2 on the subscription (notional). Hard stop if Perplexity errors
exceed 5 questions (report partial, do not swap providers mid-run).

## Results

*(at completion)*
