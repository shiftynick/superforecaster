# NNN — Title (what is being compared)

- Status: draft | frozen | complete | aborted
- Registered: YYYY-MM-DD
- Frozen: YYYY-MM-DD (commit hash of the freeze)
- Owner sign-off: (who/when, or "checkpoint review")

## Hypothesis

One falsifiable sentence. Example: "The candidate-B ensemble (5 members, agreement-scaled
extremization) achieves a lower mean Brier score than candidate A (single GLM-5.2 pass) on
the same questions."

## Arms

| Arm | Config (exact) |
|---|---|
| A | provider/model/effort/max_tokens; settings diff or snapshot reference |
| B | member roster, extremization mode + coefficients, pooling |

Everything an arm needs to be re-run exactly, including the research/retrieval settings
shared by both arms.

## Question set

- Selection rule (frozen before looking at any outcomes): e.g. "all binary questions in the
  snapshot store that resolve between DATE1 and DATE2, N ≥ …".
- Leakage tier (docs/02): frozen-context forward / post-cutoff-resolved / live.
- Leakage-risk label and why.
- Once selected: the exact question ids (append them at freeze time).

## Metric & decision rule

- Primary: (e.g. mean paired Brier delta, B − A).
- Secondaries: (log score, ECE, cost/question — diagnostic only, cannot flip the decision).
- Decision rule, pre-committed: (e.g. "adopt B as the default candidate if delta < −0.01
  and the 95% bootstrap CI on the paired delta excludes 0; otherwise keep A and log").

## Power check

Expected effect size, N available, and whether N suffices. If underpowered, state the
downgraded conclusion language now ("directional evidence only, no config change").

## Budget

Max LLM spend $X; max AskNews credits Y.

## Results (filled at completion)

- Scores per arm, delta + CI, secondaries.
- Decision taken (per the rule, verbatim).
- Surprises / anomalies / postmortem links.
