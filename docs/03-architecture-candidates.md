# Architecture Candidates

The designs to race, each a hypothesis to be adjudicated by the evaluation harness (docs/02). All are scaffolds around frontier models sharing common infrastructure: question ingestion, web retrieval with date filtering, and probability extraction. Updated 2026-07-09 from the prior-art dossier ([01-prior-art.md](01-prior-art.md)).

**Cross-cutting components** (evidence-confirmed, apply to every candidate):

- **Selective forecasting**: Halawi et al.'s system *beats* the crowd when restricted to questions with crowd uncertainty 0.3–0.7 and good retrieval — every candidate should emit a confidence-in-forecast signal and the system should defer/abstain on low-signal questions.
- **Fine-grained output**: 1%-granularity probabilities; never snap to round numbers (rounding supers to 0.10 measurably hurt their Brier).
- **Frequent re-forecasting**: update on question lifecycle events (new news, market moves) — the strongest human behavioral predictor, trivially mechanized.
- **Post-hoc calibration**: Platt scaling as a final layer to counter LLM hedging bias (AIA Forecaster's confirmed technique).

## A — Checklist single-pass (baseline)

One strong pass: retrieve relevant news/data, then a prompt that walks the superforecaster checklist — reference class → base rate anchor → inside-view adjustments → premortem → final probability. This is roughly the Halawi et al. 2024 recipe minus ensembling; it must exist first because every other candidate is measured against it.

**Prior:** already near human-crowd level in published work. Cheapest to run.

## B — Diverse ensemble + aggregation

N independent passes engineered for diversity (different personas, different retrieval seeds, different reasoning orders: outside-view-first vs. inside-view-first), aggregated **in log-odds space** (Satopää et al. 2014 — confirmed superior to probability averaging), then *mildly* extremized. Key dossier correction: extremization is conditional on information independence, and an LLM ensemble is a **shared-information** ensemble (common training data) — expect the optimal coefficient well below the classic human-crowd √3, possibly ≤1 (attenuation). Tune on a held-out backtest split; weight members by tracked performance (mechanized "tracking", the best-evidenced human intervention).

**Prior:** a 12-LLM median ensemble already matches a 925-person human crowd (Brier 0.20 vs 0.19, Science Advances). The "competence weighting" intuition lives here, applied across forecasts rather than facts.

## C — Recursive decomposition graph (the original idea)

In the user's words: a hierarchical graph that breaks the primary question into the questions that must be answered first, then sub-questions of those, and so on down to a **bedrock level** of researchable facts and base rates. From the bottom of that pyramid, confidence and prediction scores at the lower levels **bubble up** to produce the final probability. Leaves carry per-node confidence; probabilities combine upward with explicit conjunction/disjunction structure.

**Prior:** genuinely open — the dossier's dedicated search angle found **no confirmed evidence either way** on recursive decomposition for forecast calibration; our backtest will be among the first direct tests. The theoretical risk (conjunctive error compounding) stands, and the confirmed winners (diversity+aggregation, supervisor reconciliation) are alternatives, not decomposition. It's the only candidate producing a *legible argument* for the number. Even if it loses on Brier, evaluate it as (a) an explainability layer over B/E's number, and (b) a generator of diverse passes to feed B.

**Design guards:** cap depth; require every node to also carry an outside-view anchor; never multiply more than ~3 raw conjuncts without a sanity-check node comparing against a holistic estimate.

## D — Market/community-anchored

Where a community or market price exists, start there as the prior and model only the *adjustment*: is there information the market hasn't priced in (recency, niche data), or a known market bias (longshot bias, low liquidity)? Output = anchor ± modeled edge.

**Prior:** strongly confirmed as a *component*: AIA Forecaster alone loses to market consensus (Brier 0.1258 vs 0.1106) but a ~⅓-model/⅔-market ensemble beats the market (0.106). Halawi found the same shape (system+crowd beats crowd). Useless on questions with no market.

**Framing (per decision review):** the market price is **another data point — potentially heavily weighted, never the sole reliance**. The weight is learned on backtest, not assumed; the system must work (and be measured) on questions with no market at all.

## E — Supervisor-reconciled agentic scaffold (AIA-style) ★ new from dossier

The confirmed 2025 SOTA architecture (AIA Forecaster, [arXiv:2511.07678](https://arxiv.org/abs/2511.07678)): (1) agentic search over high-quality news sources, (2) a **supervisor agent that reconciles disparate forecasts** for the same event — an *intelligent* aggregator replacing B's fixed formula, and (3) statistical calibration countering LLM biases. Effectively B with a reasoning layer where the math layer was.

**Prior:** the strongest published system. More expensive per question than B; race it *after* B establishes whether reconciliation beats formula aggregation at our scale.

## Portfolio decision (2026-07-09)

**We pursue all candidates and ensemble across them.** Rather than racing to crown one survivor, every candidate implements the common engine interface so its forecast (probability + confidence + trace) can join an **architecture-level pool** — the same log-odds machinery, one level up. The harness's job shifts from "pick the winner" to "learn each candidate's weight" (possibly per question category — the pyramid may earn weight on decomposable questions, market-anchoring on liquid ones). A candidate "losing" means its weight goes to ~zero, not that it gets deleted; it can still serve as explanation layer or diversity source. Build order below still matters for cost and baseline reasons, but the end state is an ensemble of architectures.

## Grafts from the field survey (2026-07-09)

Competitor ideas worth testing, from the AIB bot-plans survey ([01-prior-art.md §6](01-prior-art.md)). Each is a **hypothesis** — it enters a candidate only via a pre-registered backtest experiment (docs/02), never as a straight patch:

- **B:** agreement-scaled extremization (extremize on lens convergence, attenuate on divergence — replaces the fixed coefficient); diversity via heterogeneous reasoning frames (outside-view / inside-view / status-quo-inertia lenses, domain doctrines) rather than temperature; variance-based routing (cheap single call when passes agree, full ensemble only on disagreement) as the cost lever.
- **E:** crux-resolution stacking — on sharp divergence the supervisor extracts the crux of disagreement and commissions targeted research before reconciling.
- **Calibration layer (all candidates):** empirical per-bin recalibration fit on our own resolved forecasts, gated by an objectivity check (mechanical-probability questions exempt).
- **Cross-cutting retrieval/reasoning:** resolution-source grounding (fetch the links the question will actually resolve by); code execution for quantitative steps (base rates, distribution fits, simulations); fragility test ("what must go wrong for this forecast to be wrong" — single-fragile-assumption forecasts pulled toward uncertainty).
- **Parked until MC questions:** option-order shuffling + in-code normalization (the field says multiple-choice is where bots bleed points vs pros).

## Cross-cutting: refinement modules (owner decision, 2026-07-12)

A **refiner** is an optional post-pass that any candidate's forecast can flow through:
input = the question, its research bundle, and the candidate's forecast (probability +
confidence + rationale); output = an adjusted forecast plus the refiner's own reasoning,
recorded in the trace. Architecturally it is the reasoning sibling of the extremization
step (a math post-pass on the pool): same slot in the pipeline, one level richer. Refiners
compose with every candidate A–E, are toggled per task in config (config-over-code), and
are evaluated the only way a post-pass honestly can be — a pre-registered paired
experiment of *arm ± refiner* on identical base forecasts.

- **Game-theoretic trajectory refiner** (owner, 2026-07-09; shaped 2026-07-12): the first
  planned refiner. From the current data/state, play out the interval between forecast
  date and resolution date as a strategic interaction — who are the decision-makers, what
  are their moves and incentives, what path is stable — and adjust the base forecast
  accordingly. Plausibly strongest on actor-driven questions (geopolitics, negotiations,
  elections, corporate moves), so question-type routing may gate it. Prior art to check
  when building: MASX-AI's doctrinal council uses game-theoretic frames as ensemble
  lenses; Spagbot mixes "game theoretical approaches" into sociopolitical questions
  ([01 §6](01-prior-art.md)) — their season results may yield evidence first. Full
  machinery (payoff matrices, equilibrium solving) stays out until the cheap LLM-reasoned
  version shows signal. Like every idea: enters the system only through a pre-registered
  experiment.

Candidate C's node schema must stay compatible: nodes carry a `reasoning_type`, so a
proven refiner can also serve as a *node-level* reasoning mode inside the pyramid where a
sub-question is actor-driven.

## Race protocol

- Common question set and retrieval budget (per docs/02)
- Primary metric: paired Brier delta vs. baseline A
- Secondary: cost/question, calibration curve, legibility of reasoning
- Race order: A first (it's the baseline and the live-tournament entry), then B, then E (only if B's formula aggregation looks like the bottleneck), C and D in parallel with B
- Predicted finish order from the dossier (to be checked against reality): **E > B ≈ D > A > C** on Brier (D only where markets exist); **C** first on legibility

## Results

_Empty until the harness runs._
