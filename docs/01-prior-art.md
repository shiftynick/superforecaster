# Prior Art Dossier

Distilled from a deep-research pass (2026-07-09): 5 search angles → 23 sources fetched → 108 claims extracted → top 25 adversarially verified (3 skeptic votes each) → **18 confirmed, 7 refuted** → 11 synthesized findings. Confidence labels below reflect that verification. Items marked **(unverified lead)** were extracted but never made the verification cut — confirm before relying on them.

## Executive summary

The evidence supports a specific, buildable design. The most robust human-craft drivers are **frequent incremental updating**, **fine-grained probabilities**, and **selecting/weighting top performers** — not the training/teaming interventions pop accounts emphasize (those effect sizes are actively disputed). For aggregation, **log-odds pooling beats probability averaging**, but extremization is *conditional* — correlated ensembles (like LLMs sharing training data) warrant mild or no extremization. The best current AI systems are retrieval-augmented scaffolds; the 2025 state of the art (AIA Forecaster) adds a **supervisor agent that reconciles disparate forecasts** plus **statistical calibration**, and wins by *ensembling with market consensus* rather than beating it alone. As of Q2 2025, Metaculus Pro humans still beat all bots decisively on the live benchmark — the gap is real and is the target.

---

## 1. The human craft — what actually drives accuracy

| Finding | Confidence | Source |
|---|---|---|
| **Tracking** (selecting top 2% into elite teams) improved accuracy more than debiasing training or teaming, alone or combined | High (3-0, 2-1) | Mellers et al. 2014 ([Psych Science](https://journals.sagepub.com/doi/10.1177/0956797614524255)); Mellers et al. 2015 ([PoPS](https://stanford.edu/~knutson/nfc/mellers15.pdf)) |
| **Belief-updating frequency is the single strongest behavioral predictor of accuracy** — supers made 2.77 forecasts/question vs 1.47 (Year 1), rising to 5.64–6.70 | High (2-1) | Mellers et al. 2015 |
| **Granularity is real skill**: rounding supers to nearest 0.10 significantly worsened Brier; others unaffected until 0.33. Supers used ~57 distinct probability values | High (3-0) | Mellers et al. 2015 |
| **Skill persists** (no regression to mean): AUC 96% vs 84%/75%; supers at 300+ days out matched regulars at 60 days | High (3-0) | Mellers et al. 2015 |
| ⚠️ **The training/teaming effect sizes are disputed**: a 2024/25 IRT reanalysis found controlling for method variance (self-selecting easy questions, forecasting near resolution) eliminated, reduced, or reversed those effects. Live controversy — Mellers rebuts that question selection *is* skill | High (2-1, 3-0, 2-1) | Hauenstein et al. 2025 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39630638/)) |

**Refuted by verification** (don't build on these): the claim that training/teaming/tracking *each causally* improved accuracy (0-3), and the specific mechanization story for probability training (0-3) — both casualties of the IRT controversy.

**Scaffold implications:** re-forecast frequently as news arrives (don't one-shot); output 1%-granularity probabilities (never snap to round numbers); track per-configuration performance and weight the ensemble by it (mechanized "tracking"); treat "teach the model the checklist" as unproven rather than the core bet.

## 2. AI forecasting systems — what won and why

| Finding | Confidence | Source |
|---|---|---|
| **Halawi et al. 2024** (retrieval-augmented GPT-4 scaffold): Brier .179 vs crowd .149 — approaches but doesn't beat the crowd. **But**: on selectively-forecast questions (crowd uncertain 0.3–0.7, good retrieval) it *beats* the crowd (.238 vs .240), and system+crowd ensemble (.146) beats crowd alone | High (3-0, 3-0) | [arXiv:2402.18563](https://arxiv.org/abs/2402.18563) (NeurIPS 2024) |
| **12-LLM median ensemble = human crowd**: Brier 0.20 vs 0.19 (925 humans, live Metaculus tournament, statistically equivalent) | High (3-0) | Schoenegger et al. ([Science Advances](https://www.science.org/doi/10.1126/sciadv.adp1528)) |
| **AIA Forecaster (Bridgewater, Nov 2025)** — current SOTA architecture: (1) agentic news search, (2) **supervisor agent reconciling disparate forecasts**, (3) statistical calibration (e.g. Platt scaling to counter LLM hedging). Alone: Brier 0.1258 vs market 0.1106 (loses). Ensembled ~⅓ AIA / ⅔ market: 0.106 (**beats market** — additive information) | High (3-0, 3-0) | [arXiv:2511.07678](https://arxiv.org/abs/2511.07678) |
| ⚠️ **Live benchmark reality check**: Metaculus Pros beat bots in *every* quarter through Dec 2025; Q2 2025 bot-vs-pro head-to-head −20.03 (95% CI [−28.63, −11.41], p=1e-5) **(unverified lead)** | — | [metaculus.com/aib](https://www.metaculus.com/aib/), [LessWrong Q2 results](https://www.lesswrong.com/posts/Surnjh8A4WjgtQTkZ/q2-ai-benchmark-results-pros-maintain-clear-lead) |

**Refuted by verification:** Halawi's *specific* ablation magnitudes (retrieval −.027 Brier etc., 0-3); AIA's headline "parity with superforecasters on ForecastBench" (0-3); ForecastBench's specific human-vs-LLM figures (0-3). The architectures are confirmed; the precise numbers around them are not.

**Scaffold implications:** retrieval-augmentation is table stakes; **selective forecasting** (defer/abstain on low-signal questions) is a confirmed edge; ensemble diversity reaches crowd parity cheaply; the frontier move is a **reconciling supervisor + post-hoc calibration**; when a market price exists, model the *adjustment*, not the level.

## 3. Aggregation & calibration — the math layer

| Finding | Confidence | Source |
|---|---|---|
| **Log-odds (logit) aggregation with one extremization parameter beats arithmetic probability averaging** — validated on 1,300+ GJP forecasters, 69 events | High (3-0, 3-0) | Satopää et al. 2014 ([IJF](https://www.sciencedirect.com/science/article/abs/pii/S0169207013001635)) |
| **The average forecast is systematically underconfident** — the theoretical basis for extremizing | High (3-0) | [arXiv:1705.02391](https://arxiv.org/pdf/1705.02391); Satopää 2014 |
| **Extremization is conditional, not universal**: optimal Bayesian aggregators *anti-extremize* in 18–73% of cases; **shared information among forecasters reduces optimal extremization or flips it to attenuation**. Private info → extremize; shared info → don't | High (3-0, 3-0) | Lichtendahl et al. ([arXiv:1705.02391](https://arxiv.org/pdf/1705.02391), Oper. Res. 2022) |
| Neyman & Roughgarden 2021: principled extremization factor d → √3 ≈ 1.73 as n grows (d≈1.62 at n=10) **(unverified lead)** | — | via [EA Forum treatment](https://forum.effectivealtruism.org/posts/biL94PKfeHmgHY6qe/principled-extremizing-of-aggregated-forecasts) |

**Scaffold implications:** aggregate in log-odds space, never average probabilities. Critically for us: **an LLM ensemble is a shared-information ensemble** (common training data) — use mild or no extremization, tune the coefficient on held-out backtest data, and expect the optimum well below the classic human-crowd values. Add post-hoc calibration (Platt scaling) as a separate layer.

## 4. Question decomposition — the evidence gap

Notable negative result of the research pass: **no confirmed findings directly adjudicate recursive decomposition for forecast calibration**, in either human or LLM studies. The decomposition search angle surfaced sources (incl. a decomposition-skeptical Springer paper and [arXiv:2602.04853](https://arxiv.org/pdf/2602.04853)) but none of their claims survived to the confirmed set. The theoretical concern (conjunctive error compounding) stands, and the indirect evidence favors alternatives: what demonstrably works is diversity + aggregation (§3) and supervisor reconciliation (§2), not deep trees.

**Implication for hypothesis C:** the decomposition graph is genuinely open — neither validated nor killed by the literature. It stays in the race, but the burden of proof is on it, and our backtest will be among the first direct tests. Design guards (depth caps, per-node outside-view anchors, holistic sanity check) remain mandatory.

## 5. Evaluation infrastructure — entry points (mostly unverified leads)

The verification pass was hardest on this area — treat everything here as a lead to confirm hands-on, not settled fact:

- **[metac-bot-template](https://github.com/Metaculus/metac-bot-template)** — official Metaculus starter bot; two paths: `main.py` on the `forecasting-tools` package, or `main_with_no_framework.py` (minimal deps); auth via `METACULUS_TOKEN`. Fastest path to live scoring.
- **[Metaculus AI Benchmark](https://www.metaculus.com/aib/)** — the live tournament; quarterly seasons; Pros-vs-bots comparison published.
- **[Autocast](https://arxiv.org/abs/2206.15474)** — thousands of tournament questions + a *date-organized news corpus* enabling leakage-free information-at-time simulation.
- **Frozen context snapshots** ([OpenReview, ICLR 2026 sub](https://openreview.net/forum?id=Q5o249Z3Je)) — forward-only backtesting: continuously scrape *unresolved* Kalshi questions and store contemporaneous search-result snapshots; forecast later against the frozen snapshot. Elegant leakage-proof design worth copying.
- **Known leakage failure modes** ([arXiv:2507.19477](https://arxiv.org/pdf/2507.19477)): (1) evaluating on events resolved before the model's cutoff, (2) retrieval mixing post-resolution documents into results. Both must be explicitly guarded in our harness.
- ⚠️ **Refuted**: ForecastBench's "leakage-hardened by construction" claim and its specific superforecaster-vs-LLM Brier figures (0-3 each). Re-derive from the primary paper before using ForecastBench numbers.

## 6. Field survey — what AIB competitors are building (added 2026-07-09)

Source: Metaculus's ["What are other bots doing?"](https://www.metaculus.com/notebooks/43497/bot-plans/) notebook — self-descriptions from ~40 credit-applicant bots across Fall 2025 / Spring 2026 / Summer 2026. Self-reported plans, not verified results; several change plans mid-season. Read as a map of the competitive landscape, not as evidence of what works.

**The commodity recipe.** The field has converged, and says so itself (aroswift: "Most bots this season and last seem to follow the same basic recipe"): retrieval (AskNews + Perplexity + Exa) → multiple forecasts diversified by model/temperature/prompt → median or log-odds aggregation → mild extremization → iterate on MiniBench's 2-week cycles. That recipe is our **candidate B**. Multi-agent debate/panel variants (a dozen bots) and supervisor-synthesizer patterns (**candidate E**) are also well-populated. Recursive decomposition (**candidate C**) exists but is rare: r2d2 (recursive Fermi decomposition with tractability stopping) and Blueberry Pi's EvoGraph (persistent evidence graph that routes research toward weak nodes) are the two serious implementations. Market anchoring (**candidate D**) appears mostly as one input among many (signal2noise, MASX, tejvn, WobblyBot), not as a dedicated anchor-and-adjust architecture.

**Where we are genuinely differentiated** (nothing in the survey does these):

1. **The architecture-level portfolio.** Everyone ensembles models/prompts *within* one architecture; nobody described ensembling *across architectures* with learned per-category weights. Our portfolio decision remains the distinctive bet.
2. **Pre-registered evaluation discipline.** Several bots have feedback loops (Sloth's scoring ledger, VIGILJANUS's calibration reviews), but none described frozen-metric pre-registration or lessons-as-hypotheses-verified-on-backtest. Most iterate by MiniBench intuition — exactly the single-miss-patching failure mode our postmortem protocol guards against.
3. Honesty requires noting: "personal forecasting platform with a bot arm" is *not* unique (Sloth/SlothAI, VIGILJANUS/SYGNITO, dtfr's centaur OSINT demo), and neither is any individual candidate A–E.

**Ideas worth stealing** (grafts recorded in [03-architecture-candidates.md](03-architecture-candidates.md)):

| Idea | From | What it is |
|---|---|---|
| Agreement-scaled extremization | pancratic-bot | Extremize only when independent lenses converge; pull toward uncertainty when they diverge. Turns our fixed extremization coefficient into a function of ensemble agreement — directly testable in candidate B's math |
| Empirical per-bin recalibration + objectivity gate | aroswift, PressPause | Fit a recalibration map on own resolved forecasts (33% forecasts resolving true 20% of the time → shift that bin down), but first ask "is this probability mechanical?" (dice-odds questions must not be dragged toward historical base rates) |
| Crux-resolution stacking | tejvn-bot | When ensemble members diverge sharply, a meta-model extracts the *crux* of disagreement and runs targeted research to resolve it before synthesizing — a smarter supervisor for candidate E |
| Variance-based routing | amigon21/MiniSim | Measure ensemble variance first; low variance → one cheap call (deliberation adds noise), high variance → full expensive deliberation. Cost control that fits our config-over-code requirement |
| Code execution for quantitative grounding | tejvn, aroswift, smingers | Let the forecaster write/run Python for base rates, distribution fitting, simulation — LLMs reasoning about numbers in prose make systematic arithmetic errors |
| Diversity via reasoning frames, not temperature | MASX-AI, pancratic | Ensemble members anchored to *heterogeneous analytical lenses* (outside view / inside view / status-quo inertia; or doctrine-level: balance-of-power, prospect theory) rather than N identical prompts at different temperatures |
| Resolution-source grounding | GreeneiBot2 | Fetch/screenshot the question's actual resolution links so the forecast reasons against the source that will decide it |
| Status-quo lens + fragility test | pancratic, VIGILJANUS, RRRRRR-bot | Explicit "what happens if nothing changes" pass, plus "what would have to go wrong for this forecast to be wrong" — single-fragile-assumption forecasts get pulled toward uncertainty |
| MC option-order debiasing | aroswift, PressPause | Multiple-choice is where bots reportedly lose most vs pros; shuffle option orders across runs and average, normalize in code not in-model. Not relevant until we do MC questions — parked |

**Meta-lessons from the field** (consistent with our dossier): Metaculus's own finding that *SOTA model choice matters more than scaffolding* is repeated to applicants; CumulativeBot's writeup independently reports that a home-built deep-researcher was the source of "pretty much every hallucination" and agentic pipelines "expensive and underperformed" — supporting our choice of simple retrieval + strong models over elaborate agentic research, at least initially.

## Bottom line — evidence-ranked scaffold priorities

1. **Retrieval + news search** (confirmed core of every competitive system)
2. **Diverse ensemble → log-odds aggregation → mild/no extremization → Platt calibration** (each layer separately confirmed)
3. **Selective forecasting** — know when not to forecast (confirmed edge over crowd)
4. **Supervisor/reconciler agent** (SOTA differentiator, confirmed architecture)
5. **Market anchoring where prices exist** — be the ⅓ in a ⅓-model/⅔-market ensemble (confirmed additive)
6. **Frequent re-forecasting** on question lifecycle (strongest human predictor, mechanizes trivially)
7. Decomposition graph — open hypothesis, ours to test

## Open questions carried forward

1. Current (mid-2026) AIB tournament rules/entry mechanics and metac-bot template state — verify hands-on (it's a `git clone` away)
2. True apples-to-apples ForecastBench standings after the refutations
3. AIA Forecaster reproducibility outside its ~322-question liquid-market window
4. Optimal extremization coefficient for a *correlated* LLM ensemble — tune empirically on our backtest
5. Which leakage-hardening techniques are validated — Autocast + frozen-snapshots are the leads
