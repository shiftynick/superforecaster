"""Render a stored forecast into a self-contained HTML argument view.

This is the interim argument UX until the web app exists (spec §6): a
deterministic template — no LLM call, no server, no external assets — that turns
one :class:`~superforecaster.forecaster.ForecastRecord` into a page a
non-statistician can read.

- A ``pyramid-c`` record renders in full: the decomposition tree as nested cards
  (each node's question, estimate, anchor, grounding, rationale, and the
  combination math), a header showing structural / holistic / blended side by
  side, the confidence and its two raw signals, the prune notes, and the cost.
- Any other method (``checklist-a``, ``ensemble-b`` and its members) degrades
  gracefully — the page shows whatever the record carries (probability,
  confidence, rationale, and the ensemble members when present).

The visual style is a restrained operational tool: embedded CSS, no gradients,
plain labels. Its template is intended to become the web app's first component.
"""

from __future__ import annotations

from html import escape

from .forecaster import ForecastRecord, PyramidDetail
from .pyramid import CombinationStep, PyramidNode

_CSS = """\
:root {
  --ink: #1c1c1c; --muted: #5a5a5a; --line: #d9d9d9; --bg: #ffffff;
  --card: #fafafa; --accent: #2b5b84; --warn: #8a4b08; --evidence: #1f6f43;
  --model: #6a3fa0;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  color: var(--ink); background: var(--bg); margin: 0; padding: 2rem;
  line-height: 1.5;
}
main { max-width: 900px; margin: 0 auto; }
h1 { font-size: 1.4rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.05rem; margin: 1.75rem 0 0.6rem; border-bottom: 1px solid var(--line);
     padding-bottom: 0.25rem; }
.sub { color: var(--muted); font-size: 0.9rem; margin: 0 0 1rem; }
.headline { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }
.stat { border: 1px solid var(--line); border-radius: 6px; padding: 0.6rem 0.9rem;
        background: var(--card); min-width: 8rem; }
.stat .label { color: var(--muted); font-size: 0.75rem; text-transform: uppercase;
               letter-spacing: 0.03em; }
.stat .value { font-size: 1.3rem; font-weight: 600; }
.badge { display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0.1rem 0.45rem;
         border-radius: 4px; border: 1px solid var(--line); text-transform: uppercase;
         letter-spacing: 0.03em; }
.badge.evidence { color: var(--evidence); border-color: var(--evidence); }
.badge.model-knowledge { color: var(--model); border-color: var(--model); }
.badge.op { color: var(--accent); border-color: var(--accent); }
.badge.failed { color: #b00020; border-color: #b00020; }
.badge.verified-true { color: var(--evidence); border-color: var(--evidence); }
.badge.verified-false { color: #b00020; border-color: #b00020; }
.badge.unverified { color: var(--muted); border-color: var(--muted); }
.badge.fragile { color: var(--warn); border-color: var(--warn); }
.assumptions { list-style: none; margin: 0.35rem 0 0; padding: 0; }
.assumptions li { font-size: 0.85rem; margin: 0.2rem 0; color: #333; }
.assumptions .ev { color: var(--muted); }
.node { border: 1px solid var(--line); border-left: 3px solid var(--accent);
        border-radius: 6px; padding: 0.75rem 0.9rem; margin: 0.6rem 0; background: var(--card); }
.node.leaf { border-left-color: var(--evidence); }
.node .q { font-weight: 600; margin: 0 0 0.35rem; }
.node .meta { color: var(--muted); font-size: 0.85rem; margin: 0.2rem 0; }
.node .rationale { font-size: 0.9rem; white-space: pre-wrap; margin: 0.4rem 0 0;
                   color: #333; }
.children { margin-left: 1.1rem; margin-top: 0.5rem; }
.math { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.85rem;
        background: #f0f0f0; border-radius: 4px; padding: 0.35rem 0.5rem; margin: 0.4rem 0 0; }
.note { color: var(--warn); font-size: 0.88rem; margin: 0.25rem 0; }
ul.notes { margin: 0.3rem 0; padding-left: 1.2rem; }
.member { border: 1px solid var(--line); border-radius: 6px; padding: 0.5rem 0.75rem;
          margin: 0.4rem 0; background: var(--card); font-size: 0.9rem; }
.prob { font-variant-numeric: tabular-nums; }
footer { color: var(--muted); font-size: 0.8rem; margin-top: 2rem;
         border-top: 1px solid var(--line); padding-top: 0.6rem; }
"""


def render_forecast_html(record: ForecastRecord) -> str:
    """Return a complete, self-contained HTML document for one forecast record.

    Dispatches on ``record.method``: a full pyramid view when the record carries a
    :class:`PyramidDetail`, otherwise the graceful-degrade view.
    """
    title = f"Forecast — question {record.question_id}"
    body = _pyramid_body(record) if record.pyramid is not None else _generic_body(record)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f"<main>\n{body}\n{_footer(record)}\n</main>\n</body>\n</html>\n"
    )


def _header(record: ForecastRecord, method_label: str) -> str:
    return (
        f"<h1>{escape(record.title)}</h1>\n"
        f'<p class="sub">Question {record.question_id} · post {record.post_id} · '
        f"{method_label} · {escape(record.created_at.isoformat())}</p>"
    )


def _stat(label: str, value: str) -> str:
    return (
        f'<div class="stat"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div></div>'
    )


def _pct(probability: float) -> str:
    return f"{probability * 100:.0f}%"


def _cost_str(record: ForecastRecord) -> str:
    return "unknown" if record.cost_usd is None else f"${record.cost_usd:.4f}"


def _pyramid_body(record: ForecastRecord) -> str:
    detail = record.pyramid
    assert detail is not None  # guarded by the caller

    parts = [_header(record, "decomposition pyramid (candidate C)")]

    parts.append('<div class="headline">')
    parts.append(_stat("Structural", _pct(detail.structural_probability)))
    parts.append(_stat("Holistic", _pct(detail.holistic_probability)))
    parts.append(_stat("Blended (final)", _pct(record.probability)))
    parts.append(_stat("Confidence", record.confidence))
    parts.append("</div>")

    parts.append("<h2>How the final number was formed</h2>")
    parts.append(
        f'<p class="meta">The <strong>structural</strong> estimate '
        f"({_pct(detail.structural_probability)}) comes from combining the leaf "
        f"estimates up the tree. The <strong>holistic</strong> estimate "
        f"({_pct(detail.holistic_probability)}) is one direct forecast of the whole "
        f"question. The final <strong>blended</strong> number ({_pct(record.probability)}) "
        f"is a log-odds blend of the two, weighting the holistic estimate "
        f"{detail.holistic_blend_weight:.2f}.</p>"
    )

    parts.append("<h2>Confidence signals</h2>")
    parts.append(
        f'<p class="meta">Reported confidence <strong>{escape(record.confidence)}</strong> is the '
        "more cautious of three raw signals (all provisional):</p><ul>"
        f"<li>Weakest leaf confidence on a multiplied path: "
        f"<strong>{escape(detail.weakest_leaf_confidence)}</strong></li>"
        f"<li>Structural-vs-holistic disagreement (log-odds gap): "
        f"<strong>{detail.structural_holistic_gap:.2f}</strong> "
        "(smaller = the two agree more)</li>"
        f"<li>Unverified load-bearing assumptions: "
        f"<strong>{detail.unverified_load_bearing_count}</strong> "
        "(more = confidence capped lower)</li></ul>"
    )

    parts.append("<h2>Verification &amp; retrieval budget</h2>")
    parts.append(
        f'<p class="meta">Assumption verifications run: '
        f"<strong>{detail.assumption_verifications_run}</strong> "
        f"(search credits {detail.assumption_search_credits}). "
        f"Leaf top-up searches: <strong>{detail.leaf_topup_searches_run}</strong> "
        f"(search credits {detail.leaf_topup_search_credits}).</p>"
    )

    if detail.decompose_failed:
        parts.append(
            '<p class="note">Decomposition failed — this ran as a single careful '
            "leaf rather than a full tree.</p>"
        )

    parts.append("<h2>The argument tree</h2>")
    steps = {step.node_id: step for step in detail.combination_steps}
    parts.append(_render_node(detail.tree, detail, steps))

    if detail.prune_notes:
        parts.append("<h2>Structural-cap prune notes</h2><ul class='notes'>")
        for note in detail.prune_notes:
            parts.append(f'<li class="note">[{escape(note.cap)}] {escape(note.message)}</li>')
        parts.append("</ul>")

    if detail.notes:
        parts.append("<h2>Trace notes</h2><ul class='notes'>")
        for trace_note in detail.notes:
            parts.append(f'<li class="note">{escape(trace_note)}</li>')
        parts.append("</ul>")

    return "\n".join(parts)


def _render_node(
    node: PyramidNode,
    detail: PyramidDetail,
    steps: dict[str, CombinationStep],
) -> str:
    """Recursively render one tree node (and its children) as nested cards."""
    is_leaf = node.node_type == "leaf"
    cls = "node leaf" if is_leaf else "node"
    parts = [f'<div class="{cls}">']
    parts.append(f'<p class="q">{escape(node.question)}</p>')

    if is_leaf:
        estimate = next((e for e in detail.leaf_estimates if e.node_id == node.id), None)
        grounding = node.grounding or (estimate.grounding if estimate else "evidence")
        parts.append(
            f'<p class="meta"><span class="badge {escape(grounding)}">{escape(grounding)}</span> '
        )
        if estimate is not None:
            failed = ' <span class="badge failed">failed</span>' if estimate.failed else ""
            fragile = (
                ' <span class="badge fragile">fragility pull</span>'
                if estimate.fragility_pull_applied
                else ""
            )
            parts.append(
                f'<span class="prob"><strong>{_pct(estimate.probability)}</strong></span> · '
                f"confidence {escape(estimate.confidence)}{failed}{fragile}</p>"
            )
            if estimate.fragility_pull_applied and estimate.pre_pull_probability is not None:
                parts.append(
                    f'<p class="meta">Fragility pull (unverified load-bearing assumption): '
                    f"{_pct(estimate.pre_pull_probability)} → {_pct(estimate.probability)} "
                    "toward 0.5.</p>"
                )
            if estimate.topup_search_used:
                parts.append(
                    '<p class="meta">Leaf top-up: one extra targeted search was appended '
                    "(zero bundle coverage).</p>"
                )
            if estimate.reference_class:
                parts.append(
                    f'<p class="meta">Anchor / reference class: '
                    f"{escape(estimate.reference_class)}</p>"
                )
            if estimate.rationale:
                parts.append(f'<p class="rationale">{escape(estimate.rationale)}</p>')
        else:
            parts.append("(no estimate recorded)</p>")
    else:
        parts.append(f'<p class="meta"><span class="badge op">{escape(node.node_type)}</span></p>')

    parts.append(_render_assumptions(node))

    step = steps.get(node.id)
    if step is not None and not is_leaf:
        clipped = ", ".join(f"{v:.3f}" for v in step.clipped_inputs)
        raw = ", ".join(f"{v:.3f}" for v in step.inputs)
        parts.append(
            f'<p class="math">{escape(node.node_type)}: '
            f"inputs [{raw}] → clipped [{clipped}] → <strong>{step.output:.3f}</strong></p>"
        )

    if node.children:
        parts.append('<div class="children">')
        for child in node.children:
            parts.append(_render_node(child, detail, steps))
        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


_ASSUMPTION_GLYPH = {"verified-true": "✓", "verified-false": "✗", "unverified": "○"}


def _render_assumptions(node: PyramidNode) -> str:
    """Render a node's assumptions as status badges with a one-line evidence note.

    ✓ green (verified-true), ✗ red (verified-false), ○ gray (unverified) —
    the "which assumption broke?" postmortem view (spec §5). Returns an empty
    string when the node surfaced no assumptions.
    """
    if not node.assumptions:
        return ""
    items = ['<ul class="assumptions">']
    for assumption in node.assumptions:
        status = assumption.status
        glyph = _ASSUMPTION_GLYPH.get(status, "○")
        lb = " · load-bearing" if assumption.load_bearing else ""
        evidence = (
            f' — <span class="ev">{escape(assumption.evidence)}</span>'
            if assumption.evidence
            else ""
        )
        items.append(
            f'<li><span class="badge {escape(status)}">{glyph} {escape(status)}</span> '
            f"{escape(assumption.text)}{lb}{evidence}</li>"
        )
    items.append("</ul>")
    return "\n".join(items)


def _generic_body(record: ForecastRecord) -> str:
    """The graceful-degrade view for non-pyramid records."""
    parts = [_header(record, f"method: {record.method}")]

    parts.append('<div class="headline">')
    parts.append(_stat("Probability", _pct(record.probability)))
    parts.append(_stat("Confidence", record.confidence))
    parts.append(_stat("Cost", _cost_str(record)))
    parts.append("</div>")

    if record.reference_class:
        parts.append(f'<p class="meta">Reference class: {escape(record.reference_class)}</p>')

    if record.rationale:
        parts.append("<h2>Rationale</h2>")
        parts.append(f'<p class="rationale">{escape(record.rationale)}</p>')

    if record.ensemble is not None:
        ens = record.ensemble
        parts.append("<h2>Ensemble members</h2>")
        parts.append(
            f'<p class="meta">Pooled over {len(ens.member_probabilities)} members; '
            f"spread {ens.spread:.3f}; coefficient {ens.coefficient_applied:.3f} "
            f"({escape(ens.extremization_mode)}).</p>"
        )
        for frame, model, prob in zip(
            ens.member_frames, ens.member_models, ens.member_probabilities, strict=False
        ):
            label = model or "(default model)"
            parts.append(
                f'<div class="member"><strong>{escape(frame)}</strong> / {escape(label)} → '
                f'<span class="prob">{_pct(prob)}</span></div>'
            )
        for failure in ens.member_failures:
            parts.append(f'<p class="note">Skipped: {escape(failure)}</p>')

    return "\n".join(parts)


def _footer(record: ForecastRecord) -> str:
    dry = "dry run" if record.dry_run else "submitted"
    return (
        f"<footer>Model {escape(record.model)} · {escape(dry)} · "
        f"cost {escape(_cost_str(record))} · "
        f"{record.input_tokens + record.output_tokens} tokens · "
        f"search credits {record.research_search_credits}</footer>"
    )
