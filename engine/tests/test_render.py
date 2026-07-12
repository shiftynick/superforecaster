"""Tests for the self-contained HTML forecast renderer (golden substring checks).

Deterministic template, no LLM. We assert that the key content appears in the
output for both a pyramid-c record (tree, structural/holistic/blended, grounding,
combination math, prune notes) and a plain checklist-a record (graceful degrade),
rather than pinning a full-file golden that would be brittle to style tweaks.
"""

from __future__ import annotations

from datetime import UTC, datetime

from superforecaster.forecaster import ForecastRecord, LeafEstimate, PyramidDetail
from superforecaster.pyramid import CombinationStep, PruneNote, PyramidNode
from superforecaster.render import render_forecast_html

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)


def _pyramid_record() -> ForecastRecord:
    tree = PyramidNode(
        id="root",
        question="Will the whole thing happen?",
        node_type="conjunction",
        children=[
            PyramidNode(
                id="a",
                question="Will step A happen?",
                node_type="leaf",
                grounding="evidence",
            ),
            PyramidNode(
                id="b",
                question="Is the base rate high?",
                node_type="leaf",
                grounding="model-knowledge",
            ),
        ],
    )
    detail = PyramidDetail(
        tree=tree,
        prune_notes=[
            PruneNote(node_id="root", cap="children", message="dropped 1 child beyond the cap")
        ],
        leaf_estimates=[
            LeafEstimate(
                node_id="a",
                question="Will step A happen?",
                grounding="evidence",
                probability=0.5,
                confidence="high",
                rationale="Recent filings suggest progress.",
                reference_class="comparable mergers",
            ),
            LeafEstimate(
                node_id="b",
                question="Is the base rate high?",
                grounding="model-knowledge",
                probability=0.4,
                confidence="medium",
                rationale="Historically about 40%.",
                reference_class="deal base rate",
            ),
        ],
        combination_steps=[
            CombinationStep(
                node_id="root",
                operation="conjunction",
                inputs=[0.5, 0.4],
                clipped_inputs=[0.5, 0.4],
                output=0.2,
            ),
            CombinationStep(
                node_id="a", operation="leaf", inputs=[0.5], clipped_inputs=[0.5], output=0.5
            ),
            CombinationStep(
                node_id="b", operation="leaf", inputs=[0.4], clipped_inputs=[0.4], output=0.4
            ),
        ],
        structural_probability=0.2,
        holistic_probability=0.3,
        holistic_blend_weight=0.5,
        weakest_leaf_confidence="medium",
        structural_holistic_gap=0.47,
        decompose_failed=False,
        notes=["holistic pass failed; blended structural against itself"],
    )
    return ForecastRecord(
        created_at=NOW,
        question_id=301,
        post_id=201,
        title="Will the merger close by year end?",
        probability=0.24,
        raw_probability=0.24,
        confidence="medium",
        rationale="Decomposition pyramid summary.",
        reference_class="comparable mergers; deal base rate",
        model="pyramid-c",
        input_tokens=400,
        output_tokens=200,
        cost_usd=0.04,
        method="pyramid-c",
        pyramid=detail,
    )


def _checklist_record() -> ForecastRecord:
    return ForecastRecord(
        created_at=NOW,
        question_id=999,
        post_id=888,
        title="Will it rain tomorrow?",
        probability=0.42,
        raw_probability=0.42,
        confidence="low",
        rationale="Reference class of rainy days, then adjust.",
        reference_class="rainy days in July",
        model="z-ai/glm-5.2",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.01,
        method="checklist-a",
    )


def test_render_pyramid_record_contains_key_content() -> None:
    html = render_forecast_html(_pyramid_record())

    assert html.startswith("<!doctype html>")
    assert "Will the merger close by year end?" in html
    # Header shows structural / holistic / blended side by side (as percentages).
    assert "Structural" in html and "20%" in html
    assert "Holistic" in html and "30%" in html
    assert "Blended (final)" in html and "24%" in html
    # Tree content: both leaf questions, grounding badges, an anchor, a rationale.
    assert "Will step A happen?" in html
    assert "Is the base rate high?" in html
    assert "evidence" in html and "model-knowledge" in html
    assert "comparable mergers" in html
    assert "Recent filings suggest progress." in html
    # Combination math for the internal node, including the clipped inputs.
    assert "conjunction:" in html
    assert "0.500" in html and "0.400" in html and "0.200" in html
    # Confidence raw signals and the prune note both surface.
    assert "Weakest leaf confidence" in html
    assert "0.47" in html
    assert "dropped 1 child beyond the cap" in html
    # Self-contained: no external asset references.
    assert "http://" not in html
    assert "src=" not in html and "<link" not in html


def test_render_checklist_record_degrades_gracefully() -> None:
    html = render_forecast_html(_checklist_record())

    assert html.startswith("<!doctype html>")
    assert "Will it rain tomorrow?" in html
    assert "42%" in html  # probability
    assert "low" in html  # confidence
    assert "Reference class of rainy days, then adjust." in html
    assert "rainy days in July" in html
    # No pyramid-only scaffolding for a non-pyramid record.
    assert "The argument tree" not in html
    assert "Structural" not in html


def test_render_escapes_html_in_content() -> None:
    record = _checklist_record()
    record = record.model_copy(update={"rationale": "1 < 2 & <script>alert('x')</script>"})
    html = render_forecast_html(record)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
