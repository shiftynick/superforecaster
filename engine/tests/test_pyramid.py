"""Tests for the decomposition pyramid: parser, cap enforcement, combination math."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from superforecaster.config import PyramidSettings, load_pyramid_settings
from superforecaster.pyramid import (
    Assumption,
    PyramidNode,
    PyramidParseError,
    blend,
    combine_conditional,
    combine_conjunction,
    combine_disjunction,
    combine_tree,
    decomposition_json_schema,
    parse_decomposition,
)

_SETTINGS = PyramidSettings()  # defaults: depth 2, children 4, conjuncts 3, clip 0.03/0.97

# Probabilities safely inside the default clip band, so clipping is the identity
# and the pure monotonicity properties hold against the raw inputs.
in_band = st.floats(min_value=0.05, max_value=0.95)


def _leaf(node_id: str, grounding: str = "evidence") -> dict[str, object]:
    return {"id": node_id, "question": f"q-{node_id}", "node_type": "leaf", "grounding": grounding}


# --- Parser: valid trees ------------------------------------------------------


def test_parse_single_leaf_tree() -> None:
    tree, notes = parse_decomposition(_leaf("root", "model-knowledge"), _SETTINGS)
    assert notes == []
    assert tree.node_type == "leaf"
    assert tree.grounding == "model-knowledge"
    assert tree.reasoning_type == "checklist"  # default applied
    assert tree.children == []


def test_parse_valid_two_level_tree() -> None:
    raw = {
        "id": "root",
        "question": "Will X happen?",
        "node_type": "conjunction",
        "children": [_leaf("a"), _leaf("b", "model-knowledge")],
    }
    tree, notes = parse_decomposition(raw, _SETTINGS)
    assert notes == []
    assert tree.node_type == "conjunction"
    assert tree.grounding is None  # internal nodes carry no grounding
    assert [c.id for c in tree.children] == ["a", "b"]
    assert tree.children[0].grounding == "evidence"


# --- Parser: malformed input --------------------------------------------------


def test_parse_leaf_missing_grounding_rejected() -> None:
    bad_leaf = {"id": "root", "question": "q", "node_type": "leaf"}
    with pytest.raises(PyramidParseError, match="grounding"):
        parse_decomposition(bad_leaf, _SETTINGS)


def test_parse_leaf_missing_grounding_nested_rejected() -> None:
    raw = {
        "id": "root",
        "question": "q",
        "node_type": "disjunction",
        "children": [_leaf("a"), {"id": "b", "question": "q", "node_type": "leaf"}],
    }
    with pytest.raises(PyramidParseError, match="grounding"):
        parse_decomposition(raw, _SETTINGS)


def test_parse_internal_node_without_children_rejected() -> None:
    raw = {"id": "root", "question": "q", "node_type": "conjunction", "children": []}
    with pytest.raises(PyramidParseError, match="no children"):
        parse_decomposition(raw, _SETTINGS)


def test_parse_leaf_with_children_rejected() -> None:
    raw = {
        "id": "root",
        "question": "q",
        "node_type": "leaf",
        "grounding": "evidence",
        "children": [_leaf("a")],
    }
    with pytest.raises(PyramidParseError, match="must not have children"):
        parse_decomposition(raw, _SETTINGS)


def test_parse_unknown_node_type_rejected() -> None:
    raw = {"id": "root", "question": "q", "node_type": "nand", "grounding": "evidence"}
    with pytest.raises(PyramidParseError, match="not a valid tree"):
        parse_decomposition(raw, _SETTINGS)


# --- Parser: cap pruning ------------------------------------------------------


def test_depth_cap_prunes_subtree_to_leaf() -> None:
    # root(0) -> mid(1) -> deep(2, would need children at depth 3). With max_depth
    # 2, the depth-2 internal node is collapsed to a leaf.
    settings = PyramidSettings(max_depth=2)
    raw = {
        "id": "root",
        "question": "q",
        "node_type": "conjunction",
        "children": [
            {
                "id": "mid",
                "question": "q",
                "node_type": "conjunction",
                "children": [
                    {
                        "id": "deep",
                        "question": "q",
                        "node_type": "conjunction",
                        "children": [_leaf("x"), _leaf("y")],
                    }
                ],
            }
        ],
    }
    tree, notes = parse_decomposition(raw, settings)
    assert [n.cap for n in notes] == ["depth"]
    assert notes[0].node_id == "deep"
    deep = tree.children[0].children[0]
    assert deep.node_type == "leaf"
    assert deep.grounding == "evidence"  # pruned leaves default to evidence
    assert deep.children == []


def test_children_cap_drops_overflow() -> None:
    settings = PyramidSettings(max_children=2)
    raw = {
        "id": "root",
        "question": "q",
        "node_type": "disjunction",
        "children": [_leaf("a"), _leaf("b"), _leaf("c"), _leaf("d")],
    }
    tree, notes = parse_decomposition(raw, settings)
    assert [n.cap for n in notes] == ["children"]
    assert [c.id for c in tree.children] == ["a", "b"]


def test_conjunct_cap_trims_conjunction() -> None:
    # 4 children is within max_children (4) but over max_conjuncts (3), so only
    # the conjunct cap fires — a clean, isolated conjunct-cap case.
    settings = PyramidSettings(max_children=4, max_conjuncts=3)
    raw = {
        "id": "root",
        "question": "q",
        "node_type": "conjunction",
        "children": [_leaf("a"), _leaf("b"), _leaf("c"), _leaf("d")],
    }
    tree, notes = parse_decomposition(raw, settings)
    assert [n.cap for n in notes] == ["conjunct"]
    assert [c.id for c in tree.children] == ["a", "b", "c"]


def test_conjunct_cap_not_applied_to_disjunction() -> None:
    # A disjunction is bounded only by max_children, never max_conjuncts.
    settings = PyramidSettings(max_children=4, max_conjuncts=3)
    raw = {
        "id": "root",
        "question": "q",
        "node_type": "disjunction",
        "children": [_leaf("a"), _leaf("b"), _leaf("c"), _leaf("d")],
    }
    tree, notes = parse_decomposition(raw, settings)
    assert notes == []
    assert len(tree.children) == 4


# --- Assumptions: schema + guard ----------------------------------------------


def _assumption(
    text: str, load_bearing: bool = True, verifiability: str = "researchable"
) -> dict[str, object]:
    return {"text": text, "load_bearing": load_bearing, "verifiability": verifiability}


def test_parse_leaf_with_assumptions() -> None:
    raw = {
        "id": "root",
        "question": "q",
        "node_type": "leaf",
        "grounding": "evidence",
        "assumptions": [
            _assumption("The vote happens on schedule", True, "researchable"),
            _assumption("Nobody redefines the metric", False, "unverifiable"),
        ],
    }
    tree, notes = parse_decomposition(raw, _SETTINGS)
    assert notes == []
    assert len(tree.assumptions) == 2
    a0 = tree.assumptions[0]
    assert a0.text == "The vote happens on schedule"
    assert a0.load_bearing is True
    assert a0.verifiability == "researchable"
    # Runtime fields default until the verification pass fills them.
    assert a0.status == "unverified"
    assert a0.evidence == ""
    assert a0.node_id == ""
    assert tree.assumptions[1].verifiability == "unverifiable"


def test_internal_node_carries_assumptions() -> None:
    raw = {
        "id": "root",
        "question": "q",
        "node_type": "conjunction",
        "assumptions": [_assumption("The regime holds")],
        "children": [_leaf("a"), _leaf("b", "model-knowledge")],
    }
    tree, notes = parse_decomposition(raw, _SETTINGS)
    assert notes == []
    assert [a.text for a in tree.assumptions] == ["The regime holds"]


def test_assumptions_over_cap_truncated_with_note() -> None:
    settings = PyramidSettings(max_assumptions_per_node=3)
    raw = {
        "id": "root",
        "question": "q",
        "node_type": "leaf",
        "grounding": "evidence",
        "assumptions": [_assumption(f"assumption {i}") for i in range(5)],
    }
    tree, notes = parse_decomposition(raw, settings)
    assert [n.cap for n in notes] == ["assumptions"]
    assert notes[0].node_id == "root"
    assert "dropped 2" in notes[0].message
    assert len(tree.assumptions) == 3
    assert [a.text for a in tree.assumptions] == [f"assumption {i}" for i in range(3)]


def test_assumptions_at_cap_not_truncated() -> None:
    settings = PyramidSettings(max_assumptions_per_node=3)
    raw = {
        "id": "root",
        "question": "q",
        "node_type": "leaf",
        "grounding": "evidence",
        "assumptions": [_assumption(f"a{i}") for i in range(3)],
    }
    tree, notes = parse_decomposition(raw, settings)
    assert notes == []
    assert len(tree.assumptions) == 3


def test_assumption_defaults_are_runtime_neutral() -> None:
    a = Assumption(text="x", load_bearing=True, verifiability="researchable")
    assert a.status == "unverified"
    assert a.evidence == ""
    assert a.node_id == ""


# --- JSON schema --------------------------------------------------------------


def test_decomposition_json_schema_shape() -> None:
    schema = decomposition_json_schema()
    assert isinstance(schema, dict)
    # Recursive model -> pydantic emits a $ref/$defs form; the node fields live in
    # one of those definitions. Confirm the key fields appear somewhere in it.
    text = str(schema)
    for field in ("node_type", "grounding", "children", "reasoning_type", "assumptions"):
        assert field in text
    # The Assumption sub-model's fields appear in the schema's $defs too.
    for field in ("load_bearing", "verifiability"):
        assert field in text


# --- Combination math: properties ---------------------------------------------


@given(ps=st.lists(in_band, min_size=1, max_size=5))
def test_conjunction_at_most_min_child(ps: list[float]) -> None:
    result = combine_conjunction(ps, 0.03, 0.97)
    assert result <= min(ps) + 1e-12


@given(ps=st.lists(in_band, min_size=1, max_size=5))
def test_disjunction_at_least_max_child(ps: list[float]) -> None:
    result = combine_disjunction(ps, 0.03, 0.97)
    assert result >= max(ps) - 1e-12


@given(ps=st.lists(in_band, min_size=1, max_size=5))
def test_conjunction_and_disjunction_in_unit_interval(ps: list[float]) -> None:
    assert 0.0 <= combine_conjunction(ps, 0.03, 0.97) <= 1.0
    assert 0.0 <= combine_disjunction(ps, 0.03, 0.97) <= 1.0


def test_conditional_equals_product_of_clipped() -> None:
    ps = [0.5, 0.4, 0.8]
    assert math.isclose(
        combine_conditional(ps, 0.03, 0.97),
        0.5 * 0.4 * 0.8,
        rel_tol=1e-12,
    )


@given(ps=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=5))
def test_clipping_bounds_respected_conjunction(ps: list[float]) -> None:
    low, high, n = 0.1, 0.9, len(ps)
    result = combine_conjunction(ps, low, high)
    # Every multiplied term is clipped into [low, high], so the product is bounded
    # by low**n below and high (each term <= high) above.
    assert low**n - 1e-12 <= result <= high + 1e-12


def test_clipping_extremes_use_bounds() -> None:
    # A single-term conjunction returns exactly the clipped value.
    assert combine_conjunction([1e-9], 0.03, 0.97) == 0.03
    assert combine_conjunction([1.0], 0.03, 0.97) == 0.97


# --- Combination math: combine_tree -------------------------------------------


def test_combine_tree_matches_hand_computed() -> None:
    # root = conjunction(a, disjunction(b, c)); all leaves in-band so no clipping.
    tree = PyramidNode(
        id="root",
        question="root?",
        node_type="conjunction",
        children=[
            PyramidNode(id="a", question="a?", node_type="leaf", grounding="evidence"),
            PyramidNode(
                id="d",
                question="d?",
                node_type="disjunction",
                children=[
                    PyramidNode(id="b", question="b?", node_type="leaf", grounding="evidence"),
                    PyramidNode(id="c", question="c?", node_type="leaf", grounding="evidence"),
                ],
            ),
        ],
    )
    estimates = {"a": 0.5, "b": 0.4, "c": 0.3}
    structural, trace = combine_tree(tree, estimates, clip_low=0.03, clip_high=0.97)

    disj = 1.0 - (1.0 - 0.4) * (1.0 - 0.3)  # 0.58
    expected = 0.5 * disj
    assert math.isclose(structural, expected, rel_tol=1e-12)

    # Trace covers every node with the right operations and outputs.
    assert set(trace) == {"root", "a", "d", "b", "c"}
    assert trace["d"].operation == "disjunction"
    assert math.isclose(trace["d"].output, disj, rel_tol=1e-12)
    assert trace["root"].operation == "conjunction"
    assert trace["a"].operation == "leaf"
    assert trace["a"].output == 0.5


def test_combine_tree_single_leaf() -> None:
    tree = PyramidNode(id="root", question="q", node_type="leaf", grounding="evidence")
    structural, trace = combine_tree(tree, {"root": 0.62}, clip_low=0.03, clip_high=0.97)
    assert structural == 0.62
    assert list(trace) == ["root"]


def test_combine_tree_missing_leaf_estimate_raises() -> None:
    tree = PyramidNode(id="root", question="q", node_type="leaf", grounding="evidence")
    with pytest.raises(ValueError, match="no leaf estimate"):
        combine_tree(tree, {}, clip_low=0.03, clip_high=0.97)


def test_combine_tree_llm_synthesis_mode_not_implemented() -> None:
    tree = PyramidNode(
        id="root",
        question="q",
        node_type="conjunction",
        children=[
            PyramidNode(id="a", question="a", node_type="leaf", grounding="evidence"),
            PyramidNode(id="b", question="b", node_type="leaf", grounding="evidence"),
        ],
    )
    with pytest.raises(NotImplementedError, match="llm-synthesis"):
        combine_tree(
            tree, {"a": 0.5, "b": 0.5}, clip_low=0.03, clip_high=0.97, mode="llm-synthesis"
        )


# --- blend --------------------------------------------------------------------


@given(p=in_band, w=st.floats(min_value=0.0, max_value=1.0))
def test_blend_identical_inputs_returns_input(p: float, w: float) -> None:
    assert math.isclose(blend(p, p, w), p, rel_tol=1e-9, abs_tol=1e-12)


@given(w=st.floats(min_value=0.0, max_value=1.0))
def test_blend_endpoints(w: float) -> None:
    # weight 0 -> pure structural, weight 1 -> pure holistic.
    assert math.isclose(blend(0.3, 0.8, 0.0), 0.3, rel_tol=1e-9)
    assert math.isclose(blend(0.3, 0.8, 1.0), 0.8, rel_tol=1e-9)


def test_blend_monotone_in_weight() -> None:
    # holistic (0.8) > structural (0.2): raising the holistic weight raises blend.
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    values = [blend(0.2, 0.8, w) for w in weights]
    assert all(a < b for a, b in itertools.pairwise(values))


def test_blend_rejects_out_of_range_weight() -> None:
    with pytest.raises(ValueError, match="weight"):
        blend(0.5, 0.5, 1.5)
    with pytest.raises(ValueError, match="weight"):
        blend(0.5, 0.5, -0.1)


# --- Config: PyramidSettings --------------------------------------------------


def test_load_pyramid_settings_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("[tournament]\nlive = 'x'\n", encoding="utf-8")
    settings = load_pyramid_settings(path)
    assert settings == PyramidSettings()


def test_pyramid_settings_assumption_defaults() -> None:
    settings = PyramidSettings()
    assert settings.max_assumptions_per_node == 8
    assert settings.max_assumption_verifications == 6


def test_load_pyramid_settings_assumption_knobs(tmp_path: Path) -> None:
    toml = "[pyramid]\nmax_assumptions_per_node = 5\nmax_assumption_verifications = 2\n"
    path = tmp_path / "settings.toml"
    path.write_text(toml, encoding="utf-8")
    settings = load_pyramid_settings(path)
    assert settings.max_assumptions_per_node == 5
    assert settings.max_assumption_verifications == 2
    # Untouched knobs keep their defaults.
    assert settings.max_depth == PyramidSettings().max_depth


def test_load_pyramid_settings_full_table(tmp_path: Path) -> None:
    toml = """\
[pyramid]
max_depth = 3
max_children = 5
max_conjuncts = 2
leaf_clip_low = 0.05
leaf_clip_high = 0.95
holistic_blend_weight = 0.4
combination_mode = "guarded-product"
"""
    path = tmp_path / "settings.toml"
    path.write_text(toml, encoding="utf-8")
    settings = load_pyramid_settings(path)
    assert settings == PyramidSettings(
        max_depth=3,
        max_children=5,
        max_conjuncts=2,
        leaf_clip_low=0.05,
        leaf_clip_high=0.95,
        holistic_blend_weight=0.4,
        combination_mode="guarded-product",
    )


def test_load_pyramid_settings_rejects_bad_mode(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[pyramid]\ncombination_mode = "magic"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="combination_mode"):
        load_pyramid_settings(path)


def test_conditional_chain_also_capped_at_max_conjuncts() -> None:
    """Conditional chains multiply like conjunctions, so the same cap applies."""
    raw = {
        "id": "root",
        "question": "Root?",
        "node_type": "conditional",
        "children": [_leaf(f"c{i}", grounding="evidence") for i in range(4)],
    }
    tree, notes = parse_decomposition(raw, PyramidSettings())
    assert len(tree.children) == 3
    assert any(n.cap == "conjunct" and n.node_id == "root" for n in notes)
