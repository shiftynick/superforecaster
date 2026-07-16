"""Tests for the decomposition-pyramid runner, driven by fake LLM clients.

No live LLM or network. The single seam is ``client_factory``: it hands the
runner a canned client per task config (decompose / leaf / holistic). Fakes emit
either a decomposition tree JSON, a forecast-pass JSON, or raise ``LlmError`` to
exercise the degrade paths. Task models come from a temp ``settings.toml`` pinned
through ``SUPERFORECASTER_CONFIG`` so the tests never touch the real control panel.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from superforecaster.aggregation import pool_log_odds
from superforecaster.config import TaskModelConfig
from superforecaster.forecaster import ForecastRecord
from superforecaster.llm import LlmError, LlmResult
from superforecaster.models import QuestionSnapshot
from superforecaster.pyramid import Assumption
from superforecaster.pyramid_runner import (
    _select_assumptions_to_verify,
    run_pyramid_forecast,
)
from superforecaster.research import ResearchBundle
from superforecaster.search import SearchResult

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)

# A settings.toml with the three task tables the runner needs (decompose, the
# leaf pass, and holistic). Models are dummy strings — the fakes ignore them.
_SETTINGS_TOML = """\
[tasks.decompose]
provider = "claude-cli"
model = "decompose-model"
max_tokens = 1000

[tasks.forecast_pass]
provider = "claude-cli"
model = "leaf-model"
max_tokens = 1000

[tasks.holistic]
provider = "claude-cli"
model = "holistic-model"
max_tokens = 1000

[tasks.assumption_verdict]
provider = "claude-cli"
model = "verdict-model"
max_tokens = 1000
"""


@pytest.fixture
def _pin_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "settings.toml"
    config.write_text(_SETTINGS_TOML, encoding="utf-8")
    monkeypatch.setenv("SUPERFORECASTER_CONFIG", str(config))


def _forecast_json(probability: float, confidence: str = "medium") -> str:
    return (
        "Reasoning about the base rate and adjustments.\n\n"
        "```json\n"
        f'{{"probability": {probability}, "confidence": "{confidence}", '
        '"rationale": "leaf base rate then a small adjustment", '
        '"reference_class": "comparable cases"}\n'
        "```"
    )


def _tree_json(tree: str) -> str:
    return f"A decomposition follows.\n\n```json\n{tree}\n```"


# A two-leaf conjunction: root(conjunction) -> [leaf a (evidence), leaf b (model)].
_TWO_LEAF_TREE = (
    '{"id": "root", "question": "Will the whole thing happen?", '
    '"node_type": "conjunction", "children": ['
    '{"id": "a", "question": "Will step A happen?", "node_type": "leaf", '
    '"grounding": "evidence"}, '
    '{"id": "b", "question": "Is the base rate high?", "node_type": "leaf", '
    '"grounding": "model-knowledge"}]}'
)


class _RecordingLlm:
    """A fake client that returns a scripted reply and records every prompt it saw.

    The runner builds one client per task config via ``client_factory``; each such
    client here yields its next scripted reply in order. Prompts are captured so a
    test can assert what context a leaf was (or was not) shown.
    """

    def __init__(self, replies: list[object]) -> None:
        # Each reply is either a string (returned as text) or an LlmError (raised).
        self._replies = list(replies)
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        self.calls.append({"system": system, "prompt": prompt, "model": model})
        reply = self._replies.pop(0)
        if isinstance(reply, LlmError):
            raise reply
        assert isinstance(reply, str)
        return LlmResult(
            text=reply,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            stop_reason="end_turn",
            model=model,
        )


def _factory_by_model(
    clients: dict[str, _RecordingLlm],
) -> Callable[[TaskModelConfig], _RecordingLlm]:
    """Route each task config to a client keyed by its model name."""

    def factory(config: TaskModelConfig) -> _RecordingLlm:
        return clients[config.model]

    return factory


def _snapshot() -> QuestionSnapshot:
    return QuestionSnapshot(
        scraped_at=NOW,
        post_id=201,
        question_id=301,
        title="Will the merger close by year end?",
        description="A corporate question.",
        resolution_criteria="Resolves YES if the deal legally closes before Jan 1.",
        fine_print="",
        scheduled_close_time=datetime(2026, 12, 1, tzinfo=UTC),
        status="open",
    )


def _bundle() -> ResearchBundle:
    return ResearchBundle(
        queries=["merger news"],
        results=[
            SearchResult(
                url="https://news.example/deal",
                title="Deal update",
                snippet="The merger is progressing.",
                published_at=None,
                source_domain="news.example",
                provider="asknews",
                raw={},
            )
        ],
        docs=[],
        tier_stats={},
        search_credits_used=3,
        retrieved_at=NOW,
    )


def test_happy_path_two_leaf_tree(_pin_settings: None) -> None:
    decompose = _RecordingLlm([_tree_json(_TWO_LEAF_TREE)])
    # Leaf a -> 0.5, leaf b -> 0.4; both clients share the leaf model, so ONE
    # client serves both leaves in order.
    leaf = _RecordingLlm([_forecast_json(0.5, "high"), _forecast_json(0.4, "medium")])
    holistic = _RecordingLlm([_forecast_json(0.3, "low")])
    clients = {
        "decompose-model": decompose,
        "leaf-model": leaf,
        "holistic-model": holistic,
    }

    record = run_pyramid_forecast(
        _snapshot(),
        research=_bundle(),
        client_factory=_factory_by_model(clients),
        now=NOW,
    )

    assert isinstance(record, ForecastRecord)
    assert record.method == "pyramid-c"
    assert record.model == "pyramid-c"
    detail = record.pyramid
    assert detail is not None

    # Structural = product of clipped leaves 0.5 * 0.4 = 0.20.
    assert detail.structural_probability == pytest.approx(0.5 * 0.4)
    assert detail.holistic_probability == pytest.approx(0.3)
    # Blended is the log-odds pool of structural and holistic (weight 0.5).
    assert 0.0 < record.probability < 1.0
    assert record.raw_probability == pytest.approx(
        pool_log_odds([detail.structural_probability, detail.holistic_probability], [0.5, 0.5])
    )

    # Trace: two leaves, one combination step per node (root + 2 leaves).
    assert len(detail.leaf_estimates) == 2
    assert {e.node_id for e in detail.leaf_estimates} == {"a", "b"}
    assert {s.node_id for s in detail.combination_steps} == {"root", "a", "b"}
    assert not detail.decompose_failed
    assert detail.prune_notes == []

    # Both raw confidence signals are stored.
    assert detail.weakest_leaf_confidence in {"low", "medium", "high"}
    assert detail.structural_holistic_gap >= 0.0

    # Cost sums decompose + 2 leaves + holistic = 4 calls x $0.01.
    assert record.cost_usd == pytest.approx(0.04)
    assert record.input_tokens == 400
    assert record.research_search_credits == 3


def test_model_knowledge_leaf_gets_no_bundle(_pin_settings: None) -> None:
    # Leaf a is evidence, leaf b is model-knowledge (see _TWO_LEAF_TREE).
    decompose = _RecordingLlm([_tree_json(_TWO_LEAF_TREE)])
    leaf = _RecordingLlm([_forecast_json(0.6), _forecast_json(0.6)])
    holistic = _RecordingLlm([_forecast_json(0.6)])
    clients = {"decompose-model": decompose, "leaf-model": leaf, "holistic-model": holistic}

    run_pyramid_forecast(
        _snapshot(),
        research=_bundle(),
        client_factory=_factory_by_model(clients),
        now=NOW,
    )

    # The leaf client served two calls: first the evidence leaf (a), then the
    # model-knowledge leaf (b). The evidence leaf's prompt carries the SOURCES
    # block; the model-knowledge one must NOT.
    evidence_prompt = leaf.calls[0]["prompt"]
    model_prompt = leaf.calls[1]["prompt"]
    assert isinstance(evidence_prompt, str) and isinstance(model_prompt, str)
    assert "SOURCES" in evidence_prompt
    assert "news.example" in evidence_prompt
    assert "SOURCES" not in model_prompt


def test_decompose_parse_failure_falls_back_to_single_leaf(_pin_settings: None) -> None:
    # Decompose returns junk twice -> after one retry, degrade to a single leaf.
    decompose = _RecordingLlm(["not json at all", "still not json"])
    leaf = _RecordingLlm([_forecast_json(0.42)])
    holistic = _RecordingLlm([_forecast_json(0.42)])
    clients = {"decompose-model": decompose, "leaf-model": leaf, "holistic-model": holistic}

    record = run_pyramid_forecast(
        _snapshot(),
        research=_bundle(),
        client_factory=_factory_by_model(clients),
        now=NOW,
    )

    detail = record.pyramid
    assert detail is not None
    assert detail.decompose_failed is True
    assert detail.tree.node_type == "leaf"
    assert len(detail.leaf_estimates) == 1
    # The single fallback leaf poses the original question and is grounded in evidence.
    assert detail.leaf_estimates[0].question == "Will the merger close by year end?"
    assert detail.leaf_estimates[0].grounding == "evidence"
    assert any("did not parse" in note for note in detail.notes)
    # Decompose was retried once (two calls).
    assert len(decompose.calls) == 2


def test_decompose_retry_then_succeeds(_pin_settings: None) -> None:
    # First decompose reply is junk; the retry returns a valid single-leaf tree.
    good = '{"id": "root", "question": "Direct?", "node_type": "leaf", "grounding": "evidence"}'
    decompose = _RecordingLlm(["junk", _tree_json(good)])
    leaf = _RecordingLlm([_forecast_json(0.55)])
    holistic = _RecordingLlm([_forecast_json(0.55)])
    clients = {"decompose-model": decompose, "leaf-model": leaf, "holistic-model": holistic}

    record = run_pyramid_forecast(
        _snapshot(),
        research=_bundle(),
        client_factory=_factory_by_model(clients),
        now=NOW,
    )
    detail = record.pyramid
    assert detail is not None
    assert detail.decompose_failed is False
    assert detail.tree.node_type == "leaf"
    assert len(decompose.calls) == 2  # one failure + one successful retry


def test_failed_leaf_degrades_not_crashes(_pin_settings: None) -> None:
    decompose = _RecordingLlm([_tree_json(_TWO_LEAF_TREE)])
    # An LlmError propagates straight out of run_forecast_pass (it only retries on
    # PARSE failures), so leaf a fails on its single call; leaf b then succeeds.
    leaf = _RecordingLlm([LlmError("leaf a boom"), _forecast_json(0.8)])
    holistic = _RecordingLlm([_forecast_json(0.5)])
    clients = {"decompose-model": decompose, "leaf-model": leaf, "holistic-model": holistic}

    record = run_pyramid_forecast(
        _snapshot(),
        research=_bundle(),
        client_factory=_factory_by_model(clients),
        now=NOW,
    )

    detail = record.pyramid
    assert detail is not None
    a = next(e for e in detail.leaf_estimates if e.node_id == "a")
    b = next(e for e in detail.leaf_estimates if e.node_id == "b")
    assert a.failed is True
    assert a.probability == 0.5
    assert a.confidence == "low"
    assert b.failed is False and b.probability == pytest.approx(0.8)
    assert any("leaf 'a' failed" in note for note in detail.notes)
    # Weakest-leaf confidence on the multiplied conjunction path is the failed
    # leaf's "low".
    assert detail.weakest_leaf_confidence == "low"


def test_holistic_failure_blends_structural_against_itself(_pin_settings: None) -> None:
    decompose = _RecordingLlm([_tree_json(_TWO_LEAF_TREE)])
    leaf = _RecordingLlm([_forecast_json(0.5), _forecast_json(0.4)])
    holistic = _RecordingLlm([LlmError("holistic down")])
    clients = {"decompose-model": decompose, "leaf-model": leaf, "holistic-model": holistic}

    record = run_pyramid_forecast(
        _snapshot(),
        research=_bundle(),
        client_factory=_factory_by_model(clients),
        now=NOW,
    )

    detail = record.pyramid
    assert detail is not None
    # Holistic degrades to the structural value, so the blend equals structural.
    assert detail.holistic_probability == pytest.approx(detail.structural_probability)
    assert record.raw_probability == pytest.approx(detail.structural_probability)
    assert detail.structural_holistic_gap == pytest.approx(0.0)
    assert any("holistic pass failed" in note for note in detail.notes)


def test_blind_run_has_no_bundle_anywhere(_pin_settings: None) -> None:
    # research=None: every call runs blind, and no leaf prompt shows SOURCES.
    decompose = _RecordingLlm([_tree_json(_TWO_LEAF_TREE)])
    leaf = _RecordingLlm([_forecast_json(0.5), _forecast_json(0.5)])
    holistic = _RecordingLlm([_forecast_json(0.5)])
    clients = {"decompose-model": decompose, "leaf-model": leaf, "holistic-model": holistic}

    record = run_pyramid_forecast(
        _snapshot(),
        research=None,
        client_factory=_factory_by_model(clients),
        now=NOW,
    )
    assert record.research is None
    assert record.research_search_credits == 0
    for call in leaf.calls:
        prompt = call["prompt"]
        assert isinstance(prompt, str) and "SOURCES" not in prompt


def test_leaf_snapshot_carries_root_resolution_context() -> None:
    """P1 regression: leaves must see the root's resolution criteria as context."""
    from superforecaster.pyramid_runner import _leaf_snapshot

    root = _snapshot()
    leaf = _leaf_snapshot(root, "Did the price ever exceed the threshold?")
    assert leaf.title == "Did the price ever exceed the threshold?"
    assert root.resolution_criteria in leaf.description
    assert "ROOT" in leaf.description
    # The root criteria travel as labeled context, not as the leaf's own criteria.
    assert leaf.resolution_criteria == ""


# --- Assumption verification --------------------------------------------------

# A two-leaf conjunction where leaf a carries one load-bearing RESEARCHABLE
# assumption and leaf b carries one UNVERIFIABLE assumption (never searched).
_ASSUMPTION_TREE = (
    '{"id": "root", "question": "Will the whole thing happen?", '
    '"node_type": "conjunction", "children": ['
    '{"id": "a", "question": "Will step A happen?", "node_type": "leaf", '
    '"grounding": "evidence", "assumptions": ['
    '{"text": "The vote happens on schedule", "load_bearing": true, '
    '"verifiability": "researchable"}]}, '
    '{"id": "b", "question": "Is the base rate high?", "node_type": "leaf", '
    '"grounding": "model-knowledge", "assumptions": ['
    '{"text": "No one redefines the metric", "load_bearing": false, '
    '"verifiability": "unverifiable"}]}]}'
)


def _verdict_json(verdict: str, evidence: str = "settled by the official calendar") -> str:
    return (
        "Weighing the evidence.\n\n```json\n"
        f'{{"verdict": "{verdict}", "evidence": "{evidence}"}}\n```'
    )


class _FakeSearch:
    """A fake search client: records queries, returns canned results, meters credits.

    Mirrors the ``SearchClient`` protocol (``search(query, *, max_results, ...)``).
    When ``per_call_credits`` > 0 it also exposes ``credits_used_this_month`` (like
    the metered AskNews client) so the runner's credit accounting can be exercised.
    """

    def __init__(
        self,
        results: list[SearchResult] | None = None,
        *,
        error: Exception | None = None,
        per_call_credits: int = 0,
    ) -> None:
        self._results = results if results is not None else [_search_result()]
        self._error = error
        self._per_call = per_call_credits
        self._credits = 0
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        max_results: int,
        start_date: object = None,
        end_date: object = None,
    ) -> list[SearchResult]:
        self.queries.append(query)
        if self._error is not None:
            raise self._error
        self._credits += self._per_call
        return list(self._results)

    def credits_used_this_month(self) -> int:
        return self._credits


def _search_result() -> SearchResult:
    return SearchResult(
        url="https://gov.example/calendar",
        title="Official vote calendar",
        snippet="The vote is scheduled as planned.",
        published_at=None,
        source_domain="gov.example",
        provider="tavily",
        raw={},
    )


def _verification_clients(verdict_reply: object) -> dict[str, _RecordingLlm]:
    """The four task clients a verification run drives (verdict reply scripted)."""
    return {
        "decompose-model": _RecordingLlm([_tree_json(_ASSUMPTION_TREE)]),
        "leaf-model": _RecordingLlm([_forecast_json(0.5), _forecast_json(0.4)]),
        "holistic-model": _RecordingLlm([_forecast_json(0.3)]),
        "verdict-model": _RecordingLlm([verdict_reply]),
    }


def test_assumption_verification_happy_path(_pin_settings: None) -> None:
    clients = _verification_clients(_verdict_json("verified-true"))
    search = _FakeSearch(per_call_credits=2)

    record = run_pyramid_forecast(
        _snapshot(),
        research=_bundle(),
        search_client=search,
        client_factory=_factory_by_model(clients),
        now=NOW,
    )
    detail = record.pyramid
    assert detail is not None

    # Both assumptions are stored flat, with node ids filled by the collector.
    assert {a.node_id for a in detail.assumptions} == {"a", "b"}
    researchable = next(a for a in detail.assumptions if a.node_id == "a")
    unverifiable = next(a for a in detail.assumptions if a.node_id == "b")
    assert researchable.status == "verified-true"
    assert researchable.evidence == "settled by the official calendar"
    # The unverifiable one is never searched — it stays unverified.
    assert unverifiable.status == "unverified"

    # Exactly one researchable assumption was verified: one search issued.
    assert search.queries == ["The vote happens on schedule"]
    assert detail.assumption_verifications_run == 1
    assert detail.assumption_search_credits == 2

    # Verification search credits fold into the record's total (bundle 3 + 2).
    assert record.research_search_credits == 5
    # Cost: decompose + 2 leaves + holistic + 1 verdict = 5 calls x $0.01.
    assert record.cost_usd == pytest.approx(0.05)


def test_assumption_verdict_failure_degrades_to_unverified(_pin_settings: None) -> None:
    # The verdict LLM raises: the assumption is left unverified, with a note.
    clients = _verification_clients(LlmError("verdict boom"))
    search = _FakeSearch(per_call_credits=2)

    record = run_pyramid_forecast(
        _snapshot(),
        research=_bundle(),
        search_client=search,
        client_factory=_factory_by_model(clients),
        now=NOW,
    )
    detail = record.pyramid
    assert detail is not None
    researchable = next(a for a in detail.assumptions if a.node_id == "a")
    assert researchable.status == "unverified"
    assert any("verdict call failed" in note for note in detail.notes)
    # The search still ran (credit drawn) even though the verdict failed.
    assert detail.assumption_search_credits == 2


def test_assumption_search_failure_degrades_to_unverified(_pin_settings: None) -> None:
    from superforecaster.search import SearchError

    clients = _verification_clients(_verdict_json("verified-true"))
    search = _FakeSearch(error=SearchError("search down"))

    record = run_pyramid_forecast(
        _snapshot(),
        research=_bundle(),
        search_client=search,
        client_factory=_factory_by_model(clients),
        now=NOW,
    )
    detail = record.pyramid
    assert detail is not None
    researchable = next(a for a in detail.assumptions if a.node_id == "a")
    assert researchable.status == "unverified"
    assert any("verification search failed" in note for note in detail.notes)
    # The verdict client's scripted reply was never consumed (no verdict call).
    assert clients["verdict-model"].calls == []


def test_verified_false_injected_into_leaf_prompt(_pin_settings: None) -> None:
    clients = _verification_clients(_verdict_json("verified-false", "the vote was postponed"))
    search = _FakeSearch(per_call_credits=1)

    record = run_pyramid_forecast(
        _snapshot(),
        research=_bundle(),
        search_client=search,
        client_factory=_factory_by_model(clients),
        now=NOW,
    )
    detail = record.pyramid
    assert detail is not None
    researchable = next(a for a in detail.assumptions if a.node_id == "a")
    assert researchable.status == "verified-false"

    # Leaf a owns the falsified assumption; its estimation SYSTEM prompt must carry
    # the corrected-assumption injection. Leaf b (no falsified assumption) must not.
    leaf = clients["leaf-model"]
    a_system = leaf.calls[0]["system"]
    b_system = leaf.calls[1]["system"]
    assert isinstance(a_system, str) and isinstance(b_system, str)
    assert "CORRECTED ASSUMPTIONS" in a_system
    assert "The vote happens on schedule" in a_system
    assert "the vote was postponed" in a_system
    assert "CORRECTED ASSUMPTIONS" not in b_system


def test_no_search_client_leaves_assumptions_unverified(_pin_settings: None) -> None:
    # search_client=None disables verification: assumptions surfaced, none checked.
    clients = _verification_clients(_verdict_json("verified-true"))

    record = run_pyramid_forecast(
        _snapshot(),
        research=_bundle(),
        search_client=None,
        client_factory=_factory_by_model(clients),
        now=NOW,
    )
    detail = record.pyramid
    assert detail is not None
    assert {a.node_id for a in detail.assumptions} == {"a", "b"}
    assert all(a.status == "unverified" for a in detail.assumptions)
    assert detail.assumption_verifications_run == 0
    assert detail.assumption_search_credits == 0
    assert record.research_search_credits == 3  # bundle only, no verification draw
    assert any("no search client" in note for note in detail.notes)
    # The verdict client was never used.
    assert clients["verdict-model"].calls == []


def test_select_assumptions_dedupe_ordering_and_cap() -> None:
    # Tree order: nlb1, lb1 (dup of lb-a later), nlb2, lb2. Load-bearing first,
    # tree order preserved within each group, near-identical text deduped, cap 2.
    assumptions = [
        Assumption(text="Not load bearing one", load_bearing=False, verifiability="researchable"),
        Assumption(text="Load bearing A", load_bearing=True, verifiability="researchable"),
        Assumption(text="  load  BEARING a  ", load_bearing=True, verifiability="researchable"),
        Assumption(text="Load bearing B", load_bearing=True, verifiability="researchable"),
    ]
    selected = _select_assumptions_to_verify(assumptions, cap=2)
    # Deduped ["Not load bearing one", "Load bearing A", "Load bearing B"] then
    # load-bearing-first -> [A, B, Not...], capped at 2 -> [A, B].
    assert [a.text for a in selected] == ["Load bearing A", "Load bearing B"]


def test_select_assumptions_orders_load_bearing_first() -> None:
    assumptions = [
        Assumption(text="cheap check", load_bearing=False, verifiability="researchable"),
        Assumption(text="pivotal check", load_bearing=True, verifiability="researchable"),
    ]
    selected = _select_assumptions_to_verify(assumptions, cap=5)
    assert [a.text for a in selected] == ["pivotal check", "cheap check"]
