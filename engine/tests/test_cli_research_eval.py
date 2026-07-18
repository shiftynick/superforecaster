"""CLI tests for the research-eval commands — no network, no LLM, no browser.

The judge LLM, the search clients, and the extractor are all injected through
their monkeypatch seams; bundles come from a fake research context so no real
retrieval happens.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from superforecaster import cli
from superforecaster.config import RetrievalSettings
from superforecaster.extract import TIER_HTTPX_TRAFILATURA, ExtractedDoc
from superforecaster.llm import LlmResult
from superforecaster.models import QuestionSnapshot
from superforecaster.research import ResearchBundle
from superforecaster.research_eval import BundleEvalStore
from superforecaster.search import SearchResult
from superforecaster.store import SnapshotStore

runner = CliRunner()

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _snapshot(post_id: int = 900, question_id: int = 800) -> QuestionSnapshot:
    return QuestionSnapshot(
        scraped_at=NOW,
        post_id=post_id,
        question_id=question_id,
        title="Will the Fed cut rates in September 2026?",
        description="",
        resolution_criteria=(
            "Resolves YES if the FOMC lowers the target range, per federalreserve.gov."
        ),
        fine_print="",
        scheduled_close_time=datetime(2026, 9, 30, tzinfo=UTC),
        status="open",
    )


def _bundle(domain: str = "federalreserve.gov") -> ResearchBundle:
    result = SearchResult(
        url=f"https://{domain}/x",
        title="FOMC rate decision",
        snippet="The FOMC target range",
        published_at=NOW,
        source_domain=domain,
        provider="asknews",
        raw={},
    )
    doc = ExtractedDoc(
        url=f"https://{domain}/x",
        text="rate cut target range details",
        title="Doc",
        published_at=NOW,
        tier=TIER_HTTPX_TRAFILATURA,
        ok=True,
    )
    return ResearchBundle(
        queries=["fed rate cut"],
        results=[result],
        docs=[doc],
        tier_stats={TIER_HTTPX_TRAFILATURA: 1},
        search_credits_used=2,
        news_quota_exhausted=False,
        retrieved_at=NOW,
    )


class _FakeJudge:
    """A judge LLM returning a fixed reply, recording each prompt."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[str] = []

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        self.calls.append(prompt)
        return LlmResult(
            text=self._text,
            input_tokens=100,
            output_tokens=40,
            cost_usd=0.001,
            stop_reason="end_turn",
            model=model,
        )


_RUBRIC_TEXT = (
    "{"
    '"coverage": 4, "coverage_justification": "Directly on the rate decision.", '
    '"source_quality": 5, "source_quality_justification": "Official Fed source present.", '
    '"recency": 5, "recency_justification": "Latest FOMC news is here.", '
    '"relevance": 4, "relevance_justification": "Mostly about this question.", '
    '"resolution_grounding": 5, "resolution_grounding_justification": "fed source present.", '
    '"overall": 4, "missing": "A dissent-scenario source would help."'
    "}"
)

_PAIRWISE_TEXT = (
    "{"
    '"coverage": "X", "source_quality": "X", "recency": "Y", "relevance": "tie", '
    '"resolution_grounding": "X", "overall": "X", "reason": "X carries the resolution source."'
    "}"
)


def _seed_snapshot(data_dir: Path, snapshot: QuestionSnapshot) -> None:
    SnapshotStore(data_dir).append([snapshot])


def _make_config_file(
    tmp_path: Path, name: str, news_provider: str, *, with_tasks: bool = False
) -> Path:
    path = tmp_path / name
    body = (
        "[retrieval]\n"
        f'news_provider = "{news_provider}"\n'
        'general_provider = "tavily"\n'
        "max_results = 5\n"
    )
    if with_tasks:
        body += (
            "\n[tasks.forecast_pass]\n"
            'provider = "openrouter"\n'
            'model = "z-ai/glm-5.2"\n'
            "max_tokens = 2048\n"
            "\n[tasks.query_gen]\n"
            'provider = "openrouter"\n'
            'model = "z-ai/glm-5.2"\n'
            "max_tokens = 1024\n"
        )
    path.write_text(body, encoding="utf-8")
    return path


# --- eval-research ------------------------------------------------------------


def test_eval_research_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "snapshots"
    snapshot = _snapshot()
    _seed_snapshot(data_dir, snapshot)
    # Prime the research cache the forecast command fills (keyed by question id).
    from superforecaster.research_cache import ResearchCache

    ResearchCache(data_dir.parent / "research").save(snapshot.question_id, _bundle())

    judge = _FakeJudge(_RUBRIC_TEXT)
    monkeypatch.setattr(cli, "_build_bundle_judge_llm", lambda: judge)
    monkeypatch.setattr(cli, "_require_llm_cli", lambda *a, **k: None)

    result = runner.invoke(
        cli.app, ["eval-research", "--post-id", "900", "--cached", "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "overall=4" in result.output
    assert "resolution_source=yes" in result.output
    assert len(judge.calls) == 1
    # The judge saw no provider identity.
    assert "asknews" not in judge.calls[0]

    store = BundleEvalStore(data_dir.parent / "bundle-evals")
    (record,) = list(store.iter_all())
    assert record.kind == "rubric"
    assert record.rubric is not None
    assert record.rubric.overall == 4
    assert record.metrics.resolution_source_present is True


def test_eval_research_cache_miss_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "snapshots"
    _seed_snapshot(data_dir, _snapshot())
    monkeypatch.setattr(cli, "_require_llm_cli", lambda *a, **k: None)

    result = runner.invoke(
        cli.app, ["eval-research", "--post-id", "900", "--cached", "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 1
    assert "no cached research bundle" in result.output


def test_eval_research_unknown_post_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "snapshots"
    _seed_snapshot(data_dir, _snapshot())
    monkeypatch.setattr(cli, "_require_llm_cli", lambda *a, **k: None)

    result = runner.invoke(
        cli.app, ["eval-research", "--post-id", "999", "--cached", "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 1
    assert "no tracked snapshot" in result.output


# --- compare-research ---------------------------------------------------------


def test_compare_research_builds_two_bundles_under_different_configs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "snapshots"
    snapshot = _snapshot()
    _seed_snapshot(data_dir, snapshot)

    # Main config drives bundle A; alt config drives bundle B — differ only in the
    # news provider. Point the engine's default config resolution at the main file.
    main_config = _make_config_file(tmp_path, "main.toml", "asknews", with_tasks=True)
    alt_config = _make_config_file(tmp_path, "alt.toml", "perplexity")
    monkeypatch.setenv("SUPERFORECASTER_CONFIG", str(main_config))

    # Record the retrieval settings each _build_search_clients call is given, and
    # hand back trivial fake clients (never actually searched — run_research is
    # faked below).
    seen_providers: list[str] = []

    def fake_build_search_clients(settings: RetrievalSettings, data_root: Path) -> tuple[Any, Any]:
        seen_providers.append(settings.news_provider)
        return object(), object()

    monkeypatch.setattr(cli, "_build_search_clients", fake_build_search_clients)
    monkeypatch.setattr(cli, "_build_extractor", lambda settings: object())
    monkeypatch.setattr(cli, "_build_research_llm", lambda: object())
    monkeypatch.setattr(cli, "build_llm_client", lambda config: object())
    monkeypatch.setattr(cli, "_require_llm_cli", lambda *a, **k: None)

    # run_research returns a bundle tagged by the settings' news provider so we can
    # tell the two bundles apart in the assertions.
    def fake_run_research(
        question: Any,
        llm: Any,
        news: Any,
        general: Any,
        extractor: Any,
        settings: RetrievalSettings,
        *,
        now: Any = None,
    ) -> ResearchBundle:
        domain = (
            "federalreserve.gov" if settings.news_provider == "asknews" else "perplexity.example"
        )
        return _bundle(domain)

    monkeypatch.setattr(cli, "run_research", fake_run_research)

    judge = _FakeJudge(_PAIRWISE_TEXT)
    monkeypatch.setattr(cli, "_build_bundle_judge_llm", lambda: judge)

    result = runner.invoke(
        cli.app,
        [
            "compare-research",
            "--post-id",
            "900",
            "--alt-config",
            str(alt_config),
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    # The two bundles were built under two DIFFERENT retrieval configs.
    assert seen_providers == ["asknews", "perplexity"]
    assert "pairwise" in result.output

    store = BundleEvalStore(data_dir.parent / "bundle-evals")
    records = list(store.iter_all())
    assert len(records) == 2
    kinds = {r.kind for r in records}
    assert kinds == {"pairwise"}
    # Record A (current settings) carries the verdict; both retrieval configs stored.
    with_verdict = [r for r in records if r.pairwise is not None]
    assert len(with_verdict) == 1
    assert with_verdict[0].retrieval.news_provider == "asknews"
    assert with_verdict[0].pairwise is not None
    assert with_verdict[0].pairwise.other_retrieval.news_provider == "perplexity"
    providers = {r.retrieval.news_provider for r in records}
    assert providers == {"asknews", "perplexity"}
