"""CLI smoke tests using a fake client injected through the _build_client seam."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from superforecaster import cli
from superforecaster.extract import TIER_HTTPX_TRAFILATURA, ExtractedDoc
from superforecaster.llm import LlmError, LlmResult
from superforecaster.metaculus import MetaculusClient
from superforecaster.research import ResearchBundle
from superforecaster.search import SearchResult

from . import fixtures

runner = CliRunner()

# Anchor for "open, soon-closing" fixture questions. Computed relative to the
# test run's clock so these snapshots never age out of the CLI's
# now <= scheduled_close_time <= now + closing_within_days window (cli.py's
# _select_questions). Fixed absolute dates used to rot once the calendar
# caught up to them (that's what broke this file — see 2026-08-06 fix).
_NOW = datetime.now(UTC)


class FakeClient:
    """Stands in for MetaculusClient; serves fixture posts, no network."""

    def __init__(self, posts: list[dict[str, Any]]) -> None:
        self._posts = posts
        self.closed = False

    def iter_open_binary_posts(
        self, page_size: int = 100, max_posts: int | None = None
    ) -> list[dict[str, Any]]:
        return self._posts if max_posts is None else self._posts[:max_posts]

    def get_post(self, post_id: int) -> dict[str, Any]:
        for post in self._posts:
            if post["id"] == post_id:
                return post
        raise KeyError(post_id)

    def close(self) -> None:
        self.closed = True


def _install_fake(monkeypatch: pytest.MonkeyPatch, posts: list[dict[str, Any]]) -> FakeClient:
    fake = FakeClient(posts)
    monkeypatch.setattr(cli, "_build_client", lambda token: fake)
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")
    return fake


def test_pyramid_cached_run_keeps_verification_and_topup_credits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a CACHED bundle only the bundle's credits are zeroed; fresh verification
    and leaf-top-up searches (drawn every run) must still be counted (spec §5)."""
    from datetime import UTC, datetime

    from superforecaster.config import PyramidSettings
    from superforecaster.forecaster import ForecastRecord, PyramidDetail
    from superforecaster.pyramid import PyramidNode

    now = datetime(2026, 7, 12, tzinfo=UTC)
    tree = PyramidNode(id="root", question="Q?", node_type="leaf", grounding="evidence")
    detail = PyramidDetail(
        tree=tree,
        prune_notes=[],
        leaf_estimates=[],
        combination_steps=[],
        structural_probability=0.5,
        holistic_probability=0.5,
        holistic_blend_weight=0.5,
        weakest_leaf_confidence="low",
        structural_holistic_gap=0.0,
        decompose_failed=False,
        assumption_search_credits=2,  # fresh verification draw
        leaf_topup_search_credits=1,  # fresh top-up draw
        notes=[],
    )
    record = ForecastRecord(
        created_at=now,
        question_id=201,
        post_id=101,
        title="Q?",
        probability=0.5,
        raw_probability=0.5,
        confidence="low",
        rationale="r",
        reference_class="",
        model="pyramid-c",
        input_tokens=0,
        output_tokens=0,
        cost_usd=None,
        method="pyramid-c",
        research_search_credits=3 + 2 + 1,  # bundle 3 + verify 2 + top-up 1
        pyramid=detail,
    )
    monkeypatch.setattr(cli, "run_pyramid_forecast", lambda *a, **k: record)

    to_store: list[ForecastRecord] = []
    out = cli._forecast_pyramid_question(
        _minimal_snapshot(now),
        PyramidSettings(),
        None,
        cast(Any, lambda cfg: None),
        None,  # bundle
        "cached",
        search_client=None,
        now=now,
        to_store=to_store,
        failures=[],
    )
    assert out is not None
    # Bundle's 3 credits dropped; the 2 verification + 1 top-up credits survive.
    assert out.research_search_credits == 3
    assert to_store[0].research_search_credits == 3


def _minimal_snapshot(now: Any) -> Any:
    from superforecaster.models import QuestionSnapshot

    return QuestionSnapshot(
        scraped_at=now,
        post_id=101,
        question_id=201,
        title="Q?",
        description="",
        resolution_criteria="",
        fine_print="",
        scheduled_close_time=now,
        status="open",
    )


def test_scrape_writes_and_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    posts = [
        fixtures.open_binary_post(101, 201),
        fixtures.open_binary_post_no_cp(102, 202),
        fixtures.group_post(103),  # skipped
    ]
    fake = _install_fake(monkeypatch, posts)

    result = runner.invoke(cli.app, ["scrape", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "fetched=3" in result.output
    assert "written=2" in result.output
    assert "skipped=1" in result.output
    assert fake.closed


def test_scrape_without_token_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)  # ensure no stray .env is picked up
    result = runner.invoke(cli.app, ["scrape", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "METACULUS_TOKEN" in result.output


def test_update_resolutions_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Seed an open snapshot, then have the fake return the resolved version.
    open_post = fixtures.open_binary_post(104, 204)
    _install_fake(monkeypatch, [open_post])
    runner.invoke(cli.app, ["scrape", "--data-dir", str(tmp_path)])

    resolved = fixtures.resolved_yes_post(104, 204)
    _install_fake(monkeypatch, [resolved])
    result = runner.invoke(cli.app, ["update-resolutions", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "newly_resolved=1" in result.output


def test_stats_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake(monkeypatch, [fixtures.open_binary_post(101, 201)])
    runner.invoke(cli.app, ["scrape", "--data-dir", str(tmp_path)])
    result = runner.invoke(cli.app, ["stats", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "total records:      1" in result.output
    assert "unique questions:   1" in result.output


def test_stats_shows_asknews_credits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json
    from datetime import date

    snap_dir = tmp_path / "snapshots"
    _install_fake(monkeypatch, [fixtures.open_binary_post(101, 201)])
    runner.invoke(cli.app, ["scrape", "--data-dir", str(snap_dir)])

    # The counter lives in the data root (parent of the snapshot dir).
    month = date.today().strftime("%Y-%m")
    (tmp_path / "asknews_quota.json").write_text(
        json.dumps({"month": month, "credits_used": 17}), encoding="utf-8"
    )

    result = runner.invoke(cli.app, ["stats", "--data-dir", str(snap_dir)])
    assert result.exit_code == 0, result.output
    assert "asknews credits:    17/900" in result.output


def test_build_client_returns_real_client() -> None:
    """The seam builds a real client when not monkeypatched (constructor smoke)."""
    client = cli._build_client("tok")
    assert isinstance(client, MetaculusClient)
    client.close()


_FORECAST_JSON = (
    "Reasoning about the reference class and adjustments.\n"
    '```json\n{"probability": 0.42, "confidence": "medium", '
    '"rationale": "reference class then adjust", '
    '"reference_class": "comparable questions"}\n```'
)


class _FakeLlm:
    """Stands in for the LLM client; returns one fixed forecast envelope."""

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        return LlmResult(
            text=_FORECAST_JSON,
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.02,
            stop_reason="end_turn",
            model=model,
        )


def _pass_cli_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the provider CLI check succeed without a real 'claude' binary."""
    monkeypatch.setattr("superforecaster.llm.check_claude_cli", lambda path="claude": True)


def _seed_open_question(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Scrape one open, soon-closing question into the snapshot store."""
    post = fixtures.open_binary_post(101, 201)
    post["question"]["scheduled_close_time"] = (_NOW + timedelta(days=5)).isoformat()
    _install_fake(monkeypatch, [post])
    runner.invoke(cli.app, ["scrape", "--data-dir", str(tmp_path)])


class _FakeQueryLlm:
    """Stands in for the query-gen LLM; returns a fixed queries JSON object."""

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        return LlmResult(
            text='{"queries": ["rate cut news", "fomc statement"]}',
            input_tokens=80,
            output_tokens=20,
            cost_usd=0.0005,
            stop_reason="end_turn",
            model=model,
        )


class _FakeSearch:
    """A fake search client returning one fixed result; meters credits."""

    def __init__(self, url: str, *, provider: str, credits: int = 0) -> None:
        self._url = url
        self._provider = provider
        self._credits = credits
        self.used = 0

    def search(self, query: str, *, max_results: int, **_: object) -> list[SearchResult]:
        self.used += self._credits
        return [
            SearchResult(
                url=self._url,
                title=f"Result for {query}",
                snippet="short",
                published_at=None,
                source_domain="example.com",
                provider=self._provider,
                raw={},
            )
        ]

    def credits_used_this_month(self) -> int:
        return self.used


class _FakeExtractor:
    def extract(self, url: str) -> ExtractedDoc:
        return ExtractedDoc(
            url=url,
            text="Extracted body about the FOMC decision and rate path.",
            title="Extracted",
            published_at=None,
            tier=TIER_HTTPX_TRAFILATURA,
            ok=True,
        )


def _install_research_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    news = _FakeSearch("https://news.example/a", provider="asknews", credits=2)
    general = _FakeSearch("https://web.example/b", provider="tavily")
    monkeypatch.setattr(cli, "_build_research_llm", lambda: _FakeQueryLlm())
    monkeypatch.setattr(cli, "_build_search_clients", lambda settings, data_dir: (news, general))
    monkeypatch.setattr(cli, "_build_extractor", lambda settings: _FakeExtractor())


def test_forecast_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _seed_open_question(monkeypatch, snap_dir)

    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FakeLlm())
    _pass_cli_check(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--no-research",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "p=0.42" in result.output
    assert "confidence=medium" in result.output
    assert "n=1" in result.output
    assert "blind" in result.output

    from superforecaster.forecast_store import ForecastStore

    stored = list(ForecastStore(fc_dir).iter_all())
    assert len(stored) == 1
    assert stored[0].dry_run is True
    assert stored[0].reference_class == "comparable questions"
    assert stored[0].research is None  # --no-research: blind control arm


def test_forecast_experiment_id_stamps_single_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--experiment-id stamps the single-pass record and is echoed in the summary."""
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _seed_open_question(monkeypatch, snap_dir)

    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FakeLlm())
    _pass_cli_check(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--no-research",
            "--experiment-id",
            "001",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "experiment=001" in result.output

    from superforecaster.forecast_store import ForecastStore

    (stored,) = list(ForecastStore(fc_dir).iter_all())
    assert stored.experiment_id == "001"


def test_forecast_without_experiment_id_leaves_records_unstamped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _seed_open_question(monkeypatch, snap_dir)

    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FakeLlm())
    _pass_cli_check(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--no-research",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "experiment=" not in result.output

    from superforecaster.forecast_store import ForecastStore

    (stored,) = list(ForecastStore(fc_dir).iter_all())
    assert stored.experiment_id is None


def test_forecast_with_research_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _seed_open_question(monkeypatch, snap_dir)

    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FakeLlm())
    _install_research_fakes(monkeypatch)
    _pass_cli_check(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "forecast",  # research ON by default
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "researched" in result.output
    assert "search_credits=4" in result.output  # 2 queries x 2 credits/news call
    assert "sources=2" in result.output

    from superforecaster.forecast_store import ForecastStore

    (stored,) = list(ForecastStore(fc_dir).iter_all())
    assert stored.research is not None
    assert stored.research.queries == ["rate cut news", "fomc statement"]
    assert stored.research_search_credits == 4
    assert {r.url for r in stored.research.results} == {
        "https://news.example/a",
        "https://web.example/b",
    }


def _fresh_bundle() -> ResearchBundle:
    from datetime import UTC, datetime

    return ResearchBundle(
        queries=["cached query"],
        results=[
            SearchResult(
                url="https://cached.example/a",
                title="Cached",
                snippet="short",
                published_at=None,
                source_domain="cached.example",
                provider="asknews",
                raw={},
            )
        ],
        docs=[],
        tier_stats={},
        search_credits_used=0,
        retrieved_at=datetime.now(UTC),
    )


def test_forecast_reuses_cached_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _seed_open_question(monkeypatch, snap_dir)

    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FakeLlm())
    _install_research_fakes(monkeypatch)
    _pass_cli_check(monkeypatch)

    # Pre-seed a fresh cached bundle for question 201 (the fixture's question id).
    from superforecaster.research_cache import ResearchCache

    ResearchCache(tmp_path / "research").save(201, _fresh_bundle())

    # run_research must NOT be called when a fresh bundle is on disk.
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("run_research called despite a fresh cached bundle")

    monkeypatch.setattr(cli, "run_research", _boom)

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "research=cached" in result.output

    from superforecaster.forecast_store import ForecastStore

    (stored,) = list(ForecastStore(fc_dir).iter_all())
    assert stored.research is not None
    assert stored.research.queries == ["cached query"]


def test_forecast_setting_zero_bypasses_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _seed_open_question(monkeypatch, snap_dir)

    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FakeLlm())
    _install_research_fakes(monkeypatch)
    _pass_cli_check(monkeypatch)

    # A fresh bundle is on disk, but reuse is disabled -> must re-research.
    from superforecaster.research_cache import ResearchCache

    ResearchCache(tmp_path / "research").save(201, _fresh_bundle())

    from superforecaster.config import RetrievalSettings

    monkeypatch.setattr(
        cli,
        "load_retrieval_settings",
        lambda: RetrievalSettings(reuse_research_max_age_hours=0),
    )

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "research=fresh" in result.output

    from superforecaster.forecast_store import ForecastStore

    (stored,) = list(ForecastStore(fc_dir).iter_all())
    assert stored.research is not None
    # The freshly-researched bundle (from the fakes), not the cached one.
    assert stored.research.queries == ["rate cut news", "fomc statement"]


def test_forecast_saves_bundle_before_failing_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _seed_open_question(monkeypatch, snap_dir)

    from superforecaster.llm import LlmError

    class _FailingLlm:
        def run(self, *args: object, **kwargs: object) -> LlmResult:
            raise LlmError("forecast blew up")

    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FailingLlm())
    _install_research_fakes(monkeypatch)
    _pass_cli_check(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    # The pass failed, so exit non-zero and no record stored...
    assert result.exit_code == 1
    assert "failed=1" in result.output

    from superforecaster.forecast_store import ForecastStore

    assert list(ForecastStore(fc_dir).iter_all()) == []
    # ...but the paid-for research bundle survived on disk.
    from superforecaster.research_cache import ResearchCache

    saved = ResearchCache(tmp_path / "research").load(201, max_age=timedelta(hours=6))
    assert saved is not None
    assert saved.queries == ["rate cut news", "fomc statement"]


def test_forecast_one_failure_still_stores_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Two open, soon-closing questions with distinct close times for stable order.
    post_a = fixtures.open_binary_post(101, 201)
    post_a["question"]["scheduled_close_time"] = (_NOW + timedelta(days=1)).isoformat()
    post_b = fixtures.open_binary_post(102, 202)
    post_b["question"]["scheduled_close_time"] = (_NOW + timedelta(days=2)).isoformat()
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _install_fake(monkeypatch, [post_a, post_b])
    runner.invoke(cli.app, ["scrape", "--data-dir", str(snap_dir)])

    from superforecaster.llm import LlmError

    class _FirstFailsLlm:
        def __init__(self) -> None:
            self.calls = 0

        def run(
            self,
            system: str,
            prompt: str,
            model: str,
            max_tokens: int | None = None,
            json_schema: dict[str, object] | None = None,
            effort: str | None = None,
        ) -> LlmResult:
            self.calls += 1
            if self.calls == 1:
                raise LlmError("first question blew up")
            return _FakeLlm().run(system, prompt, model, max_tokens, json_schema, effort)

    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FirstFailsLlm())
    _pass_cli_check(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--no-research",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 1  # a question failed
    assert "failed=1" in result.output
    assert "n=1" in result.output

    from superforecaster.forecast_store import ForecastStore

    stored = list(ForecastStore(fc_dir).iter_all())
    # The second question still forecast and was stored despite the first failing.
    assert len(stored) == 1
    assert stored[0].question_id == 202


def test_forecast_without_cli_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Pin the task config to claude-cli so this test does not depend on which
    # provider the real settings.toml currently selects (the control panel is
    # the owner's to change; tests must not break when it does).
    config = tmp_path / "settings.toml"
    config.write_text(
        '[tasks.forecast_pass]\nprovider = "claude-cli"\nmodel = "m"\nmax_tokens = 100\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPERFORECASTER_CONFIG", str(config))
    monkeypatch.setattr("superforecaster.llm.check_claude_cli", lambda path="claude": False)
    monkeypatch.chdir(tmp_path)  # avoid picking up a stray .env
    result = runner.invoke(cli.app, ["forecast", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "Claude Code" in result.output


def test_forecast_openrouter_without_key_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "settings.toml"
    config.write_text(
        '[tasks.forecast_pass]\nprovider = "openrouter"\nmodel = "m"\nmax_tokens = 100\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPERFORECASTER_CONFIG", str(config))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # avoid picking up a stray .env
    result = runner.invoke(cli.app, ["forecast", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY" in result.output


def test_forecast_no_matching_questions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_build_llm_client", lambda: _FakeLlm())
    _pass_cli_check(monkeypatch)
    result = runner.invoke(cli.app, ["forecast", "--data-dir", str(tmp_path / "empty")])
    assert result.exit_code == 0, result.output
    assert "no matching open questions" in result.output


class _FakeMemberLlm:
    """Ensemble-member fake: returns a fixed probability, or raises to fail."""

    def __init__(self, probability: float | None = None, *, fail: bool = False) -> None:
        self._probability = probability
        self._fail = fail

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        if self._fail:
            raise LlmError("member boom")
        assert self._probability is not None
        return LlmResult(
            text=(
                "Reasoning.\n```json\n"
                f'{{"probability": {self._probability}, "confidence": "medium", '
                '"rationale": "base rate then adjust, briefly", "reference_class": "c"}\n```'
            ),
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            stop_reason="end_turn",
            model=model,
        )


def _member_factory(*fakes: _FakeMemberLlm) -> Any:
    """A client_factory that hands out the given member fakes in order."""
    it = iter(fakes)

    def factory(config: Any) -> _FakeMemberLlm:
        return next(it)

    return factory


_THREE_CLAUDE_MEMBERS = (
    '[[ensemble.members]]\nframe = "outside-first"\nprovider = "claude-cli"\n'
    'model = "m0"\nmax_tokens = 100\n\n'
    '[[ensemble.members]]\nframe = "inside-first"\nprovider = "claude-cli"\n'
    'model = "m1"\nmax_tokens = 100\n\n'
    '[[ensemble.members]]\nframe = "status-quo"\nprovider = "claude-cli"\n'
    'model = "m2"\nmax_tokens = 100\n'
)


def _pin_ensemble_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, members_toml: str
) -> None:
    """Pin SUPERFORECASTER_CONFIG to an [ensemble] roster so tests don't depend on
    the real settings.toml (which the owner may retune at will)."""
    config = tmp_path / "settings.toml"
    config.write_text(f"[ensemble]\n{members_toml}", encoding="utf-8")
    monkeypatch.setenv("SUPERFORECASTER_CONFIG", str(config))


def test_forecast_ensemble_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _pin_ensemble_config(monkeypatch, tmp_path, _THREE_CLAUDE_MEMBERS)
    _seed_open_question(monkeypatch, snap_dir)

    monkeypatch.setattr(
        cli,
        "_build_member_client_factory",
        lambda: _member_factory(_FakeMemberLlm(0.4), _FakeMemberLlm(0.5), _FakeMemberLlm(0.6)),
    )
    _pass_cli_check(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--ensemble",
            "--no-research",
            "--experiment-id",
            "001",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 0, result.output
    # Log-odds of 0.4/0.5/0.6 average to 0 -> pooled 0.5 (fixed coeff 1.0).
    assert "p=0.50" in result.output
    assert "experiment=001" in result.output
    assert "members=3ok/0failed" in result.output
    assert "ensemble" in result.output
    assert "n=1" in result.output

    from superforecaster.forecast_store import ForecastStore

    stored = list(ForecastStore(fc_dir).iter_all())
    members = [r for r in stored if r.method == "ensemble-b-member"]
    pooled = [r for r in stored if r.method == "ensemble-b"]
    assert len(members) == 3
    assert len(pooled) == 1
    assert {m.frame for m in members} == {"outside-first", "inside-first", "status-quo"}
    # --experiment-id stamps the pooled record AND every member record.
    assert all(r.experiment_id == "001" for r in stored)
    (p,) = pooled
    assert p.model == "ensemble-b"
    assert p.probability == pytest.approx(0.5)
    assert p.ensemble is not None
    assert p.ensemble.member_probabilities == [0.4, 0.5, 0.6]
    # Pooled cost sums the three members ($0.01 each); members are stored too.
    assert p.cost_usd == pytest.approx(0.03)


def test_forecast_ensemble_tolerates_member_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _pin_ensemble_config(monkeypatch, tmp_path, _THREE_CLAUDE_MEMBERS)
    _seed_open_question(monkeypatch, snap_dir)

    # Middle member fails; the pool forms from the two survivors.
    monkeypatch.setattr(
        cli,
        "_build_member_client_factory",
        lambda: _member_factory(
            _FakeMemberLlm(0.4), _FakeMemberLlm(fail=True), _FakeMemberLlm(0.6)
        ),
    )
    _pass_cli_check(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--ensemble",
            "--no-research",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "members=2ok/1failed" in result.output
    assert "n=1" in result.output

    from superforecaster.forecast_store import ForecastStore

    stored = list(ForecastStore(fc_dir).iter_all())
    members = [r for r in stored if r.method == "ensemble-b-member"]
    (pooled,) = [r for r in stored if r.method == "ensemble-b"]
    assert len(members) == 2
    assert pooled.ensemble is not None
    assert pooled.ensemble.member_probabilities == [0.4, 0.6]
    assert len(pooled.ensemble.member_failures) == 1


def test_forecast_ensemble_hard_failure_is_isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Two open questions with distinct close times for a stable order.
    post_a = fixtures.open_binary_post(101, 201)
    post_a["question"]["scheduled_close_time"] = (_NOW + timedelta(days=1)).isoformat()
    post_b = fixtures.open_binary_post(102, 202)
    post_b["question"]["scheduled_close_time"] = (_NOW + timedelta(days=2)).isoformat()
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _pin_ensemble_config(monkeypatch, tmp_path, _THREE_CLAUDE_MEMBERS)
    _install_fake(monkeypatch, [post_a, post_b])
    runner.invoke(cli.app, ["scrape", "--data-dir", str(snap_dir)])

    # Q1: two of three members fail -> only one survivor -> ensemble raises.
    # Q2: all three succeed. One factory serves both questions, in order.
    monkeypatch.setattr(
        cli,
        "_build_member_client_factory",
        lambda: _member_factory(
            _FakeMemberLlm(0.5),
            _FakeMemberLlm(fail=True),
            _FakeMemberLlm(fail=True),
            _FakeMemberLlm(0.4),
            _FakeMemberLlm(0.5),
            _FakeMemberLlm(0.6),
        ),
    )
    _pass_cli_check(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--ensemble",
            "--no-research",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 1  # a question's ensemble failed
    assert "failed=1" in result.output
    assert "n=1" in result.output

    from superforecaster.forecast_store import ForecastStore

    stored = list(ForecastStore(fc_dir).iter_all())
    # Q1 stored nothing; Q2 stored its 3 members + 1 pooled, all for question 202.
    assert {r.question_id for r in stored} == {202}
    assert len([r for r in stored if r.method == "ensemble-b-member"]) == 3
    assert len([r for r in stored if r.method == "ensemble-b"]) == 1


def test_forecast_ensemble_checks_each_distinct_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Roster spans claude-cli (available) and openrouter (key missing): the
    # readiness check must catch the openrouter member and exit with its help.
    members = (
        '[[ensemble.members]]\nframe = "outside-first"\nprovider = "claude-cli"\n'
        'model = "m0"\nmax_tokens = 100\n\n'
        '[[ensemble.members]]\nframe = "inside-first"\nprovider = "openrouter"\n'
        'model = "m1"\nmax_tokens = 100\n'
    )
    _pin_ensemble_config(monkeypatch, tmp_path, members)
    monkeypatch.setattr("superforecaster.llm.check_claude_cli", lambda path="claude": True)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # avoid picking up a stray .env

    result = runner.invoke(cli.app, ["forecast", "--ensemble", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY" in result.output


_PYRAMID_TASKS_TOML = (
    '[tasks.decompose]\nprovider = "claude-cli"\nmodel = "d"\nmax_tokens = 100\n\n'
    '[tasks.forecast_pass]\nprovider = "claude-cli"\nmodel = "leaf"\nmax_tokens = 100\n\n'
    '[tasks.holistic]\nprovider = "claude-cli"\nmodel = "hol"\nmax_tokens = 100\n'
)

_PYRAMID_TREE_JSON = (
    "Decomposition.\n```json\n"
    '{"id": "root", "question": "Root?", "node_type": "conjunction", "children": ['
    '{"id": "a", "question": "Step A?", "node_type": "leaf", "grounding": "evidence"}, '
    '{"id": "b", "question": "Base rate?", "node_type": "leaf", "grounding": "model-knowledge"}]}'
    "\n```"
)


class _FakeTaskLlm:
    """Serves scripted replies in order; used to script a pyramid run's tasks."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)

    def run(
        self,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        json_schema: dict[str, object] | None = None,
        effort: str | None = None,
    ) -> LlmResult:
        text = self._replies.pop(0)
        return LlmResult(
            text=text,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            stop_reason="end_turn",
            model=model,
        )


def _pin_pyramid_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "settings.toml"
    config.write_text(_PYRAMID_TASKS_TOML, encoding="utf-8")
    monkeypatch.setenv("SUPERFORECASTER_CONFIG", str(config))


def test_forecast_pyramid_and_ensemble_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = runner.invoke(
        cli.app,
        ["forecast", "--pyramid", "--ensemble", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_forecast_pyramid_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snap_dir = tmp_path / "snapshots"
    fc_dir = tmp_path / "forecasts"
    _pin_pyramid_config(monkeypatch, tmp_path)
    _seed_open_question(monkeypatch, snap_dir)

    leaf_json = (
        "Reasoning.\n```json\n"
        '{"probability": 0.5, "confidence": "medium", '
        '"rationale": "base rate then adjust, briefly", '
        '"reference_class": "c"}\n```'
    )
    clients = {
        "d": _FakeTaskLlm([_PYRAMID_TREE_JSON]),
        "leaf": _FakeTaskLlm([leaf_json, leaf_json]),  # two leaves
        "hol": _FakeTaskLlm([leaf_json]),
    }

    def _factory() -> Any:
        def factory(config: Any) -> _FakeTaskLlm:
            return clients[config.model]

        return factory

    monkeypatch.setattr(cli, "_build_pyramid_client_factory", _factory)
    _pass_cli_check(monkeypatch)

    result = runner.invoke(
        cli.app,
        [
            "forecast",
            "--pyramid",
            "--no-research",
            "--experiment-id",
            "002",
            "--data-dir",
            str(snap_dir),
            "--forecast-dir",
            str(fc_dir),
            "--closing-within-days",
            "365",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "pyramid" in result.output
    assert "experiment=002" in result.output
    assert "structural=" in result.output
    assert "holistic=" in result.output
    assert "leaves=2" in result.output
    assert "n=1" in result.output

    from superforecaster.forecast_store import ForecastStore

    stored = list(ForecastStore(fc_dir).iter_all())
    assert len(stored) == 1  # ONE record per pyramid question
    (rec,) = stored
    assert rec.method == "pyramid-c"
    assert rec.experiment_id == "002"
    assert rec.pyramid is not None
    assert len(rec.pyramid.leaf_estimates) == 2
    # Decompose + 2 leaves + holistic = 4 calls x $0.01.
    assert rec.cost_usd == pytest.approx(0.04)


def test_render_forecast_command_writes_html(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from datetime import UTC, datetime

    from superforecaster.forecast_store import ForecastStore
    from superforecaster.forecaster import ForecastRecord

    fc_dir = tmp_path / "forecasts"
    record = ForecastRecord(
        created_at=datetime(2026, 7, 12, tzinfo=UTC),
        question_id=555,
        post_id=666,
        title="Will the CLI render this?",
        probability=0.42,
        raw_probability=0.42,
        confidence="medium",
        rationale="A checklist rationale.",
        reference_class="rendered questions",
        model="z-ai/glm-5.2",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.01,
        method="checklist-a",
    )
    ForecastStore(fc_dir).append([record])

    out = tmp_path / "out.html"
    result = runner.invoke(
        cli.app,
        [
            "render-forecast",
            "555",
            "--forecast-dir",
            str(fc_dir),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert str(out) in result.output
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "Will the CLI render this?" in html
    assert "42%" in html


def test_render_forecast_missing_question_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = runner.invoke(
        cli.app,
        ["render-forecast", "12345", "--forecast-dir", str(tmp_path / "empty")],
    )
    assert result.exit_code == 1
    assert "no stored forecast" in result.output


def test_data_dir_comes_from_settings_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no --data-dir flag or env, the snapshot dir comes from settings.toml."""
    snap_dir = tmp_path / "from-settings"
    settings = tmp_path / "settings.toml"
    settings.write_text(f'[data]\nsnapshot_dir = "{snap_dir.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("SUPERFORECASTER_CONFIG", str(settings))
    monkeypatch.delenv("SUPERFORECASTER_DATA_DIR", raising=False)
    _install_fake(monkeypatch, [fixtures.open_binary_post(101, 201)])

    result = runner.invoke(cli.app, ["scrape"])  # no --data-dir -> settings dir
    assert result.exit_code == 0, result.output
    assert "written=1" in result.output

    stats = runner.invoke(cli.app, ["stats"])
    assert str(snap_dir) in stats.output
