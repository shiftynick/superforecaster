"""Command-line entry point for scraping and inspecting Metaculus snapshots.

Run these from the repository root (``N:/superforecaster``) so the default data
directory (``./data/snapshots``) lands in the right place. Commands:

- ``scrape`` — fetch open binary questions and save a snapshot of each.
- ``update-resolutions`` — re-check questions we already track to capture their
  outcomes once they resolve.
- ``stats`` — print a summary of what's stored.

The Metaculus API token comes from the ``METACULUS_TOKEN`` environment variable
(an ``engine/.env`` file with ``KEY=VALUE`` lines is also read, if present).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import typer

from .config import (
    EnsembleSettings,
    PyramidSettings,
    RetrievalSettings,
    TaskModelConfig,
    load_ensemble_settings,
    load_pyramid_settings,
    load_retrieval_settings,
    load_settings,
    load_task_config,
)
from .dotenv import load_dotenv as _load_dotenv
from .ensemble import EnsembleResult, pooled_forecast_record, run_ensemble
from .experiments import (
    ExperimentManifest,
    ExperimentStatus,
    compute_status,
    load_manifests,
    resolutions_from_store,
)
from .extract import Extractor, build_extractor
from .forecast_store import ForecastStore
from .forecaster import ForecastRecord, run_forecast_pass
from .llm import LlmClient, LlmError, build_llm_client, check_provider_cli
from .metaculus import MetaculusClient
from .models import QuestionSnapshot, SkipPost
from .pyramid_runner import run_pyramid_forecast
from .render import render_forecast_html
from .research import ResearchBundle, run_research
from .research_cache import ResearchCache
from .search import SearchClient, build_search_client, read_asknews_credits_used
from .store import SnapshotStore
from .submission import SubmissionClient, SubmissionError

app = typer.Typer(
    add_completion=False,
    help="Scrape and inspect Metaculus question snapshots. Run from the repo root.",
)

_BOT_TOKEN_ENV = "METACULUS_BOT_TOKEN"

# One-line disclosure prefixed to every rationale comment, so a reader can see at
# a glance which bot and method produced the forecast.
_DISCLOSURE = "[bot: candidate-A baseline, no retrieval]"

_BOT_TOKEN_HELP = (
    "No submission token found. A live tournament submission uses METACULUS_BOT_TOKEN "
    "if set, otherwise falls back to METACULUS_TOKEN (the same token can serve both if "
    "it belongs to the bot account). Set one, e.g.\n"
    "  PowerShell:  $env:METACULUS_BOT_TOKEN = '<token>'\n"
    "  bash:        export METACULUS_BOT_TOKEN=<token>\n"
    "or add a line METACULUS_BOT_TOKEN=<token> (or METACULUS_TOKEN=<token>) to engine/.env."
)
_TOKEN_HELP = (
    "Metaculus API token is missing. Create a free account at metaculus.com, open "
    "Settings > API, copy your token, then set it, e.g.\n"
    "  PowerShell:  $env:METACULUS_TOKEN = '<token>'\n"
    "  bash:        export METACULUS_TOKEN=<token>\n"
    "or add a line METACULUS_TOKEN=<token> to engine/.env."
)
_CLAUDE_CLI_HELP = (
    "The Claude Code CLI ('claude') was not found on your PATH. Forecast passes "
    "run through it on the owner's Claude subscription (no API key needed). "
    "Install Claude Code (https://claude.com/claude-code) and log in with "
    "'claude login', then re-run. Verify with:  claude --version"
)
_CODEX_CLI_HELP = (
    "The Codex CLI ('codex') was not found on your PATH. Some passes run through "
    "it on the owner's Codex subscription. Install the Codex CLI and log in with "
    "your Codex subscription, then re-run. Verify with:  codex --version"
)
_OPENROUTER_KEY_HELP = (
    "OPENROUTER_API_KEY is not set. The 'openrouter' provider needs it — "
    "add OPENROUTER_API_KEY=<key> to engine/.env."
)


def _resolve_data_dir(data_dir: Path | None) -> Path:
    """Pick the snapshot directory by precedence.

    Explicit ``--data-dir`` flag, then the ``SUPERFORECASTER_DATA_DIR`` env
    override, then ``settings.toml`` ([data] snapshot_dir), then its built-in
    fallback (all handled inside :func:`load_settings`).
    """
    if data_dir is not None:
        return data_dir
    env = os.environ.get("SUPERFORECASTER_DATA_DIR")
    if env:
        return Path(env)
    return load_settings().data.snapshot_dir


def _require_token() -> str:
    """Return the Metaculus token or exit with clear setup instructions."""
    _load_dotenv(Path("engine/.env"))
    _load_dotenv(Path(".env"))
    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        typer.echo(_TOKEN_HELP, err=True)
        raise typer.Exit(code=1)
    return token


# Seam for tests: monkeypatched to return a fake client.
def _build_client(token: str) -> MetaculusClient:
    """Construct the real Metaculus client. Tests replace this to inject a fake."""
    return MetaculusClient(token=token)


def _provider_unavailable_help(provider: str) -> str:
    """The setup-help message for a provider that is configured but not ready."""
    if provider == "openrouter":
        return _OPENROUTER_KEY_HELP
    if provider == "codex-cli":
        return _CODEX_CLI_HELP
    return _CLAUDE_CLI_HELP


def _require_llm_cli(configs: list[TaskModelConfig] | None = None) -> None:
    """Ensure every configured provider is ready to use, or exit with help.

    Provider-aware: for CLI providers this checks the binary is on PATH; for
    OpenRouter it checks the API key is set (so ``.env`` is loaded first — the
    readiness check itself only reads ``os.environ``). An unknown provider name
    in config surfaces here as a clear error rather than deep in a forecast.

    ``configs`` defaults to the single ``forecast_pass`` task (the baseline pass).
    Ensemble mode passes the members' configs so each DISTINCT provider among them
    is checked — the roster can span claude-cli, codex-cli, and openrouter, and
    any one being unavailable exits with that provider's help.
    """
    _load_dotenv(Path("engine/.env"))
    _load_dotenv(Path(".env"))
    checks = configs if configs is not None else [load_task_config("forecast_pass")]
    seen: set[str] = set()
    for config in checks:
        if config.provider in seen:
            continue
        seen.add(config.provider)
        try:
            available = check_provider_cli(config)
        except LlmError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        if not available:
            typer.echo(_provider_unavailable_help(config.provider), err=True)
            raise typer.Exit(code=1)


# Seam for tests: monkeypatched to inject a fake LLM client.
def _build_llm_client() -> LlmClient | None:
    """Return ``None`` so the forecaster builds the provider's client itself.

    Tests monkeypatch this to inject a fake client with a ``.run(...)`` method,
    exactly as ``_build_client`` does for the Metaculus side.
    """
    return None


# Seam for tests: monkeypatched to inject a fake ensemble member factory.
def _build_member_client_factory() -> Callable[[TaskModelConfig], LlmClient]:
    """Return the factory the ensemble uses to build each member's LLM client.

    Defaults to :func:`build_llm_client` (dispatches on the member's provider).
    Tests monkeypatch this to inject a fake factory that hands out canned clients
    per member, mirroring :func:`_build_llm_client` for the single pass.
    """
    return build_llm_client


# Seam for tests: monkeypatched to inject a fake pyramid client factory.
def _build_pyramid_client_factory() -> Callable[[TaskModelConfig], LlmClient]:
    """Return the factory the pyramid runner uses to build each task's LLM client.

    Defaults to :func:`build_llm_client` (dispatches on each task's provider, so
    the decompose / leaf / holistic tasks can each name their own model). Tests
    monkeypatch this to hand out canned clients keyed by the task config.
    """
    return build_llm_client


# Seam for tests: monkeypatched to inject fake search clients.
def _build_search_clients(
    settings: RetrievalSettings, data_dir: Path
) -> tuple[SearchClient, SearchClient]:
    """Build the (news, general) search clients from ``[retrieval]`` settings.

    The news client is told it fills the news role (``news_role=True``) so a
    Tavily-in-the-news-slot config still gets publish dates. Keys come from the
    environment inside ``build_search_client``. Tests monkeypatch this seam.
    """
    news = build_search_client(
        settings.news_provider, settings=settings, data_dir=data_dir, news_role=True
    )
    general = build_search_client(
        settings.general_provider, settings=settings, data_dir=data_dir, news_role=False
    )
    return news, general


# Seam for tests: monkeypatched to inject a fake extractor.
def _build_extractor(settings: RetrievalSettings) -> Extractor:
    """Build the content extractor from ``[retrieval]`` settings (test seam)."""
    return build_extractor(settings)


# Seam for tests: monkeypatched to inject a fake query-gen LLM client.
def _build_research_llm() -> LlmClient | None:
    """Return ``None`` so the forecast command builds the query-gen client itself.

    Mirrors :func:`_build_llm_client`: tests monkeypatch this to inject a fake
    that returns a queries JSON object, distinct from the forecast-pass fake.
    """
    return None


# Seam for tests: monkeypatched to return a fake submission client.
def _build_submission_client(token: str) -> SubmissionClient:
    """Construct the real submission client. Tests replace this to inject a fake."""
    return SubmissionClient(bot_token=token)


def _bot_token() -> str | None:
    """Return the bot token from the environment (or ``.env``), or ``None``."""
    _load_dotenv(Path("engine/.env"))
    _load_dotenv(Path(".env"))
    return os.environ.get(_BOT_TOKEN_ENV)


_DataDirOption = typer.Option(
    None,
    "--data-dir",
    help=(
        "Directory for snapshot files. Defaults to $SUPERFORECASTER_DATA_DIR, "
        "then ./data/snapshots (run from the repo root)."
    ),
)


@app.command()
def scrape(
    max_questions: int = typer.Option(500, "--max-questions", help="Max posts to fetch."),
    data_dir: Path | None = _DataDirOption,
) -> None:
    """Fetch open binary questions and append a fresh snapshot of each."""
    token = _require_token()
    store = SnapshotStore(_resolve_data_dir(data_dir))
    scraped_at = datetime.now(UTC)

    snapshots: list[QuestionSnapshot] = []
    fetched = 0
    skipped = 0
    client = _build_client(token)
    try:
        for post in client.iter_open_binary_posts(max_posts=max_questions):
            fetched += 1
            try:
                snapshots.append(QuestionSnapshot.from_api_post(post, scraped_at))
            except SkipPost:
                skipped += 1
    finally:
        client.close()

    written = store.append(snapshots)
    typer.echo(f"scrape: fetched={fetched} written={written} skipped={skipped}")


@app.command("update-resolutions")
def update_resolutions(
    limit: int = typer.Option(100, "--limit", help="Max unresolved questions to re-check."),
    data_dir: Path | None = _DataDirOption,
) -> None:
    """Re-fetch tracked, still-open questions to capture any new resolutions."""
    token = _require_token()
    store = SnapshotStore(_resolve_data_dir(data_dir))
    post_ids = store.unresolved_question_post_ids()[:limit]
    scraped_at = datetime.now(UTC)

    snapshots: list[QuestionSnapshot] = []
    newly_resolved = 0
    client = _build_client(token)
    try:
        for post_id in post_ids:
            post = client.get_post(post_id)
            try:
                snapshot = QuestionSnapshot.from_api_post(post, scraped_at)
            except SkipPost:
                continue
            snapshots.append(snapshot)
            if snapshot.status.lower() == "resolved":
                newly_resolved += 1
    finally:
        client.close()

    written = store.append(snapshots)
    typer.echo(
        f"update-resolutions: checked={len(post_ids)} written={written} "
        f"newly_resolved={newly_resolved}"
    )


@app.command()
def stats(data_dir: Path | None = _DataDirOption) -> None:
    """Print a human-readable summary of the stored snapshots."""
    snap_dir = _resolve_data_dir(data_dir)
    store = SnapshotStore(snap_dir)
    s = store.stats()
    typer.echo(f"data dir:           {store.dir}")
    typer.echo(f"total records:      {s.total_records}")
    typer.echo(f"unique questions:   {s.unique_questions}")
    typer.echo(f"resolved questions: {s.resolved_questions}")
    typer.echo(f"first scrape:       {s.first_scrape}")
    typer.echo(f"last scrape:        {s.last_scrape}")

    # AskNews credit draw for the current month, read from the data-root counter
    # (parent of the snapshot dir). No API key or network needed — just the file.
    retrieval = load_retrieval_settings()
    used = read_asknews_credits_used(snap_dir.parent)
    cap = retrieval.asknews_monthly_credit_cap
    typer.echo(f"asknews credits:    {used}/{cap} (this month)")


_ForecastDirOption = typer.Option(
    None,
    "--forecast-dir",
    help=(
        "Directory for forecast files. Defaults to $SUPERFORECASTER_FORECAST_DIR, "
        "then ./data/forecasts (run from the repo root)."
    ),
)


_PostIdOption = typer.Option(
    [],
    "--post-id",
    help="Forecast these exact post ids, overriding selection. Repeatable.",
)


def _resolve_forecast_dir(forecast_dir: Path | None) -> Path:
    """Pick the forecast directory by precedence.

    Explicit ``--forecast-dir`` flag, then the ``SUPERFORECASTER_FORECAST_DIR``
    env override, then ``settings.toml`` ([data] forecast_dir), then its built-in
    fallback (all handled inside :func:`load_settings`).
    """
    if forecast_dir is not None:
        return forecast_dir
    env = os.environ.get("SUPERFORECASTER_FORECAST_DIR")
    if env:
        return Path(env)
    return load_settings().data.forecast_dir


def _select_questions(
    store: SnapshotStore,
    *,
    max_questions: int,
    closing_within_days: int,
    post_ids: list[int],
    now: datetime,
) -> list[QuestionSnapshot]:
    """Choose which questions to forecast.

    If ``post_ids`` is given, those exact posts (from their latest snapshot) are
    used, in the order requested. Otherwise: open questions whose scheduled close
    time is within ``closing_within_days``, soonest-closing first, capped at
    ``max_questions``.
    """
    latest = store.latest_by_question()
    if post_ids:
        by_post = {s.post_id: s for s in latest.values()}
        return [by_post[pid] for pid in post_ids if pid in by_post]

    cutoff = now + timedelta(days=closing_within_days)
    candidates = [
        s
        for s in latest.values()
        if s.status.lower() == "open"
        and s.scheduled_close_time is not None
        and now <= s.scheduled_close_time <= cutoff
    ]
    candidates.sort(key=lambda s: cast("datetime", s.scheduled_close_time))
    return candidates[:max_questions]


@app.command()
def forecast(
    max_questions: int = typer.Option(5, "--max-questions", help="Max questions to forecast."),
    closing_within_days: int = typer.Option(
        90, "--closing-within-days", help="Only questions closing within this many days."
    ),
    no_research: bool = typer.Option(
        False,
        "--no-research",
        help="Skip retrieval and forecast blind (the baseline control arm).",
    ),
    ensemble: bool = typer.Option(
        False,
        "--ensemble",
        help=(
            "Run the diverse ensemble (candidate B): N frames/models pooled over "
            "one shared research bundle, instead of a single pass. Composes with "
            "--no-research."
        ),
    ),
    pyramid: bool = typer.Option(
        False,
        "--pyramid",
        help=(
            "Run the decomposition pyramid (candidate C): decompose the question "
            "into a tree, estimate its leaves, combine upward, blend with a "
            "holistic pass. Mutually exclusive with --ensemble. Composes with "
            "--no-research."
        ),
    ),
    data_dir: Path | None = _DataDirOption,
    forecast_dir: Path | None = _ForecastDirOption,
    post_id: list[int] = _PostIdOption,
) -> None:
    """Forecast open tournament questions, with web research on by default.

    Each question is researched (query generation -> news + general search ->
    content extraction) into a frozen bundle that is rendered into the prompt and
    stored with the forecast. ``--no-research`` forecasts blind, preserving the
    baseline for side-by-side comparison. ``--ensemble`` swaps the single pass for
    the diverse ensemble; ``--pyramid`` swaps it for the decomposition pyramid
    (the two are mutually exclusive). All forecasts are dry runs — nothing is
    submitted. Per-question LLM cost and search-credit draw are both logged.
    """
    if ensemble and pyramid:
        typer.echo(
            "forecast: --ensemble and --pyramid are mutually exclusive; pick one method.",
            err=True,
        )
        raise typer.Exit(code=1)

    ensemble_settings = load_ensemble_settings() if ensemble else None
    pyramid_settings = load_pyramid_settings() if pyramid else None
    pyramid_ensemble_settings = load_ensemble_settings() if pyramid else None
    if ensemble_settings is not None:
        _require_llm_cli([m.task_config() for m in ensemble_settings.members])
    elif pyramid_settings is not None:
        _require_llm_cli(_pyramid_task_configs())
    else:
        _require_llm_cli()
    snap_dir = _resolve_data_dir(data_dir)
    snap_store = SnapshotStore(snap_dir)
    fc_store = ForecastStore(_resolve_forecast_dir(forecast_dir))
    now = datetime.now(UTC)

    questions = _select_questions(
        snap_store,
        max_questions=max_questions,
        closing_within_days=closing_within_days,
        post_ids=post_id,
        now=now,
    )
    if not questions:
        typer.echo("forecast: no matching open questions to forecast.")
        return

    # The single pass needs the forecast-pass config + one client; the ensemble
    # and pyramid drive their own task configs via a per-task client factory.
    single_pass = not ensemble and not pyramid
    config = load_task_config("forecast_pass") if single_pass else None
    client = _build_llm_client() if single_pass else None
    member_factory = _build_member_client_factory() if ensemble else None
    pyramid_factory = _build_pyramid_client_factory() if pyramid else None

    # Research context. Built once and reused across questions; the quota counter
    # lives in the data root (the parent of the snapshot dir), config-derived so
    # there is never a hard-coded absolute path.
    research_ctx = None if no_research else _open_research_context(snap_dir)

    # Research bundles are persisted next to the quota file (same data root) so a
    # failed forecast pass never throws away paid-for search credits, and a later
    # run can reuse a fresh bundle instead of re-researching.
    cache = None if no_research else ResearchCache(snap_dir.parent / "research")

    # ``summaries`` holds one record per question (the single pass, or the pooled
    # ensemble record) so cost/credit totals are counted exactly once — a pooled
    # record's cost/credits already sum its members, so summing members again
    # would double-count. Records are PERSISTED PER QUESTION (an ensemble question
    # is minutes of paid LLM time; a crash or kill mid-batch must not lose the
    # questions already finished).
    summaries: list[ForecastRecord] = []
    failures: list[str] = []
    try:
        for snapshot in questions:
            to_store: list[ForecastRecord] = []
            bundle: ResearchBundle | None = None
            research_state = "off"
            if research_ctx is not None and cache is not None:
                bundle, research_state = _research_for_question(
                    snapshot, research_ctx, cache, now=now
                )

            if pyramid_settings is not None:
                assert pyramid_factory is not None  # set whenever pyramid is on
                # The verification pass issues its searches through the general-web
                # client (unmetered) when research is on; blind runs verify nothing.
                verify_client = research_ctx.general if research_ctx is not None else None
                record = _forecast_pyramid_question(
                    snapshot,
                    pyramid_settings,
                    pyramid_ensemble_settings,
                    pyramid_factory,
                    bundle,
                    research_state,
                    search_client=verify_client,
                    now=now,
                    to_store=to_store,
                    failures=failures,
                )
                if record is not None:
                    summaries.append(record)
                fc_store.append(to_store)
                continue

            if ensemble_settings is not None:
                assert member_factory is not None  # set whenever ensemble is on
                pooled = _forecast_ensemble_question(
                    snapshot,
                    ensemble_settings,
                    member_factory,
                    bundle,
                    research_state,
                    now=now,
                    to_store=to_store,
                    failures=failures,
                )
                if pooled is not None:
                    summaries.append(pooled)
                fc_store.append(to_store)
                continue

            try:
                record = run_forecast_pass(snapshot, client=client, config=config, research=bundle)
            except LlmError as exc:
                failures.append(f"q{snapshot.question_id}: {exc}")
                typer.echo(
                    f"q{snapshot.question_id} (post {snapshot.post_id}): forecast FAILED ({exc})",
                    err=True,
                )
                continue
            if research_state == "cached":
                # The bundle's credits were drawn when it was first built; this
                # run drew none. Zero the record's draw so summaries and quota
                # accounting never double-count a reused bundle.
                record = record.model_copy(update={"research_search_credits": 0})
            fc_store.append([record])
            summaries.append(record)
            _echo_forecast_summary(record, research_state)
    finally:
        if research_ctx is not None:
            research_ctx.close()
    total_cost = sum(r.cost_usd for r in summaries if r.cost_usd is not None)
    total_credits = sum(r.research_search_credits for r in summaries)
    if ensemble:
        mode = "ensemble"
    elif pyramid:
        mode = "pyramid"
    else:
        mode = "blind" if no_research else "researched"
    typer.echo(
        f"forecast ({mode}): n={len(summaries)} failed={len(failures)} "
        f"total_cost=${total_cost:.4f} search_credits={total_credits} (all dry-run)"
    )
    if failures:
        raise typer.Exit(code=1)


def _forecast_ensemble_question(
    snapshot: QuestionSnapshot,
    settings: EnsembleSettings,
    client_factory: Callable[[TaskModelConfig], LlmClient],
    bundle: ResearchBundle | None,
    research_state: str,
    *,
    now: datetime,
    to_store: list[ForecastRecord],
    failures: list[str],
) -> ForecastRecord | None:
    """Run the ensemble for one question; append member + pooled records to store.

    Returns the pooled record (for the run's cost/credit totals), or ``None`` when
    the ensemble failed for this question — a failed ensemble is isolated to its
    question exactly like a failed single pass: it is logged and the loop goes on.
    The shared bundle's search credits are attributed to the pooled record once
    and zeroed on the members (and zeroed on the pooled record too when the bundle
    was cached), so a run's ``search_credits`` never multiplies by member count.
    """
    try:
        result = run_ensemble(
            snapshot,
            settings=settings,
            research=bundle,
            client_factory=client_factory,
            now=now,
        )
    except LlmError as exc:
        failures.append(f"q{snapshot.question_id}: {exc}")
        typer.echo(
            f"q{snapshot.question_id} (post {snapshot.post_id}): ensemble FAILED ({exc})",
            err=True,
        )
        return None

    pooled = pooled_forecast_record(snapshot, result, research=bundle, now=now)
    if research_state == "cached":
        pooled = pooled.model_copy(update={"research_search_credits": 0})
    # Members share the one bundle; its credits are counted once on the pooled
    # record, so zero them on every member to avoid multiplying the draw by N.
    # The bundle itself is also stored once, on the pooled record: members drop
    # their copy (they all shared the identical bundle), or a 5-member ensemble
    # would persist six copies of the same multi-KB research per question.
    members = [
        member.model_copy(update={"research_search_credits": 0, "research": None})
        for member in result.member_records
    ]
    to_store.extend(members)
    to_store.append(pooled)
    _echo_ensemble_summary(pooled, result, research_state)
    return pooled


def _pyramid_task_configs() -> list[TaskModelConfig]:
    """The distinct task configs a pyramid run drives, for the provider readiness check.

    Decompose, leaf (``forecast_pass``), and holistic (``holistic`` if configured,
    else it reuses ``forecast_pass``). ``_require_llm_cli`` dedupes by provider, so
    listing all three is safe even when they share one.
    """
    configs = [load_task_config("decompose"), load_task_config("forecast_pass")]
    # holistic falls back to forecast_pass (already listed) when its table is absent.
    with contextlib.suppress(KeyError):
        configs.append(load_task_config("holistic"))
    return configs


def _forecast_pyramid_question(
    snapshot: QuestionSnapshot,
    settings: PyramidSettings,
    ensemble_settings: EnsembleSettings | None,
    client_factory: Callable[[TaskModelConfig], LlmClient],
    bundle: ResearchBundle | None,
    research_state: str,
    *,
    search_client: SearchClient | None,
    now: datetime,
    to_store: list[ForecastRecord],
    failures: list[str],
) -> ForecastRecord | None:
    """Run the pyramid for one question; append its single record to the store.

    Returns the record (for the run's cost/credit totals), or ``None`` when the
    pyramid raised for this question — a failure is isolated to its question like
    the ensemble and single-pass paths. The runner itself degrades model failures
    internally, so a raise here is unusual (e.g. a config problem); it is logged
    and the loop moves on. A cached bundle's credits are zeroed on the record so a
    reused bundle is never double-counted.
    """
    try:
        record = run_pyramid_forecast(
            snapshot,
            settings=settings,
            ensemble_settings=ensemble_settings,
            research=bundle,
            search_client=search_client,
            client_factory=client_factory,
            now=now,
        )
    except LlmError as exc:
        failures.append(f"q{snapshot.question_id}: {exc}")
        typer.echo(
            f"q{snapshot.question_id} (post {snapshot.post_id}): pyramid FAILED ({exc})",
            err=True,
        )
        return None

    if research_state == "cached":
        record = record.model_copy(update={"research_search_credits": 0})
    to_store.append(record)
    _echo_pyramid_summary(record, research_state)
    return record


def _research_for_question(
    snapshot: QuestionSnapshot,
    research_ctx: _ResearchContext,
    cache: ResearchCache,
    *,
    now: datetime,
) -> tuple[ResearchBundle, str]:
    """Return this question's research bundle and whether it was cached or fresh.

    Reuses a cached bundle whose ``retrieved_at`` is within the configured
    max-age window (``reuse_research_max_age_hours``; ``0`` disables reuse). On a
    miss, runs fresh research and saves it to the cache *before* returning — so a
    later forecast crash cannot lose the paid-for research.
    """
    max_age_hours = research_ctx.settings.reuse_research_max_age_hours
    if max_age_hours > 0:
        cached = cache.load(
            snapshot.question_id,
            max_age=timedelta(hours=max_age_hours),
            now=now,
        )
        if cached is not None:
            return cached, "cached"

    bundle = run_research(
        snapshot,
        research_ctx.llm,
        research_ctx.news,
        research_ctx.general,
        research_ctx.extractor,
        research_ctx.settings,
    )
    cache.save(snapshot.question_id, bundle)
    return bundle, "fresh"


def _echo_forecast_summary(record: ForecastRecord, research_state: str) -> None:
    """Print one per-question forecast result line (id, probability, cost, sources)."""
    cost = "n/a" if record.cost_usd is None else f"${record.cost_usd:.4f}"
    sources = "-" if record.research is None else str(len(record.research.results))
    typer.echo(
        f"q{record.question_id} (post {record.post_id}): "
        f"p={record.probability:.2f} confidence={record.confidence} "
        f"cost={cost} sources={sources} research={research_state}"
    )


def _echo_ensemble_summary(
    pooled: ForecastRecord, result: EnsembleResult, research_state: str
) -> None:
    """Print one per-question pooled-ensemble result line."""
    cost = "n/a" if pooled.cost_usd is None else f"${pooled.cost_usd:.4f}"
    n_ok = len(result.member_records)
    n_failed = len(result.failures)
    typer.echo(
        f"q{pooled.question_id} (post {pooled.post_id}): "
        f"p={pooled.probability:.2f} confidence={pooled.confidence} "
        f"spread={result.spread:.2f} members={n_ok}ok/{n_failed}failed "
        f"cost={cost} research={research_state}"
    )


def _echo_pyramid_summary(record: ForecastRecord, research_state: str) -> None:
    """Print one per-question pyramid result line (pooled/structural/holistic, etc.)."""
    cost = "n/a" if record.cost_usd is None else f"${record.cost_usd:.4f}"
    detail = record.pyramid
    if detail is None:  # defensive: a pyramid record always carries its detail
        typer.echo(f"q{record.question_id} (post {record.post_id}): p={record.probability:.2f}")
        return
    typer.echo(
        f"q{record.question_id} (post {record.post_id}): "
        f"p={record.probability:.2f} structural={detail.structural_probability:.2f} "
        f"holistic={detail.holistic_probability:.2f} confidence={record.confidence} "
        f"leaves={len(detail.leaf_estimates)} pruned={len(detail.prune_notes)} "
        f"cost={cost} research={research_state}"
    )


class _ResearchContext:
    """The clients the research pipeline needs, built once and closed together."""

    def __init__(
        self,
        llm: LlmClient,
        news: SearchClient,
        general: SearchClient,
        extractor: Extractor,
        settings: RetrievalSettings,
    ) -> None:
        self.llm = llm
        self.news = news
        self.general = general
        self.extractor = extractor
        self.settings = settings

    def close(self) -> None:
        """Close each underlying client that exposes a ``close`` method."""
        for client in (self.news, self.general, self.extractor):
            closer = getattr(client, "close", None)
            if callable(closer):
                closer()


def _open_research_context(snapshot_dir: Path) -> _ResearchContext:
    """Build the research clients from settings, keyed to the data-root quota file."""
    # The search keys live in .env; the token helpers load it for their commands,
    # but a dry-run forecast never calls those — load it here too.
    _load_dotenv(Path("engine/.env"))
    _load_dotenv(Path(".env"))
    settings = load_retrieval_settings()
    data_root = snapshot_dir.parent
    news, general = _build_search_clients(settings, data_root)
    extractor = _build_extractor(settings)
    llm = _build_research_llm()
    if llm is None:
        llm = build_llm_client(load_task_config("query_gen"))
    return _ResearchContext(llm, news, general, extractor, settings)


@app.command("submit-tournament")
def submit_tournament(
    tournament: str | None = typer.Option(
        None,
        "--tournament",
        help="Tournament slug to forecast. Defaults to the unscored sandbox from settings.toml.",
    ),
    max_questions: int = typer.Option(
        5, "--max-questions", help="Max open questions to forecast this run."
    ),
    live: bool = typer.Option(
        False,
        "--live/--no-live",
        help="Actually submit forecasts. Off by default: nothing is POSTed.",
    ),
    forecast_dir: Path | None = _ForecastDirOption,
) -> None:
    """Forecast open tournament questions and (with ``--live``) submit each once.

    Dry-run by default: forecasts are computed and stored but nothing is sent to
    Metaculus. Two hard rules are enforced: each question is forecast at most once
    ever (already-submitted questions are skipped), and every live submission is
    paired with a mandatory rationale comment.
    """
    # Tournament slugs come from settings.toml; --tournament overrides the default.
    settings = load_settings()
    tournament = tournament if tournament is not None else settings.tournament.sandbox

    # Double gate: the scored tournament must never be entered by accident.
    if tournament == settings.tournament.live and not live:
        typer.echo(
            f"Refusing to run against the scored tournament {tournament!r} without "
            "--live. This is a real, money-scored entry: re-run with --live once you "
            "have made the autonomy decision (spec checkpoint B3).",
            err=True,
        )
        raise typer.Exit(code=1)

    _require_llm_cli()

    # Token resolution. Submissions prefer the dedicated bot token but fall back to
    # METACULUS_TOKEN, which for this project is itself the bot account's token.
    # Listing works with either; a live run additionally needs a submission token.
    bot_token = _bot_token()
    human_token = os.environ.get("METACULUS_TOKEN")
    submission_token = bot_token or human_token
    if live and not submission_token:
        typer.echo(_BOT_TOKEN_HELP, err=True)
        raise typer.Exit(code=1)
    listing_token = bot_token or human_token
    if not listing_token:
        typer.echo(_TOKEN_HELP, err=True)
        raise typer.Exit(code=1)

    fc_store = ForecastStore(_resolve_forecast_dir(forecast_dir))
    now = datetime.now(UTC)

    # One-forecast-per-question rule: skip any question already submitted.
    already_submitted = {r.question_id for r in fc_store.iter_all() if not r.dry_run}

    list_client = _build_client(listing_token)
    posts: list[dict[str, object]] = []
    try:
        posts = list(
            list_client.iter_tournament_binary_posts(tournament, max_posts=max_questions * 3)
        )
    finally:
        list_client.close()

    config = load_task_config("forecast_pass")
    llm_client = _build_llm_client()
    sub_client: SubmissionClient | None = None
    if live:
        assert submission_token is not None  # guaranteed by the guard above
        sub_client = _build_submission_client(submission_token)

    records: list[ForecastRecord] = []
    failures: list[str] = []
    forecast_count = 0
    submitted_count = 0
    try:
        for post in posts:
            if forecast_count >= max_questions:
                break
            try:
                snapshot = QuestionSnapshot.from_api_post(post, now)
            except SkipPost:
                continue
            if snapshot.question_id in already_submitted:
                typer.echo(f"q{snapshot.question_id}: skipped (already submitted)")
                continue

            record = run_forecast_pass(snapshot, client=llm_client, config=config)
            forecast_count += 1

            status = "dry"
            if live and sub_client is not None:
                try:
                    sub_client.submit_binary_forecast(snapshot.question_id, record.probability)
                    comment = f"{_DISCLOSURE}\n\n{record.rationale}"
                    sub_client.post_rationale_comment(snapshot.post_id, comment)
                except SubmissionError as exc:
                    failures.append(f"q{snapshot.question_id}: {exc}")
                    typer.echo(f"q{snapshot.question_id}: submission FAILED ({exc})", err=True)
                    records.append(record)  # stored as dry_run: not submitted
                    _echo_forecast_line(record, "dry")
                    continue
                record = record.model_copy(update={"dry_run": False, "submitted_at": now})
                submitted_count += 1
                status = "SUBMITTED"

            records.append(record)
            _echo_forecast_line(record, status)
    finally:
        if sub_client is not None:
            sub_client.close()

    fc_store.append(records)
    total_cost = sum(r.cost_usd for r in records if r.cost_usd is not None)
    mode = "live" if live else "dry-run"
    typer.echo(
        f"submit-tournament ({mode}): forecast={forecast_count} "
        f"submitted={submitted_count} failed={len(failures)} "
        f"total_cost=${total_cost:.4f}"
    )
    if failures:
        typer.echo(f"{len(failures)} submission(s) failed:", err=True)
        for line in failures:
            typer.echo(f"  {line}", err=True)
        raise typer.Exit(code=1)


def _echo_forecast_line(record: ForecastRecord, status: str) -> None:
    """Print one per-question result line (id, probability, confidence, cost, status)."""
    cost = "n/a" if record.cost_usd is None else f"${record.cost_usd:.4f}"
    typer.echo(
        f"q{record.question_id} (post {record.post_id}): "
        f"p={record.probability:.2f} confidence={record.confidence} cost={cost} [{status}]"
    )


_ExperimentsDirOption = typer.Option(
    None,
    "--experiments-dir",
    help="Directory of experiment manifests (*.toml). Defaults to ./docs/experiments.",
)


def _refresh_pinned_resolutions(
    store: SnapshotStore,
    manifests: list[ExperimentManifest],
) -> int:
    """Re-check resolutions for JUST the pinned questions still open in the store.

    Reuses the ``update-resolutions`` machinery (the Metaculus client's
    ``get_post`` + ``QuestionSnapshot.from_api_post`` + append), but scoped to the
    union of every manifest's pinned questions whose latest snapshot is not yet
    resolved — so a refresh spends network only on the experiments' own open
    questions, not the whole store. Returns how many newly resolved.
    """
    pinned: set[int] = set()
    for manifest in manifests:
        pinned |= manifest.pinned

    latest = store.latest_by_question()
    # Map pinned, still-open question ids to their post ids for the API fetch.
    post_ids = [
        snapshot.post_id
        for qid, snapshot in latest.items()
        if qid in pinned and snapshot.status.lower() != "resolved"
    ]
    if not post_ids:
        return 0

    token = _require_token()
    scraped_at = datetime.now(UTC)
    snapshots: list[QuestionSnapshot] = []
    newly_resolved = 0
    client = _build_client(token)
    try:
        for post_id in post_ids:
            post = client.get_post(post_id)
            try:
                snapshot = QuestionSnapshot.from_api_post(post, scraped_at)
            except SkipPost:
                continue
            snapshots.append(snapshot)
            if snapshot.status.lower() == "resolved":
                newly_resolved += 1
    finally:
        client.close()

    store.append(snapshots)
    return newly_resolved


def _echo_experiment_status(status: ExperimentStatus) -> None:
    """Print one experiment's compact interim report."""
    m = status.manifest
    n_pinned = len(m.question_ids)
    typer.echo(f"=== Experiment {m.id}: {m.title}")
    typer.echo(f"    status={m.status} frozen={m.frozen.isoformat()} questions={n_pinned}")

    for arm in status.arms:
        unknown = f" (+{arm.unknown_cost_records} unknown-cost)" if arm.unknown_cost_records else ""
        typer.echo(
            f"    arm {arm.spec.name} [{arm.spec.label}]: "
            f"coverage={arm.coverage}/{n_pinned} records={arm.matched_records} "
            f"cost=${arm.total_cost_usd:.4f}{unknown}"
        )

    n_resolved = len(status.resolved_qids)
    typer.echo(f"    resolved: {n_resolved}/{n_pinned} pinned questions resolved yes/no")

    delta = status.paired_brier_delta
    if delta is None:
        typer.echo("    paired Brier delta (B-A): n/a (no question resolved with both arms yet)")
    else:
        typer.echo(
            f"    paired Brier delta (B-A): {delta:+.4f} over N={len(status.paired)} "
            "resolved paired questions (directional only; see the experiment doc's power check)"
        )

    typer.echo(
        f"    budget: ${status.total_known_cost_usd:.4f} known spend of ${m.budget_usd:.2f} "
        f"(${status.budget_remaining_usd:.2f} remaining)"
    )


@app.command("experiment-status")
def experiment_status(
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help=(
            "Re-check resolutions for the experiments' pinned, still-open questions "
            "before reporting (spends network). Off by default."
        ),
    ),
    data_dir: Path | None = _DataDirOption,
    forecast_dir: Path | None = _ForecastDirOption,
    experiments_dir: Path | None = _ExperimentsDirOption,
) -> None:
    """Report interim progress for every pre-registered experiment.

    Reads the manifests in ``docs/experiments/*.toml``, matches stored forecast
    records to each arm, and prints coverage, resolutions, the interim paired
    Brier delta, and budget per experiment. ``--refresh`` first re-checks
    resolutions for just the pinned unresolved questions (reusing the
    ``update-resolutions`` machinery), so an interim score reflects the latest
    outcomes.
    """
    manifests = load_manifests(experiments_dir)
    if not manifests:
        typer.echo("experiment-status: no experiment manifests found.")
        return

    snap_store = SnapshotStore(_resolve_data_dir(data_dir))
    fc_store = ForecastStore(_resolve_forecast_dir(forecast_dir))

    if refresh:
        newly = _refresh_pinned_resolutions(snap_store, manifests)
        typer.echo(f"refresh: re-checked pinned open questions, newly_resolved={newly}")

    latest = snap_store.latest_by_question()
    records = list(fc_store.iter_all())
    for manifest in manifests:
        resolutions = resolutions_from_store(latest, manifest.question_ids)
        status = compute_status(manifest, records, resolutions)
        _echo_experiment_status(status)


_RenderOutOption = typer.Option(
    None,
    "--out",
    help="Output HTML path. Defaults to docs/reviews/forecast-<question_id>.html.",
)
_RenderQuestionIdArgument = typer.Argument(
    ..., help="The question id whose latest forecast to render."
)


@app.command("render-forecast")
def render_forecast(
    question_id: int = _RenderQuestionIdArgument,
    out: Path | None = _RenderOutOption,
    forecast_dir: Path | None = _ForecastDirOption,
) -> None:
    """Render the latest stored forecast for a question into a self-contained HTML page.

    Reads every stored record, picks the most recent one for ``question_id``, and
    writes a deterministic, no-LLM argument view (the decomposition tree for a
    pyramid record; a graceful-degrade view for any other method). Prints the
    path it wrote.
    """
    fc_store = ForecastStore(_resolve_forecast_dir(forecast_dir))
    matching = [r for r in fc_store.iter_all() if r.question_id == question_id]
    if not matching:
        typer.echo(f"render-forecast: no stored forecast for question {question_id}.", err=True)
        raise typer.Exit(code=1)

    # The latest record for this question (records are appended over time).
    record = max(matching, key=lambda r: r.created_at)
    html = render_forecast_html(record)

    out_path = out if out is not None else Path("docs/reviews") / f"forecast-{question_id}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    typer.echo(f"render-forecast: wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
