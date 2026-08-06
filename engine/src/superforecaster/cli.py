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
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import typer

from .config import (
    EnsembleSettings,
    MarketCorpusSettings,
    PyramidSettings,
    RetrievalSettings,
    TaskModelConfig,
    load_ensemble_settings,
    load_market_corpus_settings,
    load_pyramid_settings,
    load_retrieval_settings,
    load_settings,
    load_task_config,
)
from .dotenv import load_dotenv as _load_dotenv
from .ensemble import EnsembleResult, pooled_forecast_record, run_ensemble
from .experiments import (
    DEFAULT_EXPERIMENTS_DIR,
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
from .market_corpus import (
    MarketClient,
    MarketCorpusStore,
    MarketQuestion,
    build_market_client,
    filtered,
)
from .metaculus import MetaculusClient
from .models import QuestionSnapshot, SkipPost
from .pyramid_runner import run_pyramid_forecast
from .render import render_forecast_html
from .research import ResearchBundle, run_research
from .research_cache import ResearchCache
from .research_eval import (
    BundleEvalRecord,
    BundleEvalStore,
    BundleMetrics,
    PairwiseRecord,
    bundle_fingerprint,
    compute_metrics,
    load_bundle_judge_config,
    run_pairwise_judge,
    run_rubric_judge,
    summarize_retrieval,
)
from .search import SearchClient, build_search_client, read_asknews_credits_used
from .store import SnapshotStore
from .submission import SubmissionClient, SubmissionError
from .web_resolutions import (
    WebResolution,
    WebResolutionStore,
    default_page_loader,
    fetch_resolution,
)

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


# Web-resolution pass tunables. The volume is tiny (a few questions resolve per
# week across ~45 pinned questions), so we crawl politely and cap hard: at most
# one page every few seconds, and never more than a fixed number of pages a run.
_WEB_THROTTLE_SECONDS = 3.0
_WEB_PAGE_CAP = 20


def _resolve_web_resolution_dir(data_dir: Path | None) -> Path:
    """Pick the web-resolution directory.

    With an explicit ``--data-dir``, web resolutions sit beside the snapshots
    under the same data root (``<data-dir>/../web-resolutions``), so a test's
    artifacts stay together. Otherwise the config default
    (``[data] web_resolution_dir``) is used.
    """
    if data_dir is not None:
        return data_dir.parent / "web-resolutions"
    return load_settings().data.web_resolution_dir


# Seam for tests: monkeypatched to inject a fake page loader (no browser).
def _build_web_page_loader() -> Callable[[str], str]:
    """Return the page loader the web-resolution pass uses (real headless browser).

    Tests monkeypatch this to hand back a fake loader that returns canned page
    text, so the web pass is exercised end-to-end without launching Chromium.
    """
    return default_page_loader


# Seam for tests: monkeypatched to a spy/no-op so the throttle is observed
# without real waiting.
def _web_sleep(seconds: float) -> None:  # pragma: no cover - thin time.sleep wrapper
    """Sleep between web-resolution page loads (the polite crawl throttle)."""
    time.sleep(seconds)


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


# Seam for tests: monkeypatched to inject a fake bundle-judge LLM client.
def _build_bundle_judge_llm() -> LlmClient | None:
    """Return ``None`` so the eval commands build the judge client themselves.

    Mirrors :func:`_build_llm_client`: tests monkeypatch this to inject a fake that
    returns a rubric/pairwise JSON object, distinct from the forecast-pass fake.
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
    from_web: bool = typer.Option(
        True,
        "--from-web/--no-from-web",
        help=(
            "After the API pass, fill still-missing resolutions by reading the "
            "public Metaculus question pages with a headless browser (the API "
            "hides 'resolution' unless the account forecast on the question). "
            "On by default."
        ),
    ),
    data_dir: Path | None = _DataDirOption,
) -> None:
    """Re-fetch tracked, still-open questions to capture any new resolutions.

    Two passes. First the API pass re-fetches each still-open tracked question and
    appends a fresh snapshot. Then, unless ``--no-from-web`` is given, a web pass
    reads outcomes off the public question pages for tracked questions whose latest
    snapshot is resolved/closed yet still carries no ``resolution`` (the API's
    access-tier gap) — throttled and hard-capped, and recorded with provenance in
    the web-resolution store.
    """
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

    if from_web:
        _web_resolution_pass(store, data_dir)


def _web_resolution_pass(store: SnapshotStore, data_dir: Path | None) -> None:
    """Fill resolution gaps from the public pages for resolved/closed-but-blank questions.

    Candidates are the tracked questions whose *latest* snapshot has a status of
    ``resolved`` or ``closed`` but no ``resolution`` — exactly the gap the API's
    access tier leaves. Each is read from its public page, throttled to one page
    per :data:`_WEB_THROTTLE_SECONDS` and hard-capped at :data:`_WEB_PAGE_CAP`
    pages per run. Found outcomes are appended (with provenance) to the
    web-resolution store; a page with no banner is counted as a miss, never
    guessed. Prints a ``checked/found/failed`` summary line.
    """
    latest = store.latest_by_question()
    candidates = [
        snapshot
        for snapshot in latest.values()
        if snapshot.status.lower() in ("resolved", "closed") and snapshot.resolution is None
    ]
    candidates.sort(key=lambda s: s.post_id)

    web_store = WebResolutionStore(_resolve_web_resolution_dir(data_dir))
    loader = _build_web_page_loader()

    found: list[WebResolution] = []
    checked = 0
    failed = 0
    for snapshot in candidates:
        if checked >= _WEB_PAGE_CAP:
            break
        if checked > 0:
            _web_sleep(_WEB_THROTTLE_SECONDS)
        checked += 1
        try:
            resolution = fetch_resolution(
                snapshot.post_id,
                page_loader=loader,
                question_id=snapshot.question_id,
            )
        except Exception as exc:
            # One bad page (browser crash, navigation error) must not sink the run.
            failed += 1
            typer.echo(
                f"q{snapshot.question_id} (post {snapshot.post_id}): web-resolution FAILED ({exc})",
                err=True,
            )
            continue
        if resolution is None:
            failed += 1
            continue
        found.append(resolution)

    web_store.append(found)
    typer.echo(f"update-resolutions (web): checked={checked} found={len(found)} failed={failed}")


_ExperimentsDirOption = typer.Option(
    None,
    "--experiments-dir",
    help="Directory of experiment manifests (*.toml/*.md). Defaults to ./docs/experiments.",
)


# A pinned-question-id line: PURELY digits/commas/whitespace, with an optional
# trailing period (the doc convention: a wrapped list ending the sentence). Prose
# lines mix digits with words and never match, so this is safe to run over every
# doc in the directory, not just the doc-only experiments.
_DOC_ID_LIST_LINE_RE = re.compile(r"^\s*\d[\d,\s]*\.?\s*$")
_DOC_ID_RE = re.compile(r"\d{4,6}")


def _pinned_ids_from_docs(directory: Path) -> set[int]:
    """Collect pinned question ids recorded only in a doc, not a `.toml` manifest.

    Some pre-registered experiments (005-effort-levels, 006-perplexity-search) pin
    their question set solely in their markdown doc — no manifest is registered
    because their machinery (a script-scored effort comparison; a bundle-eval-only
    pairwise judge) isn't the arm-matching `.toml` shape (each doc's own
    "Machinery note" / "No `.toml` manifest is registered" line explains why).

    Rather than hardcoding those ids, this scans every ``*.md`` file in
    ``directory`` for lines that are a comma-separated list of 4-6 digit numbers
    (Metaculus question ids) and nothing else, e.g.::

        43716, 43507, 44373, 43892, 44422, 43873, 44011, 44017, 44256, 35710,
        44139, 44698, 43852, 44248, 43854, 39432, 43522, 43506, 44250, 43737.

    This also picks up 001's and 002's own id lists from their docs, which is
    harmless — they are identical to those experiments' ``.toml`` manifests, so
    the union below is a no-op for them. Returns an empty set if the directory
    does not exist.
    """
    ids: set[int] = set()
    if not directory.exists():
        return ids
    for path in sorted(directory.glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if _DOC_ID_LIST_LINE_RE.match(line):
                ids.update(int(m) for m in _DOC_ID_RE.findall(line))
    return ids


_SweepMaxOption = typer.Option(
    25,
    "--max",
    help=(
        "Max pages to read this run. Kept small so a run stays short and safely "
        "re-runnable — the web-resolution store's dedupe (already-resolved "
        "questions are skipped) means a later run just resumes."
    ),
)
_SweepThrottleOption = typer.Option(
    _WEB_THROTTLE_SECONDS,
    "--throttle-seconds",
    help="Seconds to sleep between page loads (politeness throttle).",
)


@app.command("sweep-resolutions")
def sweep_resolutions(
    max_pages: int = _SweepMaxOption,
    throttle_seconds: float = _SweepThrottleOption,
    data_dir: Path | None = _DataDirOption,
    experiments_dir: Path | None = _ExperimentsDirOption,
) -> None:
    """Read resolutions for every PINNED experiment question via the public-page web reader.

    Unlike ``update-resolutions --from-web`` (which only mops up gaps left by an
    API pass over the WHOLE tracked set), this sweeps the pinned question set of
    every pre-registered experiment directly and ONLY via the public-page web
    reader — never the Metaculus API, which both 429s at this volume and hides
    the ``resolution`` field behind account access.

    Pinned ids are the union of every manifest-backed experiment's question set
    (``docs/experiments/*.toml``, via :func:`load_manifests`) and every doc-only
    experiment's question set (see :func:`_pinned_ids_from_docs`). Each pinned id
    is mapped to its post id via the snapshot store's latest snapshot; ids with no
    tracked snapshot are silently unmapped (not an error — a question outside the
    scrape window just can't be checked here). Questions already resolved in the
    web-resolution store are skipped. Remaining questions are read from their
    public page, throttled and capped at ``--max`` pages; a reader error on one
    question is logged and the sweep continues rather than aborting.
    """
    manifests = load_manifests(experiments_dir)
    manifest_ids: set[int] = set()
    for manifest in manifests:
        manifest_ids |= manifest.pinned
    docs_dir = experiments_dir if experiments_dir is not None else Path(DEFAULT_EXPERIMENTS_DIR)
    pinned = manifest_ids | _pinned_ids_from_docs(docs_dir)

    snap_store = SnapshotStore(_resolve_data_dir(data_dir))
    latest = snap_store.latest_by_question()
    by_qid = {qid: latest[qid].post_id for qid in pinned if qid in latest}

    web_store = WebResolutionStore(_resolve_web_resolution_dir(data_dir))
    already = {qid for qid, r in web_store.latest_by_question().items() if r.resolution}

    todo = sorted(qid for qid in by_qid if qid not in already)[:max_pages]

    loader = _build_web_page_loader()
    found: list[WebResolution] = []
    for i, question_id in enumerate(todo):
        if i > 0:
            _web_sleep(throttle_seconds)
        post_id = by_qid[question_id]
        try:
            resolution = fetch_resolution(post_id, page_loader=loader, question_id=question_id)
        except Exception as exc:
            # One bad page must never sink the sweep.
            typer.echo(f"q{question_id} (post {post_id}): web-resolution FAILED ({exc})", err=True)
            continue
        if resolution is None:
            continue
        web_store.append([resolution])
        found.append(resolution)
        typer.echo(f"q{question_id}: RESOLVED {resolution.resolution!r}")

    typer.echo(
        f"sweep-resolutions: pinned={len(pinned)} mapped={len(by_qid)} "
        f"already={len(already)} checked={len(todo)} found={len(found)}"
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


def _resolve_market_corpus_dir(data_dir: Path | None) -> Path:
    """Pick the market-corpus directory.

    With an explicit ``--data-dir`` the corpus files sit beside the snapshots under
    the same data root (``<data-dir>/../market-corpus``), so a test's artifacts stay
    together (mirrors :func:`_resolve_web_resolution_dir`). Otherwise the corpus lands
    under the config data root (``<snapshot_dir>/../market-corpus`` -> the
    ``./data/market-corpus`` the spec names).
    """
    if data_dir is not None:
        return data_dir.parent / "market-corpus"
    return load_settings().data.snapshot_dir.parent / "market-corpus"


# Seam for tests: monkeypatched to inject a fake market client (no network).
def _build_market_client(platform: str, settings: MarketCorpusSettings) -> MarketClient:
    """Construct the real market client for a platform. Tests replace this seam."""
    return build_market_client(platform, settings=settings)


_MarketPlatformOption = typer.Option(
    "both",
    "--platform",
    help="Which market(s) to pull: 'polymarket', 'manifold', or 'both'.",
)
_MarketMaxOption = typer.Option(
    300,
    "--max",
    help="Max resolved-binary markets to pull PER platform (before filtering).",
)


@app.command("build-market-corpus")
def build_market_corpus(
    platform: str = _MarketPlatformOption,
    max_markets: int = _MarketMaxOption,
    data_dir: Path | None = _DataDirOption,
) -> None:
    """Pull resolved binary markets, run mechanical quality filters, and store them.

    Read-only public APIs (Polymarket Gamma + Manifold v0, no auth): each platform
    is paginated for up to ``--max`` resolved binaries, every parsed market is
    filtered (criteria length, volume/bettors, Manifold self-resolution) with its
    pass/fail outcome recorded on the record, and ALL parsed markets — kept and
    dropped alike — are appended (deduped by id) to ``data/market-corpus/<platform>``.
    A per-platform stats line reports pulled / parsed / kept / dropped-by-reason,
    the resolution date range, and median volume.
    """
    if platform not in ("polymarket", "manifold", "both"):
        typer.echo(
            f"build-market-corpus: --platform must be polymarket|manifold|both, got {platform!r}.",
            err=True,
        )
        raise typer.Exit(code=1)

    settings = load_market_corpus_settings()
    store = MarketCorpusStore(_resolve_market_corpus_dir(data_dir))
    platforms = ["polymarket", "manifold"] if platform == "both" else [platform]

    for plat in platforms:
        client = _build_market_client(plat, settings)
        records: list[MarketQuestion] = []
        try:
            for record in client.iter_resolved_binaries(max_markets):
                records.append(filtered(record, settings))
        finally:
            client.close()

        written = store.append(records)
        _echo_market_corpus_summary(
            plat,
            pulled=getattr(client, "pulled_count", len(records)),
            skipped=getattr(client, "skipped_count", 0),
            records=records,
            written=written,
        )


def _echo_market_corpus_summary(
    platform: str,
    *,
    pulled: int,
    skipped: int,
    records: list[MarketQuestion],
    written: int,
) -> None:
    """Print one platform's pull stats: counts, dropped-by-reason, dates, median volume."""
    parsed = len(records)
    kept = [r for r in records if r.kept]
    dropped = [r for r in records if not r.kept]

    reason_counts: dict[str, int] = {}
    for record in dropped:
        for reason in record.filters_failed:
            name = reason.split(":", 1)[0]
            reason_counts[name] = reason_counts.get(name, 0) + 1

    res_times = sorted(r.resolution_time for r in kept if r.resolution_time is not None)
    if res_times:
        date_range = f"{res_times[0].date().isoformat()}..{res_times[-1].date().isoformat()}"
    else:
        date_range = "n/a"

    volumes = sorted(r.volume for r in kept if r.volume is not None)
    median_vol = f"{volumes[len(volumes) // 2]:.0f}" if volumes else "n/a"

    typer.echo(
        f"build-market-corpus [{platform}]: pulled={pulled} parsed={parsed} "
        f"skipped_at_parse={skipped} kept={len(kept)} dropped={len(dropped)} "
        f"written={written}"
    )
    if reason_counts:
        by_reason = " ".join(f"{name}={count}" for name, count in sorted(reason_counts.items()))
        typer.echo(f"    dropped by: {by_reason}")
    typer.echo(f"    kept resolution dates: {date_range}   median volume: {median_vol}")


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
    experiment_id: str | None = typer.Option(
        None,
        "--experiment-id",
        help=(
            "Stamp every record this run persists (single, pooled ensemble AND "
            "members, pyramid) with this pre-registered experiment id (e.g. "
            "'001'). Stamped records are scored ONLY by their own experiment's "
            "manifest, so overlapping question sets never displace each other."
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
                fc_store.append(_stamped(to_store, experiment_id))
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
                fc_store.append(_stamped(to_store, experiment_id))
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
            fc_store.append(_stamped([record], experiment_id))
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
    experiment_note = f" experiment={experiment_id}" if experiment_id is not None else ""
    typer.echo(
        f"forecast ({mode}): n={len(summaries)} failed={len(failures)} "
        f"total_cost=${total_cost:.4f} search_credits={total_credits}"
        f"{experiment_note} (all dry-run)"
    )
    if failures:
        raise typer.Exit(code=1)


def _stamped(records: list[ForecastRecord], experiment_id: str | None) -> list[ForecastRecord]:
    """Return the records stamped with the run's experiment id (no-op when ``None``).

    Applied at every persist site of the ``forecast`` command, so a run made for
    a pre-registered experiment marks EVERY record it stores — the single pass,
    the pooled ensemble record and its members, and the pyramid record alike.
    """
    if experiment_id is None:
        return records
    return [record.model_copy(update={"experiment_id": experiment_id}) for record in records]


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
    and the loop moves on. On a cached bundle only the *bundle's* credits are
    subtracted: the verification and leaf-top-up searches are drawn fresh on every
    run (even a cached one), so their credits must always be counted.
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
        # Zero only the reused bundle's share; keep the fresh verification + top-up
        # credits (spec §5 / stage-1 known limitation fix). A pyramid record always
        # carries its detail, but guard defensively.
        detail = record.pyramid
        fresh_credits = (
            detail.assumption_search_credits + detail.leaf_topup_search_credits
            if detail is not None
            else 0
        )
        record = record.model_copy(update={"research_search_credits": fresh_credits})
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


def _open_research_context(
    snapshot_dir: Path, settings: RetrievalSettings | None = None
) -> _ResearchContext:
    """Build the research clients from settings, keyed to the data-root quota file.

    ``settings`` defaults to the current ``[retrieval]`` config; passing explicit
    settings (loaded from an alt config path) is how ``compare-research`` builds
    the alternative bundle under a different retrieval config without mutating the
    ``SUPERFORECASTER_CONFIG`` env var mid-process.
    """
    # The search keys live in .env; the token helpers load it for their commands,
    # but a dry-run forecast never calls those — load it here too.
    _load_dotenv(Path("engine/.env"))
    _load_dotenv(Path(".env"))
    retrieval = settings if settings is not None else load_retrieval_settings()
    data_root = snapshot_dir.parent
    news, general = _build_search_clients(retrieval, data_root)
    extractor = _build_extractor(retrieval)
    llm = _build_research_llm()
    if llm is None:
        llm = build_llm_client(load_task_config("query_gen"))
    return _ResearchContext(llm, news, general, extractor, retrieval)


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

    # Web-resolution overlay: outcomes read off the public pages fill any gap the
    # API left (an API resolution always wins; see resolutions_from_store).
    web_store = WebResolutionStore(_resolve_web_resolution_dir(data_dir))
    web_overlay = {qid: wr.resolution for qid, wr in web_store.latest_by_question().items()}

    for manifest in manifests:
        resolutions = resolutions_from_store(latest, manifest.question_ids, web_overlay)
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


_AltConfigOption = typer.Option(
    ...,
    "--alt-config",
    help=(
        "Path to an alternative settings.toml. Bundle A is built under the current "
        "settings, bundle B under this one (differing only in retrieval providers) "
        "— the Perplexity test's entry point."
    ),
)


def _resolve_bundle_eval_dir(data_dir: Path | None) -> Path:
    """Pick the bundle-eval directory.

    With an explicit ``--data-dir`` the eval records sit beside the snapshots under
    the same data root (``<data-dir>/../bundle-evals``), so a test's artifacts stay
    together. Otherwise the config default (``[data] bundle_eval_dir``) is used.
    """
    if data_dir is not None:
        return data_dir.parent / "bundle-evals"
    return load_settings().data.bundle_eval_dir


def _snapshot_for_post(snap_store: SnapshotStore, post_id: int) -> QuestionSnapshot | None:
    """Return the latest snapshot for a post id, or ``None`` if untracked."""
    by_post = {s.post_id: s for s in snap_store.latest_by_question().values()}
    return by_post.get(post_id)


def _bundle_for_eval(
    snapshot: QuestionSnapshot,
    snap_dir: Path,
    *,
    fresh: bool,
    now: datetime,
    settings: RetrievalSettings | None = None,
) -> ResearchBundle | None:
    """Load a cached bundle for a question, or build a fresh one under ``settings``.

    With ``fresh=False`` the bundle is read from the research cache (the one the
    ``forecast`` command fills); a miss returns ``None`` so the caller can report
    it. With ``fresh=True`` a new bundle is researched (and cached) under
    ``settings`` (defaulting to the current ``[retrieval]`` config).
    """
    if not fresh:
        cache = ResearchCache(snap_dir.parent / "research")
        return cache.load(snapshot.question_id, max_age=timedelta.max, now=now)

    research_ctx = _open_research_context(snap_dir, settings)
    try:
        bundle = run_research(
            snapshot,
            research_ctx.llm,
            research_ctx.news,
            research_ctx.general,
            research_ctx.extractor,
            research_ctx.settings,
            now=now,
        )
    finally:
        research_ctx.close()
    ResearchCache(snap_dir.parent / "research").save(snapshot.question_id, bundle)
    return bundle


@app.command("eval-research")
def eval_research(
    post_id: int = typer.Option(..., "--post-id", help="The post id whose bundle to evaluate."),
    fresh: bool = typer.Option(
        False,
        "--fresh/--cached",
        help=(
            "Build a fresh research bundle under the current settings (spends "
            "search credits), or evaluate the cached bundle from a prior run "
            "(default)."
        ),
    ),
    data_dir: Path | None = _DataDirOption,
) -> None:
    """Evaluate one question's research bundle: mechanical metrics + a rubric judge.

    Loads (or freshly builds) the bundle for ``--post-id``, computes the free
    mechanical metrics, runs the one-call rubric judge over a provider-blinded
    rendering, stores a :class:`BundleEvalRecord`, and echoes a summary line with
    the five dimension scores.
    """
    snap_dir = _resolve_data_dir(data_dir)
    snap_store = SnapshotStore(snap_dir)
    snapshot = _snapshot_for_post(snap_store, post_id)
    if snapshot is None:
        typer.echo(f"eval-research: no tracked snapshot for post {post_id}.", err=True)
        raise typer.Exit(code=1)

    judge_config = load_bundle_judge_config()
    check_configs = [judge_config]
    if fresh:
        check_configs.append(load_task_config("query_gen"))
    _require_llm_cli(check_configs)

    now = datetime.now(UTC)
    bundle = _bundle_for_eval(snapshot, snap_dir, fresh=fresh, now=now)
    if bundle is None:
        typer.echo(
            f"eval-research: no cached research bundle for post {post_id} "
            f"(question {snapshot.question_id}); re-run with --fresh to build one.",
            err=True,
        )
        raise typer.Exit(code=1)

    metrics = compute_metrics(snapshot, bundle, now=now)
    retrieval = load_retrieval_settings()
    judge_client = _build_bundle_judge_llm()
    try:
        result = run_rubric_judge(snapshot, bundle, client=judge_client, config=judge_config)
    except LlmError as exc:
        typer.echo(f"eval-research: rubric judge FAILED ({exc})", err=True)
        raise typer.Exit(code=1) from exc

    record = BundleEvalRecord(
        created_at=now,
        question_id=snapshot.question_id,
        post_id=snapshot.post_id,
        title=snapshot.title,
        kind="rubric",
        bundle_fingerprint=bundle_fingerprint(bundle),
        retrieval=summarize_retrieval(retrieval),
        metrics=metrics,
        rubric=result.judgment,
        judge_model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
    )
    BundleEvalStore(_resolve_bundle_eval_dir(data_dir)).append([record])
    _echo_rubric_summary(record)


def _echo_rubric_summary(record: BundleEvalRecord) -> None:
    """Print one eval-research result line (metrics headline + rubric scores)."""
    m = record.metrics
    cost = "n/a" if record.cost_usd is None else f"${record.cost_usd:.4f}"
    typer.echo(
        f"q{record.question_id} (post {record.post_id}): "
        f"results={m.result_count} domains={m.unique_domains} "
        f"docs_ok={m.doc_ok_count}/{m.doc_count} "
        f"coverage={m.key_terms_covered}/{m.key_terms_total} "
        f"resolution_source={'yes' if m.resolution_source_present else 'no'} "
        f"credits={m.search_credits_used} cost={cost}"
    )
    j = record.rubric
    if j is not None:
        typer.echo(
            f"    rubric: coverage={j.coverage} source_quality={j.source_quality} "
            f"recency={j.recency} relevance={j.relevance} "
            f"resolution_grounding={j.resolution_grounding} overall={j.overall}"
        )
        typer.echo(f"    missing: {j.missing}")


@app.command("compare-research")
def compare_research(
    post_id: int = typer.Option(..., "--post-id", help="The post id whose bundles to compare."),
    alt_config: Path = _AltConfigOption,
    data_dir: Path | None = _DataDirOption,
) -> None:
    """Compare two research bundles for one question: metrics on both + blinded pairwise.

    Builds bundle A under the current settings and bundle B under ``--alt-config``
    (both fresh, at the same moment), computes metrics on each, runs one blinded
    pairwise judge (X/Y assignment randomized and recorded), stores an eval record
    for each bundle (the verdict attached to A), and echoes a summary.
    """
    snap_dir = _resolve_data_dir(data_dir)
    snap_store = SnapshotStore(snap_dir)
    snapshot = _snapshot_for_post(snap_store, post_id)
    if snapshot is None:
        typer.echo(f"compare-research: no tracked snapshot for post {post_id}.", err=True)
        raise typer.Exit(code=1)

    judge_config = load_bundle_judge_config()
    _require_llm_cli([judge_config, load_task_config("query_gen")])

    settings_a = load_retrieval_settings()
    # Load the alt retrieval config straight from its path — no env mutation.
    settings_b = load_retrieval_settings(alt_config)

    now = datetime.now(UTC)
    bundle_a = _bundle_for_eval(snapshot, snap_dir, fresh=True, now=now, settings=settings_a)
    bundle_b = _bundle_for_eval(snapshot, snap_dir, fresh=True, now=now, settings=settings_b)
    assert bundle_a is not None and bundle_b is not None  # fresh always builds

    metrics_a = compute_metrics(snapshot, bundle_a, now=now)
    metrics_b = compute_metrics(snapshot, bundle_b, now=now)

    judge_client = _build_bundle_judge_llm()
    try:
        result = run_pairwise_judge(
            snapshot, bundle_a, bundle_b, client=judge_client, config=judge_config, now=now
        )
    except LlmError as exc:
        typer.echo(f"compare-research: pairwise judge FAILED ({exc})", err=True)
        raise typer.Exit(code=1) from exc

    fp_a = bundle_fingerprint(bundle_a)
    fp_b = bundle_fingerprint(bundle_b)
    summary_a = summarize_retrieval(settings_a)
    summary_b = summarize_retrieval(settings_b)
    # The randomization mapping is stored relative to A (this record's own bundle):
    # A was shown as X iff the judge's X was bundle A.
    pairwise = PairwiseRecord(
        verdict=result.verdict,
        x_source="self" if result.x_source == "a" else "other",
        other_fingerprint=fp_b,
        other_retrieval=summary_b,
    )
    record_a = BundleEvalRecord(
        created_at=now,
        question_id=snapshot.question_id,
        post_id=snapshot.post_id,
        title=snapshot.title,
        kind="pairwise",
        bundle_fingerprint=fp_a,
        retrieval=summary_a,
        metrics=metrics_a,
        pairwise=pairwise,
        judge_model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
    )
    record_b = BundleEvalRecord(
        created_at=now,
        question_id=snapshot.question_id,
        post_id=snapshot.post_id,
        title=snapshot.title,
        kind="pairwise",
        bundle_fingerprint=fp_b,
        retrieval=summary_b,
        metrics=metrics_b,
    )
    BundleEvalStore(_resolve_bundle_eval_dir(data_dir)).append([record_a, record_b])
    _echo_pairwise_summary(record_a, metrics_b)


def _echo_pairwise_summary(record_a: BundleEvalRecord, metrics_b: BundleMetrics) -> None:
    """Print one compare-research result line (both bundles' headline + winners)."""
    ma = record_a.metrics
    pr = record_a.pairwise
    cost = "n/a" if record_a.cost_usd is None else f"${record_a.cost_usd:.4f}"
    ra = record_a.retrieval
    other = pr.other_retrieval if pr is not None else ra
    typer.echo(f"q{record_a.question_id} (post {record_a.post_id}): comparing research")
    typer.echo(
        f"    A [{ra.news_provider}/{ra.general_provider}]: results={ma.result_count} "
        f"domains={ma.unique_domains} docs_ok={ma.doc_ok_count}/{ma.doc_count} "
        f"coverage={ma.key_terms_covered}/{ma.key_terms_total}"
    )
    mb = metrics_b
    typer.echo(
        f"    B [{other.news_provider}/{other.general_provider}]: results={mb.result_count} "
        f"domains={mb.unique_domains} docs_ok={mb.doc_ok_count}/{mb.doc_count} "
        f"coverage={mb.key_terms_covered}/{mb.key_terms_total}"
    )
    if pr is not None:
        v = pr.verdict
        x_is = "A" if pr.x_source == "self" else "B"
        typer.echo(
            f"    pairwise (X={x_is}): coverage={v.coverage} source_quality={v.source_quality} "
            f"recency={v.recency} relevance={v.relevance} "
            f"resolution_grounding={v.resolution_grounding} overall={v.overall} cost={cost}"
        )
        typer.echo(f"    reason: {v.reason}")


if __name__ == "__main__":  # pragma: no cover
    app()
