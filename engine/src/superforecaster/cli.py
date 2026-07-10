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

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import typer

from .config import RetrievalSettings, load_retrieval_settings, load_settings, load_task_config
from .dotenv import load_dotenv as _load_dotenv
from .extract import Extractor, build_extractor
from .forecast_store import ForecastStore
from .forecaster import ForecastRecord, run_forecast_pass
from .llm import LlmClient, LlmError, build_llm_client, check_provider_cli
from .metaculus import MetaculusClient
from .models import QuestionSnapshot, SkipPost
from .research import ResearchBundle, run_research
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


def _require_llm_cli() -> None:
    """Ensure the configured provider's CLI is installed, or exit with help.

    Provider-aware: it checks whichever binary the ``forecast_pass`` task's
    provider needs (today that is the ``claude`` CLI). An unknown provider name
    in config surfaces here as a clear error rather than deep in a forecast.
    """
    config = load_task_config("forecast_pass")
    try:
        available = check_provider_cli(config)
    except LlmError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if not available:
        typer.echo(_CLAUDE_CLI_HELP, err=True)
        raise typer.Exit(code=1)


# Seam for tests: monkeypatched to inject a fake LLM client.
def _build_llm_client() -> LlmClient | None:
    """Return ``None`` so the forecaster builds the provider's client itself.

    Tests monkeypatch this to inject a fake client with a ``.run(...)`` method,
    exactly as ``_build_client`` does for the Metaculus side.
    """
    return None


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
    data_dir: Path | None = _DataDirOption,
    forecast_dir: Path | None = _ForecastDirOption,
    post_id: list[int] = _PostIdOption,
) -> None:
    """Forecast open tournament questions, with web research on by default.

    Each question is researched (query generation -> news + general search ->
    content extraction) into a frozen bundle that is rendered into the prompt and
    stored with the forecast. ``--no-research`` forecasts blind, preserving the
    baseline for side-by-side comparison. All forecasts are dry runs — nothing is
    submitted. Per-question LLM cost and search-credit draw are both logged.
    """
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

    config = load_task_config("forecast_pass")
    client = _build_llm_client()

    # Research context. Built once and reused across questions; the quota counter
    # lives in the data root (the parent of the snapshot dir), config-derived so
    # there is never a hard-coded absolute path.
    research_ctx = None if no_research else _open_research_context(snap_dir)

    records: list[ForecastRecord] = []
    try:
        for snapshot in questions:
            bundle: ResearchBundle | None = None
            if research_ctx is not None:
                bundle = run_research(
                    snapshot,
                    research_ctx.llm,
                    research_ctx.news,
                    research_ctx.general,
                    research_ctx.extractor,
                    research_ctx.settings,
                )
            record = run_forecast_pass(snapshot, client=client, config=config, research=bundle)
            records.append(record)
            _echo_forecast_summary(record)
    finally:
        if research_ctx is not None:
            research_ctx.close()

    fc_store.append(records)
    total_cost = sum(r.cost_usd for r in records if r.cost_usd is not None)
    total_credits = sum(r.research_search_credits for r in records)
    mode = "blind" if no_research else "researched"
    typer.echo(
        f"forecast ({mode}): n={len(records)} total_cost=${total_cost:.4f} "
        f"search_credits={total_credits} (all dry-run)"
    )


def _echo_forecast_summary(record: ForecastRecord) -> None:
    """Print one per-question forecast result line (id, probability, cost, sources)."""
    cost = "n/a" if record.cost_usd is None else f"${record.cost_usd:.4f}"
    sources = "-" if record.research is None else str(len(record.research.results))
    typer.echo(
        f"q{record.question_id} (post {record.post_id}): "
        f"p={record.probability:.2f} confidence={record.confidence} "
        f"cost={cost} sources={sources}"
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


if __name__ == "__main__":  # pragma: no cover
    app()
