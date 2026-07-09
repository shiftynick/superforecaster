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

from .config import load_task_config
from .dotenv import load_dotenv as _load_dotenv
from .forecast_store import DEFAULT_FORECAST_DIR, ForecastStore
from .forecaster import ForecastRecord, run_forecast_pass
from .llm import LlmClient, LlmError, check_provider_cli
from .metaculus import MetaculusClient
from .models import QuestionSnapshot, SkipPost
from .store import SnapshotStore
from .submission import SubmissionClient, SubmissionError

app = typer.Typer(
    add_completion=False,
    help="Scrape and inspect Metaculus question snapshots. Run from the repo root.",
)

_DEFAULT_DATA_DIR = "./data/snapshots"

# Tournament slugs, verified during recon.
LIVE_TOURNAMENT = "summer-futureeval-2026"
SANDBOX_TOURNAMENT = "bot-testing-area"
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
    """Pick the snapshot directory: flag > ``SUPERFORECASTER_DATA_DIR`` > default."""
    if data_dir is not None:
        return data_dir
    env = os.environ.get("SUPERFORECASTER_DATA_DIR")
    if env:
        return Path(env)
    return Path(_DEFAULT_DATA_DIR)


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
    store = SnapshotStore(_resolve_data_dir(data_dir))
    s = store.stats()
    typer.echo(f"data dir:           {store.dir}")
    typer.echo(f"total records:      {s.total_records}")
    typer.echo(f"unique questions:   {s.unique_questions}")
    typer.echo(f"resolved questions: {s.resolved_questions}")
    typer.echo(f"first scrape:       {s.first_scrape}")
    typer.echo(f"last scrape:        {s.last_scrape}")


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
    """Pick the forecast directory: flag > ``SUPERFORECASTER_FORECAST_DIR`` > default."""
    if forecast_dir is not None:
        return forecast_dir
    env = os.environ.get("SUPERFORECASTER_FORECAST_DIR")
    if env:
        return Path(env)
    return Path(DEFAULT_FORECAST_DIR)


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
    data_dir: Path | None = _DataDirOption,
    forecast_dir: Path | None = _ForecastDirOption,
    post_id: list[int] = _PostIdOption,
) -> None:
    """Run a baseline (no-retrieval) forecast pass over open tournament questions.

    All forecasts are dry runs in v0 — nothing is submitted anywhere. Each
    question costs one Claude call; the per-call token cost is logged.
    """
    _require_llm_cli()
    snap_store = SnapshotStore(_resolve_data_dir(data_dir))
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

    records: list[ForecastRecord] = []
    for snapshot in questions:
        record = run_forecast_pass(snapshot, client=client, config=config)
        records.append(record)
        cost = "n/a" if record.cost_usd is None else f"${record.cost_usd:.4f}"
        typer.echo(
            f"q{record.question_id} (post {record.post_id}): "
            f"p={record.probability:.2f} confidence={record.confidence} cost={cost}"
        )

    fc_store.append(records)
    total_cost = sum(r.cost_usd for r in records if r.cost_usd is not None)
    typer.echo(f"forecast: n={len(records)} total_cost=${total_cost:.4f} (all dry-run)")


@app.command("submit-tournament")
def submit_tournament(
    tournament: str = typer.Option(
        SANDBOX_TOURNAMENT,
        "--tournament",
        help="Tournament slug to forecast. Defaults to the unscored sandbox.",
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
    # Double gate: the scored tournament must never be entered by accident.
    if tournament == LIVE_TOURNAMENT and not live:
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
