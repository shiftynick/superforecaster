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
from datetime import UTC, datetime
from pathlib import Path

import typer

from .metaculus import MetaculusClient
from .models import QuestionSnapshot, SkipPost
from .store import SnapshotStore

app = typer.Typer(
    add_completion=False,
    help="Scrape and inspect Metaculus question snapshots. Run from the repo root.",
)

_DEFAULT_DATA_DIR = "./data/snapshots"
_TOKEN_HELP = (
    "Metaculus API token is missing. Create a free account at metaculus.com, open "
    "Settings > API, copy your token, then set it, e.g.\n"
    "  PowerShell:  $env:METACULUS_TOKEN = '<token>'\n"
    "  bash:        export METACULUS_TOKEN=<token>\n"
    "or add a line METACULUS_TOKEN=<token> to engine/.env."
)


def _load_dotenv(path: Path) -> None:
    """Read simple ``KEY=VALUE`` lines from ``path`` into ``os.environ``.

    Blank lines and ``#`` comments are ignored. Existing environment variables
    win, so a real shell export always overrides the file. No dependency needed.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


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


if __name__ == "__main__":  # pragma: no cover
    app()
