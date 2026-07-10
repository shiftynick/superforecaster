"""Load all non-secret engine settings from ``engine/config/settings.toml``.

This one file is the engine's control panel — the config-over-code rule in
practice. The forecasting code never names a model, token count, data directory,
or tournament slug itself; it asks the loaders here for them. Secrets (API
tokens) deliberately live elsewhere, in ``engine/.env``.

Four things live in that TOML file and are read here:

- **Data locations** (``[data]``): where snapshots and forecast records are
  stored. :func:`load_settings` returns these as :class:`Path` values.
- **Tournaments** (``[tournament]``): the scored ``live`` slug (a real,
  money-scored entry) and the unscored ``sandbox`` slug. Also via
  :func:`load_settings`.
- **Task settings** (``[tasks.<name>]``): which model a task calls, its token
  budget, and an optional "effort" level. Read by :func:`load_task_config`.
- **Pricing** (``[pricing.<model>]``): dollars per million tokens, used only to
  turn a call's token counts into a logged cost. :func:`price_for` does that
  arithmetic and returns ``None`` when a model has no price on file.
- **Retrieval** (``[retrieval]``): provider roles, budgets, and extraction
  quality gates for the research stack. Read by :func:`load_retrieval_settings`.

The config file is found relative to this source file by default
(``src/superforecaster/config.py`` -> ``engine/config/settings.toml``), but the
``SUPERFORECASTER_CONFIG`` environment variable overrides that, and every
function takes an explicit ``path`` for tests. Missing sections fall back to the
built-in defaults below, so a minimal file (or none of a given section) works.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Effort = Literal["low", "medium", "high"]
_VALID_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high"})

# Built-in fallbacks used when a section (or the whole file) is absent. These are
# the last link in the settings precedence chain and keep a minimal config valid.
_DEFAULT_SNAPSHOT_DIR = "./data/snapshots"
_DEFAULT_FORECAST_DIR = "./data/forecasts"
_DEFAULT_LIVE_TOURNAMENT = "summer-futureeval-2026"
_DEFAULT_SANDBOX_TOURNAMENT = "bot-testing-area"


@dataclass(frozen=True)
class TaskModelConfig:
    """The model settings for one task type.

    Attributes:
        model: The model id or alias to call (e.g. ``"claude-sonnet-5"``).
        max_tokens: Maximum tokens the response may use — required by the API.
        effort: Optional reasoning-effort hint (``"low"``/``"medium"``/``"high"``)
            passed straight to the API. ``None`` means "use the API default".
        provider: Which LLM backend runs the call. Defaults to ``"claude-cli"``
            (the Claude Code CLI on the owner's subscription). Additional
            providers (a Codex CLI, OpenRouter) will be added over time; the
            forecaster dispatches on this via ``llm.build_llm_client``.
    """

    model: str
    max_tokens: int
    effort: Effort | None = None
    provider: str = "claude-cli"


@dataclass(frozen=True)
class DataSettings:
    """Where the engine reads and writes its files.

    Attributes:
        snapshot_dir: Directory of question snapshots (the scraper's output).
        forecast_dir: Directory of forecast records.
    """

    snapshot_dir: Path
    forecast_dir: Path


@dataclass(frozen=True)
class TournamentSettings:
    """The Metaculus tournament slugs the engine forecasts against.

    Attributes:
        live: The scored (money) tournament — a run against it requires ``--live``.
        sandbox: The unscored test tournament, safe to submit to freely.
    """

    live: str
    sandbox: str


@dataclass(frozen=True)
class RetrievalSettings:
    """Tunables for the research/retrieval stack (``[retrieval]`` section).

    Every field is config-over-code: the search and extraction layers read these
    rather than hard-coding provider names, budgets, or quality thresholds.

    Attributes:
        news_provider: Which search client fills the *news* role (``"asknews"``).
        general_provider: Which search client fills the *general web* role
            (``"tavily"``).
        max_results: Default number of results to request per search call.
        max_fetches_per_question: Cap on content fetches per forecast question
            (a budget guard used by the slice-2 research pipeline).
        asknews_monthly_credit_cap: Hard monthly cap on AskNews credits; the
            client refuses to call once this would be exceeded. Default 900
            leaves headroom under the 1,000-credit free tier.
        extract_min_chars: Quality gate — extracted text shorter than this many
            characters counts as an extraction failure (fall to the next tier).
        extract_min_words: Quality gate — extracted text with fewer than this
            many words counts as a failure.
        max_doc_chars: Cap on the extracted-document text stored in a research
            bundle (and handed to the prompt). Longer bodies are truncated so
            forecast records stay reviewable.
        timeout_seconds: Per-request HTTP timeout for search and extraction.
        reuse_research_max_age_hours: How many hours a cached research bundle
            stays reusable. The ``forecast`` command reuses a saved bundle whose
            ``retrieved_at`` is within this window instead of re-researching
            (spending search credits). ``0`` disables reuse: always re-research.
    """

    news_provider: str = "asknews"
    general_provider: str = "tavily"
    max_results: int = 5
    max_fetches_per_question: int = 8
    asknews_monthly_credit_cap: int = 900
    extract_min_chars: int = 500
    extract_min_words: int = 80
    max_doc_chars: int = 8000
    timeout_seconds: float = 20.0
    reuse_research_max_age_hours: float = 6.0


@dataclass(frozen=True)
class Settings:
    """The non-secret engine settings, grouped by section."""

    data: DataSettings
    tournament: TournamentSettings


def _default_config_path() -> Path:
    """Locate ``engine/config/settings.toml`` next to this package.

    Precedence: the ``SUPERFORECASTER_CONFIG`` environment variable, then the
    file resolved relative to this source file. This file lives at
    ``engine/src/superforecaster/config.py``; going up three parents lands on
    ``engine/``, where ``config/settings.toml`` sits.
    """
    env = os.environ.get("SUPERFORECASTER_CONFIG")
    if env:
        return Path(env)
    engine_dir = Path(__file__).resolve().parents[2]
    return engine_dir / "config" / "settings.toml"


def _load_toml(path: Path | None) -> dict[str, object]:
    """Parse the config TOML, defaulting to the resolved standard location."""
    resolved = path if path is not None else _default_config_path()
    with resolved.open("rb") as fh:
        return tomllib.load(fh)


def load_task_config(task: str, path: Path | None = None) -> TaskModelConfig:
    """Return the :class:`TaskModelConfig` for ``task`` from the config file.

    Args:
        task: The task-table name, e.g. ``"forecast_pass"``.
        path: Optional explicit config path (defaults to the resolved location).

    Raises:
        KeyError: If the task is not present in the config.
        ValueError: If required fields are missing or the effort value is not one
            of ``"low"``/``"medium"``/``"high"``.
    """
    data = _load_toml(path)
    tasks = data.get("tasks")
    if not isinstance(tasks, dict) or task not in tasks:
        raise KeyError(f"No [tasks.{task}] table in config")
    table = tasks[task]
    if not isinstance(table, dict):
        raise ValueError(f"[tasks.{task}] is not a table")

    model = table.get("model")
    max_tokens = table.get("max_tokens")
    # Empty string is allowed: it means "use the provider CLI's own default model".
    if not isinstance(model, str):
        raise ValueError(f"[tasks.{task}] is missing a 'model' string")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ValueError(f"[tasks.{task}] is missing an integer 'max_tokens'")

    effort_raw = table.get("effort")
    effort: Effort | None
    if effort_raw is None:
        effort = None
    elif isinstance(effort_raw, str) and effort_raw in _VALID_EFFORTS:
        # Narrow to the Literal type for mypy.
        effort = effort_raw  # type: ignore[assignment]
    else:
        raise ValueError(
            f"[tasks.{task}] effort must be one of low/medium/high, got {effort_raw!r}"
        )

    provider_raw = table.get("provider")
    if provider_raw is None:
        provider = "claude-cli"
    elif isinstance(provider_raw, str) and provider_raw:
        provider = provider_raw
    else:
        raise ValueError(f"[tasks.{task}] provider must be a non-empty string if set")

    return TaskModelConfig(model=model, max_tokens=max_tokens, effort=effort, provider=provider)


def _str_field(table: object, key: str, default: str) -> str:
    """Read a non-empty string ``key`` from a TOML table, else return ``default``."""
    if isinstance(table, dict):
        value = table.get(key)
        if isinstance(value, str) and value:
            return value
    return default


def _int_field(table: object, key: str, default: int) -> int:
    """Read an integer ``key`` from a TOML table, else return ``default``.

    ``bool`` is rejected (it is an ``int`` subclass in Python but never a valid
    count here), matching the strictness of :func:`load_task_config`.
    """
    if isinstance(table, dict):
        value = table.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return default


def _float_field(table: object, key: str, default: float) -> float:
    """Read a numeric ``key`` from a TOML table as ``float``, else ``default``."""
    if isinstance(table, dict):
        value = table.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return default


def load_retrieval_settings(path: Path | None = None) -> RetrievalSettings:
    """Return the :class:`RetrievalSettings` from the ``[retrieval]`` section.

    Every field is optional: a missing ``[retrieval]`` table (or any missing
    field within it) falls back to the built-in defaults on
    :class:`RetrievalSettings`, so a minimal config still loads. The
    ``SUPERFORECASTER_CONFIG`` env override and default-path resolution match
    :func:`load_settings`.

    Args:
        path: Optional explicit config path (defaults to the resolved location).
    """
    table = _load_toml(path).get("retrieval")
    d = RetrievalSettings()  # defaults
    return RetrievalSettings(
        news_provider=_str_field(table, "news_provider", d.news_provider),
        general_provider=_str_field(table, "general_provider", d.general_provider),
        max_results=_int_field(table, "max_results", d.max_results),
        max_fetches_per_question=_int_field(
            table, "max_fetches_per_question", d.max_fetches_per_question
        ),
        asknews_monthly_credit_cap=_int_field(
            table, "asknews_monthly_credit_cap", d.asknews_monthly_credit_cap
        ),
        extract_min_chars=_int_field(table, "extract_min_chars", d.extract_min_chars),
        extract_min_words=_int_field(table, "extract_min_words", d.extract_min_words),
        max_doc_chars=_int_field(table, "max_doc_chars", d.max_doc_chars),
        timeout_seconds=_float_field(table, "timeout_seconds", d.timeout_seconds),
        reuse_research_max_age_hours=_float_field(
            table, "reuse_research_max_age_hours", d.reuse_research_max_age_hours
        ),
    )


def load_settings(path: Path | None = None) -> Settings:
    """Return the engine :class:`Settings` from the config file.

    Every section is optional: a missing ``[data]`` or ``[tournament]`` table (or
    a missing field within one, or the whole file having neither) falls back to
    the built-in defaults, so a minimal config still loads. The
    ``SUPERFORECASTER_CONFIG`` env override and default-path resolution match
    :func:`load_task_config` — both read the same ``settings.toml``.

    Args:
        path: Optional explicit config path (defaults to the resolved location).
    """
    data_raw = _load_toml(path)
    data_table = data_raw.get("data")
    tournament_table = data_raw.get("tournament")

    data = DataSettings(
        snapshot_dir=Path(_str_field(data_table, "snapshot_dir", _DEFAULT_SNAPSHOT_DIR)),
        forecast_dir=Path(_str_field(data_table, "forecast_dir", _DEFAULT_FORECAST_DIR)),
    )
    tournament = TournamentSettings(
        live=_str_field(tournament_table, "live", _DEFAULT_LIVE_TOURNAMENT),
        sandbox=_str_field(tournament_table, "sandbox", _DEFAULT_SANDBOX_TOURNAMENT),
    )
    return Settings(data=data, tournament=tournament)


def price_for(
    model: str,
    input_tokens: int,
    output_tokens: int,
    path: Path | None = None,
) -> float | None:
    """Compute the dollar cost of a call, or ``None`` if the model is unpriced.

    Pricing is per million tokens, split into input and output rates. A model
    with no ``[pricing.<model>]`` table returns ``None`` so callers can record
    "cost unknown" rather than guessing.
    """
    data = _load_toml(path)
    pricing = data.get("pricing")
    if not isinstance(pricing, dict):
        return None
    table = pricing.get(model)
    if not isinstance(table, dict):
        return None
    input_rate = table.get("input_per_mtok")
    output_rate = table.get("output_per_mtok")
    if not isinstance(input_rate, int | float) or not isinstance(output_rate, int | float):
        return None
    return (input_tokens * float(input_rate) + output_tokens * float(output_rate)) / 1_000_000
