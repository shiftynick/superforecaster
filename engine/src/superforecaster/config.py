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

ExtremizationMode = Literal["fixed", "agreement"]
_VALID_EXTREMIZATION_MODES: frozenset[str] = frozenset({"fixed", "agreement"})

CombinationMode = Literal["guarded-product", "llm-synthesis"]
_VALID_COMBINATION_MODES: frozenset[str] = frozenset({"guarded-product", "llm-synthesis"})

# Built-in fallbacks used when a section (or the whole file) is absent. These are
# the last link in the settings precedence chain and keep a minimal config valid.
_DEFAULT_SNAPSHOT_DIR = "./data/snapshots"
_DEFAULT_FORECAST_DIR = "./data/forecasts"
_DEFAULT_LIVE_TOURNAMENT = "summer-futureeval-2026"
_DEFAULT_SANDBOX_TOURNAMENT = "bot-testing-area"
_DEFAULT_WEB_RESOLUTION_DIR = "./data/web-resolutions"
_DEFAULT_BUNDLE_EVAL_DIR = "./data/bundle-evals"


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
class EnsembleMemberConfig:
    """One member of the diverse ensemble: a reasoning frame plus its model.

    The model fields mirror :class:`TaskModelConfig` (a member is just a forecast
    pass with a chosen frame), and :meth:`task_config` hands back exactly that so
    the ensemble runner can drive the existing ``run_forecast_pass`` machinery.

    Attributes:
        frame: The reasoning-frame key (see ``ensemble.FRAMES``) layered onto the
            shared checklist — the diversity axis that is not the model.
        model: The model id or alias to call (empty string = provider default).
        max_tokens: Maximum tokens the response may use.
        effort: Optional reasoning-effort hint passed through to the provider.
        provider: Which LLM backend runs this member's call.
    """

    frame: str
    model: str
    max_tokens: int
    effort: Effort | None = None
    provider: str = "claude-cli"

    def task_config(self) -> TaskModelConfig:
        """Return the :class:`TaskModelConfig` for driving this member's pass."""
        return TaskModelConfig(
            model=self.model,
            max_tokens=self.max_tokens,
            effort=self.effort,
            provider=self.provider,
        )


# The v1 default member set (used when ``[ensemble]`` omits its members): the
# three frames on the cheap GLM-5.2, plus the default outside-first frame on two
# other models to diversify across providers. Dominated by the cheap model.
_DEFAULT_ENSEMBLE_MEMBERS: tuple[EnsembleMemberConfig, ...] = (
    EnsembleMemberConfig("outside-first", "z-ai/glm-5.2", 16384, "low", "openrouter"),
    EnsembleMemberConfig("inside-first", "z-ai/glm-5.2", 16384, "low", "openrouter"),
    EnsembleMemberConfig("status-quo", "z-ai/glm-5.2", 16384, "low", "openrouter"),
    EnsembleMemberConfig("outside-first", "x-ai/grok-4.5", 16384, "low", "openrouter"),
    EnsembleMemberConfig("outside-first", "claude-sonnet-5", 4096, "low", "claude-cli"),
)


@dataclass(frozen=True)
class EnsembleSettings:
    """Tunables for the diverse ensemble (``[ensemble]`` section).

    Config-over-code: the ensemble runner reads pooling/extremization behaviour
    and its member roster from here rather than hard-coding any of it.

    Attributes:
        extremization_mode: ``"fixed"`` applies ``extremization_coefficient``
            directly; ``"agreement"`` derives the coefficient from member spread
            (see :func:`aggregation.agreement_scaled_coefficient`).
        extremization_coefficient: The coefficient used when mode is ``"fixed"``
            (``1.0`` = no extremization).
        agreement_min_coefficient: Coefficient at maximum member disagreement,
            used when mode is ``"agreement"``.
        agreement_max_coefficient: Coefficient at full member agreement, used when
            mode is ``"agreement"``.
        confidence_high_max_spread: Member-spread (std dev of log-odds) at or below
            which the ensemble reports ``"high"`` confidence.
        confidence_low_min_spread: Member-spread at or above which the ensemble
            reports ``"low"`` confidence (``"medium"`` between the two thresholds).
        members: The ensemble roster; defaults to :data:`_DEFAULT_ENSEMBLE_MEMBERS`.
    """

    extremization_mode: ExtremizationMode = "fixed"
    extremization_coefficient: float = 1.0
    agreement_min_coefficient: float = 1.0
    agreement_max_coefficient: float = 1.3
    confidence_high_max_spread: float = 0.4
    confidence_low_min_spread: float = 1.2
    members: tuple[EnsembleMemberConfig, ...] = _DEFAULT_ENSEMBLE_MEMBERS


@dataclass(frozen=True)
class PyramidSettings:
    """Tunables for candidate C, the decomposition pyramid (``[pyramid]`` section).

    Config-over-code: the pyramid parser and combination math read their
    structural caps, clip bounds, blend weight, and combination mode from here
    rather than hard-coding any of them. Every field is optional with the default
    shown; a missing ``[pyramid]`` section uses all defaults.

    Attributes:
        max_depth: Deepest level the tree may reach (root is depth 0). An internal
            node at or beyond this depth is pruned to a leaf. Default ``2`` — a
            testing default; expect to raise it for real runs.
        max_children: Most children any internal node may keep; the overflow is
            dropped (pruned). Default ``4``.
        max_conjuncts: Most multiplied conjuncts a ``conjunction`` node may keep —
            the guard against conjunctive error compounding. Default ``3``.
        leaf_clip_low: Lower bound each child probability is clipped to before it
            is multiplied. Default ``0.03``.
        leaf_clip_high: Upper bound for the same clip. Default ``0.97``.
        holistic_blend_weight: Weight placed on the holistic (direct) forecast when
            blending it with the structural estimate in log-odds space. Default
            ``0.5`` — the honest prior that we do not yet know which to trust.
        combination_mode: How internal nodes combine children. ``"guarded-product"``
            (v1) or the reserved ``"llm-synthesis"``.
        max_assumptions_per_node: Runaway-output guard (v2) — a node's surfaced
            assumption list is truncated to this many, with a prune note. Default
            ``8``. Surfacing is cheap, so this only stops pathological output.
        max_assumption_verifications: Budget cap (v2) on how many researchable
            assumptions are verified per question (one search + one verdict call
            each). Load-bearing researchable assumptions are verified first.
            Default ``6``.
        leaf_topup_max_searches: Budget cap (v2 §4) on how many evidence leaves,
            per question, may draw ONE extra targeted search when the shared
            bundle gives their key terms zero coverage. Verification has priority
            (it runs first); top-up uses whatever of this budget remains. Default
            ``2``. ``0`` disables leaf top-up.
        fragility_pull_weight: Weight (v2 §3) by which a leaf carrying ≥1
            *unverified load-bearing* assumption is pulled toward the maximum-
            uncertainty anchor (0.5) in log-odds space — the fragility graft. The
            pull is recomputable: leaves store their pre-pull probability, so an
            experiment can re-score with the pull off. Default ``0.15``; ``0.0``
            disables it.
        unverified_lb_low_cap: Third confidence signal (v2 §3) — this many or more
            unverified load-bearing assumptions caps the record's confidence at
            ``"low"``. Default ``2``.
        unverified_lb_medium_cap: This many or more unverified load-bearing
            assumptions caps the record's confidence at ``"medium"`` (the low cap
            takes precedence when both fire). Default ``1``.
    """

    max_depth: int = 2
    max_children: int = 4
    max_conjuncts: int = 3
    leaf_clip_low: float = 0.03
    leaf_clip_high: float = 0.97
    holistic_blend_weight: float = 0.5
    combination_mode: CombinationMode = "guarded-product"
    max_assumptions_per_node: int = 8
    max_assumption_verifications: int = 6
    leaf_topup_max_searches: int = 2
    fragility_pull_weight: float = 0.15
    unverified_lb_low_cap: int = 2
    unverified_lb_medium_cap: int = 1


@dataclass(frozen=True)
class DataSettings:
    """Where the engine reads and writes its files.

    Attributes:
        snapshot_dir: Directory of question snapshots (the scraper's output).
        forecast_dir: Directory of forecast records.
        web_resolution_dir: Directory of web-scraped resolution records (the
            gap-filler that reads outcomes off the public Metaculus pages).
        bundle_eval_dir: Directory of research-bundle evaluation records (the
            research-quality yardstick's rubric + pairwise verdicts).
    """

    snapshot_dir: Path
    forecast_dir: Path
    web_resolution_dir: Path = Path("./data/web-resolutions")
    bundle_eval_dir: Path = Path("./data/bundle-evals")


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

    effort = _parse_effort(table, f"[tasks.{task}]")
    provider = _parse_provider(table, f"[tasks.{task}]")
    return TaskModelConfig(model=model, max_tokens=max_tokens, effort=effort, provider=provider)


def _parse_effort(table: dict[str, object], label: str) -> Effort | None:
    """Read an optional ``effort`` from a TOML table, validated against the enum.

    ``None`` (absent) is allowed and means "use the provider default". Any other
    value must be one of ``low``/``medium``/``high``; ``label`` names the table in
    the error so a bad value points at the offending section.
    """
    effort_raw = table.get("effort")
    if effort_raw is None:
        return None
    if isinstance(effort_raw, str) and effort_raw in _VALID_EFFORTS:
        # Narrow to the Literal type for mypy.
        return effort_raw  # type: ignore[return-value]
    raise ValueError(f"{label} effort must be one of low/medium/high, got {effort_raw!r}")


def _parse_provider(table: dict[str, object], label: str) -> str:
    """Read an optional ``provider`` from a TOML table, defaulting to claude-cli."""
    provider_raw = table.get("provider")
    if provider_raw is None:
        return "claude-cli"
    if isinstance(provider_raw, str) and provider_raw:
        return provider_raw
    raise ValueError(f"{label} provider must be a non-empty string if set")


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


def _parse_ensemble_member(raw: object) -> EnsembleMemberConfig:
    """Build one :class:`EnsembleMemberConfig` from a ``[[ensemble.members]]`` table.

    Mirrors :func:`load_task_config`'s field strictness (empty ``model`` allowed =
    provider default; ``max_tokens`` required; effort/provider validated), plus a
    required non-empty ``frame`` naming the reasoning lens.
    """
    if not isinstance(raw, dict):
        raise ValueError("[[ensemble.members]] entries must be tables")
    frame = raw.get("frame")
    if not isinstance(frame, str) or not frame:
        raise ValueError("[[ensemble.members]] entry is missing a non-empty 'frame' string")
    model = raw.get("model")
    if not isinstance(model, str):
        raise ValueError("[[ensemble.members]] entry is missing a 'model' string")
    max_tokens = raw.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ValueError("[[ensemble.members]] entry is missing an integer 'max_tokens'")
    effort = _parse_effort(raw, "[[ensemble.members]]")
    provider = _parse_provider(raw, "[[ensemble.members]]")
    return EnsembleMemberConfig(
        frame=frame, model=model, max_tokens=max_tokens, effort=effort, provider=provider
    )


def load_ensemble_settings(path: Path | None = None) -> EnsembleSettings:
    """Return the :class:`EnsembleSettings` from the ``[ensemble]`` section.

    Every field is optional: a missing ``[ensemble]`` table (or any missing field
    within it) falls back to the built-in defaults on :class:`EnsembleSettings`,
    including the default member roster. When ``[[ensemble.members]]`` tables are
    present they replace the default roster entirely; when absent (or empty) the
    default roster is kept. The ``SUPERFORECASTER_CONFIG`` env override and
    default-path resolution match :func:`load_settings`.

    Args:
        path: Optional explicit config path (defaults to the resolved location).
    """
    table = _load_toml(path).get("ensemble")
    d = EnsembleSettings()  # defaults, including the default member roster
    if not isinstance(table, dict):
        return d

    mode_raw = table.get("extremization_mode")
    if mode_raw is None:
        mode: ExtremizationMode = d.extremization_mode
    elif isinstance(mode_raw, str) and mode_raw in _VALID_EXTREMIZATION_MODES:
        mode = mode_raw  # type: ignore[assignment]
    else:
        raise ValueError(
            f"[ensemble] extremization_mode must be one of fixed/agreement, got {mode_raw!r}"
        )

    members_raw = table.get("members")
    if isinstance(members_raw, list) and members_raw:
        members = tuple(_parse_ensemble_member(m) for m in members_raw)
    else:
        members = d.members

    return EnsembleSettings(
        extremization_mode=mode,
        extremization_coefficient=_float_field(
            table, "extremization_coefficient", d.extremization_coefficient
        ),
        agreement_min_coefficient=_float_field(
            table, "agreement_min_coefficient", d.agreement_min_coefficient
        ),
        agreement_max_coefficient=_float_field(
            table, "agreement_max_coefficient", d.agreement_max_coefficient
        ),
        confidence_high_max_spread=_float_field(
            table, "confidence_high_max_spread", d.confidence_high_max_spread
        ),
        confidence_low_min_spread=_float_field(
            table, "confidence_low_min_spread", d.confidence_low_min_spread
        ),
        members=members,
    )


def load_pyramid_settings(path: Path | None = None) -> PyramidSettings:
    """Return the :class:`PyramidSettings` from the ``[pyramid]`` section.

    Every field is optional: a missing ``[pyramid]`` table (or any missing field
    within it) falls back to the built-in defaults on :class:`PyramidSettings`,
    so a minimal config still loads. Follows :func:`load_ensemble_settings`'
    pattern. The ``SUPERFORECASTER_CONFIG`` env override and default-path
    resolution match :func:`load_settings`.

    Args:
        path: Optional explicit config path (defaults to the resolved location).

    Raises:
        ValueError: If ``combination_mode`` is present but not one of
            ``guarded-product`` / ``llm-synthesis``.
    """
    table = _load_toml(path).get("pyramid")
    d = PyramidSettings()  # defaults
    if not isinstance(table, dict):
        return d

    mode_raw = table.get("combination_mode")
    if mode_raw is None:
        mode: CombinationMode = d.combination_mode
    elif isinstance(mode_raw, str) and mode_raw in _VALID_COMBINATION_MODES:
        mode = mode_raw  # type: ignore[assignment]
    else:
        raise ValueError(
            "[pyramid] combination_mode must be one of guarded-product/llm-synthesis, "
            f"got {mode_raw!r}"
        )

    return PyramidSettings(
        max_depth=_int_field(table, "max_depth", d.max_depth),
        max_children=_int_field(table, "max_children", d.max_children),
        max_conjuncts=_int_field(table, "max_conjuncts", d.max_conjuncts),
        leaf_clip_low=_float_field(table, "leaf_clip_low", d.leaf_clip_low),
        leaf_clip_high=_float_field(table, "leaf_clip_high", d.leaf_clip_high),
        holistic_blend_weight=_float_field(table, "holistic_blend_weight", d.holistic_blend_weight),
        combination_mode=mode,
        max_assumptions_per_node=_int_field(
            table, "max_assumptions_per_node", d.max_assumptions_per_node
        ),
        max_assumption_verifications=_int_field(
            table, "max_assumption_verifications", d.max_assumption_verifications
        ),
        leaf_topup_max_searches=_int_field(
            table, "leaf_topup_max_searches", d.leaf_topup_max_searches
        ),
        fragility_pull_weight=_float_field(table, "fragility_pull_weight", d.fragility_pull_weight),
        unverified_lb_low_cap=_int_field(table, "unverified_lb_low_cap", d.unverified_lb_low_cap),
        unverified_lb_medium_cap=_int_field(
            table, "unverified_lb_medium_cap", d.unverified_lb_medium_cap
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
        web_resolution_dir=Path(
            _str_field(data_table, "web_resolution_dir", _DEFAULT_WEB_RESOLUTION_DIR)
        ),
        bundle_eval_dir=Path(_str_field(data_table, "bundle_eval_dir", _DEFAULT_BUNDLE_EVAL_DIR)),
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
