"""Tests for the settings/model/pricing config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from superforecaster.config import (
    DataSettings,
    EnsembleMemberConfig,
    RetrievalSettings,
    Settings,
    TaskModelConfig,
    TournamentSettings,
    load_ensemble_settings,
    load_retrieval_settings,
    load_settings,
    load_task_config,
    price_for,
)

_TOML = """\
[tasks.forecast_pass]
model = "claude-sonnet-5"
max_tokens = 4096

[tasks.with_effort]
model = "claude-opus-5"
max_tokens = 8192
effort = "high"

[tasks.bad_effort]
model = "claude-sonnet-5"
max_tokens = 100
effort = "turbo"

[pricing.claude-sonnet-5]
input_per_mtok = 3.0
output_per_mtok = 15.0
"""


_FULL_TOML = """\
[data]
snapshot_dir = "./snaps"
forecast_dir = "./casts"

[tournament]
live = "scored-2027"
sandbox = "playpen"

[tasks.forecast_pass]
model = "claude-sonnet-5"
max_tokens = 4096
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text(_TOML, encoding="utf-8")
    return path


def test_load_task_config_reads_fields(config_path: Path) -> None:
    cfg = load_task_config("forecast_pass", config_path)
    assert cfg == TaskModelConfig(model="claude-sonnet-5", max_tokens=4096, effort=None)


def test_load_task_config_effort_passthrough(config_path: Path) -> None:
    cfg = load_task_config("with_effort", config_path)
    assert cfg.effort == "high"


def test_load_task_config_rejects_bad_effort(config_path: Path) -> None:
    with pytest.raises(ValueError, match="effort"):
        load_task_config("bad_effort", config_path)


def test_load_task_config_missing_task(config_path: Path) -> None:
    with pytest.raises(KeyError):
        load_task_config("nope", config_path)


def test_env_override(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERFORECASTER_CONFIG", str(config_path))
    cfg = load_task_config("forecast_pass")  # no explicit path -> uses env
    assert cfg.model == "claude-sonnet-5"


def test_price_for_math(config_path: Path) -> None:
    # 1M input @ $3 + 1M output @ $15 = $18.
    assert price_for("claude-sonnet-5", 1_000_000, 1_000_000, config_path) == 18.0
    # Small counts: 1000 in, 500 out -> 1000*3/1e6 + 500*15/1e6 = 0.003 + 0.0075.
    got = price_for("claude-sonnet-5", 1000, 500, config_path)
    assert got is not None
    assert got == pytest.approx(0.0105)


def test_price_for_unpriced_model(config_path: Path) -> None:
    assert price_for("claude-opus-5", 1000, 1000, config_path) is None


def test_repo_config_loads() -> None:
    """The committed engine/config/settings.toml must parse and define forecast_pass."""
    cfg = load_task_config("forecast_pass")
    assert cfg.model
    assert cfg.max_tokens > 0


# --- Settings loader --------------------------------------------------------


def test_load_settings_reads_sections(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(_FULL_TOML, encoding="utf-8")
    settings = load_settings(path)
    assert settings == Settings(
        data=DataSettings(snapshot_dir=Path("./snaps"), forecast_dir=Path("./casts")),
        tournament=TournamentSettings(live="scored-2027", sandbox="playpen"),
    )


def test_load_settings_defaults_when_sections_missing(config_path: Path) -> None:
    # _TOML has [tasks] and [pricing] but no [data]/[tournament] -> all defaults.
    settings = load_settings(config_path)
    assert settings.data.snapshot_dir == Path("./data/snapshots")
    assert settings.data.forecast_dir == Path("./data/forecasts")
    assert settings.tournament.live == "summer-futureeval-2026"
    assert settings.tournament.sandbox == "bot-testing-area"


def test_load_settings_partial_section_fills_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[data]\nsnapshot_dir = "./only-snaps"\n', encoding="utf-8")
    settings = load_settings(path)
    assert settings.data.snapshot_dir == Path("./only-snaps")
    # forecast_dir absent -> default.
    assert settings.data.forecast_dir == Path("./data/forecasts")


def test_load_settings_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(_FULL_TOML, encoding="utf-8")
    monkeypatch.setenv("SUPERFORECASTER_CONFIG", str(path))
    settings = load_settings()  # no explicit path -> uses env
    assert settings.tournament.sandbox == "playpen"


def test_load_settings_repo_default_resolves() -> None:
    """The committed settings.toml resolves via the default path and parses."""
    settings = load_settings()
    assert settings.tournament.live
    assert settings.data.snapshot_dir


# --- Retrieval settings loader ----------------------------------------------


def test_load_retrieval_settings_reads_section(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(
        "[retrieval]\n"
        'news_provider = "asknews"\n'
        'general_provider = "tavily"\n'
        "max_results = 7\n"
        "max_fetches_per_question = 3\n"
        "asknews_monthly_credit_cap = 500\n"
        "extract_min_chars = 300\n"
        "extract_min_words = 50\n"
        "timeout_seconds = 12.5\n",
        encoding="utf-8",
    )
    r = load_retrieval_settings(path)
    assert r == RetrievalSettings(
        news_provider="asknews",
        general_provider="tavily",
        max_results=7,
        max_fetches_per_question=3,
        asknews_monthly_credit_cap=500,
        extract_min_chars=300,
        extract_min_words=50,
        timeout_seconds=12.5,
    )


def test_load_retrieval_settings_defaults_when_missing(config_path: Path) -> None:
    # _TOML has no [retrieval] section -> all defaults.
    assert load_retrieval_settings(config_path) == RetrievalSettings()


def test_load_retrieval_settings_partial_fills_missing(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("[retrieval]\nmax_results = 9\n", encoding="utf-8")
    r = load_retrieval_settings(path)
    assert r.max_results == 9
    assert r.asknews_monthly_credit_cap == 900  # default preserved


def test_load_retrieval_settings_repo_default_resolves() -> None:
    """The committed settings.toml [retrieval] section parses via default path."""
    r = load_retrieval_settings()
    assert r.news_provider == "asknews"
    assert r.asknews_monthly_credit_cap > 0


# --- Ensemble settings loader -----------------------------------------------


def test_load_ensemble_settings_defaults_when_missing(config_path: Path) -> None:
    # _TOML has no [ensemble] section -> all defaults, incl. the 5-member roster.
    e = load_ensemble_settings(config_path)
    assert e.extremization_mode == "fixed"
    assert e.extremization_coefficient == 1.0
    assert e.agreement_max_coefficient == 1.3
    assert e.confidence_high_max_spread == 0.4
    assert e.confidence_low_min_spread == 1.2
    assert len(e.members) == 5
    # The default roster is dominated by the cheap model on all three frames.
    frames = {m.frame for m in e.members}
    assert frames == {"outside-first", "inside-first", "status-quo"}


def test_load_ensemble_settings_parses_explicit_section(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(
        "[ensemble]\n"
        'extremization_mode = "agreement"\n'
        "extremization_coefficient = 1.1\n"
        "agreement_min_coefficient = 0.9\n"
        "agreement_max_coefficient = 1.4\n"
        "confidence_high_max_spread = 0.3\n"
        "confidence_low_min_spread = 1.0\n"
        "\n"
        "[[ensemble.members]]\n"
        'frame = "inside-first"\n'
        'provider = "openrouter"\n'
        'model = "z-ai/glm-5.2"\n'
        'effort = "low"\n'
        "max_tokens = 8192\n"
        "\n"
        "[[ensemble.members]]\n"
        'frame = "status-quo"\n'
        'model = "claude-sonnet-5"\n'
        "max_tokens = 4096\n",
        encoding="utf-8",
    )
    e = load_ensemble_settings(path)
    assert e.extremization_mode == "agreement"
    assert e.extremization_coefficient == pytest.approx(1.1)
    assert e.agreement_min_coefficient == pytest.approx(0.9)
    assert e.agreement_max_coefficient == pytest.approx(1.4)
    assert e.confidence_high_max_spread == pytest.approx(0.3)
    assert e.confidence_low_min_spread == pytest.approx(1.0)
    assert e.members == (
        EnsembleMemberConfig(
            frame="inside-first",
            model="z-ai/glm-5.2",
            max_tokens=8192,
            effort="low",
            provider="openrouter",
        ),
        # provider/effort absent -> defaults (claude-cli, no effort).
        EnsembleMemberConfig(frame="status-quo", model="claude-sonnet-5", max_tokens=4096),
    )


def test_load_ensemble_settings_rejects_bad_mode(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[ensemble]\nextremization_mode = "turbo"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="extremization_mode"):
        load_ensemble_settings(path)


def test_load_ensemble_settings_rejects_member_without_frame(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("[[ensemble.members]]\nmodel = 'x'\nmax_tokens = 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frame"):
        load_ensemble_settings(path)


def test_load_ensemble_settings_repo_default_resolves() -> None:
    """The committed settings.toml [ensemble] section parses via the default path."""
    e = load_ensemble_settings()
    assert len(e.members) >= 2
    assert all(m.frame for m in e.members)
