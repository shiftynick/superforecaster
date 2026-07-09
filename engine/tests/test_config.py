"""Tests for the settings/model/pricing config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from superforecaster.config import (
    DataSettings,
    Settings,
    TaskModelConfig,
    TournamentSettings,
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
