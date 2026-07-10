"""CLI tests for `experiment-status`, with and without --refresh (no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from superforecaster import cli
from superforecaster.forecast_store import ForecastStore
from superforecaster.forecaster import ForecastRecord
from superforecaster.store import SnapshotStore

runner = CliRunner()

ON = datetime(2026, 7, 10, 9, 0, tzinfo=UTC)

MANIFEST = """
id = "001"
title = "Ensemble vs baseline"
status = "frozen"
frozen = 2026-07-10
budget_usd = 40.0
asknews_credit_budget = 250
question_ids = [201, 202]

[arms.A]
label = "single"
kind = "single"
method = "checklist-a"
model = "z-ai/glm-5.2"

[arms.B]
label = "ensemble"
kind = "ensemble-pooled"
method = "ensemble-b"
"""


def _record(qid: int, *, method: str, model: str, probability: float) -> ForecastRecord:
    return ForecastRecord(
        created_at=ON,
        question_id=qid,
        post_id=qid - 100,
        title=f"Q{qid}",
        probability=probability,
        raw_probability=probability,
        confidence="medium",
        rationale="r",
        reference_class="c",
        model=model,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.02,
        method=method,
    )


def _write_manifest(tmp_path: Path) -> Path:
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    (exp_dir / "001-x.toml").write_text(MANIFEST, encoding="utf-8")
    return exp_dir


def _seed_forecasts(forecast_dir: Path) -> None:
    ForecastStore(forecast_dir).append(
        [
            _record(201, method="checklist-a", model="z-ai/glm-5.2", probability=0.8),
            _record(201, method="ensemble-b", model="ensemble-b", probability=0.6),
            _record(202, method="checklist-a", model="z-ai/glm-5.2", probability=0.3),
        ]
    )


def _open_snapshot(qid: int, post_id: int) -> dict[str, Any]:
    return {
        "id": post_id,
        "title": f"Q{qid}",
        "status": "open",
        "question": {
            "id": qid,
            "type": "binary",
            "title": f"Q{qid}",
            "status": "open",
            "resolution": None,
        },
    }


def test_experiment_status_no_refresh(tmp_path: Path) -> None:
    exp_dir = _write_manifest(tmp_path)
    data_dir = tmp_path / "snapshots"
    forecast_dir = tmp_path / "forecasts"
    _seed_forecasts(forecast_dir)

    # Snapshot store: q201 resolved YES, q202 still open.
    store = SnapshotStore(data_dir)
    from superforecaster.models import QuestionSnapshot

    store.append(
        [
            QuestionSnapshot.from_api_post(
                {
                    "id": 101,
                    "title": "Q201",
                    "status": "resolved",
                    "question": {
                        "id": 201,
                        "type": "binary",
                        "title": "Q201",
                        "status": "resolved",
                        "resolution": "yes",
                    },
                },
                ON,
            ),
            QuestionSnapshot.from_api_post(_open_snapshot(202, 102), ON),
        ]
    )

    result = runner.invoke(
        cli.app,
        [
            "experiment-status",
            "--data-dir",
            str(data_dir),
            "--forecast-dir",
            str(forecast_dir),
            "--experiments-dir",
            str(exp_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Experiment 001" in result.output
    assert "coverage=2/2" in result.output  # arm A forecast both pinned
    assert "coverage=1/2" in result.output  # arm B forecast only q201
    assert "resolved: 1/2" in result.output
    # q201 resolved YES: A said 0.8 (brier .04), B said 0.6 (brier .16) -> +0.12
    assert "+0.1200" in result.output


def test_experiment_status_no_manifests(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["experiment-status", "--experiments-dir", str(tmp_path / "empty")],
    )
    assert result.exit_code == 0, result.output
    assert "no experiment manifests found" in result.output


class _FakeClient:
    """Serves a resolved post for the refresh path; no network."""

    def __init__(self) -> None:
        self.closed = False

    def get_post(self, post_id: int) -> dict[str, Any]:
        return {
            "id": post_id,
            "title": "Q201",
            "status": "resolved",
            "question": {
                "id": 201,
                "type": "binary",
                "title": "Q201",
                "status": "resolved",
                "resolution": "yes",
            },
        }

    def close(self) -> None:
        self.closed = True


def test_experiment_status_refresh_rechecks_pinned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exp_dir = _write_manifest(tmp_path)
    data_dir = tmp_path / "snapshots"
    forecast_dir = tmp_path / "forecasts"
    _seed_forecasts(forecast_dir)

    # Store starts with q201 OPEN (post 101); refresh should flip it to resolved.
    from superforecaster.models import QuestionSnapshot

    SnapshotStore(data_dir).append([QuestionSnapshot.from_api_post(_open_snapshot(201, 101), ON)])

    fake = _FakeClient()
    monkeypatch.setattr(cli, "_build_client", lambda token: fake)
    monkeypatch.setenv("METACULUS_TOKEN", "test-token")

    result = runner.invoke(
        cli.app,
        [
            "experiment-status",
            "--refresh",
            "--data-dir",
            str(data_dir),
            "--forecast-dir",
            str(forecast_dir),
            "--experiments-dir",
            str(exp_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "newly_resolved=1" in result.output
    assert "resolved: 1/2" in result.output
    assert fake.closed
