"""Tests for experiment evidence status generation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "refresh_experiment_status.py"
EXPERIMENT_STATUS = REPO_ROOT / "docs" / "data" / "experiment_status.json"
RESULTS_SNAPSHOT = REPO_ROOT / "docs" / "data" / "results_snapshot.json"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("refresh_experiment_status", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_experiment_status_includes_claim_sources_and_seat_wr(tmp_path: Path) -> None:
    module = _load_script_module()

    fleet_manifest = tmp_path / "fleet.json"
    fleet_manifest.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "name": "gh200-test",
                        "role": "trainer",
                        "target_config": "square8_3p",
                        "work_dir": "data/minimal_loop_square8_3p",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_snapshot = tmp_path / "results.json"
    results_snapshot.write_text(
        json.dumps(
            {
                "headline": [
                    {
                        "config": "square8_3p",
                        "start_elo": 1500.0,
                        "best_elo": 1534.9,
                        "promotions": 1,
                    }
                ],
                "current_experiments": [],
            }
        ),
        encoding="utf-8",
    )
    training_status = tmp_path / "training_status.json"
    training_status.write_text(
        json.dumps(
            [
                {
                    "node": "gh200-test",
                    "config": "square8_3p",
                    "process_alive": True,
                    "iteration": 24,
                    "elo": 1534.9,
                    "promotions": 1,
                    "latest_metrics": {
                        "iteration": 23,
                        "estimated_elo": 1534.9,
                        "total_promotions": 1,
                        "evaluation": {
                            "decision": "reject",
                            "win_rate": 0.3,
                            "games_played": 50,
                            "seat_wr": {"1": 0.58, "2": 0.22, "3": 0.28},
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    artifact = module.build_experiment_status(
        fleet_manifest_path=fleet_manifest,
        results_snapshot_path=results_snapshot,
        training_status_path=training_status,
        observed_at="2026-04-17T00:00:00+00:00",
    )

    assert artifact["generated_at"] == "2026-04-17T00:00:00+00:00"
    assert artifact["headline_claims"][0]["field"] == "headline[0].best_elo"
    experiment = artifact["experiments"][0]
    assert experiment["status"] == "running"
    assert experiment["latest_decision"] == "reject"
    assert experiment["seat_fairness"]["seat_wr"] == {"1": 0.58, "2": 0.22, "3": 0.28}
    assert experiment["evidence"]["source_type"] == "training_status"


def test_refresh_experiment_status_cli_writes_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "experiment_status.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--output",
            str(output),
            "--observed-at",
            "2026-04-17T00:00:00+00:00",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["generated_at"] == "2026-04-17T00:00:00+00:00"
    assert data["headline_claims"]
    assert data["experiments"]


def test_checked_in_experiment_status_cites_snapshot_claims() -> None:
    assert EXPERIMENT_STATUS.exists()
    artifact = json.loads(EXPERIMENT_STATUS.read_text(encoding="utf-8"))
    snapshot = json.loads(RESULTS_SNAPSHOT.read_text(encoding="utf-8"))

    assert "external_status_inputs" not in artifact
    claim_ids = {claim["claim_id"] for claim in artifact["headline_claims"]}
    for item in snapshot["headline"]:
        config = item["config"]
        assert f"{config}_best_elo" in claim_ids
        assert f"{config}_promotions" in claim_ids

    for claim in artifact["headline_claims"]:
        assert claim["evidence"]["path"] == "docs/data/results_snapshot.json"
        assert claim["field"].startswith("headline[")

    for source_file in artifact["source_files"]:
        assert (REPO_ROOT / source_file).exists()
