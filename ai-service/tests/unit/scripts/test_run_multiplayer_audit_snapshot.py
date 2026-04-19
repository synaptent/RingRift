"""Tests for the standard multiplayer audit snapshot wrapper."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = (
    REPO_ROOT
    / "ai-service"
    / "scripts"
    / "experiments"
    / "run_multiplayer_audit_snapshot.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_multiplayer_audit_snapshot",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "name": "gh200-9",
                        "target_config": "square8_2p",
                        "work_dir": "data/minimal_loop_square8_2p",
                    },
                    {
                        "name": "gh200-10",
                        "target_config": "hex8_3p",
                        "work_dir": "data/minimal_loop_hex8_3p",
                    },
                    {
                        "name": "gh200-12",
                        "target_config": "square8_3p",
                        "work_dir": "data/minimal_loop_square8_3p",
                    },
                ],
                "script_only_canaries": [
                    {
                        "target_config": "square8_4p",
                        "work_dir": "data/minimal_loop_square8_4p",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_discover_targets_uses_manifest_multiplayer_defaults(tmp_path: Path) -> None:
    module = _load_module()
    ai_service_root = tmp_path / "ai-service"
    manifest_path = tmp_path / "docs" / "data" / "training_fleet_manifest.json"
    _write_manifest(manifest_path)

    _write_metrics(
        ai_service_root / "data" / "minimal_loop_hex8_3p" / "metrics.jsonl",
        [],
    )
    _write_metrics(
        ai_service_root / "data" / "minimal_loop_square8_3p" / "metrics.jsonl",
        [],
    )

    targets, skipped = module.discover_targets(
        manifest_path=manifest_path,
        ai_service_root=ai_service_root,
    )

    assert [target.config for target in targets] == ["hex8_3p", "square8_3p"]
    assert [target.node for target in targets] == ["gh200-10", "gh200-12"]
    assert skipped == [
        {
            "config": "square8_4p",
            "source": "script_only_canary",
            "node": None,
            "metrics_path": str(
                ai_service_root / "data" / "minimal_loop_square8_4p" / "metrics.jsonl"
            ),
            "reason": "metrics_missing",
        }
    ]


def test_cli_writes_timestamped_snapshot(tmp_path: Path) -> None:
    ai_service_root = tmp_path / "ai-service"
    manifest_path = tmp_path / "docs" / "data" / "training_fleet_manifest.json"
    output_dir = tmp_path / "snapshots"
    _write_manifest(manifest_path)

    _write_metrics(
        ai_service_root / "data" / "minimal_loop_hex8_3p" / "metrics.jsonl",
        [
            {
                "iteration": 15,
                "promoted": False,
                "estimated_elo": 1510.0,
                "evaluation": {"decision": "reject", "win_rate": 0.24, "games_played": 50},
                "quality_gate": {
                    "warnings": [],
                    "details": {
                        "seat_fairness": {
                            "seat_wr": {"1": 0.24, "2": 0.25, "3": 0.23},
                            "selfplay_baseline_seat_wr": {"1": 0.31, "2": 0.34, "3": 0.35},
                            "expected_seat_wins": {"1": 3.7, "2": 4.1, "3": 4.2},
                            "chi_square_stat": 0.19,
                            "chi_square_p_value": 0.91,
                        }
                    },
                },
            }
        ],
    )
    _write_metrics(
        ai_service_root / "data" / "minimal_loop_square8_3p" / "metrics.jsonl",
        [
            {
                "iteration": 25,
                "promoted": True,
                "estimated_elo": 1679.1,
                "evaluation": {"decision": "promote", "win_rate": 0.355, "games_played": 400},
                "quality_gate": {
                    "warnings": [],
                    "details": {
                        "seat_fairness": {
                            "seat_wr": {"1": 0.36, "2": 0.34, "3": 0.37},
                            "selfplay_baseline_seat_wr": {"1": 0.33, "2": 0.33, "3": 0.34},
                            "expected_seat_wins": {"1": 47.5, "2": 47.5, "3": 49.0},
                            "chi_square_stat": 0.08,
                            "chi_square_p_value": 0.96,
                        }
                    },
                },
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(manifest_path),
            "--ai-service-root",
            str(ai_service_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    outputs = sorted(output_dir.glob("multiplayer_audit_*.json"))
    assert len(outputs) == 1

    artifact = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert artifact["selected_configs"] == ["hex8_3p", "square8_3p"]
    assert [report["source"] for report in artifact["reports"]] == [
        "manifest_node",
        "manifest_node",
    ]
    assert artifact["reports"][1]["corrected_promotion_deltas"] == [
        {
            "iteration": 25,
            "win_rate": 0.355,
            "corrected_elo_delta": 16.7,
        }
    ]
    assert artifact["skipped_targets"] == [
        {
            "config": "square8_4p",
            "source": "script_only_canary",
            "node": None,
            "metrics_path": str(
                ai_service_root / "data" / "minimal_loop_square8_4p" / "metrics.jsonl"
            ),
            "reason": "metrics_missing",
        }
    ]
