"""Tests for the read-only multiplayer metrics audit script."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "ai-service" / "scripts" / "experiments" / "multiplayer_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("multiplayer_audit", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_metrics(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_infer_config_from_metrics_path() -> None:
    module = _load_module()
    metrics_path = Path("data/minimal_loop_square8_3p/metrics.jsonl")

    assert module._infer_config_from_path(metrics_path) == "square8_3p"


def test_promotion_delta_matches_corrected_multiplayer_formula() -> None:
    module = _load_module()

    delta = module._promotion_elo_delta(0.355, 3)

    assert delta == pytest.approx(16.68, abs=0.05)


def test_build_audit_report_classifies_iterations(tmp_path: Path) -> None:
    module = _load_module()
    metrics_path = tmp_path / "minimal_loop_square8_3p" / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True)
    _write_metrics(
        metrics_path,
        [
            {
                "iteration": 15,
                "promoted": False,
                "estimated_elo": 1679.1,
                "evaluation": {"decision": "reject", "win_rate": 0.24, "games_played": 50},
                "quality_gate": {
                    "warnings": [],
                    "details": {
                        "seat_fairness": {
                            "seat_wr": {"1": 0.25, "2": 0.24, "3": 0.22},
                            "selfplay_baseline_seat_wr": {"1": 0.33, "2": 0.33, "3": 0.34},
                            "expected_seat_wins": {"1": 3.0, "2": 3.0, "3": 3.1},
                            "chi_square_stat": 0.19,
                            "chi_square_p_value": 0.91,
                        }
                    },
                },
            },
            {
                "iteration": 25,
                "promoted": True,
                "estimated_elo": 1317.9,
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
            },
            {
                "iteration": 26,
                "promoted": False,
                "estimated_elo": 1334.0,
                "evaluation": {"decision": "reject", "win_rate": 0.28, "games_played": 100},
                "quality_gate": {
                    "warnings": [
                        "SEAT_WR_IMBALANCE: per-seat wins deviate from selfplay baseline"
                    ],
                    "details": {
                        "seat_fairness": {
                            "seat_wr": {"1": 0.58, "2": 0.22, "3": 0.20},
                            "selfplay_baseline_seat_wr": {"1": 0.33, "2": 0.33, "3": 0.34},
                            "expected_seat_wins": {"1": 6.0, "2": 6.0, "3": 6.2},
                            "chi_square_stat": 12.6,
                            "chi_square_p_value": 0.0018,
                        }
                    },
                },
            },
        ],
    )

    report = module.build_audit_report(metrics_path=metrics_path, config="square8_3p")

    assert report["config"] == "square8_3p"
    assert report["verdict_counts"] == {
        "clean_rejection": 1,
        "threshold_audit_candidate": 1,
        "chi_square_fired": 1,
    }
    assert report["latest_recomputed_estimated_elo"] == pytest.approx(1516.7, abs=0.1)
    assert report["chi_square_p_value_distribution"]["count"] == 3
    assert report["rows"][0]["verdict"] == "clean_rejection"
    assert report["rows"][1]["verdict"] == "threshold_audit_candidate"
    assert report["rows"][1]["corrected_elo_delta"] == pytest.approx(16.7, abs=0.1)
    assert report["rows"][2]["verdict"] == "chi_square_fired"
    assert report["rows"][2]["chi_square_fired"] is True


def test_cli_writes_json_report(tmp_path: Path) -> None:
    metrics_path = tmp_path / "minimal_loop_hex8_4p" / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True)
    _write_metrics(
        metrics_path,
        [
            {
                "iteration": 6,
                "promoted": False,
                "estimated_elo": 1500.0,
                "evaluation": {"decision": "reject", "win_rate": 0.22, "games_played": 100},
                "quality_gate": {
                    "warnings": [],
                    "details": {
                        "seat_fairness": {
                            "seat_wr": {"1": 0.20, "2": 0.23, "3": 0.29, "4": 0.28},
                            "selfplay_baseline_seat_wr": {
                                "1": 0.17,
                                "2": 0.25,
                                "3": 0.28,
                                "4": 0.30,
                            },
                            "expected_seat_wins": {"1": 3.4, "2": 5.0, "3": 5.6, "4": 6.0},
                            "chi_square_stat": 5.1,
                            "chi_square_p_value": 0.165,
                        }
                    },
                },
            }
        ],
    )
    output = tmp_path / "audit.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(metrics_path), "--output", str(output)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert len(artifact["reports"]) == 1
    report = artifact["reports"][0]
    assert report["config"] == "hex8_4p"
    assert report["rows"][0]["chi_square_p_value"] == 0.165
    assert report["rows"][0]["verdict"] == "clean_rejection"
