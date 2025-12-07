"""Tests for the baseline Square-8 2-player NN training script."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.run_nn_training_baseline as nn_baseline  # noqa: E402

# Keep demo runs fast to avoid heavy work in CI.
TEST_TIMEOUT_SECONDS = 60


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_run_nn_training_baseline_demo_writes_report(tmp_path) -> None:
    """Smoke test that the baseline NN script writes a demo report and
    artefacts."""
    run_dir = tmp_path / "nn_demo"
    argv = [
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--demo",
        "--seed",
        "123",
    ]

    rc = nn_baseline.main(argv)
    assert rc == 0

    report_path = run_dir / "nn_training_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["board"] == "square8"
    assert report["num_players"] == 2
    assert report["mode"] == "demo"
    assert isinstance(report.get("model_id"), str)

    env_cfg = report["env"]
    assert env_cfg["board_type"] == "SQUARE8"
    assert env_cfg["num_players"] == 2
    assert env_cfg["reward_mode"] == "terminal"

    training_params = report["training_params"]
    assert training_params["mode"] == "demo"
    assert training_params["board_type"] == "SQUARE8"
    assert training_params["model_id"] == report["model_id"]
    assert training_params["epochs"] == 1
    assert training_params["iterations"] == 1

    # Demo run should generate a tiny synthetic dataset in the run directory.
    data_path = Path(training_params["data_path"])
    assert data_path.exists()
    assert data_path.parent == run_dir

    # Metrics should at least record a non-zero training_steps count.
    metrics = report["metrics"]
    assert metrics["training_steps"] == 1


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_run_nn_training_baseline_stub_mode(tmp_path) -> None:
    """Non-demo mode should write a stub config without running heavy
    training."""
    run_dir = tmp_path / "nn_stub"
    argv = [
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--seed",
        "7",
    ]

    rc = nn_baseline.main(argv)
    assert rc == 0

    report_path = run_dir / "nn_training_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["board"] == "square8"
    assert report["num_players"] == 2
    assert report["mode"] == "stub"

    training_params = report["training_params"]
    assert training_params["mode"] == "neural_full_stub"
    assert training_params["board_type"] == "SQUARE8"
    # Stub mode should not attempt to generate a dataset or checkpoint yet.
    assert "data_path" not in training_params
    assert "save_path" not in training_params