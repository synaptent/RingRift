"""Tests for the Square-8 2-player NN baseline training script."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure `app.*` / `scripts.*` imports resolve when running pytest from
# ai-service/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.run_nn_training_baseline as nn_baseline  # noqa: E402


TEST_TIMEOUT_SECONDS = 60


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_run_nn_training_baseline_demo_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demo run should write a small nn_training_report.json for Square-8 2p.

    The test patches train_model to avoid heavy torch work so that CI runs
    remain cheap while still exercising the CLI wiring and JSON schema.
    """

    run_dir = tmp_path / "nn_demo"

    # Patch train_model in the script module to a lightweight stub.
    def _fake_train_model(
        config,
        data_path,
        save_path,
        early_stopping_patience=0,
        checkpoint_dir="checkpoints",
        checkpoint_interval=1,
        warmup_epochs=0,
        lr_scheduler="none",
        **_kwargs,
    ):
        # Ensure the save path directory exists and create an empty file to
        # mirror the presence of a checkpoint without doing any real work.
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(b"")

    monkeypatch.setattr(
        nn_baseline,
        "train_model",
        _fake_train_model,
        raising=True,
    )

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
    assert report["model_id"]

    params = report["training_params"]
    assert params["board_type"] == "SQUARE8"
    assert params["epochs_per_iter"] == 1
    assert params["seed"] == 123

    metrics = report["metrics"]
    assert "final_loss" in metrics
    assert metrics["train_epochs_run"] == 1