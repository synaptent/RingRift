"""Focused tests for minimal_alphazero_loop experiment harness flags."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts import minimal_alphazero_loop as loop


def _write_iteration_npz(path: Path, marker: float) -> None:
    features = np.array([[[[marker]]]], dtype=np.float32)
    policy_targets = np.array([[1.0]], dtype=np.float32)
    value_targets = np.array([0.0], dtype=np.float32)
    np.savez_compressed(
        path,
        features=features,
        policy_targets=policy_targets,
        value_targets=value_targets,
    )


def _run_loop_once(
    monkeypatch,
    tmp_path: Path,
    *,
    extra_args: list[str],
    existing_npz_markers: list[float] | None = None,
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    start_model = tmp_path / "start.pth"
    start_model.write_bytes(b"seed-model")

    if existing_npz_markers:
        for index, marker in enumerate(existing_npz_markers, start=1):
            _write_iteration_npz(work_dir / f"iter_{index:03d}.npz", marker)

    heartbeats: list[dict] = []
    train_calls: list[dict] = []
    export_markers: list[float] = []

    def fake_run_selfplay(model_path, n_games, out_path, budget, randomness=0.25):
        return {
            "completed": n_games,
            "elapsed_s": 0.1,
            "budget_used": budget,
            "randomness": randomness,
        }

    def fake_export_npz(jsonl_path, npz_path):
        marker = float(npz_path.stem.split("_")[-1])
        export_markers.append(marker)
        _write_iteration_npz(npz_path, marker)
        return True

    def fake_train_model(npz_path, out_path, init_path, epochs, batch_size, lr):
        with np.load(npz_path, allow_pickle=True) as data:
            markers = data["features"][:, 0, 0, 0].tolist()
        train_calls.append(
            {
                "npz_name": Path(npz_path).name,
                "markers": markers,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
            }
        )
        Path(out_path).write_bytes(b"candidate-model")
        return {"last_epoch_line": "Epoch [1/1], Train Loss: 0.1, Val Loss: 0.2"}

    def fake_evaluate(candidate_path, best_path, eval_games, budget, tracker=None):
        return {"win_rate": 0.50, "eval_games": eval_games, "budget": budget}

    def fake_push_heartbeat(node_id, config_key, iteration, elo, promos, data_quality_score=None, *, stage="iteration_done", experiment_params=None):
        heartbeats.append(
            {
                "node_id": node_id,
                "config_key": config_key,
                "iteration": iteration,
                "elo": elo,
                "promos": promos,
                "data_quality_score": data_quality_score,
                "stage": stage,
                "experiment_params": experiment_params,
            }
        )

    monkeypatch.setattr(loop, "run_selfplay", fake_run_selfplay)
    monkeypatch.setattr(loop, "export_npz", fake_export_npz)
    monkeypatch.setattr(loop, "train_model", fake_train_model)
    monkeypatch.setattr(loop, "evaluate", fake_evaluate)
    monkeypatch.setattr(loop, "_push_heartbeat_s3", fake_push_heartbeat)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "minimal_alphazero_loop.py",
            "--model",
            str(start_model),
            "--board-type",
            "hex8",
            "--num-players",
            "2",
            "--iterations",
            "1",
            "--games-per-iter",
            "1",
            "--eval-games",
            "1",
            "--work-dir",
            str(work_dir),
            "--skip-quality-check",
            "--skip-probes",
            "--skip-quality-gate",
            *extra_args,
        ],
    )

    loop.main()

    metrics_path = work_dir / "metrics.jsonl"
    metrics_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]

    return {
        "work_dir": work_dir,
        "heartbeats": heartbeats,
        "train_calls": train_calls,
        "metrics": metrics_rows,
        "export_markers": export_markers,
    }


def test_loop_records_split_budgets_fixed_lr_and_heartbeat_context(monkeypatch, tmp_path):
    result = _run_loop_once(
        monkeypatch,
        tmp_path,
        extra_args=[
            "--selfplay-budget",
            "200",
            "--eval-budget",
            "128",
            "--lr",
            "5e-5",
            "--lr-schedule",
            "fixed",
            "--train-window",
            "3",
        ],
    )

    assert len(result["metrics"]) == 1
    metrics = result["metrics"][0]
    assert metrics["selfplay_budget"] == 200
    assert metrics["eval_budget"] == 128
    assert metrics["base_lr"] == 5e-5
    assert metrics["effective_lr"] == 5e-5
    assert metrics["lr_schedule"] == "fixed"
    assert metrics["train_window"] == 3
    assert isinstance(metrics["git_sha"], str) and metrics["git_sha"]

    assert len(result["train_calls"]) == 1
    assert result["train_calls"][0]["lr"] == 5e-5

    stages = {heartbeat["stage"]: heartbeat for heartbeat in result["heartbeats"]}
    assert set(stages) == {"selfplay_done", "training_done", "iteration_done"}

    selfplay_params = stages["selfplay_done"]["experiment_params"]
    assert selfplay_params["selfplay_budget"] == 200
    assert selfplay_params["eval_budget"] == 128
    assert selfplay_params["base_lr"] == 5e-5
    assert "effective_lr" not in selfplay_params

    training_params = stages["training_done"]["experiment_params"]
    assert training_params["effective_lr"] == 5e-5
    assert training_params["git_sha"] == metrics["git_sha"]

    final_params = stages["iteration_done"]["experiment_params"]
    assert final_params["effective_lr"] == 5e-5


def test_loop_respects_train_window_and_sqrt_decay(monkeypatch, tmp_path):
    result = _run_loop_once(
        monkeypatch,
        tmp_path,
        extra_args=[
            "--lr",
            "1e-4",
            "--lr-schedule",
            "sqrt_decay",
            "--lr-floor",
            "1e-6",
            "--train-window",
            "2",
        ],
        existing_npz_markers=[1.0, 2.0, 3.0],
    )

    assert len(result["train_calls"]) == 1
    train_call = result["train_calls"][0]
    assert train_call["npz_name"] == "combined_004.npz"
    assert train_call["markers"] == [3.0, 4.0]
    assert train_call["lr"] == 5e-5

    metrics = result["metrics"][0]
    assert metrics["train_window"] == 2
    assert metrics["effective_lr"] == 5e-5
    assert metrics["base_lr"] == 1e-4

    training_params = next(
        heartbeat["experiment_params"]
        for heartbeat in result["heartbeats"]
        if heartbeat["stage"] == "training_done"
    )
    assert training_params["effective_lr"] == 5e-5
