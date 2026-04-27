"""Focused tests for minimal_alphazero_loop experiment harness flags."""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import minimal_alphazero_loop as loop


REPO_ROOT = Path(__file__).resolve().parents[4]
CRITICAL_LOOP_FILES = (
    "ai-service/scripts/minimal_alphazero_loop.py",
    "ai-service/scripts/jsonl_to_npz.py",
    "ai-service/scripts/lib/loop_self_healing.py",
    "ai-service/scripts/lib/minimal_loop_strategy.py",
    "ai-service/scripts/lib/model_quality_gate.py",
    "ai-service/scripts/lib/training_probes.py",
    "ai-service/app/training/train.py",
    "ai-service/app/training/env.py",
    "ai-service/app/ai/gumbel_mcts_ai.py",
)


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
    iterations: int = 1,
    num_players: int = 2,
    skip_quality_gate: bool = True,
    existing_npz_markers: list[float] | None = None,
    existing_metrics: list[dict] | None = None,
    supplemental_markers: list[float] | None = None,
    staged_eval_result: dict | None = None,
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    start_model = tmp_path / "start.pth"
    start_model.write_bytes(b"seed-model")

    if existing_npz_markers:
        for index, marker in enumerate(existing_npz_markers, start=1):
            _write_iteration_npz(work_dir / f"iter_{index:03d}.npz", marker)

    if existing_metrics:
        metrics_path = work_dir / "metrics.jsonl"
        metrics_path.write_text(
            "".join(json.dumps(row) + "\n" for row in existing_metrics),
            encoding="utf-8",
        )

    supplemental_dir = tmp_path / "supplemental"
    if supplemental_markers:
        supplemental_dir.mkdir()
        for index, marker in enumerate(supplemental_markers, start=1):
            _write_iteration_npz(supplemental_dir / f"supplemental_{index:03d}.npz", marker)

    heartbeats: list[dict] = []
    train_calls: list[dict] = []
    eval_calls: list[dict] = []
    export_markers: list[float] = []

    def fake_run_selfplay(model_path, n_games, out_path, budget, randomness=0.25):
        result = {
            "completed": n_games,
            "elapsed_s": 0.1,
            "budget_used": budget,
            "randomness": randomness,
        }
        for player in range(1, num_players + 1):
            result[f"p{player}_wins"] = player
        return result

    def fake_export_npz(jsonl_path, npz_path):
        marker = float(npz_path.stem.split("_")[-1])
        export_markers.append(marker)
        _write_iteration_npz(npz_path, marker)
        return True

    def fake_train_model(
        npz_path,
        out_path,
        init_path,
        epochs,
        batch_size,
        lr,
        train_lr_scheduler,
        **train_kwargs,
    ):
        with np.load(npz_path, allow_pickle=True) as data:
            markers = data["features"][:, 0, 0, 0].tolist()
        train_calls.append(
            {
                "npz_name": Path(npz_path).name,
                "markers": markers,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "train_lr_scheduler": train_lr_scheduler,
                **train_kwargs,
            }
        )
        Path(out_path).write_bytes(b"candidate-model")
        return {"last_epoch_line": "Epoch [1/1], Train Loss: 0.1, Val Loss: 0.2"}

    def fake_evaluate(candidate_path, best_path, eval_games, budget, tracker=None):
        return {
            "win_rate": 0.50,
            "eval_games": eval_games,
            "budget": budget,
            "candidate_wins": 1,
            "best_wins": 1,
            "draws": 0,
        }

    def fake_staged_evaluate(
        candidate_path,
        best_path,
        budget,
        tracker=None,
        promote_threshold_cap=None,
        checkpoint_path=None,
        **extra_kwargs,
    ):
        # checkpoint_path was added by commit 69fc25aa0 (game-granular resume
        # for selfplay + eval). Accept it here so these tests keep working
        # after production adds new keyword-only params to staged_evaluate.
        # Any additional future kwargs fall into **extra_kwargs — we record
        # them so a surprise param is visible in the test output rather than
        # silently crashing the whole loop.
        eval_calls.append(
            {
                "candidate_path": candidate_path,
                "best_path": best_path,
                "budget": budget,
                "promote_threshold_cap": promote_threshold_cap,
                "checkpoint_path": checkpoint_path,
                "extra_kwargs": dict(extra_kwargs) if extra_kwargs else {},
            }
        )
        result = {
            "win_rate": 0.50,
            "budget": budget,
            "candidate_wins": 1,
            "best_wins": 1,
            "draws": 0,
            "games_played": 50,
            "decision": "reject",
            "decision_stage": 1,
        }
        if promote_threshold_cap is not None:
            result["promote_threshold_cap"] = promote_threshold_cap
        if staged_eval_result:
            result.update(staged_eval_result)
        return {
            **result,
        }

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
    monkeypatch.setattr(loop, "staged_evaluate", fake_staged_evaluate)
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
            str(num_players),
            "--iterations",
            str(iterations),
            "--games-per-iter",
            "1",
            "--eval-games",
            "1",
            "--work-dir",
            str(work_dir),
            "--skip-quality-check",
            "--skip-probes",
            *(["--skip-quality-gate"] if skip_quality_gate else []),
            *(
                ["--supplemental-data-dir", str(supplemental_dir)]
                if supplemental_markers
                else []
            ),
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
        "eval_calls": eval_calls,
        "metrics": metrics_rows,
        "export_markers": export_markers,
    }


@pytest.mark.parametrize("win_rate", [0.2, 0.37, 0.5, 0.62, 0.8])
def test_promotion_elo_delta_matches_legacy_formula_exactly_for_2p(win_rate: float) -> None:
    legacy = 400.0 * math.log10(win_rate / (1.0 - win_rate))

    assert loop._promotion_elo_delta(win_rate, 2) == legacy


@pytest.mark.parametrize(
    ("num_players", "fair_win_rate", "above_fair", "below_fair"),
    [
        (2, 0.5, 0.6, 0.4),
        (3, 1.0 / 3.0, 0.5, 0.25),
        (4, 0.25, 0.4, 0.2),
    ],
)
def test_promotion_elo_delta_uses_fair_multiplayer_baseline(
    num_players: int,
    fair_win_rate: float,
    above_fair: float,
    below_fair: float,
) -> None:
    assert loop._promotion_elo_delta(fair_win_rate, num_players) == 0.0
    assert loop._promotion_elo_delta(above_fair, num_players) > 0.0
    assert loop._promotion_elo_delta(below_fair, num_players) < 0.0


def test_export_npz_requests_heuristics_for_v5_heavy(monkeypatch, tmp_path):
    """V5-heavy minimal-loop exports must include heuristic features."""
    jsonl_path = tmp_path / "games.jsonl"
    jsonl_path.write_text("{}\n", encoding="utf-8")
    npz_path = tmp_path / "games.npz"
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        np.savez_compressed(
            npz_path,
            features=np.zeros((1, 64, 9, 9), dtype=np.float32),
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    monkeypatch.setattr(loop, "MODEL_VERSION", "v5-heavy")
    monkeypatch.setattr(loop, "BOARD_TYPE", "hex8")
    monkeypatch.setattr(loop, "NUM_PLAYERS", 2)

    assert loop.export_npz(jsonl_path, npz_path) is True
    assert "--encoder-version" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--encoder-version") + 1] == "v3"
    assert "--include-heuristics" in captured["cmd"]


def test_export_npz_threads_feature_version_flag(monkeypatch, tmp_path):
    """FEATURE_VERSION global must reach the jsonl_to_npz subprocess call."""
    jsonl_path = tmp_path / "games.jsonl"
    jsonl_path.write_text("{}\n", encoding="utf-8")
    npz_path = tmp_path / "games.npz"
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        np.savez_compressed(
            npz_path,
            features=np.zeros((1, 64, 9, 9), dtype=np.float32),
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    monkeypatch.setattr(loop, "MODEL_VERSION", "v5-heavy")
    monkeypatch.setattr(loop, "BOARD_TYPE", "hex8")
    monkeypatch.setattr(loop, "NUM_PLAYERS", 2)
    monkeypatch.setattr(loop, "FEATURE_VERSION", 3)

    assert loop.export_npz(jsonl_path, npz_path) is True
    assert "--feature-version" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--feature-version") + 1] == "3"


def test_train_model_threads_feature_version_flag(monkeypatch, tmp_path):
    """FEATURE_VERSION global must reach the app.training.train subprocess call."""
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="val_loss=0.1\n", stderr="")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    monkeypatch.setattr(loop, "BOARD_TYPE", "hex8")
    monkeypatch.setattr(loop, "NUM_PLAYERS", 2)
    monkeypatch.setattr(loop, "MODEL_VERSION", "v5-heavy")
    monkeypatch.setattr(loop, "FEATURE_VERSION", 3)

    result = loop.train_model(
        tmp_path / "train.npz",
        tmp_path / "candidate.pth",
        tmp_path / "init.pth",
        epochs=2,
        bs=64,
        lr=5e-5,
        train_lr_scheduler="none",
    )

    assert "error" not in result
    assert "--feature-version" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--feature-version") + 1] == "3"


def test_make_ai_threads_feature_version_to_runtime_config(monkeypatch):
    """FEATURE_VERSION must also reach live MCTS inference, not just export/train."""
    captured: dict = {}

    class FakeGumbelMCTSAI:
        def __init__(self, player, cfg, board_type):
            captured["player"] = player
            captured["cfg"] = cfg
            captured["board_type"] = board_type

    monkeypatch.setattr(loop, "GumbelMCTSAI", FakeGumbelMCTSAI)
    monkeypatch.setattr(loop, "MODEL_VERSION", "v5-heavy")
    monkeypatch.setattr(loop, "FEATURE_VERSION", 3)

    loop._make_ai(1, "models/candidate.pth", 32)

    assert captured["player"] == 1
    assert captured["cfg"].nn_model_version == "v5-heavy"
    assert captured["cfg"].feature_version == 3


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
    assert metrics["train_lr_scheduler"] == "none"
    assert metrics["train_window"] == 3
    assert isinstance(metrics["git_sha"], str) and metrics["git_sha"]

    assert len(result["train_calls"]) == 1
    assert result["train_calls"][0]["lr"] == 5e-5
    assert result["train_calls"][0]["train_lr_scheduler"] == "none"

    stages = {heartbeat["stage"]: heartbeat for heartbeat in result["heartbeats"]}
    assert set(stages) == {"selfplay_done", "training_done", "iteration_done"}

    selfplay_params = stages["selfplay_done"]["experiment_params"]
    assert selfplay_params["selfplay_budget"] == 200
    assert selfplay_params["eval_budget"] == 128
    assert selfplay_params["base_lr"] == 5e-5
    assert selfplay_params["train_lr_scheduler"] == "none"
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
    assert train_call["train_lr_scheduler"] == "cosine"

    metrics = result["metrics"][0]
    assert metrics["train_window"] == 2
    assert metrics["effective_lr"] == 5e-5
    assert metrics["base_lr"] == 1e-4
    assert metrics["train_lr_scheduler"] == "cosine"

    training_params = next(
        heartbeat["experiment_params"]
        for heartbeat in result["heartbeats"]
        if heartbeat["stage"] == "training_done"
    )
    assert training_params["effective_lr"] == 5e-5
    assert training_params["train_lr_scheduler"] == "cosine"


def test_loop_allows_explicit_training_scheduler_override(monkeypatch, tmp_path):
    result = _run_loop_once(
        monkeypatch,
        tmp_path,
        extra_args=[
            "--lr",
            "5e-5",
            "--lr-schedule",
            "fixed",
            "--train-lr-scheduler",
            "step",
        ],
    )

    assert result["train_calls"][0]["train_lr_scheduler"] == "step"
    assert result["metrics"][0]["train_lr_scheduler"] == "step"


def test_loop_threads_training_loss_knobs_to_train_model(monkeypatch, tmp_path):
    result = _run_loop_once(
        monkeypatch,
        tmp_path,
        extra_args=[
            "--policy-weight",
            "0.8",
            "--value-weight",
            "1.8",
            "--rank-dist-weight",
            "0.05",
            "--gradient-clip-max-norm",
            "0.5",
        ],
    )

    train_call = result["train_calls"][0]
    assert train_call["policy_weight"] == 0.8
    assert train_call["value_weight"] == 1.8
    assert train_call["rank_dist_weight"] == 0.05
    assert train_call["gradient_clip_max_norm"] == 0.5

    metrics = result["metrics"][0]
    assert metrics["policy_weight"] == 0.8
    assert metrics["value_weight"] == 1.8
    assert metrics["rank_dist_weight"] == 0.05
    assert metrics["gradient_clip_max_norm"] == 0.5


def test_loop_merges_supplemental_npz_without_touching_iteration_namespace(monkeypatch, tmp_path):
    result = _run_loop_once(
        monkeypatch,
        tmp_path,
        extra_args=[
            "--train-window",
            "2",
        ],
        existing_npz_markers=[1.0, 2.0],
        supplemental_markers=[9.0],
    )

    assert len(result["train_calls"]) == 1
    train_call = result["train_calls"][0]
    assert train_call["npz_name"] == "combined_003.npz"
    assert train_call["markers"] == [2.0, 3.0, 9.0]


def test_cap_promote_thresholds_only_lowers_stricter_stages() -> None:
    stages = [(50, 0.60, 0.42), (100, 0.56, 0.46), (400, 0.501, 0.0)]

    capped = loop._cap_promote_thresholds(stages, 0.52)

    assert capped == [(50, 0.52, 0.42), (100, 0.52, 0.46), (400, 0.501, 0.0)]
    assert loop._cap_promote_thresholds(stages, None) is stages


def test_auto_plateau_relax_arms_next_three_iterations(monkeypatch, tmp_path):
    existing_metrics = [
        {"iteration": i, "promoted": False}
        for i in range(1, 20)
    ]

    result = _run_loop_once(
        monkeypatch,
        tmp_path,
        iterations=2,
        existing_npz_markers=[float(i) for i in range(1, 20)],
        existing_metrics=existing_metrics,
        extra_args=["--auto-plateau-relax"],
    )

    assert result["eval_calls"][0]["promote_threshold_cap"] is None
    assert result["eval_calls"][1]["promote_threshold_cap"] == 0.52

    new_metrics = result["metrics"][-2:]
    assert new_metrics[0]["iteration"] == 20
    assert new_metrics[0]["plateau"]["detected"] is True
    assert new_metrics[0]["plateau"]["relax_until_iteration"] == 23
    assert new_metrics[1]["iteration"] == 21
    assert new_metrics[1]["evaluation"]["promote_threshold_cap"] == 0.52


def test_loop_recomputes_resumed_multiplayer_elo_from_promotion_history(monkeypatch, tmp_path):
    existing_metrics = [
        {
            "iteration": 1,
            "promoted": True,
            "estimated_elo": 1500.0,
            "total_promotions": 1,
            "evaluation": {"win_rate": 0.5},
        }
    ]

    result = _run_loop_once(
        monkeypatch,
        tmp_path,
        iterations=1,
        num_players=3,
        existing_npz_markers=[1.0],
        existing_metrics=existing_metrics,
        staged_eval_result={"decision": "reject", "win_rate": 1.0 / 3.0},
        extra_args=[],
    )

    resumed_metrics = result["metrics"][-1]
    expected_elo = round(1500.0 + loop._promotion_elo_delta(0.5, 3), 1)

    assert resumed_metrics["iteration"] == 2
    assert resumed_metrics["promoted"] is False
    assert resumed_metrics["total_promotions"] == 1
    assert resumed_metrics["estimated_elo"] == expected_elo


def test_loop_passes_selfplay_seat_baseline_into_quality_gate(monkeypatch, tmp_path):
    captured: dict[str, dict[int, int]] = {}

    def fake_check_model_quality(tracker):
        captured["baseline"] = dict(tracker._selfplay_seat_wins)
        return SimpleNamespace(
            passed=True,
            critical=False,
            warnings=[],
            summary="quality gate passed",
            details={},
        )

    monkeypatch.setattr(loop, "check_model_quality", fake_check_model_quality)

    _run_loop_once(
        monkeypatch,
        tmp_path,
        extra_args=[],
        num_players=4,
        skip_quality_gate=False,
    )

    assert captured["baseline"] == {1: 1, 2: 2, 3: 3, 4: 4}


def test_train_model_passes_requested_lr_scheduler(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="val_loss=0.1\n", stderr="")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    monkeypatch.setattr(loop, "BOARD_TYPE", "square8")
    monkeypatch.setattr(loop, "NUM_PLAYERS", 2)

    result = loop.train_model(
        tmp_path / "train.npz",
        tmp_path / "candidate.pth",
        tmp_path / "init.pth",
        epochs=2,
        bs=64,
        lr=5e-5,
        train_lr_scheduler="none",
    )

    assert "error" not in result
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[1:3] == ["-m", "app.training.train"]
    assert cmd[cmd.index("--lr-scheduler") + 1] == "none"


def test_train_model_passes_loss_weight_knobs(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="val_loss=0.1\n", stderr="")

    monkeypatch.setattr(loop.subprocess, "run", fake_run)
    monkeypatch.setattr(loop, "BOARD_TYPE", "hex8")
    monkeypatch.setattr(loop, "NUM_PLAYERS", 3)

    result = loop.train_model(
        tmp_path / "train.npz",
        tmp_path / "candidate.pth",
        tmp_path / "init.pth",
        epochs=2,
        bs=64,
        lr=3e-5,
        train_lr_scheduler="none",
        policy_weight=0.8,
        value_weight=1.8,
        rank_dist_weight=0.05,
        gradient_clip_max_norm=0.5,
    )

    assert "error" not in result
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--policy-weight") + 1] == "0.8"
    assert cmd[cmd.index("--value-weight") + 1] == "1.8"
    assert cmd[cmd.index("--rank-dist-weight") + 1] == "0.05"
    assert cmd[cmd.index("--gradient-clip-max-norm") + 1] == "0.5"


def test_minimal_loop_help_renders() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/minimal_alphazero_loop.py", "--help"],
        cwd=REPO_ROOT / "ai-service",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "--value-weight" in result.stdout
    assert "52%" in result.stdout


def test_critical_minimal_loop_modules_avoid_top_level_facade_imports() -> None:
    for relative_path in CRITICAL_LOOP_FILES:
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {
                "app.training",
                "app.coordination",
                "app.distributed",
            }:
                raise AssertionError(relative_path)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {
                        "app.training",
                        "app.coordination",
                        "app.distributed",
                    }:
                        raise AssertionError(relative_path)


def test_staged_evaluate_rotates_candidate_seat_evenly(monkeypatch):
    candidate_by_game: dict[int, int] = {}
    assignment_log: list[tuple[int, int]] = []
    current_game = {"index": -1}
    fake_move = object()

    class FakeAI:
        def __init__(self, player, model_path):
            self.player_number = player
            self.model_path = model_path

        def reset_for_new_game(self, rng_seed=None):
            return None

        def select_move(self, state):
            return fake_move

    class FakeEnv:
        num_players = 4

        def reset(self, seed=None):
            current_game["index"] += 1
            return SimpleNamespace(
                game_status=loop.GameStatus.ACTIVE,
                current_player=1,
                winner=None,
            )

        def legal_moves(self):
            return [fake_move]

        def step(self, move):
            winner = candidate_by_game[current_game["index"]]
            return (
                SimpleNamespace(
                    game_status=loop.GameStatus.COMPLETED,
                    current_player=1,
                    winner=winner,
                ),
                0.0,
                True,
                {},
            )

    def fake_make_env():
        return FakeEnv()

    def fake_make_ai(player, model_path, budget, randomness=0.0):
        game_idx = len(assignment_log) // 4
        assignment_log.append((game_idx, player))
        if model_path == "cand-model":
            candidate_by_game[game_idx] = player
        return FakeAI(player, model_path)

    monkeypatch.setattr(loop, "_make_env", fake_make_env)
    monkeypatch.setattr(loop, "_make_ai", fake_make_ai)
    monkeypatch.setattr(loop, "_get_eval_stages", lambda: [(8, 0.99, 0.0)])

    result = loop.staged_evaluate("cand-model", "best-model", budget=64)

    assert result["candidate_wins"] == 8
    assert result["best_wins"] == 0
    assert result["win_rate"] == 1.0
    assert [candidate_by_game[i] for i in range(8)] == [1, 2, 3, 4, 1, 2, 3, 4]
