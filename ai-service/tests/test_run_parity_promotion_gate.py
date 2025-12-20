"""Tests for the cross-board parity promotion gate helper."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import pytest

# Ensure app/ is importable when running tests directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.run_parity_promotion_gate as parity_gate  # type: ignore[import]  # noqa: E402


def _make_formatted(
    *,
    board: str,
    games: int,
    win_rate: float,
    ci: tuple[float, float],
    piece_advantage: float,
) -> Dict[str, Any]:
    """Build a minimal ``format_results_json``-like payload."""
    return {
        "config": {
            "player1": "candidate",
            "player2": "baseline",
            "games": games,
            "board": board,
        },
        "results": {
            "player1_wins": int(round(win_rate * games)),
            "player2_wins": games - int(round(win_rate * games)),
            "draws": 0,
            "player1_win_rate": win_rate,
            "player1_win_rate_ci95": [ci[0], ci[1]],
            "avg_game_length": 40.0,
            "avg_game_length_std": 5.0,
            "avg_decision_time_p1": 0.01,
            "avg_decision_time_p2": 0.01,
            "total_runtime_seconds": 1.0,
            "victory_types": {},
            "avg_p1_final_pieces": 10.0,
            "avg_p2_final_pieces": 9.0,
            "piece_advantage_p1": piece_advantage,
        },
        "games": [],
    }


def test_evaluate_promotion_all_matrices_pass() -> None:
    """Gate should pass when every matrix has CI lower bound above threshold."""
    matrices = {
        "square8_2p": _make_formatted(
            board="square8",
            games=200,
            win_rate=0.62,
            ci=(0.55, 0.68),
            piece_advantage=1.5,
        ),
        "square19_2p": _make_formatted(
            board="square19",
            games=200,
            win_rate=0.58,
            ci=(0.51, 0.65),
            piece_advantage=0.7,
        ),
    }

    summary = parity_gate._evaluate_promotion(matrices, min_ci_lower_bound=0.5)

    assert summary["overall_pass"] is True
    assert summary["thresholds"]["min_ci_lower_bound"] == 0.5
    assert summary["worst_case_ci_lower_bound"] == pytest.approx(0.51)

    sq8 = summary["matrices"]["square8_2p"]
    assert sq8["board"] == "square8"
    assert sq8["games"] == 200
    assert sq8["player1_win_rate"] == pytest.approx(0.62)
    assert sq8["player1_win_rate_ci95"] == [pytest.approx(0.55), pytest.approx(0.68)]
    assert sq8["piece_advantage_p1"] == pytest.approx(1.5)
    assert sq8["passes"] is True


def test_evaluate_promotion_any_failure_fails_gate() -> None:
    """Gate should fail when any matrix has CI lower bound below threshold."""
    matrices = {
        "square8_2p": _make_formatted(
            board="square8",
            games=200,
            win_rate=0.55,
            ci=(0.49, 0.61),
            piece_advantage=0.2,
        ),
        "square19_2p": _make_formatted(
            board="square19",
            games=200,
            win_rate=0.6,
            ci=(0.53, 0.67),
            piece_advantage=0.9,
        ),
    }

    summary = parity_gate._evaluate_promotion(matrices, min_ci_lower_bound=0.5)

    assert summary["overall_pass"] is False
    assert summary["matrices"]["square8_2p"]["passes"] is False
    assert summary["matrices"]["square19_2p"]["passes"] is True
    assert summary["worst_case_ci_lower_bound"] == pytest.approx(0.49)


def test_evaluate_promotion_empty_matrix_is_conservative_failure() -> None:
    """With no matrices provided, gate should fail rather than pass vacuously."""
    summary = parity_gate._evaluate_promotion({}, min_ci_lower_bound=0.5)

    assert summary["overall_pass"] is False
    assert summary["worst_case_ci_lower_bound"] == pytest.approx(0.0)
    assert summary["matrices"] == {}


def test_main_uses_matrix_stub_and_writes_report(tmp_path, monkeypatch) -> None:
    """Smoke test the CLI path with a stubbed matrix runner."""
    output_path = tmp_path / "parity_gate.json"

    # Provide a fake _run_matrix to avoid heavy evaluations.
    def _fake_run_matrix(
        spec: Any,
        *,
        player1_type: str,
        player2_type: str,
        seed: int | None,
        checkpoint: str | None,
        checkpoint2: str | None,
        cmaes_weights: str | None,
        minimax_depth: int,
    ) -> Dict[str, Any]:
        # Minimal format_results_json-like structure.
        return _make_formatted(
            board=spec.board_key,
            games=spec.games,
            win_rate=0.6,
            ci=(0.55, 0.65),
            piece_advantage=1.0,
        )

    monkeypatch.setattr(
        parity_gate,
        "_run_matrix",
        _fake_run_matrix,
    )

    argv = [
        "--player1",
        "random",
        "--player2",
        "random",
        "--boards",
        "square8",
        "--games-per-matrix",
        "10",
        "--max-moves",
        "50",
        "--seed",
        "123",
        "--output-json",
        str(output_path),
    ]

    rc = parity_gate.main(argv)
    assert rc == 0
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    cfg = payload["config"]
    gate = payload["gate"]

    assert cfg["player1"] == "random"
    assert cfg["player2"] == "random"
    assert cfg["boards"] == ["square8"]
    assert cfg["games_per_matrix"] == 10
    assert cfg["max_moves"] == 50
    assert cfg["seed"] == 123

    assert gate["overall_pass"] is True
    assert gate["thresholds"]["min_ci_lower_bound"] == 0.5

    matrices = gate["matrices"]
    assert "square8_2p" in matrices
    m_sq8 = matrices["square8_2p"]
    assert m_sq8["board"] == "square8"
    assert m_sq8["games"] == 10
    assert m_sq8["passes"] is True
