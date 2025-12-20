"""Tests for promotion / gating helpers in evaluate_ai_models."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pytest

# Ensure app/ is importable when running tests directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.evaluate_ai_models import (  # type: ignore[import]  # noqa: E402
    EvaluationResults,
    format_results_json,
    wilson_score_interval,
)


def _make_results(player1_wins: int, player2_wins: int, draws: int = 0) -> EvaluationResults:
    total_games = player1_wins + player2_wins + draws
    return EvaluationResults(
        config={"player1": "candidate", "player2": "baseline", "games": total_games, "board": "square8"},
        player1_wins=player1_wins,
        player2_wins=player2_wins,
        draws=draws,
    )


@pytest.mark.parametrize(
    "wins,total,expected_rate",
    [
        (0, 0, 0.0),
        (0, 10, 0.0),
        (5, 10, 0.5),
        (9, 10, 0.9),
    ],
)
def test_wilson_score_interval_basic_monotonicity(wins: int, total: int, expected_rate: float) -> None:
    """Wilson interval should be centred near the empirical win rate."""
    lo, hi = wilson_score_interval(wins, total)
    assert 0.0 <= lo <= hi <= 1.0
    # In normal regimes the true rate should sit inside the CI.
    if total > 0:
        assert lo <= expected_rate <= hi


def test_format_results_json_includes_ci_and_win_rate() -> None:
    """format_results_json exposes win-rate and CI needed for promotion decisions."""
    results = _make_results(player1_wins=7, player2_wins=3)

    payload: dict[str, Any] = format_results_json(results)
    res = payload["results"]

    assert res["player1_wins"] == 7
    assert res["player2_wins"] == 3
    assert res["draws"] == 0

    # Win-rate and CI fields must be present and sane.
    assert 0.0 <= res["player1_win_rate"] <= 1.0
    ci_lo, ci_hi = res["player1_win_rate_ci95"]
    assert 0.0 <= ci_lo <= ci_hi <= 1.0

