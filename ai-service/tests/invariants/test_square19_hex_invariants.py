from __future__ import annotations

"""Minimal invariant smoke-tests for square19 and hex boards.

These tests exercise the Python TrainingEnv + GameEngine on non-square8
boards for a handful of moves to flush out any gross rules issues (such
as ACTIVE states with no legal moves) without requiring long self-play
soaks or TS parity harnesses.
"""

import os
import sys
from typing import Optional

import pytest

# Ensure app package is importable when running tests directly under ai-service/.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import BoardType, GameStatus  # type: ignore  # noqa: E402
from app.training.env import RingRiftEnv  # type: ignore  # noqa: E402


def _play_short_game(
    board_type: BoardType,
    num_players: int,
    max_moves: int,
    seed: int,
    max_steps: int = 20,
) -> None:
    """Play a short self-play sequence and assert basic invariants.

    Invariants checked:
      - On every step where state.game_status == ACTIVE, env.legal_moves()
        is non-empty (no immediate ACTIVE-no-moves pathologies).
      - The loop terminates within max_steps or when the game finishes.
    """
    env = RingRiftEnv(board_type=board_type, max_moves=max_moves, num_players=num_players)
    state = env.reset(seed=seed)

    assert state.board_type == board_type
    assert state.game_status in {GameStatus.WAITING, GameStatus.ACTIVE}

    steps = 0
    while state.game_status == GameStatus.ACTIVE and steps < max_steps:
        legal = env.legal_moves()
        # For ACTIVE states, there should always be at least one legal move.
        assert len(legal) > 0, (
            f"Encountered ACTIVE state with no legal moves on {board_type.value} "
            f"at step={steps}, current_player={state.current_player}, "
            f"phase={state.current_phase}"
        )

        move = legal[0]
        state, _reward, done, _info = env.step(move)
        steps += 1
        if done:
            break


@pytest.mark.slow
def test_square19_short_self_play_has_no_active_no_moves() -> None:
    """Short 2p square19 self-play should not hit ACTIVE-no-moves early."""
    _play_short_game(
        board_type=BoardType.SQUARE19,
        num_players=2,
        max_moves=100,
        seed=123,
        max_steps=20,
    )


@pytest.mark.slow
def test_hexagonal_short_self_play_has_no_active_no_moves() -> None:
    """Short 2p hex self-play should not hit ACTIVE-no-moves early."""
    _play_short_game(
        board_type=BoardType.HEXAGONAL,
        num_players=2,
        max_moves=150,
        seed=456,
        max_steps=20,
    )

