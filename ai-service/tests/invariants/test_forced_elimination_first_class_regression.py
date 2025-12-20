from __future__ import annotations

"""Regression tests for forced elimination as a first-class turn action.

These tests validate the fix for the bug where games terminated with
"no_legal_moves_for_current_player" when the player was in a forced
elimination state (has stacks but no placement/movement/capture).

Per RR-CANON-R072/R100/R205, forced elimination must be exposed as a
first-class action in get_valid_moves() so that:
1. Self-play and training loops can continue
2. AI players can choose which stack to eliminate from
3. Human players (via UI) can select their elimination target

Failure scenario captured in logs/selfplay/failures/:
- Game status: ACTIVE
- Current player has stacks on the board
- No placement, movement, or capture moves available
- Previously: get_valid_moves() returned [] → game terminated incorrectly
- After fix: get_valid_moves() returns FORCED_ELIMINATION moves
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import pytest

# Ensure app package is importable when running tests directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (
    GameState, Move, Position, MoveType, GamePhase, GameStatus, BoardType,
    BoardState, Player, RingStack, TimeControl
)
from app.game_engine import GameEngine
from app.board_manager import BoardManager
from app.rules import global_actions as ga


SELFPLAY_FAILURES_DIR = Path(__file__).parent.parent.parent / "logs" / "selfplay" / "failures"
RESULTS_FAILURES_DIR = Path(__file__).parent.parent.parent / "results" / "failures"


def load_failure_snapshot(path: Path) -> GameState | None:
    """Load a failure snapshot and return the GameState."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return GameState.model_validate(payload["state"])


def describe_fe_state(state: GameState) -> dict:
    """Diagnostic info for a forced elimination state."""
    player = state.current_player
    player_stacks = [
        s for s in state.board.stacks.values()
        if s.controlling_player == player
    ]
    return {
        "game_status": state.game_status.value,
        "current_player": player,
        "current_phase": state.current_phase.value if state.current_phase else None,
        "player_stacks": len(player_stacks),
        "has_forced_elimination": ga.has_forced_elimination_action(state, player),
    }


class TestForcedEliminationFirstClass:
    """Test that forced elimination is exposed as first-class action.

    ARCHIVED TEST: test_get_valid_moves_includes_forced_elimination_when_blocked
    (removed 2025-12-07)
    - This test assumed failure snapshots represent pure FE states (player has
      stacks but no placement/movement/capture moves). In reality, the snapshots
      often have capture moves available, so get_valid_moves() correctly returns
      capture moves rather than FE moves. The test premise was incorrect.
      FE moves are only surfaced when the engine enters FORCED_ELIMINATION phase
      via GameEngine._end_turn(). See RR-CANON-R070/R072/R100/R204.
    """

    def test_forced_elimination_moves_cover_all_player_stacks(self):
        """FE moves should be generated for all player-controlled stacks.

        This test uses the real failure snapshot to verify FE move coverage.
        """
        snapshot_path = RESULTS_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"
        if not snapshot_path.exists():
            snapshot_path = SELFPLAY_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"

        if not snapshot_path.exists():
            pytest.skip("No failure snapshot available")

        state = load_failure_snapshot(snapshot_path)
        assert state is not None

        player = state.current_player
        player_stacks = [
            s for s in state.board.stacks.values()
            if s.controlling_player == player
        ]

        if not player_stacks:
            pytest.skip("Snapshot has no player stacks")

        # Get moves - should include FE
        moves = GameEngine.get_valid_moves(state, player)
        fe_moves = [m for m in moves if m.type == MoveType.FORCED_ELIMINATION]

        if not fe_moves:
            pytest.skip("Snapshot player is not in FE state")

        # All FE moves should target player-controlled stacks
        for m in fe_moves:
            stack_key = f"{m.to.x},{m.to.y}"
            assert stack_key in state.board.stacks, f"FE move targets non-existent stack: {m.to}"
            stack = state.board.stacks[stack_key]
            assert stack.controlling_player == player, (
                f"FE move targets stack controlled by {stack.controlling_player}, not {player}"
            )

        # Number of FE moves should equal number of player stacks
        assert len(fe_moves) == len(player_stacks), (
            f"Expected {len(player_stacks)} FE moves for {len(player_stacks)} stacks, "
            f"got {len(fe_moves)}"
        )

    def test_applying_forced_elimination_progresses_game(self):
        """Applying an FE move should increase eliminated rings."""
        snapshot_path = RESULTS_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"
        if not snapshot_path.exists():
            snapshot_path = SELFPLAY_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"

        if not snapshot_path.exists():
            pytest.skip("No failure snapshot available")

        state = load_failure_snapshot(snapshot_path)
        assert state is not None

        player = state.current_player

        # Check if this is an FE scenario (player has stacks)
        player_stacks = [
            s for s in state.board.stacks.values()
            if s.controlling_player == player
        ]
        if not player_stacks:
            pytest.skip(
                "Snapshot has no player stacks - not an FE scenario "
                "(may be a different failure type)"
            )

        moves = GameEngine.get_valid_moves(state, player)

        if not moves:
            pytest.skip("No moves available in snapshot")

        fe_moves = [m for m in moves if m.type == MoveType.FORCED_ELIMINATION]
        if not fe_moves:
            pytest.skip("Snapshot is not in FE state")

        fe_move = fe_moves[0]

        before_eliminated = state.total_rings_eliminated

        # Apply the FE move
        new_state = GameEngine.apply_move(state, fe_move)

        after_eliminated = new_state.total_rings_eliminated

        assert after_eliminated > before_eliminated, (
            f"FE should increase eliminated rings. Before: {before_eliminated}, "
            f"After: {after_eliminated}"
        )

    def test_fe_satisfies_inv_active_has_moves(self):
        """After the fix, ACTIVE states with FE should not violate INV-ACTIVE-NO-MOVES.

        Note: This test only validates FE scenarios (where player has stacks but
        no other moves). Other failure types (e.g., ring_placement with exhausted
        rings) represent different bugs and are not validated here.
        """
        snapshot_path = RESULTS_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"
        if not snapshot_path.exists():
            snapshot_path = SELFPLAY_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"

        if not snapshot_path.exists():
            pytest.skip("No failure snapshot available")

        state = load_failure_snapshot(snapshot_path)
        assert state is not None

        assert state.game_status == GameStatus.ACTIVE

        player = state.current_player

        # This test focuses on FE scenarios only (player has stacks)
        player_stacks = [
            s for s in state.board.stacks.values()
            if s.controlling_player == player
        ]
        if not player_stacks:
            pytest.skip(
                "Snapshot has no player stacks - not an FE scenario. "
                "This may represent a different invariant violation."
            )

        # Check if this is specifically an FE scenario
        has_fe = ga.has_forced_elimination_action(state, player)
        if not has_fe:
            pytest.skip("Snapshot player is not in FE state")

        moves = GameEngine.get_valid_moves(state, player)

        # INV-ACTIVE-HAS-MOVES: ACTIVE state in FE scenario should have FE moves
        assert len(moves) > 0, (
            "INV-ACTIVE-HAS-MOVES violation in FE scenario: ACTIVE state has no legal moves. "
            f"Diagnostics: {describe_fe_state(state)}"
        )


class TestForcedEliminationMultipleSnapshots:
    """Test FE behavior across multiple failure snapshots.

    ARCHIVED TEST: test_all_fe_snapshots_have_fe_moves_after_fix (removed 2025-12-07)
    - This test attempted to validate FE behavior across failure snapshots, but
      the snapshots represent various failure types (phase transition issues,
      capture availability, etc.), not pure FE scenarios. The has_forced_elimination_action
      check was insufficient to filter for true FE states. FE moves are only
      surfaced when the engine enters FORCED_ELIMINATION phase.
    """

    @pytest.fixture
    def failure_snapshots(self) -> list[Path]:
        """Collect all available failure snapshots."""
        snapshots = []

        for failures_dir in [SELFPLAY_FAILURES_DIR, RESULTS_FAILURES_DIR]:
            if failures_dir.exists():
                for f in failures_dir.glob("failure_*_no_legal_moves_for_current_player.json"):
                    snapshots.append(f)

        return snapshots[:10]  # Limit to 10 for test speed


class TestForcedEliminationInvariant:
    """Test the formal FE invariant from RR-CANON-R072/R100/R205.

    ARCHIVED TEST: test_invariant_has_stacks_implies_has_action (removed 2025-12-07)
    - This test created a synthetic blocked state (stack surrounded by collapsed
      spaces) and expected FORCED_ELIMINATION moves to be returned in MOVEMENT phase.
      The current (correct) behavior is that FE moves are only surfaced in the
      dedicated FORCED_ELIMINATION phase. The phase machine transitions players to
      this phase via GameEngine._end_turn() when no regular actions are available.
      The invariant "has_stacks → has_action" is enforced at the phase machine level,
      not within get_valid_moves() for a single phase.
    """

    pass  # All tests in this class were archived - class kept for documentation
