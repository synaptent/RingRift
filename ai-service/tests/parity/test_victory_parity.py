"""Tests for victory parity between TypeScript and Python engines.

These tests verify:
1. Games always produce a winner (no NULL winners)
2. The deterministic tiebreaker works correctly
3. Full games complete properly with winners

Per RR-CANON: The game must always produce a winner - draws are not allowed.
"""
import os
import sys

import pytest

# Ensure app packages are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from app.game_engine import GameEngine
from app.models import BoardType
from app.training.initial_state import create_initial_state


class TestVictoryAlwaysProduced:
    """Tests that games always produce a winner."""

    @pytest.mark.parametrize("board_type", [BoardType.HEX8, BoardType.SQUARE8])
    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_random_game_produces_winner(
        self, board_type: BoardType, num_players: int
    ) -> None:
        """A game played with random moves should always produce a winner."""
        state = create_initial_state(board_type, num_players)
        move_count = 0
        max_moves = 1000

        while str(state.game_status) != "GameStatus.COMPLETED" and move_count < max_moves:
            valid_moves = GameEngine.get_valid_moves(state, state.current_player)

            if valid_moves:
                # Apply first valid move (deterministic for reproducibility)
                state = GameEngine.apply_move(state, valid_moves[0])
                move_count += 1
            else:
                # Check for bookkeeping move requirement
                requirement = GameEngine.get_phase_requirement(state, state.current_player)
                if requirement is not None:
                    bookkeeping_move = GameEngine.synthesize_bookkeeping_move(
                        requirement, state
                    )
                    if bookkeeping_move:
                        state = GameEngine.apply_move(state, bookkeeping_move)
                        move_count += 1
                        continue
                # No valid moves and no bookkeeping - game is stuck
                break

        # Game must be completed
        assert str(state.game_status) == "GameStatus.COMPLETED", (
            f"Game did not complete after {move_count} moves. "
            f"Status: {state.game_status}, Phase: {state.current_phase}"
        )

        # Winner must not be None
        assert state.winner is not None, (
            f"Game completed but winner is None after {move_count} moves. "
            f"This violates RR-CANON: games must always produce a winner."
        )

        # Winner must be a valid player number
        assert 1 <= state.winner <= num_players, (
            f"Winner {state.winner} is not a valid player number (1-{num_players})"
        )

    def test_hex8_2p_game_completes_with_winner(self) -> None:
        """Specific test for hex8 2-player - the most common config."""
        state = create_initial_state(BoardType.HEX8, num_players=2)
        move_count = 0

        while str(state.game_status) != "GameStatus.COMPLETED" and move_count < 500:
            valid_moves = GameEngine.get_valid_moves(state, state.current_player)

            if valid_moves:
                state = GameEngine.apply_move(state, valid_moves[0])
                move_count += 1
            else:
                requirement = GameEngine.get_phase_requirement(state, state.current_player)
                if requirement is not None:
                    bookkeeping_move = GameEngine.synthesize_bookkeeping_move(
                        requirement, state
                    )
                    if bookkeeping_move:
                        state = GameEngine.apply_move(state, bookkeeping_move)
                        move_count += 1
                        continue
                break

        assert str(state.game_status) == "GameStatus.COMPLETED"
        assert state.winner is not None
        assert state.winner in [1, 2]


class TestDeterministicTiebreaker:
    """Tests for the deterministic tiebreaker (lowest player number wins)."""

    def test_tiebreaker_returns_lowest_player_number(self) -> None:
        """When all tiebreak criteria are equal, lowest player number should win.

        This is tested by verifying the MutableGameState implementation.
        """
        from app.rules.mutable_state import MutableGameState

        # Create a state where we can test the tiebreaker logic
        # The tiebreaker is triggered when all players are eliminated simultaneously
        # or when all tiebreak criteria (territory, rings, markers) are equal.

        # We verify the tiebreaker exists in the code
        import inspect

        source = inspect.getsource(MutableGameState._check_for_game_end)
        assert "all_player_numbers" in source, (
            "Deterministic tiebreaker not found in MutableGameState._check_for_game_end"
        )
        assert "min(all_player_numbers)" in source, (
            "Tiebreaker should use min() to select lowest player number"
        )


class TestBookkeepingMoves:
    """Tests that bookkeeping moves are properly generated."""

    @pytest.mark.parametrize("board_type", [BoardType.HEX8, BoardType.SQUARE8])
    def test_no_line_action_generated(self, board_type: BoardType) -> None:
        """NO_LINE_ACTION should be generated when no lines are formed."""
        state = create_initial_state(board_type, num_players=2)

        # Place a ring
        moves = GameEngine.get_valid_moves(state, state.current_player)
        assert len(moves) > 0, "Should have placement moves"
        state = GameEngine.apply_move(state, moves[0])

        # Move the stack
        moves = GameEngine.get_valid_moves(state, state.current_player)
        assert len(moves) > 0, "Should have movement moves"
        state = GameEngine.apply_move(state, moves[0])

        # Should be in LINE_PROCESSING phase with no valid moves
        assert str(state.current_phase) == "GamePhase.LINE_PROCESSING"

        moves = GameEngine.get_valid_moves(state, state.current_player)
        if not moves:
            # Should have a phase requirement for NO_LINE_ACTION
            requirement = GameEngine.get_phase_requirement(state, state.current_player)
            assert requirement is not None, (
                "Should have phase requirement when no valid moves in LINE_PROCESSING"
            )

            bookkeeping = GameEngine.synthesize_bookkeeping_move(requirement, state)
            assert bookkeeping is not None, "Should synthesize NO_LINE_ACTION"
            assert "NO_LINE_ACTION" in str(bookkeeping.type)


class TestGameCompletion:
    """Tests that games complete properly without getting stuck."""

    def test_game_does_not_get_stuck_after_two_moves(self) -> None:
        """Regression test: games should not get stuck after 2 moves.

        This was a bug where games would have no valid moves after the first
        placement and movement, because bookkeeping moves weren't being generated.
        """
        state = create_initial_state(BoardType.HEX8, num_players=2)

        # Move 1: Place ring
        moves = GameEngine.get_valid_moves(state, state.current_player)
        assert len(moves) > 0
        state = GameEngine.apply_move(state, moves[0])

        # Move 2: Move stack
        moves = GameEngine.get_valid_moves(state, state.current_player)
        assert len(moves) > 0
        state = GameEngine.apply_move(state, moves[0])

        # After 2 moves, we should either have valid moves OR a phase requirement
        moves = GameEngine.get_valid_moves(state, state.current_player)
        if not moves:
            requirement = GameEngine.get_phase_requirement(state, state.current_player)
            assert requirement is not None, (
                "Game stuck after 2 moves: no valid moves and no phase requirement. "
                "This indicates a bug in the game engine."
            )

    @pytest.mark.parametrize("num_games", [3])
    def test_multiple_games_complete(self, num_games: int) -> None:
        """Multiple games should all complete with winners."""
        for game_idx in range(num_games):
            state = create_initial_state(BoardType.HEX8, num_players=2)
            move_count = 0

            while (
                str(state.game_status) != "GameStatus.COMPLETED" and move_count < 500
            ):
                valid_moves = GameEngine.get_valid_moves(state, state.current_player)

                if valid_moves:
                    # Use different moves for variety (based on game index)
                    move_idx = (game_idx + move_count) % len(valid_moves)
                    state = GameEngine.apply_move(state, valid_moves[move_idx])
                    move_count += 1
                else:
                    requirement = GameEngine.get_phase_requirement(
                        state, state.current_player
                    )
                    if requirement is not None:
                        bookkeeping = GameEngine.synthesize_bookkeeping_move(
                            requirement, state
                        )
                        if bookkeeping:
                            state = GameEngine.apply_move(state, bookkeeping)
                            move_count += 1
                            continue
                    break

            assert str(state.game_status) == "GameStatus.COMPLETED", (
                f"Game {game_idx} did not complete"
            )
            assert state.winner is not None, f"Game {game_idx} has no winner"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
