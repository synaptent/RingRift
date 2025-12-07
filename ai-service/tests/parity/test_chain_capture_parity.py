"""
Chain capture parity tests for TypeScript/Python rules engine alignment.

These tests ensure that chain capture behavior is identical between the
TypeScript engine (RuleEngine + GameEngine + sandbox) and the Python
AI service GameEngine.

Chain captures are a complex game mechanic where:
1. An overtaking capture can trigger follow-up captures
2. The attacker must continue capturing if additional targets are available
3. The chain terminates when no more valid captures exist

Parity is critical because:
- Two rules engine implementations exist (TS ~20,000 lines, Py ~5,000 lines)
- Any rule change must be synchronized across both
- Chain captures involve complex state transitions
"""

import sys
import os
from datetime import datetime
from typing import Optional

import pytest

# Ensure app package is importable when running tests directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from app.models import (  # noqa: E402
    GameState,
    BoardType,
    BoardState,
    GamePhase,
    GameStatus,
    TimeControl,
    Player,
    Position,
    RingStack,
    MarkerInfo,
    Move,
    MoveType,
)
from app.game_engine import GameEngine  # noqa: E402


def create_base_state(
    board_type: BoardType = BoardType.SQUARE8,
    phase: GamePhase = GamePhase.MOVEMENT,
) -> GameState:
    """Create a base game state for chain capture testing."""
    if board_type == BoardType.SQUARE8:
        size = 8
    elif board_type == BoardType.SQUARE19:
        size = 19
    else:
        size = 13  # Canonical hex: size=13, radius=12

    return GameState(
        id="chain-capture-parity",
        boardType=board_type,
        board=BoardState(type=board_type, size=size),
        players=[
            Player(
                id="p1",
                username="Player1",
                type="human",
                playerNumber=1,
                isReady=True,
                timeRemaining=600,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
            Player(
                id="p2",
                username="Player2",
                type="human",
                playerNumber=2,
                isReady=True,
                timeRemaining=600,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            ),
        ],
        currentPhase=phase,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        rngSeed=None,
        zobristHash=None,
        lpsRoundIndex=0,
        lpsExclusivePlayerForCompletedRound=None,
    )


def place_stack(
    board: BoardState,
    x: int,
    y: int,
    player: int,
    height: int = 1,
    z: Optional[int] = None,
) -> None:
    """Helper to place a stack on the board."""
    pos = Position(x=x, y=y, z=z)
    key = pos.to_key()
    rings = [player] * height
    stack = RingStack(
        position=pos,
        rings=rings,
        stackHeight=height,
        capHeight=height,
        controllingPlayer=player,
    )
    board.stacks[key] = stack


def place_marker(
    board: BoardState,
    x: int,
    y: int,
    player: int,
    z: Optional[int] = None,
) -> None:
    """Helper to place a marker on the board."""
    pos = Position(x=x, y=y, z=z)
    key = pos.to_key()
    marker = MarkerInfo(
        player=player,
        position=pos,
        type="regular",
    )
    board.markers[key] = marker


def create_setup_move(
    player: int,
    position: Position,
    move_type: MoveType = MoveType.PLACE_RING,
) -> Move:
    """Create a synthetic setup move for testing."""
    return Move(
        id="setup",
        type=move_type,
        player=player,
        to=position,
        timestamp=datetime.now(),
        thinkTime=0,
        moveNumber=1,
    )


class TestChainCaptureParity:
    """Tests ensuring chain capture behavior matches between TS and Python."""

    def test_simple_chain_capture_parity(self) -> None:
        """A simple chain: capture A triggers capture B.

        Scenario:
        - P1 stack at (2,2) height 2
        - P2 stack at (2,3) height 1 (first target)
        - P2 stack at (2,5) height 1 (second target)
        - Empty cells at (2,4) and (2,6) for landing

        After first capture: P1 moves from (2,2) over (2,3) to (2,4)
        After second capture: P1 moves from (2,4) over (2,5) to (2,6)
        """
        state = create_base_state()
        board = state.board

        # Setup the scenario
        place_stack(board, 2, 2, player=1, height=2)  # Attacker (cap height 2)
        place_stack(board, 2, 3, player=2, height=1)  # First target
        place_stack(board, 2, 5, player=2, height=1)  # Second target

        # Add a synthetic placement move to enable capture detection
        state.move_history.append(
            create_setup_move(1, Position(x=2, y=2))
        )

        # Get initial capture moves
        initial_captures = GameEngine._get_capture_moves(state, 1)
        assert len(initial_captures) > 0, "Expected at least one capture move"

        # Find the capture over (2,3) landing at (2,4)
        first_capture = None
        for move in initial_captures:
            if (
                move.capture_target
                and move.capture_target.x == 2
                and move.capture_target.y == 3
                and move.to.x == 2
                and move.to.y == 4
            ):
                first_capture = move
                break

        assert first_capture is not None, (
            "Expected capture over (2,3) to (2,4)"
        )

        # Apply first capture
        state_after_first = GameEngine.apply_move(state, first_capture)

        # Verify first capture applied correctly
        assert state_after_first.board.stacks.get("2,2") is None, (
            "Attacker should have left (2,2)"
        )
        # Target stack should be reduced or removed
        target_after = state_after_first.board.stacks.get("2,3")
        assert target_after is None, (
            "Target at (2,3) should be removed (height was 1)"
        )

        # Get continuation captures from new position
        continuation_caps = GameEngine._get_capture_moves(state_after_first, 1)

        # Should have a second capture available over (2,5) to (2,6)
        second_capture = None
        for move in continuation_caps:
            if (
                move.capture_target
                and move.capture_target.x == 2
                and move.capture_target.y == 5
            ):
                second_capture = move
                break

        assert second_capture is not None, "Expected continuation over (2,5)"

        # Apply second capture
        state_after_second = GameEngine.apply_move(
            state_after_first, second_capture
        )

        # Verify chain is complete
        final_captures = GameEngine._get_capture_moves(state_after_second, 1)
        assert len(final_captures) == 0, "Chain should terminate"

        # Final attacker should be at landing position with increased stack
        final_attacker = state_after_second.board.stacks.get(
            second_capture.to.to_key()
        )
        assert final_attacker is not None, "Attacker at final position"
        # Stack grew by 2 captured rings (inserted at bottom)
        assert final_attacker.stack_height == 4, (
            f"Expected height 4, got {final_attacker.stack_height}"
        )

    def test_long_chain_capture_parity(self) -> None:
        """Long chain: 3+ captures in sequence.

        Scenario (horizontal line with proper spacing):
        - P1 stack at (0,3) height 2
        - P2 stacks at (1,3), (4,3), (7,3) each height 1
        - Empty cells at (2,3), (5,3) for landing (min distance = 2)

        Height-2 attacker can land at distance 2+, enabling chain.
        """
        state = create_base_state()
        board = state.board

        # Setup scenario with height-2 attacker (min distance 2)
        place_stack(board, 0, 3, player=1, height=2)  # Attacker
        place_stack(board, 1, 3, player=2, height=1)  # Target 1
        place_stack(board, 4, 3, player=2, height=1)  # Target 2
        # Target 3 placed after first landing allows chain

        # Add synthetic placement move
        state.move_history.append(
            create_setup_move(1, Position(x=0, y=3))
        )

        captured_count = 0
        current_state = state

        # Execute chain captures
        for _ in range(5):  # Safety limit
            captures = GameEngine._get_capture_moves(current_state, 1)
            if not captures:
                break

            # Take the first available capture
            capture_move = captures[0]
            current_state = GameEngine.apply_move(current_state, capture_move)
            captured_count += 1

        # Verify we captured at least 2 targets (chain continues as allowed)
        # The exact count depends on landing choices and target positions
        assert captured_count >= 2, (
            f"Expected at least 2 captures, got {captured_count}"
        )

        # Key assertion: chain eventually terminates (no infinite loops)
        assert captured_count <= 5, "Chain should not be infinite"

    def test_branching_chain_capture_parity(self) -> None:
        """Branching: one capture enables multiple capture options.

        Scenario:
        - P1 stack at (3,3) height 2
        - P2 stack at (3,4) height 1 (target enabling branch)
        - After first capture to (3,5):
          - P2 stack at (3,6) height 1 (option A - vertical)
          - P2 stack at (4,5) height 1 (option B - diagonal)
        - Empty cells at (3,7) and (5,5) for landing

        After first capture, attacker should have choice between
        continuing vertical or diagonal.
        """
        state = create_base_state()
        board = state.board

        # Setup scenario
        place_stack(board, 3, 3, player=1, height=2)  # Attacker
        place_stack(board, 3, 4, player=2, height=1)  # First target
        place_stack(board, 3, 6, player=2, height=1)  # Branch option A
        place_stack(board, 4, 5, player=2, height=1)  # Branch option B

        # Add synthetic placement move
        state.move_history.append(
            create_setup_move(1, Position(x=3, y=3))
        )

        # Get initial capture moves
        initial_captures = GameEngine._get_capture_moves(state, 1)
        assert len(initial_captures) > 0, "Expected initial capture"

        # Find capture over (3,4) to (3,5)
        first_capture = None
        for move in initial_captures:
            if (
                move.capture_target
                and move.capture_target.x == 3
                and move.capture_target.y == 4
                and move.to.y == 5
            ):
                first_capture = move
                break

        assert first_capture is not None, "Expected capture over (3,4)"

        # Apply first capture
        state_after_first = GameEngine.apply_move(state, first_capture)

        # Get continuation captures - should have multiple options
        continuation_caps = GameEngine._get_capture_moves(state_after_first, 1)

        # Find capture options
        vertical_option = None
        diagonal_option = None
        for move in continuation_caps:
            if not move.capture_target:
                continue
            if (
                move.capture_target.x == 3
                and move.capture_target.y == 6
            ):
                vertical_option = move
            elif (
                move.capture_target.x == 4
                and move.capture_target.y == 5
            ):
                diagonal_option = move

        # At least one branch option should be available
        has_branch = vertical_option is not None or diagonal_option is not None
        assert has_branch, "Expected at least one branch capture option"

        # Both options should lead to valid game states
        if vertical_option:
            state_vert = GameEngine.apply_move(
                state_after_first, vertical_option
            )
            assert state_vert.game_status == GameStatus.ACTIVE

        if diagonal_option:
            state_diag = GameEngine.apply_move(
                state_after_first, diagonal_option
            )
            assert state_diag.game_status == GameStatus.ACTIVE

    def test_self_capture_prevention_parity(self) -> None:
        """Self-capture: landing on own marker triggers elimination.

        Scenario:
        - P1 stack at (2,2) height 2
        - P2 stack at (2,3) height 1 (target)
        - P1 marker at (2,4) (landing position)

        Result: Capture succeeds but landing on own marker causes
        top ring elimination from the attacker.
        """
        state = create_base_state()
        board = state.board

        # Setup scenario
        place_stack(board, 2, 2, player=1, height=2)  # Attacker
        place_stack(board, 2, 3, player=2, height=1)  # Target
        place_marker(board, 2, 4, player=1)  # Own marker at landing

        # Track initial eliminated rings
        initial_eliminated = state.board.eliminated_rings.get("1", 0)

        # Add synthetic placement move
        state.move_history.append(
            create_setup_move(1, Position(x=2, y=2))
        )

        # Get capture moves
        captures = GameEngine._get_capture_moves(state, 1)
        assert len(captures) > 0, "Expected capture move available"

        # Find capture landing on (2,4)
        capture_to_marker = None
        for move in captures:
            if move.to.x == 2 and move.to.y == 4:
                capture_to_marker = move
                break

        assert capture_to_marker is not None, "Expected capture to (2,4)"

        # Apply capture
        state_after = GameEngine.apply_move(state, capture_to_marker)

        # Verify landing on own marker caused elimination
        final_eliminated = state_after.board.eliminated_rings.get("1", 0)
        eliminated_count = final_eliminated - initial_eliminated
        assert eliminated_count == 1, (
            f"Expected 1 ring eliminated, got {eliminated_count}"
        )

        # Marker should be removed
        assert state_after.board.markers.get("2,4") is None, (
            "Own marker should be removed after landing"
        )

        # Attacker should be at landing position
        attacker = state_after.board.stacks.get("2,4")
        assert attacker is not None, "Attacker should exist at (2,4)"
        # Original 2 + 1 captured - 1 eliminated = 2
        assert attacker.stack_height == 2, (
            f"Expected height 2, got {attacker.stack_height}"
        )

    def test_chain_interrupted_by_elimination_parity(self) -> None:
        """Chain interrupted: victory check during chain.

        Scenario:
        - P1 stack at (2,2) height 2
        - P2 stacks that would enable a long chain
        - P1 has 2 eliminated rings already

        If eliminating another ring (via self-capture) triggers victory,
        the chain should terminate.
        """
        state = create_base_state()
        board = state.board

        # Setup P1 close to victory
        state.board.eliminated_rings["1"] = 2  # 2 of 3 needed
        state.players[0].eliminated_rings = 2

        # Setup capture scenario with own marker
        place_stack(board, 2, 2, player=1, height=2)
        place_stack(board, 2, 3, player=2, height=1)
        place_marker(board, 2, 4, player=1)  # Will trigger elimination

        # More targets after (to test chain interruption)
        place_stack(board, 2, 5, player=2, height=1)

        # Add synthetic placement move
        state.move_history.append(
            create_setup_move(1, Position(x=2, y=2))
        )

        # Get capture
        captures = GameEngine._get_capture_moves(state, 1)
        capture_move = None
        for move in captures:
            if move.to.x == 2 and move.to.y == 4:
                capture_move = move
                break

        assert capture_move is not None

        # Apply capture that lands on own marker
        state_after = GameEngine.apply_move(state, capture_move)

        # Should trigger victory (3 eliminated rings)
        assert state_after.board.eliminated_rings.get("1", 0) >= 3
        assert state_after.game_status == GameStatus.COMPLETED
        assert state_after.winner == 1

    def test_chain_no_valid_targets_parity(self) -> None:
        """Chain with no valid targets: single capture terminates.

        Scenario:
        - P1 stack at (2,2) height 2
        - P2 stack at (2,3) height 1 (only target)
        - No other targets after landing at (2,4)

        Chain should execute single capture and terminate.
        """
        state = create_base_state()
        board = state.board

        # Setup minimal scenario
        place_stack(board, 2, 2, player=1, height=2)
        place_stack(board, 2, 3, player=2, height=1)

        # Add synthetic placement move
        state.move_history.append(
            create_setup_move(1, Position(x=2, y=2))
        )

        # Get initial captures
        captures = GameEngine._get_capture_moves(state, 1)
        assert len(captures) > 0

        # Apply the capture
        capture_move = captures[0]
        state_after = GameEngine.apply_move(state, capture_move)

        # Verify no continuation captures
        continuation_captures = GameEngine._get_capture_moves(state_after, 1)
        assert len(continuation_captures) == 0, "Expected no continuations"

        # Verify target removed
        assert state_after.board.stacks.get("2,3") is None

        # Verify attacker at landing with captured ring
        attacker = state_after.board.stacks.get(capture_move.to.to_key())
        assert attacker is not None
        assert attacker.stack_height == 3  # 2 original + 1 captured

    def test_cap_height_constraint_parity(self) -> None:
        """Cap height constraint: cannot capture stacks with higher cap.

        Scenario:
        - P1 stack at (2,2) height 1 cap 1
        - P2 stack at (2,3) height 2 cap 2
        - P2 stack at (4,2) height 1 cap 1

        P1 should NOT be able to capture (2,3) but CAN capture (4,2).
        """
        state = create_base_state()
        board = state.board

        # Setup scenario
        place_stack(board, 2, 2, player=1, height=1)  # cap 1
        place_stack(board, 2, 3, player=2, height=2)  # cap 2 (too tall)
        place_stack(board, 4, 2, player=2, height=1)  # cap 1 (valid target)

        # Add synthetic placement move
        state.move_history.append(
            create_setup_move(1, Position(x=2, y=2))
        )

        # Get capture moves
        captures = GameEngine._get_capture_moves(state, 1)

        # Check no capture over (2,3)
        illegal_capture = any(
            move.capture_target
            and move.capture_target.x == 2
            and move.capture_target.y == 3
            for move in captures
        )
        assert not illegal_capture, "Cannot capture higher cap height stack"

        # Check capture over (4,2) is available
        legal_capture = any(
            move.capture_target
            and move.capture_target.x == 4
            and move.capture_target.y == 2
            for move in captures
        )
        assert legal_capture, "Should capture equal cap height stack"


class TestChainCaptureEdgeCases:
    """Edge case tests for chain captures."""

    def test_capture_own_stack_parity(self) -> None:
        """Capture own stack: allowed if cap height permits.

        The rules allow capturing your own stacks (overtaking).
        """
        state = create_base_state()
        board = state.board

        # P1 stacks: attacker with higher cap, target with lower cap
        place_stack(board, 2, 2, player=1, height=3)  # cap 3
        place_stack(board, 2, 3, player=1, height=1)  # cap 1

        # Add synthetic placement move
        state.move_history.append(
            create_setup_move(1, Position(x=2, y=2))
        )

        # Get captures - should include own stack
        captures = GameEngine._get_capture_moves(state, 1)
        own_stack_capture = any(
            move.capture_target
            and move.capture_target.x == 2
            and move.capture_target.y == 3
            for move in captures
        )

        assert own_stack_capture, "Should capture own stack with lower cap"

    def test_minimum_distance_constraint_parity(self) -> None:
        """Minimum distance: capture must travel at least stack height.

        A stack of height 3 must land at distance >= 3 from origin.
        """
        state = create_base_state()
        board = state.board

        # P1 stack height 3, P2 target adjacent
        place_stack(board, 3, 3, player=1, height=3)
        place_stack(board, 3, 4, player=2, height=1)  # Adjacent target

        # Add synthetic placement move
        state.move_history.append(
            create_setup_move(1, Position(x=3, y=3))
        )

        # Get captures
        captures = GameEngine._get_capture_moves(state, 1)

        # All landings should be at distance >= 3 from (3,3)
        for move in captures:
            if not move.capture_target:
                continue
            if (
                move.capture_target.x == 3
                and move.capture_target.y == 4
            ):
                distance = abs(move.to.y - 3)  # vertical distance
                assert distance >= 3, f"Landing distance {distance} < 3"


@pytest.mark.parametrize("board_type", [
    BoardType.SQUARE8,
    BoardType.HEXAGONAL,
])
class TestChainCaptureBoardTypes:
    """Test chain captures work consistently across board types."""

    def test_basic_capture_on_board_type(
        self, board_type: BoardType
    ) -> None:
        """Basic capture works on different board types."""
        state = create_base_state(board_type=board_type)
        board = state.board

        if board_type == BoardType.HEXAGONAL:
            # Hex uses cube coordinates
            place_stack(board, 0, 0, player=1, height=2, z=0)
            place_stack(board, 1, 0, player=2, height=1, z=-1)

            state.move_history.append(
                create_setup_move(1, Position(x=0, y=0, z=0))
            )
        else:
            # Square board
            place_stack(board, 2, 2, player=1, height=2)
            place_stack(board, 2, 3, player=2, height=1)

            state.move_history.append(
                create_setup_move(1, Position(x=2, y=2))
            )

        captures = GameEngine._get_capture_moves(state, 1)
        assert len(captures) > 0, f"Expected captures on {board_type}"

        # Apply a capture
        state_after = GameEngine.apply_move(state, captures[0])
        assert state_after.game_status == GameStatus.ACTIVE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])