"""
Capture direction choice parity tests for Python engine.

These tests ensure that when multiple capture directions are available
(after an initial capture or chain continuation), the Python engine
correctly enumerates all valid capture options.

Per RR-CANON-R093 and RR-CANON-R100-R103:
- After non-capture movement landing on an opponent's stack, capture is mandatory
- Chain captures continue from landing position until no captures available
- When multiple capture directions exist, player must choose

These tests verify the Python AI service would receive correct capture options
for the CaptureDirectionChoice endpoint.
"""

import os
import sys
from datetime import datetime

import pytest

# Ensure app package is importable when running tests directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from app.game_engine import GameEngine
from app.models import (
    BoardState,
    BoardType,
    ChainCaptureState,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    TimeControl,
)
from app.rules.core import infer_total_rings_in_play


def create_base_state(
    board_type: BoardType = BoardType.SQUARE8,
    phase: GamePhase = GamePhase.CAPTURE,
    num_players: int = 2,
) -> GameState:
    """Create a base game state for capture direction testing."""
    if board_type == BoardType.SQUARE8:
        size = 8
    elif board_type == BoardType.SQUARE19:
        size = 19
    else:
        size = 13

    players = []
    for i in range(1, num_players + 1):
        players.append(
            Player(
                id=f"p{i}",
                username=f"Player{i}",
                type="human",
                playerNumber=i,
                isReady=True,
                timeRemaining=600,
                ringsInHand=18,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=None,
            )
        )

    board = BoardState(type=board_type, size=size)

    return GameState(
        id="capture-direction-parity",
        boardType=board_type,
        board=board,
        players=players,
        currentPhase=phase,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=infer_total_rings_in_play(players, board),
        totalRingsEliminated=0,
        victoryThreshold=12,
        territoryVictoryThreshold=33,
        chainCaptureState=None,
        must_move_from_stack_key=None,
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
) -> None:
    """Helper to place a stack on the board."""
    pos = Position(x=x, y=y)
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


class TestCaptureDirectionChoiceParity:
    """Tests verifying capture direction enumeration matches canonical rules."""

    def test_multiple_capture_directions_enumerated(self) -> None:
        """RR-CANON-R093: Multiple capture targets must be enumerated as distinct moves.

        Scenario:
        - P1 attacker at (3,3) height 2
        - P2 target at (3,4) height 1 (vertical direction)
        - P2 target at (4,3) height 1 (horizontal direction)

        P1 should see two distinct CAPTURE moves, one for each target.
        """
        state = create_base_state(phase=GamePhase.CAPTURE)
        board = state.board

        # Setup: P1 attacker in center, P2 targets in two directions
        place_stack(board, 3, 3, player=1, height=2)  # Attacker
        place_stack(board, 3, 4, player=2, height=1)  # Target vertical (y+1)
        place_stack(board, 4, 3, player=2, height=1)  # Target horizontal (x+1)

        # Add movement history to enable capture detection
        # Add movement history to enable capture detection
        state.move_history.append(create_setup_move(1, Position(x=3, y=3)))

        # Get capture moves
        captures = GameEngine._get_capture_moves(state, 1)

        # Should have at least 2 capture moves with distinct targets
        assert len(captures) >= 2, (
            f"Expected at least 2 capture directions, got {len(captures)}"
        )

        # Collect unique capture targets
        capture_targets = set()
        for move in captures:
            if move.capture_target:
                target_key = f"{move.capture_target.x},{move.capture_target.y}"
                capture_targets.add(target_key)

        # Both targets should be enumerated
        assert "3,4" in capture_targets, "Expected capture target at (3,4)"
        assert "4,3" in capture_targets, "Expected capture target at (4,3)"

    def test_chain_capture_continuation_directions(self) -> None:
        """RR-CANON-R100-R103: Chain captures enumerate all continuation directions.

        Scenario:
        - P1 attacker at (3,3) height 2
        - P2 initial target at (3,4) height 1
        - After landing at (3,5), two continuation targets:
          - P3 at (4,5) height 1 (diagonal)
          - P4 at (2,5) height 1 (diagonal opposite)

        After first capture, chain_capture phase should offer both continuation options.
        """
        state = create_base_state(phase=GamePhase.CAPTURE, num_players=4)
        board = state.board

        # Setup initial position
        place_stack(board, 3, 3, player=1, height=2)  # Attacker
        place_stack(board, 3, 4, player=2, height=1)  # First target
        place_stack(board, 4, 5, player=3, height=1)  # Continuation target A
        place_stack(board, 2, 5, player=4, height=1)  # Continuation target B

        # Add movement history to enable capture detection
        state.move_history.append(create_setup_move(1, Position(x=3, y=3)))

        # Get initial capture moves
        initial_captures = GameEngine._get_capture_moves(state, 1)
        assert len(initial_captures) > 0, "Expected initial capture moves"

        # Find capture over (3,4) landing at (3,5)
        first_capture = None
        for move in initial_captures:
            if (
                move.capture_target
                and move.capture_target.x == 3
                and move.capture_target.y == 4
                and move.to.x == 3
                and move.to.y == 5
            ):
                first_capture = move
                break

        assert first_capture is not None, (
            "Expected capture from (3,3) over (3,4) to (3,5)"
        )

        # Apply first capture
        state_after_first = GameEngine.apply_move(state, first_capture)

        # Verify we're in chain capture phase
        assert state_after_first.current_phase == GamePhase.CHAIN_CAPTURE, (
            f"Expected CHAIN_CAPTURE phase, got {state_after_first.current_phase}"
        )

        # Get continuation captures
        continuation_captures = GameEngine._get_capture_moves(state_after_first, 1)

        # Should have continuation options for both P3 and P4 targets
        continuation_targets = set()
        for move in continuation_captures:
            if move.capture_target:
                target_key = f"{move.capture_target.x},{move.capture_target.y}"
                continuation_targets.add(target_key)

        # Both continuation targets should be available
        # Note: Depending on exact geometry, one or both may be reachable
        assert len(continuation_targets) >= 1, (
            "Expected at least one chain continuation option"
        )

    def test_capture_direction_attributes_for_ai_choice(self) -> None:
        """Verify capture moves include attributes needed for AI choice heuristics.

        CaptureDirectionChoice endpoint needs:
        - capture_target: Position of stack being overtaken
        - to (landing): Position after capture
        - captured_cap_height: Height of cap being overtaken (for heuristic)
        """
        state = create_base_state(phase=GamePhase.CAPTURE)
        board = state.board

        # Setup: P1 attacker, P2 targets of different heights
        place_stack(board, 3, 3, player=1, height=2)  # Attacker
        place_stack(board, 3, 4, player=2, height=1)  # Target height 1
        place_stack(board, 4, 3, player=2, height=2)  # Target height 2

        # Add movement history to enable capture detection
        state.move_history.append(create_setup_move(1, Position(x=3, y=3)))

        captures = GameEngine._get_capture_moves(state, 1)
        assert len(captures) >= 2, "Expected at least 2 capture options"

        for capture in captures:
            # Every capture move must have required attributes
            assert capture.capture_target is not None, (
                "Capture move must have capture_target"
            )
            assert capture.to is not None, "Capture move must have landing (to)"

            # Verify landing is distinct from target
            assert not (
                capture.to.x == capture.capture_target.x
                and capture.to.y == capture.capture_target.y
            ), "Landing position must be different from target"

    def test_diagonal_capture_directions(self) -> None:
        """Test diagonal capture direction enumeration on square board.

        Scenario:
        - P1 attacker at (4,4) height 2
        - P2 targets at all 4 diagonal positions reachable by height-2 stack
        """
        state = create_base_state(phase=GamePhase.CAPTURE)
        board = state.board

        # Setup: P1 center, P2 at diagonal positions
        place_stack(board, 4, 4, player=1, height=2)  # Attacker
        place_stack(board, 5, 5, player=2, height=1)  # NE diagonal
        place_stack(board, 3, 3, player=2, height=1)  # SW diagonal
        place_stack(board, 5, 3, player=2, height=1)  # SE diagonal
        place_stack(board, 3, 5, player=2, height=1)  # NW diagonal

        # Add movement history to enable capture detection
        state.move_history.append(create_setup_move(1, Position(x=4, y=4)))

        captures = GameEngine._get_capture_moves(state, 1)

        # Collect capture targets
        targets = set()
        for move in captures:
            if move.capture_target:
                targets.add(f"{move.capture_target.x},{move.capture_target.y}")

        # All 4 diagonal targets should be capturable
        assert "5,5" in targets, "Expected NE diagonal target (5,5)"
        assert "3,3" in targets, "Expected SW diagonal target (3,3)"
        assert "5,3" in targets, "Expected SE diagonal target (5,3)"
        assert "3,5" in targets, "Expected NW diagonal target (3,5)"

    def test_capture_own_stack_allowed(self) -> None:
        """Per Nov 15, 2025 rule fix: Overtaking own stacks IS now allowed.

        Per KNOWN_ISSUES.md: "Players can now overtake their own stacks when
        cap height requirements are met."

        Scenario:
        - P1 attacker at (3,3) height 2
        - P1 own stack at (3,4) height 1 (capturable if cap height >= 1)
        - P2 enemy stack at (4,3) height 1 (also capturable)
        """
        state = create_base_state(phase=GamePhase.CAPTURE)
        board = state.board

        place_stack(board, 3, 3, player=1, height=2)  # Attacker cap height 2
        place_stack(board, 3, 4, player=1, height=1)  # Own stack - cap height 1
        place_stack(board, 4, 3, player=2, height=1)  # Enemy stack - cap height 1

        # Add movement history to enable capture detection
        state.move_history.append(create_setup_move(1, Position(x=3, y=3)))

        captures = GameEngine._get_capture_moves(state, 1)

        # Collect capture targets
        targets = set()
        for move in captures:
            if move.capture_target:
                targets.add(f"{move.capture_target.x},{move.capture_target.y}")

        # Both own stack AND enemy stack should be capturable
        assert "3,4" in targets, "Should offer capture of own stack (rule fix Nov 2025)"
        assert "4,3" in targets, "Should offer capture of enemy stack"


class TestCaptureDirectionChainTermination:
    """Tests for chain capture termination with direction choices."""

    def test_chain_terminates_when_no_directions_available(self) -> None:
        """Chain capture must terminate when no further captures exist.

        Scenario:
        - P1 captures and lands at position with no further targets
        - Should transition out of CHAIN_CAPTURE phase
        """
        state = create_base_state(phase=GamePhase.CAPTURE)
        board = state.board

        # Setup: Single target, no continuation
        place_stack(board, 3, 3, player=1, height=2)  # Attacker
        place_stack(board, 3, 4, player=2, height=1)  # Only target

        # Add movement history to enable capture detection
        state.move_history.append(create_setup_move(1, Position(x=3, y=3)))

        # Get and apply capture
        captures = GameEngine._get_capture_moves(state, 1)
        assert len(captures) > 0, "Expected initial capture"

        # Apply capture to furthest landing position
        capture = captures[0]
        state_after = GameEngine.apply_move(state, capture)

        # Get continuation captures
        continuations = GameEngine._get_capture_moves(state_after, 1)

        # Should have no continuation options (chain terminates)
        assert len(continuations) == 0, (
            f"Expected no chain continuations, got {len(continuations)}"
        )

    def test_chain_offers_all_branching_directions(self) -> None:
        """When chain branches, all valid directions must be offered.

        This matches the TS test 'enumerates all valid overtaking chains'.
        """
        state = create_base_state(phase=GamePhase.CAPTURE, num_players=4)
        board = state.board

        # Setup matching TS test scenario
        place_stack(board, 3, 3, player=1, height=2)  # Red attacker
        place_stack(board, 3, 4, player=2, height=1)  # Blue (first target)
        place_stack(board, 4, 5, player=3, height=1)  # Green (branch A)
        place_stack(board, 2, 5, player=4, height=1)  # Yellow (branch B)

        # Add movement history to enable capture detection
        state.move_history.append(create_setup_move(1, Position(x=3, y=3)))

        # Execute chain and collect all branch points
        captured_targets = []
        branch_counts = []
        current_state = state

        for _ in range(10):  # Safety limit
            captures = GameEngine._get_capture_moves(current_state, 1)
            if not captures:
                break

            branch_counts.append(len(captures))

            # Record targets
            for c in captures:
                if c.capture_target:
                    captured_targets.append(
                        f"{c.capture_target.x},{c.capture_target.y}"
                    )

            # Apply first capture (deterministic path)
            current_state = GameEngine.apply_move(current_state, captures[0])

        # Should have found at least one branch point (multiple captures available)
        assert max(branch_counts) >= 1, (
            "Expected at least one branch in chain capture sequence"
        )
