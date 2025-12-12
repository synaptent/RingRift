"""
Unit tests for the recovery action implementation.

Tests the Python recovery module against the canonical RR-CANON-R110–R115 rules.
"""

import pytest
from datetime import datetime

from app.models import (
    GameState,
    Move,
    Position,
    MoveType,
    GamePhase,
    GameStatus,
    BoardState,
    BoardType,
    RingStack,
    MarkerInfo,
    Player,
    TimeControl,
)
from app.rules.core import (
    count_buried_rings,
    player_has_markers,
    player_controls_any_stack,
    is_eligible_for_recovery,
)
from app.rules.recovery import (
    calculate_recovery_cost,
    enumerate_recovery_slide_targets,
    has_any_recovery_move,
    validate_recovery_slide,
    apply_recovery_slide,
    get_recovery_moves,
)


def create_test_state(
    board_type: BoardType = BoardType.SQUARE8,
    current_player: int = 1,
    current_phase: GamePhase = GamePhase.MOVEMENT,
    rings_in_hand_p1: int = 0,
    rings_in_hand_p2: int = 0,
) -> GameState:
    """Create a basic test GameState for recovery tests."""
    board = BoardState(type=board_type, size=8)
    players = [
        Player(
            id="p1",
            username="p1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=rings_in_hand_p1,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="p2",
            username="p2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=rings_in_hand_p2,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    now = datetime.now()

    return GameState(
        id="test",
        boardType=board_type,
        board=board,
        players=players,
        currentPhase=current_phase,
        currentPlayer=current_player,
        timeControl=TimeControl(initialTime=60, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=now,
        lastMoveAt=now,
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )


def add_stack(state: GameState, pos: Position, rings: list[int]) -> None:
    """Add a stack to the board."""
    controlling_player = rings[-1] if rings else 0
    cap_height = 0
    for r in reversed(rings):
        if r == controlling_player:
            cap_height += 1
        else:
            break

    pos_key = pos.to_key()
    state.board.stacks[pos_key] = RingStack(
        position=pos,
        rings=rings,
        controlling_player=controlling_player,
        stack_height=len(rings),
        cap_height=cap_height,
    )


def add_marker(state: GameState, pos: Position, player: int) -> None:
    """Add a marker to the board."""
    pos_key = pos.to_key()
    state.board.markers[pos_key] = MarkerInfo(position=pos, player=player, type="regular")


class TestCountBuriedRings:
    """Tests for count_buried_rings helper."""

    def test_returns_zero_for_empty_board(self):
        state = create_test_state()
        assert count_buried_rings(state.board, 1) == 0

    def test_returns_zero_when_player_controls_all_stacks(self):
        state = create_test_state()
        add_stack(state, Position(x=3, y=3), [1, 1, 1])
        assert count_buried_rings(state.board, 1) == 0
        assert count_buried_rings(state.board, 2) == 0

    def test_counts_buried_rings_correctly(self):
        state = create_test_state()
        # P1 controls with P2 buried: [P2, P1] - P2 has 1 buried
        add_stack(state, Position(x=3, y=3), [2, 1])
        # P2 controls with P1 buried: [P1, P1, P2] - P1 has 2 buried
        add_stack(state, Position(x=4, y=4), [1, 1, 2])
        # P1 controls, pure: [P1] - no buried
        add_stack(state, Position(x=5, y=5), [1])

        assert count_buried_rings(state.board, 1) == 2
        assert count_buried_rings(state.board, 2) == 1


class TestPlayerHasMarkers:
    """Tests for player_has_markers helper."""

    def test_returns_false_for_no_markers(self):
        state = create_test_state()
        assert player_has_markers(state.board, 1) is False

    def test_returns_true_when_player_has_marker(self):
        state = create_test_state()
        add_marker(state, Position(x=3, y=3), 1)
        assert player_has_markers(state.board, 1) is True
        assert player_has_markers(state.board, 2) is False


class TestPlayerControlsAnyStack:
    """Tests for player_controls_any_stack helper."""

    def test_returns_false_for_empty_board(self):
        state = create_test_state()
        assert player_controls_any_stack(state.board, 1) is False

    def test_returns_true_when_player_controls_stack(self):
        state = create_test_state()
        add_stack(state, Position(x=3, y=3), [1])
        assert player_controls_any_stack(state.board, 1) is True
        assert player_controls_any_stack(state.board, 2) is False


class TestIsEligibleForRecovery:
    """Tests for is_eligible_for_recovery helper."""

    def test_eligible_with_rings_in_hand(self):
        """Per RR-CANON-R201: Recovery eligibility is independent of rings in hand."""
        state = create_test_state(rings_in_hand_p1=1)
        add_marker(state, Position(x=3, y=3), 1)
        add_stack(state, Position(x=4, y=4), [1, 2])  # P1 ring buried under P2
        # Player controls no stacks, has marker, has buried ring -> eligible
        # Rings in hand do NOT prevent recovery eligibility per RR-CANON-R201
        assert is_eligible_for_recovery(state, 1) is True

    def test_not_eligible_when_controls_stack(self):
        state = create_test_state()
        add_marker(state, Position(x=3, y=3), 1)
        add_stack(state, Position(x=4, y=4), [2, 1])  # P1 controls this
        add_stack(state, Position(x=5, y=5), [1, 2])  # P1 ring buried
        assert is_eligible_for_recovery(state, 1) is False

    def test_not_eligible_without_markers(self):
        state = create_test_state()
        add_stack(state, Position(x=4, y=4), [1, 2])  # P1 ring buried
        assert is_eligible_for_recovery(state, 1) is False

    def test_not_eligible_without_buried_rings(self):
        state = create_test_state()
        add_marker(state, Position(x=3, y=3), 1)
        assert is_eligible_for_recovery(state, 1) is False

    def test_eligible_when_all_conditions_met(self):
        state = create_test_state()
        # P1 has no rings in hand (default)
        # P1 has a marker
        add_marker(state, Position(x=3, y=3), 1)
        # P1 has a buried ring (under P2)
        add_stack(state, Position(x=4, y=4), [1, 2])
        # P1 controls no stacks
        assert is_eligible_for_recovery(state, 1) is True


class TestCalculateRecoveryCost:
    """Tests for calculate_recovery_cost.

    New cost model (Option 1 / Option 2):
    - Option 1: Always costs 1 buried ring (collapse all markers)
    - Option 2: Always costs 0 (free, collapse lineLength markers, overlength only)
    """

    def test_option1_always_costs_one(self):
        # Option 1 always costs 1 buried ring regardless of markers
        assert calculate_recovery_cost(1) == 1

    def test_option2_always_free(self):
        # Option 2 is free (costs 0) - only available for overlength lines
        assert calculate_recovery_cost(2) == 0


class TestEnumerateRecoverySlideTargets:
    """Tests for enumerate_recovery_slide_targets."""

    def test_returns_empty_when_not_eligible(self):
        state = create_test_state()
        # Not eligible - no markers or buried rings
        targets = enumerate_recovery_slide_targets(state, 1)
        assert len(targets) == 0

    def test_returns_empty_when_no_line_can_be_completed(self):
        state = create_test_state()
        # Make eligible but with only 1 marker (can't form line of 3)
        add_marker(state, Position(x=3, y=3), 1)
        add_stack(state, Position(x=4, y=4), [1, 2])  # P1 ring buried
        targets = enumerate_recovery_slide_targets(state, 1)
        assert len(targets) == 0

    def test_finds_valid_slide_completing_line(self):
        state = create_test_state()
        # P1 has markers that can form a horizontal line of 4 (required for square8 2-player per RR-CANON-R112)
        # Markers at (1,3), (2,3), (3,3) - need one more at (4,3) or (0,3)
        add_marker(state, Position(x=1, y=3), 1)
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        # P1 has a slideable marker that can complete the line
        add_marker(state, Position(x=4, y=2), 1)  # Can slide to (4,3)
        # P1 has buried ring
        add_stack(state, Position(x=0, y=0), [1, 2])

        targets = enumerate_recovery_slide_targets(state, 1)
        # Should find the slide from (4,2) to (4,3) completing the 4-marker line
        assert len(targets) > 0
        slide_to_4_3 = [t for t in targets if t.to_pos.x == 4 and t.to_pos.y == 3]
        assert len(slide_to_4_3) > 0


class TestHasAnyRecoveryMove:
    """Tests for has_any_recovery_move."""

    def test_returns_false_when_not_eligible(self):
        state = create_test_state()
        assert has_any_recovery_move(state, 1) is False

    def test_returns_true_when_eligible_with_valid_moves(self):
        state = create_test_state()
        # Setup for valid recovery - need 4 markers for square8 2-player per RR-CANON-R112
        add_marker(state, Position(x=1, y=3), 1)
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        add_marker(state, Position(x=4, y=2), 1)  # Can slide to (4,3)
        add_stack(state, Position(x=0, y=0), [1, 2])

        assert has_any_recovery_move(state, 1) is True


class TestValidateRecoverySlide:
    """Tests for validate_recovery_slide."""

    def test_invalid_when_not_in_movement_phase(self):
        state = create_test_state(current_phase=GamePhase.RING_PLACEMENT)
        # Setup for otherwise valid recovery
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        add_marker(state, Position(x=4, y=2), 1)
        add_stack(state, Position(x=0, y=0), [1, 2])

        move = Move(
            id="test",
            type=MoveType.RECOVERY_SLIDE,
            player=1,
            from_pos=Position(x=4, y=2),
            to=Position(x=4, y=3),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        result = validate_recovery_slide(state, move)
        assert result.valid is False
        assert "phase" in result.reason.lower()

    def test_invalid_when_not_eligible_due_to_controlled_stack(self):
        """Player who controls a stack is NOT eligible for recovery."""
        state = create_test_state(rings_in_hand_p1=0)
        # Not eligible - P1 controls a stack (top ring is P1)
        add_marker(state, Position(x=3, y=3), 1)
        add_stack(state, Position(x=0, y=0), [2, 1])  # P1 controls this stack

        move = Move(
            id="test",
            type=MoveType.RECOVERY_SLIDE,
            player=1,
            from_pos=Position(x=3, y=3),
            to=Position(x=3, y=4),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        result = validate_recovery_slide(state, move)
        assert result.valid is False
        assert "eligible" in result.reason.lower()

    def test_valid_slide_completing_line(self):
        state = create_test_state()
        # Need 4 markers for square8 2-player per RR-CANON-R112
        add_marker(state, Position(x=1, y=3), 1)
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        add_marker(state, Position(x=4, y=2), 1)  # Will slide to (4,3)
        add_stack(state, Position(x=0, y=0), [1, 2])

        move = Move(
            id="test",
            type=MoveType.RECOVERY_SLIDE,
            player=1,
            from_pos=Position(x=4, y=2),
            to=Position(x=4, y=3),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        result = validate_recovery_slide(state, move)
        assert result.valid is True
        assert result.markers_in_line >= 4  # 4 markers for square8 2-player
        assert result.cost >= 1


class TestApplyRecoverySlide:
    """Tests for apply_recovery_slide."""

    def test_moves_marker(self):
        state = create_test_state()
        # Need 4 markers for square8 2-player per RR-CANON-R112
        add_marker(state, Position(x=1, y=3), 1)
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        add_marker(state, Position(x=4, y=2), 1)  # Will slide to (4,3)
        add_stack(state, Position(x=0, y=0), [1, 2])

        move = Move(
            id="test",
            type=MoveType.RECOVERY_SLIDE,
            player=1,
            from_pos=Position(x=4, y=2),
            to=Position(x=4, y=3),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        outcome = apply_recovery_slide(state, move)
        assert outcome.success is True
        assert "4,2" not in state.board.markers
        # After recovery slide, the line is completed and markers are collapsed
        # into territory (collapsed_spaces), not remaining as markers.
        assert "4,3" in state.board.collapsed_spaces

    def test_extracts_buried_rings(self):
        """Per RR-CANON-R113: Extracted rings are self-eliminated, NOT returned to hand."""
        state = create_test_state()
        # Need 4 markers for square8 2-player per RR-CANON-R112
        add_marker(state, Position(x=1, y=3), 1)
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        add_marker(state, Position(x=4, y=2), 1)  # Will slide to (4,3)
        add_stack(state, Position(x=0, y=0), [1, 2])

        initial_buried = count_buried_rings(state.board, 1)
        initial_eliminated = state.players[0].eliminated_rings

        move = Move(
            id="test",
            type=MoveType.RECOVERY_SLIDE,
            player=1,
            from_pos=Position(x=4, y=2),
            to=Position(x=4, y=3),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        outcome = apply_recovery_slide(state, move)
        assert outcome.success is True
        assert outcome.rings_extracted > 0

        # Buried rings should decrease
        final_buried = count_buried_rings(state.board, 1)
        assert final_buried < initial_buried

        # Per RR-CANON-R113: Extracted rings are ELIMINATED, not returned to hand
        # This is the mandatory cost for recovery actions
        final_eliminated = state.players[0].eliminated_rings
        assert final_eliminated == initial_eliminated + outcome.rings_extracted


class TestGetRecoveryMoves:
    """Tests for get_recovery_moves integration."""

    def test_returns_empty_when_not_eligible(self):
        state = create_test_state()
        moves = get_recovery_moves(state, 1)
        assert len(moves) == 0

    def test_returns_moves_with_correct_type(self):
        state = create_test_state()
        # Need 4 markers for square8 2-player per RR-CANON-R112
        add_marker(state, Position(x=1, y=3), 1)
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        add_marker(state, Position(x=4, y=2), 1)  # Can slide to (4,3)
        add_stack(state, Position(x=0, y=0), [1, 2])

        moves = get_recovery_moves(state, 1)
        assert len(moves) > 0
        for move in moves:
            assert move.type == MoveType.RECOVERY_SLIDE
            assert move.player == 1


class TestExpandedRecovery:
    """Tests for expanded recovery with fallback mode (RR-CANON-R112)."""

    def test_fallback_mode_when_no_line_possible(self):
        """Test that fallback repositioning is available when no line can be formed."""
        from app.rules.recovery import get_expanded_recovery_moves, enumerate_expanded_recovery_targets

        state = create_test_state()
        # Add markers that can't form a line (need 4 for square8 2P)
        add_marker(state, Position(x=3, y=3), 1)
        add_marker(state, Position(x=5, y=5), 1)  # Too far apart for a line
        add_stack(state, Position(x=0, y=0), [1, 2])  # P1 buried, P2 on top

        # Get expanded recovery targets
        targets = enumerate_expanded_recovery_targets(state, 1)

        # Should have fallback targets since no line is possible
        fallback_targets = [t for t in targets if t.recovery_mode == "fallback"]
        assert len(fallback_targets) > 0, "Should have fallback recovery options"

        # Get expanded moves
        moves = get_expanded_recovery_moves(state, 1)

        # Should include skip_recovery
        skip_moves = [m for m in moves if m.type == MoveType.SKIP_RECOVERY]
        assert len(skip_moves) == 1, "Should have exactly one skip_recovery option"

        # Should include fallback recovery slides
        fallback_slides = [m for m in moves if m.type == MoveType.RECOVERY_SLIDE and m.recovery_mode == "fallback"]
        assert len(fallback_slides) > 0, "Should have fallback recovery slide options"

    def test_stack_strike_targets_when_enabled_and_no_line(self, monkeypatch):
        """Stack-strike v1: targets appear when enabled and no line exists."""
        from app.rules.recovery import enumerate_expanded_recovery_targets, get_expanded_recovery_moves

        monkeypatch.setenv("RINGRIFT_RECOVERY_STACK_STRIKE_V1", "1")
        state = create_test_state()

        # Single marker adjacent to an opponent-controlled stack; no line possible.
        add_marker(state, Position(x=3, y=2), 1)
        add_stack(state, Position(x=3, y=3), [2, 2])  # attacked stack controlled by P2
        add_stack(state, Position(x=0, y=0), [1, 2])  # buried P1 ring for extraction

        targets = enumerate_expanded_recovery_targets(state, 1)
        strike_targets = [t for t in targets if t.recovery_mode == "stack_strike"]
        assert len(strike_targets) > 0

        moves = get_expanded_recovery_moves(state, 1)
        strike_moves = [m for m in moves if m.type == MoveType.RECOVERY_SLIDE and m.recovery_mode == "stack_strike"]
        assert len(strike_moves) > 0

    def test_apply_stack_strike_eliminates_top_ring_and_sacrifices_marker(self, monkeypatch):
        """Stack-strike v1: applying removes marker and top ring."""
        from app.rules.recovery import apply_recovery_slide

        monkeypatch.setenv("RINGRIFT_RECOVERY_STACK_STRIKE_V1", "1")
        state = create_test_state()
        add_marker(state, Position(x=3, y=2), 1)
        add_stack(state, Position(x=3, y=3), [2, 2])  # attacked stack
        add_stack(state, Position(x=0, y=0), [1, 2])  # extraction stack (buried P1)

        move = Move(
            id="recovery-stack-strike-test",
            type=MoveType.RECOVERY_SLIDE,
            player=1,
            from_pos=Position(x=3, y=2),
            to=Position(x=3, y=3),
            recoveryMode="stack_strike",
            extraction_stacks=("0,0",),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=len(state.move_history) + 1,
        )

        outcome = apply_recovery_slide(state, move)
        assert outcome.success is True
        assert "3,2" not in state.board.markers
        assert "3,3" not in state.board.markers
        assert state.board.stacks["3,3"].stack_height == 1

    def test_line_mode_prioritized_over_fallback(self):
        """Test that line recovery is returned instead of fallback when a line can be formed."""
        from app.rules.recovery import enumerate_expanded_recovery_targets

        state = create_test_state()
        # Add markers that CAN form a line (4 for square8 2P)
        add_marker(state, Position(x=1, y=3), 1)
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        add_marker(state, Position(x=4, y=2), 1)  # Can slide to (4,3) to complete line
        add_stack(state, Position(x=0, y=0), [1, 2])

        targets = enumerate_expanded_recovery_targets(state, 1)

        # Should have line targets
        line_targets = [t for t in targets if t.recovery_mode == "line"]
        assert len(line_targets) > 0, "Should have line recovery options"

        # Should NOT have fallback targets when line is possible
        fallback_targets = [t for t in targets if t.recovery_mode == "fallback"]
        assert len(fallback_targets) == 0, "Should not have fallback when line recovery is available"

    def test_skip_recovery_always_available(self):
        """Test that skip_recovery is always available when eligible for recovery."""
        from app.rules.recovery import get_expanded_recovery_moves

        state = create_test_state()
        add_marker(state, Position(x=3, y=3), 1)
        add_stack(state, Position(x=0, y=0), [1, 2])

        moves = get_expanded_recovery_moves(state, 1)

        skip_moves = [m for m in moves if m.type == MoveType.SKIP_RECOVERY]
        assert len(skip_moves) == 1, "skip_recovery should always be available"

        skip_move = skip_moves[0]
        assert skip_move.player == 1
