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

    def test_not_eligible_with_rings_in_hand(self):
        state = create_test_state(rings_in_hand_p1=1)
        add_marker(state, Position(x=3, y=3), 1)
        add_stack(state, Position(x=4, y=4), [1, 2])  # P1 ring buried under P2
        assert is_eligible_for_recovery(state, 1) is False

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
    """Tests for calculate_recovery_cost."""

    def test_base_cost_is_one(self):
        state = create_test_state()
        # line_length for square8 is 3
        assert calculate_recovery_cost(state.board, 1, 3) == 1

    def test_excess_markers_increase_cost(self):
        state = create_test_state()
        # 4 markers = 1 excess = cost 2
        assert calculate_recovery_cost(state.board, 1, 4) == 2
        # 5 markers = 2 excess = cost 3
        assert calculate_recovery_cost(state.board, 1, 5) == 3


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
        # P1 has markers that can form a horizontal line
        # Markers at (2,3), (3,3) - need one more at (4,3) or (1,3)
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        # P1 has a slideable marker that can complete the line
        add_marker(state, Position(x=4, y=2), 1)  # Can slide to (4,3)
        # P1 has buried ring
        add_stack(state, Position(x=0, y=0), [1, 2])

        targets = enumerate_recovery_slide_targets(state, 1)
        # Should find the slide from (4,2) to (4,3) completing the line
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
        # Setup for valid recovery
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        add_marker(state, Position(x=4, y=2), 1)
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

    def test_invalid_when_not_eligible(self):
        state = create_test_state(rings_in_hand_p1=5)
        # Not eligible - has rings in hand
        add_marker(state, Position(x=3, y=3), 1)
        add_stack(state, Position(x=0, y=0), [1, 2])

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
        assert result.valid is True
        assert result.markers_in_line >= 3
        assert result.cost >= 1


class TestApplyRecoverySlide:
    """Tests for apply_recovery_slide."""

    def test_moves_marker(self):
        state = create_test_state()
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

        outcome = apply_recovery_slide(state, move)
        assert outcome.success is True
        assert "4,2" not in state.board.markers
        assert "4,3" in state.board.markers

    def test_extracts_buried_rings(self):
        state = create_test_state()
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        add_marker(state, Position(x=4, y=2), 1)
        add_stack(state, Position(x=0, y=0), [1, 2])

        initial_buried = count_buried_rings(state.board, 1)
        initial_hand = state.players[0].rings_in_hand

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

        # Rings in hand should increase
        final_hand = state.players[0].rings_in_hand
        assert final_hand == initial_hand + outcome.rings_extracted


class TestGetRecoveryMoves:
    """Tests for get_recovery_moves integration."""

    def test_returns_empty_when_not_eligible(self):
        state = create_test_state()
        moves = get_recovery_moves(state, 1)
        assert len(moves) == 0

    def test_returns_moves_with_correct_type(self):
        state = create_test_state()
        add_marker(state, Position(x=2, y=3), 1)
        add_marker(state, Position(x=3, y=3), 1)
        add_marker(state, Position(x=4, y=2), 1)
        add_stack(state, Position(x=0, y=0), [1, 2])

        moves = get_recovery_moves(state, 1)
        assert len(moves) > 0
        for move in moves:
            assert move.type == MoveType.RECOVERY_SLIDE
            assert move.player == 1
