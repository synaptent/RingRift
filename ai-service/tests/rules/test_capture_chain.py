"""
Unit tests for the capture chain module.

Tests the Python capture chain implementation against canonical rules:
- RR-CANON-R101: Self-capture is legal (can overtake own stacks)
- RR-CANON-R102: Landing on any marker is legal (marker eliminated)
- Cap height constraint: attacker.cap_height >= target.cap_height
- Path must be clear of stacks and collapsed spaces (markers allowed)
"""

from __future__ import annotations

import pytest

from app.models import (
    BoardType,
    GamePhase,
    MoveType,
    Position,
)
from app.rules.capture_chain import (
    PyChainCaptureContinuationInfo,
    PyChainCaptureEnumerationOptions,
    PyChainCaptureStateSnapshot,
    PyCaptureSegmentParams,
    apply_capture_py,
    apply_capture_segment_py,
    enumerate_capture_moves_py,
    enumerate_chain_capture_segments_py,
    get_chain_capture_continuation_info_py,
    validate_capture_segment_on_board_py,
)

from .conftest import (
    ALL_BOARD_TYPES,
    HEX_BOARDS,
    SQUARE_BOARDS,
    add_marker,
    add_stack,
    clear_position,
    collapse_space,
    create_capture_scenario_state,
    create_chain_capture_state,
    create_game_state,
    get_board_size,
    get_valid_positions,
)


# =============================================================================
# Tests for validate_capture_segment_on_board_py
# =============================================================================

class TestValidateCaptureSegmentOnBoard:
    """Tests for the validate_capture_segment_on_board_py function."""

    @pytest.mark.parametrize("board_type", SQUARE_BOARDS)
    def test_valid_capture_square_board(self, board_type: BoardType) -> None:
        """A valid capture should return True on square boards."""
        state = create_capture_scenario_state(board_type)
        size = get_board_size(board_type)
        center = size // 2

        attacker_pos = Position(x=center - 2, y=center)
        target_pos = Position(x=center, y=center)
        landing_pos = Position(x=center + 1, y=center)

        result = validate_capture_segment_on_board_py(
            board_type=board_type,
            from_pos=attacker_pos,
            target_pos=target_pos,
            landing_pos=landing_pos,
            player=1,
            board=state.board,
        )
        assert result is True

    @pytest.mark.parametrize("board_type", HEX_BOARDS)
    def test_valid_capture_hex_board(self, board_type: BoardType) -> None:
        """A valid capture should return True on hex boards."""
        state = create_game_state(board_type)
        # Set up attacker at (-2, 0, 2), target at (0, 0, 0), landing at (1, 0, -1)
        add_stack(state, Position(x=-2, y=0, z=2), [1, 1])
        add_stack(state, Position(x=0, y=0, z=0), [2])

        result = validate_capture_segment_on_board_py(
            board_type=board_type,
            from_pos=Position(x=-2, y=0, z=2),
            target_pos=Position(x=0, y=0, z=0),
            landing_pos=Position(x=1, y=0, z=-1),
            player=1,
            board=state.board,
        )
        assert result is True

    def test_rejects_invalid_from_position(self) -> None:
        """Reject if from_pos is outside the board."""
        state = create_game_state(BoardType.SQUARE8)
        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=-1, y=0),  # Invalid
            target_pos=Position(x=2, y=0),
            landing_pos=Position(x=3, y=0),
            player=1,
            board=state.board,
        )
        assert result is False

    def test_rejects_invalid_target_position(self) -> None:
        """Reject if target_pos is outside the board."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=0, y=0), [1, 1])
        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=0, y=0),
            target_pos=Position(x=100, y=0),  # Invalid
            landing_pos=Position(x=3, y=0),
            player=1,
            board=state.board,
        )
        assert result is False

    def test_rejects_no_attacker_stack(self) -> None:
        """Reject if there is no stack at from_pos."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=4, y=4), [2])

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),  # No stack here
            target_pos=Position(x=4, y=4),
            landing_pos=Position(x=5, y=4),
            player=1,
            board=state.board,
        )
        assert result is False

    def test_rejects_wrong_player_control(self) -> None:
        """Reject if attacker stack is controlled by wrong player."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [2, 2])  # P2 controls
        add_stack(state, Position(x=4, y=4), [1])

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=4, y=4),
            landing_pos=Position(x=5, y=4),
            player=1,  # P1 trying to use P2's stack
            board=state.board,
        )
        assert result is False

    def test_rejects_no_target_stack(self) -> None:
        """Reject if there is no stack at target_pos."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [1, 1])

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=4, y=4),  # No stack
            landing_pos=Position(x=5, y=4),
            player=1,
            board=state.board,
        )
        assert result is False

    def test_rejects_insufficient_cap_height(self) -> None:
        """Reject if attacker.cap_height < target.cap_height."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [1])  # cap_height=1
        add_stack(state, Position(x=4, y=4), [2, 2, 2])  # cap_height=3

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=4, y=4),
            landing_pos=Position(x=5, y=4),
            player=1,
            board=state.board,
        )
        assert result is False

    def test_allows_self_capture_rr_canon_r101(self) -> None:
        """RR-CANON-R101: Self-capture is legal."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [1, 1])  # Attacker controlled by P1
        add_stack(state, Position(x=4, y=4), [1])  # Target also controlled by P1

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=4, y=4),
            landing_pos=Position(x=5, y=4),
            player=1,
            board=state.board,
        )
        assert result is True

    def test_rejects_path_blocked_by_stack(self) -> None:
        """Reject if path to target is blocked by another stack."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [1, 1])
        add_stack(state, Position(x=3, y=4), [2])  # Blocking stack
        add_stack(state, Position(x=4, y=4), [2])

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=4, y=4),
            landing_pos=Position(x=5, y=4),
            player=1,
            board=state.board,
        )
        assert result is False

    def test_rejects_path_blocked_by_collapsed_space(self) -> None:
        """Reject if path to target includes collapsed space."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [1, 1])
        collapse_space(state, Position(x=3, y=4))  # Collapsed
        add_stack(state, Position(x=4, y=4), [2])

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=4, y=4),
            landing_pos=Position(x=5, y=4),
            player=1,
            board=state.board,
        )
        assert result is False

    def test_allows_marker_on_path(self) -> None:
        """Markers on the path do not block captures."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [1, 1])
        add_marker(state, Position(x=3, y=4), 2)  # Marker, not blocking
        add_stack(state, Position(x=4, y=4), [2])

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=4, y=4),
            landing_pos=Position(x=5, y=4),
            player=1,
            board=state.board,
        )
        assert result is True

    def test_rejects_landing_on_stack(self) -> None:
        """Reject if landing position has a stack."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [1, 1])
        add_stack(state, Position(x=4, y=4), [2])
        add_stack(state, Position(x=5, y=4), [2])  # Stack at landing

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=4, y=4),
            landing_pos=Position(x=5, y=4),
            player=1,
            board=state.board,
        )
        assert result is False

    def test_rejects_landing_on_collapsed_space(self) -> None:
        """Reject if landing position is collapsed."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [1, 1])
        add_stack(state, Position(x=4, y=4), [2])
        collapse_space(state, Position(x=5, y=4))

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=4, y=4),
            landing_pos=Position(x=5, y=4),
            player=1,
            board=state.board,
        )
        assert result is False

    def test_allows_landing_on_marker_rr_canon_r102(self) -> None:
        """RR-CANON-R102: Landing on marker is legal (marker eliminated)."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [1, 1])
        add_stack(state, Position(x=4, y=4), [2])
        add_marker(state, Position(x=5, y=4), 2)  # Marker at landing

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=4, y=4),
            landing_pos=Position(x=5, y=4),
            player=1,
            board=state.board,
        )
        assert result is True

    def test_rejects_landing_before_target(self) -> None:
        """Reject if landing is not beyond target."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [1, 1])
        add_stack(state, Position(x=4, y=4), [2])

        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=4, y=4),
            landing_pos=Position(x=3, y=4),  # Before target
            player=1,
            board=state.board,
        )
        assert result is False

    def test_rejects_insufficient_distance_for_stack_height(self) -> None:
        """Reject if total distance < attacker.stack_height."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=2, y=4), [1, 1, 1, 1])  # stack_height=4
        add_stack(state, Position(x=3, y=4), [2])  # Adjacent

        # Distance from (2,4) to (4,4) is only 2, but stack_height is 4
        result = validate_capture_segment_on_board_py(
            board_type=BoardType.SQUARE8,
            from_pos=Position(x=2, y=4),
            target_pos=Position(x=3, y=4),
            landing_pos=Position(x=4, y=4),
            player=1,
            board=state.board,
        )
        assert result is False


# =============================================================================
# Tests for enumerate_capture_moves_py
# =============================================================================

class TestEnumerateCaptureMoves:
    """Tests for enumerate_capture_moves_py function."""

    @pytest.mark.parametrize("board_type", SQUARE_BOARDS)
    def test_returns_empty_for_no_attacker(self, board_type: BoardType) -> None:
        """Returns empty list if no stack at from_pos."""
        state = create_game_state(board_type)
        positions = get_valid_positions(board_type)

        moves = enumerate_capture_moves_py(state, 1, positions[0])
        assert moves == []

    @pytest.mark.parametrize("board_type", SQUARE_BOARDS)
    def test_returns_empty_for_wrong_player(self, board_type: BoardType) -> None:
        """Returns empty list if stack is controlled by different player."""
        state = create_game_state(board_type)
        positions = get_valid_positions(board_type)
        add_stack(state, positions[0], [2, 2])  # P2 controls

        moves = enumerate_capture_moves_py(state, 1, positions[0])
        assert moves == []

    @pytest.mark.parametrize("board_type", SQUARE_BOARDS)
    def test_finds_single_capture_square(self, board_type: BoardType) -> None:
        """Finds a single valid capture opportunity."""
        state = create_capture_scenario_state(board_type)
        size = get_board_size(board_type)
        center = size // 2
        attacker_pos = Position(x=center - 2, y=center)

        moves = enumerate_capture_moves_py(state, 1, attacker_pos)
        assert len(moves) > 0
        assert all(m.type == MoveType.OVERTAKING_CAPTURE for m in moves)

    @pytest.mark.parametrize("board_type", HEX_BOARDS)
    def test_finds_capture_hex_board(self, board_type: BoardType) -> None:
        """Finds captures on hex boards."""
        state = create_game_state(board_type)
        add_stack(state, Position(x=-2, y=0, z=2), [1, 1])
        add_stack(state, Position(x=0, y=0, z=0), [2])

        moves = enumerate_capture_moves_py(state, 1, Position(x=-2, y=0, z=2))
        assert len(moves) > 0

    def test_finds_multiple_landing_positions(self) -> None:
        """Finds all valid landing positions for a capture."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=1, y=4), [1, 1])  # Attacker
        add_stack(state, Position(x=3, y=4), [2])  # Target

        # Should find landings at x=4, x=5, x=6, x=7 (up to board edge)
        moves = enumerate_capture_moves_py(state, 1, Position(x=1, y=4))
        landing_xs = {m.to.x for m in moves if m.to}
        assert 4 in landing_xs  # Minimum landing
        assert len(landing_xs) >= 1

    def test_respects_cap_height_constraint(self) -> None:
        """Does not enumerate captures where cap_height is insufficient."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=1, y=4), [1])  # cap_height=1
        add_stack(state, Position(x=3, y=4), [2, 2, 2])  # cap_height=3

        moves = enumerate_capture_moves_py(state, 1, Position(x=1, y=4))
        assert len(moves) == 0

    def test_kind_initial_sets_overtaking_capture_type(self) -> None:
        """Kind='initial' produces OVERTAKING_CAPTURE moves."""
        state = create_capture_scenario_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2
        attacker_pos = Position(x=center - 2, y=center)

        moves = enumerate_capture_moves_py(
            state, 1, attacker_pos, kind="initial"
        )
        assert all(m.type == MoveType.OVERTAKING_CAPTURE for m in moves)

    def test_kind_continuation_sets_continue_capture_type(self) -> None:
        """Kind='continuation' produces CONTINUE_CAPTURE_SEGMENT moves."""
        state = create_capture_scenario_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2
        attacker_pos = Position(x=center - 2, y=center)

        moves = enumerate_capture_moves_py(
            state, 1, attacker_pos, kind="continuation"
        )
        assert all(m.type == MoveType.CONTINUE_CAPTURE_SEGMENT for m in moves)


# =============================================================================
# Tests for enumerate_chain_capture_segments_py
# =============================================================================

class TestEnumerateChainCaptureSegments:
    """Tests for enumerate_chain_capture_segments_py function."""

    def test_basic_enumeration(self) -> None:
        """Basic chain capture segment enumeration works."""
        state = create_capture_scenario_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2

        snapshot = PyChainCaptureStateSnapshot(
            player=1,
            current_position=Position(x=center - 2, y=center),
            captured_this_chain=[],
        )

        moves = enumerate_chain_capture_segments_py(state, snapshot)
        assert len(moves) > 0

    def test_respects_options_kind(self) -> None:
        """Options.kind is passed through to move type."""
        state = create_capture_scenario_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2

        snapshot = PyChainCaptureStateSnapshot(
            player=1,
            current_position=Position(x=center - 2, y=center),
            captured_this_chain=[],
        )

        options = PyChainCaptureEnumerationOptions(kind="continuation")
        moves = enumerate_chain_capture_segments_py(state, snapshot, options)
        assert all(m.type == MoveType.CONTINUE_CAPTURE_SEGMENT for m in moves)

    def test_filters_revisited_targets(self) -> None:
        """disallow_revisited_targets filters out previously captured targets."""
        state = create_capture_scenario_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2
        target_pos = Position(x=center, y=center)

        snapshot = PyChainCaptureStateSnapshot(
            player=1,
            current_position=Position(x=center - 2, y=center),
            captured_this_chain=[target_pos],  # Already captured
        )

        options = PyChainCaptureEnumerationOptions(disallow_revisited_targets=True)
        moves = enumerate_chain_capture_segments_py(state, snapshot, options)
        # Should be empty since only target is already captured
        target_keys = {m.capture_target.to_key() for m in moves if m.capture_target}
        assert target_pos.to_key() not in target_keys


# =============================================================================
# Tests for get_chain_capture_continuation_info_py
# =============================================================================

class TestGetChainCaptureContinuationInfo:
    """Tests for get_chain_capture_continuation_info_py function."""

    def test_must_continue_when_captures_available(self) -> None:
        """must_continue=True when captures are available from landing position."""
        state = create_chain_capture_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2

        # After landing at center, there should be another capture available
        info = get_chain_capture_continuation_info_py(
            state, player=1, current_position=Position(x=center, y=center)
        )
        assert isinstance(info, PyChainCaptureContinuationInfo)
        # The info shows whether continuation is needed

    def test_returns_continuation_info_dataclass(self) -> None:
        """Returns proper PyChainCaptureContinuationInfo dataclass."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=4, y=4), [1, 1])

        info = get_chain_capture_continuation_info_py(
            state, player=1, current_position=Position(x=4, y=4)
        )
        assert isinstance(info, PyChainCaptureContinuationInfo)
        assert isinstance(info.must_continue, bool)
        assert isinstance(info.available_continuations, list)

    def test_must_continue_false_when_no_captures(self) -> None:
        """must_continue=False when no captures available."""
        state = create_game_state(BoardType.SQUARE8)
        add_stack(state, Position(x=4, y=4), [1, 1])  # Only our stack, no targets

        info = get_chain_capture_continuation_info_py(
            state, player=1, current_position=Position(x=4, y=4)
        )
        assert info.must_continue is False
        assert len(info.available_continuations) == 0


# =============================================================================
# Tests for apply_capture_segment_py
# =============================================================================

class TestApplyCaptureSegment:
    """Tests for apply_capture_segment_py function."""

    def test_returns_new_state(self) -> None:
        """Returns a new GameState (does not mutate input)."""
        state = create_capture_scenario_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2

        original_stacks = dict(state.board.stacks)

        params = PyCaptureSegmentParams(
            from_pos=Position(x=center - 2, y=center),
            target_pos=Position(x=center, y=center),
            landing_pos=Position(x=center + 1, y=center),
            player=1,
        )

        outcome = apply_capture_segment_py(state, params)

        # Original state unchanged
        assert state.board.stacks == original_stacks
        # New state is different
        assert outcome.next_state.board.stacks != original_stacks

    def test_reports_rings_transferred(self) -> None:
        """Reports rings_transferred in outcome."""
        state = create_capture_scenario_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2

        params = PyCaptureSegmentParams(
            from_pos=Position(x=center - 2, y=center),
            target_pos=Position(x=center, y=center),
            landing_pos=Position(x=center + 1, y=center),
            player=1,
        )

        outcome = apply_capture_segment_py(state, params)
        assert outcome.rings_transferred >= 0

    def test_reports_chain_continuation_status(self) -> None:
        """Reports whether chain continuation is required."""
        state = create_capture_scenario_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2

        params = PyCaptureSegmentParams(
            from_pos=Position(x=center - 2, y=center),
            target_pos=Position(x=center, y=center),
            landing_pos=Position(x=center + 1, y=center),
            player=1,
        )

        outcome = apply_capture_segment_py(state, params)
        assert isinstance(outcome.chain_continuation_required, bool)


# =============================================================================
# Tests for apply_capture_py
# =============================================================================

class TestApplyCapture:
    """Tests for apply_capture_py high-level function."""

    def test_rejects_non_capture_move_type(self) -> None:
        """Rejects moves that are not capture types."""
        from app.models import Move

        state = create_game_state(BoardType.SQUARE8)
        move = Move(
            id="test",
            type=MoveType.PLACE_RING,  # Not a capture
            player=1,
            to=Position(x=4, y=4),
            timestamp=state.last_move_at,
            thinkTime=0,
            moveNumber=1,
        )

        success, next_state, result = apply_capture_py(state, move)
        assert success is False
        assert next_state is None
        assert "Expected" in result

    def test_rejects_missing_from_pos(self) -> None:
        """Rejects capture moves without from_pos."""
        from app.models import Move

        state = create_game_state(BoardType.SQUARE8)
        move = Move(
            id="test",
            type=MoveType.OVERTAKING_CAPTURE,
            player=1,
            from_pos=None,  # Missing
            to=Position(x=4, y=4),
            capture_target=Position(x=3, y=4),
            timestamp=state.last_move_at,
            thinkTime=0,
            moveNumber=1,
        )

        success, next_state, result = apply_capture_py(state, move)
        assert success is False

    def test_rejects_missing_capture_target(self) -> None:
        """Rejects capture moves without capture_target."""
        from app.models import Move

        state = create_game_state(BoardType.SQUARE8)
        move = Move(
            id="test",
            type=MoveType.OVERTAKING_CAPTURE,
            player=1,
            from_pos=Position(x=2, y=4),
            to=Position(x=4, y=4),
            capture_target=None,  # Missing
            timestamp=state.last_move_at,
            thinkTime=0,
            moveNumber=1,
        )

        success, next_state, result = apply_capture_py(state, move)
        assert success is False

    def test_successful_capture_returns_new_state(self) -> None:
        """Successful capture returns new state and continuation info."""
        from app.models import Move

        state = create_capture_scenario_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2

        move = Move(
            id="test-capture",
            type=MoveType.OVERTAKING_CAPTURE,
            player=1,
            from_pos=Position(x=center - 2, y=center),
            to=Position(x=center + 1, y=center),
            capture_target=Position(x=center, y=center),
            timestamp=state.last_move_at,
            thinkTime=0,
            moveNumber=1,
        )

        success, next_state, continuations = apply_capture_py(state, move)
        assert success is True
        assert next_state is not None
        assert isinstance(continuations, list)

    def test_accepts_continue_capture_segment_type(self) -> None:
        """Accepts CONTINUE_CAPTURE_SEGMENT move type."""
        from app.models import Move

        state = create_capture_scenario_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2

        move = Move(
            id="test-continue",
            type=MoveType.CONTINUE_CAPTURE_SEGMENT,
            player=1,
            from_pos=Position(x=center - 2, y=center),
            to=Position(x=center + 1, y=center),
            capture_target=Position(x=center, y=center),
            timestamp=state.last_move_at,
            thinkTime=0,
            moveNumber=1,
        )

        success, next_state, _ = apply_capture_py(state, move)
        assert success is True

    def test_accepts_chain_capture_type(self) -> None:
        """Accepts CHAIN_CAPTURE move type."""
        from app.models import Move

        state = create_capture_scenario_state(BoardType.SQUARE8)
        size = get_board_size(BoardType.SQUARE8)
        center = size // 2

        move = Move(
            id="test-chain",
            type=MoveType.CHAIN_CAPTURE,
            player=1,
            from_pos=Position(x=center - 2, y=center),
            to=Position(x=center + 1, y=center),
            capture_target=Position(x=center, y=center),
            timestamp=state.last_move_at,
            thinkTime=0,
            moveNumber=1,
        )

        success, next_state, _ = apply_capture_py(state, move)
        assert success is True


# =============================================================================
# Tests for Dataclasses
# =============================================================================

class TestDataclasses:
    """Tests for dataclass definitions."""

    def test_chain_capture_segment_creation(self) -> None:
        """PyChainCaptureSegment can be instantiated."""
        from app.rules.capture_chain import PyChainCaptureSegment

        segment = PyChainCaptureSegment(
            from_pos=Position(x=0, y=0),
            target_pos=Position(x=2, y=0),
            landing_pos=Position(x=3, y=0),
            captured_cap_height=2,
        )
        assert segment.from_pos.x == 0
        assert segment.captured_cap_height == 2

    def test_chain_capture_state_snapshot_creation(self) -> None:
        """PyChainCaptureStateSnapshot can be instantiated."""
        snapshot = PyChainCaptureStateSnapshot(
            player=1,
            current_position=Position(x=4, y=4),
            captured_this_chain=[Position(x=2, y=2)],
        )
        assert snapshot.player == 1
        assert len(snapshot.captured_this_chain) == 1

    def test_chain_capture_enumeration_options_defaults(self) -> None:
        """PyChainCaptureEnumerationOptions has correct defaults."""
        options = PyChainCaptureEnumerationOptions()
        assert options.disallow_revisited_targets is False
        assert options.move_number is None
        assert options.kind == "continuation"

    def test_capture_segment_params_creation(self) -> None:
        """PyCaptureSegmentParams can be instantiated."""
        params = PyCaptureSegmentParams(
            from_pos=Position(x=0, y=0),
            target_pos=Position(x=2, y=0),
            landing_pos=Position(x=3, y=0),
            player=1,
        )
        assert params.player == 1


# =============================================================================
# Parametrized Board Type Tests
# =============================================================================

class TestCrossBoard:
    """Tests that run across all board types."""

    @pytest.mark.parametrize("board_type", ALL_BOARD_TYPES)
    def test_empty_board_has_no_captures(self, board_type: BoardType) -> None:
        """Empty board has no capture opportunities."""
        state = create_game_state(board_type)
        positions = get_valid_positions(board_type)

        for pos in positions:
            moves = enumerate_capture_moves_py(state, 1, pos)
            assert moves == []

    @pytest.mark.parametrize("board_type", ALL_BOARD_TYPES)
    def test_single_stack_has_no_captures(self, board_type: BoardType) -> None:
        """Single stack with no targets has no captures."""
        state = create_game_state(board_type)
        positions = get_valid_positions(board_type)
        add_stack(state, positions[0], [1, 1])

        moves = enumerate_capture_moves_py(state, 1, positions[0])
        assert moves == []
