"""Tests for app.rules.generators.line module.

Tests the LineGenerator which enumerates line processing moves.

Canonical Spec References:
- RR-CANON-R076: Interactive decision moves only
- RR-CANON-R120: Line formation conditions
- RR-CANON-R123: Line elimination reward (pending_line_reward_elimination)

Note: Lines are formed by MARKERS, not stacks. Stacks act as blockers for lines.
Line length requirements per RR-CANON-R120:
- square8 2-player: 4 markers in a row
- square8 3-4 player: 3 markers in a row
"""

import pytest

from app.models import MarkerInfo, MoveType, Position
from app.rules.generators.line import LineGenerator
from app.testing.fixtures import (
    create_board_state,
    create_game_state,
)


def create_marker(x: int, y: int, player: int) -> MarkerInfo:
    """Create a MarkerInfo for testing."""
    return MarkerInfo(
        position=Position(x=x, y=y),
        player=player,
        type="regular",
    )


class TestLineGeneratorBasics:
    """Basic tests for LineGenerator initialization and interface."""

    def test_generator_instantiation(self):
        """Test LineGenerator can be instantiated."""
        generator = LineGenerator()
        assert generator is not None

    def test_generate_returns_list(self):
        """Test generate() returns a list."""
        generator = LineGenerator()
        state = create_game_state()
        moves = generator.generate(state, player=1)
        assert isinstance(moves, list)

    def test_generate_empty_board_no_lines(self):
        """Test no line moves on empty board."""
        generator = LineGenerator()
        state = create_game_state()
        moves = generator.generate(state, player=1)
        assert len(moves) == 0


class TestLineDetection:
    """Tests for line detection during move generation.

    Note: square8 2-player requires 4 markers for a line (RR-CANON-R120).
    square8 3+ player requires 3 markers.
    """

    @pytest.fixture
    def generator(self):
        return LineGenerator()

    def test_no_lines_when_insufficient_markers(self, generator):
        """Test no lines detected when fewer than required markers in a row."""
        # 2-player requires 4 markers, so 3 is insufficient
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,0": create_marker(1, 0, 1),
                "2,0": create_marker(2, 0, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=2,
        )
        moves = generator.generate(state, player=1)
        assert len(moves) == 0

    def test_line_detected_horizontal_four(self, generator):
        """Test horizontal line of 4 markers generates move (2-player)."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,0": create_marker(1, 0, 1),
                "2,0": create_marker(2, 0, 1),
                "3,0": create_marker(3, 0, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=2,
        )
        moves = generator.generate(state, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1

    def test_line_detected_three_with_three_players(self, generator):
        """Test 3-marker line detected in 3-player game."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,0": create_marker(1, 0, 1),
                "2,0": create_marker(2, 0, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=3,  # 3-player requires only 3 markers
        )
        moves = generator.generate(state, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1

    def test_line_not_detected_different_players(self, generator):
        """Test line not detected when markers owned by different players."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),  # Player 1
                "1,0": create_marker(1, 0, 2),  # Player 2 - breaks the line
                "2,0": create_marker(2, 0, 1),  # Player 1
                "3,0": create_marker(3, 0, 1),  # Player 1
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=2,
        )
        moves = generator.generate(state, player=1)
        assert len(moves) == 0

    def test_line_detected_vertical(self, generator):
        """Test vertical line detection (3-player)."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "0,1": create_marker(0, 1, 1),
                "0,2": create_marker(0, 2, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=3,  # 3-player requires only 3 markers
        )
        moves = generator.generate(state, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1

    def test_line_detected_diagonal(self, generator):
        """Test diagonal line detection (3-player)."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,1": create_marker(1, 1, 1),
                "2,2": create_marker(2, 2, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=3,  # 3-player requires only 3 markers
        )
        moves = generator.generate(state, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1


class TestMoveProperties:
    """Tests for generated move properties and structure."""

    @pytest.fixture
    def generator(self):
        return LineGenerator()

    @pytest.fixture
    def state_with_line(self):
        """Create state with a valid line for player 1 (3-player game)."""
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,0": create_marker(1, 0, 1),
                "2,0": create_marker(2, 0, 1),
            }
        )
        return create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=3,
        )

    def test_move_has_correct_player(self, generator, state_with_line):
        """Test generated moves have correct player attribute."""
        moves = generator.generate(state_with_line, player=1)
        assert len(moves) > 0  # Verify we got moves
        for move in moves:
            assert move.player == 1

    def test_move_has_id(self, generator, state_with_line):
        """Test generated moves have non-empty id."""
        moves = generator.generate(state_with_line, player=1)
        assert len(moves) > 0  # Verify we got moves
        for move in moves:
            assert move.id is not None
            assert len(move.id) > 0

    def test_process_line_move_type(self, generator, state_with_line):
        """Test PROCESS_LINE moves have correct type."""
        moves = generator.generate(state_with_line, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1
        for move in process_moves:
            assert move.type == MoveType.PROCESS_LINE


class TestEdgeCases:
    """Edge case tests for line detection boundary conditions."""

    @pytest.fixture
    def generator(self):
        return LineGenerator()

    def test_multiple_lines_same_marker(self, generator):
        """Test handling when one marker is part of multiple lines.

        Creates an L-shaped pattern where the corner marker (1,1) is part of
        both a horizontal and vertical line. The generator should detect both
        lines and generate moves for each.
        """
        # L-shape pattern in 3-player game (requires 3 markers per line):
        # Horizontal: (0,1), (1,1), (2,1)
        # Vertical:   (1,0), (1,1), (1,2)
        # The corner marker (1,1) is shared by both lines
        board = create_board_state(
            markers={
                # Horizontal line
                "0,1": create_marker(0, 1, 1),
                "1,1": create_marker(1, 1, 1),  # Corner - shared marker
                "2,1": create_marker(2, 1, 1),
                # Vertical line
                "1,0": create_marker(1, 0, 1),
                # (1,1) already added above
                "1,2": create_marker(1, 2, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=3,  # 3-player requires 3 markers per line
        )
        moves = generator.generate(state, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        # Should detect 2 lines (horizontal and vertical)
        assert len(process_moves) >= 2

    def test_line_at_board_edge(self, generator):
        """Test line detection at board edges.

        Tests that lines formed at the edge of the board (y=0 row) are
        correctly detected. Board edge should not prevent line detection.
        """
        # Line at bottom edge of board (y=0)
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,0": create_marker(1, 0, 1),
                "2,0": create_marker(2, 0, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=3,  # 3-player requires 3 markers
        )
        moves = generator.generate(state, player=1)
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1, "Line at board edge should be detected"

    def test_longer_than_minimum_line(self, generator):
        """Test lines longer than minimum length (e.g., 5 in a row).

        When a line is longer than the minimum required length, the generator
        should offer both collapse-all and minimum-collapse options via
        CHOOSE_LINE_OPTION moves.
        """
        # 5 markers in a row in a 3-player game (minimum is 3)
        board = create_board_state(
            markers={
                "0,0": create_marker(0, 0, 1),
                "1,0": create_marker(1, 0, 1),
                "2,0": create_marker(2, 0, 1),
                "3,0": create_marker(3, 0, 1),
                "4,0": create_marker(4, 0, 1),
            }
        )
        state = create_game_state(
            board=board,
            current_phase="line_processing",
            num_players=3,  # 3-player requires 3 markers
        )
        moves = generator.generate(state, player=1)

        # Should have PROCESS_LINE move
        process_moves = [m for m in moves if m.type == MoveType.PROCESS_LINE]
        assert len(process_moves) >= 1

        # Should have multiple CHOOSE_LINE_OPTION moves for overlength line:
        # - 1 collapse-all option (all 5 markers)
        # - 3 minimum-collapse options (positions 0-2, 1-3, 2-4)
        choose_moves = [m for m in moves if m.type == MoveType.CHOOSE_LINE_OPTION]
        assert len(choose_moves) >= 4, (
            f"Expected at least 4 CHOOSE_LINE_OPTION moves for 5-marker line "
            f"(1 collapse-all + 3 min-collapse), got {len(choose_moves)}"
        )

        # Verify collapse-all option exists (should have all 5 positions)
        collapse_all = [m for m in choose_moves if len(m.collapsed_markers) == 5]
        assert len(collapse_all) >= 1, "Should have collapse-all option"

        # Verify minimum-collapse options exist (should have 3 positions each)
        min_collapse = [m for m in choose_moves if len(m.collapsed_markers) == 3]
        assert len(min_collapse) >= 3, "Should have multiple min-collapse options"
