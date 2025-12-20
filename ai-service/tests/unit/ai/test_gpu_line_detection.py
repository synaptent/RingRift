"""Tests for GPU line detection module.

Tests the vectorized line detection and processing functions.
"""

import pytest
import torch

from app.ai.gpu_batch_state import BatchGameState
from app.ai.gpu_game_types import get_required_line_length
from app.ai.gpu_line_detection import (
    detect_lines_batch,
    detect_lines_vectorized,
    detect_lines_with_metadata,
    has_lines_batch_vectorized,
    process_lines_batch,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get test device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def board_size():
    return 8


@pytest.fixture
def num_players():
    return 2


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_state(batch_size: int, board_size: int, num_players: int, device: torch.device):
    """Create a test BatchGameState."""
    return BatchGameState.create_batch(
        batch_size=batch_size,
        board_size=board_size,
        num_players=num_players,
        device=device,
    )


def place_marker_line(state, game_idx: int, player: int, start_y: int, start_x: int,
                      length: int, direction: tuple):
    """Place a line of markers for testing."""
    dy, dx = direction
    for i in range(length):
        y = start_y + i * dy
        x = start_x + i * dx
        if 0 <= y < state.board_size and 0 <= x < state.board_size:
            state.marker_owner[game_idx, y, x] = player
            # Ensure no stack at this position (lines are markers only)
            state.stack_owner[game_idx, y, x] = 0


# =============================================================================
# Tests for detect_lines_vectorized
# =============================================================================


class TestDetectLinesVectorized:
    """Tests for the vectorized line detection function."""

    def test_no_markers_returns_zero_counts(self, device, board_size, num_players):
        """Empty board should have no lines."""
        state = create_test_state(4, board_size, num_players, device)

        in_line_mask, line_counts = detect_lines_vectorized(state, player=1)

        assert in_line_mask.shape == (4, board_size, board_size)
        assert line_counts.shape == (4,)
        assert (line_counts == 0).all()
        assert not in_line_mask.any()

    def test_detects_horizontal_line(self, device, board_size, num_players):
        """Should detect a horizontal line of markers."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place horizontal line for player 1 in game 0
        place_marker_line(state, 0, player=1, start_y=3, start_x=1,
                         length=required_length, direction=(0, 1))

        in_line_mask, line_counts = detect_lines_vectorized(state, player=1)

        # Game 0 should have line positions
        assert line_counts[0].item() >= required_length
        # Game 1 should have no lines
        assert line_counts[1].item() == 0

        # Check that correct positions are marked
        for i in range(required_length):
            assert in_line_mask[0, 3, 1 + i].item()

    def test_detects_vertical_line(self, device, board_size, num_players):
        """Should detect a vertical line of markers."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place vertical line for player 1
        place_marker_line(state, 0, player=1, start_y=1, start_x=4,
                         length=required_length, direction=(1, 0))

        in_line_mask, line_counts = detect_lines_vectorized(state, player=1)

        assert line_counts[0].item() >= required_length

        # Check positions
        for i in range(required_length):
            assert in_line_mask[0, 1 + i, 4].item()

    def test_detects_diagonal_line(self, device, board_size, num_players):
        """Should detect a diagonal line (dy=1, dx=1)."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place diagonal line
        place_marker_line(state, 0, player=1, start_y=0, start_x=0,
                         length=required_length, direction=(1, 1))

        _in_line_mask, line_counts = detect_lines_vectorized(state, player=1)

        assert line_counts[0].item() >= required_length

    def test_detects_anti_diagonal_line(self, device, board_size, num_players):
        """Should detect an anti-diagonal line (dy=1, dx=-1)."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place anti-diagonal line starting from top-right area
        start_x = required_length - 1  # Ensure room for anti-diagonal
        place_marker_line(state, 0, player=1, start_y=0, start_x=start_x,
                         length=required_length, direction=(1, -1))

        _in_line_mask, line_counts = detect_lines_vectorized(state, player=1)

        assert line_counts[0].item() >= required_length

    def test_short_sequence_not_detected(self, device, board_size, num_players):
        """Sequence shorter than required length should not be detected."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place line that's one short
        place_marker_line(state, 0, player=1, start_y=3, start_x=1,
                         length=required_length - 1, direction=(0, 1))

        _in_line_mask, line_counts = detect_lines_vectorized(state, player=1)

        # Should not count as a line
        assert line_counts[0].item() == 0

    def test_stacks_block_lines(self, device, board_size, num_players):
        """Stacks on marker positions should break line detection."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place markers for a line
        for i in range(required_length):
            state.marker_owner[0, 3, i] = 1

        # Put a stack in the middle - this breaks the line
        middle = required_length // 2
        state.stack_owner[0, 3, middle] = 1
        state.stack_height[0, 3, middle] = 2

        _in_line_mask, line_counts = detect_lines_vectorized(state, player=1)

        # The stack breaks the line
        assert line_counts[0].item() < required_length

    def test_game_mask_filtering(self, device, board_size, num_players):
        """Game mask should filter which games are checked."""
        state = create_test_state(4, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place lines in games 0 and 2
        place_marker_line(state, 0, player=1, start_y=3, start_x=1,
                         length=required_length, direction=(0, 1))
        place_marker_line(state, 2, player=1, start_y=3, start_x=1,
                         length=required_length, direction=(0, 1))

        # Only check games 1 and 3
        game_mask = torch.tensor([False, True, False, True], dtype=torch.bool, device=device)

        _in_line_mask, line_counts = detect_lines_vectorized(state, player=1, game_mask=game_mask)

        # Games 0 and 2 should show 0 (masked out)
        assert line_counts[0].item() == 0
        assert line_counts[2].item() == 0

    def test_different_players_independent(self, device, board_size, num_players):
        """Lines for different players should be detected independently."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Player 1 has a line
        place_marker_line(state, 0, player=1, start_y=2, start_x=0,
                         length=required_length, direction=(0, 1))

        # Player 2 has a different line
        place_marker_line(state, 0, player=2, start_y=5, start_x=0,
                         length=required_length, direction=(0, 1))

        _, p1_counts = detect_lines_vectorized(state, player=1)
        _, p2_counts = detect_lines_vectorized(state, player=2)

        assert p1_counts[0].item() >= required_length
        assert p2_counts[0].item() >= required_length


# =============================================================================
# Tests for has_lines_batch_vectorized
# =============================================================================


class TestHasLinesBatchVectorized:
    """Tests for the fast line existence check."""

    def test_returns_bool_tensor(self, device, board_size, num_players):
        """Should return a boolean tensor."""
        state = create_test_state(4, board_size, num_players, device)

        result = has_lines_batch_vectorized(state, player=1)

        assert result.shape == (4,)
        assert result.dtype == torch.bool

    def test_detects_line_presence(self, device, board_size, num_players):
        """Should detect when a player has lines."""
        state = create_test_state(4, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Only game 1 has a line
        place_marker_line(state, 1, player=1, start_y=3, start_x=0,
                         length=required_length, direction=(0, 1))

        result = has_lines_batch_vectorized(state, player=1)

        assert not result[0].item()
        assert result[1].item()
        assert not result[2].item()
        assert not result[3].item()


# =============================================================================
# Tests for detect_lines_with_metadata
# =============================================================================


class TestDetectLinesWithMetadata:
    """Tests for line detection with full metadata."""

    def test_returns_list_of_detected_lines(self, device, board_size, num_players):
        """Should return DetectedLine objects with full metadata."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place a line
        place_marker_line(state, 0, player=1, start_y=3, start_x=0,
                         length=required_length, direction=(0, 1))

        lines = detect_lines_with_metadata(state, player=1)

        assert len(lines) == 2  # One list per game
        assert len(lines[0]) >= 1  # Game 0 has at least one line
        assert len(lines[1]) == 0  # Game 1 has no lines

        # Check DetectedLine structure
        line = lines[0][0]
        assert hasattr(line, 'positions')
        assert hasattr(line, 'length')
        assert hasattr(line, 'is_overlength')
        assert hasattr(line, 'direction')

    def test_detects_overlength_lines(self, device, board_size, num_players):
        """Should mark lines longer than required as overlength."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place an overlength line
        place_marker_line(state, 0, player=1, start_y=3, start_x=0,
                         length=required_length + 1, direction=(0, 1))

        lines = detect_lines_with_metadata(state, player=1)

        assert len(lines[0]) >= 1
        line = lines[0][0]
        assert line.is_overlength
        assert line.length >= required_length + 1

    def test_exact_length_not_overlength(self, device, board_size, num_players):
        """Exact required length should not be marked as overlength."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place exact-length line
        place_marker_line(state, 0, player=1, start_y=3, start_x=0,
                         length=required_length, direction=(0, 1))

        lines = detect_lines_with_metadata(state, player=1)

        assert len(lines[0]) >= 1
        # Find the line we placed
        for line in lines[0]:
            if line.length == required_length:
                assert not line.is_overlength
                break

    def test_includes_direction(self, device, board_size, num_players):
        """Should include the direction of each line."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place horizontal line
        place_marker_line(state, 0, player=1, start_y=3, start_x=0,
                         length=required_length, direction=(0, 1))

        lines = detect_lines_with_metadata(state, player=1)

        line = lines[0][0]
        assert line.direction in [(0, 1), (1, 0), (1, 1), (1, -1)]


# =============================================================================
# Tests for detect_lines_batch
# =============================================================================


class TestDetectLinesBatch:
    """Tests for line detection returning position lists."""

    def test_returns_position_lists(self, device, board_size, num_players):
        """Should return lists of (y, x) position tuples."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        place_marker_line(state, 0, player=1, start_y=3, start_x=0,
                         length=required_length, direction=(0, 1))

        lines = detect_lines_batch(state, player=1)

        assert len(lines) == 2
        assert len(lines[0]) >= required_length  # Positions in line
        assert len(lines[1]) == 0  # No lines in game 1

        # Positions should be tuples
        if lines[0]:
            assert isinstance(lines[0][0], tuple)
            assert len(lines[0][0]) == 2

    def test_empty_when_no_lines(self, device, board_size, num_players):
        """Should return empty lists when no lines exist."""
        state = create_test_state(4, board_size, num_players, device)

        lines = detect_lines_batch(state, player=1)

        assert all(len(game_lines) == 0 for game_lines in lines)


# =============================================================================
# Tests for process_lines_batch
# =============================================================================


class TestProcessLinesBatch:
    """Tests for line processing (collapsing and elimination)."""

    def test_collapses_line_markers(self, device, board_size, num_players):
        """Processing should collapse markers into territory."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place a line
        place_marker_line(state, 0, player=1, start_y=3, start_x=0,
                         length=required_length, direction=(0, 1))

        # Add a stack for elimination cost
        state.stack_owner[0, 0, 0] = 1
        state.stack_height[0, 0, 0] = 2

        # Process lines
        process_lines_batch(state)

        # Markers should be collapsed
        for i in range(required_length):
            assert state.is_collapsed[0, 3, i].item()
            assert state.territory_owner[0, 3, i].item() == 1
            assert state.marker_owner[0, 3, i].item() == 0

    def test_elimination_cost_for_exact_length(self, device, board_size, num_players):
        """Exact-length lines should cost one ring elimination."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place exact-length line
        place_marker_line(state, 0, player=1, start_y=3, start_x=0,
                         length=required_length, direction=(0, 1))

        # Add a stack for elimination
        state.stack_owner[0, 0, 0] = 1
        state.stack_height[0, 0, 0] = 3
        initial_height = state.stack_height[0, 0, 0].item()

        process_lines_batch(state, option2_probability=0.0)  # Force option 1

        # One ring should be eliminated
        assert state.stack_height[0, 0, 0].item() == initial_height - 1

    def test_option2_no_elimination_cost(self, device, board_size, num_players):
        """Option 2 for overlength lines should have no elimination cost."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place overlength line
        place_marker_line(state, 0, player=1, start_y=3, start_x=0,
                         length=required_length + 2, direction=(0, 1))

        # Add a stack
        state.stack_owner[0, 0, 0] = 1
        state.stack_height[0, 0, 0] = 3
        initial_height = state.stack_height[0, 0, 0].item()

        # Force Option 2
        process_lines_batch(state, option2_probability=1.0)

        # No elimination cost for option 2
        assert state.stack_height[0, 0, 0].item() == initial_height

    def test_territory_count_increases(self, device, board_size, num_players):
        """Territory count should increase when line is processed."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        initial_territory = state.territory_count[0, 1].item()

        # Place a line
        place_marker_line(state, 0, player=1, start_y=3, start_x=0,
                         length=required_length, direction=(0, 1))

        # Add a stack for elimination
        state.stack_owner[0, 0, 0] = 1
        state.stack_height[0, 0, 0] = 2

        process_lines_batch(state)

        # Territory should have increased
        assert state.territory_count[0, 1].item() >= initial_territory + required_length

    def test_respects_game_mask(self, device, board_size, num_players):
        """Should only process games in the mask."""
        state = create_test_state(4, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place lines in games 0 and 2
        for g in [0, 2]:
            place_marker_line(state, g, player=1, start_y=3, start_x=0,
                             length=required_length, direction=(0, 1))
            state.stack_owner[g, 0, 0] = 1
            state.stack_height[g, 0, 0] = 2

        # Only process game 0
        game_mask = torch.tensor([True, False, False, False], dtype=torch.bool, device=device)
        process_lines_batch(state, game_mask=game_mask)

        # Game 0 should be processed
        assert state.is_collapsed[0, 3, 0].item()
        # Game 2 should NOT be processed
        assert not state.is_collapsed[2, 3, 0].item()


# =============================================================================
# Edge Cases
# =============================================================================


class TestLineDetectionEdgeCases:
    """Edge cases and corner scenarios."""

    def test_multiple_lines_same_game(self, device, board_size, num_players):
        """Should detect multiple lines in the same game."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place two parallel horizontal lines
        place_marker_line(state, 0, player=1, start_y=2, start_x=0,
                         length=required_length, direction=(0, 1))
        place_marker_line(state, 0, player=1, start_y=5, start_x=0,
                         length=required_length, direction=(0, 1))

        _, line_counts = detect_lines_vectorized(state, player=1)

        # Should have positions from both lines
        assert line_counts[0].item() >= required_length * 2

    def test_intersecting_lines(self, device, board_size, num_players):
        """Should handle intersecting lines correctly."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place horizontal line
        place_marker_line(state, 0, player=1, start_y=3, start_x=0,
                         length=required_length, direction=(0, 1))
        # Place vertical line intersecting it
        place_marker_line(state, 0, player=1, start_y=0, start_x=2,
                         length=required_length, direction=(1, 0))

        _, line_counts = detect_lines_vectorized(state, player=1)

        # Should count all unique positions
        assert line_counts[0].item() > 0

    def test_line_at_board_edge(self, device, board_size, num_players):
        """Lines at board edges should be detected."""
        state = create_test_state(2, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place line at top edge
        place_marker_line(state, 0, player=1, start_y=0, start_x=0,
                         length=required_length, direction=(0, 1))

        _, line_counts = detect_lines_vectorized(state, player=1)

        assert line_counts[0].item() >= required_length

    def test_large_batch(self, device, board_size, num_players):
        """Should handle large batches efficiently."""
        batch_size = 500
        state = create_test_state(batch_size, board_size, num_players, device)
        required_length = get_required_line_length(board_size, num_players)

        # Place a line in every 10th game
        for g in range(0, batch_size, 10):
            place_marker_line(state, g, player=1, start_y=3, start_x=0,
                             length=required_length, direction=(0, 1))

        _, line_counts = detect_lines_vectorized(state, player=1)

        # Check every 10th game has lines
        for g in range(0, batch_size, 10):
            assert line_counts[g].item() >= required_length
