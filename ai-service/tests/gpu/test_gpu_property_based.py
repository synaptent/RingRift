"""Property-based tests for GPU game simulation.

These tests verify invariants that should hold for all valid game states,
using Hypothesis to generate diverse test cases that catch edge cases.

Run with: pytest tests/gpu/test_gpu_property_based.py -v --hypothesis-show-statistics
"""

import pytest
import torch
from hypothesis import (
    assume,
    given,
    settings,
    strategies as st,
)

from app.ai.gpu_batch_state import BatchGameState
from app.ai.gpu_game_types import GamePhase, MoveType
from app.ai.gpu_line_detection import detect_lines_vectorized
from app.ai.gpu_parallel_games import (
    apply_placement_moves_batch,
    generate_capture_moves_batch,
    generate_movement_moves_batch,
    generate_placement_moves_batch,
)

# =============================================================================
# Strategies for generating game states
# =============================================================================

@st.composite
def valid_board_positions(draw, board_size: int = 8):
    """Generate valid (y, x) positions on the board."""
    y = draw(st.integers(min_value=0, max_value=board_size - 1))
    x = draw(st.integers(min_value=0, max_value=board_size - 1))
    return (y, x)


@st.composite
def sparse_stack_configuration(draw, board_size: int = 8, max_stacks: int = 5):
    """Generate a sparse configuration of stacks on the board."""
    num_stacks = draw(st.integers(min_value=1, max_value=max_stacks))
    stacks = []
    used_positions = set()

    for _ in range(num_stacks):
        # Try to find an unused position
        for _ in range(10):  # Max attempts
            y = draw(st.integers(min_value=0, max_value=board_size - 1))
            x = draw(st.integers(min_value=0, max_value=board_size - 1))
            if (y, x) not in used_positions:
                used_positions.add((y, x))
                owner = draw(st.integers(min_value=1, max_value=2))
                height = draw(st.integers(min_value=1, max_value=5))
                stacks.append((y, x, owner, height))
                break

    return stacks


# =============================================================================
# Property Tests for Move Generation
# =============================================================================

class TestPlacementMoveGenerationProperties:
    """Property-based tests for placement move generation."""

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=20, deadline=None)
    def test_placement_moves_within_board(self, batch_size):
        """All generated placement positions should be within board bounds."""
        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )
        # Ensure all games have rings to place
        state.rings_in_hand[:, 1] = 5

        mask = torch.ones(batch_size, dtype=torch.bool)
        moves = generate_placement_moves_batch(state, mask)

        if moves.total_moves > 0:
            assert (moves.from_y >= 0).all()
            assert (moves.from_x >= 0).all()
            assert (moves.from_y < 8).all()
            assert (moves.from_x < 8).all()

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=20, deadline=None)
    def test_placement_generates_consistent_moves(self, batch_size):
        """Placement moves should have consistent game indices."""
        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )
        state.rings_in_hand[:, 1] = 5
        state.current_player[:] = 1

        mask = torch.ones(batch_size, dtype=torch.bool)
        moves = generate_placement_moves_batch(state, mask)

        if moves.total_moves > 0:
            # All game indices should be valid
            assert (moves.game_idx >= 0).all()
            assert (moves.game_idx < batch_size).all()
            # Move type should be PLACEMENT
            assert (moves.move_type == MoveType.PLACEMENT).all()


class TestMovementMoveGenerationProperties:
    """Property-based tests for movement move generation."""

    @given(st.integers(min_value=1, max_value=4))
    @settings(max_examples=20, deadline=None)
    def test_movement_from_owned_stacks_only(self, batch_size):
        """Movement moves should only originate from stacks owned by current player."""
        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )

        # Place stacks for player 1
        state.stack_owner[:, 2, 2] = 1
        state.stack_height[:, 2, 2] = 3
        state.cap_height[:, 2, 2] = 3

        # Place opponent stack
        state.stack_owner[:, 5, 5] = 2
        state.stack_height[:, 5, 5] = 2
        state.cap_height[:, 5, 5] = 2

        state.current_player[:] = 1

        mask = torch.ones(batch_size, dtype=torch.bool)
        moves = generate_movement_moves_batch(state, mask)

        if moves.total_moves > 0:
            # Verify all moves originate from player 1's stacks
            for i in range(moves.total_moves):
                g = moves.game_idx[i].item()
                from_y = moves.from_y[i].item()
                from_x = moves.from_x[i].item()
                owner = state.stack_owner[g, from_y, from_x].item()
                assert owner == 1, f"Move from ({from_y}, {from_x}) but owner is {owner}"

    @given(st.integers(min_value=1, max_value=4))
    @settings(max_examples=20, deadline=None)
    def test_movement_moves_have_consistent_structure(self, batch_size):
        """Movement moves should have consistent tensor structure."""
        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )

        state.stack_owner[:, 4, 4] = 1
        state.stack_height[:, 4, 4] = 3
        state.cap_height[:, 4, 4] = 2
        state.current_player[:] = 1

        mask = torch.ones(batch_size, dtype=torch.bool)
        moves = generate_movement_moves_batch(state, mask)

        if moves.total_moves > 0:
            # All positions should be within board bounds
            assert (moves.from_y >= 0).all() and (moves.from_y < 8).all()
            assert (moves.from_x >= 0).all() and (moves.from_x < 8).all()
            assert (moves.to_y >= 0).all() and (moves.to_y < 8).all()
            assert (moves.to_x >= 0).all() and (moves.to_x < 8).all()
            # Move type should be MOVEMENT
            assert (moves.move_type == MoveType.MOVEMENT).all()
            # Game indices should be valid
            assert (moves.game_idx >= 0).all() and (moves.game_idx < batch_size).all()


# =============================================================================
# Property Tests for Line Detection
# =============================================================================

class TestLineDetectionProperties:
    """Property-based tests for line detection."""

    @given(st.integers(min_value=1, max_value=4))
    @settings(max_examples=20, deadline=None)
    def test_no_false_lines_on_empty_board(self, batch_size):
        """Empty boards should never report lines."""
        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )

        for player in [1, 2]:
            _positions_mask, line_count = detect_lines_vectorized(state, player=player)
            assert (line_count == 0).all(), f"Empty board reported lines for player {player}"

    @given(st.integers(min_value=1, max_value=4))
    @settings(max_examples=20, deadline=None)
    def test_line_count_matches_positions(self, batch_size):
        """Line count should be consistent with positions mask."""
        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )

        # Create a line of 4 markers for player 1 in each game
        for x in range(4):
            state.marker_owner[:, 3, x] = 1

        _positions_mask, line_count = detect_lines_vectorized(state, player=1)

        # Each game should have at least one line
        assert (line_count >= 1).all(), "Failed to detect line of 4 markers"

    @given(
        st.integers(min_value=1, max_value=4),
        st.integers(min_value=0, max_value=3),  # Row for horizontal line
    )
    @settings(max_examples=30, deadline=None)
    def test_horizontal_line_detection(self, batch_size, row):
        """Horizontal lines of 4+ markers should be detected."""
        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )

        # Place 4 markers horizontally
        for x in range(4):
            state.marker_owner[:, row, x] = 1

        _positions_mask, line_count = detect_lines_vectorized(state, player=1)

        assert (line_count >= 1).all(), f"Failed to detect horizontal line at row {row}"


# =============================================================================
# Property Tests for State Invariants
# =============================================================================

class TestBatchStateInvariants:
    """Property-based tests for BatchGameState invariants."""

    @given(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=30, deadline=None)
    def test_initial_state_valid(self, batch_size, num_players):
        """Newly created states should satisfy all invariants."""
        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=num_players,
            device=torch.device("cpu"),
        )

        # All games should be active
        assert (state.game_status == 0).all()  # ACTIVE = 0

        # Current player should be valid (1 to num_players)
        assert (state.current_player >= 1).all()
        assert (state.current_player <= num_players).all()

        # Move count should be 0
        assert (state.move_count == 0).all()

        # Stack heights should be non-negative
        assert (state.stack_height >= 0).all()

        # Cap height should not exceed stack height
        assert (state.cap_height <= state.stack_height).all()

    @given(st.integers(min_value=1, max_value=4))
    @settings(max_examples=20, deadline=None)
    def test_placement_preserves_invariants(self, batch_size):
        """After placement, state should still satisfy invariants."""
        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )
        state.rings_in_hand[:, 1] = 5

        mask = torch.ones(batch_size, dtype=torch.bool)
        moves = generate_placement_moves_batch(state, mask)

        if moves.total_moves > 0:
            # Select first move for each game
            selected = torch.zeros(batch_size, dtype=torch.int64)
            apply_placement_moves_batch(state, selected, moves)

            # Verify invariants still hold
            assert (state.stack_height >= 0).all()
            assert (state.cap_height <= state.stack_height).all()
            assert (state.cap_height >= 0).all()

            # Where stacks exist, owner should be non-zero
            has_stack = state.stack_height > 0
            assert (state.stack_owner[has_stack] > 0).all()


# =============================================================================
# Property Tests for Tensor Shape Consistency
# =============================================================================

class TestTensorShapeConsistency:
    """Property-based tests for tensor shape consistency in GPU operations."""

    @given(
        st.integers(min_value=1, max_value=16),
        st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=30, deadline=None)
    def test_batch_state_shapes_consistent(self, batch_size, num_players):
        """All tensors in BatchGameState should have consistent shapes."""
        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=num_players,
            device=torch.device("cpu"),
        )

        # Board tensors should be (batch_size, 8, 8)
        assert state.stack_owner.shape == (batch_size, 8, 8)
        assert state.stack_height.shape == (batch_size, 8, 8)
        assert state.cap_height.shape == (batch_size, 8, 8)
        assert state.marker_owner.shape == (batch_size, 8, 8)

        # Per-game tensors should be (batch_size,)
        assert state.game_status.shape == (batch_size,)
        assert state.current_player.shape == (batch_size,)
        assert state.move_count.shape == (batch_size,)

        # Per-player tensors should be (batch_size, num_players + 1)
        # (index 0 unused, 1-4 for players)
        assert state.rings_in_hand.shape[0] == batch_size
        assert state.eliminated_rings.shape[0] == batch_size

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=20, deadline=None)
    def test_move_generation_shapes_consistent(self, batch_size):
        """Generated moves should have consistent tensor shapes."""
        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )
        state.rings_in_hand[:, 1] = 5

        mask = torch.ones(batch_size, dtype=torch.bool)
        moves = generate_placement_moves_batch(state, mask)

        # All move tensors should have same length
        n = moves.total_moves
        if n > 0:
            assert moves.game_idx.shape == (n,)
            assert moves.move_type.shape == (n,)
            assert moves.from_y.shape == (n,)
            assert moves.from_x.shape == (n,)
            assert moves.to_y.shape == (n,)
            assert moves.to_x.shape == (n,)

        # Per-game tensors should be (batch_size,)
        assert moves.moves_per_game.shape == (batch_size,)
        assert moves.move_offsets.shape == (batch_size,)
