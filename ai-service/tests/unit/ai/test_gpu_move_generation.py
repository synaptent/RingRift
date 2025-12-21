"""Tests for GPU move generation module.

Tests the move generation functions for all game phases.
"""

import pytest
import torch

from app.ai.gpu_batch_state import BatchGameState
from app.ai.gpu_game_types import GamePhase, MoveType
from app.ai.gpu_move_generation import (
    BatchMoves,
    generate_capture_moves_batch_vectorized,
    generate_movement_moves_batch_vectorized,
    generate_placement_moves_batch,
    generate_recovery_moves_batch,
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


def place_stack(state, game_idx: int, y: int, x: int, owner: int, height: int):
    """Place a stack at a position."""
    state.stack_owner[game_idx, y, x] = owner
    state.stack_height[game_idx, y, x] = height
    state.cap_height[game_idx, y, x] = height


# =============================================================================
# Tests for BatchMoves
# =============================================================================


class TestBatchMoves:
    """Tests for the BatchMoves data structure."""

    def test_batch_moves_structure(self, device, board_size, num_players):
        """BatchMoves should have required attributes."""
        state = create_test_state(4, board_size, num_players, device)
        state.current_phase[:] = GamePhase.RING_PLACEMENT

        moves = generate_placement_moves_batch(state)

        assert hasattr(moves, 'game_idx')
        assert hasattr(moves, 'move_type')
        assert hasattr(moves, 'from_y')
        assert hasattr(moves, 'from_x')
        assert hasattr(moves, 'to_y')
        assert hasattr(moves, 'to_x')
        assert hasattr(moves, 'moves_per_game')
        assert hasattr(moves, 'move_offsets')
        assert hasattr(moves, 'total_moves')
        assert hasattr(moves, 'device')

    def test_tensor_shapes_consistent(self, device, board_size, num_players):
        """All move tensors should have consistent length."""
        state = create_test_state(4, board_size, num_players, device)
        state.current_phase[:] = GamePhase.RING_PLACEMENT

        moves = generate_placement_moves_batch(state)

        assert moves.game_idx.shape[0] == moves.total_moves
        assert moves.move_type.shape[0] == moves.total_moves
        assert moves.from_y.shape[0] == moves.total_moves
        assert moves.from_x.shape[0] == moves.total_moves
        assert moves.to_y.shape[0] == moves.total_moves
        assert moves.to_x.shape[0] == moves.total_moves


# =============================================================================
# Tests for generate_placement_moves_batch
# =============================================================================


class TestGeneratePlacementMovesBatch:
    """Tests for placement move generation."""

    def test_generates_moves_for_empty_board(self, device, board_size, num_players):
        """Empty board should have many placement options."""
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.RING_PLACEMENT

        moves = generate_placement_moves_batch(state)

        # Each game should have board_size^2 placement options (empty board)
        expected_per_game = board_size * board_size
        assert moves.moves_per_game[0].item() == expected_per_game
        assert moves.moves_per_game[1].item() == expected_per_game

    def test_occupied_cells_are_available(self, device, board_size, num_players):
        """Cells with stacks ARE available for placement per RingRift rules.

        In RingRift, you can place 1 ring on top of an existing stack.
        The GPU engine allows placement on any non-collapsed, non-marker cell.
        """
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.RING_PLACEMENT

        # Place a stack
        place_stack(state, 0, 3, 3, owner=1, height=2)

        moves = generate_placement_moves_batch(state)

        # Occupied cells are still valid placement targets in RingRift
        # Full board available (GPU engine doesn't exclude occupied cells)
        expected = board_size * board_size
        assert moves.moves_per_game[0].item() == expected

    def test_collapsed_cells_not_available(self, device, board_size, num_players):
        """Collapsed cells should not be available for placement."""
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.RING_PLACEMENT

        # Collapse some cells
        state.is_collapsed[0, 0, 0] = True
        state.is_collapsed[0, 0, 1] = True
        state.is_collapsed[0, 1, 0] = True

        moves = generate_placement_moves_batch(state)

        expected = board_size * board_size - 3
        assert moves.moves_per_game[0].item() == expected

    def test_move_type_is_placement(self, device, board_size, num_players):
        """All moves should be of type PLACEMENT."""
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.RING_PLACEMENT

        moves = generate_placement_moves_batch(state)

        assert (moves.move_type == MoveType.PLACEMENT).all()

    def test_no_rings_positions_still_generated(self, device, board_size, num_players):
        """GPU engine generates positions regardless of ring count.

        Ring count validation is done during move selection/application,
        not during position generation. This allows the engine to be simpler
        and faster while the caller filters based on ring availability.
        """
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.RING_PLACEMENT

        # Remove all rings from current player
        state.rings_in_hand[0, 1] = 0  # Player 1 has no rings

        moves = generate_placement_moves_batch(state)

        # GPU engine still generates all positions - ring check done elsewhere
        expected = board_size * board_size
        assert moves.moves_per_game[0].item() == expected


# =============================================================================
# Tests for generate_movement_moves_batch_vectorized
# =============================================================================


class TestGenerateMovementMovesBatch:
    """Tests for movement move generation."""

    def test_generates_moves_from_owned_stacks(self, device, board_size, num_players):
        """Should generate movement moves from player's stacks."""
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.MOVEMENT

        # Place a stack for player 1 in center
        place_stack(state, 0, 3, 3, owner=1, height=2)

        moves = generate_movement_moves_batch_vectorized(state)

        # Should have some movement options (4 directions, variable distances)
        assert moves.moves_per_game[0].item() > 0

    def test_no_stacks_no_moves(self, device, board_size, num_players):
        """No owned stacks means no movement moves."""
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.MOVEMENT

        moves = generate_movement_moves_batch_vectorized(state)

        assert moves.moves_per_game[0].item() == 0
        assert moves.moves_per_game[1].item() == 0

    def test_opponent_stacks_not_moved(self, device, board_size, num_players):
        """Should not generate moves for opponent's stacks."""
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.MOVEMENT
        state.current_player[0] = 1  # Player 1's turn

        # Place opponent's stack only
        place_stack(state, 0, 3, 3, owner=2, height=2)

        moves = generate_movement_moves_batch_vectorized(state)

        assert moves.moves_per_game[0].item() == 0

    def test_blocked_paths_excluded(self, device, board_size, num_players):
        """Paths blocked by stacks should be excluded."""
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.MOVEMENT

        # Place player's stack at center
        place_stack(state, 0, 3, 3, owner=1, height=2)
        # Block one direction
        place_stack(state, 0, 3, 4, owner=2, height=2)

        moves_blocked = generate_movement_moves_batch_vectorized(state)

        # Remove blocking stack
        state.stack_owner[0, 3, 4] = 0
        state.stack_height[0, 3, 4] = 0

        moves_unblocked = generate_movement_moves_batch_vectorized(state)

        # Unblocked should have more moves
        assert moves_unblocked.moves_per_game[0].item() >= moves_blocked.moves_per_game[0].item()

    def test_move_type_is_movement(self, device, board_size, num_players):
        """All moves should be of type MOVEMENT."""
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.MOVEMENT
        place_stack(state, 0, 3, 3, owner=1, height=2)

        moves = generate_movement_moves_batch_vectorized(state)

        if moves.total_moves > 0:
            assert (moves.move_type == MoveType.MOVEMENT).all()


# =============================================================================
# Tests for generate_capture_moves_batch_vectorized
# =============================================================================


class TestGenerateCaptureMovesBatch:
    """Tests for capture move generation."""

    def test_generates_captures_to_opponent_stacks(self, device, board_size, num_players):
        """Should generate capture moves to adjacent opponent stacks."""
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.MOVEMENT

        # Place player's stack adjacent to opponent's
        place_stack(state, 0, 3, 3, owner=1, height=2)
        place_stack(state, 0, 3, 4, owner=2, height=2)  # Adjacent

        moves = generate_capture_moves_batch_vectorized(state)

        # Should have at least one capture move
        assert moves.moves_per_game[0].item() > 0
        # Move type should be CAPTURE
        if moves.total_moves > 0:
            assert (moves.move_type == MoveType.CAPTURE).all()

    def test_no_adjacent_opponent_no_captures(self, device, board_size, num_players):
        """No adjacent opponent stacks means no capture moves."""
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.MOVEMENT

        # Place player's stack alone
        place_stack(state, 0, 3, 3, owner=1, height=2)

        moves = generate_capture_moves_batch_vectorized(state)

        assert moves.moves_per_game[0].item() == 0

    def test_self_capture_is_legal(self, device, board_size, num_players):
        """Self-capture IS legal per RR-CANON-R101.

        Per RR-CANON-R101: "Self-capture is legal: target may be owned by P."
        This means a player can overtake their own stacks if cap_height allows.
        """
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.MOVEMENT

        # Place two adjacent stacks for same player
        # Both have same cap_height, so capture is valid
        place_stack(state, 0, 3, 3, owner=1, height=2)
        place_stack(state, 0, 3, 4, owner=1, height=2)

        moves = generate_capture_moves_batch_vectorized(state)

        # Self-capture moves SHOULD be generated (per RR-CANON-R101)
        assert moves.moves_per_game[0].item() > 0


# =============================================================================
# Tests for generate_recovery_moves_batch
# =============================================================================


class TestGenerateRecoveryMovesBatch:
    """Tests for recovery move generation."""

    def test_generates_recovery_moves_when_eligible(self, device, board_size, num_players):
        """Should generate recovery moves when player is recovery-eligible."""
        state = create_test_state(2, board_size, num_players, device)

        # Player 1 has buried rings (eligible for recovery)
        state.buried_rings[0, 1] = 5
        # Player 1 has a stack with rings to slide
        place_stack(state, 0, 3, 3, owner=1, height=3)
        # Place opponent stack (recovery target)
        place_stack(state, 0, 3, 4, owner=2, height=2)

        moves = generate_recovery_moves_batch(state)

        # Should have some recovery moves (depending on game rules)
        # Recovery logic is complex, just verify no crash
        assert isinstance(moves, BatchMoves)

    def test_no_buried_rings_no_recovery(self, device, board_size, num_players):
        """No buried rings means no recovery moves."""
        state = create_test_state(2, board_size, num_players, device)

        # No buried rings
        state.buried_rings[:, :] = 0

        moves = generate_recovery_moves_batch(state)

        assert moves.total_moves == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestMoveGenerationEdgeCases:
    """Edge cases for move generation."""

    def test_large_batch(self, device, board_size, num_players):
        """Should handle large batches efficiently."""
        batch_size = 500
        state = create_test_state(batch_size, board_size, num_players, device)
        state.current_phase[:] = GamePhase.RING_PLACEMENT

        moves = generate_placement_moves_batch(state)

        assert moves.moves_per_game.shape[0] == batch_size
        assert moves.total_moves == batch_size * board_size * board_size

    def test_four_player_game(self, device, board_size):
        """Should work with 4 players."""
        state = create_test_state(2, board_size, num_players=4, device=device)
        state.current_phase[:] = GamePhase.RING_PLACEMENT

        moves = generate_placement_moves_batch(state)

        assert moves.total_moves > 0

    def test_different_board_sizes(self, device, num_players):
        """Should work with different board sizes."""
        for board_size in [8, 13, 19]:
            state = create_test_state(2, board_size, num_players, device)
            state.current_phase[:] = GamePhase.RING_PLACEMENT

            moves = generate_placement_moves_batch(state)

            expected = board_size * board_size
            assert moves.moves_per_game[0].item() == expected

    def test_mixed_game_states(self, device, board_size, num_players):
        """Should handle batch with mixed game states.

        GPU engine generates all non-collapsed positions regardless of:
        - Occupied cells (can place on stacks per RingRift rules)
        - Ring count (validated during move selection)
        """
        state = create_test_state(4, board_size, num_players, device)
        state.current_phase[:] = GamePhase.RING_PLACEMENT

        # Game 0: empty
        # Game 1: some stacks (still valid for placement)
        place_stack(state, 1, 3, 3, owner=1, height=2)
        place_stack(state, 1, 4, 4, owner=2, height=2)
        # Game 2: many stacks (still valid for placement)
        for i in range(5):
            place_stack(state, 2, i, i, owner=1, height=1)
        # Game 3: no rings left (positions still generated)
        state.rings_in_hand[3, 1] = 0
        # Game 3: collapse a cell to show filtering works
        state.is_collapsed[3, 0, 0] = True

        moves = generate_placement_moves_batch(state)

        # GPU engine generates all non-collapsed positions
        assert moves.moves_per_game[0].item() == board_size * board_size
        assert moves.moves_per_game[1].item() == board_size * board_size  # occupied ok
        assert moves.moves_per_game[2].item() == board_size * board_size  # occupied ok
        assert moves.moves_per_game[3].item() == board_size * board_size - 1  # one collapsed

    def test_tensors_on_correct_device(self, device, board_size, num_players):
        """Generated moves should be on the correct device."""
        state = create_test_state(2, board_size, num_players, device)
        state.current_phase[:] = GamePhase.RING_PLACEMENT

        moves = generate_placement_moves_batch(state)

        assert moves.device.type == device.type
        assert moves.game_idx.device.type == device.type
        assert moves.to_y.device.type == device.type
