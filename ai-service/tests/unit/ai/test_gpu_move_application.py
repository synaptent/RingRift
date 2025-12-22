"""Tests for gpu_move_application module.

Tests move application functions per RR-CANON rules:
- Placement moves (R080)
- Movement moves (R090-R092)
- Capture moves (R100-R103)
- Recovery slide moves (R110-R115)
"""

from dataclasses import dataclass

import pytest
import torch

from app.ai.gpu_game_types import GamePhase, MoveType
from app.ai.gpu_move_application import (
    # Legacy functions were removed - using current implementations
    apply_capture_moves_batch,
    apply_capture_moves_batch_vectorized,
    apply_capture_moves_vectorized,
    apply_movement_moves_batch,
    apply_movement_moves_batch_vectorized,
    apply_movement_moves_vectorized,
    apply_no_action_moves_batch,
    apply_placement_moves_batch,
    apply_placement_moves_batch_vectorized,
    apply_recovery_moves_vectorized,
)


@dataclass
class MockBatchGameState:
    """Mock BatchGameState for testing move application.

    Provides minimal interface needed by move application functions.
    """

    batch_size: int
    board_size: int
    device: torch.device
    num_players: int = 2
    max_history_moves: int = 500

    # Core state tensors
    stack_owner: torch.Tensor = None
    stack_height: torch.Tensor = None
    cap_height: torch.Tensor = None
    marker_owner: torch.Tensor = None
    is_collapsed: torch.Tensor = None
    territory_owner: torch.Tensor = None

    # Player state
    current_player: torch.Tensor = None
    rings_in_hand: torch.Tensor = None
    eliminated_rings: torch.Tensor = None
    buried_rings: torch.Tensor = None
    rings_caused_eliminated: torch.Tensor = None
    territory_count: torch.Tensor = None

    # Game state
    game_over: torch.Tensor = None
    winner: torch.Tensor = None
    move_count: torch.Tensor = None
    current_phase: torch.Tensor = None
    game_status: torch.Tensor = None

    # Buried ring tracking (December 2025)
    buried_at: torch.Tensor = None

    # Constraints
    must_move_from_y: torch.Tensor = None
    must_move_from_x: torch.Tensor = None

    # Capture chain tracking (December 2025)
    in_capture_chain: torch.Tensor = None
    capture_chain_depth: torch.Tensor = None

    # History
    move_history: torch.Tensor = None

    def __post_init__(self):
        """Initialize tensors if not provided."""
        bs, bz = self.batch_size, self.board_size

        if self.stack_owner is None:
            self.stack_owner = torch.zeros((bs, bz, bz), dtype=torch.int8, device=self.device)
        if self.stack_height is None:
            self.stack_height = torch.zeros((bs, bz, bz), dtype=torch.int8, device=self.device)
        if self.cap_height is None:
            self.cap_height = torch.zeros((bs, bz, bz), dtype=torch.int8, device=self.device)
        if self.marker_owner is None:
            self.marker_owner = torch.zeros((bs, bz, bz), dtype=torch.int8, device=self.device)
        if self.is_collapsed is None:
            self.is_collapsed = torch.zeros((bs, bz, bz), dtype=torch.bool, device=self.device)
        if self.territory_owner is None:
            self.territory_owner = torch.zeros((bs, bz, bz), dtype=torch.int8, device=self.device)

        if self.current_player is None:
            self.current_player = torch.ones(bs, dtype=torch.int8, device=self.device)
        if self.rings_in_hand is None:
            self.rings_in_hand = torch.zeros((bs, self.num_players + 1), dtype=torch.int8, device=self.device)
            # Give each player some rings
            self.rings_in_hand[:, 1] = 10
            self.rings_in_hand[:, 2] = 10
        if self.eliminated_rings is None:
            self.eliminated_rings = torch.zeros((bs, self.num_players + 1), dtype=torch.int32, device=self.device)
        if self.buried_rings is None:
            self.buried_rings = torch.zeros((bs, self.num_players + 1), dtype=torch.int32, device=self.device)
        if self.rings_caused_eliminated is None:
            self.rings_caused_eliminated = torch.zeros((bs, self.num_players + 1), dtype=torch.int32, device=self.device)
        if self.territory_count is None:
            self.territory_count = torch.zeros((bs, self.num_players + 1), dtype=torch.int32, device=self.device)

        if self.game_over is None:
            self.game_over = torch.zeros(bs, dtype=torch.bool, device=self.device)
        if self.winner is None:
            self.winner = torch.zeros(bs, dtype=torch.int8, device=self.device)
        if self.move_count is None:
            self.move_count = torch.zeros(bs, dtype=torch.int32, device=self.device)
        if self.current_phase is None:
            self.current_phase = torch.full((bs,), GamePhase.RING_PLACEMENT, dtype=torch.int8, device=self.device)
        if self.game_status is None:
            self.game_status = torch.zeros(bs, dtype=torch.int8, device=self.device)

        if self.buried_at is None:
            self.buried_at = torch.zeros((bs, self.num_players + 1, bz, bz), dtype=torch.bool, device=self.device)

        if self.must_move_from_y is None:
            self.must_move_from_y = torch.full((bs,), -1, dtype=torch.int32, device=self.device)
        if self.must_move_from_x is None:
            self.must_move_from_x = torch.full((bs,), -1, dtype=torch.int32, device=self.device)

        if self.in_capture_chain is None:
            self.in_capture_chain = torch.zeros(bs, dtype=torch.bool, device=self.device)
        if self.capture_chain_depth is None:
            self.capture_chain_depth = torch.zeros(bs, dtype=torch.int16, device=self.device)

        if self.move_history is None:
            # 9 columns: move_type, player, from_y, from_x, to_y, to_x, phase, capture_target_y, capture_target_x
            self.move_history = torch.full((bs, self.max_history_moves, 9), -1, dtype=torch.int16, device=self.device)

    def get_active_mask(self) -> torch.Tensor:
        """Return mask of active (non-finished) games."""
        return ~self.game_over

    def place_stack(self, game_idx: int, y: int, x: int, owner: int, height: int, cap: int | None = None):
        """Helper to place a stack on the board."""
        if cap is None:
            cap = height
        self.stack_owner[game_idx, y, x] = owner
        self.stack_height[game_idx, y, x] = height
        self.cap_height[game_idx, y, x] = cap

    def place_marker(self, game_idx: int, y: int, x: int, owner: int):
        """Helper to place a marker on the board."""
        self.marker_owner[game_idx, y, x] = owner

    def collapse_cell(self, game_idx: int, y: int, x: int, territory_owner: int = 0):
        """Helper to collapse a cell."""
        self.is_collapsed[game_idx, y, x] = True
        if territory_owner > 0:
            self.territory_owner[game_idx, y, x] = territory_owner


@dataclass
class MockBatchMoves:
    """Mock BatchMoves for testing move application."""

    batch_size: int
    total_moves: int
    device: torch.device

    # Move data tensors
    from_y: torch.Tensor = None
    from_x: torch.Tensor = None
    to_y: torch.Tensor = None
    to_x: torch.Tensor = None
    move_type: torch.Tensor = None
    game_indices: torch.Tensor = None

    # Per-game info
    moves_per_game: torch.Tensor = None
    move_offsets: torch.Tensor = None

    def __post_init__(self):
        """Initialize tensors if not provided."""
        if self.from_y is None:
            self.from_y = torch.zeros(self.total_moves, dtype=torch.int32, device=self.device)
        if self.from_x is None:
            self.from_x = torch.zeros(self.total_moves, dtype=torch.int32, device=self.device)
        if self.to_y is None:
            self.to_y = torch.zeros(self.total_moves, dtype=torch.int32, device=self.device)
        if self.to_x is None:
            self.to_x = torch.zeros(self.total_moves, dtype=torch.int32, device=self.device)
        if self.move_type is None:
            self.move_type = torch.zeros(self.total_moves, dtype=torch.int8, device=self.device)
        if self.game_indices is None:
            self.game_indices = torch.zeros(self.total_moves, dtype=torch.int32, device=self.device)
        if self.moves_per_game is None:
            self.moves_per_game = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        if self.move_offsets is None:
            self.move_offsets = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)


def create_single_move(
    device: torch.device,
    batch_size: int,
    game_idx: int,
    from_y: int,
    from_x: int,
    to_y: int,
    to_x: int,
    move_type: int = MoveType.PLACEMENT,
) -> MockBatchMoves:
    """Create a MockBatchMoves with a single move for one game."""
    moves = MockBatchMoves(batch_size=batch_size, total_moves=1, device=device)
    moves.from_y[0] = from_y
    moves.from_x[0] = from_x
    moves.to_y[0] = to_y
    moves.to_x[0] = to_x
    moves.move_type[0] = move_type
    moves.game_indices[0] = game_idx
    moves.moves_per_game[game_idx] = 1
    # Set offsets for all games
    for g in range(batch_size):
        if g <= game_idx:
            moves.move_offsets[g] = 0
        else:
            moves.move_offsets[g] = 1
    return moves


# =============================================================================
# Test apply_no_action_moves_batch
# =============================================================================


class TestApplyNoActionMovesBatch:
    """Tests for apply_no_action_moves_batch."""

    def test_no_action_records_history(self):
        """Test that NO_ACTION is recorded in move history."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=2, board_size=7, device=device)
        state.current_player[0] = 1
        state.current_player[1] = 2

        mask = torch.tensor([True, True], device=device)
        apply_no_action_moves_batch(state, mask)

        # Check move count incremented
        assert state.move_count[0].item() == 1
        assert state.move_count[1].item() == 1

        # Check history recorded correctly
        # December 2025: Canonical phases use phase-specific NO_*_ACTION types
        # Default phase is RING_PLACEMENT → NO_PLACEMENT_ACTION
        assert state.move_history[0, 0, 0].item() == MoveType.NO_PLACEMENT_ACTION
        assert state.move_history[0, 0, 1].item() == 1  # player 1
        assert state.move_history[1, 0, 0].item() == MoveType.NO_PLACEMENT_ACTION
        assert state.move_history[1, 0, 1].item() == 2  # player 2

    def test_no_action_respects_mask(self):
        """Test that NO_ACTION only applies to masked games."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=3, board_size=7, device=device)

        mask = torch.tensor([True, False, True], device=device)
        apply_no_action_moves_batch(state, mask)

        assert state.move_count[0].item() == 1
        assert state.move_count[1].item() == 0  # Not masked
        assert state.move_count[2].item() == 1

    def test_no_action_respects_game_over(self):
        """Test that NO_ACTION doesn't apply to finished games."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=2, board_size=7, device=device)
        state.game_over[1] = True

        mask = torch.tensor([True, True], device=device)
        apply_no_action_moves_batch(state, mask)

        assert state.move_count[0].item() == 1
        assert state.move_count[1].item() == 0  # Game over

    def test_no_action_with_all_false_mask(self):
        """Test NO_ACTION with empty mask."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=2, board_size=7, device=device)

        mask = torch.tensor([False, False], device=device)
        apply_no_action_moves_batch(state, mask)

        assert state.move_count[0].item() == 0
        assert state.move_count[1].item() == 0


# =============================================================================
# Test apply_placement_moves_batch
# =============================================================================


class TestApplyPlacementMovesBatch:
    """Tests for apply_placement_moves_batch."""

    def test_placement_on_empty_cell(self):
        """Test placing a ring on an empty cell."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.rings_in_hand[0, 1] = 10

        moves = create_single_move(device, 1, 0, 3, 3, 3, 3, MoveType.PLACEMENT)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_placement_moves_batch(state, move_indices, moves)

        # Check stack created
        assert state.stack_owner[0, 3, 3].item() == 1
        assert state.stack_height[0, 3, 3].item() == 1
        assert state.cap_height[0, 3, 3].item() == 1

        # Check rings_in_hand decremented
        assert state.rings_in_hand[0, 1].item() == 9

        # Check must_move_from set
        assert state.must_move_from_y[0].item() == 3
        assert state.must_move_from_x[0].item() == 3

        # Check move_count incremented
        assert state.move_count[0].item() == 1

    def test_placement_on_own_stack(self):
        """Test placing a ring on own stack increases cap height."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.rings_in_hand[0, 1] = 10
        state.place_stack(0, 3, 3, owner=1, height=2, cap=2)

        moves = create_single_move(device, 1, 0, 3, 3, 3, 3, MoveType.PLACEMENT)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_placement_moves_batch(state, move_indices, moves)

        assert state.stack_owner[0, 3, 3].item() == 1
        assert state.stack_height[0, 3, 3].item() == 3
        assert state.cap_height[0, 3, 3].item() == 3

    def test_placement_on_opponent_stack(self):
        """Test placing a ring on opponent stack resets cap height and tracks buried."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.rings_in_hand[0, 1] = 10
        state.place_stack(0, 3, 3, owner=2, height=2, cap=2)

        moves = create_single_move(device, 1, 0, 3, 3, 3, 3, MoveType.PLACEMENT)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_placement_moves_batch(state, move_indices, moves)

        assert state.stack_owner[0, 3, 3].item() == 1
        assert state.stack_height[0, 3, 3].item() == 3
        assert state.cap_height[0, 3, 3].item() == 1  # Reset for new owner
        assert state.buried_rings[0, 2].item() == 1  # Opponent ring buried

    def test_placement_skips_finished_games(self):
        """Test that placement skips finished games."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=2, board_size=7, device=device)
        state.current_player[:] = 1
        state.game_over[1] = True

        moves = MockBatchMoves(batch_size=2, total_moves=2, device=device)
        moves.from_y[0] = 3
        moves.from_x[0] = 3
        moves.to_y[0] = 3
        moves.to_x[0] = 3
        moves.from_y[1] = 3
        moves.from_x[1] = 3
        moves.to_y[1] = 3
        moves.to_x[1] = 3
        moves.moves_per_game[:] = 1
        moves.move_offsets[0] = 0
        moves.move_offsets[1] = 1

        move_indices = torch.tensor([0, 0], dtype=torch.int64, device=device)
        apply_placement_moves_batch(state, move_indices, moves)

        assert state.stack_height[0, 3, 3].item() == 1
        assert state.stack_height[1, 3, 3].item() == 0  # Game was over

    def test_placement_records_history(self):
        """Test that placement is recorded in move history."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1

        moves = create_single_move(device, 1, 0, 4, 5, 4, 5, MoveType.PLACEMENT)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_placement_moves_batch(state, move_indices, moves)

        assert state.move_history[0, 0, 0].item() == MoveType.PLACEMENT
        assert state.move_history[0, 0, 1].item() == 1  # player
        assert state.move_history[0, 0, 2].item() == 4  # y
        assert state.move_history[0, 0, 3].item() == 5  # x


@pytest.mark.skip(reason="Legacy functions removed - use vectorized implementations")
class TestApplyPlacementLegacy:
    """Tests for _apply_placement_moves_batch_legacy."""

    def test_legacy_placement_on_empty_cell(self):
        """Test legacy placement on empty cell."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.rings_in_hand[0, 1] = 10

        moves = create_single_move(device, 1, 0, 3, 3, 3, 3, MoveType.PLACEMENT)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        _apply_placement_moves_batch_legacy(state, move_indices, moves)

        assert state.stack_owner[0, 3, 3].item() == 1
        assert state.stack_height[0, 3, 3].item() == 1
        assert state.cap_height[0, 3, 3].item() == 1


# =============================================================================
# Test apply_movement_moves_batch
# =============================================================================


class TestApplyMovementMovesBatch:
    """Tests for apply_movement_moves_batch."""

    def test_movement_basic(self):
        """Test basic movement from one cell to another."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_stack(0, 2, 2, owner=1, height=2, cap=2)

        moves = create_single_move(device, 1, 0, 2, 2, 4, 2, MoveType.MOVEMENT)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_movement_moves_batch(state, move_indices, moves)

        # Origin should be empty
        assert state.stack_owner[0, 2, 2].item() == 0
        assert state.stack_height[0, 2, 2].item() == 0

        # Destination should have stack
        assert state.stack_owner[0, 4, 2].item() == 1
        assert state.stack_height[0, 4, 2].item() == 2

        # Move count incremented
        assert state.move_count[0].item() == 1

    def test_movement_flips_opponent_markers(self):
        """Test that movement flips opponent markers along path."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_stack(0, 2, 2, owner=1, height=2, cap=2)
        state.place_marker(0, 3, 2, owner=2)  # Opponent marker on path

        moves = create_single_move(device, 1, 0, 2, 2, 4, 2, MoveType.MOVEMENT)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_movement_moves_batch(state, move_indices, moves)

        # Opponent marker should be flipped to player 1
        assert state.marker_owner[0, 3, 2].item() == 1

    def test_movement_records_history(self):
        """Test that movement is recorded in history."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_stack(0, 2, 2, owner=1, height=2, cap=2)

        moves = create_single_move(device, 1, 0, 2, 2, 4, 4, MoveType.MOVEMENT)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_movement_moves_batch(state, move_indices, moves)

        assert state.move_history[0, 0, 0].item() == MoveType.MOVEMENT
        assert state.move_history[0, 0, 1].item() == 1
        assert state.move_history[0, 0, 2].item() == 2  # from_y
        assert state.move_history[0, 0, 3].item() == 2  # from_x
        assert state.move_history[0, 0, 4].item() == 4  # to_y
        assert state.move_history[0, 0, 5].item() == 4  # to_x

    def test_movement_skips_no_moves(self):
        """Test that movement skips games with no moves."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)

        moves = MockBatchMoves(batch_size=1, total_moves=0, device=device)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_movement_moves_batch(state, move_indices, moves)
        assert state.move_count[0].item() == 0


@pytest.mark.skip(reason="Legacy functions removed - use vectorized implementations")
class TestApplyMovementLegacy:
    """Tests for _apply_movement_moves_batch_legacy."""

    def test_legacy_movement_basic(self):
        """Test legacy movement."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_stack(0, 2, 2, owner=1, height=2, cap=2)

        moves = create_single_move(device, 1, 0, 2, 2, 4, 2, MoveType.MOVEMENT)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        _apply_movement_moves_batch_legacy(state, move_indices, moves)

        assert state.stack_owner[0, 2, 2].item() == 0
        assert state.stack_owner[0, 4, 2].item() == 1


# =============================================================================
# Test apply_capture_moves_batch
# =============================================================================


class TestApplyCaptureMovesBatch:
    """Tests for apply_capture_moves_batch."""

    def test_capture_basic(self):
        """Test basic capture: attacker captures target and lands beyond."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_stack(0, 2, 2, owner=1, height=2, cap=2)  # Attacker
        state.place_stack(0, 3, 2, owner=2, height=1, cap=1)  # Target

        # Move: from (2,2), landing at (4,2), target at (3,2)
        moves = create_single_move(device, 1, 0, 2, 2, 4, 2, MoveType.CAPTURE)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_capture_moves_batch(state, move_indices, moves)

        # Origin cleared
        assert state.stack_owner[0, 2, 2].item() == 0
        assert state.stack_height[0, 2, 2].item() == 0

        # Target reduced
        assert state.stack_height[0, 3, 2].item() == 0  # Was height 1, now 0

        # Attacker at landing
        assert state.stack_owner[0, 4, 2].item() == 1

        # Move count incremented
        assert state.move_count[0].item() == 1

        # Rings tracking updated
        # Captured rings are BURIED, not eliminated (can be liberated if stack is re-captured)
        assert state.buried_rings[0, 2].item() == 1
        # rings_caused_eliminated tracks any ring eliminations caused by this player
        # (captures bury rings, so this wouldn't increment for captures alone)

    def test_capture_clears_must_move(self):
        """Test that capture clears must_move constraint."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_stack(0, 2, 2, owner=1, height=2, cap=2)
        state.place_stack(0, 3, 2, owner=2, height=1, cap=1)
        state.must_move_from_y[0] = 2
        state.must_move_from_x[0] = 2

        moves = create_single_move(device, 1, 0, 2, 2, 4, 2, MoveType.CAPTURE)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_capture_moves_batch(state, move_indices, moves)

        assert state.must_move_from_y[0].item() == -1
        assert state.must_move_from_x[0].item() == -1

    def test_capture_records_history(self):
        """Test that capture is recorded in history."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_stack(0, 2, 2, owner=1, height=2, cap=2)
        state.place_stack(0, 3, 2, owner=2, height=1, cap=1)

        moves = create_single_move(device, 1, 0, 2, 2, 4, 2, MoveType.CAPTURE)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_capture_moves_batch(state, move_indices, moves)

        # December 2025: Canonical phases record initial captures as OVERTAKING_CAPTURE
        assert state.move_history[0, 0, 0].item() == MoveType.OVERTAKING_CAPTURE
        assert state.move_history[0, 0, 1].item() == 1
        assert state.move_history[0, 0, 2].item() == 2  # from_y
        assert state.move_history[0, 0, 3].item() == 2  # from_x


@pytest.mark.skip(reason="Legacy functions removed - use vectorized implementations")
class TestApplyCaptureLegacy:
    """Tests for _apply_capture_moves_batch_legacy."""

    def test_legacy_capture_basic(self):
        """Test legacy capture."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_stack(0, 2, 2, owner=1, height=2, cap=2)
        state.place_stack(0, 3, 2, owner=2, height=1, cap=1)

        moves = create_single_move(device, 1, 0, 2, 2, 3, 2, MoveType.CAPTURE)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        _apply_capture_moves_batch_legacy(state, move_indices, moves)

        assert state.stack_owner[0, 2, 2].item() == 0


# =============================================================================
# Test apply_capture_moves_vectorized
# =============================================================================


class TestApplyCaptureMovesVectorized:
    """Tests for apply_capture_moves_vectorized."""

    def test_capture_vectorized_empty_selection(self):
        """Test capture with no selected moves."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=2, board_size=7, device=device)

        moves = MockBatchMoves(batch_size=2, total_moves=0, device=device)
        selected_local_idx = torch.tensor([-1, -1], dtype=torch.int64, device=device)
        active_mask = torch.tensor([True, True], device=device)

        apply_capture_moves_vectorized(state, selected_local_idx, moves, active_mask)

        # Nothing should change
        assert state.move_count.sum().item() == 0

    def test_capture_vectorized_respects_mask(self):
        """Test that capture respects active mask."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=2, board_size=7, device=device)
        state.current_player[:] = 1
        state.place_stack(0, 2, 2, owner=1, height=2, cap=2)
        state.place_stack(0, 3, 2, owner=2, height=1, cap=1)
        state.place_stack(1, 2, 2, owner=1, height=2, cap=2)
        state.place_stack(1, 3, 2, owner=2, height=1, cap=1)

        moves = MockBatchMoves(batch_size=2, total_moves=2, device=device)
        moves.from_y[0] = 2
        moves.from_x[0] = 2
        moves.to_y[0] = 4
        moves.to_x[0] = 2
        moves.from_y[1] = 2
        moves.from_x[1] = 2
        moves.to_y[1] = 4
        moves.to_x[1] = 2
        moves.moves_per_game[:] = 1
        moves.move_offsets[0] = 0
        moves.move_offsets[1] = 1

        selected_local_idx = torch.tensor([0, 0], dtype=torch.int64, device=device)
        active_mask = torch.tensor([True, False], device=device)

        apply_capture_moves_vectorized(state, selected_local_idx, moves, active_mask)

        # Only game 0 should have move applied
        assert state.move_count[0].item() == 1
        assert state.move_count[1].item() == 0


# =============================================================================
# Test apply_movement_moves_vectorized
# =============================================================================


class TestApplyMovementMovesVectorized:
    """Tests for apply_movement_moves_vectorized."""

    def test_movement_vectorized_empty_selection(self):
        """Test movement with no selected moves."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=2, board_size=7, device=device)

        moves = MockBatchMoves(batch_size=2, total_moves=0, device=device)
        selected_local_idx = torch.tensor([-1, -1], dtype=torch.int64, device=device)
        active_mask = torch.tensor([True, True], device=device)

        apply_movement_moves_vectorized(state, selected_local_idx, moves, active_mask)

        assert state.move_count.sum().item() == 0

    def test_movement_vectorized_clears_must_move(self):
        """Test that movement clears must_move constraint."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_stack(0, 2, 2, owner=1, height=2, cap=2)
        state.must_move_from_y[0] = 2
        state.must_move_from_x[0] = 2

        moves = MockBatchMoves(batch_size=1, total_moves=1, device=device)
        moves.from_y[0] = 2
        moves.from_x[0] = 2
        moves.to_y[0] = 4
        moves.to_x[0] = 2
        moves.moves_per_game[0] = 1

        selected_local_idx = torch.tensor([0], dtype=torch.int64, device=device)
        active_mask = torch.tensor([True], device=device)

        apply_movement_moves_vectorized(state, selected_local_idx, moves, active_mask)

        assert state.must_move_from_y[0].item() == -1
        assert state.must_move_from_x[0].item() == -1


# =============================================================================
# Test apply_recovery_moves_vectorized
# =============================================================================


class TestApplyRecoveryMovesVectorized:
    """Tests for apply_recovery_moves_vectorized."""

    def test_recovery_normal_slide(self):
        """Test normal recovery slide moves marker."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_marker(0, 3, 3, owner=1)

        moves = MockBatchMoves(batch_size=1, total_moves=1, device=device)
        moves.from_y[0] = 3
        moves.from_x[0] = 3
        moves.to_y[0] = 4
        moves.to_x[0] = 3
        moves.moves_per_game[0] = 1

        selected_local_idx = torch.tensor([0], dtype=torch.int64, device=device)
        active_mask = torch.tensor([True], device=device)

        apply_recovery_moves_vectorized(state, selected_local_idx, moves, active_mask)

        # Source marker cleared
        assert state.marker_owner[0, 3, 3].item() == 0
        # Destination marker set
        assert state.marker_owner[0, 4, 3].item() == 1
        # Move count incremented
        assert state.move_count[0].item() == 1

    def test_recovery_stack_strike(self):
        """Test recovery slide hitting a stack (stack-strike)."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_marker(0, 3, 3, owner=1)
        state.place_stack(0, 4, 3, owner=2, height=2, cap=2)  # Target stack

        moves = MockBatchMoves(batch_size=1, total_moves=1, device=device)
        moves.from_y[0] = 3
        moves.from_x[0] = 3
        moves.to_y[0] = 4
        moves.to_x[0] = 3
        moves.moves_per_game[0] = 1

        selected_local_idx = torch.tensor([0], dtype=torch.int64, device=device)
        active_mask = torch.tensor([True], device=device)

        apply_recovery_moves_vectorized(state, selected_local_idx, moves, active_mask)

        # Source marker cleared
        assert state.marker_owner[0, 3, 3].item() == 0
        # Stack height reduced
        assert state.stack_height[0, 4, 3].item() == 1
        # Eliminated ring tracked
        assert state.eliminated_rings[0, 2].item() == 1
        assert state.rings_caused_eliminated[0, 1].item() == 1

    def test_recovery_deducts_buried_ring(self):
        """Test that recovery deducts buried ring cost."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_marker(0, 3, 3, owner=1)
        state.buried_rings[0, 1] = 2  # Player has 2 buried rings

        moves = MockBatchMoves(batch_size=1, total_moves=1, device=device)
        moves.from_y[0] = 3
        moves.from_x[0] = 3
        moves.to_y[0] = 4
        moves.to_x[0] = 3
        moves.moves_per_game[0] = 1

        selected_local_idx = torch.tensor([0], dtype=torch.int64, device=device)
        active_mask = torch.tensor([True], device=device)

        apply_recovery_moves_vectorized(state, selected_local_idx, moves, active_mask)

        # Buried ring count decremented
        assert state.buried_rings[0, 1].item() == 1
        # Self-elimination tracked
        assert state.eliminated_rings[0, 1].item() == 1
        assert state.rings_caused_eliminated[0, 1].item() == 1

    def test_recovery_records_history(self):
        """Test that recovery is recorded in history."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.place_marker(0, 3, 3, owner=1)

        moves = MockBatchMoves(batch_size=1, total_moves=1, device=device)
        moves.from_y[0] = 3
        moves.from_x[0] = 3
        moves.to_y[0] = 4
        moves.to_x[0] = 3
        moves.moves_per_game[0] = 1

        selected_local_idx = torch.tensor([0], dtype=torch.int64, device=device)
        active_mask = torch.tensor([True], device=device)

        apply_recovery_moves_vectorized(state, selected_local_idx, moves, active_mask)

        assert state.move_history[0, 0, 0].item() == MoveType.RECOVERY_SLIDE
        assert state.move_history[0, 0, 1].item() == 1
        assert state.move_history[0, 0, 2].item() == 3  # from_y
        assert state.move_history[0, 0, 3].item() == 3  # from_x
        assert state.move_history[0, 0, 4].item() == 4  # to_y
        assert state.move_history[0, 0, 5].item() == 3  # to_x

    def test_recovery_empty_mask(self):
        """Test recovery with empty active mask."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.place_marker(0, 3, 3, owner=1)

        moves = MockBatchMoves(batch_size=1, total_moves=1, device=device)
        moves.from_y[0] = 3
        moves.from_x[0] = 3
        moves.to_y[0] = 4
        moves.to_x[0] = 3
        moves.moves_per_game[0] = 1

        selected_local_idx = torch.tensor([0], dtype=torch.int64, device=device)
        active_mask = torch.tensor([False], device=device)

        apply_recovery_moves_vectorized(state, selected_local_idx, moves, active_mask)

        # Nothing should change
        assert state.marker_owner[0, 3, 3].item() == 1
        assert state.move_count[0].item() == 0


# =============================================================================
# Test multi-game batch processing
# =============================================================================


class TestMultiGameBatchProcessing:
    """Tests for processing multiple games in parallel."""

    def test_placement_multiple_games(self):
        """Test placement across multiple games."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=3, board_size=7, device=device)
        state.current_player[0] = 1
        state.current_player[1] = 2
        state.current_player[2] = 1

        moves = MockBatchMoves(batch_size=3, total_moves=3, device=device)
        moves.from_y[:] = torch.tensor([1, 2, 3])
        moves.from_x[:] = torch.tensor([1, 2, 3])
        moves.to_y[:] = torch.tensor([1, 2, 3])
        moves.to_x[:] = torch.tensor([1, 2, 3])
        moves.move_type[:] = MoveType.PLACEMENT
        moves.moves_per_game[:] = 1
        moves.move_offsets[:] = torch.tensor([0, 1, 2])

        move_indices = torch.tensor([0, 0, 0], dtype=torch.int64, device=device)
        apply_placement_moves_batch(state, move_indices, moves)

        assert state.stack_owner[0, 1, 1].item() == 1
        assert state.stack_owner[1, 2, 2].item() == 2
        assert state.stack_owner[2, 3, 3].item() == 1
        assert state.move_count.tolist() == [1, 1, 1]

    def test_mixed_active_games(self):
        """Test that finished games are properly skipped."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=4, board_size=7, device=device)
        state.current_player[:] = 1
        state.game_over[1] = True
        state.game_over[3] = True

        moves = MockBatchMoves(batch_size=4, total_moves=4, device=device)
        moves.from_y[:] = torch.tensor([2, 2, 2, 2])
        moves.from_x[:] = torch.tensor([2, 2, 2, 2])
        moves.to_y[:] = torch.tensor([2, 2, 2, 2])
        moves.to_x[:] = torch.tensor([2, 2, 2, 2])
        moves.move_type[:] = MoveType.PLACEMENT
        moves.moves_per_game[:] = 1
        moves.move_offsets[:] = torch.tensor([0, 1, 2, 3])

        move_indices = torch.tensor([0, 0, 0, 0], dtype=torch.int64, device=device)
        apply_placement_moves_batch(state, move_indices, moves)

        assert state.stack_height[0, 2, 2].item() == 1
        assert state.stack_height[1, 2, 2].item() == 0  # Game over
        assert state.stack_height[2, 2, 2].item() == 1
        assert state.stack_height[3, 2, 2].item() == 0  # Game over


# =============================================================================
# Test edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in move application."""

    def test_invalid_move_index_skipped(self):
        """Test that invalid move indices are skipped."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1

        moves = MockBatchMoves(batch_size=1, total_moves=1, device=device)
        moves.from_y[0] = 3
        moves.from_x[0] = 3
        moves.moves_per_game[0] = 1

        # Index out of range
        move_indices = torch.tensor([5], dtype=torch.int64, device=device)
        apply_placement_moves_batch(state, move_indices, moves)

        assert state.stack_height[0, 3, 3].item() == 0
        assert state.move_count[0].item() == 0

    def test_zero_moves_per_game(self):
        """Test that games with zero moves are skipped."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)

        moves = MockBatchMoves(batch_size=1, total_moves=0, device=device)
        moves.moves_per_game[0] = 0

        move_indices = torch.tensor([0], dtype=torch.int64, device=device)
        apply_placement_moves_batch(state, move_indices, moves)

        assert state.move_count[0].item() == 0

    def test_history_overflow_handled(self):
        """Test that move history overflow is handled gracefully."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device, max_history_moves=2)
        state.current_player[0] = 1
        state.move_count[0] = 2  # Already at max

        moves = create_single_move(device, 1, 0, 3, 3, 3, 3, MoveType.PLACEMENT)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)

        apply_placement_moves_batch(state, move_indices, moves)

        # Move should still be applied
        assert state.stack_height[0, 3, 3].item() == 1
        assert state.move_count[0].item() == 3
        # History at index 2 should be unchanged (overflow)


class TestMoveApplicationIntegration:
    """Integration tests for move application module."""

    def test_full_turn_sequence(self):
        """Test a full turn: placement -> movement."""
        device = torch.device('cpu')
        state = MockBatchGameState(batch_size=1, board_size=7, device=device)
        state.current_player[0] = 1
        state.rings_in_hand[0, 1] = 10

        # Placement
        placement_moves = create_single_move(device, 1, 0, 3, 3, 3, 3, MoveType.PLACEMENT)
        move_indices = torch.tensor([0], dtype=torch.int64, device=device)
        apply_placement_moves_batch(state, move_indices, placement_moves)

        assert state.stack_height[0, 3, 3].item() == 1
        assert state.must_move_from_y[0].item() == 3
        assert state.must_move_from_x[0].item() == 3

        # Movement (from the just-placed stack)
        movement_moves = create_single_move(device, 1, 0, 3, 3, 5, 5, MoveType.MOVEMENT)
        apply_movement_moves_batch(state, move_indices, movement_moves)

        assert state.stack_height[0, 3, 3].item() == 0
        assert state.stack_height[0, 5, 5].item() == 1
        assert state.move_count[0].item() == 2

    def test_module_exports(self):
        """Test that all expected functions are exported."""
        from app.ai.gpu_move_application import (
            # Legacy functions removed - only current implementations tested
            apply_capture_moves_batch,
            apply_capture_moves_batch_vectorized,
            apply_capture_moves_vectorized,
            apply_movement_moves_batch,
            apply_movement_moves_batch_vectorized,
            apply_movement_moves_vectorized,
            apply_no_action_moves_batch,
            apply_placement_moves_batch,
            apply_placement_moves_batch_vectorized,
            apply_recovery_moves_vectorized,
        )

        # All should be callable
        assert callable(apply_capture_moves_vectorized)
        assert callable(apply_movement_moves_vectorized)
        assert callable(apply_recovery_moves_vectorized)
        assert callable(apply_no_action_moves_batch)
        assert callable(apply_placement_moves_batch_vectorized)
        assert callable(apply_placement_moves_batch)
        assert callable(apply_movement_moves_batch_vectorized)
        assert callable(apply_movement_moves_batch)
        assert callable(apply_capture_moves_batch_vectorized)
        assert callable(apply_capture_moves_batch)
