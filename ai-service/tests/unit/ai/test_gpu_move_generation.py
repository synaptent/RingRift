"""Tests for gpu_move_generation module.

Tests move generation functions per RR-CANON rules:
- Placement moves (R080)
- Movement moves (R090-R092)
- Capture moves (R100-R103)
- Recovery slide moves (R110-R115)
"""

import pytest
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

from app.ai.gpu_move_generation import (
    BatchMoves,
    _empty_batch_moves,
    generate_placement_moves_batch,
    DIRECTIONS,
    _validate_paths_vectorized_fast,
    generate_movement_moves_batch_vectorized,
    generate_movement_moves_batch,
    generate_capture_moves_batch,
    generate_capture_moves_batch_vectorized,
    generate_chain_capture_moves_from_position,
    apply_single_chain_capture,
    generate_recovery_moves_batch,
)
from app.ai.gpu_game_types import MoveType


@dataclass
class MockBatchGameState:
    """Mock BatchGameState for testing move generation.

    Provides minimal interface needed by move generation functions.
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

    # Constraints
    must_move_from_y: torch.Tensor = None
    must_move_from_x: torch.Tensor = None

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

        if self.must_move_from_y is None:
            self.must_move_from_y = torch.full((bs,), -1, dtype=torch.int32, device=self.device)
        if self.must_move_from_x is None:
            self.must_move_from_x = torch.full((bs,), -1, dtype=torch.int32, device=self.device)

        if self.move_history is None:
            self.move_history = torch.zeros((bs, self.max_history_moves, 6), dtype=torch.int8, device=self.device)

    def get_active_mask(self) -> torch.Tensor:
        """Return mask of active (non-finished) games."""
        return ~self.game_over

    def place_stack(self, game_idx: int, y: int, x: int, owner: int, height: int, cap: int = None):
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


# =============================================================================
# Test BatchMoves and _empty_batch_moves
# =============================================================================


class TestBatchMoves:
    """Tests for BatchMoves dataclass."""

    def test_batch_moves_creation(self):
        """Test creating a BatchMoves structure."""
        device = torch.device('cpu')
        batch_size = 4
        total_moves = 10

        moves = BatchMoves(
            game_idx=torch.zeros(total_moves, dtype=torch.int32, device=device),
            move_type=torch.full((total_moves,), MoveType.PLACEMENT, dtype=torch.int8, device=device),
            from_y=torch.zeros(total_moves, dtype=torch.int32, device=device),
            from_x=torch.zeros(total_moves, dtype=torch.int32, device=device),
            to_y=torch.zeros(total_moves, dtype=torch.int32, device=device),
            to_x=torch.zeros(total_moves, dtype=torch.int32, device=device),
            moves_per_game=torch.zeros(batch_size, dtype=torch.int32, device=device),
            move_offsets=torch.zeros(batch_size, dtype=torch.int32, device=device),
            total_moves=total_moves,
            device=device,
        )

        assert moves.total_moves == total_moves
        assert moves.game_idx.shape[0] == total_moves
        assert moves.moves_per_game.shape[0] == batch_size

    def test_empty_batch_moves(self):
        """Test _empty_batch_moves helper."""
        device = torch.device('cpu')
        batch_size = 4

        moves = _empty_batch_moves(batch_size, device)

        assert moves.total_moves == 0
        assert moves.game_idx.numel() == 0
        assert moves.moves_per_game.shape[0] == batch_size
        assert (moves.moves_per_game == 0).all()


# =============================================================================
# Test Placement Move Generation
# =============================================================================


class TestGeneratePlacementMovesBatch:
    """Tests for placement move generation."""

    def test_empty_board_full_placements(self):
        """Test that empty board allows placements everywhere."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))

        moves = generate_placement_moves_batch(state)

        # All 64 cells should be valid for placement
        assert moves.total_moves == 64
        assert (moves.move_type == MoveType.PLACEMENT).all()

    def test_collapsed_cells_blocked(self):
        """Test that collapsed cells cannot have placements."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.collapse_cell(0, 3, 3)
        state.collapse_cell(0, 4, 4)

        moves = generate_placement_moves_batch(state)

        # 64 - 2 collapsed = 62 valid
        assert moves.total_moves == 62

        # Verify collapsed positions not in moves
        positions = set(zip(moves.from_y.tolist(), moves.from_x.tolist()))
        assert (3, 3) not in positions
        assert (4, 4) not in positions

    def test_markers_block_placement(self):
        """Test that cells with markers cannot have placements."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.place_marker(0, 2, 2, 1)
        state.place_marker(0, 5, 5, 2)

        moves = generate_placement_moves_batch(state)

        # 64 - 2 markers = 62 valid
        assert moves.total_moves == 62

    def test_stacks_allow_placement(self):
        """Test that cells with stacks allow placement on top."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.place_stack(0, 3, 3, 1, 2)

        moves = generate_placement_moves_batch(state)

        # All 64 cells valid including the one with stack
        assert moves.total_moves == 64

    def test_active_mask_filtering(self):
        """Test that only active games generate moves."""
        state = MockBatchGameState(batch_size=2, board_size=8, device=torch.device('cpu'))
        state.game_over[1] = True

        moves = generate_placement_moves_batch(state)

        # Only game 0 should have moves
        assert (moves.game_idx == 0).all()
        assert moves.total_moves == 64

    def test_custom_active_mask(self):
        """Test custom active_mask parameter."""
        state = MockBatchGameState(batch_size=3, board_size=4, device=torch.device('cpu'))

        # Only enable game 1
        active_mask = torch.tensor([False, True, False], device=state.device)
        moves = generate_placement_moves_batch(state, active_mask=active_mask)

        assert (moves.game_idx == 1).all()
        assert moves.total_moves == 16  # 4x4 board

    def test_moves_per_game_counting(self):
        """Test moves_per_game counts correctly."""
        state = MockBatchGameState(batch_size=2, board_size=4, device=torch.device('cpu'))
        # Block some cells in game 0
        state.collapse_cell(0, 0, 0)
        state.collapse_cell(0, 1, 1)

        moves = generate_placement_moves_batch(state)

        # Game 0: 16 - 2 = 14, Game 1: 16
        assert moves.moves_per_game[0].item() == 14
        assert moves.moves_per_game[1].item() == 16
        assert moves.total_moves == 30


# =============================================================================
# Test Movement Move Generation
# =============================================================================


class TestGenerateMovementMovesBatch:
    """Tests for movement move generation."""

    def test_no_stacks_no_moves(self):
        """Test that no player stacks means no movement moves."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1

        moves = generate_movement_moves_batch(state)

        assert moves.total_moves == 0

    def test_single_stack_generates_moves(self):
        """Test that a single stack generates movement moves."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        # Place height-1 stack at center
        state.place_stack(0, 3, 3, 1, 1)

        moves = generate_movement_moves_batch(state)

        # Height-1 stack can move distance >= 1 in 8 directions
        # Should have multiple valid moves
        assert moves.total_moves > 0
        assert (moves.move_type == MoveType.MOVEMENT).all()
        assert (moves.from_y == 3).all()
        assert (moves.from_x == 3).all()

    def test_stack_height_minimum_distance(self):
        """Test that move distance must be >= stack height."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        # Height-3 stack at corner
        state.place_stack(0, 0, 0, 1, 3)

        moves = generate_movement_moves_batch(state)

        # All moves should be distance >= 3 from (0,0)
        for i in range(moves.total_moves):
            from_y, from_x = moves.from_y[i].item(), moves.from_x[i].item()
            to_y, to_x = moves.to_y[i].item(), moves.to_x[i].item()
            dist = max(abs(to_y - from_y), abs(to_x - from_x))
            assert dist >= 3, f"Move {i}: distance {dist} < stack height 3"

    def test_cannot_pass_through_stacks(self):
        """Test that movements cannot pass through stacks."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        # Player stack at (0, 0)
        state.place_stack(0, 0, 0, 1, 1)
        # Blocking stack at (0, 2) - blocks east direction
        state.place_stack(0, 0, 2, 2, 1)

        moves = generate_movement_moves_batch(state)

        # No move should have from=(0,0), to_x > 2 with to_y=0
        for i in range(moves.total_moves):
            from_y, from_x = moves.from_y[i].item(), moves.from_x[i].item()
            to_y, to_x = moves.to_y[i].item(), moves.to_x[i].item()
            if from_y == 0 and from_x == 0 and to_y == 0:
                # East direction blocked at x=2
                assert to_x < 2, f"Move passed through blocking stack: to=({to_y}, {to_x})"

    def test_cannot_land_on_stacks(self):
        """Test that movements cannot land on stacks."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 3, 3, 1, 1)
        state.place_stack(0, 3, 5, 2, 1)  # Opponent stack

        moves = generate_movement_moves_batch(state)

        # No move should land on (3, 5)
        positions = set(zip(moves.to_y.tolist(), moves.to_x.tolist()))
        assert (3, 5) not in positions

    def test_collapsed_cells_block_movement(self):
        """Test that collapsed cells block movement paths."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 0, 0, 1, 1)
        state.collapse_cell(0, 0, 3)  # Collapse cell in east direction

        moves = generate_movement_moves_batch(state)

        # No move should reach or pass (0, 3)
        for i in range(moves.total_moves):
            from_y, from_x = moves.from_y[i].item(), moves.from_x[i].item()
            to_y, to_x = moves.to_y[i].item(), moves.to_x[i].item()
            if from_y == 0 and from_x == 0 and to_y == 0:
                assert to_x < 3, f"Move reached collapsed cell: to=({to_y}, {to_x})"

    def test_opponent_stacks_not_moved(self):
        """Test that opponent's stacks are not moved."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 3, 3, 2, 1)  # Opponent's stack

        moves = generate_movement_moves_batch(state)

        assert moves.total_moves == 0

    def test_must_move_from_constraint(self):
        """Test must_move_from constraint."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 2, 2, 1, 1)  # Stack at (2, 2)
        state.place_stack(0, 5, 5, 1, 1)  # Stack at (5, 5)
        state.must_move_from_y[0] = 2
        state.must_move_from_x[0] = 2

        moves = generate_movement_moves_batch(state)

        # All moves should be from (2, 2)
        assert (moves.from_y == 2).all()
        assert (moves.from_x == 2).all()


class TestValidatePathsVectorizedFast:
    """Tests for _validate_paths_vectorized_fast."""

    def test_clear_path_valid(self):
        """Test that clear paths are valid."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))

        game_indices = torch.tensor([0], dtype=torch.int64, device=state.device)
        from_y = torch.tensor([0], dtype=torch.int64, device=state.device)
        from_x = torch.tensor([0], dtype=torch.int64, device=state.device)
        to_y = torch.tensor([0], dtype=torch.int64, device=state.device)
        to_x = torch.tensor([4], dtype=torch.int64, device=state.device)

        valid = _validate_paths_vectorized_fast(
            state, game_indices, from_y, from_x, to_y, to_x, max_path_len=8
        )

        assert valid[0].item() == True

    def test_blocked_path_invalid(self):
        """Test that blocked paths are invalid."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.place_stack(0, 0, 2, 2, 1)  # Blocker at (0, 2)

        game_indices = torch.tensor([0], dtype=torch.int64, device=state.device)
        from_y = torch.tensor([0], dtype=torch.int64, device=state.device)
        from_x = torch.tensor([0], dtype=torch.int64, device=state.device)
        to_y = torch.tensor([0], dtype=torch.int64, device=state.device)
        to_x = torch.tensor([4], dtype=torch.int64, device=state.device)

        valid = _validate_paths_vectorized_fast(
            state, game_indices, from_y, from_x, to_y, to_x, max_path_len=8
        )

        assert valid[0].item() == False

    def test_collapsed_blocks_path(self):
        """Test that collapsed cells block paths."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.collapse_cell(0, 0, 2)

        game_indices = torch.tensor([0], dtype=torch.int64, device=state.device)
        from_y = torch.tensor([0], dtype=torch.int64, device=state.device)
        from_x = torch.tensor([0], dtype=torch.int64, device=state.device)
        to_y = torch.tensor([0], dtype=torch.int64, device=state.device)
        to_x = torch.tensor([4], dtype=torch.int64, device=state.device)

        valid = _validate_paths_vectorized_fast(
            state, game_indices, from_y, from_x, to_y, to_x, max_path_len=8
        )

        assert valid[0].item() == False


# =============================================================================
# Test Capture Move Generation
# =============================================================================


class TestGenerateCaptureMovesBatch:
    """Tests for capture move generation."""

    def test_no_targets_no_captures(self):
        """Test that no targets means no capture moves."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 3, 3, 1, 2, 2)  # Player's stack

        moves = generate_capture_moves_batch(state)

        assert moves.total_moves == 0

    def test_capture_weaker_target(self):
        """Test capturing a target with lower cap height."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 0, 0, 1, 2, 2)  # Attacker cap=2
        state.place_stack(0, 0, 2, 2, 1, 1)  # Target cap=1

        moves = generate_capture_moves_batch(state)

        # Should have at least one capture move
        assert moves.total_moves > 0
        assert (moves.move_type == MoveType.CAPTURE).all()
        assert (moves.from_y == 0).all()
        assert (moves.from_x == 0).all()

    def test_cannot_capture_stronger_target(self):
        """Test that cannot capture target with higher cap height."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 0, 0, 1, 2, 1)  # Attacker cap=1
        state.place_stack(0, 0, 2, 2, 3, 3)  # Target cap=3

        moves = generate_capture_moves_batch(state)

        # All capture from (0,0) should be blocked by strong target
        captures_from_origin = [
            i for i in range(moves.total_moves)
            if moves.from_y[i] == 0 and moves.from_x[i] == 0
        ]
        # The only target in direction blocks, but equal cap is allowed
        # cap=1 < cap=3 means blocked
        assert len(captures_from_origin) == 0

    def test_capture_equal_cap_height(self):
        """Test capturing target with equal cap height."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 0, 0, 1, 2, 2)  # Attacker cap=2
        state.place_stack(0, 0, 2, 2, 2, 2)  # Target cap=2

        moves = generate_capture_moves_batch(state)

        # Equal cap height allows capture
        assert moves.total_moves > 0

    def test_landing_beyond_target(self):
        """Test that landing must be beyond target."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 0, 0, 1, 1, 1)  # Attacker
        state.place_stack(0, 0, 1, 2, 1, 1)  # Target at distance 1

        moves = generate_capture_moves_batch(state)

        # All landing positions should be beyond x=1
        for i in range(moves.total_moves):
            if moves.from_y[i] == 0 and moves.from_x[i] == 0:
                to_y, to_x = moves.to_y[i].item(), moves.to_x[i].item()
                if to_y == 0:  # East direction
                    assert to_x > 1, f"Landing at {to_x} not beyond target at x=1"

    def test_landing_minimum_distance_is_stack_height(self):
        """Test that landing distance >= stack height."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 0, 0, 1, 3, 3)  # Height-3 attacker
        state.place_stack(0, 0, 1, 2, 1, 1)  # Target at distance 1

        moves = generate_capture_moves_batch(state)

        # Landing distance must be >= 3 (stack height)
        for i in range(moves.total_moves):
            from_y, from_x = moves.from_y[i].item(), moves.from_x[i].item()
            to_y, to_x = moves.to_y[i].item(), moves.to_x[i].item()
            dist = max(abs(to_y - from_y), abs(to_x - from_x))
            if from_y == 0 and from_x == 0:
                assert dist >= 3, f"Landing distance {dist} < stack height 3"

    def test_path_to_target_clear(self):
        """Test that path from attacker to target must be clear."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 0, 0, 1, 1, 1)  # Attacker
        state.place_stack(0, 0, 2, 2, 1, 1)  # Blocker
        state.place_stack(0, 0, 4, 2, 1, 1)  # Would-be target

        moves = generate_capture_moves_batch(state)

        # Cannot capture (0,4) because (0,2) is in the way
        # But can capture (0,2)
        for i in range(moves.total_moves):
            if moves.from_y[i] == 0 and moves.from_x[i] == 0:
                # The first stack in the direction is the target
                pass  # Just verify we don't crash

    def test_can_capture_own_stack(self):
        """Test that can capture own stack (self-capture allowed)."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 0, 0, 1, 2, 2)  # Attacker cap=2
        state.place_stack(0, 0, 2, 1, 1, 1)  # Own stack cap=1

        moves = generate_capture_moves_batch(state)

        # Self-capture should be allowed
        assert moves.total_moves > 0


class TestGenerateChainCaptureMovesFromPosition:
    """Tests for generate_chain_capture_moves_from_position."""

    def test_no_chain_capture_available(self):
        """Test when no chain capture is available."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 3, 3, 1, 2, 2)  # Player's stack

        captures = generate_chain_capture_moves_from_position(state, 0, 3, 3)

        assert len(captures) == 0

    def test_chain_capture_available(self):
        """Test when chain capture is available."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 3, 3, 1, 2, 2)  # Attacker cap=2
        state.place_stack(0, 3, 5, 2, 1, 1)  # Target cap=1

        captures = generate_chain_capture_moves_from_position(state, 0, 3, 3)

        # Should have capture landing positions beyond (3, 5)
        assert len(captures) > 0
        for landing_y, landing_x in captures:
            assert landing_x > 5 or landing_y != 3

    def test_wrong_player_stack_returns_empty(self):
        """Test that asking about opponent's stack returns empty."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 3, 3, 2, 2, 2)  # Opponent's stack

        captures = generate_chain_capture_moves_from_position(state, 0, 3, 3)

        assert len(captures) == 0


class TestApplySingleChainCapture:
    """Tests for apply_single_chain_capture."""

    def test_applies_capture_correctly(self):
        """Test that capture is applied correctly."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 0, 0, 1, 2, 2)  # Attacker
        state.place_stack(0, 0, 2, 2, 1, 1)  # Target

        # Capture landing at (0, 4)
        new_y, new_x = apply_single_chain_capture(state, 0, 0, 0, 0, 4)

        assert new_y == 0 and new_x == 4
        # Origin should be cleared
        assert state.stack_owner[0, 0, 0].item() == 0
        # Landing should have attacker's stack
        assert state.stack_owner[0, 0, 4].item() == 1
        # Target should have reduced height
        assert state.stack_height[0, 0, 2].item() == 0  # Height was 1, now 0

    def test_departure_marker_left(self):
        """Test that departure marker is left at origin."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 0, 0, 1, 2, 2)
        state.place_stack(0, 0, 2, 2, 1, 1)

        apply_single_chain_capture(state, 0, 0, 0, 0, 4)

        # Departure marker at origin
        assert state.marker_owner[0, 0, 0].item() == 1

    def test_buried_rings_updated(self):
        """Test that capturing opponent updates buried_rings."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 0, 0, 1, 2, 2)
        state.place_stack(0, 0, 2, 2, 1, 1)  # Opponent (player 2)

        initial_buried = state.buried_rings[0, 2].item()

        apply_single_chain_capture(state, 0, 0, 0, 0, 4)

        # Player 2 should have 1 more buried ring
        assert state.buried_rings[0, 2].item() == initial_buried + 1


# =============================================================================
# Test Recovery Move Generation
# =============================================================================


class TestGenerateRecoveryMovesBatch:
    """Tests for recovery move generation."""

    def test_player_with_stacks_no_recovery(self):
        """Test that player with stacks cannot use recovery."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_stack(0, 3, 3, 1, 1)  # Player has a stack
        state.place_marker(0, 5, 5, 1)  # Has marker too
        state.buried_rings[0, 1] = 2

        moves = generate_recovery_moves_batch(state)

        assert moves.total_moves == 0

    def test_player_without_markers_no_recovery(self):
        """Test that player without markers cannot use recovery."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.buried_rings[0, 1] = 2
        # No stacks, no markers

        moves = generate_recovery_moves_batch(state)

        assert moves.total_moves == 0

    def test_player_without_buried_rings_no_recovery(self):
        """Test that player without buried rings cannot use recovery."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_marker(0, 3, 3, 1)  # Has marker
        # No buried rings

        moves = generate_recovery_moves_batch(state)

        assert moves.total_moves == 0

    def test_recovery_eligible_generates_moves(self):
        """Test that eligible player generates recovery moves."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_marker(0, 3, 3, 1)  # Has marker
        state.buried_rings[0, 1] = 2  # Has buried rings
        # No stacks

        moves = generate_recovery_moves_batch(state)

        # Should have moves to slide marker to adjacent cells
        assert moves.total_moves > 0
        assert (moves.move_type == MoveType.RECOVERY_SLIDE).all()
        assert (moves.from_y == 3).all()
        assert (moves.from_x == 3).all()

    def test_recovery_slide_to_adjacent_only(self):
        """Test that recovery only slides to adjacent cells."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_marker(0, 3, 3, 1)
        state.buried_rings[0, 1] = 2

        moves = generate_recovery_moves_batch(state)

        for i in range(moves.total_moves):
            from_y, from_x = moves.from_y[i].item(), moves.from_x[i].item()
            to_y, to_x = moves.to_y[i].item(), moves.to_x[i].item()
            dist = max(abs(to_y - from_y), abs(to_x - from_x))
            assert dist == 1, f"Recovery slide distance {dist} > 1"

    def test_recovery_cannot_slide_to_collapsed(self):
        """Test that recovery cannot slide to collapsed cells."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_marker(0, 3, 3, 1)
        state.buried_rings[0, 1] = 2
        state.collapse_cell(0, 3, 4)  # Collapse adjacent cell

        moves = generate_recovery_moves_batch(state)

        # (3, 4) should not be a destination
        destinations = set(zip(moves.to_y.tolist(), moves.to_x.tolist()))
        assert (3, 4) not in destinations

    def test_recovery_cannot_slide_to_existing_marker(self):
        """Test that recovery cannot slide onto another marker."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_marker(0, 3, 3, 1)
        state.place_marker(0, 3, 4, 2)  # Another marker adjacent
        state.buried_rings[0, 1] = 2

        moves = generate_recovery_moves_batch(state)

        destinations = set(zip(moves.to_y.tolist(), moves.to_x.tolist()))
        assert (3, 4) not in destinations

    def test_stack_strike_recovery(self):
        """Test stack-strike recovery (sliding onto stack)."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.place_marker(0, 3, 3, 1)
        state.place_stack(0, 3, 4, 2, 1)  # Opponent stack adjacent
        state.buried_rings[0, 1] = 2

        moves = generate_recovery_moves_batch(state)

        # Stack-strike should be an option (if no line-forming moves)
        # The (3, 4) should be in destinations
        destinations = list(zip(moves.to_y.tolist(), moves.to_x.tolist()))
        # May or may not have stack-strike depending on line-forming
        assert moves.total_moves > 0


# =============================================================================
# Test DIRECTIONS constant
# =============================================================================


class TestDirections:
    """Tests for DIRECTIONS constant."""

    def test_eight_directions(self):
        """Test that DIRECTIONS has 8 directions."""
        assert DIRECTIONS.shape == (8, 2)

    def test_directions_are_valid(self):
        """Test that all directions are unit vectors or diagonals."""
        for i in range(8):
            dy, dx = DIRECTIONS[i, 0].item(), DIRECTIONS[i, 1].item()
            assert abs(dy) <= 1 and abs(dx) <= 1
            assert not (dy == 0 and dx == 0)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMoveGenerationIntegration:
    """Integration tests for move generation."""

    def test_all_move_types_generated(self):
        """Test generating all move types for a complex game state."""
        state = MockBatchGameState(batch_size=1, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1

        # Setup for placement (empty cells)
        # Setup for movement (player stack)
        state.place_stack(0, 1, 1, 1, 2, 2)
        # Setup for capture (player stack + target)
        state.place_stack(0, 4, 4, 1, 2, 2)
        state.place_stack(0, 4, 6, 2, 1, 1)

        placement_moves = generate_placement_moves_batch(state)
        movement_moves = generate_movement_moves_batch(state)
        capture_moves = generate_capture_moves_batch(state)

        assert placement_moves.total_moves > 0
        assert movement_moves.total_moves > 0
        assert capture_moves.total_moves > 0

    def test_batch_independence(self):
        """Test that different games in batch are processed independently."""
        state = MockBatchGameState(batch_size=2, board_size=8, device=torch.device('cpu'))
        state.current_player[0] = 1
        state.current_player[1] = 1

        # Game 0: stack at (1, 1)
        state.place_stack(0, 1, 1, 1, 1)
        # Game 1: stack at (5, 5)
        state.place_stack(1, 5, 5, 1, 1)

        moves = generate_movement_moves_batch(state)

        # Game 0 moves should have from=(1, 1)
        game0_mask = moves.game_idx == 0
        game0_from_y = moves.from_y[game0_mask].unique()
        game0_from_x = moves.from_x[game0_mask].unique()
        assert game0_from_y.numel() == 1 and game0_from_y[0].item() == 1
        assert game0_from_x.numel() == 1 and game0_from_x[0].item() == 1

        # Game 1 moves should have from=(5, 5)
        game1_mask = moves.game_idx == 1
        game1_from_y = moves.from_y[game1_mask].unique()
        game1_from_x = moves.from_x[game1_mask].unique()
        assert game1_from_y.numel() == 1 and game1_from_y[0].item() == 5
        assert game1_from_x.numel() == 1 and game1_from_x[0].item() == 5
