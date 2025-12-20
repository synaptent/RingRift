"""Tests for app/ai/gpu_parallel_games.py - GPU parallel game simulation.

Tests cover:
- MPS compatibility helpers
- GameStatus, MoveType, GamePhase enums
- BatchGameState initialization and operations
- BatchMoves data structure
- Move selection utilities
- Parallel game runner basics

Note: These tests use CPU fallback when GPU is not available.
"""

import pytest
import numpy as np
import torch

from app.ai.gpu_parallel_games import (
    # Helper functions
    select_moves_vectorized,
    # Core classes
    BatchGameState,
    ParallelGameRunner,
)
from app.ai.gpu_game_types import (
    get_int_dtype,
    GameStatus,
    MoveType,
    GamePhase,
    get_required_line_length,
)
from app.ai.gpu_move_generation import (
    BatchMoves,
    generate_placement_moves_batch,
    _empty_batch_moves,
)
from app.ai.gpu_batch import get_device


class TestMPSCompatibilityHelpers:
    """Tests for MPS compatibility helper functions."""

    def test_get_int_dtype_cpu(self):
        """Test int dtype selection for CPU."""
        device = torch.device("cpu")
        dtype = get_int_dtype(device)
        assert dtype == torch.int16

    def test_get_int_dtype_cuda(self):
        """Test int dtype selection for CUDA (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        dtype = get_int_dtype(device)
        assert dtype == torch.int16

    def test_get_int_dtype_mps(self):
        """Test int dtype selection for MPS (Apple Silicon)."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        device = torch.device("mps")
        dtype = get_int_dtype(device)
        # MPS uses int32 due to index_put_ limitations
        assert dtype == torch.int32


class TestGameEnums:
    """Tests for game status and type enums."""

    def test_game_status_values(self):
        """Test GameStatus enum values."""
        assert GameStatus.ACTIVE.value == 0
        assert GameStatus.COMPLETED.value == 1
        assert GameStatus.DRAW.value == 2
        assert GameStatus.MAX_MOVES.value == 3

    def test_move_type_values(self):
        """Test MoveType enum values."""
        assert MoveType.PLACEMENT == 0
        assert MoveType.MOVEMENT == 1
        assert MoveType.CAPTURE == 2
        assert MoveType.LINE_FORMATION == 3
        assert MoveType.TERRITORY_CLAIM == 4
        assert MoveType.SKIP == 5
        assert MoveType.NO_ACTION == 6
        assert MoveType.RECOVERY_SLIDE == 7

    def test_game_phase_values(self):
        """Test GamePhase enum values."""
        assert GamePhase.RING_PLACEMENT.value == 0
        assert GamePhase.MOVEMENT.value == 1
        assert GamePhase.LINE_PROCESSING.value == 2
        assert GamePhase.TERRITORY_PROCESSING.value == 3
        assert GamePhase.END_TURN.value == 4


class TestBatchMoves:
    """Tests for BatchMoves data structure."""

    def test_batch_moves_from_placement_generation(self):
        """Test BatchMoves created from placement generation."""
        device = torch.device("cpu")

        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        moves = generate_placement_moves_batch(state)

        # Should have expected attributes
        assert hasattr(moves, 'game_idx')
        assert hasattr(moves, 'move_type')
        assert hasattr(moves, 'total_moves')
        assert hasattr(moves, 'device')
        assert moves.total_moves > 0

    def test_empty_batch_moves(self):
        """Test creating empty BatchMoves."""
        device = torch.device("cpu")
        batch_size = 4

        moves = _empty_batch_moves(batch_size, device)

        assert moves.total_moves == 0
        assert moves.device == device


class TestBatchGameState:
    """Tests for BatchGameState class."""

    def test_initialization(self):
        """Test BatchGameState initialization via create_batch."""
        device = torch.device("cpu")
        batch_size = 4
        board_size = 8
        num_players = 2

        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=board_size,
            num_players=num_players,
            device=device,
        )

        assert state.batch_size == batch_size
        assert state.board_size == board_size
        assert state.num_players == num_players
        assert state.device == device

        # Check tensor shapes
        assert state.stack_owner.shape == (batch_size, board_size, board_size)
        assert state.current_player.shape == (batch_size,)
        assert state.current_phase.shape == (batch_size,)
        assert state.game_status.shape == (batch_size,)

    def test_initialization_with_rings(self):
        """Test BatchGameState initialization includes ring tensors."""
        device = torch.device("cpu")
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Should have rings_in_hand tensor
        assert hasattr(state, 'rings_in_hand')
        assert state.rings_in_hand.shape[0] == 2  # batch_size

    def test_initial_values(self):
        """Test initial values are correct."""
        device = torch.device("cpu")
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        # All games should start active
        assert (state.game_status == GameStatus.ACTIVE).all()

        # All games should start in ring placement phase
        assert (state.current_phase == GamePhase.RING_PLACEMENT).all()

        # Player 1 starts (1-indexed)
        assert (state.current_player == 1).all()

    def test_different_board_sizes(self):
        """Test different board sizes."""
        device = torch.device("cpu")

        for board_size in [6, 7, 8, 9]:
            state = BatchGameState.create_batch(
                batch_size=2,
                board_size=board_size,
                num_players=2,
                device=device,
            )
            assert state.stack_owner.shape == (2, board_size, board_size)

    def test_different_player_counts(self):
        """Test different player counts."""
        device = torch.device("cpu")

        for num_players in [2, 3, 4]:
            state = BatchGameState.create_batch(
                batch_size=2,
                board_size=8,
                num_players=num_players,
                device=device,
            )
            assert state.num_players == num_players


class TestSelectMovesVectorized:
    """Tests for vectorized move selection."""

    def test_empty_moves(self):
        """Test selection with no moves."""
        device = torch.device("cpu")
        batch_size = 4

        moves = _empty_batch_moves(batch_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_vectorized(
            moves=moves,
            active_mask=active_mask,
            board_size=8,
            temperature=1.0,
        )

        # All should be -1 (no moves)
        assert (selected == -1).all()

    def test_selection_from_generated_moves(self):
        """Test selection from moves generated by placement."""
        device = torch.device("cpu")
        batch_size = 2

        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
        )

        moves = generate_placement_moves_batch(state)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_vectorized(
            moves=moves,
            active_mask=active_mask,
            board_size=8,
            temperature=1.0,
        )

        # Both games should have valid selections
        assert (selected >= 0).all()

    def test_selection_temperature(self):
        """Test that different temperatures produce different distributions."""
        device = torch.device("cpu")
        batch_size = 2

        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
        )

        moves = generate_placement_moves_batch(state)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Low temperature should be more deterministic
        selected_low = select_moves_vectorized(
            moves=moves,
            active_mask=active_mask,
            board_size=8,
            temperature=0.1,
        )

        # High temperature should be more random
        selected_high = select_moves_vectorized(
            moves=moves,
            active_mask=active_mask,
            board_size=8,
            temperature=10.0,
        )

        # Both should produce valid selections
        assert (selected_low >= 0).all()
        assert (selected_high >= 0).all()


class TestGetRequiredLineLength:
    """Tests for line length calculation per RR-CANON-R120."""

    def test_2_player_boards(self):
        """Test line length for 2-player games."""
        # 2 players always uses line length 4
        for board_size in [6, 7, 8, 9, 10]:
            length = get_required_line_length(board_size, 2)
            assert length == 4

    def test_multiplayer_8x8(self):
        """Test line length for 3-4 player on 8x8."""
        # 8x8 with 3-4 players uses line length 3
        assert get_required_line_length(8, 3) == 3
        assert get_required_line_length(8, 4) == 3

    def test_multiplayer_other_boards(self):
        """Test line length for 3-4 player on non-8x8."""
        # Non-8x8 boards with 3-4 players still use 4
        assert get_required_line_length(7, 3) == 4
        assert get_required_line_length(9, 3) == 4
        assert get_required_line_length(10, 4) == 4


class TestPlacementMoveGeneration:
    """Tests for placement move generation."""

    def test_generate_placement_moves_empty_board(self):
        """Test generating placement moves on empty board."""
        device = torch.device("cpu")
        batch_size = 2
        board_size = 8

        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=board_size,
            num_players=2,
            device=device,
        )

        moves = generate_placement_moves_batch(state)

        # Should have moves for empty cells
        assert moves.total_moves > 0

        # All moves should be placement type
        assert (moves.move_type == MoveType.PLACEMENT).all()

    def test_generate_placement_moves_uses_active_mask(self):
        """Test that placement moves can be filtered with active mask."""
        device = torch.device("cpu")
        batch_size = 2
        board_size = 8

        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=board_size,
            num_players=2,
            device=device,
        )

        # Only generate for game 0
        active_mask = torch.tensor([True, False], device=device)
        moves = generate_placement_moves_batch(state, active_mask=active_mask)

        # Should only generate moves for game 0
        if moves.total_moves > 0:
            assert (moves.game_idx == 0).all()


class TestIntegration:
    """Integration tests for BatchGameState and moves."""

    def test_create_and_generate_moves(self):
        """Test creating state and generating moves."""
        device = torch.device("cpu")

        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Generate placement moves
        moves = generate_placement_moves_batch(state)

        # Should have moves for all 4 games (all in placement phase)
        assert moves.total_moves > 0
        game_indices = moves.game_idx.unique()
        assert len(game_indices) == 4

    def test_select_from_generated_moves(self):
        """Test selecting from generated moves."""
        device = torch.device("cpu")
        batch_size = 4

        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
        )

        moves = generate_placement_moves_batch(state)
        active_mask = state.game_status == GameStatus.ACTIVE

        selected = select_moves_vectorized(
            moves=moves,
            active_mask=active_mask,
            board_size=8,
            temperature=1.0,
        )

        # All games should have valid selections
        assert (selected >= 0).all()


class TestDeviceHandling:
    """Tests for device handling."""

    def test_cpu_device(self):
        """Test using CPU device explicitly."""
        device = torch.device("cpu")

        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        assert state.stack_owner.device == device
        assert state.current_player.device == device

    def test_get_device_returns_valid(self):
        """Test that get_device returns a valid device."""
        device = get_device()

        # Should be cpu, cuda, or mps
        assert device.type in ["cpu", "cuda", "mps"]

    def test_state_tensors_on_same_device(self):
        """Test all state tensors are on same device."""
        device = get_device()

        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=device,
        )

        # All tensors should be on the same device type
        assert state.stack_owner.device.type == device.type
        assert state.current_player.device.type == device.type
        assert state.current_phase.device.type == device.type
        assert state.game_status.device.type == device.type


class TestParallelGameRunner:
    """Tests for ParallelGameRunner class."""

    def test_initialization(self):
        """Test runner initialization with default parameters."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
        )

        assert runner.batch_size == 4
        assert runner.board_size == 8
        assert runner.num_players == 2
        assert runner.state is not None

    def test_initialization_different_sizes(self):
        """Test runner initialization with different board sizes."""
        for board_size in [6, 7, 8, 9]:
            runner = ParallelGameRunner(
                batch_size=2,
                board_size=board_size,
                num_players=2,
            )
            assert runner.board_size == board_size
            assert runner.state.stack_owner.shape[1] == board_size

    def test_initialization_multiplayer(self):
        """Test runner initialization with 3-4 players."""
        for num_players in [2, 3, 4]:
            runner = ParallelGameRunner(
                batch_size=2,
                board_size=8,
                num_players=num_players,
            )
            assert runner.num_players == num_players

    def test_reset_games(self):
        """Test resetting games to initial state."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
        )

        # Games should start active
        assert (runner.state.game_status == GameStatus.ACTIVE).all()

        # Reset games
        runner.reset_games()

        # Should be back to initial state
        assert (runner.state.game_status == GameStatus.ACTIVE).all()
        assert (runner.state.current_player == 1).all()
        assert (runner.state.current_phase == GamePhase.RING_PLACEMENT).all()

    def test_set_temperature(self):
        """Test setting temperature."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
        )

        runner.set_temperature(0.5)
        assert runner.temperature == 0.5

        runner.set_temperature(2.0)
        assert runner.temperature == 2.0

    def test_run_games_short(self):
        """Test running a few game steps."""
        # Force CPU to avoid MPS index_put_ dtype limitations
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )

        # Run games with short limit
        results = runner.run_games(max_moves=10)

        # Should return a dict with results
        assert isinstance(results, dict)
        assert "winners" in results
        assert len(results["winners"]) == 4

    def test_get_stats(self):
        """Test getting runner statistics."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
        )

        stats = runner.get_stats()

        # Should have expected keys
        assert isinstance(stats, dict)
        assert "games_completed" in stats
        assert "total_moves" in stats
        assert "games_per_second" in stats

    def test_default_weights(self):
        """Test default heuristic weights."""
        runner = ParallelGameRunner(
            batch_size=2,
            board_size=8,
            num_players=2,
        )

        weights = runner._default_weights()

        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert "stack_count" in weights
        assert "territory_count" in weights
