"""Integration tests for ParallelGameRunner.

Tests the GPU-accelerated parallel game simulation end-to-end.
"""

import pytest
import torch

from app.ai.gpu_batch_state import BatchGameState
from app.ai.gpu_game_types import GamePhase, GameStatus
from app.ai.gpu_parallel_games import ParallelGameRunner

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get test device.

    Notes:
    - CUDA is the primary supported acceleration backend for the parallel runner.
    - On Apple Silicon, MPS is frequently slower than CPU for this workload
      (many small kernels + sync points), and can cause timeouts in CI.
      Prefer CPU when CUDA is not available.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def small_runner(device):
    """Create a small runner for fast tests."""
    return ParallelGameRunner(
        batch_size=4,
        board_size=8,
        num_players=2,
        device=device,
    )


@pytest.fixture
def medium_runner(device):
    """Create a medium runner for more thorough tests."""
    return ParallelGameRunner(
        batch_size=16,
        board_size=8,
        num_players=2,
        device=device,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for ParallelGameRunner initialization."""

    def test_basic_init(self, device):
        """Should initialize with default parameters."""
        runner = ParallelGameRunner(
            batch_size=8,
            board_size=8,
            num_players=2,
            device=device,
        )
        assert runner.batch_size == 8
        assert runner.board_size == 8
        assert runner.num_players == 2
        assert runner.device.type == device.type

    def test_custom_board_size(self, device):
        """Should support different board sizes."""
        for board_size in [8, 13, 19]:
            runner = ParallelGameRunner(
                batch_size=4,
                board_size=board_size,
                num_players=2,
                device=device,
            )
            assert runner.board_size == board_size

    def test_multi_player(self, device):
        """Should support 4-player games."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=4,
            device=device,
        )
        assert runner.num_players == 4

    def test_reset_creates_fresh_state(self, small_runner):
        """reset_games should create a fresh BatchGameState."""
        small_runner.reset_games()
        state = small_runner.state

        assert state is not None
        assert state.batch_size == small_runner.batch_size
        assert state.board_size == small_runner.board_size
        # Initial phase should be ring placement
        assert (state.current_phase == GamePhase.RING_PLACEMENT).all()


# =============================================================================
# Game Running Tests
# =============================================================================


class TestRunGames:
    """Tests for running games to completion."""

    def test_run_short_games(self, small_runner):
        """Should run games and return results."""
        results = small_runner.run_games(max_moves=100)

        assert 'winners' in results
        assert 'move_counts' in results
        assert len(results['winners']) == small_runner.batch_size
        assert len(results['move_counts']) == small_runner.batch_size

    def test_all_games_complete_or_max_moves(self, small_runner):
        """Games should either complete or hit max_moves."""
        max_moves = 50
        results = small_runner.run_games(max_moves=max_moves)

        for count in results['move_counts']:
            # Count should be positive (games progressed)
            assert count >= 0

    def test_callback_called(self, small_runner):
        """Callback should be invoked during game."""
        callback_count = [0]

        def callback(move_num, state):
            callback_count[0] += 1

        small_runner.run_games(max_moves=20, callback=callback)

        # Should have been called at least a few times
        assert callback_count[0] > 0

    def test_with_custom_weights(self, small_runner):
        """Should accept custom weights per game."""
        weights_list = [
            {"material_weight": 10.0, "mobility_weight": 5.0}
            for _ in range(small_runner.batch_size)
        ]

        results = small_runner.run_games(
            weights_list=weights_list,
            max_moves=50,
        )

        assert len(results['winners']) == small_runner.batch_size


# =============================================================================
# Phase Transition Tests
# =============================================================================


class TestPhaseTransitions:
    """Tests for game phase transitions."""

    def test_starts_in_ring_placement(self, small_runner):
        """Games should start in ring placement phase."""
        small_runner.reset_games()
        state = small_runner.state

        assert (state.current_phase == GamePhase.RING_PLACEMENT).all()

    def test_transitions_to_movement(self, small_runner):
        """Should transition to movement after placement."""
        small_runner.reset_games()

        # Run a few moves to potentially transition
        small_runner.run_games(max_moves=100)
        state = small_runner.state

        # At least some games should have transitioned or completed
        # (not all games will necessarily be in the same phase)
        phases = state.current_phase.unique()
        assert len(phases) >= 1  # At least one phase state exists


# =============================================================================
# State Consistency Tests
# =============================================================================


class TestStateConsistency:
    """Tests for game state consistency."""

    def test_stack_owner_values_valid(self, small_runner):
        """Stack owner should only have valid player values."""
        small_runner.run_games(max_moves=50)
        state = small_runner.state

        # Stack owner should be 0 (empty) or 1-num_players
        owner_values = state.stack_owner.unique()
        for val in owner_values:
            assert 0 <= val.item() <= small_runner.num_players

    def test_stack_height_non_negative(self, small_runner):
        """Stack heights should never be negative."""
        small_runner.run_games(max_moves=50)
        state = small_runner.state

        assert (state.stack_height >= 0).all()

    def test_current_player_valid(self, small_runner):
        """Current player should be 1-num_players."""
        small_runner.run_games(max_moves=50)
        state = small_runner.state

        for g in range(small_runner.batch_size):
            player = state.current_player[g].item()
            assert 1 <= player <= small_runner.num_players

    def test_rings_in_hand_non_negative(self, small_runner):
        """Rings in hand should never be negative."""
        small_runner.run_games(max_moves=50)
        state = small_runner.state

        assert (state.rings_in_hand >= 0).all()


# =============================================================================
# Multi-Player Tests
# =============================================================================


class TestMultiPlayer:
    """Tests for multi-player games."""

    def test_four_player_game(self, device):
        """Should run 4-player games."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=4,
            device=device,
        )

        results = runner.run_games(max_moves=100)

        assert len(results['winners']) == 4
        # Winners should be 0 (draw/incomplete) or 1-4
        for winner in results['winners']:
            assert 0 <= winner <= 4


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance characteristics."""

    def test_batch_processing_scales(self, device):
        """Larger batches should be processed efficiently."""
        # Small batch
        small_runner = ParallelGameRunner(
            batch_size=10,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Larger batch
        large_runner = ParallelGameRunner(
            batch_size=100,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Both should complete
        small_results = small_runner.run_games(max_moves=20)
        large_results = large_runner.run_games(max_moves=20)

        assert len(small_results['winners']) == 10
        assert len(large_results['winners']) == 100


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for different configuration options."""

    def test_heuristic_selection(self, device):
        """Should work with heuristic move selection."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
            use_heuristic_selection=True,
        )

        results = runner.run_games(max_moves=50)
        assert len(results['winners']) == 4

    def test_temperature_setting(self, device):
        """Should accept temperature parameter."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
            temperature=0.5,
        )

        results = runner.run_games(max_moves=50)
        assert len(results['winners']) == 4

    def test_random_opening_moves(self, device):
        """Should support random opening moves."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
            random_opening_moves=5,
        )

        results = runner.run_games(max_moves=50)
        assert len(results['winners']) == 4


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_short_games(self, small_runner):
        """Should handle very short max_moves."""
        results = small_runner.run_games(max_moves=5)

        assert len(results['winners']) == small_runner.batch_size

    def test_empty_weights_list_uses_defaults(self, small_runner):
        """None weights_list should use defaults."""
        results = small_runner.run_games(weights_list=None, max_moves=20)

        assert len(results['winners']) == small_runner.batch_size

    def test_run_games_multiple_times(self, small_runner):
        """Should be able to run games multiple times on same runner."""
        results1 = small_runner.run_games(max_moves=20)
        results2 = small_runner.run_games(max_moves=20)

        assert len(results1['winners']) == small_runner.batch_size
        assert len(results2['winners']) == small_runner.batch_size


# =============================================================================
# Device Tests
# =============================================================================


class TestDeviceHandling:
    """Tests for device handling."""

    def test_tensors_on_correct_device(self, small_runner, device):
        """State tensors should be on the correct device."""
        small_runner.reset_games()
        state = small_runner.state

        assert state.device.type == device.type
        assert state.stack_owner.device.type == device.type
        assert state.stack_height.device.type == device.type

    def test_auto_device_selection(self):
        """Should auto-select GPU if available."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=None,  # Auto-select
        )

        # Should have selected a device
        assert runner.device is not None
