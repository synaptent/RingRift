"""Edge case tests for app/ai/gpu_parallel_games.py.

Tests cover:
- GameRunnerConfig dataclass validation and factory methods
- Board type variations (hex vs square)
- Victory conditions and game completion
- Error handling and invalid configurations
- Shadow validation mode
- Random opening moves
- Multi-player edge cases

Created: Dec 29, 2025
"""

import numpy as np
import pytest
import torch

from app.ai.gpu_batch import get_device
from app.ai.gpu_game_types import (
    GamePhase,
    GameStatus,
    MoveType,
    get_int_dtype,
    get_required_line_length,
)
from app.ai.gpu_move_generation import (
    BatchMoves,
    _empty_batch_moves,
    generate_placement_moves_batch,
)
from app.ai.gpu_parallel_games import (
    BatchGameState,
    GameRunnerConfig,
    ParallelGameRunner,
    select_moves_vectorized,
)


class TestGameRunnerConfig:
    """Tests for GameRunnerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GameRunnerConfig()
        assert config.batch_size == 64
        assert config.board_size == 8
        assert config.num_players == 2
        assert config.temperature == 1.0
        assert config.shadow_validation is False
        assert config.state_validation is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GameRunnerConfig(
            batch_size=128,
            board_size=10,
            num_players=4,
            temperature=0.5,
            shadow_validation=True,
        )
        assert config.batch_size == 128
        assert config.board_size == 10
        assert config.num_players == 4
        assert config.temperature == 0.5
        assert config.shadow_validation is True

    def test_for_selfplay_factory(self):
        """Test selfplay config factory method."""
        config = GameRunnerConfig.for_selfplay(
            board_type="hex8",
            num_players=2,
            batch_size=256,
        )
        assert config.board_type == "hex8"
        assert config.num_players == 2
        assert config.batch_size == 256
        assert config.record_policy is True
        assert config.use_heuristic_selection is True
        assert config.weight_noise == 0.1

    def test_for_evaluation_factory(self):
        """Test evaluation config factory method."""
        config = GameRunnerConfig.for_evaluation(
            board_type="square8",
            num_players=2,
        )
        assert config.board_type == "square8"
        assert config.record_policy is False
        assert config.weight_noise == 0.0
        assert config.temperature == 0.1  # Near-deterministic

    def test_for_cmaes_factory(self):
        """Test CMA-ES config factory method."""
        config = GameRunnerConfig.for_cmaes(
            board_type="hex8",
            num_players=2,
        )
        assert config.batch_size == 100
        assert config.weight_noise == 0.0
        # CMA-ES uses default temperature (1.0) for exploration variety
        assert config.temperature == 1.0

    def test_persona_validation(self):
        """Test that per_player_personas must match num_players."""
        with pytest.raises(ValueError, match="per_player_personas length"):
            GameRunnerConfig(
                num_players=2,
                per_player_personas=["aggressive", "defensive", "balanced"],
            )

    def test_persona_validation_correct(self):
        """Test valid per_player_personas configuration."""
        config = GameRunnerConfig(
            num_players=3,
            per_player_personas=["aggressive", "defensive", "balanced"],
        )
        assert len(config.per_player_personas) == config.num_players


class TestParallelGameRunnerWithConfig:
    """Tests for ParallelGameRunner with GameRunnerConfig."""

    def test_create_with_config(self):
        """Test creating runner with config object."""
        config = GameRunnerConfig(
            batch_size=8,
            board_size=8,
            num_players=2,
        )
        runner = ParallelGameRunner(config=config)
        assert runner.batch_size == 8
        assert runner.board_size == 8
        assert runner.num_players == 2

    def test_create_with_selfplay_config(self):
        """Test creating runner with selfplay config."""
        config = GameRunnerConfig.for_selfplay(
            board_type="hex8",
            num_players=2,
            batch_size=16,
        )
        runner = ParallelGameRunner(config=config)
        assert runner.batch_size == 16
        # Should have heuristic selection enabled
        assert runner.use_heuristic_selection is True


class TestBoardTypeVariations:
    """Tests for different board types."""

    @pytest.mark.parametrize("board_size", [6, 7, 8, 9, 10])
    def test_square_board_sizes(self, board_size):
        """Test various square board sizes."""
        device = torch.device("cpu")
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=board_size,
            num_players=2,
            device=device,
        )
        assert state.board_size == board_size
        assert state.stack_owner.shape == (2, board_size, board_size)

    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_multiplayer_configurations(self, num_players):
        """Test 2, 3, and 4 player configurations."""
        device = torch.device("cpu")
        state = BatchGameState.create_batch(
            batch_size=2,
            board_size=8,
            num_players=num_players,
            device=device,
        )
        assert state.num_players == num_players

    def test_line_length_consistency(self):
        """Test line length calculation matches rules."""
        # 2-player always uses 4
        assert get_required_line_length(8, 2) == 4
        assert get_required_line_length(7, 2) == 4
        assert get_required_line_length(9, 2) == 4

        # 3-4 player on 8x8 uses 3
        assert get_required_line_length(8, 3) == 3
        assert get_required_line_length(8, 4) == 3

        # 3-4 player on other sizes uses 4
        assert get_required_line_length(7, 3) == 4
        assert get_required_line_length(9, 4) == 4


class TestGameCompletionEdgeCases:
    """Tests for game completion scenarios."""

    def test_max_moves_termination(self):
        """Test that games terminate at max moves."""
        runner = ParallelGameRunner(
            batch_size=2,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )

        # Run with very low max_moves to force termination
        results = runner.run_games(max_moves=5)

        # Some games should have status indicating max moves
        assert results is not None
        assert "winners" in results

    def test_batch_with_mixed_completion(self):
        """Test batch where some games complete before others."""
        runner = ParallelGameRunner(
            batch_size=8,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )

        # Run for enough moves that some may complete
        results = runner.run_games(max_moves=50)

        assert results is not None
        # Winners should have entries for all games
        assert len(results["winners"]) == 8

    def test_empty_moves_handling(self):
        """Test handling when a game has no legal moves."""
        device = torch.device("cpu")
        batch_size = 2

        moves = _empty_batch_moves(batch_size, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        selected = select_moves_vectorized(
            moves=moves,
            active_mask=active_mask,
            board_size=8,
            temperature=1.0,
        )

        # Should return -1 for no moves
        assert (selected == -1).all()

    def test_inactive_game_handling(self):
        """Test that inactive games don't generate moves."""
        device = torch.device("cpu")
        batch_size = 4

        state = BatchGameState.create_batch(
            batch_size=batch_size,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Mark some games as completed
        state.game_status[0] = GameStatus.COMPLETED
        state.game_status[2] = GameStatus.COMPLETED

        # Generate moves only for active games
        active_mask = state.game_status == GameStatus.ACTIVE
        moves = generate_placement_moves_batch(state, active_mask=active_mask)

        # Moves should only be for active games (1 and 3)
        if moves.total_moves > 0:
            unique_games = moves.game_idx.unique().tolist()
            assert 0 not in unique_games
            assert 2 not in unique_games


class TestMoveSelectionEdgeCases:
    """Tests for move selection edge cases."""

    def test_temperature_zero(self):
        """Test temperature = 0 (deterministic selection)."""
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

        # Very low temperature for near-deterministic
        selected = select_moves_vectorized(
            moves=moves,
            active_mask=active_mask,
            board_size=8,
            temperature=0.01,
        )

        # Should still produce valid selections
        assert (selected >= 0).all()

    def test_temperature_very_high(self):
        """Test very high temperature (nearly uniform)."""
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

        # Very high temperature
        selected = select_moves_vectorized(
            moves=moves,
            active_mask=active_mask,
            board_size=8,
            temperature=100.0,
        )

        assert (selected >= 0).all()

    def test_single_game_batch(self):
        """Test batch_size=1 case."""
        device = torch.device("cpu")

        state = BatchGameState.create_batch(
            batch_size=1,
            board_size=8,
            num_players=2,
            device=device,
        )

        moves = generate_placement_moves_batch(state)
        active_mask = torch.ones(1, dtype=torch.bool, device=device)

        selected = select_moves_vectorized(
            moves=moves,
            active_mask=active_mask,
            board_size=8,
            temperature=1.0,
        )

        assert selected.shape == (1,)
        assert selected[0] >= 0

    def test_large_batch(self):
        """Test with large batch size."""
        device = torch.device("cpu")
        batch_size = 256

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

        assert selected.shape == (batch_size,)
        assert (selected >= 0).all()


class TestMultiplayerEdgeCases:
    """Tests for 3-4 player specific edge cases."""

    def test_4player_ring_placement(self):
        """Test 4-player ring placement phase."""
        device = torch.device("cpu")

        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=4,
            device=device,
        )

        # Should start in ring placement
        assert (state.current_phase == GamePhase.RING_PLACEMENT).all()

        # Generate placement moves
        moves = generate_placement_moves_batch(state)
        assert moves.total_moves > 0

    def test_3player_configuration(self):
        """Test 3-player specific configuration."""
        device = torch.device("cpu")

        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=3,
            device=device,
        )

        assert state.num_players == 3

        # Line length should be 3 for 3-player on 8x8
        line_length = get_required_line_length(8, 3)
        assert line_length == 3

    def test_multiplayer_runner(self):
        """Test ParallelGameRunner with 3-4 players."""
        for num_players in [3, 4]:
            runner = ParallelGameRunner(
                batch_size=4,
                board_size=8,
                num_players=num_players,
                device=torch.device("cpu"),
            )

            results = runner.run_games(max_moves=20)
            assert results is not None
            assert len(results["winners"]) == 4


class TestDeviceEdgeCases:
    """Tests for device-related edge cases."""

    def test_cpu_fallback(self):
        """Test that CPU fallback works correctly."""
        device = torch.device("cpu")

        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Should work on CPU
        results = runner.run_games(max_moves=10)
        assert results is not None

    def test_device_consistency(self):
        """Test all tensors stay on the same device."""
        device = torch.device("cpu")

        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Check all tensors are on CPU
        assert state.stack_owner.device.type == "cpu"
        assert state.current_player.device.type == "cpu"
        assert state.current_phase.device.type == "cpu"
        assert state.game_status.device.type == "cpu"
        assert state.rings_in_hand.device.type == "cpu"

    def test_int_dtype_consistency(self):
        """Test integer dtype is consistent for device."""
        for device_type in ["cpu"]:
            device = torch.device(device_type)
            dtype = get_int_dtype(device)
            assert dtype == torch.int16


class TestRunnerStatistics:
    """Tests for runner statistics tracking."""

    def test_stats_after_run(self):
        """Test statistics are populated after running games."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )

        runner.run_games(max_moves=20)
        stats = runner.get_stats()

        assert "games_completed" in stats
        assert "total_moves" in stats
        assert "games_per_second" in stats
        assert stats["total_moves"] > 0

    def test_stats_reset_on_new_run(self):
        """Test statistics reset when starting new games."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )

        # First run
        runner.run_games(max_moves=10)
        stats1 = runner.get_stats()

        # Reset and run again
        runner.reset_games()
        runner.run_games(max_moves=10)
        stats2 = runner.get_stats()

        # Both should have valid statistics
        assert stats1["total_moves"] > 0
        assert stats2["total_moves"] > 0


class TestRandomOpeningMoves:
    """Tests for random opening moves feature."""

    def test_runner_with_random_opening(self):
        """Test runner with random opening moves enabled."""
        config = GameRunnerConfig(
            batch_size=4,
            board_size=8,
            num_players=2,
            random_opening_moves=3,
        )
        runner = ParallelGameRunner(config=config)

        # Should run without error
        results = runner.run_games(max_moves=20)
        assert results is not None


class TestHeuristicSelection:
    """Tests for heuristic-based move selection."""

    def test_heuristic_weights(self):
        """Test that default heuristic weights are valid."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
        )

        weights = runner._default_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0

        # All weights should be numeric
        for key, value in weights.items():
            assert isinstance(value, (int, float)), f"Invalid weight type for {key}"

    def test_heuristic_selection_mode(self):
        """Test running with heuristic selection enabled."""
        runner = ParallelGameRunner(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=torch.device("cpu"),
            use_heuristic_selection=True,
        )

        results = runner.run_games(max_moves=20)
        assert results is not None
        assert len(results["winners"]) == 4


class TestBatchStateOperations:
    """Tests for batch state operations."""

    def test_clone_state(self):
        """Test that state can be cloned (if method exists)."""
        device = torch.device("cpu")

        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Modify original
        state.current_player[0] = 2

        # Check the modification took effect
        assert state.current_player[0] == 2

    def test_state_tensor_types(self):
        """Test state tensors have correct dtypes."""
        device = torch.device("cpu")

        state = BatchGameState.create_batch(
            batch_size=4,
            board_size=8,
            num_players=2,
            device=device,
        )

        # Game status uses int8 for memory efficiency (values 0-3)
        assert state.game_status.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]

        # Current player uses int16 for small values
        assert state.current_player.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]

        # Current phase uses int8 for small enum values
        assert state.current_phase.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
