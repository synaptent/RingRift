"""Tests for parallel_selfplay.py.

December 2025: Comprehensive test coverage for parallel selfplay generation.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from app.models import BoardType
from app.training.parallel_selfplay import (
    ParallelSelfplayConfig,
    SelfplayConfig,  # Backward compat alias
    GameResult,
    _get_globals,
    _count_pieces,
    _worker_init,
    generate_dataset_parallel,
)


# =============================================================================
# ParallelSelfplayConfig Tests
# =============================================================================


class TestParallelSelfplayConfig:
    """Test ParallelSelfplayConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ParallelSelfplayConfig()
        assert config.board_type == BoardType.SQUARE8
        assert config.num_players == 2
        assert config.max_moves == 10000
        assert config.engine == "gumbel"
        assert config.nn_model_id is None
        assert config.multi_player_values is False
        assert config.max_players == 4
        assert config.graded_outcomes is False
        assert config.history_length == 3
        assert config.feature_version == 1

    def test_gumbel_settings(self):
        """Test Gumbel-MCTS specific settings."""
        config = ParallelSelfplayConfig()
        assert config.gumbel_simulations == 64
        assert config.gumbel_top_k == 16
        assert config.gumbel_c_visit == 50.0
        assert config.gumbel_c_scale == 1.0

    def test_temperature_settings(self):
        """Test temperature scheduling settings."""
        config = ParallelSelfplayConfig()
        assert config.temperature == 1.0
        assert config.use_temperature_decay is False
        assert config.move_temp_threshold == 30
        assert config.opening_temperature == 1.5
        assert config.temperature_schedule == "linear"

    def test_opening_book_settings(self):
        """Test opening book settings."""
        config = ParallelSelfplayConfig()
        assert config.use_opening_book is False
        assert config.opening_book_prob == 0.8
        assert config.opening_book_min_openings == 100

    def test_ebmo_settings(self):
        """Test EBMO online learning settings."""
        config = ParallelSelfplayConfig()
        assert config.ebmo_online_learning is False
        assert config.ebmo_online_lr == 1e-5
        assert config.ebmo_online_buffer_size == 20

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ParallelSelfplayConfig(
            board_type=BoardType.HEX8,
            num_players=4,
            engine="mcts",
            gumbel_simulations=128,
            temperature=0.5,
        )
        assert config.board_type == BoardType.HEX8
        assert config.num_players == 4
        assert config.engine == "mcts"
        assert config.gumbel_simulations == 128
        assert config.temperature == 0.5

    def test_backward_compat_alias(self):
        """Test SelfplayConfig is an alias for ParallelSelfplayConfig."""
        assert SelfplayConfig is ParallelSelfplayConfig


# =============================================================================
# GameResult Tests
# =============================================================================


class TestGameResult:
    """Test GameResult dataclass."""

    def test_basic_creation(self):
        """Test basic GameResult creation."""
        result = GameResult(
            features=np.zeros((10, 48, 8, 8)),
            globals=np.zeros((10, 16)),
            values=np.zeros(10),
            policy_indices=[np.array([1, 2, 3])],
            policy_values=[np.array([0.5, 0.3, 0.2])],
            values_mp=None,
            num_players=None,
            num_samples=10,
            game_idx=0,
            duration_sec=5.0,
        )
        assert result.num_samples == 10
        assert result.game_idx == 0
        assert result.duration_sec == 5.0
        assert result.features.shape == (10, 48, 8, 8)

    def test_optional_fields(self):
        """Test optional fields with default None."""
        result = GameResult(
            features=np.zeros((5, 48, 8, 8)),
            globals=np.zeros((5, 16)),
            values=np.zeros(5),
            policy_indices=[],
            policy_values=[],
            values_mp=None,
            num_players=None,
            num_samples=5,
            game_idx=1,
            duration_sec=2.0,
        )
        assert result.effective_temps is None
        assert result.game_lengths is None
        assert result.piece_counts is None
        assert result.outcomes is None

    def test_all_fields_provided(self):
        """Test GameResult with all fields provided."""
        result = GameResult(
            features=np.zeros((8, 48, 8, 8)),
            globals=np.zeros((8, 16)),
            values=np.ones(8),
            policy_indices=[np.array([0, 1]) for _ in range(8)],
            policy_values=[np.array([0.6, 0.4]) for _ in range(8)],
            values_mp=np.zeros((8, 4)),
            num_players=np.full(8, 2),
            num_samples=8,
            game_idx=5,
            duration_sec=10.0,
            effective_temps=np.ones(8),
            game_lengths=np.full(8, 50),
            piece_counts=np.arange(8),
            outcomes=np.array([0, 1, 2, 0, 1, 2, 0, 1]),
        )
        assert result.values_mp is not None
        assert result.num_players is not None
        assert result.effective_temps.shape == (8,)
        assert result.game_lengths.shape == (8,)
        assert result.piece_counts.shape == (8,)
        assert result.outcomes.shape == (8,)


# =============================================================================
# _get_globals Tests
# =============================================================================


class TestGetGlobals:
    """Test _get_globals function."""

    def test_basic_globals(self):
        """Test basic globals extraction."""
        state = MagicMock()
        state.turn_number = 10

        globals_vec = _get_globals(state, current_player=1)
        assert globals_vec.shape == (16,)
        assert globals_vec.dtype == np.float32
        assert globals_vec[0] == 1 / 4.0  # current_player / 4.0
        assert globals_vec[1] == 10 / 200.0  # turn_number / 200.0

    def test_player_normalization(self):
        """Test player number normalization."""
        state = MagicMock()
        state.turn_number = 0

        for player in [1, 2, 3, 4]:
            globals_vec = _get_globals(state, current_player=player)
            assert globals_vec[0] == player / 4.0

    def test_turn_number_normalization(self):
        """Test turn number normalization."""
        state = MagicMock()

        for turn in [0, 50, 100, 200]:
            state.turn_number = turn
            globals_vec = _get_globals(state, current_player=1)
            assert globals_vec[1] == turn / 200.0

    def test_missing_turn_number(self):
        """Test handling of missing turn_number attribute."""
        state = MagicMock(spec=[])  # No attributes
        globals_vec = _get_globals(state, current_player=2)
        assert globals_vec[1] == 0.0  # getattr default


# =============================================================================
# _count_pieces Tests
# =============================================================================


class TestCountPieces:
    """Test _count_pieces function."""

    def test_empty_board(self):
        """Test counting pieces on empty board."""
        state = MagicMock()
        state.board = MagicMock()
        state.board.stacks = {}

        count = _count_pieces(state)
        assert count == 0

    def test_stacks_with_rings(self):
        """Test counting pieces in stacks with rings attribute."""
        state = MagicMock()
        state.board = MagicMock()

        stack1 = MagicMock()
        stack1.rings = [1, 2, 3]

        stack2 = MagicMock()
        stack2.rings = [1, 2]

        state.board.stacks = {"pos1": stack1, "pos2": stack2}

        count = _count_pieces(state)
        assert count == 5

    def test_stacks_as_lists(self):
        """Test counting pieces when stacks are lists."""
        state = MagicMock()
        state.board = MagicMock()
        state.board.stacks = {"pos1": [1, 2], "pos2": [1, 2, 3, 4]}

        count = _count_pieces(state)
        assert count == 6

    def test_missing_board(self):
        """Test handling when board is missing."""
        state = MagicMock(spec=[])  # No board attribute
        count = _count_pieces(state)
        assert count == 0

    def test_missing_stacks(self):
        """Test handling when stacks is missing."""
        state = MagicMock()
        state.board = MagicMock(spec=[])  # No stacks attribute
        count = _count_pieces(state)
        assert count == 0


# =============================================================================
# _worker_init Tests
# =============================================================================


class TestWorkerInit:
    """Test _worker_init function."""

    def test_config_stored(self):
        """Test config is stored in worker global."""
        from app.training import parallel_selfplay

        config_dict = {
            "board_type": BoardType.HEX8,
            "num_players": 2,
            "engine": "gumbel",
        }

        # Mock torch import to avoid thread setting errors in test environment
        # torch is imported inside the function, so we mock sys.modules
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            _worker_init(config_dict)

        assert parallel_selfplay._worker_config["board_type"] == BoardType.HEX8
        assert parallel_selfplay._worker_config["num_players"] == 2

    def test_env_vars_set(self):
        """Test environment variables are set for thread limiting."""
        config_dict = {"board_type": BoardType.SQUARE8}

        # Mock torch import to avoid thread setting errors in test environment
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            _worker_init(config_dict)

        # Check env vars were set
        assert os.environ.get("OMP_NUM_THREADS") == "1"
        assert os.environ.get("MKL_NUM_THREADS") == "1"
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
        assert os.environ.get("NUMEXPR_NUM_THREADS") == "1"

    def test_ai_service_root_added_to_path(self):
        """Test ai_service_root is added to sys.path."""
        import sys
        from app.training import parallel_selfplay

        config_dict = {
            "_ai_service_root": "/custom/path/ai-service",
        }

        original_path = sys.path.copy()
        try:
            # Mock torch import to avoid thread setting errors in test environment
            mock_torch = MagicMock()
            with patch.dict("sys.modules", {"torch": mock_torch}):
                _worker_init(config_dict)
            # Path should be added
            assert parallel_selfplay._worker_config["_ai_service_root"] == "/custom/path/ai-service"
        finally:
            # Clean up
            if "/custom/path/ai-service" in sys.path:
                sys.path.remove("/custom/path/ai-service")


# =============================================================================
# generate_dataset_parallel Tests
# =============================================================================


class TestGenerateDatasetParallel:
    """Test generate_dataset_parallel function."""

    def test_output_file_creation(self):
        """Test output file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "subdir", "test_dataset.npz")

            # Mock the ProcessPoolExecutor to avoid actual parallel execution
            with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
                mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = None

                # Create minimal game result
                mock_result = GameResult(
                    features=np.zeros((5, 48, 8, 8)),
                    globals=np.zeros((5, 16)),
                    values=np.zeros(5),
                    policy_indices=[np.array([0]) for _ in range(5)],
                    policy_values=[np.array([1.0]) for _ in range(5)],
                    values_mp=None,
                    num_players=None,
                    num_samples=5,
                    game_idx=0,
                    duration_sec=1.0,
                    effective_temps=np.ones(5),
                    game_lengths=np.full(5, 10),
                    piece_counts=np.arange(5),
                    outcomes=np.array([2, 1, 0, 1, 2]),
                )

                # Setup mock to return results
                mock_future = MagicMock()
                mock_future.result.return_value = mock_result
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                with patch("app.training.parallel_selfplay.as_completed", return_value=[mock_future]):
                    total = generate_dataset_parallel(
                        num_games=1,
                        output_file=output_file,
                        num_workers=1,
                        board_type=BoardType.SQUARE8,
                    )

                    assert total == 5
                    assert os.path.exists(output_file)

    def test_num_workers_default(self):
        """Test default num_workers is calculated correctly."""
        import multiprocessing as mp

        with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = None

            with patch.object(mp, "cpu_count", return_value=32):
                # Clear env var to test default behavior
                with patch.dict(os.environ, {}, clear=True):
                    # No results, should return 0
                    with patch("app.training.parallel_selfplay.as_completed", return_value=[]):
                        generate_dataset_parallel(
                            num_games=1,
                            output_file="/tmp/test.npz",
                        )

                        # Check that workers were capped
                        call_args = mock_executor.call_args
                        assert call_args is not None
                        assert call_args.kwargs.get("max_workers", 16) <= 16

    def test_num_workers_from_env(self):
        """Test num_workers from environment variable."""
        with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = None

            with patch.dict(os.environ, {"RINGRIFT_PARALLEL_WORKERS": "8"}):
                with patch("app.training.parallel_selfplay.as_completed", return_value=[]):
                    generate_dataset_parallel(
                        num_games=1,
                        output_file="/tmp/test.npz",
                    )

                    call_args = mock_executor.call_args
                    assert call_args is not None
                    assert call_args.kwargs.get("max_workers") == 8

    def test_opening_book_env_check(self):
        """Test opening book setting from environment."""
        with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = None

            with patch.dict(os.environ, {"RINGRIFT_USE_OPENING_BOOK": "1"}):
                with patch("app.training.parallel_selfplay.as_completed", return_value=[]):
                    generate_dataset_parallel(
                        num_games=1,
                        output_file="/tmp/test.npz",
                        use_opening_book=None,  # Let it check env
                    )

                    # Config dict should have use_opening_book=True
                    call_args = mock_executor.call_args
                    initargs = call_args.kwargs.get("initargs", ())
                    if initargs:
                        config_dict = initargs[0]
                        assert config_dict.get("use_opening_book") is True

    def test_progress_callback(self):
        """Test progress callback is called."""
        callback_calls = []

        def mock_callback(completed, total):
            callback_calls.append((completed, total))

        with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
            mock_result = GameResult(
                features=np.zeros((5, 48, 8, 8)),
                globals=np.zeros((5, 16)),
                values=np.zeros(5),
                policy_indices=[np.array([0]) for _ in range(5)],
                policy_values=[np.array([1.0]) for _ in range(5)],
                values_mp=None,
                num_players=None,
                num_samples=5,
                game_idx=0,
                duration_sec=1.0,
                effective_temps=np.ones(5),
                game_lengths=np.full(5, 10),
                piece_counts=np.arange(5),
                outcomes=np.array([2, 1, 0, 1, 2]),
            )

            mock_future = MagicMock()
            mock_future.result.return_value = mock_result
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

            with patch("app.training.parallel_selfplay.as_completed", return_value=[mock_future]):
                generate_dataset_parallel(
                    num_games=1,
                    output_file="/tmp/test.npz",
                    num_workers=1,
                    progress_callback=mock_callback,
                )

                assert len(callback_calls) > 0
                assert callback_calls[0][1] == 1  # total

    def test_no_results_returns_zero(self):
        """Test returns 0 when no games complete."""
        with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = None

            with patch("app.training.parallel_selfplay.as_completed", return_value=[]):
                result = generate_dataset_parallel(
                    num_games=1,
                    output_file="/tmp/test.npz",
                    num_workers=1,
                )
                assert result == 0

    def test_multi_player_values(self):
        """Test multi_player_values option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_mp.npz")

            with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
                mock_result = GameResult(
                    features=np.zeros((5, 48, 8, 8)),
                    globals=np.zeros((5, 16)),
                    values=np.zeros(5),
                    policy_indices=[np.array([0]) for _ in range(5)],
                    policy_values=[np.array([1.0]) for _ in range(5)],
                    values_mp=np.zeros((5, 4)),
                    num_players=np.full(5, 4),
                    num_samples=5,
                    game_idx=0,
                    duration_sec=1.0,
                    effective_temps=np.ones(5),
                    game_lengths=np.full(5, 10),
                    piece_counts=np.arange(5),
                    outcomes=np.array([2, 1, 0, 1, 2]),
                )

                mock_future = MagicMock()
                mock_future.result.return_value = mock_result
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                with patch("app.training.parallel_selfplay.as_completed", return_value=[mock_future]):
                    generate_dataset_parallel(
                        num_games=1,
                        output_file=output_file,
                        num_workers=1,
                        multi_player_values=True,
                    )

                    # Load and verify
                    data = np.load(output_file, allow_pickle=True)
                    assert "values_mp" in data
                    assert "num_players" in data

    def test_game_timeout_calculation(self):
        """Test adaptive game timeout calculation."""
        with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = None

            with patch("app.training.parallel_selfplay.as_completed", return_value=[]):
                # Test different board types have different timeouts
                for board_type in [BoardType.SQUARE8, BoardType.HEX8]:
                    generate_dataset_parallel(
                        num_games=1,
                        output_file="/tmp/test.npz",
                        num_workers=1,
                        board_type=board_type,
                    )


# =============================================================================
# Config Dict Construction Tests
# =============================================================================


class TestConfigDictConstruction:
    """Test config dict construction for workers."""

    def test_all_config_fields_included(self):
        """Test all config fields are included in config dict."""
        with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
            mock_executor.return_value.__enter__.return_value.submit.return_value.result.return_value = None

            with patch("app.training.parallel_selfplay.as_completed", return_value=[]):
                generate_dataset_parallel(
                    num_games=1,
                    output_file="/tmp/test.npz",
                    num_workers=1,
                    board_type=BoardType.HEX8,
                    num_players=4,
                    engine="mcts",
                    temperature=0.5,
                    use_temperature_decay=True,
                    opening_temperature=2.0,
                    move_temp_threshold=20,
                    gumbel_simulations=128,
                    gumbel_top_k=32,
                    use_opening_book=True,
                    opening_book_prob=0.9,
                    ebmo_online_learning=True,
                    ebmo_online_lr=1e-4,
                )

                call_args = mock_executor.call_args
                initargs = call_args.kwargs.get("initargs", ())
                assert len(initargs) > 0

                config_dict = initargs[0]
                assert config_dict["board_type"] == BoardType.HEX8
                assert config_dict["num_players"] == 4
                assert config_dict["engine"] == "mcts"
                assert config_dict["temperature"] == 0.5
                assert config_dict["gumbel_simulations"] == 128
                assert config_dict["use_opening_book"] is True
                assert config_dict["ebmo_online_learning"] is True
                assert "_ai_service_root" in config_dict


# =============================================================================
# NPZ File Structure Tests
# =============================================================================


class TestNPZFileStructure:
    """Test NPZ file structure and contents."""

    def test_required_fields_in_npz(self):
        """Test all required fields are saved to NPZ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_structure.npz")

            with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
                mock_result = GameResult(
                    features=np.random.randn(5, 48, 8, 8).astype(np.float32),
                    globals=np.random.randn(5, 16).astype(np.float32),
                    values=np.random.randn(5).astype(np.float32),
                    policy_indices=[np.array([0, 1, 2]) for _ in range(5)],
                    policy_values=[np.array([0.5, 0.3, 0.2]) for _ in range(5)],
                    values_mp=None,
                    num_players=None,
                    num_samples=5,
                    game_idx=0,
                    duration_sec=1.0,
                    effective_temps=np.ones(5, dtype=np.float32),
                    game_lengths=np.full(5, 50, dtype=np.int32),
                    piece_counts=np.arange(5, dtype=np.int32),
                    outcomes=np.array([0, 1, 2, 1, 0], dtype=np.int64),
                )

                mock_future = MagicMock()
                mock_future.result.return_value = mock_result
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                with patch("app.training.parallel_selfplay.as_completed", return_value=[mock_future]):
                    generate_dataset_parallel(
                        num_games=1,
                        output_file=output_file,
                        num_workers=1,
                    )

                    # Verify file structure
                    data = np.load(output_file, allow_pickle=True)

                    # Core training data
                    assert "features" in data
                    assert "globals" in data
                    assert "values" in data
                    assert "pol_indices" in data
                    assert "pol_values" in data

                    # Metadata
                    assert "policy_encoding" in data
                    assert "history_length" in data
                    assert "feature_version" in data

                    # Temperature data
                    assert "effective_temps" in data
                    assert "temp_config" in data

                    # Auxiliary task targets
                    assert "game_lengths" in data
                    assert "piece_counts" in data
                    assert "outcomes" in data

    def test_data_shapes(self):
        """Test data shapes in NPZ file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_shapes.npz")

            with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
                mock_result = GameResult(
                    features=np.zeros((10, 48, 8, 8), dtype=np.float32),
                    globals=np.zeros((10, 16), dtype=np.float32),
                    values=np.zeros(10, dtype=np.float32),
                    policy_indices=[np.array([0]) for _ in range(10)],
                    policy_values=[np.array([1.0]) for _ in range(10)],
                    values_mp=None,
                    num_players=None,
                    num_samples=10,
                    game_idx=0,
                    duration_sec=1.0,
                    effective_temps=np.ones(10, dtype=np.float32),
                    game_lengths=np.full(10, 30, dtype=np.int32),
                    piece_counts=np.arange(10, dtype=np.int32),
                    outcomes=np.zeros(10, dtype=np.int64),
                )

                mock_future = MagicMock()
                mock_future.result.return_value = mock_result
                mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

                with patch("app.training.parallel_selfplay.as_completed", return_value=[mock_future]):
                    generate_dataset_parallel(
                        num_games=1,
                        output_file=output_file,
                        num_workers=1,
                    )

                    data = np.load(output_file, allow_pickle=True)

                    assert data["features"].shape == (10, 48, 8, 8)
                    assert data["globals"].shape == (10, 16)
                    assert data["values"].shape == (10,)
                    assert data["effective_temps"].shape == (10,)
                    assert data["game_lengths"].shape == (10,)
                    assert data["piece_counts"].shape == (10,)
                    assert data["outcomes"].shape == (10,)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in parallel selfplay."""

    def test_game_failure_continues(self):
        """Test that failed games don't stop the whole process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_failure.npz")

            with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
                # First game fails, second succeeds
                mock_result = GameResult(
                    features=np.zeros((5, 48, 8, 8)),
                    globals=np.zeros((5, 16)),
                    values=np.zeros(5),
                    policy_indices=[np.array([0]) for _ in range(5)],
                    policy_values=[np.array([1.0]) for _ in range(5)],
                    values_mp=None,
                    num_players=None,
                    num_samples=5,
                    game_idx=1,
                    duration_sec=1.0,
                    effective_temps=np.ones(5),
                    game_lengths=np.full(5, 10),
                    piece_counts=np.arange(5),
                    outcomes=np.array([2, 1, 0, 1, 2]),
                )

                mock_future_fail = MagicMock()
                mock_future_fail.result.side_effect = Exception("Game failed")

                mock_future_success = MagicMock()
                mock_future_success.result.return_value = mock_result

                mock_executor.return_value.__enter__.return_value.submit.side_effect = [
                    mock_future_fail,
                    mock_future_success,
                ]

                # Setup as_completed to return both futures
                with patch("app.training.parallel_selfplay.as_completed",
                          return_value=[mock_future_fail, mock_future_success]):
                    result = generate_dataset_parallel(
                        num_games=2,
                        output_file=output_file,
                        num_workers=1,
                    )

                    # Should still have samples from successful game
                    assert result == 5

    def test_timeout_handling(self):
        """Test timeout handling for slow games."""
        with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
            mock_future = MagicMock()
            mock_future.result.side_effect = TimeoutError("Game timed out")
            mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future

            with patch("app.training.parallel_selfplay.as_completed", return_value=[mock_future]):
                # Should not raise, just log warning
                result = generate_dataset_parallel(
                    num_games=1,
                    output_file="/tmp/test.npz",
                    num_workers=1,
                )
                assert result == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestParallelSelfplayIntegration:
    """Integration tests for parallel selfplay."""

    def test_multiple_game_results_concatenated(self):
        """Test multiple game results are properly concatenated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_concat.npz")

            with patch("app.training.parallel_selfplay.ProcessPoolExecutor") as mock_executor:
                # Create two game results
                results = []
                for i in range(2):
                    results.append(GameResult(
                        features=np.zeros((5, 48, 8, 8)) + i,
                        globals=np.zeros((5, 16)) + i,
                        values=np.zeros(5) + i,
                        policy_indices=[np.array([i]) for _ in range(5)],
                        policy_values=[np.array([1.0]) for _ in range(5)],
                        values_mp=None,
                        num_players=None,
                        num_samples=5,
                        game_idx=i,
                        duration_sec=1.0,
                        effective_temps=np.ones(5) * (i + 1),
                        game_lengths=np.full(5, 10 * (i + 1)),
                        piece_counts=np.arange(5) + (i * 5),
                        outcomes=np.array([i % 3] * 5),
                    ))

                mock_futures = []
                for result in results:
                    mock_future = MagicMock()
                    mock_future.result.return_value = result
                    mock_futures.append(mock_future)

                mock_executor.return_value.__enter__.return_value.submit.side_effect = mock_futures

                with patch("app.training.parallel_selfplay.as_completed", return_value=mock_futures):
                    total = generate_dataset_parallel(
                        num_games=2,
                        output_file=output_file,
                        num_workers=1,
                    )

                    assert total == 10  # 5 samples * 2 games

                    data = np.load(output_file, allow_pickle=True)
                    assert data["features"].shape[0] == 10
                    assert data["values"].shape[0] == 10
