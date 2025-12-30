"""Unit tests for GPU MCTS selfplay module.

Tests cover:
- Configuration dataclass validation
- Sample and game record dataclasses
- Runner initialization and component setup
- Move-to-key conversion
- Value assignment for different player counts
- NPZ export format
- Convenience function
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from app.training.gpu_mcts_selfplay import (
    GPUMCTSSelfplayConfig,
    SelfplaySample,
    GameRecord,
    GPUMCTSSelfplayRunner,
    run_gpu_mcts_selfplay,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Create default GPU MCTS selfplay config."""
    return GPUMCTSSelfplayConfig()


@pytest.fixture
def hex8_2p_config():
    """Create hex8 2-player config."""
    return GPUMCTSSelfplayConfig(
        board_type="hex8",
        num_players=2,
        batch_size=4,
        simulation_budget=100,  # Lower for tests
        max_moves_per_game=50,
        device="cpu",
    )


@pytest.fixture
def sample_sample():
    """Create a sample SelfplaySample."""
    return SelfplaySample(
        features=np.zeros((12, 9, 9), dtype=np.float32),
        globals_vec=np.zeros(10, dtype=np.float32),
        policy_indices=np.array([0, 5, 10], dtype=np.int64),
        policy_values=np.array([0.6, 0.3, 0.1], dtype=np.float32),
        value=0.5,
        player=1,
        move_number=3,
        game_id="test_game_001",
    )


@pytest.fixture
def mock_game_state():
    """Create a mock game state."""
    from app.models import BoardType, GamePhase, GameStatus

    state = MagicMock()
    state.current_player = 1
    state.phase = GamePhase.RING_PLACEMENT
    state.game_status = GameStatus.ACTIVE
    state.winner = None
    state.game_over = False

    # Mock board
    state.board = MagicMock()
    state.board.type = BoardType.HEX8
    state.board.size = 9

    # Mock players
    player1 = MagicMock()
    player1.player_number = 1
    player1.territory_spaces = 10
    player1.eliminated_rings = 0

    player2 = MagicMock()
    player2.player_number = 2
    player2.territory_spaces = 8
    player2.eliminated_rings = 1

    state.players = [player1, player2]
    state.num_players = 2

    return state


# =============================================================================
# GPUMCTSSelfplayConfig Tests
# =============================================================================


class TestGPUMCTSSelfplayConfig:
    """Tests for GPUMCTSSelfplayConfig dataclass."""

    def test_default_values(self, default_config):
        """Test default configuration values."""
        assert default_config.board_type == "hex8"
        assert default_config.num_players == 2
        assert default_config.num_sampled_actions == 16
        assert default_config.simulation_budget == 800
        assert default_config.max_nodes == 1024
        assert default_config.batch_size == 64
        assert default_config.max_moves_per_game == 300
        assert default_config.encoder_version == "v3"
        assert default_config.feature_version == 2
        assert default_config.history_length == 3
        assert default_config.model_path is None
        assert default_config.device == "cuda"
        assert default_config.record_state_snapshots is True
        assert default_config.sample_every == 1
        assert default_config.engine_mode == "gumbel-mcts"

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = GPUMCTSSelfplayConfig(
            board_type="square19",
            num_players=4,
            simulation_budget=1600,
            batch_size=32,
            device="cpu",
            model_path="/path/to/model.pth",
        )
        assert config.board_type == "square19"
        assert config.num_players == 4
        assert config.simulation_budget == 1600
        assert config.batch_size == 32
        assert config.device == "cpu"
        assert config.model_path == "/path/to/model.pth"

    def test_all_board_types(self):
        """Test configuration with all board types."""
        for board_type in ["hex8", "hexagonal", "square8", "square19"]:
            config = GPUMCTSSelfplayConfig(board_type=board_type)
            assert config.board_type == board_type

    def test_all_player_counts(self):
        """Test configuration with different player counts."""
        for num_players in [2, 3, 4]:
            config = GPUMCTSSelfplayConfig(num_players=num_players)
            assert config.num_players == num_players


# =============================================================================
# SelfplaySample Tests
# =============================================================================


class TestSelfplaySample:
    """Tests for SelfplaySample dataclass."""

    def test_sample_creation(self, sample_sample):
        """Test sample creation with valid data."""
        assert sample_sample.features.shape == (12, 9, 9)
        assert sample_sample.globals_vec.shape == (10,)
        assert len(sample_sample.policy_indices) == 3
        assert len(sample_sample.policy_values) == 3
        assert sample_sample.value == 0.5
        assert sample_sample.player == 1
        assert sample_sample.move_number == 3
        assert sample_sample.game_id == "test_game_001"

    def test_sample_dtypes(self, sample_sample):
        """Test sample array dtypes."""
        assert sample_sample.features.dtype == np.float32
        assert sample_sample.globals_vec.dtype == np.float32
        assert sample_sample.policy_indices.dtype == np.int64
        assert sample_sample.policy_values.dtype == np.float32

    def test_sample_value_range(self):
        """Test sample can have values in valid range."""
        # Winner (value = 1)
        winner_sample = SelfplaySample(
            features=np.zeros((1,), dtype=np.float32),
            globals_vec=np.zeros((1,), dtype=np.float32),
            policy_indices=np.array([0]),
            policy_values=np.array([1.0]),
            value=1.0,
            player=1,
            move_number=0,
            game_id="test",
        )
        assert winner_sample.value == 1.0

        # Loser (value = -1)
        loser_sample = SelfplaySample(
            features=np.zeros((1,), dtype=np.float32),
            globals_vec=np.zeros((1,), dtype=np.float32),
            policy_indices=np.array([0]),
            policy_values=np.array([1.0]),
            value=-1.0,
            player=2,
            move_number=0,
            game_id="test",
        )
        assert loser_sample.value == -1.0


# =============================================================================
# GameRecord Tests
# =============================================================================


class TestGameRecord:
    """Tests for GameRecord dataclass."""

    def test_empty_game_record(self):
        """Test empty game record creation."""
        record = GameRecord(game_id="test_001")
        assert record.game_id == "test_001"
        assert record.samples == []
        assert record.winner is None
        assert record.total_moves == 0
        assert record.termination_reason == "normal"

    def test_game_record_with_samples(self, sample_sample):
        """Test game record with samples."""
        record = GameRecord(
            game_id="test_002",
            samples=[sample_sample],
            winner=1,
            total_moves=50,
            termination_reason="normal",
        )
        assert len(record.samples) == 1
        assert record.winner == 1
        assert record.total_moves == 50

    def test_game_record_termination_reasons(self):
        """Test different termination reasons."""
        for reason in ["normal", "move_limit", "error: test", "draw"]:
            record = GameRecord(game_id="test", termination_reason=reason)
            assert record.termination_reason == reason


# =============================================================================
# GPUMCTSSelfplayRunner Tests
# =============================================================================


class TestGPUMCTSSelfplayRunner:
    """Tests for GPUMCTSSelfplayRunner class."""

    def test_runner_initialization(self, hex8_2p_config):
        """Test runner initialization."""
        runner = GPUMCTSSelfplayRunner(hex8_2p_config)
        assert runner.config == hex8_2p_config
        # Lazy init - components should be None
        assert runner._mcts is None
        assert runner._encoder is None
        assert runner._neural_net is None
        assert runner._engine is None

    def test_runner_device_selection_cuda(self):
        """Test runner selects CUDA when available."""
        config = GPUMCTSSelfplayConfig(device="cuda:0")
        with patch("torch.cuda.is_available", return_value=True):
            runner = GPUMCTSSelfplayRunner(config)
            assert "cuda" in str(runner.device)

    def test_runner_device_fallback_cpu(self):
        """Test runner falls back to CPU when CUDA unavailable."""
        config = GPUMCTSSelfplayConfig(device="cuda")
        with patch("torch.cuda.is_available", return_value=False):
            runner = GPUMCTSSelfplayRunner(config)
            assert runner.device.type == "cpu"

    def test_move_to_key_with_positions(self, hex8_2p_config):
        """Test move-to-key conversion with positions."""
        runner = GPUMCTSSelfplayRunner(hex8_2p_config)

        # Create mock move with positions
        from app.models import Move, MoveType, Position

        move = MagicMock(spec=Move)
        move.type = MoveType.PLACE_RING
        move.from_pos = Position(x=2, y=3)
        move.to = Position(x=4, y=5)

        # No placement_count attribute
        del move.placement_count

        key = runner._move_to_key(move)
        assert key == "place_ring_2,3_4,5"

    def test_move_to_key_without_from_pos(self, hex8_2p_config):
        """Test move-to-key with None from_pos."""
        runner = GPUMCTSSelfplayRunner(hex8_2p_config)

        from app.models import Move, MoveType, Position

        move = MagicMock(spec=Move)
        move.type = MoveType.PLACE_RING
        move.from_pos = None
        move.to = Position(x=4, y=5)
        del move.placement_count

        key = runner._move_to_key(move)
        assert key == "place_ring_none_4,5"

    def test_move_to_key_with_placement_count(self, hex8_2p_config):
        """Test move-to-key with placement count."""
        runner = GPUMCTSSelfplayRunner(hex8_2p_config)

        from app.models import Move, MoveType, Position

        move = MagicMock(spec=Move)
        move.type = MoveType.PLACE_RING
        move.from_pos = None
        move.to = Position(x=1, y=2)
        move.placement_count = 3

        key = runner._move_to_key(move)
        assert key == "place_ring_none_1,2_3"


class TestGPUMCTSSelfplayRunnerValueAssignment:
    """Tests for value assignment logic."""

    @pytest.fixture
    def runner(self):
        """Create runner for tests."""
        config = GPUMCTSSelfplayConfig(device="cpu")
        return GPUMCTSSelfplayRunner(config)

    def test_assign_values_2p_winner(self, runner):
        """Test value assignment for 2-player game with clear winner."""
        from app.models import GameStatus

        # Create game with samples from both players
        sample1 = SelfplaySample(
            features=np.zeros((1,)),
            globals_vec=np.zeros((1,)),
            policy_indices=np.array([0]),
            policy_values=np.array([1.0]),
            value=0.0,
            player=1,
            move_number=0,
            game_id="test",
        )
        sample2 = SelfplaySample(
            features=np.zeros((1,)),
            globals_vec=np.zeros((1,)),
            policy_indices=np.array([0]),
            policy_values=np.array([1.0]),
            value=0.0,
            player=2,
            move_number=1,
            game_id="test",
        )
        game = GameRecord(game_id="test", samples=[sample1, sample2])

        # Mock final state
        final_state = MagicMock()
        final_state.game_status = GameStatus.COMPLETED
        final_state.winner = 1

        player1 = MagicMock()
        player1.player_number = 1
        player1.territory_spaces = 10
        player1.eliminated_rings = 0

        player2 = MagicMock()
        player2.player_number = 2
        player2.territory_spaces = 5
        player2.eliminated_rings = 2

        final_state.players = [player1, player2]

        runner._assign_values(game, final_state)

        # Player 1 won, should get +1
        assert game.samples[0].value == 1.0
        # Player 2 lost, should get -1
        assert game.samples[1].value == -1.0

    def test_assign_values_incomplete(self, runner):
        """Test value assignment for incomplete/abandoned game."""
        from app.models import GameStatus

        sample = SelfplaySample(
            features=np.zeros((1,)),
            globals_vec=np.zeros((1,)),
            policy_indices=np.array([0]),
            policy_values=np.array([1.0]),
            value=0.5,  # Non-zero initial value
            player=1,
            move_number=0,
            game_id="test",
        )
        game = GameRecord(game_id="test", samples=[sample])

        # Incomplete/abandoned game (not COMPLETED)
        final_state = MagicMock()
        final_state.game_status = GameStatus.ABANDONED  # Not COMPLETED
        final_state.winner = None

        runner._assign_values(game, final_state)

        # Non-completed should set value to 0
        assert game.samples[0].value == 0.0

    def test_assign_values_4p_rankings(self, runner):
        """Test value assignment for 4-player game with rankings."""
        from app.models import GameStatus

        # Create samples for 4 players
        samples = []
        for player in range(1, 5):
            samples.append(
                SelfplaySample(
                    features=np.zeros((1,)),
                    globals_vec=np.zeros((1,)),
                    policy_indices=np.array([0]),
                    policy_values=np.array([1.0]),
                    value=0.0,
                    player=player,
                    move_number=0,
                    game_id="test",
                )
            )
        game = GameRecord(game_id="test", samples=samples)

        # Player 3 wins
        final_state = MagicMock()
        final_state.game_status = GameStatus.COMPLETED
        final_state.winner = 3

        # Rankings: P3=1st, P1=2nd, P4=3rd, P2=4th
        players = []
        for pn, territory, rings in [(1, 15, 0), (2, 5, 3), (3, 0, 0), (4, 10, 1)]:
            p = MagicMock()
            p.player_number = pn
            p.territory_spaces = territory
            p.eliminated_rings = rings
            players.append(p)
        final_state.players = players

        runner._assign_values(game, final_state)

        # 4p: 1st=+1, 2nd=+0.33, 3rd=-0.33, 4th=-1
        # P3 (winner) -> 1st -> +1.0
        assert abs(samples[2].value - 1.0) < 0.01
        # P1 (2nd by territory) -> +0.33
        assert abs(samples[0].value - 0.333) < 0.02
        # P4 (3rd) -> -0.33
        assert abs(samples[3].value - (-0.333)) < 0.02
        # P2 (4th) -> -1.0
        assert abs(samples[1].value - (-1.0)) < 0.01


class TestGPUMCTSSelfplayRunnerExport:
    """Tests for NPZ export functionality."""

    def test_export_to_npz(self, hex8_2p_config, sample_sample):
        """Test NPZ export creates valid file."""
        runner = GPUMCTSSelfplayRunner(hex8_2p_config)
        runner._init_components()

        games = [
            GameRecord(
                game_id="test_001",
                samples=[sample_sample],
                winner=1,
                total_moves=10,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_export.npz")
            count = runner.export_to_npz(output_path, games)

            assert count == 1
            assert os.path.exists(output_path)

            # Verify NPZ contents
            with np.load(output_path) as data:
                assert "features" in data
                assert "globals" in data
                assert "values" in data
                assert "policy_indices" in data
                assert "policy_values" in data
                assert "board_type" in data
                assert "encoder_version" in data

    def test_export_empty_samples(self, hex8_2p_config):
        """Test export with no samples returns 0."""
        runner = GPUMCTSSelfplayRunner(hex8_2p_config)
        runner._init_components()

        games = [GameRecord(game_id="test", samples=[])]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "empty.npz")
            count = runner.export_to_npz(output_path, games)
            assert count == 0
            assert not os.path.exists(output_path)

    def test_export_multiple_games(self, hex8_2p_config, sample_sample):
        """Test export with multiple games."""
        runner = GPUMCTSSelfplayRunner(hex8_2p_config)
        runner._init_components()

        # Create sample2 with different data
        sample2 = SelfplaySample(
            features=np.ones((12, 9, 9), dtype=np.float32),
            globals_vec=np.ones(10, dtype=np.float32),
            policy_indices=np.array([1, 2], dtype=np.int64),
            policy_values=np.array([0.7, 0.3], dtype=np.float32),
            value=-0.5,
            player=2,
            move_number=5,
            game_id="test_game_002",
        )

        games = [
            GameRecord(game_id="g1", samples=[sample_sample]),
            GameRecord(game_id="g2", samples=[sample2]),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "multi.npz")
            count = runner.export_to_npz(output_path, games)

            assert count == 2

            with np.load(output_path) as data:
                assert data["features"].shape[0] == 2
                assert data["values"].shape[0] == 2


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestRunGPUMCTSSelfplay:
    """Tests for run_gpu_mcts_selfplay convenience function."""

    @patch.object(GPUMCTSSelfplayRunner, "run_batch")
    @patch.object(GPUMCTSSelfplayRunner, "_init_components")
    def test_basic_invocation(self, mock_init, mock_run_batch):
        """Test basic function invocation."""
        mock_run_batch.return_value = [GameRecord(game_id="test")]

        with patch("torch.cuda.is_available", return_value=False):
            games = run_gpu_mcts_selfplay(
                board_type="hex8",
                num_players=2,
                num_games=4,
                device="cpu",
            )

        assert len(games) == 1
        mock_run_batch.assert_called_once_with(4)

    @patch.object(GPUMCTSSelfplayRunner, "run_batch")
    @patch.object(GPUMCTSSelfplayRunner, "export_to_npz")
    @patch.object(GPUMCTSSelfplayRunner, "_init_components")
    def test_with_output_path(self, mock_init, mock_export, mock_run_batch):
        """Test function with output path triggers export."""
        mock_run_batch.return_value = [GameRecord(game_id="test")]
        mock_export.return_value = 10

        with patch("torch.cuda.is_available", return_value=False):
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "output.npz")
                games = run_gpu_mcts_selfplay(
                    board_type="hex8",
                    num_players=2,
                    num_games=4,
                    output_path=output_path,
                    device="cpu",
                )

        mock_export.assert_called_once()

    @patch.object(GPUMCTSSelfplayRunner, "run_batch")
    @patch.object(GPUMCTSSelfplayRunner, "_init_components")
    def test_all_board_types(self, mock_init, mock_run_batch):
        """Test function works with all board types."""
        mock_run_batch.return_value = []

        with patch("torch.cuda.is_available", return_value=False):
            for board_type in ["hex8", "hexagonal", "square8", "square19"]:
                games = run_gpu_mcts_selfplay(
                    board_type=board_type,
                    num_games=1,
                    device="cpu",
                )
                assert isinstance(games, list)


# =============================================================================
# Integration-like Tests (with mocked components)
# =============================================================================


class TestGPUMCTSSelfplayIntegration:
    """Integration-style tests with mocked external components."""

    @patch("app.ai.tensor_gumbel_tree.MultiTreeMCTS")
    @patch("app.rules.default_engine.DefaultRulesEngine")
    def test_init_components_board_types(self, mock_engine_cls, mock_mcts_cls):
        """Test component initialization for all board types."""
        mock_engine_cls.return_value = MagicMock()
        mock_mcts_cls.return_value = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            for board_type in ["hex8", "hexagonal", "square8", "square19"]:
                config = GPUMCTSSelfplayConfig(board_type=board_type, device="cpu")
                runner = GPUMCTSSelfplayRunner(config)
                runner._init_components()

                assert runner._mcts is not None
                assert runner._engine is not None

    def test_invalid_board_type(self):
        """Test that invalid board type raises error."""
        config = GPUMCTSSelfplayConfig(board_type="invalid", device="cpu")
        runner = GPUMCTSSelfplayRunner(config)

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(ValueError, match="Unknown board type"):
                runner._init_components()
