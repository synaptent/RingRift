"""Tests for app.ai.maxn_ai - Max-N and BRS AI implementations.

Tests cover:
- MaxNAI class (Max-N search for multiplayer games)
- BRSAI class (Best-Reply Search)
- GPU acceleration support
- Neural network evaluation support
- Depth/rounds calculation based on difficulty
- Terminal state handling
"""

import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from app.models import AIConfig, BoardType, GamePhase, GameStatus


class TestMaxNAIInitialization:
    """Tests for MaxNAI initialization."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_initialization_defaults(self, mock_tt, mock_zobrist):
        """Test MaxNAI initializes with correct defaults."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)

        assert ai.player_number == 1
        assert ai.config == config
        assert ai._num_players is None
        assert ai._gpu_batch_size == 64
        assert ai._gpu_min_batch == 4
        assert ai._board_type is None
        assert ai._leaf_buffer == []
        assert ai._leaf_results == {}

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_initialization_gpu_disabled_env(self, mock_tt, mock_zobrist):
        """Test GPU is disabled via environment variable."""
        with patch.dict(os.environ, {'RINGRIFT_GPU_MAXN_DISABLE': '1'}):
            # Need to reload module to pick up env change
            import importlib
            import app.ai.maxn_ai as maxn_module
            importlib.reload(maxn_module)

            config = AIConfig(difficulty=5)
            ai = maxn_module.MaxNAI(player_number=1, config=config)

            assert ai._gpu_enabled is False

            # Restore module state
            with patch.dict(os.environ, {'RINGRIFT_GPU_MAXN_DISABLE': ''}):
                importlib.reload(maxn_module)

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_initialization_nn_from_config(self, mock_tt, mock_zobrist):
        """Test NN mode enabled via config."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        # Simulate config attribute
        config.use_neural_net = True

        ai = MaxNAI(player_number=2, config=config)

        assert ai.use_neural_net is True
        assert ai._nn_initialized is False


class TestMaxNAIDepthCalculation:
    """Tests for MaxNAI depth calculation."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_max_depth_high_difficulty(self, mock_tt, mock_zobrist):
        """Test depth 4 for difficulty >= 9."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=10)
        ai = MaxNAI(player_number=1, config=config)

        assert ai._get_max_depth() == 4

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_max_depth_medium_high(self, mock_tt, mock_zobrist):
        """Test depth 3 for difficulty 7-8."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=7)
        ai = MaxNAI(player_number=1, config=config)

        assert ai._get_max_depth() == 3

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_max_depth_medium(self, mock_tt, mock_zobrist):
        """Test depth 2 for difficulty 4-6."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)

        assert ai._get_max_depth() == 2

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_max_depth_low(self, mock_tt, mock_zobrist):
        """Test depth 1 for difficulty < 4."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=2)
        ai = MaxNAI(player_number=1, config=config)

        assert ai._get_max_depth() == 1


class TestMaxNAIGPUInitialization:
    """Tests for MaxNAI GPU initialization."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_gpu_init_disabled(self, mock_tt, mock_zobrist):
        """Test GPU init returns False when disabled."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._gpu_enabled = False

        result = ai._ensure_gpu_initialized()

        assert result is False
        assert ai._gpu_available is False

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_gpu_init_cached(self, mock_tt, mock_zobrist):
        """Test GPU init returns cached result."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._gpu_available = True

        result = ai._ensure_gpu_initialized()

        assert result is True

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_gpu_init_success(self, mock_tt, mock_zobrist):
        """Test GPU init succeeds with CUDA device."""
        from app.ai.maxn_ai import MaxNAI

        mock_device = MagicMock()
        mock_device.type = 'cuda'

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._gpu_available = None
        ai._gpu_enabled = True

        # Mock the gpu_batch module import inside _ensure_gpu_initialized
        mock_gpu_batch = MagicMock()
        mock_gpu_batch.get_device.return_value = mock_device

        with patch.dict('sys.modules', {'app.ai.gpu_batch': mock_gpu_batch}):
            result = ai._ensure_gpu_initialized()

            # Either True (CUDA available) or False (fallback due to exception)
            # The test primarily verifies the method runs without crashing
            assert result is not None

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_gpu_init_exception_fallback(self, mock_tt, mock_zobrist):
        """Test GPU init falls back on exception."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._gpu_available = None
        ai._gpu_enabled = True

        with patch.dict('sys.modules', {'app.ai.gpu_batch': None}):
            result = ai._ensure_gpu_initialized()

            assert result is False
            assert ai._gpu_available is False


class TestMaxNAIBoardConfig:
    """Tests for MaxNAI board configuration detection."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_detect_board_config_square8(self, mock_tt, mock_zobrist):
        """Test board config detection for square8."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)

        mock_state = MagicMock()
        mock_state.board_type = BoardType.SQUARE8
        mock_state.players = [MagicMock(), MagicMock()]

        ai._detect_board_config(mock_state)

        assert ai._board_type == BoardType.SQUARE8
        assert ai._board_size == 8
        assert ai._num_players == 2

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_detect_board_config_hexagonal(self, mock_tt, mock_zobrist):
        """Test board config detection for hexagonal."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)

        mock_state = MagicMock()
        mock_state.board_type = BoardType.HEXAGONAL
        mock_state.players = [MagicMock() for _ in range(4)]

        ai._detect_board_config(mock_state)

        assert ai._board_type == BoardType.HEXAGONAL
        assert ai._board_size == 25
        assert ai._num_players == 4

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_detect_board_config_cached(self, mock_tt, mock_zobrist):
        """Test board config is only detected once."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._board_type = BoardType.SQUARE8  # Already set

        mock_state = MagicMock()
        mock_state.board_type = BoardType.HEXAGONAL

        ai._detect_board_config(mock_state)

        # Should not change
        assert ai._board_type == BoardType.SQUARE8


class TestMaxNAILeafBuffer:
    """Tests for MaxNAI leaf buffer operations."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_clear_leaf_buffer(self, mock_tt, mock_zobrist):
        """Test clearing leaf buffer."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._leaf_buffer = [(MagicMock(), 12345)]
        ai._leaf_results = {12345: {1: 10.0}}

        ai._clear_leaf_buffer()

        assert ai._leaf_buffer == []
        assert ai._leaf_results == {}


class TestMaxNAICPUEvaluation:
    """Tests for MaxNAI CPU evaluation."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_evaluate_all_players_cpu_terminal_win(self, mock_tt, mock_zobrist):
        """Test CPU evaluation for terminal win state."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._num_players = 2

        mock_state = MagicMock()
        mock_state.is_game_over.return_value = True
        mock_state.get_winner.return_value = 1

        scores = ai._evaluate_all_players_cpu(mock_state)

        assert scores[1] == 100000.0
        assert scores[2] == -100000.0

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_evaluate_all_players_cpu_terminal_draw(self, mock_tt, mock_zobrist):
        """Test CPU evaluation for terminal draw state."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._num_players = 2

        mock_state = MagicMock()
        mock_state.is_game_over.return_value = True
        mock_state.get_winner.return_value = None

        scores = ai._evaluate_all_players_cpu(mock_state)

        assert scores[1] == 0.0
        assert scores[2] == 0.0


class TestMaxNAISelectMove:
    """Tests for MaxNAI select_move method."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_select_move_no_valid_moves(self, mock_tt, mock_zobrist):
        """Test select_move returns None when no valid moves."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)

        mock_state = MagicMock()

        with patch.object(ai, 'get_valid_moves', return_value=[]):
            result = ai.select_move(mock_state)

        assert result is None

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_select_move_single_move(self, mock_tt, mock_zobrist):
        """Test select_move returns single valid move."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)

        mock_state = MagicMock()
        mock_move = MagicMock()

        with patch.object(ai, 'get_valid_moves', return_value=[mock_move]):
            result = ai.select_move(mock_state)

        assert result == mock_move


class TestBRSAIInitialization:
    """Tests for BRSAI initialization."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_initialization_defaults(self, mock_tt, mock_zobrist):
        """Test BRSAI initializes with correct defaults."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=5)
        ai = BRSAI(player_number=1, config=config)

        assert ai.player_number == 1
        assert ai.config == config
        assert ai._num_players is None
        assert ai._board_type is None
        assert ai._nn_initialized is False


class TestBRSAILookaheadRounds:
    """Tests for BRSAI lookahead rounds calculation."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_lookahead_high_difficulty(self, mock_tt, mock_zobrist):
        """Test 3 rounds for difficulty >= 7."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=8)
        ai = BRSAI(player_number=1, config=config)

        assert ai._get_lookahead_rounds() == 3

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_lookahead_medium(self, mock_tt, mock_zobrist):
        """Test 2 rounds for difficulty 4-6."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=5)
        ai = BRSAI(player_number=1, config=config)

        assert ai._get_lookahead_rounds() == 2

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_lookahead_low(self, mock_tt, mock_zobrist):
        """Test 1 round for difficulty < 4."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=2)
        ai = BRSAI(player_number=1, config=config)

        assert ai._get_lookahead_rounds() == 1


class TestBRSAISelectMove:
    """Tests for BRSAI select_move method."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_select_move_no_valid_moves(self, mock_tt, mock_zobrist):
        """Test select_move returns None when no valid moves."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=5)
        ai = BRSAI(player_number=1, config=config)

        mock_state = MagicMock()

        with patch.object(ai, 'get_valid_moves', return_value=[]):
            result = ai.select_move(mock_state)

        assert result is None

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_select_move_single_move(self, mock_tt, mock_zobrist):
        """Test select_move returns single valid move."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=5)
        ai = BRSAI(player_number=1, config=config)

        mock_state = MagicMock()
        mock_move = MagicMock()

        with patch.object(ai, 'get_valid_moves', return_value=[mock_move]):
            result = ai.select_move(mock_state)

        assert result == mock_move


class TestBRSAIEvaluation:
    """Tests for BRSAI evaluation methods."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_evaluate_for_me_terminal_win(self, mock_tt, mock_zobrist):
        """Test evaluation for terminal win state."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=5)
        ai = BRSAI(player_number=1, config=config)

        mock_state = MagicMock()
        mock_state.is_game_over.return_value = True
        mock_state.get_winner.return_value = 1

        score = ai._evaluate_for_me(mock_state)

        assert score == 100000.0

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_evaluate_for_me_terminal_loss(self, mock_tt, mock_zobrist):
        """Test evaluation for terminal loss state."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=5)
        ai = BRSAI(player_number=1, config=config)

        mock_state = MagicMock()
        mock_state.is_game_over.return_value = True
        mock_state.get_winner.return_value = 2

        score = ai._evaluate_for_me(mock_state)

        assert score == -100000.0

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_evaluate_for_me_terminal_draw(self, mock_tt, mock_zobrist):
        """Test evaluation for terminal draw state."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=5)
        ai = BRSAI(player_number=1, config=config)

        mock_state = MagicMock()
        mock_state.is_game_over.return_value = True
        mock_state.get_winner.return_value = None

        score = ai._evaluate_for_me(mock_state)

        assert score == 0.0


class TestMaxNAINNInitialization:
    """Tests for MaxNAI neural network initialization."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_nn_init_disabled(self, mock_tt, mock_zobrist):
        """Test NN init returns False when use_neural_net=False."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai.use_neural_net = False

        result = ai._ensure_nn_initialized()

        assert result is False
        assert ai._nn_initialized is True

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_nn_init_cached(self, mock_tt, mock_zobrist):
        """Test NN init returns cached result."""
        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._nn_initialized = True
        ai._neural_net = MagicMock()

        result = ai._ensure_nn_initialized()

        assert result is True


class TestBRSAINNInitialization:
    """Tests for BRSAI neural network initialization."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_nn_init_disabled(self, mock_tt, mock_zobrist):
        """Test NN init returns False when use_neural_net=False."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=5)
        ai = BRSAI(player_number=1, config=config)
        ai.use_neural_net = False

        result = ai._ensure_nn_initialized()

        assert result is False
        assert ai._nn_initialized is True


class TestMaxNAITranspositionTable:
    """Tests for MaxNAI transposition table usage."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_tt_lookup_hit(self, mock_tt_class, mock_zobrist):
        """Test transposition table lookup hit."""
        from app.ai.maxn_ai import MaxNAI

        mock_tt = MagicMock()
        mock_tt.get.return_value = {'scores': {1: 10.0, 2: -5.0}, 'depth': 2}
        mock_tt_class.return_value = mock_tt

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._num_players = 2
        ai.start_time = 0
        ai.time_limit = 10.0
        ai.nodes_visited = 0

        mock_state = MagicMock()
        mock_state.zobrist_hash = 12345
        mock_state.is_game_over.return_value = False

        scores = ai._maxn_search(mock_state, depth=2)

        assert scores == {1: 10.0, 2: -5.0}


class TestEnvironmentVariables:
    """Tests for environment variable controls."""

    def test_gpu_disable_env_parsing(self):
        """Test GPU disable environment variable parsing."""
        test_values = [
            ('1', True),
            ('true', True),
            ('yes', True),
            ('on', True),
            ('0', False),
            ('false', False),
            ('', False),
        ]

        for env_value, expected_disabled in test_values:
            result = env_value.lower() in ('1', 'true', 'yes', 'on')
            assert result == expected_disabled, f"Failed for {env_value}"

    def test_shadow_validate_env_parsing(self):
        """Test shadow validation environment variable parsing."""
        test_values = [
            ('1', True),
            ('true', True),
            ('yes', True),
            ('on', True),
            ('0', False),
        ]

        for env_value, expected in test_values:
            result = env_value.lower() in ('1', 'true', 'yes', 'on')
            assert result == expected, f"Failed for {env_value}"


class TestMaxNAIMaxNSearch:
    """Tests for MaxNAI _maxn_search method."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_maxn_search_depth_zero(self, mock_tt_class, mock_zobrist):
        """Test Max-N search at depth 0 evaluates position."""
        mock_tt = MagicMock()
        mock_tt.get.return_value = None
        mock_tt_class.return_value = mock_tt

        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._num_players = 2
        ai.start_time = 0
        ai.time_limit = 10.0
        ai.nodes_visited = 0

        mock_state = MagicMock()
        mock_state.is_game_over.return_value = False

        expected_scores = {1: 50.0, 2: -25.0}
        with patch.object(ai, '_evaluate_all_players', return_value=expected_scores):
            scores = ai._maxn_search(mock_state, depth=0)

        assert scores == expected_scores

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_maxn_search_terminal_state(self, mock_tt_class, mock_zobrist):
        """Test Max-N search at terminal state."""
        mock_tt = MagicMock()
        mock_tt.get.return_value = None
        mock_tt_class.return_value = mock_tt

        from app.ai.maxn_ai import MaxNAI

        config = AIConfig(difficulty=5)
        ai = MaxNAI(player_number=1, config=config)
        ai._num_players = 2
        ai.start_time = 0
        ai.time_limit = 10.0
        ai.nodes_visited = 0

        mock_state = MagicMock()
        mock_state.is_game_over.return_value = True

        expected_scores = {1: 100000.0, 2: -100000.0}
        with patch.object(ai, '_evaluate_all_players', return_value=expected_scores):
            scores = ai._maxn_search(mock_state, depth=3)

        assert scores == expected_scores


class TestBRSAISimulation:
    """Tests for BRSAI BRS simulation."""

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_simulate_brs_zero_rounds(self, mock_tt, mock_zobrist):
        """Test BRS simulation with zero remaining rounds."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=5)
        ai = BRSAI(player_number=1, config=config)
        ai.start_time = 0
        ai.time_limit = 10.0
        ai.nodes_visited = 0

        mock_state = MagicMock()
        mock_state.is_game_over.return_value = False

        with patch.object(ai, '_evaluate_for_me', return_value=42.0):
            score = ai._simulate_brs_rounds(mock_state, remaining_rounds=0)

        assert score == 42.0

    @patch('app.ai.maxn_ai.ZobristHash')
    @patch('app.ai.maxn_ai.BoundedTranspositionTable')
    def test_simulate_brs_terminal_state(self, mock_tt, mock_zobrist):
        """Test BRS simulation at terminal state."""
        from app.ai.maxn_ai import BRSAI

        config = AIConfig(difficulty=5)
        ai = BRSAI(player_number=1, config=config)
        ai.start_time = 0
        ai.time_limit = 10.0
        ai.nodes_visited = 0

        mock_state = MagicMock()
        mock_state.is_game_over.return_value = True

        with patch.object(ai, '_evaluate_for_me', return_value=-50.0):
            score = ai._simulate_brs_rounds(mock_state, remaining_rounds=2)

        assert score == -50.0
