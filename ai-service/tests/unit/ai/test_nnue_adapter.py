"""Unit tests for NNUE adapter module.

Tests the NNUEMCTSAdapter and NNUEWithPolicyAdapter classes that enable
NNUE models to be used with MCTS/Gumbel search algorithms.

December 2025: Created for Unified AI Evaluation Architecture.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from app.models import BoardType, GameState, Position


class TestPolicyFromValueConfig:
    """Tests for PolicyFromValueConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.ai.nnue_adapter import PolicyFromValueConfig

        config = PolicyFromValueConfig()
        assert config.temperature == 1.0
        assert config.use_softmax is True
        assert config.min_logit == -10.0
        assert config.max_logit == 10.0
        assert config.batch_child_eval is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from app.ai.nnue_adapter import PolicyFromValueConfig

        config = PolicyFromValueConfig(
            temperature=0.5,
            use_softmax=False,
            min_logit=-5.0,
            max_logit=5.0,
            batch_child_eval=False,
        )
        assert config.temperature == 0.5
        assert config.use_softmax is False
        assert config.min_logit == -5.0
        assert config.max_logit == 5.0
        assert config.batch_child_eval is False


class TestNNUEMCTSAdapter:
    """Tests for NNUEMCTSAdapter (value-only NNUE)."""

    @pytest.fixture
    def mock_nnue_model(self):
        """Create a mock NNUE model."""
        model = Mock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.eval.return_value = None

        # Mock forward to return values
        def mock_forward(x):
            batch_size = x.shape[0]
            return torch.zeros(batch_size, 1)

        model.forward = mock_forward
        return model

    @pytest.fixture
    def mock_game_state(self):
        """Create a mock game state."""
        state = Mock(spec=GameState)
        state.current_player = 1
        state.board = Mock()
        state.board.type = BoardType.HEX8

        # Mock board dimensions
        state.board.width = 9
        state.board.height = 9
        return state

    def test_initialization(self, mock_nnue_model):
        """Test adapter initialization."""
        from app.ai.nnue_adapter import NNUEMCTSAdapter

        adapter = NNUEMCTSAdapter(
            nnue_model=mock_nnue_model,
            board_type=BoardType.HEX8,
            num_players=2,
        )

        assert adapter.nnue == mock_nnue_model
        assert adapter.board_type == BoardType.HEX8
        assert adapter.num_players == 2
        assert adapter._device == torch.device("cpu")

    def test_device_detection(self):
        """Test device detection from model parameters."""
        from app.ai.nnue_adapter import NNUEMCTSAdapter

        # Mock model on CUDA (if available)
        model = Mock()
        cuda_tensor = torch.zeros(1)
        if torch.cuda.is_available():
            cuda_tensor = cuda_tensor.cuda()
        model.parameters.return_value = iter([cuda_tensor])

        adapter = NNUEMCTSAdapter(
            nnue_model=model,
            board_type=BoardType.SQUARE8,
        )

        assert adapter._device == cuda_tensor.device

    def test_version(self):
        """Test adapter version constant."""
        from app.ai.nnue_adapter import NNUEMCTSAdapter

        assert hasattr(NNUEMCTSAdapter, "ADAPTER_VERSION")
        assert NNUEMCTSAdapter.ADAPTER_VERSION == "1.0.0"

    @patch("app.ai.nnue.extract_features_from_gamestate")
    def test_evaluate_values(self, mock_extract, mock_nnue_model, mock_game_state):
        """Test value evaluation."""
        from app.ai.nnue_adapter import NNUEMCTSAdapter

        mock_extract.return_value = np.zeros(100)

        adapter = NNUEMCTSAdapter(
            nnue_model=mock_nnue_model,
            board_type=BoardType.HEX8,
        )

        values = adapter._evaluate_values([mock_game_state], value_head=1)

        assert len(values) == 1
        assert isinstance(values[0], float)

    @patch("app.ai.neural_net.get_action_encoder")
    def test_encode_move(self, mock_get_encoder, mock_nnue_model, mock_game_state):
        """Test move encoding."""
        from app.ai.nnue_adapter import NNUEMCTSAdapter

        # Setup mock action encoder
        mock_encoder = Mock()
        mock_encoder.encode_move.return_value = 42  # Return a valid move index
        mock_get_encoder.return_value = mock_encoder

        adapter = NNUEMCTSAdapter(
            nnue_model=mock_nnue_model,
            board_type=BoardType.HEX8,
        )

        # Create a mock move
        mock_move = Mock()
        mock_move.from_pos = Position(x=0, y=0)
        mock_move.to = Position(x=1, y=1)

        # encode_move should return an integer
        idx = adapter.encode_move(mock_move, mock_game_state.board)
        assert isinstance(idx, int)
        assert idx == 42


class TestNNUEWithPolicyAdapter:
    """Tests for NNUEWithPolicyAdapter (NNUE with policy head)."""

    @pytest.fixture
    def mock_nnue_policy_model(self):
        """Create a mock NNUE model with policy head."""
        model = Mock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.eval.return_value = None
        model.board_size = 9

        # Mock forward to return value + from/to logits
        def mock_forward(x, return_policy=False):
            batch_size = x.shape[0]
            if return_policy:
                return (
                    torch.zeros(batch_size, 1),  # value
                    torch.zeros(batch_size, 81),  # from_logits (9x9)
                    torch.zeros(batch_size, 81),  # to_logits (9x9)
                )
            return torch.zeros(batch_size, 1)

        model.forward = mock_forward
        return model

    def test_initialization(self, mock_nnue_policy_model):
        """Test adapter initialization."""
        from app.ai.nnue_adapter import NNUEWithPolicyAdapter

        adapter = NNUEWithPolicyAdapter(
            nnue_policy_model=mock_nnue_policy_model,
            board_type=BoardType.HEX8,
            num_players=2,
        )

        assert adapter.nnue == mock_nnue_policy_model
        assert adapter.board_type == BoardType.HEX8
        assert adapter.board_size == 9

    def test_version(self):
        """Test adapter version constant."""
        from app.ai.nnue_adapter import NNUEWithPolicyAdapter

        assert hasattr(NNUEWithPolicyAdapter, "ADAPTER_VERSION")
        assert NNUEWithPolicyAdapter.ADAPTER_VERSION == "1.0.0"


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_wrap_nnue_value_only(self):
        """Test wrapping a value-only NNUE model."""
        from app.ai.nnue_adapter import wrap_nnue_value_only, NNUEMCTSAdapter

        mock_model = Mock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        adapter = wrap_nnue_value_only(
            nnue_model=mock_model,
            board_type=BoardType.SQUARE8,
            num_players=2,
            temperature=0.5,
        )

        assert isinstance(adapter, NNUEMCTSAdapter)
        assert adapter.config.temperature == 0.5


class TestAITypeIntegration:
    """Tests for AIType enum integration."""

    def test_nnue_ai_types_exist(self):
        """Test that NNUE AI types are defined."""
        from app.models.core import AIType

        assert hasattr(AIType, "NNUE_GUMBEL")
        assert hasattr(AIType, "NNUE_MCTS")
        assert hasattr(AIType, "NNUE_BRS")
        assert hasattr(AIType, "NNUE_MAXN")

    def test_nnue_ai_type_values(self):
        """Test NNUE AI type string values."""
        from app.models.core import AIType

        assert AIType.NNUE_GUMBEL.value == "nnue_gumbel"
        assert AIType.NNUE_MCTS.value == "nnue_mcts"
        assert AIType.NNUE_BRS.value == "nnue_brs"
        assert AIType.NNUE_MAXN.value == "nnue_maxn"


class TestNNUESearchAI:
    """Tests for NNUE search AI classes."""

    def test_nnue_gumbel_ai_creation(self):
        """Test NNUEGumbelAI can be created."""
        from app.ai.nnue_search_ai import NNUEGumbelAI
        from app.models import AIConfig

        config = AIConfig(difficulty=8)
        ai = NNUEGumbelAI(player_number=1, config=config)

        assert ai.player_number == 1
        assert ai.AI_NAME == "NNUE_GUMBEL"

    def test_nnue_mcts_ai_creation(self):
        """Test NNUEMCTSAI can be created."""
        from app.ai.nnue_search_ai import NNUEMCTSAI
        from app.models import AIConfig

        config = AIConfig(difficulty=7)
        ai = NNUEMCTSAI(player_number=2, config=config)

        assert ai.player_number == 2
        assert ai.AI_NAME == "NNUE_MCTS"

    def test_nnue_brs_ai_creation(self):
        """Test NNUEBRSAI can be created."""
        from app.ai.nnue_search_ai import NNUEBRSAI
        from app.models import AIConfig

        config = AIConfig(difficulty=5)
        ai = NNUEBRSAI(player_number=1, config=config, num_players=3)

        assert ai.player_number == 1
        assert ai._num_players >= 3
        assert ai.AI_NAME == "NNUE_BRS"

    def test_nnue_maxn_ai_creation(self):
        """Test NNUEMaxNAI can be created."""
        from app.ai.nnue_search_ai import NNUEMaxNAI
        from app.models import AIConfig

        config = AIConfig(difficulty=6)
        ai = NNUEMaxNAI(player_number=1, config=config, num_players=4)

        assert ai.player_number == 1
        assert ai._num_players == 4
        assert ai.AI_NAME == "NNUE_MAXN"


class TestFactoryIntegration:
    """Tests for factory integration with NNUE AI types."""

    def test_factory_recognizes_nnue_types(self):
        """Test that AIFactory._get_ai_class handles NNUE types."""
        from app.ai.factory import AIFactory
        from app.models.core import AIType

        # These should not raise ValueError
        for ai_type in [
            AIType.NNUE_GUMBEL,
            AIType.NNUE_MCTS,
            AIType.NNUE_BRS,
            AIType.NNUE_MAXN,
        ]:
            ai_class = AIFactory._get_ai_class(ai_type)
            assert ai_class is not None
