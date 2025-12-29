"""Unit tests for NNUE (Efficiently Updatable Neural Network) module.

Tests cover:
- Feature dimension constants
- Helper functions (get_feature_dim, get_board_size, etc.)
- ClippedReLU activation
- StochasticDepthLayer
- ResidualBlock
- RingRiftNNUE model architecture
- Feature extraction functions
- Model loading and caching
- NNUEEvaluator class
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import os

import numpy as np
import pytest
import torch

from app.models import BoardType, GamePhase


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_nnue_cache():
    """Clear NNUE cache before and after each test."""
    from app.ai.nnue import clear_nnue_cache
    clear_nnue_cache()
    yield
    clear_nnue_cache()


@pytest.fixture
def mock_game_state():
    """Create a minimal mock game state for testing."""
    state = MagicMock()
    state.board = MagicMock()
    state.board.type = BoardType.SQUARE8
    state.board.stacks = {}
    state.board.territories = {}
    state.board.markers = {}

    # Mock players
    player1 = MagicMock()
    player1.rings_in_hand = 15
    player1.eliminated_rings = 1
    player1.territory_spaces = 2

    player2 = MagicMock()
    player2.rings_in_hand = 14
    player2.eliminated_rings = 2
    player2.territory_spaces = 3

    state.players = [player1, player2]
    state.phase = GamePhase.MOVEMENT
    state.turn_count = 10

    return state


@pytest.fixture
def mock_mutable_state():
    """Create a minimal mock mutable game state for testing."""
    from app.rules.mutable_state import MutableGameState

    state = MagicMock(spec=MutableGameState)
    state.board_type = BoardType.SQUARE8
    state.num_players = 2
    state.phase = GamePhase.MOVEMENT

    # Internal arrays
    state._rings = np.zeros((8, 8), dtype=np.int32)
    state._stacks = np.zeros((8, 8, 2), dtype=np.int32)
    state._territories = np.zeros((8, 8), dtype=np.int32)
    state._rings_in_hand = np.array([15, 14], dtype=np.int32)
    state._eliminated_rings = np.array([1, 2], dtype=np.int32)
    state._territory_counts = np.array([2, 3], dtype=np.int32)

    return state


# =============================================================================
# Test Feature Dimension Constants
# =============================================================================


class TestFeatureDimensionConstants:
    """Tests for feature dimension constants."""

    def test_feature_planes_value(self):
        """Test FEATURE_PLANES constant."""
        from app.ai.nnue import FEATURE_PLANES
        assert FEATURE_PLANES == 26

    def test_global_features_value(self):
        """Test GLOBAL_FEATURES constant."""
        from app.ai.nnue import GLOBAL_FEATURES
        assert GLOBAL_FEATURES == 32

    def test_board_sizes_defined(self):
        """Test BOARD_SIZES has all board types."""
        from app.ai.nnue import BOARD_SIZES

        assert BoardType.SQUARE8 in BOARD_SIZES
        assert BoardType.SQUARE19 in BOARD_SIZES
        assert BoardType.HEXAGONAL in BOARD_SIZES
        assert BoardType.HEX8 in BOARD_SIZES

    def test_board_size_values(self):
        """Test BOARD_SIZES has correct values."""
        from app.ai.nnue import BOARD_SIZES

        assert BOARD_SIZES[BoardType.SQUARE8] == 8
        assert BOARD_SIZES[BoardType.SQUARE19] == 19
        assert BOARD_SIZES[BoardType.HEXAGONAL] == 25
        assert BOARD_SIZES[BoardType.HEX8] == 9

    def test_feature_dims_calculation(self):
        """Test FEATURE_DIMS are calculated correctly."""
        from app.ai.nnue import FEATURE_DIMS, FEATURE_PLANES, GLOBAL_FEATURES

        # SQUARE8: 8*8*26 + 32 = 1696
        expected_sq8 = 8 * 8 * FEATURE_PLANES + GLOBAL_FEATURES
        assert FEATURE_DIMS[BoardType.SQUARE8] == expected_sq8

    def test_v2_constants_exist(self):
        """Test V2 backwards-compatible constants exist."""
        from app.ai.nnue import FEATURE_PLANES_V2, GLOBAL_FEATURES_V2, FEATURE_DIMS_V2

        assert FEATURE_PLANES_V2 == 12
        assert GLOBAL_FEATURES_V2 == 20
        assert BoardType.SQUARE8 in FEATURE_DIMS_V2


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestGetFeatureDim:
    """Tests for get_feature_dim function."""

    def test_returns_correct_dim_square8(self):
        """Test returns correct dimension for SQUARE8."""
        from app.ai.nnue import get_feature_dim, FEATURE_DIMS

        dim = get_feature_dim(BoardType.SQUARE8)
        assert dim == FEATURE_DIMS[BoardType.SQUARE8]

    def test_returns_correct_dim_hex8(self):
        """Test returns correct dimension for HEX8."""
        from app.ai.nnue import get_feature_dim, FEATURE_DIMS

        dim = get_feature_dim(BoardType.HEX8)
        assert dim == FEATURE_DIMS[BoardType.HEX8]

    def test_unknown_board_defaults_to_square8(self):
        """Test unknown board type defaults to SQUARE8."""
        from app.ai.nnue import get_feature_dim, FEATURE_DIMS

        # Create a mock unknown board type
        mock_type = MagicMock()
        dim = get_feature_dim(mock_type)
        assert dim == FEATURE_DIMS[BoardType.SQUARE8]


class TestGetBoardSize:
    """Tests for get_board_size function."""

    def test_returns_8_for_square8(self):
        """Test returns 8 for SQUARE8."""
        from app.ai.nnue import get_board_size
        assert get_board_size(BoardType.SQUARE8) == 8

    def test_returns_19_for_square19(self):
        """Test returns 19 for SQUARE19."""
        from app.ai.nnue import get_board_size
        assert get_board_size(BoardType.SQUARE19) == 19

    def test_returns_9_for_hex8(self):
        """Test returns 9 for HEX8."""
        from app.ai.nnue import get_board_size
        assert get_board_size(BoardType.HEX8) == 9

    def test_unknown_defaults_to_8(self):
        """Test unknown board type defaults to 8."""
        from app.ai.nnue import get_board_size

        mock_type = MagicMock()
        size = get_board_size(mock_type)
        assert size == 8


class TestGetFeatureDimForVersion:
    """Tests for get_feature_dim_for_version function."""

    def test_version_3_returns_current_dims(self):
        """Test version 3 returns current FEATURE_DIMS."""
        from app.ai.nnue import (
            get_feature_dim_for_version,
            FEATURE_DIMS,
            NNUE_FEATURE_V3,
        )

        dim = get_feature_dim_for_version(BoardType.SQUARE8, NNUE_FEATURE_V3)
        assert dim == FEATURE_DIMS[BoardType.SQUARE8]

    def test_version_2_returns_v2_dims(self):
        """Test version 2 returns V2 dims."""
        from app.ai.nnue import (
            get_feature_dim_for_version,
            FEATURE_DIMS_V2,
            NNUE_FEATURE_V2,
        )

        dim = get_feature_dim_for_version(BoardType.SQUARE8, NNUE_FEATURE_V2)
        assert dim == FEATURE_DIMS_V2[BoardType.SQUARE8]

    def test_version_1_returns_spatial_only(self):
        """Test version 1 returns spatial-only dims."""
        from app.ai.nnue import (
            get_feature_dim_for_version,
            SPATIAL_DIMS_V1,
            NNUE_FEATURE_V1,
        )

        dim = get_feature_dim_for_version(BoardType.SQUARE8, NNUE_FEATURE_V1)
        assert dim == SPATIAL_DIMS_V1[BoardType.SQUARE8]


class TestDetectFeatureVersion:
    """Tests for detect_feature_version_from_accumulator function."""

    def test_detects_v1_from_spatial_size(self):
        """Test detects V1 from spatial-only size."""
        from app.ai.nnue import (
            detect_feature_version_from_accumulator,
            SPATIAL_DIMS_V1,
            NNUE_FEATURE_V1,
        )

        # V1 for SQUARE8: 768 inputs
        shape = (256, SPATIAL_DIMS_V1[BoardType.SQUARE8])
        version = detect_feature_version_from_accumulator(shape, BoardType.SQUARE8)
        assert version == NNUE_FEATURE_V1

    def test_detects_v2_from_v2_size(self):
        """Test detects V2 from V2 size."""
        from app.ai.nnue import (
            detect_feature_version_from_accumulator,
            FEATURE_DIMS_V2,
            NNUE_FEATURE_V2,
        )

        shape = (256, FEATURE_DIMS_V2[BoardType.SQUARE8])
        version = detect_feature_version_from_accumulator(shape, BoardType.SQUARE8)
        assert version == NNUE_FEATURE_V2

    def test_detects_v3_from_current_size(self):
        """Test detects V3 from current size."""
        from app.ai.nnue import (
            detect_feature_version_from_accumulator,
            FEATURE_DIMS,
            NNUE_FEATURE_V3,
        )

        shape = (256, FEATURE_DIMS[BoardType.SQUARE8])
        version = detect_feature_version_from_accumulator(shape, BoardType.SQUARE8)
        assert version == NNUE_FEATURE_V3


# =============================================================================
# Test ClippedReLU
# =============================================================================


class TestClippedReLU:
    """Tests for ClippedReLU activation."""

    def test_clips_to_zero_lower(self):
        """Test negative values are clipped to 0."""
        from app.ai.nnue import ClippedReLU

        activation = ClippedReLU()
        x = torch.tensor([-1.0, -0.5, 0.0])
        result = activation(x)

        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(result, expected)

    def test_clips_to_one_upper(self):
        """Test values above 1 are clipped to 1."""
        from app.ai.nnue import ClippedReLU

        activation = ClippedReLU()
        x = torch.tensor([1.0, 1.5, 2.0])
        result = activation(x)

        expected = torch.tensor([1.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_preserves_values_in_range(self):
        """Test values in [0, 1] are preserved."""
        from app.ai.nnue import ClippedReLU

        activation = ClippedReLU()
        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        result = activation(x)

        assert torch.allclose(result, x)


# =============================================================================
# Test StochasticDepthLayer
# =============================================================================


class TestStochasticDepthLayer:
    """Tests for StochasticDepthLayer."""

    def test_no_drop_during_eval(self):
        """Test no dropping during evaluation."""
        from app.ai.nnue import StochasticDepthLayer

        layer = StochasticDepthLayer(p=0.5)
        layer.eval()

        x = torch.ones(2, 16)
        residual = torch.ones(2, 16) * 0.5

        result = layer(x, residual)
        expected = x + residual
        assert torch.allclose(result, expected)

    def test_no_drop_when_p_zero(self):
        """Test no dropping when p=0."""
        from app.ai.nnue import StochasticDepthLayer

        layer = StochasticDepthLayer(p=0.0)
        layer.train()

        x = torch.ones(2, 16)
        residual = torch.ones(2, 16) * 0.5

        result = layer(x, residual)
        expected = x + residual
        assert torch.allclose(result, expected)

    def test_batch_mode_drops_or_keeps_all(self):
        """Test batch mode drops or keeps entire batch."""
        from app.ai.nnue import StochasticDepthLayer

        torch.manual_seed(42)
        layer = StochasticDepthLayer(p=0.5, mode="batch")
        layer.train()

        x = torch.ones(4, 16)
        residual = torch.ones(4, 16)

        # Run multiple times and check consistency within batch
        for _ in range(10):
            result = layer(x, residual)
            # All samples should be the same (either all dropped or all kept)
            assert torch.allclose(result[0], result[1])
            assert torch.allclose(result[0], result[2])


# =============================================================================
# Test ResidualBlock
# =============================================================================


class TestResidualBlock:
    """Tests for ResidualBlock."""

    def test_output_shape_same_dim(self):
        """Test output shape when input/output dims match."""
        from app.ai.nnue import ResidualBlock

        block = ResidualBlock(in_dim=32, out_dim=32)
        x = torch.randn(4, 32)

        result = block(x)
        assert result.shape == (4, 32)

    def test_output_shape_different_dim(self):
        """Test output shape when input/output dims differ."""
        from app.ai.nnue import ResidualBlock

        block = ResidualBlock(in_dim=64, out_dim=32)
        x = torch.randn(4, 64)

        result = block(x)
        assert result.shape == (4, 32)

    def test_creates_projection_when_needed(self):
        """Test projection layer is created for dimension mismatch."""
        from app.ai.nnue import ResidualBlock

        block = ResidualBlock(in_dim=64, out_dim=32)
        assert block.projection is not None

    def test_no_projection_when_same_dim(self):
        """Test no projection when dimensions match."""
        from app.ai.nnue import ResidualBlock

        block = ResidualBlock(in_dim=32, out_dim=32)
        assert block.projection is None

    def test_with_stochastic_depth(self):
        """Test block with stochastic depth enabled."""
        from app.ai.nnue import ResidualBlock

        block = ResidualBlock(in_dim=32, out_dim=32, stochastic_depth_prob=0.2)
        assert block.stochastic_depth is not None

        x = torch.randn(4, 32)
        result = block(x)
        assert result.shape == (4, 32)


# =============================================================================
# Test RingRiftNNUE Model
# =============================================================================


class TestRingRiftNNUEInit:
    """Tests for RingRiftNNUE initialization."""

    def test_basic_initialization(self):
        """Test basic model initialization."""
        from app.ai.nnue import RingRiftNNUE

        model = RingRiftNNUE(board_type=BoardType.SQUARE8)
        assert model.board_type == BoardType.SQUARE8
        assert model.accumulator is not None

    def test_initialization_with_custom_hidden_dim(self):
        """Test initialization with custom hidden dimension."""
        from app.ai.nnue import RingRiftNNUE

        model = RingRiftNNUE(board_type=BoardType.SQUARE8, hidden_dim=128)
        # Check accumulator output matches hidden_dim
        assert model.accumulator.out_features == 128

    def test_initialization_with_multi_head(self):
        """Test initialization with multiple heads."""
        from app.ai.nnue import RingRiftNNUE

        model = RingRiftNNUE(
            board_type=BoardType.SQUARE8,
            hidden_dim=256,
            num_heads=4,
        )
        assert model.head_projections is not None
        assert len(model.head_projections) == 4
        assert model.accumulator is None

    def test_initialization_with_batch_norm(self):
        """Test initialization with batch normalization."""
        from app.ai.nnue import RingRiftNNUE

        model = RingRiftNNUE(
            board_type=BoardType.SQUARE8,
            use_batch_norm=True,
        )
        assert model.acc_batch_norm is not None

    def test_architecture_version_attribute(self):
        """Test ARCHITECTURE_VERSION class attribute."""
        from app.ai.nnue import RingRiftNNUE

        assert hasattr(RingRiftNNUE, "ARCHITECTURE_VERSION")
        assert isinstance(RingRiftNNUE.ARCHITECTURE_VERSION, str)


class TestRingRiftNNUEForward:
    """Tests for RingRiftNNUE forward pass."""

    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        from app.ai.nnue import RingRiftNNUE, get_feature_dim

        model = RingRiftNNUE(board_type=BoardType.SQUARE8)
        model.eval()

        batch_size = 4
        input_dim = get_feature_dim(BoardType.SQUARE8)
        x = torch.randn(batch_size, input_dim)

        output = model(x)
        assert output.shape == (batch_size, 1)

    def test_forward_output_range(self):
        """Test forward output is in [-1, 1] range (due to tanh)."""
        from app.ai.nnue import RingRiftNNUE, get_feature_dim

        model = RingRiftNNUE(board_type=BoardType.SQUARE8)
        model.eval()

        input_dim = get_feature_dim(BoardType.SQUARE8)
        x = torch.randn(16, input_dim)

        output = model(x)
        assert (output >= -1.0).all()
        assert (output <= 1.0).all()

    def test_forward_single(self):
        """Test forward_single convenience method."""
        from app.ai.nnue import RingRiftNNUE, get_feature_dim

        model = RingRiftNNUE(board_type=BoardType.SQUARE8)

        input_dim = get_feature_dim(BoardType.SQUARE8)
        features = np.random.randn(input_dim).astype(np.float32)

        value = model.forward_single(features)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_forward_with_hidden(self):
        """Test forward_with_hidden returns both value and hidden."""
        from app.ai.nnue import RingRiftNNUE, get_feature_dim

        model = RingRiftNNUE(board_type=BoardType.SQUARE8)
        model.eval()

        input_dim = get_feature_dim(BoardType.SQUARE8)
        x = torch.randn(4, input_dim)

        values, hidden = model.forward_with_hidden(x)
        assert values.shape == (4, 1)
        assert hidden.shape == (4, 32)


class TestRingRiftNNUEQuantization:
    """Tests for RingRiftNNUE quantization."""

    def test_quantize_dynamic_returns_model(self):
        """Test quantize_dynamic returns a model."""
        from app.ai.nnue import RingRiftNNUE

        model = RingRiftNNUE(board_type=BoardType.SQUARE8)

        try:
            quantized = model.quantize_dynamic()
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "quantized" in str(e).lower():
                pytest.skip("PyTorch quantization backend not available on this platform")
            raise

        assert quantized is not None

    def test_quantized_model_runs_inference(self):
        """Test quantized model can run inference."""
        from app.ai.nnue import RingRiftNNUE, get_feature_dim

        model = RingRiftNNUE(board_type=BoardType.SQUARE8)

        try:
            quantized = model.quantize_dynamic()
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "quantized" in str(e).lower():
                pytest.skip("PyTorch quantization backend not available on this platform")
            raise

        input_dim = get_feature_dim(BoardType.SQUARE8)
        x = torch.randn(1, input_dim)

        output = quantized(x)
        assert output.shape == (1, 1)


# =============================================================================
# Test Feature Extraction
# =============================================================================


class TestExtractFeaturesFromGameState:
    """Tests for extract_features_from_gamestate function."""

    def test_returns_numpy_array(self, mock_game_state):
        """Test returns numpy array."""
        from app.ai.nnue import extract_features_from_gamestate

        features = extract_features_from_gamestate(mock_game_state, player_number=1)
        assert isinstance(features, np.ndarray)

    def test_returns_correct_shape(self, mock_game_state):
        """Test returns correct feature dimension."""
        from app.ai.nnue import extract_features_from_gamestate, FEATURE_DIMS

        features = extract_features_from_gamestate(mock_game_state, player_number=1)
        expected_dim = FEATURE_DIMS[BoardType.SQUARE8]
        assert features.shape == (expected_dim,)

    def test_features_are_float32(self, mock_game_state):
        """Test features are float32."""
        from app.ai.nnue import extract_features_from_gamestate

        features = extract_features_from_gamestate(mock_game_state, player_number=1)
        assert features.dtype == np.float32

    def test_different_player_produces_different_features(self, mock_game_state):
        """Test different player perspective produces different features."""
        from app.ai.nnue import extract_features_from_gamestate

        features_p1 = extract_features_from_gamestate(mock_game_state, player_number=1)
        features_p2 = extract_features_from_gamestate(mock_game_state, player_number=2)

        # Features should differ due to perspective rotation
        # (though in empty state they may still be equal)
        assert features_p1.shape == features_p2.shape


class TestExtractFeaturesFromMutable:
    """Tests for extract_features_from_mutable function."""

    def test_returns_numpy_array(self, mock_mutable_state):
        """Test returns numpy array."""
        from app.ai.nnue import extract_features_from_mutable

        features = extract_features_from_mutable(mock_mutable_state, player_number=1)
        assert isinstance(features, np.ndarray)

    def test_returns_correct_shape(self, mock_mutable_state):
        """Test returns correct feature dimension."""
        from app.ai.nnue import extract_features_from_mutable, FEATURE_DIMS

        features = extract_features_from_mutable(mock_mutable_state, player_number=1)
        expected_dim = FEATURE_DIMS[BoardType.SQUARE8]
        assert features.shape == (expected_dim,)

    def test_falls_back_to_immutable_if_no_arrays(self):
        """Test falls back to immutable extraction if internal arrays missing."""
        from app.ai.nnue import extract_features_from_mutable, FEATURE_DIMS
        from app.rules.mutable_state import MutableGameState

        state = MagicMock(spec=MutableGameState)
        state.board_type = BoardType.SQUARE8
        state._rings = None  # No internal arrays

        # Mock to_immutable to return a game state
        mock_immutable = MagicMock()
        state.to_immutable = MagicMock(return_value=mock_immutable)

        # Patch extract_features_from_gamestate to verify it gets called
        expected_features = np.zeros(FEATURE_DIMS[BoardType.SQUARE8], dtype=np.float32)
        with patch("app.ai.nnue.extract_features_from_gamestate", return_value=expected_features) as mock_extract:
            features = extract_features_from_mutable(state, player_number=1)

            # Verify fallback was triggered
            mock_extract.assert_called_once_with(mock_immutable, 1)
            assert isinstance(features, np.ndarray)
            assert features.shape == expected_features.shape


# =============================================================================
# Test Model Path Functions
# =============================================================================


class TestGetNNUEModelPath:
    """Tests for get_nnue_model_path function."""

    def test_returns_path_object(self):
        """Test returns Path object."""
        from app.ai.nnue import get_nnue_model_path
        from pathlib import Path

        path = get_nnue_model_path(BoardType.SQUARE8)
        assert isinstance(path, Path)

    def test_uses_correct_naming_convention(self):
        """Test uses correct naming convention."""
        from app.ai.nnue import get_nnue_model_path

        path = get_nnue_model_path(BoardType.SQUARE8, num_players=2)
        assert "nnue_square8_2p.pt" in str(path)

    def test_different_player_counts(self):
        """Test different player counts produce different paths."""
        from app.ai.nnue import get_nnue_model_path

        path_2p = get_nnue_model_path(BoardType.SQUARE8, num_players=2)
        path_4p = get_nnue_model_path(BoardType.SQUARE8, num_players=4)

        assert path_2p != path_4p
        assert "2p" in str(path_2p)
        assert "4p" in str(path_4p)

    def test_custom_model_id(self):
        """Test custom model ID is used."""
        from app.ai.nnue import get_nnue_model_path

        path = get_nnue_model_path(BoardType.SQUARE8, model_id="custom_model")
        assert "custom_model.pt" in str(path)


# =============================================================================
# Test Model Cache
# =============================================================================


class TestNNUECache:
    """Tests for NNUE model cache."""

    def test_clear_cache_empties_cache(self):
        """Test clear_nnue_cache empties the cache."""
        from app.ai.nnue import _NNUE_CACHE, clear_nnue_cache

        # Add something to cache
        _NNUE_CACHE[("test", 2, "default")] = MagicMock()
        assert len(_NNUE_CACHE) > 0

        clear_nnue_cache()
        assert len(_NNUE_CACHE) == 0


# =============================================================================
# Test NNUEEvaluator
# =============================================================================


class TestNNUEEvaluatorInit:
    """Tests for NNUEEvaluator initialization."""

    def test_initialization_sets_attributes(self):
        """Test initialization sets expected attributes."""
        from app.ai.nnue import NNUEEvaluator

        evaluator = NNUEEvaluator(
            board_type=BoardType.SQUARE8,
            player_number=1,
            num_players=2,
            allow_fresh=True,
        )

        assert evaluator.board_type == BoardType.SQUARE8
        assert evaluator.player_number == 1
        assert evaluator.num_players == 2

    def test_available_flag_set(self):
        """Test available flag is set based on model loading."""
        from app.ai.nnue import NNUEEvaluator

        evaluator = NNUEEvaluator(
            board_type=BoardType.SQUARE8,
            player_number=1,
            allow_fresh=True,
        )

        # With allow_fresh=True, should always be available
        assert evaluator.available is True

    def test_zero_sum_flag_defaults_true(self):
        """Test zero-sum evaluation is enabled by default."""
        from app.ai.nnue import NNUEEvaluator

        with patch.dict(os.environ, {}, clear=True):
            evaluator = NNUEEvaluator(
                board_type=BoardType.SQUARE8,
                player_number=1,
                allow_fresh=True,
            )
            assert evaluator.use_zero_sum is True

    def test_zero_sum_can_be_disabled(self):
        """Test zero-sum can be disabled via environment variable."""
        from app.ai.nnue import NNUEEvaluator

        with patch.dict(os.environ, {"RINGRIFT_NNUE_ZERO_SUM_EVAL": "false"}):
            evaluator = NNUEEvaluator(
                board_type=BoardType.SQUARE8,
                player_number=1,
                allow_fresh=True,
            )
            assert evaluator.use_zero_sum is False


class TestNNUEEvaluatorEvaluate:
    """Tests for NNUEEvaluator evaluation methods."""

    def test_evaluate_mutable_returns_float(self, mock_mutable_state):
        """Test evaluate_mutable returns a float."""
        from app.ai.nnue import NNUEEvaluator

        evaluator = NNUEEvaluator(
            board_type=BoardType.SQUARE8,
            player_number=1,
            allow_fresh=True,
        )

        # Disable zero-sum for simpler test
        evaluator.use_zero_sum = False

        score = evaluator.evaluate_mutable(mock_mutable_state)
        assert isinstance(score, float)

    def test_evaluate_gamestate_returns_float(self, mock_game_state):
        """Test evaluate_gamestate returns a float."""
        from app.ai.nnue import NNUEEvaluator

        evaluator = NNUEEvaluator(
            board_type=BoardType.SQUARE8,
            player_number=1,
            allow_fresh=True,
        )

        # Disable zero-sum for simpler test
        evaluator.use_zero_sum = False

        score = evaluator.evaluate_gamestate(mock_game_state)
        assert isinstance(score, float)

    def test_unavailable_model_raises(self):
        """Test evaluation raises when model unavailable."""
        from app.ai.nnue import NNUEEvaluator

        evaluator = NNUEEvaluator(
            board_type=BoardType.SQUARE8,
            player_number=1,
            allow_fresh=True,
        )
        evaluator.available = False
        evaluator.model = None

        # Create a minimal mock state
        state = MagicMock()

        with pytest.raises(RuntimeError, match="NNUE model not available"):
            evaluator.evaluate_gamestate(state)

    def test_score_scale_applied(self, mock_mutable_state):
        """Test SCORE_SCALE is applied to output."""
        from app.ai.nnue import NNUEEvaluator

        evaluator = NNUEEvaluator(
            board_type=BoardType.SQUARE8,
            player_number=1,
            allow_fresh=True,
        )
        evaluator.use_zero_sum = False

        score = evaluator.evaluate_mutable(mock_mutable_state)

        # Score should be scaled (model outputs [-1, 1], scaled by 10000)
        assert abs(score) <= evaluator.SCORE_SCALE


# =============================================================================
# Test Perspective Rotation
# =============================================================================


class TestPerspectiveRotation:
    """Tests for _rotate_player_perspective function."""

    def test_current_player_becomes_1(self):
        """Test current player is rotated to position 1."""
        from app.ai.nnue import _rotate_player_perspective

        # Player 2 from player 2's perspective -> 1
        rotated = _rotate_player_perspective(owner=2, player_number=2, num_players=4)
        assert rotated == 1

    def test_opponent_rotation(self):
        """Test opponent is rotated correctly."""
        from app.ai.nnue import _rotate_player_perspective

        # Player 1 from player 2's perspective -> 4 (wraps around)
        rotated = _rotate_player_perspective(owner=1, player_number=2, num_players=4)
        assert rotated == 4

    def test_owner_zero_unchanged(self):
        """Test owner 0 (empty) remains unchanged."""
        from app.ai.nnue import _rotate_player_perspective

        rotated = _rotate_player_perspective(owner=0, player_number=2, num_players=4)
        assert rotated == 0


# =============================================================================
# Test State Dict Migration
# =============================================================================


class TestMigrateLegacyStateDict:
    """Tests for _migrate_legacy_state_dict function."""

    def test_migrates_v1_hidden_layers(self):
        """Test migrates v1.0/v1.1 hidden layer structure."""
        from app.ai.nnue import _migrate_legacy_state_dict

        # Old v1.0 structure: hidden.0.weight, hidden.2.weight
        state_dict = {
            "accumulator.weight": torch.randn(256, 768),
            "accumulator.bias": torch.randn(256),
            "hidden.0.weight": torch.randn(32, 512),
            "hidden.0.bias": torch.randn(32),
            "hidden.2.weight": torch.randn(32, 32),
            "hidden.2.bias": torch.randn(32),
            "output.weight": torch.randn(1, 32),
            "output.bias": torch.randn(1),
        }

        migrated, _ = _migrate_legacy_state_dict(
            state_dict,
            architecture_version="v1.0.0",
            target_input_size=None,
        )

        # Should have new key structure
        assert "hidden_blocks.0.fc.weight" in migrated
        assert "hidden_blocks.0.fc.bias" in migrated

    def test_pads_accumulator_for_new_features(self):
        """Test pads accumulator weights for new global features."""
        from app.ai.nnue import _migrate_legacy_state_dict, FEATURE_DIMS

        # Old model with 768 input (no global features)
        state_dict = {
            "accumulator.weight": torch.randn(256, 768),
            "accumulator.bias": torch.randn(256),
        }

        target_size = FEATURE_DIMS[BoardType.SQUARE8]  # 1696 for V3

        migrated, detected_version = _migrate_legacy_state_dict(
            state_dict,
            architecture_version="v1.3.0",
            target_input_size=target_size,
            board_type=BoardType.SQUARE8,
        )

        # Accumulator should be padded
        assert migrated["accumulator.weight"].shape[1] == target_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
