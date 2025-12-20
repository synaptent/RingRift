"""Tests for app/ai/neural_net.py - Neural network AI implementation.

Tests cover:
- Policy encoding constants and sizes
- Move encoding/decoding for all board types
- Board-specific policy layouts
- NeuralNetAI initialization and configuration
- Feature extraction basics
- CNN architecture forward passes (with mocking)
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Optional

from app.ai.neural_net import (
    # Constants
    POLICY_SIZE_8x8,
    POLICY_SIZE_19x19,
    POLICY_SIZE_HEX8,
    P_HEX,
    MAX_N,
    MAX_PLAYERS,
    INVALID_MOVE_INDEX,
    # Square8 policy layout constants
    SQUARE8_PLACEMENT_SPAN,
    SQUARE8_MOVEMENT_BASE,
    SQUARE8_LINE_FORM_BASE,
    SQUARE8_TERRITORY_CLAIM_BASE,
    SQUARE8_SKIP_PLACEMENT_IDX,
    SQUARE8_SWAP_SIDES_IDX,
    # Hex constants
    HEX_BOARD_SIZE,
    HEX8_BOARD_SIZE,
    NUM_HEX_DIRS,
    HEX_DIRS,
    # Functions
    get_policy_size_for_board,
    get_spatial_size_for_board,
    encode_move_for_board,
    decode_move_for_board,
    _encode_move_square8,
    _decode_move_square8,
    # Dataclass
    DecodedPolicyIndex,
)
from app.models import BoardType, Move, MoveType, Position, BoardState


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockAIConfig:
    """Mock AI config for testing."""
    nn_model_id: Optional[str] = None
    allow_fresh_weights: bool = True
    history_length: int = 3
    feature_version: int = 1
    nn_state_dict: Optional[dict] = None
    rng_seed: Optional[int] = 42
    board_type: Optional[str] = None
    num_players: int = 2


@pytest.fixture
def mock_config():
    """Create a mock AI config for testing."""
    return MockAIConfig()


@pytest.fixture
def square8_board():
    """Create a mock BoardState for square8 board."""
    board = MagicMock(spec=BoardState)
    board.type = BoardType.SQUARE8
    board.width = 8
    board.height = 8
    board.positions = {
        Position(x=x, y=y): MagicMock() for x in range(8) for y in range(8)
    }
    return board


@pytest.fixture
def square19_board():
    """Create a mock BoardState for square19 board."""
    board = MagicMock(spec=BoardState)
    board.type = BoardType.SQUARE19
    board.width = 19
    board.height = 19
    return board


@pytest.fixture
def hex_board():
    """Create a mock BoardState for hexagonal board."""
    board = MagicMock(spec=BoardState)
    board.type = BoardType.HEXAGONAL
    board.width = 25
    board.height = 25
    return board


# =============================================================================
# Test Policy Size Constants
# =============================================================================


class TestPolicySizeConstants:
    """Tests for policy size constants."""

    def test_square8_policy_size(self):
        """Test SQUARE8 policy size is correct."""
        assert POLICY_SIZE_8x8 == 7000

    def test_square19_policy_size(self):
        """Test SQUARE19 policy size is correct."""
        assert POLICY_SIZE_19x19 == 67000

    def test_hex8_policy_size(self):
        """Test HEX8 policy size is correct."""
        assert POLICY_SIZE_HEX8 == 4500

    def test_hex_policy_size(self):
        """Test HEXAGONAL policy size is correct."""
        # P_HEX = 91876 (placements + movements + special)
        assert P_HEX == 91876

    def test_max_n_constant(self):
        """Test MAX_N constant for legacy encoding."""
        assert MAX_N == 19

    def test_max_players_constant(self):
        """Test MAX_PLAYERS for multi-player value head."""
        assert MAX_PLAYERS == 4


class TestGetPolicySizeForBoard:
    """Tests for get_policy_size_for_board function."""

    def test_square8_size(self):
        """Test policy size for SQUARE8."""
        assert get_policy_size_for_board(BoardType.SQUARE8) == 7000

    def test_square19_size(self):
        """Test policy size for SQUARE19."""
        assert get_policy_size_for_board(BoardType.SQUARE19) == 67000

    def test_hex8_size(self):
        """Test policy size for HEX8."""
        assert get_policy_size_for_board(BoardType.HEX8) == 4500

    def test_hexagonal_size(self):
        """Test policy size for HEXAGONAL."""
        assert get_policy_size_for_board(BoardType.HEXAGONAL) == 91876


class TestGetSpatialSizeForBoard:
    """Tests for get_spatial_size_for_board function."""

    def test_square8_spatial(self):
        """Test spatial size for SQUARE8."""
        assert get_spatial_size_for_board(BoardType.SQUARE8) == 8

    def test_square19_spatial(self):
        """Test spatial size for SQUARE19."""
        assert get_spatial_size_for_board(BoardType.SQUARE19) == 19

    def test_hex8_spatial(self):
        """Test spatial size for HEX8."""
        assert get_spatial_size_for_board(BoardType.HEX8) == 9

    def test_hexagonal_spatial(self):
        """Test spatial size for HEXAGONAL."""
        assert get_spatial_size_for_board(BoardType.HEXAGONAL) == 25


# =============================================================================
# Test Square8 Policy Layout
# =============================================================================


class TestSquare8PolicyLayout:
    """Tests for Square8 policy layout constants."""

    def test_placement_span(self):
        """Test placement span is 3 * 8 * 8 = 192."""
        assert SQUARE8_PLACEMENT_SPAN == 192

    def test_movement_base(self):
        """Test movement base follows placement."""
        assert SQUARE8_MOVEMENT_BASE == SQUARE8_PLACEMENT_SPAN
        assert SQUARE8_MOVEMENT_BASE == 192

    def test_line_form_base(self):
        """Test line formation base follows movement."""
        # Movement span: 8 * 8 * 8 * 7 = 3584
        assert SQUARE8_LINE_FORM_BASE > SQUARE8_MOVEMENT_BASE
        assert SQUARE8_LINE_FORM_BASE == 192 + 3584  # 3776

    def test_territory_claim_base(self):
        """Test territory claim base follows line formation."""
        # Line formation span: 8 * 8 * 4 = 256
        assert SQUARE8_TERRITORY_CLAIM_BASE > SQUARE8_LINE_FORM_BASE
        assert SQUARE8_TERRITORY_CLAIM_BASE == 3776 + 256  # 4032

    def test_skip_placement_idx(self):
        """Test skip placement index is in special section."""
        assert SQUARE8_SKIP_PLACEMENT_IDX > SQUARE8_TERRITORY_CLAIM_BASE
        assert SQUARE8_SKIP_PLACEMENT_IDX < POLICY_SIZE_8x8

    def test_swap_sides_idx(self):
        """Test swap sides index follows skip placement."""
        assert SQUARE8_SWAP_SIDES_IDX == SQUARE8_SKIP_PLACEMENT_IDX + 1


# =============================================================================
# Test Move Encoding - Square8
# =============================================================================


class TestMoveEncodingSquare8:
    """Tests for move encoding on Square8 board."""

    def test_encode_placement_move(self, square8_board):
        """Test encoding a placement move."""
        move = Move(
            id="test1",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=0, y=0),
            placement_count=1,
        )
        idx = _encode_move_square8(move, square8_board)

        # First placement at (0,0) with count=1 should be index 0
        assert idx == 0
        assert 0 <= idx < SQUARE8_PLACEMENT_SPAN

    def test_encode_placement_move_center(self, square8_board):
        """Test encoding placement at board center."""
        move = Move(
            id="test2",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=4, y=4),
            placement_count=1,
        )
        idx = _encode_move_square8(move, square8_board)

        # Position index = y * 8 + x = 4 * 8 + 4 = 36
        # Placement index = pos_idx * 3 + (count - 1) = 36 * 3 + 0 = 108
        assert idx == 108
        assert 0 <= idx < SQUARE8_PLACEMENT_SPAN

    def test_encode_placement_with_count_2(self, square8_board):
        """Test encoding placement with count=2."""
        move = Move(
            id="test3",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=0, y=0),
            placement_count=2,
        )
        idx = _encode_move_square8(move, square8_board)

        # pos_idx = 0, count_idx = 1
        assert idx == 1

    def test_encode_placement_with_count_3(self, square8_board):
        """Test encoding placement with count=3."""
        move = Move(
            id="test4",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=0, y=0),
            placement_count=3,
        )
        idx = _encode_move_square8(move, square8_board)

        # pos_idx = 0, count_idx = 2
        assert idx == 2

    def test_encode_movement_move(self, square8_board):
        """Test encoding a movement move."""
        move = Move(
            id="test5",
            type=MoveType.MOVE_STACK,
            player=1,
            from_pos=Position(x=0, y=0),
            to=Position(x=1, y=0),  # Move right
        )
        idx = _encode_move_square8(move, square8_board)

        # Should be in movement range
        assert idx >= SQUARE8_MOVEMENT_BASE
        assert idx < SQUARE8_LINE_FORM_BASE

    def test_encode_movement_diagonal(self, square8_board):
        """Test encoding diagonal movement."""
        move = Move(
            id="test6",
            type=MoveType.MOVE_STACK,
            player=1,
            from_pos=Position(x=3, y=3),
            to=Position(x=5, y=5),  # Move diagonally 2 squares
        )
        idx = _encode_move_square8(move, square8_board)

        assert idx >= SQUARE8_MOVEMENT_BASE
        assert idx < SQUARE8_LINE_FORM_BASE

    def test_encode_skip_placement(self, square8_board):
        """Test encoding skip placement move."""
        move = Move(id="test7", type=MoveType.SKIP_PLACEMENT, player=1)
        idx = _encode_move_square8(move, square8_board)

        assert idx == SQUARE8_SKIP_PLACEMENT_IDX

    def test_encode_swap_sides(self, square8_board):
        """Test encoding swap sides move."""
        move = Move(id="test8", type=MoveType.SWAP_SIDES, player=1)
        idx = _encode_move_square8(move, square8_board)

        assert idx == SQUARE8_SWAP_SIDES_IDX

    def test_encode_invalid_position(self, square8_board):
        """Test encoding with out-of-bounds position."""
        move = Move(
            id="test9",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=10, y=10),  # Out of bounds
            placement_count=1,
        )
        idx = _encode_move_square8(move, square8_board)

        assert idx == INVALID_MOVE_INDEX

    def test_encode_movement_no_from_pos(self, square8_board):
        """Test encoding movement without from_pos."""
        move = Move(
            id="test10",
            type=MoveType.MOVE_STACK,
            player=1,
            to=Position(x=1, y=1),
        )
        idx = _encode_move_square8(move, square8_board)

        assert idx == INVALID_MOVE_INDEX


class TestMoveDecodingSquare8:
    """Tests for move decoding on Square8 board."""

    def test_decode_placement_index_0(self):
        """Test decoding index 0 (first placement)."""
        decoded = _decode_move_square8(0)

        assert decoded is not None
        assert decoded.action_type == "placement"
        assert (decoded.x, decoded.y) == (0, 0)
        assert decoded.count_idx == 0  # count_idx = placement_count - 1

    def test_decode_placement_center(self):
        """Test decoding center placement."""
        # Position (4, 4), count=1: idx = 36 * 3 + 0 = 108
        decoded = _decode_move_square8(108)

        assert decoded is not None
        assert decoded.action_type == "placement"
        assert (decoded.x, decoded.y) == (4, 4)
        assert decoded.count_idx == 0

    def test_decode_skip_placement(self):
        """Test decoding skip placement."""
        decoded = _decode_move_square8(SQUARE8_SKIP_PLACEMENT_IDX)

        assert decoded is not None
        assert decoded.action_type == "skip_placement"

    def test_decode_swap_sides(self):
        """Test decoding swap sides."""
        decoded = _decode_move_square8(SQUARE8_SWAP_SIDES_IDX)

        assert decoded is not None
        assert decoded.action_type == "swap_sides"

    def test_decode_invalid_index(self):
        """Test decoding invalid index."""
        decoded = _decode_move_square8(999999)  # Way out of bounds

        assert decoded is None


class TestMoveEncodingRoundTrip:
    """Tests for encoding/decoding round-trip consistency."""

    def test_placement_round_trip(self, square8_board):
        """Test placement encoding round-trips correctly."""
        for x in range(8):
            for y in range(8):
                for count in [1, 2, 3]:
                    move = Move(
                        id=f"move_{x}_{y}_{count}",
                        type=MoveType.PLACE_RING,
                        player=1,
                        to=Position(x=x, y=y),
                        placement_count=count,
                    )
                    idx = _encode_move_square8(move, square8_board)
                    decoded = _decode_move_square8(idx)

                    assert decoded is not None
                    assert decoded.action_type == "placement"
                    assert (decoded.x, decoded.y) == (x, y)
                    assert decoded.count_idx == count - 1


# =============================================================================
# Test encode_move_for_board Dispatcher
# =============================================================================


class TestEncodeMoveForBoard:
    """Tests for encode_move_for_board dispatcher function."""

    def test_dispatch_square8(self, square8_board):
        """Test dispatching to square8 encoder."""
        move = Move(
            id="test_dispatch1",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=0, y=0),
            placement_count=1,
        )
        idx = encode_move_for_board(move, square8_board)

        assert idx == 0
        assert idx < POLICY_SIZE_8x8

    def test_dispatch_square19(self, square19_board):
        """Test dispatching to square19 encoder."""
        move = Move(
            id="test_dispatch2",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=0, y=0),
            placement_count=1,
        )
        idx = encode_move_for_board(move, square19_board)

        assert idx >= 0
        assert idx < POLICY_SIZE_19x19


# =============================================================================
# Test Hex Constants
# =============================================================================


class TestHexConstants:
    """Tests for hexagonal board constants."""

    def test_hex_board_size(self):
        """Test HEX_BOARD_SIZE is 25 (2*12 + 1)."""
        assert HEX_BOARD_SIZE == 25

    def test_hex8_board_size(self):
        """Test HEX8_BOARD_SIZE is 9 (2*4 + 1)."""
        assert HEX8_BOARD_SIZE == 9

    def test_num_hex_dirs(self):
        """Test there are 6 hex directions."""
        assert NUM_HEX_DIRS == 6

    def test_hex_dirs(self):
        """Test hex direction vectors."""
        assert len(HEX_DIRS) == 6
        # Check all directions are unit-ish vectors
        for dx, dy in HEX_DIRS:
            assert abs(dx) <= 1
            assert abs(dy) <= 1


# =============================================================================
# Test DecodedPolicyIndex Dataclass
# =============================================================================


class TestDecodedPolicyIndex:
    """Tests for DecodedPolicyIndex dataclass."""

    def test_placement_creation(self):
        """Test creating a decoded placement."""
        decoded = DecodedPolicyIndex(
            action_type="placement",
            board_size=8,
            x=3,
            y=4,
            count_idx=1,  # count_idx = placement_count - 1
        )

        assert decoded.action_type == "placement"
        assert decoded.x == 3
        assert decoded.y == 4
        assert decoded.count_idx == 1

    def test_movement_creation(self):
        """Test creating a decoded movement."""
        decoded = DecodedPolicyIndex(
            action_type="movement",
            board_size=8,
            x=1,
            y=1,
            dir_idx=4,
            dist=2,
        )

        assert decoded.action_type == "movement"
        assert decoded.x == 1
        assert decoded.y == 1
        assert decoded.dir_idx == 4
        assert decoded.dist == 2

    def test_special_action_creation(self):
        """Test creating a special action."""
        decoded = DecodedPolicyIndex(
            action_type="skip_placement",
            board_size=8,
            is_special=True,
        )

        assert decoded.action_type == "skip_placement"
        assert decoded.is_special is True


# =============================================================================
# Test NeuralNetAI Initialization
# =============================================================================


class TestNeuralNetAIInit:
    """Tests for NeuralNetAI initialization."""

    @patch("app.ai._neural_net_legacy.torch.backends.mps.is_available", return_value=False)
    @patch("app.ai._neural_net_legacy.torch.cuda.is_available", return_value=False)
    def test_init_cpu_device(self, mock_cuda, mock_mps, mock_config):
        """Test initialization selects CPU when no GPU available."""
        from app.ai.neural_net import NeuralNetAI

        ai = NeuralNetAI(player_number=1, config=mock_config)

        assert ai.device.type == "cpu"
        assert ai.player_number == 1
        assert ai.model is None  # Lazy initialization
        assert ai.history_length == 3

    @patch("app.ai._neural_net_legacy.torch.backends.mps.is_available", return_value=True)
    @patch("app.ai._neural_net_legacy.torch.cuda.is_available", return_value=False)
    @patch.dict("os.environ", {"RINGRIFT_NN_ARCHITECTURE": "auto"}, clear=False)
    def test_init_mps_device(self, mock_cuda, mock_mps, mock_config):
        """Test initialization selects MPS when available and architecture is auto."""
        from app.ai.neural_net import NeuralNetAI

        ai = NeuralNetAI(player_number=1, config=mock_config)

        assert ai.device.type == "mps"
        assert ai._use_mps_arch is True

    @patch("app.ai._neural_net_legacy.torch.backends.mps.is_available", return_value=False)
    @patch("app.ai._neural_net_legacy.torch.cuda.is_available", return_value=True)
    def test_init_cuda_device(self, mock_cuda, mock_mps, mock_config):
        """Test initialization selects CUDA when available."""
        from app.ai.neural_net import NeuralNetAI

        ai = NeuralNetAI(player_number=1, config=mock_config)

        assert ai.device.type == "cuda"

    @patch("app.ai._neural_net_legacy.torch.backends.mps.is_available", return_value=False)
    @patch("app.ai._neural_net_legacy.torch.cuda.is_available", return_value=True)
    @patch.dict("os.environ", {"RINGRIFT_FORCE_CPU": "1"}, clear=False)
    def test_init_force_cpu(self, mock_cuda, mock_mps, mock_config):
        """Test RINGRIFT_FORCE_CPU environment variable."""
        from app.ai.neural_net import NeuralNetAI

        ai = NeuralNetAI(player_number=1, config=mock_config)

        assert ai.device.type == "cpu"

    @patch("app.ai._neural_net_legacy.torch.backends.mps.is_available", return_value=False)
    @patch("app.ai._neural_net_legacy.torch.cuda.is_available", return_value=False)
    def test_init_with_board_type(self, mock_cuda, mock_mps, mock_config):
        """Test initialization with explicit board_type triggers early model init."""
        from app.ai.neural_net import NeuralNetAI

        # Note: This will try to load a model which may not exist
        # In allow_fresh_weights=True mode, it should create fresh weights
        ai = NeuralNetAI(
            player_number=1,
            config=mock_config,
            board_type=BoardType.SQUARE8,
        )

        # Model should be initialized (or at least attempted)
        assert ai._initialized_board_type == BoardType.SQUARE8
        assert ai.board_size == 8

    @patch("app.ai._neural_net_legacy.torch.backends.mps.is_available", return_value=False)
    @patch("app.ai._neural_net_legacy.torch.cuda.is_available", return_value=False)
    def test_init_feature_version(self, mock_cuda, mock_mps):
        """Test feature version is read from config."""
        from app.ai.neural_net import NeuralNetAI

        config = MockAIConfig(feature_version=2)
        ai = NeuralNetAI(player_number=1, config=config)

        assert ai.feature_version == 2


class TestNeuralNetAIGameHistory:
    """Tests for NeuralNetAI game history management."""

    @patch("app.ai._neural_net_legacy.torch.backends.mps.is_available", return_value=False)
    @patch("app.ai._neural_net_legacy.torch.cuda.is_available", return_value=False)
    def test_game_history_initialized_empty(self, mock_cuda, mock_mps, mock_config):
        """Test game history starts empty."""
        from app.ai.neural_net import NeuralNetAI

        ai = NeuralNetAI(player_number=1, config=mock_config)

        assert ai.game_history == {}

    @patch("app.ai._neural_net_legacy.torch.backends.mps.is_available", return_value=False)
    @patch("app.ai._neural_net_legacy.torch.cuda.is_available", return_value=False)
    def test_history_length_default(self, mock_cuda, mock_mps, mock_config):
        """Test default history length is 3."""
        from app.ai.neural_net import NeuralNetAI

        ai = NeuralNetAI(player_number=1, config=mock_config)

        assert ai.history_length == 3


# =============================================================================
# Test CNN Architecture Constants
# =============================================================================


class TestCNNArchitectureConstants:
    """Tests for CNN architecture-related constants."""

    def test_invalid_move_index(self):
        """Test INVALID_MOVE_INDEX constant."""
        assert INVALID_MOVE_INDEX == -1

    def test_policy_sizes_are_positive(self):
        """Test all policy sizes are positive."""
        assert POLICY_SIZE_8x8 > 0
        assert POLICY_SIZE_19x19 > 0
        assert POLICY_SIZE_HEX8 > 0
        assert P_HEX > 0

    def test_policy_sizes_ordered(self):
        """Test policy sizes are in expected order."""
        # HEX8 < SQUARE8 < SQUARE19 < HEXAGONAL
        assert POLICY_SIZE_HEX8 < POLICY_SIZE_8x8
        assert POLICY_SIZE_8x8 < POLICY_SIZE_19x19
        assert POLICY_SIZE_19x19 < P_HEX


class TestEnsureModelInitialized:
    """Tests for _ensure_model_initialized method."""

    @patch("app.ai._neural_net_legacy.torch.backends.mps.is_available", return_value=False)
    @patch("app.ai._neural_net_legacy.torch.cuda.is_available", return_value=False)
    def test_double_init_same_board_type_ok(self, mock_cuda, mock_mps, mock_config):
        """Test calling _ensure_model_initialized twice with same board type is ok."""
        from app.ai.neural_net import NeuralNetAI

        ai = NeuralNetAI(player_number=1, config=mock_config)

        # First init
        ai._ensure_model_initialized(BoardType.SQUARE8)

        # Second init with same type should not raise
        ai._ensure_model_initialized(BoardType.SQUARE8)

        assert ai._initialized_board_type == BoardType.SQUARE8

    @patch("app.ai._neural_net_legacy.torch.backends.mps.is_available", return_value=False)
    @patch("app.ai._neural_net_legacy.torch.cuda.is_available", return_value=False)
    def test_double_init_different_board_type_raises(self, mock_cuda, mock_mps, mock_config):
        """Test calling _ensure_model_initialized with different board type raises."""
        from app.ai.neural_net import NeuralNetAI

        ai = NeuralNetAI(player_number=1, config=mock_config)

        # First init
        ai._ensure_model_initialized(BoardType.SQUARE8)

        # Second init with different type should raise
        with pytest.raises(RuntimeError, match="different board types"):
            ai._ensure_model_initialized(BoardType.SQUARE19)
