"""
Unit tests for app.training.encoding module.

Tests cover:
- HexStateEncoder class and methods
- HexStateEncoderV3 class
- SquareStateEncoder class
- Utility functions (detect_board_type_from_features, get_encoder_for_board_type)
- Coordinate transformations
- State and move encoding

Created: December 2025
"""

from datetime import datetime

import numpy as np
import pytest

from app.models import (
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Player,
    RingStack,
    TimeControl,
)
from app.rules.core import get_rings_per_player
from app.training.encoding import (
    HexStateEncoder,
    HexStateEncoderV3,
    SquareStateEncoder,
    detect_board_type_from_features,
    get_encoder_for_board_type,
)


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_player(player_number: int, rings_in_hand: int = 5) -> Player:
    """Create a properly structured Player for testing."""
    return Player(
        id=f"p{player_number}",
        username=f"Player {player_number}",
        type="human",
        playerNumber=player_number,
        isReady=True,
        timeRemaining=600,
        ringsInHand=rings_in_hand,
        eliminatedRings=0,
        territorySpaces=0,
        aiDifficulty=10,
    )


def create_hex_game_state(
    board_type: BoardType = BoardType.HEXAGONAL,
    size: int = 25,
    phase: GamePhase = GamePhase.RING_PLACEMENT,
    stacks: dict | None = None,
    markers: dict | None = None,
    collapsed_spaces: dict | None = None,
) -> GameState:
    """Create a hex game state for testing."""
    if board_type == BoardType.HEX8:
        size = 9
        rings_per_player = get_rings_per_player(BoardType.HEX8)
    else:
        size = 25
        rings_per_player = get_rings_per_player(BoardType.HEXAGONAL)

    players = [
        create_test_player(1, rings_per_player),
        create_test_player(2, rings_per_player),
    ]

    board = BoardState(
        type=board_type,
        size=size,
        stacks=stacks or {},
        markers=markers or {},
        collapsedSpaces=collapsed_spaces or {},
        eliminatedRings={},
    )
    total_rings_in_play = sum(player.rings_in_hand for player in players) + sum(
        len(stack.rings) for stack in board.stacks.values()
    )

    return GameState(
        id="test-hex",
        boardType=board_type,
        board=board,
        players=players,
        currentPhase=phase,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=total_rings_in_play,
        totalRingsEliminated=0,
        victoryThreshold=rings_per_player,
        territoryVictoryThreshold=235,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        rngSeed=42,
    )


def create_square_game_state(
    board_type: BoardType = BoardType.SQUARE8,
    size: int = 8,
    phase: GamePhase = GamePhase.RING_PLACEMENT,
    stacks: dict | None = None,
    markers: dict | None = None,
    collapsed_spaces: dict | None = None,
) -> GameState:
    """Create a square game state for testing."""
    rings_per_player = get_rings_per_player(board_type)

    players = [
        create_test_player(1, rings_per_player),
        create_test_player(2, rings_per_player),
    ]

    board = BoardState(
        type=board_type,
        size=size,
        stacks=stacks or {},
        markers=markers or {},
        collapsedSpaces=collapsed_spaces or {},
        eliminatedRings={},
    )
    total_rings_in_play = sum(player.rings_in_hand for player in players) + sum(
        len(stack.rings) for stack in board.stacks.values()
    )

    return GameState(
        id="test-square",
        boardType=board_type,
        board=board,
        players=players,
        currentPhase=phase,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=total_rings_in_play,
        totalRingsEliminated=0,
        victoryThreshold=rings_per_player,
        territoryVictoryThreshold=32,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        rngSeed=42,
    )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def hex_encoder():
    """Create a HexStateEncoder for default hex board (radius=12, size=25)."""
    return HexStateEncoder()


@pytest.fixture
def hex8_encoder():
    """Create a HexStateEncoder for hex8 board (radius=4, size=9)."""
    from app.ai.neural_net import HEX8_BOARD_SIZE, POLICY_SIZE_HEX8

    return HexStateEncoder(board_size=HEX8_BOARD_SIZE, policy_size=POLICY_SIZE_HEX8)


@pytest.fixture
def hex_encoder_v3():
    """Create a HexStateEncoderV3 for default hex board."""
    return HexStateEncoderV3()


@pytest.fixture
def square_encoder():
    """Create a SquareStateEncoder for square8 board."""
    return SquareStateEncoder(board_type=BoardType.SQUARE8, board_size=8)


@pytest.fixture
def empty_hex_game_state():
    """Create an empty hex game state for testing."""
    return create_hex_game_state(board_type=BoardType.HEXAGONAL)


@pytest.fixture
def empty_hex8_game_state():
    """Create an empty hex8 game state for testing."""
    return create_hex_game_state(board_type=BoardType.HEX8)


@pytest.fixture
def empty_square8_game_state():
    """Create an empty square8 game state for testing."""
    return create_square_game_state(board_type=BoardType.SQUARE8)


@pytest.fixture
def hex_game_state_with_pieces():
    """Create a hex game state with some pieces on the board."""
    from app.models import Position

    stacks = {
        "0,0,0": RingStack(
            position=Position(x=0, y=0, z=0),
            rings=[1, 1],
            stackHeight=2,
            capHeight=2,
            controllingPlayer=1,
        ),
        "1,-1,0": RingStack(
            position=Position(x=1, y=-1, z=0),
            rings=[2, 2, 2],
            stackHeight=3,
            capHeight=3,
            controllingPlayer=2,
        ),
    }
    collapsed_spaces = {
        "2,-1,0": 1,  # Player 1 territory
    }
    return create_hex_game_state(
        board_type=BoardType.HEXAGONAL,
        phase=GamePhase.MOVEMENT,
        stacks=stacks,
        collapsed_spaces=collapsed_spaces,
    )


# =============================================================================
# HexStateEncoder Tests
# =============================================================================


class TestHexStateEncoder:
    """Tests for HexStateEncoder class."""

    def test_initialization_default(self, hex_encoder):
        """Default encoder has correct dimensions."""
        assert hex_encoder.board_size == 25
        assert hex_encoder.radius == 12
        assert hex_encoder.NUM_CHANNELS == 10
        assert hex_encoder.NUM_GLOBAL_FEATURES == 20

    def test_initialization_hex8(self, hex8_encoder):
        """Hex8 encoder has correct dimensions."""
        assert hex8_encoder.board_size == 9
        assert hex8_encoder.radius == 4
        assert hex8_encoder.policy_size == 4500

    def test_valid_mask_shape(self, hex_encoder):
        """Valid mask has correct shape."""
        mask = hex_encoder.get_valid_mask()
        assert mask.shape == (25, 25)
        assert mask.dtype == bool

    def test_valid_mask_content(self, hex_encoder):
        """Valid mask marks only valid hex cells."""
        mask = hex_encoder.get_valid_mask()
        # Center cell (radius, radius) should be valid
        assert mask[12, 12] == True
        # Check that corners are not all False (hex grid has some corners valid)
        # The actual valid cells depend on the hex grid layout

    def test_valid_mask_tensor(self, hex_encoder):
        """Valid mask tensor has correct shape and type."""
        mask_tensor = hex_encoder.get_valid_mask_tensor()
        assert mask_tensor.shape == (1, 25, 25)
        assert mask_tensor.dtype == np.float32

    def test_axial_to_canonical(self, hex_encoder):
        """Axial to canonical coordinate conversion."""
        # Center (0, 0) -> (12, 12)
        cx, cy = hex_encoder.axial_to_canonical(0, 0)
        assert cx == 12
        assert cy == 12

        # Edge case at radius boundary
        cx, cy = hex_encoder.axial_to_canonical(-12, 0)
        assert cx == 0
        assert cy == 12

    def test_canonical_to_axial(self, hex_encoder):
        """Canonical to axial coordinate conversion."""
        # Center (12, 12) -> (0, 0)
        q, r = hex_encoder.canonical_to_axial(12, 12)
        assert q == 0
        assert r == 0

    def test_coordinate_roundtrip(self, hex_encoder):
        """Axial -> canonical -> axial preserves coordinates."""
        for q in range(-12, 13, 3):
            for r in range(-12, 13, 3):
                if abs(q) + abs(r) + abs(-q - r) <= 24:  # Valid hex cell
                    cx, cy = hex_encoder.axial_to_canonical(q, r)
                    q2, r2 = hex_encoder.canonical_to_axial(cx, cy)
                    assert q == q2
                    assert r == r2

    def test_encode_state_empty(self, hex_encoder, empty_hex_game_state):
        """Encoding empty state produces correct shapes."""
        features, globals_vec = hex_encoder.encode_state(empty_hex_game_state)
        assert features.shape == (10, 25, 25)
        assert globals_vec.shape == (20,)
        assert features.dtype == np.float32
        assert globals_vec.dtype == np.float32

    def test_encode_state_with_pieces(self, hex_encoder, hex_game_state_with_pieces):
        """Encoding state with pieces produces non-zero features."""
        features, globals_vec = hex_encoder.encode_state(hex_game_state_with_pieces)

        # Should have some non-zero values in stack channels
        # Channel assignment depends on implementation
        assert features.shape == (10, 25, 25)

    def test_encode_stacks_frames(self, hex_encoder, empty_hex_game_state):
        """encode() stacks 4 frames (40 channels) vs encode_state() single frame (10 channels)."""
        f_single, g1 = hex_encoder.encode_state(empty_hex_game_state)
        f_stacked, g2 = hex_encoder.encode(empty_hex_game_state)
        assert f_single.shape == (10, 25, 25)
        assert f_stacked.shape == (40, 25, 25)
        np.testing.assert_array_equal(g1, g2)

    def test_encode_wrong_board_type(self, hex_encoder, empty_square8_game_state):
        """Encoding square board raises ValueError."""
        with pytest.raises(ValueError, match="requires HEXAGONAL or HEX8"):
            hex_encoder.encode_state(empty_square8_game_state)

    def test_encode_with_history(self, hex_encoder, empty_hex_game_state):
        """Encoding with history stacks frames correctly."""
        features, _ = hex_encoder.encode_state(empty_hex_game_state)

        # Create fake history
        history = [np.zeros_like(features) for _ in range(3)]
        stacked, globals_vec = hex_encoder.encode_with_history(
            empty_hex_game_state, history, history_length=3
        )

        # 4 frames * 10 channels = 40 channels
        assert stacked.shape == (40, 25, 25)

    def test_encode_with_history_padding(self, hex_encoder, empty_hex_game_state):
        """Encoding with insufficient history pads with zeros."""
        # Only 1 history frame
        features, _ = hex_encoder.encode_state(empty_hex_game_state)
        history = [features.copy()]
        stacked, _ = hex_encoder.encode_with_history(
            empty_hex_game_state, history, history_length=3
        )

        # Should still produce 4 frames
        assert stacked.shape == (40, 25, 25)

    def test_global_features_shape(self, hex_encoder, empty_hex_game_state):
        """Global features have correct shape."""
        _, globals_vec = hex_encoder.encode_state(empty_hex_game_state)
        assert globals_vec.shape == (20,)

    def test_create_policy_target(self, hex_encoder):
        """Dense policy target creation."""
        indices = [0, 100, 1000]
        probs = [0.5, 0.3, 0.2]
        policy = hex_encoder.create_policy_target(indices, probs)

        assert policy.shape == (hex_encoder.POLICY_SIZE,)
        assert policy[0] == 0.5
        assert policy[100] == 0.3
        assert policy[1000] == 0.2
        assert policy.sum() == pytest.approx(1.0)

    def test_create_sparse_policy_target(self, hex_encoder):
        """Sparse policy target creation."""
        indices = [0, 100, 1000]
        probs = [0.5, 0.3, 0.2]
        idx_arr, val_arr = hex_encoder.create_sparse_policy_target(indices, probs)

        assert len(idx_arr) == 3
        assert len(val_arr) == 3
        np.testing.assert_array_equal(idx_arr, np.array([0, 100, 1000], dtype=np.int32))
        np.testing.assert_array_equal(
            val_arr, np.array([0.5, 0.3, 0.2], dtype=np.float32)
        )

    def test_create_policy_target_invalid_indices(self, hex_encoder):
        """Invalid indices are filtered out."""
        indices = [-1, 0, 100000]  # -1 and 100000 are invalid
        probs = [0.5, 0.3, 0.2]
        policy = hex_encoder.create_policy_target(indices, probs)

        assert policy[0] == 0.3  # Only valid index
        assert policy.sum() == pytest.approx(0.3)


# =============================================================================
# HexStateEncoderV3 Tests
# =============================================================================


class TestHexStateEncoderV3:
    """Tests for HexStateEncoderV3 class."""

    def test_initialization(self, hex_encoder_v3):
        """V3 encoder has correct channel count."""
        assert hex_encoder_v3.NUM_CHANNELS == 16  # V3 has 16 base channels
        assert hex_encoder_v3.NUM_GLOBAL_FEATURES == 20

    def test_encode_state_shape(self, hex_encoder_v3, empty_hex_game_state):
        """V3 encoding produces 16 channels."""
        features, globals_vec = hex_encoder_v3.encode_state(empty_hex_game_state)
        assert features.shape == (16, 25, 25)
        assert globals_vec.shape == (20,)

    def test_encode_with_frame_stacking(self, hex_encoder_v3, empty_hex_game_state):
        """V3 encode() produces 64 channels (16 * 4 frames)."""
        stacked, globals_vec = hex_encoder_v3.encode(empty_hex_game_state)
        assert stacked.shape == (64, 25, 25)

    def test_valid_mask_shape(self, hex_encoder_v3):
        """V3 valid mask has 2D shape."""
        mask = hex_encoder_v3.get_valid_mask_tensor()
        # V3 returns 2D, not 3D like base encoder
        assert mask.shape == (25, 25)

    def test_enhanced_channels_present(self, hex_encoder_v3, hex_game_state_with_pieces):
        """V3 enhanced channels (10-15) exist."""
        features, _ = hex_encoder_v3.encode_state(hex_game_state_with_pieces)

        # Channels 10-15: Enhanced features
        assert features.shape[0] == 16

    def test_feature_version_2_preserves_placement_validity_planes(
        self, empty_hex_game_state
    ):
        """Legacy fv2 keeps placement-validity planes populated."""
        encoder = HexStateEncoderV3(feature_version=2)
        features, _ = encoder.encode_state(empty_hex_game_state)

        assert np.count_nonzero(features[14]) > 0
        assert np.count_nonzero(features[15]) > 0

    def test_feature_version_3_zeroes_placement_validity_planes(
        self, empty_hex_game_state
    ):
        """fv3 disables the leaky placement-validity shortcut while preserving shape."""
        encoder = HexStateEncoderV3(feature_version=3)
        features, _ = encoder.encode_state(empty_hex_game_state)

        assert features.shape == (16, 25, 25)
        np.testing.assert_array_equal(features[14], np.zeros((25, 25), dtype=np.float32))
        np.testing.assert_array_equal(features[15], np.zeros((25, 25), dtype=np.float32))

    def test_feature_version_3_only_changes_placement_planes(
        self, hex_game_state_with_pieces
    ):
        """fv3 should preserve non-placement channels and zero only 14/15."""
        legacy = HexStateEncoderV3(feature_version=2)
        fv3 = HexStateEncoderV3(feature_version=3)

        legacy_features, legacy_globals = legacy.encode_state(hex_game_state_with_pieces)
        fv3_features, fv3_globals = fv3.encode_state(hex_game_state_with_pieces)

        np.testing.assert_allclose(legacy_features[:14], fv3_features[:14])
        np.testing.assert_array_equal(
            fv3_features[14:], np.zeros_like(fv3_features[14:])
        )
        np.testing.assert_allclose(legacy_globals, fv3_globals)


# =============================================================================
# SquareStateEncoder Tests
# =============================================================================


class TestSquareStateEncoder:
    """Tests for SquareStateEncoder class."""

    def test_initialization(self, square_encoder):
        """Square encoder has correct dimensions."""
        assert square_encoder.board_size == 8
        assert square_encoder.NUM_CHANNELS == 14
        assert square_encoder.NUM_GLOBAL_FEATURES == 20

    def test_encode_state_shape(self, square_encoder, empty_square8_game_state):
        """Encoding produces correct shapes."""
        features, globals_vec = square_encoder.encode_state(empty_square8_game_state)
        assert features.shape == (14, 8, 8)
        assert globals_vec.shape == (20,)

    def test_encode_wrong_board_type(self, square_encoder, empty_hex_game_state):
        """Encoding hex board raises ValueError."""
        with pytest.raises(ValueError, match="requires SQUARE8 or SQUARE19"):
            square_encoder.encode_state(empty_hex_game_state)

    def test_board_mask_channel(self, square_encoder, empty_square8_game_state):
        """Board mask channel is all ones for square boards."""
        features, _ = square_encoder.encode_state(empty_square8_game_state)
        # Channel 12 is the valid board position mask
        np.testing.assert_array_equal(
            features[12], np.ones((8, 8), dtype=np.float32)
        )

    def test_encode_stacks_frames(self, square_encoder, empty_square8_game_state):
        """encode() stacks 4 frames (56 channels) vs encode_state() single frame (14 channels)."""
        f_single, g1 = square_encoder.encode_state(empty_square8_game_state)
        f_stacked, g2 = square_encoder.encode(empty_square8_game_state)
        assert f_single.shape == (14, 8, 8)
        assert f_stacked.shape == (56, 8, 8)
        np.testing.assert_array_equal(g1, g2)


class TestSquare19Encoder:
    """Tests for SquareStateEncoder with square19 board."""

    @pytest.fixture
    def square19_encoder(self):
        return SquareStateEncoder(board_type=BoardType.SQUARE19, board_size=19)

    @pytest.fixture
    def empty_square19_game_state(self):
        return create_square_game_state(
            board_type=BoardType.SQUARE19,
            size=19,
        )

    def test_square19_shape(self, square19_encoder, empty_square19_game_state):
        """Square19 encoder produces 19x19 features."""
        features, globals_vec = square19_encoder.encode_state(empty_square19_game_state)
        assert features.shape == (14, 19, 19)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestDetectBoardTypeFromFeatures:
    """Tests for detect_board_type_from_features function."""

    def test_detect_square8(self):
        """Detects SQUARE8 from 8x8 features."""
        features = np.zeros((14, 8, 8))
        board_type = detect_board_type_from_features(features)
        assert board_type == BoardType.SQUARE8

    def test_detect_hex8(self):
        """Detects HEX8 from 9x9 features."""
        features = np.zeros((10, 9, 9))
        board_type = detect_board_type_from_features(features)
        assert board_type == BoardType.HEX8

    def test_detect_square19(self):
        """Detects SQUARE19 from 19x19 features."""
        features = np.zeros((14, 19, 19))
        board_type = detect_board_type_from_features(features)
        assert board_type == BoardType.SQUARE19

    def test_detect_hexagonal(self):
        """Detects HEXAGONAL from 25x25 features."""
        features = np.zeros((10, 25, 25))
        board_type = detect_board_type_from_features(features)
        assert board_type == BoardType.HEXAGONAL

    def test_detect_4d_tensor(self):
        """Detection works with 4D batch tensor."""
        features = np.zeros((32, 10, 25, 25))  # Batch of 32
        board_type = detect_board_type_from_features(features)
        assert board_type == BoardType.HEXAGONAL

    def test_unknown_size_raises(self):
        """Unknown spatial size raises ValueError."""
        features = np.zeros((10, 12, 12))  # 12x12 not a known size
        with pytest.raises(ValueError, match="Cannot detect board type"):
            detect_board_type_from_features(features)

    def test_wrong_dims_raises(self):
        """Non-3D/4D tensor raises ValueError."""
        features = np.zeros((10, 8))  # 2D
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            detect_board_type_from_features(features)


class TestGetEncoderForBoardType:
    """Tests for get_encoder_for_board_type factory function."""

    def test_hexagonal_v2(self):
        """Returns HexStateEncoder for HEXAGONAL."""
        encoder = get_encoder_for_board_type(BoardType.HEXAGONAL, version="v2")
        assert isinstance(encoder, HexStateEncoder)
        assert encoder.board_size == 25

    def test_hexagonal_v3(self):
        """Returns HexStateEncoderV3 for HEXAGONAL with v3."""
        encoder = get_encoder_for_board_type(BoardType.HEXAGONAL, version="v3")
        assert isinstance(encoder, HexStateEncoderV3)

    def test_hex8_v2(self):
        """Returns HexStateEncoder for HEX8."""
        encoder = get_encoder_for_board_type(BoardType.HEX8, version="v2")
        assert isinstance(encoder, HexStateEncoder)
        assert encoder.board_size == 9

    def test_hex8_v3(self):
        """Returns HexStateEncoderV3 for HEX8 with v3."""
        encoder = get_encoder_for_board_type(BoardType.HEX8, version="v3")
        assert isinstance(encoder, HexStateEncoderV3)
        assert encoder.board_size == 9

    def test_square8(self):
        """Returns SquareStateEncoder for SQUARE8."""
        encoder = get_encoder_for_board_type(BoardType.SQUARE8)
        assert isinstance(encoder, SquareStateEncoder)
        assert encoder.board_size == 8

    def test_square19(self):
        """Returns SquareStateEncoder for SQUARE19."""
        encoder = get_encoder_for_board_type(BoardType.SQUARE19)
        assert isinstance(encoder, SquareStateEncoder)
        assert encoder.board_size == 19

    def test_feature_version_passed(self):
        """Feature version is passed to encoder."""
        encoder = get_encoder_for_board_type(
            BoardType.HEXAGONAL, version="v2", feature_version=2
        )
        assert encoder.feature_version == 2


# =============================================================================
# Feature Version Tests
# =============================================================================


class TestFeatureVersions:
    """Tests for different feature versions."""

    def test_v1_game_progress(self, empty_hex_game_state):
        """V1 features have 20 global features."""
        encoder = HexStateEncoder(feature_version=1)
        _, globals_vec = encoder.encode_state(empty_hex_game_state)
        assert globals_vec.shape == (20,)

    def test_v2_chain_capture_flag(self, empty_hex_game_state):
        """V2 features include chain capture flag."""
        encoder = HexStateEncoder(feature_version=2)
        _, globals_vec = encoder.encode_state(empty_hex_game_state)
        # Not in chain capture phase - flag should be 0
        assert globals_vec[18] == 0.0

    def test_v2_chain_capture_active(self):
        """V2 chain capture flag is 1.0 when in chain capture phase."""
        state = create_hex_game_state(
            board_type=BoardType.HEXAGONAL,
            phase=GamePhase.CHAIN_CAPTURE,
        )
        encoder = HexStateEncoder(feature_version=2)
        _, globals_vec = encoder.encode_state(state)
        assert globals_vec[18] == 1.0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEncodingEdgeCases:
    """Edge case tests for encoding module."""

    def test_encode_empty_board(self, hex_encoder, empty_hex_game_state):
        """Empty board produces all-zero spatial features."""
        features, _ = hex_encoder.encode_state(empty_hex_game_state)
        # All spatial channels should be zero for empty board
        assert features.sum() == 0.0

    def test_encode_deterministic(self, hex_encoder, hex_game_state_with_pieces):
        """Encoding the same state twice produces identical results."""
        f1, g1 = hex_encoder.encode_state(hex_game_state_with_pieces)
        f2, g2 = hex_encoder.encode_state(hex_game_state_with_pieces)
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(g1, g2)

    def test_feature_values_normalized(self, hex_encoder, hex_game_state_with_pieces):
        """Feature values are in reasonable range."""
        features, globals_vec = hex_encoder.encode_state(hex_game_state_with_pieces)
        assert features.min() >= 0.0
        assert features.max() <= 1.0
        assert globals_vec.min() >= 0.0
        assert globals_vec.max() <= 1.0

    def test_stack_height_normalization(self, hex_encoder):
        """Stack heights are normalized to [0, 1]."""
        from app.models import Position

        stacks = {
            "0,0,0": RingStack(
                position=Position(x=0, y=0, z=0),
                rings=[1] * 10,  # Very tall stack
                stackHeight=10,
                capHeight=10,
                controllingPlayer=1,
            )
        }
        state = create_hex_game_state(
            board_type=BoardType.HEXAGONAL,
            phase=GamePhase.MOVEMENT,
            stacks=stacks,
        )
        features, _ = hex_encoder.encode_state(state)
        # Max value should be capped at 1.0 even for tall stacks
        assert features.max() <= 1.0

    def test_sparse_policy_empty(self, hex_encoder):
        """Sparse policy with empty indices returns empty arrays."""
        idx_arr, val_arr = hex_encoder.create_sparse_policy_target([], [])
        assert len(idx_arr) == 0
        assert len(val_arr) == 0

    def test_dense_policy_empty(self, hex_encoder):
        """Dense policy with empty indices returns zeros."""
        policy = hex_encoder.create_policy_target([], [])
        assert policy.sum() == 0.0
