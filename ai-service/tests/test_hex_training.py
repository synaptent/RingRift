"""
Tests for Hex Board Training Pipeline.

Tests the complete hex training infrastructure:
- HexStateEncoder correctness
- Hex data generation
- HexNeuralNet training on small samples
- Integration with training infrastructure
"""

import numpy as np
import pytest
import torch
from datetime import datetime

from app.ai.neural_net import (
    HexNeuralNet,
    ActionEncoderHex,
    HEX_BOARD_SIZE,
    P_HEX,
)
from app.models import (
    BoardType,
    BoardState,
    GameState,
    GamePhase,
    GameStatus,
    TimeControl,
    Player,
    Position,
    Move,
    MoveType,
    RingStack,
    MarkerInfo,
)
from app.training.encoding import (
    HexStateEncoder,
    detect_board_type_from_features,
    get_encoder_for_board_type,
)
from app.training.hex_augmentation import (
    HexSymmetryTransform,
    augment_hex_sample,
)


def create_hex_game_state(
    size: int = 11,
    current_player: int = 1,
    phase: GamePhase = GamePhase.MOVEMENT,
) -> GameState:
    """Create a basic hex game state for testing."""
    players = [
        Player(
            id="p1",
            username="Player 1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=600,
            ringsInHand=30,
            eliminatedRings=0,
            territorySpaces=0,
            aiDifficulty=10,
        ),
        Player(
            id="p2",
            username="Player 2",
            type="human",
            playerNumber=2,
            isReady=True,
            timeRemaining=600,
            ringsInHand=30,
            eliminatedRings=0,
            territorySpaces=0,
            aiDifficulty=10,
        ),
    ]

    board = BoardState(
        type=BoardType.HEXAGONAL,
        size=size,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
    )

    return GameState(
        id="test-hex",
        boardType=BoardType.HEXAGONAL,
        board=board,
        players=players,
        currentPhase=phase,
        currentPlayer=current_player,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=60,
        totalRingsEliminated=0,
        victoryThreshold=31,
        territoryVictoryThreshold=166,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        rngSeed=42,
        zobristHash=0,
        lpsRoundIndex=0,
        lpsExclusivePlayerForCompletedRound=None,
    )


class TestHexStateEncoder:
    """Test suite for HexStateEncoder class."""

    @pytest.fixture
    def encoder(self):
        """Create a HexStateEncoder instance."""
        return HexStateEncoder()

    @pytest.fixture
    def hex_state(self):
        """Create a test hex game state."""
        return create_hex_game_state()

    def test_encoder_initialization(self, encoder):
        """Test encoder initializes with correct parameters."""
        assert encoder.board_size == HEX_BOARD_SIZE  # 21
        assert encoder.radius == 10
        assert encoder.POLICY_SIZE == P_HEX
        assert encoder.NUM_CHANNELS == 10
        assert encoder.NUM_GLOBAL_FEATURES == 10

    def test_valid_mask_shape(self, encoder):
        """Test valid mask has correct shape."""
        mask = encoder.get_valid_mask()
        assert mask.shape == (HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        assert mask.dtype == bool

    def test_valid_mask_center_is_valid(self, encoder):
        """Test that center cell (0,0) is marked as valid."""
        mask = encoder.get_valid_mask()
        center = encoder.radius  # 10
        assert mask[center, center]  # True for valid hex cell

    def test_valid_mask_corners_invalid(self, encoder):
        """Test that extreme corners of bounding box are invalid."""
        mask = encoder.get_valid_mask()
        # Corners like (0,0) and (20,20) should be outside hex region
        # q=-10, r=-10, s=20 → s > radius → invalid
        assert not mask[0, 0]
        assert not mask[20, 20]
        # But (0,20) is edge: q=10, r=-10, s=0 → all <= radius → valid
        assert mask[0, 20]
        assert mask[20, 0]

    def test_valid_mask_cell_count(self, encoder):
        """Test correct number of valid hex cells."""
        mask = encoder.get_valid_mask()
        # For radius N: 3N^2 + 3N + 1 = 331 for N=10
        expected_count = 3 * 10 * 10 + 3 * 10 + 1
        assert np.sum(mask) == expected_count

    def test_axial_to_canonical_center(self, encoder):
        """Test axial (0,0) maps to center of grid."""
        cx, cy = encoder.axial_to_canonical(0, 0)
        assert cx == 10
        assert cy == 10

    def test_canonical_to_axial_center(self, encoder):
        """Test center of grid maps to axial (0,0)."""
        q, r = encoder.canonical_to_axial(10, 10)
        assert q == 0
        assert r == 0

    def test_roundtrip_coordinates(self, encoder):
        """Test axial->canonical->axial roundtrip."""
        for q in range(-5, 6):
            for r in range(-5, 6):
                if abs(q) + abs(r) + abs(-q - r) <= 20:  # Within radius
                    cx, cy = encoder.axial_to_canonical(q, r)
                    q2, r2 = encoder.canonical_to_axial(cx, cy)
                    assert q == q2 and r == r2

    def test_encode_empty_state(self, encoder, hex_state):
        """Test encoding an empty hex game state."""
        features, globals_vec = encoder.encode(hex_state)

        # Check shapes
        assert features.shape == (10, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        assert globals_vec.shape == (10,)

        # All features should be zero for empty board
        assert np.allclose(features, 0.0)

        # Global features should have phase encoding
        phase_vec = globals_vec[:5]
        assert np.sum(phase_vec) == 1.0  # Exactly one phase active

    def test_encode_with_stacks(self, encoder, hex_state):
        """Test encoding state with stacks."""
        # Add a stack at center
        hex_state.board.stacks["0,0,0"] = RingStack(
            position=Position(x=0, y=0, z=0),
            rings=[1, 1, 1],
            stackHeight=3,
            capHeight=3,
            controllingPlayer=1,
        )

        features, _ = encoder.encode(hex_state)

        # Check player 1's stack channel (index 0)
        center = encoder.radius
        assert features[0, center, center] > 0  # Player 1 stack

        # Player 2 should have no stacks
        assert np.sum(features[1]) == 0

    def test_encode_with_markers(self, encoder, hex_state):
        """Test encoding state with markers."""
        # Add markers at different positions
        pos1 = Position(x=0, y=0, z=0)
        pos2 = Position(x=1, y=0, z=-1)
        hex_state.board.markers["0,0,0"] = MarkerInfo(
            player=1, position=pos1, type="regular"
        )
        hex_state.board.markers["1,0,-1"] = MarkerInfo(
            player=2, position=pos2, type="regular"
        )

        features, _ = encoder.encode(hex_state)

        # Check marker channels (2 and 3)
        center = encoder.radius
        assert features[2, center, center] == 1.0  # Player 1 marker
        assert features[3, center, center + 1] == 1.0  # Player 2 marker

    def test_encode_with_collapsed_spaces(self, encoder, hex_state):
        """Test encoding state with collapsed spaces."""
        hex_state.board.collapsed_spaces["0,0,0"] = 1

        features, _ = encoder.encode(hex_state)

        center = encoder.radius
        assert features[4, center, center] == 1.0  # Player 1 territory

    def test_encode_global_features(self, encoder, hex_state):
        """Test global feature encoding."""
        # Set up players with specific ring counts
        hex_state.players[0].rings_in_hand = 18
        hex_state.players[1].rings_in_hand = 20

        _, globals_vec = encoder.encode(hex_state)

        # Check normalized ring counts
        assert globals_vec[5] == 18.0 / 36.0  # Current player rings
        assert globals_vec[6] == 20.0 / 36.0  # Opponent rings

    def test_encode_with_history(self, encoder, hex_state):
        """Test encoding with history frames."""
        # Create current frame
        current_features, globals_vec = encoder.encode(hex_state)

        # Create dummy history
        history = [
            np.random.rand(10, HEX_BOARD_SIZE, HEX_BOARD_SIZE).astype(
                np.float32
            )
            for _ in range(3)
        ]

        stacked, _ = encoder.encode_with_history(hex_state, history)

        # Should have 10 * (3 + 1) = 40 channels
        assert stacked.shape == (40, HEX_BOARD_SIZE, HEX_BOARD_SIZE)

    def test_rejects_non_hex_board(self, encoder):
        """Test encoder rejects non-hexagonal boards."""
        state = create_hex_game_state()
        state.board.type = BoardType.SQUARE8

        with pytest.raises(ValueError, match="HEXAGONAL"):
            encoder.encode(state)


class TestActionEncoderHex:
    """Test suite for ActionEncoderHex class."""

    @pytest.fixture
    def encoder(self):
        """Create an ActionEncoderHex instance."""
        return ActionEncoderHex()

    @pytest.fixture
    def hex_board(self):
        """Create a test hex board."""
        return BoardState(
            type=BoardType.HEXAGONAL,
            size=11,
            stacks={},
            markers={},
            collapsedSpaces={},
            eliminatedRings={},
        )

    @pytest.fixture
    def hex_state(self):
        """Create a test hex game state."""
        return create_hex_game_state()

    def test_encode_placement_center(self, encoder, hex_board):
        """Test encoding placement at center."""
        move = Move(
            id="test",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=0, y=0, z=0),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
            placementCount=1,
        )

        idx = encoder.encode_move(move, hex_board)

        # Center (cx=10, cy=10) with count=1
        # pos_idx = 10 * 21 + 10 = 220
        # idx = 220 * 3 + 0 = 660
        expected = 10 * HEX_BOARD_SIZE * 3 + 10 * 3 + 0
        assert idx == expected

    def test_encode_placement_with_count(self, encoder, hex_board):
        """Test encoding placement with different counts."""
        for count in [1, 2, 3]:
            move = Move(
                id="test",
                type=MoveType.PLACE_RING,
                player=1,
                to=Position(x=0, y=0, z=0),
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=1,
                placementCount=count,
            )

            idx = encoder.encode_move(move, hex_board)

            # Check count offset
            expected_offset = count - 1
            assert idx % 3 == expected_offset

    def test_decode_placement(self, encoder, hex_state):
        """Test decoding placement moves."""
        # Center placement with count 1
        idx = 10 * HEX_BOARD_SIZE * 3 + 10 * 3 + 0

        move = encoder.decode_move(idx, hex_state)

        assert move is not None
        assert move.type == MoveType.PLACE_RING
        assert move.to.x == 0
        assert move.to.y == 0
        assert move.placement_count == 1

    def test_encode_skip_placement(self, encoder, hex_board):
        """Test encoding skip placement action."""
        move = Move(
            id="test",
            type=MoveType.SKIP_PLACEMENT,
            player=1,
            to=Position(x=0, y=0),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
        )

        idx = encoder.encode_move(move, hex_board)

        # Skip is at HEX_SPECIAL_BASE = P_HEX - 1
        assert idx == P_HEX - 1

    def test_roundtrip_placement(self, encoder, hex_state):
        """Test encode-decode roundtrip for placements."""
        hex_board = hex_state.board

        # Test various positions within hex
        for q in range(-5, 6, 2):
            for r in range(-5, 6, 2):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= 10:
                    move = Move(
                        id="test",
                        type=MoveType.PLACE_RING,
                        player=1,
                        to=Position(x=q, y=r, z=s),
                        timestamp=datetime.now(),
                        thinkTime=0,
                        moveNumber=1,
                        placementCount=1,
                    )

                    idx = encoder.encode_move(move, hex_board)
                    decoded = encoder.decode_move(idx, hex_state)

                    assert decoded is not None
                    assert decoded.to.x == q
                    assert decoded.to.y == r

    def test_rejects_square_board(self, encoder):
        """Test encoder rejects square boards."""
        board = BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks={},
            markers={},
            collapsedSpaces={},
            eliminatedRings={},
        )

        move = Move(
            id="test",
            type=MoveType.PLACE_RING,
            player=1,
            to=Position(x=0, y=0),
            timestamp=datetime.now(),
            thinkTime=0,
            moveNumber=1,
            placementCount=1,
        )

        idx = encoder.encode_move(move, board)
        assert idx == -1  # INVALID_MOVE_INDEX


class TestHexNeuralNet:
    """Test suite for HexNeuralNet model."""

    @pytest.fixture
    def model(self):
        """Create a HexNeuralNet instance."""
        return HexNeuralNet(
            in_channels=40,  # 10 * (3 history + 1 current)
            global_features=10,
            num_res_blocks=2,  # Smaller for testing
            num_filters=32,  # Smaller for testing
        )

    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.board_size == HEX_BOARD_SIZE
        assert model.policy_size == P_HEX

    def test_forward_pass(self, model):
        """Test forward pass produces correct output shapes."""
        batch_size = 4
        x = torch.randn(batch_size, 40, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        globals_vec = torch.randn(batch_size, 10)

        value, policy = model(x, globals_vec)

        assert value.shape == (batch_size, 1)
        assert policy.shape == (batch_size, P_HEX)

    def test_forward_with_mask(self, model):
        """Test forward pass with hex mask."""
        batch_size = 2
        x = torch.randn(batch_size, 40, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        globals_vec = torch.randn(batch_size, 10)

        # Create valid hex mask
        encoder = HexStateEncoder()
        mask = encoder.get_valid_mask_tensor()
        hex_mask = torch.from_numpy(mask).unsqueeze(0)
        hex_mask = hex_mask.expand(batch_size, -1, -1, -1)

        value, policy = model(x, globals_vec, hex_mask=hex_mask)

        assert value.shape == (batch_size, 1)
        assert policy.shape == (batch_size, P_HEX)

    def test_value_in_range(self, model):
        """Test value output is in [-1, 1] range."""
        x = torch.randn(4, 40, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        globals_vec = torch.randn(4, 10)

        value, _ = model(x, globals_vec)

        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_training_step(self, model):
        """Test a single training step."""
        model.train()

        batch_size = 4
        x = torch.randn(batch_size, 40, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        globals_vec = torch.randn(batch_size, 10)

        # Targets
        value_target = torch.randn(batch_size, 1)
        policy_target = torch.softmax(torch.randn(batch_size, P_HEX), dim=1)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Forward
        value_pred, policy_logits = model(x, globals_vec)

        # Loss
        value_loss = torch.nn.MSELoss()(value_pred, value_target)
        policy_log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = torch.nn.KLDivLoss(reduction='batchmean')(
            policy_log_probs, policy_target
        )
        loss = value_loss + policy_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert loss.item() > 0  # Loss should be positive

    def test_gradient_flow(self, model):
        """Test gradients flow through entire network."""
        x = torch.randn(
            2, 40, HEX_BOARD_SIZE, HEX_BOARD_SIZE, requires_grad=True
        )
        globals_vec = torch.randn(2, 10, requires_grad=True)

        value, policy = model(x, globals_vec)

        # Compute loss
        loss = value.sum() + policy.sum()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad: {name}"
                zero_grad = torch.all(param.grad == 0)
                assert not zero_grad, f"Zero grad: {name}"


class TestHexDataAugmentation:
    """Test hex data augmentation integration."""

    @pytest.fixture
    def transform(self):
        """Create a HexSymmetryTransform instance."""
        return HexSymmetryTransform()

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        features = np.random.rand(10, HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        features = features.astype(np.float32)
        globals_vec = np.random.rand(10).astype(np.float32)
        policy_indices = np.array([0, 100, 500], dtype=np.int32)
        policy_values = np.array([0.3, 0.5, 0.2], dtype=np.float32)
        return features, globals_vec, policy_indices, policy_values

    def test_augmentation_produces_12_samples(self, sample_data):
        """Test that augmentation produces exactly 12 samples."""
        features, globals_vec, policy_indices, policy_values = sample_data

        results = augment_hex_sample(
            features, globals_vec, policy_indices, policy_values
        )

        assert len(results) == 12

    def test_augmentation_preserves_shapes(self, sample_data):
        """Test that all augmented samples have correct shapes."""
        features, globals_vec, policy_indices, policy_values = sample_data

        results = augment_hex_sample(
            features, globals_vec, policy_indices, policy_values
        )

        for i, (aug_feat, aug_glob, aug_idx, aug_val) in enumerate(results):
            assert aug_feat.shape == features.shape, f"Sample {i}"
            assert aug_glob.shape == globals_vec.shape, f"Sample {i}"

    def test_augmentation_first_is_identity(self, sample_data):
        """Test that first augmented sample is identity."""
        features, globals_vec, policy_indices, policy_values = sample_data

        results = augment_hex_sample(
            features, globals_vec, policy_indices, policy_values
        )

        aug_feat, aug_glob, aug_idx, aug_val = results[0]

        np.testing.assert_array_almost_equal(aug_feat, features)
        np.testing.assert_array_almost_equal(aug_glob, globals_vec)

    def test_all_12_symmetries_different(self, sample_data, transform):
        """Test that all 12 transformations produce different boards."""
        features, globals_vec, policy_indices, policy_values = sample_data

        # Create hex mask to focus on valid cells
        mask = np.zeros((HEX_BOARD_SIZE, HEX_BOARD_SIZE), dtype=bool)
        radius = 10
        for cy in range(HEX_BOARD_SIZE):
            for cx in range(HEX_BOARD_SIZE):
                q = cx - radius
                r = cy - radius
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= radius:
                    mask[cy, cx] = True

        results = augment_hex_sample(
            features, globals_vec, policy_indices, policy_values
        )

        # Check that at least non-identity transforms are different
        for i in range(1, 12):
            aug_feat_i = results[i][0]
            # Should be different from identity
            diff = np.abs(aug_feat_i[:, mask] - features[:, mask]).sum()
            assert diff > 0.1, f"Transform {i} too similar to identity"


class TestBoardTypeDetection:
    """Test board type detection from feature shapes."""

    def test_detect_square8(self):
        """Test detection of square8 from 8x8 features."""
        features = np.random.rand(10, 8, 8)
        board_type = detect_board_type_from_features(features)
        assert board_type == BoardType.SQUARE8

    def test_detect_square19(self):
        """Test detection of square19 from 19x19 features."""
        features = np.random.rand(10, 19, 19)
        board_type = detect_board_type_from_features(features)
        assert board_type == BoardType.SQUARE19

    def test_detect_hexagonal(self):
        """Test detection of hexagonal from 21x21 features."""
        features = np.random.rand(10, 21, 21)
        board_type = detect_board_type_from_features(features)
        assert board_type == BoardType.HEXAGONAL

    def test_detect_with_batch_dimension(self):
        """Test detection with batch dimension."""
        features = np.random.rand(32, 10, 21, 21)
        board_type = detect_board_type_from_features(features)
        assert board_type == BoardType.HEXAGONAL

    def test_unknown_size_raises(self):
        """Test unknown sizes raise ValueError."""
        features = np.random.rand(10, 15, 15)
        with pytest.raises(ValueError, match="Cannot detect"):
            detect_board_type_from_features(features)


class TestEncoderFactory:
    """Test encoder factory function."""

    def test_returns_hex_encoder_for_hexagonal(self):
        """Test factory returns HexStateEncoder for hex boards."""
        encoder = get_encoder_for_board_type(BoardType.HEXAGONAL)
        assert isinstance(encoder, HexStateEncoder)

    def test_returns_none_for_square(self):
        """Test factory returns None for square boards."""
        encoder = get_encoder_for_board_type(BoardType.SQUARE8)
        assert encoder is None

        encoder = get_encoder_for_board_type(BoardType.SQUARE19)
        assert encoder is None


class TestHexTrainingIntegration:
    """Integration tests for complete hex training pipeline."""

    def test_full_encode_augment_train_cycle(self):
        """Test complete cycle: encode -> augment -> train."""
        # Create game state
        state = create_hex_game_state()
        state.board.stacks["0,0,0"] = RingStack(
            position=Position(x=0, y=0, z=0),
            rings=[1, 1],
            stackHeight=2,
            capHeight=2,
            controllingPlayer=1,
        )

        # Encode
        encoder = HexStateEncoder()
        features, globals_vec = encoder.encode(state)

        # Create dummy policy
        policy_indices = np.array([100, 200, 300], dtype=np.int32)
        policy_values = np.array([0.5, 0.3, 0.2], dtype=np.float32)

        # Augment
        augmented = augment_hex_sample(
            features, globals_vec, policy_indices, policy_values
        )

        assert len(augmented) == 12

        # Create model
        model = HexNeuralNet(
            in_channels=10,  # Single frame
            global_features=10,
            num_res_blocks=1,
            num_filters=16,
        )
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train on augmented samples
        for aug_feat, aug_glob, aug_idx, aug_val in augmented:
            x = torch.from_numpy(aug_feat).unsqueeze(0)
            g = torch.from_numpy(aug_glob).unsqueeze(0)

            value_pred, policy_logits = model(x, g)

            # Simple loss
            loss = value_pred.mean() + policy_logits.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Should complete without error


class TestMoveEncodingEdgeCases:
    """Test edge cases in move encoding."""

    @pytest.fixture
    def encoder(self):
        return ActionEncoderHex()

    @pytest.fixture
    def hex_state(self):
        return create_hex_game_state()

    def test_boundary_positions(self, encoder, hex_state):
        """Test encoding moves at hex boundary."""
        hex_board = hex_state.board
        radius = 10

        boundary_positions = [
            (radius, 0, -radius),
            (-radius, 0, radius),
            (0, radius, -radius),
            (0, -radius, radius),
            (radius, -radius, 0),
            (-radius, radius, 0),
        ]

        for q, r, s in boundary_positions:
            move = Move(
                id="test",
                type=MoveType.PLACE_RING,
                player=1,
                to=Position(x=q, y=r, z=s),
                timestamp=datetime.now(),
                thinkTime=0,
                moveNumber=1,
                placementCount=1,
            )

            idx = encoder.encode_move(move, hex_board)
            assert 0 <= idx < P_HEX, f"Invalid idx for ({q},{r},{s})"

            decoded = encoder.decode_move(idx, hex_state)
            assert decoded is not None
            assert decoded.to.x == q
            assert decoded.to.y == r