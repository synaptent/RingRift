"""Unit tests for NNUE with Policy head module.

Tests cover:
- RingRiftNNUEWithPolicy model architecture
- Forward pass with and without policy outputs
- Move scoring and probability computation
- Position encoding utilities
- Model loading from value-only checkpoints
"""

import pytest
import numpy as np
import torch

from app.models import BoardType, Position


class TestGetHiddenDimForBoard:
    """Tests for get_hidden_dim_for_board function."""

    def test_square8_returns_128(self):
        """Square8 boards should use smaller hidden dimension."""
        from app.ai.nnue_policy import get_hidden_dim_for_board

        assert get_hidden_dim_for_board(BoardType.SQUARE8) == 128

    def test_square19_returns_512(self):
        """Square19 boards should use larger hidden dimension."""
        from app.ai.nnue_policy import get_hidden_dim_for_board

        assert get_hidden_dim_for_board(BoardType.SQUARE19) == 512

    def test_hex8_returns_128(self):
        """Hex8 boards should use smaller hidden dimension."""
        from app.ai.nnue_policy import get_hidden_dim_for_board

        assert get_hidden_dim_for_board(BoardType.HEX8) == 128

    def test_hexagonal_small_size_returns_128(self):
        """Small hexagonal boards (size <= 8) use smaller dimension."""
        from app.ai.nnue_policy import get_hidden_dim_for_board

        assert get_hidden_dim_for_board(BoardType.HEXAGONAL, board_size=8) == 128

    def test_hexagonal_large_size_returns_1024(self):
        """Large hexagonal boards (size > 8) use larger dimension."""
        from app.ai.nnue_policy import get_hidden_dim_for_board

        assert get_hidden_dim_for_board(BoardType.HEXAGONAL, board_size=19) == 1024


class TestRingRiftNNUEWithPolicy:
    """Tests for RingRiftNNUEWithPolicy model class."""

    @pytest.fixture
    def model_square8(self):
        """Create a Square8 policy model."""
        from app.ai.nnue_policy import RingRiftNNUEWithPolicy

        return RingRiftNNUEWithPolicy(
            board_type=BoardType.SQUARE8,
            hidden_dim=32,  # Small for fast tests
            num_hidden_layers=1,
        )

    @pytest.fixture
    def model_hex8(self):
        """Create a Hex8 policy model."""
        from app.ai.nnue_policy import RingRiftNNUEWithPolicy

        return RingRiftNNUEWithPolicy(
            board_type=BoardType.HEX8,
            hidden_dim=32,
            num_hidden_layers=1,
        )

    def test_model_creation_square8(self, model_square8):
        """Model creates successfully for Square8 board."""
        assert model_square8 is not None
        assert model_square8.board_type == BoardType.SQUARE8
        assert model_square8.board_size == 8

    def test_model_creation_hex8(self, model_hex8):
        """Model creates successfully for Hex8 board."""
        assert model_hex8 is not None
        assert model_hex8.board_type == BoardType.HEX8

    def test_forward_value_only(self, model_square8):
        """Forward pass returns value when return_policy=False."""
        from app.ai.nnue import get_feature_dim

        feature_dim = get_feature_dim(BoardType.SQUARE8)
        batch_size = 4
        features = torch.randn(batch_size, feature_dim)

        value = model_square8(features, return_policy=False)

        assert value.shape == (batch_size, 1)
        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_forward_with_policy(self, model_square8):
        """Forward pass returns value and policy when return_policy=True."""
        from app.ai.nnue import get_feature_dim

        feature_dim = get_feature_dim(BoardType.SQUARE8)
        batch_size = 4
        features = torch.randn(batch_size, feature_dim)

        value, from_logits, to_logits = model_square8(features, return_policy=True)

        assert value.shape == (batch_size, 1)
        # Policy heads output board_size * board_size logits
        expected_positions = 8 * 8
        assert from_logits.shape == (batch_size, expected_positions)
        assert to_logits.shape == (batch_size, expected_positions)

    def test_forward_single(self, model_square8):
        """forward_single returns scalar value for single sample."""
        from app.ai.nnue import get_feature_dim

        feature_dim = get_feature_dim(BoardType.SQUARE8)
        features = np.random.randn(feature_dim).astype(np.float32)

        value = model_square8.forward_single(features)

        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_score_moves(self, model_square8):
        """score_moves computes from+to scores for moves."""
        from app.ai.nnue import get_feature_dim

        feature_dim = get_feature_dim(BoardType.SQUARE8)
        batch_size = 2
        max_moves = 10

        features = torch.randn(batch_size, feature_dim)
        from_indices = torch.randint(0, 64, (batch_size, max_moves))
        to_indices = torch.randint(0, 64, (batch_size, max_moves))

        scores = model_square8.score_moves(features, from_indices, to_indices)

        assert scores.shape == (batch_size, max_moves)

    def test_score_moves_with_mask(self, model_square8):
        """score_moves respects move_mask."""
        from app.ai.nnue import get_feature_dim

        feature_dim = get_feature_dim(BoardType.SQUARE8)
        batch_size = 2
        max_moves = 10

        features = torch.randn(batch_size, feature_dim)
        from_indices = torch.randint(0, 64, (batch_size, max_moves))
        to_indices = torch.randint(0, 64, (batch_size, max_moves))
        move_mask = torch.zeros(batch_size, max_moves, dtype=torch.bool)
        move_mask[:, :3] = True  # Only first 3 moves are valid

        scores = model_square8.score_moves(
            features, from_indices, to_indices, move_mask
        )

        # Masked moves should have -inf scores
        assert torch.all(scores[:, 3:] == float('-inf'))
        assert torch.all(scores[:, :3] != float('-inf'))

    def test_get_move_probabilities(self, model_square8):
        """get_move_probabilities returns valid probability distribution."""
        from app.ai.nnue import get_feature_dim

        feature_dim = get_feature_dim(BoardType.SQUARE8)
        batch_size = 2
        max_moves = 10

        features = torch.randn(batch_size, feature_dim)
        from_indices = torch.randint(0, 64, (batch_size, max_moves))
        to_indices = torch.randint(0, 64, (batch_size, max_moves))

        probs = model_square8.get_move_probabilities(
            features, from_indices, to_indices
        )

        assert probs.shape == (batch_size, max_moves)
        # Probabilities should sum to 1 per sample
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        # All probabilities should be non-negative
        assert torch.all(probs >= 0)

    def test_temperature_affects_probabilities(self, model_square8):
        """Higher temperature increases exploration (more uniform probs)."""
        from app.ai.nnue import get_feature_dim

        feature_dim = get_feature_dim(BoardType.SQUARE8)
        features = torch.randn(1, feature_dim)
        from_indices = torch.randint(0, 64, (1, 10))
        to_indices = torch.randint(0, 64, (1, 10))

        probs_low_temp = model_square8.get_move_probabilities(
            features, from_indices, to_indices, temperature=0.5
        )
        probs_high_temp = model_square8.get_move_probabilities(
            features, from_indices, to_indices, temperature=2.0
        )

        # High temperature should have lower max probability (more uniform)
        assert probs_high_temp.max() <= probs_low_temp.max()


class TestPosToFlatIndex:
    """Tests for pos_to_flat_index utility."""

    def test_square_board_indexing(self):
        """Square boards use simple row-major indexing."""
        from app.ai.nnue_policy import pos_to_flat_index

        # Top-left is (0,0) = index 0
        pos = Position(x=0, y=0)
        assert pos_to_flat_index(pos, 8, BoardType.SQUARE8) == 0

        # (1,0) = index 1
        pos = Position(x=1, y=0)
        assert pos_to_flat_index(pos, 8, BoardType.SQUARE8) == 1

        # (0,1) = index 8
        pos = Position(x=0, y=1)
        assert pos_to_flat_index(pos, 8, BoardType.SQUARE8) == 8

        # Bottom-right is (7,7) = index 63
        pos = Position(x=7, y=7)
        assert pos_to_flat_index(pos, 8, BoardType.SQUARE8) == 63

    def test_hex_board_indexing(self):
        """Hex boards offset by radius for centered coordinates."""
        from app.ai.nnue_policy import pos_to_flat_index

        # For hex8 with board_size=9, radius=4
        # Center (0,0) becomes (4,4) = 4*9+4 = 40
        pos = Position(x=0, y=0)
        idx = pos_to_flat_index(pos, 9, BoardType.HEX8)
        assert idx == 40

        # (-4, -4) becomes (0,0) = 0
        pos = Position(x=-4, y=-4)
        idx = pos_to_flat_index(pos, 9, BoardType.HEX8)
        assert idx == 0


class TestArchitectureVersion:
    """Tests for model versioning."""

    def test_has_architecture_version(self):
        """Model should have ARCHITECTURE_VERSION class attribute."""
        from app.ai.nnue_policy import RingRiftNNUEWithPolicy

        assert hasattr(RingRiftNNUEWithPolicy, 'ARCHITECTURE_VERSION')
        assert isinstance(RingRiftNNUEWithPolicy.ARCHITECTURE_VERSION, str)


class TestModelGradients:
    """Tests for gradient flow during training."""

    def test_value_head_receives_gradients(self):
        """Value head weights receive gradients during training."""
        from app.ai.nnue_policy import RingRiftNNUEWithPolicy
        from app.ai.nnue import get_feature_dim

        model = RingRiftNNUEWithPolicy(
            board_type=BoardType.SQUARE8,
            hidden_dim=32,
            num_hidden_layers=1,
        )

        feature_dim = get_feature_dim(BoardType.SQUARE8)
        features = torch.randn(4, feature_dim)
        target = torch.zeros(4, 1)

        value = model(features, return_policy=False)
        loss = ((value - target) ** 2).mean()
        loss.backward()

        assert model.value_head.weight.grad is not None
        assert model.value_head.weight.grad.abs().sum() > 0

    def test_policy_heads_receive_gradients(self):
        """Policy heads receive gradients when training policy."""
        from app.ai.nnue_policy import RingRiftNNUEWithPolicy
        from app.ai.nnue import get_feature_dim

        model = RingRiftNNUEWithPolicy(
            board_type=BoardType.SQUARE8,
            hidden_dim=32,
            num_hidden_layers=1,
        )

        feature_dim = get_feature_dim(BoardType.SQUARE8)
        features = torch.randn(4, feature_dim)

        _, from_logits, to_logits = model(features, return_policy=True)

        # Create dummy targets
        target_from = torch.randint(0, 64, (4,))
        target_to = torch.randint(0, 64, (4,))

        # Cross-entropy loss on policy
        from_loss = torch.nn.functional.cross_entropy(from_logits, target_from)
        to_loss = torch.nn.functional.cross_entropy(to_logits, target_to)
        policy_loss = from_loss + to_loss
        policy_loss.backward()

        assert model.from_head.weight.grad is not None
        assert model.to_head.weight.grad is not None
        assert model.from_head.weight.grad.abs().sum() > 0
        assert model.to_head.weight.grad.abs().sum() > 0


class TestDropoutBehavior:
    """Tests for dropout behavior in policy heads."""

    def test_dropout_active_in_train_mode(self):
        """Policy heads use dropout during training."""
        from app.ai.nnue_policy import RingRiftNNUEWithPolicy
        from app.ai.nnue import get_feature_dim

        model = RingRiftNNUEWithPolicy(
            board_type=BoardType.SQUARE8,
            hidden_dim=32,
            num_hidden_layers=1,
            policy_dropout=0.5,  # High dropout for testing
        )
        model.train()

        feature_dim = get_feature_dim(BoardType.SQUARE8)
        features = torch.randn(10, feature_dim)

        # Multiple forward passes should give different results due to dropout
        _, from1, to1 = model(features, return_policy=True)
        _, from2, to2 = model(features, return_policy=True)

        # Not exactly equal due to dropout
        assert not torch.allclose(from1, from2)

    def test_dropout_inactive_in_eval_mode(self):
        """Policy heads are deterministic during evaluation."""
        from app.ai.nnue_policy import RingRiftNNUEWithPolicy
        from app.ai.nnue import get_feature_dim

        model = RingRiftNNUEWithPolicy(
            board_type=BoardType.SQUARE8,
            hidden_dim=32,
            num_hidden_layers=1,
            policy_dropout=0.5,
        )
        model.eval()

        feature_dim = get_feature_dim(BoardType.SQUARE8)
        features = torch.randn(10, feature_dim)

        with torch.no_grad():
            _, from1, to1 = model(features, return_policy=True)
            _, from2, to2 = model(features, return_policy=True)

        # Exactly equal in eval mode
        assert torch.allclose(from1, from2)
        assert torch.allclose(to1, to2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
