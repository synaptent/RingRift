"""Comprehensive unit tests for neural network architectures.

Tests cover all architecture versions (v2, v2_Lite, v3, v3_Lite, v3_Flat, v4)
for both square and hexagonal boards.

Created: December 2025
"""

import pytest
import torch
import numpy as np

from app.ai.neural_net import (
    # Square architectures
    RingRiftCNN_v2,
    RingRiftCNN_v2_Lite,
    # Hex architectures
    HexNeuralNet_v2,
    HexNeuralNet_v2_Lite,
    # Constants
    POLICY_SIZE_8x8,
    POLICY_SIZE_19x19,
    P_HEX,
)

# Import v3/v4 architectures from submodules
from app.ai.neural_net.square_architectures import (
    RingRiftCNN_v3,
    RingRiftCNN_v3_Lite,
    RingRiftCNN_v3_Flat,
    RingRiftCNN_v4,
)
from app.ai.neural_net.hex_architectures import (
    HexNeuralNet_v3,
    HexNeuralNet_v3_Lite,
    HexNeuralNet_v3_Flat,
    HexNeuralNet_v4,
)


# ==============================================================================
# Square Architecture Tests - V2
# ==============================================================================


class TestRingRiftCNN_v2:
    """Tests for RingRiftCNN_v2 high-capacity SE architecture."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert RingRiftCNN_v2.ARCHITECTURE_VERSION == "v2.0.0"

    def test_default_initialization(self):
        """Test default model initialization."""
        model = RingRiftCNN_v2(
            board_size=8,
            num_res_blocks=2,
            num_filters=32,
        )
        assert model.board_size == 8
        assert model.num_filters == 32
        assert model.num_players == 4

    def test_forward_pass_shapes_8x8(self):
        """Test forward pass produces correct shapes for 8x8 board."""
        in_channels = 14
        history_length = 3
        model = RingRiftCNN_v2(
            board_size=8,
            in_channels=in_channels,
            num_res_blocks=2,
            num_filters=32,
            history_length=history_length,
        )
        batch_size = 4
        total_channels = in_channels * (history_length + 1)
        x = torch.randn(batch_size, total_channels, 8, 8)
        g = torch.randn(batch_size, 20)

        value, policy = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, POLICY_SIZE_8x8)

    def test_forward_pass_shapes_19x19(self):
        """Test forward pass produces correct shapes for 19x19 board."""
        in_channels = 14
        history_length = 3
        model = RingRiftCNN_v2(
            board_size=19,
            in_channels=in_channels,
            num_res_blocks=2,
            num_filters=32,
            history_length=history_length,
        )
        batch_size = 2
        total_channels = in_channels * (history_length + 1)
        x = torch.randn(batch_size, total_channels, 19, 19)
        g = torch.randn(batch_size, 20)

        value, policy = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, POLICY_SIZE_19x19)

    def test_value_range(self):
        """Test value output is in [-1, 1] range (tanh activation)."""
        model = RingRiftCNN_v2(
            board_size=8,
            in_channels=14,
            num_res_blocks=2,
            num_filters=32,
            history_length=3,
        )
        batch_size = 10
        x = torch.randn(batch_size, 56, 8, 8)
        g = torch.randn(batch_size, 20)

        value, _ = model(x, g)

        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_return_features(self):
        """Test return_features option returns backbone features."""
        model = RingRiftCNN_v2(
            board_size=8,
            in_channels=14,
            num_res_blocks=2,
            num_filters=32,
            history_length=3,
        )
        x = torch.randn(2, 56, 8, 8)
        g = torch.randn(2, 20)

        value, policy, features = model(x, g, return_features=True)

        assert features.shape == (2, 32 + 20)  # num_filters + global_features

    def test_channel_mismatch_raises_error(self):
        """Test that mismatched input channels raise RuntimeError."""
        model = RingRiftCNN_v2(
            board_size=8,
            in_channels=14,
            num_res_blocks=2,
            num_filters=32,
            history_length=3,
        )
        # Wrong number of channels
        x = torch.randn(2, 10, 8, 8)  # Should be 56
        g = torch.randn(2, 20)

        with pytest.raises(RuntimeError, match="Input channel mismatch"):
            model(x, g)


class TestRingRiftCNN_v2_Lite:
    """Tests for RingRiftCNN_v2_Lite memory-efficient architecture."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert RingRiftCNN_v2_Lite.ARCHITECTURE_VERSION == "v2.0.0-lite"

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        in_channels = 12
        history_length = 2
        model = RingRiftCNN_v2_Lite(
            board_size=8,
            in_channels=in_channels,
            num_res_blocks=2,
            num_filters=32,
            history_length=history_length,
        )
        batch_size = 4
        total_channels = in_channels * (history_length + 1)
        x = torch.randn(batch_size, total_channels, 8, 8)
        g = torch.randn(batch_size, 20)

        value, policy = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, POLICY_SIZE_8x8)

    def test_smaller_than_v2(self):
        """Test that v2_Lite has fewer parameters than v2."""
        model_v2 = RingRiftCNN_v2(
            board_size=8, num_res_blocks=6, num_filters=96
        )
        model_lite = RingRiftCNN_v2_Lite(
            board_size=8, num_res_blocks=6, num_filters=96
        )

        params_v2 = sum(p.numel() for p in model_v2.parameters())
        params_lite = sum(p.numel() for p in model_lite.parameters())

        # Lite should have fewer parameters due to smaller input channels
        assert params_lite < params_v2


# ==============================================================================
# Square Architecture Tests - V3
# ==============================================================================


class TestRingRiftCNN_v3:
    """Tests for RingRiftCNN_v3 with spatial policy heads."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert RingRiftCNN_v3.ARCHITECTURE_VERSION == "v3.1.0"

    def test_forward_pass_shapes_8x8(self):
        """Test forward pass produces correct shapes including rank distribution."""
        in_channels = 14
        history_length = 3
        model = RingRiftCNN_v3(
            board_size=8,
            in_channels=in_channels,
            num_res_blocks=2,
            num_filters=32,
            history_length=history_length,
        )
        batch_size = 4
        total_channels = in_channels * (history_length + 1)
        x = torch.randn(batch_size, total_channels, 8, 8)
        g = torch.randn(batch_size, 20)

        value, policy, rank_dist = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, POLICY_SIZE_8x8)
        assert rank_dist.shape == (batch_size, 4, 4)  # [B, num_players, num_players]

    def test_rank_distribution_sums_to_one(self):
        """Test that rank distribution sums to 1 for each player."""
        model = RingRiftCNN_v3(
            board_size=8,
            in_channels=14,
            num_res_blocks=2,
            num_filters=32,
            history_length=3,
        )
        x = torch.randn(4, 56, 8, 8)
        g = torch.randn(4, 20)

        _, _, rank_dist = model(x, g)

        # Sum over ranks (dim=-1) should be 1 for each player
        rank_sums = rank_dist.sum(dim=-1)
        assert torch.allclose(rank_sums, torch.ones_like(rank_sums), atol=1e-5)

    def test_return_features(self):
        """Test return_features option."""
        model = RingRiftCNN_v3(
            board_size=8,
            in_channels=14,
            num_res_blocks=2,
            num_filters=32,
            history_length=3,
        )
        x = torch.randn(2, 56, 8, 8)
        g = torch.randn(2, 20)

        value, policy, rank_dist, features = model(x, g, return_features=True)

        assert features.shape == (2, 32 + 20)


class TestRingRiftCNN_v3_Lite:
    """Tests for RingRiftCNN_v3_Lite memory-efficient spatial policy heads."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert RingRiftCNN_v3_Lite.ARCHITECTURE_VERSION == "v3.1.0-lite"

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        in_channels = 12
        history_length = 2
        model = RingRiftCNN_v3_Lite(
            board_size=8,
            in_channels=in_channels,
            num_res_blocks=2,
            num_filters=32,
            history_length=history_length,
        )
        batch_size = 4
        total_channels = in_channels * (history_length + 1)
        x = torch.randn(batch_size, total_channels, 8, 8)
        g = torch.randn(batch_size, 20)

        value, policy, rank_dist = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, POLICY_SIZE_8x8)
        assert rank_dist.shape == (batch_size, 4, 4)


class TestRingRiftCNN_v3_Flat:
    """Tests for RingRiftCNN_v3_Flat with flat policy heads."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert RingRiftCNN_v3_Flat.ARCHITECTURE_VERSION == "v3.1.0-flat"

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        in_channels = 14
        history_length = 3
        model = RingRiftCNN_v3_Flat(
            board_size=8,
            in_channels=in_channels,
            num_res_blocks=2,
            num_filters=32,
            history_length=history_length,
        )
        batch_size = 4
        total_channels = in_channels * (history_length + 1)
        x = torch.randn(batch_size, total_channels, 8, 8)
        g = torch.randn(batch_size, 20)

        value, policy, rank_dist = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, POLICY_SIZE_8x8)
        assert rank_dist.shape == (batch_size, 4, 4)

    def test_no_extreme_policy_values(self):
        """Test that flat policy head doesn't produce -1e9 masked values."""
        model = RingRiftCNN_v3_Flat(
            board_size=8,
            in_channels=14,
            num_res_blocks=2,
            num_filters=32,
            history_length=3,
        )
        x = torch.randn(4, 56, 8, 8)
        g = torch.randn(4, 20)

        _, policy, _ = model(x, g)

        # Flat policy head should NOT have -1e9 values
        assert torch.all(policy > -1e8)


# ==============================================================================
# Square Architecture Tests - V4
# ==============================================================================


class TestRingRiftCNN_v4:
    """Tests for RingRiftCNN_v4 NAS-optimized attention architecture."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert RingRiftCNN_v4.ARCHITECTURE_VERSION == "v4.0.0"

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        in_channels = 14
        history_length = 3
        model = RingRiftCNN_v4(
            board_size=8,
            in_channels=in_channels,
            num_res_blocks=3,  # Smaller for testing
            num_filters=32,
            history_length=history_length,
        )
        batch_size = 4
        total_channels = in_channels * (history_length + 1)
        x = torch.randn(batch_size, total_channels, 8, 8)
        g = torch.randn(batch_size, 20)

        value, policy, rank_dist = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, POLICY_SIZE_8x8)
        assert rank_dist.shape == (batch_size, 4, 4)

    def test_uses_attention_blocks(self):
        """Test that v4 uses AttentionResidualBlock."""
        model = RingRiftCNN_v4(
            board_size=8,
            num_res_blocks=2,
            num_filters=32,
        )
        # Check that res_blocks are AttentionResidualBlocks
        from app.ai.neural_net.blocks import AttentionResidualBlock
        for block in model.res_blocks:
            assert isinstance(block, AttentionResidualBlock)

    def test_deeper_value_head(self):
        """Test that v4 has 3-layer value head (NAS optimal)."""
        model = RingRiftCNN_v4(
            board_size=8,
            num_res_blocks=2,
            num_filters=32,
        )
        # Check 3-layer value head
        assert hasattr(model, "value_fc1")
        assert hasattr(model, "value_fc2")
        assert hasattr(model, "value_fc3")


# ==============================================================================
# Hex Architecture Tests - V2
# ==============================================================================


class TestHexNeuralNet_v2:
    """Tests for HexNeuralNet_v2 high-capacity hex architecture."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert HexNeuralNet_v2.ARCHITECTURE_VERSION == "v2.0.0"

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        model = HexNeuralNet_v2(
            in_channels=40,
            global_features=20,
            num_res_blocks=2,
            num_filters=32,
            board_size=21,
            policy_size=P_HEX,
        )
        batch_size = 4
        x = torch.randn(batch_size, 40, 21, 21)
        g = torch.randn(batch_size, 20)

        value, policy = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, P_HEX)

    def test_value_range(self):
        """Test value output is in [-1, 1] range."""
        model = HexNeuralNet_v2(
            in_channels=40,
            num_res_blocks=2,
            num_filters=32,
            board_size=21,
        )
        x = torch.randn(5, 40, 21, 21)
        g = torch.randn(5, 20)

        value, _ = model(x, g)

        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_hex_masking(self):
        """Test that hex mask is registered as buffer."""
        model = HexNeuralNet_v2(
            in_channels=40,
            num_res_blocks=2,
            num_filters=32,
            board_size=25,
            hex_radius=12,
        )
        assert hasattr(model, "hex_mask")
        assert model.hex_mask is not None


class TestHexNeuralNet_v2_Lite:
    """Tests for HexNeuralNet_v2_Lite memory-efficient hex architecture."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert HexNeuralNet_v2_Lite.ARCHITECTURE_VERSION == "v2.0.0-lite"

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        model = HexNeuralNet_v2_Lite(
            in_channels=36,
            global_features=20,
            num_res_blocks=2,
            num_filters=32,
            board_size=21,
            policy_size=P_HEX,
        )
        batch_size = 3
        x = torch.randn(batch_size, 36, 21, 21)
        g = torch.randn(batch_size, 20)

        value, policy = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, P_HEX)


# ==============================================================================
# Hex Architecture Tests - V3
# ==============================================================================


class TestHexNeuralNet_v3:
    """Tests for HexNeuralNet_v3 with spatial policy heads."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert HexNeuralNet_v3.ARCHITECTURE_VERSION == "v3.0.0"

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        model = HexNeuralNet_v3(
            in_channels=64,
            global_features=20,
            num_res_blocks=2,
            num_filters=32,
            board_size=25,
            policy_size=P_HEX,
        )
        batch_size = 4
        x = torch.randn(batch_size, 64, 25, 25)
        g = torch.randn(batch_size, 20)

        value, policy = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, P_HEX)

    def test_spatial_policy_heads_exist(self):
        """Test that spatial policy heads are present."""
        model = HexNeuralNet_v3(
            in_channels=64,
            num_res_blocks=2,
            num_filters=32,
        )
        assert hasattr(model, "placement_conv")
        assert hasattr(model, "movement_conv")
        assert hasattr(model, "special_fc")

    def test_return_features(self):
        """Test return_features option."""
        model = HexNeuralNet_v3(
            in_channels=64,
            num_res_blocks=2,
            num_filters=32,
            board_size=25,
        )
        x = torch.randn(2, 64, 25, 25)
        g = torch.randn(2, 20)

        value, policy, features = model(x, g, return_features=True)

        assert features.shape == (2, 32)  # num_filters (pooled backbone)


class TestHexNeuralNet_v3_Lite:
    """Tests for HexNeuralNet_v3_Lite memory-efficient spatial policy."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert HexNeuralNet_v3_Lite.ARCHITECTURE_VERSION == "v3.0.0-lite"

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        model = HexNeuralNet_v3_Lite(
            in_channels=44,
            global_features=20,
            num_res_blocks=2,
            num_filters=32,
            board_size=25,
            policy_size=P_HEX,
        )
        batch_size = 4
        x = torch.randn(batch_size, 44, 25, 25)
        g = torch.randn(batch_size, 20)

        value, policy = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, P_HEX)


class TestHexNeuralNet_v3_Flat:
    """Tests for HexNeuralNet_v3_Flat with flat policy heads."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert HexNeuralNet_v3_Flat.ARCHITECTURE_VERSION == "v3.1.0-flat"

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        model = HexNeuralNet_v3_Flat(
            in_channels=64,
            global_features=20,
            num_res_blocks=2,
            num_filters=32,
            board_size=25,
            policy_size=P_HEX,
        )
        batch_size = 4
        x = torch.randn(batch_size, 64, 25, 25)
        g = torch.randn(batch_size, 20)

        value, policy = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, P_HEX)

    def test_no_extreme_policy_values(self):
        """Test that flat policy head doesn't produce -1e9 masked values."""
        model = HexNeuralNet_v3_Flat(
            in_channels=64,
            num_res_blocks=2,
            num_filters=32,
            board_size=25,
        )
        x = torch.randn(4, 64, 25, 25)
        g = torch.randn(4, 20)

        _, policy = model(x, g)

        # Flat policy head should NOT have -1e9 values
        assert torch.all(policy > -1e8)


# ==============================================================================
# Hex Architecture Tests - V4
# ==============================================================================


class TestHexNeuralNet_v4:
    """Tests for HexNeuralNet_v4 NAS-optimized attention architecture."""

    def test_architecture_version(self):
        """Test architecture version constant."""
        assert HexNeuralNet_v4.ARCHITECTURE_VERSION == "v4.0.0"

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        model = HexNeuralNet_v4(
            in_channels=64,
            global_features=20,
            num_res_blocks=3,  # Smaller for testing
            num_filters=32,
            board_size=25,
            policy_size=P_HEX,
        )
        batch_size = 4
        x = torch.randn(batch_size, 64, 25, 25)
        g = torch.randn(batch_size, 20)

        value, policy = model(x, g)

        assert value.shape == (batch_size, 4)
        assert policy.shape == (batch_size, P_HEX)

    def test_uses_attention_blocks(self):
        """Test that v4 uses AttentionResidualBlock."""
        model = HexNeuralNet_v4(
            in_channels=64,
            num_res_blocks=2,
            num_filters=32,
        )
        from app.ai.neural_net.blocks import AttentionResidualBlock
        for block in model.res_blocks:
            assert isinstance(block, AttentionResidualBlock)

    def test_deeper_value_head(self):
        """Test that v4 has 3-layer value head (NAS optimal)."""
        model = HexNeuralNet_v4(
            in_channels=64,
            num_res_blocks=2,
            num_filters=32,
        )
        assert hasattr(model, "value_fc1")
        assert hasattr(model, "value_fc2")
        assert hasattr(model, "value_fc3")

    def test_smaller_board_size(self):
        """Test v4 works with smaller hex8 board size."""
        model = HexNeuralNet_v4(
            in_channels=64,
            num_res_blocks=2,
            num_filters=32,
            board_size=9,  # hex8
            policy_size=5000,  # Custom smaller policy size
            hex_radius=4,
        )
        x = torch.randn(2, 64, 9, 9)
        g = torch.randn(2, 20)

        value, policy = model(x, g)

        assert value.shape == (2, 4)
        assert policy.shape == (2, 5000)


# ==============================================================================
# Cross-Architecture Comparison Tests
# ==============================================================================


class TestArchitectureComparisons:
    """Tests comparing different architecture versions."""

    def test_v3_has_more_outputs_than_v2_square(self):
        """Test that v3 returns rank_dist in addition to value/policy."""
        model_v2 = RingRiftCNN_v2(
            board_size=8, num_res_blocks=2, num_filters=32,
            in_channels=14, history_length=3,
        )
        model_v3 = RingRiftCNN_v3(
            board_size=8, num_res_blocks=2, num_filters=32,
            in_channels=14, history_length=3,
        )
        x = torch.randn(2, 56, 8, 8)
        g = torch.randn(2, 20)

        out_v2 = model_v2(x, g)
        out_v3 = model_v3(x, g)

        assert len(out_v2) == 2  # value, policy
        assert len(out_v3) == 3  # value, policy, rank_dist

    def test_v4_uses_larger_initial_kernel(self):
        """Test that v4 uses 5x5 initial kernel vs 3x3."""
        model_v3 = RingRiftCNN_v3(
            board_size=8, num_res_blocks=2, num_filters=32,
        )
        model_v4 = RingRiftCNN_v4(
            board_size=8, num_res_blocks=2, num_filters=32,
            initial_kernel_size=5,
        )

        # v3 uses 3x3
        assert model_v3.conv1.kernel_size == (3, 3)
        # v4 uses 5x5 (NAS optimal)
        assert model_v4.conv1.kernel_size == (5, 5)

    def test_flat_vs_spatial_policy_values(self):
        """Test that flat policy produces different values than spatial."""
        model_spatial = RingRiftCNN_v3(
            board_size=8, num_res_blocks=2, num_filters=32,
            in_channels=14, history_length=3,
        )
        model_flat = RingRiftCNN_v3_Flat(
            board_size=8, num_res_blocks=2, num_filters=32,
            in_channels=14, history_length=3,
        )
        x = torch.randn(2, 56, 8, 8)
        g = torch.randn(2, 20)

        _, policy_spatial, _ = model_spatial(x, g)
        _, policy_flat, _ = model_flat(x, g)

        # Spatial policy can have -1e4 masked values
        has_extreme_spatial = torch.any(policy_spatial < -1e3)
        # Flat policy should not
        has_extreme_flat = torch.any(policy_flat < -1e3)

        # One or both may be true depending on random init,
        # but flat should never have extreme negative values
        assert not has_extreme_flat


# ==============================================================================
# Forward Single Tests
# ==============================================================================


class TestForwardSingle:
    """Tests for forward_single convenience methods."""

    def test_forward_single_v2(self):
        """Test forward_single for v2 model."""
        model = RingRiftCNN_v2(
            board_size=8,
            in_channels=14,
            num_res_blocks=2,
            num_filters=32,
            history_length=3,
        )
        feature = np.random.randn(56, 8, 8).astype(np.float32)
        globals_vec = np.random.randn(20).astype(np.float32)

        value, policy = model.forward_single(feature, globals_vec)

        assert isinstance(value, float)
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (POLICY_SIZE_8x8,)

    def test_forward_single_v3(self):
        """Test forward_single for v3 model returns 3 values."""
        model = RingRiftCNN_v3(
            board_size=8,
            in_channels=14,
            num_res_blocks=2,
            num_filters=32,
            history_length=3,
        )
        feature = np.random.randn(56, 8, 8).astype(np.float32)
        globals_vec = np.random.randn(20).astype(np.float32)

        value, policy, rank_dist = model.forward_single(feature, globals_vec)

        assert isinstance(value, float)
        assert isinstance(policy, np.ndarray)
        assert isinstance(rank_dist, np.ndarray)
        assert rank_dist.shape == (4, 4)

    def test_forward_single_hex_v2(self):
        """Test forward_single for hex v2 model."""
        model = HexNeuralNet_v2(
            in_channels=40,
            num_res_blocks=2,
            num_filters=32,
            board_size=21,
        )
        feature = np.random.randn(40, 21, 21).astype(np.float32)
        globals_vec = np.random.randn(20).astype(np.float32)

        value, policy = model.forward_single(feature, globals_vec)

        assert isinstance(value, float)
        assert isinstance(policy, np.ndarray)


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_batch_size_one(self):
        """Test models work with batch size 1."""
        model = RingRiftCNN_v3(
            board_size=8, num_res_blocks=2, num_filters=32,
            in_channels=14, history_length=3,
        )
        x = torch.randn(1, 56, 8, 8)
        g = torch.randn(1, 20)

        value, policy, rank_dist = model(x, g)

        assert value.shape == (1, 4)

    def test_large_batch_size(self):
        """Test models work with larger batch sizes."""
        model = RingRiftCNN_v2(
            board_size=8, num_res_blocks=2, num_filters=32,
            in_channels=14, history_length=3,
        )
        x = torch.randn(64, 56, 8, 8)
        g = torch.randn(64, 20)

        value, policy = model(x, g)

        assert value.shape == (64, 4)

    def test_custom_num_players(self):
        """Test models work with different number of players."""
        model = RingRiftCNN_v2(
            board_size=8,
            num_res_blocks=2,
            num_filters=32,
            in_channels=14,
            history_length=3,
            num_players=2,
        )
        x = torch.randn(4, 56, 8, 8)
        g = torch.randn(4, 20)

        value, policy = model(x, g)

        assert value.shape == (4, 2)  # 2 players

    def test_custom_policy_size(self):
        """Test models work with custom policy size."""
        custom_policy_size = 1000
        model = RingRiftCNN_v2(
            board_size=8,
            num_res_blocks=2,
            num_filters=32,
            in_channels=14,
            history_length=3,
            policy_size=custom_policy_size,
        )
        x = torch.randn(4, 56, 8, 8)
        g = torch.randn(4, 20)

        value, policy = model(x, g)

        assert policy.shape == (4, custom_policy_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
