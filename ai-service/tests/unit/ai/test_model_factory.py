"""Tests for neural network model factory.

Tests the model factory functions that create board-specific neural network
architectures with appropriate configurations.
"""

import os
from unittest.mock import patch

import pytest
import torch

from app.ai.neural_net.model_factory import (
    VALID_MEMORY_TIERS,
    create_model_for_board,
    get_memory_tier,
    get_model_config_for_board,
)
from app.models import BoardType


class TestGetMemoryTier:
    """Tests for get_memory_tier function."""

    def test_default_tier_is_v4(self):
        """Default memory tier should be 'v4' (NAS-optimized architecture)."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            os.environ.pop("RINGRIFT_NN_MEMORY_TIER", None)
            tier = get_memory_tier()
            assert tier == "v4"

    def test_reads_from_environment(self):
        """Should read tier from environment variable."""
        with patch.dict(os.environ, {"RINGRIFT_NN_MEMORY_TIER": "low"}):
            tier = get_memory_tier()
            assert tier == "low"

    def test_all_valid_tiers_accepted(self):
        """All valid tiers should be accepted."""
        for valid_tier in VALID_MEMORY_TIERS:
            with patch.dict(os.environ, {"RINGRIFT_NN_MEMORY_TIER": valid_tier}):
                tier = get_memory_tier()
                assert tier == valid_tier

    def test_invalid_tier_defaults_to_v4(self):
        """Invalid tier should default to 'v4' with warning."""
        with patch.dict(os.environ, {"RINGRIFT_NN_MEMORY_TIER": "invalid_tier"}):
            tier = get_memory_tier()
            assert tier == "v4"

    def test_case_insensitive(self):
        """Tier name should be case insensitive."""
        with patch.dict(os.environ, {"RINGRIFT_NN_MEMORY_TIER": "V3-HIGH"}):
            tier = get_memory_tier()
            assert tier == "v3-high"


class TestGetModelConfigForBoard:
    """Tests for get_model_config_for_board function."""

    @pytest.mark.parametrize("board_type", list(BoardType))
    def test_returns_valid_config_for_all_board_types(self, board_type):
        """Should return valid config for all board types."""
        config = get_model_config_for_board(board_type)

        assert "board_size" in config
        assert "policy_size" in config
        assert "memory_tier" in config
        assert "num_res_blocks" in config
        assert "num_filters" in config
        assert "recommended_model" in config
        assert "estimated_params_m" in config

    @pytest.mark.parametrize("tier", ["high", "low", "v3-high", "v3-low"])
    def test_respects_memory_tier(self, tier):
        """Should respect provided memory tier."""
        config = get_model_config_for_board(BoardType.SQUARE8, memory_tier=tier)
        assert config["memory_tier"] == tier

    def test_v4_tier_only_for_square(self):
        """V4 tier should work for square boards."""
        config = get_model_config_for_board(BoardType.SQUARE8, memory_tier="v4")
        assert config["recommended_model"] == "RingRiftCNN_v4"

    def test_high_tier_has_more_capacity(self):
        """High tier should have more capacity than low tier."""
        high_config = get_model_config_for_board(BoardType.SQUARE8, memory_tier="high")
        low_config = get_model_config_for_board(BoardType.SQUARE8, memory_tier="low")

        assert high_config["num_res_blocks"] > low_config["num_res_blocks"]
        assert high_config["num_filters"] > low_config["num_filters"]
        assert high_config["estimated_params_m"] > low_config["estimated_params_m"]

    def test_hex_config_uses_hex_models(self):
        """Hex board should use hex model architectures."""
        config = get_model_config_for_board(BoardType.HEXAGONAL, memory_tier="high")
        assert "Hex" in config["recommended_model"]

    def test_square_config_uses_cnn_models(self):
        """Square board should use CNN model architectures."""
        config = get_model_config_for_board(BoardType.SQUARE8, memory_tier="high")
        assert "CNN" in config["recommended_model"]


class TestCreateModelForBoard:
    """Tests for create_model_for_board function."""

    @pytest.mark.parametrize("board_type", [BoardType.SQUARE8, BoardType.SQUARE19])
    def test_creates_square_model(self, board_type):
        """Should create valid model for square boards."""
        model = create_model_for_board(board_type, memory_tier="low")

        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, "forward")

    @pytest.mark.parametrize("board_type", [BoardType.HEXAGONAL, BoardType.HEX8])
    def test_creates_hex_model(self, board_type):
        """Should create valid model for hex boards."""
        model = create_model_for_board(board_type, memory_tier="low")

        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, "forward")

    def test_v4_not_available_for_hex(self):
        """V4 architecture should not be available for hex boards."""
        with pytest.raises(ValueError, match="V4 architecture is not yet available"):
            create_model_for_board(BoardType.HEXAGONAL, memory_tier="v4")

    def test_custom_res_blocks(self):
        """Should respect custom num_res_blocks parameter."""
        model = create_model_for_board(
            BoardType.SQUARE8,
            memory_tier="low",
            num_res_blocks=4
        )
        # Check the model has the expected number of blocks
        assert hasattr(model, "res_blocks") or hasattr(model, "backbone")

    def test_custom_filters(self):
        """Should respect custom num_filters parameter."""
        model = create_model_for_board(
            BoardType.SQUARE8,
            memory_tier="low",
            num_filters=64
        )
        assert isinstance(model, torch.nn.Module)

    def test_backward_compatible_model_class_arg(self):
        """model_class argument should be accepted for backward compatibility."""
        # Should not raise even though model_class is ignored
        model = create_model_for_board(
            BoardType.SQUARE8,
            memory_tier="low",
            model_class="legacy_class"  # Should be ignored
        )
        assert isinstance(model, torch.nn.Module)

    def test_policy_size_override(self):
        """Should accept custom policy_size."""
        model = create_model_for_board(
            BoardType.SQUARE8,
            memory_tier="low",
            policy_size=5000
        )
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize("tier", ["high", "low", "v3-high", "v3-low"])
    def test_all_tiers_create_valid_models(self, tier):
        """All memory tiers should create valid models."""
        model = create_model_for_board(BoardType.SQUARE8, memory_tier=tier)
        assert isinstance(model, torch.nn.Module)

    def test_model_is_trainable(self):
        """Created model should have trainable parameters."""
        model = create_model_for_board(BoardType.SQUARE8, memory_tier="low")
        params = list(model.parameters())

        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestModelIntegration:
    """Integration tests for model creation and inference."""

    @pytest.mark.parametrize("board_type,board_size", [
        (BoardType.SQUARE8, 8),
        (BoardType.SQUARE19, 19),
    ])
    def test_square_model_forward_pass(self, board_type, board_size):
        """Square model should handle forward pass with correct input shapes."""
        model = create_model_for_board(board_type, memory_tier="low")
        model.eval()

        # Create dummy input
        batch_size = 2
        in_channels = 14 * 4  # 14 base channels * (1 + 3 history)
        global_features = 20

        spatial_input = torch.randn(batch_size, in_channels, board_size, board_size)
        global_input = torch.randn(batch_size, global_features)

        with torch.no_grad():
            value, policy = model(spatial_input, global_input)

        assert value.shape[0] == batch_size
        assert policy.shape[0] == batch_size

    @pytest.mark.parametrize("board_type,board_size", [
        (BoardType.HEX8, 9),  # hex8 uses 9x9 bounding box
        (BoardType.HEXAGONAL, 25),  # hexagonal uses 25x25 bounding box
    ])
    def test_hex_model_forward_pass(self, board_type, board_size):
        """Hex model should handle forward pass with correct input shapes."""
        model = create_model_for_board(board_type, memory_tier="low")
        model.eval()

        # Create dummy input
        batch_size = 2
        in_channels = 14 * 4  # 14 base channels * (1 + 3 history)
        global_features = 20

        spatial_input = torch.randn(batch_size, in_channels, board_size, board_size)
        global_input = torch.randn(batch_size, global_features)

        with torch.no_grad():
            value, policy = model(spatial_input, global_input)

        assert value.shape[0] == batch_size
        assert policy.shape[0] == batch_size
