"""
Tests for V2 neural network architecture MPS compatibility.

These tests verify that V2 architectures (RingRiftCNN_v2, HexNeuralNet_v2, etc.)
are fully MPS-compatible and can run on Apple Silicon GPUs.

V2 architectures use torch.mean() for global pooling instead of AdaptiveAvgPool2d,
which ensures compatibility with PyTorch's MPS backend.
"""

import pytest
import torch
import os
from unittest.mock import Mock

from app.ai.neural_net import (
    RingRiftCNN_v2,
    RingRiftCNN_v2_Lite,
    HexNeuralNet_v2,
    HexNeuralNet_v2_Lite,
    create_model_for_board,
    get_memory_tier,
)
from app.models import BoardType


class TestV2ArchitectureMPSCompatibility:
    """Tests for V2 architecture MPS compatibility."""

    def test_v2_architecture_creation(self):
        """Test that V2 architectures can be instantiated."""
        model = RingRiftCNN_v2(
            board_size=8,
            in_channels=14,
            global_features=20,
            num_res_blocks=2,
            num_filters=32,
            history_length=4
        )
        assert model is not None
        assert isinstance(model, RingRiftCNN_v2)

    def test_v2_lite_architecture_creation(self):
        """Test that V2 Lite architectures can be instantiated."""
        model = RingRiftCNN_v2_Lite(
            board_size=8,
            in_channels=12,
            global_features=20,
            num_res_blocks=2,
            num_filters=32,
            history_length=3
        )
        assert model is not None
        assert isinstance(model, RingRiftCNN_v2_Lite)

    def test_v2_no_adaptive_pooling(self):
        """Verify that V2 architecture doesn't use AdaptiveAvgPool2d."""
        model = RingRiftCNN_v2(
            board_size=8, in_channels=14, global_features=20,
            num_res_blocks=2, num_filters=32
        )

        # V2 uses global average pooling via torch.mean, not AdaptiveAvgPool2d
        assert not hasattr(model, 'adaptive_pool')

        # Verify model can be converted to MPS device (if available)
        if torch.backends.mps.is_available():
            try:
                model_mps = model.to('mps')
                assert model_mps is not None
            except Exception as e:
                pytest.fail(f"Failed to move V2 model to MPS device: {e}")

    def test_hex_v2_no_adaptive_pooling(self):
        """Verify that Hex V2 architecture doesn't use AdaptiveAvgPool2d."""
        model = HexNeuralNet_v2(
            in_channels=14, global_features=20,
            num_res_blocks=2, num_filters=32,
            board_size=21, policy_size=80000
        )

        assert not hasattr(model, 'adaptive_pool')

        if torch.backends.mps.is_available():
            try:
                model_mps = model.to('mps')
                assert model_mps is not None
            except Exception as e:
                pytest.fail(f"Failed to move Hex V2 model to MPS device: {e}")


class TestV2DeviceCompatibility:
    """Tests for V2 model device compatibility (MPS/CUDA/CPU)."""

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available on this system"
    )
    def test_v2_mps_forward_pass(self):
        """Test V2 forward pass on MPS device (if available)."""
        model = RingRiftCNN_v2(
            board_size=8, in_channels=14, global_features=20,
            num_res_blocks=2, num_filters=32, history_length=4
        )
        model.eval()
        model = model.to('mps')

        batch_size = 2
        total_channels = 14 * (4 + 1)  # in_channels * (history_length + 1)
        x = torch.randn(batch_size, total_channels, 8, 8).to('mps')
        globals_vec = torch.randn(batch_size, 20).to('mps')

        with torch.no_grad():
            value, policy = model(x, globals_vec)

        assert value.device.type == 'mps'
        assert policy.device.type == 'mps'
        assert value.shape == (batch_size, 4)  # Multi-player value head

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available on this system"
    )
    def test_hex_v2_mps_forward_pass(self):
        """Test Hex V2 forward pass on MPS device (if available)."""
        model = HexNeuralNet_v2(
            in_channels=14, global_features=20,
            num_res_blocks=2, num_filters=32,
            board_size=21, policy_size=80000
        )
        model.eval()
        model = model.to('mps')

        batch_size = 2
        x = torch.randn(batch_size, 14, 21, 21).to('mps')
        globals_vec = torch.randn(batch_size, 20).to('mps')

        with torch.no_grad():
            value, policy = model(x, globals_vec)

        assert value.device.type == 'mps'
        assert policy.device.type == 'mps'
        assert value.shape == (batch_size, 4)

    def test_cpu_forward_pass(self):
        """Test V2 forward pass on CPU."""
        model = RingRiftCNN_v2(
            board_size=8, in_channels=14, global_features=20,
            num_res_blocks=2, num_filters=32, history_length=4
        )
        model.eval()

        batch_size = 2
        total_channels = 14 * (4 + 1)
        x = torch.randn(batch_size, total_channels, 8, 8)
        globals_vec = torch.randn(batch_size, 20)

        with torch.no_grad():
            value, policy = model(x, globals_vec)

        assert value.device.type == 'cpu'
        assert value.shape == (batch_size, 4)


class TestMemoryTierSelection:
    """Tests for memory tier configuration."""

    def test_get_memory_tier_default(self):
        """Test that default memory tier is 'high'."""
        os.environ.pop('RINGRIFT_NN_MEMORY_TIER', None)
        tier = get_memory_tier()
        assert tier == "high"

    def test_get_memory_tier_high(self):
        """Test explicit high memory tier selection."""
        os.environ['RINGRIFT_NN_MEMORY_TIER'] = 'high'
        try:
            tier = get_memory_tier()
            assert tier == "high"
        finally:
            os.environ.pop('RINGRIFT_NN_MEMORY_TIER', None)

    def test_get_memory_tier_low(self):
        """Test explicit low memory tier selection."""
        os.environ['RINGRIFT_NN_MEMORY_TIER'] = 'low'
        try:
            tier = get_memory_tier()
            assert tier == "low"
        finally:
            os.environ.pop('RINGRIFT_NN_MEMORY_TIER', None)

    def test_invalid_tier_defaults_to_high(self):
        """Test that invalid tier values default to 'high'."""
        os.environ['RINGRIFT_NN_MEMORY_TIER'] = 'invalid'
        try:
            tier = get_memory_tier()
            assert tier == "high"
        finally:
            os.environ.pop('RINGRIFT_NN_MEMORY_TIER', None)


class TestCreateModelForBoard:
    """Tests for the create_model_for_board factory function."""

    def test_create_square8_model_high_tier(self):
        """Test creating a square8 model with high memory tier."""
        os.environ['RINGRIFT_NN_MEMORY_TIER'] = 'high'
        try:
            model = create_model_for_board(BoardType.SQUARE8)
            assert isinstance(model, RingRiftCNN_v2)
        finally:
            os.environ.pop('RINGRIFT_NN_MEMORY_TIER', None)

    def test_create_square8_model_low_tier(self):
        """Test creating a square8 model with low memory tier."""
        os.environ['RINGRIFT_NN_MEMORY_TIER'] = 'low'
        try:
            model = create_model_for_board(BoardType.SQUARE8)
            assert isinstance(model, RingRiftCNN_v2_Lite)
        finally:
            os.environ.pop('RINGRIFT_NN_MEMORY_TIER', None)

    def test_create_hexagonal_model_high_tier(self):
        """Test creating a hexagonal model with high memory tier."""
        os.environ['RINGRIFT_NN_MEMORY_TIER'] = 'high'
        try:
            model = create_model_for_board(BoardType.HEXAGONAL)
            assert isinstance(model, HexNeuralNet_v2)
        finally:
            os.environ.pop('RINGRIFT_NN_MEMORY_TIER', None)

    def test_create_hexagonal_model_low_tier(self):
        """Test creating a hexagonal model with low memory tier."""
        os.environ['RINGRIFT_NN_MEMORY_TIER'] = 'low'
        try:
            model = create_model_for_board(BoardType.HEXAGONAL)
            assert isinstance(model, HexNeuralNet_v2_Lite)
        finally:
            os.environ.pop('RINGRIFT_NN_MEMORY_TIER', None)


class TestArchitectureVersioning:
    """Tests for V2 architecture version strings."""

    def test_v2_architecture_version(self):
        """Test that V2 architectures have correct version strings."""
        model = RingRiftCNN_v2(
            board_size=8, in_channels=14, global_features=20,
            num_res_blocks=2, num_filters=32
        )
        assert hasattr(model, 'ARCHITECTURE_VERSION')
        assert model.ARCHITECTURE_VERSION == "v2.0.0"

    def test_v2_lite_architecture_version(self):
        """Test that V2 Lite architectures have correct version strings."""
        model = RingRiftCNN_v2_Lite(
            board_size=8, in_channels=12, global_features=20,
            num_res_blocks=2, num_filters=32
        )
        assert hasattr(model, 'ARCHITECTURE_VERSION')
        assert model.ARCHITECTURE_VERSION == "v2.0.0-lite"
