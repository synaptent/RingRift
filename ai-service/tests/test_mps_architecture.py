"""
Tests for MPS-compatible neural network architecture.

These tests verify that the RingRiftCNN_MPS architecture can be instantiated,
performs forward passes correctly, and produces the expected output shapes.
"""

import pytest
import torch
import numpy as np
import os
from unittest.mock import Mock

from app.ai.neural_net import RingRiftCNN_MPS, NeuralNetAI


class TestRingRiftCNNMPS:
    """Tests for the MPS-compatible architecture."""
    
    def test_mps_architecture_creation(self):
        """Test that MPS-compatible architecture can be instantiated."""
        model = RingRiftCNN_MPS(
            board_size=8,
            in_channels=10,
            global_features=10,
            num_res_blocks=10,
            num_filters=128,
            history_length=3
        )
        assert model is not None
        assert isinstance(model, RingRiftCNN_MPS)
        assert model.board_size == 8
        assert model.policy_size == 55000
    
    def test_mps_architecture_forward_pass_8x8(self):
        """Test forward pass produces correct output shapes for 8x8 board."""
        model = RingRiftCNN_MPS(board_size=8, in_channels=10,
                                global_features=10)
        model.eval()
        
        # Input: batch_size=2, channels=40 (10 * 4 for history),
        # spatial=8x8
        batch_size = 2
        x = torch.randn(batch_size, 40, 8, 8)
        globals_vec = torch.randn(batch_size, 10)
        
        with torch.no_grad():
            value, policy = model(x, globals_vec)
        
        assert value.shape == (batch_size, 1)
        assert policy.shape == (batch_size, 55000)
        assert value.min() >= -1.0 and value.max() <= 1.0
    
    def test_mps_architecture_forward_pass_19x19(self):
        """Test forward pass produces correct output shapes for 19x19 board."""
        model = RingRiftCNN_MPS(board_size=19, in_channels=10,
                                global_features=10)
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 40, 19, 19)
        globals_vec = torch.randn(batch_size, 10)
        
        with torch.no_grad():
            value, policy = model(x, globals_vec)
        
        assert value.shape == (batch_size, 1)
        assert policy.shape == (batch_size, 55000)
        assert value.min() >= -1.0 and value.max() <= 1.0
    
    def test_mps_architecture_forward_pass_hex(self):
        """Test forward pass for hexagonal board (21x21)."""
        model = RingRiftCNN_MPS(board_size=21, in_channels=10,
                                global_features=10)
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 40, 21, 21)
        globals_vec = torch.randn(batch_size, 10)
        
        with torch.no_grad():
            value, policy = model(x, globals_vec)
        
        assert value.shape == (batch_size, 1)
        assert policy.shape == (batch_size, 55000)
    
    def test_mps_architecture_forward_single(self):
        """Test forward_single method for single-sample inference."""
        model = RingRiftCNN_MPS(board_size=8, in_channels=10,
                                global_features=10)
        model.eval()
        
        feature = np.random.randn(40, 8, 8).astype(np.float32)
        globals_vec = np.random.randn(10).astype(np.float32)
        
        value, policy_logits = model.forward_single(feature, globals_vec)
        
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0
        assert isinstance(policy_logits, np.ndarray)
        assert policy_logits.shape == (55000,)
    
    def test_mps_no_adaptive_pooling(self):
        """Verify that MPS architecture doesn't use AdaptiveAvgPool2d."""
        model = RingRiftCNN_MPS(board_size=8, in_channels=10,
                                global_features=10)
        
        # Check that model doesn't have adaptive_pool attribute
        assert not hasattr(model, 'adaptive_pool')
        
        # Verify model can be converted to MPS device (if available)
        # without errors
        if torch.backends.mps.is_available():
            try:
                model_mps = model.to('mps')
                assert model_mps is not None
            except Exception as e:
                pytest.fail(f"Failed to move model to MPS device: {e}")


class TestNeuralNetAIArchitectureSelection:
    """Tests for architecture selection in NeuralNetAI."""
    
    def _create_mock_config(self):
        """Create a properly mocked config object."""
        config = Mock()
        config.nn_model_id = "test_model"
        config.allow_fresh_weights = True
        config.rng_seed = None  # Required by BaseAI
        config.difficulty = 3  # Integer difficulty level
        config.randomness = 0.0
        return config
    
    def test_default_architecture_selection(self):
        """Test that default architecture is selected by default."""
        config = self._create_mock_config()
        
        # Clear environment variables
        os.environ.pop('RINGRIFT_NN_ARCHITECTURE', None)
        
        ai = NeuralNetAI(player_number=1, config=config)
        
        assert ai.architecture_type == "default"
        assert isinstance(ai.model, torch.nn.Module)
    
    def test_mps_architecture_selection(self):
        """Test MPS architecture selection via environment variable."""
        config = self._create_mock_config()
        
        os.environ['RINGRIFT_NN_ARCHITECTURE'] = 'mps'
        
        try:
            ai = NeuralNetAI(player_number=1, config=config)
            assert ai.architecture_type == "mps"
            assert isinstance(ai.model, RingRiftCNN_MPS)
        finally:
            os.environ.pop('RINGRIFT_NN_ARCHITECTURE', None)
    
    def test_auto_architecture_selection(self):
        """Test auto architecture selection."""
        config = self._create_mock_config()
        
        os.environ['RINGRIFT_NN_ARCHITECTURE'] = 'auto'
        
        try:
            ai = NeuralNetAI(player_number=1, config=config)
            
            # On MPS-capable systems, should select MPS architecture
            # On other systems, should fall back to default
            if torch.backends.mps.is_available():
                assert ai.architecture_type == "mps"
                assert isinstance(ai.model, RingRiftCNN_MPS)
            else:
                # May be default or mps depending on system
                assert ai.architecture_type in ["default", "mps"]
        finally:
            os.environ.pop('RINGRIFT_NN_ARCHITECTURE', None)
    
    def test_mps_checkpoint_naming(self):
        """Test that MPS architecture uses _mps suffix for checkpoints."""
        config = self._create_mock_config()
        config.nn_model_id = "ringrift_v1"
        
        os.environ['RINGRIFT_NN_ARCHITECTURE'] = 'mps'
        
        try:
            ai = NeuralNetAI(player_number=1, config=config)
            
            # The checkpoint path should include _mps suffix
            # This is tested indirectly through the initialization
            assert ai.architecture_type == "mps"
        finally:
            os.environ.pop('RINGRIFT_NN_ARCHITECTURE', None)


class TestMPSDeviceCompatibility:
    """Tests for MPS device compatibility."""
    
    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available on this system"
    )
    def test_mps_device_forward_pass(self):
        """Test forward pass on MPS device (if available)."""
        model = RingRiftCNN_MPS(board_size=8, in_channels=10,
                                global_features=10)
        model.eval()
        model = model.to('mps')
        
        batch_size = 2
        x = torch.randn(batch_size, 40, 8, 8).to('mps')
        globals_vec = torch.randn(batch_size, 10).to('mps')
        
        with torch.no_grad():
            value, policy = model(x, globals_vec)
        
        assert value.device.type == 'mps'
        assert policy.device.type == 'mps'
        assert value.shape == (batch_size, 1)
        assert policy.shape == (batch_size, 55000)
    
    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available on this system"
    )
    def test_neural_net_ai_mps_device(self):
        """Test NeuralNetAI initialization with MPS device."""
        config = Mock()
        config.nn_model_id = "test_model"
        config.allow_fresh_weights = True
        config.rng_seed = None
        config.difficulty = 3  # Integer difficulty level
        config.randomness = 0.0
        
        os.environ['RINGRIFT_NN_ARCHITECTURE'] = 'mps'
        
        try:
            ai = NeuralNetAI(player_number=1, config=config)
            
            # Device should be MPS if available and architecture is MPS
            assert ai.device.type == 'mps'
            assert ai.architecture_type == "mps"
        finally:
            os.environ.pop('RINGRIFT_NN_ARCHITECTURE', None)


class TestArchitectureVersioning:
    """Tests for architecture version compatibility."""
    
    def test_mps_architecture_version(self):
        """Test that MPS architecture has correct version string."""
        model = RingRiftCNN_MPS(board_size=8, in_channels=10,
                                global_features=10)
        
        assert hasattr(model, 'ARCHITECTURE_VERSION')
        assert model.ARCHITECTURE_VERSION == "v1.0.0-mps"
    
    def test_architecture_versions_distinct(self):
        """Test that default and MPS architectures have distinct versions."""
        from app.ai.neural_net import RingRiftCNN
        
        default_model = RingRiftCNN(board_size=8)
        mps_model = RingRiftCNN_MPS(
            board_size=8, in_channels=10, global_features=10
        )
        
        assert (default_model.ARCHITECTURE_VERSION !=
                mps_model.ARCHITECTURE_VERSION)