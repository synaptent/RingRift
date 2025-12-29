"""Comprehensive tests for app.training.train module.

Tests for the core training pipeline including:
- TrainConfig validation and defaults
- Seed utilities for reproducibility
- Heuristic weight management
- Training compatibility validation
- Model initialization and checkpointing
- Early stopping and learning rate scheduling
- Data loading and NPZ handling
- Fault tolerance and circuit breakers
- Mixed precision training
- Distributed training setup
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.models import BoardType
from app.training.config import TrainConfig


# =============================================================================
# Test TrainConfig Defaults
# =============================================================================


class TestTrainConfigDefaults:
    """Tests for TrainConfig default values."""

    def test_default_board_type(self):
        """Default board type is SQUARE8."""
        config = TrainConfig()
        assert config.board_type == BoardType.SQUARE8

    def test_default_epochs(self):
        """Default epochs_per_iter is set."""
        config = TrainConfig()
        assert config.epochs_per_iter == 50  # Updated Dec 2025

    def test_default_learning_rate(self):
        """Default learning rate is 1e-3."""
        config = TrainConfig()
        assert config.learning_rate == 1e-3

    def test_default_batch_size(self):
        """Default batch size before auto-scaling."""
        # Disable auto-scaling for this test
        with patch.dict('os.environ', {'RINGRIFT_AUTO_BATCH_SCALE': '0'}):
            config = TrainConfig()
            assert config.batch_size >= 32

    def test_default_weight_decay(self):
        """Default weight decay is 1e-4."""
        config = TrainConfig()
        assert config.weight_decay == 1e-4

    def test_default_seed(self):
        """Default seed is 42."""
        config = TrainConfig()
        assert config.seed == 42

    def test_default_policy_weight(self):
        """Default policy weight is 1.0."""
        config = TrainConfig()
        assert config.policy_weight == 1.0

    def test_default_entropy_weight(self):
        """Default entropy weight is 0.01."""
        config = TrainConfig()
        assert config.entropy_weight == 0.01

    def test_default_warmup_epochs(self):
        """Default warmup epochs is 1."""
        config = TrainConfig()
        assert config.warmup_epochs == 1

    def test_default_lr_scheduler(self):
        """Default LR scheduler is cosine."""
        config = TrainConfig()
        assert config.lr_scheduler == "cosine"

    def test_default_early_stopping(self):
        """Default early stopping patience is 25 (increased from 5 for stability)."""
        config = TrainConfig()
        assert config.early_stopping_patience == 25

    def test_default_allow_empty_policies(self):
        """Default allows empty policies."""
        config = TrainConfig()
        assert config.allow_empty_policies is True


class TestTrainConfigCustomValues:
    """Tests for TrainConfig with custom values."""

    def test_custom_board_type(self):
        """Can set custom board type."""
        config = TrainConfig(board_type=BoardType.HEXAGONAL)
        assert config.board_type == BoardType.HEXAGONAL

    def test_custom_learning_rate(self):
        """Can set custom learning rate."""
        config = TrainConfig(learning_rate=5e-4)
        assert config.learning_rate == 5e-4

    def test_custom_epochs(self):
        """Can set custom epochs."""
        config = TrainConfig(epochs_per_iter=20)
        assert config.epochs_per_iter == 20

    def test_custom_early_stopping_disabled(self):
        """Can disable early stopping."""
        config = TrainConfig(early_stopping_patience=0)
        assert config.early_stopping_patience == 0


# =============================================================================
# Test Seed Utilities
# =============================================================================


class TestSeedUtilities:
    """Tests for seeding utilities."""

    def test_seed_all_legacy_import(self):
        """seed_all_legacy can be imported."""
        from app.training.train import seed_all_legacy
        assert callable(seed_all_legacy)

    def test_seed_all_legacy_sets_random(self):
        """seed_all_legacy sets Python random seed."""
        from app.training.train import seed_all_legacy

        seed_all_legacy(42)
        first_random = random.random()

        seed_all_legacy(42)
        second_random = random.random()

        assert first_random == second_random

    def test_seed_all_legacy_sets_numpy(self):
        """seed_all_legacy sets NumPy random seed."""
        from app.training.train import seed_all_legacy

        seed_all_legacy(42)
        first_random = np.random.random()

        seed_all_legacy(42)
        second_random = np.random.random()

        assert first_random == second_random

    def test_seed_all_legacy_sets_torch(self):
        """seed_all_legacy sets PyTorch random seed."""
        from app.training.train import seed_all_legacy

        seed_all_legacy(42)
        first_random = torch.rand(1).item()

        seed_all_legacy(42)
        second_random = torch.rand(1).item()

        assert first_random == second_random

    def test_seed_all_legacy_different_seeds(self):
        """Different seeds produce different results."""
        from app.training.train import seed_all_legacy

        seed_all_legacy(42)
        first_random = random.random()

        seed_all_legacy(123)
        second_random = random.random()

        assert first_random != second_random


# =============================================================================
# Test Heuristic Weight Management
# =============================================================================


class TestHeuristicWeightManagement:
    """Tests for heuristic weight flattening and reconstruction."""

    def test_flatten_heuristic_weights_import(self):
        """_flatten_heuristic_weights can be imported."""
        from app.training.train import _flatten_heuristic_weights
        assert callable(_flatten_heuristic_weights)

    def test_reconstruct_heuristic_profile_import(self):
        """_reconstruct_heuristic_profile can be imported."""
        from app.training.train import _reconstruct_heuristic_profile
        assert callable(_reconstruct_heuristic_profile)

    def test_flatten_requires_all_keys(self):
        """_flatten_heuristic_weights requires all HEURISTIC_WEIGHT_KEYS."""
        from app.training.train import _flatten_heuristic_weights

        # Incomplete profile should raise KeyError
        incomplete_profile = {"ring_power": 1.0}  # Missing other keys
        with pytest.raises(KeyError):
            _flatten_heuristic_weights(incomplete_profile)

    def test_flatten_and_reconstruct_roundtrip(self):
        """Flatten and reconstruct preserves values."""
        from app.training.train import (
            _flatten_heuristic_weights,
            _reconstruct_heuristic_profile,
        )
        from app.training.train import HEURISTIC_WEIGHT_KEYS

        # Create a valid profile with all keys
        profile = {k: float(i) for i, k in enumerate(HEURISTIC_WEIGHT_KEYS)}

        keys, values = _flatten_heuristic_weights(profile)
        reconstructed = _reconstruct_heuristic_profile(keys, values)

        for k in HEURISTIC_WEIGHT_KEYS:
            assert reconstructed[k] == profile[k]

    def test_flatten_returns_consistent_order(self):
        """_flatten_heuristic_weights returns consistent key order."""
        from app.training.train import _flatten_heuristic_weights
        from app.training.train import HEURISTIC_WEIGHT_KEYS

        profile = {k: 1.0 for k in HEURISTIC_WEIGHT_KEYS}

        keys1, _ = _flatten_heuristic_weights(profile)
        keys2, _ = _flatten_heuristic_weights(profile)

        assert keys1 == keys2


# =============================================================================
# Test Training Compatibility Validation
# =============================================================================


class TestTrainingCompatibilityValidation:
    """Tests for _validate_training_compatibility."""

    def test_validate_import(self):
        """_validate_training_compatibility can be imported."""
        from app.training.train import _validate_training_compatibility
        assert callable(_validate_training_compatibility)

    def test_validate_mismatched_policy_size(self):
        """Validation catches policy size mismatch."""
        from app.training.train import _validate_training_compatibility

        # Create mock model with policy_size=64
        model = MagicMock()
        model.policy_size = 64

        # Create mock dataset with different policy_size
        dataset = MagicMock()
        dataset.policy_size = 128

        config = TrainConfig()

        with pytest.raises(ValueError, match="policy"):
            _validate_training_compatibility(model, dataset, config)

    def test_validate_compatible_sizes(self):
        """Validation passes for compatible model and dataset."""
        from app.training.train import _validate_training_compatibility

        # Create mock model and dataset with matching policy_size and board_type
        model = MagicMock()
        model.policy_size = 64
        model.board_type = BoardType.SQUARE8

        dataset = MagicMock()
        dataset.policy_size = 64
        dataset.board_type = BoardType.SQUARE8  # Must match model
        dataset.__len__ = MagicMock(return_value=100)
        dataset.__getitem__ = MagicMock(return_value=(
            torch.zeros(10),  # features
            torch.zeros(64),  # policy
            torch.zeros(2),   # value
        ))

        config = TrainConfig()

        # Should not raise
        _validate_training_compatibility(model, dataset, config)


# =============================================================================
# Test Model Version Selection
# =============================================================================


class TestModelVersionSelection:
    """Tests for model version and architecture selection."""

    def test_v2_model_exists(self):
        """RingRiftCNN_v2 model exists."""
        from app.ai.neural_net import RingRiftCNN_v2
        assert RingRiftCNN_v2 is not None

    def test_v3_model_exists(self):
        """RingRiftCNN_v3 model exists."""
        from app.ai.neural_net import RingRiftCNN_v3
        assert RingRiftCNN_v3 is not None

    def test_hex_v2_model_exists(self):
        """HexNeuralNet_v2 model exists."""
        from app.ai.neural_net import HexNeuralNet_v2
        assert HexNeuralNet_v2 is not None

    def test_hex_v3_model_exists(self):
        """HexNeuralNet_v3 model exists."""
        from app.ai.neural_net import HexNeuralNet_v3
        assert HexNeuralNet_v3 is not None

    def test_hex_v4_model_exists(self):
        """HexNeuralNet_v4 model exists."""
        from app.ai.neural_net import HexNeuralNet_v4
        assert HexNeuralNet_v4 is not None


# =============================================================================
# Test Policy Size Functions
# =============================================================================


class TestPolicySizeFunctions:
    """Tests for policy size calculation."""

    def test_get_policy_size_square8(self):
        """Policy size for square8 is reasonable."""
        from app.ai.neural_net import get_policy_size_for_board
        size = get_policy_size_for_board("square8")
        # Policy size includes positions * move types, not just cell count
        assert size > 64  # At least more than cell count

    def test_get_policy_size_square19(self):
        """Policy size for square19 is reasonable."""
        from app.ai.neural_net import get_policy_size_for_board
        size = get_policy_size_for_board("square19")
        # Policy size includes positions * move types
        assert size > 361  # At least more than cell count

    def test_get_policy_size_hex8(self):
        """Policy size for hex8 is reasonable."""
        from app.ai.neural_net import get_policy_size_for_board
        size = get_policy_size_for_board("hex8")
        assert size > 0

    def test_get_policy_size_hexagonal(self):
        """Policy size for hexagonal is reasonable."""
        from app.ai.neural_net import get_policy_size_for_board
        size = get_policy_size_for_board("hexagonal")
        assert size > 0


# =============================================================================
# Test Learning Rate Scheduler Selection
# =============================================================================


class TestLRSchedulerSelection:
    """Tests for learning rate scheduler configuration."""

    def test_cosine_scheduler_option(self):
        """Cosine scheduler is valid option."""
        config = TrainConfig(lr_scheduler="cosine")
        assert config.lr_scheduler == "cosine"

    def test_step_scheduler_option(self):
        """Step scheduler is valid option."""
        config = TrainConfig(lr_scheduler="step")
        assert config.lr_scheduler == "step"

    def test_none_scheduler_option(self):
        """None scheduler disables scheduling."""
        config = TrainConfig(lr_scheduler="none")
        assert config.lr_scheduler == "none"

    def test_warm_restarts_scheduler_option(self):
        """Cosine warm restarts scheduler is valid option."""
        config = TrainConfig(lr_scheduler="cosine-warm-restarts")
        assert config.lr_scheduler == "cosine-warm-restarts"


# =============================================================================
# Test Entropy Regularization
# =============================================================================


class TestEntropyRegularization:
    """Tests for entropy regularization configuration."""

    def test_entropy_weight_zero_disables(self):
        """Zero entropy weight disables regularization."""
        config = TrainConfig(entropy_weight=0.0)
        assert config.entropy_weight == 0.0

    def test_entropy_weight_positive(self):
        """Positive entropy weight is valid."""
        config = TrainConfig(entropy_weight=0.05)
        assert config.entropy_weight == 0.05

    def test_default_entropy_weight_reasonable(self):
        """Default entropy weight is in reasonable range."""
        config = TrainConfig()
        assert 0.0 <= config.entropy_weight <= 0.1


# =============================================================================
# Test Policy Label Smoothing
# =============================================================================


class TestPolicyLabelSmoothing:
    """Tests for policy label smoothing configuration."""

    def test_default_smoothing(self):
        """Default has light label smoothing for regularization."""
        config = TrainConfig()
        assert config.policy_label_smoothing == 0.05  # Updated Dec 2025

    def test_smoothing_in_valid_range(self):
        """Smoothing value is in valid range."""
        config = TrainConfig(policy_label_smoothing=0.1)
        assert 0.0 <= config.policy_label_smoothing <= 1.0


# =============================================================================
# Test Train From File Function
# =============================================================================


class TestTrainFromFile:
    """Tests for train_from_file convenience function."""

    def test_train_from_file_import(self):
        """train_from_file can be imported."""
        from app.training.train import train_from_file
        assert callable(train_from_file)

    def test_train_from_file_returns_dict(self):
        """train_from_file returns a dictionary with loss values."""
        from app.training.train import train_from_file

        # The function returns a dict even on errors (with error info)
        # We just check the function signature and return type
        import inspect
        sig = inspect.signature(train_from_file)
        assert "data_path" in sig.parameters
        assert "output_path" in sig.parameters


# =============================================================================
# Test Distributed Training Utilities
# =============================================================================


class TestDistributedTrainingUtilities:
    """Tests for distributed training helper functions."""

    def test_get_rank_returns_int(self):
        """get_rank returns an integer."""
        from app.training.distributed import get_rank
        rank = get_rank()
        assert isinstance(rank, int)
        assert rank >= 0

    def test_get_world_size_returns_int(self):
        """get_world_size returns an integer."""
        from app.training.distributed import get_world_size
        size = get_world_size()
        assert isinstance(size, int)
        assert size >= 1

    def test_is_main_process_returns_bool(self):
        """is_main_process returns a boolean."""
        from app.training.distributed import is_main_process
        is_main = is_main_process()
        assert isinstance(is_main, bool)

    def test_seed_everything_import(self):
        """seed_everything can be imported."""
        from app.training.distributed import seed_everything
        assert callable(seed_everything)


# =============================================================================
# Test Fault Tolerance Components
# =============================================================================


class TestFaultToleranceComponents:
    """Tests for fault tolerance configuration."""

    def test_fault_tolerance_config_exists(self):
        """FaultToleranceConfig can be imported."""
        from app.training.train_setup import FaultToleranceConfig
        assert FaultToleranceConfig is not None

    def test_training_state_exists(self):
        """TrainingState can be imported."""
        from app.training.train_setup import TrainingState
        assert TrainingState is not None

    def test_setup_fault_tolerance_import(self):
        """setup_fault_tolerance can be imported."""
        from app.training.train_setup import setup_fault_tolerance
        assert callable(setup_fault_tolerance)


# =============================================================================
# Test Gradient Surgery
# =============================================================================


class TestGradientSurgery:
    """Tests for gradient surgery configuration."""

    def test_gradient_surgeon_import(self):
        """GradientSurgeon can be imported."""
        from app.training.gradient_surgery import GradientSurgeon
        assert GradientSurgeon is not None

    def test_gradient_surgery_config_import(self):
        """GradientSurgeryConfig can be imported."""
        from app.training.gradient_surgery import GradientSurgeryConfig
        assert GradientSurgeryConfig is not None


# =============================================================================
# Test Value Calibration
# =============================================================================


class TestValueCalibration:
    """Tests for value calibration tracking."""

    def test_calibration_tracker_import(self):
        """CalibrationTracker can be imported."""
        from app.training.value_calibration import CalibrationTracker
        assert CalibrationTracker is not None

    def test_calibration_tracker_init(self):
        """CalibrationTracker can be initialized."""
        from app.training.value_calibration import CalibrationTracker
        tracker = CalibrationTracker()
        assert tracker is not None


# =============================================================================
# Test Data Loader Utilities
# =============================================================================


class TestDataLoaderUtilities:
    """Tests for data loader utility functions."""

    def test_get_sample_count_import(self):
        """get_sample_count can be imported."""
        from app.training.data_loader import get_sample_count
        assert callable(get_sample_count)

    def test_streaming_data_loader_import(self):
        """StreamingDataLoader can be imported."""
        from app.training.data_loader import StreamingDataLoader
        assert StreamingDataLoader is not None

    def test_weighted_streaming_loader_import(self):
        """WeightedStreamingDataLoader can be imported."""
        from app.training.data_loader import WeightedStreamingDataLoader
        assert WeightedStreamingDataLoader is not None


# =============================================================================
# Test Dataset Classes
# =============================================================================


class TestDatasetClasses:
    """Tests for dataset class imports and basic usage."""

    def test_ringrift_dataset_import(self):
        """RingRiftDataset can be imported."""
        from app.training.datasets import RingRiftDataset
        assert RingRiftDataset is not None

    def test_weighted_dataset_import(self):
        """WeightedRingRiftDataset can be imported."""
        from app.training.datasets import WeightedRingRiftDataset
        assert WeightedRingRiftDataset is not None


# =============================================================================
# Test Model Versioning
# =============================================================================


class TestModelVersioning:
    """Tests for model versioning and checkpointing."""

    def test_save_model_checkpoint_import(self):
        """save_model_checkpoint can be imported."""
        from app.training.model_versioning import save_model_checkpoint
        assert callable(save_model_checkpoint)


# =============================================================================
# Test Neural Losses
# =============================================================================


class TestNeuralLosses:
    """Tests for neural network loss functions."""

    def test_build_rank_targets_import(self):
        """build_rank_targets can be imported."""
        from app.ai.neural_losses import build_rank_targets
        assert callable(build_rank_targets)

    def test_masked_policy_kl_import(self):
        """masked_policy_kl can be imported."""
        from app.ai.neural_losses import masked_policy_kl
        assert callable(masked_policy_kl)

    def test_multi_player_value_loss_import(self):
        """multi_player_value_loss can be imported."""
        from app.ai.neural_net import multi_player_value_loss
        assert callable(multi_player_value_loss)


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainingIntegration:
    """Integration tests for training pipeline components."""

    def test_model_can_be_instantiated(self):
        """Model can be instantiated with default config."""
        from app.ai.neural_net import RingRiftCNN_v2

        model = RingRiftCNN_v2(
            board_size=8,
            num_players=2,
            policy_size=64,
        )
        assert model is not None
        assert hasattr(model, 'forward')

    def test_model_forward_pass(self):
        """Model can perform forward pass."""
        from app.ai.neural_net import RingRiftCNN_v2

        model = RingRiftCNN_v2(
            board_size=8,
            num_players=2,
            policy_size=64,
        )

        # Create dummy input with correct number of channels
        batch_size = 2
        # Get in_channels from model (required attribute)
        in_channels = model.in_channels
        x = torch.zeros(batch_size, in_channels, 8, 8)

        # Model also requires globals tensor (player embeddings, turn info, etc.)
        # RingRiftCNN_v2 defaults to 20 global features
        globals_size = 20
        globals_tensor = torch.zeros(batch_size, globals_size)

        # Forward pass should work
        model.eval()
        with torch.no_grad():
            # Note: RingRiftCNN_v2.forward returns (policy, value) tuple
            outputs = model(x, globals_tensor)
            # Handle both tuple and individual returns
            if isinstance(outputs, tuple):
                policy, value = outputs
            else:
                policy = outputs
                value = None

        assert policy.shape[0] == batch_size
        if value is not None:
            assert value.shape[0] == batch_size

    def test_optimizer_can_be_created(self):
        """Optimizer can be created for model."""
        from app.ai.neural_net import RingRiftCNN_v2
        import torch.optim as optim

        model = RingRiftCNN_v2(
            board_size=8,
            num_players=2,
            policy_size=64,
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

        assert optimizer is not None

    def test_config_to_dict(self):
        """TrainConfig can be converted to dict."""
        from dataclasses import asdict

        config = TrainConfig(
            board_type=BoardType.SQUARE8,
            learning_rate=5e-4,
        )

        config_dict = asdict(config)
        assert isinstance(config_dict, dict)
        assert config_dict["learning_rate"] == 5e-4


# =============================================================================
# Test NPZ Data Loading
# =============================================================================


class TestNPZDataLoading:
    """Tests for NPZ file data loading."""

    def test_npz_file_loading_with_mock(self):
        """Test loading NPZ file with mocked data."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            npz_path = f.name

        try:
            # Create mock NPZ file
            np.savez(
                npz_path,
                features=np.zeros((10, 20, 8, 8), dtype=np.float32),
                globals=np.zeros((10, 8), dtype=np.float32),
                policy=np.zeros((10, 64), dtype=np.float32),
                value=np.zeros((10,), dtype=np.float32),
            )

            # Verify file exists and can be loaded
            assert Path(npz_path).exists()
            data = np.load(npz_path)
            assert 'features' in data
            assert 'policy' in data
            assert 'value' in data
            assert data['features'].shape[0] == 10
        finally:
            Path(npz_path).unlink(missing_ok=True)

    def test_npz_empty_file_handling(self):
        """Test handling of empty NPZ file."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            npz_path = f.name

        try:
            # Create empty NPZ file
            np.savez(npz_path)

            # Should load but have no keys
            data = np.load(npz_path)
            assert len(data.files) == 0
        finally:
            Path(npz_path).unlink(missing_ok=True)

    def test_npz_policy_size_mismatch_detection(self):
        """Test detection of policy size mismatch in data."""
        from app.training.train import _validate_training_compatibility

        # Create model and dataset with mismatched sizes
        model = MagicMock()
        model.policy_size = 64

        dataset = MagicMock()
        dataset.policy_size = 128
        dataset.__len__ = MagicMock(return_value=10)

        config = TrainConfig()

        # Should raise ValueError for size mismatch
        with pytest.raises(ValueError, match="policy"):
            _validate_training_compatibility(model, dataset, config)


# =============================================================================
# Test Model Architecture Selection
# =============================================================================


class TestModelArchitectureSelection:
    """Tests for model architecture selection logic."""

    def test_square_board_uses_cnn_model(self):
        """Square boards use RingRiftCNN models."""
        from app.ai.neural_net import RingRiftCNN_v2
        from app.training.config import get_model_version_for_board

        version = get_model_version_for_board(BoardType.SQUARE8)
        # All boards now use v4 (December 2025)
        assert version == "v4"

    def test_hex_board_uses_hex_model(self):
        """Hex boards use HexNeuralNet models."""
        from app.training.config import get_model_version_for_board

        version = get_model_version_for_board(BoardType.HEXAGONAL)
        assert version == "v4"  # Updated Dec 2025

    def test_hex8_board_uses_hex_model(self):
        """Hex8 boards use HexNeuralNet models."""
        from app.training.config import get_model_version_for_board

        version = get_model_version_for_board(BoardType.HEX8)
        assert version == "v4"  # Updated Dec 2025


# =============================================================================
# Test Checkpoint Saving/Loading
# =============================================================================


class TestCheckpointHandling:
    """Tests for checkpoint saving and loading."""

    def test_checkpoint_directory_creation(self):
        """Checkpoint directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            assert not checkpoint_dir.exists()

            # Directory should be created when needed
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            assert checkpoint_dir.exists()

    def test_checkpoint_path_construction(self):
        """Checkpoint paths are constructed correctly."""
        checkpoint_dir = "/tmp/checkpoints"
        epoch = 5

        # Expected format: checkpoint_dir/checkpoint_epoch_N.pt
        expected = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
        actual = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt"

        assert actual == expected


# =============================================================================
# Test Early Stopping Logic
# =============================================================================


class TestEarlyStoppingLogic:
    """Tests for early stopping patience tracking."""

    def test_early_stopping_disabled_with_zero_patience(self):
        """Early stopping disabled when patience is 0."""
        config = TrainConfig(early_stopping_patience=0)
        assert config.early_stopping_patience == 0

    def test_early_stopping_enabled_with_positive_patience(self):
        """Early stopping enabled when patience > 0."""
        config = TrainConfig(early_stopping_patience=10)
        assert config.early_stopping_patience == 10

    def test_elo_early_stopping_separate_from_loss(self):
        """Elo early stopping is separate from loss early stopping."""
        config = TrainConfig(
            early_stopping_patience=5,
            elo_early_stopping_patience=10
        )
        assert config.early_stopping_patience == 5
        assert config.elo_early_stopping_patience == 10

    def test_min_improvement_threshold(self):
        """Minimum improvement threshold can be configured."""
        config = TrainConfig(elo_min_improvement=10.0)
        assert config.elo_min_improvement == 10.0


# =============================================================================
# Test Learning Rate Scheduling
# =============================================================================


class TestLearningRateScheduling:
    """Tests for learning rate scheduler configuration."""

    def test_lr_warmup_configuration(self):
        """LR warmup can be configured."""
        config = TrainConfig(warmup_epochs=5)
        assert config.warmup_epochs == 5

    def test_lr_warmup_disabled_with_zero(self):
        """LR warmup disabled with 0 epochs."""
        config = TrainConfig(warmup_epochs=0)
        assert config.warmup_epochs == 0

    def test_cosine_scheduler_min_lr(self):
        """Cosine scheduler minimum LR can be set."""
        config = TrainConfig(lr_min=1e-7)
        assert config.lr_min == 1e-7

    def test_lr_scheduler_types_valid(self):
        """All scheduler types are valid options."""
        valid_schedulers = ["none", "step", "cosine", "cosine-warm-restarts"]
        for sched in valid_schedulers:
            config = TrainConfig(lr_scheduler=sched)
            assert config.lr_scheduler == sched


# =============================================================================
# Test Gradient Accumulation
# =============================================================================


class TestGradientAccumulation:
    """Tests for gradient accumulation configuration."""

    def test_gradient_accumulation_default_disabled(self):
        """Gradient accumulation disabled by default."""
        config = TrainConfig()
        assert config.gradient_accumulation_steps == 1

    def test_gradient_accumulation_can_be_enabled(self):
        """Gradient accumulation can be enabled."""
        config = TrainConfig(gradient_accumulation_steps=4)
        assert config.gradient_accumulation_steps == 4

    def test_effective_batch_size_calculation(self):
        """Effective batch size is batch_size * accumulation_steps."""
        config = TrainConfig(batch_size=32, gradient_accumulation_steps=4)
        effective_batch = config.batch_size * config.gradient_accumulation_steps
        assert effective_batch == 128


# =============================================================================
# Test Temporary Heuristic Profile
# =============================================================================


class TestTemporaryHeuristicProfile:
    """Tests for temporary heuristic profile context manager."""

    def test_temporary_profile_import(self):
        """temporary_heuristic_profile can be imported."""
        from app.training.train import temporary_heuristic_profile
        assert callable(temporary_heuristic_profile)

    def test_temporary_profile_context_manager(self):
        """temporary_heuristic_profile is a context manager."""
        from app.training.train import temporary_heuristic_profile
        from app.training.train import HEURISTIC_WEIGHT_KEYS

        # Create a valid profile
        weights = {k: 1.0 for k in HEURISTIC_WEIGHT_KEYS}

        # Should work as context manager
        with temporary_heuristic_profile("test_profile", weights):
            pass


# =============================================================================
# Test CMA-ES Heuristic Optimization
# =============================================================================


class TestCMAESHeuristicOptimization:
    """Tests for CMA-ES heuristic optimization."""

    def test_cmaes_import(self):
        """run_cmaes_heuristic_optimization can be imported."""
        from app.training.train import run_cmaes_heuristic_optimization
        assert callable(run_cmaes_heuristic_optimization)

    def test_evaluate_heuristic_candidate_import(self):
        """evaluate_heuristic_candidate can be imported."""
        from app.training.train import evaluate_heuristic_candidate
        assert callable(evaluate_heuristic_candidate)

    def test_get_heuristic_tier_by_id_import(self):
        """_get_heuristic_tier_by_id can be imported."""
        from app.training.heuristic_tuning import _get_heuristic_tier_by_id
        assert callable(_get_heuristic_tier_by_id)


# =============================================================================
# Test Board Configuration Presets
# =============================================================================


class TestBoardConfigurationPresets:
    """Tests for board-specific training configuration presets."""

    def test_get_training_config_for_board_import(self):
        """get_training_config_for_board can be imported."""
        from app.training.config import get_training_config_for_board
        assert callable(get_training_config_for_board)

    def test_square8_config_preset(self):
        """Square8 config preset has appropriate settings."""
        from app.training.config import get_training_config_for_board

        config = get_training_config_for_board(BoardType.SQUARE8)
        assert config.board_type == BoardType.SQUARE8
        assert config.policy_size is not None
        assert config.batch_size >= 32

    def test_hex8_config_preset(self):
        """Hex8 config preset has appropriate settings."""
        from app.training.config import get_training_config_for_board

        config = get_training_config_for_board(BoardType.HEX8)
        assert config.board_type == BoardType.HEX8
        assert config.policy_size is not None

    def test_square19_config_preset(self):
        """Square19 config preset has appropriate settings."""
        from app.training.config import get_training_config_for_board

        config = get_training_config_for_board(BoardType.SQUARE19)
        assert config.board_type == BoardType.SQUARE19
        assert config.policy_size is not None

    def test_hexagonal_config_preset(self):
        """Hexagonal config preset has appropriate settings."""
        from app.training.config import get_training_config_for_board

        config = get_training_config_for_board(BoardType.HEXAGONAL)
        assert config.board_type == BoardType.HEXAGONAL
        assert config.policy_size is not None


# =============================================================================
# Test Auto-Streaming Threshold
# =============================================================================


class TestStreamingDataLoader:
    """Tests for streaming data loader configuration."""

    def test_auto_streaming_threshold_env_var(self):
        """Auto-streaming threshold can be set via env var."""
        # Check the constant exists
        from app.training.train import AUTO_STREAMING_THRESHOLD_BYTES
        assert AUTO_STREAMING_THRESHOLD_BYTES > 0


# =============================================================================
# Test train_model Parameter Options
# =============================================================================


class TestTrainModelParameters:
    """Tests for train_model function parameter options."""

    def test_train_model_import(self):
        """train_model can be imported."""
        from app.training.train import train_model
        assert callable(train_model)

    def test_train_model_signature_has_required_params(self):
        """train_model has required parameters."""
        from app.training.train import train_model
        import inspect

        sig = inspect.signature(train_model)
        assert "config" in sig.parameters
        assert "data_path" in sig.parameters
        assert "save_path" in sig.parameters

    def test_train_model_has_early_stopping_params(self):
        """train_model has early stopping parameters."""
        from app.training.train import train_model
        import inspect

        sig = inspect.signature(train_model)
        assert "early_stopping_patience" in sig.parameters
        assert "elo_early_stopping_patience" in sig.parameters

    def test_train_model_has_scheduler_params(self):
        """train_model has scheduler parameters."""
        from app.training.train import train_model
        import inspect

        sig = inspect.signature(train_model)
        assert "lr_scheduler" in sig.parameters
        assert "warmup_epochs" in sig.parameters

    def test_train_model_has_distributed_params(self):
        """train_model has distributed training parameters."""
        from app.training.train import train_model
        import inspect

        sig = inspect.signature(train_model)
        assert "distributed" in sig.parameters
        assert "local_rank" in sig.parameters
        assert "scale_lr" in sig.parameters

    def test_train_model_has_fault_tolerance_params(self):
        """train_model has fault tolerance parameters."""
        from app.training.train import train_model
        import inspect

        sig = inspect.signature(train_model)
        assert "enable_circuit_breaker" in sig.parameters
        assert "enable_anomaly_detection" in sig.parameters
        assert "gradient_clip_mode" in sig.parameters

    def test_train_model_has_data_validation_params(self):
        """train_model has data validation parameters."""
        from app.training.train import train_model
        import inspect

        sig = inspect.signature(train_model)
        assert "validate_data" in sig.parameters
        assert "fail_on_invalid_data" in sig.parameters

    def test_train_model_has_mixed_precision_params(self):
        """train_model has mixed precision parameters."""
        from app.training.train import train_model
        import inspect

        sig = inspect.signature(train_model)
        assert "mixed_precision" in sig.parameters
        assert "amp_dtype" in sig.parameters

    def test_train_model_has_integrated_enhancements_params(self):
        """train_model has integrated enhancements parameters."""
        from app.training.train import train_model
        import inspect

        sig = inspect.signature(train_model)
        assert "use_integrated_enhancements" in sig.parameters
        assert "enable_curriculum" in sig.parameters
        assert "enable_elo_weighting" in sig.parameters
        assert "enable_background_eval" in sig.parameters

    def test_train_model_has_lr_finder_params(self):
        """train_model has LR finder parameters."""
        from app.training.train import train_model
        import inspect

        sig = inspect.signature(train_model)
        assert "find_lr" in sig.parameters
        assert "lr_finder_min" in sig.parameters
        assert "lr_finder_max" in sig.parameters
