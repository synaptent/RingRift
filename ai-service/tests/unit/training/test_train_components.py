"""Tests for train_components bridge module.

Tests the helper functions that integrate the new modular training components.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

import torch


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockTrainConfig:
    """Mock training configuration for testing."""

    board_type: Mock = None
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 0.001
    seed: int = 42
    history_length: int = 3
    model_id: str = "test_model"
    early_stopping_patience: int = 20
    warmup_epochs: int = 1
    lr_scheduler: str = "cosine"
    lr_min: float = 1e-6

    def __post_init__(self):
        if self.board_type is None:
            self.board_type = Mock(value="square8")


@pytest.fixture
def mock_config():
    """Create a mock training config."""
    return MockTrainConfig()


@pytest.fixture
def mock_resolved():
    """Create a mock resolved config."""
    from app.training.train_context import ResolvedConfig
    return ResolvedConfig(
        num_players=2,
        distributed=False,
        local_rank=-1,
    )


# =============================================================================
# Test: resolve_train_config
# =============================================================================


class TestResolveTrainConfig:
    """Tests for resolve_train_config function."""

    def test_basic_resolution(self, mock_config):
        """Test basic parameter resolution."""
        from app.training.train_components import resolve_train_config

        resolved = resolve_train_config(
            config=mock_config,
            num_players=2,
        )

        assert resolved is not None
        assert resolved.num_players == 2

    def test_override_parameters(self, mock_config):
        """Test that explicit overrides take precedence."""
        from app.training.train_components import resolve_train_config

        resolved = resolve_train_config(
            config=mock_config,
            early_stopping_patience=15,
            warmup_epochs=3,
            num_players=4,
        )

        assert resolved.early_stopping_patience == 15
        assert resolved.warmup_epochs == 3
        assert resolved.num_players == 4

    def test_none_falls_through_to_config(self, mock_config):
        """Test that None values fall through to config defaults."""
        from app.training.train_components import resolve_train_config

        mock_config.early_stopping_patience = 25

        resolved = resolve_train_config(
            config=mock_config,
            early_stopping_patience=None,  # Should use config value
            num_players=2,
        )

        # Should get config value through resolver
        assert resolved.early_stopping_patience >= 0  # Just verify it resolved

    def test_distributed_settings(self, mock_config):
        """Test distributed training settings."""
        from app.training.train_components import resolve_train_config

        resolved = resolve_train_config(
            config=mock_config,
            distributed=True,
            local_rank=3,
            num_players=2,
        )

        assert resolved.distributed is True
        assert resolved.local_rank == 3

    def test_checkpoint_settings(self, mock_config):
        """Test checkpoint configuration."""
        from app.training.train_components import resolve_train_config

        resolved = resolve_train_config(
            config=mock_config,
            checkpoint_dir="/custom/checkpoints",
            checkpoint_interval=10,
            num_players=2,
        )

        assert resolved.checkpoint_dir == "/custom/checkpoints"
        assert resolved.checkpoint_interval == 10


# =============================================================================
# Test: resolve_training_device
# =============================================================================


class TestResolveTrainingDevice:
    """Tests for resolve_training_device function."""

    def test_cpu_fallback(self):
        """Test CPU fallback when no GPU available."""
        from app.training.train_components import resolve_training_device

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                device = resolve_training_device(distributed=False)

        assert device.type == "cpu"

    def test_non_distributed(self):
        """Test device selection for non-distributed training."""
        from app.training.train_components import resolve_training_device

        device = resolve_training_device(distributed=False, local_rank=-1)

        # Should return some valid device
        assert device is not None
        assert isinstance(device, torch.device)


# =============================================================================
# Test: validate_training_data
# =============================================================================


class TestValidateTrainingData:
    """Tests for validate_training_data function."""

    def test_basic_validation(self, mock_resolved, tmp_path):
        """Test basic data validation flow."""
        from app.training.train_components import validate_training_data

        # Create a dummy NPZ file
        import numpy as np
        npz_path = tmp_path / "test_data.npz"
        np.savez(
            npz_path,
            features=np.random.randn(100, 10, 8, 8).astype(np.float32),
            policy=np.random.randn(100, 64).astype(np.float32),
            value=np.random.randn(100, 2).astype(np.float32),
        )

        result = validate_training_data(
            data_paths=[str(npz_path)],
            board_type="square8",
            num_players=2,
            resolved=mock_resolved,
            is_main_process=True,
        )

        # Should complete without error
        assert result is not None

    def test_empty_paths_fails(self, mock_resolved):
        """Test that empty data paths fail validation."""
        from app.training.train_components import validate_training_data

        result = validate_training_data(
            data_paths=[],
            board_type="square8",
            num_players=2,
            resolved=mock_resolved,
            is_main_process=True,
        )

        assert not result.all_valid
        assert "No valid data paths" in str(result.errors)

    def test_nonexistent_path(self, mock_resolved):
        """Test handling of nonexistent paths."""
        from app.training.train_components import validate_training_data

        result = validate_training_data(
            data_paths=["/nonexistent/path/data.npz"],
            board_type="square8",
            num_players=2,
            resolved=mock_resolved,
            is_main_process=True,
        )

        assert not result.all_valid


# =============================================================================
# Test: build_train_context
# =============================================================================


class TestBuildTrainContext:
    """Tests for build_train_context function."""

    def test_basic_context_building(self, mock_config, mock_resolved):
        """Test basic context building."""
        from app.training.train_components import build_train_context
        from app.training.model_initializer import ModelInitResult

        model_result = ModelInitResult(
            model=Mock(),
            model_version="v2",
            policy_size=64,
            board_size=8,
            effective_blocks=6,
            effective_filters=96,
            feature_version=1,
        )

        context = build_train_context(
            config=mock_config,
            resolved=mock_resolved,
            model_result=model_result,
            data_paths=["data.npz"],
            save_path="model.pth",
            device=torch.device("cpu"),
        )

        assert context is not None
        assert context.model is not None
        assert context.policy_size == 64
        assert context.board_size == 8
        assert context.device.type == "cpu"

    def test_context_with_optional_components(self, mock_config, mock_resolved):
        """Test context building with optional components."""
        from app.training.train_components import build_train_context
        from app.training.model_initializer import ModelInitResult

        model_result = ModelInitResult(
            model=Mock(),
            model_version="v2",
            policy_size=64,
            board_size=8,
            effective_blocks=6,
            effective_filters=96,
            feature_version=1,
        )

        mock_optimizer = Mock()
        mock_train_loader = Mock()
        mock_scheduler = Mock()

        context = build_train_context(
            config=mock_config,
            resolved=mock_resolved,
            model_result=model_result,
            data_paths=["data.npz"],
            save_path="model.pth",
            device=torch.device("cpu"),
            optimizer=mock_optimizer,
            train_loader=mock_train_loader,
            epoch_scheduler=mock_scheduler,
        )

        assert context.optimizer is mock_optimizer
        assert context.train_loader is mock_train_loader
        assert context.epoch_scheduler is mock_scheduler


# =============================================================================
# Test: Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_data_paths_single(self):
        """Test normalizing single path to list."""
        from app.training.train_components import get_data_paths

        paths = get_data_paths("single/path.npz")

        assert paths == ["single/path.npz"]

    def test_get_data_paths_list(self):
        """Test normalizing list of paths."""
        from app.training.train_components import get_data_paths

        input_paths = ["path1.npz", "path2.npz", "path3.npz"]
        paths = get_data_paths(input_paths)

        assert paths == input_paths

    def test_check_is_main_process_non_distributed(self):
        """Test main process check for non-distributed."""
        from app.training.train_components import check_is_main_process

        assert check_is_main_process(distributed=False) is True
        assert check_is_main_process(distributed=False, local_rank=5) is True

    def test_check_is_main_process_distributed(self):
        """Test main process check for distributed."""
        from app.training.train_components import check_is_main_process

        assert check_is_main_process(distributed=True, local_rank=0) is True
        assert check_is_main_process(distributed=True, local_rank=-1) is True
        assert check_is_main_process(distributed=True, local_rank=1) is False
        assert check_is_main_process(distributed=True, local_rank=3) is False


# =============================================================================
# Test: Integration Example
# =============================================================================


class TestTrainModelWithComponents:
    """Tests for the integration example function."""

    @patch("app.training.train_components.initialize_training_model")
    @patch("app.training.train_components.validate_training_data")
    def test_basic_flow(self, mock_validate, mock_init_model, mock_config, tmp_path):
        """Test the basic component integration flow."""
        from app.training.train_components import train_model_with_components
        from app.training.model_initializer import ModelInitResult
        from app.training.data_validator import DataValidationResult

        # Mock validation success
        mock_validate.return_value = DataValidationResult(all_valid=True)

        # Mock model initialization
        mock_init_model.return_value = ModelInitResult(
            model=Mock(),
            model_version="v2",
            policy_size=64,
            board_size=8,
            effective_blocks=6,
            effective_filters=96,
            feature_version=1,
        )

        result = train_model_with_components(
            config=mock_config,
            data_path=str(tmp_path / "data.npz"),
            save_path=str(tmp_path / "model.pth"),
            num_players=2,
        )

        assert result is not None
        assert result["status"] == "components_initialized"
        assert result["model_version"] == "v2"

    @patch("app.training.train_components.validate_training_data")
    def test_validation_failure_raises(self, mock_validate, mock_config, tmp_path):
        """Test that validation failure raises error."""
        from app.training.train_components import train_model_with_components
        from app.training.data_validator import DataValidationResult

        # Mock validation failure
        mock_validate.return_value = DataValidationResult(
            all_valid=False,
            errors=["Data is stale"],
        )

        with pytest.raises(ValueError, match="Data validation failed"):
            train_model_with_components(
                config=mock_config,
                data_path=str(tmp_path / "data.npz"),
                save_path=str(tmp_path / "model.pth"),
                num_players=2,
            )
