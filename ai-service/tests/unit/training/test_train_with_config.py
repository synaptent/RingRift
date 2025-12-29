"""Unit tests for train_with_config() function.

Tests cover:
- FullTrainingConfig structure and defaults
- Config to parameter mapping
- train_with_config function signature

Created: December 2025
"""

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# FullTrainingConfig Tests
# =============================================================================


class TestFullTrainingConfig:
    """Tests for FullTrainingConfig dataclass."""

    def test_default_creation(self):
        """FullTrainingConfig can be created with defaults."""
        from app.training.train_config import FullTrainingConfig

        config = FullTrainingConfig()

        assert config.board_type == "square8"
        assert config.num_players == 2
        assert config.epochs == 50
        assert config.batch_size == 512

    def test_sub_configs_exist(self):
        """FullTrainingConfig has all sub-config objects."""
        from app.training.train_config import FullTrainingConfig

        config = FullTrainingConfig()

        assert hasattr(config, 'data')
        assert hasattr(config, 'distributed')
        assert hasattr(config, 'checkpoint')
        assert hasattr(config, 'lr')
        assert hasattr(config, 'enhancements')
        assert hasattr(config, 'fault_tolerance')
        assert hasattr(config, 'model')
        assert hasattr(config, 'early_stopping')
        assert hasattr(config, 'mixed_precision')
        assert hasattr(config, 'augmentation')
        assert hasattr(config, 'heartbeat')

    def test_data_config_defaults(self):
        """TrainingDataConfig has correct defaults."""
        from app.training.train_config import FullTrainingConfig

        config = FullTrainingConfig()

        assert config.data.validate_data is True
        assert config.data.skip_freshness_check is False
        assert config.data.max_data_age_hours == 1.0

    def test_enhancement_config_defaults(self):
        """EnhancementConfig has correct defaults."""
        from app.training.train_config import FullTrainingConfig

        config = FullTrainingConfig()

        assert config.enhancements.use_integrated_enhancements is True
        assert config.enhancements.enable_curriculum is True
        assert config.enhancements.enable_elo_weighting is True
        assert config.enhancements.enable_quality_weighting is True

    def test_outcome_weighted_policy_defaults(self):
        """Outcome weighted policy config has correct defaults."""
        from app.training.train_config import FullTrainingConfig

        config = FullTrainingConfig()

        assert config.enhancements.enable_outcome_weighted_policy is True
        assert config.enhancements.outcome_weight_scale == 0.5

    def test_checkpoint_averaging_defaults(self):
        """Checkpoint averaging config has correct defaults."""
        from app.training.train_config import FullTrainingConfig

        config = FullTrainingConfig()

        assert config.checkpoint.enable_checkpoint_averaging is True
        assert config.checkpoint.num_checkpoints_to_average == 5

    def test_custom_values(self):
        """FullTrainingConfig accepts custom values."""
        from app.training.train_config import (
            FullTrainingConfig,
            EnhancementConfig,
        )

        config = FullTrainingConfig(
            board_type="hex8",
            num_players=4,
            epochs=100,
            enhancements=EnhancementConfig(
                enable_curriculum=False,
            ),
        )

        assert config.board_type == "hex8"
        assert config.num_players == 4
        assert config.epochs == 100
        assert config.enhancements.enable_curriculum is False


# =============================================================================
# Sub-Config Tests
# =============================================================================


class TestTrainingDataConfig:
    """Tests for TrainingDataConfig dataclass."""

    def test_creation(self):
        """TrainingDataConfig can be created."""
        from app.training.train_config import TrainingDataConfig

        config = TrainingDataConfig(
            data_path="data/training/test.npz",
            validate_data=True,
        )

        assert config.data_path == "data/training/test.npz"
        assert config.validate_data is True


class TestDistributedConfig:
    """Tests for DistributedConfig dataclass."""

    def test_creation(self):
        """DistributedConfig can be created."""
        from app.training.train_config import DistributedConfig

        config = DistributedConfig(
            distributed=True,
            local_rank=0,
        )

        assert config.distributed is True
        assert config.local_rank == 0


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_creation(self):
        """CheckpointConfig can be created."""
        from app.training.train_config import CheckpointConfig

        config = CheckpointConfig(
            save_path="models/test.pth",
            checkpoint_dir="checkpoints/test",
        )

        assert config.save_path == "models/test.pth"
        assert config.checkpoint_dir == "checkpoints/test"

    def test_checkpoint_averaging_fields(self):
        """CheckpointConfig has checkpoint averaging fields."""
        from app.training.train_config import CheckpointConfig

        config = CheckpointConfig(
            enable_checkpoint_averaging=True,
            num_checkpoints_to_average=10,
        )

        assert config.enable_checkpoint_averaging is True
        assert config.num_checkpoints_to_average == 10


class TestEnhancementConfig:
    """Tests for EnhancementConfig dataclass."""

    def test_creation(self):
        """EnhancementConfig can be created."""
        from app.training.train_config import EnhancementConfig

        config = EnhancementConfig(
            enable_curriculum=True,
            enable_quality_weighting=True,
        )

        assert config.enable_curriculum is True
        assert config.enable_quality_weighting is True

    def test_outcome_weighted_fields(self):
        """EnhancementConfig has outcome weighted fields."""
        from app.training.train_config import EnhancementConfig

        config = EnhancementConfig(
            enable_outcome_weighted_policy=True,
            outcome_weight_scale=0.7,
        )

        assert config.enable_outcome_weighted_policy is True
        assert config.outcome_weight_scale == 0.7


# =============================================================================
# train_with_config Function Tests
# =============================================================================


class TestTrainWithConfig:
    """Tests for train_with_config function."""

    def test_function_exists(self):
        """train_with_config function exists and is callable."""
        from app.training.train import train_with_config

        assert callable(train_with_config)

    def test_function_signature(self):
        """train_with_config accepts FullTrainingConfig."""
        import inspect
        from app.training.train import train_with_config

        sig = inspect.signature(train_with_config)
        params = list(sig.parameters.keys())

        assert 'full_config' in params

    @patch('app.training.train.train_model')
    def test_calls_train_model(self, mock_train_model):
        """train_with_config calls train_model with unpacked parameters."""
        from app.training.train import train_with_config
        from app.training.train_config import FullTrainingConfig

        mock_train_model.return_value = {
            'best_val_loss': 0.5,
            'final_train_loss': 0.4,
            'final_val_loss': 0.5,
            'epochs_completed': 10,
            'epoch_losses': [],
        }

        config = FullTrainingConfig(
            board_type="hex8",
            num_players=2,
        )
        config.data.data_path = "test.npz"
        config.checkpoint.save_path = "test.pth"

        result = train_with_config(config)

        mock_train_model.assert_called_once()
        assert result['best_val_loss'] == 0.5

    @patch('app.training.train.train_model')
    def test_passes_enhancement_params(self, mock_train_model):
        """train_with_config passes enhancement parameters."""
        from app.training.train import train_with_config
        from app.training.train_config import FullTrainingConfig, EnhancementConfig

        mock_train_model.return_value = {'best_val_loss': 0.5}

        config = FullTrainingConfig(
            enhancements=EnhancementConfig(
                enable_curriculum=True,
                enable_quality_weighting=True,
                enable_outcome_weighted_policy=True,
            ),
        )
        config.data.data_path = "test.npz"
        config.checkpoint.save_path = "test.pth"

        train_with_config(config)

        call_kwargs = mock_train_model.call_args.kwargs
        assert call_kwargs['enable_curriculum'] is True
        assert call_kwargs['enable_quality_weighting'] is True
        assert call_kwargs['enable_outcome_weighted_policy'] is True

    @patch('app.training.train.train_model')
    def test_passes_checkpoint_averaging(self, mock_train_model):
        """train_with_config passes checkpoint averaging parameters."""
        from app.training.train import train_with_config
        from app.training.train_config import FullTrainingConfig, CheckpointConfig

        mock_train_model.return_value = {'best_val_loss': 0.5}

        config = FullTrainingConfig(
            checkpoint=CheckpointConfig(
                enable_checkpoint_averaging=True,
                num_checkpoints_to_average=10,
            ),
        )
        config.data.data_path = "test.npz"
        config.checkpoint.save_path = "test.pth"

        train_with_config(config)

        call_kwargs = mock_train_model.call_args.kwargs
        assert call_kwargs['enable_checkpoint_averaging'] is True
        assert call_kwargs['num_checkpoints_to_average'] == 10


# =============================================================================
# config_from_legacy_params Tests
# =============================================================================


class TestConfigFromLegacyParams:
    """Tests for config_from_legacy_params helper."""

    def test_function_exists(self):
        """config_from_legacy_params function exists."""
        from app.training.train_config import config_from_legacy_params

        assert callable(config_from_legacy_params)

    def test_converts_data_path(self):
        """config_from_legacy_params converts data_path."""
        from app.training.train_config import config_from_legacy_params

        config = config_from_legacy_params(
            data_path="data/test.npz",
        )

        assert config.data.data_path == "data/test.npz"

    def test_converts_distributed(self):
        """config_from_legacy_params converts distributed params."""
        from app.training.train_config import config_from_legacy_params

        config = config_from_legacy_params(
            distributed=True,
            local_rank=0,
        )

        assert config.distributed.distributed is True
        assert config.distributed.local_rank == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
