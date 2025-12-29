"""Unit tests for train_config.py dataclasses.

Tests cover:
- Individual config dataclass defaults and validation
- FullTrainingConfig composition
- config_from_legacy_params() backwards compatibility
- Serialization and deserialization

Created: December 2025
"""

import pytest
from dataclasses import asdict, fields

from app.training.train_config import (
    TrainingDataConfig,
    DistributedConfig,
    CheckpointConfig,
    LearningRateConfig,
    EnhancementConfig,
    FaultToleranceConfig,
    ModelArchConfig,
    EarlyStoppingConfig,
    MixedPrecisionConfig,
    AugmentationConfig,
    HeartbeatConfig,
    FullTrainingConfig,
    config_from_legacy_params,
)


# ==============================================================================
# TrainingDataConfig Tests
# ==============================================================================


class TestTrainingDataConfig:
    """Tests for TrainingDataConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = TrainingDataConfig()

        assert config.data_path == ""
        assert config.use_streaming is False
        assert config.data_dir is None
        assert config.sampling_weights == "uniform"
        assert config.validate_data is True
        assert config.fail_on_invalid_data is False

    def test_freshness_defaults(self):
        """Test freshness check defaults."""
        config = TrainingDataConfig()

        assert config.skip_freshness_check is False
        assert config.max_data_age_hours == 1.0
        assert config.allow_stale_data is False

    def test_quality_discovery_defaults(self):
        """Test quality-aware discovery defaults."""
        config = TrainingDataConfig()

        assert config.discover_synced_data is False
        assert config.min_quality_score == 0.0
        assert config.include_local_data is True
        assert config.include_nfs_data is True

    def test_custom_values(self):
        """Test custom values."""
        config = TrainingDataConfig(
            data_path="data/training/hex8_2p.npz",
            use_streaming=True,
            sampling_weights="late_game",
            max_data_age_hours=2.5,
        )

        assert config.data_path == "data/training/hex8_2p.npz"
        assert config.use_streaming is True
        assert config.sampling_weights == "late_game"
        assert config.max_data_age_hours == 2.5

    def test_list_data_path(self):
        """Test that data_path can be a list of paths."""
        config = TrainingDataConfig(
            data_path=["path1.npz", "path2.npz", "path3.npz"]
        )

        assert isinstance(config.data_path, list)
        assert len(config.data_path) == 3


# ==============================================================================
# DistributedConfig Tests
# ==============================================================================


class TestDistributedConfig:
    """Tests for DistributedConfig dataclass."""

    def test_defaults(self):
        """Test default values (no distributed training)."""
        config = DistributedConfig()

        assert config.distributed is False
        assert config.local_rank == -1
        assert config.scale_lr is False
        assert config.lr_scale_mode == "linear"
        assert config.find_unused_parameters is False

    def test_distributed_enabled(self):
        """Test distributed training configuration."""
        config = DistributedConfig(
            distributed=True,
            local_rank=0,
            scale_lr=True,
            lr_scale_mode="sqrt",
        )

        assert config.distributed is True
        assert config.local_rank == 0
        assert config.scale_lr is True
        assert config.lr_scale_mode == "sqrt"


# ==============================================================================
# CheckpointConfig Tests
# ==============================================================================


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = CheckpointConfig()

        assert config.save_path == ""
        assert config.checkpoint_dir == "checkpoints"
        assert config.checkpoint_interval == 5
        assert config.save_all_epochs is True
        assert config.resume_path is None
        assert config.init_weights_path is None
        assert config.init_weights_strict is False

    def test_checkpoint_averaging_defaults(self):
        """Test checkpoint averaging is enabled by default."""
        config = CheckpointConfig()

        assert config.enable_checkpoint_averaging is True
        assert config.num_checkpoints_to_average == 5

    def test_custom_checkpoint_settings(self):
        """Test custom checkpoint configuration."""
        config = CheckpointConfig(
            save_path="models/best.pth",
            checkpoint_dir="custom_checkpoints",
            checkpoint_interval=10,
            resume_path="models/checkpoint_epoch5.pth",
            num_checkpoints_to_average=3,
        )

        assert config.save_path == "models/best.pth"
        assert config.checkpoint_dir == "custom_checkpoints"
        assert config.checkpoint_interval == 10
        assert config.resume_path == "models/checkpoint_epoch5.pth"
        assert config.num_checkpoints_to_average == 3


# ==============================================================================
# LearningRateConfig Tests
# ==============================================================================


class TestLearningRateConfig:
    """Tests for LearningRateConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = LearningRateConfig()

        assert config.warmup_epochs == 1
        assert config.lr_scheduler == "cosine"
        assert config.lr_min == 1e-6
        assert config.lr_t0 == 10
        assert config.lr_t_mult == 2

    def test_lr_finder_defaults(self):
        """Test LR finder defaults (disabled)."""
        config = LearningRateConfig()

        assert config.find_lr is False
        assert config.lr_finder_min == 1e-7
        assert config.lr_finder_max == 1.0
        assert config.lr_finder_iterations == 100

    def test_scheduler_types(self):
        """Test various scheduler types."""
        for scheduler in ["none", "step", "cosine", "cosine-warm-restarts"]:
            config = LearningRateConfig(lr_scheduler=scheduler)
            assert config.lr_scheduler == scheduler

    def test_cyclic_lr(self):
        """Test cyclic LR configuration."""
        config = LearningRateConfig(
            cyclic_lr=True,
            cyclic_lr_period=10,
        )

        assert config.cyclic_lr is True
        assert config.cyclic_lr_period == 10


# ==============================================================================
# EnhancementConfig Tests
# ==============================================================================


class TestEnhancementConfig:
    """Tests for EnhancementConfig dataclass."""

    def test_december_2025_defaults(self):
        """Test that December 2025 defaults enable enhancements."""
        config = EnhancementConfig()

        # These should be enabled by default as of Dec 28, 2025
        assert config.use_integrated_enhancements is True
        assert config.enable_curriculum is True
        assert config.enable_augmentation is True
        assert config.enable_elo_weighting is True
        assert config.enable_auxiliary_tasks is True
        assert config.enable_background_eval is True

    def test_quality_weighting_defaults(self):
        """Test quality weighting is enabled by default."""
        config = EnhancementConfig()

        assert config.enable_quality_weighting is True
        assert config.quality_weight_blend == 0.7
        assert config.quality_ranking_weight == 0.1

    def test_outcome_weighted_policy_enabled(self):
        """Test outcome-weighted policy is enabled by default."""
        config = EnhancementConfig()

        assert config.enable_outcome_weighted_policy is True
        assert config.outcome_weight_scale == 0.5

    def test_hot_data_buffer_disabled_by_default(self):
        """Test hot data buffer is disabled by default."""
        config = EnhancementConfig()

        assert config.use_hot_data_buffer is False
        assert config.hot_buffer_size == 10000
        assert config.hot_buffer_mix_ratio == 0.3
        assert config.external_hot_buffer is None


# ==============================================================================
# FaultToleranceConfig Tests
# ==============================================================================


class TestFaultToleranceConfig:
    """Tests for FaultToleranceConfig dataclass."""

    def test_defaults_enable_safety(self):
        """Test that safety features are enabled by default."""
        config = FaultToleranceConfig()

        assert config.enable_circuit_breaker is True
        assert config.enable_anomaly_detection is True
        assert config.enable_graceful_shutdown is True

    def test_gradient_clipping_defaults(self):
        """Test gradient clipping configuration."""
        config = FaultToleranceConfig()

        assert config.gradient_clip_mode == "adaptive"
        assert config.gradient_clip_max_norm == 1.0

    def test_anomaly_thresholds(self):
        """Test anomaly detection thresholds."""
        config = FaultToleranceConfig()

        assert config.anomaly_spike_threshold == 3.0
        assert config.anomaly_gradient_threshold == 100.0


# ==============================================================================
# ModelArchConfig Tests
# ==============================================================================


class TestModelArchConfig:
    """Tests for ModelArchConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = ModelArchConfig()

        assert config.model_version == "v2"
        assert config.model_type == "cnn"
        assert config.num_res_blocks is None
        assert config.num_filters is None
        assert config.freeze_policy is False

    def test_december_2025_dropout_increase(self):
        """Test dropout increased to 0.15 as of Dec 28, 2025."""
        config = ModelArchConfig()

        # Increased from 0.08 to reduce overfitting
        assert config.dropout == 0.15

    def test_advanced_features_disabled_by_default(self):
        """Test advanced features are disabled by default."""
        config = ModelArchConfig()

        assert config.spectral_norm is False
        assert config.stochastic_depth is False
        assert config.stochastic_depth_prob == 0.1


# ==============================================================================
# EarlyStoppingConfig Tests
# ==============================================================================


class TestEarlyStoppingConfig:
    """Tests for EarlyStoppingConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = EarlyStoppingConfig()

        assert config.patience == 5
        assert config.elo_patience == 10
        assert config.elo_min_improvement == 5.0


# ==============================================================================
# MixedPrecisionConfig Tests
# ==============================================================================


class TestMixedPrecisionConfig:
    """Tests for MixedPrecisionConfig dataclass."""

    def test_defaults(self):
        """Test mixed precision is disabled by default."""
        config = MixedPrecisionConfig()

        assert config.enabled is False
        assert config.amp_dtype == "bfloat16"

    def test_dtype_options(self):
        """Test dtype options."""
        for dtype in ["bfloat16", "float16"]:
            config = MixedPrecisionConfig(enabled=True, amp_dtype=dtype)
            assert config.amp_dtype == dtype


# ==============================================================================
# AugmentationConfig Tests
# ==============================================================================


class TestAugmentationConfig:
    """Tests for AugmentationConfig dataclass."""

    def test_defaults(self):
        """Test defaults (no augmentation)."""
        config = AugmentationConfig()

        assert config.augment_hex_symmetry is False
        assert config.policy_label_smoothing == 0.0


# ==============================================================================
# HeartbeatConfig Tests
# ==============================================================================


class TestHeartbeatConfig:
    """Tests for HeartbeatConfig dataclass."""

    def test_defaults(self):
        """Test defaults (no heartbeat file)."""
        config = HeartbeatConfig()

        assert config.heartbeat_file is None
        assert config.heartbeat_interval == 30.0


# ==============================================================================
# FullTrainingConfig Tests
# ==============================================================================


class TestFullTrainingConfig:
    """Tests for FullTrainingConfig composition."""

    def test_defaults(self):
        """Test default values for core settings."""
        config = FullTrainingConfig()

        assert config.board_type == "square8"
        assert config.num_players == 2
        assert config.multi_player is False
        assert config.epochs == 50
        assert config.batch_size == 512
        assert config.learning_rate == 0.001

    def test_sub_configs_initialized(self):
        """Test that all sub-configurations are initialized."""
        config = FullTrainingConfig()

        assert isinstance(config.data, TrainingDataConfig)
        assert isinstance(config.distributed, DistributedConfig)
        assert isinstance(config.checkpoint, CheckpointConfig)
        assert isinstance(config.lr, LearningRateConfig)
        assert isinstance(config.enhancements, EnhancementConfig)
        assert isinstance(config.fault_tolerance, FaultToleranceConfig)
        assert isinstance(config.model, ModelArchConfig)
        assert isinstance(config.early_stopping, EarlyStoppingConfig)
        assert isinstance(config.mixed_precision, MixedPrecisionConfig)
        assert isinstance(config.augmentation, AugmentationConfig)
        assert isinstance(config.heartbeat, HeartbeatConfig)

    def test_customize_sub_config(self):
        """Test customizing sub-configurations."""
        config = FullTrainingConfig(
            board_type="hex8",
            num_players=4,
            epochs=100,
            data=TrainingDataConfig(
                data_path="data/training/hex8_4p.npz",
                validate_data=False,
            ),
            distributed=DistributedConfig(
                distributed=True,
                local_rank=0,
            ),
        )

        assert config.board_type == "hex8"
        assert config.num_players == 4
        assert config.epochs == 100
        assert config.data.data_path == "data/training/hex8_4p.npz"
        assert config.data.validate_data is False
        assert config.distributed.distributed is True

    def test_additional_settings(self):
        """Test additional settings outside sub-configs."""
        config = FullTrainingConfig()

        # Value whitening
        assert config.value_whitening is False
        assert config.value_whitening_momentum == 0.99

        # EMA
        assert config.ema is False
        assert config.ema_decay == 0.999

        # Other settings
        assert config.adaptive_warmup is False
        assert config.hard_example_mining is False
        assert config.auto_tune_batch_size is True
        assert config.track_calibration is False

    def test_serialization(self):
        """Test that config can be serialized to dict."""
        config = FullTrainingConfig(
            board_type="hexagonal",
            epochs=75,
        )

        as_dict = asdict(config)

        assert as_dict["board_type"] == "hexagonal"
        assert as_dict["epochs"] == 75
        assert isinstance(as_dict["data"], dict)
        assert as_dict["data"]["data_path"] == ""


# ==============================================================================
# config_from_legacy_params Tests
# ==============================================================================


class TestConfigFromLegacyParams:
    """Tests for backwards compatibility helper."""

    def test_empty_params(self):
        """Test with no parameters returns defaults."""
        config = config_from_legacy_params()

        assert isinstance(config, FullTrainingConfig)
        assert config.data.data_path == ""

    def test_data_path_mapping(self):
        """Test data_path is mapped correctly."""
        config = config_from_legacy_params(
            data_path="data/training/test.npz",
            validate_data=False,
            skip_freshness_check=True,
        )

        assert config.data.data_path == "data/training/test.npz"
        assert config.data.validate_data is False
        assert config.data.skip_freshness_check is True

    def test_distributed_mapping(self):
        """Test distributed params are mapped correctly."""
        config = config_from_legacy_params(
            distributed=True,
            local_rank=0,
            scale_lr=True,
        )

        assert config.distributed.distributed is True
        assert config.distributed.local_rank == 0
        assert config.distributed.scale_lr is True

    def test_checkpoint_mapping(self):
        """Test checkpoint params are mapped correctly."""
        config = config_from_legacy_params(
            save_path="models/best.pth",
            checkpoint_dir="my_checkpoints",
            resume_path="models/resume.pth",
        )

        assert config.checkpoint.save_path == "models/best.pth"
        assert config.checkpoint.checkpoint_dir == "my_checkpoints"
        assert config.checkpoint.resume_path == "models/resume.pth"

    def test_lr_mapping(self):
        """Test learning rate params are mapped correctly."""
        config = config_from_legacy_params(
            warmup_epochs=5,
            lr_scheduler="step",
        )

        assert config.lr.warmup_epochs == 5
        assert config.lr.lr_scheduler == "step"

    def test_enhancement_mapping(self):
        """Test enhancement params are mapped correctly."""
        config = config_from_legacy_params(
            enable_curriculum=False,
            enable_elo_weighting=False,
        )

        assert config.enhancements.enable_curriculum is False
        assert config.enhancements.enable_elo_weighting is False

    def test_fault_tolerance_mapping(self):
        """Test fault tolerance params are mapped correctly."""
        config = config_from_legacy_params(
            enable_circuit_breaker=False,
            gradient_clip_max_norm=2.0,
        )

        assert config.fault_tolerance.enable_circuit_breaker is False
        assert config.fault_tolerance.gradient_clip_max_norm == 2.0

    def test_model_mapping(self):
        """Test model params are mapped correctly."""
        config = config_from_legacy_params(
            model_version="v3",
            model_type="gnn",
        )

        assert config.model.model_version == "v3"
        assert config.model.model_type == "gnn"

    def test_early_stopping_mapping(self):
        """Test early stopping params are mapped correctly."""
        config = config_from_legacy_params(
            early_stopping_patience=10,
            elo_early_stopping_patience=15,
        )

        assert config.early_stopping.patience == 10
        assert config.early_stopping.elo_patience == 15

    def test_mixed_precision_mapping(self):
        """Test mixed precision params are mapped correctly."""
        config = config_from_legacy_params(
            mixed_precision=True,
            amp_dtype="float16",
        )

        assert config.mixed_precision.enabled is True
        assert config.mixed_precision.amp_dtype == "float16"

    def test_augmentation_mapping(self):
        """Test augmentation params are mapped correctly."""
        config = config_from_legacy_params(
            augment_hex_symmetry=True,
        )

        assert config.augmentation.augment_hex_symmetry is True

    def test_heartbeat_mapping(self):
        """Test heartbeat params are mapped correctly."""
        config = config_from_legacy_params(
            heartbeat_file="/tmp/training_heartbeat",
        )

        assert config.heartbeat.heartbeat_file == "/tmp/training_heartbeat"

    def test_unknown_params_ignored(self):
        """Test that unknown parameters don't cause errors."""
        # This should not raise even with unknown params
        config = config_from_legacy_params(
            unknown_param="value",
            another_unknown=123,
        )

        assert isinstance(config, FullTrainingConfig)


# ==============================================================================
# Field Count and Coverage Tests
# ==============================================================================


class TestConfigCompleteness:
    """Tests to verify config dataclasses are complete."""

    def test_full_config_has_all_sub_configs(self):
        """Verify FullTrainingConfig includes all sub-config types."""
        config = FullTrainingConfig()

        # Count sub-config fields
        sub_config_types = [
            TrainingDataConfig,
            DistributedConfig,
            CheckpointConfig,
            LearningRateConfig,
            EnhancementConfig,
            FaultToleranceConfig,
            ModelArchConfig,
            EarlyStoppingConfig,
            MixedPrecisionConfig,
            AugmentationConfig,
            HeartbeatConfig,
        ]

        for config_type in sub_config_types:
            found = False
            for field in fields(config):
                if field.type == config_type or str(field.type) == config_type.__name__:
                    found = True
                    break
                # Check if it's a default_factory for this type
                if hasattr(field.default_factory, '__call__'):
                    try:
                        if isinstance(field.default_factory(), config_type):
                            found = True
                            break
                    except Exception:
                        pass
            assert found, f"FullTrainingConfig missing {config_type.__name__}"

    def test_all_configs_are_dataclasses(self):
        """Verify all config classes are proper dataclasses."""
        from dataclasses import is_dataclass

        configs = [
            TrainingDataConfig,
            DistributedConfig,
            CheckpointConfig,
            LearningRateConfig,
            EnhancementConfig,
            FaultToleranceConfig,
            ModelArchConfig,
            EarlyStoppingConfig,
            MixedPrecisionConfig,
            AugmentationConfig,
            HeartbeatConfig,
            FullTrainingConfig,
        ]

        for config_cls in configs:
            assert is_dataclass(config_cls), f"{config_cls.__name__} is not a dataclass"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
