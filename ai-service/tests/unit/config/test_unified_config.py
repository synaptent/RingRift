"""
Unit tests for app.config.unified_config module.

Tests cover:
- DataIngestionConfig dataclass
- TrainingConfig dataclass
- EvaluationConfig dataclass
- PromotionConfig dataclass
- BoardConfig dataclass
- ClusterConfig dataclass
- UnifiedConfig main class
- get_config singleton function

Created: December 2025
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import fields


# =============================================================================
# DataIngestionConfig Tests
# =============================================================================


class TestDataIngestionConfig:
    """Tests for DataIngestionConfig dataclass."""

    def test_default_creation(self):
        """DataIngestionConfig can be created with defaults."""
        from app.config.unified_config import DataIngestionConfig

        config = DataIngestionConfig()

        assert config.poll_interval_seconds == 60
        assert config.sync_method == "incremental"
        assert config.deduplication is True
        assert config.min_games_per_sync == 5

    def test_custom_values(self):
        """DataIngestionConfig accepts custom values."""
        from app.config.unified_config import DataIngestionConfig

        config = DataIngestionConfig(
            poll_interval_seconds=120,
            sync_method="full",
            deduplication=False,
        )

        assert config.poll_interval_seconds == 120
        assert config.sync_method == "full"
        assert config.deduplication is False

    def test_ephemeral_poll_interval(self):
        """DataIngestionConfig has ephemeral poll interval."""
        from app.config.unified_config import DataIngestionConfig

        config = DataIngestionConfig()

        assert config.ephemeral_poll_interval_seconds == 15

    def test_checksum_validation_default(self):
        """Checksum validation is enabled by default."""
        from app.config.unified_config import DataIngestionConfig

        config = DataIngestionConfig()

        assert config.checksum_validation is True


# =============================================================================
# TrainingConfig Tests
# =============================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_creation(self):
        """TrainingConfig can be created with defaults."""
        from app.config.unified_config import TrainingConfig

        config = TrainingConfig()

        assert config.trigger_threshold_games == 500
        assert config.min_interval_seconds == 1200  # 20 minutes
        assert config.max_concurrent_jobs == 1

    def test_warm_start_enabled(self):
        """Warm start is enabled by default."""
        from app.config.unified_config import TrainingConfig

        config = TrainingConfig()

        assert config.warm_start is True

    def test_validation_split(self):
        """Validation split has correct default."""
        from app.config.unified_config import TrainingConfig

        config = TrainingConfig()

        assert config.validation_split == 0.1

    def test_nnue_thresholds(self):
        """NNUE thresholds are configured."""
        from app.config.unified_config import TrainingConfig

        config = TrainingConfig()

        assert config.nnue_min_games >= 1000
        assert config.nnue_policy_min_games >= 1000
        assert config.cmaes_min_games >= 1000

    def test_mixed_precision_default(self):
        """Mixed precision is enabled by default."""
        from app.config.unified_config import TrainingConfig

        config = TrainingConfig()

        assert config.use_mixed_precision is True


# =============================================================================
# EvaluationConfig Tests
# =============================================================================


class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""

    def test_class_exists(self):
        """EvaluationConfig class exists."""
        from app.config.unified_config import EvaluationConfig
        assert EvaluationConfig is not None

    def test_default_creation(self):
        """EvaluationConfig can be created with defaults."""
        from app.config.unified_config import EvaluationConfig

        config = EvaluationConfig()
        assert config is not None


# =============================================================================
# PromotionConfig Tests
# =============================================================================


class TestPromotionConfig:
    """Tests for PromotionConfig dataclass."""

    def test_class_exists(self):
        """PromotionConfig class exists."""
        from app.config.unified_config import PromotionConfig
        assert PromotionConfig is not None

    def test_default_creation(self):
        """PromotionConfig can be created with defaults."""
        from app.config.unified_config import PromotionConfig

        config = PromotionConfig()
        assert config is not None


# =============================================================================
# CurriculumConfig Tests
# =============================================================================


class TestCurriculumConfig:
    """Tests for CurriculumConfig dataclass."""

    def test_class_exists(self):
        """CurriculumConfig class exists."""
        from app.config.unified_config import CurriculumConfig
        assert CurriculumConfig is not None


# =============================================================================
# BoardConfig Tests
# =============================================================================


class TestBoardConfig:
    """Tests for BoardConfig dataclass."""

    def test_class_exists(self):
        """BoardConfig class exists."""
        from app.config.unified_config import BoardConfig
        assert BoardConfig is not None

    def test_creation_with_values(self):
        """BoardConfig can be created with values."""
        from app.config.unified_config import BoardConfig

        config = BoardConfig(
            board_type="hex8",
            num_players=2,
        )

        assert config.board_type == "hex8"
        assert config.num_players == 2

    def test_config_key_property(self):
        """BoardConfig has config_key property."""
        from app.config.unified_config import BoardConfig

        config = BoardConfig(
            board_type="square8",
            num_players=4,
        )

        assert config.config_key == "square8_4p"


# =============================================================================
# ClusterConfig Tests
# =============================================================================


class TestClusterConfig:
    """Tests for ClusterConfig dataclass."""

    def test_class_exists(self):
        """ClusterConfig class exists."""
        from app.config.unified_config import ClusterConfig
        assert ClusterConfig is not None

    def test_default_creation(self):
        """ClusterConfig can be created with defaults."""
        from app.config.unified_config import ClusterConfig

        config = ClusterConfig()
        assert config is not None


# =============================================================================
# SafeguardsConfig Tests
# =============================================================================


class TestSafeguardsConfig:
    """Tests for SafeguardsConfig dataclass."""

    def test_class_exists(self):
        """SafeguardsConfig class exists."""
        from app.config.unified_config import SafeguardsConfig
        assert SafeguardsConfig is not None


# =============================================================================
# DistributedConfig Tests
# =============================================================================


class TestDistributedConfig:
    """Tests for DistributedConfig dataclass."""

    def test_class_exists(self):
        """DistributedConfig class exists."""
        from app.config.unified_config import DistributedConfig
        assert DistributedConfig is not None


# =============================================================================
# UnifiedConfig Tests
# =============================================================================


class TestUnifiedConfig:
    """Tests for UnifiedConfig main class."""

    def test_class_exists(self):
        """UnifiedConfig class exists."""
        from app.config.unified_config import UnifiedConfig
        assert UnifiedConfig is not None

    def test_default_creation(self):
        """UnifiedConfig can be created with defaults."""
        from app.config.unified_config import UnifiedConfig

        config = UnifiedConfig()
        assert config is not None

    def test_has_training_config(self):
        """UnifiedConfig has training sub-config."""
        from app.config.unified_config import UnifiedConfig

        config = UnifiedConfig()
        assert hasattr(config, 'training')
        assert config.training is not None

    def test_has_evaluation_config(self):
        """UnifiedConfig has evaluation sub-config."""
        from app.config.unified_config import UnifiedConfig

        config = UnifiedConfig()
        assert hasattr(config, 'evaluation')

    def test_has_promotion_config(self):
        """UnifiedConfig has promotion sub-config."""
        from app.config.unified_config import UnifiedConfig

        config = UnifiedConfig()
        assert hasattr(config, 'promotion')

    def test_has_cluster_config(self):
        """UnifiedConfig has cluster sub-config."""
        from app.config.unified_config import UnifiedConfig

        config = UnifiedConfig()
        assert hasattr(config, 'cluster')

    def test_get_all_board_configs_method(self):
        """UnifiedConfig has get_all_board_configs method."""
        from app.config.unified_config import UnifiedConfig

        config = UnifiedConfig()
        assert hasattr(config, 'get_all_board_configs')
        assert callable(config.get_all_board_configs)


# =============================================================================
# get_config Singleton Tests
# =============================================================================


class TestGetConfig:
    """Tests for get_config singleton function."""

    def test_function_exists(self):
        """get_config function exists."""
        from app.config.unified_config import get_config
        assert callable(get_config)

    def test_returns_unified_config(self):
        """get_config returns UnifiedConfig instance."""
        from app.config.unified_config import get_config, UnifiedConfig

        config = get_config()
        assert isinstance(config, UnifiedConfig)

    def test_singleton_behavior(self):
        """get_config returns same instance on repeated calls."""
        from app.config.unified_config import get_config

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2


# =============================================================================
# Configuration Loading Tests
# =============================================================================


class TestConfigLoading:
    """Tests for configuration loading from files."""

    def test_default_config_path_defined(self):
        """DEFAULT_CONFIG_PATH is defined."""
        from app.config.unified_config import DEFAULT_CONFIG_PATH
        assert DEFAULT_CONFIG_PATH is not None
        assert isinstance(DEFAULT_CONFIG_PATH, str)

    def test_env_override_constants(self):
        """Environment override constants are documented."""
        # These are documented in module docstring
        from app.config.unified_config import get_config

        # Just verify module loads successfully
        assert get_config() is not None


# =============================================================================
# All Config Classes Tests
# =============================================================================


class TestAllConfigClasses:
    """Tests verifying all config dataclasses exist."""

    def test_data_ingestion_config_exists(self):
        """DataIngestionConfig is importable."""
        from app.config.unified_config import DataIngestionConfig
        assert DataIngestionConfig is not None

    def test_training_config_exists(self):
        """TrainingConfig is importable."""
        from app.config.unified_config import TrainingConfig
        assert TrainingConfig is not None

    def test_evaluation_config_exists(self):
        """EvaluationConfig is importable."""
        from app.config.unified_config import EvaluationConfig
        assert EvaluationConfig is not None

    def test_promotion_config_exists(self):
        """PromotionConfig is importable."""
        from app.config.unified_config import PromotionConfig
        assert PromotionConfig is not None

    def test_curriculum_config_exists(self):
        """CurriculumConfig is importable."""
        from app.config.unified_config import CurriculumConfig
        assert CurriculumConfig is not None

    def test_safeguards_config_exists(self):
        """SafeguardsConfig is importable."""
        from app.config.unified_config import SafeguardsConfig
        assert SafeguardsConfig is not None

    def test_board_config_exists(self):
        """BoardConfig is importable."""
        from app.config.unified_config import BoardConfig
        assert BoardConfig is not None

    def test_regression_config_exists(self):
        """RegressionConfig is importable."""
        from app.config.unified_config import RegressionConfig
        assert RegressionConfig is not None

    def test_alerting_config_exists(self):
        """AlertingConfig is importable."""
        from app.config.unified_config import AlertingConfig
        assert AlertingConfig is not None

    def test_safety_config_exists(self):
        """SafetyConfig is importable."""
        from app.config.unified_config import SafetyConfig
        assert SafetyConfig is not None

    def test_plateau_detection_config_exists(self):
        """PlateauDetectionConfig is importable."""
        from app.config.unified_config import PlateauDetectionConfig
        assert PlateauDetectionConfig is not None

    def test_replay_buffer_config_exists(self):
        """ReplayBufferConfig is importable."""
        from app.config.unified_config import ReplayBufferConfig
        assert ReplayBufferConfig is not None

    def test_cluster_config_exists(self):
        """ClusterConfig is importable."""
        from app.config.unified_config import ClusterConfig
        assert ClusterConfig is not None

    def test_ssh_config_exists(self):
        """SSHConfig is importable."""
        from app.config.unified_config import SSHConfig
        assert SSHConfig is not None

    def test_slurm_config_exists(self):
        """SlurmConfig is importable."""
        from app.config.unified_config import SlurmConfig
        assert SlurmConfig is not None

    def test_distributed_config_exists(self):
        """DistributedConfig is importable."""
        from app.config.unified_config import DistributedConfig
        assert DistributedConfig is not None

    def test_selfplay_defaults_exists(self):
        """SelfplayDefaults is importable."""
        from app.config.unified_config import SelfplayDefaults
        assert SelfplayDefaults is not None

    def test_tournament_config_exists(self):
        """TournamentConfig is importable."""
        from app.config.unified_config import TournamentConfig
        assert TournamentConfig is not None

    def test_integrated_enhancements_config_exists(self):
        """IntegratedEnhancementsConfig is importable."""
        from app.config.unified_config import IntegratedEnhancementsConfig
        assert IntegratedEnhancementsConfig is not None

    def test_health_config_exists(self):
        """HealthConfig is importable."""
        from app.config.unified_config import HealthConfig
        assert HealthConfig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
