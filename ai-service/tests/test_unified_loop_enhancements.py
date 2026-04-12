"""Tests for unified loop enhancements (2025-12).

Tests for:
- TrainingConfig validation
- Auto-recovery retry logic
- Data quality gate enforcement
- Model lifecycle integration
"""

from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTrainingConfigValidation:
    """Tests for TrainingConfig __post_init__ validation."""

    def test_default_config_valid(self):
        """Default TrainingConfig should pass validation."""
        from scripts.unified_loop.config import TrainingConfig

        config = TrainingConfig()
        # Should not raise
        assert config.trigger_threshold_games == 500
        assert config.batch_size == 256

    def test_invalid_threshold_games(self):
        """trigger_threshold_games < 1 should fail."""
        from scripts.unified_loop.config import TrainingConfig

        with pytest.raises(ValueError, match="trigger_threshold_games must be >= 1"):
            TrainingConfig(trigger_threshold_games=0)

    def test_invalid_batch_size(self):
        """batch_size < 1 should fail."""
        from scripts.unified_loop.config import TrainingConfig

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            TrainingConfig(batch_size=0)

    def test_invalid_batch_size_range(self):
        """max_batch_size < min_batch_size should fail."""
        from scripts.unified_loop.config import TrainingConfig

        with pytest.raises(ValueError, match="max_batch_size.*must be >= min_batch_size"):
            TrainingConfig(min_batch_size=512, max_batch_size=256)

    def test_invalid_ratio_above_one(self):
        """Ratio fields > 1.0 should fail."""
        from scripts.unified_loop.config import TrainingConfig

        with pytest.raises(ValueError, match="swa_start_fraction must be between 0.0 and 1.0"):
            TrainingConfig(swa_start_fraction=1.5)

    def test_invalid_ratio_below_zero(self):
        """Ratio fields < 0.0 should fail."""
        from scripts.unified_loop.config import TrainingConfig

        with pytest.raises(ValueError, match="ema_decay must be between 0.0 and 1.0"):
            TrainingConfig(ema_decay=-0.1)

    def test_invalid_retry_settings(self):
        """Invalid retry settings should fail."""
        from scripts.unified_loop.config import TrainingConfig

        with pytest.raises(ValueError, match="training_max_retries must be >= 0"):
            TrainingConfig(training_max_retries=-1)

        with pytest.raises(ValueError, match="training_retry_backoff_base must be > 0"):
            TrainingConfig(training_retry_backoff_base=0)

        with pytest.raises(ValueError, match="training_retry_backoff_multiplier must be >= 1.0"):
            TrainingConfig(training_retry_backoff_multiplier=0.5)

    def test_multiple_errors_reported(self):
        """Multiple validation errors should all be reported."""
        from scripts.unified_loop.config import TrainingConfig

        with pytest.raises(ValueError) as exc_info:
            TrainingConfig(
                batch_size=0,
                min_interval_seconds=-1,
                swa_start_fraction=2.0,
            )

        error_msg = str(exc_info.value)
        assert "batch_size" in error_msg
        assert "min_interval_seconds" in error_msg
        assert "swa_start_fraction" in error_msg


class TestDataQualityGateConfig:
    """Tests for data quality gate configuration."""

    def test_default_gate_settings(self):
        """Default data quality gate settings should be sensible."""
        from scripts.unified_loop.config import TrainingConfig

        config = TrainingConfig()
        assert config.enforce_data_quality_gate is True
        assert config.min_data_quality_for_training == 0.7
        assert config.validate_training_data is True
        assert config.fail_on_invalid_training_data is False

    def test_custom_gate_settings(self):
        """Custom data quality gate settings should work."""
        from scripts.unified_loop.config import TrainingConfig

        config = TrainingConfig(
            enforce_data_quality_gate=False,
            min_data_quality_for_training=0.5,
        )
        assert config.enforce_data_quality_gate is False
        assert config.min_data_quality_for_training == 0.5


class TestAutoRecoveryRetryLogic:
    """Tests for training auto-recovery retry scheduling."""

    @pytest.fixture
    def mock_training_scheduler(self):
        """Create a mock training scheduler with retry state."""
        from scripts.unified_loop.config import TrainingConfig

        # Create minimal mock
        scheduler = MagicMock()
        scheduler.config = TrainingConfig()
        scheduler._retry_attempts = {}
        scheduler._last_failure_time = {}
        scheduler._pending_retries = []
        scheduler._retry_policy = None

        # Import the actual methods
        from scripts.unified_loop.training import TrainingScheduler

        # Bind methods
        scheduler.schedule_training_retry = TrainingScheduler.schedule_training_retry.__get__(scheduler)
        scheduler.reset_retry_count = TrainingScheduler.reset_retry_count.__get__(scheduler)
        scheduler.get_pending_retry = TrainingScheduler.get_pending_retry.__get__(scheduler)

        return scheduler

    def test_schedule_retry_increments_count(self, mock_training_scheduler):
        """schedule_training_retry should increment retry count."""
        scheduler = mock_training_scheduler

        # First retry
        scheduled_time = scheduler.schedule_training_retry("square8_2p")
        assert scheduled_time is not None
        assert scheduler._retry_attempts["square8_2p"] == 1

        # Second retry
        scheduled_time = scheduler.schedule_training_retry("square8_2p")
        assert scheduler._retry_attempts["square8_2p"] == 2

    def test_schedule_retry_exponential_backoff(self, mock_training_scheduler):
        """Retry delay should use exponential backoff."""
        import time
        scheduler = mock_training_scheduler

        # First retry: base_delay * 2^0 = 60
        t1 = scheduler.schedule_training_retry("square8_2p")

        # Second retry: base_delay * 2^1 = 120
        t2 = scheduler.schedule_training_retry("square8_2p")

        # Check delays are increasing
        delay1 = t1 - time.time()
        delay2 = t2 - time.time()
        assert delay2 > delay1

    def test_max_retries_returns_none(self, mock_training_scheduler):
        """schedule_training_retry should return None after max retries."""
        scheduler = mock_training_scheduler

        # Exhaust retries
        for _ in range(3):
            scheduler.schedule_training_retry("square8_2p")

        # Next should return None
        result = scheduler.schedule_training_retry("square8_2p")
        assert result is None

    def test_reset_retry_count_clears_state(self, mock_training_scheduler):
        """reset_retry_count should clear retry state."""
        scheduler = mock_training_scheduler

        # Add some retries
        scheduler.schedule_training_retry("square8_2p")
        scheduler.schedule_training_retry("square8_2p")

        assert scheduler._retry_attempts["square8_2p"] == 2

        # Reset
        scheduler.reset_retry_count("square8_2p")

        assert "square8_2p" not in scheduler._retry_attempts
        assert "square8_2p" not in scheduler._last_failure_time

    def test_get_pending_retry_returns_due(self, mock_training_scheduler):
        """get_pending_retry should return configs that are due."""
        import time
        scheduler = mock_training_scheduler

        # Add a retry that's already due
        scheduler._pending_retries = [("square8_2p", time.time() - 10)]

        result = scheduler.get_pending_retry()
        assert result == "square8_2p"
        assert len(scheduler._pending_retries) == 0

    def test_get_pending_retry_respects_schedule(self, mock_training_scheduler):
        """get_pending_retry should not return configs not yet due."""
        import time
        scheduler = mock_training_scheduler

        # Add a retry in the future
        scheduler._pending_retries = [("square8_2p", time.time() + 3600)]

        result = scheduler.get_pending_retry()
        assert result is None
        assert len(scheduler._pending_retries) == 1


class TestModelLifecycleIntegration:
    """Tests for model lifecycle manager integration."""

    def test_lifecycle_manager_initialization(self):
        """ModelLifecycleManager should initialize in ModelPromoter."""
        from scripts.unified_loop.config import PromotionConfig
        from scripts.unified_loop.promotion import HAS_LIFECYCLE_MANAGER, ModelPromoter

        if not HAS_LIFECYCLE_MANAGER:
            pytest.skip("ModelLifecycleManager not available")

        config = PromotionConfig()
        state = MagicMock()
        event_bus = MagicMock()

        promoter = ModelPromoter(config, state, event_bus)
        assert promoter._lifecycle_manager is not None

    def test_lifecycle_manager_optional(self):
        """ModelPromoter should work without lifecycle manager."""
        from scripts.unified_loop.config import PromotionConfig
        from scripts.unified_loop.promotion import ModelPromoter

        config = PromotionConfig()
        state = MagicMock()
        event_bus = MagicMock()

        with patch('scripts.unified_loop.promotion.HAS_LIFECYCLE_MANAGER', False):
            promoter = ModelPromoter(config, state, event_bus)
            # Should still initialize without error
            assert promoter._lifecycle_manager is None


class TestRetentionPolicyValidation:
    """Tests for RetentionPolicy from model_lifecycle."""

    def test_default_policy(self):
        """Default RetentionPolicy should have sensible values."""
        from app.training.model_lifecycle import RetentionPolicy

        policy = RetentionPolicy()
        assert policy.max_models_per_config == 100
        assert policy.keep_top_by_elo == 50
        assert policy.archive_after_days == 30
        assert policy.delete_archived_after_days == 90

    def test_custom_policy(self):
        """Custom RetentionPolicy values should work."""
        from app.training.model_lifecycle import RetentionPolicy

        policy = RetentionPolicy(
            max_models_per_config=50,
            keep_top_by_elo=10,
            min_models_to_keep=5,
        )
        assert policy.max_models_per_config == 50
        assert policy.keep_top_by_elo == 10
        assert policy.min_models_to_keep == 5
