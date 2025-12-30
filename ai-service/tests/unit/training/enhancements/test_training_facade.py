"""Tests for app.training.enhancements.training_facade.

Tests the unified training enhancements facade including:
- FacadeConfig and EpochStatistics dataclasses
- TrainingEnhancementsFacade class methods
- EnhancementConfig and EnhancementComponents
- initialize_all_enhancements() factory
- Singleton accessor functions
"""

import math
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.training.enhancements.training_facade import (
    EnhancementComponents,
    EnhancementConfig,
    EpochStatistics,
    FacadeConfig,
    TrainingEnhancementsFacade,
    get_facade,
    initialize_all_enhancements,
    reset_facade,
)


class TestFacadeConfig:
    """Tests for FacadeConfig dataclass."""

    def test_default_values(self):
        """Test FacadeConfig has sensible defaults."""
        config = FacadeConfig()

        assert config.enable_hard_mining is True
        assert config.hard_fraction == 0.3
        assert config.hard_buffer_size == 10000
        assert config.hard_min_samples_before_mining == 1000
        assert config.hard_base_weight == 1.0
        assert config.hard_upweight == 2.0

        assert config.track_per_sample_loss is True
        assert config.loss_tracker_max_samples == 10000

        assert config.enable_curriculum_lr is True
        assert config.curriculum_lr_min_scale == 0.8
        assert config.curriculum_lr_max_scale == 1.2
        assert config.curriculum_warmup_fraction == 0.1

        assert config.enable_freshness_weighting is True
        assert config.freshness_decay_hours == 24.0
        assert config.freshness_weight == 0.2

        assert config.policy_weight == 1.0

    def test_custom_values(self):
        """Test FacadeConfig accepts custom values."""
        config = FacadeConfig(
            enable_hard_mining=False,
            hard_fraction=0.5,
            hard_buffer_size=5000,
            curriculum_lr_max_scale=1.5,
        )

        assert config.enable_hard_mining is False
        assert config.hard_fraction == 0.5
        assert config.hard_buffer_size == 5000
        assert config.curriculum_lr_max_scale == 1.5


class TestEpochStatistics:
    """Tests for EpochStatistics dataclass."""

    def test_default_values(self):
        """Test EpochStatistics has sensible defaults."""
        stats = EpochStatistics()

        assert stats.mean_loss == 0.0
        assert stats.mean_per_sample_loss == 0.0
        assert stats.hard_examples_fraction == 0.0
        assert stats.mining_active is False
        assert stats.tracked_samples == 0
        assert stats.curriculum_lr_scale == 1.0

    def test_custom_values(self):
        """Test EpochStatistics accepts custom values."""
        stats = EpochStatistics(
            mean_loss=1.5,
            hard_examples_fraction=0.3,
            mining_active=True,
            tracked_samples=1000,
        )

        assert stats.mean_loss == 1.5
        assert stats.hard_examples_fraction == 0.3
        assert stats.mining_active is True
        assert stats.tracked_samples == 1000


class TestTrainingEnhancementsFacade:
    """Tests for TrainingEnhancementsFacade class."""

    def test_initialization_default_config(self):
        """Test facade initializes with default config."""
        facade = TrainingEnhancementsFacade()

        assert facade.config is not None
        assert facade._epoch == 0
        assert facade._total_epochs == 1
        assert facade._total_samples_seen == 0

    def test_initialization_custom_config(self):
        """Test facade initializes with custom config."""
        config = FacadeConfig(enable_hard_mining=False)
        facade = TrainingEnhancementsFacade(config=config)

        assert facade.config.enable_hard_mining is False
        assert facade._miner is None

    def test_set_total_epochs(self):
        """Test set_total_epochs updates internal state."""
        facade = TrainingEnhancementsFacade()

        facade.set_total_epochs(100)
        assert facade._total_epochs == 100

        # Test minimum of 1
        facade.set_total_epochs(0)
        assert facade._total_epochs == 1

    def test_compute_per_sample_loss(self):
        """Test compute_per_sample_loss returns correct shape."""
        facade = TrainingEnhancementsFacade()

        batch_size = 32
        num_actions = 64

        policy_logits = torch.randn(batch_size, num_actions)
        policy_targets = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        value_pred = torch.randn(batch_size)
        value_targets = torch.rand(batch_size) * 2 - 1

        losses = facade.compute_per_sample_loss(
            policy_logits, policy_targets, value_pred, value_targets
        )

        assert losses.shape == (batch_size,)
        assert torch.all(losses >= 0)

    def test_compute_per_sample_loss_reduction_mean(self):
        """Test compute_per_sample_loss with mean reduction."""
        facade = TrainingEnhancementsFacade()

        batch_size = 32
        num_actions = 64

        policy_logits = torch.randn(batch_size, num_actions)
        policy_targets = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        value_pred = torch.randn(batch_size)
        value_targets = torch.rand(batch_size) * 2 - 1

        loss = facade.compute_per_sample_loss(
            policy_logits, policy_targets, value_pred, value_targets,
            reduction="mean"
        )

        assert loss.ndim == 0  # Scalar

    def test_get_weighted_loss_no_mining(self):
        """Test get_weighted_loss without mining returns mean."""
        config = FacadeConfig(enable_hard_mining=False)
        facade = TrainingEnhancementsFacade(config=config)

        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weighted = facade.get_weighted_loss(losses)

        assert torch.isclose(weighted, torch.tensor(2.5))

    def test_get_weighted_loss_with_mining(self):
        """Test get_weighted_loss with mining enabled."""
        config = FacadeConfig(enable_hard_mining=True)
        facade = TrainingEnhancementsFacade(config=config)

        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        indices = [0, 1, 2, 3]

        weighted = facade.get_weighted_loss(losses, batch_indices=indices)

        # Should return a tensor
        assert isinstance(weighted, torch.Tensor)
        assert weighted.ndim == 0  # Scalar

    def test_record_batch(self):
        """Test record_batch updates internal state."""
        facade = TrainingEnhancementsFacade()

        batch_indices = [0, 1, 2, 3]
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])

        facade.record_batch(batch_indices, losses)

        assert facade._total_samples_seen == 4
        assert facade._epoch_sample_count == 4
        assert facade._epoch_loss_sum > 0

    def test_record_batch_with_tensor_indices(self):
        """Test record_batch accepts tensor indices."""
        facade = TrainingEnhancementsFacade()

        batch_indices = torch.tensor([0, 1, 2, 3])
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])

        facade.record_batch(batch_indices, losses)

        assert facade._total_samples_seen == 4

    def test_get_curriculum_lr_scale_warmup(self):
        """Test curriculum LR during warmup phase."""
        config = FacadeConfig(
            curriculum_warmup_fraction=0.1,
            curriculum_lr_min_scale=0.8,
        )
        facade = TrainingEnhancementsFacade(config=config)

        # At start of warmup
        scale = facade.get_curriculum_lr_scale(0.0)
        assert scale == pytest.approx(0.8, rel=0.01)

        # Mid warmup (5% progress)
        scale = facade.get_curriculum_lr_scale(0.05)
        assert 0.8 < scale < 1.0

    def test_get_curriculum_lr_scale_main_phase(self):
        """Test curriculum LR during main training phase."""
        config = FacadeConfig(
            curriculum_warmup_fraction=0.1,
            curriculum_lr_max_scale=1.2,
        )
        facade = TrainingEnhancementsFacade(config=config)

        # Just after warmup
        scale = facade.get_curriculum_lr_scale(0.1)
        assert scale == pytest.approx(1.0, rel=0.01)

        # End of training
        scale = facade.get_curriculum_lr_scale(1.0)
        assert scale == pytest.approx(1.2, rel=0.01)

    def test_get_curriculum_lr_scale_disabled(self):
        """Test curriculum LR when disabled."""
        config = FacadeConfig(enable_curriculum_lr=False)
        facade = TrainingEnhancementsFacade(config=config)

        assert facade.get_curriculum_lr_scale(0.0) == 1.0
        assert facade.get_curriculum_lr_scale(0.5) == 1.0
        assert facade.get_curriculum_lr_scale(1.0) == 1.0

    def test_compute_freshness_weight_recent(self):
        """Test freshness weight for recent games."""
        facade = TrainingEnhancementsFacade()

        now = time.time()
        weight = facade.compute_freshness_weight(now, now)

        assert weight == pytest.approx(1.0, rel=0.01)

    def test_compute_freshness_weight_old(self):
        """Test freshness weight for old games."""
        config = FacadeConfig(freshness_decay_hours=24.0)
        facade = TrainingEnhancementsFacade(config=config)

        now = time.time()
        one_day_ago = now - 24 * 3600

        weight = facade.compute_freshness_weight(one_day_ago, now)

        # Should be approximately 1/e ≈ 0.368
        assert weight == pytest.approx(1/math.e, rel=0.01)

    def test_compute_freshness_weight_disabled(self):
        """Test freshness weight when disabled."""
        config = FacadeConfig(enable_freshness_weighting=False)
        facade = TrainingEnhancementsFacade(config=config)

        now = time.time()
        old_time = now - 7 * 24 * 3600

        weight = facade.compute_freshness_weight(old_time, now)

        assert weight == 1.0

    def test_compute_freshness_weight_future(self):
        """Test freshness weight handles future timestamps."""
        facade = TrainingEnhancementsFacade()

        now = time.time()
        future = now + 3600

        weight = facade.compute_freshness_weight(future, now)

        assert weight == 1.0

    def test_get_epoch_statistics(self):
        """Test get_epoch_statistics returns valid stats."""
        facade = TrainingEnhancementsFacade()

        # Record some samples
        for i in range(100):
            facade.record_batch([i], torch.tensor([float(i) / 100]))

        stats = facade.get_epoch_statistics()

        assert isinstance(stats, EpochStatistics)
        assert stats.tracked_samples >= 0
        assert stats.mean_loss >= 0

    def test_on_epoch_end(self):
        """Test on_epoch_end advances epoch and returns stats."""
        facade = TrainingEnhancementsFacade()

        # Record some samples
        facade.record_batch([0, 1, 2], torch.tensor([1.0, 2.0, 3.0]))

        result = facade.on_epoch_end()

        assert facade._epoch == 1
        assert facade._epoch_loss_sum == 0.0
        assert facade._epoch_sample_count == 0
        assert 'mean_loss' in result
        assert 'epoch' in result
        assert result['epoch'] == 0

    def test_get_hard_indices_no_mining(self):
        """Test get_hard_indices without mining returns empty."""
        config = FacadeConfig(enable_hard_mining=False)
        facade = TrainingEnhancementsFacade(config=config)

        indices = facade.get_hard_indices(100)

        assert len(indices) == 0

    def test_get_hardest_samples_no_tracking(self):
        """Test get_hardest_samples without tracking returns empty."""
        config = FacadeConfig(track_per_sample_loss=False)
        facade = TrainingEnhancementsFacade(config=config)

        samples = facade.get_hardest_samples(10)

        assert len(samples) == 0

    def test_reset(self):
        """Test reset clears all state."""
        facade = TrainingEnhancementsFacade()

        # Build up some state
        facade.record_batch([0, 1, 2], torch.tensor([1.0, 2.0, 3.0]))
        facade.on_epoch_end()
        facade.record_batch([3, 4, 5], torch.tensor([4.0, 5.0, 6.0]))

        facade.reset()

        assert facade._epoch == 0
        assert facade._total_samples_seen == 0
        assert facade._epoch_loss_sum == 0.0
        assert facade._epoch_sample_count == 0

    def test_is_mining_active(self):
        """Test is_mining_active property."""
        config = FacadeConfig(
            enable_hard_mining=True,
            hard_min_samples_before_mining=100,
        )
        facade = TrainingEnhancementsFacade(config=config)

        assert facade.is_mining_active is False

        # Record enough samples
        for i in range(100):
            facade.record_batch([i], torch.tensor([1.0]))

        assert facade.is_mining_active is True

    def test_is_mining_active_disabled(self):
        """Test is_mining_active when mining disabled."""
        config = FacadeConfig(enable_hard_mining=False)
        facade = TrainingEnhancementsFacade(config=config)

        assert facade.is_mining_active is False


class TestEnhancementConfig:
    """Tests for EnhancementConfig dataclass."""

    def test_default_values(self):
        """Test EnhancementConfig has sensible defaults."""
        config = EnhancementConfig()

        assert config.enable_hot_buffer is True
        assert config.hot_buffer_max_size == 100000
        assert config.hot_buffer_update_interval == 60.0
        assert config.enable_quality_bridge is True
        assert config.quality_threshold == 0.5
        assert config.enable_enhancements_manager is True
        assert config.start_background_services is True

    def test_custom_values(self):
        """Test EnhancementConfig accepts custom values."""
        facade_config = FacadeConfig(enable_hard_mining=False)
        config = EnhancementConfig(
            facade_config=facade_config,
            enable_hot_buffer=False,
            quality_threshold=0.7,
        )

        assert config.facade_config.enable_hard_mining is False
        assert config.enable_hot_buffer is False
        assert config.quality_threshold == 0.7


class TestEnhancementComponents:
    """Tests for EnhancementComponents dataclass."""

    def test_default_values(self):
        """Test EnhancementComponents has None defaults."""
        components = EnhancementComponents()

        assert components.facade is None
        assert components.hot_buffer is None
        assert components.quality_bridge is None
        assert components.enhancements_manager is None
        assert components.started is False

    def test_start_background_services_no_manager(self):
        """Test start_background_services does nothing without manager."""
        components = EnhancementComponents()

        # Should not raise
        components.start_background_services()

        assert components.started is False

    def test_stop_background_services_no_manager(self):
        """Test stop_background_services does nothing without manager."""
        components = EnhancementComponents()

        # Should not raise
        components.stop_background_services()

        assert components.started is False

    def test_start_background_services_with_manager(self):
        """Test start_background_services with mock manager."""
        components = EnhancementComponents()

        mock_manager = MagicMock()
        components.enhancements_manager = mock_manager

        components.start_background_services()

        mock_manager.start_background_services.assert_called_once()
        assert components.started is True

    def test_stop_background_services_with_manager(self):
        """Test stop_background_services with mock manager."""
        components = EnhancementComponents()
        components.started = True

        mock_manager = MagicMock()
        components.enhancements_manager = mock_manager

        components.stop_background_services()

        mock_manager.stop_background_services.assert_called_once()
        assert components.started is False


class TestInitializeAllEnhancements:
    """Tests for initialize_all_enhancements factory function."""

    def test_default_initialization(self):
        """Test initialize_all_enhancements with defaults."""
        components = initialize_all_enhancements()

        assert components.facade is not None
        assert isinstance(components.facade, TrainingEnhancementsFacade)

    def test_with_custom_config(self):
        """Test initialize_all_enhancements with custom config."""
        config = EnhancementConfig(
            enable_hot_buffer=False,
            enable_quality_bridge=False,
            enable_enhancements_manager=False,
            start_background_services=False,
        )

        components = initialize_all_enhancements(config=config)

        assert components.facade is not None
        assert components.hot_buffer is None  # Disabled
        assert components.quality_bridge is None  # Disabled
        assert components.enhancements_manager is None  # Disabled
        assert components.started is False

    def test_with_board_info(self):
        """Test initialize_all_enhancements with board info."""
        config = EnhancementConfig(
            enable_hot_buffer=False,
            enable_quality_bridge=False,
            enable_enhancements_manager=False,
            start_background_services=False,
        )

        components = initialize_all_enhancements(
            config=config,
            board_type="hex8",
            num_players=2,
        )

        assert components.facade is not None

    def test_non_main_process(self):
        """Test initialize_all_enhancements on non-main process."""
        config = EnhancementConfig(
            start_background_services=True,
        )

        components = initialize_all_enhancements(
            config=config,
            is_main_process=False,
        )

        # Background services should not start on non-main process
        assert components.started is False


class TestSingletonAccessors:
    """Tests for get_facade and reset_facade singleton functions."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_facade()

    def test_get_facade_creates_instance(self):
        """Test get_facade creates a new instance."""
        facade = get_facade()

        assert facade is not None
        assert isinstance(facade, TrainingEnhancementsFacade)

    def test_get_facade_returns_same_instance(self):
        """Test get_facade returns same instance."""
        facade1 = get_facade()
        facade2 = get_facade()

        assert facade1 is facade2

    def test_get_facade_ignores_config_after_creation(self):
        """Test get_facade ignores config after initial creation."""
        config1 = FacadeConfig(enable_hard_mining=True)
        facade1 = get_facade(config1)

        config2 = FacadeConfig(enable_hard_mining=False)
        facade2 = get_facade(config2)

        # Should still have hard mining enabled
        assert facade2.config.enable_hard_mining is True

    def test_reset_facade(self):
        """Test reset_facade clears singleton."""
        facade1 = get_facade()
        reset_facade()
        facade2 = get_facade()

        # Should be different instances
        assert facade1 is not facade2

    def test_reset_facade_calls_reset(self):
        """Test reset_facade resets internal state."""
        facade = get_facade()
        facade.record_batch([0, 1, 2], torch.tensor([1.0, 2.0, 3.0]))

        reset_facade()

        new_facade = get_facade()
        assert new_facade._total_samples_seen == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_batch(self):
        """Test handling of empty batches."""
        facade = TrainingEnhancementsFacade()

        empty_indices = []
        empty_losses = torch.tensor([])

        # Should not raise
        facade.record_batch(empty_indices, empty_losses)

    def test_large_batch(self):
        """Test handling of large batches."""
        facade = TrainingEnhancementsFacade()

        batch_size = 10000
        indices = list(range(batch_size))
        losses = torch.rand(batch_size)

        facade.record_batch(indices, losses)

        assert facade._total_samples_seen == batch_size

    def test_many_epochs(self):
        """Test many epoch transitions."""
        facade = TrainingEnhancementsFacade()

        for epoch in range(100):
            facade.record_batch([0], torch.tensor([1.0]))
            facade.on_epoch_end()

        assert facade._epoch == 100

    def test_curriculum_lr_clamps_progress(self):
        """Test curriculum LR clamps progress to [0, 1]."""
        facade = TrainingEnhancementsFacade()

        # Negative progress
        scale_neg = facade.get_curriculum_lr_scale(-0.5)
        scale_zero = facade.get_curriculum_lr_scale(0.0)
        assert scale_neg == scale_zero

        # Progress > 1
        scale_over = facade.get_curriculum_lr_scale(1.5)
        scale_one = facade.get_curriculum_lr_scale(1.0)
        assert scale_over == scale_one

    def test_freshness_weight_very_old(self):
        """Test freshness weight for very old games."""
        facade = TrainingEnhancementsFacade()

        now = time.time()
        year_ago = now - 365 * 24 * 3600

        weight = facade.compute_freshness_weight(year_ago, now)

        # Should be very close to 0
        assert weight < 0.01

    def test_numpy_array_indices(self):
        """Test record_batch accepts numpy array indices."""
        facade = TrainingEnhancementsFacade()

        indices = np.array([0, 1, 2, 3])
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])

        facade.record_batch(indices, losses)

        assert facade._total_samples_seen == 4
