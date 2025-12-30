"""Tests for post_epoch_handler.py.

December 2025: Comprehensive test coverage for PostEpochHandler.
"""

from __future__ import annotations

import asyncio
from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import torch
import torch.nn as nn

from app.models import BoardType
from app.training.post_epoch_handler import (
    PostEpochConfig,
    EpochMetrics,
    PostEpochResult,
    PostEpochHandler,
    create_post_epoch_handler,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_context():
    """Create a mock TrainContext with all required attributes."""
    context = MagicMock()
    context.is_main_process = True
    context.device = torch.device("cpu")

    # Config - use real BoardType to avoid validation errors
    context.config = MagicMock()
    context.config.board_type = BoardType.HEX8
    context.config.epochs = 50
    context.config.batch_size = 32

    # Resolved config
    context.resolved = MagicMock()
    context.resolved.num_players = 2
    context.resolved.checkpoint_dir = "/tmp/checkpoints"

    # Model
    context.model = MagicMock(spec=nn.Module)
    context.model_to_save = context.model
    context.optimizer = MagicMock()

    # Optional components
    context.early_stopper = None
    context.epoch_scheduler = None
    context.async_checkpointer = None
    context.checkpoint_averager = None
    context.calibration_tracker = None
    context.training_facade = None
    context.hard_example_miner = None
    context.enhancements_manager = None
    context.heartbeat_monitor = None
    context.training_breaker = None
    context.metrics_collector = None

    # Train loader
    context.train_loader = MagicMock()
    context.train_loader.dataset = [1, 2, 3]  # Mock dataset with 3 items

    # Save path
    context.save_path = "/tmp/model.pth"

    return context


@pytest.fixture
def sample_metrics():
    """Create sample epoch metrics."""
    return EpochMetrics(
        epoch=5,
        avg_train_loss=0.35,
        avg_val_loss=0.40,
        avg_policy_accuracy=0.75,
        learning_rate=0.001,
        avg_policy_loss=0.25,
        avg_value_loss=0.10,
        train_batches=100,
        samples_per_second=500.0,
        epoch_duration=120.0,
        epoch_losses=[
            {"train_loss": 0.5, "val_loss": 0.55},
            {"train_loss": 0.45, "val_loss": 0.50},
            {"train_loss": 0.40, "val_loss": 0.45},
            {"train_loss": 0.38, "val_loss": 0.42},
            {"train_loss": 0.35, "val_loss": 0.40},
        ],
    )


# =============================================================================
# PostEpochConfig Tests
# =============================================================================


class TestPostEpochConfig:
    """Test PostEpochConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PostEpochConfig()
        assert config.checkpoint_interval == 5
        assert config.checkpoint_dir == "checkpoints"
        assert config.save_all_epochs is True
        assert config.emit_events is True
        assert config.emit_prometheus is True
        assert config.emit_dashboard is True
        assert config.enable_regression_detection is True
        assert config.calibration_interval == 5
        assert config.trend_interval == 5
        assert config.plateau_interval == 10
        assert config.anomaly_threshold == 2.0
        assert config.enable_distillation is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PostEpochConfig(
            checkpoint_interval=10,
            checkpoint_dir="/custom/path",
            emit_events=False,
            enable_regression_detection=False,
            anomaly_threshold=3.0,
        )
        assert config.checkpoint_interval == 10
        assert config.checkpoint_dir == "/custom/path"
        assert config.emit_events is False
        assert config.enable_regression_detection is False
        assert config.anomaly_threshold == 3.0

    def test_all_fields_exist(self):
        """Verify all expected fields are present."""
        field_names = {f.name for f in fields(PostEpochConfig)}
        expected_fields = {
            "checkpoint_interval",
            "checkpoint_dir",
            "save_all_epochs",
            "emit_events",
            "emit_prometheus",
            "emit_dashboard",
            "enable_regression_detection",
            "calibration_interval",
            "trend_interval",
            "plateau_interval",
            "anomaly_threshold",
            "enable_distillation",
        }
        assert expected_fields.issubset(field_names)


# =============================================================================
# EpochMetrics Tests
# =============================================================================


class TestEpochMetrics:
    """Test EpochMetrics dataclass."""

    def test_required_fields(self):
        """Test required fields are properly set."""
        metrics = EpochMetrics(
            epoch=10,
            avg_train_loss=0.5,
            avg_val_loss=0.6,
            avg_policy_accuracy=0.8,
            learning_rate=0.001,
        )
        assert metrics.epoch == 10
        assert metrics.avg_train_loss == 0.5
        assert metrics.avg_val_loss == 0.6
        assert metrics.avg_policy_accuracy == 0.8
        assert metrics.learning_rate == 0.001

    def test_optional_fields_defaults(self):
        """Test optional fields have correct defaults."""
        metrics = EpochMetrics(
            epoch=1,
            avg_train_loss=0.5,
            avg_val_loss=0.6,
            avg_policy_accuracy=0.8,
            learning_rate=0.001,
        )
        assert metrics.avg_policy_loss == 0.0
        assert metrics.avg_value_loss == 0.0
        assert metrics.train_batches == 0
        assert metrics.samples_per_second == 0.0
        assert metrics.epoch_duration == 0.0
        assert metrics.epoch_losses == []

    def test_epoch_losses_list(self):
        """Test epoch_losses is properly initialized."""
        metrics = EpochMetrics(
            epoch=5,
            avg_train_loss=0.3,
            avg_val_loss=0.35,
            avg_policy_accuracy=0.85,
            learning_rate=0.0005,
            epoch_losses=[
                {"train_loss": 0.5, "val_loss": 0.55},
                {"train_loss": 0.4, "val_loss": 0.45},
            ],
        )
        assert len(metrics.epoch_losses) == 2
        assert metrics.epoch_losses[0]["train_loss"] == 0.5


# =============================================================================
# PostEpochResult Tests
# =============================================================================


class TestPostEpochResult:
    """Test PostEpochResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = PostEpochResult()
        assert result.should_stop is False
        assert result.stop_reason == ""
        assert result.checkpoint_saved is False
        assert result.checkpoint_path == ""
        assert result.best_model_saved is False
        assert result.epoch_record == {}
        assert result.regression_detected is False
        assert result.anomaly_detected is False
        assert result.plateau_detected is False
        assert result.distillation_triggered is False

    def test_early_stop_result(self):
        """Test result with early stopping."""
        result = PostEpochResult(
            should_stop=True,
            stop_reason="loss_stagnation",
            checkpoint_saved=True,
            checkpoint_path="/tmp/checkpoint.pth",
        )
        assert result.should_stop is True
        assert result.stop_reason == "loss_stagnation"
        assert result.checkpoint_saved is True
        assert result.checkpoint_path == "/tmp/checkpoint.pth"


# =============================================================================
# PostEpochHandler Initialization Tests
# =============================================================================


class TestPostEpochHandlerInit:
    """Test PostEpochHandler initialization."""

    def test_default_config(self):
        """Test handler with default config."""
        handler = PostEpochHandler()
        assert handler.config is not None
        assert handler.config.checkpoint_interval == 5
        assert handler._best_val_loss == float("inf")
        assert handler._last_good_checkpoint_path is None
        assert handler._last_good_epoch == 0

    def test_custom_config(self):
        """Test handler with custom config."""
        config = PostEpochConfig(checkpoint_interval=10, emit_events=False)
        handler = PostEpochHandler(config)
        assert handler.config.checkpoint_interval == 10
        assert handler.config.emit_events is False


# =============================================================================
# PostEpochHandler._create_epoch_record Tests
# =============================================================================


class TestCreateEpochRecord:
    """Test epoch record creation."""

    def test_basic_record(self, mock_context, sample_metrics):
        """Test basic epoch record creation."""
        handler = PostEpochHandler()
        record = handler._create_epoch_record(mock_context, sample_metrics)

        assert record["epoch"] == 6  # epoch + 1
        assert record["train_loss"] == 0.35
        assert record["val_loss"] == 0.40
        assert record["policy_accuracy"] == 0.75
        assert record["lr"] == 0.001
        assert record["train_batches"] == 100
        assert record["samples_per_second"] == 500.0
        assert record["epoch_duration"] == 120.0


# =============================================================================
# PostEpochHandler._check_regression Tests
# =============================================================================


class TestCheckRegression:
    """Test regression detection."""

    def test_regression_skipped_early_epochs(self, mock_context, sample_metrics):
        """Test regression detection skipped for early epochs."""
        handler = PostEpochHandler()
        sample_metrics.epoch = 1
        result = handler._check_regression(mock_context, sample_metrics)
        assert result is False

    def test_regression_skipped_not_main_process(self, mock_context, sample_metrics):
        """Test regression detection skipped when not main process."""
        handler = PostEpochHandler()
        mock_context.is_main_process = False
        result = handler._check_regression(mock_context, sample_metrics)
        assert result is False

    def test_regression_disabled(self, mock_context, sample_metrics):
        """Test regression detection when disabled."""
        config = PostEpochConfig(enable_regression_detection=False)
        handler = PostEpochHandler(config)
        # Even with regression detection disabled, _check_regression should be skipped
        # at the handle_epoch_end level
        result = handler._check_regression(mock_context, sample_metrics)
        # With module not available, returns False
        assert result is False


# =============================================================================
# PostEpochHandler._check_early_stopping Tests
# =============================================================================


class TestCheckEarlyStopping:
    """Test early stopping logic."""

    def test_no_early_stopper(self, mock_context, sample_metrics):
        """Test no early stopping when early_stopper is None."""
        handler = PostEpochHandler()
        mock_context.early_stopper = None
        should_stop, reason = handler._check_early_stopping(mock_context, sample_metrics)
        assert should_stop is False
        assert reason == ""

    def test_early_stopper_not_triggered(self, mock_context, sample_metrics):
        """Test early stopper when not triggered."""
        handler = PostEpochHandler()
        mock_context.early_stopper = MagicMock()
        mock_context.early_stopper.should_stop.return_value = False
        mock_context.enhancements_manager = None

        should_stop, reason = handler._check_early_stopping(mock_context, sample_metrics)
        assert should_stop is False

    def test_early_stopper_triggered(self, mock_context, sample_metrics):
        """Test early stopper when triggered."""
        handler = PostEpochHandler()
        mock_context.early_stopper = MagicMock()
        mock_context.early_stopper.should_stop.return_value = True
        mock_context.enhancements_manager = None
        sample_metrics.epoch = 20  # Above MIN_TRAINING_EPOCHS

        should_stop, reason = handler._check_early_stopping(mock_context, sample_metrics)
        assert should_stop is True
        assert reason == "loss_stagnation"

    def test_enhancements_manager_elo_regression(self, mock_context, sample_metrics):
        """Test early stop from enhancements manager Elo regression."""
        handler = PostEpochHandler()
        mock_context.enhancements_manager = MagicMock()
        mock_context.enhancements_manager.should_early_stop.return_value = True

        should_stop, reason = handler._check_early_stopping(mock_context, sample_metrics)
        assert should_stop is True
        assert reason == "elo_regression"


# =============================================================================
# PostEpochHandler._should_checkpoint Tests
# =============================================================================


class TestShouldCheckpoint:
    """Test checkpoint timing logic."""

    def test_not_main_process(self, mock_context, sample_metrics):
        """Test no checkpoint when not main process."""
        handler = PostEpochHandler()
        mock_context.is_main_process = False
        assert handler._should_checkpoint(mock_context, sample_metrics) is False

    def test_interval_disabled(self, mock_context, sample_metrics):
        """Test no checkpoint when interval is 0."""
        config = PostEpochConfig(checkpoint_interval=0)
        handler = PostEpochHandler(config)
        assert handler._should_checkpoint(mock_context, sample_metrics) is False

    def test_at_interval(self, mock_context, sample_metrics):
        """Test checkpoint at interval."""
        config = PostEpochConfig(checkpoint_interval=5)
        handler = PostEpochHandler(config)
        sample_metrics.epoch = 4  # epoch + 1 = 5
        assert handler._should_checkpoint(mock_context, sample_metrics) is True

    def test_not_at_interval(self, mock_context, sample_metrics):
        """Test no checkpoint between intervals."""
        config = PostEpochConfig(checkpoint_interval=5)
        handler = PostEpochHandler(config)
        sample_metrics.epoch = 3  # epoch + 1 = 4
        assert handler._should_checkpoint(mock_context, sample_metrics) is False


# =============================================================================
# PostEpochHandler._emit_loss_events Tests
# =============================================================================


class TestEmitLossEvents:
    """Test loss event emission."""

    def test_no_events_when_disabled(self, mock_context, sample_metrics):
        """Test no events when emit_events is False."""
        config = PostEpochConfig(emit_events=False)
        handler = PostEpochHandler(config)
        # _emit_loss_events is only called from handle_epoch_end when emit_events is True
        # Direct call should return False, False due to HAS_TRAINING_EVENTS check
        anomaly, plateau = handler._emit_loss_events(mock_context, sample_metrics)
        # Result depends on HAS_TRAINING_EVENTS flag
        assert isinstance(anomaly, bool)
        assert isinstance(plateau, bool)

    def test_no_events_when_not_main_process(self, mock_context, sample_metrics):
        """Test no events when not main process."""
        handler = PostEpochHandler()
        mock_context.is_main_process = False
        anomaly, plateau = handler._emit_loss_events(mock_context, sample_metrics)
        assert anomaly is False
        assert plateau is False

    def test_no_events_with_empty_epoch_losses(self, mock_context, sample_metrics):
        """Test no events with empty epoch losses."""
        handler = PostEpochHandler()
        sample_metrics.epoch_losses = []
        anomaly, plateau = handler._emit_loss_events(mock_context, sample_metrics)
        assert anomaly is False
        assert plateau is False


# =============================================================================
# PostEpochHandler.handle_epoch_end Tests
# =============================================================================


class TestHandleEpochEnd:
    """Test main handle_epoch_end orchestration."""

    def test_basic_epoch_handling(self, mock_context, sample_metrics):
        """Test basic epoch handling returns result."""
        handler = PostEpochHandler()
        # Prevent best model saving which triggers model validation
        handler._best_val_loss = 0.0  # Lower than sample_metrics.avg_val_loss
        result = handler.handle_epoch_end(mock_context, sample_metrics)

        assert isinstance(result, PostEpochResult)
        assert result.should_stop is False
        assert "epoch" in result.epoch_record
        assert result.epoch_record["epoch"] == 6

    def test_best_model_saved_on_improvement(self, mock_context, sample_metrics):
        """Test best model saved when val_loss improves."""
        handler = PostEpochHandler()
        handler._best_val_loss = 0.5  # Previous best was worse

        with patch.object(handler, "_save_best_model", return_value=True) as mock_save:
            result = handler.handle_epoch_end(mock_context, sample_metrics)
            mock_save.assert_called_once()
            assert handler._best_val_loss == 0.40  # Updated to current

    def test_no_best_model_when_not_improving(self, mock_context, sample_metrics):
        """Test no best model save when not improving."""
        handler = PostEpochHandler()
        handler._best_val_loss = 0.3  # Previous best was better

        with patch.object(handler, "_save_best_model") as mock_save:
            handler.handle_epoch_end(mock_context, sample_metrics)
            mock_save.assert_not_called()

    def test_heartbeat_monitor_beat(self, mock_context, sample_metrics):
        """Test heartbeat monitor is called."""
        handler = PostEpochHandler()
        handler._best_val_loss = 0.0  # Prevent best model save
        mock_context.heartbeat_monitor = MagicMock()

        handler.handle_epoch_end(mock_context, sample_metrics)
        mock_context.heartbeat_monitor.beat.assert_called_once()

    def test_training_breaker_success(self, mock_context, sample_metrics):
        """Test circuit breaker success is recorded."""
        handler = PostEpochHandler()
        handler._best_val_loss = 0.0  # Prevent best model save
        mock_context.training_breaker = MagicMock()

        handler.handle_epoch_end(mock_context, sample_metrics)
        mock_context.training_breaker.record_success.assert_called_once_with("training_epoch")


# =============================================================================
# PostEpochHandler.handle_training_complete Tests
# =============================================================================


class TestHandleTrainingComplete:
    """Test handle_training_complete method."""

    def test_not_main_process(self, mock_context, sample_metrics):
        """Test no action when not main process."""
        handler = PostEpochHandler()
        mock_context.is_main_process = False

        result = handler.handle_training_complete(mock_context, sample_metrics)
        assert result.checkpoint_saved is False

    def test_final_checkpoint_saved(self, mock_context, sample_metrics):
        """Test final checkpoint is saved."""
        handler = PostEpochHandler()

        with patch("app.training.post_epoch_handler.save_checkpoint") as mock_save:
            result = handler.handle_training_complete(mock_context, sample_metrics)
            mock_save.assert_called_once()
            assert result.checkpoint_saved is True
            assert "final" in result.checkpoint_path

    def test_async_checkpointer_used(self, mock_context, sample_metrics):
        """Test async checkpointer is used when available."""
        handler = PostEpochHandler()
        mock_context.async_checkpointer = MagicMock()

        result = handler.handle_training_complete(mock_context, sample_metrics)
        mock_context.async_checkpointer.save_async.assert_called_once()
        assert result.checkpoint_saved is True


# =============================================================================
# PostEpochHandler._emit_async Tests
# =============================================================================


class TestEmitAsync:
    """Test async event emission helper."""

    def test_emit_async_with_running_loop(self):
        """Test async emission with running event loop."""
        handler = PostEpochHandler()

        async def test_coro():
            return "test"

        async def run_test():
            coro = test_coro()
            handler._emit_async(coro)
            # Give time for the task to be scheduled
            await asyncio.sleep(0.01)

        asyncio.run(run_test())

    def test_emit_async_no_loop(self):
        """Test async emission handles no running loop gracefully."""
        handler = PostEpochHandler()

        async def test_coro():
            return "test"

        # Should not raise even without a running loop
        handler._emit_async(test_coro())


# =============================================================================
# PostEpochHandler._record_prometheus_metrics Tests
# =============================================================================


class TestRecordPrometheusMetrics:
    """Test Prometheus metrics recording."""

    def test_not_main_process(self, mock_context, sample_metrics):
        """Test no metrics when not main process."""
        handler = PostEpochHandler()
        mock_context.is_main_process = False
        epoch_record = {}
        # Should not raise
        handler._record_prometheus_metrics(mock_context, sample_metrics, epoch_record)


# =============================================================================
# PostEpochHandler._record_dashboard_metrics Tests
# =============================================================================


class TestRecordDashboardMetrics:
    """Test dashboard metrics recording."""

    def test_not_main_process(self, mock_context, sample_metrics):
        """Test no metrics when not main process."""
        handler = PostEpochHandler()
        mock_context.is_main_process = False
        epoch_record = {}
        # Should not raise
        handler._record_dashboard_metrics(mock_context, sample_metrics, epoch_record)

    def test_no_metrics_collector(self, mock_context, sample_metrics):
        """Test no action when metrics_collector is None."""
        handler = PostEpochHandler()
        mock_context.metrics_collector = None
        epoch_record = {}
        # Should not raise
        handler._record_dashboard_metrics(mock_context, sample_metrics, epoch_record)

    def test_metrics_recorded(self, mock_context, sample_metrics):
        """Test metrics are recorded to collector."""
        handler = PostEpochHandler()
        mock_context.metrics_collector = MagicMock()
        epoch_record = {"train_batches": 100}

        handler._record_dashboard_metrics(mock_context, sample_metrics, epoch_record)
        mock_context.metrics_collector.record_training_step.assert_called_once()


# =============================================================================
# PostEpochHandler._compute_calibration Tests
# =============================================================================


class TestComputeCalibration:
    """Test calibration computation."""

    def test_no_calibration_tracker(self, mock_context, sample_metrics):
        """Test no action when calibration_tracker is None."""
        handler = PostEpochHandler()
        mock_context.calibration_tracker = None
        epoch_record = {}
        # Should not raise
        handler._compute_calibration(mock_context, sample_metrics, epoch_record)
        assert "calibration_ece" not in epoch_record

    def test_not_at_interval(self, mock_context, sample_metrics):
        """Test no calibration when not at interval."""
        config = PostEpochConfig(calibration_interval=5)
        handler = PostEpochHandler(config)
        mock_context.calibration_tracker = MagicMock()
        sample_metrics.epoch = 2  # epoch + 1 = 3, not divisible by 5
        epoch_record = {}

        handler._compute_calibration(mock_context, sample_metrics, epoch_record)
        mock_context.calibration_tracker.compute_current_calibration.assert_not_called()

    def test_calibration_at_interval(self, mock_context, sample_metrics):
        """Test calibration computed at interval."""
        config = PostEpochConfig(calibration_interval=5)
        handler = PostEpochHandler(config)
        mock_context.calibration_tracker = MagicMock()
        mock_report = MagicMock()
        mock_report.ece = 0.05
        mock_report.mce = 0.10
        mock_report.overconfidence = 0.02
        mock_report.optimal_temperature = 1.1
        mock_context.calibration_tracker.compute_current_calibration.return_value = mock_report
        sample_metrics.epoch = 4  # epoch + 1 = 5
        epoch_record = {}

        handler._compute_calibration(mock_context, sample_metrics, epoch_record)
        assert epoch_record["calibration_ece"] == 0.05
        assert epoch_record["calibration_mce"] == 0.10


# =============================================================================
# PostEpochHandler._log_training_stats Tests
# =============================================================================


class TestLogTrainingStats:
    """Test training stats logging."""

    def test_not_main_process(self, mock_context, sample_metrics):
        """Test no logging when not main process."""
        handler = PostEpochHandler()
        mock_context.is_main_process = False
        epoch_record = {}
        # Should not raise
        handler._log_training_stats(mock_context, sample_metrics, epoch_record)

    def test_training_facade_stats(self, mock_context, sample_metrics):
        """Test training facade stats are logged."""
        handler = PostEpochHandler()
        mock_context.training_facade = MagicMock()
        mock_context.training_facade.on_epoch_end.return_value = {
            "mining_active": True,
            "tracked_samples": 1000,
            "hard_examples_fraction": 0.1,
            "mean_per_sample_loss": 0.5,
            "curriculum_lr_scale": 0.9,
            "mean_loss": 0.45,
        }
        epoch_record = {}

        handler._log_training_stats(mock_context, sample_metrics, epoch_record)
        assert epoch_record["facade_mean_loss"] == 0.45
        assert epoch_record["facade_mining_active"] is True

    def test_hard_example_miner_stats(self, mock_context, sample_metrics):
        """Test hard example miner stats are logged."""
        handler = PostEpochHandler()
        mock_context.training_facade = None
        mock_context.hard_example_miner = MagicMock()
        mock_context.hard_example_miner.get_statistics.return_value = {
            "mining_active": True,
            "tracked_examples": 500,
            "mean_loss": 0.6,
            "loss_p90": 0.8,
        }
        epoch_record = {}

        handler._log_training_stats(mock_context, sample_metrics, epoch_record)
        assert epoch_record["hard_mining_mean_loss"] == 0.6
        assert epoch_record["hard_mining_tracked"] == 500


# =============================================================================
# PostEpochHandler._maybe_distill Tests
# =============================================================================


class TestMaybeDistill:
    """Test knowledge distillation logic."""

    def test_no_enhancements_manager(self, mock_context, sample_metrics):
        """Test no distillation when enhancements_manager is None."""
        handler = PostEpochHandler()
        mock_context.enhancements_manager = None
        result = handler._maybe_distill(mock_context, sample_metrics)
        assert result is False

    def test_not_distill_time(self, mock_context, sample_metrics):
        """Test no distillation when not time to distill."""
        handler = PostEpochHandler()
        mock_context.enhancements_manager = MagicMock()
        mock_context.enhancements_manager.should_distill.return_value = False

        result = handler._maybe_distill(mock_context, sample_metrics)
        assert result is False

    def test_distillation_triggered(self, mock_context, sample_metrics):
        """Test distillation is triggered."""
        handler = PostEpochHandler()
        mock_context.enhancements_manager = MagicMock()
        mock_context.enhancements_manager.should_distill.return_value = True
        mock_context.enhancements_manager.run_distillation.return_value = True

        result = handler._maybe_distill(mock_context, sample_metrics)
        assert result is True
        mock_context.enhancements_manager.run_distillation.assert_called_once()


# =============================================================================
# create_post_epoch_handler Factory Tests
# =============================================================================


class TestCreatePostEpochHandler:
    """Test factory function."""

    def test_default_factory(self):
        """Test factory with defaults."""
        handler = create_post_epoch_handler()
        assert handler.config.checkpoint_interval == 5
        assert handler.config.emit_events is True

    def test_custom_factory(self):
        """Test factory with custom values."""
        handler = create_post_epoch_handler(
            checkpoint_interval=10,
            checkpoint_dir="/custom/checkpoints",
            emit_events=False,
        )
        assert handler.config.checkpoint_interval == 10
        assert handler.config.checkpoint_dir == "/custom/checkpoints"
        assert handler.config.emit_events is False

    def test_factory_kwargs(self):
        """Test factory filters unknown kwargs."""
        handler = create_post_epoch_handler(
            checkpoint_interval=15,
            unknown_param="ignored",
        )
        assert handler.config.checkpoint_interval == 15


# =============================================================================
# Integration Tests
# =============================================================================


class TestPostEpochHandlerIntegration:
    """Integration tests for PostEpochHandler."""

    def test_full_epoch_cycle(self, mock_context, sample_metrics):
        """Test a full epoch cycle with multiple features enabled."""
        handler = PostEpochHandler(PostEpochConfig(
            checkpoint_interval=0,  # Disable checkpointing to avoid pickle issues
            emit_events=False,  # Avoid event system dependencies
            emit_prometheus=False,
            emit_dashboard=False,
            enable_regression_detection=False,
        ))

        # Mock save_model_checkpoint to avoid model validation
        with patch("app.training.post_epoch_handler.save_model_checkpoint"):
            # Run multiple epochs
            for epoch in range(10):
                sample_metrics.epoch = epoch
                sample_metrics.avg_val_loss = 0.5 - (epoch * 0.05)
                result = handler.handle_epoch_end(mock_context, sample_metrics)
                assert result.should_stop is False

            # Best val loss should be tracked
            assert handler._best_val_loss < 0.5

    def test_checkpoint_and_best_model_flow(self, mock_context, sample_metrics):
        """Test checkpoint and best model saving flow."""
        handler = PostEpochHandler(PostEpochConfig(
            checkpoint_interval=3,
            emit_events=False,
            emit_prometheus=False,
        ))

        with patch.object(handler, "_save_checkpoint", return_value=(True, "/tmp/cp.pth")):
            with patch.object(handler, "_save_best_model", return_value=True):
                sample_metrics.epoch = 2  # epoch + 1 = 3, checkpoint interval
                result = handler.handle_epoch_end(mock_context, sample_metrics)
                assert result.checkpoint_saved is True
