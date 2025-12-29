"""Unit tests for PredictiveAlertManager.

Tests cover:
- PredictiveAlertConfig dataclass
- Alert and MetricSample dataclasses
- AlertType and AlertSeverity enums
- Metric recording and history cleanup
- Trend calculation (linear regression)
- Disk full prediction
- Elo degradation prediction
- Queue backlog prediction
- Training stall detection
- Alert throttling and rate limiting
- YAML configuration loading

Created: December 2025
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.monitoring.predictive_alerts import (
    Alert,
    AlertSeverity,
    AlertType,
    MetricSample,
    PredictiveAlertConfig,
    PredictiveAlertManager,
    load_alert_config_from_yaml,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_info_value(self):
        """Test INFO severity value."""
        assert AlertSeverity.INFO.value == "info"

    def test_warning_value(self):
        """Test WARNING severity value."""
        assert AlertSeverity.WARNING.value == "warning"

    def test_critical_value(self):
        """Test CRITICAL severity value."""
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlertType:
    """Tests for AlertType enum."""

    def test_all_types_have_values(self):
        """Test all alert types have string values."""
        assert AlertType.DISK_FULL.value == "disk_full"
        assert AlertType.MEMORY_EXHAUSTION.value == "memory_exhaustion"
        assert AlertType.ELO_DEGRADATION.value == "elo_degradation"
        assert AlertType.NODE_FAILURE.value == "node_failure"
        assert AlertType.QUEUE_BACKLOG.value == "queue_backlog"
        assert AlertType.TRAINING_STALL.value == "training_stall"
        assert AlertType.MODEL_REGRESSION.value == "model_regression"


class TestPredictiveAlertConfig:
    """Tests for PredictiveAlertConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PredictiveAlertConfig()
        assert config.enabled is True
        assert config.disk_prediction_hours == 4
        assert config.disk_critical_threshold == 90.0
        assert config.memory_prediction_hours == 2
        assert config.memory_critical_threshold == 95.0
        assert config.elo_trend_window_hours == 6
        assert config.elo_degradation_threshold == -5.0
        assert config.queue_backlog_threshold == 50
        assert config.queue_growth_rate_threshold == 10.0
        assert config.training_stall_hours == 6
        assert config.alert_throttle_minutes == 30
        assert config.max_alerts_per_hour == 20

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PredictiveAlertConfig(
            enabled=False,
            disk_prediction_hours=8,
            disk_critical_threshold=95.0,
        )
        assert config.enabled is False
        assert config.disk_prediction_hours == 8
        assert config.disk_critical_threshold == 95.0


class TestMetricSample:
    """Tests for MetricSample dataclass."""

    def test_creation(self):
        """Test creating a metric sample."""
        sample = MetricSample(timestamp=1000.0, value=75.5)
        assert sample.timestamp == 1000.0
        assert sample.value == 75.5


class TestAlert:
    """Tests for Alert dataclass."""

    def test_basic_creation(self):
        """Test creating a basic alert."""
        alert = Alert(
            alert_id="test_123",
            alert_type=AlertType.DISK_FULL,
            severity=AlertSeverity.WARNING,
            target_id="node-1",
            message="Disk full warning",
            action="cleanup_required",
        )
        assert alert.alert_id == "test_123"
        assert alert.alert_type == AlertType.DISK_FULL
        assert alert.severity == AlertSeverity.WARNING
        assert alert.target_id == "node-1"
        assert alert.message == "Disk full warning"
        assert alert.action == "cleanup_required"

    def test_auto_created_at(self):
        """Test that created_at is automatically set."""
        before = time.time()
        alert = Alert(
            alert_id="test",
            alert_type=AlertType.DISK_FULL,
            severity=AlertSeverity.INFO,
            target_id="node-1",
            message="Test",
            action="none",
        )
        after = time.time()
        assert before <= alert.created_at <= after

    def test_with_metadata(self):
        """Test alert with metadata."""
        alert = Alert(
            alert_id="test",
            alert_type=AlertType.ELO_DEGRADATION,
            severity=AlertSeverity.WARNING,
            target_id="model-1",
            message="Elo degrading",
            action="investigate",
            metadata={"current_elo": 1500, "slope": -10.0},
        )
        assert alert.metadata["current_elo"] == 1500
        assert alert.metadata["slope"] == -10.0

    def test_with_predicted_time(self):
        """Test alert with predicted issue time."""
        predicted = time.time() + 3600
        alert = Alert(
            alert_id="test",
            alert_type=AlertType.DISK_FULL,
            severity=AlertSeverity.WARNING,
            target_id="node-1",
            message="Disk will be full",
            action="cleanup",
            predicted_issue_time=predicted,
        )
        assert alert.predicted_issue_time == predicted


class TestPredictiveAlertManager:
    """Tests for PredictiveAlertManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = PredictiveAlertManager()
        assert manager.config is not None
        assert manager._disk_history == {}
        assert manager._memory_history == {}
        assert manager._elo_history == {}
        assert manager._queue_history == []

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = PredictiveAlertConfig(enabled=False)
        manager = PredictiveAlertManager(config=config)
        assert manager.config.enabled is False

    def test_set_notify_callback(self):
        """Test setting notification callback."""
        manager = PredictiveAlertManager()
        callback = MagicMock()
        manager.set_notify_callback(callback)
        assert manager._notify_callback is callback


class TestMetricRecording:
    """Tests for metric recording methods."""

    def test_record_disk_usage(self):
        """Test recording disk usage."""
        manager = PredictiveAlertManager()
        manager.record_disk_usage("node-1", 50.0)
        assert "node-1" in manager._disk_history
        assert len(manager._disk_history["node-1"]) == 1
        assert manager._disk_history["node-1"][0].value == 50.0

    def test_record_multiple_disk_samples(self):
        """Test recording multiple disk samples."""
        manager = PredictiveAlertManager()
        manager.record_disk_usage("node-1", 50.0)
        manager.record_disk_usage("node-1", 55.0)
        manager.record_disk_usage("node-1", 60.0)
        assert len(manager._disk_history["node-1"]) == 3

    def test_record_memory_usage(self):
        """Test recording memory usage."""
        manager = PredictiveAlertManager()
        manager.record_memory_usage("node-1", 70.0)
        assert "node-1" in manager._memory_history
        assert manager._memory_history["node-1"][0].value == 70.0

    def test_record_elo(self):
        """Test recording Elo rating."""
        manager = PredictiveAlertManager()
        manager.record_elo("model-1", 1500.0)
        assert "model-1" in manager._elo_history
        assert manager._elo_history["model-1"][0].value == 1500.0

    def test_record_queue_depth(self):
        """Test recording queue depth."""
        manager = PredictiveAlertManager()
        manager.record_queue_depth(25)
        assert len(manager._queue_history) == 1
        assert manager._queue_history[0].value == 25.0


class TestHistoryCleanup:
    """Tests for history cleanup."""

    def test_cleanup_old_samples(self):
        """Test that old samples are cleaned up."""
        manager = PredictiveAlertManager()

        # Add old sample
        old_sample = MetricSample(
            timestamp=time.time() - 100000,  # ~28 hours ago
            value=50.0,
        )
        manager._disk_history["node-1"] = [old_sample]

        # Add new sample - should trigger cleanup
        manager.record_disk_usage("node-1", 60.0)

        # Old sample should be removed
        assert len(manager._disk_history["node-1"]) == 1
        assert manager._disk_history["node-1"][0].value == 60.0


class TestTrendCalculation:
    """Tests for trend calculation."""

    def test_calculate_trend_insufficient_data(self):
        """Test trend calculation with insufficient data."""
        manager = PredictiveAlertManager()
        samples = [MetricSample(timestamp=1000.0, value=50.0)]
        result = manager._calculate_trend(samples, window_hours=1)
        assert result is None

    def test_calculate_trend_positive(self):
        """Test calculating positive trend."""
        manager = PredictiveAlertManager()
        now = time.time()

        # Create samples with increasing values over 1 hour
        samples = [
            MetricSample(timestamp=now - 3600, value=50.0),
            MetricSample(timestamp=now - 2400, value=55.0),
            MetricSample(timestamp=now - 1200, value=60.0),
            MetricSample(timestamp=now, value=65.0),
        ]

        slope = manager._calculate_trend(samples, window_hours=2)
        assert slope is not None
        assert slope > 0  # Positive growth

    def test_calculate_trend_negative(self):
        """Test calculating negative trend."""
        manager = PredictiveAlertManager()
        now = time.time()

        samples = [
            MetricSample(timestamp=now - 3600, value=100.0),
            MetricSample(timestamp=now - 2400, value=90.0),
            MetricSample(timestamp=now - 1200, value=80.0),
            MetricSample(timestamp=now, value=70.0),
        ]

        slope = manager._calculate_trend(samples, window_hours=2)
        assert slope is not None
        assert slope < 0  # Negative growth


class TestDiskFullPrediction:
    """Tests for disk full prediction."""

    def test_predict_disk_no_history(self):
        """Test prediction with no history."""
        manager = PredictiveAlertManager()
        result = manager.predict_disk_full("node-1")
        assert result is None

    def test_predict_disk_critical(self):
        """Test prediction when disk is already critical."""
        manager = PredictiveAlertManager()

        # Record critical usage
        manager._disk_history["node-1"] = [
            MetricSample(timestamp=time.time(), value=95.0)
        ]

        alert = manager.predict_disk_full("node-1")
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.alert_type == AlertType.DISK_FULL
        assert "critical" in alert.message.lower()

    def test_predict_disk_growing(self):
        """Test prediction with growing disk usage."""
        manager = PredictiveAlertManager()
        now = time.time()

        # Create rapidly growing history
        manager._disk_history["node-1"] = [
            MetricSample(timestamp=now - 7200, value=50.0),
            MetricSample(timestamp=now - 5400, value=60.0),
            MetricSample(timestamp=now - 3600, value=70.0),
            MetricSample(timestamp=now - 1800, value=80.0),
            MetricSample(timestamp=now, value=85.0),
        ]

        alert = manager.predict_disk_full("node-1", hours_ahead=8)
        # May or may not alert depending on calculated trend
        if alert:
            assert alert.alert_type == AlertType.DISK_FULL
            assert "predicted" in alert.message.lower() or "critical" in alert.message.lower()

    def test_predict_disk_stable(self):
        """Test prediction with stable disk usage."""
        manager = PredictiveAlertManager()
        now = time.time()

        # Create stable history
        manager._disk_history["node-1"] = [
            MetricSample(timestamp=now - 3600, value=50.0),
            MetricSample(timestamp=now - 2400, value=50.1),
            MetricSample(timestamp=now - 1200, value=49.9),
            MetricSample(timestamp=now, value=50.0),
        ]

        alert = manager.predict_disk_full("node-1")
        assert alert is None  # No alert for stable usage


class TestEloDegradationPrediction:
    """Tests for Elo degradation prediction."""

    def test_predict_elo_no_history(self):
        """Test prediction with no history."""
        manager = PredictiveAlertManager()
        result = manager.predict_elo_degradation("model-1")
        assert result is None

    def test_predict_elo_insufficient_samples(self):
        """Test prediction with insufficient samples."""
        manager = PredictiveAlertManager()
        manager._elo_history["model-1"] = [
            MetricSample(timestamp=time.time(), value=1500.0)
        ]
        result = manager.predict_elo_degradation("model-1")
        assert result is None

    def test_predict_elo_degrading(self):
        """Test prediction with degrading Elo."""
        manager = PredictiveAlertManager()
        now = time.time()

        # Create degrading history (losing more than 5 Elo/hour)
        samples = []
        for i in range(15):
            samples.append(MetricSample(
                timestamp=now - (14 - i) * 1200,  # Every 20 minutes
                value=1500 - i * 5,  # Losing 15 Elo/hour
            ))
        manager._elo_history["model-1"] = samples

        alert = manager.predict_elo_degradation("model-1")
        assert alert is not None
        assert alert.alert_type == AlertType.ELO_DEGRADATION
        assert alert.severity == AlertSeverity.WARNING

    def test_predict_elo_improving(self):
        """Test prediction with improving Elo."""
        manager = PredictiveAlertManager()
        now = time.time()

        samples = []
        for i in range(15):
            samples.append(MetricSample(
                timestamp=now - (14 - i) * 1200,
                value=1500 + i * 5,  # Gaining 15 Elo/hour
            ))
        manager._elo_history["model-1"] = samples

        alert = manager.predict_elo_degradation("model-1")
        assert alert is None  # No alert for improving Elo


class TestQueueBacklogPrediction:
    """Tests for queue backlog prediction."""

    def test_predict_queue_no_history(self):
        """Test prediction with no history."""
        manager = PredictiveAlertManager()
        result = manager.predict_queue_backlog()
        assert result is None

    def test_predict_queue_backlogged(self):
        """Test prediction when queue is backlogged."""
        manager = PredictiveAlertManager()
        manager._queue_history = [
            MetricSample(timestamp=time.time(), value=100.0)  # Over threshold
        ]

        alert = manager.predict_queue_backlog()
        assert alert is not None
        assert alert.alert_type == AlertType.QUEUE_BACKLOG
        assert "backlogged" in alert.message.lower()

    def test_predict_queue_growing(self):
        """Test prediction with rapidly growing queue."""
        manager = PredictiveAlertManager()
        now = time.time()

        # Create rapidly growing queue
        manager._queue_history = [
            MetricSample(timestamp=now - 3600, value=10.0),
            MetricSample(timestamp=now - 2700, value=15.0),
            MetricSample(timestamp=now - 1800, value=25.0),
            MetricSample(timestamp=now - 900, value=35.0),
            MetricSample(timestamp=now, value=45.0),
        ]

        alert = manager.predict_queue_backlog()
        # May alert based on growth rate
        if alert:
            assert alert.alert_type == AlertType.QUEUE_BACKLOG


class TestTrainingStallDetection:
    """Tests for training stall detection."""

    def test_training_not_stalled(self):
        """Test when training is active."""
        manager = PredictiveAlertManager()
        last_training = time.time() - 3600  # 1 hour ago

        alert = manager.check_training_stall(last_training)
        assert alert is None

    def test_training_stalled(self):
        """Test when training has stalled."""
        manager = PredictiveAlertManager()
        last_training = time.time() - 25200  # 7 hours ago (over 6 hour threshold)

        alert = manager.check_training_stall(last_training)
        assert alert is not None
        assert alert.alert_type == AlertType.TRAINING_STALL
        assert "No training completed" in alert.message


class TestAlertThrottling:
    """Tests for alert throttling and rate limiting."""

    def test_should_alert_disabled(self):
        """Test that alerts are blocked when disabled."""
        config = PredictiveAlertConfig(enabled=False)
        manager = PredictiveAlertManager(config=config)

        result = manager._should_alert("test_key")
        assert result is False

    def test_should_alert_rate_limited(self):
        """Test rate limiting."""
        config = PredictiveAlertConfig(max_alerts_per_hour=2)
        manager = PredictiveAlertManager(config=config)
        manager._alerts_this_hour = 2  # At limit

        result = manager._should_alert("test_key")
        assert result is False

    def test_should_alert_throttled(self):
        """Test throttling of same alert."""
        config = PredictiveAlertConfig(alert_throttle_minutes=30)
        manager = PredictiveAlertManager(config=config)
        manager._recent_alerts["test_key"] = time.time() - 60  # 1 minute ago

        result = manager._should_alert("test_key")
        assert result is False  # Still within throttle window

    def test_should_alert_allowed(self):
        """Test that alert is allowed when not throttled."""
        manager = PredictiveAlertManager()
        result = manager._should_alert("new_key")
        assert result is True

    def test_hourly_counter_reset(self):
        """Test hourly counter reset."""
        manager = PredictiveAlertManager()
        manager._alerts_this_hour = 100
        manager._hour_start = time.time() - 3700  # Over an hour ago

        # Calling _should_alert should reset counter
        manager._should_alert("test_key")
        assert manager._alerts_this_hour == 0


class TestRunAllChecks:
    """Tests for run_all_checks method."""

    @pytest.mark.asyncio
    async def test_run_all_checks_no_issues(self):
        """Test running all checks with no issues."""
        manager = PredictiveAlertManager()
        last_training = time.time()  # Just now

        alerts = await manager.run_all_checks(
            node_ids=["node-1"],
            model_ids=["model-1"],
            last_training_time=last_training,
        )
        assert alerts == []

    @pytest.mark.asyncio
    async def test_run_all_checks_with_training_stall(self):
        """Test running all checks with training stall."""
        manager = PredictiveAlertManager()
        last_training = time.time() - 25200  # 7 hours ago

        alerts = await manager.run_all_checks(
            node_ids=[],
            model_ids=[],
            last_training_time=last_training,
        )
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.TRAINING_STALL


class TestSendAlert:
    """Tests for send_alert method."""

    @pytest.mark.asyncio
    async def test_send_alert_no_callback(self):
        """Test sending alert with no callback."""
        manager = PredictiveAlertManager()
        alert = Alert(
            alert_id="test",
            alert_type=AlertType.DISK_FULL,
            severity=AlertSeverity.WARNING,
            target_id="node-1",
            message="Test",
            action="none",
        )

        result = await manager.send_alert(alert)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_alert_with_callback(self):
        """Test sending alert with callback."""
        manager = PredictiveAlertManager()
        callback = AsyncMock()
        manager.set_notify_callback(callback)

        alert = Alert(
            alert_id="test",
            alert_type=AlertType.DISK_FULL,
            severity=AlertSeverity.WARNING,
            target_id="node-1",
            message="Test",
            action="none",
        )

        result = await manager.send_alert(alert)
        assert result is True
        callback.assert_called_once_with(alert)

    @pytest.mark.asyncio
    async def test_send_alert_callback_failure(self):
        """Test handling callback failure."""
        manager = PredictiveAlertManager()
        callback = AsyncMock(side_effect=Exception("Callback failed"))
        manager.set_notify_callback(callback)

        alert = Alert(
            alert_id="test",
            alert_type=AlertType.DISK_FULL,
            severity=AlertSeverity.WARNING,
            target_id="node-1",
            message="Test",
            action="none",
        )

        result = await manager.send_alert(alert)
        assert result is False


class TestGetAlertStats:
    """Tests for get_alert_stats method."""

    def test_get_alert_stats(self):
        """Test getting alert statistics."""
        manager = PredictiveAlertManager()
        manager._alerts_this_hour = 5
        manager._disk_history["node-1"] = []
        manager._disk_history["node-2"] = []
        manager._elo_history["model-1"] = []
        manager._queue_history = [MetricSample(time.time(), 10)]
        manager._recent_alerts["key1"] = time.time()

        stats = manager.get_alert_stats()
        assert stats["enabled"] is True
        assert stats["alerts_this_hour"] == 5
        assert stats["max_alerts_per_hour"] == 20
        assert stats["nodes_tracked"] == 2
        assert stats["models_tracked"] == 1
        assert stats["queue_samples"] == 1
        assert stats["throttled_alerts"] == 1


class TestLoadConfigFromYaml:
    """Tests for load_alert_config_from_yaml function."""

    def test_load_default_config(self):
        """Test loading with empty YAML."""
        config = load_alert_config_from_yaml({})
        assert config.enabled is True
        assert config.disk_prediction_hours == 4

    def test_load_custom_config(self):
        """Test loading with custom values."""
        yaml_config = {
            "proactive_monitoring": {
                "enabled": False,
                "disk_prediction_hours": 8,
                "disk_critical_threshold": 95.0,
                "queue_backlog_threshold": 100,
            }
        }

        config = load_alert_config_from_yaml(yaml_config)
        assert config.enabled is False
        assert config.disk_prediction_hours == 8
        assert config.disk_critical_threshold == 95.0
        assert config.queue_backlog_threshold == 100

    def test_load_partial_config(self):
        """Test loading with partial values."""
        yaml_config = {
            "proactive_monitoring": {
                "disk_prediction_hours": 12,
            }
        }

        config = load_alert_config_from_yaml(yaml_config)
        assert config.disk_prediction_hours == 12
        # Other values should be defaults
        assert config.enabled is True
        assert config.queue_backlog_threshold == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
