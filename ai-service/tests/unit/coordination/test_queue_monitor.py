"""Tests for Queue Depth Monitor with Backpressure.

Tests the queue monitoring system that tracks queue depths and applies
backpressure to prevent queue overflow.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.coordination.queue_monitor import (
    DEFAULT_QUEUE_CONFIG,
    BackpressureLevel,
    QueueMetric,
    QueueMonitor,
    QueueStatus,
    QueueType,
    check_backpressure,
    get_queue_monitor,
    report_queue_depth,
    should_throttle_production,
)


class TestQueueType:
    """Tests for QueueType enum."""

    def test_all_queue_types_defined(self):
        """All expected queue types should be defined."""
        assert QueueType.TRAINING_DATA.value == "training_data"
        assert QueueType.PENDING_GAMES.value == "pending_games"
        assert QueueType.EVALUATION_QUEUE.value == "evaluation"
        assert QueueType.PROMOTION_QUEUE.value == "promotion"
        assert QueueType.SYNC_QUEUE.value == "sync"
        assert QueueType.EXPORT_QUEUE.value == "export"

    def test_queue_type_count(self):
        """Should have exactly 6 queue types."""
        assert len(QueueType) == 6


class TestBackpressureLevel:
    """Tests for BackpressureLevel enum."""

    def test_all_levels_defined(self):
        """All backpressure levels should be defined."""
        assert BackpressureLevel.NONE.value == "none"
        assert BackpressureLevel.SOFT.value == "soft"
        assert BackpressureLevel.HARD.value == "hard"
        assert BackpressureLevel.STOP.value == "stop"

    def test_level_count(self):
        """Should have exactly 4 levels."""
        assert len(BackpressureLevel) == 4


class TestQueueStatus:
    """Tests for QueueStatus dataclass."""

    def test_create_status(self):
        """Should create QueueStatus with all fields."""
        status = QueueStatus(
            queue_type=QueueType.TRAINING_DATA,
            current_depth=50000,
            soft_limit=100000,
            hard_limit=500000,
            target_depth=50000,
            backpressure=BackpressureLevel.NONE,
            last_updated=time.time(),
            trend="stable",
        )
        assert status.current_depth == 50000
        assert status.backpressure == BackpressureLevel.NONE

    def test_to_dict(self):
        """Should serialize to dictionary correctly."""
        now = time.time()
        status = QueueStatus(
            queue_type=QueueType.TRAINING_DATA,
            current_depth=50000,
            soft_limit=100000,
            hard_limit=500000,
            target_depth=50000,
            backpressure=BackpressureLevel.SOFT,
            last_updated=now,
            trend="rising",
        )
        d = status.to_dict()
        assert d["queue_type"] == "training_data"
        assert d["current_depth"] == 50000
        assert d["backpressure"] == "soft"
        assert d["trend"] == "rising"


class TestQueueMetric:
    """Tests for QueueMetric dataclass."""

    def test_create_metric(self):
        """Should create QueueMetric with all fields."""
        metric = QueueMetric(
            queue_type="training_data",
            depth=10000,
            host="test-host",
            timestamp=time.time(),
        )
        assert metric.depth == 10000
        assert metric.host == "test-host"


class TestQueueMonitor:
    """Tests for QueueMonitor class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_queue_monitor.db"
            yield db_path

    @pytest.fixture
    def monitor(self, temp_db):
        """Create a QueueMonitor instance."""
        return QueueMonitor(db_path=temp_db)

    def test_init_creates_db(self, temp_db):
        """Should create database on init."""
        QueueMonitor(db_path=temp_db)
        assert temp_db.exists()

    def test_report_depth_returns_backpressure(self, monitor):
        """report_depth should return backpressure level."""
        level = monitor.report_depth(QueueType.TRAINING_DATA, 10000)
        assert level == BackpressureLevel.NONE

    def test_report_depth_soft_threshold(self, monitor):
        """Should return SOFT at soft limit."""
        # Training data soft limit is 100000
        level = monitor.report_depth(QueueType.TRAINING_DATA, 100000)
        assert level == BackpressureLevel.SOFT

    def test_report_depth_hard_threshold(self, monitor):
        """Should return HARD near hard limit."""
        # Training data hard limit is 500000, 90% = 450000
        level = monitor.report_depth(QueueType.TRAINING_DATA, 450000)
        assert level == BackpressureLevel.HARD

    def test_report_depth_stop_threshold(self, monitor):
        """Should return STOP at hard limit."""
        # Training data hard limit is 500000
        level = monitor.report_depth(QueueType.TRAINING_DATA, 500000)
        assert level == BackpressureLevel.STOP

    def test_get_status_returns_none_for_unreported(self, monitor):
        """Should return None for unreported queue."""
        status = monitor.get_status(QueueType.TRAINING_DATA)
        assert status is None

    def test_get_status_returns_status_after_report(self, monitor):
        """Should return status after depth is reported."""
        monitor.report_depth(QueueType.TRAINING_DATA, 50000)
        status = monitor.get_status(QueueType.TRAINING_DATA)
        assert status is not None
        assert status.current_depth == 50000
        assert status.backpressure == BackpressureLevel.NONE

    def test_get_all_status_empty_initially(self, monitor):
        """Should return empty dict initially."""
        statuses = monitor.get_all_status()
        assert statuses == {}

    def test_get_all_status_returns_reported_queues(self, monitor):
        """Should return all reported queues."""
        monitor.report_depth(QueueType.TRAINING_DATA, 10000)
        monitor.report_depth(QueueType.PENDING_GAMES, 500)
        statuses = monitor.get_all_status()
        assert "training_data" in statuses
        assert "pending_games" in statuses
        assert len(statuses) == 2

    def test_check_backpressure_none_for_unreported(self, monitor):
        """Should return NONE for unreported queue."""
        level = monitor.check_backpressure(QueueType.TRAINING_DATA)
        assert level == BackpressureLevel.NONE

    def test_check_backpressure_returns_current_level(self, monitor):
        """Should return current backpressure level."""
        monitor.report_depth(QueueType.TRAINING_DATA, 100000)
        level = monitor.check_backpressure(QueueType.TRAINING_DATA)
        assert level == BackpressureLevel.SOFT

    def test_should_throttle_false_below_soft(self, monitor):
        """should_throttle should be False below soft limit."""
        monitor.report_depth(QueueType.TRAINING_DATA, 10000)
        assert not monitor.should_throttle(QueueType.TRAINING_DATA)

    def test_should_throttle_true_at_soft(self, monitor):
        """should_throttle should be True at soft limit."""
        monitor.report_depth(QueueType.TRAINING_DATA, 100000)
        assert monitor.should_throttle(QueueType.TRAINING_DATA)

    def test_should_stop_false_below_hard(self, monitor):
        """should_stop should be False below hard limit."""
        monitor.report_depth(QueueType.TRAINING_DATA, 400000)
        assert not monitor.should_stop(QueueType.TRAINING_DATA)

    def test_should_stop_true_at_hard(self, monitor):
        """should_stop should be True at hard limit."""
        monitor.report_depth(QueueType.TRAINING_DATA, 500000)
        assert monitor.should_stop(QueueType.TRAINING_DATA)

    def test_get_throttle_factor_none(self, monitor):
        """Throttle factor should be 1.0 for NONE."""
        monitor.report_depth(QueueType.TRAINING_DATA, 10000)
        assert monitor.get_throttle_factor(QueueType.TRAINING_DATA) == 1.0

    def test_get_throttle_factor_soft(self, monitor):
        """Throttle factor should be 0.5 for SOFT."""
        monitor.report_depth(QueueType.TRAINING_DATA, 100000)
        assert monitor.get_throttle_factor(QueueType.TRAINING_DATA) == 0.5

    def test_get_throttle_factor_hard(self, monitor):
        """Throttle factor should be 0.1 for HARD."""
        monitor.report_depth(QueueType.TRAINING_DATA, 450000)
        assert monitor.get_throttle_factor(QueueType.TRAINING_DATA) == 0.1

    def test_get_throttle_factor_stop(self, monitor):
        """Throttle factor should be 0.0 for STOP."""
        monitor.report_depth(QueueType.TRAINING_DATA, 500000)
        assert monitor.get_throttle_factor(QueueType.TRAINING_DATA) == 0.0

    def test_register_callback(self, monitor):
        """Should register callback for backpressure changes."""
        callback = MagicMock()
        monitor.register_callback(QueueType.TRAINING_DATA, callback)

        # First report - no callback (no change)
        monitor.report_depth(QueueType.TRAINING_DATA, 10000)
        callback.assert_not_called()

        # Second report crossing soft threshold - should call callback
        monitor.report_depth(QueueType.TRAINING_DATA, 100000)
        callback.assert_called_once_with(BackpressureLevel.SOFT)

    def test_callback_on_level_change(self, monitor):
        """Callback should be called when level changes."""
        levels_seen = []
        def callback(level):
            return levels_seen.append(level)
        monitor.register_callback(QueueType.TRAINING_DATA, callback)

        # NONE -> SOFT
        monitor.report_depth(QueueType.TRAINING_DATA, 10000)  # NONE
        monitor.report_depth(QueueType.TRAINING_DATA, 100000)  # SOFT

        # SOFT -> HARD
        monitor.report_depth(QueueType.TRAINING_DATA, 450000)  # HARD

        # HARD -> STOP
        monitor.report_depth(QueueType.TRAINING_DATA, 500000)  # STOP

        assert levels_seen == [
            BackpressureLevel.SOFT,
            BackpressureLevel.HARD,
            BackpressureLevel.STOP,
        ]

    def test_get_history_empty_initially(self, monitor):
        """Should return empty history initially."""
        history = monitor.get_history(QueueType.TRAINING_DATA)
        assert history == []

    def test_get_history_returns_data_after_reports(self, monitor):
        """Should return history data after reports."""
        monitor.report_depth(QueueType.TRAINING_DATA, 10000)
        monitor.report_depth(QueueType.TRAINING_DATA, 20000)
        monitor.report_depth(QueueType.TRAINING_DATA, 15000)

        history = monitor.get_history(QueueType.TRAINING_DATA, hours=1)
        # Should have at least one bucket
        assert len(history) >= 1

    def test_trend_calculation_stable(self, monitor):
        """Should calculate stable trend correctly."""
        # Report same depth twice
        monitor.report_depth(QueueType.TRAINING_DATA, 10000)
        status = monitor.get_status(QueueType.TRAINING_DATA)
        # First report has no history, defaults to stable
        assert status.trend == "stable"


class TestDefaultQueueConfig:
    """Tests for default queue configuration."""

    def test_all_queue_types_have_config(self):
        """All queue types should have default config."""
        for qt in QueueType:
            assert qt in DEFAULT_QUEUE_CONFIG
            config = DEFAULT_QUEUE_CONFIG[qt]
            assert "soft_limit" in config
            assert "hard_limit" in config
            assert "target_depth" in config
            assert "drain_rate" in config

    def test_soft_less_than_hard(self):
        """Soft limit should always be less than hard limit."""
        for qt in QueueType:
            config = DEFAULT_QUEUE_CONFIG[qt]
            assert config["soft_limit"] < config["hard_limit"]

    def test_target_less_than_soft(self):
        """Target depth should be less than or equal to soft limit."""
        for qt in QueueType:
            config = DEFAULT_QUEUE_CONFIG[qt]
            assert config["target_depth"] <= config["soft_limit"]


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_global_monitor(self):
        """Reset global monitor between tests."""
        import app.coordination.queue_monitor as qm
        qm._monitor_instance = None
        yield
        qm._monitor_instance = None

    def test_get_queue_monitor_returns_singleton(self):
        """get_queue_monitor should return same instance."""
        m1 = get_queue_monitor()
        m2 = get_queue_monitor()
        assert m1 is m2

    def test_check_backpressure_convenience(self):
        """check_backpressure should work as convenience function."""
        level = check_backpressure(QueueType.TRAINING_DATA)
        assert level == BackpressureLevel.NONE

    def test_report_queue_depth_convenience(self):
        """report_queue_depth should work as convenience function."""
        level = report_queue_depth(QueueType.TRAINING_DATA, depth=10000)
        assert level == BackpressureLevel.NONE

    def test_should_throttle_production_convenience(self):
        """should_throttle_production should work as convenience function."""
        result = should_throttle_production(QueueType.TRAINING_DATA)
        assert result is False


class TestCustomConfig:
    """Tests for custom queue configuration."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_custom_config.db"
            yield db_path

    def test_custom_thresholds(self, temp_db):
        """Should respect custom thresholds."""
        custom_config = {
            QueueType.TRAINING_DATA: {
                "soft_limit": 100,
                "hard_limit": 200,
                "target_depth": 50,
                "drain_rate": 10,
            },
        }
        monitor = QueueMonitor(db_path=temp_db, config=custom_config)

        # Below soft = NONE
        level = monitor.report_depth(QueueType.TRAINING_DATA, 50)
        assert level == BackpressureLevel.NONE

        # At soft = SOFT
        level = monitor.report_depth(QueueType.TRAINING_DATA, 100)
        assert level == BackpressureLevel.SOFT

        # At hard = STOP
        level = monitor.report_depth(QueueType.TRAINING_DATA, 200)
        assert level == BackpressureLevel.STOP
