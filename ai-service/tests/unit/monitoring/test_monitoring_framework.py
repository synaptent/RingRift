"""Tests for the unified monitoring framework.

Tests cover:
- Alert thresholds configuration (thresholds.py)
- Base monitoring classes (base.py)
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from app.config.thresholds import (
    AlertLevel,
    THRESHOLDS,
    get_threshold,
    should_alert,
    get_all_thresholds,
    update_threshold,
)
from app.monitoring.base import (
    HealthStatus,
    Alert,
    MonitoringResult,
    HealthMonitor,
    CompositeMonitor,
)


class TestAlertLevel:
    """Test AlertLevel enum."""

    def test_alert_levels_exist(self):
        """All expected alert levels should exist."""
        assert AlertLevel.INFO == "info"
        assert AlertLevel.WARNING == "warning"
        assert AlertLevel.CRITICAL == "critical"
        assert AlertLevel.FATAL == "fatal"

    def test_alert_level_is_string_enum(self):
        """AlertLevel should be usable as string."""
        assert str(AlertLevel.WARNING) == "AlertLevel.WARNING"
        assert AlertLevel.CRITICAL.value == "critical"


class TestThresholds:
    """Test threshold configuration."""

    def test_thresholds_has_required_categories(self):
        """THRESHOLDS should have all expected categories."""
        required = ["disk", "gpu_utilization", "gpu_memory", "training",
                    "data_quality", "cluster", "selfplay", "network", "memory"]
        for category in required:
            assert category in THRESHOLDS, f"Missing category: {category}"

    def test_disk_thresholds(self):
        """Disk thresholds should have expected values."""
        disk = THRESHOLDS["disk"]
        assert disk["warning"] == 65  # DISK_WARNING_PERCENT
        assert disk["critical"] == 70  # DISK_CRITICAL_PERCENT
        assert disk["fatal"] == 95
        assert disk["unit"] == "percent"

    def test_gpu_thresholds(self):
        """GPU thresholds should be properly configured."""
        gpu_util = THRESHOLDS["gpu_utilization"]
        assert gpu_util["idle"] == 5
        assert gpu_util["low"] == 20
        assert gpu_util["normal"] == 50

        gpu_mem = THRESHOLDS["gpu_memory"]
        assert gpu_mem["warning"] == 85
        assert gpu_mem["critical"] == 95

    def test_training_thresholds(self):
        """Training thresholds should exist."""
        training = THRESHOLDS["training"]
        assert training["stale_hours"] == 6.0  # TRAINING_STALENESS_HOURS
        assert training["model_stale_hours"] == 48
        assert training["min_batch_rate"] == 10

    def test_cluster_thresholds(self):
        """Cluster thresholds should exist."""
        cluster = THRESHOLDS["cluster"]
        assert cluster["min_nodes_online"] == 5
        assert cluster["node_timeout_seconds"] == 90  # PEER_TIMEOUT


class TestGetThreshold:
    """Test get_threshold function."""

    def test_get_existing_threshold(self):
        """Should return correct threshold value."""
        assert get_threshold("disk", "warning") == 65  # DISK_WARNING_PERCENT
        assert get_threshold("disk", "critical") == 70  # DISK_CRITICAL_PERCENT

    def test_get_nonexistent_category(self):
        """Should return default for missing category."""
        assert get_threshold("nonexistent", "warning") is None
        assert get_threshold("nonexistent", "warning", default=50) == 50

    def test_get_nonexistent_key(self):
        """Should return default for missing key."""
        assert get_threshold("disk", "nonexistent") is None
        assert get_threshold("disk", "nonexistent", default=100) == 100


class TestShouldAlert:
    """Test should_alert function."""

    def test_should_alert_gte_above_threshold(self):
        """Should alert when value >= threshold (default gte)."""
        assert should_alert("disk", 75, "warning") is True  # 75 >= 65
        assert should_alert("disk", 65, "warning") is True  # 65 >= 65

    def test_should_not_alert_gte_below_threshold(self):
        """Should not alert when value < threshold."""
        assert should_alert("disk", 60, "warning") is False  # 60 < 65

    def test_should_alert_lte_comparison(self):
        """Should alert with lte comparison."""
        # gpu_utilization.idle = 5
        assert should_alert("gpu_utilization", 3, "idle", comparison="lte") is True
        assert should_alert("gpu_utilization", 5, "idle", comparison="lte") is True
        assert should_alert("gpu_utilization", 10, "idle", comparison="lte") is False

    def test_should_alert_gt_comparison(self):
        """Should alert with gt comparison (strictly greater)."""
        assert should_alert("disk", 66, "warning", comparison="gt") is True
        assert should_alert("disk", 65, "warning", comparison="gt") is False

    def test_should_alert_lt_comparison(self):
        """Should alert with lt comparison (strictly less)."""
        assert should_alert("disk", 64, "warning", comparison="lt") is True
        assert should_alert("disk", 65, "warning", comparison="lt") is False

    def test_should_alert_nonexistent_category(self):
        """Should return False for nonexistent category."""
        assert should_alert("nonexistent", 100, "warning") is False

    def test_should_alert_nonexistent_level(self):
        """Should return False for nonexistent level."""
        assert should_alert("disk", 100, "nonexistent_level") is False


class TestGetAllThresholds:
    """Test get_all_thresholds function."""

    def test_returns_copy(self):
        """Should return a copy, not the original."""
        result = get_all_thresholds()
        result["test"] = {"value": 123}
        assert "test" not in THRESHOLDS


class TestUpdateThreshold:
    """Test update_threshold function."""

    def test_update_existing_threshold(self):
        """Should update existing threshold."""
        original = get_threshold("disk", "warning")
        try:
            update_threshold("disk", "warning", 75)
            assert get_threshold("disk", "warning") == 75
        finally:
            update_threshold("disk", "warning", original)

    def test_update_nonexistent_category(self):
        """Should not raise for nonexistent category."""
        update_threshold("nonexistent", "key", 100)  # Should not raise


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_health_statuses_exist(self):
        """All expected health statuses should exist."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        assert HealthStatus.UNKNOWN == "unknown"


class TestAlert:
    """Test Alert dataclass."""

    def test_alert_creation(self):
        """Should create alert with required fields."""
        alert = Alert(
            level=AlertLevel.WARNING,
            category="disk",
            message="Disk usage high",
        )
        assert alert.level == AlertLevel.WARNING
        assert alert.category == "disk"
        assert alert.message == "Disk usage high"
        assert isinstance(alert.timestamp, datetime)

    def test_alert_with_optional_fields(self):
        """Should create alert with optional fields."""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            category="gpu",
            message="GPU memory critical",
            node="lambda-gh200-e",
            metric_name="gpu_memory_percent",
            metric_value=96.5,
            threshold=95.0,
            details={"gpu_id": 0},
        )
        assert alert.node == "lambda-gh200-e"
        assert alert.metric_value == 96.5
        assert alert.threshold == 95.0
        assert alert.details == {"gpu_id": 0}

    def test_alert_to_dict(self):
        """Should convert to dictionary."""
        alert = Alert(
            level=AlertLevel.WARNING,
            category="test",
            message="Test message",
            metric_value=75.0,
            threshold=70.0,
        )
        d = alert.to_dict()
        assert d["level"] == "warning"
        assert d["category"] == "test"
        assert d["message"] == "Test message"
        assert "timestamp" in d

    def test_alert_str(self):
        """Should format as readable string."""
        alert = Alert(
            level=AlertLevel.WARNING,
            category="disk",
            message="Disk usage high",
            metric_value=75.0,
            threshold=70.0,
        )
        s = str(alert)
        assert "[WARNING]" in s
        assert "Disk usage high" in s
        assert "75.0/70.0" in s

    def test_alert_str_with_node(self):
        """Should include node in string if present."""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            category="gpu",
            message="GPU error",
            node="node-1",
        )
        s = str(alert)
        assert "(node-1)" in s


class TestMonitoringResult:
    """Test MonitoringResult dataclass."""

    def test_result_creation(self):
        """Should create result with status."""
        result = MonitoringResult(status=HealthStatus.HEALTHY)
        assert result.status == HealthStatus.HEALTHY
        assert result.is_healthy is True
        assert result.has_alerts is False

    def test_result_with_alerts(self):
        """Should handle alerts."""
        alerts = [
            Alert(level=AlertLevel.WARNING, category="test", message="warn"),
            Alert(level=AlertLevel.CRITICAL, category="test", message="critical"),
        ]
        result = MonitoringResult(
            status=HealthStatus.DEGRADED,
            alerts=alerts,
        )
        assert result.has_alerts is True
        assert len(result.alerts) == 2
        assert len(result.critical_alerts) == 1

    def test_result_is_healthy(self):
        """is_healthy should only be True for HEALTHY status."""
        assert MonitoringResult(status=HealthStatus.HEALTHY).is_healthy is True
        assert MonitoringResult(status=HealthStatus.DEGRADED).is_healthy is False
        assert MonitoringResult(status=HealthStatus.UNHEALTHY).is_healthy is False
        assert MonitoringResult(status=HealthStatus.UNKNOWN).is_healthy is False

    def test_result_to_dict(self):
        """Should convert to dictionary."""
        result = MonitoringResult(
            status=HealthStatus.HEALTHY,
            metrics={"cpu": 50.0, "memory": 60.0},
            details={"info": "test"},
        )
        d = result.to_dict()
        assert d["status"] == "healthy"
        assert d["metrics"]["cpu"] == 50.0
        assert d["details"]["info"] == "test"


class TestHealthMonitor:
    """Test HealthMonitor abstract base class."""

    def test_concrete_implementation(self):
        """Should work with concrete implementation."""
        class TestMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(
                    status=HealthStatus.HEALTHY,
                    metrics={"test_metric": 42},
                )

        monitor = TestMonitor("TestMonitor")
        assert monitor.name == "TestMonitor"
        assert monitor.last_result is None

        result = monitor.run_check()
        assert result.status == HealthStatus.HEALTHY
        assert result.metrics["test_metric"] == 42
        assert monitor.last_result is result
        assert result.duration_ms is not None

    def test_default_name(self):
        """Should use class name as default."""
        class MyCustomMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(status=HealthStatus.HEALTHY)

        monitor = MyCustomMonitor()
        assert monitor.name == "MyCustomMonitor"

    def test_run_check_handles_exception(self):
        """Should catch exceptions and return error result."""
        class FailingMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                raise RuntimeError("Test error")

        monitor = FailingMonitor()
        result = monitor.run_check()
        assert result.status == HealthStatus.UNKNOWN
        assert len(result.alerts) == 1
        assert "Test error" in result.alerts[0].message

    def test_should_alert_returns_most_severe(self):
        """should_alert should return most severe alert."""
        class AlertingMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(
                    status=HealthStatus.DEGRADED,
                    alerts=[
                        Alert(level=AlertLevel.WARNING, category="a", message="warn"),
                        Alert(level=AlertLevel.CRITICAL, category="b", message="crit"),
                        Alert(level=AlertLevel.INFO, category="c", message="info"),
                    ],
                )

        monitor = AlertingMonitor()
        monitor.run_check()
        alert = monitor.should_alert()
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_should_alert_returns_none_when_no_alerts(self):
        """should_alert should return None when no alerts."""
        class HealthyMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(status=HealthStatus.HEALTHY)

        monitor = HealthyMonitor()
        monitor.run_check()
        assert monitor.should_alert() is None

    def test_format_report(self):
        """Should format readable report."""
        class MetricsMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(
                    status=HealthStatus.HEALTHY,
                    metrics={"cpu": 45.5, "count": 10},
                    alerts=[Alert(level=AlertLevel.INFO, category="test", message="info")],
                )

        monitor = MetricsMonitor("TestMetrics")
        monitor.run_check()
        report = monitor.format_report()
        assert "TestMetrics" in report
        assert "healthy" in report
        assert "cpu: 45.50" in report
        assert "count: 10" in report
        assert "Alerts (1)" in report

    def test_format_report_no_data(self):
        """Should handle case with no data."""
        class EmptyMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(status=HealthStatus.HEALTHY)

        monitor = EmptyMonitor("Empty")
        report = monitor.format_report()
        assert "Empty: No data" in report


class TestCompositeMonitor:
    """Test CompositeMonitor for aggregating sub-monitors."""

    def test_composite_aggregates_metrics(self):
        """Should aggregate metrics from sub-monitors."""
        class MonitorA(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(
                    status=HealthStatus.HEALTHY,
                    metrics={"metric_a": 10},
                )

        class MonitorB(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(
                    status=HealthStatus.HEALTHY,
                    metrics={"metric_b": 20},
                )

        composite = CompositeMonitor("Aggregate")
        composite.add_monitor(MonitorA("A"))
        composite.add_monitor(MonitorB("B"))

        result = composite.run_check()
        assert result.status == HealthStatus.HEALTHY
        assert result.metrics["A.metric_a"] == 10
        assert result.metrics["B.metric_b"] == 20

    def test_composite_aggregates_alerts(self):
        """Should collect all alerts from sub-monitors."""
        class WarnMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(
                    status=HealthStatus.DEGRADED,
                    alerts=[Alert(level=AlertLevel.WARNING, category="w", message="warn")],
                )

        class CritMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(
                    status=HealthStatus.UNHEALTHY,
                    alerts=[Alert(level=AlertLevel.CRITICAL, category="c", message="crit")],
                )

        composite = CompositeMonitor()
        composite.add_monitor(WarnMonitor())
        composite.add_monitor(CritMonitor())

        result = composite.run_check()
        assert len(result.alerts) == 2

    def test_composite_worst_status(self):
        """Should return worst status from sub-monitors."""
        class HealthyMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(status=HealthStatus.HEALTHY)

        class UnhealthyMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(status=HealthStatus.UNHEALTHY)

        composite = CompositeMonitor()
        composite.add_monitor(HealthyMonitor())
        composite.add_monitor(UnhealthyMonitor())

        result = composite.run_check()
        assert result.status == HealthStatus.UNHEALTHY

    def test_composite_handles_sub_monitor_failure(self):
        """Should handle exceptions from sub-monitors gracefully."""
        class FailingMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                raise ValueError("Sub-monitor error")

        class GoodMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(status=HealthStatus.HEALTHY)

        composite = CompositeMonitor()
        composite.add_monitor(FailingMonitor("Failing"))
        composite.add_monitor(GoodMonitor("Good"))

        result = composite.run_check()
        # Failing monitor's exception is caught by run_check() and converted to alert
        # The alert message contains "Health check failed" from the base class wrapper
        assert any("Sub-monitor error" in a.message for a in result.alerts)
        # Good monitor should still be included
        assert "Good" in result.details
        # Failing monitor should also be in details (with error result)
        assert "Failing" in result.details

    def test_remove_monitor(self):
        """Should be able to remove sub-monitors."""
        class TestMonitor(HealthMonitor):
            def check_health(self) -> MonitoringResult:
                return MonitoringResult(status=HealthStatus.HEALTHY)

        composite = CompositeMonitor()
        monitor = TestMonitor("ToRemove")
        composite.add_monitor(monitor)
        composite.remove_monitor(monitor)

        result = composite.run_check()
        assert "ToRemove" not in result.details
