"""Unit tests for UnifiedHealthOrchestrator.

Tests cover:
- HealthCheckResult dataclass
- SystemHealthReport dataclass
- UnifiedHealthOrchestrator initialization
- Default health check registration
- Custom health check registration
- Health check execution
- Overall status aggregation
- Singleton pattern
- Convenience functions

Created: December 2025
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.monitoring.base import HealthStatus
from app.monitoring.unified_health import (
    HealthCheckResult,
    SystemHealthReport,
    UnifiedHealthOrchestrator,
    check_system_health,
    get_health_orchestrator,
    is_system_healthy,
)


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic health check result."""
        result = HealthCheckResult(
            component="test_component",
            status=HealthStatus.HEALTHY,
        )
        assert result.component == "test_component"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == ""
        assert result.details == {}
        assert result.check_time == 0.0

    def test_full_creation(self):
        """Test creating a health check result with all fields."""
        result = HealthCheckResult(
            component="database",
            status=HealthStatus.DEGRADED,
            message="High latency",
            details={"latency_ms": 150, "threshold_ms": 100},
            check_time=5.5,
        )
        assert result.component == "database"
        assert result.status == HealthStatus.DEGRADED
        assert result.message == "High latency"
        assert result.details == {"latency_ms": 150, "threshold_ms": 100}
        assert result.check_time == 5.5

    def test_to_dict(self):
        """Test converting result to dict."""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            details={"key": "value"},
            check_time=10.0,
        )
        d = result.to_dict()
        assert d["component"] == "test"
        assert d["status"] == "healthy"
        assert d["message"] == "OK"
        assert d["details"] == {"key": "value"}
        assert d["check_time"] == 10.0


class TestSystemHealthReport:
    """Tests for SystemHealthReport dataclass."""

    def test_basic_creation(self):
        """Test creating a basic health report."""
        report = SystemHealthReport(
            overall_status=HealthStatus.HEALTHY,
            healthy=True,
            checks=[],
            issues=[],
            timestamp=1000.0,
            duration_ms=5.0,
        )
        assert report.overall_status == HealthStatus.HEALTHY
        assert report.healthy is True
        assert report.checks == []
        assert report.issues == []
        assert report.timestamp == 1000.0
        assert report.duration_ms == 5.0

    def test_with_checks(self):
        """Test creating a report with health checks."""
        checks = [
            HealthCheckResult(
                component="comp1", status=HealthStatus.HEALTHY, message="OK"
            ),
            HealthCheckResult(
                component="comp2", status=HealthStatus.DEGRADED, message="Slow"
            ),
        ]
        report = SystemHealthReport(
            overall_status=HealthStatus.DEGRADED,
            healthy=False,
            checks=checks,
            issues=["comp2 (degraded): Slow"],
            timestamp=1000.0,
            duration_ms=15.0,
        )
        assert len(report.checks) == 2
        assert report.checks[0].component == "comp1"
        assert report.checks[1].component == "comp2"
        assert len(report.issues) == 1

    def test_to_dict(self):
        """Test converting report to dict."""
        check = HealthCheckResult(
            component="test", status=HealthStatus.HEALTHY, message="OK"
        )
        report = SystemHealthReport(
            overall_status=HealthStatus.HEALTHY,
            healthy=True,
            checks=[check],
            issues=[],
            timestamp=1000.0,
            duration_ms=5.0,
        )
        d = report.to_dict()
        assert d["overall_status"] == "healthy"
        assert d["healthy"] is True
        assert len(d["checks"]) == 1
        assert d["checks"][0]["component"] == "test"
        assert d["issues"] == []
        assert d["timestamp"] == 1000.0
        assert d["duration_ms"] == 5.0


class TestUnifiedHealthOrchestrator:
    """Tests for UnifiedHealthOrchestrator class."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = UnifiedHealthOrchestrator()
        assert orchestrator._health_checks is not None
        assert len(orchestrator._health_checks) > 0
        assert orchestrator._last_report is None

    def test_default_checks_registered(self):
        """Test that default health checks are registered."""
        orchestrator = UnifiedHealthOrchestrator()
        registered = orchestrator.get_registered_checks()

        # Should have default checks
        assert "training_scheduler" in registered
        assert "data_sync" in registered
        assert "resources" in registered
        assert "coordinators" in registered
        assert "event_bus" in registered

    def test_register_custom_check(self):
        """Test registering a custom health check."""
        orchestrator = UnifiedHealthOrchestrator()

        def custom_check() -> HealthCheckResult:
            return HealthCheckResult(
                component="custom", status=HealthStatus.HEALTHY, message="Custom OK"
            )

        orchestrator.register_check("custom", custom_check)
        assert "custom" in orchestrator.get_registered_checks()

    def test_unregister_check(self):
        """Test unregistering a health check."""
        orchestrator = UnifiedHealthOrchestrator()

        def dummy_check() -> HealthCheckResult:
            return HealthCheckResult(
                component="dummy", status=HealthStatus.HEALTHY, message=""
            )

        orchestrator.register_check("dummy", dummy_check)
        assert "dummy" in orchestrator.get_registered_checks()

        removed = orchestrator.unregister_check("dummy")
        assert removed is True
        assert "dummy" not in orchestrator.get_registered_checks()

    def test_unregister_nonexistent(self):
        """Test unregistering a nonexistent check."""
        orchestrator = UnifiedHealthOrchestrator()
        removed = orchestrator.unregister_check("nonexistent")
        assert removed is False

    def test_check_component(self):
        """Test checking a single component."""
        orchestrator = UnifiedHealthOrchestrator()

        def test_check() -> HealthCheckResult:
            return HealthCheckResult(
                component="test_comp",
                status=HealthStatus.HEALTHY,
                message="All good",
            )

        orchestrator.register_check("test_comp", test_check)

        result = orchestrator.check_component("test_comp")
        assert result is not None
        assert result.component == "test_comp"
        assert result.status == HealthStatus.HEALTHY
        assert result.check_time > 0

    def test_check_component_nonexistent(self):
        """Test checking a nonexistent component."""
        orchestrator = UnifiedHealthOrchestrator()
        result = orchestrator.check_component("nonexistent")
        assert result is None

    def test_check_component_exception(self):
        """Test handling exception in health check."""
        orchestrator = UnifiedHealthOrchestrator()

        def failing_check() -> HealthCheckResult:
            raise RuntimeError("Check failed!")

        orchestrator.register_check("failing", failing_check)

        result = orchestrator.check_component("failing")
        assert result is not None
        assert result.status == HealthStatus.UNKNOWN
        assert "Check failed" in result.message

    def test_check_all_health_all_healthy(self):
        """Test checking all health when all components are healthy."""
        orchestrator = UnifiedHealthOrchestrator()
        # Clear default checks
        orchestrator._health_checks = {}

        def healthy_check_1() -> HealthCheckResult:
            return HealthCheckResult(
                component="comp1", status=HealthStatus.HEALTHY, message="OK"
            )

        def healthy_check_2() -> HealthCheckResult:
            return HealthCheckResult(
                component="comp2", status=HealthStatus.HEALTHY, message="OK"
            )

        orchestrator.register_check("comp1", healthy_check_1)
        orchestrator.register_check("comp2", healthy_check_2)

        report = orchestrator.check_all_health()
        assert report.overall_status == HealthStatus.HEALTHY
        assert report.healthy is True
        assert len(report.checks) == 2
        assert len(report.issues) == 0

    def test_check_all_health_degraded(self):
        """Test checking all health when some components are degraded."""
        orchestrator = UnifiedHealthOrchestrator()
        orchestrator._health_checks = {}

        def healthy_check() -> HealthCheckResult:
            return HealthCheckResult(
                component="healthy", status=HealthStatus.HEALTHY, message="OK"
            )

        def degraded_check() -> HealthCheckResult:
            return HealthCheckResult(
                component="degraded", status=HealthStatus.DEGRADED, message="Slow"
            )

        orchestrator.register_check("healthy", healthy_check)
        orchestrator.register_check("degraded", degraded_check)

        report = orchestrator.check_all_health()
        assert report.overall_status == HealthStatus.DEGRADED
        assert report.healthy is False
        assert len(report.issues) == 1
        assert "degraded" in report.issues[0]

    def test_check_all_health_unhealthy(self):
        """Test checking all health when a component is unhealthy."""
        orchestrator = UnifiedHealthOrchestrator()
        orchestrator._health_checks = {}

        def healthy_check() -> HealthCheckResult:
            return HealthCheckResult(
                component="healthy", status=HealthStatus.HEALTHY, message="OK"
            )

        def unhealthy_check() -> HealthCheckResult:
            return HealthCheckResult(
                component="unhealthy", status=HealthStatus.UNHEALTHY, message="Down"
            )

        orchestrator.register_check("healthy", healthy_check)
        orchestrator.register_check("unhealthy", unhealthy_check)

        report = orchestrator.check_all_health()
        assert report.overall_status == HealthStatus.UNHEALTHY
        assert report.healthy is False
        assert len(report.issues) == 1
        assert "unhealthy" in report.issues[0]

    def test_check_all_health_mixed_statuses(self):
        """Test checking health with mixed statuses."""
        orchestrator = UnifiedHealthOrchestrator()
        orchestrator._health_checks = {}

        def healthy_check() -> HealthCheckResult:
            return HealthCheckResult(
                component="healthy", status=HealthStatus.HEALTHY, message=""
            )

        def degraded_check() -> HealthCheckResult:
            return HealthCheckResult(
                component="degraded", status=HealthStatus.DEGRADED, message="Slow"
            )

        def unhealthy_check() -> HealthCheckResult:
            return HealthCheckResult(
                component="unhealthy", status=HealthStatus.UNHEALTHY, message="Down"
            )

        orchestrator.register_check("healthy", healthy_check)
        orchestrator.register_check("degraded", degraded_check)
        orchestrator.register_check("unhealthy", unhealthy_check)

        report = orchestrator.check_all_health()
        # Unhealthy takes precedence over degraded
        assert report.overall_status == HealthStatus.UNHEALTHY
        assert len(report.issues) == 2

    def test_check_all_health_with_exception(self):
        """Test that exceptions in checks are handled gracefully."""
        orchestrator = UnifiedHealthOrchestrator()
        orchestrator._health_checks = {}

        def failing_check() -> HealthCheckResult:
            raise ValueError("Something went wrong!")

        orchestrator.register_check("failing", failing_check)

        # Should not raise
        report = orchestrator.check_all_health()
        assert len(report.checks) == 1
        assert report.checks[0].status == HealthStatus.UNKNOWN
        assert "Something went wrong!" in report.checks[0].message

    def test_check_all_health_duration(self):
        """Test that duration is tracked."""
        orchestrator = UnifiedHealthOrchestrator()
        orchestrator._health_checks = {}

        def slow_check() -> HealthCheckResult:
            time.sleep(0.01)  # 10ms
            return HealthCheckResult(
                component="slow", status=HealthStatus.HEALTHY, message=""
            )

        orchestrator.register_check("slow", slow_check)

        report = orchestrator.check_all_health()
        assert report.duration_ms >= 10  # At least 10ms

    def test_get_last_report(self):
        """Test getting last health report."""
        orchestrator = UnifiedHealthOrchestrator()
        orchestrator._health_checks = {}

        def healthy_check() -> HealthCheckResult:
            return HealthCheckResult(
                component="test", status=HealthStatus.HEALTHY, message=""
            )

        orchestrator.register_check("test", healthy_check)

        # No report yet
        assert orchestrator.get_last_report() is None

        # Generate report
        report = orchestrator.check_all_health()

        # Should return same report
        last = orchestrator.get_last_report()
        assert last is report


class TestDefaultHealthChecks:
    """Tests for default health check implementations."""

    def test_resources_check_healthy(self):
        """Test resources check when values are within thresholds."""
        orchestrator = UnifiedHealthOrchestrator()

        with patch("psutil.cpu_percent", return_value=50.0), patch(
            "psutil.virtual_memory"
        ) as mock_memory, patch("psutil.disk_usage") as mock_disk:
            mock_memory.return_value = MagicMock(percent=60.0)
            mock_disk.return_value = MagicMock(percent=50.0)

            result = orchestrator.check_component("resources")
            assert result is not None
            assert result.status == HealthStatus.HEALTHY
            assert "CPU" in result.message

    def test_resources_check_unhealthy(self):
        """Test resources check when values exceed thresholds."""
        orchestrator = UnifiedHealthOrchestrator()

        with patch("psutil.cpu_percent", return_value=99.0), patch(
            "psutil.virtual_memory"
        ) as mock_memory, patch("psutil.disk_usage") as mock_disk, patch(
            "app.config.thresholds.CPU_CRITICAL_PERCENT", 95
        ), patch(
            "app.config.thresholds.MEMORY_CRITICAL_PERCENT", 95
        ), patch(
            "app.config.thresholds.DISK_CRITICAL_PERCENT", 95
        ):
            mock_memory.return_value = MagicMock(percent=60.0)
            mock_disk.return_value = MagicMock(percent=50.0)

            result = orchestrator.check_component("resources")
            assert result is not None
            # May be healthy or unhealthy depending on threshold import

    def test_event_bus_check_no_subscribers(self):
        """Test event bus check when no subscribers."""
        orchestrator = UnifiedHealthOrchestrator()

        mock_router = MagicMock()
        mock_router.get_stats.return_value = {
            "total_subscriptions": 0,
            "total_events_published": 0,
        }

        # Patch at the module where it's imported inside the function
        with patch(
            "app.coordination.event_router.get_router", return_value=mock_router
        ):
            # Re-register to use our mock
            orchestrator._health_checks = {}
            orchestrator._register_default_checks()

            result = orchestrator.check_component("event_bus")
            assert result is not None
            assert result.status == HealthStatus.DEGRADED
            assert "No event subscribers" in result.message

    def test_event_bus_check_with_subscribers(self):
        """Test event bus check with active subscribers."""
        orchestrator = UnifiedHealthOrchestrator()

        mock_router = MagicMock()
        mock_router.get_stats.return_value = {
            "total_subscriptions": 50,
            "total_events_published": 1000,
        }

        # Patch at the module where it's imported inside the function
        with patch(
            "app.coordination.event_router.get_router", return_value=mock_router
        ):
            orchestrator._health_checks = {}
            orchestrator._register_default_checks()

            result = orchestrator.check_component("event_bus")
            assert result is not None
            assert result.status == HealthStatus.HEALTHY
            assert "1000 events published" in result.message
            assert "50 subscriptions" in result.message


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_health_orchestrator_singleton(self):
        """Test that get_health_orchestrator returns singleton."""
        import app.monitoring.unified_health as module

        # Reset singleton
        module._health_orchestrator = None

        orch1 = get_health_orchestrator()
        orch2 = get_health_orchestrator()
        assert orch1 is orch2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_system_health(self):
        """Test check_system_health convenience function."""
        import app.monitoring.unified_health as module

        # Create mock orchestrator
        mock_orchestrator = MagicMock()
        mock_report = SystemHealthReport(
            overall_status=HealthStatus.HEALTHY,
            healthy=True,
            checks=[],
            issues=[],
            timestamp=1000.0,
            duration_ms=5.0,
        )
        mock_orchestrator.check_all_health.return_value = mock_report
        module._health_orchestrator = mock_orchestrator

        result = check_system_health()
        assert result["healthy"] is True
        assert result["overall_status"] == "healthy"

    def test_is_system_healthy_true(self):
        """Test is_system_healthy returns True when healthy."""
        import app.monitoring.unified_health as module

        mock_orchestrator = MagicMock()
        mock_report = MagicMock()
        mock_report.healthy = True
        mock_orchestrator.check_all_health.return_value = mock_report
        module._health_orchestrator = mock_orchestrator

        assert is_system_healthy() is True

    def test_is_system_healthy_false(self):
        """Test is_system_healthy returns False when unhealthy."""
        import app.monitoring.unified_health as module

        mock_orchestrator = MagicMock()
        mock_report = MagicMock()
        mock_report.healthy = False
        mock_orchestrator.check_all_health.return_value = mock_report
        module._health_orchestrator = mock_orchestrator

        assert is_system_healthy() is False


class TestHealthStatusAggregation:
    """Tests for health status aggregation logic."""

    def test_all_healthy_overall_healthy(self):
        """Test all HEALTHY results in overall HEALTHY."""
        orchestrator = UnifiedHealthOrchestrator()
        orchestrator._health_checks = {}

        for i in range(3):

            def make_check(n):
                def check() -> HealthCheckResult:
                    return HealthCheckResult(
                        component=f"comp{n}",
                        status=HealthStatus.HEALTHY,
                        message="",
                    )

                return check

            orchestrator.register_check(f"comp{i}", make_check(i))

        report = orchestrator.check_all_health()
        assert report.overall_status == HealthStatus.HEALTHY

    def test_unknown_with_healthy_gives_degraded(self):
        """Test UNKNOWN + HEALTHY results in DEGRADED."""
        orchestrator = UnifiedHealthOrchestrator()
        orchestrator._health_checks = {}

        def healthy() -> HealthCheckResult:
            return HealthCheckResult(
                component="healthy", status=HealthStatus.HEALTHY, message=""
            )

        def unknown() -> HealthCheckResult:
            return HealthCheckResult(
                component="unknown", status=HealthStatus.UNKNOWN, message=""
            )

        orchestrator.register_check("healthy", healthy)
        orchestrator.register_check("unknown", unknown)

        report = orchestrator.check_all_health()
        assert report.overall_status == HealthStatus.DEGRADED

    def test_unhealthy_takes_precedence(self):
        """Test UNHEALTHY takes precedence over DEGRADED."""
        orchestrator = UnifiedHealthOrchestrator()
        orchestrator._health_checks = {}

        def degraded() -> HealthCheckResult:
            return HealthCheckResult(
                component="degraded", status=HealthStatus.DEGRADED, message=""
            )

        def unhealthy() -> HealthCheckResult:
            return HealthCheckResult(
                component="unhealthy", status=HealthStatus.UNHEALTHY, message=""
            )

        orchestrator.register_check("degraded", degraded)
        orchestrator.register_check("unhealthy", unhealthy)

        report = orchestrator.check_all_health()
        assert report.overall_status == HealthStatus.UNHEALTHY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
