"""Tests for contracts.py - Coordination protocols and types.

Tests cover:
1. CoordinatorStatus enum
2. HealthCheckResult dataclass and factory methods
3. CoordinatorMetrics dataclass
4. Protocol type checking (CoordinatorProtocol, DaemonProtocol, EventDrivenProtocol)
5. Coordinator registry functions

December 2025: Created to ensure contracts.py has proper test coverage.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.coordination.contracts import (
    ConfigurableProtocol,
    CoordinatorMetrics,
    CoordinatorProtocol,
    CoordinatorStatus,
    DaemonProtocol,
    EventDrivenProtocol,
    HealthCheckResult,
    get_coordinator,
    get_registered_coordinators,
    is_coordinator,
    is_daemon,
    is_event_driven,
    register_coordinator,
    unregister_coordinator,
)


class TestCoordinatorStatus:
    """Tests for CoordinatorStatus enum."""

    def test_all_statuses_are_strings(self):
        """All statuses should be string enums."""
        for status in CoordinatorStatus:
            assert isinstance(status.value, str)

    def test_status_values(self):
        """Status values should match expected strings."""
        assert CoordinatorStatus.INITIALIZING.value == "initializing"
        assert CoordinatorStatus.STARTING.value == "starting"
        assert CoordinatorStatus.READY.value == "ready"
        assert CoordinatorStatus.RUNNING.value == "running"
        assert CoordinatorStatus.PAUSED.value == "paused"
        assert CoordinatorStatus.DRAINING.value == "draining"
        assert CoordinatorStatus.STOPPING.value == "stopping"
        assert CoordinatorStatus.STOPPED.value == "stopped"
        assert CoordinatorStatus.ERROR.value == "error"
        assert CoordinatorStatus.DEGRADED.value == "degraded"

    def test_status_count(self):
        """Should have expected number of statuses."""
        assert len(CoordinatorStatus) == 10

    def test_status_is_str_enum(self):
        """Status should be usable as string."""
        assert f"Status: {CoordinatorStatus.RUNNING}" == "Status: running"


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_basic_creation(self):
        """Should create with required field."""
        result = HealthCheckResult(healthy=True)
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert result.message == ""
        assert isinstance(result.timestamp, float)
        assert result.details == {}

    def test_creation_with_all_fields(self):
        """Should create with all fields."""
        details = {"key": "value"}
        result = HealthCheckResult(
            healthy=False,
            status=CoordinatorStatus.ERROR,
            message="Test error",
            timestamp=12345.0,
            details=details,
        )
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert result.message == "Test error"
        assert result.timestamp == 12345.0
        assert result.details == details

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        result = HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="OK",
            timestamp=12345.0,
            details={"uptime": 100},
        )
        d = result.to_dict()
        assert d["healthy"] is True
        assert d["status"] == "running"
        assert d["message"] == "OK"
        assert d["timestamp"] == 12345.0
        assert d["details"] == {"uptime": 100}

    def test_unhealthy_factory(self):
        """Should create unhealthy result."""
        result = HealthCheckResult.unhealthy("Error occurred", code=500)
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert result.message == "Error occurred"
        assert result.details["code"] == 500

    def test_degraded_factory(self):
        """Should create degraded result."""
        result = HealthCheckResult.degraded("High latency", latency_ms=500)
        assert result.healthy is True  # Still operational
        assert result.status == CoordinatorStatus.DEGRADED
        assert result.message == "High latency"
        assert result.details["latency_ms"] == 500


class TestHealthCheckResultFromMetrics:
    """Tests for HealthCheckResult.from_metrics factory method."""

    def test_healthy_from_metrics(self):
        """Should create healthy result from good metrics."""
        result = HealthCheckResult.from_metrics(
            uptime_seconds=3600,
            events_processed=1000,
            errors_count=5,
        )
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert result.message == "OK"
        assert result.details["uptime_seconds"] == 3600
        assert result.details["events_processed"] == 1000
        assert result.details["errors_count"] == 5
        assert result.details["error_rate"] == 0.005

    def test_unhealthy_high_error_rate(self):
        """Should mark unhealthy with high error rate."""
        result = HealthCheckResult.from_metrics(
            uptime_seconds=3600,
            events_processed=100,
            errors_count=15,  # 15% error rate > 10% threshold
        )
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert "error rate" in result.message.lower()

    def test_degraded_elevated_error_rate(self):
        """Should mark degraded with elevated error rate."""
        result = HealthCheckResult.from_metrics(
            uptime_seconds=3600,
            events_processed=100,
            errors_count=8,  # 8% error rate > 5% but < 10%
        )
        assert result.healthy is True
        assert result.status == CoordinatorStatus.DEGRADED
        assert "error rate" in result.message.lower()

    def test_degraded_inactivity(self):
        """Should mark degraded with high inactivity."""
        result = HealthCheckResult.from_metrics(
            uptime_seconds=3600,
            events_processed=100,
            errors_count=0,
            last_activity_ago=400,  # > 300s threshold
        )
        assert result.healthy is True
        assert result.status == CoordinatorStatus.DEGRADED
        assert "inactive" in result.message.lower()

    def test_custom_thresholds(self):
        """Should respect custom thresholds."""
        result = HealthCheckResult.from_metrics(
            uptime_seconds=3600,
            events_processed=100,
            errors_count=25,
            max_error_rate=0.3,  # 30% threshold
        )
        # 25% error rate < 30% threshold
        assert result.healthy is True
        assert result.status == CoordinatorStatus.DEGRADED

    def test_explicit_error_rate(self):
        """Should use explicit error_rate if provided."""
        result = HealthCheckResult.from_metrics(
            uptime_seconds=3600,
            events_processed=100,
            errors_count=5,  # Would be 5%
            error_rate=0.02,  # But we provide 2%
        )
        assert result.details["error_rate"] == 0.02

    def test_zero_events(self):
        """Should handle zero events gracefully."""
        result = HealthCheckResult.from_metrics(
            uptime_seconds=3600,
            events_processed=0,
            errors_count=0,
        )
        assert result.healthy is True
        assert result.details["error_rate"] == 0.0

    def test_extra_details(self):
        """Should include extra details."""
        result = HealthCheckResult.from_metrics(
            uptime_seconds=3600,
            events_processed=100,
            errors_count=1,
            custom_field="test",
            another=123,
        )
        assert result.details["custom_field"] == "test"
        assert result.details["another"] == 123


class TestHealthCheckResultHealthScore:
    """Tests for HealthCheckResult.health_score method."""

    def test_unhealthy_returns_zero(self):
        """Unhealthy results should return 0.0."""
        result = HealthCheckResult(healthy=False, status=CoordinatorStatus.ERROR)
        assert result.health_score() == 0.0

    def test_error_status_returns_low_score(self):
        """Error status with healthy=True should return 0.1."""
        result = HealthCheckResult(healthy=True, status=CoordinatorStatus.ERROR)
        assert result.health_score() == 0.1

    def test_degraded_returns_0_7(self):
        """Degraded status should return 0.7."""
        result = HealthCheckResult(healthy=True, status=CoordinatorStatus.DEGRADED)
        assert result.health_score() == 0.7

    def test_running_returns_1_0(self):
        """Running status should return 1.0."""
        result = HealthCheckResult(healthy=True, status=CoordinatorStatus.RUNNING)
        assert result.health_score() == 1.0

    def test_ready_returns_1_0(self):
        """Ready status should return 1.0."""
        result = HealthCheckResult(healthy=True, status=CoordinatorStatus.READY)
        assert result.health_score() == 1.0

    def test_paused_returns_0_5(self):
        """Paused status should return 0.5."""
        result = HealthCheckResult(healthy=True, status=CoordinatorStatus.PAUSED)
        assert result.health_score() == 0.5

    def test_draining_returns_0_6(self):
        """Draining status should return 0.6."""
        result = HealthCheckResult(healthy=True, status=CoordinatorStatus.DRAINING)
        assert result.health_score() == 0.6

    def test_stopping_returns_0_3(self):
        """Stopping status should return 0.3."""
        result = HealthCheckResult(healthy=True, status=CoordinatorStatus.STOPPING)
        assert result.health_score() == 0.3

    def test_stopped_returns_0_0(self):
        """Stopped status should return 0.0."""
        result = HealthCheckResult(healthy=True, status=CoordinatorStatus.STOPPED)
        assert result.health_score() == 0.0

    def test_initializing_returns_0_8(self):
        """Initializing status should return 0.8."""
        result = HealthCheckResult(healthy=True, status=CoordinatorStatus.INITIALIZING)
        assert result.health_score() == 0.8


class TestHealthCheckResultWithDetails:
    """Tests for HealthCheckResult.with_details method."""

    def test_adds_details(self):
        """Should add new details."""
        result = HealthCheckResult(healthy=True)
        new_result = result.with_details(key="value")
        assert new_result.details["key"] == "value"
        assert result.details == {}  # Original unchanged

    def test_merges_with_existing(self):
        """Should merge with existing details."""
        result = HealthCheckResult(healthy=True, details={"existing": 1})
        new_result = result.with_details(new_key=2)
        assert new_result.details["existing"] == 1
        assert new_result.details["new_key"] == 2

    def test_overrides_existing(self):
        """Should override existing keys."""
        result = HealthCheckResult(healthy=True, details={"key": "old"})
        new_result = result.with_details(key="new")
        assert new_result.details["key"] == "new"

    def test_preserves_other_fields(self):
        """Should preserve other fields."""
        result = HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.DEGRADED,
            message="Test",
            timestamp=12345.0,
        )
        new_result = result.with_details(extra=1)
        assert new_result.healthy == result.healthy
        assert new_result.status == result.status
        assert new_result.message == result.message
        assert new_result.timestamp == result.timestamp


class TestCoordinatorMetrics:
    """Tests for CoordinatorMetrics dataclass."""

    def test_basic_creation(self):
        """Should create with required field."""
        metrics = CoordinatorMetrics(
            name="test",
            status=CoordinatorStatus.RUNNING,
        )
        assert metrics.name == "test"
        assert metrics.status == CoordinatorStatus.RUNNING
        assert metrics.uptime_seconds == 0.0
        assert metrics.events_processed == 0
        assert metrics.errors_count == 0
        assert metrics.last_error == ""
        assert metrics.custom_metrics == {}

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        metrics = CoordinatorMetrics(
            name="test",
            status=CoordinatorStatus.RUNNING,
            uptime_seconds=100.0,
            events_processed=50,
            errors_count=2,
            last_error="Some error",
            custom_metrics={"key": "value"},
        )
        d = metrics.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "running"
        assert d["uptime_seconds"] == 100.0
        assert d["events_processed"] == 50
        assert d["errors_count"] == 2
        assert d["last_error"] == "Some error"
        assert d["key"] == "value"  # Custom metrics merged


class TestCoordinatorRegistry:
    """Tests for coordinator registry functions."""

    def setup_method(self):
        """Clear registry before each test."""
        from app.coordination import contracts
        contracts._COORDINATOR_REGISTRY.clear()

    def test_register_and_get(self):
        """Should register and retrieve coordinator."""
        mock = MagicMock()
        mock.name = "test_coordinator"
        register_coordinator(mock)
        result = get_coordinator("test_coordinator")
        assert result is mock

    def test_register_with_custom_name(self):
        """Should register with custom name."""
        mock = MagicMock()
        mock.name = "original"
        register_coordinator(mock, name="custom")
        assert get_coordinator("custom") is mock
        assert get_coordinator("original") is None

    def test_register_without_name_attribute(self):
        """Should use class name if no name attribute."""
        class MyCoordinator:
            pass
        coord = MyCoordinator()
        register_coordinator(coord)
        assert get_coordinator("MyCoordinator") is coord

    def test_unregister(self):
        """Should unregister coordinator."""
        mock = MagicMock()
        mock.name = "test"
        register_coordinator(mock)
        assert get_coordinator("test") is mock
        unregister_coordinator("test")
        assert get_coordinator("test") is None

    def test_unregister_nonexistent(self):
        """Should handle unregistering nonexistent coordinator."""
        unregister_coordinator("nonexistent")  # Should not raise

    def test_get_registered_coordinators(self):
        """Should return all registered coordinators."""
        mock1 = MagicMock()
        mock1.name = "coord1"
        mock2 = MagicMock()
        mock2.name = "coord2"
        register_coordinator(mock1)
        register_coordinator(mock2)
        all_coords = get_registered_coordinators()
        assert len(all_coords) == 2
        assert all_coords["coord1"] is mock1
        assert all_coords["coord2"] is mock2

    def test_get_registered_returns_copy(self):
        """Should return a copy of the registry."""
        mock = MagicMock()
        mock.name = "test"
        register_coordinator(mock)
        all_coords = get_registered_coordinators()
        all_coords["new"] = MagicMock()
        # Original registry unchanged
        assert get_coordinator("new") is None


class TestProtocolTypeChecking:
    """Tests for protocol type checking functions."""

    def test_is_coordinator_true(self):
        """Should identify coordinator protocol."""
        class ValidCoordinator:
            @property
            def name(self) -> str:
                return "test"

            @property
            def status(self) -> CoordinatorStatus:
                return CoordinatorStatus.RUNNING

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

            def health_check(self) -> HealthCheckResult:
                return HealthCheckResult(healthy=True)

        coord = ValidCoordinator()
        assert is_coordinator(coord) is True

    def test_is_coordinator_false(self):
        """Should reject non-coordinator."""
        class NotCoordinator:
            pass
        assert is_coordinator(NotCoordinator()) is False

    def test_is_daemon_true(self):
        """Should identify daemon protocol."""
        class ValidDaemon:
            @property
            def name(self) -> str:
                return "test"

            @property
            def status(self) -> CoordinatorStatus:
                return CoordinatorStatus.RUNNING

            @property
            def is_running(self) -> bool:
                return True

            @property
            def uptime_seconds(self) -> float:
                return 100.0

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            async def restart(self) -> None:
                pass

            async def pause(self) -> None:
                pass

            async def resume(self) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

            def health_check(self) -> HealthCheckResult:
                return HealthCheckResult(healthy=True)

        daemon = ValidDaemon()
        assert is_daemon(daemon) is True

    def test_is_daemon_false(self):
        """Should reject non-daemon."""
        class NotDaemon:
            pass
        assert is_daemon(NotDaemon()) is False

    def test_is_event_driven_true(self):
        """Should identify event-driven protocol."""
        class ValidEventDriven:
            @property
            def name(self) -> str:
                return "test"

            @property
            def status(self) -> CoordinatorStatus:
                return CoordinatorStatus.RUNNING

            @property
            def subscribed_events(self) -> list[str]:
                return ["event1", "event2"]

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

            def health_check(self) -> HealthCheckResult:
                return HealthCheckResult(healthy=True)

            def subscribe_to_events(self) -> bool:
                return True

            def unsubscribe_from_events(self) -> None:
                pass

        ed = ValidEventDriven()
        assert is_event_driven(ed) is True

    def test_is_event_driven_false(self):
        """Should reject non-event-driven."""
        class NotEventDriven:
            pass
        assert is_event_driven(NotEventDriven()) is False


class TestConfigurableProtocol:
    """Tests for ConfigurableProtocol."""

    def test_is_runtime_checkable(self):
        """Should be runtime checkable."""
        class ValidConfigurable:
            def get_config(self) -> dict[str, Any]:
                return {"key": "value"}

            def update_config(self, config: dict[str, Any]) -> bool:
                return True

        assert isinstance(ValidConfigurable(), ConfigurableProtocol)

    def test_not_configurable(self):
        """Should reject non-configurable."""
        class NotConfigurable:
            pass
        assert not isinstance(NotConfigurable(), ConfigurableProtocol)
