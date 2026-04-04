"""Tests for app.coordination.protocols - Coordination Protocols (December 2025)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from app.coordination.protocols import (
    BaseCoordinator,
    BaseDaemon,
    ConfigurableProtocol,
    CoordinatorMetrics,
    CoordinatorProtocol,
    CoordinatorStatus,
    DaemonProtocol,
    EventDrivenProtocol,
    HealthCheckResult,
    get_all_metrics,
    get_coordinator,
    get_registered_coordinators,
    health_check_all,
    is_coordinator,
    is_daemon,
    is_event_driven,
    register_coordinator,
    unregister_coordinator,
    validate_coordinator,
)


class TestCoordinatorStatus:
    """Tests for CoordinatorStatus enum."""

    def test_all_status_values_exist(self):
        """Test all expected status values are defined."""
        assert CoordinatorStatus.INITIALIZING == "initializing"
        assert CoordinatorStatus.RUNNING == "running"
        assert CoordinatorStatus.PAUSED == "paused"
        assert CoordinatorStatus.STOPPING == "stopping"
        assert CoordinatorStatus.STOPPED == "stopped"
        assert CoordinatorStatus.ERROR == "error"
        assert CoordinatorStatus.DEGRADED == "degraded"

    def test_status_is_string_enum(self):
        """Test CoordinatorStatus is a string enum."""
        assert isinstance(CoordinatorStatus.RUNNING.value, str)
        # Can be compared to strings
        assert CoordinatorStatus.RUNNING == "running"


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_basic_creation(self):
        """Test HealthCheckResult can be created."""
        result = HealthCheckResult(healthy=True)
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert result.message == ""
        assert isinstance(result.timestamp, float)
        assert result.details == {}

    def test_creation_with_all_fields(self):
        """Test HealthCheckResult with all fields specified."""
        result = HealthCheckResult(
            healthy=False,
            status=CoordinatorStatus.ERROR,
            message="Something went wrong",
            timestamp=1704067200.0,
            details={"error_code": 500},
        )
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert result.message == "Something went wrong"
        assert result.timestamp == 1704067200.0
        assert result.details == {"error_code": 500}

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="OK",
            details={"uptime": 100},
        )
        d = result.to_dict()
        assert d["healthy"] is True
        assert d["status"] == "running"
        assert d["message"] == "OK"
        assert d["details"] == {"uptime": 100}
        assert "timestamp" in d

    def test_unhealthy_factory(self):
        """Test unhealthy factory method."""
        result = HealthCheckResult.unhealthy("Connection failed", retries=3)
        assert result.healthy is False
        assert result.status == CoordinatorStatus.ERROR
        assert result.message == "Connection failed"
        assert result.details == {"retries": 3}

    def test_degraded_factory(self):
        """Test degraded factory method."""
        result = HealthCheckResult.degraded("High latency", latency_ms=500)
        assert result.healthy is True  # Still operational
        assert result.status == CoordinatorStatus.DEGRADED
        assert result.message == "High latency"
        assert result.details == {"latency_ms": 500}


class TestCoordinatorMetrics:
    """Tests for CoordinatorMetrics dataclass."""

    def test_basic_creation(self):
        """Test CoordinatorMetrics can be created."""
        metrics = CoordinatorMetrics(
            name="TestCoordinator",
            status=CoordinatorStatus.RUNNING,
        )
        assert metrics.name == "TestCoordinator"
        assert metrics.status == CoordinatorStatus.RUNNING
        assert metrics.uptime_seconds == 0.0
        assert metrics.events_processed == 0
        assert metrics.errors_count == 0

    def test_creation_with_all_fields(self):
        """Test CoordinatorMetrics with all fields."""
        metrics = CoordinatorMetrics(
            name="TestDaemon",
            status=CoordinatorStatus.RUNNING,
            uptime_seconds=3600.0,
            start_time=1704067200.0,
            events_processed=100,
            errors_count=5,
            last_error="Timeout",
            last_activity_time=1704070800.0,
            custom_metrics={"queue_depth": 10},
        )
        assert metrics.uptime_seconds == 3600.0
        assert metrics.events_processed == 100
        assert metrics.custom_metrics == {"queue_depth": 10}

    def test_to_dict(self):
        """Test to_dict serialization."""
        metrics = CoordinatorMetrics(
            name="TestCoordinator",
            status=CoordinatorStatus.RUNNING,
            custom_metrics={"custom_key": "custom_value"},
        )
        d = metrics.to_dict()
        assert d["name"] == "TestCoordinator"
        assert d["status"] == "running"
        assert d["custom_key"] == "custom_value"


class TestBaseCoordinator:
    """Tests for BaseCoordinator abstract class."""

    class ConcreteCoordinator(BaseCoordinator):
        """Concrete implementation for testing."""

        @property
        def name(self) -> str:
            return "ConcreteCoordinator"

        async def _on_start(self) -> None:
            pass

        async def _on_stop(self) -> None:
            pass

    @pytest.fixture
    def coordinator(self):
        """Create a concrete coordinator for testing."""
        return self.ConcreteCoordinator()

    def test_initial_state(self, coordinator):
        """Test initial state is INITIALIZING."""
        assert coordinator.status == CoordinatorStatus.INITIALIZING
        assert coordinator._events_processed == 0
        assert coordinator._errors_count == 0

    @pytest.mark.asyncio
    async def test_start(self, coordinator):
        """Test start transitions to RUNNING."""
        await coordinator.start()
        assert coordinator.status == CoordinatorStatus.RUNNING
        assert coordinator._start_time > 0

    @pytest.mark.asyncio
    async def test_start_idempotent(self, coordinator):
        """Test start is idempotent."""
        await coordinator.start()
        start_time = coordinator._start_time
        await coordinator.start()
        # Start time should not change
        assert coordinator._start_time == start_time

    @pytest.mark.asyncio
    async def test_stop(self, coordinator):
        """Test stop transitions to STOPPED."""
        await coordinator.start()
        await coordinator.stop()
        assert coordinator.status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, coordinator):
        """Test stop is idempotent."""
        await coordinator.start()
        await coordinator.stop()
        await coordinator.stop()  # Should not raise
        assert coordinator.status == CoordinatorStatus.STOPPED

    def test_uptime_before_start(self, coordinator):
        """Test uptime is 0 before start."""
        assert coordinator.uptime_seconds == 0.0

    @pytest.mark.asyncio
    async def test_uptime_after_start(self, coordinator):
        """Test uptime increases after start."""
        await coordinator.start()
        await asyncio.sleep(0.01)
        assert coordinator.uptime_seconds > 0

    def test_get_metrics(self, coordinator):
        """Test get_metrics returns proper structure."""
        metrics = coordinator.get_metrics()
        assert metrics["name"] == "ConcreteCoordinator"
        assert metrics["status"] == "initializing"
        assert "uptime_seconds" in metrics
        assert "events_processed" in metrics

    def test_health_check_initializing(self, coordinator):
        """Test health check when initializing."""
        result = coordinator.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.INITIALIZING

    @pytest.mark.asyncio
    async def test_health_check_running(self, coordinator):
        """Test health check when running."""
        await coordinator.start()
        result = coordinator.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    @pytest.mark.asyncio
    async def test_health_check_stopped(self, coordinator):
        """Test health check when stopped."""
        await coordinator.start()
        await coordinator.stop()
        result = coordinator.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.STOPPED

    def test_health_check_error(self, coordinator):
        """Test health check when in error state."""
        coordinator._status = CoordinatorStatus.ERROR
        coordinator._last_error = "Database connection failed"
        result = coordinator.health_check()
        assert result.healthy is False
        assert "Database connection failed" in result.message

    def test_record_event(self, coordinator):
        """Test _record_event increments counter."""
        assert coordinator._events_processed == 0
        coordinator._record_event()
        assert coordinator._events_processed == 1
        assert coordinator._last_activity > 0

    def test_record_error(self, coordinator):
        """Test _record_error tracks errors."""
        assert coordinator._errors_count == 0
        coordinator._record_error("Test error")
        assert coordinator._errors_count == 1
        assert coordinator._last_error == "Test error"


class TestBaseDaemon:
    """Tests for BaseDaemon abstract class."""

    class ConcreteDaemon(BaseDaemon):
        """Concrete implementation for testing."""

        @property
        def name(self) -> str:
            return "ConcreteDaemon"

        async def _on_start(self) -> None:
            pass

        async def _on_stop(self) -> None:
            pass

    @pytest.fixture
    def daemon(self):
        """Create a concrete daemon for testing."""
        with pytest.deprecated_call(match="protocols\\.BaseDaemon is deprecated"):
            return self.ConcreteDaemon()

    def test_is_running_initially_false(self, daemon):
        """Test is_running is False initially."""
        assert daemon.is_running is False

    @pytest.mark.asyncio
    async def test_is_running_after_start(self, daemon):
        """Test is_running is True after start."""
        await daemon.start()
        assert daemon.is_running is True

    @pytest.mark.asyncio
    async def test_restart(self, daemon):
        """Test restart stops and starts."""
        await daemon.start()
        assert daemon.is_running is True
        await daemon.restart()
        assert daemon.is_running is True

    @pytest.mark.asyncio
    async def test_pause(self, daemon):
        """Test pause transitions to PAUSED."""
        await daemon.start()
        await daemon.pause()
        assert daemon.status == CoordinatorStatus.PAUSED
        assert daemon._paused is True

    @pytest.mark.asyncio
    async def test_pause_when_not_running(self, daemon):
        """Test pause does nothing when not running."""
        await daemon.pause()
        assert daemon.status != CoordinatorStatus.PAUSED

    @pytest.mark.asyncio
    async def test_resume(self, daemon):
        """Test resume transitions from PAUSED to RUNNING."""
        await daemon.start()
        await daemon.pause()
        assert daemon.status == CoordinatorStatus.PAUSED
        await daemon.resume()
        assert daemon.status == CoordinatorStatus.RUNNING
        assert daemon._paused is False

    @pytest.mark.asyncio
    async def test_resume_when_not_paused(self, daemon):
        """Test resume does nothing when not paused."""
        await daemon.start()
        initial_status = daemon.status
        await daemon.resume()  # Not paused, should be no-op
        assert daemon.status == initial_status


class TestValidationHelpers:
    """Tests for validation helper functions."""

    class ValidCoordinator:
        """A class that implements CoordinatorProtocol."""

        @property
        def name(self) -> str:
            return "ValidCoordinator"

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

    class InvalidCoordinator:
        """A class missing some protocol methods."""

        @property
        def name(self) -> str:
            return "InvalidCoordinator"

        # Missing: status, start, stop, get_metrics, health_check

    def test_validate_coordinator_valid(self):
        """Test validate_coordinator with valid coordinator."""
        coord = self.ValidCoordinator()
        missing = validate_coordinator(coord)
        assert missing == []

    def test_validate_coordinator_invalid(self):
        """Test validate_coordinator with invalid coordinator."""
        coord = self.InvalidCoordinator()
        missing = validate_coordinator(coord)
        # Should report missing methods
        assert "status" in missing or "start" in missing

    def test_is_coordinator_true(self):
        """Test is_coordinator returns True for valid coordinator."""
        coord = self.ValidCoordinator()
        assert is_coordinator(coord) is True

    def test_is_coordinator_false(self):
        """Test is_coordinator returns False for invalid object."""
        assert is_coordinator({"not": "a coordinator"}) is False
        assert is_coordinator("string") is False
        assert is_coordinator(None) is False

    def test_is_daemon(self):
        """Test is_daemon detection."""

        class SimpleDaemon:
            @property
            def name(self) -> str:
                return "SimpleDaemon"

            @property
            def status(self) -> CoordinatorStatus:
                return CoordinatorStatus.RUNNING

            @property
            def is_running(self) -> bool:
                return True

            @property
            def uptime_seconds(self) -> float:
                return 0.0

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

        daemon = SimpleDaemon()
        assert is_daemon(daemon) is True

    def test_is_event_driven(self):
        """Test is_event_driven detection."""

        class SimpleEventDriven:
            @property
            def name(self) -> str:
                return "SimpleEventDriven"

            @property
            def status(self) -> CoordinatorStatus:
                return CoordinatorStatus.RUNNING

            @property
            def subscribed_events(self) -> list[str]:
                return ["event1"]

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

        ed = SimpleEventDriven()
        assert is_event_driven(ed) is True


class TestRegistry:
    """Tests for coordinator registry functions."""

    class TestCoord(BaseCoordinator):
        """Test coordinator for registry tests."""

        __test__ = False

        def __init__(self, name: str):
            super().__init__()
            self._name = name

        @property
        def name(self) -> str:
            return self._name

        async def _on_start(self) -> None:
            pass

        async def _on_stop(self) -> None:
            pass

    @pytest.fixture(autouse=True)
    def cleanup_registry(self):
        """Clean up registry before and after each test."""
        import app.coordination.protocols as protocols

        # Save current state
        saved = dict(protocols._coordinator_registry)
        protocols._coordinator_registry.clear()
        yield
        # Restore
        protocols._coordinator_registry.clear()
        protocols._coordinator_registry.update(saved)

    def test_register_and_get(self):
        """Test registering and retrieving a coordinator."""
        coord = self.TestCoord("TestCoord1")
        register_coordinator(coord)

        result = get_coordinator("TestCoord1")
        assert result is coord

    def test_get_nonexistent(self):
        """Test getting a nonexistent coordinator."""
        result = get_coordinator("NonExistent")
        assert result is None

    def test_unregister(self):
        """Test unregistering a coordinator."""
        coord = self.TestCoord("TestCoord2")
        register_coordinator(coord)

        unregister_coordinator("TestCoord2")
        assert get_coordinator("TestCoord2") is None

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent coordinator doesn't raise."""
        unregister_coordinator("NonExistent")  # Should not raise

    def test_get_registered_coordinators(self):
        """Test getting all registered coordinators."""
        coord1 = self.TestCoord("Coord1")
        coord2 = self.TestCoord("Coord2")
        register_coordinator(coord1)
        register_coordinator(coord2)

        all_coords = get_registered_coordinators()
        assert len(all_coords) == 2
        assert "Coord1" in all_coords
        assert "Coord2" in all_coords

    def test_health_check_all(self):
        """Test health_check_all runs health checks on all coordinators."""
        coord1 = self.TestCoord("HealthCoord1")
        coord2 = self.TestCoord("HealthCoord2")
        register_coordinator(coord1)
        register_coordinator(coord2)

        results = health_check_all()
        assert len(results) == 2
        assert "HealthCoord1" in results
        assert "HealthCoord2" in results
        assert all(isinstance(r, HealthCheckResult) for r in results.values())

    def test_get_all_metrics(self):
        """Test get_all_metrics returns metrics from all coordinators."""
        coord1 = self.TestCoord("MetricsCoord1")
        coord2 = self.TestCoord("MetricsCoord2")
        register_coordinator(coord1)
        register_coordinator(coord2)

        metrics = get_all_metrics()
        assert len(metrics) == 2
        assert "MetricsCoord1" in metrics
        assert "MetricsCoord2" in metrics
        assert all(isinstance(m, dict) for m in metrics.values())


class TestConfigurableProtocol:
    """Tests for ConfigurableProtocol."""

    class ConfigurableCoord:
        """A coordinator that implements ConfigurableProtocol."""

        def __init__(self):
            self._config = {"threshold": 100, "enabled": True}

        @property
        def name(self) -> str:
            return "ConfigurableCoord"

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

        def get_config(self) -> dict[str, Any]:
            return dict(self._config)

        def update_config(self, config: dict[str, Any]) -> bool:
            self._config.update(config)
            return True

    def test_get_config(self):
        """Test getting configuration."""
        coord = self.ConfigurableCoord()
        config = coord.get_config()
        assert config["threshold"] == 100
        assert config["enabled"] is True

    def test_update_config(self):
        """Test updating configuration."""
        coord = self.ConfigurableCoord()
        result = coord.update_config({"threshold": 200})
        assert result is True
        assert coord.get_config()["threshold"] == 200


class TestRuntimeCheckable:
    """Tests for runtime_checkable protocols."""

    def test_coordinator_protocol_isinstance(self):
        """Test CoordinatorProtocol is runtime checkable."""

        class SimpleCoord:
            @property
            def name(self) -> str:
                return "SimpleCoord"

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

        coord = SimpleCoord()
        assert isinstance(coord, CoordinatorProtocol)

    def test_daemon_protocol_isinstance(self):
        """Test DaemonProtocol is runtime checkable."""

        class SimpleDaemon:
            @property
            def name(self) -> str:
                return "SimpleDaemon"

            @property
            def status(self) -> CoordinatorStatus:
                return CoordinatorStatus.RUNNING

            @property
            def is_running(self) -> bool:
                return True

            @property
            def uptime_seconds(self) -> float:
                return 0.0

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

        daemon = SimpleDaemon()
        assert isinstance(daemon, DaemonProtocol)
