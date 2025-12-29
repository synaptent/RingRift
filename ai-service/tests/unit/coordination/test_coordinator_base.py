"""Tests for coordinator_base.py - base classes and registry for coordinators."""

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.coordination.coordinator_base import (
    CoordinatorBase,
    CoordinatorStats,
    CoordinatorStatus,
    SQLitePersistenceMixin,
    SingletonMixin,
    CallbackMixin,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class ConcreteCoordinator(CoordinatorBase):
    """Concrete implementation for testing."""

    def __init__(self, name: str = "TestCoordinator"):
        super().__init__(name)
        self.start_called = False
        self.stop_called = False
        self.initialize_called = False

    async def get_stats(self) -> dict[str, Any]:
        base = await super().get_stats()
        base["custom"] = "value"
        return base

    async def _do_initialize(self) -> None:
        self.initialize_called = True

    async def _do_start(self) -> None:
        self.start_called = True

    async def _do_stop(self) -> None:
        self.stop_called = True


class FailingCoordinator(CoordinatorBase):
    """Coordinator that fails on start."""

    async def get_stats(self) -> dict[str, Any]:
        return {}

    async def _do_start(self) -> None:
        raise RuntimeError("Start failed")


# =============================================================================
# CoordinatorBase Property Tests
# =============================================================================


class TestCoordinatorBaseProperties:
    """Tests for CoordinatorBase properties."""

    def test_name_property(self):
        coord = ConcreteCoordinator("MyCoord")
        assert coord.name == "MyCoord"

    def test_name_defaults_to_class_name(self):
        coord = ConcreteCoordinator()
        assert coord.name == "TestCoordinator"

    def test_initial_status_is_initializing(self):
        coord = ConcreteCoordinator()
        assert coord.status == CoordinatorStatus.INITIALIZING

    def test_is_running_initially_false(self):
        coord = ConcreteCoordinator()
        assert coord.is_running is False

    def test_uptime_zero_when_not_started(self):
        coord = ConcreteCoordinator()
        assert coord.uptime_seconds == 0.0

    @pytest.mark.asyncio
    async def test_uptime_increases_after_start(self):
        coord = ConcreteCoordinator()
        await coord.start()
        time.sleep(0.1)
        assert coord.uptime_seconds > 0.05


# =============================================================================
# CoordinatorBase Dependency Injection Tests
# =============================================================================


class TestDependencyInjection:
    """Tests for dependency injection methods."""

    def test_set_and_get_dependency(self):
        coord = ConcreteCoordinator()
        coord.set_dependency("db", "sqlite://test")
        assert coord.get_dependency("db") == "sqlite://test"

    def test_get_nonexistent_dependency_returns_default(self):
        coord = ConcreteCoordinator()
        assert coord.get_dependency("missing") is None
        assert coord.get_dependency("missing", "default") == "default"

    def test_has_dependency(self):
        coord = ConcreteCoordinator()
        assert coord.has_dependency("cache") is False
        coord.set_dependency("cache", {})
        assert coord.has_dependency("cache") is True


# =============================================================================
# CoordinatorBase Lifecycle Tests
# =============================================================================


class TestLifecycle:
    """Tests for lifecycle methods."""

    @pytest.mark.asyncio
    async def test_initialize_changes_status_to_ready(self):
        coord = ConcreteCoordinator()
        await coord.initialize()
        assert coord.status == CoordinatorStatus.READY
        assert coord.initialize_called

    @pytest.mark.asyncio
    async def test_initialize_only_runs_once(self):
        coord = ConcreteCoordinator()
        await coord.initialize()
        coord.initialize_called = False
        await coord.initialize()  # Second call should be no-op
        assert coord.initialize_called is False

    @pytest.mark.asyncio
    async def test_start_changes_status_to_running(self):
        coord = ConcreteCoordinator()
        await coord.start()
        assert coord.status == CoordinatorStatus.RUNNING
        assert coord.is_running is True
        assert coord.start_called

    @pytest.mark.asyncio
    async def test_start_only_runs_once_when_already_running(self):
        coord = ConcreteCoordinator()
        await coord.start()
        coord.start_called = False
        await coord.start()  # Second call should be no-op
        assert coord.start_called is False

    @pytest.mark.asyncio
    async def test_start_auto_initializes(self):
        coord = ConcreteCoordinator()
        await coord.start()
        assert coord.initialize_called  # Should auto-init

    @pytest.mark.asyncio
    async def test_stop_changes_status_to_stopped(self):
        coord = ConcreteCoordinator()
        await coord.start()
        await coord.stop()
        assert coord.status == CoordinatorStatus.STOPPED
        assert coord.is_running is False
        assert coord.stop_called

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_noop(self):
        coord = ConcreteCoordinator()
        await coord.stop()
        assert coord.stop_called is False

    @pytest.mark.asyncio
    async def test_shutdown_calls_stop(self):
        coord = ConcreteCoordinator()
        await coord.start()
        await coord.shutdown()
        assert coord.stop_called
        assert coord.status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_pause_and_resume(self):
        coord = ConcreteCoordinator()
        await coord.start()

        await coord.pause()
        assert coord.status == CoordinatorStatus.PAUSED

        await coord.resume()
        assert coord.status == CoordinatorStatus.RUNNING


# =============================================================================
# CoordinatorBase Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in lifecycle."""

    @pytest.mark.asyncio
    async def test_start_failure_sets_error_status(self):
        coord = FailingCoordinator()
        with pytest.raises(RuntimeError):
            await coord.start()
        assert coord.status == CoordinatorStatus.ERROR
        assert coord.is_running is False


# =============================================================================
# CoordinatorBase Operation Recording Tests
# =============================================================================


class TestOperationRecording:
    """Tests for operation recording methods."""

    def test_record_operation_increments_count(self):
        coord = ConcreteCoordinator()
        assert coord._operations_count == 0
        coord.record_operation()
        assert coord._operations_count == 1
        coord.record_operation()
        assert coord._operations_count == 2

    def test_record_operation_updates_time(self):
        coord = ConcreteCoordinator()
        assert coord._last_operation_time == 0.0
        coord.record_operation()
        assert coord._last_operation_time > 0.0

    def test_record_error_increments_count(self):
        coord = ConcreteCoordinator()
        assert coord._errors_count == 0
        coord.record_error()
        assert coord._errors_count == 1

    def test_record_error_stores_message(self):
        coord = ConcreteCoordinator()
        coord.record_error(ValueError("test error"))
        assert "test error" in coord._last_error


# =============================================================================
# CoordinatorBase Status Dict Tests
# =============================================================================


class TestStatusDict:
    """Tests for get_status_dict method."""

    @pytest.mark.asyncio
    async def test_status_dict_contains_base_fields(self):
        coord = ConcreteCoordinator("TestCoord")
        await coord.start()
        status = coord.get_status_dict()

        assert status["name"] == "TestCoord"
        assert status["running"] is True
        assert "uptime_seconds" in status
        assert status["status"] == "running"
        assert "error_count" in status

    def test_status_dict_includes_custom_fields(self):
        coord = ConcreteCoordinator()
        status = coord.get_status_dict(jobs_pending=5, last_config="hex8_2p")

        assert status["jobs_pending"] == 5
        assert status["last_config"] == "hex8_2p"


# =============================================================================
# CoordinatorBase Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_running(self):
        coord = ConcreteCoordinator()
        await coord.start()
        result = coord.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING

    def test_health_check_unhealthy_when_not_running(self):
        coord = ConcreteCoordinator()
        result = coord.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower() or coord.status.value in result.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_degraded_on_high_error_rate(self):
        coord = ConcreteCoordinator()
        await coord.start()

        # Record 100 operations and 20 errors (20% error rate)
        for _ in range(100):
            coord.record_operation()
        for _ in range(20):
            coord.record_error()

        result = coord.health_check()
        # With >10% error rate, should be degraded
        assert result.status == CoordinatorStatus.DEGRADED or not result.healthy

    @pytest.mark.asyncio
    async def test_health_check_includes_details(self):
        coord = ConcreteCoordinator("DetailedCoord")
        await coord.start()
        result = coord.health_check()

        assert "name" in result.details
        assert result.details["name"] == "DetailedCoord"
        assert "is_running" in result.details
        assert "uptime_seconds" in result.details


# =============================================================================
# SingletonMixin Tests
# =============================================================================


class TestSingletonMixin:
    """Tests for SingletonMixin."""

    def test_singleton_returns_same_instance(self):
        class MySingleton(SingletonMixin):
            pass

        # Reset for clean test
        MySingleton._clear_instance()

        a = MySingleton._get_or_create_instance()
        b = MySingleton._get_or_create_instance()
        assert a is b

    def test_reset_instance_clears_singleton(self):
        class ResetSingleton(SingletonMixin):
            pass

        # Reset for clean test
        ResetSingleton._clear_instance()

        a = ResetSingleton._get_or_create_instance()
        ResetSingleton._clear_instance()
        b = ResetSingleton._get_or_create_instance()
        assert a is not b


# =============================================================================
# CallbackMixin Tests
# =============================================================================


class TestCallbackMixin:
    """Tests for CallbackMixin."""

    @pytest.mark.asyncio
    async def test_register_and_call_callback(self):
        class MyClass(CallbackMixin):
            def __init__(self):
                super().__init__()
                self._callbacks = {}

        obj = MyClass()
        results = []

        def my_callback(value):
            results.append(value)

        obj.register_callback("test_event", my_callback)
        await obj.invoke_callbacks("test_event", "hello")

        assert "hello" in results

    @pytest.mark.asyncio
    async def test_unregister_callback(self):
        class MyClass(CallbackMixin):
            def __init__(self):
                super().__init__()
                self._callbacks = {}

        obj = MyClass()
        results = []

        def my_callback(value):
            results.append(value)

        obj.register_callback("event", my_callback)
        obj.unregister_callback("event", my_callback)
        await obj.invoke_callbacks("event", "value")

        assert len(results) == 0


# =============================================================================
# Original Import Tests (Preserved)
# =============================================================================


class TestCoordinatorBaseImports:
    """Test that coordinator_base provides correct imports."""

    def test_import_enums_and_dataclasses(self):
        """Test importing enums and dataclasses."""
        from app.coordination.coordinator_base import (
            CoordinatorStats,
            CoordinatorStatus,
        )
        assert CoordinatorStatus is not None
        assert CoordinatorStats is not None

    def test_import_protocols(self):
        """Test importing protocols."""
        from app.coordination.coordinator_base import (
            CoordinatorProtocol,
        )
        assert CoordinatorProtocol is not None

    def test_import_base_classes(self):
        """Test importing base classes."""
        from app.coordination.coordinator_base import (
            CoordinatorBase,
        )
        assert CoordinatorBase is not None

    def test_import_mixins(self):
        """Test importing mixins."""
        from app.coordination.coordinator_base import (
            CallbackMixin,
            SingletonMixin,
            SQLitePersistenceMixin,
        )
        assert SQLitePersistenceMixin is not None
        assert SingletonMixin is not None
        assert CallbackMixin is not None

    def test_import_registry(self):
        """Test importing registry."""
        from app.coordination.coordinator_base import (
            CoordinatorRegistry,
            get_coordinator_registry,
        )
        assert CoordinatorRegistry is not None
        assert get_coordinator_registry is not None

    def test_import_helper_functions(self):
        """Test importing new helper functions."""
        from app.coordination.coordinator_base import (
            get_all_coordinators,
            get_coordinator_statuses,
            shutdown_all_coordinators,
        )
        assert callable(get_all_coordinators)
        assert callable(get_coordinator_statuses)
        assert callable(shutdown_all_coordinators)


class TestCoordinatorStatus:
    """Test CoordinatorStatus enum."""

    def test_status_values_exist(self):
        """Test that expected status values exist."""
        from app.coordination.coordinator_base import CoordinatorStatus

        assert hasattr(CoordinatorStatus, 'INITIALIZING')
        assert hasattr(CoordinatorStatus, 'READY')
        assert hasattr(CoordinatorStatus, 'RUNNING')
        assert hasattr(CoordinatorStatus, 'PAUSED')
        assert hasattr(CoordinatorStatus, 'DRAINING')
        assert hasattr(CoordinatorStatus, 'ERROR')
        assert hasattr(CoordinatorStatus, 'STOPPED')

    def test_status_is_string_enum(self):
        """Test that status values are strings."""
        from app.coordination.coordinator_base import CoordinatorStatus

        assert CoordinatorStatus.RUNNING.value == "running"
        assert CoordinatorStatus.STOPPED.value == "stopped"


class TestCoordinatorStats:
    """Test CoordinatorStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        from app.coordination.coordinator_base import (
            CoordinatorStats,
            CoordinatorStatus,
        )

        stats = CoordinatorStats()
        assert stats.status == CoordinatorStatus.INITIALIZING
        assert stats.uptime_seconds == 0.0
        assert stats.operations_count == 0
        assert stats.errors_count == 0

    def test_to_dict(self):
        """Test converting to dict."""
        from app.coordination.coordinator_base import (
            CoordinatorStats,
            CoordinatorStatus,
        )

        stats = CoordinatorStats(
            status=CoordinatorStatus.RUNNING,
            uptime_seconds=120.5,
            operations_count=50,
        )

        data = stats.to_dict()
        assert isinstance(data, dict)
        assert data["status"] == "running"
        assert data["uptime_seconds"] == 120.5
        assert data["operations_count"] == 50


class TestGetAllCoordinators:
    """Test get_all_coordinators function."""

    def test_returns_dict(self):
        """Test that function returns a dict."""
        from app.coordination.coordinator_base import get_all_coordinators

        result = get_all_coordinators()
        assert isinstance(result, dict)

    def test_returns_copy(self):
        """Test that function returns a copy, not the original."""
        from app.coordination.coordinator_base import (
            get_all_coordinators,
            get_coordinator_registry,
        )

        result1 = get_all_coordinators()
        result2 = get_all_coordinators()

        # Should be equal but not the same object
        assert result1 == result2
        # Modifying one shouldn't affect the other
        result1["test"] = "value"
        assert "test" not in result2


class TestGetCoordinatorStatuses:
    """Test get_coordinator_statuses function."""

    def test_returns_dict(self):
        """Test that function returns a dict."""
        from app.coordination.coordinator_base import get_coordinator_statuses

        result = get_coordinator_statuses()
        assert isinstance(result, dict)


class TestCoordinatorRegistry:
    """Test CoordinatorRegistry class."""

    def test_singleton_pattern(self):
        """Test that get_coordinator_registry returns singleton."""
        from app.coordination.coordinator_base import get_coordinator_registry

        r1 = get_coordinator_registry()
        r2 = get_coordinator_registry()
        assert r1 is r2

    def test_get_health_summary(self):
        """Test getting health summary."""
        from app.coordination.coordinator_base import get_coordinator_registry

        registry = get_coordinator_registry()
        summary = registry.get_health_summary()

        assert isinstance(summary, dict)
        assert "healthy" in summary
        assert "coordinator_count" in summary
        assert "coordinators" in summary
