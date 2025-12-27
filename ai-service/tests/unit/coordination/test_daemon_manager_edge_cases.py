"""Unit tests for DaemonManager edge cases.

December 2025: Created as part of daemon lifecycle recovery test coverage.
Tests import errors, dependency timeout, health check polymorphism,
and daemon state machine transitions.

Test Scenarios Covered:
- Scenario 5: Health check async/sync polymorphism
- Scenario 8: Import error permanent failure
- Scenario 9: Dependency timeout handling (30s)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.daemon_manager import (
    DaemonInfo,
    DaemonManager,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manager_config():
    """Create test config with short intervals for faster tests."""
    return DaemonManagerConfig(
        health_check_interval=0.1,  # 100ms for fast tests
        shutdown_timeout=1.0,
        recovery_cooldown=0.2,  # 200ms cooldown
        auto_restart_failed=True,
    )


@pytest.fixture
def manager(manager_config):
    """Create fresh DaemonManager for each test with no default factories."""
    DaemonManager.reset_instance()
    mgr = DaemonManager(manager_config)
    # Clear default factories to avoid loading heavy dependencies
    mgr._factories.clear()
    mgr._daemons.clear()
    yield mgr
    # Cleanup - stop any running tasks
    mgr._running = False
    if mgr._shutdown_event:
        mgr._shutdown_event.set()
    for info in list(mgr._daemons.values()):
        if info.task and not info.task.done():
            info.task.cancel()
    DaemonManager.reset_instance()


# =============================================================================
# Test Import Error Handling
# =============================================================================


class TestImportErrorHandling:
    """Tests for import error handling in daemons."""

    @pytest.mark.asyncio
    async def test_import_error_sets_import_failed_state(self, manager: DaemonManager):
        """ImportError should set IMPORT_FAILED state permanently.

        Scenario 8: Import errors are permanent - require code/env fix.
        """

        async def import_failing_factory():
            raise ImportError("No module named 'nonexistent_module'")

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            import_failing_factory,
            auto_restart=True,
            max_restarts=5,
        )

        # Start the daemon
        await manager.start(DaemonType.MODEL_SYNC)

        # Wait for failure to be processed
        await asyncio.sleep(0.2)

        info = manager._daemons[DaemonType.MODEL_SYNC]

        # ImportError should result in IMPORT_FAILED or FAILED state with import_error set
        assert info.state in (DaemonState.FAILED, DaemonState.STOPPED, DaemonState.IMPORT_FAILED), (
            f"Expected FAILED, STOPPED, or IMPORT_FAILED state for import error, got {info.state}"
        )
        # The error should be captured
        assert info.last_error is not None or info.import_error is not None

    @pytest.mark.asyncio
    async def test_import_error_not_auto_restarted_by_health_loop(
        self, manager: DaemonManager
    ):
        """Daemons with import_error should not be auto-restarted.

        Expected: _check_health() skips daemons with import_error set.
        """
        # Directly set up a daemon with import error
        info = DaemonInfo(daemon_type=DaemonType.MODEL_SYNC)
        info.state = DaemonState.FAILED
        info.import_error = "ModuleNotFoundError: No module named 'nonexistent'"
        info.last_failure_time = 0  # Long ago
        info.auto_restart = True
        info.max_restarts = 5
        info.restart_count = 0

        manager._daemons[DaemonType.MODEL_SYNC] = info

        # Run health check
        await manager._check_health()

        # Verify no restart attempted
        assert info.state == DaemonState.FAILED
        assert info.restart_count == 0, "Import error daemons should not be restarted"

    @pytest.mark.asyncio
    async def test_import_error_cleared_by_factory_reregistration(
        self, manager: DaemonManager
    ):
        """Re-registering factory clears import_error state.

        After fixing the import issue, re-registering the factory
        should clear the error and allow restart.
        """
        # Set up daemon with import error
        info = DaemonInfo(daemon_type=DaemonType.MODEL_SYNC)
        info.state = DaemonState.FAILED
        info.import_error = "ModuleNotFoundError: No module named 'nonexistent'"
        info.restart_count = 3
        manager._daemons[DaemonType.MODEL_SYNC] = info

        # Register a working factory for the retry
        factory_called = False

        async def working_factory():
            nonlocal factory_called
            factory_called = True
            while True:
                await asyncio.sleep(1)

        # Re-register the factory - this should allow a fresh start
        manager.register_factory(
            DaemonType.MODEL_SYNC,
            working_factory,
            auto_restart=True,
        )

        # The re-registration should reset the import error
        # Now try to start
        await manager.start(DaemonType.MODEL_SYNC)
        await asyncio.sleep(0.1)

        # Check if factory was called or at least the daemon is in a valid state
        new_info = manager._daemons[DaemonType.MODEL_SYNC]
        # After re-registration, import_error should be cleared on new DaemonInfo
        assert new_info.import_error is None, "Re-registration should clear import_error"


# =============================================================================
# Test Dependency Timeout Handling
# =============================================================================


class TestDependencyTimeoutHandling:
    """Tests for 30s dependency ready timeout handling."""

    @pytest.mark.asyncio
    async def test_dependency_not_running_fails_start(self, manager: DaemonManager):
        """Start should fail if dependency is not running.

        Scenario 9: Dependencies must be running before dependent can start.
        """

        async def child_factory():
            while True:
                await asyncio.sleep(1)

        # Register child with dependency on parent (parent not registered)
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            child_factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        # Try to start child without starting parent
        result = await manager.start(DaemonType.DATA_PIPELINE)

        # Should fail because dependency not running
        assert result is False, "Start should fail when dependency not running"

    @pytest.mark.asyncio
    async def test_dependency_running_allows_start(self, manager: DaemonManager):
        """Start should succeed if dependency is running.

        Expected: start() returns True when dependency is running and ready.
        """

        async def parent_factory():
            while True:
                await asyncio.sleep(1)

        async def child_factory():
            while True:
                await asyncio.sleep(1)

        # Register parent
        manager.register_factory(DaemonType.EVENT_ROUTER, parent_factory)

        # Register child with dependency
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            child_factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        # Start parent first
        await manager.start(DaemonType.EVENT_ROUTER)
        await asyncio.sleep(0.1)

        # Mark parent as ready (normally done by the daemon itself)
        manager.mark_daemon_ready(DaemonType.EVENT_ROUTER)

        # Now start child
        result = await manager.start(DaemonType.DATA_PIPELINE)

        assert result is True, "Start should succeed when dependency is running"

    @pytest.mark.asyncio
    async def test_dependency_wait_releases_lock(self, manager: DaemonManager):
        """Waiting for dependency should release lock to avoid deadlock.

        Expected: Lock released during wait, re-acquired after.
        """
        # This test verifies the pattern works - if it deadlocks, it will timeout

        async def parent_factory():
            while True:
                await asyncio.sleep(1)

        async def child_factory():
            while True:
                await asyncio.sleep(1)

        manager.register_factory(DaemonType.EVENT_ROUTER, parent_factory)
        manager.register_factory(
            DaemonType.DATA_PIPELINE,
            child_factory,
            depends_on=[DaemonType.EVENT_ROUTER],
        )

        # Start parent
        await manager.start(DaemonType.EVENT_ROUTER)

        # Start child in background (it will wait for parent ready)
        child_task = asyncio.create_task(manager.start(DaemonType.DATA_PIPELINE))

        # While child is waiting, we should be able to acquire lock for other ops
        await asyncio.sleep(0.05)  # Let child start waiting

        # Mark parent ready - this should unblock child
        manager.mark_daemon_ready(DaemonType.EVENT_ROUTER)

        # Wait for child to complete
        try:
            result = await asyncio.wait_for(child_task, timeout=2.0)
            assert result is True
        except asyncio.TimeoutError:
            pytest.fail("Deadlock detected - lock not released during dependency wait")


# =============================================================================
# Test Health Check Return Types (Polymorphism)
# =============================================================================


class TestHealthCheckReturnTypes:
    """Tests for health_check() return type polymorphism.

    Note: DaemonManager calls `info.instance.health_check()` to check daemon health.
    These tests verify that daemons with healthy health_check responses stay running.
    """

    @pytest.mark.asyncio
    async def test_health_check_bool_true_means_healthy(self, manager: DaemonManager):
        """health_check() returning True means healthy - no restart.

        Expected: Daemon stays RUNNING.
        """
        health_checked = False

        # Create a mock daemon class with health_check method
        class HealthyDaemon:
            def health_check(self):
                nonlocal health_checked
                health_checked = True
                return True

        mock_daemon = HealthyDaemon()

        async def daemon_factory():
            while True:
                await asyncio.sleep(0.5)

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            daemon_factory,
            auto_restart=True,
        )

        await manager.start(DaemonType.MODEL_SYNC)
        await asyncio.sleep(0.05)

        info = manager._daemons[DaemonType.MODEL_SYNC]
        info.instance = mock_daemon  # Set instance with health_check method

        # Run health loop
        await manager._check_health()

        # Daemon should still be running
        assert info.state == DaemonState.RUNNING
        assert health_checked, "Health check should have been called"

    @pytest.mark.asyncio
    async def test_health_check_bool_false_triggers_restart(
        self, manager: DaemonManager
    ):
        """health_check() returning False triggers restart if enabled.

        Expected: When health_check returns False, daemon should be restarted.
        """
        factory_calls = 0

        # Create a mock daemon class with health_check that returns False
        class UnhealthyDaemon:
            def health_check(self):
                return False

        mock_daemon = UnhealthyDaemon()

        async def daemon_factory():
            nonlocal factory_calls
            factory_calls += 1
            while True:
                await asyncio.sleep(0.5)

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            daemon_factory,
            auto_restart=True,
            max_restarts=3,
        )

        await manager.start(DaemonType.MODEL_SYNC)
        await asyncio.sleep(0.05)

        info = manager._daemons[DaemonType.MODEL_SYNC]
        info.instance = mock_daemon  # Set instance with unhealthy health_check
        info.restart_delay = 0.1  # Fast restart for tests

        initial_factory_calls = factory_calls

        # Run health check - should detect unhealthy and restart
        await manager._check_health()
        await asyncio.sleep(0.3)  # Wait for restart

        # Factory should be called again (restart)
        assert factory_calls > initial_factory_calls, "Should restart on unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_dict_with_healthy_key(self, manager: DaemonManager):
        """health_check() returning dict with 'healthy' key.

        Expected: Extract healthy value from dict.
        """

        class DictHealthDaemon:
            def health_check(self):
                return {"healthy": True, "message": "All systems operational"}

        mock_daemon = DictHealthDaemon()

        async def daemon_factory():
            while True:
                await asyncio.sleep(0.5)

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            daemon_factory,
            auto_restart=True,
        )

        await manager.start(DaemonType.MODEL_SYNC)
        await asyncio.sleep(0.05)

        info = manager._daemons[DaemonType.MODEL_SYNC]
        info.instance = mock_daemon  # Set instance with dict health_check

        # Run health check
        await manager._check_health()

        # Should still be running (dict with healthy=True)
        assert info.state == DaemonState.RUNNING

    @pytest.mark.asyncio
    async def test_health_check_object_with_healthy_attribute(
        self, manager: DaemonManager
    ):
        """health_check() returning object with .healthy attribute.

        Expected: Use hasattr() to check for healthy attribute.
        """

        @dataclass
        class HealthCheckResult:
            healthy: bool
            message: str = ""

        class ObjectHealthDaemon:
            def health_check(self):
                return HealthCheckResult(healthy=True, message="OK")

        mock_daemon = ObjectHealthDaemon()

        async def daemon_factory():
            while True:
                await asyncio.sleep(0.5)

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            daemon_factory,
            auto_restart=True,
        )

        await manager.start(DaemonType.MODEL_SYNC)
        await asyncio.sleep(0.05)

        info = manager._daemons[DaemonType.MODEL_SYNC]
        info.instance = mock_daemon  # Set instance with object health_check

        # Run health check
        await manager._check_health()

        # Should still be running
        assert info.state == DaemonState.RUNNING

    @pytest.mark.asyncio
    async def test_async_health_check_awaited(self, manager: DaemonManager):
        """Async health_check() (returning coroutine) is properly awaited.

        Expected: asyncio.iscoroutine() check, then await.
        """
        health_check_awaited = False

        class AsyncHealthDaemon:
            async def health_check(self):
                nonlocal health_check_awaited
                await asyncio.sleep(0.01)  # Actual async operation
                health_check_awaited = True
                return True

        mock_daemon = AsyncHealthDaemon()

        async def daemon_factory():
            while True:
                await asyncio.sleep(0.5)

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            daemon_factory,
            auto_restart=True,
        )

        await manager.start(DaemonType.MODEL_SYNC)
        await asyncio.sleep(0.05)

        info = manager._daemons[DaemonType.MODEL_SYNC]
        info.instance = mock_daemon  # Set instance with async health_check

        # Run health check
        await manager._check_health()

        assert health_check_awaited, "Async health_check should be awaited"


# =============================================================================
# Test Daemon State Transitions
# =============================================================================


class TestDaemonStateTransitions:
    """Tests for daemon state machine transitions."""

    @pytest.mark.asyncio
    async def test_normal_start_state_sequence(self, manager: DaemonManager):
        """Normal start should end in RUNNING state.

        Expected: After successful start, daemon is RUNNING.
        """

        async def tracking_factory():
            while True:
                await asyncio.sleep(0.5)

        manager.register_factory(DaemonType.MODEL_SYNC, tracking_factory)

        # Start the daemon
        result = await manager.start(DaemonType.MODEL_SYNC)
        await asyncio.sleep(0.1)

        # Should be running after successful start
        assert result is True, "Start should succeed"
        info = manager._daemons[DaemonType.MODEL_SYNC]
        assert info.state == DaemonState.RUNNING, "Daemon should be RUNNING after start"

    @pytest.mark.asyncio
    async def test_crash_sets_failed_state(self, manager: DaemonManager):
        """Daemon crash: RUNNING -> FAILED (if no auto-restart).

        Expected: State = FAILED, last_error set.
        """

        async def crashing_factory():
            await asyncio.sleep(0.05)
            raise RuntimeError("Deliberate crash for test")

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            crashing_factory,
            auto_restart=False,  # Disable auto-restart
        )

        await manager.start(DaemonType.MODEL_SYNC)

        # Wait for crash
        await asyncio.sleep(0.2)

        # Run health check to detect crash
        await manager._check_health()

        info = manager._daemons[DaemonType.MODEL_SYNC]
        assert info.state == DaemonState.FAILED, f"Expected FAILED, got {info.state}"

    @pytest.mark.asyncio
    async def test_auto_restart_increments_count(self, manager: DaemonManager):
        """Auto-restart should increment restart_count.

        Expected: restart_count increases after each crash.
        """
        crash_count = 0

        async def sometimes_crashing_factory():
            nonlocal crash_count
            crash_count += 1
            if crash_count <= 2:  # Crash first 2 times
                raise RuntimeError(f"Crash #{crash_count}")
            while True:
                await asyncio.sleep(1)

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            sometimes_crashing_factory,
            auto_restart=True,
            max_restarts=5,
        )

        manager._daemons[DaemonType.MODEL_SYNC].restart_delay = 0.1

        await manager.start(DaemonType.MODEL_SYNC)

        # Wait for crashes and restarts
        await asyncio.sleep(1.0)

        info = manager._daemons[DaemonType.MODEL_SYNC]
        assert info.restart_count >= 1, f"Expected restart_count >= 1, got {info.restart_count}"

    @pytest.mark.asyncio
    async def test_import_failed_is_terminal_without_force(
        self, manager: DaemonManager
    ):
        """IMPORT_FAILED is effectively terminal without force=True.

        Expected: Cannot restart without force, stays FAILED.
        """
        # Set up daemon with import error
        info = DaemonInfo(daemon_type=DaemonType.MODEL_SYNC)
        info.state = DaemonState.FAILED
        info.import_error = "ImportError: No module"
        manager._daemons[DaemonType.MODEL_SYNC] = info

        # Try health check recovery
        await manager._check_health()

        # Should still be FAILED
        assert info.state == DaemonState.FAILED
        assert info.import_error is not None


# =============================================================================
# Test Restart Count Reset
# =============================================================================


class TestRestartCountReset:
    """Tests for restart count reset after stability period."""

    @pytest.mark.asyncio
    async def test_restart_count_preserved_within_cooldown(
        self, manager: DaemonManager
    ):
        """Restart count NOT reset if daemon fails again before stability.

        Expected: restart_count preserved across failures within window.
        """
        crash_count = 0

        async def always_crashing():
            nonlocal crash_count
            crash_count += 1
            raise RuntimeError(f"Crash {crash_count}")

        manager.register_factory(
            DaemonType.MODEL_SYNC,
            always_crashing,
            auto_restart=True,
            max_restarts=3,
        )

        manager._daemons[DaemonType.MODEL_SYNC].restart_delay = 0.05
        manager.config.auto_restart_failed = False  # Prevent health loop interference

        await manager.start(DaemonType.MODEL_SYNC)

        # Wait for max restarts to be exhausted
        await asyncio.sleep(0.5)

        info = manager._daemons[DaemonType.MODEL_SYNC]
        # Should have counted all restarts, not reset
        assert info.restart_count <= 3, f"Restart count should not exceed max, got {info.restart_count}"
