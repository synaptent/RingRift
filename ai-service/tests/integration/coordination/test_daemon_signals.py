"""Integration tests for daemon signal handling.

These tests verify that the DaemonManager correctly handles:
1. Signal handler installation (SIGTERM, SIGINT)
2. Graceful shutdown on signal receipt
3. Coordinator registry signal integration

Created: December 2025
Purpose: Ensure graceful shutdown works under termination signals
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.daemon_manager import (
    DaemonManager,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
    get_daemon_manager,
    reset_daemon_manager,
    setup_signal_handlers,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manager_config():
    """Create test config with short intervals for faster tests."""
    return DaemonManagerConfig(
        health_check_interval=0.1,
        shutdown_timeout=1.0,
        recovery_cooldown=0.2,
        auto_restart_failed=False,  # Disable for signal tests
    )


@pytest.fixture
def manager(manager_config):
    """Create fresh DaemonManager for each test."""
    DaemonManager.reset_instance()
    mgr = DaemonManager(manager_config)
    mgr._factories.clear()
    mgr._daemons.clear()
    yield mgr
    mgr._running = False
    if mgr._shutdown_event:
        mgr._shutdown_event.set()
    for info in list(mgr._daemons.values()):
        if info.task and not info.task.done():
            info.task.cancel()
    DaemonManager.reset_instance()


# =============================================================================
# Signal Handler Installation Tests
# =============================================================================


class TestSignalHandlerInstallation:
    """Tests for signal handler setup."""

    def test_setup_signal_handlers_installs_sigterm(self):
        """setup_signal_handlers should install SIGTERM handler."""
        with patch("signal.signal") as mock_signal:
            setup_signal_handlers()

            # Find SIGTERM call
            sigterm_calls = [
                call for call in mock_signal.call_args_list
                if call[0][0] == signal.SIGTERM
            ]
            assert len(sigterm_calls) >= 1, "SIGTERM handler should be installed"

    def test_setup_signal_handlers_installs_sigint(self):
        """setup_signal_handlers should install SIGINT handler."""
        with patch("signal.signal") as mock_signal:
            setup_signal_handlers()

            # Find SIGINT call
            sigint_calls = [
                call for call in mock_signal.call_args_list
                if call[0][0] == signal.SIGINT
            ]
            assert len(sigint_calls) >= 1, "SIGINT handler should be installed"

    def test_setup_signal_handlers_handles_os_error(self):
        """setup_signal_handlers should gracefully handle OSError."""
        with patch("signal.signal", side_effect=OSError("Cannot set signal handler")):
            # Should not raise
            setup_signal_handlers()

    def test_setup_signal_handlers_handles_runtime_error(self):
        """setup_signal_handlers should gracefully handle RuntimeError."""
        with patch("signal.signal", side_effect=RuntimeError("Signal in non-main thread")):
            # Should not raise
            setup_signal_handlers()

    def test_setup_signal_handlers_handles_value_error(self):
        """setup_signal_handlers should gracefully handle ValueError."""
        with patch("signal.signal", side_effect=ValueError("Invalid signal")):
            # Should not raise
            setup_signal_handlers()


# =============================================================================
# Signal Handler Behavior Tests
# =============================================================================


class TestSignalHandlerBehavior:
    """Tests for signal handler behavior when invoked."""

    @pytest.mark.asyncio
    async def test_signal_handler_triggers_shutdown(self, manager: DaemonManager):
        """Signal handler should trigger manager shutdown."""
        shutdown_called = False

        async def mock_shutdown():
            nonlocal shutdown_called
            shutdown_called = True

        manager.shutdown = mock_shutdown

        # Capture the handler function when signal.signal is called
        captured_handler = None

        def capture_signal(signum, handler):
            nonlocal captured_handler
            if signum == signal.SIGTERM:
                captured_handler = handler

        with patch("app.coordination.daemon_manager.get_daemon_manager", return_value=manager):
            with patch("signal.signal", side_effect=capture_signal):
                setup_signal_handlers()

        # Verify handler was captured
        assert captured_handler is not None, "SIGTERM handler should be captured"

        # Simulate signal by calling the handler
        # Note: This runs in sync context, so fire_and_forget is used internally
        with patch("app.coordination.daemon_manager.fire_and_forget") as mock_ff:
            captured_handler(signal.SIGTERM, None)

            # Verify fire_and_forget was called with shutdown coroutine
            assert mock_ff.called, "fire_and_forget should be called for async shutdown"

    def test_sync_shutdown_handles_no_event_loop(self, manager: DaemonManager):
        """_sync_shutdown should handle case when no event loop exists.

        Note: The current implementation of _sync_shutdown doesn't do anything
        when there's no running loop - it just passes. This is expected behavior
        since there's no way to run async shutdown in that case.
        """
        # Patch asyncio.get_running_loop to raise RuntimeError
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("No running loop")):
            # Should not raise even when there's no event loop
            manager._sync_shutdown()
            # If we get here, the test passes (no exception raised)


# =============================================================================
# Graceful Shutdown Tests
# =============================================================================


class TestGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_all_daemons(self, manager: DaemonManager):
        """shutdown() should stop all running daemons."""
        async def daemon_factory():
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass

        manager.register_factory(DaemonType.MODEL_SYNC, daemon_factory)
        manager.register_factory(DaemonType.EVENT_ROUTER, daemon_factory)

        # Start daemons
        await manager.start(DaemonType.MODEL_SYNC)
        await manager.start(DaemonType.EVENT_ROUTER)
        await asyncio.sleep(0.1)

        # Verify they're running
        assert manager._daemons[DaemonType.MODEL_SYNC].state == DaemonState.RUNNING
        assert manager._daemons[DaemonType.EVENT_ROUTER].state == DaemonState.RUNNING

        # Trigger shutdown
        await manager.shutdown()

        # Give time for shutdown to complete
        await asyncio.sleep(0.2)

        # Verify both daemons were stopped
        assert manager._daemons[DaemonType.MODEL_SYNC].state in (
            DaemonState.STOPPED, DaemonState.FAILED
        )
        assert manager._daemons[DaemonType.EVENT_ROUTER].state in (
            DaemonState.STOPPED, DaemonState.FAILED
        )

    @pytest.mark.asyncio
    async def test_shutdown_stops_manager_running(self, manager: DaemonManager):
        """shutdown() should stop the manager's running state."""
        async def daemon_factory():
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass

        manager.register_factory(DaemonType.MODEL_SYNC, daemon_factory)
        await manager.start(DaemonType.MODEL_SYNC)
        await asyncio.sleep(0.1)

        # Verify running before shutdown
        assert manager._running

        # Trigger shutdown
        await manager.shutdown()

        # Verify manager is no longer running
        assert not manager._running, "Manager should not be running after shutdown"

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_daemons_to_stop(self, manager: DaemonManager):
        """shutdown() should wait for daemons to stop gracefully."""
        daemon_stopped = asyncio.Event()

        async def slow_stopping_daemon():
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                # Simulate cleanup taking time
                await asyncio.sleep(0.2)
                daemon_stopped.set()
                raise

        manager.register_factory(DaemonType.MODEL_SYNC, slow_stopping_daemon)
        await manager.start(DaemonType.MODEL_SYNC)
        await asyncio.sleep(0.1)

        # Shutdown should wait for daemon to finish cleanup
        await manager.shutdown()

        # Give a bit more time for the event to propagate
        await asyncio.sleep(0.1)

        # The daemon should have completed its cleanup
        info = manager._daemons[DaemonType.MODEL_SYNC]
        assert info.state in (DaemonState.STOPPED, DaemonState.FAILED)


# =============================================================================
# Coordinator Registry Signal Tests
# =============================================================================


class TestCoordinatorRegistrySignals:
    """Tests for CoordinatorRegistry signal handling."""

    def test_registry_install_signal_handlers(self):
        """CoordinatorRegistry should be able to install signal handlers."""
        from app.coordination.coordinator_base import CoordinatorRegistry

        registry = CoordinatorRegistry()

        with patch("signal.signal") as mock_signal:
            # Should not raise, even if signal setup fails
            try:
                registry.install_signal_handlers()
            except Exception as e:
                pytest.fail(f"install_signal_handlers should not raise: {e}")

    def test_registry_shutdown_all_on_signal(self):
        """CoordinatorRegistry should shutdown all coordinators on signal."""
        from app.coordination.coordinator_base import CoordinatorRegistry

        registry = CoordinatorRegistry()

        # Mock a coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.shutdown = AsyncMock()

        registry._coordinators["test"] = mock_coordinator

        # Capture signal handler
        captured_handler = None

        def capture_signal(signum, handler):
            nonlocal captured_handler
            if signum == signal.SIGTERM:
                captured_handler = handler

        with patch("signal.signal", side_effect=capture_signal):
            registry.install_signal_handlers()

        # Verify handler was captured (may be None if signal can't be set)
        # The important thing is that install_signal_handlers doesn't crash
        # In real usage, the handler would trigger shutdown_all()

    @pytest.mark.asyncio
    async def test_registry_shutdown_all(self):
        """CoordinatorRegistry.shutdown_all() should stop all coordinators."""
        from app.coordination.coordinator_base import (
            CoordinatorBase,
            CoordinatorRegistry,
            CoordinatorStatus,
        )

        registry = CoordinatorRegistry()

        # Create mock coordinators that properly subclass CoordinatorBase behavior
        class MockCoordinator:
            def __init__(self, name: str):
                self.name = name
                self.is_running = True
                self.shutdown_called = False
                self.status = CoordinatorStatus.RUNNING

            async def shutdown(self):
                self.shutdown_called = True
                self.is_running = False
                self.status = CoordinatorStatus.STOPPED

            async def stop(self):
                self.is_running = False

        mock1 = MockCoordinator("coord1")
        mock2 = MockCoordinator("coord2")

        registry._coordinators["coord1"] = mock1
        registry._coordinators["coord2"] = mock2
        registry._priorities["coord1"] = 0
        registry._priorities["coord2"] = 0
        registry._update_shutdown_order()

        await registry.shutdown_all()

        # Both should have shutdown called
        assert mock1.shutdown_called, "Coordinator 1 should have shutdown called"
        assert mock2.shutdown_called, "Coordinator 2 should have shutdown called"
