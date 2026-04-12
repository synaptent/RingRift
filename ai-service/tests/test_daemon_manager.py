"""Tests for the Unified Daemon Manager.

Tests the daemon lifecycle management system that coordinates:
- Sync daemons (data, model, elo)
- Health check services
- Event watchers
- Background pipelines
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.daemon_manager import (
    DaemonInfo,
    DaemonManager,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
    get_daemon_manager,
    reset_daemon_manager,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return DaemonManagerConfig(
        auto_start=False,
        health_check_interval=1.0,  # Fast for tests
        shutdown_timeout=2.0,
        auto_restart_failed=False,  # Disable for predictable tests
        max_restart_attempts=2,
        enable_coordination_wiring=False,
    )


@pytest.fixture
def manager(config):
    """Create a fresh DaemonManager for each test."""
    reset_daemon_manager()
    instance = DaemonManager(config)
    yield instance
    reset_daemon_manager()


@pytest.fixture
def simple_daemon():
    """A simple daemon factory for testing."""
    async def daemon():
        while True:
            await asyncio.sleep(0.1)
    return daemon


@pytest.fixture
def failing_daemon():
    """A daemon that fails immediately."""
    async def daemon():
        raise RuntimeError("Intentional failure")
    return daemon


class TestDaemonManagerSingleton:
    """Tests for singleton pattern."""

    def test_get_daemon_manager_returns_singleton(self):
        """get_daemon_manager should return same instance."""
        reset_daemon_manager()
        m1 = get_daemon_manager()
        m2 = get_daemon_manager()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        """reset_daemon_manager should create new instance."""
        m1 = get_daemon_manager()
        reset_daemon_manager()
        m2 = get_daemon_manager()
        assert m1 is not m2


class TestDaemonRegistration:
    """Tests for daemon factory registration."""

    def test_register_factory(self, manager, simple_daemon):
        """Should register daemon factory."""
        manager.register_factory(
            DaemonType.SYNC_COORDINATOR,
            simple_daemon,
        )

        assert DaemonType.SYNC_COORDINATOR in manager._factories
        assert DaemonType.SYNC_COORDINATOR in manager._daemons

    def test_register_with_dependencies(self, manager, simple_daemon):
        """Should register daemon with dependencies."""
        manager.register_factory(
            DaemonType.HIGH_QUALITY_SYNC,
            simple_daemon,
            depends_on=[DaemonType.SYNC_COORDINATOR],
        )

        info = manager._daemons[DaemonType.HIGH_QUALITY_SYNC]
        assert DaemonType.SYNC_COORDINATOR in info.depends_on

    def test_register_with_options(self, manager, simple_daemon):
        """Should register daemon with custom options."""
        manager.register_factory(
            DaemonType.HEALTH_CHECK,
            simple_daemon,
            health_check_interval=30.0,
            auto_restart=False,
            max_restarts=10,
        )

        info = manager._daemons[DaemonType.HEALTH_CHECK]
        assert info.health_check_interval == 30.0
        assert info.auto_restart is False
        assert info.max_restarts == 10


class TestDaemonLifecycle:
    """Tests for daemon start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_daemon(self, manager, simple_daemon):
        """Should start a registered daemon."""
        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)

        result = await manager.start(DaemonType.SYNC_COORDINATOR)

        assert result is True
        assert manager.is_running(DaemonType.SYNC_COORDINATOR)
        info = manager._daemons[DaemonType.SYNC_COORDINATOR]
        assert info.state == DaemonState.RUNNING
        assert info.task is not None

        # Cleanup
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_start_unknown_daemon_fails(self, manager):
        """Should fail to start unregistered daemon."""
        manager._factories.clear()
        manager._daemons.clear()

        result = await manager.start(DaemonType.SYNC_COORDINATOR)
        assert result is False

    @pytest.mark.asyncio
    async def test_start_already_running_returns_true(self, manager, simple_daemon):
        """Should return True if daemon already running."""
        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)

        await manager.start(DaemonType.SYNC_COORDINATOR)
        result = await manager.start(DaemonType.SYNC_COORDINATOR)

        assert result is True

        # Cleanup
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_stop_daemon(self, manager, simple_daemon):
        """Should stop a running daemon."""
        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)

        await manager.start(DaemonType.SYNC_COORDINATOR)
        result = await manager.stop(DaemonType.SYNC_COORDINATOR)

        assert result is True
        assert not manager.is_running(DaemonType.SYNC_COORDINATOR)
        info = manager._daemons[DaemonType.SYNC_COORDINATOR]
        assert info.state == DaemonState.STOPPED
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_stop_already_stopped_returns_true(self, manager, simple_daemon):
        """Should return True if daemon already stopped."""
        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)

        result = await manager.stop(DaemonType.SYNC_COORDINATOR)
        assert result is True
        await manager.shutdown()


class TestDependencyOrdering:
    """Tests for daemon dependency management."""

    @pytest.mark.asyncio
    async def test_start_respects_dependencies(self, manager, simple_daemon):
        """Should not start daemon if dependencies not running."""
        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)
        manager.register_factory(
            DaemonType.HIGH_QUALITY_SYNC,
            simple_daemon,
            depends_on=[DaemonType.SYNC_COORDINATOR],
        )

        # Try to start dependent without starting dependency
        result = await manager.start(DaemonType.HIGH_QUALITY_SYNC)
        assert result is False

        # Start dependency first
        await manager.start(DaemonType.SYNC_COORDINATOR)
        result = await manager.start(DaemonType.HIGH_QUALITY_SYNC)
        assert result is True

        # Cleanup
        await manager.shutdown()

    def test_sort_by_dependencies(self, manager, simple_daemon):
        """Should sort daemons by dependency order."""
        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)
        manager.register_factory(
            DaemonType.HIGH_QUALITY_SYNC,
            simple_daemon,
            depends_on=[DaemonType.SYNC_COORDINATOR],
        )
        manager.register_factory(
            DaemonType.ELO_SYNC,
            simple_daemon,
            depends_on=[DaemonType.HIGH_QUALITY_SYNC],
        )

        types = [
            DaemonType.ELO_SYNC,
            DaemonType.SYNC_COORDINATOR,
            DaemonType.HIGH_QUALITY_SYNC,
        ]
        sorted_types = manager._sort_by_dependencies(types)

        # SYNC_COORDINATOR should come before HIGH_QUALITY_SYNC
        # HIGH_QUALITY_SYNC should come before ELO_SYNC
        sync_idx = sorted_types.index(DaemonType.SYNC_COORDINATOR)
        hq_idx = sorted_types.index(DaemonType.HIGH_QUALITY_SYNC)
        elo_idx = sorted_types.index(DaemonType.ELO_SYNC)

        assert sync_idx < hq_idx < elo_idx


class TestStartStopAll:
    """Tests for batch start/stop operations."""

    @pytest.mark.asyncio
    async def test_start_all(self, manager, simple_daemon):
        """Should start all registered daemons."""
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)
        manager.register_factory(DaemonType.HEALTH_CHECK, simple_daemon)

        results = await manager.start_all()

        assert results[DaemonType.SYNC_COORDINATOR] is True
        assert results[DaemonType.HEALTH_CHECK] is True
        assert manager.is_running(DaemonType.SYNC_COORDINATOR)
        assert manager.is_running(DaemonType.HEALTH_CHECK)

        # Cleanup
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_start_all_specific_types(self, manager, simple_daemon):
        """Should start only specified daemon types."""
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)
        manager.register_factory(DaemonType.HEALTH_CHECK, simple_daemon)

        results = await manager.start_all(types=[DaemonType.SYNC_COORDINATOR])

        assert DaemonType.SYNC_COORDINATOR in results
        assert DaemonType.HEALTH_CHECK not in results
        assert manager.is_running(DaemonType.SYNC_COORDINATOR)
        assert not manager.is_running(DaemonType.HEALTH_CHECK)

        # Cleanup
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_stop_all(self, manager, simple_daemon):
        """Should stop all running daemons."""
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)
        manager.register_factory(DaemonType.HEALTH_CHECK, simple_daemon)

        await manager.start_all()
        results = await manager.stop_all()

        assert results[DaemonType.SYNC_COORDINATOR] is True
        assert results[DaemonType.HEALTH_CHECK] is True
        assert not manager.is_running(DaemonType.SYNC_COORDINATOR)
        assert not manager.is_running(DaemonType.HEALTH_CHECK)
        await manager.shutdown()


class TestDaemonStatus:
    """Tests for status reporting."""

    @pytest.mark.asyncio
    async def test_get_status(self, manager, simple_daemon):
        """Should return comprehensive status."""
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)
        await manager.start(DaemonType.SYNC_COORDINATOR)

        status = manager.get_status()

        assert "running" in status
        assert "daemons" in status
        assert "summary" in status

        daemon_status = status["daemons"]["sync_coordinator"]
        assert daemon_status["state"] == "running"
        assert "uptime_seconds" in daemon_status

        # Cleanup
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_status_summary(self, manager, simple_daemon):
        """Should include summary counts."""
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)
        manager.register_factory(DaemonType.HEALTH_CHECK, simple_daemon)

        await manager.start(DaemonType.SYNC_COORDINATOR)

        status = manager.get_status()
        summary = status["summary"]

        assert summary["total"] == 2
        assert summary["running"] == 1
        assert summary["stopped"] == 1

        # Cleanup
        await manager.shutdown()

    def test_is_running(self, manager, simple_daemon):
        """is_running should reflect daemon state."""
        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)

        assert not manager.is_running(DaemonType.SYNC_COORDINATOR)


class TestDaemonInfo:
    """Tests for DaemonInfo dataclass."""

    def test_uptime_when_running(self):
        """uptime_seconds should return elapsed time when running."""
        import time
        info = DaemonInfo(
            daemon_type=DaemonType.SYNC_COORDINATOR,
            state=DaemonState.RUNNING,
            start_time=time.time() - 10,
        )

        assert info.uptime_seconds >= 9.0

    def test_uptime_when_stopped(self):
        """uptime_seconds should return 0 when stopped."""
        info = DaemonInfo(
            daemon_type=DaemonType.SYNC_COORDINATOR,
            state=DaemonState.STOPPED,
        )

        assert info.uptime_seconds == 0.0


class TestDaemonState:
    """Tests for DaemonState enum."""

    def test_all_states_exist(self):
        """All expected states should exist."""
        assert DaemonState.STOPPED
        assert DaemonState.STARTING
        assert DaemonState.RUNNING
        assert DaemonState.STOPPING
        assert DaemonState.FAILED
        assert DaemonState.RESTARTING


class TestDaemonType:
    """Tests for DaemonType enum."""

    def test_sync_daemons_exist(self):
        """Sync daemon types should exist."""
        assert DaemonType.SYNC_COORDINATOR
        assert DaemonType.HIGH_QUALITY_SYNC
        assert DaemonType.ELO_SYNC
        assert DaemonType.MODEL_SYNC

    def test_health_daemons_exist(self):
        """Health daemon types should exist."""
        assert DaemonType.HEALTH_CHECK
        assert DaemonType.CLUSTER_MONITOR
        assert DaemonType.QUEUE_MONITOR

    def test_event_daemons_exist(self):
        """Event daemon types should exist."""
        assert DaemonType.EVENT_ROUTER
        assert DaemonType.CROSS_PROCESS_POLLER


class TestShutdown:
    """Tests for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_all_daemons(self, manager, simple_daemon):
        """Shutdown should stop all running daemons."""
        manager._factories.clear()
        manager._daemons.clear()

        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)
        manager.register_factory(DaemonType.HEALTH_CHECK, simple_daemon)

        await manager.start_all()
        await manager.shutdown()

        assert not manager.is_running(DaemonType.SYNC_COORDINATOR)
        assert not manager.is_running(DaemonType.HEALTH_CHECK)

    @pytest.mark.asyncio
    async def test_shutdown_sets_event(self, manager, simple_daemon):
        """Shutdown should set the shutdown event."""
        manager.register_factory(DaemonType.SYNC_COORDINATOR, simple_daemon)

        await manager.start(DaemonType.SYNC_COORDINATOR)
        await manager.shutdown()

        assert manager._shutdown_event.is_set()


class TestAutoRestart:
    """Tests for auto-restart functionality."""

    @pytest.mark.asyncio
    async def test_restart_count_increments(self, manager, failing_daemon):
        """Restart count should increment on failure."""
        manager.config.auto_restart_failed = True
        manager.config.max_restart_attempts = 2
        manager._coordination_wired = True
        manager._factories.clear()
        manager._daemons.clear()
        manager._persisted_restart_counts.clear()
        manager._restart_timestamps.clear()
        manager._permanently_failed.clear()
        manager._degraded_daemons.clear()
        manager.register_factory(
            DaemonType.SYNC_COORDINATOR,
            failing_daemon,
            auto_restart=True,
            max_restarts=2,
        )

        # Start will trigger the daemon which fails immediately
        await manager.start(DaemonType.SYNC_COORDINATOR)

        # Give it time to fail and attempt restart
        await asyncio.sleep(0.5)

        info = manager._daemons[DaemonType.SYNC_COORDINATOR]
        # Should have attempted restarts
        assert info.restart_count > 0 or info.state == DaemonState.FAILED

        await manager.shutdown()
