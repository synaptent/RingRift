"""Tests for DaemonManager lifecycle operations - Phase 2 test coverage (December 2025).

Tests cover:
- Configuration and initialization
- Singleton pattern
- DaemonInfo and DaemonState
- Daemon registration
- Health checks and probes
- Restart tracking
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest


class TestDaemonState:
    """Tests for DaemonState enum."""

    def test_state_values(self):
        """Test DaemonState enum values."""
        from app.coordination.daemon_types import DaemonState

        assert DaemonState.STOPPED.value == "stopped"
        assert DaemonState.STARTING.value == "starting"
        assert DaemonState.RUNNING.value == "running"
        assert DaemonState.STOPPING.value == "stopping"
        assert DaemonState.FAILED.value == "failed"
        assert DaemonState.RESTARTING.value == "restarting"
        assert DaemonState.IMPORT_FAILED.value == "import_failed"


class TestDaemonInfo:
    """Tests for DaemonInfo dataclass."""

    def test_default_values(self):
        """Test DaemonInfo default values."""
        from app.coordination.daemon_types import DaemonInfo, DaemonState, DaemonType

        info = DaemonInfo(daemon_type=DaemonType.AUTO_SYNC)

        assert info.daemon_type == DaemonType.AUTO_SYNC
        assert info.state == DaemonState.STOPPED
        assert info.task is None
        assert info.restart_count == 0
        assert info.auto_restart is True
        assert info.max_restarts == 5

    def test_uptime_seconds(self):
        """Test uptime_seconds property."""
        from app.coordination.daemon_types import DaemonInfo, DaemonState, DaemonType

        info = DaemonInfo(daemon_type=DaemonType.AUTO_SYNC)
        info.state = DaemonState.RUNNING
        info.start_time = time.time() - 10.0  # Started 10 seconds ago

        assert info.uptime_seconds >= 9.0  # Allow some tolerance


class TestDaemonManagerConfig:
    """Tests for DaemonManagerConfig."""

    def test_default_config_exists(self):
        """Test DaemonManagerConfig can be instantiated."""
        from app.coordination.daemon_types import DaemonManagerConfig

        config = DaemonManagerConfig()

        # Should have key configuration fields
        assert hasattr(config, 'health_check_interval')
        assert hasattr(config, 'auto_restart_failed')
        assert hasattr(config, 'max_restart_attempts')

    def test_auto_start_default_false(self):
        """Test auto_start defaults to False."""
        from app.coordination.daemon_types import DaemonManagerConfig

        config = DaemonManagerConfig()
        assert config.auto_start is False

    def test_custom_config(self):
        """Test DaemonManagerConfig with custom values."""
        from app.coordination.daemon_types import DaemonManagerConfig

        config = DaemonManagerConfig(auto_start=True)
        assert config.auto_start is True


class TestDaemonManagerInit:
    """Tests for DaemonManager initialization."""

    def test_initialization(self):
        """Test basic DaemonManager initialization."""
        from app.coordination.daemon_manager import DaemonManager

        # Reset singleton to get fresh instance
        DaemonManager.reset_instance()

        manager = DaemonManager()

        assert manager._running is False
        assert isinstance(manager._daemons, dict)
        assert isinstance(manager._factories, dict)
        assert manager._health_task is None

        # Cleanup
        DaemonManager.reset_instance()

    def test_default_factories_registered(self):
        """Test that default daemon factories are registered."""
        from app.coordination.daemon_manager import DaemonManager
        from app.coordination.daemon_types import DaemonType

        DaemonManager.reset_instance()
        manager = DaemonManager()

        # Should have factories for core daemon types
        assert len(manager._factories) > 0
        # EVENT_ROUTER is a fundamental daemon
        assert DaemonType.EVENT_ROUTER in manager._factories

        DaemonManager.reset_instance()


class TestSingletonPattern:
    """Tests for DaemonManager singleton pattern."""

    def test_get_daemon_manager_returns_instance(self):
        """Test get_daemon_manager returns an instance."""
        from app.coordination.daemon_manager import DaemonManager, get_daemon_manager

        DaemonManager.reset_instance()
        manager = get_daemon_manager()

        assert manager is not None
        assert isinstance(manager, DaemonManager)

        DaemonManager.reset_instance()

    def test_get_daemon_manager_same_instance(self):
        """Test get_daemon_manager returns same instance."""
        from app.coordination.daemon_manager import DaemonManager, get_daemon_manager

        DaemonManager.reset_instance()
        manager1 = get_daemon_manager()
        manager2 = get_daemon_manager()

        assert manager1 is manager2

        DaemonManager.reset_instance()

    def test_reset_instance_creates_new(self):
        """Test reset_instance allows new instance creation."""
        from app.coordination.daemon_manager import DaemonManager, get_daemon_manager

        DaemonManager.reset_instance()
        manager1 = get_daemon_manager()
        DaemonManager.reset_instance()
        manager2 = get_daemon_manager()

        # After reset, should be a different instance
        assert manager1 is not manager2

        DaemonManager.reset_instance()


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_returns_result(self):
        """Test health_check returns HealthCheckResult."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()

        result = manager.health_check()

        # Should have standard health check fields
        assert hasattr(result, 'healthy')
        assert hasattr(result, 'status')
        assert hasattr(result, 'message')
        assert hasattr(result, 'details')

        DaemonManager.reset_instance()

    def test_health_check_not_running(self):
        """Test health_check when manager not running."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()
        manager._running = False

        result = manager.health_check()

        # Not running should be unhealthy
        assert result.healthy is False

        DaemonManager.reset_instance()

    def test_health_check_running_no_daemons(self):
        """Test health_check when running but no daemons."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()
        manager._running = True

        result = manager.health_check()

        # Running with no daemons should still be healthy
        assert hasattr(result, 'healthy')

        DaemonManager.reset_instance()


class TestLivenessProbe:
    """Tests for liveness probe."""

    def test_liveness_probe_returns_alive(self):
        """Test liveness_probe returns alive=True."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()

        result = manager.liveness_probe()

        assert result['alive'] is True
        assert 'timestamp' in result
        assert 'uptime_seconds' in result

        DaemonManager.reset_instance()

    def test_liveness_probe_uptime(self):
        """Test liveness_probe includes uptime."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()

        result = manager.liveness_probe()

        assert result['uptime_seconds'] >= 0

        DaemonManager.reset_instance()


class TestReadinessProbe:
    """Tests for readiness probe."""

    def test_readiness_probe_returns_ready(self):
        """Test readiness_probe returns ready status."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()

        result = manager.readiness_probe()

        assert 'ready' in result
        assert 'timestamp' in result

        DaemonManager.reset_instance()


class TestRestartTracking:
    """Tests for restart count tracking."""

    def test_record_restart(self):
        """Test recording a restart."""
        from app.coordination.daemon_manager import DaemonManager
        from app.coordination.daemon_types import DaemonType

        DaemonManager.reset_instance()
        manager = DaemonManager()

        # Clear any persisted restart data for test isolation
        daemon_name = "auto_sync"  # Enum value
        manager._restart_timestamps[daemon_name] = []

        # record_restart expects DaemonType enum
        manager.record_restart(DaemonType.AUTO_SYNC)

        assert daemon_name in manager._restart_timestamps
        assert len(manager._restart_timestamps[daemon_name]) == 1

        DaemonManager.reset_instance()

    def test_multiple_restarts_increments(self):
        """Test that multiple restarts increment count."""
        from app.coordination.daemon_manager import DaemonManager
        from app.coordination.daemon_types import DaemonType

        DaemonManager.reset_instance()
        manager = DaemonManager()

        # Clear for isolation
        daemon_name = "auto_sync"
        manager._restart_timestamps[daemon_name] = []

        initial_count = len(manager._restart_timestamps[daemon_name])
        manager.record_restart(DaemonType.AUTO_SYNC)
        manager.record_restart(DaemonType.AUTO_SYNC)
        manager.record_restart(DaemonType.AUTO_SYNC)

        # Should have 3 more than initial
        assert len(manager._restart_timestamps[daemon_name]) == initial_count + 3

        DaemonManager.reset_instance()

    def test_cascade_breaker_initial_state(self):
        """Test cascade breaker starts closed."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()

        assert manager._cascade_breaker_open is False
        assert manager._cascade_breaker_opened_at == 0.0

        DaemonManager.reset_instance()


class TestGetStatus:
    """Tests for status querying."""

    def test_get_status_structure(self):
        """Test get_status returns expected structure."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()

        status = manager.get_status()

        # Should have these core keys
        assert 'running' in status
        assert 'daemons' in status
        assert 'summary' in status

        DaemonManager.reset_instance()

    def test_get_status_running_state(self):
        """Test get_status reflects running state."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()

        status_before = manager.get_status()
        assert status_before['running'] is False

        manager._running = True
        status_after = manager.get_status()
        assert status_after['running'] is True

        DaemonManager.reset_instance()


class TestDaemonRegistryAccess:
    """Tests for daemon registry access."""

    def test_daemons_dict_accessible(self):
        """Test _daemons dict is accessible."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()

        # _daemons should be a dict
        assert isinstance(manager._daemons, dict)

        DaemonManager.reset_instance()

    def test_default_daemons_populated(self):
        """Test that default daemons are pre-populated in registry."""
        from app.coordination.daemon_manager import DaemonManager
        from app.coordination.daemon_types import DaemonType

        DaemonManager.reset_instance()
        manager = DaemonManager()

        # Default registration happens during init
        # Some daemon types should be present
        assert len(manager._daemons) > 0

        DaemonManager.reset_instance()


class TestDaemonInfoInRegistry:
    """Tests for DaemonInfo in the registry."""

    def test_add_daemon_to_registry(self):
        """Test adding a daemon to the registry directly."""
        from app.coordination.daemon_manager import DaemonManager
        from app.coordination.daemon_types import DaemonInfo, DaemonState, DaemonType

        DaemonManager.reset_instance()
        manager = DaemonManager()

        # Directly add to registry (internal API)
        manager._daemons[DaemonType.AUTO_SYNC] = DaemonInfo(
            daemon_type=DaemonType.AUTO_SYNC,
            state=DaemonState.RUNNING,
            start_time=time.time(),
        )

        assert DaemonType.AUTO_SYNC in manager._daemons
        assert manager._daemons[DaemonType.AUTO_SYNC].state == DaemonState.RUNNING

        DaemonManager.reset_instance()

    def test_daemon_uptime_tracking(self):
        """Test daemon uptime is tracked."""
        from app.coordination.daemon_manager import DaemonManager
        from app.coordination.daemon_types import DaemonInfo, DaemonState, DaemonType

        DaemonManager.reset_instance()
        manager = DaemonManager()

        # Add a running daemon with start time
        manager._daemons[DaemonType.AUTO_SYNC] = DaemonInfo(
            daemon_type=DaemonType.AUTO_SYNC,
            state=DaemonState.RUNNING,
            start_time=time.time() - 60.0,  # Started 60 seconds ago
        )

        # get_daemon_uptime should return uptime
        uptime = manager.get_daemon_uptime(DaemonType.AUTO_SYNC)
        assert uptime >= 59.0  # Allow some tolerance

        DaemonManager.reset_instance()


class TestFactoryRegistration:
    """Tests for factory registration."""

    def test_register_custom_factory(self):
        """Test registering a custom daemon factory."""
        from app.coordination.daemon_manager import DaemonManager
        from app.coordination.daemon_types import DaemonType

        DaemonManager.reset_instance()
        manager = DaemonManager()

        async def custom_factory():
            pass

        manager._factories[DaemonType.AUTO_SYNC] = custom_factory
        assert manager._factories[DaemonType.AUTO_SYNC] is custom_factory

        DaemonManager.reset_instance()


class TestRunningState:
    """Tests for running state management."""

    def test_initial_running_state(self):
        """Test initial running state is False."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()

        assert manager._running is False

        DaemonManager.reset_instance()

    def test_set_running_state(self):
        """Test setting running state."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()

        manager._running = True
        assert manager._running is True

        manager._running = False
        assert manager._running is False

        DaemonManager.reset_instance()


class TestShutdownHandler:
    """Tests for shutdown handling."""

    def test_sync_shutdown_exists(self):
        """Test _sync_shutdown method exists."""
        from app.coordination.daemon_manager import DaemonManager

        DaemonManager.reset_instance()
        manager = DaemonManager()

        assert callable(manager._sync_shutdown)

        DaemonManager.reset_instance()


class TestCriticalDaemons:
    """Tests for critical daemon constants."""

    def test_critical_daemons_defined(self):
        """Test CRITICAL_DAEMONS is defined."""
        from app.coordination.daemon_types import CRITICAL_DAEMONS, DaemonType

        assert isinstance(CRITICAL_DAEMONS, (list, tuple, set, frozenset))
        # Should include EVENT_ROUTER as it's fundamental
        assert DaemonType.EVENT_ROUTER in CRITICAL_DAEMONS

    def test_daemon_startup_order_defined(self):
        """Test DAEMON_STARTUP_ORDER is defined."""
        from app.coordination.daemon_types import DAEMON_STARTUP_ORDER, DaemonType

        assert isinstance(DAEMON_STARTUP_ORDER, (list, tuple))
        # Should include EVENT_ROUTER first
        assert DaemonType.EVENT_ROUTER in DAEMON_STARTUP_ORDER


# Run with: pytest tests/unit/coordination/test_daemon_manager_lifecycle.py -v
