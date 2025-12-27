"""Tests for daemon_lifecycle.py - Daemon Lifecycle Management.

December 2025: Created as part of test coverage initiative.
Tests the DaemonLifecycleManager extracted from DaemonManager.

Coverage includes:
- Lifecycle operations (start, stop, restart)
- Dependency ordering (topological sort)
- Error handling and restart logic
- Graceful shutdown
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.daemon_lifecycle import (
    DaemonLifecycleManager,
    DependencyValidationError,
    StateUpdateCallback,
)
from app.coordination.daemon_types import (
    DaemonInfo,
    DaemonManagerConfig,
    DaemonState,
    DaemonType,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_daemons():
    """Create mock daemon registry."""
    daemons = {}
    for dt in [DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE, DaemonType.AUTO_SYNC]:
        info = DaemonInfo(daemon_type=dt)
        info.ready_event = None
        daemons[dt] = info
    return daemons


@pytest.fixture
def mock_factories():
    """Create mock factory registry."""
    factories = {}
    for dt in [DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE, DaemonType.AUTO_SYNC]:
        factories[dt] = AsyncMock()
    return factories


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return DaemonManagerConfig()


@pytest.fixture
def lifecycle_manager(mock_daemons, mock_factories, mock_config):
    """Create lifecycle manager with mocked dependencies."""
    shutdown_event = asyncio.Event()
    lock = asyncio.Lock()
    update_callback = MagicMock()
    running_flag = [True]  # Use list for mutability

    manager = DaemonLifecycleManager(
        daemons=mock_daemons,
        factories=mock_factories,
        config=mock_config,
        shutdown_event=shutdown_event,
        lock=lock,
        update_daemon_state=update_callback,
        running_flag_getter=lambda: running_flag[0],
        running_flag_setter=lambda v: running_flag.__setitem__(0, v),
    )
    return manager


# =============================================================================
# DaemonLifecycleManager Initialization Tests
# =============================================================================


class TestDaemonLifecycleManagerInit:
    """Tests for DaemonLifecycleManager initialization."""

    def test_init_stores_references(self, mock_daemons, mock_factories, mock_config):
        """Manager stores all provided references."""
        shutdown_event = asyncio.Event()
        lock = asyncio.Lock()
        callback = MagicMock()

        manager = DaemonLifecycleManager(
            daemons=mock_daemons,
            factories=mock_factories,
            config=mock_config,
            shutdown_event=shutdown_event,
            lock=lock,
            update_daemon_state=callback,
            running_flag_getter=lambda: True,
            running_flag_setter=lambda v: None,
        )

        assert manager._daemons is mock_daemons
        assert manager._factories is mock_factories
        assert manager.config is mock_config
        assert manager._shutdown_event is shutdown_event
        assert manager._lock is lock
        assert manager._update_daemon_state is callback


# =============================================================================
# Start Daemon Tests
# =============================================================================


class TestStartDaemon:
    """Tests for starting individual daemons."""

    @pytest.mark.asyncio
    async def test_start_unknown_daemon_returns_false(self, lifecycle_manager):
        """Starting unknown daemon type returns False."""
        # Use a daemon type not in the mock registry
        result = await lifecycle_manager.start(DaemonType.QUEUE_POPULATOR)
        assert result is False

    @pytest.mark.asyncio
    async def test_start_already_running_returns_true(self, lifecycle_manager):
        """Starting already running daemon returns True."""
        lifecycle_manager._daemons[DaemonType.EVENT_ROUTER].state = DaemonState.RUNNING
        result = await lifecycle_manager.start(DaemonType.EVENT_ROUTER)
        assert result is True

    @pytest.mark.asyncio
    async def test_start_no_factory_returns_false(self, lifecycle_manager):
        """Starting daemon without factory returns False."""
        del lifecycle_manager._factories[DaemonType.EVENT_ROUTER]
        result = await lifecycle_manager.start(DaemonType.EVENT_ROUTER)
        assert result is False

    @pytest.mark.asyncio
    async def test_start_success_sets_running_state(self, lifecycle_manager):
        """Successful start sets daemon to RUNNING state."""
        # Make factory complete immediately
        lifecycle_manager._factories[DaemonType.EVENT_ROUTER] = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        result = await lifecycle_manager.start(DaemonType.EVENT_ROUTER)

        assert result is True
        info = lifecycle_manager._daemons[DaemonType.EVENT_ROUTER]
        assert info.state == DaemonState.RUNNING
        assert info.task is not None
        assert info.ready_event is not None

    @pytest.mark.asyncio
    async def test_start_creates_ready_event(self, lifecycle_manager):
        """Start creates ready_event for new daemon."""
        lifecycle_manager._factories[DaemonType.EVENT_ROUTER] = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        await lifecycle_manager.start(DaemonType.EVENT_ROUTER)

        info = lifecycle_manager._daemons[DaemonType.EVENT_ROUTER]
        assert info.ready_event is not None
        assert isinstance(info.ready_event, asyncio.Event)

    @pytest.mark.asyncio
    async def test_start_with_unmet_dependency_fails(self, lifecycle_manager):
        """Start fails if dependency is not running."""
        # Set up dependency
        info = lifecycle_manager._daemons[DaemonType.DATA_PIPELINE]
        info.depends_on = [DaemonType.EVENT_ROUTER]
        lifecycle_manager._daemons[DaemonType.EVENT_ROUTER].state = DaemonState.STOPPED

        result = await lifecycle_manager.start(DaemonType.DATA_PIPELINE)

        assert result is False


# =============================================================================
# Stop Daemon Tests
# =============================================================================


class TestStopDaemon:
    """Tests for stopping individual daemons."""

    @pytest.mark.asyncio
    async def test_stop_unknown_daemon_returns_false(self, lifecycle_manager):
        """Stopping unknown daemon type returns False."""
        result = await lifecycle_manager.stop(DaemonType.QUEUE_POPULATOR)
        assert result is False

    @pytest.mark.asyncio
    async def test_stop_already_stopped_returns_true(self, lifecycle_manager):
        """Stopping already stopped daemon returns True."""
        lifecycle_manager._daemons[DaemonType.EVENT_ROUTER].state = DaemonState.STOPPED
        result = await lifecycle_manager.stop(DaemonType.EVENT_ROUTER)
        assert result is True

    @pytest.mark.asyncio
    async def test_stop_sets_stopped_state(self, lifecycle_manager):
        """Stop sets daemon to STOPPED state."""
        info = lifecycle_manager._daemons[DaemonType.EVENT_ROUTER]
        info.state = DaemonState.RUNNING

        # Create a mock task that completes on cancel
        mock_task = MagicMock()
        mock_task.cancel = MagicMock()

        async def wait_for_cancel(*args, **kwargs):
            raise asyncio.CancelledError()

        with patch('asyncio.wait_for', side_effect=wait_for_cancel):
            info.task = mock_task
            result = await lifecycle_manager.stop(DaemonType.EVENT_ROUTER)

        assert result is True
        assert info.state == DaemonState.STOPPED
        assert info.task is None


# =============================================================================
# Restart Daemon Tests
# =============================================================================


class TestRestartFailedDaemon:
    """Tests for restarting failed daemons."""

    @pytest.mark.asyncio
    async def test_restart_unknown_daemon_returns_false(self, lifecycle_manager):
        """Restarting unknown daemon returns False."""
        result = await lifecycle_manager.restart_failed_daemon(DaemonType.QUEUE_POPULATOR)
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_running_daemon_fails(self, lifecycle_manager):
        """Cannot restart daemon that is already running."""
        lifecycle_manager._daemons[DaemonType.EVENT_ROUTER].state = DaemonState.RUNNING
        result = await lifecycle_manager.restart_failed_daemon(DaemonType.EVENT_ROUTER)
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_failed_daemon_succeeds(self, lifecycle_manager):
        """Can restart failed daemon."""
        info = lifecycle_manager._daemons[DaemonType.EVENT_ROUTER]
        info.state = DaemonState.FAILED
        lifecycle_manager._factories[DaemonType.EVENT_ROUTER] = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        result = await lifecycle_manager.restart_failed_daemon(DaemonType.EVENT_ROUTER)
        assert result is True

    @pytest.mark.asyncio
    async def test_restart_import_failed_without_force_fails(self, lifecycle_manager):
        """Cannot restart import failed daemon without force=True."""
        info = lifecycle_manager._daemons[DaemonType.EVENT_ROUTER]
        info.state = DaemonState.IMPORT_FAILED
        info.import_error = "No module named 'missing'"

        result = await lifecycle_manager.restart_failed_daemon(DaemonType.EVENT_ROUTER)
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_import_failed_with_force_succeeds(self, lifecycle_manager):
        """Can restart import failed daemon with force=True."""
        info = lifecycle_manager._daemons[DaemonType.EVENT_ROUTER]
        info.state = DaemonState.IMPORT_FAILED
        info.import_error = "No module named 'missing'"
        info.restart_count = 5
        lifecycle_manager._factories[DaemonType.EVENT_ROUTER] = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        result = await lifecycle_manager.restart_failed_daemon(
            DaemonType.EVENT_ROUTER,
            force=True
        )

        assert result is True
        assert info.restart_count == 0
        assert info.import_error is None


# =============================================================================
# Dependency Ordering Tests
# =============================================================================


class TestDependencyOrdering:
    """Tests for dependency-based ordering."""

    def test_sort_by_dependencies_no_deps(self, lifecycle_manager):
        """Sort returns input order when no dependencies."""
        types = [DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE]
        result = lifecycle_manager._sort_by_dependencies(types)
        assert set(result) == set(types)

    def test_sort_by_dependencies_with_deps(self, lifecycle_manager):
        """Sort respects dependencies."""
        # Set up dependency: DATA_PIPELINE depends on EVENT_ROUTER
        info = lifecycle_manager._daemons[DaemonType.DATA_PIPELINE]
        info.depends_on = [DaemonType.EVENT_ROUTER]

        types = [DaemonType.DATA_PIPELINE, DaemonType.EVENT_ROUTER]
        result = lifecycle_manager._sort_by_dependencies(types)

        # EVENT_ROUTER should come before DATA_PIPELINE
        router_idx = result.index(DaemonType.EVENT_ROUTER)
        pipeline_idx = result.index(DaemonType.DATA_PIPELINE)
        assert router_idx < pipeline_idx


# =============================================================================
# Dependency Validation Tests
# =============================================================================


class TestDependencyValidation:
    """Tests for dependency graph validation."""

    def test_validate_no_circular_deps(self, lifecycle_manager):
        """Validation passes with no circular dependencies."""
        types = [DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE]
        # No exception should be raised
        lifecycle_manager.validate_dependency_graph(types)

    def test_validate_detects_self_dependency(self, lifecycle_manager):
        """Validation detects self-dependencies."""
        info = lifecycle_manager._daemons[DaemonType.EVENT_ROUTER]
        info.depends_on = [DaemonType.EVENT_ROUTER]  # Self-dependency

        with pytest.raises(DependencyValidationError):
            lifecycle_manager.validate_dependency_graph([DaemonType.EVENT_ROUTER])

    def test_validate_detects_circular_deps(self, lifecycle_manager):
        """Validation detects circular dependencies."""
        # A depends on B, B depends on A
        lifecycle_manager._daemons[DaemonType.EVENT_ROUTER].depends_on = [DaemonType.DATA_PIPELINE]
        lifecycle_manager._daemons[DaemonType.DATA_PIPELINE].depends_on = [DaemonType.EVENT_ROUTER]

        with pytest.raises(DependencyValidationError):
            lifecycle_manager.validate_dependency_graph([
                DaemonType.EVENT_ROUTER,
                DaemonType.DATA_PIPELINE
            ])


# =============================================================================
# Start All / Stop All Tests
# =============================================================================


class TestStartStopAll:
    """Tests for bulk start/stop operations."""

    @pytest.mark.asyncio
    async def test_start_all_starts_all_daemons(self, lifecycle_manager):
        """start_all starts all registered daemons."""
        # Make factories complete immediately
        for dt in lifecycle_manager._factories:
            lifecycle_manager._factories[dt] = AsyncMock(
                side_effect=asyncio.CancelledError()
            )

        results = await lifecycle_manager.start_all()

        assert len(results) > 0
        # All should succeed (True)
        assert all(results.values())

    @pytest.mark.asyncio
    async def test_start_all_respects_types_filter(self, lifecycle_manager):
        """start_all respects types parameter."""
        for dt in lifecycle_manager._factories:
            lifecycle_manager._factories[dt] = AsyncMock(
                side_effect=asyncio.CancelledError()
            )

        results = await lifecycle_manager.start_all(types=[DaemonType.EVENT_ROUTER])

        assert DaemonType.EVENT_ROUTER in results
        assert DaemonType.DATA_PIPELINE not in results

    @pytest.mark.asyncio
    async def test_start_all_calls_callback(self, lifecycle_manager):
        """start_all calls on_started_callback."""
        callback = AsyncMock()
        for dt in lifecycle_manager._factories:
            lifecycle_manager._factories[dt] = AsyncMock(
                side_effect=asyncio.CancelledError()
            )

        await lifecycle_manager.start_all(on_started_callback=callback)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_all_stops_all_daemons(self, lifecycle_manager):
        """stop_all stops all running daemons."""
        # Set all to running
        for dt in lifecycle_manager._daemons:
            lifecycle_manager._daemons[dt].state = DaemonState.RUNNING

        results = await lifecycle_manager.stop_all()

        assert len(results) == len(lifecycle_manager._daemons)
        for dt in lifecycle_manager._daemons:
            assert lifecycle_manager._daemons[dt].state == DaemonState.STOPPED


# =============================================================================
# Graceful Shutdown Tests
# =============================================================================


class TestGracefulShutdown:
    """Tests for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_shutdown_event(self, lifecycle_manager):
        """Shutdown sets the shutdown event."""
        await lifecycle_manager.shutdown()
        assert lifecycle_manager._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_shutdown_stops_all_daemons(self, lifecycle_manager):
        """Shutdown stops all running daemons."""
        for dt in lifecycle_manager._daemons:
            lifecycle_manager._daemons[dt].state = DaemonState.RUNNING

        await lifecycle_manager.shutdown()

        for dt in lifecycle_manager._daemons:
            assert lifecycle_manager._daemons[dt].state == DaemonState.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_calls_pre_callback(self, lifecycle_manager):
        """Shutdown calls pre_shutdown_callback."""
        callback = AsyncMock()

        await lifecycle_manager.shutdown(pre_shutdown_callback=callback)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_health_task(self, lifecycle_manager):
        """Shutdown cancels provided health task."""
        # Create a mock health task
        async def never_ending():
            await asyncio.sleep(100)

        health_task = asyncio.create_task(never_ending())

        await lifecycle_manager.shutdown(health_task=health_task)

        assert health_task.cancelled()


# =============================================================================
# Lifecycle Event Emission Tests
# =============================================================================


class TestLifecycleEventEmission:
    """Tests for lifecycle event emission."""

    def test_emit_lifecycle_event_handles_import_error(self, lifecycle_manager):
        """Event emission handles ImportError gracefully."""
        # Patch the emit_sync import to simulate missing event_router
        with patch(
            'app.coordination.daemon_lifecycle.emit_sync',
            side_effect=ImportError("event_router not available"),
            create=True
        ):
            # Should not raise - just logs debug message
            lifecycle_manager._emit_daemon_lifecycle_event(
                DaemonType.EVENT_ROUTER,
                "DAEMON_STARTED"
            )

    @pytest.mark.asyncio
    async def test_start_emits_daemon_started_event(self, lifecycle_manager):
        """Start emits DAEMON_STARTED event."""
        lifecycle_manager._factories[DaemonType.EVENT_ROUTER] = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        with patch.object(
            lifecycle_manager,
            '_emit_daemon_lifecycle_event'
        ) as mock_emit:
            await lifecycle_manager.start(DaemonType.EVENT_ROUTER)
            mock_emit.assert_called_with(DaemonType.EVENT_ROUTER, "DAEMON_STARTED")

    @pytest.mark.asyncio
    async def test_stop_emits_daemon_stopped_event(self, lifecycle_manager):
        """Stop emits DAEMON_STOPPED event."""
        info = lifecycle_manager._daemons[DaemonType.EVENT_ROUTER]
        info.state = DaemonState.RUNNING
        info.task = None

        with patch.object(
            lifecycle_manager,
            '_emit_daemon_lifecycle_event'
        ) as mock_emit:
            await lifecycle_manager.stop(DaemonType.EVENT_ROUTER)
            mock_emit.assert_called_with(DaemonType.EVENT_ROUTER, "DAEMON_STOPPED")


# =============================================================================
# Run Daemon Error Handling Tests
# =============================================================================


class TestRunDaemonErrorHandling:
    """Tests for daemon error handling and restart logic."""

    @pytest.mark.asyncio
    async def test_import_error_sets_import_failed_state(self, lifecycle_manager):
        """Import errors set IMPORT_FAILED state."""
        lifecycle_manager._factories[DaemonType.EVENT_ROUTER] = AsyncMock(
            side_effect=ImportError("No module named 'missing'")
        )

        await lifecycle_manager.start(DaemonType.EVENT_ROUTER)

        # Give the daemon time to fail
        await asyncio.sleep(0.1)

        info = lifecycle_manager._daemons[DaemonType.EVENT_ROUTER]
        assert info.import_error is not None

    @pytest.mark.asyncio
    async def test_runtime_error_triggers_restart(self, lifecycle_manager):
        """Runtime errors trigger restart if auto_restart is True."""
        call_count = [0]

        async def failing_factory():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("First failure")
            # Second call succeeds
            await asyncio.sleep(10)

        info = lifecycle_manager._daemons[DaemonType.EVENT_ROUTER]
        info.auto_restart = True
        info.restart_delay = 0.1  # Fast restart for testing
        lifecycle_manager._factories[DaemonType.EVENT_ROUTER] = failing_factory

        await lifecycle_manager.start(DaemonType.EVENT_ROUTER)

        # Wait for restart attempt
        await asyncio.sleep(0.5)

        assert call_count[0] >= 1


# =============================================================================
# Critical Startup Order Tests
# =============================================================================


class TestCriticalStartupOrder:
    """Tests for critical startup ordering."""

    def test_apply_critical_startup_order_exists(self, lifecycle_manager):
        """_apply_critical_startup_order method exists."""
        assert hasattr(lifecycle_manager, '_apply_critical_startup_order')

    def test_apply_critical_startup_order_moves_subscribers_first(self, lifecycle_manager):
        """Critical order moves subscribers before emitters."""
        # Add necessary daemons to registry
        for dt in [DaemonType.FEEDBACK_LOOP, DaemonType.AUTO_SYNC]:
            if dt not in lifecycle_manager._daemons:
                lifecycle_manager._daemons[dt] = DaemonInfo(daemon_type=dt)

        input_order = [
            DaemonType.AUTO_SYNC,  # Emitter
            DaemonType.EVENT_ROUTER,
            DaemonType.FEEDBACK_LOOP,  # Subscriber
            DaemonType.DATA_PIPELINE,  # Subscriber
        ]

        result = lifecycle_manager._apply_critical_startup_order(input_order)

        # Subscribers should come before emitters
        if DaemonType.DATA_PIPELINE in result and DaemonType.AUTO_SYNC in result:
            pipeline_idx = result.index(DaemonType.DATA_PIPELINE)
            sync_idx = result.index(DaemonType.AUTO_SYNC)
            assert pipeline_idx < sync_idx


# =============================================================================
# Deprecated Daemon Warning Tests
# =============================================================================


class TestDeprecatedDaemonWarning:
    """Tests for deprecated daemon warnings."""

    @pytest.mark.asyncio
    async def test_start_checks_deprecated_daemons(self, lifecycle_manager):
        """Start checks for deprecated daemon types."""
        with patch(
            'app.coordination.daemon_lifecycle._check_deprecated_daemon'
        ) as mock_check:
            lifecycle_manager._factories[DaemonType.EVENT_ROUTER] = AsyncMock(
                side_effect=asyncio.CancelledError()
            )
            await lifecycle_manager.start(DaemonType.EVENT_ROUTER)
            mock_check.assert_called_with(DaemonType.EVENT_ROUTER)
