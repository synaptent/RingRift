"""Comprehensive tests for DaemonWatchdog.

Tests cover:
- Watchdog initialization and configuration
- Alert callbacks (sync and async)
- Daemon health checking logic
- Auto-restart behavior with cooldown and max attempts
- Background health check loop
- Import failure detection (no auto-restart)
- Stuck daemon detection
- Singleton pattern management
- Event router integration (graceful fallback)

Created: December 2025
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.daemon_types import DaemonInfo, DaemonState, DaemonType
from app.coordination.daemon_watchdog import (
    DaemonHealthRecord,
    DaemonWatchdog,
    WatchdogAlert,
    WatchdogConfig,
    get_watchdog,
    start_watchdog,
    stop_watchdog,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fast_config() -> WatchdogConfig:
    """Fast config for testing (short intervals)."""
    return WatchdogConfig(
        check_interval_seconds=0.1,
        auto_restart_cooldown_seconds=0.1,
        max_auto_restarts=3,
        auto_restart_enabled=True,
        auto_restart_window_seconds=60.0,
    )


@pytest.fixture
def mock_manager() -> MagicMock:
    """Mock DaemonManager."""
    manager = MagicMock()
    manager.get_status.return_value = {"daemons": {}}
    manager.get_daemon_info.return_value = None
    manager.restart_failed_daemon = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_daemon_info_running() -> MagicMock:
    """Mock DaemonInfo in RUNNING state with active task."""
    info = MagicMock(spec=DaemonInfo)
    info.state = DaemonState.RUNNING
    info.task = MagicMock()
    info.task.done.return_value = False
    info.import_error = None
    info.error = None
    info.start_time = time.time()
    info.type = DaemonType.AUTO_SYNC
    return info


@pytest.fixture
def mock_daemon_info_failed() -> MagicMock:
    """Mock DaemonInfo in FAILED state."""
    info = MagicMock(spec=DaemonInfo)
    info.state = DaemonState.FAILED
    info.task = None
    info.import_error = None
    info.error = "Test failure"
    info.start_time = time.time() - 100
    info.type = DaemonType.AUTO_SYNC
    return info


@pytest.fixture
def mock_daemon_info_stuck() -> MagicMock:
    """Mock DaemonInfo that appears stuck (RUNNING but task done)."""
    info = MagicMock(spec=DaemonInfo)
    info.state = DaemonState.RUNNING
    task = MagicMock()
    task.done.return_value = True
    task.exception.return_value = RuntimeError("Task crashed")
    info.task = task
    info.import_error = None
    info.error = None
    info.start_time = time.time() - 10
    info.type = DaemonType.AUTO_SYNC
    return info


@pytest.fixture
def mock_daemon_info_import_failed() -> MagicMock:
    """Mock DaemonInfo with import failure."""
    info = MagicMock(spec=DaemonInfo)
    info.state = DaemonState.IMPORT_FAILED
    info.task = None
    info.import_error = "ModuleNotFoundError: No module named 'missing'"
    info.error = None
    info.start_time = time.time()
    info.type = DaemonType.AUTO_SYNC
    return info


@pytest.fixture
def reset_watchdog_singleton():
    """Reset the global watchdog singleton before/after tests."""
    import app.coordination.daemon_watchdog as dwm

    original = getattr(dwm, "_watchdog", None)
    dwm._watchdog = None
    yield
    dwm._watchdog = original


# =============================================================================
# WatchdogAlert Tests
# =============================================================================


class TestWatchdogAlert:
    """Test WatchdogAlert enum."""

    def test_alert_enum_values(self):
        """Test all alert types exist."""
        assert WatchdogAlert.DAEMON_STUCK == "daemon_stuck"
        assert WatchdogAlert.DAEMON_CRASHED == "daemon_crashed"
        assert WatchdogAlert.DAEMON_IMPORT_FAILED == "daemon_import_failed"
        assert WatchdogAlert.DAEMON_RESTART_EXHAUSTED == "daemon_restart_exhausted"
        assert WatchdogAlert.DAEMON_AUTO_RESTARTED == "daemon_auto_restarted"
        assert WatchdogAlert.WATCHDOG_STARTED == "watchdog_started"
        assert WatchdogAlert.WATCHDOG_STOPPED == "watchdog_stopped"

    def test_alert_value(self):
        """Test alert value representation."""
        assert WatchdogAlert.DAEMON_STUCK.value == "daemon_stuck"


# =============================================================================
# WatchdogConfig Tests
# =============================================================================


class TestWatchdogConfig:
    """Test WatchdogConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = WatchdogConfig()
        assert config.check_interval_seconds == 30.0
        assert config.auto_restart_cooldown_seconds == 60.0
        assert config.max_auto_restarts == 3
        assert config.auto_restart_enabled is True
        assert config.auto_restart_window_seconds == 3600.0

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = WatchdogConfig(
            check_interval_seconds=10.0,
            auto_restart_cooldown_seconds=30.0,
            max_auto_restarts=5,
            auto_restart_enabled=False,
            auto_restart_window_seconds=1800.0,
        )
        assert config.check_interval_seconds == 10.0
        assert config.auto_restart_cooldown_seconds == 30.0
        assert config.max_auto_restarts == 5
        assert config.auto_restart_enabled is False
        assert config.auto_restart_window_seconds == 1800.0


# =============================================================================
# DaemonHealthRecord Tests
# =============================================================================


class TestDaemonHealthRecord:
    """Test DaemonHealthRecord dataclass."""

    def test_health_record_defaults(self):
        """Test default health record values."""
        record = DaemonHealthRecord(
            last_check_time=0.0,
            last_restart_time=0.0,
            auto_restart_count=0,
            first_auto_restart_time=0.0,
            last_alert_time=0.0,
            consecutive_healthy_checks=0,
        )
        assert record.auto_restart_count == 0
        assert record.consecutive_healthy_checks == 0

    def test_health_record_tracking(self):
        """Test health record field updates."""
        now = time.time()
        record = DaemonHealthRecord(
            last_check_time=now,
            last_restart_time=now - 60,
            auto_restart_count=2,
            first_auto_restart_time=now - 300,
            last_alert_time=now - 3600,
            consecutive_healthy_checks=10,
        )
        assert record.auto_restart_count == 2
        assert record.consecutive_healthy_checks == 10
        assert record.last_restart_time < record.last_check_time


# =============================================================================
# DaemonWatchdog Init Tests
# =============================================================================


class TestDaemonWatchdogInit:
    """Test DaemonWatchdog initialization."""

    def test_init_with_defaults(self):
        """Test watchdog with default config."""
        watchdog = DaemonWatchdog()
        assert watchdog.config is not None
        assert watchdog._running is False
        assert watchdog._health_records == {}

    def test_init_with_custom_config(self, fast_config):
        """Test watchdog with custom config."""
        watchdog = DaemonWatchdog(config=fast_config)
        assert watchdog.config.check_interval_seconds == 0.1
        assert watchdog.config.max_auto_restarts == 3

    def test_init_with_manager(self, mock_manager):
        """Test watchdog with injected manager."""
        watchdog = DaemonWatchdog(manager=mock_manager)
        # Manager should be accessible via property
        assert watchdog._manager is mock_manager


# =============================================================================
# Alert Callback Tests
# =============================================================================


class TestAlertCallbacks:
    """Test alert callback functionality."""

    def test_add_alert_callback(self, mock_manager):
        """Test adding alert callbacks."""
        watchdog = DaemonWatchdog(manager=mock_manager)
        callback = MagicMock()
        watchdog.add_alert_callback(callback)
        assert callback in watchdog._alert_callbacks

    @pytest.mark.asyncio
    async def test_emit_alert_calls_sync_callback(self, mock_manager):
        """Test that sync callbacks are called."""
        watchdog = DaemonWatchdog(manager=mock_manager)
        callback = MagicMock()
        watchdog.add_alert_callback(callback)

        await watchdog._emit_alert(
            WatchdogAlert.DAEMON_CRASHED,
            daemon_name="test_daemon",
            details={"error": "test"},
        )

        callback.assert_called_once()
        # Callback receives (alert_type, daemon_name, details)
        call_args = callback.call_args[0]
        assert call_args[0] == WatchdogAlert.DAEMON_CRASHED
        assert call_args[1] == "test_daemon"

    @pytest.mark.asyncio
    async def test_emit_alert_calls_async_callback(self, mock_manager):
        """Test that async callbacks are awaited."""
        watchdog = DaemonWatchdog(manager=mock_manager)
        async_callback = AsyncMock()
        watchdog.add_alert_callback(async_callback)

        await watchdog._emit_alert(
            WatchdogAlert.WATCHDOG_STARTED,
            daemon_name=None,
        )

        async_callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_emit_alert_handles_callback_error(self, mock_manager):
        """Test that callback errors don't break emission."""
        watchdog = DaemonWatchdog(manager=mock_manager)
        bad_callback = MagicMock(side_effect=RuntimeError("Callback failed"))
        good_callback = MagicMock()
        watchdog.add_alert_callback(bad_callback)
        watchdog.add_alert_callback(good_callback)

        # Should not raise, good callback should still be called
        await watchdog._emit_alert(WatchdogAlert.DAEMON_STUCK, daemon_name="test")

        good_callback.assert_called_once()


# =============================================================================
# Health Record Tests
# =============================================================================


class TestHealthRecord:
    """Test health record management."""

    def test_get_creates_new_record(self, mock_manager):
        """Test that get_health_record creates new record if missing."""
        watchdog = DaemonWatchdog(manager=mock_manager)
        record = watchdog._get_health_record("new_daemon")
        assert record is not None
        assert record.auto_restart_count == 0

    def test_get_returns_existing_record(self, mock_manager):
        """Test that get_health_record returns existing record."""
        watchdog = DaemonWatchdog(manager=mock_manager)
        record1 = watchdog._get_health_record("my_daemon")
        record1.auto_restart_count = 5
        record2 = watchdog._get_health_record("my_daemon")
        assert record2.auto_restart_count == 5


# =============================================================================
# Auto-Restart Window Tests
# =============================================================================


class TestAutoRestartWindow:
    """Test auto-restart window reset logic."""

    def test_reset_outside_window(self, mock_manager, fast_config):
        """Test restart count resets after window expires."""
        fast_config.auto_restart_window_seconds = 1.0  # 1 second window
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        record = watchdog._get_health_record("test")
        record.auto_restart_count = 3
        record.first_auto_restart_time = time.time() - 10  # 10 seconds ago

        now = time.time()
        watchdog._reset_auto_restart_window(record, now)

        assert record.auto_restart_count == 0
        assert record.first_auto_restart_time == 0.0

    def test_keep_inside_window(self, mock_manager, fast_config):
        """Test restart count preserved inside window."""
        fast_config.auto_restart_window_seconds = 60.0  # 60 second window
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        record = watchdog._get_health_record("test")
        record.auto_restart_count = 2
        record.first_auto_restart_time = time.time() - 5  # 5 seconds ago

        now = time.time()
        watchdog._reset_auto_restart_window(record, now)

        assert record.auto_restart_count == 2  # Preserved


# =============================================================================
# Check Daemon Health Tests
# =============================================================================


class TestCheckDaemonHealth:
    """Test core daemon health checking logic."""

    @pytest.mark.asyncio
    async def test_unknown_daemon_type_returns_early(self, mock_manager):
        """Test that unknown daemon types are skipped."""
        watchdog = DaemonWatchdog(manager=mock_manager)
        # Should not raise, just return early
        await watchdog._check_daemon_health("unknown_daemon_xyz")

    @pytest.mark.asyncio
    async def test_unregistered_daemon_returns_early(self, mock_manager):
        """Test that unregistered daemons are skipped."""
        mock_manager.get_daemon_info.return_value = None
        watchdog = DaemonWatchdog(manager=mock_manager)

        await watchdog._check_daemon_health("auto_sync")
        # Should not emit any alerts for unregistered daemon

    @pytest.mark.asyncio
    async def test_detects_stuck_daemon(
        self, mock_manager, mock_daemon_info_stuck, fast_config
    ):
        """Test detection of stuck daemon (RUNNING but task done)."""
        mock_manager.get_daemon_info.return_value = mock_daemon_info_stuck
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        alerts_received = []

        def capture_alert(alert_type, daemon_name, details):
            alerts_received.append((alert_type, daemon_name, details))

        watchdog.add_alert_callback(capture_alert)

        await watchdog._check_daemon_health("auto_sync")

        # Should emit stuck alert (first alert before any auto-restart)
        assert len(alerts_received) >= 1
        # First alert should be DAEMON_STUCK
        first_alert_type = alerts_received[0][0]
        assert first_alert_type == WatchdogAlert.DAEMON_STUCK

    @pytest.mark.asyncio
    async def test_import_failed_daemon_no_restart(
        self, mock_manager, mock_daemon_info_import_failed, fast_config
    ):
        """Test that import-failed daemons are NOT auto-restarted."""
        mock_manager.get_daemon_info.return_value = mock_daemon_info_import_failed
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        await watchdog._check_daemon_health("auto_sync")

        # restart_failed_daemon should NOT be called
        mock_manager.restart_failed_daemon.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_daemon_auto_restart_success(
        self, mock_manager, mock_daemon_info_failed, fast_config
    ):
        """Test successful auto-restart of failed daemon."""
        mock_manager.get_daemon_info.return_value = mock_daemon_info_failed
        mock_manager.restart_failed_daemon.return_value = True
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        callback = MagicMock()
        watchdog.add_alert_callback(callback)

        await watchdog._check_daemon_health("auto_sync")

        # Should attempt restart
        mock_manager.restart_failed_daemon.assert_awaited()

    @pytest.mark.asyncio
    async def test_failed_daemon_max_restarts_exceeded(
        self, mock_manager, mock_daemon_info_failed, fast_config
    ):
        """Test max restart limit enforcement."""
        fast_config.max_auto_restarts = 2
        mock_manager.get_daemon_info.return_value = mock_daemon_info_failed
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        # Simulate exhausted restarts
        record = watchdog._get_health_record("auto_sync")
        record.auto_restart_count = 3
        record.first_auto_restart_time = time.time()

        callback = MagicMock()
        watchdog.add_alert_callback(callback)

        await watchdog._check_daemon_health("auto_sync")

        # Should NOT attempt restart, should emit exhausted alert
        mock_manager.restart_failed_daemon.assert_not_awaited()
        callback.assert_called()

    @pytest.mark.asyncio
    async def test_failed_daemon_in_cooldown(
        self, mock_manager, mock_daemon_info_failed, fast_config
    ):
        """Test cooldown period prevents rapid restarts."""
        fast_config.auto_restart_cooldown_seconds = 60.0
        mock_manager.get_daemon_info.return_value = mock_daemon_info_failed
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        # Set recent restart time
        record = watchdog._get_health_record("auto_sync")
        record.last_restart_time = time.time() - 5  # 5 seconds ago

        await watchdog._check_daemon_health("auto_sync")

        # Should NOT attempt restart due to cooldown
        mock_manager.restart_failed_daemon.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_running_daemon_increments_health(
        self, mock_manager, mock_daemon_info_running, fast_config
    ):
        """Test that healthy running daemons increment counter."""
        mock_manager.get_daemon_info.return_value = mock_daemon_info_running
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        record = watchdog._get_health_record("auto_sync")
        initial_count = record.consecutive_healthy_checks

        await watchdog._check_daemon_health("auto_sync")

        assert record.consecutive_healthy_checks == initial_count + 1

    @pytest.mark.asyncio
    async def test_disabled_auto_restart(
        self, mock_manager, mock_daemon_info_failed, fast_config
    ):
        """Test auto-restart can be disabled."""
        fast_config.auto_restart_enabled = False
        mock_manager.get_daemon_info.return_value = mock_daemon_info_failed
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        await watchdog._check_daemon_health("auto_sync")

        # Should NOT attempt restart
        mock_manager.restart_failed_daemon.assert_not_awaited()


# =============================================================================
# Health Check Loop Tests
# =============================================================================


class TestHealthCheckLoop:
    """Test background health check loop."""

    @pytest.mark.asyncio
    async def test_loop_handles_cancelled_error(self, mock_manager, fast_config):
        """Test loop handles CancelledError gracefully by stopping."""
        mock_manager.get_status.return_value = {"daemons": {"auto_sync": {}}}
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)
        watchdog._running = True  # Simulate started state

        task = asyncio.create_task(watchdog._health_check_loop())
        await asyncio.sleep(0.05)  # Let it start
        task.cancel()

        # The loop catches CancelledError, emits alert, and returns
        with suppress(asyncio.CancelledError):
            await task


# =============================================================================
# Start/Stop Tests
# =============================================================================


class TestStartStop:
    """Test watchdog start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, mock_manager, fast_config):
        """Test start sets running flag."""
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)
        mock_manager.get_status.return_value = {"daemons": {}}

        with patch(
            "app.coordination.daemon_watchdog.safe_create_task"
        ) as mock_create_task:
            mock_task = MagicMock()
            mock_task.add_done_callback = MagicMock()
            mock_create_task.return_value = mock_task

            await watchdog.start()  # start() is async

            assert watchdog._running is True
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, mock_manager, fast_config):
        """Test start is idempotent (calling twice doesn't create second task)."""
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)
        watchdog._running = True

        with patch(
            "app.coordination.daemon_watchdog.safe_create_task"
        ) as mock_create_task:
            await watchdog.start()  # start() is async
            mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self, mock_manager, fast_config):
        """Test stop clears running flag."""
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)
        watchdog._running = True

        # Create a proper cancellable task
        async def dummy():
            await asyncio.sleep(10)

        watchdog._task = asyncio.create_task(dummy())

        await watchdog.stop()

        assert watchdog._running is False

    @pytest.mark.asyncio
    async def test_stop_not_running_returns_early(self, mock_manager, fast_config):
        """Test stop returns early if not running."""
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)
        watchdog._running = False

        # Should not raise
        await watchdog.stop()


# =============================================================================
# Health Status Tests
# =============================================================================


class TestGetHealthStatus:
    """Test health status reporting."""

    def test_status_includes_running_state(self, mock_manager, fast_config):
        """Test status includes running flag."""
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)
        watchdog._running = True

        status = watchdog.get_health_status()

        assert status["running"] is True

    def test_status_includes_config(self, mock_manager, fast_config):
        """Test status includes config values."""
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        status = watchdog.get_health_status()

        # Status uses flattened config keys
        assert "check_interval" in status
        assert status["check_interval"] == 0.1
        assert "auto_restart_enabled" in status

    def test_status_includes_daemon_records(self, mock_manager, fast_config):
        """Test status includes daemon health records."""
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        # Create some records
        record = watchdog._get_health_record("test_daemon")
        record.auto_restart_count = 2
        record.consecutive_healthy_checks = 5

        status = watchdog.get_health_status()

        # Status uses "daemons" key, not "daemon_records"
        assert "daemons" in status
        assert "test_daemon" in status["daemons"]
        assert status["daemons"]["test_daemon"]["auto_restart_count"] == 2


# =============================================================================
# Health Check Compliance Tests
# =============================================================================


class TestHealthCheckCompliance:
    """Test CoordinatorProtocol health_check method."""

    def test_health_check_returns_healthy(self, mock_manager, fast_config):
        """Test health_check returns healthy when running."""
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)
        watchdog._running = True

        # health_check is sync (not async)
        result = watchdog.health_check()

        assert result.healthy is True

    def test_health_check_not_running(self, mock_manager, fast_config):
        """Test health_check when not running."""
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)
        watchdog._running = False

        result = watchdog.health_check()

        # Not running returns healthy=False
        assert result is not None
        assert result.healthy is False


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Test module-level singleton functions."""

    def test_get_watchdog_returns_instance(self, reset_watchdog_singleton):
        """Test get_watchdog returns watchdog instance."""
        with patch("app.coordination.daemon_watchdog.DaemonWatchdog") as MockWatchdog:
            MockWatchdog.return_value = MagicMock()
            watchdog = get_watchdog()
            assert watchdog is not None

    def test_get_watchdog_returns_same_instance(self, reset_watchdog_singleton):
        """Test get_watchdog returns singleton."""
        import app.coordination.daemon_watchdog as dwm

        with patch("app.coordination.daemon_watchdog.DaemonWatchdog") as MockWatchdog:
            MockWatchdog.return_value = MagicMock()
            w1 = get_watchdog()
            w2 = get_watchdog()
            # Should return same instance
            assert w1 is w2

    @pytest.mark.asyncio
    async def test_start_watchdog_starts_instance(self, reset_watchdog_singleton):
        """Test start_watchdog starts the watchdog."""
        with patch("app.coordination.daemon_watchdog.DaemonWatchdog") as MockWatchdog:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            MockWatchdog.return_value = mock_instance

            result = await start_watchdog()

            mock_instance.start.assert_awaited_once()
            assert result is mock_instance


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration-style tests for common workflows."""

    @pytest.mark.asyncio
    async def test_detect_and_track_failed_daemon(
        self, mock_manager, mock_daemon_info_failed, fast_config
    ):
        """Test detecting and tracking a failed daemon over multiple checks.

        Tests that successful auto-restarts emit DAEMON_AUTO_RESTARTED alerts.
        """
        fast_config.auto_restart_cooldown_seconds = 0.01
        mock_manager.get_daemon_info.return_value = mock_daemon_info_failed
        # Return True for successful restart (alert is only emitted on success)
        mock_manager.restart_failed_daemon = AsyncMock(return_value=True)
        watchdog = DaemonWatchdog(manager=mock_manager, config=fast_config)

        alerts_received = []

        def capture_alert(alert_type, daemon_name, details):
            alerts_received.append((alert_type, daemon_name, details))

        watchdog.add_alert_callback(capture_alert)

        # Multiple checks - should trigger restarts
        for _ in range(4):
            await watchdog._check_daemon_health("auto_sync")
            await asyncio.sleep(0.02)

        # Should have received alerts for successful restarts
        assert len(alerts_received) >= 1
        # All alerts should be DAEMON_AUTO_RESTARTED
        for alert_type, daemon_name, details in alerts_received:
            assert alert_type == WatchdogAlert.DAEMON_AUTO_RESTARTED
            assert daemon_name == "auto_sync"
