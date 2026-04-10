"""Tests for SocketLeakRecoveryDaemon.

Tests cover:
- Configuration loading from environment
- Leak detection with threshold logic
- Cleanup action triggering
- Event emission (SOCKET_LEAK_DETECTED, SOCKET_LEAK_RECOVERED)
- Health check reporting
- Grace period handling
- Consecutive critical counting
- Cooldown between events
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.socket_leak_recovery_daemon import (
    SocketLeakConfig,
    SocketLeakRecoveryDaemon,
    SocketStatus,
    get_socket_leak_recovery_daemon,
    reset_socket_leak_recovery_daemon,
)

pytestmark = pytest.mark.timeout(30)


class TestSocketLeakConfig:
    """Tests for SocketLeakConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SocketLeakConfig()

        assert config.enabled is True
        assert config.check_interval_seconds == 30.0
        assert config.cleanup_enabled is True
        assert config.event_cooldown_seconds == 120.0
        assert config.cleanup_threshold_count == 3
        assert config.startup_grace_period == 60.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SocketLeakConfig(
            enabled=False,
            check_interval_seconds=60.0,
            cleanup_enabled=False,
            event_cooldown_seconds=300.0,
            cleanup_threshold_count=5,
            startup_grace_period=120.0,
        )

        assert config.enabled is False
        assert config.check_interval_seconds == 60.0
        assert config.cleanup_enabled is False
        assert config.event_cooldown_seconds == 300.0
        assert config.cleanup_threshold_count == 5
        assert config.startup_grace_period == 120.0

    def test_from_env_defaults(self) -> None:
        """Test config creation from environment with defaults."""
        with patch.dict("os.environ", {}, clear=True):
            config = SocketLeakConfig.from_env()

        assert config.enabled is True
        assert config.check_interval_seconds == 30.0

    def test_from_env_custom(self) -> None:
        """Test config creation from environment variables."""
        env_vars = {
            "RINGRIFT_SOCKET_ENABLED": "false",
            "RINGRIFT_SOCKET_CHECK_INTERVAL": "45",
            "RINGRIFT_SOCKET_CLEANUP_ENABLED": "false",
            "RINGRIFT_SOCKET_EVENT_COOLDOWN": "180",
            "RINGRIFT_SOCKET_CLEANUP_THRESHOLD": "4",
            "RINGRIFT_SOCKET_GRACE_PERIOD": "90",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = SocketLeakConfig.from_env()

        assert config.enabled is False
        assert config.check_interval_seconds == 45.0
        assert config.cleanup_enabled is False
        assert config.event_cooldown_seconds == 180.0
        assert config.cleanup_threshold_count == 4
        assert config.startup_grace_period == 90.0


class TestSocketStatus:
    """Tests for SocketStatus dataclass."""

    def test_default_values(self) -> None:
        """Test default status values."""
        status = SocketStatus()

        assert status.total_sockets == 0
        assert status.time_wait_count == 0
        assert status.close_wait_count == 0
        assert status.established_count == 0
        assert status.fd_count == 0
        assert status.fd_limit == 1024
        assert status.fd_percent == 0.0
        assert status.socket_warning is False
        assert status.socket_critical is False
        assert status.fd_warning is False
        assert status.fd_critical is False
        assert status.issues == []

    def test_any_critical_false(self) -> None:
        """Test any_critical returns False when healthy."""
        status = SocketStatus()
        assert status.any_critical is False

    def test_any_critical_socket(self) -> None:
        """Test any_critical returns True for socket critical."""
        status = SocketStatus(socket_critical=True)
        assert status.any_critical is True

    def test_any_critical_fd(self) -> None:
        """Test any_critical returns True for FD critical."""
        status = SocketStatus(fd_critical=True)
        assert status.any_critical is True

    def test_any_warning_false(self) -> None:
        """Test any_warning returns False when healthy."""
        status = SocketStatus()
        assert status.any_warning is False

    def test_any_warning_socket(self) -> None:
        """Test any_warning returns True for socket warning."""
        status = SocketStatus(socket_warning=True)
        assert status.any_warning is True

    def test_any_warning_fd(self) -> None:
        """Test any_warning returns True for FD warning."""
        status = SocketStatus(fd_warning=True)
        assert status.any_warning is True

    def test_any_warning_includes_critical(self) -> None:
        """Test any_warning includes critical state."""
        status = SocketStatus(socket_critical=True)
        assert status.any_warning is True


class TestSocketLeakRecoveryDaemonInit:
    """Tests for daemon initialization."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_socket_leak_recovery_daemon()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_socket_leak_recovery_daemon()

    def test_default_initialization(self) -> None:
        """Test default daemon initialization."""
        daemon = SocketLeakRecoveryDaemon()

        assert daemon.config.enabled is True
        assert daemon._consecutive_criticals == 0
        assert daemon._last_event_time == 0.0
        assert daemon._leaks_detected == 0
        assert daemon._cleanups_performed == 0

    def test_custom_config_initialization(self) -> None:
        """Test daemon with custom config."""
        config = SocketLeakConfig(
            enabled=False,
            check_interval_seconds=120.0,
        )
        daemon = SocketLeakRecoveryDaemon(config=config)

        assert daemon.config.enabled is False
        assert daemon.config.check_interval_seconds == 120.0

    def test_event_subscriptions(self) -> None:
        """Test event subscriptions."""
        daemon = SocketLeakRecoveryDaemon()
        subscriptions = daemon._get_event_subscriptions()

        assert "RESOURCE_CONSTRAINT" in subscriptions
        assert callable(subscriptions["RESOURCE_CONSTRAINT"])


class TestSocketLeakRecoverySingleton:
    """Tests for singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_socket_leak_recovery_daemon()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_socket_leak_recovery_daemon()

    def test_get_returns_same_instance(self) -> None:
        """Test get_socket_leak_recovery_daemon returns same instance."""
        daemon1 = get_socket_leak_recovery_daemon()
        daemon2 = get_socket_leak_recovery_daemon()

        assert daemon1 is daemon2

    def test_reset_clears_instance(self) -> None:
        """Test reset_socket_leak_recovery_daemon clears singleton."""
        daemon1 = get_socket_leak_recovery_daemon()
        reset_socket_leak_recovery_daemon()
        daemon2 = get_socket_leak_recovery_daemon()

        assert daemon1 is not daemon2


class TestSocketLeakHealthCheck:
    """Tests for health check functionality."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_socket_leak_recovery_daemon()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_socket_leak_recovery_daemon()

    def test_health_check_disabled(self) -> None:
        """Test health check when daemon is disabled."""
        config = SocketLeakConfig(enabled=False)
        daemon = SocketLeakRecoveryDaemon(config=config)

        result = daemon.health_check()

        assert result.healthy is True
        assert "disabled" in result.message.lower()
        assert result.details["enabled"] is False

    def test_health_check_healthy(self) -> None:
        """Test health check when system is healthy."""
        daemon = SocketLeakRecoveryDaemon()
        # Ensure healthy status
        daemon._current_status = SocketStatus()

        result = daemon.health_check()

        assert result.healthy is True
        assert "healthy" in result.message.lower()

    def test_health_check_critical(self) -> None:
        """Test health check when system is critical."""
        daemon = SocketLeakRecoveryDaemon()
        daemon._current_status = SocketStatus(
            socket_critical=True,
            issues=["TIME_WAIT exceeded"],
        )

        result = daemon.health_check()

        assert result.healthy is False
        assert "critical" in result.message.lower()

    def test_health_check_warning(self) -> None:
        """Test health check when system has warnings."""
        daemon = SocketLeakRecoveryDaemon()
        daemon._current_status = SocketStatus(socket_warning=True)

        result = daemon.health_check()

        assert result.healthy is True
        assert "recovering" in result.message.lower()

    def test_health_check_details(self) -> None:
        """Test health check includes detailed stats."""
        daemon = SocketLeakRecoveryDaemon()
        daemon._current_status = SocketStatus(
            total_sockets=100,
            time_wait_count=50,
            fd_count=500,
            fd_limit=1024,
        )
        daemon._leaks_detected = 3
        daemon._cleanups_performed = 1

        result = daemon.health_check()

        assert result.details["total_sockets"] == 100
        assert result.details["time_wait"] == 50
        assert result.details["fd_count"] == 500
        assert result.details["leaks_detected"] == 3
        assert result.details["cleanups_performed"] == 1


class TestSocketLeakDetection:
    """Tests for leak detection logic."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_socket_leak_recovery_daemon()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_socket_leak_recovery_daemon()

    @pytest.mark.asyncio
    async def test_run_cycle_disabled(self) -> None:
        """Test run cycle does nothing when disabled."""
        config = SocketLeakConfig(enabled=False)
        daemon = SocketLeakRecoveryDaemon(config=config)

        with patch.object(daemon, "_check_and_recover", new_callable=AsyncMock) as mock:
            await daemon._run_cycle()
            mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_enabled(self) -> None:
        """Test run cycle calls check_and_recover when enabled."""
        daemon = SocketLeakRecoveryDaemon()

        with patch.object(daemon, "_check_and_recover", new_callable=AsyncMock) as mock:
            await daemon._run_cycle()
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_status_import_error(self) -> None:
        """Test graceful handling of import error."""
        daemon = SocketLeakRecoveryDaemon()

        with patch(
            "app.coordination.socket_leak_recovery_daemon.asyncio.to_thread",
            side_effect=ImportError("health_checks not available"),
        ):
            status = await daemon._get_current_status()

        assert "Import error" in str(status.issues)

    @pytest.mark.asyncio
    async def test_get_current_status_runtime_error(self) -> None:
        """Test graceful handling of runtime error."""
        daemon = SocketLeakRecoveryDaemon()

        with patch(
            "app.coordination.socket_leak_recovery_daemon.asyncio.to_thread",
            side_effect=RuntimeError("psutil failed"),
        ):
            status = await daemon._get_current_status()

        assert "Check error" in str(status.issues)


class TestSocketLeakEventEmission:
    """Tests for event emission."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_socket_leak_recovery_daemon()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_socket_leak_recovery_daemon()

    @pytest.mark.asyncio
    async def test_emit_leak_event_cooldown_respected(self) -> None:
        """Test cooldown prevents rapid event emission."""
        config = SocketLeakConfig(event_cooldown_seconds=60.0)
        daemon = SocketLeakRecoveryDaemon(config=config)
        daemon._last_event_time = time.time()  # Just emitted

        status = SocketStatus(socket_critical=True)

        with patch.object(daemon, "_safe_emit_event_async", new_callable=AsyncMock) as mock:
            await daemon._maybe_emit_leak_event(status)
            mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_emit_leak_event_after_cooldown(self) -> None:
        """Test event emission after cooldown expires."""
        config = SocketLeakConfig(event_cooldown_seconds=1.0)
        daemon = SocketLeakRecoveryDaemon(config=config)
        daemon._last_event_time = time.time() - 2.0  # Cooldown expired

        status = SocketStatus(
            socket_critical=True,
            time_wait_count=100,
            close_wait_count=50,
        )

        with patch.object(daemon, "_safe_emit_event_async", new_callable=AsyncMock) as mock:
            await daemon._maybe_emit_leak_event(status)
            mock.assert_called_once()
            args = mock.call_args[0]
            assert args[0] == "SOCKET_LEAK_DETECTED"

    @pytest.mark.asyncio
    async def test_emit_recovery_event(self) -> None:
        """Test recovery event emission."""
        daemon = SocketLeakRecoveryDaemon()
        daemon._leaks_detected = 1
        daemon._cleanups_performed = 0

        status = SocketStatus()  # Healthy

        with patch.object(daemon, "_safe_emit_event_async", new_callable=AsyncMock) as mock:
            await daemon._emit_recovery_event(status)
            mock.assert_called_once()
            args = mock.call_args[0]
            assert args[0] == "SOCKET_LEAK_RECOVERED"


class TestSocketLeakCleanup:
    """Tests for cleanup action triggering."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_socket_leak_recovery_daemon()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_socket_leak_recovery_daemon()

    @pytest.mark.asyncio
    async def test_trigger_cleanup_increments_count(self) -> None:
        """Test cleanup increments counter and resets criticals."""
        daemon = SocketLeakRecoveryDaemon()
        daemon._consecutive_criticals = 5

        status = SocketStatus(socket_critical=True, close_wait_count=10)

        with patch.object(daemon, "_cleanup_connection_pools", new_callable=AsyncMock, return_value=""):
            with patch.object(daemon, "_cleanup_http_sessions", new_callable=AsyncMock, return_value=""):
                await daemon._trigger_cleanup(status)

        assert daemon._cleanups_performed == 1
        assert daemon._consecutive_criticals == 0

    @pytest.mark.asyncio
    async def test_cleanup_connection_pools_success(self) -> None:
        """Test connection pool cleanup success."""
        daemon = SocketLeakRecoveryDaemon()

        mock_pool = MagicMock()
        mock_pool.cleanup_idle_connections = AsyncMock()

        # Patch the import inside the method
        with patch.dict(
            "sys.modules",
            {"scripts.p2p.connection_pool": MagicMock(get_connection_pool=MagicMock(return_value=mock_pool))},
        ):
            result = await daemon._cleanup_connection_pools()

        assert result == "connection_pool_cleanup"
        mock_pool.cleanup_idle_connections.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_cleanup_connection_pools_import_error(self) -> None:
        """Test connection pool cleanup handles import error."""
        daemon = SocketLeakRecoveryDaemon()

        # Remove module from sys.modules to trigger ImportError
        import sys
        original = sys.modules.get("scripts.p2p.connection_pool")
        sys.modules["scripts.p2p.connection_pool"] = None  # type: ignore
        try:
            result = await daemon._cleanup_connection_pools()
        finally:
            if original is not None:
                sys.modules["scripts.p2p.connection_pool"] = original
            elif "scripts.p2p.connection_pool" in sys.modules:
                del sys.modules["scripts.p2p.connection_pool"]

        assert result == ""

    @pytest.mark.asyncio
    async def test_cleanup_http_sessions_success(self) -> None:
        """Test HTTP session cleanup success."""
        daemon = SocketLeakRecoveryDaemon()

        mock_client = MagicMock()
        mock_client.cleanup_idle = AsyncMock()

        # Patch the import inside the method
        with patch.dict(
            "sys.modules",
            {"app.distributed.http_client": MagicMock(get_http_client=MagicMock(return_value=mock_client))},
        ):
            result = await daemon._cleanup_http_sessions()

        assert result == "http_session_cleanup"
        mock_client.cleanup_idle.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_p2p_reset(self) -> None:
        """Test P2P connection reset request."""
        daemon = SocketLeakRecoveryDaemon()
        daemon._current_status = SocketStatus(close_wait_count=100)

        with patch.object(daemon, "_safe_emit_event_async", new_callable=AsyncMock) as mock:
            result = await daemon._request_p2p_connection_reset()

        assert result == "p2p_reset_requested"
        mock.assert_called_once()
        args = mock.call_args[0]
        assert args[0] == "P2P_CONNECTION_RESET_REQUESTED"


class TestSocketLeakGracePeriod:
    """Tests for startup grace period handling."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_socket_leak_recovery_daemon()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_socket_leak_recovery_daemon()

    @pytest.mark.asyncio
    async def test_cleanup_not_triggered_during_grace(self) -> None:
        """Test cleanup not triggered during startup grace period."""
        config = SocketLeakConfig(
            startup_grace_period=3600.0,  # 1 hour
            cleanup_threshold_count=1,
        )
        daemon = SocketLeakRecoveryDaemon(config=config)
        daemon._consecutive_criticals = 5  # Above threshold

        # Mock started_at to be very recent
        daemon.stats.started_at = time.time()

        mock_status = SocketStatus(socket_critical=True)

        with patch.object(daemon, "_get_current_status", new_callable=AsyncMock, return_value=mock_status):
            with patch.object(daemon, "_trigger_cleanup", new_callable=AsyncMock) as mock_cleanup:
                await daemon._check_and_recover()
                mock_cleanup.assert_not_called()


class TestResourceConstraintEvent:
    """Tests for RESOURCE_CONSTRAINT event handling."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_socket_leak_recovery_daemon()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_socket_leak_recovery_daemon()

    @pytest.mark.asyncio
    async def test_on_resource_constraint_socket(self) -> None:
        """Test handling socket resource constraint event."""
        daemon = SocketLeakRecoveryDaemon()

        event = {
            "payload": {"resource_type": "socket"},
        }

        with patch.object(daemon, "_check_and_recover", new_callable=AsyncMock) as mock:
            await daemon._on_resource_constraint(event)
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_resource_constraint_fd(self) -> None:
        """Test handling file descriptor resource constraint event."""
        daemon = SocketLeakRecoveryDaemon()

        event = {
            "payload": {"resource_type": "file_descriptor"},
        }

        with patch.object(daemon, "_check_and_recover", new_callable=AsyncMock) as mock:
            await daemon._on_resource_constraint(event)
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_resource_constraint_ignored(self) -> None:
        """Test ignoring unrelated resource constraint events."""
        daemon = SocketLeakRecoveryDaemon()

        event = {
            "payload": {"resource_type": "memory"},
        }

        with patch.object(daemon, "_check_and_recover", new_callable=AsyncMock) as mock:
            await daemon._on_resource_constraint(event)
            mock.assert_not_called()


class TestConsecutiveCriticals:
    """Tests for consecutive critical counting."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_socket_leak_recovery_daemon()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_socket_leak_recovery_daemon()

    @pytest.mark.asyncio
    async def test_critical_increments_count(self) -> None:
        """Test critical status increments consecutive count."""
        config = SocketLeakConfig(
            startup_grace_period=0.0,  # No grace period
            cleanup_threshold_count=10,  # High threshold
        )
        daemon = SocketLeakRecoveryDaemon(config=config)
        daemon.stats.started_at = time.time() - 100  # Past grace

        mock_status = SocketStatus(socket_critical=True)

        with patch.object(daemon, "_get_current_status", new_callable=AsyncMock, return_value=mock_status):
            with patch.object(daemon, "_maybe_emit_leak_event", new_callable=AsyncMock):
                await daemon._check_and_recover()

        assert daemon._consecutive_criticals == 1

    @pytest.mark.asyncio
    async def test_warning_decrements_count(self) -> None:
        """Test warning status decrements consecutive count."""
        daemon = SocketLeakRecoveryDaemon()
        daemon._consecutive_criticals = 5

        mock_status = SocketStatus(socket_warning=True)

        with patch.object(daemon, "_get_current_status", new_callable=AsyncMock, return_value=mock_status):
            await daemon._check_and_recover()

        assert daemon._consecutive_criticals == 4

    @pytest.mark.asyncio
    async def test_healthy_resets_count(self) -> None:
        """Test healthy status resets consecutive count."""
        daemon = SocketLeakRecoveryDaemon()
        daemon._consecutive_criticals = 5
        daemon._leaks_detected = 0  # No previous leaks, no recovery event

        mock_status = SocketStatus()  # Healthy

        with patch.object(daemon, "_get_current_status", new_callable=AsyncMock, return_value=mock_status):
            await daemon._check_and_recover()

        assert daemon._consecutive_criticals == 0
