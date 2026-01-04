"""Socket Leak Recovery Daemon (January 2026).

Monitors socket connections and file descriptor usage to detect and recover
from resource leaks before they cause system instability.

Key features:
- Monitors TIME_WAIT, CLOSE_WAIT socket buildup
- Monitors file descriptor exhaustion
- Triggers connection pool cleanup when critical
- Emits SOCKET_LEAK_DETECTED / SOCKET_LEAK_RECOVERED events
- Part of 48-hour autonomous operation infrastructure

Thresholds (from app.config.thresholds):
- TIME_WAIT: 100 warning, 500 critical
- CLOSE_WAIT: 20 warning, 50 critical
- Total sockets: 200 warning, 500 critical
- File descriptors: 80% warning, 90% critical

Usage:
    from app.coordination.socket_leak_recovery_daemon import (
        SocketLeakRecoveryDaemon,
        get_socket_leak_recovery_daemon,
    )

    daemon = get_socket_leak_recovery_daemon()
    await daemon.start()

Environment Variables:
    RINGRIFT_SOCKET_ENABLED=true          - Enable/disable daemon
    RINGRIFT_SOCKET_CHECK_INTERVAL=30     - Check interval seconds
    RINGRIFT_SOCKET_CLEANUP_ENABLED=true  - Enable cleanup actions
    RINGRIFT_SOCKET_EVENT_COOLDOWN=120    - Cooldown between events
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)


# Environment variable prefix
_ENV_PREFIX = "RINGRIFT_SOCKET_"


def _env_float(name: str, default: float) -> float:
    """Get float from environment."""
    return float(os.environ.get(f"{_ENV_PREFIX}{name}", str(default)))


def _env_int(name: str, default: int) -> int:
    """Get int from environment."""
    return int(os.environ.get(f"{_ENV_PREFIX}{name}", str(default)))


def _env_bool(name: str, default: bool) -> bool:
    """Get bool from environment."""
    val = os.environ.get(f"{_ENV_PREFIX}{name}", str(default).lower())
    return val.lower() in ("true", "1", "yes")


@dataclass
class SocketLeakConfig:
    """Configuration for SocketLeakRecoveryDaemon."""

    enabled: bool = True
    check_interval_seconds: float = 30.0

    # Whether to attempt cleanup actions
    cleanup_enabled: bool = True

    # Cooldown between SOCKET_LEAK_DETECTED events (seconds)
    event_cooldown_seconds: float = 120.0

    # Number of consecutive criticals before triggering cleanup
    cleanup_threshold_count: int = 3

    # Grace period after startup before triggering cleanup (seconds)
    startup_grace_period: float = 60.0

    @classmethod
    def from_env(cls) -> SocketLeakConfig:
        """Create config from environment variables."""
        return cls(
            enabled=_env_bool("ENABLED", True),
            check_interval_seconds=_env_float("CHECK_INTERVAL", 30.0),
            cleanup_enabled=_env_bool("CLEANUP_ENABLED", True),
            event_cooldown_seconds=_env_float("EVENT_COOLDOWN", 120.0),
            cleanup_threshold_count=_env_int("CLEANUP_THRESHOLD", 3),
            startup_grace_period=_env_float("GRACE_PERIOD", 60.0),
        )


@dataclass
class SocketStatus:
    """Current socket/fd status snapshot."""

    # Socket counts
    total_sockets: int = 0
    time_wait_count: int = 0
    close_wait_count: int = 0
    established_count: int = 0

    # File descriptors
    fd_count: int = 0
    fd_limit: int = 1024
    fd_percent: float = 0.0

    # Status flags
    socket_warning: bool = False
    socket_critical: bool = False
    fd_warning: bool = False
    fd_critical: bool = False

    # Issues detected
    issues: list[str] = field(default_factory=list)

    @property
    def any_critical(self) -> bool:
        """Check if any resource is in critical state."""
        return self.socket_critical or self.fd_critical

    @property
    def any_warning(self) -> bool:
        """Check if any resource is in warning state."""
        return self.socket_warning or self.fd_warning or self.any_critical


class SocketLeakRecoveryDaemon(HandlerBase):
    """Daemon for detecting and recovering from socket/fd leaks."""

    _event_source = "SocketLeakRecoveryDaemon"

    def __init__(self, config: SocketLeakConfig | None = None):
        """Initialize the daemon.

        Args:
            config: Configuration. Uses environment defaults if not provided.
        """
        config = config or SocketLeakConfig.from_env()
        super().__init__(
            name="socket_leak_recovery",
            cycle_interval=config.check_interval_seconds,
        )
        self.config = config

        # State tracking
        self._consecutive_criticals: int = 0
        self._last_event_time: float = 0.0
        self._last_cleanup_time: float = 0.0
        self._cleanups_performed: int = 0
        self._leaks_detected: int = 0
        self._recoveries_completed: int = 0
        self._current_status: SocketStatus = SocketStatus()

    def _get_event_subscriptions(self) -> dict:
        """Return event subscriptions.

        This daemon primarily monitors and emits, but can respond to
        recovery-related events from other daemons.
        """
        return {
            "RESOURCE_CONSTRAINT": self._on_resource_constraint,
        }

    async def _on_resource_constraint(self, event: dict) -> None:
        """Handle resource constraint events from other daemons."""
        payload = event.get("payload", {})
        resource_type = payload.get("resource_type", "")

        if resource_type in ("socket", "fd", "file_descriptor"):
            logger.info(
                f"Received RESOURCE_CONSTRAINT for {resource_type}, "
                "triggering immediate check"
            )
            # Run check immediately
            await self._check_and_recover()

    async def _run_cycle(self) -> None:
        """Main monitoring cycle."""
        if not self.config.enabled:
            return

        await self._check_and_recover()

    async def _check_and_recover(self) -> None:
        """Check socket/fd status and trigger recovery if needed."""
        status = await self._get_current_status()
        self._current_status = status

        # Check if in startup grace period
        uptime = time.time() - self.stats.started_at
        in_grace_period = uptime < self.config.startup_grace_period

        # Log status
        if status.any_critical:
            logger.warning(
                f"Socket/FD critical: {status.issues}, "
                f"sockets={status.total_sockets}, fd={status.fd_count}/{status.fd_limit}"
            )
            self._consecutive_criticals += 1

            # Emit event (with cooldown)
            await self._maybe_emit_leak_event(status)

            # Trigger cleanup if threshold reached and not in grace period
            if (
                not in_grace_period
                and self.config.cleanup_enabled
                and self._consecutive_criticals >= self.config.cleanup_threshold_count
            ):
                await self._trigger_cleanup(status)

        elif status.any_warning:
            logger.info(
                f"Socket/FD warning: {status.issues}, "
                f"sockets={status.total_sockets}, fd={status.fd_count}/{status.fd_limit}"
            )
            # Reset consecutive count on non-critical
            self._consecutive_criticals = max(0, self._consecutive_criticals - 1)

        else:
            # Healthy - reset and potentially emit recovery
            if self._leaks_detected > 0 and self._consecutive_criticals > 0:
                await self._emit_recovery_event(status)
            self._consecutive_criticals = 0

    async def _get_current_status(self) -> SocketStatus:
        """Get current socket/fd status using health_checks utilities."""
        try:
            # Import lazily to avoid circular imports
            from app.distributed.health_checks import (
                check_file_descriptors,
                check_socket_connections,
            )

            # Run checks in thread pool (they use psutil which can block)
            fd_result = await asyncio.to_thread(check_file_descriptors)
            socket_result = await asyncio.to_thread(check_socket_connections)

            status = SocketStatus(
                total_sockets=socket_result.get("total", 0),
                time_wait_count=socket_result.get("by_status", {}).get("TIME_WAIT", 0),
                close_wait_count=socket_result.get("by_status", {}).get("CLOSE_WAIT", 0),
                established_count=socket_result.get("by_status", {}).get("ESTABLISHED", 0),
                fd_count=fd_result.get("count", 0),
                fd_limit=fd_result.get("limit", 1024),
                fd_percent=fd_result.get("percent_used", 0.0),
                socket_warning=socket_result.get("status") == "warning",
                socket_critical=socket_result.get("status") == "critical",
                fd_warning=fd_result.get("status") == "warning",
                fd_critical=fd_result.get("status") == "critical",
                issues=socket_result.get("issues", [])
                + ([fd_result.get("message", "")] if fd_result.get("status") != "ok" else []),
            )
            return status

        except ImportError as e:
            logger.warning(f"health_checks module not available: {e}")
            return SocketStatus(issues=[f"Import error: {e}"])
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"Failed to get socket status: {e}")
            return SocketStatus(issues=[f"Check error: {e}"])

    async def _maybe_emit_leak_event(self, status: SocketStatus) -> None:
        """Emit SOCKET_LEAK_DETECTED event with cooldown."""
        now = time.time()
        if now - self._last_event_time < self.config.event_cooldown_seconds:
            return

        self._last_event_time = now
        self._leaks_detected += 1

        await self._safe_emit_event_async(
            "SOCKET_LEAK_DETECTED",
            {
                "total_sockets": status.total_sockets,
                "time_wait": status.time_wait_count,
                "close_wait": status.close_wait_count,
                "fd_count": status.fd_count,
                "fd_limit": status.fd_limit,
                "fd_percent": status.fd_percent,
                "issues": status.issues,
                "consecutive_criticals": self._consecutive_criticals,
                "source": self._event_source,
            },
        )
        logger.warning(
            f"Emitted SOCKET_LEAK_DETECTED: "
            f"TIME_WAIT={status.time_wait_count}, CLOSE_WAIT={status.close_wait_count}"
        )

    async def _emit_recovery_event(self, status: SocketStatus) -> None:
        """Emit SOCKET_LEAK_RECOVERED event."""
        self._recoveries_completed += 1

        await self._safe_emit_event_async(
            "SOCKET_LEAK_RECOVERED",
            {
                "total_sockets": status.total_sockets,
                "fd_count": status.fd_count,
                "fd_percent": status.fd_percent,
                "leaks_detected": self._leaks_detected,
                "cleanups_performed": self._cleanups_performed,
                "source": self._event_source,
            },
        )
        logger.info("Emitted SOCKET_LEAK_RECOVERED - system healthy")

    async def _trigger_cleanup(self, status: SocketStatus) -> None:
        """Trigger cleanup actions to recover from socket leak."""
        logger.warning(
            f"Triggering socket leak cleanup after {self._consecutive_criticals} "
            f"consecutive critical readings"
        )

        cleanup_actions: list[str] = []

        # 1. Request connection pool cleanup
        cleanup_actions.append(await self._cleanup_connection_pools())

        # 2. Request HTTP session cleanup
        cleanup_actions.append(await self._cleanup_http_sessions())

        # 3. Request P2P connection cleanup (if critical)
        if status.socket_critical and status.close_wait_count > 30:
            cleanup_actions.append(await self._request_p2p_connection_reset())

        self._cleanups_performed += 1
        self._last_cleanup_time = time.time()
        self._consecutive_criticals = 0

        # Log results
        successful = [a for a in cleanup_actions if a]
        logger.info(
            f"Socket cleanup completed: {len(successful)} actions, "
            f"total cleanups: {self._cleanups_performed}"
        )

    async def _cleanup_connection_pools(self) -> str:
        """Request connection pool cleanup from P2P infrastructure."""
        try:
            # Try to access connection pool and request cleanup
            from scripts.p2p.connection_pool import get_connection_pool

            pool = get_connection_pool()
            if hasattr(pool, "cleanup_idle_connections"):
                await pool.cleanup_idle_connections(force=True)
                return "connection_pool_cleanup"
            return ""
        except ImportError:
            logger.debug("Connection pool not available for cleanup")
            return ""
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"Connection pool cleanup skipped: {e}")
            return ""

    async def _cleanup_http_sessions(self) -> str:
        """Request cleanup of idle HTTP sessions."""
        try:
            # Try to clean up aiohttp sessions via known singletons
            from app.distributed.http_client import get_http_client

            client = get_http_client()
            if hasattr(client, "cleanup_idle"):
                await client.cleanup_idle()
                return "http_session_cleanup"
            return ""
        except ImportError:
            logger.debug("HTTP client not available for cleanup")
            return ""
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"HTTP session cleanup skipped: {e}")
            return ""

    async def _request_p2p_connection_reset(self) -> str:
        """Request P2P orchestrator to reset problematic connections."""
        try:
            # Emit event for P2P recovery daemon to handle
            await self._safe_emit_event_async(
                "P2P_CONNECTION_RESET_REQUESTED",
                {
                    "reason": "socket_leak_recovery",
                    "close_wait_count": self._current_status.close_wait_count,
                    "source": self._event_source,
                },
            )
            return "p2p_reset_requested"
        except (RuntimeError, ValueError) as e:
            logger.debug(f"P2P reset request failed: {e}")
            return ""

    def health_check(self) -> HealthCheckResult:
        """Return current health status."""
        if not self.config.enabled:
            return HealthCheckResult(
                healthy=True,
                message="Socket leak recovery disabled",
                details={"enabled": False},
            )

        # Determine health based on current status
        status = self._current_status
        is_healthy = not status.any_critical

        if status.any_critical:
            message = f"Socket leak critical: {', '.join(status.issues[:2])}"
        elif status.any_warning:
            message = "Socket/FD warning but recovering"
        else:
            message = "Socket/FD healthy"

        return HealthCheckResult(
            healthy=is_healthy,
            message=message,
            details={
                "total_sockets": status.total_sockets,
                "time_wait": status.time_wait_count,
                "close_wait": status.close_wait_count,
                "established": status.established_count,
                "fd_count": status.fd_count,
                "fd_limit": status.fd_limit,
                "fd_percent": round(status.fd_percent, 1),
                "consecutive_criticals": self._consecutive_criticals,
                "leaks_detected": self._leaks_detected,
                "cleanups_performed": self._cleanups_performed,
                "recoveries": self._recoveries_completed,
                "enabled": self.config.enabled,
                "cleanup_enabled": self.config.cleanup_enabled,
            },
        )


# =============================================================================
# Singleton Access
# =============================================================================

_instance: SocketLeakRecoveryDaemon | None = None


def get_socket_leak_recovery_daemon() -> SocketLeakRecoveryDaemon:
    """Get or create the singleton daemon instance."""
    global _instance
    if _instance is None:
        _instance = SocketLeakRecoveryDaemon()
    return _instance


def reset_socket_leak_recovery_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
