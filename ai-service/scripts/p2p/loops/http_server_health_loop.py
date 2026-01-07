"""Internal HTTP server health monitoring loop.

January 2026: Created to detect zombie state where HTTP server crashes but process continues.

Problem:
The P2P orchestrator can enter a zombie state where:
1. The HTTP server (aiohttp) crashes due to request handling errors
2. The main Python process continues running (consuming CPU/RAM)
3. Port 8770 stops listening
4. External watchdogs (pgrep) see process as alive, but health endpoint fails

Solution:
This loop probes localhost:8770/health every 10 seconds. After 4 consecutive failures
(40 seconds), it attempts recovery. If recovery fails after 2 attempts, it terminates
the process with exit code 3, allowing systemd to restart it.

Detection Timeline (updated Jan 2026):
    T+0s:    HTTP server crashes
    T+10s:   First probe fails (counter=1)
    T+40s:   Fourth probe fails (counter=4, threshold reached)
    T+45s:   Recovery attempt 1 (wait 5s, re-probe)
    T+50s:   Recovery attempt 2 (wait 5s, re-probe)
    T+55s:   Force exit with code 3
    T+70s:   systemd RestartSec=15 restarts process
    Total MTTR: ~70 seconds (improved from ~90s)

Usage:
    from scripts.p2p.loops import HttpServerHealthLoop, HttpServerHealthConfig

    loop = HttpServerHealthLoop(port=8770)
    await loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scripts.p2p.loops.base import BaseLoop

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Exit code for HTTP server failure (distinguishes from normal exit)
EXIT_CODE_HTTP_SERVER_FAILED = 3


@dataclass
class HttpServerHealthConfig:
    """Configuration for HTTP server health monitoring.

    Attributes:
        probe_interval_seconds: How often to probe localhost health (default: 10s)
        probe_timeout_seconds: HTTP request timeout (default: 5s)
        failure_threshold: Consecutive failures before recovery attempt (default: 6 = 60s)
        recovery_attempts: Number of recovery re-probes before termination (default: 2)
        recovery_delay_seconds: Delay between recovery attempts (default: 5s)
        startup_grace_period_seconds: Skip probes during startup (default: 60s)
        exit_code_http_server_failed: Exit code when terminating (default: 3)
    """

    probe_interval_seconds: float = 10.0
    probe_timeout_seconds: float = 5.0
    failure_threshold: int = 4  # Reduced from 6: faster detection (40s vs 60s)
    recovery_attempts: int = 2
    recovery_delay_seconds: float = 5.0
    startup_grace_period_seconds: float = 30.0  # Reduced from 60s: faster detection
    exit_code_http_server_failed: int = EXIT_CODE_HTTP_SERVER_FAILED


class HttpServerHealthLoop(BaseLoop):
    """Monitors local HTTP server health and terminates on unrecoverable failure.

    This loop provides internal self-monitoring for the P2P orchestrator's HTTP server.
    Unlike external watchdogs (cron, systemd), this can detect the zombie state where
    the process is running but the HTTP server has crashed.

    The loop:
    1. Probes localhost:PORT/health every probe_interval_seconds
    2. Tracks consecutive failures
    3. After failure_threshold consecutive failures, attempts recovery
    4. If recovery fails after recovery_attempts, terminates with exit code 3
    5. systemd Restart=always will restart the process

    Attributes:
        port: The HTTP port to probe (default: 8770)
        config: Configuration for probe intervals and thresholds
    """

    def __init__(
        self,
        port: int = 8770,
        config: HttpServerHealthConfig | None = None,
    ):
        """Initialize the HTTP server health loop.

        Args:
            port: HTTP port to probe for health endpoint
            config: Configuration for monitoring behavior (uses defaults if None)
        """
        self._config = config or HttpServerHealthConfig()
        super().__init__(
            name="http_server_health",
            interval=self._config.probe_interval_seconds,
        )
        self._port = port
        self._consecutive_failures = 0
        self._start_time = time.time()
        self._last_success_time = 0.0
        self._total_probes = 0
        self._total_failures = 0
        self._recovery_triggered = False

    async def _run_once(self) -> None:
        """Execute one health probe iteration.

        Skips probing during startup grace period. On probe failure, increments
        failure counter. When threshold is reached, initiates recovery sequence.
        """
        # Skip during startup grace period
        elapsed = time.time() - self._start_time
        if elapsed < self._config.startup_grace_period_seconds:
            logger.debug(
                f"[{self.name}] Startup grace period: {elapsed:.1f}s / "
                f"{self._config.startup_grace_period_seconds}s"
            )
            return

        self._total_probes += 1

        if await self._probe_local_health():
            # Success - reset failure counter
            if self._consecutive_failures > 0:
                logger.info(
                    f"[{self.name}] HTTP server recovered after "
                    f"{self._consecutive_failures} failures"
                )
            self._consecutive_failures = 0
            self._last_success_time = time.time()
            self._recovery_triggered = False
            return

        # Failure - increment counter
        self._consecutive_failures += 1
        self._total_failures += 1
        logger.warning(
            f"[{self.name}] Health probe failed "
            f"(consecutive: {self._consecutive_failures}/{self._config.failure_threshold})"
        )

        if self._consecutive_failures >= self._config.failure_threshold:
            await self._handle_server_failure()

    async def _probe_local_health(self) -> bool:
        """Probe localhost health endpoint with independent session.

        Creates a fresh aiohttp session for each probe to avoid connection pooling
        issues that could mask real failures.

        Returns:
            True if health endpoint returns 200 OK, False otherwise
        """
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=self._config.probe_timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"http://127.0.0.1:{self._port}/health"
                ) as resp:
                    return resp.status == 200
        except ImportError:
            # aiohttp not available - can't probe, assume healthy
            logger.warning(
                f"[{self.name}] aiohttp not available, skipping health probe"
            )
            return True
        except asyncio.TimeoutError:
            logger.debug(f"[{self.name}] Health probe timed out")
            return False
        except Exception as e:
            # Catch aiohttp client errors and connection errors
            logger.debug(f"[{self.name}] Health probe error: {type(e).__name__}: {e}")
            return False

    async def _handle_server_failure(self) -> None:
        """Handle HTTP server failure with recovery attempts.

        Attempts recovery by waiting and re-probing. If all recovery attempts
        fail, emits zombie detection event and terminates the process.
        """
        self._recovery_triggered = True
        logger.error(
            f"[{self.name}] HTTP server unresponsive after "
            f"{self._consecutive_failures} consecutive failures. "
            f"Attempting recovery ({self._config.recovery_attempts} attempts)..."
        )

        for attempt in range(1, self._config.recovery_attempts + 1):
            logger.info(
                f"[{self.name}] Recovery attempt {attempt}/{self._config.recovery_attempts}"
            )
            await asyncio.sleep(self._config.recovery_delay_seconds)

            if await self._probe_local_health():
                logger.info(
                    f"[{self.name}] HTTP server recovered on attempt {attempt}"
                )
                self._consecutive_failures = 0
                self._last_success_time = time.time()
                self._recovery_triggered = False
                return

        # Recovery failed - terminate process
        logger.critical(
            f"[{self.name}] HTTP server unresponsive after "
            f"{self._config.failure_threshold} failures and "
            f"{self._config.recovery_attempts} recovery attempts. "
            f"Terminating process (exit code {self._config.exit_code_http_server_failed}) "
            f"to trigger systemd restart."
        )

        # Emit event before terminating (best effort)
        await self._emit_zombie_detected_event()

        # Use os._exit() to force immediate termination
        # sys.exit() may be caught by exception handlers, os._exit() is guaranteed
        os._exit(self._config.exit_code_http_server_failed)

    async def _emit_zombie_detected_event(self) -> None:
        """Emit P2P_ZOMBIE_DETECTED event for monitoring/alerting.

        Best-effort emission - failures are logged but don't prevent termination.
        """
        try:
            from app.distributed.data_events import DataEventType

            # Try to get event router
            try:
                from app.coordination.event_router import emit_event

                await asyncio.to_thread(
                    emit_event,
                    DataEventType.P2P_ZOMBIE_DETECTED,
                    {
                        "port": self._port,
                        "consecutive_failures": self._consecutive_failures,
                        "total_failures": self._total_failures,
                        "total_probes": self._total_probes,
                        "last_success_time": self._last_success_time,
                        "uptime_seconds": time.time() - self._start_time,
                        "exit_code": self._config.exit_code_http_server_failed,
                    },
                )
                logger.info(f"[{self.name}] Emitted P2P_ZOMBIE_DETECTED event")
            except ImportError:
                logger.warning(f"[{self.name}] event_router not available")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to emit event: {e}")
        except ImportError:
            logger.warning(f"[{self.name}] data_events not available")

    def health_check(self) -> "HealthCheckResult":
        """Check loop health for DaemonManager integration.

        Returns:
            HealthCheckResult with HTTP server probe statistics
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            return {  # type: ignore[return-value]
                "healthy": self._running and not self._recovery_triggered,
                "status": "running" if self._running else "stopped",
                "message": f"HTTP health loop: {self._consecutive_failures} consecutive failures",
                "details": self._get_health_details(),
            }

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message=f"Loop {self.name} is stopped",
            )

        # Critical: recovery in progress
        if self._recovery_triggered:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"HTTP server recovery in progress ({self._consecutive_failures} failures)",
                details=self._get_health_details(),
            )

        # Warning: failures but not yet at threshold
        if self._consecutive_failures > 0:
            return HealthCheckResult(
                healthy=True,  # Still operational
                status=CoordinatorStatus.DEGRADED,
                message=f"HTTP server probe failures: {self._consecutive_failures}/{self._config.failure_threshold}",
                details=self._get_health_details(),
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"HTTP server healthy (port {self._port})",
            details=self._get_health_details(),
        )

    def _get_health_details(self) -> dict[str, Any]:
        """Get detailed health statistics."""
        uptime = time.time() - self._start_time
        success_rate = (
            ((self._total_probes - self._total_failures) / self._total_probes * 100)
            if self._total_probes > 0
            else 100.0
        )

        return {
            "port": self._port,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self._config.failure_threshold,
            "total_probes": self._total_probes,
            "total_failures": self._total_failures,
            "success_rate": f"{success_rate:.1f}%",
            "last_success_time": self._last_success_time,
            "uptime_seconds": uptime,
            "recovery_triggered": self._recovery_triggered,
            "in_grace_period": uptime < self._config.startup_grace_period_seconds,
        }
