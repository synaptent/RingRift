"""P2P Recovery Daemon for 48-Hour Autonomous Operation.

This daemon monitors the P2P cluster health and automatically restarts
the P2P orchestrator when it becomes unhealthy or partitioned.

Key features:
- Health checks via /status endpoint
- Consecutive failure tracking before restart
- Cooldown period between restarts
- Event emission for monitoring

December 2025: Created for 48-hour autonomous operation enablement.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

from app.coordination.base_daemon import BaseDaemon, DaemonConfig
from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(kw_only=True)
class P2PRecoveryConfig(DaemonConfig):
    """Configuration for P2P Recovery daemon.

    Attributes:
        check_interval_seconds: How often to check P2P health (default: 60s)
        health_endpoint: P2P status endpoint URL
        max_consecutive_failures: Failures before triggering restart (default: 3)
        restart_cooldown_seconds: Minimum time between restarts (default: 5 min)
        health_timeout_seconds: Timeout for health check request (default: 10s)
        min_alive_peers: Minimum peers required for healthy cluster (default: 3)
        startup_grace_seconds: Grace period after restart before checking (default: 30s)
    """

    check_interval_seconds: int = 60
    health_endpoint: str = "http://localhost:8770/status"
    max_consecutive_failures: int = 3
    restart_cooldown_seconds: int = 300  # 5 minutes
    health_timeout_seconds: float = 10.0
    min_alive_peers: int = 3
    startup_grace_seconds: int = 30

    @classmethod
    def from_env(cls) -> "P2PRecoveryConfig":
        """Load config from environment variables."""
        config = cls()

        if os.environ.get("RINGRIFT_P2P_RECOVERY_ENABLED"):
            config.enabled = os.environ.get("RINGRIFT_P2P_RECOVERY_ENABLED", "1") == "1"

        if os.environ.get("RINGRIFT_P2P_RECOVERY_INTERVAL"):
            try:
                config.check_interval_seconds = int(
                    os.environ.get("RINGRIFT_P2P_RECOVERY_INTERVAL", "60")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_P2P_HEALTH_ENDPOINT"):
            config.health_endpoint = os.environ.get(
                "RINGRIFT_P2P_HEALTH_ENDPOINT", config.health_endpoint
            )

        if os.environ.get("RINGRIFT_P2P_MAX_FAILURES"):
            try:
                config.max_consecutive_failures = int(
                    os.environ.get("RINGRIFT_P2P_MAX_FAILURES", "3")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_P2P_RESTART_COOLDOWN"):
            try:
                config.restart_cooldown_seconds = int(
                    os.environ.get("RINGRIFT_P2P_RESTART_COOLDOWN", "300")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_P2P_MIN_PEERS"):
            try:
                config.min_alive_peers = int(
                    os.environ.get("RINGRIFT_P2P_MIN_PEERS", "3")
                )
            except ValueError:
                pass

        return config


# =============================================================================
# P2P Recovery Daemon
# =============================================================================


class P2PRecoveryDaemon(BaseDaemon[P2PRecoveryConfig]):
    """Daemon that monitors P2P cluster health and triggers auto-recovery.

    Workflow:
    1. Periodically check P2P /status endpoint
    2. Track consecutive failures
    3. After max_consecutive_failures, restart P2P orchestrator
    4. Respect cooldown period between restarts

    Events Emitted:
    - P2P_RESTART_TRIGGERED: When restart is initiated
    - P2P_HEALTH_RECOVERED: When P2P becomes healthy after being unhealthy
    """

    _instance: "P2PRecoveryDaemon | None" = None

    def __init__(self, config: P2PRecoveryConfig | None = None):
        super().__init__(config)
        self._consecutive_failures = 0
        self._last_restart_time = 0.0
        self._total_restarts = 0
        self._last_healthy_time = time.time()
        self._was_unhealthy = False
        self._last_status: dict[str, Any] = {}
        self._startup_time = time.time()

    @staticmethod
    def _get_default_config() -> P2PRecoveryConfig:
        """Return default config."""
        return P2PRecoveryConfig.from_env()

    def _get_daemon_name(self) -> str:
        """Return daemon name."""
        return "P2PRecovery"

    @classmethod
    def get_instance(cls) -> "P2PRecoveryDaemon":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    # =========================================================================
    # Main Cycle
    # =========================================================================

    async def _run_cycle(self) -> None:
        """Main daemon cycle - check P2P health and recover if needed."""
        # Skip during startup grace period
        if time.time() - self._startup_time < self.config.startup_grace_seconds:
            logger.debug("P2P Recovery: In startup grace period, skipping check")
            return

        # Check P2P health
        is_healthy, status = await self._check_p2p_health()
        self._last_status = status

        if is_healthy:
            # Reset failure counter
            if self._consecutive_failures > 0:
                logger.info(
                    f"P2P healthy after {self._consecutive_failures} failures"
                )
            self._consecutive_failures = 0
            self._last_healthy_time = time.time()

            # Emit recovery event if was previously unhealthy
            if self._was_unhealthy:
                await self._emit_recovery_event(status)
                self._was_unhealthy = False
        else:
            # Increment failure counter
            self._consecutive_failures += 1
            logger.warning(
                f"P2P health check failed ({self._consecutive_failures}/{self.config.max_consecutive_failures})"
            )

            # Check if we should restart
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                if self._can_restart():
                    await self._restart_p2p()
                    self._consecutive_failures = 0
                    self._was_unhealthy = True
                else:
                    cooldown_remaining = self._get_cooldown_remaining()
                    logger.warning(
                        f"P2P restart needed but in cooldown ({cooldown_remaining:.0f}s remaining)"
                    )

    async def _check_p2p_health(self) -> tuple[bool, dict[str, Any]]:
        """Check P2P orchestrator health via /status endpoint.

        Returns:
            Tuple of (is_healthy, status_dict)
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.config.health_endpoint,
                    timeout=aiohttp.ClientTimeout(total=self.config.health_timeout_seconds),
                ) as resp:
                    if resp.status != 200:
                        return False, {"error": f"HTTP {resp.status}"}

                    data = await resp.json()

                    # Check key health indicators
                    alive_peers = data.get("alive_peers", 0)
                    leader_id = data.get("leader_id")
                    role = data.get("role", "unknown")

                    # Consider healthy if:
                    # 1. Has minimum peers
                    # 2. Has a leader (or is leader)
                    is_healthy = (
                        alive_peers >= self.config.min_alive_peers
                        and leader_id is not None
                    )

                    return is_healthy, {
                        "alive_peers": alive_peers,
                        "leader_id": leader_id,
                        "role": role,
                        "is_healthy": is_healthy,
                    }
        except asyncio.TimeoutError:
            return False, {"error": "timeout"}
        except Exception as e:
            return False, {"error": str(e)}

    def _can_restart(self) -> bool:
        """Check if we can restart (cooldown has passed)."""
        if self._last_restart_time == 0:
            return True
        elapsed = time.time() - self._last_restart_time
        return elapsed >= self.config.restart_cooldown_seconds

    def _get_cooldown_remaining(self) -> float:
        """Get seconds remaining in cooldown period."""
        if self._last_restart_time == 0:
            return 0
        elapsed = time.time() - self._last_restart_time
        return max(0, self.config.restart_cooldown_seconds - elapsed)

    async def _restart_p2p(self) -> None:
        """Restart the P2P orchestrator process."""
        logger.warning("Initiating P2P orchestrator restart")

        # Update tracking
        self._last_restart_time = time.time()
        self._total_restarts += 1
        self._startup_time = time.time()  # Reset for grace period

        # Emit restart event
        await self._emit_restart_event()

        try:
            # Kill existing P2P process
            kill_result = subprocess.run(
                ["pkill", "-f", "p2p_orchestrator.py"],
                capture_output=True,
                timeout=10,
            )
            if kill_result.returncode == 0:
                logger.info("Killed existing P2P process")
            else:
                logger.debug("No existing P2P process to kill")

            # Wait for process to fully terminate
            await asyncio.sleep(5)

            # Start new P2P process
            # The P2P orchestrator should be started by the master loop,
            # but we can also start it directly if needed
            p2p_script = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "scripts",
                "p2p_orchestrator.py",
            )

            if os.path.exists(p2p_script):
                # Start in background
                process = subprocess.Popen(
                    [sys.executable, p2p_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                logger.info(f"Started new P2P process (PID {process.pid})")
            else:
                logger.warning(
                    "P2P script not found, relying on master_loop to restart"
                )

        except subprocess.TimeoutExpired:
            logger.error("Timeout killing P2P process")
        except Exception as e:
            logger.error(f"Error restarting P2P: {e}")
            self._errors_count += 1
            self._last_error = str(e)

    # =========================================================================
    # Event Emission
    # =========================================================================

    async def _emit_restart_event(self) -> None:
        """Emit event when P2P restart is triggered."""
        try:
            from app.distributed.data_events import emit_event

            await emit_event(
                "P2P_RESTART_TRIGGERED",
                {
                    "consecutive_failures": self._consecutive_failures,
                    "last_status": self._last_status,
                    "total_restarts": self._total_restarts,
                    "unhealthy_duration_seconds": time.time() - self._last_healthy_time,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit P2P_RESTART_TRIGGERED: {e}")

    async def _emit_recovery_event(self, status: dict[str, Any]) -> None:
        """Emit event when P2P recovers."""
        try:
            from app.distributed.data_events import emit_event

            await emit_event(
                "P2P_HEALTH_RECOVERED",
                {
                    "alive_peers": status.get("alive_peers", 0),
                    "leader_id": status.get("leader_id"),
                    "recovery_duration_seconds": time.time() - self._last_healthy_time,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit P2P_HEALTH_RECOVERED: {e}")

    # =========================================================================
    # Health & Status
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="P2PRecovery not running",
                details={},
            )

        # Check if P2P is currently unhealthy
        if self._consecutive_failures >= self.config.max_consecutive_failures:
            return HealthCheckResult(
                healthy=True,  # Daemon is healthy, P2P is not
                status=CoordinatorStatus.RUNNING,
                message=f"P2P unhealthy, {self._consecutive_failures} consecutive failures",
                details={
                    "p2p_healthy": False,
                    "consecutive_failures": self._consecutive_failures,
                    "last_status": self._last_status,
                    "total_restarts": self._total_restarts,
                },
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="P2P cluster healthy",
            details={
                "p2p_healthy": True,
                "consecutive_failures": self._consecutive_failures,
                "cycles_completed": self._cycles_completed,
                "total_restarts": self._total_restarts,
                "last_status": self._last_status,
            },
        )

    def get_status(self) -> dict[str, Any]:
        """Return detailed status."""
        base_status = super().get_status()

        base_status["p2p_status"] = {
            "consecutive_failures": self._consecutive_failures,
            "last_restart_time": self._last_restart_time,
            "total_restarts": self._total_restarts,
            "last_healthy_time": self._last_healthy_time,
            "unhealthy_duration": time.time() - self._last_healthy_time if self._was_unhealthy else 0,
            "cooldown_remaining": self._get_cooldown_remaining(),
            "last_status": self._last_status,
        }

        return base_status

    def is_p2p_healthy(self) -> bool:
        """Check if P2P is currently healthy based on last check."""
        return self._consecutive_failures < self.config.max_consecutive_failures


# =============================================================================
# Singleton Accessor
# =============================================================================


def get_p2p_recovery_daemon() -> P2PRecoveryDaemon:
    """Get the singleton P2PRecovery instance."""
    return P2PRecoveryDaemon.get_instance()
