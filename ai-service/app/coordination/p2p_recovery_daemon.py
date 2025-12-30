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
        isolation_check_enabled: Enable network isolation detection (default: True)
        min_peer_ratio: Trigger isolation recovery if P2P/Tailscale ratio below this (default: 0.5)
        isolation_consecutive_checks: Consecutive isolation checks before action (default: 3)
        # Dec 29, 2025: Self-healing enhancements for quorum and leader gaps
        max_leader_gap_seconds: Maximum seconds without a leader before forcing election (default: 120)
        quorum_recovery_enabled: Enable automatic quorum recovery (default: True)
        leader_election_endpoint: Endpoint to trigger leader election
        # Dec 30, 2025: Exponential backoff for restart attempts
        restart_backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        max_cooldown_seconds: Maximum cooldown between restarts (default: 30 min)
    """

    check_interval_seconds: int = 60
    health_endpoint: str = "http://localhost:8770/status"
    max_consecutive_failures: int = 3
    restart_cooldown_seconds: int = 300  # 5 minutes (initial cooldown)
    restart_backoff_multiplier: float = 2.0  # Double cooldown each attempt
    max_cooldown_seconds: int = 1800  # 30 minutes max cooldown
    health_timeout_seconds: float = 10.0
    min_alive_peers: int = 3
    startup_grace_seconds: int = 30
    # Network isolation detection (December 2025)
    isolation_check_enabled: bool = True
    min_peer_ratio: float = 0.5  # Trigger if P2P sees < 50% of Tailscale peers
    isolation_consecutive_checks: int = 3  # Require 3 checks (~3 min) before action
    # Dec 29, 2025: Self-healing for quorum and leader gaps
    max_leader_gap_seconds: int = 120  # Force election if no leader for 2 minutes
    quorum_recovery_enabled: bool = True
    leader_election_endpoint: str = "http://localhost:8770/election/start"

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

        # Network isolation detection config
        if os.environ.get("RINGRIFT_P2P_ISOLATION_ENABLED"):
            config.isolation_check_enabled = os.environ.get(
                "RINGRIFT_P2P_ISOLATION_ENABLED", "1"
            ) == "1"

        if os.environ.get("RINGRIFT_P2P_MIN_PEER_RATIO"):
            try:
                config.min_peer_ratio = float(
                    os.environ.get("RINGRIFT_P2P_MIN_PEER_RATIO", "0.5")
                )
            except ValueError:
                pass

        if os.environ.get("RINGRIFT_P2P_ISOLATION_CHECKS"):
            try:
                config.isolation_consecutive_checks = int(
                    os.environ.get("RINGRIFT_P2P_ISOLATION_CHECKS", "3")
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
        # Network isolation detection (December 2025)
        self._consecutive_isolation_checks = 0
        self._last_tailscale_count = 0
        self._isolation_triggered_restarts = 0
        # Dec 29, 2025: Self-healing for leader gaps and quorum
        self._last_leader_seen_time = time.time()
        self._leader_gap_elections_triggered = 0
        self._quorum_recovery_attempts = 0
        # Dec 30, 2025: Exponential backoff for restarts
        self._restart_attempt_count = 0  # Consecutive restart attempts without recovery

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

        # Also check for network isolation (December 2025)
        alive_peers = status.get("alive_peers", 0)
        is_isolated, isolation_details = await self._check_network_isolation(alive_peers)
        status["isolation"] = isolation_details

        # Dec 29, 2025: Track leader presence for leader gap detection
        leader_id = status.get("leader_id")
        if leader_id:
            self._last_leader_seen_time = time.time()

        if is_healthy and not is_isolated:
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
                # Dec 30, 2025: Reset exponential backoff on recovery
                if self._restart_attempt_count > 0:
                    logger.info(
                        f"P2P recovered after {self._restart_attempt_count} restart attempts, "
                        "resetting backoff"
                    )
                    self._restart_attempt_count = 0

        # Dec 29, 2025: Check for leader gap and trigger election if needed
        leader_gap_seconds = time.time() - self._last_leader_seen_time
        if not leader_id and leader_gap_seconds > self.config.max_leader_gap_seconds:
            logger.warning(
                f"Leader gap detected: no leader for {leader_gap_seconds:.0f}s "
                f"(threshold: {self.config.max_leader_gap_seconds}s), triggering election"
            )
            await self._trigger_leader_election()
            return  # Skip other recovery actions this cycle

        elif is_isolated and self._consecutive_isolation_checks >= self.config.isolation_consecutive_checks:
            # Network isolation confirmed - trigger restart
            logger.warning(
                f"Network isolation confirmed after {self._consecutive_isolation_checks} checks, "
                f"triggering P2P restart"
            )
            await self._emit_isolation_event(isolation_details)

            if self._can_restart():
                await self._restart_p2p()
                self._consecutive_isolation_checks = 0
                self._isolation_triggered_restarts += 1
                self._was_unhealthy = True
            else:
                cooldown_remaining = self._get_cooldown_remaining()
                logger.warning(
                    f"P2P restart needed for isolation but in cooldown ({cooldown_remaining:.0f}s remaining)"
                )

        elif not is_healthy:
            # Standard health check failure
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

    async def _get_tailscale_online_count(self) -> int:
        """Get count of online Tailscale peers from local Tailscale status.

        Uses TailscaleChecker to query the mesh network and count peers
        that are marked as online.

        Returns:
            Number of online Tailscale peers, or 0 if query fails.
        """
        try:
            from app.coordination.node_availability.providers.tailscale_checker import (
                TailscaleChecker,
            )
            from app.config.cluster_config import load_cluster_config

            checker = TailscaleChecker()
            if not await checker.check_api_availability():
                logger.debug("Tailscale CLI not available for isolation check")
                return 0

            config = load_cluster_config()
            online_nodes = await checker.get_online_nodes(config.hosts_raw)
            count = len(online_nodes)
            self._last_tailscale_count = count
            return count

        except Exception as e:
            logger.debug(f"Failed to get Tailscale peer count: {e}")
            return 0  # Fail open - don't trigger isolation on error

    async def _check_network_isolation(
        self, alive_peers: int
    ) -> tuple[bool, dict[str, Any]]:
        """Check if we're experiencing network isolation.

        Compares P2P visible peers to Tailscale online peers.
        If P2P sees significantly fewer than Tailscale, we may be isolated.

        Args:
            alive_peers: Number of peers visible in P2P cluster

        Returns:
            Tuple of (is_isolated, details_dict)
        """
        if not self.config.isolation_check_enabled:
            return False, {"isolation_check": "disabled"}

        tailscale_count = await self._get_tailscale_online_count()

        if tailscale_count == 0:
            # Can't determine isolation without Tailscale data
            return False, {"isolation_check": "no_tailscale_data"}

        peer_ratio = alive_peers / tailscale_count
        is_isolated = peer_ratio < self.config.min_peer_ratio

        details = {
            "p2p_peers": alive_peers,
            "tailscale_peers": tailscale_count,
            "peer_ratio": round(peer_ratio, 3),
            "min_ratio": self.config.min_peer_ratio,
            "is_isolated": is_isolated,
        }

        if is_isolated:
            self._consecutive_isolation_checks += 1
            details["consecutive_isolation_checks"] = self._consecutive_isolation_checks
            logger.warning(
                f"Network isolation detected: P2P sees {alive_peers} peers "
                f"but Tailscale shows {tailscale_count} online "
                f"(ratio={peer_ratio:.2f}, check {self._consecutive_isolation_checks}/"
                f"{self.config.isolation_consecutive_checks})"
            )
        else:
            if self._consecutive_isolation_checks > 0:
                logger.info(
                    f"Network isolation cleared after {self._consecutive_isolation_checks} checks"
                )
            self._consecutive_isolation_checks = 0

        return is_isolated, details

    def _get_current_cooldown(self) -> float:
        """Calculate current cooldown with exponential backoff.

        Dec 30, 2025: Implements exponential backoff for consecutive restart
        attempts. Cooldown doubles after each failed restart until max is reached.

        Formula: base_cooldown * (multiplier ** attempt_count)
        Capped at max_cooldown_seconds.
        """
        if self._restart_attempt_count == 0:
            return float(self.config.restart_cooldown_seconds)

        backoff_cooldown = (
            self.config.restart_cooldown_seconds
            * (self.config.restart_backoff_multiplier ** self._restart_attempt_count)
        )
        return min(backoff_cooldown, float(self.config.max_cooldown_seconds))

    def _can_restart(self) -> bool:
        """Check if we can restart (cooldown has passed)."""
        if self._last_restart_time == 0:
            return True
        elapsed = time.time() - self._last_restart_time
        current_cooldown = self._get_current_cooldown()
        return elapsed >= current_cooldown

    def _get_cooldown_remaining(self) -> float:
        """Get seconds remaining in cooldown period (with exponential backoff)."""
        if self._last_restart_time == 0:
            return 0
        elapsed = time.time() - self._last_restart_time
        current_cooldown = self._get_current_cooldown()
        return max(0, current_cooldown - elapsed)

    async def _restart_p2p(self) -> None:
        """Restart the P2P orchestrator process.

        December 2025: Uses asyncio.create_subprocess_exec to avoid blocking the event loop.
        Dec 30, 2025: Tracks restart attempts for exponential backoff.
        """
        # Increment restart attempt counter for exponential backoff
        self._restart_attempt_count += 1
        current_cooldown = self._get_current_cooldown()
        logger.warning(
            f"Initiating P2P orchestrator restart (attempt {self._restart_attempt_count}, "
            f"next cooldown: {current_cooldown:.0f}s)"
        )

        # Update tracking
        self._last_restart_time = time.time()
        self._total_restarts += 1
        self._startup_time = time.time()  # Reset for grace period

        # Emit restart event
        await self._emit_restart_event()

        try:
            # Kill existing P2P process (non-blocking)
            proc = await asyncio.create_subprocess_exec(
                "pkill", "-f", "p2p_orchestrator.py",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
                if proc.returncode == 0:
                    logger.info("Killed existing P2P process")
                else:
                    logger.debug("No existing P2P process to kill")
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error("Timeout killing P2P process")

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
                # December 30, 2025: Wrap Popen in asyncio.to_thread for consistency
                # Popen is mostly non-blocking, but fork can briefly block
                def _start_p2p() -> subprocess.Popen[bytes]:
                    return subprocess.Popen(
                        [sys.executable, p2p_script],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )

                process = await asyncio.to_thread(_start_p2p)
                logger.info(f"Started new P2P process (PID {process.pid})")
            else:
                logger.warning(
                    "P2P script not found, relying on master_loop to restart"
                )

        except Exception as e:
            logger.error(f"Error restarting P2P: {e}")
            self._errors_count += 1
            self._last_error = str(e)

    async def _trigger_leader_election(self) -> bool:
        """Trigger a leader election via the P2P orchestrator API.

        Dec 29, 2025: Added for self-healing when leader gaps are detected.
        This allows the daemon to proactively trigger elections without
        requiring a full P2P restart.

        Returns:
            True if election was triggered successfully, False otherwise.
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.leader_election_endpoint,
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    if resp.status == 200:
                        self._leader_gap_elections_triggered += 1
                        logger.info(
                            f"Leader election triggered successfully "
                            f"(total: {self._leader_gap_elections_triggered})"
                        )
                        await self._emit_leader_gap_event()
                        return True
                    else:
                        logger.warning(
                            f"Leader election trigger failed: HTTP {resp.status}"
                        )
                        return False

        except asyncio.TimeoutError:
            logger.warning("Leader election trigger timed out")
            return False
        except Exception as e:
            logger.error(f"Error triggering leader election: {e}")
            return False

    # =========================================================================
    # Event Emission
    # =========================================================================

    async def _emit_restart_event(self) -> None:
        """Emit event when P2P restart is triggered."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.P2P_RESTART_TRIGGERED,
                consecutive_failures=self._consecutive_failures,
                last_status=self._last_status,
                total_restarts=self._total_restarts,
                unhealthy_duration_seconds=time.time() - self._last_healthy_time,
                source="P2PRecoveryDaemon",
            )
        except Exception as e:
            logger.debug(f"Failed to emit P2P_RESTART_TRIGGERED: {e}")

    async def _emit_recovery_event(self, status: dict[str, Any]) -> None:
        """Emit event when P2P recovers."""
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.P2P_HEALTH_RECOVERED,
                alive_peers=status.get("alive_peers", 0),
                leader_id=status.get("leader_id"),
                recovery_duration_seconds=time.time() - self._last_healthy_time,
                source="P2PRecoveryDaemon",
            )
        except Exception as e:
            logger.debug(f"Failed to emit P2P_HEALTH_RECOVERED: {e}")

    async def _emit_isolation_event(self, isolation_details: dict[str, Any]) -> None:
        """Emit event when network isolation is detected.

        December 2025: New event for monitoring network partition scenarios.
        """
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            emit_data_event(
                DataEventType.NETWORK_ISOLATION_DETECTED,
                p2p_peers=isolation_details.get("p2p_peers", 0),
                tailscale_peers=isolation_details.get("tailscale_peers", 0),
                peer_ratio=isolation_details.get("peer_ratio", 0),
                consecutive_checks=self._consecutive_isolation_checks,
                isolation_triggered_restarts=self._isolation_triggered_restarts,
                source="P2PRecoveryDaemon",
            )
        except Exception as e:
            logger.debug(f"Failed to emit NETWORK_ISOLATION_DETECTED: {e}")

    async def _emit_leader_gap_event(self) -> None:
        """Emit event when leader gap triggers an election.

        Dec 29, 2025: New event for monitoring leader gap scenarios.
        """
        try:
            from app.distributed.data_events import DataEventType, emit_data_event

            leader_gap_seconds = time.time() - self._last_leader_seen_time
            emit_data_event(
                DataEventType.LEADER_ELECTION_TRIGGERED,
                reason="leader_gap",
                leader_gap_seconds=leader_gap_seconds,
                threshold_seconds=self.config.max_leader_gap_seconds,
                total_leader_gap_elections=self._leader_gap_elections_triggered,
                source="P2PRecoveryDaemon",
            )
        except Exception as e:
            logger.debug(f"Failed to emit LEADER_ELECTION_TRIGGERED: {e}")

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
                    "isolation_checks": self._consecutive_isolation_checks,
                    "restart_attempt_count": self._restart_attempt_count,
                    "current_cooldown_seconds": self._get_current_cooldown(),
                    "cooldown_remaining_seconds": self._get_cooldown_remaining(),
                },
            )

        # Check if network isolation is being detected
        if self._consecutive_isolation_checks > 0:
            return HealthCheckResult(
                healthy=True,  # Daemon is healthy, but isolation detected
                status=CoordinatorStatus.RUNNING,
                message=f"Network isolation detected ({self._consecutive_isolation_checks} checks)",
                details={
                    "p2p_healthy": True,
                    "network_isolated": True,
                    "isolation_checks": self._consecutive_isolation_checks,
                    "isolation_threshold": self.config.isolation_consecutive_checks,
                    "last_tailscale_count": self._last_tailscale_count,
                    "last_status": self._last_status,
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
                "isolation_triggered_restarts": self._isolation_triggered_restarts,
                "leader_gap_elections_triggered": self._leader_gap_elections_triggered,
                "restart_attempt_count": self._restart_attempt_count,
                "current_cooldown_seconds": self._get_current_cooldown(),
                "cooldown_remaining_seconds": self._get_cooldown_remaining(),
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

        # Network isolation detection status (December 2025)
        base_status["isolation_status"] = {
            "enabled": self.config.isolation_check_enabled,
            "consecutive_checks": self._consecutive_isolation_checks,
            "threshold": self.config.isolation_consecutive_checks,
            "min_peer_ratio": self.config.min_peer_ratio,
            "last_tailscale_count": self._last_tailscale_count,
            "isolation_triggered_restarts": self._isolation_triggered_restarts,
        }

        # Dec 29, 2025: Leader gap detection status
        leader_gap_seconds = time.time() - self._last_leader_seen_time
        base_status["leader_gap_status"] = {
            "last_leader_seen_time": self._last_leader_seen_time,
            "current_leader_gap_seconds": leader_gap_seconds,
            "max_leader_gap_seconds": self.config.max_leader_gap_seconds,
            "elections_triggered": self._leader_gap_elections_triggered,
            "quorum_recovery_attempts": self._quorum_recovery_attempts,
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
