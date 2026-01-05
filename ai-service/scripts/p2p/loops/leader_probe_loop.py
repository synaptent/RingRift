"""Fast Leader Health Probe Loop.

Jan 4, 2026 - Phase 5 of P2P Cluster Resilience.

Problem: When the leader becomes unreachable, the cluster relies on gossip timeouts
and periodic election checks to detect the failure. This can take 60-180 seconds,
during which training and selfplay dispatch is blocked.

Solution: LeaderProbeLoop actively probes the leader every 10 seconds. After 6
consecutive failures (60s total), it triggers a forced election to recover from
unreachable leader conditions faster.

Features:
- Probes leader health via HTTP /health endpoint
- Configurable failure threshold (default: 6 consecutive failures = 60s)
- Triggers election via _start_election() on threshold breach
- Skips probing when this node is the leader
- Emits events for observability (LEADER_PROBE_FAILED, LEADER_PROBE_RECOVERED)

Usage:
    from scripts.p2p.loops.leader_probe_loop import LeaderProbeLoop

    loop = LeaderProbeLoop(orchestrator)
    loop.start_background()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from .base import BaseLoop

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LeaderProbeLoop(BaseLoop):
    """Actively probe leader and trigger election if unreachable.

    This loop runs on ALL non-leader nodes and actively checks if the current
    leader is reachable. If the leader becomes unreachable for too long, it
    triggers a forced election to quickly recover from leader failures.

    This is a proactive measure that complements the reactive gossip-based
    failure detection, providing faster recovery for 48-hour autonomous operation.
    """

    # Class-level attributes for LoopManager registration
    depends_on: list[str] = []  # No dependencies - runs independently

    def __init__(
        self,
        orchestrator: Any,
        *,
        probe_interval: float = 10.0,
        failure_threshold: int = 6,
        probe_timeout: float = 5.0,
    ) -> None:
        """Initialize the leader probe loop.

        Args:
            orchestrator: P2POrchestrator instance
            probe_interval: Seconds between probes (default: 10s)
            failure_threshold: Consecutive failures before triggering election (default: 6 = 60s)
            probe_timeout: Timeout for each probe request (default: 5s)
        """
        super().__init__(
            name="leader_probe",
            interval=probe_interval,
            enabled=True,
            depends_on=[],
        )

        self._orchestrator = orchestrator
        self._failure_threshold = failure_threshold
        self._probe_timeout = probe_timeout
        self._consecutive_failures = 0
        self._last_success_time = time.time()
        self._election_triggered_recently = False
        self._election_cooldown = 120.0  # Don't trigger elections too frequently

    async def _run_once(self) -> None:
        """Single probe iteration - check if leader is reachable.

        Called by BaseLoop.run_forever() at probe_interval intervals.
        """
        # Skip if we are the leader
        if self._is_leader():
            self._consecutive_failures = 0
            return

        # Skip if no known leader
        leader_id = self._get_leader_id()
        if not leader_id:
            # No leader known - don't probe, let election process handle it
            return

        # Probe the leader
        leader_reachable = await self._probe_leader(leader_id)

        if leader_reachable:
            self._on_probe_success()
        else:
            await self._on_probe_failure(leader_id)

    def _is_leader(self) -> bool:
        """Check if this node is the current leader."""
        try:
            return bool(getattr(self._orchestrator, "is_leader", False))
        except Exception:
            return False

    def _get_leader_id(self) -> str | None:
        """Get the current leader ID."""
        try:
            return getattr(self._orchestrator, "leader_id", None)
        except Exception:
            return None

    async def _probe_leader(self, leader_id: str) -> bool:
        """Check if leader is reachable via HTTP health check.

        Args:
            leader_id: The leader node ID to probe

        Returns:
            True if leader responded successfully, False otherwise
        """
        try:
            # Get leader URLs
            urls_for_peer = getattr(self._orchestrator, "_urls_for_peer", None)
            if urls_for_peer is None:
                logger.debug(f"[LeaderProbe] No urls_for_peer method available")
                return True  # Assume healthy if we can't probe

            urls = urls_for_peer(leader_id, "health")
            if not urls:
                logger.debug(f"[LeaderProbe] No health URLs for leader {leader_id}")
                return True  # Assume healthy if no URLs configured

            # Try each URL
            import aiohttp

            async with aiohttp.ClientSession() as session:
                for url in urls:
                    try:
                        async with session.get(
                            url,
                            timeout=aiohttp.ClientTimeout(total=self._probe_timeout),
                        ) as response:
                            if response.status == 200:
                                logger.debug(f"[LeaderProbe] Leader {leader_id} reachable via {url}")
                                return True
                    except asyncio.TimeoutError:
                        logger.debug(f"[LeaderProbe] Timeout probing {url}")
                    except Exception as e:
                        logger.debug(f"[LeaderProbe] Error probing {url}: {e}")

            # All URLs failed
            return False

        except Exception as e:
            logger.debug(f"[LeaderProbe] Probe error: {e}")
            return False

    def _on_probe_success(self) -> None:
        """Handle successful probe - reset failure counter."""
        if self._consecutive_failures > 0:
            logger.info(
                f"[LeaderProbe] Leader recovered after {self._consecutive_failures} failures"
            )
            self._emit_event("LEADER_PROBE_RECOVERED", {
                "failures_before_recovery": self._consecutive_failures,
                "downtime_seconds": time.time() - self._last_success_time,
            })

        self._consecutive_failures = 0
        self._last_success_time = time.time()
        self._election_triggered_recently = False

    async def _on_probe_failure(self, leader_id: str) -> None:
        """Handle probe failure - increment counter and maybe trigger election.

        Args:
            leader_id: The leader ID that failed to respond
        """
        self._consecutive_failures += 1
        logger.warning(
            f"[LeaderProbe] Leader {leader_id} unreachable "
            f"({self._consecutive_failures}/{self._failure_threshold})"
        )

        self._emit_event("LEADER_PROBE_FAILED", {
            "leader_id": leader_id,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self._failure_threshold,
        })

        # Check if we should trigger election
        if self._consecutive_failures >= self._failure_threshold:
            await self._trigger_election(leader_id)

    async def _trigger_election(self, unreachable_leader: str) -> None:
        """Trigger a forced election due to unreachable leader.

        Args:
            unreachable_leader: The leader ID that became unreachable
        """
        # Cooldown check to prevent election storms
        if self._election_triggered_recently:
            logger.debug("[LeaderProbe] Election cooldown active, skipping")
            return

        downtime = time.time() - self._last_success_time
        logger.error(
            f"[LeaderProbe] Leader {unreachable_leader} unreachable for {downtime:.0f}s, "
            f"triggering election"
        )

        self._election_triggered_recently = True

        # Schedule cooldown reset
        async def reset_cooldown():
            await asyncio.sleep(self._election_cooldown)
            self._election_triggered_recently = False

        asyncio.create_task(reset_cooldown())

        # Trigger election
        try:
            # Jan 5, 2026: Check quorum before starting election
            # If quorum is LOST, election cannot succeed - skip to avoid election storms
            check_quorum = getattr(self._orchestrator, "_check_quorum_health", None)
            if check_quorum:
                from scripts.p2p.leader_election import QuorumHealthLevel
                quorum_health = check_quorum()
                if quorum_health == QuorumHealthLevel.LOST:
                    logger.warning(
                        "[LeaderProbe] Quorum LOST - cannot hold election. "
                        "Wait for voters to recover."
                    )
                    self._emit_event("ELECTION_BLOCKED_QUORUM_LOST", {
                        "unreachable_leader": unreachable_leader,
                        "quorum_health": quorum_health.value,
                    })
                    return

            start_election = getattr(self._orchestrator, "_start_election", None)
            if start_election:
                await start_election(reason="leader_unreachable_probe")
                logger.info("[LeaderProbe] Election triggered successfully")
            else:
                logger.warning("[LeaderProbe] No _start_election method available")
        except Exception as e:
            logger.error(f"[LeaderProbe] Failed to start election: {e}")

        # Reset failure counter after triggering election
        self._consecutive_failures = 0

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit an event for observability.

        Args:
            event_type: Event type name
            payload: Event payload dictionary
        """
        try:
            from app.coordination.event_router import emit_event

            emit_event(event_type, {
                "source": "leader_probe_loop",
                "timestamp": time.time(),
                **payload,
            })
        except Exception as e:
            logger.debug(f"[LeaderProbe] Failed to emit event {event_type}: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get loop status for health checks."""
        return {
            "name": self.name,
            "running": self.running,
            "enabled": self.enabled,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self._failure_threshold,
            "last_success_time": self._last_success_time,
            "seconds_since_success": time.time() - self._last_success_time,
            "election_triggered_recently": self._election_triggered_recently,
        }

    def health_check(self) -> Any:
        """Check loop health with leader-specific status.

        Jan 2026: Added for DaemonManager integration and better observability.
        Provides more detailed health info than base class implementation.

        Returns:
            HealthCheckResult with leader probe status
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            return {
                "healthy": self.running,
                "status": "running" if self.running else "stopped",
                "message": f"LeaderProbeLoop {'running' if self.running else 'stopped'}",
                "details": self.get_status(),
            }

        # If we're the leader, we don't probe - always healthy
        if self._is_leader():
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message="This node is the leader (no probing needed)",
                details={"is_leader": True, "running": self.running},
            )

        # Not running
        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="LeaderProbeLoop is stopped",
                details={"running": False},
            )

        # Check if leader is unreachable (approaching election trigger)
        if self._consecutive_failures >= self._failure_threshold:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Leader unreachable ({self._consecutive_failures} failures), election triggered",
                details={
                    "consecutive_failures": self._consecutive_failures,
                    "failure_threshold": self._failure_threshold,
                    "seconds_since_success": time.time() - self._last_success_time,
                    "election_triggered_recently": self._election_triggered_recently,
                },
            )

        # Check if leader probe is degraded (some failures but not threshold)
        if self._consecutive_failures > 0:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Leader probe experiencing failures ({self._consecutive_failures}/{self._failure_threshold})",
                details={
                    "consecutive_failures": self._consecutive_failures,
                    "failure_threshold": self._failure_threshold,
                    "seconds_since_success": time.time() - self._last_success_time,
                },
            )

        # Healthy - leader reachable
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="Leader probe healthy",
            details={
                "consecutive_failures": 0,
                "seconds_since_success": time.time() - self._last_success_time,
                "leader_id": self._get_leader_id(),
            },
        )
