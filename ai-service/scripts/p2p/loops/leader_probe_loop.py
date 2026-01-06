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
from collections import deque
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
        probe_backup_candidates: bool = True,
    ) -> None:
        """Initialize the leader probe loop.

        Args:
            orchestrator: P2POrchestrator instance
            probe_interval: Seconds between probes (default: 10s)
            failure_threshold: Consecutive failures before triggering election (default: 6 = 60s)
            probe_timeout: Timeout for each probe request (default: 5s)
            probe_backup_candidates: Whether to probe backup candidates in parallel (default: True)
                Session 17.33 Phase 17: Enables faster failover by pre-probing backup candidates
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

        # Phase 17.2: Backup candidate tracking
        self._probe_backup_candidates = probe_backup_candidates
        self._backup_candidate_status: dict[str, bool] = {}  # node_id -> is_reachable

        # Jan 5, 2026: Latency trending for early warning
        # Track last 10 probe latencies to detect degradation before complete failure
        self._latency_history: deque[float] = deque(maxlen=10)
        self._latency_warning_emitted = False  # Avoid spamming warnings

        # Session 17.39: Backup candidate probing (Phase 10.3)
        # Probe backup candidates every N cycles to maintain warm connections
        # This reduces MTTR from 60s to 20-30s on leader failure
        self._probe_cycle_counter = 0
        self._backup_probe_frequency = 5  # Probe every 5th cycle (~50s at 10s interval)

    async def _run_once(self) -> None:
        """Single probe iteration - check if leader is reachable.

        Called by BaseLoop.run_forever() at probe_interval intervals.
        """
        # Increment cycle counter for backup probing
        self._probe_cycle_counter += 1

        # Skip if we are the leader
        if self._is_leader():
            self._consecutive_failures = 0
            return

        # Skip if no known leader
        leader_id = self._get_leader_id()
        if not leader_id:
            # No leader known - don't probe, let election process handle it
            return

        # Jan 5, 2026: Track probe latency for early warning
        start_time = time.time()

        # Probe the leader
        leader_reachable = await self._probe_leader(leader_id)

        # Record latency for trending analysis
        latency = time.time() - start_time
        self._latency_history.append(latency)

        if leader_reachable:
            self._on_probe_success()
            # Check for latency degradation even on success
            self._check_latency_trend()
        else:
            await self._on_probe_failure(leader_id)

        # Session 17.39 Phase 10.3: Probe backup candidates every Nth cycle
        # This maintains warm connections to potential leader candidates,
        # reducing MTTR from 60s to 20-30s on leader failure
        if (
            self._probe_backup_candidates
            and self._probe_cycle_counter % self._backup_probe_frequency == 0
        ):
            await self._probe_and_update_backup_candidates(leader_id)

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

        Session 17.33 Phase 17.1: Parallelized URL probing using asyncio.gather().
        Instead of sequentially trying each URL and waiting for timeouts,
        we now probe all URLs in parallel and return True as soon as any succeeds.
        This reduces failover detection time from 60s → 30-40s.

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

            # Phase 17.1: Probe all URLs in parallel
            return await self._probe_urls_parallel(leader_id, urls)

        except Exception as e:
            logger.debug(f"[LeaderProbe] Probe error: {e}")
            return False

    async def _probe_urls_parallel(self, node_id: str, urls: list[str]) -> bool:
        """Probe multiple URLs in parallel and return on first success.

        Session 17.33 Phase 17.1: Uses asyncio.gather() with return_exceptions=True
        to probe all URLs simultaneously. Returns True immediately when any URL
        responds with 200 OK. This is much faster than sequential probing when
        some URLs are unreachable (avoids waiting for timeouts).

        Args:
            node_id: Node ID being probed (for logging)
            urls: List of health check URLs to probe

        Returns:
            True if any URL responded successfully, False if all failed
        """
        import aiohttp

        async def probe_single_url(url: str) -> tuple[str, bool]:
            """Probe a single URL and return (url, success)."""
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=self._probe_timeout),
                    ) as response:
                        if response.status == 200:
                            return (url, True)
                        return (url, False)
            except asyncio.TimeoutError:
                logger.debug(f"[LeaderProbe] Timeout probing {url}")
                return (url, False)
            except Exception as e:
                logger.debug(f"[LeaderProbe] Error probing {url}: {e}")
                return (url, False)

        # Probe all URLs in parallel
        results = await asyncio.gather(
            *[probe_single_url(url) for url in urls],
            return_exceptions=True,
        )

        # Check results - return True if any succeeded
        for result in results:
            if isinstance(result, Exception):
                continue
            url, success = result
            if success:
                logger.debug(f"[LeaderProbe] Node {node_id} reachable via {url}")
                return True

        # All URLs failed
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

    def _check_latency_trend(self) -> None:
        """Check if probe latency is trending upward - early warning for leader degradation.

        Jan 5, 2026: Detects latency trends before complete failure.
        If recent probes are 2x slower than older probes, emit a warning event.
        This gives operators early warning that the leader may be struggling.
        """
        # Need at least 6 samples (3 old + 3 recent) for meaningful comparison
        if len(self._latency_history) < 6:
            return

        # Split into older and recent halves
        history = list(self._latency_history)
        mid_point = len(history) // 2
        older_samples = history[:mid_point]
        recent_samples = history[mid_point:]

        older_avg = sum(older_samples) / len(older_samples)
        recent_avg = sum(recent_samples) / len(recent_samples)

        # Avoid division by zero and filter out noise
        if older_avg < 0.01:  # < 10ms is healthy, ignore
            if self._latency_warning_emitted:
                self._latency_warning_emitted = False
                self._emit_event("LEADER_LATENCY_RECOVERED", {
                    "older_avg_ms": older_avg * 1000,
                    "recent_avg_ms": recent_avg * 1000,
                })
            return

        # Check if latency has doubled
        if recent_avg > older_avg * 2:
            if not self._latency_warning_emitted:
                logger.warning(
                    f"[LeaderProbe] Leader latency trending up: "
                    f"{older_avg*1000:.0f}ms → {recent_avg*1000:.0f}ms "
                    f"({recent_avg/older_avg:.1f}x increase)"
                )
                self._emit_event("LEADER_LATENCY_WARNING", {
                    "older_avg_ms": older_avg * 1000,
                    "recent_avg_ms": recent_avg * 1000,
                    "ratio": recent_avg / older_avg,
                    "samples": len(history),
                    "leader_id": self._get_leader_id(),
                })
                self._latency_warning_emitted = True
        else:
            # Latency recovered
            if self._latency_warning_emitted:
                logger.info(
                    f"[LeaderProbe] Leader latency recovered: "
                    f"{recent_avg*1000:.0f}ms (was trending up)"
                )
                self._emit_event("LEADER_LATENCY_RECOVERED", {
                    "older_avg_ms": older_avg * 1000,
                    "recent_avg_ms": recent_avg * 1000,
                })
                self._latency_warning_emitted = False

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

    # =========================================================================
    # Backup Candidate Probing (Session 17.39 - Phase 10.3)
    # =========================================================================

    def _get_backup_candidates(self, current_leader: str, limit: int = 3) -> list[str]:
        """Get top backup candidates (voters excluding current leader).

        Session 17.39 Phase 10.3: Returns voters who could become the next leader.
        These are probed periodically to maintain warm connections, enabling
        faster failover when the current leader becomes unreachable.

        Args:
            current_leader: Current leader to exclude from candidates
            limit: Maximum number of candidates to return (default: 3)

        Returns:
            List of voter node IDs that could become the next leader
        """
        try:
            # Get voter list from orchestrator
            voter_ids = getattr(self._orchestrator, "voter_node_ids", []) or []
            if not voter_ids:
                return []

            # Filter out current leader
            candidates = [v for v in voter_ids if v != current_leader]

            # Also filter out this node (we can't vote for ourselves effectively)
            node_id = getattr(self._orchestrator, "node_id", None)
            if node_id:
                candidates = [v for v in candidates if v != node_id]

            # Return top N candidates (first N alphabetically or by configured order)
            return candidates[:limit]

        except Exception as e:
            logger.debug(f"[LeaderProbe] Failed to get backup candidates: {e}")
            return []

    async def _probe_and_update_backup_candidates(self, current_leader: str) -> None:
        """Probe backup candidates and update their reachability status.

        Session 17.39 Phase 10.3: Proactively probes backup leader candidates
        to maintain warm TCP connections. When the leader fails, we already know
        which candidates are reachable, enabling faster failover.

        Args:
            current_leader: Current leader ID (to exclude from probing)
        """
        candidates = self._get_backup_candidates(current_leader)
        if not candidates:
            return

        logger.debug(
            f"[LeaderProbe] Probing {len(candidates)} backup candidates: {candidates}"
        )

        # Probe each candidate and update status
        warm_count = 0
        for candidate_id in candidates:
            try:
                reachable = await self._probe_node_health(candidate_id)
                old_status = self._backup_candidate_status.get(candidate_id)
                self._backup_candidate_status[candidate_id] = reachable

                if reachable:
                    warm_count += 1
                    if old_status is False:
                        logger.info(
                            f"[LeaderProbe] Backup candidate {candidate_id} is now reachable"
                        )
                else:
                    if old_status is True:
                        logger.warning(
                            f"[LeaderProbe] Backup candidate {candidate_id} became unreachable"
                        )
            except Exception as e:
                logger.debug(f"[LeaderProbe] Error probing candidate {candidate_id}: {e}")
                self._backup_candidate_status[candidate_id] = False

        # Emit event for observability
        self._emit_event("BACKUP_CANDIDATES_PROBED", {
            "candidates": candidates,
            "warm_count": warm_count,
            "total_count": len(candidates),
            "status": {k: v for k, v in self._backup_candidate_status.items()},
            "current_leader": current_leader,
        })

    async def _probe_node_health(self, node_id: str) -> bool:
        """Probe a node's health endpoint.

        Session 17.39: Reuses the parallel URL probing logic from leader probing.

        Args:
            node_id: Node ID to probe

        Returns:
            True if node responded successfully, False otherwise
        """
        try:
            urls_for_peer = getattr(self._orchestrator, "_urls_for_peer", None)
            if urls_for_peer is None:
                return False

            urls = urls_for_peer(node_id, "health")
            if not urls:
                return False

            # Reuse the parallel probing logic
            return await self._probe_urls_parallel(node_id, urls)

        except Exception as e:
            logger.debug(f"[LeaderProbe] Error probing node {node_id}: {e}")
            return False

    def get_warm_backup_candidates(self) -> list[str]:
        """Get list of backup candidates that are currently reachable.

        Session 17.39: Returns candidates with warm connections. These are
        preferred for failover when the leader fails.

        Returns:
            List of reachable backup candidate node IDs
        """
        return [
            node_id
            for node_id, is_reachable in self._backup_candidate_status.items()
            if is_reachable
        ]

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

            # Jan 5, 2026: Clear stale leader references from gossip before election
            # This helps prevent split-brain where old leader info persists
            gossip_mixin = getattr(self._orchestrator, "_gossip_mixin", None)
            if gossip_mixin and hasattr(gossip_mixin, "clear_stale_leader_from_gossip"):
                current_epoch = getattr(self._orchestrator, "_election_epoch", 1) + 1
                cleared = gossip_mixin.clear_stale_leader_from_gossip(
                    unreachable_leader, current_epoch
                )
                if cleared > 0:
                    logger.info(f"[LeaderProbe] Cleared {cleared} stale leader gossip refs")

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
        # Calculate latency stats
        latency_stats = {}
        if self._latency_history:
            latencies = list(self._latency_history)
            latency_stats = {
                "avg_latency_ms": sum(latencies) / len(latencies) * 1000,
                "recent_latency_ms": latencies[-1] * 1000 if latencies else 0,
                "latency_samples": len(latencies),
                "latency_warning_active": self._latency_warning_emitted,
            }

        return {
            "name": self.name,
            "running": self.running,
            "enabled": self.enabled,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self._failure_threshold,
            "last_success_time": self._last_success_time,
            "seconds_since_success": time.time() - self._last_success_time,
            "election_triggered_recently": self._election_triggered_recently,
            **latency_stats,
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
