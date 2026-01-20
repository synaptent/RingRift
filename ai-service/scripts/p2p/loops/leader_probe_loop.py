"""Fast Leader Health Probe Loop.

Jan 4, 2026 - Phase 5 of P2P Cluster Resilience.

Problem: When the leader becomes unreachable, the cluster relies on gossip timeouts
and periodic election checks to detect the failure. This can take 60-180 seconds,
during which training and selfplay dispatch is blocked.

Solution: LeaderProbeLoop actively probes the leader every 10 seconds. After 8
consecutive failures (80s total), it triggers a forced election to recover from
unreachable leader conditions faster. The increased timeout (from 60s) accommodates
Lambda GH200 and Vast.ai nodes which can have 2-5s baseline latency.

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
import random
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
        failure_threshold: int = 8,  # Jan 20, 2026: Increased from 6 (80s total vs 60s)
        probe_timeout: float = 10.0,  # Jan 20, 2026: Increased from 5s to match probe interval
        probe_backup_candidates: bool = True,
        startup_grace_period: float = 20.0,
    ) -> None:
        """Initialize the leader probe loop.

        Args:
            orchestrator: P2POrchestrator instance
            probe_interval: Seconds between probes (default: 10s)
            failure_threshold: Consecutive failures before triggering election (default: 8 = 80s)
            probe_timeout: Timeout for each probe request (default: 10s, matches probe interval)
            probe_backup_candidates: Whether to probe backup candidates in parallel (default: True)
                Session 17.33 Phase 17: Enables faster failover by pre-probing backup candidates
            startup_grace_period: Seconds to wait after startup before triggering elections (default: 20s)
                Jan 7, 2026: Reduced from 45s to 20s for faster leader detection.
                Still allows cluster to converge after mass restarts.
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

        # Jan 7, 2026: Adaptive cooldown based on startup phase
        # During startup: 15-20s cooldown to allow cluster to converge quickly
        # During normal operation: 120s cooldown to prevent election storms
        self._startup_grace_period = startup_grace_period
        self._startup_time = time.time()
        self._startup_cooldown = 15.0  # Short cooldown during startup
        self._normal_cooldown = 120.0  # Normal cooldown after startup
        self._cooldown_jitter = 0.2  # ±20% jitter to stagger elections

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

        # Session 17.43: Split-brain detection (Phase 2)
        # Detects when claimed leader doesn't acknowledge its own leadership
        self._split_brain_detected = False
        self._split_brain_check_interval = 3  # Check every 3rd cycle (~30s at 10s interval)

        # Jan 13, 2026: Dynamic failure threshold scaling (P2.1)
        # Scale failure threshold based on quorum health and latency
        # Range: 4-12 failures (45-120s at 10s probe interval)
        self._min_failure_threshold = 4  # 45s - healthy cluster, fast failover
        self._max_failure_threshold = 12  # 120s - degraded cluster, avoid false elections
        self._dynamic_threshold_enabled = True

    def _is_in_startup_phase(self) -> bool:
        """Check if we're still in the startup grace period.

        Jan 7, 2026: During startup, the cluster is converging and leader probes
        may fail due to timing rather than actual leader failures. This grace
        period allows nodes to come online and establish connectivity.

        Returns:
            True if within startup grace period, False otherwise
        """
        elapsed = time.time() - self._startup_time
        return elapsed < self._startup_grace_period

    def _get_effective_cooldown(self) -> float:
        """Get the current election cooldown based on cluster state.

        Jan 7, 2026: Returns a shorter cooldown during startup to allow faster
        cluster convergence, and a longer cooldown during normal operation to
        prevent election storms. Adds jitter to stagger election attempts.

        Returns:
            Cooldown duration in seconds with ±20% jitter
        """
        if self._is_in_startup_phase():
            base_cooldown = self._startup_cooldown
        else:
            base_cooldown = self._normal_cooldown

        # Add random jitter (±20%) to stagger elections from different voters
        jitter = base_cooldown * self._cooldown_jitter * (2 * random.random() - 1)
        return base_cooldown + jitter

    def _compute_dynamic_failure_threshold(self) -> int:
        """Compute failure threshold based on cluster health metrics.

        Jan 13, 2026 - P2.1 Dynamic Quorum Timeout Scaling.

        Problem: Fixed 60s timeout doesn't account for cluster load. During high
        latency or degraded quorum, false leader elections can destabilize the cluster.

        Solution: Scale the failure threshold (and thus timeout) based on:
        1. Quorum health level (HEALTHY → fast, DEGRADED → slow)
        2. Probe latency trending (low latency → fast, high latency → slow)

        Returns:
            Failure threshold in number of consecutive failures (4-12)
            At 10s probe interval: 4=40s, 6=60s, 8=80s, 12=120s
        """
        if not self._dynamic_threshold_enabled:
            return self._failure_threshold

        # Start with base threshold
        threshold = self._failure_threshold

        # Factor 1: Quorum health level
        # HEALTHY: Use faster timeout (reduce threshold)
        # DEGRADED/MINIMUM: Use slower timeout (increase threshold)
        try:
            check_quorum = getattr(self._orchestrator, "_check_quorum_health", None)
            if check_quorum:
                from scripts.p2p.leader_election import QuorumHealthLevel
                quorum_health = check_quorum()

                if quorum_health == QuorumHealthLevel.HEALTHY:
                    # Cluster is healthy - can afford faster failover
                    threshold = max(threshold - 2, self._min_failure_threshold)
                elif quorum_health == QuorumHealthLevel.DEGRADED:
                    # Cluster is degraded - be more cautious
                    threshold = min(threshold + 2, self._max_failure_threshold)
                elif quorum_health == QuorumHealthLevel.MINIMUM:
                    # Cluster is at minimum - be very cautious
                    threshold = min(threshold + 4, self._max_failure_threshold)
                # LOST is handled separately - elections are blocked entirely
        except Exception as e:
            logger.debug(f"[LeaderProbe] Failed to check quorum health: {e}")

        # Factor 2: Latency trending
        # High latency suggests network issues - use slower timeout
        if len(self._latency_history) >= 3:
            recent_latencies = list(self._latency_history)[-3:]
            avg_latency = sum(recent_latencies) / len(recent_latencies)

            # Jan 20, 2026: Relaxed latency thresholds for Lambda/Vast.ai nodes
            # which can have 2-5s baseline latency due to NAT relay
            if avg_latency > 5.0:
                # Very high latency (>5s) - network is struggling
                threshold = min(threshold + 3, self._max_failure_threshold)
            elif avg_latency > 3.0:
                # High latency (>3s) - be more cautious
                threshold = min(threshold + 1, self._max_failure_threshold)
            elif avg_latency < 0.1:
                # Very low latency (<100ms) - network is healthy
                threshold = max(threshold - 1, self._min_failure_threshold)

        # Clamp to valid range
        threshold = max(self._min_failure_threshold, min(threshold, self._max_failure_threshold))

        return threshold

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

            # Session 17.43: Split-brain detection - check if leader claims leadership
            # Every Nth cycle, verify the leader actually thinks it's the leader
            if self._probe_cycle_counter % self._split_brain_check_interval == 0:
                await self._check_for_split_brain(leader_id)
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
        except (AttributeError, TypeError):
            # Orchestrator unavailable or wrong type
            return False

    def _get_leader_id(self) -> str | None:
        """Get the current leader ID."""
        try:
            return getattr(self._orchestrator, "leader_id", None)
        except (AttributeError, TypeError):
            # Orchestrator unavailable or wrong type
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

        # Jan 7, 2026: Skip election during startup grace period
        # During startup, probe failures are often transient as the cluster converges.
        # We still count failures but don't trigger elections until startup completes.
        if self._is_in_startup_phase():
            elapsed = time.time() - self._startup_time
            remaining = self._startup_grace_period - elapsed
            logger.debug(
                f"[LeaderProbe] Startup grace period active ({remaining:.0f}s remaining). "
                f"Leader {leader_id} unreachable ({self._consecutive_failures}/{self._failure_threshold}) - "
                f"deferring election"
            )
            return

        # Jan 13, 2026: Use dynamic threshold based on cluster health
        dynamic_threshold = self._compute_dynamic_failure_threshold()

        logger.warning(
            f"[LeaderProbe] Leader {leader_id} unreachable "
            f"({self._consecutive_failures}/{dynamic_threshold})"
        )

        self._emit_event("LEADER_PROBE_FAILED", {
            "leader_id": leader_id,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": dynamic_threshold,
            "base_threshold": self._failure_threshold,
            "dynamic_threshold_enabled": self._dynamic_threshold_enabled,
        })

        # Check if we should trigger election using dynamic threshold
        if self._consecutive_failures >= dynamic_threshold:
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

    # =========================================================================
    # Split-Brain Detection (Session 17.43 - Phase 2)
    # =========================================================================

    async def _check_for_split_brain(self, leader_id: str) -> None:
        """Check if the claimed leader actually thinks it's the leader.

        Session 17.43 Phase 2: Detects split-brain scenarios where this node
        thinks a peer is the leader, but that peer doesn't claim leadership.
        This can happen after network partitions where different nodes have
        divergent views of the cluster state.

        If split-brain is detected:
        1. Emit SPLIT_BRAIN_DETECTED event for observability
        2. Clear our leader reference to trigger re-election
        3. Force election to establish consensus on leadership

        Args:
            leader_id: The node ID we believe is the leader
        """
        try:
            # Get leader's status to check if they claim leadership
            leader_status = await self._get_node_status(leader_id)

            if leader_status is None:
                # Couldn't reach status endpoint - handled by regular probe failure
                return

            # Check if the claimed leader actually thinks it's the leader
            claims_leadership = leader_status.get("is_leader", False)

            if not claims_leadership:
                # Split-brain detected! This node thinks leader_id is leader,
                # but leader_id doesn't claim leadership
                await self._on_split_brain_detected(leader_id, leader_status)

            elif self._split_brain_detected:
                # Split-brain was previously detected but now resolved
                logger.info(
                    f"[LeaderProbe] Split-brain resolved: {leader_id} now claims leadership"
                )
                self._split_brain_detected = False
                self._emit_event("SPLIT_BRAIN_RESOLVED", {
                    "leader_id": leader_id,
                    "resolution": "leader_confirmed",
                })

        except Exception as e:
            logger.debug(f"[LeaderProbe] Split-brain check error: {e}")

    async def _get_node_status(self, node_id: str) -> dict[str, Any] | None:
        """Get a node's full status to check its leadership claim.

        Args:
            node_id: Node ID to query

        Returns:
            Status dict with is_leader, leader_id, etc., or None if unreachable
        """
        import aiohttp

        try:
            urls_for_peer = getattr(self._orchestrator, "_urls_for_peer", None)
            if urls_for_peer is None:
                return None

            # Use /status endpoint which includes is_leader field
            urls = urls_for_peer(node_id, "status")
            if not urls:
                return None

            # Try each URL until one succeeds
            for url in urls:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            url,
                            timeout=aiohttp.ClientTimeout(total=self._probe_timeout),
                        ) as response:
                            if response.status == 200:
                                return await response.json()
                except asyncio.TimeoutError:
                    continue
                except (aiohttp.ClientError, OSError, ValueError):
                    # Connection error, network error, or invalid response
                    continue

            return None

        except (aiohttp.ClientError, OSError, ValueError, RuntimeError) as e:
            logger.debug(f"[LeaderProbe] Failed to get status from {node_id}: {e}")
            return None

    async def _on_split_brain_detected(
        self, claimed_leader: str, leader_status: dict[str, Any]
    ) -> None:
        """Handle split-brain detection - leader doesn't claim leadership.

        Session 17.43 Phase 2: When we detect that our claimed leader doesn't
        think it's the leader, we have a split-brain situation. This typically
        happens after network partitions where gossip state has diverged.

        Recovery strategy:
        1. Log the split-brain for observability
        2. Emit SPLIT_BRAIN_DETECTED event
        3. Clear local leader reference
        4. Trigger election to establish consensus

        Args:
            claimed_leader: Node ID we thought was leader
            leader_status: Status response from the claimed leader
        """
        # Only trigger once per split-brain detection
        if self._split_brain_detected:
            return

        self._split_brain_detected = True
        actual_leader = leader_status.get("leader_id")

        logger.warning(
            f"[LeaderProbe] SPLIT-BRAIN DETECTED: "
            f"We think {claimed_leader} is leader, but it claims {actual_leader} is leader. "
            f"Triggering election to resolve."
        )

        # Emit event for observability and alerting
        self._emit_event("SPLIT_BRAIN_DETECTED", {
            "our_leader": claimed_leader,
            "their_leader": actual_leader,
            "claimed_leader_is_leader": leader_status.get("is_leader", False),
            "claimed_leader_node_id": leader_status.get("node_id"),
            "source": "leader_probe_loop",
        })

        # Clear local leader reference to force re-election
        try:
            clear_leader = getattr(self._orchestrator, "_clear_leader", None)
            if clear_leader:
                clear_leader()
                logger.info("[LeaderProbe] Cleared local leader reference")
        except Exception as e:
            logger.debug(f"[LeaderProbe] Failed to clear leader: {e}")

        # Trigger election to establish consensus
        # Use a slight delay to allow other nodes to also detect and participate
        await asyncio.sleep(0.5)
        await self._trigger_election(claimed_leader)

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
        is_startup = self._is_in_startup_phase()
        effective_cooldown = self._get_effective_cooldown()

        logger.error(
            f"[LeaderProbe] Leader {unreachable_leader} unreachable for {downtime:.0f}s, "
            f"triggering election (cooldown={effective_cooldown:.0f}s, startup={is_startup})"
        )

        self._election_triggered_recently = True

        # Schedule cooldown reset with adaptive duration
        async def reset_cooldown():
            await asyncio.sleep(effective_cooldown)
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

        # Phase 1.2 (Jan 7, 2026): Verify elected leader has consensus
        # Schedule consensus verification after election settles
        asyncio.create_task(self._verify_elected_leader_after_delay(3.0))

    async def _verify_elected_leader_after_delay(self, delay: float) -> None:
        """Wait for election to settle, then verify consensus.

        Jan 7, 2026 - Phase 1.2 P2P Stability: After triggering an election,
        wait for it to settle and then verify the new leader has quorum support.

        Args:
            delay: Seconds to wait before verification (allows election to complete)
        """
        await asyncio.sleep(delay)
        new_leader = self._get_leader_id()
        if not new_leader:
            logger.warning("[LeaderProbe] No leader after election - cannot verify consensus")
            return

        has_consensus = await self._verify_elected_leader_consensus()
        if has_consensus:
            logger.info(f"[LeaderProbe] Leader {new_leader} verified with voter consensus")
        else:
            logger.warning(
                f"[LeaderProbe] Leader {new_leader} does NOT have voter consensus - "
                f"may indicate ongoing split-brain"
            )
            self._emit_event("LEADER_CONSENSUS_FAILED", {
                "leader_id": new_leader,
                "reason": "insufficient_voter_agreement",
            })

    async def _verify_elected_leader_consensus(self) -> bool:
        """Verify 3+ voters agree on the current leader.

        Jan 7, 2026 - Phase 1.2 P2P Stability: After split-brain detection triggers
        election, verify that the new leader has quorum support. This prevents
        accepting a leader that only a minority of voters recognize.

        Returns:
            True if 3+ voters agree on current leader, False otherwise
        """
        current_leader = self._get_leader_id()
        if not current_leader:
            return False

        voter_ids = self._get_voter_ids()
        if not voter_ids:
            # No voter list available - assume consensus
            logger.debug("[LeaderProbe] No voter list available for consensus check")
            return True

        # Probe up to 5 voters to check their view of the leader
        agreed_voters = 0
        probed_voters = 0

        for voter_id in voter_ids[:5]:
            # Skip ourselves
            node_id = getattr(self._orchestrator, "node_id", None)
            if voter_id == node_id:
                # We agree with ourselves by definition
                agreed_voters += 1
                probed_voters += 1
                continue

            voter_leader = await self._probe_voter_for_leader(voter_id)
            if voter_leader is None:
                # Couldn't reach voter - don't count
                continue

            probed_voters += 1
            if voter_leader == current_leader:
                agreed_voters += 1

        logger.info(
            f"[LeaderProbe] Consensus check: {agreed_voters}/{probed_voters} voters "
            f"agree on leader {current_leader}"
        )

        # Require 3+ voters to agree
        return agreed_voters >= 3

    def _get_voter_ids(self) -> list[str]:
        """Get list of voter node IDs from orchestrator.

        Returns:
            List of voter node IDs
        """
        try:
            return getattr(self._orchestrator, "voter_node_ids", []) or []
        except (AttributeError, TypeError):
            # Orchestrator unavailable or wrong type
            return []

    async def _probe_voter_for_leader(self, voter_id: str) -> str | None:
        """Probe a voter to get their view of who the leader is.

        Args:
            voter_id: Voter node ID to probe

        Returns:
            Leader ID that the voter believes is leader, or None if unreachable
        """
        status = await self._get_node_status(voter_id)
        if status is None:
            return None
        return status.get("leader_id")

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

        # Jan 13, 2026: Include dynamic threshold info
        dynamic_threshold = self._compute_dynamic_failure_threshold()
        timeout_seconds = dynamic_threshold * self.interval

        return {
            "name": self.name,
            "running": self.running,
            "enabled": self.enabled,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self._failure_threshold,
            "dynamic_threshold": dynamic_threshold,
            "dynamic_timeout_seconds": timeout_seconds,
            "dynamic_threshold_enabled": self._dynamic_threshold_enabled,
            "last_success_time": self._last_success_time,
            "seconds_since_success": time.time() - self._last_success_time,
            "election_triggered_recently": self._election_triggered_recently,
            "split_brain_detected": self._split_brain_detected,
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

        # Jan 7, 2026: Collect startup phase and cooldown info for all responses
        is_startup = self._is_in_startup_phase()
        effective_cooldown = self._get_effective_cooldown()
        startup_info = {
            "is_startup_phase": is_startup,
            "effective_cooldown": round(effective_cooldown, 1),
            "cooldown_type": "startup" if is_startup else "normal",
        }
        if is_startup:
            startup_info["startup_remaining_seconds"] = round(
                self._startup_grace_period - (time.time() - self._startup_time), 1
            )

        # Jan 13, 2026: Use dynamic threshold for health check
        dynamic_threshold = self._compute_dynamic_failure_threshold()

        # Check if leader is unreachable (approaching election trigger)
        if self._consecutive_failures >= dynamic_threshold:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Leader unreachable ({self._consecutive_failures} failures), election triggered",
                details={
                    "consecutive_failures": self._consecutive_failures,
                    "failure_threshold": dynamic_threshold,
                    "base_threshold": self._failure_threshold,
                    "seconds_since_success": time.time() - self._last_success_time,
                    "election_triggered_recently": self._election_triggered_recently,
                    **startup_info,
                },
            )

        # Session 17.43: Check for split-brain condition
        if self._split_brain_detected:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message="Split-brain detected: claimed leader denies leadership",
                details={
                    "split_brain_detected": True,
                    "leader_id": self._get_leader_id(),
                    "seconds_since_success": time.time() - self._last_success_time,
                    **startup_info,
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
                    **startup_info,
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
                "split_brain_detected": False,
                **startup_info,
            },
        )
