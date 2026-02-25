"""Recovery Manager for P2P Orchestrator.

January 2026: Phase 12 of Aggressive Decomposition.
Consolidates NAT recovery and node recovery into a single manager.

This module provides:
- NAT type detection and relay management
- NAT-blocked peer recovery probing
- Node health detection and automatic recovery
- Connectivity scoring for leader eligibility

Extracted from p2p_orchestrator.py to reduce complexity and improve testability.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import ClientTimeout

from app.core.async_context import safe_create_task

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator
    from scripts.p2p.types import NodeInfo


logger = logging.getLogger(__name__)

# Recovery constants
NAT_BLOCKED_RECOVERY_TIMEOUT = 300  # 5 minutes before probing NAT-blocked peers
NAT_BLOCKED_PROBE_TIMEOUT = 5.0  # Timeout for NAT probe requests
NAT_BLOCKED_PROBE_INTERVAL = 60  # Minimum interval between probes
NAT_RELAY_PREFERENCE_THRESHOLD = 3  # Consecutive failures before marking for relay
MAX_CONSECUTIVE_FAILURES = 5  # Maximum failures before node considered unhealthy


# ============================================================================
# Singleton Pattern
# ============================================================================

_recovery_manager: RecoveryManager | None = None


def get_recovery_manager() -> RecoveryManager | None:
    """Get the global RecoveryManager singleton."""
    return _recovery_manager


def set_recovery_manager(manager: RecoveryManager) -> None:
    """Set the global RecoveryManager singleton."""
    global _recovery_manager
    _recovery_manager = manager


def reset_recovery_manager() -> None:
    """Reset the global RecoveryManager singleton (for testing)."""
    global _recovery_manager
    _recovery_manager = None


def create_recovery_manager(
    config: RecoveryConfig | None = None,
    orchestrator: P2POrchestrator | None = None,
) -> RecoveryManager:
    """Factory function to create and register a RecoveryManager.

    Args:
        config: Optional configuration. Uses defaults if not provided.
        orchestrator: The P2P orchestrator instance.

    Returns:
        The created RecoveryManager instance.
    """
    manager = RecoveryManager(
        config=config or RecoveryConfig(),
        orchestrator=orchestrator,
    )
    set_recovery_manager(manager)
    return manager


# ============================================================================
# Configuration and Statistics
# ============================================================================


@dataclass
class RecoveryConfig:
    """Configuration for RecoveryManager."""

    # NAT recovery settings
    nat_probe_timeout: float = NAT_BLOCKED_PROBE_TIMEOUT
    nat_probe_interval: float = NAT_BLOCKED_PROBE_INTERVAL
    nat_recovery_timeout: float = NAT_BLOCKED_RECOVERY_TIMEOUT
    relay_preference_threshold: int = NAT_RELAY_PREFERENCE_THRESHOLD

    # Node recovery settings
    node_recovery_cooldown: float = 600.0  # 10 minutes between recovery attempts
    node_recovery_check_interval: float = 120.0  # Check every 2 minutes
    max_recoveries_per_cycle: int = 2  # Max nodes to recover per cycle
    max_concurrent_recoveries: int = 2  # Max in-flight recovery tasks
    stale_gossip_threshold: float = 300.0  # 5 minutes
    high_failure_threshold: int = 5  # Consecutive failures
    disk_full_threshold: float = 95.0  # Disk usage percentage

    # SSH recovery settings
    ssh_timeout: float = 20.0  # Feb 24: Reduced from 45s — no reason to wait longer for dead nodes
    ssh_connect_timeout: float = 10.0
    service_name: str = "ringrift-p2p"

    # Exponential backoff for failed recoveries
    max_recovery_cooldown: float = 3600.0  # Cap at 60 minutes


@dataclass
class RecoveryStats:
    """Statistics for recovery operations."""

    # NAT recovery stats
    nat_probes_performed: int = 0
    peers_recovered_from_nat: int = 0
    relay_assignments_updated: int = 0

    # Node recovery stats
    node_recovery_attempts: int = 0
    node_recovery_successes: int = 0
    node_recovery_failures: int = 0


@dataclass
class NodeRecoveryMetrics:
    """Metrics for node recovery tracking."""

    total_attempts: int = 0
    successes: int = 0
    failures: int = 0
    nodes_in_cooldown: int = 0


# ============================================================================
# RecoveryManager
# ============================================================================


class RecoveryManager:
    """Manages NAT recovery and node recovery for P2P cluster.

    This manager consolidates recovery-related functionality:
    - NAT type detection (symmetric vs cone)
    - NAT-blocked peer probing and recovery
    - Relay node selection and assignment
    - Node health detection and automatic restart
    - Connectivity scoring for leader eligibility

    Attributes:
        config: Recovery configuration parameters.
        stats: Recovery statistics.
    """

    def __init__(
        self,
        config: RecoveryConfig,
        orchestrator: P2POrchestrator | None = None,
    ):
        """Initialize RecoveryManager.

        Args:
            config: Recovery configuration.
            orchestrator: The P2P orchestrator instance.
        """
        self.config = config
        self._orchestrator = orchestrator
        self._stats = RecoveryStats()

        # NAT state
        self._nat_type: str = "unknown"

        # Node recovery state
        self._node_recovery_attempts: dict[str, float] = {}  # node_id -> last_attempt_time
        self._node_recovery_metrics: dict[str, int] = {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
        }
        self._last_node_recovery_check: float = 0
        self._active_recoveries: set[str] = set()  # node_ids with in-flight recovery tasks
        self._recovery_failure_counts: dict[str, int] = {}  # node_id -> consecutive failure count

    @property
    def nat_type(self) -> str:
        """Get the detected NAT type."""
        return self._nat_type

    # ========================================================================
    # NAT Detection
    # ========================================================================

    async def detect_nat_type(self) -> str:
        """Detect NAT type using STUN-like probing.

        NAT Types:
        - Full Cone: Any external host can send packets to internal host
        - Restricted Cone: Only hosts that internal has contacted can respond
        - Port Restricted: Only hosts+ports that internal has contacted can respond
        - Symmetric: Different external IP:port for each destination (breaks P2P)

        Returns:
            Detected NAT type string.
        """
        if not self._orchestrator:
            return "unknown"

        external_ips = set()

        # Probe multiple peers to detect if we get different external IPs
        alive_peers = [
            p for p in self._orchestrator._peer_snapshot.get_snapshot().values()
            if p.is_alive() and p.node_id != self._orchestrator.node_id
        ]

        for peer in alive_peers[:5]:  # Probe up to 5 peers
            try:
                peer_scheme = getattr(peer, "scheme", "http") or "http"
                async with aiohttp.ClientSession() as session, session.get(
                    f"{peer_scheme}://{peer.host}:{peer.port}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                    headers=self._orchestrator._auth_headers()
                ) as resp:
                    if resp.status == 200:
                        await resp.json()
                        external_ips.add(peer.host)
            except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                continue

        # If we see ourselves with different IPs from different vantage points,
        # we likely have symmetric NAT
        if len(external_ips) > 1:
            self._nat_type = "symmetric"
            logger.info("Detected symmetric NAT (multiple external IPs seen)")
        elif len(external_ips) == 1:
            self._nat_type = "cone"
        else:
            self._nat_type = "unknown"

        return self._nat_type

    # ========================================================================
    # NAT-Blocked Peer Recovery
    # ========================================================================

    async def probe_nat_blocked_peers(self) -> int:
        """Probe NAT-blocked peers to see if they've become reachable.

        Returns:
            Number of peers that recovered.
        """
        if not self._orchestrator:
            return 0

        nat_blocked_peers = [
            p for p in self._orchestrator._peer_snapshot.get_snapshot().values()
            if p.nat_blocked and p.node_id != self._orchestrator.node_id
        ]

        recovered = 0
        for peer in nat_blocked_peers:
            # Check if enough time has passed since blocking
            blocked_duration = time.time() - (peer.nat_blocked_since or 0)
            if blocked_duration < self.config.nat_recovery_timeout:
                continue

            # Try to reach the peer
            peer_scheme = getattr(peer, "scheme", "http") or "http"

            # Try multiple endpoints
            endpoints_to_try = [(peer.host, peer.port)]

            # Add Tailscale IP
            ts_ip = self._orchestrator._get_tailscale_ip_for_peer(peer.node_id)
            if ts_ip and ts_ip != peer.host:
                endpoints_to_try.insert(0, (ts_ip, peer.port))

            # Add reported endpoint
            rh = str(getattr(peer, "reported_host", "") or "").strip()
            rp = int(getattr(peer, "reported_port", 0) or 0)
            if rh and rp:
                endpoints_to_try.append((rh, rp))

            for host, port in endpoints_to_try:
                try:
                    async with aiohttp.ClientSession() as session, session.get(
                        f"{peer_scheme}://{host}:{port}/health",
                        timeout=aiohttp.ClientTimeout(total=self.config.nat_probe_timeout),
                        headers=self._orchestrator._auth_headers()
                    ) as resp:
                        if resp.status == 200:
                            # Peer is reachable! Clear NAT-blocked status
                            logger.info(f"NAT-blocked peer {peer.node_id} is now reachable at {host}:{port}")
                            with self._orchestrator.peers_lock:
                                if peer.node_id in self._orchestrator.peers:
                                    self._orchestrator.peers[peer.node_id].nat_blocked = False
                                    self._orchestrator.peers[peer.node_id].nat_blocked_since = 0.0
                                    self._orchestrator.peers[peer.node_id].host = host
                                    self._orchestrator.peers[peer.node_id].consecutive_failures = 0
                            recovered += 1
                            self._stats.peers_recovered_from_nat += 1
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError):
                    continue

        self._stats.nat_probes_performed += len(nat_blocked_peers)
        return recovered

    async def update_relay_preferences(self) -> int:
        """Update relay preferences based on connectivity patterns.

        Returns:
            Number of peers marked for relay.
        """
        if not self._orchestrator:
            return 0

        # Identify peers that consistently fail direct connections
        peers_needing_relay = [
            p for p in self._orchestrator._peer_snapshot.get_snapshot().values()
            if (getattr(p, "consecutive_failures", 0) or 0) >= self.config.relay_preference_threshold
            and not p.nat_blocked
            and p.node_id != self._orchestrator.node_id
        ]

        marked = 0
        for peer in peers_needing_relay:
            if not peer.nat_blocked:
                logger.info(f"Peer {peer.node_id} has {peer.consecutive_failures} consecutive failures, marking as NAT-blocked")
                with self._orchestrator.peers_lock:
                    if peer.node_id in self._orchestrator.peers:
                        self._orchestrator.peers[peer.node_id].nat_blocked = True
                        self._orchestrator.peers[peer.node_id].nat_blocked_since = time.time()
                        # Set relay to best available relay node
                        relay_node = self.select_best_relay(for_peer=peer.node_id)
                        if relay_node:
                            self._orchestrator.peers[peer.node_id].relay_via = relay_node
                        marked += 1

        self._stats.relay_assignments_updated += marked
        return marked

    async def validate_relay_assignments(self) -> int:
        """Validate and update relay assignments for NAT-blocked peers.

        Returns:
            Number of relay reassignments.
        """
        if not self._orchestrator:
            return 0

        peers_snapshot = self._orchestrator._peer_snapshot.get_snapshot()
        nat_blocked_peers = [
            p for p in peers_snapshot.values()
            if getattr(p, "nat_blocked", False)
            and getattr(p, "relay_via", "")
            and p.node_id != self._orchestrator.node_id
        ]

        reassigned = 0
        for peer in nat_blocked_peers:
            relay_id = str(getattr(peer, "relay_via", "") or "")
            if not relay_id:
                continue

            relay_peer = peers_snapshot.get(relay_id)

            # Check if relay is healthy
            relay_healthy = (
                relay_peer is not None
                and relay_peer.is_alive()
                and not getattr(relay_peer, "nat_blocked", False)
                and (getattr(relay_peer, "consecutive_failures", 0) or 0) < 2
            )

            if not relay_healthy:
                # Find a new relay
                new_relay = self.select_best_relay(for_peer=peer.node_id)
                if new_relay and new_relay != relay_id:
                    logger.info(
                        f"[RelayHealthCheck] Relay {relay_id} unhealthy for {peer.node_id}, "
                        f"switching to {new_relay}"
                    )
                    with self._orchestrator.peers_lock:
                        if peer.node_id in self._orchestrator.peers:
                            self._orchestrator.peers[peer.node_id].relay_via = new_relay
                    reassigned += 1

        return reassigned

    def select_best_relay(self, for_peer: str = "") -> str:
        """Select the best relay node based on connectivity and load.

        Args:
            for_peer: Optional peer node_id to look up configured relay preferences.

        Returns:
            Node ID of the best relay, or empty string if none available.
        """
        if not self._orchestrator:
            return ""

        # Check configured relay preferences for this peer
        if for_peer:
            configured_relays = self._get_configured_relays(for_peer)
            for relay_id in configured_relays:
                with self._orchestrator.peers_lock:
                    relay_peer = self._orchestrator.peers.get(relay_id)
                if relay_peer and self._is_valid_relay(relay_peer):
                    return relay_id

        # Fall back to automatic selection
        with self._orchestrator.peers_lock:
            candidates = [
                p for p in self._orchestrator.peers.values()
                if self._is_valid_relay(p)
            ]

        if not candidates:
            return ""

        # Prefer leader, then voters, then lowest load
        leader_peer = next((p for p in candidates if p.node_id == self._orchestrator.leader_id), None)
        if leader_peer:
            return leader_peer.node_id

        voter_peer = next((p for p in candidates if p.node_id in self._orchestrator.voter_node_ids), None)
        if voter_peer:
            return voter_peer.node_id

        # Lowest load
        candidates.sort(key=lambda p: getattr(p, "load_score", 100))
        return candidates[0].node_id if candidates else ""

    def _is_valid_relay(self, peer: NodeInfo) -> bool:
        """Check if a peer is a valid relay candidate."""
        if not self._orchestrator:
            return False
        return (
            peer.is_alive()
            and not getattr(peer, "nat_blocked", False)
            and peer.node_id != self._orchestrator.node_id
            and (getattr(peer, "consecutive_failures", 0) or 0) < 2
        )

    def _get_configured_relays(self, peer_id: str) -> list[str]:
        """Get configured relay nodes for a peer from distributed_hosts.yaml.

        Returns:
            List of relay node IDs in priority order.
        """
        try:
            from app.config.cluster_config import load_cluster_config

            config = load_cluster_config()
            hosts_raw = getattr(config, "hosts_raw", {}) or {}
            host_config = hosts_raw.get(peer_id, {})

            relays = []
            for key in ["relay_primary", "relay_secondary", "relay_tertiary"]:
                relay_id = host_config.get(key, "")
                if relay_id:
                    relays.append(relay_id)
            return relays
        except (ImportError, AttributeError, KeyError, TypeError):
            return []

    async def probe_nat_blocked_peer(self, peer: NodeInfo) -> bool:
        """Probe a NAT-blocked peer to check if it's now directly reachable.

        Args:
            peer: The peer to probe.

        Returns:
            True if peer is reachable and NAT-blocked status was cleared.
        """
        if not self._orchestrator:
            return False

        if not peer.nat_blocked:
            return False

        now = time.time()
        nat_blocked_since = float(getattr(peer, "nat_blocked_since", 0.0) or 0.0)
        last_probe = float(getattr(peer, "last_nat_probe", 0.0) or 0.0)

        # Don't probe too frequently
        if now - last_probe < self.config.nat_probe_interval:
            return False

        # Don't probe if not blocked long enough
        if nat_blocked_since > 0 and (now - nat_blocked_since) < self.config.nat_recovery_timeout:
            return False

        # Update last probe time
        with self._orchestrator.peers_lock:
            existing = self._orchestrator.peers.get(peer.node_id)
            if existing:
                existing.last_nat_probe = now

        try:
            # Import session helper
            try:
                from scripts.p2p.utils.http_utils import get_client_session
            except ImportError:
                get_client_session = lambda t: aiohttp.ClientSession(timeout=t)

            url = self._orchestrator._url_for_peer(peer, "/status")
            timeout = ClientTimeout(total=self.config.nat_probe_timeout)
            async with get_client_session(timeout) as session:
                async with session.get(url, headers=self._orchestrator._auth_headers()) as resp:
                    if resp.status == 200:
                        # Peer is reachable! Clear NAT-blocked status
                        with self._orchestrator.peers_lock:
                            existing = self._orchestrator.peers.get(peer.node_id)
                            if existing and existing.nat_blocked:
                                existing.nat_blocked = False
                                existing.nat_blocked_since = 0.0
                                existing.relay_via = ""
                                logger.info(f"NAT recovery: {peer.node_id} is now directly reachable")
                                return True
        except (aiohttp.ClientError, asyncio.TimeoutError, AttributeError) as e:
            logger.debug(f"NAT recovery probe failed for {peer.node_id}: {type(e).__name__}")

        return False

    async def sweep_nat_recovery(self) -> int:
        """Periodically probe NAT-blocked peers to check if they've become reachable.

        Returns:
            Number of peers that recovered from NAT-blocked state.
        """
        if not self._orchestrator:
            return 0

        recovered = 0
        nat_blocked_peers = [
            p for p in self._orchestrator._peer_snapshot.get_snapshot().values()
            if p.nat_blocked and p.is_alive()
        ]

        if not nat_blocked_peers:
            return 0

        # Probe in parallel but limit concurrency
        tasks = [self.probe_nat_blocked_peer(p) for p in nat_blocked_peers[:10]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                peer_id = nat_blocked_peers[i].node_id if i < len(nat_blocked_peers) else "unknown"
                logger.debug(f"NAT probe failed for {peer_id}: {result}")
            elif result is True:
                recovered += 1

        if recovered > 0:
            logger.info(f"NAT recovery sweep: {recovered} peer(s) recovered")

        return recovered

    # ========================================================================
    # Connectivity Scoring
    # ========================================================================

    def compute_connectivity_score(self, peer: NodeInfo) -> float:
        """Compute connectivity score for leader eligibility ranking.

        Score components (0.0 to 1.0):
        - 0.4: Direct reachability (not NAT-blocked, not force_relay)
        - 0.3: Transport success rate (based on consecutive_failures)
        - 0.2: Connected peers (based on visible_peers if available)
        - 0.1: Role weight (leaders get slight boost)

        Args:
            peer: The peer to score.

        Returns:
            Connectivity score between 0.0 and 1.0.
        """
        if not self._orchestrator:
            return 0.0

        # Import NodeRole
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            NodeRole = None

        score = 0.0

        # Direct reachability: 0.4 points
        nat_blocked = getattr(peer, "nat_blocked", False)
        force_relay = getattr(peer, "force_relay_mode", False)
        if not nat_blocked and not force_relay:
            score += 0.4
        elif not nat_blocked:
            score += 0.2

        # Transport success rate: 0.3 points
        failures = int(getattr(peer, "consecutive_failures", 0) or 0)
        if failures == 0:
            score += 0.3
        elif failures < 3:
            score += 0.2
        elif failures < MAX_CONSECUTIVE_FAILURES:
            score += 0.1

        # Connected peers: 0.2 points
        visible_peers = int(getattr(peer, "visible_peers", 0) or 0)
        if visible_peers >= 10:
            score += 0.2
        elif visible_peers >= 5:
            score += 0.15
        elif visible_peers >= 2:
            score += 0.1

        # Role weight: 0.1 points
        if NodeRole and hasattr(peer, "role"):
            if peer.role == NodeRole.LEADER:
                score += 0.1
            elif peer.role == NodeRole.FOLLOWER:
                score += 0.05

        return min(1.0, score)

    # ========================================================================
    # Node Recovery
    # ========================================================================

    async def check_node_recovery(self) -> None:
        """Detect and recover stuck nodes via gossip.

        AUTOMATIC NODE RECOVERY: Uses gossip to detect nodes that are:
        - Unresponsive (stale gossip timestamp)
        - Stuck (high failure count, no job progress)
        - Resource-exhausted (high disk/memory)

        Recovery actions:
        - SSH to node and restart the ringrift-p2p service
        - Only leader attempts recovery to avoid duplicate restarts
        - Rate limit recovery attempts
        """
        if not self._orchestrator:
            return

        # Import NodeRole
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            return

        # Only leader performs recovery to avoid duplicate restarts
        if self._orchestrator.role != NodeRole.LEADER:
            return

        now = time.time()

        # Rate limit check interval
        if now - self._last_node_recovery_check < self.config.node_recovery_check_interval:
            return
        self._last_node_recovery_check = now

        # Check each peer for health issues
        gossip_states = getattr(self._orchestrator, "_gossip_peer_states", {})
        nodes_to_recover = []

        # Feb 23, 2026: Use non-blocking cached snapshot to avoid blocking event
        # loop on peers_lock contention (was causing 62s check_node_recovery)
        _snapshot_fn = getattr(self._orchestrator, "_get_peers_snapshot_nonblocking", None)
        if _snapshot_fn is not None:
            _peers_snapshot = _snapshot_fn()
        else:
            with self._orchestrator.peers_lock:
                _peers_snapshot = list(self._orchestrator.peers.values())

        for peer in _peers_snapshot:
            node_id = peer.node_id
            if node_id == self._orchestrator.node_id:
                continue

            # Skip recently recovered nodes
            last_attempt = self._node_recovery_attempts.get(node_id, 0)
            if now - last_attempt < self.config.node_recovery_cooldown:
                continue

            # Check for unhealthy indicators
            needs_recovery = False
            reason = ""

            # 1. Peer not alive
            if not peer.is_alive():
                needs_recovery = True
                reason = "not responding to heartbeat"

            # 2. Stale gossip state
            elif node_id in gossip_states:
                state = gossip_states[node_id]
                state_age = now - state.get("timestamp", 0)
                if state_age > self.config.stale_gossip_threshold:
                    needs_recovery = True
                    reason = f"stale gossip ({int(state_age)}s old)"

            # 3. High consecutive failures
            elif getattr(peer, "consecutive_failures", 0) >= self.config.high_failure_threshold:
                needs_recovery = True
                reason = f"high failure count ({peer.consecutive_failures})"

            # 4. Disk nearly full
            elif getattr(peer, "disk_percent", 0) > self.config.disk_full_threshold:
                needs_recovery = True
                reason = f"disk full ({peer.disk_percent}%)"

            if needs_recovery:
                nodes_to_recover.append((node_id, peer, reason))

        # Feb 24, 2026: Fire-and-forget recovery — detection loop completes immediately,
        # slow SSH recovery runs in background without blocking LeaderOps.
        for node_id, peer, reason in nodes_to_recover[:self.config.max_recoveries_per_cycle]:
            # Skip nodes already being recovered
            if node_id in self._active_recoveries:
                logger.debug(f"NODE RECOVERY: Skipping {node_id}, recovery already in flight")
                continue

            # Cap concurrent recoveries to prevent thread pool exhaustion
            if len(self._active_recoveries) >= self.config.max_concurrent_recoveries:
                logger.debug(
                    f"NODE RECOVERY: Skipping {node_id}, "
                    f"concurrent recovery cap ({self.config.max_concurrent_recoveries}) reached"
                )
                break

            # Apply exponential backoff: after failed recovery, double cooldown
            failure_count = self._recovery_failure_counts.get(node_id, 0)
            if failure_count > 0:
                backoff_cooldown = min(
                    self.config.node_recovery_cooldown * (2 ** failure_count),
                    self.config.max_recovery_cooldown,
                )
                last_attempt = self._node_recovery_attempts.get(node_id, 0)
                if now - last_attempt < backoff_cooldown:
                    logger.debug(
                        f"NODE RECOVERY: Skipping {node_id}, in backoff cooldown "
                        f"({backoff_cooldown:.0f}s after {failure_count} failures)"
                    )
                    continue

            logger.info(f"NODE RECOVERY: Launching background recovery for {node_id} ({reason})")
            self._node_recovery_attempts[node_id] = now
            self._node_recovery_metrics["attempts"] += 1
            self._stats.node_recovery_attempts += 1

            safe_create_task(
                self._background_recovery(node_id, peer, reason),
                name=f"recovery-{node_id}",
            )

    async def _background_recovery(self, node_id: str, peer: Any, reason: str) -> None:
        """Run node recovery in the background without blocking the detection loop.

        Feb 24, 2026: Extracted from check_node_recovery() to make detection fast (<1s).
        SSH recovery (slow) runs as a background task, tracked via _active_recoveries.

        Args:
            node_id: The node identifier.
            peer: The peer object with host information.
            reason: Why recovery was triggered.
        """
        self._active_recoveries.add(node_id)
        try:
            # Notify on recovery attempt
            safe_create_task(self._orchestrator.notifier.send(
                title="Node Recovery Initiated",
                message=f"Attempting to recover node {node_id}: {reason}",
                level="warning",
                fields={
                    "Node": node_id,
                    "Reason": reason,
                    "Host": getattr(peer, "host", "unknown"),
                },
                node_id=self._orchestrator.node_id,
            ), name=f"recovery-notify-attempt-{node_id}")

            success = await self.attempt_node_recovery(node_id, peer)
            if success:
                self._node_recovery_metrics["successes"] += 1
                self._stats.node_recovery_successes += 1
                self._recovery_failure_counts.pop(node_id, None)  # Reset on success
                logger.info(f"NODE RECOVERY: Successfully restarted {node_id}")
                safe_create_task(self._orchestrator.notifier.send(
                    title="Node Recovery Success",
                    message=f"Successfully recovered node {node_id}",
                    level="info",
                    fields={"Node": node_id, "Reason": reason},
                    node_id=self._orchestrator.node_id,
                ), name=f"recovery-notify-success-{node_id}")
            else:
                self._node_recovery_metrics["failures"] += 1
                self._stats.node_recovery_failures += 1
                self._recovery_failure_counts[node_id] = (
                    self._recovery_failure_counts.get(node_id, 0) + 1
                )
                logger.info(
                    f"NODE RECOVERY: Failed to restart {node_id} "
                    f"(failures={self._recovery_failure_counts[node_id]})"
                )
                safe_create_task(self._orchestrator.notifier.send(
                    title="Node Recovery Failed",
                    message=f"Failed to recover node {node_id} ({reason})",
                    level="error",
                    fields={
                        "Node": node_id,
                        "Reason": reason,
                        "Action": "Manual intervention may be required",
                    },
                    node_id=self._orchestrator.node_id,
                ), name=f"recovery-notify-failure-{node_id}")
        except Exception as e:  # noqa: BLE001
            self._node_recovery_metrics["failures"] += 1
            self._stats.node_recovery_failures += 1
            self._recovery_failure_counts[node_id] = (
                self._recovery_failure_counts.get(node_id, 0) + 1
            )
            logger.warning(f"NODE RECOVERY: Background recovery error for {node_id}: {e}")
        finally:
            self._active_recoveries.discard(node_id)

    async def attempt_node_recovery(self, node_id: str, peer: Any) -> bool:
        """Attempt to recover a node by restarting its service via SSH.

        Args:
            node_id: The node identifier.
            peer: The peer object with host information.

        Returns:
            True if recovery command succeeded, False otherwise.
        """
        host = getattr(peer, "host", None)
        if not host:
            return False

        try:
            # Try to restart the service via SSH
            cmd = (
                f"timeout {int(self.config.ssh_timeout - 5)} "
                f"ssh -o ConnectTimeout={int(self.config.ssh_connect_timeout)} "
                f"-o StrictHostKeyChecking=no {host} "
                f"'sudo systemctl restart {self.config.service_name}'"
            )
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.ssh_timeout)

            if proc.returncode == 0:
                return True
            else:
                logger.info(f"NODE RECOVERY: SSH failed for {node_id}: {stderr.decode()[:100]}")
                return False

        except asyncio.TimeoutError:
            logger.info(f"NODE RECOVERY: SSH timeout for {node_id}")
            return False
        except Exception as e:  # noqa: BLE001
            logger.info(f"NODE RECOVERY: Error recovering {node_id}: {e}")
            return False

    def get_node_recovery_metrics(self) -> NodeRecoveryMetrics:
        """Get node recovery metrics for /status endpoint.

        Returns:
            NodeRecoveryMetrics with recovery statistics.
        """
        now = time.time()
        in_cooldown = sum(
            1 for t in self._node_recovery_attempts.values()
            if now - t < self.config.node_recovery_cooldown
        )

        return NodeRecoveryMetrics(
            total_attempts=self._node_recovery_metrics.get("attempts", 0),
            successes=self._node_recovery_metrics.get("successes", 0),
            failures=self._node_recovery_metrics.get("failures", 0),
            nodes_in_cooldown=in_cooldown,
        )

    # ========================================================================
    # Health Check
    # ========================================================================

    def health_check(self) -> dict[str, Any]:
        """Get health status for DaemonManager integration.

        Returns:
            Dict with health status and metrics.
        """
        metrics = self.get_node_recovery_metrics()
        return {
            "healthy": True,
            "nat_type": self._nat_type,
            "nat_probes_performed": self._stats.nat_probes_performed,
            "peers_recovered_from_nat": self._stats.peers_recovered_from_nat,
            "relay_assignments_updated": self._stats.relay_assignments_updated,
            "node_recovery_attempts": metrics.total_attempts,
            "node_recovery_successes": metrics.successes,
            "node_recovery_failures": metrics.failures,
            "nodes_in_cooldown": metrics.nodes_in_cooldown,
        }
