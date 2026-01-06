"""
Partition Healer for P2P Cluster.

January 2026: Automated network partition detection and healing without full restarts.

Philosophy:
- Union discovery: If ANY source finds a peer, include it
- Gossip everything: Discovered peers are gossiped to the mesh
- Heal don't restart: Inject peers into partitions, don't restart processes
- Convergence: Eventually all nodes should see all other nodes

Prometheus Metrics (January 2026):
- ringrift_partition_healing_duration_seconds: Time spent healing partitions
- ringrift_partitions_detected_total: Number of network partitions detected
- ringrift_partitions_healed_total: Number of partitions successfully healed
- ringrift_nodes_reconnected_total: Number of nodes reconnected during healing
- ringrift_partition_healing_success_total: Successful healing passes
- ringrift_partition_healing_failures_total: Failed healing passes
- ringrift_partition_healing_escalation_level: Current escalation level (0-5)

Convergence Metrics (Sprint 15.1.5, January 3, 2026):
- ringrift_partition_convergence_duration_seconds: Time to achieve convergence
- ringrift_partition_convergence_attempts_total: Total convergence check attempts
- ringrift_partition_convergence_failures_total: Convergence failures (timeout/disagreement)
- ringrift_partition_convergence_leader_disagreements_total: Leader disagreements during check

Usage:
    # Run as a background daemon
    python scripts/p2p/partition_healer.py

    # Single healing pass
    python scripts/p2p/partition_healer.py --once

    # Check for partitions only (no action)
    python scripts/p2p/partition_healer.py --check-only

    # Triggered automatically from P2P orchestrator on NETWORK_ISOLATION_DETECTED
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config.coordination_defaults import PartitionHealingDefaults
from scripts.p2p.union_discovery import UnionDiscovery, DiscoveredPeer

# Jan 2026: Centralized timeouts for partition healing
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
    _PARTITION_DISCOVERY_TIMEOUT = LoopTimeouts.PARTITION_DISCOVERY
except ImportError:
    _PARTITION_DISCOVERY_TIMEOUT = 30.0  # Fallback

# Jan 2026 Session 8: Circuit breaker check before injection
try:
    from scripts.p2p.network import check_peer_transport_circuit
    HAS_TRANSPORT_CB = True
except ImportError:
    HAS_TRANSPORT_CB = False
    def check_peer_transport_circuit(peer_host: str, transport: str = "http") -> bool:
        return True  # Fallback: assume transport is available

logger = logging.getLogger(__name__)

# =============================================================================
# Prometheus Metrics (January 2026)
# =============================================================================

try:
    from prometheus_client import REGISTRY, Counter, Gauge, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    REGISTRY = None


def _get_or_create_histogram(name: str, description: str, labels: list, buckets: tuple | None = None):
    """Get existing histogram or create new one, handling re-registration."""
    if not HAS_PROMETHEUS:
        return None
    try:
        if buckets:
            return Histogram(name, description, labels, buckets=buckets)
        return Histogram(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


def _get_or_create_counter(name: str, description: str, labels: list):
    """Get existing counter or create new one, handling re-registration."""
    if not HAS_PROMETHEUS:
        return None
    try:
        return Counter(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


def _get_or_create_gauge(name: str, description: str, labels: list):
    """Get existing gauge or create new one, handling re-registration."""
    if not HAS_PROMETHEUS:
        return None
    try:
        return Gauge(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)


# Define metrics
PROM_HEALING_DURATION = None
PROM_PARTITIONS_DETECTED = None
PROM_PARTITIONS_HEALED = None
PROM_NODES_RECONNECTED = None
PROM_HEALING_SUCCESS = None
PROM_HEALING_FAILURES = None
PROM_ESCALATION_LEVEL = None
# Jan 3, 2026 Sprint 15.1.5: Convergence-specific metrics
PROM_CONVERGENCE_DURATION = None
PROM_CONVERGENCE_ATTEMPTS = None
PROM_CONVERGENCE_FAILURES = None
PROM_CONVERGENCE_LEADER_DISAGREEMENTS = None

if HAS_PROMETHEUS:
    PROM_HEALING_DURATION = _get_or_create_histogram(
        'ringrift_partition_healing_duration_seconds',
        'Time spent healing partitions',
        [],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
    )
    PROM_PARTITIONS_DETECTED = _get_or_create_counter(
        'ringrift_partitions_detected_total',
        'Number of network partitions detected',
        []
    )
    PROM_PARTITIONS_HEALED = _get_or_create_counter(
        'ringrift_partitions_healed_total',
        'Number of partitions successfully healed',
        []
    )
    PROM_NODES_RECONNECTED = _get_or_create_counter(
        'ringrift_nodes_reconnected_total',
        'Number of nodes reconnected during healing',
        []
    )
    PROM_HEALING_SUCCESS = _get_or_create_counter(
        'ringrift_partition_healing_success_total',
        'Successful partition healing passes',
        []
    )
    PROM_HEALING_FAILURES = _get_or_create_counter(
        'ringrift_partition_healing_failures_total',
        'Failed partition healing passes',
        []
    )
    PROM_ESCALATION_LEVEL = _get_or_create_gauge(
        'ringrift_partition_healing_escalation_level',
        'Current escalation level for partition healing (0-5)',
        []
    )
    # Jan 3, 2026 Sprint 15.1.5: Convergence-specific metrics
    PROM_CONVERGENCE_DURATION = _get_or_create_histogram(
        'ringrift_partition_convergence_duration_seconds',
        'Time to achieve convergence after partition healing',
        [],
        buckets=(1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
    )
    PROM_CONVERGENCE_ATTEMPTS = _get_or_create_counter(
        'ringrift_partition_convergence_attempts_total',
        'Total convergence check attempts',
        []
    )
    PROM_CONVERGENCE_FAILURES = _get_or_create_counter(
        'ringrift_partition_convergence_failures_total',
        'Convergence check failures (timeout or disagreement)',
        []
    )
    PROM_CONVERGENCE_LEADER_DISAGREEMENTS = _get_or_create_counter(
        'ringrift_partition_convergence_leader_disagreements_total',
        'Times nodes disagreed on cluster leader during convergence',
        []
    )

# Singleton instance for access from P2P orchestrator
_partition_healer_instance: PartitionHealer | None = None
_partition_healer_lock = threading.Lock()


@dataclass
class PartitionInfo:
    """Information about a detected partition."""

    partition_id: str  # Identifier for this partition
    leader: str | None  # Leader of this partition
    members: set[str] = field(default_factory=set)  # Node IDs in this partition
    addresses: dict[str, str] = field(default_factory=dict)  # node_id -> address

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class HealingResult:
    """Result of a healing operation."""

    success: bool
    partitions_found: int
    partitions_healed: int
    nodes_reconnected: int
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def record_metrics(self) -> None:
        """Record Prometheus metrics for this healing result.

        January 2026: Added for P2P observability.
        """
        # Duration in seconds
        if PROM_HEALING_DURATION:
            PROM_HEALING_DURATION.observe(self.duration_ms / 1000.0)

        # Partitions detected
        if PROM_PARTITIONS_DETECTED and self.partitions_found > 0:
            PROM_PARTITIONS_DETECTED.inc(self.partitions_found)

        # Partitions healed
        if PROM_PARTITIONS_HEALED and self.partitions_healed > 0:
            PROM_PARTITIONS_HEALED.inc(self.partitions_healed)

        # Nodes reconnected
        if PROM_NODES_RECONNECTED and self.nodes_reconnected > 0:
            PROM_NODES_RECONNECTED.inc(self.nodes_reconnected)

        # Success/failure counters
        if self.success:
            if PROM_HEALING_SUCCESS:
                PROM_HEALING_SUCCESS.inc()
        else:
            if PROM_HEALING_FAILURES:
                PROM_HEALING_FAILURES.inc()


class PartitionHealer:
    """
    Detects and heals network partitions in the P2P cluster.

    Strategy:
    1. Use UnionDiscovery to find ALL known peers from all sources
    2. Query each peer's /status to map which peers they see
    3. Identify partitions (groups of nodes that don't see each other)
    4. Inject bridge peers into each partition via /admin/add_peer endpoint
    5. Wait for gossip to propagate and verify healing

    January 2026: Added auto-triggering from P2P orchestrator with rate limiting.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        p2p_port: int = 8770,
        timeout: float | None = None,
        orchestrator: Any = None,
    ):
        self._config_path = config_path or (
            Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml"
        )
        self._p2p_port = p2p_port
        self._timeout = timeout if timeout is not None else PartitionHealingDefaults.PEER_TIMEOUT
        self._union_discovery = UnionDiscovery()
        self._session: aiohttp.ClientSession | None = None

        # January 2026: Rate limiting for auto-triggered healing
        self._last_healing_time: float = 0.0
        self._healing_lock = threading.Lock()
        self._pending_trigger: bool = False

        # Sprint 10 (Jan 3, 2026): Recovery escalation on consecutive convergence failures
        self._consecutive_failures: int = 0
        self._escalation_level: int = 0

        # Session 17.41 (Jan 6, 2026): Orchestrator reference for forced elections
        self._orchestrator = orchestrator
        self._last_election_trigger: float = 0.0
        self._election_cooldown: float = 120.0  # Seconds between forced elections

    def set_orchestrator(self, orchestrator: Any) -> None:
        """Set the orchestrator reference for forced elections.

        Session 17.41 (Jan 6, 2026): Allows wiring the orchestrator after initialization,
        useful when PartitionHealer is created before orchestrator is fully configured.

        Args:
            orchestrator: P2P orchestrator instance with _start_election() method.
        """
        self._orchestrator = orchestrator
        logger.info(
            f"[PartitionHealer] Orchestrator reference set "
            f"(has_start_election={hasattr(orchestrator, '_start_election')})"
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def discover_all_peers(self) -> dict[str, DiscoveredPeer]:
        """Use union discovery to find all known peers."""
        return await self._union_discovery.discover_all_peers(timeout=_PARTITION_DISCOVERY_TIMEOUT)

    async def get_peer_view(self, address: str) -> dict[str, Any] | None:
        """Get a peer's view of the cluster from its /status endpoint."""
        url = f"http://{address}:{self._p2p_port}/status"
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.debug(f"Could not reach {address}: {e}")
        return None

    async def detect_partitions(
        self,
        known_peers: dict[str, DiscoveredPeer],
    ) -> list[PartitionInfo]:
        """
        Detect partitions by querying each peer's view.

        Returns list of PartitionInfo, one per partition.
        """
        # Query all peers in parallel
        peer_views: dict[str, set[str]] = {}

        async def query_peer(node_id: str, peer: DiscoveredPeer) -> tuple[str, set[str] | None]:
            """Query a peer's view of the cluster."""
            addr = peer.best_address
            if not addr:
                return node_id, None

            view = await self.get_peer_view(addr)
            if view:
                # Get the set of peers this node sees
                peers = view.get("peers", {})
                seen = set(peers.keys())
                seen.add(view.get("node_id", node_id))  # Include self
                return node_id, seen
            return node_id, None

        tasks = [query_peer(node_id, peer) for node_id, peer in known_peers.items()]
        results = await asyncio.gather(*tasks)

        for node_id, seen in results:
            if seen is not None:
                peer_views[node_id] = seen

        if not peer_views:
            logger.warning("Could not reach any peers")
            return []

        # Build partition graph using union-find
        # Two nodes are in the same partition if they can see each other (directly or indirectly)
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union nodes that can see each other
        for node_id, seen in peer_views.items():
            for peer_id in seen:
                if peer_id in peer_views:  # Only union with peers we can query
                    union(node_id, peer_id)

        # Group nodes by partition root
        partitions_map: dict[str, set[str]] = {}
        for node_id in peer_views:
            root = find(node_id)
            if root not in partitions_map:
                partitions_map[root] = set()
            partitions_map[root].add(node_id)

        # Build PartitionInfo objects
        partitions: list[PartitionInfo] = []
        for i, (root, members) in enumerate(sorted(partitions_map.items(), key=lambda x: -len(x[1]))):
            # Get leader from any member's view
            leader = None
            for member in members:
                view_peer = known_peers.get(member)
                if view_peer and view_peer.best_address:
                    view = await self.get_peer_view(view_peer.best_address)
                    if view:
                        leader = view.get("leader_id")
                        break

            # Build address map
            addresses = {}
            for member in members:
                peer = known_peers.get(member)
                if peer and peer.best_address:
                    addresses[member] = peer.best_address

            partitions.append(PartitionInfo(
                partition_id=f"partition_{i}",
                leader=leader,
                members=members,
                addresses=addresses,
            ))

        return partitions

    async def inject_peer(
        self,
        target_address: str,
        peer_to_inject: str,
        peer_address: str,
    ) -> bool:
        """
        Inject a peer into a node's peer list via /admin/add_peer endpoint.

        This allows the target to discover the injected peer without restart.

        Jan 2026 Session 8: Check circuit breaker state before attempting injection.
        If HTTP transport to target is circuit-broken, skip injection to avoid
        wasted attempts and faster failover to alternate bridge targets.
        """
        # Jan 2026 Session 8: Check if HTTP transport to target is circuit-broken
        if HAS_TRANSPORT_CB and not check_peer_transport_circuit(target_address, "http"):
            logger.debug(
                f"HTTP transport to {target_address} is circuit-broken, skipping injection"
            )
            return False

        url = f"http://{target_address}:{self._p2p_port}/admin/add_peer"
        payload = {
            "node_id": peer_to_inject,
            "address": peer_address,
            "port": self._p2p_port,
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    logger.info(f"Injected {peer_to_inject} into {target_address}")
                    return True
                else:
                    logger.warning(f"Failed to inject peer: HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"Error injecting peer: {e}")

        return False

    async def heal_partitions(
        self,
        partitions: list[PartitionInfo],
        known_peers: dict[str, DiscoveredPeer],
    ) -> HealingResult:
        """
        Heal partitions by injecting bridge peers between them.

        Strategy: For each pair of partitions, inject peers from one into the other.
        This creates bridges that allow gossip to flow between partitions.
        """
        start_time = time.time()
        result = HealingResult(
            success=True,
            partitions_found=len(partitions),
            partitions_healed=0,
            nodes_reconnected=0,
        )

        if len(partitions) <= 1:
            logger.info("Only one partition found, no healing needed")
            result.duration_ms = (time.time() - start_time) * 1000
            return result

        # For each partition pair, inject some peers from one into the other
        for i, p1 in enumerate(partitions):
            for j, p2 in enumerate(partitions[i + 1:], start=i + 1):
                # Get bridge candidates from each partition
                p1_bridges = list(p1.addresses.items())[:3]  # Up to 3 bridges
                p2_bridges = list(p2.addresses.items())[:3]

                injections_succeeded = 0

                # Inject p2 peers into p1 nodes
                for p1_node, p1_addr in p1_bridges:
                    for p2_node, p2_addr in p2_bridges:
                        if await self.inject_peer(p1_addr, p2_node, p2_addr):
                            injections_succeeded += 1
                            result.nodes_reconnected += 1

                # Inject p1 peers into p2 nodes
                for p2_node, p2_addr in p2_bridges:
                    for p1_node, p1_addr in p1_bridges:
                        if await self.inject_peer(p2_addr, p1_node, p1_addr):
                            injections_succeeded += 1
                            result.nodes_reconnected += 1

                if injections_succeeded > 0:
                    result.partitions_healed += 1
                    logger.info(
                        f"Created {injections_succeeded} bridges between "
                        f"{p1.partition_id} and {p2.partition_id}"
                    )

                    # Sprint 16.1 (Jan 3, 2026): Validate gossip propagation after injection
                    # Collect target nodes and injected peers for validation
                    target_nodes = [
                        (node_id, addr) for node_id, addr in p1_bridges
                    ] + [
                        (node_id, addr) for node_id, addr in p2_bridges
                    ]
                    injected_peers = [p2_node for p2_node, _ in p2_bridges] + [
                        p1_node for p1_node, _ in p1_bridges
                    ]

                    propagated, visibility = await self._validate_injection_propagated(
                        target_nodes, injected_peers
                    )

                    if not propagated:
                        logger.warning(
                            f"Injection between {p1.partition_id} and {p2.partition_id} "
                            f"may not have propagated fully (visibility={visibility*100:.1f}%)"
                        )
                        # Continue with other partition pairs, don't fail early

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    async def _validate_injection_propagated(
        self,
        target_nodes: list[tuple[str, str]],  # [(node_id, address), ...]
        injected_peers: list[str],  # peer_ids that were injected
    ) -> tuple[bool, float]:
        """
        Validate that injected peers are visible on target nodes.

        Sprint 16.1 (Jan 3, 2026): Quick gossip propagation check after T0 injection.
        Instead of waiting for full cluster convergence, this validates that the
        specific injected peers are now visible on the target nodes.

        Args:
            target_nodes: List of (node_id, address) tuples for nodes that received injections
            injected_peers: List of peer_ids that were injected

        Returns:
            Tuple of (success, visibility_ratio) where visibility_ratio is the
            fraction of (target, peer) pairs that are visible.
        """
        if not target_nodes or not injected_peers:
            return True, 1.0  # Nothing to validate

        # Wait one gossip round for propagation
        gossip_interval = PartitionHealingDefaults.CONVERGENCE_CHECK_INTERVAL
        logger.debug(
            f"[InjectionValidation] Waiting {gossip_interval}s for gossip propagation..."
        )
        await asyncio.sleep(gossip_interval)

        # Check visibility on each target
        visible_count = 0
        total_checks = 0
        timeout = PartitionHealingDefaults.INJECTION_PROPAGATION_TIMEOUT

        async def check_peer_visibility(
            node_id: str, addr: str, peer_ids: list[str]
        ) -> int:
            """Check how many injected peers are visible on a target node."""
            try:
                view = await self.get_peer_view(addr)
                if view and "peers" in view:
                    visible_peers = set(view["peers"].keys())
                    visible_peers.add(view.get("node_id", ""))  # Include self
                    return sum(1 for p in peer_ids if p in visible_peers)
            except Exception as e:
                logger.debug(f"[InjectionValidation] Error checking {node_id}: {e}")
            return 0

        # Run visibility checks with timeout
        start_time = time.time()
        check_tasks = []
        for node_id, addr in target_nodes:
            check_tasks.append(check_peer_visibility(node_id, addr, injected_peers))

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*check_tasks, return_exceptions=True),
                timeout=timeout,
            )

            for i, result in enumerate(results):
                node_id = target_nodes[i][0]
                expected = len(injected_peers)
                total_checks += expected

                if isinstance(result, Exception):
                    logger.debug(
                        f"[InjectionValidation] {node_id}: check failed ({result})"
                    )
                else:
                    visible_count += result
                    if result < expected:
                        logger.debug(
                            f"[InjectionValidation] {node_id}: {result}/{expected} "
                            f"injected peers visible"
                        )

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(
                f"[InjectionValidation] Timed out after {elapsed:.1f}s checking visibility"
            )

        visibility_ratio = visible_count / total_checks if total_checks > 0 else 0
        success = visibility_ratio >= PartitionHealingDefaults.INJECTION_VISIBILITY_THRESHOLD

        logger.info(
            f"[InjectionValidation] Result: {visible_count}/{total_checks} visible "
            f"({visibility_ratio*100:.1f}%), threshold={PartitionHealingDefaults.INJECTION_VISIBILITY_THRESHOLD*100:.0f}%, "
            f"success={success}"
        )

        return success, visibility_ratio

    async def _validate_convergence(
        self,
        known_peers: dict[str, DiscoveredPeer],
        timeout: float | None = None,
    ) -> tuple[bool, str]:
        """
        Validate that gossip has converged after healing.

        Sprint 4 (Jan 2, 2026): Verify that all nodes agree on peer membership
        after healing, instead of assuming success.

        Sprint 15.1.5 (Jan 3, 2026): Added leader consensus verification and
        convergence metrics for better observability.

        Args:
            known_peers: Known peers from union discovery
            timeout: Timeout for convergence (defaults to CONVERGENCE_TIMEOUT)

        Returns:
            Tuple of (converged, message)
        """
        timeout = timeout or PartitionHealingDefaults.CONVERGENCE_TIMEOUT
        check_interval = PartitionHealingDefaults.CONVERGENCE_CHECK_INTERVAL
        agreement_threshold = PartitionHealingDefaults.CONVERGENCE_AGREEMENT_THRESHOLD

        start_time = time.time()
        check_count = 0
        leader_disagreement_count = 0
        reachable_nodes = [
            (node_id, peer.best_address)
            for node_id, peer in known_peers.items()
            if peer.best_address
        ]

        if not reachable_nodes:
            if PROM_CONVERGENCE_FAILURES:
                PROM_CONVERGENCE_FAILURES.inc()
            return False, "No reachable nodes for convergence check"

        logger.info(
            f"[ConvergenceCheck] Starting convergence validation "
            f"(timeout={timeout}s, threshold={agreement_threshold*100:.0f}%)"
        )

        avg_agreement = 0.0  # Track for timeout message

        while (time.time() - start_time) < timeout:
            check_count += 1
            if PROM_CONVERGENCE_ATTEMPTS:
                PROM_CONVERGENCE_ATTEMPTS.inc()

            # Sprint 15.1.5: Collect both peer views AND leader info from all nodes
            peer_views: dict[str, set[str]] = {}
            leader_views: dict[str, str | None] = {}

            async def get_view_with_leader(
                node_id: str, addr: str
            ) -> tuple[str, set[str] | None, str | None]:
                """Get peer view and leader_id from node's /status endpoint."""
                view = await self.get_peer_view(addr)
                if view:
                    peers = view.get("peers", {})
                    seen = set(peers.keys())
                    seen.add(view.get("node_id", node_id))
                    # Sprint 15.1.5: Extract leader_id for consensus check
                    leader_id = view.get("leader_id") or view.get("effective_leader_id")
                    return node_id, seen, leader_id
                return node_id, None, None

            tasks = [get_view_with_leader(node_id, addr) for node_id, addr in reachable_nodes]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, tuple) and len(result) == 3:
                    node_id, seen, leader_id = result
                    if seen is not None:
                        peer_views[node_id] = seen
                    if leader_id is not None:
                        leader_views[node_id] = leader_id

            if len(peer_views) < 2:
                # Not enough responses for comparison
                logger.debug("[ConvergenceCheck] Not enough responses, waiting...")
                await asyncio.sleep(check_interval)
                continue

            # Calculate agreement: what percentage of known peers do nodes agree on?
            # All nodes should see the same set of peers if converged
            all_seen_peers: set[str] = set()
            for seen in peer_views.values():
                all_seen_peers.update(seen)

            # Count how many nodes see each peer
            peer_counts: dict[str, int] = {}
            for seen in peer_views.values():
                for peer_id in seen:
                    peer_counts[peer_id] = peer_counts.get(peer_id, 0) + 1

            # Agreement = average(how many nodes see each peer / total responding nodes)
            total_responding = len(peer_views)
            agreement_scores = [
                count / total_responding for count in peer_counts.values()
            ]
            avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0

            # Sprint 15.1.5: Check leader consensus
            unique_leaders = set(leader_views.values())
            unique_leaders.discard(None)  # Remove None values
            leader_consensus = len(unique_leaders) <= 1

            if not leader_consensus and len(unique_leaders) > 1:
                leader_disagreement_count += 1
                if PROM_CONVERGENCE_LEADER_DISAGREEMENTS:
                    PROM_CONVERGENCE_LEADER_DISAGREEMENTS.inc()
                logger.debug(
                    f"[ConvergenceCheck] Leader disagreement: {len(unique_leaders)} "
                    f"different leaders reported: {unique_leaders}"
                )

            logger.debug(
                f"[ConvergenceCheck] Progress: {avg_agreement*100:.1f}% peer agreement, "
                f"leader_consensus={leader_consensus} "
                f"({total_responding} nodes responding, {len(all_seen_peers)} peers seen)"
            )

            # Sprint 15.1.5: Require BOTH peer agreement AND leader consensus
            if avg_agreement >= agreement_threshold and leader_consensus:
                elapsed = time.time() - start_time
                # Record convergence duration metric
                if PROM_CONVERGENCE_DURATION:
                    PROM_CONVERGENCE_DURATION.observe(elapsed)

                leader_info = ""
                if unique_leaders:
                    leader_info = f", leader={list(unique_leaders)[0]}"

                msg = (
                    f"Convergence achieved: {avg_agreement*100:.1f}% peer agreement"
                    f"{leader_info} in {elapsed:.1f}s "
                    f"({total_responding} nodes, {len(all_seen_peers)} peers)"
                )
                logger.info(f"[ConvergenceCheck] {msg}")
                return True, msg

            await asyncio.sleep(check_interval)

        # Timeout reached
        elapsed = time.time() - start_time
        if PROM_CONVERGENCE_FAILURES:
            PROM_CONVERGENCE_FAILURES.inc()
        if PROM_CONVERGENCE_DURATION:
            PROM_CONVERGENCE_DURATION.observe(elapsed)

        # Build detailed failure message
        unique_leaders = set(leader_views.values())
        unique_leaders.discard(None)
        leader_consensus = len(unique_leaders) <= 1

        failure_reasons = []
        if avg_agreement < agreement_threshold:
            failure_reasons.append(f"peer agreement {avg_agreement*100:.1f}% < {agreement_threshold*100:.0f}%")
        if not leader_consensus:
            failure_reasons.append(f"leader disagreement ({len(unique_leaders)} leaders)")

        msg = (
            f"Convergence timeout after {elapsed:.1f}s: {', '.join(failure_reasons) or 'unknown'}. "
            f"Leader disagreements during check: {leader_disagreement_count}"
        )
        logger.warning(f"[ConvergenceCheck] {msg}")
        return False, msg

    async def verify_update_convergence(
        self,
        updated_nodes: list[str],
        timeout_seconds: float = 120.0,
        agreement_threshold: float | None = None,
    ) -> bool:
        """Verify updated nodes have rejoined mesh and gossip converged.

        January 3, 2026 - Sprint 16.2: Added for quorum-safe rolling updates.

        This method is called by QuorumSafeUpdateCoordinator after restarting
        nodes to verify they've rejoined the P2P mesh and gossip has converged.

        Args:
            updated_nodes: Node IDs that were just updated/restarted
            timeout_seconds: Maximum time to wait for convergence (default: 120s)
            agreement_threshold: Override for convergence threshold (default: from config)

        Returns:
            True if all nodes visible and gossip converged, False on timeout

        Example:
            >>> healer = get_partition_healer()
            >>> converged = await healer.verify_update_convergence(
            ...     updated_nodes=["lambda-gh200-1", "lambda-gh200-2"],
            ...     timeout_seconds=120.0,
            ... )
            >>> if not converged:
            ...     logger.error("Nodes failed to rejoin mesh after update")
        """
        if not updated_nodes:
            return True  # Nothing to verify

        agreement_threshold = (
            agreement_threshold
            or PartitionHealingDefaults.CONVERGENCE_AGREEMENT_THRESHOLD
        )
        check_interval = PartitionHealingDefaults.CONVERGENCE_CHECK_INTERVAL
        updated_set = set(updated_nodes)

        logger.info(
            f"[UpdateConvergence] Waiting for {len(updated_nodes)} nodes to rejoin mesh "
            f"(timeout={timeout_seconds}s, threshold={agreement_threshold*100:.0f}%)"
        )

        start_time = time.time()
        check_count = 0

        while (time.time() - start_time) < timeout_seconds:
            check_count += 1

            # Discover all currently visible peers
            known_peers = await self.discover_all_peers()

            # Check which updated nodes are now visible
            visible_updated = set()
            for node_id, peer in known_peers.items():
                if node_id in updated_set and peer.best_address:
                    # Try to reach the node
                    view = await self.get_peer_view(peer.best_address)
                    if view is not None:
                        visible_updated.add(node_id)

            missing = updated_set - visible_updated
            visibility_ratio = len(visible_updated) / len(updated_set)

            logger.debug(
                f"[UpdateConvergence] Check {check_count}: "
                f"{len(visible_updated)}/{len(updated_set)} updated nodes visible "
                f"({visibility_ratio*100:.1f}%)"
            )

            if missing:
                logger.debug(f"[UpdateConvergence] Missing nodes: {missing}")

            # All updated nodes visible, now verify full gossip convergence
            if not missing:
                # Use existing convergence validation
                remaining_timeout = timeout_seconds - (time.time() - start_time)
                if remaining_timeout > 10:  # Need at least 10s for convergence check
                    converged, msg = await self._validate_convergence(
                        known_peers,
                        timeout=remaining_timeout,
                    )
                    if converged:
                        elapsed = time.time() - start_time
                        logger.info(
                            f"[UpdateConvergence] All {len(updated_nodes)} nodes rejoined "
                            f"and gossip converged in {elapsed:.1f}s"
                        )
                        return True
                    else:
                        # Convergence failed but nodes are visible - keep trying
                        logger.debug(
                            f"[UpdateConvergence] Nodes visible but gossip not converged: {msg}"
                        )

            await asyncio.sleep(check_interval)

        # Timeout reached
        elapsed = time.time() - start_time
        visible_updated = set()
        try:
            known_peers = await self.discover_all_peers()
            for node_id in updated_set:
                if node_id in known_peers:
                    peer = known_peers[node_id]
                    if peer.best_address:
                        view = await self.get_peer_view(peer.best_address)
                        if view is not None:
                            visible_updated.add(node_id)
        except Exception as e:
            logger.debug(f"[UpdateConvergence] Error in final check: {e}")

        missing = updated_set - visible_updated
        logger.warning(
            f"[UpdateConvergence] Timeout after {elapsed:.1f}s. "
            f"{len(visible_updated)}/{len(updated_set)} nodes visible. "
            f"Missing: {missing or 'none (convergence issue)'}"
        )
        return False

    async def run_healing_pass(self) -> HealingResult:
        """
        Run a single healing pass: discover, detect partitions, heal.
        """
        logger.info("Starting partition healing pass")

        # Step 1: Discover all peers from all sources
        logger.info("Discovering peers from all sources...")
        known_peers = await self.discover_all_peers()
        logger.info(f"Found {len(known_peers)} peers from union discovery")

        if len(known_peers) < 2:
            return HealingResult(
                success=True,
                partitions_found=0,
                partitions_healed=0,
                nodes_reconnected=0,
                errors=["Not enough peers discovered"],
            )

        # Step 2: Detect partitions
        logger.info("Detecting partitions...")
        partitions = await self.detect_partitions(known_peers)
        logger.info(f"Found {len(partitions)} partitions")

        for p in partitions:
            logger.info(f"  {p.partition_id}: {p.size} members, leader={p.leader}")

        # Step 3: Heal partitions
        if len(partitions) > 1:
            logger.info("Healing partitions...")
            result = await self.heal_partitions(partitions, known_peers)

            # Sprint 4 (Jan 2, 2026): Validate convergence after healing
            if result.partitions_healed > 0:
                logger.info("Validating convergence after healing...")
                converged, convergence_msg = await self._validate_convergence(known_peers)

                if converged:
                    # Emit success event and reset escalation state
                    self._emit_healing_event(result)
                    self._reset_escalation()
                else:
                    # Convergence failed - mark result as partial success
                    result.success = False
                    result.errors.append(f"Convergence failed: {convergence_msg}")
                    self._emit_healing_failed_event(convergence_msg)
                    logger.warning(
                        f"Partition healing completed but convergence not achieved: "
                        f"{convergence_msg}"
                    )
                    # Sprint 10 (Jan 3, 2026): Escalate recovery on repeated failures
                    await self._escalate_recovery(convergence_msg)

            return result
        else:
            return HealingResult(
                success=True,
                partitions_found=len(partitions),
                partitions_healed=0,
                nodes_reconnected=0,
            )

    def _emit_healing_event(self, result: HealingResult) -> None:
        """Emit PARTITION_HEALED event after successful healing."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(
                DataEventType.PARTITION_HEALED,
                {
                    "partitions_found": result.partitions_found,
                    "partitions_healed": result.partitions_healed,
                    "nodes_reconnected": result.nodes_reconnected,
                    "duration_ms": result.duration_ms,
                    "timestamp": time.time(),
                },
            )
            logger.info(
                f"Emitted PARTITION_HEALED event: "
                f"{result.partitions_healed} partitions healed"
            )
        except ImportError:
            logger.debug("Event emission not available (missing event_router)")
        except Exception as e:
            logger.warning(f"Failed to emit partition healed event: {e}")

    async def trigger_healing_pass(
        self,
        delay: float | None = None,
        force: bool = False,
    ) -> HealingResult | None:
        """Trigger a healing pass with rate limiting.

        January 2026: Called from P2P orchestrator on NETWORK_ISOLATION_DETECTED.
        Uses rate limiting to prevent healing loops.

        Args:
            delay: Optional delay before starting (defaults to DETECTION_DELAY)
            force: If True, bypass rate limiting

        Returns:
            HealingResult if healing was performed, None if rate-limited
        """
        now = time.time()

        with self._healing_lock:
            # Check rate limit unless forced
            if not force:
                time_since_last = now - self._last_healing_time
                if time_since_last < PartitionHealingDefaults.MIN_INTERVAL:
                    remaining = PartitionHealingDefaults.MIN_INTERVAL - time_since_last
                    logger.info(
                        f"Partition healing rate-limited, {remaining:.0f}s until next allowed"
                    )
                    return None

            # Mark that we're about to heal
            self._last_healing_time = now

        # Emit start event
        self._emit_healing_start_event()

        # Apply delay if specified, with jitter to prevent thundering herd
        # Jan 2026 Session 8: Add random jitter (±50% of base delay) to stagger
        # healing triggers when multiple nodes detect partition simultaneously
        delay = delay if delay is not None else PartitionHealingDefaults.DETECTION_DELAY
        if delay > 0:
            jitter = random.uniform(0, delay * 0.5)  # 0 to 50% of base delay
            total_delay = delay + jitter
            logger.info(
                f"Waiting {total_delay:.1f}s before starting healing pass "
                f"(base {delay:.0f}s + jitter {jitter:.1f}s)..."
            )
            await asyncio.sleep(total_delay)

        # Run the healing pass with overall timeout
        # Jan 3, 2026: Added timeout to prevent healing operations from hanging indefinitely
        try:
            result = await asyncio.wait_for(
                self.run_healing_pass(),
                timeout=PartitionHealingDefaults.TOTAL_TIMEOUT,
            )
            # January 2026: Record Prometheus metrics
            result.record_metrics()
            self._update_escalation_metric()
            return result
        except asyncio.TimeoutError:
            error_msg = (
                f"Healing pass timed out after {PartitionHealingDefaults.TOTAL_TIMEOUT}s"
            )
            logger.error(error_msg)
            self._emit_healing_failed_event(error_msg)
            result = HealingResult(
                success=False,
                partitions_found=0,
                partitions_healed=0,
                nodes_reconnected=0,
                errors=[error_msg],
            )
            result.record_metrics()
            return result
        except Exception as e:
            logger.error(f"Healing pass failed: {e}")
            self._emit_healing_failed_event(str(e))
            result = HealingResult(
                success=False,
                partitions_found=0,
                partitions_healed=0,
                nodes_reconnected=0,
                errors=[str(e)],
            )
            result.record_metrics()
            return result

    def _update_escalation_metric(self) -> None:
        """Update the escalation level Prometheus gauge.

        January 2026: Added for P2P observability.
        """
        if PROM_ESCALATION_LEVEL:
            PROM_ESCALATION_LEVEL.set(self._escalation_level)

    def _emit_healing_start_event(self) -> None:
        """Emit PARTITION_HEALING_STARTED event."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(
                DataEventType.PARTITION_HEALING_STARTED,
                {"timestamp": time.time()},
            )
        except (ImportError, Exception):
            pass  # Best effort

    def _emit_healing_failed_event(self, error: str) -> None:
        """Emit PARTITION_HEALING_FAILED event."""
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(
                DataEventType.PARTITION_HEALING_FAILED,
                {"error": error, "timestamp": time.time()},
            )
        except (ImportError, Exception):
            pass  # Best effort

    async def _check_leader_reachable(self, leader_address: str | None = None) -> bool:
        """Check if the current leader is reachable via HTTP /health endpoint.

        Session 17.41 (Jan 6, 2026): Added for forced election on partition detection.
        If the leader cannot respond within timeout, workers can't claim jobs.

        Args:
            leader_address: Optional leader address. If not provided, discovers from gossip.

        Returns:
            True if leader responds to /health, False otherwise.
        """
        if not leader_address:
            # Try to discover leader from local gossip state
            if self._orchestrator:
                leader_id = getattr(self._orchestrator, "_current_leader_id", None)
                if leader_id:
                    peers = getattr(self._orchestrator, "_gossip_peer_states", {})
                    leader_info = peers.get(leader_id, {})
                    leader_address = leader_info.get("address")

            if not leader_address:
                logger.debug("[ForceElection] No leader address available to check")
                return False

        url = f"http://{leader_address}:{self._p2p_port}/health"
        try:
            session = await self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                if resp.status == 200:
                    logger.debug(f"[ForceElection] Leader at {leader_address} is reachable")
                    return True
        except asyncio.TimeoutError:
            logger.warning(f"[ForceElection] Leader at {leader_address} timed out")
        except Exception as e:
            logger.warning(f"[ForceElection] Leader at {leader_address} unreachable: {e}")

        return False

    async def _try_trigger_election(self, reason: str) -> bool:
        """Attempt to trigger a forced election via the orchestrator.

        Session 17.41 (Jan 6, 2026): When partition healing detects an unreachable
        leader, trigger an election to establish a new leader quickly.

        Args:
            reason: Description of why election is being triggered.

        Returns:
            True if election was triggered, False if cooldown or no orchestrator.
        """
        if not self._orchestrator:
            logger.debug("[ForceElection] No orchestrator reference, cannot trigger election")
            return False

        # Check cooldown to prevent election storms
        now = time.time()
        if now - self._last_election_trigger < self._election_cooldown:
            remaining = self._election_cooldown - (now - self._last_election_trigger)
            logger.debug(
                f"[ForceElection] Election cooldown active, {remaining:.0f}s remaining"
            )
            return False

        # Get the start_election method from orchestrator (same pattern as LeaderProbeLoop)
        start_election = getattr(self._orchestrator, "_start_election", None)
        if not start_election:
            logger.warning("[ForceElection] Orchestrator has no _start_election method")
            return False

        try:
            logger.warning(
                f"[ForceElection] Triggering forced election due to partition: {reason}"
            )
            await start_election(reason=f"partition_healing:{reason}")
            self._last_election_trigger = now

            # Emit event for observability
            try:
                from app.distributed.data_events import DataEventType
                from app.coordination.event_router import emit_event

                emit_event(
                    DataEventType.LEADER_ELECTION_STARTED,
                    {
                        "reason": f"partition_healing:{reason}",
                        "triggered_by": "partition_healer",
                        "escalation_level": self._escalation_level,
                        "timestamp": now,
                    },
                )
            except (ImportError, Exception):
                pass  # Best effort

            return True
        except Exception as e:
            logger.error(f"[ForceElection] Failed to trigger election: {e}")
            return False

    async def _escalate_recovery(self, reason: str) -> None:
        """Escalate recovery when convergence repeatedly fails.

        Sprint 10 (Jan 3, 2026): When gossip convergence fails multiple times,
        escalate by increasing wait times and eventually emitting P2P_RECOVERY_NEEDED.

        Escalation tiers:
        - Level 1-2: Log warning, increase probe interval
        - Level 3-4: Log error, longer backoff
        - Level 5+: Emit P2P_RECOVERY_NEEDED for external intervention

        Args:
            reason: Description of the convergence failure
        """
        self._consecutive_failures += 1

        # Determine escalation level
        if self._consecutive_failures >= PartitionHealingDefaults.ESCALATION_THRESHOLD:
            self._escalation_level = min(
                self._escalation_level + 1,
                PartitionHealingDefaults.ESCALATION_MAX_LEVEL,
            )
        else:
            # Not enough consecutive failures for escalation
            logger.info(
                f"[RecoveryEscalation] Convergence failure {self._consecutive_failures}/"
                f"{PartitionHealingDefaults.ESCALATION_THRESHOLD} (not yet escalating)"
            )
            return

        # Calculate wait time for next probe
        wait_time = (
            PartitionHealingDefaults.ESCALATION_BASE_WAIT
            * self._escalation_level
        )

        logger.warning(
            f"[RecoveryEscalation] Escalated to level {self._escalation_level} "
            f"after {self._consecutive_failures} consecutive failures. "
            f"Next probe in {wait_time:.0f}s. Reason: {reason}"
        )

        # Session 17.41 (Jan 6, 2026): Check leader reachability and trigger election if needed
        # At escalation level >= 1, check if leader is unreachable and trigger election
        if self._escalation_level >= 1 and self._orchestrator:
            leader_reachable = await self._check_leader_reachable()
            if not leader_reachable:
                logger.warning(
                    f"[RecoveryEscalation] Leader unreachable at escalation level "
                    f"{self._escalation_level}, attempting forced election"
                )
                election_triggered = await self._try_trigger_election(
                    reason=f"leader_unreachable_escalation_level_{self._escalation_level}"
                )
                if election_triggered:
                    logger.info(
                        "[RecoveryEscalation] Forced election triggered, "
                        "may resolve partition by establishing new leader"
                    )

        # At max escalation, emit event for external recovery
        if (
            self._escalation_level >= PartitionHealingDefaults.ESCALATION_MAX_LEVEL
            and PartitionHealingDefaults.EMIT_RECOVERY_EVENT_AT_MAX
        ):
            try:
                from app.distributed.data_events import DataEventType
                from app.coordination.event_router import emit_event

                emit_event(
                    DataEventType.P2P_RECOVERY_NEEDED,
                    {
                        "reason": f"Partition healing convergence failed repeatedly: {reason}",
                        "escalation_level": self._escalation_level,
                        "consecutive_failures": self._consecutive_failures,
                        "timestamp": time.time(),
                    },
                )
                logger.error(
                    f"[RecoveryEscalation] Emitted P2P_RECOVERY_NEEDED at max escalation. "
                    f"Manual intervention may be required."
                )
            except ImportError:
                logger.warning(
                    "[RecoveryEscalation] Could not emit P2P_RECOVERY_NEEDED "
                    "(event_router not available)"
                )
            except Exception as e:
                logger.warning(f"[RecoveryEscalation] Failed to emit recovery event: {e}")

    def _reset_escalation(self) -> None:
        """Reset escalation state after successful convergence.

        Sprint 10 (Jan 3, 2026): Called when healing succeeds to reset counters.
        """
        if self._consecutive_failures > 0 or self._escalation_level > 0:
            logger.info(
                f"[RecoveryEscalation] Reset after success "
                f"(was level {self._escalation_level}, {self._consecutive_failures} failures)"
            )
        self._consecutive_failures = 0
        self._escalation_level = 0

    def get_status(self) -> dict[str, Any]:
        """Get partition healer status for /status endpoint."""
        now = time.time()
        return {
            "auto_enabled": PartitionHealingDefaults.AUTO_ENABLED,
            "min_interval": PartitionHealingDefaults.MIN_INTERVAL,
            "last_healing_time": self._last_healing_time,
            "time_since_last": now - self._last_healing_time if self._last_healing_time > 0 else None,
            "ready_for_trigger": (
                now - self._last_healing_time >= PartitionHealingDefaults.MIN_INTERVAL
                if self._last_healing_time > 0
                else True
            ),
            # Sprint 10 (Jan 3, 2026): Escalation state
            "consecutive_failures": self._consecutive_failures,
            "escalation_level": self._escalation_level,
            "escalation_threshold": PartitionHealingDefaults.ESCALATION_THRESHOLD,
            "max_escalation_level": PartitionHealingDefaults.ESCALATION_MAX_LEVEL,
            # Session 17.41 (Jan 6, 2026): Election trigger state
            "last_election_trigger": self._last_election_trigger,
            "election_cooldown": self._election_cooldown,
            "election_ready": now - self._last_election_trigger >= self._election_cooldown if self._last_election_trigger > 0 else True,
            "has_orchestrator": self._orchestrator is not None,
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Sprint 15 (Jan 3, 2026): Added for unified health monitoring.
        Session 17.41 (Jan 6, 2026): Added election trigger state.
        """
        # Determine health status based on escalation level
        max_escalation = PartitionHealingDefaults.ESCALATION_MAX_LEVEL
        is_critical = self._escalation_level >= max_escalation
        is_degraded = self._escalation_level > 0
        now = time.time()

        if is_critical:
            status = "critical"
            healthy = False
        elif is_degraded:
            status = "degraded"
            healthy = True  # Still functional, just elevated failure rate
        else:
            status = "healthy"
            healthy = True

        return {
            "healthy": healthy,
            "status": status,
            "details": {
                "auto_enabled": PartitionHealingDefaults.AUTO_ENABLED,
                "escalation_level": self._escalation_level,
                "max_escalation_level": max_escalation,
                "consecutive_failures": self._consecutive_failures,
                "last_healing_time": self._last_healing_time,
                "ready_for_trigger": (
                    now - self._last_healing_time >= PartitionHealingDefaults.MIN_INTERVAL
                    if self._last_healing_time > 0
                    else True
                ),
                # Session 17.41 (Jan 6, 2026): Election trigger state
                "last_election_trigger": self._last_election_trigger,
                "election_cooldown": self._election_cooldown,
                "has_orchestrator": self._orchestrator is not None,
            },
        }

    async def run_daemon(self, interval: float = 60.0) -> None:
        """Run as a background daemon, healing partitions periodically."""
        logger.info(f"Starting partition healer daemon (interval: {interval}s)")

        while True:
            try:
                result = await self.run_healing_pass()
                if result.partitions_healed > 0:
                    logger.info(
                        f"Healing complete: {result.partitions_healed} partitions healed, "
                        f"{result.nodes_reconnected} nodes reconnected"
                    )
            except Exception as e:
                logger.error(f"Error in healing pass: {e}")

            await asyncio.sleep(interval)


# =============================================================================
# Singleton accessors for P2P orchestrator integration
# =============================================================================


def get_partition_healer() -> PartitionHealer:
    """Get or create the singleton PartitionHealer instance.

    January 2026: Used by P2P orchestrator for auto-triggered healing.
    """
    global _partition_healer_instance
    with _partition_healer_lock:
        if _partition_healer_instance is None:
            _partition_healer_instance = PartitionHealer()
        return _partition_healer_instance


def reset_partition_healer() -> None:
    """Reset the singleton instance (for testing)."""
    global _partition_healer_instance
    with _partition_healer_lock:
        _partition_healer_instance = None


def wire_orchestrator_to_partition_healer(orchestrator: Any) -> None:
    """Wire the P2P orchestrator to the partition healer singleton.

    Session 17.41 (Jan 6, 2026): Enables forced elections when partition healing
    detects an unreachable leader. Should be called during orchestrator initialization.

    Args:
        orchestrator: P2P orchestrator instance with _start_election() method.
    """
    healer = get_partition_healer()
    healer.set_orchestrator(orchestrator)


async def trigger_partition_healing(
    delay: float | None = None,
    force: bool = False,
) -> HealingResult | None:
    """Convenience function to trigger healing from P2P orchestrator.

    January 2026: Called on NETWORK_ISOLATION_DETECTED event.

    Args:
        delay: Optional delay before starting (defaults to DETECTION_DELAY)
        force: If True, bypass rate limiting

    Returns:
        HealingResult if healing was performed, None if rate-limited or disabled
    """
    if not PartitionHealingDefaults.AUTO_ENABLED:
        logger.info("Partition auto-healing is disabled")
        return None

    healer = get_partition_healer()
    return await healer.trigger_healing_pass(delay=delay, force=force)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="P2P Partition Healer")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--check-only", action="store_true", help="Check for partitions only")
    parser.add_argument("--interval", type=float, default=60.0, help="Daemon interval (seconds)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    healer = PartitionHealer()

    try:
        if args.check_only:
            known_peers = await healer.discover_all_peers()
            partitions = await healer.detect_partitions(known_peers)
            print(f"\nFound {len(partitions)} partition(s):")
            for p in partitions:
                print(f"  {p.partition_id}: {p.size} members, leader={p.leader}")
                for member in sorted(p.members):
                    addr = p.addresses.get(member, "unknown")
                    print(f"    - {member} ({addr})")
        elif args.once:
            result = await healer.run_healing_pass()
            print(f"\nHealing result:")
            print(f"  Partitions found: {result.partitions_found}")
            print(f"  Partitions healed: {result.partitions_healed}")
            print(f"  Nodes reconnected: {result.nodes_reconnected}")
            print(f"  Duration: {result.duration_ms:.1f}ms")
        else:
            await healer.run_daemon(interval=args.interval)
    finally:
        await healer.close()


if __name__ == "__main__":
    asyncio.run(main())
