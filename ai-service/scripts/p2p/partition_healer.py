"""
Partition Healer for P2P Cluster.

January 2026: Automated network partition detection and healing without full restarts.

Philosophy:
- Union discovery: If ANY source finds a peer, include it
- Gossip everything: Discovered peers are gossiped to the mesh
- Heal don't restart: Inject peers into partitions, don't restart processes
- Convergence: Eventually all nodes should see all other nodes

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

logger = logging.getLogger(__name__)

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
        """
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

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    async def _validate_convergence(
        self,
        known_peers: dict[str, DiscoveredPeer],
        timeout: float | None = None,
    ) -> tuple[bool, str]:
        """
        Validate that gossip has converged after healing.

        Sprint 4 (Jan 2, 2026): Verify that all nodes agree on peer membership
        after healing, instead of assuming success.

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
        reachable_nodes = [
            (node_id, peer.best_address)
            for node_id, peer in known_peers.items()
            if peer.best_address
        ]

        if not reachable_nodes:
            return False, "No reachable nodes for convergence check"

        logger.info(
            f"[ConvergenceCheck] Starting convergence validation "
            f"(timeout={timeout}s, threshold={agreement_threshold*100:.0f}%)"
        )

        while (time.time() - start_time) < timeout:
            # Collect peer views from all reachable nodes
            peer_views: dict[str, set[str]] = {}

            async def get_view(node_id: str, addr: str) -> tuple[str, set[str] | None]:
                view = await self.get_peer_view(addr)
                if view:
                    peers = view.get("peers", {})
                    seen = set(peers.keys())
                    seen.add(view.get("node_id", node_id))
                    return node_id, seen
                return node_id, None

            tasks = [get_view(node_id, addr) for node_id, addr in reachable_nodes]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, tuple):
                    node_id, seen = result
                    if seen is not None:
                        peer_views[node_id] = seen

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

            logger.debug(
                f"[ConvergenceCheck] Progress: {avg_agreement*100:.1f}% agreement "
                f"({total_responding} nodes responding, {len(all_seen_peers)} peers seen)"
            )

            if avg_agreement >= agreement_threshold:
                elapsed = time.time() - start_time
                msg = (
                    f"Convergence achieved: {avg_agreement*100:.1f}% agreement "
                    f"in {elapsed:.1f}s ({total_responding} nodes, {len(all_seen_peers)} peers)"
                )
                logger.info(f"[ConvergenceCheck] {msg}")
                return True, msg

            await asyncio.sleep(check_interval)

        elapsed = time.time() - start_time
        msg = f"Convergence timeout after {elapsed:.1f}s (last agreement: {avg_agreement*100:.1f}%)"
        logger.warning(f"[ConvergenceCheck] {msg}")
        return False, msg

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
                    # Emit success event
                    self._emit_healing_event(result)
                else:
                    # Convergence failed - mark result as partial success
                    result.success = False
                    result.errors.append(f"Convergence failed: {convergence_msg}")
                    self._emit_healing_failed_event(convergence_msg)
                    logger.warning(
                        f"Partition healing completed but convergence not achieved: "
                        f"{convergence_msg}"
                    )

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

        # Apply delay if specified
        delay = delay if delay is not None else PartitionHealingDefaults.DETECTION_DELAY
        if delay > 0:
            logger.info(f"Waiting {delay:.0f}s before starting healing pass...")
            await asyncio.sleep(delay)

        # Run the healing pass
        try:
            result = await self.run_healing_pass()
            return result
        except Exception as e:
            logger.error(f"Healing pass failed: {e}")
            self._emit_healing_failed_event(str(e))
            return HealingResult(
                success=False,
                partitions_found=0,
                partitions_healed=0,
                nodes_reconnected=0,
                errors=[str(e)],
            )

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

    def get_status(self) -> dict[str, Any]:
        """Get partition healer status for /status endpoint."""
        return {
            "auto_enabled": PartitionHealingDefaults.AUTO_ENABLED,
            "min_interval": PartitionHealingDefaults.MIN_INTERVAL,
            "last_healing_time": self._last_healing_time,
            "time_since_last": time.time() - self._last_healing_time if self._last_healing_time > 0 else None,
            "ready_for_trigger": (
                time.time() - self._last_healing_time >= PartitionHealingDefaults.MIN_INTERVAL
                if self._last_healing_time > 0
                else True
            ),
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
