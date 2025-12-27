"""SWIM-based membership manager for leaderless P2P coordination.

This module provides a gossip-based membership layer using the SWIM protocol,
which offers:
- O(1) message complexity per node (constant bandwidth)
- Failure detection in <5 seconds (vs 60+ seconds with heartbeat-based)
- No single leader required - truly distributed
- Suspicion mechanism to reduce false positives

Integration with existing P2P orchestrator:
- SWIM handles: membership, failure detection, peer discovery
- aiohttp handles: HTTP APIs, work distribution, data transfer

Usage:
    from app.p2p.swim_adapter import SwimMembershipManager

    manager = SwimMembershipManager(
        node_id="nebius-backbone-1",
        bind_port=7947,
        seeds=["nebius-h100-1:7947", "hetzner-cpu1:7947"]
    )
    await manager.start()

    # Check if peer is alive
    if manager.is_peer_alive("runpod-h100"):
        await send_work_to_peer(...)

    # Get all alive peers
    for peer_id in manager.get_alive_peers():
        print(f"{peer_id} is alive")
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

logger = logging.getLogger(__name__)

# Try to import swim-p2p, fall back gracefully if not available
# Note: swim-p2p v1.2.x uses Node/Member classes with a complex builder pattern.
# Full integration requires proper factory setup which is deferred to future work.
# For now, SWIM_AVAILABLE is False to use HTTP heartbeats as fallback.
try:
    from swim import Node as SwimNode, Member as SwimMember, MemberState
    # Check if the API matches our expected interface
    # swim-p2p 1.2.x requires complex builder pattern, not simple constructor
    if not hasattr(SwimNode, '__init__'):
        raise ImportError("swim-p2p API mismatch")
    # Defer full SWIM integration - the library needs builder pattern setup
    # See https://pypi.org/project/swim-p2p/ for full API documentation
    SWIM_AVAILABLE = False  # Temporarily disabled pending factory implementation
    logger.info("swim-p2p installed but SWIM integration deferred (using HTTP heartbeats)")
except ImportError:
    SWIM_AVAILABLE = False
    logger.warning("swim-p2p not installed. Install with: pip install swim-p2p")
    # Define stub classes for type hints
    SwimNode = None
    SwimMember = None
    MemberState = None


@dataclass
class SwimConfig:
    """Configuration for SWIM membership protocol."""

    bind_host: str = "0.0.0.0"
    bind_port: int = 7947  # Default SWIM port (different from aiohttp 8770)

    # Failure detection tuning
    failure_timeout: float = 5.0  # Seconds before marking node as failed
    suspicion_timeout: float = 3.0  # Seconds in suspicion state before failed
    ping_interval: float = 1.0  # Seconds between protocol rounds

    # Protocol tuning
    ping_request_group_size: int = 3  # Number of nodes to ask for indirect ping
    max_transmissions: int = 10  # Max times to transmit membership updates

    # Seeds (initial peers to bootstrap from)
    seeds: list[tuple[str, int]] = field(default_factory=list)


class SwimMembershipManager:
    """Manages cluster membership using SWIM gossip protocol.

    This provides a leaderless alternative to the current voter-based
    leader election. Instead of electing a leader, all nodes have
    an eventually-consistent view of cluster membership.

    Benefits over current leader election:
    1. No leader election needed - all nodes are equal
    2. Faster failure detection (5s vs 60s+)
    3. Lower false positive rate (suspicion mechanism)
    4. Constant bandwidth (O(1) messages per node)
    """

    def __init__(
        self,
        node_id: str,
        bind_port: int = 7947,
        config: Optional[SwimConfig] = None,
        on_member_alive: Optional[Callable[[str], None]] = None,
        on_member_failed: Optional[Callable[[str], None]] = None,
    ):
        """Initialize SWIM membership manager.

        Args:
            node_id: Unique identifier for this node
            bind_port: UDP port for SWIM protocol
            config: Optional SwimConfig, uses defaults if not provided
            on_member_alive: Callback when a member becomes alive
            on_member_failed: Callback when a member is detected as failed
        """
        self.node_id = node_id
        self.config = config or SwimConfig(bind_port=bind_port)
        self.on_member_alive = on_member_alive
        self.on_member_failed = on_member_failed

        self._swim: Optional[SwimNode] = None
        self._started = False
        self._members: dict[str, str] = {}  # node_id -> state

    @classmethod
    def from_distributed_hosts(
        cls,
        node_id: str,
        config_path: Optional[Path] = None,
        bind_port: int = 7947,
    ) -> "SwimMembershipManager":
        """Create manager with seeds from distributed_hosts.yaml.

        Args:
            node_id: This node's identifier
            config_path: Path to distributed_hosts.yaml
            bind_port: UDP port for SWIM protocol

        Returns:
            Configured SwimMembershipManager
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml"

        seeds = []
        if config_path.exists():
            try:
                with open(config_path) as f:
                    hosts_config = yaml.safe_load(f)

                # Use voter nodes as SWIM seeds for bootstrap
                for host_name, host_config in hosts_config.get("hosts", {}).items():
                    if host_config.get("voter", False) and host_name != node_id:
                        ssh_host = host_config.get("ssh_host", "")
                        if ssh_host and ssh_host != "localhost":
                            seeds.append((ssh_host, bind_port))

                # If no voters, use first 3 p2p-enabled nodes as seeds
                if not seeds:
                    for host_name, host_config in hosts_config.get("hosts", {}).items():
                        if host_config.get("p2p_enabled", False) and host_name != node_id:
                            ssh_host = host_config.get("ssh_host", "")
                            if ssh_host and ssh_host != "localhost":
                                seeds.append((ssh_host, bind_port))
                            if len(seeds) >= 3:
                                break

                logger.info(f"SWIM seeds from config: {seeds}")
            except Exception as e:
                logger.warning(f"Could not load SWIM seeds from config: {e}")

        config = SwimConfig(bind_port=bind_port, seeds=seeds)
        return cls(node_id=node_id, bind_port=bind_port, config=config)

    async def start(self) -> bool:
        """Start SWIM membership protocol.

        Returns:
            True if started successfully, False otherwise
        """
        if not SWIM_AVAILABLE:
            logger.error("swim-p2p not installed. Cannot start SWIM membership.")
            return False

        if self._started:
            logger.warning("SWIM membership already started")
            return True

        try:
            self._swim = SwimNode(
                node_id=self.node_id,
                bind_address=(self.config.bind_host, self.config.bind_port),
                failure_timeout=self.config.failure_timeout,
                suspicion_timeout=self.config.suspicion_timeout,
                ping_interval=self.config.ping_interval,
                ping_request_group_size=self.config.ping_request_group_size,
                max_transmissions=self.config.max_transmissions,
            )

            # Register callbacks
            if self.on_member_alive:
                self._swim.on_member_alive(self._handle_member_alive)
            if self.on_member_failed:
                self._swim.on_member_failed(self._handle_member_failed)

            # Start the SWIM node
            await self._swim.start()

            # Bootstrap from seeds
            if self.config.seeds:
                await self._swim.bootstrap(self.config.seeds)
                logger.info(f"SWIM bootstrapped from {len(self.config.seeds)} seeds")

            self._started = True
            logger.info(
                f"SWIM membership started on {self.config.bind_host}:{self.config.bind_port} "
                f"(failure_timeout={self.config.failure_timeout}s)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start SWIM membership: {e}", exc_info=True)
            return False

    async def stop(self):
        """Stop SWIM membership protocol."""
        if self._swim and self._started:
            try:
                await self._swim.stop()
            except Exception as e:
                logger.warning(f"Error stopping SWIM: {e}")
            finally:
                self._started = False
                self._swim = None
                logger.info("SWIM membership stopped")

    def _handle_member_alive(self, member: SwimMember):
        """Handle member becoming alive."""
        self._members[member.id] = "alive"
        logger.info(f"SWIM: {member.id} is now ALIVE")
        if self.on_member_alive:
            try:
                self.on_member_alive(member.id)
            except Exception as e:
                logger.error(f"Error in on_member_alive callback: {e}")

    def _handle_member_failed(self, member: SwimMember):
        """Handle member failure detection."""
        self._members[member.id] = "failed"
        logger.warning(f"SWIM: {member.id} is now FAILED")
        if self.on_member_failed:
            try:
                self.on_member_failed(member.id)
            except Exception as e:
                logger.error(f"Error in on_member_failed callback: {e}")

    def get_alive_peers(self) -> list[str]:
        """Get list of all alive peer node IDs.

        Returns:
            List of node IDs that are currently alive
        """
        if not self._swim or not self._started:
            return []

        try:
            return [
                member.id
                for member in self._swim.members
                if member.state == "alive" and member.id != self.node_id
            ]
        except Exception as e:
            logger.error(f"Error getting alive peers: {e}")
            return []

    def is_peer_alive(self, peer_id: str) -> bool:
        """Check if a specific peer is alive.

        Args:
            peer_id: Node ID to check

        Returns:
            True if peer is alive, False otherwise
        """
        if not self._swim or not self._started:
            return False

        try:
            member = next(
                (m for m in self._swim.members if m.id == peer_id),
                None
            )
            return member is not None and member.state == "alive"
        except Exception as e:
            logger.error(f"Error checking peer {peer_id}: {e}")
            return False

    def get_member_count(self) -> int:
        """Get total number of known members (alive + suspected).

        Returns:
            Number of members in the cluster
        """
        if not self._swim or not self._started:
            return 0

        try:
            return len(list(self._swim.members))
        except Exception:
            return 0

    def get_membership_summary(self) -> dict:
        """Get summary of current membership state.

        Returns:
            Dict with membership statistics
        """
        if not self._swim or not self._started:
            return {
                "node_id": self.node_id,
                "started": False,
                "members": 0,
                "alive": 0,
                "suspected": 0,
                "failed": 0,
            }

        try:
            members = list(self._swim.members)
            alive = sum(1 for m in members if m.state == "alive")
            suspected = sum(1 for m in members if m.state == "suspected")
            failed = sum(1 for m in members if m.state == "failed")

            return {
                "node_id": self.node_id,
                "started": True,
                "members": len(members),
                "alive": alive,
                "suspected": suspected,
                "failed": failed,
                "bind_port": self.config.bind_port,
                "failure_timeout": self.config.failure_timeout,
            }
        except Exception as e:
            logger.error(f"Error getting membership summary: {e}")
            return {
                "node_id": self.node_id,
                "started": True,
                "error": str(e),
            }


class HybridMembershipManager:
    """Hybrid membership manager using SWIM + fallback to HTTP heartbeats.

    This provides a migration path from the current HTTP-based heartbeats
    to SWIM-based membership. It runs both protocols in parallel and
    uses SWIM for membership decisions when available.

    Usage:
        manager = HybridMembershipManager(node_id="my-node")
        await manager.start()

        # Uses SWIM if available, falls back to HTTP heartbeats
        alive_peers = manager.get_alive_peers()
    """

    def __init__(
        self,
        node_id: str,
        swim_port: int = 7947,
        http_peers: Optional[dict[str, str]] = None,  # node_id -> http_endpoint
    ):
        self.node_id = node_id
        self.swim_port = swim_port
        self.http_peers = http_peers or {}

        self.swim_manager: Optional[SwimMembershipManager] = None
        self._http_heartbeats: dict[str, float] = {}  # node_id -> last_heartbeat_time

    async def start(self) -> bool:
        """Start hybrid membership (SWIM + HTTP fallback)."""
        # Try to start SWIM
        if SWIM_AVAILABLE:
            self.swim_manager = SwimMembershipManager.from_distributed_hosts(
                node_id=self.node_id,
                bind_port=self.swim_port,
            )
            swim_started = await self.swim_manager.start()
            if swim_started:
                logger.info("Hybrid membership: SWIM active, HTTP fallback available")
                return True
            else:
                logger.warning("Hybrid membership: SWIM failed, using HTTP only")
                self.swim_manager = None
        else:
            logger.warning("Hybrid membership: swim-p2p not installed, using HTTP only")

        return True  # HTTP fallback is always available

    async def stop(self):
        """Stop hybrid membership."""
        if self.swim_manager:
            await self.swim_manager.stop()

    def record_http_heartbeat(self, node_id: str, timestamp: float):
        """Record an HTTP heartbeat from a peer.

        Called by the HTTP heartbeat handler to track peer liveness.
        """
        self._http_heartbeats[node_id] = timestamp

    def get_alive_peers(self, http_timeout: float = 120.0) -> list[str]:
        """Get list of alive peers using SWIM or HTTP fallback.

        Args:
            http_timeout: Seconds after which HTTP heartbeat is considered stale

        Returns:
            List of alive peer node IDs
        """
        # Prefer SWIM if available
        if self.swim_manager and self.swim_manager._started:
            return self.swim_manager.get_alive_peers()

        # Fallback to HTTP heartbeats
        import time
        now = time.time()
        return [
            node_id
            for node_id, last_hb in self._http_heartbeats.items()
            if (now - last_hb) < http_timeout
        ]

    def is_peer_alive(self, peer_id: str, http_timeout: float = 120.0) -> bool:
        """Check if peer is alive using SWIM or HTTP fallback."""
        if self.swim_manager and self.swim_manager._started:
            return self.swim_manager.is_peer_alive(peer_id)

        # Fallback to HTTP heartbeats
        import time
        last_hb = self._http_heartbeats.get(peer_id, 0)
        return (time.time() - last_hb) < http_timeout

    def get_membership_summary(self) -> dict:
        """Get membership summary including SWIM status."""
        summary = {
            "node_id": self.node_id,
            "mode": "swim" if (self.swim_manager and self.swim_manager._started) else "http",
            "swim_available": SWIM_AVAILABLE,
        }

        if self.swim_manager:
            summary["swim"] = self.swim_manager.get_membership_summary()

        summary["http_peers"] = len(self._http_heartbeats)

        return summary
