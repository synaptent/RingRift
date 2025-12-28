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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from app.config.ports import SWIM_PORT

# Dec 2025: Use cluster_config helpers instead of inline YAML parsing
try:
    from app.config.cluster_config import get_cluster_nodes, get_p2p_voters
    HAS_CLUSTER_CONFIG = True
except ImportError:
    HAS_CLUSTER_CONFIG = False
    get_cluster_nodes = None
    get_p2p_voters = None

logger = logging.getLogger(__name__)

# Try to import swim-p2p, fall back gracefully if not available
# swim-p2p v1.2.x uses Node.create() factory method with UDPTransport
try:
    from swim import Node as SwimNode, Member as SwimMember, MemberState
    from swim.transport.udp import UDPTransport
    # Verify the factory method exists
    if not hasattr(SwimNode, 'create'):
        raise ImportError("swim-p2p API mismatch: Node.create not found")
    SWIM_AVAILABLE = True
    logger.info("swim-p2p loaded successfully (SWIM membership available)")
except ImportError as e:
    SWIM_AVAILABLE = False
    logger.warning(f"swim-p2p not available: {e}. Install with: pip install swim-p2p")
    # Define stub classes for type hints
    SwimNode = None
    SwimMember = None
    MemberState = None
    UDPTransport = None


@dataclass
class SwimConfig:
    """Configuration for SWIM membership protocol."""

    bind_host: str = "0.0.0.0"
    bind_port: int = SWIM_PORT  # Default SWIM port (different from aiohttp 8770)

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
        bind_port: int = SWIM_PORT,
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
        bind_port: int = SWIM_PORT,
    ) -> "SwimMembershipManager":
        """Create manager with seeds from distributed_hosts.yaml.

        Args:
            node_id: This node's identifier
            config_path: Path to distributed_hosts.yaml
            bind_port: UDP port for SWIM protocol

        Returns:
            Configured SwimMembershipManager
        """
        seeds = []

        # Dec 2025: Use cluster_config helpers instead of inline YAML parsing
        if HAS_CLUSTER_CONFIG and get_p2p_voters is not None:
            try:
                # Use voter nodes as SWIM seeds for bootstrap
                voters = get_p2p_voters()
                nodes = get_cluster_nodes()

                for voter_name in voters:
                    if voter_name != node_id and voter_name in nodes:
                        node = nodes[voter_name]
                        ssh_host = node.best_ip
                        if ssh_host and ssh_host != "localhost":
                            seeds.append((ssh_host, bind_port))

                # If no voters, use first 3 active GPU nodes as seeds
                if not seeds:
                    for host_name, node in nodes.items():
                        if host_name != node_id and node.is_active:
                            ssh_host = node.best_ip
                            if ssh_host and ssh_host != "localhost":
                                seeds.append((ssh_host, bind_port))
                            if len(seeds) >= 3:
                                break

                logger.info(f"SWIM seeds from cluster_config: {seeds}")

            except Exception as e:
                logger.warning(f"Could not load SWIM seeds from cluster_config: {e}")

        # Fallback: Load from YAML directly if cluster_config unavailable
        if not seeds:
            if config_path is None:
                config_path = Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml"

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

                    logger.info(f"SWIM seeds from YAML fallback: {seeds}")
                except Exception as e:
                    logger.warning(f"Could not load SWIM seeds from YAML: {e}")

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
            # swim-p2p 1.2.x uses factory pattern with UDPTransport
            transport = UDPTransport()
            bind_addr = (self.config.bind_host, self.config.bind_port)

            # Configuration for SWIM protocol tuning
            config = {
                "failure_timeout": self.config.failure_timeout,
                "suspicion_timeout": self.config.suspicion_timeout,
                "ping_interval": self.config.ping_interval,
                "ping_request_group_size": self.config.ping_request_group_size,
                "max_transmissions": self.config.max_transmissions,
            }

            # Create node using factory method
            # Dec 28, 2025: Fixed missing await - SwimNode.create() is async
            self._swim = await SwimNode.create(
                bind_addr=bind_addr,
                transport=transport,
                seed_addrs=self.config.seeds if self.config.seeds else None,
                config=config,
                validate_ports=False,  # We handle port validation ourselves
            )

            # Start the SWIM node
            await self._swim.start()
            logger.info(f"SWIM bootstrapped from {len(self.config.seeds)} seeds" if self.config.seeds else "SWIM started (no seeds)")

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
            # Dec 28, 2025: swim-p2p 1.2.x MemberList is not directly iterable
            # Use .all() or .list() method if available, otherwise try .values()
            members = self._swim.members
            if hasattr(members, 'all'):
                members_list = members.all()
            elif hasattr(members, 'list'):
                members_list = members.list()
            elif hasattr(members, 'values'):
                members_list = list(members.values())
            elif hasattr(members, '__iter__'):
                members_list = list(members)
            else:
                # Last resort: try to access via alive/suspect/faulty methods
                alive = []
                if hasattr(members, 'alive'):
                    alive = list(members.alive()) if callable(members.alive) else list(members.alive)
                return [m.id if hasattr(m, 'id') else str(m) for m in alive if str(m) != self.node_id]

            return [
                member.id if hasattr(member, 'id') else str(member)
                for member in members_list
                if (member.state if hasattr(member, 'state') else "alive") == "alive"
                and (member.id if hasattr(member, 'id') else str(member)) != self.node_id
            ]
        except Exception as e:
            logger.error(f"Error getting alive peers: {e}")
            return []

    def _get_members_list(self) -> list:
        """Get members as a list, handling swim-p2p API variations.

        Returns:
            List of member objects
        """
        if not self._swim:
            return []

        members = self._swim.members
        if hasattr(members, 'all'):
            return members.all()
        elif hasattr(members, 'list'):
            return members.list()
        elif hasattr(members, 'values'):
            return list(members.values())
        elif hasattr(members, '__iter__'):
            try:
                return list(members)
            except TypeError:
                pass
        # Last resort: return empty list
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
            members_list = self._get_members_list()
            member = next(
                (m for m in members_list
                 if (m.id if hasattr(m, 'id') else str(m)) == peer_id),
                None
            )
            if member is None:
                return False
            state = member.state if hasattr(member, 'state') else "unknown"
            return state == "alive"
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
            return len(self._get_members_list())
        except (AttributeError, TypeError, StopIteration) as e:
            # Dec 2025: SWIM member iteration may fail if not fully initialized
            logger.debug(f"[SWIMAdapter] Member count failed: {e}")
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
            members = self._get_members_list()
            alive = sum(1 for m in members if (m.state if hasattr(m, 'state') else "unknown") == "alive")
            suspected = sum(1 for m in members if (m.state if hasattr(m, 'state') else "unknown") == "suspected")
            failed = sum(1 for m in members if (m.state if hasattr(m, 'state') else "unknown") == "failed")

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
        swim_port: int = SWIM_PORT,
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
