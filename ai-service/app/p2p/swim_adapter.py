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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from app.config.ports import SWIM_PORT

# Jan 22, 2026: Import tuned SWIM timeouts from constants.py (canonical source)
# These values are calibrated for high-latency cross-cloud networks (P99 RTT ~2.6s)
# Fallback values must match constants.py defaults to avoid timeout disagreement
try:
    from app.p2p.constants import (
        SWIM_FAILURE_TIMEOUT,
        SWIM_SUSPICION_TIMEOUT,
        SWIM_PING_INTERVAL,
        SWIM_INDIRECT_PING_COUNT,
    )
except ImportError:
    # Fallback to match constants.py defaults (30s/20s for cross-cloud latency)
    SWIM_FAILURE_TIMEOUT = 30.0
    SWIM_SUSPICION_TIMEOUT = 20.0
    SWIM_PING_INTERVAL = 1.0
    SWIM_INDIRECT_PING_COUNT = 7

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
# swim-p2p v1.2.x uses Node.create() factory method with transport layer
# December 30, 2025: Added TCP transport support for environments where UDP is blocked
try:
    from swim import Node as SwimNode, Member as SwimMember, MemberState
    from swim.transport.udp import UDPTransport
    from swim.transport.tcp import TCPTransport
    # Verify the factory method exists
    if not hasattr(SwimNode, 'create'):
        raise ImportError("swim-p2p API mismatch: Node.create not found")
    SWIM_AVAILABLE = True
    TCP_TRANSPORT_AVAILABLE = True
    logger.info("swim-p2p loaded successfully (SWIM membership available, TCP+UDP transports)")
except ImportError as e:
    SWIM_AVAILABLE = False
    TCP_TRANSPORT_AVAILABLE = False
    logger.warning(f"swim-p2p not available: {e}. Install with: pip install swim-p2p")
    # Define stub classes for type hints
    SwimNode = None
    SwimMember = None
    MemberState = None
    UDPTransport = None
    TCPTransport = None

# Transport selection via environment variable
# December 30, 2025: Default to TCP since UDP port 7947 is often blocked by firewalls
SWIM_TRANSPORT = os.environ.get("RINGRIFT_SWIM_TRANSPORT", "tcp").lower()
if SWIM_TRANSPORT not in ("tcp", "udp"):
    logger.warning(f"Invalid SWIM_TRANSPORT '{SWIM_TRANSPORT}', defaulting to tcp")
    SWIM_TRANSPORT = "tcp"


@dataclass
class SwimBootstrapConfig:
    """Configuration for SWIM bootstrap retry behavior.

    December 29, 2025: Added for resilient SWIM initialization.
    Implements exponential backoff for joining the SWIM cluster.
    """

    max_attempts: int = 5  # Maximum bootstrap attempts
    initial_delay_seconds: float = 1.0  # Initial delay before first retry
    max_delay_seconds: float = 30.0  # Maximum delay cap
    backoff_multiplier: float = 2.0  # Delay multiplier per attempt
    seed_rotation: bool = True  # Try different seeds on retry


@dataclass
class SwimConfig:
    """Configuration for SWIM membership protocol."""

    bind_host: str = "0.0.0.0"
    bind_port: int = SWIM_PORT  # Default SWIM port (different from aiohttp 8770)

    # Failure detection tuning (December 30, 2025: Use tuned values from constants.py)
    # These are calibrated for high-latency cross-cloud networks (P99 RTT ~2.6s)
    failure_timeout: float = SWIM_FAILURE_TIMEOUT  # 10.0s default for cross-cloud
    suspicion_timeout: float = SWIM_SUSPICION_TIMEOUT  # 6.0s default for cross-cloud
    ping_interval: float = SWIM_PING_INTERVAL  # 1.0s between protocol rounds

    # Protocol tuning (December 30, 2025: Increased indirect pings for reliability)
    ping_request_group_size: int = SWIM_INDIRECT_PING_COUNT  # 7 nodes for indirect ping
    max_transmissions: int = 10  # Max times to transmit membership updates

    # Seeds (initial peers to bootstrap from)
    seeds: list[tuple[str, int]] = field(default_factory=list)

    # Bootstrap retry configuration (December 29, 2025)
    bootstrap: SwimBootstrapConfig = field(default_factory=SwimBootstrapConfig)

    # Transport type (December 30, 2025: TCP for firewall-blocked environments)
    # If not specified, uses RINGRIFT_SWIM_TRANSPORT env var (default: tcp)
    transport: str = field(default_factory=lambda: SWIM_TRANSPORT)


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

            except (ImportError, yaml.YAMLError, OSError, KeyError, AttributeError) as e:
                # Import/config loading errors - fall back to direct YAML loading
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
                except (yaml.YAMLError, OSError, KeyError, TypeError) as e:
                    # YAML parsing or file access errors
                    logger.warning(f"Could not load SWIM seeds from YAML: {e}")

        config = SwimConfig(bind_port=bind_port, seeds=seeds)
        return cls(node_id=node_id, bind_port=bind_port, config=config)

    async def start(self) -> bool:
        """Start SWIM membership protocol with retry logic.

        December 29, 2025: Enhanced with exponential backoff retry for bootstrap.
        Uses SwimBootstrapConfig for retry parameters.

        Returns:
            True if started successfully, False otherwise
        """
        if not SWIM_AVAILABLE:
            logger.error("swim-p2p not installed. Cannot start SWIM membership.")
            return False

        if self._started:
            logger.warning("SWIM membership already started")
            return True

        bootstrap_config = self.config.bootstrap
        last_error: Exception | None = None

        for attempt in range(bootstrap_config.max_attempts):
            try:
                # swim-p2p 1.2.x uses factory pattern with transport layer
                # December 30, 2025: Support TCP transport for firewall-blocked environments
                transport_type = self.config.transport
                if transport_type == "tcp" and TCP_TRANSPORT_AVAILABLE:
                    transport = TCPTransport()
                    logger.info(f"SWIM using TCP transport (attempt {attempt + 1})")
                else:
                    transport = UDPTransport()
                    if transport_type == "tcp":
                        logger.warning(f"TCP transport requested but unavailable, falling back to UDP")
                    logger.info(f"SWIM using UDP transport (attempt {attempt + 1})")
                bind_addr = (self.config.bind_host, self.config.bind_port)

                # Configuration for SWIM protocol tuning
                config = {
                    "failure_timeout": self.config.failure_timeout,
                    "suspicion_timeout": self.config.suspicion_timeout,
                    "ping_interval": self.config.ping_interval,
                    "ping_request_group_size": self.config.ping_request_group_size,
                    "max_transmissions": self.config.max_transmissions,
                }

                # December 29, 2025: Seed rotation for retry resilience
                # Rotate seeds on retry to try different bootstrap nodes
                seeds_to_use = self.config.seeds
                if bootstrap_config.seed_rotation and attempt > 0 and self.config.seeds:
                    # Rotate seeds by attempt number to try different bootstrap order
                    rotation = attempt % len(self.config.seeds)
                    seeds_to_use = self.config.seeds[rotation:] + self.config.seeds[:rotation]
                    logger.debug(f"SWIM bootstrap attempt {attempt + 1}: rotated seeds")

                # Create node using factory method
                # Dec 28, 2025: Fixed missing await - SwimNode.create() is async
                self._swim = await SwimNode.create(
                    bind_addr=bind_addr,
                    transport=transport,
                    seed_addrs=seeds_to_use if seeds_to_use else None,
                    config=config,
                    validate_ports=False,  # We handle port validation ourselves
                )

                # Start the SWIM node
                await self._swim.start()
                logger.info(
                    f"SWIM bootstrapped from {len(self.config.seeds)} seeds"
                    if self.config.seeds else "SWIM started (no seeds)"
                )

                self._started = True
                if attempt > 0:
                    logger.info(f"SWIM membership started on attempt {attempt + 1}")
                logger.info(
                    f"SWIM membership started on {self.config.bind_host}:{self.config.bind_port} "
                    f"(failure_timeout={self.config.failure_timeout}s)"
                )
                return True

            except (OSError, ConnectionError, TimeoutError) as e:
                # Network-related errors are retryable
                last_error = e
                logger.warning(
                    f"SWIM bootstrap attempt {attempt + 1}/{bootstrap_config.max_attempts} "
                    f"failed (network): {e}"
                )
            except (RuntimeError, ValueError) as e:
                # Configuration/initialization errors may not be retryable
                last_error = e
                logger.warning(
                    f"SWIM bootstrap attempt {attempt + 1}/{bootstrap_config.max_attempts} "
                    f"failed: {e}"
                )

            # Don't sleep after final failed attempt
            if attempt < bootstrap_config.max_attempts - 1:
                # Calculate delay with exponential backoff
                import asyncio
                delay = min(
                    bootstrap_config.initial_delay_seconds * (bootstrap_config.backoff_multiplier ** attempt),
                    bootstrap_config.max_delay_seconds,
                )
                logger.info(f"SWIM retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        # All attempts failed
        logger.error(
            f"SWIM membership failed to start after {bootstrap_config.max_attempts} attempts: {last_error}",
            exc_info=last_error is not None,
        )
        return False

    async def stop(self):
        """Stop SWIM membership protocol."""
        if self._swim and self._started:
            try:
                await self._swim.stop()
            except (RuntimeError, asyncio.CancelledError) as e:
                logger.warning(f"Error stopping SWIM: {e}")
            finally:
                self._started = False
                self._swim = None
                logger.info("SWIM membership stopped")

    async def restart(self) -> bool:
        """Restart SWIM membership protocol.

        December 29, 2025: Added for connection recovery.
        Stops the current instance and starts a fresh one with retry logic.

        Returns:
            True if restarted successfully, False otherwise
        """
        logger.info("SWIM membership restarting...")
        await self.stop()
        return await self.start()

    def is_healthy(self) -> bool:
        """Check if SWIM membership is healthy.

        December 29, 2025: Added for health monitoring.

        Returns:
            True if SWIM is running and has active members, False otherwise
        """
        if not self._swim or not self._started:
            return False

        try:
            # Consider healthy if we have at least one other member
            member_count = self.get_member_count()
            return member_count > 0
        except (RuntimeError, OSError, AttributeError) as e:
            # Dec 2025: Narrowed from broad Exception - SWIM protocol errors
            logger.debug(f"SWIM health check failed: {e}")
            return False

    def get_health_status(self) -> dict:
        """Get detailed SWIM health status.

        December 29, 2025: Added for observability and debugging.

        Returns:
            Dict with health metrics and diagnostics
        """
        if not SWIM_AVAILABLE:
            return {
                "healthy": False,
                "started": False,
                "reason": "swim-p2p package not installed",
                "swim_available": False,
            }

        if not self._swim or not self._started:
            return {
                "healthy": False,
                "started": False,
                "reason": "SWIM not started",
                "swim_available": True,
            }

        try:
            summary = self.get_membership_summary()
            alive_count = summary.get("alive", 0)
            member_count = summary.get("members", 0)

            # Health criteria:
            # - Must have at least one other member (unless no seeds configured)
            # - Alive ratio should be > 50% of known members
            has_members = member_count > 0 or not self.config.seeds
            good_alive_ratio = member_count == 0 or (alive_count / max(1, member_count)) >= 0.5

            healthy = has_members and good_alive_ratio

            return {
                "healthy": healthy,
                "started": True,
                "swim_available": True,
                "node_id": self.node_id,
                "bind_port": self.config.bind_port,
                "members": member_count,
                "alive": alive_count,
                "suspected": summary.get("suspected", 0),
                "failed": summary.get("failed", 0),
                "seeds_configured": len(self.config.seeds),
                "failure_timeout": self.config.failure_timeout,
                "reason": None if healthy else (
                    "no members discovered" if not has_members
                    else "low alive ratio" if not good_alive_ratio
                    else "unknown"
                ),
            }
        except (RuntimeError, AttributeError, ValueError) as e:
            return {
                "healthy": False,
                "started": True,
                "swim_available": True,
                "reason": f"health check error: {e}",
                "error": str(e),
            }

    def _handle_member_alive(self, member: SwimMember):
        """Handle member becoming alive."""
        member_id = self._get_member_id(member)
        self._members[member_id] = "alive"
        logger.info(f"SWIM: {member_id} is now ALIVE")
        if self.on_member_alive:
            try:
                self.on_member_alive(member_id)
            except (TypeError, AttributeError, ValueError) as e:
                logger.error(f"Error in on_member_alive callback: {e}")

    def _handle_member_failed(self, member: SwimMember):
        """Handle member failure detection.

        Jan 29, 2026: SWIM failure detection is now INFORMATIONAL ONLY.
        Raft is the authoritative source for peer liveness. SWIM still
        tracks failure state internally but does NOT emit liveness events.
        This prevents split-brain from SWIM detecting failure at 30-50s
        while Raft/HTTP uses 180s timeout.

        SWIM is still used for peer DISCOVERY (on_member_alive).
        """
        member_id = self._get_member_id(member)
        self._members[member_id] = "failed"
        # Log only - Raft is authoritative for liveness
        logger.debug(f"SWIM: {member_id} detected as FAILED (informational only, Raft is authoritative)")
        # Jan 29, 2026: DISABLED callback - causes split-brain when SWIM timeout (30-50s)
        # disagrees with Raft/HTTP timeout (180s). Raft handles liveness authoritatively.
        # if self.on_member_failed:
        #     try:
        #         self.on_member_failed(member_id)
        #     except (TypeError, AttributeError, ValueError) as e:
        #         logger.error(f"Error in on_member_failed callback: {e}")

    @staticmethod
    def _get_member_id(member) -> str:
        """Extract a string ID from a SWIM Member object.

        Jan 2, 2026: SWIM Members have 'addr' attribute (host, port tuple),
        not 'id'. This helper extracts a consistent string identifier.

        Args:
            member: SWIM Member object

        Returns:
            String identifier like "100.113.2.45:7947"
        """
        # Try explicit id attribute first (some SWIM variants)
        if hasattr(member, 'id') and member.id:
            return str(member.id)

        # SWIM 1.2.x uses 'addr' attribute: (host, port) tuple
        if hasattr(member, 'addr') and member.addr:
            host, port = member.addr
            return f"{host}:{port}"

        # Try 'address' attribute (alternative naming)
        if hasattr(member, 'address') and member.address:
            if isinstance(member.address, tuple):
                host, port = member.address
                return f"{host}:{port}"
            return str(member.address)

        # Last resort: str() but try to parse address from it
        # Format: "Member(100.113.2.45:7947, ALIVE, inc=1)"
        member_str = str(member)
        if member_str.startswith("Member(") and ":" in member_str:
            # Extract IP:port from "Member(IP:port, ...)"
            import re
            match = re.search(r'Member\(([^,]+)', member_str)
            if match:
                return match.group(1).strip()

        return member_str

    def get_alive_peers(self) -> list[str]:
        """Get list of all alive peer node IDs.

        December 30, 2025: Updated to use correct swim-p2p 1.2.x MemberList API.
        Jan 2, 2026: Fixed to use _get_member_id() helper for proper ID extraction.

        Returns:
            List of node IDs that are currently alive
        """
        if not self._swim or not self._started:
            return []

        try:
            members = self._swim.members
            # Dec 30, 2025: swim-p2p 1.2.x uses get_alive_members()
            if hasattr(members, 'get_alive_members'):
                alive_members = members.get_alive_members()
                return [
                    self._get_member_id(member)
                    for member in alive_members
                    if self._get_member_id(member) != self.node_id
                ]

            # Fallback: get all members and filter by state
            members_list = self._get_members_list()
            return [
                self._get_member_id(member)
                for member in members_list
                if (member.state if hasattr(member, 'state') else "alive") == "alive"
                and self._get_member_id(member) != self.node_id
            ]
        except (RuntimeError, AttributeError, StopIteration) as e:
            logger.error(f"Error getting alive peers: {e}")
            return []

    def _get_members_list(self) -> list:
        """Get members as a list, handling swim-p2p API variations.

        December 30, 2025: Updated to use correct swim-p2p 1.2.x MemberList API.
        The MemberList class uses get_all_members() and get_alive_members().

        Returns:
            List of member objects
        """
        if not self._swim:
            return []

        members = self._swim.members
        # Dec 30, 2025: swim-p2p 1.2.x uses get_all_members() or get_alive_members()
        if hasattr(members, 'get_all_members'):
            return members.get_all_members()
        elif hasattr(members, 'get_alive_members'):
            return members.get_alive_members()
        elif hasattr(members, 'all'):
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
                 if self._get_member_id(m) == peer_id),
                None
            )
            if member is None:
                return False
            state = member.state if hasattr(member, 'state') else "unknown"
            return state == "alive"
        except (RuntimeError, AttributeError, StopIteration) as e:
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
                "transport": self.config.transport,
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
                "transport": self.config.transport,
                "members": len(members),
                "alive": alive,
                "suspected": suspected,
                "failed": failed,
                "bind_port": self.config.bind_port,
                "failure_timeout": self.config.failure_timeout,
            }
        except (RuntimeError, AttributeError, ValueError) as e:
            logger.error(f"Error getting membership summary: {e}")
            return {
                "node_id": self.node_id,
                "started": True,
                "transport": self.config.transport,
                "error": str(e),
            }


class HybridMembershipManager:
    """Hybrid membership manager using SWIM + fallback to HTTP heartbeats.

    December 29, 2025: Enhanced with true hybrid coordination.
    This provides a migration path from the current HTTP-based heartbeats
    to SWIM-based membership. It runs both protocols in parallel and
    uses SWIM for membership decisions when available, with HTTP fallback.

    Features:
    - True hybrid mode: SWIM for fast detection, HTTP for resilience
    - Auto-recovery: Attempts SWIM restart on failures
    - State synchronization: HTTP state updates SWIM expectations
    - Graceful degradation: Falls back to HTTP when SWIM unavailable

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
        prefer_swim: bool = True,  # Dec 29, 2025: Whether to prefer SWIM over HTTP
    ):
        self.node_id = node_id
        self.swim_port = swim_port
        self.http_peers = http_peers or {}
        self.prefer_swim = prefer_swim

        self.swim_manager: Optional[SwimMembershipManager] = None
        self._http_heartbeats: dict[str, float] = {}  # node_id -> last_heartbeat_time
        self._mode_switches: int = 0  # Dec 29, 2025: Track mode transitions
        self._last_swim_check: float = 0.0  # Dec 29, 2025: Track health check timing
        self._swim_failure_count: int = 0  # Dec 29, 2025: Track consecutive failures

    async def start(self) -> bool:
        """Start hybrid membership (SWIM + HTTP fallback).

        December 29, 2025: Enhanced with retry logic via SwimBootstrapConfig.
        """
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
                self._swim_failure_count += 1
        else:
            logger.warning("Hybrid membership: swim-p2p not installed, using HTTP only")

        return True  # HTTP fallback is always available

    async def stop(self):
        """Stop hybrid membership."""
        if self.swim_manager:
            await self.swim_manager.stop()

    async def check_and_recover(self) -> bool:
        """Check SWIM health and attempt recovery if needed.

        December 29, 2025: Added for proactive SWIM recovery.
        Should be called periodically (e.g., every 60 seconds).

        Returns:
            True if healthy or recovered, False if recovery failed
        """
        import time
        now = time.time()

        # Rate limit health checks (every 60 seconds max)
        if now - self._last_swim_check < 60:
            return True
        self._last_swim_check = now

        # Check if SWIM should be running but isn't
        if SWIM_AVAILABLE and self.swim_manager is None:
            # SWIM available but not initialized - try to start
            logger.info("Hybrid membership: Attempting to initialize SWIM")
            self.swim_manager = SwimMembershipManager.from_distributed_hosts(
                node_id=self.node_id,
                bind_port=self.swim_port,
            )
            if await self.swim_manager.start():
                self._swim_failure_count = 0
                self._mode_switches += 1
                logger.info("Hybrid membership: SWIM initialized successfully")
                return True
            else:
                self.swim_manager = None
                self._swim_failure_count += 1
                logger.warning(f"Hybrid membership: SWIM init failed (failures: {self._swim_failure_count})")

        # Check if SWIM is running but unhealthy
        if self.swim_manager and self.swim_manager._started:
            if hasattr(self.swim_manager, "is_healthy") and not self.swim_manager.is_healthy():
                logger.warning("Hybrid membership: SWIM unhealthy, attempting restart")
                if hasattr(self.swim_manager, "restart"):
                    if await self.swim_manager.restart():
                        self._swim_failure_count = 0
                        logger.info("Hybrid membership: SWIM restarted successfully")
                        return True
                    else:
                        self._swim_failure_count += 1
                        logger.error(f"Hybrid membership: SWIM restart failed (failures: {self._swim_failure_count})")

        return True  # HTTP fallback always available

    def record_http_heartbeat(self, node_id: str, timestamp: float):
        """Record an HTTP heartbeat from a peer.

        Called by the HTTP heartbeat handler to track peer liveness.
        """
        self._http_heartbeats[node_id] = timestamp

    def get_alive_peers(self, http_timeout: float = 120.0) -> list[str]:
        """Get list of alive peers using hybrid SWIM + HTTP.

        December 29, 2025: Now returns union of SWIM and HTTP alive peers
        for true hybrid operation (not just fallback).

        Args:
            http_timeout: Seconds after which HTTP heartbeat is considered stale

        Returns:
            List of alive peer node IDs (union of both sources)
        """
        import time
        now = time.time()

        swim_alive: set[str] = set()
        http_alive: set[str] = set()

        # Get SWIM alive peers if available
        if self.swim_manager and self.swim_manager._started:
            try:
                swim_alive = set(self.swim_manager.get_alive_peers())
            except (RuntimeError, AttributeError, StopIteration) as e:
                logger.debug(f"SWIM get_alive_peers error: {e}")

        # Get HTTP alive peers
        http_alive = {
            node_id
            for node_id, last_hb in self._http_heartbeats.items()
            if (now - last_hb) < http_timeout
        }

        # Return union of both sources for true hybrid
        return list(swim_alive | http_alive)

    def is_peer_alive(self, peer_id: str, http_timeout: float = 120.0) -> bool:
        """Check if peer is alive using hybrid SWIM + HTTP.

        December 29, 2025: Now uses OR logic - alive if EITHER source reports alive.

        Args:
            peer_id: Node ID to check
            http_timeout: HTTP heartbeat timeout in seconds

        Returns:
            True if peer is alive via SWIM OR HTTP
        """
        swim_alive = False
        http_alive = False

        # Check SWIM
        if self.swim_manager and self.swim_manager._started:
            try:
                swim_alive = self.swim_manager.is_peer_alive(peer_id)
            except Exception as e:
                logger.debug(f"SWIM is_peer_alive error: {e}")

        # Check HTTP
        import time
        last_hb = self._http_heartbeats.get(peer_id, 0)
        http_alive = (time.time() - last_hb) < http_timeout

        # True hybrid: alive if EITHER source reports alive
        return swim_alive or http_alive

    def get_membership_summary(self) -> dict:
        """Get membership summary including SWIM status."""
        import time
        now = time.time()

        summary = {
            "node_id": self.node_id,
            "mode": "swim" if (self.swim_manager and self.swim_manager._started) else "http",
            "swim_available": SWIM_AVAILABLE,
            "hybrid_enabled": True,  # Dec 29, 2025
            "prefer_swim": self.prefer_swim,
            "mode_switches": self._mode_switches,
            "swim_failure_count": self._swim_failure_count,
        }

        if self.swim_manager:
            summary["swim"] = self.swim_manager.get_membership_summary()

        # HTTP stats
        http_timeout = 120.0
        http_alive_count = sum(
            1 for last_hb in self._http_heartbeats.values()
            if (now - last_hb) < http_timeout
        )
        summary["http_peers"] = len(self._http_heartbeats)
        summary["http_alive"] = http_alive_count

        return summary

    def get_health_status(self) -> dict:
        """Get health status for hybrid membership.

        December 29, 2025: Added for unified health monitoring.

        Returns:
            Dict with health status and detailed metrics
        """
        import time
        now = time.time()
        http_timeout = 120.0

        # Get individual health statuses
        swim_health = {}
        if self.swim_manager:
            if hasattr(self.swim_manager, "get_health_status"):
                swim_health = self.swim_manager.get_health_status()
            else:
                swim_health = {"healthy": self.swim_manager._started}

        # HTTP health
        http_alive_count = sum(
            1 for last_hb in self._http_heartbeats.values()
            if (now - last_hb) < http_timeout
        )

        # Hybrid is healthy if EITHER source is healthy
        swim_healthy = swim_health.get("healthy", False)
        http_healthy = http_alive_count > 0 or not self.http_peers

        return {
            "healthy": swim_healthy or http_healthy,
            "swim_healthy": swim_healthy,
            "http_healthy": http_healthy,
            "mode": "swim" if swim_healthy else "http",
            "swim_available": SWIM_AVAILABLE,
            "swim_failure_count": self._swim_failure_count,
            "http_peers": len(self._http_heartbeats),
            "http_alive": http_alive_count,
            "swim": swim_health,
        }
