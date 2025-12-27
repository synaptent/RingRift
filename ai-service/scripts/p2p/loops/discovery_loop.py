"""Peer Discovery Loops for P2P Orchestrator.

December 2025: Background loops for peer discovery and network formation.

Loops:
- UdpDiscoveryLoop: Broadcast UDP discovery messages to find peers on local network
- FollowerDiscoveryLoop: Discover peers by querying known nodes (for followers)

Usage:
    from scripts.p2p.loops import UdpDiscoveryLoop

    discovery_loop = UdpDiscoveryLoop(
        get_node_id=lambda: orchestrator.node_id,
        get_host=lambda: orchestrator.self_info.host,
        get_port=lambda: orchestrator.port,
        get_known_peers=lambda: orchestrator.known_peers,
        add_peer=lambda addr: orchestrator.known_peers.append(addr),
    )
    await discovery_loop.run_forever()
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from .base import BaseLoop

logger = logging.getLogger(__name__)

# Default discovery port for UDP broadcast
DEFAULT_DISCOVERY_PORT = 8771


@dataclass
class UdpDiscoveryConfig:
    """Configuration for UDP discovery loop."""

    discovery_port: int = DEFAULT_DISCOVERY_PORT
    broadcast_interval_seconds: float = 30.0  # Broadcast every 30 seconds
    listen_timeout_seconds: float = 1.0
    max_message_size: int = 1024

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.discovery_port < 1 or self.discovery_port > 65535:
            raise ValueError("discovery_port must be between 1 and 65535")
        if self.broadcast_interval_seconds <= 0:
            raise ValueError("broadcast_interval_seconds must be > 0")
        if self.listen_timeout_seconds <= 0:
            raise ValueError("listen_timeout_seconds must be > 0")
        if self.max_message_size <= 0:
            raise ValueError("max_message_size must be > 0")


class UdpDiscoveryLoop(BaseLoop):
    """Background loop that broadcasts UDP discovery messages to find peers.

    This loop:
    1. Broadcasts presence on local network via UDP
    2. Listens for peer broadcasts
    3. Adds discovered peers to known_peers list

    Useful for:
    - LAN deployments without central discovery
    - Initial peer bootstrap
    - Network partition recovery
    """

    def __init__(
        self,
        get_node_id: Callable[[], str],
        get_host: Callable[[], str],
        get_port: Callable[[], int],
        get_known_peers: Callable[[], list[str]],
        add_peer: Callable[[str], None],
        config: UdpDiscoveryConfig | None = None,
    ):
        """Initialize UDP discovery loop.

        Args:
            get_node_id: Callback returning this node's ID
            get_host: Callback returning this node's host address
            get_port: Callback returning this node's port
            get_known_peers: Callback returning list of known peer addresses
            add_peer: Callback to add a newly discovered peer
            config: Discovery configuration
        """
        self.config = config or UdpDiscoveryConfig()
        super().__init__(
            name="udp_discovery",
            interval=self.config.broadcast_interval_seconds,
        )
        self._get_node_id = get_node_id
        self._get_host = get_host
        self._get_port = get_port
        self._get_known_peers = get_known_peers
        self._add_peer = add_peer
        self._peers_discovered = 0

    async def _run_once(self) -> None:
        """Broadcast discovery and listen for peer responses."""
        sock = None
        try:
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(self.config.listen_timeout_seconds)

            # Broadcast our presence
            node_id = self._get_node_id()
            host = self._get_host()
            port = self._get_port()

            message = json.dumps({
                "type": "p2p_discovery",
                "node_id": node_id,
                "host": host,
                "port": port,
            }).encode()

            try:
                sock.sendto(message, ('<broadcast>', self.config.discovery_port))
            except OSError as e:
                # Broadcast may fail on some networks (no broadcast route)
                logger.debug(f"[UdpDiscovery] Broadcast failed: {e}")

            # Listen for responses
            known_peers = self._get_known_peers()
            try:
                while True:
                    data, _addr = sock.recvfrom(self.config.max_message_size)
                    try:
                        msg = json.loads(data.decode())
                        if msg.get("type") == "p2p_discovery" and msg.get("node_id") != node_id:
                            # Found a peer!
                            peer_host = msg.get("host")
                            peer_port = msg.get("port")
                            if peer_host and peer_port:
                                peer_addr = f"{peer_host}:{peer_port}"
                                if peer_addr not in known_peers:
                                    self._add_peer(peer_addr)
                                    self._peers_discovered += 1
                                    logger.info(
                                        f"[UdpDiscovery] Discovered peer: "
                                        f"{msg.get('node_id')} at {peer_addr}"
                                    )
                    except json.JSONDecodeError:
                        pass  # Ignore malformed messages
            except TimeoutError:
                pass  # Normal - no more messages in timeout window

        except Exception as e:
            logger.debug(f"[UdpDiscovery] Discovery iteration error: {e}")
        finally:
            if sock:
                try:
                    sock.close()
                except OSError:
                    pass  # Socket cleanup - ignore close errors

    def get_discovery_stats(self) -> dict[str, Any]:
        """Get discovery statistics."""
        return {
            "peers_discovered": self._peers_discovered,
            **self.stats.to_dict(),
        }


@dataclass
class FollowerDiscoveryConfig:
    """Configuration for follower discovery loop."""

    discovery_interval_seconds: float = 60.0
    query_timeout_seconds: float = 10.0
    max_nodes_to_query: int = 5

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.discovery_interval_seconds <= 0:
            raise ValueError("discovery_interval_seconds must be > 0")
        if self.query_timeout_seconds <= 0:
            raise ValueError("query_timeout_seconds must be > 0")
        if self.max_nodes_to_query <= 0:
            raise ValueError("max_nodes_to_query must be > 0")


class FollowerDiscoveryLoop(BaseLoop):
    """Background loop that discovers peers by querying known nodes.

    Used by follower nodes to discover the full cluster membership
    by querying known peers for their peer lists.

    This is more reliable than UDP broadcast for:
    - Cross-subnet deployments
    - Cloud environments
    - When UDP broadcast is disabled
    """

    def __init__(
        self,
        get_known_peers: Callable[[], list[str]],
        query_peer_list: Callable[[str], Coroutine[Any, Any, list[str] | None]],
        add_peer: Callable[[str], None],
        is_leader: Callable[[], bool],
        config: FollowerDiscoveryConfig | None = None,
    ):
        """Initialize follower discovery loop.

        Args:
            get_known_peers: Callback returning list of known peer addresses
            query_peer_list: Async callback to query a peer for its peer list
            add_peer: Callback to add a newly discovered peer
            is_leader: Callback returning whether this node is leader
            config: Discovery configuration
        """
        self.config = config or FollowerDiscoveryConfig()
        super().__init__(
            name="follower_discovery",
            interval=self.config.discovery_interval_seconds,
        )
        self._get_known_peers = get_known_peers
        self._query_peer_list = query_peer_list
        self._add_peer = add_peer
        self._is_leader = is_leader
        self._peers_discovered = 0
        self._queries_made = 0

    async def _run_once(self) -> None:
        """Query known peers for their peer lists to discover new peers."""
        # Leaders don't need to discover - they receive peer registrations
        if self._is_leader():
            return

        known_peers = self._get_known_peers()
        if not known_peers:
            return

        # Query a subset of known peers
        peers_to_query = known_peers[:self.config.max_nodes_to_query]
        current_peer_set = set(known_peers)

        for peer_addr in peers_to_query:
            try:
                self._queries_made += 1
                peer_list = await asyncio.wait_for(
                    self._query_peer_list(peer_addr),
                    timeout=self.config.query_timeout_seconds,
                )

                if peer_list:
                    for new_peer in peer_list:
                        if new_peer not in current_peer_set:
                            self._add_peer(new_peer)
                            current_peer_set.add(new_peer)
                            self._peers_discovered += 1
                            logger.info(f"[FollowerDiscovery] Discovered peer: {new_peer}")

            except asyncio.TimeoutError:
                logger.debug(f"[FollowerDiscovery] Timeout querying {peer_addr}")
            except Exception as e:
                logger.debug(f"[FollowerDiscovery] Failed to query {peer_addr}: {e}")

    def get_discovery_stats(self) -> dict[str, Any]:
        """Get discovery statistics."""
        return {
            "peers_discovered": self._peers_discovered,
            "queries_made": self._queries_made,
            **self.stats.to_dict(),
        }


__all__ = [
    "DEFAULT_DISCOVERY_PORT",
    "UdpDiscoveryConfig",
    "UdpDiscoveryLoop",
    "FollowerDiscoveryConfig",
    "FollowerDiscoveryLoop",
]
