"""Gossip Peer Promotion Loop for P2P Orchestrator.

January 2026: Addresses the critical gap where gossip-learned peers are stored
in _gossip_learned_endpoints but never automatically promoted to self.peers.

Problem:
- Peers discovered via gossip messages are stored in _gossip_learned_endpoints
- The _try_connect_gossip_peer method is called reactively when gossip arrives
- But if the initial connection attempt fails, there's no retry mechanism
- This leads to gossip-learned peers never becoming active even if they become
  reachable later

Solution:
- This loop periodically scans _gossip_learned_endpoints for unconnected peers
- Attempts to connect to each peer via heartbeat
- Successfully connected peers are added to self.peers
- Uses exponential backoff per-peer to avoid hammering unreachable nodes

Usage:
    from scripts.p2p.loops.gossip_peer_promotion_loop import GossipPeerPromotionLoop

    loop = GossipPeerPromotionLoop(orchestrator)
    await loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .base import BaseLoop, BackoffConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class GossipPeerPromotionConfig:
    """Configuration for gossip peer promotion loop."""

    # How often to run the promotion loop
    interval_seconds: float = 30.0

    # Max peers to try connecting to per cycle (to avoid hammering network)
    max_peers_per_cycle: int = 5

    # Minimum time between connection attempts to the same peer
    min_retry_interval_seconds: float = 60.0

    # Max consecutive failures before backing off aggressively
    max_consecutive_failures: int = 5

    # Backoff after max consecutive failures
    failure_backoff_seconds: float = 300.0  # 5 minutes

    # Timeout for connection attempts
    connection_timeout_seconds: float = 10.0


@dataclass
class PeerConnectionState:
    """Tracks connection attempt state for a peer."""

    node_id: str
    host: str
    port: int
    tailscale_ip: str = ""
    last_attempt_time: float = 0.0
    consecutive_failures: int = 0
    total_attempts: int = 0
    last_success_time: float = 0.0


class GossipPeerPromotionLoop(BaseLoop):
    """Background loop that promotes gossip-learned peers to active peers.

    This loop:
    1. Periodically scans _gossip_learned_endpoints for unconnected peers
    2. Attempts to connect to each peer via heartbeat
    3. Successfully connected peers are added to self.peers
    4. Uses per-peer backoff to avoid hammering unreachable nodes

    Benefits:
    - Ensures gossip-learned peers eventually become active
    - Recovers from transient network issues
    - Respects rate limits with per-peer backoff
    """

    def __init__(
        self,
        orchestrator: Any,
        config: GossipPeerPromotionConfig | None = None,
    ):
        """Initialize gossip peer promotion loop.

        Args:
            orchestrator: P2P orchestrator instance with _gossip_learned_endpoints,
                          peers, and _try_connect_gossip_peer method
            config: Loop configuration
        """
        self.config = config or GossipPeerPromotionConfig()
        super().__init__(
            name="gossip_peer_promotion",
            interval=self.config.interval_seconds,
        )

        self._orchestrator = orchestrator

        # Track connection state per peer for smart backoff
        self._peer_states: dict[str, PeerConnectionState] = {}

        # Stats for observability (don't override parent's _stats!)
        self._promotion_stats = {
            "total_promotion_attempts": 0,
            "successful_promotions": 0,
            "failed_promotions": 0,
            "peers_currently_tracked": 0,
        }

    def health_check(self) -> dict[str, Any]:
        """Return health check data for DaemonManager integration."""
        return {
            "name": self.name,
            "running": self._running,
            "stats": self._stats.to_dict() if hasattr(self._stats, 'to_dict') else {},
            "promotion_stats": self._promotion_stats.copy(),
            "tracked_peers": len(self._peer_states),
        }

    async def _run_once(self) -> None:
        """Run one iteration of the gossip peer promotion loop."""
        orchestrator = self._orchestrator
        if orchestrator is None:
            return

        # Get gossip-learned endpoints
        endpoints = getattr(orchestrator, "_gossip_learned_endpoints", {})
        if not endpoints:
            logger.debug("[GossipPromotion] No gossip-learned endpoints to process")
            return

        # Get currently connected peers
        peers = getattr(orchestrator, "peers", {})

        now = time.time()
        candidates = []

        # Find peers that are in gossip endpoints but not in active peers
        for node_id, endpoint in endpoints.items():
            # Skip if already connected
            if node_id in peers and hasattr(peers[node_id], "is_alive"):
                if peers[node_id].is_alive():
                    # Already connected and alive - clear any failure state
                    if node_id in self._peer_states:
                        self._peer_states[node_id].consecutive_failures = 0
                        self._peer_states[node_id].last_success_time = now
                    continue

            # Skip self
            if node_id == getattr(orchestrator, "node_id", None):
                continue

            # Get or create connection state
            if node_id not in self._peer_states:
                host = endpoint.get("host", "")
                port = endpoint.get("port", 8770)
                tailscale_ip = endpoint.get("tailscale_ip", "")
                self._peer_states[node_id] = PeerConnectionState(
                    node_id=node_id,
                    host=host,
                    port=port,
                    tailscale_ip=tailscale_ip,
                )

            state = self._peer_states[node_id]

            # Check if we should retry
            time_since_last = now - state.last_attempt_time

            # Apply backoff based on failure count
            if state.consecutive_failures >= self.config.max_consecutive_failures:
                required_wait = self.config.failure_backoff_seconds
            else:
                required_wait = self.config.min_retry_interval_seconds

            if time_since_last < required_wait:
                continue

            candidates.append(state)

        if not candidates:
            logger.debug("[GossipPromotion] No candidates for promotion this cycle")
            return

        # Limit candidates per cycle
        candidates = candidates[: self.config.max_peers_per_cycle]
        self._promotion_stats["peers_currently_tracked"] = len(self._peer_states)

        logger.info(
            f"[GossipPromotion] Attempting to promote {len(candidates)} gossip peers"
        )

        # Try connecting to each candidate
        for state in candidates:
            await self._try_promote_peer(state, orchestrator, now)

    async def _try_promote_peer(
        self,
        state: PeerConnectionState,
        orchestrator: Any,
        now: float,
    ) -> None:
        """Attempt to promote a single gossip peer to active status."""
        state.last_attempt_time = now
        state.total_attempts += 1
        self._promotion_stats["total_promotion_attempts"] += 1

        # Try multiple addresses: tailscale_ip first, then host
        addresses_to_try = []
        if state.tailscale_ip:
            addresses_to_try.append((state.tailscale_ip, state.port))
        if state.host and state.host != state.tailscale_ip:
            addresses_to_try.append((state.host, state.port))

        success = False

        for host, port in addresses_to_try:
            try:
                logger.debug(
                    f"[GossipPromotion] Trying {state.node_id} at {host}:{port}"
                )

                # Use the orchestrator's _send_heartbeat_to_peer method
                send_heartbeat = getattr(
                    orchestrator, "_send_heartbeat_to_peer", None
                )
                if send_heartbeat is None:
                    logger.warning(
                        "[GossipPromotion] Orchestrator has no _send_heartbeat_to_peer"
                    )
                    break

                # Send heartbeat with timeout
                info = await asyncio.wait_for(
                    send_heartbeat(host, port),
                    timeout=self.config.connection_timeout_seconds,
                )

                if info:
                    # Successfully connected - add to peers
                    peers_lock = getattr(orchestrator, "peers_lock", None)
                    peers = getattr(orchestrator, "peers", {})

                    if peers_lock:
                        async with peers_lock:
                            peers[info.node_id] = info
                    else:
                        peers[info.node_id] = info

                    state.consecutive_failures = 0
                    state.last_success_time = now
                    self._promotion_stats["successful_promotions"] += 1
                    success = True

                    logger.info(
                        f"[GossipPromotion] Successfully promoted {info.node_id} "
                        f"via {host}:{port}"
                    )

                    # Save to peer cache if available
                    save_cache = getattr(orchestrator, "_save_peer_to_cache", None)
                    if save_cache:
                        save_cache(
                            info.node_id,
                            host,
                            port,
                            state.tailscale_ip,
                        )

                    break  # Success, no need to try other addresses

            except asyncio.TimeoutError:
                logger.debug(
                    f"[GossipPromotion] Timeout connecting to {state.node_id} "
                    f"at {host}:{port}"
                )
            except (OSError, ConnectionError) as e:
                logger.debug(
                    f"[GossipPromotion] Connection error for {state.node_id} "
                    f"at {host}:{port}: {e}"
                )
            except Exception as e:
                # Catch-all for unexpected errors - log but don't crash
                logger.warning(
                    f"[GossipPromotion] Unexpected error for {state.node_id}: {e}"
                )

        if not success:
            state.consecutive_failures += 1
            self._promotion_stats["failed_promotions"] += 1

            if state.consecutive_failures >= self.config.max_consecutive_failures:
                logger.debug(
                    f"[GossipPromotion] {state.node_id} has {state.consecutive_failures} "
                    f"consecutive failures, backing off for "
                    f"{self.config.failure_backoff_seconds}s"
                )

    def get_promotion_stats(self) -> dict[str, Any]:
        """Get promotion statistics for observability."""
        return {
            **self._promotion_stats,
            "peer_states": {
                node_id: {
                    "consecutive_failures": state.consecutive_failures,
                    "total_attempts": state.total_attempts,
                    "last_attempt_time": state.last_attempt_time,
                    "last_success_time": state.last_success_time,
                }
                for node_id, state in self._peer_states.items()
            },
        }

    def clear_peer_state(self, node_id: str) -> None:
        """Clear connection state for a peer (e.g., when it's retired)."""
        if node_id in self._peer_states:
            del self._peer_states[node_id]
