"""
P2P relay transport implementation.

Tier 4 (RELAY): Route through P2P leader or other relay nodes.

Jan 2, 2026: Enhanced with per-relay health tracking and health-based selection
to avoid single relay bottleneck and cascade failures.

Jan 5, 2026: Added distributed relay load balancing via consistent hashing.
NAT-blocked nodes are now assigned preferred relays based on their node_id hash,
preventing all nodes from targeting the same relay (reduces single relay bottleneck
by ~50% per the plan).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import aiohttp
    from aiohttp import ClientTimeout
except ImportError:
    aiohttp = None  # type: ignore
    ClientTimeout = None  # type: ignore

from ..transport_cascade import BaseTransport, TransportResult, TransportTier

logger = logging.getLogger(__name__)


# =============================================================================
# Per-Relay Health Tracking (Jan 2, 2026)
# =============================================================================


@dataclass
class RelayHealth:
    """Health tracking for a single relay node.

    Jan 2, 2026: Added to enable health-based relay selection and avoid
    cascade failures when a relay becomes unhealthy.
    """

    node_id: str
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    total_latency_ms: float = 0.0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    last_error: str = ""

    # Circuit breaker settings
    # Jan 2026: Increased failure_threshold (3→4) and reduced recovery_timeout (60→45)
    failure_threshold: int = 4  # Failures before marking unhealthy (was 3)
    recovery_timeout: float = 45.0  # Seconds before retrying unhealthy relay (was 60)

    @property
    def is_healthy(self) -> bool:
        """Check if relay is considered healthy."""
        if self.consecutive_failures >= self.failure_threshold:
            # Check if recovery timeout has passed
            if self.last_failure_time > 0:
                elapsed = time.time() - self.last_failure_time
                if elapsed < self.recovery_timeout:
                    return False
        return True

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        total = self.successes + self.failures
        if total == 0:
            return 1.0  # Assume healthy if no data
        return self.successes / total

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.successes == 0:
            return 0.0
        return self.total_latency_ms / self.successes

    @property
    def health_score(self) -> float:
        """Calculate health score for relay selection (higher = better).

        Score factors:
        - Success rate (0-1) weighted by 0.6
        - Recency of success (0-1) weighted by 0.3
        - Low latency bonus (0-1) weighted by 0.1
        """
        # Success rate component
        success_component = self.success_rate * 0.6

        # Recency component (how recently did we succeed?)
        if self.last_success_time > 0:
            since_success = time.time() - self.last_success_time
            # Decay over 5 minutes
            recency = max(0.0, 1.0 - (since_success / 300.0))
        else:
            recency = 0.5  # Unknown - assume moderate
        recency_component = recency * 0.3

        # Latency component (lower is better, normalize to 100-2000ms range)
        if self.avg_latency_ms > 0:
            latency_score = max(0.0, 1.0 - (self.avg_latency_ms - 100) / 1900)
        else:
            latency_score = 0.5  # Unknown
        latency_component = latency_score * 0.1

        return success_component + recency_component + latency_component

    def record_success(self, latency_ms: float) -> None:
        """Record a successful relay request."""
        self.successes += 1
        self.consecutive_failures = 0
        self.total_latency_ms += latency_ms
        self.last_success_time = time.time()

    def record_failure(self, error: str) -> None:
        """Record a failed relay request."""
        self.failures += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self.last_error = error


class P2PRelayTransport(BaseTransport):
    """
    Relay transport through P2P leader or relay nodes.

    Tier 4 (RELAY): For nodes that can't be reached directly.
    The payload is sent to a relay node which forwards to the target.

    Jan 2, 2026: Enhanced with per-relay health tracking. Relays are sorted
    by health score before use, avoiding cascade failures from unhealthy relays.

    Jan 5, 2026: Added distributed relay load balancing. NAT-blocked nodes get
    a preferred relay based on consistent hash of their node_id. This distributes
    load across relays instead of all nodes targeting the healthiest one.
    """

    name = "p2p_relay"
    tier = TransportTier.TIER_4_RELAY

    # Jan 5, 2026: Priority boost for hash-assigned preferred relay
    # Higher value = stronger preference for assigned relay over health-only selection
    PREFERRED_RELAY_BOOST = float(os.environ.get("RINGRIFT_RELAY_BOOST", "0.3"))

    def __init__(
        self,
        relay_nodes: list[str] | None = None,
        port: int = 8770,
        timeout: float = 20.0,
        source_node_id: str | None = None,
    ):
        self._relay_nodes = relay_nodes or []
        self._port = port
        self._timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._leader_node: str | None = None
        # Per-relay health tracking (Jan 2, 2026)
        self._relay_health: dict[str, RelayHealth] = {}
        # Jan 5, 2026: Source node ID for distributed relay assignment
        self._source_node_id = source_node_id

    def set_leader_node(self, leader_node: str) -> None:
        """Set the current P2P leader for relay."""
        self._leader_node = leader_node

    def add_relay_node(self, node: str) -> None:
        """Add a relay node."""
        if node not in self._relay_nodes:
            self._relay_nodes.append(node)

    def set_source_node_id(self, source_node_id: str) -> None:
        """Set the source node ID for distributed relay assignment.

        Jan 5, 2026: Allows setting source node after initialization.
        Used by failover_integration to configure the transport with node_id.
        """
        self._source_node_id = source_node_id

    def _get_preferred_relay(self, candidates: list[str]) -> str | None:
        """Get the preferred relay for this source node via consistent hashing.

        Jan 5, 2026: Distributes NAT-blocked nodes across available relays.
        Uses SHA-256 hash of source_node_id to consistently assign relays.

        This prevents the "thundering herd" problem where all NAT-blocked nodes
        target the same (healthiest) relay, overloading it.

        Args:
            candidates: List of available relay nodes to choose from

        Returns:
            Preferred relay node_id, or None if no source_node_id configured
        """
        if not self._source_node_id or not candidates:
            return None

        # Use SHA-256 hash for consistent assignment
        # Same source node always gets the same relay (until relay list changes)
        hash_input = f"{self._source_node_id}:relay_assignment"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        # Use first 8 bytes as integer for index calculation
        hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")

        # Select relay based on hash (consistent hashing)
        relay_index = hash_int % len(candidates)
        return candidates[relay_index]

    def _get_or_create_relay_health(self, node_id: str) -> RelayHealth:
        """Get or create health tracking for a relay node."""
        if node_id not in self._relay_health:
            self._relay_health[node_id] = RelayHealth(node_id=node_id)
        return self._relay_health[node_id]

    def _get_sorted_relays(self, target: str) -> list[str]:
        """Get relay list sorted by health score with preferred relay boost.

        Jan 2, 2026: Replaces simple leader-first ordering with health-based
        sorting. Healthy relays with good success rates are tried first.

        Jan 5, 2026: Added preferred relay boost for distributed load balancing.
        Each NAT-blocked node gets a hash-assigned preferred relay that receives
        a score boost (default 0.3). This distributes load across relays while
        still respecting health metrics for failover.
        """
        # Build candidate list: leader + configured relays (excluding target)
        candidates = []
        if self._leader_node and self._leader_node != target:
            candidates.append(self._leader_node)
        candidates.extend([r for r in self._relay_nodes if r != target and r not in candidates])

        if not candidates:
            return []

        # Jan 5, 2026: Get preferred relay for this source node (distributed assignment)
        preferred_relay = self._get_preferred_relay(candidates)

        # Filter to healthy relays first
        healthy_relays = []
        unhealthy_relays = []

        for node_id in candidates:
            health = self._get_or_create_relay_health(node_id)
            base_score = health.health_score

            # Jan 5, 2026: Apply boost for preferred relay
            # This biases selection toward the hash-assigned relay while still
            # allowing health to override if preferred relay is unhealthy
            if node_id == preferred_relay:
                adjusted_score = min(1.0, base_score + self.PREFERRED_RELAY_BOOST)
            else:
                adjusted_score = base_score

            if health.is_healthy:
                healthy_relays.append((node_id, adjusted_score))
            else:
                # Keep unhealthy relays as fallback (may have recovered)
                unhealthy_relays.append((node_id, adjusted_score))

        # Sort healthy relays by score (descending), keep unhealthy as fallback
        healthy_relays.sort(key=lambda x: x[1], reverse=True)
        unhealthy_relays.sort(key=lambda x: x[1], reverse=True)

        result = [node_id for node_id, _ in healthy_relays] + [node_id for node_id, _ in unhealthy_relays]

        # Log relay selection for debugging (only if we have a preferred relay)
        if preferred_relay and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[RelayTransport] Source={self._source_node_id}, "
                f"Preferred={preferred_relay}, Order={result[:3]}..."
            )

        return result

    def get_relay_health_summary(self) -> dict[str, Any]:
        """Get summary of all relay health states.

        Returns dict with overall stats and per-relay details.

        Jan 5, 2026: Added source_node_id and preferred_relay for debugging
        distributed load balancing.
        """
        total = len(self._relay_health)
        healthy = sum(1 for h in self._relay_health.values() if h.is_healthy)

        # Jan 5, 2026: Include preferred relay for this source node
        candidates = list(self._relay_health.keys()) or self._relay_nodes
        preferred = self._get_preferred_relay(candidates) if candidates else None

        return {
            "total_relays": total,
            "healthy_relays": healthy,
            "unhealthy_relays": total - healthy,
            # Jan 5, 2026: Distributed load balancing info
            "source_node_id": self._source_node_id,
            "preferred_relay": preferred,
            "preferred_relay_boost": self.PREFERRED_RELAY_BOOST,
            "relays": {
                node_id: {
                    "is_healthy": h.is_healthy,
                    "health_score": round(h.health_score, 3),
                    "success_rate": round(h.success_rate, 3),
                    "avg_latency_ms": round(h.avg_latency_ms, 1),
                    "consecutive_failures": h.consecutive_failures,
                    "last_error": h.last_error,
                    "is_preferred": node_id == preferred,  # Jan 5, 2026
                }
                for node_id, h in self._relay_health.items()
            },
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send payload via relay.

        Jan 2, 2026: Enhanced to use health-based relay selection. Healthy relays
        with good success rates are tried first, reducing cascade failures.
        """
        if aiohttp is None:
            return self._make_result(
                success=False, latency_ms=0, error="aiohttp not installed"
            )

        # Get relays sorted by health (healthy first, best scores first)
        relay_list = self._get_sorted_relays(target)

        if not relay_list:
            return self._make_result(
                success=False,
                latency_ms=0,
                error="No relay nodes available",
            )

        # Try each relay in health order
        errors = []
        for relay in relay_list:
            result = await self._send_via_relay(relay, target, payload)
            if result.success:
                return result
            errors.append(f"{relay}: {result.error}")

        return self._make_result(
            success=False,
            latency_ms=0,
            error=f"All relays failed: {'; '.join(errors)}",
        )

    async def _send_via_relay(
        self, relay: str, target: str, payload: bytes
    ) -> TransportResult:
        """Send payload through a specific relay node.

        Jan 2, 2026: Records success/failure in per-relay health tracking.
        """
        # Relay endpoint expects target in header
        url = f"http://{relay}:{self._port}/relay/forward"
        start_time = time.time()
        health = self._get_or_create_relay_health(relay)

        try:
            session = await self._get_session()
            async with session.post(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Relay-Target": target,
                },
            ) as resp:
                latency_ms = (time.time() - start_time) * 1000
                response_data = await resp.read()

                if resp.status == 200:
                    health.record_success(latency_ms)
                    return self._make_result(
                        success=True,
                        latency_ms=latency_ms,
                        response=response_data,
                        relay_node=relay,
                    )
                else:
                    error_msg = f"Relay HTTP {resp.status}"
                    health.record_failure(error_msg)
                    return self._make_result(
                        success=False,
                        latency_ms=latency_ms,
                        error=error_msg,
                    )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            health.record_failure(error_msg)
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=error_msg,
            )

    async def is_available(self, target: str) -> bool:
        """Check if relay transport is available."""
        # Available if we have any relay nodes
        return bool(self._leader_node) or bool(self._relay_nodes)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
