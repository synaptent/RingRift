"""Peer Recovery Loop for P2P Orchestrator.

December 2025: Active peer recovery for cluster node availability.

Problem: Retired/dead peers only recover when they re-announce themselves.
With RETRY_RETIRED_NODE_INTERVAL=3600 (1 hour), recovery takes too long.

Solution: Actively probe retired peers at a configurable interval (default: 120s)
and emit NODE_RECOVERED event when they respond.

Usage:
    from scripts.p2p.loops import PeerRecoveryLoop

    recovery_loop = PeerRecoveryLoop(
        get_retired_peers=lambda: orchestrator.get_retired_peers(),
        probe_peer=lambda addr: orchestrator.probe_peer_health(addr),
        recover_peer=lambda peer: orchestrator.recover_peer(peer),
        emit_event=lambda event, data: orchestrator.emit_event(event, data),
    )
    await recovery_loop.run_forever()

Events:
    NODE_RECOVERED: Emitted when a retired/dead peer is successfully recovered
    NODE_PROBE_FAILED: Emitted when a probe fails (for metrics)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Protocol

from .base import BaseLoop

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PeerRecoveryConfig:
    """Configuration for peer recovery loop."""

    # Base interval between recovery cycles (seconds)
    # Dec 2025: Changed default from 3600 (1 hour) to 120 (2 minutes)
    # Jan 2026: Reduced from 120s to 45s for faster CB recovery (MTTR 210s → 90s)
    recovery_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_P2P_PEER_RECOVERY_INTERVAL", "45")
        )
    )

    # Maximum peers to probe per cycle (limit API load)
    # Jan 2026: Increased from 10 to 20 for faster cluster recovery after outages
    max_probes_per_cycle: int = 20

    # HTTP timeout for health probes (seconds)
    # January 5, 2026: Increased from 10.0 to 15.0 for slow providers (Lambda NAT, Vast.ai)
    # Provider-specific multipliers are applied via LoopTimeouts.get_provider_multiplier()
    probe_timeout_seconds: float = 15.0

    # Exponential backoff: max interval between probes for a single peer (seconds)
    # After repeated failures, we probe less frequently for that peer
    # Dec 2025: Reduced from 3600 (1 hour) to 600 (10 min) for faster cluster recovery
    max_backoff_interval: float = 600.0  # 10 minutes max

    # Backoff multiplier for repeated failures
    # Dec 2025: Reduced from 2.0 to 1.5, then to 1.2 for even gentler backoff.
    # With 1.2x: 120s → 144s → 173s → 207s → 249s → 299s → 358s → 430s → 516s → 600s (cap)
    # Reaches cap in ~9 iterations vs 4 with 1.5x, enabling faster peer recovery.
    backoff_multiplier: float = 1.2

    # Number of failures before applying backoff
    # Jan 2026: Changed back to 3 (was reduced to 1, caused peer thrashing)
    # Threshold of 1 was too aggressive - a single transient failure triggered backoff,
    # causing healthy peers to be deprioritized. 3 failures provides resilience
    # against transient network blips while still backing off for persistent issues.
    backoff_threshold: int = 3

    # Whether to emit events on recovery/failure
    emit_events: bool = True

    # Whether the loop is enabled
    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_P2P_PEER_RECOVERY_ENABLED", "true"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.recovery_interval_seconds <= 0:
            raise ValueError("recovery_interval_seconds must be > 0")
        if self.max_probes_per_cycle <= 0:
            raise ValueError("max_probes_per_cycle must be > 0")
        if self.probe_timeout_seconds <= 0:
            raise ValueError("probe_timeout_seconds must be > 0")
        if self.backoff_multiplier <= 1:
            raise ValueError("backoff_multiplier must be > 1")


# =============================================================================
# Type Protocols
# =============================================================================


class PeerInfo(Protocol):
    """Protocol for peer information."""

    @property
    def node_id(self) -> str:
        """Unique peer identifier."""
        ...

    @property
    def address(self) -> str:
        """Peer's HTTP address (e.g., 'http://10.0.0.1:8770')."""
        ...

    @property
    def last_seen(self) -> float:
        """Timestamp of last successful contact."""
        ...

    @property
    def state(self) -> str:
        """Peer state: 'alive', 'suspect', 'dead', 'retired'."""
        ...


# =============================================================================
# Recovery Loop
# =============================================================================


class PeerRecoveryLoop(BaseLoop):
    """Background loop that actively probes retired/dead peers for recovery.

    Key features:
    - Probes retired peers at configurable interval (default: 120s)
    - Tracks probe failures per peer with exponential backoff
    - Emits NODE_RECOVERED event on successful recovery
    - Respects circuit breaker state (skip peers with open circuit)
    - Limited probes per cycle to avoid API overload
    """

    def __init__(
        self,
        get_retired_peers: Callable[[], list[Any]],
        probe_peer: Callable[[str], Coroutine[Any, Any, bool]],
        recover_peer: Callable[[Any], Coroutine[Any, Any, bool]],
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        get_circuit_state: Callable[[str], str] | None = None,
        reset_circuit: Callable[[str], bool] | None = None,
        config: PeerRecoveryConfig | None = None,
    ):
        """Initialize peer recovery loop.

        Args:
            get_retired_peers: Callback returning list of retired/dead peers
            probe_peer: Async callback to probe peer health by address.
                        Returns True if peer is healthy.
            recover_peer: Async callback to recover a peer (update state, reconnect).
                         Returns True if recovery succeeded.
            emit_event: Optional callback to emit events (event_name, event_data)
            get_circuit_state: Optional callback to get circuit breaker state for peer.
                               Returns 'OPEN', 'HALF_OPEN', or 'CLOSED'.
            reset_circuit: Optional callback to reset circuit breaker for a peer.
                          Sprint 12 Session 8: Enables proactive circuit recovery.
            config: Recovery configuration
        """
        self.config = config or PeerRecoveryConfig()
        super().__init__(
            name="peer_recovery",
            interval=self.config.recovery_interval_seconds,
            enabled=self.config.enabled,
        )

        # Callbacks
        self._get_retired_peers = get_retired_peers
        self._probe_peer = probe_peer
        self._recover_peer = recover_peer
        self._emit_event = emit_event
        self._get_circuit_state = get_circuit_state
        self._reset_circuit = reset_circuit

        # Per-peer failure tracking for backoff
        self._peer_failures: dict[str, int] = {}  # node_id -> consecutive failures
        self._peer_last_probe: dict[str, float] = {}  # node_id -> last probe timestamp
        self._peer_next_probe: dict[str, float] = {}  # node_id -> next allowed probe time

        # Statistics
        self._stats_probes_sent = 0
        self._stats_recoveries = 0
        self._stats_probe_failures = 0

    async def _run_once(self) -> None:
        """Execute one recovery cycle."""
        if not self.config.enabled:
            return

        # Get retired/dead peers
        peers = self._get_retired_peers()
        if not peers:
            logger.debug("[PeerRecovery] No retired peers to probe")
            return

        # Filter peers that are ready to be probed (respecting backoff)
        now = time.time()
        ready_peers = []
        circuit_broken_peers = []  # Sprint 12 Session 8: Track CB-open peers separately

        for peer in peers:
            node_id = self._get_peer_id(peer)

            # Check circuit breaker state
            if self._get_circuit_state:
                circuit_state = self._get_circuit_state(node_id)
                if circuit_state == "OPEN":
                    # Sprint 12 Session 8: Don't skip - try proactive recovery
                    if self._reset_circuit:
                        circuit_broken_peers.append(peer)
                    else:
                        logger.debug(f"[PeerRecovery] Skipping {node_id}: circuit OPEN")
                    continue

            # Check backoff timer
            next_probe = self._peer_next_probe.get(node_id, 0)
            if now < next_probe:
                remaining = next_probe - now
                logger.debug(
                    f"[PeerRecovery] Skipping {node_id}: backoff ({remaining:.0f}s remaining)"
                )
                continue

            ready_peers.append(peer)

        # Sprint 12 Session 8: Try proactive circuit recovery first
        # This reduces mean recovery time from 180s to 5-15s
        await self._try_proactive_circuit_recovery(circuit_broken_peers, now)

        if not ready_peers:
            logger.debug("[PeerRecovery] All retired peers in backoff")
            return

        # Limit probes per cycle
        probes_this_cycle = ready_peers[: self.config.max_probes_per_cycle]
        logger.info(
            f"[PeerRecovery] Probing {len(probes_this_cycle)}/{len(peers)} retired peers"
        )

        # Probe each peer
        recovered_count = 0
        for peer in probes_this_cycle:
            node_id = self._get_peer_id(peer)
            address = self._get_peer_address(peer)

            try:
                self._stats_probes_sent += 1
                self._peer_last_probe[node_id] = now

                # Probe with timeout
                is_healthy = await asyncio.wait_for(
                    self._probe_peer(address),
                    timeout=self.config.probe_timeout_seconds,
                )

                if is_healthy:
                    # Attempt recovery
                    recovered = await self._recover_peer(peer)
                    if recovered:
                        recovered_count += 1
                        self._stats_recoveries += 1
                        self._peer_failures.pop(node_id, None)
                        self._peer_next_probe.pop(node_id, None)
                        logger.info(f"[PeerRecovery] Recovered peer: {node_id}")

                        if self._emit_event and self.config.emit_events:
                            self._emit_event(
                                "NODE_RECOVERED",
                                {
                                    "node_id": node_id,
                                    "address": address,
                                    "recovery_source": "peer_recovery_loop",
                                    "timestamp": now,
                                },
                            )
                    else:
                        self._record_probe_failure(node_id, "recovery_failed")
                else:
                    self._record_probe_failure(node_id, "probe_unhealthy")

            except asyncio.TimeoutError:
                self._record_probe_failure(node_id, "timeout")
            except Exception as e:
                self._record_probe_failure(node_id, f"error: {e}")
                logger.debug(f"[PeerRecovery] Probe failed for {node_id}: {e}")

        if recovered_count > 0:
            logger.info(f"[PeerRecovery] Recovered {recovered_count} peers this cycle")

    def _record_probe_failure(self, node_id: str, reason: str) -> None:
        """Record a probe failure and apply backoff if needed."""
        self._stats_probe_failures += 1
        failures = self._peer_failures.get(node_id, 0) + 1
        self._peer_failures[node_id] = failures

        # Apply exponential backoff after threshold
        if failures >= self.config.backoff_threshold:
            backoff_count = failures - self.config.backoff_threshold + 1
            backoff = self.config.recovery_interval_seconds * (
                self.config.backoff_multiplier ** backoff_count
            )
            backoff = min(backoff, self.config.max_backoff_interval)
            self._peer_next_probe[node_id] = time.time() + backoff
            logger.debug(
                f"[PeerRecovery] Backoff for {node_id}: {backoff:.0f}s "
                f"(failures: {failures}, reason: {reason})"
            )

        if self._emit_event and self.config.emit_events:
            self._emit_event(
                "NODE_PROBE_FAILED",
                {
                    "node_id": node_id,
                    "reason": reason,
                    "consecutive_failures": failures,
                    "timestamp": time.time(),
                },
            )

    def _get_peer_id(self, peer: Any) -> str:
        """Extract node_id from peer object."""
        if hasattr(peer, "node_id"):
            return peer.node_id
        if isinstance(peer, dict):
            return peer.get("node_id", str(peer))
        return str(peer)

    def _get_peer_address(self, peer: Any) -> str:
        """Extract address from peer object."""
        if hasattr(peer, "address"):
            return peer.address
        if isinstance(peer, dict):
            return peer.get("address", "")
        return ""

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics."""
        return {
            "probes_sent": self._stats_probes_sent,
            "recoveries": self._stats_recoveries,
            "probe_failures": self._stats_probe_failures,
            "peers_in_backoff": len(self._peer_next_probe),
            "total_tracked_failures": sum(self._peer_failures.values()),
        }

    def reset_peer_backoff(self, node_id: str) -> None:
        """Reset backoff state for a specific peer.

        Call this when a peer comes back online through other means
        (e.g., re-announced itself via gossip).
        """
        self._peer_failures.pop(node_id, None)
        self._peer_next_probe.pop(node_id, None)
        self._peer_last_probe.pop(node_id, None)
        logger.debug(f"[PeerRecovery] Reset backoff for {node_id}")

    def reset_all_backoffs(self) -> None:
        """Reset backoff state for all peers."""
        self._peer_failures.clear()
        self._peer_next_probe.clear()
        self._peer_last_probe.clear()
        logger.info("[PeerRecovery] Reset all peer backoffs")

    def get_peer_backoff_info(self, node_id: str) -> dict[str, Any]:
        """Get backoff information for a specific peer."""
        now = time.time()
        next_probe = self._peer_next_probe.get(node_id, 0)
        return {
            "node_id": node_id,
            "consecutive_failures": self._peer_failures.get(node_id, 0),
            "last_probe": self._peer_last_probe.get(node_id, 0),
            "next_probe_in": max(0, next_probe - now) if next_probe else 0,
            "in_backoff": now < next_probe if next_probe else False,
        }

    async def _try_proactive_circuit_recovery(
        self, circuit_broken_peers: list[Any], now: float
    ) -> None:
        """Proactively probe circuit-broken peers and reset CB if healthy.

        Sprint 12 Session 8: Reduces mean recovery time from 180s (CB default timeout)
        to 5-15s by actively probing peers with open circuit breakers.

        When a peer recovers (e.g., SSH node comes back online), the circuit stays
        open until the CB timeout expires. This wastes recovery time. By proactively
        probing and resetting the circuit, we can restore workload distribution faster.

        Args:
            circuit_broken_peers: List of peers with OPEN circuit breakers
            now: Current timestamp for logging
        """
        if not circuit_broken_peers or not self._reset_circuit:
            return

        # Use a shorter timeout for proactive recovery probes
        quick_timeout = min(5.0, self.config.probe_timeout_seconds)
        recovered_count = 0

        for peer in circuit_broken_peers[:self.config.max_probes_per_cycle]:
            node_id = self._get_peer_id(peer)
            address = self._get_peer_address(peer)

            try:
                self._stats_probes_sent += 1
                self._peer_last_probe[node_id] = now

                # Quick probe with shorter timeout
                is_healthy = await asyncio.wait_for(
                    self._probe_peer(address),
                    timeout=quick_timeout,
                )

                if is_healthy:
                    # Reset the circuit breaker to allow immediate traffic
                    cb_reset = self._reset_circuit(node_id)
                    if cb_reset:
                        recovered_count += 1
                        self._stats_recoveries += 1
                        self._peer_failures.pop(node_id, None)
                        self._peer_next_probe.pop(node_id, None)
                        logger.info(
                            f"[PeerRecovery] Proactive CB reset for {node_id} "
                            f"(was circuit-broken, now healthy)"
                        )

                        if self._emit_event and self.config.emit_events:
                            self._emit_event(
                                "CIRCUIT_RESET",
                                {
                                    "node_id": node_id,
                                    "address": address,
                                    "recovery_source": "proactive_cb_recovery",
                                    "timestamp": now,
                                },
                            )

                        # Also attempt full peer recovery
                        try:
                            await self._recover_peer(peer)
                        except Exception as e:
                            logger.debug(
                                f"[PeerRecovery] Peer recovery after CB reset "
                                f"failed for {node_id}: {e}"
                            )
                else:
                    logger.debug(
                        f"[PeerRecovery] Proactive probe unhealthy for {node_id}"
                    )

            except asyncio.TimeoutError:
                logger.debug(
                    f"[PeerRecovery] Proactive probe timeout for {node_id} "
                    f"({quick_timeout}s)"
                )
            except Exception as e:
                logger.debug(
                    f"[PeerRecovery] Proactive probe error for {node_id}: {e}"
                )

        if recovered_count > 0:
            logger.info(
                f"[PeerRecovery] Proactive CB recovery: {recovered_count} "
                f"circuits reset this cycle"
            )

    def health_check(self) -> Any:
        """Check loop health with peer recovery-specific status.

        Jan 2026: Added for DaemonManager integration.
        Reports recovery success rate and backoff status.

        Returns:
            HealthCheckResult with peer recovery status
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            stats = self.get_recovery_stats()
            return {
                "healthy": self.running,
                "status": "running" if self.running else "stopped",
                "message": f"PeerRecoveryLoop {'running' if self.running else 'stopped'}",
                "details": stats,
            }

        # Not running
        if not self.running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="PeerRecoveryLoop is stopped",
                details={"running": False},
            )

        # Not enabled
        if not self.config.enabled:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="PeerRecoveryLoop is disabled via config",
                details={"enabled": False, "running": self.running},
            )

        # Get stats
        stats = self.get_recovery_stats()
        probes_sent = stats.get("probes_sent", 0)
        recoveries = stats.get("recoveries", 0)
        probe_failures = stats.get("probe_failures", 0)
        peers_in_backoff = stats.get("peers_in_backoff", 0)

        # Calculate recovery rate
        total_attempts = probes_sent
        recovery_rate = (recoveries / total_attempts * 100) if total_attempts > 0 else 0

        # Check if too many peers stuck in backoff (degraded)
        if peers_in_backoff > 20:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"PeerRecoveryLoop has {peers_in_backoff} peers in backoff",
                details={
                    "probes_sent": probes_sent,
                    "recoveries": recoveries,
                    "probe_failures": probe_failures,
                    "peers_in_backoff": peers_in_backoff,
                    "recovery_rate_pct": round(recovery_rate, 1),
                },
            )

        # Healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"PeerRecoveryLoop healthy ({recoveries} recovered, {recovery_rate:.1f}% rate)",
            details={
                "probes_sent": probes_sent,
                "recoveries": recoveries,
                "probe_failures": probe_failures,
                "peers_in_backoff": peers_in_backoff,
                "recovery_rate_pct": round(recovery_rate, 1),
                "interval_seconds": self.config.recovery_interval_seconds,
            },
        )
