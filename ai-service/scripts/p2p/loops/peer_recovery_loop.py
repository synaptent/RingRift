"""Peer Recovery Loop for P2P Orchestrator.

December 2025: Active peer recovery for cluster node availability.

Problem: Retired/dead peers only recover when they re-announce themselves.
With RETRY_RETIRED_NODE_INTERVAL=3600 (1 hour), recovery takes too long.

Solution: Actively probe retired peers at a configurable interval (default: 120s)
and emit NODE_RECOVERED event when they respond.

January 2026 (Phase 7.2): Provider-aware probe timeouts.
Different cloud providers have different network latency characteristics:
- Vast.ai: Consumer networks with high variance (35s timeout)
- Lambda GH200: NAT-blocked nodes with relay latency (25s timeout)
- Nebius/Hetzner: Stable infrastructure with consistent latency (20s timeout)

This reduces false positive rate from 22-30% to <5% for slow providers.

Usage:
    from scripts.p2p.loops import PeerRecoveryLoop

    recovery_loop = PeerRecoveryLoop(
        get_retired_peers=lambda: orchestrator.get_retired_peers(),
        probe_peer=lambda addr: orchestrator.probe_peer_health(addr),
        recover_peer=lambda peer: orchestrator.recover_peer(peer),
        emit_event=lambda event, data: orchestrator.emit_event(event, data),
    )
    await recovery_loop.run_forever()

    # Get provider-specific timeout for a node
    from scripts.p2p.loops.peer_recovery_loop import get_provider_probe_timeout
    timeout = get_provider_probe_timeout("vast-29031159")  # Returns 35.0

Events:
    NODE_RECOVERED: Emitted when a retired/dead peer is successfully recovered
    NODE_PROBE_FAILED: Emitted when a probe fails (for metrics)
    CIRCUIT_RESET: Emitted on proactive circuit breaker recovery
"""
from __future__ import annotations

__all__ = [
    "PeerRecoveryLoop",
    "PeerRecoveryConfig",
    "PeerInfo",
    # Phase 7.2: Provider-aware timeouts
    "PROVIDER_PROBE_TIMEOUTS",
    "get_provider_probe_timeout",
    # Jan 7, 2026: TCP connectivity check for faster backoff reset
    "check_tcp_connectivity",
]

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Protocol
from urllib.parse import urlparse

from .base import BaseLoop

# Jan 16, 2026: Use centralized provider timeout configuration
from app.config.provider_timeouts import ProviderTimeouts

logger = logging.getLogger(__name__)


# =============================================================================
# Provider-Specific Timeouts (Phase 7.2 - Jan 5, 2026)
# =============================================================================
# Jan 16, 2026: Now delegated to centralized ProviderTimeouts config.
# These aliases are kept for backward compatibility with existing callers.

# Backward-compatible aliases using centralized config
PROVIDER_PROBE_TIMEOUTS: dict[str, float] = ProviderTimeouts.PROBE_TIMEOUTS

# Default timeout for unknown providers (conservative)
DEFAULT_PROVIDER_TIMEOUT: float = ProviderTimeouts.DEFAULT_PROBE_TIMEOUT


def _extract_provider_from_node_id(node_id: str) -> str:
    """Extract provider name from node_id prefix.

    Jan 16, 2026: Delegated to centralized ProviderTimeouts.extract_provider().
    """
    return ProviderTimeouts.extract_provider(node_id)


def get_provider_probe_timeout(node_id: str, base_timeout: float = 20.0) -> float:
    """Get provider-specific probe timeout for a node.

    Phase 7.2 (Jan 5, 2026): Provider-aware timeouts to reduce false positives.
    Different providers have different network characteristics:
    - Vast.ai: Consumer networks with 25-30s response times under load
    - Lambda GH200: NAT-blocked, relay adds ~5s latency
    - Nebius/Hetzner: Stable cloud/bare metal with consistent <20s responses

    Jan 16, 2026: Delegated to centralized ProviderTimeouts.get_probe_timeout().

    Args:
        node_id: Node identifier to look up provider for
        base_timeout: Base timeout to use if provider not found (ignored - uses centralized config)

    Returns:
        Provider-appropriate probe timeout in seconds
    """
    return ProviderTimeouts.get_probe_timeout(node_id)


def check_tcp_connectivity(address: str, timeout: float = 2.0) -> bool:
    """Check raw TCP connectivity to an address.

    Jan 7, 2026: Module-level convenience function for external callers.
    Useful for quick connectivity checks before attempting full HTTP probes.

    Args:
        address: Peer address (e.g., 'http://10.0.0.1:8770' or '10.0.0.1:8770')
        timeout: TCP connect timeout in seconds (default: 2.0s)

    Returns:
        True if TCP connection succeeded, False otherwise
    """
    try:
        # Parse address to extract host and port
        if address.startswith("http://") or address.startswith("https://"):
            parsed = urlparse(address)
            host = parsed.hostname
            port = parsed.port or 8770
        else:
            # Assume host:port format
            if ":" in address:
                host, port_str = address.rsplit(":", 1)
                port = int(port_str)
            else:
                host = address
                port = 8770

        if not host:
            return False

        # Create socket and attempt connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0

    except (socket.error, socket.timeout, ValueError, OSError):
        return False


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PeerRecoveryConfig:
    """Configuration for peer recovery loop."""

    # Base interval between recovery cycles (seconds)
    # Dec 2025: Changed default from 3600 (1 hour) to 120 (2 minutes)
    # Jan 2026: Reduced from 120s to 45s for faster CB recovery (MTTR 210s → 90s)
    # January 5, 2026 (Phase 7.10): Reduced from 45s to 15s for faster initial retries.
    # Combined with 1.5x backoff multiplier: 15→22→34→51→76→114s (3.5 min to 114s vs 10+ min)
    # Dead nodes are now retried in 3-4 min instead of 10+ min total backoff time.
    recovery_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_P2P_PEER_RECOVERY_INTERVAL", "15")
        )
    )

    # Maximum peers to probe per cycle (limit API load)
    # Jan 2026: Increased from 10 to 20 for faster cluster recovery after outages
    max_probes_per_cycle: int = 20

    # HTTP timeout for health probes (seconds)
    # January 5, 2026: Increased from 10.0 to 15.0 for slow providers (Lambda NAT, Vast.ai)
    # January 5, 2026 (Phase 3): Further increased to 20.0 to reduce 22% false positive rate
    # Provider-specific multipliers are applied via LoopTimeouts.get_provider_multiplier()
    probe_timeout_seconds: float = 20.0

    # Exponential backoff: max interval between probes for a single peer (seconds)
    # After repeated failures, we probe less frequently for that peer
    # Dec 2025: Reduced from 3600 (1 hour) to 600 (10 min) for faster cluster recovery
    max_backoff_interval: float = 600.0  # 10 minutes max

    # Backoff multiplier for repeated failures
    # Dec 2025: Reduced from 2.0 to 1.5, then to 1.2 for even gentler backoff.
    # January 5, 2026 (Phase 7.10): Increased back to 1.5 combined with shorter base interval.
    # With 1.5x and 15s base: 15→22→34→51→76→114→171→256→384→576→600s (cap)
    # Reaches cap faster but starts retrying sooner after failures.
    # Dead nodes retried in 3-4 min vs 10+ min with old 1.2x multiplier.
    backoff_multiplier: float = 1.5

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

    # ==========================================================================
    # Burst Recovery Mode Configuration (January 2026 - P2P Stability Plan Phase 1.3)
    # ==========================================================================
    # When a significant fraction of peers are retired (mass failure event like
    # network partition recovery or provider outage), accelerate recovery by:
    # - Increasing max probes per cycle (20 → 50)
    # - Reducing recovery interval (15s → 5s)
    # This helps the cluster converge faster after mass disconnection events.

    # Threshold for activating burst mode (fraction of total peers that are retired)
    # When retired_peers / total_peers > threshold, burst mode activates
    burst_mode_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_P2P_BURST_MODE_THRESHOLD", "0.30")
        )
    )

    # Max probes per cycle during burst mode (increased from normal 20)
    burst_mode_max_probes: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_P2P_BURST_MODE_MAX_PROBES", "50")
        )
    )

    # Recovery interval during burst mode (reduced from normal 15s)
    burst_mode_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_P2P_BURST_MODE_INTERVAL", "5")
        )
    )

    # Whether burst mode is enabled
    burst_mode_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_P2P_BURST_MODE_ENABLED", "true"
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
        # Burst mode validation
        if not (0.0 < self.burst_mode_threshold <= 1.0):
            raise ValueError("burst_mode_threshold must be between 0 and 1")
        if self.burst_mode_max_probes <= 0:
            raise ValueError("burst_mode_max_probes must be > 0")
        if self.burst_mode_interval_seconds <= 0:
            raise ValueError("burst_mode_interval_seconds must be > 0")


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
    - Burst recovery mode for mass failure events (Jan 2026)
    """

    def __init__(
        self,
        get_retired_peers: Callable[[], list[Any]],
        probe_peer: Callable[[str], Coroutine[Any, Any, bool]],
        recover_peer: Callable[[Any], Coroutine[Any, Any, bool]],
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        get_circuit_state: Callable[[str], str] | None = None,
        reset_circuit: Callable[[str], bool] | None = None,
        get_total_peer_count: Callable[[], int] | None = None,
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
            get_total_peer_count: Optional callback to get total known peer count.
                                  Required for burst mode detection. If not provided,
                                  burst mode will use retired peer count as estimate.
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
        self._get_total_peer_count = get_total_peer_count

        # Per-peer failure tracking for backoff
        self._peer_failures: dict[str, int] = {}  # node_id -> consecutive failures
        self._peer_last_probe: dict[str, float] = {}  # node_id -> last probe timestamp
        self._peer_next_probe: dict[str, float] = {}  # node_id -> next allowed probe time

        # Statistics
        self._stats_probes_sent = 0
        self._stats_recoveries = 0
        self._stats_probe_failures = 0

        # Burst mode state (January 2026 - P2P Stability Plan Phase 1.3)
        self._burst_mode_active = False
        self._burst_mode_activated_at: float | None = None
        self._stats_burst_mode_activations = 0

    def _check_burst_mode(self, retired_count: int) -> bool:
        """Check if burst recovery mode should be active.

        January 2026 - P2P Stability Plan Phase 1.3:
        Activates burst mode when >30% of peers are retired, indicating a mass
        failure event (network partition recovery, provider outage, etc.).

        Args:
            retired_count: Number of retired/dead peers

        Returns:
            True if burst mode should be active
        """
        if not self.config.burst_mode_enabled:
            return False

        # Get total peer count
        if self._get_total_peer_count:
            total_peers = self._get_total_peer_count()
        else:
            # Fallback: estimate total as retired + some baseline (20 nodes typical)
            # This is imperfect but allows burst mode without the callback
            total_peers = max(retired_count + 10, 20)

        if total_peers <= 0:
            return False

        retired_ratio = retired_count / total_peers
        should_activate = retired_ratio > self.config.burst_mode_threshold

        # Track state transitions
        was_active = self._burst_mode_active
        self._burst_mode_active = should_activate

        if should_activate and not was_active:
            # Just activated burst mode
            self._burst_mode_activated_at = time.time()
            self._stats_burst_mode_activations += 1
            logger.warning(
                f"[PeerRecovery] BURST MODE ACTIVATED: {retired_count}/{total_peers} "
                f"peers retired ({retired_ratio:.1%} > {self.config.burst_mode_threshold:.0%} threshold). "
                f"Accelerating recovery: interval {self.config.burst_mode_interval_seconds}s, "
                f"max_probes {self.config.burst_mode_max_probes}"
            )

            if self._emit_event and self.config.emit_events:
                self._emit_event(
                    "BURST_RECOVERY_MODE_ACTIVATED",
                    {
                        "retired_count": retired_count,
                        "total_peers": total_peers,
                        "retired_ratio": retired_ratio,
                        "threshold": self.config.burst_mode_threshold,
                        "burst_interval": self.config.burst_mode_interval_seconds,
                        "burst_max_probes": self.config.burst_mode_max_probes,
                        "timestamp": time.time(),
                    },
                )

        elif not should_activate and was_active:
            # Just deactivated burst mode
            duration = time.time() - (self._burst_mode_activated_at or time.time())
            logger.info(
                f"[PeerRecovery] Burst mode deactivated after {duration:.1f}s. "
                f"Retired ratio {retired_ratio:.1%} <= {self.config.burst_mode_threshold:.0%} threshold"
            )
            self._burst_mode_activated_at = None

            if self._emit_event and self.config.emit_events:
                self._emit_event(
                    "BURST_RECOVERY_MODE_DEACTIVATED",
                    {
                        "retired_count": retired_count,
                        "total_peers": total_peers,
                        "retired_ratio": retired_ratio,
                        "duration_seconds": duration,
                        "timestamp": time.time(),
                    },
                )

        return should_activate

    def _get_effective_params(self, burst_mode: bool) -> tuple[int, float]:
        """Get effective max_probes and interval based on mode.

        Args:
            burst_mode: Whether burst mode is active

        Returns:
            Tuple of (max_probes_per_cycle, recovery_interval)
        """
        if burst_mode:
            return (
                self.config.burst_mode_max_probes,
                self.config.burst_mode_interval_seconds,
            )
        return (
            self.config.max_probes_per_cycle,
            self.config.recovery_interval_seconds,
        )

    async def _run_once(self) -> None:
        """Execute one recovery cycle."""
        if not self.config.enabled:
            return

        # Get retired/dead peers
        peers = self._get_retired_peers()
        if not peers:
            logger.debug("[PeerRecovery] No retired peers to probe")
            # Deactivate burst mode if no retired peers
            if self._burst_mode_active:
                self._check_burst_mode(0)
            return

        # Check for burst mode (mass failure detection)
        burst_mode = self._check_burst_mode(len(peers))
        max_probes, _ = self._get_effective_params(burst_mode)

        # Filter peers that are ready to be probed (respecting backoff)
        now = time.time()
        ready_peers = []
        peers_in_backoff = []  # Jan 7, 2026: Track peers in backoff for TCP check
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
                # Jan 7, 2026: Track for TCP connectivity check
                peers_in_backoff.append(peer)
                continue

            ready_peers.append(peer)

        # Jan 7, 2026: Check TCP connectivity for peers in backoff
        # This reduces recovery time by detecting reachable hosts faster (2s vs 20s)
        if peers_in_backoff:
            await self._reset_backoff_for_tcp_reachable_peers(peers_in_backoff, now)

        # Sprint 12 Session 8: Try proactive circuit recovery first
        # This reduces mean recovery time from 180s to 5-15s
        await self._try_proactive_circuit_recovery(circuit_broken_peers, now)

        if not ready_peers:
            logger.debug("[PeerRecovery] All retired peers in backoff")
            return

        # Limit probes per cycle (use burst mode max if active)
        probes_this_cycle = ready_peers[:max_probes]
        mode_str = " [BURST]" if burst_mode else ""
        logger.info(
            f"[PeerRecovery]{mode_str} Probing {len(probes_this_cycle)}/{len(peers)} retired peers"
        )

        # Probe each peer
        recovered_count = 0
        for peer in probes_this_cycle:
            node_id = self._get_peer_id(peer)
            address = self._get_peer_address(peer)

            try:
                self._stats_probes_sent += 1
                self._peer_last_probe[node_id] = now

                # Phase 7.2 (Jan 5, 2026): Provider-aware probe timeout
                # Reduces false positives from 22-30% to <5% by allowing more
                # time for slow providers (Vast.ai, Lambda NAT-relayed nodes)
                probe_timeout = get_provider_probe_timeout(
                    node_id, self.config.probe_timeout_seconds
                )

                # Probe with provider-specific timeout
                is_healthy = await asyncio.wait_for(
                    self._probe_peer(address),
                    timeout=probe_timeout,
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
        stats = {
            "probes_sent": self._stats_probes_sent,
            "recoveries": self._stats_recoveries,
            "probe_failures": self._stats_probe_failures,
            "peers_in_backoff": len(self._peer_next_probe),
            "total_tracked_failures": sum(self._peer_failures.values()),
            # Burst mode stats (January 2026)
            "burst_mode_active": self._burst_mode_active,
            "burst_mode_activations": self._stats_burst_mode_activations,
        }
        if self._burst_mode_activated_at:
            stats["burst_mode_duration"] = time.time() - self._burst_mode_activated_at
        return stats

    def is_burst_mode_active(self) -> bool:
        """Check if burst recovery mode is currently active.

        January 2026 - P2P Stability Plan Phase 1.3:
        Returns True when >30% of peers are retired and the loop is
        operating with accelerated recovery parameters.
        """
        return self._burst_mode_active

    def get_burst_mode_info(self) -> dict[str, Any]:
        """Get detailed burst mode status information.

        Returns:
            Dict with burst mode configuration and current state
        """
        info = {
            "enabled": self.config.burst_mode_enabled,
            "active": self._burst_mode_active,
            "threshold": self.config.burst_mode_threshold,
            "burst_interval_seconds": self.config.burst_mode_interval_seconds,
            "burst_max_probes": self.config.burst_mode_max_probes,
            "normal_interval_seconds": self.config.recovery_interval_seconds,
            "normal_max_probes": self.config.max_probes_per_cycle,
            "activations_total": self._stats_burst_mode_activations,
        }
        if self._burst_mode_activated_at:
            info["active_since"] = self._burst_mode_activated_at
            info["active_duration_seconds"] = time.time() - self._burst_mode_activated_at
        return info

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

    def on_quorum_restored(self, event: dict[str, Any] | None = None) -> None:
        """Handle quorum restoration by resetting all peer backoffs.

        Session 17.31 (Jan 5, 2026): Called when voter quorum is restored.
        This enables faster cluster convergence by allowing immediate
        re-probing of all peers that were in backoff during the outage.

        Args:
            event: Optional quorum_restored event data (for logging context)
        """
        online_voters = event.get("online_voters", "?") if event else "?"
        total_voters = event.get("total_voters", "?") if event else "?"
        logger.info(
            f"[PeerRecovery] Quorum restored ({online_voters}/{total_voters}) - "
            "resetting all peer backoffs for faster convergence"
        )
        self.reset_all_backoffs()

        # Emit event for observability
        if self._emit_event:
            self._emit_event(
                "PEER_BACKOFFS_RESET",
                {
                    "reason": "quorum_restored",
                    "online_voters": online_voters,
                    "total_voters": total_voters,
                    "peers_reset": len(self._peer_failures) + len(self._peer_next_probe),
                },
            )

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

    def _check_tcp_connectivity(
        self, address: str, timeout: float = 2.0
    ) -> bool:
        """Check raw TCP connectivity to a peer (faster than HTTP).

        Jan 7, 2026: Added for faster backoff reset detection.
        TCP connect is much faster than HTTP (2s vs 20s timeout), so we can
        detect recovered hosts more quickly and reset their backoff state.

        Args:
            address: Peer address (e.g., 'http://10.0.0.1:8770' or '10.0.0.1:8770')
            timeout: TCP connect timeout in seconds (default: 2.0s)

        Returns:
            True if TCP connection succeeded, False otherwise
        """
        try:
            # Parse address to extract host and port
            if address.startswith("http://") or address.startswith("https://"):
                parsed = urlparse(address)
                host = parsed.hostname
                port = parsed.port or 8770
            else:
                # Assume host:port format
                if ":" in address:
                    host, port_str = address.rsplit(":", 1)
                    port = int(port_str)
                else:
                    host = address
                    port = 8770

            if not host:
                return False

            # Create socket and attempt connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0

        except (socket.error, socket.timeout, ValueError, OSError):
            return False

    async def _check_tcp_connectivity_async(
        self, address: str, timeout: float = 2.0
    ) -> bool:
        """Async wrapper for TCP connectivity check.

        Runs the blocking socket operation in a thread pool to avoid
        blocking the event loop.

        Args:
            address: Peer address
            timeout: TCP connect timeout in seconds

        Returns:
            True if TCP connection succeeded, False otherwise
        """
        return await asyncio.to_thread(
            self._check_tcp_connectivity, address, timeout
        )

    async def _reset_backoff_for_tcp_reachable_peers(
        self, peers_in_backoff: list[Any], now: float
    ) -> int:
        """Check TCP connectivity for peers in backoff and reset if reachable.

        Jan 7, 2026: Reduces recovery time by detecting TCP-reachable hosts
        before the HTTP probe timeout. If a host responds to TCP connect,
        we reset its backoff so it will be probed on the next cycle.

        This is especially useful for nodes that are rebooting - they become
        TCP-reachable (port listening) before the HTTP endpoint is ready.

        Args:
            peers_in_backoff: List of peers currently in backoff state
            now: Current timestamp for logging

        Returns:
            Number of peers that had backoff reset
        """
        if not peers_in_backoff:
            return 0

        reset_count = 0

        # Check up to 10 peers per cycle to avoid overwhelming the network
        for peer in peers_in_backoff[:10]:
            node_id = self._get_peer_id(peer)
            address = self._get_peer_address(peer)

            if not address:
                continue

            try:
                # Quick TCP check (2s timeout)
                tcp_reachable = await self._check_tcp_connectivity_async(address, 2.0)

                if tcp_reachable:
                    # Reset backoff so peer will be HTTP-probed on next cycle
                    self.reset_peer_backoff(node_id)
                    reset_count += 1
                    logger.info(
                        f"[PeerRecovery] TCP alive - reset backoff for {node_id}"
                    )

                    if self._emit_event and self.config.emit_events:
                        self._emit_event(
                            "PEER_BACKOFF_RESET_TCP",
                            {
                                "node_id": node_id,
                                "address": address,
                                "reason": "tcp_connectivity_restored",
                                "timestamp": now,
                            },
                        )

                    # Jan 16, 2026: Also attempt full recovery (auto-unretire)
                    # This allows TCP-reachable peers to be recovered immediately
                    # instead of waiting for the next HTTP probe cycle.
                    try:
                        recovered = await self._recover_peer(peer)
                        if recovered:
                            logger.info(
                                f"[PeerRecovery] Auto-unretired TCP-reachable peer: {node_id}"
                            )
                            if self._emit_event and self.config.emit_events:
                                self._emit_event(
                                    "NODE_RECOVERED",
                                    {
                                        "node_id": node_id,
                                        "address": address,
                                        "recovery_method": "tcp_auto_unretire",
                                        "timestamp": now,
                                    },
                                )
                    except Exception as e:
                        # Recovery may fail if HTTP isn't ready yet (port listening but
                        # server still starting). That's OK - the backoff reset means
                        # we'll HTTP-probe on the next cycle.
                        logger.debug(
                            f"[PeerRecovery] Auto-unretire after TCP failed for {node_id}: "
                            f"{type(e).__name__}: {e}"
                        )

            except Exception as e:
                logger.debug(
                    f"[PeerRecovery] TCP check error for {node_id}: {e}"
                )

        if reset_count > 0:
            logger.info(
                f"[PeerRecovery] Reset backoff for {reset_count} TCP-reachable peers"
            )

        return reset_count

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
        # Session 17.31 (Jan 5, 2026): Reduced from 20 to 10 for earlier intervention
        if peers_in_backoff > 10:
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

        # Healthy - include burst mode status
        burst_info = ""
        if self._burst_mode_active:
            burst_info = " [BURST MODE]"

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"PeerRecoveryLoop healthy ({recoveries} recovered, {recovery_rate:.1f}% rate){burst_info}",
            details={
                "probes_sent": probes_sent,
                "recoveries": recoveries,
                "probe_failures": probe_failures,
                "peers_in_backoff": peers_in_backoff,
                "recovery_rate_pct": round(recovery_rate, 1),
                "interval_seconds": self.config.recovery_interval_seconds,
                # Burst mode info (January 2026)
                "burst_mode_active": self._burst_mode_active,
                "burst_mode_activations": self._stats_burst_mode_activations,
            },
        )
