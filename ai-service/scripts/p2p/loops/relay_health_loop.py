"""Relay Health Monitoring Loop for P2P Orchestrator.

January 5, 2026 (Task 8.5): Monitor DERP relay health for NAT-traversal nodes.

This loop provides proactive monitoring of relay nodes that NAT-blocked nodes
depend on for connectivity. When relays become degraded or unreachable, the loop
emits events to trigger failover before complete connectivity loss.

Features:
- Periodic probing of all active relay nodes
- Latency tracking with trend detection
- Automatic relay failover recommendation
- Health events: RELAY_DEGRADED, RELAY_HEALTHY, RELAY_FAILOVER

Usage:
    from scripts.p2p.loops.relay_health_loop import RelayHealthLoop

    relay_loop = RelayHealthLoop(
        get_relay_nodes=lambda: ["hetzner-cpu1", "nebius-backbone-1"],
        get_node_info=lambda node_id: {"tailscale_ip": "100.x.x.x"},
        get_nat_blocked_peers=lambda: {"lambda-gh200-1": "hetzner-cpu1"},
        trigger_relay_failover=async_failover_callback,
    )
    await relay_loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from .base import BaseLoop

logger = logging.getLogger(__name__)


@dataclass
class RelayHealthConfig:
    """Configuration for relay health monitoring."""

    check_interval_seconds: float = 30.0  # How often to check relays
    probe_timeout_seconds: float = 5.0  # Timeout for individual probe
    latency_warning_threshold_ms: float = 200.0  # Warn if latency exceeds this
    latency_critical_threshold_ms: float = 500.0  # Critical if latency exceeds this
    # Jan 2026: Reduced thresholds for preemptive failover
    consecutive_failures_for_degraded: int = 2  # Mark degraded after N failures (was 3)
    consecutive_failures_for_failover: int = 3  # Trigger failover after N failures (was 5)
    latency_history_size: int = 10  # Rolling window for latency trending

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.probe_timeout_seconds <= 0:
            raise ValueError("probe_timeout_seconds must be > 0")
        if self.latency_warning_threshold_ms <= 0:
            raise ValueError("latency_warning_threshold_ms must be > 0")


@dataclass
class RelayHealthStatus:
    """Health status for a single relay node."""

    node_id: str
    healthy: bool = True
    latency_ms: float = 0.0
    consecutive_failures: int = 0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    latency_history: deque = field(default_factory=lambda: deque(maxlen=10))
    degraded_event_emitted: bool = False
    dependent_peers: set = field(default_factory=set)

    def record_success(self, latency_ms: float) -> None:
        """Record a successful probe."""
        self.healthy = True
        self.latency_ms = latency_ms
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.latency_history.append(latency_ms)

    def record_failure(self) -> None:
        """Record a failed probe."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= 3:
            self.healthy = False

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency from history."""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)

    @property
    def is_latency_trending_up(self) -> bool:
        """Check if latency is trending upward."""
        if len(self.latency_history) < 6:
            return False
        history = list(self.latency_history)
        mid = len(history) // 2
        older_avg = sum(history[:mid]) / mid
        recent_avg = sum(history[mid:]) / (len(history) - mid)
        return recent_avg > older_avg * 1.5  # 50% increase is concerning


class RelayHealthLoop(BaseLoop):
    """Monitor health of relay nodes for NAT-traversal.

    Provides early warning when relay nodes become degraded, allowing
    proactive failover before NAT-blocked nodes lose connectivity.

    January 5, 2026: Created as part of Task 8.5 (Add relay health monitoring).
    """

    def __init__(
        self,
        get_relay_nodes: Callable[[], list[str]],
        get_node_info: Callable[[str], dict[str, Any] | None],
        get_nat_blocked_peers: Callable[[], dict[str, str]],
        trigger_relay_failover: Callable[[str, str, str], Coroutine[Any, Any, bool]] | None = None,
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        config: RelayHealthConfig | None = None,
    ):
        """Initialize relay health loop.

        Args:
            get_relay_nodes: Returns list of active relay node IDs
            get_node_info: Returns node info dict for a node ID (with tailscale_ip, etc.)
            get_nat_blocked_peers: Returns dict of {peer_id: primary_relay_id} for NAT-blocked peers
            trigger_relay_failover: Async callback to trigger failover (peer_id, old_relay, new_relay)
            emit_event: Callback to emit events (event_name, payload)
            config: Health monitoring configuration
        """
        self.config = config or RelayHealthConfig()
        super().__init__(
            name="relay_health",
            interval=self.config.check_interval_seconds,
        )

        self._get_relay_nodes = get_relay_nodes
        self._get_node_info = get_node_info
        self._get_nat_blocked_peers = get_nat_blocked_peers
        self._trigger_relay_failover = trigger_relay_failover
        self._emit_event = emit_event

        # Track health status per relay
        self._relay_health: dict[str, RelayHealthStatus] = {}

        # Statistics
        self._probes_sent = 0
        self._probes_succeeded = 0
        self._failovers_triggered = 0
        self._degraded_events_emitted = 0
        self._healthy_events_emitted = 0

    async def _on_start(self) -> None:
        """Log startup and initialize relay tracking."""
        relay_nodes = self._get_relay_nodes()
        logger.info(
            f"[RelayHealth] Starting relay health monitoring for {len(relay_nodes)} relays"
        )

        # Initialize health tracking for all relays
        for relay_id in relay_nodes:
            self._relay_health[relay_id] = RelayHealthStatus(node_id=relay_id)

    async def _run_once(self) -> None:
        """Perform relay health check cycle."""
        relay_nodes = self._get_relay_nodes()
        if not relay_nodes:
            logger.debug("[RelayHealth] No relay nodes to monitor")
            return

        # Update dependent peer mapping
        nat_blocked_peers = self._get_nat_blocked_peers()
        for peer_id, primary_relay in nat_blocked_peers.items():
            if primary_relay in self._relay_health:
                self._relay_health[primary_relay].dependent_peers.add(peer_id)

        # Probe all relay nodes in parallel
        probe_tasks = [
            self._probe_relay(relay_id) for relay_id in relay_nodes
        ]
        await asyncio.gather(*probe_tasks, return_exceptions=True)

        # Check for relays needing failover
        await self._check_for_failovers(nat_blocked_peers)

    async def _probe_relay(self, relay_id: str) -> None:
        """Probe a single relay node and update health status.

        Args:
            relay_id: The relay node ID to probe
        """
        node_info = self._get_node_info(relay_id)
        if not node_info:
            logger.debug(f"[RelayHealth] No info for relay {relay_id}")
            return

        # Get the IP to probe (prefer Tailscale, fallback to SSH host)
        ip = node_info.get("tailscale_ip") or node_info.get("ssh_host")
        if not ip:
            logger.debug(f"[RelayHealth] No IP for relay {relay_id}")
            return

        # Ensure we have health status tracking
        if relay_id not in self._relay_health:
            self._relay_health[relay_id] = RelayHealthStatus(node_id=relay_id)

        status = self._relay_health[relay_id]
        self._probes_sent += 1

        # Probe the relay (try HTTP health endpoint first, then TCP)
        start_time = time.time()
        success = await self._do_probe(ip, relay_id)
        latency_ms = (time.time() - start_time) * 1000

        if success:
            self._probes_succeeded += 1
            was_unhealthy = not status.healthy or status.degraded_event_emitted
            status.record_success(latency_ms)

            # Check if latency is concerning
            if latency_ms > self.config.latency_critical_threshold_ms:
                self._emit_relay_event("RELAY_LATENCY_HIGH", relay_id, {
                    "latency_ms": latency_ms,
                    "threshold_ms": self.config.latency_critical_threshold_ms,
                    "dependent_peers": list(status.dependent_peers),
                })
            elif status.is_latency_trending_up:
                self._emit_relay_event("RELAY_LATENCY_WARNING", relay_id, {
                    "avg_latency_ms": status.avg_latency_ms,
                    "recent_latency_ms": latency_ms,
                    "dependent_peers": list(status.dependent_peers),
                })

            # Emit recovery event if was degraded
            if was_unhealthy:
                status.degraded_event_emitted = False
                self._healthy_events_emitted += 1
                self._emit_relay_event("RELAY_HEALTHY", relay_id, {
                    "latency_ms": latency_ms,
                    "recovery_time_seconds": time.time() - status.last_failure_time,
                    "dependent_peers": list(status.dependent_peers),
                })
                logger.info(
                    f"[RelayHealth] Relay {relay_id} recovered (latency: {latency_ms:.0f}ms)"
                )
        else:
            status.record_failure()

            # Emit degraded event if threshold reached
            if (
                status.consecutive_failures >= self.config.consecutive_failures_for_degraded
                and not status.degraded_event_emitted
            ):
                status.degraded_event_emitted = True
                self._degraded_events_emitted += 1
                self._emit_relay_event("RELAY_DEGRADED", relay_id, {
                    "consecutive_failures": status.consecutive_failures,
                    "last_success_time": status.last_success_time,
                    "downtime_seconds": time.time() - status.last_success_time,
                    "dependent_peers": list(status.dependent_peers),
                })
                logger.warning(
                    f"[RelayHealth] Relay {relay_id} degraded "
                    f"({status.consecutive_failures} consecutive failures, "
                    f"{len(status.dependent_peers)} dependent peers)"
                )

    async def _do_probe(self, ip: str, relay_id: str) -> bool:
        """Perform the actual probe of a relay node.

        Args:
            ip: IP address to probe
            relay_id: The relay node ID (for logging)

        Returns:
            True if probe succeeded, False otherwise
        """
        # Try HTTP health endpoint first (port 8770)
        try:
            import aiohttp
            url = f"http://{ip}:8770/health"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.config.probe_timeout_seconds),
                ) as response:
                    if response.status == 200:
                        return True
        except ImportError:
            pass  # aiohttp not available, fall back to TCP
        except asyncio.TimeoutError:
            logger.debug(f"[RelayHealth] HTTP probe timeout for {relay_id} ({ip})")
        except Exception as e:
            logger.debug(f"[RelayHealth] HTTP probe failed for {relay_id}: {e}")

        # Fall back to TCP probe (SSH port)
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, 22),
                timeout=self.config.probe_timeout_seconds,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, ConnectionError, OSError) as e:
            logger.debug(f"[RelayHealth] TCP probe failed for {relay_id} ({ip}): {e}")
            return False

    async def _check_for_failovers(
        self, nat_blocked_peers: dict[str, str]
    ) -> None:
        """Check if any NAT-blocked peers need relay failover.

        Jan 16, 2026: Now uses chain-aware rotation to follow configured
        relay chain order (primary -> secondary -> tertiary -> quaternary).

        Args:
            nat_blocked_peers: Dict of {peer_id: primary_relay_id}
        """
        if not self._trigger_relay_failover:
            return

        for peer_id, primary_relay in nat_blocked_peers.items():
            if primary_relay not in self._relay_health:
                continue

            status = self._relay_health[primary_relay]
            if status.consecutive_failures >= self.config.consecutive_failures_for_failover:
                # Use chain-aware rotation (Jan 16, 2026)
                alternative = await self._rotate_relay_chain(peer_id, primary_relay)
                if alternative:
                    success = await self._trigger_relay_failover(
                        peer_id, primary_relay, alternative
                    )
                    if success:
                        self._failovers_triggered += 1
                        # Get chain position for the new relay
                        chain = self._get_peer_relay_chain(peer_id)
                        new_position = chain.index(alternative) if alternative in chain else -1
                        self._emit_relay_event("RELAY_FAILOVER", primary_relay, {
                            "peer_id": peer_id,
                            "old_relay": primary_relay,
                            "new_relay": alternative,
                            "new_relay_chain_position": new_position,
                            "failures_before_failover": status.consecutive_failures,
                            "rotation_method": "chain_aware",
                        })
                        logger.info(
                            f"[RelayHealth] Triggered chain failover for {peer_id}: "
                            f"{primary_relay} -> {alternative} (chain position {new_position})"
                        )

    def _find_healthy_alternative_relay(self, exclude_relay: str) -> str | None:
        """Find a healthy relay to use as alternative.

        Args:
            exclude_relay: Relay to exclude from consideration

        Returns:
            Node ID of a healthy relay, or None if none available
        """
        for relay_id, status in self._relay_health.items():
            if relay_id == exclude_relay:
                continue
            if status.healthy and status.consecutive_failures == 0:
                return relay_id

        # Second pass: accept relays with minor issues
        for relay_id, status in self._relay_health.items():
            if relay_id == exclude_relay:
                continue
            if status.healthy:
                return relay_id

        return None

    def _get_peer_relay_chain(self, peer_id: str) -> list[str | None]:
        """Get the configured relay chain for a peer.

        Jan 16, 2026: Returns the ordered relay chain from distributed_hosts.yaml.

        Args:
            peer_id: The peer node ID

        Returns:
            List of relay node IDs [primary, secondary, tertiary, quaternary],
            with None for unconfigured relays.
        """
        node_info = self._get_node_info(peer_id)
        if not node_info:
            return []

        return [
            node_info.get("relay_primary"),
            node_info.get("relay_secondary"),
            node_info.get("relay_tertiary"),
            node_info.get("relay_quaternary"),
        ]

    async def _rotate_relay_chain(
        self, peer_id: str, failed_relay: str
    ) -> str | None:
        """Rotate to the next healthy relay in the configured chain.

        Jan 16, 2026: Chain-aware relay rotation for faster failover.

        Follows the configured chain order:
        relay_primary -> relay_secondary -> relay_tertiary -> relay_quaternary

        Args:
            peer_id: The peer that needs a new relay
            failed_relay: The relay that failed

        Returns:
            The next healthy relay ID, or None if no healthy relays remain
        """
        chain = self._get_peer_relay_chain(peer_id)
        if not chain:
            logger.debug(f"[RelayHealth] No relay chain configured for {peer_id}")
            return self._find_healthy_alternative_relay(failed_relay)

        # Find current position in chain
        try:
            current_idx = chain.index(failed_relay) if failed_relay in chain else -1
        except ValueError:
            current_idx = -1

        # Try relays after the failed one first (chain order)
        for i in range(current_idx + 1, len(chain)):
            candidate = chain[i]
            if not candidate:
                continue

            # Check if this relay is healthy
            if candidate in self._relay_health:
                status = self._relay_health[candidate]
                if status.healthy and status.consecutive_failures == 0:
                    logger.info(
                        f"[RelayHealth] Chain rotation for {peer_id}: "
                        f"{failed_relay} -> {candidate} (position {i})"
                    )
                    return candidate

        # Try relays before the failed one (wrap around)
        for i in range(0, current_idx):
            candidate = chain[i]
            if not candidate or candidate == failed_relay:
                continue

            if candidate in self._relay_health:
                status = self._relay_health[candidate]
                if status.healthy and status.consecutive_failures == 0:
                    logger.info(
                        f"[RelayHealth] Chain rotation (wrap) for {peer_id}: "
                        f"{failed_relay} -> {candidate} (position {i})"
                    )
                    return candidate

        # Last resort: any healthy relay not in chain
        logger.warning(
            f"[RelayHealth] No healthy relays in chain for {peer_id}, "
            f"falling back to any healthy relay"
        )
        return self._find_healthy_alternative_relay(failed_relay)

    def _emit_relay_event(
        self, event_name: str, relay_id: str, payload: dict[str, Any]
    ) -> None:
        """Emit a relay health event.

        Args:
            event_name: Event type (RELAY_DEGRADED, RELAY_HEALTHY, etc.)
            relay_id: The relay node this event is about
            payload: Event payload
        """
        full_payload = {
            "relay_id": relay_id,
            "timestamp": time.time(),
            **payload,
        }

        if self._emit_event:
            try:
                self._emit_event(event_name, full_payload)
            except Exception as e:
                logger.warning(f"[RelayHealth] Failed to emit {event_name}: {e}")
        else:
            logger.debug(f"[RelayHealth] Event: {event_name} - {full_payload}")

    def get_relay_stats(self) -> dict[str, Any]:
        """Get relay health statistics.

        Returns:
            Dictionary with relay health stats and per-relay status
        """
        relay_statuses = {}
        for relay_id, status in self._relay_health.items():
            relay_statuses[relay_id] = {
                "healthy": status.healthy,
                "latency_ms": status.latency_ms,
                "avg_latency_ms": status.avg_latency_ms,
                "consecutive_failures": status.consecutive_failures,
                "dependent_peers_count": len(status.dependent_peers),
                "last_success_time": status.last_success_time,
                "latency_trending_up": status.is_latency_trending_up,
            }

        return {
            "probes_sent": self._probes_sent,
            "probes_succeeded": self._probes_succeeded,
            "success_rate": (
                self._probes_succeeded / self._probes_sent * 100
                if self._probes_sent > 0 else 100.0
            ),
            "failovers_triggered": self._failovers_triggered,
            "degraded_events_emitted": self._degraded_events_emitted,
            "healthy_events_emitted": self._healthy_events_emitted,
            "relay_count": len(self._relay_health),
            "healthy_relay_count": sum(
                1 for s in self._relay_health.values() if s.healthy
            ),
            "relays": relay_statuses,
            **self.stats.to_dict(),
        }

    def get_status(self) -> dict[str, Any]:
        """Get current loop status for monitoring.

        Returns:
            Extended status with relay health information
        """
        base_status = super().get_status()
        base_status["relay_stats"] = self.get_relay_stats()
        return base_status

    def health_check(self) -> "HealthCheckResult":
        """Check loop health for DaemonManager integration.

        Returns:
            HealthCheckResult with relay health metrics
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            # Fallback if protocols not available
            return {  # type: ignore[return-value]
                "healthy": self._running,
                "status": "running" if self._running else "stopped",
                "message": f"RelayHealthLoop {'running' if self._running else 'stopped'}",
                "details": self.get_relay_stats(),
            }

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="RelayHealthLoop is stopped",
            )

        stats = self.get_relay_stats()
        healthy_count = stats.get("healthy_relay_count", 0)
        total_count = stats.get("relay_count", 0)

        # All relays unhealthy is critical
        if total_count > 0 and healthy_count == 0:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message="All relays unhealthy - NAT connectivity at risk",
                details=stats,
            )

        # Some relays unhealthy is degraded
        if healthy_count < total_count:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.DEGRADED,
                message=f"Relay health degraded ({healthy_count}/{total_count} healthy)",
                details=stats,
            )

        # All healthy
        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"RelayHealthLoop healthy ({healthy_count} relays)",
            details=stats,
        )


# Convenience export
__all__ = [
    "RelayHealthConfig",
    "RelayHealthLoop",
    "RelayHealthStatus",
]
