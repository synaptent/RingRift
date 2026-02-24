"""
Leader Health Probing Module.

Dec 30, 2025: Part of Phase 6 - Bully Algorithm Enhancements.

Provides multi-path leader health probing for faster failover detection.
Voter nodes probe the leader through multiple transports to ensure
accurate health status.

Features:
- Multi-transport probing (HTTP, Tailscale, gossip)
- Adaptive probe intervals based on leader stability
- Successor preparation for fast failover
- Graceful step-down support

Usage:
    from scripts.p2p.leader_health import (
        LeaderHealthProbe,
        LeaderHealthResult,
        get_leader_health_probe,
    )

    probe = get_leader_health_probe(node_id="voter-1", leader_id="leader-1")
    result = await probe.probe_leader(leader_addr="192.168.1.10:8770")
    if not result.is_healthy:
        # Trigger election
        pass
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from app.core.async_context import safe_create_task

logger = logging.getLogger(__name__)

# January 2026: Use centralized timeouts from loop_constants
try:
    from scripts.p2p.loops.loop_constants import LoopTimeouts
    DEFAULT_PROBE_TIMEOUT = LoopTimeouts.LEADER_PROBE  # 5.0 seconds per probe
except ImportError:
    DEFAULT_PROBE_TIMEOUT = 5.0  # Fallback

# Probe configuration
DEFAULT_PROBE_INTERVAL = 15.0  # seconds between probes
FAST_PROBE_INTERVAL = 5.0  # seconds when leader is unstable
MIN_PROBES_FOR_DECISION = 2  # require 2+ transports to agree
CONSECUTIVE_FAILURES_THRESHOLD = 3  # failures before declaring unhealthy


class ProbeTransport(str, Enum):
    """Transport types for leader probing."""

    HTTP_DIRECT = "http_direct"
    HTTP_TAILSCALE = "http_tailscale"
    GOSSIP = "gossip"
    TCP_PING = "tcp_ping"
    WORK_ACCEPTANCE = "work_acceptance"  # January 2, 2026: Frozen leader detection


class LeaderHealthStatus(str, Enum):
    """Health status of the leader."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Some probes failing
    UNHEALTHY = "unhealthy"  # All probes failing
    FROZEN = "frozen"  # Leader responds to heartbeat but not accepting work (stuck event loop)
    UNKNOWN = "unknown"  # Not enough data


@dataclass
class ProbeResult:
    """Result of a single probe attempt."""

    transport: ProbeTransport
    success: bool
    latency_ms: float = 0.0
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class LeaderHealthResult:
    """Aggregated health result from all probes."""

    is_healthy: bool
    status: LeaderHealthStatus
    successful_probes: list[ProbeResult] = field(default_factory=list)
    failed_probes: list[ProbeResult] = field(default_factory=list)
    best_latency_ms: float | None = None
    best_transport: ProbeTransport | None = None
    consecutive_failures: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_healthy": self.is_healthy,
            "status": self.status.value,
            "successful_probes": len(self.successful_probes),
            "failed_probes": len(self.failed_probes),
            "best_latency_ms": self.best_latency_ms,
            "best_transport": self.best_transport.value if self.best_transport else None,
            "consecutive_failures": self.consecutive_failures,
            "timestamp": self.timestamp,
        }


@dataclass
class LeaderProbeConfig:
    """Configuration for leader health probing."""

    probe_timeout: float = DEFAULT_PROBE_TIMEOUT
    probe_interval: float = DEFAULT_PROBE_INTERVAL
    fast_probe_interval: float = FAST_PROBE_INTERVAL
    min_probes_for_decision: int = MIN_PROBES_FOR_DECISION
    consecutive_failures_threshold: int = CONSECUTIVE_FAILURES_THRESHOLD
    enable_http_probe: bool = True
    enable_tailscale_probe: bool = True
    enable_gossip_probe: bool = True
    enable_tcp_probe: bool = True
    # January 2, 2026: Frozen leader detection
    enable_work_acceptance_probe: bool = True
    frozen_leader_consecutive_failures: int = 3
    frozen_leader_grace_period: float = 60.0


class LeaderHealthProbe:
    """
    Multi-path leader health prober for voter nodes.

    Probes the leader through multiple transports and aggregates results
    to determine leader health status.
    """

    def __init__(
        self,
        node_id: str,
        leader_id: str | None = None,
        config: LeaderProbeConfig | None = None,
    ) -> None:
        """Initialize the leader health probe.

        Args:
            node_id: This node's ID (voter node)
            leader_id: Current leader's ID
            config: Probe configuration
        """
        self._node_id = node_id
        self._leader_id = leader_id
        self._config = config or LeaderProbeConfig()
        self._consecutive_failures = 0
        self._last_probe_time = 0.0
        self._last_successful_probe = 0.0
        self._probe_history: list[LeaderHealthResult] = []
        self._running = False
        self._probe_task: asyncio.Task | None = None

        # Transport probers (initialized lazily)
        self._http_session: Any = None

        # January 2, 2026: Frozen leader detection
        # Track work acceptance probe failures separately from heartbeat failures
        self._work_acceptance_failures = 0
        self._last_work_acceptance_success = 0.0
        self._leader_became_leader_at = 0.0  # For grace period tracking

    async def probe_leader(
        self,
        leader_addr: str | None = None,
        tailscale_addr: str | None = None,
    ) -> LeaderHealthResult:
        """Probe the leader through all available transports.

        Args:
            leader_addr: Leader's HTTP address (host:port)
            tailscale_addr: Leader's Tailscale address (optional)

        Returns:
            Aggregated health result
        """
        if not self._leader_id:
            return LeaderHealthResult(
                is_healthy=False,
                status=LeaderHealthStatus.UNKNOWN,
            )

        # Run all enabled probes in parallel
        probe_tasks = []

        if self._config.enable_http_probe and leader_addr:
            probe_tasks.append(self._probe_http(leader_addr))

        if self._config.enable_tailscale_probe and tailscale_addr:
            probe_tasks.append(self._probe_tailscale(tailscale_addr))

        if self._config.enable_gossip_probe:
            probe_tasks.append(self._probe_gossip())

        if self._config.enable_tcp_probe and leader_addr:
            host = leader_addr.split(":")[0]
            port = int(leader_addr.split(":")[1]) if ":" in leader_addr else 8770
            probe_tasks.append(self._probe_tcp(host, port))

        # January 2, 2026: Work acceptance probe for frozen leader detection
        if self._config.enable_work_acceptance_probe and leader_addr:
            probe_tasks.append(self._probe_work_acceptance(leader_addr))

        if not probe_tasks:
            return LeaderHealthResult(
                is_healthy=False,
                status=LeaderHealthStatus.UNKNOWN,
            )

        # Sprint 4 (Jan 2, 2026): Fail-fast probing with early exit
        # Use asyncio.wait with FIRST_COMPLETED to return quickly on success
        # instead of waiting for all probes (which could include slow timeouts)
        successful = []
        failed = []
        pending_tasks = {safe_create_task(t, name="leader-health-probe") for t in probe_tasks}
        completed_tasks: set[asyncio.Task] = set()

        try:
            while pending_tasks:
                # Wait for the first probe to complete
                done, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    timeout=self._config.probe_timeout + 1.0,  # Slightly longer than individual probe timeout
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    # All remaining tasks timed out
                    break

                completed_tasks.update(done)

                # Process completed probes
                for task in done:
                    try:
                        result = task.result()
                        if isinstance(result, ProbeResult):
                            if result.success:
                                successful.append(result)
                            else:
                                failed.append(result)
                    except Exception as e:
                        failed.append(ProbeResult(
                            transport=ProbeTransport.HTTP_DIRECT,
                            success=False,
                            error=str(e),
                        ))

                # Fail-fast: If we have enough successful probes, cancel remaining
                # and return immediately (skip slow/timing-out probes)
                heartbeat_successful = [p for p in successful if p.transport != ProbeTransport.WORK_ACCEPTANCE]
                if len(heartbeat_successful) >= self._config.min_probes_for_decision:
                    # We have enough for a healthy decision
                    logger.debug(
                        f"[LeaderProbe] Fail-fast exit: {len(heartbeat_successful)} successful probes "
                        f"(threshold: {self._config.min_probes_for_decision}), "
                        f"cancelling {len(pending_tasks)} remaining"
                    )
                    break

                # Also check if remaining probes can't change outcome
                max_possible_successes = len(heartbeat_successful) + len(pending_tasks)
                if max_possible_successes < self._config.min_probes_for_decision:
                    # Can't possibly reach threshold even if all remaining succeed
                    logger.debug(
                        f"[LeaderProbe] Early fail: max possible {max_possible_successes} < "
                        f"threshold {self._config.min_probes_for_decision}"
                    )
                    break

        finally:
            # Cancel any remaining tasks to avoid resource leaks
            for task in pending_tasks:
                task.cancel()
            # Give cancelled tasks a chance to clean up
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)

        # January 2, 2026: Separate work acceptance probe from heartbeat probes
        # for frozen leader detection
        heartbeat_successful = [p for p in successful if p.transport != ProbeTransport.WORK_ACCEPTANCE]

        # Determine overall health based on heartbeat probes
        if len(heartbeat_successful) >= self._config.min_probes_for_decision:
            is_healthy = True
            status = LeaderHealthStatus.HEALTHY
            self._consecutive_failures = 0
            self._last_successful_probe = time.time()
        elif len(heartbeat_successful) > 0:
            is_healthy = True  # Partial success is still healthy
            status = LeaderHealthStatus.DEGRADED
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._config.consecutive_failures_threshold:
                is_healthy = False
                status = LeaderHealthStatus.UNHEALTHY
            else:
                is_healthy = True  # Still within tolerance
                status = LeaderHealthStatus.DEGRADED

        # January 2, 2026: Check for frozen leader (heartbeat OK but work acceptance failing)
        # A frozen leader is unhealthy even if heartbeats succeed
        if status in (LeaderHealthStatus.HEALTHY, LeaderHealthStatus.DEGRADED):
            if self.is_leader_frozen():
                is_healthy = False
                status = LeaderHealthStatus.FROZEN
                logger.warning(
                    f"Leader {self._leader_id} detected as FROZEN: heartbeat OK but "
                    f"work acceptance failed {self._work_acceptance_failures} times"
                )

        # Find best transport
        best_latency = None
        best_transport = None
        for probe in successful:
            if best_latency is None or probe.latency_ms < best_latency:
                best_latency = probe.latency_ms
                best_transport = probe.transport

        result = LeaderHealthResult(
            is_healthy=is_healthy,
            status=status,
            successful_probes=successful,
            failed_probes=failed,
            best_latency_ms=best_latency,
            best_transport=best_transport,
            consecutive_failures=self._consecutive_failures,
        )

        self._last_probe_time = time.time()
        self._probe_history.append(result)

        # Keep last 100 results
        if len(self._probe_history) > 100:
            self._probe_history = self._probe_history[-100:]

        return result

    async def _probe_http(self, addr: str) -> ProbeResult:
        """Probe leader via HTTP /status endpoint."""
        start = time.time()
        try:
            import aiohttp

            if self._http_session is None:
                self._http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._config.probe_timeout)
                )

            url = f"http://{addr}/status"
            async with self._http_session.get(url) as resp:
                if resp.status == 200:
                    latency = (time.time() - start) * 1000
                    return ProbeResult(
                        transport=ProbeTransport.HTTP_DIRECT,
                        success=True,
                        latency_ms=latency,
                    )
                else:
                    return ProbeResult(
                        transport=ProbeTransport.HTTP_DIRECT,
                        success=False,
                        error=f"HTTP {resp.status}",
                    )

        except asyncio.TimeoutError:
            return ProbeResult(
                transport=ProbeTransport.HTTP_DIRECT,
                success=False,
                error="timeout",
            )
        except Exception as e:
            # Close session on connection errors so it gets recreated on next probe
            if self._http_session:
                try:
                    await self._http_session.close()
                except Exception:
                    pass
                self._http_session = None
            return ProbeResult(
                transport=ProbeTransport.HTTP_DIRECT,
                success=False,
                error=str(e),
            )

    async def _probe_tailscale(self, addr: str) -> ProbeResult:
        """Probe leader via Tailscale network."""
        start = time.time()
        try:
            import aiohttp

            if self._http_session is None:
                self._http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._config.probe_timeout)
                )

            # Tailscale addresses typically don't need port specification
            if ":" not in addr:
                addr = f"{addr}:8770"

            url = f"http://{addr}/status"
            async with self._http_session.get(url) as resp:
                if resp.status == 200:
                    latency = (time.time() - start) * 1000
                    return ProbeResult(
                        transport=ProbeTransport.HTTP_TAILSCALE,
                        success=True,
                        latency_ms=latency,
                    )
                else:
                    return ProbeResult(
                        transport=ProbeTransport.HTTP_TAILSCALE,
                        success=False,
                        error=f"HTTP {resp.status}",
                    )

        except asyncio.TimeoutError:
            return ProbeResult(
                transport=ProbeTransport.HTTP_TAILSCALE,
                success=False,
                error="timeout",
            )
        except Exception as e:
            # Close session on connection errors so it gets recreated on next probe
            if self._http_session:
                try:
                    await self._http_session.close()
                except Exception:
                    pass
                self._http_session = None
            return ProbeResult(
                transport=ProbeTransport.HTTP_TAILSCALE,
                success=False,
                error=str(e),
            )

    async def _probe_gossip(self) -> ProbeResult:
        """Probe leader via gossip protocol (if available).

        Checks if the gossip protocol considers the leader alive.
        """
        start = time.time()
        try:
            # Try to use gossip protocol to check leader status
            # This is a passive check - we're checking our gossip state
            from scripts.p2p.gossip_protocol import get_gossip_protocol

            gossip = get_gossip_protocol()
            if gossip and self._leader_id:
                is_alive = await gossip.is_peer_alive(self._leader_id)
                latency = (time.time() - start) * 1000
                return ProbeResult(
                    transport=ProbeTransport.GOSSIP,
                    success=is_alive,
                    latency_ms=latency,
                    error=None if is_alive else "peer_not_alive_in_gossip",
                )

            return ProbeResult(
                transport=ProbeTransport.GOSSIP,
                success=False,
                error="gossip_not_available",
            )

        except ImportError:
            return ProbeResult(
                transport=ProbeTransport.GOSSIP,
                success=False,
                error="gossip_not_available",
            )
        except Exception as e:
            return ProbeResult(
                transport=ProbeTransport.GOSSIP,
                success=False,
                error=str(e),
            )

    async def _probe_tcp(self, host: str, port: int) -> ProbeResult:
        """Probe leader via TCP connection test."""
        start = time.time()
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self._config.probe_timeout,
            )
            writer.close()
            await writer.wait_closed()

            latency = (time.time() - start) * 1000
            return ProbeResult(
                transport=ProbeTransport.TCP_PING,
                success=True,
                latency_ms=latency,
            )

        except asyncio.TimeoutError:
            return ProbeResult(
                transport=ProbeTransport.TCP_PING,
                success=False,
                error="timeout",
            )
        except Exception as e:
            return ProbeResult(
                transport=ProbeTransport.TCP_PING,
                success=False,
                error=str(e),
            )

    async def _probe_work_acceptance(self, addr: str) -> ProbeResult:
        """Probe leader via /admin/ping_work endpoint.

        January 2, 2026: Detects frozen leaders that respond to heartbeats
        but can't accept new work (stuck event loop).

        This probe POSTs a ping_work request that requires the leader's
        event loop to process a simple task within timeout. If the leader
        is frozen (deadlock, long-running sync, etc.), this will fail
        even though /status still responds.

        Args:
            addr: Leader's HTTP address (host:port)

        Returns:
            ProbeResult indicating work acceptance capability
        """
        start = time.time()
        try:
            import aiohttp

            if self._http_session is None:
                self._http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._config.probe_timeout)
                )

            # POST to /admin/ping_work - requires event loop processing
            url = f"http://{addr}/admin/ping_work"
            payload = {
                "probe_id": f"{self._node_id}-{int(time.time()*1000)}",
                "prober_node": self._node_id,
                "timestamp": time.time(),
            }

            async with self._http_session.post(url, json=payload) as resp:
                if resp.status == 200:
                    latency = (time.time() - start) * 1000
                    self._work_acceptance_failures = 0
                    self._last_work_acceptance_success = time.time()
                    return ProbeResult(
                        transport=ProbeTransport.WORK_ACCEPTANCE,
                        success=True,
                        latency_ms=latency,
                    )
                else:
                    self._work_acceptance_failures += 1
                    return ProbeResult(
                        transport=ProbeTransport.WORK_ACCEPTANCE,
                        success=False,
                        error=f"HTTP {resp.status}",
                    )

        except asyncio.TimeoutError:
            self._work_acceptance_failures += 1
            return ProbeResult(
                transport=ProbeTransport.WORK_ACCEPTANCE,
                success=False,
                error="timeout",
            )
        except Exception as e:
            self._work_acceptance_failures += 1
            # Close session on connection errors so it gets recreated on next probe
            if self._http_session:
                try:
                    await self._http_session.close()
                except Exception:
                    pass
                self._http_session = None
            return ProbeResult(
                transport=ProbeTransport.WORK_ACCEPTANCE,
                success=False,
                error=str(e),
            )

    def is_leader_frozen(self) -> bool:
        """Check if the leader appears frozen (heartbeating but not accepting work).

        January 2, 2026: A leader is considered frozen if:
        1. Enough consecutive work acceptance failures have occurred
        2. We're past the grace period for new leaders
        3. Heartbeat probes are still succeeding (distinguishes from UNHEALTHY)

        Returns:
            True if leader appears frozen
        """
        # Check if past grace period
        if self._leader_became_leader_at > 0:
            time_as_leader = time.time() - self._leader_became_leader_at
            if time_as_leader < self._config.frozen_leader_grace_period:
                return False

        # Check if enough work acceptance failures
        return self._work_acceptance_failures >= self._config.frozen_leader_consecutive_failures

    def notify_leader_change(self, leader_id: str | None) -> None:
        """Notify that the leader has changed.

        January 2, 2026: Resets frozen leader tracking when leader changes.
        """
        if leader_id != self._leader_id:
            self._leader_id = leader_id
            self._consecutive_failures = 0
            self._work_acceptance_failures = 0
            self._last_work_acceptance_success = 0.0
            self._leader_became_leader_at = time.time()
            self._probe_history.clear()

    async def start_continuous_probing(
        self,
        leader_addr: str,
        on_unhealthy: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        """Start continuous leader probing in background.

        Args:
            leader_addr: Leader's HTTP address
            on_unhealthy: Async callback when leader becomes unhealthy
        """
        if self._running:
            return

        self._running = True
        self._probe_task = safe_create_task(
            self._probe_loop(leader_addr, on_unhealthy),
            name="leader-health-continuous-probe",
        )
        logger.debug(f"Started continuous probing of leader {self._leader_id}")

    async def _probe_loop(
        self,
        leader_addr: str,
        on_unhealthy: Callable[[], Coroutine[Any, Any, None]] | None,
    ) -> None:
        """Background loop for continuous probing."""
        while self._running:
            try:
                # Determine probe interval based on health
                if self._consecutive_failures > 0:
                    interval = self._config.fast_probe_interval
                else:
                    interval = self._config.probe_interval

                # Wait for interval
                await asyncio.sleep(interval)

                if not self._running:
                    break

                # Probe leader
                result = await self.probe_leader(leader_addr)

                if not result.is_healthy and on_unhealthy:
                    logger.warning(
                        f"Leader {self._leader_id} is unhealthy "
                        f"({result.consecutive_failures} failures)"
                    )
                    await on_unhealthy()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Probe loop error: {e}")

    async def stop_continuous_probing(self) -> None:
        """Stop continuous leader probing."""
        self._running = False
        if self._probe_task:
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass

        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        logger.debug("Stopped continuous leader probing")

    def set_leader(self, leader_id: str | None) -> None:
        """Update the leader being probed.

        January 2, 2026: Also resets frozen leader tracking.
        """
        if leader_id != self._leader_id:
            self._leader_id = leader_id
            self._consecutive_failures = 0
            self._work_acceptance_failures = 0
            self._last_work_acceptance_success = 0.0
            self._leader_became_leader_at = time.time()
            self._probe_history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get probe statistics."""
        successful_count = sum(
            1 for r in self._probe_history if r.is_healthy
        )
        total_count = len(self._probe_history)

        return {
            "leader_id": self._leader_id,
            "consecutive_failures": self._consecutive_failures,
            "last_probe_time": self._last_probe_time,
            "last_successful_probe": self._last_successful_probe,
            "probe_history_count": total_count,
            "success_rate": successful_count / total_count if total_count > 0 else 0.0,
            "is_probing": self._running,
            # January 2, 2026: Frozen leader detection stats
            "work_acceptance_failures": self._work_acceptance_failures,
            "last_work_acceptance_success": self._last_work_acceptance_success,
            "is_leader_frozen": self.is_leader_frozen(),
            "leader_became_leader_at": self._leader_became_leader_at,
        }


# Singleton management
_probe_instances: dict[str, LeaderHealthProbe] = {}


def get_leader_health_probe(
    node_id: str,
    leader_id: str | None = None,
    config: LeaderProbeConfig | None = None,
) -> LeaderHealthProbe:
    """Get or create a leader health probe for a node.

    Args:
        node_id: Voter node ID
        leader_id: Current leader ID
        config: Probe configuration

    Returns:
        LeaderHealthProbe instance
    """
    global _probe_instances

    if node_id not in _probe_instances:
        _probe_instances[node_id] = LeaderHealthProbe(
            node_id=node_id,
            leader_id=leader_id,
            config=config,
        )
    else:
        # Update leader if changed
        _probe_instances[node_id].set_leader(leader_id)

    return _probe_instances[node_id]


def clear_probe_instances() -> None:
    """Clear all probe instances (for testing)."""
    global _probe_instances
    _probe_instances.clear()
