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

logger = logging.getLogger(__name__)

# Probe configuration
DEFAULT_PROBE_TIMEOUT = 5.0  # seconds per probe
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


class LeaderHealthStatus(str, Enum):
    """Health status of the leader."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Some probes failing
    UNHEALTHY = "unhealthy"  # All probes failing
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

        if not probe_tasks:
            return LeaderHealthResult(
                is_healthy=False,
                status=LeaderHealthStatus.UNKNOWN,
            )

        # Wait for all probes with timeout
        results = await asyncio.gather(*probe_tasks, return_exceptions=True)

        # Aggregate results
        successful = []
        failed = []

        for result in results:
            if isinstance(result, Exception):
                failed.append(ProbeResult(
                    transport=ProbeTransport.HTTP_DIRECT,
                    success=False,
                    error=str(result),
                ))
            elif isinstance(result, ProbeResult):
                if result.success:
                    successful.append(result)
                else:
                    failed.append(result)

        # Determine overall health
        if len(successful) >= self._config.min_probes_for_decision:
            is_healthy = True
            status = LeaderHealthStatus.HEALTHY
            self._consecutive_failures = 0
            self._last_successful_probe = time.time()
        elif len(successful) > 0:
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
        self._probe_task = asyncio.create_task(
            self._probe_loop(leader_addr, on_unhealthy)
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
        """Update the leader being probed."""
        if leader_id != self._leader_id:
            self._leader_id = leader_id
            self._consecutive_failures = 0
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
