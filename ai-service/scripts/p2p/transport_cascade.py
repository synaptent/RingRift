"""
Comprehensive multi-layer transport failover cascade.

Dec 30, 2025: Implements exhaustive transport failover - tries every possible
communication method before declaring a node unreachable.

Design Philosophy:
- Union over intersection: try ALL transports, not just the "best" one
- Tiered approach: fast transports first, expensive transports last
- Self-healing: learns from successes/failures to optimize future attempts
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class TransportTier(Enum):
    """Transport tiers ordered by preference (fastest/cheapest first)."""

    TIER_1_FAST = auto()  # UDP, direct HTTP (sub-100ms)
    TIER_2_RELIABLE = auto()  # TCP, Tailscale mesh (100-500ms)
    TIER_3_TUNNELED = auto()  # SSH tunnel, Cloudflare (500ms-2s)
    TIER_4_RELAY = auto()  # P2P relay, TURN server (1-5s)
    TIER_5_EXTERNAL = auto()  # Slack/Discord webhook, email API (5-30s)
    TIER_6_MANUAL = auto()  # SMS, PagerDuty, human escalation (30s+)


@dataclass
class TransportResult:
    """Result of a transport attempt."""

    success: bool
    transport_name: str
    latency_ms: float
    response: bytes | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransportHealth:
    """Health statistics for a transport to a specific target."""

    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0-1.0)."""
        total = self.successes + self.failures
        if total == 0:
            return 0.5  # Unknown, assume 50%
        return self.successes / total

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in ms."""
        if self.successes == 0:
            return float("inf")
        return self.total_latency_ms / self.successes

    def record_success(self, latency_ms: float) -> None:
        """Record a successful transport attempt."""
        self.successes += 1
        self.total_latency_ms += latency_ms
        self.last_success_time = time.time()
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed transport attempt."""
        self.failures += 1
        self.last_failure_time = time.time()
        self.consecutive_failures += 1


@runtime_checkable
class TransportProtocol(Protocol):
    """Protocol for all transport implementations."""

    name: str
    tier: TransportTier

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Attempt to send payload to target."""
        ...

    async def is_available(self, target: str) -> bool:
        """Check if this transport can reach target."""
        ...


class BaseTransport(ABC):
    """Base class for transport implementations."""

    name: str = "base"
    tier: TransportTier = TransportTier.TIER_1_FAST

    @abstractmethod
    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Attempt to send payload to target."""
        pass

    async def is_available(self, target: str) -> bool:
        """Check if this transport can reach target. Override for custom logic."""
        return True

    def _make_result(
        self,
        success: bool,
        latency_ms: float,
        response: bytes | None = None,
        error: str | None = None,
        **metadata: Any,
    ) -> TransportResult:
        """Helper to create TransportResult."""
        return TransportResult(
            success=success,
            transport_name=self.name,
            latency_ms=latency_ms,
            response=response,
            error=error,
            metadata=metadata,
        )


class TransportCascade:
    """
    Exhaustive transport failover - try everything until something works.

    Features:
    - Tiered cascade: fast transports first, expensive ones last
    - Health-aware ordering: prioritize historically successful transports
    - Parallel probing: can probe multiple transports simultaneously
    - Circuit breaker integration: stops cascade if global CB opens
    """

    def __init__(self):
        self._transports: list[BaseTransport] = []
        # target -> transport_name -> TransportHealth
        self._health: dict[str, dict[str, TransportHealth]] = {}
        self._global_circuit_breaker: GlobalCircuitBreaker | None = None

        # Configuration from environment
        self._enabled = os.environ.get("RINGRIFT_TRANSPORT_CASCADE_ENABLED", "true").lower() == "true"
        self._min_tier = int(os.environ.get("RINGRIFT_TRANSPORT_MIN_TIER", "1"))
        self._max_tier = int(os.environ.get("RINGRIFT_TRANSPORT_MAX_TIER", "5"))
        self._timeout_per_transport = float(os.environ.get("RINGRIFT_TRANSPORT_TIMEOUT", "10"))

    def set_circuit_breaker(self, cb: GlobalCircuitBreaker) -> None:
        """Set global circuit breaker for cascade control."""
        self._global_circuit_breaker = cb

    def register_transport(self, transport: BaseTransport) -> None:
        """Register a transport in the cascade."""
        self._transports.append(transport)
        # Sort by tier (lower = faster/cheaper)
        self._transports.sort(key=lambda t: t.tier.value)
        logger.info(f"Registered transport: {transport.name} (tier {transport.tier.name})")

    def get_transports(self) -> list[BaseTransport]:
        """Get all registered transports."""
        return list(self._transports)

    def get_health(self, target: str, transport_name: str) -> TransportHealth:
        """Get health stats for a transport to a target."""
        if target not in self._health:
            self._health[target] = {}
        if transport_name not in self._health[target]:
            self._health[target][transport_name] = TransportHealth()
        return self._health[target][transport_name]

    async def send_with_cascade(
        self,
        target: str,
        payload: bytes,
        *,
        min_tier: TransportTier | None = None,
        max_tier: TransportTier | None = None,
        timeout_per_transport: float | None = None,
        parallel_probe: bool = False,
    ) -> TransportResult:
        """
        Try all transports in tier order until one succeeds.

        Args:
            target: Target node identifier (node_id or address)
            payload: Data to send
            min_tier: Minimum tier to try (default: TIER_1_FAST)
            max_tier: Maximum tier to try (default: TIER_5_EXTERNAL)
            timeout_per_transport: Timeout per transport attempt
            parallel_probe: If True, try all transports in each tier simultaneously

        Returns:
            TransportResult with success/failure details
        """
        if not self._enabled:
            return TransportResult(
                success=False,
                transport_name="cascade_disabled",
                latency_ms=0,
                error="Transport cascade is disabled",
            )

        # Check global circuit breaker
        if self._global_circuit_breaker and self._global_circuit_breaker.is_open:
            return TransportResult(
                success=False,
                transport_name="circuit_breaker_open",
                latency_ms=0,
                error="Global circuit breaker is open",
            )

        # Use defaults from config if not specified
        effective_min = min_tier or TransportTier(self._min_tier)
        effective_max = max_tier or TransportTier(self._max_tier)
        effective_timeout = timeout_per_transport or self._timeout_per_transport

        # Filter transports by tier
        eligible = [
            t
            for t in self._transports
            if effective_min.value <= t.tier.value <= effective_max.value
        ]

        if not eligible:
            return TransportResult(
                success=False,
                transport_name="no_eligible_transports",
                latency_ms=0,
                error=f"No transports in tier range {effective_min.name}-{effective_max.name}",
            )

        # Sort by health (success rate, then latency)
        sorted_transports = self._sort_by_health(eligible, target)

        if parallel_probe:
            return await self._cascade_parallel(
                target, payload, sorted_transports, effective_timeout
            )
        else:
            return await self._cascade_sequential(
                target, payload, sorted_transports, effective_timeout
            )

    async def _cascade_sequential(
        self,
        target: str,
        payload: bytes,
        transports: list[BaseTransport],
        timeout: float,
    ) -> TransportResult:
        """Try transports sequentially."""
        errors = []

        for transport in transports:
            # Check availability first
            try:
                if not await transport.is_available(target):
                    errors.append(f"{transport.name}: not available")
                    continue
            except Exception as e:
                errors.append(f"{transport.name}: availability check failed: {e}")
                continue

            # Try to send
            start_time = time.time()
            try:
                result = await asyncio.wait_for(
                    transport.send(target, payload),
                    timeout=timeout,
                )
                latency_ms = (time.time() - start_time) * 1000

                if result.success:
                    self._record_success(target, transport.name, latency_ms)
                    return result
                else:
                    self._record_failure(target, transport.name)
                    errors.append(f"{transport.name}: {result.error}")

            except asyncio.TimeoutError:
                self._record_failure(target, transport.name)
                errors.append(f"{transport.name}: timeout ({timeout}s)")

            except Exception as e:
                self._record_failure(target, transport.name)
                errors.append(f"{transport.name}: {type(e).__name__}: {e}")

        # All transports failed
        return TransportResult(
            success=False,
            transport_name="cascade_exhausted",
            latency_ms=0,
            error=f"All {len(transports)} transports failed: {'; '.join(errors)}",
        )

    async def _cascade_parallel(
        self,
        target: str,
        payload: bytes,
        transports: list[BaseTransport],
        timeout: float,
    ) -> TransportResult:
        """Try all transports in parallel within each tier."""
        # Group by tier
        by_tier: dict[TransportTier, list[BaseTransport]] = {}
        for t in transports:
            by_tier.setdefault(t.tier, []).append(t)

        errors = []

        # Try each tier
        for tier in sorted(by_tier.keys(), key=lambda t: t.value):
            tier_transports = by_tier[tier]

            # Create tasks for all transports in this tier
            tasks = []
            for transport in tier_transports:
                task = asyncio.create_task(
                    self._try_transport(target, payload, transport, timeout)
                )
                tasks.append((transport.name, task))

            # Wait for first success or all failures
            pending = set(t for _, t in tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    result = task.result()
                    if result.success:
                        # Cancel remaining tasks
                        for p in pending:
                            p.cancel()
                        return result
                    else:
                        errors.append(f"{result.transport_name}: {result.error}")

        # All tiers exhausted
        return TransportResult(
            success=False,
            transport_name="cascade_exhausted",
            latency_ms=0,
            error=f"All transports failed: {'; '.join(errors)}",
        )

    async def _try_transport(
        self,
        target: str,
        payload: bytes,
        transport: BaseTransport,
        timeout: float,
    ) -> TransportResult:
        """Helper to try a single transport with error handling."""
        start_time = time.time()
        try:
            if not await transport.is_available(target):
                return TransportResult(
                    success=False,
                    transport_name=transport.name,
                    latency_ms=0,
                    error="not available",
                )

            result = await asyncio.wait_for(
                transport.send(target, payload),
                timeout=timeout,
            )
            latency_ms = (time.time() - start_time) * 1000

            if result.success:
                self._record_success(target, transport.name, latency_ms)
            else:
                self._record_failure(target, transport.name)

            return result

        except asyncio.TimeoutError:
            self._record_failure(target, transport.name)
            return TransportResult(
                success=False,
                transport_name=transport.name,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"timeout ({timeout}s)",
            )

        except Exception as e:
            self._record_failure(target, transport.name)
            return TransportResult(
                success=False,
                transport_name=transport.name,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    def _sort_by_health(
        self, transports: list[BaseTransport], target: str
    ) -> list[BaseTransport]:
        """Sort transports by health for a target (best first)."""

        def health_score(t: BaseTransport) -> tuple[int, float, float]:
            health = self.get_health(target, t.name)
            # Primary: tier (lower = better)
            # Secondary: success rate (higher = better, negated for sort)
            # Tertiary: avg latency (lower = better)
            return (
                t.tier.value,
                -health.success_rate,
                health.avg_latency_ms,
            )

        return sorted(transports, key=health_score)

    def _record_success(self, target: str, transport_name: str, latency_ms: float) -> None:
        """Record successful transport attempt."""
        health = self.get_health(target, transport_name)
        health.record_success(latency_ms)

        if self._global_circuit_breaker:
            self._global_circuit_breaker.record_success(transport_name, target)

        logger.debug(
            f"Transport success: {transport_name} -> {target} "
            f"({latency_ms:.1f}ms, rate={health.success_rate:.2f})"
        )

    def _record_failure(self, target: str, transport_name: str) -> None:
        """Record failed transport attempt."""
        health = self.get_health(target, transport_name)
        health.record_failure()

        if self._global_circuit_breaker:
            self._global_circuit_breaker.record_failure(transport_name, target)

        logger.debug(
            f"Transport failure: {transport_name} -> {target} "
            f"(consecutive={health.consecutive_failures}, rate={health.success_rate:.2f})"
        )


@dataclass
class GlobalCircuitState:
    """Cross-transport circuit breaker state."""

    total_failures: int = 0
    total_successes: int = 0
    failures_by_transport: dict[str, int] = field(default_factory=dict)
    failures_by_target: dict[str, int] = field(default_factory=dict)
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    state: str = "closed"  # "closed", "half_open", "open"
    opened_at: float = 0.0


class GlobalCircuitBreaker:
    """
    Global circuit breaker monitoring failures across ALL transports.

    Opens when cluster-wide failure rate exceeds threshold,
    triggering emergency notifications.
    """

    def __init__(
        self,
        failure_threshold: int = 50,
        recovery_timeout: float = 60.0,
        success_threshold: int = 5,
    ):
        """
        Args:
            failure_threshold: Total failures before opening
            recovery_timeout: Seconds before half-open
            success_threshold: Successes needed to close
        """
        self._state = GlobalCircuitState()
        self._failure_threshold = int(
            os.environ.get("RINGRIFT_GLOBAL_CB_FAILURE_THRESHOLD", str(failure_threshold))
        )
        self._recovery_timeout = float(
            os.environ.get("RINGRIFT_GLOBAL_CB_RECOVERY_TIMEOUT", str(recovery_timeout))
        )
        self._success_threshold = success_threshold
        self._half_open_successes = 0
        self._notification_callback: Any = None

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._state.state == "open":
            # Check if recovery timeout has passed
            if time.time() - self._state.opened_at > self._recovery_timeout:
                self._state.state = "half_open"
                self._half_open_successes = 0
                logger.info("Global circuit breaker entering half-open state")
                return False
            return True
        return False

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state.state

    def set_notification_callback(self, callback: Any) -> None:
        """Set callback for emergency notifications."""
        self._notification_callback = callback

    def record_failure(self, transport: str, target: str) -> None:
        """Record a failure and potentially open circuit."""
        self._state.total_failures += 1
        self._state.last_failure_time = time.time()
        self._state.failures_by_transport[transport] = (
            self._state.failures_by_transport.get(transport, 0) + 1
        )
        self._state.failures_by_target[target] = (
            self._state.failures_by_target.get(target, 0) + 1
        )

        # Check if we should open
        if (
            self._state.state == "closed"
            and self._state.total_failures >= self._failure_threshold
        ):
            self._open_circuit()
        elif self._state.state == "half_open":
            # Single failure in half-open re-opens
            self._open_circuit()

    def record_success(self, transport: str, target: str) -> None:
        """Record a success."""
        self._state.total_successes += 1
        self._state.last_success_time = time.time()

        if self._state.state == "half_open":
            self._half_open_successes += 1
            if self._half_open_successes >= self._success_threshold:
                self._close_circuit()

    def _open_circuit(self) -> None:
        """Open the circuit breaker."""
        if self._state.state != "open":
            self._state.state = "open"
            self._state.opened_at = time.time()
            logger.critical(
                f"Global circuit breaker OPENED. "
                f"Total failures: {self._state.total_failures}"
            )

            # Trigger emergency notifications
            if self._notification_callback:
                asyncio.create_task(self._emit_emergency_alerts())

    def _close_circuit(self) -> None:
        """Close the circuit breaker."""
        self._state.state = "closed"
        self._state.total_failures = 0
        self._half_open_successes = 0
        logger.info("Global circuit breaker CLOSED")

    async def _emit_emergency_alerts(self) -> None:
        """Send alerts via every possible channel."""
        worst_transport = max(
            self._state.failures_by_transport.items(),
            key=lambda x: x[1],
            default=("unknown", 0),
        )
        worst_target = max(
            self._state.failures_by_target.items(),
            key=lambda x: x[1],
            default=("unknown", 0),
        )

        message = (
            f"CRITICAL: Global circuit breaker opened. "
            f"Total failures: {self._state.total_failures}. "
            f"Worst transport: {worst_transport[0]} ({worst_transport[1]} failures). "
            f"Worst target: {worst_target[0]} ({worst_target[1]} failures)."
        )

        if self._notification_callback:
            try:
                await self._notification_callback(message)
            except Exception as e:
                logger.error(f"Failed to send emergency alert: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self._state.state,
            "total_failures": self._state.total_failures,
            "total_successes": self._state.total_successes,
            "failures_by_transport": dict(self._state.failures_by_transport),
            "failures_by_target": dict(self._state.failures_by_target),
            "last_success_time": self._state.last_success_time,
            "last_failure_time": self._state.last_failure_time,
        }


# Singleton instances
_cascade: TransportCascade | None = None
_circuit_breaker: GlobalCircuitBreaker | None = None


def get_transport_cascade() -> TransportCascade:
    """Get the singleton TransportCascade instance."""
    global _cascade
    if _cascade is None:
        _cascade = TransportCascade()
    return _cascade


def get_global_circuit_breaker() -> GlobalCircuitBreaker:
    """Get the singleton GlobalCircuitBreaker instance."""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = GlobalCircuitBreaker()
    return _circuit_breaker


def reset_singletons() -> None:
    """Reset singletons (for testing)."""
    global _cascade, _circuit_breaker
    _cascade = None
    _circuit_breaker = None
