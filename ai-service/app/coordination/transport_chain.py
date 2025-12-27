#!/usr/bin/env python3
"""Transport Chain Coordinator for Multi-Transport Failover.

Manages multiple transport implementations with intelligent failover,
shared circuit breaker state, and unified health tracking.

Usage:
    from app.coordination.transport_chain import TransportChain
    from app.coordination.transport_base import TransportBase

    # Define transports in priority order
    chain = TransportChain([
        TailscaleTransport(),  # Try first
        SSHTransport(),        # Fallback
        HTTPTransport(),       # Last resort
    ])

    # Execute with automatic failover
    result = await chain.transfer(source, dest, target)

    # Check chain health
    health = chain.health_check()

December 2025: Consolidates transport coordination from:
- cluster_transport.py (multi-transport failover)
- hybrid_transport.py (priority-based routing)
- resilient_transfer.py (retry coordination)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from app.coordination.retry_strategies import (
    ExponentialBackoffStrategy,
    RetryContext,
    RetryStrategy,
)
from app.coordination.transport_base import (
    CircuitBreakerConfig,
    TargetStatus,
    TimeoutConfig,
    TransportBase,
    TransportError,
    TransportResult,
    TransportState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


class FailoverMode(Enum):
    """How to handle transport failures."""

    SEQUENTIAL = "sequential"  # Try each in order until success
    PARALLEL = "parallel"  # Try all simultaneously, use first success
    PRIORITY = "priority"  # Use priority scores to order attempts
    ADAPTIVE = "adaptive"  # Adjust order based on recent success rates


@dataclass
class TransportMetrics:
    """Metrics for a single transport in the chain."""

    transport_name: str
    total_attempts: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    total_latency_ms: float = 0.0
    last_attempt_time: float = 0.0
    last_success_time: float = 0.0
    priority_score: float = 1.0  # Higher = more preferred

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successes / self.total_attempts if self.total_attempts > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        return self.total_latency_ms / self.successes if self.successes > 0 else 0.0


@dataclass
class ChainResult:
    """Result of a chain transfer operation."""

    success: bool
    result: TransportResult | None = None
    transport_used: str = ""
    transports_attempted: list[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    error: str | None = None

    @classmethod
    def from_transport_result(
        cls,
        result: TransportResult,
        transports_attempted: list[str],
        total_latency: float,
    ) -> ChainResult:
        """Create from a successful TransportResult."""
        return cls(
            success=result.success,
            result=result,
            transport_used=result.transport_used,
            transports_attempted=transports_attempted,
            total_latency_ms=total_latency,
            error=result.error if not result.success else None,
        )

    @classmethod
    def failure(
        cls,
        error: str,
        transports_attempted: list[str],
        total_latency: float,
    ) -> ChainResult:
        """Create a failure result."""
        return cls(
            success=False,
            transports_attempted=transports_attempted,
            total_latency_ms=total_latency,
            error=error,
        )


# =============================================================================
# Transport Chain
# =============================================================================


class TransportChain:
    """Manages multi-transport failover with shared state.

    Coordinates multiple transport implementations, trying each in
    order (or based on configured strategy) until one succeeds.
    Maintains shared circuit breaker state across all transports.

    Features:
    - Sequential, parallel, priority, or adaptive failover modes
    - Shared circuit breaker for per-target tracking
    - Metrics collection for each transport
    - Configurable retry strategy per operation
    - Health check aggregation across all transports
    """

    def __init__(
        self,
        transports: list[TransportBase],
        failover_mode: FailoverMode = FailoverMode.SEQUENTIAL,
        retry_strategy: RetryStrategy | None = None,
        timeout_config: TimeoutConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        """Initialize the transport chain.

        Args:
            transports: List of transports in priority order
            failover_mode: How to handle transport failures
            retry_strategy: Optional retry strategy for failed operations
            timeout_config: Timeout configuration
            circuit_breaker_config: Circuit breaker configuration
        """
        if not transports:
            raise ValueError("At least one transport is required")

        self._transports = transports
        self._failover_mode = failover_mode
        self._retry_strategy = retry_strategy or ExponentialBackoffStrategy(
            max_retries=2,
            base_delay=0.5,
            max_delay=10.0,
        )

        # Timeout and circuit breaker config
        self._timeout_config = timeout_config or TimeoutConfig()
        self._cb_config = circuit_breaker_config or CircuitBreakerConfig()

        # Shared circuit breaker state (per target, across all transports)
        self._shared_status: dict[str, TargetStatus] = {}

        # Per-transport metrics
        self._transport_metrics: dict[str, TransportMetrics] = {
            t.name: TransportMetrics(transport_name=t.name)
            for t in transports
        }

        # Callbacks
        self._on_transport_success: list[Callable[[str, str, float], None]] = []
        self._on_transport_failure: list[Callable[[str, str, Exception], None]] = []

    @property
    def transport_names(self) -> list[str]:
        """Get names of all transports in the chain."""
        return [t.name for t in self._transports]

    # =========================================================================
    # Main Transfer Method
    # =========================================================================

    async def transfer(
        self,
        source: str | Path,
        destination: str | Path,
        target: str,
        **kwargs,
    ) -> ChainResult:
        """Perform transfer with automatic failover.

        Args:
            source: Source path or URL
            destination: Destination path or URL
            target: Target identifier for circuit breaker
            **kwargs: Transport-specific options

        Returns:
            ChainResult with success status and metadata
        """
        # Check shared circuit breaker
        if not self._can_attempt_target(target):
            return ChainResult.failure(
                error=f"Circuit open for target: {target}",
                transports_attempted=[],
                total_latency=0.0,
            )

        # Execute based on failover mode
        if self._failover_mode == FailoverMode.SEQUENTIAL:
            return await self._transfer_sequential(source, destination, target, **kwargs)
        elif self._failover_mode == FailoverMode.PARALLEL:
            return await self._transfer_parallel(source, destination, target, **kwargs)
        elif self._failover_mode == FailoverMode.PRIORITY:
            return await self._transfer_priority(source, destination, target, **kwargs)
        elif self._failover_mode == FailoverMode.ADAPTIVE:
            return await self._transfer_adaptive(source, destination, target, **kwargs)
        else:
            # Default to sequential
            return await self._transfer_sequential(source, destination, target, **kwargs)

    async def _transfer_sequential(
        self,
        source: str | Path,
        destination: str | Path,
        target: str,
        **kwargs,
    ) -> ChainResult:
        """Try each transport in order until one succeeds."""
        start_time = time.time()
        transports_attempted = []
        last_error: str | None = None

        for transport in self._transports:
            if not transport.can_attempt(target):
                logger.debug(
                    f"[TransportChain] Skipping {transport.name} - circuit open for {target}"
                )
                continue

            transports_attempted.append(transport.name)

            try:
                result = await self._execute_with_metrics(
                    transport, source, destination, target, **kwargs
                )

                if result.success:
                    self._record_shared_success(target)
                    total_latency = (time.time() - start_time) * 1000
                    return ChainResult.from_transport_result(
                        result, transports_attempted, total_latency
                    )

                last_error = result.error

            except TransportError as e:
                last_error = str(e)
                logger.debug(f"[TransportChain] {transport.name} failed: {e}")

            except asyncio.TimeoutError:
                last_error = f"Timeout on {transport.name}"
                self._transport_metrics[transport.name].timeouts += 1

            except (OSError, ValueError, TypeError, RuntimeError) as e:
                last_error = str(e)
                logger.debug(f"[TransportChain] {transport.name} error: {e}")

        # All transports failed
        self._record_shared_failure(target)
        total_latency = (time.time() - start_time) * 1000

        return ChainResult.failure(
            error=last_error or "All transports failed",
            transports_attempted=transports_attempted,
            total_latency=total_latency,
        )

    async def _transfer_parallel(
        self,
        source: str | Path,
        destination: str | Path,
        target: str,
        **kwargs,
    ) -> ChainResult:
        """Try all transports simultaneously, use first success."""
        start_time = time.time()
        available = [t for t in self._transports if t.can_attempt(target)]

        if not available:
            return ChainResult.failure(
                error="No transports available",
                transports_attempted=[],
                total_latency=0.0,
            )

        # Create tasks for all available transports
        tasks = [
            asyncio.create_task(
                self._execute_with_metrics(t, source, destination, target, **kwargs)
            )
            for t in available
        ]

        transports_attempted = [t.name for t in available]

        # Wait for first successful result
        done: set[asyncio.Task] = set()
        pending = set(tasks)

        while pending:
            done_now, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            done.update(done_now)

            for task in done_now:
                try:
                    result = task.result()
                    if result.success:
                        # Cancel remaining tasks
                        for t in pending:
                            t.cancel()

                        self._record_shared_success(target)
                        total_latency = (time.time() - start_time) * 1000

                        return ChainResult.from_transport_result(
                            result, transports_attempted, total_latency
                        )
                except (asyncio.CancelledError, TransportError, OSError):
                    continue

        # All failed
        self._record_shared_failure(target)
        total_latency = (time.time() - start_time) * 1000

        return ChainResult.failure(
            error="All parallel transports failed",
            transports_attempted=transports_attempted,
            total_latency=total_latency,
        )

    async def _transfer_priority(
        self,
        source: str | Path,
        destination: str | Path,
        target: str,
        **kwargs,
    ) -> ChainResult:
        """Try transports ordered by priority score."""
        # Sort by priority (higher first), then by success rate
        sorted_transports = sorted(
            self._transports,
            key=lambda t: (
                self._transport_metrics[t.name].priority_score,
                self._transport_metrics[t.name].success_rate,
            ),
            reverse=True,
        )

        # Use sequential logic with sorted order
        original_order = self._transports
        self._transports = sorted_transports

        try:
            return await self._transfer_sequential(source, destination, target, **kwargs)
        finally:
            self._transports = original_order

    async def _transfer_adaptive(
        self,
        source: str | Path,
        destination: str | Path,
        target: str,
        **kwargs,
    ) -> ChainResult:
        """Adaptively order transports based on recent success rates."""
        # Sort by recent success rate
        sorted_transports = sorted(
            self._transports,
            key=lambda t: (
                self._transport_metrics[t.name].success_rate,
                -self._transport_metrics[t.name].avg_latency_ms,
            ),
            reverse=True,
        )

        original_order = self._transports
        self._transports = sorted_transports

        try:
            return await self._transfer_sequential(source, destination, target, **kwargs)
        finally:
            self._transports = original_order

    # =========================================================================
    # Metrics and Execution
    # =========================================================================

    async def _execute_with_metrics(
        self,
        transport: TransportBase,
        source: str | Path,
        destination: str | Path,
        target: str,
        **kwargs,
    ) -> TransportResult:
        """Execute transfer on a single transport with metrics tracking."""
        metrics = self._transport_metrics[transport.name]
        metrics.total_attempts += 1
        metrics.last_attempt_time = time.time()

        start = time.time()

        try:
            result = await asyncio.wait_for(
                transport.transfer(source, destination, target, **kwargs),
                timeout=self._timeout_config.operation_timeout,
            )

            latency_ms = (time.time() - start) * 1000

            if result.success:
                metrics.successes += 1
                metrics.total_latency_ms += latency_ms
                metrics.last_success_time = time.time()
                transport.record_success(target, latency_ms)

                # Notify callbacks
                for callback in self._on_transport_success:
                    try:
                        callback(transport.name, target, latency_ms)
                    except (TypeError, RuntimeError):
                        pass
            else:
                metrics.failures += 1
                transport.record_failure(target)

            return result

        except asyncio.TimeoutError:
            metrics.timeouts += 1
            transport.record_failure(target)
            raise

        except Exception as e:
            metrics.failures += 1
            transport.record_failure(target, e)

            # Notify callbacks
            for callback in self._on_transport_failure:
                try:
                    callback(transport.name, target, e)
                except (TypeError, RuntimeError):
                    pass

            raise TransportError(
                message=str(e),
                transport=transport.name,
                target=target,
                cause=e,
            ) from e

    # =========================================================================
    # Shared Circuit Breaker
    # =========================================================================

    def _can_attempt_target(self, target: str) -> bool:
        """Check shared circuit breaker for target."""
        status = self._shared_status.get(target)

        if status is None:
            return True

        if status.state == TransportState.CLOSED:
            return True

        if status.state == TransportState.OPEN:
            # Check recovery timeout
            elapsed = time.time() - status.last_failure_time
            if elapsed >= self._cb_config.recovery_timeout:
                status.state = TransportState.HALF_OPEN
                return True
            return False

        return status.state == TransportState.HALF_OPEN

    def _record_shared_success(self, target: str) -> None:
        """Record success in shared circuit breaker."""
        if target not in self._shared_status:
            self._shared_status[target] = TargetStatus()

        status = self._shared_status[target]
        status.state = TransportState.CLOSED
        status.success_count += 1
        status.failure_count = 0
        status.last_success_time = time.time()

    def _record_shared_failure(self, target: str) -> None:
        """Record failure in shared circuit breaker."""
        if target not in self._shared_status:
            self._shared_status[target] = TargetStatus()

        status = self._shared_status[target]
        status.failure_count += 1
        status.last_failure_time = time.time()

        if status.failure_count >= self._cb_config.failure_threshold:
            status.state = TransportState.OPEN
            logger.warning(
                f"[TransportChain] Shared circuit opened for {target} "
                f"(failures={status.failure_count})"
            )

    # =========================================================================
    # Priority Management
    # =========================================================================

    def set_transport_priority(self, transport_name: str, priority: float) -> None:
        """Set priority score for a transport.

        Args:
            transport_name: Name of the transport
            priority: Priority score (higher = more preferred)
        """
        if transport_name in self._transport_metrics:
            self._transport_metrics[transport_name].priority_score = priority

    def boost_transport(self, transport_name: str, factor: float = 1.5) -> None:
        """Temporarily boost a transport's priority."""
        if transport_name in self._transport_metrics:
            current = self._transport_metrics[transport_name].priority_score
            self._transport_metrics[transport_name].priority_score = current * factor

    def penalize_transport(self, transport_name: str, factor: float = 0.5) -> None:
        """Temporarily reduce a transport's priority."""
        if transport_name in self._transport_metrics:
            current = self._transport_metrics[transport_name].priority_score
            self._transport_metrics[transport_name].priority_score = current * factor

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_success(self, callback: Callable[[str, str, float], None]) -> None:
        """Register callback for successful transfers.

        Args:
            callback: Function(transport_name, target, latency_ms)
        """
        self._on_transport_success.append(callback)

    def on_failure(self, callback: Callable[[str, str, Exception], None]) -> None:
        """Register callback for failed transfers.

        Args:
            callback: Function(transport_name, target, exception)
        """
        self._on_transport_failure.append(callback)

    # =========================================================================
    # Health and Statistics
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Check health of all transports in the chain.

        Returns:
            Aggregated health status
        """
        transport_health = {}
        total_success = 0
        total_attempts = 0

        for transport in self._transports:
            health = transport.health_check()
            transport_health[transport.name] = health

            metrics = self._transport_metrics[transport.name]
            total_success += metrics.successes
            total_attempts += metrics.total_attempts

        # Aggregate status
        healthy_count = sum(
            1 for h in transport_health.values() if h.get("healthy", False)
        )
        all_healthy = healthy_count == len(self._transports)

        success_rate = total_success / total_attempts if total_attempts > 0 else 1.0

        return {
            "healthy": healthy_count > 0,
            "status": "healthy" if all_healthy else ("degraded" if healthy_count > 0 else "unhealthy"),
            "message": f"{healthy_count}/{len(self._transports)} transports healthy",
            "details": {
                "transports": transport_health,
                "failover_mode": self._failover_mode.value,
                "shared_circuits": {
                    target: {
                        "state": status.state.value,
                        "failures": status.failure_count,
                    }
                    for target, status in self._shared_status.items()
                },
                "overall_success_rate": round(success_rate, 3),
            },
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics for all transports."""
        return {
            "failover_mode": self._failover_mode.value,
            "transports": {
                name: {
                    "attempts": m.total_attempts,
                    "successes": m.successes,
                    "failures": m.failures,
                    "timeouts": m.timeouts,
                    "success_rate": round(m.success_rate, 3),
                    "avg_latency_ms": round(m.avg_latency_ms, 1),
                    "priority_score": m.priority_score,
                }
                for name, m in self._transport_metrics.items()
            },
            "shared_circuits": len(self._shared_status),
            "open_circuits": sum(
                1
                for s in self._shared_status.values()
                if s.state == TransportState.OPEN
            ),
        }

    def reset_statistics(self) -> None:
        """Reset all metrics and circuit breakers."""
        for metrics in self._transport_metrics.values():
            metrics.total_attempts = 0
            metrics.successes = 0
            metrics.failures = 0
            metrics.timeouts = 0
            metrics.total_latency_ms = 0.0
            metrics.priority_score = 1.0

        self._shared_status.clear()

        for transport in self._transports:
            transport.reset_all_circuits()


# =============================================================================
# Factory Functions
# =============================================================================


def create_sequential_chain(transports: list[TransportBase]) -> TransportChain:
    """Create a chain with sequential failover."""
    return TransportChain(transports, failover_mode=FailoverMode.SEQUENTIAL)


def create_parallel_chain(transports: list[TransportBase]) -> TransportChain:
    """Create a chain with parallel execution."""
    return TransportChain(transports, failover_mode=FailoverMode.PARALLEL)


def create_adaptive_chain(transports: list[TransportBase]) -> TransportChain:
    """Create a chain with adaptive ordering."""
    return TransportChain(transports, failover_mode=FailoverMode.ADAPTIVE)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Main class
    "TransportChain",
    # Data classes
    "ChainResult",
    "TransportMetrics",
    "FailoverMode",
    # Factory functions
    "create_sequential_chain",
    "create_parallel_chain",
    "create_adaptive_chain",
]
