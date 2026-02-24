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

from app.core.async_context import safe_create_task
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# =============================================================================
# Per-Transport Circuit Breaker Integration (January 2026 - Phase 2)
# =============================================================================
# Import per-transport circuit breaker functions to enable transport-level failover
try:
    from scripts.p2p.network import (
        check_peer_transport_circuit,
        record_peer_transport_success,
        record_peer_transport_failure,
        HAS_TRANSPORT_BREAKER,
    )
except ImportError:
    # Graceful fallback if network module not available
    HAS_TRANSPORT_BREAKER = False

    def check_peer_transport_circuit(peer_host: str, transport: str = "http") -> bool:
        return True  # Allow all if breakers not available

    def record_peer_transport_success(peer_host: str, transport: str = "http") -> None:
        pass

    def record_peer_transport_failure(
        peer_host: str, transport: str = "http", error: Exception | None = None
    ) -> None:
        pass


# =============================================================================
# NAT-Aware Transport Selection (January 19, 2026)
# =============================================================================
# Import NAT detection for adaptive transport ordering
try:
    from scripts.p2p.nat_detection import (
        NATType,
        get_cached_nat_type,
        get_recommended_transport_order,
        is_nat_traversal_difficult,
    )

    HAS_NAT_DETECTION = True
except ImportError:
    HAS_NAT_DETECTION = False
    NATType = None  # type: ignore

    def get_cached_nat_type():
        return None

    def get_recommended_transport_order(nat_type):
        return []

    def is_nat_traversal_difficult(nat_type):
        return False


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


class AdaptiveTimeoutTracker:
    """Learn per-peer timeouts from historical latency.

    Jan 2026: Part of Phase 2.3 - Adaptive Timeout Learning.

    Uses a rolling window of latency samples to compute adaptive timeouts:
    - Target: p95 latency * multiplier (default 1.5)
    - Clamped to [min_timeout, max_timeout] range

    This prevents:
    - Fast peers waiting too long for responses
    - Slow peers being falsely marked as dead
    """

    def __init__(
        self,
        min_timeout: float = 5.0,
        max_timeout: float = 60.0,
        default_timeout: float = 10.0,
        multiplier: float = 1.5,
        window_size: int = 100,
    ):
        """Initialize adaptive timeout tracker.

        Args:
            min_timeout: Minimum timeout in seconds
            max_timeout: Maximum timeout in seconds
            default_timeout: Timeout for unknown targets
            multiplier: Multiply p95 latency by this factor
            window_size: Number of latency samples to track per target
        """
        self._min_timeout = min_timeout
        self._max_timeout = max_timeout
        self._default_timeout = default_timeout
        self._multiplier = multiplier
        self._window_size = window_size

        # target -> list of latency samples (ms)
        self._latency_samples: dict[str, list[float]] = {}

        # Environment overrides
        self._min_timeout = float(
            os.environ.get("RINGRIFT_ADAPTIVE_TIMEOUT_MIN", str(min_timeout))
        )
        self._max_timeout = float(
            os.environ.get("RINGRIFT_ADAPTIVE_TIMEOUT_MAX", str(max_timeout))
        )
        self._multiplier = float(
            os.environ.get("RINGRIFT_ADAPTIVE_TIMEOUT_MULTIPLIER", str(multiplier))
        )

    def record_latency(self, target: str, latency_ms: float) -> None:
        """Record a successful latency sample.

        Args:
            target: Target identifier (node_id, IP, etc.)
            latency_ms: Observed latency in milliseconds
        """
        if target not in self._latency_samples:
            self._latency_samples[target] = []

        samples = self._latency_samples[target]
        samples.append(latency_ms)

        # Keep only the most recent samples
        if len(samples) > self._window_size:
            self._latency_samples[target] = samples[-self._window_size:]

    def get_timeout(self, target: str) -> float:
        """Get adaptive timeout for a target.

        Returns:
            Timeout in seconds based on historical latency, clamped to valid range.
        """
        samples = self._latency_samples.get(target, [])

        if len(samples) < 3:
            # Not enough data, use default
            return self._default_timeout

        # Calculate p95 latency
        sorted_samples = sorted(samples)
        p95_index = int(len(sorted_samples) * 0.95)
        p95_latency_ms = sorted_samples[p95_index]

        # Convert to seconds and apply multiplier
        timeout_seconds = (p95_latency_ms / 1000.0) * self._multiplier

        # Clamp to valid range
        return min(self._max_timeout, max(self._min_timeout, timeout_seconds))

    def get_stats(self, target: str) -> dict[str, Any]:
        """Get statistics for a target.

        Args:
            target: Target identifier

        Returns:
            Dictionary with latency statistics
        """
        samples = self._latency_samples.get(target, [])

        if not samples:
            return {
                "target": target,
                "sample_count": 0,
                "timeout": self._default_timeout,
            }

        sorted_samples = sorted(samples)
        return {
            "target": target,
            "sample_count": len(samples),
            "min_latency_ms": sorted_samples[0],
            "max_latency_ms": sorted_samples[-1],
            "p50_latency_ms": sorted_samples[len(sorted_samples) // 2],
            "p95_latency_ms": sorted_samples[int(len(sorted_samples) * 0.95)],
            "timeout": self.get_timeout(target),
        }

    def get_all_stats(self) -> list[dict[str, Any]]:
        """Get statistics for all tracked targets."""
        return [self.get_stats(target) for target in self._latency_samples]

    def clear(self, target: str | None = None) -> None:
        """Clear latency samples.

        Args:
            target: Specific target to clear, or None for all targets
        """
        if target is None:
            self._latency_samples.clear()
        elif target in self._latency_samples:
            del self._latency_samples[target]


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

    Sprint 5 (Jan 2, 2026): Added metrics persistence for historical analysis.
    """

    # Class-level state manager for metrics persistence
    _state_manager: Any = None
    _last_metrics_persist: float = 0.0
    _metrics_persist_interval: float = 300.0  # Persist every 5 minutes

    def __init__(self):
        self._transports: list[BaseTransport] = []
        # target -> transport_name -> TransportHealth
        self._health: dict[str, dict[str, TransportHealth]] = {}
        self._global_circuit_breaker: GlobalCircuitBreaker | None = None

        # Configuration from environment
        self._enabled = os.environ.get("RINGRIFT_TRANSPORT_CASCADE_ENABLED", "true").lower() == "true"
        self._min_tier = int(os.environ.get("RINGRIFT_TRANSPORT_MIN_TIER", "1"))
        self._max_tier = int(os.environ.get("RINGRIFT_TRANSPORT_MAX_TIER", "5"))
        # Jan 2026: Reduced from 10s to 5s per transport for faster failover (50% reduction)
        # Jan 5, 2026 (Phase 3): Increased to 8s to reduce false positives on slow providers
        self._timeout_per_transport = float(os.environ.get("RINGRIFT_TRANSPORT_TIMEOUT", "8"))
        # Total timeout for the entire cascade (prevents 360s+ hangs with 6 tiers)
        # Jan 2026: Reduced from 30s to 15s for faster overall failover
        # Jan 5, 2026 (Phase 3): Increased to 25s to allow more transport attempts
        self._total_timeout = float(os.environ.get("RINGRIFT_TRANSPORT_TOTAL_TIMEOUT", "25"))

        # Jan 2026: Adaptive timeout learning (Phase 2.3)
        self._adaptive_timeouts_enabled = (
            os.environ.get("RINGRIFT_ADAPTIVE_TIMEOUTS_ENABLED", "true").lower() == "true"
        )
        self._adaptive_timeout_tracker = AdaptiveTimeoutTracker(
            min_timeout=5.0,
            max_timeout=60.0,
            default_timeout=self._timeout_per_transport,
        )

        # Sprint 5: Load historical metrics on startup
        self._load_metrics_from_persistence()

    @classmethod
    def set_state_manager(cls, state_manager: Any) -> None:
        """Set the state manager for transport metrics persistence.

        Called during P2P orchestrator initialization.

        Args:
            state_manager: StateManager instance
        """
        cls._state_manager = state_manager
        logger.debug("[TransportCascade] State manager configured for metrics persistence")

    def _load_metrics_from_persistence(self) -> int:
        """Load historical transport metrics on startup.

        Returns:
            Number of metrics loaded
        """
        if not self._state_manager:
            return 0

        try:
            metrics = self._state_manager.load_transport_metrics(max_age_seconds=3600.0)
            loaded = 0
            for m in metrics:
                target = m.get("target", "")
                transport_name = m.get("transport_name", "")
                if not target or not transport_name:
                    continue

                health = self.get_health(target, transport_name)
                health.successes = m.get("successes", 0)
                health.failures = m.get("failures", 0)
                health.total_latency_ms = m.get("total_latency_ms", 0.0)
                health.consecutive_failures = m.get("consecutive_failures", 0)
                health.last_success_time = m.get("last_success_time", 0.0)
                health.last_failure_time = m.get("last_failure_time", 0.0)
                loaded += 1

            if loaded > 0:
                logger.info(f"[TransportCascade] Loaded {loaded} historical transport metrics")
            return loaded
        except (AttributeError, TypeError) as e:
            logger.debug(f"[TransportCascade] Failed to load metrics: {e}")
            return 0

    def persist_metrics(self, force: bool = False) -> int:
        """Persist current transport metrics to database.

        Called periodically or on significant events.

        Args:
            force: Force persistence even if interval hasn't elapsed

        Returns:
            Number of metrics persisted
        """
        if not self._state_manager:
            return 0

        now = time.time()
        if not force and (now - self._last_metrics_persist) < self._metrics_persist_interval:
            return 0

        try:
            metrics_list = []
            for target, transports in self._health.items():
                for transport_name, health in transports.items():
                    # Only persist if there's been activity
                    if health.successes + health.failures == 0:
                        continue
                    metrics_list.append({
                        "target": target,
                        "transport_name": transport_name,
                        "successes": health.successes,
                        "failures": health.failures,
                        "total_latency_ms": health.total_latency_ms,
                        "consecutive_failures": health.consecutive_failures,
                        "last_success_time": health.last_success_time,
                        "last_failure_time": health.last_failure_time,
                    })

            if metrics_list:
                saved = self._state_manager.save_transport_metrics_batch(metrics_list)
                self._last_metrics_persist = now
                return saved
            return 0
        except (AttributeError, TypeError) as e:
            logger.debug(f"[TransportCascade] Failed to persist metrics: {e}")
            return 0

    def get_recommended_transport(self, target: str) -> str | None:
        """Get the recommended transport for a target based on historical performance.

        Sprint 5 (Jan 2, 2026): Uses persisted metrics for ranking.

        Args:
            target: Target node identifier

        Returns:
            Best transport name, or None if no data
        """
        if self._state_manager:
            try:
                return self._state_manager.get_best_transport_for_target(target)
            except (AttributeError, TypeError):
                pass

        # Fall back to in-memory metrics
        if target not in self._health:
            return None

        best_transport = None
        best_score = -1.0

        for transport_name, health in self._health[target].items():
            # Score = success_rate - penalty for consecutive failures
            score = health.success_rate - (health.consecutive_failures * 0.1)
            if score > best_score:
                best_score = score
                best_transport = transport_name

        return best_transport

    def get_transport_rankings(self, target: str) -> list[tuple[str, float]]:
        """Get ranked list of transports for a target.

        Sprint 5 (Jan 2, 2026): Returns all transports sorted by score.

        Args:
            target: Target node identifier

        Returns:
            List of (transport_name, score) tuples, sorted best first
        """
        rankings: list[tuple[str, float]] = []

        # Try persisted metrics first
        if self._state_manager and target:
            try:
                metrics = self._state_manager.load_transport_metrics(target=target)
                for m in metrics:
                    name = m.get("transport_name", "")
                    success_rate = m.get("success_rate", 0.5)
                    consec_fail = m.get("consecutive_failures", 0)
                    score = success_rate - (consec_fail * 0.1)
                    rankings.append((name, score))
            except (AttributeError, TypeError):
                pass

        # Fall back to in-memory if no persisted data
        if not rankings and target in self._health:
            for transport_name, health in self._health[target].items():
                score = health.success_rate - (health.consecutive_failures * 0.1)
                rankings.append((transport_name, score))

        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_all_transport_rankings(self) -> dict[str, list[tuple[str, float]]]:
        """Get transport rankings for all known targets.

        Sprint 5 (Jan 2, 2026): Aggregates rankings across all targets
        for cluster-wide transport health visibility.

        Returns:
            Dict mapping target -> list of (transport_name, score) tuples
        """
        all_rankings: dict[str, list[tuple[str, float]]] = {}

        # Get all targets from in-memory health data
        for target in self._health.keys():
            all_rankings[target] = self.get_transport_rankings(target)

        return all_rankings

    def get_transport_health_summary(self) -> dict[str, Any]:
        """Get summary of transport health across all targets.

        Sprint 5 (Jan 2, 2026): Unified metrics view for monitoring.

        Returns:
            Dict with overall transport health statistics
        """
        summary: dict[str, Any] = {
            "targets": len(self._health),
            "transports": {},
            "global_circuit_breaker": None,
        }

        # Aggregate transport stats
        transport_stats: dict[str, dict[str, float]] = {}
        for target, health_by_transport in self._health.items():
            for transport_name, health in health_by_transport.items():
                if transport_name not in transport_stats:
                    transport_stats[transport_name] = {
                        "total_successes": 0,
                        "total_failures": 0,
                        "target_count": 0,
                        "avg_success_rate": 0.0,
                        "total_latency_ms": 0.0,
                    }
                stats = transport_stats[transport_name]
                stats["total_successes"] += health.successes
                stats["total_failures"] += health.failures
                stats["target_count"] += 1
                stats["total_latency_ms"] += health.total_latency_ms

        # Calculate averages
        for name, stats in transport_stats.items():
            total = stats["total_successes"] + stats["total_failures"]
            if total > 0:
                stats["avg_success_rate"] = stats["total_successes"] / total
            if stats["total_successes"] > 0:
                stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_successes"]
            else:
                stats["avg_latency_ms"] = 0.0

        summary["transports"] = transport_stats

        # Include global circuit breaker state
        if self._global_circuit_breaker:
            summary["global_circuit_breaker"] = {
                "open": self._global_circuit_breaker.is_open(),
                "failure_count": self._global_circuit_breaker._failure_count,
                "last_failure_time": self._global_circuit_breaker._last_failure_time,
            }

        return summary

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

    # Jan 5, 2026: Stagger delay between starting each tier in staggered probe mode
    # This reduces failover time from 50s (sequential) to ~15s (staggered)
    STAGGER_DELAY = float(os.environ.get("RINGRIFT_CASCADE_STAGGER_DELAY", "5.0"))

    async def send_with_cascade(
        self,
        target: str,
        payload: bytes,
        *,
        min_tier: TransportTier | None = None,
        max_tier: TransportTier | None = None,
        timeout_per_transport: float | None = None,
        total_timeout: float | None = None,
        parallel_probe: bool = False,
        staggered_probe: bool = False,
    ) -> TransportResult:
        """
        Try all transports in tier order until one succeeds.

        Args:
            target: Target node identifier (node_id or address)
            payload: Data to send
            min_tier: Minimum tier to try (default: TIER_1_FAST)
            max_tier: Maximum tier to try (default: TIER_5_EXTERNAL)
            timeout_per_transport: Timeout per transport attempt
            total_timeout: Maximum total time for entire cascade (default: 30s)
            parallel_probe: If True, try all transports in each tier simultaneously
            staggered_probe: If True, start lower tiers with a delay while higher
                tiers are still probing. Reduces failover time from 50s to ~15s.
                (Jan 5, 2026: New mode for faster failover)

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
        effective_total_timeout = total_timeout or self._total_timeout

        # Track cascade start time for total timeout enforcement
        cascade_start_time = time.time()

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

        # Jan 19, 2026: Apply NAT-aware ordering for CGNAT bypass
        # This prioritizes P2PD hole punching when behind difficult NAT
        sorted_transports = self._apply_nat_aware_ordering(sorted_transports)

        # Jan 5, 2026: Staggered probe takes priority over parallel probe
        if staggered_probe:
            return await self._cascade_staggered(
                target, payload, sorted_transports, effective_timeout,
                cascade_start_time, effective_total_timeout,
            )
        elif parallel_probe:
            return await self._cascade_parallel(
                target, payload, sorted_transports, effective_timeout,
                cascade_start_time, effective_total_timeout,
            )
        else:
            return await self._cascade_sequential(
                target, payload, sorted_transports, effective_timeout,
                cascade_start_time, effective_total_timeout,
            )

    async def _cascade_sequential(
        self,
        target: str,
        payload: bytes,
        transports: list[BaseTransport],
        timeout: float,
        cascade_start_time: float,
        total_timeout: float,
    ) -> TransportResult:
        """Try transports sequentially with total timeout enforcement."""
        errors = []

        # Jan 2026: Use adaptive timeout if enabled and we have historical data
        effective_timeout = timeout
        if self._adaptive_timeouts_enabled:
            adaptive_timeout = self._adaptive_timeout_tracker.get_timeout(target)
            # Use the larger of fixed and adaptive to be safe during learning
            effective_timeout = max(timeout, adaptive_timeout)

        for transport in transports:
            # Check total timeout before each attempt
            elapsed = time.time() - cascade_start_time
            remaining = total_timeout - elapsed
            if remaining <= 0:
                logger.debug(
                    f"[TransportCascade] Total timeout ({total_timeout}s) exceeded after {elapsed:.1f}s"
                )
                return TransportResult(
                    success=False,
                    transport_name="total_timeout_exceeded",
                    latency_ms=elapsed * 1000,
                    error=f"Total cascade timeout ({total_timeout}s) exceeded after trying {len(errors)} transports",
                )

            # Clamp per-transport timeout to remaining time
            clamped_timeout = min(effective_timeout, remaining)
            # Jan 2026 Phase 2: Check per-transport circuit breaker BEFORE attempting
            # This enables transport-level failover - skip transports with OPEN circuits
            if not check_peer_transport_circuit(target, transport.name):
                errors.append(f"{transport.name}: circuit OPEN (skipped)")
                logger.debug(
                    f"[TransportCascade] Skipping {transport.name} -> {target}: "
                    f"circuit breaker is OPEN"
                )
                continue

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
                    timeout=clamped_timeout,
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
                errors.append(f"{transport.name}: timeout ({clamped_timeout:.1f}s)")

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
        cascade_start_time: float,
        total_timeout: float,
    ) -> TransportResult:
        """Try all transports in parallel within each tier with total timeout enforcement."""
        # Group by tier
        by_tier: dict[TransportTier, list[BaseTransport]] = {}
        for t in transports:
            by_tier.setdefault(t.tier, []).append(t)

        errors = []

        # Jan 2026: Use adaptive timeout if enabled and we have historical data
        effective_timeout = timeout
        if self._adaptive_timeouts_enabled:
            adaptive_timeout = self._adaptive_timeout_tracker.get_timeout(target)
            # Use the larger of fixed and adaptive to be safe during learning
            effective_timeout = max(timeout, adaptive_timeout)

        # Try each tier
        for tier in sorted(by_tier.keys(), key=lambda t: t.value):
            # Check total timeout before each tier
            elapsed = time.time() - cascade_start_time
            remaining = total_timeout - elapsed
            if remaining <= 0:
                logger.debug(
                    f"[TransportCascade] Total timeout ({total_timeout}s) exceeded after {elapsed:.1f}s"
                )
                return TransportResult(
                    success=False,
                    transport_name="total_timeout_exceeded",
                    latency_ms=elapsed * 1000,
                    error=f"Total cascade timeout ({total_timeout}s) exceeded after trying {len(errors)} transports",
                )

            # Clamp per-transport timeout to remaining time
            clamped_timeout = min(effective_timeout, remaining)
            tier_transports = by_tier[tier]

            # Create tasks for all transports in this tier
            tasks = []
            for transport in tier_transports:
                task = safe_create_task(
                    self._try_transport(target, payload, transport, clamped_timeout),
                    name=f"cascade-transport-{transport.name}",
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

    async def _cascade_staggered(
        self,
        target: str,
        payload: bytes,
        transports: list[BaseTransport],
        timeout: float,
        cascade_start_time: float,
        total_timeout: float,
    ) -> TransportResult:
        """Try transports with staggered tier starts for faster failover.

        Jan 5, 2026: New probing mode that starts each tier with a delay
        while previous tiers are still trying. This reduces failover time
        from 50s (sequential) to ~15s by overlapping tier attempts.

        How it works:
        - Tier 1 starts immediately at T+0s
        - Tier 2 starts at T+5s (while Tier 1 may still be probing)
        - Tier 3 starts at T+10s (while earlier tiers may still be probing)
        - First successful response cancels all pending tasks

        This is particularly effective for NAT-blocked nodes where Tier 1-2
        (direct/Tailscale) typically timeout after 10s, but we can start
        Tier 4 (relay) probing at T+15s to reduce total failover time.
        """
        # Group transports by tier
        by_tier: dict[TransportTier, list[BaseTransport]] = {}
        for t in transports:
            by_tier.setdefault(t.tier, []).append(t)

        if not by_tier:
            return TransportResult(
                success=False,
                transport_name="no_transports",
                latency_ms=0,
                error="No transports available for staggered probe",
            )

        # Jan 2026: Use adaptive timeout if enabled
        effective_timeout = timeout
        if self._adaptive_timeouts_enabled:
            adaptive_timeout = self._adaptive_timeout_tracker.get_timeout(target)
            effective_timeout = max(timeout, adaptive_timeout)

        sorted_tiers = sorted(by_tier.keys(), key=lambda t: t.value)
        errors: list[str] = []
        all_tasks: list[asyncio.Task] = []
        tier_start_tasks: list[asyncio.Task] = []

        # Create a result event to signal first success
        result_event = asyncio.Event()
        successful_result: list[TransportResult] = []  # Use list for mutability in nested function

        async def probe_tier(tier: TransportTier, delay: float) -> None:
            """Probe all transports in a tier after the specified delay."""
            nonlocal errors

            if delay > 0:
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return

            # Check if we already have a successful result
            if result_event.is_set():
                return

            # Check total timeout
            elapsed = time.time() - cascade_start_time
            remaining = total_timeout - elapsed
            if remaining <= 0:
                return

            tier_transports = by_tier[tier]
            clamped_timeout = min(effective_timeout, remaining)

            # Create tasks for all transports in this tier
            tasks = []
            for transport in tier_transports:
                task = safe_create_task(
                    self._try_transport(target, payload, transport, clamped_timeout),
                    name=f"cascade-stagger-{transport.name}",
                )
                tasks.append(task)
                all_tasks.append(task)

            # Wait for any result from this tier
            pending = set(tasks)
            while pending and not result_event.is_set():
                try:
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=0.1,  # Check result_event periodically
                    )
                except asyncio.CancelledError:
                    for p in pending:
                        p.cancel()
                    return

                for task in done:
                    try:
                        result = task.result()
                        if result.success:
                            successful_result.append(result)
                            result_event.set()
                            return
                        else:
                            errors.append(f"{result.transport_name}: {result.error}")
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        errors.append(f"tier_{tier.name}: {type(e).__name__}: {e}")

        # Start each tier with staggered delay
        for tier_idx, tier in enumerate(sorted_tiers):
            delay = tier_idx * self.STAGGER_DELAY
            task = safe_create_task(probe_tier(tier, delay), name=f"cascade-probe-tier-{tier.name}")
            tier_start_tasks.append(task)

        # Wait for first success or total timeout
        try:
            await asyncio.wait_for(
                result_event.wait(),
                timeout=total_timeout - (time.time() - cascade_start_time),
            )
        except asyncio.TimeoutError:
            pass  # Continue to cleanup and return failure

        # Cancel all remaining tasks
        for task in all_tasks:
            if not task.done():
                task.cancel()
        for task in tier_start_tasks:
            if not task.done():
                task.cancel()

        # Wait for cancellation to complete (brief)
        await asyncio.gather(*tier_start_tasks, *all_tasks, return_exceptions=True)

        # Return successful result or failure
        if successful_result:
            logger.debug(
                f"[TransportCascade] Staggered probe succeeded via "
                f"{successful_result[0].transport_name} in "
                f"{(time.time() - cascade_start_time) * 1000:.0f}ms"
            )
            return successful_result[0]

        elapsed = (time.time() - cascade_start_time) * 1000
        return TransportResult(
            success=False,
            transport_name="staggered_cascade_exhausted",
            latency_ms=elapsed,
            error=f"All staggered probes failed ({len(sorted_tiers)} tiers, "
                  f"{len(transports)} transports): {'; '.join(errors[:5])}"
                  f"{'...' if len(errors) > 5 else ''}",
        )

    async def _try_transport(
        self,
        target: str,
        payload: bytes,
        transport: BaseTransport,
        timeout: float,
    ) -> TransportResult:
        """Helper to try a single transport with error handling."""
        # Jan 2026 Phase 2: Check per-transport circuit breaker first
        if not check_peer_transport_circuit(target, transport.name):
            return TransportResult(
                success=False,
                transport_name=transport.name,
                latency_ms=0,
                error="circuit OPEN",
            )

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

    def _apply_nat_aware_ordering(
        self, transports: list[BaseTransport]
    ) -> list[BaseTransport]:
        """Reorder transports based on local NAT type.

        Jan 19, 2026: Added for CGNAT bypass optimization. When we detect
        that our node is behind CGNAT (common on Vast.ai), we prioritize
        P2PD UDP hole punching over transports that won't work well.

        Args:
            transports: List of transports sorted by tier/health

        Returns:
            Reordered list optimized for local NAT environment
        """
        if not HAS_NAT_DETECTION:
            return transports

        # Get cached NAT type (won't block - returns UNKNOWN if not cached)
        nat_type = get_cached_nat_type()
        if nat_type is None:
            return transports

        # Get recommended transport order for this NAT type
        recommended_order = get_recommended_transport_order(nat_type)
        if not recommended_order:
            return transports

        # Build name-to-transport mapping
        transport_by_name = {t.name: t for t in transports}

        # Reorder based on NAT-aware recommendations
        result = []

        # First, add transports in recommended order (if they exist)
        for name in recommended_order:
            if name in transport_by_name:
                result.append(transport_by_name.pop(name))

        # Then add remaining transports in their original order
        for t in transports:
            if t.name in transport_by_name:
                result.append(t)

        # Log if NAT-aware reordering changed anything
        if is_nat_traversal_difficult(nat_type):
            original_names = [t.name for t in transports[:3]]
            new_names = [t.name for t in result[:3]]
            if original_names != new_names:
                logger.debug(
                    f"NAT-aware reorder for {nat_type.value}: "
                    f"{original_names} -> {new_names}"
                )

        return result

    def _record_success(self, target: str, transport_name: str, latency_ms: float) -> None:
        """Record successful transport attempt."""
        health = self.get_health(target, transport_name)
        health.record_success(latency_ms)

        if self._global_circuit_breaker:
            self._global_circuit_breaker.record_success(transport_name, target)

        # Jan 2026 Phase 2: Record success to per-transport circuit breaker
        # This helps the circuit transition from half-open to closed
        record_peer_transport_success(target, transport_name)

        # Jan 2026: Record latency for adaptive timeout learning
        if self._adaptive_timeouts_enabled:
            self._adaptive_timeout_tracker.record_latency(target, latency_ms)

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

        # Jan 2026 Phase 2: Record failure to per-transport circuit breaker
        # This may trip the circuit, enabling failover to other transports
        record_peer_transport_failure(target, transport_name)

        logger.debug(
            f"Transport failure: {transport_name} -> {target} "
            f"(consecutive={health.consecutive_failures}, rate={health.success_rate:.2f})"
        )

    def get_adaptive_timeout_stats(self) -> dict[str, Any]:
        """Get adaptive timeout learning statistics.

        Returns:
            Dictionary with adaptive timeout stats per target.
        """
        return {
            "enabled": self._adaptive_timeouts_enabled,
            "targets": self._adaptive_timeout_tracker.get_all_stats(),
            "config": {
                "min_timeout": self._adaptive_timeout_tracker._min_timeout,
                "max_timeout": self._adaptive_timeout_tracker._max_timeout,
                "default_timeout": self._adaptive_timeout_tracker._default_timeout,
                "multiplier": self._adaptive_timeout_tracker._multiplier,
            },
        }

    def get_adaptive_timeout(self, target: str) -> float:
        """Get the adaptive timeout for a specific target.

        Args:
            target: Target identifier

        Returns:
            Adaptive timeout in seconds
        """
        if not self._adaptive_timeouts_enabled:
            return self._timeout_per_transport
        return self._adaptive_timeout_tracker.get_timeout(target)

    def get_transport_latency_summary(self) -> dict[str, Any]:
        """Get latency summary aggregated by transport type.

        Jan 3, 2026: Added for /status endpoint transport latency visibility.
        Enables diagnosis of slow transports across all targets.

        Returns:
            Dictionary with per-transport latency stats:
            {
                "by_transport": {
                    "http": {"avg_latency_ms": 45.2, "success_rate": 0.98, ...},
                    "ssh": {"avg_latency_ms": 120.5, "success_rate": 0.95, ...},
                },
                "by_target": {
                    "node-1": {"best_transport": "http", "avg_latency_ms": 32.1},
                    ...
                },
                "summary": {
                    "total_targets": 25,
                    "transports_used": ["http", "ssh", "tailscale"],
                    "overall_avg_latency_ms": 67.3,
                }
            }
        """
        by_transport: dict[str, dict[str, float | int]] = {}
        by_target: dict[str, dict[str, Any]] = {}

        # Aggregate stats by transport
        for target, transports in self._health.items():
            best_transport = None
            best_latency = float("inf")

            for transport_name, health in transports.items():
                # Skip transports with no data
                if health.successes + health.failures == 0:
                    continue

                # Initialize transport aggregate if needed
                if transport_name not in by_transport:
                    by_transport[transport_name] = {
                        "total_successes": 0,
                        "total_failures": 0,
                        "total_latency_ms": 0.0,
                        "targets_used": 0,
                    }

                agg = by_transport[transport_name]
                agg["total_successes"] += health.successes
                agg["total_failures"] += health.failures
                agg["total_latency_ms"] += health.total_latency_ms
                agg["targets_used"] += 1

                # Track best transport per target
                if health.successes > 0 and health.avg_latency_ms < best_latency:
                    best_latency = health.avg_latency_ms
                    best_transport = transport_name

            if best_transport:
                by_target[target] = {
                    "best_transport": best_transport,
                    "avg_latency_ms": round(best_latency, 1),
                }

        # Calculate averages and success rates per transport
        transport_summary: dict[str, dict[str, Any]] = {}
        overall_latency_total = 0.0
        overall_success_total = 0

        for transport_name, agg in by_transport.items():
            successes = agg["total_successes"]
            failures = agg["total_failures"]
            total = successes + failures

            avg_latency = 0.0
            if successes > 0:
                avg_latency = agg["total_latency_ms"] / successes
                overall_latency_total += agg["total_latency_ms"]
                overall_success_total += successes

            transport_summary[transport_name] = {
                "avg_latency_ms": round(avg_latency, 1),
                "success_rate": round(successes / total, 3) if total > 0 else 0.0,
                "total_requests": total,
                "targets_used": agg["targets_used"],
            }

        # Overall summary
        overall_avg = 0.0
        if overall_success_total > 0:
            overall_avg = overall_latency_total / overall_success_total

        return {
            "by_transport": transport_summary,
            "by_target": by_target,
            "summary": {
                "total_targets": len(by_target),
                "transports_used": list(transport_summary.keys()),
                "overall_avg_latency_ms": round(overall_avg, 1),
            },
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Sprint 15 (Jan 3, 2026): Added for unified health monitoring.
        """
        # Calculate overall transport health
        total_successes = 0
        total_failures = 0
        transport_count = 0

        for target_health in self._health.values():
            for health in target_health.values():
                total_successes += health.successes
                total_failures += health.failures
                transport_count += 1

        total = total_successes + total_failures
        success_rate = total_successes / total if total > 0 else 1.0

        # Check circuit breaker state
        cb_open = False
        if self._global_circuit_breaker:
            cb_open = self._global_circuit_breaker._state.state == "open"

        # Determine health status
        if cb_open:
            status = "critical"
            healthy = False
        elif success_rate < 0.5:
            status = "degraded"
            healthy = True  # Still functioning
        else:
            status = "healthy"
            healthy = True

        return {
            "healthy": healthy,
            "status": status,
            "details": {
                "enabled": self._enabled,
                "total_transports": len(self._transports),
                "targets_tracked": len(self._health),
                "total_requests": total,
                "success_rate": round(success_rate, 3),
                "circuit_breaker_open": cb_open,
                "adaptive_timeouts_enabled": self._adaptive_timeouts_enabled,
                "min_tier": self._min_tier,
                "max_tier": self._max_tier,
            },
        }


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

    Sprint 4 (Jan 2, 2026): Added state persistence for crash recovery.
    Circuit state is saved to StateManager on transitions and restored on startup.
    """

    # Class-level state manager reference for persistence
    # Set via set_state_manager() during P2P orchestrator initialization
    _state_manager: Any = None
    _persistence_key = "global_circuit_breaker"

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

        # Sprint 4: Restore state from persistence if available
        self._load_state()

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

            # Sprint 4: Persist state for crash recovery
            self._save_state()

            # Trigger emergency notifications
            if self._notification_callback:
                safe_create_task(self._emit_emergency_alerts(), name="cascade-emergency-alerts")

    def _close_circuit(self) -> None:
        """Close the circuit breaker."""
        self._state.state = "closed"
        self._state.total_failures = 0
        self._half_open_successes = 0
        logger.info("Global circuit breaker CLOSED")

        # Sprint 4: Persist state for crash recovery
        self._save_state()

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

    # =========================================================================
    # State Persistence (Sprint 4 - Jan 2, 2026)
    # =========================================================================

    @classmethod
    def set_state_manager(cls, state_manager: Any) -> None:
        """Set the state manager for circuit breaker persistence.

        Called during P2P orchestrator initialization to enable state persistence.

        Args:
            state_manager: StateManager instance from p2p/managers/state_manager.py
        """
        cls._state_manager = state_manager
        logger.debug("[GlobalCircuitBreaker] State manager configured for persistence")

    def _save_state(self) -> bool:
        """Save circuit breaker state to persistence.

        Sprint 4 (Jan 2, 2026): Persist state on every transition to enable
        crash recovery. Uses StateManager's peer health infrastructure.

        Returns:
            True if saved successfully, False otherwise
        """
        if not self._state_manager:
            return False

        try:
            # Use PeerHealthState with a synthetic peer_id for the global CB
            from .managers.state_manager import PeerHealthState

            health_state = PeerHealthState(
                node_id=self._persistence_key,
                state="global_circuit_breaker",  # Special marker
                failure_count=self._state.total_failures,
                gossip_failure_count=0,
                last_seen=self._state.last_success_time,
                last_failure=self._state.last_failure_time,
                circuit_state=self._state.state,
                circuit_opened_at=self._state.opened_at,
            )

            return self._state_manager.save_peer_health(health_state)
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"[GlobalCircuitBreaker] Failed to save state: {e}")
            return False

    def _load_state(self) -> bool:
        """Load circuit breaker state from persistence.

        Sprint 4 (Jan 2, 2026): Restore state on startup to prevent
        "circuit breaker amnesia" after P2P restart.

        Returns:
            True if state was restored, False otherwise
        """
        if not self._state_manager:
            return False

        try:
            health_state = self._state_manager.load_peer_health(self._persistence_key)
            if health_state is None:
                return False

            # Only restore if the persisted state is recent (within recovery timeout * 2)
            age = time.time() - health_state.updated_at
            if age > self._recovery_timeout * 2:
                logger.info(
                    f"[GlobalCircuitBreaker] Persisted state too old ({age:.0f}s), "
                    f"starting fresh"
                )
                return False

            # Restore state
            self._state.state = health_state.circuit_state
            self._state.opened_at = health_state.circuit_opened_at
            self._state.total_failures = health_state.failure_count
            self._state.last_failure_time = health_state.last_failure
            self._state.last_success_time = health_state.last_seen

            if self._state.state == "open":
                logger.warning(
                    f"[GlobalCircuitBreaker] Restored OPEN state from persistence "
                    f"(opened {age:.0f}s ago, failures={self._state.total_failures})"
                )
            else:
                logger.info(
                    f"[GlobalCircuitBreaker] Restored state={self._state.state} "
                    f"from persistence"
                )

            return True
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"[GlobalCircuitBreaker] Failed to load state: {e}")
            return False


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
