"""
Unified Backpressure Signal for Cluster Coordination.

Consolidates multiple pressure sources into a single signal:
- Queue pressure: Selfplay queue depth
- Training pressure: Active training jobs
- Disk pressure: Cluster disk usage
- Sync pressure: Pending data syncs

This module provides a unified spawn_rate_multiplier (0.0-1.0) that
daemons can use to throttle work generation.

Usage:
    from app.coordination.backpressure import (
        get_backpressure_monitor,
        BackpressureSignal,
    )

    # Get current backpressure
    monitor = get_backpressure_monitor()
    signal = await monitor.get_signal()

    # Use spawn rate multiplier
    if signal.spawn_rate_multiplier < 0.1:
        logger.info("High backpressure, pausing spawns")
        return

    # Adjust spawn rate
    games_to_spawn = int(base_games * signal.spawn_rate_multiplier)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

# December 29, 2025: Import DynamicThreshold for adaptive queue depth
try:
    from app.coordination.dynamic_thresholds import (
        AdjustmentStrategy,
        DynamicThreshold,
    )
    HAS_DYNAMIC_THRESHOLDS = True
except ImportError:
    HAS_DYNAMIC_THRESHOLDS = False
    DynamicThreshold = None
    AdjustmentStrategy = None

logger = logging.getLogger(__name__)


@dataclass
class BackpressureSignal:
    """Unified backpressure signal from all sources.

    Each pressure value is normalized to 0.0-1.0:
    - 0.0 = no pressure (system healthy)
    - 1.0 = maximum pressure (system overloaded)
    """

    # Individual pressure sources
    queue_pressure: float = 0.0      # Selfplay queue depth
    training_pressure: float = 0.0   # Active training jobs
    disk_pressure: float = 0.0       # Cluster disk usage
    sync_pressure: float = 0.0       # Pending data syncs
    memory_pressure: float = 0.0     # GPU memory usage

    # Metadata
    timestamp: float = field(default_factory=time.time)
    source_details: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_pressure(self) -> float:
        """Compute overall pressure as weighted average.

        Weights (Jan 2026 - increased memory weight for OOM prevention):
        - Queue: 25% (directly affects training data availability)
        - Memory: 25% (system RAM - critical for OOM prevention)
        - Training: 20% (GPU utilization)
        - Disk: 15% (storage constraints)
        - Sync: 15% (data availability across cluster)

        Note: Memory weight was increased from 10% to 25% after a cluster
        failure caused by memory exhaustion reaching 100% without triggering
        adequate backpressure response. See Session 16 cluster resilience plan.
        """
        return (
            0.25 * self.queue_pressure +
            0.25 * self.memory_pressure +
            0.20 * self.training_pressure +
            0.15 * self.disk_pressure +
            0.15 * self.sync_pressure
        )

    @property
    def spawn_rate_multiplier(self) -> float:
        """Get spawn rate multiplier based on pressure.

        Returns:
            1.0 = full speed (pressure < 0.2)
            0.0 = stopped (pressure > 0.7)
            Linear interpolation in between

        Jan 19, 2026: Lowered thresholds to prevent node saturation.
        Previous values (0.3-0.9) allowed nodes to hit 100% CPU before
        throttling, causing P2P heartbeat failures. New values (0.2-0.7)
        start throttling earlier to keep nodes responsive.
        """
        pressure = self.overall_pressure

        if pressure < 0.2:
            return 1.0
        if pressure > 0.7:
            return 0.0

        # Linear interpolation between 0.2 and 0.7
        # At 0.2 → 1.0, at 0.7 → 0.0
        return 1.0 - (pressure - 0.2) / 0.5

    @property
    def should_pause(self) -> bool:
        """Check if operations should pause entirely."""
        return self.spawn_rate_multiplier < 0.1

    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy (no significant pressure)."""
        return self.overall_pressure < 0.3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "queue_pressure": self.queue_pressure,
            "training_pressure": self.training_pressure,
            "disk_pressure": self.disk_pressure,
            "sync_pressure": self.sync_pressure,
            "memory_pressure": self.memory_pressure,
            "overall_pressure": self.overall_pressure,
            "spawn_rate_multiplier": self.spawn_rate_multiplier,
            "should_pause": self.should_pause,
            "is_healthy": self.is_healthy,
            "timestamp": self.timestamp,
        }


@dataclass
class BackpressureConfig:
    """Configuration for backpressure monitoring.

    Jan 19, 2026: Lowered thresholds to prevent node saturation and
    maintain P2P heartbeat responsiveness. Previous high thresholds
    (50%/90%) allowed nodes to hit 100% CPU before throttling,
    causing heartbeat timeouts and cluster instability.
    """

    # Queue pressure thresholds
    queue_low_threshold: int = 5       # Below this = 0 pressure (was 10)
    queue_high_threshold: int = 50     # Above this = 1.0 pressure (was 100)

    # Training pressure thresholds
    training_low_threshold: int = 1    # Below this = 0 pressure (was 2)
    training_high_threshold: int = 5   # Above this = 1.0 pressure (was 10)

    # Disk pressure thresholds (percentage)
    disk_low_threshold: float = 0.4    # Below 40% = 0 pressure (was 50%)
    disk_high_threshold: float = 0.75  # Above 75% = 1.0 pressure (was 90%)

    # Sync pressure thresholds
    sync_low_threshold: int = 3        # Below this = 0 pressure (was 5)
    sync_high_threshold: int = 30      # Above this = 1.0 pressure (was 50)

    # Memory pressure thresholds (percentage)
    memory_low_threshold: float = 0.3  # Below 30% = 0 pressure (was 50%)
    memory_high_threshold: float = 0.7 # Above 70% = 1.0 pressure (was 90%)

    # Cache settings
    cache_ttl_seconds: float = 10.0    # How long to cache the signal


class BackpressureMonitor:
    """Monitors cluster and produces unified backpressure signal.

    Collects metrics from:
    - QueueMonitor for queue depth
    - ClusterMonitor for training jobs
    - P2P mesh for disk/sync status
    """

    def __init__(self, config: BackpressureConfig | None = None):
        """Initialize the backpressure monitor.

        Args:
            config: Configuration for thresholds. Uses defaults if not provided.
        """
        self.config = config or BackpressureConfig()
        self._cached_signal: BackpressureSignal | None = None
        self._cache_time: float = 0
        self._lock = asyncio.Lock()
        # December 2025: Track state for event emission
        self._was_paused: bool = False
        self._last_event_time: float = 0.0
        self._event_cooldown: float = 60.0  # Min 60s between same event type

        # December 29, 2025: Dynamic queue depth threshold
        # Adjusts queue_high_threshold based on observed queue behavior
        # - When queue overflows frequently, raise threshold (more tolerant)
        # - When queue is consistently healthy, lower threshold (more responsive)
        self._dynamic_queue_threshold: DynamicThreshold | None = None
        if HAS_DYNAMIC_THRESHOLDS and DynamicThreshold is not None:
            self._dynamic_queue_threshold = DynamicThreshold(
                name="queue_depth_high",
                initial_value=float(self.config.queue_high_threshold),
                min_value=20.0,   # Don't go below 20 items
                max_value=500.0,  # Don't exceed 500 items
                target_success_rate=0.85,  # 85% of observations should be below threshold
                adjustment_strategy=AdjustmentStrategy.ADAPTIVE,
                adjustment_factor=0.05,  # 5% adjustment per cycle
                window_size=100,
                cooldown_seconds=120.0,  # 2 min between adjustments
                higher_is_more_permissive=True,
            )
            logger.debug(
                f"[Backpressure] Dynamic queue threshold enabled: "
                f"initial={self.config.queue_high_threshold}"
            )

    async def get_signal(self, force_refresh: bool = False) -> BackpressureSignal:
        """Get current backpressure signal.

        Args:
            force_refresh: Force refresh even if cache is valid

        Returns:
            Current backpressure signal
        """
        async with self._lock:
            now = time.time()

            # Return cached value if still valid
            if (
                not force_refresh
                and self._cached_signal is not None
                and now - self._cache_time < self.config.cache_ttl_seconds
            ):
                return self._cached_signal

            # Collect metrics from various sources
            signal = await self._collect_metrics()
            self._cached_signal = signal
            self._cache_time = now

            # December 2025: Emit state change events
            await self._check_and_emit_state_change(signal, now)

            return signal

    async def _check_and_emit_state_change(
        self, signal: BackpressureSignal, now: float
    ) -> None:
        """Check for backpressure state changes and emit events.

        December 2025: Added to enable sync router and selfplay scheduler to
        react to backpressure changes. Emits BACKPRESSURE_ACTIVATED when
        spawning should pause, and BACKPRESSURE_RELEASED when it's safe again.

        Args:
            signal: Current backpressure signal
            now: Current timestamp
        """
        is_paused = signal.should_pause

        # Check if state changed
        if is_paused == self._was_paused:
            return

        # Apply cooldown to prevent event spam
        if now - self._last_event_time < self._event_cooldown:
            return

        try:
            from app.coordination.event_router import publish

            if is_paused and not self._was_paused:
                # Backpressure activated
                await publish(
                    event_type="BACKPRESSURE_ACTIVATED",
                    payload={
                        "overall_pressure": signal.overall_pressure,
                        "spawn_rate_multiplier": signal.spawn_rate_multiplier,
                        "queue_pressure": signal.queue_pressure,
                        "training_pressure": signal.training_pressure,
                        "disk_pressure": signal.disk_pressure,
                        "sync_pressure": signal.sync_pressure,
                        "memory_pressure": signal.memory_pressure,
                        "details": signal.source_details,
                    },
                    source="backpressure_monitor",
                )
                logger.info(
                    f"[Backpressure] ACTIVATED: pressure={signal.overall_pressure:.2f}, "
                    f"multiplier={signal.spawn_rate_multiplier:.2f}"
                )

            elif not is_paused and self._was_paused:
                # Backpressure released
                await publish(
                    event_type="BACKPRESSURE_RELEASED",
                    payload={
                        "overall_pressure": signal.overall_pressure,
                        "spawn_rate_multiplier": signal.spawn_rate_multiplier,
                    },
                    source="backpressure_monitor",
                )
                logger.info(
                    f"[Backpressure] RELEASED: pressure={signal.overall_pressure:.2f}, "
                    f"multiplier={signal.spawn_rate_multiplier:.2f}"
                )

            self._was_paused = is_paused
            self._last_event_time = now

        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"[Backpressure] Failed to emit event: {e}")

    async def _collect_metrics(self) -> BackpressureSignal:
        """Collect metrics from all sources."""
        signal = BackpressureSignal()
        details: dict[str, Any] = {}

        # Queue pressure from QueueMonitor
        try:
            from app.coordination.queue_monitor import (
                QueueType,
                get_queue_monitor,
            )

            monitor = get_queue_monitor()
            status = monitor.get_status(QueueType.TRAINING_DATA)
            if status:
                queue_depth = status.current_depth

                # December 29, 2025: Use dynamic threshold if available
                queue_high = self.config.queue_high_threshold
                if self._dynamic_queue_threshold is not None:
                    queue_high = int(self._dynamic_queue_threshold.value)
                    # Record observation: success if queue < 80% of threshold
                    is_healthy = queue_depth < queue_high * 0.8
                    self._dynamic_queue_threshold.record_outcome(
                        success=is_healthy,
                        measured_value=float(queue_depth),
                    )
                    details["dynamic_queue_threshold"] = queue_high
                    details["queue_threshold_adjusted"] = (
                        queue_high != self.config.queue_high_threshold
                    )

                signal.queue_pressure = self._normalize(
                    queue_depth,
                    self.config.queue_low_threshold,
                    queue_high,
                )
                details["queue_depth"] = queue_depth
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"[Backpressure] Queue monitor unavailable: {e}")

        # Training pressure from DaemonManager or P2P
        try:
            from app.coordination.daemon_manager import get_daemon_manager

            dm = get_daemon_manager()
            status = dm.get_status()
            running_daemons = sum(
                1 for d in status.get("daemons", {}).values()
                if d.get("state") == "RUNNING"
            )
            signal.training_pressure = self._normalize(
                running_daemons,
                self.config.training_low_threshold,
                self.config.training_high_threshold,
            )
            details["running_daemons"] = running_daemons
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"[Backpressure] Daemon manager unavailable: {e}")

        # Disk pressure from P2P status
        # Dec 2025: Updated to use p2p_integration.get_p2p_status (async, with caching)
        try:
            from app.coordination.p2p_integration import get_p2p_status

            # get_p2p_status() is async with caching
            status = await get_p2p_status()
            if status:
                # Aggregate disk usage across all nodes
                peers = status.get("peers", [])
                if peers:
                    disk_usages = [
                        p.get("disk_usage_percent", 0)
                        for p in peers
                        if isinstance(p.get("disk_usage_percent"), (int, float))
                    ]
                    if disk_usages:
                        avg_disk_usage = sum(disk_usages) / len(disk_usages) / 100.0
                        signal.disk_pressure = self._normalize(
                            avg_disk_usage,
                            self.config.disk_low_threshold,
                            self.config.disk_high_threshold,
                        )
                        details["disk_usage_percent"] = avg_disk_usage * 100
                        details["nodes_reporting_disk"] = len(disk_usages)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"[Backpressure] P2P status unavailable: {e}")

        # Sync pressure from sync router
        try:
            from app.coordination.sync_router import get_sync_router

            router = get_sync_router()
            pending_syncs = len(router.get_pending_syncs()) if hasattr(router, 'get_pending_syncs') else 0
            signal.sync_pressure = self._normalize(
                pending_syncs,
                self.config.sync_low_threshold,
                self.config.sync_high_threshold,
            )
            details["pending_syncs"] = pending_syncs
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"[Backpressure] Sync router unavailable: {e}")

        # Memory pressure from CUDA
        try:
            import torch
            if torch.cuda.is_available():
                max_mem = torch.cuda.max_memory_allocated()
                if max_mem > 0:
                    memory_allocated = torch.cuda.memory_allocated() / max_mem
                    signal.memory_pressure = self._normalize(
                        memory_allocated,
                        self.config.memory_low_threshold,
                        self.config.memory_high_threshold,
                    )
                    details["gpu_memory_percent"] = memory_allocated * 100
        except (ImportError, RuntimeError, ZeroDivisionError) as e:
            logger.debug(f"[Backpressure] CUDA unavailable: {e}")

        signal.source_details = details
        return signal

    def _normalize(self, value: float, low: float, high: float) -> float:
        """Normalize a value to 0.0-1.0 based on thresholds.

        Args:
            value: Raw value
            low: Threshold below which pressure is 0
            high: Threshold above which pressure is 1

        Returns:
            Normalized pressure (0.0-1.0)
        """
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        # Guard against division by zero when thresholds are equal
        if high <= low:
            return 0.0
        return (value - low) / (high - low)

    def get_cached_signal(self) -> BackpressureSignal | None:
        """Get cached signal without refreshing."""
        return self._cached_signal

    def get_dynamic_queue_threshold(self) -> int:
        """Get current dynamic queue depth threshold.

        December 29, 2025: Returns the dynamically adjusted queue threshold
        based on observed queue behavior.

        Returns:
            Current queue_high threshold (may differ from config)
        """
        if self._dynamic_queue_threshold is not None:
            return int(self._dynamic_queue_threshold.value)
        return self.config.queue_high_threshold

    def get_dynamic_queue_stats(self) -> dict[str, Any]:
        """Get statistics about dynamic queue threshold adjustments.

        Returns:
            Dict with threshold value, adjustment history, etc.
        """
        if self._dynamic_queue_threshold is None:
            return {
                "enabled": False,
                "current_threshold": self.config.queue_high_threshold,
            }

        threshold = self._dynamic_queue_threshold
        return {
            "enabled": True,
            "current_threshold": int(threshold.value),
            "initial_threshold": self.config.queue_high_threshold,
            "min_threshold": int(threshold.min_value),
            "max_threshold": int(threshold.max_value),
            "adjustment_count": threshold._adjustment_count,
            "last_adjusted": threshold._last_adjustment_time,
            "observation_count": len(threshold._observations),
            "current_success_rate": threshold.success_rate,
            "target_success_rate": threshold.target_success_rate,
        }

    def health_check(self) -> "HealthCheckResult":
        """Check monitor health for CoordinatorProtocol compliance.

        December 2025 Phase 9: Added for daemon health monitoring.
        December 2025 Session 2: Added exception handling.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        try:
            has_cached_signal = self._cached_signal is not None
            cache_age = time.time() - self._cache_time if self._cache_time > 0 else float('inf')
            cache_valid = cache_age < self.config.cache_ttl_seconds

            return HealthCheckResult(
                healthy=True,  # Monitor is stateless, always healthy if instantiated
                status=CoordinatorStatus.RUNNING if has_cached_signal else CoordinatorStatus.READY,
                message=f"BackpressureMonitor: cache_age={cache_age:.1f}s",
                details={
                    "has_cached_signal": has_cached_signal,
                    "cache_age_seconds": cache_age,
                    "cache_valid": cache_valid,
                    "config": {
                        "cache_ttl": self.config.cache_ttl_seconds,
                        "queue_low": self.config.queue_low_threshold,
                        "queue_high": self.config.queue_high_threshold,
                    },
                    "dynamic_queue": self.get_dynamic_queue_stats(),
                },
            )
        except (RuntimeError, OSError, AttributeError, KeyError) as e:
            logger.warning(f"[BackpressureMonitor] health_check error: {e}")
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"Health check error: {e}",
                details={"error": str(e)},
            )


# Module-level singleton
_backpressure_monitor: BackpressureMonitor | None = None


def get_backpressure_monitor(
    config: BackpressureConfig | None = None
) -> BackpressureMonitor:
    """Get the singleton backpressure monitor.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        BackpressureMonitor instance
    """
    global _backpressure_monitor
    if _backpressure_monitor is None:
        _backpressure_monitor = BackpressureMonitor(config=config)
    return _backpressure_monitor


async def get_spawn_rate_multiplier() -> float:
    """Convenience function to get current spawn rate multiplier.

    Returns:
        Multiplier (0.0-1.0) for spawn rate
    """
    monitor = get_backpressure_monitor()
    signal = await monitor.get_signal()
    return signal.spawn_rate_multiplier


async def should_pause_spawning() -> bool:
    """Convenience function to check if spawning should pause.

    Returns:
        True if spawning should pause
    """
    monitor = get_backpressure_monitor()
    signal = await monitor.get_signal()
    return signal.should_pause


def reset_backpressure_monitor() -> None:
    """Reset the singleton for testing."""
    global _backpressure_monitor
    _backpressure_monitor = None


__all__ = [
    "BackpressureConfig",
    "BackpressureMonitor",
    "BackpressureSignal",
    "get_backpressure_monitor",
    "get_spawn_rate_multiplier",
    "reset_backpressure_monitor",
    "should_pause_spawning",
]
