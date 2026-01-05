"""Fast Failure Detector.

Jan 4, 2026 - Phase 4 of P2P Cluster Resilience.

Problem: The progress watchdog takes 6 hours to detect cluster-wide failures.
This is too slow for acute failures where GPU nodes sit idle.

Solution: Tiered detection with escalating response:
- Tier 1 (5 min): Warning log only
- Tier 2 (10 min): Emit FAST_FAILURE_ALERT, boost selfplay 1.5x
- Tier 3 (30 min): Trigger autonomous queue, boost selfplay 2x

Failure Signals Monitored:
- No leader for extended period
- Work queue empty/near-empty
- Low selfplay completion rate (<0.1 games/sec)
- High GPU idle percentage (>70%)

Events Emitted:
- FAST_FAILURE_ALERT: 10-minute failure detected
- FAST_FAILURE_RECOVERY: 30-minute escalation triggered
- FAST_FAILURE_RECOVERED: Cluster returned to healthy state

Usage:
    from app.coordination.fast_failure_detector import (
        FastFailureDetector,
        get_fast_failure_detector,
    )

    detector = get_fast_failure_detector()
    await detector.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from app.coordination.handler_base import HandlerBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Configuration defaults (seconds)
DEFAULT_TIER1_THRESHOLD = 300.0  # 5 minutes - warning log
DEFAULT_TIER2_THRESHOLD = 600.0  # 10 minutes - alert + boost
DEFAULT_TIER3_THRESHOLD = 1800.0  # 30 minutes - recovery mode
DEFAULT_RECOVERY_THRESHOLD = 120.0  # 2 minutes of health before clearing
DEFAULT_CHECK_INTERVAL = 30.0


class FailureTier(Enum):
    """Tier levels for failure detection."""
    HEALTHY = 0
    WARNING = 1  # Tier 1: 5 min
    ALERT = 2    # Tier 2: 10 min
    RECOVERY = 3  # Tier 3: 30 min


@dataclass
class FailureTierConfig:
    """Configuration for a single failure tier."""
    tier: FailureTier
    threshold_seconds: float
    action: str  # "log", "emit", "recover"
    event_type: str | None = None
    selfplay_boost: float = 1.0

    def __post_init__(self) -> None:
        if self.action not in ("log", "emit", "recover"):
            raise ValueError(f"Invalid action: {self.action}")


@dataclass
class FastFailureConfig:
    """Configuration for fast failure detection.

    Attributes:
        tiers: List of tier configurations
        recovery_threshold_seconds: Time of health before clearing failure state
        check_interval_seconds: How often to check cluster health
        enabled: Whether this detector is active
    """
    tiers: list[FailureTierConfig] = field(default_factory=list)
    recovery_threshold_seconds: float = DEFAULT_RECOVERY_THRESHOLD
    check_interval_seconds: float = DEFAULT_CHECK_INTERVAL
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.tiers:
            self.tiers = [
                FailureTierConfig(
                    tier=FailureTier.WARNING,
                    threshold_seconds=DEFAULT_TIER1_THRESHOLD,
                    action="log",
                ),
                FailureTierConfig(
                    tier=FailureTier.ALERT,
                    threshold_seconds=DEFAULT_TIER2_THRESHOLD,
                    action="emit",
                    event_type="FAST_FAILURE_ALERT",
                    selfplay_boost=1.5,
                ),
                FailureTierConfig(
                    tier=FailureTier.RECOVERY,
                    threshold_seconds=DEFAULT_TIER3_THRESHOLD,
                    action="recover",
                    event_type="FAST_FAILURE_RECOVERY",
                    selfplay_boost=2.0,
                ),
            ]

    @classmethod
    def from_env(cls) -> "FastFailureConfig":
        """Create config from environment variables."""
        import os

        return cls(
            recovery_threshold_seconds=float(
                os.environ.get("RINGRIFT_FAST_FAILURE_RECOVERY_THRESHOLD", DEFAULT_RECOVERY_THRESHOLD)
            ),
            check_interval_seconds=float(
                os.environ.get("RINGRIFT_FAST_FAILURE_CHECK_INTERVAL", DEFAULT_CHECK_INTERVAL)
            ),
            enabled=os.environ.get("RINGRIFT_FAST_FAILURE_ENABLED", "true").lower() == "true",
        )


@dataclass
class FailureSignals:
    """Collected failure signals from cluster state."""
    no_leader: bool = False
    queue_empty: bool = False
    queue_depth: int = 0
    low_selfplay_rate: bool = False
    selfplay_rate: float = 0.0
    high_idle_percent: bool = False
    idle_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_failing(self) -> bool:
        """Returns True if any failure signal is active."""
        return self.no_leader or self.queue_empty or self.low_selfplay_rate or self.high_idle_percent

    @property
    def signal_count(self) -> int:
        """Number of active failure signals."""
        return sum([self.no_leader, self.queue_empty, self.low_selfplay_rate, self.high_idle_percent])


@dataclass
class DetectorStats:
    """Statistics for the fast failure detector."""
    checks_performed: int = 0
    failures_detected: int = 0
    alerts_emitted: int = 0
    recoveries_triggered: int = 0
    tier_escalations: int = 0
    current_tier: FailureTier = FailureTier.HEALTHY
    failure_start_time: float = 0.0
    last_healthy_time: float = field(default_factory=time.time)
    last_check_time: float = 0.0
    last_signals: FailureSignals | None = None


class FastFailureDetector(HandlerBase):
    """Detects cluster-wide failures within 5-10 minutes.

    This detector monitors multiple failure signals and escalates
    through tiers based on failure duration. Each tier triggers
    progressively stronger recovery actions.

    Features:
    - Multi-signal failure detection (leader, queue, selfplay rate, idle %)
    - Tiered escalation (5 min warning → 10 min alert → 30 min recovery)
    - Selfplay boost multipliers to break failure loops
    - Recovery detection to clear failure state
    """

    # Singleton instance
    _instance: "FastFailureDetector | None" = None

    def __init__(
        self,
        config: FastFailureConfig | None = None,
        get_leader_id: Callable[[], str | None] | None = None,
        get_work_queue_depth: Callable[[], int] | None = None,
        get_selfplay_rate: Callable[[], float] | None = None,
        get_cluster_utilization: Callable[[], float] | None = None,
        trigger_autonomous_queue: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            config: Detector configuration
            get_leader_id: Callback to get current leader ID (None if no leader)
            get_work_queue_depth: Callback to get work queue depth
            get_selfplay_rate: Callback to get selfplay games/second rate
            get_cluster_utilization: Callback to get cluster utilization (0.0-1.0)
            trigger_autonomous_queue: Callback to activate autonomous queue
        """
        resolved_config = config or FastFailureConfig.from_env()
        super().__init__(
            name="fast_failure_detector",
            config=resolved_config,
            cycle_interval=resolved_config.check_interval_seconds,
        )

        # Callbacks
        self._get_leader_id = get_leader_id
        self._get_work_queue_depth = get_work_queue_depth
        self._get_selfplay_rate = get_selfplay_rate
        self._get_cluster_utilization = get_cluster_utilization
        self._trigger_autonomous_queue = trigger_autonomous_queue

        # State
        self._stats = DetectorStats()
        self._current_boost = 1.0

    @classmethod
    def get_instance(
        cls,
        config: FastFailureConfig | None = None,
        **kwargs: Any,
    ) -> "FastFailureDetector":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config=config, **kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def set_callbacks(
        self,
        get_leader_id: Callable[[], str | None] | None = None,
        get_work_queue_depth: Callable[[], int] | None = None,
        get_selfplay_rate: Callable[[], float] | None = None,
        get_cluster_utilization: Callable[[], float] | None = None,
        trigger_autonomous_queue: Callable[[], None] | None = None,
    ) -> None:
        """Set callbacks for late binding."""
        if get_leader_id is not None:
            self._get_leader_id = get_leader_id
        if get_work_queue_depth is not None:
            self._get_work_queue_depth = get_work_queue_depth
        if get_selfplay_rate is not None:
            self._get_selfplay_rate = get_selfplay_rate
        if get_cluster_utilization is not None:
            self._get_cluster_utilization = get_cluster_utilization
        if trigger_autonomous_queue is not None:
            self._trigger_autonomous_queue = trigger_autonomous_queue

    async def _run_cycle(self) -> None:
        """Main cycle - check cluster health and escalate if needed."""
        if not self._config.enabled:
            return

        self._stats.checks_performed += 1
        self._stats.last_check_time = time.time()

        # Collect failure signals
        signals = self._collect_failure_signals()
        self._stats.last_signals = signals

        if signals.is_failing:
            await self._handle_failure(signals)
        else:
            await self._handle_recovery()

    def _collect_failure_signals(self) -> FailureSignals:
        """Collect all failure signals from cluster state."""
        signals = FailureSignals()

        # Leader status
        if self._get_leader_id:
            try:
                leader_id = self._get_leader_id()
                signals.no_leader = leader_id is None
            except Exception:
                signals.no_leader = True

        # Work queue depth
        if self._get_work_queue_depth:
            try:
                depth = self._get_work_queue_depth()
                signals.queue_depth = depth
                signals.queue_empty = depth < 5  # Near-empty threshold
            except Exception:
                signals.queue_empty = True
                signals.queue_depth = 0

        # Selfplay rate
        if self._get_selfplay_rate:
            try:
                rate = self._get_selfplay_rate()
                signals.selfplay_rate = rate
                signals.low_selfplay_rate = rate < 0.1  # Less than 0.1 games/sec
            except Exception:
                signals.low_selfplay_rate = True

        # Cluster utilization
        if self._get_cluster_utilization:
            try:
                util = self._get_cluster_utilization()
                signals.idle_percent = 1.0 - util
                signals.high_idle_percent = util < 0.3  # Less than 30% utilized
            except Exception:
                signals.high_idle_percent = True

        return signals

    async def _handle_failure(self, signals: FailureSignals) -> None:
        """Handle detected failure - escalate through tiers."""
        now = time.time()

        # Start tracking failure if not already
        if self._stats.current_tier == FailureTier.HEALTHY:
            self._stats.failure_start_time = now
            self._stats.failures_detected += 1
            logger.warning(
                f"[FastFailureDetector] Failure detected: "
                f"{signals.signal_count} signals active (no_leader={signals.no_leader}, "
                f"queue_empty={signals.queue_empty}, low_rate={signals.low_selfplay_rate}, "
                f"high_idle={signals.high_idle_percent})"
            )

        # Calculate failure duration
        failure_duration = now - self._stats.failure_start_time

        # Check each tier (highest to lowest)
        for tier_config in sorted(self._config.tiers, key=lambda t: t.threshold_seconds, reverse=True):
            if failure_duration >= tier_config.threshold_seconds:
                if self._stats.current_tier != tier_config.tier:
                    await self._escalate_to_tier(tier_config, signals, failure_duration)
                break

    async def _escalate_to_tier(
        self,
        tier_config: FailureTierConfig,
        signals: FailureSignals,
        failure_duration: float,
    ) -> None:
        """Escalate to a new failure tier."""
        old_tier = self._stats.current_tier
        self._stats.current_tier = tier_config.tier
        self._stats.tier_escalations += 1

        logger.warning(
            f"[FastFailureDetector] Escalating from {old_tier.name} to {tier_config.tier.name} "
            f"after {failure_duration:.0f}s of failure"
        )

        if tier_config.action == "log":
            # Tier 1: Just log warning
            logger.warning(
                f"[FastFailureDetector] Tier 1 WARNING: Cluster unhealthy for "
                f"{failure_duration:.0f}s ({tier_config.threshold_seconds}s threshold)"
            )

        elif tier_config.action == "emit":
            # Tier 2: Emit alert event, apply boost
            self._stats.alerts_emitted += 1
            self._current_boost = tier_config.selfplay_boost

            self._emit_failure_event(
                tier_config.event_type or "FAST_FAILURE_ALERT",
                tier_config.tier,
                signals,
                failure_duration,
            )

            logger.warning(
                f"[FastFailureDetector] Tier 2 ALERT: Emitted {tier_config.event_type}, "
                f"selfplay boost={tier_config.selfplay_boost}x"
            )

        elif tier_config.action == "recover":
            # Tier 3: Full recovery mode
            self._stats.recoveries_triggered += 1
            self._current_boost = tier_config.selfplay_boost

            self._emit_failure_event(
                tier_config.event_type or "FAST_FAILURE_RECOVERY",
                tier_config.tier,
                signals,
                failure_duration,
            )

            # Trigger autonomous queue if available
            if self._trigger_autonomous_queue:
                try:
                    self._trigger_autonomous_queue()
                    logger.warning("[FastFailureDetector] Triggered autonomous queue activation")
                except Exception as e:
                    logger.error(f"[FastFailureDetector] Failed to trigger autonomous queue: {e}")

            logger.warning(
                f"[FastFailureDetector] Tier 3 RECOVERY: Full recovery mode activated, "
                f"selfplay boost={tier_config.selfplay_boost}x"
            )

    async def _handle_recovery(self) -> None:
        """Handle cluster returning to healthy state."""
        if self._stats.current_tier == FailureTier.HEALTHY:
            self._stats.last_healthy_time = time.time()
            return

        now = time.time()
        healthy_duration = now - self._stats.last_healthy_time

        # Check if we've been healthy long enough to clear failure state
        if healthy_duration >= self._config.recovery_threshold_seconds:
            old_tier = self._stats.current_tier
            self._stats.current_tier = FailureTier.HEALTHY
            self._stats.failure_start_time = 0.0
            self._current_boost = 1.0

            self._emit_recovered_event(old_tier)

            logger.info(
                f"[FastFailureDetector] Cluster recovered from {old_tier.name} after "
                f"{healthy_duration:.0f}s of healthy operation"
            )

        self._stats.last_healthy_time = now

    def _emit_failure_event(
        self,
        event_type: str,
        tier: FailureTier,
        signals: FailureSignals,
        failure_duration: float,
    ) -> None:
        """Emit a failure event."""
        from app.coordination.event_emission_helpers import safe_emit_event

        safe_emit_event(
            event_type,
            {
                "tier": tier.name,
                "failure_duration_seconds": failure_duration,
                "signals": {
                    "no_leader": signals.no_leader,
                    "queue_empty": signals.queue_empty,
                    "queue_depth": signals.queue_depth,
                    "low_selfplay_rate": signals.low_selfplay_rate,
                    "selfplay_rate": signals.selfplay_rate,
                    "high_idle_percent": signals.high_idle_percent,
                    "idle_percent": signals.idle_percent,
                },
                "selfplay_boost": self._current_boost,
                "timestamp": time.time(),
            },
            context="FastFailureDetector",
        )

    def _emit_recovered_event(self, from_tier: FailureTier) -> None:
        """Emit FAST_FAILURE_RECOVERED event."""
        from app.coordination.event_emission_helpers import safe_emit_event

        safe_emit_event(
            "FAST_FAILURE_RECOVERED",
            {
                "from_tier": from_tier.name,
                "total_failure_duration_seconds": time.time() - self._stats.failure_start_time
                if self._stats.failure_start_time > 0
                else 0,
                "timestamp": time.time(),
            },
            context="FastFailureDetector",
        )

    def get_current_boost(self) -> float:
        """Get the current selfplay boost multiplier."""
        return self._current_boost

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics."""
        return {
            "enabled": self._config.enabled,
            "current_tier": self._stats.current_tier.name,
            "current_boost": self._current_boost,
            "checks_performed": self._stats.checks_performed,
            "failures_detected": self._stats.failures_detected,
            "alerts_emitted": self._stats.alerts_emitted,
            "recoveries_triggered": self._stats.recoveries_triggered,
            "tier_escalations": self._stats.tier_escalations,
            "failure_start_time": self._stats.failure_start_time,
            "last_healthy_time": self._stats.last_healthy_time,
            "last_check_time": self._stats.last_check_time,
            "last_signals": {
                "no_leader": self._stats.last_signals.no_leader,
                "queue_empty": self._stats.last_signals.queue_empty,
                "queue_depth": self._stats.last_signals.queue_depth,
                "low_selfplay_rate": self._stats.last_signals.low_selfplay_rate,
                "selfplay_rate": self._stats.last_signals.selfplay_rate,
                "high_idle_percent": self._stats.last_signals.high_idle_percent,
                "idle_percent": self._stats.last_signals.idle_percent,
            } if self._stats.last_signals else None,
        }

    def health_check(self) -> dict[str, Any]:
        """Return health check result."""
        is_healthy = self._stats.current_tier == FailureTier.HEALTHY
        return {
            "healthy": is_healthy,
            "status": "degraded" if not is_healthy else "healthy",
            "details": self.get_stats(),
        }


# Singleton accessor
def get_fast_failure_detector(
    config: FastFailureConfig | None = None,
    **kwargs: Any,
) -> FastFailureDetector:
    """Get the singleton fast failure detector."""
    return FastFailureDetector.get_instance(config=config, **kwargs)
