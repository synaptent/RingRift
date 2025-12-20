"""Dynamic Threshold Adjustment System (December 2025).

This module provides automatic threshold adjustment based on observed
system behavior. Thresholds are adjusted to maintain target success
rates and avoid unnecessary alerts.

Features:
- Automatic adjustment based on observed behavior
- Min/max bounds to prevent runaway adjustments
- Smoothing to avoid oscillation
- Multiple adjustment strategies (linear, exponential, adaptive)
- Per-metric threshold management

Usage:
    from app.coordination.dynamic_thresholds import (
        DynamicThreshold,
        ThresholdManager,
        get_threshold_manager,
    )

    # Create a dynamic threshold
    threshold = DynamicThreshold(
        name="handler_timeout",
        initial_value=30.0,
        min_value=5.0,
        max_value=120.0,
        target_success_rate=0.95,
    )

    # Record observations
    threshold.record_outcome(success=True, duration=25.0)
    threshold.record_outcome(success=False, duration=45.0)

    # Get current threshold value
    print(f"Current timeout: {threshold.value}s")
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AdjustmentStrategy(Enum):
    """Strategy for threshold adjustment."""

    LINEAR = "linear"  # Fixed step adjustments
    EXPONENTIAL = "exponential"  # Proportional adjustments
    ADAPTIVE = "adaptive"  # Based on deviation from target


@dataclass
class ThresholdObservation:
    """Observation for threshold adjustment."""

    timestamp: float
    success: bool
    measured_value: float | None = None  # e.g., actual duration
    metadata: dict[str, Any] = field(default_factory=dict)


class DynamicThreshold:
    """Dynamic threshold that adjusts based on observations.

    The threshold adjusts to maintain a target success rate while
    staying within configured bounds.
    """

    def __init__(
        self,
        name: str,
        initial_value: float,
        min_value: float,
        max_value: float,
        target_success_rate: float = 0.95,
        adjustment_strategy: AdjustmentStrategy = AdjustmentStrategy.ADAPTIVE,
        adjustment_factor: float = 0.1,
        window_size: int = 100,
        cooldown_seconds: float = 60.0,
        higher_is_more_permissive: bool = True,
    ):
        """Initialize DynamicThreshold.

        Args:
            name: Threshold name for logging
            initial_value: Starting threshold value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            target_success_rate: Target success rate (0.0-1.0)
            adjustment_strategy: How to adjust (LINEAR, EXPONENTIAL, ADAPTIVE)
            adjustment_factor: Step size for adjustments
            window_size: Number of observations to consider
            cooldown_seconds: Minimum time between adjustments
            higher_is_more_permissive: If True, higher values allow more successes
        """
        self.name = name
        self._value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.target_success_rate = target_success_rate
        self.adjustment_strategy = adjustment_strategy
        self.adjustment_factor = adjustment_factor
        self.window_size = window_size
        self.cooldown_seconds = cooldown_seconds
        self.higher_is_more_permissive = higher_is_more_permissive

        # Observation history
        self._observations: deque[ThresholdObservation] = deque(maxlen=window_size)

        # Adjustment tracking
        self._last_adjustment_time: float = 0.0
        self._adjustment_count: int = 0
        self._total_observations: int = 0

    @property
    def value(self) -> float:
        """Current threshold value."""
        return self._value

    @property
    def success_rate(self) -> float:
        """Current success rate in the observation window."""
        if not self._observations:
            return 1.0
        successes = sum(1 for o in self._observations if o.success)
        return successes / len(self._observations)

    def record_outcome(
        self,
        success: bool,
        measured_value: float | None = None,
        **metadata,
    ) -> None:
        """Record an observation for threshold adjustment.

        Args:
            success: Whether the operation succeeded
            measured_value: Actual measured value (e.g., duration)
            **metadata: Additional context
        """
        observation = ThresholdObservation(
            timestamp=time.time(),
            success=success,
            measured_value=measured_value,
            metadata=metadata,
        )
        self._observations.append(observation)
        self._total_observations += 1

        # Check if adjustment needed
        self._maybe_adjust()

    def _maybe_adjust(self) -> None:
        """Check if threshold should be adjusted."""
        now = time.time()

        # Check cooldown
        if now - self._last_adjustment_time < self.cooldown_seconds:
            return

        # Need enough observations
        if len(self._observations) < self.window_size // 2:
            return

        current_rate = self.success_rate
        rate_diff = current_rate - self.target_success_rate

        # Determine if adjustment needed
        adjustment_threshold = 0.05  # 5% tolerance
        if abs(rate_diff) < adjustment_threshold:
            return

        # Calculate adjustment
        old_value = self._value
        new_value = self._calculate_adjustment(rate_diff)

        # Apply bounds
        new_value = max(self.min_value, min(self.max_value, new_value))

        if new_value != old_value:
            self._value = new_value
            self._last_adjustment_time = now
            self._adjustment_count += 1

            logger.info(
                f"[DynamicThreshold] {self.name}: {old_value:.2f} -> {new_value:.2f} "
                f"(success_rate={current_rate:.1%}, target={self.target_success_rate:.1%})"
            )

    def _calculate_adjustment(self, rate_diff: float) -> float:
        """Calculate new threshold value based on rate difference."""
        # rate_diff > 0 means we're above target (too permissive if higher_is_more_permissive)
        # rate_diff < 0 means we're below target (too restrictive if higher_is_more_permissive)

        if self.adjustment_strategy == AdjustmentStrategy.LINEAR:
            step = self.adjustment_factor * (self.max_value - self.min_value)
            if self.higher_is_more_permissive:
                # High success -> can tighten (decrease)
                # Low success -> need to loosen (increase)
                if rate_diff > 0:
                    return self._value - step
                else:
                    return self._value + step
            else:
                if rate_diff > 0:
                    return self._value + step
                else:
                    return self._value - step

        elif self.adjustment_strategy == AdjustmentStrategy.EXPONENTIAL:
            # Proportional to current value
            if self.higher_is_more_permissive:
                if rate_diff > 0:
                    return self._value * (1 - self.adjustment_factor)
                else:
                    return self._value * (1 + self.adjustment_factor)
            else:
                if rate_diff > 0:
                    return self._value * (1 + self.adjustment_factor)
                else:
                    return self._value * (1 - self.adjustment_factor)

        else:  # ADAPTIVE
            # Adjustment proportional to how far we are from target
            adjustment_magnitude = abs(rate_diff) * self.adjustment_factor * (self.max_value - self.min_value)

            if self.higher_is_more_permissive:
                if rate_diff > 0:
                    return self._value - adjustment_magnitude
                else:
                    return self._value + adjustment_magnitude
            else:
                if rate_diff > 0:
                    return self._value + adjustment_magnitude
                else:
                    return self._value - adjustment_magnitude

    def reset(self) -> None:
        """Reset threshold to initial state."""
        self._observations.clear()
        self._last_adjustment_time = 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get threshold statistics."""
        measured_values = [
            o.measured_value for o in self._observations
            if o.measured_value is not None
        ]

        return {
            "name": self.name,
            "current_value": round(self._value, 2),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "success_rate": round(self.success_rate, 4),
            "target_success_rate": self.target_success_rate,
            "observations_in_window": len(self._observations),
            "total_observations": self._total_observations,
            "adjustment_count": self._adjustment_count,
            "strategy": self.adjustment_strategy.value,
            "measured_mean": round(statistics.mean(measured_values), 2) if measured_values else None,
            "measured_p95": round(sorted(measured_values)[int(len(measured_values) * 0.95)], 2) if len(measured_values) > 20 else None,
        }


class ThresholdManager:
    """Manager for multiple dynamic thresholds.

    Provides centralized management and monitoring of thresholds
    across the coordination system.
    """

    def __init__(self):
        self._thresholds: dict[str, DynamicThreshold] = {}

    def register(self, threshold: DynamicThreshold) -> None:
        """Register a threshold.

        Args:
            threshold: Threshold to register
        """
        self._thresholds[threshold.name] = threshold
        logger.debug(f"[ThresholdManager] Registered threshold: {threshold.name}")

    def get(self, name: str) -> DynamicThreshold | None:
        """Get a threshold by name.

        Args:
            name: Threshold name

        Returns:
            DynamicThreshold or None if not found
        """
        return self._thresholds.get(name)

    def get_value(self, name: str, default: float | None = None) -> float | None:
        """Get current value of a threshold.

        Args:
            name: Threshold name
            default: Default if threshold not found

        Returns:
            Current threshold value
        """
        threshold = self._thresholds.get(name)
        if threshold:
            return threshold.value
        return default

    def record(self, name: str, success: bool, measured_value: float | None = None) -> bool:
        """Record an observation for a threshold.

        Args:
            name: Threshold name
            success: Whether operation succeeded
            measured_value: Actual measured value

        Returns:
            True if threshold exists and observation recorded
        """
        threshold = self._thresholds.get(name)
        if threshold:
            threshold.record_outcome(success, measured_value)
            return True
        return False

    def reset_all(self) -> None:
        """Reset all thresholds."""
        for threshold in self._thresholds.values():
            threshold.reset()

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all thresholds."""
        return {
            name: threshold.get_stats()
            for name, threshold in self._thresholds.items()
        }

    def get_health(self) -> dict[str, Any]:
        """Get health summary for all thresholds."""
        stats = self.get_all_stats()

        unhealthy = []
        for name, s in stats.items():
            if s["success_rate"] < s["target_success_rate"] - 0.1:
                unhealthy.append(name)

        return {
            "threshold_count": len(self._thresholds),
            "unhealthy_thresholds": unhealthy,
            "all_healthy": len(unhealthy) == 0,
            "thresholds": stats,
        }


# =============================================================================
# Global Manager and Pre-configured Thresholds
# =============================================================================

_threshold_manager: ThresholdManager | None = None


def get_threshold_manager() -> ThresholdManager:
    """Get the global threshold manager."""
    global _threshold_manager
    if _threshold_manager is None:
        _threshold_manager = ThresholdManager()
        _initialize_default_thresholds(_threshold_manager)
    return _threshold_manager


def reset_threshold_manager() -> None:
    """Reset the global threshold manager (for testing)."""
    global _threshold_manager
    _threshold_manager = None


def _initialize_default_thresholds(manager: ThresholdManager) -> None:
    """Initialize default system thresholds."""

    # Handler timeout threshold
    manager.register(DynamicThreshold(
        name="handler_timeout",
        initial_value=30.0,
        min_value=5.0,
        max_value=120.0,
        target_success_rate=0.95,
        adjustment_strategy=AdjustmentStrategy.ADAPTIVE,
        higher_is_more_permissive=True,
    ))

    # Heartbeat threshold
    manager.register(DynamicThreshold(
        name="heartbeat_threshold",
        initial_value=60.0,
        min_value=20.0,
        max_value=300.0,
        target_success_rate=0.99,  # Want very high success for heartbeats
        adjustment_strategy=AdjustmentStrategy.EXPONENTIAL,
        higher_is_more_permissive=True,
    ))

    # Plateau detection window
    manager.register(DynamicThreshold(
        name="plateau_window",
        initial_value=10.0,
        min_value=5.0,
        max_value=30.0,
        target_success_rate=0.90,  # 90% should be valid plateau detections
        adjustment_strategy=AdjustmentStrategy.LINEAR,
        higher_is_more_permissive=True,
    ))

    # Memory warning threshold
    manager.register(DynamicThreshold(
        name="memory_warning",
        initial_value=0.8,
        min_value=0.5,
        max_value=0.95,
        target_success_rate=0.95,
        adjustment_strategy=AdjustmentStrategy.ADAPTIVE,
        higher_is_more_permissive=True,
    ))

    logger.info("[ThresholdManager] Initialized with default thresholds")


__all__ = [
    "AdjustmentStrategy",
    "DynamicThreshold",
    "ThresholdManager",
    "ThresholdObservation",
    "get_threshold_manager",
    "reset_threshold_manager",
]
