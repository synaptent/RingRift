"""Unified Regression Detection for RingRift AI Training.

Consolidates regression detection logic from multiple scattered locations:
- rollback_manager.py (Elo drop, win rate drop, error rate)
- promotion_controller.py (consecutive regression tracking)
- feedback_accelerator.py (momentum-based regression state)

This module provides a SINGLE SOURCE OF TRUTH for regression detection.
Other modules should subscribe to regression events rather than implementing
their own detection logic.

Usage:
    from app.training.regression_detector import (
        RegressionDetector,
        RegressionConfig,
        RegressionEvent,
        RegressionSeverity,
    )

    detector = RegressionDetector(config)

    # Check for regression
    event = detector.check_regression(
        model_id="square8_2p_v42",
        current_elo=1450,
        baseline_elo=1520,
        current_win_rate=0.42,
        baseline_win_rate=0.55,
        games_played=100,
    )

    if event:
        print(f"Regression detected: {event.severity.name}")
        # Notify listeners
        for listener in detector.listeners:
            listener.on_regression(event)

See docs/CONSOLIDATION_ROADMAP.md for consolidation context.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from app.distributed.data_events import EventBus

# Import canonical thresholds
try:
    from app.config.thresholds import (
        CONSECUTIVE_REGRESSIONS_FORCE,
        ELO_DROP_ROLLBACK,
        ERROR_RATE_ROLLBACK,
        MIN_GAMES_REGRESSION,
        WIN_RATE_DROP_ROLLBACK,
    )
except ImportError:
    # Fallback defaults
    ELO_DROP_ROLLBACK = 50
    WIN_RATE_DROP_ROLLBACK = 0.10
    ERROR_RATE_ROLLBACK = 0.05
    MIN_GAMES_REGRESSION = 50
    CONSECUTIVE_REGRESSIONS_FORCE = 3

logger = logging.getLogger(__name__)


class RegressionSeverity(Enum):
    """Severity levels for detected regressions."""
    MINOR = "minor"      # Small dip, might recover
    MODERATE = "moderate"  # Significant drop, needs attention
    SEVERE = "severe"    # Critical drop, requires action
    CRITICAL = "critical"  # Immediate rollback recommended


@dataclass
class RegressionConfig:
    """Configuration for regression detection.

    Uses canonical threshold values from app/config/thresholds.py
    """
    # Elo thresholds
    elo_drop_minor: float = 20.0  # Minor regression
    elo_drop_moderate: float = 35.0  # Moderate regression
    elo_drop_severe: float = ELO_DROP_ROLLBACK  # Severe (default: 50)
    elo_drop_critical: float = 75.0  # Critical

    # Win rate thresholds
    win_rate_drop_minor: float = 0.05  # 5%
    win_rate_drop_moderate: float = WIN_RATE_DROP_ROLLBACK / 2  # 5%
    win_rate_drop_severe: float = WIN_RATE_DROP_ROLLBACK  # 10%
    win_rate_drop_critical: float = 0.15  # 15%

    # Error rate threshold
    error_rate_threshold: float = ERROR_RATE_ROLLBACK  # 5%

    # Minimum data requirements
    min_games_for_detection: int = MIN_GAMES_REGRESSION  # 50

    # Consecutive regression tracking
    consecutive_regressions_for_escalation: int = 2
    consecutive_regressions_force_action: int = CONSECUTIVE_REGRESSIONS_FORCE  # 3

    # Time window for regression detection (hours)
    detection_window_hours: float = 24.0

    # Cooldown between regression events (seconds)
    cooldown_seconds: float = 300.0  # 5 minutes


@dataclass
class RegressionEvent:
    """A detected regression event."""
    model_id: str
    severity: RegressionSeverity
    timestamp: float

    # Metrics
    current_elo: float
    baseline_elo: float
    elo_drop: float

    current_win_rate: float | None = None
    baseline_win_rate: float | None = None
    win_rate_drop: float | None = None

    error_rate: float | None = None
    games_played: int = 0

    # Tracking
    consecutive_count: int = 1

    # Metadata
    reason: str = ""
    recommended_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "current_elo": self.current_elo,
            "baseline_elo": self.baseline_elo,
            "elo_drop": self.elo_drop,
            "current_win_rate": self.current_win_rate,
            "baseline_win_rate": self.baseline_win_rate,
            "win_rate_drop": self.win_rate_drop,
            "error_rate": self.error_rate,
            "games_played": self.games_played,
            "consecutive_count": self.consecutive_count,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
        }


class RegressionListener(Protocol):
    """Protocol for regression event listeners."""

    def on_regression(self, event: RegressionEvent) -> None:
        """Called when a regression is detected."""
        ...


class RegressionDetector:
    """Unified regression detection component.

    Consolidates regression detection from:
    - rollback_manager.py
    - promotion_controller.py
    - feedback_accelerator.py

    Subscribe to events instead of implementing detection logic elsewhere.

    When an EventBus is provided, regression events are automatically published
    to the event bus for cross-process coordination. Other modules should
    subscribe to REGRESSION_* events instead of implementing their own detection.
    """

    def __init__(
        self,
        config: RegressionConfig | None = None,
        event_bus: EventBus | None = None,
    ):
        self.config = config or RegressionConfig()
        self._listeners: list[RegressionListener] = []
        self._event_bus = event_bus

        # State tracking
        self._baselines: dict[str, dict[str, Any]] = {}
        self._consecutive_counts: dict[str, int] = {}
        self._last_event_times: dict[str, float] = {}
        self._event_history: list[RegressionEvent] = []

        logger.info("[RegressionDetector] Initialized with canonical thresholds")

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Set the event bus for publishing regression events."""
        self._event_bus = event_bus
        logger.debug("[RegressionDetector] Event bus connected")

    def add_listener(self, listener: RegressionListener) -> None:
        """Add a regression event listener."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: RegressionListener) -> None:
        """Remove a regression event listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def set_baseline(
        self,
        model_id: str,
        elo: float,
        win_rate: float | None = None,
        games: int = 0,
    ) -> None:
        """Set baseline metrics for a model.

        Should be called when a model is promoted or starts evaluation.
        """
        self._baselines[model_id] = {
            "elo": elo,
            "win_rate": win_rate,
            "games": games,
            "timestamp": time.time(),
        }
        # Reset consecutive count on new baseline
        self._consecutive_counts[model_id] = 0
        logger.debug(f"[RegressionDetector] Baseline set for {model_id}: elo={elo}")

    def check_regression(
        self,
        model_id: str,
        current_elo: float,
        baseline_elo: float | None = None,
        current_win_rate: float | None = None,
        baseline_win_rate: float | None = None,
        error_rate: float | None = None,
        games_played: int = 0,
    ) -> RegressionEvent | None:
        """Check if a regression has occurred.

        Args:
            model_id: Model identifier
            current_elo: Current Elo rating
            baseline_elo: Baseline Elo (uses stored if not provided)
            current_win_rate: Current win rate (0-1)
            baseline_win_rate: Baseline win rate (uses stored if not provided)
            error_rate: Current error rate (0-1)
            games_played: Games played since baseline

        Returns:
            RegressionEvent if regression detected, None otherwise
        """
        # Get baseline
        if baseline_elo is None:
            baseline = self._baselines.get(model_id, {})
            baseline_elo = baseline.get("elo", current_elo)
            if baseline_win_rate is None:
                baseline_win_rate = baseline.get("win_rate")

        # Check minimum games requirement
        if games_played < self.config.min_games_for_detection:
            return None

        # Check cooldown
        last_event_time = self._last_event_times.get(model_id, 0)
        if time.time() - last_event_time < self.config.cooldown_seconds:
            return None

        # Calculate drops
        elo_drop = baseline_elo - current_elo
        win_rate_drop = None
        if current_win_rate is not None and baseline_win_rate is not None:
            win_rate_drop = baseline_win_rate - current_win_rate

        # Determine severity
        severity = self._determine_severity(elo_drop, win_rate_drop, error_rate)

        if severity is None:
            # No regression detected, decrement consecutive count
            if model_id in self._consecutive_counts:
                old_count = self._consecutive_counts[model_id]
                self._consecutive_counts[model_id] = max(0, old_count - 1)
                # Publish cleared event when regression fully clears
                if old_count > 0 and self._consecutive_counts[model_id] == 0:
                    self._publish_regression_cleared(model_id)
                    logger.info(f"[RegressionDetector] Regression cleared for {model_id}")
            return None

        # Increment consecutive count
        self._consecutive_counts[model_id] = self._consecutive_counts.get(model_id, 0) + 1
        consecutive = self._consecutive_counts[model_id]

        # Escalate severity based on consecutive regressions
        if consecutive >= self.config.consecutive_regressions_force_action:
            severity = RegressionSeverity.CRITICAL
        elif consecutive >= self.config.consecutive_regressions_for_escalation:
            if severity == RegressionSeverity.MINOR:
                severity = RegressionSeverity.MODERATE
            elif severity == RegressionSeverity.MODERATE:
                severity = RegressionSeverity.SEVERE

        # Create event
        event = RegressionEvent(
            model_id=model_id,
            severity=severity,
            timestamp=time.time(),
            current_elo=current_elo,
            baseline_elo=baseline_elo,
            elo_drop=elo_drop,
            current_win_rate=current_win_rate,
            baseline_win_rate=baseline_win_rate,
            win_rate_drop=win_rate_drop,
            error_rate=error_rate,
            games_played=games_played,
            consecutive_count=consecutive,
            reason=self._build_reason(elo_drop, win_rate_drop, error_rate, consecutive),
            recommended_action=self._recommend_action(severity),
        )

        # Record event
        self._last_event_times[model_id] = event.timestamp
        self._event_history.append(event)

        # Notify listeners (local protocol-based)
        self._notify_listeners(event)

        # Publish to event bus (cross-process coordination)
        self._publish_to_event_bus(event)

        logger.warning(
            f"[RegressionDetector] {severity.name} regression for {model_id}: "
            f"elo_drop={elo_drop:.1f}, consecutive={consecutive}"
        )

        return event

    def _determine_severity(
        self,
        elo_drop: float,
        win_rate_drop: float | None,
        error_rate: float | None,
    ) -> RegressionSeverity | None:
        """Determine regression severity from metrics."""
        cfg = self.config

        # Check critical thresholds
        if elo_drop >= cfg.elo_drop_critical:
            return RegressionSeverity.CRITICAL
        if win_rate_drop is not None and win_rate_drop >= cfg.win_rate_drop_critical:
            return RegressionSeverity.CRITICAL

        # Check severe thresholds
        if elo_drop >= cfg.elo_drop_severe:
            return RegressionSeverity.SEVERE
        if win_rate_drop is not None and win_rate_drop >= cfg.win_rate_drop_severe:
            return RegressionSeverity.SEVERE
        if error_rate is not None and error_rate >= cfg.error_rate_threshold:
            return RegressionSeverity.SEVERE

        # Check moderate thresholds
        if elo_drop >= cfg.elo_drop_moderate:
            return RegressionSeverity.MODERATE
        if win_rate_drop is not None and win_rate_drop >= cfg.win_rate_drop_moderate:
            return RegressionSeverity.MODERATE

        # Check minor thresholds
        if elo_drop >= cfg.elo_drop_minor:
            return RegressionSeverity.MINOR
        if win_rate_drop is not None and win_rate_drop >= cfg.win_rate_drop_minor:
            return RegressionSeverity.MINOR

        return None

    def _build_reason(
        self,
        elo_drop: float,
        win_rate_drop: float | None,
        error_rate: float | None,
        consecutive: int,
    ) -> str:
        """Build human-readable reason string."""
        reasons = []

        if elo_drop > 0:
            reasons.append(f"Elo dropped by {elo_drop:.1f}")

        if win_rate_drop is not None and win_rate_drop > 0:
            reasons.append(f"Win rate dropped by {win_rate_drop*100:.1f}%")

        if error_rate is not None and error_rate > self.config.error_rate_threshold:
            reasons.append(f"Error rate at {error_rate*100:.1f}%")

        if consecutive > 1:
            reasons.append(f"Consecutive regression #{consecutive}")

        return "; ".join(reasons) if reasons else "Unknown regression"

    def _recommend_action(self, severity: RegressionSeverity) -> str:
        """Recommend action based on severity."""
        actions = {
            RegressionSeverity.MINOR: "Monitor closely, may recover",
            RegressionSeverity.MODERATE: "Investigate causes, consider retraining",
            RegressionSeverity.SEVERE: "Halt promotion, investigate immediately",
            RegressionSeverity.CRITICAL: "Rollback recommended, stop deployments",
        }
        return actions.get(severity, "Investigate")

    def _notify_listeners(self, event: RegressionEvent) -> None:
        """Notify all registered listeners."""
        for listener in self._listeners:
            try:
                listener.on_regression(event)
            except Exception as e:
                logger.error(f"[RegressionDetector] Listener error: {e}")

    def _publish_to_event_bus(self, event: RegressionEvent) -> None:
        """Publish regression event to the event bus for cross-process coordination."""
        if self._event_bus is None:
            return

        try:
            from app.distributed.data_events import DataEvent, DataEventType

            # Map severity to event type
            severity_to_event = {
                RegressionSeverity.MINOR: DataEventType.REGRESSION_MINOR,
                RegressionSeverity.MODERATE: DataEventType.REGRESSION_MODERATE,
                RegressionSeverity.SEVERE: DataEventType.REGRESSION_SEVERE,
                RegressionSeverity.CRITICAL: DataEventType.REGRESSION_CRITICAL,
            }
            specific_event_type = severity_to_event.get(
                event.severity, DataEventType.REGRESSION_DETECTED
            )

            payload = event.to_dict()

            # Publish the general regression event
            general_event = DataEvent(
                event_type=DataEventType.REGRESSION_DETECTED,
                payload=payload,
                source="regression_detector",
            )

            # Also publish severity-specific event for targeted subscribers
            specific_event = DataEvent(
                event_type=specific_event_type,
                payload=payload,
                source="regression_detector",
            )

            # Use synchronous publish if no event loop, async otherwise
            try:
                from app.utils.async_utils import fire_and_forget
                asyncio.get_running_loop()
                fire_and_forget(self._event_bus.publish(general_event), name="regression_general")
                fire_and_forget(self._event_bus.publish(specific_event), name="regression_specific")
            except RuntimeError:
                # No running event loop - use sync version if available
                if hasattr(self._event_bus, 'publish_sync'):
                    self._event_bus.publish_sync(general_event)
                    self._event_bus.publish_sync(specific_event)

        except Exception as e:
            logger.error(f"[RegressionDetector] Event bus publish error: {e}")

    def _publish_regression_cleared(self, model_id: str) -> None:
        """Publish event when a model recovers from regression."""
        if self._event_bus is None:
            return

        try:
            from app.distributed.data_events import DataEvent, DataEventType

            event = DataEvent(
                event_type=DataEventType.REGRESSION_CLEARED,
                payload={
                    "model_id": model_id,
                    "timestamp": time.time(),
                },
                source="regression_detector",
            )

            try:
                from app.utils.async_utils import fire_and_forget
                asyncio.get_running_loop()
                fire_and_forget(self._event_bus.publish(event), name="regression_cleared")
            except RuntimeError:
                if hasattr(self._event_bus, 'publish_sync'):
                    self._event_bus.publish_sync(event)

        except Exception as e:
            logger.error(f"[RegressionDetector] Event bus publish error: {e}")

    def get_status(self, model_id: str) -> dict[str, Any]:
        """Get regression status for a model."""
        baseline = self._baselines.get(model_id, {})
        consecutive = self._consecutive_counts.get(model_id, 0)

        recent_events = [
            e.to_dict() for e in self._event_history
            if e.model_id == model_id
        ][-5:]  # Last 5 events

        return {
            "model_id": model_id,
            "baseline_elo": baseline.get("elo"),
            "baseline_win_rate": baseline.get("win_rate"),
            "baseline_timestamp": baseline.get("timestamp"),
            "consecutive_regressions": consecutive,
            "recent_events": recent_events,
            "is_regressing": consecutive > 0,
        }

    def reset_model(self, model_id: str) -> None:
        """Reset tracking state for a model."""
        self._baselines.pop(model_id, None)
        self._consecutive_counts.pop(model_id, None)
        self._last_event_times.pop(model_id, None)
        logger.info(f"[RegressionDetector] Reset tracking for {model_id}")

    def get_all_regressing_models(self) -> list[str]:
        """Get list of models currently showing regression."""
        return [
            model_id for model_id, count in self._consecutive_counts.items()
            if count > 0
        ]


# =============================================================================
# Factory Function
# =============================================================================

_detector_instance: RegressionDetector | None = None
_detector_lock = threading.Lock()


def get_regression_detector(
    config: RegressionConfig | None = None,
    connect_event_bus: bool = True,
) -> RegressionDetector:
    """Get or create the singleton regression detector (thread-safe).

    Args:
        config: Optional configuration (only used on first call)
        connect_event_bus: If True, connects to the global event bus for
            cross-process regression event publishing (default: True)

    Returns:
        RegressionDetector instance
    """
    global _detector_instance

    if _detector_instance is None:
        with _detector_lock:
            # Double-check locking pattern
            if _detector_instance is None:
                _detector_instance = RegressionDetector(config)

    # Connect to event bus if requested and not already connected
    if connect_event_bus and _detector_instance._event_bus is None:
        try:
            from app.distributed.data_events import get_event_bus
            _detector_instance.set_event_bus(get_event_bus())
        except ImportError:
            pass  # Event bus not available

    return _detector_instance


def create_regression_detector(
    config: RegressionConfig | None = None,
    event_bus: EventBus | None = None,
) -> RegressionDetector:
    """Create a new regression detector instance.

    Unlike get_regression_detector(), this always creates a new instance.
    Useful for testing or isolated use cases.

    Args:
        config: Optional configuration
        event_bus: Optional event bus for cross-process coordination
    """
    return RegressionDetector(config, event_bus=event_bus)
