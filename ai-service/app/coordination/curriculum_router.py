"""Curriculum Signal Bridge - Base class for event→curriculum weight feedback.

Provides a generic pattern for watchers that respond to events by adjusting
curriculum weights. This consolidates the common subscribe/unsubscribe,
weight adjustment, and event emission logic shared across multiple watcher classes.

December 30, 2025: Created as part of Priority 4 consolidation effort.
Reduces ~1,200 LOC across 5 watcher classes to ~200 LOC of shared infrastructure.

Usage:
    from app.coordination.curriculum_router import CurriculumSignalBridge

    class MyEventWatcher(CurriculumSignalBridge):
        WATCHER_NAME = "my_event_watcher"
        EVENT_TYPES = ["MY_EVENT_TYPE"]  # Can be DataEventType or RingRiftEventType

        def _compute_weight_multiplier(
            self, config_key: str, payload: dict
        ) -> float | None:
            # Return multiplier (e.g., 1.2 for 20% increase)
            # Return None to skip weight adjustment
            severity = payload.get("severity", "low")
            if severity == "high":
                return 1.5  # 50% increase
            return 1.2  # 20% increase

        def _extract_event_details(self, payload: dict) -> dict:
            # Extract details for logging and CURRICULUM_REBALANCED event
            return {
                "severity": payload.get("severity", "unknown"),
                "reason": payload.get("reason", ""),
            }

    # Use the watcher
    watcher = MyEventWatcher()
    watcher.subscribe()
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from app.coordination.event_utils import extract_config_key

if TYPE_CHECKING:
    from app.coordination.health_facade import HealthCheckResult

__all__ = [
    "CurriculumSignalBridge",
    "CurriculumSignalConfig",
    "WeightAdjustment",
    "create_signal_bridge",
]

logger = logging.getLogger(__name__)


@dataclass
class CurriculumSignalConfig:
    """Configuration for a CurriculumSignalBridge.

    Attributes:
        max_weight_multiplier: Maximum weight multiplier allowed (default 3.0)
        min_weight_multiplier: Minimum weight multiplier allowed (default 0.3)
        emit_rebalance_events: Whether to emit CURRICULUM_REBALANCED events
        log_level: Logging level for weight adjustments (INFO or DEBUG)
    """

    max_weight_multiplier: float = 3.0
    min_weight_multiplier: float = 0.3
    emit_rebalance_events: bool = True
    log_level: int = logging.INFO


@dataclass
class WeightAdjustment:
    """Result of a weight adjustment operation.

    Attributes:
        config_key: The config that was adjusted
        old_weight: Previous curriculum weight
        new_weight: New curriculum weight after adjustment
        multiplier: The multiplier that was applied
        trigger: What triggered the adjustment (e.g., "regression_critical")
        details: Additional context for logging/events
    """

    config_key: str
    old_weight: float
    new_weight: float
    multiplier: float
    trigger: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def was_adjusted(self) -> bool:
        """Whether the weight actually changed."""
        return abs(self.new_weight - self.old_weight) > 0.001


class CurriculumSignalBridge(ABC):
    """Base class for event-driven curriculum weight adjustment watchers.

    Provides common infrastructure for:
    - Event subscription/unsubscription
    - Config key extraction from event payloads
    - Weight multiplier application with bounds checking
    - CURRICULUM_REBALANCED event emission
    - Health check reporting

    Subclasses must implement:
    - WATCHER_NAME: Unique identifier for logging and health checks
    - EVENT_TYPES: List of event type names or enums to subscribe to
    - _compute_weight_multiplier(): Calculate weight change from event payload
    - _extract_event_details(): Extract details for logging and events (optional)

    Example:
        class RegressionWatcher(CurriculumSignalBridge):
            WATCHER_NAME = "regression_watcher"
            EVENT_TYPES = ["REGRESSION_CRITICAL"]

            def _compute_weight_multiplier(self, config_key, payload):
                severity = payload.get("severity", "moderate")
                return 1.5 if severity == "severe" else 1.25
    """

    # Subclasses must set these
    WATCHER_NAME: str = "base_signal_bridge"
    EVENT_TYPES: list[str] = []  # Event type names or enums

    def __init__(self, config: CurriculumSignalConfig | None = None):
        """Initialize the signal bridge.

        Args:
            config: Optional configuration overrides
        """
        self._config = config or CurriculumSignalConfig()
        self._subscribed = False
        self._event_count = 0
        self._adjustment_count = 0
        self._last_event_time: float = 0.0
        self._state: dict[str, Any] = {}  # Subclass-specific state

    # =========================================================================
    # Abstract Methods - Subclasses Must Implement
    # =========================================================================

    @abstractmethod
    def _compute_weight_multiplier(
        self,
        config_key: str,
        payload: dict[str, Any],
    ) -> float | None:
        """Compute weight multiplier for a config based on event payload.

        Args:
            config_key: The config being adjusted (e.g., "hex8_2p")
            payload: Event payload dictionary

        Returns:
            Weight multiplier (e.g., 1.2 for 20% increase, 0.8 for 20% decrease)
            Return None to skip weight adjustment for this event.
        """
        ...

    # =========================================================================
    # Optional Overrides
    # =========================================================================

    def _extract_event_details(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract details from event payload for logging and events.

        Override in subclasses to include event-specific fields.

        Args:
            payload: Event payload dictionary

        Returns:
            Dictionary of details to include in logs and CURRICULUM_REBALANCED event
        """
        return {}

    def _on_weight_adjusted(
        self,
        adjustment: WeightAdjustment,
    ) -> None:
        """Called after a weight is successfully adjusted.

        Override in subclasses for additional post-adjustment logic.

        Args:
            adjustment: Details of the weight adjustment
        """
        pass

    # =========================================================================
    # Subscription Management
    # =========================================================================

    def subscribe(self) -> bool:
        """Subscribe to configured event types.

        Returns:
            True if successfully subscribed, False otherwise
        """
        if self._subscribed:
            return True

        if not self.EVENT_TYPES:
            logger.warning(f"[{self.WATCHER_NAME}] No EVENT_TYPES configured")
            return False

        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()
            if router is None:
                logger.debug(f"[{self.WATCHER_NAME}] Event router not available")
                return False

            # Subscribe to each event type
            for event_type in self.EVENT_TYPES:
                # Handle both string and enum event types
                if isinstance(event_type, str):
                    # Try DataEventType first, then use as string
                    try:
                        event_type = DataEventType[event_type]
                    except (KeyError, AttributeError):
                        pass  # Use string as-is

                router.subscribe(event_type, self._handle_event)

            self._subscribed = True
            event_names = [
                e.name if hasattr(e, "name") else str(e) for e in self.EVENT_TYPES
            ]
            logger.info(f"[{self.WATCHER_NAME}] Subscribed to {event_names}")
            return True

        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[{self.WATCHER_NAME}] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from all event types."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()
            if router:
                for event_type in self.EVENT_TYPES:
                    if isinstance(event_type, str):
                        try:
                            event_type = DataEventType[event_type]
                        except (KeyError, AttributeError):
                            pass
                    router.unsubscribe(event_type, self._handle_event)

            self._subscribed = False
            logger.info(f"[{self.WATCHER_NAME}] Unsubscribed from events")

        except (ImportError, AttributeError, TypeError, RuntimeError):
            pass

    # =========================================================================
    # Event Handling
    # =========================================================================

    def _handle_event(self, event: Any) -> None:
        """Handle incoming event - extract payload and adjust weight.

        Args:
            event: Event object with payload attribute or dict
        """
        try:
            self._event_count += 1
            self._last_event_time = time.time()

            # Extract payload from event
            payload = event.payload if hasattr(event, "payload") else event
            if not isinstance(payload, dict):
                payload = {}

            # Extract config key
            config_key = extract_config_key(payload)
            if not config_key:
                logger.debug(
                    f"[{self.WATCHER_NAME}] Event missing config_key, skipping"
                )
                return

            # Compute weight multiplier
            multiplier = self._compute_weight_multiplier(config_key, payload)
            if multiplier is None:
                logger.debug(
                    f"[{self.WATCHER_NAME}] Skipping weight adjustment for {config_key}"
                )
                return

            # Apply bounds
            multiplier = max(
                self._config.min_weight_multiplier,
                min(self._config.max_weight_multiplier, multiplier),
            )

            # Extract event details for logging
            details = self._extract_event_details(payload)

            # Apply weight adjustment
            adjustment = self._adjust_curriculum_weight(config_key, multiplier, details)

            if adjustment and adjustment.was_adjusted:
                self._adjustment_count += 1
                self._on_weight_adjusted(adjustment)

                if self._config.emit_rebalance_events:
                    self._emit_rebalance_event(adjustment)

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"[{self.WATCHER_NAME}] Error handling event: {e}")

    def _adjust_curriculum_weight(
        self,
        config_key: str,
        multiplier: float,
        details: dict[str, Any],
    ) -> WeightAdjustment | None:
        """Adjust curriculum weight for a config.

        Args:
            config_key: Config to adjust (e.g., "hex8_2p")
            multiplier: Weight multiplier to apply
            details: Additional context for logging

        Returns:
            WeightAdjustment with results, or None if adjustment failed
        """
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            old_weight = feedback._current_weights.get(config_key, 1.0)

            # Apply multiplier
            new_weight = old_weight * multiplier

            # Clamp to feedback bounds
            new_weight = max(feedback.weight_min, min(feedback.weight_max, new_weight))

            # Create adjustment record
            adjustment = WeightAdjustment(
                config_key=config_key,
                old_weight=old_weight,
                new_weight=new_weight,
                multiplier=multiplier,
                trigger=self.WATCHER_NAME,
                details=details,
            )

            # Update if changed
            if adjustment.was_adjusted:
                feedback._current_weights[config_key] = new_weight
                logger.log(
                    self._config.log_level,
                    f"[{self.WATCHER_NAME}] Adjusted curriculum weight for {config_key}: "
                    f"{old_weight:.3f} → {new_weight:.3f} (multiplier={multiplier:.2f})",
                )

            return adjustment

        except ImportError as e:
            logger.debug(f"[{self.WATCHER_NAME}] curriculum_feedback import error: {e}")
            return None
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.warning(f"[{self.WATCHER_NAME}] Error adjusting weight: {e}")
            return None

    def _emit_rebalance_event(self, adjustment: WeightAdjustment) -> None:
        """Emit CURRICULUM_REBALANCED event after weight adjustment.

        Args:
            adjustment: The weight adjustment that was made
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            if router is None:
                return

            payload = {
                "trigger": adjustment.trigger,
                "changed_configs": [adjustment.config_key],
                "new_weights": {adjustment.config_key: adjustment.new_weight},
                "old_weights": {adjustment.config_key: adjustment.old_weight},
                "multiplier": adjustment.multiplier,
                "timestamp": time.time(),
                **adjustment.details,
            }

            router.publish_sync(
                "CURRICULUM_REBALANCED",
                payload,
                source=self.WATCHER_NAME,
            )

        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[{self.WATCHER_NAME}] Failed to emit rebalance event: {e}")

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    def health_check(self) -> "HealthCheckResult":
        """Check watcher health for DaemonManager integration.

        Returns:
            HealthCheckResult indicating current health status
        """
        try:
            from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

            if not self._subscribed:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.STOPPED,
                    message=f"{self.WATCHER_NAME} not subscribed",
                )

            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"Processed {self._event_count} events, made {self._adjustment_count} adjustments",
                details={
                    "event_count": self._event_count,
                    "adjustment_count": self._adjustment_count,
                    "last_event_time": self._last_event_time,
                    "state_keys": list(self._state.keys()),
                },
            )

        except ImportError:
            # Return a simple dict if HealthCheckResult not available
            return {
                "healthy": self._subscribed,
                "status": "running" if self._subscribed else "stopped",
                "message": f"Processed {self._event_count} events",
            }

    # =========================================================================
    # State Management
    # =========================================================================

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value for a key."""
        return self._state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set state value for a key."""
        self._state[key] = value

    def reset_state(self, config_key: str | None = None) -> None:
        """Reset state, optionally for a specific config only.

        Args:
            config_key: If provided, only reset state related to this config
        """
        if config_key:
            # Remove config-specific state entries
            to_remove = [k for k in self._state if k.startswith(f"{config_key}:")]
            for k in to_remove:
                del self._state[k]
        else:
            self._state.clear()

    @property
    def is_subscribed(self) -> bool:
        """Whether currently subscribed to events."""
        return self._subscribed

    @property
    def stats(self) -> dict[str, Any]:
        """Get current statistics."""
        return {
            "watcher_name": self.WATCHER_NAME,
            "subscribed": self._subscribed,
            "event_count": self._event_count,
            "adjustment_count": self._adjustment_count,
            "last_event_time": self._last_event_time,
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_signal_bridge(
    watcher_name: str,
    event_types: list[str],
    compute_multiplier: Callable[[str, dict], float | None],
    extract_details: Callable[[dict], dict] | None = None,
    config: CurriculumSignalConfig | None = None,
) -> CurriculumSignalBridge:
    """Create a CurriculumSignalBridge from functions.

    Convenience factory for simple watchers that don't need a custom class.

    Args:
        watcher_name: Unique name for logging and health checks
        event_types: List of event type names to subscribe to
        compute_multiplier: Function to compute weight multiplier from payload
        extract_details: Optional function to extract event details
        config: Optional configuration overrides

    Returns:
        Configured CurriculumSignalBridge instance

    Example:
        bridge = create_signal_bridge(
            watcher_name="quality_boost",
            event_types=["QUALITY_SCORE_UPDATED"],
            compute_multiplier=lambda ck, p: 1.1 if p.get("score", 0) > 0.8 else None,
        )
        bridge.subscribe()
    """

    class FunctionalBridge(CurriculumSignalBridge):
        WATCHER_NAME = watcher_name
        EVENT_TYPES = event_types

        def _compute_weight_multiplier(
            self, config_key: str, payload: dict
        ) -> float | None:
            return compute_multiplier(config_key, payload)

        def _extract_event_details(self, payload: dict) -> dict:
            if extract_details:
                return extract_details(payload)
            return {}

    return FunctionalBridge(config=config)
