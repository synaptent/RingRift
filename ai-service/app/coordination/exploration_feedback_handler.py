"""Exploration Feedback Handler - Handles exploration boost signals.

Extracted from FeedbackLoopController (December 2025) to reduce file size
and improve maintainability.

This handler manages:
- Exploration boost for loss anomalies
- Exploration boost for training stalls
- Exploration reduction after improvement
- Temperature scheduler wiring

Usage:
    from app.coordination.exploration_feedback_handler import (
        ExplorationFeedbackHandler,
        get_exploration_feedback_handler,
    )

    handler = get_exploration_feedback_handler()
    await handler.start()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from app.config.thresholds import (
    EXPLORATION_BOOST_DECAY,
    EXPLORATION_BOOST_MAX,
    EXPLORATION_BOOST_PER_ANOMALY,
    EXPLORATION_BOOST_PER_STALL_GROUP,
    EXPLORATION_BOOST_STALL_MAX,
    TREND_DURATION_SEVERE,
)
from app.coordination.handler_base import HandlerBase
from app.coordination.protocols import HealthCheckResult

if TYPE_CHECKING:
    from app.coordination.feedback_loop_controller import FeedbackState

logger = logging.getLogger(__name__)

# Event emitter availability flags
try:
    from app.coordination.event_router import emit_exploration_boost
    HAS_EXPLORATION_EVENTS = True
except ImportError:
    emit_exploration_boost = None
    HAS_EXPLORATION_EVENTS = False


def _handle_task_error(task: asyncio.Task, context: str = "") -> None:
    """Handle errors from fire-and-forget tasks."""
    try:
        exc = task.exception()
        if exc is not None:
            logger.error(f"[ExplorationFeedbackHandler] Task error{' in ' + context if context else ''}: {exc}")
    except asyncio.CancelledError:
        pass
    except asyncio.InvalidStateError:
        pass


def _safe_create_task(coro, context: str = "") -> asyncio.Task | None:
    """Create a task with error handling callback."""
    try:
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda t: _handle_task_error(t, context))
        return task
    except RuntimeError as e:
        logger.debug(f"[ExplorationFeedbackHandler] Could not create task for {context}: {e}")
        return None


class ExplorationFeedbackHandler(HandlerBase):
    """Handles exploration-related feedback signals.

    Extracted from FeedbackLoopController to reduce file size.
    This handler processes exploration boost events and manages temperature
    scheduler integration.

    Key behaviors:
    - Boost exploration when loss anomalies detected
    - Boost exploration when training stalls
    - Gradually reduce exploration after improvement
    - Wire exploration boost to temperature schedulers
    """

    _instance: ExplorationFeedbackHandler | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        states: dict[str, "FeedbackState"] | None = None,
        get_or_create_state_fn: callable | None = None,
    ):
        """Initialize the exploration feedback handler.

        Args:
            states: Shared state dictionary from FeedbackLoopController
            get_or_create_state_fn: Function to get or create FeedbackState
        """
        super().__init__(name="exploration_feedback_handler", cycle_interval=60.0)

        # State management (shared with FeedbackLoopController)
        self._states = states if states is not None else {}
        self._get_or_create_state_fn = get_or_create_state_fn

        # Metrics
        self._exploration_boosts_applied = 0
        self._exploration_reductions_applied = 0
        self._last_boost_time = 0.0
        self._last_cycle_time = 0.0

    @classmethod
    def get_instance(
        cls,
        states: dict[str, "FeedbackState"] | None = None,
        get_or_create_state_fn: callable | None = None,
    ) -> "ExplorationFeedbackHandler":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(states, get_or_create_state_fn)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._running = False
            cls._instance = None

    def _get_event_subscriptions(self) -> dict[str, callable]:
        """Return event subscriptions for this handler."""
        return {
            "LOSS_ANOMALY_DETECTED": self._on_loss_anomaly_detected,
            "TRAINING_STALL_DETECTED": self._on_training_stall_detected,
            "TRAINING_IMPROVED": self._on_training_improved,
            "ELO_PLATEAU_DETECTED": self._on_elo_plateau_detected,
        }

    async def _run_cycle(self) -> None:
        """Run one cycle of the exploration feedback handler.

        This handler is primarily event-driven, so the cycle just updates metrics.
        """
        self._last_cycle_time = time.time()

    def _get_or_create_state(self, config_key: str) -> "FeedbackState":
        """Get or create a FeedbackState for the given config."""
        if self._get_or_create_state_fn:
            return self._get_or_create_state_fn(config_key)

        # Local fallback
        from app.coordination.feedback_loop_controller import FeedbackState

        if config_key not in self._states:
            self._states[config_key] = FeedbackState(config_key=config_key)
        return self._states[config_key]

    # =========================================================================
    # Exploration Boost Methods
    # =========================================================================

    def boost_exploration_for_anomaly(self, config_key: str, anomaly_count: int) -> None:
        """Boost exploration in response to loss anomalies.

        Increases exploration when training loss shows anomalies,
        which may indicate model needs more diverse data.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            anomaly_count: Number of anomalies detected
        """
        # Calculate boost: EXPLORATION_BOOST_PER_ANOMALY per anomaly, up to max
        boost = min(EXPLORATION_BOOST_MAX, 1.0 + EXPLORATION_BOOST_PER_ANOMALY * anomaly_count)

        # Always update local state (fallback for when schedulers aren't available)
        state = self._get_or_create_state(config_key)
        state.current_exploration_boost = boost

        self._exploration_boosts_applied += 1
        self._last_boost_time = time.time()

        logger.info(
            f"[ExplorationFeedbackHandler] Exploration boost set to {boost:.2f}x "
            f"for {config_key} (anomaly count: {anomaly_count})"
        )

        # Emit EXPLORATION_BOOST event to notify selfplay/temperature schedulers
        self._emit_exploration_boost(config_key, boost, "loss_anomaly", anomaly_count)

        # Try to wire to active temperature schedulers
        self._wire_to_temperature_schedulers(config_key, boost)

    def boost_exploration_for_stall(self, config_key: str, stall_epochs: int) -> None:
        """Boost exploration in response to training stall.

        Increases exploration when training metrics plateau,
        which may indicate model is stuck in local optimum.

        Args:
            config_key: Configuration key
            stall_epochs: Number of epochs in stall
        """
        # Boost per TREND_DURATION_SEVERE epochs of stall, up to max
        boost = min(
            EXPLORATION_BOOST_STALL_MAX,
            1.0 + EXPLORATION_BOOST_PER_STALL_GROUP * (stall_epochs // TREND_DURATION_SEVERE)
        )

        # Always update local state (fallback)
        state = self._get_or_create_state(config_key)
        state.current_exploration_boost = max(state.current_exploration_boost, boost)

        self._exploration_boosts_applied += 1
        self._last_boost_time = time.time()

        logger.info(
            f"[ExplorationFeedbackHandler] Exploration boost set to {boost:.2f}x "
            f"for {config_key} (stalled for {stall_epochs} epochs)"
        )

        # Emit EXPLORATION_BOOST event
        self._emit_exploration_boost(
            config_key, boost, "stall", stall_epochs // 5  # Use stall count as pseudo-anomaly
        )

        # Try to wire to active temperature schedulers
        self._wire_to_temperature_schedulers(config_key, boost)

    def reduce_exploration_after_improvement(self, config_key: str) -> None:
        """Gradually reduce exploration boost when training is improving.

        Called when training metrics improve, indicating model is making
        progress and doesn't need as much exploration.

        Args:
            config_key: Configuration key
        """
        # Get current boost from local state
        state = self._get_or_create_state(config_key)
        current_boost = state.current_exploration_boost

        if current_boost > 1.0:
            # Reduce towards 1.0 using decay factor
            new_boost = max(1.0, current_boost * EXPLORATION_BOOST_DECAY)
            state.current_exploration_boost = new_boost

            self._exploration_reductions_applied += 1

            logger.debug(
                f"[ExplorationFeedbackHandler] Reduced exploration boost to {new_boost:.2f}x "
                f"for {config_key} (training improving)"
            )

            # Try to wire to active temperature schedulers
            self._wire_to_temperature_schedulers(config_key, new_boost)

    def _emit_exploration_boost(
        self,
        config_key: str,
        boost_factor: float,
        reason: str,
        anomaly_count: int = 0,
    ) -> None:
        """Emit EXPLORATION_BOOST event."""
        if HAS_EXPLORATION_EVENTS and emit_exploration_boost:
            try:
                _safe_create_task(
                    emit_exploration_boost(
                        config_key=config_key,
                        boost_factor=boost_factor,
                        reason=reason,
                        anomaly_count=anomaly_count,
                        source="ExplorationFeedbackHandler",
                    ),
                    context=f"emit_exploration_boost:{reason}:{config_key}",
                )
                logger.debug(
                    f"[ExplorationFeedbackHandler] Emitted EXPLORATION_BOOST event for {config_key}"
                )
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.warning(f"[ExplorationFeedbackHandler] Failed to emit EXPLORATION_BOOST: {e}")

    def _wire_to_temperature_schedulers(self, config_key: str, boost: float) -> None:
        """Wire exploration boost to active temperature schedulers."""
        try:
            from app.training.temperature_scheduling import get_active_schedulers

            schedulers = get_active_schedulers()
            wired = False
            for scheduler in schedulers:
                if hasattr(scheduler, 'config_key') and scheduler.config_key == config_key:
                    if hasattr(scheduler, 'set_exploration_boost'):
                        scheduler.set_exploration_boost(boost)
                        wired = True
            if wired:
                logger.debug("[ExplorationFeedbackHandler] Wired boost to temperature scheduler")
        except ImportError:
            logger.debug("[ExplorationFeedbackHandler] Temperature scheduling not available")
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[ExplorationFeedbackHandler] Could not wire to scheduler: {e}")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_loss_anomaly_detected(self, event: Any) -> None:
        """Handle LOSS_ANOMALY_DETECTED event."""
        # December 30, 2025: Use consolidated extraction from HandlerBase
        payload = self._get_payload(event)
        config_key = payload.get("config_key", "")
        anomaly_count = payload.get("anomaly_count", 1)

        if config_key:
            self.boost_exploration_for_anomaly(config_key, anomaly_count)

    def _on_training_stall_detected(self, event: Any) -> None:
        """Handle TRAINING_STALL_DETECTED event."""
        # December 30, 2025: Use consolidated extraction from HandlerBase
        payload = self._get_payload(event)
        config_key = payload.get("config_key", "")
        stall_epochs = payload.get("stall_epochs", 5)

        if config_key:
            self.boost_exploration_for_stall(config_key, stall_epochs)

    def _on_training_improved(self, event: Any) -> None:
        """Handle TRAINING_IMPROVED event."""
        # December 30, 2025: Use consolidated extraction from HandlerBase
        payload = self._get_payload(event)
        config_key = payload.get("config_key", "")

        if config_key:
            self.reduce_exploration_after_improvement(config_key)

    def _on_elo_plateau_detected(self, event: Any) -> None:
        """Handle ELO_PLATEAU_DETECTED event.

        When Elo plateaus, boost exploration to try different strategies.
        """
        # December 30, 2025: Use consolidated extraction from HandlerBase
        payload = self._get_payload(event)
        config_key = payload.get("config_key", "")
        plateau_duration_hours = payload.get("duration_hours", 24)

        if config_key:
            # Convert hours to equivalent stall epochs (rough approximation)
            stall_epochs = max(5, int(plateau_duration_hours / 2))
            self.boost_exploration_for_stall(config_key, stall_epochs)
            logger.info(
                f"[ExplorationFeedbackHandler] Boosted exploration for Elo plateau: "
                f"{config_key} (plateau for {plateau_duration_hours}h)"
            )

    # =========================================================================
    # Public API for other components
    # =========================================================================

    def get_exploration_boost(self, config_key: str) -> float:
        """Get current exploration boost for a config.

        Args:
            config_key: Configuration key

        Returns:
            Current exploration boost factor (1.0 = normal)
        """
        state = self._get_or_create_state(config_key)
        return state.current_exploration_boost

    def set_exploration_boost(self, config_key: str, boost: float) -> None:
        """Set exploration boost for a config.

        Args:
            config_key: Configuration key
            boost: Boost factor (1.0 = normal, >1.0 = more exploration)
        """
        state = self._get_or_create_state(config_key)
        state.current_exploration_boost = min(EXPLORATION_BOOST_MAX, max(1.0, boost))
        logger.debug(f"[ExplorationFeedbackHandler] Set exploration boost for {config_key}: {boost:.2f}")

    def health_check(self) -> HealthCheckResult:
        """Return health check result for DaemonManager integration."""
        base_result = super().health_check()

        # Add exploration-specific details
        base_result.details["exploration_boosts_applied"] = self._exploration_boosts_applied
        base_result.details["exploration_reductions_applied"] = self._exploration_reductions_applied
        base_result.details["last_boost_time"] = self._last_boost_time
        base_result.details["states_tracked"] = len(self._states)

        return base_result


# Singleton accessor
_handler_instance: ExplorationFeedbackHandler | None = None


def get_exploration_feedback_handler(
    states: dict[str, "FeedbackState"] | None = None,
    get_or_create_state_fn: callable | None = None,
) -> ExplorationFeedbackHandler:
    """Get or create the singleton ExplorationFeedbackHandler instance."""
    return ExplorationFeedbackHandler.get_instance(states, get_or_create_state_fn)


def reset_exploration_feedback_handler() -> None:
    """Reset the singleton for testing."""
    ExplorationFeedbackHandler.reset_instance()


__all__ = [
    "ExplorationFeedbackHandler",
    "get_exploration_feedback_handler",
    "reset_exploration_feedback_handler",
]
