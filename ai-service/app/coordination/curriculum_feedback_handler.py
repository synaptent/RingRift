"""Curriculum Feedback Handler - Handles curriculum weight adjustments.

Extracted from FeedbackLoopController (December 2025) to reduce file size
and improve maintainability.

This handler manages:
- Curriculum weight updates based on selfplay quality
- Curriculum feedback from training metrics
- Recording training completion in curriculum
- Adaptive promotion threshold calculation

Usage:
    from app.coordination.curriculum_feedback_handler import (
        CurriculumFeedbackHandler,
        get_curriculum_feedback_handler,
    )

    handler = get_curriculum_feedback_handler()
    await handler.start()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from app.config.thresholds import (
    CURRICULUM_WEIGHT_ADJUSTMENT_DOWN,
    CURRICULUM_WEIGHT_ADJUSTMENT_UP,
    ELO_FAST_IMPROVEMENT_PER_HOUR,
    ELO_PLATEAU_PER_HOUR,
    POLICY_HIGH_THRESHOLD,
    POLICY_LOW_THRESHOLD,
)
from app.coordination.handler_base import HandlerBase
from app.coordination.protocols import HealthCheckResult
from app.coordination.event_handler_utils import extract_config_key
from app.coordination.event_emission_helpers import safe_emit_event

if TYPE_CHECKING:
    from app.coordination.feedback_loop_controller import FeedbackState

logger = logging.getLogger(__name__)


def _handle_task_error(task: asyncio.Task, context: str = "") -> None:
    """Handle errors from fire-and-forget tasks."""
    try:
        exc = task.exception()
        if exc is not None:
            logger.error(f"[CurriculumFeedbackHandler] Task error{' in ' + context if context else ''}: {exc}")
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
        logger.debug(f"[CurriculumFeedbackHandler] Could not create task for {context}: {e}")
        return None


class CurriculumFeedbackHandler(HandlerBase):
    """Handles curriculum-related feedback signals.

    Extracted from FeedbackLoopController to reduce file size.
    This handler processes curriculum weight adjustments and training feedback.

    Key behaviors:
    - Update curriculum weights based on selfplay quality
    - Adjust curriculum based on training accuracy
    - Record training completion in curriculum system
    - Calculate adaptive promotion thresholds
    """

    _instance: CurriculumFeedbackHandler | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        states: dict[str, "FeedbackState"] | None = None,
        get_or_create_state_fn: callable | None = None,
    ):
        """Initialize the curriculum feedback handler.

        Args:
            states: Shared state dictionary from FeedbackLoopController
            get_or_create_state_fn: Function to get or create FeedbackState
        """
        super().__init__(name="curriculum_feedback_handler", cycle_interval=60.0)

        # State management (shared with FeedbackLoopController)
        self._states = states if states is not None else {}
        self._get_or_create_state_fn = get_or_create_state_fn

        # Metrics
        self._curriculum_adjustments = 0
        self._training_recordings = 0
        self._last_adjustment_time = 0.0
        self._last_cycle_time = 0.0

    @classmethod
    def get_instance(
        cls,
        states: dict[str, "FeedbackState"] | None = None,
        get_or_create_state_fn: callable | None = None,
    ) -> "CurriculumFeedbackHandler":
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
        """Return event subscriptions for this handler.

        December 30, 2025: Added EVALUATION_STARTED to prepare curriculum updates.
        """
        return {
            "SELFPLAY_QUALITY_ASSESSED": self._on_selfplay_quality_assessed,
            "TRAINING_METRICS_AVAILABLE": self._on_training_metrics_available,
            "TRAINING_COMPLETED": self._on_training_completed,
            "evaluation_started": self._on_evaluation_started,
        }

    async def _on_evaluation_started(self, event) -> None:
        """Handle EVALUATION_STARTED event to prepare curriculum updates.

        December 30, 2025: Added to close the feedback loop gap.
        When evaluation starts, we can pre-compute curriculum adjustments
        that will be applied once EVALUATION_COMPLETED arrives.

        Args:
            event: Event with payload containing config_key, model_path, etc.
        """
        payload = self._get_payload(event)
        config_key = self._extract_config_key(payload)

        if config_key and config_key != "unknown":
            # Mark that evaluation is in progress for this config
            state = self._get_or_create_state(config_key)
            state.evaluation_in_progress = True
            logger.debug(
                f"[CurriculumFeedbackHandler] Evaluation started for {config_key}, "
                f"preparing curriculum updates"
            )

    async def _run_cycle(self) -> None:
        """Run one cycle of the curriculum feedback handler.

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
    # Curriculum Weight Methods
    # =========================================================================

    def update_curriculum_weight_from_selfplay(
        self, config_key: str, quality_score: float
    ) -> None:
        """Update curriculum weight based on selfplay quality.

        Creates feedback path from selfplay quality to curriculum weights.
        Configs with quality issues get more training attention (higher weight)
        while stable configs can have slightly reduced priority.

        Logic:
        - Low quality (< 0.5): Increase weight by 15% (needs attention)
        - Medium quality (0.5-0.7): No change
        - High quality (>= 0.7): Decrease weight by 5% (stable, less urgent)

        Args:
            config_key: Configuration key
            quality_score: Quality score (0-1)
        """
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            state = self._get_or_create_state(config_key)

            old_weight = state.current_curriculum_weight
            new_weight = old_weight

            if quality_score < 0.5:
                # Low quality - needs more training focus
                new_weight = min(2.0, old_weight * 1.15)
            elif quality_score >= 0.7:
                # High quality - can reduce priority slightly
                new_weight = max(0.5, old_weight * 0.95)
            # Medium quality - no change

            if new_weight != old_weight:
                state.current_curriculum_weight = new_weight
                self._curriculum_adjustments += 1
                self._last_adjustment_time = time.time()

                # Propagate to curriculum feedback system
                if hasattr(feedback, "_current_weights"):
                    feedback._current_weights[config_key] = new_weight

                logger.info(
                    f"[CurriculumFeedbackHandler] Curriculum weight for {config_key}: "
                    f"{old_weight:.2f} → {new_weight:.2f} (quality={quality_score:.2f})"
                )

                # Emit CURRICULUM_REBALANCED event for SelfplayScheduler
                self._emit_curriculum_rebalanced(
                    config_key, new_weight, f"selfplay_quality_{quality_score:.2f}"
                )

        except ImportError:
            pass
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Failed to update curriculum weight: {e}")

    def emit_curriculum_training_feedback(
        self,
        config_key: str,
        policy_accuracy: float,
        value_accuracy: float,
    ) -> None:
        """Emit curriculum feedback event based on training metrics.

        Closes the training → curriculum feedback loop.
        Low training accuracy indicates the model needs more/better data,
        triggering curriculum weight adjustments.

        Args:
            config_key: Configuration key
            policy_accuracy: Policy head accuracy
            value_accuracy: Value head accuracy
        """
        try:
            from app.coordination.event_router import DataEvent, DataEventType, get_event_bus
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()

            # Determine curriculum adjustment based on accuracy
            adjustment = 0.0
            if policy_accuracy < POLICY_LOW_THRESHOLD:
                adjustment = CURRICULUM_WEIGHT_ADJUSTMENT_UP  # Boost - needs more training
            elif policy_accuracy > POLICY_HIGH_THRESHOLD:
                adjustment = CURRICULUM_WEIGHT_ADJUSTMENT_DOWN  # Reduce - learning well

            if adjustment != 0.0:
                # Update internal weights
                current_weight = feedback._current_weights.get(config_key, 1.0)
                new_weight = max(
                    feedback.weight_min,
                    min(feedback.weight_max, current_weight + adjustment)
                )
                feedback._current_weights[config_key] = new_weight

                self._curriculum_adjustments += 1

                logger.info(
                    f"[CurriculumFeedbackHandler] Curriculum adjusted for {config_key}: "
                    f"policy_acc={policy_accuracy:.2%} → weight {current_weight:.2f} → {new_weight:.2f}"
                )

            # Emit CURRICULUM_REBALANCED event for downstream listeners
            bus = get_event_bus()
            event = DataEvent(
                event_type=DataEventType.CURRICULUM_REBALANCED,
                payload={
                    "config": config_key,
                    "trigger": "training_complete",
                    "policy_accuracy": policy_accuracy,
                    "value_accuracy": value_accuracy,
                    "adjustment": adjustment,
                    "new_weights": dict(feedback._current_weights),
                    "timestamp": time.time(),
                },
                source="curriculum_feedback_handler",
            )

            try:
                _safe_create_task(bus.publish(event), "curriculum_event_publish")
            except RuntimeError:
                asyncio.run(bus.publish(event))

            logger.debug("[CurriculumFeedbackHandler] Emitted CURRICULUM_REBALANCED (training_complete)")

        except ImportError:
            pass  # Event system not available
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[CurriculumFeedbackHandler] Failed to emit curriculum event: {e}")

    def record_training_in_curriculum(self, config_key: str) -> None:
        """Record training completion in curriculum system.

        Args:
            config_key: Configuration key
        """
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            feedback.record_training(config_key)
            self._training_recordings += 1
        except ImportError:
            pass
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"[CurriculumFeedbackHandler] Failed to record training: {e}")

    def _emit_curriculum_rebalanced(
        self, config_key: str, new_weight: float, reason: str
    ) -> None:
        """Emit CURRICULUM_REBALANCED event."""
        safe_emit_event(
            "CURRICULUM_REBALANCED",
            {
                "config_key": config_key,
                "weight": new_weight,
                "reason": reason,
            },
            source="curriculum_feedback_handler",
            context="curriculum_rebalance",
        )

    # =========================================================================
    # Adaptive Thresholds
    # =========================================================================

    def get_adaptive_promotion_threshold(
        self, elo: float, state: "FeedbackState"
    ) -> float:
        """Get Elo-adaptive promotion threshold.

        Higher Elo models need stronger win rates to be promoted.
        This prevents promoting models that are only marginally better at high
        Elo levels, ensuring real progress before promotion.

        Elo-based threshold tiers:
        - Beginner (< 1300): 0.55 (55%) - easier to promote beginners
        - Intermediate (1300-1600): 0.60 (60%) - standard threshold
        - Advanced (1600-1800): 0.65 (65%) - need to show real progress
        - Elite (> 1800): 0.70 (70%) - near target, must be clearly better

        Additional modifiers:
        - Consecutive successes (3+): -0.03 (reward momentum)
        - Consecutive failures (2+): +0.02 (require more evidence)
        - Fast velocity (improving): -0.02 (capitalize on momentum)
        - Plateau (low velocity): +0.02 (wait for real improvement)

        Args:
            elo: Current Elo rating
            state: FeedbackState with velocity and success/failure tracking

        Returns:
            Adaptive promotion threshold (0.50 - 0.75 range)
        """
        # Base threshold based on Elo tier
        if elo < 1300:
            base_threshold = 0.55  # Beginner tier
        elif elo < 1600:
            base_threshold = 0.60  # Intermediate tier
        elif elo < 1800:
            base_threshold = 0.65  # Advanced tier
        else:
            base_threshold = 0.70  # Elite tier

        # Apply modifiers
        modifier = 0.0

        # Momentum modifier: reward consecutive successes
        if state.consecutive_successes >= 3:
            modifier -= 0.03
            logger.debug(
                f"[CurriculumFeedbackHandler] Threshold modifier -0.03 for "
                f"{state.consecutive_successes} consecutive successes"
            )

        # Caution modifier: require more evidence after failures
        if state.consecutive_failures >= 2:
            modifier += 0.02
            logger.debug(
                f"[CurriculumFeedbackHandler] Threshold modifier +0.02 for "
                f"{state.consecutive_failures} consecutive failures"
            )

        # Velocity modifier: adjust based on improvement speed
        elo_history = getattr(state, "elo_history", None) or []
        if len(elo_history) >= 3:
            velocity = getattr(state, "elo_velocity", 0.0)
            if velocity > ELO_FAST_IMPROVEMENT_PER_HOUR:
                modifier -= 0.02  # Fast improvement
                logger.debug(
                    f"[CurriculumFeedbackHandler] Threshold modifier -0.02 for "
                    f"fast velocity ({velocity:.1f} Elo/hr)"
                )
            elif velocity < ELO_PLATEAU_PER_HOUR:
                modifier += 0.02  # Plateau
                logger.debug(
                    f"[CurriculumFeedbackHandler] Threshold modifier +0.02 for "
                    f"plateau velocity ({velocity:.1f} Elo/hr)"
                )

        # Clamp final threshold
        threshold = max(0.50, min(0.75, base_threshold + modifier))

        logger.debug(
            f"[CurriculumFeedbackHandler] Adaptive threshold: base={base_threshold:.2f}, "
            f"modifier={modifier:+.2f}, final={threshold:.2f} (elo={elo:.0f})"
        )

        return threshold

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_selfplay_quality_assessed(self, event: Any) -> None:
        """Handle SELFPLAY_QUALITY_ASSESSED event."""
        # December 30, 2025: Use consolidated extraction from HandlerBase
        payload = self._get_payload(event)
        config_key = extract_config_key(payload)
        quality_score = payload.get("quality_score", 0.5)

        if config_key:
            self.update_curriculum_weight_from_selfplay(config_key, quality_score)

    def _on_training_metrics_available(self, event: Any) -> None:
        """Handle TRAINING_METRICS_AVAILABLE event."""
        # December 30, 2025: Use consolidated extraction from HandlerBase
        payload = self._get_payload(event)
        config_key = payload.get("config", "")
        policy_accuracy = payload.get("policy_accuracy", 0.0)
        value_accuracy = payload.get("value_accuracy", 0.0)

        if config_key:
            self.emit_curriculum_training_feedback(
                config_key, policy_accuracy, value_accuracy
            )

    def _on_training_completed(self, event: Any) -> None:
        """Handle TRAINING_COMPLETED event."""
        # December 30, 2025: Use consolidated extraction from HandlerBase
        payload = self._get_payload(event)
        config_key = payload.get("config", "")

        if config_key:
            self.record_training_in_curriculum(config_key)

    # =========================================================================
    # Public API
    # =========================================================================

    def get_curriculum_weight(self, config_key: str) -> float:
        """Get current curriculum weight for a config.

        Args:
            config_key: Configuration key

        Returns:
            Current curriculum weight (1.0 = normal)
        """
        state = self._get_or_create_state(config_key)
        return state.current_curriculum_weight

    def set_curriculum_weight(self, config_key: str, weight: float) -> None:
        """Set curriculum weight for a config.

        Args:
            config_key: Configuration key
            weight: Weight value (0.5 - 2.0 range)
        """
        state = self._get_or_create_state(config_key)
        state.current_curriculum_weight = max(0.5, min(2.0, weight))
        logger.debug(
            f"[CurriculumFeedbackHandler] Set curriculum weight for {config_key}: {weight:.2f}"
        )

    def health_check(self) -> HealthCheckResult:
        """Return health check result for DaemonManager integration."""
        base_result = super().health_check()

        # Add curriculum-specific details
        base_result.details["curriculum_adjustments"] = self._curriculum_adjustments
        base_result.details["training_recordings"] = self._training_recordings
        base_result.details["last_adjustment_time"] = self._last_adjustment_time
        base_result.details["states_tracked"] = len(self._states)

        return base_result


# Singleton accessor
_handler_instance: CurriculumFeedbackHandler | None = None


def get_curriculum_feedback_handler(
    states: dict[str, "FeedbackState"] | None = None,
    get_or_create_state_fn: callable | None = None,
) -> CurriculumFeedbackHandler:
    """Get or create the singleton CurriculumFeedbackHandler instance."""
    return CurriculumFeedbackHandler.get_instance(states, get_or_create_state_fn)


def reset_curriculum_feedback_handler() -> None:
    """Reset the singleton for testing."""
    CurriculumFeedbackHandler.reset_instance()


__all__ = [
    "CurriculumFeedbackHandler",
    "get_curriculum_feedback_handler",
    "reset_curriculum_feedback_handler",
]
