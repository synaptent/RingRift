"""Feedback Loop Controller - Central orchestration of training feedback signals.

This is the "central nervous system" of the training improvement loop.
It orchestrates feedback between:
- Selfplay quality → Training intensity
- Training metrics → Curriculum adjustment
- Evaluation results → Promotion decisions
- Promotion outcomes → Selfplay exploration

The goal is to create a positive feedback loop where:
1. Good models train faster (accelerated intensity)
2. Struggling models get more exploration
3. Failed promotions trigger more aggressive training
4. Successful promotions shift curriculum to harder tasks

Usage:
    from app.coordination.feedback_loop_controller import (
        FeedbackLoopController,
        get_feedback_loop_controller,
    )

    # Get singleton instance
    controller = get_feedback_loop_controller()

    # Start the feedback loop (as daemon)
    await controller.start()

    # Manual signals (usually automatic via events)
    controller.signal_selfplay_quality("square8_2p", quality_score=0.85)
    controller.signal_training_complete("square8_2p", policy_accuracy=0.78)

December 2025: Created as part of Phase 7 feedback loop integration.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FeedbackState:
    """State tracking for a single config's feedback loop."""

    config_key: str

    # Quality metrics
    last_selfplay_quality: float = 0.0
    last_training_accuracy: float = 0.0
    last_evaluation_win_rate: float = 0.0

    # Status
    last_promotion_success: bool | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # Timing
    last_selfplay_time: float = 0.0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    last_promotion_time: float = 0.0

    # Feedback signals applied
    current_training_intensity: str = "normal"  # normal, accelerated, hot_path, reduced
    current_exploration_boost: float = 1.0  # 1.0 = normal, >1.0 = more exploration
    current_curriculum_weight: float = 1.0  # curriculum priority multiplier


class FeedbackLoopController:
    """Central controller orchestrating all feedback signals.

    Subscribes to:
    - SELFPLAY_COMPLETE: Assess data quality, adjust training intensity
    - TRAINING_COMPLETE: Trigger evaluation, adjust curriculum
    - EVALUATION_COMPLETED: Consider promotion, record results
    - PROMOTION_COMPLETE: Adjust curriculum and exploration based on outcome

    Emits decisions that are consumed by:
    - FeedbackAccelerator: Training intensity signals
    - CurriculumFeedback: Curriculum weight adjustments
    - TemperatureScheduler: Exploration rate adjustments
    """

    def __init__(self):
        self._states: dict[str, FeedbackState] = {}
        self._running = False
        self._subscribed = False
        self._lock = threading.Lock()

        # Configuration
        self.policy_accuracy_threshold = 0.75  # Trigger evaluation above this
        self.promotion_threshold = 0.55  # Win rate for promotion consideration
        self.failure_exploration_boost = 1.3  # Boost exploration on failure
        self.success_intensity_reduction = 0.9  # Reduce intensity on success

    def _get_or_create_state(self, config_key: str) -> FeedbackState:
        """Get or create state for a config."""
        with self._lock:
            if config_key not in self._states:
                self._states[config_key] = FeedbackState(config_key=config_key)
            return self._states[config_key]

    async def start(self) -> None:
        """Start the feedback loop controller.

        Subscribes to all relevant events and begins orchestration.
        """
        if self._running:
            logger.warning("FeedbackLoopController already running")
            return

        self._running = True

        # Subscribe to events
        self._subscribe_to_events()

        # Wire all curriculum feedback integrations
        self._wire_curriculum_feedback()

        logger.info("[FeedbackLoopController] Started feedback loop orchestration")

    async def stop(self) -> None:
        """Stop the feedback loop controller."""
        self._running = False
        self._unsubscribe_from_events()
        logger.info("[FeedbackLoopController] Stopped")

    def is_running(self) -> bool:
        """Check if controller is running."""
        return self._running

    def _subscribe_to_events(self) -> None:
        """Subscribe to all relevant training events."""
        if self._subscribed:
            return

        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            bus.subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)
            bus.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
            bus.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_complete)
            bus.subscribe(DataEventType.PROMOTION_COMPLETE, self._on_promotion_complete)

            self._subscribed = True
            logger.info("[FeedbackLoopController] Subscribed to 4 event types")
        except Exception as e:
            logger.warning(f"[FeedbackLoopController] Failed to subscribe: {e}")

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)
            bus.unsubscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
            bus.unsubscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_complete)
            bus.unsubscribe(DataEventType.PROMOTION_COMPLETE, self._on_promotion_complete)

            self._subscribed = False
        except Exception as e:
            logger.debug(f"[FeedbackLoopController] Error unsubscribing: {e}")

    def _wire_curriculum_feedback(self) -> None:
        """Wire all curriculum feedback integrations."""
        try:
            from app.training.curriculum_feedback import wire_all_curriculum_feedback

            watchers = wire_all_curriculum_feedback(
                elo_significant_change=30.0,
                plateau_cooldown_seconds=600.0,
                tournament_cooldown_seconds=300.0,
                promotion_failure_urgency="high",
                auto_export=True,
            )

            logger.info(
                f"[FeedbackLoopController] Wired {len(watchers)} curriculum feedback integrations"
            )
        except ImportError:
            logger.warning(
                "[FeedbackLoopController] curriculum_feedback module not available"
            )
        except Exception as e:
            logger.warning(f"[FeedbackLoopController] Failed to wire curriculum feedback: {e}")

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle selfplay completion.

        Actions:
        1. Assess data quality
        2. Update training intensity based on quality
        3. Signal training readiness if quality is sufficient
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = payload.get("config", "")
            games_count = payload.get("games_count", 0)
            db_path = payload.get("db_path", "")

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_selfplay_time = time.time()

            # Assess data quality
            quality_score = self._assess_selfplay_quality(db_path, games_count)
            state.last_selfplay_quality = quality_score

            logger.info(
                f"[FeedbackLoopController] Selfplay complete for {config_key}: "
                f"{games_count} games, quality={quality_score:.2f}"
            )

            # Update training intensity based on quality
            self._update_training_intensity(config_key, quality_score)

            # Signal training readiness if quality is good
            if quality_score >= 0.6:
                self._signal_training_ready(config_key, quality_score)

        except Exception as e:
            logger.error(f"[FeedbackLoopController] Error handling selfplay complete: {e}")

    def _on_training_complete(self, event: Any) -> None:
        """Handle training completion.

        Actions:
        1. Record training metrics
        2. Trigger evaluation if accuracy threshold met
        3. Adjust curriculum based on metrics
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = payload.get("config", "")
            policy_accuracy = payload.get("policy_accuracy", 0.0)
            value_accuracy = payload.get("value_accuracy", 0.0)
            model_path = payload.get("model_path", "")

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_training_time = time.time()
            state.last_training_accuracy = policy_accuracy

            logger.info(
                f"[FeedbackLoopController] Training complete for {config_key}: "
                f"policy_acc={policy_accuracy:.2%}, value_acc={value_accuracy:.2%}"
            )

            # Trigger evaluation if accuracy is good enough
            if policy_accuracy >= self.policy_accuracy_threshold:
                self._trigger_evaluation(config_key, model_path)

            # Record training in curriculum
            self._record_training_in_curriculum(config_key)

            # Emit curriculum adjustment event based on training metrics
            # Low accuracy = model needs more training data (boost weight)
            # High accuracy = model is learning well (maintain or reduce weight)
            self._emit_curriculum_training_feedback(
                config_key, policy_accuracy, value_accuracy
            )

        except Exception as e:
            logger.error(f"[FeedbackLoopController] Error handling training complete: {e}")

    def _emit_curriculum_training_feedback(
        self,
        config_key: str,
        policy_accuracy: float,
        value_accuracy: float,
    ) -> None:
        """Emit curriculum feedback event based on training metrics.

        December 2025: Closes the training → curriculum feedback loop.
        Low training accuracy indicates the model needs more/better data,
        triggering curriculum weight adjustments.
        """
        try:
            from app.coordination.event_router import DataEvent, DataEventType, get_event_bus
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()

            # Determine curriculum adjustment based on accuracy
            # Low accuracy (<0.65) = boost training weight
            # High accuracy (>0.80) = reduce training weight
            adjustment = 0.0
            if policy_accuracy < 0.65:
                adjustment = 0.15  # Boost - needs more training
            elif policy_accuracy > 0.80:
                adjustment = -0.10  # Reduce - learning well

            if adjustment != 0.0:
                # Update internal weights
                current_weight = feedback._current_weights.get(config_key, 1.0)
                new_weight = max(
                    feedback.weight_min,
                    min(feedback.weight_max, current_weight + adjustment)
                )
                feedback._current_weights[config_key] = new_weight

                logger.info(
                    f"[FeedbackLoopController] Curriculum adjusted for {config_key}: "
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
                source="feedback_loop_controller",
            )

            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(bus.publish(event))
            except RuntimeError:
                asyncio.run(bus.publish(event))

            logger.debug(
                f"[FeedbackLoopController] Emitted CURRICULUM_REBALANCED (training_complete)"
            )

        except ImportError:
            pass  # Event system not available
        except Exception as e:
            logger.debug(f"[FeedbackLoopController] Failed to emit curriculum event: {e}")

    def _on_evaluation_complete(self, event: Any) -> None:
        """Handle evaluation completion.

        Actions:
        1. Record evaluation results
        2. Consider promotion if win rate threshold met
        3. Adjust curriculum based on performance
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            config_key = payload.get("config", "")
            win_rate = payload.get("win_rate", 0.0)
            elo = payload.get("elo", 1500.0)
            model_path = payload.get("model_path", "")

            if not config_key:
                return

            state = self._get_or_create_state(config_key)
            state.last_evaluation_time = time.time()
            state.last_evaluation_win_rate = win_rate

            logger.info(
                f"[FeedbackLoopController] Evaluation complete for {config_key}: "
                f"win_rate={win_rate:.2%}, elo={elo:.0f}"
            )

            # Consider promotion if threshold met
            if win_rate >= self.promotion_threshold:
                self._consider_promotion(config_key, model_path, win_rate, elo)

        except Exception as e:
            logger.error(f"[FeedbackLoopController] Error handling evaluation complete: {e}")

    def _on_promotion_complete(self, event: Any) -> None:
        """Handle promotion completion.

        Actions:
        1. Record promotion outcome
        2. Adjust exploration rate based on outcome
        3. Update training intensity based on outcome
        4. Signal curriculum adjustments
        """
        payload = event.payload if hasattr(event, "payload") else {}
        metadata = payload.get("metadata", payload)

        config_key = metadata.get("config") or metadata.get("config_key", "")
        promoted = metadata.get("promoted", False)
        new_elo = metadata.get("new_elo") or metadata.get("elo")

        if not config_key:
            return

        state = self._get_or_create_state(config_key)
        state.last_promotion_time = time.time()
        state.last_promotion_success = promoted

        if promoted:
            state.consecutive_successes += 1
            state.consecutive_failures = 0
            logger.info(
                f"[FeedbackLoopController] Promotion SUCCESS for {config_key} "
                f"(streak: {state.consecutive_successes})"
            )

            # Reduce intensity on success (model is performing well)
            state.current_training_intensity = "normal"
            state.current_exploration_boost = max(1.0, state.current_exploration_boost - 0.1)

        else:
            state.consecutive_failures += 1
            state.consecutive_successes = 0
            logger.info(
                f"[FeedbackLoopController] Promotion FAILED for {config_key} "
                f"(streak: {state.consecutive_failures})"
            )

            # Boost exploration and intensity on failure
            state.current_exploration_boost = min(2.0, state.current_exploration_boost + 0.2)

            if state.consecutive_failures >= 3:
                state.current_training_intensity = "hot_path"
            else:
                state.current_training_intensity = "accelerated"

            # Signal urgent training needed
            self._signal_urgent_training(config_key, state.consecutive_failures)

        # Apply feedback to FeedbackAccelerator
        self._apply_intensity_feedback(config_key, state)

    # =========================================================================
    # Internal Actions
    # =========================================================================

    def _assess_selfplay_quality(self, db_path: str, games_count: int) -> float:
        """Assess quality of selfplay data.

        Returns:
            Quality score 0.0-1.0
        """
        # Basic quality assessment based on game count
        if games_count < 100:
            return 0.3
        elif games_count < 500:
            return 0.6
        elif games_count < 1000:
            return 0.8
        else:
            return 0.95

    def _update_training_intensity(self, config_key: str, quality_score: float) -> None:
        """Update training intensity based on data quality."""
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            accelerator = get_feedback_accelerator()

            if quality_score >= 0.8:
                accelerator.signal_training_needed(
                    config_key=config_key,
                    urgency="high",
                    reason="high_quality_selfplay_data",
                )
            elif quality_score >= 0.6:
                accelerator.signal_training_needed(
                    config_key=config_key,
                    urgency="normal",
                    reason="adequate_selfplay_data",
                )

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to update training intensity: {e}")

    def _signal_training_ready(self, config_key: str, quality_score: float) -> None:
        """Signal that training data is ready."""
        try:
            from app.coordination.event_emitters import emit_data_quality_assessed

            asyncio.create_task(
                emit_data_quality_assessed(
                    config=config_key,
                    quality_score=quality_score,
                    samples_available=0,  # Unknown here
                    ready_for_training=True,
                )
            )
        except (ImportError, RuntimeError):
            pass

    def _trigger_evaluation(self, config_key: str, model_path: str) -> None:
        """Trigger model evaluation."""
        logger.info(f"[FeedbackLoopController] Triggering evaluation for {config_key}")

        try:
            from app.coordination.pipeline_actions import trigger_evaluation

            asyncio.create_task(
                trigger_evaluation({
                    "config": config_key,
                    "model_path": model_path,
                })
            )
        except (ImportError, RuntimeError):
            pass

    def _record_training_in_curriculum(self, config_key: str) -> None:
        """Record training completion in curriculum."""
        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            feedback.record_training(config_key)
        except ImportError:
            pass

    def _consider_promotion(
        self,
        config_key: str,
        model_path: str,
        win_rate: float,
        elo: float,
    ) -> None:
        """Consider model for promotion."""
        logger.info(
            f"[FeedbackLoopController] Considering promotion for {config_key}: "
            f"win_rate={win_rate:.2%}, elo={elo:.0f}"
        )

        try:
            from app.coordination.pipeline_actions import trigger_promotion

            asyncio.create_task(
                trigger_promotion({
                    "config": config_key,
                    "model_path": model_path,
                    "win_rate": win_rate,
                    "elo": elo,
                })
            )
        except (ImportError, RuntimeError):
            pass

    def _signal_urgent_training(self, config_key: str, failure_count: int) -> None:
        """Signal that urgent training is needed after failures."""
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            accelerator = get_feedback_accelerator()

            if failure_count >= 3:
                urgency = "critical"
            elif failure_count >= 2:
                urgency = "high"
            else:
                urgency = "normal"

            accelerator.signal_training_needed(
                config_key=config_key,
                urgency=urgency,
                reason=f"consecutive_promotion_failures:{failure_count}",
            )
        except ImportError:
            pass

    def _apply_intensity_feedback(self, config_key: str, state: FeedbackState) -> None:
        """Apply current feedback state to FeedbackAccelerator."""
        try:
            from app.training.feedback_accelerator import (
                TrainingIntensity,
                get_feedback_accelerator,
            )

            accelerator = get_feedback_accelerator()

            # Map intensity string to enum
            intensity_map = {
                "paused": TrainingIntensity.PAUSED,
                "reduced": TrainingIntensity.REDUCED,
                "normal": TrainingIntensity.NORMAL,
                "accelerated": TrainingIntensity.ACCELERATED,
                "hot_path": TrainingIntensity.HOT_PATH,
            }

            intensity = intensity_map.get(
                state.current_training_intensity,
                TrainingIntensity.NORMAL,
            )

            # Update accelerator state
            accelerator.set_intensity(config_key, intensity)

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to apply intensity feedback: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def get_state(self, config_key: str) -> FeedbackState | None:
        """Get current feedback state for a config."""
        return self._states.get(config_key)

    def get_all_states(self) -> dict[str, FeedbackState]:
        """Get all feedback states."""
        return dict(self._states)

    def get_summary(self) -> dict[str, Any]:
        """Get feedback loop summary."""
        return {
            "running": self._running,
            "subscribed": self._subscribed,
            "configs_tracked": len(self._states),
            "states": {
                k: {
                    "training_intensity": v.current_training_intensity,
                    "exploration_boost": v.current_exploration_boost,
                    "consecutive_successes": v.consecutive_successes,
                    "consecutive_failures": v.consecutive_failures,
                    "last_selfplay_quality": v.last_selfplay_quality,
                    "last_training_accuracy": v.last_training_accuracy,
                    "last_evaluation_win_rate": v.last_evaluation_win_rate,
                }
                for k, v in self._states.items()
            },
        }

    def signal_selfplay_quality(self, config_key: str, quality_score: float) -> None:
        """Manually signal selfplay quality (for testing/manual intervention)."""
        state = self._get_or_create_state(config_key)
        state.last_selfplay_quality = quality_score
        state.last_selfplay_time = time.time()
        self._update_training_intensity(config_key, quality_score)

    def signal_training_complete(
        self,
        config_key: str,
        policy_accuracy: float,
        value_accuracy: float = 0.0,
    ) -> None:
        """Manually signal training completion (for testing/manual intervention)."""
        state = self._get_or_create_state(config_key)
        state.last_training_accuracy = policy_accuracy
        state.last_training_time = time.time()

        if policy_accuracy >= self.policy_accuracy_threshold:
            logger.info(f"Manual training signal: {config_key} ready for evaluation")


# =============================================================================
# Singleton
# =============================================================================

_controller: FeedbackLoopController | None = None
_controller_lock = threading.Lock()


def get_feedback_loop_controller() -> FeedbackLoopController:
    """Get the singleton FeedbackLoopController instance."""
    global _controller
    if _controller is None:
        with _controller_lock:
            if _controller is None:
                _controller = FeedbackLoopController()
    return _controller


def reset_feedback_loop_controller() -> None:
    """Reset the singleton (for testing)."""
    global _controller
    if _controller is not None:
        asyncio.create_task(_controller.stop())
    _controller = None


__all__ = [
    "FeedbackLoopController",
    "FeedbackState",
    "get_feedback_loop_controller",
    "reset_feedback_loop_controller",
]
