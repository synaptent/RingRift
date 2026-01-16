"""Training curriculum feedback handling for FeedbackLoopController.

Sprint 17.9 (Jan 16, 2026): Extracted from feedback_loop_controller.py (~120 LOC)

This mixin provides training → curriculum feedback logic that:
- Emits curriculum rebalancing events based on training accuracy
- Records training feedback to Elo database for velocity tracking

Usage:
    class FeedbackLoopController(TrainingCurriculumFeedbackMixin, ...):
        pass
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from app.config.thresholds import (
    POLICY_LOW_THRESHOLD,
    POLICY_HIGH_THRESHOLD,
    CURRICULUM_WEIGHT_ADJUSTMENT_UP,
    CURRICULUM_WEIGHT_ADJUSTMENT_DOWN,
)
from app.coordination.event_utils import parse_config_key

if TYPE_CHECKING:
    from app.coordination.feedback_loop_controller import FeedbackState

logger = logging.getLogger(__name__)


def _safe_create_task(coro, context: str = "") -> asyncio.Task | None:
    """Create a task with basic error handling.

    Note: This is a local helper. The main controller has a more sophisticated
    version with error tracking. This is used for mixin independence.
    """
    try:
        task = asyncio.create_task(coro)
        task.add_done_callback(
            lambda t: logger.debug(f"[TrainingCurriculum] Task {context} done")
            if not t.cancelled() and t.exception() is None
            else logger.warning(f"[TrainingCurriculum] Task {context} failed: {t.exception()}")
            if t.exception() else None
        )
        return task
    except RuntimeError as e:
        logger.debug(f"[TrainingCurriculum] Could not create task for {context}: {e}")
        return None


class TrainingCurriculumFeedbackMixin:
    """Mixin for training curriculum feedback handling in FeedbackLoopController.

    Requires the host class to implement:
    - _get_or_create_state(config_key: str) -> FeedbackState

    Provides:
    - _emit_curriculum_training_feedback(config_key, policy_accuracy, value_accuracy)
    - _record_training_feedback_to_db(config_key, state)
    """

    def _get_or_create_state(self, config_key: str) -> "FeedbackState":
        """Get or create state for a config. Must be implemented by host class."""
        raise NotImplementedError("Host class must implement _get_or_create_state")

    def _emit_curriculum_training_feedback(
        self,
        config_key: str,
        policy_accuracy: float,
        value_accuracy: float,
    ) -> None:
        """Emit curriculum feedback event based on training metrics.

        December 2025: Closes the training -> curriculum feedback loop.
        Low training accuracy indicates the model needs more/better data,
        triggering curriculum weight adjustments.
        """
        try:
            from app.coordination.event_router import DataEvent, DataEventType, get_event_bus
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()

            # Determine curriculum adjustment based on accuracy
            # Low accuracy = boost training weight, high accuracy = reduce
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

                logger.info(
                    f"[TrainingCurriculum] Curriculum adjusted for {config_key}: "
                    f"policy_acc={policy_accuracy:.2%} -> weight {current_weight:.2f} -> {new_weight:.2f}"
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
                source="training_curriculum_mixin",
            )

            try:
                _safe_create_task(bus.publish(event), "curriculum_event_publish")
            except RuntimeError:
                asyncio.run(bus.publish(event))

            logger.debug("[TrainingCurriculum] Emitted CURRICULUM_REBALANCED (training_complete)")

        except ImportError:
            pass  # Event system not available
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[TrainingCurriculum] Failed to emit curriculum event: {e}")

    def _record_training_feedback_to_db(
        self,
        config_key: str,
        state: "FeedbackState",
    ) -> None:
        """Record training feedback to Elo database for velocity tracking.

        January 2026: Added to fix empty training_feedback table.
        This enables Elo velocity tracking and curriculum progression signals.
        """
        try:
            from app.training.elo_service import EloService

            parsed = parse_config_key(config_key)
            if not parsed:
                return

            elo_service = EloService.get_instance()

            # Get current best Elo for this config
            best_elo = state.last_elo or 1500.0

            # Compute Elo delta from previous training
            prev_elo_key = f"_prev_elo_{config_key}"
            prev_elo = getattr(self, prev_elo_key, 1500.0)
            elo_delta = best_elo - prev_elo
            setattr(self, prev_elo_key, best_elo)

            # Get iteration count
            iter_key = f"_training_iteration_{config_key}"
            iteration = getattr(self, iter_key, 0) + 1
            setattr(self, iter_key, iteration)

            # Get curriculum stage from state
            curriculum_stage = getattr(state, "curriculum_stage", 0)

            elo_service.record_training_feedback(
                board_type=parsed.board_type,
                num_players=parsed.num_players,
                iteration=iteration,
                best_elo=best_elo,
                elo_delta=elo_delta,
                curriculum_stage=curriculum_stage,
            )

            logger.debug(
                f"[TrainingCurriculum] Recorded training feedback for {config_key}: "
                f"iter={iteration}, elo={best_elo:.0f}, delta={elo_delta:+.0f}"
            )

        except ImportError:
            pass  # EloService not available
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[TrainingCurriculum] Failed to record training feedback: {e}")


__all__ = ["TrainingCurriculumFeedbackMixin"]
