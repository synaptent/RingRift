"""Quality Feedback Handler - Handles quality-related feedback signals.

Extracted from FeedbackLoopController (December 2025) to reduce file size
and improve maintainability.

This handler manages:
- QUALITY_DEGRADED events → training acceleration
- QUALITY_CHECK_FAILED events → request more selfplay
- QUALITY_FEEDBACK_ADJUSTED events → adjust training intensity
- QUALITY_SCORE_UPDATED events → exploration adjustments
- HIGH_QUALITY_DATA_AVAILABLE events → training acceleration

Usage:
    from app.coordination.quality_feedback_handler import (
        QualityFeedbackHandler,
        get_quality_feedback_handler,
    )

    handler = get_quality_feedback_handler()
    await handler.start()
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config.thresholds import (
    EXPLORATION_BOOST_MAX,
    EXPLORATION_BOOST_PER_ANOMALY,
    MEDIUM_QUALITY_THRESHOLD,
)
from app.coordination.handler_base import HandlerBase
from app.coordination.protocols import HealthCheckResult

if TYPE_CHECKING:
    from app.coordination.feedback_loop_controller import FeedbackState

logger = logging.getLogger(__name__)

# Event emitter availability flags
try:
    from app.coordination.event_router import emit_selfplay_target_updated
    HAS_SELFPLAY_EVENTS = True
except ImportError:
    emit_selfplay_target_updated = None
    HAS_SELFPLAY_EVENTS = False

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
            logger.error(f"[QualityFeedbackHandler] Task error{' in ' + context if context else ''}: {exc}")
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
        logger.debug(f"[QualityFeedbackHandler] Could not create task for {context}: {e}")
        return None


class QualityFeedbackHandler(HandlerBase):
    """Handles quality-related feedback signals.

    Extracted from FeedbackLoopController to reduce file size.
    This handler processes quality events and adjusts training/exploration accordingly.

    Event Subscriptions:
    - QUALITY_DEGRADED: Accelerate training to fix quality issues
    - QUALITY_CHECK_FAILED: Request more selfplay data
    - QUALITY_FEEDBACK_ADJUSTED: Adjust training intensity
    - QUALITY_SCORE_UPDATED: Emit exploration adjustments
    - HIGH_QUALITY_DATA_AVAILABLE: Accelerate training
    """

    _instance: QualityFeedbackHandler | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        states: dict[str, "FeedbackState"] | None = None,
        get_or_create_state_fn: callable | None = None,
        boost_exploration_fn: callable | None = None,
    ):
        """Initialize the quality feedback handler.

        Args:
            states: Shared state dictionary from FeedbackLoopController
            get_or_create_state_fn: Function to get or create FeedbackState
            boost_exploration_fn: Function to boost exploration for stalls
        """
        super().__init__(name="quality_feedback_handler", cycle_interval=60.0)

        # State management (shared with FeedbackLoopController)
        self._states = states if states is not None else {}
        self._get_or_create_state_fn = get_or_create_state_fn
        self._boost_exploration_fn = boost_exploration_fn

        # Metrics
        self._quality_events_processed = 0
        self._last_quality_event_time = 0.0
        self._last_cycle_time = 0.0

    @classmethod
    def get_instance(
        cls,
        states: dict[str, "FeedbackState"] | None = None,
        get_or_create_state_fn: callable | None = None,
        boost_exploration_fn: callable | None = None,
    ) -> "QualityFeedbackHandler":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(states, get_or_create_state_fn, boost_exploration_fn)
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
            "QUALITY_DEGRADED": self._on_quality_degraded_for_training,
            "QUALITY_CHECK_FAILED": self._on_quality_check_failed,
            "QUALITY_FEEDBACK_ADJUSTED": self._on_quality_feedback_adjusted,
            "QUALITY_SCORE_UPDATED": self._on_quality_score_updated,
            "HIGH_QUALITY_DATA_AVAILABLE": self._on_high_quality_data_available,
        }

    async def _run_cycle(self) -> None:
        """Run one cycle of the quality feedback handler.

        This handler is primarily event-driven, so the cycle just updates metrics.
        """
        # Update health metrics
        self._last_cycle_time = time.time()

    def _get_or_create_state(self, config_key: str) -> "FeedbackState":
        """Get or create a FeedbackState for the given config.

        Uses the injected function if available, otherwise manages locally.
        """
        if self._get_or_create_state_fn:
            return self._get_or_create_state_fn(config_key)

        # Local fallback
        from app.coordination.feedback_loop_controller import FeedbackState

        if config_key not in self._states:
            self._states[config_key] = FeedbackState(config_key=config_key)
        return self._states[config_key]

    def _boost_exploration_for_stall(self, config_key: str, trend_duration_epochs: int = 3) -> None:
        """Boost exploration when quality stalls.

        Uses the injected function if available, otherwise implements locally.
        """
        if self._boost_exploration_fn:
            self._boost_exploration_fn(config_key, trend_duration_epochs)
            return

        # Local implementation
        state = self._get_or_create_state(config_key)
        boost = min(EXPLORATION_BOOST_MAX, 1.0 + 0.1 * trend_duration_epochs)
        state.current_exploration_boost = boost
        logger.info(f"[QualityFeedbackHandler] Set exploration boost to {boost:.2f}x for {config_key}")

    # =========================================================================
    # Event Handlers - Quality Events
    # =========================================================================

    def _on_quality_degraded_for_training(self, event: Any) -> None:
        """Handle QUALITY_DEGRADED events to adjust training thresholds.

        When quality degrades, we want to train MORE to fix the problem.
        This reduces the training threshold via ImprovementOptimizer.

        Actions:
        - Record low data quality score in ImprovementOptimizer
        - This triggers faster training cycles (lower threshold)
        - Also boost exploration to gather more diverse data
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            # Source tracking loop guard - skip events we emitted
            source = payload.get("source", "")
            if source in ("feedback_loop_controller", "FeedbackLoopController", "quality_feedback_handler"):
                logger.debug("[QualityFeedbackHandler] Skipping self-emitted QUALITY_DEGRADED event")
                return

            config_key = payload.get("config_key", "")
            quality_score = payload.get("quality_score", 0.5)
            threshold = payload.get("threshold", MEDIUM_QUALITY_THRESHOLD)

            if not config_key:
                return

            self._quality_events_processed += 1
            self._last_quality_event_time = time.time()

            logger.info(
                f"[QualityFeedbackHandler] Quality degraded for {config_key}: "
                f"score={quality_score:.2f} < threshold={threshold:.2f}, "
                f"triggering training acceleration"
            )

            # Update ImprovementOptimizer to reduce training threshold
            try:
                from app.training.improvement_optimizer import ImprovementOptimizer

                optimizer = ImprovementOptimizer.get_instance()
                optimizer.record_data_quality(
                    data_quality_score=quality_score,
                    parity_success_rate=quality_score,  # Use quality as proxy
                )
                logger.info(
                    f"[QualityFeedbackHandler] Updated ImprovementOptimizer for {config_key}: "
                    f"quality={quality_score:.2f}"
                )
            except ImportError:
                logger.debug("[QualityFeedbackHandler] ImprovementOptimizer not available")
            except (TypeError, AttributeError) as e:
                logger.debug(f"[QualityFeedbackHandler] ImprovementOptimizer error: {e}")

            # Also boost exploration for this config
            self._boost_exploration_for_stall(config_key, trend_duration_epochs=3)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[QualityFeedbackHandler] Error handling quality degraded: {e}")

    def _trigger_quality_check(self, config_key: str, reason: str) -> None:
        """Trigger a quality check for the given config.

        Emits QUALITY_CHECK_REQUESTED event to be handled by QualityMonitorDaemon,
        completing the feedback loop from training loss anomalies to data quality
        verification.
        """
        try:
            from app.coordination.event_router import emit_quality_check_requested

            logger.info(f"[QualityFeedbackHandler] Triggering quality check for {config_key}: {reason}")

            priority = "high" if reason in ("training_loss_anomaly", "training_loss_degrading") else "normal"

            try:
                _safe_create_task(
                    emit_quality_check_requested(
                        config_key=config_key,
                        reason=reason,
                        source="QualityFeedbackHandler",
                        priority=priority,
                    ),
                    context=f"emit_quality_check_requested:{config_key}",
                )
            except RuntimeError:
                asyncio.run(emit_quality_check_requested(
                    config_key=config_key,
                    reason=reason,
                    source="QualityFeedbackHandler",
                    priority=priority,
                ))

        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[QualityFeedbackHandler] Error triggering quality check: {e}")

    def _assess_selfplay_quality(self, db_path: str, games_count: int) -> float:
        """Assess quality of selfplay data using UnifiedQualityScorer.

        Uses the proper quality scoring system that evaluates game content,
        not just game count. Falls back to count-based heuristics if the
        unified scorer is unavailable.

        Returns:
            Quality score 0.0-1.0
        """
        try:
            from app.quality.unified_quality import compute_game_quality_from_params

            db = Path(db_path)
            if not db.exists():
                logger.debug(f"Database not found: {db_path}")
                return 0.3

            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT game_id, game_status, winner, termination_reason, total_moves
                    FROM games
                    WHERE game_status IN ('complete', 'finished', 'COMPLETE', 'FINISHED')
                    ORDER BY created_at DESC
                    LIMIT 50
                """)
                games = cursor.fetchall()

            if not games:
                logger.debug(f"No completed games in {db_path}")
                return 0.3

            quality_scores = []
            for game in games:
                try:
                    quality = compute_game_quality_from_params(
                        game_id=game["game_id"],
                        game_status=game["game_status"],
                        winner=game["winner"],
                        termination_reason=game["termination_reason"],
                        total_moves=game["total_moves"] or 0,
                    )
                    quality_scores.append(quality.quality_score)
                except (AttributeError, TypeError, KeyError, ValueError) as e:
                    logger.debug(
                        f"[QualityFeedbackHandler] Failed to compute quality for game "
                        f"{game.get('game_id', 'unknown')}: {e}"
                    )
                    continue

            if not quality_scores:
                return 0.3

            avg_quality = sum(quality_scores) / len(quality_scores)
            count_factor = min(1.0, games_count / 500)
            final_quality = 0.3 + (avg_quality - 0.3) * count_factor

            logger.debug(
                f"[QualityFeedbackHandler] Quality: avg={avg_quality:.3f}, "
                f"count_factor={count_factor:.2f}, final={final_quality:.3f}"
            )
            return final_quality

        except ImportError:
            logger.debug("UnifiedQualityScorer not available, using count heuristic")
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Quality assessment error: {e}")

        # Fallback to count-based heuristic
        if games_count < 100:
            return 0.3
        elif games_count < 500:
            return 0.6
        elif games_count < 1000:
            return 0.8
        else:
            return 0.95

    def _compute_intensity_from_quality(self, quality_score: float) -> str:
        """Map continuous quality score to intensity level.

        Returns:
            Intensity level: paused, reduced, normal, accelerated, hot_path
        """
        if quality_score >= 0.90:
            return "hot_path"
        elif quality_score >= 0.80:
            return "accelerated"
        elif quality_score >= 0.65:
            return "normal"
        elif quality_score >= 0.50:
            return "reduced"
        else:
            return "paused"

    def _update_training_intensity(self, config_key: str, quality_score: float) -> None:
        """Update training intensity based on data quality."""
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            accelerator = get_feedback_accelerator()
            intensity = self._compute_intensity_from_quality(quality_score)

            urgency_map = {
                "hot_path": "critical",
                "accelerated": "high",
                "normal": "normal",
                "reduced": "low",
                "paused": "none",
            }

            urgency = urgency_map.get(intensity, "normal")

            if urgency != "none":
                accelerator.signal_training_needed(
                    config_key=config_key,
                    urgency=urgency,
                    reason=f"quality_score_{quality_score:.2f}_intensity_{intensity}",
                )

            logger.info(
                f"[QualityFeedbackHandler] Updated training intensity for {config_key}: "
                f"quality={quality_score:.2f} → intensity={intensity}"
            )

        except ImportError:
            logger.debug("[QualityFeedbackHandler] FeedbackAccelerator not available")
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[QualityFeedbackHandler] Error updating training intensity: {e}")

    def _on_quality_check_failed(self, event: Any) -> None:
        """Handle QUALITY_CHECK_FAILED - data quality check failed.

        Triggers additional selfplay to improve data quality and adjusts
        training parameters to be more conservative.
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config = payload.get("config", "")
        quality_score = payload.get("quality_score", 0.0)
        threshold = payload.get("threshold", 0.6)

        logger.warning(
            f"[QualityFeedbackHandler] Quality check failed: {config} "
            f"(score={quality_score:.2f}, threshold={threshold:.2f})"
        )

        if config:
            state = self._get_or_create_state(config)
            state.last_selfplay_quality = quality_score

            self._quality_events_processed += 1
            self._last_quality_event_time = time.time()

            # Trigger more selfplay to improve quality
            if HAS_SELFPLAY_EVENTS and emit_selfplay_target_updated:
                try:
                    _safe_create_task(
                        emit_selfplay_target_updated(
                            config_key=config,
                            target_games=1000,
                            reason="quality_check_failed",
                            priority=8,
                            source="quality_feedback_handler",
                        ),
                        "selfplay_target_emit"
                    )
                except (ImportError, AttributeError) as e:
                    logger.debug(f"[QualityFeedbackHandler] Emitter not available: {e}")
                except (TypeError, ValueError) as e:
                    logger.error(f"[QualityFeedbackHandler] Invalid selfplay target parameters: {e}")
                except RuntimeError as e:
                    logger.warning(f"[QualityFeedbackHandler] Failed to emit selfplay target: {e}")

    def _on_quality_feedback_adjusted(self, event: Any) -> None:
        """Handle QUALITY_FEEDBACK_ADJUSTED - quality assessment triggered adjustments.

        Adjusts training intensity and exploration based on data quality feedback.
        When quality improves, accelerate training. When quality degrades, boost exploration.
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config_key = payload.get("config_key", "")
        quality_score = payload.get("quality_score", 0.5)
        budget_multiplier = payload.get("budget_multiplier", 1.0)
        adjustment_type = payload.get("adjustment_type", "unknown")

        if not config_key:
            return

        state = self._get_or_create_state(config_key)
        state.last_selfplay_quality = quality_score

        self._quality_events_processed += 1
        self._last_quality_event_time = time.time()

        if budget_multiplier > 1.0:
            state.current_training_intensity = "accelerated"
            logger.info(
                f"[QualityFeedbackHandler] Quality feedback positive for {config_key}: "
                f"score={quality_score:.2f}, multiplier={budget_multiplier:.2f} → accelerated training"
            )
        elif budget_multiplier < 0.8:
            state.current_training_intensity = "conservative"
            logger.info(
                f"[QualityFeedbackHandler] Quality feedback negative for {config_key}: "
                f"score={quality_score:.2f}, multiplier={budget_multiplier:.2f} → conservative training"
            )

            # Trigger exploration boost for poor quality
            if HAS_EXPLORATION_EVENTS and emit_exploration_boost:
                try:
                    _safe_create_task(
                        emit_exploration_boost(
                            config_key=config_key,
                            boost_factor=1.5,
                            reason=f"quality_feedback_{adjustment_type}",
                            source="quality_feedback_handler",
                        ),
                        "quality_exploration_boost_emit"
                    )
                except (ImportError, RuntimeError, asyncio.CancelledError) as e:
                    logger.debug(f"Failed to emit exploration boost: {e}")
        else:
            state.current_training_intensity = "normal"

    def _on_high_quality_data_available(self, event: Any) -> None:
        """Handle HIGH_QUALITY_DATA_AVAILABLE - quality recovered above threshold.

        When data quality recovers to "good" levels, this handler:
        1. Reduces exploration boost (no longer needed)
        2. Accelerates training intensity
        3. Updates quality tracking metrics
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config_key = payload.get("config_key", "")
        quality_score = payload.get("quality_score", 0.0)
        sample_count = payload.get("sample_count", 0)

        if not config_key:
            return

        state = self._get_or_create_state(config_key)

        prev_quality = state.last_selfplay_quality
        state.last_selfplay_quality = quality_score
        state.last_selfplay_time = time.time()

        state.current_training_intensity = "accelerated"

        self._quality_events_processed += 1
        self._last_quality_event_time = time.time()

        logger.info(
            f"[QualityFeedbackHandler] High quality data available for {config_key}: "
            f"score={quality_score:.2f} (prev={prev_quality:.2f}), "
            f"samples={sample_count} → accelerated training"
        )

    def _on_quality_score_updated(self, event: Any) -> None:
        """Handle QUALITY_SCORE_UPDATED - game quality recalculated.

        When quality scores are updated, this handler:
        1. Updates per-config quality tracking
        2. Adjusts training intensity based on quality trends
        3. Triggers exploration boost if quality is declining
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config_key = payload.get("config_key", "")
        quality_score = payload.get("quality_score", 0.0)
        trend = payload.get("trend", "stable")
        sample_count = payload.get("sample_count", 0)

        if not config_key:
            return

        state = self._get_or_create_state(config_key)

        prev_quality = state.last_selfplay_quality
        state.last_selfplay_quality = quality_score
        state.last_selfplay_time = time.time()

        self._quality_events_processed += 1
        self._last_quality_event_time = time.time()

        if trend == "declining" and quality_score < 0.6:
            state.current_training_intensity = "conservative"
            logger.warning(
                f"[QualityFeedbackHandler] Quality declining for {config_key}: "
                f"score={quality_score:.2f}, trend={trend} → conservative training"
            )
        elif trend == "improving" and quality_score > 0.8:
            state.current_training_intensity = "accelerated"
            logger.info(
                f"[QualityFeedbackHandler] Quality improving for {config_key}: "
                f"score={quality_score:.2f} → accelerated training"
            )
        else:
            logger.debug(
                f"[QualityFeedbackHandler] Quality update for {config_key}: "
                f"score={quality_score:.2f} (prev={prev_quality:.2f})"
            )

        # Emit exploration adjustment based on quality
        self._emit_exploration_adjustment(config_key, quality_score, trend)

    def _emit_exploration_adjustment(
        self, config_key: str, quality_score: float, trend: str
    ) -> None:
        """Emit exploration adjustment signal based on quality score.

        Quality-driven selfplay exploration signals:
        - Low quality → harder positions, more MCTS budget
        - High quality → standard exploration, normal budget
        - Declining trend → boost exploration temperature
        """
        try:
            from app.coordination.event_router import DataEventType, get_event_bus
            from app.distributed.data_events import DataEvent

            # Determine exploration adjustments based on quality
            if quality_score < 0.5:
                position_difficulty = "hard"
                mcts_budget_multiplier = 1.5
                exploration_temp_boost = 1.3
            elif quality_score < 0.7:
                position_difficulty = "medium-hard"
                mcts_budget_multiplier = 1.2
                exploration_temp_boost = 1.15
            elif quality_score > 0.9:
                position_difficulty = "normal"
                mcts_budget_multiplier = 0.8
                exploration_temp_boost = 1.0
            else:
                position_difficulty = "normal"
                mcts_budget_multiplier = 1.0
                exploration_temp_boost = 1.0

            # Boost exploration if trend is declining
            if trend == "declining":
                exploration_temp_boost *= 1.2
                mcts_budget_multiplier = max(mcts_budget_multiplier, 1.3)

            # Only emit if adjustments differ from baseline
            if (mcts_budget_multiplier != 1.0 or exploration_temp_boost != 1.0 or
                position_difficulty != "normal"):

                payload = {
                    "config_key": config_key,
                    "quality_score": quality_score,
                    "trend": trend,
                    "position_difficulty": position_difficulty,
                    "mcts_budget_multiplier": mcts_budget_multiplier,
                    "exploration_temp_boost": exploration_temp_boost,
                    "timestamp": time.time(),
                }

                bus = get_event_bus()
                event = DataEvent(
                    event_type=DataEventType.EXPLORATION_ADJUSTED,
                    payload=payload,
                    source="QualityFeedbackHandler",
                )
                bus.publish(event)

                logger.info(
                    f"[QualityFeedbackHandler] Exploration adjusted for {config_key}: "
                    f"difficulty={position_difficulty}, mcts_mult={mcts_budget_multiplier:.1f}, "
                    f"temp_boost={exploration_temp_boost:.2f} (quality={quality_score:.2f})"
                )

        except (ImportError, AttributeError, RuntimeError) as e:
            logger.debug(f"[QualityFeedbackHandler] Failed to emit exploration adjustment: {e}")

    def health_check(self) -> HealthCheckResult:
        """Return health check result for DaemonManager integration."""
        base_result = super().health_check()

        # Add quality-specific details
        base_result.details["quality_events_processed"] = self._quality_events_processed
        base_result.details["last_quality_event_time"] = self._last_quality_event_time
        base_result.details["states_tracked"] = len(self._states)

        return base_result


# Singleton accessor
_handler_instance: QualityFeedbackHandler | None = None


def get_quality_feedback_handler(
    states: dict[str, "FeedbackState"] | None = None,
    get_or_create_state_fn: callable | None = None,
    boost_exploration_fn: callable | None = None,
) -> QualityFeedbackHandler:
    """Get or create the singleton QualityFeedbackHandler instance."""
    return QualityFeedbackHandler.get_instance(
        states, get_or_create_state_fn, boost_exploration_fn
    )


def reset_quality_feedback_handler() -> None:
    """Reset the singleton for testing."""
    QualityFeedbackHandler.reset_instance()


__all__ = [
    "QualityFeedbackHandler",
    "get_quality_feedback_handler",
    "reset_quality_feedback_handler",
]
