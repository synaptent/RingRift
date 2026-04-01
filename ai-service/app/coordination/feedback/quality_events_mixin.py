"""Quality events mixin for FeedbackLoopController.

Extracted from feedback_loop_controller.py (Sprint 18, Feb 2026).
Handles quality-related event subscriptions: QUALITY_DEGRADED, QUALITY_CHECK_FAILED,
QUALITY_FEEDBACK_ADJUSTED, HIGH_QUALITY_DATA_AVAILABLE, QUALITY_SCORE_UPDATED.

~380 LOC extracted.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.config.thresholds import MEDIUM_QUALITY_THRESHOLD
from app.coordination.event_handler_utils import extract_config_key

logger = logging.getLogger(__name__)


class QualityEventsMixin:
    """Mixin providing quality event handling for FeedbackLoopController."""

    def _on_quality_degraded_for_training(self, event: Any) -> None:
        """Handle QUALITY_DEGRADED events to adjust training thresholds (P1.1).

        When quality degrades, we want to train MORE to fix the problem.
        This reduces the training threshold via ImprovementOptimizer.

        Actions:
        - Record low data quality score in ImprovementOptimizer
        - This triggers faster training cycles (lower threshold)
        - Also boost exploration to gather more diverse data
        """
        try:
            payload = event.payload if hasattr(event, "payload") else {}

            # Dec 2025: Source tracking loop guard - skip events we emitted
            source = payload.get("source", "")
            if source in ("feedback_loop_controller", "FeedbackLoopController"):
                logger.debug("[FeedbackLoopController] Skipping self-emitted QUALITY_DEGRADED event")
                return

            config_key = extract_config_key(payload)
            quality_score = payload.get("quality_score", 0.5)
            threshold = payload.get("threshold", MEDIUM_QUALITY_THRESHOLD)

            if not config_key:
                return

            logger.info(
                f"[FeedbackLoopController] Quality degraded for {config_key}: "
                f"score={quality_score:.2f} < threshold={threshold:.2f}, "
                f"triggering training acceleration"
            )

            # Update ImprovementOptimizer to reduce training threshold
            try:
                from app.training.improvement_optimizer import ImprovementOptimizer

                optimizer = ImprovementOptimizer.get_instance()
                # Record low data quality - this reduces threshold_multiplier
                optimizer.record_data_quality(
                    config_key=config_key,
                    data_quality_score=quality_score,
                    parity_success_rate=quality_score,  # Use quality as proxy
                )
                logger.info(
                    f"[FeedbackLoopController] Updated ImprovementOptimizer for {config_key}: "
                    f"quality={quality_score:.2f}"
                )
            except ImportError:
                logger.debug("[FeedbackLoopController] ImprovementOptimizer not available")

            # Also boost exploration for this config
            self._boost_exploration_for_stall(config_key, trend_duration_epochs=3)

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[FeedbackLoopController] Error handling quality degraded: {e}")

    def _trigger_quality_check(self, config_key: str, reason: str) -> None:
        """Trigger a quality check for the given config.

        Phase 9 (Dec 2025): Emits QUALITY_CHECK_REQUESTED event to be handled
        by QualityMonitorDaemon, completing the feedback loop from training
        loss anomalies to data quality verification.
        """
        try:
            from app.coordination.event_router import emit_quality_check_requested
            from app.coordination.feedback_loop_controller import _safe_create_task

            logger.info(
                f"[FeedbackLoopController] Triggering quality check for {config_key}: {reason}"
            )

            # Determine priority based on reason
            priority = "high" if reason in ("training_loss_anomaly", "training_loss_degrading") else "normal"

            # Emit the event (handle both sync and async contexts)
            try:
                task = _safe_create_task(
                    emit_quality_check_requested(
                        config_key=config_key,
                        reason=reason,
                        source="FeedbackLoopController",
                        priority=priority,
                    ),
                    context=f"emit_quality_check_requested:{config_key}",
                )
                if task is None:
                    asyncio.run(emit_quality_check_requested(
                        config_key=config_key,
                        reason=reason,
                        source="FeedbackLoopController",
                        priority=priority,
                    ))
            except RuntimeError:
                logger.warning(
                    f"[FeedbackLoopController] Failed to emit quality check request for {config_key}"
                )

        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[FeedbackLoopController] Error triggering quality check: {e}")

    def _on_quality_check_failed(self, event) -> None:
        """Handle QUALITY_CHECK_FAILED - data quality check failed.

        Triggers additional selfplay to improve data quality and adjusts
        training parameters to be more conservative.

        Added: December 2025
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config = extract_config_key(payload)
        quality_score = payload.get("quality_score", 0.0)
        threshold = payload.get("threshold", 0.6)

        logger.warning(
            f"[FeedbackLoopController] Quality check failed: {config} "
            f"(score={quality_score:.2f}, threshold={threshold:.2f})"
        )

        if config:
            state = self._get_or_create_state(config)
            state.last_selfplay_quality = quality_score

            # Trigger more selfplay to improve quality
            try:
                from app.coordination.event_router import emit_selfplay_target_updated

                from app.coordination.feedback_loop_controller import _safe_create_task

                task = _safe_create_task(
                    emit_selfplay_target_updated(
                        config_key=config,
                        target_games=1000,  # Request more games
                        reason="quality_check_failed",
                        priority=8,
                        source="feedback_loop_controller",
                    ),
                    "selfplay_target_emit"
                )
                if task is None:
                    asyncio.run(
                        emit_selfplay_target_updated(
                            config_key=config,
                            target_games=1000,
                            reason="quality_check_failed",
                            priority=8,
                            source="feedback_loop_controller",
                        )
                    )
            except (ImportError, AttributeError) as e:
                logger.debug(f"[FeedbackLoop] Emitter not available: {e}")
            except (TypeError, ValueError) as e:
                logger.error(f"[FeedbackLoop] Invalid selfplay target parameters: {e}")
            except RuntimeError as e:
                logger.warning(f"[FeedbackLoop] Failed to emit selfplay target: {e}")

    def _on_quality_feedback_adjusted(self, event) -> None:
        """Handle QUALITY_FEEDBACK_ADJUSTED - quality assessment triggered adjustments.

        Adjusts training intensity and exploration based on data quality feedback.
        When quality improves, accelerate training. When quality degrades, boost exploration.

        Added: December 2025 - Closes critical feedback loop gap
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config_key = extract_config_key(payload)
        quality_score = payload.get("quality_score", 0.5)
        budget_multiplier = payload.get("budget_multiplier", 1.0)
        adjustment_type = payload.get("adjustment_type", "unknown")

        if not config_key:
            return

        state = self._get_or_create_state(config_key)

        # Update quality tracking
        state.last_selfplay_quality = quality_score

        # Adjust training intensity based on quality feedback
        if budget_multiplier > 1.0:
            # Quality is good - accelerate training
            state.current_training_intensity = "accelerated"
            logger.info(
                f"[FeedbackLoopController] Quality feedback positive for {config_key}: "
                f"score={quality_score:.2f}, multiplier={budget_multiplier:.2f} -> accelerated training"
            )
        elif budget_multiplier < 0.8:
            # Quality is poor - boost exploration, slow training
            state.current_training_intensity = "conservative"
            logger.info(
                f"[FeedbackLoopController] Quality feedback negative for {config_key}: "
                f"score={quality_score:.2f}, multiplier={budget_multiplier:.2f} -> conservative training"
            )

            # Trigger exploration boost for poor quality
            try:
                from app.coordination.event_router import emit_exploration_boost

                from app.coordination.feedback_loop_controller import _safe_create_task

                task = _safe_create_task(
                    emit_exploration_boost(
                        config_key=config_key,
                        boost_factor=1.5,
                        reason=f"quality_feedback_{adjustment_type}",
                        source="feedback_loop_controller",
                    ),
                    "quality_exploration_boost_emit"
                )
                if task is None:
                    asyncio.run(
                        emit_exploration_boost(
                            config_key=config_key,
                            boost_factor=1.5,
                            reason=f"quality_feedback_{adjustment_type}",
                            source="feedback_loop_controller",
                        )
                    )
            except (ImportError, RuntimeError, asyncio.CancelledError) as e:
                logger.debug(f"Failed to emit exploration boost: {e}")
        else:
            # Quality is normal
            state.current_training_intensity = "normal"

    def _on_high_quality_data_available(self, event) -> None:
        """Handle HIGH_QUALITY_DATA_AVAILABLE - quality recovered above threshold.

        When data quality recovers to "good" levels, this handler:
        1. Reduces exploration boost (no longer needed)
        2. Accelerates training intensity
        3. Updates quality tracking metrics

        Added: December 2025 - Closes quality recovery feedback loop
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config_key = extract_config_key(payload)
        quality_score = payload.get("quality_score", 0.0)
        sample_count = payload.get("sample_count", 0)

        if not config_key:
            return

        state = self._get_or_create_state(config_key)

        # Update quality tracking
        prev_quality = state.last_selfplay_quality
        state.last_selfplay_quality = quality_score
        state.last_selfplay_time = time.time()

        # Quality recovered - accelerate training
        state.current_training_intensity = "accelerated"

        logger.info(
            f"[FeedbackLoopController] High quality data available for {config_key}: "
            f"score={quality_score:.2f} (prev={prev_quality:.2f}), "
            f"samples={sample_count} -> accelerated training"
        )

        # Track metrics
        if hasattr(self, "_metrics") and self._metrics:
            self._metrics.increment("high_quality_events", {"config_key": config_key})

    def _on_quality_score_updated(self, event) -> None:
        """Handle QUALITY_SCORE_UPDATED - game quality recalculated.

        Dec 27, 2025: Closes quality monitoring -> training feedback loop.
        When quality scores are updated, this handler:
        1. Updates per-config quality tracking
        2. Adjusts training intensity based on quality trends
        3. Triggers exploration boost if quality is declining
        """
        payload = event.payload if hasattr(event, "payload") else {}
        config_key = extract_config_key(payload)
        quality_score = payload.get("quality_score", 0.0)
        trend = payload.get("trend", "stable")  # improving, declining, stable
        sample_count = payload.get("sample_count", 0)

        if not config_key:
            return

        state = self._get_or_create_state(config_key)

        # Update quality tracking
        prev_quality = state.last_selfplay_quality
        state.last_selfplay_quality = quality_score
        state.last_selfplay_time = time.time()

        # Adjust training intensity based on trend
        if trend == "declining" and quality_score < 0.6:
            state.current_training_intensity = "conservative"
            logger.warning(
                f"[FeedbackLoopController] Quality declining for {config_key}: "
                f"score={quality_score:.2f}, trend={trend} -> conservative training"
            )
        elif trend == "improving" and quality_score > 0.8:
            state.current_training_intensity = "accelerated"
            logger.info(
                f"[FeedbackLoopController] Quality improving for {config_key}: "
                f"score={quality_score:.2f} -> accelerated training"
            )
        else:
            logger.debug(
                f"[FeedbackLoopController] Quality update for {config_key}: "
                f"score={quality_score:.2f} (prev={prev_quality:.2f})"
            )

        # Dec 29, 2025: Emit EXPLORATION_ADJUSTED for quality-driven selfplay adjustment
        self._emit_exploration_adjustment(config_key, quality_score, trend)

    def _emit_exploration_adjustment(
        self, config_key: str, quality_score: float, trend: str
    ) -> None:
        """Emit exploration adjustment signal based on quality score.

        Dec 29, 2025: Quality-driven selfplay exploration signals.
        Adjusts exploration parameters to match data quality needs.
        """
        try:
            from app.coordination.event_router import DataEventType, publish_sync

            # Determine exploration adjustments based on quality
            if quality_score < 0.5:
                # Very low quality -> aggressive exploration needed
                position_difficulty = "hard"
                mcts_budget_multiplier = 1.5
                exploration_temp_boost = 1.3
            elif quality_score < 0.7:
                # Medium quality -> slightly harder positions
                position_difficulty = "medium-hard"
                mcts_budget_multiplier = 1.2
                exploration_temp_boost = 1.15
            elif quality_score > 0.9:
                # High quality -> can reduce budget for efficiency
                position_difficulty = "normal"
                mcts_budget_multiplier = 0.8
                exploration_temp_boost = 1.0
            else:
                # Normal quality
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

                publish_sync(
                    DataEventType.EXPLORATION_ADJUSTED,
                    payload,
                    source="FeedbackLoopController",
                )

                logger.info(
                    f"[FeedbackLoopController] Exploration adjusted for {config_key}: "
                    f"difficulty={position_difficulty}, mcts_mult={mcts_budget_multiplier:.1f}, "
                    f"temp_boost={exploration_temp_boost:.2f} (quality={quality_score:.2f})"
                )

        except Exception as e:
            logger.debug(f"[FeedbackLoopController] Failed to emit exploration adjustment: {e}")
