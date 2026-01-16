"""Quality feedback handling for FeedbackLoopController.

Sprint 17.9 (Jan 16, 2026): Extracted from feedback_loop_controller.py (~290 LOC)

This mixin provides quality assessment and feedback logic that responds to:
- Selfplay data quality (game content analysis)
- Quality → training intensity mapping
- Quality → curriculum weight adjustments
- Quality degradation events

The quality feedback affects training scheduling and selfplay allocation
to ensure high-quality training data.

Usage:
    class FeedbackLoopController(QualityFeedbackMixin, ...):
        pass
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from app.coordination.event_router import safe_emit_event

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
            lambda t: logger.debug(f"[QualityFeedback] Task {context} done")
            if not t.cancelled() and t.exception() is None
            else logger.warning(f"[QualityFeedback] Task {context} failed: {t.exception()}")
            if t.exception() else None
        )
        return task
    except RuntimeError as e:
        logger.debug(f"[QualityFeedback] Could not create task for {context}: {e}")
        return None


class QualityFeedbackMixin:
    """Mixin for quality feedback handling in FeedbackLoopController.

    Requires the host class to implement:
    - _get_or_create_state(config_key: str) -> FeedbackState

    Provides:
    - _assess_selfplay_quality(db_path, games_count) -> quality score
    - _assess_selfplay_quality_async(db_path, games_count) -> quality score
    - _compute_intensity_from_quality(quality_score) -> intensity level
    - _update_training_intensity(config_key, quality_score)
    - _update_curriculum_weight_from_selfplay(config_key, quality_score)
    - _signal_training_ready(config_key, quality_score)
    - _emit_quality_degraded(config_key, quality_score, threshold, previous_score)
    """

    def _get_or_create_state(self, config_key: str) -> "FeedbackState":
        """Get or create state for a config. Must be implemented by host class."""
        raise NotImplementedError("Host class must implement _get_or_create_state")

    def _assess_selfplay_quality(self, db_path: str, games_count: int) -> float:
        """Assess quality of selfplay data using UnifiedQualityScorer.

        Uses the proper quality scoring system that evaluates game content,
        not just game count. Falls back to count-based heuristics if the
        unified scorer is unavailable.

        Returns:
            Quality score 0.0-1.0
        """
        try:
            from pathlib import Path
            import sqlite3
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
                    logger.debug(f"[QualityFeedback] Failed to compute quality for game {game.get('game_id', 'unknown')}: {e}")
                    continue

            if not quality_scores:
                return 0.3

            avg_quality = sum(quality_scores) / len(quality_scores)
            count_factor = min(1.0, games_count / 500)
            final_quality = 0.3 + (avg_quality - 0.3) * count_factor

            logger.debug(
                f"[QualityFeedback] Quality: avg={avg_quality:.3f}, "
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

    async def _assess_selfplay_quality_async(
        self, db_path: str, games_count: int
    ) -> float:
        """Async version of _assess_selfplay_quality.

        Sprint 17.9: Wraps blocking SQLite operation in asyncio.to_thread()
        to avoid blocking the event loop when called from async event handlers.

        Returns:
            Quality score 0.0-1.0
        """
        return await asyncio.to_thread(
            self._assess_selfplay_quality, db_path, games_count
        )

    def _compute_intensity_from_quality(self, quality_score: float) -> str:
        """Map continuous quality score to intensity level.

        December 2025 - Phase 2C.1: Continuous quality-to-intensity scaling.
        Replaces binary 0.6/0.8 thresholds with a 5-tier gradient.

        Returns:
            Intensity level: paused, reduced, normal, accelerated, hot_path
        """
        if quality_score >= 0.90:
            return "hot_path"  # Excellent quality -> maximum training speed
        elif quality_score >= 0.80:
            return "accelerated"  # Very good quality
        elif quality_score >= 0.65:
            return "normal"  # Adequate quality
        elif quality_score >= 0.50:
            return "reduced"  # Poor quality -> slower training
        else:
            return "paused"  # Very poor quality -> pause training

    def _update_training_intensity(self, config_key: str, quality_score: float) -> None:
        """Update training intensity based on data quality.

        December 2025 - Phase 2C.1: Now uses continuous quality gradient
        instead of binary thresholds.
        """
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            accelerator = get_feedback_accelerator()

            # Compute intensity using continuous gradient
            intensity = self._compute_intensity_from_quality(quality_score)

            # Map to urgency for accelerator signaling
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

            # Update state with computed intensity
            state = self._get_or_create_state(config_key)
            state.current_training_intensity = intensity

            logger.debug(
                f"[QualityFeedback] Quality {quality_score:.2f} -> intensity {intensity}"
            )

        except ImportError:
            pass
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.warning(f"Failed to update training intensity: {e}")

    def _update_curriculum_weight_from_selfplay(
        self, config_key: str, quality_score: float
    ) -> None:
        """Update curriculum weight based on selfplay quality.

        Gap 4 fix (December 2025): Creates feedback path from selfplay quality
        to curriculum weights. This ensures configs with quality issues get
        more training attention (higher weight) while stable configs can
        have slightly reduced priority.

        Logic:
        - Low quality (< 0.5): Increase weight by 15% (needs attention)
        - Medium quality (0.5-0.7): No change
        - High quality (>= 0.7): Decrease weight by 5% (stable, less urgent)
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

                # Propagate to curriculum feedback system
                if hasattr(feedback, "_current_weights"):
                    feedback._current_weights[config_key] = new_weight

                logger.info(
                    f"[QualityFeedback] Curriculum weight for {config_key}: "
                    f"{old_weight:.2f} -> {new_weight:.2f} (quality={quality_score:.2f})"
                )

                # Dec 2025: Emit CURRICULUM_REBALANCED event for SelfplayScheduler
                # This closes the feedback loop from selfplay quality -> scheduler priorities
                try:
                    # Jan 2026: Migrated to event_router (app.coordination.data_events deprecated Q2 2026)
                    from app.coordination.event_router import DataEventType, get_router

                    router = get_router()
                    # router.emit() is synchronous - no task wrapper needed (Dec 28, 2025 fix)
                    router.emit(
                        event_type=DataEventType.CURRICULUM_REBALANCED,
                        payload={
                            "config_key": config_key,
                            "weight": new_weight,
                            "reason": f"selfplay_quality_{quality_score:.2f}",
                        },
                    )
                except (ImportError, AttributeError):
                    pass  # Event system not available

        except ImportError:
            pass
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Failed to update curriculum weight: {e}")

    def _signal_training_ready(self, config_key: str, quality_score: float) -> None:
        """Signal that training data is ready (January 2026 - migrated to event_router)."""
        safe_emit_event(
            "DATA_QUALITY_ASSESSED",
            {
                "config": config_key,
                "quality_score": quality_score,
                "samples_available": 0,  # Unknown here
                "ready_for_training": True,
            },
            context="quality_feedback_mixin",
        )

    def _emit_quality_degraded(
        self,
        config_key: str,
        quality_score: float,
        threshold: float,
        previous_score: float,
    ) -> None:
        """Emit QUALITY_DEGRADED event when quality drops below threshold.

        Phase 5 (Dec 2025): Connects quality monitoring to selfplay scheduling.
        When quality degrades, SelfplayScheduler can reduce allocation for this
        config, forcing attention to fixing the underlying issue.

        Args:
            config_key: Configuration key
            quality_score: Current quality score
            threshold: Quality threshold that was crossed
            previous_score: Previous quality score for delta calculation
        """
        try:
            from app.coordination.event_router import emit_quality_degraded

            _safe_create_task(
                emit_quality_degraded(
                    config_key=config_key,
                    quality_score=quality_score,
                    threshold=threshold,
                    previous_score=previous_score,
                    source="quality_feedback_mixin",
                ),
                f"emit_quality_degraded({config_key})",
            )

            logger.warning(
                f"[QualityFeedback] Quality degraded for {config_key}: "
                f"{quality_score:.2f} < {threshold:.2f} (prev: {previous_score:.2f})"
            )

        except ImportError:
            logger.debug("[QualityFeedback] emit_quality_degraded not available")
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[QualityFeedback] Error emitting quality degraded: {e}")


__all__ = ["QualityFeedbackMixin"]
