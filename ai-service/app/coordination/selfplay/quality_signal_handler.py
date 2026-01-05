"""SelfplayQualitySignalHandler - Quality feedback event handlers for selfplay scheduling.

January 2026 Sprint 17.9: Extracted from selfplay_scheduler.py as part of
the Phase 3 decomposition effort. This mixin provides handlers for quality
feedback events that affect selfplay allocation decisions.

Responsibilities:
- Apply quality penalties when quality drops below threshold
- Handle regression detection with exploration boost
- Boost selfplay when training is blocked by quality
- Respond to quality feedback adjustments
- Handle opponent mastered events for curriculum adjustment
- Boost selfplay on training early stop
- Throttle selfplay on low quality warning

Usage:
    class SelfplayScheduler(SelfplayQualitySignalMixin, SelfplayHealthMonitorMixin, HandlerBase):
        def __init__(self):
            super().__init__()
            # Mixin requires _config_priorities dict to be set

Event Subscriptions (via _get_quality_event_subscriptions):
- QUALITY_DEGRADED
- REGRESSION_DETECTED
- TRAINING_BLOCKED_BY_QUALITY
- QUALITY_FEEDBACK_ADJUSTED
- OPPONENT_MASTERED
- TRAINING_EARLY_STOPPED
- LOW_QUALITY_DATA_WARNING
"""

from __future__ import annotations

__all__ = [
    "SelfplayQualitySignalMixin",
]

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from app.coordination.selfplay.types import ConfigPriority

logger = logging.getLogger(__name__)

# Import at module level for type hints
STALE_DATA_THRESHOLD = 2.0  # Default, real value from coordination_defaults


class SelfplayQualitySignalMixin:
    """Mixin providing quality feedback event handlers for selfplay scheduling.

    This mixin handles quality-related events that affect selfplay allocation
    decisions, including quality degradation, regression detection, and
    training blockers.

    Required attributes from parent class:
    - _config_priorities: dict[str, ConfigPriority]
    - invalidate_quality_cache(config_key: str | None) -> int
    - subscribe_to_events(): Callable

    Event types handled:
    - QUALITY_DEGRADED: Apply quality penalty proportional to degradation
    - REGRESSION_DETECTED: Apply exploration boost based on severity
    - TRAINING_BLOCKED_BY_QUALITY: Boost selfplay allocation by 1.5x
    - QUALITY_FEEDBACK_ADJUSTED: Invalidate cache, adjust priority immediately
    - OPPONENT_MASTERED: Reduce curriculum weight when opponent mastered
    - TRAINING_EARLY_STOPPED: Aggressively boost selfplay for fresh data
    - LOW_QUALITY_DATA_WARNING: Throttle selfplay to avoid more low-quality data
    """

    # Type hints for attributes from parent class
    _config_priorities: dict[str, "ConfigPriority"]

    def _get_quality_event_subscriptions(self) -> dict[str, Callable[[Any], None]]:
        """Get event subscriptions for quality signal handlers.

        Returns:
            dict mapping event names to handler methods
        """
        return {
            "QUALITY_DEGRADED": self._on_quality_degraded,
            "REGRESSION_DETECTED": self._on_regression_detected,
            "TRAINING_BLOCKED_BY_QUALITY": self._on_training_blocked_by_quality,
            "QUALITY_FEEDBACK_ADJUSTED": self._on_quality_feedback_adjusted,
            "OPPONENT_MASTERED": self._on_opponent_mastered,
            "TRAINING_EARLY_STOPPED": self._on_training_early_stopped,
            "LOW_QUALITY_DATA_WARNING": self._on_low_quality_warning,
        }

    def _on_quality_degraded(self, event: Any) -> None:
        """Handle quality degradation event.

        Phase 5 (Dec 2025): When game quality drops below threshold,
        apply a penalty to reduce selfplay allocation for this config.
        """
        try:
            from app.coordination.event_utils import extract_config_key

            config_key = extract_config_key(event.payload if hasattr(event, "payload") else event)
            payload = event.payload if hasattr(event, "payload") else event
            quality_score = payload.get("quality_score", 0.0)
            threshold = payload.get("threshold", 0.6)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                # Apply quality penalty proportional to how far below threshold
                # quality_score=0.4, threshold=0.6 → penalty = -0.20 * (0.6-0.4)/0.6 = -0.067
                if quality_score < threshold:
                    degradation = (threshold - quality_score) / threshold
                    priority.quality_penalty = -0.20 * degradation
                    logger.warning(
                        f"[SelfplayScheduler] Quality degradation for {config_key}: "
                        f"score={quality_score:.2f} < {threshold:.2f}, "
                        f"penalty={priority.quality_penalty:.3f}"
                    )
                else:
                    # Quality recovered, clear penalty
                    if priority.quality_penalty < 0:
                        logger.info(
                            f"[SelfplayScheduler] Quality recovered for {config_key}: "
                            f"score={quality_score:.2f}, clearing penalty"
                        )
                    priority.quality_penalty = 0.0

            # Dec 30, 2025: Invalidate quality cache when quality data changes
            self.invalidate_quality_cache(config_key)
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling quality degraded: {e}")

    def _on_regression_detected(self, event: Any) -> None:
        """Handle model regression detection.

        Dec 2025: When evaluation detects model regression (win rate dropping),
        trigger curriculum rebalancing to adjust selfplay priorities.
        This helps recover from training instability by increasing exploration.
        """
        try:
            from app.coordination.event_utils import extract_config_key

            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            severity = payload.get("severity", "moderate")  # mild, moderate, severe
            win_rate_drop = payload.get("win_rate_drop", 0.0)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                # Apply exploration boost based on regression severity
                boost_factor = {"mild": 0.1, "moderate": 0.2, "severe": 0.3}.get(severity, 0.15)
                priority.exploration_boost = boost_factor

                logger.warning(
                    f"[SelfplayScheduler] Regression detected for {config_key}: "
                    f"severity={severity}, win_rate_drop={win_rate_drop:.2%}, "
                    f"exploration_boost={boost_factor}"
                )

                # Emit curriculum rebalance event to trigger downstream updates
                try:
                    from app.coordination.event_router import publish_sync

                    publish_sync("CURRICULUM_REBALANCED", {
                        "config_key": config_key,
                        "reason": "regression_detected",
                        "severity": severity,
                        "new_exploration_boost": boost_factor,
                        "source": "selfplay_scheduler",  # Loop guard identifier
                    })
                except ImportError:
                    pass  # Event system not available

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling regression: {e}")

    def _on_training_blocked_by_quality(self, event: Any) -> None:
        """Handle training blocked by quality event.

        P1.1 (Dec 2025): Strategic Improvement Plan Gap 1.1 fix.
        When training is blocked due to stale/low-quality data, boost selfplay
        allocation by 1.5x to accelerate data generation for that config.

        This closes the critical feedback loop:
        TRAINING_BLOCKED_BY_QUALITY → Selfplay acceleration → Fresh data → Training resumes
        """
        try:
            from app.coordination.event_utils import extract_config_key

            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            reason = payload.get("reason", "unknown")
            data_age_hours = payload.get("data_age_hours", 0.0)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                # Boost selfplay allocation by 1.5x for this config
                old_boost = priority.exploration_boost
                priority.exploration_boost = max(1.5, old_boost * 1.5)
                priority.training_pending = True

                # Log the quality block
                logger.warning(
                    f"[SelfplayScheduler] Training blocked for {config_key} "
                    f"(reason: {reason}, age: {data_age_hours:.1f}h). "
                    f"Boosted selfplay: exploration {old_boost:.2f} → {priority.exploration_boost:.2f}"
                )

                # Emit a target update event to propagate the boost
                try:
                    from app.coordination.event_router import DataEventType, get_event_bus

                    bus = get_event_bus()
                    if bus:
                        bus.emit(DataEventType.SELFPLAY_TARGET_UPDATED, {
                            "config_key": config_key,
                            "priority": "high",
                            "reason": f"training_blocked:{reason}",
                            "exploration_boost": priority.exploration_boost,
                        })
                except Exception as emit_err:
                    logger.debug(f"[SelfplayScheduler] Failed to emit target update: {emit_err}")
            else:
                logger.debug(
                    f"[SelfplayScheduler] Received training blocked for unknown config: {config_key}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling training blocked by quality: {e}")

    def _on_quality_feedback_adjusted(self, event: Any) -> None:
        """Handle quality feedback adjustment - update priority immediately.

        Dec 30, 2025: When quality changes, invalidate cache and adjust
        priority weights immediately instead of waiting for cache TTL (30s).
        This enables faster response to quality degradation (+12-18 Elo).

        Quality thresholds:
        - < 0.50: Low quality - boost exploration to generate diverse games
        - > 0.80: High quality - can reduce exploration
        """
        try:
            from app.coordination.event_utils import extract_config_key

            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            quality_score = payload.get("quality_score", 0.7)

            if not config_key:
                logger.debug("[SelfplayScheduler] QUALITY_FEEDBACK_ADJUSTED missing config_key")
                return

            # Invalidate quality cache immediately
            self.invalidate_quality_cache(config_key)

            # Apply priority boost for low-quality configs
            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                if quality_score < 0.50:
                    # Low quality - boost exploration to generate more diverse games
                    old_boost = priority.exploration_boost
                    priority.exploration_boost = max(priority.exploration_boost, 0.2)
                    logger.info(
                        f"[SelfplayScheduler] Quality feedback: {config_key} "
                        f"score={quality_score:.2f} (low), boosting exploration "
                        f"{old_boost:.2f} → {priority.exploration_boost:.2f}"
                    )
                elif quality_score > 0.80:
                    # High quality - can reduce exploration to focus on exploitation
                    old_boost = priority.exploration_boost
                    priority.exploration_boost = max(0.0, priority.exploration_boost - 0.1)
                    logger.debug(
                        f"[SelfplayScheduler] Quality feedback: {config_key} "
                        f"score={quality_score:.2f} (high), reducing exploration "
                        f"{old_boost:.2f} → {priority.exploration_boost:.2f}"
                    )
                else:
                    logger.debug(
                        f"[SelfplayScheduler] Quality feedback: {config_key} "
                        f"score={quality_score:.2f} (normal), cache invalidated"
                    )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling quality feedback: {e}")

    def _on_opponent_mastered(self, event: Any) -> None:
        """Handle opponent mastered event.

        P1.4 (Dec 2025): When a model has mastered its current opponent level,
        advance the curriculum and slightly reduce selfplay allocation for this
        config (it's now "easier" and should allocate resources elsewhere).

        This closes the feedback loop:
        OPPONENT_MASTERED → Curriculum advancement → Harder opponents → Better training
        """
        try:
            from app.coordination.event_utils import extract_config_key

            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            opponent_level = payload.get("opponent_level", "unknown")
            win_rate = payload.get("win_rate", 0.0)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                # Reduce curriculum weight slightly since opponent was mastered
                old_weight = priority.curriculum_weight
                priority.curriculum_weight = max(0.5, old_weight * 0.9)

                # Also slightly reduce exploration (we don't need as much variance)
                old_boost = priority.exploration_boost
                priority.exploration_boost = max(0.8, old_boost * 0.95)

                logger.info(
                    f"[SelfplayScheduler] Opponent mastered for {config_key} "
                    f"(level={opponent_level}, win_rate={win_rate:.1%}). "
                    f"curriculum_weight {old_weight:.2f} → {priority.curriculum_weight:.2f}, "
                    f"exploration {old_boost:.2f} → {priority.exploration_boost:.2f}"
                )

                # Emit curriculum rebalanced event (Jan 2026: migrated to safe_emit_event)
                from app.coordination.event_emission_helpers import safe_emit_event

                safe_emit_event(
                    "CURRICULUM_UPDATED",
                    {
                        "config_key": config_key,
                        "new_weight": priority.curriculum_weight,
                        "trigger": f"opponent_mastered:{opponent_level}",
                    },
                    context="quality_signal_handler.opponent_mastered",
                )
            else:
                logger.debug(
                    f"[SelfplayScheduler] Received opponent mastered for unknown config: {config_key}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling opponent mastered: {e}")

    def _on_training_early_stopped(self, event: Any) -> None:
        """Handle training early stopped event.

        P10-LOOP-1 (Dec 2025): When training early stops due to stagnation or regression,
        aggressively boost selfplay to generate fresh, diverse training data.

        This closes the feedback loop:
        TRAINING_EARLY_STOPPED → Selfplay boost → More diverse data → Better next training run

        Early stopping typically indicates:
        - Loss plateau (need more diverse positions)
        - Overfitting (need fresher data)
        - Regression (need different exploration)
        """
        try:
            from app.config.coordination_defaults import SelfplayDefaults
            from app.coordination.event_utils import extract_config_key

            payload = event.payload if hasattr(event, "payload") else event
            config_key = extract_config_key(payload)
            reason = payload.get("reason", "unknown")
            final_loss = payload.get("final_loss", 0.0)
            epochs_completed = payload.get("epochs_completed", 0)

            # Get stale data threshold from config
            stale_threshold = getattr(SelfplayDefaults, "STALE_DATA_THRESHOLD", STALE_DATA_THRESHOLD)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]

                # Boost exploration significantly - we need diverse data
                old_boost = priority.exploration_boost
                if "regression" in reason.lower():
                    # Regression needs more aggressive exploration
                    priority.exploration_boost = min(2.0, old_boost * 1.5)
                elif "plateau" in reason.lower() or "stagnation" in reason.lower():
                    # Plateau needs moderate exploration boost
                    priority.exploration_boost = min(1.8, old_boost * 1.3)
                else:
                    # General early stop - moderate boost
                    priority.exploration_boost = min(1.5, old_boost * 1.2)

                # Also boost curriculum weight to prioritize this config
                old_weight = priority.curriculum_weight
                priority.curriculum_weight = min(2.0, old_weight * 1.3)

                # Mark as needing urgent data (increases staleness factor effect)
                priority.staleness_hours = max(priority.staleness_hours, stale_threshold)

                logger.info(
                    f"[SelfplayScheduler] Training early stopped for {config_key} "
                    f"(reason={reason}, epochs={epochs_completed}, loss={final_loss:.4f}). "
                    f"Boosted exploration {old_boost:.2f} → {priority.exploration_boost:.2f}, "
                    f"curriculum_weight {old_weight:.2f} → {priority.curriculum_weight:.2f}"
                )

                # Emit SELFPLAY_TARGET_UPDATED to trigger immediate selfplay allocation
                try:
                    from app.coordination.event_router import DataEventType, get_event_bus

                    bus = get_event_bus()
                    if bus:
                        bus.emit(DataEventType.SELFPLAY_TARGET_UPDATED, {
                            "config_key": config_key,
                            "priority": "urgent",
                            "reason": f"training_early_stopped:{reason}",
                            "exploration_boost": priority.exploration_boost,
                            "curriculum_weight": priority.curriculum_weight,
                        })
                        logger.debug(
                            f"[SelfplayScheduler] Emitted urgent SELFPLAY_TARGET_UPDATED for {config_key}"
                        )
                except Exception as emit_err:
                    logger.debug(f"[SelfplayScheduler] Failed to emit target update: {emit_err}")
            else:
                logger.debug(
                    f"[SelfplayScheduler] Received training early stopped for unknown config: {config_key}"
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling training early stopped: {e}")

    def _on_low_quality_warning(self, event: Any) -> None:
        """Handle LOW_QUALITY_DATA_WARNING to throttle selfplay allocation.

        Wire orphaned event (Dec 2025): When QualityMonitorDaemon detects quality
        below warning threshold, reduce selfplay allocation to avoid generating
        more low-quality data.

        This closes the feedback loop:
        Low quality detected → Throttle selfplay → Focus on training existing data

        Actions:
        - Reduce exploration_boost by 0.7x (throttle by 30%)
        - Apply temporary quality penalty
        - Log throttling action
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            quality_score = payload.get("quality_score", 0.0)
            old_state = payload.get("old_state", "unknown")
            new_state = payload.get("new_state", "unknown")

            logger.warning(
                f"[SelfplayScheduler] Low quality warning: "
                f"score={quality_score:.2f}, state={old_state} → {new_state}"
            )

            # Throttle all configs proportionally based on quality
            # Worse quality = more aggressive throttling
            if quality_score < 0.4:
                throttle_factor = 0.5  # 50% reduction for very poor quality
            elif quality_score < 0.5:
                throttle_factor = 0.6  # 40% reduction for poor quality
            else:
                throttle_factor = 0.7  # 30% reduction for marginal quality

            throttled_count = 0
            for config_key, priority in self._config_priorities.items():
                old_boost = priority.exploration_boost

                # Apply throttling to exploration boost
                priority.exploration_boost = max(0.5, priority.exploration_boost * throttle_factor)

                # Apply quality penalty
                old_penalty = priority.quality_penalty
                priority.quality_penalty = -0.15 * (1.0 - quality_score)  # -0.15 at quality=0, 0 at quality=1

                if abs(priority.exploration_boost - old_boost) > 0.01:
                    throttled_count += 1
                    logger.debug(
                        f"[SelfplayScheduler] Throttled {config_key}: "
                        f"exploration {old_boost:.2f} → {priority.exploration_boost:.2f}, "
                        f"quality_penalty {old_penalty:.3f} → {priority.quality_penalty:.3f}"
                    )

            logger.info(
                f"[SelfplayScheduler] Throttled selfplay for {throttled_count} configs "
                f"due to low quality (score={quality_score:.2f}, throttle={throttle_factor:.2f}x)"
            )

            # Emit QUALITY_PENALTY_APPLIED event for curriculum feedback (Dec 2025)
            if throttled_count > 0:
                self._emit_quality_penalty_applied(
                    quality_score=quality_score,
                    throttle_factor=throttle_factor,
                    throttled_count=throttled_count,
                )

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling low quality warning: {e}")

    def _emit_quality_penalty_applied(
        self,
        quality_score: float,
        throttle_factor: float,
        throttled_count: int,
    ) -> None:
        """Emit QUALITY_PENALTY_APPLIED event for each throttled config.

        Closes the feedback loop: LOW_QUALITY_DATA_WARNING → penalty applied → curriculum adjustment.

        January 2026: Migrated to safe_emit_event for consistent event handling.
        """
        from app.coordination.event_emission_helpers import safe_emit_event

        emitted_count = 0
        for config_key, priority in self._config_priorities.items():
            if priority.quality_penalty < 0:  # Only emit for penalized configs
                if safe_emit_event(
                    "QUALITY_PENALTY_APPLIED",
                    {
                        "config_key": config_key,
                        "penalty": -priority.quality_penalty,
                        "reason": "low_quality_selfplay_data",
                        "current_weight": priority.exploration_boost,
                        "source": "selfplay_scheduler",
                        "quality_score": quality_score,
                        "throttle_factor": throttle_factor,
                    },
                    context="quality_signal_handler.emit_quality_penalties",
                ):
                    emitted_count += 1

        logger.debug(
            f"[SelfplayScheduler] Emitted QUALITY_PENALTY_APPLIED for {emitted_count} configs"
        )

    # Required method from parent class (type hint only for mixin)
    def invalidate_quality_cache(self, config: str | None = None) -> int:
        """Invalidate quality cache for a config or all configs.

        This method must be implemented by the parent class.

        Args:
            config: Specific config to invalidate, or None to clear all

        Returns:
            Number of entries invalidated
        """
        raise NotImplementedError("Parent class must implement invalidate_quality_cache")
