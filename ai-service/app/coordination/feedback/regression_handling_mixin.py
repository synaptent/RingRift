"""Regression handling mixin for FeedbackLoopController.

Sprint 17.9 (Jan 16, 2026): Extracted from feedback_loop_controller.py (~230 LOC)

This mixin provides regression-related feedback logic that:
- Calculates response amplitude based on regression severity
- Handles REGRESSION_DETECTED events
- Triggers exploration boosts and curriculum rollbacks
- Emits SELFPLAY_TARGET_UPDATED for recovery

The regression feedback loop closes the training recovery cycle:
REGRESSION_DETECTED -> _on_regression_detected -> exploration boost -> selfplay -> recovery

Usage:
    class FeedbackLoopController(RegressionHandlingMixin, ...):
        pass
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from app.coordination.event_handler_utils import extract_config_key

if TYPE_CHECKING:
    from app.coordination.feedback_loop_controller import FeedbackState

logger = logging.getLogger(__name__)

# Import exploration boost constants
try:
    from app.coordination.feedback.exploration_boost import (
        EXPLORATION_BOOST_RECOVERY,
        EXPLORATION_BOOST_MAX,
    )
except ImportError:
    EXPLORATION_BOOST_RECOVERY = 1.5
    EXPLORATION_BOOST_MAX = 2.0

# Optional selfplay event emitter
HAS_SELFPLAY_EVENTS = False
emit_selfplay_target_updated = None
try:
    from app.coordination.selfplay_events import emit_selfplay_target_updated

    HAS_SELFPLAY_EVENTS = True
except ImportError:
    pass


def _safe_create_task(coro, context: str = "") -> asyncio.Task | None:
    """Create a task with basic error handling.

    Note: This is a local helper. The main controller has a more sophisticated
    version with error tracking. This is used for mixin independence.
    """
    try:
        task = asyncio.create_task(coro)
        task.add_done_callback(
            lambda t: logger.debug(f"[RegressionHandling] Task {context} done")
            if not t.cancelled() and t.exception() is None
            else logger.warning(f"[RegressionHandling] Task {context} failed: {t.exception()}")
            if t.exception() else None
        )
        return task
    except RuntimeError as e:
        if hasattr(coro, "close"):
            coro.close()
        logger.debug(f"[RegressionHandling] Could not create task for {context}: {e}")
        return None


class RegressionHandlingMixin:
    """Mixin for regression handling in FeedbackLoopController.

    Requires the host class to implement (typically via HandlerBase or other mixins):
    - _get_or_create_state(config_key: str) -> FeedbackState
    - _is_duplicate_event(payload: dict) -> bool (provided by HandlerBase)
    - _trigger_gauntlet_all_baselines(config_key: str) (from EvaluationFeedbackMixin)

    Provides:
    - _calculate_regression_amplitude(elo_drop, consecutive_count, severity_str) -> (boost, games)
    - _on_regression_detected(event) - Handle REGRESSION_DETECTED events

    Note: Stub methods are NOT provided here to avoid MRO conflicts with HandlerBase.
    The host class must ensure these methods are available through inheritance.
    """

    def _calculate_regression_amplitude(
        self, elo_drop: float, consecutive_count: int, severity_str: str = "minor"
    ) -> tuple[float, int]:
        """Calculate response amplitude based on regression severity.

        January 3, 2026: Implements proportional response to regression severity.
        Larger Elo drops and repeated regressions trigger stronger responses.

        January 2026: Now uses RegressionDetector's severity directly instead of
        recalculating from elo_drop. This consolidates severity logic in one place.

        The amplitude formula:
        - Base boost: 1.5x (EXPLORATION_BOOST_RECOVERY)
        - Severity scaling: MINOR +0.0, MODERATE +0.2, SEVERE +0.35, CRITICAL +0.5
        - Consecutive scaling: +0.15x per consecutive regression

        Args:
            elo_drop: Magnitude of Elo regression (positive value)
            consecutive_count: Number of consecutive regression events (from RegressionDetector)
            severity_str: Severity level from RegressionDetector ("minor", "moderate", "severe", "critical")

        Returns:
            Tuple of (exploration_boost, target_games):
            - exploration_boost: Scaled boost factor (1.5x to 2.5x range)
            - target_games: Number of selfplay games to request (500 to 2000 range)
        """
        # Base values
        base_boost = EXPLORATION_BOOST_RECOVERY  # 1.5x
        base_games = 500

        # January 2026: Use severity from RegressionDetector instead of recalculating
        severity_boost_map = {
            "minor": 0.0,
            "moderate": 0.2,
            "severe": 0.35,
            "critical": 0.5,
        }
        severity_boost = severity_boost_map.get(severity_str, 0.0)

        # Scale by consecutive count (+0.15x each, capped at +0.5x)
        consecutive_boost = min(0.5, consecutive_count * 0.15)

        # Combined boost, capped at EXPLORATION_BOOST_MAX (2.0)
        exploration_boost = min(
            EXPLORATION_BOOST_MAX,
            base_boost + severity_boost + consecutive_boost
        )

        # Scale target games based on severity
        # Use severity-based multiplier instead of raw elo_drop calculation
        severity_game_multiplier = {
            "minor": 1.0,
            "moderate": 1.5,
            "severe": 2.0,
            "critical": 2.5,
        }
        severity_factor = severity_game_multiplier.get(severity_str, 1.0) + consecutive_count * 0.3
        target_games = min(2000, int(base_games * severity_factor))

        return exploration_boost, target_games

    # Mar 20, 2026: Per-config cooldown to prevent regression response cascade.
    # Without this, REGRESSION_DETECTED fires hundreds of times per minute when
    # 8/12 configs are regressed, creating an event amplification cascade that
    # burns 100% CPU. 30-minute cooldown ensures at most one response per config
    # per evaluation cycle.
    _regression_cooldowns: dict[str, float] = {}
    _REGRESSION_COOLDOWN_SECONDS: float = 1800.0  # 30 minutes

    def _on_regression_detected(self, event: Any) -> None:
        """Handle REGRESSION_DETECTED event.

        Dec 2025: When regression is detected, trigger:
        1. Exploration boost (1.5x to generate more diverse data)
        2. SELFPLAY_TARGET_UPDATED event to request additional games
        3. Log the action for monitoring

        January 3, 2026: Added amplitude scaling - response intensity is now
        proportional to regression severity (Elo drop magnitude and consecutive count).

        January 3, 2026: Added deduplication to prevent duplicate handling when
        both FeedbackLoopController and UnifiedFeedbackOrchestrator are running.

        Mar 20, 2026: Added 30-minute per-config cooldown. Without this, with
        8/12 configs regressed, the handler fires hundreds of times per minute
        (each EVALUATION_PROGRESS triggers record_metric → regression detection
        → 13+ subscribers × secondary events = event cascade → 100% CPU).

        This closes the feedback loop: REGRESSION_DETECTED -> exploration boost ->
        more diverse selfplay -> better training data -> recovery.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event if isinstance(event, dict) else {}

            # January 3, 2026: Check for duplicate event (via HandlerBase)
            if self._is_duplicate_event(payload):
                logger.debug("[RegressionHandling] Skipping duplicate REGRESSION_DETECTED")
                return

            # Mar 20, 2026: Per-config cooldown (30 min) to prevent CPU burn cascade
            config_key = payload.get("config_key", payload.get("config", ""))
            if config_key:
                import time as _time
                last_handled = self._regression_cooldowns.get(config_key, 0.0)
                if (_time.time() - last_handled) < self._REGRESSION_COOLDOWN_SECONDS:
                    logger.debug(
                        f"[RegressionHandling] Cooldown active for {config_key}, "
                        f"skipping ({self._REGRESSION_COOLDOWN_SECONDS}s)"
                    )
                    return
                self._regression_cooldowns[config_key] = _time.time()

            config_key = extract_config_key(payload)
            elo_drop = payload.get("elo_drop", 0.0)
            # January 2026: Use consecutive_count from RegressionDetector (single source of truth)
            consecutive_count = payload.get("consecutive_count", payload.get("consecutive_regressions", 1))
            severity_str = payload.get("severity", "minor")

            if not config_key:
                logger.debug("[RegressionHandling] REGRESSION_DETECTED missing config_key")
                return

            state = self._get_or_create_state(config_key)
            # Sync with RegressionDetector's consecutive count instead of maintaining separate tracking
            state.consecutive_failures = consecutive_count

            # Jan 3, 2026: Calculate amplitude-scaled response based on severity
            # January 2026: Pass severity for severity-aware amplitude calculation
            exploration_boost, target_games = self._calculate_regression_amplitude(
                elo_drop, consecutive_count, severity_str
            )

            logger.warning(
                f"[RegressionHandling] Regression detected for {config_key}: "
                f"elo_drop={elo_drop:.0f}, consecutive={consecutive_count}, severity={severity_str}, "
                f"total_failures={state.consecutive_failures}, "
                f"amplitude_boost={exploration_boost:.2f}x, target_games={target_games}"
            )

            # Set exploration boost to generate more diverse data
            # Use max to preserve any higher existing boost
            old_boost = state.current_exploration_boost
            new_boost = max(state.current_exploration_boost, exploration_boost)
            state.current_exploration_boost = new_boost

            logger.info(
                f"[RegressionHandling] Increased exploration boost for {config_key}: "
                f"{old_boost:.2f}x -> {new_boost:.2f}x (regression amplitude response)"
            )

            # Emit SELFPLAY_TARGET_UPDATED to request more diverse selfplay games
            if HAS_SELFPLAY_EVENTS and emit_selfplay_target_updated:
                try:
                    # Dec 2025: Use get_running_loop() instead of deprecated get_event_loop()
                    loop = asyncio.get_running_loop()
                    task = _safe_create_task(
                        emit_selfplay_target_updated(
                            config_key=config_key,
                            target_games=target_games,
                            reason=f"regression_detected_elo_drop_{elo_drop:.0f}",
                            priority=2,  # High priority
                            source="regression_handling_mixin.py",
                        ),
                        f"emit_selfplay_target_updated_regression({config_key})",
                    )
                    if task:
                        logger.info(
                            f"[RegressionHandling] Emitted SELFPLAY_TARGET_UPDATED for {config_key}: "
                            f"{target_games} games (exploration_boost={exploration_boost:.1f}x, priority=2)"
                        )
                except RuntimeError:
                    logger.debug("[RegressionHandling] No event loop for SELFPLAY_TARGET_UPDATED")

            # Emit EXPLORATION_BOOST event for temperature schedulers
            try:
                from app.coordination.event_router import emit_exploration_boost

                _safe_create_task(emit_exploration_boost(
                    config_key=config_key,
                    boost_factor=new_boost,
                    reason="regression_detected",
                    anomaly_count=consecutive_count,
                    source="RegressionHandlingMixin",
                ), f"emit_exploration_boost_regression({config_key})")
                logger.debug(
                    f"[RegressionHandling] Emitted EXPLORATION_BOOST event for {config_key}"
                )
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.debug(f"[RegressionHandling] Failed to emit EXPLORATION_BOOST: {e}")

            # January 3, 2026 (Sprint 12 P1): Curriculum tier rollback for major regressions
            # When Elo drops significantly, roll back to a previous curriculum tier to allow
            # the model to relearn from simpler positions before advancing again.
            if elo_drop > 50 and state.curriculum_tier > 0:
                old_tier = state.curriculum_tier
                new_tier = max(0, old_tier - 1)  # Roll back one tier
                state.curriculum_tier = new_tier

                tier_names = ["Beginner", "Intermediate", "Advanced", "Expert"]
                logger.warning(
                    f"[RegressionHandling] Curriculum rollback for {config_key}: "
                    f"{tier_names[old_tier]} -> {tier_names[new_tier]} "
                    f"(elo_drop={elo_drop:.0f} > 50 threshold)"
                )

                # Emit CURRICULUM_ROLLBACK event for downstream consumers
                try:
                    from app.coordination.event_router import DataEventType, publish_sync

                    publish_sync(
                        DataEventType.CURRICULUM_ROLLBACK
                        if hasattr(DataEventType, "CURRICULUM_ROLLBACK")
                        else DataEventType.CURRICULUM_ADVANCED,
                        {
                            "config_key": config_key,
                            "old_tier": old_tier,
                            "new_tier": new_tier,
                            "trigger": "regression_detected",
                            "elo_drop": elo_drop,
                            "consecutive_regressions": consecutive_count,  # Fixed: was undefined variable
                            "direction": "rollback",
                            "source": "RegressionHandlingMixin",
                        },
                        source="RegressionHandlingMixin",
                    )
                    logger.info(
                        f"[RegressionHandling] Emitted CURRICULUM_ROLLBACK for {config_key}"
                    )
                except (AttributeError, TypeError, ImportError, RuntimeError) as emit_err:
                    logger.debug(
                        f"[RegressionHandling] Failed to emit curriculum event: {emit_err}"
                    )

                # Mar 20, 2026: REMOVED gauntlet re-trigger on regression.
                # This was the root cause of a CPU burn loop: regression detected →
                # trigger gauntlet → gauntlet detects same regression → loop forever.
                # The evaluation daemon already runs gauntlets on its regular schedule.
                # Removing this breaks the infinite loop without losing functionality.

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"[RegressionHandling] Error handling regression detected: {e}")


__all__ = ["RegressionHandlingMixin"]
