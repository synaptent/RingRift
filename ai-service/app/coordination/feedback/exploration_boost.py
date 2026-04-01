"""Exploration boost handling for FeedbackLoopController.

Sprint 17.9 (Jan 16, 2026): Extracted from feedback_loop_controller.py (~170 LOC)

This mixin provides exploration boost logic that responds to:
- Loss anomalies during training
- Training stalls / plateaus
- Training improvement (decay)

The exploration boost affects selfplay temperature scheduling to encourage
more diverse move selection when the model is struggling.

Usage:
    class FeedbackLoopController(ExplorationBoostMixin, ...):
        pass
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from app.config.thresholds import (
    EXPLORATION_BOOST_DECAY,
    EXPLORATION_BOOST_MAX,
    EXPLORATION_BOOST_PER_ANOMALY,
    EXPLORATION_BOOST_PER_STALL_GROUP,
    EXPLORATION_BOOST_STALL_MAX,
    TREND_DURATION_SEVERE,
)

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
            lambda t: logger.debug(f"[ExplorationBoost] Task {context} done")
            if not t.cancelled() and t.exception() is None
            else logger.warning(f"[ExplorationBoost] Task {context} failed: {t.exception()}")
            if t.exception() else None
        )
        return task
    except RuntimeError as e:
        if hasattr(coro, "close"):
            coro.close()
        logger.debug(f"[ExplorationBoost] Could not create task for {context}: {e}")
        return None


class ExplorationBoostMixin:
    """Mixin for exploration boost handling in FeedbackLoopController.

    Requires the host class to implement:
    - _get_or_create_state(config_key: str) -> FeedbackState

    Provides:
    - _boost_exploration_for_anomaly(config_key, anomaly_count)
    - _boost_exploration_for_stall(config_key, stall_epochs)
    - _reduce_exploration_after_improvement(config_key)
    """

    def _get_or_create_state(self, config_key: str) -> "FeedbackState":
        """Get or create state for a config. Must be implemented by host class."""
        raise NotImplementedError("Host class must implement _get_or_create_state")

    def _boost_exploration_for_anomaly(self, config_key: str, anomaly_count: int) -> None:
        """Boost exploration in response to loss anomalies.

        December 2025: Now includes fallback to store boost in FeedbackState
        even if temperature_scheduling module is not available. This ensures
        SelfplayScheduler can still use the exploration boost signal.

        P11-CRITICAL-1 (Dec 2025): Now emits EXPLORATION_BOOST event to close
        the feedback loop to selfplay exploration rates.
        """
        # Calculate boost: EXPLORATION_BOOST_PER_ANOMALY per anomaly, up to max
        boost = min(EXPLORATION_BOOST_MAX, 1.0 + EXPLORATION_BOOST_PER_ANOMALY * anomaly_count)

        # Always update local state (fallback for when schedulers aren't available)
        state = self._get_or_create_state(config_key)
        state.current_exploration_boost = boost
        logger.info(
            f"[FeedbackLoopController] Exploration boost set to {boost:.2f}x "
            f"for {config_key} (anomaly count: {anomaly_count})"
        )

        # P11-CRITICAL-1: Emit EXPLORATION_BOOST event to notify selfplay/temperature schedulers
        try:
            from app.coordination.event_router import emit_exploration_boost

            _safe_create_task(
                emit_exploration_boost(
                    config_key=config_key,
                    boost_factor=boost,
                    reason="loss_anomaly",
                    anomaly_count=anomaly_count,
                    source="FeedbackLoopController",
                ),
                context=f"emit_exploration_boost:loss_anomaly:{config_key}",
            )
            logger.debug(
                f"[FeedbackLoopController] Emitted EXPLORATION_BOOST event for {config_key}"
            )
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[FeedbackLoopController] Failed to emit EXPLORATION_BOOST: {e}")

        # Try to wire to active temperature schedulers (optional, may not be running)
        self._wire_exploration_to_schedulers(config_key, boost)

    def _boost_exploration_for_stall(self, config_key: str, stall_epochs: int) -> None:
        """Boost exploration in response to training stall.

        December 2025: Now includes fallback to store boost in FeedbackState.
        P11-CRITICAL-1 (Dec 2025): Now emits EXPLORATION_BOOST event.
        """
        # Boost per TREND_DURATION_SEVERE epochs of stall, up to max
        boost = min(
            EXPLORATION_BOOST_STALL_MAX,
            1.0 + EXPLORATION_BOOST_PER_STALL_GROUP * (stall_epochs // TREND_DURATION_SEVERE)
        )

        # Always update local state (fallback)
        state = self._get_or_create_state(config_key)
        state.current_exploration_boost = max(state.current_exploration_boost, boost)
        logger.info(
            f"[FeedbackLoopController] Exploration boost set to {boost:.2f}x "
            f"for {config_key} (stalled for {stall_epochs} epochs)"
        )

        # P11-CRITICAL-1: Emit EXPLORATION_BOOST event
        try:
            from app.coordination.event_router import emit_exploration_boost

            _safe_create_task(
                emit_exploration_boost(
                    config_key=config_key,
                    boost_factor=boost,
                    reason="stall",
                    anomaly_count=stall_epochs // 5,  # Use stall count as pseudo-anomaly count
                    source="FeedbackLoopController",
                ),
                context=f"emit_exploration_boost:stall:{config_key}",
            )
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.warning(f"[FeedbackLoopController] Failed to emit EXPLORATION_BOOST: {e}")

        # Try to wire to active temperature schedulers
        self._wire_exploration_to_schedulers(config_key, boost)

    def _reduce_exploration_after_improvement(self, config_key: str) -> None:
        """Gradually reduce exploration boost when training is improving.

        December 2025: Now includes fallback to update FeedbackState.
        Sprint 12 Session 8 (Jan 2026): Adaptive decay based on Elo velocity.
        - Fast improvement (>2 Elo/hour): 2% decay (preserve good exploration)
        - Normal improvement (0.5-2 Elo/hour): 10% decay (default)
        - Stalled (<0.5 Elo/hour): 0% decay, may boost exploration
        """
        # Get current boost from local state
        state = self._get_or_create_state(config_key)
        current_boost = state.current_exploration_boost

        if current_boost > 1.0:
            # Sprint 12: Adaptive decay based on Elo velocity
            velocity = state.elo_velocity

            if velocity > 2.0:
                # Fast improvement: slow decay to preserve what's working
                decay_factor = 0.98  # 2% decay
                decay_reason = "fast_improvement"
            elif velocity >= 0.5:
                # Normal improvement: standard decay
                decay_factor = EXPLORATION_BOOST_DECAY  # 10% decay
                decay_reason = "normal_improvement"
            elif velocity >= 0.0:
                # Stalled: no decay, hold exploration constant
                decay_factor = 1.0  # 0% decay
                decay_reason = "stalled"
            else:
                # Regression: boost exploration instead
                decay_factor = 1.05  # 5% increase
                decay_reason = "regression"

            # Apply decay (or boost in regression case)
            new_boost = current_boost * decay_factor
            # Clamp between 1.0 and EXPLORATION_BOOST_MAX (2.0)
            new_boost = max(1.0, min(EXPLORATION_BOOST_MAX, new_boost))
            state.current_exploration_boost = new_boost

            logger.debug(
                f"[FeedbackLoopController] Adaptive decay: {current_boost:.2f}→{new_boost:.2f}x "
                f"for {config_key} (velocity={velocity:.2f}, reason={decay_reason})"
            )

            # Try to wire to active temperature schedulers
            self._wire_exploration_to_schedulers(config_key, new_boost)

    def _wire_exploration_to_schedulers(self, config_key: str, boost: float) -> None:
        """Wire exploration boost to active temperature schedulers.

        This is a helper extracted to reduce duplication.
        """
        try:
            from app.training.temperature_scheduling import get_active_schedulers

            schedulers = get_active_schedulers()
            for scheduler in schedulers:
                if hasattr(scheduler, 'config_key') and scheduler.config_key == config_key:
                    if hasattr(scheduler, 'set_exploration_boost'):
                        scheduler.set_exploration_boost(boost)
                        logger.debug(
                            f"[FeedbackLoopController] Wired boost {boost:.2f}x to scheduler "
                            f"for {config_key}"
                        )
        except ImportError:
            logger.debug("[FeedbackLoopController] Temperature scheduling not available (using fallback)")
        except (AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[FeedbackLoopController] Could not wire to scheduler: {e}")


__all__ = ["ExplorationBoostMixin"]
