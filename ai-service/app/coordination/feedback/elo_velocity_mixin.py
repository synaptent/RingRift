"""Elo velocity and adaptation handling for FeedbackLoopController.

Sprint 17.9 (Jan 16, 2026): Extracted from feedback_loop_controller.py (~166 LOC)

This mixin provides Elo velocity tracking and adaptation logic that responds to:
- Elo velocity (improvement rate per hour)
- Plateau detection and response
- Adaptive training parameter adjustments based on evaluation results

The Elo feedback affects selfplay search budget, exploration boost, and
training parameters to optimize model improvement.

Usage:
    class FeedbackLoopController(EloVelocityAdaptationMixin, ...):
        pass
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.config.thresholds import (
    ELO_TARGET_ALL_CONFIGS,
    ELO_PLATEAU_PER_HOUR,
    ELO_FAST_IMPROVEMENT_PER_HOUR,
    EXPLORATION_BOOST_MAX,
    EXPLORATION_BOOST_MULTIPLIER,
)

if TYPE_CHECKING:
    from app.coordination.feedback_loop_controller import (
        FeedbackState,
        AdaptiveTrainingSignal,
    )

logger = logging.getLogger(__name__)


class EloVelocityAdaptationMixin:
    """Mixin for Elo velocity and adaptation handling in FeedbackLoopController.

    Requires the host class to implement:
    - _get_or_create_state(config_key: str) -> FeedbackState
    - AdaptiveTrainingSignal (imported from feedback_loop_controller)

    Provides:
    - _adjust_selfplay_for_velocity(config_key, state, elo, velocity)
    - _emit_selfplay_adjustment(config_key, state, elo_gap, velocity)
    - _compute_adaptive_signal(config_key, state, eval_result) -> AdaptiveTrainingSignal
    - _emit_adaptive_training_signal(config_key, signal)
    """

    def _get_or_create_state(self, config_key: str) -> "FeedbackState":
        """Get or create state for a config. Must be implemented by host class."""
        raise NotImplementedError("Host class must implement _get_or_create_state")

    def _adjust_selfplay_for_velocity(
        self, config_key: str, state: "FeedbackState", elo: float, velocity: float
    ) -> None:
        """Adjust selfplay intensity based on Elo velocity.

        Dec 28 2025: Key feedback loop for reaching 2000+ Elo.
        - Low velocity (plateau) -> Increase search budget, boost exploration
        - High velocity (improving) -> Maintain current settings
        - Near 2000 Elo goal -> Fine-tune with higher budget
        """
        elo_gap = ELO_TARGET_ALL_CONFIGS - elo
        is_plateau = velocity < ELO_PLATEAU_PER_HOUR and len(state.elo_history) >= 3
        is_fast = velocity > ELO_FAST_IMPROVEMENT_PER_HOUR

        # Determine new search budget based on velocity and gap
        if is_plateau:
            # Plateau detected - boost search budget
            old_budget = state.current_search_budget
            new_budget = min(800, old_budget + 100)  # Increase by 100, cap at 800
            state.current_search_budget = new_budget
            state.current_exploration_boost = min(
                EXPLORATION_BOOST_MAX, state.current_exploration_boost * EXPLORATION_BOOST_MULTIPLIER
            )

            logger.warning(
                f"[EloVelocity] PLATEAU DETECTED for {config_key}: "
                f"velocity={velocity:.1f} Elo/hr, elo_gap={elo_gap:.0f}. "
                f"Increasing budget {old_budget}->{new_budget}, "
                f"exploration_boost={state.current_exploration_boost:.2f}"
            )
        elif is_fast:
            # Fast improvement - maintain current settings
            logger.info(
                f"[EloVelocity] Fast improvement for {config_key}: "
                f"velocity={velocity:.1f} Elo/hr. Maintaining settings."
            )
        elif elo > 1800:
            # Near goal - fine-tune with higher budget
            state.current_search_budget = max(600, state.current_search_budget)
            logger.info(
                f"[EloVelocity] Near goal for {config_key}: "
                f"elo={elo:.0f}, using budget={state.current_search_budget}"
            )

        # Emit selfplay target update event
        self._emit_selfplay_adjustment(config_key, state, elo_gap, velocity)

    def _emit_selfplay_adjustment(
        self, config_key: str, state: "FeedbackState", elo_gap: float, velocity: float
    ) -> None:
        """Emit SELFPLAY_TARGET_UPDATED event with adjusted parameters.

        Dec 28 2025: This event is consumed by SelfplayScheduler to adjust
        search budget and game allocation.
        """
        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus:
                bus.emit(DataEventType.SELFPLAY_TARGET_UPDATED, {
                    "config_key": config_key,
                    "search_budget": state.current_search_budget,
                    "exploration_boost": state.current_exploration_boost,
                    "elo_gap": elo_gap,
                    "velocity": velocity,
                    "priority": "HIGH" if elo_gap > 500 or velocity < ELO_PLATEAU_PER_HOUR else "NORMAL",
                    "reason": "velocity_feedback",
                })
                logger.debug(
                    f"[EloVelocity] Emitted SELFPLAY_TARGET_UPDATED for {config_key}"
                )
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.debug(f"[EloVelocity] Could not emit selfplay adjustment: {e}")

    def _compute_adaptive_signal(
        self, config_key: str, state: "FeedbackState", eval_result: dict
    ) -> "AdaptiveTrainingSignal":
        """Compute adaptive training parameters based on evaluation results.

        December 29, 2025: Phase 6 - Training parameters adapt to eval outcomes.

        Strategy:
        - Strong improvement (>50 Elo): Extend training epochs to capitalize
        - Plateau (<10 Elo improvement): Reduce LR, increase batch size
        - Regression (<-30 Elo): Aggressive LR reduction, enable gradient clipping

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            state: Current feedback state for this config
            eval_result: Evaluation result dict with elo, win_rate, etc.

        Returns:
            AdaptiveTrainingSignal with adjusted training parameters
        """
        # Import here to avoid circular dependency
        from app.coordination.feedback_loop_controller import AdaptiveTrainingSignal

        signal = AdaptiveTrainingSignal()

        # Compute Elo improvement from history
        current_elo = eval_result.get("elo", 1500.0)
        prev_elo = state.last_elo if state.last_elo > 0 else current_elo

        # Only compute improvement if we have history
        if len(state.elo_history) >= 2:
            # Use second-to-last entry as previous reference
            _, prev_elo = state.elo_history[-2]

        elo_improvement = current_elo - prev_elo

        # Strong improvement: extend training
        if elo_improvement > 50:
            signal.epochs_extension = 10
            signal.reason = f"Strong improvement ({elo_improvement:.0f} Elo) - extending training"
            logger.info(
                f"[EloVelocity] Adaptive signal for {config_key}: "
                f"{signal.reason}"
            )

        # Plateau: reduce LR, increase batch size
        elif elo_improvement < 10:
            signal.learning_rate_multiplier = 0.5
            signal.batch_size_multiplier = 1.5
            signal.gradient_clip_enabled = True
            signal.reason = f"Plateau ({elo_improvement:.0f} Elo) - reducing LR, enabling grad clip"
            logger.info(
                f"[EloVelocity] Adaptive signal for {config_key}: "
                f"{signal.reason}"
            )

        # Regression: aggressive LR reduction
        if elo_improvement < -30:
            signal.learning_rate_multiplier = 0.2
            signal.gradient_clip_enabled = True
            signal.epochs_extension = 0  # Don't extend on regression
            signal.reason = f"Regression ({elo_improvement:.0f} Elo) - aggressive LR reduction"
            logger.warning(
                f"[EloVelocity] Adaptive signal for {config_key}: "
                f"{signal.reason}"
            )

        return signal

    def _emit_adaptive_training_signal(
        self, config_key: str, signal: "AdaptiveTrainingSignal"
    ) -> None:
        """Emit ADAPTIVE_PARAMS_CHANGED event with training adjustments.

        December 29, 2025: Phase 6 - Consumed by training system.
        """
        if signal.reason == "":
            # No adjustment needed
            return

        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus and hasattr(DataEventType, 'ADAPTIVE_PARAMS_CHANGED'):
                bus.emit(DataEventType.ADAPTIVE_PARAMS_CHANGED, {
                    "config_key": config_key,
                    **signal.to_dict(),
                })
                logger.debug(
                    f"[EloVelocity] Emitted ADAPTIVE_PARAMS_CHANGED for {config_key}"
                )
        except (ImportError, AttributeError, TypeError, RuntimeError) as e:
            logger.debug(
                f"[EloVelocity] Could not emit adaptive signal: {e}"
            )


__all__ = ["EloVelocityAdaptationMixin"]
