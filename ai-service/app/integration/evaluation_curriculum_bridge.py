"""
Evaluation → Curriculum Bridge for RingRift AI Self-Improvement Loop.

Connects evaluation results to selfplay curriculum adjustments:
1. Receives evaluation completion events
2. Analyzes per-config performance (Elo, win rate)
3. Updates curriculum weights to prioritize weak configs
4. Signals selfplay coordinator to adjust config selection

This closes the loop: eval results → curriculum → selfplay → training → eval

Usage:
    from app.integration.evaluation_curriculum_bridge import (
        EvaluationCurriculumBridge,
        create_evaluation_bridge,
    )

    # Create and start the bridge
    bridge = create_evaluation_bridge(
        feedback_controller=feedback_controller,
        feedback_router=feedback_router,
        selfplay_coordinator=selfplay_coordinator
    )
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CurriculumState:
    """Tracks curriculum state for all configurations."""
    # Config key -> current weight (1.0 = normal priority)
    weights: dict[str, float] = field(default_factory=dict)
    # Config key -> recent Elo ratings
    elo_history: dict[str, list[float]] = field(default_factory=dict)
    # Config key -> recent win rates
    win_rate_history: dict[str, list[float]] = field(default_factory=dict)
    # Last update timestamp
    last_update: float = 0.0
    # Number of updates
    update_count: int = 0


class EvaluationCurriculumBridge:
    """
    Bridges evaluation results to curriculum weight adjustments.

    The bridge:
    1. Listens to evaluation completion events
    2. Tracks per-config performance trends
    3. Computes curriculum weights based on weakness
    4. Pushes weights to selfplay coordinator
    """

    # All supported configs
    ALL_CONFIGS = [
        "square8_2p", "square8_3p", "square8_4p",
        "square19_2p", "square19_3p", "square19_4p",
        "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    ]

    def __init__(
        self,
        selfplay_coordinator=None,
        min_weight: float = 0.5,
        max_weight: float = 2.0,
        weak_threshold: float = 0.45,
        strong_threshold: float = 0.55,
        history_size: int = 20,
    ):
        """
        Initialize the bridge.

        Args:
            selfplay_coordinator: SelfplayCoordinator to update
            min_weight: Minimum curriculum weight
            max_weight: Maximum curriculum weight
            weak_threshold: Win rate below this = weak config
            strong_threshold: Win rate above this = strong config
            history_size: Number of results to keep per config
        """
        self.selfplay_coordinator = selfplay_coordinator
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weak_threshold = weak_threshold
        self.strong_threshold = strong_threshold
        self.history_size = history_size
        self.state = CurriculumState()
        self._callbacks: list[Callable] = []

        # Initialize all configs with default weight
        for config in self.ALL_CONFIGS:
            self.state.weights[config] = 1.0

    def add_evaluation_result(
        self,
        config_key: str,
        elo: float | None = None,
        win_rate: float | None = None,
        games_played: int = 0,
    ) -> dict[str, float]:
        """
        Add an evaluation result and recompute curriculum weights.

        Args:
            config_key: Configuration key (e.g., "square8_2p")
            elo: Elo rating from evaluation
            win_rate: Win rate from evaluation
            games_played: Number of games played

        Returns:
            Updated curriculum weights dict
        """
        # Track Elo history
        if elo is not None:
            if config_key not in self.state.elo_history:
                self.state.elo_history[config_key] = []
            self.state.elo_history[config_key].append(elo)
            if len(self.state.elo_history[config_key]) > self.history_size:
                self.state.elo_history[config_key] = self.state.elo_history[config_key][-self.history_size:]

        # Track win rate history
        if win_rate is not None:
            if config_key not in self.state.win_rate_history:
                self.state.win_rate_history[config_key] = []
            self.state.win_rate_history[config_key].append(win_rate)
            if len(self.state.win_rate_history[config_key]) > self.history_size:
                self.state.win_rate_history[config_key] = self.state.win_rate_history[config_key][-self.history_size:]

        # Recompute weights
        self._recompute_weights()
        self.state.last_update = time.time()
        self.state.update_count += 1

        # Push to selfplay coordinator
        if self.selfplay_coordinator:
            self.selfplay_coordinator.update_curriculum_weights(self.state.weights)
            logger.info(f"[Eval→Curriculum] Updated weights for {config_key}: {self.state.weights.get(config_key, 1.0):.2f}")

        # Fire callbacks
        for callback in self._callbacks:
            try:
                callback(config_key, self.state.weights.copy())
            except Exception as e:
                logger.error(f"[Eval→Curriculum] Callback error: {e}")

        return self.state.weights.copy()

    def _recompute_weights(self) -> None:
        """Recompute curriculum weights based on performance."""
        for config in self.ALL_CONFIGS:
            win_rates = self.state.win_rate_history.get(config, [])
            elos = self.state.elo_history.get(config, [])

            if not win_rates and not elos:
                # No data - keep default weight
                continue

            # Compute performance score
            avg_win_rate = sum(win_rates[-5:]) / len(win_rates[-5:]) if win_rates else 0.5
            elo_trend = self._compute_elo_trend(config)

            # Determine weight adjustment
            if avg_win_rate < self.weak_threshold:
                # Weak config - increase weight (more selfplay)
                weakness_factor = (self.weak_threshold - avg_win_rate) / self.weak_threshold
                new_weight = 1.0 + (weakness_factor * (self.max_weight - 1.0))
            elif avg_win_rate > self.strong_threshold:
                # Strong config - decrease weight (less selfplay)
                strength_factor = (avg_win_rate - self.strong_threshold) / (1.0 - self.strong_threshold)
                new_weight = 1.0 - (strength_factor * (1.0 - self.min_weight))
            else:
                # Normal performance - gradual adjustment based on trend
                if elo_trend < 10:  # Stagnating
                    new_weight = 1.1
                elif elo_trend < 0:  # Declining
                    new_weight = 1.2
                else:
                    new_weight = 1.0

            # Clamp to bounds
            self.state.weights[config] = max(self.min_weight, min(self.max_weight, new_weight))

    def _compute_elo_trend(self, config: str, lookback: int = 5) -> float:
        """Compute Elo trend (positive = improving)."""
        elos = self.state.elo_history.get(config, [])
        if len(elos) < lookback:
            return 0.0
        recent = elos[-lookback:]
        return recent[-1] - recent[0]

    def get_weak_configs(self) -> list[str]:
        """Get list of weak configurations that need more attention."""
        weak = []
        for config in self.ALL_CONFIGS:
            win_rates = self.state.win_rate_history.get(config, [])
            if win_rates and len(win_rates) >= 3:
                avg = sum(win_rates[-5:]) / len(win_rates[-5:])
                if avg < self.weak_threshold:
                    weak.append(config)
        return weak

    def get_curriculum_weights(self) -> dict[str, float]:
        """Get current curriculum weights."""
        return self.state.weights.copy()

    def register_callback(self, callback: Callable[[str, dict[str, float]], None]) -> None:
        """Register callback for weight updates."""
        self._callbacks.append(callback)

    def get_status(self) -> dict[str, Any]:
        """Get bridge status."""
        return {
            "weights": self.state.weights.copy(),
            "weak_configs": self.get_weak_configs(),
            "update_count": self.state.update_count,
            "last_update": self.state.last_update,
            "configs_tracked": len(self.state.elo_history),
        }


def create_evaluation_bridge(
    feedback_controller=None,
    feedback_router=None,
    selfplay_coordinator=None,
) -> EvaluationCurriculumBridge:
    """
    Create and wire up an evaluation-curriculum bridge.

    This function:
    1. Creates the bridge
    2. Registers it with the feedback router for INCREASE_CURRICULUM_WEIGHT signals
    3. Subscribes to evaluation events from feedback controller

    Args:
        feedback_controller: FeedbackController for subscribing to events
        feedback_router: FeedbackSignalRouter for signal handling
        selfplay_coordinator: SelfplayCoordinator to update

    Returns:
        Configured EvaluationCurriculumBridge
    """
    bridge = EvaluationCurriculumBridge(selfplay_coordinator=selfplay_coordinator)

    # Register with feedback router for curriculum weight signals
    if feedback_router:
        try:
            from app.integration.pipeline_feedback import FeedbackAction, FeedbackSignal

            async def handle_curriculum_signal(signal: FeedbackSignal) -> bool:
                """Handle curriculum weight increase signal."""
                metadata = signal.metadata or {}
                config = metadata.get('config')
                if config:
                    # This signal comes from evaluation analysis
                    # We don't need to recompute - just acknowledge
                    logger.debug(f"[Eval→Curriculum] Received curriculum signal for {config}")
                    return True
                return False

            feedback_router.register_handler(
                FeedbackAction.INCREASE_CURRICULUM_WEIGHT,
                handle_curriculum_signal,
                name="evaluation_curriculum_bridge"
            )
            logger.info("[Eval→Curriculum] Registered with feedback router")

        except ImportError:
            logger.warning("[Eval→Curriculum] pipeline_feedback not available")

    # Subscribe to evaluation events from feedback controller
    if feedback_controller and hasattr(feedback_controller, 'register_handler'):
        async def on_evaluation_result(result: dict[str, Any]) -> None:
            """Handle evaluation result from feedback controller."""
            config_key = result.get('config_key')
            if not config_key:
                return

            bridge.add_evaluation_result(
                config_key=config_key,
                elo=result.get('elo'),
                win_rate=result.get('win_rate'),
                games_played=result.get('games_played', 0)
            )

        feedback_controller.register_handler('evaluation', on_evaluation_result)
        logger.info("[Eval→Curriculum] Subscribed to evaluation events")

    return bridge


def integrate_evaluation_with_curriculum(
    feedback_controller,
    selfplay_coordinator,
) -> EvaluationCurriculumBridge:
    """
    One-line integration of evaluation → curriculum feedback loop.

    Usage:
        bridge = integrate_evaluation_with_curriculum(
            feedback_controller,
            selfplay_coordinator
        )

    Args:
        feedback_controller: FeedbackController instance
        selfplay_coordinator: SelfplayCoordinator instance

    Returns:
        Configured bridge
    """
    # Get the feedback router from the controller if available
    feedback_router = getattr(feedback_controller, 'signal_router', None)

    bridge = create_evaluation_bridge(
        feedback_controller=feedback_controller,
        feedback_router=feedback_router,
        selfplay_coordinator=selfplay_coordinator
    )

    logger.info("[Eval→Curriculum] Integration complete: evaluation → curriculum → selfplay")
    return bridge


# Convenience: Global bridge instance
_global_bridge: EvaluationCurriculumBridge | None = None


def get_evaluation_bridge() -> EvaluationCurriculumBridge | None:
    """Get the global evaluation bridge instance."""
    return _global_bridge


def set_evaluation_bridge(bridge: EvaluationCurriculumBridge) -> None:
    """Set the global evaluation bridge instance."""
    global _global_bridge
    _global_bridge = bridge
