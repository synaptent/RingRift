"""Adaptive controller for self-improvement loop optimization.

This module provides dynamic control over the improvement loop based on
convergence detection and performance trends. It implements:

1. Plateau detection: Stop after N iterations without improvement
2. Dynamic scaling: More selfplay games when improving, fewer when stable
3. Early stopping: Confidence-based evaluation termination
4. Win rate tracking: Historical analysis for trend detection
5. Quality-aware adaptation: Reduce games when data quality is poor (Dec 2025)

Event Integration:
- Emits PLATEAU_DETECTED when training stalls
- Emits HYPERPARAMETER_UPDATED when game counts change
- Subscribes to TRAINING_COMPLETED, EVALUATION_FAILED for automatic updates
- Subscribes to LOW_QUALITY_DATA_WARNING to reduce game generation (Dec 2025)
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Event system integration (optional - graceful fallback if not available)
try:
    from app.distributed.data_events import (
        DataEventType,
        get_event_bus,
        emit_plateau_detected,
        emit_hyperparameter_updated,
    )
    HAS_EVENT_SYSTEM = True
except ImportError:
    HAS_EVENT_SYSTEM = False
    logger.debug("Event system not available - AdaptiveController running standalone")


@dataclass
class IterationResult:
    """Result of a single improvement iteration."""
    iteration: int
    win_rate: float
    promoted: bool
    games_played: int
    eval_games: int


@dataclass
class AdaptiveController:
    """Controller for adaptive improvement loop behavior.

    Attributes:
        plateau_threshold: Number of iterations without improvement before stopping
        min_games: Minimum selfplay games per iteration
        max_games: Maximum selfplay games per iteration
        min_eval_games: Minimum evaluation games
        max_eval_games: Maximum evaluation games
        history: List of iteration results
        base_win_rate: Win rate threshold considered as "improvement"
        config_name: Configuration identifier for event tagging
        enable_events: Whether to emit events (default True if event system available)
    """

    plateau_threshold: int = 5
    min_games: int = 50
    max_games: int = 200
    min_eval_games: int = 50
    max_eval_games: int = 200
    base_win_rate: float = 0.55
    history: List[IterationResult] = field(default_factory=list)
    config_name: str = "default"
    enable_events: bool = True
    _last_game_count: int = field(default=0, repr=False)
    _last_eval_count: int = field(default=0, repr=False)
    _plateau_emitted: bool = field(default=False, repr=False)
    # Quality degradation factor (0.0 = no penalty, 1.0 = max penalty)
    _quality_penalty: float = field(default=0.0, repr=False)
    _quality_penalty_decay: float = field(default=0.1, repr=False)  # Per-iteration decay

    def record_iteration(
        self,
        iteration: int,
        win_rate: float,
        promoted: bool,
        games_played: int,
        eval_games: int,
    ) -> None:
        """Record the result of an iteration."""
        self.history.append(IterationResult(
            iteration=iteration,
            win_rate=win_rate,
            promoted=promoted,
            games_played=games_played,
            eval_games=eval_games,
        ))
        # Reset plateau emitted flag if we got a promotion
        if promoted:
            self._plateau_emitted = False
        # Decay quality penalty each iteration
        self._quality_penalty = max(0.0, self._quality_penalty - self._quality_penalty_decay)

    async def record_iteration_async(
        self,
        iteration: int,
        win_rate: float,
        promoted: bool,
        games_played: int,
        eval_games: int,
    ) -> None:
        """Record iteration result and emit events if appropriate.

        Async version that checks for plateau and emits events.
        """
        self.record_iteration(iteration, win_rate, promoted, games_played, eval_games)

        # Check for plateau and emit if detected
        if not self.should_continue():
            await self.emit_plateau_if_detected()

        # Check if game counts changed and emit
        await self.emit_game_count_update()

    async def emit_plateau_if_detected(self) -> bool:
        """Emit PLATEAU_DETECTED event if we've hit the plateau threshold.

        Returns True if event was emitted, False otherwise.
        Only emits once per plateau period.
        """
        if not HAS_EVENT_SYSTEM or not self.enable_events:
            return False

        if self._plateau_emitted:
            return False

        plateau_count = self.get_plateau_count()
        if plateau_count >= self.plateau_threshold:
            stats = self.get_statistics()
            await emit_plateau_detected(
                config=self.config_name,
                iterations_without_improvement=plateau_count,
                avg_win_rate=stats.get("avg_win_rate", 0.0),
                recommended_action="Consider hyperparameter search or curriculum adjustment",
                source="adaptive_controller",
            )
            self._plateau_emitted = True
            logger.info(
                f"Plateau detected for {self.config_name}: "
                f"{plateau_count} iterations without improvement"
            )
            return True
        return False

    async def emit_game_count_update(self) -> bool:
        """Emit HYPERPARAMETER_UPDATED if game counts have changed.

        Returns True if event was emitted, False otherwise.
        """
        if not HAS_EVENT_SYSTEM or not self.enable_events:
            return False

        new_games = self.compute_games()
        new_eval = self.compute_eval_games()

        if new_games == self._last_game_count and new_eval == self._last_eval_count:
            return False

        old_games = self._last_game_count
        old_eval = self._last_eval_count
        self._last_game_count = new_games
        self._last_eval_count = new_eval

        # Only emit if this isn't the initial computation
        if old_games == 0 and old_eval == 0:
            return False

        await emit_hyperparameter_updated(
            config=self.config_name,
            parameter="adaptive_game_counts",
            old_value={"games": old_games, "eval_games": old_eval},
            new_value={"games": new_games, "eval_games": new_eval},
            reason=f"Trend-based adjustment (trend_factor={self._compute_trend_factor():.2f})",
            source="adaptive_controller",
        )
        logger.debug(
            f"Game counts updated for {self.config_name}: "
            f"games {old_games}->{new_games}, eval {old_eval}->{new_eval}"
        )
        return True

    def setup_event_subscriptions(
        self,
        on_training_completed: Optional[Callable] = None,
        on_evaluation_failed: Optional[Callable] = None,
        on_quality_warning: Optional[Callable] = None,
    ) -> None:
        """Set up subscriptions to relevant events.

        Args:
            on_training_completed: Optional custom handler for training completion
            on_evaluation_failed: Optional custom handler for evaluation failures
            on_quality_warning: Optional custom handler for quality warnings
        """
        if not HAS_EVENT_SYSTEM:
            logger.warning("Event system not available - cannot set up subscriptions")
            return

        bus = get_event_bus()

        # Subscribe to training completion to track iterations
        async def handle_training_completed(event):
            payload = event.payload
            if payload.get("config") == self.config_name:
                if on_training_completed:
                    await on_training_completed(event)
                logger.debug(f"Training completed for {self.config_name}")

        bus.subscribe(DataEventType.TRAINING_COMPLETED, handle_training_completed)

        # Subscribe to evaluation failures for tracking
        async def handle_evaluation_failed(event):
            payload = event.payload
            if payload.get("config") == self.config_name:
                if on_evaluation_failed:
                    await on_evaluation_failed(event)
                logger.debug(f"Evaluation failed for {self.config_name}")

        bus.subscribe(DataEventType.EVALUATION_FAILED, handle_evaluation_failed)

        # Subscribe to quality warnings for adaptive game reduction (December 2025)
        async def handle_low_quality_warning(event):
            payload = event.payload
            if payload.get("config") == self.config_name:
                # Compute penalty from quality ratio
                quality_ratio = payload.get("quality_ratio", 0.0)
                low_count = payload.get("low_quality_count", 0)

                # Higher quality_ratio = more low quality data = higher penalty
                penalty = min(1.0, quality_ratio * 1.5)  # Scale up ratio
                self.apply_quality_penalty(
                    penalty,
                    reason=f"{low_count} low quality games ({quality_ratio:.1%})"
                )

                if on_quality_warning:
                    await on_quality_warning(event)

        bus.subscribe(DataEventType.LOW_QUALITY_DATA_WARNING, handle_low_quality_warning)

        logger.info(f"Event subscriptions set up for AdaptiveController({self.config_name})")

    def should_continue(self) -> bool:
        """Determine if the improvement loop should continue.

        Returns True if:
        - Not enough history yet (< plateau_threshold iterations)
        - At least one recent iteration showed improvement
        """
        if len(self.history) < self.plateau_threshold:
            return True

        recent = self.history[-self.plateau_threshold:]
        return any(r.promoted for r in recent)

    def get_plateau_count(self) -> int:
        """Count consecutive iterations without improvement."""
        count = 0
        for result in reversed(self.history):
            if result.promoted:
                break
            count += 1
        return count

    def compute_games(self, recent_win_rate: Optional[float] = None) -> int:
        """Compute optimal number of selfplay games for next iteration.

        Strategy:
        - Marginal win rates (0.45-0.55): More games needed for better signal
        - Clear results (< 0.45 or > 0.55): Fewer games needed
        - Rising trend: More games to capitalize on momentum
        - Plateau: Fewer games to reduce wasted compute
        """
        if recent_win_rate is None and self.history:
            recent_win_rate = self.history[-1].win_rate

        if recent_win_rate is None:
            return (self.min_games + self.max_games) // 2

        # Marginal results need more data
        if 0.45 < recent_win_rate < 0.55:
            base = self.max_games
        else:
            base = self.min_games

        # Adjust based on trend
        trend_factor = self._compute_trend_factor()
        adjusted = int(base * trend_factor)

        return max(self.min_games, min(self.max_games, adjusted))

    def compute_eval_games(self, recent_win_rate: Optional[float] = None) -> int:
        """Compute optimal number of evaluation games.

        Strategy:
        - Marginal win rates: More eval games for confidence
        - Clear winners/losers: Fewer eval games (obvious result)
        """
        if recent_win_rate is None and self.history:
            recent_win_rate = self.history[-1].win_rate

        if recent_win_rate is None:
            return (self.min_eval_games + self.max_eval_games) // 2

        # Marginal results need more evaluation
        if 0.48 < recent_win_rate < 0.52:
            return self.max_eval_games
        elif 0.45 < recent_win_rate < 0.55:
            return (self.min_eval_games + self.max_eval_games) // 2
        else:
            return self.min_eval_games

    def _compute_trend_factor(self) -> float:
        """Compute trend factor based on recent history and quality.

        Returns:
            > 1.0: Improving trend (boost games)
            < 1.0: Declining/plateau trend (reduce games)
            = 1.0: No clear trend

        Quality penalty reduces the trend factor when data quality is poor,
        preventing wasteful game generation with bad data.
        """
        if len(self.history) < 3:
            base_factor = 1.0
        else:
            recent_wins = [r.promoted for r in self.history[-5:]]
            win_count = sum(1 for w in recent_wins if w)
            win_ratio = win_count / len(recent_wins)

            if win_ratio > 0.6:
                # Strong improvement - boost games
                base_factor = 1.3
            elif win_ratio > 0.4:
                # Moderate improvement - slight boost
                base_factor = 1.1
            elif win_ratio > 0.2:
                # Weak improvement - maintain
                base_factor = 1.0
            else:
                # Plateau - reduce games
                base_factor = 0.8

        # Apply quality penalty (reduce games when quality is poor)
        # Penalty of 1.0 reduces factor by 30%
        quality_adjustment = 1.0 - (self._quality_penalty * 0.3)
        return base_factor * quality_adjustment

    def apply_quality_penalty(self, penalty: float, reason: str = "") -> None:
        """Apply a quality penalty to reduce game generation.

        Called when LOW_QUALITY_DATA_WARNING is received. The penalty
        decays over iterations as quality may improve.

        Args:
            penalty: Penalty value (0.0 to 1.0, higher = worse quality)
            reason: Optional reason for logging
        """
        old_penalty = self._quality_penalty
        self._quality_penalty = max(self._quality_penalty, min(1.0, penalty))
        if self._quality_penalty > old_penalty:
            logger.info(
                f"Quality penalty applied for {self.config_name}: "
                f"{old_penalty:.2f} -> {self._quality_penalty:.2f}"
                f"{f' ({reason})' if reason else ''}"
            )

    def get_statistics(self) -> dict:
        """Get summary statistics of the improvement loop."""
        if not self.history:
            return {
                "total_iterations": 0,
                "total_promotions": 0,
                "promotion_rate": 0.0,
                "avg_win_rate": 0.0,
                "plateau_count": 0,
                "trend": "unknown",
            }

        total = len(self.history)
        promotions = sum(1 for r in self.history if r.promoted)
        win_rates = [r.win_rate for r in self.history]

        # Compute trend
        if len(win_rates) >= 3:
            recent_avg = statistics.mean(win_rates[-3:])
            older_avg = statistics.mean(win_rates[:-3]) if len(win_rates) > 3 else win_rates[0]
            if recent_avg > older_avg + 0.02:
                trend = "improving"
            elif recent_avg < older_avg - 0.02:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "total_iterations": total,
            "total_promotions": promotions,
            "promotion_rate": promotions / total if total > 0 else 0.0,
            "avg_win_rate": statistics.mean(win_rates),
            "plateau_count": self.get_plateau_count(),
            "trend": trend,
            "quality_penalty": self._quality_penalty,
        }

    def save(self, path: Path) -> None:
        """Save controller state to JSON file."""
        state = {
            "plateau_threshold": self.plateau_threshold,
            "min_games": self.min_games,
            "max_games": self.max_games,
            "min_eval_games": self.min_eval_games,
            "max_eval_games": self.max_eval_games,
            "base_win_rate": self.base_win_rate,
            "config_name": self.config_name,
            "enable_events": self.enable_events,
            "history": [asdict(r) for r in self.history],
            "quality_penalty": self._quality_penalty,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: Path, config_name: Optional[str] = None) -> "AdaptiveController":
        """Load controller state from JSON file.

        Args:
            path: Path to JSON state file
            config_name: Override config name (useful when loading shared state)
        """
        if not path.exists():
            return cls(config_name=config_name or "default")

        try:
            state = json.loads(path.read_text())
            history = [
                IterationResult(**r) for r in state.get("history", [])
            ]
            controller = cls(
                plateau_threshold=state.get("plateau_threshold", 5),
                min_games=state.get("min_games", 50),
                max_games=state.get("max_games", 200),
                min_eval_games=state.get("min_eval_games", 50),
                max_eval_games=state.get("max_eval_games", 200),
                base_win_rate=state.get("base_win_rate", 0.55),
                config_name=config_name or state.get("config_name", "default"),
                enable_events=state.get("enable_events", True),
                history=history,
            )
            # Restore quality penalty if saved (December 2025)
            controller._quality_penalty = state.get("quality_penalty", 0.0)
            return controller
        except (json.JSONDecodeError, TypeError, KeyError):
            return cls()


def create_adaptive_controller(
    *,
    plateau_threshold: int = 5,
    min_games: int = 50,
    max_games: int = 200,
    config_name: str = "default",
    enable_events: bool = True,
    setup_subscriptions: bool = False,
    state_path: Optional[Path] = None,
) -> AdaptiveController:
    """Factory function to create an adaptive controller.

    If state_path is provided and exists, loads existing state.
    Otherwise creates a new controller with the given parameters.

    Args:
        plateau_threshold: Iterations without improvement before stopping
        min_games: Minimum selfplay games per iteration
        max_games: Maximum selfplay games per iteration
        config_name: Configuration identifier for event tagging
        enable_events: Whether to emit events
        setup_subscriptions: Whether to set up event subscriptions
        state_path: Path to persist/load state
    """
    if state_path and state_path.exists():
        controller = AdaptiveController.load(state_path, config_name=config_name)
        # Update parameters (allows reconfiguration)
        controller.plateau_threshold = plateau_threshold
        controller.min_games = min_games
        controller.max_games = max_games
        controller.enable_events = enable_events
    else:
        controller = AdaptiveController(
            plateau_threshold=plateau_threshold,
            min_games=min_games,
            max_games=max_games,
            config_name=config_name,
            enable_events=enable_events,
        )

    if setup_subscriptions:
        controller.setup_event_subscriptions()

    return controller
