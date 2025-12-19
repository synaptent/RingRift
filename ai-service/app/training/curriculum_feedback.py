"""Curriculum Feedback Loop for Training.

Closes the loop between selfplay performance and curriculum weights:
1. Track selfplay metrics (win rates, game counts) per config
2. Feed back to curriculum weights more frequently
3. Adjust selfplay allocation based on model performance

This creates a responsive system where:
- Weak configs get more training attention
- Strong configs get less training (resources reallocated)
- Metrics update in near real-time (not just hourly)

Usage:
    from app.training.curriculum_feedback import CurriculumFeedback

    feedback = CurriculumFeedback()

    # Record selfplay results
    feedback.record_game("square8_2p", winner=1, model_elo=1650)

    # Get updated curriculum weights
    weights = feedback.get_curriculum_weights()
    # {"square8_2p": 0.8, "hexagonal_2p": 1.2, ...}

    # Export weights for P2P orchestrator
    feedback.export_weights_json("curriculum_weights.json")
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.paths import AI_SERVICE_ROOT

logger = logging.getLogger(__name__)

# Constants
DEFAULT_WEIGHT_MIN = 0.5
DEFAULT_WEIGHT_MAX = 2.0
DEFAULT_LOOKBACK_MINUTES = 30
DEFAULT_TARGET_WIN_RATE = 0.55


@dataclass
class ConfigMetrics:
    """Metrics for a single config."""
    games_total: int = 0
    games_recent: int = 0  # In lookback window
    wins_recent: int = 0
    losses_recent: int = 0
    draws_recent: int = 0
    avg_elo: float = 1500.0
    win_rate: float = 0.5
    elo_trend: float = 0.0  # Positive = improving
    last_game_time: float = 0
    last_training_time: float = 0
    model_count: int = 0

    @property
    def recent_win_rate(self) -> float:
        """Win rate over recent games."""
        total = self.wins_recent + self.losses_recent + self.draws_recent
        if total == 0:
            return 0.5
        return (self.wins_recent + 0.5 * self.draws_recent) / total


@dataclass
class GameRecord:
    """A single game record for tracking."""
    config_key: str
    timestamp: float
    winner: int  # 1 = model won, -1 = model lost, 0 = draw
    model_elo: float = 1500.0
    opponent_type: str = "baseline"  # baseline, selfplay, etc.


class CurriculumFeedback:
    """Manages curriculum feedback loop with real-time metrics."""

    def __init__(
        self,
        lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
        weight_min: float = DEFAULT_WEIGHT_MIN,
        weight_max: float = DEFAULT_WEIGHT_MAX,
        target_win_rate: float = DEFAULT_TARGET_WIN_RATE,
    ):
        self.lookback_minutes = lookback_minutes
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.target_win_rate = target_win_rate

        # Game history (circular buffer per config)
        self._game_history: Dict[str, List[GameRecord]] = defaultdict(list)
        self._max_history_per_config = 1000

        # Cached metrics per config
        self._config_metrics: Dict[str, ConfigMetrics] = {}

        # Last update time for change detection
        self._last_update_time: float = 0

    def record_game(
        self,
        config_key: str,
        winner: int,
        model_elo: float = 1500.0,
        opponent_type: str = "baseline",
    ) -> None:
        """Record a game result.

        Args:
            config_key: Config identifier (e.g., "square8_2p")
            winner: 1 = model won, -1 = model lost, 0 = draw
            model_elo: Current model Elo rating
            opponent_type: Type of opponent (baseline, selfplay, etc.)
        """
        record = GameRecord(
            config_key=config_key,
            timestamp=time.time(),
            winner=winner,
            model_elo=model_elo,
            opponent_type=opponent_type,
        )

        # Add to history
        history = self._game_history[config_key]
        history.append(record)

        # Trim to max size
        if len(history) > self._max_history_per_config:
            self._game_history[config_key] = history[-self._max_history_per_config:]

        self._last_update_time = time.time()

    def record_training(self, config_key: str) -> None:
        """Record that training ran for a config."""
        metrics = self._get_or_create_metrics(config_key)
        metrics.last_training_time = time.time()
        self._last_update_time = time.time()

    def update_model_count(self, config_key: str, count: int) -> None:
        """Update the model count for a config."""
        metrics = self._get_or_create_metrics(config_key)
        metrics.model_count = count

    def _get_or_create_metrics(self, config_key: str) -> ConfigMetrics:
        """Get or create metrics for a config."""
        if config_key not in self._config_metrics:
            self._config_metrics[config_key] = ConfigMetrics()
        return self._config_metrics[config_key]

    def _compute_metrics(self, config_key: str) -> ConfigMetrics:
        """Compute metrics for a config from game history."""
        metrics = self._get_or_create_metrics(config_key)
        history = self._game_history.get(config_key, [])

        if not history:
            return metrics

        now = time.time()
        lookback_cutoff = now - (self.lookback_minutes * 60)

        # Count recent games
        recent_games = [g for g in history if g.timestamp >= lookback_cutoff]

        metrics.games_total = len(history)
        metrics.games_recent = len(recent_games)
        metrics.wins_recent = sum(1 for g in recent_games if g.winner == 1)
        metrics.losses_recent = sum(1 for g in recent_games if g.winner == -1)
        metrics.draws_recent = sum(1 for g in recent_games if g.winner == 0)

        # Compute win rate
        if metrics.games_recent > 0:
            metrics.win_rate = metrics.recent_win_rate

        # Compute average Elo (from recent games)
        if recent_games:
            metrics.avg_elo = sum(g.model_elo for g in recent_games) / len(recent_games)

            # Compute Elo trend (compare first half to second half)
            if len(recent_games) >= 10:
                mid = len(recent_games) // 2
                first_half_elo = sum(g.model_elo for g in recent_games[:mid]) / mid
                second_half_elo = sum(g.model_elo for g in recent_games[mid:]) / (len(recent_games) - mid)
                metrics.elo_trend = second_half_elo - first_half_elo

        if history:
            metrics.last_game_time = max(g.timestamp for g in history)

        return metrics

    def get_config_metrics(self, config_key: str) -> ConfigMetrics:
        """Get current metrics for a config."""
        return self._compute_metrics(config_key)

    def get_all_metrics(self) -> Dict[str, ConfigMetrics]:
        """Get metrics for all configs."""
        result = {}
        for config_key in set(self._game_history.keys()) | set(self._config_metrics.keys()):
            result[config_key] = self._compute_metrics(config_key)
        return result

    def get_curriculum_weights(self) -> Dict[str, float]:
        """Compute curriculum weights based on current metrics.

        Weighting strategy:
        - Low win rate (< target) → Higher weight (more training needed)
        - High win rate (> target) → Lower weight (already strong)
        - Few models → Higher weight (bootstrap priority)
        - Declining Elo → Higher weight (regression detected)

        Returns:
            Dict mapping config_key → weight (0.5 to 2.0)
        """
        all_metrics = self.get_all_metrics()

        if not all_metrics:
            return {}

        weights = {}

        for config_key, metrics in all_metrics.items():
            weight = 1.0

            # Win rate adjustment
            win_rate_diff = self.target_win_rate - metrics.win_rate
            if metrics.games_recent >= 10:  # Only adjust if enough data
                # Scale: -0.2 diff → 0.6 weight, +0.2 diff → 1.4 weight
                weight += win_rate_diff * 2.0

            # Model count adjustment (bootstrap priority)
            if metrics.model_count == 0:
                weight *= 1.5  # Major boost for new configs
            elif metrics.model_count == 1:
                weight *= 1.2  # Moderate boost

            # Elo trend adjustment
            if metrics.elo_trend < -20:  # Significant regression
                weight *= 1.2
            elif metrics.elo_trend > 30:  # Significant improvement
                weight *= 0.9

            # Time since training adjustment
            if metrics.last_training_time > 0:
                hours_since_training = (time.time() - metrics.last_training_time) / 3600
                if hours_since_training > 6:
                    weight *= 1.1  # Slight boost for stale configs

            # Clamp to bounds
            weight = max(self.weight_min, min(self.weight_max, weight))
            weights[config_key] = round(weight, 3)

        return weights

    def export_weights_json(self, output_path: str) -> None:
        """Export curriculum weights to JSON for P2P orchestrator.

        Args:
            output_path: Path to output JSON file
        """
        weights = self.get_curriculum_weights()
        metrics = self.get_all_metrics()

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "weights": weights,
            "metrics": {
                config_key: {
                    "games_recent": m.games_recent,
                    "win_rate": round(m.win_rate, 3),
                    "avg_elo": round(m.avg_elo, 1),
                    "elo_trend": round(m.elo_trend, 1),
                    "model_count": m.model_count,
                }
                for config_key, m in metrics.items()
            },
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported curriculum weights to {output_path}")

    def should_update_curriculum(self, min_games: int = 50) -> bool:
        """Check if curriculum should be updated based on new data.

        Args:
            min_games: Minimum new games since last update

        Returns:
            True if enough new data to warrant curriculum update
        """
        total_recent = sum(
            m.games_recent for m in self.get_all_metrics().values()
        )
        return total_recent >= min_games


# Singleton instance
_feedback_instance: Optional[CurriculumFeedback] = None


def get_curriculum_feedback() -> CurriculumFeedback:
    """Get the global curriculum feedback instance."""
    global _feedback_instance
    if _feedback_instance is None:
        _feedback_instance = CurriculumFeedback()
    return _feedback_instance


def record_selfplay_game(
    config_key: str,
    winner: int,
    model_elo: float = 1500.0,
) -> None:
    """Record a selfplay game result (convenience function)."""
    get_curriculum_feedback().record_game(
        config_key, winner, model_elo, opponent_type="selfplay"
    )


def get_curriculum_weights() -> Dict[str, float]:
    """Get current curriculum weights (convenience function)."""
    return get_curriculum_feedback().get_curriculum_weights()


# =============================================================================
# ELO Change → Curriculum Rebalance Integration (December 2025)
# =============================================================================

class EloToCurriculumWatcher:
    """Watches for ELO changes and triggers curriculum rebalancing.

    Subscribes to ELO_UPDATED events from the event bus and triggers
    curriculum weight recalculation when significant ELO changes occur.

    Usage:
        from app.training.curriculum_feedback import (
            wire_elo_to_curriculum,
            get_elo_curriculum_watcher,
        )

        # Wire ELO events to curriculum
        watcher = wire_elo_to_curriculum()

        # Or get existing watcher
        watcher = get_elo_curriculum_watcher()
        if watcher:
            watcher.force_rebalance()
    """

    def __init__(
        self,
        feedback: Optional[CurriculumFeedback] = None,
        significant_elo_change: float = 30.0,
        rebalance_cooldown_seconds: float = 300.0,
        auto_export: bool = True,
        export_path: str = "data/curriculum_weights.json",
    ):
        """Initialize the ELO-to-curriculum watcher.

        Args:
            feedback: CurriculumFeedback instance (uses singleton if None)
            significant_elo_change: ELO change threshold to trigger rebalance
            rebalance_cooldown_seconds: Minimum time between rebalances
            auto_export: Whether to auto-export weights after rebalance
            export_path: Path to export weights JSON
        """
        self.feedback = feedback or get_curriculum_feedback()
        self.significant_elo_change = significant_elo_change
        self.rebalance_cooldown_seconds = rebalance_cooldown_seconds
        self.auto_export = auto_export
        self.export_path = export_path

        self._last_rebalance_time: float = 0.0
        self._elo_history: Dict[str, List[float]] = defaultdict(list)
        self._subscribed = False

    def subscribe_to_elo_events(self) -> bool:
        """Subscribe to ELO update events from the event bus.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.subscribe(DataEventType.ELO_UPDATED, self._on_elo_updated)
            self._subscribed = True
            logger.info("[EloToCurriculumWatcher] Subscribed to ELO_UPDATED events")
            return True
        except Exception as e:
            logger.warning(f"[EloToCurriculumWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from ELO events."""
        if not self._subscribed:
            return

        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.ELO_UPDATED, self._on_elo_updated)
            self._subscribed = False
        except Exception:
            pass

    def _on_elo_updated(self, event: Any) -> None:
        """Handle ELO_UPDATED event.

        Checks if the ELO change is significant and triggers rebalance if needed.
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config_key", "")
        new_elo = payload.get("new_elo") or payload.get("elo")
        old_elo = payload.get("old_elo")

        if not config_key or new_elo is None:
            return

        # Track ELO history
        self._elo_history[config_key].append(new_elo)
        if len(self._elo_history[config_key]) > 20:
            self._elo_history[config_key] = self._elo_history[config_key][-20:]

        # Calculate ELO change
        elo_change = 0.0
        if old_elo is not None:
            elo_change = abs(new_elo - old_elo)
        elif len(self._elo_history[config_key]) >= 2:
            elo_change = abs(new_elo - self._elo_history[config_key][-2])

        # Check if significant change
        if elo_change >= self.significant_elo_change:
            logger.info(
                f"[EloToCurriculumWatcher] Significant ELO change for {config_key}: "
                f"{elo_change:.1f} points"
            )
            self._maybe_rebalance(config_key, new_elo, elo_change)

    def _maybe_rebalance(self, config_key: str, new_elo: float, elo_change: float) -> bool:
        """Potentially trigger curriculum rebalance.

        Respects cooldown to prevent excessive rebalancing.

        Returns:
            True if rebalance was triggered
        """
        now = time.time()

        # Check cooldown
        if now - self._last_rebalance_time < self.rebalance_cooldown_seconds:
            logger.debug(
                f"[EloToCurriculumWatcher] Rebalance cooldown active, skipping"
            )
            return False

        # Update metrics in feedback
        metrics = self.feedback.get_config_metrics(config_key)
        metrics.avg_elo = new_elo

        # Calculate ELO trend from history
        history = self._elo_history.get(config_key, [])
        if len(history) >= 3:
            recent_trend = sum(history[-3:]) / 3 - sum(history[:3]) / min(3, len(history))
            metrics.elo_trend = recent_trend

        # Rebalance
        weights = self.feedback.get_curriculum_weights()
        self._last_rebalance_time = now

        logger.info(
            f"[EloToCurriculumWatcher] Rebalanced curriculum weights: "
            f"{len(weights)} configs, trigger: {config_key}"
        )

        # Auto-export if enabled
        if self.auto_export:
            try:
                self.feedback.export_weights_json(self.export_path)
            except Exception as e:
                logger.warning(f"Failed to export weights: {e}")

        # Publish rebalance event
        self._publish_rebalance_event(weights, config_key, elo_change)

        return True

    def _publish_rebalance_event(
        self,
        weights: Dict[str, float],
        trigger_config: str,
        elo_change: float,
    ) -> None:
        """Publish curriculum rebalance event."""
        try:
            from app.distributed.data_events import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            event = DataEvent(
                event_type=DataEventType.CURRICULUM_REBALANCED,
                payload={
                    "weights": weights,
                    "trigger_config": trigger_config,
                    "trigger_elo_change": elo_change,
                    "timestamp": time.time(),
                },
                source="curriculum_feedback",
            )

            bus = get_event_bus()
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(bus.publish(event))
            except RuntimeError:
                if hasattr(bus, 'publish_sync'):
                    bus.publish_sync(event)

        except Exception as e:
            logger.debug(f"Failed to publish rebalance event: {e}")

    def force_rebalance(self) -> Dict[str, float]:
        """Force an immediate curriculum rebalance.

        Returns:
            New curriculum weights
        """
        self._last_rebalance_time = 0  # Reset cooldown
        weights = self.feedback.get_curriculum_weights()

        if self.auto_export:
            try:
                self.feedback.export_weights_json(self.export_path)
            except Exception as e:
                logger.warning(f"Failed to export weights: {e}")

        self._publish_rebalance_event(weights, "manual", 0.0)
        return weights


# Singleton watcher
_elo_watcher: Optional[EloToCurriculumWatcher] = None


def wire_elo_to_curriculum(
    significant_elo_change: float = 30.0,
    auto_export: bool = True,
) -> EloToCurriculumWatcher:
    """Wire ELO change events to curriculum rebalancing.

    This is the main entry point for connecting ELO updates to automatic
    curriculum weight adjustments.

    Args:
        significant_elo_change: ELO change threshold to trigger rebalance
        auto_export: Whether to auto-export weights after rebalance

    Returns:
        EloToCurriculumWatcher instance
    """
    global _elo_watcher

    _elo_watcher = EloToCurriculumWatcher(
        significant_elo_change=significant_elo_change,
        auto_export=auto_export,
    )
    _elo_watcher.subscribe_to_elo_events()

    logger.info(
        f"[wire_elo_to_curriculum] ELO events wired to curriculum rebalance "
        f"(threshold={significant_elo_change})"
    )

    return _elo_watcher


def get_elo_curriculum_watcher() -> Optional[EloToCurriculumWatcher]:
    """Get the global ELO-to-curriculum watcher if configured."""
    return _elo_watcher


# =============================================================================
# PLATEAU_DETECTED → Curriculum Rebalance Integration (December 2025)
# =============================================================================

class PlateauToCurriculumWatcher:
    """Watches for training plateaus and triggers curriculum rebalancing.

    When a plateau is detected (training progress stalls), this watcher
    triggers a curriculum rebalance to potentially shift training focus
    to other configs or adjust weights to break through the plateau.

    Usage:
        from app.training.curriculum_feedback import wire_plateau_to_curriculum

        # Wire plateau events to curriculum
        watcher = wire_plateau_to_curriculum()
    """

    def __init__(
        self,
        feedback: Optional[CurriculumFeedback] = None,
        rebalance_cooldown_seconds: float = 600.0,
        auto_export: bool = True,
        export_path: str = "data/curriculum_weights.json",
        plateau_weight_boost: float = 0.3,
    ):
        """Initialize the plateau-to-curriculum watcher.

        Args:
            feedback: CurriculumFeedback instance (uses singleton if None)
            rebalance_cooldown_seconds: Minimum time between rebalances
            auto_export: Whether to auto-export weights after rebalance
            export_path: Path to export weights JSON
            plateau_weight_boost: Extra weight to add for plateaued configs
        """
        self.feedback = feedback or get_curriculum_feedback()
        self.rebalance_cooldown_seconds = rebalance_cooldown_seconds
        self.auto_export = auto_export
        self.export_path = export_path
        self.plateau_weight_boost = plateau_weight_boost

        self._last_rebalance_time: float = 0.0
        self._plateau_configs: Dict[str, float] = {}  # config -> plateau_time
        self._subscribed = False

    def subscribe_to_plateau_events(self) -> bool:
        """Subscribe to PLATEAU_DETECTED events from the event bus.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.subscribe(DataEventType.PLATEAU_DETECTED, self._on_plateau_detected)
            self._subscribed = True
            logger.info("[PlateauToCurriculumWatcher] Subscribed to PLATEAU_DETECTED events")
            return True
        except Exception as e:
            logger.warning(f"[PlateauToCurriculumWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from plateau events."""
        if not self._subscribed:
            return

        try:
            from app.distributed.data_events import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.PLATEAU_DETECTED, self._on_plateau_detected)
            self._subscribed = False
        except Exception:
            pass

    def _on_plateau_detected(self, event: Any) -> None:
        """Handle PLATEAU_DETECTED event.

        Triggers curriculum rebalance with boosted weight for plateaued config.
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config", "")
        current_elo = payload.get("current_elo", 0)
        plateau_duration_games = payload.get("plateau_duration_games", 0)
        plateau_duration_seconds = payload.get("plateau_duration_seconds", 0)

        if not config_key:
            return

        logger.info(
            f"[PlateauToCurriculumWatcher] Plateau detected for {config_key}: "
            f"{plateau_duration_games} games, {plateau_duration_seconds:.0f}s"
        )

        # Track plateau
        self._plateau_configs[config_key] = time.time()

        # Trigger rebalance
        self._maybe_rebalance(config_key, current_elo, plateau_duration_games)

    def _maybe_rebalance(
        self,
        config_key: str,
        current_elo: float,
        plateau_duration_games: int,
    ) -> bool:
        """Potentially trigger curriculum rebalance.

        Returns:
            True if rebalance was triggered
        """
        now = time.time()

        # Check cooldown
        if now - self._last_rebalance_time < self.rebalance_cooldown_seconds:
            logger.debug("[PlateauToCurriculumWatcher] Rebalance cooldown active, skipping")
            return False

        # Update metrics - boost the plateaued config's priority
        metrics = self.feedback.get_config_metrics(config_key)
        metrics.avg_elo = current_elo
        metrics.elo_trend = -10.0  # Mark as stagnant/declining

        # Get weights and apply plateau boost
        weights = self.feedback.get_curriculum_weights()

        # Boost weight for plateaued config (to try different training emphasis)
        if config_key in weights:
            boosted = min(
                self.feedback.weight_max,
                weights[config_key] + self.plateau_weight_boost
            )
            weights[config_key] = round(boosted, 3)

        self._last_rebalance_time = now

        logger.info(
            f"[PlateauToCurriculumWatcher] Rebalanced curriculum (plateau trigger): "
            f"{config_key} weight boosted to {weights.get(config_key, 1.0)}"
        )

        # Auto-export if enabled
        if self.auto_export:
            try:
                self.feedback.export_weights_json(self.export_path)
            except Exception as e:
                logger.warning(f"Failed to export weights: {e}")

        # Publish rebalance event
        self._publish_rebalance_event(weights, config_key, plateau_duration_games)

        return True

    def _publish_rebalance_event(
        self,
        weights: Dict[str, float],
        trigger_config: str,
        plateau_games: int,
    ) -> None:
        """Publish curriculum rebalance event."""
        try:
            from app.distributed.data_events import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            event = DataEvent(
                event_type=DataEventType.CURRICULUM_REBALANCED,
                payload={
                    "weights": weights,
                    "trigger_config": trigger_config,
                    "trigger_reason": "plateau_detected",
                    "plateau_duration_games": plateau_games,
                    "timestamp": time.time(),
                },
                source="curriculum_feedback_plateau",
            )

            bus = get_event_bus()
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(bus.publish(event))
            except RuntimeError:
                if hasattr(bus, 'publish_sync'):
                    bus.publish_sync(event)

        except Exception as e:
            logger.debug(f"Failed to publish rebalance event: {e}")

    def get_plateau_configs(self) -> Dict[str, float]:
        """Get configs currently in plateau state.

        Returns:
            Dict mapping config_key to plateau detection time
        """
        return dict(self._plateau_configs)

    def clear_plateau(self, config_key: str) -> bool:
        """Clear plateau status for a config (e.g., after breakthrough).

        Returns:
            True if config was in plateau state
        """
        if config_key in self._plateau_configs:
            del self._plateau_configs[config_key]
            logger.info(f"[PlateauToCurriculumWatcher] Cleared plateau for {config_key}")
            return True
        return False


# Singleton plateau watcher
_plateau_watcher: Optional[PlateauToCurriculumWatcher] = None


def wire_plateau_to_curriculum(
    rebalance_cooldown_seconds: float = 600.0,
    auto_export: bool = True,
    plateau_weight_boost: float = 0.3,
) -> PlateauToCurriculumWatcher:
    """Wire PLATEAU_DETECTED events to curriculum rebalancing.

    This connects plateau detection to automatic curriculum weight adjustments,
    boosting training priority for configs that have stalled.

    Args:
        rebalance_cooldown_seconds: Minimum time between rebalances
        auto_export: Whether to auto-export weights after rebalance
        plateau_weight_boost: Extra weight to add for plateaued configs

    Returns:
        PlateauToCurriculumWatcher instance
    """
    global _plateau_watcher

    _plateau_watcher = PlateauToCurriculumWatcher(
        rebalance_cooldown_seconds=rebalance_cooldown_seconds,
        auto_export=auto_export,
        plateau_weight_boost=plateau_weight_boost,
    )
    _plateau_watcher.subscribe_to_plateau_events()

    logger.info(
        f"[wire_plateau_to_curriculum] PLATEAU_DETECTED events wired to curriculum rebalance "
        f"(cooldown={rebalance_cooldown_seconds}s, boost={plateau_weight_boost})"
    )

    return _plateau_watcher


def get_plateau_curriculum_watcher() -> Optional[PlateauToCurriculumWatcher]:
    """Get the global plateau-to-curriculum watcher if configured."""
    return _plateau_watcher
