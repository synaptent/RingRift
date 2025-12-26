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
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
        opponent_tracker: Any | None = None,
    ):
        self.lookback_minutes = lookback_minutes
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.target_win_rate = target_win_rate

        # OpponentWinRateTracker for weak opponent detection (Dec 2025)
        self._opponent_tracker = opponent_tracker

        # Game history (circular buffer per config)
        self._game_history: dict[str, list[GameRecord]] = defaultdict(list)
        self._max_history_per_config = 1000

        # Cached metrics per config
        self._config_metrics: dict[str, ConfigMetrics] = {}

        # Current curriculum weights per config (December 2025)
        # Updated by record_promotion() and get_curriculum_weights()
        self._current_weights: dict[str, float] = {}

        # Curriculum stage tracking per config (December 2025 Phase 5)
        # Updated by _on_curriculum_advanced() when CURRICULUM_ADVANCED events received
        self._curriculum_stages: dict[str, int] = {}

        # Last update time for change detection
        self._last_update_time: float = 0

        # Auto-wire to CURRICULUM_ADVANCED events (Phase 5)
        self._curriculum_advanced_subscribed = False
        self._auto_wire_curriculum_advanced()

    def set_opponent_tracker(self, tracker: Any) -> None:
        """Set the opponent win rate tracker for weak opponent detection.

        Args:
            tracker: OpponentWinRateTracker instance from app.integration.pipeline_feedback
        """
        self._opponent_tracker = tracker
        logger.info("CurriculumFeedback: OpponentWinRateTracker integrated")

    def _auto_wire_curriculum_advanced(self) -> None:
        """Auto-wire to CURRICULUM_ADVANCED events for closed-loop feedback.

        Phase 5 (December 2025): Subscribe to CURRICULUM_ADVANCED events
        emitted by GauntletFeedbackController to synchronize curriculum
        stage tracking and weight adjustments.
        """
        if self._curriculum_advanced_subscribed:
            return

        try:
            from app.coordination.event_router import get_router
            from app.distributed.data_events import DataEventType

            router = get_router()
            if router is None:
                logger.debug("[CurriculumFeedback] Event router not available, deferring subscription")
                return

            router.subscribe(DataEventType.CURRICULUM_ADVANCED.value, self._on_curriculum_advanced)
            router.subscribe(DataEventType.TRAINING_EARLY_STOPPED.value, self._on_training_early_stopped)
            self._curriculum_advanced_subscribed = True
            logger.info("[CurriculumFeedback] Subscribed to CURRICULUM_ADVANCED + TRAINING_EARLY_STOPPED events")
        except Exception as e:
            logger.debug(f"[CurriculumFeedback] Event subscription deferred: {e}")

    def _on_curriculum_advanced(self, event) -> None:
        """Handle CURRICULUM_ADVANCED events from GauntletFeedbackController.

        When curriculum advances (model is performing well), this:
        1. Tracks the new curriculum stage
        2. Adjusts weights to maintain training focus
        3. Emits CURRICULUM_REBALANCED for downstream listeners

        Phase 5 (December 2025): Close the feedback loop from gauntlet to curriculum.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config", payload.get("config_key", ""))
            new_stage = payload.get("new_stage", payload.get("stage", 0))
            reason = payload.get("reason", "gauntlet_feedback")

            if not config_key:
                return

            # Update internal stage tracking
            old_stage = self._curriculum_stages.get(config_key, 0)
            self._curriculum_stages[config_key] = new_stage

            logger.info(
                f"[CurriculumFeedback] Curriculum advanced for {config_key}: "
                f"stage {old_stage} → {new_stage} (reason: {reason})"
            )

            # Adjust weight based on curriculum advancement
            # Higher stages indicate stronger model, slightly reduce training priority
            # but maintain enough weight to continue challenging the model
            current_weight = self._current_weights.get(config_key, 1.0)
            stage_adjustment = -0.05 * (new_stage - old_stage)  # Reduce by 5% per stage
            new_weight = max(self.weight_min, min(self.weight_max, current_weight + stage_adjustment))

            if abs(new_weight - current_weight) > 0.01:
                self._current_weights[config_key] = new_weight
                logger.info(
                    f"[CurriculumFeedback] Weight adjusted for {config_key}: "
                    f"{current_weight:.2f} → {new_weight:.2f} (curriculum stage {new_stage})"
                )

                # Emit curriculum rebalanced event
                self._emit_curriculum_updated(config_key, new_weight, f"curriculum_advanced_stage_{new_stage}")

            self._last_update_time = time.time()

        except Exception as e:
            logger.warning(f"[CurriculumFeedback] Failed to handle CURRICULUM_ADVANCED: {e}")

    def _on_training_early_stopped(self, event) -> None:
        """Handle TRAINING_EARLY_STOPPED - boost curriculum weight for stalled configs.

        December 2025: Closes the training → curriculum feedback loop.
        When training early stops (stagnation or regression), this handler
        boosts the curriculum weight for that config to prioritize more
        training data generation.

        This helps recover from training plateaus by:
        1. Increasing selfplay priority for the config
        2. Triggering more diverse exploration
        """
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key", payload.get("config", ""))
            reason = payload.get("reason", "unknown")
            epoch = payload.get("epoch", 0)
            final_loss = payload.get("final_loss", 0.0)

            if not config_key:
                return

            # Boost curriculum weight when training stalls
            current_weight = self._current_weights.get(config_key, 1.0)

            # Boost weight by 30% for early stopping (helps generate more diverse data)
            boost_factor = 1.3
            new_weight = min(self.weight_max, current_weight * boost_factor)

            if new_weight > current_weight:
                self._current_weights[config_key] = new_weight
                logger.info(
                    f"[CurriculumFeedback] Boosted weight for {config_key}: "
                    f"{current_weight:.2f} → {new_weight:.2f} (training early stopped at epoch {epoch}, reason: {reason})"
                )

                # Emit curriculum rebalanced event
                self._emit_curriculum_updated(config_key, new_weight, f"early_stopped_{reason}")

            self._last_update_time = time.time()

        except Exception as e:
            logger.warning(f"[CurriculumFeedback] Failed to handle TRAINING_EARLY_STOPPED: {e}")

    def advance_stage(self, config_key: str, new_stage: int) -> None:
        """Manually advance curriculum stage for a config.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")
            new_stage: New curriculum stage number
        """
        old_stage = self._curriculum_stages.get(config_key, 0)
        self._curriculum_stages[config_key] = new_stage

        if new_stage != old_stage:
            logger.info(f"[CurriculumFeedback] Manual stage advance: {config_key} {old_stage} → {new_stage}")
            self._last_update_time = time.time()

    def get_curriculum_stage(self, config_key: str) -> int:
        """Get the current curriculum stage for a config.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")

        Returns:
            Current curriculum stage (0 = initial, higher = more advanced)
        """
        return self._curriculum_stages.get(config_key, 0)

    def get_all_curriculum_stages(self) -> dict[str, int]:
        """Get curriculum stages for all configs.

        Returns:
            Dict mapping config_key → curriculum stage
        """
        return dict(self._curriculum_stages)

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

    def record_promotion(
        self,
        config_key: str,
        promoted: bool,
        new_elo: float | None = None,
        promotion_reason: str = "",
    ) -> None:
        """Record a model promotion result and adjust curriculum weights.

        When a model is promoted (or fails promotion), this adjusts the
        curriculum weights to reallocate training resources:
        - Successful promotion: Reduce weight (model is strong enough)
        - Failed promotion: Increase weight (needs more training)

        Args:
            config_key: Config identifier (e.g., "square8_2p")
            promoted: Whether the model was promoted
            new_elo: New Elo rating after promotion
            promotion_reason: Reason for promotion decision

        December 2025: Added to close the feedback loop from evaluation
        to curriculum weights.
        """
        metrics = self._get_or_create_metrics(config_key)
        metrics.model_count += 1 if promoted else 0

        if new_elo is not None:
            # Update Elo tracking
            old_elo = metrics.avg_elo
            metrics.avg_elo = new_elo
            metrics.elo_trend = new_elo - old_elo

        # Adjust weight based on promotion result
        current_weight = self._current_weights.get(config_key, 1.0)

        if promoted:
            # Successful promotion: Model is performing well
            # Reduce weight slightly to reallocate resources to weaker configs
            adjustment = -0.1
            logger.info(
                f"CurriculumFeedback: {config_key} promoted, reducing weight "
                f"({current_weight:.2f} -> {max(self.weight_min, current_weight + adjustment):.2f})"
            )
        else:
            # Failed promotion: Model needs more training
            # Increase weight to allocate more resources
            adjustment = 0.15
            logger.info(
                f"CurriculumFeedback: {config_key} failed promotion, increasing weight "
                f"({current_weight:.2f} -> {min(self.weight_max, current_weight + adjustment):.2f})"
            )

        new_weight = max(self.weight_min, min(self.weight_max, current_weight + adjustment))
        self._current_weights[config_key] = new_weight
        self._last_update_time = time.time()

        # Emit curriculum update event (December 2025)
        self._emit_curriculum_updated(config_key, new_weight, "promotion")

    def _emit_curriculum_updated(
        self,
        config_key: str,
        new_weight: float,
        trigger: str,
    ) -> None:
        """Emit CURRICULUM_REBALANCED event for downstream listeners."""
        try:
            from app.coordination.event_router import DataEvent, DataEventType, get_event_bus

            event = DataEvent(
                event_type=DataEventType.CURRICULUM_REBALANCED,
                payload={
                    "config_key": config_key,
                    "new_weight": new_weight,
                    "trigger": trigger,
                    "timestamp": time.time(),
                    "all_weights": dict(self._current_weights),
                },
                source="curriculum_feedback",
            )

            import asyncio
            bus = get_event_bus()
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(bus.publish(event))
            except RuntimeError:
                asyncio.run(bus.publish(event))

            logger.debug(f"CurriculumFeedback: Emitted CURRICULUM_REBALANCED for {config_key}")
        except Exception as e:
            logger.warning(f"CurriculumFeedback: Failed to emit event: {e}")

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

    def get_all_metrics(self) -> dict[str, ConfigMetrics]:
        """Get metrics for all configs."""
        result = {}
        for config_key in set(self._game_history.keys()) | set(self._config_metrics.keys()):
            result[config_key] = self._compute_metrics(config_key)
        return result

    def get_curriculum_weights(self) -> dict[str, float]:
        """Compute curriculum weights based on current metrics.

        Weighting strategy:
        - Low win rate (< target) → Higher weight (more training needed)
        - High win rate (> target) → Lower weight (already strong)
        - Few models → Higher weight (bootstrap priority)
        - Declining Elo → Higher weight (regression detected)
        - Weak opponent performance → Higher weight (+50-150 Elo impact, Dec 2025)

        Returns:
            Dict mapping config_key → weight (0.5 to 2.0)
        """
        all_metrics = self.get_all_metrics()

        if not all_metrics:
            return {}

        weights = {}

        # Get weak opponent info if tracker is available
        weak_opponent_boost = self._compute_weak_opponent_boosts()

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

            # Weak opponent adjustment (Dec 2025)
            # Boost training for configs where we struggle against specific opponents
            if config_key in weak_opponent_boost:
                weight *= weak_opponent_boost[config_key]

            # Clamp to bounds
            weight = max(self.weight_min, min(self.weight_max, weight))
            weights[config_key] = round(weight, 3)

        return weights

    def _compute_weak_opponent_boosts(self) -> dict[str, float]:
        """Compute training weight boosts based on weak opponent performance.

        Uses OpponentWinRateTracker to identify opponents where model struggles
        and boosts training weight accordingly.

        Returns:
            Dict mapping config_key → boost multiplier (1.0 = no boost, 1.3 = 30% boost)
        """
        if self._opponent_tracker is None:
            return {}

        boosts = {}
        try:
            # Get all models being tracked
            # The opponent tracker tracks by model_id, we need to map to config_key
            # Convention: model_id format is "{config_key}_model_{version}"
            for config_key in set(self._game_history.keys()) | set(self._config_metrics.keys()):
                # Look for weak opponents for models in this config
                # Try common model naming patterns
                model_patterns = [
                    config_key,
                    f"{config_key}_best",
                    f"{config_key}_latest",
                ]

                weak_count = 0
                total_weakness = 0.0

                for model_id in model_patterns:
                    weak_opponents = self._opponent_tracker.get_weak_opponents(model_id)
                    for opponent_id, win_rate in weak_opponents:
                        weak_count += 1
                        # More weakness (lower win rate) = higher boost
                        total_weakness += (0.45 - win_rate)

                if weak_count > 0:
                    # Scale boost: 1 weak opponent at 25% win rate → 1.2x boost
                    # Multiple weak opponents → up to 1.4x boost
                    avg_weakness = total_weakness / weak_count
                    boost = 1.0 + min(0.4, avg_weakness * 2.0 + weak_count * 0.05)
                    boosts[config_key] = boost
                    logger.debug(
                        f"CurriculumFeedback: {config_key} has {weak_count} weak opponents, "
                        f"boost={boost:.2f}"
                    )

        except Exception as e:
            logger.warning(f"Error computing weak opponent boosts: {e}")

        return boosts

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

    def detect_stuck_configs(
        self,
        min_plateau_hours: int = 48,
        elo_change_threshold: float = 5.0,
        min_models: int = 3,
    ) -> dict[str, dict]:
        """Detect configs that are permanently stuck (Elo plateau).

        December 2025: Added for Phase 4 weak config detection.
        Identifies configs that have been stuck for the specified time period
        and should have their resource allocation reduced.

        Args:
            min_plateau_hours: Minimum hours of plateau to be considered stuck
            elo_change_threshold: Max Elo change to be considered plateau
            min_models: Minimum model count to consider (bootstrapping configs exempt)

        Returns:
            Dict mapping config_key to stuck info:
            {
                "is_stuck": True,
                "stuck_hours": 52.3,
                "elo_slope": 0.3,
                "total_elo_change": 2.1,
                "model_count": 5,
                "current_weight": 0.8,
                "recommended_weight": 0.5,
            }
        """
        stuck_configs = {}

        try:
            from app.training.elo_service import get_elo_trend_for_config
        except ImportError:
            logger.warning("Cannot detect stuck configs: elo_service not available")
            return stuck_configs

        all_metrics = self.get_all_metrics()

        for config_key, metrics in all_metrics.items():
            # Skip configs still bootstrapping
            if metrics.model_count < min_models:
                continue

            # Get Elo trend for this config
            trend = get_elo_trend_for_config(config_key, hours=min_plateau_hours)

            if trend.get("error"):
                continue

            # Check if stuck (plateau with enough data)
            is_plateau = trend.get("is_plateau", False)
            duration_hours = trend.get("duration_hours", 0)
            total_change = abs(trend.get("total_change", 0))
            confidence = trend.get("confidence", 0)

            # A config is stuck if:
            # 1. It's in plateau state
            # 2. Has been tracked for min_plateau_hours
            # 3. Total Elo change is below threshold
            # 4. Confidence is reasonable
            is_stuck = (
                is_plateau
                and duration_hours >= min_plateau_hours * 0.8  # Allow 20% slack
                and total_change <= elo_change_threshold
                and confidence >= 0.3
            )

            if is_stuck:
                current_weight = self._current_weights.get(config_key, 1.0)
                # Recommend reducing to minimum weight
                recommended_weight = max(self.weight_min, current_weight * 0.5)

                stuck_configs[config_key] = {
                    "is_stuck": True,
                    "stuck_hours": duration_hours,
                    "elo_slope": trend.get("slope", 0),
                    "total_elo_change": trend.get("total_change", 0),
                    "start_elo": trend.get("start_elo", 0),
                    "end_elo": trend.get("end_elo", 0),
                    "model_count": metrics.model_count,
                    "current_weight": current_weight,
                    "recommended_weight": recommended_weight,
                    "confidence": confidence,
                }

                logger.info(
                    f"[CurriculumFeedback] Stuck config detected: {config_key} "
                    f"(stuck {duration_hours:.1f}h, Elo change {total_change:.1f})"
                )

        return stuck_configs

    def get_stuck_config_keys(
        self,
        min_plateau_hours: int = 48,
    ) -> set[str]:
        """Get set of config keys that are currently stuck.

        December 2025: Convenience method for selfplay_orchestrator filtering.

        Returns:
            Set of config keys identified as stuck
        """
        stuck = self.detect_stuck_configs(min_plateau_hours=min_plateau_hours)
        return set(stuck.keys())

    def apply_stuck_config_weights(
        self,
        min_plateau_hours: int = 48,
    ) -> dict[str, float]:
        """Detect stuck configs and apply reduced weights.

        December 2025: Proactive stuck config handling.
        Detects stuck configs and updates their weights to minimum.

        Returns:
            Dict of config_key -> new_weight for modified configs
        """
        stuck_configs = self.detect_stuck_configs(min_plateau_hours=min_plateau_hours)
        modified = {}

        for config_key, info in stuck_configs.items():
            old_weight = self._current_weights.get(config_key, 1.0)
            new_weight = info["recommended_weight"]

            if new_weight < old_weight:
                self._current_weights[config_key] = new_weight
                modified[config_key] = new_weight

                logger.info(
                    f"[CurriculumFeedback] Reduced weight for stuck config "
                    f"{config_key}: {old_weight:.2f} → {new_weight:.2f}"
                )

                # Emit curriculum update
                self._emit_curriculum_updated(config_key, new_weight, "stuck_config")

        if modified:
            self._last_update_time = time.time()

        return modified


# Singleton instance (thread-safe)
_feedback_instance: CurriculumFeedback | None = None
_feedback_lock = threading.Lock()


def get_curriculum_feedback() -> CurriculumFeedback:
    """Get the global curriculum feedback instance (thread-safe)."""
    global _feedback_instance
    if _feedback_instance is None:
        with _feedback_lock:
            # Double-check locking pattern
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


def get_curriculum_weights() -> dict[str, float]:
    """Get current curriculum weights (convenience function)."""
    return get_curriculum_feedback().get_curriculum_weights()


def sync_curriculum_weights_to_p2p() -> bool:
    """Sync curriculum weights to JSON file for P2P orchestrator consumption.

    This exports the current curriculum weights to the standard path
    (data/curriculum_weights.json) that the P2P orchestrator reads for
    selfplay job prioritization.

    Returns:
        True if export succeeded, False otherwise

    Usage:
        from app.training.curriculum_feedback import sync_curriculum_weights_to_p2p
        sync_curriculum_weights_to_p2p()
    """
    try:
        from app.utils.paths import AI_SERVICE_ROOT

        output_path = AI_SERVICE_ROOT / "data" / "curriculum_weights.json"
        feedback = get_curriculum_feedback()
        weights = feedback.get_curriculum_weights()

        # Write in format expected by p2p_orchestrator
        import time
        data = {
            "weights": weights,
            "updated_at": time.time(),
            "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "curriculum_feedback",
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(output_path)

        logger.debug(f"Synced curriculum weights to {output_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to sync curriculum weights: {e}")
        return False


def wire_opponent_tracker_to_curriculum() -> None:
    """Wire OpponentWinRateTracker to CurriculumFeedback for weak opponent detection.

    This integrates opponent-specific performance tracking into curriculum weights.
    Configs where the model struggles against specific opponents will get training
    priority boosts of up to 40%.

    Expected Elo impact: +50-150 Elo through targeted weakness exploitation.

    Usage:
        from app.training.curriculum_feedback import wire_opponent_tracker_to_curriculum
        wire_opponent_tracker_to_curriculum()

        # Now get_curriculum_weights() includes weak opponent boosts
        weights = get_curriculum_weights()
    """
    try:
        from app.integration.pipeline_feedback import create_opponent_tracker
        tracker = create_opponent_tracker(min_games=10, weak_threshold=0.45)
        get_curriculum_feedback().set_opponent_tracker(tracker)
        logger.info("Wired OpponentWinRateTracker to CurriculumFeedback")
    except ImportError as e:
        logger.warning(f"Could not wire opponent tracker: {e}")


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
        feedback: CurriculumFeedback | None = None,
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
        self._elo_history: dict[str, list[float]] = defaultdict(list)
        self._subscribed = False

    def subscribe_to_elo_events(self) -> bool:
        """Subscribe to ELO update events from the event bus.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.coordination.event_router import (
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
            from app.coordination.event_router import (
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

        config_key = payload.get("config_key") or payload.get("config", "")
        new_elo = payload.get("new_elo")
        if new_elo is None:
            new_elo = payload.get("elo")
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
                "[EloToCurriculumWatcher] Rebalance cooldown active, skipping"
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
        weights: dict[str, float],
        trigger_config: str,
        elo_change: float,
    ) -> None:
        """Publish curriculum rebalance event via centralized emitters.

        December 2025: Migrated to use event_emitters.py for unified routing.
        """
        try:
            import asyncio
            from app.coordination.event_emitters import emit_curriculum_rebalanced

            # Get old weights for comparison (empty dict if not available)
            old_weights: dict[str, float] = {}

            try:
                asyncio.get_running_loop()
                asyncio.create_task(
                    emit_curriculum_rebalanced(
                        config=trigger_config,
                        old_weights=old_weights,
                        new_weights=weights,
                        reason=f"elo_change_{elo_change:.0f}",
                        trigger="elo_change",
                        elo_change=elo_change,
                    )
                )
            except RuntimeError:
                asyncio.run(
                    emit_curriculum_rebalanced(
                        config=trigger_config,
                        old_weights=old_weights,
                        new_weights=weights,
                        reason=f"elo_change_{elo_change:.0f}",
                        trigger="elo_change",
                        elo_change=elo_change,
                    )
                )
            logger.debug("Emitted CURRICULUM_REBALANCED via centralized emitters")
        except ImportError:
            logger.debug("Centralized event emitters not available")
        except Exception as e:
            logger.debug(f"Failed to publish rebalance event: {e}")

    def force_rebalance(self) -> dict[str, float]:
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
_elo_watcher: EloToCurriculumWatcher | None = None


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


def get_elo_curriculum_watcher() -> EloToCurriculumWatcher | None:
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
        feedback: CurriculumFeedback | None = None,
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
        self._plateau_configs: dict[str, float] = {}  # config -> plateau_time
        self._subscribed = False

    def subscribe_to_plateau_events(self) -> bool:
        """Subscribe to PLATEAU_DETECTED events from the event bus.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.coordination.event_router import (
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
            from app.coordination.event_router import (
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
        weights: dict[str, float],
        trigger_config: str,
        plateau_games: int,
    ) -> None:
        """Publish curriculum rebalance event via centralized emitters.

        December 2025: Migrated to use event_emitters.py for unified routing.
        """
        try:
            import asyncio
            from app.coordination.event_emitters import emit_curriculum_rebalanced

            # Get old weights for comparison (empty dict if not available)
            old_weights: dict[str, float] = {}

            try:
                asyncio.get_running_loop()
                asyncio.create_task(
                    emit_curriculum_rebalanced(
                        config=trigger_config,
                        old_weights=old_weights,
                        new_weights=weights,
                        reason="plateau_detected",
                        trigger="plateau",
                        plateau_duration_games=plateau_games,
                    )
                )
            except RuntimeError:
                asyncio.run(
                    emit_curriculum_rebalanced(
                        config=trigger_config,
                        old_weights=old_weights,
                        new_weights=weights,
                        reason="plateau_detected",
                        trigger="plateau",
                        plateau_duration_games=plateau_games,
                    )
                )
            logger.debug("Emitted CURRICULUM_REBALANCED (plateau) via centralized emitters")
        except ImportError:
            logger.debug("Centralized event emitters not available")
        except Exception as e:
            logger.debug(f"Failed to publish rebalance event: {e}")

    def get_plateau_configs(self) -> dict[str, float]:
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
_plateau_watcher: PlateauToCurriculumWatcher | None = None


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


def get_plateau_curriculum_watcher() -> PlateauToCurriculumWatcher | None:
    """Get the global plateau-to-curriculum watcher if configured."""
    return _plateau_watcher


# =============================================================================
# Tournament Results → Curriculum Feedback Integration (December 2025)
# =============================================================================

class TournamentToCurriculumWatcher:
    """Watches for tournament/evaluation results and adjusts curriculum weights.

    Subscribes to EVALUATION_COMPLETED events and uses tournament results
    (win rates, Elo gains) to influence curriculum weight adjustments:
    - High-performing configs may get reduced weight (less training needed)
    - Low-performing configs get increased weight (more training needed)

    Usage:
        from app.training.curriculum_feedback import wire_tournament_to_curriculum

        # Wire tournament events to curriculum
        watcher = wire_tournament_to_curriculum()
    """

    def __init__(
        self,
        feedback: CurriculumFeedback | None = None,
        rebalance_cooldown_seconds: float = 300.0,
        auto_export: bool = True,
        export_path: str = "data/curriculum_weights.json",
        win_rate_threshold_low: float = 0.45,
        win_rate_threshold_high: float = 0.65,
        weight_adjustment: float = 0.15,
    ):
        """Initialize the tournament-to-curriculum watcher.

        Args:
            feedback: CurriculumFeedback instance (uses singleton if None)
            rebalance_cooldown_seconds: Minimum time between rebalances
            auto_export: Whether to auto-export weights after rebalance
            export_path: Path to export weights JSON
            win_rate_threshold_low: Below this, boost weight
            win_rate_threshold_high: Above this, reduce weight
            weight_adjustment: Amount to adjust weight by
        """
        self.feedback = feedback or get_curriculum_feedback()
        self.rebalance_cooldown_seconds = rebalance_cooldown_seconds
        self.auto_export = auto_export
        self.export_path = export_path
        self.win_rate_threshold_low = win_rate_threshold_low
        self.win_rate_threshold_high = win_rate_threshold_high
        self.weight_adjustment = weight_adjustment

        self._last_rebalance_time: float = 0.0
        self._tournament_results: dict[str, list[dict]] = defaultdict(list)
        self._subscribed = False
        self._adjustment_count = 0

    def subscribe_to_tournament_events(self) -> bool:
        """Subscribe to tournament/evaluation events from the event bus.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.coordination.event_router import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.subscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_completed)
            self._subscribed = True
            logger.info("[TournamentToCurriculumWatcher] Subscribed to EVALUATION_COMPLETED events")
            return True
        except Exception as e:
            logger.warning(f"[TournamentToCurriculumWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from tournament events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.EVALUATION_COMPLETED, self._on_evaluation_completed)
            self._subscribed = False
        except Exception:
            pass

    def _on_evaluation_completed(self, event: Any) -> None:
        """Handle EVALUATION_COMPLETED event.

        Records tournament results and potentially triggers curriculum rebalance.
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        config = payload.get("config", "")
        elo = payload.get("elo", 0)
        games_played = payload.get("games_played", 0)
        win_rate = payload.get("win_rate", 0.5)

        if not config or games_played < 5:
            return  # Need meaningful sample

        logger.debug(
            f"[TournamentToCurriculumWatcher] Evaluation completed for {config}: "
            f"elo={elo:.0f}, win_rate={win_rate:.2%}, games={games_played}"
        )

        # Track result
        result = {
            "elo": elo,
            "games": games_played,
            "win_rate": win_rate,
            "timestamp": time.time(),
        }
        self._tournament_results[config].append(result)

        # Keep only recent results
        if len(self._tournament_results[config]) > 10:
            self._tournament_results[config] = self._tournament_results[config][-10:]

        # Record game results in feedback
        wins = int(games_played * win_rate)
        losses = games_played - wins
        for _ in range(wins):
            self.feedback.record_game(config, winner=1, model_elo=elo, opponent_type="tournament")
        for _ in range(losses):
            self.feedback.record_game(config, winner=-1, model_elo=elo, opponent_type="tournament")

        # Check if curriculum adjustment needed
        self._maybe_adjust_curriculum(config, win_rate, elo)

    def _maybe_adjust_curriculum(
        self,
        config_key: str,
        win_rate: float,
        elo: float,
    ) -> bool:
        """Potentially adjust curriculum weights based on tournament results.

        Returns:
            True if adjustment was made
        """
        now = time.time()

        # Check cooldown
        if now - self._last_rebalance_time < self.rebalance_cooldown_seconds:
            return False

        adjustment = 0.0

        # Low win rate = boost training for this config
        if win_rate < self.win_rate_threshold_low:
            adjustment = self.weight_adjustment
            reason = f"low_win_rate_{win_rate:.2%}"
        # High win rate = reduce training (already performing well)
        elif win_rate > self.win_rate_threshold_high:
            adjustment = -self.weight_adjustment
            reason = f"high_win_rate_{win_rate:.2%}"
        else:
            return False  # No adjustment needed

        # Update metrics
        metrics = self.feedback.get_config_metrics(config_key)
        metrics.avg_elo = elo
        metrics.win_rate = win_rate

        # Get and adjust weights
        weights = self.feedback.get_curriculum_weights()

        if config_key in weights:
            new_weight = weights[config_key] + adjustment
            new_weight = max(self.feedback.weight_min, min(self.feedback.weight_max, new_weight))
            weights[config_key] = round(new_weight, 3)

            logger.info(
                f"[TournamentToCurriculumWatcher] Adjusted curriculum weight for {config_key}: "
                f"{adjustment:+.2f} → {new_weight:.2f} (reason: {reason})"
            )

        self._last_rebalance_time = now
        self._adjustment_count += 1

        # Auto-export if enabled
        if self.auto_export:
            try:
                self.feedback.export_weights_json(self.export_path)
            except Exception as e:
                logger.warning(f"Failed to export weights: {e}")

        # Publish rebalance event
        self._publish_rebalance_event(weights, config_key, win_rate, reason)

        return True

    def _publish_rebalance_event(
        self,
        weights: dict[str, float],
        trigger_config: str,
        win_rate: float,
        reason: str,
    ) -> None:
        """Publish curriculum rebalance event via centralized emitters.

        December 2025: Migrated to use event_emitters.py for unified routing.
        """
        try:
            import asyncio
            from app.coordination.event_emitters import emit_curriculum_rebalanced

            # Get old weights for comparison (empty dict if not available)
            old_weights: dict[str, float] = {}

            try:
                asyncio.get_running_loop()
                asyncio.create_task(
                    emit_curriculum_rebalanced(
                        config=trigger_config,
                        old_weights=old_weights,
                        new_weights=weights,
                        reason=f"tournament_{reason}",
                        trigger="tournament",
                        win_rate=win_rate,
                    )
                )
            except RuntimeError:
                asyncio.run(
                    emit_curriculum_rebalanced(
                        config=trigger_config,
                        old_weights=old_weights,
                        new_weights=weights,
                        reason=f"tournament_{reason}",
                        trigger="tournament",
                        win_rate=win_rate,
                    )
                )
            logger.debug("Emitted CURRICULUM_REBALANCED (tournament) via centralized emitters")
        except ImportError:
            logger.debug("Centralized event emitters not available")
        except Exception as e:
            logger.debug(f"Failed to publish rebalance event: {e}")

    def get_tournament_summary(self, config_key: str) -> dict[str, Any]:
        """Get tournament results summary for a config.

        Returns:
            Dict with tournament statistics
        """
        results = self._tournament_results.get(config_key, [])
        if not results:
            return {"config": config_key, "tournaments": 0}

        return {
            "config": config_key,
            "tournaments": len(results),
            "avg_win_rate": sum(r["win_rate"] for r in results) / len(results),
            "avg_elo": sum(r["elo"] for r in results) / len(results),
            "total_games": sum(r["games"] for r in results),
            "last_tournament": results[-1] if results else None,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get watcher statistics.

        Returns:
            Dict with statistics
        """
        return {
            "subscribed": self._subscribed,
            "adjustment_count": self._adjustment_count,
            "configs_tracked": len(self._tournament_results),
            "thresholds": {
                "low_win_rate": self.win_rate_threshold_low,
                "high_win_rate": self.win_rate_threshold_high,
                "weight_adjustment": self.weight_adjustment,
            },
        }


# Singleton tournament watcher
_tournament_watcher: TournamentToCurriculumWatcher | None = None


def wire_tournament_to_curriculum(
    rebalance_cooldown_seconds: float = 300.0,
    auto_export: bool = True,
    win_rate_threshold_low: float = 0.45,
    win_rate_threshold_high: float = 0.65,
    weight_adjustment: float = 0.15,
) -> TournamentToCurriculumWatcher:
    """Wire tournament/evaluation results to curriculum weight adjustments.

    This connects tournament results to automatic curriculum rebalancing:
    - Configs with low win rates get boosted training priority
    - Configs with high win rates get reduced training priority

    Args:
        rebalance_cooldown_seconds: Minimum time between rebalances
        auto_export: Whether to auto-export weights after rebalance
        win_rate_threshold_low: Below this win rate, boost weight
        win_rate_threshold_high: Above this win rate, reduce weight
        weight_adjustment: Amount to adjust weight by

    Returns:
        TournamentToCurriculumWatcher instance
    """
    global _tournament_watcher

    _tournament_watcher = TournamentToCurriculumWatcher(
        rebalance_cooldown_seconds=rebalance_cooldown_seconds,
        auto_export=auto_export,
        win_rate_threshold_low=win_rate_threshold_low,
        win_rate_threshold_high=win_rate_threshold_high,
        weight_adjustment=weight_adjustment,
    )
    _tournament_watcher.subscribe_to_tournament_events()

    logger.info(
        f"[wire_tournament_to_curriculum] EVALUATION_COMPLETED events wired to curriculum "
        f"(low={win_rate_threshold_low}, high={win_rate_threshold_high})"
    )

    return _tournament_watcher


def get_tournament_curriculum_watcher() -> TournamentToCurriculumWatcher | None:
    """Get the global tournament-to-curriculum watcher if configured."""
    return _tournament_watcher


# =============================================================================
# Promotion Outcome → Curriculum & Training Intensity Integration (December 2025)
# =============================================================================

class PromotionToCurriculumWatcher:
    """Watches for promotion outcomes and adjusts curriculum + training intensity.

    Subscribes to PROMOTION_COMPLETE events and:
    1. Updates curriculum weights based on promotion success/failure
    2. Signals FeedbackAccelerator with urgency when promotion fails

    This creates a tight feedback loop:
    - Successful promotion → reduce training weight (model is strong)
    - Failed promotion → boost training weight + signal high urgency training

    Usage:
        from app.training.curriculum_feedback import wire_promotion_to_curriculum

        # Wire promotion events
        watcher = wire_promotion_to_curriculum()
    """

    def __init__(
        self,
        feedback: CurriculumFeedback | None = None,
        auto_export: bool = True,
        export_path: str = "data/curriculum_weights.json",
        failure_urgency: str = "high",
    ):
        """Initialize the promotion-to-curriculum watcher.

        Args:
            feedback: CurriculumFeedback instance (uses singleton if None)
            auto_export: Whether to auto-export weights after promotion
            export_path: Path to export weights JSON
            failure_urgency: Training urgency level on promotion failure
        """
        self.feedback = feedback or get_curriculum_feedback()
        self.auto_export = auto_export
        self.export_path = export_path
        self.failure_urgency = failure_urgency

        self._promotion_history: dict[str, list[dict]] = defaultdict(list)
        self._subscribed = False
        self._promotions_processed = 0
        self._failures_processed = 0

    def subscribe_to_promotion_events(self) -> bool:
        """Subscribe to PROMOTION_COMPLETE events from the event bus.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.coordination.event_router import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.subscribe(DataEventType.PROMOTION_COMPLETE, self._on_promotion_complete)
            # P1.2 (Dec 2025): Also subscribe to MODEL_PROMOTED for complete coverage
            # MODEL_PROMOTED is emitted when a model is actually promoted to production
            if hasattr(DataEventType, 'MODEL_PROMOTED'):
                bus.subscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
            self._subscribed = True
            logger.info("[PromotionToCurriculumWatcher] Subscribed to PROMOTION_COMPLETE + MODEL_PROMOTED events")
            return True
        except Exception as e:
            logger.warning(f"[PromotionToCurriculumWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from promotion events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import (
                DataEventType,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.unsubscribe(DataEventType.PROMOTION_COMPLETE, self._on_promotion_complete)
            # P1.2: Unsubscribe from MODEL_PROMOTED
            if hasattr(DataEventType, 'MODEL_PROMOTED'):
                bus.unsubscribe(DataEventType.MODEL_PROMOTED, self._on_model_promoted)
            self._subscribed = False
        except Exception:
            pass

    def _on_model_promoted(self, event: Any) -> None:
        """Handle MODEL_PROMOTED event (P1.2).

        When a model is promoted to production, reset curriculum weights
        for that config to give other configs a chance.
        """
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config_key", "") or payload.get("config", "")
        new_elo = payload.get("new_elo", 0) or payload.get("elo", 0)

        if not config_key:
            return

        logger.info(
            f"[PromotionToCurriculumWatcher] MODEL_PROMOTED for {config_key}, "
            f"resetting curriculum weight to 1.0"
        )

        # Reset curriculum weight to baseline (1.0) after successful promotion
        # This ensures other configs get fair allocation
        self.feedback._config_metrics[config_key].model_count += 1
        old_weight = self.feedback._weights.get(config_key, 1.0)
        self.feedback._weights[config_key] = 1.0  # Reset to baseline

        # Emit curriculum update
        self.feedback._emit_curriculum_updated(config_key, 1.0, "model_promoted")

        logger.info(
            f"[PromotionToCurriculumWatcher] Reset {config_key} weight: {old_weight:.2f} → 1.0"
        )

    def _on_promotion_complete(self, event: Any) -> None:
        """Handle PROMOTION_COMPLETE event.

        Updates curriculum weights and signals training intensity.
        """
        payload = event.payload if hasattr(event, 'payload') else {}
        metadata = payload.get("metadata", payload)

        # Extract promotion details
        config_key = metadata.get("config") or metadata.get("config_key", "")
        board_type = metadata.get("board_type", "")
        num_players = metadata.get("num_players", 2)
        promoted = metadata.get("promoted", False)
        new_elo = metadata.get("new_elo") or metadata.get("elo")
        reason = metadata.get("reason", "")

        # Build config_key if not provided
        if not config_key and board_type:
            config_key = f"{board_type}_{num_players}p"

        if not config_key:
            logger.debug("[PromotionToCurriculumWatcher] No config_key in event, skipping")
            return

        logger.info(
            f"[PromotionToCurriculumWatcher] Promotion result for {config_key}: "
            f"promoted={promoted}, elo={new_elo}, reason={reason}"
        )

        # Track in history
        self._promotion_history[config_key].append({
            "promoted": promoted,
            "elo": new_elo,
            "reason": reason,
            "timestamp": time.time(),
        })
        if len(self._promotion_history[config_key]) > 20:
            self._promotion_history[config_key] = self._promotion_history[config_key][-20:]

        # Update counters
        if promoted:
            self._promotions_processed += 1
        else:
            self._failures_processed += 1

        # Update curriculum feedback
        self.feedback.record_promotion(
            config_key=config_key,
            promoted=promoted,
            new_elo=new_elo,
            promotion_reason=reason,
        )

        # Signal FeedbackAccelerator on failure
        if not promoted:
            self._signal_training_needed(config_key, reason)

        # Auto-export if enabled
        if self.auto_export:
            try:
                self.feedback.export_weights_json(self.export_path)
                # Also sync to standard P2P path for orchestrator consumption
                sync_curriculum_weights_to_p2p()
            except Exception as e:
                logger.warning(f"Failed to export weights: {e}")

    def _signal_training_needed(self, config_key: str, reason: str) -> None:
        """Signal FeedbackAccelerator that training is urgently needed.

        Called when promotion fails to accelerate the training loop.
        """
        try:
            from app.training.feedback_accelerator import get_feedback_accelerator

            accelerator = get_feedback_accelerator()
            accelerator.signal_training_needed(
                config_key=config_key,
                urgency=self.failure_urgency,
                reason=f"promotion_failed:{reason}",
            )

            logger.info(
                f"[PromotionToCurriculumWatcher] Signaled {self.failure_urgency} urgency "
                f"training for {config_key} after failed promotion"
            )
        except ImportError:
            logger.debug("FeedbackAccelerator not available for training signal")
        except Exception as e:
            logger.warning(f"Failed to signal training urgency: {e}")

    def get_promotion_stats(self, config_key: str | None = None) -> dict[str, Any]:
        """Get promotion statistics.

        Args:
            config_key: Specific config to get stats for, or None for all

        Returns:
            Dict with promotion statistics
        """
        if config_key:
            history = self._promotion_history.get(config_key, [])
            if not history:
                return {"config": config_key, "promotions": 0, "failures": 0}

            promotions = sum(1 for h in history if h["promoted"])
            failures = len(history) - promotions

            return {
                "config": config_key,
                "promotions": promotions,
                "failures": failures,
                "success_rate": promotions / len(history) if history else 0,
                "last_result": history[-1] if history else None,
            }
        else:
            return {
                "total_processed": self._promotions_processed + self._failures_processed,
                "promotions": self._promotions_processed,
                "failures": self._failures_processed,
                "success_rate": (
                    self._promotions_processed / max(1, self._promotions_processed + self._failures_processed)
                ),
                "configs_tracked": len(self._promotion_history),
            }


# Singleton promotion watcher
_promotion_watcher: PromotionToCurriculumWatcher | None = None


def wire_promotion_to_curriculum(
    auto_export: bool = True,
    failure_urgency: str = "high",
) -> PromotionToCurriculumWatcher:
    """Wire promotion outcomes to curriculum + training intensity adjustments.

    This closes the feedback loop from promotion decisions:
    - Successful promotion → reduce training weight (model is performing)
    - Failed promotion → boost weight + signal urgent training

    Args:
        auto_export: Whether to auto-export weights after promotion
        failure_urgency: Training urgency level on failure ("low", "normal", "high", "critical")

    Returns:
        PromotionToCurriculumWatcher instance
    """
    global _promotion_watcher

    _promotion_watcher = PromotionToCurriculumWatcher(
        auto_export=auto_export,
        failure_urgency=failure_urgency,
    )
    _promotion_watcher.subscribe_to_promotion_events()

    logger.info(
        f"[wire_promotion_to_curriculum] PROMOTION_COMPLETE events wired to curriculum "
        f"(failure_urgency={failure_urgency})"
    )

    return _promotion_watcher


def get_promotion_curriculum_watcher() -> PromotionToCurriculumWatcher | None:
    """Get the global promotion-to-curriculum watcher if configured."""
    return _promotion_watcher


# =============================================================================
# All-in-one wiring function (December 2025)
# =============================================================================


def wire_all_curriculum_feedback(
    elo_significant_change: float = 30.0,
    plateau_cooldown_seconds: float = 600.0,
    tournament_cooldown_seconds: float = 300.0,
    promotion_failure_urgency: str = "high",
    auto_export: bool = True,
) -> dict[str, Any]:
    """Wire all curriculum feedback integrations at once.

    This is the recommended way to set up the full curriculum feedback loop,
    connecting ELO changes, plateaus, tournaments, and promotions to
    curriculum weight adjustments.

    Args:
        elo_significant_change: ELO change threshold for rebalance
        plateau_cooldown_seconds: Cooldown between plateau rebalances
        tournament_cooldown_seconds: Cooldown between tournament rebalances
        promotion_failure_urgency: Urgency level on promotion failure
        auto_export: Auto-export weights after changes

    Returns:
        Dict with all watcher instances:
        {
            "elo": EloToCurriculumWatcher,
            "plateau": PlateauToCurriculumWatcher,
            "tournament": TournamentToCurriculumWatcher,
            "promotion": PromotionToCurriculumWatcher,
        }
    """
    watchers = {}

    # Wire ELO changes
    try:
        watchers["elo"] = wire_elo_to_curriculum(
            significant_elo_change=elo_significant_change,
            auto_export=auto_export,
        )
    except Exception as e:
        logger.warning(f"Failed to wire ELO to curriculum: {e}")

    # Wire plateau detection
    try:
        watchers["plateau"] = wire_plateau_to_curriculum(
            rebalance_cooldown_seconds=plateau_cooldown_seconds,
            auto_export=auto_export,
        )
    except Exception as e:
        logger.warning(f"Failed to wire plateau to curriculum: {e}")

    # Wire tournament results
    try:
        watchers["tournament"] = wire_tournament_to_curriculum(
            rebalance_cooldown_seconds=tournament_cooldown_seconds,
            auto_export=auto_export,
        )
    except Exception as e:
        logger.warning(f"Failed to wire tournament to curriculum: {e}")

    # Wire promotion outcomes
    try:
        watchers["promotion"] = wire_promotion_to_curriculum(
            auto_export=auto_export,
            failure_urgency=promotion_failure_urgency,
        )
    except Exception as e:
        logger.warning(f"Failed to wire promotion to curriculum: {e}")

    # Wire quality feedback
    try:
        watchers["quality"] = wire_quality_to_curriculum(
            auto_export=auto_export,
        )
    except Exception as e:
        logger.warning(f"Failed to wire quality to curriculum: {e}")

    # Wire epoch events for mid-training updates (December 2025)
    try:
        watchers["epoch"] = wire_epoch_to_curriculum(
            check_interval_epochs=5,
            auto_export=auto_export,
        )
    except Exception as e:
        logger.warning(f"Failed to wire epoch to curriculum: {e}")

    # Wire opponent tracker for weak opponent detection (December 2025)
    # This enables +50-150 ELO gain through targeted weakness exploitation
    try:
        wire_opponent_tracker_to_curriculum()
        watchers["opponent"] = True  # Boolean since it doesn't return a watcher object
        logger.info("[wire_all_curriculum_feedback] Wired opponent tracker to curriculum")
    except Exception as e:
        logger.warning(f"Failed to wire opponent tracker to curriculum: {e}")

    # Wire early stopping events to curriculum boost (December 2025)
    try:
        watchers["early_stop"] = wire_early_stop_to_curriculum(auto_export=auto_export)
    except Exception as e:
        logger.warning(f"Failed to wire early stop to curriculum: {e}")

    logger.info(
        f"[wire_all_curriculum_feedback] Wired {len(watchers)} curriculum feedback integrations"
    )

    return watchers


# =============================================================================
# TRAINING_EARLY_STOPPED → Curriculum Boost (December 2025)
# =============================================================================


class EarlyStopToCurriculumWatcher:
    """Watches for early stopping events and boosts curriculum weight.

    When training early-stops due to loss stagnation, the config needs more/better
    training data. This watcher increases the curriculum weight to allocate more
    selfplay resources to that config.

    Boost amounts:
    - loss_stagnation: +0.3 weight boost
    - elo_stagnation: +0.4 weight boost (Elo stagnation is harder to fix)
    - regression: +0.5 weight boost (regression needs aggressive correction)
    """

    def __init__(
        self,
        feedback: CurriculumFeedback | None = None,
        auto_export: bool = True,
        export_path: str = "data/curriculum_weights.json",
    ):
        self.feedback = feedback or get_curriculum_feedback()
        self.auto_export = auto_export
        self.export_path = export_path
        self._events_processed = 0
        self._subscribed = False

        # Boost amounts by reason
        self._boost_by_reason = {
            "loss_stagnation": 0.3,
            "elo_stagnation": 0.4,
            "regression": 0.5,
        }

    def subscribe(self) -> bool:
        """Subscribe to TRAINING_EARLY_STOPPED events."""
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()
            router.subscribe(
                DataEventType.TRAINING_EARLY_STOPPED.value,
                self._on_early_stopped,
            )
            self._subscribed = True
            logger.info("[EarlyStopToCurriculumWatcher] Subscribed to TRAINING_EARLY_STOPPED events")
            return True
        except ImportError:
            logger.warning("[EarlyStopToCurriculumWatcher] event_router not available")
            return False
        except Exception as e:
            logger.warning(f"[EarlyStopToCurriculumWatcher] Failed to subscribe: {e}")
            return False

    async def _on_early_stopped(self, event) -> None:
        """Handle TRAINING_EARLY_STOPPED - boost curriculum weight for stuck config."""
        payload = event.payload
        config_key = payload.get("config_key", "")
        reason = payload.get("reason", "loss_stagnation")
        epoch = payload.get("epoch", 0)
        epochs_without_improvement = payload.get("epochs_without_improvement", 0)

        if not config_key:
            return

        self._events_processed += 1

        # Get boost amount based on reason
        boost = self._boost_by_reason.get(reason, 0.3)

        # Scale boost by stagnation severity
        if epochs_without_improvement > 10:
            boost *= 1.2  # Longer stagnation = bigger boost
        elif epochs_without_improvement > 5:
            boost *= 1.1

        # Apply boost
        current_weight = self.feedback.get_weight(config_key)
        new_weight = min(2.0, current_weight + boost)  # Cap at 2.0

        self.feedback.set_weight(config_key, new_weight)

        logger.info(
            f"[EarlyStopToCurriculumWatcher] Boosted {config_key} weight: "
            f"{current_weight:.2f}→{new_weight:.2f} (reason={reason}, epoch={epoch})"
        )

        # Auto-export if enabled
        if self.auto_export:
            try:
                self.feedback.export_weights_json(self.export_path)
                sync_curriculum_weights_to_p2p()
            except Exception as e:
                logger.warning(f"Failed to export weights: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get watcher statistics."""
        return {
            "events_processed": self._events_processed,
            "subscribed": self._subscribed,
            "boost_amounts": dict(self._boost_by_reason),
        }


# Singleton early stop watcher
_early_stop_watcher: EarlyStopToCurriculumWatcher | None = None


def wire_early_stop_to_curriculum(
    auto_export: bool = True,
) -> EarlyStopToCurriculumWatcher:
    """Wire early stopping events to curriculum weight boost.

    When training early-stops, this boosts the curriculum weight for that config
    to allocate more training resources.

    Args:
        auto_export: Whether to auto-export weights after boost

    Returns:
        EarlyStopToCurriculumWatcher instance
    """
    global _early_stop_watcher

    _early_stop_watcher = EarlyStopToCurriculumWatcher(
        auto_export=auto_export,
    )
    _early_stop_watcher.subscribe()

    logger.info(
        "[wire_early_stop_to_curriculum] TRAINING_EARLY_STOPPED events wired to curriculum boost"
    )

    return _early_stop_watcher


def get_early_stop_watcher() -> EarlyStopToCurriculumWatcher | None:
    """Get the singleton early stop watcher."""
    return _early_stop_watcher


# =============================================================================
# QUALITY_SCORE → Selfplay Budget Adjustment (December 2025)
# =============================================================================

class QualityFeedbackWatcher:
    """Watches quality scores and adjusts selfplay opponent budget.

    Low-quality games indicate that opponents are too weak or games are
    too short/predictable. This watcher increases opponent search budget
    when quality is low to produce better training data.

    Quality-to-Budget Mapping:
    - quality < 0.4: 1.5x opponent budget (boost weak opponents)
    - quality 0.4-0.6: 1.2x opponent budget (moderate boost)
    - quality 0.6-0.8: 1.0x opponent budget (normal)
    - quality > 0.8: 0.8x opponent budget (save compute, games are good)

    Usage:
        from app.training.curriculum_feedback import (
            wire_quality_to_curriculum,
            get_quality_budget_multiplier,
        )

        # Wire quality events to budget adjustment
        watcher = wire_quality_to_curriculum()

        # Get budget multiplier for selfplay
        multiplier = get_quality_budget_multiplier("hex8_2p")
        budget = base_budget * multiplier
    """

    # Quality thresholds for budget adjustment
    LOW_QUALITY_THRESHOLD = 0.4
    MEDIUM_QUALITY_THRESHOLD = 0.6
    HIGH_QUALITY_THRESHOLD = 0.8

    # Budget multipliers
    LOW_QUALITY_MULTIPLIER = 1.5
    MEDIUM_QUALITY_MULTIPLIER = 1.2
    NORMAL_QUALITY_MULTIPLIER = 1.0
    HIGH_QUALITY_MULTIPLIER = 0.8

    def __init__(
        self,
        feedback: CurriculumFeedback | None = None,
        rolling_window_size: int = 100,
        auto_export: bool = True,
        export_path: str = "data/quality_budget_multipliers.json",
    ):
        """Initialize the quality feedback watcher.

        Args:
            feedback: CurriculumFeedback instance (uses singleton if None)
            rolling_window_size: Number of games for rolling average
            auto_export: Whether to auto-export multipliers
            export_path: Path to export multipliers JSON
        """
        self.feedback = feedback or get_curriculum_feedback()
        self.rolling_window_size = rolling_window_size
        self.auto_export = auto_export
        self.export_path = export_path

        # Quality history per config (rolling window)
        self._quality_history: dict[str, list[float]] = defaultdict(list)

        # Computed multipliers per config
        self._budget_multipliers: dict[str, float] = {}

        # Last update times
        self._last_update_time: float = 0.0

        self._subscribed = False

    def subscribe_to_quality_events(self) -> bool:
        """Subscribe to quality score events from the event bus.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.subscribe("QUALITY_SCORE_UPDATED", self._on_quality_score_updated)
            router.subscribe("quality_score_updated", self._on_quality_score_updated)
            self._subscribed = True
            logger.info("[QualityFeedbackWatcher] Subscribed to QUALITY_SCORE_UPDATED events")
            return True
        except Exception as e:
            logger.warning(f"[QualityFeedbackWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from quality events."""
        if not self._subscribed:
            return

        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.unsubscribe("QUALITY_SCORE_UPDATED", self._on_quality_score_updated)
            router.unsubscribe("quality_score_updated", self._on_quality_score_updated)
            self._subscribed = False
        except Exception:
            pass

    def _on_quality_score_updated(self, event: Any) -> None:
        """Handle QUALITY_SCORE_UPDATED event."""
        payload = event.payload if hasattr(event, 'payload') else {}

        config_key = payload.get("config") or payload.get("config_key", "")
        quality_score = payload.get("quality_score", payload.get("new_score", 0.5))

        if not config_key:
            return

        # Track in rolling window
        history = self._quality_history[config_key]
        history.append(quality_score)

        # Trim to window size
        if len(history) > self.rolling_window_size:
            self._quality_history[config_key] = history[-self.rolling_window_size:]

        # Update multiplier
        self._update_multiplier(config_key)
        self._last_update_time = time.time()

    def _update_multiplier(self, config_key: str) -> None:
        """Update budget multiplier based on rolling quality average."""
        history = self._quality_history.get(config_key, [])

        if not history:
            self._budget_multipliers[config_key] = self.NORMAL_QUALITY_MULTIPLIER
            return

        avg_quality = sum(history) / len(history)

        # Compute multiplier based on quality
        if avg_quality < self.LOW_QUALITY_THRESHOLD:
            multiplier = self.LOW_QUALITY_MULTIPLIER
        elif avg_quality < self.MEDIUM_QUALITY_THRESHOLD:
            multiplier = self.MEDIUM_QUALITY_MULTIPLIER
        elif avg_quality < self.HIGH_QUALITY_THRESHOLD:
            multiplier = self.NORMAL_QUALITY_MULTIPLIER
        else:
            multiplier = self.HIGH_QUALITY_MULTIPLIER

        old_multiplier = self._budget_multipliers.get(config_key, 1.0)
        if abs(multiplier - old_multiplier) > 0.05:
            logger.info(
                f"[QualityFeedbackWatcher] {config_key} quality={avg_quality:.2f}, "
                f"budget multiplier: {old_multiplier:.2f} → {multiplier:.2f}"
            )

            # Emit event for downstream listeners
            self._emit_quality_feedback_adjusted(config_key, multiplier, avg_quality)

        self._budget_multipliers[config_key] = multiplier

    def _emit_quality_feedback_adjusted(
        self,
        config_key: str,
        multiplier: float,
        avg_quality: float,
    ) -> None:
        """Emit quality feedback adjusted event."""
        try:
            from app.coordination.event_router import get_router, RouterEvent, EventSource
            from app.distributed.data_events import DataEventType

            router = get_router()
            # P0.6 Dec 2025: Use DataEventType enum for type-safe event emission
            event = RouterEvent(
                event_type=DataEventType.QUALITY_FEEDBACK_ADJUSTED,
                payload={
                    "config_key": config_key,
                    "budget_multiplier": multiplier,
                    "avg_quality": avg_quality,
                    "timestamp": time.time(),
                },
                source="quality_feedback_watcher",
                origin=EventSource.ROUTER,
            )
            router.publish_sync(DataEventType.QUALITY_FEEDBACK_ADJUSTED, event.payload, "quality_feedback_watcher")
            logger.debug(f"Emitted QUALITY_FEEDBACK_ADJUSTED for {config_key}")
        except Exception as e:
            logger.debug(f"Failed to emit quality feedback event: {e}")

    def get_budget_multiplier(self, config_key: str) -> float:
        """Get the current budget multiplier for a config.

        Args:
            config_key: Config identifier (e.g., "hex8_2p")

        Returns:
            Budget multiplier (1.0 = normal, >1 = boost, <1 = reduce)
        """
        return self._budget_multipliers.get(config_key, self.NORMAL_QUALITY_MULTIPLIER)

    def get_all_multipliers(self) -> dict[str, float]:
        """Get all current budget multipliers."""
        return dict(self._budget_multipliers)

    def get_quality_summary(self, config_key: str) -> dict[str, Any]:
        """Get quality summary for a config.

        Returns:
            Dict with avg_quality, multiplier, history_size
        """
        history = self._quality_history.get(config_key, [])
        avg_quality = sum(history) / len(history) if history else 0.5

        return {
            "config_key": config_key,
            "avg_quality": avg_quality,
            "history_size": len(history),
            "budget_multiplier": self.get_budget_multiplier(config_key),
            "last_update": self._last_update_time,
        }

    def export_multipliers_json(self, output_path: str | None = None) -> None:
        """Export budget multipliers to JSON.

        Args:
            output_path: Path to output file (uses default if None)
        """
        output_path = output_path or self.export_path

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "multipliers": self._budget_multipliers,
            "summaries": {
                config_key: self.get_quality_summary(config_key)
                for config_key in self._budget_multipliers
            },
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.debug(f"Exported quality budget multipliers to {output_path}")


# Singleton watcher
_quality_watcher: QualityFeedbackWatcher | None = None


def wire_quality_to_curriculum(
    rolling_window_size: int = 100,
    auto_export: bool = True,
) -> QualityFeedbackWatcher:
    """Wire quality score events to selfplay budget adjustment.

    This is the main entry point for connecting quality metrics to
    automatic selfplay opponent budget adjustment.

    Args:
        rolling_window_size: Number of games for rolling quality average
        auto_export: Whether to auto-export multipliers

    Returns:
        QualityFeedbackWatcher instance
    """
    global _quality_watcher

    _quality_watcher = QualityFeedbackWatcher(
        rolling_window_size=rolling_window_size,
        auto_export=auto_export,
    )
    _quality_watcher.subscribe_to_quality_events()

    logger.info(
        f"[wire_quality_to_curriculum] Quality events wired to selfplay budget adjustment "
        f"(window={rolling_window_size})"
    )

    return _quality_watcher


def get_quality_feedback_watcher() -> QualityFeedbackWatcher | None:
    """Get the global quality feedback watcher if configured."""
    return _quality_watcher


def get_quality_budget_multiplier(config_key: str) -> float:
    """Get the budget multiplier for a config (convenience function).

    If no watcher is configured, returns 1.0 (no adjustment).

    Args:
        config_key: Config identifier (e.g., "hex8_2p")

    Returns:
        Budget multiplier for selfplay opponent budget
    """
    if _quality_watcher is None:
        return 1.0
    return _quality_watcher.get_budget_multiplier(config_key)


# =============================================================================
# Training Epoch → Curriculum Updates (December 2025)
# =============================================================================

class EpochToCurriculumWatcher:
    """Watches for training epoch completions and updates curriculum mid-training.

    Subscribes to training.epoch.completed events and adjusts curriculum weights
    based on training progress:
    - Rapid improvement (>10% loss reduction) → reduce weight (training is working)
    - Stalled progress (<2% loss reduction over 5 epochs) → boost weight (needs help)
    - Regression (loss increasing) → significantly boost weight (urgent)

    This enables the system to adapt curriculum weights during long training runs
    rather than waiting until training completes.

    Usage:
        from app.training.curriculum_feedback import wire_epoch_to_curriculum

        # Wire epoch events to curriculum
        watcher = wire_epoch_to_curriculum()
    """

    # Progress thresholds
    RAPID_IMPROVEMENT_THRESHOLD = 0.10  # 10% loss reduction
    STALLED_THRESHOLD = 0.02  # 2% or less improvement
    REGRESSION_THRESHOLD = 0.05  # 5% loss increase

    # Weight adjustments
    RAPID_IMPROVEMENT_ADJUSTMENT = -0.1  # Reduce priority (training working)
    STALLED_ADJUSTMENT = 0.15  # Boost priority (needs more focus)
    REGRESSION_ADJUSTMENT = 0.25  # Major boost (urgent intervention)

    def __init__(
        self,
        feedback: CurriculumFeedback | None = None,
        check_interval_epochs: int = 5,
        auto_export: bool = True,
        export_path: str = "data/curriculum_weights.json",
    ):
        """Initialize the epoch-to-curriculum watcher.

        Args:
            feedback: CurriculumFeedback instance (uses singleton if None)
            check_interval_epochs: How often to evaluate progress (in epochs)
            auto_export: Whether to auto-export weights after adjustment
            export_path: Path to export weights JSON
        """
        self.feedback = feedback or get_curriculum_feedback()
        self.check_interval_epochs = check_interval_epochs
        self.auto_export = auto_export
        self.export_path = export_path

        # Track loss history per config
        self._loss_history: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._max_history = 20  # Keep last 20 epochs

        self._subscribed = False
        self._adjustments_made = 0

    def subscribe_to_epoch_events(self) -> bool:
        """Subscribe to training.epoch.completed events from the event bus.

        Returns:
            True if successfully subscribed
        """
        try:
            from app.training.event_integration import TrainingTopics
            from app.events.bus import get_bus

            bus = get_bus()
            if bus is None:
                logger.debug("[EpochToCurriculumWatcher] Event bus not available")
                return False

            bus.subscribe(TrainingTopics.EPOCH_COMPLETED, self._on_epoch_completed)
            self._subscribed = True
            logger.info("[EpochToCurriculumWatcher] Subscribed to EPOCH_COMPLETED events")
            return True
        except ImportError as e:
            logger.debug(f"[EpochToCurriculumWatcher] Import failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"[EpochToCurriculumWatcher] Failed to subscribe: {e}")
            return False

    def unsubscribe(self) -> None:
        """Unsubscribe from epoch events."""
        if not self._subscribed:
            return

        try:
            from app.training.event_integration import TrainingTopics
            from app.events.bus import get_bus

            bus = get_bus()
            if bus:
                bus.unsubscribe(TrainingTopics.EPOCH_COMPLETED, self._on_epoch_completed)
            self._subscribed = False
        except Exception:
            pass

    def _on_epoch_completed(self, event: Any) -> None:
        """Handle training.epoch.completed event.

        Records loss history and potentially adjusts curriculum weights
        at checkpoint intervals.
        """
        # Extract event data
        if hasattr(event, 'payload'):
            data = event.payload
        elif hasattr(event, '__dict__'):
            data = event.__dict__
        else:
            data = event if isinstance(event, dict) else {}

        config_key = data.get("config_key", "")
        epoch = data.get("epoch", 0)
        val_loss = data.get("val_loss")
        train_loss = data.get("train_loss", 0)

        if not config_key or epoch == 0:
            return

        # Use val_loss if available, otherwise train_loss
        loss = val_loss if val_loss is not None else train_loss

        # Track loss history
        history = self._loss_history[config_key]
        history.append((epoch, loss))

        # Trim to max history
        if len(history) > self._max_history:
            self._loss_history[config_key] = history[-self._max_history:]

        # Check progress at interval
        if epoch % self.check_interval_epochs == 0 and len(history) >= self.check_interval_epochs:
            self._evaluate_progress(config_key, epoch)

    def _evaluate_progress(self, config_key: str, current_epoch: int) -> None:
        """Evaluate training progress and potentially adjust curriculum weights.

        Args:
            config_key: Config identifier
            current_epoch: Current training epoch
        """
        history = self._loss_history.get(config_key, [])
        if len(history) < self.check_interval_epochs:
            return

        # Get recent losses
        recent = history[-self.check_interval_epochs:]
        older = history[-self.check_interval_epochs * 2:-self.check_interval_epochs] if len(history) >= self.check_interval_epochs * 2 else []

        current_loss = recent[-1][1]
        start_loss = recent[0][1]

        # Calculate improvement rate
        if start_loss > 0:
            improvement_rate = (start_loss - current_loss) / start_loss
        else:
            improvement_rate = 0.0

        # Compare to older period if available
        older_improvement = 0.0
        if older:
            older_start = older[0][1]
            older_end = older[-1][1]
            if older_start > 0:
                older_improvement = (older_start - older_end) / older_start

        # Determine adjustment
        adjustment = 0.0
        reason = ""

        if improvement_rate > self.RAPID_IMPROVEMENT_THRESHOLD:
            # Rapid improvement - training is working well
            adjustment = self.RAPID_IMPROVEMENT_ADJUSTMENT
            reason = f"rapid_improvement_{improvement_rate:.1%}"
        elif improvement_rate < -self.REGRESSION_THRESHOLD:
            # Regression - loss is increasing
            adjustment = self.REGRESSION_ADJUSTMENT
            reason = f"regression_{improvement_rate:.1%}"
        elif abs(improvement_rate) < self.STALLED_THRESHOLD:
            # Stalled - very little progress
            # Only boost if we were previously improving
            if older_improvement > self.STALLED_THRESHOLD:
                adjustment = self.STALLED_ADJUSTMENT
                reason = f"stalled_after_progress_{improvement_rate:.1%}"

        if adjustment != 0.0:
            self._apply_adjustment(config_key, adjustment, reason, current_epoch, improvement_rate)

    def _apply_adjustment(
        self,
        config_key: str,
        adjustment: float,
        reason: str,
        epoch: int,
        improvement_rate: float,
    ) -> None:
        """Apply curriculum weight adjustment.

        Args:
            config_key: Config identifier
            adjustment: Weight adjustment amount
            reason: Reason for adjustment
            epoch: Current epoch
            improvement_rate: Training improvement rate
        """
        # Get current weight
        weights = self.feedback.get_curriculum_weights()
        current_weight = weights.get(config_key, 1.0)

        # Apply adjustment with bounds
        new_weight = max(
            self.feedback.weight_min,
            min(self.feedback.weight_max, current_weight + adjustment)
        )

        # Update weight in feedback
        self.feedback._current_weights[config_key] = new_weight
        self._adjustments_made += 1

        logger.info(
            f"[EpochToCurriculumWatcher] Mid-training curriculum adjustment for {config_key} "
            f"at epoch {epoch}: weight {current_weight:.2f} -> {new_weight:.2f} "
            f"(reason: {reason})"
        )

        # Auto-export if enabled
        if self.auto_export:
            try:
                self.feedback.export_weights_json(self.export_path)
            except Exception as e:
                logger.warning(f"Failed to export weights: {e}")

        # Emit curriculum updated event
        self.feedback._emit_curriculum_updated(config_key, new_weight, f"epoch_{reason}")

    def get_loss_history(self, config_key: str) -> list[tuple[int, float]]:
        """Get loss history for a config.

        Args:
            config_key: Config identifier

        Returns:
            List of (epoch, loss) tuples
        """
        return list(self._loss_history.get(config_key, []))

    def get_stats(self) -> dict[str, Any]:
        """Get watcher statistics.

        Returns:
            Dict with statistics
        """
        return {
            "subscribed": self._subscribed,
            "adjustments_made": self._adjustments_made,
            "configs_tracked": len(self._loss_history),
            "check_interval_epochs": self.check_interval_epochs,
            "thresholds": {
                "rapid_improvement": self.RAPID_IMPROVEMENT_THRESHOLD,
                "stalled": self.STALLED_THRESHOLD,
                "regression": self.REGRESSION_THRESHOLD,
            },
        }


# Singleton epoch watcher
_epoch_watcher: EpochToCurriculumWatcher | None = None


def wire_epoch_to_curriculum(
    check_interval_epochs: int = 5,
    auto_export: bool = True,
) -> EpochToCurriculumWatcher:
    """Wire training epoch events to curriculum weight adjustments.

    This enables mid-training curriculum updates based on training progress:
    - Rapid improvement → reduce training priority (training is working)
    - Stalled progress → boost priority (needs more focus)
    - Regression → major boost (urgent intervention)

    Args:
        check_interval_epochs: How often to evaluate progress (in epochs)
        auto_export: Whether to auto-export weights after adjustment

    Returns:
        EpochToCurriculumWatcher instance
    """
    global _epoch_watcher

    _epoch_watcher = EpochToCurriculumWatcher(
        check_interval_epochs=check_interval_epochs,
        auto_export=auto_export,
    )
    _epoch_watcher.subscribe_to_epoch_events()

    logger.info(
        f"[wire_epoch_to_curriculum] EPOCH_COMPLETED events wired to curriculum "
        f"(check_interval={check_interval_epochs} epochs)"
    )

    return _epoch_watcher


def get_epoch_curriculum_watcher() -> EpochToCurriculumWatcher | None:
    """Get the global epoch-to-curriculum watcher if configured."""
    return _epoch_watcher
