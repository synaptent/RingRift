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

logger = logging.getLogger(__name__)

# Constants
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
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
