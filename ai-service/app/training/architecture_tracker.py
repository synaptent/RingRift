"""Architecture Performance Tracker for RingRift Model Training.

Tracks performance metrics across different neural network architectures (v2, v4, v5, etc.)
to enable intelligent training compute allocation. Higher-performing architectures get
more selfplay games and training resources.

Key Metrics:
    - avg_elo: Average Elo rating across all evaluations
    - best_elo: Best observed Elo rating
    - elo_per_training_hour: Efficiency metric (Elo gain / training time)
    - games_evaluated: Total games used in evaluations

Usage:
    from app.training.architecture_tracker import (
        get_architecture_tracker,
        record_evaluation,
        get_best_architecture,
    )

    # Record evaluation result
    record_evaluation(
        architecture="v5",
        board_type="hex8",
        num_players=2,
        elo=1450,
        training_hours=2.5,
        games_evaluated=100,
    )

    # Get best architecture for allocation decisions
    best = get_best_architecture(board_type="hex8", num_players=2)
    print(f"Best: {best.architecture} with Elo {best.avg_elo:.0f}")
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureStats:
    """Performance statistics for a single architecture.

    Tracks Elo ratings, training efficiency, and evaluation history
    to enable data-driven architecture selection.
    """

    architecture: str  # e.g., "v2", "v4", "v5_heavy", "nnue_v1"
    board_type: str  # e.g., "hex8", "square8"
    num_players: int

    # Elo metrics
    avg_elo: float = 1000.0
    best_elo: float = 1000.0
    worst_elo: float = 1000.0
    elo_variance: float = 0.0

    # Training efficiency
    training_hours: float = 0.0
    elo_per_training_hour: float = 0.0

    # Evaluation history
    games_evaluated: int = 0
    evaluation_count: int = 0
    last_evaluation_time: float = 0.0

    # Running statistics for online mean/variance
    _elo_sum: float = 0.0
    _elo_sum_sq: float = 0.0

    @property
    def config_key(self) -> str:
        """Configuration key for this architecture."""
        return f"{self.board_type}_{self.num_players}p"

    @property
    def full_key(self) -> str:
        """Full key including architecture."""
        return f"{self.architecture}:{self.config_key}"

    @property
    def efficiency_score(self) -> float:
        """Efficiency score: Elo per training hour invested.

        Higher is better - indicates architecture learns faster.
        """
        if self.training_hours <= 0:
            return 0.0
        # Use Elo above baseline (1000) per hour
        return max(0.0, self.avg_elo - 1000.0) / self.training_hours

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """95% confidence interval on avg_elo.

        Uses standard error with t-distribution approximation.
        """
        if self.evaluation_count < 2:
            return (self.avg_elo - 200.0, self.avg_elo + 200.0)

        std_dev = math.sqrt(max(0.0, self.elo_variance))
        std_error = std_dev / math.sqrt(self.evaluation_count)
        margin = 1.96 * std_error  # 95% CI

        return (self.avg_elo - margin, self.avg_elo + margin)

    def record_evaluation(
        self,
        elo: float,
        training_hours: float,
        games_evaluated: int,
    ) -> None:
        """Record a new evaluation result.

        Updates running statistics using Welford's online algorithm
        for numerically stable mean and variance.

        Args:
            elo: Elo rating from evaluation
            training_hours: Training time invested in this architecture
            games_evaluated: Number of games in this evaluation
        """
        self.evaluation_count += 1
        self.games_evaluated += games_evaluated
        self.training_hours += training_hours
        self.last_evaluation_time = time.time()

        # Update Elo bounds
        if elo > self.best_elo:
            self.best_elo = elo
        if elo < self.worst_elo or self.evaluation_count == 1:
            self.worst_elo = elo

        # Welford's online algorithm for mean and variance
        self._elo_sum += elo
        self._elo_sum_sq += elo * elo

        n = self.evaluation_count
        self.avg_elo = self._elo_sum / n

        if n >= 2:
            # Population variance
            mean_sq = self.avg_elo * self.avg_elo
            self.elo_variance = (self._elo_sum_sq / n) - mean_sq

        # Update efficiency
        if self.training_hours > 0:
            self.elo_per_training_hour = max(0.0, self.avg_elo - 1000.0) / self.training_hours

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "architecture": self.architecture,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "avg_elo": self.avg_elo,
            "best_elo": self.best_elo,
            "worst_elo": self.worst_elo,
            "elo_variance": self.elo_variance,
            "training_hours": self.training_hours,
            "elo_per_training_hour": self.elo_per_training_hour,
            "games_evaluated": self.games_evaluated,
            "evaluation_count": self.evaluation_count,
            "last_evaluation_time": self.last_evaluation_time,
            "_elo_sum": self._elo_sum,
            "_elo_sum_sq": self._elo_sum_sq,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureStats:
        """Deserialize from dictionary."""
        stats = cls(
            architecture=data["architecture"],
            board_type=data["board_type"],
            num_players=data["num_players"],
        )
        stats.avg_elo = data.get("avg_elo", 1000.0)
        stats.best_elo = data.get("best_elo", 1000.0)
        stats.worst_elo = data.get("worst_elo", 1000.0)
        stats.elo_variance = data.get("elo_variance", 0.0)
        stats.training_hours = data.get("training_hours", 0.0)
        stats.elo_per_training_hour = data.get("elo_per_training_hour", 0.0)
        stats.games_evaluated = data.get("games_evaluated", 0)
        stats.evaluation_count = data.get("evaluation_count", 0)
        stats.last_evaluation_time = data.get("last_evaluation_time", 0.0)
        stats._elo_sum = data.get("_elo_sum", 0.0)
        stats._elo_sum_sq = data.get("_elo_sum_sq", 0.0)
        return stats


class ArchitectureTracker:
    """Singleton tracker for architecture performance across all configs.

    Thread-safe registry that persists to disk and enables:
    - Recording evaluation results per architecture
    - Querying best/most-efficient architectures
    - Providing allocation weights for SelfplayScheduler
    """

    _instance: ArchitectureTracker | None = None
    _lock = threading.Lock()

    def __init__(self, state_path: str | Path | None = None):
        """Initialize architecture tracker.

        Args:
            state_path: Path to persist state (default: data/architecture_stats.json)
        """
        self._stats: dict[str, ArchitectureStats] = {}
        self._state_path = Path(state_path or "data/architecture_stats.json")
        self._dirty = False
        self._load_state()

    @classmethod
    def get_instance(cls) -> ArchitectureTracker:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _load_state(self) -> None:
        """Load state from disk."""
        if not self._state_path.exists():
            return

        try:
            with open(self._state_path) as f:
                data = json.load(f)

            for key, stats_data in data.get("stats", {}).items():
                self._stats[key] = ArchitectureStats.from_dict(stats_data)

            logger.info(
                f"ArchitectureTracker: Loaded {len(self._stats)} architecture stats"
            )
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
            logger.warning(f"ArchitectureTracker: Could not load state: {e}")

    def _save_state(self) -> None:
        """Persist state to disk."""
        if not self._dirty:
            return

        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "stats": {key: stats.to_dict() for key, stats in self._stats.items()},
                "last_updated": time.time(),
            }
            with open(self._state_path, "w") as f:
                json.dump(data, f, indent=2)
            self._dirty = False
        except OSError as e:
            logger.error(f"ArchitectureTracker: Could not save state: {e}")

    def record_evaluation(
        self,
        architecture: str,
        board_type: str,
        num_players: int,
        elo: float,
        training_hours: float = 0.0,
        games_evaluated: int = 0,
    ) -> ArchitectureStats:
        """Record an evaluation result for an architecture.

        Args:
            architecture: Architecture version (e.g., "v4", "v5_heavy")
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players
            elo: Elo rating from evaluation
            training_hours: Additional training time for this evaluation
            games_evaluated: Games used in evaluation

        Returns:
            Updated ArchitectureStats
        """
        key = f"{architecture}:{board_type}_{num_players}p"

        with self._lock:
            if key not in self._stats:
                self._stats[key] = ArchitectureStats(
                    architecture=architecture,
                    board_type=board_type,
                    num_players=num_players,
                )

            self._stats[key].record_evaluation(elo, training_hours, games_evaluated)
            self._dirty = True
            self._save_state()

            return self._stats[key]

    def get_stats(
        self,
        architecture: str,
        board_type: str,
        num_players: int,
    ) -> ArchitectureStats | None:
        """Get stats for a specific architecture/config combination."""
        key = f"{architecture}:{board_type}_{num_players}p"
        return self._stats.get(key)

    def get_all_stats(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[ArchitectureStats]:
        """Get all stats, optionally filtered by config.

        Args:
            board_type: Filter by board type (optional)
            num_players: Filter by player count (optional)

        Returns:
            List of ArchitectureStats matching filters
        """
        results = []
        for stats in self._stats.values():
            if board_type and stats.board_type != board_type:
                continue
            if num_players and stats.num_players != num_players:
                continue
            results.append(stats)
        return results

    def get_best_architecture(
        self,
        board_type: str,
        num_players: int,
        metric: str = "avg_elo",
    ) -> ArchitectureStats | None:
        """Get best-performing architecture for a config.

        Args:
            board_type: Board type
            num_players: Player count
            metric: Metric to rank by ("avg_elo", "best_elo", "efficiency_score")

        Returns:
            Best ArchitectureStats or None if no data
        """
        candidates = self.get_all_stats(board_type, num_players)
        if not candidates:
            return None

        def get_metric(stats: ArchitectureStats) -> float:
            if metric == "efficiency_score":
                return stats.efficiency_score
            elif metric == "best_elo":
                return stats.best_elo
            else:
                return stats.avg_elo

        return max(candidates, key=get_metric)

    def get_efficiency_ranking(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[tuple[str, float]]:
        """Get architectures ranked by training efficiency.

        Args:
            board_type: Filter by board type (optional)
            num_players: Filter by player count (optional)

        Returns:
            List of (architecture, efficiency_score) tuples, highest first
        """
        candidates = self.get_all_stats(board_type, num_players)
        ranking = [
            (stats.architecture, stats.efficiency_score)
            for stats in candidates
            if stats.evaluation_count > 0
        ]
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def compute_allocation_weights(
        self,
        board_type: str,
        num_players: int,
        temperature: float = 0.5,
    ) -> dict[str, float]:
        """Compute allocation weights for architectures using softmax.

        Higher-performing architectures get more weight for selfplay allocation.
        Uses efficiency_score as the basis for weighting.

        Args:
            board_type: Board type
            num_players: Player count
            temperature: Softmax temperature (lower = more concentrated on best)

        Returns:
            Dictionary mapping architecture -> allocation weight (sums to 1.0)
        """
        candidates = self.get_all_stats(board_type, num_players)
        if not candidates:
            return {}

        # Filter to architectures with evaluations
        candidates = [s for s in candidates if s.evaluation_count > 0]
        if not candidates:
            return {}

        # Compute softmax weights based on efficiency score
        scores = [s.efficiency_score for s in candidates]
        max_score = max(scores) if scores else 0.0

        # Numerical stability: subtract max before exp
        exp_scores = []
        for score in scores:
            exp_scores.append(math.exp((score - max_score) / temperature))

        total = sum(exp_scores)
        if total <= 0:
            # Uniform weights if all scores are 0
            uniform = 1.0 / len(candidates)
            return {s.architecture: uniform for s in candidates}

        weights = {}
        for stats, exp_score in zip(candidates, exp_scores, strict=False):
            weights[stats.architecture] = exp_score / total

        return weights

    def get_architecture_boost(
        self,
        architecture: str,
        board_type: str,
        num_players: int,
        threshold_elo_diff: float = 50.0,
    ) -> float:
        """Get boost factor for an architecture based on relative performance.

        Returns a factor > 1.0 if this architecture is better than average,
        < 1.0 if worse, exactly 1.0 if at average.

        Args:
            architecture: Architecture to check
            board_type: Board type
            num_players: Player count
            threshold_elo_diff: Minimum Elo difference for boost

        Returns:
            Boost factor (1.0 = no boost)
        """
        current = self.get_stats(architecture, board_type, num_players)
        if not current or current.evaluation_count == 0:
            return 1.0

        best = self.get_best_architecture(board_type, num_players, metric="avg_elo")
        if not best or best.architecture == architecture:
            return 1.0

        elo_diff = best.avg_elo - current.avg_elo
        if elo_diff < threshold_elo_diff:
            return 1.0

        # Boost proportional to Elo difference
        # Every 100 Elo difference = 0.1 boost
        return 1.0 + (elo_diff / 1000.0)


# ============================================
# Convenience Functions
# ============================================


def get_architecture_tracker() -> ArchitectureTracker:
    """Get the singleton architecture tracker."""
    return ArchitectureTracker.get_instance()


def record_evaluation(
    architecture: str,
    board_type: str,
    num_players: int,
    elo: float,
    training_hours: float = 0.0,
    games_evaluated: int = 0,
) -> ArchitectureStats:
    """Record an evaluation result for an architecture.

    Convenience wrapper for ArchitectureTracker.record_evaluation().
    """
    return get_architecture_tracker().record_evaluation(
        architecture=architecture,
        board_type=board_type,
        num_players=num_players,
        elo=elo,
        training_hours=training_hours,
        games_evaluated=games_evaluated,
    )


def get_best_architecture(
    board_type: str,
    num_players: int,
    metric: str = "avg_elo",
) -> ArchitectureStats | None:
    """Get best-performing architecture for a config.

    Convenience wrapper for ArchitectureTracker.get_best_architecture().
    """
    return get_architecture_tracker().get_best_architecture(
        board_type=board_type,
        num_players=num_players,
        metric=metric,
    )


def get_allocation_weights(
    board_type: str,
    num_players: int,
    temperature: float = 0.5,
) -> dict[str, float]:
    """Get allocation weights for architectures.

    Convenience wrapper for ArchitectureTracker.compute_allocation_weights().
    """
    return get_architecture_tracker().compute_allocation_weights(
        board_type=board_type,
        num_players=num_players,
        temperature=temperature,
    )
