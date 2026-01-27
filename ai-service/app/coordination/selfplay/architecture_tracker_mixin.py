"""ArchitectureTrackerMixin - Architecture performance tracking for SelfplayScheduler.

January 2026 Sprint 17.9: Extracted from selfplay_scheduler.py (~150 LOC)
to reduce main file size toward ~1,800 LOC target.

This mixin provides:
- Architecture allocation weight calculation
- Evaluation result recording
- Architecture performance boost calculation
- Best architecture selection

Usage:
    class SelfplayScheduler(ArchitectureTrackerMixin, ...):
        pass

No instance attributes required from main class - all methods delegate to
the app.training.architecture_tracker module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


class ArchitectureTrackerMixin:
    """Mixin providing architecture performance tracking methods.

    This mixin extracts architecture tracking from SelfplayScheduler.
    All methods delegate to the app.training.architecture_tracker module,
    which manages the actual performance data storage and computation.

    Methods provided:
        get_architecture_weights: Get allocation weights based on Elo
        record_architecture_evaluation: Record evaluation results
        get_architecture_boost: Get performance-based boost factor
        get_best_architecture: Get best architecture for a config
    """

    # =========================================================================
    # Architecture Performance Tracking (December 29, 2025)
    # =========================================================================

    def get_architecture_weights(
        self,
        board_type: str,
        num_players: int,
        temperature: float = 0.5,
    ) -> dict[str, float]:
        """Get allocation weights for architectures based on Elo performance.

        Higher-performing architectures get more weight for selfplay/training allocation.
        Uses softmax with temperature to control concentration on best architecture.

        Args:
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)
            temperature: Softmax temperature (lower = more concentrated on best)

        Returns:
            Dictionary mapping architecture name to allocation weight (sums to 1.0)

        Example:
            >>> weights = scheduler.get_architecture_weights("hex8", 2)
            >>> # {"v5": 0.4, "v4": 0.3, "v3": 0.2, "v2": 0.1}
        """
        try:
            from app.training.architecture_tracker import get_allocation_weights

            return get_allocation_weights(
                board_type=board_type,
                num_players=num_players,
                temperature=temperature,
            )
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
            return {}
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting architecture weights: {e}")
            return {}

    def record_architecture_evaluation(
        self,
        architecture: str,
        board_type: str,
        num_players: int,
        elo: float,
        training_hours: float = 0.0,
        games_evaluated: int = 0,
    ) -> None:
        """Record an evaluation result for an architecture.

        Called after gauntlet evaluation to track architecture performance.
        The architecture tracker uses this data to compute allocation weights.

        Args:
            architecture: Architecture version (e.g., "v4", "v5_heavy")
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players
            elo: Elo rating from evaluation
            training_hours: Additional training time for this evaluation
            games_evaluated: Games used in evaluation
        """
        try:
            from app.training.architecture_tracker import record_evaluation

            stats = record_evaluation(
                architecture=architecture,
                board_type=board_type,
                num_players=num_players,
                elo=elo,
                training_hours=training_hours,
                games_evaluated=games_evaluated,
            )
            logger.info(
                f"[SelfplayScheduler] Architecture evaluation recorded: "
                f"{architecture} on {board_type}_{num_players}p -> Elo {elo:.0f} "
                f"(avg: {stats.avg_elo:.0f}, best: {stats.best_elo:.0f})"
            )
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
        except Exception as e:
            logger.warning(f"[SelfplayScheduler] Failed to record architecture evaluation: {e}")

    def get_architecture_boost(
        self,
        architecture: str,
        board_type: str,
        num_players: int,
        threshold_elo_diff: float = 50.0,
    ) -> float:
        """Get boost factor for an architecture based on relative performance.

        Returns a factor > 1.0 if this architecture is better than average,
        < 1.0 if worse, exactly 1.0 if at average or no data available.

        Args:
            architecture: Architecture to check (e.g., "v4", "v5")
            board_type: Board type
            num_players: Player count
            threshold_elo_diff: Minimum Elo difference for boost

        Returns:
            Boost factor (1.0 = no boost)
        """
        try:
            from app.training.architecture_tracker import get_architecture_tracker

            tracker = get_architecture_tracker()
            return tracker.get_architecture_boost(
                architecture=architecture,
                board_type=board_type,
                num_players=num_players,
                threshold_elo_diff=threshold_elo_diff,
            )
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
            return 1.0
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting architecture boost: {e}")
            return 1.0

    def get_best_architecture(
        self,
        board_type: str,
        num_players: int,
        metric: str = "avg_elo",
    ) -> str | None:
        """Get the best-performing architecture for a configuration.

        Args:
            board_type: Board type
            num_players: Player count
            metric: Metric to rank by ("avg_elo", "best_elo", "efficiency_score")

        Returns:
            Architecture name (e.g., "v5") or None if no data available
        """
        try:
            from app.training.architecture_tracker import get_best_architecture

            stats = get_best_architecture(
                board_type=board_type,
                num_players=num_players,
                metric=metric,
            )
            if stats:
                return stats.architecture
            return None
        except ImportError:
            logger.debug("[SelfplayScheduler] architecture_tracker not available")
            return None
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting best architecture: {e}")
            return None
