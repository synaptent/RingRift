"""Status monitoring mixin for SelfplayScheduler.

Provides health checks, metrics, and status reporting functionality.

Sprint 17.3 (Jan 4, 2026): Extracted from selfplay_scheduler.py (4,743 LOC).

This module uses a mixin pattern to allow the main SelfplayScheduler to
inherit these methods while keeping them in a separate file for maintainability.

Usage:
    from app.coordination.selfplay.status_monitoring import StatusMonitoringMixin

    class SelfplayScheduler(HandlerBase, StatusMonitoringMixin):
        ...
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.coordination.handler_base import HealthCheckResult
    from app.coordination.selfplay_priority_types import ConfigPriority

logger = logging.getLogger(__name__)


class StatusMonitoringMixin:
    """Mixin providing status monitoring functionality for SelfplayScheduler.

    This mixin expects the following attributes on self:
    - _subscribed: bool
    - _last_priority_update: float
    - _node_capabilities: dict
    - _config_priorities: dict[str, ConfigPriority]
    - _metrics_collector: SchedulerMetricsCollector
    - _overloaded_nodes: dict[str, float] (optional)
    """

    # =========================================================================
    # Node Backoff Management
    # =========================================================================

    def is_node_under_backoff(self, node_id: str) -> bool:
        """Check if a node is under backoff due to overload.

        Dec 29, 2025: Used by job dispatch to avoid overloaded nodes.

        Args:
            node_id: Node identifier to check

        Returns:
            True if node is in backoff period, False otherwise
        """
        if not hasattr(self, "_overloaded_nodes"):
            return False

        current_time = time.time()
        backoff_until = self._overloaded_nodes.get(node_id, 0)
        return backoff_until > current_time

    def get_overloaded_nodes(self) -> list[str]:
        """Get list of nodes currently under backoff.

        Dec 29, 2025: Returns nodes that should be avoided for job dispatch.

        Returns:
            List of node IDs currently in backoff period
        """
        if not hasattr(self, "_overloaded_nodes"):
            return []

        current_time = time.time()

        # Clean up expired backoffs and return active ones
        active = []
        expired = []
        for node_id, backoff_until in self._overloaded_nodes.items():
            if backoff_until > current_time:
                active.append(node_id)
            else:
                expired.append(node_id)

        for node_id in expired:
            del self._overloaded_nodes[node_id]

        return active

    # =========================================================================
    # Status Reporting
    # =========================================================================

    def get_metrics(self) -> dict[str, Any]:
        """Get throughput metrics for monitoring.

        Dec 30, 2025: Refactored to use SchedulerMetricsCollector.
        """
        return self._metrics_collector.get_metrics()

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status."""
        return {
            "subscribed": self._subscribed,
            "last_priority_update": self._last_priority_update,
            "node_count": len(self._node_capabilities),
            "overloaded_nodes": self.get_overloaded_nodes(),
            "config_priorities": {
                cfg: {
                    "priority_score": p.priority_score,
                    "staleness_hours": p.staleness_hours,
                    "elo_velocity": p.elo_velocity,
                    "exploration_boost": p.exploration_boost,
                    "curriculum_weight": p.curriculum_weight,
                    "games_allocated": p.games_allocated,
                }
                for cfg, p in self._config_priorities.items()
            },
        }

    def get_top_priorities(self, n: int = 5) -> list[dict[str, Any]]:
        """Get top N priority configurations with details."""
        sorted_configs = sorted(
            self._config_priorities.values(),
            key=lambda p: -p.priority_score,
        )

        return [
            {
                "config": p.config_key,
                "priority": p.priority_score,
                "staleness_hours": p.staleness_hours,
                "elo_velocity": p.elo_velocity,
                "exploration_boost": p.exploration_boost,
                "curriculum_weight": p.curriculum_weight,
            }
            for p in sorted_configs[:n]
        ]

    # =========================================================================
    # Health Checks
    # =========================================================================

    def _health_check_impl(self) -> "HealthCheckResult":
        """Implementation of health check for selfplay scheduler.

        This is a helper that can be called from the main health_check method.
        The main class should call super().health_check() and then merge results.

        December 30, 2025: Incorporates scheduler-specific metrics.

        Returns:
            Health check result with scheduler status and metrics.
        """
        from app.coordination.handler_base import HealthCheckResult

        # Get metrics from collector
        games_in_window = self._metrics_collector.get_games_in_window()
        games_total = self._metrics_collector._games_allocated_total

        # Determine health status
        current_time = time.time()
        stale_priority = current_time - self._last_priority_update > 300  # 5 min
        healthy = self._subscribed and not stale_priority

        message = "Running" if healthy else (
            "Not subscribed to events" if not self._subscribed else
            "Priority data stale (>5 min)"
        )

        details = {
            "subscribed": self._subscribed,
            "configs_tracked": len(self._config_priorities),
            "nodes_tracked": len(self._node_capabilities),
            "last_priority_update": self._last_priority_update,
            "priority_age_seconds": current_time - self._last_priority_update,
            "games_allocated_total": games_total,
            "games_in_last_hour": games_in_window,
        }

        return HealthCheckResult(
            healthy=healthy,
            message=message,
            details=details,
        )

    # =========================================================================
    # Architecture Performance Tracking
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
        """
        try:
            from app.training.architecture_tracker import get_allocation_weights

            return get_allocation_weights(
                board_type=board_type,
                num_players=num_players,
                temperature=temperature,
            )
        except ImportError:
            logger.debug("[StatusMonitoring] architecture_tracker not available")
            return {}
        except Exception as e:
            logger.debug(f"[StatusMonitoring] Error getting architecture weights: {e}")
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
                f"[StatusMonitoring] Architecture evaluation recorded: "
                f"{architecture} on {board_type}_{num_players}p -> Elo {elo:.0f} "
                f"(avg: {stats.avg_elo:.0f}, best: {stats.best_elo:.0f})"
            )
        except ImportError:
            logger.debug("[StatusMonitoring] architecture_tracker not available")
        except Exception as e:
            logger.warning(f"[StatusMonitoring] Failed to record architecture evaluation: {e}")

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
            logger.debug("[StatusMonitoring] architecture_tracker not available")
            return 1.0
        except Exception as e:
            logger.debug(f"[StatusMonitoring] Error getting architecture boost: {e}")
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
            logger.debug("[StatusMonitoring] architecture_tracker not available")
            return None
        except Exception as e:
            logger.debug(f"[StatusMonitoring] Error getting best architecture: {e}")
            return None


__all__ = ["StatusMonitoringMixin"]
