"""SelfplayScheduler - Priority-based selfplay allocation across cluster.

This module provides intelligent scheduling of selfplay jobs across the cluster,
prioritizing configurations based on:
- Data staleness (fresher data gets lower priority)
- ELO improvement velocity (fast-improving configs get more resources)
- Training pipeline needs (configs waiting for training get boosted)
- Node capabilities (allocate based on GPU power)

Architecture:
    SelfplayScheduler
    ├── TrainingFreshness: Track data age per config
    ├── ELO Velocity: Track improvement rate
    ├── Node Allocator: Distribute based on capabilities
    └── Event Integration: React to pipeline events

Usage:
    from app.coordination.selfplay_scheduler import (
        SelfplayScheduler,
        get_selfplay_scheduler,
    )

    scheduler = get_selfplay_scheduler()

    # Get priority-ordered configs for selfplay
    priorities = await scheduler.get_priority_configs()

    # Allocate games across nodes
    allocation = await scheduler.allocate_selfplay_batch(games_per_config=500)

    # Start jobs based on allocation
    for config, nodes in allocation.items():
        for node_id, num_games in nodes.items():
            await start_selfplay_job(node_id, config, num_games)

December 2025: Created as part of strategic integration plan.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config.thresholds import (
    GPU_MEMORY_WEIGHTS,
    SELFPLAY_GAMES_PER_NODE,
    is_ephemeral_node,
    get_gpu_weight,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# All supported configurations
ALL_CONFIGS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]

# Priority calculation weights
STALENESS_WEIGHT = 0.35  # Data freshness importance
ELO_VELOCITY_WEIGHT = 0.25  # ELO improvement velocity
TRAINING_NEED_WEIGHT = 0.15  # Waiting for training
EXPLORATION_BOOST_WEIGHT = 0.10  # Feedback loop exploration signal
CURRICULUM_WEIGHT = 0.15  # Curriculum-based priority (Phase 2C.3)

# Staleness thresholds (hours)
FRESH_DATA_THRESHOLD = 1.0  # Data < 1hr old is fresh
STALE_DATA_THRESHOLD = 4.0  # Data > 4hr old is stale
MAX_STALENESS_HOURS = 24.0  # Cap staleness factor

# Default allocation
DEFAULT_GAMES_PER_CONFIG = 500
MIN_GAMES_PER_ALLOCATION = 100


@dataclass
class ConfigPriority:
    """Priority information for a configuration."""
    config_key: str

    # Priority factors
    staleness_hours: float = 0.0
    elo_velocity: float = 0.0  # ELO points per day
    training_pending: bool = False
    exploration_boost: float = 1.0
    curriculum_weight: float = 1.0  # Phase 2C.3: Curriculum-based weight

    # Computed priority
    priority_score: float = 0.0

    # Allocation
    games_allocated: int = 0
    nodes_allocated: list[str] = field(default_factory=list)

    @property
    def staleness_factor(self) -> float:
        """Compute staleness factor (0-1, higher = more stale)."""
        if self.staleness_hours < FRESH_DATA_THRESHOLD:
            return 0.0
        if self.staleness_hours > MAX_STALENESS_HOURS:
            return 1.0
        # Linear interpolation
        return min(1.0, (self.staleness_hours - FRESH_DATA_THRESHOLD) /
                   (STALE_DATA_THRESHOLD - FRESH_DATA_THRESHOLD))

    @property
    def velocity_factor(self) -> float:
        """Compute velocity factor (0-1, higher = faster improvement)."""
        # Positive velocity means improvement, negative means regression
        if self.elo_velocity <= 0:
            return 0.0
        # Cap at 100 ELO/day for factor calculation
        return min(1.0, self.elo_velocity / 100.0)


@dataclass
class NodeCapability:
    """Capability information for a cluster node."""
    node_id: str
    gpu_type: str = "unknown"
    gpu_memory_gb: float = 0.0
    is_ephemeral: bool = False
    current_load: float = 0.0  # 0-1, current utilization
    data_lag_seconds: float = 0.0  # Sync lag from coordinator

    @property
    def capacity_weight(self) -> float:
        """Get capacity weight based on GPU type."""
        return get_gpu_weight(self.gpu_type)

    @property
    def available_capacity(self) -> float:
        """Get available capacity (0-1)."""
        return max(0.0, 1.0 - self.current_load) * self.capacity_weight


class SelfplayScheduler:
    """Priority-based selfplay scheduler across cluster nodes.

    Responsibilities:
    - Track data freshness per configuration
    - Calculate priority scores for each config
    - Allocate selfplay games based on node capabilities
    - Integrate with feedback loop signals
    """

    def __init__(self):
        # Priority tracking
        self._config_priorities: dict[str, ConfigPriority] = {
            cfg: ConfigPriority(config_key=cfg) for cfg in ALL_CONFIGS
        }

        # Node tracking
        self._node_capabilities: dict[str, NodeCapability] = {}

        # Timing
        self._last_priority_update = 0.0
        self._priority_update_interval = 60.0  # Update every minute

        # Event subscription
        self._subscribed = False

        # Lazy dependencies
        self._training_freshness = None
        self._cluster_manifest = None

    # =========================================================================
    # Priority Calculation
    # =========================================================================

    async def get_priority_configs(self, top_n: int = 6) -> list[tuple[str, float]]:
        """Get configs ranked by priority for selfplay allocation.

        Args:
            top_n: Number of top priority configs to return

        Returns:
            List of (config_key, priority_score) tuples, sorted by priority
        """
        await self._update_priorities()

        priorities = [
            (cfg, p.priority_score)
            for cfg, p in self._config_priorities.items()
        ]

        # Sort by priority (descending)
        priorities.sort(key=lambda x: -x[1])

        return priorities[:top_n]

    async def _update_priorities(self) -> None:
        """Update priority scores for all configurations."""
        now = time.time()
        if now - self._last_priority_update < self._priority_update_interval:
            return

        self._last_priority_update = now

        # Get data freshness
        freshness_data = await self._get_data_freshness()

        # Get ELO velocities
        elo_data = await self._get_elo_velocities()

        # Get feedback loop signals
        feedback_data = await self._get_feedback_signals()

        # Get curriculum weights (Phase 2C.3)
        curriculum_data = await self._get_curriculum_weights()

        # Update each config
        for config_key, priority in self._config_priorities.items():
            # Update staleness
            if config_key in freshness_data:
                priority.staleness_hours = freshness_data[config_key]

            # Update ELO velocity
            if config_key in elo_data:
                priority.elo_velocity = elo_data[config_key]

            # Update feedback signals
            if config_key in feedback_data:
                priority.exploration_boost = feedback_data[config_key].get("exploration_boost", 1.0)
                priority.training_pending = feedback_data[config_key].get("training_pending", False)

            # Update curriculum weight (Phase 2C.3)
            if config_key in curriculum_data:
                priority.curriculum_weight = curriculum_data[config_key]

            # Compute priority score
            priority.priority_score = self._compute_priority_score(priority)

        logger.debug(f"[SelfplayScheduler] Updated priorities for {len(self._config_priorities)} configs")

    def _compute_priority_score(self, priority: ConfigPriority) -> float:
        """Compute overall priority score for a configuration.

        Higher score = higher priority for selfplay allocation.

        December 2025 - Phase 2C.3: Now includes curriculum weight factor.
        """
        # Base factors
        staleness = priority.staleness_factor * STALENESS_WEIGHT
        velocity = priority.velocity_factor * ELO_VELOCITY_WEIGHT
        training = (1.0 if priority.training_pending else 0.0) * TRAINING_NEED_WEIGHT
        exploration = (priority.exploration_boost - 1.0) * EXPLORATION_BOOST_WEIGHT

        # Phase 2C.3: Curriculum weight factor (normalized around 1.0)
        # Higher curriculum weight = more data needed for this config
        curriculum = (priority.curriculum_weight - 1.0) * CURRICULUM_WEIGHT

        # Combine factors
        score = staleness + velocity + training + exploration + curriculum

        # Apply exploration boost as multiplier
        score *= priority.exploration_boost

        return score

    async def _get_data_freshness(self) -> dict[str, float]:
        """Get data freshness (hours since last export) per config.

        Returns:
            Dict mapping config_key to hours since last export
        """
        result = {}

        try:
            # Try using TrainingFreshness if available
            if self._training_freshness is None:
                try:
                    from app.coordination.training_freshness import get_training_freshness
                    self._training_freshness = get_training_freshness()
                except ImportError:
                    pass

            if self._training_freshness:
                for config_key in ALL_CONFIGS:
                    staleness = await self._training_freshness.get_staleness(config_key)
                    result[config_key] = staleness
            else:
                # Fallback: check NPZ file modification times
                from pathlib import Path
                now = time.time()

                for config_key in ALL_CONFIGS:
                    parts = config_key.rsplit("_", 1)
                    if len(parts) != 2:
                        continue

                    board_type = parts[0]
                    npz_path = Path(f"data/training/{board_type}_{parts[1]}.npz")

                    if npz_path.exists():
                        mtime = npz_path.stat().st_mtime
                        result[config_key] = (now - mtime) / 3600.0
                    else:
                        result[config_key] = MAX_STALENESS_HOURS

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting freshness: {e}")

        return result

    async def _get_elo_velocities(self) -> dict[str, float]:
        """Get ELO improvement velocities per config.

        Returns:
            Dict mapping config_key to ELO points per day
        """
        result = {}

        try:
            # Try using QueuePopulator's ConfigTarget if available
            from app.coordination.queue_populator import get_queue_populator

            populator = get_queue_populator()
            if populator:
                for config_key, target in populator._configs.items():
                    result[config_key] = target.elo_velocity
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting ELO velocities: {e}")

        return result

    async def _get_feedback_signals(self) -> dict[str, dict[str, Any]]:
        """Get feedback loop signals per config.

        Returns:
            Dict mapping config_key to feedback data
        """
        result = {}

        try:
            from app.coordination.feedback_loop_controller import get_feedback_loop_controller

            controller = get_feedback_loop_controller()
            if controller:
                for config_key in ALL_CONFIGS:
                    state = controller._get_or_create_state(config_key)
                    result[config_key] = {
                        "exploration_boost": state.current_exploration_boost,
                        "training_pending": state.current_training_intensity == "accelerated",
                    }
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting feedback signals: {e}")

        return result

    async def _get_curriculum_weights(self) -> dict[str, float]:
        """Get curriculum weights per config.

        December 2025 - Phase 2C.3: Wire curriculum weights into priority calculation.
        Higher curriculum weight = config needs more training data.

        Returns:
            Dict mapping config_key to curriculum weight (default 1.0)
        """
        result = {}

        try:
            from app.training.curriculum_feedback import get_curriculum_feedback

            feedback = get_curriculum_feedback()
            if feedback:
                # Use get_curriculum_weights() which computes weights from metrics
                # (win rates, Elo trends, weak opponent data) instead of accessing
                # _current_weights directly which may be empty initially
                computed_weights = feedback.get_curriculum_weights()
                if computed_weights:
                    result = computed_weights
                else:
                    # Fall back to manually tracked weights if metrics unavailable
                    result = dict(feedback._current_weights)

        except ImportError:
            logger.debug("[SelfplayScheduler] curriculum_feedback not available")
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error getting curriculum weights: {e}")

        # Ensure all configs have a default weight
        for config_key in ALL_CONFIGS:
            if config_key not in result:
                result[config_key] = 1.0

        return result

    # =========================================================================
    # Node Allocation
    # =========================================================================

    async def allocate_selfplay_batch(
        self,
        games_per_config: int = DEFAULT_GAMES_PER_CONFIG,
        max_configs: int = 6,
    ) -> dict[str, dict[str, int]]:
        """Allocate selfplay games across cluster nodes.

        Args:
            games_per_config: Target games per config
            max_configs: Maximum configs to allocate

        Returns:
            Dict mapping config_key to {node_id: num_games}
        """
        # Get priority configs
        priorities = await self.get_priority_configs(top_n=max_configs)

        # Get available nodes
        await self._update_node_capabilities()

        allocation: dict[str, dict[str, int]] = {}

        for config_key, priority_score in priorities:
            if priority_score <= 0:
                continue

            # Allocate based on priority weight
            config_games = int(games_per_config * (0.5 + priority_score))
            config_games = max(MIN_GAMES_PER_ALLOCATION, config_games)

            # Distribute across nodes
            node_allocation = self._allocate_to_nodes(config_key, config_games)
            if node_allocation:
                allocation[config_key] = node_allocation

                # Update priority tracking
                priority = self._config_priorities[config_key]
                priority.games_allocated = sum(node_allocation.values())
                priority.nodes_allocated = list(node_allocation.keys())

        logger.info(
            f"[SelfplayScheduler] Allocated {len(allocation)} configs: "
            f"{', '.join(f'{k}={sum(v.values())}' for k, v in allocation.items())}"
        )

        return allocation

    def _allocate_to_nodes(
        self,
        config_key: str,
        total_games: int,
    ) -> dict[str, int]:
        """Allocate games for a config across available nodes.

        Args:
            config_key: Configuration key
            total_games: Total games to allocate

        Returns:
            Dict mapping node_id to num_games
        """
        # Get available nodes sorted by capacity
        available_nodes = sorted(
            [n for n in self._node_capabilities.values() if n.available_capacity > 0.1],
            key=lambda n: (-n.available_capacity, n.data_lag_seconds),
        )

        if not available_nodes:
            return {}

        allocation: dict[str, int] = {}
        remaining = total_games

        # Calculate total capacity
        total_capacity = sum(n.available_capacity for n in available_nodes)

        for node in available_nodes:
            if remaining <= 0:
                break

            # Proportional allocation based on capacity
            proportion = node.available_capacity / total_capacity
            node_games = max(
                MIN_GAMES_PER_ALLOCATION,
                int(total_games * proportion),
            )

            # Adjust based on GPU type
            gpu_games = SELFPLAY_GAMES_PER_NODE.get(node.gpu_type, 500)
            node_games = min(node_games, gpu_games)

            # Cap at remaining games
            node_games = min(node_games, remaining)

            if node_games >= MIN_GAMES_PER_ALLOCATION:
                allocation[node.node_id] = node_games
                remaining -= node_games

        return allocation

    async def _update_node_capabilities(self) -> None:
        """Update node capability information from cluster."""
        try:
            # Try getting from cluster monitor
            from app.distributed.cluster_monitor import ClusterMonitor

            monitor = ClusterMonitor()
            status = monitor.get_cluster_status()

            for node_id, node_status in status.nodes.items():
                if node_id not in self._node_capabilities:
                    self._node_capabilities[node_id] = NodeCapability(node_id=node_id)

                cap = self._node_capabilities[node_id]
                cap.gpu_type = node_status.gpu or "unknown"
                cap.gpu_memory_gb = node_status.gpu_memory_total_gb
                cap.is_ephemeral = is_ephemeral_node(node_id)
                cap.current_load = node_status.gpu_utilization_percent / 100.0
                cap.data_lag_seconds = node_status.sync_lag_seconds

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error updating node capabilities: {e}")

    # =========================================================================
    # Event Integration
    # =========================================================================

    def subscribe_to_events(self) -> None:
        """Subscribe to relevant pipeline events."""
        if self._subscribed:
            return

        try:
            from app.coordination.event_router import DataEventType, get_event_bus

            bus = get_event_bus()
            if bus:
                bus.subscribe(DataEventType.SELFPLAY_COMPLETE, self._on_selfplay_complete)
                bus.subscribe(DataEventType.TRAINING_COMPLETED, self._on_training_complete)
                bus.subscribe(DataEventType.PROMOTION_COMPLETE, self._on_promotion_complete)
                self._subscribed = True
                logger.info("[SelfplayScheduler] Subscribed to pipeline events")

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Failed to subscribe: {e}")

    def _on_selfplay_complete(self, event: Any) -> None:
        """Handle selfplay completion event."""
        try:
            config_key = event.payload.get("config_key", "")
            if config_key in self._config_priorities:
                # Reset staleness for this config
                self._config_priorities[config_key].staleness_hours = 0.0
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling selfplay complete: {e}")

    def _on_training_complete(self, event: Any) -> None:
        """Handle training completion event."""
        try:
            config_key = event.payload.get("config_key", "")
            if config_key in self._config_priorities:
                # Clear training pending flag
                self._config_priorities[config_key].training_pending = False
        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error handling training complete: {e}")

    def _on_promotion_complete(self, event: Any) -> None:
        """Handle promotion completion event."""
        try:
            config_key = event.payload.get("config_key", "")
            success = event.payload.get("success", False)

            if config_key in self._config_priorities:
                priority = self._config_priorities[config_key]
                if success:
                    # Boost for continued improvement
                    priority.exploration_boost = 1.0
                else:
                    # Increase exploration on failure
                    priority.exploration_boost = min(2.0, priority.exploration_boost * 1.3)
        except Exception:
            pass

    # =========================================================================
    # Status & Metrics
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status."""
        return {
            "subscribed": self._subscribed,
            "last_priority_update": self._last_priority_update,
            "node_count": len(self._node_capabilities),
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


# =============================================================================
# Singleton
# =============================================================================

_scheduler_instance: SelfplayScheduler | None = None


def get_selfplay_scheduler() -> SelfplayScheduler:
    """Get the singleton SelfplayScheduler instance."""
    global _scheduler_instance

    if _scheduler_instance is None:
        _scheduler_instance = SelfplayScheduler()
        _scheduler_instance.subscribe_to_events()

    return _scheduler_instance


def reset_selfplay_scheduler() -> None:
    """Reset the scheduler singleton (for testing)."""
    global _scheduler_instance
    _scheduler_instance = None
