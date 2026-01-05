"""AllocationEngine - Selfplay allocation logic extracted from SelfplayScheduler.

January 2026 Sprint 17.9: Extracted ~450 LOC of allocation logic to improve
testability and reduce SelfplayScheduler complexity (3,156 LOC -> ~2,700 LOC).

This module provides:
- AllocationContext: Runtime context for allocation decisions
- AllocationEngine: Stateless service for computing game allocations

Design principles:
- Pure functions where possible (easier to test)
- Dependency injection for external dependencies
- AllocationEngine receives snapshots, not live references
- Results are returned, not mutated in place

Usage:
    from app.coordination.selfplay.allocation_engine import (
        AllocationContext,
        AllocationEngine,
    )

    engine = AllocationEngine(
        config_priorities=scheduler._config_priorities,
        node_capabilities=scheduler._node_capabilities,
        metrics_collector=scheduler._metrics_collector,
    )

    context = AllocationContext(
        unhealthy_nodes=scheduler._unhealthy_nodes,
        cluster_health_factor=scheduler._cluster_health_factor,
        backpressure_signal=await backpressure_monitor.get_signal(),
    )

    allocation = engine.allocate_selfplay_batch(
        games_per_config=500,
        max_configs=10,
        context=context,
    )
"""

from __future__ import annotations

__all__ = [
    "AllocationContext",
    "AllocationEngine",
    "AllocationResult",
]

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from app.config.coordination_defaults import (
    SelfplayAllocationDefaults,
    SelfplayPriorityWeightDefaults,
)
from app.config.thresholds import SELFPLAY_GAMES_PER_NODE

# Get starvation thresholds from centralized config
_priority_weight_defaults = SelfplayPriorityWeightDefaults()
DATA_STARVATION_ULTRA_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_ULTRA_THRESHOLD
DATA_STARVATION_EMERGENCY_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_EMERGENCY_THRESHOLD
DATA_STARVATION_CRITICAL_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_CRITICAL_THRESHOLD
DATA_POVERTY_THRESHOLD = _priority_weight_defaults.DATA_POVERTY_THRESHOLD

if TYPE_CHECKING:
    from app.coordination.backpressure import BackpressureSignal
    from app.coordination.node_allocator import NodeCapability
    from app.coordination.scheduler_metrics import SchedulerMetricsCollector
    from app.coordination.selfplay_priority_types import ConfigPriority


logger = logging.getLogger(__name__)


# =============================================================================
# Constants (from coordination_defaults)
# =============================================================================

MIN_GAMES_PER_ALLOCATION = SelfplayAllocationDefaults.MIN_GAMES_PER_ALLOCATION


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AllocationContext:
    """Runtime context for allocation decisions.

    Contains cluster-wide state that affects allocation:
    - Node health status
    - Cluster health factor (scaling for degraded cluster)
    - Backpressure signal (throttling based on queue depth)

    This is passed to AllocationEngine methods as a snapshot of current state,
    avoiding direct coupling to SelfplayScheduler's mutable state.
    """

    # Nodes to exclude from allocation
    unhealthy_nodes: set[str] = field(default_factory=set)

    # Cluster-wide health factor (0.0-1.0, lower = more degraded)
    cluster_health_factor: float = 1.0

    # Current backpressure signal (optional, for throttling)
    backpressure_signal: Optional[BackpressureSignal] = None

    # Demoted nodes (reduced allocation weight)
    demoted_nodes: set[str] = field(default_factory=set)

    # Whether to apply 4-player allocation minimums
    enforce_4p_minimums: bool = True

    # Timestamp for freshness
    timestamp: float = field(default_factory=time.time)

    @property
    def should_throttle(self) -> bool:
        """Check if backpressure indicates throttling needed."""
        if self.backpressure_signal is None:
            return False
        return self.backpressure_signal.spawn_rate_multiplier < 1.0

    @property
    def throttle_factor(self) -> float:
        """Get throttle factor from backpressure (0.0-1.0)."""
        if self.backpressure_signal is None:
            return 1.0
        return self.backpressure_signal.spawn_rate_multiplier


@dataclass
class AllocationResult:
    """Result of a batch allocation operation.

    Contains the allocation mapping plus metadata about the allocation.
    """

    # config_key -> {node_id: games}
    allocations: dict[str, dict[str, int]] = field(default_factory=dict)

    # Total games allocated across all configs
    total_games: int = 0

    # Number of configs allocated to
    configs_allocated: int = 0

    # Number of nodes involved
    nodes_involved: int = 0

    # What triggered this allocation
    trigger: str = "allocate_batch"

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    # Whether backpressure was active
    backpressure_active: bool = False

    # Applied throttle factor
    throttle_factor: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for event payload."""
        return {
            "allocations": self.allocations,
            "total_games": self.total_games,
            "configs_allocated": self.configs_allocated,
            "nodes_involved": self.nodes_involved,
            "trigger": self.trigger,
            "timestamp": self.timestamp,
            "backpressure_active": self.backpressure_active,
            "throttle_factor": self.throttle_factor,
        }


class AllocationEngine:
    """Stateless service for computing selfplay game allocations.

    Receives snapshots of priorities and capabilities, computes allocations,
    and returns results. Does not modify any external state.

    January 2026 Sprint 17.9: Extracted from SelfplayScheduler to improve
    testability and reduce complexity.

    Methods to implement (Phase 2-5):
    - _steal_from_donors(): Pure function for redistributing games
    - _enforce_starvation_floor(): Ensure data-starved configs get allocation
    - _enforce_4p_allocation_minimums(): Redistribute from 2p to 3p/4p
    - _allocate_to_nodes(): Distribute games for a config across nodes
    - allocate_selfplay_batch(): Main entry point
    """

    def __init__(
        self,
        config_priorities: dict[str, ConfigPriority],
        node_capabilities: dict[str, NodeCapability],
        metrics_collector: Optional[SchedulerMetricsCollector] = None,
        # Optional event emission callback
        emit_event_fn: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ):
        """Initialize AllocationEngine with snapshots of current state.

        Args:
            config_priorities: Snapshot of config priorities (config_key -> ConfigPriority)
            node_capabilities: Snapshot of node capabilities (node_id -> NodeCapability)
            metrics_collector: Optional metrics collector for recording allocations
            emit_event_fn: Optional callback for emitting events (event_name, payload)
        """
        self._config_priorities = config_priorities
        self._node_capabilities = node_capabilities
        self._metrics_collector = metrics_collector
        self._emit_event_fn = emit_event_fn

    # =========================================================================
    # Main Entry Point (Phase 5)
    # =========================================================================

    def allocate_selfplay_batch(
        self,
        priorities: list[tuple[str, float]],
        games_per_config: int,
        context: AllocationContext,
    ) -> AllocationResult:
        """Allocate selfplay games across configs and nodes.

        This is the main entry point for batch allocation. It:
        1. Applies backpressure throttling if needed
        2. Allocates games to each config based on priority
        3. Enforces starvation floor for data-starved configs
        4. Enforces 4-player allocation minimums
        5. Emits allocation event

        Note: Priority calculation is done by the caller (SelfplayScheduler).
        This engine focuses on allocation logic.

        Args:
            priorities: List of (config_key, priority_score) tuples
            games_per_config: Base games per config
            context: Runtime allocation context

        Returns:
            AllocationResult with config -> node -> games mapping
        """
        # Apply backpressure throttling
        effective_games_per_config = games_per_config
        if context.should_throttle:
            effective_games_per_config = max(
                MIN_GAMES_PER_ALLOCATION,
                int(games_per_config * context.throttle_factor),
            )
            logger.info(
                f"[AllocationEngine] Backpressure scaling: {games_per_config} -> "
                f"{effective_games_per_config} (throttle={context.throttle_factor:.2f})"
            )

        allocation: dict[str, dict[str, int]] = {}
        skipped_configs: list[tuple[str, float]] = []

        # Allocate based on priorities
        for config_key, priority_score in priorities:
            if priority_score <= 0:
                skipped_configs.append((config_key, priority_score))
                continue

            # Allocate based on priority weight
            config_games = int(effective_games_per_config * (0.5 + priority_score))
            config_games = max(MIN_GAMES_PER_ALLOCATION, config_games)

            # Distribute across nodes
            node_allocation = self.allocate_to_nodes(config_key, config_games, context)
            if node_allocation:
                allocation[config_key] = node_allocation

                # Update priority tracking in config_priorities
                if config_key in self._config_priorities:
                    priority = self._config_priorities[config_key]
                    priority.games_allocated = sum(node_allocation.values())
                    priority.nodes_allocated = list(node_allocation.keys())

        # Enforce starvation floor allocations
        allocation = self.enforce_starvation_floor(
            allocation, effective_games_per_config, context
        )

        # Enforce 4p allocation minimums
        allocation = self.enforce_4p_allocation_minimums(
            allocation, effective_games_per_config, context
        )

        # Calculate totals
        total_games = sum(
            sum(node_games.values()) for node_games in allocation.values()
        )
        nodes_involved: set[str] = set()
        for node_games in allocation.values():
            nodes_involved.update(node_games.keys())

        # Record metrics
        self._record_allocation(total_games)

        # Build result
        result = AllocationResult(
            allocations=allocation,
            total_games=total_games,
            configs_allocated=len(allocation),
            nodes_involved=len(nodes_involved),
            trigger="allocate_batch",
            backpressure_active=context.should_throttle,
            throttle_factor=context.throttle_factor,
        )

        # Log allocation summary
        if allocation:
            logger.info(
                f"[AllocationEngine] Allocated {len(allocation)} configs: "
                f"{', '.join(f'{k}={sum(v.values())}' for k, v in allocation.items())}"
            )

        # Log skipped configs
        if skipped_configs:
            logger.warning(
                f"[AllocationEngine] {len(skipped_configs)} configs skipped (priority <= 0): "
                f"{', '.join(f'{k}({s:.3f})' for k, s in skipped_configs)}"
            )

        # Emit event
        if total_games > 0:
            self._emit_allocation_updated(result)

        return result

    # =========================================================================
    # Starvation Floor Enforcement (Phase 3)
    # =========================================================================

    def enforce_starvation_floor(
        self,
        allocation: dict[str, dict[str, int]],
        games_per_config: int,
        context: AllocationContext,
    ) -> dict[str, dict[str, int]]:
        """Enforce minimum allocation for data-starved configs.

        Dec 29, 2025: Extracted from SelfplayScheduler._enforce_starvation_floor().

        Configs with very few games (ULTRA, EMERGENCY, CRITICAL, POVERTY tiers)
        get guaranteed minimum allocation to prevent permanent starvation.

        Tiers:
        - ULTRA (<20 games): 3x allocation
        - EMERGENCY (<100 games): 2x allocation
        - CRITICAL (<1000 games): 1.5x allocation
        - POVERTY (<5000 games): 1.25x allocation

        Args:
            allocation: Nested dict config_key -> {node_id: games}
            games_per_config: Base games per config
            context: Runtime allocation context

        Returns:
            Updated allocation with starvation floor enforced
        """
        added_configs = []

        for config_key, priority in self._config_priorities.items():
            game_count = priority.game_count

            # Calculate current allocation for this config
            current_allocation = sum(
                allocation.get(config_key, {}).values()
            )

            # Determine minimum floor based on starvation level
            if game_count < DATA_STARVATION_ULTRA_THRESHOLD:
                # ULTRA: <20 games - 3x allocation (highest priority)
                min_floor = int(games_per_config * 3.0)
                level = "ULTRA"
            elif game_count < DATA_STARVATION_EMERGENCY_THRESHOLD:
                # EMERGENCY: <100 games - 2x allocation
                min_floor = int(games_per_config * 2.0)
                level = "EMERGENCY"
            elif game_count < DATA_STARVATION_CRITICAL_THRESHOLD:
                # CRITICAL: <1000 games - 1.5x allocation
                min_floor = int(games_per_config * 1.5)
                level = "CRITICAL"
            elif game_count < DATA_POVERTY_THRESHOLD:
                # POVERTY: <5000 games - 1.25x allocation
                min_floor = int(games_per_config * 1.25)
                level = "POVERTY"
            else:
                # Not starved, no floor needed
                continue

            if current_allocation >= min_floor:
                continue  # Already meets floor

            # Need to allocate more for this starved config
            shortfall = min_floor - current_allocation

            # Allocate shortfall to available nodes
            additional_allocation = self.allocate_to_nodes(
                config_key, shortfall, context
            )

            if additional_allocation:
                if config_key not in allocation:
                    allocation[config_key] = {}

                # Merge with existing allocation
                for node_id, games in additional_allocation.items():
                    allocation[config_key][node_id] = (
                        allocation[config_key].get(node_id, 0) + games
                    )

                added_configs.append(
                    f"{config_key}({level}:{game_count}g->+{shortfall})"
                )
                logger.warning(
                    f"[AllocationEngine] Starvation floor: {config_key} "
                    f"({level}, {game_count} games) allocated +{shortfall} games"
                )

        if added_configs:
            logger.info(
                f"[AllocationEngine] Starvation floor enforcement: {', '.join(added_configs)}"
            )

        return allocation

    # =========================================================================
    # 4-Player Allocation Enforcement (Phase 3)
    # =========================================================================

    def enforce_4p_allocation_minimums(
        self,
        allocation: dict[str, dict[str, int]],
        games_per_config: int,
        context: AllocationContext,
    ) -> dict[str, dict[str, int]]:
        """Enforce minimum allocation for 3-player and 4-player configs.

        Dec 30, 2025: Extracted from SelfplayScheduler._enforce_4p_allocation_minimums().

        Redistributes games from 2-player configs (which tend to dominate)
        to ensure 3p and 4p configs get adequate training data.

        Targets:
        - 4p configs get at least 2.5x their proportional share
        - 3p configs get at least 1.5x their proportional share

        Args:
            allocation: Nested dict config_key -> {node_id: games}
            games_per_config: Base games per config
            context: Runtime allocation context

        Returns:
            Updated allocation with 4p minimums enforced
        """
        if not context.enforce_4p_minimums or not allocation:
            return allocation

        # Calculate per-config totals
        totals: dict[str, int] = {}
        for config_key, node_alloc in allocation.items():
            totals[config_key] = sum(node_alloc.values())

        # Identify configs by player count
        four_p_configs = [c for c in totals if "_4p" in c]
        three_p_configs = [c for c in totals if "_3p" in c]
        two_p_configs = [c for c in totals if "_2p" in c]

        if not two_p_configs:
            return allocation  # No donors available

        # Dec 30, 2025: Aggressive targets for multiplayer
        # 4p: 2.5x (these have fewest games)
        # 3p: 1.5x (also underrepresented)
        min_4p_games = int(games_per_config * 2.5)
        min_3p_games = int(games_per_config * 1.5)
        redistributed = 0

        # Process 4p first (higher priority - most starved)
        for config in four_p_configs:
            current = totals.get(config, 0)
            shortfall = max(0, min_4p_games - current)

            if shortfall > 0:
                stolen = self.steal_from_donors(
                    allocation, totals, config, shortfall, two_p_configs, games_per_config
                )
                redistributed += stolen
                if stolen > 0:
                    logger.info(
                        f"[AllocationEngine] 4p enforcement: boosted {config} by {stolen} games "
                        f"(now {totals.get(config, 0)})"
                    )

        # Process 3p next (lower priority than 4p)
        for config in three_p_configs:
            current = totals.get(config, 0)
            shortfall = max(0, min_3p_games - current)

            if shortfall > 0:
                stolen = self.steal_from_donors(
                    allocation, totals, config, shortfall, two_p_configs, games_per_config
                )
                redistributed += stolen
                if stolen > 0:
                    logger.info(
                        f"[AllocationEngine] 3p enforcement: boosted {config} by {stolen} games "
                        f"(now {totals.get(config, 0)})"
                    )

        if redistributed > 0:
            logger.warning(
                f"[AllocationEngine] Multiplayer allocation enforcement: "
                f"redistributed {redistributed} games from 2p -> 3p/4p configs"
            )

        return allocation

    # =========================================================================
    # Steal From Donors (Phase 2 - Pure Function)
    # =========================================================================

    @staticmethod
    def steal_from_donors(
        allocation: dict[str, dict[str, int]],
        totals: dict[str, int],
        recipient: str,
        shortfall: int,
        donors: list[str],
        games_per_config: int,
    ) -> int:
        """Redistribute games from donor configs to recipient.

        Pure function that mutates the allocation and totals dicts to
        redistribute games from donors to the recipient config.

        Dec 30, 2025: Extracted from SelfplayScheduler._steal_from_donors().
        2p configs keep at least 40% (was 50%) to allow more redistribution.

        Args:
            allocation: Nested dict config_key -> {node_id: games} (mutated)
            totals: Running totals for configs (mutated)
            recipient: Config key to receive games
            shortfall: Games needed to meet minimum
            donors: List of config keys to steal from (priority order)
            games_per_config: Base games per config (for calculating floor)

        Returns:
            Number of games stolen (added to recipient)
        """
        stolen_total = 0

        for donor in donors:
            if shortfall <= 0:
                break

            donor_current = totals.get(donor, 0)
            # Dec 30, 2025: 2p keeps at least 40% to allow more redistribution
            donor_min = int(games_per_config * 0.4)

            available = max(0, donor_current - donor_min)
            steal = min(shortfall, available)

            if steal > 0 and donor in allocation:
                # Remove from donor (proportionally from nodes)
                for node_id in allocation[donor]:
                    node_games = allocation[donor][node_id]
                    if donor_current > 0:
                        node_steal = int(steal * node_games / donor_current)
                        allocation[donor][node_id] = max(0, node_games - node_steal)

                # Add to recipient config (to first available node)
                if recipient in allocation and allocation[recipient]:
                    first_node = next(iter(allocation[recipient]))
                    allocation[recipient][first_node] += steal
                elif recipient not in allocation:
                    # Assign to a node from the donor
                    if allocation[donor]:
                        first_donor_node = next(iter(allocation[donor]))
                        allocation[recipient] = {first_donor_node: steal}

                totals[donor] -= steal
                totals[recipient] = totals.get(recipient, 0) + steal
                shortfall -= steal
                stolen_total += steal

                logger.debug(
                    f"[AllocationEngine] Multiplayer enforcement: "
                    f"{donor} -> {recipient}: {steal} games"
                )

        return stolen_total

    # =========================================================================
    # Node Allocation (Phase 4)
    # =========================================================================

    def allocate_to_nodes(
        self,
        config_key: str,
        total_games: int,
        context: AllocationContext,
    ) -> dict[str, int]:
        """Allocate games for a config across available nodes.

        Distributes games based on:
        - Node capacity (GPU type, available resources)
        - Ephemeral node status (boost for short jobs, reduce for long jobs)
        - Cluster health factor
        - Unhealthy node exclusion

        Args:
            config_key: Configuration to allocate for
            total_games: Total games to distribute
            context: Runtime allocation context

        Returns:
            Dict mapping node_id to num_games
        """
        # Apply cluster health factor
        if context.cluster_health_factor < 1.0:
            adjusted_games = max(
                MIN_GAMES_PER_ALLOCATION,
                int(total_games * context.cluster_health_factor),
            )
            logger.debug(
                f"[AllocationEngine] Cluster health {context.cluster_health_factor:.2f} "
                f"reducing allocation: {total_games} -> {adjusted_games} games"
            )
            total_games = adjusted_games

        # Get available nodes (excluding unhealthy)
        available_nodes = sorted(
            [
                n for n in self._node_capabilities.values()
                if n.available_capacity > 0.1
                and n.node_id not in context.unhealthy_nodes
            ],
            key=lambda n: (-n.available_capacity, n.data_lag_seconds),
        )

        if not available_nodes:
            logger.debug(
                f"[AllocationEngine] No nodes available for {config_key}: "
                f"unhealthy={len(context.unhealthy_nodes)}"
            )
            return {}

        allocation: dict[str, int] = {}
        remaining = total_games

        # Phase 2B.4: Determine if this is a short job
        is_short_job = config_key.startswith(("square8", "hex8"))

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

            # Adjust based on GPU type limits
            gpu_limit = SELFPLAY_GAMES_PER_NODE.get(node.gpu_type, 500)
            node_games = min(node_games, gpu_limit)

            # Ephemeral node adjustment
            if node.is_ephemeral:
                if is_short_job:
                    node_games = int(node_games * 1.5)
                    logger.debug(
                        f"[AllocationEngine] Boosted {node.node_id} (ephemeral) "
                        f"for short job {config_key}"
                    )
                else:
                    node_games = int(node_games * 0.5)
                    logger.debug(
                        f"[AllocationEngine] Reduced {node.node_id} (ephemeral) "
                        f"for long job {config_key}"
                    )

            # Cap at remaining games
            node_games = min(node_games, remaining)

            if node_games >= MIN_GAMES_PER_ALLOCATION:
                allocation[node.node_id] = node_games
                remaining -= node_games

        # Log allocation breakdown
        if allocation and logger.isEnabledFor(logging.DEBUG):
            total_assigned = sum(allocation.values())
            logger.debug(
                f"[AllocationEngine] Node allocation for {config_key}: "
                f"{total_assigned}/{total_games} games across {len(allocation)} nodes"
            )

        return allocation

    # =========================================================================
    # Metrics and Events (Phase 7)
    # =========================================================================

    def _record_allocation(self, games_allocated: int) -> None:
        """Record allocation metrics."""
        if self._metrics_collector is not None:
            self._metrics_collector.record_allocation(games_allocated)

    def _emit_allocation_updated(
        self,
        result: AllocationResult,
    ) -> None:
        """Emit SELFPLAY_ALLOCATION_UPDATED event."""
        if self._emit_event_fn is None:
            return

        try:
            nodes_involved: set[str] = set()
            for node_games in result.allocations.values():
                nodes_involved.update(node_games.keys())

            payload = {
                "trigger": result.trigger,
                "total_games": result.total_games,
                "configs": list(result.allocations.keys()),
                "node_count": len(nodes_involved),
                "timestamp": result.timestamp,
                "backpressure_active": result.backpressure_active,
                "throttle_factor": result.throttle_factor,
            }

            self._emit_event_fn("SELFPLAY_ALLOCATION_UPDATED", payload)
            logger.debug(
                f"[AllocationEngine] Emitted SELFPLAY_ALLOCATION_UPDATED: "
                f"games={result.total_games}, configs={result.configs_allocated}"
            )
        except Exception as e:
            logger.debug(f"[AllocationEngine] Failed to emit allocation update: {e}")
