"""Node allocation and starvation enforcement mixin for SelfplayScheduler.

Sprint 17.9+ (Feb 2026): Extracted from selfplay_scheduler.py to reduce file size.

Provides methods for:
- _allocate_to_nodes: Distribute games across available nodes
- _enforce_starvation_floor: Ensure minimum allocation for starved configs
- _enforce_4p_allocation_minimums: Ensure multiplayer configs get adequate allocation
- _steal_from_donors: Redistribute games from 2p to 3p/4p configs
- _update_node_capabilities: Refresh node capability data from cluster
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.config.thresholds import (
    SELFPLAY_GAMES_PER_NODE,
    get_gpu_cost_per_hour,
    is_ephemeral_node,
)
from app.coordination.node_allocator import NodeCapability
from app.coordination.priority_calculator import ALL_CONFIGS

# January 5, 2026 (Phase 7.4): Node circuit breaker for work allocation filtering
from app.coordination.node_circuit_breaker import get_node_circuit_registry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AllocationMixin:
    """Mixin providing node allocation and starvation enforcement.

    Expects the host class to have:
    - _node_capabilities: dict[str, NodeCapability]
    - _config_priorities: dict[str, ConfigPriority]
    - _last_node_capability_update: float
    - _node_capability_update_interval: float
    - _is_selfplay_enabled(node_id): method from NodeTargetingMixin
    - _safe_emit_event(event_name, payload): method from HandlerBase
    """

    def _enforce_starvation_floor(
        self,
        allocation: dict[str, dict[str, int]],
        games_per_config: int,
    ) -> dict[str, dict[str, int]]:
        """Enforce minimum allocation for data-starved configs.

        Dec 29, 2025 - Phase 5: Data starvation floor enforcement.
        Any config with <100 games MUST receive allocation, even if it wasn't
        in the initial priority list. This addresses critical gaps like:
        - square19_4p: 1 game
        - hexagonal_4p: 400 games

        For emergency configs (<100 games): allocate 2x base games
        For critical configs (<1000 games): allocate 1.5x base games

        Args:
            allocation: Current allocation {config_key: {node_id: games}}
            games_per_config: Base games per config

        Returns:
            Adjusted allocation with starvation floors enforced
        """
        from app.coordination.selfplay_scheduler import (
            DATA_STARVATION_ULTRA_THRESHOLD,
            DATA_STARVATION_EMERGENCY_THRESHOLD,
            DATA_STARVATION_CRITICAL_THRESHOLD,
            DATA_POVERTY_THRESHOLD,
        )

        added_configs = []

        for config_key in ALL_CONFIGS:
            priority = self._config_priorities.get(config_key)
            if not priority:
                continue

            game_count = priority.game_count

            # Skip if already adequately allocated
            current_allocation = sum(
                allocation.get(config_key, {}).values()
            )

            # Determine minimum floor based on starvation level
            # Dec 29, 2025: Added ULTRA tier for critically starved configs
            if game_count < DATA_STARVATION_ULTRA_THRESHOLD:
                # ULTRA: <20 games - must get 3x allocation (highest priority)
                min_floor = int(games_per_config * 3.0)
                level = "ULTRA"
            elif game_count < DATA_STARVATION_EMERGENCY_THRESHOLD:
                # EMERGENCY: <100 games - must get 2x allocation
                min_floor = int(games_per_config * 2.0)
                level = "EMERGENCY"
            elif game_count < DATA_STARVATION_CRITICAL_THRESHOLD:
                # CRITICAL: <1000 games - must get 1.5x allocation
                min_floor = int(games_per_config * 1.5)
                level = "CRITICAL"
            elif game_count < DATA_POVERTY_THRESHOLD:
                # POVERTY (Dec 30, 2025): <5000 games - must get 1.25x allocation
                min_floor = int(games_per_config * 1.25)
                level = "POVERTY"
            else:
                # Not starved, no floor needed
                continue

            if current_allocation >= min_floor:
                continue  # Already meets floor

            # Need to allocate more for this starved config
            shortfall = min_floor - current_allocation

            # Allocate to available nodes
            additional_allocation = self._allocate_to_nodes(config_key, shortfall)

            # Feb 2026: If no nodes available with strict filtering, retry with
            # circuit breakers ignored. Starving configs must get allocation even
            # if nodes had recent failures - those nodes may have recovered.
            if not additional_allocation and level in ("ULTRA", "EMERGENCY", "CRITICAL"):
                logger.warning(
                    f"[SelfplayScheduler] Starvation floor: {config_key} "
                    f"({level}, {game_count} games) - no nodes available with "
                    f"strict filtering, retrying with CB nodes included"
                )
                additional_allocation = self._allocate_to_nodes(
                    config_key, shortfall, ignore_circuit_breakers=True,
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
                    f"[SelfplayScheduler] Starvation floor: {config_key} "
                    f"({level}, {game_count} games) allocated +{shortfall} games"
                )
            elif level in ("ULTRA", "EMERGENCY"):
                # Feb 2026: Emit event for unresolvable starvation
                logger.error(
                    f"[SelfplayScheduler] STARVATION UNRESOLVED: {config_key} "
                    f"({level}, {game_count} games) - no nodes available even "
                    f"with relaxed filters. Manual intervention may be needed."
                )
                self._safe_emit_event(
                    "STARVATION_UNRESOLVED",
                    {
                        "config_key": config_key,
                        "level": level,
                        "game_count": game_count,
                        "shortfall": shortfall,
                    },
                )

        if added_configs:
            logger.info(
                f"[SelfplayScheduler] Starvation floor enforcement: {', '.join(added_configs)}"
            )

        return allocation

    def _enforce_4p_allocation_minimums(
        self,
        allocation: dict[str, dict[str, int]],
        games_per_config: int,
    ) -> dict[str, dict[str, int]]:
        """Enforce minimum allocations for 3-player and 4-player configs.

        Dec 29, 2025 - Phase 4: Multiplayer allocation enforcement.
        Dec 30, 2025: Extended to 3p configs + increased targets.

        The 3p/4p multipliers can't be satisfied if cluster has limited GPUs.
        This method redistributes from 2p configs to ensure multiplayer gets minimum.

        Targets (Dec 30, 2025 - more aggressive):
        - 4p configs get at least 5.0x their proportional share
        - 3p configs get at least 2.5x their proportional share
        If short, steal from 2p configs (which are easiest to generate).

        Args:
            allocation: Current allocation {config_key: {node_id: games}}
            games_per_config: Base games per config

        Returns:
            Adjusted allocation with enforced multiplayer minimums
        """
        if not allocation:
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

        # Feb 2026: Reduced multiplayer targets since 4p configs already beat
        # heuristic (hex8_4p +243, square8_4p +227). Shift focus to 2p donors.
        min_4p_games = int(games_per_config * 3.0)
        min_3p_games = int(games_per_config * 1.5)
        redistributed = 0

        # Process 4p first (higher priority - most starved)
        for config in four_p_configs:
            current = totals.get(config, 0)
            shortfall = max(0, min_4p_games - current)

            if shortfall > 0:
                stolen = self._steal_from_donors(
                    allocation, totals, config, shortfall, two_p_configs, games_per_config
                )
                redistributed += stolen
                if stolen > 0:
                    logger.info(
                        f"[SelfplayScheduler] 4p enforcement: boosted {config} by {stolen} games "
                        f"(now {totals.get(config, 0)})"
                    )

        # Process 3p next (lower priority than 4p)
        for config in three_p_configs:
            current = totals.get(config, 0)
            shortfall = max(0, min_3p_games - current)

            if shortfall > 0:
                stolen = self._steal_from_donors(
                    allocation, totals, config, shortfall, two_p_configs, games_per_config
                )
                redistributed += stolen
                if stolen > 0:
                    logger.info(
                        f"[SelfplayScheduler] 3p enforcement: boosted {config} by {stolen} games "
                        f"(now {totals.get(config, 0)})"
                    )

        if redistributed > 0:
            logger.warning(
                f"[SelfplayScheduler] Multiplayer allocation enforcement: "
                f"redistributed {redistributed} games from 2p -> 3p/4p configs"
            )

        return allocation

    def _steal_from_donors(
        self,
        allocation: dict[str, dict[str, int]],
        totals: dict[str, int],
        recipient: str,
        shortfall: int,
        donors: list[str],
        games_per_config: int,
    ) -> int:
        """Steal games from donor configs to fill recipient shortfall.

        Dec 30, 2025: Extracted helper for 3p/4p enforcement.

        Args:
            allocation: Current allocation dict
            totals: Current totals per config
            recipient: Config to receive games
            shortfall: Games needed
            donors: Configs to steal from
            games_per_config: Base allocation

        Returns:
            Number of games stolen
        """
        stolen_total = 0

        for donor in donors:
            if shortfall <= 0:
                break

            donor_current = totals.get(donor, 0)
            # Jan 14, 2026: Reduced to 20% to allow aggressive redistribution to 4p
            # 4p configs are at 5% of 2p levels and need urgent catchup
            donor_min = int(games_per_config * 0.2)

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
                    # Find a node that had this config's board type
                    allocation[recipient] = {list(allocation[donor].keys())[0]: steal}

                totals[donor] -= steal
                totals[recipient] = totals.get(recipient, 0) + steal
                shortfall -= steal
                stolen_total += steal

                logger.debug(
                    f"[SelfplayScheduler] Multiplayer enforcement: {donor} -> {recipient}: {steal} games"
                )

        return stolen_total

    def _allocate_to_nodes(
        self,
        config_key: str,
        total_games: int,
        *,
        ignore_circuit_breakers: bool = False,
    ) -> dict[str, int]:
        """Allocate games for a config across available nodes.

        December 2025 - Phase 2B.4: Ephemeral node short-job prioritization.
        Short jobs (square8, hex8) are boosted for ephemeral nodes.
        Long jobs (square19, hexagonal) are reduced for ephemeral nodes.

        December 2025 - P2P Integration: Excludes unhealthy nodes and applies
        cluster health factor to allocation.

        Feb 2026: Added ignore_circuit_breakers for starvation floor fallback.
        When starving configs can't find any nodes, we retry with CB nodes
        included rather than leaving the config with zero allocation.

        Args:
            config_key: Configuration key
            total_games: Total games to allocate
            ignore_circuit_breakers: If True, include circuit-broken nodes in
                allocation (used by starvation floor as fallback).

        Returns:
            Dict mapping node_id to num_games
        """
        from app.coordination.selfplay_scheduler import MIN_GAMES_PER_ALLOCATION

        # Get unhealthy nodes from P2P health tracking
        unhealthy_nodes = getattr(self, "_unhealthy_nodes", set())

        # January 5, 2026 (Phase 7.4): Get circuit-broken nodes from registry
        # This reduces failed dispatches from 30-40% to <5% by skipping
        # nodes with open circuit breakers that are known to be failing.
        cb_registry = get_node_circuit_registry()
        circuit_broken_nodes: set[str] = set()
        try:
            for node_id in self._node_capabilities.keys():
                if cb_registry.is_circuit_open(node_id):
                    circuit_broken_nodes.add(node_id)
            if circuit_broken_nodes:
                logger.debug(
                    f"[SelfplayScheduler] Excluding {len(circuit_broken_nodes)} "
                    f"circuit-broken nodes: {circuit_broken_nodes}"
                )
        except (ImportError, AttributeError, RuntimeError, KeyError, TypeError) as e:
            # Graceful fallback if CB registry unavailable or returns unexpected types
            logger.debug(f"[SelfplayScheduler] CB registry check failed: {e}")
            pass

        # Apply cluster health factor to total games
        cluster_health = getattr(self, "_cluster_health_factor", 1.0)
        if cluster_health < 1.0:
            adjusted_games = max(MIN_GAMES_PER_ALLOCATION, int(total_games * cluster_health))
            logger.debug(
                f"[SelfplayScheduler] Cluster health {cluster_health:.2f} reducing allocation: "
                f"{total_games} -> {adjusted_games} games"
            )
            total_games = adjusted_games

        # January 2026: Build set of training-only nodes (selfplay_enabled=False)
        training_only_nodes: set[str] = set()
        for node_id in self._node_capabilities.keys():
            if not self._is_selfplay_enabled(node_id):
                training_only_nodes.add(node_id)
        if training_only_nodes:
            logger.debug(
                f"[SelfplayScheduler] Excluding {len(training_only_nodes)} "
                f"training-only nodes: {training_only_nodes}"
            )

        # Get available nodes sorted by cost efficiency then capacity
        # Feb 2026: Sort by cost efficiency to prefer cheaper nodes at equivalent capacity
        available_nodes = sorted(
            [
                n for n in self._node_capabilities.values()
                if n.available_capacity > 0.1
                and n.node_id not in unhealthy_nodes  # Exclude unhealthy nodes
                and (ignore_circuit_breakers or n.node_id not in circuit_broken_nodes)
                and n.node_id not in training_only_nodes  # Jan 2026: Exclude training-only nodes
            ],
            key=lambda n: (-n.cost_efficiency, -n.available_capacity, n.data_lag_seconds),
        )

        if not available_nodes:
            return {}

        allocation: dict[str, int] = {}
        remaining = total_games

        # Phase 2B.4: Determine if this is a short job (quick selfplay)
        # Small boards complete in <10 minutes, good for ephemeral nodes
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

            # Adjust based on GPU type
            gpu_games = SELFPLAY_GAMES_PER_NODE.get(node.gpu_type, 500)
            node_games = min(node_games, gpu_games)

            # Phase 2B.4: Ephemeral node job-duration matching
            if node.is_ephemeral:
                if is_short_job:
                    # Boost allocation to ephemeral for short jobs
                    node_games = int(node_games * 1.5)
                    logger.debug(
                        f"[SelfplayScheduler] Boosted {node.node_id} (ephemeral) "
                        f"for short job {config_key}"
                    )
                else:
                    # Reduce for long jobs (risk of termination)
                    # Feb 2026: Relaxed from 0.5 to 0.75 - 50% penalty was too
                    # harsh and contributed to square19/hexagonal starvation
                    node_games = int(node_games * 0.75)
                    logger.debug(
                        f"[SelfplayScheduler] Reduced {node.node_id} (ephemeral) "
                        f"for long job {config_key}"
                    )

            # Cap at remaining games
            node_games = min(node_games, remaining)

            if node_games >= MIN_GAMES_PER_ALLOCATION:
                allocation[node.node_id] = node_games
                remaining -= node_games

        # Dec 29, 2025: Log node allocation breakdown for this config
        if allocation and logger.isEnabledFor(logging.DEBUG):
            total_assigned = sum(allocation.values())
            logger.debug(
                f"[SelfplayScheduler] Node allocation for {config_key}: "
                f"{total_assigned}/{total_games} games across {len(allocation)} nodes "
                f"({', '.join(f'{n}={g}' for n, g in allocation.items())})"
            )
        elif not allocation:
            logger.debug(
                f"[SelfplayScheduler] No allocation for {config_key}: "
                f"no nodes available (unhealthy={len(unhealthy_nodes)}, "
                f"total_requested={total_games})"
            )

        return allocation

    async def _update_node_capabilities(self) -> None:
        """Update node capability information from cluster.

        This is intentionally rate-limited because ClusterMonitor may probe remote
        hosts via subprocess/SSH. In unit tests and some callers, node capabilities
        may be pre-seeded directly on the instance; in that case we treat them as
        fresh and skip expensive probing.
        """
        import time

        now = time.time()

        # If capabilities were pre-seeded externally (common in tests) and we have
        # never performed a refresh, treat the injected snapshot as up-to-date.
        if self._node_capabilities and self._last_node_capability_update == 0.0:
            self._last_node_capability_update = now
            return

        if self._node_capabilities and (now - self._last_node_capability_update) < self._node_capability_update_interval:
            return

        # Mark refresh time up-front to avoid repeated expensive probes if the
        # monitor fails repeatedly.
        self._last_node_capability_update = now

        try:
            # Try getting from cluster monitor
            from app.coordination.cluster_status_monitor import ClusterMonitor
            from app.config.cluster_config import load_cluster_config

            monitor = ClusterMonitor()
            status = monitor.get_cluster_status()

            # Filter out nodes with selfplay disabled in config
            cluster_cfg = load_cluster_config()
            hosts_raw = cluster_cfg.hosts_raw

            for node_id, node_status in status.nodes.items():
                host_info = hosts_raw.get(node_id, {})
                if host_info.get("selfplay_enabled") is False:
                    # Remove from capabilities if previously added
                    self._node_capabilities.pop(node_id, None)
                    continue
                node_cfg_status = host_info.get("status", "")
                if node_cfg_status in ("offline", "archived", "terminated"):
                    self._node_capabilities.pop(node_id, None)
                    continue

                if node_id not in self._node_capabilities:
                    self._node_capabilities[node_id] = NodeCapability(node_id=node_id)

                cap = self._node_capabilities[node_id]
                cap.gpu_type = node_status.gpu or "unknown"
                cap.gpu_memory_gb = node_status.gpu_memory_total_gb
                cap.is_ephemeral = is_ephemeral_node(node_id)
                cap.current_load = node_status.gpu_utilization_percent / 100.0
                cap.data_lag_seconds = node_status.sync_lag_seconds
                # Dec 29 2025: Track job count for dynamic weight calculation
                cap.current_jobs = getattr(node_status, 'selfplay_jobs', 0) or 0
                # Feb 2026: Populate cost data from provider or GPU lookup
                reported_cost = getattr(node_status, 'cost_per_hour', 0.0) or 0.0
                cap.cost_per_hour = reported_cost if reported_cost > 0 else get_gpu_cost_per_hour(cap.gpu_type)

        except Exception as e:
            logger.debug(f"[SelfplayScheduler] Error updating node capabilities: {e}")
