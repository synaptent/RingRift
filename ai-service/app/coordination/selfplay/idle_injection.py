"""IdleWorkInjectionMixin - Idle node work injection for SelfplayScheduler.

January 2026 Sprint 17.9: Extracted from selfplay_scheduler.py (~220 LOC)
to reduce main file size toward ~1,800 LOC target.

This mixin provides:
- Idle node detection and tracking
- Emergency work injection for idle GPU nodes
- Config priority-based work assignment for underserved configs

Usage:
    class SelfplayScheduler(IdleWorkInjectionMixin, ...):
        pass

The mixin expects the following attributes on the class:
- _node_capabilities: dict[str, NodeCapability]
- _node_idle_since: dict[str, float]
- _idle_threshold_seconds: float
- _last_idle_injection: float
- _idle_injection_cooldown: float
- _config_priorities: dict[str, ConfigPriority]
- _unhealthy_nodes: set[str] (optional via getattr)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

# Import DATA_STARVATION_CRITICAL_THRESHOLD for config selection
from app.config.coordination_defaults import SelfplayPriorityWeightDefaults

if TYPE_CHECKING:
    from app.coordination.node_allocator import NodeCapability
    from app.coordination.selfplay_priority_types import ConfigPriority

logger = logging.getLogger(__name__)

# Get threshold from centralized config
_priority_weight_defaults = SelfplayPriorityWeightDefaults()
DATA_STARVATION_CRITICAL_THRESHOLD = _priority_weight_defaults.DATA_STARVATION_CRITICAL_THRESHOLD


class IdleWorkInjectionMixin:
    """Mixin providing idle node work injection methods.

    This mixin extracts idle node handling from SelfplayScheduler:
    - Tracks which nodes have been idle (no active jobs)
    - Detects nodes idle beyond threshold
    - Injects priority selfplay work for underserved configs

    Attributes expected from main class:
        _node_capabilities: Dict of node capabilities
        _node_idle_since: Dict tracking when nodes became idle
        _idle_threshold_seconds: Threshold before work injection
        _last_idle_injection: Last injection timestamp
        _idle_injection_cooldown: Cooldown between injections
        _config_priorities: Dict of config priority tracking
    """

    # Type hints for attributes provided by SelfplayScheduler
    _node_capabilities: dict[str, "NodeCapability"]
    _node_idle_since: dict[str, float]
    _idle_threshold_seconds: float
    _last_idle_injection: float
    _idle_injection_cooldown: float
    _config_priorities: dict[str, "ConfigPriority"]

    # =========================================================================
    # Idle Node Work Injection (Jan 5, 2026 - Sprint 17.30)
    # =========================================================================

    def _update_node_idle_tracking(self) -> None:
        """Update idle tracking state for all nodes.

        Called during node capability updates. Tracks when each node
        first became idle (current_jobs == 0).

        Jan 5, 2026: Part of idle node work injection feature.
        Jan 5, 2026 (Task 8.4): Now includes CPU-only nodes for heuristic selfplay.
        """
        now = time.time()

        for node_id, cap in self._node_capabilities.items():
            # Jan 5, 2026: Include CPU-only nodes now that they can run heuristic selfplay
            # Previously skipped CPU-only nodes, but they can contribute training data

            if cap.current_jobs == 0:
                # Node is idle - start tracking if not already
                if node_id not in self._node_idle_since:
                    self._node_idle_since[node_id] = now
                    logger.debug(
                        f"[SelfplayScheduler] Node {node_id} became idle, "
                        f"will inject work after {self._idle_threshold_seconds}s"
                    )
            else:
                # Node is working - clear idle tracking
                if node_id in self._node_idle_since:
                    del self._node_idle_since[node_id]

    def _detect_idle_nodes(self) -> list[str]:
        """Get list of nodes that have been idle longer than threshold.

        Returns:
            List of node IDs that are idle and eligible for work injection
        """
        now = time.time()
        idle_nodes = []

        # Get unhealthy nodes to skip
        unhealthy_nodes = getattr(self, "_unhealthy_nodes", set())

        for node_id, idle_since in self._node_idle_since.items():
            idle_duration = now - idle_since

            if idle_duration >= self._idle_threshold_seconds:
                # Skip unhealthy nodes
                if node_id in unhealthy_nodes:
                    logger.debug(
                        f"[SelfplayScheduler] Skipping unhealthy idle node: {node_id}"
                    )
                    continue

                idle_nodes.append(node_id)
                logger.debug(
                    f"[SelfplayScheduler] Node {node_id} idle for {idle_duration:.0f}s "
                    f"(threshold: {self._idle_threshold_seconds}s)"
                )

        return idle_nodes

    def _get_most_underserved_config(self) -> str | None:
        """Get the config with lowest game count that needs more data.

        Jan 5, 2026: Used for idle node work injection. Prioritizes
        configs in ULTRA/EMERGENCY starvation tiers.

        Returns:
            Config key of most underserved config, or None if all are healthy
        """
        most_underserved = None
        lowest_game_count = float("inf")

        for config_key, priority in self._config_priorities.items():
            game_count = priority.game_count

            # Skip configs that are adequately served
            if game_count >= DATA_STARVATION_CRITICAL_THRESHOLD:
                continue

            # Prefer the config with lowest game count
            if game_count < lowest_game_count:
                lowest_game_count = game_count
                most_underserved = config_key

        return most_underserved

    async def inject_work_for_idle_nodes(self) -> dict[str, Any]:
        """Inject priority selfplay work for nodes that have been idle.

        Jan 5, 2026 - Sprint 17.30: When GPU nodes are idle for more than
        IDLE_THRESHOLD_SECONDS (default 5 min), allocate emergency selfplay
        work for the most underserved config.

        This improves:
        - GPU utilization (+15-25% during backpressure periods)
        - Elo improvement (+8-15 Elo from additional data for starved configs)
        - Cluster efficiency (idle resources put to productive use)

        Returns:
            Dict with injection statistics
        """
        now = time.time()

        # Rate limit work injection
        if now - self._last_idle_injection < self._idle_injection_cooldown:
            return {"skipped": True, "reason": "cooldown"}

        # Update idle tracking from latest capabilities
        self._update_node_idle_tracking()

        # Detect nodes that are idle past threshold
        idle_nodes = self._detect_idle_nodes()

        if not idle_nodes:
            return {"idle_nodes": 0, "injected": 0}

        # Get most underserved config to work on
        target_config = self._get_most_underserved_config()

        if not target_config:
            # All configs have sufficient data
            logger.debug(
                f"[SelfplayScheduler] {len(idle_nodes)} idle nodes but no underserved configs"
            )
            return {
                "idle_nodes": len(idle_nodes),
                "injected": 0,
                "reason": "no_underserved_configs",
            }

        # Inject work for idle nodes
        injected_count = 0
        cpu_node_count = 0
        games_per_node = 50  # Emergency allocation per node
        # Jan 5, 2026 (Task 8.4): Lower batch for CPU-only nodes (heuristic is slower)
        games_per_cpu_node = 25

        allocation: dict[str, dict[str, int]] = {target_config: {}}

        for node_id in idle_nodes:
            # Jan 5, 2026 (Task 8.4): CPU nodes get fewer games (heuristic is slower)
            is_cpu = self._is_cpu_only_node(node_id)
            node_games = games_per_cpu_node if is_cpu else games_per_node

            # Allocate emergency games to this node
            allocation[target_config][node_id] = node_games
            injected_count += 1
            if is_cpu:
                cpu_node_count += 1

            # Clear idle tracking since we're injecting work
            if node_id in self._node_idle_since:
                del self._node_idle_since[node_id]

        self._last_idle_injection = now

        # Log the injection
        priority = self._config_priorities.get(target_config)
        game_count = priority.game_count if priority else 0
        total_games = sum(allocation[target_config].values())

        # Jan 5, 2026 (Task 8.4): Log CPU vs GPU node breakdown
        gpu_node_count = injected_count - cpu_node_count
        node_breakdown = f"{gpu_node_count} GPU, {cpu_node_count} CPU" if cpu_node_count > 0 else f"{gpu_node_count} GPU"

        logger.info(
            f"[SelfplayScheduler] Idle work injection: "
            f"{len(idle_nodes)} idle nodes ({node_breakdown}) → {target_config} ({game_count} games, "
            f"+{total_games} games allocated)"
        )

        # Emit event for observability
        try:
            from app.coordination.event_router import DataEventType, emit_event

            emit_event(
                DataEventType.IDLE_NODE_WORK_INJECTED,
                {
                    "idle_nodes": idle_nodes,
                    "target_config": target_config,
                    "games_injected": total_games,
                    "timestamp": now,
                    # Jan 5, 2026 (Task 8.4): Include CPU vs GPU breakdown
                    "gpu_nodes": gpu_node_count,
                    "cpu_nodes": cpu_node_count,
                },
            )
        except (ImportError, AttributeError):
            pass

        return {
            "idle_nodes": len(idle_nodes),
            "injected": injected_count,
            "target_config": target_config,
            "games_per_node": games_per_node,
            "games_per_cpu_node": games_per_cpu_node,
            "total_games": total_games,
            "gpu_nodes": gpu_node_count,
            "cpu_nodes": cpu_node_count,
        }
