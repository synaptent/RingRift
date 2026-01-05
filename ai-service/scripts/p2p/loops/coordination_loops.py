"""Coordination Loops for P2P Orchestrator.

December 2025: Background loops for cluster coordination.

Loops:
- AutoScalingLoop: Scales cluster resources based on demand
- HealthAggregationLoop: Aggregates health metrics from all nodes

Usage:
    from scripts.p2p.loops import AutoScalingLoop, HealthAggregationLoop

    scaling = AutoScalingLoop(
        get_pending_work=lambda: orchestrator.pending_work_count,
        get_active_nodes=lambda: orchestrator.active_node_count,
        scale_up=orchestrator.provision_node,
        scale_down=orchestrator.terminate_node,
    )
    await scaling.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from .base import BaseLoop

logger = logging.getLogger(__name__)


@dataclass
class AutoScalingConfig:
    """Configuration for auto-scaling loop."""

    check_interval_seconds: float = 120.0  # 2 minutes
    scale_up_threshold: int = 10  # Pending items per node to trigger scale up
    scale_down_threshold: int = 2  # Pending items per node to trigger scale down
    min_nodes: int = 2
    max_nodes: int = 20
    scale_cooldown_seconds: float = 600.0  # 10 minutes
    max_scale_per_cycle: int = 3
    idle_threshold_seconds: float = 900.0  # 15 minutes

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.scale_up_threshold <= 0:
            raise ValueError("scale_up_threshold must be > 0")
        if self.scale_down_threshold <= 0:
            raise ValueError("scale_down_threshold must be > 0")
        if self.scale_down_threshold > self.scale_up_threshold:
            raise ValueError("scale_down_threshold must be <= scale_up_threshold")
        if self.min_nodes < 0:
            raise ValueError("min_nodes must be >= 0")
        if self.max_nodes < self.min_nodes:
            raise ValueError("max_nodes must be >= min_nodes")
        if self.scale_cooldown_seconds < 0:
            raise ValueError("scale_cooldown_seconds must be >= 0")
        if self.max_scale_per_cycle <= 0:
            raise ValueError("max_scale_per_cycle must be > 0")
        if self.idle_threshold_seconds <= 0:
            raise ValueError("idle_threshold_seconds must be > 0")


class AutoScalingLoop(BaseLoop):
    """Background loop that manages cluster scaling.

    Monitors work queue depth and node utilization to automatically
    scale the cluster up or down to match demand.
    """

    def __init__(
        self,
        get_pending_work: Callable[[], int],
        get_active_nodes: Callable[[], int],
        get_idle_nodes: Callable[[], list[str]],
        scale_up: Callable[[int], Coroutine[Any, Any, list[str]]],
        scale_down: Callable[[list[str]], Coroutine[Any, Any, int]],
        config: AutoScalingConfig | None = None,
    ):
        """Initialize auto-scaling loop.

        Args:
            get_pending_work: Callback returning pending work item count
            get_active_nodes: Callback returning active node count
            get_idle_nodes: Callback returning list of idle node IDs
            scale_up: Async callback to provision N new nodes, returns node IDs
            scale_down: Async callback to terminate nodes, returns count terminated
            config: Scaling configuration
        """
        self.config = config or AutoScalingConfig()
        super().__init__(
            name="auto_scaling",
            interval=self.config.check_interval_seconds,
        )
        self._get_pending_work = get_pending_work
        self._get_active_nodes = get_active_nodes
        self._get_idle_nodes = get_idle_nodes
        self._scale_up = scale_up
        self._scale_down = scale_down
        self._last_scale_time: float = 0.0
        self._scaling_stats = {
            "scale_up_events": 0,
            "scale_down_events": 0,
            "nodes_added": 0,
            "nodes_removed": 0,
        }

    async def _run_once(self) -> None:
        """Evaluate and execute scaling decisions."""
        now = time.time()

        # Check cooldown
        if now - self._last_scale_time < self.config.scale_cooldown_seconds:
            return

        pending = self._get_pending_work()
        active_nodes = self._get_active_nodes()

        if active_nodes == 0:
            # No nodes, need to scale up
            if pending > 0:
                await self._do_scale_up(1)
            return

        # Calculate work per node
        work_per_node = pending / active_nodes

        # Decide scaling action
        if work_per_node > self.config.scale_up_threshold:
            # Scale up
            if active_nodes < self.config.max_nodes:
                nodes_needed = min(
                    int((pending / self.config.scale_up_threshold) - active_nodes + 1),
                    self.config.max_scale_per_cycle,
                    self.config.max_nodes - active_nodes,
                )
                if nodes_needed > 0:
                    await self._do_scale_up(nodes_needed)

        elif work_per_node < self.config.scale_down_threshold:
            # Check for scale down
            idle_nodes = self._get_idle_nodes()
            if idle_nodes and active_nodes > self.config.min_nodes:
                # Scale down idle nodes
                nodes_to_remove = min(
                    len(idle_nodes),
                    self.config.max_scale_per_cycle,
                    active_nodes - self.config.min_nodes,
                )
                if nodes_to_remove > 0:
                    await self._do_scale_down(idle_nodes[:nodes_to_remove])

    async def _do_scale_up(self, count: int) -> None:
        """Execute scale up operation."""
        try:
            logger.info(f"[AutoScaling] Scaling up by {count} nodes")
            new_nodes = await self._scale_up(count)
            self._last_scale_time = time.time()
            self._scaling_stats["scale_up_events"] += 1
            self._scaling_stats["nodes_added"] += len(new_nodes)
            logger.info(f"[AutoScaling] Added {len(new_nodes)} nodes: {new_nodes}")
        except Exception as e:
            logger.error(f"[AutoScaling] Scale up failed: {e}")

    async def _do_scale_down(self, nodes: list[str]) -> None:
        """Execute scale down operation."""
        try:
            logger.info(f"[AutoScaling] Scaling down {len(nodes)} nodes: {nodes}")
            removed = await self._scale_down(nodes)
            self._last_scale_time = time.time()
            self._scaling_stats["scale_down_events"] += 1
            self._scaling_stats["nodes_removed"] += removed
            logger.info(f"[AutoScaling] Removed {removed} nodes")
        except Exception as e:
            logger.error(f"[AutoScaling] Scale down failed: {e}")

    def get_scaling_stats(self) -> dict[str, Any]:
        """Get scaling statistics."""
        return {
            **self._scaling_stats,
            "last_scale_time": self._last_scale_time,
            **self.stats.to_dict(),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        scale_stats = self.get_scaling_stats()
        scale_ups = scale_stats.get("scale_up_events", 0)
        scale_downs = scale_stats.get("scale_down_events", 0)
        nodes_added = scale_stats.get("nodes_added", 0)
        nodes_removed = scale_stats.get("nodes_removed", 0)

        if not self.is_running():
            status = "ERROR"
            message = "Auto-scaling loop not running"
        else:
            status = "HEALTHY"
            message = f"Scale ups: {scale_ups}, Scale downs: {scale_downs}"

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.is_running(),
                "scale_up_events": scale_ups,
                "scale_down_events": scale_downs,
                "nodes_added": nodes_added,
                "nodes_removed": nodes_removed,
                "last_scale_time": self._last_scale_time,
                "run_count": self.stats.run_count,
            },
        }


@dataclass
class HealthAggregationConfig:
    """Configuration for health aggregation loop."""

    check_interval_seconds: float = 30.0
    health_timeout_seconds: float = 10.0
    max_nodes_per_cycle: int = 50
    unhealthy_threshold_seconds: float = 120.0  # Mark unhealthy after this

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.health_timeout_seconds <= 0:
            raise ValueError("health_timeout_seconds must be > 0")
        if self.max_nodes_per_cycle <= 0:
            raise ValueError("max_nodes_per_cycle must be > 0")
        if self.unhealthy_threshold_seconds <= 0:
            raise ValueError("unhealthy_threshold_seconds must be > 0")


class HealthAggregationLoop(BaseLoop):
    """Background loop that aggregates health metrics from cluster nodes.

    Collects CPU, memory, GPU, and disk metrics from all nodes and
    maintains a unified health view of the cluster.

    January 5, 2026 (Session 17.26): Added on_recover_unhealthy callback to
    automatically recover nodes from unhealthy state after transient failures.
    """

    def __init__(
        self,
        get_node_ids: Callable[[], list[str]],
        fetch_node_health: Callable[[str], Coroutine[Any, Any, dict[str, Any]]],
        on_health_updated: Callable[[dict[str, dict[str, Any]]], None] | None = None,
        on_recover_unhealthy: Callable[[], list[str]] | None = None,
        config: HealthAggregationConfig | None = None,
    ):
        """Initialize health aggregation loop.

        Args:
            get_node_ids: Callback returning list of node IDs to check
            fetch_node_health: Async callback to fetch health from a node
            on_health_updated: Optional callback when health data is updated
            on_recover_unhealthy: Optional callback to attempt recovery of unhealthy nodes.
                Should return list of recovered node IDs. Called after each health
                aggregation cycle. January 5, 2026 (Session 17.26): Added to fix
                permanently excluded nodes issue - nodes marked unhealthy were never
                automatically recovered.
            config: Aggregation configuration
        """
        self.config = config or HealthAggregationConfig()
        super().__init__(
            name="health_aggregation",
            interval=self.config.check_interval_seconds,
        )
        self._get_node_ids = get_node_ids
        self._fetch_node_health = fetch_node_health
        self._on_health_updated = on_health_updated
        self._on_recover_unhealthy = on_recover_unhealthy
        self._node_health: dict[str, dict[str, Any]] = {}
        self._last_health_time: dict[str, float] = {}
        self._recovery_stats = {
            "total_recovered": 0,
            "last_recovery_time": 0.0,
            "last_recovered_nodes": [],
        }

    async def _run_once(self) -> None:
        """Collect health metrics from all nodes."""
        node_ids = self._get_node_ids()
        if not node_ids:
            return

        now = time.time()
        updated_health: dict[str, dict[str, Any]] = {}

        # Fetch health from each node concurrently
        async def fetch_one(node_id: str) -> tuple[str, dict[str, Any] | None]:
            try:
                health = await asyncio.wait_for(
                    self._fetch_node_health(node_id),
                    timeout=self.config.health_timeout_seconds,
                )
                return node_id, health
            except Exception as e:
                logger.debug(f"[HealthAggregation] Failed to fetch from {node_id}: {e}")
                return node_id, None

        # Process in batches
        for i in range(0, len(node_ids), self.config.max_nodes_per_cycle):
            batch = node_ids[i:i + self.config.max_nodes_per_cycle]
            results = await asyncio.gather(*[fetch_one(nid) for nid in batch])

            for node_id, health in results:
                if health:
                    health["timestamp"] = now
                    health["healthy"] = True
                    updated_health[node_id] = health
                    self._last_health_time[node_id] = now
                else:
                    # Check if we should mark as unhealthy
                    last_time = self._last_health_time.get(node_id, 0)
                    if now - last_time > self.config.unhealthy_threshold_seconds:
                        updated_health[node_id] = {
                            "healthy": False,
                            "last_seen": last_time,
                            "error": "Health check timeout",
                        }

        # Update stored health
        self._node_health.update(updated_health)

        # Call callback if provided
        if self._on_health_updated and updated_health:
            try:
                self._on_health_updated(updated_health)
            except Exception as e:
                logger.warning(f"[HealthAggregation] Callback failed: {e}")

        # January 5, 2026 (Session 17.26): Attempt to recover unhealthy nodes
        # This fixes the critical issue where nodes were permanently excluded
        # from work distribution after transient failures.
        if self._on_recover_unhealthy:
            try:
                recovered = self._on_recover_unhealthy()
                if recovered:
                    self._recovery_stats["total_recovered"] += len(recovered)
                    self._recovery_stats["last_recovery_time"] = time.time()
                    self._recovery_stats["last_recovered_nodes"] = recovered
                    logger.info(
                        f"[HealthAggregation] Auto-recovered {len(recovered)} nodes: {recovered}"
                    )
            except Exception as e:
                logger.warning(f"[HealthAggregation] Recovery callback failed: {e}")

    def get_cluster_health(self) -> dict[str, dict[str, Any]]:
        """Get current cluster health data."""
        return dict(self._node_health)

    def get_unhealthy_nodes(self) -> list[str]:
        """Get list of unhealthy node IDs."""
        return [
            node_id
            for node_id, health in self._node_health.items()
            if not health.get("healthy", True)
        ]

    def get_aggregation_stats(self) -> dict[str, Any]:
        """Get aggregation statistics."""
        healthy_count = sum(
            1 for h in self._node_health.values()
            if h.get("healthy", True)
        )
        return {
            "nodes_tracked": len(self._node_health),
            "healthy_nodes": healthy_count,
            "unhealthy_nodes": len(self._node_health) - healthy_count,
            "total_recovered": self._recovery_stats["total_recovered"],
            "last_recovery_time": self._recovery_stats["last_recovery_time"],
            **self.stats.to_dict(),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        agg_stats = self.get_aggregation_stats()
        nodes_tracked = agg_stats.get("nodes_tracked", 0)
        healthy_nodes = agg_stats.get("healthy_nodes", 0)
        unhealthy_nodes = agg_stats.get("unhealthy_nodes", 0)

        if not self.is_running():
            status = "ERROR"
            message = "Health aggregation loop not running"
        elif nodes_tracked > 0 and unhealthy_nodes > healthy_nodes:
            status = "DEGRADED"
            message = f"Majority unhealthy: {unhealthy_nodes}/{nodes_tracked}"
        elif unhealthy_nodes > 0:
            status = "HEALTHY"
            message = f"Tracking {nodes_tracked} nodes, {unhealthy_nodes} unhealthy"
        else:
            status = "HEALTHY"
            message = f"Tracking {nodes_tracked} healthy nodes"

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.is_running(),
                "nodes_tracked": nodes_tracked,
                "healthy_nodes": healthy_nodes,
                "unhealthy_nodes": unhealthy_nodes,
                "run_count": self.stats.run_count,
                "total_recovered": self._recovery_stats["total_recovered"],
                "last_recovery_time": self._recovery_stats["last_recovery_time"],
            },
        }


__all__ = [
    "AutoScalingConfig",
    "AutoScalingLoop",
    "HealthAggregationConfig",
    "HealthAggregationLoop",
]
