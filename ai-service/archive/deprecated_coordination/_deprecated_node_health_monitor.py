"""Node Health Monitor with Eviction (December 2025).

.. deprecated:: December 2025
    This module is deprecated. Use :mod:`app.coordination.health_check_orchestrator`
    instead, which provides more comprehensive multi-layer health checks including
    SSH, P2P, Tailscale, and provider API checks.

    Migration guide:
        # OLD (deprecated)
        from app.coordination.node_health_monitor import get_node_health_monitor
        monitor = get_node_health_monitor()
        available = monitor.is_node_available("node-1")

        # NEW (recommended)
        from app.coordination.health_check_orchestrator import get_health_orchestrator
        orchestrator = get_health_orchestrator()
        health = orchestrator.get_node_health("node-1")
        available = health.is_available() if health else False

Monitors cluster nodes for health and automatically evicts unhealthy nodes
from task assignment to maintain cluster reliability.

Features:
- Periodic health checks every 30s to all nodes
- Mark unhealthy after 3 consecutive failures
- Evict from task assignment after 5 consecutive failures
- Auto-recover when node responds again
- Integration with event router for alerts

Usage (DEPRECATED):
    from app.coordination.node_health_monitor import (
        NodeHealthMonitor,
        get_node_health_monitor,
    )

    monitor = get_node_health_monitor()
    await monitor.start()

    # Check if a node is available for tasks
    if monitor.is_node_available("runpod-h100"):
        # Can assign task
        pass

    # Get cluster health summary
    summary = monitor.get_cluster_summary()
"""

from __future__ import annotations

import warnings

warnings.warn(
    "node_health_monitor module is deprecated as of December 2025. "
    "Use health_check_orchestrator module instead for comprehensive health checks.",
    DeprecationWarning,
    stacklevel=2,
)

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp

from app.config.ports import HEALTH_CHECK_PORT
from app.core.async_context import safe_create_task

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Node health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # 1-2 consecutive failures
    UNHEALTHY = "unhealthy"  # 3+ consecutive failures
    EVICTED = "evicted"  # 5+ consecutive failures, no tasks assigned


@dataclass
class NodeHealth:
    """Health state for a cluster node."""
    node_id: str
    host: str
    port: int = HEALTH_CHECK_PORT
    status: NodeStatus = NodeStatus.HEALTHY
    consecutive_failures: int = 0
    last_check: float = 0.0
    last_success: float = 0.0
    last_error: str | None = None
    check_count: int = 0
    failure_count: int = 0
    evicted_at: float | None = None

    def is_available(self) -> bool:
        """Check if node is available for task assignment."""
        return self.status in (NodeStatus.HEALTHY, NodeStatus.DEGRADED)

    def record_success(self) -> None:
        """Record a successful health check."""
        self.consecutive_failures = 0
        self.last_success = time.time()
        self.last_error = None
        self.check_count += 1

        # Auto-recover from evicted state
        if self.status == NodeStatus.EVICTED:
            logger.info(f"[NodeHealth] Node {self.node_id} recovered from eviction")
            self.evicted_at = None

        self.status = NodeStatus.HEALTHY

    def record_failure(self, error: str) -> None:
        """Record a failed health check."""
        self.consecutive_failures += 1
        self.failure_count += 1
        self.last_error = error
        self.check_count += 1

        # Update status based on failure count
        if self.consecutive_failures >= 5:
            if self.status != NodeStatus.EVICTED:
                logger.warning(f"[NodeHealth] Evicting node {self.node_id} after {self.consecutive_failures} failures")
                self.evicted_at = time.time()
            self.status = NodeStatus.EVICTED
        elif self.consecutive_failures >= 3:
            self.status = NodeStatus.UNHEALTHY
        else:
            self.status = NodeStatus.DEGRADED


class NodeHealthMonitor:
    """Monitors cluster node health and manages eviction.

    Features:
    - Periodic health checks to all nodes
    - Automatic status updates based on consecutive failures
    - Eviction of unreachable nodes
    - Auto-recovery when nodes come back online
    """

    # Thresholds
    DEGRADED_THRESHOLD = 1  # Mark degraded after 1 failure
    UNHEALTHY_THRESHOLD = 3  # Mark unhealthy after 3 failures
    EVICTION_THRESHOLD = 5  # Evict after 5 failures

    def __init__(
        self,
        check_interval: float = 30.0,
        timeout: float = 10.0,
    ):
        self.check_interval = check_interval
        self.timeout = timeout
        self.nodes: dict[str, NodeHealth] = {}
        self._running = False
        self._task: asyncio.Task | None = None

        # Initialize with known nodes from config
        self._load_nodes_from_config()

    def _load_nodes_from_config(self) -> None:
        """Load node list from distributed_hosts.yaml."""
        try:
            from app.config.distributed_hosts import load_hosts_config
            config = load_hosts_config()

            for host_config in config.get("hosts", []):
                node_id = host_config.get("name", host_config.get("host"))
                host = host_config.get("host")
                port = host_config.get("port", HEALTH_CHECK_PORT)

                if node_id and host:
                    self.nodes[node_id] = NodeHealth(
                        node_id=node_id,
                        host=host,
                        port=port,
                    )

            logger.info(f"[NodeHealthMonitor] Loaded {len(self.nodes)} nodes from config")

        except Exception as e:
            logger.warning(f"[NodeHealthMonitor] Could not load config: {e}")

    def add_node(self, node_id: str, host: str, port: int = HEALTH_CHECK_PORT) -> None:
        """Add a node to monitor."""
        self.nodes[node_id] = NodeHealth(
            node_id=node_id,
            host=host,
            port=port,
        )

    def remove_node(self, node_id: str) -> None:
        """Remove a node from monitoring."""
        self.nodes.pop(node_id, None)

    async def start(self) -> None:
        """Start the health monitor."""
        if self._running:
            return

        self._running = True
        self._task = safe_create_task(
            self._monitor_loop(),
            name="node_health_monitor_loop",
        )
        logger.info(f"[NodeHealthMonitor] Started (interval={self.check_interval}s)")

    async def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[NodeHealthMonitor] Stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self.check_all_nodes()
            except Exception as e:
                logger.error(f"[NodeHealthMonitor] Check cycle error: {e}")

            await asyncio.sleep(self.check_interval)

    async def check_all_nodes(self) -> dict[str, bool]:
        """Check health of all nodes.

        Returns:
            Dict mapping node_id to health status (True = healthy)
        """
        results = {}

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            tasks = [
                self._check_node(session, node)
                for node in self.nodes.values()
            ]
            check_results = await asyncio.gather(*tasks, return_exceptions=True)

            for node, result in zip(self.nodes.values(), check_results):
                if isinstance(result, Exception):
                    results[node.node_id] = False
                else:
                    results[node.node_id] = result

        return results

    async def _check_node(self, session: aiohttp.ClientSession, node: NodeHealth) -> bool:
        """Check health of a single node.

        Returns:
            True if healthy, False otherwise
        """
        node.last_check = time.time()
        url = f"http://{node.host}:{node.port}/health"

        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    node.record_success()
                    return True
                else:
                    node.record_failure(f"HTTP {resp.status}")
                    return False
        except asyncio.TimeoutError:
            node.record_failure("Timeout")
            return False
        except aiohttp.ClientError as e:
            node.record_failure(str(e))
            return False
        except Exception as e:
            node.record_failure(f"Error: {e}")
            return False

    def is_node_available(self, node_id: str) -> bool:
        """Check if a node is available for task assignment."""
        node = self.nodes.get(node_id)
        return node.is_available() if node else False

    def get_available_nodes(self) -> list[str]:
        """Get list of node IDs available for task assignment."""
        return [
            node_id for node_id, node in self.nodes.items()
            if node.is_available()
        ]

    def get_evicted_nodes(self) -> list[str]:
        """Get list of evicted node IDs."""
        return [
            node_id for node_id, node in self.nodes.items()
            if node.status == NodeStatus.EVICTED
        ]

    def get_node_health(self, node_id: str) -> NodeHealth | None:
        """Get health info for a specific node."""
        return self.nodes.get(node_id)

    def get_cluster_summary(self) -> dict[str, Any]:
        """Get summary of cluster health.

        Returns:
            Dict with cluster health summary
        """
        by_status = {status: 0 for status in NodeStatus}
        for node in self.nodes.values():
            by_status[node.status] += 1

        return {
            "total_nodes": len(self.nodes),
            "healthy": by_status[NodeStatus.HEALTHY],
            "degraded": by_status[NodeStatus.DEGRADED],
            "unhealthy": by_status[NodeStatus.UNHEALTHY],
            "evicted": by_status[NodeStatus.EVICTED],
            "available_nodes": self.get_available_nodes(),
            "evicted_nodes": self.get_evicted_nodes(),
            "timestamp": datetime.now().isoformat(),
        }

    def force_evict(self, node_id: str) -> bool:
        """Force evict a node.

        Returns:
            True if node was evicted, False if not found
        """
        node = self.nodes.get(node_id)
        if node:
            node.status = NodeStatus.EVICTED
            node.evicted_at = time.time()
            logger.info(f"[NodeHealthMonitor] Force evicted node {node_id}")
            return True
        return False

    def force_recover(self, node_id: str) -> bool:
        """Force recover an evicted node.

        Returns:
            True if node was recovered, False if not found
        """
        node = self.nodes.get(node_id)
        if node:
            node.status = NodeStatus.HEALTHY
            node.consecutive_failures = 0
            node.evicted_at = None
            logger.info(f"[NodeHealthMonitor] Force recovered node {node_id}")
            return True
        return False


# Global instance
_node_health_monitor: NodeHealthMonitor | None = None


def get_node_health_monitor() -> NodeHealthMonitor:
    """Get or create the global node health monitor."""
    global _node_health_monitor

    if _node_health_monitor is None:
        _node_health_monitor = NodeHealthMonitor()

    return _node_health_monitor


def is_node_available(node_id: str) -> bool:
    """Check if a node is available for task assignment.

    Convenience function that uses the global monitor.
    """
    return get_node_health_monitor().is_node_available(node_id)


def get_available_nodes() -> list[str]:
    """Get list of available nodes.

    Convenience function that uses the global monitor.
    """
    return get_node_health_monitor().get_available_nodes()
