"""Unified Health Check Orchestrator.

Comprehensive health checking across all providers with graduated responses
and integration with the provider manager layer.

Features:
- Multi-layered health checks (P2P, SSH, Tailscale, provider API)
- Parallel checking for speed
- Graduated health states (healthy → degraded → unhealthy → offline)
- Utilization metrics collection
- Event emission for alerting

Usage:
    from app.coordination.health_check_orchestrator import (
        HealthCheckOrchestrator,
        get_health_orchestrator,
    )

    orchestrator = get_health_orchestrator()
    await orchestrator.start()

    # Get cluster health summary
    summary = await orchestrator.get_cluster_health()

    # Check specific node
    health = await orchestrator.check_node("lambda-gh200-a")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.providers import (
    AWSManager,
    HetznerManager,
    HealthCheckResult,
    InstanceState,
    LambdaManager,
    Provider,
    ProviderInstance,
    TailscaleManager,
    VastManager,
)

logger = logging.getLogger(__name__)


class NodeHealthState(str, Enum):
    """Graduated node health states."""

    HEALTHY = "healthy"  # All checks pass
    DEGRADED = "degraded"  # 1 check failing
    UNHEALTHY = "unhealthy"  # 2+ checks failing
    OFFLINE = "offline"  # SSH unreachable
    PROVIDER_DOWN = "provider_down"  # Provider reports down
    RETIRED = "retired"  # Manually removed


@dataclass
class NodeHealthDetails:
    """Detailed health information for a node."""

    node_id: str
    provider: Provider
    state: NodeHealthState = NodeHealthState.OFFLINE
    last_check: datetime = field(default_factory=datetime.now)

    # Individual check results
    ssh_healthy: bool = False
    p2p_healthy: bool = False
    tailscale_healthy: bool = False
    provider_healthy: bool = False

    # Latency metrics
    ssh_latency_ms: float | None = None
    p2p_latency_ms: float | None = None

    # Utilization (if available)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_percent: float = 0.0

    # Workload info
    selfplay_jobs: int = 0
    training_running: bool = False

    # Error info
    last_error: str | None = None
    consecutive_failures: int = 0

    # Instance info
    instance: ProviderInstance | None = None

    def is_available(self) -> bool:
        """Check if node is available for task assignment."""
        return self.state in (NodeHealthState.HEALTHY, NodeHealthState.DEGRADED)

    def compute_state(self) -> NodeHealthState:
        """Compute health state from individual checks."""
        if not self.ssh_healthy:
            return NodeHealthState.OFFLINE

        checks = [self.p2p_healthy, self.tailscale_healthy]
        failing = sum(1 for c in checks if not c)

        if failing == 0:
            return NodeHealthState.HEALTHY
        elif failing == 1:
            return NodeHealthState.DEGRADED
        else:
            return NodeHealthState.UNHEALTHY


@dataclass
class ClusterHealthSummary:
    """Summary of cluster health across all providers."""

    timestamp: datetime = field(default_factory=datetime.now)
    total_nodes: int = 0
    healthy: int = 0
    degraded: int = 0
    unhealthy: int = 0
    offline: int = 0
    retired: int = 0

    # By provider
    by_provider: dict[str, dict[str, int]] = field(default_factory=dict)

    # Available capacity
    total_gpus: int = 0
    available_gpus: int = 0
    total_cpu_cores: int = 0
    available_cpu_cores: int = 0

    # Cost
    hourly_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_nodes": self.total_nodes,
            "healthy": self.healthy,
            "degraded": self.degraded,
            "unhealthy": self.unhealthy,
            "offline": self.offline,
            "retired": self.retired,
            "by_provider": self.by_provider,
            "total_gpus": self.total_gpus,
            "available_gpus": self.available_gpus,
            "total_cpu_cores": self.total_cpu_cores,
            "available_cpu_cores": self.available_cpu_cores,
            "hourly_cost": self.hourly_cost,
            "availability_percent": (
                self.healthy + self.degraded
            ) / max(self.total_nodes - self.retired, 1) * 100,
        }


class HealthCheckOrchestrator:
    """Orchestrates health checks across all providers.

    Runs periodic health checks and maintains current health state
    for all nodes in the cluster.
    """

    # Check intervals (seconds)
    P2P_CHECK_INTERVAL = 60
    SSH_CHECK_INTERVAL = 300
    PROVIDER_CHECK_INTERVAL = 300
    UTILIZATION_CHECK_INTERVAL = 60

    def __init__(
        self,
        check_interval: float = 60.0,
        p2p_port: int = 8770,
    ):
        """Initialize health check orchestrator.

        Args:
            check_interval: Seconds between health check cycles
            p2p_port: P2P daemon port to check
        """
        self.check_interval = check_interval
        self.p2p_port = p2p_port

        # Provider managers
        self.lambda_mgr = LambdaManager()
        self.vast_mgr = VastManager()
        self.hetzner_mgr = HetznerManager()
        self.aws_mgr = AWSManager()
        self.tailscale_mgr = TailscaleManager()

        # Health state
        self.node_health: dict[str, NodeHealthDetails] = {}
        self._instances: dict[str, ProviderInstance] = {}

        # Control
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_full_check = 0.0

    async def start(self) -> None:
        """Start the health check orchestrator."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info(
            f"[HealthCheckOrchestrator] Started (interval={self.check_interval}s)"
        )

    async def stop(self) -> None:
        """Stop the health check orchestrator."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Close managers
        await self.lambda_mgr.close()

        logger.info("[HealthCheckOrchestrator] Stopped")

    async def _check_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            try:
                await self.run_full_health_check()
            except Exception as e:
                logger.error(f"[HealthCheckOrchestrator] Check cycle error: {e}")

            await asyncio.sleep(self.check_interval)

    async def run_full_health_check(self) -> dict[str, NodeHealthDetails]:
        """Run complete health check cycle.

        1. Discover all instances from all providers
        2. Run health checks in parallel
        3. Update health state
        4. Emit events if needed

        Returns:
            Dict mapping node_id to health details
        """
        start = time.time()
        logger.info("[HealthCheckOrchestrator] Starting health check cycle...")

        # Phase 1: Discover instances from all providers
        await self._discover_all_instances()

        # Phase 2: Run health checks in parallel
        check_tasks = []
        for node_id, instance in self._instances.items():
            check_tasks.append(self._check_node_health(node_id, instance))

        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Process results
        for node_id, result in zip(self._instances.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"[HealthCheckOrchestrator] Check failed for {node_id}: {result}")
                if node_id in self.node_health:
                    self.node_health[node_id].consecutive_failures += 1
                    self.node_health[node_id].last_error = str(result)
                    self.node_health[node_id].state = NodeHealthState.UNHEALTHY
            else:
                self.node_health[node_id] = result

        self._last_full_check = time.time()
        elapsed = (time.time() - start) * 1000

        # Summary log
        summary = await self.get_cluster_health()
        logger.info(
            f"[HealthCheckOrchestrator] Check complete in {elapsed:.0f}ms: "
            f"{summary.healthy} healthy, {summary.degraded} degraded, "
            f"{summary.unhealthy} unhealthy, {summary.offline} offline"
        )

        return self.node_health

    async def _discover_all_instances(self) -> None:
        """Discover instances from all providers."""
        logger.debug("[HealthCheckOrchestrator] Discovering instances...")

        # Run all discoveries in parallel
        results = await asyncio.gather(
            self.lambda_mgr.list_instances(),
            self.vast_mgr.list_instances(),
            self.hetzner_mgr.list_instances(),
            self.aws_mgr.list_instances(),
            return_exceptions=True,
        )

        self._instances.clear()

        provider_names = ["Lambda", "Vast", "Hetzner", "AWS"]
        for provider_name, result in zip(provider_names, results):
            if isinstance(result, Exception):
                logger.warning(
                    f"[HealthCheckOrchestrator] {provider_name} discovery failed: {result}"
                )
                continue

            for instance in result:
                # Only include running instances
                if instance.state == InstanceState.RUNNING:
                    node_id = instance.name or instance.instance_id
                    self._instances[node_id] = instance

        logger.debug(
            f"[HealthCheckOrchestrator] Discovered {len(self._instances)} running instances"
        )

    async def _check_node_health(
        self,
        node_id: str,
        instance: ProviderInstance,
    ) -> NodeHealthDetails:
        """Check health of a single node.

        Performs multiple health checks in parallel and aggregates results.
        """
        details = NodeHealthDetails(
            node_id=node_id,
            provider=instance.provider,
            instance=instance,
        )

        # Get the appropriate manager
        manager = self._get_manager_for_provider(instance.provider)
        if not manager:
            details.state = NodeHealthState.OFFLINE
            details.last_error = "No manager for provider"
            return details

        # Run checks in parallel
        check_results = await asyncio.gather(
            manager.check_ssh_connectivity(instance),
            manager.check_p2p_health(instance, port=self.p2p_port),
            manager.check_tailscale(instance),
            return_exceptions=True,
        )

        # Process SSH result
        ssh_result = check_results[0]
        if isinstance(ssh_result, Exception):
            details.ssh_healthy = False
            details.last_error = f"SSH: {ssh_result}"
        elif isinstance(ssh_result, HealthCheckResult):
            details.ssh_healthy = ssh_result.healthy
            details.ssh_latency_ms = ssh_result.latency_ms
            if not ssh_result.healthy:
                details.last_error = ssh_result.message

        # Process P2P result
        p2p_result = check_results[1]
        if isinstance(p2p_result, Exception):
            details.p2p_healthy = False
        elif isinstance(p2p_result, HealthCheckResult):
            details.p2p_healthy = p2p_result.healthy
            details.p2p_latency_ms = p2p_result.latency_ms

        # Process Tailscale result
        ts_result = check_results[2]
        if isinstance(ts_result, Exception):
            details.tailscale_healthy = False
        elif isinstance(ts_result, HealthCheckResult):
            details.tailscale_healthy = ts_result.healthy

        # Compute overall state
        details.state = details.compute_state()
        details.last_check = datetime.now()

        # If healthy, get utilization metrics
        if details.ssh_healthy:
            try:
                util = await manager.get_utilization(instance)
                details.cpu_percent = util.get("cpu_percent", 0.0)
                details.memory_percent = util.get("memory_percent", 0.0)
                details.gpu_percent = util.get("gpu_percent", 0.0)
                details.gpu_memory_percent = util.get("gpu_memory_percent", 0.0)
                details.disk_percent = util.get("disk_percent", 0.0)
            except Exception as e:
                logger.debug(f"[HealthCheckOrchestrator] Utilization check failed for {node_id}: {e}")

        # Reset or increment failure counter
        if details.state == NodeHealthState.HEALTHY:
            details.consecutive_failures = 0
        else:
            old_details = self.node_health.get(node_id)
            if old_details:
                details.consecutive_failures = old_details.consecutive_failures + 1

        return details

    def _get_manager_for_provider(self, provider: Provider):
        """Get the appropriate manager for a provider."""
        managers = {
            Provider.LAMBDA: self.lambda_mgr,
            Provider.VAST: self.vast_mgr,
            Provider.HETZNER: self.hetzner_mgr,
            Provider.AWS: self.aws_mgr,
        }
        return managers.get(provider)

    async def check_node(self, node_id: str) -> NodeHealthDetails | None:
        """Check health of a specific node.

        Returns:
            NodeHealthDetails if node found, None otherwise
        """
        instance = self._instances.get(node_id)
        if not instance:
            # Try to find by partial match
            for nid, inst in self._instances.items():
                if node_id.lower() in nid.lower():
                    instance = inst
                    node_id = nid
                    break

        if not instance:
            return None

        return await self._check_node_health(node_id, instance)

    async def get_cluster_health(self) -> ClusterHealthSummary:
        """Get cluster health summary.

        Returns:
            ClusterHealthSummary with aggregated health data
        """
        summary = ClusterHealthSummary()
        summary.by_provider = {}

        for node_id, health in self.node_health.items():
            summary.total_nodes += 1

            # Count by state
            if health.state == NodeHealthState.HEALTHY:
                summary.healthy += 1
            elif health.state == NodeHealthState.DEGRADED:
                summary.degraded += 1
            elif health.state == NodeHealthState.UNHEALTHY:
                summary.unhealthy += 1
            elif health.state == NodeHealthState.OFFLINE:
                summary.offline += 1
            elif health.state == NodeHealthState.RETIRED:
                summary.retired += 1

            # Count by provider
            provider_name = health.provider.value if health.provider else "unknown"
            if provider_name not in summary.by_provider:
                summary.by_provider[provider_name] = {
                    "total": 0,
                    "healthy": 0,
                    "available": 0,
                }
            summary.by_provider[provider_name]["total"] += 1
            if health.state == NodeHealthState.HEALTHY:
                summary.by_provider[provider_name]["healthy"] += 1
            if health.is_available():
                summary.by_provider[provider_name]["available"] += 1

            # Aggregate capacity
            if health.instance:
                summary.total_gpus += health.instance.gpu_count
                summary.total_cpu_cores += health.instance.cpu_count
                summary.hourly_cost += health.instance.hourly_cost

                if health.is_available():
                    summary.available_gpus += health.instance.gpu_count
                    summary.available_cpu_cores += health.instance.cpu_count

        return summary

    def get_available_nodes(self) -> list[str]:
        """Get list of nodes available for task assignment."""
        return [
            node_id
            for node_id, health in self.node_health.items()
            if health.is_available()
        ]

    def get_nodes_by_state(self, state: NodeHealthState) -> list[str]:
        """Get nodes in a specific health state."""
        return [
            node_id
            for node_id, health in self.node_health.items()
            if health.state == state
        ]

    def get_underutilized_nodes(
        self,
        gpu_threshold: float = 20.0,
        cpu_threshold: float = 30.0,
    ) -> list[str]:
        """Get nodes that are underutilized.

        Args:
            gpu_threshold: GPU utilization below this is underutilized
            cpu_threshold: CPU utilization below this is underutilized

        Returns:
            List of node IDs that are underutilized
        """
        underutilized = []
        for node_id, health in self.node_health.items():
            if not health.is_available():
                continue

            # Check GPU utilization if node has GPU
            if health.instance and health.instance.gpu_count > 0:
                if health.gpu_percent < gpu_threshold:
                    underutilized.append(node_id)
            else:
                # CPU-only node
                if health.cpu_percent < cpu_threshold:
                    underutilized.append(node_id)

        return underutilized

    def get_overloaded_nodes(
        self,
        gpu_threshold: float = 95.0,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
    ) -> list[str]:
        """Get nodes that are overloaded.

        Returns:
            List of node IDs that are overloaded
        """
        overloaded = []
        for node_id, health in self.node_health.items():
            if not health.is_available():
                continue

            if (
                health.gpu_percent > gpu_threshold
                or health.cpu_percent > cpu_threshold
                or health.memory_percent > memory_threshold
            ):
                overloaded.append(node_id)

        return overloaded

    def get_node_health(self, node_id: str) -> NodeHealthDetails | None:
        """Get health details for a specific node."""
        return self.node_health.get(node_id)

    def mark_retired(self, node_id: str) -> bool:
        """Mark a node as retired (removed from active use)."""
        if node_id in self.node_health:
            self.node_health[node_id].state = NodeHealthState.RETIRED
            logger.info(f"[HealthCheckOrchestrator] Marked {node_id} as retired")
            return True
        return False

    def unmark_retired(self, node_id: str) -> bool:
        """Unmark a retired node."""
        if node_id in self.node_health:
            if self.node_health[node_id].state == NodeHealthState.RETIRED:
                self.node_health[node_id].state = NodeHealthState.OFFLINE
                logger.info(f"[HealthCheckOrchestrator] Unmarked {node_id} from retired")
                return True
        return False


# Global instance
_health_orchestrator: HealthCheckOrchestrator | None = None


def get_health_orchestrator() -> HealthCheckOrchestrator:
    """Get or create the global health check orchestrator."""
    global _health_orchestrator

    if _health_orchestrator is None:
        _health_orchestrator = HealthCheckOrchestrator()

    return _health_orchestrator


async def get_cluster_health() -> ClusterHealthSummary:
    """Get cluster health summary.

    Convenience function that uses the global orchestrator.
    """
    return await get_health_orchestrator().get_cluster_health()


def get_available_nodes() -> list[str]:
    """Get list of available nodes.

    Convenience function that uses the global orchestrator.
    """
    return get_health_orchestrator().get_available_nodes()
