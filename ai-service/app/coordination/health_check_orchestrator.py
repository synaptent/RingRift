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
    health = await orchestrator.check_node("my-node-1")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.core.async_context import safe_create_task
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

# December 2025: Import canonical NodeHealthState from node_status.py
# to avoid duplicate enum definitions
from app.coordination.node_status import NodeHealthState

logger = logging.getLogger(__name__)


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
    last_failure_time: float = 0.0  # December 2025: For failure decay

    # Instance info
    instance: ProviderInstance | None = None

    def decay_failures(self, decay_half_life_hours: float = 1.0) -> None:
        """Decay consecutive failures over time.

        Nodes recover reputation over time - a node that had failures
        an hour ago shouldn't be penalized as heavily as one that just failed.

        Args:
            decay_half_life_hours: Time in hours for failures to halve (default: 1 hour)
        """
        if self.consecutive_failures > 0 and self.last_failure_time > 0:
            elapsed_hours = (time.time() - self.last_failure_time) / 3600
            decay_factor = 0.5 ** (elapsed_hours / decay_half_life_hours)
            self.consecutive_failures = max(0, int(self.consecutive_failures * decay_factor))

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

    Check intervals are configurable via environment variables (December 28, 2025):
    - RINGRIFT_P2P_CHECK_INTERVAL (default: 60)
    - RINGRIFT_SSH_CHECK_INTERVAL (default: 120)
    - RINGRIFT_PROVIDER_CHECK_INTERVAL (default: 120)
    - RINGRIFT_UTILIZATION_CHECK_INTERVAL (default: 60)
    """

    # Check intervals (seconds) - loaded from centralized defaults (December 28, 2025)
    try:
        from app.config.coordination_defaults import HealthCheckOrchestratorDefaults
        P2P_CHECK_INTERVAL = HealthCheckOrchestratorDefaults.P2P_CHECK_INTERVAL
        SSH_CHECK_INTERVAL = HealthCheckOrchestratorDefaults.SSH_CHECK_INTERVAL
        PROVIDER_CHECK_INTERVAL = HealthCheckOrchestratorDefaults.PROVIDER_CHECK_INTERVAL
        UTILIZATION_CHECK_INTERVAL = HealthCheckOrchestratorDefaults.UTILIZATION_CHECK_INTERVAL
    except ImportError:
        # Fallback for standalone testing
        P2P_CHECK_INTERVAL = 60
        SSH_CHECK_INTERVAL = 120
        PROVIDER_CHECK_INTERVAL = 120
        UTILIZATION_CHECK_INTERVAL = 60

    def __init__(
        self,
        check_interval: float = 60.0,
        p2p_port: int | None = None,
    ):
        """Initialize health check orchestrator.

        Args:
            check_interval: Seconds between health check cycles
            p2p_port: P2P daemon port to check. If None, uses get_p2p_port()
                     (default 8770, override with RINGRIFT_P2P_PORT env var)
        """
        from app.config.cluster_config import get_p2p_port
        self.check_interval = check_interval
        self.p2p_port = p2p_port if p2p_port is not None else get_p2p_port()

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
        self._task = safe_create_task(
            self._check_loop(),
            name="health_check_loop",
        )

        # December 2025: Subscribe to P2P events for fast failure detection
        self._subscribe_to_events()

        logger.info(
            f"[HealthCheckOrchestrator] Started (interval={self.check_interval}s)"
        )

    def _subscribe_to_events(self) -> None:
        """Subscribe to P2P node death events for fast failure detection.

        December 2025: Reduces failure detection from 120s to ~10s by responding
        to P2P heartbeat failures immediately rather than waiting for next health check.
        """
        try:
            from app.coordination.event_router import get_router
            from app.distributed.data_events import DataEventType

            router = get_router()
            router.subscribe(DataEventType.P2P_NODE_DEAD, self._on_node_dead)
            router.subscribe(DataEventType.P2P_NODES_DEAD, self._on_nodes_dead)
            router.subscribe(DataEventType.HOST_OFFLINE, self._on_node_dead)
            logger.info("[HealthCheckOrchestrator] Subscribed to P2P node death events")
        except ImportError as e:
            logger.debug(f"[HealthCheckOrchestrator] Event router not available: {e}")
        except Exception as e:
            logger.warning(f"[HealthCheckOrchestrator] Failed to subscribe to events: {e}")

    async def _on_node_dead(self, event) -> None:
        """Handle P2P_NODE_DEAD or HOST_OFFLINE - immediate node status update.

        This provides ~10x faster failure detection compared to waiting for the
        next health check cycle (120s default -> ~10s P2P heartbeat timeout).
        """
        # December 29, 2025: Fix 'RouterEvent' object has no attribute 'get' bug
        payload = event if isinstance(event, dict) else getattr(event, "payload", {})
        if not isinstance(payload, dict):
            payload = {}
        node_id = payload.get("node_id")
        if not node_id:
            return

        if node_id in self.node_health:
            health = self.node_health[node_id]
            health.state = NodeHealthState.OFFLINE
            health.consecutive_failures += 1
            health.last_failure_time = time.time()
            health.last_error = payload.get("reason", "P2P dead")
            logger.warning(
                f"[HealthCheckOrchestrator] Node {node_id} marked OFFLINE via P2P event "
                f"(failures={health.consecutive_failures})"
            )
        else:
            # Create entry for unknown node
            self.node_health[node_id] = NodeHealthDetails(
                node_id=node_id,
                provider=Provider.LAMBDA,  # Default, will be updated on next check
                state=NodeHealthState.OFFLINE,
                last_error=payload.get("reason", "P2P dead"),
                consecutive_failures=1,
                last_failure_time=time.time(),
            )
            logger.warning(
                f"[HealthCheckOrchestrator] Unknown node {node_id} marked OFFLINE via P2P event"
            )

    async def _on_nodes_dead(self, event) -> None:
        """Handle batch P2P_NODES_DEAD event."""
        # December 29, 2025: Fix 'RouterEvent' object has no attribute 'get' bug
        payload = event if isinstance(event, dict) else getattr(event, "payload", {})
        if not isinstance(payload, dict):
            payload = {}
        for node_id in payload.get("node_ids", []):
            await self._on_node_dead({"payload": {"node_id": node_id, "reason": "batch P2P dead"}})

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

        1. Decay failure counters (December 2025: nodes recover reputation over time)
        2. Discover all instances from all providers
        3. Run health checks in parallel
        4. Update health state
        5. Emit events if needed

        Returns:
            Dict mapping node_id to health details
        """
        start = time.time()
        logger.info("[HealthCheckOrchestrator] Starting health check cycle...")

        # Phase 0: Decay failure counters (December 2025)
        # Nodes that haven't failed recently should have their failure count reduced
        for health in self.node_health.values():
            health.decay_failures()

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
                    self.node_health[node_id].last_failure_time = time.time()  # December 2025
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

                # P0.2 Dec 2025: Emit NODE_CAPACITY_UPDATED for resource scheduling
                self._emit_node_capacity_updated(
                    node_id=node_id,
                    gpu_memory_gb=util.get("gpu_memory_gb", 0.0),
                    gpu_utilization=details.gpu_percent,
                    cpu_utilization=details.cpu_percent,
                    available_slots=util.get("available_slots", 1),
                )
            except Exception as e:
                logger.debug(f"[HealthCheckOrchestrator] Utilization check failed for {node_id}: {e}")

        # Reset or increment failure counter
        if details.state == NodeHealthState.HEALTHY:
            details.consecutive_failures = 0
        else:
            old_details = self.node_health.get(node_id)
            if old_details:
                details.consecutive_failures = old_details.consecutive_failures + 1
            details.last_failure_time = time.time()  # December 2025: Track for decay

        return details

    def _emit_node_capacity_updated(
        self,
        node_id: str,
        gpu_memory_gb: float,
        gpu_utilization: float,
        cpu_utilization: float,
        available_slots: int,
    ) -> None:
        """Emit NODE_CAPACITY_UPDATED event for resource scheduling.

        P0.2 (December 2025): Enables SelfplayScheduler and ResourceMonitoringCoordinator
        to track available capacity for job scheduling.

        Args:
            node_id: Node identifier
            gpu_memory_gb: Available GPU memory in GB
            gpu_utilization: GPU utilization percentage
            cpu_utilization: CPU utilization percentage
            available_slots: Number of available task slots
        """
        try:
            from app.coordination.event_router import publish_sync

            publish_sync("NODE_CAPACITY_UPDATED", {
                "node_id": node_id,
                "gpu_memory_gb": gpu_memory_gb,
                "gpu_utilization": gpu_utilization,
                "cpu_utilization": cpu_utilization,
                "available_slots": available_slots,
                "reason": "health_check",
                "source": "health_check_orchestrator",
            })
        except ImportError:
            pass  # Event router not available
        except Exception as e:
            logger.debug(f"[HealthCheckOrchestrator] Failed to emit capacity update: {e}")

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

    def health_check(self) -> "HealthCheckResult":
        """Perform health check on this orchestrator (CoordinatorProtocol compliance).

        Returns standardized HealthCheckResult for unified monitoring.

        Returns:
            HealthCheckResult with health status and details
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        # Check if orchestrator is running and has data
        is_running = self._running
        has_data = len(self.node_health) > 0
        check_recent = (time.time() - self._last_full_check) < (self.check_interval * 3)

        is_healthy = is_running and (has_data or check_recent)

        if is_healthy:
            status = CoordinatorStatus.RUNNING
            message = ""
        elif is_running and not has_data:
            status = CoordinatorStatus.DEGRADED
            message = "No node health data collected yet"
        else:
            status = CoordinatorStatus.STOPPED
            message = "Health check orchestrator not running"

        # Count node states
        healthy_count = sum(
            1 for h in self.node_health.values()
            if h.state == NodeHealthState.HEALTHY
        )
        total_count = len(self.node_health)

        return HealthCheckResult(
            healthy=is_healthy,
            status=status,
            message=message,
            details={
                "is_running": is_running,
                "nodes_tracked": total_count,
                "nodes_healthy": healthy_count,
                "last_check_seconds_ago": round(time.time() - self._last_full_check, 1),
                "check_interval": self.check_interval,
            },
        )


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
