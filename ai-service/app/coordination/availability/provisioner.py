"""Auto-provisioning daemon for cluster capacity management.

This module automatically provisions new instances when cluster capacity
drops below configured thresholds. Respects budget constraints before
provisioning.

Created: Dec 28, 2025
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus

if TYPE_CHECKING:
    from app.coordination.providers.base import GPUType, Instance

logger = logging.getLogger(__name__)


@dataclass
class ProvisionResult:
    """Result of a provisioning attempt."""
    success: bool
    instances_created: int
    instance_ids: list[str] = field(default_factory=list)
    error: str | None = None
    provider: str = ""
    gpu_type: str = ""
    cost_per_hour: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for event emission."""
        return {
            "success": self.success,
            "instances_created": self.instances_created,
            "instance_ids": self.instance_ids,
            "error": self.error,
            "provider": self.provider,
            "gpu_type": self.gpu_type,
            "cost_per_hour": self.cost_per_hour,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProvisionerConfig:
    """Configuration for Provisioner.

    December 2025: Simplified - no longer inherits from DaemonConfig.
    HandlerBase uses cycle_interval directly.
    """
    check_interval_seconds: int = 300  # 5 minutes
    min_gpu_capacity: int = 4  # Minimum active GPU nodes
    target_gpu_capacity: int = 10  # Target GPU nodes
    max_provision_per_cycle: int = 2  # Max instances to create per cycle
    preferred_provider: str = "lambda"
    fallback_providers: list[str] = field(default_factory=lambda: ["vast", "runpod"])
    preferred_gpu_types: list[str] = field(
        default_factory=lambda: ["GH200_96GB", "H100_80GB", "A100_80GB"]
    )
    wait_after_provision_seconds: float = 120.0
    dry_run: bool = False  # Log actions without executing


@dataclass
class ClusterCapacity:
    """Current cluster capacity metrics."""
    total_gpu_nodes: int = 0
    active_gpu_nodes: int = 0
    healthy_gpu_nodes: int = 0
    total_gpus: int = 0
    gpu_utilization: float = 0.0
    providers: dict[str, int] = field(default_factory=dict)


class Provisioner(HandlerBase):
    """Auto-provisioning daemon for maintaining cluster capacity.

    December 2025: Migrated to HandlerBase pattern.
    - Uses HandlerBase singleton (get_instance/reset_instance)
    - Uses _stats for metrics tracking
    - Inherits safe event emission from HandlerBase (via SafeEventEmitterMixin)

    Monitors cluster GPU capacity and automatically provisions new
    instances when capacity drops below minimum thresholds. Respects
    budget constraints via CapacityPlanner integration.

    Example:
        provisioner = Provisioner()
        await provisioner.start()
    """

    _event_source = "Provisioner"

    def __init__(self, config: ProvisionerConfig | None = None):
        self._daemon_config = config or ProvisionerConfig()

        super().__init__(
            name="Provisioner",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        self._provision_history: list[ProvisionResult] = []
        self._pending_provisions: int = 0

    @property
    def config(self) -> ProvisionerConfig:
        """Get daemon configuration."""
        return self._daemon_config

    def _get_event_subscriptions(self) -> dict:
        """Subscribe to capacity-related events."""
        return {
            "CAPACITY_LOW": self._on_capacity_low,
            "NODE_FAILED_PERMANENTLY": self._on_node_failed,
        }

    async def _on_capacity_low(self, event: dict) -> None:
        """Handle CAPACITY_LOW event."""
        payload = event.get("payload", event)
        needed = payload.get("needed_gpus", 1)
        logger.info(f"Provisioner: Capacity low, need {needed} GPUs")
        self._pending_provisions = max(self._pending_provisions, needed)

    async def _on_node_failed(self, event: dict) -> None:
        """Handle NODE_FAILED_PERMANENTLY event."""
        payload = event.get("payload", event)
        node_id = payload.get("node_id")
        logger.info(f"Provisioner: Node {node_id} failed permanently, may need replacement")
        # Don't auto-provision here; let the capacity check decide

    async def _run_cycle(self) -> None:
        """Run one provisioning cycle."""
        # Get current capacity
        capacity = await self._get_cluster_capacity()

        # Check if we need to provision
        if capacity.healthy_gpu_nodes >= self.config.min_gpu_capacity:
            if self._pending_provisions == 0:
                logger.debug(
                    f"Provisioner: Capacity OK ({capacity.healthy_gpu_nodes} "
                    f">= {self.config.min_gpu_capacity} min)"
                )
                return

        # Calculate how many nodes we need
        needed = max(
            self.config.min_gpu_capacity - capacity.healthy_gpu_nodes,
            self._pending_provisions,
        )
        needed = min(needed, self.config.max_provision_per_cycle)

        if needed <= 0:
            self._pending_provisions = 0
            return

        # Check budget before provisioning
        if not await self._check_budget(needed):
            logger.warning("Provisioner: Budget exceeded, skipping provisioning")
            await self._emit_budget_exceeded()
            return

        # Attempt provisioning
        logger.info(f"Provisioner: Attempting to provision {needed} GPU node(s)")

        result = await self._provision_nodes(needed)

        if result.success:
            self._pending_provisions = max(0, self._pending_provisions - result.instances_created)
            await self._emit_nodes_provisioned(result)
        else:
            await self._emit_provision_failed(result)

        self._provision_history.append(result)

    async def _get_cluster_capacity(self) -> ClusterCapacity:
        """Get current cluster capacity metrics."""
        capacity = ClusterCapacity()

        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()

            for node_id, node in nodes.items():
                if getattr(node, "is_gpu_node", False):
                    capacity.total_gpu_nodes += 1
                    provider = getattr(node, "provider", "unknown")
                    capacity.providers[provider] = capacity.providers.get(provider, 0) + 1

                    if getattr(node, "is_active", True):
                        capacity.active_gpu_nodes += 1

            # Check health via NodeMonitor
            try:
                from .node_monitor import get_node_monitor

                monitor = get_node_monitor()
                statuses = monitor.get_all_node_statuses()

                for node_id, status in statuses.items():
                    if status.get("healthy", False):
                        node = nodes.get(node_id)
                        if node and getattr(node, "is_gpu_node", False):
                            capacity.healthy_gpu_nodes += 1

            except Exception as e:
                logger.warning(f"Failed to get health status: {e}")
                # Fall back to active count
                capacity.healthy_gpu_nodes = capacity.active_gpu_nodes

        except Exception as e:
            logger.error(f"Failed to get cluster capacity: {e}")

        return capacity

    async def _check_budget(self, count: int) -> bool:
        """Check if we have budget to provision."""
        try:
            from .capacity_planner import get_capacity_planner

            planner = get_capacity_planner()
            return await planner.should_scale_up(count)
        except Exception as e:
            logger.warning(f"Budget check failed: {e}")
            # Default to allowing provisioning
            return True

    async def _provision_nodes(self, count: int) -> ProvisionResult:
        """Provision new nodes across providers."""
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would provision {count} node(s)")
            return ProvisionResult(
                success=True,
                instances_created=0,
                error="dry_run",
            )

        # Try preferred provider first
        providers_to_try = [self.config.preferred_provider] + self.config.fallback_providers

        for provider_name in providers_to_try:
            result = await self._provision_from_provider(provider_name, count)
            if result.success:
                return result

        return ProvisionResult(
            success=False,
            instances_created=0,
            error="All providers failed",
        )

    async def _provision_from_provider(
        self,
        provider_name: str,
        count: int,
    ) -> ProvisionResult:
        """Provision nodes from a specific provider."""
        try:
            from app.coordination.providers.registry import get_provider
            from app.coordination.providers.base import GPUType

            provider = get_provider(provider_name)
            if not provider or not provider.is_configured():
                return ProvisionResult(
                    success=False,
                    instances_created=0,
                    error=f"Provider {provider_name} not configured",
                    provider=provider_name,
                )

            # Try preferred GPU types in order
            for gpu_type_str in self.config.preferred_gpu_types:
                try:
                    gpu_type = GPUType[gpu_type_str]
                except KeyError:
                    continue

                try:
                    instances = await provider.scale_up(
                        gpu_type=gpu_type,
                        count=count,
                        name_prefix="ringrift-auto",
                    )

                    if instances:
                        total_cost = sum(inst.cost_per_hour for inst in instances)

                        logger.info(
                            f"Provisioned {len(instances)} {gpu_type.value} instance(s) "
                            f"from {provider_name} (${total_cost:.2f}/hr)"
                        )

                        # Wait for instances to be ready
                        await asyncio.sleep(self.config.wait_after_provision_seconds)

                        # Register new nodes in cluster config
                        await self._register_new_nodes(instances)

                        return ProvisionResult(
                            success=True,
                            instances_created=len(instances),
                            instance_ids=[inst.id for inst in instances],
                            provider=provider_name,
                            gpu_type=gpu_type.value,
                            cost_per_hour=total_cost,
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to provision {gpu_type.value} from {provider_name}: {e}"
                    )
                    continue

            return ProvisionResult(
                success=False,
                instances_created=0,
                error=f"No available GPU types from {provider_name}",
                provider=provider_name,
            )

        except Exception as e:
            return ProvisionResult(
                success=False,
                instances_created=0,
                error=str(e),
                provider=provider_name,
            )

    async def _register_new_nodes(self, instances: list[Instance]) -> None:
        """Register newly provisioned nodes in cluster configuration."""
        from app.config.cluster_config import add_or_update_node

        for instance in instances:
            logger.info(
                f"Registering new node: id={instance.id}, ip={instance.ip_address}, "
                f"gpu={instance.gpu_type.value}"
            )
            # Build node config based on instance info
            node_config = {
                "status": "setup",  # Will become 'ready' when P2P connects
                "ssh_host": instance.ip_address,
                "ssh_user": "root",  # Most cloud providers use root
                "gpu": instance.gpu_type.value if instance.gpu_type else "",
                "gpu_vram_gb": self._get_gpu_vram(instance.gpu_type),
                "cuda_capable": instance.gpu_type is not None,
                "selfplay_enabled": True,
                "training_enabled": self._get_gpu_vram(instance.gpu_type) >= 48,
                "role": "gpu-worker" if instance.gpu_type else "worker",
            }

            # Generate node name from instance ID
            node_name = self._generate_node_name(instance)

            if add_or_update_node(node_name, node_config):
                logger.info(f"Added node {node_name} to cluster config")
            else:
                logger.warning(f"Failed to add node {node_name} to cluster config")

    def _generate_node_name(self, instance: "Instance") -> str:
        """Generate a node name from instance info."""
        # Use provider-based naming convention
        provider = self.config.preferred_provider
        if hasattr(instance, "id") and instance.id:
            # Truncate ID if too long
            short_id = instance.id[:8] if len(instance.id) > 8 else instance.id
            return f"{provider}-{short_id}"
        return f"{provider}-{id(instance)}"

    def _get_gpu_vram(self, gpu_type: "GPUType | None") -> int:
        """Get VRAM in GB for a GPU type."""
        if gpu_type is None:
            return 0
        # Map GPU types to VRAM (approximations)
        vram_map = {
            "GH200_96GB": 96,
            "H100_80GB": 80,
            "H100_SXM": 80,
            "A100_80GB": 80,
            "A100_40GB": 40,
            "L40S": 48,
            "L40": 48,
            "RTX_4090": 24,
            "RTX_3090": 24,
            "RTX_5090": 32,
            "A40": 48,
        }
        return vram_map.get(str(gpu_type.value), 24)

    async def _emit_nodes_provisioned(self, result: ProvisionResult) -> None:
        """Emit NODE_PROVISIONED events."""
        for instance_id in result.instance_ids:
            await self._safe_emit_event_async(
                "NODE_PROVISIONED",
                {
                    "instance_id": instance_id,
                    "provider": result.provider,
                    "gpu_type": result.gpu_type,
                    "cost_per_hour": result.cost_per_hour / len(result.instance_ids),
                },
            )

    async def _emit_provision_failed(self, result: ProvisionResult) -> None:
        """Emit PROVISION_FAILED event."""
        await self._safe_emit_event_async(
            "PROVISION_FAILED",
            {
                "error": result.error,
                "provider": result.provider,
            },
        )

    async def _emit_budget_exceeded(self) -> None:
        """Emit BUDGET_EXCEEDED event."""
        await self._safe_emit_event_async(
            "BUDGET_EXCEEDED",
            {
                "reason": "provisioning_blocked",
                "timestamp": datetime.now().isoformat(),
            },
        )

    def get_provision_history(self, limit: int = 20) -> list[dict]:
        """Get recent provisioning history."""
        return [r.to_dict() for r in self._provision_history[-limit:]]

    def health_check(self) -> HealthCheckResult:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult with status based on recent provision failures.
        """
        recent_failures = sum(
            1 for r in self._provision_history[-10:]
            if not r.success
        )

        is_healthy = recent_failures < 5

        return HealthCheckResult(
            healthy=is_healthy,
            status=CoordinatorStatus.RUNNING if self._running else CoordinatorStatus.STOPPED,
            message=f"Provisioner: {self._pending_provisions} pending, {len(self._provision_history)} total attempts",
            details={
                "pending_provisions": self._pending_provisions,
                "total_provisions": len(self._provision_history),
                "recent_failures": recent_failures,
                "cycles_completed": self._stats.cycles_completed,
                "errors_count": self._stats.errors_count,
            },
        )


# =============================================================================
# Singleton Access (using HandlerBase class methods)
# =============================================================================


def get_provisioner() -> Provisioner:
    """Get or create the singleton Provisioner instance.

    Uses HandlerBase.get_instance() for thread-safe singleton access.
    """
    return Provisioner.get_instance()


def reset_provisioner() -> None:
    """Reset the singleton instance (for testing).

    Uses HandlerBase.reset_instance() for thread-safe cleanup.
    """
    Provisioner.reset_instance()
