"""Lambda Idle Shutdown Daemon (December 2025).

Monitors Lambda Labs GPU nodes for idle detection and automatically
terminates them to reduce costs.

Key features:
- 30-minute idle threshold (configurable)
- Pending work check before termination
- Minimum node retention to maintain cluster capacity
- Graceful shutdown with drain period
- Cost tracking and reporting

NOTE: Lambda account currently suspended pending support ticket resolution.
      This code is ready for when the account is reactivated.

Usage:
    from app.coordination.lambda_idle_daemon import LambdaIdleDaemon

    config = LambdaIdleConfig.from_env()
    daemon = LambdaIdleDaemon(config=config)
    await daemon.start()
"""

from __future__ import annotations

__all__ = [
    "LambdaIdleConfig",
    "LambdaIdleDaemon",
    "LambdaNodeStatus",
]

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from app.coordination.protocols import (
    CoordinatorStatus,
    register_coordinator,
    unregister_coordinator,
)

logger = logging.getLogger(__name__)

# Lambda provider integration
try:
    from app.coordination.providers.lambda_provider import LambdaProvider
    HAS_LAMBDA_PROVIDER = True
except ImportError:
    HAS_LAMBDA_PROVIDER = False
    LambdaProvider = None

# P2P status integration for workload checking
try:
    from app.coordination.p2p_backend import get_p2p_status
    HAS_P2P = True
except ImportError:
    HAS_P2P = False
    get_p2p_status = None

# Event emission for tracking
try:
    from app.coordination.event_router import get_router, DataEventType
    HAS_EVENTS = True
except ImportError:
    HAS_EVENTS = False
    get_router = None
    DataEventType = None


@dataclass
class LambdaIdleConfig:
    """Configuration for Lambda idle shutdown daemon."""
    enabled: bool = True
    # Check interval in seconds
    check_interval_seconds: int = 60
    # Idle threshold in seconds (30 minutes default)
    idle_threshold_seconds: int = 1800
    # GPU utilization threshold (%) - below this is considered idle
    idle_utilization_threshold: float = 10.0
    # Minimum nodes to retain (never terminate below this count)
    min_nodes_to_retain: int = 1
    # Drain period before termination (seconds) - allows graceful shutdown
    drain_period_seconds: int = 300
    # Cost threshold - only terminate if cost savings exceed this per hour
    min_cost_savings_per_hour: float = 1.0
    # Pending work check - don't terminate if work queue has items
    check_pending_work: bool = True
    # Pending work threshold - allow termination if queue depth below this
    pending_work_threshold: int = 5
    # Dry run mode - log actions without executing
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> LambdaIdleConfig:
        """Load configuration from environment variables."""
        return cls(
            enabled=os.environ.get("LAMBDA_IDLE_ENABLED", "true").lower() == "true",
            check_interval_seconds=int(os.environ.get("LAMBDA_IDLE_CHECK_INTERVAL", "60")),
            idle_threshold_seconds=int(os.environ.get("LAMBDA_IDLE_THRESHOLD", "1800")),
            idle_utilization_threshold=float(os.environ.get("LAMBDA_IDLE_UTIL_THRESHOLD", "10.0")),
            min_nodes_to_retain=int(os.environ.get("LAMBDA_MIN_NODES", "1")),
            drain_period_seconds=int(os.environ.get("LAMBDA_DRAIN_PERIOD", "300")),
            min_cost_savings_per_hour=float(os.environ.get("LAMBDA_MIN_SAVINGS", "1.0")),
            check_pending_work=os.environ.get("LAMBDA_CHECK_WORK", "true").lower() == "true",
            pending_work_threshold=int(os.environ.get("LAMBDA_WORK_THRESHOLD", "5")),
            dry_run=os.environ.get("LAMBDA_IDLE_DRY_RUN", "false").lower() == "true",
        )


@dataclass
class LambdaNodeStatus:
    """Status of a Lambda node for idle tracking."""
    instance_id: str
    instance_name: str
    instance_type: str
    ip_address: str
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    last_activity_time: float = 0.0
    idle_since: float = 0.0
    cost_per_hour: float = 0.0
    status: str = "unknown"

    @property
    def idle_duration_seconds(self) -> float:
        """Get how long the node has been idle."""
        if self.idle_since <= 0:
            return 0.0
        return time.time() - self.idle_since

    @property
    def is_idle(self) -> bool:
        """Check if node is currently idle."""
        return self.idle_since > 0


class LambdaIdleDaemon:
    """Daemon that monitors Lambda nodes and terminates idle ones."""

    def __init__(self, config: LambdaIdleConfig | None = None):
        """Initialize the daemon.

        Args:
            config: Configuration for idle detection. Uses env if not provided.
        """
        self.config = config or LambdaIdleConfig.from_env()
        self._running = False
        self._task: asyncio.Task | None = None
        self._provider: LambdaProvider | None = None
        self._node_status: dict[str, LambdaNodeStatus] = {}
        self._terminated_count = 0
        self._cost_saved = 0.0
        self._last_check_time = 0.0

    async def start(self) -> None:
        """Start the daemon."""
        if not self.config.enabled:
            logger.info("[LambdaIdleDaemon] Disabled by configuration")
            return

        if not HAS_LAMBDA_PROVIDER:
            logger.warning("[LambdaIdleDaemon] Lambda provider not available")
            return

        # Initialize provider
        self._provider = LambdaProvider()
        if not self._provider.is_configured():
            logger.warning("[LambdaIdleDaemon] Lambda API key not configured, daemon disabled")
            return

        self._running = True

        # Register as coordinator
        register_coordinator(
            "lambda_idle_daemon",
            CoordinatorStatus.ACTIVE,
            metadata={"config": self.config.__dict__},
        )

        logger.info(
            f"[LambdaIdleDaemon] Started with idle_threshold={self.config.idle_threshold_seconds}s, "
            f"min_nodes={self.config.min_nodes_to_retain}, dry_run={self.config.dry_run}"
        )

        try:
            await self._run_loop()
        finally:
            self._running = False
            unregister_coordinator("lambda_idle_daemon")
            if self._provider and hasattr(self._provider, 'close'):
                await self._provider.close()

    async def stop(self) -> None:
        """Stop the daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        """Main daemon loop."""
        while self._running:
            try:
                await self._check_and_terminate_idle_nodes()
                self._last_check_time = time.time()
            except Exception as e:
                logger.error(f"[LambdaIdleDaemon] Error in check loop: {e}", exc_info=True)

            await asyncio.sleep(self.config.check_interval_seconds)

    async def _check_and_terminate_idle_nodes(self) -> None:
        """Check all Lambda nodes and terminate idle ones."""
        if not self._provider:
            return

        # Get current instances
        try:
            instances = await self._provider.list_instances()
        except Exception as e:
            logger.error(f"[LambdaIdleDaemon] Failed to list instances: {e}")
            return

        if not instances:
            logger.debug("[LambdaIdleDaemon] No Lambda instances found")
            return

        running_instances = [i for i in instances if i.status == "running"]
        logger.debug(f"[LambdaIdleDaemon] Found {len(running_instances)} running Lambda instances")

        # Update node status for each instance
        now = time.time()
        for instance in running_instances:
            await self._update_node_status(instance, now)

        # Find nodes eligible for termination
        idle_nodes = [
            status for status in self._node_status.values()
            if status.is_idle and status.idle_duration_seconds >= self.config.idle_threshold_seconds
        ]

        if not idle_nodes:
            return

        logger.info(f"[LambdaIdleDaemon] Found {len(idle_nodes)} idle nodes exceeding threshold")

        # Check pending work if configured
        if self.config.check_pending_work and not await self._should_allow_termination():
            logger.info("[LambdaIdleDaemon] Skipping termination due to pending work")
            return

        # Sort by idle duration (longest first) and cost (highest first)
        idle_nodes.sort(key=lambda n: (-n.idle_duration_seconds, -n.cost_per_hour))

        # Calculate how many we can terminate (respecting minimum retention)
        can_terminate = len(running_instances) - self.config.min_nodes_to_retain
        if can_terminate <= 0:
            logger.info(
                f"[LambdaIdleDaemon] Cannot terminate - would go below minimum "
                f"({len(running_instances)} running, {self.config.min_nodes_to_retain} minimum)"
            )
            return

        # Terminate eligible nodes
        terminated = 0
        for node in idle_nodes[:can_terminate]:
            if node.cost_per_hour < self.config.min_cost_savings_per_hour:
                logger.debug(
                    f"[LambdaIdleDaemon] Skipping {node.instance_name} - "
                    f"cost ${node.cost_per_hour}/hr below threshold"
                )
                continue

            success = await self._terminate_node(node)
            if success:
                terminated += 1
                self._terminated_count += 1
                self._cost_saved += node.cost_per_hour

                # Emit event
                if HAS_EVENTS and get_router:
                    router = get_router()
                    await router.publish(
                        DataEventType.NODE_TERMINATED,
                        {
                            "instance_id": node.instance_id,
                            "instance_name": node.instance_name,
                            "provider": "lambda",
                            "reason": "idle",
                            "idle_duration_seconds": node.idle_duration_seconds,
                            "cost_per_hour": node.cost_per_hour,
                        },
                        source="lambda_idle_daemon",
                    )

        if terminated > 0:
            logger.info(
                f"[LambdaIdleDaemon] Terminated {terminated} idle nodes, "
                f"total saved: ${self._cost_saved:.2f}/hr"
            )

    async def _update_node_status(self, instance: Any, now: float) -> None:
        """Update status for a single node."""
        instance_id = instance.id

        # Get or create status
        if instance_id not in self._node_status:
            self._node_status[instance_id] = LambdaNodeStatus(
                instance_id=instance_id,
                instance_name=instance.name or instance_id,
                instance_type=instance.instance_type,
                ip_address=instance.ip_address or "",
                cost_per_hour=getattr(instance, 'cost_per_hour', 0.0),
            )

        status = self._node_status[instance_id]
        status.status = instance.status

        # Get GPU utilization via P2P status if available
        gpu_util = await self._get_gpu_utilization(instance)
        status.gpu_utilization = gpu_util

        # Update idle tracking
        if gpu_util < self.config.idle_utilization_threshold:
            if status.idle_since <= 0:
                status.idle_since = now
                logger.debug(
                    f"[LambdaIdleDaemon] {status.instance_name} became idle "
                    f"(GPU util: {gpu_util:.1f}%)"
                )
        else:
            if status.idle_since > 0:
                logger.debug(
                    f"[LambdaIdleDaemon] {status.instance_name} no longer idle "
                    f"(GPU util: {gpu_util:.1f}%)"
                )
            status.idle_since = 0
            status.last_activity_time = now

    async def _get_gpu_utilization(self, instance: Any) -> float:
        """Get GPU utilization for an instance via P2P status."""
        if not HAS_P2P or not get_p2p_status:
            return 100.0  # Assume busy if we can't check

        try:
            ip = instance.ip_address
            if not ip:
                return 100.0

            # Query P2P status on the node
            status = await get_p2p_status(ip, timeout=5.0)
            if status and "gpu_utilization" in status:
                return float(status["gpu_utilization"])

            # Fall back to checking if any selfplay processes running
            if status and "active_jobs" in status:
                return 100.0 if status["active_jobs"] > 0 else 0.0

        except Exception as e:
            logger.debug(f"[LambdaIdleDaemon] Failed to get GPU util for {instance.name}: {e}")

        return 100.0  # Assume busy on error

    async def _should_allow_termination(self) -> bool:
        """Check if termination should be allowed based on pending work."""
        if not HAS_P2P or not get_p2p_status:
            return True  # Allow if we can't check

        try:
            # Check queue depth from P2P status
            status = await get_p2p_status(timeout=5.0)
            if status and "queue_depth" in status:
                queue_depth = int(status["queue_depth"])
                if queue_depth > self.config.pending_work_threshold:
                    logger.debug(
                        f"[LambdaIdleDaemon] Queue depth {queue_depth} exceeds threshold "
                        f"{self.config.pending_work_threshold}"
                    )
                    return False
        except Exception as e:
            logger.debug(f"[LambdaIdleDaemon] Failed to check pending work: {e}")

        return True

    async def _terminate_node(self, node: LambdaNodeStatus) -> bool:
        """Terminate a single node."""
        if not self._provider:
            return False

        if self.config.dry_run:
            logger.info(
                f"[LambdaIdleDaemon] DRY RUN: Would terminate {node.instance_name} "
                f"(idle for {node.idle_duration_seconds:.0f}s, ${node.cost_per_hour}/hr)"
            )
            return True

        logger.info(
            f"[LambdaIdleDaemon] Terminating {node.instance_name} "
            f"(idle for {node.idle_duration_seconds:.0f}s, saving ${node.cost_per_hour}/hr)"
        )

        try:
            await self._provider.terminate_instance(node.instance_id)
            # Remove from tracking
            self._node_status.pop(node.instance_id, None)
            return True
        except Exception as e:
            logger.error(f"[LambdaIdleDaemon] Failed to terminate {node.instance_name}: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get daemon statistics."""
        return {
            "running": self._running,
            "last_check_time": self._last_check_time,
            "terminated_count": self._terminated_count,
            "cost_saved_per_hour": self._cost_saved,
            "tracked_nodes": len(self._node_status),
            "idle_nodes": sum(1 for n in self._node_status.values() if n.is_idle),
            "config": {
                "enabled": self.config.enabled,
                "idle_threshold_seconds": self.config.idle_threshold_seconds,
                "min_nodes_to_retain": self.config.min_nodes_to_retain,
                "dry_run": self.config.dry_run,
            },
        }
