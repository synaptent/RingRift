"""Vast.ai Idle Shutdown Daemon (December 2025).

Monitors Vast.ai GPU instances for idle detection and automatically
terminates them to reduce costs. Especially important for ephemeral
marketplace instances that charge by the hour.

Key features:
- 15-minute idle threshold (configurable, shorter than Lambda due to hourly billing)
- Pending work check before termination
- Minimum node retention to maintain cluster capacity
- Graceful shutdown with drain period
- Cost tracking and reporting
- Ephemeral node prioritization

Adapted from LambdaIdleDaemon pattern.

Usage:
    from app.coordination.vast_idle_daemon import VastIdleDaemon

    config = VastIdleConfig.from_env()
    daemon = VastIdleDaemon(config=config)
    await daemon.start()
"""

from __future__ import annotations

__all__ = [
    "VastIdleConfig",
    "VastIdleDaemon",
    "VastNodeStatus",
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

# Vast provider integration
try:
    from app.coordination.providers.vast_provider import VastProvider
    HAS_VAST_PROVIDER = True
except ImportError:
    HAS_VAST_PROVIDER = False
    VastProvider = None

# P2P status integration for workload checking
try:
    from app.coordination.p2p_backend import get_p2p_status
    HAS_P2P = True
except ImportError:
    HAS_P2P = False

    async def get_p2p_status() -> dict:
        return {}

# Event emission for tracking
try:
    from app.coordination.event_router import get_router, DataEventType
    HAS_EVENTS = True
except ImportError:
    HAS_EVENTS = False
    get_router = None
    DataEventType = None


@dataclass
class VastIdleConfig:
    """Configuration for Vast.ai idle shutdown daemon."""
    enabled: bool = True
    # Check interval in seconds
    check_interval_seconds: int = 60
    # Idle threshold in seconds (15 minutes - shorter than Lambda due to hourly billing)
    idle_threshold_seconds: int = 900
    # GPU utilization threshold (%) - below this is considered idle
    idle_utilization_threshold: float = 10.0
    # Minimum nodes to retain per GPU type (never terminate all of a type)
    min_nodes_to_retain: int = 0
    # Drain period before termination (seconds) - allows graceful shutdown
    drain_period_seconds: int = 180
    # Cost threshold - only terminate if cost savings exceed this per hour
    min_cost_savings_per_hour: float = 0.10  # Lower threshold for Vast
    # Pending work check - don't terminate if work queue has items
    check_pending_work: bool = True
    # Pending work threshold - allow termination if queue depth below this
    pending_work_threshold: int = 3
    # Dry run mode - log actions without executing
    dry_run: bool = False
    # Only terminate instances with these labels (empty = all)
    instance_label_filter: str = ""
    # Prioritize terminating expensive instances first
    prioritize_expensive: bool = True

    @classmethod
    def from_env(cls) -> VastIdleConfig:
        """Load configuration from environment variables."""
        return cls(
            enabled=os.environ.get("VAST_IDLE_ENABLED", "true").lower() == "true",
            check_interval_seconds=int(os.environ.get("VAST_IDLE_CHECK_INTERVAL", "60")),
            idle_threshold_seconds=int(os.environ.get("VAST_IDLE_THRESHOLD", "900")),
            idle_utilization_threshold=float(os.environ.get("VAST_IDLE_UTIL_THRESHOLD", "10.0")),
            min_nodes_to_retain=int(os.environ.get("VAST_MIN_NODES", "0")),
            drain_period_seconds=int(os.environ.get("VAST_DRAIN_PERIOD", "180")),
            min_cost_savings_per_hour=float(os.environ.get("VAST_MIN_SAVINGS", "0.10")),
            check_pending_work=os.environ.get("VAST_CHECK_WORK", "true").lower() == "true",
            pending_work_threshold=int(os.environ.get("VAST_WORK_THRESHOLD", "3")),
            dry_run=os.environ.get("VAST_IDLE_DRY_RUN", "false").lower() == "true",
            instance_label_filter=os.environ.get("VAST_LABEL_FILTER", ""),
            prioritize_expensive=os.environ.get("VAST_PRIORITIZE_EXPENSIVE", "true").lower() == "true",
        )


@dataclass
class VastNodeStatus:
    """Status of a Vast.ai node for idle tracking."""
    instance_id: str
    instance_name: str
    gpu_name: str
    ip_address: str
    ssh_port: int = 22
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    last_activity_time: float = 0.0
    idle_since: float = 0.0
    cost_per_hour: float = 0.0
    status: str = "unknown"
    # Vast-specific fields
    reliability: float = 1.0
    geolocation: str = ""

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


class VastIdleDaemon:
    """Daemon that monitors Vast.ai nodes and terminates idle ones."""

    def __init__(self, config: VastIdleConfig | None = None):
        """Initialize the daemon.

        Args:
            config: Configuration for idle detection. Uses env if not provided.
        """
        self.config = config or VastIdleConfig.from_env()
        self._running = False
        self._task: asyncio.Task | None = None
        self._provider: VastProvider | None = None
        self._node_status: dict[str, VastNodeStatus] = {}
        self._terminated_count = 0
        self._cost_saved = 0.0
        self._last_check_time = 0.0
        self._draining_nodes: dict[str, float] = {}  # instance_id -> drain_started

    async def start(self) -> None:
        """Start the daemon."""
        if not self.config.enabled:
            logger.info("[VastIdleDaemon] Disabled by configuration")
            return

        if not HAS_VAST_PROVIDER:
            logger.warning("[VastIdleDaemon] Vast provider not available")
            return

        # Initialize provider
        self._provider = VastProvider()
        if not self._provider.is_configured():
            logger.warning("[VastIdleDaemon] Vast.ai CLI not configured, daemon disabled")
            return

        self._running = True

        # Register as coordinator
        register_coordinator(
            "vast_idle_daemon",
            CoordinatorStatus.ACTIVE,
            metadata={"config": self.config.__dict__},
        )

        logger.info(
            f"[VastIdleDaemon] Started with idle_threshold={self.config.idle_threshold_seconds}s, "
            f"min_nodes={self.config.min_nodes_to_retain}, dry_run={self.config.dry_run}"
        )

        try:
            await self._run_loop()
        finally:
            self._running = False
            unregister_coordinator("vast_idle_daemon")

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
                logger.error(f"[VastIdleDaemon] Error in check loop: {e}", exc_info=True)

            await asyncio.sleep(self.config.check_interval_seconds)

    async def _get_node_utilization(self, instance: Any) -> tuple[float, float, float]:
        """Get GPU utilization for an instance via SSH.

        Returns:
            Tuple of (utilization_percent, memory_used_gb, memory_total_gb)
        """
        # Try to get utilization via SSH nvidia-smi
        try:
            from app.core.ssh import run_ssh_command_async, SSHConfig

            ssh_config = SSHConfig(
                host=instance.ip_address,
                port=instance.ssh_port or 22,
                user=instance.ssh_user or "root",
            )

            result = await run_ssh_command_async(
                ssh_config,
                "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
                "--format=csv,noheader,nounits",
                timeout=10,
            )

            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split("\n")
                # Sum across GPUs if multiple
                total_util = 0.0
                total_mem_used = 0.0
                total_mem_total = 0.0
                gpu_count = 0

                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 3:
                        total_util += float(parts[0].strip())
                        total_mem_used += float(parts[1].strip()) / 1024  # MB to GB
                        total_mem_total += float(parts[2].strip()) / 1024
                        gpu_count += 1

                if gpu_count > 0:
                    return total_util / gpu_count, total_mem_used, total_mem_total

        except Exception as e:
            logger.debug(f"[VastIdleDaemon] Failed to get utilization for {instance.name}: {e}")

        return 0.0, 0.0, 0.0

    async def _update_node_status(self, instance: Any, now: float) -> None:
        """Update status tracking for an instance."""
        instance_id = instance.id

        # Get current utilization
        util, mem_used, mem_total = await self._get_node_utilization(instance)

        # Get or create status entry
        if instance_id not in self._node_status:
            self._node_status[instance_id] = VastNodeStatus(
                instance_id=instance_id,
                instance_name=instance.name,
                gpu_name=instance.raw_data.get("gpu_name", ""),
                ip_address=instance.ip_address or "",
                ssh_port=instance.ssh_port or 22,
                cost_per_hour=instance.cost_per_hour or 0.0,
                reliability=instance.raw_data.get("reliability", 1.0),
                geolocation=instance.region or "",
            )

        status = self._node_status[instance_id]
        status.gpu_utilization = util
        status.gpu_memory_used_gb = mem_used
        status.gpu_memory_total_gb = mem_total
        status.status = str(instance.status.value) if hasattr(instance.status, "value") else str(instance.status)

        # Check if idle
        is_currently_idle = util < self.config.idle_utilization_threshold

        if is_currently_idle:
            if status.idle_since <= 0:
                # Just became idle
                status.idle_since = now
                logger.debug(f"[VastIdleDaemon] {instance.name} became idle (util={util:.1f}%)")
        else:
            if status.idle_since > 0:
                logger.debug(f"[VastIdleDaemon] {instance.name} became active (util={util:.1f}%)")
            status.idle_since = 0
            status.last_activity_time = now

    async def _should_allow_termination(self) -> bool:
        """Check if termination should be allowed based on pending work."""
        if not HAS_P2P:
            return True

        try:
            status = await get_p2p_status()
            work_queue_size = status.get("work_queue_size", 0)

            if work_queue_size > self.config.pending_work_threshold:
                logger.info(
                    f"[VastIdleDaemon] Work queue has {work_queue_size} items, "
                    f"above threshold {self.config.pending_work_threshold}"
                )
                return False
            return True
        except Exception as e:
            logger.debug(f"[VastIdleDaemon] Failed to check P2P status: {e}")
            return True  # Allow termination if we can't check

    async def _terminate_node(self, node: VastNodeStatus) -> bool:
        """Terminate an idle node."""
        if not self._provider:
            return False

        if self.config.dry_run:
            logger.info(
                f"[VastIdleDaemon] DRY RUN: Would terminate {node.instance_name} "
                f"(idle {node.idle_duration_seconds:.0f}s, ${node.cost_per_hour:.2f}/hr)"
            )
            return True

        # Check if in drain period
        if node.instance_id in self._draining_nodes:
            drain_started = self._draining_nodes[node.instance_id]
            if time.time() - drain_started < self.config.drain_period_seconds:
                remaining = self.config.drain_period_seconds - (time.time() - drain_started)
                logger.debug(f"[VastIdleDaemon] {node.instance_name} draining ({remaining:.0f}s remaining)")
                return False
            # Drain complete, proceed with termination
        else:
            # Start drain period
            self._draining_nodes[node.instance_id] = time.time()
            logger.info(
                f"[VastIdleDaemon] Starting drain period for {node.instance_name} "
                f"({self.config.drain_period_seconds}s)"
            )
            return False

        try:
            results = await self._provider.scale_down([node.instance_id])
            success = results.get(node.instance_id, False)

            if success:
                logger.info(
                    f"[VastIdleDaemon] Terminated {node.instance_name} "
                    f"(idle {node.idle_duration_seconds:.0f}s, saving ${node.cost_per_hour:.2f}/hr)"
                )
                # Clean up
                if node.instance_id in self._draining_nodes:
                    del self._draining_nodes[node.instance_id]
                if node.instance_id in self._node_status:
                    del self._node_status[node.instance_id]
            else:
                logger.warning(f"[VastIdleDaemon] Failed to terminate {node.instance_name}")

            return success
        except Exception as e:
            logger.error(f"[VastIdleDaemon] Error terminating {node.instance_name}: {e}")
            return False

    async def _check_and_terminate_idle_nodes(self) -> None:
        """Check all Vast.ai nodes and terminate idle ones."""
        if not self._provider:
            return

        # Get current instances
        try:
            instances = await self._provider.list_instances()
        except Exception as e:
            logger.error(f"[VastIdleDaemon] Failed to list instances: {e}")
            return

        if not instances:
            logger.debug("[VastIdleDaemon] No Vast.ai instances found")
            return

        # Filter by label if configured
        if self.config.instance_label_filter:
            label = self.config.instance_label_filter
            instances = [
                i for i in instances
                if label in (i.tags.get("label", "") or i.name or "")
            ]

        # Filter to running instances
        from app.coordination.providers.base import InstanceStatus
        running_instances = [
            i for i in instances
            if i.status == InstanceStatus.RUNNING
        ]

        logger.debug(f"[VastIdleDaemon] Found {len(running_instances)} running Vast.ai instances")

        # Update node status for each instance
        now = time.time()
        for instance in running_instances:
            await self._update_node_status(instance, now)

        # Find nodes eligible for termination
        idle_nodes = [
            status for status in self._node_status.values()
            if (
                status.is_idle
                and status.idle_duration_seconds >= self.config.idle_threshold_seconds
                and status.status == "running"
            )
        ]

        if not idle_nodes:
            return

        logger.info(f"[VastIdleDaemon] Found {len(idle_nodes)} idle nodes exceeding threshold")

        # Check pending work if configured
        if self.config.check_pending_work and not await self._should_allow_termination():
            logger.info("[VastIdleDaemon] Skipping termination due to pending work")
            return

        # Sort by priority (expensive first if configured, else longest idle)
        if self.config.prioritize_expensive:
            idle_nodes.sort(key=lambda n: (-n.cost_per_hour, -n.idle_duration_seconds))
        else:
            idle_nodes.sort(key=lambda n: (-n.idle_duration_seconds, -n.cost_per_hour))

        # Calculate how many we can terminate (respecting minimum retention)
        can_terminate = len(running_instances) - self.config.min_nodes_to_retain
        if can_terminate <= 0:
            logger.info(
                f"[VastIdleDaemon] Cannot terminate - would go below minimum "
                f"({len(running_instances)} running, {self.config.min_nodes_to_retain} minimum)"
            )
            return

        # Terminate eligible nodes
        terminated = 0
        for node in idle_nodes[:can_terminate]:
            if node.cost_per_hour < self.config.min_cost_savings_per_hour:
                logger.debug(
                    f"[VastIdleDaemon] Skipping {node.instance_name} - "
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
                    try:
                        router = get_router()
                        await router.publish(
                            DataEventType.NODE_TERMINATED,
                            {
                                "instance_id": node.instance_id,
                                "instance_name": node.instance_name,
                                "provider": "vast",
                                "reason": "idle",
                                "idle_duration_seconds": node.idle_duration_seconds,
                                "cost_per_hour": node.cost_per_hour,
                                "gpu_name": node.gpu_name,
                            },
                            source="vast_idle_daemon",
                        )
                    except Exception as e:
                        logger.debug(f"[VastIdleDaemon] Failed to emit event: {e}")

        if terminated > 0:
            logger.info(
                f"[VastIdleDaemon] Terminated {terminated} idle nodes. "
                f"Total: {self._terminated_count} terminated, ${self._cost_saved:.2f}/hr saved"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get current daemon statistics."""
        return {
            "running": self._running,
            "terminated_count": self._terminated_count,
            "cost_saved_per_hour": self._cost_saved,
            "tracked_nodes": len(self._node_status),
            "draining_nodes": len(self._draining_nodes),
            "last_check_time": self._last_check_time,
            "config": {
                "enabled": self.config.enabled,
                "idle_threshold_seconds": self.config.idle_threshold_seconds,
                "min_nodes_to_retain": self.config.min_nodes_to_retain,
                "dry_run": self.config.dry_run,
            },
        }
