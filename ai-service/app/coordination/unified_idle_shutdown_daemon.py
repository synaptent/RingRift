"""Unified Idle Shutdown Daemon (December 2025).

Provider-agnostic idle detection and shutdown for cloud GPU instances.
Consolidates lambda_idle_daemon.py and vast_idle_daemon.py into a single
implementation with provider-specific configuration.

Key features:
- Provider-agnostic design using CloudProvider interface
- Configurable idle thresholds per provider
- Pending work check before termination
- Minimum node retention to maintain cluster capacity
- Graceful shutdown with drain period
- Cost tracking and reporting

Usage:
    from app.coordination.unified_idle_shutdown_daemon import (
        UnifiedIdleShutdownDaemon,
        IdleShutdownConfig,
    )
    from app.coordination.providers.vast_provider import VastProvider

    config = IdleShutdownConfig.for_provider("vast")
    daemon = UnifiedIdleShutdownDaemon(
        provider=VastProvider(),
        config=config,
    )
    await daemon.start()
"""

from __future__ import annotations

__all__ = [
    "IdleShutdownConfig",
    "NodeIdleStatus",
    "UnifiedIdleShutdownDaemon",
    "get_idle_shutdown_daemon",
    "reset_idle_shutdown_daemon",
]

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.providers.base import (
    CloudProvider,
    Instance,
    InstanceStatus,
    ProviderType,
)

logger = logging.getLogger(__name__)

# Protocol registration
try:
    from app.coordination.protocols import (
        CoordinatorStatus,
        register_coordinator,
        unregister_coordinator,
    )
    HAS_PROTOCOLS = True
except ImportError:
    HAS_PROTOCOLS = False

# P2P status integration for workload checking
try:
    from app.coordination.p2p_backend import get_p2p_status
    HAS_P2P = True
except ImportError:
    HAS_P2P = False

    async def get_p2p_status(*args, **kwargs) -> dict:
        return {}

# Event emission for tracking
try:
    from app.coordination.event_router import get_router, DataEventType
    HAS_EVENTS = True
except ImportError:
    HAS_EVENTS = False
    get_router = None
    DataEventType = None


# Provider-specific default configurations
PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "lambda": {
        "idle_threshold_seconds": 1800,  # 30 minutes
        "min_cost_savings_per_hour": 1.0,
        "drain_period_seconds": 300,
        "min_nodes_to_retain": 1,
        "pending_work_threshold": 5,
        "env_prefix": "LAMBDA",
    },
    "vast": {
        # Jan 2, 2026 (Sprint 3.5): Increased from 900s to 1800s
        # 15 minutes was too aggressive - terminated nodes during normal scheduling gaps
        # between selfplay job batches. 30 minutes provides buffer for job dispatch.
        "idle_threshold_seconds": 1800,  # 30 minutes (increased from 15)
        "min_cost_savings_per_hour": 0.10,
        "drain_period_seconds": 180,
        "min_nodes_to_retain": 0,
        "pending_work_threshold": 3,
        "env_prefix": "VAST",
    },
    "runpod": {
        "idle_threshold_seconds": 1200,  # 20 minutes
        "min_cost_savings_per_hour": 0.50,
        "drain_period_seconds": 240,
        "min_nodes_to_retain": 1,
        "pending_work_threshold": 5,
        "env_prefix": "RUNPOD",
    },
    "vultr": {
        "idle_threshold_seconds": 1800,  # 30 minutes
        "min_cost_savings_per_hour": 0.50,
        "drain_period_seconds": 300,
        "min_nodes_to_retain": 1,
        "pending_work_threshold": 5,
        "env_prefix": "VULTR",
    },
}


@dataclass
class IdleShutdownConfig:
    """Configuration for idle shutdown daemon."""
    enabled: bool = True
    # Check interval in seconds
    check_interval_seconds: int = 60
    # Idle threshold in seconds
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
    # Instance label filter (empty = all)
    instance_label_filter: str = ""
    # Prioritize terminating expensive instances first
    prioritize_expensive: bool = True
    # Provider name for logging and events
    provider_name: str = "unknown"

    @classmethod
    def for_provider(cls, provider_name: str) -> IdleShutdownConfig:
        """Create config with provider-specific defaults."""
        provider_key = provider_name.lower()
        defaults = PROVIDER_DEFAULTS.get(provider_key, {})
        env_prefix = defaults.get("env_prefix", provider_name.upper())

        return cls(
            enabled=os.environ.get(f"{env_prefix}_IDLE_ENABLED", "true").lower() == "true",
            check_interval_seconds=int(os.environ.get(f"{env_prefix}_IDLE_CHECK_INTERVAL", "60")),
            idle_threshold_seconds=int(os.environ.get(
                f"{env_prefix}_IDLE_THRESHOLD",
                str(defaults.get("idle_threshold_seconds", 1800))
            )),
            idle_utilization_threshold=float(os.environ.get(
                f"{env_prefix}_IDLE_UTIL_THRESHOLD", "10.0"
            )),
            min_nodes_to_retain=int(os.environ.get(
                f"{env_prefix}_MIN_NODES",
                str(defaults.get("min_nodes_to_retain", 1))
            )),
            drain_period_seconds=int(os.environ.get(
                f"{env_prefix}_DRAIN_PERIOD",
                str(defaults.get("drain_period_seconds", 300))
            )),
            min_cost_savings_per_hour=float(os.environ.get(
                f"{env_prefix}_MIN_SAVINGS",
                str(defaults.get("min_cost_savings_per_hour", 1.0))
            )),
            check_pending_work=os.environ.get(
                f"{env_prefix}_CHECK_WORK", "true"
            ).lower() == "true",
            pending_work_threshold=int(os.environ.get(
                f"{env_prefix}_WORK_THRESHOLD",
                str(defaults.get("pending_work_threshold", 5))
            )),
            dry_run=os.environ.get(f"{env_prefix}_IDLE_DRY_RUN", "false").lower() == "true",
            instance_label_filter=os.environ.get(f"{env_prefix}_LABEL_FILTER", ""),
            prioritize_expensive=os.environ.get(
                f"{env_prefix}_PRIORITIZE_EXPENSIVE", "true"
            ).lower() == "true",
            provider_name=provider_name,
        )


@dataclass
class NodeIdleStatus:
    """Status of a node for idle tracking."""
    instance_id: str
    instance_name: str
    provider: str
    ip_address: str
    ssh_port: int = 22
    gpu_name: str = ""
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


class UnifiedIdleShutdownDaemon(HandlerBase):
    """Provider-agnostic daemon that monitors nodes and terminates idle ones.

    January 2026: Migrated to HandlerBase for unified lifecycle management.
    """

    def __init__(
        self,
        provider: CloudProvider,
        config: IdleShutdownConfig | None = None,
    ):
        """Initialize the daemon.

        Args:
            provider: Cloud provider instance (e.g., VastProvider, LambdaProvider)
            config: Configuration for idle detection. Uses provider defaults if not provided.
        """
        self._provider = provider
        provider_name = provider.name if hasattr(provider, 'name') else str(provider.provider_type.name).lower()
        self._daemon_config = config or IdleShutdownConfig.for_provider(provider_name)
        self._daemon_config.provider_name = provider_name
        self._daemon_name = f"{provider_name}_idle_daemon"

        # Initialize HandlerBase with cycle_interval from config
        super().__init__(
            name=self._daemon_name,
            config={"provider": provider_name},
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )

        self._node_status: dict[str, NodeIdleStatus] = {}
        self._draining_nodes: dict[str, float] = {}  # instance_id -> drain_started
        self._terminated_count = 0
        self._cost_saved = 0.0
        self._last_check_time = 0.0

    @property
    def config(self) -> IdleShutdownConfig:
        """Backward-compatible config access."""
        return self._daemon_config

    @config.setter
    def config(self, value: IdleShutdownConfig) -> None:
        """Backward-compatible config setter."""
        self._daemon_config = value

    def _get_event_subscriptions(self) -> dict[str, Callable]:
        """Return event subscriptions for HandlerBase.

        January 2026: Migrated from _subscribe_to_events().
        """
        return {
            "training_started": self._on_training_started,
            "selfplay_complete": self._on_selfplay_completed,
            "work_started": self._on_work_started,
            "work_claimed": self._on_work_claimed,
            "task_spawned": self._on_task_spawned,
        }

    async def _on_start(self) -> None:
        """Initialize daemon on startup (HandlerBase lifecycle hook)."""
        if not self._daemon_config.enabled:
            logger.info(f"[{self._daemon_name}] Disabled by configuration")
            self._running = False
            return

        if not self._provider.is_configured():
            logger.warning(f"[{self._daemon_name}] Provider not configured, daemon disabled")
            self._running = False
            return

        # Register as coordinator
        if HAS_PROTOCOLS:
            register_coordinator(
                self._daemon_name,
                CoordinatorStatus.ACTIVE,
                metadata={"config": self._daemon_config.__dict__},
            )

        logger.info(
            f"[{self._daemon_name}] Started with idle_threshold={self._daemon_config.idle_threshold_seconds}s, "
            f"min_nodes={self._daemon_config.min_nodes_to_retain}, dry_run={self._daemon_config.dry_run}"
        )


    def _cancel_drain_for_host(self, hostname: str, reason: str) -> bool:
        """Cancel drain period for a node if it's currently draining.

        Dec 2025: When activity is detected on a draining node, cancel the drain
        to prevent premature termination.

        Args:
            hostname: Host name or IP that had activity
            reason: Reason for cancelling drain (for logging)

        Returns:
            True if drain was cancelled, False if node wasn't draining
        """
        # Find matching node in draining_nodes by checking instance name
        for instance_id, drain_started in list(self._draining_nodes.items()):
            # Check if this instance matches the hostname
            # Node status includes instance_name which should match hostname
            status = self._node_status.get(instance_id)
            if status and status.instance_name in hostname or hostname in (status.instance_name or ""):
                del self._draining_nodes[instance_id]
                drain_duration = time.time() - drain_started
                logger.info(
                    f"[{self._daemon_name}] Cancelled drain for {status.instance_name} "
                    f"after {drain_duration:.0f}s: {reason}"
                )
                return True
        return False

    async def _on_training_started(self, event) -> None:
        """Handle training started - cancel drain and don't terminate training nodes."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            host = payload.get("host", "unknown")
            logger.debug(f"[{self._daemon_name}] Training started on {host}")

            # Dec 2025: Cancel drain if node was draining
            self._cancel_drain_for_host(host, "training_started")
        except Exception as e:
            logger.debug(f"[{self._daemon_name}] Error handling training event: {e}")

    async def _on_selfplay_completed(self, event) -> None:
        """Handle selfplay completed - refresh utilization data."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            host = payload.get("host", payload.get("node", ""))
            logger.debug(f"[{self._daemon_name}] Selfplay completed on {host}")

            # Dec 2025: Cancel drain if node was draining (selfplay completed means it was active)
            if host:
                self._cancel_drain_for_host(host, "selfplay_completed")
        except Exception as e:
            logger.debug(f"[{self._daemon_name}] Error handling selfplay event: {e}")

    async def _on_work_started(self, event) -> None:
        """Handle work started - cancel drain for active worker.

        Dec 2025: When work starts on a node, cancel any pending drain.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            host = payload.get("host", payload.get("worker_id", payload.get("node", "")))
            if host:
                self._cancel_drain_for_host(host, "work_started")
        except Exception as e:
            logger.debug(f"[{self._daemon_name}] Error handling work_started: {e}")

    async def _on_work_claimed(self, event) -> None:
        """Handle work claimed - cancel drain for worker claiming work.

        Dec 2025: When a node claims work from the queue, cancel any pending drain.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            host = payload.get("worker_id", payload.get("node", payload.get("host", "")))
            if host:
                self._cancel_drain_for_host(host, "work_claimed")
        except Exception as e:
            logger.debug(f"[{self._daemon_name}] Error handling work_claimed: {e}")

    async def _on_task_spawned(self, event) -> None:
        """Handle task spawned - cancel drain for node running new task.

        Dec 2025: When a task is spawned on a node, cancel any pending drain.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            host = payload.get("node", payload.get("host", payload.get("worker_id", "")))
            if host:
                self._cancel_drain_for_host(host, "task_spawned")
        except Exception as e:
            logger.debug(f"[{self._daemon_name}] Error handling task_spawned: {e}")

    async def _on_stop(self) -> None:
        """Cleanup on shutdown (HandlerBase lifecycle hook)."""
        if HAS_PROTOCOLS:
            unregister_coordinator(self._daemon_name)
        logger.info(f"[{self._daemon_name}] Stopped")

    async def _run_cycle(self) -> None:
        """Main daemon cycle (HandlerBase lifecycle hook).

        January 2026: Migrated from _run_loop().
        """
        if not self._daemon_config.enabled:
            return
        if not self._provider.is_configured():
            return

        try:
            await self._check_and_terminate_idle_nodes()
            self._last_check_time = time.time()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[{self._daemon_name}] Error in check cycle: {e}", exc_info=True)
            self._record_error(f"Cycle error: {e}", e)

    async def _get_gpu_utilization(self, instance: Instance) -> tuple[float, float, float]:
        """Get GPU utilization for an instance.

        Tries P2P status first, falls back to SSH nvidia-smi.

        Returns:
            Tuple of (utilization_percent, memory_used_gb, memory_total_gb)
        """
        # Try P2P status first (faster, no SSH overhead)
        if HAS_P2P and instance.ip_address:
            try:
                status = await get_p2p_status(instance.ip_address, timeout=5.0)
                if status and "gpu_utilization" in status:
                    mem_used = status.get("gpu_memory_used_gb", 0.0)
                    mem_total = status.get("gpu_memory_total_gb", 0.0)
                    return float(status["gpu_utilization"]), mem_used, mem_total

                # Fall back to checking if any selfplay processes running
                if status and "active_jobs" in status:
                    return (100.0, 0.0, 0.0) if status["active_jobs"] > 0 else (0.0, 0.0, 0.0)
            except asyncio.TimeoutError:
                pass  # P2P timed out, fall through to SSH
            except (OSError, IOError):
                pass  # Network error, fall through to SSH
            except (ValueError, KeyError, TypeError):
                pass  # Malformed response, fall through to SSH

        # Try SSH nvidia-smi
        if instance.ip_address:
            try:
                from app.core.ssh import run_ssh_command_async, SSHConfig

                ssh_config = SSHConfig(
                    host=instance.ip_address,
                    port=instance.ssh_port,
                    user=instance.ssh_user,
                )

                result = await run_ssh_command_async(
                    ssh_config,
                    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
                    "--format=csv,noheader,nounits",
                    timeout=10,
                )

                if result.returncode == 0 and result.stdout:
                    lines = result.stdout.strip().split("\n")
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
                logger.debug(f"[{self._daemon_name}] Failed to get utilization for {instance.name}: {e}")

        return 0.0, 0.0, 0.0  # Return 0 on error (will be marked idle)

    async def _update_node_status(self, instance: Instance, now: float) -> None:
        """Update status tracking for an instance."""
        instance_id = instance.id

        # Get current utilization
        util, mem_used, mem_total = await self._get_gpu_utilization(instance)

        # Get or create status entry
        if instance_id not in self._node_status:
            self._node_status[instance_id] = NodeIdleStatus(
                instance_id=instance_id,
                instance_name=instance.name,
                provider=self.config.provider_name,
                ip_address=instance.ip_address or "",
                ssh_port=instance.ssh_port,
                gpu_name=instance.raw_data.get("gpu_name", str(instance.gpu_type.value)),
                cost_per_hour=instance.cost_per_hour,
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
                logger.debug(f"[{self._daemon_name}] {instance.name} became idle (util={util:.1f}%)")
        else:
            if status.idle_since > 0:
                logger.debug(f"[{self._daemon_name}] {instance.name} became active (util={util:.1f}%)")
            status.idle_since = 0
            status.last_activity_time = now

    async def _should_allow_termination(self) -> bool:
        """Check if termination should be allowed based on pending work."""
        if not self.config.check_pending_work:
            return True

        if not HAS_P2P:
            return True

        try:
            status = await get_p2p_status(timeout=5.0)
            queue_depth = status.get("work_queue_size", status.get("queue_depth", 0))

            if queue_depth > self.config.pending_work_threshold:
                logger.debug(
                    f"[{self._daemon_name}] Queue depth {queue_depth} exceeds threshold "
                    f"{self.config.pending_work_threshold}"
                )
                return False
            return True
        except Exception as e:
            logger.debug(f"[{self._daemon_name}] Failed to check P2P status: {e}")
            return True  # Allow termination if we can't check

    async def _terminate_node(self, node: NodeIdleStatus) -> bool:
        """Terminate an idle node with drain period."""
        if self.config.dry_run:
            logger.info(
                f"[{self._daemon_name}] DRY RUN: Would terminate {node.instance_name} "
                f"(idle {node.idle_duration_seconds:.0f}s, ${node.cost_per_hour:.2f}/hr)"
            )
            return True

        # Check if in drain period
        if node.instance_id in self._draining_nodes:
            drain_started = self._draining_nodes[node.instance_id]
            if time.time() - drain_started < self.config.drain_period_seconds:
                remaining = self.config.drain_period_seconds - (time.time() - drain_started)
                logger.debug(f"[{self._daemon_name}] {node.instance_name} draining ({remaining:.0f}s remaining)")
                return False
            # Drain complete, proceed with termination
        else:
            # Start drain period
            self._draining_nodes[node.instance_id] = time.time()
            logger.info(
                f"[{self._daemon_name}] Starting drain period for {node.instance_name} "
                f"({self.config.drain_period_seconds}s)"
            )
            return False

        try:
            results = await self._provider.scale_down([node.instance_id])
            success = results.get(node.instance_id, False)

            if success:
                logger.info(
                    f"[{self._daemon_name}] Terminated {node.instance_name} "
                    f"(idle {node.idle_duration_seconds:.0f}s, saving ${node.cost_per_hour:.2f}/hr)"
                )
                # Clean up
                self._draining_nodes.pop(node.instance_id, None)
                self._node_status.pop(node.instance_id, None)
            else:
                logger.warning(f"[{self._daemon_name}] Failed to terminate {node.instance_name}")

            return success
        except Exception as e:
            logger.error(f"[{self._daemon_name}] Error terminating {node.instance_name}: {e}")
            return False

    async def _check_and_terminate_idle_nodes(self) -> None:
        """Check all nodes and terminate idle ones."""
        # Get current instances
        try:
            instances = await self._provider.list_instances()
        except Exception as e:
            logger.error(f"[{self._daemon_name}] Failed to list instances: {e}")
            return

        if not instances:
            logger.debug(f"[{self._daemon_name}] No instances found")
            return

        # Filter by label if configured
        if self.config.instance_label_filter:
            label = self.config.instance_label_filter
            instances = [
                i for i in instances
                if label in (i.tags.get("label", "") or i.name or "")
            ]

        # Filter to running instances
        running_instances = [i for i in instances if i.status == InstanceStatus.RUNNING]

        logger.debug(f"[{self._daemon_name}] Found {len(running_instances)} running instances")

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

        logger.info(f"[{self._daemon_name}] Found {len(idle_nodes)} idle nodes exceeding threshold")

        # Check pending work if configured
        if not await self._should_allow_termination():
            logger.info(f"[{self._daemon_name}] Skipping termination due to pending work")
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
                f"[{self._daemon_name}] Cannot terminate - would go below minimum "
                f"({len(running_instances)} running, {self.config.min_nodes_to_retain} minimum)"
            )
            return

        # Terminate eligible nodes
        terminated = 0
        for node in idle_nodes[:can_terminate]:
            if node.cost_per_hour < self.config.min_cost_savings_per_hour:
                logger.debug(
                    f"[{self._daemon_name}] Skipping {node.instance_name} - "
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
                                "provider": self.config.provider_name,
                                "reason": "idle",
                                "idle_duration_seconds": node.idle_duration_seconds,
                                "cost_per_hour": node.cost_per_hour,
                                "gpu_name": node.gpu_name,
                            },
                            source=self._daemon_name,
                        )
                    except Exception as e:
                        logger.debug(f"[{self._daemon_name}] Failed to emit event: {e}")

        if terminated > 0:
            logger.info(
                f"[{self._daemon_name}] Terminated {terminated} idle nodes. "
                f"Total: {self._terminated_count} terminated, ${self._cost_saved:.2f}/hr saved"
            )

    def health_check(self) -> HealthCheckResult:
        """Check daemon health.

        January 2026: Updated for HandlerBase migration.

        Returns:
            HealthCheckResult with status and details
        """
        if not self._running:
            return HealthCheckResult(
                healthy=True,
                message=f"{self._daemon_name} not running",
            )

        # Check if checks are running
        if self._last_check_time > 0:
            time_since_last = time.time() - self._last_check_time
            if time_since_last > self._daemon_config.check_interval_seconds * 3:
                return HealthCheckResult(
                    healthy=False,
                    message=f"{self._daemon_name} stale ({time_since_last:.0f}s since last check)",
                    details=self.get_stats(),
                )

        uptime = time.time() - self._stats.started_at if self._stats.started_at > 0 else 0
        return HealthCheckResult(
            healthy=True,
            message=f"{self._daemon_name} healthy ({self._terminated_count} terminated, ${self._cost_saved:.2f}/hr saved, uptime={uptime:.0f}s)",
            details=self.get_stats(),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get current daemon statistics."""
        uptime = time.time() - self._stats.started_at if self._stats.started_at > 0 else 0
        return {
            "name": self._daemon_name,
            "running": self._running,
            "uptime_seconds": uptime,
            "provider": self._daemon_config.provider_name,
            "terminated_count": self._terminated_count,
            "cost_saved_per_hour": self._cost_saved,
            "tracked_nodes": len(self._node_status),
            "draining_nodes": len(self._draining_nodes),
            "last_check_time": self._last_check_time,
            "cycles_completed": self._stats.cycles_completed,
            "errors_count": self._stats.errors_count,
            "config": {
                "enabled": self._daemon_config.enabled,
                "idle_threshold_seconds": self._daemon_config.idle_threshold_seconds,
                "min_nodes_to_retain": self._daemon_config.min_nodes_to_retain,
                "drain_period_seconds": self._daemon_config.drain_period_seconds,
                "dry_run": self._daemon_config.dry_run,
            },
        }


# Factory functions for backward compatibility
def create_lambda_idle_daemon() -> UnifiedIdleShutdownDaemon | None:
    """Create a Lambda idle shutdown daemon with Lambda-specific defaults.

    NOTE: Lambda Labs account was terminated December 2025. This function
    now returns None with a warning. Retained for backward compatibility.
    """
    import warnings
    warnings.warn(
        "Lambda Labs account terminated December 2025. "
        "create_lambda_idle_daemon() returns None. Use create_vast_idle_daemon() "
        "or create_runpod_idle_daemon() for active providers.",
        DeprecationWarning,
        stacklevel=2,
    )
    return None


def create_vast_idle_daemon() -> UnifiedIdleShutdownDaemon:
    """Create a Vast.ai idle shutdown daemon with Vast-specific defaults."""
    try:
        from app.coordination.providers.vast_provider import VastProvider
        return UnifiedIdleShutdownDaemon(
            provider=VastProvider(),
            config=IdleShutdownConfig.for_provider("vast"),
        )
    except ImportError:
        raise ImportError("Vast provider not available")


def create_runpod_idle_daemon() -> UnifiedIdleShutdownDaemon:
    """Create a RunPod idle shutdown daemon with RunPod-specific defaults."""
    try:
        from app.coordination.providers.runpod_provider import RunPodProvider
        return UnifiedIdleShutdownDaemon(
            provider=RunPodProvider(),
            config=IdleShutdownConfig.for_provider("runpod"),
        )
    except ImportError:
        raise ImportError("RunPod provider not available")


# =============================================================================
# Module-level singleton accessors (January 2026)
# =============================================================================

# Per-provider singleton instances
_daemon_instances: dict[str, UnifiedIdleShutdownDaemon] = {}


def get_idle_shutdown_daemon(
    provider: CloudProvider,
    config: IdleShutdownConfig | None = None,
) -> UnifiedIdleShutdownDaemon:
    """Get or create the singleton idle shutdown daemon for a provider.

    Args:
        provider: Cloud provider instance
        config: Optional configuration

    Returns:
        UnifiedIdleShutdownDaemon singleton for the given provider
    """
    provider_name = provider.name if hasattr(provider, 'name') else str(provider.provider_type.name).lower()

    if provider_name not in _daemon_instances:
        _daemon_instances[provider_name] = UnifiedIdleShutdownDaemon(
            provider=provider,
            config=config,
        )
    return _daemon_instances[provider_name]


def reset_idle_shutdown_daemon(provider_name: str | None = None) -> None:
    """Reset the singleton daemon instance(s).

    Args:
        provider_name: If provided, only reset the daemon for this provider.
                       If None, reset all daemon instances.
    """
    global _daemon_instances
    if provider_name is not None:
        _daemon_instances.pop(provider_name, None)
    else:
        _daemon_instances.clear()
