"""Scale adapters for cloud provider auto-scaling.

December 2025 (Wave 7 Phase 2.1): Provides a unified interface for scaling
GPU compute nodes across multiple cloud providers (Lambda, Vast.ai, RunPod, etc.).

Design Principles:
1. RELUCTANT TERMINATION: Default behavior is to utilize all available nodes.
   Nodes are only terminated after confirmed unusability via repeated failures.
2. CONSERVATIVE SCALING: Scale-up is aggressive (respond to workload quickly),
   but scale-down requires extended idle periods and confirmation.
3. MULTI-PROVIDER: Support heterogeneous clusters with different providers.

Usage:
    from scripts.p2p.adapters.scale_adapters import (
        CompositeScaleAdapter,
        AutoScalingConfig,
    )

    adapter = CompositeScaleAdapter(
        work_queue=my_work_queue,
        peers_getter=lambda: orchestrator.peers,
        lambda_api_key=os.environ.get("LAMBDA_API_KEY"),
    )

    # Use in AutoScalingLoop
    auto_scaling = AutoScalingLoop(
        get_pending_work=adapter.get_pending_work,
        get_active_nodes=adapter.get_active_nodes,
        get_idle_nodes=adapter.get_idle_nodes,
        scale_up=adapter.scale_up,
        scale_down=adapter.scale_down,
        config=AutoScalingConfig.conservative(),  # Reluctant termination
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class AutoScalingConfig:
    """Configuration for auto-scaling behavior.

    Default values are CONSERVATIVE to prevent accidental termination.
    """

    # Minimum nodes to maintain (never scale below this)
    min_nodes: int = 2

    # Maximum nodes to scale to (cost protection)
    max_nodes: int = 50

    # Work queue threshold to trigger scale-up (pending items)
    scale_up_threshold: int = 100

    # Idle time required before considering scale-down (seconds)
    # Default: 30 minutes - gives ample time for work to arrive
    scale_down_idle_threshold: float = 1800.0

    # Number of consecutive health check failures required before termination
    # Default: 5 - very reluctant to terminate
    termination_failure_threshold: int = 5

    # Minimum time a node must be idle before termination consideration (seconds)
    # Even with failures, don't terminate recently-used nodes
    min_idle_before_termination: float = 3600.0  # 1 hour

    # Cooldown between scale operations (seconds)
    scale_up_cooldown: float = 60.0
    scale_down_cooldown: float = 300.0  # 5 minutes between terminations

    # Maximum nodes to terminate in a single cycle
    max_terminations_per_cycle: int = 1  # Very conservative

    # Dry run mode - log actions without executing
    dry_run: bool = False

    @classmethod
    def conservative(cls) -> "AutoScalingConfig":
        """Create a conservative config - very reluctant to terminate."""
        return cls(
            min_nodes=2,
            max_nodes=50,
            scale_up_threshold=100,
            scale_down_idle_threshold=3600.0,  # 1 hour idle
            termination_failure_threshold=10,  # 10 consecutive failures
            min_idle_before_termination=7200.0,  # 2 hours
            scale_down_cooldown=600.0,  # 10 minutes between terminations
            max_terminations_per_cycle=1,
        )

    @classmethod
    def aggressive(cls) -> "AutoScalingConfig":
        """Create an aggressive config - faster scale-down for cost savings."""
        return cls(
            min_nodes=1,
            max_nodes=100,
            scale_up_threshold=50,
            scale_down_idle_threshold=600.0,  # 10 minutes
            termination_failure_threshold=3,
            min_idle_before_termination=1200.0,  # 20 minutes
            scale_down_cooldown=120.0,
            max_terminations_per_cycle=3,
        )

    @classmethod
    def from_env(cls) -> "AutoScalingConfig":
        """Create config from environment variables."""
        return cls(
            min_nodes=int(os.environ.get("RINGRIFT_MIN_NODES", "2")),
            max_nodes=int(os.environ.get("RINGRIFT_MAX_NODES", "50")),
            scale_up_threshold=int(os.environ.get("RINGRIFT_SCALE_UP_THRESHOLD", "100")),
            scale_down_idle_threshold=float(
                os.environ.get("RINGRIFT_SCALE_DOWN_IDLE_THRESHOLD", "1800")
            ),
            termination_failure_threshold=int(
                os.environ.get("RINGRIFT_TERMINATION_FAILURE_THRESHOLD", "5")
            ),
            min_idle_before_termination=float(
                os.environ.get("RINGRIFT_MIN_IDLE_BEFORE_TERMINATION", "3600")
            ),
            dry_run=os.environ.get("RINGRIFT_AUTOSCALE_DRY_RUN", "").lower() == "true",
        )


@dataclass
class NodeIdleStatus:
    """Track idle status for a node."""

    node_id: str
    last_activity_time: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    gpu_utilization: float = 0.0
    active_jobs: int = 0
    is_marked_for_termination: bool = False
    termination_check_count: int = 0

    @property
    def idle_duration(self) -> float:
        """Seconds since last activity."""
        return time.time() - self.last_activity_time

    def is_truly_idle(self, threshold: float) -> bool:
        """Check if node is truly idle (no jobs, low GPU, past threshold)."""
        return (
            self.active_jobs == 0
            and self.gpu_utilization < 5.0
            and self.idle_duration > threshold
        )

    def record_activity(self) -> None:
        """Record that this node had activity."""
        self.last_activity_time = time.time()
        self.consecutive_failures = 0
        self.is_marked_for_termination = False
        self.termination_check_count = 0

    def record_failure(self) -> None:
        """Record a health check failure."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()


# ============================================================================
# Protocol Definitions
# ============================================================================


@runtime_checkable
class ScaleAdapterProtocol(Protocol):
    """Protocol for cloud provider scaling adapters."""

    def get_pending_work(self) -> int:
        """Get count of pending work items across all queues."""
        ...

    def get_active_nodes(self) -> int:
        """Get count of currently active GPU nodes."""
        ...

    def get_idle_nodes(self) -> list[str]:
        """Get list of node IDs that are idle and can be terminated.

        IMPORTANT: Should only return nodes that meet ALL criteria:
        1. No active jobs
        2. Low GPU utilization (<5%)
        3. Idle for longer than threshold
        4. Multiple consecutive health check failures (reluctant termination)
        """
        ...

    async def scale_up(self, count: int) -> list[str]:
        """Provision new nodes. Returns list of new node IDs."""
        ...

    async def scale_down(self, node_ids: list[str]) -> int:
        """Terminate specified nodes. Returns count terminated."""
        ...


@runtime_checkable
class CloudProviderProtocol(Protocol):
    """Protocol for individual cloud provider implementations."""

    @property
    def provider_name(self) -> str:
        """Name of this provider (e.g., 'lambda', 'vast')."""
        ...

    @property
    def hourly_cost(self) -> float:
        """Estimated hourly cost per node in USD."""
        ...

    async def list_instances(self) -> list[dict[str, Any]]:
        """List all instances for this provider."""
        ...

    async def launch_instance(self, instance_type: str | None = None) -> str:
        """Launch a new instance. Returns instance ID."""
        ...

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance. Returns True if successful."""
        ...

    def get_instance_status(self, instance_id: str) -> str:
        """Get instance status (running, stopped, terminated, etc.)."""
        ...


# ============================================================================
# Stub Provider Implementations
# ============================================================================


class LambdaScaleAdapter:
    """Scale adapter for Lambda Labs instances.

    Note: Requires LAMBDA_API_KEY environment variable.
    """

    provider_name = "lambda"
    hourly_cost = 1.49  # GH200 pricing

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("LAMBDA_API_KEY")
        if not self.api_key:
            logger.warning("LambdaScaleAdapter: No API key provided")

    async def list_instances(self) -> list[dict[str, Any]]:
        """List Lambda instances via API."""
        if not self.api_key:
            return []
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                async with session.get(
                    "https://cloud.lambdalabs.com/api/v1/instances",
                    headers=headers,
                    timeout=30,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("data", [])
                    logger.warning(f"Lambda API error: {resp.status}")
                    return []
        except Exception as e:  # noqa: BLE001
            logger.error(f"Lambda list_instances failed: {e}")
            return []

    async def launch_instance(self, instance_type: str | None = None) -> str:
        """Launch a new Lambda instance."""
        if not self.api_key:
            logger.error("Cannot launch Lambda instance: no API key")
            return ""

        instance_type = instance_type or "gpu_1x_gh200"
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                payload = {
                    "region_name": "us-west-1",
                    "instance_type_name": instance_type,
                    "ssh_key_names": [],  # Would need to be configured
                    "quantity": 1,
                }
                async with session.post(
                    "https://cloud.lambdalabs.com/api/v1/instance-operations/launch",
                    headers=headers,
                    json=payload,
                    timeout=60,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        instance_ids = data.get("data", {}).get("instance_ids", [])
                        return instance_ids[0] if instance_ids else ""
                    logger.error(f"Lambda launch failed: {resp.status} - {await resp.text()}")
                    return ""
        except Exception as e:  # noqa: BLE001
            logger.error(f"Lambda launch_instance failed: {e}")
            return ""

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a Lambda instance."""
        if not self.api_key:
            return False

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                payload = {"instance_ids": [instance_id]}
                async with session.post(
                    "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate",
                    headers=headers,
                    json=payload,
                    timeout=60,
                ) as resp:
                    return resp.status == 200
        except Exception as e:  # noqa: BLE001
            logger.error(f"Lambda terminate_instance failed: {e}")
            return False

    def get_instance_status(self, instance_id: str) -> str:
        """Get instance status (sync version for quick checks)."""
        # Would need async implementation in real usage
        return "unknown"


class VastAIScaleAdapter:
    """Scale adapter for Vast.ai instances.

    Note: Requires VAST_API_KEY environment variable.
    """

    provider_name = "vast"
    hourly_cost = 0.50  # Varies by instance type

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("VAST_API_KEY")
        if not self.api_key:
            logger.warning("VastAIScaleAdapter: No API key provided")

    async def list_instances(self) -> list[dict[str, Any]]:
        """List Vast.ai instances."""
        # Placeholder - would use Vast.ai API
        return []

    async def launch_instance(self, instance_type: str | None = None) -> str:
        """Launch a Vast.ai instance."""
        logger.warning("VastAIScaleAdapter.launch_instance: Not implemented")
        return ""

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a Vast.ai instance."""
        logger.warning("VastAIScaleAdapter.terminate_instance: Not implemented")
        return False

    def get_instance_status(self, instance_id: str) -> str:
        """Get instance status."""
        return "unknown"


class RunPodScaleAdapter:
    """Scale adapter for RunPod instances.

    Note: Requires RUNPOD_API_KEY environment variable.
    """

    provider_name = "runpod"
    hourly_cost = 1.00  # Varies

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            logger.warning("RunPodScaleAdapter: No API key provided")

    async def list_instances(self) -> list[dict[str, Any]]:
        """List RunPod pods."""
        return []

    async def launch_instance(self, instance_type: str | None = None) -> str:
        """Launch a RunPod pod."""
        return ""

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate a RunPod pod."""
        return False

    def get_instance_status(self, instance_id: str) -> str:
        """Get pod status."""
        return "unknown"


# ============================================================================
# Composite Adapter
# ============================================================================


class CompositeScaleAdapter:
    """Adapter that combines multiple provider-specific adapters.

    Implements ScaleAdapterProtocol with RELUCTANT TERMINATION behavior.
    """

    def __init__(
        self,
        work_queue: Any = None,
        peers_getter: Callable[[], dict[str, Any]] | None = None,
        lambda_api_key: str | None = None,
        vast_api_key: str | None = None,
        runpod_api_key: str | None = None,
        config: AutoScalingConfig | None = None,
    ):
        """Initialize composite adapter.

        Args:
            work_queue: Work queue instance for pending work counts
            peers_getter: Callable that returns dict of node_id -> peer info
            lambda_api_key: Optional Lambda Labs API key
            vast_api_key: Optional Vast.ai API key
            runpod_api_key: Optional RunPod API key
            config: Auto-scaling configuration (default: conservative)
        """
        self.work_queue = work_queue
        self.peers_getter = peers_getter or (lambda: {})
        self.config = config or AutoScalingConfig.conservative()

        # Initialize providers
        self.providers: dict[str, CloudProviderProtocol] = {}
        if lambda_api_key:
            self.providers["lambda"] = LambdaScaleAdapter(lambda_api_key)
        if vast_api_key:
            self.providers["vast"] = VastAIScaleAdapter(vast_api_key)
        if runpod_api_key:
            self.providers["runpod"] = RunPodScaleAdapter(runpod_api_key)

        # Track node idle status for reluctant termination
        self._node_idle_status: dict[str, NodeIdleStatus] = {}

        # Track last scale operations for cooldown
        self._last_scale_up: float = 0.0
        self._last_scale_down: float = 0.0

        logger.info(
            f"CompositeScaleAdapter initialized with providers: {list(self.providers.keys())}"
        )

    def get_pending_work(self) -> int:
        """Get count of pending work items across all queues."""
        if self.work_queue is None:
            return 0

        try:
            # Try different methods to get pending count
            if hasattr(self.work_queue, "get_pending_count"):
                return self.work_queue.get_pending_count()
            if hasattr(self.work_queue, "get_pending_items"):
                return len(self.work_queue.get_pending_items())
            if hasattr(self.work_queue, "pending_count"):
                return self.work_queue.pending_count
            return 0
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Error getting pending work: {e}")
            return 0

    def get_active_nodes(self) -> int:
        """Get count of currently active GPU nodes."""
        peers = self.peers_getter()
        return len([p for p in peers.values() if p.get("alive", False)])

    def get_idle_nodes(self) -> list[str]:
        """Get list of node IDs that are confirmed idle and safe to terminate.

        RELUCTANT TERMINATION: Only returns nodes that meet ALL criteria:
        1. No active jobs for extended period
        2. Low GPU utilization (<5%)
        3. Idle for longer than scale_down_idle_threshold
        4. Multiple consecutive health check failures (termination_failure_threshold)
        5. Not recently used (min_idle_before_termination)
        """
        peers = self.peers_getter()
        now = time.time()
        idle_candidates: list[str] = []

        for node_id, peer in peers.items():
            if not peer.get("alive", False):
                continue

            # Get or create idle status tracker
            if node_id not in self._node_idle_status:
                self._node_idle_status[node_id] = NodeIdleStatus(node_id=node_id)

            status = self._node_idle_status[node_id]

            # Update status from peer info
            gpu_util = peer.get("gpu_percent", 0.0) or 0.0
            active_jobs = peer.get("selfplay_jobs", 0) + peer.get("training_jobs", 0)

            status.gpu_utilization = gpu_util
            status.active_jobs = active_jobs

            # If node has activity, reset idle tracking
            if active_jobs > 0 or gpu_util > 5.0:
                status.record_activity()
                continue

            # Check if node meets ALL idle criteria (RELUCTANT TERMINATION)
            is_idle_enough = status.idle_duration > self.config.scale_down_idle_threshold
            is_confirmed_idle = status.idle_duration > self.config.min_idle_before_termination
            has_enough_failures = status.consecutive_failures >= self.config.termination_failure_threshold

            # Increment termination check count
            status.termination_check_count += 1

            # Only consider for termination if ALL criteria are met
            if is_idle_enough and is_confirmed_idle and has_enough_failures:
                # Mark for termination and add to list
                status.is_marked_for_termination = True
                idle_candidates.append(node_id)
                logger.info(
                    f"Node {node_id} marked for termination: "
                    f"idle={status.idle_duration:.0f}s, failures={status.consecutive_failures}, "
                    f"checks={status.termination_check_count}"
                )
            elif is_idle_enough:
                # Node is idle but not confirmed - record a failure for tracking
                if status.termination_check_count % 5 == 0:  # Only record every 5th check
                    status.record_failure()
                    logger.debug(
                        f"Node {node_id} idle check: failures={status.consecutive_failures}/{self.config.termination_failure_threshold}"
                    )

        # Respect max_terminations_per_cycle
        return idle_candidates[: self.config.max_terminations_per_cycle]

    async def scale_up(self, count: int) -> list[str]:
        """Provision new nodes from available providers.

        Uses cheapest providers first. Respects cooldown period.
        """
        now = time.time()

        # Check cooldown
        if now - self._last_scale_up < self.config.scale_up_cooldown:
            remaining = self.config.scale_up_cooldown - (now - self._last_scale_up)
            logger.debug(f"Scale-up in cooldown, {remaining:.0f}s remaining")
            return []

        # Check max nodes
        current_nodes = self.get_active_nodes()
        if current_nodes >= self.config.max_nodes:
            logger.warning(f"At max nodes ({current_nodes}/{self.config.max_nodes}), cannot scale up")
            return []

        # Limit count to stay within max
        count = min(count, self.config.max_nodes - current_nodes)

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would scale up {count} nodes")
            return []

        # Sort providers by cost (cheapest first)
        sorted_providers = sorted(
            self.providers.values(),
            key=lambda p: p.hourly_cost,
        )

        new_nodes: list[str] = []
        for provider in sorted_providers:
            if len(new_nodes) >= count:
                break

            needed = count - len(new_nodes)
            for _ in range(needed):
                try:
                    instance_id = await provider.launch_instance()
                    if instance_id:
                        new_nodes.append(instance_id)
                        logger.info(f"Launched {provider.provider_name} instance: {instance_id}")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to launch {provider.provider_name} instance: {e}")

        if new_nodes:
            self._last_scale_up = now
            logger.info(f"Scale-up complete: launched {len(new_nodes)} nodes")

        return new_nodes

    async def scale_down(self, node_ids: list[str]) -> int:
        """Terminate specified nodes.

        RELUCTANT TERMINATION: Only terminates after extensive verification.
        """
        now = time.time()

        # Check cooldown
        if now - self._last_scale_down < self.config.scale_down_cooldown:
            remaining = self.config.scale_down_cooldown - (now - self._last_scale_down)
            logger.debug(f"Scale-down in cooldown, {remaining:.0f}s remaining")
            return 0

        # Check min nodes
        current_nodes = self.get_active_nodes()
        if current_nodes <= self.config.min_nodes:
            logger.info(f"At min nodes ({current_nodes}/{self.config.min_nodes}), cannot scale down")
            return 0

        # Limit terminations
        node_ids = node_ids[: self.config.max_terminations_per_cycle]

        # Don't terminate more than would take us below min_nodes
        max_terminable = current_nodes - self.config.min_nodes
        node_ids = node_ids[:max_terminable]

        if not node_ids:
            return 0

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would terminate nodes: {node_ids}")
            return 0

        terminated = 0
        for node_id in node_ids:
            # Final verification before termination
            status = self._node_idle_status.get(node_id)
            if status and not status.is_marked_for_termination:
                logger.warning(f"Node {node_id} not marked for termination, skipping")
                continue

            # Find provider for this node
            provider = self._get_provider_for_node(node_id)
            if provider:
                try:
                    success = await provider.terminate_instance(node_id)
                    if success:
                        terminated += 1
                        logger.info(f"Terminated {node_id}")
                        # Clean up tracking
                        self._node_idle_status.pop(node_id, None)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to terminate {node_id}: {e}")
            else:
                logger.warning(f"No provider found for node {node_id}")

        if terminated > 0:
            self._last_scale_down = now
            logger.info(f"Scale-down complete: terminated {terminated} nodes")

        return terminated

    def _get_provider_for_node(self, node_id: str) -> CloudProviderProtocol | None:
        """Determine which provider owns a node based on its ID prefix."""
        node_lower = node_id.lower()

        if "lambda" in node_lower or "gh200" in node_lower:
            return self.providers.get("lambda")
        if "vast" in node_lower:
            return self.providers.get("vast")
        if "runpod" in node_lower:
            return self.providers.get("runpod")

        # Try all providers
        for provider in self.providers.values():
            return provider  # Return first available

        return None

    def record_node_activity(self, node_id: str) -> None:
        """Record that a node had activity (used by job dispatchers)."""
        if node_id in self._node_idle_status:
            self._node_idle_status[node_id].record_activity()
        else:
            self._node_idle_status[node_id] = NodeIdleStatus(node_id=node_id)

    def record_node_failure(self, node_id: str) -> None:
        """Record a health check failure for a node."""
        if node_id not in self._node_idle_status:
            self._node_idle_status[node_id] = NodeIdleStatus(node_id=node_id)
        self._node_idle_status[node_id].record_failure()

    def get_node_idle_status(self, node_id: str) -> NodeIdleStatus | None:
        """Get idle status for a specific node."""
        return self._node_idle_status.get(node_id)

    def get_all_idle_statuses(self) -> dict[str, NodeIdleStatus]:
        """Get idle status for all tracked nodes."""
        return self._node_idle_status.copy()


# ============================================================================
# Factory Functions
# ============================================================================


def create_scale_adapter(
    work_queue: Any = None,
    peers_getter: Callable[[], dict[str, Any]] | None = None,
    config: AutoScalingConfig | None = None,
) -> CompositeScaleAdapter:
    """Create a scale adapter from environment variables.

    Automatically detects available providers based on API keys in environment.
    """
    return CompositeScaleAdapter(
        work_queue=work_queue,
        peers_getter=peers_getter,
        lambda_api_key=os.environ.get("LAMBDA_API_KEY"),
        vast_api_key=os.environ.get("VAST_API_KEY"),
        runpod_api_key=os.environ.get("RUNPOD_API_KEY"),
        config=config or AutoScalingConfig.from_env(),
    )


def get_default_config() -> AutoScalingConfig:
    """Get default conservative auto-scaling configuration."""
    return AutoScalingConfig.conservative()
