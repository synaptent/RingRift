"""Node Recovery Daemon (December 2025 - Phase 21).

Monitors cluster for terminated or failed nodes and automatically triggers
recovery actions.

Key features:
- Monitors P2P cluster for node failures
- Detects terminated Lambda instances via API
- Auto-restarts terminated instances (requires API credentials)
- Proactive recovery based on resource trends
- Integration with event system for cluster health events

Usage:
    from app.coordination.node_recovery_daemon import NodeRecoveryDaemon

    daemon = NodeRecoveryDaemon()
    await daemon.start()

Environment variables:
    LAMBDA_API_KEY: Lambda Cloud API key for instance management
    RINGRIFT_NODE_RECOVERY_ENABLED: Enable/disable recovery (default: 1)
    RINGRIFT_NODE_RECOVERY_INTERVAL: Check interval in seconds (default: 300)
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.coordination.protocols import (
    CoordinatorStatus,
    register_coordinator,
    unregister_coordinator,
)
from app.core.async_context import safe_create_task

logger = logging.getLogger(__name__)

# Health event emission imports (December 2025 - Phase 21)
try:
    from app.distributed.data_events import (
        emit_node_unhealthy,
        emit_node_recovered,
    )
    HAS_HEALTH_EVENTS = True
except ImportError:
    HAS_HEALTH_EVENTS = False
    emit_node_unhealthy = None
    emit_node_recovered = None


class RecoveryAction(Enum):
    """Types of recovery actions."""
    NONE = "none"
    RESTART = "restart"
    PREEMPTIVE_RESTART = "preemptive_restart"
    NOTIFY = "notify"  # Just notify, don't auto-recover
    FAILOVER = "failover"  # Migrate workload to another node


class NodeProvider(Enum):
    """Cloud provider types."""
    LAMBDA = "lambda"
    VAST = "vast"
    RUNPOD = "runpod"
    HETZNER = "hetzner"
    UNKNOWN = "unknown"


@dataclass
class NodeRecoveryConfig:
    """Configuration for node recovery."""
    enabled: bool = True
    check_interval_seconds: int = 300  # 5 minutes
    # Provider-specific settings
    lambda_api_key: str = ""
    vast_api_key: str = ""
    runpod_api_key: str = ""
    # Recovery thresholds
    max_consecutive_failures: int = 3
    recovery_cooldown_seconds: int = 600  # 10 minutes between recovery attempts
    # Proactive recovery settings
    memory_exhaustion_threshold: float = 0.02  # 2% per minute memory growth
    memory_exhaustion_window_minutes: int = 30
    preemptive_recovery_enabled: bool = True

    @classmethod
    def from_env(cls) -> NodeRecoveryConfig:
        """Load configuration from environment variables."""
        config = cls()
        config.enabled = os.environ.get("RINGRIFT_NODE_RECOVERY_ENABLED", "1") == "1"
        config.check_interval_seconds = int(
            os.environ.get("RINGRIFT_NODE_RECOVERY_INTERVAL", "300")
        )
        config.lambda_api_key = os.environ.get("LAMBDA_API_KEY", "")
        config.vast_api_key = os.environ.get("VAST_API_KEY", "")
        config.runpod_api_key = os.environ.get("RUNPOD_API_KEY", "")
        config.preemptive_recovery_enabled = (
            os.environ.get("RINGRIFT_PREEMPTIVE_RECOVERY", "1") == "1"
        )
        return config


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    host: str
    provider: NodeProvider = NodeProvider.UNKNOWN
    status: str = "unknown"  # running, terminated, failed, unreachable
    last_seen: float = 0.0
    consecutive_failures: int = 0
    last_recovery_attempt: float = 0.0
    instance_id: str = ""  # Provider-specific instance ID
    # Resource tracking for proactive recovery
    memory_samples: list[float] = field(default_factory=list)
    sample_timestamps: list[float] = field(default_factory=list)


@dataclass
class RecoveryStats:
    """Statistics for recovery operations."""
    total_checks: int = 0
    nodes_recovered: int = 0
    recovery_failures: int = 0
    preemptive_recoveries: int = 0
    last_check_time: float = 0.0
    last_error: str | None = None


class NodeRecoveryDaemon:
    """Daemon that monitors nodes and triggers recovery actions.

    Continuously monitors cluster for terminated or failing nodes
    and automatically triggers recovery when possible.
    """

    def __init__(self, config: NodeRecoveryConfig | None = None):
        self.config = config or NodeRecoveryConfig.from_env()
        self.node_id = socket.gethostname()
        self._running = False
        self._stats = RecoveryStats()
        self._monitor_task: asyncio.Task | None = None

        # Track node states
        self._node_states: dict[str, NodeInfo] = {}

        # CoordinatorProtocol state
        self._coordinator_status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._events_processed: int = 0
        self._errors_count: int = 0
        self._last_error: str = ""

        # HTTP session for API calls
        self._http_session = None

        logger.info(
            f"NodeRecoveryDaemon initialized: node={self.node_id}, "
            f"interval={self.config.check_interval_seconds}s, "
            f"lambda_api={'set' if self.config.lambda_api_key else 'not set'}"
        )

    def is_running(self) -> bool:
        """Check if the daemon is running."""
        return self._running

    async def start(self) -> None:
        """Start the node recovery daemon."""
        if not self.config.enabled:
            self._coordinator_status = CoordinatorStatus.STOPPED
            logger.info("NodeRecoveryDaemon disabled by config")
            return

        if self._coordinator_status == CoordinatorStatus.RUNNING:
            return  # Already running

        self._running = True
        self._coordinator_status = CoordinatorStatus.RUNNING
        self._start_time = time.time()
        logger.info(f"Starting NodeRecoveryDaemon on {self.node_id}")

        # Register with coordinator protocol
        try:
            register_coordinator("node_recovery", self)
        except Exception as e:
            logger.debug(f"Failed to register coordinator: {e}")

        # Subscribe to P2P node death events
        self._subscribe_to_events()

        # Start monitoring loop
        self._monitor_task = safe_create_task(
            self._monitor_loop(),
            name="node_recovery_monitor"
        )

    async def stop(self) -> None:
        """Stop the node recovery daemon."""
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return  # Already stopped

        self._coordinator_status = CoordinatorStatus.STOPPING
        logger.info("Stopping NodeRecoveryDaemon...")
        self._running = False

        # Stop monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Close HTTP session
        if self._http_session:
            await self._http_session.close()

        # Unregister coordinator
        try:
            unregister_coordinator("node_recovery")
        except (KeyError, RuntimeError, AttributeError):
            pass

        self._coordinator_status = CoordinatorStatus.STOPPED
        logger.info(
            f"NodeRecoveryDaemon stopped. Stats: "
            f"{self._stats.nodes_recovered} recovered, "
            f"{self._stats.recovery_failures} failures"
        )

    def _subscribe_to_events(self) -> None:
        """Subscribe to cluster health events."""
        try:
            from app.coordination.event_router import get_router
            from app.distributed.data_events import DataEventType

            router = get_router()

            # Subscribe to node death events
            router.subscribe(
                DataEventType.P2P_NODES_DEAD.value,
                self._on_nodes_dead
            )

            logger.info("[NodeRecoveryDaemon] Subscribed to P2P_NODES_DEAD events")

        except ImportError:
            logger.debug("Event router not available")
        except Exception as e:
            logger.warning(f"Failed to subscribe to events: {e}")

    def _on_nodes_dead(self, event) -> None:
        """Handle P2P_NODES_DEAD event."""
        payload = event.payload if hasattr(event, 'payload') else event
        dead_nodes = payload.get("nodes", [])

        for node_id in dead_nodes:
            if node_id in self._node_states:
                node = self._node_states[node_id]
                node.consecutive_failures += 1
                node.status = "unreachable"
                logger.info(
                    f"[NodeRecoveryDaemon] Node {node_id} marked unreachable "
                    f"(failures: {node.consecutive_failures})"
                )

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_nodes()
                self._stats.total_checks += 1
                self._stats.last_check_time = time.time()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors_count += 1
                self._last_error = str(e)
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _check_nodes(self) -> None:
        """Check all known nodes and trigger recovery if needed."""
        try:
            # Get current cluster state
            await self._update_node_states()

            # Check each node for recovery needs
            for node_id, node in list(self._node_states.items()):
                action = self._determine_recovery_action(node)

                if action != RecoveryAction.NONE:
                    await self._execute_recovery(node, action)

        except Exception as e:
            logger.warning(f"Node check error: {e}")

    async def _update_node_states(self) -> None:
        """Update node states from P2P cluster."""
        try:
            from app.distributed.p2p_orchestrator import get_p2p_orchestrator

            p2p = get_p2p_orchestrator()
            if p2p is None:
                return

            status = await p2p.get_status()
            if not status:
                return

            # Get alive and dead peers
            alive_peers = status.get("alive_peers", [])
            dead_peers = status.get("dead_peers", [])

            # Update alive nodes
            if isinstance(alive_peers, list):
                for peer in alive_peers:
                    if isinstance(peer, dict):
                        node_id = peer.get("node_id", "")
                        if node_id:
                            self._update_node_info(node_id, peer, "running")

            # Update dead nodes
            if isinstance(dead_peers, list):
                for peer_id in dead_peers:
                    if isinstance(peer_id, str) and peer_id in self._node_states:
                        node = self._node_states[peer_id]
                        node.status = "unreachable"
                        node.consecutive_failures += 1

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to update node states: {e}")

    def _update_node_info(
        self,
        node_id: str,
        info: dict[str, Any],
        status: str,
    ) -> None:
        """Update or create node info from P2P data."""
        if node_id not in self._node_states:
            self._node_states[node_id] = NodeInfo(
                node_id=node_id,
                host=info.get("host", ""),
                provider=self._detect_provider(node_id, info),
                instance_id=info.get("instance_id", ""),
            )

        node = self._node_states[node_id]
        node.status = status
        node.last_seen = time.time()

        # Reset failure count on successful contact
        if status == "running":
            node.consecutive_failures = 0

        # Track memory for proactive recovery
        memory_used = info.get("memory_used_percent", 0.0)
        if memory_used > 0:
            now = time.time()
            node.memory_samples.append(memory_used)
            node.sample_timestamps.append(now)

            # Keep last N samples
            max_samples = 60  # 1 sample per check, 60 = 5 hours at 5-min intervals
            if len(node.memory_samples) > max_samples:
                node.memory_samples = node.memory_samples[-max_samples:]
                node.sample_timestamps = node.sample_timestamps[-max_samples:]

    def _detect_provider(
        self,
        node_id: str,
        info: dict[str, Any],
    ) -> NodeProvider:
        """Detect the cloud provider for a node."""
        provider = info.get("provider", "").lower()

        if "lambda" in provider or "lambda" in node_id.lower():
            return NodeProvider.LAMBDA
        elif "vast" in provider or "vast" in node_id.lower():
            return NodeProvider.VAST
        elif "runpod" in provider or "runpod" in node_id.lower():
            return NodeProvider.RUNPOD
        elif "hetzner" in provider or "hetzner" in node_id.lower():
            return NodeProvider.HETZNER

        return NodeProvider.UNKNOWN

    def _determine_recovery_action(self, node: NodeInfo) -> RecoveryAction:
        """Determine what recovery action to take for a node."""
        now = time.time()

        # Check cooldown
        if node.last_recovery_attempt > 0:
            time_since_recovery = now - node.last_recovery_attempt
            if time_since_recovery < self.config.recovery_cooldown_seconds:
                return RecoveryAction.NONE

        # Check for terminated/failed status
        if node.status in ("terminated", "failed"):
            if node.consecutive_failures >= self.config.max_consecutive_failures:
                return RecoveryAction.RESTART
            else:
                return RecoveryAction.NOTIFY

        # Check for unreachable nodes
        if node.status == "unreachable":
            if node.consecutive_failures >= self.config.max_consecutive_failures:
                return RecoveryAction.RESTART
            else:
                return RecoveryAction.NOTIFY

        # Proactive recovery based on resource trends
        if self.config.preemptive_recovery_enabled:
            trend_action = self._check_resource_trends(node)
            if trend_action != RecoveryAction.NONE:
                return trend_action

        return RecoveryAction.NONE

    def _check_resource_trends(self, node: NodeInfo) -> RecoveryAction:
        """Check for resource exhaustion trends for preemptive recovery."""
        if len(node.memory_samples) < 5:
            return RecoveryAction.NONE

        now = time.time()
        window_seconds = self.config.memory_exhaustion_window_minutes * 60

        # Filter to samples within the window
        recent_samples = [
            (t, m) for t, m in zip(node.sample_timestamps, node.memory_samples)
            if now - t < window_seconds
        ]

        if len(recent_samples) < 5:
            return RecoveryAction.NONE

        # Calculate memory growth rate (% per minute)
        first_time, first_mem = recent_samples[0]
        last_time, last_mem = recent_samples[-1]

        time_diff_minutes = (last_time - first_time) / 60
        if time_diff_minutes < 5:
            return RecoveryAction.NONE

        growth_rate = (last_mem - first_mem) / time_diff_minutes

        # Check if memory is growing too fast
        if growth_rate > self.config.memory_exhaustion_threshold:
            # Project when memory will be exhausted (assuming 100% = exhaustion)
            remaining_memory = 100.0 - last_mem
            if remaining_memory > 0:
                minutes_to_exhaustion = remaining_memory / growth_rate
                if minutes_to_exhaustion < 60:  # Less than 60 minutes
                    logger.warning(
                        f"[NodeRecoveryDaemon] Node {node.node_id} memory exhaustion "
                        f"projected in {minutes_to_exhaustion:.0f} minutes "
                        f"(growth rate: {growth_rate:.2f}%/min)"
                    )
                    return RecoveryAction.PREEMPTIVE_RESTART

        return RecoveryAction.NONE

    async def _execute_recovery(
        self,
        node: NodeInfo,
        action: RecoveryAction,
    ) -> bool:
        """Execute a recovery action for a node."""
        node.last_recovery_attempt = time.time()

        if action == RecoveryAction.NOTIFY:
            # Just emit an event
            self._emit_recovery_event(node, action, success=True)
            return True

        if action == RecoveryAction.RESTART:
            success = await self._restart_node(node)
            if success:
                self._stats.nodes_recovered += 1
            else:
                self._stats.recovery_failures += 1
            self._emit_recovery_event(node, action, success)
            return success

        if action == RecoveryAction.PREEMPTIVE_RESTART:
            success = await self._restart_node(node)
            if success:
                self._stats.preemptive_recoveries += 1
            else:
                self._stats.recovery_failures += 1
            self._emit_recovery_event(node, action, success)
            return success

        return False

    async def _restart_node(self, node: NodeInfo) -> bool:
        """Restart a node via its provider's API."""
        if node.provider == NodeProvider.LAMBDA:
            return await self._restart_lambda_node(node)
        elif node.provider == NodeProvider.VAST:
            return await self._restart_vast_node(node)
        elif node.provider == NodeProvider.RUNPOD:
            return await self._restart_runpod_node(node)
        else:
            logger.warning(
                f"[NodeRecoveryDaemon] Cannot restart {node.node_id}: "
                f"unknown provider {node.provider}"
            )
            return False

    async def _restart_lambda_node(self, node: NodeInfo) -> bool:
        """Restart a Lambda Cloud instance."""
        if not self.config.lambda_api_key:
            logger.warning(
                f"[NodeRecoveryDaemon] Cannot restart Lambda node {node.node_id}: "
                "LAMBDA_API_KEY not set"
            )
            return False

        if not node.instance_id:
            logger.warning(
                f"[NodeRecoveryDaemon] Cannot restart Lambda node {node.node_id}: "
                "no instance_id"
            )
            return False

        try:
            import aiohttp

            if self._http_session is None:
                self._http_session = aiohttp.ClientSession()

            headers = {
                "Authorization": f"Bearer {self.config.lambda_api_key}",
                "Content-Type": "application/json",
            }

            # Lambda Cloud API for instance restart
            # Note: This is a placeholder - Lambda API may differ
            url = f"https://cloud.lambdalabs.com/api/v1/instances/{node.instance_id}/restart"

            async with self._http_session.post(url, headers=headers) as resp:
                if resp.status == 200:
                    logger.info(
                        f"[NodeRecoveryDaemon] Successfully restarted Lambda node "
                        f"{node.node_id} (instance: {node.instance_id})"
                    )
                    return True
                else:
                    body = await resp.text()
                    logger.error(
                        f"[NodeRecoveryDaemon] Failed to restart Lambda node "
                        f"{node.node_id}: {resp.status} - {body}"
                    )
                    return False

        except ImportError:
            logger.error("[NodeRecoveryDaemon] aiohttp not available for API calls")
            return False
        except Exception as e:
            logger.error(f"[NodeRecoveryDaemon] Lambda restart failed: {e}")
            return False

    async def _restart_vast_node(self, node: NodeInfo) -> bool:
        """Restart a Vast.ai instance.

        Uses the VastManager to start/stop instances via vastai CLI.
        The vast_api_key env var is read by vastai CLI directly.
        """
        try:
            from app.providers.vast_manager import VastManager

            manager = VastManager()

            # Extract instance ID from node metadata or node_id
            instance_id = node.metadata.get("instance_id") or node.node_id

            # Check if it's a valid Vast.ai instance ID (numeric)
            if not instance_id or not str(instance_id).isdigit():
                # Try to find by host match
                instances = await manager.list_instances()
                for inst in instances:
                    if node.host in [inst.public_ip, inst.metadata.get("ssh_host", "")]:
                        instance_id = inst.instance_id
                        break

            if not instance_id:
                logger.warning(
                    f"[NodeRecoveryDaemon] Cannot find Vast instance for {node.node_id}"
                )
                return False

            logger.info(
                f"[NodeRecoveryDaemon] Restarting Vast.ai instance {instance_id} "
                f"for node {node.node_id}"
            )

            # Stop then start for full restart
            stop_success = await manager.stop_instance(str(instance_id))
            if not stop_success:
                logger.warning(
                    f"[NodeRecoveryDaemon] Failed to stop Vast instance {instance_id}, "
                    "attempting start anyway"
                )

            # Wait briefly for stop to complete
            await asyncio.sleep(5)

            # Start the instance
            start_success = await manager.start_instance(str(instance_id))
            if start_success:
                logger.info(
                    f"[NodeRecoveryDaemon] Successfully restarted Vast instance {instance_id}"
                )
                return True
            else:
                logger.error(
                    f"[NodeRecoveryDaemon] Failed to start Vast instance {instance_id}"
                )
                return False

        except ImportError:
            logger.warning(
                "[NodeRecoveryDaemon] VastManager not available for Vast restart"
            )
            return False
        except Exception as e:
            logger.error(f"[NodeRecoveryDaemon] Vast restart error: {e}")
            return False

    async def _restart_runpod_node(self, node: NodeInfo) -> bool:
        """Restart a RunPod instance.

        December 2025: Implements RunPod API integration for node recovery.
        Uses the RunPod REST API to stop and start pods.

        API Reference: https://docs.runpod.io/reference/api-reference
        """
        if not self.config.runpod_api_key:
            logger.warning(
                f"[NodeRecoveryDaemon] Cannot restart RunPod node {node.node_id}: "
                "RUNPOD_API_KEY not set"
            )
            return False

        try:
            import aiohttp

            # Extract pod ID from node metadata or node_id
            # RunPod pod IDs are typically in format like "abc123xyz"
            pod_id = node.metadata.get("pod_id") or node.metadata.get("runpod_id")

            if not pod_id:
                # Try to extract from node_id if it contains runpod pattern
                if "runpod-" in node.node_id.lower():
                    # Format: runpod-abc123xyz or similar
                    pod_id = node.node_id.lower().replace("runpod-", "").split("-")[0]
                else:
                    pod_id = node.node_id

            logger.info(
                f"[NodeRecoveryDaemon] Attempting to restart RunPod pod {pod_id} "
                f"for node {node.node_id}"
            )

            headers = {
                "Authorization": f"Bearer {self.config.runpod_api_key}",
                "Content-Type": "application/json",
            }

            base_url = "https://api.runpod.io/v2"

            async with aiohttp.ClientSession() as session:
                # First, check pod status
                async with session.get(
                    f"{base_url}/pods/{pod_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 404:
                        logger.warning(
                            f"[NodeRecoveryDaemon] RunPod pod {pod_id} not found"
                        )
                        return False
                    elif resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(
                            f"[NodeRecoveryDaemon] RunPod API error: {resp.status} - {error_text}"
                        )
                        return False

                    pod_data = await resp.json()
                    current_status = pod_data.get("status", "unknown")
                    logger.debug(f"[NodeRecoveryDaemon] Pod {pod_id} status: {current_status}")

                # Stop the pod
                async with session.post(
                    f"{base_url}/pods/{pod_id}/stop",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status not in (200, 202, 204):
                        error_text = await resp.text()
                        logger.warning(
                            f"[NodeRecoveryDaemon] Failed to stop RunPod pod: "
                            f"{resp.status} - {error_text}"
                        )
                        # Continue anyway - pod might already be stopped

                # Wait for stop to complete
                logger.debug(f"[NodeRecoveryDaemon] Waiting for pod {pod_id} to stop...")
                await asyncio.sleep(10)

                # Start the pod
                async with session.post(
                    f"{base_url}/pods/{pod_id}/start",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status in (200, 202):
                        logger.info(
                            f"[NodeRecoveryDaemon] Successfully restarted RunPod pod {pod_id}"
                        )
                        return True
                    else:
                        error_text = await resp.text()
                        logger.error(
                            f"[NodeRecoveryDaemon] Failed to start RunPod pod: "
                            f"{resp.status} - {error_text}"
                        )
                        return False

        except ImportError:
            logger.warning(
                "[NodeRecoveryDaemon] aiohttp not available for RunPod API calls"
            )
            return False
        except asyncio.TimeoutError:
            logger.error(f"[NodeRecoveryDaemon] RunPod API timeout for pod {pod_id}")
            return False
        except Exception as e:
            logger.error(f"[NodeRecoveryDaemon] RunPod restart error: {e}")
            return False

    def _emit_recovery_event(
        self,
        node: NodeInfo,
        action: RecoveryAction,
        success: bool,
    ) -> None:
        """Emit event for recovery action."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            # Phase 22.2 fix: Use publish_sync instead of emit (which doesn't exist)
            router.publish_sync(
                "node_recovery_triggered",
                {
                    "node_id": node.node_id,
                    "provider": node.provider.value,
                    "action": action.value,
                    "success": success,
                    "consecutive_failures": node.consecutive_failures,
                    "timestamp": time.time(),
                },
                source="node_recovery_daemon",
            )
        except Exception as e:
            logger.debug(f"Could not publish recovery event: {e}")

        # Emit standard health events (December 2025 - Phase 21)
        if HAS_HEALTH_EVENTS:
            self._emit_health_event(node, action, success)

    def _emit_health_event(
        self,
        node: NodeInfo,
        action: RecoveryAction,
        success: bool,
    ) -> None:
        """Emit standard cluster health events for coordination."""
        import asyncio

        async def _emit():
            try:
                if success and action in (RecoveryAction.RESTART, RecoveryAction.PREEMPTIVE_RESTART):
                    # Node was successfully recovered
                    await emit_node_recovered(
                        node_id=node.node_id,
                        node_ip=node.host,
                        recovery_time_seconds=0.0,  # Could track this if needed
                        source="node_recovery_daemon",
                    )
                elif not success or node.status in ("terminated", "failed", "unreachable"):
                    # Node is unhealthy
                    await emit_node_unhealthy(
                        node_id=node.node_id,
                        reason=f"Status: {node.status}, Action: {action.value}",
                        node_ip=node.host,
                        consecutive_failures=node.consecutive_failures,
                        source="node_recovery_daemon",
                    )
            except Exception as e:
                logger.debug(f"Could not emit health event: {e}")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_emit())
        except RuntimeError:
            # No running loop - try to run directly
            try:
                asyncio.run(_emit())
            except Exception:
                pass  # Best effort

    def get_stats(self) -> dict[str, Any]:
        """Get daemon statistics."""
        return {
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "total_checks": self._stats.total_checks,
            "nodes_recovered": self._stats.nodes_recovered,
            "recovery_failures": self._stats.recovery_failures,
            "preemptive_recoveries": self._stats.preemptive_recoveries,
            "last_check_time": self._stats.last_check_time,
            "last_error": self._stats.last_error,
            "tracked_nodes": len(self._node_states),
            "errors_count": self._errors_count,
        }

    def get_node_states(self) -> dict[str, dict[str, Any]]:
        """Get current node states."""
        return {
            node_id: {
                "node_id": node.node_id,
                "host": node.host,
                "provider": node.provider.value,
                "status": node.status,
                "consecutive_failures": node.consecutive_failures,
                "last_seen": node.last_seen,
            }
            for node_id, node in self._node_states.items()
        }

    # CoordinatorProtocol methods
    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self._running,
            "status": self._coordinator_status.value,
            "stats": self.get_stats(),
            "nodes": self.get_node_states(),
        }

    def get_status(self) -> CoordinatorStatus:
        """Get coordinator status."""
        return self._coordinator_status
