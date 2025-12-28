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

Dec 2025: Refactored to use BaseDaemon base class (~100 LOC reduction).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.coordination.base_daemon import BaseDaemon, DaemonConfig
from app.coordination.contracts import HealthCheckResult
from app.coordination.protocols import CoordinatorStatus
# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import JobDaemonStats

logger = logging.getLogger(__name__)

# Health event emission (uses safe fallbacks internally)
from app.coordination.event_emitters import (
    emit_node_unhealthy,
    emit_node_recovered,
)


class NodeRecoveryAction(Enum):
    """Types of node-level recovery actions.

    NOTE (Dec 2025): Renamed from RecoveryAction to avoid collision with
    JobRecoveryAction in unified_health_manager.py and SystemRecoveryAction
    in recovery_orchestrator.py which have different semantics.
    """
    NONE = "none"
    RESTART = "restart"
    PREEMPTIVE_RESTART = "preemptive_restart"
    NOTIFY = "notify"  # Just notify, don't auto-recover
    FAILOVER = "failover"  # Migrate workload to another node


# Backward-compat alias (deprecated)
RecoveryAction = NodeRecoveryAction


class NodeProvider(Enum):
    """Cloud provider types."""
    LAMBDA = "lambda"
    VAST = "vast"
    RUNPOD = "runpod"
    HETZNER = "hetzner"
    UNKNOWN = "unknown"


@dataclass
class NodeRecoveryConfig(DaemonConfig):
    """Configuration for node recovery.

    Extends DaemonConfig with recovery-specific settings.
    """

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
    def from_env(cls, prefix: str = "RINGRIFT_NODE_RECOVERY") -> "NodeRecoveryConfig":
        """Load configuration from environment variables."""
        config = cls()
        config.enabled = os.environ.get(f"{prefix}_ENABLED", "1") == "1"
        if os.environ.get(f"{prefix}_INTERVAL"):
            config.check_interval_seconds = int(os.environ.get(f"{prefix}_INTERVAL", "300"))
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
class RecoveryStats(JobDaemonStats):
    """Statistics for recovery operations.

    December 2025: Now extends JobDaemonStats for consistent tracking.
    Inherits: jobs_processed, jobs_succeeded, jobs_failed, is_healthy(), etc.
    """

    # Recovery-specific fields
    preemptive_recoveries: int = 0

    # Backward compatibility aliases
    @property
    def total_checks(self) -> int:
        """Alias for jobs_processed (backward compatibility)."""
        return self.jobs_processed

    @property
    def nodes_recovered(self) -> int:
        """Alias for jobs_succeeded (backward compatibility)."""
        return self.jobs_succeeded

    @property
    def recovery_failures(self) -> int:
        """Alias for jobs_failed (backward compatibility)."""
        return self.jobs_failed

    def record_check(self) -> None:
        """Record a node health check."""
        self.last_job_time = time.time()
        self.jobs_processed += 1

    def record_recovery_success(self, preemptive: bool = False) -> None:
        """Record a successful recovery."""
        self.record_job_success()
        if preemptive:
            self.preemptive_recoveries += 1

    def record_recovery_failure(self, error: str) -> None:
        """Record a failed recovery."""
        self.record_job_failure(error)


class NodeRecoveryDaemon(BaseDaemon[NodeRecoveryConfig]):
    """Daemon that monitors nodes and triggers recovery actions.

    Continuously monitors cluster for terminated or failing nodes
    and automatically triggers recovery when possible.

    Inherits from BaseDaemon which provides:
    - Lifecycle management (start/stop)
    - Coordinator protocol registration
    - Protected main loop with error handling
    - Health check interface
    """

    def __init__(self, config: NodeRecoveryConfig | None = None):
        super().__init__(config)
        self._stats = RecoveryStats()

        # Track node states
        self._node_states: dict[str, NodeInfo] = {}

        # HTTP session for API calls
        self._http_session = None

    @staticmethod
    def _get_default_config() -> NodeRecoveryConfig:
        """Return default configuration."""
        return NodeRecoveryConfig.from_env()

    async def _on_start(self) -> None:
        """Initialize on startup."""
        logger.info(
            f"[{self._get_daemon_name()}] Config: "
            f"interval={self.config.check_interval_seconds}s, "
            f"lambda_api={'set' if self.config.lambda_api_key else 'not set'}"
        )
        # Subscribe to P2P node death events
        self._subscribe_to_events()

    def _subscribe_to_events(self) -> None:
        """Subscribe to cluster health events."""
        try:
            from app.coordination.event_router import get_router, DataEventType

            router = get_router()

            # Subscribe to node death events (batch)
            router.subscribe(
                DataEventType.P2P_NODES_DEAD.value,
                self._on_nodes_dead
            )

            # Subscribe to single node death events (Dec 2025 fix)
            if hasattr(DataEventType, 'P2P_NODE_DEAD'):
                router.subscribe(
                    DataEventType.P2P_NODE_DEAD.value,
                    self._on_single_node_dead
                )
                logger.info("[NodeRecoveryDaemon] Subscribed to P2P_NODE_DEAD events")

            logger.info("[NodeRecoveryDaemon] Subscribed to P2P_NODES_DEAD events")

        except ImportError:
            logger.debug("Event router not available")
        except Exception as e:
            logger.warning(f"Failed to subscribe to events: {e}")

    def _on_nodes_dead(self, event) -> None:
        """Handle P2P_NODES_DEAD event (batch of dead nodes)."""
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

    def _on_single_node_dead(self, event) -> None:
        """Handle P2P_NODE_DEAD event (single node).

        Dec 2025: Added to handle single node death events from P2P orchestrator.
        Previously only batch P2P_NODES_DEAD was handled, causing single node
        failures to be missed until batch heartbeat timeout.
        """
        payload = event.payload if hasattr(event, 'payload') else event
        node_id = payload.get("node_id")
        reason = payload.get("reason", "unknown")

        if not node_id:
            logger.warning("[NodeRecoveryDaemon] Received P2P_NODE_DEAD without node_id")
            return

        if node_id in self._node_states:
            node = self._node_states[node_id]
            node.consecutive_failures += 1
            node.status = "unreachable"
            node.last_failure_reason = reason
            logger.info(
                f"[NodeRecoveryDaemon] Single node {node_id} confirmed dead "
                f"(reason: {reason}, failures: {node.consecutive_failures})"
            )
        else:
            # Create new state for unknown node
            logger.info(
                f"[NodeRecoveryDaemon] Unknown node {node_id} reported dead, "
                f"will track on next cluster scan"
            )

    async def _on_stop(self) -> None:
        """Graceful shutdown handler.

        December 2025: Added for clean daemon shutdown with event emission.
        """
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()

        # Log stats
        logger.info(
            f"[{self._get_daemon_name()}] Stats: "
            f"{self._stats.nodes_recovered} recovered, "
            f"{self._stats.recovery_failures} failures"
        )

        try:
            # Emit shutdown event
            from app.coordination.event_emitters import emit_coordinator_shutdown

            await emit_coordinator_shutdown(
                coordinator_name=self._get_daemon_name(),
                reason="graceful",
                remaining_tasks=0,
                state_snapshot={
                    "uptime_seconds": self.uptime_seconds,
                    "total_checks": self._stats.total_checks,
                    "nodes_recovered": self._stats.nodes_recovered,
                    "recovery_failures": self._stats.recovery_failures,
                    "tracked_nodes": len(self._node_states),
                },
            )
            logger.info(f"[{self._get_daemon_name()}] Graceful shutdown complete")
        except ImportError:
            logger.debug(f"[{self._get_daemon_name()}] Event emitters not available for shutdown")
        except Exception as e:
            logger.warning(f"[{self._get_daemon_name()}] Error during shutdown: {e}")

    async def _run_cycle(self) -> None:
        """Run one recovery cycle.

        Called by BaseDaemon's protected main loop.
        """
        await self._check_nodes()
        self._stats.record_check()  # Updates jobs_processed and last_job_time

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
            from app.coordination.p2p_integration import get_p2p_orchestrator

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
        """Execute a recovery action for a node.

        P0.5 Dec 2025: Now includes post-restart health verification to ensure
        the node actually came back up after restart API call.
        """
        node.last_recovery_attempt = time.time()

        if action == RecoveryAction.NOTIFY:
            # Just emit an event
            self._emit_recovery_event(node, action, success=True)
            return True

        if action == RecoveryAction.RESTART:
            api_success = await self._restart_node(node)
            if api_success:
                # P0.5 Dec 2025: Verify node actually came back up
                verified = await self._verify_node_health_after_restart(node)
                if verified:
                    self._stats.record_recovery_success()
                    self._emit_recovery_event(node, action, success=True)
                    return True
                else:
                    # API succeeded but node didn't come back up
                    self._stats.record_recovery_failure(
                        f"verification_failed:{node.node_id}"
                    )
                    self._emit_recovery_event(node, action, success=False)
                    return False
            else:
                self._stats.record_recovery_failure(f"restart_failed:{node.node_id}")
                self._emit_recovery_event(node, action, success=False)
                return False

        if action == RecoveryAction.PREEMPTIVE_RESTART:
            api_success = await self._restart_node(node)
            if api_success:
                # P0.5 Dec 2025: Verify node actually came back up
                verified = await self._verify_node_health_after_restart(node)
                if verified:
                    self._stats.record_recovery_success(preemptive=True)
                    self._emit_recovery_event(node, action, success=True)
                    return True
                else:
                    # API succeeded but node didn't come back up
                    self._stats.record_recovery_failure(
                        f"preemptive_verification_failed:{node.node_id}"
                    )
                    self._emit_recovery_event(node, action, success=False)
                    return False
            else:
                self._stats.record_recovery_failure(
                    f"preemptive_restart_failed:{node.node_id}"
                )
                self._emit_recovery_event(node, action, success=False)
                return False

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
        """Restart a Lambda Cloud instance.

        Uses the Lambda Cloud API instance-operations/restart endpoint.
        API Reference: https://cloud.lambdalabs.com/api/v1/docs
        """
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
                # Use 60s default timeout for Lambda API operations
                timeout = aiohttp.ClientTimeout(total=60)
                self._http_session = aiohttp.ClientSession(timeout=timeout)

            # Lambda API uses Basic Auth with API key as username
            # The trailing colon indicates no password
            auth = aiohttp.BasicAuth(self.config.lambda_api_key, "")
            headers = {
                "Content-Type": "application/json",
            }

            # Lambda Cloud API restart endpoint
            # POST https://cloud.lambdalabs.com/api/v1/instance-operations/restart
            # Body: {"instance_ids": ["instance-id-here"]}
            url = "https://cloud.lambdalabs.com/api/v1/instance-operations/restart"
            payload = {"instance_ids": [node.instance_id]}

            async with self._http_session.post(
                url, headers=headers, auth=auth, json=payload, timeout=30
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    restarted = result.get("restarted_instances", [])
                    if node.instance_id in [i.get("id") for i in restarted]:
                        logger.info(
                            f"[NodeRecoveryDaemon] Successfully restarted Lambda node "
                            f"{node.node_id} (instance: {node.instance_id})"
                        )
                        return True
                    else:
                        logger.warning(
                            f"[NodeRecoveryDaemon] Lambda restart response OK but instance "
                            f"not in restarted list: {result}"
                        )
                        return True  # API call succeeded, assume restart initiated
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

    async def _verify_node_health_after_restart(
        self,
        node: NodeInfo,
        poll_intervals: list[float] | None = None,
    ) -> bool:
        """Verify node health after restart by polling at specified intervals.

        P0.5 Dec 2025: Ensures restart actually succeeded by verifying node
        becomes reachable after restart API call completes.

        Args:
            node: The node to verify
            poll_intervals: List of seconds to wait between polls.
                           Default: [10, 20, 30] (poll at 10s, 30s, 60s after restart)

        Returns:
            True if node becomes healthy within the poll window, False otherwise.
        """
        if poll_intervals is None:
            poll_intervals = [10.0, 20.0, 30.0]

        logger.info(
            f"[NodeRecoveryDaemon] Verifying health of {node.node_id} "
            f"after restart (intervals: {poll_intervals}s)"
        )

        for i, interval in enumerate(poll_intervals):
            logger.debug(
                f"[NodeRecoveryDaemon] Waiting {interval}s before health check {i + 1}"
            )
            await asyncio.sleep(interval)

            # Check node health via P2P status or direct SSH
            is_healthy = await self._check_single_node_health(node)

            if is_healthy:
                total_wait = sum(poll_intervals[:i + 1])
                logger.info(
                    f"[NodeRecoveryDaemon] Node {node.node_id} verified healthy "
                    f"after {total_wait:.0f}s"
                )
                # Reset failure state on successful verification
                node.status = "running"
                node.consecutive_failures = 0
                return True

            logger.debug(
                f"[NodeRecoveryDaemon] Node {node.node_id} not yet healthy "
                f"after {sum(poll_intervals[:i + 1]):.0f}s"
            )

        # All polls failed
        total_wait = sum(poll_intervals)
        logger.warning(
            f"[NodeRecoveryDaemon] Node {node.node_id} failed health verification "
            f"after {total_wait:.0f}s"
        )
        return False

    async def _check_single_node_health(self, node: NodeInfo) -> bool:
        """Check if a single node is healthy via P2P or SSH.

        P0.5 Dec 2025: Helper for post-restart verification.

        Args:
            node: The node to check

        Returns:
            True if node responds to health check, False otherwise.
        """
        # Try P2P status first
        try:
            from app.coordination.p2p_integration import get_p2p_orchestrator

            p2p = get_p2p_orchestrator()
            if p2p is not None:
                status = await p2p.get_status()
                if status:
                    alive_peers = status.get("alive_peers", [])
                    if isinstance(alive_peers, list):
                        for peer in alive_peers:
                            if isinstance(peer, dict):
                                if peer.get("node_id") == node.node_id:
                                    return True
                            elif isinstance(peer, str) and peer == node.node_id:
                                return True
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"[NodeRecoveryDaemon] P2P check failed: {e}")

        # Fallback: Try SSH health check
        if node.host:
            try:
                from app.core.ssh import run_ssh_command_async

                result = await asyncio.wait_for(
                    run_ssh_command_async(
                        node.node_id,  # Uses cluster config for SSH details
                        "echo healthy",
                        timeout=10,
                    ),
                    timeout=15.0,
                )
                if result and result.success and "healthy" in result.stdout:
                    return True
            except (ImportError, asyncio.TimeoutError, RuntimeError, OSError) as e:
                logger.debug(f"[NodeRecoveryDaemon] SSH health check failed: {e}")

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

        # Emit standard health events
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
            except RuntimeError as e:
                logger.debug(f"Failed to run recovery event emission (nested loop?): {e}")
            except (OSError, IOError) as e:
                logger.debug(f"Failed to emit recovery event (I/O error): {e}")

    async def health_check(self) -> HealthCheckResult:
        """Check if the daemon is healthy.

        Returns HealthCheckResult for protocol compliance.
        Used by DaemonManager for crash detection and auto-restart.

        Dec 2025: Fixed to return HealthCheckResult instead of bool.
        """
        is_healthy = True
        message = "Node recovery daemon running"
        status = CoordinatorStatus.RUNNING

        # Check if daemon is running
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Node recovery daemon not running",
                details={"running": False},
            )

        # Check if we have recent check data (within 2x check interval)
        if self._stats.last_job_time > 0:
            max_age = self.config.check_interval_seconds * 2
            age = time.time() - self._stats.last_job_time
            if age > max_age:
                is_healthy = False
                status = CoordinatorStatus.DEGRADED
                message = f"Stale check data (age={age:.0f}s, max={max_age}s)"
                logger.warning(f"[{self._get_daemon_name()}] Health check: {message}")

        # Check for excessive failures
        if is_healthy and self._stats.recovery_failures > 10:
            # Allow if we also have successes (not just all failures)
            if self._stats.nodes_recovered == 0:
                is_healthy = False
                status = CoordinatorStatus.DEGRADED
                message = f"Only failures, no recoveries (failures={self._stats.recovery_failures})"
                logger.warning(f"[{self._get_daemon_name()}] Health check: {message}")

        # Check HTTP session health if used
        if is_healthy and self._http_session is not None and self._http_session.closed:
            is_healthy = False
            status = CoordinatorStatus.DEGRADED
            message = "HTTP session closed unexpectedly"
            logger.warning(f"[{self._get_daemon_name()}] Health check: {message}")

        return HealthCheckResult(
            healthy=is_healthy,
            status=status,
            message=message,
            details={
                "running": self._running,
                "nodes_recovered": self._stats.nodes_recovered,
                "recovery_failures": self._stats.recovery_failures,
                "last_job_time": self._stats.last_job_time,
            },
        )

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

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring.

        Extends base class status with recovery-specific fields.
        """
        status = super().get_status()

        # Add recovery-specific stats
        status["recovery_stats"] = {
            "total_checks": self._stats.total_checks,
            "nodes_recovered": self._stats.nodes_recovered,
            "recovery_failures": self._stats.recovery_failures,
            "preemptive_recoveries": self._stats.preemptive_recoveries,
            "last_check_time": self._stats.last_job_time,  # Use underlying field
            "last_error": self._stats.last_error,
        }
        status["tracked_nodes"] = len(self._node_states)
        status["nodes"] = self.get_node_states()

        return status


# =============================================================================
# Module-level Singleton Factory (December 2025)
# =============================================================================

_node_recovery_daemon: NodeRecoveryDaemon | None = None


def get_node_recovery_daemon() -> NodeRecoveryDaemon:
    """Get the singleton NodeRecoveryDaemon instance.

    December 2025: Added for consistency with other daemon factories.
    """
    global _node_recovery_daemon
    if _node_recovery_daemon is None:
        _node_recovery_daemon = NodeRecoveryDaemon()
    return _node_recovery_daemon


def reset_node_recovery_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _node_recovery_daemon
    _node_recovery_daemon = None
