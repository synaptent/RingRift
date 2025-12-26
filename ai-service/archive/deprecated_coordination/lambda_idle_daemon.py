"""Lambda Idle Shutdown Daemon.

Automatically terminates idle Lambda Labs GPU nodes to reduce costs.
Monitors cluster for nodes that have been idle beyond the threshold and
safely terminates them after confirming no pending work.

NOTE: Lambda Labs GPU account currently suspended pending support ticket resolution.
      This daemon will resume normal operation once the account is reinstated.

Key features:
- Monitors Lambda nodes for idle detection (30+ minutes at <5% GPU utilization)
- Checks for pending work before termination
- Graceful shutdown with pending job drain
- Cost tracking and savings reporting
- Event emission for observability

Usage:
    from app.coordination.lambda_idle_daemon import LambdaIdleDaemon

    daemon = LambdaIdleDaemon()
    await daemon.start()
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.protocols import (
    CoordinatorStatus,
    register_coordinator,
    unregister_coordinator,
)
from app.core.async_context import safe_create_task

logger = logging.getLogger(__name__)


@dataclass
class LambdaIdleConfig:
    """Configuration for Lambda idle shutdown."""
    enabled: bool = True
    # Check interval (5 minutes)
    check_interval_seconds: int = 300
    # Idle threshold - node must be below this GPU utilization
    idle_threshold_percent: float = 5.0
    # Time node must be idle before termination (30 minutes)
    idle_duration_seconds: int = 1800
    # Grace period after announcing shutdown (60 seconds)
    shutdown_grace_seconds: int = 60
    # Minimum nodes to keep running (always keep at least 1)
    min_nodes_to_keep: int = 1
    # Maximum nodes to terminate per cycle (prevent mass termination)
    max_terminations_per_cycle: int = 2
    # Don't terminate nodes that were recently active (cooldown)
    recent_activity_cooldown_seconds: int = 300
    # Lambda provider name patterns
    lambda_provider_patterns: list[str] = field(
        default_factory=lambda: ["lambda", "Lambda", "gh200"]
    )

    @classmethod
    def from_env(cls) -> LambdaIdleConfig:
        """Load configuration from environment variables."""
        config = cls()
        config.enabled = os.environ.get("RINGRIFT_LAMBDA_IDLE_ENABLED", "1") == "1"
        config.check_interval_seconds = int(
            os.environ.get("RINGRIFT_LAMBDA_IDLE_INTERVAL", "300")
        )
        config.idle_threshold_percent = float(
            os.environ.get("RINGRIFT_LAMBDA_IDLE_THRESHOLD", "5.0")
        )
        config.idle_duration_seconds = int(
            os.environ.get("RINGRIFT_LAMBDA_IDLE_DURATION", "1800")
        )
        config.min_nodes_to_keep = int(
            os.environ.get("RINGRIFT_LAMBDA_MIN_NODES", "1")
        )
        return config


@dataclass
class LambdaNodeState:
    """Tracked state for a Lambda node."""
    node_id: str
    host: str
    provider: str = "lambda"
    instance_id: str | None = None
    gpu_utilization: float = 0.0
    gpu_memory_total_gb: float = 0.0
    last_seen: float = 0.0
    idle_since: float = 0.0
    last_active: float = 0.0
    hourly_cost: float = 0.0
    is_terminating: bool = False


@dataclass
class TerminationEvent:
    """Record of a node termination."""
    node_id: str
    host: str
    timestamp: float
    idle_duration_seconds: float
    reason: str
    success: bool
    cost_saved_estimate: float = 0.0
    error: str | None = None


@dataclass
class DaemonStats:
    """Statistics for the daemon."""
    nodes_terminated: int = 0
    nodes_skipped: int = 0
    termination_failures: int = 0
    estimated_savings: float = 0.0  # Total $ saved
    last_check_time: float = 0.0
    last_termination_time: float = 0.0


class LambdaIdleDaemon:
    """Daemon that monitors and terminates idle Lambda nodes.

    Continuously monitors Lambda GPU nodes for idleness and automatically
    terminates them to reduce costs. Ensures safety by checking for pending
    work and maintaining minimum node counts.
    """

    def __init__(self, config: LambdaIdleConfig | None = None):
        self.config = config or LambdaIdleConfig.from_env()
        self.node_id = socket.gethostname()
        self._running = False
        self._stats = DaemonStats()
        self._monitor_task: asyncio.Task | None = None

        # Track Lambda node states
        self._node_states: dict[str, LambdaNodeState] = {}

        # Termination history
        self._termination_history: list[TerminationEvent] = []
        self._max_history: int = 100

        # CoordinatorProtocol state
        self._coordinator_status = CoordinatorStatus.INITIALIZING
        self._start_time: float = 0.0
        self._errors_count: int = 0
        self._last_error: str = ""

        logger.info(
            f"LambdaIdleDaemon initialized: node={self.node_id}, "
            f"idle_threshold={self.config.idle_threshold_percent}%, "
            f"idle_duration={self.config.idle_duration_seconds}s"
        )

    def is_running(self) -> bool:
        """Check if the daemon is running."""
        return self._running

    async def start(self) -> None:
        """Start the idle shutdown daemon."""
        if not self.config.enabled:
            self._coordinator_status = CoordinatorStatus.STOPPED
            logger.info("LambdaIdleDaemon disabled by config")
            return

        if self._coordinator_status == CoordinatorStatus.RUNNING:
            return

        self._running = True
        self._coordinator_status = CoordinatorStatus.RUNNING
        self._start_time = time.time()
        logger.info(f"Starting LambdaIdleDaemon on {self.node_id}")

        # Register with coordinator protocol
        try:
            register_coordinator("lambda_idle", self)
        except Exception as e:
            logger.debug(f"Failed to register coordinator: {e}")

        # Start monitoring loop
        self._monitor_task = safe_create_task(
            self._monitor_loop(),
            name="lambda_idle_monitor"
        )

    async def stop(self) -> None:
        """Stop the idle shutdown daemon."""
        if self._coordinator_status == CoordinatorStatus.STOPPED:
            return

        self._coordinator_status = CoordinatorStatus.STOPPING
        logger.info("Stopping LambdaIdleDaemon...")
        self._running = False

        # Stop monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Unregister coordinator
        try:
            unregister_coordinator("lambda_idle")
        except Exception as e:
            logger.debug(f"Failed to unregister coordinator: {e}")

        self._coordinator_status = CoordinatorStatus.STOPPED
        logger.info(
            f"LambdaIdleDaemon stopped. "
            f"Terminated {self._stats.nodes_terminated} nodes, "
            f"saved ${self._stats.estimated_savings:.2f}"
        )

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_and_terminate()
                self._stats.last_check_time = time.time()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors_count += 1
                self._last_error = str(e)
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(60)

    async def _check_and_terminate(self) -> None:
        """Check for idle Lambda nodes and terminate them."""
        try:
            # Get cluster node status
            nodes = await self._get_cluster_nodes()

            if not nodes:
                logger.debug("No cluster nodes found")
                return

            # Filter to Lambda nodes only
            lambda_nodes = [
                n for n in nodes
                if self._is_lambda_node(n)
            ]

            if not lambda_nodes:
                logger.debug("No Lambda nodes in cluster")
                return

            # Check for pending work
            pending_work = await self._get_pending_work_count()

            # Get idle Lambda nodes that should be terminated
            termination_candidates = self._get_termination_candidates(
                lambda_nodes, pending_work
            )

            if not termination_candidates:
                return

            # Log decision
            logger.info(
                f"[LambdaIdleDaemon] Found {len(termination_candidates)} idle nodes, "
                f"pending_work={pending_work}, will terminate up to "
                f"{self.config.max_terminations_per_cycle}"
            )

            # Terminate nodes (up to max per cycle)
            for node in termination_candidates[:self.config.max_terminations_per_cycle]:
                await self._terminate_node(node)

        except Exception as e:
            logger.warning(f"Check and terminate error: {e}")

    async def _get_cluster_nodes(self) -> list[LambdaNodeState]:
        """Get status of all cluster nodes."""
        nodes: list[LambdaNodeState] = []
        now = time.time()

        try:
            from app.distributed.p2p_orchestrator import get_p2p_orchestrator

            p2p = get_p2p_orchestrator()
            if p2p is None:
                return nodes

            # Get cluster status
            status = await p2p.get_status()
            if not status:
                return nodes

            # Parse alive peers
            alive_peers = status.get("alive_peers", [])
            if isinstance(alive_peers, int):
                return nodes

            for peer_info in alive_peers:
                if isinstance(peer_info, dict):
                    node = LambdaNodeState(
                        node_id=peer_info.get("node_id", ""),
                        host=peer_info.get("host", ""),
                        provider=peer_info.get("provider", "unknown"),
                        instance_id=peer_info.get("instance_id"),
                        gpu_utilization=peer_info.get("gpu_utilization", 0.0),
                        gpu_memory_total_gb=peer_info.get("gpu_memory_total", 0.0),
                        last_seen=peer_info.get("last_seen", now),
                        hourly_cost=peer_info.get("hourly_cost", 0.0),
                    )

                    # Update idle tracking
                    self._update_node_state(node)
                    nodes.append(node)

        except ImportError:
            logger.debug("P2P orchestrator not available")
        except Exception as e:
            logger.debug(f"Failed to get cluster nodes: {e}")

        return nodes

    def _update_node_state(self, node: LambdaNodeState) -> None:
        """Update tracked node state for idle duration tracking."""
        now = time.time()
        existing = self._node_states.get(node.node_id)

        if existing:
            # Check if node transitioned to idle
            if node.gpu_utilization < self.config.idle_threshold_percent:
                if existing.gpu_utilization >= self.config.idle_threshold_percent:
                    # Just became idle
                    node.idle_since = now
                else:
                    # Still idle, preserve idle_since
                    node.idle_since = existing.idle_since
            else:
                # Not idle, update last_active
                node.idle_since = 0.0
                node.last_active = now
        else:
            # New node
            if node.gpu_utilization < self.config.idle_threshold_percent:
                node.idle_since = now
            else:
                node.last_active = now

        self._node_states[node.node_id] = node

    def _is_lambda_node(self, node: LambdaNodeState) -> bool:
        """Check if a node is a Lambda Labs node."""
        # Check provider field
        for pattern in self.config.lambda_provider_patterns:
            if pattern.lower() in node.provider.lower():
                return True
            if pattern.lower() in node.node_id.lower():
                return True

        return False

    async def _get_pending_work_count(self) -> int:
        """Get count of pending work items in the queue."""
        try:
            from app.coordination.work_queue import get_work_queue

            queue = get_work_queue()
            if queue:
                status = queue.get_queue_status()
                return status.get("by_status", {}).get("pending", 0)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to get pending work: {e}")

        return 0

    def _get_termination_candidates(
        self,
        nodes: list[LambdaNodeState],
        pending_work: int,
    ) -> list[LambdaNodeState]:
        """Get list of Lambda nodes that should be terminated.

        Returns:
            List of nodes to terminate, sorted by idle duration (longest first)
        """
        now = time.time()
        candidates: list[LambdaNodeState] = []

        # Don't terminate if there's pending work
        if pending_work > 0:
            logger.debug(
                f"[LambdaIdleDaemon] Skipping termination: {pending_work} pending work items"
            )
            return []

        # Get active Lambda node count
        active_lambda_count = len(nodes)

        # Ensure we keep minimum nodes
        max_to_terminate = active_lambda_count - self.config.min_nodes_to_keep
        if max_to_terminate <= 0:
            logger.debug(
                f"[LambdaIdleDaemon] At minimum nodes ({self.config.min_nodes_to_keep}), "
                f"skipping termination"
            )
            return []

        for node in nodes:
            state = self._node_states.get(node.node_id, node)

            # Skip if already terminating
            if state.is_terminating:
                continue

            # Skip if not idle long enough
            if state.idle_since <= 0:
                continue

            idle_duration = now - state.idle_since
            if idle_duration < self.config.idle_duration_seconds:
                continue

            # Skip if recently active (cooldown)
            if state.last_active > 0:
                time_since_active = now - state.last_active
                if time_since_active < self.config.recent_activity_cooldown_seconds:
                    continue

            candidates.append(state)

        # Sort by idle duration (longest idle first)
        candidates.sort(key=lambda n: n.idle_since)

        # Cap at max_to_terminate
        return candidates[:max_to_terminate]

    async def _terminate_node(self, node: LambdaNodeState) -> bool:
        """Terminate a Lambda node.

        Returns:
            True if termination was successful
        """
        now = time.time()
        idle_duration = now - node.idle_since if node.idle_since > 0 else 0

        logger.info(
            f"[LambdaIdleDaemon] Terminating node {node.node_id} "
            f"(idle for {idle_duration:.0f}s, host={node.host})"
        )

        # Mark as terminating
        node.is_terminating = True
        self._node_states[node.node_id] = node

        # Emit shutdown announcement event
        self._emit_shutdown_event(node, "announcing")

        # Wait grace period
        await asyncio.sleep(self.config.shutdown_grace_seconds)

        # Attempt termination
        try:
            success = await self._call_termination_api(node)

            # Record event
            event = TerminationEvent(
                node_id=node.node_id,
                host=node.host,
                timestamp=now,
                idle_duration_seconds=idle_duration,
                reason="idle_threshold_exceeded",
                success=success,
                cost_saved_estimate=node.hourly_cost,  # 1 hour saved estimate
            )
            self._termination_history.append(event)
            if len(self._termination_history) > self._max_history:
                self._termination_history.pop(0)

            if success:
                self._stats.nodes_terminated += 1
                self._stats.estimated_savings += node.hourly_cost
                self._stats.last_termination_time = now

                # Remove from tracking
                self._node_states.pop(node.node_id, None)

                # Emit success event
                self._emit_shutdown_event(node, "terminated")

                logger.info(
                    f"[LambdaIdleDaemon] Successfully terminated {node.node_id}"
                )
                return True
            else:
                self._stats.termination_failures += 1
                node.is_terminating = False
                return False

        except Exception as e:
            self._stats.termination_failures += 1
            node.is_terminating = False

            # Record failure
            event = TerminationEvent(
                node_id=node.node_id,
                host=node.host,
                timestamp=now,
                idle_duration_seconds=idle_duration,
                reason="idle_threshold_exceeded",
                success=False,
                error=str(e),
            )
            self._termination_history.append(event)

            logger.error(f"[LambdaIdleDaemon] Failed to terminate {node.node_id}: {e}")
            return False

    async def _call_termination_api(self, node: LambdaNodeState) -> bool:
        """Call Lambda Labs API to terminate the instance.

        Returns:
            True if API call was successful
        """
        if not node.instance_id:
            logger.warning(
                f"[LambdaIdleDaemon] No instance_id for node {node.node_id}, "
                f"attempting SSH shutdown"
            )
            return await self._ssh_shutdown(node)

        try:
            # Try Lambda Labs API
            from app.distributed.lambda_client import get_lambda_client

            client = get_lambda_client()
            if client:
                result = await client.terminate_instance(node.instance_id)
                return result.get("success", False)

        except ImportError:
            logger.debug("Lambda client not available, falling back to SSH shutdown")
        except Exception as e:
            logger.warning(f"Lambda API termination failed: {e}, trying SSH")

        # Fallback to SSH shutdown
        return await self._ssh_shutdown(node)

    async def _ssh_shutdown(self, node: LambdaNodeState) -> bool:
        """Shutdown node via SSH command."""
        try:
            import subprocess

            # Get SSH config for this host
            ssh_config = await self._get_ssh_config(node.node_id)
            if not ssh_config:
                logger.warning(f"No SSH config for {node.node_id}")
                return False

            # Build SSH command for graceful shutdown
            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
            ]

            if ssh_config.get("key_file"):
                ssh_cmd.extend(["-i", ssh_config["key_file"]])

            if ssh_config.get("port"):
                ssh_cmd.extend(["-p", str(ssh_config["port"])])

            user = ssh_config.get("user", "root")
            host = ssh_config.get("host", node.host)
            ssh_cmd.append(f"{user}@{host}")

            # Graceful shutdown command
            ssh_cmd.append("sudo shutdown -h now")

            # Execute with timeout
            result = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                await asyncio.wait_for(result.wait(), timeout=30)
                return result.returncode == 0
            except asyncio.TimeoutError:
                result.kill()
                # Shutdown command may disconnect before returning
                return True

        except Exception as e:
            logger.warning(f"SSH shutdown failed for {node.node_id}: {e}")
            return False

    async def _get_ssh_config(self, node_id: str) -> dict[str, Any] | None:
        """Get SSH configuration for a node from distributed_hosts.yaml."""
        try:
            import yaml

            config_path = Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml"
            if not config_path.exists():
                return None

            with open(config_path) as f:
                config = yaml.safe_load(f)

            hosts = config.get("hosts", {})
            host_config = hosts.get(node_id, {})

            if not host_config:
                return None

            return {
                "host": host_config.get("host"),
                "port": host_config.get("port", 22),
                "user": host_config.get("user", "root"),
                "key_file": host_config.get("key_file"),
            }

        except Exception as e:
            logger.debug(f"Failed to get SSH config: {e}")
            return None

    def _emit_shutdown_event(self, node: LambdaNodeState, status: str) -> None:
        """Emit event for node shutdown."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.publish_sync(
                "lambda_node_shutdown",
                {
                    "node_id": node.node_id,
                    "host": node.host,
                    "status": status,
                    "reason": "idle",
                    "idle_duration_seconds": time.time() - node.idle_since if node.idle_since else 0,
                    "hourly_cost": node.hourly_cost,
                    "timestamp": time.time(),
                },
                source="lambda_idle_daemon",
            )
        except Exception as e:
            logger.debug(f"Could not emit shutdown event: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get daemon statistics."""
        tracked_nodes = len(self._node_states)
        idle_nodes = sum(
            1 for n in self._node_states.values()
            if n.idle_since > 0
        )
        terminating_nodes = sum(
            1 for n in self._node_states.values()
            if n.is_terminating
        )

        return {
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "nodes_terminated": self._stats.nodes_terminated,
            "nodes_skipped": self._stats.nodes_skipped,
            "termination_failures": self._stats.termination_failures,
            "estimated_savings_usd": round(self._stats.estimated_savings, 2),
            "last_check_time": self._stats.last_check_time,
            "last_termination_time": self._stats.last_termination_time,
            "tracked_nodes": tracked_nodes,
            "idle_nodes": idle_nodes,
            "terminating_nodes": terminating_nodes,
            "errors_count": self._errors_count,
        }

    def get_termination_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent termination history."""
        events = self._termination_history[-limit:]
        return [
            {
                "node_id": e.node_id,
                "host": e.host,
                "timestamp": e.timestamp,
                "idle_duration_seconds": e.idle_duration_seconds,
                "reason": e.reason,
                "success": e.success,
                "cost_saved": e.cost_saved_estimate,
                "error": e.error,
            }
            for e in events
        ]

    # CoordinatorProtocol methods
    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self._running,
            "status": self._coordinator_status.value,
            "stats": self.get_stats(),
        }

    def get_status(self) -> CoordinatorStatus:
        """Get coordinator status."""
        return self._coordinator_status


# =============================================================================
# Module-level singleton
# =============================================================================

_daemon_instance: LambdaIdleDaemon | None = None


def get_lambda_idle_daemon() -> LambdaIdleDaemon:
    """Get or create the Lambda idle daemon singleton."""
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = LambdaIdleDaemon()
    return _daemon_instance


def reset_lambda_idle_daemon() -> None:
    """Reset the singleton (for testing)."""
    global _daemon_instance
    _daemon_instance = None


__all__ = [
    "LambdaIdleDaemon",
    "LambdaIdleConfig",
    "LambdaNodeState",
    "TerminationEvent",
    "DaemonStats",
    "get_lambda_idle_daemon",
    "reset_lambda_idle_daemon",
]
