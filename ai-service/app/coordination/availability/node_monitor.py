"""Multi-layer node health monitoring.

This module implements proactive health monitoring for cluster nodes with
multi-layer checks:
1. P2P heartbeat (fastest, 15s timeout)
2. SSH connectivity (30s timeout)
3. GPU health check (for GPU nodes)
4. Provider API status

Created: Dec 28, 2025
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from app.coordination.base_daemon import BaseDaemon, DaemonConfig
from app.config.ports import P2P_DEFAULT_PORT

if TYPE_CHECKING:
    from app.config.cluster_config import ClusterNode

logger = logging.getLogger(__name__)


class HealthCheckLayer(str, Enum):
    """Health check layer types."""
    P2P = "p2p"
    SSH = "ssh"
    GPU = "gpu"
    PROVIDER_API = "provider_api"
    ALL = "all"


@dataclass
class NodeHealthResult:
    """Result of a node health check."""
    node_id: str
    layer: HealthCheckLayer
    healthy: bool
    latency_ms: float
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for event emission."""
        return {
            "node_id": self.node_id,
            "layer": self.layer.value,
            "healthy": self.healthy,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass(kw_only=True)
class NodeMonitorConfig(DaemonConfig):
    """Configuration for NodeMonitor."""
    check_interval_seconds: int = 30  # Overrides DaemonConfig default
    p2p_timeout_seconds: float = 15.0
    ssh_timeout_seconds: float = 30.0
    gpu_check_enabled: bool = True
    provider_check_enabled: bool = True
    consecutive_failures_before_unhealthy: int = 3
    consecutive_failures_before_recovery: int = 5
    p2p_port: int = P2P_DEFAULT_PORT


class NodeMonitor(BaseDaemon):
    """Multi-layer node health monitoring daemon.

    Monitors all cluster nodes at regular intervals using multiple health
    check layers. Emits NODE_UNHEALTHY events when nodes fail consistently.

    Example:
        monitor = NodeMonitor()
        await monitor.start()
    """

    def __init__(
        self,
        config: NodeMonitorConfig | None = None,
        nodes: list[ClusterNode] | None = None,
    ):
        super().__init__(config)
        self._nodes: list[ClusterNode] = nodes or []
        self._failure_counts: dict[str, int] = {}
        self._last_healthy: dict[str, datetime] = {}
        self._health_history: dict[str, list[NodeHealthResult]] = {}

    def _get_default_config(self) -> NodeMonitorConfig:
        """Return default configuration."""
        return NodeMonitorConfig()

    def _get_daemon_name(self) -> str:
        """Return daemon name for logging."""
        return "NodeMonitor"

    def set_nodes(self, nodes: list[ClusterNode]) -> None:
        """Update the list of nodes to monitor."""
        self._nodes = nodes
        # Initialize failure counts for new nodes
        for node in nodes:
            if node.name not in self._failure_counts:
                self._failure_counts[node.name] = 0
                self._health_history[node.name] = []

    async def _run_cycle(self) -> None:
        """Run one monitoring cycle."""
        if not self._nodes:
            # Try to load nodes from config
            await self._load_nodes_from_config()

        if not self._nodes:
            logger.debug("NodeMonitor: No nodes configured")
            return

        # Check all nodes concurrently
        tasks = [self._check_node_health(node) for node in self._nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for node, result in zip(self._nodes, results):
            if isinstance(result, Exception):
                logger.error(f"Error checking node {node.name}: {result}")
                result = NodeHealthResult(
                    node_id=node.name,
                    layer=HealthCheckLayer.P2P,
                    healthy=False,
                    latency_ms=0.0,
                    error=str(result),
                )

            await self._process_health_result(node, result)

    async def _load_nodes_from_config(self) -> None:
        """Load nodes from cluster configuration."""
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes_dict = get_cluster_nodes()
            self._nodes = list(nodes_dict.values())
            logger.info(f"NodeMonitor: Loaded {len(self._nodes)} nodes from config")
        except Exception as e:
            logger.warning(f"Failed to load nodes from config: {e}")

    async def _check_node_health(self, node: ClusterNode) -> NodeHealthResult:
        """Check health of a node through all layers."""
        start_time = time.time()

        # Layer 1: P2P heartbeat (fastest check)
        p2p_result = await self._check_p2p(node)
        if not p2p_result.healthy:
            return p2p_result

        # Layer 2: SSH connectivity
        ssh_result = await self._check_ssh(node)
        if not ssh_result.healthy:
            return ssh_result

        # Layer 3: GPU health (for GPU nodes)
        if self.config.gpu_check_enabled and getattr(node, "is_gpu_node", False):
            gpu_result = await self._check_gpu(node)
            if not gpu_result.healthy:
                return gpu_result

        # Layer 4: Provider API status
        if self.config.provider_check_enabled:
            provider_result = await self._check_provider_status(node)
            if not provider_result.healthy:
                return provider_result

        # All checks passed
        latency = (time.time() - start_time) * 1000
        return NodeHealthResult(
            node_id=node.name,
            layer=HealthCheckLayer.ALL,
            healthy=True,
            latency_ms=latency,
        )

    async def _check_p2p(self, node: ClusterNode) -> NodeHealthResult:
        """Check P2P heartbeat connectivity."""
        import aiohttp

        start_time = time.time()
        ip = getattr(node, "best_ip", None) or getattr(node, "tailscale_ip", None)

        if not ip:
            return NodeHealthResult(
                node_id=node.name,
                layer=HealthCheckLayer.P2P,
                healthy=False,
                latency_ms=0.0,
                error="No IP address configured",
            )

        url = f"http://{ip}:{self.config.p2p_port}/status"

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.p2p_timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    latency = (time.time() - start_time) * 1000
                    if resp.status == 200:
                        data = await resp.json()
                        return NodeHealthResult(
                            node_id=node.name,
                            layer=HealthCheckLayer.P2P,
                            healthy=True,
                            latency_ms=latency,
                            details={"status": data.get("status", "unknown")},
                        )
                    else:
                        return NodeHealthResult(
                            node_id=node.name,
                            layer=HealthCheckLayer.P2P,
                            healthy=False,
                            latency_ms=latency,
                            error=f"HTTP {resp.status}",
                        )
        except asyncio.TimeoutError:
            return NodeHealthResult(
                node_id=node.name,
                layer=HealthCheckLayer.P2P,
                healthy=False,
                latency_ms=self.config.p2p_timeout_seconds * 1000,
                error="P2P timeout",
            )
        except Exception as e:
            return NodeHealthResult(
                node_id=node.name,
                layer=HealthCheckLayer.P2P,
                healthy=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def _check_ssh(self, node: ClusterNode) -> NodeHealthResult:
        """Check SSH connectivity."""
        start_time = time.time()
        ip = getattr(node, "best_ip", None) or getattr(node, "tailscale_ip", None)
        user = getattr(node, "ssh_user", "root")
        port = getattr(node, "ssh_port", 22)

        if not ip:
            return NodeHealthResult(
                node_id=node.name,
                layer=HealthCheckLayer.SSH,
                healthy=False,
                latency_ms=0.0,
                error="No IP address configured",
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o", f"ConnectTimeout={int(self.config.ssh_timeout_seconds)}",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-p", str(port),
                f"{user}@{ip}",
                "echo ok",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                await asyncio.wait_for(
                    proc.wait(),
                    timeout=self.config.ssh_timeout_seconds + 5
                )
            except asyncio.TimeoutError:
                proc.kill()
                return NodeHealthResult(
                    node_id=node.name,
                    layer=HealthCheckLayer.SSH,
                    healthy=False,
                    latency_ms=self.config.ssh_timeout_seconds * 1000,
                    error="SSH timeout",
                )

            latency = (time.time() - start_time) * 1000

            if proc.returncode == 0:
                return NodeHealthResult(
                    node_id=node.name,
                    layer=HealthCheckLayer.SSH,
                    healthy=True,
                    latency_ms=latency,
                )
            else:
                stderr = ""
                if proc.stderr:
                    stderr_data = await proc.stderr.read()
                    stderr = stderr_data.decode().strip()[:100]
                return NodeHealthResult(
                    node_id=node.name,
                    layer=HealthCheckLayer.SSH,
                    healthy=False,
                    latency_ms=latency,
                    error=f"SSH exit code {proc.returncode}: {stderr}",
                )
        except Exception as e:
            return NodeHealthResult(
                node_id=node.name,
                layer=HealthCheckLayer.SSH,
                healthy=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def _check_gpu(self, node: ClusterNode) -> NodeHealthResult:
        """Check GPU health via SSH."""
        start_time = time.time()
        ip = getattr(node, "best_ip", None) or getattr(node, "tailscale_ip", None)
        user = getattr(node, "ssh_user", "root")
        port = getattr(node, "ssh_port", 22)

        if not ip:
            return NodeHealthResult(
                node_id=node.name,
                layer=HealthCheckLayer.GPU,
                healthy=False,
                latency_ms=0.0,
                error="No IP address",
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-p", str(port),
                f"{user}@{ip}",
                "nvidia-smi --query-gpu=gpu_name,memory.total,utilization.gpu --format=csv,noheader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=30
                )
            except asyncio.TimeoutError:
                proc.kill()
                return NodeHealthResult(
                    node_id=node.name,
                    layer=HealthCheckLayer.GPU,
                    healthy=False,
                    latency_ms=30000,
                    error="GPU check timeout",
                )

            latency = (time.time() - start_time) * 1000

            if proc.returncode == 0:
                gpu_info = stdout.decode().strip()
                return NodeHealthResult(
                    node_id=node.name,
                    layer=HealthCheckLayer.GPU,
                    healthy=True,
                    latency_ms=latency,
                    details={"gpu_info": gpu_info},
                )
            else:
                return NodeHealthResult(
                    node_id=node.name,
                    layer=HealthCheckLayer.GPU,
                    healthy=False,
                    latency_ms=latency,
                    error="nvidia-smi failed",
                )
        except Exception as e:
            return NodeHealthResult(
                node_id=node.name,
                layer=HealthCheckLayer.GPU,
                healthy=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def _check_provider_status(self, node: ClusterNode) -> NodeHealthResult:
        """Check instance status via provider API."""
        start_time = time.time()
        provider = getattr(node, "provider", None)

        if not provider:
            # Skip provider check if no provider configured
            return NodeHealthResult(
                node_id=node.name,
                layer=HealthCheckLayer.PROVIDER_API,
                healthy=True,
                latency_ms=0.0,
                details={"skipped": "no provider configured"},
            )

        try:
            from app.coordination.providers.registry import get_provider

            provider_client = get_provider(provider)
            if not provider_client:
                return NodeHealthResult(
                    node_id=node.name,
                    layer=HealthCheckLayer.PROVIDER_API,
                    healthy=True,
                    latency_ms=0.0,
                    details={"skipped": f"provider {provider} not available"},
                )

            # Get instance status
            instance_id = getattr(node, "instance_id", None)
            if instance_id:
                from app.coordination.providers.base import InstanceStatus

                status = await provider_client.get_instance_status(instance_id)
                latency = (time.time() - start_time) * 1000

                healthy = status == InstanceStatus.RUNNING
                return NodeHealthResult(
                    node_id=node.name,
                    layer=HealthCheckLayer.PROVIDER_API,
                    healthy=healthy,
                    latency_ms=latency,
                    details={"status": status.value},
                    error=None if healthy else f"Instance status: {status.value}",
                )
            else:
                return NodeHealthResult(
                    node_id=node.name,
                    layer=HealthCheckLayer.PROVIDER_API,
                    healthy=True,
                    latency_ms=0.0,
                    details={"skipped": "no instance_id"},
                )

        except Exception as e:
            return NodeHealthResult(
                node_id=node.name,
                layer=HealthCheckLayer.PROVIDER_API,
                healthy=True,  # Don't fail on provider API errors
                latency_ms=(time.time() - start_time) * 1000,
                details={"warning": str(e)},
            )

    async def _process_health_result(
        self,
        node: ClusterNode,
        result: NodeHealthResult,
    ) -> None:
        """Process a health check result and emit events if needed."""
        node_id = node.name

        # Track history
        if node_id not in self._health_history:
            self._health_history[node_id] = []
        self._health_history[node_id].append(result)
        # Keep last 100 results
        self._health_history[node_id] = self._health_history[node_id][-100:]

        if result.healthy:
            # Reset failure count on success
            if self._failure_counts.get(node_id, 0) > 0:
                logger.info(
                    f"Node {node_id} recovered (was at {self._failure_counts[node_id]} failures)"
                )
                await self._emit_node_recovered(node, result)

            self._failure_counts[node_id] = 0
            self._last_healthy[node_id] = datetime.now()
        else:
            # Increment failure count
            self._failure_counts[node_id] = self._failure_counts.get(node_id, 0) + 1
            failures = self._failure_counts[node_id]

            logger.warning(
                f"Node {node_id} unhealthy ({failures} consecutive failures): "
                f"layer={result.layer.value}, error={result.error}"
            )

            # Emit unhealthy event after threshold
            if failures >= self.config.consecutive_failures_before_unhealthy:
                await self._emit_node_unhealthy(node, result)

            # Emit recovery trigger after higher threshold
            if failures >= self.config.consecutive_failures_before_recovery:
                await self._emit_recovery_needed(node, result)

    async def _emit_node_unhealthy(
        self,
        node: ClusterNode,
        result: NodeHealthResult,
    ) -> None:
        """Emit NODE_UNHEALTHY event."""
        try:
            from app.distributed.data_events import DataEventType

            await self._safe_emit_event(
                DataEventType.NODE_UNHEALTHY.value,
                {
                    "node_id": node.name,
                    "health_result": result.to_dict(),
                    "consecutive_failures": self._failure_counts.get(node.name, 0),
                    "last_healthy": (
                        self._last_healthy[node.name].isoformat()
                        if node.name in self._last_healthy
                        else None
                    ),
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit NODE_UNHEALTHY: {e}")

    async def _emit_node_recovered(
        self,
        node: ClusterNode,
        result: NodeHealthResult,
    ) -> None:
        """Emit NODE_RECOVERED event."""
        try:
            from app.distributed.data_events import DataEventType

            await self._safe_emit_event(
                DataEventType.NODE_RECOVERED.value,
                {
                    "node_id": node.name,
                    "health_result": result.to_dict(),
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit NODE_RECOVERED: {e}")

    async def _emit_recovery_needed(
        self,
        node: ClusterNode,
        result: NodeHealthResult,
    ) -> None:
        """Emit event requesting recovery action."""
        try:
            from app.distributed.data_events import DataEventType

            await self._safe_emit_event(
                DataEventType.RECOVERY_INITIATED.value,
                {
                    "node_id": node.name,
                    "health_result": result.to_dict(),
                    "consecutive_failures": self._failure_counts.get(node.name, 0),
                    "reason": "consecutive_health_failures",
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit RECOVERY_INITIATED: {e}")

    async def _safe_emit_event(self, event_type: str, payload: dict) -> None:
        """Safely emit an event via the event router."""
        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus:
                from app.distributed.data_events import DataEvent

                event = DataEvent(
                    event_type=event_type,
                    payload=payload,
                    source="NodeMonitor",
                )
                bus.publish(event)
        except Exception as e:
            logger.debug(f"Event emission failed: {e}")

    def get_node_status(self, node_id: str) -> dict:
        """Get current status for a specific node."""
        history = self._health_history.get(node_id, [])
        last_result = history[-1] if history else None

        return {
            "node_id": node_id,
            "healthy": last_result.healthy if last_result else None,
            "consecutive_failures": self._failure_counts.get(node_id, 0),
            "last_healthy": (
                self._last_healthy[node_id].isoformat()
                if node_id in self._last_healthy
                else None
            ),
            "last_check": last_result.to_dict() if last_result else None,
        }

    def get_all_node_statuses(self) -> dict[str, dict]:
        """Get status for all monitored nodes."""
        return {node.name: self.get_node_status(node.name) for node in self._nodes}

    def health_check(self) -> dict:
        """Return health status for DaemonManager integration."""
        unhealthy_count = sum(
            1 for node in self._nodes
            if self._failure_counts.get(node.name, 0) >= self.config.consecutive_failures_before_unhealthy
        )

        return {
            "healthy": unhealthy_count == 0,
            "message": f"Monitoring {len(self._nodes)} nodes, {unhealthy_count} unhealthy",
            "details": {
                "nodes_monitored": len(self._nodes),
                "unhealthy_count": unhealthy_count,
                "cycles_completed": self._cycles_completed,
            },
        }


# Singleton instance
_node_monitor: NodeMonitor | None = None


def get_node_monitor() -> NodeMonitor:
    """Get the singleton NodeMonitor instance."""
    global _node_monitor
    if _node_monitor is None:
        _node_monitor = NodeMonitor()
    return _node_monitor


def reset_node_monitor() -> None:
    """Reset the singleton (for testing)."""
    global _node_monitor
    _node_monitor = None
