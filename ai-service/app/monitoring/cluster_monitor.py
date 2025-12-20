"""Cluster health monitoring implementation.

This module provides concrete HealthMonitor implementations for cluster health
monitoring, using the centralized thresholds and base classes.

Usage:
    from app.monitoring.cluster_monitor import (
        ClusterHealthMonitor,
        NodeHealthMonitor,
        create_cluster_monitor,
    )

    # Create composite monitor for the cluster
    monitor = create_cluster_monitor(nodes=["node1", "node2", "node3"])
    result = monitor.run_check()

    if not result.is_healthy:
        for alert in result.alerts:
            print(f"ALERT: {alert}")
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.monitoring.base import (
    Alert,
    CompositeMonitor,
    HealthMonitor,
    HealthStatus,
    MonitoringResult,
)
from app.config.thresholds import (
    AlertLevel,
    get_threshold,
    should_alert,
)

# Event emission for cluster health changes
try:
    import asyncio

    from app.distributed.data_events import DataEvent, DataEventType, get_event_bus
    HAS_CLUSTER_EVENTS = True
except ImportError:
    HAS_CLUSTER_EVENTS = False
    DataEvent = None
    DataEventType = None
    get_event_bus = None

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    address: str
    last_heartbeat: datetime | None = None
    is_leader: bool = False
    gpu_utilization: float = 0.0
    disk_usage_pct: float = 0.0
    memory_usage_pct: float = 0.0
    selfplay_games_per_hour: float = 0.0
    status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


class DiskHealthMonitor(HealthMonitor):
    """Monitor disk space on local or remote paths."""

    def __init__(self, path: str = "/", name: str | None = None):
        super().__init__(name or f"DiskMonitor({path})")
        self.path = path

    def check_health(self) -> MonitoringResult:
        """Check disk space against thresholds."""
        alerts: list[Alert] = []
        metrics: dict[str, Any] = {}

        try:
            usage = shutil.disk_usage(self.path)
            total_gb = usage.total / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            pct_used = (usage.used / usage.total) * 100

            metrics = {
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": free_gb,
                "percent_used": pct_used,
            }

            # Check thresholds
            if should_alert("disk", pct_used, "fatal"):
                alerts.append(Alert(
                    level=AlertLevel.FATAL,
                    category="disk",
                    message=f"Disk critically full at {pct_used:.1f}% - stopping writes",
                    metric_name="disk_percent_used",
                    metric_value=pct_used,
                    threshold=get_threshold("disk", "fatal"),
                ))
                status = HealthStatus.UNHEALTHY
            elif should_alert("disk", pct_used, "critical"):
                alerts.append(Alert(
                    level=AlertLevel.CRITICAL,
                    category="disk",
                    message=f"Disk high at {pct_used:.1f}% - pausing operations",
                    metric_name="disk_percent_used",
                    metric_value=pct_used,
                    threshold=get_threshold("disk", "critical"),
                ))
                status = HealthStatus.DEGRADED
            elif should_alert("disk", pct_used, "warning"):
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    category="disk",
                    message=f"Disk usage at {pct_used:.1f}%",
                    metric_name="disk_percent_used",
                    metric_value=pct_used,
                    threshold=get_threshold("disk", "warning"),
                ))
                status = HealthStatus.HEALTHY  # Warning doesn't degrade status
            else:
                status = HealthStatus.HEALTHY

        except OSError as e:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                category="disk",
                message=f"Failed to check disk: {e}",
            ))
            status = HealthStatus.UNKNOWN

        return MonitoringResult(
            status=status,
            metrics=metrics,
            alerts=alerts,
        )


class MemoryHealthMonitor(HealthMonitor):
    """Monitor system memory usage."""

    def __init__(self, name: str = "MemoryMonitor"):
        super().__init__(name)

    def check_health(self) -> MonitoringResult:
        """Check memory usage against thresholds."""
        alerts: list[Alert] = []
        metrics: dict[str, Any] = {}

        try:
            # Try to use psutil if available
            import psutil
            mem = psutil.virtual_memory()
            pct_used = mem.percent
            total_gb = mem.total / (1024 ** 3)
            available_gb = mem.available / (1024 ** 3)

            metrics = {
                "total_gb": total_gb,
                "available_gb": available_gb,
                "percent_used": pct_used,
            }

            if should_alert("memory", pct_used, "critical"):
                alerts.append(Alert(
                    level=AlertLevel.CRITICAL,
                    category="memory",
                    message=f"Memory critical at {pct_used:.1f}%",
                    metric_name="memory_percent_used",
                    metric_value=pct_used,
                    threshold=get_threshold("memory", "critical"),
                ))
                status = HealthStatus.DEGRADED
            elif should_alert("memory", pct_used, "warning"):
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    category="memory",
                    message=f"Memory high at {pct_used:.1f}%",
                    metric_name="memory_percent_used",
                    metric_value=pct_used,
                    threshold=get_threshold("memory", "warning"),
                ))
                status = HealthStatus.HEALTHY
            else:
                status = HealthStatus.HEALTHY

        except ImportError:
            # psutil not available - try /proc/meminfo on Linux
            try:
                with open("/proc/meminfo") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip().split()[0]
                            meminfo[key] = int(value)

                    total = meminfo.get("MemTotal", 0)
                    available = meminfo.get("MemAvailable", 0)
                    if total > 0:
                        pct_used = (1 - available / total) * 100
                        metrics = {
                            "total_gb": total / (1024 ** 2),
                            "available_gb": available / (1024 ** 2),
                            "percent_used": pct_used,
                        }
                        status = HealthStatus.HEALTHY
                    else:
                        status = HealthStatus.UNKNOWN
            except (OSError, KeyError):
                status = HealthStatus.UNKNOWN
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    category="memory",
                    message="Unable to check memory (psutil not available)",
                ))

        return MonitoringResult(
            status=status,
            metrics=metrics,
            alerts=alerts,
        )


class GPUHealthMonitor(HealthMonitor):
    """Monitor GPU utilization and memory."""

    def __init__(self, device_id: int = 0, name: str | None = None):
        super().__init__(name or f"GPUMonitor({device_id})")
        self.device_id = device_id

    def check_health(self) -> MonitoringResult:
        """Check GPU health using nvidia-smi or pynvml."""
        alerts: list[Alert] = []
        metrics: dict[str, Any] = {}
        status = HealthStatus.HEALTHY

        try:
            # Try pynvml first
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

            # Get GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            mem_util = util.memory

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_gb = mem_info.used / (1024 ** 3)
            mem_total_gb = mem_info.total / (1024 ** 3)
            mem_pct = (mem_info.used / mem_info.total) * 100

            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temp = None

            metrics = {
                "gpu_utilization": gpu_util,
                "memory_utilization": mem_util,
                "memory_used_gb": mem_used_gb,
                "memory_total_gb": mem_total_gb,
                "memory_percent": mem_pct,
            }
            if temp is not None:
                metrics["temperature_c"] = temp

            pynvml.nvmlShutdown()

            # Check GPU memory thresholds
            if should_alert("gpu_memory", mem_pct, "critical"):
                alerts.append(Alert(
                    level=AlertLevel.CRITICAL,
                    category="gpu_memory",
                    message=f"GPU memory critical at {mem_pct:.1f}%",
                    metric_name="gpu_memory_percent",
                    metric_value=mem_pct,
                    threshold=get_threshold("gpu_memory", "critical"),
                ))
                status = HealthStatus.DEGRADED
            elif should_alert("gpu_memory", mem_pct, "warning"):
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    category="gpu_memory",
                    message=f"GPU memory high at {mem_pct:.1f}%",
                    metric_name="gpu_memory_percent",
                    metric_value=mem_pct,
                    threshold=get_threshold("gpu_memory", "warning"),
                ))

            # Check for idle GPU (potential issue during selfplay)
            idle_threshold = get_threshold("gpu_utilization", "idle", 5)
            if gpu_util <= idle_threshold:
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    category="gpu_utilization",
                    message=f"GPU appears idle at {gpu_util}%",
                    metric_name="gpu_utilization",
                    metric_value=gpu_util,
                    threshold=idle_threshold,
                ))

        except ImportError:
            # pynvml not available
            alerts.append(Alert(
                level=AlertLevel.INFO,
                category="gpu",
                message="pynvml not available - GPU monitoring disabled",
            ))
            status = HealthStatus.UNKNOWN
        except Exception as e:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                category="gpu",
                message=f"GPU monitoring failed: {e}",
            ))
            status = HealthStatus.UNKNOWN

        return MonitoringResult(
            status=status,
            metrics=metrics,
            alerts=alerts,
        )


class NodeHealthMonitor(HealthMonitor):
    """Monitor health of a single cluster node."""

    def __init__(
        self,
        node_id: str,
        address: str,
        check_gpu: bool = True,
        check_disk: bool = True,
        check_memory: bool = True,
    ):
        super().__init__(f"Node({node_id})")
        self.node_id = node_id
        self.address = address
        self.check_gpu = check_gpu
        self.check_disk = check_disk
        self.check_memory = check_memory

        # Sub-monitors
        self._monitors: list[HealthMonitor] = []
        if check_disk:
            self._monitors.append(DiskHealthMonitor("/"))
        if check_memory:
            self._monitors.append(MemoryHealthMonitor())
        if check_gpu:
            self._monitors.append(GPUHealthMonitor())

    def check_health(self) -> MonitoringResult:
        """Check health of this node."""
        all_metrics: dict[str, Any] = {"node_id": self.node_id}
        all_alerts: list[Alert] = []
        worst_status = HealthStatus.HEALTHY

        for monitor in self._monitors:
            try:
                result = monitor.run_check()
                # Aggregate metrics
                for key, value in result.metrics.items():
                    all_metrics[f"{monitor.name}.{key}"] = value
                # Collect alerts with node info
                for alert in result.alerts:
                    alert.node = self.node_id
                    all_alerts.append(alert)
                # Track worst status
                if result.status == HealthStatus.UNHEALTHY:
                    worst_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and worst_status != HealthStatus.UNHEALTHY:
                    worst_status = HealthStatus.DEGRADED
                elif result.status == HealthStatus.UNKNOWN and worst_status == HealthStatus.HEALTHY:
                    worst_status = HealthStatus.UNKNOWN
            except Exception as e:
                all_alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    category="monitor_error",
                    message=f"Monitor {monitor.name} failed: {e}",
                    node=self.node_id,
                ))

        return MonitoringResult(
            status=worst_status,
            metrics=all_metrics,
            alerts=all_alerts,
        )


class ClusterHealthMonitor(CompositeMonitor):
    """Monitor health of the entire cluster.

    Combines node-level monitors and checks cluster-wide thresholds
    like minimum nodes online. Emits P2P_CLUSTER_HEALTHY/UNHEALTHY events
    when health status changes.
    """

    def __init__(self, min_nodes: int | None = None):
        super().__init__("ClusterHealthMonitor")
        self.min_nodes = min_nodes or get_threshold("cluster", "min_nodes_online", 5)
        self.node_timeout_seconds = get_threshold("cluster", "node_timeout_seconds", 30)
        self._previous_status: HealthStatus | None = None

    def add_node(
        self,
        node_id: str,
        address: str,
        check_gpu: bool = True,
    ) -> None:
        """Add a node to monitor."""
        node_monitor = NodeHealthMonitor(
            node_id=node_id,
            address=address,
            check_gpu=check_gpu,
        )
        self.add_monitor(node_monitor)

    def check_health(self) -> MonitoringResult:
        """Check cluster health including node count thresholds."""
        # Get base result from composite monitor
        result = super().check_health()

        # Check minimum nodes threshold
        nodes_online = len([
            m for m in self._monitors
            if isinstance(m, NodeHealthMonitor)
        ])

        result.metrics["nodes_online"] = nodes_online
        result.metrics["min_nodes_required"] = self.min_nodes

        if nodes_online < self.min_nodes:
            result.alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                category="cluster",
                message=f"Only {nodes_online}/{self.min_nodes} nodes online",
                metric_name="nodes_online",
                metric_value=float(nodes_online),
                threshold=float(self.min_nodes),
            ))
            result.status = HealthStatus.DEGRADED

        # Emit events on status change for event-driven coordination
        self._emit_health_event(result)

        return result

    def _emit_health_event(self, result: MonitoringResult) -> None:
        """Emit cluster health events when status changes.

        This enables other components (training, promotion, selfplay) to
        react to cluster health changes via the EventBus.
        """
        if not HAS_CLUSTER_EVENTS:
            return

        current_status = result.status
        previous_status = self._previous_status

        # Only emit on status change (or first check)
        if current_status == previous_status:
            return

        self._previous_status = current_status

        try:
            # Determine event type based on health transition
            is_healthy = current_status == HealthStatus.HEALTHY
            event_type = (
                DataEventType.P2P_CLUSTER_HEALTHY if is_healthy
                else DataEventType.P2P_CLUSTER_UNHEALTHY
            )

            payload = {
                "status": current_status.value if hasattr(current_status, 'value') else str(current_status),
                "previous_status": previous_status.value if previous_status and hasattr(previous_status, 'value') else str(previous_status),
                "nodes_online": result.metrics.get("nodes_online", 0),
                "min_nodes_required": result.metrics.get("min_nodes_required", self.min_nodes),
                "alerts": [str(a) for a in result.alerts[:5]],  # First 5 alerts
            }

            # Schedule event emission
            try:
                asyncio.get_running_loop()
                event_bus = get_event_bus()
                asyncio.ensure_future(event_bus.publish(DataEvent(
                    event_type=event_type,
                    payload=payload,
                    source="cluster_monitor",
                )))
                logger.info(f"Emitted {event_type.value}: cluster is now {current_status}")
            except RuntimeError:
                # No running loop - skip event emission in sync context
                logger.debug("No event loop available for cluster health event")

        except Exception as e:
            logger.warning(f"Failed to emit cluster health event: {e}")


def create_cluster_monitor(
    nodes: list[dict[str, Any]] | None = None,
    min_nodes: int | None = None,
) -> ClusterHealthMonitor:
    """Create a ClusterHealthMonitor with the specified nodes.

    Args:
        nodes: List of node configurations with keys:
            - node_id: Unique node identifier
            - address: Node address (IP or hostname)
            - check_gpu: Whether to monitor GPU (default True)
        min_nodes: Minimum nodes required for healthy cluster

    Returns:
        Configured ClusterHealthMonitor
    """
    monitor = ClusterHealthMonitor(min_nodes=min_nodes)

    if nodes:
        for node_config in nodes:
            monitor.add_node(
                node_id=node_config["node_id"],
                address=node_config["address"],
                check_gpu=node_config.get("check_gpu", True),
            )

    return monitor


# Convenience function for quick local check
def check_local_health() -> MonitoringResult:
    """Quick health check of the local machine."""
    monitor = NodeHealthMonitor(
        node_id="local",
        address="localhost",
    )
    return monitor.run_check()


__all__ = [
    "ClusterHealthMonitor",
    "DiskHealthMonitor",
    "GPUHealthMonitor",
    "MemoryHealthMonitor",
    "NodeHealthMonitor",
    "NodeInfo",
    "check_local_health",
    "create_cluster_monitor",
]
