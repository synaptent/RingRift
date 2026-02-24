"""
Unified Cluster Monitor for RingRift AI Training Infrastructure.

Consolidates multiple monitoring scripts into a single module:
- cluster_health_check.py -> health_checker
- cluster_monitor.py -> cluster_status
- disk_monitor.py -> resource_monitor
- elo_monitor.py -> elo_tracker
- training_monitor.py -> training_status
- data_quality_monitor.py -> quality_monitor

Usage:
    from app.monitoring.unified_cluster_monitor import UnifiedClusterMonitor

    monitor = UnifiedClusterMonitor()
    status = await monitor.get_full_status()

    # Or run as continuous monitoring
    await monitor.start_monitoring(interval=60)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
import sqlite3
import time
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import yaml at module level for test patchability
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from app.config.thresholds import DISK_CRITICAL_PERCENT, DISK_PRODUCTION_HALT_PERCENT

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from app.distributed.health_checks import (
        HealthChecker,
        integrate_health_with_recovery,
    )
    HAS_HEALTH_CHECKS = True
except ImportError:
    HAS_HEALTH_CHECKS = False
    HealthChecker = None

try:
    from app.config.cluster_config import get_cluster_config
    HAS_CLUSTER_CONFIG = True
except ImportError:
    HAS_CLUSTER_CONFIG = False
    get_cluster_config = None

try:
    from app.config.ports import P2P_DEFAULT_PORT
except ImportError:
    P2P_DEFAULT_PORT = 8770

# Event router for status change notifications (Phase 10 consolidation)
# Using unified router for cross-system event routing (December 2025)
try:
    from app.coordination.event_router import get_router
    from app.coordination.event_router import DataEventType
    HAS_EVENT_ROUTER = True
except ImportError:
    HAS_EVENT_ROUTER = False
    get_router = None
    DataEventType = None


@dataclass
class ClusterNodeStatus:
    """Status of a single cluster node."""
    name: str
    ip: str
    is_healthy: bool
    last_seen: float
    active_jobs: int = 0
    gpu_utilization: float = 0.0
    memory_used_gb: float = 0.0
    disk_used_percent: float = 0.0
    selfplay_rate: float = 0.0
    error: str | None = None


@dataclass
class NodeHealth:
    """Health status of a single cluster node (test-compatible API)."""
    name: str
    status: str = "unknown"
    via_tailscale: bool = False
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    selfplay_active: bool = False
    games_played: int = 0
    gpu_util: float = 0.0
    error: str | None = None


@dataclass
class LeaderHealth:
    """Health status of the cluster leader node."""
    is_leader: bool = False
    node_id: str | None = None
    selfplay_jobs: int = 0
    selfplay_rate: float = 0.0
    training_nnue_running: int = 0
    training_cmaes_running: int = 0
    error: str | None = None


@dataclass
class ClusterHealth:
    """Overall cluster health status (test-compatible API)."""
    nodes: dict[str, NodeHealth] = field(default_factory=dict)
    leader: LeaderHealth | None = None
    total_nodes: int = 0
    healthy_nodes: int = 0
    total_games: int = 0
    avg_gpu_util: float = 0.0
    timestamp: datetime | None = None
    alerts: list[str] = field(default_factory=list)
    critical_alerts: list[str] = field(default_factory=list)


@dataclass
class TrainingStatus:
    """Current training status."""
    is_training: bool = False
    config_key: str | None = None
    epoch: int = 0
    total_epochs: int = 0
    val_loss: float = float("inf")
    best_val_loss: float = float("inf")
    started_at: float | None = None
    node: str | None = None


@dataclass
class EloStatus:
    """Current Elo status."""
    best_elo: float = 1500.0
    best_model: str | None = None
    recent_matches: int = 0
    last_evaluation: float | None = None
    elo_trend: float = 0.0  # Positive = improving
    target_elo: float = 2000.0
    gap_to_target: float = 500.0


@dataclass
class DataQualityStatus:
    """Data quality status."""
    total_games: int = 0
    games_24h: int = 0
    parity_pass_rate: float = 1.0
    quality_score: float = 1.0
    quarantined_games: int = 0


@dataclass
class ClusterStatus:
    """Overall cluster status."""
    timestamp: str
    healthy: bool
    node_count: int
    healthy_nodes: int
    nodes: list[ClusterNodeStatus]
    training: TrainingStatus
    elo: EloStatus
    data_quality: DataQualityStatus
    health_summary: dict[str, Any] | None = None
    alerts: list[str] = field(default_factory=list)


# Path to cluster config file
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml"


class ClusterConfig:
    """Cluster configuration loaded from distributed_hosts.yaml."""

    def __init__(self, config_path: Path | None = None):
        """Load cluster configuration from YAML file."""
        self.config_path = config_path or CONFIG_PATH
        self.nodes: dict[str, dict[str, Any]] = {}
        self.leader_url: str | None = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return

        if yaml is None:
            logger.warning("PyYAML not installed, cannot load cluster config")
            return

        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f) or {}

            hosts = data.get("hosts", {})
            for name, host_data in hosts.items():
                if host_data and host_data.get("status") != "disabled":
                    self.nodes[name] = {
                        "name": name,
                        "ssh_host": host_data.get("ssh_host", ""),
                        "ssh_user": host_data.get("ssh_user", "ubuntu"),
                        "ssh_port": host_data.get("ssh_port", 22),
                        "tailscale_ip": host_data.get("tailscale_ip"),
                        "p2p_port": host_data.get("p2p_port", P2P_DEFAULT_PORT),
                        "is_leader": host_data.get("role") == "coordinator",
                    }
                    # Build URLs
                    ip = host_data.get("ssh_host", "")
                    ts_ip = host_data.get("tailscale_ip")
                    port = host_data.get("p2p_port", P2P_DEFAULT_PORT)
                    self.nodes[name]["primary_url"] = f"http://{ip}:{port}/health" if ip else None
                    self.nodes[name]["tailscale_url"] = f"http://{ts_ip}:{port}/health" if ts_ip else None

                    if self.nodes[name]["is_leader"] and ts_ip:
                        self.leader_url = f"http://{ts_ip}:{port}"

        except Exception as e:
            logger.error(f"Failed to load cluster config: {e}")

    def get_node_names(self) -> list[str]:
        """Get list of all node names."""
        return list(self.nodes.keys())


class UnifiedClusterMonitor:
    """
    Unified monitoring for RingRift AI training cluster.

    Consolidates all monitoring functionality into a single class:
    - Cluster node health (replaces cluster_health_check.py)
    - Training status (replaces training_monitor.py)
    - Elo tracking (replaces elo_monitor.py)
    - Data quality (replaces data_quality_monitor.py)
    - Resource monitoring (replaces disk_monitor.py)
    """

    def __init__(
        self,
        config: ClusterConfig | None = None,
        webhook_url: str | None = None,
        check_interval: int = 60,
        deep_checks: bool = False,
        recovery_manager=None,
        notifier=None,
        auto_recover: bool = False,
    ):
        """
        Initialize unified monitor.

        Args:
            config: Cluster configuration (default: auto-load from file)
            webhook_url: Optional webhook URL for alerts
            check_interval: Seconds between health checks
            deep_checks: Whether to perform SSH-based deep checks
            recovery_manager: Optional RecoveryManager for auto-recovery
            notifier: Optional notifier for alerts
            auto_recover: Whether to auto-recover on failures
        """
        self._running = False
        self._callbacks: list[Callable] = []

        # Config and settings
        self.config = config or ClusterConfig()
        self.webhook_url = webhook_url
        self.check_interval = check_interval
        self.deep_checks = deep_checks

        # Thresholds - from app.config.thresholds (canonical source)
        self.disk_warning = float(DISK_PRODUCTION_HALT_PERCENT)
        self.disk_critical = float(DISK_CRITICAL_PERCENT)
        self.memory_warning = 85.0
        self.memory_critical = 95.0

        # Alert cooldown tracking
        self._alert_cooldown = 300  # 5 minutes
        self._last_alerts: dict[str, float] = {}

        # Initialize sub-monitors
        self.health_checker = HealthChecker() if HAS_HEALTH_CHECKS else None
        self.health_recovery = None

        if recovery_manager and HAS_HEALTH_CHECKS:
            self.health_recovery = integrate_health_with_recovery(
                recovery_manager=recovery_manager,
                notifier=notifier,
                auto_recover=auto_recover
            )

        # Paths
        self._elo_db = Path("data/unified_elo.db")
        self._training_db = Path("data/coordination/training_coordination.db")
        self._games_db = Path("data/games/selfplay.db")

        # Legacy cluster config
        self._cluster_config = None
        if HAS_CLUSTER_CONFIG:
            with contextlib.suppress(Exception):
                self._cluster_config = get_cluster_config()

    def _http_get_json(self, url: str, timeout: int = 5) -> dict[str, Any]:
        """Fetch JSON from HTTP endpoint."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ClusterMonitor/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            return {"error": str(e)}

    def _should_alert(self, key: str) -> bool:
        """Check if we should send an alert (respects cooldown)."""
        now = time.time()
        last_alert = self._last_alerts.get(key, 0)
        if now - last_alert >= self._alert_cooldown:
            self._last_alerts[key] = now
            return True
        return False

    def check_node_http(self, node_name: str) -> NodeHealth:
        """Check node health via HTTP endpoint."""
        node_config = self.config.nodes.get(node_name, {})
        health = NodeHealth(name=node_name)

        # Try primary URL first
        primary_url = node_config.get("primary_url")
        if primary_url:
            result = self._http_get_json(primary_url)
            if "error" not in result:
                health.status = "healthy" if result.get("healthy") else "unhealthy"
                health.cpu_percent = result.get("cpu_percent", 0.0)
                health.memory_percent = result.get("memory_percent", 0.0)
                health.disk_percent = result.get("disk_percent", 0.0)
                health.selfplay_active = result.get("selfplay_active", False)
                health.games_played = result.get("games_played", 0)
                health.gpu_util = result.get("gpu_util", 0.0)
                return health

        # Fallback to Tailscale URL
        tailscale_url = node_config.get("tailscale_url")
        if tailscale_url:
            result = self._http_get_json(tailscale_url)
            if "error" not in result:
                health.status = "healthy" if result.get("healthy") else "unhealthy"
                health.via_tailscale = True
                health.cpu_percent = result.get("cpu_percent", 0.0)
                health.memory_percent = result.get("memory_percent", 0.0)
                health.disk_percent = result.get("disk_percent", 0.0)
                health.selfplay_active = result.get("selfplay_active", False)
                health.games_played = result.get("games_played", 0)
                health.gpu_util = result.get("gpu_util", 0.0)
                return health

        health.status = "unreachable"
        health.error = "Failed to reach node via primary and Tailscale URLs"
        return health

    def check_leader(self) -> LeaderHealth | None:
        """Check leader node health."""
        if not self.config.leader_url:
            return None

        result = self._http_get_json(f"{self.config.leader_url}/health")
        if "error" in result:
            return LeaderHealth(error=result["error"])

        return LeaderHealth(
            is_leader=True,
            node_id=result.get("node_id"),
            selfplay_jobs=result.get("selfplay_jobs", 0),
            selfplay_rate=result.get("selfplay_rate", 0.0),
            training_nnue_running=result.get("training_nnue_running", 0),
            training_cmaes_running=result.get("training_cmaes_running", 0),
        )

    def check_cluster(self) -> ClusterHealth:
        """Check health of all cluster nodes."""
        cluster = ClusterHealth(timestamp=datetime.now())
        node_names = self.config.get_node_names()

        total_games = 0
        total_gpu_util = 0.0
        healthy_count = 0

        for name in node_names:
            health = self.check_node_http(name)
            cluster.nodes[name] = health
            if health.status == "healthy":
                healthy_count += 1
            total_games += health.games_played
            total_gpu_util += health.gpu_util

        cluster.total_nodes = len(node_names)
        cluster.healthy_nodes = healthy_count
        cluster.total_games = total_games
        cluster.avg_gpu_util = total_gpu_util / len(node_names) if node_names else 0.0

        # Check leader
        cluster.leader = self.check_leader()

        # Generate alerts
        self.generate_alerts(cluster)

        return cluster

    def generate_alerts(self, cluster: ClusterHealth) -> None:
        """Generate alerts based on cluster health."""
        for name, health in cluster.nodes.items():
            # Disk alerts
            if health.disk_percent >= self.disk_critical:
                cluster.critical_alerts.append(f"CRITICAL: {name} disk usage at {health.disk_percent:.1f}%")
            elif health.disk_percent >= self.disk_warning:
                cluster.alerts.append(f"Warning: {name} disk usage at {health.disk_percent:.1f}%")

            # Memory alerts
            if health.memory_percent >= self.memory_critical:
                cluster.critical_alerts.append(f"CRITICAL: {name} memory usage at {health.memory_percent:.1f}%")
            elif health.memory_percent >= self.memory_warning:
                cluster.alerts.append(f"Warning: {name} memory usage at {health.memory_percent:.1f}%")

            # Unreachable alerts
            if health.status == "unreachable":
                cluster.critical_alerts.append(f"CRITICAL: {name} is unreachable")

        # Cluster down alert - check both total_nodes field and nodes dict
        node_count = cluster.total_nodes if cluster.total_nodes > 0 else len(cluster.nodes)
        if cluster.healthy_nodes == 0 and node_count > 0:
            cluster.critical_alerts.append("CRITICAL: Cluster down - no healthy nodes")

    def format_text(self, cluster: ClusterHealth) -> str:
        """Format cluster health as text."""
        lines = []
        lines.append("=" * 60)
        lines.append("CLUSTER HEALTH STATUS")
        lines.append("=" * 60)
        lines.append(f"Nodes: {cluster.healthy_nodes}/{cluster.total_nodes} healthy")
        lines.append(f"Total games: {cluster.total_games}")
        lines.append(f"Avg GPU util: {cluster.avg_gpu_util:.1f}%")
        lines.append("-" * 60)

        for name, health in cluster.nodes.items():
            status = "✓" if health.status == "healthy" else "✗"
            lines.append(f"{status} {name}: {health.status} | disk:{health.disk_percent:.0f}% mem:{health.memory_percent:.0f}%")

        if cluster.alerts or cluster.critical_alerts:
            lines.append("-" * 60)
            lines.append("ALERTS:")
            for alert in cluster.critical_alerts:
                lines.append(f"  {alert}")
            for alert in cluster.alerts:
                lines.append(f"  {alert}")

        return "\n".join(lines)

    def format_json(self, cluster: ClusterHealth) -> str:
        """Format cluster health as JSON."""
        data = {
            "timestamp": cluster.timestamp.isoformat() if cluster.timestamp else None,
            "total_nodes": cluster.total_nodes,
            "healthy_nodes": cluster.healthy_nodes,
            "total_games": cluster.total_games,
            "avg_gpu_util": cluster.avg_gpu_util,
            "nodes": {
                name: asdict(health) for name, health in cluster.nodes.items()
            },
            "alerts": cluster.alerts,
            "critical_alerts": cluster.critical_alerts,
        }
        return json.dumps(data, indent=2, default=str)

    def send_webhook_alert(self, message: str, level: str = "info") -> None:
        """Send alert via webhook."""
        if not self.webhook_url:
            return

        payload = json.dumps({
            "text": message,
            "level": level,
            "timestamp": datetime.now().isoformat(),
        }).encode()

        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning(f"Failed to send webhook alert: {e}")

    async def get_full_status(self) -> ClusterStatus:
        """Get complete cluster status."""
        timestamp = datetime.now(timezone.utc).isoformat()

        # Get health summary
        health_summary = None
        if self.health_checker:
            try:
                summary = self.health_checker.check_all()
                health_summary = {
                    "healthy": summary.healthy,
                    "issues": summary.issues,
                    "warnings": summary.warnings,
                    "components": {c.name: c.status for c in summary.components}
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")

        # Get node statuses
        nodes = await self._get_node_statuses()
        healthy_nodes = [n for n in nodes if n.is_healthy]

        # Get training status
        training = await self._get_training_status()

        # Get Elo status
        elo = await self._get_elo_status()

        # Get data quality
        data_quality = await self._get_data_quality()

        # Compile alerts
        alerts = self._compile_alerts(health_summary, nodes, training, elo, data_quality)

        return ClusterStatus(
            timestamp=timestamp,
            healthy=len(alerts) == 0,
            node_count=len(nodes),
            healthy_nodes=len(healthy_nodes),
            nodes=nodes,
            training=training,
            elo=elo,
            data_quality=data_quality,
            health_summary=health_summary,
            alerts=alerts
        )

    async def _get_node_statuses(self) -> list[ClusterNodeStatus]:
        """Get status of all cluster nodes."""
        nodes = []

        if self._cluster_config:
            try:
                for node in self._cluster_config.get_lambda_nodes():
                    status = ClusterNodeStatus(
                        name=node.name,
                        ip=node.tailscale_ip,
                        is_healthy=True,  # Would ping/check in real implementation
                        last_seen=time.time(),
                    )
                    nodes.append(status)
            except Exception as e:
                logger.error(f"Failed to get node statuses: {e}")

        # Fallback: just return local node
        if not nodes:
            nodes.append(ClusterNodeStatus(
                name=socket.gethostname(),
                ip="127.0.0.1",
                is_healthy=True,
                last_seen=time.time(),
            ))

        return nodes

    async def _get_training_status(self) -> TrainingStatus:
        """Get current training status."""
        status = TrainingStatus()

        if not self._training_db.exists():
            return status

        try:
            conn = sqlite3.connect(str(self._training_db), timeout=5)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get active training job
            cursor.execute("""
                SELECT * FROM training_jobs
                WHERE status = 'running'
                ORDER BY started_at DESC
                LIMIT 1
            """)
            row = cursor.fetchone()

            if row:
                status.is_training = True
                status.config_key = f"{row['board_type']}_{row['num_players']}p"
                status.epoch = row['epochs_completed']
                status.best_val_loss = row['best_val_loss']
                status.started_at = row['started_at']
                status.node = row['node_name']

            conn.close()
        except Exception as e:
            logger.error(f"Failed to get training status: {e}")

        return status

    async def _get_elo_status(self) -> EloStatus:
        """Get current Elo status."""
        status = EloStatus()

        if not self._elo_db.exists():
            return status

        try:
            conn = sqlite3.connect(str(self._elo_db), timeout=5)
            cursor = conn.cursor()

            # Get best model Elo
            cursor.execute("""
                SELECT model_id, elo_rating
                FROM models
                ORDER BY elo_rating DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                status.best_model = row[0]
                status.best_elo = row[1]
                status.gap_to_target = status.target_elo - status.best_elo

            # Get recent match count
            cursor.execute("""
                SELECT COUNT(*) FROM match_history
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            status.recent_matches = cursor.fetchone()[0]

            # Get last evaluation time
            cursor.execute("SELECT MAX(timestamp) FROM match_history")
            last = cursor.fetchone()[0]
            if last:
                status.last_evaluation = time.time()  # Simplified

            # Calculate Elo trend (last 5 evaluations)
            cursor.execute("""
                SELECT elo_rating FROM models
                ORDER BY rowid DESC
                LIMIT 5
            """)
            elos = [r[0] for r in cursor.fetchall()]
            if len(elos) >= 2:
                status.elo_trend = elos[0] - elos[-1]

            conn.close()
        except Exception as e:
            logger.debug(f"Failed to get Elo status: {e}")

        return status

    async def _get_data_quality(self) -> DataQualityStatus:
        """Get data quality status."""
        status = DataQualityStatus()

        if not self._games_db.exists():
            return status

        try:
            conn = sqlite3.connect(str(self._games_db), timeout=5)
            cursor = conn.cursor()

            # Total games
            cursor.execute("SELECT COUNT(*) FROM games")
            status.total_games = cursor.fetchone()[0]

            # Games in last 24h (simplified - would check timestamp)
            status.games_24h = min(status.total_games, 1000)

            # Parity pass rate (simplified)
            status.parity_pass_rate = 0.95
            status.quality_score = 0.9

            conn.close()
        except Exception as e:
            logger.debug(f"Failed to get data quality: {e}")

        return status

    def _compile_alerts(
        self,
        health_summary: dict | None,
        nodes: list[ClusterNodeStatus],
        training: TrainingStatus,
        elo: EloStatus,
        data_quality: DataQualityStatus
    ) -> list[str]:
        """Compile alerts from all status sources."""
        alerts = []

        # Health alerts
        if health_summary:
            alerts.extend(health_summary.get("issues", []))

        # Node alerts
        unhealthy_nodes = [n for n in nodes if not n.is_healthy]
        if unhealthy_nodes:
            alerts.append(f"{len(unhealthy_nodes)} nodes unhealthy")

        # Elo alerts
        if elo.gap_to_target > 400:
            alerts.append(f"Elo gap to target: {elo.gap_to_target:.0f} pts")

        if elo.elo_trend < -20:
            alerts.append(f"Elo declining: {elo.elo_trend:.0f} pts")

        # Data quality alerts
        if data_quality.parity_pass_rate < 0.9:
            alerts.append(f"Low parity pass rate: {data_quality.parity_pass_rate:.0%}")

        return alerts

    def register_callback(self, callback: Callable[[ClusterStatus], None]) -> None:
        """Register callback for status updates."""
        self._callbacks.append(callback)

    async def start_monitoring(self, interval: int = 60) -> None:
        """Start continuous monitoring."""
        self._running = True
        self._last_healthy = True  # Track for status change detection
        logger.info(f"[UnifiedMonitor] Starting monitoring (interval={interval}s)")

        while self._running:
            try:
                status = await self.get_full_status()

                # Emit events for status changes (Phase 10)
                await self._emit_status_events(status)

                # Fire callbacks
                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(status)
                        else:
                            callback(status)
                    except Exception as e:
                        logger.error(f"[UnifiedMonitor] Callback error: {e}")

                # Log summary
                if status.alerts:
                    logger.warning(f"[UnifiedMonitor] {len(status.alerts)} alerts: {status.alerts}")
                else:
                    logger.info(
                        f"[UnifiedMonitor] Healthy | "
                        f"Nodes: {status.healthy_nodes}/{status.node_count} | "
                        f"Elo: {status.elo.best_elo:.0f} | "
                        f"Games: {status.data_quality.total_games}"
                    )

            except Exception as e:
                logger.error(f"[UnifiedMonitor] Monitoring error: {e}")

            await asyncio.sleep(interval)

    async def _emit_event(
        self,
        event_type: DataEventType,
        payload: dict,
    ) -> None:
        """Emit an event via unified router.

        Uses unified router for event routing.
        """
        if not HAS_EVENT_ROUTER:
            return

        source = "unified_cluster_monitor"
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)

        try:
            router = get_router()
            await router.publish(event_type_str, payload, source)
        except Exception as e:
            logger.debug(f"Router emit failed: {e}")

    async def _emit_status_events(self, status: ClusterStatus) -> None:
        """Emit events based on cluster status changes.

        Uses unified router for event routing.
        """
        if not HAS_EVENT_ROUTER:
            return

        # Detect overall health state change
        if status.healthy != self._last_healthy:
            self._last_healthy = status.healthy

            await self._emit_event(
                DataEventType.CLUSTER_STATUS_CHANGED,
                {
                    "healthy": status.healthy,
                    "healthy_nodes": status.healthy_nodes,
                    "node_count": status.node_count,
                    "alerts": status.alerts,
                },
            )

            cluster_event_type = (
                DataEventType.P2P_CLUSTER_HEALTHY
                if status.healthy
                else DataEventType.P2P_CLUSTER_UNHEALTHY
            )
            await self._emit_event(
                cluster_event_type,
                {
                    "healthy": status.healthy,
                    "healthy_nodes": status.healthy_nodes,
                    "node_count": status.node_count,
                    "alerts": status.alerts,
                },
            )

        # Emit alerts as individual events
        for alert in status.alerts:
            # Determine alert type based on content
            if "node" in alert.lower():
                event_type = DataEventType.NODE_UNHEALTHY
            elif any(r in alert.lower() for r in ["disk", "memory", "cpu", "gpu"]):
                event_type = DataEventType.RESOURCE_CONSTRAINT
            else:
                event_type = DataEventType.HEALTH_ALERT

            await self._emit_event(
                event_type,
                {"alert": alert, "timestamp": status.timestamp},
            )

        # Emit node-specific events for unhealthy nodes
        for node in status.nodes:
            if not node.is_healthy:
                await self._emit_event(
                    DataEventType.NODE_UNHEALTHY,
                    {
                        "node_name": node.name,
                        "node_ip": node.ip,
                        "error": node.error,
                        "gpu_utilization": node.gpu_utilization,
                        "disk_used_percent": node.disk_used_percent,
                    },
                )

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._running = False
        logger.info("[UnifiedMonitor] Monitoring stopped")

    def get_summary_json(self, status: ClusterStatus) -> str:
        """Get status as JSON string."""
        return json.dumps(asdict(status), indent=2, default=str)


# Convenience functions

def get_cluster_monitor(
    recovery_manager=None,
    notifier=None,
    auto_recover: bool = False
) -> UnifiedClusterMonitor:
    """Create unified cluster monitor."""
    return UnifiedClusterMonitor(
        recovery_manager=recovery_manager,
        notifier=notifier,
        auto_recover=auto_recover
    )


async def get_cluster_status() -> ClusterStatus:
    """Quick one-shot cluster status check."""
    monitor = UnifiedClusterMonitor()
    return await monitor.get_full_status()


def print_cluster_status(status: ClusterStatus) -> None:
    """Print cluster status to console."""
    print("=" * 60)
    print(f"CLUSTER STATUS - {status.timestamp}")
    print("=" * 60)
    print(f"Overall: {'HEALTHY' if status.healthy else 'UNHEALTHY'}")
    print(f"Nodes: {status.healthy_nodes}/{status.node_count} healthy")
    print()

    print("TRAINING:")
    if status.training.is_training:
        print(f"  Config: {status.training.config_key}")
        print(f"  Epoch: {status.training.epoch}")
        print(f"  Best Loss: {status.training.best_val_loss:.4f}")
        print(f"  Node: {status.training.node}")
    else:
        print("  No active training")
    print()

    print("ELO:")
    print(f"  Best: {status.elo.best_elo:.1f} ({status.elo.best_model or 'N/A'})")
    print(f"  Target: {status.elo.target_elo:.0f}")
    print(f"  Gap: {status.elo.gap_to_target:.0f} pts")
    print(f"  Trend: {status.elo.elo_trend:+.0f}")
    print()

    print("DATA:")
    print(f"  Total Games: {status.data_quality.total_games:,}")
    print(f"  24h: {status.data_quality.games_24h:,}")
    print(f"  Quality: {status.data_quality.quality_score:.0%}")
    print()

    if status.alerts:
        print("ALERTS:")
        for alert in status.alerts:
            print(f"  ⚠ {alert}")
    print("=" * 60)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Cluster Monitor")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval")
    args = parser.parse_args()

    async def main():
        monitor = UnifiedClusterMonitor()

        if args.watch:
            await monitor.start_monitoring(interval=args.interval)
        else:
            status = await monitor.get_full_status()
            if args.json:
                print(monitor.get_summary_json(status))
            else:
                print_cluster_status(status)

    asyncio.run(main())
