"""Cluster Status Monitor for RingRift Coordination Infrastructure.

December 2025: Relocated from app.distributed.cluster_monitor to
app.coordination.cluster_status_monitor for better organization.

This module provides SSH-based real-time monitoring of the entire cluster:
- Game counts across all nodes
- Training process status
- Disk usage monitoring
- Data sync status
- Node health and connectivity

Usage:
    # Programmatic API
    from app.coordination.cluster_status_monitor import ClusterMonitor

    monitor = ClusterMonitor()
    status = monitor.get_cluster_status()
    print(f"Total games: {status.total_games}")
    print(f"Active nodes: {status.active_nodes}")

    # CLI with real-time watch mode
    python -m app.coordination.cluster_status_monitor --watch --interval 10

    # Single snapshot
    python -m app.coordination.cluster_status_monitor

Note:
    A backward-compatible shim exists at the old location
    (app.distributed.cluster_monitor) with a deprecation warning.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Dec 29, 2025: Use canonical SSH module for all SSH operations
from app.core.ssh import SSHClient, SSHConfig, get_ssh_client
from app.coordination.event_emission_helpers import safe_emit_event

logger = logging.getLogger(__name__)


@dataclass
class NodeStatus:
    """Status information for a single cluster node."""

    host_name: str
    reachable: bool = False
    response_time_ms: float = 0.0

    # Game counts
    game_counts: dict[str, int] = field(default_factory=dict)
    total_games: int = 0

    # Training status
    training_active: bool = False
    training_processes: list[dict[str, Any]] = field(default_factory=list)

    # Resource usage
    disk_usage_percent: float = 0.0
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0

    # Memory and CPU (if available)
    memory_usage_percent: float = 0.0
    cpu_percent: float = 0.0

    # GPU metrics
    gpu_utilization_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0

    # Data sync status
    last_sync_time: datetime | None = None
    sync_lag_seconds: float = 0.0
    pending_files: int = 0

    # Host info
    role: str = "unknown"
    gpu: str = ""
    status: str = "unknown"

    # Error tracking
    error: str | None = None
    last_check: datetime = field(default_factory=datetime.now)


@dataclass
class NodeStallState:
    """State tracking for stall detection on a single node.

    December 2025: Harvested from deprecated robust_cluster_monitor.py.
    Tracks game count across consecutive checks to detect stuck jobs.
    """

    last_game_count: int = 0
    stall_count: int = 0  # Consecutive checks with unchanged game count
    last_check_time: float = 0.0
    alerted: bool = False  # Whether we've already alerted for this stall


@dataclass
class ClusterStatus:
    """Aggregated status for the entire cluster."""

    timestamp: datetime = field(default_factory=datetime.now)

    # Node counts
    total_nodes: int = 0
    active_nodes: int = 0
    unreachable_nodes: int = 0

    # Game counts
    total_games: int = 0
    games_by_config: dict[str, int] = field(default_factory=dict)

    # Training status
    nodes_training: int = 0
    total_training_processes: int = 0

    # Resource aggregates
    avg_disk_usage: float = 0.0
    total_disk_free_gb: float = 0.0
    total_disk_capacity_gb: float = 0.0

    # Individual node statuses
    nodes: dict[str, NodeStatus] = field(default_factory=dict)

    # Query metadata
    query_duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def healthy_nodes(self) -> int:
        """Alias for active_nodes for backward compatibility."""
        return self.active_nodes


class ClusterMonitor:
    """Monitor and query cluster-wide status.

    This class integrates with:
    - RemoteGameDiscovery for game counts
    - SSH for process and resource monitoring
    - distributed_hosts.yaml for cluster configuration
    """

    def __init__(
        self,
        hosts_config_path: Path | str | None = None,
        ssh_timeout: int = 15,
        parallel: bool = True,
    ):
        """Initialize cluster monitor.

        Args:
            hosts_config_path: Path to distributed_hosts.yaml
            ssh_timeout: Timeout for SSH operations (seconds)
            parallel: Query nodes in parallel
        """
        self.ssh_timeout = ssh_timeout
        self.parallel = parallel

        # Auto-detect hosts config
        if hosts_config_path is None:
            candidates = [
                Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml",
                Path.cwd() / "config" / "distributed_hosts.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    hosts_config_path = candidate
                    break

        self.hosts_config_path = Path(hosts_config_path) if hosts_config_path else None
        self._hosts: dict[str, dict[str, Any]] = {}
        self._load_hosts()

        # Initialize RemoteGameDiscovery
        try:
            from app.utils.game_discovery import RemoteGameDiscovery
            self.game_discovery = RemoteGameDiscovery(
                hosts_config_path=self.hosts_config_path,
                cache_ttl=0,  # Disable caching for real-time monitoring
            )
        except ImportError:
            logger.warning("RemoteGameDiscovery not available")
            self.game_discovery = None

        # December 2025: Stall detection state (harvested from robust_cluster_monitor)
        self._stall_states: dict[str, NodeStallState] = {}
        self._stall_threshold: int = 3  # Consecutive checks before alerting
        self._running = False

    def _load_hosts(self):
        """Load hosts configuration from cluster_config helpers.

        Updated Dec 2025 to use consolidated cluster_config helpers instead of
        inline YAML parsing, while preserving backward compatibility.
        """
        # Try cluster_config helpers first (Dec 2025 consolidation)
        try:
            from app.config.cluster_config import load_cluster_config
            config = load_cluster_config(self.hosts_config_path)
            self._hosts = config.hosts_raw
            logger.info(f"Loaded {len(self._hosts)} hosts via cluster_config")
            return
        except ImportError:
            logger.debug("cluster_config not available, falling back to direct YAML")
        except Exception as e:
            logger.debug(f"cluster_config load failed, falling back: {e}")

        # Fallback: direct YAML parsing for backward compatibility
        if not self.hosts_config_path or not self.hosts_config_path.exists():
            logger.warning(f"Hosts config not found: {self.hosts_config_path}")
            return

        if not HAS_YAML:
            logger.error("PyYAML not installed, cannot load hosts config")
            return

        try:
            with open(self.hosts_config_path) as f:
                config = yaml.safe_load(f) or {}
            self._hosts = config.get("hosts", {})
            logger.info(f"Loaded {len(self._hosts)} hosts from config")
        except Exception as e:
            logger.error(f"Error loading hosts config: {e}")

    async def _async_run_ssh_command(
        self,
        host_name: str,
        command: str,
        timeout: float | None = None,
    ) -> tuple[int, str, str]:
        """Run SSH command asynchronously using asyncio.create_subprocess_exec.

        This is used for async contexts (e.g., run_forever) to avoid blocking
        the event loop. For sync contexts, use subprocess.run directly.

        Args:
            host_name: Host to connect to
            command: Command to run on remote host
            timeout: Optional timeout (default: self.ssh_timeout)

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        host_info = self._hosts.get(host_name, {})
        ssh_host = host_info.get("tailscale_ip") or host_info.get("ssh_host")
        ssh_user = host_info.get("ssh_user", "ubuntu")
        ssh_key = host_info.get("ssh_key", "~/.ssh/id_cluster")
        ssh_port = host_info.get("ssh_port", 22)

        if timeout is None:
            timeout = self.ssh_timeout

        cmd = [
            "ssh",
            "-i", os.path.expanduser(ssh_key),
            "-p", str(ssh_port),
            "-o", f"ConnectTimeout={min(int(timeout), 10)}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "LogLevel=ERROR",
            f"{ssh_user}@{ssh_host}",
            command,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
                return (
                    proc.returncode or 0,
                    stdout.decode("utf-8", errors="replace"),
                    stderr.decode("utf-8", errors="replace"),
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return -1, "", "Timeout"

        except Exception as e:
            logger.debug(f"Async SSH command failed for {host_name}: {e}")
            return -1, "", str(e)

    async def _async_check_host_connectivity(self, host_name: str) -> bool:
        """Async version of connectivity check."""
        returncode, _, _ = await self._async_run_ssh_command(
            host_name,
            "true",
            timeout=min(self.ssh_timeout, 5),
        )
        return returncode == 0

    async def get_node_status_async(
        self,
        host_name: str,
        include_game_counts: bool = True,
        include_training_status: bool = True,
        include_disk_usage: bool = True,
        include_sync_status: bool = False,
    ) -> NodeStatus:
        """Get status for a single node asynchronously.

        This is the async version of get_node_status, used in run_forever.
        """
        # Run sync version in thread pool to avoid code duplication
        # while still being non-blocking
        return await asyncio.to_thread(
            self.get_node_status,
            host_name,
            include_game_counts=include_game_counts,
            include_training_status=include_training_status,
            include_disk_usage=include_disk_usage,
            include_sync_status=include_sync_status,
        )

    async def get_cluster_status_async(
        self,
        hosts: list[str] | None = None,
        include_game_counts: bool = True,
        include_training_status: bool = True,
        include_disk_usage: bool = True,
        include_sync_status: bool = False,
    ) -> ClusterStatus:
        """Get comprehensive cluster status asynchronously.

        This is the async version of get_cluster_status, used in run_forever.
        Uses asyncio.gather for parallel execution.
        """
        if hosts is None:
            hosts = self.get_active_hosts()

        status = ClusterStatus(total_nodes=len(hosts))

        # Run all node queries in parallel
        tasks = [
            self.get_node_status_async(
                host,
                include_game_counts=include_game_counts,
                include_training_status=include_training_status,
                include_disk_usage=include_disk_usage,
                include_sync_status=include_sync_status,
            )
            for host in hosts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for host, result in zip(hosts, results):
            if isinstance(result, Exception):
                logger.error(f"Error querying {host}: {result}")
                status.nodes[host] = NodeStatus(
                    host_name=host,
                    reachable=False,
                    error=str(result),
                )
            else:
                status.nodes[host] = result

        # Aggregate statistics (same as sync version)
        status.active_nodes = sum(1 for n in status.nodes.values() if n.reachable)
        status.unreachable_nodes = status.total_nodes - status.active_nodes

        for node in status.nodes.values():
            if node.reachable:
                status.total_games += node.total_games
                for config, count in node.game_counts.items():
                    status.games_by_config[config] = (
                        status.games_by_config.get(config, 0) + count
                    )

        status.nodes_training = sum(1 for n in status.nodes.values() if n.training_active)
        status.total_training_processes = sum(
            len(n.training_processes) for n in status.nodes.values()
        )

        return status

    def get_active_hosts(self) -> list[str]:
        """Get list of hosts marked as active/ready in config."""
        return [
            name for name, info in self._hosts.items()
            if info.get("status") in ("ready", "active")
        ]

    def get_cluster_status(
        self,
        hosts: list[str] | None = None,
        include_game_counts: bool = True,
        include_training_status: bool = True,
        include_disk_usage: bool = True,
        include_sync_status: bool = False,  # Expensive, disabled by default
    ) -> ClusterStatus:
        """Get comprehensive cluster status.

        Args:
            hosts: List of hosts to query (default: all active hosts)
            include_game_counts: Query game database counts
            include_training_status: Check for active training processes
            include_disk_usage: Query disk usage
            include_sync_status: Check data sync status (expensive)

        Returns:
            ClusterStatus with aggregated metrics
        """
        start_time = time.time()

        if hosts is None:
            hosts = self.get_active_hosts()

        status = ClusterStatus(
            total_nodes=len(hosts),
        )

        # Query all nodes
        if self.parallel:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = {
                    executor.submit(
                        self.get_node_status,
                        host,
                        include_game_counts=include_game_counts,
                        include_training_status=include_training_status,
                        include_disk_usage=include_disk_usage,
                        include_sync_status=include_sync_status,
                    ): host
                    for host in hosts
                }
                for future in concurrent.futures.as_completed(futures):
                    host = futures[future]
                    try:
                        node_status = future.result()
                        status.nodes[host] = node_status
                    except Exception as e:
                        logger.error(f"Error querying {host}: {e}")
                        status.nodes[host] = NodeStatus(
                            host_name=host,
                            reachable=False,
                            error=str(e),
                        )
        else:
            for host in hosts:
                try:
                    node_status = self.get_node_status(
                        host,
                        include_game_counts=include_game_counts,
                        include_training_status=include_training_status,
                        include_disk_usage=include_disk_usage,
                        include_sync_status=include_sync_status,
                    )
                    status.nodes[host] = node_status
                except Exception as e:
                    logger.error(f"Error querying {host}: {e}")
                    status.nodes[host] = NodeStatus(
                        host_name=host,
                        reachable=False,
                        error=str(e),
                    )

        # Aggregate statistics
        status.active_nodes = sum(1 for n in status.nodes.values() if n.reachable)
        status.unreachable_nodes = status.total_nodes - status.active_nodes

        # Aggregate game counts
        for node in status.nodes.values():
            if node.reachable:
                status.total_games += node.total_games
                for config, count in node.game_counts.items():
                    status.games_by_config[config] = (
                        status.games_by_config.get(config, 0) + count
                    )

        # Aggregate training status
        status.nodes_training = sum(1 for n in status.nodes.values() if n.training_active)
        status.total_training_processes = sum(
            len(n.training_processes) for n in status.nodes.values()
        )

        # Aggregate disk usage
        reachable_nodes = [n for n in status.nodes.values() if n.reachable]
        if reachable_nodes:
            status.avg_disk_usage = sum(
                n.disk_usage_percent for n in reachable_nodes
            ) / len(reachable_nodes)
            status.total_disk_free_gb = sum(n.disk_free_gb for n in reachable_nodes)
            status.total_disk_capacity_gb = sum(n.disk_total_gb for n in reachable_nodes)

        # Collect errors
        for node in status.nodes.values():
            if node.error:
                status.errors.append(f"{node.host_name}: {node.error}")

        status.query_duration_seconds = time.time() - start_time
        return status

    def get_node_status(
        self,
        host_name: str,
        include_game_counts: bool = True,
        include_training_status: bool = True,
        include_disk_usage: bool = True,
        include_gpu_metrics: bool = True,
        include_sync_status: bool = False,
    ) -> NodeStatus:
        """Get detailed status for a single node.

        Args:
            host_name: Name of the host in distributed_hosts.yaml
            include_game_counts: Query game database counts
            include_training_status: Check for active training processes
            include_disk_usage: Query disk usage
            include_gpu_metrics: Query GPU utilization and memory
            include_sync_status: Check data sync status

        Returns:
            NodeStatus with all requested metrics
        """
        start_time = time.time()

        host_info = self._hosts.get(host_name, {})
        node_status = NodeStatus(
            host_name=host_name,
            role=host_info.get("role", "unknown"),
            gpu=host_info.get("gpu", ""),
            status=host_info.get("status", "unknown"),
        )

        # Quick connectivity check
        if not self._check_connectivity(host_name):
            node_status.reachable = False
            node_status.error = "Host unreachable"
            return node_status

        node_status.reachable = True
        node_status.response_time_ms = (time.time() - start_time) * 1000

        # Game counts (via RemoteGameDiscovery)
        if include_game_counts and self.game_discovery:
            try:
                counts = self.game_discovery.get_remote_game_counts(
                    host_name,
                    timeout=self.ssh_timeout,
                    use_cache=False,
                )
                node_status.game_counts = counts
                node_status.total_games = sum(counts.values())
            except Exception as e:
                logger.debug(f"Error getting game counts for {host_name}: {e}")

        # Training status
        if include_training_status:
            try:
                training_info = self._check_training_status(host_name)
                node_status.training_active = training_info["active"]
                node_status.training_processes = training_info["processes"]
            except Exception as e:
                logger.debug(f"Error checking training status for {host_name}: {e}")

        # Disk usage
        if include_disk_usage:
            try:
                disk_info = self._check_disk_usage(host_name)
                node_status.disk_usage_percent = disk_info["percent"]
                node_status.disk_free_gb = disk_info["free_gb"]
                node_status.disk_total_gb = disk_info["total_gb"]
            except Exception as e:
                logger.debug(f"Error checking disk usage for {host_name}: {e}")

        # GPU metrics
        if include_gpu_metrics and node_status.gpu:
            try:
                gpu_info = self._check_gpu_metrics(host_name)
                node_status.gpu_utilization_percent = gpu_info["utilization_percent"]
                node_status.gpu_memory_used_gb = gpu_info["memory_used_gb"]
                node_status.gpu_memory_total_gb = gpu_info["memory_total_gb"]
            except Exception as e:
                logger.debug(f"Error checking GPU metrics for {host_name}: {e}")

        # Sync status
        if include_sync_status:
            try:
                sync_info = self._check_sync_status(host_name)
                node_status.last_sync_time = sync_info.get("last_sync_time")
                node_status.sync_lag_seconds = sync_info.get("lag_seconds", 0.0)
                node_status.pending_files = sync_info.get("pending_files", 0)
            except Exception as e:
                logger.debug(f"Error checking sync status for {host_name}: {e}")

        node_status.last_check = datetime.now()
        return node_status

    def _get_ssh_client(self, host_name: str) -> SSHClient | None:
        """Get or create SSHClient for a host.

        Dec 29, 2025: Uses canonical get_ssh_client from app.core.ssh.
        This provides connection pooling, circuit breakers, and metrics.
        """
        try:
            return get_ssh_client(host_name)
        except Exception as e:
            logger.debug(f"Failed to get SSH client for {host_name}: {e}")
            return None

    def _check_connectivity(self, host_name: str) -> bool:
        """Quick connectivity check using SSH.

        Dec 29, 2025: Migrated to use canonical SSHClient.
        """
        client = self._get_ssh_client(host_name)
        if not client:
            return False

        try:
            return client.is_alive()
        except Exception as e:
            logger.warning(f"Connectivity check failed for {host_name}: {e}")
            return False

    def _check_training_status(self, host_name: str) -> dict[str, Any]:
        """Check for active training processes on a node.

        Dec 29, 2025: Migrated to use canonical SSHClient.
        """
        client = self._get_ssh_client(host_name)
        if not client:
            return {"active": False, "processes": []}

        # Look for training processes (train.py, train_*.py, etc.)
        command = "ps aux | grep -E '(train\\.py|train_|training)' | grep -v grep || true"

        try:
            result = client.run(command, timeout=self.ssh_timeout)

            if not result.success:
                return {"active": False, "processes": []}

            processes = []
            for line in result.stdout.strip().split("\n"):
                if line and "python" in line.lower():
                    # Parse process info
                    parts = line.split()
                    if len(parts) >= 11:
                        processes.append({
                            "user": parts[0],
                            "pid": parts[1],
                            "cpu": parts[2],
                            "mem": parts[3],
                            "command": " ".join(parts[10:])[:100],  # Truncate
                        })

            return {
                "active": len(processes) > 0,
                "processes": processes,
            }
        except Exception as e:
            logger.debug(f"Error checking training status: {e}")
            return {"active": False, "processes": []}

    def _check_disk_usage(self, host_name: str) -> dict[str, float]:
        """Check disk usage on a node.

        Dec 29, 2025: Migrated to use canonical SSHClient.
        """
        client = self._get_ssh_client(host_name)
        if not client:
            return {"percent": 0.0, "free_gb": 0.0, "total_gb": 0.0}

        host_info = self._hosts.get(host_name, {})
        ringrift_path = host_info.get("ringrift_path", "~/ringrift/ai-service")

        # Check disk usage of the ringrift directory's filesystem
        command = f"df -BG {ringrift_path} | tail -1"

        try:
            result = client.run(command, timeout=self.ssh_timeout)

            if not result.success:
                return {"percent": 0.0, "free_gb": 0.0, "total_gb": 0.0}

            # Parse df output: Filesystem Size Used Avail Use% Mounted
            parts = result.stdout.strip().split()
            if len(parts) >= 5:
                total_str = parts[1].rstrip("G")
                used_str = parts[2].rstrip("G")
                avail_str = parts[3].rstrip("G")
                percent_str = parts[4].rstrip("%")

                try:
                    return {
                        "percent": float(percent_str),
                        "free_gb": float(avail_str),
                        "total_gb": float(total_str),
                    }
                except ValueError:
                    pass

            return {"percent": 0.0, "free_gb": 0.0, "total_gb": 0.0}
        except Exception as e:
            logger.debug(f"Error checking disk usage: {e}")
            return {"percent": 0.0, "free_gb": 0.0, "total_gb": 0.0}

    def _check_gpu_metrics(self, host_name: str) -> dict[str, float]:
        """Check GPU utilization and memory on a node.

        Uses nvidia-smi to query:
        - GPU utilization percentage
        - GPU memory used/total

        Returns:
            Dict with utilization_percent, memory_used_gb, memory_total_gb

        Dec 29, 2025: Migrated to use canonical SSHClient.
        """
        client = self._get_ssh_client(host_name)
        if not client:
            return {"utilization_percent": 0.0, "memory_used_gb": 0.0, "memory_total_gb": 0.0}

        # Query nvidia-smi for GPU metrics
        # Format: utilization.gpu [%], memory.used [MiB], memory.total [MiB]
        command = (
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits 2>/dev/null | head -1"
        )

        try:
            result = client.run(command, timeout=self.ssh_timeout)

            if not result.success or not result.stdout.strip():
                return {"utilization_percent": 0.0, "memory_used_gb": 0.0, "memory_total_gb": 0.0}

            # Parse: "45, 8192, 81920" (util%, mem_used_mib, mem_total_mib)
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 3:
                try:
                    util_percent = float(parts[0])
                    mem_used_mib = float(parts[1])
                    mem_total_mib = float(parts[2])
                    return {
                        "utilization_percent": util_percent,
                        "memory_used_gb": mem_used_mib / 1024,
                        "memory_total_gb": mem_total_mib / 1024,
                    }
                except ValueError:
                    pass

            return {"utilization_percent": 0.0, "memory_used_gb": 0.0, "memory_total_gb": 0.0}
        except Exception as e:
            logger.debug(f"Error checking GPU metrics for {host_name}: {e}")
            return {"utilization_percent": 0.0, "memory_used_gb": 0.0, "memory_total_gb": 0.0}

    def _check_sync_status(self, host_name: str) -> dict[str, Any]:
        """Check data sync status on a node.

        This checks for:
        - Last sync timestamp (from selfplay.db mtime)
        - Pending files in sync queue
        - Sync lag compared to coordinator (based on db mtime difference)

        Dec 29, 2025: Migrated to use canonical SSHClient.
        """
        client = self._get_ssh_client(host_name)
        if not client:
            return {"pending_files": 0, "lag_seconds": 0.0, "last_sync_time": None}

        host_info = self._hosts.get(host_name, {})
        ringrift_path = host_info.get("ringrift_path", "~/ringrift/ai-service")

        # Get local coordinator's selfplay.db mtime for lag calculation
        local_db_mtime = self._get_local_db_mtime()

        # Query remote node for:
        # 1. Pending sync files count
        # 2. selfplay.db mtime (for lag calculation)
        # 3. Last sync timestamp from sync state file (if exists)
        command = (
            f"cd {ringrift_path} && "
            f"echo PENDING:$(find data/sync -name '*.pending' 2>/dev/null | wc -l) && "
            f"echo DBMTIME:$(stat -c %Y data/games/selfplay.db 2>/dev/null || echo 0) && "
            f"echo SYNCSTATE:$(cat data/sync/.last_sync_time 2>/dev/null || echo 0)"
        )

        try:
            result = client.run(command, timeout=self.ssh_timeout)

            pending_files = 0
            remote_db_mtime = 0.0
            last_sync_time = None
            lag_seconds = 0.0

            if result.success:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("PENDING:"):
                        try:
                            pending_files = int(line.split(":")[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith("DBMTIME:"):
                        try:
                            remote_db_mtime = float(line.split(":")[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith("SYNCSTATE:"):
                        try:
                            sync_ts = float(line.split(":")[1].strip())
                            if sync_ts > 0:
                                last_sync_time = datetime.fromtimestamp(sync_ts)
                        except (ValueError, IndexError):
                            pass

                # Calculate lag: difference between local and remote db mtimes
                # Positive lag means remote is behind local
                if local_db_mtime > 0 and remote_db_mtime > 0:
                    lag_seconds = local_db_mtime - remote_db_mtime
                    # Cap at 0 if remote is somehow ahead (clock skew)
                    lag_seconds = max(0.0, lag_seconds)

                # If no sync state file, use db mtime as last sync time
                if last_sync_time is None and remote_db_mtime > 0:
                    last_sync_time = datetime.fromtimestamp(remote_db_mtime)

            return {
                "pending_files": pending_files,
                "lag_seconds": lag_seconds,
                "last_sync_time": last_sync_time,
            }
        except Exception as e:
            logger.debug(f"Error checking sync status: {e}")
            return {
                "pending_files": 0,
                "lag_seconds": 0.0,
                "last_sync_time": None,
            }

    def _get_local_db_mtime(self) -> float:
        """Get the mtime of the local coordinator's selfplay.db.

        Used as reference point for calculating sync lag on remote nodes.
        """
        try:
            from app.utils.paths import AI_SERVICE_ROOT
            local_db = AI_SERVICE_ROOT / "data" / "games" / "selfplay.db"
            if local_db.exists():
                return local_db.stat().st_mtime
        except Exception as e:
            logger.debug(f"Failed to get local db mtime: {e}")
        return 0.0

    def print_dashboard(self, status: ClusterStatus | None = None) -> None:
        """Print formatted dashboard output.

        Args:
            status: ClusterStatus to display (will query if not provided)
        """
        if status is None:
            status = self.get_cluster_status()

        # Print header
        print("\n" + "=" * 100)
        print(f"RINGRIFT CLUSTER MONITOR - {status.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)

        # Cluster summary
        print(f"\nCluster Summary:")
        print(f"  Nodes: {status.active_nodes}/{status.total_nodes} online")
        print(f"  Total Games: {status.total_games:,}")
        print(f"  Training: {status.nodes_training} nodes active ({status.total_training_processes} processes)")
        print(f"  Disk: {status.avg_disk_usage:.1f}% avg usage, {status.total_disk_free_gb:.0f}GB free")
        print(f"  Query Time: {status.query_duration_seconds:.2f}s")

        # Games by config
        if status.games_by_config:
            print(f"\nGame Counts by Configuration:")
            for config in sorted(status.games_by_config.keys()):
                count = status.games_by_config[config]
                print(f"  {config:20} {count:>12,}")

        # Node table
        print(f"\nNode Status:")
        print(f"{'Host':<25} {'Status':<10} {'Games':<12} {'GPU':<18} {'Training':<10} {'Disk':<15}")
        print("-" * 100)

        for host in sorted(status.nodes.keys()):
            node = status.nodes[host]

            # Status indicator
            if node.reachable:
                status_str = "ONLINE"
            else:
                status_str = "OFFLINE"

            # Games
            games_str = f"{node.total_games:,}" if node.total_games > 0 else "-"

            # GPU utilization
            if node.gpu and node.gpu_utilization_percent > 0:
                gpu_str = f"{node.gpu_utilization_percent:.0f}% ({node.gpu_memory_used_gb:.0f}/{node.gpu_memory_total_gb:.0f}GB)"
            elif node.gpu:
                gpu_str = f"0% (idle)"
            else:
                gpu_str = "-"

            # Training
            if node.training_active:
                training_str = f"YES ({len(node.training_processes)})"
            else:
                training_str = "-"

            # Disk
            if node.disk_total_gb > 0:
                disk_str = f"{node.disk_usage_percent:.1f}% ({node.disk_free_gb:.0f}GB)"
            else:
                disk_str = "-"

            print(f"{host:<25} {status_str:<10} {games_str:<12} {gpu_str:<18} {training_str:<10} {disk_str:<15}")

        # Errors
        if status.errors:
            print(f"\nErrors:")
            for error in status.errors[:10]:  # Limit to 10
                print(f"  - {error}")

        print("=" * 100 + "\n")

    def watch(self, interval: int = 10, clear_screen: bool = True) -> None:
        """Continuously monitor cluster with periodic updates.

        Args:
            interval: Seconds between updates
            clear_screen: Clear screen before each update
        """
        try:
            while True:
                if clear_screen:
                    # Clear screen (works on Unix and Windows)
                    os.system('clear' if os.name == 'posix' else 'cls')

                status = self.get_cluster_status()
                self.print_dashboard(status)

                print(f"Next update in {interval}s (Ctrl+C to exit)...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

    def _check_for_stalls(self, status: ClusterStatus) -> list[str]:
        """Check for stalled nodes (no game progress).

        December 2025: Harvested from deprecated robust_cluster_monitor.py.
        Tracks game count across consecutive checks. If game count doesn't
        change for N consecutive checks, the node is considered stalled.

        Args:
            status: Current cluster status

        Returns:
            List of stalled node names (for alerting)
        """
        now = time.time()
        stalled_nodes: list[str] = []

        for node_name, node_status in status.nodes.items():
            if not node_status.reachable:
                # Clear stall state for unreachable nodes
                if node_name in self._stall_states:
                    del self._stall_states[node_name]
                continue

            current_games = node_status.total_games

            # Initialize state if first check
            if node_name not in self._stall_states:
                self._stall_states[node_name] = NodeStallState(
                    last_game_count=current_games,
                    stall_count=0,
                    last_check_time=now,
                    alerted=False,
                )
                continue

            state = self._stall_states[node_name]

            if current_games > state.last_game_count:
                # Progress made - reset stall counter
                state.stall_count = 0
                state.last_game_count = current_games
                state.last_check_time = now
                state.alerted = False
            else:
                # No progress - increment stall counter
                state.stall_count += 1
                state.last_check_time = now

                # Check if stall threshold reached
                if state.stall_count >= self._stall_threshold and not state.alerted:
                    stalled_nodes.append(node_name)
                    state.alerted = True
                    logger.warning(
                        f"[ClusterMonitor] Node {node_name} stalled: "
                        f"game count {current_games} unchanged for "
                        f"{state.stall_count} consecutive checks"
                    )

        return stalled_nodes

    async def _emit_stall_alert(self, stalled_nodes: list[str]) -> None:
        """Emit stall detection event.

        December 2025: Integrated with event system for pipeline coordination.
        """
        if not stalled_nodes:
            return

        safe_emit_event(
            "CLUSTER_STALL_DETECTED",
            {
                "stalled_nodes": stalled_nodes,
                "stall_threshold": self._stall_threshold,
                "timestamp": time.time(),
            },
            source="cluster_status_monitor",
            context="stall_detection",
        )

    def get_stall_status(self) -> dict[str, dict[str, Any]]:
        """Get current stall detection status for all tracked nodes.

        Returns:
            Dict mapping node name to stall state info
        """
        return {
            node: {
                "game_count": state.last_game_count,
                "stall_count": state.stall_count,
                "threshold": self._stall_threshold,
                "is_stalled": state.stall_count >= self._stall_threshold,
                "alerted": state.alerted,
                "last_check": state.last_check_time,
            }
            for node, state in self._stall_states.items()
        }

    async def run_forever(self, interval: int = 30) -> None:
        """Run cluster monitoring forever (for DaemonManager compatibility).

        Collects cluster status periodically without printing.
        Emits events when cluster health changes.

        Args:
            interval: Seconds between checks

        Note:
            Uses get_cluster_status_async() to avoid blocking the event loop
            with subprocess.run calls (Dec 2025 fix).
        """
        self._running = True
        logger.info(f"[ClusterMonitor] Starting background monitoring (interval={interval}s)")

        while self._running:
            try:
                # Use async version to avoid blocking event loop
                status = await self.get_cluster_status_async()
                # Log summary instead of printing
                logger.debug(
                    f"[ClusterMonitor] {status.total_nodes} nodes, "
                    f"{status.healthy_nodes} healthy, {len(status.errors)} errors"
                )

                # December 2025: Check for stalled nodes
                stalled = self._check_for_stalls(status)
                if stalled:
                    await self._emit_stall_alert(stalled)

            except Exception as e:
                logger.warning(f"[ClusterMonitor] Status check failed: {e}")

            await asyncio.sleep(interval)

    def stop(self) -> None:
        """Stop the background monitoring loop."""
        self._running = False

    def health_check(self) -> "HealthCheckResult":
        """Check monitor health for CoordinatorProtocol compliance.

        December 2025 Phase 9: Added for daemon health monitoring.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        is_running = getattr(self, "_running", False)
        hosts_loaded = len(self._hosts) > 0

        return HealthCheckResult(
            healthy=hosts_loaded,
            status=CoordinatorStatus.RUNNING if is_running else CoordinatorStatus.READY,
            message=f"ClusterMonitor: {len(self._hosts)} hosts configured",
            details={
                "running": is_running,
                "hosts_configured": len(self._hosts),
                "hosts_config_path": str(self.hosts_config_path),
            },
        )


# =============================================================================
# HandlerBase Integration (January 2026)
# =============================================================================

try:
    from app.coordination.handler_base import HandlerBase
    from app.coordination.contracts import HealthCheckResult as HBHealthCheckResult

    HAS_HANDLER_BASE = True
except ImportError:
    HAS_HANDLER_BASE = False

if HAS_HANDLER_BASE:

    class ClusterMonitorHandler(HandlerBase):
        """HandlerBase wrapper for ClusterMonitor.

        January 2026: Added for unified daemon lifecycle management.
        Runs cluster status checks on a periodic cycle and emits events
        for stalled nodes or degraded cluster health.
        """

        # Default check interval (60 seconds for cluster-wide status)
        DEFAULT_CYCLE_INTERVAL = 60.0

        def __init__(
            self,
            cycle_interval: float = DEFAULT_CYCLE_INTERVAL,
            ssh_timeout: int = 15,
        ):
            super().__init__(
                name="cluster_monitor",
                cycle_interval=cycle_interval,
            )
            self._monitor = ClusterMonitor(ssh_timeout=ssh_timeout)
            self._last_status: ClusterStatus | None = None

        async def _run_cycle(self) -> None:
            """Run one cluster status check cycle."""
            try:
                # Get cluster status (runs synchronously, wrap in thread)
                status = await asyncio.to_thread(
                    self._monitor.get_cluster_status,
                    include_game_counts=True,
                    include_training_status=True,
                    include_disk_usage=True,
                )
                self._last_status = status

                # Check for stalls and emit events
                stalled_nodes = self._monitor._check_for_stalls(status)
                if stalled_nodes:
                    safe_emit_event(
                        "CLUSTER_STALL_DETECTED",
                        {
                            "stalled_nodes": stalled_nodes,
                            "active_nodes": status.active_nodes,
                            "total_nodes": status.total_nodes,
                        },
                    )

                # Emit health summary event
                health_ratio = status.active_nodes / max(status.total_nodes, 1)
                if health_ratio < 0.5:
                    safe_emit_event(
                        "CLUSTER_HEALTH_DEGRADED",
                        {
                            "health_ratio": health_ratio,
                            "active_nodes": status.active_nodes,
                            "total_nodes": status.total_nodes,
                        },
                    )

                logger.debug(
                    f"[ClusterMonitorHandler] Cycle complete: "
                    f"{status.active_nodes}/{status.total_nodes} nodes active"
                )

            except Exception as e:
                logger.error(f"[ClusterMonitorHandler] Cycle error: {e}")

        def _get_event_subscriptions(self) -> dict:
            """Get event subscriptions for cluster monitor."""
            return {
                "NODE_FAILED": self._on_node_failed,
                "NODE_RECOVERED": self._on_node_recovered,
            }

        async def _on_node_failed(self, event: dict) -> None:
            """Handle node failure - trigger immediate status check."""
            from app.coordination.event_router import get_event_payload
            payload = get_event_payload(event)
            node_id = payload.get("node_id", "unknown")
            logger.info(f"[ClusterMonitorHandler] Node {node_id} failed, scheduling check")
            # Schedule immediate cycle
            await self._run_cycle()

        async def _on_node_recovered(self, event: dict) -> None:
            """Handle node recovery - update status."""
            from app.coordination.event_router import get_event_payload
            payload = get_event_payload(event)
            node_id = payload.get("node_id", "unknown")
            logger.info(f"[ClusterMonitorHandler] Node {node_id} recovered")

        def health_check(self) -> HBHealthCheckResult:
            """Health check for DaemonManager integration."""
            from app.coordination.protocols import CoordinatorStatus

            is_healthy = self._last_status is not None
            details = {
                "has_status": self._last_status is not None,
                "cycle_interval": self._cycle_interval,
            }

            if self._last_status:
                details["active_nodes"] = self._last_status.active_nodes
                details["total_nodes"] = self._last_status.total_nodes
                details["total_games"] = self._last_status.total_games

            return HBHealthCheckResult(
                healthy=is_healthy,
                status=CoordinatorStatus.RUNNING if is_healthy else CoordinatorStatus.STARTING,
                message=f"ClusterMonitorHandler: {'OK' if is_healthy else 'No status yet'}",
                details=details,
            )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RingRift Cluster Monitoring Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single snapshot
  python -m app.distributed.cluster_monitor

  # Watch mode with 30s updates
  python -m app.distributed.cluster_monitor --watch --interval 30

  # Query specific hosts only
  python -m app.distributed.cluster_monitor --hosts gpu-node-1 gpu-node-2

  # Skip expensive checks
  python -m app.distributed.cluster_monitor --no-training --no-disk
        """,
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor with periodic updates",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Update interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--hosts",
        nargs="+",
        help="Specific hosts to monitor (default: all active)",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear screen in watch mode",
    )
    parser.add_argument(
        "--no-games",
        action="store_true",
        help="Skip game count queries",
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Skip training status checks",
    )
    parser.add_argument(
        "--no-disk",
        action="store_true",
        help="Skip disk usage checks",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Include sync status (expensive)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="SSH timeout in seconds (default: 15)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Query nodes sequentially instead of parallel",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create monitor
    monitor = ClusterMonitor(
        ssh_timeout=args.timeout,
        parallel=not args.sequential,
    )

    if args.watch:
        # Watch mode
        monitor.watch(
            interval=args.interval,
            clear_screen=not args.no_clear,
        )
    else:
        # Single snapshot
        status = monitor.get_cluster_status(
            hosts=args.hosts,
            include_game_counts=not args.no_games,
            include_training_status=not args.no_training,
            include_disk_usage=not args.no_disk,
            include_sync_status=args.sync,
        )
        monitor.print_dashboard(status)


if __name__ == "__main__":
    main()
