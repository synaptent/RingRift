"""
Cluster Operations Library

Provides utilities for interacting with the RingRift training cluster:
- SSH command execution with retries and timeouts
- Node discovery and health checking
- Data transfer utilities
- GPU monitoring

Usage:
    from scripts.lib.cluster import ClusterManager, ClusterNode

    cluster = ClusterManager()
    for node in cluster.get_healthy_nodes():
        result = node.run("nvidia-smi")
        print(f"{node.name}: {result.stdout}")
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

T = TypeVar("T")


class NodeStatus(Enum):
    """Status of a cluster node."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNREACHABLE = "unreachable"
    DEGRADED = "degraded"  # Reachable but has issues


@dataclass
class CommandResult:
    """Result of a remote command execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    node: str
    command: str

    def __bool__(self) -> bool:
        return self.success

    @property
    def output(self) -> str:
        """Return stdout, or stderr if stdout is empty."""
        return self.stdout.strip() or self.stderr.strip()


@dataclass
class GPUInfo:
    """GPU information from a node."""
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_percent: int
    temperature_c: int

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb

    @property
    def memory_utilization_percent(self) -> float:
        return (self.memory_used_mb / self.memory_total_mb * 100) if self.memory_total_mb > 0 else 0


@dataclass
class NodeHealth:
    """Health status of a cluster node."""
    status: NodeStatus
    gpus: List[GPUInfo] = field(default_factory=list)
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_free_gb: float = 0.0
    uptime_seconds: int = 0
    error_message: Optional[str] = None
    checked_at: float = field(default_factory=time.time)

    @property
    def total_gpu_memory_gb(self) -> float:
        return sum(g.memory_total_mb for g in self.gpus) / 1024

    @property
    def avg_gpu_utilization(self) -> float:
        return sum(g.utilization_percent for g in self.gpus) / len(self.gpus) if self.gpus else 0


class ClusterNode:
    """Represents a single node in the cluster."""

    def __init__(
        self,
        name: str,
        hostname: Optional[str] = None,
        ssh_user: str = "ubuntu",
        ssh_key: Optional[str] = None,
        connect_timeout: int = 10,
        command_timeout: int = 30,
        ringrift_path: str = "~/ringrift/ai-service",
    ):
        self.name = name
        self.hostname = hostname or name
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.connect_timeout = connect_timeout
        self.command_timeout = command_timeout
        self.ringrift_path = ringrift_path
        self._health: Optional[NodeHealth] = None
        self._health_checked_at: float = 0

    def run(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        check: bool = False,
    ) -> CommandResult:
        """Execute a command on this node via SSH.

        Args:
            command: The command to execute
            timeout: Command timeout in seconds (default: self.command_timeout)
            cwd: Working directory (default: ringrift_path)
            env: Environment variables to set
            check: If True, raise exception on non-zero exit

        Returns:
            CommandResult with execution details
        """
        timeout = timeout or self.command_timeout
        cwd = cwd or self.ringrift_path

        # Build the full command
        full_command = f"cd {cwd} && {command}"

        if env:
            env_str = " ".join(f"{k}={v}" for k, v in env.items())
            full_command = f"{env_str} {full_command}"

        # Build SSH command
        ssh_cmd = [
            "ssh",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
        ]

        if self.ssh_key:
            ssh_cmd.extend(["-i", self.ssh_key])

        ssh_cmd.extend([
            f"{self.ssh_user}@{self.hostname}" if self.ssh_user else self.hostname,
            full_command,
        ])

        start_time = time.time()

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            duration = time.time() - start_time

            cmd_result = CommandResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                duration_seconds=duration,
                node=self.name,
                command=command,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            cmd_result = CommandResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                exit_code=-1,
                duration_seconds=duration,
                node=self.name,
                command=command,
            )
        except Exception as e:
            duration = time.time() - start_time
            cmd_result = CommandResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                duration_seconds=duration,
                node=self.name,
                command=command,
            )

        if check and not cmd_result.success:
            raise CommandError(cmd_result)

        return cmd_result

    def run_python(
        self,
        script: str,
        args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        """Execute a Python script on this node.

        Args:
            script: Path to script (relative to ringrift_path)
            args: Script arguments
            timeout: Command timeout
        """
        args_str = " ".join(args) if args else ""
        cmd = f"source venv/bin/activate && PYTHONPATH=. python {script} {args_str}"
        return self.run(cmd, timeout=timeout)

    def check_health(self, force: bool = False, cache_seconds: int = 60) -> NodeHealth:
        """Check the health of this node.

        Args:
            force: Force refresh even if cached
            cache_seconds: How long to cache health results

        Returns:
            NodeHealth with current status
        """
        if not force and self._health and (time.time() - self._health_checked_at) < cache_seconds:
            return self._health

        health = NodeHealth(status=NodeStatus.UNKNOWN)

        # Check basic connectivity
        result = self.run("echo connected", timeout=10)
        if not result.success:
            health.status = NodeStatus.UNREACHABLE
            health.error_message = result.stderr
            self._health = health
            self._health_checked_at = time.time()
            return health

        # Get GPU info
        gpu_result = self.run(
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu "
            "--format=csv,noheader,nounits",
            timeout=15,
        )

        if gpu_result.success:
            for line in gpu_result.stdout.strip().split("\n"):
                if line.strip():
                    try:
                        parts = [p.strip() for p in line.split(",")]
                        health.gpus.append(GPUInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            memory_total_mb=int(parts[2]),
                            memory_used_mb=int(parts[3]),
                            utilization_percent=int(parts[4]),
                            temperature_c=int(parts[5]),
                        ))
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse GPU info on {self.name}: {e}")

        # Get system info
        sys_result = self.run(
            "cat /proc/loadavg && free -g | grep Mem && df -BG / | tail -1",
            timeout=10,
        )

        if sys_result.success:
            lines = sys_result.stdout.strip().split("\n")
            if len(lines) >= 1:
                load_parts = lines[0].split()[:3]
                try:
                    health.load_average = tuple(float(x) for x in load_parts)
                except ValueError:
                    pass

            if len(lines) >= 2:
                mem_parts = lines[1].split()
                try:
                    health.memory_total_gb = float(mem_parts[1])
                    health.memory_available_gb = float(mem_parts[6]) if len(mem_parts) > 6 else float(mem_parts[3])
                except (ValueError, IndexError):
                    pass

            if len(lines) >= 3:
                disk_parts = lines[2].split()
                try:
                    health.disk_free_gb = float(disk_parts[3].rstrip("G"))
                except (ValueError, IndexError):
                    pass

        # Determine overall status
        if health.gpus:
            # Check for GPU issues
            any_hot = any(g.temperature_c > 85 for g in health.gpus)
            any_oom = any(g.memory_utilization_percent > 95 for g in health.gpus)

            if any_hot or any_oom:
                health.status = NodeStatus.DEGRADED
                if any_hot:
                    health.error_message = "GPU temperature high"
                elif any_oom:
                    health.error_message = "GPU memory near capacity"
            else:
                health.status = NodeStatus.HEALTHY
        else:
            health.status = NodeStatus.DEGRADED
            health.error_message = "No GPUs detected"

        self._health = health
        self._health_checked_at = time.time()
        return health

    def transfer_file(
        self,
        local_path: Path,
        remote_path: str,
        direction: str = "upload",
        compress: bool = True,
    ) -> CommandResult:
        """Transfer a file to/from this node.

        Args:
            local_path: Local file path
            remote_path: Remote file path
            direction: "upload" or "download"
            compress: Use compression for transfer
        """
        scp_cmd = ["scp", "-o", f"ConnectTimeout={self.connect_timeout}"]

        if compress:
            scp_cmd.append("-C")
        if self.ssh_key:
            scp_cmd.extend(["-i", self.ssh_key])

        remote_full = f"{self.ssh_user}@{self.hostname}:{remote_path}"

        if direction == "upload":
            scp_cmd.extend([str(local_path), remote_full])
        else:
            scp_cmd.extend([remote_full, str(local_path)])

        start_time = time.time()

        try:
            result = subprocess.run(
                scp_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min for file transfers
            )

            return CommandResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                duration_seconds=time.time() - start_time,
                node=self.name,
                command=f"scp {direction}: {local_path} <-> {remote_path}",
            )
        except Exception as e:
            return CommandResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                duration_seconds=time.time() - start_time,
                node=self.name,
                command=f"scp {direction}: {local_path} <-> {remote_path}",
            )

    def __repr__(self) -> str:
        return f"ClusterNode({self.name})"


class CommandError(Exception):
    """Raised when a command fails with check=True."""

    def __init__(self, result: CommandResult):
        self.result = result
        super().__init__(f"Command failed on {result.node}: {result.command}\n{result.stderr}")


class ClusterManager:
    """Manages a cluster of training nodes."""

    # Default cluster nodes
    DEFAULT_NODES = [
        {"name": "lambda-gh200-e", "hostname": "100.88.176.74"},
        {"name": "lambda-gh200-f", "hostname": "100.104.165.116"},
        {"name": "lambda-gh200-g", "hostname": "100.104.126.58"},
        {"name": "lambda-gh200-h", "hostname": "100.65.88.62"},
        {"name": "lambda-gh200-i", "hostname": "100.99.27.56"},
        {"name": "lambda-gh200-k", "hostname": "100.96.142.42"},
        {"name": "lambda-gh200-l", "hostname": "100.76.145.60"},
        {"name": "lambda-2xh100", "hostname": "100.97.104.89"},
    ]

    def __init__(
        self,
        nodes: Optional[List[Dict[str, Any]]] = None,
        max_workers: int = 8,
        ssh_user: str = "ubuntu",
    ):
        """Initialize cluster manager.

        Args:
            nodes: List of node configs (uses DEFAULT_NODES if not specified)
            max_workers: Max parallel workers for cluster operations
            ssh_user: Default SSH user
        """
        self.max_workers = max_workers
        node_configs = nodes or self.DEFAULT_NODES

        self.nodes: Dict[str, ClusterNode] = {}
        for config in node_configs:
            node = ClusterNode(
                name=config["name"],
                hostname=config.get("hostname"),
                ssh_user=config.get("ssh_user", ssh_user),
                ssh_key=config.get("ssh_key"),
            )
            self.nodes[node.name] = node

    def get_node(self, name: str) -> Optional[ClusterNode]:
        """Get a node by name."""
        return self.nodes.get(name)

    def get_healthy_nodes(self, force_check: bool = False) -> List[ClusterNode]:
        """Get all healthy nodes in the cluster."""
        healthy = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(node.check_health, force_check): node
                for node in self.nodes.values()
            }

            for future in as_completed(futures):
                node = futures[future]
                try:
                    health = future.result()
                    if health.status == NodeStatus.HEALTHY:
                        healthy.append(node)
                except Exception as e:
                    logger.warning(f"Health check failed for {node.name}: {e}")

        return healthy

    def run_on_all(
        self,
        command: str,
        nodes: Optional[List[str]] = None,
        parallel: bool = True,
        continue_on_error: bool = True,
    ) -> Dict[str, CommandResult]:
        """Run a command on multiple nodes.

        Args:
            command: Command to run
            nodes: List of node names (default: all nodes)
            parallel: Run in parallel
            continue_on_error: Continue even if some nodes fail

        Returns:
            Dict mapping node names to results
        """
        target_nodes = [self.nodes[n] for n in (nodes or self.nodes.keys())]
        results = {}

        if parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(node.run, command): node
                    for node in target_nodes
                }

                for future in as_completed(futures):
                    node = futures[future]
                    try:
                        results[node.name] = future.result()
                    except Exception as e:
                        results[node.name] = CommandResult(
                            success=False,
                            stdout="",
                            stderr=str(e),
                            exit_code=-1,
                            duration_seconds=0,
                            node=node.name,
                            command=command,
                        )
                        if not continue_on_error:
                            raise
        else:
            for node in target_nodes:
                try:
                    results[node.name] = node.run(command)
                except Exception as e:
                    results[node.name] = CommandResult(
                        success=False,
                        stdout="",
                        stderr=str(e),
                        exit_code=-1,
                        duration_seconds=0,
                        node=node.name,
                        command=command,
                    )
                    if not continue_on_error:
                        raise

        return results

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all nodes."""
        metrics = {
            "timestamp": time.time(),
            "nodes": {},
            "total_gpus": 0,
            "healthy_nodes": 0,
            "total_gpu_memory_gb": 0,
            "avg_gpu_utilization": 0,
        }

        gpu_utils = []

        for name, node in self.nodes.items():
            health = node.check_health()

            metrics["nodes"][name] = {
                "status": health.status.value,
                "gpus": len(health.gpus),
                "gpu_memory_gb": health.total_gpu_memory_gb,
                "gpu_utilization": health.avg_gpu_utilization,
                "load_average": health.load_average,
                "memory_available_gb": health.memory_available_gb,
            }

            if health.status == NodeStatus.HEALTHY:
                metrics["healthy_nodes"] += 1

            metrics["total_gpus"] += len(health.gpus)
            metrics["total_gpu_memory_gb"] += health.total_gpu_memory_gb

            if health.gpus:
                gpu_utils.append(health.avg_gpu_utilization)

        if gpu_utils:
            metrics["avg_gpu_utilization"] = sum(gpu_utils) / len(gpu_utils)

        return metrics

    def deploy_file(
        self,
        local_path: Path,
        remote_path: str,
        nodes: Optional[List[str]] = None,
    ) -> Dict[str, CommandResult]:
        """Deploy a file to multiple nodes.

        Args:
            local_path: Local file to deploy
            remote_path: Remote destination path
            nodes: Target nodes (default: all)

        Returns:
            Dict mapping node names to transfer results
        """
        target_nodes = [self.nodes[n] for n in (nodes or self.nodes.keys())]
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(node.transfer_file, local_path, remote_path, "upload"): node
                for node in target_nodes
            }

            for future in as_completed(futures):
                node = futures[future]
                try:
                    results[node.name] = future.result()
                except Exception as e:
                    results[node.name] = CommandResult(
                        success=False,
                        stdout="",
                        stderr=str(e),
                        exit_code=-1,
                        duration_seconds=0,
                        node=node.name,
                        command=f"deploy {local_path}",
                    )

        return results


def get_cluster() -> ClusterManager:
    """Get a default cluster manager instance."""
    return ClusterManager()


# =============================================================================
# Multi-Provider Cluster Automation
# =============================================================================


class VastNodeManager:
    """Manages Vast.ai nodes for automated provisioning and lifecycle."""

    def __init__(self):
        self.ssh_key = os.path.expanduser("~/.ssh/id_cluster")

    def list_instances(self) -> List[Dict[str, Any]]:
        """List all Vast.ai instances."""
        try:
            result = subprocess.run(
                ["vastai", "show", "instances", "--raw"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
            logger.warning(f"vastai failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"Failed to list Vast.ai instances: {e}")
        return []

    def get_instance_ssh(self, instance_id: int) -> Optional[Tuple[str, int]]:
        """Get SSH host and port for a Vast.ai instance."""
        for inst in self.list_instances():
            if inst.get("id") == instance_id:
                return (inst.get("ssh_host", ""), inst.get("ssh_port", 22))
        return None

    def setup_tailscale(self, ssh_host: str, ssh_port: int, auth_key: str) -> bool:
        """Setup Tailscale on a Vast.ai instance."""
        install_cmd = """
        if ! command -v tailscale &> /dev/null; then
            curl -fsSL https://tailscale.com/install.sh | sh
        fi
        sudo tailscale up --authkey {auth_key} --ssh
        """
        try:
            result = subprocess.run(
                [
                    "ssh", "-o", "ConnectTimeout=30", "-o", "BatchMode=yes",
                    "-i", self.ssh_key, "-p", str(ssh_port),
                    f"root@{ssh_host}",
                    install_cmd.format(auth_key=auth_key),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to setup Tailscale: {e}")
            return False

    def start_p2p_orchestrator(
        self,
        ssh_host: str,
        ssh_port: int,
        node_id: str,
        peers: List[str],
    ) -> bool:
        """Start P2P orchestrator on a Vast.ai instance."""
        peers_str = ",".join(peers)
        cmd = f"""
        cd ~/ringrift/ai-service &&
        pkill -f p2p_orchestrator || true &&
        mkdir -p logs &&
        PYTHONPATH=. nohup python3 scripts/p2p_orchestrator.py \\
            --node-id {node_id} --port 8770 \\
            --peers {peers_str} \\
            --ringrift-path ~/ringrift \\
            > logs/p2p_orchestrator.log 2>&1 &
        sleep 2 && pgrep -f p2p_orchestrator
        """
        try:
            result = subprocess.run(
                [
                    "ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
                    "-i", self.ssh_key, "-p", str(ssh_port),
                    f"root@{ssh_host}", cmd,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to start P2P orchestrator: {e}")
            return False


class ClusterAutomation:
    """Automated cluster management across multiple providers."""

    DEFAULT_PEERS = [
        "http://100.78.101.123:8770",  # lambda-h100
        "http://100.123.183.70:8770",  # lambda-gh200-a
    ]

    def __init__(self):
        self.cluster = ClusterManager()
        self.vast = VastNodeManager()

    def discover_all_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Discover all nodes across all providers.

        Returns:
            Dict mapping node_id to node info including:
            - provider: "lambda" | "vast" | "aws" | "hetzner"
            - tailscale_ip: str or None
            - p2p_status: "running" | "stopped" | "unknown"
            - gpu_name: str
        """
        nodes = {}

        # Get Tailscale nodes
        try:
            result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                ts_data = json.loads(result.stdout)
                for peer in ts_data.get("Peer", {}).values():
                    name = peer.get("HostName", "")
                    ip = peer.get("TailscaleIPs", [""])[0] if peer.get("TailscaleIPs") else ""
                    online = peer.get("Online", False)

                    # Determine provider from name
                    provider = "unknown"
                    if "lambda" in name.lower() or "gh200" in name.lower():
                        provider = "lambda"
                    elif "vast" in name.lower():
                        provider = "vast"
                    elif "aws" in name.lower() or name.startswith("ip-"):
                        provider = "aws"

                    nodes[name] = {
                        "provider": provider,
                        "tailscale_ip": ip,
                        "online": online,
                        "p2p_status": "unknown",
                    }
        except Exception as e:
            logger.warning(f"Failed to get Tailscale status: {e}")

        # Add Vast.ai instances
        for inst in self.vast.list_instances():
            node_id = f"vast-{inst.get('id')}"
            if node_id not in nodes:
                nodes[node_id] = {
                    "provider": "vast",
                    "tailscale_ip": None,
                    "ssh_host": inst.get("ssh_host"),
                    "ssh_port": inst.get("ssh_port"),
                    "gpu_name": inst.get("gpu_name"),
                    "online": inst.get("actual_status") == "running",
                    "p2p_status": "unknown",
                }

        return nodes

    def check_p2p_status(self, tailscale_ip: str) -> str:
        """Check if P2P orchestrator is running on a node."""
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"http://{tailscale_ip}:8770/status",
                timeout=5,
            ) as resp:
                if resp.status == 200:
                    return "running"
        except Exception:
            pass
        return "stopped"

    def ensure_all_orchestrators_running(
        self,
        peers: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Ensure P2P orchestrators are running on all nodes.

        Returns:
            Dict mapping node_id to whether orchestrator was started/verified.
        """
        peers = peers or self.DEFAULT_PEERS
        results = {}

        nodes = self.discover_all_nodes()

        for node_id, info in nodes.items():
            if not info.get("online"):
                results[node_id] = False
                continue

            ip = info.get("tailscale_ip")
            if ip:
                status = self.check_p2p_status(ip)
                if status == "running":
                    results[node_id] = True
                    logger.info(f"{node_id}: P2P already running")
                    continue

            # Try to start orchestrator
            if info.get("provider") == "vast" and info.get("ssh_host"):
                success = self.vast.start_p2p_orchestrator(
                    info["ssh_host"],
                    info["ssh_port"],
                    node_id,
                    peers,
                )
                results[node_id] = success
                if success:
                    logger.info(f"{node_id}: Started P2P orchestrator")
                else:
                    logger.warning(f"{node_id}: Failed to start P2P")
            elif ip and node_id in self.cluster.nodes:
                # Use cluster node SSH
                node = self.cluster.nodes[node_id]
                result = node.run(f"""
                    pgrep -f p2p_orchestrator || (
                        cd ~/ringrift/ai-service &&
                        source venv/bin/activate &&
                        mkdir -p logs &&
                        PYTHONPATH=. nohup python scripts/p2p_orchestrator.py \\
                            --node-id {node_id} --port 8770 \\
                            --peers {','.join(peers)} \\
                            > logs/p2p_orchestrator.log 2>&1 &
                    )
                """, timeout=30)
                results[node_id] = result.success
            else:
                results[node_id] = False
                logger.warning(f"{node_id}: Cannot start orchestrator (no SSH access)")

        return results

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of entire cluster status."""
        nodes = self.discover_all_nodes()

        summary = {
            "total_nodes": len(nodes),
            "online_nodes": 0,
            "p2p_running": 0,
            "by_provider": {},
            "nodes": {},
        }

        for node_id, info in nodes.items():
            provider = info.get("provider", "unknown")

            if provider not in summary["by_provider"]:
                summary["by_provider"][provider] = {"total": 0, "online": 0}

            summary["by_provider"][provider]["total"] += 1

            if info.get("online"):
                summary["online_nodes"] += 1
                summary["by_provider"][provider]["online"] += 1

                # Check P2P status
                ip = info.get("tailscale_ip")
                if ip:
                    status = self.check_p2p_status(ip)
                    info["p2p_status"] = status
                    if status == "running":
                        summary["p2p_running"] += 1

            summary["nodes"][node_id] = info

        return summary


def get_automation() -> ClusterAutomation:
    """Get a cluster automation instance."""
    return ClusterAutomation()
