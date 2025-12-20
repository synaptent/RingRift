"""P2P Orchestrator Client Library.

Provides a Python interface for interacting with the P2P orchestrator
cluster management system. Use this instead of direct SSH when you
want to coordinate with the cluster's job scheduling.

Usage:
    from scripts.p2p.client import P2PClient, JobRequest

    client = P2PClient()

    # Get cluster status
    status = client.get_status()
    print(f"Leader: {status.leader_id}")
    print(f"Online nodes: {len(status.online_nodes)}")

    # Submit a job
    job = JobRequest(
        job_type="selfplay",
        config_key="square8_2p",
        num_games=1000,
    )
    job_id = client.submit_job(job)

    # Wait for completion
    result = client.wait_for_job(job_id, timeout=3600)
    print(f"Job completed: {result.status}")
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

DEFAULT_PORT = 8770
DEFAULT_TIMEOUT = 30


class JobType(Enum):
    """Types of jobs that can be submitted to the cluster."""
    SELFPLAY = "selfplay"
    TRAINING = "training"
    GAUNTLET = "gauntlet"
    TOURNAMENT = "tournament"
    BENCHMARK = "benchmark"


class JobStatus(Enum):
    """Status of a submitted job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRequest:
    """Request to submit a job to the cluster."""
    job_type: str
    config_key: str = "square8_2p"
    num_games: int = 100
    node_id: str | None = None  # Specific node, or None for auto-assign
    priority: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """Result of a completed job."""
    job_id: str
    status: JobStatus
    node_id: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output_path: str | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    host: str
    online: bool
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    role: str = "follower"
    gpu_name: str = ""


@dataclass
class ClusterStatus:
    """Status of the P2P cluster."""
    leader_id: str | None = None
    role: str = "unknown"
    total_nodes: int = 0
    online_nodes: list[NodeInfo] = field(default_factory=list)
    active_selfplay_count: int = 0
    active_training_count: int = 0
    avg_gpu_util: float = 0.0
    version: str = ""
    uptime_seconds: int = 0


class P2PClientError(Exception):
    """Error from P2P client operations."""


class P2PClient:
    """Client for interacting with the P2P orchestrator."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        timeout: int = DEFAULT_TIMEOUT,
        auto_discover: bool = True,
    ):
        """Initialize P2P client.

        Args:
            host: P2P orchestrator host
            port: P2P orchestrator port
            timeout: Request timeout in seconds
            auto_discover: Try to discover orchestrator if localhost fails
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.auto_discover = auto_discover
        self._base_url: str | None = None

    @property
    def base_url(self) -> str:
        """Get base URL for API requests."""
        if self._base_url is None:
            self._base_url = f"http://{self.host}:{self.port}"
        return self._base_url

    def _request(
        self,
        path: str,
        method: str = "GET",
        data: dict | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request to orchestrator."""
        url = f"{self.base_url}{path}"

        try:
            if data:
                body = json.dumps(data).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=body,
                    method=method,
                    headers={"Content-Type": "application/json"},
                )
            else:
                req = urllib.request.Request(url, method=method)

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())

        except urllib.error.URLError as e:
            # Try to auto-discover if localhost fails
            if self.auto_discover and self.host == "localhost":
                discovered = self._try_discover()
                if discovered:
                    self.host = discovered
                    self._base_url = None
                    return self._request(path, method, data)

            raise P2PClientError(f"Failed to connect to {url}: {e}")

        except json.JSONDecodeError as e:
            raise P2PClientError(f"Invalid JSON response: {e}")

    def _try_discover(self) -> str | None:
        """Try to discover P2P orchestrator on common hosts."""
        # Try loading from config
        try:
            from scripts.lib.hosts import get_hosts
            for host in get_hosts(p2p_voter=True):
                if host.tailscale_ip:
                    try:
                        test_url = f"http://{host.tailscale_ip}:{self.port}/health"
                        req = urllib.request.Request(test_url)
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            if resp.status == 200:
                                return host.tailscale_ip
                    except Exception:
                        continue
        except Exception:
            pass

        return None

    def health_check(self) -> bool:
        """Check if orchestrator is healthy."""
        try:
            result = self._request("/health")
            return result.get("status") == "healthy"
        except P2PClientError:
            return False

    def get_status(self) -> ClusterStatus:
        """Get current cluster status."""
        data = self._request("/status")

        # Parse nodes
        nodes = []
        self_info = data.get("self", {})
        if self_info:
            nodes.append(NodeInfo(
                node_id=self_info.get("node_id", "unknown"),
                host=self_info.get("host", ""),
                online=True,
                gpu_percent=self_info.get("gpu_percent", 0),
                gpu_memory_percent=self_info.get("gpu_memory_percent", 0),
                cpu_percent=self_info.get("cpu_percent", 0),
                memory_percent=self_info.get("memory_percent", 0),
                disk_percent=self_info.get("disk_percent", 0),
                selfplay_jobs=self_info.get("selfplay_jobs", 0),
                training_jobs=self_info.get("training_jobs", 0),
                role=data.get("role", "follower"),
                gpu_name=self_info.get("gpu_name", ""),
            ))

        peers = data.get("peers", {})
        for peer_id, peer_info in peers.items():
            last_hb = peer_info.get("last_heartbeat", 0)
            is_online = (time.time() - last_hb) < 180

            nodes.append(NodeInfo(
                node_id=peer_id,
                host=peer_info.get("host", ""),
                online=is_online,
                gpu_percent=peer_info.get("gpu_percent", 0),
                gpu_memory_percent=peer_info.get("gpu_memory_percent", 0),
                cpu_percent=peer_info.get("cpu_percent", 0),
                memory_percent=peer_info.get("memory_percent", 0),
                disk_percent=peer_info.get("disk_percent", 0),
                selfplay_jobs=peer_info.get("selfplay_jobs", 0),
                training_jobs=peer_info.get("training_jobs", 0),
                role=peer_info.get("role", "follower"),
                gpu_name=peer_info.get("gpu_name", ""),
            ))

        online_nodes = [n for n in nodes if n.online]
        total_selfplay = sum(n.selfplay_jobs for n in nodes)
        total_training = sum(n.training_jobs for n in nodes)

        gpu_nodes = [n for n in online_nodes if n.gpu_name]
        avg_gpu = sum(n.gpu_percent for n in gpu_nodes) / len(gpu_nodes) if gpu_nodes else 0

        return ClusterStatus(
            leader_id=data.get("leader_id") or data.get("effective_leader_id"),
            role=data.get("role", "unknown"),
            total_nodes=len(nodes),
            online_nodes=online_nodes,
            active_selfplay_count=total_selfplay,
            active_training_count=total_training,
            avg_gpu_util=avg_gpu,
            version=data.get("version", ""),
            uptime_seconds=data.get("uptime_seconds", 0),
        )

    def get_node(self, node_id: str) -> NodeInfo | None:
        """Get info about a specific node."""
        status = self.get_status()
        for node in status.online_nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_jobs(self, status: JobStatus | None = None) -> list[dict[str, Any]]:
        """Get list of jobs from orchestrator."""
        data = self._request("/jobs")
        jobs = data.get("jobs", [])

        if status:
            jobs = [j for j in jobs if j.get("status") == status.value]

        return jobs

    def submit_job(self, job: JobRequest) -> str:
        """Submit a job to the cluster.

        Args:
            job: Job request details

        Returns:
            Job ID
        """
        data = {
            "job_type": job.job_type,
            "config_key": job.config_key,
            "num_games": job.num_games,
            "priority": job.priority,
            "metadata": job.metadata,
        }

        if job.node_id:
            data["node_id"] = job.node_id

        result = self._request("/jobs", method="POST", data=data)
        return result.get("job_id", "")

    def get_job(self, job_id: str) -> JobResult | None:
        """Get details of a specific job."""
        try:
            data = self._request(f"/jobs/{job_id}")

            status = JobStatus(data.get("status", "pending"))

            return JobResult(
                job_id=job_id,
                status=status,
                node_id=data.get("node_id", ""),
                started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
                completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                output_path=data.get("output_path"),
                error=data.get("error"),
                metrics=data.get("metrics", {}),
            )
        except P2PClientError:
            return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        try:
            self._request(f"/jobs/{job_id}/cancel", method="POST")
            return True
        except P2PClientError:
            return False

    def wait_for_job(
        self,
        job_id: str,
        timeout: int = 3600,
        poll_interval: int = 10,
    ) -> JobResult | None:
        """Wait for a job to complete.

        Args:
            job_id: Job ID to wait for
            timeout: Maximum wait time in seconds
            poll_interval: How often to check status

        Returns:
            JobResult when complete, or None on timeout
        """
        start = time.time()

        while time.time() - start < timeout:
            result = self.get_job(job_id)

            if result and result.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return result

            time.sleep(poll_interval)

        return None


# Convenience functions
_default_client: P2PClient | None = None


def get_client() -> P2PClient:
    """Get the default P2P client instance."""
    global _default_client
    if _default_client is None:
        _default_client = P2PClient()
    return _default_client


def get_cluster_status() -> ClusterStatus:
    """Get current cluster status using default client."""
    return get_client().get_status()


def submit_selfplay_job(
    config_key: str = "square8_2p",
    num_games: int = 1000,
    node_id: str | None = None,
) -> str:
    """Submit a self-play job to the cluster.

    Args:
        config_key: Board/player configuration
        num_games: Number of games to generate
        node_id: Specific node, or None for auto-assign

    Returns:
        Job ID
    """
    job = JobRequest(
        job_type="selfplay",
        config_key=config_key,
        num_games=num_games,
        node_id=node_id,
    )
    return get_client().submit_job(job)


def submit_training_job(
    config_key: str = "square8_2p",
    epochs: int = 50,
    node_id: str | None = None,
) -> str:
    """Submit a training job to the cluster.

    Args:
        config_key: Board/player configuration
        epochs: Number of training epochs
        node_id: Specific node, or None for auto-assign

    Returns:
        Job ID
    """
    job = JobRequest(
        job_type="training",
        config_key=config_key,
        node_id=node_id,
        metadata={"epochs": epochs},
    )
    return get_client().submit_job(job)
