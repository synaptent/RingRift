"""P2P Backend client for communication with P2P orchestrator cluster.

This module provides a REST API client for the P2P orchestrator, enabling
job dispatch and cluster management without direct SSH access.

Consolidated from:
- scripts/archive/pipeline_orchestrator.py (P2PBackend, discover_p2p_leader_url)

Usage:
    from app.coordination.p2p_backend import P2PBackend, discover_p2p_leader_url

    # Discover the current P2P leader
    leader_url = await discover_p2p_leader_url(["http://node1:8770", "http://node2:8770"])

    # Connect to the P2P backend
    backend = P2PBackend(leader_url)
    nodes = await backend.get_healthy_nodes()

    # Start a job
    result = await backend.start_canonical_selfplay(
        board_type="square8",
        num_players=2,
        games_per_node=500
    )

    # Wait for completion
    await backend.wait_for_pipeline_completion(result["job_id"])
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from app.config.ports import P2P_DEFAULT_PORT

logger = logging.getLogger(__name__)

# P2P orchestrator timing constants
P2P_HTTP_TIMEOUT = 30  # seconds
P2P_JOB_POLL_INTERVAL = 10  # seconds
MAX_PHASE_WAIT_MINUTES = 120  # Maximum wait for any phase

# Try to import aiohttp
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None  # type: ignore
    HAS_AIOHTTP = False


@dataclass
class P2PNodeInfo:
    """Information about a node from P2P cluster."""

    node_id: str
    host: str
    port: int
    role: str
    has_gpu: bool
    gpu_name: str
    memory_gb: int
    capabilities: list[str]
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    is_alive: bool = True
    is_healthy: bool = True

    def to_worker_config(self) -> dict[str, Any]:
        """Convert to WorkerConfig-compatible dict for compatibility."""
        return {
            "name": self.node_id,
            "host": self.host,
            "role": "mixed" if self.has_gpu else "selfplay",
            "capabilities": self.capabilities,
            "ssh_port": 22,
            "remote_path": "~/ringrift/ai-service",
            "max_parallel_jobs": 4 if self.has_gpu else 2,
            "gpu": self.gpu_name,
            "memory_gb": self.memory_gb,
        }


class P2PBackend:
    """Backend for communicating with P2P orchestrator cluster.

    Provides interface to P2P orchestrator REST API for job dispatch without SSH.
    """

    def __init__(
        self,
        leader_url: str,
        auth_token: str | None = None,
        timeout: float = P2P_HTTP_TIMEOUT,
    ):
        """Initialize P2P backend.

        Args:
            leader_url: URL of the P2P leader node (e.g., http://192.168.1.100:8770)
            auth_token: Optional authentication token
            timeout: HTTP request timeout in seconds
        """
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for P2P backend: pip install aiohttp")
        self.leader_url = leader_url.rstrip("/")
        self.auth_token = auth_token or os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN", "")
        self.timeout = timeout
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> P2PBackend:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def get_cluster_status(self) -> dict[str, Any]:
        """Get cluster status from the leader node."""
        session = await self._get_session()
        async with session.get(f"{self.leader_url}/api/cluster/status") as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to get cluster status: {resp.status}")
            return await resp.json()

    async def get_nodes(self) -> list[P2PNodeInfo]:
        """Get all nodes in the cluster."""
        status = await self.get_cluster_status()
        nodes = []
        for n in status.get("nodes", []):
            nodes.append(
                P2PNodeInfo(
                    node_id=n.get("node_id", ""),
                    host=n.get("host", ""),
                    port=n.get("port", P2P_DEFAULT_PORT),
                    role=n.get("role", "follower"),
                    has_gpu=n.get("has_gpu", False),
                    gpu_name=n.get("gpu_name", ""),
                    memory_gb=n.get("memory_gb", 0),
                    capabilities=n.get("capabilities", []),
                    cpu_percent=n.get("cpu_percent", 0),
                    memory_percent=n.get("memory_percent", 0),
                    disk_percent=n.get("disk_percent", 0),
                    selfplay_jobs=n.get("selfplay_jobs", 0),
                    training_jobs=n.get("training_jobs", 0),
                    is_alive=n.get("is_alive", True),
                    is_healthy=n.get("is_healthy", True),
                )
            )
        return nodes

    async def get_healthy_nodes(self) -> list[P2PNodeInfo]:
        """Get all healthy nodes in the cluster."""
        return [n for n in await self.get_nodes() if n.is_alive and n.is_healthy]

    async def get_gpu_nodes(self) -> list[P2PNodeInfo]:
        """Get all healthy GPU nodes."""
        return [n for n in await self.get_healthy_nodes() if n.has_gpu]

    async def get_cpu_nodes(self) -> list[P2PNodeInfo]:
        """Get all healthy CPU-only nodes."""
        return [n for n in await self.get_healthy_nodes() if not n.has_gpu]

    async def start_canonical_selfplay(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        games_per_node: int = 500,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Start canonical selfplay across the cluster.

        Args:
            board_type: Board type (square8, square19, hexagonal)
            num_players: Number of players (2, 3, 4)
            games_per_node: Number of games per node
            seed: Random seed for reproducibility

        Returns:
            Dict with job_id and other metadata
        """
        session = await self._get_session()
        payload = {
            "phase": "canonical_selfplay",
            "board_type": board_type,
            "num_players": num_players,
            "games_per_node": games_per_node,
            "seed": seed,
        }
        async with session.post(f"{self.leader_url}/pipeline/start", json=payload) as resp:
            result = await resp.json()
            if not result.get("success"):
                raise RuntimeError(f"Failed to start canonical selfplay: {result.get('error')}")
            return result

    async def start_parity_validation(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        db_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Start parity validation on generated games.

        Args:
            board_type: Board type
            num_players: Number of players
            db_paths: Optional list of database paths to validate

        Returns:
            Dict with job_id and other metadata
        """
        session = await self._get_session()
        payload = {
            "phase": "parity_validation",
            "board_type": board_type,
            "num_players": num_players,
        }
        if db_paths:
            payload["db_paths"] = db_paths
        async with session.post(f"{self.leader_url}/pipeline/start", json=payload) as resp:
            result = await resp.json()
            if not result.get("success"):
                raise RuntimeError(f"Failed to start parity validation: {result.get('error')}")
            return result

    async def start_npz_export(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        output_dir: str = "data/training",
    ) -> dict[str, Any]:
        """Start NPZ export of validated games.

        Args:
            board_type: Board type
            num_players: Number of players
            output_dir: Output directory for NPZ files

        Returns:
            Dict with job_id and other metadata
        """
        session = await self._get_session()
        payload = {
            "phase": "npz_export",
            "board_type": board_type,
            "num_players": num_players,
            "output_dir": output_dir,
        }
        async with session.post(f"{self.leader_url}/pipeline/start", json=payload) as resp:
            result = await resp.json()
            if not result.get("success"):
                raise RuntimeError(f"Failed to start NPZ export: {result.get('error')}")
            return result

    async def start_training(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        **kwargs,
    ) -> dict[str, Any]:
        """Start neural network training.

        Args:
            board_type: Board type
            num_players: Number of players
            **kwargs: Additional training parameters

        Returns:
            Dict with job_id and other metadata
        """
        session = await self._get_session()
        payload = {
            "phase": "training",
            "board_type": board_type,
            "num_players": num_players,
            **kwargs,
        }
        async with session.post(f"{self.leader_url}/pipeline/start", json=payload) as resp:
            result = await resp.json()
            if not result.get("success"):
                raise RuntimeError(f"Failed to start training: {result.get('error')}")
            return result

    async def get_pipeline_status(self) -> dict[str, Any]:
        """Get current pipeline status."""
        session = await self._get_session()
        async with session.get(f"{self.leader_url}/pipeline/status") as resp:
            return await resp.json()

    async def wait_for_pipeline_completion(
        self,
        job_id: str,
        poll_interval: float = P2P_JOB_POLL_INTERVAL,
        timeout_minutes: float = MAX_PHASE_WAIT_MINUTES,
    ) -> dict[str, Any]:
        """Wait for a pipeline job to complete.

        Args:
            job_id: Job ID to wait for
            poll_interval: Polling interval in seconds
            timeout_minutes: Maximum wait time in minutes

        Returns:
            Final pipeline status

        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout_minutes * 60:
            status = await self.get_pipeline_status()
            current = status.get("current_job", {})
            if current.get("job_id") == job_id and current.get("status") in (
                "completed",
                "failed",
            ):
                return status
            await asyncio.sleep(poll_interval)
        raise TimeoutError(
            f"Pipeline job {job_id} did not complete within {timeout_minutes} minutes"
        )

    async def trigger_data_sync(self) -> dict[str, Any]:
        """Trigger data synchronization across the cluster."""
        session = await self._get_session()
        async with session.post(f"{self.leader_url}/sync/start") as resp:
            return await resp.json()

    async def trigger_git_update(self, node_id: str | None = None) -> dict[str, Any]:
        """Trigger git update on cluster nodes.

        Args:
            node_id: Optional specific node to update (all nodes if None)

        Returns:
            Result of git update operation
        """
        session = await self._get_session()
        payload = {"node_id": node_id} if node_id else {}
        async with session.post(f"{self.leader_url}/git/update", json=payload) as resp:
            return await resp.json()

    async def get_job_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent job history.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of recent jobs
        """
        session = await self._get_session()
        async with session.get(f"{self.leader_url}/jobs/history?limit={limit}") as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return data.get("jobs", [])

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            Result of cancel operation
        """
        session = await self._get_session()
        async with session.post(f"{self.leader_url}/jobs/{job_id}/cancel") as resp:
            return await resp.json()

    async def health_check(self) -> bool:
        """Check if the P2P leader is healthy.

        Returns:
            True if leader is healthy
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.leader_url}/health") as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"P2P health check failed: {e}")
            return False


def _normalize_p2p_seed_url(raw: str) -> str:
    """Normalize a P2P seed URL."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


def _is_loopback(host: str) -> bool:
    """Check if host is a loopback address."""
    host = (host or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _is_tailscale_ip(host: str) -> bool:
    """Check if host is a Tailscale IP (100.64.0.0/10 range)."""
    host = (host or "").strip()
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.version != 4:
            return False
        return ip in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        return False  # Invalid IP address format


async def discover_p2p_leader_url(
    seed_urls: list[str],
    *,
    auth_token: str = "",
    timeout_seconds: float = 5.0,
) -> str | None:
    """Discover the current effective P2P leader URL from one or more seed nodes.

    This keeps orchestration scripts resilient to leader churn: any reachable
    seed node can be used to locate the current leader without hard-coding a
    specific instance.

    Args:
        seed_urls: List of seed node URLs to try
        auth_token: Optional authentication token
        timeout_seconds: Timeout for each connection attempt

    Returns:
        URL of the current leader, or None if not found

    Example:
        leader = await discover_p2p_leader_url([
            "http://192.168.1.100:8770",
            "http://192.168.1.101:8770",
            "http://node1.internal:8770",
        ])
        if leader:
            backend = P2PBackend(leader)
    """
    if not HAS_AIOHTTP:
        raise ImportError("aiohttp required for P2P leader discovery: pip install aiohttp")

    seeds = [_normalize_p2p_seed_url(s) for s in (seed_urls or [])]
    seeds = [s for s in seeds if s]
    if not seeds:
        return None

    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))

    def _candidate_base_urls(info: dict[str, Any]) -> list[str]:
        """Generate candidate URLs for the leader."""
        scheme = str(info.get("scheme") or "http").strip() or "http"
        host = str(info.get("host") or "").strip()
        rh = str(info.get("reported_host") or "").strip()
        try:
            port = int(info.get("port"))
        except (ValueError, TypeError):
            port = None
        try:
            rp = int(info.get("reported_port"))
        except (ValueError, TypeError):
            rp = None

        candidates: list[str] = []

        def _add(h: str, p: int | None) -> None:
            h = (h or "").strip()
            if not h or _is_loopback(h):
                return
            if not p or p <= 0:
                return
            base = f"{scheme}://{h}:{p}"
            if base not in candidates:
                candidates.append(base)

        # Prefer mesh addresses first when present (best NAT traversal)
        if rh and rp and _is_tailscale_ip(rh):
            _add(rh, rp)
        _add(host, port)
        if rh and rp and (rh != host or rp != port):
            _add(rh, rp)

        return candidates

    async def _first_reachable_base(
        session: aiohttp.ClientSession, candidates: list[str]
    ) -> str | None:
        """Find the first reachable URL from candidates."""
        for base in candidates:
            try:
                async with session.get(f"{base}/health") as resp:
                    if resp.status == 200:
                        return base
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                continue
        return None

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        for seed in seeds:
            try:
                async with session.get(f"{seed}/status") as resp:
                    if resp.status != 200:
                        continue
                    status = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                continue

            if not isinstance(status, dict):
                continue

            leader_id = (
                status.get("effective_leader_id") or status.get("leader_id") or ""
            ).strip()
            if not leader_id:
                continue

            self_block = (
                status.get("self") if isinstance(status.get("self"), dict) else {}
            )
            peers_block = (
                status.get("peers") if isinstance(status.get("peers"), dict) else {}
            )

            leader_info: dict[str, Any] = {}
            node_id = (status.get("node_id") or "").strip()
            if leader_id == node_id or leader_id == (
                self_block.get("node_id") or ""
            ).strip():
                leader_info = self_block
            else:
                leader_info = (
                    peers_block.get(leader_id)
                    if isinstance(peers_block.get(leader_id), dict)
                    else {}
                )

            host = (leader_info.get("host") or "").strip()
            scheme = (leader_info.get("scheme") or "http").strip() or "http"
            try:
                port_i = int(leader_info.get("port", None))
            except (ValueError, TypeError):
                port_i = None

            candidates = _candidate_base_urls(leader_info)
            reachable = await _first_reachable_base(session, candidates)
            if reachable:
                return reachable

            if host and port_i and not _is_loopback(host):
                # Back-compat fallback (may be unreachable in NAT/proxy setups)
                return f"{scheme}://{host}:{port_i}"

            # Fallback: use the seed itself if it claims to be leader
            if leader_id == node_id:
                return seed

    return None


async def get_p2p_backend(
    seed_urls: list[str] | None = None,
    leader_url: str | None = None,
    auth_token: str | None = None,
) -> P2PBackend:
    """Get a P2P backend instance with automatic leader discovery.

    Args:
        seed_urls: List of seed URLs for leader discovery
        leader_url: Direct leader URL (skips discovery if provided)
        auth_token: Optional authentication token

    Returns:
        Configured P2PBackend instance

    Raises:
        RuntimeError: If no leader can be found
    """
    if leader_url:
        return P2PBackend(leader_url, auth_token)

    # Try environment variable for seeds
    if not seed_urls:
        env_seeds = os.environ.get("RINGRIFT_P2P_SEEDS", "")
        seed_urls = [s.strip() for s in env_seeds.split(",") if s.strip()]

    if not seed_urls:
        raise RuntimeError("No P2P seed URLs provided and RINGRIFT_P2P_SEEDS not set")

    discovered = await discover_p2p_leader_url(seed_urls, auth_token=auth_token or "")
    if not discovered:
        raise RuntimeError(f"Could not discover P2P leader from seeds: {seed_urls}")

    return P2PBackend(discovered, auth_token)


# =============================================================================
# OrchestratorRegistry Integration (December 2025)
# =============================================================================

# Import OrchestratorRegistry for P2P leader sync
try:
    from app.coordination.orchestrator_registry import (
        OrchestratorInfo,
        OrchestratorRegistry,
        OrchestratorRole,
    )
    HAS_ORCHESTRATOR_REGISTRY = True
except ImportError:
    HAS_ORCHESTRATOR_REGISTRY = False
    OrchestratorRegistry = None
    OrchestratorRole = None
    OrchestratorInfo = None


def register_p2p_leader_in_registry(
    leader_url: str,
    leader_id: str = "",
) -> bool:
    """Register the discovered P2P leader in OrchestratorRegistry.

    This syncs the P2P leader election with the centralized orchestrator registry,
    enabling coordination between P2P-based and registry-based systems.

    Args:
        leader_url: URL of the P2P leader
        leader_id: Optional P2P node ID of the leader

    Returns:
        True if registered successfully
    """
    if not HAS_ORCHESTRATOR_REGISTRY:
        logger.debug("OrchestratorRegistry not available, skipping P2P leader registration")
        return False

    try:
        registry = OrchestratorRegistry.get_instance()

        # Try to acquire the P2P_LEADER role with leader info as metadata
        acquired = registry.acquire_role(
            OrchestratorRole.P2P_LEADER,
            force=False,  # Don't forcefully take over
            metadata={
                "p2p_leader_url": leader_url,
                "p2p_leader_id": leader_id,
                "discovered_at": time.time(),
            }
        )

        if acquired:
            logger.info(f"Registered P2P leader in registry: {leader_url}")
            return True
        else:
            # Just update the metadata if we can't acquire the role
            holder = registry.get_role_holder(OrchestratorRole.P2P_LEADER)
            if holder:
                logger.debug(f"P2P_LEADER role held by {holder.hostname}:{holder.pid}")
            return False

    except (RuntimeError, ValueError, KeyError, AttributeError) as e:
        logger.warning(f"Failed to register P2P leader in registry: {e}")
        return False


def get_p2p_leader_from_registry() -> str | None:
    """Get P2P leader URL from OrchestratorRegistry if available.

    Returns:
        P2P leader URL if found in registry, None otherwise
    """
    if not HAS_ORCHESTRATOR_REGISTRY:
        return None

    try:
        registry = OrchestratorRegistry.get_instance()
        holder = registry.get_role_holder(OrchestratorRole.P2P_LEADER)

        if holder and holder.is_alive():
            leader_url = holder.metadata.get("p2p_leader_url")
            if leader_url:
                logger.debug(f"Found P2P leader in registry: {leader_url}")
                return leader_url

    except (RuntimeError, ValueError, KeyError, AttributeError) as e:
        logger.debug(f"Could not get P2P leader from registry: {e}")

    return None


def sync_p2p_leader_heartbeat(leader_url: str, leader_id: str = "") -> None:
    """Update P2P leader heartbeat in registry.

    Call this periodically to keep the registry updated with leader status.

    Args:
        leader_url: URL of the P2P leader
        leader_id: Optional P2P node ID
    """
    if not HAS_ORCHESTRATOR_REGISTRY:
        return

    try:
        registry = OrchestratorRegistry.get_instance()

        # Update metadata with heartbeat
        registry.heartbeat(metadata_update={
            "p2p_leader_url": leader_url,
            "p2p_leader_id": leader_id,
            "last_sync": time.time(),
        })

    except (RuntimeError, ValueError, KeyError, AttributeError) as e:
        logger.debug(f"Failed to sync P2P leader heartbeat: {e}")


async def get_p2p_backend_with_registry(
    seed_urls: list[str] | None = None,
    leader_url: str | None = None,
    auth_token: str | None = None,
    use_registry: bool = True,
) -> P2PBackend:
    """Get a P2P backend with OrchestratorRegistry integration.

    This version first checks the registry for a known leader before
    falling back to seed discovery.

    Args:
        seed_urls: List of seed URLs for leader discovery
        leader_url: Direct leader URL (skips discovery if provided)
        auth_token: Optional authentication token
        use_registry: Whether to check registry first

    Returns:
        Configured P2PBackend instance

    Raises:
        RuntimeError: If no leader can be found
    """
    # Direct leader URL provided
    if leader_url:
        return P2PBackend(leader_url, auth_token)

    # Try registry first for cached leader
    if use_registry:
        registry_leader = get_p2p_leader_from_registry()
        if registry_leader:
            try:
                backend = P2PBackend(registry_leader, auth_token)
                # Verify it's still reachable
                if HAS_AIOHTTP:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{registry_leader}/health", timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                            if resp.status == 200:
                                logger.info(f"Using P2P leader from registry: {registry_leader}")
                                return backend
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as e:
                logger.debug(f"Registry leader not reachable: {e}")

    # Fall back to seed discovery
    if not seed_urls:
        env_seeds = os.environ.get("RINGRIFT_P2P_SEEDS", "")
        seed_urls = [s.strip() for s in env_seeds.split(",") if s.strip()]

    if not seed_urls:
        raise RuntimeError("No P2P seed URLs provided and RINGRIFT_P2P_SEEDS not set")

    discovered = await discover_p2p_leader_url(seed_urls, auth_token=auth_token or "")
    if not discovered:
        raise RuntimeError(f"Could not discover P2P leader from seeds: {seed_urls}")

    # Register discovered leader in registry
    register_p2p_leader_in_registry(discovered)

    return P2PBackend(discovered, auth_token)


__all__ = [
    "HAS_AIOHTTP",
    "HAS_ORCHESTRATOR_REGISTRY",
    "MAX_PHASE_WAIT_MINUTES",
    "P2P_DEFAULT_PORT",
    "P2P_HTTP_TIMEOUT",
    "P2P_JOB_POLL_INTERVAL",
    "P2PBackend",
    "P2PNodeInfo",
    "discover_p2p_leader_url",
    "get_p2p_backend",
    "get_p2p_backend_with_registry",
    "get_p2p_leader_from_registry",
    # OrchestratorRegistry integration
    "register_p2p_leader_in_registry",
    "sync_p2p_leader_heartbeat",
]
