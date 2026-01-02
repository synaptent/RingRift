"""P2P Integration Facade for Coordination Modules (December 2025).

This module provides a unified interface for coordination modules to interact
with the P2P orchestrator cluster. It consolidates scattered P2P communication
patterns into a single, well-tested facade.

Usage:
    from app.coordination.p2p_integration import (
        get_p2p_status,
        get_p2p_nodes,
        submit_p2p_job,
        is_p2p_available,
    )

    # Check if P2P is available
    if await is_p2p_available():
        status = await get_p2p_status()
        nodes = await get_p2p_nodes()

    # Submit a job
    result = await submit_p2p_job({
        "type": "selfplay",
        "board_type": "hex8",
        "num_players": 2,
        "num_games": 1000,
    })

This replaces the non-existent app.distributed.p2p_orchestrator import pattern
that was causing ImportError in idle_resource_daemon.py and node_recovery_daemon.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# P2P configuration
P2P_DEFAULT_TIMEOUT = 10.0  # seconds
P2P_STATUS_CACHE_TTL = 5.0  # seconds
P2P_MAX_RETRIES = 2

# Try to import P2PBackend
try:
    from app.coordination.p2p_backend import (
        P2PBackend,
        discover_p2p_leader_url,
        get_p2p_backend,
        HAS_AIOHTTP,
    )
    HAS_P2P_BACKEND = True
except ImportError:
    HAS_P2P_BACKEND = False
    HAS_AIOHTTP = False
    P2PBackend = None  # type: ignore
    discover_p2p_leader_url = None  # type: ignore
    get_p2p_backend = None  # type: ignore

# Dec 2025: Use centralized P2P port config
try:
    from app.config.cluster_config import get_p2p_port
except ImportError:
    def get_p2p_port() -> int:
        return int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))

# Try to import aiohttp directly for raw HTTP calls
if not HAS_AIOHTTP:
    try:
        import aiohttp
        HAS_AIOHTTP = True
    except ImportError:
        aiohttp = None  # type: ignore


def _get_default_p2p_port() -> int:
    """Get default P2P port - wrapper for field default_factory."""
    return get_p2p_port()


@dataclass
class P2PNodeStatus:
    """Node status from P2P cluster."""

    node_id: str
    host: str
    port: int = field(default_factory=_get_default_p2p_port)
    is_alive: bool = True
    is_healthy: bool = True
    gpu_utilization: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    active_jobs: int = 0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    provider: str = "unknown"
    last_seen: float = 0.0
    has_gpu: bool = False
    gpu_name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> P2PNodeStatus:
        """Create from P2P status dict."""
        return cls(
            node_id=data.get("node_id", ""),
            host=data.get("host", ""),
            port=data.get("port", get_p2p_port()),  # Dec 2025: Use centralized config
            is_alive=data.get("is_alive", True),
            is_healthy=data.get("is_healthy", True),
            gpu_utilization=data.get("gpu_utilization", 0.0),
            gpu_memory_total_gb=data.get("gpu_memory_total", data.get("gpu_memory_total_gb", 0.0)),
            gpu_memory_used_gb=data.get("gpu_memory_used", data.get("gpu_memory_used_gb", 0.0)),
            cpu_percent=data.get("cpu_percent", 0.0),
            memory_percent=data.get("memory_percent", 0.0),
            disk_percent=data.get("disk_percent", 0.0),
            active_jobs=data.get("active_jobs", 0),
            selfplay_jobs=data.get("selfplay_jobs", 0),
            training_jobs=data.get("training_jobs", 0),
            provider=data.get("provider", "unknown"),
            last_seen=data.get("last_seen", time.time()),
            has_gpu=data.get("has_gpu", False),
            gpu_name=data.get("gpu_name", ""),
        )


@dataclass
class P2PJobResult:
    """Result of a P2P job submission."""

    success: bool
    job_id: str = ""
    error: str = ""
    details: dict[str, Any] | None = None


@dataclass
class DistributedLockResult:
    """Result of a distributed lock operation.

    January 2026 (Sprint 3): Added for global training lock support.
    """

    acquired: bool
    lock_name: str
    holder: str | None = None
    error: str = ""
    timestamp: float = 0.0


# Cached status to avoid hammering P2P orchestrator
_status_cache: dict[str, Any] = {}
_status_cache_time: float = 0.0
_backend_instance: P2PBackend | None = None


def _get_p2p_seeds() -> list[str]:
    """Get P2P seed URLs from environment."""
    seeds_str = os.environ.get("RINGRIFT_P2P_SEEDS", "")
    if seeds_str:
        return [s.strip() for s in seeds_str.split(",") if s.strip()]

    # Dec 2025: Use centralized P2P URL helper
    from app.config.ports import get_local_p2p_url
    default_url = get_local_p2p_url()
    default_seeds = os.environ.get(
        "RINGRIFT_P2P_DEFAULT_SEEDS",
        default_url
    )
    return [s.strip() for s in default_seeds.split(",") if s.strip()]


async def _get_backend() -> P2PBackend | None:
    """Get or create P2P backend instance."""
    global _backend_instance

    if not HAS_P2P_BACKEND:
        return None

    if _backend_instance is not None:
        # Check if still healthy
        try:
            health_result = await _backend_instance.health_check()
            # December 2025: health_check now returns HealthCheckResult
            is_healthy = health_result.healthy if hasattr(health_result, "healthy") else bool(health_result)
            if is_healthy:
                return _backend_instance
        except (asyncio.TimeoutError, OSError, RuntimeError):
            pass

    # Create new backend
    seeds = _get_p2p_seeds()
    if not seeds:
        return None

    try:
        _backend_instance = await get_p2p_backend(seed_urls=seeds)
        return _backend_instance
    except (RuntimeError, ImportError) as e:
        logger.debug(f"Failed to create P2P backend: {e}")
        return None


async def is_p2p_available() -> bool:
    """Check if P2P orchestrator is available.

    Returns:
        True if P2P is available and reachable
    """
    if not HAS_AIOHTTP:
        return False

    backend = await _get_backend()
    if backend is None:
        return False

    try:
        health_result = await backend.health_check()
        # December 2025: health_check now returns HealthCheckResult
        return health_result.healthy if hasattr(health_result, "healthy") else bool(health_result)
    except (asyncio.TimeoutError, OSError, RuntimeError):
        return False


async def get_p2p_status(
    use_cache: bool = True,
    cache_ttl: float = P2P_STATUS_CACHE_TTL,
) -> dict[str, Any] | None:
    """Get P2P cluster status.

    This is a replacement for the non-existent app.distributed.p2p_orchestrator
    get_p2p_orchestrator().get_status() pattern.

    Args:
        use_cache: Whether to use cached status
        cache_ttl: Cache TTL in seconds

    Returns:
        Cluster status dict or None if unavailable
    """
    global _status_cache, _status_cache_time

    # Check cache
    if use_cache and _status_cache:
        if time.time() - _status_cache_time < cache_ttl:
            return _status_cache

    backend = await _get_backend()
    if backend is None:
        return None

    try:
        status = await backend.get_cluster_status()
        _status_cache = status
        _status_cache_time = time.time()
        return status
    except (asyncio.TimeoutError, OSError, RuntimeError) as e:
        logger.debug(f"Failed to get P2P status: {e}")
        return None


async def get_p2p_nodes() -> list[P2PNodeStatus]:
    """Get all nodes from P2P cluster.

    Returns:
        List of P2PNodeStatus objects
    """
    status = await get_p2p_status()
    if status is None:
        return []

    nodes: list[P2PNodeStatus] = []

    # Parse alive_peers
    alive_peers = status.get("alive_peers", [])
    if isinstance(alive_peers, list):
        for peer in alive_peers:
            if isinstance(peer, dict):
                nodes.append(P2PNodeStatus.from_dict(peer))
            elif isinstance(peer, str):
                # Just a node ID string
                nodes.append(P2PNodeStatus(node_id=peer, host=""))

    return nodes


async def get_p2p_alive_nodes() -> list[P2PNodeStatus]:
    """Get all alive nodes from P2P cluster.

    Returns:
        List of alive P2PNodeStatus objects
    """
    return [n for n in await get_p2p_nodes() if n.is_alive]


async def get_p2p_healthy_nodes() -> list[P2PNodeStatus]:
    """Get all healthy nodes from P2P cluster.

    Returns:
        List of healthy P2PNodeStatus objects
    """
    return [n for n in await get_p2p_nodes() if n.is_alive and n.is_healthy]


async def get_p2p_gpu_nodes() -> list[P2PNodeStatus]:
    """Get all healthy GPU nodes from P2P cluster.

    Returns:
        List of healthy GPU nodes
    """
    return [n for n in await get_p2p_healthy_nodes() if n.has_gpu]


async def submit_p2p_job(job_spec: dict[str, Any]) -> P2PJobResult:
    """Submit a job to the P2P cluster work queue.

    This is a replacement for the non-existent app.distributed.p2p_orchestrator
    get_p2p_orchestrator().submit_job() pattern.

    IMPORTANT: The job_spec format must match what /work/add expects:

    Args:
        job_spec: Job specification dict with:
            - work_type: str - Job type ("selfplay", "training", "tournament", "gpu_cmaes")
            - priority: int - Priority 0-100, higher = more urgent (default: 50)
            - config: dict - Job-specific parameters:
                - board_type: Board type (e.g., "hex8", "square8")
                - num_players: Number of players (2, 3, or 4)
                - num_games: Number of games (for selfplay)
                - engine_mode: Engine mode (e.g., "gumbel-mcts", "heuristic-only")
                - target_node: Optional target node ID

    Example:
        job_spec = {
            "work_type": "selfplay",
            "priority": 60,
            "config": {
                "board_type": "hex8",
                "num_players": 2,
                "num_games": 500,
                "engine_mode": "gumbel-mcts",
            }
        }

    Returns:
        P2PJobResult with success status and job ID
    """
    if not HAS_AIOHTTP:
        return P2PJobResult(success=False, error="aiohttp not available")

    backend = await _get_backend()
    if backend is None:
        return P2PJobResult(success=False, error="P2P backend not available")

    try:
        # Use the backend's HTTP session to submit job
        session = await backend._get_session()
        async with session.post(
            f"{backend.leader_url}/work/add",
            json=job_spec,
        ) as resp:
            result = await resp.json()
            if result.get("success"):
                return P2PJobResult(
                    success=True,
                    job_id=result.get("job_id", ""),
                    details=result,
                )
            else:
                return P2PJobResult(
                    success=False,
                    error=result.get("error", "Unknown error"),
                    details=result,
                )

    except asyncio.TimeoutError:
        return P2PJobResult(success=False, error="Request timeout")
    except (OSError, RuntimeError) as e:
        return P2PJobResult(success=False, error=str(e))


async def cancel_p2p_job(job_id: str) -> P2PJobResult:
    """Cancel a job in the P2P cluster.

    December 2025: Added as part of P2P facade expansion.

    Args:
        job_id: ID of the job to cancel

    Returns:
        P2PJobResult with success status
    """
    if not HAS_AIOHTTP:
        return P2PJobResult(success=False, error="aiohttp not available")

    backend = await _get_backend()
    if backend is None:
        return P2PJobResult(success=False, error="P2P backend not available")

    try:
        session = await backend._get_session()
        async with session.post(
            f"{backend.leader_url}/work/cancel",
            json={"job_id": job_id},
        ) as resp:
            result = await resp.json()
            return P2PJobResult(
                success=result.get("success", False),
                job_id=job_id,
                error=result.get("error", ""),
                details=result,
            )
    except asyncio.TimeoutError:
        return P2PJobResult(success=False, job_id=job_id, error="Request timeout")
    except (OSError, RuntimeError) as e:
        return P2PJobResult(success=False, job_id=job_id, error=str(e))


async def dispatch_selfplay_direct(
    target_node: str,
    host: str,
    port: int,
    board_type: str,
    num_players: int,
    num_games: int,
    engine_mode: str = "gumbel-mcts",
    engine_extra_args: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> P2PJobResult:
    """Dispatch selfplay directly to a node's /selfplay/start endpoint.

    This bypasses the work queue for immediate execution on idle nodes.
    Used by IdleResourceDaemon for responsive selfplay spawning.

    December 2025: Added to fix autonomous selfplay dispatch. The work queue
    model (submit_job -> /work/add) doesn't work well because WorkerPullLoop
    only pulls when nodes are idle (training_jobs == 0), but training nodes
    are always busy.

    Args:
        target_node: Node ID for logging/tracking
        host: Node HTTP host
        port: Node HTTP port (default P2P port is 8770)
        board_type: Board type (e.g., "hex8", "square8")
        num_players: Number of players (2, 3, or 4)
        num_games: Number of games to run
        engine_mode: Engine mode (e.g., "gumbel-mcts", "heuristic-only")
        engine_extra_args: Optional extra arguments for engine
        timeout: Request timeout in seconds

    Returns:
        P2PJobResult with success status and job_id
    """
    if not HAS_AIOHTTP:
        return P2PJobResult(success=False, error="aiohttp not available")

    url = f"http://{host}:{port}/selfplay/start"
    payload = {
        "board_type": board_type,
        "num_players": num_players,
        "num_games": num_games,
        "engine_mode": engine_mode,
    }
    if engine_extra_args:
        payload["engine_extra_args"] = engine_extra_args

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                result = await resp.json()
                if result.get("success"):
                    logger.debug(
                        f"[dispatch_selfplay_direct] Started selfplay on {target_node}: "
                        f"{board_type}_{num_players}p, {num_games} games, job_id={result.get('job_id')}"
                    )
                    return P2PJobResult(
                        success=True,
                        job_id=result.get("job_id", ""),
                        details=result,
                    )
                else:
                    return P2PJobResult(
                        success=False,
                        error=result.get("error", "Unknown error"),
                        details=result,
                    )
    except asyncio.TimeoutError:
        logger.debug(f"[dispatch_selfplay_direct] Timeout dispatching to {target_node}")
        return P2PJobResult(success=False, error=f"Timeout connecting to {target_node}")
    except aiohttp.ClientError as e:
        logger.debug(f"[dispatch_selfplay_direct] HTTP error for {target_node}: {e}")
        return P2PJobResult(success=False, error=f"HTTP error: {e}")
    except (OSError, RuntimeError) as e:
        logger.debug(f"[dispatch_selfplay_direct] Error for {target_node}: {e}")
        return P2PJobResult(success=False, error=str(e))


async def get_p2p_job_status(job_id: str) -> dict[str, Any] | None:
    """Get status of a specific job in the P2P cluster.

    December 2025: Added as part of P2P facade expansion.

    Args:
        job_id: ID of the job to query

    Returns:
        Job status dict or None if not found
    """
    if not HAS_AIOHTTP:
        return None

    backend = await _get_backend()
    if backend is None:
        return None

    try:
        session = await backend._get_session()
        async with session.get(
            f"{backend.leader_url}/work/status/{job_id}",
        ) as resp:
            if resp.status == 404:
                return None
            result = await resp.json()
            return result
    except asyncio.TimeoutError:
        logger.debug(f"Timeout getting job status for {job_id}")
        return None
    except (OSError, RuntimeError) as e:
        logger.debug(f"Error getting job status for {job_id}: {e}")
        return None


async def list_p2p_jobs(
    status_filter: str | None = None,
    job_type: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List jobs in the P2P cluster.

    December 2025: Added as part of P2P facade expansion.

    Args:
        status_filter: Optional status to filter by (pending, running, completed, failed)
        job_type: Optional job type to filter by (selfplay, training, tournament)
        limit: Maximum number of jobs to return

    Returns:
        List of job status dicts
    """
    if not HAS_AIOHTTP:
        return []

    backend = await _get_backend()
    if backend is None:
        return []

    try:
        params: dict[str, Any] = {"limit": limit}
        if status_filter:
            params["status"] = status_filter
        if job_type:
            params["type"] = job_type

        session = await backend._get_session()
        async with session.get(
            f"{backend.leader_url}/work/list",
            params=params,
        ) as resp:
            result = await resp.json()
            return result.get("jobs", [])
    except asyncio.TimeoutError:
        logger.debug("Timeout listing P2P jobs")
        return []
    except (OSError, RuntimeError) as e:
        logger.debug(f"Error listing P2P jobs: {e}")
        return []


async def update_p2p_lease(node_id: str | None = None) -> bool:
    """Update the P2P lease/heartbeat for this node.

    December 2025: Added as part of P2P facade expansion.
    This is used to signal to the cluster that this node is still alive.

    Args:
        node_id: Optional node ID (uses local node if not specified)

    Returns:
        True if lease was updated successfully
    """
    if not HAS_AIOHTTP:
        return False

    backend = await _get_backend()
    if backend is None:
        return False

    try:
        # If no node_id specified, try to get from environment
        if node_id is None:
            node_id = os.environ.get("RINGRIFT_NODE_ID", "")
            if not node_id:
                # Try to get from status
                status = await get_p2p_status()
                if status:
                    node_id = status.get("node_id", "")

        if not node_id:
            logger.debug("No node_id available for lease update")
            return False

        session = await backend._get_session()
        async with session.post(
            f"{backend.leader_url}/lease/update",
            json={"node_id": node_id, "timestamp": time.time()},
        ) as resp:
            result = await resp.json()
            return result.get("success", False)
    except asyncio.TimeoutError:
        logger.debug("Timeout updating P2P lease")
        return False
    except (OSError, RuntimeError) as e:
        logger.debug(f"Error updating P2P lease: {e}")
        return False


async def submit_batch_p2p_jobs(job_specs: list[dict[str, Any]]) -> list[P2PJobResult]:
    """Submit multiple jobs to the P2P cluster in a batch.

    December 2025: Added as part of P2P facade expansion.
    This is more efficient than submitting jobs one at a time.

    Args:
        job_specs: List of job specification dicts

    Returns:
        List of P2PJobResult for each job
    """
    if not job_specs:
        return []

    if not HAS_AIOHTTP:
        return [P2PJobResult(success=False, error="aiohttp not available") for _ in job_specs]

    backend = await _get_backend()
    if backend is None:
        return [P2PJobResult(success=False, error="P2P backend not available") for _ in job_specs]

    try:
        session = await backend._get_session()
        async with session.post(
            f"{backend.leader_url}/work/add_batch",
            json={"jobs": job_specs},
        ) as resp:
            result = await resp.json()

            if result.get("success"):
                # Parse individual job results
                job_results = result.get("results", [])
                return [
                    P2PJobResult(
                        success=jr.get("success", False),
                        job_id=jr.get("job_id", ""),
                        error=jr.get("error", ""),
                        details=jr,
                    )
                    for jr in job_results
                ]
            else:
                # Batch failed entirely
                error_msg = result.get("error", "Batch submission failed")
                return [P2PJobResult(success=False, error=error_msg) for _ in job_specs]

    except asyncio.TimeoutError:
        return [P2PJobResult(success=False, error="Request timeout") for _ in job_specs]
    except (OSError, RuntimeError) as e:
        return [P2PJobResult(success=False, error=str(e)) for _ in job_specs]


async def get_p2p_leader_id() -> str | None:
    """Get the current P2P leader ID.

    Returns:
        Leader node ID or None if not available
    """
    status = await get_p2p_status()
    if status is None:
        return None

    return status.get("leader_id") or status.get("effective_leader_id")


async def get_p2p_leader_url() -> str | None:
    """Get the current P2P leader URL.

    Returns:
        Leader URL or None if not available
    """
    backend = await _get_backend()
    if backend is None:
        return None
    return backend.leader_url


def clear_p2p_cache() -> None:
    """Clear the P2P status cache."""
    global _status_cache, _status_cache_time
    _status_cache = {}
    _status_cache_time = 0.0


async def close_p2p_connection() -> None:
    """Close the P2P backend connection."""
    global _backend_instance
    if _backend_instance is not None:
        await _backend_instance.close()
        _backend_instance = None


# =============================================================================
# Distributed Lock Functions (January 2026 - Sprint 3)
# =============================================================================


async def acquire_distributed_lock(
    lock_name: str,
    timeout_seconds: float = 300.0,
) -> DistributedLockResult:
    """Acquire a distributed lock via P2P Raft consensus.

    January 2026 (Sprint 3): Added for global training lock support.
    Prevents duplicate training jobs across cluster nodes.

    Args:
        lock_name: Name of the lock (e.g., "training:hex8:2")
        timeout_seconds: Lock timeout/TTL in seconds (default: 300)

    Returns:
        DistributedLockResult with acquired=True if lock was acquired

    Example:
        lock_name = f"training:{board_type}:{num_players}"
        result = await acquire_distributed_lock(lock_name)
        if result.acquired:
            try:
                await run_training(...)
            finally:
                await release_distributed_lock(lock_name)
    """
    if not HAS_AIOHTTP:
        return DistributedLockResult(
            acquired=False,
            lock_name=lock_name,
            error="aiohttp not available",
        )

    try:
        leader_url = await get_p2p_leader_url()
        if not leader_url:
            return DistributedLockResult(
                acquired=False,
                lock_name=lock_name,
                error="P2P leader not available",
            )

        # POST to /raft/lock/{name}
        import aiohttp

        url = f"{leader_url}/raft/lock/{lock_name}"
        payload = {"timeout": timeout_seconds}

        # Get auth token if available
        auth_token = os.environ.get("RINGRIFT_P2P_AUTH_TOKEN", "")
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                data = await response.json()

                if response.status == 200:
                    return DistributedLockResult(
                        acquired=data.get("acquired", False),
                        lock_name=lock_name,
                        holder=data.get("holder"),
                        timestamp=data.get("timestamp", time.time()),
                    )
                else:
                    return DistributedLockResult(
                        acquired=False,
                        lock_name=lock_name,
                        error=data.get("error", f"HTTP {response.status}"),
                    )

    except asyncio.TimeoutError:
        logger.warning(f"[P2P] Timeout acquiring lock {lock_name}")
        return DistributedLockResult(
            acquired=False,
            lock_name=lock_name,
            error="timeout",
        )
    except Exception as e:
        logger.error(f"[P2P] Error acquiring lock {lock_name}: {e}")
        return DistributedLockResult(
            acquired=False,
            lock_name=lock_name,
            error=str(e),
        )


async def release_distributed_lock(lock_name: str) -> bool:
    """Release a distributed lock via P2P Raft consensus.

    January 2026 (Sprint 3): Added for global training lock support.

    Args:
        lock_name: Name of the lock to release

    Returns:
        True if lock was released successfully
    """
    if not HAS_AIOHTTP:
        logger.warning(f"[P2P] Cannot release lock {lock_name}: aiohttp not available")
        return False

    try:
        leader_url = await get_p2p_leader_url()
        if not leader_url:
            logger.warning(f"[P2P] Cannot release lock {lock_name}: no leader")
            return False

        # DELETE to /raft/lock/{name}
        import aiohttp

        url = f"{leader_url}/raft/lock/{lock_name}"

        # Get auth token if available
        auth_token = os.environ.get("RINGRIFT_P2P_AUTH_TOKEN", "")
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.delete(url, headers=headers) as response:
                data = await response.json()
                released = data.get("released", False)

                if released:
                    logger.debug(f"[P2P] Released lock {lock_name}")
                else:
                    logger.warning(
                        f"[P2P] Failed to release lock {lock_name}: {data.get('error', 'unknown')}"
                    )

                return released

    except asyncio.TimeoutError:
        logger.warning(f"[P2P] Timeout releasing lock {lock_name}")
        return False
    except Exception as e:
        logger.error(f"[P2P] Error releasing lock {lock_name}: {e}")
        return False


class DistributedLockContext:
    """Async context manager for distributed locks.

    January 2026 (Sprint 3): Safe lock management with auto-release.

    Usage:
        async with DistributedLockContext("training:hex8:2") as lock:
            if lock.acquired:
                await run_training(...)
            else:
                logger.info("Lock not acquired, skipping")
    """

    def __init__(self, lock_name: str, timeout_seconds: float = 300.0):
        self.lock_name = lock_name
        self.timeout_seconds = timeout_seconds
        self.result: DistributedLockResult | None = None

    async def __aenter__(self) -> DistributedLockResult:
        self.result = await acquire_distributed_lock(
            self.lock_name,
            timeout_seconds=self.timeout_seconds,
        )
        return self.result

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.result and self.result.acquired:
            await release_distributed_lock(self.lock_name)


def with_training_lock(config_key: str, timeout_seconds: float = 300.0) -> DistributedLockContext:
    """Create a distributed lock context for training operations.

    January 2026 (Sprint 3): Convenience function for training locks.

    Args:
        config_key: Configuration key (e.g., "hex8_2p")
        timeout_seconds: Lock TTL (default: 300s = 5 minutes)

    Returns:
        DistributedLockContext for use with async with

    Example:
        async with with_training_lock("hex8_2p") as lock:
            if lock.acquired:
                await trigger_training("hex8_2p")
    """
    lock_name = f"training:{config_key.replace('_', ':')}"
    return DistributedLockContext(lock_name, timeout_seconds)


# =============================================================================
# Compatibility shim for broken imports
# =============================================================================


class P2POrchestratorShim:
    """Compatibility shim for app.distributed.p2p_orchestrator.

    This provides the interface that idle_resource_daemon.py and
    node_recovery_daemon.py expect from get_p2p_orchestrator().
    """

    async def get_status(self) -> dict[str, Any] | None:
        """Get cluster status (shim for p2p.get_status())."""
        return await get_p2p_status()

    async def submit_job(self, job_spec: dict[str, Any]) -> dict[str, Any]:
        """Submit a job (shim for p2p.submit_job())."""
        result = await submit_p2p_job(job_spec)
        return {
            "success": result.success,
            "job_id": result.job_id,
            "error": result.error,
            **(result.details or {}),
        }


_shim_instance: P2POrchestratorShim | None = None


class P2PIntegration:
    """P2P Integration daemon for DaemonManager (December 2025).

    This class provides the daemon interface (start/stop/health_check) needed
    by daemon_runners.py for the P2P_BACKEND daemon type.

    It wraps the P2P status checking functionality and provides health monitoring
    for the P2P cluster connection.
    """

    def __init__(self):
        """Initialize P2P integration."""
        self._running = False
        self._last_status_check: float = 0.0
        self._last_status: dict[str, Any] | None = None
        self._error_count = 0
        self._check_interval = 30.0  # Check status every 30 seconds
        self._check_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the P2P integration daemon."""
        if self._running:
            logger.debug("[P2PIntegration] Already running")
            return

        logger.info("[P2PIntegration] Starting P2P integration daemon")
        self._running = True
        self._error_count = 0

        # Start background status check loop
        self._check_task = asyncio.create_task(self._status_check_loop())
        logger.info("[P2PIntegration] Started")

    async def stop(self) -> None:
        """Stop the P2P integration daemon."""
        if not self._running:
            return

        logger.info("[P2PIntegration] Stopping P2P integration daemon")
        self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None

        logger.info("[P2PIntegration] Stopped")

    async def _status_check_loop(self) -> None:
        """Background loop to periodically check P2P status."""
        while self._running:
            try:
                status = await get_p2p_status()
                self._last_status = status
                self._last_status_check = time.time()
                if status:
                    self._error_count = 0
                else:
                    self._error_count += 1
            except asyncio.CancelledError:
                break
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"[P2PIntegration] Status check failed: {e}")
                self._error_count += 1
            except Exception as e:
                logger.error(f"[P2PIntegration] Unexpected error in status check: {e}")
                self._error_count += 1

            await asyncio.sleep(self._check_interval)

    def health_check(self) -> "HealthCheckResult":
        """Check health of P2P integration.

        Returns:
            HealthCheckResult with health status
        """
        from app.coordination.protocols import HealthCheckResult, CoordinatorStatus

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="P2P integration not running",
            )

        # Check if we've had successful status checks recently
        time_since_check = time.time() - self._last_status_check
        if time_since_check > self._check_interval * 3:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"No P2P status check in {time_since_check:.0f}s",
                details={"error_count": self._error_count},
            )

        if self._error_count > 5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"P2P status checks failing ({self._error_count} errors)",
                details={"error_count": self._error_count},
            )

        leader_id = self._last_status.get("leader_id", "unknown") if self._last_status else "unknown"
        alive_count = self._last_status.get("alive_peers", 0) if self._last_status else 0

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Connected to P2P cluster (leader: {leader_id}, peers: {alive_count})",
            details={
                "leader_id": leader_id,
                "alive_peers": alive_count,
                "error_count": self._error_count,
                "last_check": self._last_status_check,
            },
        )


def get_p2p_orchestrator() -> P2POrchestratorShim | None:
    """Get P2P orchestrator shim.

    This is the compatibility function that idle_resource_daemon.py and
    node_recovery_daemon.py were trying to import from app.distributed.p2p_orchestrator.

    Returns:
        P2POrchestratorShim instance or None if P2P is not available
    """
    global _shim_instance

    if not HAS_P2P_BACKEND:
        return None

    if _shim_instance is None:
        _shim_instance = P2POrchestratorShim()

    return _shim_instance


__all__ = [
    # Status functions
    "get_p2p_status",
    "get_p2p_nodes",
    "get_p2p_alive_nodes",
    "get_p2p_healthy_nodes",
    "get_p2p_gpu_nodes",
    "get_p2p_leader_id",
    "get_p2p_leader_url",
    # Job functions
    "submit_p2p_job",
    "cancel_p2p_job",  # December 2025
    "get_p2p_job_status",  # December 2025
    "list_p2p_jobs",  # December 2025
    "submit_batch_p2p_jobs",  # December 2025
    # Lease functions
    "update_p2p_lease",  # December 2025
    # Distributed lock functions (January 2026 - Sprint 3)
    "acquire_distributed_lock",
    "release_distributed_lock",
    "with_training_lock",
    "DistributedLockContext",
    "DistributedLockResult",
    # Utility functions
    "is_p2p_available",
    "clear_p2p_cache",
    "close_p2p_connection",
    # Data classes
    "P2PNodeStatus",
    "P2PJobResult",
    # Compatibility shim
    "get_p2p_orchestrator",
    "P2POrchestratorShim",
    # Daemon interface
    "P2PIntegration",
]
