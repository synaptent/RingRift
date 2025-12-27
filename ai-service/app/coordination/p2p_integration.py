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
from dataclasses import dataclass
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

# Try to import aiohttp directly for raw HTTP calls
if not HAS_AIOHTTP:
    try:
        import aiohttp
        HAS_AIOHTTP = True
    except ImportError:
        aiohttp = None  # type: ignore


@dataclass
class P2PNodeStatus:
    """Node status from P2P cluster."""

    node_id: str
    host: str
    port: int = 8770
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
            port=data.get("port", 8770),
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


# Cached status to avoid hammering P2P orchestrator
_status_cache: dict[str, Any] = {}
_status_cache_time: float = 0.0
_backend_instance: P2PBackend | None = None


def _get_p2p_seeds() -> list[str]:
    """Get P2P seed URLs from environment."""
    seeds_str = os.environ.get("RINGRIFT_P2P_SEEDS", "")
    if seeds_str:
        return [s.strip() for s in seeds_str.split(",") if s.strip()]

    # Default seeds based on common P2P ports (Dec 2025: consistent with RINGRIFT_P2P_URL)
    default_url = os.environ.get("RINGRIFT_P2P_URL", "http://localhost:8770")
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
            if await _backend_instance.health_check():
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
        return await backend.health_check()
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
    """Submit a job to the P2P cluster.

    This is a replacement for the non-existent app.distributed.p2p_orchestrator
    get_p2p_orchestrator().submit_job() pattern.

    Args:
        job_spec: Job specification dict with:
            - type: Job type (selfplay, training, etc.)
            - board_type: Board type
            - num_players: Number of players
            - num_games: Number of games (for selfplay)
            - target_node: Optional target node ID
            - engine_mode: Engine mode (e.g., "gumbel-mcts")

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
]
