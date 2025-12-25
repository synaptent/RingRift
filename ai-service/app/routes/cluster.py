"""Cluster Monitoring Routes for RingRift AI Service.

Provides admin/debug endpoints for cluster health and sync status:
- GET /cluster/status - Cluster health summary
- GET /cluster/sync/status - Sync daemon status
- GET /cluster/manifest - Manifest inspection
- POST /cluster/sync/trigger - Manual sync trigger
- GET /cluster/nodes - Node inventory list
- GET /cluster/nodes/{node_id} - Specific node details

Usage:
    from app.routes import cluster_router

    app.include_router(cluster_router, prefix="/api", tags=["cluster"])
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cluster", tags=["cluster"])

__all__ = ["router"]


# =============================================================================
# Response Models
# =============================================================================


class ClusterStatusResponse(BaseModel):
    """Cluster health summary response."""
    node_id: str = Field(..., description="Local node identifier")
    is_leader: bool = Field(..., description="Whether this node is the cluster leader")
    leader_id: str | None = Field(None, description="Current leader node ID")
    alive_peers: int = Field(..., description="Number of alive peers")
    total_peers: int = Field(..., description="Total number of configured peers")
    uptime_seconds: float = Field(..., description="P2P daemon uptime")
    job_count: int = Field(0, description="Active jobs on cluster")
    cluster_healthy: bool = Field(..., description="Overall cluster health status")


class SyncStatusResponse(BaseModel):
    """Sync daemon status response."""
    node_id: str = Field(..., description="Local node identifier")
    running: bool = Field(..., description="Whether sync daemon is running")
    last_sync_time: float = Field(0, description="Unix timestamp of last sync")
    pending_syncs: int = Field(0, description="Number of pending sync operations")
    games_synced: int = Field(0, description="Total games synced")
    models_synced: int = Field(0, description="Total models synced")
    bytes_transferred: int = Field(0, description="Total bytes transferred")
    errors: list[str] = Field(default_factory=list, description="Recent sync errors")


class ManifestSummaryResponse(BaseModel):
    """Manifest summary response."""
    node_id: str = Field(..., description="Local node identifier")
    total_games: int = Field(..., description="Total unique games in manifest")
    total_models: int = Field(..., description="Total models tracked")
    total_npz_files: int = Field(..., description="Total NPZ training files")
    games_by_config: dict[str, int] = Field(default_factory=dict, description="Game counts by config")
    under_replicated_count: int = Field(0, description="Games with < 2 replicas")
    nodes_with_data: int = Field(0, description="Nodes with registered data")


class NodeInventoryResponse(BaseModel):
    """Node inventory response."""
    node_id: str = Field(..., description="Node identifier")
    game_count: int = Field(0, description="Games on this node")
    model_count: int = Field(0, description="Models on this node")
    npz_count: int = Field(0, description="NPZ files on this node")
    disk_usage_percent: float | None = Field(None, description="Disk usage percentage")
    disk_free_gb: float | None = Field(None, description="Free disk space in GB")
    is_training: bool = Field(False, description="Whether node is training")
    is_selfplaying: bool = Field(False, description="Whether node is running selfplay")
    last_seen: float = Field(0, description="Unix timestamp of last contact")


class SyncTriggerRequest(BaseModel):
    """Request body for manual sync trigger."""
    board_type: str | None = Field(None, description="Optional board type filter")
    num_players: int | None = Field(None, description="Optional player count filter")
    priority: bool = Field(False, description="Use priority sync mode")


class SyncTriggerResponse(BaseModel):
    """Response for sync trigger."""
    triggered: bool = Field(..., description="Whether sync was triggered")
    message: str = Field(..., description="Status message")


# =============================================================================
# Helper Functions
# =============================================================================


def _get_p2p_status() -> dict[str, Any] | None:
    """Get P2P daemon status via HTTP."""
    try:
        import urllib.request
        import json

        url = "http://localhost:8770/status"
        with urllib.request.urlopen(url, timeout=2) as response:
            return json.loads(response.read().decode())
    except Exception:
        return None


def _get_cluster_manifest():
    """Get ClusterManifest instance."""
    try:
        from app.distributed.cluster_manifest import get_cluster_manifest
        return get_cluster_manifest()
    except ImportError:
        return None


def _get_sync_daemon():
    """Get AutoSyncDaemon instance."""
    try:
        from app.coordination.auto_sync_daemon import get_auto_sync_daemon
        return get_auto_sync_daemon()
    except ImportError:
        return None


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/status", response_model=ClusterStatusResponse)
async def get_cluster_status():
    """Get cluster health summary.

    Returns overall cluster health including:
    - Leader election status
    - Peer connectivity
    - Active job counts
    """
    p2p_status = _get_p2p_status()

    if p2p_status is None:
        # P2P daemon not running - return degraded status
        return ClusterStatusResponse(
            node_id=socket.gethostname(),
            is_leader=False,
            leader_id=None,
            alive_peers=0,
            total_peers=0,
            uptime_seconds=0,
            job_count=0,
            cluster_healthy=False,
        )

    alive_peers = sum(1 for p in p2p_status.get("peers", []) if p.get("alive"))
    total_peers = len(p2p_status.get("peers", []))

    return ClusterStatusResponse(
        node_id=p2p_status.get("node_id", socket.gethostname()),
        is_leader=p2p_status.get("is_leader", False),
        leader_id=p2p_status.get("leader_id"),
        alive_peers=alive_peers,
        total_peers=total_peers,
        uptime_seconds=p2p_status.get("uptime", 0),
        job_count=p2p_status.get("job_count", 0),
        cluster_healthy=alive_peers > 0 and p2p_status.get("leader_id") is not None,
    )


@router.get("/sync/status", response_model=SyncStatusResponse)
async def get_sync_status():
    """Get sync daemon status.

    Returns sync daemon metrics including:
    - Running state
    - Pending operations
    - Transfer statistics
    """
    daemon = _get_sync_daemon()

    if daemon is None:
        return SyncStatusResponse(
            node_id=socket.gethostname(),
            running=False,
        )

    status = daemon.get_status()
    stats = status.get("stats", {})

    return SyncStatusResponse(
        node_id=status.get("node_id", socket.gethostname()),
        running=status.get("running", False),
        last_sync_time=stats.get("last_sync_time", 0),
        pending_syncs=status.get("pending_games", 0),
        games_synced=stats.get("games_synced", 0),
        models_synced=stats.get("models_synced", 0),
        bytes_transferred=stats.get("bytes_transferred", 0),
        errors=stats.get("recent_errors", [])[:5],
    )


@router.get("/manifest", response_model=ManifestSummaryResponse)
async def get_manifest_summary():
    """Get cluster manifest summary.

    Returns manifest statistics including:
    - Data counts by type
    - Games by configuration
    - Replication health
    """
    manifest = _get_cluster_manifest()

    if manifest is None:
        return ManifestSummaryResponse(
            node_id=socket.gethostname(),
            total_games=0,
            total_models=0,
            total_npz_files=0,
        )

    stats = manifest.get_cluster_stats()

    return ManifestSummaryResponse(
        node_id=manifest.node_id,
        total_games=stats.get("total_games", 0),
        total_models=stats.get("total_models", 0),
        total_npz_files=stats.get("total_npz_files", 0),
        games_by_config=stats.get("games_by_config", {}),
        under_replicated_count=stats.get("under_replicated_count", 0),
        nodes_with_data=stats.get("nodes_with_data", 0),
    )


@router.get("/nodes", response_model=list[NodeInventoryResponse])
async def list_nodes():
    """List all nodes with their inventory.

    Returns inventory for all known nodes including:
    - Data counts
    - Disk usage
    - Activity status
    """
    manifest = _get_cluster_manifest()
    p2p_status = _get_p2p_status()

    nodes = []

    if manifest is not None:
        # Get nodes from manifest
        for node_id in manifest.get_all_node_ids():
            inventory = manifest.get_node_inventory(node_id)

            # Check if node is alive via P2P
            is_alive = False
            last_seen = 0
            if p2p_status:
                for peer in p2p_status.get("peers", []):
                    if peer.get("id") == node_id:
                        is_alive = peer.get("alive", False)
                        last_seen = peer.get("last_seen", 0)
                        break

            disk_usage = None
            disk_free = None
            if inventory.capacity:
                disk_usage = inventory.capacity.usage_percent
                disk_free = inventory.capacity.free_bytes / (1024 ** 3)

            nodes.append(NodeInventoryResponse(
                node_id=node_id,
                game_count=inventory.game_count,
                model_count=inventory.model_count,
                npz_count=inventory.npz_count,
                disk_usage_percent=disk_usage,
                disk_free_gb=disk_free,
                is_training=False,  # Would need job tracking
                is_selfplaying=False,  # Would need job tracking
                last_seen=last_seen,
            ))

    return nodes


@router.get("/nodes/{node_id}", response_model=NodeInventoryResponse)
async def get_node_details(node_id: str):
    """Get detailed inventory for a specific node.

    Args:
        node_id: Node identifier

    Returns:
        Detailed node inventory
    """
    manifest = _get_cluster_manifest()

    if manifest is None:
        raise HTTPException(status_code=503, detail="Manifest not available")

    try:
        inventory = manifest.get_node_inventory(node_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    disk_usage = None
    disk_free = None
    if inventory.capacity:
        disk_usage = inventory.capacity.usage_percent
        disk_free = inventory.capacity.free_bytes / (1024 ** 3)

    return NodeInventoryResponse(
        node_id=node_id,
        game_count=inventory.game_count,
        model_count=inventory.model_count,
        npz_count=inventory.npz_count,
        disk_usage_percent=disk_usage,
        disk_free_gb=disk_free,
        is_training=False,
        is_selfplaying=False,
        last_seen=time.time(),
    )


@router.post("/sync/trigger", response_model=SyncTriggerResponse)
async def trigger_sync(request: SyncTriggerRequest | None = None):
    """Manually trigger a sync operation.

    Optionally filter by board type and player count.

    Args:
        request: Optional sync parameters
    """
    daemon = _get_sync_daemon()

    if daemon is None:
        return SyncTriggerResponse(
            triggered=False,
            message="Sync daemon not available",
        )

    try:
        import asyncio

        # Start daemon if not running
        if not daemon._running:
            await daemon.start()

        # Trigger sync
        await daemon.trigger_sync()

        msg = "Sync triggered successfully"
        if request and request.board_type:
            msg += f" for {request.board_type}"
            if request.num_players:
                msg += f"_{request.num_players}p"

        return SyncTriggerResponse(
            triggered=True,
            message=msg,
        )

    except Exception as e:
        logger.error(f"Sync trigger failed: {e}")
        return SyncTriggerResponse(
            triggered=False,
            message=f"Sync trigger failed: {str(e)}",
        )


@router.get("/health")
async def cluster_health_check():
    """Quick cluster health check.

    Returns 200 if cluster is healthy, 503 otherwise.
    Suitable for load balancer health checks.
    """
    p2p_status = _get_p2p_status()

    if p2p_status is None:
        raise HTTPException(status_code=503, detail="P2P daemon not running")

    alive_peers = sum(1 for p in p2p_status.get("peers", []) if p.get("alive"))

    if alive_peers == 0:
        raise HTTPException(status_code=503, detail="No alive peers")

    if p2p_status.get("leader_id") is None:
        raise HTTPException(status_code=503, detail="No leader elected")

    return {
        "status": "healthy",
        "leader": p2p_status.get("leader_id"),
        "alive_peers": alive_peers,
    }


@router.get("/config")
async def get_cluster_config():
    """Get cluster configuration summary.

    Returns configured hosts and sync settings.
    """
    try:
        from pathlib import Path
        import yaml

        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "distributed_hosts.yaml"

        if not config_path.exists():
            return {"configured": False, "hosts": [], "sync_routing": {}}

        with open(config_path) as f:
            config = yaml.safe_load(f)

        hosts = list(config.get("hosts", {}).keys())
        sync_routing = config.get("sync_routing", {})

        return {
            "configured": True,
            "host_count": len(hosts),
            "hosts": hosts,
            "priority_hosts": sync_routing.get("priority_hosts", []),
            "max_disk_usage_percent": sync_routing.get("max_disk_usage_percent", 70),
        }

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {"configured": False, "error": str(e)}
