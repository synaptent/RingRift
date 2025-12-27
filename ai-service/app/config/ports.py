"""Centralized port configuration for RingRift AI Service.

All network ports should be defined here as the single source of truth.
Import from this module instead of hardcoding port numbers.

Usage:
    from app.config.ports import P2P_DEFAULT_PORT, HEALTH_CHECK_PORT

    # Build URLs
    url = f"http://localhost:{P2P_DEFAULT_PORT}/status"
"""

# =============================================================================
# P2P Cluster Communication
# =============================================================================

# Main P2P orchestrator port for cluster coordination, leader election,
# job dispatch, and status queries
P2P_DEFAULT_PORT = 8770

# Gossip protocol port for P2P data replication and peer discovery
GOSSIP_PORT = 8771

# SWIM membership protocol port for gossip-based failure detection
# Used by SwimMembershipManager for O(1) bandwidth peer health monitoring
SWIM_PORT = 7947

# =============================================================================
# Node Health & Monitoring
# =============================================================================

# Health check port for node monitoring (used by NodeHealthMonitor)
# Nodes expose health endpoints on this port for cluster health tracking
HEALTH_CHECK_PORT = 8765

# Prometheus metrics port for observability
METRICS_PORT = 9090

# =============================================================================
# Data Transfer & Sync
# =============================================================================

# Data server port for aria2 transport and HTTP-based data transfers
# Used for serving game databases and model files to other nodes
DATA_SERVER_PORT = 8766

# Distributed data transfer port for dynamic data distribution
# Used by dynamic_data_distribution.py for NPZ/DB distribution from OWC
DISTRIBUTED_DATA_PORT = 8767

# Unified data sync HTTP API port
# Used by UnifiedDataSyncService for cluster-wide data synchronization
UNIFIED_SYNC_API_PORT = 8772

# =============================================================================
# AI Service API
# =============================================================================

# Default port for the FastAPI AI service (move selection, evaluation)
# Note: Often overridden by environment or command line
AI_SERVICE_DEFAULT_PORT = 8000

# Human evaluation web server port
HUMAN_EVAL_PORT = 8081

# Training dashboard server port
TRAINING_DASHBOARD_PORT = 8080

# Keepalive dashboard API port
KEEPALIVE_DASHBOARD_PORT = 8771  # Shares with gossip (different endpoints)

# =============================================================================
# Helper Functions
# =============================================================================


def get_p2p_status_url(host: str = "localhost", port: int | None = None) -> str:
    """Build P2P status URL for a given host.

    Args:
        host: Hostname or IP address
        port: Override port (defaults to P2P_DEFAULT_PORT)

    Returns:
        Full URL to P2P status endpoint
    """
    port = port or P2P_DEFAULT_PORT
    return f"http://{host}:{port}/status"


def get_p2p_base_url(host: str = "localhost", port: int | None = None) -> str:
    """Build P2P base URL for a given host.

    Args:
        host: Hostname or IP address
        port: Override port (defaults to P2P_DEFAULT_PORT)

    Returns:
        Base URL for P2P orchestrator
    """
    port = port or P2P_DEFAULT_PORT
    return f"http://{host}:{port}"


def get_local_p2p_url() -> str:
    """Get the local P2P orchestrator URL.

    This is the canonical way to get the P2P URL - checks environment
    variables first, then falls back to localhost default.

    Environment variables checked (in order):
        - RINGRIFT_P2P_URL
        - P2P_URL
        - P2P_ORCHESTRATOR_URL

    Returns:
        P2P orchestrator base URL (e.g., "http://localhost:8770")

    Example:
        from app.config.ports import get_local_p2p_url
        url = get_local_p2p_url()  # "http://localhost:8770" or from env
    """
    import os
    return (
        os.environ.get("RINGRIFT_P2P_URL")
        or os.environ.get("P2P_URL")
        or os.environ.get("P2P_ORCHESTRATOR_URL")
        or get_p2p_base_url()
    )


def get_data_server_url(host: str, port: int | None = None, path: str = "") -> str:
    """Build data server URL for file transfers.

    Args:
        host: Hostname or IP address
        port: Override port (defaults to DATA_SERVER_PORT)
        path: Optional path to append (e.g., "/db" or "/models")

    Returns:
        Full URL to data server
    """
    port = port or DATA_SERVER_PORT
    return f"http://{host}:{port}{path}"


def get_health_check_url(host: str, port: int | None = None) -> str:
    """Build health check URL for a node.

    Args:
        host: Hostname or IP address
        port: Override port (defaults to HEALTH_CHECK_PORT)

    Returns:
        Full URL to health endpoint
    """
    port = port or HEALTH_CHECK_PORT
    return f"http://{host}:{port}/health"
