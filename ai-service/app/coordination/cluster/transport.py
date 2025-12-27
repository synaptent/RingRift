"""Cluster transport layer (December 2025).

Re-exports from cluster_transport.py for unified access.

Usage:
    from app.coordination.cluster.transport import ClusterTransport, TransportConfig
    from app.coordination.cluster.transport import TransportError, RetryableTransportError
"""

from app.coordination.cluster_transport import (
    ClusterTransport,
    NodeConfig,
    PermanentTransportError,
    RetryableTransportError,
    TransportConfig,
    TransportError,
    TransportResult,
    get_cluster_transport,
)

__all__ = [
    "ClusterTransport",
    "NodeConfig",
    "PermanentTransportError",
    "RetryableTransportError",
    "TransportConfig",
    "TransportError",
    "TransportResult",
    "get_cluster_transport",
]
