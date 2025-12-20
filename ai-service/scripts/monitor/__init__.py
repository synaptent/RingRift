"""Unified Cluster Monitoring Module.

This module consolidates the 24+ monitoring scripts into a single cohesive system.
It provides:
- CLI dashboard for quick cluster status
- Alert integration (Slack/Discord via cluster.yaml)
- Health checks for all nodes
- Metrics collection

Usage:
    # CLI dashboard
    python -m scripts.monitor status
    python -m scripts.monitor health
    python -m scripts.monitor alert "Test message"

    # Programmatic usage
    from scripts.monitor import get_cluster_status, check_cluster_health
"""

from .dashboard import (
    get_cluster_status,
    print_cluster_status,
    get_node_status,
)
from .health import (
    check_cluster_health,
    check_node_health,
    HealthStatus,
)
from .alerting import (
    send_alert,
    AlertLevel,
)

__all__ = [
    "AlertLevel",
    "HealthStatus",
    "check_cluster_health",
    "check_node_health",
    "get_cluster_status",
    "get_node_status",
    "print_cluster_status",
    "send_alert",
]
