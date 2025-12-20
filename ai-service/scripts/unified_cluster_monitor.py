"""Compatibility wrapper for unified_cluster_monitor.

This module preserves the legacy import path used by scripts/tests while
the implementation lives in app.monitoring.unified_cluster_monitor.
"""

# Re-export urllib.request for test patching compatibility
import urllib.request
from urllib.request import urlopen

from app.monitoring.unified_cluster_monitor import (
    CONFIG_PATH,
    ClusterConfig,
    ClusterHealth,
    LeaderHealth,
    NodeHealth,
    UnifiedClusterMonitor,
    print_cluster_status,
)

__all__ = [
    "CONFIG_PATH",
    "ClusterConfig",
    "ClusterHealth",
    "LeaderHealth",
    "NodeHealth",
    "UnifiedClusterMonitor",
    "print_cluster_status",
    "urllib",
    "urlopen",
]


if __name__ == "__main__":
    import runpy

    runpy.run_module("app.monitoring.unified_cluster_monitor", run_name="__main__")
