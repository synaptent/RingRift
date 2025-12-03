"""
Distributed training infrastructure for local Mac cluster and cloud deployment.

This module provides:
- Worker discovery via Bonjour/mDNS
- HTTP client for distributed task execution
- Coordinator utilities for CMA-ES population evaluation
"""

from .discovery import (
    WorkerDiscovery,
    WorkerInfo,
    discover_workers,
    wait_for_workers,
    parse_manual_workers,
    filter_healthy_workers,
)
from .client import (
    WorkerClient,
    DistributedEvaluator,
)

__all__ = [
    "WorkerDiscovery",
    "WorkerInfo",
    "discover_workers",
    "wait_for_workers",
    "parse_manual_workers",
    "filter_healthy_workers",
    "WorkerClient",
    "DistributedEvaluator",
]
