"""Cloud provider management for RingRift cluster.

This module provides unified interfaces for managing compute instances
across multiple cloud providers:
- Vast.ai (various consumer/datacenter GPUs) - Primary GPU training nodes
- RunPod (H100, A100, L40S GPUs) - High-performance GPU nodes
- Nebius (H100, L40S GPUs) - Backbone infrastructure
- Hetzner Cloud (CPU instances) - CPU-only selfplay, P2P voters
- AWS EC2 (staging/proxy) - Keep updated, light utilization
- Tailscale (mesh networking) - Cross-provider connectivity

DEPRECATED (Dec 2025): Lambda Labs account terminated.
LambdaManager is kept for backward compatibility but will emit DeprecationWarning.

Usage:
    from app.providers import VastManager, HetznerManager

    # List all Vast instances
    vast_mgr = VastManager()
    instances = await vast_mgr.list_instances()

    # Check health
    for inst in instances:
        health = await vast_mgr.check_health(inst)
        print(f"{inst.name}: {health.message}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.providers.base import (
    HealthCheckResult,
    InstanceState,
    Provider,
    ProviderInstance,
    ProviderManager,
    RecoveryResult,
)
from app.providers.aws_manager import AWSManager
from app.providers.hetzner_manager import HetznerManager
from app.providers.tailscale_manager import TailscaleManager, TailscalePeer, TailscaleStatus
from app.providers.vast_manager import VastManager, VastOffer

# Lazy import for deprecated Lambda to avoid DeprecationWarning on module load
if TYPE_CHECKING:
    from app.providers.lambda_manager import LambdaManager as _LambdaManager


def __getattr__(name: str):
    """Lazy import for deprecated modules."""
    if name == "LambdaManager":
        from app.providers.lambda_manager import LambdaManager
        return LambdaManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "Provider",
    "ProviderInstance",
    "ProviderManager",
    "InstanceState",
    "HealthCheckResult",
    "RecoveryResult",
    # Provider managers
    "LambdaManager",  # DEPRECATED - will emit warning on import
    "VastManager",
    "HetznerManager",
    "AWSManager",
    "TailscaleManager",
    # Additional types
    "TailscalePeer",
    "TailscaleStatus",
    "VastOffer",
]
