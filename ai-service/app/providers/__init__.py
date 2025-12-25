"""Cloud provider management for RingRift cluster.

This module provides unified interfaces for managing compute instances
across multiple cloud providers:
- Lambda Labs (GH200, H100, A10 GPUs) - Primary GPU training nodes
- Vast.ai (various consumer/datacenter GPUs) - Regular P2P nodes
- Hetzner Cloud (CPU instances) - CPU-only selfplay
- AWS EC2 (staging/proxy) - Keep updated, light utilization
- Tailscale (mesh networking) - Cross-provider connectivity

Usage:
    from app.providers import LambdaManager, VastManager, HetznerManager

    # List all Lambda instances
    lambda_mgr = LambdaManager()
    instances = await lambda_mgr.list_instances()

    # Check health
    for inst in instances:
        health = await lambda_mgr.check_health(inst)
        print(f"{inst.name}: {health.message}")
"""

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
from app.providers.lambda_manager import LambdaManager
from app.providers.tailscale_manager import TailscaleManager, TailscalePeer, TailscaleStatus
from app.providers.vast_manager import VastManager, VastOffer

__all__ = [
    # Base classes
    "Provider",
    "ProviderInstance",
    "ProviderManager",
    "InstanceState",
    "HealthCheckResult",
    "RecoveryResult",
    # Provider managers
    "LambdaManager",
    "VastManager",
    "HetznerManager",
    "AWSManager",
    "TailscaleManager",
    # Additional types
    "TailscalePeer",
    "TailscaleStatus",
    "VastOffer",
]
