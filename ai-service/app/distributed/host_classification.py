"""Host Classification for RingRift AI Service.

This module provides host classification based on storage characteristics,
enabling differentiated sync strategies for different host types.

Key features:
1. Storage type classification (persistent, ephemeral, SSD, network)
2. Sync profiles with optimized settings per storage type
3. Automatic detection of ephemeral hosts (RAM disk, Vast.ai)
4. Priority-based sync scheduling

Usage:
    from app.distributed.host_classification import (
        StorageType,
        HostSyncProfile,
        classify_host_storage,
        get_ephemeral_hosts,
    )

    # Classify a host
    storage_type = classify_host_storage(host_config)

    # Create sync profile
    if storage_type == StorageType.EPHEMERAL:
        profile = HostSyncProfile.for_ephemeral_host("vast-gpu-1")
    else:
        profile = HostSyncProfile.for_persistent_host("gh200-a")

    # Get all ephemeral hosts
    ephemeral = get_ephemeral_hosts(hosts_config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Storage Types
# =============================================================================


class StorageType(str, Enum):
    """Storage type classification for hosts.

    Different storage types require different sync strategies:
    - PERSISTENT: Standard disk storage, can sync at normal intervals
    - EPHEMERAL: RAM disk (lost on termination), needs aggressive sync
    - SSD: Fast SSD, can handle more parallel transfers
    - NETWORK: Network-attached, may need compression
    """
    PERSISTENT = "persistent"  # Normal disk storage
    EPHEMERAL = "ephemeral"    # RAM disk (/dev/shm) - data lost on termination
    SSD = "ssd"                # Fast SSD storage
    NETWORK = "network"        # Network-attached storage


# =============================================================================
# Host Tier Classification
# =============================================================================


class HostTier(str, Enum):
    """Hardware tier classification for hosts.

    Used for workload scheduling and resource allocation.
    """
    HIGH_END = "HIGH_END"      # High-performance GPUs (A100, H100, GH200)
    MID_TIER = "MID_TIER"      # Mid-range GPUs (RTX 4090, A10)
    LOW_TIER = "LOW_TIER"      # Entry-level GPUs
    CPU_ONLY = "CPU_ONLY"      # No GPU, CPU-only workloads


# =============================================================================
# Sync Profiles
# =============================================================================


@dataclass
class HostSyncProfile:
    """Sync profile for a host based on its characteristics.

    Provides optimized sync settings based on storage type and
    host capabilities.
    """
    host_name: str
    storage_type: StorageType = StorageType.PERSISTENT
    poll_interval_seconds: int = 60
    priority: int = 1  # Higher = sync first
    max_parallel_transfers: int = 2
    compress_in_transit: bool = False
    # Ephemeral-specific settings
    is_ephemeral: bool = False
    aggressive_sync: bool = False  # True for ephemeral hosts
    last_sync_time: float = 0.0
    games_at_risk: int = 0  # Estimated games that could be lost
    # Hardware tier
    tier: HostTier | None = None

    @classmethod
    def for_ephemeral_host(cls, host_name: str) -> HostSyncProfile:
        """Create profile for ephemeral (RAM disk) host.

        Ephemeral hosts use aggressive sync to prevent data loss:
        - 15s poll interval (vs 60s for persistent)
        - High priority (10 vs 1)
        - More parallel transfers
        - Compression enabled
        """
        return cls(
            host_name=host_name,
            storage_type=StorageType.EPHEMERAL,
            poll_interval_seconds=15,  # Aggressive: 15s instead of 60s
            priority=10,  # High priority
            max_parallel_transfers=4,  # More parallelism
            compress_in_transit=True,  # Compress to speed up
            is_ephemeral=True,
            aggressive_sync=True,
        )

    @classmethod
    def for_persistent_host(cls, host_name: str) -> HostSyncProfile:
        """Create profile for persistent storage host."""
        return cls(
            host_name=host_name,
            storage_type=StorageType.PERSISTENT,
            poll_interval_seconds=60,
            priority=1,
            max_parallel_transfers=2,
            compress_in_transit=False,
            is_ephemeral=False,
            aggressive_sync=False,
        )

    @classmethod
    def for_ssd_host(cls, host_name: str) -> HostSyncProfile:
        """Create profile for SSD storage host."""
        return cls(
            host_name=host_name,
            storage_type=StorageType.SSD,
            poll_interval_seconds=45,  # Slightly faster
            priority=3,
            max_parallel_transfers=4,  # Can handle more parallelism
            compress_in_transit=False,
            is_ephemeral=False,
            aggressive_sync=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host_name": self.host_name,
            "storage_type": self.storage_type.value,
            "poll_interval_seconds": self.poll_interval_seconds,
            "priority": self.priority,
            "max_parallel_transfers": self.max_parallel_transfers,
            "compress_in_transit": self.compress_in_transit,
            "is_ephemeral": self.is_ephemeral,
            "aggressive_sync": self.aggressive_sync,
            "last_sync_time": self.last_sync_time,
            "games_at_risk": self.games_at_risk,
            "tier": self.tier.value if self.tier else None,
        }


# =============================================================================
# Classification Functions
# =============================================================================


def classify_host_storage(host_config: dict[str, Any]) -> StorageType:
    """Classify host storage type from configuration.

    Detection priority:
    1. Explicit storage_type in config
    2. Infer from remote_path (e.g., /dev/shm indicates ephemeral)
    3. Default to PERSISTENT

    Args:
        host_config: Host configuration dictionary

    Returns:
        Detected StorageType
    """
    # Explicit storage_type in config
    storage_type = host_config.get("storage_type", "").lower()
    if storage_type in ("ram", "ephemeral"):
        return StorageType.EPHEMERAL
    if storage_type == "ssd":
        return StorageType.SSD
    if storage_type == "network":
        return StorageType.NETWORK
    if storage_type == "persistent":
        return StorageType.PERSISTENT

    # Infer from remote_path
    remote_path = host_config.get("remote_path", "")
    if "/dev/shm" in remote_path or "/run/shm" in remote_path:
        return StorageType.EPHEMERAL

    # Infer from remote_db_path
    remote_db_path = host_config.get("remote_db_path", "")
    if "/dev/shm" in remote_db_path or "/run/shm" in remote_db_path:
        return StorageType.EPHEMERAL

    return StorageType.PERSISTENT


def classify_host_tier(host_config: dict[str, Any]) -> HostTier:
    """Classify host hardware tier from configuration.

    Detection priority:
    1. Explicit tier in config
    2. Infer from GPU type
    3. Default to MID_TIER

    Args:
        host_config: Host configuration dictionary

    Returns:
        Detected HostTier
    """
    # Explicit tier in config
    tier = host_config.get("tier", "").upper()
    if tier in ("HIGH_END", "HIGH"):
        return HostTier.HIGH_END
    if tier in ("MID_TIER", "MID"):
        return HostTier.MID_TIER
    if tier in ("LOW_TIER", "LOW"):
        return HostTier.LOW_TIER
    if tier in ("CPU_ONLY", "CPU"):
        return HostTier.CPU_ONLY

    # Infer from GPU type
    gpu_type = host_config.get("gpu_type", "").lower()
    if any(x in gpu_type for x in ("h100", "a100", "gh200")):
        return HostTier.HIGH_END
    if any(x in gpu_type for x in ("4090", "a10", "3090")):
        return HostTier.MID_TIER
    if any(x in gpu_type for x in ("3080", "3070", "2080")):
        return HostTier.LOW_TIER
    if not gpu_type or gpu_type == "none":
        return HostTier.CPU_ONLY

    return HostTier.MID_TIER


def get_ephemeral_hosts(hosts_config: dict[str, Any]) -> list[str]:
    """Get list of ephemeral host names from config.

    Checks both vast_hosts (typically ephemeral) and standard_hosts.

    Args:
        hosts_config: Hosts configuration with vast_hosts and standard_hosts sections

    Returns:
        List of ephemeral host names
    """
    ephemeral = []

    # Check vast_hosts (typically ephemeral)
    for name, config in hosts_config.get("vast_hosts", {}).items():
        if classify_host_storage(config) == StorageType.EPHEMERAL:
            ephemeral.append(name)

    # Check standard_hosts for any with RAM storage
    for name, config in hosts_config.get("standard_hosts", {}).items():
        if classify_host_storage(config) == StorageType.EPHEMERAL:
            ephemeral.append(name)

    return ephemeral


def get_hosts_by_storage_type(
    hosts_config: dict[str, Any],
    storage_type: StorageType,
) -> list[str]:
    """Get hosts of a specific storage type.

    Args:
        hosts_config: Hosts configuration
        storage_type: Storage type to filter by

    Returns:
        List of matching host names
    """
    matching = []

    for section in ("standard_hosts", "vast_hosts"):
        for name, config in hosts_config.get(section, {}).items():
            if classify_host_storage(config) == storage_type:
                matching.append(name)

    return matching


def get_hosts_by_tier(
    hosts_config: dict[str, Any],
    tier: HostTier,
) -> list[str]:
    """Get hosts of a specific hardware tier.

    Args:
        hosts_config: Hosts configuration
        tier: Hardware tier to filter by

    Returns:
        List of matching host names
    """
    matching = []

    for section in ("standard_hosts", "vast_hosts"):
        for name, config in hosts_config.get(section, {}).items():
            if classify_host_tier(config) == tier:
                matching.append(name)

    return matching


def create_sync_profile(
    host_name: str,
    host_config: dict[str, Any],
) -> HostSyncProfile:
    """Create a sync profile for a host based on its configuration.

    Automatically detects storage type and creates appropriate profile.

    Args:
        host_name: Host name
        host_config: Host configuration

    Returns:
        Configured HostSyncProfile
    """
    storage_type = classify_host_storage(host_config)
    tier = classify_host_tier(host_config)

    if storage_type == StorageType.EPHEMERAL:
        profile = HostSyncProfile.for_ephemeral_host(host_name)
    elif storage_type == StorageType.SSD:
        profile = HostSyncProfile.for_ssd_host(host_name)
    else:
        profile = HostSyncProfile.for_persistent_host(host_name)

    profile.tier = tier
    return profile


def create_sync_profiles(
    hosts_config: dict[str, Any],
) -> dict[str, HostSyncProfile]:
    """Create sync profiles for all hosts in configuration.

    Args:
        hosts_config: Hosts configuration

    Returns:
        Dictionary mapping host name to HostSyncProfile
    """
    profiles = {}

    for section in ("standard_hosts", "vast_hosts"):
        for name, config in hosts_config.get(section, {}).items():
            profiles[name] = create_sync_profile(name, config)

    return profiles


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "HostSyncProfile",
    "HostTier",
    "StorageType",
    "classify_host_storage",
    "classify_host_tier",
    "create_sync_profile",
    "create_sync_profiles",
    "get_ephemeral_hosts",
    "get_hosts_by_storage_type",
    "get_hosts_by_tier",
]
