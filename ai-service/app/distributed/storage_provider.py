"""Unified Storage Provider - Abstraction for distributed storage access.

This module provides a unified interface for accessing storage across different
providers (Lambda NFS, Vast.ai ephemeral, local development). It automatically
detects the current provider and optimizes sync operations accordingly.

Key features:
1. Auto-detection of storage provider based on hostname/paths
2. NFS optimization for Lambda Labs (skip rsync between NFS nodes)
3. Provider-specific path resolution
4. Integration with aria2, SSH, and P2P sync transports
5. Shared storage awareness for training data distribution

Usage:
    provider = get_storage_provider()

    # Get paths
    selfplay_dir = provider.selfplay_dir
    models_dir = provider.models_dir

    # Check capabilities
    if provider.has_shared_storage:
        # Skip data sync, files are already shared
        pass
    else:
        # Need to sync data to this node
        await provider.sync_training_data(sources)
"""

from __future__ import annotations

import logging
import os
import platform
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StorageProviderType(Enum):
    """Storage provider types."""
    LAMBDA_NFS = "lambda"      # Lambda Labs with 14PB shared NFS
    VAST_EPHEMERAL = "vast"    # Vast.ai with ephemeral local storage
    LOCAL = "local"            # Local development (Mac/Linux)
    AWS_EFS = "aws"            # AWS with EFS (future)
    UNKNOWN = "unknown"


@dataclass
class StorageCapabilities:
    """Capabilities of a storage provider."""
    has_shared_storage: bool = False       # True if storage is shared across nodes
    skip_rsync_for_shared: bool = False    # Skip rsync for nodes with shared access
    supports_direct_nfs: bool = False      # Can access NFS directly
    ephemeral: bool = False                # Storage is ephemeral (lost on shutdown)
    has_ram_disk: bool = False             # Has fast RAM-based scratch
    max_sync_interval_seconds: int = 60    # Recommended sync interval
    priority_in_fallback: int = 50         # Priority in transport fallback (higher = try first)


@dataclass
class StoragePaths:
    """Storage paths for a provider."""
    selfplay_games: Path
    model_checkpoints: Path
    training_data: Path
    elo_database: Path
    sync_staging: Path
    local_scratch: Path
    nfs_base: Path | None = None


class StorageProvider(ABC):
    """Abstract base class for storage providers."""

    @property
    @abstractmethod
    def provider_type(self) -> StorageProviderType:
        """Get the provider type."""

    @property
    @abstractmethod
    def capabilities(self) -> StorageCapabilities:
        """Get storage capabilities."""

    @property
    @abstractmethod
    def paths(self) -> StoragePaths:
        """Get storage paths."""

    @property
    def selfplay_dir(self) -> Path:
        """Convenience accessor for selfplay directory."""
        return self.paths.selfplay_games

    @property
    def models_dir(self) -> Path:
        """Convenience accessor for models directory."""
        return self.paths.model_checkpoints

    @property
    def training_dir(self) -> Path:
        """Convenience accessor for training data directory."""
        return self.paths.training_data

    @property
    def scratch_dir(self) -> Path:
        """Convenience accessor for scratch directory."""
        return self.paths.local_scratch

    @property
    def has_shared_storage(self) -> bool:
        """Check if this provider has shared storage (no sync needed)."""
        return self.capabilities.has_shared_storage

    @property
    def is_ephemeral(self) -> bool:
        """Check if storage is ephemeral."""
        return self.capabilities.ephemeral

    def should_skip_rsync_to(self, target_node: str) -> bool:
        """Check if rsync should be skipped to target node.

        Args:
            target_node: Target node identifier

        Returns:
            True if rsync should be skipped (both nodes have shared storage)
        """
        if not self.capabilities.skip_rsync_for_shared:
            return False
        # Skip rsync between Lambda nodes (both have NFS)
        return bool(target_node.startswith("lambda-") and self.provider_type == StorageProviderType.LAMBDA_NFS)

    def ensure_directories(self) -> None:
        """Ensure all storage directories exist."""
        paths = self.paths
        for path in [
            paths.selfplay_games,
            paths.model_checkpoints,
            paths.training_data,
            paths.sync_staging,
            paths.local_scratch,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        # Ensure parent of elo database exists
        paths.elo_database.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Ensured storage directories for {self.provider_type.value}")

    def get_sync_config(self) -> dict[str, Any]:
        """Get recommended sync configuration for this provider.

        Returns config suitable for UnifiedDataSyncConfig or aria2.
        """
        caps = self.capabilities
        return {
            "poll_interval_seconds": 15 if caps.ephemeral else 60,
            "skip_rsync_for_nfs": caps.skip_rsync_for_shared,
            "enable_aria2_transport": True,
            "enable_gossip_sync": not caps.has_shared_storage,  # Gossip useful for non-NFS
            "use_ram_scratch": caps.has_ram_disk,
            "max_sync_interval": caps.max_sync_interval_seconds,
        }


class LambdaNFSProvider(StorageProvider):
    """Storage provider for Lambda Labs with 14PB shared NFS."""

    NFS_BASE = Path("/lambda/nfs/RingRift")
    NFS_TEST_FILE = ".nfs_health_check"

    def __init__(self, nfs_base: Path | None = None):
        self._nfs_base = nfs_base or self.NFS_BASE
        self._paths = StoragePaths(
            selfplay_games=self._nfs_base / "selfplay",
            model_checkpoints=self._nfs_base / "models",
            training_data=self._nfs_base / "training_data",
            elo_database=self._nfs_base / "elo" / "unified_elo.db",
            sync_staging=self._nfs_base / "sync_staging",
            local_scratch=Path("/tmp/ringrift"),
            nfs_base=self._nfs_base,
        )
        self._capabilities = StorageCapabilities(
            has_shared_storage=True,
            skip_rsync_for_shared=True,
            supports_direct_nfs=True,
            ephemeral=False,
            has_ram_disk=False,
            max_sync_interval_seconds=300,  # Less urgent with shared storage
            priority_in_fallback=90,  # High priority - reliable
        )
        self._nfs_verified = False
        self._last_nfs_check = 0.0
        self._nfs_check_interval = 60.0  # Verify every 60 seconds

    @property
    def provider_type(self) -> StorageProviderType:
        return StorageProviderType.LAMBDA_NFS

    @property
    def capabilities(self) -> StorageCapabilities:
        return self._capabilities

    @property
    def paths(self) -> StoragePaths:
        return self._paths

    @classmethod
    def is_available(cls) -> bool:
        """Check if Lambda NFS is available."""
        return cls.NFS_BASE.exists() and cls.NFS_BASE.is_dir()

    def verify_nfs_mount(self, force: bool = False) -> bool:
        """Verify NFS mount is healthy by performing a write test.

        This detects stale NFS mounts or permissions issues that would
        cause silent data loss.

        Args:
            force: Force re-verification even if cache is valid

        Returns:
            True if NFS is healthy and writable
        """
        import time

        # Check cache
        if not force:
            if self._nfs_verified and (time.time() - self._last_nfs_check < self._nfs_check_interval):
                return True

        test_path = self._nfs_base / self.NFS_TEST_FILE
        node_id = socket.gethostname()
        test_content = f"{node_id}:{time.time()}"

        try:
            # Test 1: Write a file
            test_path.write_text(test_content)

            # Test 2: Read it back
            read_content = test_path.read_text()
            if read_content != test_content:
                logger.warning(f"NFS read mismatch: wrote '{test_content}', read '{read_content}'")
                self._nfs_verified = False
                return False

            # Test 3: Delete it
            test_path.unlink()

            self._nfs_verified = True
            self._last_nfs_check = time.time()
            logger.debug(f"NFS verification passed: {self._nfs_base}")
            return True

        except PermissionError as e:
            logger.error(f"NFS verification failed (permission): {e}")
            self._nfs_verified = False
            return False
        except OSError as e:
            # Catches stale NFS handle, connection issues, etc.
            logger.error(f"NFS verification failed (OS error): {e}")
            self._nfs_verified = False
            return False
        except Exception as e:
            logger.error(f"NFS verification failed (unexpected): {e}")
            self._nfs_verified = False
            return False

    def should_skip_rsync_to(self, target_node: str) -> bool:
        """Check if rsync should be skipped to target node.

        IMPORTANT: Only skips if NFS is verified healthy.

        Args:
            target_node: Target node identifier

        Returns:
            True if rsync should be skipped (both nodes have verified NFS)
        """
        if not self.capabilities.skip_rsync_for_shared:
            return False

        # Only skip if NFS is verified healthy
        if not self.verify_nfs_mount():
            logger.warning("NFS verification failed - will use rsync fallback")
            return False

        # Skip rsync between Lambda nodes (both have NFS)
        return bool(target_node.startswith("lambda-") and self.provider_type == StorageProviderType.LAMBDA_NFS)

    @property
    def is_nfs_healthy(self) -> bool:
        """Check if NFS is currently healthy."""
        return self.verify_nfs_mount()


class VastEphemeralProvider(StorageProvider):
    """Storage provider for Vast.ai with ephemeral local storage."""

    WORKSPACE_BASE = Path("/workspace")

    def __init__(self, workspace_base: Path | None = None):
        self._workspace_base = workspace_base or self.WORKSPACE_BASE
        self._paths = StoragePaths(
            selfplay_games=self._workspace_base / "data" / "selfplay",
            model_checkpoints=self._workspace_base / "models",
            training_data=self._workspace_base / "data" / "training",
            elo_database=self._workspace_base / "data" / "unified_elo.db",
            sync_staging=self._workspace_base / "data" / "sync_staging",
            local_scratch=Path("/dev/shm/ringrift"),  # RAM disk for speed
            nfs_base=None,
        )
        self._capabilities = StorageCapabilities(
            has_shared_storage=False,
            skip_rsync_for_shared=False,
            supports_direct_nfs=False,
            ephemeral=True,  # Data lost on shutdown
            has_ram_disk=True,  # /dev/shm available
            max_sync_interval_seconds=15,  # Aggressive sync for ephemeral
            priority_in_fallback=70,  # Medium priority
        )

    @property
    def provider_type(self) -> StorageProviderType:
        return StorageProviderType.VAST_EPHEMERAL

    @property
    def capabilities(self) -> StorageCapabilities:
        return self._capabilities

    @property
    def paths(self) -> StoragePaths:
        return self._paths

    @classmethod
    def is_available(cls) -> bool:
        """Check if running on Vast.ai."""
        # Check for /workspace directory (Vast.ai standard)
        if cls.WORKSPACE_BASE.exists():
            return True
        # Check environment variable
        return bool(os.environ.get("VAST_CONTAINERLABEL"))


class LocalStorageProvider(StorageProvider):
    """Storage provider for local development."""

    def __init__(self, base_dir: Path | None = None):
        if base_dir is None:
            # Default to ai-service directory
            base_dir = Path(__file__).parent.parent.parent
        self._base_dir = base_dir
        self._paths = StoragePaths(
            selfplay_games=base_dir / "data" / "selfplay",
            model_checkpoints=base_dir / "models",
            training_data=base_dir / "data" / "training",
            elo_database=base_dir / "data" / "unified_elo.db",
            sync_staging=base_dir / "data" / "sync_staging",
            local_scratch=Path("/tmp/ringrift"),
            nfs_base=None,
        )
        self._capabilities = StorageCapabilities(
            has_shared_storage=False,
            skip_rsync_for_shared=False,
            supports_direct_nfs=False,
            ephemeral=False,
            has_ram_disk=platform.system() == "Linux",  # /dev/shm on Linux
            max_sync_interval_seconds=60,
            priority_in_fallback=50,
        )

    @property
    def provider_type(self) -> StorageProviderType:
        return StorageProviderType.LOCAL

    @property
    def capabilities(self) -> StorageCapabilities:
        return self._capabilities

    @property
    def paths(self) -> StoragePaths:
        return self._paths


# =============================================================================
# Provider Detection and Factory
# =============================================================================

_cached_provider: StorageProvider | None = None


def detect_storage_provider() -> StorageProviderType:
    """Auto-detect the current storage provider.

    Detection order:
    1. RINGRIFT_STORAGE_PROVIDER environment variable
    2. Lambda NFS mount check
    3. Vast.ai workspace check
    4. Default to local
    """
    # Check environment override
    env_provider = os.environ.get("RINGRIFT_STORAGE_PROVIDER", "").lower()
    if env_provider == "lambda":
        return StorageProviderType.LAMBDA_NFS
    elif env_provider == "vast":
        return StorageProviderType.VAST_EPHEMERAL
    elif env_provider == "local":
        return StorageProviderType.LOCAL

    # Auto-detect based on filesystem
    if LambdaNFSProvider.is_available():
        return StorageProviderType.LAMBDA_NFS

    if VastEphemeralProvider.is_available():
        return StorageProviderType.VAST_EPHEMERAL

    # Check hostname patterns
    hostname = socket.gethostname().lower()
    if "lambda" in hostname or hostname.startswith("lambda-"):
        # Lambda node but NFS not mounted yet
        return StorageProviderType.LAMBDA_NFS

    return StorageProviderType.LOCAL


def get_storage_provider(
    provider_type: StorageProviderType | None = None,
    force_refresh: bool = False,
) -> StorageProvider:
    """Get the storage provider instance.

    Args:
        provider_type: Explicit provider type (auto-detects if None)
        force_refresh: Force re-detection and re-creation

    Returns:
        StorageProvider instance
    """
    global _cached_provider

    if _cached_provider is not None and not force_refresh and provider_type is None:
        return _cached_provider

    if provider_type is None:
        provider_type = detect_storage_provider()

    if provider_type == StorageProviderType.LAMBDA_NFS:
        provider = LambdaNFSProvider()
    elif provider_type == StorageProviderType.VAST_EPHEMERAL:
        provider = VastEphemeralProvider()
    else:
        provider = LocalStorageProvider()

    _cached_provider = provider
    logger.info(f"Storage provider initialized: {provider_type.value}")

    return provider


def clear_provider_cache() -> None:
    """Clear the cached provider (for testing)."""
    global _cached_provider
    _cached_provider = None


# =============================================================================
# Sync Transport Integration
# =============================================================================

@dataclass
class TransportConfig:
    """Configuration for data sync transports."""
    # aria2 settings
    enable_aria2: bool = True
    aria2_connections_per_server: int = 16
    aria2_split: int = 16
    aria2_data_server_port: int = 8766

    # SSH/rsync settings
    enable_ssh: bool = True
    ssh_timeout: int = 300

    # P2P HTTP settings
    enable_p2p: bool = True
    p2p_timeout: int = 300

    # Gossip settings
    # Phase 9 (Dec 2025): Enabled by default for better data resilience
    enable_gossip: bool = True
    gossip_port: int = 8771
    gossip_sync_interval: int = 60

    # General
    fallback_chain: list[str] = field(default_factory=lambda: ["aria2", "ssh", "p2p"])
    total_timeout_budget: int = 900  # 15 minutes max for all fallback attempts


def get_optimal_transport_config(provider: StorageProvider | None = None) -> TransportConfig:
    """Get optimal transport configuration for the current provider.

    Args:
        provider: Storage provider (auto-detects if None)

    Returns:
        TransportConfig optimized for the provider
    """
    if provider is None:
        provider = get_storage_provider()

    caps = provider.capabilities

    config = TransportConfig()

    if caps.has_shared_storage:
        # NFS nodes: prefer direct access, disable most sync
        config.enable_ssh = False  # No need to rsync between NFS nodes
        config.enable_gossip = False  # Data already shared
        config.fallback_chain = ["aria2"]  # Only use aria2 for external data
        logger.debug("Using minimal transport config for shared storage")

    elif caps.ephemeral:
        # Ephemeral nodes: aggressive sync, use all transports
        config.enable_gossip = True  # Help distribute data quickly
        config.fallback_chain = ["aria2", "ssh", "p2p"]  # aria2 first for speed
        logger.debug("Using aggressive transport config for ephemeral storage")

    else:
        # Standard nodes: balanced approach
        config.enable_gossip = True
        config.fallback_chain = ["aria2", "ssh", "p2p"]
        logger.debug("Using balanced transport config")

    return config


# =============================================================================
# Convenience Functions for Common Operations
# =============================================================================

def get_selfplay_dir() -> Path:
    """Get the selfplay games directory."""
    return get_storage_provider().selfplay_dir


def get_models_dir() -> Path:
    """Get the models directory."""
    return get_storage_provider().models_dir


def get_training_dir() -> Path:
    """Get the training data directory."""
    return get_storage_provider().training_dir


def get_scratch_dir() -> Path:
    """Get the local scratch directory."""
    return get_storage_provider().scratch_dir


def should_sync_to_node(target_node: str) -> bool:
    """Check if data sync is needed to target node.

    Args:
        target_node: Target node identifier

    Returns:
        True if sync is needed, False if both nodes have shared storage
    """
    provider = get_storage_provider()
    return not provider.should_skip_rsync_to(target_node)


def is_nfs_available() -> bool:
    """Check if NFS storage is available."""
    provider = get_storage_provider()
    return provider.capabilities.supports_direct_nfs


def verify_nfs_health() -> bool:
    """Verify NFS storage is healthy and writable.

    This performs a write/read/delete test to ensure NFS isn't stale.
    Should be called before relying on NFS for data sync.

    Returns:
        True if NFS is healthy, False if verification failed or NFS unavailable
    """
    provider = get_storage_provider()
    if provider.provider_type != StorageProviderType.LAMBDA_NFS:
        return False

    if isinstance(provider, LambdaNFSProvider):
        return provider.verify_nfs_mount(force=True)

    return False


def get_aria2_sources(exclude_self: bool = True) -> list[str]:
    """Get list of aria2 data server URLs from cluster.

    Args:
        exclude_self: Exclude current node from sources

    Returns:
        List of URLs like ["http://host1:8766", "http://host2:8766"]
    """
    # Import here to avoid circular dependency
    try:
        from app.sync.cluster_hosts import get_data_sync_urls
        return get_data_sync_urls(exclude_self=exclude_self, reachable_only=False)
    except Exception:
        pass

    try:
        from app.config.unified_config import get_config
        from app.distributed.hosts import load_remote_hosts
        port = get_config().distributed.data_server_port
    except Exception:
        port = 8766

    sources = []
    hostname = socket.gethostname().lower()

    try:
        hosts = load_remote_hosts()
    except Exception:
        hosts = {}

    for name, host in hosts.items():
        if exclude_self and name.lower() == hostname:
            continue
        if host.worker_url:
            sources.append(host.worker_url)
            continue
        for candidate in (host.tailscale_ip, host.ssh_host):
            if not candidate:
                continue
            host_ip = candidate.split("@", 1)[1] if "@" in candidate else candidate
            sources.append(f"http://{host_ip}:{port}")
            break

    return sources
