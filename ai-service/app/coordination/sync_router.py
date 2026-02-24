"""Sync Router - Intelligent Data Routing Based on Node Capabilities.

Routes data (games, models, NPZ files) to appropriate nodes based on:
- Node role (training, selfplay, coordinator)
- Disk capacity and usage limits
- Storage type (persistent, ephemeral, NFS)
- Network topology (same provider, NFS sharing)
- Exclusion rules

Usage:
    from app.coordination.sync_router import SyncRouter, get_sync_router

    router = get_sync_router()

    # Get sync targets for game data
    targets = router.get_sync_targets(data_type="game", board_type="hex8")

    # Check if a node should receive specific data
    if router.should_sync_to_node("gpu-node-1", data_type="game"):
        sync_to_node(...)
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.config.cluster_config import load_cluster_config, get_host_provider, get_cluster_nodes
from app.config.thresholds import DISK_SYNC_TARGET_PERCENT
from app.coordination.contracts import CoordinatorStatus, HealthCheckResult
from app.distributed.cluster_manifest import (
    ClusterManifest,
    DataType,
    SyncTarget,
    get_cluster_manifest,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Data classes
    "SyncRoute",
    "NodeSyncCapability",
    "S3SyncConfig",  # Phase 2: S3 as first-class sync target (Jan 2026)
    # Main class
    "SyncRouter",
    # Singleton accessors
    "get_sync_router",
    "reset_sync_router",
]


@dataclass
class S3SyncConfig:
    """Configuration for S3 as a sync target (Phase 2 - Jan 2026).

    Enables S3 to be a first-class storage tier alongside OWC drives.
    """
    enabled: bool = True
    bucket: str = "ringrift-models-20251214"
    region: str = "us-east-1"
    prefix: str = "consolidated"
    pull_games: bool = True
    pull_training_npz: bool = True
    pull_models: bool = True
    push_enabled_hosts: list[str] = field(default_factory=list)
    sync_interval: int = 300
    # Phase 2: S3 as primary target (Jan 2026)
    primary_for_games: bool = True  # Prefer S3 over OWC for game storage
    primary_for_models: bool = True
    primary_for_npz: bool = True


@dataclass
class SyncRoute:
    """A sync route with source and target information."""
    source_node: str
    target_node: str
    data_type: DataType
    priority: int = 0
    reason: str = ""
    estimated_size_bytes: int = 0
    bandwidth_limit_mbps: int | None = None
    quality_score: float = 0.0  # Dec 2025: Quality-based priority boost


@dataclass
class NodeSyncCapability:
    """Sync capabilities for a node."""
    node_id: str
    can_receive_games: bool = True
    can_receive_models: bool = True
    can_receive_npz: bool = True
    is_training_node: bool = False
    is_priority_node: bool = False
    is_ephemeral: bool = False
    shares_nfs: bool = False
    provider: str = "unknown"
    disk_usage_percent: float = 0.0
    available_gb: float = 0.0
    last_sync_time: float = 0.0  # Dec 2025: Timestamp of last successful sync
    # Dec 2025: Added missing properties to fix AttributeError crashes
    selfplay_enabled: bool = False  # Whether node runs selfplay jobs
    has_gpu: bool = False  # Whether node has GPU (inferred from provider/role)

    @property
    def training_enabled(self) -> bool:
        """Alias for is_training_node for consistency."""
        return self.is_training_node

    @property
    def disk_percent(self) -> float:
        """Alias for disk_usage_percent for consistency."""
        return self.disk_usage_percent


class SyncRouter:
    """Routes data to nodes based on capabilities and policies.

    Integrates with ClusterManifest for:
    - Exclusion rules and sync policies
    - Disk capacity checks
    - Replication tracking
    """

    # P0.6 Dec 2025: Capacity refresh interval (30 seconds)
    # Previous 5-minute interval allowed disk to fill during active training
    CAPACITY_REFRESH_INTERVAL = 30.0

    # Dec 2025: Sync timestamp state file
    _SYNC_STATE_FILE = Path("data/sync/.node_sync_timestamps.json")

    def __init__(
        self,
        config_path: Path | None = None,
        manifest: ClusterManifest | None = None,
    ):
        """Initialize the sync router.

        Args:
            config_path: Path to distributed_hosts.yaml
            manifest: ClusterManifest instance (uses singleton if None)
        """
        self.node_id = socket.gethostname()
        self._manifest = manifest or get_cluster_manifest()

        # Load host configuration
        self._hosts_config: dict[str, Any] = {}
        self._sync_routing: dict[str, Any] = {}
        self._node_capabilities: dict[str, NodeSyncCapability] = {}
        self._load_config(config_path)

        # Phase 2 (Jan 2026): S3 as first-class sync target
        self._s3_config: S3SyncConfig = self._load_s3_config()
        self._s3_healthy: bool = True
        self._s3_last_health_check: float = 0.0
        self._s3_health_check_interval: float = 60.0  # Check S3 health every 60s

        # P2.3 Dec 2025: Capacity refresh tracking
        self._last_capacity_refresh = 0.0

        # Dec 2025: Load persisted sync timestamps
        self._load_sync_timestamps()

        logger.info(
            f"SyncRouter initialized: {len(self._node_capabilities)} nodes, "
            f"S3 enabled={self._s3_config.enabled}"
        )

    def _load_config(self, config_path: Path | None = None) -> None:
        """Load configuration from distributed_hosts.yaml using cluster_config.

        Uses the consolidated cluster_config module instead of inline yaml loading.
        """
        try:
            cluster_config = load_cluster_config(config_path)

            self._hosts_config = cluster_config.hosts_raw
            self._sync_routing = cluster_config.get_raw_section("sync_routing")

            # Dec 2025: Load allowed_external_storage for coordinator backup
            self._external_storage: list[dict[str, Any]] = [
                {
                    "host": storage.host,
                    "path": storage.path,
                    "receive_games": storage.receive_games,
                    "receive_npz": storage.receive_npz,
                    "receive_models": storage.receive_models,
                    "subdirs": storage.subdirs,
                }
                for storage in cluster_config.sync_routing.allowed_external_storage
            ]

            # Build node capabilities from hosts config
            self._build_node_capabilities()

            # Dec 2025: Log external storage config
            if self._external_storage:
                for storage in self._external_storage:
                    logger.info(
                        f"[SyncRouter] External storage configured: "
                        f"{storage.get('host')} -> {storage.get('path')}"
                    )

        except (yaml.YAMLError, OSError) as e:
            # Config file errors (malformed YAML, file not found, permission denied)
            logger.error(f"Failed to load config file: {e}")
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            # Config structure errors (invalid values, missing keys, type mismatches)
            logger.error(f"Failed to parse config structure: {e}")

    def _build_node_capabilities(self) -> None:
        """Build node capability information from config."""
        priority_hosts = set(self._sync_routing.get("priority_hosts", []))

        # Dec 2025: Build external storage lookup for coordinator backup
        external_storage_hosts = {}
        for storage in getattr(self, "_external_storage", []):
            host = storage.get("host", "")
            if host:
                external_storage_hosts[host] = storage

        for host_name, host_config in self._hosts_config.items():
            role = host_config.get("role", "selfplay")
            gpu = host_config.get("gpu", "")

            # Use consolidated provider detection from cluster_config
            provider = get_host_provider(host_name)

            # Check if shares NFS (Lambda nodes with same provider)
            shares_nfs = provider == "lambda"

            # Check if ephemeral (Vast.ai)
            is_ephemeral = provider == "vast"

            # Check if training node
            is_training = "training" in role or host_name in priority_hosts

            # Get sync policy from manifest
            policy = self._manifest.get_sync_policy(host_name)

            # Dec 2025: Override with external storage config if present
            # This enables coordinator backup via external drives
            if host_name in external_storage_hosts:
                ext_config = external_storage_hosts[host_name]
                can_receive_games = ext_config.get("receive_games", False)
                can_receive_models = ext_config.get("receive_models", False)
                can_receive_npz = ext_config.get("receive_npz", False)
                is_priority = True  # External storage is priority for backup
            else:
                can_receive_games = policy.receive_games
                can_receive_models = policy.receive_models
                can_receive_npz = policy.receive_npz
                is_priority = host_name in priority_hosts

            # Dec 2025: Determine selfplay and GPU capabilities from role/gpu config
            is_selfplay = "selfplay" in role or role == "selfplay"
            has_gpu = bool(gpu)  # Non-empty gpu string means GPU available

            cap = NodeSyncCapability(
                node_id=host_name,
                can_receive_games=can_receive_games,
                can_receive_models=can_receive_models,
                can_receive_npz=can_receive_npz,
                is_training_node=is_training,
                is_priority_node=is_priority,
                is_ephemeral=is_ephemeral,
                shares_nfs=shares_nfs,
                provider=provider,
                selfplay_enabled=is_selfplay,
                has_gpu=has_gpu,
            )

            self._node_capabilities[host_name] = cap

    def _load_s3_config(self) -> S3SyncConfig:
        """Load S3 configuration from sync_routing section.

        Phase 2 (Jan 2026): S3 as first-class sync target.
        """
        s3_cfg = self._sync_routing.get("s3", {})
        if not s3_cfg:
            return S3SyncConfig(enabled=False)

        return S3SyncConfig(
            enabled=s3_cfg.get("enabled", True),
            bucket=s3_cfg.get("bucket", "ringrift-models-20251214"),
            region=s3_cfg.get("region", "us-east-1"),
            prefix=s3_cfg.get("prefix", "consolidated"),
            pull_games=s3_cfg.get("pull_games", True),
            pull_training_npz=s3_cfg.get("pull_training_npz", True),
            pull_models=s3_cfg.get("pull_models", True),
            push_enabled_hosts=s3_cfg.get("push_enabled_hosts", []),
            sync_interval=s3_cfg.get("sync_interval", 300),
            # Phase 2: S3 as primary (Jan 2026)
            primary_for_games=s3_cfg.get("primary_for_games", True),
            primary_for_models=s3_cfg.get("primary_for_models", True),
            primary_for_npz=s3_cfg.get("primary_for_npz", True),
        )

    # =========================================================================
    # Phase 2 (Jan 2026): S3 as First-Class Sync Target
    # =========================================================================

    def is_s3_primary_for(self, data_type: str | DataType) -> bool:
        """Check if S3 should be the primary sync target for a data type.

        Phase 2: S3 as first-class storage tier.

        Args:
            data_type: Type of data ("game", "model", "npz")

        Returns:
            True if S3 should be preferred over local storage
        """
        if not self._s3_config.enabled:
            return False

        if not self._check_s3_health():
            return False

        if isinstance(data_type, str):
            data_type = DataType(data_type)

        if data_type == DataType.GAME:
            return self._s3_config.primary_for_games
        elif data_type == DataType.MODEL:
            return self._s3_config.primary_for_models
        elif data_type == DataType.NPZ:
            return self._s3_config.primary_for_npz
        return False

    def get_s3_uri(self, data_type: str | DataType, filename: str) -> str:
        """Get S3 URI for a file.

        Args:
            data_type: Type of data ("game", "model", "npz")
            filename: Filename

        Returns:
            Full S3 URI (s3://bucket/prefix/type/filename)
        """
        if isinstance(data_type, DataType):
            type_str = data_type.value
        else:
            type_str = data_type

        # Map data types to S3 subdirectories
        type_to_subdir = {
            "game": "games",
            "games": "games",
            "model": "models",
            "models": "models",
            "npz": "training",
            "training": "training",
        }
        subdir = type_to_subdir.get(type_str, type_str)

        return f"s3://{self._s3_config.bucket}/{self._s3_config.prefix}/{subdir}/{filename}"

    def _check_s3_health(self) -> bool:
        """Check S3 health (cached for performance).

        Returns:
            True if S3 is healthy
        """
        now = time.time()
        if now - self._s3_last_health_check < self._s3_health_check_interval:
            return self._s3_healthy

        # Check S3 health via aws s3 ls (fast operation)
        try:
            result = subprocess.run(
                ["aws", "s3", "ls", f"s3://{self._s3_config.bucket}/", "--max-items", "1"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._s3_healthy = result.returncode == 0
            self._s3_last_health_check = now
            if not self._s3_healthy:
                logger.warning(f"[SyncRouter] S3 health check failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            self._s3_healthy = False
            self._s3_last_health_check = now
            logger.warning(f"[SyncRouter] S3 health check error: {e}")

        return self._s3_healthy

    def should_push_to_s3(self, data_type: str | DataType) -> bool:
        """Check if this node should push data to S3.

        Args:
            data_type: Type of data

        Returns:
            True if this node should push to S3
        """
        if not self._s3_config.enabled:
            return False

        # Check if this node is in the push_enabled_hosts list
        if self._s3_config.push_enabled_hosts:
            if self.node_id not in self._s3_config.push_enabled_hosts:
                return False

        return self.is_s3_primary_for(data_type)

    def get_s3_config(self) -> S3SyncConfig:
        """Get the S3 sync configuration."""
        return self._s3_config

    def get_external_storage_path(self, host: str, data_type: str) -> str | None:
        """Get the external storage path for a host and data type.

        December 2025: Supports coordinator backup via external drives.

        Args:
            host: Hostname to check
            data_type: Type of data ("games", "models", "npz")

        Returns:
            Storage path if configured, None otherwise
        """
        for storage in getattr(self, "_external_storage", []):
            if storage.get("host") == host:
                base_path = storage.get("path", "")
                subdirs = storage.get("subdirs", {})
                subdir = subdirs.get(data_type, data_type)
                return f"{base_path}/{subdir}" if base_path else None
        return None

    def get_sync_targets(
        self,
        data_type: str | DataType,
        board_type: str | None = None,
        num_players: int | None = None,
        exclude_nodes: list[str] | None = None,
        max_targets: int = 10,
        size_bytes: int = 0,
    ) -> list[SyncTarget]:
        """Get candidate nodes for syncing data.

        Args:
            data_type: Type of data ("game", "model", "npz")
            board_type: Optional board type filter
            num_players: Optional num_players filter
            exclude_nodes: Nodes to exclude
            max_targets: Maximum number of targets to return
            size_bytes: Estimated size of data to sync (for pool capacity check)

        Returns:
            List of SyncTarget sorted by priority
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)

        exclude = set(exclude_nodes or [])
        exclude.add(self.node_id)  # Don't sync to self

        targets: list[SyncTarget] = []

        # Dec 2025: Load cluster nodes for storage routing checks
        cluster_nodes = get_cluster_nodes()

        for node_id, cap in self._node_capabilities.items():
            if node_id in exclude:
                continue

            # Dec 2025: Check if node should receive sync data
            node_config = cluster_nodes.get(node_id)
            if node_config and not node_config.should_receive_sync():
                continue

            # Check if node can receive this data type
            if not self._can_receive_data_type(cap, data_type):
                continue

            # Check disk capacity
            if not self._check_node_capacity(node_id):
                continue

            # Dec 30, 2025 (Phase 2.3): Check rotating pool capacity for coordinators
            if size_bytes > 0 and not self._check_pool_capacity(node_id, size_bytes, node_config):
                continue

            # Skip if NFS sharing applies (data already visible)
            if self._shares_storage_with(node_id):
                continue

            # Compute priority
            priority = self._compute_target_priority(cap, data_type)
            reason = self._get_target_reason(cap)

            targets.append(SyncTarget(
                node_id=node_id,
                priority=priority,
                reason=reason,
            ))

        # Sort by priority (highest first)
        targets.sort(key=lambda t: t.priority, reverse=True)

        return targets[:max_targets]

    def _check_pool_capacity(
        self,
        node_id: str,
        size_bytes: int,
        node_config: object | None = None,
    ) -> bool:
        """Check if a node has rotating pool capacity for sync data.

        Dec 30, 2025: Phase 2.3 of distributed data pipeline architecture.

        For coordinator nodes that use rotating pools, this checks if the pool
        has sufficient capacity for the incoming data. Non-coordinator nodes
        and nodes without pools always return True.

        Args:
            node_id: Node identifier
            size_bytes: Estimated size of data to sync
            node_config: Optional ClusterNode config

        Returns:
            True if node can accept the data, False otherwise
        """
        try:
            # Only check pool for coordinator nodes or nodes with skip_sync_receive
            if node_config is not None:
                role = getattr(node_config, "role", None)
                if role != "coordinator":
                    return True  # Non-coordinators don't use pools

            # Check if this is the local node
            from app.config.env import env

            if node_id != env.node_id:
                return True  # Can't check remote pool capacity here

            # Check local rotating pool
            from app.coordination.rotating_disk_pool import get_rotating_pool_manager

            pool = get_rotating_pool_manager()
            return pool.can_accept_data(size_bytes)

        except ImportError:
            # Rotating pool not available
            return True
        except Exception as e:
            logger.debug(f"[SyncRouter] Error checking pool capacity for {node_id}: {e}")
            return True  # Allow sync on error

    def get_sync_sources(
        self,
        data_type: str | DataType,
        target_node: str | None = None,
        exclude_nodes: list[str] | None = None,
        max_sources: int = 5,
    ) -> list[SyncTarget]:
        """Get nodes that can provide data (for reverse sync / pull operations).

        December 2025: Added to support coordinator pulling data from cluster nodes.
        This is the inverse of get_sync_targets() - finds nodes that generate data
        which should be synced TO the target_node.

        Args:
            data_type: Type of data ("game", "model", "npz")
            target_node: Node that needs the data (defaults to self.node_id)
            exclude_nodes: Nodes to exclude as sources
            max_sources: Maximum number of sources to return

        Returns:
            List of SyncTarget sorted by priority (nodes with most data first)
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)

        target_node = target_node or self.node_id
        exclude = set(exclude_nodes or [])
        exclude.add(target_node)  # Don't pull from self

        sources: list[SyncTarget] = []

        # Dec 2025: Load cluster nodes for storage routing checks
        cluster_nodes = get_cluster_nodes()

        for node_id, cap in self._node_capabilities.items():
            if node_id in exclude:
                continue

            # Source nodes must generate/have the data type
            if not self._can_provide_data_type(cap, data_type):
                continue

            # Check if node has data to share
            node_config = cluster_nodes.get(node_id)
            if node_config and node_config.role == "coordinator":
                # Coordinators don't generate data
                continue

            # Compute priority based on data generation rate
            priority = self._compute_source_priority(cap, data_type)
            reason = f"Has {data_type.value} data, generates selfplay"

            sources.append(SyncTarget(
                node_id=node_id,
                priority=priority,
                reason=reason,
            ))

        # Dec 30, 2025: Include external storage as high-priority sources
        # External storage (OWC drives) often contains archival data
        for storage in getattr(self, "_external_storage", []):
            host = storage.get("host", "")
            if host in exclude:
                continue

            # Check if storage can provide this data type
            if self._can_external_storage_provide(storage, data_type):
                path = storage.get("path", "")
                sources.append(SyncTarget(
                    node_id=host,
                    priority=150.0,  # Higher than cluster nodes (archival data)
                    reason=f"external_storage:{path}",
                ))

        # Sort by priority (highest first - nodes with most data)
        sources.sort(key=lambda s: s.priority, reverse=True)

        return sources[:max_sources]

    def _can_provide_data_type(
        self,
        cap: NodeSyncCapability,
        data_type: DataType,
    ) -> bool:
        """Check if a node can provide (has) a specific data type."""
        if data_type == DataType.GAME:
            # Nodes that generate selfplay have game data
            return cap.selfplay_enabled
        elif data_type == DataType.MODEL:
            # Any node can have models
            return True
        elif data_type == DataType.NPZ:
            # NPZ data comes from export, usually on training nodes
            return cap.training_enabled or cap.selfplay_enabled
        return False

    def _can_external_storage_provide(
        self,
        storage: dict[str, Any],
        data_type: DataType,
    ) -> bool:
        """Check if external storage can provide a specific data type.

        Dec 30, 2025: Added to support external storage as sync sources.
        External storage config uses receive_* flags to indicate what data is stored.
        """
        if data_type == DataType.GAME:
            return storage.get("receive_games", False)
        elif data_type == DataType.MODEL:
            return storage.get("receive_models", False)
        elif data_type == DataType.NPZ:
            return storage.get("receive_npz", False)
        return False

    def _compute_source_priority(
        self,
        cap: NodeSyncCapability,
        data_type: DataType,
    ) -> float:
        """Compute priority for a source node (higher = more data).

        December 2025: Priority based on selfplay/training activity.
        """
        priority = 0.0

        if data_type == DataType.GAME:
            if cap.selfplay_enabled:
                priority += 100.0  # Active selfplay = high priority
            if cap.training_enabled:
                priority += 50.0   # Training nodes may have consolidated data
        elif data_type == DataType.MODEL:
            if cap.training_enabled:
                priority += 100.0  # Training nodes have latest models
        elif data_type == DataType.NPZ:
            if cap.training_enabled:
                priority += 100.0  # Training nodes export NPZ

        # Boost for GPU nodes (generate more data)
        if cap.has_gpu:
            priority += 20.0

        # Boost for high disk usage (more data available)
        if cap.disk_percent > 50:
            priority += 10.0

        return priority

    def _can_receive_data_type(
        self,
        cap: NodeSyncCapability,
        data_type: DataType,
    ) -> bool:
        """Check if a node can receive a specific data type."""
        if data_type == DataType.GAME:
            return cap.can_receive_games
        elif data_type == DataType.MODEL:
            return cap.can_receive_models
        elif data_type == DataType.NPZ:
            return cap.can_receive_npz
        return False

    def _check_node_capacity(self, node_id: str) -> bool:
        """Check if a node has capacity for more data.

        P2.3 Dec 2025: Added capacity refresh check to avoid stale data.
        """
        # Refresh capacity if stale
        self._maybe_refresh_capacity()
        return self._manifest.can_receive_data(node_id, DataType.GAME)

    def _maybe_refresh_capacity(self) -> None:
        """Refresh capacity data if stale.

        P2.3 Dec 2025: Prevents routing decisions based on hours-old capacity data.
        """
        import time

        now = time.time()
        if now - self._last_capacity_refresh < self.CAPACITY_REFRESH_INTERVAL:
            return

        self._last_capacity_refresh = now

        # Refresh local node capacity
        try:
            self._manifest.update_local_capacity()
            logger.debug("[SyncRouter] Refreshed local capacity")
        except (AttributeError, RuntimeError, OSError) as e:
            # AttributeError: manifest not initialized
            # RuntimeError: capacity update failed internally
            # OSError: disk/network probe failed
            logger.debug(f"[SyncRouter] Failed to refresh local capacity: {e}")

        # Emit capacity refresh event for cluster-wide updates
        # Sprint 10 (Jan 3, 2026): Use unified emitter for consistent payloads
        try:
            from app.distributed.data_events import emit_node_capacity_updated_sync

            emit_node_capacity_updated_sync(
                node_id=self.node_id,
                reason="capacity_refresh",
                source="sync_router",
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            # ImportError: unified emitter not available
            # AttributeError: emitter misconfigured
            # RuntimeError: event emission failed
            logger.debug(f"[SyncRouter] Failed to emit capacity event: {e}")

    def refresh_all_capacity(self) -> None:
        """Force refresh of all capacity data.

        P2.3 Dec 2025: Call this when capacity data is suspected to be stale.
        """
        import time

        self._last_capacity_refresh = time.time()

        # Update local capacity
        try:
            self._manifest.update_local_capacity()
            logger.info("[SyncRouter] Force refreshed local capacity")
        except (AttributeError, RuntimeError, OSError) as e:
            # AttributeError: manifest not initialized
            # RuntimeError: capacity update failed internally
            # OSError: disk/network probe failed
            logger.warning(f"[SyncRouter] Failed to refresh capacity: {e}")

    def _shares_storage_with(self, target_node: str) -> bool:
        """Check if current node shares storage with target node.

        If both nodes are Lambda nodes with NFS, they share storage
        and don't need to sync between each other.
        """
        my_cap = self._node_capabilities.get(self.node_id)
        target_cap = self._node_capabilities.get(target_node)

        if not my_cap or not target_cap:
            return False

        # Both Lambda nodes share NFS
        if my_cap.shares_nfs and target_cap.shares_nfs:
            return True

        return False

    def _compute_target_priority(
        self,
        cap: NodeSyncCapability,
        data_type: DataType,
    ) -> int:
        """Compute sync priority for a target node.

        Higher priority = sync first.

        December 29, 2025: Added training-active priority boost (+50)
        to reduce data staleness for nodes actively running training jobs.
        """
        priority = 50  # Base priority

        # December 29, 2025: HIGHEST priority for nodes actively running training
        # This reduces data staleness from 5-15min to <1min for training nodes
        if self._is_node_training_active(cap.node_id):
            priority += 50  # Training-active gets highest boost

        # Training nodes get high priority for game/NPZ data
        if cap.is_training_node and data_type in (DataType.GAME, DataType.NPZ):
            priority += 30

        # Priority nodes get bonus
        if cap.is_priority_node:
            priority += 20

        # Ephemeral nodes get lower priority for receiving data
        # (we should sync FROM them, not TO them)
        if cap.is_ephemeral:
            priority -= 15

        # Prefer nodes with more available space
        if cap.disk_usage_percent < 50:
            priority += 10
        elif cap.disk_usage_percent > DISK_SYNC_TARGET_PERCENT:
            priority -= 20

        # December 2025: Time-since-sync priority weighting
        # Nodes not synced recently get priority boost (up to 30 points for 1 hour)
        if cap.last_sync_time > 0:
            seconds_since_sync = time.time() - cap.last_sync_time
            time_weight = min((seconds_since_sync / 3600.0) * 30, 30)
            priority += int(time_weight)
        else:
            # Never synced = maximum time weight
            priority += 30

        return priority

    def _is_node_training_active(self, node_id: str) -> bool:
        """Check if a node is actively running a training job.

        December 29, 2025: Added to support training-active priority sync.
        Reduces data staleness by prioritizing nodes with in-flight training.

        Args:
            node_id: Node to check

        Returns:
            True if node has active training job
        """
        # Check cached training-active nodes (updated by P2P status)
        if hasattr(self, "_training_active_nodes"):
            return node_id in self._training_active_nodes
        return False

    def update_training_active_nodes(self, active_nodes: set[str]) -> None:
        """Update the set of nodes actively running training.

        December 29, 2025: Called by P2P orchestrator or training activity daemon
        to update which nodes have in-flight training jobs.

        Args:
            active_nodes: Set of node IDs with active training
        """
        self._training_active_nodes = active_nodes
        logger.debug(f"[SyncRouter] Updated training-active nodes: {len(active_nodes)}")

    def _get_target_reason(self, cap: NodeSyncCapability) -> str:
        """Get human-readable reason for target selection."""
        reasons = []

        if cap.is_training_node:
            reasons.append("training")
        if cap.is_priority_node:
            reasons.append("priority")
        if cap.is_ephemeral:
            reasons.append("ephemeral")

        return ", ".join(reasons) if reasons else "available"

    # =========================================================================
    # December 2025: Sync Timestamp Persistence
    # =========================================================================

    def _load_sync_timestamps(self) -> None:
        """Load persisted sync timestamps from JSON file."""
        if not self._SYNC_STATE_FILE.exists():
            return

        try:
            with open(self._SYNC_STATE_FILE) as f:
                data = json.load(f)
            loaded_count = 0
            for node_id, timestamp in data.items():
                if node_id in self._node_capabilities:
                    self._node_capabilities[node_id].last_sync_time = timestamp
                    loaded_count += 1
            if loaded_count:
                logger.debug(f"[SyncRouter] Loaded sync timestamps for {loaded_count} nodes")
        except (
            json.JSONDecodeError,  # Corrupted JSON
            FileNotFoundError,     # File removed between exists() and open()
            PermissionError,       # Access denied
            OSError,               # General I/O error
            KeyError,              # Missing expected key
            TypeError,             # Invalid data type in JSON
            ValueError,            # Invalid timestamp value
        ) as e:
            logger.warning(f"[SyncRouter] Failed to load sync timestamps: {e}")

    def _save_sync_timestamps(self) -> None:
        """Persist sync timestamps to JSON file."""
        try:
            self._SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                node_id: cap.last_sync_time
                for node_id, cap in self._node_capabilities.items()
                if cap.last_sync_time > 0
            }
            with open(self._SYNC_STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except (
            FileNotFoundError,     # Parent directory doesn't exist (shouldn't happen with mkdir)
            PermissionError,       # Write access denied
            OSError,               # Disk full or other I/O error
            TypeError,             # Non-serializable data (shouldn't happen with float timestamps)
        ) as e:
            logger.warning(f"[SyncRouter] Failed to save sync timestamps: {e}")

    def record_sync_success(self, node_id: str) -> None:
        """Record successful sync to a node, updating timestamp.

        Args:
            node_id: The node that was successfully synced to
        """
        if node_id in self._node_capabilities:
            self._node_capabilities[node_id].last_sync_time = time.time()
            self._save_sync_timestamps()
            logger.debug(f"[SyncRouter] Recorded sync success for {node_id}")

    def should_sync_to_node(
        self,
        target_node: str,
        data_type: str | DataType,
        source_node: str | None = None,
    ) -> bool:
        """Check if data should be synced to a specific node.

        Args:
            target_node: Target node ID
            data_type: Type of data
            source_node: Source node (defaults to current node)

        Returns:
            True if sync should proceed
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type)

        source = source_node or self.node_id

        # Can't sync to self
        if target_node == source:
            return False

        # Check node capabilities
        cap = self._node_capabilities.get(target_node)
        if not cap:
            # Unknown node - use manifest policy
            return self._manifest.can_receive_data(target_node, data_type)

        # Check data type permissions
        if not self._can_receive_data_type(cap, data_type):
            return False

        # Check capacity
        if not self._check_node_capacity(target_node):
            return False

        # Check NFS sharing
        if self._shares_storage_with(target_node):
            return False

        return True

    def get_optimal_source(
        self,
        game_id: str,
        target_node: str,
    ) -> str | None:
        """Find the optimal source node for syncing a game.

        Considers:
        - Network proximity (same provider)
        - Load balancing
        - Bandwidth availability

        Args:
            game_id: Game to sync
            target_node: Destination node

        Returns:
            Best source node ID or None if not found
        """
        # Find locations for this game
        locations = self._manifest.find_game(game_id)
        if not locations:
            return None

        target_cap = self._node_capabilities.get(target_node)
        target_provider = target_cap.provider if target_cap else "unknown"

        candidates: list[tuple[str, int]] = []

        for loc in locations:
            source = loc.node_id
            if source == target_node:
                continue

            source_cap = self._node_capabilities.get(source)
            score = 50  # Base score

            if source_cap:
                # Prefer same provider (lower latency)
                if source_cap.provider == target_provider:
                    score += 20

                # Avoid ephemeral nodes as sources if possible
                if source_cap.is_ephemeral:
                    score -= 10

                # Prefer nodes with low disk usage (less IO contention)
                if source_cap.disk_usage_percent < 50:
                    score += 10

            candidates.append((source, score))

        if not candidates:
            return None

        # Return highest scoring source
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def plan_replication(
        self,
        game_id: str,
        min_copies: int = 2,
        quality_score: float | None = None,
    ) -> list[SyncRoute]:
        """Plan replication routes for a game.

        December 2025: Now supports quality-based priority boost.
        High-quality games get synced first for faster training data availability.

        Args:
            game_id: Game to replicate
            min_copies: Minimum number of copies desired
            quality_score: Optional quality score (0-1). If not provided,
                          will attempt to fetch from manifest.

        Returns:
            List of SyncRoute describing the sync plan
        """
        routes: list[SyncRoute] = []

        # Get current locations
        locations = self._manifest.find_game(game_id)
        current_nodes = {loc.node_id for loc in locations}

        copies_needed = min_copies - len(current_nodes)
        if copies_needed <= 0:
            return routes

        # Get quality score if not provided (Dec 2025: Quality-based priority)
        if quality_score is None:
            quality_score = self._get_game_quality_score(game_id)

        # Compute quality priority boost (0-30 points based on quality)
        # High-quality games get significant priority boost for faster sync
        quality_priority_boost = int(quality_score * 30) if quality_score else 0

        # Get targets
        targets = self._manifest.get_replication_targets(
            game_id,
            min_copies=min_copies,
            exclude_nodes=list(current_nodes),
        )

        for target in targets:
            # Find best source
            source = self.get_optimal_source(game_id, target.node_id)
            if not source:
                source = list(current_nodes)[0] if current_nodes else self.node_id

            # Apply quality boost to priority
            adjusted_priority = target.priority + quality_priority_boost

            routes.append(SyncRoute(
                source_node=source,
                target_node=target.node_id,
                data_type=DataType.GAME,
                priority=adjusted_priority,
                reason=target.reason + (f" (quality={quality_score:.2f})" if quality_score else ""),
                quality_score=quality_score or 0.0,
            ))

        # Sort by quality-adjusted priority (highest first)
        routes.sort(key=lambda r: r.priority, reverse=True)

        return routes

    def _get_game_quality_score(self, game_id: str) -> float:
        """Get quality score for a game from manifest or compute it.

        December 2025: Integrates with unified_quality for quality-based sync.
        """
        # Try manifest first (metadata might have quality)
        try:
            metadata = self._manifest.get_game_metadata(game_id)
            if metadata and hasattr(metadata, 'quality_score'):
                return metadata.quality_score or 0.0
        except (ValueError, KeyError, AttributeError) as e:
            # Narrow to data access errors (December 2025 exception narrowing)
            logger.debug(f"Failed to get quality score from manifest for {game_id}: {e}")

        # Try unified quality scorer
        try:
            from app.quality.unified_quality import get_quality_scorer

            # Get game details from manifest
            locations = self._manifest.find_game(game_id)
            if locations:
                loc = locations[0]
                if hasattr(loc, 'game_length') and hasattr(loc, 'winner'):
                    scorer = get_quality_scorer()
                    quality = scorer.compute_game_quality(
                        game_id=game_id,
                        game_length=getattr(loc, 'game_length', 50),
                        winner=getattr(loc, 'winner', None),
                        avg_player_elo=getattr(loc, 'avg_elo', 1200.0),
                    )
                    return quality.overall_score
        except ImportError:
            pass
        except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as e:
            # Quality scorer or computation errors (December 2025 exception narrowing)
            logger.debug(f"[SyncRouter] Could not get quality for {game_id}: {e}")

        return 0.5  # Default neutral quality

    def get_node_capability(self, node_id: str) -> NodeSyncCapability | None:
        """Get sync capability information for a node."""
        return self._node_capabilities.get(node_id)

    def get_status(self) -> dict[str, Any]:
        """Get router status."""
        return {
            "node_id": self.node_id,
            "total_nodes": len(self._node_capabilities),
            "training_nodes": sum(
                1 for c in self._node_capabilities.values() if c.is_training_node
            ),
            "priority_nodes": sum(
                1 for c in self._node_capabilities.values() if c.is_priority_node
            ),
            "ephemeral_nodes": sum(
                1 for c in self._node_capabilities.values() if c.is_ephemeral
            ),
            "nfs_nodes": sum(
                1 for c in self._node_capabilities.values() if c.shares_nfs
            ),
        }

    def health_check(self) -> HealthCheckResult:
        """Check health status of SyncRouter.

        December 27, 2025: Added to meet P2P manager health_check() standard.
        Updated to return HealthCheckResult (Dec 27, 2025).

        Returns:
            HealthCheckResult with status, node counts, and configuration health.
        """
        coordinator_status = CoordinatorStatus.RUNNING
        errors_count = 0
        message = ""

        # Check node capabilities are loaded
        total_nodes = len(self._node_capabilities)
        if total_nodes == 0:
            coordinator_status = CoordinatorStatus.DEGRADED
            message = "No node capabilities loaded"

        # Check manifest availability
        manifest_healthy = False
        try:
            manifest_healthy = self._manifest is not None and hasattr(
                self._manifest, "find_game"
            )
        except (AttributeError, TypeError) as e:
            # Manifest access errors (December 2025 exception narrowing)
            coordinator_status = CoordinatorStatus.STOPPED
            message = f"Manifest error: {e}"
            errors_count += 1

        if not manifest_healthy and coordinator_status == CoordinatorStatus.RUNNING:
            coordinator_status = CoordinatorStatus.DEGRADED
            message = "Cluster manifest not available"

        # Count enabled vs disabled nodes
        enabled_nodes = sum(
            1
            for c in self._node_capabilities.values()
            if c.can_receive_games or c.can_receive_models or c.can_receive_npz
        )

        # If all nodes are disabled, that's degraded
        if enabled_nodes == 0 and total_nodes > 0:
            coordinator_status = CoordinatorStatus.DEGRADED
            message = "All nodes have sync disabled"

        # Phase 2 (Jan 2026): Check S3 health
        s3_healthy = self._check_s3_health() if self._s3_config.enabled else None

        healthy = coordinator_status == CoordinatorStatus.RUNNING
        return HealthCheckResult(
            healthy=healthy,
            status=coordinator_status,
            message=message,
            details={
                "operations_count": total_nodes,
                "errors_count": errors_count,
                "total_nodes": total_nodes,
                "enabled_nodes": enabled_nodes,
                "manifest_available": manifest_healthy,
                "training_nodes": sum(
                    1 for c in self._node_capabilities.values() if c.is_training_node
                ),
                "priority_nodes": sum(
                    1 for c in self._node_capabilities.values() if c.is_priority_node
                ),
                # Phase 2: S3 status (Jan 2026)
                "s3_enabled": self._s3_config.enabled,
                "s3_healthy": s3_healthy,
                "s3_bucket": self._s3_config.bucket if self._s3_config.enabled else None,
                "s3_primary_for_games": self._s3_config.primary_for_games,
            },
        )

    # =========================================================================
    # Event Integration (December 2025)
    # =========================================================================

    def wire_to_event_router(self) -> None:
        """Wire this router to the event system.

        Subscribes to:
        - NEW_GAMES_AVAILABLE: Route new games to appropriate nodes
        - TRAINING_STARTED: Prioritize training nodes
        - HOST_ONLINE/OFFLINE: Update node capabilities
        - NODE_RECOVERED: Re-enable sync to recovered nodes (Dec 2025)
        - CLUSTER_CAPACITY_CHANGED: React to cluster membership changes
        """
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()

            # Subscribe to game events
            router.subscribe(
                DataEventType.NEW_GAMES_AVAILABLE.value,
                self._on_new_games_available,
            )

            # Subscribe to training events
            router.subscribe(
                DataEventType.TRAINING_STARTED.value,
                self._on_training_started,
            )

            # Subscribe to host events
            router.subscribe(
                DataEventType.HOST_ONLINE.value,
                self._on_host_online,
            )
            router.subscribe(
                DataEventType.HOST_OFFLINE.value,
                self._on_host_offline,
            )

            # Dec 2025: Subscribe to NODE_RECOVERED to re-enable sync
            router.subscribe(
                DataEventType.NODE_RECOVERED.value,
                self._on_node_recovered,
            )

            # Subscribe to cluster capacity changes (Dec 2025 - P2P integration)
            router.subscribe(
                DataEventType.CLUSTER_CAPACITY_CHANGED.value,
                self._on_cluster_capacity_changed,
            )

            # Dec 2025: Subscribe to MODEL_SYNC_REQUESTED to trigger model re-download
            router.subscribe(
                DataEventType.MODEL_SYNC_REQUESTED.value,
                self._on_model_sync_requested,
            )

            # Dec 2025: Subscribe to SYNC_STALLED to track slow/unreliable nodes
            router.subscribe(
                DataEventType.SYNC_STALLED.value,
                self._on_sync_stalled,
            )

            # Dec 29, 2025: Subscribe to SYNC_FAILURE_CRITICAL for multi-failure recovery
            router.subscribe(
                DataEventType.SYNC_FAILURE_CRITICAL.value,
                self._on_sync_failure_critical,
            )

            # Dec 27, 2025: Subscribe to backpressure events
            router.subscribe(
                DataEventType.BACKPRESSURE_ACTIVATED.value,
                self._on_backpressure_activated,
            )
            router.subscribe(
                DataEventType.BACKPRESSURE_RELEASED.value,
                self._on_backpressure_released,
            )

            # Dec 2025: Subscribe to CONFIG_UPDATED for cluster config sync
            router.subscribe(
                DataEventType.CONFIG_UPDATED.value,
                self._on_config_updated,
            )

            logger.info(
                "[SyncRouter] Wired to event router "
                "(NEW_GAMES_AVAILABLE, TRAINING_STARTED, HOST_ONLINE/OFFLINE, "
                "NODE_RECOVERED, CLUSTER_CAPACITY_CHANGED, MODEL_SYNC_REQUESTED, "
                "SYNC_STALLED, SYNC_FAILURE_CRITICAL, BACKPRESSURE_ACTIVATED/RELEASED, "
                "CONFIG_UPDATED)"
            )

        except ImportError as e:
            logger.warning(f"[SyncRouter] Event router not available: {e}")
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            # Specific errors from subscription configuration issues
            logger.error(f"[SyncRouter] Failed to wire to event router: {e}")

    async def _on_new_games_available(self, event: Any) -> None:
        """Handle NEW_GAMES_AVAILABLE event - route games to targets."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            game_count = payload.get("count", 0)
            source_node = payload.get("source", self.node_id)

            if game_count > 0:
                # Get sync targets for the new games
                targets = self.get_sync_targets(
                    data_type=DataType.GAME,
                    exclude_nodes=[source_node],
                    max_targets=5,
                )

                if targets:
                    logger.info(
                        f"[SyncRouter] New games ({game_count}) from {source_node}, "
                        f"routing to {len(targets)} targets"
                    )

                    # Emit sync routing decision
                    await self._emit_sync_routing_decision(
                        source=source_node,
                        targets=[t.node_id for t in targets],
                        data_type=DataType.GAME,
                        reason=f"new_games:{game_count}",
                    )

        except (KeyError, TypeError, AttributeError, ValueError) as e:
            # Event payload or routing errors
            logger.error(f"[SyncRouter] Error handling new games event: {e}")

    async def _on_training_started(self, event: Any) -> None:
        """Handle TRAINING_STARTED event - mark node as training priority."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            node_id = payload.get("node_id") or payload.get("host")

            if node_id and node_id in self._node_capabilities:
                cap = self._node_capabilities[node_id]
                cap.is_training_node = True
                cap.is_priority_node = True
                logger.info(f"[SyncRouter] Marked {node_id} as training priority")

        except (KeyError, TypeError, AttributeError) as e:
            # Event payload access errors
            logger.error(f"[SyncRouter] Error handling training started: {e}")

    async def _on_host_online(self, event: Any) -> None:
        """Handle HOST_ONLINE event - add/update node capabilities."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            node_id = payload.get("host")

            if node_id and node_id not in self._node_capabilities:
                # Create default capability for new node
                cap = NodeSyncCapability(node_id=node_id)
                self._node_capabilities[node_id] = cap
                logger.info(f"[SyncRouter] Added new node: {node_id}")

        except (KeyError, TypeError, AttributeError) as e:
            # Event payload access errors
            logger.debug(f"[SyncRouter] Error handling host online: {e}")

    async def _on_host_offline(self, event: Any) -> None:
        """Handle HOST_OFFLINE event - mark node as unavailable."""
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            node_id = payload.get("host")

            if node_id and node_id in self._node_capabilities:
                # Mark node as unavailable rather than removing
                cap = self._node_capabilities[node_id]
                cap.can_receive_games = False
                cap.can_receive_models = False
                cap.can_receive_npz = False
                logger.info(f"[SyncRouter] Marked {node_id} as offline")

        except (KeyError, TypeError, AttributeError) as e:
            # Event payload access errors
            logger.debug(f"[SyncRouter] Error handling host offline: {e}")

    async def _on_node_recovered(self, event: Any) -> None:
        """Handle NODE_RECOVERED event - re-enable sync to recovered node.

        December 2025: Added to complete health event integration.
        When a node recovers after being offline, restore its sync capabilities.
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            node_id = payload.get("node_id") or payload.get("host")

            if node_id and node_id in self._node_capabilities:
                cap = self._node_capabilities[node_id]
                # Re-enable based on original policy from manifest
                policy = self._manifest.get_sync_policy(node_id)
                cap.can_receive_games = policy.receive_games
                cap.can_receive_models = policy.receive_models
                cap.can_receive_npz = policy.receive_npz
                logger.info(
                    f"[SyncRouter] Restored sync capabilities for recovered node: {node_id}"
                )
            elif node_id:
                # New node recovered - add with default capabilities
                cap = NodeSyncCapability(node_id=node_id)
                self._node_capabilities[node_id] = cap
                logger.info(f"[SyncRouter] Added recovered node: {node_id}")

        except (KeyError, TypeError, AttributeError) as e:
            # Event payload or manifest access errors
            logger.debug(f"[SyncRouter] Error handling node recovered: {e}")

    async def _on_cluster_capacity_changed(self, event: Any) -> None:
        """Handle CLUSTER_CAPACITY_CHANGED event - refresh capacity data and recalculate routes.

        December 2025: Enables real-time reaction to cluster membership changes.
        When nodes join or leave, we refresh capacity data to ensure sync
        targets are current and appropriately prioritized.

        Args:
            event: Event with payload containing change_type, node_id, total_nodes, gpu_nodes
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            change_type = payload.get("change_type", "unknown")
            node_id = payload.get("node_id", "unknown")
            total_nodes = payload.get("total_nodes", 0)
            gpu_nodes = payload.get("gpu_nodes", 0)
            reason = payload.get("reason", "")

            logger.info(
                f"[SyncRouter] Cluster capacity changed: {change_type} "
                f"node={node_id}, total={total_nodes}, gpu={gpu_nodes}, reason={reason}"
            )

            # Refresh capacity data from manifest
            if self._manifest:
                self._manifest.refresh_capacity_data()

            # Update node capability based on change
            if change_type == "node_removed":
                if node_id in self._node_capabilities:
                    cap = self._node_capabilities[node_id]
                    cap.can_receive_games = False
                    cap.can_receive_models = False
                    cap.can_receive_npz = False
                    logger.debug(f"[SyncRouter] Disabled sync to removed node: {node_id}")

            elif change_type == "node_added":
                if node_id not in self._node_capabilities:
                    # Add new node with default capabilities
                    self._node_capabilities[node_id] = NodeSyncCapability(node_id=node_id)
                else:
                    # Re-enable existing node
                    cap = self._node_capabilities[node_id]
                    cap.can_receive_games = True
                    cap.can_receive_models = True
                    cap.can_receive_npz = True
                logger.debug(f"[SyncRouter] Enabled sync to added node: {node_id}")

            # Emit capacity refresh event for downstream consumers
            await self._emit_capacity_refresh(
                change_type=change_type,
                node_id=node_id,
                total_nodes=total_nodes,
                gpu_nodes=gpu_nodes,
            )

        except (KeyError, TypeError, AttributeError, ValueError) as e:
            # Event payload or node capability update errors
            logger.warning(f"[SyncRouter] Error handling cluster capacity changed: {e}")

    async def _on_model_sync_requested(self, event: Any) -> None:
        """Handle MODEL_SYNC_REQUESTED event - trigger model re-download from healthy nodes.

        December 2025: Wired to address critical gap where model sync requests were
        emitted but had no subscribers. This handler routes models from healthy nodes
        to requesting nodes.

        Args:
            event: Event with payload containing model_id, requesting_node, reason
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            model_id = payload.get("model_id", "")
            requesting_node = payload.get("node_id", "") or payload.get("requesting_node", "")
            reason = payload.get("reason", "sync_requested")

            if not model_id or not requesting_node:
                logger.debug(
                    f"[SyncRouter] MODEL_SYNC_REQUESTED missing model_id or node: {payload}"
                )
                return

            logger.info(
                f"[SyncRouter] Model sync requested: {model_id} for {requesting_node}, "
                f"reason: {reason}"
            )

            # Find nodes that have this model and can serve as sources
            sources = self.get_sync_targets(
                data_type=DataType.MODEL,
                exclude_nodes=[requesting_node],
                max_targets=3,
            )

            if not sources:
                logger.warning(
                    f"[SyncRouter] No sources found for model {model_id}"
                )
                return

            # Select the best source (first one - highest priority)
            source_node = sources[0].node_id

            logger.info(
                f"[SyncRouter] Routing model {model_id} from {source_node} "
                f"to {requesting_node}"
            )

            # Emit sync routing decision
            await self._emit_sync_routing_decision(
                source=source_node,
                targets=[requesting_node],
                data_type=DataType.MODEL,
                reason=f"model_sync:{model_id}:{reason}",
            )

            # Also emit a MODEL_SYNC_STARTED event for tracking
            try:
                from app.coordination.event_router import get_router

                router = get_router()
                await router.publish(
                    "MODEL_SYNC_STARTED",
                    {
                        "model_id": model_id,
                        "source_node": source_node,
                        "target_node": requesting_node,
                        "reason": reason,
                    },
                )
            except (ImportError, RuntimeError, OSError) as e:
                # Event emission infrastructure errors
                logger.debug(f"[SyncRouter] Could not emit MODEL_SYNC_STARTED: {e}")

        except (KeyError, TypeError, AttributeError, ValueError) as e:
            # Event payload or routing errors
            logger.error(f"[SyncRouter] Error handling model sync request: {e}")

    # Transport escalation order (Dec 28, 2025)
    # When a transport fails, try the next one in the chain
    TRANSPORT_ESCALATION_ORDER = ["p2p", "http", "rsync", "base64"]

    async def _on_sync_stalled(self, event: Any) -> None:
        """Handle SYNC_STALLED event - escalate to alternate transport.

        December 2025: Added to complete sync reliability integration.
        December 28, 2025: Fixed to include transport escalation.
        When a sync operation times out, try the next transport in the chain
        before disabling the node entirely.

        Transport escalation order: P2P -> HTTP -> rsync -> base64
        Only disable sync after all transports have failed.

        Args:
            event: Event with payload containing target_host, timeout_seconds,
                   retry_count, failed_transport
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            target_node = payload.get("target_host", "")
            timeout_seconds = payload.get("timeout_seconds", 0)
            retry_count = payload.get("retry_count", 0)
            data_type = payload.get("data_type", "game")
            failed_transport = payload.get("transport", "unknown")

            if not target_node:
                return

            # Initialize transport failure tracking
            if not hasattr(self, '_failed_transports'):
                self._failed_transports: dict[str, set[str]] = {}
            if not hasattr(self, '_stall_counts'):
                self._stall_counts: dict[str, int] = {}

            # Track this failure
            if target_node not in self._failed_transports:
                self._failed_transports[target_node] = set()

            if failed_transport != "unknown":
                self._failed_transports[target_node].add(failed_transport)

            self._stall_counts[target_node] = self._stall_counts.get(target_node, 0) + 1
            stall_count = self._stall_counts[target_node]

            # Find next transport to try
            failed_set = self._failed_transports.get(target_node, set())
            next_transport = None
            for transport in self.TRANSPORT_ESCALATION_ORDER:
                if transport not in failed_set:
                    next_transport = transport
                    break

            if next_transport:
                # There's another transport to try - emit SYNC_RETRY_WITH_TRANSPORT
                logger.info(
                    f"[SyncRouter] Sync stalled to {target_node} via {failed_transport}, "
                    f"escalating to {next_transport}"
                )
                try:
                    from app.coordination.event_router import safe_emit_event

                    safe_emit_event(
                        "SYNC_RETRY_REQUESTED",
                        {
                            "target_host": target_node,
                            "data_type": data_type,
                            "preferred_transport": next_transport,
                            "failed_transports": list(failed_set),
                        },
                        source="SyncRouter",
                    )
                except (ImportError, RuntimeError, OSError, AttributeError) as e:
                    # Event emission infrastructure errors (December 2025 exception narrowing)
                    logger.debug(f"[SyncRouter] Could not emit SYNC_RETRY_REQUESTED: {e}")
            else:
                # All transports have failed - disable sync to this node
                logger.warning(
                    f"[SyncRouter] All transports failed to {target_node}: "
                    f"timeout={timeout_seconds}s, retries={retry_count}, type={data_type}"
                )

                if target_node in self._node_capabilities:
                    cap = self._node_capabilities[target_node]
                    # Disable based on data type that stalled
                    if data_type == "game":
                        cap.can_receive_games = False
                    elif data_type == "model":
                        cap.can_receive_models = False
                    elif data_type == "npz":
                        cap.can_receive_npz = False
                    else:
                        # Disable all sync for generic stalls
                        cap.can_receive_games = False
                        cap.can_receive_models = False
                        cap.can_receive_npz = False

                    logger.warning(
                        f"[SyncRouter] Disabled {data_type} sync to {target_node} "
                        f"after exhausting all transports (stalls={stall_count})"
                    )

        except (AttributeError, KeyError) as e:
            logger.debug(f"[SyncRouter] Error handling sync stalled: {e}")

    async def _on_sync_failure_critical(self, event: Any) -> None:
        """Handle SYNC_FAILURE_CRITICAL event - trigger emergency sync recovery.

        December 29, 2025: Added to handle sustained sync failures.
        When multiple consecutive sync failures occur, this indicates a systemic
        issue that requires intervention:
        - Alert operators via logging
        - Reset failed transport tracking to retry all transports
        - Emit CLUSTER_HEALTH_DEGRADED if threshold exceeded

        Args:
            event: Event with payload containing consecutive_failures,
                   last_success, time_since_success_seconds
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            consecutive_failures = payload.get("consecutive_failures", 0)
            time_since_success = payload.get("time_since_success_seconds")
            source = payload.get("source", "unknown")

            logger.error(
                f"[SyncRouter] CRITICAL: {consecutive_failures} consecutive sync failures "
                f"from {source}. "
                + (f"Last success was {time_since_success:.0f}s ago. " if time_since_success else "")
                + "Initiating recovery..."
            )

            # Reset failed transport tracking to give nodes a fresh chance
            if hasattr(self, '_failed_transports'):
                reset_count = len(self._failed_transports)
                self._failed_transports.clear()
                logger.info(
                    f"[SyncRouter] Reset failed transport tracking for {reset_count} nodes"
                )

            # Reset stall counts
            if hasattr(self, '_stall_counts'):
                self._stall_counts.clear()

            # Re-enable sync for nodes that were disabled
            for node_id, cap in self._node_capabilities.items():
                if not cap.can_receive_games or not cap.can_receive_models or not cap.can_receive_npz:
                    cap.can_receive_games = True
                    cap.can_receive_models = True
                    cap.can_receive_npz = True
                    logger.info(f"[SyncRouter] Re-enabled sync for {node_id}")

            # If failures are severe, emit cluster health degraded
            if consecutive_failures >= 5:
                try:
                    from app.coordination.event_router import safe_emit_event

                    safe_emit_event(
                        "CLUSTER_HEALTH_DEGRADED",
                        {
                            "reason": "sync_failure_critical",
                            "consecutive_failures": consecutive_failures,
                            "time_since_success_seconds": time_since_success,
                        },
                        source="SyncRouter",
                    )
                except (ImportError, RuntimeError, OSError, AttributeError) as emit_err:
                    logger.debug(f"[SyncRouter] Could not emit CLUSTER_HEALTH_DEGRADED: {emit_err}")

        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"[SyncRouter] Error handling sync failure critical: {e}")

    async def _on_backpressure_activated(self, event: Any) -> None:
        """Handle BACKPRESSURE_ACTIVATED event - track state but DO NOT reduce sync priority.

        December 27, 2025: Added to prevent overwhelming nodes under pressure.
        December 28, 2025 FIX: Removed priority reduction that was causing feedback loops.

        CRITICAL: We MUST NOT reduce sync priority when backpressure is activated.
        Reducing sync priority causes training data to become stale, which causes
        more training failures, which causes circuit breakers to stay open longer,
        creating a self-reinforcing negative feedback loop.

        Instead, we only track backpressure state for monitoring/alerting.
        Data sync should ALWAYS continue regardless of backpressure state.

        Args:
            event: Event with payload containing source_node, queue_depth, threshold
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            source_node = payload.get("source_node", "")
            queue_depth = payload.get("queue_depth", 0)
            threshold = payload.get("threshold", 100)

            # Track backpressure state for monitoring
            if not hasattr(self, '_backpressure_active'):
                self._backpressure_active: set[str] = set()

            if source_node:
                self._backpressure_active.add(source_node)
                logger.warning(
                    f"[SyncRouter] Backpressure activated on {source_node}: "
                    f"queue_depth={queue_depth}, threshold={threshold}. "
                    f"NOTE: Sync priority NOT reduced to avoid feedback loops."
                )
                # Dec 28, 2025: REMOVED priority reduction code that was causing feedback loops
                # The old code reduced sync priority, which made data stale, causing more failures.
                # See CLAUDE.md "Circuit Breaker Feedback Loop Fix" for details.
            else:
                # Global backpressure - just track it
                self._backpressure_active.add("__global__")
                logger.warning(
                    f"[SyncRouter] Global backpressure activated: "
                    f"queue_depth={queue_depth}. Sync continues normally."
                )

        except (AttributeError, KeyError) as e:
            logger.debug(f"[SyncRouter] Error handling backpressure activated: {e}")

    async def _on_backpressure_released(self, event: Any) -> None:
        """Handle BACKPRESSURE_RELEASED event - clear tracking state.

        December 27, 2025: Added to restore sync throughput after recovery.
        December 28, 2025: Simplified - sync was never throttled, just tracking cleared.

        Args:
            event: Event with payload containing source_node
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            source_node = payload.get("source_node", "")

            if not hasattr(self, '_backpressure_active'):
                return

            if source_node:
                self._backpressure_active.discard(source_node)
                logger.info(
                    f"[SyncRouter] Backpressure released on {source_node}"
                )
                # Dec 28, 2025: No priority restoration needed - we don't reduce priority anymore
            else:
                # Global backpressure released
                self._backpressure_active.discard("__global__")
                logger.info("[SyncRouter] Global backpressure released")

        except (AttributeError, KeyError) as e:
            logger.debug(f"[SyncRouter] Error handling backpressure released: {e}")

    async def _on_config_updated(self, event: Any) -> None:
        """Handle CONFIG_UPDATED event - reload cluster config when updated.

        December 30, 2025: Added as part of distributed config sync infrastructure.
        When a peer has a newer config version (detected via gossip), it triggers
        a config pull and emits this event. Subscribers should reload their cached
        config to pick up the changes.

        Args:
            event: Event with payload containing source_node, timestamp
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event
            source_node = payload.get("source_node", "unknown")
            timestamp = payload.get("timestamp", 0)

            logger.info(
                f"[SyncRouter] Config updated from {source_node} at {timestamp:.0f}, "
                "reloading cluster config"
            )

            # Force reload cluster config to pick up changes
            try:
                from app.config.cluster_config import get_config_cache

                cache = get_config_cache()
                cache.get_config(force_reload=True)
                logger.debug("[SyncRouter] Successfully reloaded cluster config")
            except ImportError:
                logger.warning("[SyncRouter] Could not import config cache for reload")
            except (OSError, ValueError) as reload_err:
                logger.warning(f"[SyncRouter] Config reload failed: {reload_err}")

            # Refresh node capabilities from new config
            self._refresh_from_config()

        except (AttributeError, KeyError, TypeError) as e:
            logger.debug(f"[SyncRouter] Error handling config updated: {e}")

    def is_under_backpressure(self, node_id: str = "") -> bool:
        """Check if a node (or globally) is under backpressure.

        Args:
            node_id: Specific node to check, or empty for global check

        Returns:
            True if under backpressure
        """
        if not hasattr(self, '_backpressure_active'):
            return False

        if "__global__" in self._backpressure_active:
            return True

        if node_id:
            return node_id in self._backpressure_active

        return len(self._backpressure_active) > 0

    async def _emit_capacity_refresh(
        self,
        change_type: str,
        node_id: str,
        total_nodes: int,
        gpu_nodes: int,
    ) -> None:
        """Emit a capacity refresh event for downstream consumers."""
        try:
            from app.coordination.event_router import get_router

            router = get_router()
            router.publish(
                "SYNC_CAPACITY_REFRESHED",
                {
                    "change_type": change_type,
                    "node_id": node_id,
                    "total_nodes": total_nodes,
                    "gpu_nodes": gpu_nodes,
                    "router": "SyncRouter",
                },
            )
        except (ImportError, RuntimeError, OSError, AttributeError) as e:
            # Event emission infrastructure errors
            logger.debug(f"[SyncRouter] Could not emit capacity refresh: {e}")

    async def _emit_sync_routing_decision(
        self,
        source: str,
        targets: list[str],
        data_type: DataType,
        reason: str,
    ) -> None:
        """Emit a sync routing decision event."""
        try:
            from app.coordination.event_router import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )

            # Dec 2025: Explicit null check before publish
            bus = get_event_bus()
            if bus is not None:
                await bus.publish(DataEvent(
                    event_type=DataEventType.SYNC_REQUEST.value,
                    payload={
                        "source": source,
                        "targets": targets,
                        "data_type": data_type.value,
                        "reason": reason,
                        "router": "SyncRouter",
                    },
                    source="SyncRouter",
                ))

        except (ImportError, RuntimeError, OSError, AttributeError) as e:
            # Event emission infrastructure errors
            logger.warning(f"[SyncRouter] Could not emit routing decision: {e}")
            # Still log the decision for debugging even if event emission failed
            logger.info(
                f"[SyncRouter] Routing decision: {len(targets)} targets for "
                f"{data_type.value} data from {source}"
            )


# Module-level singleton
_sync_router: SyncRouter | None = None


def get_sync_router() -> SyncRouter:
    """Get the singleton SyncRouter instance."""
    global _sync_router
    if _sync_router is None:
        _sync_router = SyncRouter()
    return _sync_router


def reset_sync_router() -> None:
    """Reset the singleton (for testing)."""
    global _sync_router
    _sync_router = None
