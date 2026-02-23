"""Sync strategies and configuration for AutoSyncDaemon (December 2025).

This module contains the sync strategy definitions and configuration classes
extracted from auto_sync_daemon.py for better modularity.

Classes:
    SyncStrategy          - Enum-like class for sync mode selection
    AutoSyncConfig        - Configuration dataclass with all sync settings
    SyncStats             - Statistics tracking (extends SyncDaemonStats)

Constants:
    MIN_MOVES_PER_GAME    - Minimum moves per game for completeness validation
    DEFAULT_MIN_MOVES     - Fallback for unknown configurations

Usage:
    from app.coordination.sync_strategies import (
        SyncStrategy,
        AutoSyncConfig,
        SyncStats,
    )

    config = AutoSyncConfig.from_config_file()
    config.strategy = SyncStrategy.EPHEMERAL
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# December 2025: Use consolidated daemon stats base class
from app.coordination.daemon_stats import SyncDaemonStats

logger = logging.getLogger(__name__)

# Import centralized thresholds for quality filtering
try:
    from app.config.thresholds import (
        HIGH_QUALITY_THRESHOLD,
        SYNC_MIN_QUALITY,
        SYNC_QUALITY_SAMPLE_SIZE,
    )
except ImportError:
    HIGH_QUALITY_THRESHOLD = 0.7
    SYNC_MIN_QUALITY = 0.5
    SYNC_QUALITY_SAMPLE_SIZE = 20


class SyncStrategy:
    """Sync strategy enum for AutoSyncDaemon (December 2025 consolidation).

    Strategies:
    - HYBRID: Default. Push-from-generator + gossip replication (persistent hosts)
    - EPHEMERAL: Aggressive 5-second sync for Vast.ai/spot instances
    - BROADCAST: Leader-only push to all eligible nodes
    - PULL: Coordinator pulls data from cluster (for recovery/backup)
    - AUTO: Auto-detect based on node type (ephemeral detection, leader status)
    """
    HYBRID = "hybrid"
    EPHEMERAL = "ephemeral"
    BROADCAST = "broadcast"
    PULL = "pull"  # December 2025: Coordinator-side reverse sync
    AUTO = "auto"


# Minimum moves per game for completeness validation (December 2025)
# Games with fewer moves are considered incomplete and skipped during sync
MIN_MOVES_PER_GAME: dict[tuple[str, int], int] = {
    ("hex8", 2): 20, ("hex8", 3): 30, ("hex8", 4): 40,
    ("square8", 2): 20, ("square8", 3): 30, ("square8", 4): 40,
    ("square19", 2): 50, ("square19", 3): 80, ("square19", 4): 100,
    ("hexagonal", 2): 50, ("hexagonal", 3): 80, ("hexagonal", 4): 100,
}
DEFAULT_MIN_MOVES = 5  # Fallback for unknown configurations


@dataclass
class AutoSyncConfig:
    """Configuration for automated data sync.

    Quality thresholds are loaded from app.config.thresholds for centralized configuration.

    December 2025: Added strategy parameter for consolidated sync modes:
    - hybrid: Push-from-generator + gossip (default)
    - ephemeral: Aggressive 5s sync for Vast.ai/spot
    - broadcast: Leader-only push to all nodes
    - auto: Detect based on node type
    """
    enabled: bool = True
    # Strategy selection (December 2025 consolidation)
    strategy: str = SyncStrategy.AUTO  # auto, hybrid, ephemeral, broadcast
    interval_seconds: int = 60  # December 2025: Reduced from 300s for faster data discovery
    gossip_interval_seconds: int = 30  # December 2025: Reduced from 60s for faster replication
    exclude_hosts: list[str] = field(default_factory=list)
    skip_nfs_sync: bool = True
    # Feb 2026: Limited to 1 to prevent OOM from parallel rsyncs
    max_concurrent_syncs: int = int(os.getenv("RINGRIFT_AUTO_SYNC_MAX_CONCURRENT", "1"))
    min_games_to_sync: int = 10
    bandwidth_limit_mbps: int = 20
    # Disk usage thresholds (from sync_routing)
    max_disk_usage_percent: float = 85.0
    target_disk_usage_percent: float = 60.0
    # Enable automatic disk cleanup
    auto_cleanup_enabled: bool = True
    # Use ClusterManifest for tracking
    use_cluster_manifest: bool = True
    # Quality-based sync filtering - from centralized config
    quality_filter_enabled: bool = True
    min_quality_for_sync: float = SYNC_MIN_QUALITY
    quality_sample_size: int = SYNC_QUALITY_SAMPLE_SIZE
    # Quality extraction for priority-based training
    enable_quality_extraction: bool = True
    min_quality_score_for_priority: float = HIGH_QUALITY_THRESHOLD  # From thresholds.py
    # Ephemeral-specific settings (December 2025 consolidation)
    ephemeral_poll_seconds: int = 5  # Aggressive polling for ephemeral hosts
    ephemeral_write_through: bool = True  # Wait for push confirmation
    ephemeral_write_through_timeout: int = 60  # Max wait for confirmation
    ephemeral_wal_enabled: bool = True  # Write-ahead log for durability
    # Broadcast-specific settings (December 2025 consolidation)
    broadcast_high_priority_configs: list[str] = field(
        default_factory=lambda: ["square8_2p", "hex8_2p", "hex8_3p", "hex8_4p"]
    )

    @classmethod
    def from_config_file(cls, config_path: Path | None = None) -> AutoSyncConfig:
        """Load configuration from distributed_hosts.yaml or unified_loop.yaml."""
        from app.config.cluster_config import load_cluster_config

        config = cls()

        # Load from distributed_hosts.yaml via cluster_config helper
        try:
            cluster_cfg = load_cluster_config(config_path)

            # Get sync_routing settings
            config.max_disk_usage_percent = cluster_cfg.sync_routing.max_disk_usage_percent
            config.target_disk_usage_percent = cluster_cfg.sync_routing.target_disk_usage_percent

            # Get auto_sync settings
            auto_sync = cluster_cfg.auto_sync
            config.enabled = auto_sync.enabled
            config.interval_seconds = auto_sync.interval_seconds
            config.gossip_interval_seconds = auto_sync.gossip_interval_seconds
            config.exclude_hosts = list(auto_sync.exclude_hosts)
            config.skip_nfs_sync = auto_sync.skip_nfs_sync
            config.max_concurrent_syncs = auto_sync.max_concurrent_syncs
            config.min_games_to_sync = auto_sync.min_games_to_sync
            config.bandwidth_limit_mbps = auto_sync.bandwidth_limit_mbps

            # December 27, 2025: Auto-exclude coordinator nodes and nodes with skip_sync_receive
            # This is Layer 2 of the multi-layer coordinator disk protection plan.
            # Previously, role: coordinator and skip_sync_receive: true in config were never
            # enforced. Now we automatically add these nodes to exclude_hosts.
            #
            # EXCEPTION: Nodes with use_external_storage: true are allowed to receive sync
            # because data is routed to external storage (e.g., mac-studio with OWC drive).
            for host_name, host_config in cluster_cfg.hosts_raw.items():
                if host_name in config.exclude_hosts:
                    continue  # Already excluded

                # Check skip_sync_receive flag - always exclude these
                if host_config.get("skip_sync_receive", False):
                    config.exclude_hosts.append(host_name)
                    logger.debug(
                        f"[AutoSyncConfig] Auto-excluding node with skip_sync_receive: {host_name}"
                    )
                    continue

                # Check role - coordinators should not receive synced data
                # UNLESS they have external storage configured
                #
                # December 2025: Added hostname pattern fallback for coordinator detection.
                # Some coordinator nodes may not have role: coordinator in config,
                # so we also check common coordinator hostname patterns.
                is_coordinator_by_role = host_config.get("role") == "coordinator"
                is_coordinator_by_flag = host_config.get("is_coordinator", False)
                is_coordinator_by_hostname = any(
                    pattern in host_name.lower()
                    for pattern in ["mac-studio", "localhost", "local-mac", "coordinator", "macbook"]
                )
                is_coordinator = is_coordinator_by_role or is_coordinator_by_flag or is_coordinator_by_hostname

                if is_coordinator:
                    # Allow if external storage is configured
                    if host_config.get("use_external_storage", False):
                        logger.debug(
                            f"[AutoSyncConfig] Allowing coordinator with external storage: {host_name}"
                        )
                        continue
                    # Exclude coordinators without external storage
                    config.exclude_hosts.append(host_name)
                    detection_method = (
                        "role" if is_coordinator_by_role else
                        "flag" if is_coordinator_by_flag else
                        "hostname"
                    )
                    logger.debug(
                        f"[AutoSyncConfig] Auto-excluding coordinator node: {host_name} "
                        f"(detected via {detection_method})"
                    )

        except (OSError, ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Failed to load cluster config: {e}")

        base_dir = Path(__file__).resolve().parent.parent.parent

        # Fallback to unified_loop.yaml
        unified_config_path = base_dir / "config" / "unified_loop.yaml"
        if unified_config_path.exists():
            try:
                with open(unified_config_path) as f:
                    data = yaml.safe_load(f)

                auto_sync = data.get("auto_sync", {})
                # Only override if not already set
                if not config.exclude_hosts:
                    config.exclude_hosts = auto_sync.get("exclude_hosts", [])

                # Also check data_aggregation.excluded_nodes for compatibility
                data_agg = data.get("data_aggregation", {})
                for node in data_agg.get("excluded_nodes", []):
                    if node not in config.exclude_hosts:
                        config.exclude_hosts.append(node)

            except (OSError, ValueError) as e:
                logger.warning(f"Failed to load unified_loop.yaml: {e}")

        return config


@dataclass
class SyncProgress:
    """Real-time progress tracking for sync operations (December 2025).

    Used to track the current state of an ongoing sync operation,
    enabling UI/monitoring to show progress and estimate completion time.

    Attributes:
        is_active: Whether a sync operation is currently in progress
        current_phase: Description of current phase (e.g., "collecting", "pushing")
        current_file: Name of file currently being synced
        current_node: Node currently being synced to/from
        files_completed: Number of files completed in this sync cycle
        files_total: Total files expected in this sync cycle
        bytes_transferred: Bytes transferred so far in this sync cycle
        bytes_total: Total bytes expected (0 if unknown)
        started_at: Unix timestamp when sync started
        last_update_at: Unix timestamp of last progress update
        estimated_completion_at: Estimated completion timestamp (0 if unknown)
        error_message: Current error message (empty if no error)
    """

    is_active: bool = False
    current_phase: str = ""
    current_file: str = ""
    current_node: str = ""
    files_completed: int = 0
    files_total: int = 0
    bytes_transferred: int = 0
    bytes_total: int = 0
    started_at: float = 0.0
    last_update_at: float = 0.0
    estimated_completion_at: float = 0.0
    error_message: str = ""

    @property
    def percent_complete(self) -> float:
        """Calculate percentage completion based on available metrics."""
        if self.files_total > 0:
            return (self.files_completed / self.files_total) * 100.0
        if self.bytes_total > 0:
            return (self.bytes_transferred / self.bytes_total) * 100.0
        return 0.0

    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since sync started."""
        if self.started_at <= 0:
            return 0.0
        import time
        return time.time() - self.started_at

    @property
    def transfer_rate_bytes_per_sec(self) -> float:
        """Current transfer rate in bytes per second."""
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.bytes_transferred / elapsed

    def to_dict(self) -> dict[str, Any]:
        """Convert progress to dictionary for JSON serialization."""
        return {
            "is_active": self.is_active,
            "current_phase": self.current_phase,
            "current_file": self.current_file,
            "current_node": self.current_node,
            "files_completed": self.files_completed,
            "files_total": self.files_total,
            "bytes_transferred": self.bytes_transferred,
            "bytes_total": self.bytes_total,
            "started_at": self.started_at,
            "last_update_at": self.last_update_at,
            "estimated_completion_at": self.estimated_completion_at,
            "error_message": self.error_message,
            # Computed properties
            "percent_complete": self.percent_complete,
            "elapsed_seconds": self.elapsed_seconds,
            "transfer_rate_bytes_per_sec": self.transfer_rate_bytes_per_sec,
        }

    def reset(self) -> None:
        """Reset all progress tracking for a new sync cycle."""
        self.is_active = False
        self.current_phase = ""
        self.current_file = ""
        self.current_node = ""
        self.files_completed = 0
        self.files_total = 0
        self.bytes_transferred = 0
        self.bytes_total = 0
        self.started_at = 0.0
        self.last_update_at = 0.0
        self.estimated_completion_at = 0.0
        self.error_message = ""


@dataclass
class SyncStats(SyncDaemonStats):
    """Statistics for sync operations.

    December 2025: Now extends SyncDaemonStats for consistent tracking.
    Inherits: syncs_completed, syncs_failed, bytes_synced, last_sync_duration,
              errors_count, last_error, consecutive_failures, is_healthy(), etc.
    """

    # AutoSync-specific fields (not in base class)
    games_synced: int = 0
    # Quality filtering stats (December 2025)
    databases_skipped_quality: int = 0
    databases_quality_checked: int = 0
    # Quality extraction stats (December 2025)
    games_quality_extracted: int = 0
    games_added_to_priority: int = 0
    # Verification stats (December 2025 - Gap 4 fix)
    databases_verified: int = 0
    databases_verification_failed: int = 0
    last_verification_time: float = 0.0

    # Backward compatibility aliases
    @property
    def total_syncs(self) -> int:
        """Alias for operations_attempted (backward compatibility)."""
        return self.operations_attempted

    @property
    def successful_syncs(self) -> int:
        """Alias for syncs_completed (backward compatibility)."""
        return self.syncs_completed

    @property
    def failed_syncs(self) -> int:
        """Alias for syncs_failed (backward compatibility)."""
        return self.syncs_failed

    @property
    def bytes_transferred(self) -> int:
        """Alias for bytes_synced (backward compatibility)."""
        return self.bytes_synced

    @property
    def last_sync_time(self) -> float:
        """Alias for last_check_time (backward compatibility)."""
        return self.last_check_time

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary with AutoSync-specific fields."""
        base = super().to_dict()
        base.update({
            # AutoSync-specific
            "games_synced": self.games_synced,
            "databases_skipped_quality": self.databases_skipped_quality,
            "databases_quality_checked": self.databases_quality_checked,
            "games_quality_extracted": self.games_quality_extracted,
            "games_added_to_priority": self.games_added_to_priority,
            "databases_verified": self.databases_verified,
            "databases_verification_failed": self.databases_verification_failed,
            "last_verification_time": self.last_verification_time,
            # Backward compat aliases
            "total_syncs": self.total_syncs,
            "successful_syncs": self.successful_syncs,
            "failed_syncs": self.failed_syncs,
            "bytes_transferred": self.bytes_transferred,
            "last_sync_time": self.last_sync_time,
        })
        return base


__all__ = [
    # Strategy enum
    "SyncStrategy",
    # Configuration
    "AutoSyncConfig",
    # Stats and progress
    "SyncStats",
    "SyncProgress",
    # Constants
    "MIN_MOVES_PER_GAME",
    "DEFAULT_MIN_MOVES",
]
