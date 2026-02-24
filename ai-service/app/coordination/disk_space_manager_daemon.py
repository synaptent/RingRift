"""Disk Space Manager Daemon.

Proactive disk space management for training nodes and coordinators.
Monitors disk usage and triggers cleanup before reaching critical thresholds.

Features:
- Proactive cleanup at 60% (before 70% warning threshold)
- Removes old logs, empty databases, old checkpoints
- Config-aware: prioritizes keeping data for active training configs
- Event-driven: emits DISK_CLEANUP_TRIGGERED, DISK_SPACE_LOW events

December 2025: Created for permanent disk management solution.

Usage:
    from app.coordination.disk_space_manager_daemon import (
        DiskSpaceManagerDaemon,
        DiskSpaceConfig,
        get_disk_space_daemon,
    )

    # Start the daemon
    daemon = get_disk_space_daemon()
    await daemon.start()

    # Check current disk status
    status = daemon.get_disk_status()
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.thresholds import DISK_CRITICAL_PERCENT, DISK_PRODUCTION_HALT_PERCENT
from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.event_utils import parse_config_key
from app.coordination.contracts import CoordinatorStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Protected Patterns - NEVER DELETE THESE
# =============================================================================

# CRITICAL: Files matching these patterns must NEVER be deleted by cleanup.
# This is a safety measure to prevent accidental data loss.
PROTECTED_PATTERNS = [
    "canonical_*.db",       # Canonical game databases - source of truth
    "unified_elo*.db",      # Elo ratings database
    "registry.db",          # Model registry
    "coordination.db",      # Coordination state
    "work_queue.db",        # Active work queue
    "*.npz",                # Training data files - never auto-delete
    "*_backup*.db",         # Backup files
]

# Directories that should never have files auto-deleted
PROTECTED_DIRECTORIES = [
    "coordination",         # State databases
    "model_registry",       # Model metadata
    "registry_backups",     # Registry backups
]


def _is_protected_file(file_path: Path) -> bool:
    """Check if a file is protected from deletion.

    CRITICAL: This function must be called before any file deletion.
    Protected files include canonical databases, training data, and
    coordination state.
    """
    import fnmatch

    filename = file_path.name

    # Check filename patterns
    for pattern in PROTECTED_PATTERNS:
        if fnmatch.fnmatch(filename, pattern):
            logger.debug(f"Protected file (pattern {pattern}): {filename}")
            return True

    # Check parent directories
    for parent in file_path.parents:
        if parent.name in PROTECTED_DIRECTORIES:
            logger.debug(f"Protected file (directory {parent.name}): {filename}")
            return True

    return False


# =============================================================================
# Disk Write Gate (Phase 2.3 - Dec 29, 2025)
# =============================================================================
#
# Global write gate to prevent writes when disk is critically full.
# Writers should call can_write_to_disk() before large write operations.
# The DiskSpaceManagerDaemon automatically blocks writes when disk usage > 70%
# and allows writes again when usage drops below target (50%).

import threading

_disk_write_lock = threading.Lock()
_disk_write_allowed = True
_disk_write_block_reason: str | None = None


def can_write_to_disk() -> bool:
    """Check if disk writes are currently allowed.

    Usage in writers (e.g., GameReplayDB, selfplay_runner):
        from app.coordination.disk_space_manager_daemon import can_write_to_disk

        if not can_write_to_disk():
            raise DiskSpaceError("Disk writes blocked due to low disk space")

    Returns:
        True if writes are allowed, False if writes are blocked.
    """
    with _disk_write_lock:
        return _disk_write_allowed


def get_disk_write_status() -> tuple[bool, str | None]:
    """Get disk write status and reason if blocked.

    Returns:
        Tuple of (is_allowed, block_reason). block_reason is None if allowed.
    """
    with _disk_write_lock:
        return _disk_write_allowed, _disk_write_block_reason


def block_disk_writes(reason: str = "disk_space_critical") -> None:
    """Block disk writes globally.

    Called by DiskSpaceManagerDaemon when disk usage exceeds critical threshold.
    Can also be called programmatically for maintenance operations.

    Args:
        reason: Human-readable reason for the block (for logging/debugging)
    """
    global _disk_write_allowed, _disk_write_block_reason
    with _disk_write_lock:
        if _disk_write_allowed:
            _disk_write_allowed = False
            _disk_write_block_reason = reason
            logger.warning(f"Disk writes BLOCKED: {reason}")


def allow_disk_writes() -> None:
    """Allow disk writes globally.

    Called by DiskSpaceManagerDaemon when disk usage drops below target threshold.
    """
    global _disk_write_allowed, _disk_write_block_reason
    with _disk_write_lock:
        if not _disk_write_allowed:
            _disk_write_allowed = True
            previous_reason = _disk_write_block_reason
            _disk_write_block_reason = None
            logger.info(f"Disk writes ALLOWED (was blocked: {previous_reason})")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DiskSpaceConfig:
    """Configuration for disk space management."""

    # Daemon control
    enabled: bool = True

    # P1.3 Dec 2025: Reduced from 30 minutes to 5 minutes
    # 30-minute interval allowed disk to fill silently during active selfplay
    check_interval_seconds: int = 300

    # Thresholds (percentages) - from app.config.thresholds (canonical source)
    proactive_cleanup_threshold: int = 75  # Start cleanup before production halt (85)
    warning_threshold: int = DISK_PRODUCTION_HALT_PERCENT - 5  # 80 - warn approaching halt
    critical_threshold: int = DISK_CRITICAL_PERCENT  # 90 - block writes
    emergency_threshold: int = 95  # Emergency mode

    # Target after cleanup - aligned with auto_sync_daemon target_disk_usage_percent
    target_disk_usage: int = 50  # Clean down to 50%

    # Minimum free space (GB)
    min_free_gb: int = 50

    # Data paths to manage (relative to ai-service root)
    logs_dir: str = "logs"
    games_dir: str = "data/games"
    training_dir: str = "data/training"
    checkpoints_dir: str = "data/checkpoints"
    model_registry_dir: str = "data/model_registry"

    # Retention policies (days)
    log_retention_days: int = 7
    checkpoint_retention_days: int = 14
    empty_db_max_age_days: int = 1

    # Keep at least N newest files per category
    min_checkpoints_per_config: int = 3
    min_logs_to_keep: int = 10

    # Sync-aware cleanup settings (December 2025)
    # Require N verified copies before deleting a file
    min_copies_before_delete: int = 2
    # Enable sync-aware cleanup (requires ClusterManifest)
    enable_sync_aware_cleanup: bool = True

    # Cleanup priorities (lower = delete first)
    cleanup_priorities: list[str] = field(
        default_factory=lambda: [
            "logs",  # 1. Old logs first
            "cache",  # 2. Pip/torch cache
            "empty_dbs",  # 3. Empty databases
            "synced_games",  # 4. Games with verified N+ copies (NEW - Dec 2025)
            "old_checkpoints",  # 5. Old checkpoints (keep N newest)
            "quarantine",  # 6. Quarantined databases
        ]
    )

    # Enable actual cleanup (set False for dry-run)
    enable_cleanup: bool = True

    # Enable event emission
    emit_events: bool = True

    @classmethod
    def from_env(cls, prefix: str = "RINGRIFT_DISK_SPACE") -> "DiskSpaceConfig":
        """Load configuration from environment variables."""
        # Start with defaults
        config = cls()

        # Base config env vars
        if os.environ.get(f"{prefix}_ENABLED"):
            config.enabled = os.environ[f"{prefix}_ENABLED"].lower() == "true"
        if os.environ.get(f"{prefix}_CHECK_INTERVAL"):
            config.check_interval_seconds = int(os.environ[f"{prefix}_CHECK_INTERVAL"])

        # Disk-specific env vars
        env_vars = {
            "PROACTIVE_THRESHOLD": ("proactive_cleanup_threshold", int),
            "WARNING_THRESHOLD": ("warning_threshold", int),
            "CRITICAL_THRESHOLD": ("critical_threshold", int),
            "TARGET_USAGE": ("target_disk_usage", int),
            "MIN_FREE_GB": ("min_free_gb", int),
            "LOG_RETENTION_DAYS": ("log_retention_days", int),
            "ENABLE_CLEANUP": ("enable_cleanup", lambda x: x == "1"),
            "EMIT_EVENTS": ("emit_events", lambda x: x == "1"),
        }

        for env_suffix, (attr, converter) in env_vars.items():
            env_key = f"{prefix}_{env_suffix}"
            if os.environ.get(env_key):
                try:
                    setattr(config, attr, converter(os.environ[env_key]))
                except (ValueError, TypeError):
                    pass

        return config


# =============================================================================
# Disk Status
# =============================================================================


@dataclass
class DiskStatus:
    """Disk usage status."""

    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    usage_percent: float
    needs_cleanup: bool
    is_warning: bool
    is_critical: bool
    is_emergency: bool

    @classmethod
    def from_path(cls, path: str, config: DiskSpaceConfig) -> "DiskStatus":
        """Get disk status for a path."""
        try:
            usage = shutil.disk_usage(path)
            total_gb = usage.total / (1024**3)
            used_gb = usage.used / (1024**3)
            free_gb = usage.free / (1024**3)
            usage_percent = (usage.used / usage.total) * 100

            return cls(
                path=path,
                total_gb=round(total_gb, 2),
                used_gb=round(used_gb, 2),
                free_gb=round(free_gb, 2),
                usage_percent=round(usage_percent, 1),
                needs_cleanup=usage_percent >= config.proactive_cleanup_threshold,
                is_warning=usage_percent >= config.warning_threshold,
                is_critical=usage_percent >= config.critical_threshold,
                is_emergency=usage_percent >= config.emergency_threshold,
            )
        except OSError as e:
            logger.error(f"Failed to get disk status for {path}: {e}")
            return cls(
                path=path,
                total_gb=0,
                used_gb=0,
                free_gb=0,
                usage_percent=100,  # Assume full on error
                needs_cleanup=True,
                is_warning=True,
                is_critical=True,
                is_emergency=False,
            )


# =============================================================================
# Disk Space Manager Daemon
# =============================================================================


class DiskSpaceManagerDaemon(HandlerBase):
    """Proactive disk space management daemon.

    Monitors disk usage and automatically cleans up when thresholds are exceeded.
    Emits events for coordination with other daemons.
    """

    def __init__(self, config: DiskSpaceConfig | None = None):
        self._daemon_config = config or DiskSpaceConfig.from_env()
        super().__init__(
            name="DiskSpaceManagerDaemon",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )
        self._root_path = self._find_ai_service_root()
        self._last_cleanup_time: float = 0.0
        self._bytes_cleaned: int = 0
        self._cleanups_performed: int = 0
        self._current_status: DiskStatus | None = None
        self._manifest: "ClusterManifest | None" = None
        self._manifest_initialized: bool = False
        self._sync_cleanup_stats: dict[str, int] = {
            "files_checked": 0,
            "files_deleted": 0,
            "files_skipped_no_sync": 0,
            "files_skipped_protected": 0,
        }

    @property
    def config(self) -> DiskSpaceConfig:
        """Return daemon configuration."""
        return self._daemon_config

    def _get_manifest(self) -> "ClusterManifest | None":
        """Lazy-load ClusterManifest for sync-aware cleanup."""
        if self._manifest_initialized:
            return self._manifest

        try:
            from app.distributed.cluster_manifest import ClusterManifest

            self._manifest = ClusterManifest.get_instance()
            self._manifest_initialized = True
            logger.info(f"[{self.name}] ClusterManifest initialized for sync-aware cleanup")
        except ImportError:
            logger.warning(
                f"[{self.name}] ClusterManifest not available - "
                "sync-aware cleanup disabled"
            )
            self._manifest = None
            self._manifest_initialized = True
        except Exception as e:
            logger.warning(
                f"[{self.name}] Failed to initialize ClusterManifest: {e} - "
                "sync-aware cleanup disabled"
            )
            self._manifest = None
            self._manifest_initialized = True

        return self._manifest

    def _find_ai_service_root(self) -> Path:
        """Find ai-service root directory."""
        # Try from current file
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "app" / "coordination").exists():
                return parent

        # Fallback to environment
        env_root = os.environ.get("AI_SERVICE_ROOT")
        if env_root:
            return Path(env_root)

        # Last resort: current working directory
        return Path.cwd()

    async def _run_cycle(self) -> None:
        """Main daemon cycle - check disk and cleanup if needed."""
        self._current_status = DiskStatus.from_path(str(self._root_path), self.config)

        logger.info(
            f"[{self.name}] Disk status: "
            f"{self._current_status.usage_percent:.1f}% used "
            f"({self._current_status.free_gb:.1f}GB free)"
        )

        # Emit events if thresholds exceeded
        if self.config.emit_events:
            await self._emit_status_events(self._current_status)

        # Cleanup if needed
        if self._current_status.needs_cleanup and self.config.enable_cleanup:
            await self._perform_cleanup(self._current_status)

        # P1.2 Dec 2025: Periodic WAL checkpoint for SQLite databases
        # WAL files accumulate until checkpoint - run every cycle (5 min)
        self._checkpoint_databases()

    async def _emit_status_events(self, status: DiskStatus) -> None:
        """Emit disk status events."""
        try:
            from app.distributed.data_events import (
                emit_disk_space_low,
            )

            if status.is_critical or status.is_emergency:
                await emit_disk_space_low(
                    host=self.node_id,
                    usage_percent=status.usage_percent,
                    free_gb=status.free_gb,
                    threshold=self.config.critical_threshold,
                    source=self.name,
                )
        except ImportError:
            logger.debug("data_events module not available for event emission")
        except Exception as e:
            logger.debug(f"Failed to emit disk status event: {e}")

    async def _perform_cleanup(self, status: DiskStatus) -> None:
        """Perform cleanup operations based on priority."""
        logger.info(f"[{self.name}] Starting cleanup (target: {self.config.target_disk_usage}%)")

        # Phase 2.3 Dec 29, 2025: Block writes if disk usage exceeds critical threshold
        # This prevents new writes from competing with cleanup operations
        if status.is_critical or status.is_emergency:
            block_disk_writes(
                reason=f"disk_usage_{status.usage_percent:.1f}%_exceeds_critical_{self.config.critical_threshold}%"
            )

        bytes_freed = 0
        for priority in self.config.cleanup_priorities:
            if status.usage_percent <= self.config.target_disk_usage:
                break  # Target reached

            # December 30, 2025: Wrap sync cleanup methods with asyncio.to_thread
            # to avoid blocking the event loop during file I/O operations
            if priority == "logs":
                bytes_freed += await asyncio.to_thread(self._cleanup_old_logs)
            elif priority == "cache":
                bytes_freed += await asyncio.to_thread(self._cleanup_cache)
            elif priority == "empty_dbs":
                bytes_freed += await asyncio.to_thread(self._cleanup_empty_databases)
            elif priority == "synced_games":
                # NEW: Sync-aware cleanup (December 2025)
                if self.config.enable_sync_aware_cleanup:
                    bytes_freed += await asyncio.to_thread(self._cleanup_synced_databases)
            elif priority == "old_checkpoints":
                bytes_freed += await asyncio.to_thread(self._cleanup_old_checkpoints)
            elif priority == "quarantine":
                bytes_freed += await asyncio.to_thread(self._cleanup_quarantine)

            # Refresh status
            status = DiskStatus.from_path(str(self._root_path), self.config)

        self._bytes_cleaned += bytes_freed
        self._cleanups_performed += 1
        self._last_cleanup_time = time.time()

        freed_mb = bytes_freed / (1024 * 1024)
        logger.info(
            f"[{self.name}] Cleanup complete: "
            f"freed {freed_mb:.1f}MB, now at {status.usage_percent:.1f}%"
        )

        # Phase 2.3 Dec 29, 2025: Re-allow writes if cleanup succeeded
        # Only allow if we're now below the critical threshold
        if status.usage_percent < self.config.critical_threshold:
            allow_disk_writes()

        # Emit cleanup event
        if self.config.emit_events and bytes_freed > 0:
            await self._emit_cleanup_event(bytes_freed)

    async def _emit_cleanup_event(self, bytes_freed: int) -> None:
        """Emit cleanup completed event."""
        try:
            from app.coordination.event_router import DataEventType, get_router

            router = get_router()
            await router.publish(
                DataEventType.DISK_CLEANUP_TRIGGERED,
                {
                    "host": self.node_id,
                    "bytes_freed": bytes_freed,
                    "cleanups_performed": self._cleanups_performed,
                    "source": self.name,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit cleanup event: {e}")

    def _checkpoint_databases(self) -> None:
        """Checkpoint WAL for all SQLite databases.

        P1.2 Dec 2025: SQLite WAL files grow unbounded until checkpoint.
        This runs TRUNCATE checkpoint to reclaim space in WAL files.

        Dec 29, 2025 Enhancement: Now checkpoints ALL database directories:
        - games/ (selfplay databases)
        - coordination/ (pipeline state, delivery ledger, tasks, etc.)
        - Root data/ (unified_elo.db, cluster_manifest.db, work_queue.db, etc.)
        """
        try:
            from app.coordination.wal_sync_utils import checkpoint_database

            checkpointed = 0
            errors = 0

            # Directories to checkpoint
            checkpoint_dirs = [
                self._root_path / self.config.games_dir,  # games/
                self._root_path / "coordination",  # coordination databases
                self._root_path,  # Root level databases (unified_elo.db, etc.)
            ]

            for dir_path in checkpoint_dirs:
                if not dir_path.exists():
                    continue

                # For root path, don't recurse (avoid double-processing subdirs)
                pattern = "*.db" if dir_path == self._root_path else "**/*.db"

                for db_file in dir_path.glob(pattern):
                    if db_file.name.startswith("."):
                        continue  # Skip hidden files

                    # Check if WAL file exists (only checkpoint if needed)
                    wal_file = db_file.with_suffix(db_file.suffix + "-wal")
                    if not wal_file.exists():
                        continue  # No WAL file, skip

                    # Check WAL file size - only checkpoint if > 1MB
                    try:
                        wal_size = wal_file.stat().st_size
                        if wal_size < 1024 * 1024:  # < 1MB
                            continue  # Small WAL, skip
                    except OSError:
                        continue

                    try:
                        if checkpoint_database(db_file, truncate=True):
                            checkpointed += 1
                            logger.debug(
                                f"[{self.name}] Checkpointed {db_file.name} "
                                f"(WAL was {wal_size / 1024:.0f}KB)"
                            )
                    except Exception as e:
                        errors += 1
                        logger.debug(f"Checkpoint failed for {db_file.name}: {e}")

            if checkpointed > 0:
                logger.info(
                    f"[{self.name}] WAL checkpoint complete: "
                    f"{checkpointed} databases checkpointed"
                    + (f", {errors} errors" if errors > 0 else "")
                )

        except ImportError:
            logger.debug("wal_sync_utils not available for WAL checkpoint")
        except Exception as e:
            logger.debug(f"WAL checkpoint cycle error: {e}")

    def _cleanup_old_logs(self) -> int:
        """Remove old log files."""
        bytes_freed = 0
        logs_path = self._root_path / self.config.logs_dir
        if not logs_path.exists():
            return 0

        cutoff = time.time() - (self.config.log_retention_days * 86400)
        log_files = sorted(logs_path.glob("**/*.log"), key=lambda f: f.stat().st_mtime)

        # Keep at least min_logs_to_keep newest files
        files_to_check = log_files[:-self.config.min_logs_to_keep] if len(log_files) > self.config.min_logs_to_keep else []

        for log_file in files_to_check:
            try:
                if log_file.stat().st_mtime < cutoff:
                    size = log_file.stat().st_size
                    log_file.unlink()
                    bytes_freed += size
                    logger.debug(f"Removed old log: {log_file.name}")
            except (OSError, PermissionError) as e:
                logger.debug(f"Failed to remove log {log_file}: {e}")

        return bytes_freed

    def _cleanup_cache(self) -> int:
        """Remove pip/torch cache."""
        bytes_freed = 0
        cache_dirs = [
            Path.home() / ".cache" / "pip",
            Path.home() / ".cache" / "torch",
            Path.home() / ".cache" / "huggingface",
        ]

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                try:
                    # Get size before removal
                    size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    bytes_freed += size
                    logger.debug(f"Cleared cache: {cache_dir}")
                except (OSError, PermissionError) as e:
                    logger.debug(f"Failed to clear cache {cache_dir}: {e}")

        return bytes_freed

    def _cleanup_empty_databases(self) -> int:
        """Remove empty or near-empty database files.

        SAFETY: Never deletes:
        - canonical_*.db files (production game data)
        - Databases in cluster_* directories (synced from cluster nodes)
        - .bak backup files
        """
        bytes_freed = 0
        games_path = self._root_path / self.config.games_dir
        if not games_path.exists():
            return 0

        cutoff = time.time() - (self.config.empty_db_max_age_days * 86400)

        for db_file in games_path.glob("**/*.db"):
            # CRITICAL: Check centralized protection first
            if _is_protected_file(db_file):
                continue

            # NEVER delete canonical databases - they are protected
            if db_file.name.startswith("canonical_"):
                logger.debug(f"Skipping protected canonical database: {db_file.name}")
                continue

            # Skip databases in cluster_* directories (source data from cluster nodes)
            if any(parent.name.startswith("cluster_") for parent in db_file.parents):
                logger.debug(f"Skipping cluster source database: {db_file}")
                continue

            # Skip backup files
            if ".bak" in db_file.name:
                logger.debug(f"Skipping backup file: {db_file.name}")
                continue

            try:
                stat = db_file.stat()
                # Remove if small (<100KB) and old
                if stat.st_size < 100 * 1024 and stat.st_mtime < cutoff:
                    # Verify it's actually empty
                    result = subprocess.run(
                        ["sqlite3", str(db_file), "SELECT COUNT(*) FROM games;"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        count = int(result.stdout.strip() or "0")
                        if count == 0:
                            size = stat.st_size
                            db_file.unlink()
                            bytes_freed += size
                            logger.debug(f"Removed empty database: {db_file.name}")
            except (OSError, subprocess.TimeoutExpired, ValueError) as e:
                logger.debug(f"Failed to check/remove database {db_file}: {e}")

        return bytes_freed

    def _cleanup_old_checkpoints(self) -> int:
        """Remove old model checkpoints, keeping N newest per config."""
        bytes_freed = 0
        checkpoints_path = self._root_path / self.config.checkpoints_dir
        if not checkpoints_path.exists():
            return 0

        # Group checkpoints by config pattern
        checkpoint_groups: dict[str, list[Path]] = {}
        for pth_file in checkpoints_path.glob("**/*.pth"):
            # Extract config from filename (e.g., "hex8_2p_v5_epoch10.pth" -> "hex8_2p")
            name = pth_file.stem
            parsed = parse_config_key(name)
            if parsed:
                config_key = f"{parsed.board_type}_{parsed.num_players}p"
            else:
                # Fallback: try first two underscore-separated parts
                parts = name.split("_")
                config_key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]

            if config_key not in checkpoint_groups:
                checkpoint_groups[config_key] = []
            checkpoint_groups[config_key].append(pth_file)

        # Keep newest N per config
        for config_key, files in checkpoint_groups.items():
            sorted_files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)
            to_remove = sorted_files[self.config.min_checkpoints_per_config:]

            for old_file in to_remove:
                try:
                    size = old_file.stat().st_size
                    old_file.unlink()
                    bytes_freed += size
                    logger.debug(f"Removed old checkpoint: {old_file.name}")
                except (OSError, PermissionError) as e:
                    logger.debug(f"Failed to remove checkpoint {old_file}: {e}")

        return bytes_freed

    def _cleanup_quarantine(self) -> int:
        """Remove quarantined databases."""
        bytes_freed = 0
        quarantine_path = self._root_path / "data" / "quarantine"
        if not quarantine_path.exists():
            return 0

        for file in quarantine_path.iterdir():
            try:
                size = file.stat().st_size
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
                bytes_freed += size
                logger.debug(f"Removed quarantined: {file.name}")
            except (OSError, PermissionError) as e:
                logger.debug(f"Failed to remove quarantined {file}: {e}")

        return bytes_freed

    def _cleanup_synced_databases(self) -> int:
        """Delete databases that have been verified synced to N+ locations.

        SAFE VERSION (December 2025): Only deletes if ClusterManifest confirms
        the file exists on at least min_copies_before_delete other nodes.

        This replaces the disabled _cleanup_synced_games method that previously
        deleted databases without verifying sync actually succeeded.

        Returns:
            Number of bytes freed.
        """
        manifest = self._get_manifest()
        if manifest is None:
            logger.debug(
                f"[{self.name}] Skipping sync-aware cleanup - "
                "ClusterManifest not available"
            )
            return 0

        bytes_freed = 0
        min_copies = self.config.min_copies_before_delete
        games_path = self._root_path / self.config.games_dir

        if not games_path.exists():
            return 0

        logger.info(
            f"[{self.name}] Starting sync-aware cleanup "
            f"(require {min_copies}+ verified copies before delete)"
        )

        for db_file in games_path.glob("**/*.db"):
            self._sync_cleanup_stats["files_checked"] += 1

            # CRITICAL: Check centralized protection first
            if _is_protected_file(db_file):
                self._sync_cleanup_stats["files_skipped_protected"] += 1
                continue

            # Get relative path for manifest lookup
            try:
                relative_path = db_file.relative_to(self._root_path)
            except ValueError:
                relative_path = db_file

            # Check if file is safe to delete (has N+ verified copies)
            try:
                if manifest.is_safe_to_delete(str(relative_path), min_copies=min_copies):
                    # Verify file size before deletion
                    try:
                        size = db_file.stat().st_size
                        db_file.unlink()
                        bytes_freed += size
                        self._sync_cleanup_stats["files_deleted"] += 1
                        logger.info(
                            f"[{self.name}] Safe delete: {db_file.name} "
                            f"(verified {min_copies}+ copies, freed {size / 1024 / 1024:.1f}MB)"
                        )
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Failed to delete synced database {db_file}: {e}")
                else:
                    # Not enough verified copies - skip
                    self._sync_cleanup_stats["files_skipped_no_sync"] += 1
                    replication_count = manifest.get_verified_replication_count(str(relative_path))
                    logger.debug(
                        f"[{self.name}] Skipping {db_file.name}: "
                        f"only {replication_count} verified copies (need {min_copies})"
                    )
            except Exception as e:
                logger.debug(f"Error checking sync status for {db_file}: {e}")
                self._sync_cleanup_stats["files_skipped_no_sync"] += 1

        # Log cleanup summary
        stats = self._sync_cleanup_stats
        logger.info(
            f"[{self.name}] Sync-aware cleanup complete: "
            f"checked={stats['files_checked']}, deleted={stats['files_deleted']}, "
            f"skipped_no_sync={stats['files_skipped_no_sync']}, "
            f"skipped_protected={stats['files_skipped_protected']}"
        )

        return bytes_freed

    def get_disk_status(self) -> DiskStatus | None:
        """Get current disk status."""
        if self._current_status is None:
            self._current_status = DiskStatus.from_path(str(self._root_path), self.config)
        return self._current_status

    def health_check(self) -> HealthCheckResult:
        """Return health check result for daemon protocol."""
        status = self.get_disk_status()
        details = {
            "running": self._running,
            "disk_usage_percent": status.usage_percent if status else None,
            "free_gb": status.free_gb if status else None,
            "bytes_cleaned_total": self._bytes_cleaned,
            "cleanups_performed": self._cleanups_performed,
            "last_cleanup_time": self._last_cleanup_time,
            "uptime_seconds": self.uptime_seconds,
            "cycles_completed": self._stats.cycles_completed,
            "errors_count": self._stats.errors_count,
            "sync_cleanup_stats": self._sync_cleanup_stats,
            "sync_aware_cleanup_enabled": self.config.enable_sync_aware_cleanup,
            "min_copies_before_delete": self.config.min_copies_before_delete,
        }

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message=f"{self.name} is not running",
                details=details,
            )

        # Unhealthy if disk is critical
        if status and status.is_critical:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Disk usage critical: {status.usage_percent:.1f}%",
                details=details,
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"{self.name} healthy, disk at {status.usage_percent:.1f}%" if status else "Running",
            details=details,
        )

    def get_status(self) -> dict[str, Any]:
        """Get daemon status for monitoring."""
        health = self.health_check()
        status = self.get_disk_status()
        return {
            "name": self.name,
            "running": self._running,
            "uptime_seconds": self.uptime_seconds,
            "config": {
                "proactive_threshold": self.config.proactive_cleanup_threshold,
                "warning_threshold": self.config.warning_threshold,
                "target_usage": self.config.target_disk_usage,
                "enable_cleanup": self.config.enable_cleanup,
            },
            "disk": {
                "usage_percent": status.usage_percent if status else None,
                "free_gb": status.free_gb if status else None,
                "needs_cleanup": status.needs_cleanup if status else None,
            },
            "health": {
                "healthy": health.healthy,
                "status": health.status.value if hasattr(health.status, "value") else str(health.status),
                "message": health.message,
            },
            **health.details,
        }


# =============================================================================
# Singleton Access
# =============================================================================


def get_disk_space_daemon() -> DiskSpaceManagerDaemon:
    """Get the singleton DiskSpaceManagerDaemon instance."""
    return DiskSpaceManagerDaemon.get_instance()


def reset_disk_space_daemon() -> None:
    """Reset the singleton (for testing)."""
    DiskSpaceManagerDaemon.reset_instance()


# =============================================================================
# Coordinator-Specific Configuration (December 27, 2025)
# =============================================================================


@dataclass
class CoordinatorDiskConfig(DiskSpaceConfig):
    """More aggressive disk management for coordinator-only nodes.

    Coordinator nodes don't run training/selfplay, so they can:
    - Sync data to remote storage before cleanup
    - Be more aggressive about removing local copies
    - Have lower thresholds for triggering cleanup
    """

    # More aggressive thresholds for coordinators
    proactive_cleanup_threshold: int = 50  # Start earlier
    target_disk_usage: int = 40  # Clean down further
    min_free_gb: int = 100  # Keep more free space

    # Shorter retention for coordinators
    log_retention_days: int = 3
    checkpoint_retention_days: int = 7
    empty_db_max_age_days: int = 0  # Clean immediately

    # Remote sync configuration
    remote_sync_enabled: bool = True
    remote_host: str = "mac-studio"
    remote_base_path: str = "/Volumes/RingRift-Data"
    sync_before_cleanup: bool = True

    # Coordinator-specific cleanup priorities
    cleanup_priorities: list[str] = field(
        default_factory=lambda: [
            "logs",  # 1. Old logs first
            "cache",  # 2. Pip/torch cache
            "empty_dbs",  # 3. Empty databases
            "synced_training",  # 4. Training data that's been synced
            "synced_games",  # 5. Game databases that are synced
            "old_checkpoints",  # 6. Old checkpoints
            "quarantine",  # 7. Quarantined databases
        ]
    )

    @classmethod
    def for_coordinator(cls) -> "CoordinatorDiskConfig":
        """Create config for coordinator nodes."""
        config = cls()
        # Override from environment
        if os.environ.get("RINGRIFT_COORDINATOR_REMOTE_HOST"):
            config.remote_host = os.environ["RINGRIFT_COORDINATOR_REMOTE_HOST"]
        if os.environ.get("RINGRIFT_COORDINATOR_REMOTE_PATH"):
            config.remote_base_path = os.environ["RINGRIFT_COORDINATOR_REMOTE_PATH"]
        return config


class CoordinatorDiskManager(DiskSpaceManagerDaemon):
    """Disk manager for coordinator-only nodes with remote sync.

    December 27, 2025: Created for permanent coordinator disk management.

    Features beyond base DiskSpaceManager:
    - Syncs valuable data to remote storage (OWC) before cleanup
    - Removes local copies of synced data
    - More aggressive cleanup thresholds
    """

    def __init__(self, config: CoordinatorDiskConfig | None = None):
        self._daemon_config = config or CoordinatorDiskConfig.for_coordinator()
        # Directly call HandlerBase.__init__ to set correct name
        HandlerBase.__init__(
            self,
            name="CoordinatorDiskManager",
            config=self._daemon_config,
            cycle_interval=float(self._daemon_config.check_interval_seconds),
        )
        # Initialize base class state (without calling parent __init__ again)
        self._root_path = self._find_ai_service_root()
        self._last_cleanup_time = 0.0
        self._bytes_cleaned = 0
        self._cleanups_performed = 0
        self._current_status = None
        self._manifest = None
        self._manifest_initialized = False
        self._sync_cleanup_stats = {
            "files_checked": 0,
            "files_deleted": 0,
            "files_skipped_no_sync": 0,
            "files_skipped_protected": 0,
        }
        self._sync_stats: dict[str, int] = {
            "files_synced": 0,
            "bytes_synced": 0,
            "sync_errors": 0,
        }

    @property
    def config(self) -> CoordinatorDiskConfig:
        """Return coordinator daemon configuration."""
        return self._daemon_config  # type: ignore

    async def _perform_cleanup(self, status: DiskStatus) -> None:
        """Perform cleanup with remote sync first."""
        config = self.config
        if not isinstance(config, CoordinatorDiskConfig):
            return await super()._perform_cleanup(status)

        # Step 1: Sync valuable data to remote before cleanup
        if config.sync_before_cleanup and config.remote_sync_enabled:
            logger.info(f"[{self.name}] Syncing data to {config.remote_host} before cleanup")
            await self._sync_to_remote()

        # Step 2: Perform regular cleanup
        await super()._perform_cleanup(status)

        # Step 3: Additional coordinator-specific cleanup
        for priority in config.cleanup_priorities:
            if status.usage_percent <= config.target_disk_usage:
                break

            if priority == "synced_training":
                self._cleanup_synced_training()
            elif priority == "synced_games":
                self._cleanup_synced_games()

            status = DiskStatus.from_path(str(self._root_path), config)

    async def _sync_to_remote(self) -> None:
        """Sync valuable data to remote storage.

        December 2025: Uses asyncio.create_subprocess_exec to avoid blocking the event loop.
        """
        config = self.config
        if not isinstance(config, CoordinatorDiskConfig):
            return

        remote_host = config.remote_host
        remote_base = config.remote_base_path

        sync_dirs = [
            ("data/games", f"{remote_base}/cluster_games/coordinator_backup"),
            ("data/training", f"{remote_base}/training_data/coordinator_backup"),
            ("models", f"{remote_base}/trained_models/coordinator_backup"),
            ("data/model_registry", f"{remote_base}/model_registry"),
        ]

        for local_dir, remote_path in sync_dirs:
            local_path = self._root_path / local_dir
            if not local_path.exists():
                continue

            try:
                # Use rsync with progress for monitoring (non-blocking)
                proc = await asyncio.create_subprocess_exec(
                    "rsync",
                    "-avz",
                    "--progress",
                    "--partial",
                    f"{local_path}/",
                    f"{remote_host}:{remote_path}/",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=3600.0  # 1 hour timeout
                    )
                    if proc.returncode == 0:
                        logger.info(f"Synced {local_dir} to {remote_host}:{remote_path}")
                        self._sync_stats["files_synced"] += 1
                    else:
                        stderr_text = stderr.decode()[:200] if stderr else "unknown error"
                        logger.warning(f"Sync failed for {local_dir}: {stderr_text}")
                        self._sync_stats["sync_errors"] += 1
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    logger.warning(f"Sync timed out for {local_dir}")
                    self._sync_stats["sync_errors"] += 1
            except Exception as e:
                logger.warning(f"Sync error for {local_dir}: {e}")
                self._sync_stats["sync_errors"] += 1

    def _cleanup_synced_training(self) -> int:
        """Remove training NPZ files that have been synced.

        DISABLED: NPZ files are never auto-deleted for safety.
        Training data is critical and should only be deleted manually.

        December 27, 2025: Disabled after data loss incident.
        """
        # CRITICAL: NPZ files are in PROTECTED_PATTERNS - never auto-delete
        logger.debug("_cleanup_synced_training: Disabled for safety - NPZ files are protected")
        return 0

    def _cleanup_synced_games(self) -> int:
        """Remove game databases that have been synced.

        DISABLED: Database deletion is now disabled for safety.
        Only empty databases (<100KB with 0 games) are cleaned via _cleanup_empty_databases().

        SAFETY: This method previously deleted any DB older than 24 hours that wasn't
        canonical_*.db. This was dangerous because:
        1. It didn't verify sync actually succeeded
        2. It deleted valuable selfplay databases
        3. No confirmation or manifest tracking

        December 27, 2025: Disabled after data loss incident.
        """
        # CRITICAL: Disabled to prevent accidental data loss
        logger.debug("_cleanup_synced_games: Disabled for safety - manual deletion only")
        return 0

    def _cleanup_synced_games_DISABLED(self) -> int:
        """DEPRECATED: Original implementation kept for reference.

        DO NOT ENABLE - this deleted databases without verifying sync succeeded.
        """
        bytes_freed = 0
        games_path = self._root_path / "data" / "games"
        if not games_path.exists():
            return 0

        cutoff = time.time() - 86400  # 24 hours

        for db_file in games_path.glob("*.db"):
            # CRITICAL: Check centralized protection first
            if _is_protected_file(db_file):
                continue

            # Never delete canonical databases
            if db_file.name.startswith("canonical_"):
                continue

            # Never delete backup files
            if ".bak" in db_file.name:
                continue

            try:
                if db_file.stat().st_mtime < cutoff:
                    size = db_file.stat().st_size
                    db_file.unlink()
                    bytes_freed += size
                    logger.debug(f"Removed synced game database: {db_file.name}")
            except (OSError, PermissionError) as e:
                logger.debug(f"Failed to remove {db_file}: {e}")

        return bytes_freed

    def health_check(self) -> HealthCheckResult:
        """Return health check result with coordinator-specific sync stats.

        December 27, 2025: Override to include sync statistics in health check.
        """
        result = super().health_check()
        # Add sync stats to details
        if result.details:
            result.details["sync_stats"] = self._sync_stats
            result.details["remote_host"] = (
                self.config.remote_host
                if isinstance(self.config, CoordinatorDiskConfig)
                else None
            )
            result.details["remote_sync_enabled"] = (
                self.config.remote_sync_enabled
                if isinstance(self.config, CoordinatorDiskConfig)
                else False
            )
        # Check if sync errors are too high
        if self._sync_stats.get("sync_errors", 0) > 5:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"High sync error count: {self._sync_stats['sync_errors']}",
                details=result.details,
            )
        return result

    def get_status(self) -> dict[str, Any]:
        """Get daemon status including sync statistics."""
        status = super().get_status()
        status["sync_stats"] = self._sync_stats
        return status


# Coordinator daemon singleton


def get_coordinator_disk_daemon() -> CoordinatorDiskManager:
    """Get the singleton CoordinatorDiskManager instance."""
    return CoordinatorDiskManager.get_instance()


def reset_coordinator_disk_daemon() -> None:
    """Reset the coordinator singleton (for testing)."""
    CoordinatorDiskManager.reset_instance()
