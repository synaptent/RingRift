"""Ramdrive (tmpfs) utility for high-speed I/O operations.

This module provides utilities for using RAM-backed storage (/dev/shm)
to accelerate I/O-heavy operations like selfplay data generation and
model training. RAM storage is particularly useful on cloud instances
with limited or slow disk but ample RAM.

Key features:
- Auto-detection of ramdrive availability and capacity
- Automatic fallback to disk when ramdrive unavailable
- Periodic sync to persistent storage
- Graceful handling of space constraints

Usage:
    from app.utils.ramdrive import get_data_directory, RamdriveConfig

    # Simple usage with auto-detection
    data_dir = get_data_directory(prefer_ramdrive=True)

    # With explicit configuration
    config = RamdriveConfig(
        prefer_ramdrive=True,
        min_free_gb=2.0,
        sync_interval=300,
        sync_target="/path/to/persistent/storage"
    )
    data_dir = get_data_directory(config=config)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.utils.paths import AI_SERVICE_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    # Constants
    "RAMDRIVE_PATHS",
    "RINGRIFT_SUBDIR",
    # Configuration
    "RamdriveConfig",
    "RamdriveStatus",
    # Syncer class
    "RamdriveSyncer",
    "SystemResources",
    # Argument parsing helpers
    "add_ramdrive_args",
    # Core functions
    "detect_ramdrive",
    "get_auto_storage_path",
    "get_checkpoints_directory",
    "get_config_from_args",
    "get_data_directory",
    "get_games_directory",
    "get_logs_directory",
    "get_models_directory",
    "get_ramdrive_path",
    "get_system_resources",
    "is_ramdrive_available",
    "log_storage_recommendation",
    "should_use_ramdrive",
]

# Standard ramdrive locations
RAMDRIVE_PATHS = [
    "/dev/shm",      # Linux default
    "/run/shm",      # Some Linux distros
    "/tmp",          # macOS (often tmpfs)
]

# Default subdirectory for RingRift data
RINGRIFT_SUBDIR = "ringrift"


@dataclass
class RamdriveConfig:
    """Configuration for ramdrive usage."""

    # Whether to prefer ramdrive over disk storage
    prefer_ramdrive: bool = True

    # Minimum free space required (GB) to use ramdrive
    min_free_gb: float = 1.0

    # Maximum space to use on ramdrive (GB), 0 = no limit
    max_use_gb: float = 0.0

    # Interval for syncing to persistent storage (seconds), 0 = no sync
    sync_interval: int = 0

    # Target path for periodic sync (if empty, no sync)
    sync_target: str = ""

    # Subdirectory within ramdrive for this application
    subdirectory: str = "data"

    # Fallback path if ramdrive unavailable
    fallback_path: str = ""

    # Callback when sync completes
    on_sync_complete: Callable[[Path, Path, bool], None] | None = None


@dataclass
class RamdriveStatus:
    """Status information about ramdrive."""
    available: bool = False
    path: Path | None = None
    total_gb: float = 0.0
    free_gb: float = 0.0
    used_gb: float = 0.0
    is_tmpfs: bool = False


def detect_ramdrive() -> RamdriveStatus:
    """Detect available ramdrive and its status.

    Returns:
        RamdriveStatus with availability and capacity info.
    """
    for path_str in RAMDRIVE_PATHS:
        path = Path(path_str)
        if not path.exists():
            continue

        try:
            # Check if it's actually a tmpfs/ramfs
            is_tmpfs = False
            if os.path.exists("/proc/mounts"):
                with open("/proc/mounts") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 3 and parts[1] == path_str:
                            is_tmpfs = parts[2] in ("tmpfs", "ramfs")
                            break

            # Get disk usage
            usage = shutil.disk_usage(path)
            total_gb = usage.total / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)

            # Consider it available if writable and has some space
            if os.access(path, os.W_OK) and free_gb > 0.1:
                return RamdriveStatus(
                    available=True,
                    path=path,
                    total_gb=total_gb,
                    free_gb=free_gb,
                    used_gb=used_gb,
                    is_tmpfs=is_tmpfs,
                )
        except (OSError, PermissionError) as e:
            logger.debug(f"Cannot access {path}: {e}")
            continue

    return RamdriveStatus(available=False)


def get_data_directory(
    prefer_ramdrive: bool = False,
    config: RamdriveConfig | None = None,
    base_name: str = "data",
) -> Path:
    """Get the appropriate data directory based on configuration.

    Args:
        prefer_ramdrive: Whether to prefer ramdrive over disk.
        config: Optional detailed configuration.
        base_name: Base name for the data directory (e.g., "data", "games").

    Returns:
        Path to the data directory (created if necessary).
    """
    if config is None:
        config = RamdriveConfig(prefer_ramdrive=prefer_ramdrive)

    # Check ramdrive availability
    if config.prefer_ramdrive:
        status = detect_ramdrive()

        if status.available and status.free_gb >= config.min_free_gb:
            ramdrive_dir = status.path / RINGRIFT_SUBDIR / config.subdirectory / base_name
            try:
                ramdrive_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"Using ramdrive at {ramdrive_dir} "
                    f"(free: {status.free_gb:.1f}GB, total: {status.total_gb:.1f}GB)"
                )
                return ramdrive_dir
            except (OSError, PermissionError) as e:
                logger.warning(f"Cannot create ramdrive directory: {e}")
        elif status.available:
            logger.warning(
                f"Ramdrive available but insufficient space: "
                f"{status.free_gb:.1f}GB < {config.min_free_gb:.1f}GB required"
            )
        else:
            logger.info("Ramdrive not available, using disk storage")

    # Fallback to disk storage
    if config.fallback_path:
        fallback = Path(config.fallback_path) / base_name
    else:
        # Use ai-service/data as default
        fallback = AI_SERVICE_ROOT / "data" / base_name

    fallback.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using disk storage at {fallback}")
    return fallback


def get_games_directory(prefer_ramdrive: bool = False, config: RamdriveConfig | None = None) -> Path:
    """Get the games data directory."""
    return get_data_directory(prefer_ramdrive, config, base_name="games")


def get_logs_directory(prefer_ramdrive: bool = False, config: RamdriveConfig | None = None) -> Path:
    """Get the logs directory."""
    return get_data_directory(prefer_ramdrive, config, base_name="logs")


def get_models_directory(prefer_ramdrive: bool = False, config: RamdriveConfig | None = None) -> Path:
    """Get the models directory."""
    return get_data_directory(prefer_ramdrive, config, base_name="models")


def get_checkpoints_directory(prefer_ramdrive: bool = False, config: RamdriveConfig | None = None) -> Path:
    """Get the checkpoints directory."""
    return get_data_directory(prefer_ramdrive, config, base_name="checkpoints")


class RamdriveSyncer:
    """Background syncer for ramdrive data to persistent storage."""

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        interval: int = 300,
        patterns: list[str] | None = None,
        on_complete: Callable[[Path, Path, bool], None] | None = None,
    ):
        """Initialize the syncer.

        Args:
            source_dir: Source directory (ramdrive).
            target_dir: Target directory (persistent storage).
            interval: Sync interval in seconds.
            patterns: File patterns to sync (e.g., ["*.db", "*.jsonl"]).
            on_complete: Callback when sync completes (source, target, success).
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.interval = interval
        self.patterns = patterns or ["*.db", "*.jsonl", "*.npz"]
        self.on_complete = on_complete

        self._running = False
        self._thread: threading.Thread | None = None
        self._last_sync_time = 0.0
        self._sync_count = 0
        self._error_count = 0

    def start(self) -> None:
        """Start background sync thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started ramdrive syncer: {self.source_dir} -> {self.target_dir} (every {self.interval}s)")

    def stop(self, final_sync: bool = True) -> None:
        """Stop background sync and optionally perform final sync."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

        if final_sync:
            logger.info("Performing final sync before shutdown...")
            self.sync_now()

    def sync_now(self) -> bool:
        """Perform immediate sync."""
        try:
            self.target_dir.mkdir(parents=True, exist_ok=True)

            # Use rsync if available for efficiency
            if shutil.which("rsync"):
                success = self._rsync_files()
            else:
                success = self._copy_files()

            self._last_sync_time = time.time()
            self._sync_count += 1

            if self.on_complete:
                self.on_complete(self.source_dir, self.target_dir, success)

            return success

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self._error_count += 1
            if self.on_complete:
                self.on_complete(self.source_dir, self.target_dir, False)
            return False

    def _sync_loop(self) -> None:
        """Background sync loop."""
        while self._running:
            time.sleep(self.interval)
            if self._running:
                self.sync_now()

    def _rsync_files(self) -> bool:
        """Sync using rsync."""
        try:
            # Build include patterns
            includes = []
            for pattern in self.patterns:
                includes.extend(["--include", pattern])

            cmd = [
                "rsync", "-av", "--delete",
                *includes,
                "--exclude", "*",
                str(self.source_dir) + "/",
                str(self.target_dir) + "/",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.debug(f"Rsync completed: {self.source_dir} -> {self.target_dir}")
                return True
            else:
                logger.warning(f"Rsync failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Rsync timed out")
            return False
        except Exception as e:
            logger.error(f"Rsync error: {e}")
            return False

    def _copy_files(self) -> bool:
        """Sync using Python copy (fallback)."""
        try:
            synced = 0
            for pattern in self.patterns:
                for src_file in self.source_dir.glob(f"**/{pattern}"):
                    rel_path = src_file.relative_to(self.source_dir)
                    dst_file = self.target_dir / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    synced += 1

            logger.debug(f"Copied {synced} files: {self.source_dir} -> {self.target_dir}")
            return True

        except Exception as e:
            logger.error(f"Copy failed: {e}")
            return False

    @property
    def stats(self) -> dict:
        """Get sync statistics."""
        return {
            "sync_count": self._sync_count,
            "error_count": self._error_count,
            "last_sync_time": self._last_sync_time,
            "source_dir": str(self.source_dir),
            "target_dir": str(self.target_dir),
            "interval": self.interval,
        }


def add_ramdrive_args(parser, include_auto: bool = True) -> None:
    """Add standard ramdrive arguments to an argparse parser.

    Args:
        parser: argparse.ArgumentParser instance.
        include_auto: Whether to include --auto-ramdrive flag.
    """
    group = parser.add_argument_group("Storage Options")
    group.add_argument(
        "--ram-storage", "--ramdrive",
        action="store_true",
        help="Use RAM-backed storage (/dev/shm) for faster I/O"
    )
    if include_auto:
        group.add_argument(
            "--auto-ramdrive",
            action="store_true",
            help="Auto-detect if ramdrive should be used (RAM-rich + disk-limited machines)"
        )
    group.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Override data directory path"
    )
    group.add_argument(
        "--sync-interval",
        type=int,
        default=0,
        help="Sync ramdrive to disk every N seconds (0 = no sync)"
    )
    group.add_argument(
        "--sync-target",
        type=str,
        default="",
        help="Target directory for ramdrive sync"
    )


def get_config_from_args(args, data_path: Path | None = None) -> RamdriveConfig:
    """Create RamdriveConfig from parsed arguments.

    Args:
        args: Parsed argparse namespace with ramdrive arguments.
        data_path: Optional path to check disk space for auto-detection.

    Returns:
        RamdriveConfig instance.
    """
    # Determine if ramdrive should be used
    force_ramdrive = getattr(args, "ram_storage", False)
    auto_ramdrive = getattr(args, "auto_ramdrive", False)

    if auto_ramdrive and not force_ramdrive:
        # Auto-detect based on system resources
        prefer_ramdrive = should_use_ramdrive(data_path=data_path)
        if prefer_ramdrive:
            logger.info("Auto-detection recommends using ramdrive")
    else:
        prefer_ramdrive = force_ramdrive

    return RamdriveConfig(
        prefer_ramdrive=prefer_ramdrive,
        fallback_path=getattr(args, "data_dir", ""),
        sync_interval=getattr(args, "sync_interval", 0),
        sync_target=getattr(args, "sync_target", ""),
    )


# Module-level convenience for checking ramdrive status
_cached_status: RamdriveStatus | None = None


def is_ramdrive_available(min_free_gb: float = 1.0) -> bool:
    """Quick check if ramdrive is available with sufficient space."""
    global _cached_status
    if _cached_status is None:
        _cached_status = detect_ramdrive()
    return _cached_status.available and _cached_status.free_gb >= min_free_gb


def get_ramdrive_path() -> Path | None:
    """Get the ramdrive path if available."""
    global _cached_status
    if _cached_status is None:
        _cached_status = detect_ramdrive()
    return _cached_status.path if _cached_status.available else None


@dataclass
class SystemResources:
    """System resource information for auto-detection."""
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    total_disk_gb: float = 0.0
    free_disk_gb: float = 0.0
    disk_usage_percent: float = 0.0
    ramdrive_free_gb: float = 0.0
    is_ram_rich: bool = False
    is_disk_limited: bool = False
    recommend_ramdrive: bool = False


def get_system_resources(data_path: Path | None = None) -> SystemResources:
    """Detect system resources to determine if ramdrive should be preferred.

    Args:
        data_path: Path to check disk space for. Defaults to current directory.

    Returns:
        SystemResources with RAM/disk info and recommendations.
    """
    resources = SystemResources()

    # Get RAM info
    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
        for line in meminfo.split("\n"):
            if line.startswith("MemTotal:"):
                resources.total_ram_gb = int(line.split()[1]) / (1024 * 1024)
            elif line.startswith("MemAvailable:"):
                resources.available_ram_gb = int(line.split()[1]) / (1024 * 1024)
    except (FileNotFoundError, PermissionError):
        # macOS or other system - try sysctl
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                resources.total_ram_gb = int(result.stdout.strip()) / (1024 ** 3)
                # Estimate available as 70% of total on macOS
                resources.available_ram_gb = resources.total_ram_gb * 0.7
        except (subprocess.SubprocessError, ValueError):
            pass

    # Get disk info
    check_path = data_path or Path.cwd()
    try:
        usage = shutil.disk_usage(check_path)
        resources.total_disk_gb = usage.total / (1024 ** 3)
        resources.free_disk_gb = usage.free / (1024 ** 3)
        resources.disk_usage_percent = (usage.used / usage.total) * 100
    except (OSError, PermissionError):
        pass

    # Get ramdrive info
    ramdrive_status = detect_ramdrive()
    if ramdrive_status.available:
        resources.ramdrive_free_gb = ramdrive_status.free_gb

    # Determine if machine is RAM-rich
    # RAM-rich: > 32GB total RAM or > 16GB available
    resources.is_ram_rich = (
        resources.total_ram_gb > 32 or
        resources.available_ram_gb > 16
    )

    # Determine if disk is limited
    # Disk-limited: < 50GB free OR > DISK_SYNC_TARGET_PERCENT used OR total < 100GB
    # See app.config.thresholds for canonical disk thresholds (70/85/90%)
    resources.is_disk_limited = (
        resources.free_disk_gb < 50 or
        resources.disk_usage_percent > 70 or  # DISK_SYNC_TARGET_PERCENT
        resources.total_disk_gb < 100
    )

    # Recommend ramdrive if:
    # 1. RAM-rich AND disk-limited, OR
    # 2. Ramdrive has > 8GB free and disk usage > 60%, OR
    # 3. Ramdrive has more free space than disk
    resources.recommend_ramdrive = (
        (resources.is_ram_rich and resources.is_disk_limited) or
        (resources.ramdrive_free_gb > 8 and resources.disk_usage_percent > 60) or
        (resources.ramdrive_free_gb > resources.free_disk_gb and resources.ramdrive_free_gb > 4)
    )

    return resources


def should_use_ramdrive(
    data_path: Path | None = None,
    min_ramdrive_gb: float = 4.0,
    force: bool | None = None,
) -> bool:
    """Determine if ramdrive should be used based on system resources.

    Args:
        data_path: Path to check disk space for.
        min_ramdrive_gb: Minimum ramdrive space required.
        force: If provided, override auto-detection.

    Returns:
        True if ramdrive should be used.
    """
    if force is not None:
        return force

    resources = get_system_resources(data_path)

    if not is_ramdrive_available(min_ramdrive_gb):
        return False

    return resources.recommend_ramdrive


def get_auto_storage_path(
    base_name: str = "data",
    subdirectory: str = "",
    fallback_path: Path | None = None,
    min_ramdrive_gb: float = 4.0,
    force_ramdrive: bool | None = None,
) -> tuple[Path, bool]:
    """Automatically select storage path based on system resources.

    This is the main entry point for scripts that want automatic
    ramdrive selection based on machine characteristics.

    Args:
        base_name: Base directory name (e.g., "games", "training").
        subdirectory: Subdirectory within the storage location.
        fallback_path: Path to use if ramdrive not available/recommended.
        min_ramdrive_gb: Minimum ramdrive space required.
        force_ramdrive: Override auto-detection (True=force ramdrive, False=force disk).

    Returns:
        Tuple of (path, using_ramdrive).
    """
    use_ramdrive = should_use_ramdrive(
        data_path=fallback_path,
        min_ramdrive_gb=min_ramdrive_gb,
        force=force_ramdrive,
    )

    if use_ramdrive:
        ramdrive_status = detect_ramdrive()
        if ramdrive_status.available:
            path = ramdrive_status.path / RINGRIFT_SUBDIR
            if subdirectory:
                path = path / subdirectory
            path = path / base_name
            path.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Auto-selected ramdrive storage: {path} "
                f"(ramdrive: {ramdrive_status.free_gb:.1f}GB free)"
            )
            return path, True

    # Fall back to disk
    if fallback_path:
        path = fallback_path / base_name
    else:
        path = AI_SERVICE_ROOT / "data" / base_name

    if subdirectory:
        path = path.parent / subdirectory / path.name

    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using disk storage: {path}")
    return path, False


def log_storage_recommendation(data_path: Path | None = None) -> None:
    """Log storage recommendation based on system resources.

    Useful for debugging and understanding why ramdrive was/wasn't selected.
    """
    resources = get_system_resources(data_path)
    ramdrive_status = detect_ramdrive()

    logger.info("=" * 60)
    logger.info("STORAGE AUTO-DETECTION REPORT")
    logger.info("=" * 60)
    logger.info(f"System RAM:     {resources.total_ram_gb:.1f}GB total, {resources.available_ram_gb:.1f}GB available")
    logger.info(f"Disk:           {resources.total_disk_gb:.1f}GB total, {resources.free_disk_gb:.1f}GB free ({resources.disk_usage_percent:.1f}% used)")
    logger.info(f"Ramdrive:       {ramdrive_status.path} - {resources.ramdrive_free_gb:.1f}GB free" if ramdrive_status.available else "Ramdrive:       Not available")
    logger.info(f"RAM-rich:       {'Yes' if resources.is_ram_rich else 'No'}")
    logger.info(f"Disk-limited:   {'Yes' if resources.is_disk_limited else 'No'}")
    logger.info(f"Recommendation: {'USE RAMDRIVE' if resources.recommend_ramdrive else 'USE DISK'}")
    logger.info("=" * 60)
