"""Training Data Freshness Check - Ensures training nodes have fresh data.

This module provides pre-training data freshness validation to ensure
training nodes have up-to-date data before starting training:

- Checks age of local training data (databases and NPZ files)
- Triggers sync from cluster if data is stale (> configurable threshold)
- Waits for sync completion before allowing training to start
- Integrates with ClusterManifest for remote data discovery

Usage:
    from app.coordination.training_freshness import (
        TrainingFreshnessChecker,
        get_freshness_checker,
        ensure_fresh_data,
    )

    # Quick check and sync
    checker = get_freshness_checker()
    is_fresh = await checker.ensure_fresh_data(
        board_type="hex8",
        num_players=2,
    )

    # Pre-training check
    result = await ensure_fresh_data(
        board_type="square8",
        num_players=2,
        max_age_hours=1.0,
        wait_for_sync=True,
    )
    if result.success:
        # Proceed with training
        ...

Integration with train.py:
    # Add to training startup
    if config.ensure_data_freshness:
        await ensure_fresh_data(config.board_type, config.num_players)
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    # Data classes
    "FreshnessConfig",
    "FreshnessResult",
    "DataSourceInfo",
    "WatcherConfig",
    # Main classes
    "TrainingFreshnessChecker",
    "HighQualityDataSyncWatcher",
    # Singleton accessors
    "get_freshness_checker",
    "reset_freshness_checker",
    "get_high_quality_sync_watcher",
    "reset_high_quality_sync_watcher",
    # Convenience functions
    "check_freshness_sync",
    # Constants
    "DEFAULT_MAX_AGE_HOURS",
    "DEFAULT_SYNC_TIMEOUT_SECONDS",
    "ALL_CONFIGS",
]

# Default freshness threshold (1 hour)
DEFAULT_MAX_AGE_HOURS = 1.0

# Sync timeout (5 minutes)
DEFAULT_SYNC_TIMEOUT_SECONDS = 300

# Data directories
_AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _AI_SERVICE_ROOT / "data"
_GAMES_DIR = _DATA_DIR / "games"
_TRAINING_DIR = _DATA_DIR / "training"


@dataclass
class FreshnessConfig:
    """Configuration for data freshness checking."""
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS
    sync_timeout_seconds: int = DEFAULT_SYNC_TIMEOUT_SECONDS
    wait_for_sync: bool = True
    trigger_sync: bool = True
    check_databases: bool = True
    check_npz_files: bool = True
    min_games_required: int = 1000  # Minimum games for valid training data


@dataclass
class FreshnessResult:
    """Result of a freshness check."""
    success: bool
    is_fresh: bool
    data_age_hours: float
    games_available: int
    sync_triggered: bool = False
    sync_completed: bool = False
    sync_duration_seconds: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSourceInfo:
    """Information about a data source.

    P1.6 Dec 2025: Added content_age_hours for actual data age (vs file mtime).
    File mtime can be misleading when files are copied/synced between nodes.
    """
    path: Path
    age_hours: float  # File modification time age
    size_bytes: int
    game_count: int = 0
    is_stale: bool = False
    board_type: str | None = None
    num_players: int | None = None
    # P1.6 Dec 2025: Content-based age (newest game time or NPZ metadata)
    content_age_hours: float | None = None  # None if unknown


class TrainingFreshnessChecker:
    """Checks and ensures training data freshness.

    This class provides:
    - Local data age checking (databases and NPZ files)
    - Remote data availability check via ClusterManifest
    - Automatic sync triggering when data is stale
    - Sync completion waiting for pre-training validation
    """

    def __init__(
        self,
        config: FreshnessConfig | None = None,
        data_dir: Path | None = None,
    ):
        """Initialize the freshness checker.

        Args:
            config: Freshness check configuration
            data_dir: Base data directory (defaults to ai-service/data)
        """
        self.config = config or FreshnessConfig()
        self.data_dir = data_dir or _DATA_DIR
        self.node_id = socket.gethostname()

        # Lazy-loaded components
        self._manifest = None
        self._sync_daemon = None

        logger.debug(
            f"TrainingFreshnessChecker initialized: "
            f"max_age={self.config.max_age_hours}h, "
            f"sync_timeout={self.config.sync_timeout_seconds}s"
        )

    @property
    def games_dir(self) -> Path:
        """Get the games directory."""
        return self.data_dir / "games"

    @property
    def training_dir(self) -> Path:
        """Get the training data directory."""
        return self.data_dir / "training"

    def _get_manifest(self) -> "ClusterManifest | None":
        """Lazy-load ClusterManifest.

        Returns:
            ClusterManifest instance or None if unavailable.
        """
        if self._manifest is None:
            try:
                from app.distributed.cluster_manifest import get_cluster_manifest
                self._manifest = get_cluster_manifest()
            except ImportError:
                logger.debug("ClusterManifest not available")
        return self._manifest

    def _get_sync_daemon(self) -> "AutoSyncDaemon | None":
        """Lazy-load AutoSyncDaemon.

        Returns:
            AutoSyncDaemon instance or None if unavailable.
        """
        if self._sync_daemon is None:
            try:
                from app.coordination.auto_sync_daemon import get_auto_sync_daemon
                self._sync_daemon = get_auto_sync_daemon()
            except ImportError:
                logger.debug("AutoSyncDaemon not available")
        return self._sync_daemon

    def find_local_databases(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[DataSourceInfo]:
        """Find local game databases.

        Args:
            board_type: Filter by board type
            num_players: Filter by player count

        Returns:
            List of DataSourceInfo for matching databases
        """
        sources = []
        now = time.time()

        if not self.games_dir.exists():
            return sources

        # Search patterns
        patterns = ["*.db"]
        if board_type:
            patterns = [
                f"*{board_type}*.db",
                f"canonical_{board_type}*.db",
                f"selfplay_{board_type}*.db",
            ]

        for pattern in patterns:
            for db_path in self.games_dir.glob(pattern):
                if not db_path.is_file():
                    continue

                # Filter by num_players if specified
                name = db_path.name.lower()
                if num_players:
                    player_pattern = f"{num_players}p"
                    if player_pattern not in name:
                        continue

                stat = db_path.stat()
                age_hours = (now - stat.st_mtime) / 3600

                # Get game count
                game_count = self._get_db_game_count(db_path)

                sources.append(DataSourceInfo(
                    path=db_path,
                    age_hours=age_hours,
                    size_bytes=stat.st_size,
                    game_count=game_count,
                    is_stale=age_hours > self.config.max_age_hours,
                    board_type=board_type,
                    num_players=num_players,
                ))

        return sources

    def find_local_npz_files(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[DataSourceInfo]:
        """Find local NPZ training files.

        Args:
            board_type: Filter by board type
            num_players: Filter by player count

        Returns:
            List of DataSourceInfo for matching NPZ files
        """
        sources = []
        now = time.time()

        if not self.training_dir.exists():
            return sources

        # Search patterns
        patterns = ["*.npz"]
        if board_type:
            patterns = [
                f"*{board_type}*.npz",
                f"{board_type}_{num_players}p*.npz" if num_players else f"{board_type}*.npz",
            ]

        for pattern in patterns:
            for npz_path in self.training_dir.glob(pattern):
                if not npz_path.is_file():
                    continue

                # Filter by num_players if specified
                name = npz_path.name.lower()
                if num_players:
                    player_pattern = f"{num_players}p"
                    if player_pattern not in name:
                        continue

                stat = npz_path.stat()
                age_hours = (now - stat.st_mtime) / 3600

                # Estimate sample count from file size
                # ~200 bytes per sample for typical configurations
                estimated_samples = stat.st_size // 200

                sources.append(DataSourceInfo(
                    path=npz_path,
                    age_hours=age_hours,
                    size_bytes=stat.st_size,
                    game_count=estimated_samples,  # Actually sample count
                    is_stale=age_hours > self.config.max_age_hours,
                    board_type=board_type,
                    num_players=num_players,
                ))

        return sources

    def _get_db_game_count(self, db_path: Path) -> int:
        """Get game count from a database."""
        try:
            import sqlite3
            with sqlite3.connect(db_path, timeout=5.0) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM games")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.warning(f"Failed to get game count from {db_path}: {e}")
            return 0

    def check_freshness(
        self,
        board_type: str,
        num_players: int,
    ) -> FreshnessResult:
        """Check freshness of local training data.

        Args:
            board_type: Board type to check
            num_players: Number of players

        Returns:
            FreshnessResult with freshness status
        """
        databases = []
        npz_files = []

        if self.config.check_databases:
            databases = self.find_local_databases(board_type, num_players)

        if self.config.check_npz_files:
            npz_files = self.find_local_npz_files(board_type, num_players)

        # Compute aggregate stats
        all_sources = databases + npz_files
        if not all_sources:
            return FreshnessResult(
                success=False,
                is_fresh=False,
                data_age_hours=float("inf"),
                games_available=0,
                error="No training data found",
            )

        # Find freshest data
        freshest = min(all_sources, key=lambda s: s.age_hours)
        total_games = sum(s.game_count for s in databases)

        # Check if any fresh data exists
        fresh_sources = [s for s in all_sources if not s.is_stale]
        is_fresh = len(fresh_sources) > 0 and total_games >= self.config.min_games_required

        # Build details
        details = {
            "databases": [
                {
                    "path": str(s.path),
                    "age_hours": round(s.age_hours, 2),
                    "game_count": s.game_count,
                    "is_stale": s.is_stale,
                }
                for s in databases
            ],
            "npz_files": [
                {
                    "path": str(s.path),
                    "age_hours": round(s.age_hours, 2),
                    "sample_count": s.game_count,
                    "is_stale": s.is_stale,
                }
                for s in npz_files
            ],
            "config_key": f"{board_type}_{num_players}p",
            "threshold_hours": self.config.max_age_hours,
            "min_games_required": self.config.min_games_required,
        }

        return FreshnessResult(
            success=True,
            is_fresh=is_fresh,
            data_age_hours=freshest.age_hours,
            games_available=total_games,
            details=details,
        )

    async def trigger_sync(
        self,
        board_type: str,
        num_players: int,
    ) -> bool:
        """Trigger data sync from cluster.

        Args:
            board_type: Board type to sync
            num_players: Number of players

        Returns:
            True if sync was triggered successfully
        """
        daemon = self._get_sync_daemon()
        if daemon is None:
            logger.warning("AutoSyncDaemon not available for sync trigger")
            return False

        try:
            # Request priority sync for this configuration
            logger.info(f"Triggering priority sync for {board_type}_{num_players}p")

            # Check if daemon is running
            if not daemon._running:
                logger.warning("AutoSyncDaemon not running, starting it")
                await daemon.start()

            # Trigger sync cycle
            await daemon.trigger_sync()
            return True

        except Exception as e:
            logger.error(f"Failed to trigger sync: {e}")
            return False

    async def wait_for_fresh_data(
        self,
        board_type: str,
        num_players: int,
        timeout_seconds: int | None = None,
        poll_interval: float = 5.0,
    ) -> FreshnessResult:
        """Wait for fresh data to become available.

        Args:
            board_type: Board type to check
            num_players: Number of players
            timeout_seconds: Maximum wait time
            poll_interval: Time between checks

        Returns:
            FreshnessResult with final status
        """
        timeout = timeout_seconds or self.config.sync_timeout_seconds
        start_time = time.time()

        while True:
            result = self.check_freshness(board_type, num_players)

            if result.is_fresh:
                result.sync_completed = True
                result.sync_duration_seconds = time.time() - start_time
                logger.info(
                    f"Fresh data available for {board_type}_{num_players}p: "
                    f"{result.games_available} games, age={result.data_age_hours:.1f}h"
                )
                return result

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                result.error = f"Timeout waiting for fresh data ({timeout}s)"
                result.sync_duration_seconds = elapsed
                logger.warning(result.error)
                return result

            logger.debug(
                f"Waiting for fresh data... "
                f"elapsed={elapsed:.0f}s, games={result.games_available}"
            )
            await asyncio.sleep(poll_interval)

    async def ensure_fresh_data(
        self,
        board_type: str,
        num_players: int,
    ) -> FreshnessResult:
        """Ensure fresh training data is available.

        This is the main entry point for pre-training freshness validation.
        It checks local data, triggers sync if needed, and waits for completion.

        Args:
            board_type: Board type for training
            num_players: Number of players

        Returns:
            FreshnessResult with final status
        """
        config_key = f"{board_type}_{num_players}p"
        logger.info(f"Checking training data freshness for {config_key}")

        # Check current freshness
        result = self.check_freshness(board_type, num_players)

        if result.is_fresh:
            logger.info(
                f"Training data for {config_key} is fresh: "
                f"{result.games_available} games, age={result.data_age_hours:.1f}h"
            )
            await self._emit_freshness_event("fresh", board_type, num_players, result)
            return result

        # Data is stale - trigger sync if configured
        await self._emit_freshness_event("stale", board_type, num_players, result)

        if not self.config.trigger_sync:
            logger.warning(
                f"Training data for {config_key} is stale "
                f"(age={result.data_age_hours:.1f}h > {self.config.max_age_hours}h) "
                f"but sync is disabled"
            )
            return result

        logger.info(
            f"Training data for {config_key} is stale "
            f"(age={result.data_age_hours:.1f}h), triggering sync"
        )

        # Trigger sync
        sync_triggered = await self.trigger_sync(board_type, num_players)
        result.sync_triggered = sync_triggered

        if sync_triggered:
            await self._emit_freshness_event("sync_triggered", board_type, num_players, result)

        if not sync_triggered:
            result.error = "Failed to trigger sync"
            return result

        # Wait for fresh data if configured
        if not self.config.wait_for_sync:
            logger.info(f"Sync triggered for {config_key}, not waiting for completion")
            return result

        # Wait for sync to complete
        return await self.wait_for_fresh_data(
            board_type,
            num_players,
            timeout_seconds=self.config.sync_timeout_seconds,
        )

    async def _emit_freshness_event(
        self,
        event_kind: str,
        board_type: str,
        num_players: int,
        result: FreshnessResult,
    ) -> None:
        """Emit freshness event for feedback loop coupling.

        Args:
            event_kind: "fresh", "stale", or "sync_triggered"
            board_type: Board type
            num_players: Number of players
            result: Freshness check result
        """
        try:
            from app.coordination.event_router import get_router, DataEventType

            event_type_map = {
                "fresh": DataEventType.DATA_FRESH,
                "stale": DataEventType.DATA_STALE,
                "sync_triggered": DataEventType.SYNC_TRIGGERED,
            }
            event_type = event_type_map.get(event_kind)
            if not event_type:
                return

            router = get_router()
            if router:
                await router.publish(
                    event_type=event_type,
                    payload={
                        "node_id": self.node_id,
                        "board_type": board_type,
                        "num_players": num_players,
                        "is_fresh": result.is_fresh,
                        "data_age_hours": result.data_age_hours,
                        "games_available": result.games_available,
                        "timestamp": time.time(),
                    },
                    source="TrainingFreshnessChecker",
                )
        except Exception as e:
            logger.debug(f"Could not emit {event_kind} event: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get checker status."""
        return {
            "node_id": self.node_id,
            "config": {
                "max_age_hours": self.config.max_age_hours,
                "sync_timeout_seconds": self.config.sync_timeout_seconds,
                "wait_for_sync": self.config.wait_for_sync,
                "trigger_sync": self.config.trigger_sync,
                "min_games_required": self.config.min_games_required,
            },
            "data_dir": str(self.data_dir),
            "games_dir_exists": self.games_dir.exists(),
            "training_dir_exists": self.training_dir.exists(),
        }


# Module-level singleton
_freshness_checker: TrainingFreshnessChecker | None = None


def get_freshness_checker(
    config: FreshnessConfig | None = None,
) -> TrainingFreshnessChecker:
    """Get the singleton TrainingFreshnessChecker instance."""
    global _freshness_checker
    if _freshness_checker is None:
        _freshness_checker = TrainingFreshnessChecker(config)
    return _freshness_checker


def reset_freshness_checker() -> None:
    """Reset the singleton (for testing)."""
    global _freshness_checker
    _freshness_checker = None


async def ensure_fresh_data(
    board_type: str,
    num_players: int,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    wait_for_sync: bool = True,
    trigger_sync: bool = True,
) -> FreshnessResult:
    """Convenience function to ensure fresh training data.

    Args:
        board_type: Board type for training
        num_players: Number of players
        max_age_hours: Maximum age for "fresh" data
        wait_for_sync: Wait for sync to complete
        trigger_sync: Trigger sync if data is stale

    Returns:
        FreshnessResult with status
    """
    config = FreshnessConfig(
        max_age_hours=max_age_hours,
        wait_for_sync=wait_for_sync,
        trigger_sync=trigger_sync,
    )
    checker = TrainingFreshnessChecker(config)
    return await checker.ensure_fresh_data(board_type, num_players)


def check_freshness_sync(
    board_type: str,
    num_players: int,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> FreshnessResult:
    """Synchronous freshness check (no sync triggering).

    Args:
        board_type: Board type for training
        num_players: Number of players
        max_age_hours: Maximum age for "fresh" data

    Returns:
        FreshnessResult with status
    """
    config = FreshnessConfig(
        max_age_hours=max_age_hours,
        trigger_sync=False,
        wait_for_sync=False,
    )
    checker = TrainingFreshnessChecker(config)
    return checker.check_freshness(board_type, num_players)


# =============================================================================
# HighQualityDataSyncWatcher Daemon
# =============================================================================


@dataclass
class WatcherConfig:
    """Configuration for HighQualityDataSyncWatcher."""

    check_interval_seconds: float = 300.0  # 5 minutes
    max_age_hours: float = 1.0
    configs_to_watch: list[tuple[str, int]] | None = None  # None = all configs
    trigger_sync: bool = True
    emit_events: bool = True


# All board configurations
ALL_CONFIGS: list[tuple[str, int]] = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


class HighQualityDataSyncWatcher:
    """Daemon that monitors training data freshness and triggers sync.

    This daemon periodically checks all board configurations for stale
    training data and triggers sync when needed. It integrates with
    the coordination infrastructure for automatic data freshness.

    Usage:
        watcher = HighQualityDataSyncWatcher()
        await watcher.start()
        # ... runs until stopped
        await watcher.stop()
    """

    def __init__(self, config: WatcherConfig | None = None):
        """Initialize the watcher.

        Args:
            config: Watcher configuration
        """
        self.config = config or WatcherConfig()
        self._running = False
        self._task: asyncio.Task | None = None
        self._checker = TrainingFreshnessChecker(
            FreshnessConfig(
                max_age_hours=self.config.max_age_hours,
                trigger_sync=self.config.trigger_sync,
                wait_for_sync=False,  # Don't block on sync
            )
        )
        self._start_time = 0.0
        self._checks_completed = 0
        self._syncs_triggered = 0
        self._errors = 0

    @property
    def is_running(self) -> bool:
        """Whether the watcher is running."""
        return self._running

    async def start(self) -> None:
        """Start the watcher daemon."""
        if self._running:
            logger.warning("HighQualityDataSyncWatcher already running")
            return

        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("HighQualityDataSyncWatcher started")

    async def stop(self) -> None:
        """Stop the watcher daemon."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("HighQualityDataSyncWatcher stopped")

    async def _run_loop(self) -> None:
        """Main watcher loop."""
        while self._running:
            try:
                await self._check_all_configs()
                self._checks_completed += 1
            except Exception as e:
                self._errors += 1
                logger.error(f"Error in freshness check loop: {e}")

            # Wait for next check interval
            try:
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break

    async def _check_all_configs(self) -> None:
        """Check freshness for all watched configurations."""
        configs = self.config.configs_to_watch or ALL_CONFIGS

        for board_type, num_players in configs:
            if not self._running:
                break

            try:
                result = await self._checker.ensure_fresh_data(
                    board_type, num_players
                )
                if result.sync_triggered:
                    self._syncs_triggered += 1
                    logger.info(
                        f"Triggered sync for {board_type}_{num_players}p "
                        f"(age={result.data_age_hours:.1f}h)"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to check freshness for {board_type}_{num_players}p: {e}"
                )

    def health_check(self) -> "HealthCheckResult":
        """Return health status for DaemonManager integration.

        December 2025: Updated to return HealthCheckResult for protocol compliance.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        uptime = time.time() - self._start_time if self._start_time > 0 else 0
        is_healthy = self._running and self._errors < 10

        if not self._running:
            return HealthCheckResult(
                healthy=True,  # Stopped is not unhealthy
                status=CoordinatorStatus.STOPPED,
                message="HighQualityDataSyncWatcher not running",
            )

        if self._errors >= 10:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"Too many errors ({self._errors})",
                details={
                    "uptime_seconds": uptime,
                    "checks_completed": self._checks_completed,
                    "syncs_triggered": self._syncs_triggered,
                    "errors": self._errors,
                    "check_interval": self.config.check_interval_seconds,
                },
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Checks: {self._checks_completed}, Syncs: {self._syncs_triggered}",
            details={
                "uptime_seconds": uptime,
                "checks_completed": self._checks_completed,
                "syncs_triggered": self._syncs_triggered,
                "errors": self._errors,
                "check_interval": self.config.check_interval_seconds,
            },
        )

    def get_status(self) -> dict[str, Any]:
        """Get watcher status."""
        health = self.health_check()
        return {
            "running": self._running,
            **health.to_dict(),
            "config": {
                "check_interval_seconds": self.config.check_interval_seconds,
                "max_age_hours": self.config.max_age_hours,
                "trigger_sync": self.config.trigger_sync,
                "configs_count": len(self.config.configs_to_watch or ALL_CONFIGS),
            },
        }


# Singleton for HighQualityDataSyncWatcher
_watcher_instance: HighQualityDataSyncWatcher | None = None


def get_high_quality_sync_watcher(
    config: WatcherConfig | None = None,
) -> HighQualityDataSyncWatcher:
    """Get singleton HighQualityDataSyncWatcher instance."""
    global _watcher_instance
    if _watcher_instance is None:
        _watcher_instance = HighQualityDataSyncWatcher(config)
    return _watcher_instance


def reset_high_quality_sync_watcher() -> None:
    """Reset the singleton (for testing)."""
    global _watcher_instance
    _watcher_instance = None
