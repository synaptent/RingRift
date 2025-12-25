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
    # Main class
    "TrainingFreshnessChecker",
    # Singleton accessors
    "get_freshness_checker",
    "reset_freshness_checker",
    # Convenience functions
    "check_freshness_sync",
    # Constants
    "DEFAULT_MAX_AGE_HOURS",
    "DEFAULT_SYNC_TIMEOUT_SECONDS",
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
    """Information about a data source."""
    path: Path
    age_hours: float
    size_bytes: int
    game_count: int = 0
    is_stale: bool = False
    board_type: str | None = None
    num_players: int | None = None


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

    def _get_manifest(self):
        """Lazy-load ClusterManifest."""
        if self._manifest is None:
            try:
                from app.distributed.cluster_manifest import get_cluster_manifest
                self._manifest = get_cluster_manifest()
            except ImportError:
                logger.debug("ClusterManifest not available")
        return self._manifest

    def _get_sync_daemon(self):
        """Lazy-load AutoSyncDaemon."""
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
            return result

        # Data is stale - trigger sync if configured
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


__all__ = [
    "FreshnessConfig",
    "FreshnessResult",
    "DataSourceInfo",
    "TrainingFreshnessChecker",
    "get_freshness_checker",
    "reset_freshness_checker",
    "ensure_fresh_data",
    "check_freshness_sync",
]
