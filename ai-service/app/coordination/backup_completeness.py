"""Backup Completeness Tracker - Track what's backed up vs what exists (January 2026).

This module provides comprehensive backup coverage tracking by comparing:
- Local canonical databases (source of truth)
- S3 backup inventory
- OWC external drive inventory

Tracks per-config backup coverage and alerts when backup lags behind.

Usage:
    from app.coordination.backup_completeness import (
        BackupCompletenessTracker,
        get_backup_completeness_tracker,
        BackupStatus,
    )

    tracker = get_backup_completeness_tracker()
    status = await tracker.get_backup_status("hex8_2p")
    print(f"S3 coverage: {status.s3_coverage:.1%}")
    print(f"OWC coverage: {status.owc_coverage:.1%}")

    # Get all incomplete configs
    incomplete = await tracker.get_incomplete_configs()
    for config in incomplete:
        print(f"{config.config_key}: S3={config.s3_coverage:.1%}, OWC={config.owc_coverage:.1%}")

Part of Sprint 2: Backup Completeness in the Unified Data Consolidation plan.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.coordination.event_utils import parse_config_key
from app.utils.game_discovery import GameDiscovery

logger = logging.getLogger(__name__)

__all__ = [
    "BackupCompletenessTracker",
    "BackupCompletenessConfig",
    "ConfigBackupStatus",
    "BackupCoverage",
    "get_backup_completeness_tracker",
    "get_all_backup_status",
]


class BackupCoverage(str, Enum):
    """Backup coverage level for a configuration."""

    COMPLETE = "complete"  # >= 99% backed up
    GOOD = "good"  # >= 90% backed up
    PARTIAL = "partial"  # >= 50% backed up
    LOW = "low"  # >= 10% backed up
    MISSING = "missing"  # < 10% or no backup
    UNKNOWN = "unknown"  # Cannot determine (source unavailable)


@dataclass
class ConfigBackupStatus:
    """Backup status for a single configuration."""

    config_key: str
    board_type: str
    num_players: int

    # Local (canonical) counts
    local_game_count: int = 0
    local_database_count: int = 0

    # S3 counts
    s3_game_count: int = 0
    s3_database_count: int = 0
    s3_coverage: float = 0.0  # 0.0 to 1.0
    s3_level: BackupCoverage = BackupCoverage.UNKNOWN

    # OWC counts
    owc_game_count: int = 0
    owc_database_count: int = 0
    owc_coverage: float = 0.0  # 0.0 to 1.0
    owc_level: BackupCoverage = BackupCoverage.UNKNOWN

    # Overall status
    dual_backed_up: bool = False  # True if both S3 and OWC have >= 99%
    needs_backup: bool = True
    games_missing_from_s3: int = 0
    games_missing_from_owc: int = 0

    # Timestamps
    last_checked: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_key": self.config_key,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "local_game_count": self.local_game_count,
            "s3_game_count": self.s3_game_count,
            "s3_coverage": self.s3_coverage,
            "s3_level": self.s3_level.value,
            "owc_game_count": self.owc_game_count,
            "owc_coverage": self.owc_coverage,
            "owc_level": self.owc_level.value,
            "dual_backed_up": self.dual_backed_up,
            "needs_backup": self.needs_backup,
            "games_missing_from_s3": self.games_missing_from_s3,
            "games_missing_from_owc": self.games_missing_from_owc,
            "last_checked": self.last_checked,
        }


@dataclass
class BackupCompletenessConfig:
    """Configuration for backup completeness tracking."""

    # Coverage thresholds
    complete_threshold: float = 0.99  # >= 99% = complete
    good_threshold: float = 0.90  # >= 90% = good
    partial_threshold: float = 0.50  # >= 50% = partial
    low_threshold: float = 0.10  # >= 10% = low

    # Alert thresholds
    alert_games_behind: int = 1000  # Alert if backup is this many games behind
    alert_coverage_below: float = 0.90  # Alert if coverage drops below this

    # Cache settings
    cache_ttl_seconds: float = 60.0  # 1 minute

    # Paths
    canonical_dir: Path = field(default_factory=lambda: Path("data/games"))


@dataclass
class OverallBackupStatus:
    """Overall backup status across all configurations."""

    total_local_games: int = 0
    total_s3_games: int = 0
    total_owc_games: int = 0
    overall_s3_coverage: float = 0.0
    overall_owc_coverage: float = 0.0
    configs_complete: int = 0
    configs_incomplete: int = 0
    configs_missing: int = 0
    last_checked: float = 0.0
    by_config: dict[str, ConfigBackupStatus] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_local_games": self.total_local_games,
            "total_s3_games": self.total_s3_games,
            "total_owc_games": self.total_owc_games,
            "overall_s3_coverage": self.overall_s3_coverage,
            "overall_owc_coverage": self.overall_owc_coverage,
            "configs_complete": self.configs_complete,
            "configs_incomplete": self.configs_incomplete,
            "configs_missing": self.configs_missing,
            "last_checked": self.last_checked,
            "by_config": {k: v.to_dict() for k, v in self.by_config.items()},
        }


class BackupCompletenessTracker:
    """Tracks backup completeness across S3 and OWC.

    Compares local canonical database counts with S3 and OWC inventories
    to ensure all data is properly backed up.
    """

    def __init__(self, config: BackupCompletenessConfig | None = None) -> None:
        self.config = config or BackupCompletenessConfig()
        self._cache: OverallBackupStatus | None = None
        self._cache_time: float = 0.0
        self._lock = asyncio.Lock()

        # Lazy-loaded inventories
        self._s3_inventory: Any | None = None
        self._owc_inventory: Any | None = None
        self._discovery: GameDiscovery | None = None

    async def get_backup_status(
        self, config_key: str, force_refresh: bool = False
    ) -> ConfigBackupStatus:
        """Get backup status for a specific configuration.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            force_refresh: Force refresh of cached data

        Returns:
            ConfigBackupStatus with coverage information
        """
        overall = await self._get_cached_status(force_refresh)
        if config_key in overall.by_config:
            return overall.by_config[config_key]

        # Create empty status for unknown config
        parsed = parse_config_key(config_key)
        board_type = parsed.board_type if parsed else "unknown"
        num_players = parsed.num_players if parsed else 2

        return ConfigBackupStatus(
            config_key=config_key,
            board_type=board_type,
            num_players=num_players,
            last_checked=time.time(),
        )

    async def get_all_status(
        self, force_refresh: bool = False
    ) -> OverallBackupStatus:
        """Get overall backup status across all configurations."""
        return await self._get_cached_status(force_refresh)

    async def get_incomplete_configs(
        self, force_refresh: bool = False
    ) -> list[ConfigBackupStatus]:
        """Get list of configs with incomplete backups.

        Returns configs where either S3 or OWC coverage is below 99%.
        """
        overall = await self._get_cached_status(force_refresh)
        return [
            status
            for status in overall.by_config.values()
            if not status.dual_backed_up
        ]

    async def get_configs_missing_from_s3(
        self, force_refresh: bool = False
    ) -> list[ConfigBackupStatus]:
        """Get configs with games missing from S3."""
        overall = await self._get_cached_status(force_refresh)
        return [
            status
            for status in overall.by_config.values()
            if status.games_missing_from_s3 > 0
        ]

    async def get_configs_missing_from_owc(
        self, force_refresh: bool = False
    ) -> list[ConfigBackupStatus]:
        """Get configs with games missing from OWC."""
        overall = await self._get_cached_status(force_refresh)
        return [
            status
            for status in overall.by_config.values()
            if status.games_missing_from_owc > 0
        ]

    async def _get_cached_status(
        self, force_refresh: bool = False
    ) -> OverallBackupStatus:
        """Get cached status or refresh if stale."""
        async with self._lock:
            now = time.time()
            if (
                not force_refresh
                and self._cache is not None
                and (now - self._cache_time) < self.config.cache_ttl_seconds
            ):
                return self._cache

            status = await self._compute_status()
            self._cache = status
            self._cache_time = now
            return status

    async def _compute_status(self) -> OverallBackupStatus:
        """Compute current backup status by comparing all sources."""
        overall = OverallBackupStatus(last_checked=time.time())

        try:
            # Get local game counts
            local_counts = await self._get_local_counts()

            # Get S3 counts
            s3_counts = await self._get_s3_counts()

            # Get OWC counts
            owc_counts = await self._get_owc_counts()

            # Compute status for each config
            all_configs = set(local_counts.keys()) | set(s3_counts.keys()) | set(owc_counts.keys())

            for config_key in all_configs:
                local_count = local_counts.get(config_key, 0)
                s3_count = s3_counts.get(config_key, 0)
                owc_count = owc_counts.get(config_key, 0)

                # Parse config key using centralized utility
                parsed = parse_config_key(config_key)
                board_type = parsed.board_type if parsed else "unknown"
                num_players = parsed.num_players if parsed else 2

                # Compute coverage
                s3_coverage = s3_count / local_count if local_count > 0 else 1.0
                owc_coverage = owc_count / local_count if local_count > 0 else 1.0

                # Determine coverage levels
                s3_level = self._get_coverage_level(s3_coverage)
                owc_level = self._get_coverage_level(owc_coverage)

                status = ConfigBackupStatus(
                    config_key=config_key,
                    board_type=board_type,
                    num_players=num_players,
                    local_game_count=local_count,
                    s3_game_count=s3_count,
                    s3_coverage=min(1.0, s3_coverage),
                    s3_level=s3_level,
                    owc_game_count=owc_count,
                    owc_coverage=min(1.0, owc_coverage),
                    owc_level=owc_level,
                    dual_backed_up=(
                        s3_coverage >= self.config.complete_threshold
                        and owc_coverage >= self.config.complete_threshold
                    ),
                    needs_backup=(
                        s3_coverage < self.config.complete_threshold
                        or owc_coverage < self.config.complete_threshold
                    ),
                    games_missing_from_s3=max(0, local_count - s3_count),
                    games_missing_from_owc=max(0, local_count - owc_count),
                    last_checked=time.time(),
                )

                overall.by_config[config_key] = status
                overall.total_local_games += local_count
                overall.total_s3_games += s3_count
                overall.total_owc_games += owc_count

                if status.dual_backed_up:
                    overall.configs_complete += 1
                elif local_count > 0:
                    overall.configs_incomplete += 1
                else:
                    overall.configs_missing += 1

            # Compute overall coverage
            if overall.total_local_games > 0:
                overall.overall_s3_coverage = (
                    overall.total_s3_games / overall.total_local_games
                )
                overall.overall_owc_coverage = (
                    overall.total_owc_games / overall.total_local_games
                )

        except Exception as e:
            logger.warning(f"[BackupCompleteness] Error computing status: {e}")

        return overall

    def _get_coverage_level(self, coverage: float) -> BackupCoverage:
        """Convert coverage percentage to coverage level."""
        if coverage >= self.config.complete_threshold:
            return BackupCoverage.COMPLETE
        elif coverage >= self.config.good_threshold:
            return BackupCoverage.GOOD
        elif coverage >= self.config.partial_threshold:
            return BackupCoverage.PARTIAL
        elif coverage >= self.config.low_threshold:
            return BackupCoverage.LOW
        else:
            return BackupCoverage.MISSING

    async def _get_local_counts(self) -> dict[str, int]:
        """Get game counts from local canonical databases."""
        if self._discovery is None:
            self._discovery = GameDiscovery()

        # Clear cache to get fresh data
        self._discovery.clear_cache()

        counts: dict[str, int] = {}
        try:
            for db_info in self._discovery.find_all_databases():
                # Only count canonical databases
                if "canonical" not in db_info.path.name:
                    continue

                config_key = f"{db_info.board_type}_{db_info.num_players}p"
                if config_key not in counts:
                    counts[config_key] = 0
                counts[config_key] += db_info.game_count
        except Exception as e:
            logger.warning(f"[BackupCompleteness] Error getting local counts: {e}")

        return counts

    async def _get_s3_counts(self) -> dict[str, int]:
        """Get game counts from S3."""
        try:
            if self._s3_inventory is None:
                from app.coordination.s3_inventory import get_s3_inventory
                self._s3_inventory = get_s3_inventory()

            return await self._s3_inventory.get_game_counts(force_refresh=True)
        except ImportError:
            logger.debug("[BackupCompleteness] S3 inventory module not available")
            return {}
        except Exception as e:
            logger.warning(f"[BackupCompleteness] Error getting S3 counts: {e}")
            return {}

    async def _get_owc_counts(self) -> dict[str, int]:
        """Get game counts from OWC drive."""
        try:
            if self._owc_inventory is None:
                from app.coordination.owc_inventory import get_owc_inventory
                self._owc_inventory = get_owc_inventory()

            return await self._owc_inventory.get_game_counts(force_refresh=True)
        except ImportError:
            logger.debug("[BackupCompleteness] OWC inventory module not available")
            return {}
        except Exception as e:
            logger.warning(f"[BackupCompleteness] Error getting OWC counts: {e}")
            return {}


# Singleton instance
_instance: BackupCompletenessTracker | None = None


def get_backup_completeness_tracker() -> BackupCompletenessTracker:
    """Get the singleton BackupCompletenessTracker instance."""
    global _instance
    if _instance is None:
        _instance = BackupCompletenessTracker()
    return _instance


async def get_all_backup_status() -> OverallBackupStatus:
    """Convenience function to get overall backup status.

    Returns:
        OverallBackupStatus with all config coverage information.
    """
    tracker = get_backup_completeness_tracker()
    return await tracker.get_all_status()
