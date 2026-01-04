"""S3 Inventory - Query AWS S3 for game and training data counts (January 2026).

This module provides utilities to query S3 for:
- Game counts per configuration (board_type, num_players)
- NPZ file counts and sample totals
- Model inventory for backup verification

Used by BackupCompletenessTracker to verify that local data is properly
backed up to S3.

Usage:
    from app.coordination.s3_inventory import (
        S3Inventory,
        get_s3_inventory,
        S3InventoryConfig,
    )

    inventory = get_s3_inventory()
    counts = await inventory.get_game_counts()
    # Returns: {"hex8_2p": 50000, "square8_4p": 10000, ...}

Environment Variables:
    RINGRIFT_S3_BUCKET: S3 bucket name (default: ringrift-models-20251214)
    RINGRIFT_S3_REGION: AWS region (default: us-east-1)
    AWS_ACCESS_KEY_ID: AWS credentials
    AWS_SECRET_ACCESS_KEY: AWS credentials
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.coordination.event_utils import make_config_key

logger = logging.getLogger(__name__)

__all__ = [
    "S3Inventory",
    "S3InventoryConfig",
    "S3GameCount",
    "S3InventoryStats",
    "get_s3_inventory",
    "get_s3_game_counts",
]

# Default bucket configuration
DEFAULT_S3_BUCKET = "ringrift-models-20251214"
DEFAULT_S3_REGION = "us-east-1"


@dataclass
class S3InventoryConfig:
    """Configuration for S3 inventory queries."""

    bucket: str = field(
        default_factory=lambda: os.environ.get("RINGRIFT_S3_BUCKET", DEFAULT_S3_BUCKET)
    )
    region: str = field(
        default_factory=lambda: os.environ.get("RINGRIFT_S3_REGION", DEFAULT_S3_REGION)
    )

    # S3 prefixes for different data types
    games_prefix: str = "consolidated/games"
    training_prefix: str = "consolidated/training"
    models_prefix: str = "consolidated/models"

    # Cache settings
    cache_ttl_seconds: float = 300.0  # 5 minutes

    # Query timeout
    query_timeout: float = 60.0  # 1 minute


@dataclass
class S3GameCount:
    """Game count for a specific configuration in S3."""

    config_key: str  # e.g., "hex8_2p"
    board_type: str
    num_players: int
    game_count: int
    database_count: int  # Number of .db files
    total_size_bytes: int
    last_modified: float  # Unix timestamp


@dataclass
class S3InventoryStats:
    """Overall S3 inventory statistics."""

    total_games: int = 0
    total_databases: int = 0
    total_npz_files: int = 0
    total_npz_samples: int = 0
    total_models: int = 0
    total_size_bytes: int = 0
    last_query_time: float = 0.0
    query_duration_seconds: float = 0.0
    games_by_config: dict[str, S3GameCount] = field(default_factory=dict)
    npz_by_config: dict[str, dict] = field(default_factory=dict)


class S3Inventory:
    """Queries AWS S3 for game and training data inventory.

    Uses aws s3 ls and aws s3api to efficiently query bucket contents
    without downloading files.
    """

    def __init__(self, config: S3InventoryConfig | None = None) -> None:
        self.config = config or S3InventoryConfig()
        self._cache: S3InventoryStats | None = None
        self._cache_time: float = 0.0
        self._lock = asyncio.Lock()

    async def get_game_counts(self, force_refresh: bool = False) -> dict[str, int]:
        """Get game counts per configuration from S3.

        Returns:
            Dict mapping config_key (e.g., "hex8_2p") to game count.
        """
        stats = await self._get_cached_stats(force_refresh)
        return {k: v.game_count for k, v in stats.games_by_config.items()}

    async def get_npz_counts(self, force_refresh: bool = False) -> dict[str, dict]:
        """Get NPZ file counts per configuration from S3.

        Returns:
            Dict mapping config_key to {"count": int, "total_samples": int}.
        """
        stats = await self._get_cached_stats(force_refresh)
        return stats.npz_by_config

    async def get_full_stats(self, force_refresh: bool = False) -> S3InventoryStats:
        """Get complete S3 inventory statistics."""
        return await self._get_cached_stats(force_refresh)

    async def _get_cached_stats(self, force_refresh: bool = False) -> S3InventoryStats:
        """Get cached stats or refresh if stale."""
        async with self._lock:
            now = time.time()
            if (
                not force_refresh
                and self._cache is not None
                and (now - self._cache_time) < self.config.cache_ttl_seconds
            ):
                return self._cache

            # Query S3
            stats = await self._query_s3()
            self._cache = stats
            self._cache_time = now
            return stats

    async def _query_s3(self) -> S3InventoryStats:
        """Query S3 for inventory data."""
        start_time = time.time()
        stats = S3InventoryStats()

        try:
            # Query game databases
            await self._query_games(stats)

            # Query NPZ training files
            await self._query_npz(stats)

            # Query models
            await self._query_models(stats)

        except Exception as e:
            logger.warning(f"[S3Inventory] Error querying S3: {e}")

        stats.last_query_time = time.time()
        stats.query_duration_seconds = stats.last_query_time - start_time
        return stats

    async def _query_games(self, stats: S3InventoryStats) -> None:
        """Query S3 for game database files."""
        try:
            # List all .db files in games prefix
            cmd = [
                "aws", "s3", "ls",
                f"s3://{self.config.bucket}/{self.config.games_prefix}/",
                "--recursive",
                "--region", self.config.region,
            ]

            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run, cmd, capture_output=True, text=True
                ),
                timeout=self.config.query_timeout,
            )

            if result.returncode != 0:
                logger.warning(f"[S3Inventory] aws s3 ls failed: {result.stderr}")
                return

            # Parse output: "2024-01-02 12:34:56   123456 path/to/file.db"
            config_pattern = re.compile(r"canonical_(\w+)_(\d+)p\.db$")

            for line in result.stdout.strip().split("\n"):
                if not line or not line.strip():
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                size_bytes = int(parts[2])
                file_path = parts[3]

                if not file_path.endswith(".db"):
                    continue

                # Extract config from filename
                match = config_pattern.search(file_path)
                if match:
                    board_type = match.group(1)
                    num_players = int(match.group(2))
                    config_key = make_config_key(board_type, num_players)

                    if config_key not in stats.games_by_config:
                        stats.games_by_config[config_key] = S3GameCount(
                            config_key=config_key,
                            board_type=board_type,
                            num_players=num_players,
                            game_count=0,
                            database_count=0,
                            total_size_bytes=0,
                            last_modified=0.0,
                        )

                    stats.games_by_config[config_key].database_count += 1
                    stats.games_by_config[config_key].total_size_bytes += size_bytes
                    stats.total_databases += 1
                    stats.total_size_bytes += size_bytes

            # Estimate game counts from file sizes (rough approximation)
            # Average game is ~2KB in SQLite
            for config_key, game_count in stats.games_by_config.items():
                estimated_games = game_count.total_size_bytes // 2048
                game_count.game_count = max(1, estimated_games)
                stats.total_games += game_count.game_count

        except asyncio.TimeoutError:
            logger.warning("[S3Inventory] Timeout querying games from S3")
        except FileNotFoundError:
            logger.warning("[S3Inventory] aws CLI not found")
        except Exception as e:
            logger.warning(f"[S3Inventory] Error querying games: {e}")

    async def _query_npz(self, stats: S3InventoryStats) -> None:
        """Query S3 for NPZ training files."""
        try:
            cmd = [
                "aws", "s3", "ls",
                f"s3://{self.config.bucket}/{self.config.training_prefix}/",
                "--recursive",
                "--region", self.config.region,
            ]

            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run, cmd, capture_output=True, text=True
                ),
                timeout=self.config.query_timeout,
            )

            if result.returncode != 0:
                return

            # Parse NPZ files
            config_pattern = re.compile(r"(\w+)_(\d+)p[_\.].*\.npz$")

            for line in result.stdout.strip().split("\n"):
                if not line or not line.strip():
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                size_bytes = int(parts[2])
                file_path = parts[3]

                if not file_path.endswith(".npz"):
                    continue

                match = config_pattern.search(file_path)
                if match:
                    board_type = match.group(1)
                    num_players = int(match.group(2))
                    config_key = make_config_key(board_type, num_players)

                    if config_key not in stats.npz_by_config:
                        stats.npz_by_config[config_key] = {
                            "count": 0,
                            "total_size_bytes": 0,
                            "estimated_samples": 0,
                        }

                    stats.npz_by_config[config_key]["count"] += 1
                    stats.npz_by_config[config_key]["total_size_bytes"] += size_bytes
                    # Estimate samples: ~100 bytes per sample average
                    stats.npz_by_config[config_key]["estimated_samples"] += size_bytes // 100
                    stats.total_npz_files += 1

        except asyncio.TimeoutError:
            logger.warning("[S3Inventory] Timeout querying NPZ files from S3")
        except Exception as e:
            logger.warning(f"[S3Inventory] Error querying NPZ files: {e}")

    async def _query_models(self, stats: S3InventoryStats) -> None:
        """Query S3 for model checkpoints."""
        try:
            cmd = [
                "aws", "s3", "ls",
                f"s3://{self.config.bucket}/{self.config.models_prefix}/",
                "--recursive",
                "--region", self.config.region,
            ]

            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run, cmd, capture_output=True, text=True
                ),
                timeout=self.config.query_timeout,
            )

            if result.returncode != 0:
                return

            for line in result.stdout.strip().split("\n"):
                if not line or not line.strip():
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                file_path = parts[3]
                if file_path.endswith(".pth") or file_path.endswith(".pt"):
                    stats.total_models += 1

        except Exception as e:
            logger.warning(f"[S3Inventory] Error querying models: {e}")


# Singleton instance
_instance: S3Inventory | None = None


def get_s3_inventory() -> S3Inventory:
    """Get the singleton S3Inventory instance."""
    global _instance
    if _instance is None:
        _instance = S3Inventory()
    return _instance


async def get_s3_game_counts() -> dict[str, int]:
    """Convenience function to get game counts from S3.

    Returns:
        Dict mapping config_key to game count.
    """
    inventory = get_s3_inventory()
    return await inventory.get_game_counts()
