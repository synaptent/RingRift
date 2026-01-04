"""OWC Inventory - Query OWC external drive for game and training data counts (January 2026).

This module provides utilities to query the OWC external drive (mounted on mac-studio) for:
- Game counts per configuration (board_type, num_players)
- NPZ file counts and sample totals
- Model inventory for backup verification

The OWC drive is the primary backup destination for all training data,
providing fast local access and redundancy independent of cloud providers.

Used by BackupCompletenessTracker to verify that local data is properly
backed up to OWC.

Usage:
    from app.coordination.owc_inventory import (
        OWCInventory,
        get_owc_inventory,
        OWCInventoryConfig,
    )

    inventory = get_owc_inventory()
    counts = await inventory.get_game_counts()
    # Returns: {"hex8_2p": 50000, "square8_4p": 10000, ...}

Environment Variables:
    OWC_HOST: Host with OWC drive mounted (default: mac-studio)
    OWC_BASE_PATH: Mount path of OWC drive (default: /Volumes/RingRift-Data)
    OWC_SSH_KEY: SSH key for authentication (default: ~/.ssh/id_ed25519)
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

from app.coordination.event_utils import make_config_key
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "OWCInventory",
    "OWCInventoryConfig",
    "OWCGameCount",
    "OWCInventoryStats",
    "get_owc_inventory",
    "get_owc_game_counts",
]

# Default OWC configuration
DEFAULT_OWC_HOST = "mac-studio"
DEFAULT_OWC_BASE_PATH = "/Volumes/RingRift-Data"
DEFAULT_OWC_SSH_KEY = "~/.ssh/id_ed25519"
DEFAULT_OWC_SSH_USER = "armand"


@dataclass
class OWCInventoryConfig:
    """Configuration for OWC inventory queries."""

    host: str = field(
        default_factory=lambda: os.environ.get("OWC_HOST", DEFAULT_OWC_HOST)
    )
    base_path: str = field(
        default_factory=lambda: os.environ.get("OWC_BASE_PATH", DEFAULT_OWC_BASE_PATH)
    )
    ssh_key: str = field(
        default_factory=lambda: os.path.expanduser(
            os.environ.get("OWC_SSH_KEY", DEFAULT_OWC_SSH_KEY)
        )
    )
    ssh_user: str = field(
        default_factory=lambda: os.environ.get("OWC_SSH_USER", DEFAULT_OWC_SSH_USER)
    )

    # Subdirectories on OWC drive
    games_subdir: str = "games"
    training_subdir: str = "training"
    models_subdir: str = "models"

    # Cache settings
    cache_ttl_seconds: float = 300.0  # 5 minutes

    # Query timeout
    query_timeout: float = 120.0  # 2 minutes (SSH can be slow)


@dataclass
class OWCGameCount:
    """Game count for a specific configuration on OWC drive."""

    config_key: str  # e.g., "hex8_2p"
    board_type: str
    num_players: int
    game_count: int
    database_count: int  # Number of .db files
    total_size_bytes: int
    last_modified: float  # Unix timestamp


@dataclass
class OWCInventoryStats:
    """Overall OWC inventory statistics."""

    total_games: int = 0
    total_databases: int = 0
    total_npz_files: int = 0
    total_npz_samples: int = 0
    total_models: int = 0
    total_size_bytes: int = 0
    drive_available: bool = False
    drive_total_gb: float = 0.0
    drive_used_gb: float = 0.0
    drive_free_gb: float = 0.0
    last_query_time: float = 0.0
    query_duration_seconds: float = 0.0
    games_by_config: dict[str, OWCGameCount] = field(default_factory=dict)
    npz_by_config: dict[str, dict] = field(default_factory=dict)


class OWCInventory:
    """Queries OWC external drive for game and training data inventory.

    Uses SSH to query the mac-studio host where the OWC drive is mounted.
    All queries are done via remote commands (ls, du, sqlite3) without
    downloading files.
    """

    def __init__(self, config: OWCInventoryConfig | None = None) -> None:
        self.config = config or OWCInventoryConfig()
        self._cache: OWCInventoryStats | None = None
        self._cache_time: float = 0.0
        self._lock = asyncio.Lock()

    async def get_game_counts(self, force_refresh: bool = False) -> dict[str, int]:
        """Get game counts per configuration from OWC drive.

        Returns:
            Dict mapping config_key (e.g., "hex8_2p") to game count.
        """
        stats = await self._get_cached_stats(force_refresh)
        return {k: v.game_count for k, v in stats.games_by_config.items()}

    async def get_npz_counts(self, force_refresh: bool = False) -> dict[str, dict]:
        """Get NPZ file counts per configuration from OWC drive.

        Returns:
            Dict mapping config_key to {"count": int, "total_samples": int}.
        """
        stats = await self._get_cached_stats(force_refresh)
        return stats.npz_by_config

    async def get_full_stats(self, force_refresh: bool = False) -> OWCInventoryStats:
        """Get complete OWC inventory statistics."""
        return await self._get_cached_stats(force_refresh)

    async def check_drive_available(self) -> bool:
        """Check if OWC drive is mounted and accessible."""
        try:
            result = await self._run_ssh_command(f"test -d {self.config.base_path} && echo 'OK'")
            return result.strip() == "OK"
        except Exception:
            return False

    async def _get_cached_stats(self, force_refresh: bool = False) -> OWCInventoryStats:
        """Get cached stats or refresh if stale."""
        async with self._lock:
            now = time.time()
            if (
                not force_refresh
                and self._cache is not None
                and (now - self._cache_time) < self.config.cache_ttl_seconds
            ):
                return self._cache

            # Query OWC drive
            stats = await self._query_owc()
            self._cache = stats
            self._cache_time = now
            return stats

    async def _query_owc(self) -> OWCInventoryStats:
        """Query OWC drive for inventory data."""
        start_time = time.time()
        stats = OWCInventoryStats()

        try:
            # Check if drive is available
            stats.drive_available = await self.check_drive_available()
            if not stats.drive_available:
                logger.warning("[OWCInventory] OWC drive not available")
                return stats

            # Get disk usage
            await self._query_disk_usage(stats)

            # Query game databases
            await self._query_games(stats)

            # Query NPZ training files
            await self._query_npz(stats)

            # Query models
            await self._query_models(stats)

        except Exception as e:
            logger.warning(f"[OWCInventory] Error querying OWC: {e}")

        stats.last_query_time = time.time()
        stats.query_duration_seconds = stats.last_query_time - start_time
        return stats

    async def _run_ssh_command(self, command: str) -> str:
        """Run a command on the OWC host via SSH."""
        ssh_cmd = [
            "ssh",
            "-i", self.config.ssh_key,
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            f"{self.config.ssh_user}@{self.config.host}",
            command,
        ]

        result = await asyncio.wait_for(
            asyncio.to_thread(
                subprocess.run, ssh_cmd, capture_output=True, text=True
            ),
            timeout=self.config.query_timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(f"SSH command failed: {result.stderr}")

        return result.stdout

    async def _query_disk_usage(self, stats: OWCInventoryStats) -> None:
        """Query disk usage for OWC drive."""
        try:
            # Get disk free space using df
            output = await self._run_ssh_command(
                f"df -k '{self.config.base_path}' | tail -1"
            )
            parts = output.split()
            if len(parts) >= 4:
                total_kb = int(parts[1])
                used_kb = int(parts[2])
                free_kb = int(parts[3])
                stats.drive_total_gb = total_kb / 1024 / 1024
                stats.drive_used_gb = used_kb / 1024 / 1024
                stats.drive_free_gb = free_kb / 1024 / 1024
        except Exception as e:
            logger.debug(f"[OWCInventory] Error getting disk usage: {e}")

    async def _query_games(self, stats: OWCInventoryStats) -> None:
        """Query OWC for game database files."""
        try:
            games_path = f"{self.config.base_path}/{self.config.games_subdir}"

            # List all .db files with sizes
            output = await self._run_ssh_command(
                f"find '{games_path}' -name '*.db' -exec ls -l {{}} \\; 2>/dev/null || true"
            )

            config_pattern = re.compile(r"canonical_(\w+)_(\d+)p\.db$")

            for line in output.strip().split("\n"):
                if not line or not line.strip():
                    continue

                parts = line.split()
                if len(parts) < 9:
                    continue

                try:
                    size_bytes = int(parts[4])
                    file_path = parts[8]
                except (ValueError, IndexError):
                    continue

                if not file_path.endswith(".db"):
                    continue

                # Extract config from filename
                match = config_pattern.search(file_path)
                if match:
                    board_type = match.group(1)
                    num_players = int(match.group(2))
                    config_key = make_config_key(board_type, num_players)

                    if config_key not in stats.games_by_config:
                        stats.games_by_config[config_key] = OWCGameCount(
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

            # Query actual game counts using sqlite3
            await self._query_game_counts_sql(stats)

        except asyncio.TimeoutError:
            logger.warning("[OWCInventory] Timeout querying games from OWC")
        except Exception as e:
            logger.warning(f"[OWCInventory] Error querying games: {e}")

    async def _query_game_counts_sql(self, stats: OWCInventoryStats) -> None:
        """Query actual game counts from canonical databases."""
        for config_key in list(stats.games_by_config.keys()):
            try:
                db_path = (
                    f"{self.config.base_path}/{self.config.games_subdir}/"
                    f"canonical_{config_key.replace('_', '_')}.db"
                )
                output = await self._run_ssh_command(
                    f"sqlite3 '{db_path}' 'SELECT COUNT(*) FROM games' 2>/dev/null || echo '0'"
                )
                count = int(output.strip())
                stats.games_by_config[config_key].game_count = count
                stats.total_games += count
            except Exception:
                # If we can't query SQL, estimate from file size
                if config_key in stats.games_by_config:
                    size = stats.games_by_config[config_key].total_size_bytes
                    estimated = size // 2048  # ~2KB per game
                    stats.games_by_config[config_key].game_count = max(1, estimated)
                    stats.total_games += max(1, estimated)

    async def _query_npz(self, stats: OWCInventoryStats) -> None:
        """Query OWC for NPZ training files."""
        try:
            training_path = f"{self.config.base_path}/{self.config.training_subdir}"

            output = await self._run_ssh_command(
                f"find '{training_path}' -name '*.npz' -exec ls -l {{}} \\; 2>/dev/null || true"
            )

            config_pattern = re.compile(r"(\w+)_(\d+)p[_\.].*\.npz$")

            for line in output.strip().split("\n"):
                if not line or not line.strip():
                    continue

                parts = line.split()
                if len(parts) < 9:
                    continue

                try:
                    size_bytes = int(parts[4])
                    file_path = parts[8]
                except (ValueError, IndexError):
                    continue

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
                    stats.npz_by_config[config_key]["estimated_samples"] += size_bytes // 100
                    stats.total_npz_files += 1

        except asyncio.TimeoutError:
            logger.warning("[OWCInventory] Timeout querying NPZ files from OWC")
        except Exception as e:
            logger.warning(f"[OWCInventory] Error querying NPZ files: {e}")

    async def _query_models(self, stats: OWCInventoryStats) -> None:
        """Query OWC for model checkpoints."""
        try:
            models_path = f"{self.config.base_path}/{self.config.models_subdir}"

            output = await self._run_ssh_command(
                f"find '{models_path}' -name '*.pth' -o -name '*.pt' 2>/dev/null | wc -l || echo '0'"
            )

            stats.total_models = int(output.strip())

        except Exception as e:
            logger.warning(f"[OWCInventory] Error querying models: {e}")


# Singleton instance
_instance: OWCInventory | None = None


def get_owc_inventory() -> OWCInventory:
    """Get the singleton OWCInventory instance."""
    global _instance
    if _instance is None:
        _instance = OWCInventory()
    return _instance


async def get_owc_game_counts() -> dict[str, int]:
    """Convenience function to get game counts from OWC drive.

    Returns:
        Dict mapping config_key to game count.
    """
    inventory = get_owc_inventory()
    return await inventory.get_game_counts()
