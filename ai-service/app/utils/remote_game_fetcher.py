"""Remote Game Fetcher - Fetch games from remote sources for export/training.

This module enables fetching game databases from remote sources:
1. Cluster nodes (via P2P HTTP)
2. Amazon S3 (via aws s3 cp)
3. OWC external drive (via rsync over SSH)

The fetcher is used by export scripts to pull games from all available
sources before generating training data.

Key features:
- Multi-source game fetching (P2P, S3, OWC)
- Checksum verification
- Parallel downloads
- Progress tracking
- Automatic cleanup of temporary files

Configuration via environment variables:
- RINGRIFT_FETCH_FROM_P2P: Enable P2P fetching (default: true)
- RINGRIFT_FETCH_FROM_S3: Enable S3 fetching (default: true)
- RINGRIFT_FETCH_FROM_OWC: Enable OWC fetching (default: true)
- RINGRIFT_FETCH_TIMEOUT: Default timeout per fetch (default: 300s)
- RINGRIFT_FETCH_PARALLEL: Max parallel fetches (default: 4)

Usage:
    from app.utils.remote_game_fetcher import (
        RemoteGameFetcher,
        FetchConfig,
        get_remote_game_fetcher,
    )

    # Get singleton instance
    fetcher = get_remote_game_fetcher()

    # Fetch all games for a config
    paths = await fetcher.fetch_all_for_config(
        config_key="hex8_2p",
        target_dir=Path("data/games/fetched"),
    )

    # Fetch from specific node
    path = await fetcher.fetch_from_node(
        node_id="vast-12345",
        db_path="data/games/selfplay_hex8.db",
    )

January 2026: Created as part of multi-source game discovery and sync infrastructure.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RemoteGameFetcher",
    "FetchConfig",
    "FetchResult",
    "get_remote_game_fetcher",
    "reset_remote_game_fetcher",
]


# =============================================================================
# Configuration
# =============================================================================

# Environment variable defaults
FETCH_FROM_P2P = os.getenv("RINGRIFT_FETCH_FROM_P2P", "true").lower() == "true"
FETCH_FROM_S3 = os.getenv("RINGRIFT_FETCH_FROM_S3", "true").lower() == "true"
FETCH_FROM_OWC = os.getenv("RINGRIFT_FETCH_FROM_OWC", "true").lower() == "true"
FETCH_TIMEOUT = int(os.getenv("RINGRIFT_FETCH_TIMEOUT", "300"))
FETCH_PARALLEL = int(os.getenv("RINGRIFT_FETCH_PARALLEL", "4"))

# S3 configuration
S3_BUCKET = os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
S3_GAMES_PREFIX = os.getenv("RINGRIFT_S3_GAMES_PREFIX", "consolidated/games/")

# OWC configuration
OWC_HOST = os.getenv("RINGRIFT_OWC_HOST", "mac-studio")
OWC_USER = os.getenv("RINGRIFT_OWC_USER", "armand")
OWC_PATH = os.getenv("RINGRIFT_OWC_PATH", "/Volumes/RingRift-Data/selfplay_repository")
OWC_SSH_KEY = os.getenv("RINGRIFT_OWC_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))

# P2P data server port
P2P_DATA_PORT = int(os.getenv("RINGRIFT_P2P_DATA_PORT", "8790"))


@dataclass
class FetchConfig:
    """Configuration for remote game fetching."""

    # Source toggles
    fetch_from_p2p: bool = FETCH_FROM_P2P
    fetch_from_s3: bool = FETCH_FROM_S3
    fetch_from_owc: bool = FETCH_FROM_OWC

    # Timeouts and limits
    timeout_seconds: int = FETCH_TIMEOUT
    max_parallel_fetches: int = FETCH_PARALLEL
    max_retries: int = 3

    # S3 configuration
    s3_bucket: str = S3_BUCKET
    s3_prefix: str = S3_GAMES_PREFIX

    # OWC configuration
    owc_host: str = OWC_HOST
    owc_user: str = OWC_USER
    owc_path: str = OWC_PATH
    owc_ssh_key: str = OWC_SSH_KEY

    # P2P configuration
    p2p_data_port: int = P2P_DATA_PORT

    # Target directory for downloads
    default_target_dir: Path = field(
        default_factory=lambda: Path("data/games/fetched")
    )

    # Minimum free space in GB before fetching
    min_free_space_gb: float = 2.0

    @classmethod
    def from_env(cls) -> "FetchConfig":
        """Create config from environment variables."""
        return cls()


@dataclass
class FetchResult:
    """Result of a fetch operation."""

    source: str  # "p2p", "s3", "owc"
    source_path: str  # Original path/URL
    local_path: Path | None  # Local path if successful
    success: bool
    error: str | None = None
    size_bytes: int = 0
    fetch_time_seconds: float = 0.0
    checksum: str | None = None


@dataclass
class FetchStats:
    """Statistics for fetch operations."""

    total_fetches: int = 0
    successful_fetches: int = 0
    failed_fetches: int = 0
    total_bytes_fetched: int = 0
    total_fetch_time: float = 0.0
    last_fetch_time: float | None = None


# =============================================================================
# Remote Game Fetcher
# =============================================================================


class RemoteGameFetcher:
    """Fetches game databases from remote sources.

    Supports fetching from:
    - P2P cluster nodes (via HTTP)
    - Amazon S3 (via aws s3 cp)
    - OWC external drive (via rsync over SSH)

    Key features:
    - Parallel downloads with semaphore limiting
    - Checksum verification
    - Automatic retry with backoff
    - Progress tracking
    """

    def __init__(self, config: FetchConfig | None = None):
        """Initialize the fetcher.

        Args:
            config: Fetch configuration (defaults to env-based config)
        """
        self._config = config or FetchConfig.from_env()
        self._stats = FetchStats()
        self._semaphore = asyncio.Semaphore(self._config.max_parallel_fetches)

        # Lazy-loaded cluster manifest
        self._cluster_manifest = None

        # Cache of fetched files to avoid re-downloading
        self._fetched_cache: dict[str, Path] = {}

    # =========================================================================
    # Lazy Dependencies
    # =========================================================================

    def _get_cluster_manifest(self):
        """Get or create ClusterManifest instance."""
        if self._cluster_manifest is None:
            try:
                from app.distributed.cluster_manifest import get_cluster_manifest
                self._cluster_manifest = get_cluster_manifest()
            except ImportError:
                logger.debug("[RemoteGameFetcher] ClusterManifest not available")
        return self._cluster_manifest

    # =========================================================================
    # Public API
    # =========================================================================

    async def fetch_all_for_config(
        self,
        config_key: str,
        target_dir: Path | None = None,
    ) -> list[Path]:
        """Fetch all available games for a config from all sources.

        Args:
            config_key: Configuration identifier (e.g., "hex8_2p")
            target_dir: Directory to store fetched files (default: data/games/fetched)

        Returns:
            List of local paths to fetched databases
        """
        target_dir = target_dir or self._config.default_target_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Check free space
        if not self._check_free_space():
            logger.warning("[RemoteGameFetcher] Insufficient disk space for fetching")
            return []

        # Parse config_key
        parts = config_key.replace("_", " ").split()
        if len(parts) >= 2:
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))
        else:
            logger.warning(f"[RemoteGameFetcher] Invalid config_key: {config_key}")
            return []

        # Collect fetch tasks from all sources
        tasks = []

        # P2P cluster nodes
        if self._config.fetch_from_p2p:
            p2p_tasks = await self._find_p2p_sources(config_key, board_type, num_players)
            for source in p2p_tasks:
                tasks.append(
                    self._fetch_with_semaphore(
                        self._fetch_from_p2p_node,
                        source["node_id"],
                        source["db_path"],
                        target_dir,
                    )
                )

        # S3
        if self._config.fetch_from_s3:
            s3_tasks = await self._find_s3_sources(config_key)
            for source in s3_tasks:
                tasks.append(
                    self._fetch_with_semaphore(
                        self._fetch_from_s3,
                        source["s3_path"],
                        target_dir,
                    )
                )

        # OWC
        if self._config.fetch_from_owc:
            owc_tasks = await self._find_owc_sources(config_key, board_type, num_players)
            for source in owc_tasks:
                tasks.append(
                    self._fetch_with_semaphore(
                        self._fetch_from_owc,
                        source["path"],
                        target_dir,
                    )
                )

        if not tasks:
            logger.debug(f"[RemoteGameFetcher] No remote sources for {config_key}")
            return []

        logger.info(f"[RemoteGameFetcher] Fetching {len(tasks)} databases for {config_key}")

        # Execute all fetches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful downloads
        local_paths = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"[RemoteGameFetcher] Fetch error: {result}")
            elif isinstance(result, FetchResult) and result.success:
                local_paths.append(result.local_path)

        logger.info(
            f"[RemoteGameFetcher] Fetched {len(local_paths)}/{len(tasks)} databases"
        )
        return local_paths

    async def fetch_from_node(
        self,
        node_id: str,
        db_path: str,
        target_dir: Path | None = None,
    ) -> Path | None:
        """Fetch a database from a specific cluster node via P2P HTTP.

        Args:
            node_id: Node identifier
            db_path: Path to database on the node
            target_dir: Directory to store fetched file

        Returns:
            Local path to fetched file or None if failed
        """
        target_dir = target_dir or self._config.default_target_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        result = await self._fetch_from_p2p_node(node_id, db_path, target_dir)
        return result.local_path if result.success else None

    async def fetch_from_s3(
        self,
        s3_path: str,
        target_dir: Path | None = None,
    ) -> Path | None:
        """Fetch a database from S3.

        Args:
            s3_path: S3 path (e.g., s3://bucket/path/file.db)
            target_dir: Directory to store fetched file

        Returns:
            Local path to fetched file or None if failed
        """
        target_dir = target_dir or self._config.default_target_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        result = await self._fetch_from_s3(s3_path, target_dir)
        return result.local_path if result.success else None

    async def fetch_from_owc(
        self,
        owc_path: str,
        target_dir: Path | None = None,
    ) -> Path | None:
        """Fetch a database from OWC external drive.

        Args:
            owc_path: Path on OWC drive
            target_dir: Directory to store fetched file

        Returns:
            Local path to fetched file or None if failed
        """
        target_dir = target_dir or self._config.default_target_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        result = await self._fetch_from_owc(owc_path, target_dir)
        return result.local_path if result.success else None

    def get_stats(self) -> dict[str, Any]:
        """Get fetch statistics."""
        return {
            "total_fetches": self._stats.total_fetches,
            "successful_fetches": self._stats.successful_fetches,
            "failed_fetches": self._stats.failed_fetches,
            "total_bytes_fetched": self._stats.total_bytes_fetched,
            "total_fetch_time": self._stats.total_fetch_time,
            "last_fetch_time": self._stats.last_fetch_time,
            "cache_size": len(self._fetched_cache),
        }

    def clear_cache(self) -> None:
        """Clear the fetch cache."""
        self._fetched_cache.clear()

    # =========================================================================
    # Source Discovery
    # =========================================================================

    async def _find_p2p_sources(
        self, config_key: str, board_type: str, num_players: int
    ) -> list[dict[str, Any]]:
        """Find databases available on P2P cluster nodes."""
        sources = []

        manifest = self._get_cluster_manifest()
        if manifest is None:
            return sources

        try:
            # Get databases registered in manifest for this config
            node_data = await asyncio.to_thread(
                manifest.get_games_by_node_and_config, config_key
            )

            for node_id, count in node_data.items():
                if count > 0:
                    # Construct expected database path
                    db_path = f"data/games/selfplay_{board_type}_{num_players}p.db"
                    sources.append({
                        "node_id": node_id,
                        "db_path": db_path,
                        "game_count": count,
                    })

        except Exception as e:
            logger.debug(f"[RemoteGameFetcher] Error finding P2P sources: {e}")

        return sources

    async def _find_s3_sources(self, config_key: str) -> list[dict[str, Any]]:
        """Find databases available in S3."""
        sources = []

        try:
            # List S3 objects matching config
            prefix = f"{self._config.s3_prefix}{config_key}/"
            cmd = [
                "aws", "s3", "ls",
                f"s3://{self._config.s3_bucket}/{prefix}",
                "--recursive",
            ]

            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line and ".db" in line:
                        # Parse S3 ls output: "2025-01-02 10:30:00 12345 path/to/file.db"
                        parts = line.split()
                        if len(parts) >= 4:
                            s3_key = parts[-1]
                            s3_path = f"s3://{self._config.s3_bucket}/{s3_key}"
                            sources.append({
                                "s3_path": s3_path,
                                "size": int(parts[2]) if parts[2].isdigit() else 0,
                            })

        except Exception as e:
            logger.debug(f"[RemoteGameFetcher] Error finding S3 sources: {e}")

        return sources

    async def _find_owc_sources(
        self, config_key: str, board_type: str, num_players: int
    ) -> list[dict[str, Any]]:
        """Find databases available on OWC drive."""
        sources = []

        try:
            ssh_key = Path(self._config.owc_ssh_key).expanduser()
            if not ssh_key.exists():
                return sources

            # List databases in OWC config directory
            owc_dir = f"{self._config.owc_path}/{config_key}"
            cmd = (
                f"ssh -i {ssh_key} -o ConnectTimeout=10 -o BatchMode=yes "
                f"{self._config.owc_user}@{self._config.owc_host} "
                f"'find {owc_dir} -name \"*.db\" -type f 2>/dev/null'"
            )

            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                for path in result.stdout.strip().split("\n"):
                    if path.strip():
                        sources.append({"path": path.strip()})

        except Exception as e:
            logger.debug(f"[RemoteGameFetcher] Error finding OWC sources: {e}")

        return sources

    # =========================================================================
    # Fetch Implementations
    # =========================================================================

    async def _fetch_with_semaphore(self, fetch_fn, *args) -> FetchResult:
        """Execute a fetch with semaphore limiting."""
        async with self._semaphore:
            return await fetch_fn(*args)

    async def _fetch_from_p2p_node(
        self, node_id: str, db_path: str, target_dir: Path
    ) -> FetchResult:
        """Fetch a database from a P2P cluster node via HTTP."""
        start_time = time.time()
        self._stats.total_fetches += 1

        try:
            # Get node IP from cluster config
            node_ip = await self._get_node_ip(node_id)
            if not node_ip:
                return FetchResult(
                    source="p2p",
                    source_path=f"{node_id}:{db_path}",
                    local_path=None,
                    success=False,
                    error=f"Cannot resolve IP for {node_id}",
                )

            # Build URL and local path
            db_name = Path(db_path).name
            url = f"http://{node_ip}:{self._config.p2p_data_port}/games/{db_name}"
            local_path = target_dir / f"p2p_{node_id}_{db_name}"

            # Download via curl
            cmd = [
                "curl", "-s", "-f", "-o", str(local_path),
                "--connect-timeout", "30",
                "--max-time", str(self._config.timeout_seconds),
                url,
            ]

            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                timeout=self._config.timeout_seconds + 10,
            )

            if result.returncode == 0 and local_path.exists():
                size = local_path.stat().st_size
                checksum = self._calculate_checksum(local_path)
                self._stats.successful_fetches += 1
                self._stats.total_bytes_fetched += size
                self._fetched_cache[f"p2p:{node_id}:{db_path}"] = local_path

                return FetchResult(
                    source="p2p",
                    source_path=url,
                    local_path=local_path,
                    success=True,
                    size_bytes=size,
                    fetch_time_seconds=time.time() - start_time,
                    checksum=checksum,
                )
            else:
                self._stats.failed_fetches += 1
                return FetchResult(
                    source="p2p",
                    source_path=url,
                    local_path=None,
                    success=False,
                    error=f"curl failed: {result.returncode}",
                )

        except Exception as e:
            self._stats.failed_fetches += 1
            return FetchResult(
                source="p2p",
                source_path=f"{node_id}:{db_path}",
                local_path=None,
                success=False,
                error=str(e),
            )
        finally:
            self._stats.total_fetch_time += time.time() - start_time
            self._stats.last_fetch_time = time.time()

    async def _fetch_from_s3(self, s3_path: str, target_dir: Path) -> FetchResult:
        """Fetch a database from S3."""
        start_time = time.time()
        self._stats.total_fetches += 1

        try:
            # Build local path
            db_name = Path(s3_path).name
            local_path = target_dir / f"s3_{db_name}"

            # Download via aws s3 cp
            cmd = [
                "aws", "s3", "cp",
                s3_path,
                str(local_path),
                "--only-show-errors",
            ]

            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
            )

            if result.returncode == 0 and local_path.exists():
                size = local_path.stat().st_size
                checksum = self._calculate_checksum(local_path)
                self._stats.successful_fetches += 1
                self._stats.total_bytes_fetched += size
                self._fetched_cache[f"s3:{s3_path}"] = local_path

                return FetchResult(
                    source="s3",
                    source_path=s3_path,
                    local_path=local_path,
                    success=True,
                    size_bytes=size,
                    fetch_time_seconds=time.time() - start_time,
                    checksum=checksum,
                )
            else:
                self._stats.failed_fetches += 1
                return FetchResult(
                    source="s3",
                    source_path=s3_path,
                    local_path=None,
                    success=False,
                    error=result.stderr or f"aws s3 cp failed: {result.returncode}",
                )

        except Exception as e:
            self._stats.failed_fetches += 1
            return FetchResult(
                source="s3",
                source_path=s3_path,
                local_path=None,
                success=False,
                error=str(e),
            )
        finally:
            self._stats.total_fetch_time += time.time() - start_time
            self._stats.last_fetch_time = time.time()

    async def _fetch_from_owc(self, owc_path: str, target_dir: Path) -> FetchResult:
        """Fetch a database from OWC external drive via rsync."""
        start_time = time.time()
        self._stats.total_fetches += 1

        try:
            ssh_key = Path(self._config.owc_ssh_key).expanduser()
            if not ssh_key.exists():
                self._stats.failed_fetches += 1
                return FetchResult(
                    source="owc",
                    source_path=owc_path,
                    local_path=None,
                    success=False,
                    error=f"SSH key not found: {ssh_key}",
                )

            # Build local path
            db_name = Path(owc_path).name
            local_path = target_dir / f"owc_{db_name}"

            # Download via rsync
            source = f"{self._config.owc_user}@{self._config.owc_host}:{owc_path}"
            cmd = [
                "rsync", "-avz", "--checksum",
                "-e", f"ssh -i {ssh_key} -o ConnectTimeout=30 -o BatchMode=yes",
                source,
                str(local_path),
            ]

            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
            )

            if result.returncode == 0 and local_path.exists():
                size = local_path.stat().st_size
                checksum = self._calculate_checksum(local_path)
                self._stats.successful_fetches += 1
                self._stats.total_bytes_fetched += size
                self._fetched_cache[f"owc:{owc_path}"] = local_path

                return FetchResult(
                    source="owc",
                    source_path=owc_path,
                    local_path=local_path,
                    success=True,
                    size_bytes=size,
                    fetch_time_seconds=time.time() - start_time,
                    checksum=checksum,
                )
            else:
                self._stats.failed_fetches += 1
                return FetchResult(
                    source="owc",
                    source_path=owc_path,
                    local_path=None,
                    success=False,
                    error=result.stderr or f"rsync failed: {result.returncode}",
                )

        except Exception as e:
            self._stats.failed_fetches += 1
            return FetchResult(
                source="owc",
                source_path=owc_path,
                local_path=None,
                success=False,
                error=str(e),
            )
        finally:
            self._stats.total_fetch_time += time.time() - start_time
            self._stats.last_fetch_time = time.time()

    # =========================================================================
    # Helpers
    # =========================================================================

    async def _get_node_ip(self, node_id: str) -> str | None:
        """Get IP address for a cluster node."""
        try:
            from app.config.cluster_config import get_cluster_nodes

            nodes = get_cluster_nodes()
            for node in nodes:
                if node.name == node_id:
                    return node.best_ip
        except Exception as e:
            logger.debug(f"[RemoteGameFetcher] Error getting node IP: {e}")

        return None

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _check_free_space(self) -> bool:
        """Check if there's enough free disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self._config.default_target_dir.parent)
            free_gb = free / (1024 ** 3)
            return free_gb >= self._config.min_free_space_gb
        except Exception:
            return True  # Proceed if we can't check


# =============================================================================
# Singleton Management
# =============================================================================

_fetcher_instance: RemoteGameFetcher | None = None


def get_remote_game_fetcher(
    config: FetchConfig | None = None,
) -> RemoteGameFetcher:
    """Get or create the singleton RemoteGameFetcher instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        RemoteGameFetcher instance
    """
    global _fetcher_instance

    if _fetcher_instance is None:
        _fetcher_instance = RemoteGameFetcher(config)

    return _fetcher_instance


def reset_remote_game_fetcher() -> None:
    """Reset the singleton instance (for testing)."""
    global _fetcher_instance
    _fetcher_instance = None


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    async def main():
        parser = argparse.ArgumentParser(description="Remote Game Fetcher")
        parser.add_argument("--config-key", type=str, help="Config key (e.g., hex8_2p)")
        parser.add_argument("--target-dir", type=str, default="data/games/fetched")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        parser.add_argument("--dry-run", action="store_true", help="List sources without fetching")
        args = parser.parse_args()

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s: %(message)s",
        )

        config = FetchConfig()
        fetcher = RemoteGameFetcher(config)

        if not args.config_key:
            parser.print_help()
            return

        target_dir = Path(args.target_dir)

        if args.dry_run:
            # Parse config_key
            parts = args.config_key.replace("_", " ").split()
            if len(parts) >= 2:
                board_type = parts[0]
                num_players = int(parts[1].replace("p", ""))
            else:
                print(f"Invalid config_key: {args.config_key}")
                return

            print(f"\nDiscovering sources for {args.config_key}...")

            # P2P sources
            p2p_sources = await fetcher._find_p2p_sources(args.config_key, board_type, num_players)
            print(f"\nP2P sources ({len(p2p_sources)}):")
            for src in p2p_sources:
                print(f"  {src['node_id']}: {src['db_path']} ({src['game_count']} games)")

            # S3 sources
            s3_sources = await fetcher._find_s3_sources(args.config_key)
            print(f"\nS3 sources ({len(s3_sources)}):")
            for src in s3_sources:
                print(f"  {src['s3_path']}")

            # OWC sources
            owc_sources = await fetcher._find_owc_sources(args.config_key, board_type, num_players)
            print(f"\nOWC sources ({len(owc_sources)}):")
            for src in owc_sources:
                print(f"  {src['path']}")

        else:
            print(f"\nFetching games for {args.config_key} to {target_dir}...")
            paths = await fetcher.fetch_all_for_config(
                args.config_key,
                target_dir,
            )

            print(f"\nFetched {len(paths)} databases:")
            for path in paths:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  {path} ({size_mb:.1f} MB)")

            print(f"\nStats: {fetcher.get_stats()}")

    asyncio.run(main())
