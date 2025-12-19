"""Aria2 Transport - High-performance multi-connection download transport.

This module provides aria2-based data synchronization for the unified data sync
service. It uses aria2c for resilient, multi-connection downloads that work well
over unstable network connections.

Key features:
1. Multi-connection parallel downloads (16 connections per server by default)
2. Multi-source downloads when multiple peers have the same file
3. Auto-resume on connection drops
4. Metalink support for efficient multi-source coordination
5. Integration with the node inventory system

Usage:
    transport = Aria2Transport()

    # Sync from multiple sources
    result = await transport.sync_from_sources(
        sources=["http://node1:8766", "http://node2:8766"],
        local_dir=Path("data/games/synced"),
        patterns=["*.db"],
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Import circuit breaker for fault tolerance
try:
    from app.distributed.circuit_breaker import (
        get_operation_breaker,
        get_adaptive_timeout,
        CircuitOpenError,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False

# Maximum total timeout for batch operations (prevent hour-long stalls)
MAX_BATCH_TIMEOUT = 1800  # 30 minutes max for any batch operation
MAX_PER_FILE_TIMEOUT = 120  # 2 minutes per file max


@dataclass
class Aria2Config:
    """Configuration for aria2 transport."""
    connections_per_server: int = 16
    split: int = 16  # Number of parallel segments per file
    min_split_size: str = "1M"  # Minimum size to split
    max_concurrent_downloads: int = 5
    connect_timeout: int = 10
    timeout: int = 300
    retry_wait: int = 3
    max_tries: int = 5
    continue_download: bool = True
    check_integrity: bool = True
    allow_overwrite: bool = True
    # Data server port (matches aria2_data_sync.py)
    data_server_port: int = 8766


@dataclass
class FileInfo:
    """Information about a remote file."""
    name: str
    path: str
    size_bytes: int
    mtime: float
    category: str  # 'games', 'models', 'training', 'elo'
    checksum: Optional[str] = None
    sources: List[str] = field(default_factory=list)


@dataclass
class NodeInventory:
    """Inventory from a remote node."""
    url: str
    hostname: str = ""
    files: Dict[str, FileInfo] = field(default_factory=dict)
    reachable: bool = False
    total_size_mb: float = 0


@dataclass
class SyncResult:
    """Result of an aria2 sync operation."""
    success: bool
    files_synced: int = 0
    files_failed: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    method: str = "aria2"


def check_aria2_available() -> bool:
    """Check if aria2c is available."""
    return shutil.which("aria2c") is not None


class Aria2Transport:
    """High-performance aria2-based data transport."""

    def __init__(self, config: Optional[Aria2Config] = None):
        self.config = config or Aria2Config()
        self._aria2_available: Optional[bool] = None
        self._session: Optional[Any] = None

    def is_available(self) -> bool:
        """Check if aria2 transport is available."""
        if self._aria2_available is None:
            self._aria2_available = check_aria2_available()
        return self._aria2_available

    async def _get_session(self):
        """Get or create aiohttp session for inventory fetches."""
        if self._session is None:
            try:
                import aiohttp
                timeout = aiohttp.ClientTimeout(
                    total=self.config.timeout,
                    connect=self.config.connect_timeout,
                )
                self._session = aiohttp.ClientSession(timeout=timeout)
            except ImportError:
                logger.warning("aiohttp not available, using requests fallback")
                return None
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch_inventory(
        self,
        source_url: str,
        timeout: int = 10,
    ) -> Optional[NodeInventory]:
        """Fetch inventory from a data server node.

        Args:
            source_url: Base URL of the data server (e.g., http://node1:8766)
            timeout: Request timeout in seconds

        Returns:
            NodeInventory if successful, None otherwise
        """
        # Extract host for circuit breaker tracking
        host = source_url.split("://")[-1].split(":")[0].split("/")[0]

        # Circuit breaker check
        if HAS_CIRCUIT_BREAKER:
            breaker = get_operation_breaker("aria2")
            if not breaker.can_execute(host):
                logger.debug(f"Circuit breaker open for {host}, skipping inventory fetch")
                return None
            timeout = int(get_adaptive_timeout("aria2", host, float(timeout)))

        inventory_url = f"{source_url.rstrip('/')}/inventory.json"

        try:
            session = await self._get_session()
            if session:
                async with session.get(inventory_url, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Record success
                        if HAS_CIRCUIT_BREAKER:
                            get_operation_breaker("aria2").record_success(host)
                        return self._parse_inventory(source_url, data)
            else:
                # Fallback to requests
                import requests
                resp = requests.get(inventory_url, timeout=timeout)
                if resp.status_code == 200:
                    # Record success
                    if HAS_CIRCUIT_BREAKER:
                        get_operation_breaker("aria2").record_success(host)
                    return self._parse_inventory(source_url, resp.json())

            # Non-200 response is a failure
            if HAS_CIRCUIT_BREAKER:
                get_operation_breaker("aria2").record_failure(host)

        except Exception as e:
            logger.debug(f"Failed to fetch inventory from {source_url}: {e}")
            # Record failure
            if HAS_CIRCUIT_BREAKER:
                get_operation_breaker("aria2").record_failure(host, e)

        return None

    def _parse_inventory(self, source_url: str, data: Dict) -> NodeInventory:
        """Parse inventory JSON into NodeInventory object."""
        files = {}
        total_size = 0
        base_url = source_url.rstrip("/")

        def add_file(path: str, file_data: Dict[str, Any], category_hint: Optional[str] = None) -> None:
            nonlocal total_size
            if not path:
                return
            path = path.lstrip("/")
            name = file_data.get("name") or Path(path).name
            category = file_data.get("category") or category_hint or "unknown"
            file_info = FileInfo(
                name=name,
                path=path,
                size_bytes=file_data.get("size_bytes", 0),
                mtime=file_data.get("mtime", 0),
                category=category,
                checksum=file_data.get("checksum"),
                sources=[f"{base_url}/{path}"],
            )
            if path not in files:
                files[path] = file_info
                total_size += file_info.size_bytes

        files_map = data.get("files", {}) or {}
        for path, file_data in files_map.items():
            add_file(path, file_data, file_data.get("category"))

        for category in ["games", "models", "training", "elo"]:
            for file_data in data.get(category, []):
                path = file_data.get("path") or f"{category}/{file_data.get('name', '')}"
                add_file(path, file_data, category)

        return NodeInventory(
            url=source_url,
            hostname=data.get("hostname", ""),
            files=files,
            reachable=True,
            total_size_mb=total_size / (1024 * 1024),
        )

    async def discover_sources(
        self,
        source_urls: List[str],
        parallel: int = 10,
    ) -> Tuple[List[NodeInventory], Dict[str, List[str]]]:
        """Discover all available sources and aggregate file information.

        Returns:
            Tuple of (list of NodeInventory, dict mapping filename to list of source URLs)
        """
        # Fetch inventories in parallel
        tasks = [self.fetch_inventory(url) for url in source_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        inventories = []
        file_sources: Dict[str, List[str]] = {}

        for result in results:
            if isinstance(result, NodeInventory) and result.reachable:
                inventories.append(result)
                for file_info in result.files.values():
                    if file_info.name not in file_sources:
                        file_sources[file_info.name] = []
                    file_sources[file_info.name].extend(file_info.sources)

        logger.info(
            f"Discovered {len(inventories)} reachable nodes with {len(file_sources)} unique files"
        )
        return inventories, file_sources

    def _build_aria2_command(
        self,
        output_dir: Path,
        url_file: Optional[Path] = None,
        urls: Optional[List[str]] = None,
    ) -> List[str]:
        """Build aria2c command with optimal settings."""
        cmd = [
            "aria2c",
            f"--max-connection-per-server={self.config.connections_per_server}",
            f"--split={self.config.split}",
            f"--min-split-size={self.config.min_split_size}",
            f"--max-concurrent-downloads={self.config.max_concurrent_downloads}",
            f"--connect-timeout={self.config.connect_timeout}",
            f"--timeout={self.config.timeout}",
            f"--retry-wait={self.config.retry_wait}",
            f"--max-tries={self.config.max_tries}",
            f"--dir={output_dir}",
            "--file-allocation=falloc",
            "--console-log-level=warn",
            "--summary-interval=0",
        ]

        if self.config.continue_download:
            cmd.append("--continue=true")
        if self.config.allow_overwrite:
            cmd.append("--allow-overwrite=true")
        if self.config.check_integrity:
            cmd.append("--check-integrity=true")

        if url_file:
            cmd.append(f"--input-file={url_file}")
        elif urls:
            cmd.extend(urls)

        return cmd

    def _resolve_category_dir(self, local_dir: Path, category: str) -> Path:
        if category == "games" and local_dir.name in ("games", "selfplay"):
            return local_dir
        if category == "models" and local_dir.name == "models":
            return local_dir
        if category == "training" and local_dir.name in ("training", "training_data"):
            return local_dir
        if category == "elo" and local_dir.name == "elo":
            return local_dir
        return local_dir / category

    async def download_file(
        self,
        sources: List[str],
        output_dir: Path,
        filename: Optional[str] = None,
    ) -> Tuple[bool, int, str]:
        """Download a single file from multiple sources using aria2.

        Args:
            sources: List of URLs to download from (aria2 will try all)
            output_dir: Directory to save the file
            filename: Optional filename override

        Returns:
            Tuple of (success, bytes_downloaded, error_message)
        """
        if not self.is_available():
            return False, 0, "aria2c not available"

        if not sources:
            return False, 0, "No sources provided"

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_aria2_command(output_dir, urls=sources)
        if filename:
            cmd.append(f"--out={filename}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout,
            )

            if process.returncode == 0:
                # Get file size
                if filename:
                    filepath = output_dir / filename
                else:
                    # Extract filename from first URL
                    filepath = output_dir / Path(sources[0]).name

                size = filepath.stat().st_size if filepath.exists() else 0
                return True, size, ""
            else:
                error = stderr.decode()[:200] if stderr else "Unknown error"
                return False, 0, error

        except asyncio.TimeoutError:
            return False, 0, "Download timeout"
        except Exception as e:
            return False, 0, str(e)[:200]

    async def download_batch(
        self,
        file_sources: Dict[str, List[str]],
        output_dir: Path,
        category: Optional[str] = "games",
    ) -> SyncResult:
        """Download multiple files using aria2 with a URL list file.

        Args:
            file_sources: Dict mapping filenames to list of source URLs
            output_dir: Directory to save files
            category: Category subdirectory (games, models, training, elo)

        Returns:
            SyncResult with download statistics
        """
        if not self.is_available():
            return SyncResult(
                success=False,
                errors=["aria2c not available"],
            )

        if not file_sources:
            return SyncResult(success=True)

        start_time = time.time()
        category_dir = output_dir if not category else output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Create URL list file for aria2
        # Format: URL\n  out=filename\n
        url_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                url_file = Path(f.name)
                for filename, sources in file_sources.items():
                    # Write all sources for this file (aria2 will try them in order)
                    for source in sources:
                        f.write(f"{source}\n")
                    f.write(f"  out={filename}\n")

            cmd = self._build_aria2_command(category_dir, url_file=url_file)

            # Calculate reasonable timeout: base timeout + per-file overhead
            # Cap at MAX_BATCH_TIMEOUT to prevent hour-long stalls
            per_file_timeout = min(self.config.timeout, MAX_PER_FILE_TIMEOUT)
            total_timeout = min(
                self.config.timeout + (per_file_timeout * min(len(file_sources), 50)),
                MAX_BATCH_TIMEOUT
            )

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=total_timeout,
            )

            # Count successful downloads
            files_synced = 0
            files_failed = 0
            bytes_transferred = 0
            errors = []

            for filename in file_sources:
                filepath = category_dir / filename
                if filepath.exists():
                    files_synced += 1
                    bytes_transferred += filepath.stat().st_size
                else:
                    files_failed += 1
                    errors.append(f"Failed to download: {filename}")

            if stderr and files_failed > 0:
                errors.append(stderr.decode()[:500])

            return SyncResult(
                success=files_failed == 0,
                files_synced=files_synced,
                files_failed=files_failed,
                bytes_transferred=bytes_transferred,
                duration_seconds=time.time() - start_time,
                errors=errors,
                method="aria2",
            )

        except asyncio.TimeoutError:
            return SyncResult(
                success=False,
                duration_seconds=time.time() - start_time,
                errors=["Batch download timeout"],
                method="aria2",
            )
        except Exception as e:
            return SyncResult(
                success=False,
                duration_seconds=time.time() - start_time,
                errors=[str(e)],
                method="aria2",
            )
        finally:
            if url_file and url_file.exists():
                url_file.unlink()

    async def sync_games(
        self,
        source_urls: List[str],
        local_dir: Path,
        max_age_hours: float = 168,  # 1 week
        dry_run: bool = False,
    ) -> SyncResult:
        """Sync game databases from multiple sources.

        Args:
            source_urls: List of data server URLs
            local_dir: Local directory to sync to
            max_age_hours: Only sync files newer than this
            dry_run: If True, just report what would be synced

        Returns:
            SyncResult with sync statistics
        """
        start_time = time.time()

        # Discover all sources
        inventories, file_sources = await self.discover_sources(source_urls)

        if not inventories:
            return SyncResult(
                success=False,
                errors=["No reachable sources found"],
                duration_seconds=time.time() - start_time,
            )

        # Filter to games category and by age
        cutoff_time = time.time() - (max_age_hours * 3600)
        games_to_sync: Dict[str, List[str]] = {}

        category_dir = self._resolve_category_dir(local_dir, "games")

        for inventory in inventories:
            for file_info in inventory.files.values():
                if file_info.category != "games":
                    continue
                if file_info.mtime < cutoff_time:
                    continue

                # Check if we already have this file
                local_path = category_dir / file_info.name
                if local_path.exists():
                    local_mtime = local_path.stat().st_mtime
                    if local_mtime >= file_info.mtime:
                        continue

                if file_info.name not in games_to_sync:
                    games_to_sync[file_info.name] = []
                games_to_sync[file_info.name].extend(file_info.sources)

        if not games_to_sync:
            return SyncResult(
                success=True,
                duration_seconds=time.time() - start_time,
            )

        logger.info(f"Found {len(games_to_sync)} game files to sync")

        if dry_run:
            return SyncResult(
                success=True,
                files_synced=0,
                files_failed=len(games_to_sync),  # Reported as "would sync"
                duration_seconds=time.time() - start_time,
            )

        return await self.download_batch(games_to_sync, category_dir, None)

    async def sync_models(
        self,
        source_urls: List[str],
        local_dir: Path,
        dry_run: bool = False,
    ) -> SyncResult:
        """Sync model checkpoints from multiple sources."""
        start_time = time.time()

        inventories, file_sources = await self.discover_sources(source_urls)

        if not inventories:
            return SyncResult(
                success=False,
                errors=["No reachable sources found"],
                duration_seconds=time.time() - start_time,
            )

        models_to_sync: Dict[str, List[str]] = {}

        category_dir = self._resolve_category_dir(local_dir, "models")

        for inventory in inventories:
            for file_info in inventory.files.values():
                if file_info.category != "models":
                    continue

                local_path = category_dir / file_info.name
                if local_path.exists():
                    # Check if remote is newer or different size
                    local_stat = local_path.stat()
                    if local_stat.st_mtime >= file_info.mtime:
                        continue

                if file_info.name not in models_to_sync:
                    models_to_sync[file_info.name] = []
                models_to_sync[file_info.name].extend(file_info.sources)

        if not models_to_sync:
            return SyncResult(
                success=True,
                duration_seconds=time.time() - start_time,
            )

        logger.info(f"Found {len(models_to_sync)} model files to sync")

        if dry_run:
            return SyncResult(
                success=True,
                files_failed=len(models_to_sync),
                duration_seconds=time.time() - start_time,
            )

        return await self.download_batch(models_to_sync, category_dir, None)

    async def sync_training_data(
        self,
        source_urls: List[str],
        local_dir: Path,
        max_age_hours: float = 24,
        dry_run: bool = False,
    ) -> SyncResult:
        """Sync training data batches from multiple sources."""
        start_time = time.time()

        inventories, file_sources = await self.discover_sources(source_urls)

        if not inventories:
            return SyncResult(
                success=False,
                errors=["No reachable sources found"],
                duration_seconds=time.time() - start_time,
            )

        cutoff_time = time.time() - (max_age_hours * 3600)
        training_to_sync: Dict[str, List[str]] = {}

        category_dir = self._resolve_category_dir(local_dir, "training")

        for inventory in inventories:
            for file_info in inventory.files.values():
                if file_info.category != "training":
                    continue
                if file_info.mtime < cutoff_time:
                    continue

                local_path = category_dir / file_info.name
                if local_path.exists():
                    continue

                if file_info.name not in training_to_sync:
                    training_to_sync[file_info.name] = []
                training_to_sync[file_info.name].extend(file_info.sources)

        if not training_to_sync:
            return SyncResult(
                success=True,
                duration_seconds=time.time() - start_time,
            )

        logger.info(f"Found {len(training_to_sync)} training files to sync")

        if dry_run:
            return SyncResult(
                success=True,
                files_failed=len(training_to_sync),
                duration_seconds=time.time() - start_time,
            )

        return await self.download_batch(training_to_sync, category_dir, None)

    async def full_cluster_sync(
        self,
        source_urls: List[str],
        local_dir: Path,
        categories: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, SyncResult]:
        """Sync all data categories from cluster.

        Args:
            source_urls: List of data server URLs
            local_dir: Local base directory
            categories: Categories to sync (default: all)
            dry_run: If True, just report what would be synced

        Returns:
            Dict mapping category to SyncResult
        """
        if categories is None:
            categories = ["games", "models", "training"]

        results = {}

        if "games" in categories:
            results["games"] = await self.sync_games(
                source_urls, local_dir, dry_run=dry_run
            )

        if "models" in categories:
            results["models"] = await self.sync_models(
                source_urls, local_dir, dry_run=dry_run
            )

        if "training" in categories:
            results["training"] = await self.sync_training_data(
                source_urls, local_dir, dry_run=dry_run
            )

        return results


# Factory function for integration with unified_data_sync
def create_aria2_transport(config: Optional[Dict[str, Any]] = None) -> Aria2Transport:
    """Create an Aria2Transport instance from config dict."""
    if config:
        aria2_config = Aria2Config(
            connections_per_server=config.get("connections_per_server", 16),
            split=config.get("split", 16),
            max_concurrent_downloads=config.get("max_concurrent_downloads", 5),
            timeout=config.get("timeout", 300),
            data_server_port=config.get("data_server_port", 8766),
        )
        return Aria2Transport(aria2_config)
    return Aria2Transport()
