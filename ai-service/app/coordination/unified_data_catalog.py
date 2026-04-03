"""Unified Data Catalog - Single API for querying data across all sources (January 2026).

This module provides a unified interface for discovering and fetching training data
from all available sources:
- LOCAL: Local filesystem databases and NPZ files
- S3: AWS S3 bucket backup
- OWC: OWC external drive on mac-studio
- P2P: Other cluster nodes via P2P network

Key features:
- Content-addressed deduplication via SHA256 checksums
- Source preference ordering (latency-based)
- Lazy fetching with caching
- Background inventory refresh
- HTTP API endpoint for cluster nodes

Usage:
    from app.coordination.unified_data_catalog import (
        UnifiedDataCatalog,
        get_unified_catalog,
        DataSource,
    )

    catalog = get_unified_catalog()
    await catalog.refresh_inventory()

    # Find best source for training data
    location = await catalog.find_best_source("hex8_2p")

    # Get data, downloading if needed
    path = await catalog.get_or_fetch("hex8_2p")

Environment Variables:
    RINGRIFT_CATALOG_REFRESH_INTERVAL: Inventory refresh interval (default: 3600)
    RINGRIFT_CATALOG_HTTP_PORT: HTTP API port (default: 8771)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.protocols import CoordinatorStatus

logger = logging.getLogger(__name__)


class DataSource(str, Enum):
    """Data source types."""

    LOCAL = "local"
    S3 = "s3"
    OWC = "owc"
    P2P = "p2p"


class DataType(str, Enum):
    """Data type categories."""

    CANONICAL_DB = "canonical_db"
    TRAINING_NPZ = "training_npz"
    MODEL = "model"


@dataclass
class DataLocation:
    """Information about a data file location."""

    source: DataSource
    path: str  # Local path or S3 key or remote path
    size_bytes: int
    mtime: float
    checksum: str  # SHA256 for deduplication
    data_type: DataType
    config_key: str | None = None  # e.g., "hex8_2p"


@dataclass
class CatalogConfig:
    """Configuration for Unified Data Catalog."""

    refresh_interval: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_CATALOG_REFRESH_INTERVAL", "3600")
        )
    )
    http_port: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_CATALOG_HTTP_PORT", "8771")
        )
    )
    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_CATALOG_ENABLED", "true"
        ).lower() == "true"
    )

    # Source preference (lower number = preferred)
    source_preference: dict[DataSource, int] = field(default_factory=lambda: {
        DataSource.LOCAL: 0,
        DataSource.P2P: 1,
        DataSource.S3: 2,
        DataSource.OWC: 3,
    })

    # Local paths
    local_games_dir: str = "data/games"
    local_training_dir: str = "data/training"
    local_models_dir: str = "models"

    # OWC settings
    owc_host: str = field(
        default_factory=lambda: os.environ.get("OWC_HOST", "mac-studio")
    )
    owc_base_path: str = field(
        default_factory=lambda: os.environ.get(
            "OWC_BASE_PATH", "/Volumes/RingRift-Data"
        )
    )

    # S3 settings
    s3_bucket: str = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_S3_BUCKET", "ringrift-models-20251214"
        )
    )


@dataclass
class CatalogStats:
    """Statistics for catalog operations."""

    total_locations: int = 0
    local_files: int = 0
    s3_files: int = 0
    owc_files: int = 0
    p2p_files: int = 0
    last_refresh_time: float = 0.0
    fetch_requests: int = 0
    cache_hits: int = 0


class UnifiedDataCatalog(HandlerBase):
    """Unified catalog for discovering and fetching data across all sources.

    Provides a single API to query data availability across LOCAL, S3, OWC,
    and P2P sources, with automatic source selection and deduplication.
    """

    _instance: UnifiedDataCatalog | None = None

    def __init__(self, config: CatalogConfig | None = None):
        """Initialize unified data catalog.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        resolved_config = config or CatalogConfig()
        super().__init__(
            name="unified_catalog",
            config=resolved_config,
            cycle_interval=resolved_config.refresh_interval,
        )

        self._catalog_stats = CatalogStats()
        self._base_path = Path(os.environ.get("RINGRIFT_BASE_PATH", "."))

        # Inventory: config_key -> list of DataLocation
        self._inventory: dict[str, list[DataLocation]] = {}

        # Checksum index: checksum -> list of DataLocation (for dedup)
        self._checksum_index: dict[str, list[DataLocation]] = {}

        # Cache of recently fetched files
        self._fetch_cache: dict[str, tuple[Path, float]] = {}  # key -> (path, timestamp)

        # Source availability
        self._source_available: dict[DataSource, bool] = {
            DataSource.LOCAL: True,
            DataSource.S3: False,
            DataSource.OWC: False,
            DataSource.P2P: False,
        }

    @classmethod
    def get_instance(cls) -> UnifiedDataCatalog:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_config_key(self, filename: str) -> str | None:
        """Extract config key from filename.

        Examples:
            canonical_hex8_2p.db -> hex8_2p
            hex8_2p.npz -> hex8_2p
            canonical_hex8_2p.pth -> hex8_2p
        """
        # Remove common prefixes
        name = filename.replace("canonical_", "").replace("ringrift_best_", "")

        # Remove extension
        for ext in [".db", ".npz", ".pth", ".pt"]:
            if name.endswith(ext):
                name = name[:-len(ext)]
                break

        # Validate it looks like a config key (board_Np)
        if "_" in name and name[-1] == "p" and name[-2].isdigit():
            return name

        return None

    def _determine_data_type(self, filename: str) -> DataType:
        """Determine data type from filename."""
        if filename.endswith(".db"):
            return DataType.CANONICAL_DB
        elif filename.endswith(".npz"):
            return DataType.TRAINING_NPZ
        elif filename.endswith((".pth", ".pt")):
            return DataType.MODEL
        return DataType.CANONICAL_DB  # Default

    async def _scan_local(self) -> list[DataLocation]:
        """Scan local filesystem for data files."""
        locations = []

        # Scan canonical databases
        games_dir = self._base_path / self.config.local_games_dir
        if games_dir.exists():
            for db_path in games_dir.glob("canonical_*.db"):
                try:
                    stat = db_path.stat()
                    checksum = await asyncio.to_thread(self._compute_checksum, db_path)
                    config_key = self._extract_config_key(db_path.name)

                    locations.append(DataLocation(
                        source=DataSource.LOCAL,
                        path=str(db_path),
                        size_bytes=stat.st_size,
                        mtime=stat.st_mtime,
                        checksum=checksum,
                        data_type=DataType.CANONICAL_DB,
                        config_key=config_key,
                    ))
                except (OSError, IOError) as e:
                    logger.debug(f"[Catalog] Error scanning {db_path}: {e}")

        # Scan NPZ training files
        training_dir = self._base_path / self.config.local_training_dir
        if training_dir.exists():
            for npz_path in training_dir.glob("*.npz"):
                try:
                    stat = npz_path.stat()
                    checksum = await asyncio.to_thread(self._compute_checksum, npz_path)
                    config_key = self._extract_config_key(npz_path.name)

                    locations.append(DataLocation(
                        source=DataSource.LOCAL,
                        path=str(npz_path),
                        size_bytes=stat.st_size,
                        mtime=stat.st_mtime,
                        checksum=checksum,
                        data_type=DataType.TRAINING_NPZ,
                        config_key=config_key,
                    ))
                except (OSError, IOError) as e:
                    logger.debug(f"[Catalog] Error scanning {npz_path}: {e}")

        # Scan models
        models_dir = self._base_path / self.config.local_models_dir
        if models_dir.exists():
            for model_path in models_dir.glob("canonical_*.pth"):
                try:
                    stat = model_path.stat()
                    checksum = await asyncio.to_thread(self._compute_checksum, model_path)
                    config_key = self._extract_config_key(model_path.name)

                    locations.append(DataLocation(
                        source=DataSource.LOCAL,
                        path=str(model_path),
                        size_bytes=stat.st_size,
                        mtime=stat.st_mtime,
                        checksum=checksum,
                        data_type=DataType.MODEL,
                        config_key=config_key,
                    ))
                except (OSError, IOError) as e:
                    logger.debug(f"[Catalog] Error scanning {model_path}: {e}")

        self._catalog_stats.local_files = len(locations)
        self._source_available[DataSource.LOCAL] = True

        return locations

    async def _scan_s3(self) -> list[DataLocation]:
        """Scan S3 bucket for data files."""
        try:
            from app.coordination.s3_import_daemon import get_s3_import_daemon

            s3_daemon = get_s3_import_daemon()
            await s3_daemon.refresh_inventory()

            locations = []
            inventory = s3_daemon.get_inventory()

            for s3_key, info in inventory.items():
                filename = os.path.basename(s3_key)
                config_key = self._extract_config_key(filename)
                data_type = self._determine_data_type(filename)

                locations.append(DataLocation(
                    source=DataSource.S3,
                    path=s3_key,
                    size_bytes=info["size"],
                    mtime=0,  # S3 doesn't have mtime in same way
                    checksum=info["etag"],  # Use ETag as pseudo-checksum
                    data_type=data_type,
                    config_key=config_key,
                ))

            self._catalog_stats.s3_files = len(locations)
            self._source_available[DataSource.S3] = len(locations) > 0

            return locations

        except Exception as e:
            logger.debug(f"[Catalog] S3 scan error: {e}")
            self._source_available[DataSource.S3] = False
            return []

    async def _scan_owc(self) -> list[DataLocation]:
        """Scan OWC drive for data files."""
        try:
            from app.coordination.owc_import_daemon import get_owc_import_daemon

            owc_daemon = get_owc_import_daemon()

            # Check if OWC is available via the daemon
            if not await owc_daemon._check_owc_available():
                self._source_available[DataSource.OWC] = False
                return []

            # Use SSH to list files on OWC
            import subprocess

            locations = []
            base_path = self.config.owc_base_path

            # List canonical databases
            for subdir, data_type in [
                ("consolidated/games", DataType.CANONICAL_DB),
                ("consolidated/training", DataType.TRAINING_NPZ),
                ("models", DataType.MODEL),
            ]:
                full_path = f"{base_path}/{subdir}"
                pattern = "*.db" if data_type == DataType.CANONICAL_DB else (
                    "*.npz" if data_type == DataType.TRAINING_NPZ else "*.pth"
                )

                result = await asyncio.to_thread(
                    subprocess.run,
                    [
                        "ssh",
                        "-i", os.path.expanduser("~/.ssh/id_ed25519"),
                        "-o", "ConnectTimeout=30",
                        f"armand@{self.config.owc_host}",
                        f"ls -la {full_path}/{pattern} 2>/dev/null | awk '{{print $5, $9}}'",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                size = int(parts[0])
                                path = parts[1]
                                filename = os.path.basename(path)
                                config_key = self._extract_config_key(filename)

                                locations.append(DataLocation(
                                    source=DataSource.OWC,
                                    path=path,
                                    size_bytes=size,
                                    mtime=0,
                                    checksum="",  # Would need to compute remotely
                                    data_type=data_type,
                                    config_key=config_key,
                                ))
                            except (ValueError, IndexError):
                                continue

            self._catalog_stats.owc_files = len(locations)
            self._source_available[DataSource.OWC] = len(locations) > 0

            return locations

        except Exception as e:
            logger.debug(f"[Catalog] OWC scan error: {e}")
            self._source_available[DataSource.OWC] = False
            return []

    async def _scan_p2p(self) -> list[DataLocation]:
        """Scan P2P peers for data files."""
        try:
            from app.distributed.data_catalog import get_data_catalog

            p2p_catalog = get_data_catalog()
            sources = p2p_catalog.discover_data_sources()

            locations = []
            for source in sources:
                filename = os.path.basename(source.get("path", ""))
                config_key = self._extract_config_key(filename)
                data_type = self._determine_data_type(filename)

                locations.append(DataLocation(
                    source=DataSource.P2P,
                    path=source.get("path", ""),
                    size_bytes=source.get("size", 0),
                    mtime=source.get("mtime", 0),
                    checksum=source.get("checksum", ""),
                    data_type=data_type,
                    config_key=config_key,
                ))

            self._catalog_stats.p2p_files = len(locations)
            self._source_available[DataSource.P2P] = len(locations) > 0

            return locations

        except Exception as e:
            logger.debug(f"[Catalog] P2P scan error: {e}")
            self._source_available[DataSource.P2P] = False
            return []

    async def refresh_inventory(self) -> None:
        """Refresh inventory from all sources."""
        self._inventory.clear()
        self._checksum_index.clear()

        # Scan all sources in parallel
        results = await asyncio.gather(
            self._scan_local(),
            self._scan_s3(),
            self._scan_owc(),
            self._scan_p2p(),
            return_exceptions=True,
        )

        all_locations: list[DataLocation] = []
        for result in results:
            if isinstance(result, list):
                all_locations.extend(result)
            elif isinstance(result, Exception):
                logger.debug(f"[Catalog] Scan error: {result}")

        # Build inventory index
        for loc in all_locations:
            if loc.config_key:
                if loc.config_key not in self._inventory:
                    self._inventory[loc.config_key] = []
                self._inventory[loc.config_key].append(loc)

            # Build checksum index for deduplication
            if loc.checksum:
                if loc.checksum not in self._checksum_index:
                    self._checksum_index[loc.checksum] = []
                self._checksum_index[loc.checksum].append(loc)

        self._catalog_stats.total_locations = len(all_locations)
        self._catalog_stats.last_refresh_time = time.time()

        logger.info(
            f"[Catalog] Inventory refreshed: {len(all_locations)} locations, "
            f"{len(self._inventory)} configs"
        )

    def find_all_locations(self, config_key: str) -> list[DataLocation]:
        """Find all locations for a given config key.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")

        Returns:
            List of all DataLocation objects for this config
        """
        return self._inventory.get(config_key, [])

    async def find_best_source(
        self,
        config_key: str,
        data_type: DataType | None = None,
        prefer_source: DataSource | None = None,
    ) -> DataLocation | None:
        """Find the best source for a given config key.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            data_type: Optional filter by data type
            prefer_source: Optional preferred source

        Returns:
            Best DataLocation or None if not found
        """
        locations = self.find_all_locations(config_key)

        if not locations:
            return None

        # Filter by data type if specified
        if data_type:
            locations = [loc for loc in locations if loc.data_type == data_type]

        if not locations:
            return None

        # Filter by source availability
        locations = [
            loc for loc in locations
            if self._source_available.get(loc.source, False)
        ]

        if not locations:
            return None

        # Sort by preference
        def sort_key(loc: DataLocation) -> tuple[int, int]:
            # Prefer specified source first
            if prefer_source and loc.source == prefer_source:
                return (0, 0)
            # Then by configured preference
            pref = self.config.source_preference.get(loc.source, 999)
            # Then by size (larger = more data = better)
            return (1, pref, -loc.size_bytes)

        locations.sort(key=sort_key)

        return locations[0]

    async def get_or_fetch(
        self,
        config_key: str,
        data_type: DataType | None = None,
        prefer_source: DataSource | None = None,
    ) -> Path | None:
        """Get local path for data, fetching from remote if needed.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            data_type: Optional filter by data type
            prefer_source: Optional preferred source

        Returns:
            Local Path to data file, or None if unavailable
        """
        self._catalog_stats.fetch_requests += 1

        # Check cache first
        cache_key = f"{config_key}:{data_type}"
        if cache_key in self._fetch_cache:
            cached_path, cache_time = self._fetch_cache[cache_key]
            if cached_path.exists() and (time.time() - cache_time) < 3600:
                self._catalog_stats.cache_hits += 1
                return cached_path

        # Find best source
        location = await self.find_best_source(config_key, data_type, prefer_source)

        if not location:
            logger.warning(f"[Catalog] No source found for {config_key}")
            return None

        # If local, just return the path
        if location.source == DataSource.LOCAL:
            path = Path(location.path)
            if path.exists():
                self._fetch_cache[cache_key] = (path, time.time())
                return path

        # Otherwise, fetch from remote source
        local_path = await self._fetch_from_source(location)

        if local_path and local_path.exists():
            self._fetch_cache[cache_key] = (local_path, time.time())
            return local_path

        return None

    async def _fetch_from_source(self, location: DataLocation) -> Path | None:
        """Fetch data from a remote source.

        Args:
            location: DataLocation to fetch from

        Returns:
            Local path to fetched file, or None on failure
        """
        if location.source == DataSource.S3:
            return await self._fetch_from_s3(location)
        elif location.source == DataSource.OWC:
            return await self._fetch_from_owc(location)
        elif location.source == DataSource.P2P:
            return await self._fetch_from_p2p(location)

        return None

    async def _fetch_from_s3(self, location: DataLocation) -> Path | None:
        """Fetch from S3."""
        try:
            from app.coordination.s3_import_daemon import get_s3_import_daemon

            s3_daemon = get_s3_import_daemon()

            # Determine local path based on data type
            if location.data_type == DataType.CANONICAL_DB:
                local_dir = self._base_path / self.config.local_games_dir
            elif location.data_type == DataType.TRAINING_NPZ:
                local_dir = self._base_path / self.config.local_training_dir
            else:
                local_dir = self._base_path / self.config.local_models_dir

            local_path = local_dir / os.path.basename(location.path)

            if await s3_daemon._download_file(location.path, local_path):
                return local_path

        except Exception as e:
            logger.warning(f"[Catalog] S3 fetch error: {e}")

        return None

    async def _fetch_from_owc(self, location: DataLocation) -> Path | None:
        """Fetch from OWC drive."""
        try:
            from app.coordination.owc_import_daemon import get_owc_import_daemon

            owc_daemon = get_owc_import_daemon()

            # Determine local path
            if location.data_type == DataType.CANONICAL_DB:
                local_dir = self._base_path / self.config.local_games_dir
            elif location.data_type == DataType.TRAINING_NPZ:
                local_dir = self._base_path / self.config.local_training_dir
            else:
                local_dir = self._base_path / self.config.local_models_dir

            filename = os.path.basename(location.path)
            local_path = local_dir / filename

            # Use OWC daemon's sync method
            rel_path = location.path.replace(self.config.owc_base_path + "/", "")
            synced = await owc_daemon._sync_database(rel_path)

            if synced and synced.exists():
                # Move from staging to proper location
                import shutil
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(synced), str(local_path))
                return local_path

        except Exception as e:
            logger.warning(f"[Catalog] OWC fetch error: {e}")

        return None

    async def _fetch_from_p2p(self, location: DataLocation) -> Path | None:
        """Fetch from P2P peer."""
        try:
            from app.coordination.auto_sync_daemon import get_auto_sync_daemon

            sync_daemon = get_auto_sync_daemon()

            # Trigger sync for this specific file
            # This is a simplified approach - real P2P sync is more complex
            logger.info(f"[Catalog] P2P fetch not implemented for {location.path}")
            return None

        except Exception as e:
            logger.warning(f"[Catalog] P2P fetch error: {e}")

        return None

    def get_configs_with_data(
        self,
        data_type: DataType | None = None,
        source: DataSource | None = None,
    ) -> list[str]:
        """Get list of config keys that have data available.

        Args:
            data_type: Optional filter by data type
            source: Optional filter by source

        Returns:
            List of config keys
        """
        configs = set()

        for config_key, locations in self._inventory.items():
            for loc in locations:
                if data_type and loc.data_type != data_type:
                    continue
                if source and loc.source != source:
                    continue
                configs.add(config_key)

        return sorted(configs)

    def check_data_exists(
        self,
        config_key: str,
        data_type: DataType | None = None,
    ) -> dict[DataSource, bool]:
        """Check which sources have data for a config.

        Args:
            config_key: Configuration key
            data_type: Optional filter by data type

        Returns:
            Dict mapping source to availability
        """
        result = {source: False for source in DataSource}

        for loc in self.find_all_locations(config_key):
            if data_type and loc.data_type != data_type:
                continue
            result[loc.source] = True

        return result

    async def _run_cycle(self) -> None:
        """Run one catalog refresh cycle."""
        if not self.config.enabled:
            logger.debug("[Catalog] Disabled via config, skipping cycle")
            return

        await self.refresh_inventory()

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Get event subscriptions for this daemon."""
        return {
            "DATA_SYNC_COMPLETED": self._on_data_changed,
            "TRAINING_COMPLETED": self._on_data_changed,
            "CONSOLIDATION_COMPLETE": self._on_data_changed,
        }

    async def _on_data_changed(self, event: dict[str, Any]) -> None:
        """Handle data change events - trigger inventory refresh."""
        # Debounce refreshes
        if time.time() - self._catalog_stats.last_refresh_time < 60:
            return

        await self.refresh_inventory()

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="Unified catalog not running",
            )

        available_sources = [
            src.value for src, avail in self._source_available.items() if avail
        ]

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"Catalog active: {len(self._inventory)} configs, "
                   f"{len(available_sources)} sources",
            details={
                "cycles_completed": self._stats.cycles_completed,
                "total_locations": self._catalog_stats.total_locations,
                "configs_tracked": len(self._inventory),
                "local_files": self._catalog_stats.local_files,
                "s3_files": self._catalog_stats.s3_files,
                "owc_files": self._catalog_stats.owc_files,
                "p2p_files": self._catalog_stats.p2p_files,
                "available_sources": available_sources,
                "fetch_requests": self._catalog_stats.fetch_requests,
                "cache_hits": self._catalog_stats.cache_hits,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get current catalog statistics."""
        return {
            "total_locations": self._catalog_stats.total_locations,
            "configs_tracked": len(self._inventory),
            "local_files": self._catalog_stats.local_files,
            "s3_files": self._catalog_stats.s3_files,
            "owc_files": self._catalog_stats.owc_files,
            "p2p_files": self._catalog_stats.p2p_files,
            "last_refresh_time": self._catalog_stats.last_refresh_time,
            "fetch_requests": self._catalog_stats.fetch_requests,
            "cache_hits": self._catalog_stats.cache_hits,
            "source_available": {
                src.value: avail for src, avail in self._source_available.items()
            },
        }

    def get_inventory_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary of inventory by config."""
        summary = {}

        for config_key, locations in self._inventory.items():
            sources = {}
            for loc in locations:
                if loc.source.value not in sources:
                    sources[loc.source.value] = []
                sources[loc.source.value].append({
                    "path": loc.path,
                    "size_mb": round(loc.size_bytes / (1024 * 1024), 2),
                    "type": loc.data_type.value,
                })

            summary[config_key] = {
                "sources": sources,
                "total_locations": len(locations),
            }

        return summary


def get_unified_catalog() -> UnifiedDataCatalog:
    """Get the singleton unified data catalog instance."""
    return UnifiedDataCatalog.get_instance()


def reset_unified_catalog() -> None:
    """Reset the singleton instance (for testing)."""
    UnifiedDataCatalog.reset_instance()


# Factory function for daemon_runners.py
async def create_unified_catalog() -> None:
    """Create and run unified data catalog daemon (January 2026).

    Provides unified discovery and fetching of training data from
    all available sources (LOCAL, S3, OWC, P2P).
    """
    catalog = get_unified_catalog()
    await catalog.start()

    try:
        while catalog._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        await catalog.stop()
