"""Unified resolver for training data across all sources.

January 2026: Created as part of unified data synchronization plan.
Provides a single API for training to fetch data from the best available source.

Resolution order:
1. LOCAL - Check local filesystem first (fastest)
2. P2P - Query ClusterManifest for other nodes (requires network)
3. S3 - Download from S3 bucket (reliable, may be slow)
4. OWC - Download from OWC external drive (backup, mac-studio only)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Use DataSource from cluster_manifest (has all 4 sources)
from app.coordination.event_utils import parse_config_key
from app.distributed.cluster_manifest import (
    ClusterManifest,
    DataSource,
    get_cluster_manifest,
)
from app.utils.game_discovery import GameDiscovery

logger = logging.getLogger(__name__)


class ResolutionStatus(str, Enum):
    """Status of data resolution attempt."""

    FOUND_LOCAL = "found_local"
    FOUND_P2P = "found_p2p"
    DOWNLOADED_S3 = "downloaded_s3"
    DOWNLOADED_OWC = "downloaded_owc"
    NOT_FOUND = "not_found"
    DOWNLOAD_FAILED = "download_failed"


@dataclass
class ResolvedDataResult:
    """Result of resolving training data."""

    config_key: str
    status: ResolutionStatus
    path: Path | None = None
    source: DataSource | None = None
    game_count: int = 0
    file_size_mb: float = 0.0
    download_time_seconds: float = 0.0
    node_id: str | None = None  # For P2P sources
    error: str | None = None

    @property
    def success(self) -> bool:
        """Return True if data was found/downloaded successfully."""
        return self.status in (
            ResolutionStatus.FOUND_LOCAL,
            ResolutionStatus.FOUND_P2P,
            ResolutionStatus.DOWNLOADED_S3,
            ResolutionStatus.DOWNLOADED_OWC,
        )


@dataclass
class ResolverConfig:
    """Configuration for TrainingDataResolver."""

    # Resolution preferences
    prefer_source: DataSource | None = None
    min_size_mb: float = 10.0
    max_age_hours: float = 24.0

    # Download settings
    download_dir: Path | None = None
    s3_bucket: str = "ringrift-models-20251214"
    owc_host: str = "mac-studio"
    owc_base_path: str = "/Volumes/RingRift-Data"

    # Timeouts
    p2p_sync_timeout: float = 300.0  # 5 minutes
    s3_download_timeout: float = 600.0  # 10 minutes
    owc_download_timeout: float = 600.0  # 10 minutes

    # Fallback behavior
    try_all_sources: bool = True  # If preferred source fails, try others

    def __post_init__(self) -> None:
        if self.download_dir is None:
            self.download_dir = Path("data/training")


@dataclass
class TrainingDataResolver:
    """Unified resolver for training data across all sources.

    This is the primary API for training to get data. It abstracts away
    the complexity of finding and fetching data from multiple sources.

    Usage:
        resolver = TrainingDataResolver()
        result = await resolver.resolve_best_data("hex8_2p")
        if result.success:
            train_model(result.path)
    """

    config: ResolverConfig = field(default_factory=ResolverConfig)
    _manifest: ClusterManifest | None = field(default=None, repr=False)
    _discovery: GameDiscovery | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._manifest = get_cluster_manifest()
        self._discovery = GameDiscovery()

    async def resolve_best_data(
        self,
        config_key: str,
        min_size_mb: float | None = None,
        max_age_hours: float | None = None,
        prefer_source: DataSource | None = None,
    ) -> ResolvedDataResult:
        """Find and fetch best available training data.

        Resolution order (unless prefer_source overrides):
        1. LOCAL - Check local filesystem first
        2. P2P - Query ClusterManifest for other nodes
        3. S3 - Download from S3 bucket (uses S3Inventory for discovery)
        4. OWC - Download from OWC external drive (uses OWCInventory for discovery)

        Jan 3, 2026: Enhanced with S3Inventory/OWCInventory fallback discovery.
        If manifest doesn't have locations, we now query inventories directly.

        Args:
            config_key: Config key (e.g., 'hex8_2p')
            min_size_mb: Minimum file size in MB (default: config value)
            max_age_hours: Maximum age in hours (default: config value)
            prefer_source: Override preferred source

        Returns:
            ResolvedDataResult with path to local data file
        """
        min_size = min_size_mb if min_size_mb is not None else self.config.min_size_mb
        prefer = prefer_source or self.config.prefer_source

        # Get all available data across sources (manifest-registered)
        all_sources = self._manifest.find_across_all_sources(config_key)

        # Enhance with S3/OWC inventory discovery (Jan 3, 2026)
        all_sources = await self._enhance_with_inventory_discovery(
            config_key, all_sources
        )

        # Build resolution order
        resolution_order = self._get_resolution_order(prefer)

        for source in resolution_order:
            locations = all_sources.get(source, [])
            if not locations:
                continue

            result = await self._try_source(
                config_key=config_key,
                source=source,
                locations=locations,
                min_size_mb=min_size,
            )
            if result.success:
                return result

            # If not trying all sources, stop after preferred source fails
            if not self.config.try_all_sources and source == prefer:
                return result

        return ResolvedDataResult(
            config_key=config_key,
            status=ResolutionStatus.NOT_FOUND,
            error="No training data found in any source",
        )

    async def _enhance_with_inventory_discovery(
        self,
        config_key: str,
        all_sources: dict[DataSource, list[dict[str, Any]]],
    ) -> dict[DataSource, list[dict[str, Any]]]:
        """Enhance source locations with S3/OWC inventory discovery.

        Jan 3, 2026: Sprint 4 - Dynamic data fetching for training.

        If S3/OWC locations are not in the manifest, query the inventory
        modules directly to discover available databases.
        """
        # Parse config key using centralized utility
        parsed = parse_config_key(config_key)
        if not parsed:
            return all_sources

        board_type = parsed.board_type
        num_players = parsed.num_players

        # Check S3 if no manifest locations
        if not all_sources.get(DataSource.S3):
            s3_locations = await self._discover_s3_locations(config_key, board_type, num_players)
            if s3_locations:
                all_sources[DataSource.S3] = s3_locations

        # Check OWC if no manifest locations
        if not all_sources.get(DataSource.OWC):
            owc_locations = await self._discover_owc_locations(config_key, board_type, num_players)
            if owc_locations:
                all_sources[DataSource.OWC] = owc_locations

        return all_sources

    async def _discover_s3_locations(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
    ) -> list[dict[str, Any]]:
        """Discover S3 locations using S3Inventory.

        Jan 3, 2026: Sprint 4 - Dynamic S3 discovery.
        """
        try:
            from app.coordination.s3_inventory import get_s3_inventory

            inventory = get_s3_inventory()
            stats = await inventory.get_full_stats()

            if config_key in stats.games_by_config:
                game_info = stats.games_by_config[config_key]
                # Construct S3 key based on expected path structure
                s3_key = f"consolidated/games/canonical_{board_type}_{num_players}p.db"
                return [{
                    "s3_key": s3_key,
                    "s3_bucket": inventory.config.bucket,
                    "game_count": game_info.game_count,
                    "file_size_bytes": game_info.total_size_bytes,
                    "source": "s3_inventory_discovery",
                }]
        except ImportError:
            logger.debug("[TrainingDataResolver] S3Inventory not available")
        except Exception as e:
            logger.debug(f"[TrainingDataResolver] S3 discovery error: {e}")

        return []

    async def _discover_owc_locations(
        self,
        config_key: str,
        board_type: str,
        num_players: int,
    ) -> list[dict[str, Any]]:
        """Discover OWC locations using OWCInventory.

        Jan 3, 2026: Sprint 4 - Dynamic OWC discovery.
        """
        try:
            from app.coordination.owc_inventory import get_owc_inventory

            inventory = get_owc_inventory()
            stats = await inventory.get_full_stats()

            if not stats.drive_available:
                return []

            if config_key in stats.games_by_config:
                game_info = stats.games_by_config[config_key]
                # Construct OWC path based on expected structure
                owc_path = (
                    f"{inventory.config.base_path}/{inventory.config.games_subdir}/"
                    f"canonical_{board_type}_{num_players}p.db"
                )
                return [{
                    "owc_path": owc_path,
                    "owc_host": inventory.config.host,
                    "game_count": game_info.game_count,
                    "file_size_bytes": game_info.total_size_bytes,
                    "source": "owc_inventory_discovery",
                }]
        except ImportError:
            logger.debug("[TrainingDataResolver] OWCInventory not available")
        except Exception as e:
            logger.debug(f"[TrainingDataResolver] OWC discovery error: {e}")

        return []

    async def prefetch_for_training(
        self,
        config_keys: list[str],
        parallel: bool = True,
    ) -> dict[str, ResolvedDataResult]:
        """Prefetch data for multiple configs before training.

        This is useful for pipeline preparation - fetch all data upfront
        so training doesn't block on downloads.

        Args:
            config_keys: List of config keys to prefetch
            parallel: If True, fetch all configs in parallel

        Returns:
            Dictionary mapping config_key to ResolvedDataResult
        """
        results: dict[str, ResolvedDataResult] = {}

        if parallel:
            tasks = [
                self.resolve_best_data(config_key) for config_key in config_keys
            ]
            resolved = await asyncio.gather(*tasks, return_exceptions=True)
            for config_key, result in zip(config_keys, resolved):
                if isinstance(result, Exception):
                    results[config_key] = ResolvedDataResult(
                        config_key=config_key,
                        status=ResolutionStatus.DOWNLOAD_FAILED,
                        error=str(result),
                    )
                else:
                    results[config_key] = result
        else:
            for config_key in config_keys:
                try:
                    results[config_key] = await self.resolve_best_data(config_key)
                except Exception as e:
                    results[config_key] = ResolvedDataResult(
                        config_key=config_key,
                        status=ResolutionStatus.DOWNLOAD_FAILED,
                        error=str(e),
                    )

        return results

    def _get_resolution_order(
        self, prefer_source: DataSource | None
    ) -> list[DataSource]:
        """Get resolution order based on preference.

        Default order: LOCAL > P2P > S3 > OWC
        If prefer_source is set, it comes first.
        """
        default_order = [
            DataSource.LOCAL,
            DataSource.P2P,
            DataSource.S3,
            DataSource.OWC,
        ]

        if prefer_source is None:
            return default_order

        # Move preferred source to front
        order = [prefer_source]
        for source in default_order:
            if source != prefer_source:
                order.append(source)
        return order

    async def _try_source(
        self,
        config_key: str,
        source: DataSource,
        locations: list[dict[str, Any]],
        min_size_mb: float,
    ) -> ResolvedDataResult:
        """Try to get data from a specific source."""
        if source == DataSource.LOCAL:
            return await self._try_local(config_key, locations, min_size_mb)
        elif source == DataSource.P2P:
            return await self._try_p2p(config_key, locations, min_size_mb)
        elif source == DataSource.S3:
            return await self._try_s3(config_key, locations, min_size_mb)
        elif source == DataSource.OWC:
            return await self._try_owc(config_key, locations, min_size_mb)
        else:
            return ResolvedDataResult(
                config_key=config_key,
                status=ResolutionStatus.NOT_FOUND,
                error=f"Unknown source: {source}",
            )

    async def _try_local(
        self,
        config_key: str,
        locations: list[dict[str, Any]],
        min_size_mb: float,
    ) -> ResolvedDataResult:
        """Try to find data on local filesystem."""
        # Also check via GameDiscovery for local databases
        parsed = parse_config_key(config_key)
        if parsed:
            local_dbs = self._discovery.find_databases_for_config(
                board_type=parsed.board_type,
                num_players=parsed.num_players,
            )
            for db in local_dbs:
                db_path = Path(db.path)
                if db_path.exists():
                    size_mb = db_path.stat().st_size / (1024 * 1024)
                    if size_mb >= min_size_mb:
                        return ResolvedDataResult(
                            config_key=config_key,
                            status=ResolutionStatus.FOUND_LOCAL,
                            path=db_path,
                            source=DataSource.LOCAL,
                            game_count=db.game_count,
                            file_size_mb=size_mb,
                        )

        # Check locations from manifest
        for loc in locations:
            db_path = Path(loc.get("db_path", ""))
            if db_path.exists():
                size_mb = db_path.stat().st_size / (1024 * 1024)
                if size_mb >= min_size_mb:
                    return ResolvedDataResult(
                        config_key=config_key,
                        status=ResolutionStatus.FOUND_LOCAL,
                        path=db_path,
                        source=DataSource.LOCAL,
                        game_count=loc.get("game_count", 0),
                        file_size_mb=size_mb,
                    )

        return ResolvedDataResult(
            config_key=config_key,
            status=ResolutionStatus.NOT_FOUND,
            error="No local data found",
        )

    async def _try_p2p(
        self,
        config_key: str,
        locations: list[dict[str, Any]],
        min_size_mb: float,
    ) -> ResolvedDataResult:
        """Try to sync data from P2P cluster node."""
        # Sort by game count (most games first)
        sorted_locs = sorted(
            locations, key=lambda x: x.get("game_count", 0), reverse=True
        )

        for loc in sorted_locs:
            node_id = loc.get("node_id")
            remote_path = loc.get("db_path")
            game_count = loc.get("game_count", 0)

            if not node_id or not remote_path:
                continue

            # Try to sync from this node
            start_time = time.time()
            try:
                local_path = await self._sync_from_node(
                    node_id=node_id,
                    remote_path=remote_path,
                    config_key=config_key,
                )
                if local_path and local_path.exists():
                    size_mb = local_path.stat().st_size / (1024 * 1024)
                    return ResolvedDataResult(
                        config_key=config_key,
                        status=ResolutionStatus.FOUND_P2P,
                        path=local_path,
                        source=DataSource.P2P,
                        game_count=game_count,
                        file_size_mb=size_mb,
                        download_time_seconds=time.time() - start_time,
                        node_id=node_id,
                    )
            except Exception as e:
                logger.warning(f"[TrainingDataResolver] P2P sync failed from {node_id}: {e}")
                continue

        return ResolvedDataResult(
            config_key=config_key,
            status=ResolutionStatus.NOT_FOUND,
            error="P2P sync failed for all nodes",
        )

    async def _sync_from_node(
        self,
        node_id: str,
        remote_path: str,
        config_key: str,
    ) -> Path | None:
        """Sync a database from a cluster node."""
        # Use cluster transport for sync
        try:
            from app.coordination.cluster_transport import ClusterTransport

            transport = ClusterTransport()
            local_dir = self.config.download_dir / "synced"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / f"{config_key}_{node_id}.db"

            success = await asyncio.wait_for(
                transport.download_file(
                    node_id=node_id,
                    remote_path=remote_path,
                    local_path=str(local_path),
                ),
                timeout=self.config.p2p_sync_timeout,
            )
            return local_path if success else None
        except ImportError:
            logger.debug("[TrainingDataResolver] ClusterTransport not available")
            return None
        except asyncio.TimeoutError:
            logger.warning(f"[TrainingDataResolver] P2P sync timeout from {node_id}")
            return None

    async def _try_s3(
        self,
        config_key: str,
        locations: list[dict[str, Any]],
        min_size_mb: float,
    ) -> ResolvedDataResult:
        """Try to download data from S3."""
        # Sort by game count (most games first)
        sorted_locs = sorted(
            locations, key=lambda x: x.get("game_count", 0), reverse=True
        )

        for loc in sorted_locs:
            s3_key = loc.get("s3_key")
            s3_bucket = loc.get("s3_bucket", self.config.s3_bucket)
            game_count = loc.get("game_count", 0)

            if not s3_key:
                continue

            start_time = time.time()
            try:
                local_path = await self._download_from_s3(
                    s3_bucket=s3_bucket,
                    s3_key=s3_key,
                    config_key=config_key,
                )
                if local_path and local_path.exists():
                    size_mb = local_path.stat().st_size / (1024 * 1024)
                    return ResolvedDataResult(
                        config_key=config_key,
                        status=ResolutionStatus.DOWNLOADED_S3,
                        path=local_path,
                        source=DataSource.S3,
                        game_count=game_count,
                        file_size_mb=size_mb,
                        download_time_seconds=time.time() - start_time,
                    )
            except Exception as e:
                logger.warning(f"[TrainingDataResolver] S3 download failed: {e}")
                continue

        return ResolvedDataResult(
            config_key=config_key,
            status=ResolutionStatus.NOT_FOUND,
            error="S3 download failed for all locations",
        )

    async def _download_from_s3(
        self,
        s3_bucket: str,
        s3_key: str,
        config_key: str,
    ) -> Path | None:
        """Download a file from S3."""
        import subprocess

        local_dir = self.config.download_dir / "s3"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / f"{config_key}_s3.db"

        s3_uri = f"s3://{s3_bucket}/{s3_key}"

        def _do_download() -> bool:
            result = subprocess.run(
                ["aws", "s3", "cp", s3_uri, str(local_path)],
                capture_output=True,
                timeout=int(self.config.s3_download_timeout),
            )
            return result.returncode == 0

        try:
            success = await asyncio.wait_for(
                asyncio.to_thread(_do_download),
                timeout=self.config.s3_download_timeout,
            )
            return local_path if success else None
        except asyncio.TimeoutError:
            logger.warning(f"[TrainingDataResolver] S3 download timeout: {s3_uri}")
            return None

    async def _try_owc(
        self,
        config_key: str,
        locations: list[dict[str, Any]],
        min_size_mb: float,
    ) -> ResolvedDataResult:
        """Try to download data from OWC drive."""
        # Sort by game count (most games first)
        sorted_locs = sorted(
            locations, key=lambda x: x.get("game_count", 0), reverse=True
        )

        for loc in sorted_locs:
            owc_path = loc.get("owc_path")
            owc_host = loc.get("owc_host", self.config.owc_host)
            game_count = loc.get("game_count", 0)

            if not owc_path:
                continue

            start_time = time.time()
            try:
                local_path = await self._download_from_owc(
                    owc_host=owc_host,
                    owc_path=owc_path,
                    config_key=config_key,
                )
                if local_path and local_path.exists():
                    size_mb = local_path.stat().st_size / (1024 * 1024)
                    return ResolvedDataResult(
                        config_key=config_key,
                        status=ResolutionStatus.DOWNLOADED_OWC,
                        path=local_path,
                        source=DataSource.OWC,
                        game_count=game_count,
                        file_size_mb=size_mb,
                        download_time_seconds=time.time() - start_time,
                    )
            except Exception as e:
                logger.warning(f"[TrainingDataResolver] OWC download failed: {e}")
                continue

        return ResolvedDataResult(
            config_key=config_key,
            status=ResolutionStatus.NOT_FOUND,
            error="OWC download failed for all locations",
        )

    async def _download_from_owc(
        self,
        owc_host: str,
        owc_path: str,
        config_key: str,
    ) -> Path | None:
        """Download a file from OWC drive via SSH."""
        import subprocess

        local_dir = self.config.download_dir / "owc"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / f"{config_key}_owc.db"

        def _do_download() -> bool:
            result = subprocess.run(
                ["rsync", "-az", f"{owc_host}:{owc_path}", str(local_path)],
                capture_output=True,
                timeout=int(self.config.owc_download_timeout),
            )
            return result.returncode == 0

        try:
            success = await asyncio.wait_for(
                asyncio.to_thread(_do_download),
                timeout=self.config.owc_download_timeout,
            )
            return local_path if success else None
        except asyncio.TimeoutError:
            logger.warning(f"[TrainingDataResolver] OWC download timeout from {owc_host}")
            return None

    def get_data_availability_report(
        self,
        config_keys: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get a report of data availability across all sources.

        This is useful for monitoring and debugging data distribution.

        Args:
            config_keys: Specific configs to check (None = all known)

        Returns:
            Dictionary mapping config_key to availability info
        """
        if config_keys is None:
            # Get all known configs from manifest
            config_keys = list(
                set(
                    self._manifest.get_all_configs()
                    if hasattr(self._manifest, "get_all_configs")
                    else []
                )
            )

        report: dict[str, dict[str, Any]] = {}
        for config_key in config_keys:
            sources = self._manifest.find_across_all_sources(config_key)
            total_games = self._manifest.get_total_games_across_sources(config_key)

            report[config_key] = {
                "total_games": total_games,
                "local_count": len(sources.get(DataSource.LOCAL, [])),
                "p2p_count": len(sources.get(DataSource.P2P, [])),
                "s3_count": len(sources.get(DataSource.S3, [])),
                "owc_count": len(sources.get(DataSource.OWC, [])),
                "has_local": len(sources.get(DataSource.LOCAL, [])) > 0,
                "has_backup": (
                    len(sources.get(DataSource.S3, [])) > 0
                    or len(sources.get(DataSource.OWC, [])) > 0
                ),
            }

        return report


# Singleton instance
_resolver_instance: TrainingDataResolver | None = None


def get_training_data_resolver() -> TrainingDataResolver:
    """Get the singleton TrainingDataResolver instance."""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = TrainingDataResolver()
    return _resolver_instance


def reset_training_data_resolver() -> None:
    """Reset the singleton instance (for testing)."""
    global _resolver_instance
    _resolver_instance = None
