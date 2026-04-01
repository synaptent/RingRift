"""NodeDataAgent - Agent running on each cluster node for data discovery and fetching.

This agent provides a unified interface for cluster nodes to:
1. Discover available training data across LOCAL, S3, OWC, and P2P sources
2. Fetch data from the best available source
3. Report local inventory to the coordinator's UnifiedDataCatalog
4. Manage local data cache with LRU eviction

Part of the comprehensive data consolidation system (January 2026).
Phase 6: Enables distributed data fetching for training.

Usage:
    from app.coordination.node_data_agent import (
        NodeDataAgent,
        get_node_data_agent,
    )

    # Get singleton instance
    agent = get_node_data_agent()

    # Get training data for a config (fetches if needed)
    path = await agent.get_training_data("hex8_2p")
    if path:
        train_model(path)

    # Report local inventory to coordinator
    await agent.report_local_inventory()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult
from app.coordination.contracts import CoordinatorStatus

logger = logging.getLogger(__name__)

__all__ = [
    "NodeDataAgent",
    "NodeDataAgentConfig",
    "CacheEntry",
    "get_node_data_agent",
    "reset_node_data_agent",
]


class FetchSource(str, Enum):
    """Sources for fetching data."""
    LOCAL = "local"
    P2P = "p2p"
    S3 = "s3"
    OWC = "owc"
    COORDINATOR = "coordinator"


@dataclass
class CacheEntry:
    """Entry in the local data cache."""
    path: Path
    config_key: str
    data_type: str  # "npz", "model", "database"
    size_bytes: int
    checksum: str
    fetched_at: float
    last_accessed: float
    source: FetchSource


@dataclass
class NodeDataAgentConfig:
    """Configuration for NodeDataAgent."""

    # Feature flags
    enabled: bool = True
    auto_report_inventory: bool = True

    # Cache settings
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    max_cache_size_gb: float = 50.0  # Max cache size in GB
    cache_ttl_hours: float = 48.0  # Cache entries expire after this

    # Fetch settings
    fetch_timeout_seconds: float = 600.0  # 10 minutes
    prefer_local: bool = True
    prefer_p2p: bool = True
    fallback_to_s3: bool = True

    # Coordinator settings
    coordinator_url: str = ""  # Will be auto-discovered
    report_interval_seconds: float = 300.0  # Report inventory every 5 minutes

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 5.0

    # Cycle interval for the daemon
    cycle_interval_seconds: float = 300.0

    @classmethod
    def from_env(cls) -> "NodeDataAgentConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("RINGRIFT_NODE_AGENT_ENABLED", "true").lower() == "true",
            auto_report_inventory=os.getenv("RINGRIFT_NODE_AGENT_AUTO_REPORT", "true").lower() == "true",
            cache_dir=Path(os.getenv("RINGRIFT_NODE_AGENT_CACHE_DIR", "data/cache")),
            max_cache_size_gb=float(os.getenv("RINGRIFT_NODE_AGENT_MAX_CACHE_GB", "50.0")),
            cache_ttl_hours=float(os.getenv("RINGRIFT_NODE_AGENT_CACHE_TTL_HOURS", "48.0")),
            fetch_timeout_seconds=float(os.getenv("RINGRIFT_NODE_AGENT_FETCH_TIMEOUT", "600.0")),
            coordinator_url=os.getenv("RINGRIFT_COORDINATOR_URL", ""),
            report_interval_seconds=float(os.getenv("RINGRIFT_NODE_AGENT_REPORT_INTERVAL", "300.0")),
            cycle_interval_seconds=float(os.getenv("RINGRIFT_NODE_AGENT_CYCLE_INTERVAL", "300.0")),
        )


class NodeDataAgent(HandlerBase):
    """Agent for discovering and fetching training data on cluster nodes.

    Each cluster node runs this agent to:
    1. Query the coordinator's UnifiedDataCatalog for data availability
    2. Fetch data from the best available source (LOCAL > P2P > S3 > OWC)
    3. Maintain a local LRU cache with automatic cleanup
    4. Report local inventory back to the coordinator

    This enables distributed training where nodes can dynamically fetch
    the data they need without manual pre-staging.
    """

    def __init__(self, config: NodeDataAgentConfig | None = None):
        """Initialize the NodeDataAgent.

        Args:
            config: Configuration for the agent behavior
        """
        self._agent_config = config or NodeDataAgentConfig.from_env()
        super().__init__(
            name="NodeDataAgent",
            config=self._agent_config,
            cycle_interval=float(self._agent_config.cycle_interval_seconds),
        )

        # Cache state
        self._cache: dict[str, CacheEntry] = {}  # checksum -> entry
        self._cache_index: dict[str, str] = {}  # config_key:data_type -> checksum

        # Fetch state
        self._pending_fetches: dict[str, asyncio.Task] = {}
        self._fetch_history: list[dict[str, Any]] = []

        # Inventory state
        self._last_inventory_report: float = 0.0
        self._local_inventory: dict[str, dict[str, Any]] = {}

        # Stats
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "fetches_started": 0,
            "fetches_completed": 0,
            "fetches_failed": 0,
            "inventory_reports": 0,
            "bytes_fetched": 0,
            "bytes_evicted": 0,
        }

        # Lock for cache operations
        self._cache_lock = asyncio.Lock()

    @property
    def config(self) -> NodeDataAgentConfig:
        """Return agent configuration."""
        return self._agent_config

    async def _on_start(self) -> None:
        """Called when agent starts."""
        # Ensure cache directory exists
        self._agent_config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cache entries
        await self._scan_local_cache()

        # Initial inventory scan
        await self._scan_local_inventory()

        # Subscribe to relevant events
        await self._subscribe_to_events()

        logger.info(
            f"[NodeDataAgent] Started with cache_dir={self._agent_config.cache_dir}, "
            f"max_cache={self._agent_config.max_cache_size_gb}GB"
        )

    async def _on_stop(self) -> None:
        """Called when agent stops."""
        # Cancel pending fetches
        for task in self._pending_fetches.values():
            if not task.done():
                task.cancel()

        await self._unsubscribe_from_events()

    async def _run_cycle(self) -> None:
        """Run one agent cycle.

        Called periodically to:
        1. Clean up expired cache entries
        2. Report inventory if needed
        3. Evict LRU entries if cache is over limit
        """
        # Clean up expired entries
        await self._cleanup_expired_cache()

        # Evict if over cache limit
        await self._evict_lru_if_needed()

        # Report inventory periodically
        if self._agent_config.auto_report_inventory:
            now = time.time()
            if now - self._last_inventory_report > self._agent_config.report_interval_seconds:
                await self.report_local_inventory()

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Return event subscriptions."""
        return {
            "DATA_CATALOG_UPDATED": self._on_catalog_updated,
            "DATA_FETCH_REQUESTED": self._on_fetch_requested,
        }

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        try:
            from app.coordination.event_router import subscribe
            for event_type, handler in self._get_event_subscriptions().items():
                subscribe(event_type, handler)

            logger.debug("[NodeDataAgent] Subscribed to events")
        except ImportError:
            logger.debug("[NodeDataAgent] Event bus not available")

    async def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from events."""
        try:
            from app.coordination.event_router import unsubscribe
            for event_type, handler in self._get_event_subscriptions().items():
                unsubscribe(event_type, handler)
        except Exception:
            pass

    def _on_catalog_updated(self, event: Any) -> None:
        """Handle DATA_CATALOG_UPDATED event."""
        # Refresh local understanding of available data
        pass

    def _on_fetch_requested(self, event: Any) -> None:
        """Handle DATA_FETCH_REQUESTED event."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            config_key = payload.get("config_key")
            data_type = payload.get("data_type", "npz")

            if config_key:
                # Trigger async fetch
                asyncio.create_task(
                    self.get_training_data(config_key, data_type)
                )
        except Exception as e:
            logger.debug(f"[NodeDataAgent] Error handling fetch request: {e}")

    async def get_training_data(
        self,
        config_key: str,
        data_type: str = "npz",
        prefer_source: FetchSource | None = None,
    ) -> Path | None:
        """Get training data for a config, fetching if needed.

        Args:
            config_key: Configuration key (e.g., "hex8_2p")
            data_type: Type of data ("npz", "model", "database")
            prefer_source: Preferred source for fetching

        Returns:
            Path to local data file, or None if not found/fetchable
        """
        cache_key = f"{config_key}:{data_type}"

        # Check local cache first
        async with self._cache_lock:
            if cache_key in self._cache_index:
                checksum = self._cache_index[cache_key]
                if checksum in self._cache:
                    entry = self._cache[checksum]
                    if entry.path.exists():
                        # Update access time
                        entry.last_accessed = time.time()
                        self._stats["cache_hits"] += 1
                        logger.debug(f"[NodeDataAgent] Cache hit for {cache_key}")
                        return entry.path

        self._stats["cache_misses"] += 1

        # Check if fetch already in progress
        if cache_key in self._pending_fetches:
            task = self._pending_fetches[cache_key]
            if not task.done():
                logger.debug(f"[NodeDataAgent] Waiting for pending fetch of {cache_key}")
                try:
                    return await task
                except Exception:
                    return None

        # Start new fetch
        task = asyncio.create_task(
            self._fetch_data(config_key, data_type, prefer_source)
        )
        self._pending_fetches[cache_key] = task

        try:
            return await task
        except Exception as e:
            logger.error(f"[NodeDataAgent] Failed to fetch {cache_key}: {e}")
            return None
        finally:
            self._pending_fetches.pop(cache_key, None)

    async def _fetch_data(
        self,
        config_key: str,
        data_type: str,
        prefer_source: FetchSource | None,
    ) -> Path | None:
        """Fetch data from the best available source.

        Source preference order (unless overridden):
        1. LOCAL - Already on disk (non-cache locations)
        2. P2P - From another cluster node
        3. S3 - From AWS S3
        4. OWC - From OWC external drive via coordinator
        """
        self._stats["fetches_started"] += 1
        start_time = time.time()

        # Determine source order
        sources = self._get_source_order(prefer_source)

        for source in sources:
            try:
                path = await self._fetch_from_source(config_key, data_type, source)
                if path and path.exists():
                    # Add to cache
                    await self._add_to_cache(path, config_key, data_type, source)

                    self._stats["fetches_completed"] += 1
                    duration = time.time() - start_time
                    self._fetch_history.append({
                        "config_key": config_key,
                        "data_type": data_type,
                        "source": source.value,
                        "duration": duration,
                        "success": True,
                        "timestamp": time.time(),
                    })

                    logger.info(
                        f"[NodeDataAgent] Fetched {config_key}/{data_type} from {source.value} "
                        f"in {duration:.1f}s"
                    )
                    return path

            except Exception as e:
                logger.debug(f"[NodeDataAgent] Fetch from {source.value} failed: {e}")
                continue

        self._stats["fetches_failed"] += 1
        self._fetch_history.append({
            "config_key": config_key,
            "data_type": data_type,
            "source": None,
            "duration": time.time() - start_time,
            "success": False,
            "timestamp": time.time(),
        })

        return None

    def _get_source_order(self, prefer_source: FetchSource | None) -> list[FetchSource]:
        """Get source order based on preferences."""
        if prefer_source:
            return [prefer_source]

        sources = []
        if self._agent_config.prefer_local:
            sources.append(FetchSource.LOCAL)
        if self._agent_config.prefer_p2p:
            sources.append(FetchSource.P2P)
        if self._agent_config.fallback_to_s3:
            sources.append(FetchSource.S3)
        sources.append(FetchSource.OWC)
        sources.append(FetchSource.COORDINATOR)

        return sources

    async def _fetch_from_source(
        self,
        config_key: str,
        data_type: str,
        source: FetchSource,
    ) -> Path | None:
        """Fetch data from a specific source."""
        if source == FetchSource.LOCAL:
            return await self._fetch_local(config_key, data_type)
        elif source == FetchSource.P2P:
            return await self._fetch_p2p(config_key, data_type)
        elif source == FetchSource.S3:
            return await self._fetch_s3(config_key, data_type)
        elif source == FetchSource.OWC:
            return await self._fetch_owc(config_key, data_type)
        elif source == FetchSource.COORDINATOR:
            return await self._fetch_coordinator(config_key, data_type)
        return None

    async def _fetch_local(self, config_key: str, data_type: str) -> Path | None:
        """Check for data in local non-cache locations."""
        # Standard locations to check
        locations = self._get_local_search_paths(config_key, data_type)

        for path in locations:
            if path.exists():
                return path

        return None

    def _get_local_search_paths(self, config_key: str, data_type: str) -> list[Path]:
        """Get local paths to search for data."""
        paths = []

        if data_type == "npz":
            paths.extend([
                Path(f"data/training/{config_key}.npz"),
                Path(f"data/training/{config_key}_combined.npz"),
                Path(f"data/training/{config_key}_quality.npz"),
            ])
        elif data_type == "model":
            paths.extend([
                Path(f"models/canonical_{config_key}.pth"),
                Path(f"models/ringrift_best_{config_key}.pth"),
            ])
        elif data_type == "database":
            paths.extend([
                Path(f"data/games/canonical_{config_key}.db"),
                Path(f"data/games/{config_key}_selfplay.db"),
            ])

        return paths

    async def _fetch_p2p(self, config_key: str, data_type: str) -> Path | None:
        """Fetch data from another cluster node via P2P."""
        try:
            from app.coordination.auto_sync_daemon import AutoSyncDaemon

            # Use AutoSyncDaemon's P2P infrastructure to find and fetch
            sync = AutoSyncDaemon.get_instance()
            if not sync._running:
                return None

            # Try to get from P2P peers
            # This is a simplified implementation - full implementation would
            # query the P2P manifest and fetch from the best peer
            return None

        except ImportError:
            return None

    async def _fetch_s3(self, config_key: str, data_type: str) -> Path | None:
        """Fetch data from S3."""
        try:
            from app.coordination.s3_import_daemon import S3ImportDaemon

            # Use S3ImportDaemon to fetch
            s3_import = S3ImportDaemon.get_instance()

            # Map data_type to S3ImportDaemon's data types
            s3_data_type_map = {
                "npz": "training",
                "model": "models",
                "database": "databases",
            }
            s3_type = s3_data_type_map.get(data_type, data_type)

            result = await s3_import.import_from_s3(
                config_key=config_key,
                data_type=s3_type,
            )

            if result.get("success") and result.get("files_imported", 0) > 0:
                # Find the imported file
                return await self._fetch_local(config_key, data_type)

            return None

        except ImportError:
            logger.debug("[NodeDataAgent] S3ImportDaemon not available")
            return None
        except Exception as e:
            logger.debug(f"[NodeDataAgent] S3 fetch error: {e}")
            return None

    async def _fetch_owc(self, config_key: str, data_type: str) -> Path | None:
        """Fetch data from OWC external drive."""
        try:
            from app.coordination.owc_import_daemon import get_owc_import_daemon

            owc = get_owc_import_daemon()

            # Trigger import for this config
            result = await owc.import_data_for_config(config_key)

            if result.get("success"):
                # Find the imported file
                return await self._fetch_local(config_key, data_type)

            return None

        except ImportError:
            logger.debug("[NodeDataAgent] OWCImportDaemon not available")
            return None
        except Exception as e:
            logger.debug(f"[NodeDataAgent] OWC fetch error: {e}")
            return None

    async def _fetch_coordinator(self, config_key: str, data_type: str) -> Path | None:
        """Fetch data from coordinator's UnifiedDataCatalog."""
        try:
            coordinator_url = self._agent_config.coordinator_url
            if not coordinator_url:
                # Try to auto-discover coordinator
                coordinator_url = await self._discover_coordinator()

            if not coordinator_url:
                return None

            # Query catalog and fetch
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # First, find the best source
                find_url = f"{coordinator_url}/catalog/find"
                params = {"config": config_key, "type": data_type}

                async with session.get(find_url, params=params, timeout=30) as resp:
                    if resp.status != 200:
                        return None
                    location_data = await resp.json()

                # Fetch from the location
                fetch_url = location_data.get("fetch_url")
                if not fetch_url:
                    return None

                # Download to cache
                cache_path = self._agent_config.cache_dir / f"{config_key}_{data_type}"
                async with session.get(fetch_url, timeout=self._agent_config.fetch_timeout_seconds) as resp:
                    if resp.status != 200:
                        return None

                    # Collect chunks first, then write in thread to avoid blocking
                    chunks: list[bytes] = []
                    async for chunk in resp.content.iter_chunked(8192):
                        chunks.append(chunk)
                        self._stats["bytes_fetched"] += len(chunk)

                    def _write_file() -> None:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_path, "wb") as f:
                            for chunk in chunks:
                                f.write(chunk)

                    await asyncio.to_thread(_write_file)

                return cache_path

        except Exception as e:
            logger.debug(f"[NodeDataAgent] Coordinator fetch error: {e}")
            return None

    async def _discover_coordinator(self) -> str | None:
        """Auto-discover coordinator URL."""
        try:
            # Try well-known coordinator addresses
            candidates = [
                "http://mac-studio:8790",
                "http://localhost:8790",
            ]

            # Also try from cluster config
            try:
                from app.config.cluster_config import get_coordinator_node

                coord = get_coordinator_node()
                if coord:
                    ip = coord.best_ip
                    if ip:
                        candidates.insert(0, f"http://{ip}:8790")
            except ImportError:
                pass

            # Check each candidate
            import aiohttp

            for url in candidates:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{url}/health", timeout=5) as resp:
                            if resp.status == 200:
                                return url
                except Exception:
                    continue

            return None

        except Exception:
            return None

    async def _add_to_cache(
        self,
        path: Path,
        config_key: str,
        data_type: str,
        source: FetchSource,
    ) -> None:
        """Add a file to the cache."""
        async with self._cache_lock:
            # Calculate checksum
            checksum = await self._compute_checksum(path)
            size_bytes = path.stat().st_size

            # Create cache entry
            entry = CacheEntry(
                path=path,
                config_key=config_key,
                data_type=data_type,
                size_bytes=size_bytes,
                checksum=checksum,
                fetched_at=time.time(),
                last_accessed=time.time(),
                source=source,
            )

            cache_key = f"{config_key}:{data_type}"
            self._cache[checksum] = entry
            self._cache_index[cache_key] = checksum

    async def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        def _compute() -> str:
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()

        return await asyncio.to_thread(_compute)

    async def _scan_local_cache(self) -> None:
        """Scan cache directory for existing entries."""
        cache_dir = self._agent_config.cache_dir
        if not cache_dir.exists():
            return

        count = 0
        for path in cache_dir.iterdir():
            if path.is_file():
                try:
                    # Try to parse config_key and data_type from filename
                    # Format: {config_key}_{data_type}
                    parts = path.stem.rsplit("_", 1)
                    if len(parts) == 2:
                        config_key, data_type = parts
                        await self._add_to_cache(
                            path, config_key, data_type, FetchSource.LOCAL
                        )
                        count += 1
                except Exception:
                    continue

        if count > 0:
            logger.info(f"[NodeDataAgent] Loaded {count} entries from cache")

    async def _scan_local_inventory(self) -> None:
        """Scan local data for inventory reporting."""
        inventory: dict[str, dict[str, Any]] = {}

        # Scan data directories
        scan_dirs = [
            ("data/games", "database"),
            ("data/training", "npz"),
            ("models", "model"),
        ]

        for dir_path, data_type in scan_dirs:
            path = Path(dir_path)
            if not path.exists():
                continue

            for file_path in path.glob("*.*"):
                if file_path.is_file():
                    # Extract config_key from filename
                    config_key = self._extract_config_key(file_path.name)
                    if config_key:
                        key = f"{config_key}:{data_type}"
                        inventory[key] = {
                            "path": str(file_path),
                            "size_bytes": file_path.stat().st_size,
                            "mtime": file_path.stat().st_mtime,
                            "data_type": data_type,
                            "config_key": config_key,
                        }

        self._local_inventory = inventory

    def _extract_config_key(self, filename: str) -> str | None:
        """Extract config_key from filename."""
        # Handle various naming patterns
        # canonical_hex8_2p.db -> hex8_2p
        # hex8_2p.npz -> hex8_2p
        # ringrift_best_hex8_2p.pth -> hex8_2p

        import re

        # Try canonical_* pattern
        match = re.match(r"canonical_(\w+_\d+p)", filename)
        if match:
            return match.group(1)

        # Try ringrift_best_* pattern
        match = re.match(r"ringrift_best_(\w+_\d+p)", filename)
        if match:
            return match.group(1)

        # Try direct config pattern
        match = re.match(r"(\w+_\d+p)", filename)
        if match:
            return match.group(1)

        return None

    async def report_local_inventory(self) -> dict[str, Any]:
        """Report local inventory to coordinator's UnifiedDataCatalog."""
        await self._scan_local_inventory()

        try:
            from app.config.env import env

            node_id = env.node_id

            # Try to report to coordinator
            coordinator_url = self._agent_config.coordinator_url
            if not coordinator_url:
                coordinator_url = await self._discover_coordinator()

            if coordinator_url:
                import aiohttp

                report_data = {
                    "node_id": node_id,
                    "timestamp": time.time(),
                    "inventory": self._local_inventory,
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{coordinator_url}/catalog/report",
                        json=report_data,
                        timeout=30,
                    ) as resp:
                        if resp.status == 200:
                            self._stats["inventory_reports"] += 1
                            self._last_inventory_report = time.time()
                            logger.debug(
                                f"[NodeDataAgent] Reported inventory: {len(self._local_inventory)} items"
                            )
                            return {"success": True, "items": len(self._local_inventory)}

            # Fallback: emit event for local handling
            from app.coordination.event_emission_helpers import safe_emit_event

            if safe_emit_event(
                "LOCAL_INVENTORY_UPDATED",
                {
                    "node_id": node_id,
                    "inventory": self._local_inventory,
                    "timestamp": time.time(),
                },
                context="NodeDataAgent",
            ):
                self._stats["inventory_reports"] += 1
                self._last_inventory_report = time.time()
                return {"success": True, "items": len(self._local_inventory)}

            return {"success": False, "error": "No reporting channel available"}

        except Exception as e:
            logger.warning(f"[NodeDataAgent] Failed to report inventory: {e}")
            return {"success": False, "error": str(e)}

    async def _cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        now = time.time()
        ttl_seconds = self._agent_config.cache_ttl_hours * 3600
        expired = []

        async with self._cache_lock:
            for checksum, entry in self._cache.items():
                age = now - entry.fetched_at
                if age > ttl_seconds:
                    expired.append(checksum)

        for checksum in expired:
            await self._evict_entry(checksum)

    async def _evict_lru_if_needed(self) -> None:
        """Evict least-recently-used entries if cache exceeds limit."""
        max_bytes = self._agent_config.max_cache_size_gb * 1024 * 1024 * 1024

        async with self._cache_lock:
            total_size = sum(e.size_bytes for e in self._cache.values())

            if total_size <= max_bytes:
                return

            # Sort by last_accessed (oldest first)
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed,
            )

            # Evict until under limit
            evict_target = total_size - max_bytes
            evicted_bytes = 0

            for checksum, entry in sorted_entries:
                if evicted_bytes >= evict_target:
                    break

                await self._evict_entry(checksum)
                evicted_bytes += entry.size_bytes

    async def _evict_entry(self, checksum: str) -> None:
        """Evict a single cache entry."""
        async with self._cache_lock:
            if checksum not in self._cache:
                return

            entry = self._cache[checksum]

            # Remove file if in cache directory
            if entry.path.exists() and str(self._agent_config.cache_dir) in str(entry.path):
                try:
                    entry.path.unlink()
                    self._stats["bytes_evicted"] += entry.size_bytes
                except Exception:
                    pass

            # Remove from indexes
            cache_key = f"{entry.config_key}:{entry.data_type}"
            self._cache_index.pop(cache_key, None)
            del self._cache[checksum]

    def get_status(self) -> dict[str, Any]:
        """Get agent status."""
        total_cache_bytes = sum(e.size_bytes for e in self._cache.values())

        return {
            "running": self._running,
            "cache_entries": len(self._cache),
            "cache_size_bytes": total_cache_bytes,
            "cache_size_gb": total_cache_bytes / (1024 * 1024 * 1024),
            "max_cache_gb": self._agent_config.max_cache_size_gb,
            "pending_fetches": len(self._pending_fetches),
            "inventory_items": len(self._local_inventory),
            "last_inventory_report": self._last_inventory_report,
            "stats": self._stats,
            "recent_fetches": self._fetch_history[-10:],
        }

    def health_check(self) -> HealthCheckResult:
        """Return health check result."""
        details = self.get_status()

        if not self._running:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.STOPPED,
                message="NodeDataAgent is not running",
                details=details,
            )

        # Check for high failure rate
        total_fetches = self._stats["fetches_completed"] + self._stats["fetches_failed"]
        if total_fetches > 10:
            failure_rate = self._stats["fetches_failed"] / total_fetches
            if failure_rate > 0.5:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"High fetch failure rate: {failure_rate:.1%}",
                    details=details,
                )

        # Check cache utilization
        cache_gb = details["cache_size_gb"]
        max_gb = self._agent_config.max_cache_size_gb
        if cache_gb > max_gb * 0.9:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.RUNNING,
                message=f"Cache near full: {cache_gb:.1f}/{max_gb:.1f}GB",
                details=details,
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"NodeDataAgent healthy ({details['cache_entries']} cached)",
            details=details,
        )


# Singleton management
_node_data_agent: NodeDataAgent | None = None


def get_node_data_agent() -> NodeDataAgent:
    """Get the singleton NodeDataAgent instance."""
    global _node_data_agent
    if _node_data_agent is None:
        _node_data_agent = NodeDataAgent()
    return _node_data_agent


def reset_node_data_agent() -> None:
    """Reset the singleton instance (for testing)."""
    global _node_data_agent
    _node_data_agent = None
