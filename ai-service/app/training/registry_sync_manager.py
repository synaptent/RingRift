"""Model Registry Synchronization Manager.

Provides distributed synchronization of model registry state across cluster nodes.
Inherits from DatabaseSyncManager for common sync functionality.

December 2025: Refactored to use DatabaseSyncManager base class.
~260 LOC saved through consolidation.

Features:
- Multi-transport failover (Tailscale → SSH → HTTP) via DatabaseSyncManager
- Merge-based conflict resolution (union of models/versions)
- Periodic sync with cluster nodes
- Integration with UnifiedAILoop

Usage:
    sync_manager = RegistrySyncManager(registry_path=Path("data/model_registry.db"))
    await sync_manager.initialize()

    # Periodic sync:
    await sync_manager.sync_with_cluster()
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from app.coordination.database_sync_manager import (
    DatabaseSyncManager,
    DatabaseSyncState,
    SyncNodeInfo,
)
from app.utils.paths import AI_SERVICE_ROOT

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_REGISTRY_PATH = AI_SERVICE_ROOT / "data" / "model_registry.db"
SYNC_STATE_PATH = AI_SERVICE_ROOT / "data" / "registry_sync_state.json"


# =============================================================================
# Backward-compatible aliases (for existing callers)
# =============================================================================

# Alias: RegistrySyncState -> DatabaseSyncState
# Tests and older code may import RegistrySyncState directly
RegistrySyncState = DatabaseSyncState

# Alias: SyncState -> RegistrySyncState (older convention)
SyncState = DatabaseSyncState

# Alias: NodeInfo -> SyncNodeInfo
NodeInfo = SyncNodeInfo


# =============================================================================
# Registry Sync Manager
# =============================================================================


class RegistrySyncManager(DatabaseSyncManager):
    """Manages model registry synchronization across cluster nodes.

    Inherits from DatabaseSyncManager for common functionality:
    - Multi-transport failover (Tailscale → SSH → Vast.ai SSH → HTTP)
    - Rsync-based database transfers
    - Node discovery from P2P or YAML config
    - Circuit breaker per-node fault tolerance

    Registry-specific features:
    - Merges models and versions tables separately
    - HTTP endpoint support for JSON-based registry export
    - Tracks model_count and version_count separately
    """

    def __init__(
        self,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
        coordinator_host: str = "nebius-backbone-1",
        sync_interval: int = 600,  # 10 minutes
        p2p_url: str | None = None,
    ):
        """Initialize registry sync manager.

        Args:
            registry_path: Path to local model registry database
            coordinator_host: Primary coordinator hostname
            sync_interval: Seconds between sync cycles
            p2p_url: P2P orchestrator URL for node discovery
        """
        super().__init__(
            db_path=registry_path,
            state_path=SYNC_STATE_PATH,
            db_type="registry",
            coordinator_host=coordinator_host,
            sync_interval=float(sync_interval),
            p2p_url=p2p_url,
            enable_merge=True,
        )

        # Registry-specific state tracking
        self._model_count = 0
        self._version_count = 0

        # Track last sync source for merge
        self._current_sync_source: str | None = None

    # =========================================================================
    # Public API
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize the sync manager."""
        self._load_db_state()
        await self.discover_nodes()
        self._update_local_stats()
        logger.info(
            f"RegistrySyncManager initialized: {self._model_count} models, "
            f"{self._version_count} versions"
        )

    @property
    def state(self) -> DatabaseSyncState:
        """Get current sync state (backward compatibility)."""
        return self._db_state

    @property
    def registry_path(self) -> Path:
        """Get registry database path (backward compatibility)."""
        return self.db_path

    def get_sync_status(self) -> dict[str, Any]:
        """Get current sync status (backward compatibility wrapper)."""
        status = self.get_status()
        # Map to old format
        return {
            'last_sync': self._db_state.last_sync_timestamp,
            'local_models': self._model_count,
            'local_versions': self._version_count,
            'synced_nodes': {k: 0.0 for k in self._db_state.synced_nodes},
            'nodes_available': len(self.nodes),
            'circuit_breakers': {
                h: {'state': str(cb.state), 'failures': cb.failure_count}
                for h, cb in self.circuit_breakers.items()
            }
        }

    # =========================================================================
    # Abstract method implementations
    # =========================================================================

    def _get_remote_db_path(self) -> str:
        """Get remote database path for rsync."""
        return "ai-service/data/model_registry.db"

    def _get_remote_count_query(self) -> str:
        """Get SQL query for counting remote records.

        Returns count of models (primary entity).
        """
        return "SELECT COUNT(*) FROM models"

    def _update_local_stats(self) -> None:
        """Update local registry statistics."""
        if not self.db_path.exists():
            return

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM models")
            self._model_count = cursor.fetchone()[0]
            self._db_state.local_record_count = self._model_count

            cursor.execute("SELECT COUNT(*) FROM versions")
            self._version_count = cursor.fetchone()[0]

            conn.close()
        except Exception as e:
            logger.warning(f"Failed to update local stats: {e}")

    async def _merge_databases(self, remote_db_path: Path) -> bool:
        """Merge remote registry database into local.

        Merges both models and versions tables, preserving all unique entries.
        """
        models_merged = 0
        versions_merged = 0
        source = self._current_sync_source or "unknown"

        try:
            remote_conn = sqlite3.connect(str(remote_db_path))
            remote_conn.row_factory = sqlite3.Row
            local_conn = sqlite3.connect(str(self.db_path))

            # Get existing local model IDs
            local_cursor = local_conn.cursor()
            local_cursor.execute("SELECT model_id FROM models")
            local_models = {row[0] for row in local_cursor.fetchall()}

            # Merge models
            remote_cursor = remote_conn.cursor()
            remote_cursor.execute("SELECT * FROM models")
            for row in remote_cursor.fetchall():
                model_id = row['model_id']
                if model_id not in local_models:
                    local_conn.execute("""
                        INSERT INTO models (model_id, name, description, model_type, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (row['model_id'], row['name'], row['description'],
                          row['model_type'], row['created_at'], row['updated_at']))
                    models_merged += 1

            # Get existing local versions
            local_cursor.execute("SELECT model_id, version FROM versions")
            local_versions = {(row[0], row[1]) for row in local_cursor.fetchall()}

            # Merge versions
            remote_cursor.execute("SELECT * FROM versions")
            for row in remote_cursor.fetchall():
                version_key = (row['model_id'], row['version'])
                if version_key not in local_versions:
                    local_conn.execute("""
                        INSERT INTO versions
                        (model_id, version, stage, file_path, file_hash, file_size_bytes,
                         metrics_json, training_config_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (row['model_id'], row['version'], row['stage'], row['file_path'],
                          row['file_hash'], row['file_size_bytes'], row['metrics_json'],
                          row['training_config_json'], row['created_at'], row['updated_at']))
                    versions_merged += 1

            local_conn.commit()
            local_conn.close()
            remote_conn.close()

            if models_merged > 0 or versions_merged > 0:
                logger.info(
                    f"Registry merge from {source}: "
                    f"{models_merged} models, {versions_merged} versions merged"
                )

            return True

        except Exception as e:
            logger.error(f"Database merge failed: {e}")
            return False

    # =========================================================================
    # HTTP-specific sync (override base for JSON handling)
    # =========================================================================

    async def _sync_via_http(self, node: SyncNodeInfo) -> bool:
        """Sync via HTTP from P2P registry export endpoint.

        Registry uses JSON export format via HTTP, not raw database download.
        """
        if not node.http_url:
            return False

        try:
            import aiohttp

            url = f"{self.p2p_url}/api/registry/export?host={node.name}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        return False

                    data = await resp.json()
                    return await self._merge_registry_data(data, node.name)

        except ImportError:
            logger.debug("aiohttp not available for HTTP sync")
            return False
        except Exception as e:
            logger.warning(f"HTTP registry sync failed: {e}")
            return False

    async def _merge_registry_data(self, data: dict[str, Any], source: str) -> bool:
        """Merge registry data received via HTTP JSON.

        Args:
            data: JSON data with 'models' and 'versions' arrays
            source: Source node name

        Returns:
            True if merge succeeded
        """
        models_merged = 0
        versions_merged = 0

        try:
            local_conn = sqlite3.connect(str(self.db_path))
            cursor = local_conn.cursor()

            # Get existing IDs
            cursor.execute("SELECT model_id FROM models")
            local_models = {row[0] for row in cursor.fetchall()}

            cursor.execute("SELECT model_id, version FROM versions")
            local_versions = {(row[0], row[1]) for row in cursor.fetchall()}

            # Merge models from data
            for model in data.get('models', []):
                if model['model_id'] not in local_models:
                    cursor.execute("""
                        INSERT INTO models (model_id, name, description, model_type, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (model['model_id'], model['name'], model.get('description', ''),
                          model['model_type'], model['created_at'], model['updated_at']))
                    models_merged += 1

            # Merge versions
            for version in data.get('versions', []):
                version_key = (version['model_id'], version['version'])
                if version_key not in local_versions:
                    cursor.execute("""
                        INSERT INTO versions
                        (model_id, version, stage, file_path, file_hash, file_size_bytes,
                         metrics_json, training_config_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (version['model_id'], version['version'], version['stage'],
                          version['file_path'], version['file_hash'], version['file_size_bytes'],
                          version.get('metrics_json', '{}'), version.get('training_config_json', '{}'),
                          version['created_at'], version['updated_at']))
                    versions_merged += 1

            local_conn.commit()
            local_conn.close()

            self._update_local_stats()

            if models_merged > 0 or versions_merged > 0:
                logger.info(
                    f"Registry HTTP merge from {source}: "
                    f"{models_merged} models, {versions_merged} versions"
                )

            return True

        except Exception as e:
            logger.error(f"HTTP registry merge failed: {e}")
            return False

    # =========================================================================
    # Sync execution (override to track source)
    # =========================================================================

    async def _do_sync(self, node: str) -> bool:
        """Perform sync with a specific node, tracking source for merge."""
        # Track source for merge logging
        self._current_sync_source = node
        try:
            return await super()._do_sync(node)
        finally:
            self._current_sync_source = None


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "RegistrySyncManager",
    "RegistrySyncState",
    "SyncState",
    "NodeInfo",
]
