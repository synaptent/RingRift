"""Manifest HTTP Handlers Mixin.

Provides HTTP endpoints for distributed data manifest management.
Handles local and cluster-wide data inventory collection.

Usage:
    class P2POrchestrator(ManifestHandlersMixin, ...):
        pass

Endpoints:
    GET /data/manifest - Get this node's local data manifest
    GET /data/cluster_manifest - Get cluster-wide data manifest (leader-only)
    POST /data/refresh_manifest - Force refresh of local data manifest

Requires the implementing class to have:
    - node_id: str
    - role: NodeRole
    - leader_id: Optional[str]
    - manifest_lock: threading.Lock
    - local_data_manifest: Optional[DataManifest]
    - cluster_data_manifest: Optional[ClusterDataManifest]
    - _collect_local_data_manifest() method
    - _collect_cluster_manifest() async method
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import NodeRole for leader check
try:
    from scripts.p2p_orchestrator import NodeRole
except ImportError:
    # Fallback enum for type checking
    class NodeRole:  # type: ignore[no-redef]
        LEADER = "leader"
        FOLLOWER = "follower"


class ManifestHandlersMixin(BaseP2PHandler):
    """Mixin providing data manifest HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - node_id: str
    - role: NodeRole
    - leader_id: Optional[str]
    - manifest_lock: threading.Lock
    - local_data_manifest: Optional[DataManifest]
    - cluster_data_manifest: Optional[ClusterDataManifest]
    - _collect_local_data_manifest() method
    - _collect_cluster_manifest() async method
    """

    # Type hints for IDE support
    node_id: str
    leader_id: str | None
    manifest_lock: object  # threading.Lock
    local_data_manifest: object | None
    cluster_data_manifest: object | None

    async def handle_data_manifest(self, request: web.Request) -> web.Response:
        """Return this node's local data manifest.

        Used by leader to collect data inventory from all nodes.
        """
        try:
            local_manifest = await asyncio.to_thread(self._collect_local_data_manifest)
            with self.manifest_lock:
                self.local_data_manifest = local_manifest

            return web.json_response({
                "node_id": self.node_id,
                "manifest": local_manifest.to_dict(),
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_cluster_data_manifest(self, request: web.Request) -> web.Response:
        """Leader-only: Return cluster-wide data manifest.

        Aggregates data manifests from all nodes to show:
        - Total files across cluster
        - Total selfplay games
        - Files missing from specific nodes (for sync planning)
        """
        try:
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "Not leader",
                    "leader_id": self.leader_id,
                }, status=400)

            refresh_raw = str(request.query.get("refresh", "") or "").strip().lower()
            refresh = refresh_raw in {"1", "true", "yes", "y"}

            # Default to returning the cached manifest to keep this endpoint
            # fast and usable by daemons with tight timeouts.
            if not refresh:
                with self.manifest_lock:
                    cached = self.cluster_data_manifest
                if cached:
                    return web.json_response({
                        "cluster_manifest": cached.to_dict(),
                        "cached": True,
                    })
                # Manifest collection loop runs shortly after startup; callers
                # can retry or pass ?refresh=1 to force.
                return web.json_response({
                    "cluster_manifest": None,
                    "cached": True,
                    "error": "manifest_not_ready",
                })

            # Forced refresh: collect and update cache.
            cluster_manifest = await self._collect_cluster_manifest()
            with self.manifest_lock:
                self.cluster_data_manifest = cluster_manifest

            return web.json_response({
                "cluster_manifest": cluster_manifest.to_dict(),
                "cached": False,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_refresh_manifest(self, request: web.Request) -> web.Response:
        """Force refresh of local data manifest."""
        try:
            local_manifest = await asyncio.to_thread(self._collect_local_data_manifest)
            with self.manifest_lock:
                self.local_data_manifest = local_manifest

            return web.json_response({
                "success": True,
                "node_id": self.node_id,
                "total_files": local_manifest.total_files,
                "total_size_bytes": local_manifest.total_size_bytes,
                "selfplay_games": local_manifest.selfplay_games,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_data_inventory(self, request: web.Request) -> web.Response:
        """Return cluster-wide game inventory with counts by config and node.

        Dec 30, 2025: Added for quick cluster-wide data visibility.
        Unlike /cluster_data_manifest, this endpoint:
        - Returns simpler, smaller response focused on game counts
        - Queries ClusterManifest directly (no manifest collection)
        - Works on any node (not leader-only)

        Query params:
            refresh: If "true", bypass cache and query fresh

        Returns:
            {
                "timestamp": ...,
                "total_games": ...,
                "games_by_config": {"hex8_2p": 1234, ...},
                "games_by_node": {"node-1": 500, ...},
            }
        """
        import time
        try:
            refresh_raw = str(request.query.get("refresh", "") or "").strip().lower()
            refresh = refresh_raw in {"1", "true", "yes", "y"}

            inventory = await self._collect_data_inventory(refresh=refresh)

            return web.json_response({
                "timestamp": time.time(),
                "node_id": self.node_id,
                **inventory,
            })
        except Exception as e:  # noqa: BLE001
            logger.exception("[ManifestHandlers] Error in handle_data_inventory")
            return web.json_response({"error": str(e)}, status=500)

    async def _collect_data_inventory(self, refresh: bool = False) -> dict:
        """Collect cluster-wide data inventory from ClusterManifest.

        Dec 30, 2025: Added for quick game count aggregation.
        Dec 30, 2025: Extended with NPZ counts and nodes_with_data for Phase 1.1
                      of distributed data pipeline architecture.

        Returns:
            {
                "total_games": int,
                "games_by_config": {"hex8_2p": 1234, ...},
                "games_by_node": {"node-1": 500, ...},
                "npz_by_config": {"hex8_2p": {"count": 3, "total_samples": 150000}, ...},
                "nodes_with_data": {"hex8_2p": ["node-1", "node-2"], ...},
                "unconsolidated_by_config": {"hex8_2p": 100, ...},  # Games not yet in canonical DB
            }
        """
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest

            manifest = get_cluster_manifest()

            # Get all game locations from the registry
            games_by_config: dict[str, int] = {}
            games_by_node: dict[str, int] = {}
            npz_by_config: dict[str, dict] = {}
            nodes_with_data: dict[str, list[str]] = {}
            unconsolidated_by_config: dict[str, int] = {}
            total_games = 0

            # Query game_locations table for aggregated counts
            with manifest._connection() as conn:
                # Count by config
                cursor = conn.execute("""
                    SELECT board_type || '_' || num_players || 'p' as config, COUNT(DISTINCT game_id)
                    FROM game_locations
                    WHERE board_type IS NOT NULL
                    GROUP BY board_type, num_players
                """)
                for row in cursor:
                    config_key = row[0]
                    count = row[1]
                    if config_key and config_key != "None_Nonep":
                        games_by_config[config_key] = count
                        total_games += count

                # Count by node
                cursor = conn.execute("""
                    SELECT node_id, COUNT(DISTINCT game_id)
                    FROM game_locations
                    GROUP BY node_id
                """)
                for row in cursor:
                    node_id = row[0]
                    count = row[1]
                    games_by_node[node_id] = count

                # NPZ counts by config (Phase 1.1 addition)
                cursor = conn.execute("""
                    SELECT board_type || '_' || num_players || 'p' as config,
                           COUNT(DISTINCT npz_path) as file_count,
                           SUM(sample_count) as total_samples,
                           SUM(file_size) as total_size_bytes
                    FROM npz_locations
                    WHERE board_type IS NOT NULL
                    GROUP BY board_type, num_players
                """)
                for row in cursor:
                    config_key = row[0]
                    if config_key and config_key != "None_Nonep":
                        npz_by_config[config_key] = {
                            "count": row[1] or 0,
                            "total_samples": row[2] or 0,
                            "total_size_bytes": row[3] or 0,
                        }

                # Nodes with data per config (Phase 1.1 addition)
                cursor = conn.execute("""
                    SELECT board_type || '_' || num_players || 'p' as config,
                           node_id,
                           COUNT(DISTINCT game_id) as game_count
                    FROM game_locations
                    WHERE board_type IS NOT NULL
                    GROUP BY board_type, num_players, node_id
                    HAVING game_count > 0
                """)
                for row in cursor:
                    config_key = row[0]
                    node_id = row[1]
                    if config_key and config_key != "None_Nonep":
                        if config_key not in nodes_with_data:
                            nodes_with_data[config_key] = []
                        nodes_with_data[config_key].append(node_id)

                # Unconsolidated games by config (Phase 1.1 addition)
                cursor = conn.execute("""
                    SELECT board_type || '_' || num_players || 'p' as config,
                           COUNT(DISTINCT game_id) as unconsolidated_count
                    FROM game_locations
                    WHERE board_type IS NOT NULL
                      AND (is_consolidated = 0 OR is_consolidated IS NULL)
                    GROUP BY board_type, num_players
                """)
                for row in cursor:
                    config_key = row[0]
                    count = row[1]
                    if config_key and config_key != "None_Nonep":
                        unconsolidated_by_config[config_key] = count

            return {
                "total_games": total_games,
                "games_by_config": games_by_config,
                "games_by_node": games_by_node,
                "npz_by_config": npz_by_config,
                "nodes_with_data": nodes_with_data,
                "unconsolidated_by_config": unconsolidated_by_config,
            }
        except Exception as e:
            logger.warning(f"[ManifestHandlers] Error collecting inventory: {e}")
            return {
                "total_games": 0,
                "games_by_config": {},
                "games_by_node": {},
                "npz_by_config": {},
                "nodes_with_data": {},
                "unconsolidated_by_config": {},
                "error": str(e),
            }
