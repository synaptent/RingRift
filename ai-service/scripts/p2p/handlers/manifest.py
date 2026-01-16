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
from scripts.p2p.handlers.timeout_decorator import (
    handler_timeout,
    HANDLER_TIMEOUT_TOURNAMENT,
)

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
    _cluster_manifest_received_at: float  # When broadcast was received

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
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

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
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

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_cluster_manifest_broadcast(self, request: web.Request) -> web.Response:
        """Receive cluster manifest broadcast from leader.

        Jan 2026: Added to enable followers to have cluster-wide data visibility.

        This endpoint is called by the leader's ManifestCollectionLoop to push
        the aggregated cluster manifest to all followers. The received manifest
        is stored locally and can be queried via get_cluster_manifest().
        """
        import time
        try:
            data = await request.json()

            # Import ClusterDataManifest for deserialization
            from scripts.p2p.models import ClusterDataManifest

            manifest = ClusterDataManifest.from_dict(data)

            # Store in local cache (thread-safe)
            with self.manifest_lock:
                self.cluster_data_manifest = manifest
                self._cluster_manifest_received_at = time.time()

            # Update local unified registry if available
            try:
                await self._update_unified_registry_from_broadcast(manifest)
            except Exception as e:
                logger.debug(f"[ManifestHandlers] Registry update from broadcast failed: {e}")

            logger.debug(
                f"[ManifestHandlers] Received cluster manifest broadcast: "
                f"{manifest.total_nodes} nodes, {manifest.total_selfplay_games} games"
            )

            return web.json_response({
                "status": "ok",
                "received_at": time.time(),
                "total_nodes": manifest.total_nodes,
                "total_games": manifest.total_selfplay_games,
            })
        except Exception as e:
            logger.warning(f"[ManifestHandlers] Error receiving broadcast: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _update_unified_registry_from_broadcast(self, manifest: object) -> None:
        """Update local unified registry with received cluster manifest.

        Jan 2026: Ensures local data queries reflect cluster-wide state.
        """
        try:
            from app.distributed.data_catalog import get_data_registry

            registry = get_data_registry()
            # UnifiedDataRegistry.update_from_cluster_manifest is sync
            registry.update_from_cluster_manifest(manifest)
        except ImportError:
            # Registry not available
            logger.debug("[ManifestHandlers] Data registry not available for update")
        except Exception as e:
            logger.debug(f"[ManifestHandlers] Failed to update registry: {e}")

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
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

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_request_manifest(self, request: web.Request) -> web.Response:
        """Return cluster manifest regardless of node role.

        Jan 16, 2026: Added to enable on-demand cluster data queries from any node.

        Unlike /data/cluster_manifest (leader-only), this endpoint:
        - Returns cached cluster manifest if available (from broadcast)
        - If no manifest cached, returns leader info for redirect
        - Works on any node (leader, follower, voter)

        This enables tools like cluster_data_status.py to get cluster-wide
        data without waiting for the 5-minute broadcast cycle.

        Returns:
            - If manifest available: {cluster_manifest: {...}, age_seconds: float}
            - If no manifest: {error: "no_manifest", leader_id: str, leader_url: str}
        """
        import time
        try:
            with self.manifest_lock:
                cached_manifest = self.cluster_data_manifest
                received_at = getattr(self, '_cluster_manifest_received_at', 0)

            if cached_manifest:
                age_seconds = time.time() - received_at if received_at else -1
                manifest_dict = (
                    cached_manifest.to_dict()
                    if hasattr(cached_manifest, 'to_dict')
                    else cached_manifest
                )
                return web.json_response({
                    "cluster_manifest": manifest_dict,
                    "age_seconds": age_seconds,
                    "node_id": self.node_id,
                    "is_leader": self.role == NodeRole.LEADER,
                })

            # No manifest cached - provide leader info for redirect
            leader_url = None
            if self.leader_id:
                # Try to get leader's URL from peers
                leader_peer = getattr(self, 'peers', {}).get(self.leader_id)
                if leader_peer:
                    leader_url = getattr(leader_peer, 'http_url', None) or getattr(leader_peer, 'url', None)

            return web.json_response({
                "error": "no_manifest",
                "message": "No cluster manifest cached. Query leader directly or wait for broadcast.",
                "leader_id": self.leader_id,
                "leader_url": leader_url,
                "node_id": self.node_id,
            }, status=404)

        except Exception as e:  # noqa: BLE001
            logger.exception("[ManifestHandlers] Error in handle_request_manifest")
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
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

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_backup_status(self, request: web.Request) -> web.Response:
        """Return backup completeness status for all configurations.

        Jan 3, 2026: Sprint 2 - Backup Completeness Tracking

        Compares local canonical databases with S3 and OWC backup inventories
        to show backup coverage percentage per config.

        Query params:
            refresh: If "true", bypass cache and query fresh
            config: If specified, return only that config's status

        Returns:
            {
                "timestamp": ...,
                "overall": {
                    "total_local_games": ...,
                    "total_s3_games": ...,
                    "total_owc_games": ...,
                    "overall_s3_coverage": ...,
                    "overall_owc_coverage": ...,
                    "configs_complete": ...,
                    "configs_incomplete": ...,
                },
                "by_config": {
                    "hex8_2p": {
                        "local_game_count": ...,
                        "s3_game_count": ...,
                        "s3_coverage": ...,
                        "owc_game_count": ...,
                        "owc_coverage": ...,
                        "dual_backed_up": ...,
                        "needs_backup": ...,
                    },
                    ...
                },
            }
        """
        import time
        try:
            refresh_raw = str(request.query.get("refresh", "") or "").strip().lower()
            refresh = refresh_raw in {"1", "true", "yes", "y"}
            config_filter = request.query.get("config")

            backup_status = await self._get_backup_status(
                force_refresh=refresh,
                config_filter=config_filter,
            )

            return web.json_response({
                "timestamp": time.time(),
                "node_id": self.node_id,
                **backup_status,
            })
        except Exception as e:  # noqa: BLE001
            logger.exception("[ManifestHandlers] Error in handle_backup_status")
            return web.json_response({"error": str(e)}, status=500)

    async def _get_backup_status(
        self,
        force_refresh: bool = False,
        config_filter: str | None = None,
    ) -> dict:
        """Get backup completeness status from BackupCompletenessTracker.

        Jan 3, 2026: Sprint 2 - Backup Completeness Tracking
        """
        try:
            from app.coordination.backup_completeness import (
                get_backup_completeness_tracker,
            )

            tracker = get_backup_completeness_tracker()

            if config_filter:
                # Return status for a single config
                status = await tracker.get_backup_status(
                    config_filter,
                    force_refresh=force_refresh,
                )
                return {
                    "config": config_filter,
                    "status": status.to_dict(),
                }
            else:
                # Return overall status with all configs
                overall = await tracker.get_all_status(force_refresh=force_refresh)
                return {
                    "overall": {
                        "total_local_games": overall.total_local_games,
                        "total_s3_games": overall.total_s3_games,
                        "total_owc_games": overall.total_owc_games,
                        "overall_s3_coverage": overall.overall_s3_coverage,
                        "overall_owc_coverage": overall.overall_owc_coverage,
                        "configs_complete": overall.configs_complete,
                        "configs_incomplete": overall.configs_incomplete,
                        "configs_missing": overall.configs_missing,
                        "last_checked": overall.last_checked,
                    },
                    "by_config": {
                        k: v.to_dict() for k, v in overall.by_config.items()
                    },
                }
        except ImportError as e:
            logger.warning(f"[ManifestHandlers] BackupCompletenessTracker not available: {e}")
            return {
                "error": "backup_completeness_module_not_available",
                "details": str(e),
            }
        except Exception as e:
            logger.warning(f"[ManifestHandlers] Error getting backup status: {e}")
            return {
                "error": str(e),
            }

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_cluster_data_summary(self, request: web.Request) -> web.Response:
        """Return aggregated data summary across all storage sources.

        Jan 3, 2026: Sprint 3 - Unified Data Visibility

        Aggregates game counts from:
        - Local canonical databases
        - S3 backup inventory
        - OWC backup inventory
        - P2P cluster manifest

        Query params:
            refresh: If "true", bypass cache and query fresh

        Returns:
            {
                "timestamp": ...,
                "summary": {
                    "hex8_2p": {
                        "local": 50000,
                        "s3": 49000,
                        "owc": 48500,
                        "p2p": 52000,
                        "total": 52000,  # Max across sources
                    },
                    ...
                },
                "totals": {
                    "local": 200000,
                    "s3": 195000,
                    "owc": 190000,
                    "p2p": 210000,
                },
            }
        """
        import time
        try:
            refresh_raw = str(request.query.get("refresh", "") or "").strip().lower()
            refresh = refresh_raw in {"1", "true", "yes", "y"}

            data_summary = await self._collect_cluster_data_summary(force_refresh=refresh)

            return web.json_response({
                "timestamp": time.time(),
                "node_id": self.node_id,
                **data_summary,
            })
        except Exception as e:  # noqa: BLE001
            logger.exception("[ManifestHandlers] Error in handle_cluster_data_summary")
            return web.json_response({"error": str(e)}, status=500)

    async def _collect_cluster_data_summary(self, force_refresh: bool = False) -> dict:
        """Collect aggregated data summary from all sources.

        Jan 3, 2026: Sprint 3 - Unified Data Visibility

        Sources:
        1. Local canonical databases (GameDiscovery)
        2. S3 backup inventory (S3Inventory)
        3. OWC backup inventory (OWCInventory)
        4. P2P cluster manifest (ClusterManifest)

        Returns:
            {
                "summary": {config_key: {local, s3, owc, p2p, total}},
                "totals": {local, s3, owc, p2p},
            }
        """
        summary: dict[str, dict[str, int]] = {}
        totals = {"local": 0, "s3": 0, "owc": 0, "p2p": 0}

        # 1. Local canonical databases
        try:
            from app.utils.game_discovery import GameDiscovery
            discovery = GameDiscovery()

            for db_info in discovery.find_all_databases():
                # Only count canonical databases
                if "canonical" not in db_info.path.name:
                    continue

                config_key = f"{db_info.board_type}_{db_info.num_players}p"
                if config_key not in summary:
                    summary[config_key] = {"local": 0, "s3": 0, "owc": 0, "p2p": 0, "total": 0}

                summary[config_key]["local"] += db_info.game_count
                totals["local"] += db_info.game_count
        except Exception as e:
            logger.warning(f"[ManifestHandlers] Error getting local counts: {e}")

        # 2. S3 inventory
        try:
            from app.coordination.s3_inventory import get_s3_inventory
            s3_inventory = get_s3_inventory()
            s3_counts = await s3_inventory.get_game_counts(force_refresh=force_refresh)

            for config_key, count in s3_counts.items():
                if config_key not in summary:
                    summary[config_key] = {"local": 0, "s3": 0, "owc": 0, "p2p": 0, "total": 0}
                summary[config_key]["s3"] = count
                totals["s3"] += count
        except ImportError:
            logger.debug("[ManifestHandlers] S3 inventory not available")
        except Exception as e:
            logger.warning(f"[ManifestHandlers] Error getting S3 counts: {e}")

        # 3. OWC inventory
        try:
            from app.coordination.owc_inventory import get_owc_inventory
            owc_inventory = get_owc_inventory()
            owc_counts = await owc_inventory.get_game_counts(force_refresh=force_refresh)

            for config_key, count in owc_counts.items():
                if config_key not in summary:
                    summary[config_key] = {"local": 0, "s3": 0, "owc": 0, "p2p": 0, "total": 0}
                summary[config_key]["owc"] = count
                totals["owc"] += count
        except ImportError:
            logger.debug("[ManifestHandlers] OWC inventory not available")
        except Exception as e:
            logger.warning(f"[ManifestHandlers] Error getting OWC counts: {e}")

        # 4. P2P cluster manifest
        try:
            from app.distributed.cluster_manifest import get_cluster_manifest
            manifest = get_cluster_manifest()

            with manifest._connection() as conn:
                cursor = conn.execute("""
                    SELECT board_type || '_' || num_players || 'p' as config,
                           COUNT(DISTINCT game_id) as game_count
                    FROM game_locations
                    WHERE board_type IS NOT NULL
                    GROUP BY board_type, num_players
                """)
                for row in cursor:
                    config_key = row[0]
                    count = row[1]
                    if config_key and config_key != "None_Nonep":
                        if config_key not in summary:
                            summary[config_key] = {"local": 0, "s3": 0, "owc": 0, "p2p": 0, "total": 0}
                        summary[config_key]["p2p"] = count
                        totals["p2p"] += count
        except Exception as e:
            logger.warning(f"[ManifestHandlers] Error getting P2P counts: {e}")

        # Compute total (max across all sources) for each config
        for config_key, counts in summary.items():
            counts["total"] = max(
                counts["local"],
                counts["s3"],
                counts["owc"],
                counts["p2p"],
            )

        return {
            "summary": summary,
            "totals": totals,
        }
