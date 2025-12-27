"""Elo Sync HTTP Handlers Mixin.

Provides HTTP endpoints for synchronizing Elo rating databases across the cluster.
Enables consistent model rankings and tournament results across all nodes.

Usage:
    class P2POrchestrator(EloSyncHandlersMixin, ...):
        pass

Endpoints:
    GET /elo/sync/status - Get Elo database sync status
    POST /elo/sync/push - Push local Elo database to cluster
    POST /elo/sync/pull - Pull latest Elo database from leader
    GET /elo/ratings - Get current Elo ratings for all models

Sync Mechanism:
    Leader maintains authoritative Elo database. Nodes sync periodically
    and after gauntlet evaluations. Conflict resolution uses timestamp-based
    last-write-wins for individual model ratings.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

# Dec 2025: Use consolidated handler utilities
from scripts.p2p.handlers.handlers_base import get_event_bridge

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Event bridge manager for safe event emission (Dec 2025 consolidation)
_event_bridge = get_event_bridge()


class EloSyncHandlersMixin:
    """Mixin providing Elo sync HTTP handlers.

    Requires the implementing class to have:
    - node_id: str
    - ringrift_path: str
    - elo_sync_manager: Optional[EloSyncManager]
    - sync_in_progress: bool
    """

    # Type hints for IDE support
    node_id: str
    ringrift_path: str
    elo_sync_manager: Any  # Optional[EloSyncManager]
    sync_in_progress: bool

    async def handle_elo_sync_status(self, request: web.Request) -> web.Response:
        """GET /elo/sync/status - Get Elo database sync status."""
        try:
            if not self.elo_sync_manager:
                return web.json_response({
                    "enabled": False,
                    "error": "EloSyncManager not initialized"
                })

            status = self.elo_sync_manager.get_status()
            status["enabled"] = True
            status["node_id"] = self.node_id

            return web.json_response(status)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_elo_sync_trigger(self, request: web.Request) -> web.Response:
        """POST /elo/sync/trigger - Manually trigger Elo database sync."""
        try:
            if not self.elo_sync_manager:
                return web.json_response({
                    "success": False,
                    "error": "EloSyncManager not initialized"
                }, status=503)

            # Trigger sync
            success = await self.elo_sync_manager.sync_with_cluster()

            return web.json_response({
                "success": success,
                "status": self.elo_sync_manager.get_status()
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_elo_sync_download(self, request: web.Request) -> web.Response:
        """GET /elo/sync/db - Download unified_elo.db for cluster sync."""
        try:
            ai_root = Path(self.ringrift_path) / "ai-service"
            db_path = ai_root / "data" / "unified_elo.db"

            if not db_path.exists():
                return web.json_response({"error": "Database not found"}, status=404)

            # Read and return the database file
            with open(db_path, 'rb') as f:
                data = f.read()

            return web.Response(
                body=data,
                content_type='application/octet-stream',
                headers={
                    'Content-Disposition': 'attachment; filename="unified_elo.db"',
                    'Content-Length': str(len(data))
                }
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_elo_sync_upload(self, request: web.Request) -> web.Response:
        """POST /elo/sync/upload - Upload/merge unified_elo.db from another node.

        Dec 2025: Added optional auth token check for security.
        Set RINGRIFT_ADMIN_TOKEN to require auth, or leave unset for open access.
        """
        import os
        admin_token = os.environ.get("RINGRIFT_ADMIN_TOKEN")
        if admin_token:
            request_token = request.headers.get("X-Admin-Token", "")
            if request_token != admin_token:
                logger.warning("Unauthorized Elo sync upload attempt")
                return web.json_response({"error": "Unauthorized"}, status=401)

        try:
            if not self.elo_sync_manager:
                return web.json_response({
                    "success": False,
                    "error": "EloSyncManager not initialized"
                }, status=503)

            # Read uploaded database
            data = await request.read()
            if not data:
                return web.json_response({"error": "No data received"}, status=400)

            # Save to temp file and merge
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
                f.write(data)
                temp_path = Path(f.name)

            try:
                # Use merge if enabled
                if self.elo_sync_manager.enable_merge:
                    success = await self.elo_sync_manager._merge_databases(temp_path)
                else:
                    # Simple replace
                    shutil.copy(temp_path, self.elo_sync_manager.db_path)
                    success = True

                self.elo_sync_manager._update_local_stats()

                return web.json_response({
                    "success": success,
                    "match_count": self.elo_sync_manager.state.local_match_count
                })
            finally:
                temp_path.unlink(missing_ok=True)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _trigger_elo_sync_after_matches(self, num_matches: int = 1):
        """Trigger Elo sync after recording new matches.

        This is called after recording match results to ensure cluster-wide
        consistency. It debounces sync requests to avoid overwhelming the
        network with syncs after every individual match.
        """
        if not self.elo_sync_manager:
            return

        # Debounce: only sync if enough new matches or enough time has passed
        # This prevents sync storms when processing many matches in quick succession
        MIN_MATCHES_FOR_IMMEDIATE_SYNC = 10
        MIN_INTERVAL_BETWEEN_SYNCS = 30  # seconds

        try:
            last_sync = getattr(self, '_last_elo_sync_trigger', 0)
            now = time.time()

            # Accumulate pending matches
            pending = getattr(self, '_pending_sync_matches', 0) + num_matches
            self._pending_sync_matches = pending

            # Check if we should sync now
            should_sync = (
                pending >= MIN_MATCHES_FOR_IMMEDIATE_SYNC or
                (now - last_sync) >= MIN_INTERVAL_BETWEEN_SYNCS
            )

            if should_sync and not self.sync_in_progress:
                self._last_elo_sync_trigger = now
                self._pending_sync_matches = 0
                success = await self.elo_sync_manager.sync_with_cluster()
                if success:
                    logger.info(f"Elo sync triggered after {pending} matches: "
                          f"{self.elo_sync_manager.state.local_match_count} total")
                    # Emit Elo sync event to coordination EventRouter (Dec 2025 consolidation)
                    # Note: We don't have individual model Elo changes here,
                    # so emit a generic sync completion signal
                    await _event_bridge.emit("p2p_elo_updated", {
                        "model_id": "cluster_sync",
                        "config_key": "all",
                        "old_elo": 0.0,
                        "new_elo": 0.0,
                        "games_played": self.elo_sync_manager.state.local_match_count,
                    })
        except Exception as e:
            logger.info(f"Elo sync trigger error: {e}")
