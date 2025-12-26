"""Admin HTTP Handlers Mixin.

Extracted from p2p_orchestrator.py for modularity.
This mixin provides admin and git management endpoints.

Usage:
    class P2POrchestrator(AdminHandlersMixin, ...):
        pass
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import constants
try:
    from scripts.p2p.constants import AUTO_UPDATE_ENABLED
except ImportError:
    AUTO_UPDATE_ENABLED = False


class AdminHandlersMixin:
    """Mixin providing admin and git HTTP handlers.

    Requires the implementing class to have:
    - ringrift_path: str
    - _get_local_git_commit() method
    - _get_local_git_branch() method
    - _check_local_changes() method
    - _check_for_updates() method
    - _get_commits_behind() method
    - _perform_git_update() method
    - _restart_orchestrator() method
    """

    # Type hints for IDE support
    ringrift_path: str

    async def handle_git_status(self, request: web.Request) -> web.Response:
        """Get git status for this node.

        Returns local/remote commit info and whether updates are available.
        """
        try:
            local_commit = self._get_local_git_commit()
            local_branch = self._get_local_git_branch()
            has_local_changes = self._check_local_changes()

            # Check for remote updates (this does a git fetch)
            has_updates, _, remote_commit = self._check_for_updates()
            commits_behind = 0
            if has_updates and local_commit and remote_commit:
                commits_behind = self._get_commits_behind(local_commit, remote_commit)

            return web.json_response({
                "local_commit": local_commit[:8] if local_commit else None,
                "local_commit_full": local_commit,
                "local_branch": local_branch,
                "remote_commit": remote_commit[:8] if remote_commit else None,
                "remote_commit_full": remote_commit,
                "has_updates": has_updates,
                "commits_behind": commits_behind,
                "has_local_changes": has_local_changes,
                "auto_update_enabled": AUTO_UPDATE_ENABLED,
                "ringrift_path": self.ringrift_path,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_git_update(self, request: web.Request) -> web.Response:
        """Manually trigger a git update on this node.

        This will stop jobs, pull updates, and restart the orchestrator.
        """
        try:
            # Check for updates first
            has_updates, local_commit, remote_commit = self._check_for_updates()

            if not has_updates:
                return web.json_response({
                    "success": True,
                    "message": "Already up to date",
                    "local_commit": local_commit[:8] if local_commit else None,
                })

            # Perform the update
            success, message = await self._perform_git_update()

            if success:
                # Schedule restart
                asyncio.create_task(self._restart_orchestrator())
                return web.json_response({
                    "success": True,
                    "message": "Update successful, restarting...",
                    "old_commit": local_commit[:8] if local_commit else None,
                    "new_commit": remote_commit[:8] if remote_commit else None,
                })
            else:
                return web.json_response({
                    "success": False,
                    "message": message,
                }, status=400)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_admin_restart(self, request: web.Request) -> web.Response:
        """Force restart the orchestrator process.

        Useful after code updates when /git/update shows "already up to date"
        but the running process hasn't picked up the changes.
        """
        try:
            logger.info("Admin restart requested via API")
            # Schedule restart (gives time to return response)
            asyncio.create_task(self._restart_orchestrator())
            return web.json_response({
                "success": True,
                "message": "Restart scheduled, process will restart in 2 seconds",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
