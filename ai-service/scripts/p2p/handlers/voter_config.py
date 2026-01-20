"""Voter Configuration HTTP Handlers Mixin.

Jan 20, 2026: Provides HTTP endpoints for consensus-safe voter configuration
synchronization. Enables automated config drift detection and resolution.

Usage:
    class P2POrchestrator(VoterConfigHandlersMixin, ...):
        pass

Endpoints:
    GET /voter-config - Get current voter configuration with version/hash
    POST /voter-config/sync - Apply remote voter config (pull mechanism)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import handler_timeout

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VoterConfigHandlersMixin(BaseP2PHandler):
    """HTTP handlers for voter configuration synchronization.

    Jan 20, 2026: Implements the pull mechanism for voter config sync.
    Nodes can request current config from peers and apply newer versions.
    """

    @handler_timeout(5.0)
    async def handle_voter_config_get(self, request: web.Request) -> web.Response:
        """GET /voter-config - Return current voter configuration.

        Returns:
            JSON with:
            - version: Config version number
            - voters: List of voter node IDs
            - quorum_size: Minimum voters for quorum
            - sha256_hash: Full integrity hash
            - hash_short: First 16 chars of hash (for display)
            - created_at: Unix timestamp
            - created_by: Node that created this version
            - voter_count: Number of voters
        """
        try:
            from scripts.p2p.managers.voter_config_manager import get_voter_config_manager

            manager = get_voter_config_manager()
            config = manager.get_current()

            if config is None:
                return web.json_response(
                    {
                        "version": 0,
                        "voters": [],
                        "quorum_size": 0,
                        "sha256_hash": "",
                        "hash_short": "",
                        "created_at": 0,
                        "created_by": "",
                        "voter_count": 0,
                        "error": "no_config_loaded",
                    },
                    status=200,
                )

            return web.json_response(config.to_dict())

        except Exception as e:
            logger.error(f"Error in handle_voter_config_get: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500,
            )

    @handler_timeout(10.0)
    async def handle_voter_config_sync(self, request: web.Request) -> web.Response:
        """POST /voter-config/sync - Apply remote voter configuration.

        Allows a peer to push its voter config to this node. The config
        is only applied if it has a higher version than the local config
        and passes integrity verification.

        Request body:
            - version: int
            - voters: list[str]
            - quorum_size: int
            - sha256_hash: str
            - created_at: float
            - created_by: str

        Returns:
            JSON with:
            - applied: bool - Whether config was applied
            - reason: str - Why config was/wasn't applied
            - local_version: int - Current local version after operation
        """
        try:
            from scripts.p2p.managers.voter_config_manager import get_voter_config_manager
            from app.coordination.voter_config_types import VoterConfigVersion

            data = await request.json()

            # Validate required fields
            required_fields = ["version", "voters", "quorum_size", "sha256_hash"]
            missing = [f for f in required_fields if f not in data]
            if missing:
                return web.json_response(
                    {
                        "applied": False,
                        "reason": f"missing_fields: {missing}",
                        "local_version": self._get_voter_config_version(),
                    },
                    status=400,
                )

            # Build VoterConfigVersion from request data
            remote_config = VoterConfigVersion.from_dict(data)

            # Try to apply
            manager = get_voter_config_manager()
            source = data.get("source", "sync")
            success, reason = manager.apply_remote_config(remote_config, source=source)

            # Get current version after operation
            current_config = manager.get_current()
            local_version = current_config.version if current_config else 0

            return web.json_response(
                {
                    "applied": success,
                    "reason": reason,
                    "local_version": local_version,
                }
            )

        except Exception as e:
            logger.error(f"Error in handle_voter_config_sync: {e}")
            return web.json_response(
                {
                    "applied": False,
                    "reason": f"error: {e}",
                    "local_version": self._get_voter_config_version(),
                },
                status=500,
            )
