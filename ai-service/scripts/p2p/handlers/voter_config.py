"""Voter Configuration HTTP Handlers Mixin.

Jan 20, 2026: Provides HTTP endpoints for consensus-safe voter configuration
synchronization. Enables automated config drift detection and resolution.

Usage:
    class P2POrchestrator(VoterConfigHandlersMixin, ...):
        pass

Endpoints:
    GET /voter-config - Get current voter configuration with version/hash
    POST /voter-config/sync - Apply remote voter config (pull mechanism)
    POST /voter-config/ack - Acknowledge a change proposal (joint consensus)
    POST /voter-config/propose - Propose a voter list change (leader only)
    GET /voter-config/change-status - Get current change protocol status
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

    @handler_timeout(10.0)
    async def handle_voter_config_ack(self, request: web.Request) -> web.Response:
        """POST /voter-config/ack - Acknowledge a change proposal.

        Called by the leader during joint consensus to collect acks
        from voters in both old and new voter sets.

        Request body:
            - proposal_id: str - Unique proposal ID
            - phase: str - "old" or "new"
            - old_version: int - Current config version
            - new_version: int - Proposed new version
            - old_voters: list[str] - Current voter list
            - new_voters: list[str] - Proposed voter list

        Returns:
            JSON with:
            - ack: bool - Whether this node acknowledges
            - reason: str - Why ack was granted/denied
        """
        try:
            data = await request.json()

            # Validate required fields
            required_fields = [
                "proposal_id", "phase", "old_version",
                "new_version", "old_voters", "new_voters"
            ]
            missing = [f for f in required_fields if f not in data]
            if missing:
                return web.json_response(
                    {"ack": False, "reason": f"missing_fields: {missing}"},
                    status=400,
                )

            # Get or create the change protocol
            change_protocol = self._get_voter_config_change_protocol()
            if change_protocol is None:
                return web.json_response(
                    {"ack": False, "reason": "change_protocol_not_available"},
                    status=503,
                )

            # Process the ack request
            ack, reason = await change_protocol.handle_ack_request(
                proposal_id=data["proposal_id"],
                phase=data["phase"],
                old_version=data["old_version"],
                new_version=data["new_version"],
                old_voters=data["old_voters"],
                new_voters=data["new_voters"],
            )

            return web.json_response({"ack": ack, "reason": reason})

        except Exception as e:
            logger.error(f"Error in handle_voter_config_ack: {e}")
            return web.json_response(
                {"ack": False, "reason": f"error: {e}"},
                status=500,
            )

    @handler_timeout(180.0)  # 3 minutes - change protocol can take up to 2 min
    async def handle_voter_config_propose(self, request: web.Request) -> web.Response:
        """POST /voter-config/propose - Propose a voter list change.

        Leader-only endpoint to propose changing the voter list.
        Uses joint consensus (both old and new quorums must agree).

        Request body:
            - new_voters: list[str] - New voter node IDs
            - reason: str (optional) - Reason for the change

        Returns:
            JSON with:
            - success: bool - Whether change was applied
            - proposal_id: str - Proposal ID
            - old_version: int - Previous config version
            - new_version: int - New config version (if success)
            - reason: str - Success/failure reason
            - phase_reached: str - Last phase before completion/failure
        """
        try:
            # Check if this node is the leader
            if not self._is_leader():
                return web.json_response(
                    {
                        "success": False,
                        "proposal_id": "",
                        "reason": "not_leader",
                        "phase_reached": "idle",
                    },
                    status=403,
                )

            data = await request.json()

            # Validate required fields
            if "new_voters" not in data:
                return web.json_response(
                    {
                        "success": False,
                        "proposal_id": "",
                        "reason": "missing_new_voters",
                        "phase_reached": "idle",
                    },
                    status=400,
                )

            new_voters = data["new_voters"]
            reason = data.get("reason", "")

            # Get or create the change protocol
            change_protocol = self._get_voter_config_change_protocol()
            if change_protocol is None:
                return web.json_response(
                    {
                        "success": False,
                        "proposal_id": "",
                        "reason": "change_protocol_not_available",
                        "phase_reached": "idle",
                    },
                    status=503,
                )

            # Execute the change
            result = await change_protocol.propose_change(
                new_voters=new_voters,
                reason=reason,
            )

            return web.json_response({
                "success": result.success,
                "proposal_id": result.proposal_id,
                "old_version": result.old_version,
                "new_version": result.new_version,
                "old_voters": result.old_voters,
                "new_voters": result.new_voters,
                "reason": result.reason,
                "phase_reached": result.phase_reached.value,
                "old_acks": result.old_acks,
                "new_acks": result.new_acks,
                "nacks": result.nacks,
                "duration_seconds": result.duration_seconds,
            })

        except Exception as e:
            logger.error(f"Error in handle_voter_config_propose: {e}")
            return web.json_response(
                {
                    "success": False,
                    "proposal_id": "",
                    "reason": f"error: {e}",
                    "phase_reached": "failed",
                },
                status=500,
            )

    @handler_timeout(5.0)
    async def handle_voter_config_change_status(
        self, request: web.Request
    ) -> web.Response:
        """GET /voter-config/change-status - Get change protocol status.

        Returns:
            JSON with:
            - active: bool - Whether a change is in progress
            - proposal_id: str (if active)
            - phase: str - Current phase
            - ... other status fields
        """
        try:
            change_protocol = self._get_voter_config_change_protocol()
            if change_protocol is None:
                return web.json_response({
                    "active": False,
                    "phase": "idle",
                    "error": "change_protocol_not_available",
                })

            return web.json_response(change_protocol.get_status())

        except Exception as e:
            logger.error(f"Error in handle_voter_config_change_status: {e}")
            return web.json_response(
                {"active": False, "phase": "error", "error": str(e)},
                status=500,
            )

    def _get_voter_config_change_protocol(self):
        """Get or create the VoterConfigChangeProtocol instance.

        Returns:
            VoterConfigChangeProtocol instance or None if not available
        """
        # Check if already cached
        protocol = getattr(self, "_voter_config_change_protocol", None)
        if protocol is not None:
            return protocol

        try:
            from scripts.p2p.voter_config_change_protocol import (
                VoterConfigChangeProtocol,
            )
            protocol = VoterConfigChangeProtocol(self)
            self._voter_config_change_protocol = protocol
            return protocol
        except ImportError as e:
            logger.warning(f"VoterConfigChangeProtocol not available: {e}")
            return None
