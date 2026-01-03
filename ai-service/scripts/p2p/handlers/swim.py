"""SWIM HTTP Handlers Mixin.

Provides HTTP endpoints for SWIM membership protocol status and member listing.

December 2025: Migrated to use BaseP2PHandler for consistent response formatting.

Usage:
    class P2POrchestrator(SwimHandlersMixin, ...):
        pass

Endpoints:
    GET /swim/status  - Get SWIM membership status and configuration
    GET /swim/members - Get list of SWIM members with their states
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import (
    handler_timeout,
    HANDLER_TIMEOUT_GOSSIP,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import SWIM adapter availability check
try:
    from app.p2p.swim_adapter import SWIM_AVAILABLE
except ImportError:
    SWIM_AVAILABLE = False

# Import constants with fallbacks
try:
    from scripts.p2p.constants import (
        MEMBERSHIP_MODE,
        SWIM_BIND_PORT,
        SWIM_ENABLED,
        SWIM_FAILURE_TIMEOUT,
        SWIM_PING_INTERVAL,
        SWIM_SUSPICION_TIMEOUT,
    )
except ImportError:
    SWIM_ENABLED = False
    SWIM_BIND_PORT = 7947
    SWIM_FAILURE_TIMEOUT = 5.0
    SWIM_SUSPICION_TIMEOUT = 3.0
    SWIM_PING_INTERVAL = 1.0
    MEMBERSHIP_MODE = "http"


class SwimHandlersMixin(BaseP2PHandler):
    """Mixin providing SWIM HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - node_id: str (from BaseP2PHandler)
    - auth_token: str | None (from BaseP2PHandler)
    - _swim_manager: SwimMembershipManager | None
    - _swim_started: bool
    - _is_request_authorized(request) method
    - get_swim_membership_summary() method (from MembershipMixin)
    """

    # Type hints for IDE support
    _swim_manager: Any  # Optional[SwimMembershipManager]
    _swim_started: bool

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_swim_status(self, request: web.Request) -> web.Response:
        """GET /swim/status - Get SWIM membership status and configuration.

        Returns SWIM protocol configuration, current state, and summary statistics.
        Does not require authentication for read-only status.

        Response:
            {
                "node_id": "my-node",
                "swim_enabled": true,
                "swim_available": true,
                "swim_started": true,
                "membership_mode": "hybrid",
                "config": {
                    "bind_port": 7947,
                    "failure_timeout": 5.0,
                    "suspicion_timeout": 3.0,
                    "ping_interval": 1.0
                },
                "summary": {
                    "members": 10,
                    "alive": 8,
                    "suspected": 1,
                    "failed": 1
                },
                "timestamp": 1703500000.0
            }
        """
        try:
            # Get membership summary (uses MembershipMixin method if available)
            get_summary = getattr(self, "get_swim_membership_summary", None)
            if callable(get_summary):
                summary = get_summary()
            else:
                summary = {
                    "swim_enabled": SWIM_ENABLED,
                    "swim_available": SWIM_AVAILABLE,
                    "swim_started": getattr(self, "_swim_started", False),
                    "membership_mode": MEMBERSHIP_MODE,
                }

            response = {
                "node_id": self.node_id,
                **summary,
                "config": {
                    "bind_port": SWIM_BIND_PORT,
                    "failure_timeout": SWIM_FAILURE_TIMEOUT,
                    "suspicion_timeout": SWIM_SUSPICION_TIMEOUT,
                    "ping_interval": SWIM_PING_INTERVAL,
                },
                "timestamp": time.time(),
            }

            return self.json_response(response)

        except Exception as e:
            logger.error(f"Error in handle_swim_status: {e}", exc_info=True)
            return self.error_response(
                str(e),
                status=500,
                details={
                    "swim_enabled": SWIM_ENABLED,
                    "swim_available": SWIM_AVAILABLE,
                },
            )

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_swim_members(self, request: web.Request) -> web.Response:
        """GET /swim/members - Get list of SWIM members with their states.

        Returns detailed member information including state and metadata.
        Does not require authentication for read-only member listing.

        Response:
            {
                "node_id": "my-node",
                "swim_started": true,
                "members": [
                    {
                        "id": "peer-1",
                        "state": "alive",
                        "address": "192.168.1.10:7947"
                    },
                    {
                        "id": "peer-2",
                        "state": "suspected",
                        "address": "192.168.1.11:7947"
                    }
                ],
                "alive_count": 8,
                "suspected_count": 1,
                "failed_count": 1,
                "timestamp": 1703500000.0
            }
        """
        try:
            swim_started = getattr(self, "_swim_started", False)
            swim_manager = getattr(self, "_swim_manager", None)

            if not swim_started or swim_manager is None:
                return self.json_response(
                    {
                        "node_id": self.node_id,
                        "swim_started": False,
                        "members": [],
                        "alive_count": 0,
                        "suspected_count": 0,
                        "failed_count": 0,
                        "message": "SWIM not started or not available",
                        "timestamp": time.time(),
                    }
                )

            # Get member list from SWIM manager
            members = []
            alive_count = 0
            suspected_count = 0
            failed_count = 0

            try:
                # Access the internal swim node's members if available
                swim_node = getattr(swim_manager, "_swim", None)
                if swim_node is not None:
                    for member in swim_node.members:
                        member_info = {
                            "id": member.id,
                            "state": member.state,
                        }

                        # Add address if available
                        if hasattr(member, "address"):
                            member_info["address"] = str(member.address)

                        members.append(member_info)

                        # Count by state
                        if member.state == "alive":
                            alive_count += 1
                        elif member.state == "suspected":
                            suspected_count += 1
                        elif member.state == "failed":
                            failed_count += 1

                else:
                    # Fallback: use public API
                    alive_peers = swim_manager.get_alive_peers()
                    for peer_id in alive_peers:
                        members.append({"id": peer_id, "state": "alive"})
                        alive_count += 1

            except Exception as e:
                logger.warning(f"Error getting SWIM members: {e}")
                # Return what we have

            return self.json_response(
                {
                    "node_id": self.node_id,
                    "swim_started": swim_started,
                    "members": members,
                    "alive_count": alive_count,
                    "suspected_count": suspected_count,
                    "failed_count": failed_count,
                    "timestamp": time.time(),
                }
            )

        except Exception as e:
            logger.error(f"Error in handle_swim_members: {e}", exc_info=True)
            return self.error_response(
                str(e),
                status=500,
                details={
                    "swim_started": False,
                    "members": [],
                },
            )
