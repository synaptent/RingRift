"""Election HTTP Handlers Mixin.

Extracted from p2p_orchestrator.py for modularity.
This mixin provides leader election and voter management endpoints.

Usage:
    class P2POrchestrator(ElectionHandlersMixin, ...):
        pass
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import constants
try:
    from scripts.p2p.constants import LEADER_LEASE_DURATION
except ImportError:
    LEADER_LEASE_DURATION = 180

# Import types
try:
    from scripts.p2p.types import NodeRole
except ImportError:
    from enum import Enum

    class NodeRole(str, Enum):
        LEADER = "leader"
        FOLLOWER = "follower"
        CANDIDATE = "candidate"


class ElectionHandlersMixin:
    """Mixin providing election HTTP handlers.

    Requires the implementing class to have:
    - node_id: str
    - role: NodeRole
    - leader_id: str | None
    - election_in_progress: bool
    - peers_lock: threading.RLock
    - peers: dict[str, NodeInfo]
    - self_info: NodeInfo
    - auth_token: str | None
    - voter_node_ids: list[str]
    - voter_grant_leader_id: str
    - voter_grant_lease_id: str
    - voter_grant_expires: float
    - leader_lease_id: str
    - leader_lease_expires: float
    - last_leader_seen: float
    - last_lease_renewal: float
    - last_election_attempt: float
    - _update_self_info() method
    - _endpoint_conflict_keys() method
    - _is_leader_eligible() method
    - _has_voter_quorum() method
    - _start_election() method
    - _increment_cluster_epoch() method
    - _save_state() method
    - _is_request_authorized() method
    """

    # Type hints for IDE support
    node_id: str
    role: Any  # NodeRole
    leader_id: str | None
    election_in_progress: bool
    auth_token: str | None

    async def handle_election(self, request: web.Request) -> web.Response:
        """Handle election message from another node."""
        try:
            # Only "bully" lower-priority candidates when we're actually eligible
            # to act as a leader. Otherwise (e.g. NAT-blocked / ambiguous endpoint),
            # responding ALIVE can stall elections and leave the cluster leaderless.
            self._update_self_info()
            data = await request.json()
            candidate_id = str(data.get("candidate_id") or "")
            if not candidate_id:
                return web.json_response({"error": "missing_candidate_id"}, status=400)

            with self.peers_lock:
                peers_snapshot = [
                    p for p in self.peers.values() if p.node_id != self.node_id
                ]
            conflict_keys = self._endpoint_conflict_keys([self.self_info, *peers_snapshot])
            eligible = self._is_leader_eligible(
                self.self_info, conflict_keys, require_alive=False
            )
            voter_node_ids = list(getattr(self, "voter_node_ids", []) or [])
            if eligible and voter_node_ids:
                # When quorum gating is enabled, only configured voters can participate
                # in bully elections. Non-voters responding "ALIVE" would stall the
                # election because their own `_start_election()` returns early.
                eligible = (
                    self.node_id in voter_node_ids
                ) and self._has_voter_quorum()

            # If our ID is higher, we respond with "ALIVE" (Bully algorithm)
            if self.node_id > candidate_id and eligible:
                # Start our own election
                asyncio.create_task(self._start_election())
                return web.json_response(
                    {"response": "ALIVE", "node_id": self.node_id, "eligible": True}
                )
            else:
                return web.json_response(
                    {"response": "OK", "node_id": self.node_id, "eligible": bool(eligible)}
                )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_lease_request(self, request: web.Request) -> web.Response:
        """Voter endpoint: grant/renew an exclusive leader lease.

        A leader candidate must obtain grants from a quorum of voters before it
        may act as leader. Voters only grant to one leader at a time until the
        lease expires (or is explicitly released by stepping down).
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)
            data = await request.json()
            leader_id = str(
                data.get("leader_id") or data.get("candidate_id") or ""
            ).strip()
            lease_id = str(data.get("lease_id") or "").strip()
            duration_raw = data.get("lease_duration", LEADER_LEASE_DURATION)
            try:
                duration = int(duration_raw)
            except ValueError:
                duration = int(LEADER_LEASE_DURATION)
            duration = max(10, min(duration, int(LEADER_LEASE_DURATION * 2)))

            if not leader_id or not lease_id:
                return web.json_response(
                    {"granted": False, "reason": "missing_fields"}, status=400
                )

            voters = list(getattr(self, "voter_node_ids", []) or [])
            if voters:
                if self.node_id not in voters:
                    return web.json_response(
                        {"granted": False, "reason": "not_a_voter"}, status=403
                    )
                if leader_id not in voters:
                    return web.json_response(
                        {"granted": False, "reason": "leader_not_voter"}, status=403
                    )

            now = time.time()
            current_leader = str(getattr(self, "voter_grant_leader_id", "") or "")
            current_expires = float(getattr(self, "voter_grant_expires", 0.0) or 0.0)

            if current_leader and current_expires > now and current_leader != leader_id:
                return web.json_response(
                    {
                        "granted": False,
                        "reason": "lease_already_granted",
                        "current_leader_id": current_leader,
                        "current_lease_id": str(
                            getattr(self, "voter_grant_lease_id", "") or ""
                        ),
                        "lease_expires": current_expires,
                    },
                    status=409,
                )

            self.voter_grant_leader_id = leader_id
            self.voter_grant_lease_id = lease_id
            self.voter_grant_expires = now + float(duration)
            self._save_state()

            lease_ttl_seconds = max(0.0, float(self.voter_grant_expires) - time.time())
            return web.json_response(
                {
                    "granted": True,
                    "leader_id": leader_id,
                    "lease_id": lease_id,
                    "lease_expires": self.voter_grant_expires,
                    # Use a relative TTL for robustness under clock skew (absolute
                    # timestamps from different machines are not directly comparable).
                    "lease_ttl_seconds": lease_ttl_seconds,
                    "voter_id": self.node_id,
                }
            )
        except Exception as e:
            return web.json_response({"granted": False, "error": str(e)}, status=400)

    async def handle_voter_grant_status(self, request: web.Request) -> web.Response:
        """Read-only voter endpoint: return our currently granted leader lease.

        This lets nodes resolve split-brain by consulting a quorum of voters for
        the active lease holder, without mutating lease state.
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)
            now = time.time()
            expires = float(getattr(self, "voter_grant_expires", 0.0) or 0.0)
            return web.json_response(
                {
                    "voter_id": self.node_id,
                    "now": now,
                    "leader_id": str(getattr(self, "voter_grant_leader_id", "") or ""),
                    "lease_id": str(getattr(self, "voter_grant_lease_id", "") or ""),
                    "lease_expires": expires,
                    "lease_ttl_seconds": max(0.0, expires - now),
                }
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_election_reset(self, request: web.Request) -> web.Response:
        """Reset stuck election state to allow fresh leader election.

        This endpoint clears election-in-progress flags and cached leader state,
        allowing a new election to proceed. Use when elections are deadlocked.

        POST /election/reset
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)

            old_state = {
                "election_in_progress": self.election_in_progress,
                "role": str(self.role),
                "leader_id": self.leader_id,
                "leader_lease_id": getattr(self, "leader_lease_id", ""),
                "leader_lease_expires": getattr(self, "leader_lease_expires", 0.0),
            }

            # Reset election state
            self.election_in_progress = False
            self.leader_id = None
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
            self.last_lease_renewal = 0.0
            self.last_election_attempt = 0.0
            if self.role == NodeRole.LEADER:
                self.role = NodeRole.FOLLOWER

            # Clear voter grants if we were granting to ourselves
            if str(getattr(self, "voter_grant_leader_id", "") or "") == self.node_id:
                self.voter_grant_leader_id = ""
                self.voter_grant_lease_id = ""
                self.voter_grant_expires = 0.0

            self._save_state()

            logger.info(f"Election state reset on {self.node_id}: {old_state}")

            return web.json_response(
                {
                    "status": "reset",
                    "node_id": self.node_id,
                    "previous_state": old_state,
                    "message": "Election state cleared. New election will start on next heartbeat cycle.",
                }
            )
        except Exception as e:
            logger.error(f"Error resetting election: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_election_force_leader(self, request: web.Request) -> web.Response:
        """Force a specific node to become leader (emergency override).

        This bypasses normal election and directly sets leadership. Use only
        when normal elections are persistently failing.

        POST /election/force_leader
        Body: {"leader_id": "node-id-to-become-leader"}
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)

            data = await request.json()
            target_leader_id = str(data.get("leader_id", "")).strip()

            if not target_leader_id:
                return web.json_response({"error": "leader_id required"}, status=400)

            # If we're the target, become leader
            if target_leader_id == self.node_id:
                import uuid

                lease_id = (
                    f"{self.node_id}_{int(time.time())}_forced_{uuid.uuid4().hex[:8]}"
                )

                self.role = NodeRole.LEADER
                self.leader_id = self.node_id
                self.leader_lease_id = lease_id
                self.leader_lease_expires = time.time() + LEADER_LEASE_DURATION
                self.last_leader_seen = time.time()
                self.election_in_progress = False

                self._increment_cluster_epoch()
                self._save_state()

                logger.warning(
                    f"FORCED LEADERSHIP: {self.node_id} is now leader via override"
                )

                return web.json_response(
                    {
                        "status": "leader_forced",
                        "node_id": self.node_id,
                        "role": "leader",
                        "lease_id": lease_id,
                        "lease_expires": self.leader_lease_expires,
                        "warning": "Leadership was forced without normal election. Use with caution.",
                    }
                )
            else:
                # Store the forced leader hint so we adopt it
                self.leader_id = target_leader_id
                self.role = NodeRole.FOLLOWER
                self.election_in_progress = False
                self._save_state()

                logger.info(
                    f"Accepting forced leader {target_leader_id} on node {self.node_id}"
                )

                return web.json_response(
                    {
                        "status": "leader_accepted",
                        "node_id": self.node_id,
                        "forced_leader": target_leader_id,
                        "role": "follower",
                    }
                )
        except Exception as e:
            logger.error(f"Error forcing leader: {e}")
            return web.json_response({"error": str(e)}, status=500)
