"""Election HTTP Handlers Mixin.

Provides HTTP endpoints for Bully algorithm leader election and voter-based
quorum management. Supports lease-based leader locking with automatic expiry.

December 2025: Migrated to use BaseP2PHandler for consistent response formatting.

Usage:
    class P2POrchestrator(ElectionHandlersMixin, ...):
        pass

Endpoints:
    POST /election/start - Start a new leader election
    POST /election/vote - Request vote from this voter node
    POST /election/leader - Announce leadership claim
    POST /election/renew-lease - Renew leader lease (extend TTL)
    GET /election/status - Get current election state and leader info
    GET /voters - Get list of registered voter nodes

Leader Election Flow:
    1. Candidate calls /election/start to begin election
    2. Candidate requests votes via /election/vote from voters
    3. With quorum (3/5), candidate claims leadership via /election/leader
    4. Leader renews lease periodically via /election/renew-lease
    5. On lease expiry, followers can start new election
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import aiohttp
from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import (
    handler_timeout,
    HANDLER_TIMEOUT_GOSSIP,
)

# Dec 2025: Use consolidated handler utilities
from scripts.p2p.handlers.handlers_base import get_event_bridge

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Event bridge manager for safe event emission (Dec 2025 consolidation)
_event_bridge = get_event_bridge()

# Import constants
try:
    from scripts.p2p.constants import LEADER_LEASE_DURATION
except ImportError:
    # Jan 5, 2026: Reduced from 180s to 90s for faster failover (210s → 120s total)
    LEADER_LEASE_DURATION = 90

# Import types
try:
    from scripts.p2p.types import NodeRole
except ImportError:
    from enum import Enum

    class NodeRole(str, Enum):
        LEADER = "leader"
        FOLLOWER = "follower"
        CANDIDATE = "candidate"


class ElectionHandlersMixin(BaseP2PHandler):
    """Mixin providing election HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - node_id: str (from BaseP2PHandler)
    - auth_token: str | None (from BaseP2PHandler)
    - role: NodeRole
    - leader_id: str | None
    - election_in_progress: bool
    - peers_lock: threading.RLock
    - peers: dict[str, NodeInfo]
    - self_info: NodeInfo
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
    role: Any  # NodeRole
    leader_id: str | None
    election_in_progress: bool

    # Jan 20, 2026: Maximum voter config version delta before probation
    MAX_VOTER_CONFIG_VERSION_DELTA = 3

    def _check_voter_config_probation(
        self, leader_id: str, request_data: dict
    ) -> tuple[bool, str]:
        """Check if a leader candidate is on probation due to stale voter config.

        Jan 20, 2026: Nodes with voter config version delta > 3 must sync
        their config before they can lead. This prevents nodes with stale
        voter lists from causing cluster instability.

        Args:
            leader_id: The candidate requesting leadership
            request_data: Request payload (may contain voter_config_version)

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        try:
            # Get local config version
            local_version = self._get_voter_config_version()
            if local_version == 0:
                # No local config, can't enforce probation
                return True, "no_local_config"

            # Get candidate's version from request or peer info
            candidate_version = request_data.get("voter_config_version")

            if candidate_version is None:
                # Try to get from peer info
                peers = getattr(self, "peers", {})
                peer_info = peers.get(leader_id)
                if peer_info:
                    candidate_version = getattr(
                        peer_info, "voter_config_version", None
                    )

            if candidate_version is None:
                # Can't determine version, allow (for backward compatibility)
                return True, "version_unknown"

            # Check version delta
            version_delta = abs(local_version - candidate_version)

            if version_delta > self.MAX_VOTER_CONFIG_VERSION_DELTA:
                return False, (
                    f"config_stale: candidate v{candidate_version} "
                    f"vs local v{local_version} (delta={version_delta})"
                )

            return True, "config_ok"

        except Exception as e:
            # On error, allow (don't break elections)
            logger.debug(f"[Election] Probation check error: {e}")
            return True, f"check_error: {e}"

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_election(self, request: web.Request) -> web.Response:
        """Handle election message from another node."""
        try:
            # Only "bully" lower-priority candidates when we're actually eligible
            # to act as a leader. Otherwise (e.g. NAT-blocked / ambiguous endpoint),
            # responding ALIVE can stall elections and leave the cluster leaderless.
            self._update_self_info()
            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            candidate_id = str(data.get("candidate_id") or "")
            if not candidate_id:
                return self.bad_request("missing_candidate_id")

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
                return self.json_response({
                    "response": "ALIVE",
                    "node_id": self.node_id,
                    "eligible": True,
                })
            else:
                return self.json_response({
                    "response": "OK",
                    "node_id": self.node_id,
                    "eligible": bool(eligible),
                })
        except Exception as e:
            return self.error_response(str(e), status=400)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_lease_request(self, request: web.Request) -> web.Response:
        """Voter endpoint: grant/renew an exclusive leader lease.

        A leader candidate must obtain grants from a quorum of voters before it
        may act as leader. Voters only grant to one leader at a time until the
        lease expires (or is explicitly released by stepping down).
        """
        try:
            if not self.check_auth(request):
                return self.auth_error()
            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")
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
                return self.error_response(
                    "missing_fields",
                    status=400,
                    error_code="MISSING_FIELDS",
                    details={"granted": False},
                )

            voters = list(getattr(self, "voter_node_ids", []) or [])
            if voters:
                if self.node_id not in voters:
                    return self.error_response(
                        "not_a_voter",
                        status=403,
                        error_code="NOT_A_VOTER",
                        details={"granted": False},
                    )
                if leader_id not in voters:
                    return self.error_response(
                        "leader_not_voter",
                        status=403,
                        error_code="LEADER_NOT_VOTER",
                        details={"granted": False},
                    )

            # Jan 20, 2026: Probation check for stale voter config
            # Nodes with voter config version delta > 3 must sync before leading
            probation_result = self._check_voter_config_probation(leader_id, data)
            if not probation_result[0]:
                return self.error_response(
                    probation_result[1],
                    status=403,
                    error_code="LEADER_CONFIG_STALE",
                    details={
                        "granted": False,
                        "reason": probation_result[1],
                        "must_sync": True,
                    },
                )

            now = time.time()

            # Jan 1, 2026: Lease epoch validation (Phase 3A fix)
            # If request has higher epoch, it supersedes cached lease from old leader
            request_epoch = int(data.get("lease_epoch", 0) or 0)
            cached_epoch = int(getattr(self, "_voter_lease_epoch", 0) or 0)
            if request_epoch > cached_epoch:
                logger.info(
                    f"[{self.node_id}] Lease epoch {request_epoch} supersedes cached "
                    f"{cached_epoch}, clearing old grant to {getattr(self, 'voter_grant_leader_id', None)}"
                )
                self.voter_grant_leader_id = None
                self.voter_grant_expires = 0.0
                self._voter_lease_epoch = request_epoch

            current_leader = str(getattr(self, "voter_grant_leader_id", "") or "")
            current_expires = float(getattr(self, "voter_grant_expires", 0.0) or 0.0)

            if current_leader and current_expires > now and current_leader != leader_id:
                return self.json_response(
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
            # Jan 1, 2026: Store epoch with grant for supersession tracking
            self._voter_lease_epoch = max(request_epoch, cached_epoch)
            self._save_state()

            lease_ttl_seconds = max(0.0, float(self.voter_grant_expires) - time.time())
            return self.json_response({
                "granted": True,
                "leader_id": leader_id,
                "lease_id": lease_id,
                "lease_expires": self.voter_grant_expires,
                # Use a relative TTL for robustness under clock skew (absolute
                # timestamps from different machines are not directly comparable).
                "lease_ttl_seconds": lease_ttl_seconds,
                "voter_id": self.node_id,
            })
        except Exception as e:
            return self.error_response(
                str(e),
                status=400,
                details={"granted": False},
            )

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_voter_grant_status(self, request: web.Request) -> web.Response:
        """Read-only voter endpoint: return our currently granted leader lease.

        This lets nodes resolve split-brain by consulting a quorum of voters for
        the active lease holder, without mutating lease state.
        """
        try:
            if not self.check_auth(request):
                return self.auth_error()
            now = time.time()
            expires = float(getattr(self, "voter_grant_expires", 0.0) or 0.0)
            return self.json_response({
                "voter_id": self.node_id,
                "now": now,
                "leader_id": str(getattr(self, "voter_grant_leader_id", "") or ""),
                "lease_id": str(getattr(self, "voter_grant_lease_id", "") or ""),
                "lease_expires": expires,
                "lease_ttl_seconds": max(0.0, expires - now),
            })
        except Exception as e:
            return self.error_response(str(e), status=400)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_lease_revoke(self, request: web.Request) -> web.Response:
        """Voter endpoint: revoke lease when leader steps down.

        Jan 1, 2026: Added for Phase 3B-C fix - leadership stability.

        When a leader steps down (or restarts), it notifies voters to clear their
        cached grants. This prevents the 60s timeout waiting for lease expiry.

        POST /election/lease_revoke
        {
            "leader_id": "node-xyz",     # The leader revoking its lease
            "epoch": 5                   # Current lease epoch
        }
        """
        try:
            if not self.check_auth(request):
                return self.auth_error()
            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            revoking_leader = str(data.get("leader_id") or "").strip()
            revoke_epoch = int(data.get("epoch", 0) or 0)

            cleared = False
            current_leader = str(getattr(self, "voter_grant_leader_id", "") or "")

            # Only clear if this is from the leader we granted to
            if current_leader == revoking_leader:
                logger.info(
                    f"[{self.node_id}] Clearing revoked lease from {revoking_leader}, "
                    f"epoch={revoke_epoch}"
                )
                self.voter_grant_leader_id = None
                self.voter_grant_expires = 0.0
                # Increment epoch to prevent stale grants
                self._voter_lease_epoch = revoke_epoch + 1
                self._save_state()
                cleared = True

            return self.json_response({
                "success": True,
                "cleared": cleared,
                "voter_id": self.node_id,
                "new_epoch": getattr(self, "_voter_lease_epoch", 0),
            })
        except Exception as e:
            return self.error_response(str(e), status=400)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_election_reset(self, request: web.Request) -> web.Response:
        """Reset stuck election state to allow fresh leader election.

        This endpoint clears election-in-progress flags and cached leader state,
        allowing a new election to proceed. Use when elections are deadlocked.

        POST /election/reset
        """
        try:
            if not self.check_auth(request):
                return self.auth_error()

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

            return self.json_response({
                "status": "reset",
                "node_id": self.node_id,
                "previous_state": old_state,
                "message": "Election state cleared. New election will start on next heartbeat cycle.",
            })
        except Exception as e:
            logger.error(f"Error resetting election: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_election_force_leader(self, request: web.Request) -> web.Response:
        """Force a specific node to become leader (emergency override).

        This bypasses normal election and directly sets leadership. Use only
        when normal elections are persistently failing.

        POST /election/force_leader
        Body: {"leader_id": "node-id-to-become-leader"}
        """
        try:
            if not self.check_auth(request):
                return self.auth_error()

            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")
            target_leader_id = str(data.get("leader_id", "")).strip()

            if not target_leader_id:
                return self.bad_request("leader_id required")

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

                # Emit leader change event to coordination EventRouter (Dec 2025 consolidation)
                await _event_bridge.emit("p2p_leader_changed", {
                    "new_leader_id": self.node_id,
                    "old_leader_id": "",  # Unknown previous leader in force mode
                    "term": getattr(self, "cluster_epoch", 0),
                })

                # Jan 9, 2026: Broadcast leadership to all peers for fast propagation
                if hasattr(self, "_broadcast_leader_to_all_peers"):
                    epoch = getattr(self, "cluster_epoch", 0)
                    if hasattr(self, "_leadership_sm") and self._leadership_sm:
                        epoch = getattr(self._leadership_sm, "epoch", epoch)
                    asyncio.create_task(
                        self._broadcast_leader_to_all_peers(
                            self.node_id,
                            epoch,
                            self.leader_lease_expires,
                        )
                    )

                logger.warning(
                    f"FORCED LEADERSHIP: {self.node_id} is now leader via override"
                )

                return self.json_response({
                    "status": "leader_forced",
                    "node_id": self.node_id,
                    "role": "leader",
                    "lease_id": lease_id,
                    "lease_expires": self.leader_lease_expires,
                    "warning": "Leadership was forced without normal election. Use with caution.",
                })
            else:
                # Store the forced leader hint so we adopt it
                self.leader_id = target_leader_id
                self.role = NodeRole.FOLLOWER
                self.election_in_progress = False
                self._save_state()

                logger.info(
                    f"Accepting forced leader {target_leader_id} on node {self.node_id}"
                )

                return self.json_response({
                    "status": "leader_accepted",
                    "node_id": self.node_id,
                    "forced_leader": target_leader_id,
                    "role": "follower",
                })
        except Exception as e:
            logger.error(f"Error forcing leader: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_election_request(self, request: web.Request) -> web.Response:
        """Handle election request from non-voter nodes.

        December 29, 2025: This endpoint allows non-voters to signal to voters
        that an election is needed. This is useful when non-voters detect that
        there's no leader but can't start elections themselves.

        If this node is a voter and quorum is available, it will start an election.
        If this node is not a voter, it will forward the request to known voters.

        Request body:
            - requester_id: str (optional) - ID of the node requesting election
            - reason: str (optional) - Reason for requesting election

        Response:
            - accepted: bool - Whether the election request was accepted
            - action: str - What action was taken (started_election, forwarded, rejected)
            - details: str - Additional details
        """
        try:
            data = await self.parse_json_body(request)
            if data is None:
                data = {}

            requester_id = str(data.get("requester_id") or "unknown")
            reason = str(data.get("reason") or "no_leader")

            logger.info(f"Election request from {requester_id}: {reason}")

            voters = list(getattr(self, "voter_node_ids", []) or [])

            # If we're a voter, try to start an election
            if not voters or self.node_id in voters:
                # Check if we already have a leader
                if self.leader_id:
                    with self.peers_lock:
                        leader = self.peers.get(self.leader_id)
                    if leader and leader.is_alive():
                        return self.json_response({
                            "accepted": False,
                            "action": "rejected",
                            "details": f"Leader {self.leader_id} is already active",
                            "leader_id": self.leader_id,
                        })

                # Check quorum
                if not self._has_voter_quorum():
                    return self.json_response({
                        "accepted": False,
                        "action": "rejected",
                        "details": "No voter quorum available",
                        "voters_needed": getattr(self, "voter_quorum_size", 3),
                    })

                # Start election
                if not getattr(self, "election_in_progress", False):
                    asyncio.create_task(self._start_election())
                    return self.json_response({
                        "accepted": True,
                        "action": "started_election",
                        "details": f"Voter {self.node_id} starting election on behalf of {requester_id}",
                        "voter_id": self.node_id,
                    })
                else:
                    return self.json_response({
                        "accepted": True,
                        "action": "election_in_progress",
                        "details": "Election already in progress",
                        "voter_id": self.node_id,
                    })
            else:
                # We're not a voter, forward to known voters
                # This is best-effort - we don't wait for responses
                import aiohttp

                forwarded_to = []
                for voter_id in voters[:3]:  # Limit to 3 to avoid broadcast storm
                    with self.peers_lock:
                        voter = self.peers.get(voter_id)
                    if voter and voter.is_alive():
                        try:
                            url = self._url_for_peer(voter, "/election/request")
                            async with aiohttp.ClientSession() as session:
                                async with session.post(
                                    url,
                                    json={"requester_id": self.node_id, "reason": reason},
                                    headers=self._auth_headers(),
                                    timeout=aiohttp.ClientTimeout(total=5.0),
                                ) as resp:
                                    if resp.status == 200:
                                        forwarded_to.append(voter_id)
                        except (
                            aiohttp.ClientError,
                            asyncio.TimeoutError,
                            OSError,
                        ):
                            # Network/connection errors expected during election forwarding
                            pass

                if forwarded_to:
                    return self.json_response({
                        "accepted": True,
                        "action": "forwarded",
                        "details": f"Forwarded election request to {len(forwarded_to)} voters",
                        "forwarded_to": forwarded_to,
                    })
                else:
                    return self.json_response({
                        "accepted": False,
                        "action": "forward_failed",
                        "details": "No voters reachable to forward election request",
                    })
        except Exception as e:
            logger.error(f"Error handling election request: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_provisional_leader_claim(self, request: web.Request) -> web.Response:
        """Handle provisional leadership claim from another node.

        Jan 1, 2026: This endpoint handles provisional leadership claims during
        fallback leadership mode. When a node claims provisional leadership after
        prolonged leaderlessness (voter quorum unavailable), peers can:

        1. ACK: Acknowledge the claim (agree to follow this provisional leader)
        2. CHALLENGE: Contest the claim (if we're also claiming provisional leadership)

        If multiple nodes claim provisional leadership, the highest node_id wins.

        Request body:
            - claimant_id: str - ID of the node claiming provisional leadership
            - lease_id: str - Provisional lease ID
            - claimed_at: float - Timestamp when claim was made

        Response:
            - ack: bool - True if we acknowledge this claim
            - challenge: bool - True if we're challenging this claim
            - challenger_id: str - Our node_id if challenging
            - current_leader: str | None - Our known leader if we have one
        """
        try:
            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            claimant_id = str(data.get("claimant_id") or "").strip()
            lease_id = str(data.get("lease_id") or "").strip()
            claimed_at = float(data.get("claimed_at") or 0.0)

            if not claimant_id or not lease_id:
                return self.bad_request("missing claimant_id or lease_id")

            logger.info(f"Received provisional leadership claim from {claimant_id}")

            # If we already have a functioning leader, reject the claim
            if self.leader_id and self.leader_id != claimant_id:
                with self.peers_lock:
                    leader = self.peers.get(self.leader_id)
                if leader and leader.is_alive():
                    logger.info(f"Rejecting provisional claim: have active leader {self.leader_id}")
                    return self.json_response({
                        "ack": False,
                        "challenge": False,
                        "current_leader": self.leader_id,
                        "reason": "have_active_leader",
                    })

            # Check if we're also a provisional leader or claiming
            if self.role == NodeRole.PROVISIONAL_LEADER:
                # We're also claiming - use node_id tiebreaker
                if self.node_id > claimant_id:
                    # We win the tiebreaker, challenge their claim
                    logger.info(f"Challenging provisional claim from {claimant_id}: our ID {self.node_id} > {claimant_id}")
                    return self.json_response({
                        "ack": False,
                        "challenge": True,
                        "challenger_id": self.node_id,
                        "reason": "node_id_tiebreaker",
                    })
                else:
                    # They win the tiebreaker, step down and acknowledge
                    logger.info(f"Stepping down from provisional: {claimant_id} > {self.node_id}")
                    self._step_down_from_provisional()
                    # Fall through to acknowledge

            # If we're a candidate in an election, also use tiebreaker
            if self.role == NodeRole.CANDIDATE:
                if self.node_id > claimant_id:
                    logger.info(f"Challenging provisional claim during election: {self.node_id} > {claimant_id}")
                    return self.json_response({
                        "ack": False,
                        "challenge": True,
                        "challenger_id": self.node_id,
                        "reason": "election_in_progress",
                    })

            # Acknowledge the provisional claim
            self.leader_id = claimant_id
            self.leader_lease_id = lease_id
            self.leader_lease_expires = claimed_at + 60  # Short provisional lease
            self.role = NodeRole.FOLLOWER
            self._save_state()

            logger.info(f"Acknowledged provisional leader: {claimant_id}")

            return self.json_response({
                "ack": True,
                "challenge": False,
                "node_id": self.node_id,
            })

        except Exception as e:
            logger.error(f"Error handling provisional leader claim: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_leader_state_change(self, request: web.Request) -> web.Response:
        """Handle leader step-down broadcast from peers.

        Part of the Unified Leadership State Machine (ULSM) - Jan 2026.
        When a leader steps down, they broadcast to all peers BEFORE clearing
        their local state. This ensures peers learn about the step-down
        immediately rather than waiting for lease expiry.

        Request body:
            {
                "node_id": "stepping-down-leader-id",
                "new_state": "stepping_down",
                "epoch": 5,
                "reason": "quorum_lost",
                "timestamp": 1234567890.123
            }

        Response:
            {"ack": true} on success
        """
        try:
            data = await request.json()
            sender_id = data.get("node_id", "")
            new_state = data.get("new_state", "")
            sender_epoch = data.get("epoch", 0)
            reason = data.get("reason", "unknown")

            logger.info(
                f"Received leader state change: node={sender_id}, "
                f"state={new_state}, epoch={sender_epoch}, reason={reason}"
            )

            # Only process step-down broadcasts from our current leader
            if new_state == "stepping_down" and self.leader_id == sender_id:
                logger.info(
                    f"Leader {sender_id} stepping down (epoch {sender_epoch}), "
                    f"clearing leader_id"
                )

                # Clear our record of the leader
                self.leader_id = None

                # Update state machine if available
                if hasattr(self, "_leadership_sm"):
                    # Set invalidation window to reject stale gossip
                    self._leadership_sm._invalidation_until = time.time() + 60.0
                    # Update epoch if sender's is higher
                    if sender_epoch > self._leadership_sm._epoch:
                        self._leadership_sm._epoch = sender_epoch
                    # Clear the leader in state machine
                    self._leadership_sm.clear_leader()

                # Save state to disk
                self._save_state()

                # Emit event for other components
                _event_bridge.emit(
                    "LEADER_STEP_DOWN_RECEIVED",
                    {
                        "former_leader": sender_id,
                        "epoch": sender_epoch,
                        "reason": reason,
                        "node_id": self.node_id,
                    },
                )

            return self.json_response({"ack": True})

        except Exception as e:
            logger.error(f"Error handling leader state change: {e}")
            return self.error_response(str(e), status=500)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_leader_announcement(self, request: web.Request) -> web.Response:
        """Receive direct leader announcement from elected leader.

        Jan 9, 2026: Added for fast leader propagation (<2s vs 30s gossip).
        Non-voter nodes wait 30s for gossip to propagate leader info.
        Direct broadcast from new leader reduces this to <2s.

        POST /leader_announcement
        Body: {"leader_id": str, "epoch": int, "lease_expires": float}

        Response: {"accepted": bool, "reason": str (if rejected)}
        """
        try:
            data = await request.json()
            leader_id = data.get("leader_id")
            epoch = data.get("epoch", 0)
            lease_expires = data.get("lease_expires", 0)

            if not leader_id:
                return self.json_response({"accepted": False, "reason": "missing_leader_id"})

            # Validate epoch is newer than current
            current_epoch = getattr(self, "_election_epoch", 0)
            if epoch < current_epoch:
                logger.debug(
                    f"[Election] Rejecting stale leader announcement: "
                    f"epoch {epoch} < current {current_epoch}"
                )
                return self.json_response({"accepted": False, "reason": "stale_epoch"})

            # Update leader immediately
            self.leader_id = leader_id
            if hasattr(self, "_election_epoch"):
                self._election_epoch = epoch
            if hasattr(self, "_leader_lease_expires"):
                self._leader_lease_expires = lease_expires

            logger.info(
                f"[Election] Adopted leader {leader_id} via direct announcement (epoch {epoch})"
            )

            # Emit event for observability
            _event_bridge.emit(
                "LEADER_ADOPTED",
                {
                    "leader_id": leader_id,
                    "epoch": epoch,
                    "source": "direct_announcement",
                    "node_id": self.node_id,
                },
            )

            return self.json_response({"accepted": True})

        except Exception as e:
            logger.error(f"Error handling leader announcement: {e}")
            return self.json_response({"accepted": False, "error": str(e)}, status=400)

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_commitment_ack(self, request: web.Request) -> web.Response:
        """Handle commitment ack request for two-phase leadership commitment.

        January 2026: Part of two-phase leadership commitment protocol.
        Leader requests commitment acks from voters after winning election.
        Voters only ack if they agree on the leader.

        Request body:
            {
                "leader_id": "node-1",
                "lease_id": "uuid-1234"  // Optional
            }

        Response:
            {
                "agreed": true,
                "voter_id": "node-2",
                "leader_seen": "node-1",  // What leader this voter sees
                "reason": ""  // If disagreed, why
            }
        """
        try:
            if not self.check_auth(request):
                return self.auth_error()

            data = await self.parse_json_body(request)
            if data is None:
                return self.bad_request("Invalid JSON body")

            leader_id = str(data.get("leader_id", "")).strip()

            if not leader_id:
                return self.json_response({
                    "agreed": False,
                    "voter_id": self.node_id,
                    "leader_seen": self.leader_id,
                    "reason": "missing_leader_id",
                })

            # Check if this is a voter node
            voters = list(getattr(self, "voter_node_ids", []) or [])
            if voters and self.node_id not in voters:
                return self.json_response({
                    "agreed": False,
                    "voter_id": self.node_id,
                    "leader_seen": self.leader_id,
                    "reason": "not_a_voter",
                })

            # Check if we agree on the leader
            current_leader = self.leader_id
            agreed = current_leader == leader_id or current_leader is None

            reason = ""
            if not agreed:
                reason = f"different_leader_seen:{current_leader}"
            elif getattr(self, "election_in_progress", False):
                agreed = False
                reason = "election_in_progress"

            logger.debug(
                f"[CommitmentAck] Request from leader {leader_id}: "
                f"agreed={agreed}, our_leader={current_leader}"
            )

            return self.json_response({
                "agreed": agreed,
                "voter_id": self.node_id,
                "leader_seen": current_leader,
                "reason": reason,
            })

        except Exception as e:
            logger.error(f"Error handling commitment ack: {e}")
            return self.json_response({
                "agreed": False,
                "voter_id": self.node_id,
                "leader_seen": getattr(self, "leader_id", None),
                "reason": f"error:{e}",
            }, status=500)
