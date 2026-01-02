"""Leader Election Logic Mixin.

Extracted from p2p_orchestrator.py for modularity.
This mixin provides core leader election and voter quorum logic.

Usage:
    class P2POrchestrator(LeaderElectionMixin, ...):
        pass

Phase 2.2 extraction - Dec 26, 2025
Refactored to use P2PMixinBase - Dec 27, 2025
P5.4: Added Raft leader integration - Dec 30, 2025

Leader Election Modes (controlled by CONSENSUS_MODE):
- "bully": Traditional bully algorithm with voter quorum (default)
- "raft": Use Raft leader as authoritative P2P leader (when available)
- "hybrid": Use Raft for work queue, bully for leadership

When CONSENSUS_MODE="raft" and Raft is initialized, the Raft leader
becomes the P2P leader. This provides:
- Single source of truth for leadership
- Reduced election churn (Raft handles leader changes)
- Strong consistency with work queue operations
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from scripts.p2p.p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    from threading import RLock

    from scripts.p2p.models import NodeInfo
    from scripts.p2p.types import NodeRole

logger = logging.getLogger(__name__)

# Load constants with fallbacks using base class helper
_CONSTANTS = P2PMixinBase._load_config_constants({
    "VOTER_MIN_QUORUM": 3,
    "CONSENSUS_MODE": "bully",
    "RAFT_ENABLED": False,
    "LEADER_LEASE_EXPIRY_GRACE_SECONDS": 30,  # Jan 2026: Stale leader alerting
})

VOTER_MIN_QUORUM = _CONSTANTS["VOTER_MIN_QUORUM"]
CONSENSUS_MODE = _CONSTANTS["CONSENSUS_MODE"]
RAFT_ENABLED = _CONSTANTS["RAFT_ENABLED"]
LEADER_LEASE_EXPIRY_GRACE_SECONDS = _CONSTANTS["LEADER_LEASE_EXPIRY_GRACE_SECONDS"]

# Phase 3.2 (January 2026): Dynamic voter management
# Enable via RINGRIFT_P2P_DYNAMIC_VOTER=true
#
# FEATURE STATUS: DISABLED BY DEFAULT (Sprint 3.5, Jan 2, 2026)
# This feature auto-promotes non-voter nodes to voters when the voter quorum
# is at risk (alive voters <= quorum + 1). This provides automatic failover
# when voter nodes fail.
#
# WHY DISABLED:
# - Cluster stability: Dynamic voter changes can cause split-brain scenarios
#   during network partitions if not carefully tuned.
# - Raft mode has its own membership management.
# - The current voter set (7 nodes across providers) is stable.
#
# ENABLE WHEN:
# - You have a stable cluster with reliable network connectivity
# - You understand the risks of automatic voter promotion
# - You've tested the feature in a non-production environment
#
# CONFIGURATION:
# - RINGRIFT_P2P_DYNAMIC_VOTER=true - Enable dynamic voter management
# - RINGRIFT_P2P_DYNAMIC_VOTER_PROMOTION_DELAY=60 - Seconds to wait before promotion
#
import os
DYNAMIC_VOTER_ENABLED = os.environ.get("RINGRIFT_P2P_DYNAMIC_VOTER", "false").lower() == "true"
VOTER_QUORUM_MARGIN = 1  # Promote when alive voters <= quorum_required + margin
VOTER_MIN_UPTIME_SECONDS = 300.0  # Candidate must have 5+ minutes uptime
VOTER_MAX_ERROR_RATE = 0.10  # Candidate must have <10% error rate

# Import promotion delay from constants (with fallback)
try:
    from app.p2p.constants import DYNAMIC_VOTER_PROMOTION_DELAY
except ImportError:
    DYNAMIC_VOTER_PROMOTION_DELAY = 60


class LeaderElectionMixin(P2PMixinBase):
    """Mixin providing core leader election logic.

    Inherits from P2PMixinBase for shared peer counting helpers.

    Requires the implementing class to have:
    State:
    - node_id: str - This node's ID
    - role: NodeRole - Current node role
    - leader_id: str | None - Current leader's ID
    - leader_lease_id: str - Active lease ID
    - leader_lease_expires: float - Lease expiry timestamp
    - last_lease_renewal: float - Last lease renewal time
    - voter_node_ids: list[str] - Configured voters
    - voter_grant_leader_id: str - Voter grant recipient
    - voter_grant_lease_id: str - Voter grant lease ID
    - voter_grant_expires: float - Voter grant expiry
    - peers_lock: RLock - Lock for peers dict
    - peers: dict[str, NodeInfo] - Active peers

    Methods:
    - _start_election() - Start new election
    - _save_state() - Persist state changes
    """

    MIXIN_TYPE = "leader_election"

    # Type hints for IDE support (implemented by P2POrchestrator)
    node_id: str
    role: Any  # NodeRole
    leader_id: str | None
    leader_lease_id: str
    leader_lease_expires: float
    last_lease_renewal: float
    voter_node_ids: list[str]
    voter_grant_leader_id: str
    voter_grant_lease_id: str
    voter_grant_expires: float
    peers_lock: "RLock"
    peers: dict[str, Any]  # dict[str, NodeInfo]

    def _has_voter_quorum(self) -> bool:
        """Return True if we currently see enough voter nodes alive.

        SIMPLIFIED QUORUM: Uses fixed minimum of 3 voters instead of majority.
        This makes leader election more resilient - as long as 3 voters agree,
        leadership can be established regardless of total voter count.
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return True

        # Use fixed minimum quorum of 3 (or fewer if we have fewer voters)
        quorum = min(VOTER_MIN_QUORUM, len(voters))

        # Use base class helper for counting alive peers
        alive = self._count_alive_peers(voters)
        return alive >= quorum

    def _release_voter_grant_if_self(self) -> None:
        """Release our voter-side lease grant when stepping down.

        This shortens failover time when the leader voluntarily steps down (e.g.
        lost quorum) by not forcing other candidates to wait for the full lease
        TTL to expire.
        """
        if str(getattr(self, "voter_grant_leader_id", "") or "") != self.node_id:
            return
        self.voter_grant_leader_id = ""
        self.voter_grant_lease_id = ""
        self.voter_grant_expires = 0.0

    async def _notify_voters_lease_revoked(self) -> int:
        """Notify all voters to revoke cached lease grants.

        Jan 1, 2026: Phase 3B-C fix for leadership stability.

        When stepping down from leadership, this notifies voters to clear their
        cached grants. This prevents the 60s timeout waiting for lease expiry.

        Returns:
            Number of voters successfully notified
        """
        import aiohttp
        from aiohttp import ClientTimeout

        voter_node_ids = list(getattr(self, "voter_node_ids", []) or [])
        if not voter_node_ids:
            return 0

        # Get current epoch
        lease_epoch = int(getattr(self, "_lease_epoch", 0) or 0)

        cleared_count = 0
        timeout = ClientTimeout(total=3)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for voter_id in voter_node_ids:
                if voter_id == self.node_id:
                    # Release our own grant synchronously
                    self._release_voter_grant_if_self()
                    cleared_count += 1
                    continue

                # Get voter peer info
                peer = self.peers.get(voter_id)
                if not peer or not peer.is_alive():
                    continue

                try:
                    url = self._url_for_peer(peer, "/election/lease_revoke")
                    resp = await session.post(
                        url,
                        json={
                            "leader_id": self.node_id,
                            "epoch": lease_epoch,
                        },
                        headers=self._auth_headers(),
                    )
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("cleared"):
                            cleared_count += 1
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass  # Expected during step-down

        logger.info(f"Notified {cleared_count}/{len(voter_node_ids)} voters of lease revocation")
        return cleared_count

    # _is_leader_lease_valid: Implemented in P2POrchestrator with additional grace period logic

    def _get_voter_quorum_status(self) -> dict[str, Any]:
        """Get detailed voter quorum status for debugging.

        Returns:
            Dict with alive/total voters, quorum met status
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return {"voters": [], "alive": 0, "total": 0, "quorum_met": True}

        quorum = min(VOTER_MIN_QUORUM, len(voters))

        # Use base class helper for getting alive peer list
        alive_voters = self._get_alive_peer_list(voters)

        return {
            "voters": voters,
            "alive": len(alive_voters),
            "alive_list": alive_voters,
            "total": len(voters),
            "quorum_required": quorum,
            "quorum_met": len(alive_voters) >= quorum,
        }

    # =========================================================================
    # Phase 3.2: Dynamic Voter Management (January 2026)
    # =========================================================================

    def _should_promote_voter(self) -> bool:
        """Check if a new voter should be promoted to maintain quorum margin.

        Returns True when:
        - Dynamic voter management is enabled
        - Alive voters are at or below quorum + margin threshold
        - Not using Raft for leadership (Raft has its own membership)

        Returns:
            True if a new voter should be promoted
        """
        if not DYNAMIC_VOTER_ENABLED:
            return False

        # Don't mess with voters when using Raft
        if self._use_raft_for_leadership():
            return False

        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return False

        quorum = min(VOTER_MIN_QUORUM, len(voters))
        threshold = quorum + VOTER_QUORUM_MARGIN

        alive = self._count_alive_peers(voters)
        return alive <= threshold

    def _rank_voter_candidates(self) -> list[dict[str, Any]]:
        """Rank non-voter peers as candidates for voter promotion.

        Candidates are ranked by:
        1. Uptime (longer is better)
        2. Error rate (lower is better)
        3. Connectivity (more peers visible is better)

        Returns:
            List of candidate dicts sorted best-to-worst:
            [{"node_id": str, "score": float, "uptime": float, "error_rate": float}, ...]
        """
        voters = set(getattr(self, "voter_node_ids", []) or [])
        candidates: list[dict[str, Any]] = []
        now = time.time()

        with self.peers_lock:
            peers = dict(self.peers)

        for node_id, peer in peers.items():
            # Skip existing voters
            if node_id in voters:
                continue

            # Skip dead/unhealthy peers
            if not peer.is_alive():
                continue

            # Calculate uptime
            first_seen = getattr(peer, "first_seen", now)
            uptime = now - first_seen

            # Skip peers without minimum uptime
            if uptime < VOTER_MIN_UPTIME_SECONDS:
                continue

            # Get error rate from peer stats (default to 0 if not available)
            error_count = getattr(peer, "error_count", 0)
            request_count = getattr(peer, "request_count", 1)  # Avoid div by 0
            error_rate = error_count / max(request_count, 1)

            # Skip peers with high error rates
            if error_rate > VOTER_MAX_ERROR_RATE:
                continue

            # Get connectivity score (number of peers this node sees)
            peer_count = getattr(peer, "peer_count", 0)

            # Calculate composite score (higher is better)
            # Normalize factors: uptime in hours, 1-error_rate, peer_count
            uptime_score = min(uptime / 3600, 24)  # Cap at 24 hours
            reliability_score = (1.0 - error_rate) * 10
            connectivity_score = min(peer_count, 20)  # Cap at 20 peers

            score = uptime_score + reliability_score + connectivity_score

            candidates.append({
                "node_id": node_id,
                "score": score,
                "uptime": uptime,
                "error_rate": error_rate,
                "peer_count": peer_count,
            })

        # Sort by score descending (best first)
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    async def _promote_to_voter(self, node_id: str) -> bool:
        """Promote a node to voter status.

        This adds the node to voter_node_ids and broadcasts the update
        to the cluster via gossip.

        Args:
            node_id: The node ID to promote

        Returns:
            True if promotion was successful
        """
        # Validate node exists and is healthy
        with self.peers_lock:
            peer = self.peers.get(node_id)
            if not peer or not peer.is_alive():
                self._log_warning(f"Cannot promote {node_id}: peer not found or not alive")
                return False

        # Add to voter list
        if self.voter_node_ids is None:
            self.voter_node_ids = []

        if node_id in self.voter_node_ids:
            self._log_info(f"Node {node_id} is already a voter")
            return True

        self.voter_node_ids.append(node_id)

        # Log and emit event
        self._log_info(
            f"[DynamicVoter] Promoted {node_id} to voter "
            f"(total voters: {len(self.voter_node_ids)})"
        )

        self._safe_emit_event("VOTER_PROMOTED", {
            "node_id": node_id,
            "total_voters": len(self.voter_node_ids),
            "quorum_required": min(VOTER_MIN_QUORUM, len(self.voter_node_ids)),
        })

        # Save state
        if hasattr(self, "_save_state"):
            self._save_state()

        # Broadcast voter change via gossip
        await self._broadcast_voter_change("promote", node_id)

        return True

    async def _maybe_promote_voter(self) -> bool:
        """Check quorum margin and promote a voter if needed.

        This is the main entry point for dynamic voter management.
        Call this periodically (e.g., in membership loop or health check).

        Returns:
            True if a voter was promoted
        """
        if not self._should_promote_voter():
            return False

        # Get ranked candidates
        candidates = self._rank_voter_candidates()
        if not candidates:
            self._log_warning(
                "[DynamicVoter] Quorum at risk but no suitable candidates for promotion"
            )
            return False

        # Promote the best candidate
        best = candidates[0]
        self._log_info(
            f"[DynamicVoter] Promoting {best['node_id']} (score={best['score']:.2f}, "
            f"uptime={best['uptime']:.0f}s, error_rate={best['error_rate']:.2%})"
        )

        return await self._promote_to_voter(best["node_id"])

    async def _broadcast_voter_change(self, action: str, node_id: str) -> int:
        """Broadcast voter list change to all peers.

        Args:
            action: "promote" or "demote"
            node_id: The affected node

        Returns:
            Number of peers successfully notified
        """
        import aiohttp
        from aiohttp import ClientTimeout

        notified = 0
        timeout = ClientTimeout(total=5)

        payload = {
            "action": action,
            "node_id": node_id,
            "voters": list(self.voter_node_ids or []),
            "source_node": self.node_id,
            "timestamp": time.time(),
        }

        with self.peers_lock:
            peers = list(self.peers.values())

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for peer in peers:
                if not peer.is_alive():
                    continue

                try:
                    url = self._url_for_peer(peer, "/election/voter_update")
                    resp = await session.post(
                        url,
                        json=payload,
                        headers=self._auth_headers(),
                    )
                    if resp.status == 200:
                        notified += 1
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    pass  # Expected for some nodes

        self._log_info(f"[DynamicVoter] Notified {notified} peers of voter {action}")
        return notified

    def _check_dynamic_voter_health(self) -> dict[str, Any]:
        """Return health status for dynamic voter management.

        Returns:
            Dict with dynamic voter status and metrics
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        quorum = min(VOTER_MIN_QUORUM, len(voters)) if voters else 0
        alive = self._count_alive_peers(voters) if voters else 0

        candidates = self._rank_voter_candidates() if DYNAMIC_VOTER_ENABLED else []

        return {
            "dynamic_voter_enabled": DYNAMIC_VOTER_ENABLED,
            "total_voters": len(voters),
            "alive_voters": alive,
            "quorum_required": quorum,
            "margin_threshold": quorum + VOTER_QUORUM_MARGIN,
            "should_promote": self._should_promote_voter(),
            "candidate_count": len(candidates),
            "top_candidate": candidates[0] if candidates else None,
        }

    def _check_leader_consistency(self) -> tuple[bool, str]:
        """Check for inconsistent leadership state.

        Returns:
            (is_consistent, reason) - True if state is consistent
        """
        # Import NodeRole lazily
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            from enum import Enum

            class NodeRole(str, Enum):
                LEADER = "leader"
                FOLLOWER = "follower"

        if self.leader_id == self.node_id and self.role != NodeRole.LEADER:
            return False, "leader_id=self but role!=leader"
        if self.leader_id != self.node_id and self.role == NodeRole.LEADER:
            return False, "role=leader but leader_id!=self"
        return True, "consistent"

    def _check_lease_expiry(self) -> dict[str, Any]:
        """Check if the current leader's lease has expired without stepdown.

        January 2, 2026: Added for stale leader alerting (Sprint 3).

        This method checks if the known leader's lease has expired beyond the
        configured grace period. If expired, it emits a LEADER_LEASE_EXPIRED
        event to alert monitoring systems.

        Returns:
            dict with:
            - lease_stale: True if lease expired beyond grace
            - lease_remaining: Seconds until expiry (negative if expired)
            - grace_exceeded_by: Seconds beyond grace (0 if not exceeded)
            - event_emitted: True if LEADER_LEASE_EXPIRED was emitted this check
        """
        now = time.time()
        lease_expires = getattr(self, "leader_lease_expires", 0.0) or 0.0
        leader_id = self.leader_id or ""

        # Calculate time since expiry
        lease_remaining = lease_expires - now
        grace_exceeded_by = max(0, -lease_remaining - LEADER_LEASE_EXPIRY_GRACE_SECONDS)

        result = {
            "lease_stale": False,
            "lease_remaining": lease_remaining,
            "grace_exceeded_by": grace_exceeded_by,
            "event_emitted": False,
            "leader_id": leader_id,
        }

        # Skip if no leader known
        if not leader_id:
            return result

        # Skip if we ARE the leader (we handle our own stepdown)
        if leader_id == self.node_id:
            return result

        # Check if expired beyond grace
        if grace_exceeded_by > 0:
            result["lease_stale"] = True

            # Log warning
            self._log_warning(
                f"Leader {leader_id} lease expired {grace_exceeded_by:.1f}s ago "
                f"(lease_expires={lease_expires:.1f}, grace={LEADER_LEASE_EXPIRY_GRACE_SECONDS}s)"
            )

            # Emit event (async-safe via _safe_emit_event)
            self._safe_emit_event("LEADER_LEASE_EXPIRED", {
                "leader_id": leader_id,
                "lease_expiry_time": lease_expires,
                "current_time": now,
                "grace_seconds": LEADER_LEASE_EXPIRY_GRACE_SECONDS,
                "expired_by_seconds": -lease_remaining,
            })
            result["event_emitted"] = True

        return result

    def _has_voter_consensus_on_leader(self, proposed_leader: str) -> bool:
        """Check if voter quorum agrees on the proposed leader.

        This prevents split-brain scenarios where network partitions cause
        different nodes to see different leaders. Leadership is only valid
        if a quorum of voters agrees on the SAME leader.

        Args:
            proposed_leader: The node ID to validate as leader

        Returns:
            True if quorum of voters agrees on this leader
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return True  # No voters configured = single-node mode

        quorum = min(VOTER_MIN_QUORUM, len(voters))
        voting_for_proposed = 0

        with self.peers_lock:
            peers = dict(self.peers)

        for node_id in voters:
            if node_id == self.node_id:
                # Count self's vote
                if self.leader_id == proposed_leader:
                    voting_for_proposed += 1
            else:
                peer = peers.get(node_id)
                if peer and peer.is_alive() and getattr(peer, "leader_id", None) == proposed_leader:
                    voting_for_proposed += 1

        has_consensus = voting_for_proposed >= quorum
        if not has_consensus:
            self._log_warning(
                f"No consensus on leader {proposed_leader}: "
                f"{voting_for_proposed}/{quorum} voters agree"
            )
        return has_consensus

    def _count_votes_for_leader(self, leader_id: str) -> int:
        """Count how many voters recognize this leader.

        Args:
            leader_id: The leader to count votes for

        Returns:
            Number of voters agreeing on this leader
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return 1

        vote_count = 0
        with self.peers_lock:
            peers = dict(self.peers)

        for node_id in voters:
            if node_id == self.node_id:
                if self.leader_id == leader_id:
                    vote_count += 1
            else:
                peer = peers.get(node_id)
                if peer and peer.is_alive() and getattr(peer, "leader_id", None) == leader_id:
                    vote_count += 1

        return vote_count

    def _detect_split_brain(self) -> dict[str, Any] | None:
        """Detect if cluster is in split-brain state and trigger resolution.

        Split-brain occurs when voters report different leaders.
        This is a critical situation that must be resolved.

        Returns:
            None if no split-brain, otherwise dict with details:
            {
                "leaders_seen": {"leader1": [voter1, voter2], "leader2": [voter3]},
                "severity": "critical" | "warning",
                "recommended_action": "force_election" | "wait"
            }
        """
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return None

        leaders_seen: dict[str, list[str]] = {}

        with self.peers_lock:
            peers = dict(self.peers)

        # Collect leader reports from all alive voters
        for node_id in voters:
            if node_id == self.node_id:
                leader = self.leader_id or ""
            else:
                peer = peers.get(node_id)
                if not peer or not peer.is_alive():
                    continue
                leader = getattr(peer, "leader_id", "") or ""

            if leader:
                if leader not in leaders_seen:
                    leaders_seen[leader] = []
                leaders_seen[leader].append(node_id)

        # Check for split-brain
        if len(leaders_seen) <= 1:
            return None  # No split-brain

        # Multiple leaders detected - this is split-brain
        severity = "critical" if len(leaders_seen) >= 3 else "warning"
        self._log_error(
            f"SPLIT-BRAIN DETECTED: {len(leaders_seen)} different leaders seen: "
            f"{list(leaders_seen.keys())}"
        )

        # Emit SPLIT_BRAIN_DETECTED event
        self._safe_emit_event("SPLIT_BRAIN_DETECTED", {
            "leaders_seen": list(leaders_seen.keys()),
            "voter_count": len(voters),
            "severity": severity,
        })

        # ENFORCEMENT: Trigger resolution immediately
        self._resolve_split_brain(leaders_seen)

        return {
            "leaders_seen": leaders_seen,
            "severity": severity,
            "recommended_action": "force_election" if severity == "critical" else "wait",
        }

    def _resolve_split_brain(self, leaders_seen: dict[str, list[str]]) -> None:
        """Resolve split-brain by demoting self if not the canonical leader.

        Canonical leader is determined by:
        1. Highest peer count (most votes)
        2. Lowest node_id as tiebreaker (deterministic)

        Args:
            leaders_seen: Dict mapping leader_id to list of voters recognizing them
        """
        # Import NodeRole lazily
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            from enum import Enum

            class NodeRole(str, Enum):
                LEADER = "leader"
                FOLLOWER = "follower"

        # Only relevant if we think we're the leader
        if self.role != NodeRole.LEADER:
            return

        # Find canonical leader: highest vote count, then lowest node_id
        canonical_leader = None
        max_votes = 0
        for leader_id, voters in leaders_seen.items():
            vote_count = len(voters)
            if vote_count > max_votes or (vote_count == max_votes and (
                canonical_leader is None or leader_id < canonical_leader
            )):
                canonical_leader = leader_id
                max_votes = vote_count

        if canonical_leader is None:
            return

        # If we're not the canonical leader, step down
        if canonical_leader != self.node_id:
            self._log_warning(
                f"SPLIT-BRAIN RESOLUTION: Demoting self ({self.node_id}) in favor of "
                f"canonical leader {canonical_leader} (has {max_votes} votes)"
            )

            # Step down from leadership
            self.role = NodeRole.FOLLOWER
            self.leader_id = canonical_leader
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
            self.last_lease_renewal = 0.0
            self._release_voter_grant_if_self()

            # Save state and emit event
            if hasattr(self, "_save_state"):
                self._save_state()

            self._safe_emit_event("SPLIT_BRAIN_RESOLVED", {
                "demoted_node": self.node_id,
                "canonical_leader": canonical_leader,
                "canonical_votes": max_votes,
            })
        else:
            self._log_info(
                f"SPLIT-BRAIN RESOLUTION: This node ({self.node_id}) is the canonical leader"
            )

    # =========================================================================
    # P5.4: Raft Leader Integration (Dec 30, 2025)
    # =========================================================================

    def _use_raft_for_leadership(self) -> bool:
        """Check if Raft should be used for leadership decisions.

        Returns True when:
        - CONSENSUS_MODE is "raft" (not "bully" or "hybrid")
        - Raft is enabled and initialized
        - This node is a voter (non-voters use bully as fallback)

        In "hybrid" mode, Raft handles work queue but bully handles leadership.
        In "raft" mode, Raft leader is the authoritative P2P leader.

        Returns:
            True if Raft should determine leadership
        """
        # Only use Raft for leadership in "raft" mode (not "hybrid")
        if CONSENSUS_MODE != "raft":
            return False

        # Must have Raft enabled and initialized
        if not RAFT_ENABLED:
            return False

        # Check if Raft is initialized (set by ConsensusMixin)
        if not getattr(self, "_raft_initialized", False):
            return False

        # Only voters should use Raft for leadership
        # Non-voters use bully algorithm as fallback
        if self.node_id not in (self.voter_node_ids or []):
            return False

        return True

    def _get_raft_leader_node_id(self) -> str | None:
        """Get the node ID of the current Raft leader.

        Extracts the Raft leader address and maps it back to a node ID
        by looking up peers with matching addresses.

        Returns:
            Node ID of the Raft leader, or None if unavailable
        """
        raft_wq = getattr(self, "_raft_work_queue", None)
        if raft_wq is None:
            return None

        try:
            leader_addr = raft_wq._getLeader()
            if leader_addr is None:
                return None

            leader_addr_str = str(leader_addr)

            # Check if we are the leader
            advertise_host = getattr(self, "advertise_host", "")
            raft_port = 4321  # Default from constants
            try:
                from scripts.p2p.constants import RAFT_BIND_PORT
                raft_port = RAFT_BIND_PORT
            except ImportError:
                pass

            self_addr = f"{advertise_host}:{raft_port}"
            if leader_addr_str == self_addr:
                return self.node_id

            # Look up peer by Raft address
            # The address is in format "host:raft_port"
            leader_host = leader_addr_str.rsplit(":", 1)[0]

            with self.peers_lock:
                for node_id, peer in self.peers.items():
                    # Check tailscale_ip or host
                    peer_ip = getattr(peer, "tailscale_ip", None) or getattr(peer, "host", None)
                    if peer_ip == leader_host:
                        return node_id

            self._log_debug(f"Could not map Raft leader address {leader_addr_str} to node ID")
            return None

        except Exception as e:
            self._log_debug(f"Error getting Raft leader: {e}")
            return None

    def _sync_leader_from_raft(self) -> bool:
        """Synchronize P2P leader with Raft leader.

        When Raft is used for leadership, this method updates the P2P
        leader_id to match the Raft leader. This ensures work queue
        operations and leadership are consistent.

        Should be called periodically (e.g., in health check or membership loop).

        Returns:
            True if leader was synced/updated, False if no change or error
        """
        if not self._use_raft_for_leadership():
            return False

        raft_leader_id = self._get_raft_leader_node_id()
        if raft_leader_id is None:
            # No Raft leader elected yet - don't change anything
            return False

        # Check if leader changed
        current_leader = self.leader_id
        if current_leader == raft_leader_id:
            return False  # No change

        # Import NodeRole lazily
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            from enum import Enum

            class NodeRole(str, Enum):
                LEADER = "leader"
                FOLLOWER = "follower"

        old_leader = self.leader_id
        self.leader_id = raft_leader_id

        # Update our role based on whether we're the Raft leader
        if raft_leader_id == self.node_id:
            if self.role != NodeRole.LEADER:
                self._log_info(f"[Raft] Becoming leader (Raft leader elected)")
                self.role = NodeRole.LEADER
                self.leader_lease_expires = time.time() + 300  # Raft handles leases
                self._safe_emit_event("LEADER_ELECTED", {
                    "leader_id": self.node_id,
                    "source": "raft",
                })
        else:
            if self.role == NodeRole.LEADER:
                self._log_info(f"[Raft] Stepping down, new leader: {raft_leader_id}")
                self.role = NodeRole.FOLLOWER
                self.leader_lease_id = ""
                self.leader_lease_expires = 0.0
                self._release_voter_grant_if_self()

        # Save state
        if hasattr(self, "_save_state"):
            self._save_state()

        # Emit leader change event
        self._safe_emit_event("P2P_LEADER_CHANGED", {
            "old_leader": old_leader,
            "new_leader": raft_leader_id,
            "source": "raft_sync",
        })

        self._log_info(f"[Raft] Leader synced from Raft: {old_leader} -> {raft_leader_id}")
        return True

    def get_authoritative_leader(self) -> str | None:
        """Get the authoritative leader ID, checking Raft first.

        This is the preferred method for getting the current leader.
        It checks Raft leader first (if using Raft for leadership),
        then falls back to the bully-elected leader.

        Returns:
            Node ID of the authoritative leader, or None if no leader
        """
        if self._use_raft_for_leadership():
            raft_leader = self._get_raft_leader_node_id()
            if raft_leader is not None:
                return raft_leader

        # Fall back to bully-elected leader
        return self.leader_id

    def election_health_check(self) -> dict[str, Any]:
        """Return health status for leader election subsystem.

        Also runs split-brain detection as a side effect, triggering automatic
        resolution if multiple leaders are detected.

        December 2025: Added split-brain detection integration to ensure it runs
        periodically with health checks, enabling automatic resolution.

        December 30, 2025 (P5.4): Added Raft leader sync. When using Raft for
        leadership, syncs P2P leader with Raft leader on each health check.

        January 2026 (Phase 3.2): Added dynamic voter management. When enabled,
        automatically promotes healthy nodes to voter status when quorum is at risk.

        Returns:
            dict with is_healthy, role, leader_id, quorum status, split_brain status
        """
        # December 30, 2025 (P5.4): Sync leader from Raft if applicable
        # This ensures P2P leader stays in sync with Raft leader
        raft_leader_synced = self._sync_leader_from_raft()
        using_raft = self._use_raft_for_leadership()

        has_quorum = self._has_voter_quorum()
        voter_count = len(self.voter_node_ids) if self.voter_node_ids else 0
        alive_voters = self._count_alive_peers(self.voter_node_ids or [])
        lease_remaining = max(0, self.leader_lease_expires - time.time())

        # December 2025: Run split-brain detection (triggers resolution if needed)
        # Skip split-brain detection when using Raft - Raft handles this
        split_brain_info = None
        if not using_raft:
            split_brain_info = self._detect_split_brain()
        has_split_brain = split_brain_info is not None

        # January 2026 (Phase 3.2): Dynamic voter management health info
        dynamic_voter_info = self._check_dynamic_voter_health()

        # January 2026 (Sprint 3): Check for stale leader lease expiry
        lease_expiry_info = self._check_lease_expiry()
        has_stale_leader = lease_expiry_info.get("lease_stale", False)

        # Unhealthy if:
        # 1. No quorum and we should have voters (unless using Raft), OR
        # 2. Split-brain detected (critical severity), OR
        # 3. Leader lease expired without stepdown (stale leader)
        # When using Raft, quorum is handled by Raft consensus
        if using_raft:
            is_healthy = not has_split_brain and not has_stale_leader
        else:
            is_healthy = (has_quorum or voter_count == 0) and not has_split_brain and not has_stale_leader

        result = {
            "is_healthy": is_healthy,
            "role": str(self.role) if self.role else "unknown",
            "leader_id": self.leader_id,
            "authoritative_leader": self.get_authoritative_leader(),
            "has_quorum": has_quorum,
            "voter_count": voter_count,
            "alive_voters": alive_voters,
            "lease_remaining_seconds": lease_remaining,
            "split_brain_detected": has_split_brain,
            "split_brain_info": split_brain_info,
            # P5.4: Raft leader integration status
            "using_raft_for_leadership": using_raft,
            "raft_leader_synced": raft_leader_synced,
            # Phase 3.2: Dynamic voter management
            "dynamic_voter": dynamic_voter_info,
            # Sprint 3: Stale leader alerting
            "stale_leader_detected": has_stale_leader,
            "lease_expiry_info": lease_expiry_info,
        }

        # Add Raft leader info if available
        if using_raft:
            result["raft_leader_node_id"] = self._get_raft_leader_node_id()

        return result

    def health_check(self) -> dict[str, Any]:
        """Return health status for leader election mixin (DaemonManager integration).

        December 2025: Added for unified health check interface.
        Uses base class helper for standardized response format.

        Returns:
            dict with healthy status, message, and details
        """
        status = self.election_health_check()
        is_healthy = status.get("is_healthy", False)
        role = status.get("role", "unknown")
        leader = status.get("leader_id", "none")
        message = f"Election (role={role}, leader={leader})" if is_healthy else "No quorum"
        return self._build_health_response(is_healthy, message, status)


# Convenience functions for external use
def check_quorum(
    voters: list[str],
    alive_peers: dict[str, Any],
    self_node_id: str,
) -> bool:
    """Standalone quorum check function.

    Args:
        voters: List of voter node IDs
        alive_peers: Dict of alive peer NodeInfo objects
        self_node_id: This node's ID

    Returns:
        True if quorum is met
    """
    if not voters:
        return True

    quorum = min(VOTER_MIN_QUORUM, len(voters))
    alive = 0
    for node_id in voters:
        if node_id == self_node_id:
            alive += 1
            continue
        peer = alive_peers.get(node_id)
        if peer and hasattr(peer, "is_alive") and peer.is_alive():
            alive += 1
    return alive >= quorum
