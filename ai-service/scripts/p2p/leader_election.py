"""Leader Election Logic Mixin.

Extracted from p2p_orchestrator.py for modularity.
This mixin provides core leader election and voter quorum logic.

Usage:
    class P2POrchestrator(LeaderElectionMixin, ...):
        pass

Phase 2.2 extraction - Dec 26, 2025
Refactored to use P2PMixinBase - Dec 27, 2025
"""

from __future__ import annotations

import logging
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
})

VOTER_MIN_QUORUM = _CONSTANTS["VOTER_MIN_QUORUM"]


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
        """Detect if cluster is in split-brain state.

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

        return {
            "leaders_seen": leaders_seen,
            "severity": severity,
            "recommended_action": "force_election" if severity == "critical" else "wait",
        }


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
