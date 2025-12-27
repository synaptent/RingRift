"""Leader Election Logic Mixin.

Extracted from p2p_orchestrator.py for modularity.
This mixin provides core leader election and voter quorum logic.

Usage:
    class P2POrchestrator(LeaderElectionMixin, ...):
        pass

Phase 2.2 extraction - Dec 26, 2025
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from threading import RLock

    from scripts.p2p.models import NodeInfo
    from scripts.p2p.types import NodeRole

logger = logging.getLogger(__name__)

# Import constants with fallbacks
try:
    from scripts.p2p.constants import VOTER_MIN_QUORUM
except ImportError:
    VOTER_MIN_QUORUM = 3


class LeaderElectionMixin:
    """Mixin providing core leader election logic.

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

        alive = 0
        with self.peers_lock:
            peers = dict(self.peers)
        for node_id in voters:
            if node_id == self.node_id:
                alive += 1
                continue
            peer = peers.get(node_id)
            if peer and peer.is_alive():
                alive += 1
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
        alive_voters = []

        with self.peers_lock:
            peers = dict(self.peers)
        for node_id in voters:
            if node_id == self.node_id:
                alive_voters.append(node_id)
                continue
            peer = peers.get(node_id)
            if peer and peer.is_alive():
                alive_voters.append(node_id)

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
