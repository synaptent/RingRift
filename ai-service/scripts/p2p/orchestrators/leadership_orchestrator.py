"""Leadership Orchestrator - Handles leader election and consensus.

January 2026: Created as part of P2POrchestrator decomposition.

Responsibilities:
- Leader election state machine (CANDIDATE → LEADER → STEPPING_DOWN → FOLLOWER)
- Fence token management (generation, validation, refresh)
- Lease management (epoch, validation)
- Leadership consistency validation
- Leadership desync recovery
- Incumbent grace period (prevent flapping)
- Broadcast leadership claims via gossip
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from scripts.p2p.orchestrators.base_orchestrator import BaseOrchestrator, HealthCheckResult

if TYPE_CHECKING:
    from aiohttp import web
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)


class LeadershipOrchestrator(BaseOrchestrator):
    """Orchestrator for leader election and consensus management.

    This orchestrator handles all aspects of leadership in the P2P cluster:
    - Determining and setting the current leader
    - Managing fence tokens to prevent split-brain
    - Handling leader leases and their renewal
    - Broadcasting leadership claims to peers
    - Recovering from leadership desync situations

    The orchestrator delegates to existing managers:
    - state_manager: For persistent state
    - quorum_manager: For voter management
    - leader_probe_loop: For health monitoring

    Usage:
        # In P2POrchestrator.__init__:
        self.leadership = LeadershipOrchestrator(self)

        # Check leadership:
        if self.leadership.is_leader:
            ...

        # Get fence token:
        token = self.leadership.get_fence_token()
    """

    # Grace period constants
    INCUMBENT_GRACE_PERIOD_SECONDS = 60.0  # Prevent leader flapping
    RECENT_LEADER_WINDOW_SECONDS = 300.0  # Consider "recently leader" for 5 min

    def __init__(self, p2p: "P2POrchestrator") -> None:
        """Initialize the leadership orchestrator.

        Args:
            p2p: The parent P2POrchestrator instance.
        """
        super().__init__(p2p)

        # Leadership state (mirrors p2p state for now, will migrate)
        self._last_leader_change_time: float = 0.0
        self._last_election_time: float = 0.0
        self._election_in_progress: bool = False
        self._provisional_leader: bool = False
        self._leader_claim_broadcast_count: int = 0

    @property
    def name(self) -> str:
        """Return the name of this orchestrator."""
        return "leadership"

    def health_check(self) -> HealthCheckResult:
        """Check the health of leadership orchestrator.

        Returns:
            HealthCheckResult with leadership status details.
        """
        try:
            # Get leadership state from parent orchestrator
            if hasattr(self._p2p, "is_leader") and callable(self._p2p.is_leader):
                is_leader = self._p2p.is_leader()
            else:
                is_leader = getattr(self._p2p, "is_leader", False)
            leader_id = getattr(self._p2p, "leader_id", None)
            role = getattr(self._p2p, "role", None)

            # Check for potential issues
            issues = []

            # Check lease validity
            lease_valid = self._is_leader_lease_valid()
            if is_leader and not lease_valid:
                issues.append("Leader with invalid lease")

            # Check for leadership consistency
            if hasattr(self._p2p, "_get_leadership_consistency_metrics"):
                metrics = self._p2p._get_leadership_consistency_metrics()
                if not metrics.get("is_consistent", True):
                    issues.append(f"Leadership inconsistency: {metrics.get('reason', 'unknown')}")

            healthy = len(issues) == 0
            message = "Leadership healthy" if healthy else "; ".join(issues)

            return HealthCheckResult(
                healthy=healthy,
                message=message,
                details={
                    "is_leader": is_leader,
                    "leader_id": leader_id,
                    "role": str(role) if role else None,
                    "lease_valid": lease_valid,
                    "election_in_progress": self._election_in_progress,
                    "provisional_leader": self._provisional_leader,
                    "issues": issues,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
            )

    # =========================================================================
    # Leadership State - Delegated to P2POrchestrator for now
    # These will be migrated incrementally
    # =========================================================================

    def is_leader(self) -> bool:
        """Check if this node is the current cluster leader.

        Returns:
            True if this node is the leader with a valid lease.
        """
        if hasattr(self._p2p, "is_leader") and callable(self._p2p.is_leader):
            return self._p2p.is_leader()
        return getattr(self._p2p, "is_leader", False)

    def get_leader_id(self) -> str | None:
        """Get the current leader's node ID.

        Returns:
            The leader's node ID, or None if no leader.
        """
        return getattr(self._p2p, "leader_id", None)

    def get_fence_token(self) -> str:
        """Get the current fence token for split-brain prevention.

        Jan 28, 2026: Implementation moved from P2POrchestrator.

        Returns:
            The current fence token string, or empty if not leader.
        """
        # Try NodeRole import for role check
        try:
            from scripts.p2p.node_info import NodeRole
            if getattr(self._p2p, "role", None) != NodeRole.LEADER:
                return ""
        except ImportError:
            # Fallback: check role string
            role = getattr(self._p2p, "role", None)
            if role is None or str(role) != "leader":
                return ""

        return getattr(self._p2p, "_fence_token", "")

    def get_lease_epoch(self) -> int:
        """Get the current leader lease epoch.

        Jan 28, 2026: Implementation moved from P2POrchestrator.

        Returns:
            The current lease epoch number (0 if never been leader).
        """
        return getattr(self._p2p, "_lease_epoch", 0)

    def validate_fence_token(self, token: str) -> tuple[bool, str]:
        """Validate an incoming fence token from a claimed leader.

        Jan 28, 2026: Implementation moved from P2POrchestrator.

        Workers use this to reject commands from stale leaders.
        A token is valid if:
        1. It's from the current known leader
        2. Its epoch is >= our known epoch

        Args:
            token: Fence token to validate (format: node_id:epoch:timestamp)

        Returns:
            Tuple of (is_valid, reason).
        """
        if not token:
            return False, "empty_fence_token"

        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False, "malformed_token"

            token_node_id = parts[0]
            token_epoch = int(parts[1])

            # Check if token is from known leader
            leader_id = getattr(self._p2p, "leader_id", None)
            if leader_id and token_node_id != leader_id:
                return False, f"token_from_unknown_leader:{token_node_id}"

            # Check epoch - reject if lower than what we've seen
            last_seen_epoch = getattr(self._p2p, "_last_seen_epoch", 0)
            if token_epoch < last_seen_epoch:
                return False, f"stale_epoch:{token_epoch}<{last_seen_epoch}"

            # Update last seen epoch on P2P
            self._p2p._last_seen_epoch = max(last_seen_epoch, token_epoch)

            return True, "valid"

        except (ValueError, IndexError) as e:
            return False, f"parse_error:{e}"

    def _is_leader_lease_valid(self) -> bool:
        """Check if the current leader lease is valid.

        Returns:
            True if the lease is valid.
        """
        if hasattr(self._p2p, "_is_leader_lease_valid"):
            return self._p2p._is_leader_lease_valid()
        # Fallback: check lease expiry directly
        lease_expires = getattr(self._p2p, "leader_lease_expires", 0)
        return time.time() < lease_expires

    # =========================================================================
    # Leadership Consistency
    # =========================================================================

    def get_consistency_metrics(self) -> dict[str, Any]:
        """Get metrics for detecting leadership state desyncs.

        Jan 28, 2026: Implementation moved from P2POrchestrator.
        Used to monitor and debug the leader self-recognition bug
        where leader_id is set correctly but role doesn't match.

        Returns:
            Dictionary with consistency check results for monitoring.
        """
        try:
            from scripts.p2p.leadership_state_machine import LeaderState
        except ImportError:
            LeaderState = None

        try:
            from scripts.p2p.node_info import NodeRole
        except ImportError:
            NodeRole = None

        # Get ULSM state if available
        ulsm_state = None
        ulsm_leader = None
        leadership_sm = getattr(self._p2p, "_leadership_sm", None)
        if leadership_sm is not None:
            if LeaderState is not None:
                ulsm_state = (
                    leadership_sm._state.value
                    if hasattr(leadership_sm._state, "value")
                    else str(leadership_sm._state)
                )
            ulsm_leader = leadership_sm._leader_id

        # Check for inconsistencies
        role_ulsm_mismatch = False
        leader_ulsm_mismatch = False

        role = getattr(self._p2p, "role", None)
        leader_id = getattr(self._p2p, "leader_id", None)
        node_id = getattr(self._p2p, "node_id", "")

        if leadership_sm is not None and LeaderState is not None and NodeRole is not None:
            # Role should match ULSM state
            local_is_leader = role in (NodeRole.LEADER, NodeRole.PROVISIONAL_LEADER)
            ulsm_is_leader = leadership_sm._state == LeaderState.LEADER
            role_ulsm_mismatch = (
                (local_is_leader != ulsm_is_leader)
                and leadership_sm._state != LeaderState.STEPPING_DOWN
            )
            # Leader IDs should match
            leader_ulsm_mismatch = leadership_sm._leader_id != leader_id

        # Self-recognition check: If we're the elected leader, do we recognize it?
        leader_id_is_self = leader_id == node_id
        role_is_leader = False
        if NodeRole is not None:
            role_is_leader = role in (NodeRole.LEADER, NodeRole.PROVISIONAL_LEADER)
        is_leader_call = self.is_leader()

        # Desync conditions
        gossip_desync = leader_id_is_self and not role_is_leader
        role_desync = role_is_leader and not leader_id_is_self

        return {
            "role": role.value if hasattr(role, "value") else str(role),
            "leader_id": leader_id,
            "node_id": node_id,
            "is_leader_call": is_leader_call,
            "leader_id_is_self": leader_id_is_self,
            "role_is_leader": role_is_leader,
            "ulsm_state": ulsm_state,
            "ulsm_leader_id": ulsm_leader,
            "role_ulsm_mismatch": role_ulsm_mismatch,
            "leader_ulsm_mismatch": leader_ulsm_mismatch,
            "gossip_desync": gossip_desync,
            "role_desync": role_desync,
            "self_recognition_ok": leader_id_is_self == is_leader_call,
            "is_consistent": not (gossip_desync or role_desync or role_ulsm_mismatch),
        }

    def recover_leadership_desync(self) -> bool:
        """Auto-recover from leadership state desynchronization.

        Jan 28, 2026: Implementation moved from P2POrchestrator.

        Recovery actions:
        1. gossip_desync (leader_id=self but role!=LEADER):
           → Accept leadership since other nodes already see us as leader
        2. role_desync (role=LEADER but leader_id!=self):
           → Step down since another node is the recognized leader

        Returns:
            True if recovery action was taken, False if state was consistent.
        """
        try:
            from scripts.p2p.node_info import NodeRole
        except ImportError:
            self._log_warning("Cannot recover desync: NodeRole not available")
            return False

        leader_id = getattr(self._p2p, "leader_id", None)
        node_id = getattr(self._p2p, "node_id", "")
        role = getattr(self._p2p, "role", None)

        leader_id_is_self = leader_id == node_id
        role_is_leader = role in (NodeRole.LEADER, NodeRole.PROVISIONAL_LEADER)

        # Case 1: gossip_desync - leader_id=self but role!=LEADER
        if leader_id_is_self and not role_is_leader:
            self._log_warning(
                f"[LeadershipRecovery] Fixing gossip_desync: "
                f"leader_id={leader_id}, role={role} -> LEADER"
            )
            self._p2p.role = NodeRole.LEADER
            # Update state machine if available
            leadership_sm = getattr(self._p2p, "_leadership_sm", None)
            if leadership_sm:
                try:
                    from scripts.p2p.leadership_state_machine import LeaderState
                    leadership_sm._state = LeaderState.LEADER
                    leadership_sm._leader_id = node_id
                except Exception as e:
                    self._log_warning(f"[LeadershipRecovery] Failed to update state machine: {e}")
            return True

        # Case 2: role_desync - role=LEADER but leader_id!=self
        if role_is_leader and not leader_id_is_self:
            self._log_warning(
                f"[LeadershipRecovery] Fixing role_desync: "
                f"role={role}, leader_id={leader_id} -> FOLLOWER"
            )
            self._p2p.role = NodeRole.FOLLOWER
            # Update state machine if available
            leadership_sm = getattr(self._p2p, "_leadership_sm", None)
            if leadership_sm:
                try:
                    from scripts.p2p.leadership_state_machine import LeaderState
                    leadership_sm._state = LeaderState.FOLLOWER
                    leadership_sm._leader_id = leader_id
                except Exception as e:
                    self._log_warning(f"[LeadershipRecovery] Failed to update state machine: {e}")
            return True

        return False  # State was consistent

    def reconcile_leadership_state(self) -> bool:
        """Reconcile ULSM state with gossip consensus.

        Jan 28, 2026: Implementation moved from P2POrchestrator.

        Addresses the issue where gossip leader_consensus disagrees with ULSM state.
        Called periodically from heartbeat loop.

        Returns:
            True if reconciliation action was taken, False otherwise.
        """
        leadership_sm = getattr(self._p2p, "_leadership_sm", None)
        if not leadership_sm:
            return False

        try:
            from scripts.p2p.leadership_state_machine import LeaderState
            from scripts.p2p.node_info import NodeRole
        except ImportError:
            return False

        # Get gossip consensus on who the leader is
        if hasattr(self._p2p, "_get_cluster_leader_consensus"):
            consensus_info = self._p2p._get_cluster_leader_consensus()
        else:
            return False

        consensus_leader = consensus_info.get("consensus_leader")
        total_voters = consensus_info.get("total_voters", 0)
        agreement = consensus_info.get("leader_agreement", 0)

        # Need at least 2 voters for meaningful consensus
        if total_voters < 2:
            return False

        consensus_ratio = agreement / total_voters if total_voters > 0 else 0
        MIN_CONSENSUS_THRESHOLD = 0.50

        node_id = getattr(self._p2p, "node_id", "")
        ulsm_state = leadership_sm._state

        # Low consensus - try to help build it
        if consensus_ratio < MIN_CONSENSUS_THRESHOLD:
            if ulsm_state == LeaderState.LEADER:
                self._log_info(
                    f"[LeaderReconciliation] Low consensus ({consensus_ratio:.0%}), "
                    f"proactively announcing leadership"
                )
                self.broadcast_leadership_claim()
            return False

        # Case 1: Gossip says we're leader, ULSM doesn't know
        if consensus_leader == node_id and ulsm_state != LeaderState.LEADER:
            if self._is_leader_lease_valid():
                self._log_info(
                    f"[LeaderReconciliation] Accepting leadership from gossip "
                    f"(agreement={agreement}/{total_voters}={consensus_ratio:.0%})"
                )
                leadership_sm._state = LeaderState.LEADER
                leadership_sm._leader_id = node_id
                self._p2p.role = NodeRole.LEADER
                self._p2p.leader_id = node_id
                return True
            else:
                self._log_warning(
                    "[LeaderReconciliation] Gossip says we're leader but lease invalid"
                )
                self._p2p.leader_id = None
                return True

        # Case 2: ULSM says leader, gossip disagrees
        if ulsm_state == LeaderState.LEADER and consensus_leader and consensus_leader != node_id:
            self._log_warning(
                f"[LeaderReconciliation] Gossip says {consensus_leader} is leader "
                f"(agreement={agreement}/{total_voters}={consensus_ratio:.0%}); stepping down"
            )
            if hasattr(self._p2p, "_schedule_step_down_sync"):
                try:
                    from scripts.p2p.leadership_state_machine import TransitionReason
                    self._p2p._schedule_step_down_sync(TransitionReason.HIGHER_EPOCH_SEEN)
                except ImportError:
                    self._p2p.role = NodeRole.FOLLOWER
            return True

        return False

    # =========================================================================
    # Leadership Broadcasting
    # =========================================================================

    def broadcast_leadership_claim(self) -> None:
        """Broadcast a leadership claim to all peers."""
        if hasattr(self._p2p, "_broadcast_leadership_claim"):
            self._p2p._broadcast_leadership_claim()
            self._leader_claim_broadcast_count += 1

    async def async_broadcast_leader_claim(self) -> None:
        """Asynchronously broadcast a leadership claim to all peers."""
        if hasattr(self._p2p, "_async_broadcast_leader_claim"):
            await self._p2p._async_broadcast_leader_claim()
            self._leader_claim_broadcast_count += 1

    # =========================================================================
    # Grace Period Management
    # =========================================================================

    def was_recently_leader(self) -> bool:
        """Check if this node was the cluster leader within RECENT_LEADER_WINDOW.

        Jan 28, 2026: Implementation moved from P2POrchestrator.
        Leader stickiness - allows previous leader to reclaim leadership
        with preference during elections, reducing oscillation.

        Returns:
            True if we were leader and stepped down within RECENT_LEADER_WINDOW_SECONDS.
        """
        now = time.time()
        # Check if we became leader at some point
        last_become_leader = getattr(self._p2p, "_last_become_leader_time", 0.0)
        if last_become_leader <= 0:
            return False
        # Check if we stepped down recently (within window)
        last_step_down = getattr(self._p2p, "_last_step_down_time", 0.0)
        if last_step_down <= 0:
            return False
        time_since_step_down = now - last_step_down
        return time_since_step_down < self.RECENT_LEADER_WINDOW_SECONDS

    def in_incumbent_grace_period(self) -> bool:
        """Check if we're within the incumbent grace period after stepping down.

        Jan 28, 2026: Implementation moved from P2POrchestrator.
        During this period, the previous leader gets priority to reclaim
        leadership without competition.

        Returns:
            True if within INCUMBENT_GRACE_PERIOD_SECONDS of step-down.
        """
        last_step_down = getattr(self._p2p, "_last_step_down_time", 0.0)
        if last_step_down <= 0:
            return False
        time_since_step_down = time.time() - last_step_down
        return time_since_step_down < self.INCUMBENT_GRACE_PERIOD_SECONDS

    # =========================================================================
    # Election Management
    # =========================================================================

    async def start_election(self, reason: str = "manual") -> None:
        """Start a leader election.

        Args:
            reason: Human-readable reason for the election.
        """
        if self._election_in_progress:
            self._log_warning(f"Election already in progress, skipping: {reason}")
            return

        self._election_in_progress = True
        self._last_election_time = time.time()

        try:
            if hasattr(self._p2p, "_start_election"):
                await self._p2p._start_election()
        finally:
            self._election_in_progress = False

    async def become_leader(self) -> None:
        """Transition this node to leader state."""
        if hasattr(self._p2p, "_become_leader"):
            await self._p2p._become_leader()

    async def request_election_from_voters(self, reason: str = "non_voter_request") -> bool:
        """Request an election from voter nodes.

        Args:
            reason: Reason for the election request.

        Returns:
            True if election was triggered.
        """
        if hasattr(self._p2p, "_request_election_from_voters"):
            return await self._p2p._request_election_from_voters(reason)
        return False

    # =========================================================================
    # Leader Queries
    # =========================================================================

    def get_leader_peer(self) -> Any | None:
        """Get the NodeInfo for the current leader.

        Returns:
            NodeInfo of the leader, or None if no leader.
        """
        if hasattr(self._p2p, "_get_leader_peer"):
            return self._p2p._get_leader_peer()
        return None

    async def determine_leased_leader_from_voters(self) -> str | None:
        """Query voters to determine the leased leader.

        Returns:
            The leader ID according to voter consensus, or None.
        """
        if hasattr(self._p2p, "_determine_leased_leader_from_voters"):
            return await self._p2p._determine_leased_leader_from_voters()
        return None

    def count_peers_reporting_leader(self, leader_id: str) -> int:
        """Count how many peers report a specific node as leader.

        Args:
            leader_id: The leader ID to check.

        Returns:
            Number of peers reporting this node as leader.
        """
        if hasattr(self._p2p, "_count_peers_reporting_leader"):
            return self._p2p._count_peers_reporting_leader(leader_id)
        return 0

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_leader_hint(self) -> dict[str, Any]:
        """Get leader hint information for peer responses.

        Returns:
            Dictionary with leader_id and other hints.
        """
        if hasattr(self._p2p, "_get_leader_hint"):
            return self._p2p._get_leader_hint()
        return {"leader_id": self.get_leader_id()}

    def get_cluster_leader_consensus(self) -> dict[str, Any]:
        """Get cluster-wide leader consensus information.

        Returns:
            Dictionary with consensus status and voting details.
        """
        if hasattr(self._p2p, "_get_cluster_leader_consensus"):
            return self._p2p._get_cluster_leader_consensus()
        return {"leader_id": self.get_leader_id(), "consensus": "unknown"}
