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
import contextlib
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
    # Core Leadership Methods
    # Jan 29, 2026: Moved from P2POrchestrator
    # =========================================================================

    def set_leader(
        self,
        new_leader_id: str | None,
        reason: str = "unknown",
        *,
        sync_to_ulsm: bool = True,
        save_state: bool = True,
    ) -> bool:
        """Atomically set the leader and role to ensure consistency.

        Jan 29, 2026: Implementation moved from P2POrchestrator._set_leader().

        This is the CANONICAL method for modifying leader_id and role.
        Using this method ensures both tracking systems (direct fields and ULSM)
        stay synchronized and prevents the leader self-recognition desync bug.

        Args:
            new_leader_id: The new leader ID (None to clear leader)
            reason: Human-readable reason for logging/debugging
            sync_to_ulsm: Whether to sync state to LeadershipStateMachine
            save_state: Whether to persist state after change

        Returns:
            True if this node is now the leader
        """
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            self._log_warning("Cannot set leader: NodeRole not available")
            return False

        # C1 fix: Acquire lock to prevent race conditions during leader transitions
        leader_state_lock = getattr(self._p2p, "leader_state_lock", None)
        if leader_state_lock is None:
            self._log_warning("No leader_state_lock available, proceeding without lock")
            return self._set_leader_unlocked(new_leader_id, reason, sync_to_ulsm, save_state)

        with leader_state_lock:
            return self._set_leader_unlocked(new_leader_id, reason, sync_to_ulsm, save_state)

    def _set_leader_unlocked(
        self,
        new_leader_id: str | None,
        reason: str,
        sync_to_ulsm: bool,
        save_state: bool,
    ) -> bool:
        """Internal implementation of set_leader (called with lock held).

        Jan 29, 2026: Helper for set_leader().
        """
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            return False

        old_leader_id = getattr(self._p2p, "leader_id", None)
        old_role = getattr(self._p2p, "role", None)
        node_id = getattr(self._p2p, "node_id", "")

        # Determine new role based on leader_id
        if new_leader_id is None:
            new_role = NodeRole.FOLLOWER
            is_now_leader = False
        elif new_leader_id == node_id:
            new_role = NodeRole.LEADER
            is_now_leader = True
        else:
            new_role = NodeRole.FOLLOWER
            is_now_leader = False

        # Atomic update of both fields
        self._p2p.leader_id = new_leader_id
        self._p2p.role = new_role

        # Sync to ULSM (Unified Leadership State Machine) if enabled
        if sync_to_ulsm:
            leadership_sm = getattr(self._p2p, "_leadership_sm", None)
            if leadership_sm is not None:
                try:
                    from scripts.p2p.leadership_state_machine import LeaderState

                    leadership_sm._leader_id = new_leader_id
                    leadership_sm._state = (
                        LeaderState.LEADER if is_now_leader else LeaderState.FOLLOWER
                    )
                except (ImportError, AttributeError) as e:
                    self._log_debug(f"[LeaderSet] ULSM sync skipped: {e}")

        # Log if changed
        if old_leader_id != new_leader_id or old_role != new_role:
            self._log_info(
                f"[LeaderSet] {old_role.value if hasattr(old_role, 'value') else old_role}->"
                f"{new_role.value if hasattr(new_role, 'value') else new_role}, "
                f"leader_id={old_leader_id}->{new_leader_id}, reason={reason}"
            )

        # Persist state if requested
        if save_state and (old_leader_id != new_leader_id or old_role != new_role):
            if hasattr(self._p2p, "_save_state"):
                self._p2p._save_state()

        return is_now_leader

    def check_is_leader(self) -> bool:
        """Check if this node is the current cluster leader with valid lease.

        Jan 29, 2026: Implementation moved from P2POrchestrator._is_leader().

        Routes through Raft consensus when enabled, with fallback chain:
        1. consensus_mixin.is_raft_leader() - Deferred Raft init
        2. HybridCoordinator.is_leader() - May fall back to Bully
        3. Bully algorithm - Legacy fallback

        Returns:
            True if this node is the leader with a valid lease.
        """
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            return False

        node_id = getattr(self._p2p, "node_id", "")

        # Check forced leader override first - bypasses all consensus checks
        if getattr(self._p2p, "_forced_leader_override", False):
            if time.time() < getattr(self._p2p, "leader_lease_expires", 0):
                return True
            else:
                self._p2p._forced_leader_override = False
                self._log_info("[ForcedLeader] Forced leadership lease expired, clearing override")

        # First check consensus_mixin's Raft (supports deferred initialization)
        if getattr(self._p2p, "_raft_initialized", False):
            try:
                if hasattr(self._p2p, "is_raft_leader"):
                    is_raft_leader = self._p2p.is_raft_leader()
                    if is_raft_leader and getattr(self._p2p, "leader_id", None) != node_id:
                        self._log_info("[Raft] Syncing orchestrator state: consensus_mixin Raft says we're leader")
                        leader_state_lock = getattr(self._p2p, "leader_state_lock", None)
                        if leader_state_lock:
                            with leader_state_lock:
                                self._p2p.leader_id = node_id
                                self._p2p.role = NodeRole.LEADER
                    elif not is_raft_leader and getattr(self._p2p, "leader_id", None) == node_id:
                        self._log_debug("[Raft] consensus_mixin Raft says we're not leader")
                    return is_raft_leader
            except Exception as e:
                self._log_debug(f"[Raft] consensus_mixin.is_raft_leader() failed: {e}")

        # Route through HybridCoordinator if available
        hybrid_coordinator = getattr(self._p2p, "_hybrid_coordinator", None)
        if hybrid_coordinator is not None:
            try:
                is_raft_leader = hybrid_coordinator.is_leader()
                if is_raft_leader and getattr(self._p2p, "leader_id", None) != node_id:
                    self._log_info("[Raft] Syncing orchestrator state: HybridCoordinator says we're leader")
                    leader_state_lock = getattr(self._p2p, "leader_state_lock", None)
                    if leader_state_lock:
                        with leader_state_lock:
                            self._p2p.leader_id = node_id
                            self._p2p.role = NodeRole.LEADER
                elif not is_raft_leader and getattr(self._p2p, "leader_id", None) == node_id:
                    self._log_debug("[Raft] HybridCoordinator says we're not leader")
                return is_raft_leader
            except Exception as e:
                self._log_warning(f"[Raft] HybridCoordinator.is_leader() failed, falling back to Bully: {e}")

        # Bully algorithm fallback
        return self._check_is_leader_bully()

    def _check_is_leader_bully(self) -> bool:
        """Check leadership using Bully algorithm (legacy fallback).

        Jan 29, 2026: Extracted from _is_leader() for clarity.
        """
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            return False

        node_id = getattr(self._p2p, "node_id", "")
        leader_id = getattr(self._p2p, "leader_id", None)
        role = getattr(self._p2p, "role", None)

        # Consistency check: we should never claim role=leader while leader_id points elsewhere
        if leader_id != node_id:
            self._log_debug(
                f"[LeaderCheck] Not leader: leader_id={leader_id}, "
                f"self.node_id={node_id}, role={role.value if hasattr(role, 'value') else role}"
            )
            if role in (NodeRole.LEADER, NodeRole.PROVISIONAL_LEADER):
                self._log_info("Inconsistent leadership state (role=leader but leader_id!=self); stepping down")
                self._handle_leadership_inconsistency_step_down()
            return False

        # Consistency check: we should never claim leader_id=self while being a follower/candidate
        if role not in (NodeRole.LEADER, NodeRole.PROVISIONAL_LEADER):
            self._log_info("Inconsistent leadership state (leader_id=self but role!=leader/provisional); clearing leader_id")
            self._handle_leadership_inconsistency_clear()
            return False

        # Must have valid lease to act as leader
        leader_lease_expires = getattr(self._p2p, "leader_lease_expires", 0)
        if leader_lease_expires > 0 and time.time() >= leader_lease_expires:
            self._log_info("Leadership lease expired, stepping down via ULSM")
            self._handle_lease_expired()
            return False

        # Check voter quorum
        voter_node_ids = getattr(self._p2p, "voter_node_ids", [])
        if voter_node_ids and hasattr(self._p2p, "_has_voter_quorum") and not self._p2p._has_voter_quorum():
            if not self._handle_quorum_check_failure():
                return False

        return True

    def _handle_leadership_inconsistency_step_down(self) -> None:
        """Handle case where role=leader but leader_id!=self.

        Jan 29, 2026: Extracted from _is_leader() for clarity.
        """
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            return

        leader_state_lock = getattr(self._p2p, "leader_state_lock", None)
        if leader_state_lock:
            with leader_state_lock:
                self._p2p.role = NodeRole.FOLLOWER
                self._p2p.last_lease_renewal = 0.0
                if not getattr(self._p2p, "leader_id", None):
                    self._p2p.leader_lease_id = ""
                    self._p2p.leader_lease_expires = 0.0

        if hasattr(self._p2p, "_release_voter_grant_if_self"):
            self._p2p._release_voter_grant_if_self()
        if hasattr(self._p2p, "_save_state"):
            self._p2p._save_state()

        # Start election if no known leader and we have quorum
        if not getattr(self._p2p, "leader_id", None):
            voter_node_ids = getattr(self._p2p, "voter_node_ids", [])
            has_quorum = not voter_node_ids or (hasattr(self._p2p, "_has_voter_quorum") and self._p2p._has_voter_quorum())
            if has_quorum:
                self._schedule_election()
            else:
                self._log_warning("Skipping election: no voter quorum available")

    def _handle_leadership_inconsistency_clear(self) -> None:
        """Handle case where leader_id=self but role!=leader.

        Jan 29, 2026: Extracted from _is_leader() for clarity.
        """
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            return

        leader_state_lock = getattr(self._p2p, "leader_state_lock", None)
        if leader_state_lock:
            with leader_state_lock:
                self._p2p.role = NodeRole.FOLLOWER
                self._p2p.leader_id = None
                self._p2p.leader_lease_id = ""
                self._p2p.leader_lease_expires = 0.0
                self._p2p.last_lease_renewal = 0.0

        if hasattr(self._p2p, "_release_voter_grant_if_self"):
            self._p2p._release_voter_grant_if_self()
        if hasattr(self._p2p, "_save_state"):
            self._p2p._save_state()

        # Start election if we have quorum
        voter_node_ids = getattr(self._p2p, "voter_node_ids", [])
        has_quorum = not voter_node_ids or (hasattr(self._p2p, "_has_voter_quorum") and self._p2p._has_voter_quorum())
        if has_quorum:
            self._schedule_election()
        else:
            self._log_warning("Skipping election after inconsistent state: no voter quorum available")

    def _handle_lease_expired(self) -> None:
        """Handle leadership lease expiration.

        Jan 29, 2026: Extracted from _is_leader() for clarity.
        """
        if hasattr(self._p2p, "_schedule_step_down_sync"):
            try:
                from scripts.p2p.leadership_state_machine import TransitionReason
                self._p2p._schedule_step_down_sync(TransitionReason.LEASE_EXPIRED)
            except ImportError:
                pass

    def _handle_quorum_check_failure(self) -> bool:
        """Handle voter quorum check failure.

        Jan 29, 2026: Extracted from _is_leader() for clarity.

        Returns:
            True if we should continue as leader, False if we should step down.
        """
        voters_alive = 0
        if hasattr(self._p2p, "_count_alive_voters"):
            voters_alive = self._p2p._count_alive_voters()
        quorum_size = getattr(self._p2p, "voter_quorum_size", 0)

        # Use ULSM QuorumHealth for unified tracking
        leadership_sm = getattr(self._p2p, "_leadership_sm", None)
        if leadership_sm is None or not hasattr(leadership_sm, "quorum_health"):
            # No ULSM, just log and continue
            self._log_debug(f"[LeaderCheck] Quorum check failed: voters_alive={voters_alive}, quorum_size={quorum_size}")
            return True

        threshold_exceeded = leadership_sm.quorum_health.record_failure(voters_alive)
        fail_count = leadership_sm.quorum_health.consecutive_failures
        threshold = leadership_sm.quorum_health.failure_threshold
        self._log_debug(
            f"[LeaderCheck] Quorum check failed ({fail_count}/{threshold}): "
            f"voters_alive={voters_alive}, quorum_size={quorum_size}"
        )

        if threshold_exceeded:
            self._log_info(f"Leadership without voter quorum ({threshold} consecutive failures), stepping down via ULSM")
            if hasattr(self._p2p, "_schedule_step_down_sync"):
                try:
                    from scripts.p2p.leadership_state_machine import TransitionReason
                    self._p2p._schedule_step_down_sync(TransitionReason.QUORUM_LOST)
                except ImportError:
                    pass
            self._log_warning("Skipping election after quorum loss: no voter quorum available")

            # Trigger aggressive peer discovery during quorum crisis
            quorum_crisis_loop = getattr(self._p2p, "_quorum_crisis_loop", None)
            if quorum_crisis_loop:
                quorum_crisis_loop.enter_crisis_mode(reason="quorum_lost")
            return False

        return True

    def _record_quorum_success(self) -> None:
        """Record successful quorum check.

        Jan 29, 2026: Extracted from _is_leader() for clarity.
        """
        leadership_sm = getattr(self._p2p, "_leadership_sm", None)
        if leadership_sm is not None and hasattr(leadership_sm, "quorum_health"):
            voters_alive = 0
            if hasattr(self._p2p, "_count_alive_voters"):
                voters_alive = self._p2p._count_alive_voters()
            leadership_sm.quorum_health.record_success(voters_alive)

            # Exit crisis mode when quorum is restored
            quorum_crisis_loop = getattr(self._p2p, "_quorum_crisis_loop", None)
            if quorum_crisis_loop and quorum_crisis_loop.in_crisis_mode:
                quorum_crisis_loop.exit_crisis_mode(reason="quorum_restored")

    def _schedule_election(self) -> None:
        """Schedule an election via the event loop.

        Jan 29, 2026: Helper method for consistency.
        """
        try:
            loop = asyncio.get_running_loop()
            if hasattr(self._p2p, "_start_election"):
                with contextlib.suppress(RuntimeError):
                    loop.create_task(self._p2p._start_election())
        except RuntimeError:
            pass  # No running event loop

    def is_leader(self) -> bool:
        """Check if this node is the current cluster leader.

        Returns:
            True if this node is the leader with a valid lease.
        """
        return self.check_is_leader()

    def is_leader_lease_valid(self) -> bool:
        """Check if the current leader's lease is still valid.

        Jan 29, 2026: Implementation moved from P2POrchestrator._is_leader_lease_valid().

        Returns:
            True if the leader lease is valid.
        """
        leader_id = getattr(self._p2p, "leader_id", None)
        if not leader_id:
            return False

        # Reject proxy_only leaders - they should never have been elected
        # This forces a re-election if a proxy_only node somehow became leader
        if hasattr(self._p2p, "_is_node_proxy_only"):
            if self._p2p._is_node_proxy_only(leader_id):
                self._log_warning(
                    f"[LeaderValidation] Current leader {leader_id} is proxy_only - invalidating lease"
                )
                return False

        # Use symmetric grace period for both leader and followers
        # CRITICAL: Same grace for everyone to prevent split-brain.
        lease_expires = getattr(self._p2p, "leader_lease_expires", 0)
        grace = 30  # Same grace for leader and followers
        return time.time() < lease_expires + grace

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
            from scripts.p2p.types import NodeRole
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
            from scripts.p2p.types import NodeRole
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
            from scripts.p2p.types import NodeRole
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
            from scripts.p2p.types import NodeRole
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
        """Broadcast a leadership claim to all peers.

        Jan 28, 2026: Implementation moved from P2POrchestrator.
        Schedules async broadcast via event loop.
        """
        leader_id = getattr(self._p2p, "leader_id", None)
        if not leader_id:
            return

        # Schedule async broadcast using the event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.async_broadcast_leader_claim())
                self._leader_claim_broadcast_count += 1
        except RuntimeError:
            # No event loop available
            pass

    async def async_broadcast_leader_claim(self) -> None:
        """Asynchronously broadcast a leadership claim to all peers.

        Jan 28, 2026: Implementation moved from P2POrchestrator.
        """
        try:
            # Access peer snapshot via P2P
            peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
            if peer_snapshot is None:
                self._log_debug("No peer snapshot available for broadcast")
                return

            peers_snapshot = peer_snapshot.get_snapshot()
            tasks = []

            node_id = getattr(self._p2p, "node_id", "")
            leader_id = getattr(self._p2p, "leader_id", None)
            role = getattr(self._p2p, "role", None)

            for peer_id, peer_info in peers_snapshot.items():
                if not peer_info.is_alive():
                    continue

                url = f"http://{peer_info.host}:{peer_info.port}/heartbeat"
                payload = {
                    "node_id": node_id,
                    "leader_id": leader_id,
                    "role": role.value if hasattr(role, "value") else str(role),
                    "timestamp": time.time(),
                    "is_leadership_claim": True,
                }
                tasks.append(self._broadcast_claim_to_peer(url, payload, peer_id))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success = sum(1 for r in results if r is True)
                self._log_debug(
                    f"[LeaderReconciliation] Broadcast leadership claim to {success}/{len(tasks)} peers"
                )
        except Exception as e:
            self._log_debug(f"[LeaderReconciliation] Failed to broadcast leadership claim: {e}")

    async def _broadcast_claim_to_peer(self, url: str, payload: dict, peer_id: str) -> bool:
        """Broadcast leadership claim to a single peer via HTTP POST.

        Jan 28, 2026: Implementation moved from P2POrchestrator.
        """
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    async def broadcast_leader_state_change(
        self,
        new_state: str,
        epoch: int,
        reason: Any,
    ) -> None:
        """Broadcast leadership state change to all peers.

        Jan 28, 2026: Implementation moved from P2POrchestrator.
        Called when stepping down. Ensures peers learn about step-down
        BEFORE local state mutation.

        Args:
            new_state: New leadership state (e.g., "stepping_down")
            epoch: Current leadership epoch
            reason: Reason for the transition (TransitionReason)
        """
        message = {
            "node_id": getattr(self._p2p, "node_id", ""),
            "new_state": new_state,
            "epoch": epoch,
            "reason": reason.value if hasattr(reason, "value") else str(reason),
            "timestamp": time.time(),
        }

        # Access peer snapshot via P2P
        peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
        if peer_snapshot is None:
            return

        peers_snapshot = peer_snapshot.get_snapshot()
        tasks = []

        for peer_id, peer_info in peers_snapshot.items():
            if not peer_info.is_alive():
                continue

            url = f"http://{peer_info.host}:{peer_info.port}/leader-state-change"
            tasks.append(self._broadcast_state_to_peer(url, message, peer_id))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success = sum(1 for r in results if r is True)
            self._log_info(
                f"Broadcast step-down to {success}/{len(tasks)} peers "
                f"(epoch={epoch}, reason={reason.value if hasattr(reason, 'value') else reason})"
            )

    async def _broadcast_state_to_peer(
        self,
        url: str,
        message: dict[str, Any],
        peer_id: str,
    ) -> bool:
        """Send state change message to a single peer with timeout.

        Jan 28, 2026: Implementation moved from P2POrchestrator.
        """
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                auth_headers = {}
                if hasattr(self._p2p, "_auth_headers"):
                    auth_headers = self._p2p._auth_headers()

                async with session.post(
                    url,
                    json=message,
                    headers=auth_headers,
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    if resp.status == 200:
                        return True
                    self._log_debug(f"Broadcast to {peer_id} returned {resp.status}")
                    return False
        except Exception as e:
            self._log_debug(f"Broadcast to {peer_id} failed: {e}")
            return False

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

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_leader_hint(self) -> dict[str, Any]:
        """Get this node's leader hint for gossip propagation.

        Jan 29, 2026: Implementation moved from P2POrchestrator._get_leader_hint().

        LEADER HINTS: Share information about preferred leader candidates to
        enable faster convergence during elections. Hints include:
        - Current known leader and lease expiry
        - Preferred successor (highest-priority eligible node)
        - This node's priority rank

        Returns:
            Dictionary with leader hint information.
        """
        leader_id = getattr(self._p2p, "leader_id", None)
        node_id = getattr(self._p2p, "node_id", "")

        hint = {
            "current_leader": leader_id,
            "lease_expires": getattr(self._p2p, "leader_lease_expires", 0),
            "preferred_successor": None,
            "my_priority": 0,
        }

        # Calculate this node's priority (lower is better for Bully algorithm)
        # But we want to express it as a score (higher is better)
        peers_lock = getattr(self._p2p, "peers_lock", None)
        peers = getattr(self._p2p, "peers", {})

        if peers_lock is not None:
            with peers_lock:
                all_nodes = [node_id] + [p.node_id for p in peers.values() if p.is_alive()]
        else:
            all_nodes = [node_id]

        all_nodes_sorted = sorted(all_nodes, reverse=True)  # Bully: higher ID wins
        if node_id in all_nodes_sorted:
            hint["my_priority"] = len(all_nodes_sorted) - all_nodes_sorted.index(node_id)

        # Find preferred successor (highest priority eligible node that's not current leader)
        voter_ids = list(getattr(self._p2p, "voter_node_ids", []) or [])
        for nid in all_nodes_sorted:
            if nid == leader_id:
                continue
            if voter_ids and nid not in voter_ids:
                continue
            hint["preferred_successor"] = nid
            break

        return hint

    def get_cluster_leader_consensus(self) -> dict[str, Any]:
        """Get cluster consensus on leader from gossip hints.

        Jan 29, 2026: Implementation moved from P2POrchestrator._get_cluster_leader_consensus().

        LEADER CONSENSUS: Aggregate leader hints from all nodes to determine
        if there's agreement on who the leader is/should be.

        Returns:
            Dictionary with consensus status and voting details.
        """
        import asyncio

        leader_votes: dict[str, int] = {}
        successor_votes: dict[str, int] = {}
        gossip_states = getattr(self._p2p, "_gossip_peer_states", {})
        now = time.time()

        # Count our vote
        our_hint = self.get_leader_hint()
        if our_hint.get("current_leader"):
            leader_votes[our_hint["current_leader"]] = leader_votes.get(our_hint["current_leader"], 0) + 1
        if our_hint.get("preferred_successor"):
            successor_votes[our_hint["preferred_successor"]] = successor_votes.get(our_hint["preferred_successor"], 0) + 1

        # Count votes from gossip
        for _node_id, state in gossip_states.items():
            if state.get("timestamp", 0) < now - 120:  # Skip stale states
                continue

            hint = state.get("leader_hint", {})
            leader = hint.get("current_leader")
            successor = hint.get("preferred_successor")

            if leader:
                leader_votes[leader] = leader_votes.get(leader, 0) + 1
            if successor:
                successor_votes[successor] = successor_votes.get(successor, 0) + 1

        # Find consensus leader and successor
        consensus_leader = max(leader_votes.items(), key=lambda x: x[1])[0] if leader_votes else None
        consensus_successor = max(successor_votes.items(), key=lambda x: x[1])[0] if successor_votes else None

        result = {
            "consensus_leader": consensus_leader,
            "leader_agreement": leader_votes.get(consensus_leader, 0) if consensus_leader else 0,
            "consensus_successor": consensus_successor,
            "successor_agreement": successor_votes.get(consensus_successor, 0) if consensus_successor else 0,
            "total_voters": len(gossip_states) + 1,
        }

        # ALERTING: Check for low leader consensus (only leader alerts, rate limited)
        try:
            from scripts.p2p.types import NodeRole
            role = getattr(self._p2p, "role", None)
            if role == NodeRole.LEADER and result["total_voters"] >= 3:
                agreement_ratio = result["leader_agreement"] / result["total_voters"]
                last_low_consensus_alert = getattr(self, "_last_low_consensus_alert", 0)
                if agreement_ratio < 0.5 and now - last_low_consensus_alert > 3600:  # Alert once per hour max
                    self._last_low_consensus_alert = now
                    notifier = getattr(self._p2p, "notifier", None)
                    if notifier is not None:
                        node_id = getattr(self._p2p, "node_id", "")
                        asyncio.create_task(notifier.send(
                            title="Low Leader Consensus",
                            message=f"Only {result['leader_agreement']}/{result['total_voters']} nodes agree on leader",
                            level="warning",
                            fields={
                                "Agreement": f"{agreement_ratio*100:.0f}%",
                                "Consensus Leader": str(consensus_leader),
                                "Total Voters": str(result["total_voters"]),
                                "Action": "Check for network partitions or stale nodes",
                            },
                            node_id=node_id,
                        ))
        except ImportError:
            pass

        return result

    def count_peers_reporting_leader(self, leader_id: str, peers_snapshot: list | None = None) -> int:
        """Count how many peers report the same leader_id.

        Jan 29, 2026: Implementation moved from P2POrchestrator._count_peers_reporting_leader().

        Args:
            leader_id: The leader ID to check for consensus
            peers_snapshot: List of peer NodeInfo objects (optional, will get from P2P if not provided)

        Returns:
            Number of peers reporting this leader_id
        """
        if peers_snapshot is None:
            peer_snapshot = getattr(self._p2p, "_peer_snapshot", None)
            if peer_snapshot is None:
                return 0
            peers_snapshot = list(peer_snapshot.get_snapshot().values())

        count = 0
        for peer in peers_snapshot:
            if not peer.is_alive():
                continue
            # Check if peer reports this leader
            peer_leader = getattr(peer, "leader_id", None)
            if peer_leader == leader_id:
                count += 1
        return count

    # =========================================================================
    # Partition Local Election
    # =========================================================================

    def enable_partition_local_election(self) -> bool:
        """Enable local leader election for partitioned nodes.

        Jan 29, 2026: Implementation moved from P2POrchestrator._enable_partition_local_election().

        When a partition is detected and no voters are reachable, this method
        temporarily adds reachable nodes to the voter set so they can elect a
        local leader and continue operating autonomously.

        This is a self-healing mechanism for network splits. When connectivity
        is restored, the partition will merge back with the main cluster.

        Returns:
            True if local election was enabled
        """
        import os

        # Import constants with fallback
        try:
            from scripts.p2p.constants import VOTER_MIN_QUORUM
        except ImportError:
            VOTER_MIN_QUORUM = 3

        p2p = self._p2p

        # Don't override env-configured voters
        if (os.environ.get("RINGRIFT_P2P_VOTERS") or "").strip():
            return False

        # Check if we have any voters configured
        voters = list(getattr(p2p, "voter_node_ids", []) or [])

        # Count how many voters are reachable
        # Jan 2026: Use lock-free PeerSnapshot for read-only access
        peer_snapshot = getattr(p2p, "_peer_snapshot", None)
        if peer_snapshot is None:
            return False

        peers_by_id = peer_snapshot.get_snapshot()
        reachable_voters = 0
        for voter_id in voters:
            if voter_id == p2p.node_id:
                reachable_voters += 1
                continue
            peer = peers_by_id.get(voter_id)
            if peer and peer.is_alive():
                reachable_voters += 1

        # If we have quorum (simplified: 3 voters), no need for partition election
        quorum = min(VOTER_MIN_QUORUM, len(voters)) if voters else 1
        if reachable_voters >= quorum:
            return False

        # Build local partition voter set from reachable nodes
        local_voters = [p2p.node_id]  # Always include self
        for node_id, peer in peers_by_id.items():
            if peer.is_alive() and node_id not in local_voters:
                local_voters.append(node_id)

        if len(local_voters) < 2:
            # Need at least 2 nodes for meaningful election
            return False

        # Store original voters for restoration
        if not hasattr(p2p, "_original_voters"):
            p2p._original_voters = voters.copy()
            p2p._partition_election_started = time.time()

        # Enable partition-local election
        p2p.voter_node_ids = sorted(local_voters)
        p2p.voter_quorum_size = min(VOTER_MIN_QUORUM, len(local_voters))
        p2p.voter_config_source = "partition-local"
        self._log_info(
            f"PARTITION: Enabling local election with {len(local_voters)} nodes: "
            f"{', '.join(local_voters)} (quorum={p2p.voter_quorum_size})"
        )
        return True

    def restore_original_voters(self) -> bool:
        """Restore original voter configuration after partition heals.

        Jan 29, 2026: Implementation moved from P2POrchestrator._restore_original_voters().

        Called when connectivity to the main cluster is restored.

        Returns:
            True if voters were restored
        """
        # Import constants with fallback
        try:
            from scripts.p2p.constants import VOTER_MIN_QUORUM
        except ImportError:
            VOTER_MIN_QUORUM = 3

        p2p = self._p2p

        if not hasattr(p2p, "_original_voters"):
            return False

        original = getattr(p2p, "_original_voters", [])
        if not original:
            return False

        # Check if we can reach any original voters
        # Jan 2026: Use lock-free PeerSnapshot for read-only access
        peer_snapshot = getattr(p2p, "_peer_snapshot", None)
        if peer_snapshot is None:
            return False

        peers_by_id = peer_snapshot.get_snapshot()
        for voter_id in original:
            if voter_id == p2p.node_id:
                continue
            peer = peers_by_id.get(voter_id)
            if peer and peer.is_alive():
                # We can reach at least one original voter, restore config
                p2p.voter_node_ids = original.copy()
                # SIMPLIFIED QUORUM: Fixed at 3 voters (or less if fewer voters exist)
                p2p.voter_quorum_size = min(VOTER_MIN_QUORUM, len(original))
                p2p.voter_config_source = "restored"
                delattr(p2p, "_original_voters")
                if hasattr(p2p, "_partition_election_started"):
                    delattr(p2p, "_partition_election_started")
                self._log_info(f"Partition healed: restored original voters {', '.join(original)}")
                return True
        return False

    def step_down_from_provisional(self) -> None:
        """Step down from provisional leadership (lost to challenger).

        Jan 29, 2026: Implementation moved from P2POrchestrator._step_down_from_provisional().

        Uses ULSM for broadcast-before-mutation pattern. Clears provisional-specific
        state, schedules step-down via ULSM, and notifies voters of lease revocation.
        """
        p2p = self._p2p

        self._log_info("Stepping down from provisional leadership via ULSM")

        # Clear provisional-specific state first (ULSM doesn't know about these)
        p2p._provisional_leader_claimed_at = 0.0
        p2p._provisional_leader_acks.clear()
        p2p._provisional_leader_challengers.clear()

        # Use ULSM step-down (broadcasts to peers, then clears leader state)
        try:
            from scripts.p2p.constants import TransitionReason
            p2p._schedule_step_down_sync(TransitionReason.ARBITER_OVERRIDE)
        except ImportError:
            # Fallback if constants not available
            p2p._schedule_step_down_sync("arbiter_override")

        # Notify voters of lease revocation
        try:
            asyncio.create_task(p2p._notify_voters_lease_revoked())
        except RuntimeError:
            # Not in async context, schedule on event loop if available
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(p2p._notify_voters_lease_revoked(), loop)

    async def request_election_from_voters(self, reason: str = "non_voter_request") -> bool:
        """Non-voters can request that voters start an election.

        Jan 29, 2026: Implementation moved from P2POrchestrator._request_election_from_voters().

        Instead of silently returning when a non-voter tries to start an election,
        this method sends requests to known voters to have them start one.

        Args:
            reason: Why the election is being requested

        Returns:
            True if at least one voter accepted the request
        """
        import aiohttp

        p2p = self._p2p
        voter_node_ids = list(getattr(p2p, "voter_node_ids", []) or [])
        if not voter_node_ids:
            return False

        self._log_info(f"Non-voter {self.node_id} requesting election from voters: {reason}")

        # Rate limit election requests to avoid spamming
        now = time.time()
        last_request = getattr(p2p, "_last_election_request", 0.0)
        if now - last_request < 30:  # At most once per 30 seconds
            logger.debug("Skipping election request: rate limited")
            return False
        p2p._last_election_request = now

        accepted = False
        # Jan 12, 2026: Copy-on-write - single snapshot instead of lock-per-voter
        with p2p.peers_lock:
            peers_snapshot = dict(p2p.peers)

        async with aiohttp.ClientSession() as session:
            for voter_id in voter_node_ids[:3]:  # Limit to 3 voters
                voter = peers_snapshot.get(voter_id)
                if not voter or not voter.is_alive():
                    continue

                try:
                    url = p2p._url_for_peer(voter, "/election/request")
                    async with session.post(
                        url,
                        json={"requester_id": self.node_id, "reason": reason},
                        headers=p2p._auth_headers(),
                        timeout=aiohttp.ClientTimeout(total=5.0),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("accepted"):
                                self._log_info(
                                    f"Voter {voter_id} accepted election request: {data.get('action')}"
                                )
                                accepted = True
                                break  # One voter accepting is enough
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.debug(f"Failed to request election from {voter_id}: {e}")
                    continue

        if not accepted:
            logger.warning(f"No voters accepted election request from {self.node_id}")
        return accepted

    def maybe_adopt_leader_from_peers(self) -> bool:
        """If we can already see a healthy leader, adopt it and avoid elections.

        Jan 29, 2026: Implementation moved from P2POrchestrator._maybe_adopt_leader_from_peers().

        Returns:
            True if a leader was adopted from peers, False otherwise.
        """
        import time
        from scripts.p2p.protocols import NodeRole

        p2p = self._p2p
        if p2p.role == NodeRole.LEADER:
            return False

        # Use lock-free PeerSnapshot for read-only access
        peers = [
            p for p in p2p._peer_snapshot.get_snapshot().values()
            if p.node_id != p2p.node_id
        ]

        conflict_keys = p2p._endpoint_conflict_keys([p2p.self_info, *peers])
        leaders = [
            p for p in peers
            if p.role == NodeRole.LEADER and p2p._is_leader_eligible(p, conflict_keys)
        ]

        voter_ids = list(getattr(p2p, "voter_node_ids", []) or [])
        if voter_ids:
            leaders = [p for p in leaders if p.node_id in voter_ids]

        if not leaders:
            return False

        # If multiple leaders exist (split brain), pick the lexicographically highest
        # ID (matches bully ordering) to converge.
        leader = sorted(leaders, key=lambda p: p.node_id)[-1]

        if p2p.leader_id != leader.node_id:
            self._log_info(f"Adopted existing leader from peers: {leader.node_id}")
            # Record election latency for "adopted" outcome if we were in an election
            if getattr(p2p, "_election_started_at", 0) > 0:
                p2p._record_election_latency("adopted")

        # Use _set_leader() for atomic leadership assignment
        p2p._set_leader(leader.node_id, reason="join_existing_leader", save_state=True)
        p2p.last_leader_seen = time.time()
        return True
