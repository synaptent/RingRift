"""Unified Leadership State Machine for P2P Cluster.

Created as part of the ULSM initiative (Jan 2026) to address leader stability issues.

This module provides a formal state machine for leadership transitions with:
- Explicit state transitions via transition_to() (no direct field mutation)
- Mandatory broadcast BEFORE local state mutation (via STEPPING_DOWN state)
- Epoch-based leader validation for gossip protocol
- Unified quorum health tracking (replaces dual counters)
- Restart healing for inconsistent persisted state

Usage:
    from scripts.p2p.leadership_state_machine import (
        LeadershipStateMachine,
        LeaderState,
        TransitionReason,
    )

    # Initialize
    sm = LeadershipStateMachine(node_id="my-node")
    sm._broadcast_callback = my_broadcast_function

    # Transition (the ONLY way to change state)
    await sm.transition_to(LeaderState.STEPPING_DOWN, TransitionReason.QUORUM_LOST)
    await sm.transition_to(LeaderState.FOLLOWER, TransitionReason.STEP_DOWN_COMPLETE)

    # Validate gossip leader claims
    if sm.validate_leader_claim(claimed_leader, epoch, lease_expires):
        accept_leader(claimed_leader)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from typing import Awaitable

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class LeaderState(Enum):
    """Leadership states in the P2P cluster.

    State transitions are strictly controlled via VALID_TRANSITIONS matrix.
    The STEPPING_DOWN state ensures broadcast happens before local mutation.
    """

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    PROVISIONAL_LEADER = "provisional_leader"
    LEADER = "leader"
    STEPPING_DOWN = "stepping_down"  # Intermediate state for safe step-down


class TransitionReason(Enum):
    """Reasons for leadership state transitions.

    Used for logging, debugging, and event emission.
    """

    # Becoming leader
    ELECTION_WON = "election_won"
    PROVISIONAL_CLAIM = "provisional_claim"
    PROVISIONAL_PROMOTED = "provisional_promoted"

    # Losing leadership
    LEASE_EXPIRED = "lease_expired"
    QUORUM_LOST = "quorum_lost"
    ARBITER_OVERRIDE = "arbiter_override"
    HIGHER_EPOCH_SEEN = "higher_epoch_seen"
    STEP_DOWN_COMPLETE = "step_down_complete"
    NETWORK_PARTITION = "network_partition"

    # Recovery
    RESTART_HEALING = "restart_healing"
    ELECTION_LOST = "election_lost"
    MANUAL_OVERRIDE = "manual_override"


# Valid state transitions matrix
# Key: current state, Value: set of allowed next states
VALID_TRANSITIONS: dict[LeaderState, set[LeaderState]] = {
    LeaderState.FOLLOWER: {
        LeaderState.CANDIDATE,
        LeaderState.PROVISIONAL_LEADER,
    },
    LeaderState.CANDIDATE: {
        LeaderState.FOLLOWER,
        LeaderState.LEADER,
        LeaderState.PROVISIONAL_LEADER,
    },
    LeaderState.LEADER: {
        LeaderState.STEPPING_DOWN,  # Must go through STEPPING_DOWN for broadcast
    },
    LeaderState.PROVISIONAL_LEADER: {
        LeaderState.STEPPING_DOWN,
        LeaderState.LEADER,  # Can be promoted to full leader
    },
    LeaderState.STEPPING_DOWN: {
        LeaderState.FOLLOWER,  # Terminal transition only
    },
}


# =============================================================================
# QuorumHealth - Unified quorum failure tracking
# =============================================================================


@dataclass
class QuorumHealth:
    """Unified quorum failure tracking.

    Replaces the dual counters (_is_leader_quorum_fail_count and _quorum_fail_count)
    that were previously tracked independently in _is_leader() and _renew_leader_lease().

    Attributes:
        consecutive_failures: Number of consecutive quorum check failures
        failure_threshold: Number of failures before triggering step-down (default: 5)
        last_success_time: Timestamp of last successful quorum check
        voters_seen_last_check: Number of voters seen in last check
    """

    consecutive_failures: int = 0
    failure_threshold: int = 5  # Increased from 3 for stability
    last_success_time: float = field(default_factory=time.time)
    voters_seen_last_check: int = 0

    def record_failure(self, voters_alive: int) -> bool:
        """Record a quorum check failure.

        Args:
            voters_alive: Number of voters currently alive

        Returns:
            True if failure threshold exceeded (should trigger step-down)
        """
        self.consecutive_failures += 1
        self.voters_seen_last_check = voters_alive
        threshold_exceeded = self.consecutive_failures >= self.failure_threshold

        if threshold_exceeded:
            logger.warning(
                f"Quorum health: threshold exceeded "
                f"({self.consecutive_failures}/{self.failure_threshold} failures, "
                f"voters_alive={voters_alive})"
            )

        return threshold_exceeded

    def record_success(self, voters_alive: int) -> None:
        """Record a successful quorum check."""
        if self.consecutive_failures > 0:
            logger.info(
                f"Quorum health: recovered after {self.consecutive_failures} failures"
            )
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.voters_seen_last_check = voters_alive

    def reset(self) -> None:
        """Reset failure counter (called on state transitions)."""
        self.consecutive_failures = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for /status endpoint."""
        return {
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "last_success_time": self.last_success_time,
            "voters_seen_last_check": self.voters_seen_last_check,
            "seconds_since_success": time.time() - self.last_success_time,
        }


# =============================================================================
# LeadershipStateMachine
# =============================================================================


class LeadershipStateMachine:
    """Unified leadership state management with explicit transitions.

    This is the single source of truth for leadership state. All state changes
    must go through transition_to() - no direct field mutations allowed.

    Key guarantees:
    1. All transitions are validated against VALID_TRANSITIONS matrix
    2. STEPPING_DOWN state triggers broadcast BEFORE local state mutation
    3. Epoch increments on any leadership loss (invalidates stale claims)
    4. Restart healing demotes any leadership claims to FOLLOWER

    Attributes:
        node_id: This node's identifier
        quorum_health: Unified quorum failure tracking
    """

    def __init__(
        self,
        node_id: str,
        initial_state: LeaderState = LeaderState.FOLLOWER,
        initial_epoch: int = 0,
    ):
        """Initialize the state machine.

        Args:
            node_id: This node's identifier
            initial_state: Starting state (default: FOLLOWER)
            initial_epoch: Starting epoch (default: 0)
        """
        self.node_id = node_id
        self._state = initial_state
        self._epoch = initial_epoch
        self._leader_id: Optional[str] = None
        self._invalidation_until = 0.0
        self._last_transition_time = time.time()
        self._transition_count = 0

        # Unified quorum tracking
        self.quorum_health = QuorumHealth()

        # Broadcast callback - set by orchestrator
        # Signature: async def callback(new_state: str, epoch: int, reason: TransitionReason) -> None
        self._broadcast_callback: Optional[
            Callable[[str, int, TransitionReason], Awaitable[None]]
        ] = None

        # Set initial leader_id based on state
        if initial_state in (LeaderState.LEADER, LeaderState.PROVISIONAL_LEADER):
            self._leader_id = node_id

        logger.info(
            f"LeadershipStateMachine initialized: node_id={node_id}, "
            f"state={initial_state.value}, epoch={initial_epoch}"
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> LeaderState:
        """Current leadership state."""
        return self._state

    @property
    def epoch(self) -> int:
        """Current leadership epoch (monotonically increasing)."""
        return self._epoch

    @property
    def leader_id(self) -> Optional[str]:
        """Current leader ID (None if no leader known)."""
        return self._leader_id

    @property
    def is_leader(self) -> bool:
        """True if this node is the current leader."""
        return self._state == LeaderState.LEADER and self._leader_id == self.node_id

    @property
    def is_provisional_leader(self) -> bool:
        """True if this node is a provisional leader."""
        return (
            self._state == LeaderState.PROVISIONAL_LEADER
            and self._leader_id == self.node_id
        )

    @property
    def is_stepping_down(self) -> bool:
        """True if this node is in the stepping down process."""
        return self._state == LeaderState.STEPPING_DOWN

    @property
    def in_invalidation_window(self) -> bool:
        """True if we're in the post-step-down invalidation window."""
        return time.time() < self._invalidation_until

    @property
    def invalidation_remaining_seconds(self) -> float:
        """Seconds remaining in invalidation window (0 if not active)."""
        remaining = self._invalidation_until - time.time()
        return max(0.0, remaining)

    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------

    async def transition_to(
        self,
        new_state: LeaderState,
        reason: TransitionReason,
    ) -> bool:
        """Transition to a new leadership state.

        This is the ONLY way to change leadership state. All transitions are
        logged and validated against the VALID_TRANSITIONS matrix.

        For STEPPING_DOWN transitions, the broadcast callback is invoked BEFORE
        local state mutation to ensure peers learn about step-down even if we crash.

        Args:
            new_state: Target state
            reason: Reason for transition

        Returns:
            True if transition succeeded, False if invalid
        """
        # Validate transition
        valid_targets = VALID_TRANSITIONS.get(self._state, set())
        if new_state not in valid_targets:
            logger.warning(
                f"Invalid transition: {self._state.value} -> {new_state.value} "
                f"(valid targets: {[s.value for s in valid_targets]})"
            )
            return False

        old_state = self._state
        old_epoch = self._epoch

        # STEPPING_DOWN requires special handling: broadcast BEFORE mutation
        if new_state == LeaderState.STEPPING_DOWN:
            # Increment epoch to invalidate any stale leader claims in gossip
            self._epoch += 1

            # Set invalidation window to reject stale gossip
            self._invalidation_until = time.time() + 60.0

            # Broadcast to peers BEFORE local state change
            if self._broadcast_callback is not None:
                try:
                    await self._broadcast_callback(
                        "stepping_down",
                        self._epoch,
                        reason,
                    )
                except Exception as e:
                    logger.error(f"Broadcast callback failed: {e}")
                    # Continue with transition even if broadcast fails

        # Perform state transition
        self._state = new_state
        self._last_transition_time = time.time()
        self._transition_count += 1

        # Update leader_id based on new state
        if new_state in (LeaderState.LEADER, LeaderState.PROVISIONAL_LEADER):
            self._leader_id = self.node_id
        elif new_state == LeaderState.FOLLOWER:
            self._leader_id = None
        # STEPPING_DOWN and CANDIDATE keep current leader_id

        # Reset quorum health on state change
        self.quorum_health.reset()

        logger.info(
            f"Leadership transition: {old_state.value} -> {new_state.value} "
            f"(reason={reason.value}, epoch={old_epoch}->{self._epoch}, node={self.node_id})"
        )

        return True

    def validate_leader_claim(
        self,
        claimed_leader: str,
        claimed_epoch: int,
        lease_expires: float,
    ) -> bool:
        """Validate if a leader claim from gossip should be accepted.

        Used by the gossip protocol to filter out stale leader claims.

        Args:
            claimed_leader: Node ID claiming to be leader
            claimed_epoch: Epoch of the claim
            lease_expires: Timestamp when the leader's lease expires

        Returns:
            True if the claim should be accepted
        """
        now = time.time()

        # Reject if we're in invalidation window (just stepped down)
        if now < self._invalidation_until:
            logger.debug(
                f"Rejecting leader claim from {claimed_leader}: in invalidation window "
                f"({self.invalidation_remaining_seconds:.1f}s remaining)"
            )
            return False

        # Reject stale epochs
        if claimed_epoch < self._epoch:
            logger.debug(
                f"Rejecting leader claim from {claimed_leader}: stale epoch "
                f"({claimed_epoch} < {self._epoch})"
            )
            return False

        # Reject expired leases
        if lease_expires <= now:
            logger.debug(
                f"Rejecting leader claim from {claimed_leader}: lease expired "
                f"({lease_expires} <= {now})"
            )
            return False

        # Don't override if we already have a leader at same/higher epoch
        if self._leader_id and claimed_epoch <= self._epoch:
            logger.debug(
                f"Rejecting leader claim from {claimed_leader}: "
                f"already have leader {self._leader_id} at epoch {self._epoch}"
            )
            return False

        return True

    def set_leader(self, leader_id: str, epoch: int) -> None:
        """Set the current leader (called when accepting a leader claim from gossip).

        This does NOT change our state - we remain a follower. It just updates
        who we believe the leader is.

        Args:
            leader_id: Node ID of the leader
            epoch: Epoch of the leader
        """
        if epoch > self._epoch:
            self._epoch = epoch
        self._leader_id = leader_id
        logger.debug(f"Accepted leader {leader_id} at epoch {epoch}")

    def clear_leader(self) -> None:
        """Clear the current leader (when leader lease expires or leader is dead)."""
        if self._leader_id:
            logger.info(f"Clearing leader {self._leader_id}")
            self._leader_id = None

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_persisted_state(
        cls,
        node_id: str,
        state_dict: dict[str, Any],
    ) -> "LeadershipStateMachine":
        """Create a state machine from persisted state with healing logic.

        HEALING: If we were leader/provisional before restart, we demote to
        follower and increment epoch. This ensures we don't claim leadership
        without winning a new election.

        Args:
            node_id: This node's identifier
            state_dict: Persisted state dictionary

        Returns:
            New LeadershipStateMachine instance
        """
        initial_state = LeaderState(state_dict.get("state", "follower"))
        initial_epoch = state_dict.get("epoch", 0)

        # HEALING: Demote leadership claims on restart
        healed = False
        if initial_state in (
            LeaderState.LEADER,
            LeaderState.PROVISIONAL_LEADER,
            LeaderState.STEPPING_DOWN,
        ):
            logger.info(
                f"Restart healing: demoting from {initial_state.value} to follower, "
                f"incrementing epoch {initial_epoch} -> {initial_epoch + 1}"
            )
            initial_state = LeaderState.FOLLOWER
            initial_epoch += 1
            healed = True

        instance = cls(node_id, initial_state, initial_epoch)

        # Restore invalidation window if still valid
        invalidation_until = state_dict.get("invalidation_until", 0.0)
        if invalidation_until > time.time():
            instance._invalidation_until = invalidation_until

        # Restore quorum health if available
        if "quorum_health" in state_dict:
            qh = state_dict["quorum_health"]
            instance.quorum_health.failure_threshold = qh.get("failure_threshold", 5)
            # Don't restore consecutive_failures - start fresh

        if healed:
            logger.info(
                f"Restart healing complete: state={initial_state.value}, epoch={initial_epoch}"
            )

        return instance

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize state machine for /status endpoint and persistence."""
        return {
            "state": self._state.value,
            "epoch": self._epoch,
            "leader_id": self._leader_id,
            "node_id": self.node_id,
            "is_leader": self.is_leader,
            "is_provisional_leader": self.is_provisional_leader,
            "in_invalidation_window": self.in_invalidation_window,
            "invalidation_remaining_seconds": self.invalidation_remaining_seconds,
            "invalidation_until": self._invalidation_until,
            "quorum_health": self.quorum_health.to_dict(),
            "last_transition_time": self._last_transition_time,
            "transition_count": self._transition_count,
            "seconds_since_transition": time.time() - self._last_transition_time,
        }

    def to_persistence_dict(self) -> dict[str, Any]:
        """Serialize for state persistence (minimal footprint)."""
        return {
            "state": self._state.value,
            "epoch": self._epoch,
            "invalidation_until": self._invalidation_until,
            "quorum_health": {
                "failure_threshold": self.quorum_health.failure_threshold,
            },
        }

    def __repr__(self) -> str:
        return (
            f"LeadershipStateMachine(node_id={self.node_id!r}, "
            f"state={self._state.value}, epoch={self._epoch}, "
            f"leader_id={self._leader_id!r})"
        )
