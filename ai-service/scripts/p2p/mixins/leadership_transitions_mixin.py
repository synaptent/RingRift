"""Leadership Transitions Mixin - Step-down and state machine transitions.

January 2026: Extracted from p2p_orchestrator.py to reduce file size.

This mixin provides leadership state transition functionality:
- _schedule_step_down_sync(): Sync-to-async bridge for step-down
- _complete_step_down_async(): Full async step-down implementation

Usage:
    class P2POrchestrator(LeadershipTransitionsMixin, ...):
        pass

Dependencies on parent class attributes:
    - leader_id: str | None
    - role: NodeRole
    - leader_state_lock: threading.RLock
    - leader_lease_id: str
    - leader_lease_expires: float
    - last_lease_renewal: float
    - _leadership_sm: LeadershipStateMachine
    - _last_step_down_time: float

Dependencies on parent class methods:
    - _release_voter_grant_if_self()
    - _save_state()
    - _emit_leader_lost_sync()
    - _has_voter_quorum()
    - _start_election()
"""

from __future__ import annotations

import asyncio

from app.core.async_context import safe_create_task
import logging
import time
from typing import TYPE_CHECKING, Any

from scripts.p2p.p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    from scripts.p2p.ulsm import LeaderState, TransitionReason

logger = logging.getLogger(__name__)


class LeadershipTransitionsMixin(P2PMixinBase):
    """Mixin providing leadership state transition logic for P2P orchestrator.

    This mixin handles the step-down process when a leader needs to
    relinquish leadership due to:
    - Lease expiration
    - Quorum loss
    - Health degradation
    - Manual step-down request

    Inherits from P2PMixinBase for shared helper methods.
    """

    MIXIN_TYPE = "leadership_transitions"

    # Type hints for parent class attributes
    leader_id: str | None
    role: Any  # NodeRole
    leader_state_lock: Any  # threading.RLock
    leader_lease_id: str
    leader_lease_expires: float
    last_lease_renewal: float
    _leadership_sm: Any  # LeadershipStateMachine
    _last_step_down_time: float

    def _schedule_step_down_sync(self, reason: "TransitionReason") -> None:
        """Schedule async step-down from sync context (e.g., _is_leader()).

        This is a sync-to-async bridge that schedules the full ULSM step-down
        process via asyncio.create_task(). The step-down includes:
        1. State machine transition to STEPPING_DOWN (broadcasts to peers)
        2. Brief delay for broadcast propagation
        3. Transition to FOLLOWER
        4. Legacy field synchronization

        Args:
            reason: Why we're stepping down (LEASE_EXPIRED, QUORUM_LOST, etc.)
        """
        # Feb 2026: Don't step down if forced leader override is active
        _forced_sd = getattr(self, "_forced_leader_override", False)
        _lease_sd = time.time() < getattr(self, "leader_lease_expires", 0)
        if _forced_sd and _lease_sd:
            logger.warning(
                f"ULSM: Blocking step-down (reason={reason.value}) — "
                f"forced leader override active"
            )
            return
        try:
            asyncio.get_running_loop()  # Guard: safe_create_task needs a running loop
            safe_create_task(self._complete_step_down_async(reason), name="ulsm-step-down")
            logger.info(f"ULSM: Scheduled step-down for reason={reason.value}")
        except RuntimeError:
            # No running loop - this shouldn't happen in normal operation
            logger.warning("ULSM: No event loop to schedule step-down")

    async def _complete_step_down_async(self, reason: "TransitionReason") -> None:
        """Complete the step-down process via ULSM state machine.

        This is the async implementation that:
        1. Transitions to STEPPING_DOWN (triggers broadcast to peers)
        2. Waits briefly for broadcast propagation
        3. Transitions to FOLLOWER
        4. Syncs legacy fields and saves state

        Args:
            reason: Why we're stepping down
        """
        try:
            # Import here to avoid circular imports
            from scripts.p2p.ulsm import LeaderState, TransitionReason as TR

            # Import NodeRole
            try:
                from scripts.p2p.models import NodeRole
            except ImportError:
                class NodeRole:
                    FOLLOWER = "follower"

            # Step 1: Transition to STEPPING_DOWN (triggers broadcast)
            old_leader_id = self.leader_id
            success = await self._leadership_sm.transition_to(
                LeaderState.STEPPING_DOWN,
                reason,
            )
            if not success:
                logger.warning(
                    f"ULSM: Failed transition to STEPPING_DOWN, "
                    f"current state={self._leadership_sm.state}"
                )
                return

            # Step 2: Brief delay to allow broadcast propagation
            await asyncio.sleep(0.5)

            # Step 3: Transition to FOLLOWER
            await self._leadership_sm.transition_to(
                LeaderState.FOLLOWER,
                TR.STEP_DOWN_COMPLETE,
            )

            # Step 4: Sync legacy fields (for backward compatibility)
            with self.leader_state_lock:
                self.role = NodeRole.FOLLOWER
                self.leader_id = None
                self.leader_lease_id = ""
                self.leader_lease_expires = 0.0
                self.last_lease_renewal = 0.0
                # Reset ULSM QuorumHealth
                if self._leadership_sm:
                    self._leadership_sm.quorum_health.reset()
                # Track when we stepped down for leader stickiness
                self._last_step_down_time = time.time()

            self._release_voter_grant_if_self()
            self._save_state()

            # Emit LEADER_LOST event for other components
            self._emit_leader_lost_sync(old_leader_id, reason.value)

            logger.info(f"ULSM: Step-down complete, reason={reason.value}")

            # Step 5: Trigger new election for certain step-down reasons
            # Skip for QUORUM_LOST since election would fail anyway
            if reason != TR.QUORUM_LOST:
                if getattr(self, "voter_node_ids", []) and not self._has_voter_quorum():
                    logger.warning("ULSM: Skipping post-step-down election - no voter quorum")
                else:
                    logger.info("ULSM: Starting election after step-down")
                    safe_create_task(self._start_election(), name="ulsm-post-stepdown-election")

        except Exception as e:
            logger.error(f"ULSM: Error during step-down: {e}", exc_info=True)
            # Fallback: Force follower state
            try:
                from scripts.p2p.models import NodeRole
            except ImportError:
                class NodeRole:
                    FOLLOWER = "follower"

            with self.leader_state_lock:
                self.role = NodeRole.FOLLOWER
                self.leader_id = None
            self._save_state()
