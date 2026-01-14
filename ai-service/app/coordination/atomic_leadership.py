"""Atomic Leadership Transitions - Two-Phase Commit for Leader Election.

January 2026: Created as part of distributed architecture Phase 3.

Problem:
    Leader election takes ~100s (detection 60s + gossip convergence 40s).
    During transition, split-brain is possible as different nodes may
    have different views of who the leader is.

Solution:
    Two-phase commit for leadership transfers:
    1. PREPARE: Lock all voters, get acknowledgment
    2. COMMIT: If quorum agrees, install new leader atomically
    3. ABORT: If prepare fails, roll back

This reduces failover time and prevents split-brain by ensuring
all voters agree before the transition completes.

Usage:
    from app.coordination.atomic_leadership import (
        AtomicLeadershipCoordinator,
        get_leadership_coordinator,
    )

    coordinator = get_leadership_coordinator()

    # Transfer leadership with atomic commit
    success = await coordinator.transfer_leadership(
        old_leader="node-1",
        new_leader="node-2",
        voters=["node-1", "node-2", "node-3", "node-4", "node-5"],
    )

    # Handle leadership prepare request (from another node)
    async def handle_prepare(request):
        result = await coordinator.handle_prepare_request(request)
        return result
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TransitionState(Enum):
    """State of a leadership transition."""

    IDLE = "idle"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTED = "aborted"


class PrepareResponse(Enum):
    """Response to prepare request."""

    ACK = "ack"  # Ready to accept new leader
    NACK = "nack"  # Cannot accept (already in transition, etc.)
    ERROR = "error"  # Error processing request


@dataclass
class TransitionRecord:
    """Record of a leadership transition attempt."""

    transition_id: str
    old_leader: str
    new_leader: str
    term: int
    state: TransitionState = TransitionState.IDLE
    prepare_votes: dict[str, PrepareResponse] = field(default_factory=dict)
    commit_votes: dict[str, bool] = field(default_factory=dict)
    started_at: float = 0.0
    prepared_at: float = 0.0
    completed_at: float = 0.0
    error: str = ""


@dataclass
class AtomicLeadershipConfig:
    """Configuration for atomic leadership transitions."""

    # Timeouts
    prepare_timeout_seconds: float = 10.0
    commit_timeout_seconds: float = 10.0
    transition_timeout_seconds: float = 30.0

    # Quorum settings
    require_strict_quorum: bool = True  # Require majority for prepare

    # Retry settings
    max_prepare_retries: int = 2
    retry_delay_seconds: float = 1.0

    # Lock settings
    lock_duration_seconds: float = 30.0  # How long voters stay locked


class AtomicLeadershipCoordinator:
    """Coordinates atomic leadership transitions using two-phase commit.

    Ensures leadership changes are atomic across the cluster by:
    1. Preparing all voters (locking them)
    2. Committing if quorum agrees
    3. Aborting and rolling back if prepare fails
    """

    def __init__(self, config: AtomicLeadershipConfig | None = None):
        """Initialize leadership coordinator."""
        self.config = config or AtomicLeadershipConfig()

        # Current state
        self._current_leader: str = ""
        self._current_term: int = 0
        self._node_id: str = self._get_node_id()

        # Transition tracking
        self._active_transition: TransitionRecord | None = None
        self._transition_history: list[TransitionRecord] = []
        self._lock = asyncio.Lock()

        # Voter lock state (for when we're a voter)
        self._locked_for_transition: str | None = None  # transition_id
        self._lock_expires_at: float = 0.0

    def _get_node_id(self) -> str:
        """Get this node's identifier."""
        try:
            from app.config.node_identity import get_node_id_safe

            return get_node_id_safe()
        except ImportError:
            import os
            import socket

            return os.getenv("RINGRIFT_NODE_ID", socket.gethostname())

    async def transfer_leadership(
        self,
        old_leader: str,
        new_leader: str,
        voters: list[str],
        term: int | None = None,
    ) -> bool:
        """Atomically transfer leadership from old_leader to new_leader.

        Args:
            old_leader: Current leader node_id
            new_leader: New leader node_id
            voters: List of voter node_ids
            term: Optional new term number (auto-incremented if not provided)

        Returns:
            True if transfer succeeded, False otherwise
        """
        async with self._lock:
            if self._active_transition is not None:
                logger.warning(
                    f"Transition already in progress: {self._active_transition.transition_id}"
                )
                return False

            # Create transition record
            transition_id = str(uuid.uuid4())[:8]
            new_term = term if term is not None else self._current_term + 1

            transition = TransitionRecord(
                transition_id=transition_id,
                old_leader=old_leader,
                new_leader=new_leader,
                term=new_term,
                state=TransitionState.PREPARING,
                started_at=time.time(),
            )
            self._active_transition = transition

            logger.info(
                f"Starting leadership transition {transition_id}: "
                f"{old_leader} -> {new_leader} (term {new_term})"
            )

        try:
            # Phase 1: PREPARE - Lock all voters
            prepare_success = await self._phase_prepare(transition, voters)

            if not prepare_success:
                await self._phase_abort(transition, voters)
                return False

            # Phase 2: COMMIT - Install new leader
            commit_success = await self._phase_commit(transition, voters)

            if not commit_success:
                await self._phase_abort(transition, voters)
                return False

            # Success!
            transition.state = TransitionState.COMMITTED
            transition.completed_at = time.time()
            self._current_leader = new_leader
            self._current_term = new_term

            duration = transition.completed_at - transition.started_at
            logger.info(
                f"Leadership transition {transition_id} completed in {duration:.1f}s"
            )

            self._emit_leader_changed_event(transition)
            return True

        except Exception as e:
            logger.error(f"Leadership transition failed: {e}")
            transition.state = TransitionState.ABORTED
            transition.error = str(e)
            await self._phase_abort(transition, voters)
            return False

        finally:
            async with self._lock:
                self._transition_history.append(transition)
                self._active_transition = None

    async def _phase_prepare(
        self, transition: TransitionRecord, voters: list[str]
    ) -> bool:
        """Phase 1: Prepare voters for transition.

        Returns:
            True if quorum of voters acknowledged prepare
        """
        logger.debug(f"[{transition.transition_id}] Phase 1: PREPARE")

        # Send prepare to all voters in parallel
        prepare_tasks = [
            self._send_prepare(voter, transition) for voter in voters
        ]

        results = await asyncio.gather(*prepare_tasks, return_exceptions=True)

        # Count acks
        ack_count = 0
        for voter, result in zip(voters, results):
            if isinstance(result, Exception):
                transition.prepare_votes[voter] = PrepareResponse.ERROR
                logger.debug(f"Prepare failed for {voter}: {result}")
            elif result == PrepareResponse.ACK:
                transition.prepare_votes[voter] = PrepareResponse.ACK
                ack_count += 1
            else:
                transition.prepare_votes[voter] = result

        # Check quorum
        quorum_size = len(voters) // 2 + 1
        has_quorum = ack_count >= quorum_size

        logger.info(
            f"[{transition.transition_id}] Prepare: {ack_count}/{len(voters)} ACKs "
            f"(quorum={quorum_size}, success={has_quorum})"
        )

        if has_quorum:
            transition.state = TransitionState.PREPARED
            transition.prepared_at = time.time()

        return has_quorum

    async def _phase_commit(
        self, transition: TransitionRecord, voters: list[str]
    ) -> bool:
        """Phase 2: Commit new leader to all voters.

        Returns:
            True if quorum of voters committed
        """
        logger.debug(f"[{transition.transition_id}] Phase 2: COMMIT")
        transition.state = TransitionState.COMMITTING

        # Send commit to all voters in parallel
        commit_tasks = [
            self._send_commit(voter, transition) for voter in voters
        ]

        results = await asyncio.gather(*commit_tasks, return_exceptions=True)

        # Count commits
        commit_count = 0
        for voter, result in zip(voters, results):
            if isinstance(result, Exception):
                transition.commit_votes[voter] = False
                logger.debug(f"Commit failed for {voter}: {result}")
            elif result:
                transition.commit_votes[voter] = True
                commit_count += 1
            else:
                transition.commit_votes[voter] = False

        # Check quorum
        quorum_size = len(voters) // 2 + 1
        has_quorum = commit_count >= quorum_size

        logger.info(
            f"[{transition.transition_id}] Commit: {commit_count}/{len(voters)} "
            f"(quorum={quorum_size}, success={has_quorum})"
        )

        return has_quorum

    async def _phase_abort(
        self, transition: TransitionRecord, voters: list[str]
    ) -> None:
        """Abort phase: Unlock all voters."""
        logger.info(f"[{transition.transition_id}] ABORT: Rolling back")
        transition.state = TransitionState.ABORTED

        # Send abort to all voters (best effort)
        abort_tasks = [self._send_abort(voter, transition) for voter in voters]
        await asyncio.gather(*abort_tasks, return_exceptions=True)

    async def _send_prepare(
        self, voter: str, transition: TransitionRecord
    ) -> PrepareResponse:
        """Send prepare request to a voter."""
        try:
            import aiohttp

            # Find voter endpoint
            voter_url = await self._get_voter_url(voter)
            if not voter_url:
                return PrepareResponse.ERROR

            payload = {
                "transition_id": transition.transition_id,
                "old_leader": transition.old_leader,
                "new_leader": transition.new_leader,
                "term": transition.term,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{voter_url}/leadership/prepare",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.prepare_timeout_seconds),
                ) as resp:
                    if resp.status == 200:
                        return PrepareResponse.ACK
                    elif resp.status == 409:  # Conflict - already in transition
                        return PrepareResponse.NACK
                    else:
                        return PrepareResponse.ERROR

        except asyncio.TimeoutError:
            return PrepareResponse.ERROR
        except Exception as e:
            logger.debug(f"Prepare to {voter} failed: {e}")
            return PrepareResponse.ERROR

    async def _send_commit(
        self, voter: str, transition: TransitionRecord
    ) -> bool:
        """Send commit request to a voter."""
        try:
            import aiohttp

            voter_url = await self._get_voter_url(voter)
            if not voter_url:
                return False

            payload = {
                "transition_id": transition.transition_id,
                "new_leader": transition.new_leader,
                "term": transition.term,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{voter_url}/leadership/commit",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.commit_timeout_seconds),
                ) as resp:
                    return resp.status == 200

        except Exception as e:
            logger.debug(f"Commit to {voter} failed: {e}")
            return False

    async def _send_abort(
        self, voter: str, transition: TransitionRecord
    ) -> bool:
        """Send abort request to a voter."""
        try:
            import aiohttp

            voter_url = await self._get_voter_url(voter)
            if not voter_url:
                return False

            payload = {"transition_id": transition.transition_id}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{voter_url}/leadership/abort",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    return resp.status == 200

        except Exception:
            return False

    async def _get_voter_url(self, voter: str) -> str | None:
        """Get HTTP URL for a voter node."""
        try:
            # Try P2P peer lookup
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:8770/status",
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        peers = data.get("peers", {})
                        if voter in peers:
                            peer = peers[voter]
                            host = peer.get("host", "")
                            port = peer.get("port", 8770)
                            if host:
                                return f"http://{host}:{port}"

            # Try config lookup
            from app.config.cluster_config import load_cluster_config

            config = load_cluster_config()
            nodes = getattr(config, "hosts_raw", {}) or {}
            node_cfg = nodes.get(voter, {})
            if node_cfg:
                host = node_cfg.get("tailscale_ip") or node_cfg.get("ssh_host")
                if host:
                    return f"http://{host}:8770"

        except Exception as e:
            logger.debug(f"Could not get URL for voter {voter}: {e}")

        return None

    # === Voter-side handlers ===

    async def handle_prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle prepare request from coordinator.

        Returns:
            Response dict with "status": "ack", "nack", or "error"
        """
        transition_id = request.get("transition_id", "")
        new_leader = request.get("new_leader", "")
        term = request.get("term", 0)

        async with self._lock:
            # Check if we're already locked for another transition
            if self._locked_for_transition and time.time() < self._lock_expires_at:
                if self._locked_for_transition != transition_id:
                    return {"status": "nack", "reason": "already_locked"}

            # Lock for this transition
            self._locked_for_transition = transition_id
            self._lock_expires_at = time.time() + self.config.lock_duration_seconds

            logger.info(
                f"Prepared for transition {transition_id}: "
                f"new_leader={new_leader}, term={term}"
            )

            return {"status": "ack", "voter": self._node_id}

    async def handle_commit_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle commit request from coordinator.

        Returns:
            Response dict with "status": "ok" or "error"
        """
        transition_id = request.get("transition_id", "")
        new_leader = request.get("new_leader", "")
        term = request.get("term", 0)

        async with self._lock:
            # Verify we're locked for this transition
            if self._locked_for_transition != transition_id:
                return {"status": "error", "reason": "not_prepared"}

            # Update our view of the leader
            self._current_leader = new_leader
            self._current_term = term

            # Clear lock
            self._locked_for_transition = None
            self._lock_expires_at = 0.0

            logger.info(f"Committed leader change: {new_leader} (term {term})")

            return {"status": "ok", "voter": self._node_id}

    async def handle_abort_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle abort request from coordinator.

        Returns:
            Response dict with "status": "ok"
        """
        transition_id = request.get("transition_id", "")

        async with self._lock:
            if self._locked_for_transition == transition_id:
                self._locked_for_transition = None
                self._lock_expires_at = 0.0
                logger.info(f"Aborted transition {transition_id}")

            return {"status": "ok", "voter": self._node_id}

    def _emit_leader_changed_event(self, transition: TransitionRecord) -> None:
        """Emit event when leader changes."""
        try:
            from app.coordination.event_router import emit_event
            from app.distributed.data_events import DataEventType

            emit_event(
                DataEventType.LEADER_ELECTED,
                {
                    "old_leader": transition.old_leader,
                    "new_leader": transition.new_leader,
                    "term": transition.term,
                    "transition_id": transition.transition_id,
                    "duration_seconds": transition.completed_at - transition.started_at,
                    "source": "atomic_leadership",
                },
            )
        except ImportError:
            pass

    def get_status(self) -> dict[str, Any]:
        """Get current leadership status."""
        return {
            "node_id": self._node_id,
            "current_leader": self._current_leader,
            "current_term": self._current_term,
            "active_transition": (
                {
                    "id": self._active_transition.transition_id,
                    "state": self._active_transition.state.value,
                    "new_leader": self._active_transition.new_leader,
                }
                if self._active_transition
                else None
            ),
            "locked_for_transition": self._locked_for_transition,
            "lock_expires_at": self._lock_expires_at,
            "transition_count": len(self._transition_history),
        }


# Singleton instance
_instance: AtomicLeadershipCoordinator | None = None


def get_leadership_coordinator() -> AtomicLeadershipCoordinator:
    """Get the singleton leadership coordinator."""
    global _instance
    if _instance is None:
        _instance = AtomicLeadershipCoordinator()
    return _instance


def reset_leadership_coordinator() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
