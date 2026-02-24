"""Two-Phase Leadership Commitment Protocol.

January 2026: Created as part of P2P critical hardening (Phase 1).
Problem: After leader election, the cluster can resume operations before all
voters agree on the leader, leading to split-brain scenarios.

Solution: Two-phase commitment protocol:
1. Phase 1 (Election): Leader is elected via existing bully/Raft algorithm
2. Phase 2 (Commitment): Leader requests commitment acks from voters
   - Voters only ack if they agree on this leader
   - Operations resume ONLY after majority ack received

Usage:
    from scripts.p2p.two_phase_leader_commit import TwoPhaseLeaderCommit

    # After winning election
    commit_protocol = TwoPhaseLeaderCommit(
        node_id=self.node_id,
        voters=self.voter_node_ids,
        request_ack_fn=self._request_commitment_ack,
    )

    # Try to commit leadership (blocks until done)
    result = await commit_protocol.commit_leadership(leader_id=self.node_id)
    if result.committed:
        # Safe to resume operations - majority agrees
        self._resume_operations()
    else:
        # Failed to get commitment - step down
        self._step_down()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from app.core.async_context import safe_create_task

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CommitmentState(Enum):
    """States of the commitment protocol."""

    PENDING = "pending"  # Waiting for commitment
    COMMITTED = "committed"  # Majority acked
    REJECTED = "rejected"  # Failed to get majority
    EXPIRED = "expired"  # Timeout waiting for acks


@dataclass
class CommitmentConfig:
    """Configuration for two-phase commitment protocol."""

    # Minimum acks needed (quorum)
    quorum_size: int = 4  # For 7 voters, need 4

    # Timeout waiting for acks
    commitment_timeout: float = 30.0  # 30 seconds

    # How long a commitment is valid
    commitment_ttl: float = 120.0  # 2 minutes

    # Retry delay between ack requests
    ack_retry_delay: float = 2.0

    # Max retries per voter
    max_retries_per_voter: int = 3


@dataclass
class CommitmentResult:
    """Result of commitment attempt."""

    committed: bool
    leader_id: str
    acks_received: int
    acks_needed: int
    acked_voters: list[str] = field(default_factory=list)
    rejected_voters: list[str] = field(default_factory=list)
    timeout_voters: list[str] = field(default_factory=list)
    commitment_time: float = 0.0
    error: str | None = None


@dataclass
class VoterAck:
    """Acknowledgment from a voter."""

    voter_id: str
    agreed: bool
    leader_seen: str | None = None
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


class TwoPhaseLeaderCommit:
    """Two-phase leadership commitment protocol.

    After a leader is elected, this protocol ensures that a majority of voters
    agree on the leader before operations resume. This prevents split-brain
    scenarios where different parts of the cluster see different leaders.

    The protocol:
    1. Leader sends commitment request to all voters
    2. Each voter responds with ack (agree) or nack (disagree)
    3. If majority acks received within timeout, leadership is committed
    4. If not, leader steps down and triggers new election

    Thread Safety:
        This class is designed to be used from a single async context.
        For multi-threaded use, external synchronization is required.
    """

    def __init__(
        self,
        node_id: str,
        voters: list[str],
        request_ack_fn: Callable[[str, str], Awaitable[VoterAck | None]],
        config: CommitmentConfig | None = None,
    ):
        """Initialize commitment protocol.

        Args:
            node_id: This node's ID
            voters: List of voter node IDs
            request_ack_fn: Async function to request ack from a voter
                           Signature: (voter_id, leader_id) -> VoterAck | None
            config: Protocol configuration
        """
        self.node_id = node_id
        self.voters = voters
        self.request_ack_fn = request_ack_fn
        self.config = config or CommitmentConfig()

        self._state = CommitmentState.PENDING
        self._commitment_time: float = 0.0
        self._acked_voters: list[str] = []
        self._rejected_voters: list[str] = []
        self._timeout_voters: list[str] = []

    @property
    def state(self) -> CommitmentState:
        """Current commitment state."""
        return self._state

    @property
    def is_committed(self) -> bool:
        """Check if leadership is committed."""
        if self._state != CommitmentState.COMMITTED:
            return False
        # Check if commitment is still valid (not expired)
        if time.time() - self._commitment_time > self.config.commitment_ttl:
            self._state = CommitmentState.EXPIRED
            return False
        return True

    def _calculate_quorum_needed(self) -> int:
        """Calculate number of acks needed for commitment."""
        # Majority of voters
        return (len(self.voters) // 2) + 1

    async def commit_leadership(self, leader_id: str) -> CommitmentResult:
        """Attempt to commit leadership by getting voter acks.

        This is the main entry point. It contacts all voters in parallel,
        collects acks, and returns whether commitment succeeded.

        Args:
            leader_id: ID of the leader to commit

        Returns:
            CommitmentResult with outcome and details
        """
        start_time = time.time()
        quorum_needed = self._calculate_quorum_needed()

        logger.info(
            f"[TwoPhaseCommit] Starting commitment for leader {leader_id}, "
            f"need {quorum_needed}/{len(self.voters)} acks"
        )

        # Reset state
        self._state = CommitmentState.PENDING
        self._acked_voters = []
        self._rejected_voters = []
        self._timeout_voters = []

        # Request acks from all voters in parallel
        tasks = []
        for voter_id in self.voters:
            task = safe_create_task(
                self._request_ack_with_retry(voter_id, leader_id),
                name=f"two-phase-commit-ack-{voter_id}",
            )
            tasks.append((voter_id, task))

        # Wait for all with timeout
        try:
            done, pending = await asyncio.wait(
                [t for _, t in tasks],
                timeout=self.config.commitment_timeout,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"[TwoPhaseCommit] Error during commitment: {e}")
            self._state = CommitmentState.REJECTED
            return CommitmentResult(
                committed=False,
                leader_id=leader_id,
                acks_received=len(self._acked_voters),
                acks_needed=quorum_needed,
                acked_voters=self._acked_voters,
                rejected_voters=self._rejected_voters,
                timeout_voters=self._timeout_voters,
                error=str(e),
            )

        # Check results
        acks_received = len(self._acked_voters)

        if acks_received >= quorum_needed:
            self._state = CommitmentState.COMMITTED
            self._commitment_time = time.time()
            logger.info(
                f"[TwoPhaseCommit] Leadership COMMITTED for {leader_id} "
                f"with {acks_received}/{quorum_needed} acks in "
                f"{time.time() - start_time:.2f}s"
            )
        else:
            self._state = CommitmentState.REJECTED
            logger.warning(
                f"[TwoPhaseCommit] Leadership REJECTED for {leader_id}: "
                f"only {acks_received}/{quorum_needed} acks received"
            )

        return CommitmentResult(
            committed=self._state == CommitmentState.COMMITTED,
            leader_id=leader_id,
            acks_received=acks_received,
            acks_needed=quorum_needed,
            acked_voters=list(self._acked_voters),
            rejected_voters=list(self._rejected_voters),
            timeout_voters=list(self._timeout_voters),
            commitment_time=self._commitment_time if self._state == CommitmentState.COMMITTED else 0.0,
        )

    async def _request_ack_with_retry(
        self,
        voter_id: str,
        leader_id: str,
    ) -> VoterAck | None:
        """Request ack from a voter with retries.

        Args:
            voter_id: Voter to request ack from
            leader_id: Leader to commit

        Returns:
            VoterAck if received, None if timeout/error
        """
        for attempt in range(self.config.max_retries_per_voter):
            try:
                ack = await self.request_ack_fn(voter_id, leader_id)

                if ack is None:
                    # Timeout or error
                    if attempt < self.config.max_retries_per_voter - 1:
                        await asyncio.sleep(self.config.ack_retry_delay)
                        continue
                    self._timeout_voters.append(voter_id)
                    return None

                if ack.agreed:
                    self._acked_voters.append(voter_id)
                    logger.debug(f"[TwoPhaseCommit] Got ACK from {voter_id}")
                else:
                    self._rejected_voters.append(voter_id)
                    logger.debug(
                        f"[TwoPhaseCommit] Got NACK from {voter_id}: {ack.reason}"
                    )

                return ack

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"[TwoPhaseCommit] Error requesting ack from {voter_id}: {e}"
                )
                if attempt < self.config.max_retries_per_voter - 1:
                    await asyncio.sleep(self.config.ack_retry_delay)

        # All retries exhausted
        self._timeout_voters.append(voter_id)
        return None

    def refresh_commitment(self) -> bool:
        """Refresh commitment TTL (like lease renewal).

        Returns:
            True if refreshed, False if expired or not committed
        """
        if self._state != CommitmentState.COMMITTED:
            return False

        if time.time() - self._commitment_time > self.config.commitment_ttl:
            self._state = CommitmentState.EXPIRED
            return False

        self._commitment_time = time.time()
        return True

    def invalidate(self) -> None:
        """Invalidate current commitment (e.g., on step down)."""
        self._state = CommitmentState.PENDING
        self._commitment_time = 0.0
        self._acked_voters = []

    def get_status(self) -> dict[str, Any]:
        """Get current commitment status for monitoring."""
        now = time.time()
        return {
            "state": self._state.value,
            "is_committed": self.is_committed,
            "commitment_age": now - self._commitment_time if self._commitment_time else None,
            "commitment_ttl_remaining": (
                max(0, self.config.commitment_ttl - (now - self._commitment_time))
                if self._commitment_time else None
            ),
            "acked_voters": list(self._acked_voters),
            "rejected_voters": list(self._rejected_voters),
            "timeout_voters": list(self._timeout_voters),
            "quorum_size": self._calculate_quorum_needed(),
            "total_voters": len(self.voters),
        }


# =============================================================================
# Integration Helpers
# =============================================================================


def create_commitment_ack_handler(orchestrator: Any) -> Callable:
    """Create an HTTP handler for commitment ack requests.

    This is registered as POST /election/commitment-ack

    Args:
        orchestrator: P2POrchestrator instance

    Returns:
        Async handler function
    """
    from aiohttp import web

    async def handle_commitment_ack(request: web.Request) -> web.Response:
        """Handle commitment ack request from leader candidate.

        Request body:
            {
                "leader_id": "node-1",
                "lease_id": "uuid-1234"
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
            data = await request.json()
            leader_id = data.get("leader_id", "")
            lease_id = data.get("lease_id", "")

            if not leader_id:
                return web.json_response({
                    "agreed": False,
                    "voter_id": orchestrator.node_id,
                    "reason": "missing_leader_id",
                })

            # Check if this voter agrees on the leader
            current_leader = orchestrator.leader_id
            agreed = current_leader == leader_id or current_leader is None

            # Additional checks
            if agreed:
                # Check if we're in an election
                if getattr(orchestrator, "election_in_progress", False):
                    agreed = False
                    reason = "election_in_progress"
                else:
                    reason = ""
            else:
                reason = f"different_leader_seen:{current_leader}"

            return web.json_response({
                "agreed": agreed,
                "voter_id": orchestrator.node_id,
                "leader_seen": current_leader,
                "reason": reason,
            })

        except Exception as e:
            logger.error(f"[CommitmentAck] Error handling request: {e}")
            return web.json_response({
                "agreed": False,
                "voter_id": getattr(orchestrator, "node_id", "unknown"),
                "reason": f"error:{e}",
            }, status=500)

    return handle_commitment_ack


async def request_commitment_ack(
    orchestrator: Any,
    voter_id: str,
    leader_id: str,
    timeout: float = 10.0,
) -> VoterAck | None:
    """Request commitment ack from a voter.

    This is the default implementation that makes HTTP requests.

    Args:
        orchestrator: P2POrchestrator instance (for peer info)
        voter_id: Voter to request from
        leader_id: Leader to commit

    Returns:
        VoterAck if received, None on timeout/error
    """
    import aiohttp

    # Get voter peer info
    with orchestrator.peers_lock:
        peer = orchestrator.peers.get(voter_id)

    if not peer or not peer.is_alive():
        return None

    # Build URL
    host = getattr(peer, "tailscale_ip", None) or getattr(peer, "host", None)
    port = getattr(peer, "port", 8770)
    url = f"http://{host}:{port}/election/commitment-ack"

    try:
        async with aiohttp.ClientSession() as session:
            headers = {}
            if hasattr(orchestrator, "auth_token") and orchestrator.auth_token:
                headers["Authorization"] = f"Bearer {orchestrator.auth_token}"

            async with session.post(
                url,
                json={"leader_id": leader_id, "lease_id": ""},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()
                return VoterAck(
                    voter_id=data.get("voter_id", voter_id),
                    agreed=data.get("agreed", False),
                    leader_seen=data.get("leader_seen"),
                    reason=data.get("reason", ""),
                )

    except asyncio.TimeoutError:
        logger.debug(f"[CommitmentAck] Timeout requesting from {voter_id}")
        return None
    except Exception as e:
        logger.debug(f"[CommitmentAck] Error requesting from {voter_id}: {e}")
        return None


__all__ = [
    "TwoPhaseLeaderCommit",
    "CommitmentConfig",
    "CommitmentResult",
    "CommitmentState",
    "VoterAck",
    "create_commitment_ack_handler",
    "request_commitment_ack",
]
