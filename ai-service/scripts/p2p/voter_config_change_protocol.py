"""Voter Configuration Change Protocol with Joint Consensus.

Jan 20, 2026: Implements safe voter list changes using joint consensus.
Both OLD and NEW voter sets must achieve quorum for the transition.

This ensures:
1. No split-brain during voter set transitions
2. Automatic rollback on timeout
3. Audit trail for all changes

Usage:
    protocol = VoterConfigChangeProtocol(orchestrator)
    result = await protocol.propose_change(
        new_voters=["node1", "node2", "node3"],
        reason="Adding hetzner-cpu3 to voter set"
    )
    if result.success:
        print(f"Change applied: v{result.new_version}")
    else:
        print(f"Change failed: {result.reason}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Configuration
CHANGE_TIMEOUT_SECONDS = float(120)  # 2 minutes
ACK_TIMEOUT_SECONDS = float(30)  # Per-node ack timeout
MIN_VOTERS = 3  # Minimum voters allowed


class ChangePhase(Enum):
    """Phases of the voter config change protocol."""
    IDLE = "idle"
    PROPOSING = "proposing"
    OLD_QUORUM = "old_quorum"  # Collecting acks from old voters
    NEW_QUORUM = "new_quorum"  # Collecting acks from new voters
    COMMITTING = "committing"
    COMMITTED = "committed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"


@dataclass
class ChangeProposal:
    """A proposed voter config change."""
    proposal_id: str
    proposer_id: str
    old_voters: list[str]
    new_voters: list[str]
    old_version: int
    new_version: int
    reason: str
    created_at: float = field(default_factory=time.time)
    phase: ChangePhase = ChangePhase.PROPOSING
    old_acks: set[str] = field(default_factory=set)
    new_acks: set[str] = field(default_factory=set)
    nacks: dict[str, str] = field(default_factory=dict)  # node_id -> reason


@dataclass
class ChangeResult:
    """Result of a voter config change attempt."""
    success: bool
    proposal_id: str
    old_version: int
    new_version: int
    old_voters: list[str]
    new_voters: list[str]
    reason: str
    phase_reached: ChangePhase
    old_acks: list[str] = field(default_factory=list)
    new_acks: list[str] = field(default_factory=list)
    nacks: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0


class VoterConfigChangeProtocol:
    """Protocol for safely changing voter configuration.

    Uses joint consensus: both old AND new voter sets must achieve quorum.

    Protocol phases:
    1. PROPOSING: Leader creates proposal with new voter list
    2. OLD_QUORUM: Collect acks from old voters (old quorum required)
    3. NEW_QUORUM: Collect acks from new voters (new quorum required)
    4. COMMITTING: Apply change and broadcast to all nodes
    5. COMMITTED: Change successfully applied

    If any phase fails or times out, automatic rollback occurs.
    """

    def __init__(self, orchestrator: Any):
        """Initialize the change protocol.

        Args:
            orchestrator: P2POrchestrator instance
        """
        self._orchestrator = orchestrator
        self._current_proposal: ChangeProposal | None = None
        self._lock = asyncio.Lock()

    @property
    def node_id(self) -> str:
        """Get this node's ID."""
        return getattr(self._orchestrator, "node_id", "unknown")

    def _is_leader(self) -> bool:
        """Check if this node is the current leader."""
        return getattr(self._orchestrator, "_is_leader", lambda: False)()

    async def propose_change(
        self,
        new_voters: list[str],
        reason: str = "",
    ) -> ChangeResult:
        """Propose a voter configuration change.

        Only the leader can propose changes. Uses joint consensus:
        both old and new voter sets must achieve quorum.

        Args:
            new_voters: New list of voter node IDs
            reason: Human-readable reason for the change

        Returns:
            ChangeResult with success/failure details
        """
        start_time = time.time()

        # Validate leader status
        if not self._is_leader():
            return ChangeResult(
                success=False,
                proposal_id="",
                old_version=0,
                new_version=0,
                old_voters=[],
                new_voters=new_voters,
                reason="not_leader",
                phase_reached=ChangePhase.IDLE,
            )

        # Validate new voter list
        if len(new_voters) < MIN_VOTERS:
            return ChangeResult(
                success=False,
                proposal_id="",
                old_version=0,
                new_version=0,
                old_voters=[],
                new_voters=new_voters,
                reason=f"insufficient_voters: need >= {MIN_VOTERS}",
                phase_reached=ChangePhase.IDLE,
            )

        # Check for duplicate voters
        if len(new_voters) != len(set(new_voters)):
            return ChangeResult(
                success=False,
                proposal_id="",
                old_version=0,
                new_version=0,
                old_voters=[],
                new_voters=new_voters,
                reason="duplicate_voters",
                phase_reached=ChangePhase.IDLE,
            )

        async with self._lock:
            if self._current_proposal is not None:
                return ChangeResult(
                    success=False,
                    proposal_id=self._current_proposal.proposal_id,
                    old_version=self._current_proposal.old_version,
                    new_version=0,
                    old_voters=self._current_proposal.old_voters,
                    new_voters=new_voters,
                    reason="change_already_in_progress",
                    phase_reached=self._current_proposal.phase,
                )

            # Get current config
            from scripts.p2p.managers.voter_config_manager import get_voter_config_manager
            manager = get_voter_config_manager()
            current_config = manager.get_current()

            if current_config is None:
                return ChangeResult(
                    success=False,
                    proposal_id="",
                    old_version=0,
                    new_version=0,
                    old_voters=[],
                    new_voters=new_voters,
                    reason="no_current_config",
                    phase_reached=ChangePhase.IDLE,
                )

            old_voters = current_config.voters
            old_version = current_config.version
            new_version = old_version + 1

            # Create proposal
            proposal_id = f"vc-{self.node_id}-{int(time.time())}"
            self._current_proposal = ChangeProposal(
                proposal_id=proposal_id,
                proposer_id=self.node_id,
                old_voters=old_voters,
                new_voters=new_voters,
                old_version=old_version,
                new_version=new_version,
                reason=reason,
                phase=ChangePhase.PROPOSING,
            )

            logger.info(
                f"[VoterConfigChange] Proposal {proposal_id}: "
                f"v{old_version} -> v{new_version}, "
                f"voters: {old_voters} -> {new_voters}"
            )

            try:
                # Phase 1: Get acks from OLD voters
                self._current_proposal.phase = ChangePhase.OLD_QUORUM
                old_quorum_size = (len(old_voters) // 2) + 1

                old_acks = await self._collect_acks(
                    voters=old_voters,
                    proposal=self._current_proposal,
                    phase="old",
                )

                self._current_proposal.old_acks = old_acks

                if len(old_acks) < old_quorum_size:
                    return await self._fail_proposal(
                        f"old_quorum_failed: {len(old_acks)}/{old_quorum_size}",
                        start_time,
                    )

                logger.info(
                    f"[VoterConfigChange] Old quorum achieved: "
                    f"{len(old_acks)}/{old_quorum_size}"
                )

                # Phase 2: Get acks from NEW voters
                self._current_proposal.phase = ChangePhase.NEW_QUORUM
                new_quorum_size = (len(new_voters) // 2) + 1

                new_acks = await self._collect_acks(
                    voters=new_voters,
                    proposal=self._current_proposal,
                    phase="new",
                )

                self._current_proposal.new_acks = new_acks

                if len(new_acks) < new_quorum_size:
                    return await self._fail_proposal(
                        f"new_quorum_failed: {len(new_acks)}/{new_quorum_size}",
                        start_time,
                    )

                logger.info(
                    f"[VoterConfigChange] New quorum achieved: "
                    f"{len(new_acks)}/{new_quorum_size}"
                )

                # Phase 3: Commit
                self._current_proposal.phase = ChangePhase.COMMITTING

                success = await self._commit_change(self._current_proposal)

                if not success:
                    return await self._fail_proposal("commit_failed", start_time)

                # Success!
                self._current_proposal.phase = ChangePhase.COMMITTED

                result = ChangeResult(
                    success=True,
                    proposal_id=proposal_id,
                    old_version=old_version,
                    new_version=new_version,
                    old_voters=old_voters,
                    new_voters=new_voters,
                    reason=reason,
                    phase_reached=ChangePhase.COMMITTED,
                    old_acks=list(old_acks),
                    new_acks=list(new_acks),
                    nacks=self._current_proposal.nacks,
                    duration_seconds=time.time() - start_time,
                )

                logger.info(
                    f"[VoterConfigChange] Change committed: "
                    f"v{old_version} -> v{new_version} in {result.duration_seconds:.1f}s"
                )

                # Emit success event
                await self._emit_event(
                    "VOTER_CONFIG_CHANGE_COMMITTED",
                    {
                        "proposal_id": proposal_id,
                        "old_version": old_version,
                        "new_version": new_version,
                        "old_voters": old_voters,
                        "new_voters": new_voters,
                        "reason": reason,
                        "duration_seconds": result.duration_seconds,
                    },
                )

                return result

            except asyncio.TimeoutError:
                return await self._fail_proposal("timeout", start_time)
            except Exception as e:
                logger.error(f"[VoterConfigChange] Unexpected error: {e}")
                return await self._fail_proposal(f"error: {e}", start_time)
            finally:
                self._current_proposal = None

    async def _collect_acks(
        self,
        voters: list[str],
        proposal: ChangeProposal,
        phase: str,
    ) -> set[str]:
        """Collect acknowledgments from voters.

        Args:
            voters: List of voter node IDs to collect from
            proposal: The change proposal
            phase: "old" or "new"

        Returns:
            Set of node IDs that acknowledged
        """
        acks: set[str] = set()

        # Self-ack if we're a voter
        if self.node_id in voters:
            acks.add(self.node_id)

        # Request acks from other voters in parallel
        peers = getattr(self._orchestrator, "peers", {})

        async def request_ack(voter_id: str) -> bool:
            if voter_id == self.node_id:
                return True

            peer_info = peers.get(voter_id)
            if not peer_info:
                proposal.nacks[voter_id] = "peer_not_found"
                return False

            url = self._get_peer_url(voter_id, peer_info, "/voter-config/ack")
            if not url:
                proposal.nacks[voter_id] = "no_url"
                return False

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json={
                            "proposal_id": proposal.proposal_id,
                            "phase": phase,
                            "old_version": proposal.old_version,
                            "new_version": proposal.new_version,
                            "old_voters": proposal.old_voters,
                            "new_voters": proposal.new_voters,
                        },
                        timeout=aiohttp.ClientTimeout(total=ACK_TIMEOUT_SECONDS),
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("ack"):
                                return True
                            proposal.nacks[voter_id] = data.get("reason", "nack")
                        else:
                            proposal.nacks[voter_id] = f"http_{response.status}"
                        return False
            except asyncio.TimeoutError:
                proposal.nacks[voter_id] = "timeout"
                return False
            except Exception as e:
                proposal.nacks[voter_id] = str(e)
                return False

        # Collect acks in parallel with overall timeout
        tasks = [request_ack(v) for v in voters if v != self.node_id]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=ACK_TIMEOUT_SECONDS * 1.5,
            )

            for i, voter_id in enumerate(v for v in voters if v != self.node_id):
                if i < len(results) and results[i] is True:
                    acks.add(voter_id)

        except asyncio.TimeoutError:
            logger.warning(f"[VoterConfigChange] Ack collection timed out for {phase} phase")

        return acks

    async def _commit_change(self, proposal: ChangeProposal) -> bool:
        """Commit the configuration change.

        Creates new VoterConfigVersion and broadcasts to all nodes.

        Args:
            proposal: The approved proposal

        Returns:
            True if commit succeeded
        """
        try:
            from app.coordination.voter_config_types import VoterConfigVersion
            from scripts.p2p.managers.voter_config_manager import get_voter_config_manager

            # Create new config
            new_config = VoterConfigVersion(
                version=proposal.new_version,
                voters=proposal.new_voters,
                created_at=time.time(),
                created_by=self.node_id,
            )

            # Apply locally first
            manager = get_voter_config_manager()
            success, reason = manager.apply_remote_config(
                new_config,
                source=f"change_protocol_{proposal.proposal_id}",
            )

            if not success and reason != "version_not_newer":
                # version_not_newer is OK if we already applied
                logger.error(f"[VoterConfigChange] Local apply failed: {reason}")
                return False

            # Broadcast to all peers (best effort)
            await self._broadcast_config(new_config)

            return True

        except Exception as e:
            logger.error(f"[VoterConfigChange] Commit error: {e}")
            return False

    async def _broadcast_config(self, config: Any) -> None:
        """Broadcast new config to all peers.

        Best effort - failures don't block commit.
        Nodes will sync via VoterConfigSyncLoop anyway.
        """
        peers = getattr(self._orchestrator, "peers", {})

        async def push_to_peer(peer_id: str, peer_info: Any) -> None:
            url = self._get_peer_url(peer_id, peer_info, "/voter-config/sync")
            if not url:
                return

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=config.to_dict(),
                        timeout=aiohttp.ClientTimeout(total=10.0),
                    ) as response:
                        if response.status == 200:
                            logger.debug(f"[VoterConfigChange] Pushed config to {peer_id}")
                        else:
                            logger.debug(
                                f"[VoterConfigChange] Push to {peer_id} failed: {response.status}"
                            )
            except Exception as e:
                logger.debug(f"[VoterConfigChange] Push to {peer_id} error: {e}")

        # Push in parallel, don't wait for all
        tasks = [
            push_to_peer(peer_id, peer_info)
            for peer_id, peer_info in peers.items()
            if peer_id != self.node_id
        ]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fail_proposal(
        self,
        reason: str,
        start_time: float,
    ) -> ChangeResult:
        """Handle proposal failure and cleanup.

        Args:
            reason: Failure reason
            start_time: When the proposal started

        Returns:
            ChangeResult with failure details
        """
        proposal = self._current_proposal
        if proposal is None:
            return ChangeResult(
                success=False,
                proposal_id="",
                old_version=0,
                new_version=0,
                old_voters=[],
                new_voters=[],
                reason=reason,
                phase_reached=ChangePhase.FAILED,
            )

        proposal.phase = ChangePhase.FAILED

        logger.warning(
            f"[VoterConfigChange] Proposal {proposal.proposal_id} failed: {reason}"
        )

        # Emit failure event
        await self._emit_event(
            "VOTER_CONFIG_CHANGE_FAILED",
            {
                "proposal_id": proposal.proposal_id,
                "old_version": proposal.old_version,
                "new_version": proposal.new_version,
                "reason": reason,
                "phase_reached": proposal.phase.value,
                "old_acks": list(proposal.old_acks),
                "new_acks": list(proposal.new_acks),
                "nacks": proposal.nacks,
            },
        )

        return ChangeResult(
            success=False,
            proposal_id=proposal.proposal_id,
            old_version=proposal.old_version,
            new_version=proposal.new_version,
            old_voters=proposal.old_voters,
            new_voters=proposal.new_voters,
            reason=reason,
            phase_reached=proposal.phase,
            old_acks=list(proposal.old_acks),
            new_acks=list(proposal.new_acks),
            nacks=proposal.nacks,
            duration_seconds=time.time() - start_time,
        )

    def _get_peer_url(self, peer_id: str, peer_info: Any, path: str) -> str | None:
        """Build URL for a peer endpoint.

        Args:
            peer_id: Node ID
            peer_info: NodeInfo object
            path: URL path (e.g., "/voter-config/ack")

        Returns:
            Full URL or None if can't determine
        """
        # Try Tailscale IP first
        tailscale_ip = getattr(peer_info, "tailscale_ip", None)
        if tailscale_ip:
            port = getattr(peer_info, "port", 8770)
            return f"http://{tailscale_ip}:{port}{path}"

        # Try endpoint
        endpoint = getattr(peer_info, "endpoint", None)
        if endpoint:
            if ":" in endpoint:
                return f"http://{endpoint}{path}"
            return f"http://{endpoint}:8770{path}"

        # Try IP address
        ip = getattr(peer_info, "ip", None)
        if ip:
            port = getattr(peer_info, "port", 8770)
            return f"http://{ip}:{port}{path}"

        return None

    async def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event via the orchestrator.

        Args:
            event_type: Event type name
            data: Event payload
        """
        try:
            emit_fn = getattr(self._orchestrator, "_safe_emit_event", None)
            if emit_fn:
                await asyncio.to_thread(
                    emit_fn,
                    event_type,
                    {**data, "timestamp": time.time()},
                )
        except Exception as e:
            logger.debug(f"[VoterConfigChange] Failed to emit {event_type}: {e}")

    async def handle_ack_request(
        self,
        proposal_id: str,
        phase: str,
        old_version: int,
        new_version: int,
        old_voters: list[str],
        new_voters: list[str],
    ) -> tuple[bool, str]:
        """Handle an incoming ack request from the leader.

        Called by HTTP handler when receiving /voter-config/ack POST.

        Args:
            proposal_id: Unique proposal ID
            phase: "old" or "new"
            old_version: Current config version
            new_version: Proposed new version
            old_voters: Current voter list
            new_voters: Proposed voter list

        Returns:
            Tuple of (ack: bool, reason: str)
        """
        try:
            from scripts.p2p.managers.voter_config_manager import get_voter_config_manager

            manager = get_voter_config_manager()
            current_config = manager.get_current()

            if current_config is None:
                return False, "no_local_config"

            # Verify version matches
            if current_config.version != old_version:
                return False, f"version_mismatch: local={current_config.version}"

            # Verify we're a voter in the appropriate phase
            am_old_voter = self.node_id in old_voters
            am_new_voter = self.node_id in new_voters

            if phase == "old" and not am_old_voter:
                return False, "not_old_voter"

            if phase == "new" and not am_new_voter:
                return False, "not_new_voter"

            # Validate new voters list
            if len(new_voters) < MIN_VOTERS:
                return False, f"insufficient_voters: {len(new_voters)} < {MIN_VOTERS}"

            # All checks passed
            logger.info(
                f"[VoterConfigChange] Acking proposal {proposal_id} "
                f"(phase={phase}, v{old_version} -> v{new_version})"
            )

            return True, "acked"

        except Exception as e:
            logger.error(f"[VoterConfigChange] Ack request error: {e}")
            return False, f"error: {e}"

    def get_status(self) -> dict[str, Any]:
        """Get current protocol status.

        Returns:
            Dict with protocol state
        """
        if self._current_proposal is None:
            return {
                "active": False,
                "phase": ChangePhase.IDLE.value,
            }

        p = self._current_proposal
        return {
            "active": True,
            "proposal_id": p.proposal_id,
            "proposer_id": p.proposer_id,
            "phase": p.phase.value,
            "old_version": p.old_version,
            "new_version": p.new_version,
            "old_voters": p.old_voters,
            "new_voters": p.new_voters,
            "old_acks": list(p.old_acks),
            "new_acks": list(p.new_acks),
            "nacks": p.nacks,
            "created_at": p.created_at,
            "age_seconds": time.time() - p.created_at,
        }
