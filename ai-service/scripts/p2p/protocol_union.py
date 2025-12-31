"""
Protocol Union for Maximum Membership Coverage.

Dec 30, 2025: Part of Phase 9 - Multi-Layer Failover Architecture.

Combines all membership protocols (Gossip, SWIM, Raft, Bully) into a unified view:
- Returns UNION of alive peers from all protocols
- Any protocol saying a peer is alive counts
- Provides confidence scoring based on agreement

This ensures maximum connectivity - if ANY protocol can reach a node, we include it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)


class MembershipSource(str, Enum):
    """Sources of membership information."""

    GOSSIP = "gossip"
    SWIM = "swim"
    RAFT = "raft"
    BULLY = "bully"
    HTTP_HEARTBEAT = "http_heartbeat"
    TAILSCALE = "tailscale"


@dataclass
class UnifiedMemberState:
    """Unified membership state for a peer.

    Combines information from all membership protocols.
    """

    node_id: str
    sources: set[MembershipSource] = field(default_factory=set)
    last_seen_by: dict[MembershipSource, float] = field(default_factory=dict)
    is_alive: bool = True
    is_suspected: bool = False
    is_voter: bool = False
    is_leader: bool = False
    addresses: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0-1, based on number of sources agreeing

    def __post_init__(self) -> None:
        """Compute confidence based on sources."""
        self._update_confidence()

    def _update_confidence(self) -> None:
        """Update confidence score based on number of sources."""
        if not self.sources:
            self.confidence = 0.0
        else:
            # Higher confidence with more sources agreeing
            # 1 source = 0.3, 2 sources = 0.5, 3 sources = 0.7, 4+ = 0.9
            source_count = len(self.sources)
            self.confidence = min(0.9, 0.2 + source_count * 0.2)

    def add_source(self, source: MembershipSource) -> None:
        """Add a source that has seen this peer."""
        self.sources.add(source)
        self.last_seen_by[source] = time.time()
        self._update_confidence()

    def get_freshest_source(self) -> tuple[MembershipSource | None, float]:
        """Get the most recent source that saw this peer."""
        if not self.last_seen_by:
            return None, 0.0
        source = max(self.last_seen_by.keys(), key=lambda s: self.last_seen_by[s])
        return source, self.last_seen_by[source]


class ProtocolUnion:
    """
    Union of all membership protocols for maximum coverage.

    Provides a unified view of cluster membership by combining:
    - Gossip protocol peer list
    - SWIM membership
    - Raft cluster members
    - Bully election participants

    Any protocol saying a peer is alive means we include it.
    """

    def __init__(
        self,
        orchestrator: Any | None = None,
        gossip_getter: Callable[[], list[str]] | None = None,
        swim_getter: Callable[[], list[str]] | None = None,
        raft_getter: Callable[[], list[str]] | None = None,
        bully_getter: Callable[[], list[str]] | None = None,
        http_getter: Callable[[], list[str]] | None = None,
    ):
        """Initialize protocol union.

        Args:
            orchestrator: P2POrchestrator instance for direct access
            gossip_getter: Callback to get Gossip alive peers
            swim_getter: Callback to get SWIM alive peers
            raft_getter: Callback to get Raft cluster members
            bully_getter: Callback to get Bully election participants
            http_getter: Callback to get HTTP heartbeat alive peers
        """
        self._orchestrator = orchestrator
        self._gossip_getter = gossip_getter
        self._swim_getter = swim_getter
        self._raft_getter = raft_getter
        self._bully_getter = bully_getter
        self._http_getter = http_getter

        # Cache of unified membership
        self._unified_members: dict[str, UnifiedMemberState] = {}
        self._last_update: float = 0.0
        self._update_interval: float = 5.0  # Seconds between full refreshes

    async def get_all_alive_peers(self) -> set[str]:
        """Get union of alive peers from all protocols.

        This is the core method - returns all peers that ANY protocol
        considers alive.

        Returns:
            Set of alive peer IDs from all sources
        """
        await self._refresh_membership()

        return {
            node_id
            for node_id, state in self._unified_members.items()
            if state.is_alive and not state.is_suspected
        }

    async def get_alive_peers_with_confidence(
        self, min_confidence: float = 0.0
    ) -> dict[str, float]:
        """Get alive peers with their confidence scores.

        Args:
            min_confidence: Minimum confidence threshold (0-1)

        Returns:
            Dict mapping node_id to confidence score
        """
        await self._refresh_membership()

        return {
            node_id: state.confidence
            for node_id, state in self._unified_members.items()
            if state.is_alive and state.confidence >= min_confidence
        }

    async def get_membership_by_source(self) -> dict[MembershipSource, set[str]]:
        """Get membership breakdown by protocol source.

        Returns:
            Dict mapping source to set of peer IDs
        """
        await self._refresh_membership()

        result: dict[MembershipSource, set[str]] = {
            source: set() for source in MembershipSource
        }

        for node_id, state in self._unified_members.items():
            for source in state.sources:
                result[source].add(node_id)

        return result

    async def get_high_confidence_peers(
        self, threshold: float = 0.5
    ) -> set[str]:
        """Get peers with high confidence (multiple sources agree).

        Args:
            threshold: Confidence threshold (default 0.5 = 2+ sources)

        Returns:
            Set of peer IDs with confidence >= threshold
        """
        peers = await self.get_alive_peers_with_confidence(threshold)
        return set(peers.keys())

    async def get_voters(self) -> set[str]:
        """Get voter nodes from Bully election."""
        await self._refresh_membership()
        return {
            node_id
            for node_id, state in self._unified_members.items()
            if state.is_voter
        }

    async def get_leader(self) -> str | None:
        """Get current leader from any protocol."""
        await self._refresh_membership()
        for node_id, state in self._unified_members.items():
            if state.is_leader:
                return node_id
        return None

    async def _refresh_membership(self) -> None:
        """Refresh membership from all protocols."""
        now = time.time()
        if now - self._last_update < self._update_interval:
            return

        # Gather from all sources in parallel
        results = await asyncio.gather(
            self._get_gossip_members(),
            self._get_swim_members(),
            self._get_raft_members(),
            self._get_bully_members(),
            self._get_http_members(),
            return_exceptions=True,
        )

        # Process results
        gossip_peers = results[0] if not isinstance(results[0], Exception) else []
        swim_peers = results[1] if not isinstance(results[1], Exception) else []
        raft_peers = results[2] if not isinstance(results[2], Exception) else []
        bully_peers = results[3] if not isinstance(results[3], Exception) else []
        http_peers = results[4] if not isinstance(results[4], Exception) else []

        # Merge into unified view
        self._merge_members(MembershipSource.GOSSIP, gossip_peers)
        self._merge_members(MembershipSource.SWIM, swim_peers)
        self._merge_members(MembershipSource.RAFT, raft_peers)
        self._merge_members(MembershipSource.BULLY, bully_peers)
        self._merge_members(MembershipSource.HTTP_HEARTBEAT, http_peers)

        self._last_update = now

        # Log summary
        total = len(self._unified_members)
        alive = sum(1 for s in self._unified_members.values() if s.is_alive)
        logger.debug(
            f"ProtocolUnion: {alive}/{total} alive peers from "
            f"gossip={len(gossip_peers)}, swim={len(swim_peers)}, "
            f"raft={len(raft_peers)}, bully={len(bully_peers)}, "
            f"http={len(http_peers)}"
        )

    def _merge_members(
        self, source: MembershipSource, peer_ids: list[str]
    ) -> None:
        """Merge peer list from a source into unified view."""
        for peer_id in peer_ids:
            if peer_id not in self._unified_members:
                self._unified_members[peer_id] = UnifiedMemberState(node_id=peer_id)
            self._unified_members[peer_id].add_source(source)
            self._unified_members[peer_id].is_alive = True

    async def _get_gossip_members(self) -> list[str]:
        """Get alive members from Gossip protocol."""
        if self._gossip_getter:
            return self._gossip_getter()

        if self._orchestrator:
            try:
                # Use handlers_utils helper
                from scripts.p2p.handlers.handlers_utils import get_alive_peers

                return get_alive_peers(
                    self._orchestrator.peers,
                    self._orchestrator.peers_lock,
                )
            except Exception as e:
                logger.debug(f"Failed to get Gossip members: {e}")

        return []

    async def _get_swim_members(self) -> list[str]:
        """Get alive members from SWIM protocol."""
        if self._swim_getter:
            return self._swim_getter()

        if self._orchestrator:
            try:
                # Check if orchestrator has SWIM manager
                if hasattr(self._orchestrator, "_swim_manager") and self._orchestrator._swim_manager:
                    return self._orchestrator._swim_manager.get_alive_peers()
                # Try hybrid method
                if hasattr(self._orchestrator, "get_alive_peers_hybrid"):
                    return self._orchestrator.get_alive_peers_hybrid()
            except Exception as e:
                logger.debug(f"Failed to get SWIM members: {e}")

        return []

    async def _get_raft_members(self) -> list[str]:
        """Get cluster members from Raft consensus."""
        if self._raft_getter:
            return self._raft_getter()

        if self._orchestrator:
            try:
                # Check if Raft is initialized
                if hasattr(self._orchestrator, "_raft_state") and self._orchestrator._raft_state:
                    raft = self._orchestrator._raft_state
                    if hasattr(raft, "getClusterNodes"):
                        nodes = raft.getClusterNodes()
                        return [str(n) for n in nodes]
            except Exception as e:
                logger.debug(f"Failed to get Raft members: {e}")

        return []

    async def _get_bully_members(self) -> list[str]:
        """Get participants from Bully election."""
        if self._bully_getter:
            return self._bully_getter()

        if self._orchestrator:
            try:
                # Voters are the core Bully participants
                voters = getattr(self._orchestrator, "voter_node_ids", [])

                # Also include leader and any nodes that have voted
                leader_id = getattr(self._orchestrator, "leader_id", None)

                result = list(voters)
                if leader_id and leader_id not in result:
                    result.append(leader_id)

                # Mark voters in unified state
                for voter_id in voters:
                    if voter_id in self._unified_members:
                        self._unified_members[voter_id].is_voter = True

                # Mark leader
                if leader_id and leader_id in self._unified_members:
                    self._unified_members[leader_id].is_leader = True

                return result
            except Exception as e:
                logger.debug(f"Failed to get Bully members: {e}")

        return []

    async def _get_http_members(self) -> list[str]:
        """Get alive members from HTTP heartbeat."""
        if self._http_getter:
            return self._http_getter()

        if self._orchestrator:
            try:
                # HTTP peers are in the standard peers dict
                from scripts.p2p.handlers.handlers_utils import get_alive_peers

                return get_alive_peers(
                    self._orchestrator.peers,
                    self._orchestrator.peers_lock,
                )
            except Exception as e:
                logger.debug(f"Failed to get HTTP members: {e}")

        return []

    async def broadcast_to_all(
        self,
        message: bytes,
        via_gossip: bool = True,
        via_swim: bool = True,
        via_raft: bool = True,
    ) -> dict[MembershipSource, bool]:
        """Broadcast message via all available protocols.

        Args:
            message: Message bytes to broadcast
            via_gossip: Use Gossip protocol
            via_swim: Use SWIM protocol
            via_raft: Use Raft replication

        Returns:
            Dict mapping protocol to success status
        """
        results: dict[MembershipSource, bool] = {}

        tasks = []
        if via_gossip:
            tasks.append(("gossip", self._broadcast_via_gossip(message)))
        if via_swim:
            tasks.append(("swim", self._broadcast_via_swim(message)))
        if via_raft:
            tasks.append(("raft", self._broadcast_via_raft(message)))

        for name, coro in tasks:
            try:
                await coro
                results[MembershipSource(name)] = True
            except Exception as e:
                logger.debug(f"Broadcast via {name} failed: {e}")
                results[MembershipSource(name)] = False

        return results

    async def _broadcast_via_gossip(self, message: bytes) -> None:
        """Broadcast via Gossip protocol."""
        if self._orchestrator and hasattr(self._orchestrator, "_gossip_broadcast"):
            await self._orchestrator._gossip_broadcast(message)

    async def _broadcast_via_swim(self, message: bytes) -> None:
        """Broadcast via SWIM protocol."""
        if self._orchestrator and hasattr(self._orchestrator, "_swim_manager"):
            swim = self._orchestrator._swim_manager
            if swim and hasattr(swim, "broadcast"):
                await swim.broadcast(message)

    async def _broadcast_via_raft(self, message: bytes) -> None:
        """Replicate via Raft consensus."""
        if self._orchestrator and hasattr(self._orchestrator, "_raft_state"):
            raft = self._orchestrator._raft_state
            if raft and hasattr(raft, "set"):
                # Store in Raft replicated dict
                import hashlib

                key = f"broadcast_{hashlib.sha256(message).hexdigest()[:16]}"
                raft.set(key, message.decode("utf-8", errors="replace"))

    def force_refresh(self) -> None:
        """Force a refresh on next access."""
        self._last_update = 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get protocol union statistics."""
        sources_count: dict[str, int] = {}
        for source in MembershipSource:
            sources_count[source.value] = sum(
                1
                for s in self._unified_members.values()
                if source in s.sources
            )

        return {
            "total_members": len(self._unified_members),
            "alive_members": sum(
                1 for s in self._unified_members.values() if s.is_alive
            ),
            "suspected_members": sum(
                1 for s in self._unified_members.values() if s.is_suspected
            ),
            "voters": sum(
                1 for s in self._unified_members.values() if s.is_voter
            ),
            "sources_count": sources_count,
            "last_update": self._last_update,
            "avg_confidence": (
                sum(s.confidence for s in self._unified_members.values())
                / len(self._unified_members)
                if self._unified_members
                else 0.0
            ),
        }


# Singleton instance
_protocol_union: ProtocolUnion | None = None


def get_protocol_union(orchestrator: Any = None) -> ProtocolUnion:
    """Get or create the ProtocolUnion singleton.

    Args:
        orchestrator: Optional P2POrchestrator to initialize with

    Returns:
        ProtocolUnion singleton instance
    """
    global _protocol_union
    if _protocol_union is None:
        _protocol_union = ProtocolUnion(orchestrator=orchestrator)
    elif orchestrator and _protocol_union._orchestrator is None:
        _protocol_union._orchestrator = orchestrator
    return _protocol_union


def reset_protocol_union() -> None:
    """Reset the singleton (for testing)."""
    global _protocol_union
    _protocol_union = None
