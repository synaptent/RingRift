"""Relay-Aware Leader Propagation.

Jan 4, 2026 - Phase 1 of P2P Cluster Resilience.

Problem: NAT-blocked nodes (32/50) may not receive leader announcements directly
because they can only communicate via relay. This causes:
- Stale leader_id on NAT-blocked nodes
- Workers can't find work queue (returns early if no leader_id)
- QueuePopulatorLoop returns early on followers

Solution: Actively propagate leader_id via gossip state which IS relayed.
When a node becomes leader, it sets a dedicated `cluster_leader` entry in
gossip state that gets propagated to all nodes including NAT-blocked ones.

When nodes receive gossip, they check the cluster_leader entry and adopt
the leader if it has a valid lease and higher epoch than their current view.

Usage:
    class P2POrchestrator(RelayLeaderPropagatorMixin, GossipProtocolMixin, ...):
        pass

    # After becoming leader
    self._propagate_leader_via_gossip()

    # In gossip processing (automatic if mixin is used)
    # Leader is extracted from cluster_leader gossip entry
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.p2p.types import NodeRole

logger = logging.getLogger(__name__)


# Gossip key for leader propagation
GOSSIP_LEADER_KEY = "cluster_leader"

# How long a leader claim is valid in the gossip state (seconds)
LEADER_CLAIM_TTL_SECONDS = 300.0

# Minimum time between leader propagation broadcasts (seconds)
LEADER_PROPAGATION_COOLDOWN = 5.0


@dataclass
class LeaderClaim:
    """Represents a leader claim propagated via gossip.

    Attributes:
        leader_id: Node ID of the claimed leader
        epoch: Leadership epoch for ordering claims
        lease_expires: Unix timestamp when the leader's lease expires
        timestamp: When this claim was made
        source_node: Node that originated this claim
    """
    leader_id: str
    epoch: int
    lease_expires: float
    timestamp: float = field(default_factory=time.time)
    source_node: str = ""

    def is_valid(self) -> bool:
        """Check if this claim is still valid (lease not expired)."""
        return self.lease_expires > time.time()

    def is_fresher_than(self, other: "LeaderClaim | None") -> bool:
        """Check if this claim is fresher than another claim."""
        if other is None:
            return True
        # Higher epoch always wins
        if self.epoch > other.epoch:
            return True
        if self.epoch < other.epoch:
            return False
        # Same epoch: prefer newer timestamp
        return self.timestamp > other.timestamp

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for gossip state."""
        return {
            "leader_id": self.leader_id,
            "epoch": self.epoch,
            "lease_expires": self.lease_expires,
            "timestamp": self.timestamp,
            "source_node": self.source_node,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LeaderClaim | None":
        """Create from dictionary. Returns None if data is invalid."""
        if not data or not data.get("leader_id"):
            return None
        try:
            return cls(
                leader_id=data["leader_id"],
                epoch=int(data.get("epoch", 0)),
                lease_expires=float(data.get("lease_expires", 0)),
                timestamp=float(data.get("timestamp", 0)),
                source_node=str(data.get("source_node", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None


class RelayLeaderPropagatorMixin:
    """Mixin providing relay-aware leader propagation via gossip.

    This mixin enables NAT-blocked nodes to discover the leader through
    gossip state propagation, which works even when direct communication
    is blocked.

    Requires the implementing class to have:
    - node_id: str - This node's ID
    - leader_id: str | None - Current leader's ID
    - leader_lease_expires: float - Lease expiry timestamp
    - _gossip_peer_states: dict - Gossip state storage
    - _leadership_sm: Leadership state machine (optional, for epoch)

    Provides:
    - _propagate_leader_via_gossip() - Broadcast leader claim
    - _extract_leader_from_gossip() - Get leader from gossip states
    - _update_leader_from_gossip() - Adopt leader if fresher than current
    """

    # Type hints for IDE support (implemented by P2POrchestrator)
    node_id: str
    leader_id: str | None
    leader_lease_expires: float
    role: "NodeRole"
    _gossip_peer_states: dict[str, dict]

    # Track last propagation to avoid spamming
    _last_leader_propagation: float = 0.0

    def _propagate_leader_via_gossip(self, force: bool = False) -> bool:
        """Propagate current leadership via gossip state.

        Call this after becoming leader to ensure all nodes (including
        NAT-blocked ones) learn about the new leader through gossip.

        Args:
            force: If True, skip cooldown check

        Returns:
            True if propagation was performed
        """
        # Only the leader should propagate
        try:
            from scripts.p2p.types import NodeRole
            if self.role != NodeRole.LEADER:
                return False
        except ImportError:
            pass

        if self.leader_id != self.node_id:
            return False

        # Respect cooldown unless forced
        now = time.time()
        if not force and (now - self._last_leader_propagation) < LEADER_PROPAGATION_COOLDOWN:
            return False

        # Get current epoch from leadership state machine if available
        epoch = 0
        if hasattr(self, "_leadership_sm") and self._leadership_sm:
            epoch = getattr(self._leadership_sm, "epoch", 0)

        # Create the leader claim
        claim = LeaderClaim(
            leader_id=self.node_id,
            epoch=epoch,
            lease_expires=self.leader_lease_expires,
            timestamp=now,
            source_node=self.node_id,
        )

        # Store in our local gossip state for propagation
        # The gossip protocol will include this in outgoing messages
        if not hasattr(self, "_local_gossip_leader_claim"):
            self._local_gossip_leader_claim = None
        self._local_gossip_leader_claim = claim

        self._last_leader_propagation = now

        logger.info(
            f"[RelayLeaderPropagator] Broadcasting leader claim: "
            f"leader={claim.leader_id}, epoch={claim.epoch}, "
            f"lease_expires={claim.lease_expires:.0f}"
        )

        return True

    def _get_leader_claim_for_gossip(self) -> dict[str, Any] | None:
        """Get the leader claim to include in gossip state.

        Called by _build_local_gossip_state() to include leader claim
        in outgoing gossip messages.

        Returns:
            Leader claim dict or None if not the leader
        """
        claim = getattr(self, "_local_gossip_leader_claim", None)
        if claim is None:
            return None

        # Don't propagate expired claims
        if not claim.is_valid():
            return None

        return claim.to_dict()

    def _extract_leader_from_gossip(self) -> LeaderClaim | None:
        """Extract the best leader claim from all known gossip states.

        Scans all received gossip states to find the highest-epoch,
        freshest leader claim that hasn't expired.

        Returns:
            Best LeaderClaim or None if no valid claims found
        """
        best_claim: LeaderClaim | None = None

        # Check all gossip peer states
        gossip_states = getattr(self, "_gossip_peer_states", {})

        for node_id, state in gossip_states.items():
            # Check for dedicated cluster_leader entry
            leader_data = state.get(GOSSIP_LEADER_KEY)
            if leader_data:
                claim = LeaderClaim.from_dict(leader_data)
                if claim and claim.is_valid() and claim.is_fresher_than(best_claim):
                    best_claim = claim

            # Also check the standard leader_id field
            leader_id = state.get("leader_id")
            if leader_id:
                epoch = state.get("leadership_epoch", 0)
                lease_expires = state.get("leader_lease_expires", 0)
                timestamp = state.get("timestamp", 0)

                claim = LeaderClaim(
                    leader_id=leader_id,
                    epoch=epoch,
                    lease_expires=lease_expires,
                    timestamp=timestamp,
                    source_node=node_id,
                )

                if claim.is_valid() and claim.is_fresher_than(best_claim):
                    best_claim = claim

        return best_claim

    def _update_leader_from_gossip(self) -> bool:
        """Update local leader_id from gossip if a fresher claim exists.

        This is the key method for NAT-blocked nodes. It extracts the
        best leader claim from gossip and adopts it if:
        - We don't have a leader_id, OR
        - Our current leader's lease has expired, OR
        - The gossip claim has a higher epoch

        Returns:
            True if leader_id was updated
        """
        best_claim = self._extract_leader_from_gossip()
        if best_claim is None:
            return False

        # Determine if we should adopt this leader
        should_adopt = False
        reason = ""

        current_leader = self.leader_id
        current_lease = getattr(self, "leader_lease_expires", 0.0)
        current_epoch = 0
        if hasattr(self, "_leadership_sm") and self._leadership_sm:
            current_epoch = getattr(self._leadership_sm, "epoch", 0)

        now = time.time()

        if not current_leader:
            # No leader known - adopt the claim
            should_adopt = True
            reason = "no_leader"
        elif current_lease < now:
            # Current leader's lease expired - adopt if claim is valid
            should_adopt = True
            reason = "lease_expired"
        elif best_claim.epoch > current_epoch:
            # Higher epoch - new election completed, adopt
            should_adopt = True
            reason = f"higher_epoch ({best_claim.epoch} > {current_epoch})"
        elif best_claim.leader_id == current_leader:
            # Same leader, just update lease time
            if best_claim.lease_expires > current_lease:
                self.leader_lease_expires = best_claim.lease_expires
                # Not a leader change, just lease renewal
                return False

        if not should_adopt:
            return False

        # Update our leader_id
        old_leader = self.leader_id
        self.leader_id = best_claim.leader_id
        self.leader_lease_expires = best_claim.lease_expires

        # Update last_leader_seen if available
        if hasattr(self, "last_leader_seen"):
            self.last_leader_seen = now

        logger.info(
            f"[RelayLeaderPropagator] Adopted leader from gossip: "
            f"{old_leader} -> {best_claim.leader_id} "
            f"(reason={reason}, epoch={best_claim.epoch}, source={best_claim.source_node})"
        )

        # Emit event if available
        self._emit_leader_adopted_from_gossip(best_claim, reason)

        return True

    def _emit_leader_adopted_from_gossip(
        self, claim: LeaderClaim, reason: str
    ) -> None:
        """Emit event when leader is adopted from gossip.

        Args:
            claim: The LeaderClaim that was adopted
            reason: Why the leader was adopted
        """
        try:
            from app.distributed.data_events import DataEventType
            from app.coordination.event_router import emit_event

            emit_event(DataEventType.P2P_LEADER_CHANGED, {
                "old_leader": getattr(self, "_previous_leader_id", None),
                "new_leader": claim.leader_id,
                "epoch": claim.epoch,
                "source": "gossip_propagation",
                "reason": reason,
                "node_id": self.node_id,
                "timestamp": time.time(),
            })
        except ImportError:
            pass  # Event system not available
        except Exception as e:
            logger.debug(f"[RelayLeaderPropagator] Failed to emit event: {e}")

    def _process_gossip_leader_claim(self, claim_data: dict[str, Any]) -> bool:
        """Process a leader claim received in gossip.

        Called when receiving gossip to process the cluster_leader entry.

        Args:
            claim_data: The cluster_leader dict from gossip state

        Returns:
            True if leader was updated from this claim
        """
        claim = LeaderClaim.from_dict(claim_data)
        if claim is None or not claim.is_valid():
            return False

        # Store in gossip states for later extraction
        # This ensures the claim is available to _extract_leader_from_gossip
        if not hasattr(self, "_received_leader_claims"):
            self._received_leader_claims: list[LeaderClaim] = []

        # Keep only recent, valid claims (max 10)
        self._received_leader_claims = [
            c for c in self._received_leader_claims
            if c.is_valid()
        ][:9]
        self._received_leader_claims.append(claim)

        # Try to update our leader from this claim
        return self._update_leader_from_gossip()
