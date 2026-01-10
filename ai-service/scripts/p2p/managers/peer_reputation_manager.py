"""Peer Reputation Manager for P2P Orchestrator.

January 2026: Phase 4 P2P Orchestrator Deep Decomposition

This module extracts peer reputation tracking from p2p_orchestrator.py
for better modularity and testability.

Responsibilities:
- Track peer interaction success/failure rates
- Compute reputation scores for peer selection
- Generate reputation summaries for gossip propagation
- Reset hourly counters for recent activity tracking

Usage:
    from scripts.p2p.managers.peer_reputation_manager import (
        PeerReputationManager,
        PeerReputationConfig,
        get_peer_reputation_manager,
    )

    # Create manager
    manager = PeerReputationManager()

    # Record interactions
    manager.record_interaction("peer-1", success=True, interaction_type="sync")
    manager.record_interaction("peer-1", success=False, interaction_type="gossip")

    # Get reputation score
    score = manager.get_reputation_score("peer-1")

    # Get summary for gossip
    summary = manager.get_reputation_summary()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Module-level singleton
_peer_reputation_manager: PeerReputationManager | None = None
_manager_lock = threading.Lock()


@dataclass
class PeerReputationConfig:
    """Configuration for peer reputation tracking.

    Attributes:
        recent_window_seconds: Time window for "recent" counters (default 1 hour)
        recency_decay_hours: Hours over which recency bonus decays (default 24)
        recent_weight: Weight for recent success rate in score (default 70)
        historical_weight: Weight for historical success rate (default 20)
        recency_weight: Weight for recency bonus (default 10)
        reliable_threshold: Score threshold for "reliable" classification (default 70)
        unreliable_threshold: Score threshold for "unreliable" classification (default 30)
        top_reliable_count: Number of reliable peers to include in summary (default 5)
        bottom_unreliable_count: Number of unreliable peers to include (default 3)
    """

    recent_window_seconds: float = 3600.0
    recency_decay_hours: float = 24.0
    recent_weight: float = 70.0
    historical_weight: float = 20.0
    recency_weight: float = 10.0
    reliable_threshold: float = 70.0
    unreliable_threshold: float = 30.0
    top_reliable_count: int = 5
    bottom_unreliable_count: int = 3


@dataclass
class PeerReputation:
    """Reputation data for a single peer.

    Attributes:
        total_success: Total successful interactions
        total_failure: Total failed interactions
        recent_success: Successful interactions in recent window
        recent_failure: Failed interactions in recent window
        last_success: Timestamp of last successful interaction
        last_failure: Timestamp of last failed interaction
        last_reset: Timestamp of last recent counter reset
        by_type: Success/failure counts by interaction type
    """

    total_success: int = 0
    total_failure: int = 0
    recent_success: int = 0
    recent_failure: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    last_reset: float = field(default_factory=time.time)
    by_type: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_success": self.total_success,
            "total_failure": self.total_failure,
            "recent_success": self.recent_success,
            "recent_failure": self.recent_failure,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "last_reset": self.last_reset,
            "by_type": self.by_type.copy(),
        }


@dataclass
class ReputationSummary:
    """Summary of peer reputations for gossip propagation.

    Attributes:
        reliable_peers: List of high-reputation peers with scores
        unreliable_peers: List of low-reputation peers with scores
    """

    reliable_peers: list[dict[str, Any]] = field(default_factory=list)
    unreliable_peers: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reliable_peers": self.reliable_peers,
            "unreliable_peers": self.unreliable_peers,
        }


class PeerReputationManager:
    """Track peer reliability for better peer selection in P2P operations.

    Maintains success/failure rates for each peer across different interaction
    types (sync, gossip, heartbeat, command), and computes reputation scores
    that consider recent activity, historical performance, and recency.
    """

    def __init__(self, config: PeerReputationConfig | None = None):
        """Initialize peer reputation manager.

        Args:
            config: Reputation tracking configuration
        """
        self._config = config or PeerReputationConfig()
        self._reputations: dict[str, PeerReputation] = {}
        self._lock = threading.Lock()

    # =========================================================================
    # INTERACTION RECORDING
    # =========================================================================

    def record_interaction(
        self,
        peer_id: str,
        success: bool,
        interaction_type: str = "general",
    ) -> None:
        """Record a peer interaction for reputation tracking.

        Args:
            peer_id: ID of the peer
            success: Whether the interaction was successful
            interaction_type: Type of interaction (sync, gossip, heartbeat, command)
        """
        with self._lock:
            if peer_id not in self._reputations:
                self._reputations[peer_id] = PeerReputation()

            rep = self._reputations[peer_id]
            now = time.time()

            # Reset recent counters if window expired
            if now - rep.last_reset > self._config.recent_window_seconds:
                rep.recent_success = 0
                rep.recent_failure = 0
                rep.last_reset = now

            # Update counters
            if success:
                rep.total_success += 1
                rep.recent_success += 1
                rep.last_success = now
            else:
                rep.total_failure += 1
                rep.recent_failure += 1
                rep.last_failure = now

            # Track by type
            if interaction_type not in rep.by_type:
                rep.by_type[interaction_type] = {"success": 0, "failure": 0}
            if success:
                rep.by_type[interaction_type]["success"] += 1
            else:
                rep.by_type[interaction_type]["failure"] += 1

    def record_success(self, peer_id: str, interaction_type: str = "general") -> None:
        """Record a successful interaction.

        Args:
            peer_id: ID of the peer
            interaction_type: Type of interaction
        """
        self.record_interaction(peer_id, success=True, interaction_type=interaction_type)

    def record_failure(self, peer_id: str, interaction_type: str = "general") -> None:
        """Record a failed interaction.

        Args:
            peer_id: ID of the peer
            interaction_type: Type of interaction
        """
        self.record_interaction(peer_id, success=False, interaction_type=interaction_type)

    # =========================================================================
    # SCORE COMPUTATION
    # =========================================================================

    def get_reputation_score(self, peer_id: str) -> float:
        """Get reputation score for a peer (0-100, higher is better).

        Score combines:
        - Recent success rate (configurable weight, default 70%)
        - Historical success rate (configurable weight, default 20%)
        - Recency bonus (configurable weight, default 10%)

        Args:
            peer_id: ID of the peer

        Returns:
            Reputation score between 0 and 100
        """
        with self._lock:
            rep = self._reputations.get(peer_id)
            if not rep:
                return 50.0  # Default neutral score

            now = time.time()

            # Recent success rate (within recent window)
            recent_total = rep.recent_success + rep.recent_failure
            recent_rate = rep.recent_success / max(1, recent_total)

            # Historical success rate
            total = rep.total_success + rep.total_failure
            historical_rate = rep.total_success / max(1, total)

            # Recency bonus (active peers get a boost)
            last_interaction = max(rep.last_success, rep.last_failure)
            if last_interaction > 0:
                recency_hours = (now - last_interaction) / 3600
            else:
                recency_hours = self._config.recency_decay_hours  # No interactions

            recency_score = max(0.0, 1.0 - (recency_hours / self._config.recency_decay_hours))

            # Weighted score
            score = (
                (recent_rate * self._config.recent_weight)
                + (historical_rate * self._config.historical_weight)
                + (recency_score * self._config.recency_weight)
            )

            return min(100.0, max(0.0, score))

    def get_all_scores(self) -> dict[str, float]:
        """Get reputation scores for all tracked peers.

        Returns:
            Dict mapping peer_id to reputation score
        """
        with self._lock:
            # Use unlocked version to avoid deadlock (we already hold the lock)
            return {peer_id: self._compute_score_unlocked(peer_id) for peer_id in self._reputations}

    # =========================================================================
    # SUMMARY GENERATION
    # =========================================================================

    def get_reputation_summary(self) -> ReputationSummary:
        """Get summary of peer reputation for gossip propagation.

        Returns top reliable and bottom unreliable peers with their scores.
        Used to help cluster converge on reliable peer selection.

        Returns:
            ReputationSummary with reliable and unreliable peer lists
        """
        with self._lock:
            if not self._reputations:
                return ReputationSummary()

            scores = []
            for peer_id in self._reputations:
                # Note: get_reputation_score acquires lock, but we already hold it
                # So we compute inline
                score = self._compute_score_unlocked(peer_id)
                scores.append((peer_id, score))

            scores.sort(key=lambda x: x[1], reverse=True)

            reliable = [
                {"peer": p, "score": round(s)}
                for p, s in scores[: self._config.top_reliable_count]
                if s >= self._config.reliable_threshold
            ]

            unreliable = [
                {"peer": p, "score": round(s)}
                for p, s in scores[-self._config.bottom_unreliable_count :]
                if s < self._config.unreliable_threshold
            ]

            return ReputationSummary(
                reliable_peers=reliable,
                unreliable_peers=unreliable,
            )

    def _compute_score_unlocked(self, peer_id: str) -> float:
        """Compute reputation score without acquiring lock (caller must hold lock).

        Args:
            peer_id: ID of the peer

        Returns:
            Reputation score between 0 and 100
        """
        rep = self._reputations.get(peer_id)
        if not rep:
            return 50.0

        now = time.time()

        recent_total = rep.recent_success + rep.recent_failure
        recent_rate = rep.recent_success / max(1, recent_total)

        total = rep.total_success + rep.total_failure
        historical_rate = rep.total_success / max(1, total)

        last_interaction = max(rep.last_success, rep.last_failure)
        if last_interaction > 0:
            recency_hours = (now - last_interaction) / 3600
        else:
            recency_hours = self._config.recency_decay_hours

        recency_score = max(0.0, 1.0 - (recency_hours / self._config.recency_decay_hours))

        score = (
            (recent_rate * self._config.recent_weight)
            + (historical_rate * self._config.historical_weight)
            + (recency_score * self._config.recency_weight)
        )

        return min(100.0, max(0.0, score))

    # =========================================================================
    # PEER DATA ACCESS
    # =========================================================================

    def get_peer_data(self, peer_id: str) -> PeerReputation | None:
        """Get reputation data for a specific peer.

        Args:
            peer_id: ID of the peer

        Returns:
            PeerReputation data or None if peer not tracked
        """
        with self._lock:
            rep = self._reputations.get(peer_id)
            if rep:
                # Return a copy to avoid concurrent modification
                return PeerReputation(
                    total_success=rep.total_success,
                    total_failure=rep.total_failure,
                    recent_success=rep.recent_success,
                    recent_failure=rep.recent_failure,
                    last_success=rep.last_success,
                    last_failure=rep.last_failure,
                    last_reset=rep.last_reset,
                    by_type=rep.by_type.copy(),
                )
            return None

    def get_all_peers(self) -> list[str]:
        """Get list of all tracked peer IDs.

        Returns:
            List of peer IDs
        """
        with self._lock:
            return list(self._reputations.keys())

    def get_peer_count(self) -> int:
        """Get number of tracked peers.

        Returns:
            Number of peers with reputation data
        """
        with self._lock:
            return len(self._reputations)

    def remove_peer(self, peer_id: str) -> bool:
        """Remove reputation data for a peer.

        Args:
            peer_id: ID of the peer to remove

        Returns:
            True if peer was removed, False if not found
        """
        with self._lock:
            if peer_id in self._reputations:
                del self._reputations[peer_id]
                return True
            return False

    def clear_all(self) -> None:
        """Clear all reputation data."""
        with self._lock:
            self._reputations.clear()

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Return health status for daemon manager integration.

        Returns dict with:
        - status: "healthy" or "degraded"
        - message: Human-readable status message
        - details: Additional details for debugging
        """
        with self._lock:
            peer_count = len(self._reputations)
            scores = [self._compute_score_unlocked(p) for p in self._reputations]

            if not scores:
                return {
                    "status": "healthy",
                    "message": "PeerReputationManager operational (no peers tracked)",
                    "details": {"peer_count": 0},
                }

            avg_score = sum(scores) / len(scores)
            reliable_count = sum(1 for s in scores if s >= self._config.reliable_threshold)
            unreliable_count = sum(1 for s in scores if s < self._config.unreliable_threshold)

            status = "healthy"
            message = "PeerReputationManager operational"

            if unreliable_count > peer_count * 0.5:
                status = "degraded"
                message = f"High unreliable peer ratio ({unreliable_count}/{peer_count})"

            return {
                "status": status,
                "message": message,
                "details": {
                    "peer_count": peer_count,
                    "avg_score": round(avg_score, 1),
                    "reliable_count": reliable_count,
                    "unreliable_count": unreliable_count,
                },
            }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def get_peer_reputation_manager() -> PeerReputationManager | None:
    """Get the singleton PeerReputationManager instance.

    Returns:
        PeerReputationManager instance or None if not created
    """
    return _peer_reputation_manager


def set_peer_reputation_manager(manager: PeerReputationManager | None) -> None:
    """Set the singleton PeerReputationManager instance.

    Args:
        manager: PeerReputationManager instance or None to clear
    """
    global _peer_reputation_manager
    with _manager_lock:
        _peer_reputation_manager = manager


def create_peer_reputation_manager(
    config: PeerReputationConfig | None = None,
) -> PeerReputationManager:
    """Create and register a PeerReputationManager instance.

    Args:
        config: Reputation tracking configuration

    Returns:
        PeerReputationManager instance
    """
    manager = PeerReputationManager(config=config)
    set_peer_reputation_manager(manager)
    return manager
