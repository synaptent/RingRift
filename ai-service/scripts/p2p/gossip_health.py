"""Gossip Health Tracking - Self-contained health classes for gossip protocol.

Extracted from gossip_protocol.py for modularity.

This module provides:
- GossipHealthSummary: Thread-safe snapshot of gossip health data
- GossipHealthTracker: Per-peer failure tracking with exponential backoff
- GOSSIP_FAILURE_SUSPECT_THRESHOLD: Configurable failure threshold

February 2026: Extracted as part of gossip_protocol.py decomposition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Dec 28, 2025: Phase 6 - Gossip health tracking threshold
# Dec 30, 2025: Now uses centralized GossipDefaults for configurability
try:
    from app.config.coordination_defaults import GossipDefaults
    GOSSIP_FAILURE_SUSPECT_THRESHOLD = GossipDefaults.FAILURE_THRESHOLD
except ImportError:
    GOSSIP_FAILURE_SUSPECT_THRESHOLD = 5  # Fallback


@dataclass
class GossipHealthSummary:
    """Thread-safe health summary from GossipHealthTracker.

    Jan 3, 2026 Sprint 13: Provides a public API for HealthCoordinator
    to access gossip health data without coupling to private attributes.
    All data is copied at creation time for thread safety.
    """

    failure_counts: dict[str, int] = field(default_factory=dict)
    """Copy of per-peer failure counts."""

    last_success: dict[str, float] = field(default_factory=dict)
    """Copy of per-peer last success timestamps."""

    suspected_peers: list[str] = field(default_factory=list)
    """List of peers currently suspected (high failure count)."""

    stale_peers: list[str] = field(default_factory=list)
    """List of peers with no gossip success in >5 minutes."""

    total_tracked_peers: int = 0
    """Total number of peers being tracked."""

    failure_threshold: int = 5
    """Threshold for marking a peer as suspect."""

    @property
    def suspected_count(self) -> int:
        """Number of suspected peers."""
        return len(self.suspected_peers)

    @property
    def stale_count(self) -> int:
        """Number of stale peers."""
        return len(self.stale_peers)

    @property
    def health_score(self) -> float:
        """Calculate health score from 0.0 (unhealthy) to 1.0 (healthy).

        Score is based on:
        - 50% weight: Ratio of healthy peers (not suspected or stale)
        - 50% weight: Average failure rate across peers
        """
        if self.total_tracked_peers == 0:
            return 1.0  # No peers tracked = healthy by default

        # Healthy peer ratio (not suspected or stale)
        unhealthy_peers = set(self.suspected_peers) | set(self.stale_peers)
        healthy_ratio = 1.0 - (len(unhealthy_peers) / self.total_tracked_peers)

        # Average failure rate (capped at threshold)
        if self.failure_counts:
            avg_failures = sum(self.failure_counts.values()) / len(self.failure_counts)
            failure_ratio = 1.0 - min(avg_failures / self.failure_threshold, 1.0)
        else:
            failure_ratio = 1.0

        return 0.5 * healthy_ratio + 0.5 * failure_ratio


class GossipHealthTracker:
    """Tracks gossip protocol health per peer.

    December 2025: Part of Phase 6 cluster availability improvements.
    Tracks gossip failures per peer and emits NODE_SUSPECT when a peer
    has too many consecutive gossip failures.

    Jan 3, 2026: Added exponential backoff for failed peers.
    When a peer fails gossip, we wait progressively longer before retrying:
    - 1st failure: 1s backoff
    - 2nd failure: 2s backoff
    - 3rd failure: 4s backoff
    - 4th failure: 8s backoff
    - 5th+ failure: 16s backoff (max)

    Jan 3, 2026 Sprint 13: Added thread safety via RLock.
    All shared state (_failure_counts, _last_success, _last_failure, _suspect_emitted)
    is now protected by _state_lock to prevent data corruption from concurrent
    gossip handlers.

    Features:
    - Per-peer failure counting
    - Automatic reset on success
    - Threshold-based suspect detection
    - Last success timestamp tracking for staleness
    - Exponential backoff for failed peers (Jan 3, 2026)
    - Thread-safe access via RLock (Jan 3, 2026 Sprint 13)
    """

    # Jan 3, 2026: Backoff configuration
    # Jan 5, 2026 Session 17.31: Reduced max backoff from 16s to 8s for faster recovery
    BACKOFF_BASE_SECONDS: float = 1.0
    BACKOFF_MULTIPLIER: float = 2.0
    BACKOFF_MAX_SECONDS: float = 8.0  # Was 16.0 - reduced for faster node recovery
    # Jan 5, 2026 (Phase 6): Jitter factor to prevent thundering herd
    # Applied as multiplier: backoff * random(1-jitter, 1+jitter)
    BACKOFF_JITTER_FACTOR: float = 0.25  # +/- 25% jitter
    # January 2026: Grace period for new peers before marking them stale
    # New peers are not marked stale until grace period expires
    NEW_PEER_GRACE_PERIOD_SECONDS: float = 60.0

    def __init__(self, failure_threshold: int = GOSSIP_FAILURE_SUSPECT_THRESHOLD):
        """Initialize gossip health tracker.

        Args:
            failure_threshold: Number of consecutive failures before emitting suspect
        """
        import threading
        self._state_lock = threading.RLock()  # Sprint 13: Thread safety for shared state
        self._failure_counts: dict[str, int] = {}
        self._last_success: dict[str, float] = {}
        self._last_failure: dict[str, float] = {}  # Jan 3, 2026: Track last failure time
        self._peer_first_seen: dict[str, float] = {}  # January 2026: Track when peer was first seen
        self._failure_threshold = failure_threshold
        self._suspect_emitted: set[str] = set()  # Track which peers have been marked suspect

    def record_gossip_failure(self, peer_id: str) -> tuple[bool, int]:
        """Record a gossip failure for a peer.

        Thread-safe: Uses _state_lock to protect shared state.

        Args:
            peer_id: The peer that failed to respond to gossip

        Returns:
            Tuple of (should_emit_suspect, failure_count)
            - should_emit_suspect: True if this failure crosses the threshold
            - failure_count: Current consecutive failure count

        January 2026: Also tracks first-seen time for new peers to enable
        grace period before marking them as stale.
        """
        with self._state_lock:
            now = time.time()
            # January 2026: Track first-seen time for new peers (grace period support)
            if peer_id not in self._peer_first_seen:
                self._peer_first_seen[peer_id] = now
            self._failure_counts[peer_id] = self._failure_counts.get(peer_id, 0) + 1
            self._last_failure[peer_id] = now  # Jan 3, 2026: Track failure time
            count = self._failure_counts[peer_id]

            # Check if we should emit suspect (only once per failure streak)
            should_emit = (
                count >= self._failure_threshold
                and peer_id not in self._suspect_emitted
            )

            if should_emit:
                self._suspect_emitted.add(peer_id)

            return should_emit, count

    def get_backoff_seconds(self, peer_id: str) -> float:
        """Calculate exponential backoff delay for a peer with jitter.

        Thread-safe: Uses _state_lock to protect shared state.

        Jan 3, 2026: Returns the number of seconds to wait before retrying
        gossip to this peer based on consecutive failure count.

        Jan 5, 2026 (Phase 6): Added jitter to prevent thundering herd when
        multiple nodes back off identically after partition healing.

        Args:
            peer_id: The peer to calculate backoff for

        Returns:
            Backoff delay in seconds with jitter (0 if no failures)
        """
        import random

        with self._state_lock:
            failure_count = self._failure_counts.get(peer_id, 0)
        if failure_count == 0:
            return 0.0

        # Exponential backoff: base * multiplier^(failures-1), capped at max
        base_backoff = self.BACKOFF_BASE_SECONDS * (
            self.BACKOFF_MULTIPLIER ** (failure_count - 1)
        )
        base_backoff = min(base_backoff, self.BACKOFF_MAX_SECONDS)

        # Apply jitter: multiply by random value in range [1-jitter, 1+jitter]
        # This prevents thundering herd when multiple nodes back off identically
        jitter_multiplier = random.uniform(
            1 - self.BACKOFF_JITTER_FACTOR,
            1 + self.BACKOFF_JITTER_FACTOR,
        )
        return base_backoff * jitter_multiplier

    def should_skip_peer(self, peer_id: str) -> bool:
        """Check if a peer should be skipped due to backoff.

        Thread-safe: Uses _state_lock to protect shared state.

        Jan 3, 2026: Returns True if the peer is in backoff period.

        Args:
            peer_id: The peer to check

        Returns:
            True if peer should be skipped, False if OK to gossip
        """
        with self._state_lock:
            last_failure = self._last_failure.get(peer_id)
        if last_failure is None:
            return False  # No failures, OK to gossip

        backoff = self.get_backoff_seconds(peer_id)
        if backoff == 0:
            return False  # No backoff needed

        elapsed = time.time() - last_failure
        return elapsed < backoff

    def record_gossip_success(self, peer_id: str) -> bool:
        """Record a successful gossip exchange with a peer.

        Thread-safe: Uses _state_lock to protect shared state.

        Args:
            peer_id: The peer that responded successfully

        Returns:
            True if peer was previously suspected (recovered), False otherwise

        January 2026: Also tracks first-seen time for new peers.
        """
        with self._state_lock:
            now = time.time()
            was_suspected = peer_id in self._suspect_emitted

            # January 2026: Track first-seen time for new peers (grace period support)
            if peer_id not in self._peer_first_seen:
                self._peer_first_seen[peer_id] = now

            # Reset failure count and remove from suspect set
            self._failure_counts[peer_id] = 0
            self._last_success[peer_id] = now
            self._suspect_emitted.discard(peer_id)

            return was_suspected

    def get_failure_count(self, peer_id: str) -> int:
        """Get the current consecutive failure count for a peer.

        Thread-safe: Uses _state_lock to protect shared state.
        """
        with self._state_lock:
            return self._failure_counts.get(peer_id, 0)

    def get_last_success(self, peer_id: str) -> float | None:
        """Get the timestamp of the last successful gossip with a peer.

        Thread-safe: Uses _state_lock to protect shared state.
        """
        with self._state_lock:
            return self._last_success.get(peer_id)

    def is_suspected(self, peer_id: str) -> bool:
        """Check if a peer is currently marked as suspect due to gossip failures.

        Thread-safe: Uses _state_lock to protect shared state.
        """
        with self._state_lock:
            return peer_id in self._suspect_emitted

    def get_suspected_peers(self) -> set[str]:
        """Get the set of peers currently marked as suspect.

        Thread-safe: Uses _state_lock to protect shared state.
        """
        with self._state_lock:
            return self._suspect_emitted.copy()

    def get_stats(self) -> dict[str, int | float | list[str]]:
        """Get tracker statistics for monitoring.

        Thread-safe: Uses _state_lock to protect shared state.
        """
        with self._state_lock:
            now = time.time()
            stale_peers = [
                peer_id for peer_id, last_seen in self._last_success.items()
                if now - last_seen > 300  # 5 minutes
            ]
            return {
                "total_tracked_peers": len(self._failure_counts),
                "suspected_peers": len(self._suspect_emitted),
                "suspected_peer_ids": list(self._suspect_emitted),
                "stale_peers": len(stale_peers),
                "failure_threshold": self._failure_threshold,
            }

    def cleanup_stale_peers(self, max_age_seconds: float = 1800.0) -> int:
        """Remove stale peer tracking data.

        Thread-safe: Uses _state_lock to protect shared state.

        January 2026: Respects grace period - new peers won't be cleaned up
        until their grace period expires.
        Jan 25, 2026: Reduced from 3600 to 1800 for cleaner startup.

        Args:
            max_age_seconds: Remove peers not seen in this many seconds

        Returns:
            Number of peers cleaned up
        """
        with self._state_lock:
            now = time.time()
            cutoff = now - max_age_seconds

            # Find stale peers (excluding those still within grace period)
            stale_peers = [
                peer_id for peer_id, last_seen in self._last_success.items()
                if last_seen < cutoff
                # January 2026: Exclude peers still within grace period
                and now - self._peer_first_seen.get(peer_id, 0) > self.NEW_PEER_GRACE_PERIOD_SECONDS
            ]

            # Also include peers with high failure counts but no success
            for peer_id in list(self._failure_counts.keys()):
                if peer_id not in self._last_success:
                    # January 2026: Skip peers still within grace period
                    first_seen = self._peer_first_seen.get(peer_id, 0)
                    if now - first_seen <= self.NEW_PEER_GRACE_PERIOD_SECONDS:
                        continue
                    # Peer has never succeeded, check if it's been tracked too long
                    # We use a simple heuristic: high failure count = stale
                    if self._failure_counts[peer_id] > self._failure_threshold * 10:
                        if peer_id not in stale_peers:
                            stale_peers.append(peer_id)

            # Clean up
            for peer_id in stale_peers:
                self._failure_counts.pop(peer_id, None)
                self._last_success.pop(peer_id, None)
                self._last_failure.pop(peer_id, None)  # Jan 3, 2026: Clean up backoff tracking
                self._peer_first_seen.pop(peer_id, None)  # January 2026: Clean up first-seen tracking
                self._suspect_emitted.discard(peer_id)

            return len(stale_peers)

    def get_health_summary(self) -> GossipHealthSummary:
        """Get a thread-safe health summary for HealthCoordinator.

        Jan 3, 2026 Sprint 13: Public API to decouple HealthCoordinator
        from private attributes. All data is copied under lock.

        January 2026: Respects grace period for new peers - they are not
        marked as stale until grace period expires.

        Returns:
            GossipHealthSummary with copied state data.
        """
        with self._state_lock:
            now = time.time()
            stale_peers = [
                peer_id for peer_id, last_seen in self._last_success.items()
                if now - last_seen > 300  # 5 minutes
                # January 2026: Exclude peers still within grace period
                and now - self._peer_first_seen.get(peer_id, 0) > self.NEW_PEER_GRACE_PERIOD_SECONDS
            ]
            return GossipHealthSummary(
                failure_counts=dict(self._failure_counts),
                last_success=dict(self._last_success),
                suspected_peers=list(self._suspect_emitted),
                stale_peers=stale_peers,
                total_tracked_peers=len(self._failure_counts),
                failure_threshold=self._failure_threshold,
            )
