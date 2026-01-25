"""Peer Cleanup Loop for P2P Orchestrator.

December 2025: Automatic stale peer cleanup for cluster health.

Problem: Stale peers accumulate over time in self.peers dict because:
1. Peers are added via gossip/heartbeats but never automatically removed
2. Partition detection uses total_peers vs alive_peers ratio
3. Stale peers inflate denominator, causing false "isolated" status

This caused the cluster to show 149 total peers with only 16 alive (10.7%
health ratio), triggering readonly mode despite having healthy nodes.

Solution: Automatically purge stale peers using a multi-tier cleanup strategy:
- Tier 1 (>1 hour): Mark as cleanup candidate (tracked in stats)
- Tier 2 (>6 hours): Auto-purge from in-memory peers dict
- Tier 3 (>24 hours + retired): Purge from all caches including SQLite

Usage:
    from scripts.p2p.loops import PeerCleanupLoop, PeerCleanupConfig

    cleanup_loop = PeerCleanupLoop(
        get_all_peers=lambda: orchestrator.peers,
        purge_peer=orchestrator._purge_single_peer,
        emit_event=orchestrator._emit_event,
    )
    await cleanup_loop.run_forever()

Events:
    STALE_PEERS_PURGED: Emitted after cleanup with purge counts per tier
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from .base import BaseLoop

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PeerCleanupConfig:
    """Configuration for automatic stale peer cleanup."""

    # Interval between cleanup cycles (seconds)
    cleanup_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_PEER_CLEANUP_INTERVAL", "300")
        )
    )

    # Tier 1: Mark as stale candidate (tracked in stats, not purged)
    # Jan 12, 2026: Reduced from 1 hour to 20 min for faster dead node cleanup
    tier1_stale_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_PEER_STALE_THRESHOLD", "1200")
        )
    )

    # Tier 2: Auto-purge from memory (definitively gone)
    # Jan 12, 2026: Reduced from 6 hours to 1 hour to prevent dead node accumulation
    # Jan 24, 2026: Reduced from 1 hour to 20 minutes for faster cleanup of stale peers
    tier2_purge_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_PEER_PURGE_THRESHOLD", "1200")
        )
    )

    # Tier 3: Purge retired peers from all caches (permanently retired)
    # Jan 12, 2026: Reduced from 24 hours to 4 hours for more aggressive cleanup
    tier3_cache_purge_seconds: float = 14400.0  # 4 hours

    # Maximum peers to purge per cycle (prevent thundering herd)
    max_purge_per_cycle: int = 20

    # Whether the loop is enabled
    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_PEER_CLEANUP_ENABLED", "1"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    # Dry run mode - log what would be purged without actually purging
    dry_run: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_PEER_CLEANUP_DRY_RUN", "0"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    # Whether to emit events on cleanup
    emit_events: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.cleanup_interval_seconds <= 0:
            raise ValueError("cleanup_interval_seconds must be > 0")
        if self.tier1_stale_seconds <= 0:
            raise ValueError("tier1_stale_seconds must be > 0")
        if self.tier2_purge_seconds <= self.tier1_stale_seconds:
            raise ValueError("tier2_purge_seconds must be > tier1_stale_seconds")
        if self.tier3_cache_purge_seconds <= self.tier2_purge_seconds:
            raise ValueError("tier3_cache_purge_seconds must be > tier2_purge_seconds")
        if self.max_purge_per_cycle <= 0:
            raise ValueError("max_purge_per_cycle must be > 0")


# =============================================================================
# Statistics
# =============================================================================


@dataclass
class CleanupStats:
    """Statistics for peer cleanup operations."""

    total_purged: int = 0
    tier1_detected: int = 0  # Candidates (>1h, not yet purged)
    tier2_purged: int = 0  # Purged from memory (>6h)
    tier3_purged: int = 0  # Purged retired from cache (>24h)
    last_cleanup_time: float = 0.0
    cycles_run: int = 0
    consecutive_errors: int = 0  # Required by LoopManager.get_status()
    successful_runs: int = 0  # Required by LoopManager.get_status()
    total_run_duration: float = 0.0  # Jan 7, 2026: Added for avg_run_duration calculation

    @property
    def total_runs(self) -> int:
        """Alias for cycles_run to match LoopStats interface.

        Required by LoopManager.start_all() which checks _stats.total_runs.
        """
        return self.cycles_run

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage.

        Required by LoopManager.get_status() at base.py:410.
        """
        if self.cycles_run == 0:
            return 100.0
        return (self.successful_runs / self.cycles_run) * 100.0

    @property
    def avg_run_duration(self) -> float:
        """Calculate average run duration in seconds.

        Required by base.py:483 for performance degradation checks.
        Jan 7, 2026: Added to fix AttributeError on _check_performance_degradation.
        """
        if self.successful_runs == 0:
            return 0.0
        return self.total_run_duration / self.successful_runs

    def to_dict(self) -> dict:
        """Convert stats to dictionary for JSON serialization."""
        return {
            "total_purged": self.total_purged,
            "tier1_detected": self.tier1_detected,
            "tier2_purged": self.tier2_purged,
            "tier3_purged": self.tier3_purged,
            "last_cleanup_time": self.last_cleanup_time,
            "cycles_run": self.cycles_run,
            "total_runs": self.cycles_run,  # Include for consistency
        }


# =============================================================================
# Cleanup Loop
# =============================================================================


class PeerCleanupLoop(BaseLoop):
    """Background loop that automatically purges stale peers.

    Key features:
    - Multi-tier staleness detection (1h/6h/24h thresholds)
    - Limited purges per cycle to avoid thundering herd
    - Emits STALE_PEERS_PURGED event with cleanup statistics
    - Dry run mode for testing cleanup logic
    - Improves partition detection accuracy by keeping peer counts accurate
    """

    def __init__(
        self,
        get_all_peers: Callable[[], dict[str, Any]],
        purge_peer: Callable[[str], Coroutine[Any, Any, bool]] | Callable[[str], bool],
        get_peer_cache_manager: Callable[[], Any] | None = None,
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        config: PeerCleanupConfig | None = None,
    ):
        """Initialize peer cleanup loop.

        Args:
            get_all_peers: Callback returning dict of node_id -> peer_info
            purge_peer: Callback to purge a peer by node_id. Returns True if successful.
                       Can be sync or async.
            get_peer_cache_manager: Optional callback to get PeerManager for cache cleanup
            emit_event: Optional callback to emit events (event_name, event_data)
            config: Cleanup configuration
        """
        self.config = config or PeerCleanupConfig()
        super().__init__(
            name="peer_cleanup",
            interval=self.config.cleanup_interval_seconds,
            enabled=self.config.enabled,
        )

        # Callbacks
        self._get_all_peers = get_all_peers
        self._purge_peer = purge_peer
        self._get_peer_cache_manager = get_peer_cache_manager
        self._emit_event = emit_event

        # Statistics
        self._cleanup_stats = CleanupStats()

    async def _run_once(self) -> None:
        """Execute one cleanup cycle."""
        if not self.config.enabled:
            return

        now = time.time()
        peers = self._get_all_peers()

        if not peers:
            logger.debug("[PeerCleanup] No peers to evaluate")
            return

        # Calculate health ratio before cleanup
        total_before = len(peers)
        alive_before = sum(
            1 for p in peers.values() if self._is_peer_alive(p)
        )
        health_before = alive_before / total_before if total_before > 0 else 1.0

        # Classify peers by staleness tier
        tier1_candidates: list[str] = []  # Stale but not purged yet
        tier2_to_purge: list[str] = []  # Ready for memory purge
        tier3_to_purge: list[str] = []  # Ready for cache purge (retired)

        for node_id, peer in peers.items():
            tier = self._classify_stale_peer(peer, now)
            if tier == 1:
                tier1_candidates.append(node_id)
            elif tier == 2:
                tier2_to_purge.append(node_id)
            elif tier == 3:
                tier3_to_purge.append(node_id)

        # Update tier1 detection stats (not purged, just tracked)
        self._cleanup_stats.tier1_detected = len(tier1_candidates)

        # Combine tier2 and tier3 for purging
        all_to_purge = tier2_to_purge + tier3_to_purge

        if not all_to_purge:
            logger.debug(
                f"[PeerCleanup] No peers to purge (tier1 candidates: {len(tier1_candidates)})"
            )
            return

        # Limit purges per cycle
        purge_this_cycle = all_to_purge[: self.config.max_purge_per_cycle]
        purge_set = set(purge_this_cycle)

        # Count by tier for this cycle
        tier2_count = len([n for n in tier2_to_purge if n in purge_set])
        tier3_count = len([n for n in tier3_to_purge if n in purge_set])

        if self.config.dry_run:
            logger.info(
                f"[PeerCleanup] DRY RUN: Would purge {len(purge_this_cycle)} peers "
                f"(tier2: {tier2_count}, tier3: {tier3_count})"
            )
            for node_id in purge_this_cycle:
                logger.info(f"[PeerCleanup] DRY RUN: Would purge {node_id}")
            return

        # Execute purges
        purged_count = 0
        for node_id in purge_this_cycle:
            try:
                result = self._purge_peer(node_id)
                # Handle both sync and async callbacks
                if hasattr(result, "__await__"):
                    result = await result
                if result:
                    purged_count += 1
                    logger.debug(f"[PeerCleanup] Purged stale peer: {node_id}")
                else:
                    logger.warning(f"[PeerCleanup] Failed to purge peer: {node_id}")
            except Exception as e:
                logger.warning(f"[PeerCleanup] Error purging {node_id}: {e}")

        # Update statistics
        self._cleanup_stats.total_purged += purged_count
        self._cleanup_stats.tier2_purged += min(tier2_count, purged_count)
        remaining_for_tier3 = purged_count - tier2_count
        if remaining_for_tier3 > 0:
            self._cleanup_stats.tier3_purged += min(tier3_count, remaining_for_tier3)
        self._cleanup_stats.last_cleanup_time = now
        self._cleanup_stats.cycles_run += 1

        # Calculate health ratio after cleanup
        peers_after = self._get_all_peers()
        total_after = len(peers_after)
        alive_after = sum(
            1 for p in peers_after.values() if self._is_peer_alive(p)
        )
        health_after = alive_after / total_after if total_after > 0 else 1.0

        logger.info(
            f"[PeerCleanup] Purged {purged_count} stale peers "
            f"(tier2: {tier2_count}, tier3: {tier3_count}). "
            f"Health ratio: {health_before:.1%} → {health_after:.1%}"
        )

        # Emit event
        if self._emit_event and self.config.emit_events and purged_count > 0:
            self._emit_event(
                "STALE_PEERS_PURGED",
                {
                    "purged_count": purged_count,
                    "by_tier": {
                        "tier1_candidates": len(tier1_candidates),
                        "tier2": tier2_count,
                        "tier3": tier3_count,
                    },
                    "remaining_total": total_after,
                    "remaining_alive": alive_after,
                    "health_ratio_before": health_before,
                    "health_ratio_after": health_after,
                    "timestamp": now,
                },
            )

    def _classify_stale_peer(self, peer: Any, now: float) -> int | None:
        """Classify peer into staleness tier.

        Returns:
            1: Tier 1 - stale candidate (>tier1_stale_seconds, tracked only)
            2: Tier 2 - ready for memory purge (>tier2_purge_seconds)
            3: Tier 3 - ready for cache purge (retired + >tier3_cache_purge_seconds)
            None: Not stale
        """
        last_heartbeat = self._get_peer_heartbeat(peer)
        if last_heartbeat <= 0:
            # No valid heartbeat - consider very stale
            return 2

        age = now - last_heartbeat
        is_retired = self._is_peer_retired(peer)

        # Tier 3: Retired + very old
        if is_retired and age > self.config.tier3_cache_purge_seconds:
            return 3

        # Tier 2: Very stale (definitively gone)
        if age > self.config.tier2_purge_seconds:
            return 2

        # Tier 1: Stale candidate (might recover)
        if age > self.config.tier1_stale_seconds:
            return 1

        return None

    def _get_peer_heartbeat(self, peer: Any) -> float:
        """Extract last_heartbeat from peer object."""
        if hasattr(peer, "last_heartbeat"):
            return peer.last_heartbeat
        if isinstance(peer, dict):
            return peer.get("last_heartbeat", 0.0)
        return 0.0

    def _is_peer_retired(self, peer: Any) -> bool:
        """Check if peer is marked as retired."""
        if hasattr(peer, "retired"):
            return bool(peer.retired)
        if isinstance(peer, dict):
            return bool(peer.get("retired", False))
        return False

    def _is_peer_alive(self, peer: Any) -> bool:
        """Check if peer is considered alive."""
        if hasattr(peer, "is_alive"):
            return bool(peer.is_alive)
        if isinstance(peer, dict) and "is_alive" in peer:
            return bool(peer["is_alive"])
        # Fallback: check last_heartbeat recency
        last_heartbeat = self._get_peer_heartbeat(peer)
        if last_heartbeat > 0:
            return (time.time() - last_heartbeat) < 60  # 60s threshold
        return False

    def get_cleanup_stats(self) -> dict[str, Any]:
        """Get cleanup statistics."""
        return {
            "total_purged": self._cleanup_stats.total_purged,
            "tier1_detected": self._cleanup_stats.tier1_detected,
            "tier2_purged": self._cleanup_stats.tier2_purged,
            "tier3_purged": self._cleanup_stats.tier3_purged,
            "last_cleanup_time": self._cleanup_stats.last_cleanup_time,
            "cycles_run": self._cleanup_stats.cycles_run,
            "config": {
                "interval_seconds": self.config.cleanup_interval_seconds,
                "tier1_threshold": self.config.tier1_stale_seconds,
                "tier2_threshold": self.config.tier2_purge_seconds,
                "tier3_threshold": self.config.tier3_cache_purge_seconds,
                "max_per_cycle": self.config.max_purge_per_cycle,
                "enabled": self.config.enabled,
                "dry_run": self.config.dry_run,
            },
        }

    def reset_stats(self) -> None:
        """Reset cleanup statistics."""
        self._cleanup_stats = CleanupStats()
        logger.info("[PeerCleanup] Statistics reset")
