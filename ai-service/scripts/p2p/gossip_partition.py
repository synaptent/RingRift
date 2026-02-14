"""Gossip Partition Detection & Adaptive Intervals Mixin.

Extracted from gossip_protocol.py for modularity.

This mixin provides:
- Network partition detection (healthy/minority/isolated)
- Adaptive gossip intervals based on cluster health
- Interval jitter for thundering herd prevention

The mixin expects the implementing class to have:
- peers: dict[str, NodeInfo]
- peers_lock: threading.RLock
- node_id: str
- _peer_snapshot: PeerSnapshot (optional, for lock-free reads)
- get_gossip_suspected_peers() -> set[str]
- detect_partition_status related class constants

December 2025 (Phase 2.3): Active partition detection.
January 2026 (Sprint 13): Adaptive gossip intervals.
February 2026: Extracted as part of gossip_protocol.py decomposition.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class GossipPartitionMixin:
    """Mixin providing partition detection and adaptive gossip intervals.

    Expects the implementing class to provide:
    - peers: dict of peer NodeInfo objects
    - peers_lock: threading.RLock for peers dict
    - node_id: str
    - _peer_snapshot: optional PeerSnapshot for lock-free reads
    - get_gossip_suspected_peers(): set[str]
    - GOSSIP_INTERVAL_PARTITION/RECOVERY/STABLE: float constants
    - GOSSIP_STABILITY_THRESHOLD: int
    - GOSSIP_INTERVAL_SECONDS: float
    - GOSSIP_INTERVAL_JITTER: float
    - _log_info/debug/warning() from P2PMixinBase
    - _safe_emit_event() from P2PMixinBase
    """

    # =========================================================================
    # Partition Detection (Phase 2.3 - Dec 29, 2025)
    # =========================================================================

    def detect_partition_status(self) -> tuple[str, float]:
        """Detect if we're in a network partition.

        December 2025 (Phase 2.3): Active partition detection for cluster stability.
        December 30, 2025: Fixed to exclude long-dead and retired peers from denominator.

        The health ratio now only considers "relevant" peers:
        - Alive peers (heartbeat within PEER_TIMEOUT)
        - Recently dead peers (dead < 5 minutes) - may recover
        - Excludes: retired peers, long-dead peers (>5 min without heartbeat)

        This prevents false "minority" partition detection when the peers dict
        contains stale entries from nodes that were spun down or never connected.

        Partition detection enables the cluster to:
        - Pause writes when in minority partition (prevent split-brain divergence)
        - Alert operators about network issues
        - Automatically resume when connectivity restores

        Returns:
            Tuple of (status, health_ratio) where:
            - status: 'healthy' (>50% peers alive), 'minority' (20-50%), 'isolated' (<20%)
            - health_ratio: Float between 0.0 and 1.0 representing peer connectivity
        """
        import time

        # Long-dead threshold: peers dead for >5 minutes are excluded from calculation
        LONG_DEAD_THRESHOLD = 300  # 5 minutes

        # Feb 2026: Use lock-free PeerSnapshot for read-only access to avoid
        # lock contention that was causing /status endpoint timeouts.
        peer_snapshot = getattr(self, "_peer_snapshot", None)
        if peer_snapshot:
            peers_dict = peer_snapshot.get_snapshot()
        else:
            # Fallback to lock if snapshot not available (shouldn't happen)
            with self.peers_lock:
                peers_dict = dict(self.peers)

        if len(peers_dict) == 0:
            # No peers known at all - we're isolated
            return ("isolated", 0.0)

        now = time.time()
        alive_peers = 0
        relevant_peers = 0

        for p in peers_dict.values():
            # Skip retired peers entirely - they're intentionally offline
            if getattr(p, 'retired', False):
                continue

            time_since_heartbeat = now - p.last_heartbeat

            if p.is_alive():
                alive_peers += 1
                relevant_peers += 1
            elif time_since_heartbeat < LONG_DEAD_THRESHOLD:
                # Recently dead - still count in denominator (may recover)
                relevant_peers += 1
            # else: long-dead peer, exclude from both counts

        if relevant_peers == 0:
            # All peers are long-dead or retired - we're isolated
            return ("isolated", 0.0)

        health_ratio = alive_peers / relevant_peers

        if health_ratio > 0.5:
            return ("healthy", health_ratio)
        elif health_ratio > 0.2:
            return ("minority", health_ratio)
        else:
            return ("isolated", health_ratio)

    def get_partition_details(self) -> dict[str, Any]:
        """Get detailed partition status information.

        December 2025 (Phase 2.3): Extended partition info for monitoring.
        Feb 2026: Use lock-free PeerSnapshot to avoid endpoint timeouts.

        Returns:
            Dict with partition status, counts, and peer details.
        """
        status, ratio = self.detect_partition_status()

        # Use lock-free PeerSnapshot for read-only access
        peer_snapshot = getattr(self, "_peer_snapshot", None)
        if peer_snapshot:
            peers_dict = peer_snapshot.get_snapshot()
        else:
            with self.peers_lock:
                peers_dict = dict(self.peers)

        total_peers = len(peers_dict)
        alive_peers = [p.node_id for p in peers_dict.values() if p.is_alive()]
        dead_peers = [p.node_id for p in peers_dict.values() if not p.is_alive()]
        suspected_peers = list(self.get_gossip_suspected_peers())

        return {
            "status": status,
            "health_ratio": round(ratio, 3),
            "total_peers": total_peers,
            "alive_count": len(alive_peers),
            "dead_count": len(dead_peers),
            "suspected_count": len(suspected_peers),
            "alive_peers": alive_peers[:10],  # Limit for response size
            "dead_peers": dead_peers[:10],
            "suspected_peers": suspected_peers[:10],
        }

    # =========================================================================
    # Adaptive Gossip Intervals (Sprint 13 - Jan 3, 2026)
    # =========================================================================

    def _update_adaptive_gossip_interval(self) -> float:
        """Update and return the adaptive gossip interval based on partition status.

        Sprint 13 (Jan 3, 2026): Adaptive gossip intervals for faster partition recovery.

        The interval is adjusted based on cluster health:
        - Partition (isolated/minority): 5s - Fast gossip for rapid recovery
        - Recovery (healthy but recently partitioned): 10s - Medium interval
        - Stable (consistently healthy): 30s - Normal interval to reduce overhead

        Stability is determined by consecutive healthy checks exceeding the threshold.
        This prevents oscillation between intervals when health briefly fluctuates.

        Returns:
            The current adaptive gossip interval in seconds.
        """
        status, _ = self.detect_partition_status()

        # Track status changes for logging
        if hasattr(self, "_gossip_last_partition_status"):
            last_status = self._gossip_last_partition_status
        else:
            last_status = "unknown"
            self._gossip_last_partition_status = "unknown"
            self._gossip_consecutive_healthy = 0

        old_interval = getattr(self, "_gossip_adaptive_interval", self.GOSSIP_INTERVAL_SECONDS)

        if status in ("isolated", "minority"):
            # Partition detected - use fast interval
            self._gossip_consecutive_healthy = 0
            self._gossip_adaptive_interval = self.GOSSIP_INTERVAL_PARTITION
            if last_status == "healthy":
                self._log_warning(
                    f"Partition detected ({status}), reducing gossip interval: "
                    f"{old_interval:.1f}s -> {self._gossip_adaptive_interval:.1f}s"
                )
        elif status == "healthy":
            # Healthy - increment stability counter
            self._gossip_consecutive_healthy += 1

            if self._gossip_consecutive_healthy >= self.GOSSIP_STABILITY_THRESHOLD:
                # Stable - use normal interval
                self._gossip_adaptive_interval = self.GOSSIP_INTERVAL_STABLE
                if old_interval != self.GOSSIP_INTERVAL_STABLE:
                    self._log_info(
                        f"Cluster stable ({self._gossip_consecutive_healthy} healthy checks), "
                        f"increasing gossip interval: {old_interval:.1f}s -> {self._gossip_adaptive_interval:.1f}s"
                    )
            else:
                # Recovering - use medium interval
                self._gossip_adaptive_interval = self.GOSSIP_INTERVAL_RECOVERY
                if last_status != "healthy" and old_interval != self.GOSSIP_INTERVAL_RECOVERY:
                    self._log_info(
                        f"Cluster recovering, using medium gossip interval: "
                        f"{old_interval:.1f}s -> {self._gossip_adaptive_interval:.1f}s"
                    )

        self._gossip_last_partition_status = status
        return self._gossip_adaptive_interval

    def get_adaptive_gossip_interval(self) -> float:
        """Get the current adaptive gossip interval without updating state.

        Returns:
            The current adaptive gossip interval in seconds.
        """
        return getattr(self, "_gossip_adaptive_interval", self.GOSSIP_INTERVAL_SECONDS)

    def _get_gossip_interval_with_jitter(self) -> float:
        """Get the adaptive gossip interval with random jitter applied.

        Jan 5, 2026 (Phase 10.2): Added jitter to prevent thundering herd when
        multiple nodes gossip at the same time. The jitter is 0-50% of the
        base interval, meaning the effective interval is 1.0x to 1.5x the base.

        Returns:
            Gossip interval in seconds with jitter applied.
        """
        import random

        base_interval = self.get_adaptive_gossip_interval()
        # Apply 0-JITTER factor (multiplicative, adds delay only)
        jitter_multiplier = random.uniform(1.0, 1.0 + self.GOSSIP_INTERVAL_JITTER)
        return base_interval * jitter_multiplier

    def get_adaptive_gossip_status(self) -> dict[str, Any]:
        """Get detailed adaptive gossip interval status for monitoring.

        Returns:
            Dict with current interval, partition status, and stability info.
        """
        status, ratio = self.detect_partition_status()
        return {
            "current_interval": getattr(self, "_gossip_adaptive_interval", self.GOSSIP_INTERVAL_SECONDS),
            "partition_status": status,
            "health_ratio": round(ratio, 3),
            "consecutive_healthy": getattr(self, "_gossip_consecutive_healthy", 0),
            "stability_threshold": self.GOSSIP_STABILITY_THRESHOLD,
            "is_stable": getattr(self, "_gossip_consecutive_healthy", 0) >= self.GOSSIP_STABILITY_THRESHOLD,
            "interval_partition": self.GOSSIP_INTERVAL_PARTITION,
            "interval_recovery": self.GOSSIP_INTERVAL_RECOVERY,
            "interval_stable": self.GOSSIP_INTERVAL_STABLE,
            "jitter_factor": self.GOSSIP_INTERVAL_JITTER,  # Phase 10.2: For monitoring
        }
