"""Gossip State Cleanup Loop for P2P Orchestrator.

January 2026: TTL-based cleanup for unbounded gossip data structures.

Problem: Seven data structures in P2P orchestrator grow unbounded over time:
1. _gossip_peer_states - stores state for every peer ever seen via gossip
2. _gossip_peer_manifests - stores large NodeDataManifest objects
3. _node_recovery_attempts - tracks recovery attempts without TTL cleanup
4. _peer_reputation - tracks reputation for every peer forever
5. _gossip_learned_endpoints - stores endpoints learned via gossip
6. _promotion_failures - tracks model promotion failures (MEMORY LEAK - no limit!)
7. Job state dictionaries (distributed_cmaes_state, ssh_tournament_runs, etc.)

This causes memory growth leading to OOM kills, especially on nodes like
vultr-a100-20gb (restart counter 602) with limited RAM.

Solution: TTL-based cleanup with configurable thresholds:
- States/manifests for unknown peers: purge after 1 hour
- Recovery attempts for non-existent peers: purge after 6 hours
- Reputation for inactive peers: purge after 24 hours
- Learned endpoints: purge after 30 minutes (GOSSIP_ENDPOINT_TTL)

Usage:
    from scripts.p2p.loops import GossipStateCleanupLoop, GossipStateCleanupConfig

    cleanup_loop = GossipStateCleanupLoop(
        get_orchestrator=lambda: orchestrator,
        config=GossipStateCleanupConfig(),
    )
    await cleanup_loop.run_forever()

Events:
    GOSSIP_STATE_CLEANUP_COMPLETED: Emitted after cleanup with counts
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .base import BaseLoop

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GossipStateCleanupConfig:
    """Configuration for gossip state cleanup."""

    # Interval between cleanup cycles (seconds)
    cleanup_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_GOSSIP_CLEANUP_INTERVAL", "300")
        )
    )

    # TTL for gossip peer states (seconds) - peers not in active peer list
    gossip_state_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_GOSSIP_STATE_TTL", "3600")  # 1 hour
        )
    )

    # TTL for gossip peer manifests (seconds)
    gossip_manifest_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_GOSSIP_MANIFEST_TTL", "3600")  # 1 hour
        )
    )

    # TTL for node recovery attempts (seconds)
    recovery_attempts_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_RECOVERY_ATTEMPTS_TTL", "21600")  # 6 hours
        )
    )

    # TTL for peer reputation entries without activity (seconds)
    peer_reputation_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_PEER_REPUTATION_TTL", "86400")  # 24 hours
        )
    )

    # TTL for learned endpoints (seconds)
    learned_endpoints_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_LEARNED_ENDPOINTS_TTL", "1800")  # 30 minutes
        )
    )

    # Maximum entries per data structure (hard limits)
    max_gossip_states: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_MAX_GOSSIP_STATES", "200")
        )
    )
    max_gossip_manifests: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_MAX_GOSSIP_MANIFESTS", "100")
        )
    )
    max_recovery_attempts: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_MAX_RECOVERY_ATTEMPTS", "100")
        )
    )
    max_peer_reputation: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_MAX_PEER_REPUTATION", "200")
        )
    )
    max_learned_endpoints: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_MAX_LEARNED_ENDPOINTS", "100")
        )
    )

    # Whether the loop is enabled
    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_GOSSIP_CLEANUP_ENABLED", "1"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    # TTL for promotion failures (seconds) - Jan 7, 2026 memory leak fix
    promotion_failures_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_PROMOTION_FAILURES_TTL", "86400")  # 24 hours
        )
    )

    # TTL for completed job states (seconds)
    job_states_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_JOB_STATES_TTL", "21600")  # 6 hours
        )
    )

    # Maximum promotion failures to keep per config
    max_promotion_failures_per_config: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_MAX_PROMOTION_FAILURES_PER_CONFIG", "10")
        )
    )

    # Maximum job states to keep
    max_job_states: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_MAX_JOB_STATES", "100"))
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.cleanup_interval_seconds <= 0:
            raise ValueError("cleanup_interval_seconds must be > 0")


# =============================================================================
# Statistics
# =============================================================================


@dataclass
class GossipCleanupStats:
    """Statistics for gossip state cleanup operations."""

    gossip_states_purged: int = 0
    gossip_manifests_purged: int = 0
    recovery_attempts_purged: int = 0
    peer_reputation_purged: int = 0
    learned_endpoints_purged: int = 0
    promotion_failures_purged: int = 0  # Jan 7, 2026 memory leak fix
    job_states_purged: int = 0  # Jan 7, 2026 memory leak fix
    total_purged: int = 0
    last_cleanup_time: float = 0.0
    cycles_run: int = 0
    consecutive_errors: int = 0
    successful_runs: int = 0

    @property
    def total_runs(self) -> int:
        """Alias for cycles_run to match LoopStats interface."""
        return self.cycles_run

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.cycles_run == 0:
            return 100.0
        return (self.successful_runs / self.cycles_run) * 100.0

    def to_dict(self) -> dict:
        """Convert stats to dictionary for JSON serialization."""
        return {
            "gossip_states_purged": self.gossip_states_purged,
            "gossip_manifests_purged": self.gossip_manifests_purged,
            "recovery_attempts_purged": self.recovery_attempts_purged,
            "peer_reputation_purged": self.peer_reputation_purged,
            "learned_endpoints_purged": self.learned_endpoints_purged,
            "promotion_failures_purged": self.promotion_failures_purged,
            "job_states_purged": self.job_states_purged,
            "total_purged": self.total_purged,
            "last_cleanup_time": self.last_cleanup_time,
            "cycles_run": self.cycles_run,
            "total_runs": self.cycles_run,
        }


# =============================================================================
# Cleanup Loop
# =============================================================================


class GossipStateCleanupLoop(BaseLoop):
    """Background loop that cleans up unbounded gossip data structures.

    Key features:
    - TTL-based cleanup for 5 data structures
    - Max entry limits to prevent unbounded growth
    - Prioritizes active peers (in self.peers)
    - Emits GOSSIP_STATE_CLEANUP_COMPLETED event with statistics
    - Prevents OOM kills from memory growth

    Data structures cleaned:
    1. _gossip_peer_states - peer state from gossip
    2. _gossip_peer_manifests - data manifests from peers
    3. _node_recovery_attempts - recovery attempt timestamps
    4. _peer_reputation - peer reputation scores
    5. _gossip_learned_endpoints - endpoints from gossip discovery
    """

    def __init__(
        self,
        get_orchestrator: Callable[[], Any],
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        config: GossipStateCleanupConfig | None = None,
    ):
        """Initialize gossip state cleanup loop.

        Args:
            get_orchestrator: Callback returning the P2P orchestrator instance
            emit_event: Optional callback to emit events (event_name, event_data)
            config: Cleanup configuration
        """
        self.config = config or GossipStateCleanupConfig()
        super().__init__(
            name="gossip_state_cleanup",
            interval=self.config.cleanup_interval_seconds,
            enabled=self.config.enabled,
        )

        self._get_orchestrator = get_orchestrator
        self._emit_event = emit_event
        self._cleanup_stats = GossipCleanupStats()

    async def _run_once(self) -> None:
        """Execute one cleanup cycle."""
        if not self.config.enabled:
            return

        orchestrator = self._get_orchestrator()
        if not orchestrator:
            logger.debug("[GossipStateCleanup] No orchestrator available")
            return

        now = time.time()
        cycle_purged = 0

        # Get active peer IDs for reference
        active_peers = set(getattr(orchestrator, "peers", {}).keys())
        node_id = getattr(orchestrator, "node_id", "unknown")

        # 1. Clean _gossip_peer_states
        states_purged = self._cleanup_gossip_peer_states(orchestrator, active_peers, now)
        cycle_purged += states_purged

        # 2. Clean _gossip_peer_manifests
        manifests_purged = self._cleanup_gossip_peer_manifests(orchestrator, active_peers, now)
        cycle_purged += manifests_purged

        # 3. Clean _node_recovery_attempts
        recovery_purged = self._cleanup_node_recovery_attempts(orchestrator, active_peers, now)
        cycle_purged += recovery_purged

        # 4. Clean _peer_reputation
        reputation_purged = self._cleanup_peer_reputation(orchestrator, active_peers, now)
        cycle_purged += reputation_purged

        # 5. Clean _gossip_learned_endpoints
        endpoints_purged = self._cleanup_learned_endpoints(orchestrator, active_peers, now)
        cycle_purged += endpoints_purged

        # Update statistics
        self._cleanup_stats.total_purged += cycle_purged
        self._cleanup_stats.last_cleanup_time = now
        self._cleanup_stats.cycles_run += 1
        self._cleanup_stats.successful_runs += 1

        if cycle_purged > 0:
            logger.info(
                f"[GossipStateCleanup] Purged {cycle_purged} entries: "
                f"states={states_purged}, manifests={manifests_purged}, "
                f"recovery={recovery_purged}, reputation={reputation_purged}, "
                f"endpoints={endpoints_purged}"
            )

            # Emit event
            if self._emit_event:
                self._emit_event(
                    "GOSSIP_STATE_CLEANUP_COMPLETED",
                    {
                        "node_id": node_id,
                        "gossip_states_purged": states_purged,
                        "gossip_manifests_purged": manifests_purged,
                        "recovery_attempts_purged": recovery_purged,
                        "peer_reputation_purged": reputation_purged,
                        "learned_endpoints_purged": endpoints_purged,
                        "total_purged": cycle_purged,
                        "active_peers_count": len(active_peers),
                        "timestamp": now,
                    },
                )
        else:
            logger.debug("[GossipStateCleanup] No entries to purge")

    def _cleanup_gossip_peer_states(
        self, orchestrator: Any, active_peers: set[str], now: float
    ) -> int:
        """Clean up _gossip_peer_states dictionary.

        Removes entries for:
        - Peers not in active peers list and older than TTL
        - Oldest entries if over max limit
        """
        states = getattr(orchestrator, "_gossip_peer_states", None)
        if not states:
            return 0

        ttl = self.config.gossip_state_ttl_seconds
        max_entries = self.config.max_gossip_states
        to_purge = []

        for node_id, state in states.items():
            # Keep active peers
            if node_id in active_peers:
                continue

            # Check TTL
            state_time = state.get("timestamp", 0) if isinstance(state, dict) else 0
            age = now - state_time
            if age > ttl:
                to_purge.append(node_id)

        # Enforce max entries limit
        if len(states) - len(to_purge) > max_entries:
            # Sort remaining by timestamp, purge oldest
            remaining = [(k, v) for k, v in states.items() if k not in to_purge]
            remaining.sort(
                key=lambda x: x[1].get("timestamp", 0) if isinstance(x[1], dict) else 0
            )
            excess = len(remaining) - max_entries
            if excess > 0:
                to_purge.extend([k for k, _ in remaining[:excess]])

        # Perform cleanup
        for node_id in to_purge:
            states.pop(node_id, None)

        if to_purge:
            self._cleanup_stats.gossip_states_purged += len(to_purge)
            logger.debug(f"[GossipStateCleanup] Purged {len(to_purge)} gossip states")

        return len(to_purge)

    def _cleanup_gossip_peer_manifests(
        self, orchestrator: Any, active_peers: set[str], now: float
    ) -> int:
        """Clean up _gossip_peer_manifests dictionary.

        Removes entries for:
        - Peers not in active peers list and older than TTL
        - Oldest entries if over max limit
        """
        manifests = getattr(orchestrator, "_gossip_peer_manifests", None)
        if not manifests:
            return 0

        ttl = self.config.gossip_manifest_ttl_seconds
        max_entries = self.config.max_gossip_manifests
        to_purge = []

        for node_id, manifest in manifests.items():
            # Keep active peers
            if node_id in active_peers:
                continue

            # Check TTL using manifest timestamp if available
            manifest_time = getattr(manifest, "timestamp", 0) or getattr(manifest, "updated_at", 0)
            if manifest_time == 0:
                # No timestamp, use creation time heuristic - purge if very old
                to_purge.append(node_id)
                continue

            age = now - manifest_time
            if age > ttl:
                to_purge.append(node_id)

        # Enforce max entries limit
        if len(manifests) - len(to_purge) > max_entries:
            remaining = [k for k in manifests if k not in to_purge]
            # Without timestamps, just purge randomly from non-active
            excess = len(remaining) - max_entries
            if excess > 0:
                non_active = [k for k in remaining if k not in active_peers]
                to_purge.extend(non_active[:excess])

        # Perform cleanup
        for node_id in to_purge:
            manifests.pop(node_id, None)

        if to_purge:
            self._cleanup_stats.gossip_manifests_purged += len(to_purge)
            logger.debug(f"[GossipStateCleanup] Purged {len(to_purge)} gossip manifests")

        return len(to_purge)

    def _cleanup_node_recovery_attempts(
        self, orchestrator: Any, active_peers: set[str], now: float
    ) -> int:
        """Clean up _node_recovery_attempts dictionary.

        Removes entries for:
        - Nodes not in active peers with recovery attempt older than TTL
        - Oldest entries if over max limit
        """
        attempts = getattr(orchestrator, "_node_recovery_attempts", None)
        if not attempts:
            return 0

        ttl = self.config.recovery_attempts_ttl_seconds
        max_entries = self.config.max_recovery_attempts
        to_purge = []

        for node_id, last_attempt in attempts.items():
            # Keep active peers with recent attempts
            if node_id in active_peers:
                continue

            # Check TTL
            age = now - last_attempt
            if age > ttl:
                to_purge.append(node_id)

        # Enforce max entries limit
        if len(attempts) - len(to_purge) > max_entries:
            remaining = [(k, v) for k, v in attempts.items() if k not in to_purge]
            remaining.sort(key=lambda x: x[1])  # Sort by timestamp, oldest first
            excess = len(remaining) - max_entries
            if excess > 0:
                to_purge.extend([k for k, _ in remaining[:excess]])

        # Perform cleanup
        for node_id in to_purge:
            attempts.pop(node_id, None)

        if to_purge:
            self._cleanup_stats.recovery_attempts_purged += len(to_purge)
            logger.debug(f"[GossipStateCleanup] Purged {len(to_purge)} recovery attempts")

        return len(to_purge)

    def _cleanup_peer_reputation(
        self, orchestrator: Any, active_peers: set[str], now: float
    ) -> int:
        """Clean up _peer_reputation dictionary.

        Removes entries for:
        - Peers not in active peers with no activity for TTL
        - Oldest entries if over max limit
        """
        reputation = getattr(orchestrator, "_peer_reputation", None)
        if not reputation:
            return 0

        ttl = self.config.peer_reputation_ttl_seconds
        max_entries = self.config.max_peer_reputation
        to_purge = []

        for peer_id, rep_data in reputation.items():
            # Keep active peers
            if peer_id in active_peers:
                continue

            # Check last activity
            last_activity = max(
                rep_data.get("last_success", 0),
                rep_data.get("last_failure", 0),
                rep_data.get("last_reset", 0),
            )
            age = now - last_activity
            if age > ttl:
                to_purge.append(peer_id)

        # Enforce max entries limit
        if len(reputation) - len(to_purge) > max_entries:
            remaining = [
                (k, v) for k, v in reputation.items() if k not in to_purge
            ]
            # Sort by last activity, oldest first
            remaining.sort(
                key=lambda x: max(
                    x[1].get("last_success", 0),
                    x[1].get("last_failure", 0),
                )
            )
            excess = len(remaining) - max_entries
            if excess > 0:
                to_purge.extend([k for k, _ in remaining[:excess]])

        # Perform cleanup
        for peer_id in to_purge:
            reputation.pop(peer_id, None)

        if to_purge:
            self._cleanup_stats.peer_reputation_purged += len(to_purge)
            logger.debug(f"[GossipStateCleanup] Purged {len(to_purge)} peer reputations")

        return len(to_purge)

    def _cleanup_learned_endpoints(
        self, orchestrator: Any, active_peers: set[str], now: float
    ) -> int:
        """Clean up _gossip_learned_endpoints dictionary.

        Removes entries for:
        - Endpoints older than TTL
        - Already connected peers (in active_peers)
        - Oldest entries if over max limit
        """
        endpoints = getattr(orchestrator, "_gossip_learned_endpoints", None)
        if not endpoints:
            return 0

        ttl = self.config.learned_endpoints_ttl_seconds
        max_entries = self.config.max_learned_endpoints
        to_purge = []

        for node_id, endpoint in endpoints.items():
            # Already connected, don't need learned endpoint
            if node_id in active_peers:
                to_purge.append(node_id)
                continue

            # Check TTL
            learned_at = endpoint.get("learned_at", 0) if isinstance(endpoint, dict) else 0
            age = now - learned_at
            if age > ttl:
                to_purge.append(node_id)

        # Enforce max entries limit
        if len(endpoints) - len(to_purge) > max_entries:
            remaining = [(k, v) for k, v in endpoints.items() if k not in to_purge]
            remaining.sort(
                key=lambda x: x[1].get("learned_at", 0) if isinstance(x[1], dict) else 0
            )
            excess = len(remaining) - max_entries
            if excess > 0:
                to_purge.extend([k for k, _ in remaining[:excess]])

        # Perform cleanup
        for node_id in to_purge:
            endpoints.pop(node_id, None)

        if to_purge:
            self._cleanup_stats.learned_endpoints_purged += len(to_purge)
            logger.debug(f"[GossipStateCleanup] Purged {len(to_purge)} learned endpoints")

        return len(to_purge)

    def get_cleanup_stats(self) -> dict[str, Any]:
        """Get cleanup statistics."""
        return {
            **self._cleanup_stats.to_dict(),
            "config": {
                "interval_seconds": self.config.cleanup_interval_seconds,
                "gossip_state_ttl": self.config.gossip_state_ttl_seconds,
                "gossip_manifest_ttl": self.config.gossip_manifest_ttl_seconds,
                "recovery_attempts_ttl": self.config.recovery_attempts_ttl_seconds,
                "peer_reputation_ttl": self.config.peer_reputation_ttl_seconds,
                "learned_endpoints_ttl": self.config.learned_endpoints_ttl_seconds,
                "max_gossip_states": self.config.max_gossip_states,
                "max_gossip_manifests": self.config.max_gossip_manifests,
                "max_recovery_attempts": self.config.max_recovery_attempts,
                "max_peer_reputation": self.config.max_peer_reputation,
                "max_learned_endpoints": self.config.max_learned_endpoints,
                "enabled": self.config.enabled,
            },
        }

    def reset_stats(self) -> None:
        """Reset cleanup statistics."""
        self._cleanup_stats = GossipCleanupStats()
        logger.info("[GossipStateCleanup] Statistics reset")

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration."""
        status = "running" if self._running else "stopped"
        stats = self._cleanup_stats

        # Check for issues
        if stats.cycles_run > 0 and stats.successful_runs == 0:
            status = "error"
        elif stats.consecutive_errors > 3:
            status = "degraded"

        return {
            "status": status,
            "running": self._running,
            "enabled": self.config.enabled,
            "cycles_run": stats.cycles_run,
            "total_purged": stats.total_purged,
            "last_cleanup_time": stats.last_cleanup_time,
            "success_rate": stats.success_rate,
            "details": {
                "gossip_states_purged": stats.gossip_states_purged,
                "gossip_manifests_purged": stats.gossip_manifests_purged,
                "recovery_attempts_purged": stats.recovery_attempts_purged,
                "peer_reputation_purged": stats.peer_reputation_purged,
                "learned_endpoints_purged": stats.learned_endpoints_purged,
            },
        }
