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
    # January 8, 2026: Reduced from 3600s to 180s for faster stale peer cleanup
    # January 15, 2026: Reduced to 60s for faster partition detection and recovery
    # Jan 22, 2026: INCREASED to 240s for 40-node cluster propagation.
    # Math: 40 nodes, fanout 10, 30s interval -> 2 rounds * 30s * 4 safety = 240s minimum
    gossip_state_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_GOSSIP_STATE_TTL", "240")  # 4 minutes
        )
    )

    # TTL for gossip peer manifests (seconds)
    # Jan 22, 2026: Aligned to 5x base TTL (240s * 5 = 1200s = 20 min)
    gossip_manifest_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_GOSSIP_MANIFEST_TTL", "1200")  # 20 minutes
        )
    )

    # TTL for node recovery attempts (seconds)
    # January 2026 - P2P Stability Plan Phase 3: Reduced from 6h to 2h
    # Jan 22, 2026: Aligned to 10x base TTL (240s * 10 = 2400s = 40 min)
    recovery_attempts_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_RECOVERY_ATTEMPTS_TTL", "2400")  # 40 minutes
        )
    )

    # TTL for peer reputation entries without activity (seconds)
    # Jan 22, 2026: Aligned to 30x base TTL (240s * 30 = 7200s = 2 hours)
    # Previously 24h was causing unbounded memory growth
    peer_reputation_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_PEER_REPUTATION_TTL", "7200")  # 2 hours
        )
    )

    # TTL for learned endpoints (seconds)
    # Jan 22, 2026: Aligned to 3x base TTL (240s * 3 = 720s = 12 min)
    learned_endpoints_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_LEARNED_ENDPOINTS_TTL", "720")  # 12 minutes
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
    # Jan 12, 2026: Reduced from 6h to 1h to mitigate memory pressure
    # January 2026 - P2P Stability Plan Phase 3: Reduced from 1h to 30m
    # Keeps memory footprint smaller on long-running clusters
    job_states_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_JOB_STATES_TTL", "1800")  # 30 minutes
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
    total_run_duration: float = 0.0  # Jan 7, 2026: Added for avg_run_duration calculation

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
    - TTL-based cleanup for 7 data structures
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
    6. _promotion_failures - model promotion failures (MEMORY LEAK FIX)
    7. Job state dicts (distributed_cmaes_state, ssh_tournament_runs, etc.)
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

        # 6. Clean _promotion_failures (Jan 7, 2026 - memory leak fix)
        promotion_purged = self._cleanup_promotion_failures(orchestrator, now)
        cycle_purged += promotion_purged

        # 7. Clean job state dictionaries (Jan 7, 2026 - memory leak fix)
        jobs_purged = self._cleanup_job_states(orchestrator, now)
        cycle_purged += jobs_purged

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
                f"endpoints={endpoints_purged}, promotion={promotion_purged}, "
                f"jobs={jobs_purged}"
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
                        "promotion_failures_purged": promotion_purged,
                        "job_states_purged": jobs_purged,
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

    def _cleanup_promotion_failures(self, orchestrator: Any, now: float) -> int:
        """Clean up _promotion_failures dictionary.

        Memory leak fix (Jan 7, 2026): _promotion_failures grows unbounded unlike
        _handler_failures which has a limit of 10. This method enforces:
        - Keep only last N entries per config (max_promotion_failures_per_config)
        - Purge entries older than TTL (promotion_failures_ttl_seconds)

        The _promotion_failures dict has structure:
        {config_key: [(timestamp, model_path, reason), ...], ...}
        """
        failures = getattr(orchestrator, "_promotion_failures", None)
        if not failures:
            return 0

        ttl = self.config.promotion_failures_ttl_seconds
        max_per_config = self.config.max_promotion_failures_per_config
        total_purged = 0

        configs_to_remove = []

        for config_key, failure_list in failures.items():
            if not isinstance(failure_list, list):
                continue

            original_count = len(failure_list)

            # Filter by TTL - keep only recent failures
            fresh_failures = []
            for entry in failure_list:
                # Entry format: (timestamp, model_path, reason) or dict
                if isinstance(entry, tuple) and len(entry) >= 1:
                    timestamp = entry[0]
                elif isinstance(entry, dict):
                    timestamp = entry.get("timestamp", 0)
                else:
                    continue

                if now - timestamp <= ttl:
                    fresh_failures.append(entry)

            # Enforce max entries per config - keep most recent
            if len(fresh_failures) > max_per_config:
                # Sort by timestamp descending (newest first)
                if fresh_failures and isinstance(fresh_failures[0], tuple):
                    fresh_failures.sort(key=lambda x: x[0], reverse=True)
                elif fresh_failures and isinstance(fresh_failures[0], dict):
                    fresh_failures.sort(
                        key=lambda x: x.get("timestamp", 0), reverse=True
                    )
                fresh_failures = fresh_failures[:max_per_config]

            purged_count = original_count - len(fresh_failures)
            total_purged += purged_count

            if len(fresh_failures) == 0:
                configs_to_remove.append(config_key)
            else:
                failures[config_key] = fresh_failures

        # Remove empty config entries
        for config_key in configs_to_remove:
            failures.pop(config_key, None)

        if total_purged > 0:
            self._cleanup_stats.promotion_failures_purged += total_purged
            logger.debug(
                f"[GossipStateCleanup] Purged {total_purged} promotion failures "
                f"across {len(failures)} configs"
            )

        return total_purged

    def _cleanup_job_states(self, orchestrator: Any, now: float) -> int:
        """Clean up job state dictionaries.

        Memory leak fix (Jan 7, 2026): Job state dicts grow with completed jobs.
        Cleans: distributed_cmaes_state, distributed_tournament_state,
                ssh_tournament_runs, improvement_loop_state

        Each dict typically stores job_id -> state mappings. We purge:
        - Completed jobs older than TTL
        - Oldest entries if over max_job_states
        """
        job_dicts = [
            "_distributed_cmaes_state",
            "_distributed_tournament_state",
            "_ssh_tournament_runs",
            "_improvement_loop_state",
        ]

        ttl = self.config.job_states_ttl_seconds
        max_jobs = self.config.max_job_states
        total_purged = 0

        for dict_name in job_dicts:
            job_dict = getattr(orchestrator, dict_name, None)
            if not job_dict or not isinstance(job_dict, dict):
                continue

            to_purge = []

            for job_id, state in job_dict.items():
                # Check if state has completion/timestamp info
                if isinstance(state, dict):
                    # Look for completion indicators
                    status = state.get("status", "")
                    completed_at = state.get("completed_at", 0)
                    updated_at = state.get("updated_at", 0)
                    started_at = state.get("started_at", 0)

                    # Use most recent timestamp
                    timestamp = max(completed_at, updated_at, started_at)

                    # Purge completed jobs older than TTL
                    if status in ("completed", "failed", "cancelled", "done"):
                        if timestamp > 0 and now - timestamp > ttl:
                            to_purge.append(job_id)
                    # Purge very old jobs regardless of status (likely orphaned)
                    elif timestamp > 0 and now - timestamp > ttl * 2:
                        to_purge.append(job_id)

            # Enforce max entries if still over limit
            remaining_count = len(job_dict) - len(to_purge)
            if remaining_count > max_jobs:
                remaining = [
                    (k, v) for k, v in job_dict.items() if k not in to_purge
                ]
                # Sort by timestamp, oldest first
                remaining.sort(
                    key=lambda x: max(
                        x[1].get("completed_at", 0) if isinstance(x[1], dict) else 0,
                        x[1].get("updated_at", 0) if isinstance(x[1], dict) else 0,
                        x[1].get("started_at", 0) if isinstance(x[1], dict) else 0,
                    )
                )
                excess = remaining_count - max_jobs
                to_purge.extend([k for k, _ in remaining[:excess]])

            # Perform cleanup
            for job_id in to_purge:
                job_dict.pop(job_id, None)

            total_purged += len(to_purge)

            if to_purge:
                logger.debug(
                    f"[GossipStateCleanup] Purged {len(to_purge)} entries from {dict_name}"
                )

        if total_purged > 0:
            self._cleanup_stats.job_states_purged += total_purged

        return total_purged

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
                "promotion_failures_ttl": self.config.promotion_failures_ttl_seconds,
                "job_states_ttl": self.config.job_states_ttl_seconds,
                "max_gossip_states": self.config.max_gossip_states,
                "max_gossip_manifests": self.config.max_gossip_manifests,
                "max_recovery_attempts": self.config.max_recovery_attempts,
                "max_peer_reputation": self.config.max_peer_reputation,
                "max_learned_endpoints": self.config.max_learned_endpoints,
                "max_promotion_failures_per_config": self.config.max_promotion_failures_per_config,
                "max_job_states": self.config.max_job_states,
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
                "promotion_failures_purged": stats.promotion_failures_purged,
                "job_states_purged": stats.job_states_purged,
            },
        }

    async def force_emergency_cleanup(self) -> dict[str, int]:
        """Force immediate cleanup with aggressive thresholds for memory pressure.

        January 2026 - P2P Stability Plan Phase 3:
        Called when memory pressure reaches CRITICAL tier (>80%).
        Uses 50% of normal TTLs and 50% of max limits for aggressive cleanup.

        This method:
        1. Runs cleanup immediately (doesn't wait for next interval)
        2. Uses halved TTL values for more aggressive purging
        3. Uses halved max limits
        4. Emits EMERGENCY_GOSSIP_CLEANUP event

        Returns:
            Dict with counts of purged entries by category
        """
        logger.warning("[GossipStateCleanup] EMERGENCY CLEANUP triggered by memory pressure")

        orchestrator = self._get_orchestrator()
        if not orchestrator:
            logger.error("[GossipStateCleanup] Cannot run emergency cleanup: no orchestrator")
            return {"error": 1}

        now = time.time()
        results = {}

        # Get active peer IDs for reference
        active_peers = set(getattr(orchestrator, "peers", {}).keys())

        # Save original config values
        original_config = {
            "gossip_state_ttl": self.config.gossip_state_ttl_seconds,
            "gossip_manifest_ttl": self.config.gossip_manifest_ttl_seconds,
            "recovery_attempts_ttl": self.config.recovery_attempts_ttl_seconds,
            "peer_reputation_ttl": self.config.peer_reputation_ttl_seconds,
            "learned_endpoints_ttl": self.config.learned_endpoints_ttl_seconds,
            "job_states_ttl": self.config.job_states_ttl_seconds,
            "max_gossip_states": self.config.max_gossip_states,
            "max_gossip_manifests": self.config.max_gossip_manifests,
            "max_recovery_attempts": self.config.max_recovery_attempts,
            "max_peer_reputation": self.config.max_peer_reputation,
            "max_learned_endpoints": self.config.max_learned_endpoints,
            "max_job_states": self.config.max_job_states,
        }

        try:
            # Apply aggressive thresholds (50% of normal)
            self.config.gossip_state_ttl_seconds = original_config["gossip_state_ttl"] * 0.5
            self.config.gossip_manifest_ttl_seconds = original_config["gossip_manifest_ttl"] * 0.5
            self.config.recovery_attempts_ttl_seconds = original_config["recovery_attempts_ttl"] * 0.5
            self.config.peer_reputation_ttl_seconds = original_config["peer_reputation_ttl"] * 0.5
            self.config.learned_endpoints_ttl_seconds = original_config["learned_endpoints_ttl"] * 0.5
            self.config.job_states_ttl_seconds = original_config["job_states_ttl"] * 0.5
            self.config.max_gossip_states = original_config["max_gossip_states"] // 2
            self.config.max_gossip_manifests = original_config["max_gossip_manifests"] // 2
            self.config.max_recovery_attempts = original_config["max_recovery_attempts"] // 2
            self.config.max_peer_reputation = original_config["max_peer_reputation"] // 2
            self.config.max_learned_endpoints = original_config["max_learned_endpoints"] // 2
            self.config.max_job_states = original_config["max_job_states"] // 2

            # Run cleanup with aggressive thresholds
            results["gossip_states"] = self._cleanup_gossip_peer_states(orchestrator, active_peers, now)
            results["gossip_manifests"] = self._cleanup_gossip_peer_manifests(orchestrator, active_peers, now)
            results["recovery_attempts"] = self._cleanup_node_recovery_attempts(orchestrator, active_peers, now)
            results["peer_reputation"] = self._cleanup_peer_reputation(orchestrator, active_peers, now)
            results["learned_endpoints"] = self._cleanup_learned_endpoints(orchestrator, active_peers, now)
            results["promotion_failures"] = self._cleanup_promotion_failures(orchestrator, now)
            results["job_states"] = self._cleanup_job_states(orchestrator, now)

            total_purged = sum(results.values())
            results["total"] = total_purged

            logger.warning(
                f"[GossipStateCleanup] EMERGENCY CLEANUP complete: purged {total_purged} entries "
                f"(states={results['gossip_states']}, manifests={results['gossip_manifests']}, "
                f"jobs={results['job_states']})"
            )

            # Emit emergency cleanup event
            if self._emit_event:
                self._emit_event(
                    "EMERGENCY_GOSSIP_CLEANUP_COMPLETED",
                    {
                        "total_purged": total_purged,
                        "breakdown": results,
                        "trigger": "memory_pressure",
                        "timestamp": now,
                    },
                )

        finally:
            # Restore original config values
            self.config.gossip_state_ttl_seconds = original_config["gossip_state_ttl"]
            self.config.gossip_manifest_ttl_seconds = original_config["gossip_manifest_ttl"]
            self.config.recovery_attempts_ttl_seconds = original_config["recovery_attempts_ttl"]
            self.config.peer_reputation_ttl_seconds = original_config["peer_reputation_ttl"]
            self.config.learned_endpoints_ttl_seconds = original_config["learned_endpoints_ttl"]
            self.config.job_states_ttl_seconds = original_config["job_states_ttl"]
            self.config.max_gossip_states = original_config["max_gossip_states"]
            self.config.max_gossip_manifests = original_config["max_gossip_manifests"]
            self.config.max_recovery_attempts = original_config["max_recovery_attempts"]
            self.config.max_peer_reputation = original_config["max_peer_reputation"]
            self.config.max_learned_endpoints = original_config["max_learned_endpoints"]
            self.config.max_job_states = original_config["max_job_states"]

        return results
