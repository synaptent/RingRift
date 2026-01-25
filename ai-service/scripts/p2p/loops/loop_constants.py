"""Consolidated Constants for P2P Loops.

Sprint 10 (Jan 3, 2026): Centralized loop timing and threshold constants.
Reduces ~80 LOC of scattered magic numbers across loop files.

Usage:
    from scripts.p2p.loops.loop_constants import (
        LoopIntervals,
        LoopThresholds,
        LoopLimits,
        JobDefaults,
    )

    # Access timing constants
    interval = LoopIntervals.QUEUE_POPULATOR

    # Access thresholds
    disk_limit = LoopThresholds.MAX_DISK_USAGE_PERCENT
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Jan 16, 2026: Use centralized provider timeout configuration
from app.config.provider_timeouts import ProviderTimeouts


@dataclass(frozen=True)
class LoopIntervals:
    """Timing intervals for background loops (in seconds)."""

    # Queue and sync intervals
    QUEUE_POPULATOR: float = 60.0            # 1 minute between population attempts
    QUEUE_POPULATOR_INITIAL_DELAY: float = 30.0  # Delay before first run
    QUEUE_POPULATOR_ALL_MET: float = 300.0   # 5 minutes when all targets met

    # Sync intervals
    ELO_SYNC: float = 300.0                  # 5 minutes Elo sync
    TRAINING_SYNC: float = 300.0             # 5 minutes training sync
    MANIFEST_COLLECTION: float = 300.0       # 5 minutes manifest collection
    MANIFEST_INITIAL_DELAY: float = 60.0     # HTTP server stabilization

    # Process monitoring
    STALE_PROCESS_CHECK: float = 300.0       # 5 minutes stale check
    REENABLE_CHECK: float = 3600.0           # 1 hour re-enable check

    # Scaling and coordination
    AUTO_SCALING: float = 120.0              # 2 minutes scaling check
    HEALTH_AGGREGATION: float = 60.0         # 1 minute health aggregation

    # Network and discovery
    IP_DISCOVERY: float = 300.0              # 5 minutes IP discovery
    MODEL_SYNC: float = 120.0                # 2 minutes model sync
    DATA_AGGREGATION: float = 180.0          # 3 minutes data aggregation

    # Job management
    JOB_REAPER: float = 300.0                # 5 minutes job reaper
    IDLE_DETECTION: float = 60.0             # 1 minute idle detection
    SPAWN_VERIFICATION: float = 5.0          # 5 seconds spawn check
    JOB_REASSIGNMENT: float = 60.0           # 1 minute orphan check
    PREDICTIVE_SCALING: float = 30.0         # 30 seconds queue monitoring

    # Git updates
    GIT_UPDATE_CHECK: float = float(
        os.environ.get("RINGRIFT_GIT_UPDATE_INTERVAL", "300")
    )
    GIT_UPDATE_RETRY: float = 60.0           # 1 minute retry on error


@dataclass(frozen=True)
class LoopThresholds:
    """Threshold values for loop decisions."""

    # Disk usage
    MAX_DISK_USAGE_PERCENT: int = 70         # Don't sync if disk > 70%

    # Scaling thresholds
    SCALE_UP_THRESHOLD: int = 10             # Pending items per node to scale up
    SCALE_DOWN_THRESHOLD: int = 2            # Pending items per node to scale down
    SCALE_COOLDOWN_SECONDS: float = 600.0    # 10 minutes between scale ops
    IDLE_THRESHOLD_SECONDS: float = 900.0    # 15 minutes idle before shutdown

    # Spawn verification
    SPAWN_TIMEOUT_SECONDS: float = 30.0      # 30 seconds to verify spawn
    ORPHAN_DETECTION_SECONDS: float = 300.0  # 5 minutes orphan detection

    # Tailscale discovery (sync with app.p2p.constants)
    TAILSCALE_MIN_PEERS_MAINTENANCE: int = 5


@dataclass(frozen=True)
class LoopLimits:
    """Maximum counts and limits for loop operations."""

    # Retry and init limits
    MAX_INIT_RETRIES: int = 5                # Max initialization attempts
    MAX_JOBS_TO_REAP_PER_CYCLE: int = 10     # Max jobs reaped per cycle
    MAX_REASSIGNMENTS_PER_CYCLE: int = 5     # Max job reassignments per cycle
    MAX_SYNC_OPERATIONS_PER_CYCLE: int = 5   # Max sync ops per cycle
    MAX_SCALE_PER_CYCLE: int = 3             # Max nodes scaled per cycle

    # Node limits
    MIN_NODES: int = 2                       # Minimum cluster size
    MAX_NODES: int = 50                      # Maximum cluster size (supports 40+ node clusters)

    # DNS and network
    DNS_TIMEOUT_SECONDS: float = 10.0        # DNS lookup timeout
    MAX_NODES_PER_IP_CYCLE: int = 20         # Max nodes to check per IP cycle

    # Sync operation limits
    SYNC_TIMEOUT_SECONDS: float = 300.0      # 5 minutes sync timeout


@dataclass(frozen=True)
class JobDefaults:
    """Default values for job-related thresholds (in seconds).

    GPU jobs use faster thresholds since issues surface quickly.
    CPU jobs can wait longer since they're cheaper.
    """

    # Stale thresholds by job type
    GPU_GUMBEL_STALE: float = 600.0          # 10 min - expensive GPU
    GPU_POLICY_STALE: float = 600.0          # 10 min - GPU inference
    GPU_SELFPLAY_STALE: float = 600.0        # 10 min - general GPU
    TRAINING_STALE: float = 1800.0           # 30 min - long init time
    EVALUATION_STALE: float = 900.0          # 15 min - time-bounded
    CPU_HEURISTIC_STALE: float = 1800.0      # 30 min - CPU is cheap
    CPU_GUMBEL_STALE: float = 1200.0         # 20 min - CPU MCTS
    SELFPLAY_STALE: float = 900.0            # 15 min - generic selfplay
    DEFAULT_STALE: float = 1800.0            # 30 min fallback

    # Stuck job threshold
    STUCK_JOB_THRESHOLD: float = 7200.0      # 2 hours

    @staticmethod
    def get_stale_thresholds() -> dict[str, float]:
        """Return the stale thresholds dict for JobReaperConfig."""
        return {
            "gpu_gumbel": JobDefaults.GPU_GUMBEL_STALE,
            "gpu_policy": JobDefaults.GPU_POLICY_STALE,
            "gpu_selfplay": JobDefaults.GPU_SELFPLAY_STALE,
            "training": JobDefaults.TRAINING_STALE,
            "evaluation": JobDefaults.EVALUATION_STALE,
            "cpu_heuristic": JobDefaults.CPU_HEURISTIC_STALE,
            "cpu_gumbel": JobDefaults.CPU_GUMBEL_STALE,
            "selfplay": JobDefaults.SELFPLAY_STALE,
            "default": JobDefaults.DEFAULT_STALE,
        }


@dataclass(frozen=True)
class LoopTimeouts:
    """Centralized timeout values for P2P operations (in seconds).

    January 2026 Sprint 10: Consolidates scattered timeout magic numbers.
    Reduces technical debt and enables consistent tuning.
    """

    # Health check timeouts
    # Jan 5, 2026: Increased base from 5.0 to 8.0 to reduce 22% false positive rate
    # With provider multipliers: vast=16s, lambda/runpod=12s, nebius/vultr=9.6s
    # Jan 16, 2026: Increased HEALTH_CHECK_FAST from 3.0 to 8.0 to match standard
    # and added HEALTH_CHECK_NAT_BLOCKED for relay nodes (reduces false positives by ~50%)
    HEALTH_CHECK: float = 8.0                # Default health check timeout
    HEALTH_CHECK_FAST: float = 8.0           # Fast health probes (was 3.0, caused false positives)
    HEALTH_CHECK_SLOW: float = 15.0          # Slow/remote nodes
    HEALTH_CHECK_NAT_BLOCKED: float = 30.0   # NAT-blocked nodes via relay

    # SSH/network timeouts
    SSH_CONNECT: float = 30.0                # SSH connection timeout
    SSH_COMMAND: float = 60.0                # SSH command execution
    SSH_TRANSFER: float = 300.0              # SSH file transfer (large files)

    # HTTP/API timeouts
    HTTP_QUICK: float = 5.0                  # Quick HTTP requests
    HTTP_STANDARD: float = 15.0              # Standard HTTP requests
    HTTP_LONG: float = 30.0                  # Long HTTP requests (data)

    # P2P-specific timeouts
    # Jan 10, 2026: Reduced from 5.0 to 2.0 to reduce lock contention on 40+ node clusters
    # Jan 24, 2026: Increased from 2.0 to 4.0 for 40+ node cluster stability
    GOSSIP_LOCK: float = 4.0                 # Gossip state lock acquisition
    GOSSIP_RPC: float = 10.0                 # Gossip RPC calls
    # Jan 5, 2026: Increased from 5.0 to 8.0 for consistency with HEALTH_CHECK
    PEER_PROBE: float = 8.0                  # Peer health probe
    # Jan 7, 2026: Reduced from 120s to 45s for faster detection of NAT-blocked peers
    PEER_PROBE_NAT: float = 45.0             # NAT-blocked peer probe

    # Leader election
    ELECTION_REQUEST: float = 3.0            # Election request timeout
    # Jan 5, 2026: Increased from 5.0 to 8.0 to reduce unnecessary election triggers
    LEADER_PROBE: float = 8.0                # Leader health probe
    STATE_TRANSFER: float = 10.0             # Leader state transfer
    VOTER_PROMOTION_CB: float = 300.0        # Voter promotion circuit breaker timeout
    DRAIN_TIMEOUT: float = 30.0              # Work drain before stepdown

    # Partition healing
    CONVERGENCE_TIMEOUT: float = 120.0       # Post-healing convergence check
    PARTITION_DISCOVERY: float = 30.0        # Peer discovery during healing

    # Peer management
    # Jan 24, 2026: Increased from 60.0 to 90.0 to match PEER_TIMEOUT in app/p2p/constants.py
    # The 30s mismatch caused nodes to be marked dead too quickly, leading to network instability
    PEER_DEAD_TIMEOUT: float = 90.0          # Peer considered dead after this (matches PEER_TIMEOUT)

    # Sync and transfer
    SYNC_LOCK: float = 120.0                 # Sync operation lock
    SYNC_OPERATION: float = 300.0            # Full sync timeout
    RSYNC_TRANSFER: float = 300.0            # Rsync file transfer
    MANIFEST_COLLECTION: float = 120.0       # Cluster manifest collection

    # Process management
    PROCESS_SPAWN: float = 30.0              # Process spawn verification
    PROCESS_GRACEFUL: float = 30.0           # Graceful shutdown timeout
    SUBPROCESS_QUICK: float = 5.0            # Quick subprocess calls
    SUBPROCESS_LONG: float = 30.0            # Long subprocess calls

    # SQLite database timeouts (lock acquisition)
    SQLITE_LOCK_QUICK: float = 5.0           # Quick lock, fail-fast acceptable
    SQLITE_LOCK_STANDARD: float = 10.0       # Standard transaction lock
    SQLITE_LOCK_LONG: float = 30.0           # Long-running DB operations

    # HTTP handler timeouts (web endpoint processing)
    # Jan 22, 2026: Increased from 30s to 45s - gossip with large cluster state
    # was taking 31s, causing cascade failures in status endpoint
    HANDLER_GOSSIP: float = 45.0             # Gossip/status handlers
    HANDLER_TOURNAMENT: float = 60.0         # Tournament/gauntlet handlers
    HANDLER_DELIVERY: float = 120.0          # File delivery handlers
    HANDLER_ADMIN: float = 300.0             # Admin/long-running handlers

    # Job and work queue timeouts
    WORK_QUEUE_ITEM: float = 3600.0          # 1 hour per work queue item
    LONG_RUNNING_JOB: float = 21600.0        # 6 hours for training/gauntlet jobs

    # Lock acquisition timeouts
    LOCK_QUICK: float = 30.0                 # Short lock window
    LOCK_STANDARD: float = 60.0              # Standard lock window

    # Provider-specific multipliers (higher latency = higher multiplier)
    PROVIDER_MULTIPLIERS: dict = None  # type: ignore  # Set in __post_init__ workaround

    @staticmethod
    def get_provider_multiplier(provider: str) -> float:
        """Get timeout multiplier for a provider.

        Returns:
            Multiplier to apply to base timeouts (e.g., 2.0 for Vast.ai).

        January 5, 2026: Increased multipliers for Vast.ai and Lambda.
        - Lambda GH200 nodes are behind NAT, causing higher connection latency
        - Vast.ai containers on consumer networks have variable latency
        - These multipliers reduce false-positive disconnections by ~50%

        January 16, 2026: Delegated to centralized ProviderTimeouts config.
        """
        # Delegate to centralized config (app/config/provider_timeouts.py)
        # Note: ProviderTimeouts expects node_id, but we're given provider prefix
        # Create a fake node_id with the provider prefix to get the multiplier
        return ProviderTimeouts.get_multiplier(f"{provider.lower()}-node")

    @staticmethod
    def get_for_provider(provider: str) -> dict[str, float]:
        """Get provider-specific timeout adjustments.

        Some providers (Vast.ai) have higher latency and need longer timeouts.
        """
        multiplier = LoopTimeouts.get_provider_multiplier(provider)
        return {
            "health_check": LoopTimeouts.HEALTH_CHECK * multiplier,
            "ssh_connect": LoopTimeouts.SSH_CONNECT * multiplier,
            "http_standard": LoopTimeouts.HTTP_STANDARD * multiplier,
            "peer_probe": LoopTimeouts.PEER_PROBE * multiplier,
        }

    @staticmethod
    def get_adaptive_timeout(
        base_timeout: float,
        provider: str = "",
        adaptive_timeout: float | None = None,
        weight_adaptive: float = 0.7,
    ) -> float:
        """Get timeout combining provider defaults with learned adaptive values.

        January 2026 Sprint 10: Enables intelligent timeout selection that learns
        from actual network conditions while respecting provider-specific baselines.

        Args:
            base_timeout: Static base timeout in seconds (e.g., HEALTH_CHECK)
            provider: Provider name for static adjustment (e.g., "vast")
            adaptive_timeout: Learned timeout from AdaptiveTimeoutTracker, or None
            weight_adaptive: Weight for adaptive value (0.0 = all static, 1.0 = all adaptive)

        Returns:
            Blended timeout in seconds.

        Example:
            # In a loop that needs adaptive health check timeout:
            from scripts.p2p.transport_cascade import TransportCascade

            cascade = TransportCascade.get_instance()
            adaptive = cascade.get_adaptive_timeout(target_node)

            timeout = LoopTimeouts.get_adaptive_timeout(
                base_timeout=LoopTimeouts.HEALTH_CHECK,
                provider="vast",
                adaptive_timeout=adaptive,
            )
        """
        # Start with static provider-adjusted timeout
        multiplier = LoopTimeouts.get_provider_multiplier(provider) if provider else 1.0
        static_timeout = base_timeout * multiplier

        if adaptive_timeout is None or adaptive_timeout <= 0:
            # No adaptive data, use static
            return static_timeout

        # Blend static and adaptive: weighted average
        # - weight_adaptive=0.7 means 70% adaptive, 30% static
        # - Provides stability from static while responding to real conditions
        weight_static = 1.0 - weight_adaptive
        blended = (adaptive_timeout * weight_adaptive) + (static_timeout * weight_static)

        # Clamp to reasonable bounds (0.5x to 2.0x of base)
        min_timeout = base_timeout * 0.5
        max_timeout = base_timeout * 2.0
        return max(min_timeout, min(max_timeout, blended))

    @staticmethod
    def get_adaptive_health_check(
        provider: str = "",
        adaptive_timeout: float | None = None,
    ) -> float:
        """Convenience method for adaptive health check timeout."""
        return LoopTimeouts.get_adaptive_timeout(
            base_timeout=LoopTimeouts.HEALTH_CHECK,
            provider=provider,
            adaptive_timeout=adaptive_timeout,
        )

    @staticmethod
    def get_adaptive_peer_probe(
        provider: str = "",
        adaptive_timeout: float | None = None,
        is_nat_blocked: bool = False,
    ) -> float:
        """Convenience method for adaptive peer probe timeout.

        Args:
            provider: Provider name for static adjustment
            adaptive_timeout: Learned timeout from transport cascade
            is_nat_blocked: If True, uses longer NAT-blocked timeout as base
        """
        base = LoopTimeouts.PEER_PROBE_NAT if is_nat_blocked else LoopTimeouts.PEER_PROBE
        return LoopTimeouts.get_adaptive_timeout(
            base_timeout=base,
            provider=provider,
            adaptive_timeout=adaptive_timeout,
        )


# Network ports (re-export for convenience)
DEFAULT_DISCOVERY_PORT: int = 8771
DEFAULT_P2P_PORT: int = 8770


# Feature flags
AUTO_UPDATE_ENABLED: bool = os.environ.get(
    "RINGRIFT_P2P_AUTO_UPDATE", ""
).lower() in ("1", "true", "yes")
