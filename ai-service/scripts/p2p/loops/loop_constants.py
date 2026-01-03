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
    MAX_NODES: int = 20                      # Maximum cluster size

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


# Network ports (re-export for convenience)
DEFAULT_DISCOVERY_PORT: int = 8771
DEFAULT_P2P_PORT: int = 8770


# Feature flags
AUTO_UPDATE_ENABLED: bool = os.environ.get(
    "RINGRIFT_P2P_AUTO_UPDATE", ""
).lower() in ("1", "true", "yes")
