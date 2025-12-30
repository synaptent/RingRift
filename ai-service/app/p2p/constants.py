"""P2P Orchestrator Constants (canonical).

This module contains configuration constants used throughout the P2P orchestrator.
Many constants are configurable via environment variables for cluster tuning.

The scripts/p2p/constants.py module re-exports this file for legacy compatibility.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path

# Import canonical Elo constants
try:
    from app.config.thresholds import (
        BASELINE_ELO_RANDOM,  # Random AI pinned at 400 Elo
        ELO_K_FACTOR,
        INITIAL_ELO_RATING,
        MIN_MEMORY_GB_FOR_TRAINING as CONFIG_MIN_MEMORY_GB_FOR_TRAINING,
    )
except ImportError:
    BASELINE_ELO_RANDOM = 400  # Random AI pinned at 400 Elo
    INITIAL_ELO_RATING = 1500.0
    ELO_K_FACTOR = 32
    CONFIG_MIN_MEMORY_GB_FOR_TRAINING = 32

try:
    from app.config.ports import SWIM_PORT, P2P_DEFAULT_PORT
except ImportError:
    SWIM_PORT = 7947
    P2P_DEFAULT_PORT = int(os.environ.get("RINGRIFT_P2P_PORT", "8770"))

# ============================================
# Network Configuration
# ============================================

# Use canonical P2P port from ports.py
DEFAULT_PORT = P2P_DEFAULT_PORT

# Tailscale uses the IPv4 CGNAT range 100.64.0.0/10 for node IPs.
# Helpers treat hosts in this range as "Tailscale endpoints".
TAILSCALE_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

# Dec 2025 (Phase 2): Reduced from 30s to 15s for faster failure detection
# Matches RELAY_HEARTBEAT_INTERVAL. 10s would match voters but may cause
# false positives on congested networks.
# Dec 2025: Now configurable via environment variable for cluster tuning.
HEARTBEAT_INTERVAL = int(os.environ.get("RINGRIFT_P2P_HEARTBEAT_INTERVAL", "15") or 15)
# Dec 2025: Originally reduced from 90s to 60s for faster failure detection.
# Dec 30, 2025: Increased back to 90s for coordinator nodes behind NAT.
# Coordinators (like local-mac) behind home NAT experience higher latency
# and need longer timeouts to avoid false-positive peer deaths.
# With 15s heartbeats, 6 missed = dead for coordinators, 4 missed for others.
PEER_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_PEER_TIMEOUT", "90") or 90)
# Original fast timeout for non-coordinator nodes in well-connected DC environments
PEER_TIMEOUT_FAST = int(os.environ.get("RINGRIFT_P2P_PEER_TIMEOUT_FAST", "60") or 60)
# SUSPECT grace period: nodes transition ALIVE -> SUSPECT -> DEAD
# Dec 29, 2025: Reduced from 60s to 30s - faster suspect detection enables quicker recovery.
# With 15s heartbeats, this means 2 missed = suspect, 4 missed = dead
SUSPECT_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_SUSPECT_TIMEOUT", "30") or 30)
# Election timeout configurable for aggressive failover mode
# Dec 29, 2025: Increased from 10 to 30 to reduce leader thrashing (5 changes/6h → 1/6h)
# Still fast enough for failover but reduces spurious elections from network hiccups
ELECTION_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_ELECTION_TIMEOUT", "30") or 30)

# Leader lease must be comfortably larger than the heartbeat cadence
# LEARNED LESSONS: Increased from 90s to 180s - network latency between cloud providers
# can cause lease renewal to fail even with Tailscale.
LEADER_LEASE_DURATION = 180  # seconds
LEADER_LEASE_RENEW_INTERVAL = 15  # How often leader renews lease

# Leaderless fallback - trigger local training when no leader for this long
# Reduced from 180s (3min) to 30s for faster decentralized operation (Dec 2025)
# With Serf integration providing reliable failure detection, we can act quickly
LEADERLESS_TRAINING_TIMEOUT = 30  # 30 seconds - quick fallback for resilience

# Dec 29, 2025: Reduced from 60s to 15s for faster job status updates
JOB_CHECK_INTERVAL = int(os.environ.get("RINGRIFT_P2P_JOB_CHECK_INTERVAL", "15") or 15)
DISCOVERY_PORT = 8771  # UDP port for peer discovery
DISCOVERY_INTERVAL = 120  # seconds between discovery broadcasts

# ============================================
# Resource Thresholds (80% max utilization enforced)
# ============================================

# Disk thresholds - 75% max (raised Dec 29, 2025 to allow jobs at ~71% disk)
# Cleanup at 60%, warning at 70%, critical at 75%
DISK_CRITICAL_THRESHOLD = int(os.environ.get("RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD", "75") or 75)
DISK_WARNING_THRESHOLD = int(os.environ.get("RINGRIFT_P2P_DISK_WARNING_THRESHOLD", "70") or 70)
DISK_CLEANUP_THRESHOLD = int(os.environ.get("RINGRIFT_P2P_DISK_CLEANUP_THRESHOLD", "60") or 60)

# Memory thresholds - 80% max
MEMORY_CRITICAL_THRESHOLD = min(80, int(os.environ.get("RINGRIFT_P2P_MEMORY_CRITICAL_THRESHOLD", "80") or 80))
MEMORY_WARNING_THRESHOLD = min(75, int(os.environ.get("RINGRIFT_P2P_MEMORY_WARNING_THRESHOLD", "75") or 75))
MIN_MEMORY_GB_FOR_TASKS = int(os.environ.get("RINGRIFT_P2P_MIN_MEMORY_GB", "64") or 64)
MIN_MEMORY_GB_FOR_TRAINING = int(
    os.environ.get(
        "RINGRIFT_P2P_MIN_MEMORY_GB_TRAINING",
        str(CONFIG_MIN_MEMORY_GB_FOR_TRAINING),
    )
    or CONFIG_MIN_MEMORY_GB_FOR_TRAINING
)

# Load thresholds - 80% max
LOAD_MAX_FOR_NEW_JOBS = min(80, int(os.environ.get("RINGRIFT_P2P_LOAD_MAX_FOR_NEW_JOBS", "80") or 80))

# ============================================
# GPU Configuration
# ============================================

# GPU utilization targets from environment variables
# NOTE: Previously called get_resource_targets() here at import time, but that
# triggered database writes which fail on readonly cluster nodes. Use env vars
# at import time; call get_resource_targets() at runtime if DB-backed values needed.
TARGET_GPU_UTIL_MIN = int(os.environ.get("RINGRIFT_P2P_TARGET_GPU_UTIL_MIN", "60") or 60)
TARGET_GPU_UTIL_MAX = min(80, int(os.environ.get("RINGRIFT_P2P_TARGET_GPU_UTIL_MAX", "80") or 80))

GH200_MIN_SELFPLAY = int(os.environ.get("RINGRIFT_P2P_GH200_MIN_SELFPLAY", "20") or 20)
GH200_MAX_SELFPLAY = int(os.environ.get("RINGRIFT_P2P_GH200_MAX_SELFPLAY", "100") or 100)

# GPU Power Rankings for training node priority
# Higher score = more powerful GPU = higher priority
GPU_POWER_RANKINGS = {
    # Data center GPUs (highest priority)
    "H100": 2000,
    "H200": 2500,
    "A100": 624,
    "A10G": 250,
    "A10": 250,
    "L40": 362,
    "V100": 125,
    # Consumer GPUs - RTX 50 series
    "5090": 419,
    "5080": 300,
    "5070": 200,
    # Consumer GPUs - RTX 40 series
    "4090": 330,
    "4080": 242,
    "4070": 184,
    "4060": 120,
    # Consumer GPUs - RTX 30 series
    "3090": 142,
    "3080": 119,
    "3070": 81,
    "3060": 51,
    # Apple Silicon
    "Apple M3": 30,
    "Apple M2": 25,
    "Apple M1": 20,
    "Apple MPS": 15,
    # Fallback
    "Unknown": 10,
}

# ============================================
# Connection Robustness
# ============================================

# HTTP timeouts increased (Dec 2025) for better cross-cloud reliability
HTTP_CONNECT_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_HTTP_CONNECT_TIMEOUT", "15"))  # Was 10
HTTP_TOTAL_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_HTTP_TOTAL_TIMEOUT", "45"))      # Was 30
MAX_CONSECUTIVE_FAILURES = 5  # Mark node dead after 5 failures (increased from 3)
RETRY_DEAD_NODE_INTERVAL = 120  # Retry dead nodes every 2 minutes (reduced from 5)

# ============================================
# Gossip Protocol
# ============================================

# Gossip fanout - number of peers to forward gossip messages to
GOSSIP_FANOUT = int(os.environ.get("RINGRIFT_P2P_GOSSIP_FANOUT", "3") or 3)
# Gossip interval - seconds between gossip rounds
# Dec 2025: Reduced from 60s to 15s for faster state convergence (6 gossip rounds per PEER_TIMEOUT)
GOSSIP_INTERVAL = int(os.environ.get("RINGRIFT_P2P_GOSSIP_INTERVAL", "15") or 15)
# Gossip jitter - randomization factor to prevent thundering herd (±10%)
GOSSIP_JITTER = float(os.environ.get("RINGRIFT_P2P_GOSSIP_JITTER", "0.2") or 0.2)
# Upper bound on peer endpoints included in gossip payloads to limit message size.
GOSSIP_MAX_PEER_ENDPOINTS = int(
    os.environ.get("RINGRIFT_P2P_GOSSIP_MAX_PEER_ENDPOINTS", "25") or 25
)

# Peer lifecycle
PEER_RETIRE_AFTER_SECONDS = int(os.environ.get("RINGRIFT_P2P_PEER_RETIRE_AFTER_SECONDS", "3600") or 3600)
# Dec 2025: CRITICAL FIX - Changed from 3600 (1 hour) to 120 (2 minutes) for active peer recovery
# Renamed from RETRY_RETIRED_NODE_INTERVAL to PEER_RECOVERY_RETRY_INTERVAL for clarity
PEER_RECOVERY_RETRY_INTERVAL = int(os.environ.get("RINGRIFT_P2P_PEER_RECOVERY_INTERVAL", "120") or 120)
# Backward compat alias (deprecated - use PEER_RECOVERY_RETRY_INTERVAL)
RETRY_RETIRED_NODE_INTERVAL = PEER_RECOVERY_RETRY_INTERVAL
PEER_PURGE_AFTER_SECONDS = int(os.environ.get("RINGRIFT_P2P_PEER_PURGE_AFTER_SECONDS", "21600") or 21600)

# Peer cache / reputation settings
PEER_CACHE_TTL_SECONDS = int(os.environ.get("RINGRIFT_P2P_PEER_CACHE_TTL_SECONDS", "604800") or 604800)
PEER_CACHE_MAX_ENTRIES = int(os.environ.get("RINGRIFT_P2P_PEER_CACHE_MAX_ENTRIES", "200") or 200)
PEER_REPUTATION_ALPHA = float(os.environ.get("RINGRIFT_P2P_PEER_REPUTATION_ALPHA", "0.2") or 0.2)

# ============================================
# NAT/Relay Settings
# ============================================

# Dec 2025: Made configurable for cluster tuning
NAT_INBOUND_HEARTBEAT_STALE_SECONDS = int(os.environ.get("RINGRIFT_P2P_NAT_STALE_SECONDS", "180") or 180)
RELAY_HEARTBEAT_INTERVAL = int(os.environ.get("RINGRIFT_P2P_RELAY_HEARTBEAT_INTERVAL", "15") or 15)
RELAY_COMMAND_TTL_SECONDS = 1800
RELAY_COMMAND_MAX_BATCH = 16
RELAY_COMMAND_MAX_ATTEMPTS = 3
RELAY_MAX_PENDING_START_JOBS = 4

# NAT recovery settings
NAT_BLOCKED_RECOVERY_TIMEOUT = 300
NAT_BLOCKED_PROBE_INTERVAL = 60
NAT_BLOCKED_PROBE_TIMEOUT = 5

# Voter heartbeat settings
VOTER_HEARTBEAT_INTERVAL = 10
VOTER_HEARTBEAT_TIMEOUT = 5
VOTER_MESH_REFRESH_INTERVAL = 30
VOTER_NAT_RECOVERY_AGGRESSIVE = True

# Advanced NAT management
NAT_STUN_LIKE_PROBE_INTERVAL = 120
NAT_SYMMETRIC_DETECTION_ENABLED = True
NAT_RELAY_PREFERENCE_THRESHOLD = 3
NAT_HOLE_PUNCH_RETRY_COUNT = 3
NAT_EXTERNAL_IP_CACHE_TTL = 300

# Peer bootstrap
PEER_BOOTSTRAP_INTERVAL = 60
PEER_BOOTSTRAP_MIN_PEERS = 3
VOTER_MIN_QUORUM = int(os.environ.get("RINGRIFT_P2P_VOTER_MIN_QUORUM", "3") or 3)

# Bootstrap seeds - initial peers to contact for mesh join
# Empty by default; loaded from distributed_hosts.yaml or CLI args
_bootstrap_seeds_env = os.environ.get("RINGRIFT_P2P_BOOTSTRAP_SEEDS", "").strip()
BOOTSTRAP_SEEDS: list[str] = [s.strip() for s in _bootstrap_seeds_env.split(",") if s.strip()] if _bootstrap_seeds_env else []

# Minimum number of bootstrap attempts per seed before moving on
MIN_BOOTSTRAP_ATTEMPTS = int(os.environ.get("RINGRIFT_P2P_MIN_BOOTSTRAP_ATTEMPTS", "3") or 3)

# Interval between bootstrap attempts when node is isolated (no connected peers)
ISOLATED_BOOTSTRAP_INTERVAL = int(os.environ.get("RINGRIFT_P2P_ISOLATED_BOOTSTRAP_INTERVAL", "30") or 30)

# Minimum connected peers to not be considered isolated
MIN_CONNECTED_PEERS = int(os.environ.get("RINGRIFT_P2P_MIN_CONNECTED_PEERS", "2") or 2)

# ============================================
# Cluster Epochs
# ============================================

# Initial cluster epoch value (incremented on significant cluster events)
INITIAL_CLUSTER_EPOCH = int(os.environ.get("RINGRIFT_P2P_INITIAL_CLUSTER_EPOCH", "0") or 0)

# ============================================
# Safeguards
# ============================================

GPU_IDLE_RESTART_TIMEOUT = 300
GPU_IDLE_THRESHOLD = 2
# Dec 2025: Lowered from 500 to 100 to intervene earlier on runaway processes
# Aligns with app/config/constants.py and scripts/node_resilience.py
_runaway_threshold_env = (os.environ.get("RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD") or "").strip()
RUNAWAY_SELFPLAY_PROCESS_THRESHOLD = int(_runaway_threshold_env) if _runaway_threshold_env else 100

LOAD_AVERAGE_MAX_MULTIPLIER = float(os.environ.get("RINGRIFT_P2P_LOAD_AVG_MAX_MULT", "2.0") or 2.0)
SPAWN_RATE_LIMIT_PER_MINUTE = int(os.environ.get("RINGRIFT_P2P_SPAWN_RATE_LIMIT", "5") or 5)
COORDINATOR_URL = os.environ.get("RINGRIFT_COORDINATOR_URL", "")
AGENT_MODE_ENABLED = os.environ.get("RINGRIFT_P2P_AGENT_MODE", "").lower() in {"1", "true", "yes", "on"}
AUTO_UPDATE_ENABLED = os.environ.get("RINGRIFT_P2P_AUTO_UPDATE", "").lower() in {"1", "true", "yes", "on"}

MAX_DISK_USAGE_PERCENT = float(os.environ.get("RINGRIFT_MAX_DISK_PERCENT", "70"))

# Arbiter URL for split-brain resolution
ARBITER_URL = os.environ.get("RINGRIFT_ARBITER_URL", "") or COORDINATOR_URL

# ============================================
# Raft Consensus (optional)
# ============================================

RAFT_ENABLED = os.environ.get("RINGRIFT_RAFT_ENABLED", "").lower() in {"1", "true", "yes", "on"}
RAFT_BIND_PORT = int(os.environ.get("RINGRIFT_RAFT_BIND_PORT", "4321") or 4321)
RAFT_COMPACTION_MIN_ENTRIES = int(os.environ.get("RINGRIFT_RAFT_COMPACTION_MIN_ENTRIES", "1000") or 1000)
RAFT_AUTO_UNLOCK_TIME = float(os.environ.get("RINGRIFT_RAFT_AUTO_UNLOCK_TIME", "300") or 300)

# ============================================
# SWIM membership (optional)
# ============================================

# December 29, 2025: Auto-detect swim-p2p availability for hybrid mode
# SWIM is auto-enabled when swim-p2p package is available, unless explicitly disabled
def _detect_swim_available() -> bool:
    """Check if swim-p2p package is available for SWIM protocol."""
    try:
        from swim import Node as SwimNode
        if hasattr(SwimNode, 'create'):
            return True
    except ImportError:
        pass
    return False

_SWIM_PACKAGE_AVAILABLE = _detect_swim_available()
_swim_env = os.environ.get("RINGRIFT_SWIM_ENABLED", "").lower()
# Auto-enable if package available AND not explicitly disabled
# Explicit "false"/"off"/"0" disables; empty string with package = enabled
if _swim_env in {"0", "false", "no", "off"}:
    SWIM_ENABLED = False
elif _swim_env in {"1", "true", "yes", "on"}:
    SWIM_ENABLED = True
else:
    # Auto-detect: enable if package available
    SWIM_ENABLED = _SWIM_PACKAGE_AVAILABLE

SWIM_BIND_PORT = int(os.environ.get("RINGRIFT_SWIM_BIND_PORT", str(SWIM_PORT)) or SWIM_PORT)
# December 29, 2025: Tuned for high-latency cross-cloud networks
# Original values: 5.0s failure, 3.0s suspicion, 3 indirect pings
# Increased for P99 RTT of 2.6s observed between cloud providers
SWIM_FAILURE_TIMEOUT = float(os.environ.get("RINGRIFT_SWIM_FAILURE_TIMEOUT", "10.0") or 10.0)
SWIM_SUSPICION_TIMEOUT = float(os.environ.get("RINGRIFT_SWIM_SUSPICION_TIMEOUT", "6.0") or 6.0)
SWIM_PING_INTERVAL = float(os.environ.get("RINGRIFT_SWIM_PING_INTERVAL", "1.0") or 1.0)
# Increased indirect probes from 3 to 7 per SWIM paper for better success rate
SWIM_INDIRECT_PING_COUNT = int(os.environ.get("RINGRIFT_SWIM_INDIRECT_PING_COUNT", "7") or 7)

# ============================================
# Feature Flags
# ============================================

# December 29, 2025: Changed default from "http" to "hybrid" for faster failure detection
# Hybrid mode uses SWIM when available with HTTP fallback for compatibility
MEMBERSHIP_MODE = os.environ.get("RINGRIFT_MEMBERSHIP_MODE", "hybrid")
CONSENSUS_MODE = os.environ.get("RINGRIFT_CONSENSUS_MODE", "bully")

# ============================================
# Aggressive Failover Mode (Dec 2025)
# ============================================
# When enabled, reduces failover time from ~270s to ~120s at the cost of
# potential false positives during network congestion. Opt-in only.
#
# Default timeline (conservative): 90s peer timeout + 60s suspect + 180s lease = ~330s worst case
# Aggressive timeline: 45s peer timeout + 30s suspect + 60s lease = ~135s worst case
AGGRESSIVE_FAILOVER_ENABLED = os.environ.get("RINGRIFT_P2P_AGGRESSIVE_FAILOVER", "").lower() in {"1", "true", "yes", "on"}

# Aggressive mode timeout overrides (only used when AGGRESSIVE_FAILOVER_ENABLED=true)
AGGRESSIVE_PEER_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_AGGRESSIVE_PEER_TIMEOUT", "45") or 45)
AGGRESSIVE_SUSPECT_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_AGGRESSIVE_SUSPECT_TIMEOUT", "30") or 30)
AGGRESSIVE_LEADER_LEASE_DURATION = int(os.environ.get("RINGRIFT_P2P_AGGRESSIVE_LEASE_DURATION", "60") or 60)
AGGRESSIVE_ELECTION_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_AGGRESSIVE_ELECTION_TIMEOUT", "5") or 5)

# Helper functions to get effective timeout values based on mode
def get_effective_peer_timeout() -> int:
    """Return peer timeout based on failover mode."""
    if AGGRESSIVE_FAILOVER_ENABLED:
        return AGGRESSIVE_PEER_TIMEOUT
    return PEER_TIMEOUT

def get_effective_suspect_timeout() -> int:
    """Return suspect timeout based on failover mode."""
    if AGGRESSIVE_FAILOVER_ENABLED:
        return AGGRESSIVE_SUSPECT_TIMEOUT
    return SUSPECT_TIMEOUT

def get_effective_leader_lease_duration() -> int:
    """Return leader lease duration based on failover mode."""
    if AGGRESSIVE_FAILOVER_ENABLED:
        return AGGRESSIVE_LEADER_LEASE_DURATION
    return LEADER_LEASE_DURATION

def get_effective_election_timeout() -> int:
    """Return election timeout based on failover mode."""
    if AGGRESSIVE_FAILOVER_ENABLED:
        return AGGRESSIVE_ELECTION_TIMEOUT
    return ELECTION_TIMEOUT


def get_adaptive_peer_timeout(node_id: str = "", role: str = "") -> int:
    """Return adaptive peer timeout based on node characteristics.

    Dec 30, 2025: Added to provide longer timeouts for coordinators and
    NAT-blocked nodes while keeping fast detection for DC nodes.

    Args:
        node_id: Node identifier (e.g., "local-mac", "nebius-h100-1")
        role: Node role (e.g., "coordinator", "gpu_selfplay")

    Returns:
        Peer timeout in seconds (90s for coordinators, 60s for others)
    """
    if AGGRESSIVE_FAILOVER_ENABLED:
        return AGGRESSIVE_PEER_TIMEOUT

    # Coordinators and local nodes get longer timeout for NAT resilience
    is_coordinator = role in ("coordinator", "leader")
    is_local = node_id.startswith("local-") or node_id.startswith("mac-")

    if is_coordinator or is_local:
        return PEER_TIMEOUT  # 90s for coordinators
    return PEER_TIMEOUT_FAST  # 60s for DC nodes

# ============================================
# Environment Variable Names (for reference)
# ============================================

ADVERTISE_HOST_ENV = "RINGRIFT_P2P_ADVERTISE_HOST"
ADVERTISE_PORT_ENV = "RINGRIFT_P2P_ADVERTISE_PORT"
AUTH_TOKEN_ENV = "RINGRIFT_CLUSTER_AUTH_TOKEN"
AUTH_TOKEN_FILE_ENV = "RINGRIFT_CLUSTER_AUTH_TOKEN_FILE"
BUILD_VERSION_ENV = "RINGRIFT_BUILD_VERSION"

# ============================================
# Dynamic Voter Management
# ============================================

DYNAMIC_VOTER_ENABLED = os.environ.get("RINGRIFT_P2P_DYNAMIC_VOTER", "").lower() in {"1", "true", "yes", "on"}
DYNAMIC_VOTER_MIN = int(os.environ.get("RINGRIFT_P2P_DYNAMIC_VOTER_MIN", "3") or 3)
DYNAMIC_VOTER_TARGET = int(os.environ.get("RINGRIFT_P2P_DYNAMIC_VOTER_TARGET", "5") or 5)
DYNAMIC_VOTER_MAX_QUORUM = int(os.environ.get("RINGRIFT_P2P_DYNAMIC_VOTER_MAX_QUORUM", "7") or 7)
VOTER_DEMOTION_FAILURES = int(os.environ.get("RINGRIFT_P2P_VOTER_DEMOTION_FAILURES", "3") or 3)
VOTER_HEALTH_THRESHOLD = float(os.environ.get("RINGRIFT_P2P_VOTER_HEALTH_THRESHOLD", "0.8") or 0.8)
VOTER_PROMOTION_UPTIME = int(os.environ.get("RINGRIFT_P2P_VOTER_PROMOTION_UPTIME", "3600") or 3600)  # 1 hour

# ============================================
# Leader Health
# ============================================

LEADER_HEALTH_CHECK_INTERVAL = int(os.environ.get("RINGRIFT_P2P_LEADER_HEALTH_CHECK_INTERVAL", "30") or 30)
LEADER_MIN_RESPONSE_RATE = float(os.environ.get("RINGRIFT_P2P_LEADER_MIN_RESPONSE_RATE", "0.5") or 0.5)
LEADER_DEGRADED_STEPDOWN_DELAY = int(os.environ.get("RINGRIFT_P2P_LEADER_DEGRADED_STEPDOWN_DELAY", "60") or 60)

# ============================================
# Git Auto-Update
# ============================================

GIT_BRANCH_NAME = os.environ.get("RINGRIFT_P2P_GIT_BRANCH", "main")
GIT_REMOTE_NAME = os.environ.get("RINGRIFT_P2P_GIT_REMOTE", "origin")
GIT_UPDATE_CHECK_INTERVAL = int(os.environ.get("RINGRIFT_P2P_GIT_UPDATE_CHECK_INTERVAL", "300") or 300)
GRACEFUL_SHUTDOWN_BEFORE_UPDATE = int(os.environ.get("RINGRIFT_P2P_GRACEFUL_SHUTDOWN_BEFORE_UPDATE", "30") or 30)

# ============================================
# Idle Detection
# ============================================

IDLE_CHECK_INTERVAL = int(os.environ.get("RINGRIFT_P2P_IDLE_CHECK_INTERVAL", "60") or 60)
IDLE_GPU_THRESHOLD = float(os.environ.get("RINGRIFT_P2P_IDLE_GPU_THRESHOLD", "5.0") or 5.0)  # % GPU utilization
IDLE_GRACE_PERIOD = int(os.environ.get("RINGRIFT_P2P_IDLE_GRACE_PERIOD", "300") or 300)  # 5 minutes

# ============================================
# Data Management
# ============================================

DATA_MANAGEMENT_INTERVAL = int(os.environ.get("RINGRIFT_P2P_DATA_MANAGEMENT_INTERVAL", "300") or 300)
DB_EXPORT_THRESHOLD_MB = int(os.environ.get("RINGRIFT_P2P_DB_EXPORT_THRESHOLD_MB", "100") or 100)
TRAINING_DATA_SYNC_THRESHOLD_MB = int(os.environ.get("RINGRIFT_P2P_TRAINING_DATA_SYNC_THRESHOLD_MB", "10") or 10)
MAX_CONCURRENT_EXPORTS = int(os.environ.get("RINGRIFT_P2P_MAX_CONCURRENT_EXPORTS", "2") or 2)
AUTO_TRAINING_THRESHOLD_MB = int(os.environ.get("RINGRIFT_P2P_AUTO_TRAINING_THRESHOLD_MB", "50") or 50)

# JSONL manifest scanning
MANIFEST_JSONL_SAMPLE_BYTES = int(os.environ.get("RINGRIFT_P2P_MANIFEST_JSONL_SAMPLE_BYTES", "8192") or 8192)
MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES = int(os.environ.get("RINGRIFT_P2P_MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES", "65536") or 65536)
MANIFEST_JSONL_LINECOUNT_MAX_BYTES = int(os.environ.get("RINGRIFT_P2P_MANIFEST_JSONL_LINECOUNT_MAX_BYTES", "10485760") or 10485760)  # 10MB
STARTUP_JSONL_GRACE_PERIOD_SECONDS = int(os.environ.get("RINGRIFT_P2P_STARTUP_JSONL_GRACE_PERIOD", "120") or 120)

# ============================================
# Training Node Sync
# ============================================

TRAINING_NODE_COUNT = int(os.environ.get("RINGRIFT_P2P_TRAINING_NODE_COUNT", "5") or 5)
TRAINING_SYNC_INTERVAL = float(os.environ.get("RINGRIFT_P2P_TRAINING_SYNC_INTERVAL", "300.0") or 300.0)
MIN_GAMES_FOR_SYNC = int(os.environ.get("RINGRIFT_P2P_MIN_GAMES_FOR_SYNC", "100") or 100)
MODEL_SYNC_INTERVAL = int(os.environ.get("RINGRIFT_P2P_MODEL_SYNC_INTERVAL", "300") or 300)

# ============================================
# Adaptive Sync Intervals (P2P)
# ============================================

# Data sync (game databases)
P2P_DATA_SYNC_BASE = int(os.environ.get("RINGRIFT_P2P_DATA_SYNC_BASE", "300") or 300)  # 5 minutes
P2P_DATA_SYNC_MIN = int(os.environ.get("RINGRIFT_P2P_DATA_SYNC_MIN", "60") or 60)     # 1 minute
P2P_DATA_SYNC_MAX = int(os.environ.get("RINGRIFT_P2P_DATA_SYNC_MAX", "1800") or 1800)  # 30 minutes

# Model sync
P2P_MODEL_SYNC_BASE = int(os.environ.get("RINGRIFT_P2P_MODEL_SYNC_BASE", "600") or 600)  # 10 minutes
P2P_MODEL_SYNC_MIN = int(os.environ.get("RINGRIFT_P2P_MODEL_SYNC_MIN", "120") or 120)   # 2 minutes
P2P_MODEL_SYNC_MAX = int(os.environ.get("RINGRIFT_P2P_MODEL_SYNC_MAX", "3600") or 3600)  # 1 hour

# Training DB sync (NPZ exports)
P2P_TRAINING_DB_SYNC_BASE = int(os.environ.get("RINGRIFT_P2P_TRAINING_DB_SYNC_BASE", "600") or 600)
P2P_TRAINING_DB_SYNC_MIN = int(os.environ.get("RINGRIFT_P2P_TRAINING_DB_SYNC_MIN", "120") or 120)
P2P_TRAINING_DB_SYNC_MAX = int(os.environ.get("RINGRIFT_P2P_TRAINING_DB_SYNC_MAX", "3600") or 3600)

# Sync interval adjustment factors
P2P_SYNC_SPEEDUP_FACTOR = float(os.environ.get("RINGRIFT_P2P_SYNC_SPEEDUP_FACTOR", "0.8") or 0.8)
P2P_SYNC_BACKOFF_FACTOR = float(os.environ.get("RINGRIFT_P2P_SYNC_BACKOFF_FACTOR", "1.5") or 1.5)

# ============================================
# Stale Process Cleanup
# ============================================

STALE_PROCESS_CHECK_INTERVAL = int(os.environ.get("RINGRIFT_P2P_STALE_PROCESS_CHECK_INTERVAL", "300") or 300)
# Process name patterns to kill when stale (configurable via env as comma-separated)
_stale_patterns_env = os.environ.get("RINGRIFT_P2P_STALE_PROCESS_PATTERNS", "").strip()
STALE_PROCESS_PATTERNS: list[str] = [p.strip() for p in _stale_patterns_env.split(",") if p.strip()] if _stale_patterns_env else [
    "python.*selfplay",
    "python.*training",
    "python.*gauntlet",
    "python.*tournament",
]

# Max runtime limits for various job types
MAX_SELFPLAY_RUNTIME = int(os.environ.get("RINGRIFT_P2P_MAX_SELFPLAY_RUNTIME", "7200") or 7200)    # 2 hours
MAX_TRAINING_RUNTIME = int(os.environ.get("RINGRIFT_P2P_MAX_TRAINING_RUNTIME", "86400") or 86400)  # 24 hours
MAX_TOURNAMENT_RUNTIME = int(os.environ.get("RINGRIFT_P2P_MAX_TOURNAMENT_RUNTIME", "14400") or 14400)  # 4 hours
MAX_GAUNTLET_RUNTIME = int(os.environ.get("RINGRIFT_P2P_MAX_GAUNTLET_RUNTIME", "7200") or 7200)    # 2 hours

# ============================================
# Work Assignment
# ============================================

AUTO_ASSIGN_ENABLED = os.environ.get("RINGRIFT_P2P_AUTO_ASSIGN", "true").lower() in {"1", "true", "yes", "on"}
AUTO_WORK_BATCH_SIZE = int(os.environ.get("RINGRIFT_P2P_AUTO_WORK_BATCH_SIZE", "5") or 5)

# ============================================
# Unified Discovery
# ============================================

UNIFIED_DISCOVERY_INTERVAL = int(os.environ.get("RINGRIFT_P2P_UNIFIED_DISCOVERY_INTERVAL", "60") or 60)

# ============================================
# Selfplay Scheduler (December 2025)
# ============================================

# Exploration boost default duration (15 minutes)
EXPLORATION_BOOST_DEFAULT_DURATION = int(os.environ.get("RINGRIFT_SCHEDULER_EXPLORATION_BOOST_DURATION", "900") or 900)

# Plateau penalty default duration (30 minutes)
PLATEAU_PENALTY_DEFAULT_DURATION = int(os.environ.get("RINGRIFT_SCHEDULER_PLATEAU_PENALTY_DURATION", "1800") or 1800)

# Training completion boost duration (30 minutes)
TRAINING_BOOST_DURATION = int(os.environ.get("RINGRIFT_SCHEDULER_TRAINING_BOOST_DURATION", "1800") or 1800)

# Win rate threshold for clearing plateau status (50%)
PLATEAU_CLEAR_WIN_RATE = float(os.environ.get("RINGRIFT_SCHEDULER_PLATEAU_CLEAR_WIN_RATE", "0.50") or 0.50)

# Priority change thresholds for event emission
PRIORITY_CHANGE_THRESHOLD = int(os.environ.get("RINGRIFT_SCHEDULER_PRIORITY_CHANGE_THRESHOLD", "2") or 2)
HIGH_PRIORITY_THRESHOLD = int(os.environ.get("RINGRIFT_SCHEDULER_HIGH_PRIORITY_THRESHOLD", "5") or 5)
RELATIVE_CHANGE_THRESHOLD = float(os.environ.get("RINGRIFT_SCHEDULER_RELATIVE_CHANGE_THRESHOLD", "0.5") or 0.5)

# Target change thresholds
TARGET_CHANGE_THRESHOLD = int(os.environ.get("RINGRIFT_SCHEDULER_TARGET_CHANGE_THRESHOLD", "3") or 3)

# CPU-only job spawn threshold (min CPU count)
CPU_ONLY_JOB_MIN_CPUS = int(os.environ.get("RINGRIFT_SCHEDULER_CPU_ONLY_JOB_MIN_CPUS", "128") or 128)

# Promotion penalty durations (critical/multiple/single failure)
PROMOTION_PENALTY_DURATION_CRITICAL = int(os.environ.get("RINGRIFT_SCHEDULER_PROMOTION_PENALTY_CRITICAL", "7200") or 7200)  # 2 hours
PROMOTION_PENALTY_DURATION_MULTIPLE = int(os.environ.get("RINGRIFT_SCHEDULER_PROMOTION_PENALTY_MULTIPLE", "3600") or 3600)  # 1 hour
PROMOTION_PENALTY_DURATION_SINGLE = int(os.environ.get("RINGRIFT_SCHEDULER_PROMOTION_PENALTY_SINGLE", "1800") or 1800)  # 30 min

# Promotion penalty factors (multipliers for selfplay priority)
PROMOTION_PENALTY_FACTOR_CRITICAL = float(os.environ.get("RINGRIFT_SCHEDULER_PROMOTION_PENALTY_FACTOR_CRITICAL", "0.3") or 0.3)
PROMOTION_PENALTY_FACTOR_MULTIPLE = float(os.environ.get("RINGRIFT_SCHEDULER_PROMOTION_PENALTY_FACTOR_MULTIPLE", "0.5") or 0.5)
PROMOTION_PENALTY_FACTOR_SINGLE = float(os.environ.get("RINGRIFT_SCHEDULER_PROMOTION_PENALTY_FACTOR_SINGLE", "0.7") or 0.7)

# ============================================
# Network Helpers
# ============================================

def get_default_bind_address() -> str:
    """Get default bind address for P2P services."""
    addr = os.environ.get("RINGRIFT_P2P_BIND_ADDR", "").strip()
    if addr:
        return addr
    return "0.0.0.0"


def get_default_network() -> ipaddress.IPv4Network:
    """Return default network for peer discovery (private ranges)."""
    cidr = os.environ.get("RINGRIFT_P2P_NETWORK", "10.0.0.0/8").strip()
    try:
        return ipaddress.ip_network(cidr, strict=False)
    except ValueError:
        return ipaddress.ip_network("10.0.0.0/8", strict=False)


def get_default_cluster_name() -> str:
    """Return cluster name for P2P membership grouping."""
    return os.environ.get("RINGRIFT_CLUSTER_NAME", "ringrift-cluster")


def get_default_storage_root() -> Path:
    """Return default storage root for P2P artifacts."""
    root = os.environ.get("RINGRIFT_P2P_STORAGE_ROOT", "").strip()
    if root:
        return Path(root)
    return Path.cwd()


def get_default_state_dir() -> Path:
    """Return default directory for P2P state files."""
    state_dir = os.environ.get("RINGRIFT_P2P_STATE_DIR", "").strip()
    if state_dir:
        return Path(state_dir)
    return get_default_storage_root() / "p2p_state"


# Default state directory constant (for backward compat with scripts/p2p/__init__.py)
STATE_DIR = get_default_state_dir()
