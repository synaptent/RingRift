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
    )
except ImportError:
    BASELINE_ELO_RANDOM = 400  # Random AI pinned at 400 Elo
    INITIAL_ELO_RATING = 1500.0
    ELO_K_FACTOR = 32

# ============================================
# Network Configuration
# ============================================

DEFAULT_PORT = 8770

# Tailscale uses the IPv4 CGNAT range 100.64.0.0/10 for node IPs.
# Helpers treat hosts in this range as "Tailscale endpoints".
TAILSCALE_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

# Dec 2025 (Phase 2): Reduced from 30s to 15s for faster failure detection
# Matches RELAY_HEARTBEAT_INTERVAL. 10s would match voters but may cause
# false positives on congested networks.
HEARTBEAT_INTERVAL = 15  # seconds
# Dec 2025: Reduced from 90s to 60s - with 15s heartbeats, 4 missed = dead
# Previous 90s with 30s heartbeat meant 3 missed = dead, keeping same ratio
PEER_TIMEOUT = 60  # seconds without heartbeat = node considered dead
# SUSPECT grace period: nodes transition ALIVE -> SUSPECT -> DEAD
# Dec 2025: 30s grace prevents transient network issues from causing failover
# With 15s heartbeats, this means 2 missed = suspect, 4 missed = dead
SUSPECT_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_SUSPECT_TIMEOUT", "30") or 30)
ELECTION_TIMEOUT = 10  # seconds to wait for election responses

# Leader lease must be comfortably larger than the heartbeat cadence
# LEARNED LESSONS: Increased from 90s to 180s - network latency between cloud providers
# can cause lease renewal to fail even with Tailscale.
LEADER_LEASE_DURATION = 180  # seconds
LEADER_LEASE_RENEW_INTERVAL = 15  # How often leader renews lease

# Leaderless fallback - trigger local training when no leader for this long
# Reduced from 180s (3min) to 30s for faster decentralized operation (Dec 2025)
# With Serf integration providing reliable failure detection, we can act quickly
LEADERLESS_TRAINING_TIMEOUT = 30  # 30 seconds - quick fallback for resilience

JOB_CHECK_INTERVAL = 60  # seconds between job status checks
DISCOVERY_PORT = 8771  # UDP port for peer discovery
DISCOVERY_INTERVAL = 120  # seconds between discovery broadcasts

# ============================================
# Resource Thresholds (80% max utilization enforced)
# ============================================

# Disk thresholds - 70% max (lower than CPU/memory because cleanup takes time)
DISK_CRITICAL_THRESHOLD = int(os.environ.get("RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD", "70") or 70)
DISK_WARNING_THRESHOLD = int(os.environ.get("RINGRIFT_P2P_DISK_WARNING_THRESHOLD", "65") or 65)
DISK_CLEANUP_THRESHOLD = int(os.environ.get("RINGRIFT_P2P_DISK_CLEANUP_THRESHOLD", "65") or 65)

# Memory thresholds - 80% max
MEMORY_CRITICAL_THRESHOLD = min(80, int(os.environ.get("RINGRIFT_P2P_MEMORY_CRITICAL_THRESHOLD", "80") or 80))
MEMORY_WARNING_THRESHOLD = min(75, int(os.environ.get("RINGRIFT_P2P_MEMORY_WARNING_THRESHOLD", "75") or 75))
MIN_MEMORY_GB_FOR_TASKS = int(os.environ.get("RINGRIFT_P2P_MIN_MEMORY_GB", "64") or 64)

# Load thresholds - 80% max
LOAD_MAX_FOR_NEW_JOBS = min(80, int(os.environ.get("RINGRIFT_P2P_LOAD_MAX_FOR_NEW_JOBS", "80") or 80))

# ============================================
# GPU Configuration
# ============================================

# GPU utilization targets (unified source preferred)
try:
    from app.coordination.resource_targets import get_resource_targets
    _unified_targets = get_resource_targets()
except ImportError:
    _unified_targets = None

if _unified_targets is not None:
    TARGET_GPU_UTIL_MIN = int(_unified_targets.gpu_min)
    TARGET_GPU_UTIL_MAX = min(80, int(_unified_targets.gpu_max))  # 80% hard cap
else:
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

# Upper bound on peer endpoints included in gossip payloads to limit message size.
GOSSIP_MAX_PEER_ENDPOINTS = int(
    os.environ.get("RINGRIFT_P2P_GOSSIP_MAX_PEER_ENDPOINTS", "25") or 25
)

# Peer lifecycle
PEER_RETIRE_AFTER_SECONDS = int(os.environ.get("RINGRIFT_P2P_PEER_RETIRE_AFTER_SECONDS", "3600") or 3600)
RETRY_RETIRED_NODE_INTERVAL = int(os.environ.get("RINGRIFT_P2P_RETRY_RETIRED_NODE_INTERVAL", "3600") or 3600)
PEER_PURGE_AFTER_SECONDS = int(os.environ.get("RINGRIFT_P2P_PEER_PURGE_AFTER_SECONDS", "21600") or 21600)

# ============================================
# NAT/Relay Settings
# ============================================

NAT_INBOUND_HEARTBEAT_STALE_SECONDS = 180
RELAY_HEARTBEAT_INTERVAL = 15
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

SWIM_ENABLED = os.environ.get("RINGRIFT_SWIM_ENABLED", "").lower() in {"1", "true", "yes", "on"}

# ============================================
# Feature Flags
# ============================================

MEMBERSHIP_MODE = os.environ.get("RINGRIFT_MEMBERSHIP_MODE", "http")
CONSENSUS_MODE = os.environ.get("RINGRIFT_CONSENSUS_MODE", "bully")

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
