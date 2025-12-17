"""P2P Orchestrator Constants.

This module contains configuration constants used throughout the P2P orchestrator.
Many constants are configurable via environment variables for cluster tuning.
Extracted from p2p_orchestrator.py for better modularity.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path

# ============================================
# Network Configuration
# ============================================

DEFAULT_PORT = 8770
HEARTBEAT_INTERVAL = 30  # seconds
PEER_TIMEOUT = 90  # seconds without heartbeat = node considered dead
ELECTION_TIMEOUT = 10  # seconds to wait for election responses

# Leader lease must be comfortably larger than the heartbeat cadence
# LEARNED LESSONS: Increased from 90s to 180s - network latency between cloud providers
# can cause lease renewal to fail even with Tailscale.
LEADER_LEASE_DURATION = 180  # seconds
LEADER_LEASE_RENEW_INTERVAL = 15  # How often leader renews lease

# Leaderless fallback - trigger local training when no leader for this long
LEADERLESS_TRAINING_TIMEOUT = 180  # 3 minutes

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

HTTP_CONNECT_TIMEOUT = 10  # Fast timeout for connection phase
HTTP_TOTAL_TIMEOUT = 30    # Total request timeout
MAX_CONSECUTIVE_FAILURES = 3  # Mark node dead after 3 failures
RETRY_DEAD_NODE_INTERVAL = 300  # Retry dead nodes every 5 minutes

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
_runaway_threshold_env = (os.environ.get("RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD") or "").strip()
RUNAWAY_SELFPLAY_PROCESS_THRESHOLD = int(_runaway_threshold_env) if _runaway_threshold_env else 0

LOAD_AVERAGE_MAX_MULTIPLIER = float(os.environ.get("RINGRIFT_P2P_LOAD_AVG_MAX_MULT", "2.0") or 2.0)
SPAWN_RATE_LIMIT_PER_MINUTE = int(os.environ.get("RINGRIFT_P2P_SPAWN_RATE_LIMIT", "5") or 5)
COORDINATOR_URL = os.environ.get("RINGRIFT_COORDINATOR_URL", "")
AGENT_MODE_ENABLED = os.environ.get("RINGRIFT_P2P_AGENT_MODE", "").lower() in {"1", "true", "yes", "on"}

MAX_DISK_USAGE_PERCENT = float(os.environ.get("RINGRIFT_MAX_DISK_PERCENT", "70"))

# Arbiter URL for split-brain resolution
ARBITER_URL = os.environ.get("RINGRIFT_ARBITER_URL", "") or COORDINATOR_URL

# ============================================
# Dynamic Voter Management
# ============================================

DYNAMIC_VOTER_ENABLED = os.environ.get("RINGRIFT_DYNAMIC_VOTERS", "1").lower() in {"1", "true", "yes"}
DYNAMIC_VOTER_TARGET = int(os.environ.get("RINGRIFT_VOTER_TARGET", "7") or 7)
DYNAMIC_VOTER_MIN = int(os.environ.get("RINGRIFT_VOTER_MIN", "5") or 5)
VOTER_MIN_QUORUM = int(os.environ.get("RINGRIFT_VOTER_MIN_QUORUM", "3") or 3)
DYNAMIC_VOTER_MAX_QUORUM = int(os.environ.get("RINGRIFT_VOTER_MAX_QUORUM", "3") or 3)
VOTER_HEALTH_THRESHOLD = float(os.environ.get("RINGRIFT_VOTER_HEALTH_THRESHOLD", "0.7") or 0.7)
VOTER_PROMOTION_UPTIME = int(os.environ.get("RINGRIFT_VOTER_PROMOTION_UPTIME", "300") or 300)
VOTER_DEMOTION_FAILURES = int(os.environ.get("RINGRIFT_VOTER_DEMOTION_FAILURES", "5") or 5)

# Health-based leadership
LEADER_HEALTH_CHECK_INTERVAL = 30
LEADER_MIN_RESPONSE_RATE = float(os.environ.get("RINGRIFT_LEADER_MIN_RESPONSE_RATE", "0.6") or 0.6)
LEADER_DEGRADED_STEPDOWN_DELAY = 60

# ============================================
# Auto-Update Settings
# ============================================

GIT_UPDATE_CHECK_INTERVAL = int(os.environ.get("RINGRIFT_P2P_GIT_UPDATE_CHECK_INTERVAL", "300") or 300)
GIT_REMOTE_NAME = "origin"
GIT_BRANCH_NAME = "main"
AUTO_UPDATE_ENABLED = os.environ.get("RINGRIFT_P2P_AUTO_UPDATE", "false").strip().lower() in {"1", "true", "yes"}
GRACEFUL_SHUTDOWN_BEFORE_UPDATE = True

# ============================================
# Auth and Build Info
# ============================================

AUTH_TOKEN_ENV = "RINGRIFT_CLUSTER_AUTH_TOKEN"
AUTH_TOKEN_FILE_ENV = "RINGRIFT_CLUSTER_AUTH_TOKEN_FILE"
BUILD_VERSION_ENV = "RINGRIFT_BUILD_VERSION"
ADVERTISE_HOST_ENV = "RINGRIFT_ADVERTISE_HOST"
ADVERTISE_PORT_ENV = "RINGRIFT_ADVERTISE_PORT"

# Tailscale CGNAT space
TAILSCALE_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

# ============================================
# Data Management
# ============================================

MANIFEST_JSONL_LINECOUNT_MAX_BYTES = 50 * 1024 * 1024  # 50MB
MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES = 1024 * 1024
MANIFEST_JSONL_SAMPLE_BYTES = 256 * 1024
STARTUP_JSONL_GRACE_PERIOD_SECONDS = 120

DATA_MANAGEMENT_INTERVAL = 300
DB_EXPORT_THRESHOLD_MB = 100
TRAINING_DATA_SYNC_THRESHOLD_MB = 10
MAX_CONCURRENT_EXPORTS = 2
AUTO_TRAINING_THRESHOLD_MB = 50

# ============================================
# Training Node Sync
# ============================================

TRAINING_NODE_COUNT = 3
TRAINING_SYNC_INTERVAL = 300.0
MODEL_SYNC_INTERVAL = 300.0
MIN_GAMES_FOR_SYNC = 100

# Adaptive P2P sync intervals
P2P_DATA_SYNC_BASE = 300
P2P_DATA_SYNC_MIN = 120
P2P_DATA_SYNC_MAX = 600
P2P_MODEL_SYNC_BASE = 180
P2P_MODEL_SYNC_MIN = 60
P2P_MODEL_SYNC_MAX = 300
P2P_TRAINING_DB_SYNC_BASE = 600
P2P_TRAINING_DB_SYNC_MIN = 300
P2P_TRAINING_DB_SYNC_MAX = 900
P2P_SYNC_BACKOFF_FACTOR = 1.5
P2P_SYNC_SPEEDUP_FACTOR = 0.8

# ============================================
# State Directory
# ============================================

STATE_DIR = Path(__file__).parent.parent.parent / "logs" / "p2p_orchestrator"
STATE_DIR.mkdir(parents=True, exist_ok=True)
