"""P2P Orchestrator Constants (canonical).

This module contains configuration constants used throughout the P2P orchestrator.
Many constants are configurable via environment variables for cluster tuning.

The scripts/p2p/constants.py module re-exports this file for legacy compatibility.

Jan 29, 2026: Migrated to use TimeoutProfile for unified timeout management.
See app/config/timeout_profile.py for the single source of truth.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
from pathlib import Path

# Import unified timeout profile
try:
    from app.config.timeout_profile import (
        get_timeout_profile,
        get_jittered_timeout,
        get_jittered_peer_timeout as _profile_jittered_peer_timeout,
    )
    _TIMEOUT_PROFILE_AVAILABLE = True
except ImportError:
    _TIMEOUT_PROFILE_AVAILABLE = False

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
    CONFIG_MIN_MEMORY_GB_FOR_TRAINING = 8

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

# Jan 2026: Tailscale IPv6 network. All Tailscale nodes also get an IPv6 address
# in the fd7a:115c:a1e0::/48 range. IPv6 bypasses NAT entirely for better connectivity.
TAILSCALE_IPV6_NETWORK = ipaddress.ip_network("fd7a:115c:a1e0::/48")

# Dec 2025 (Phase 2): Reduced from 30s to 15s for faster failure detection
# Matches RELAY_HEARTBEAT_INTERVAL. 10s would match voters but may cause
# false positives on congested networks.
# Dec 2025: Now configurable via environment variable for cluster tuning.
# Jan 25, 2026: Reduced from 15s to 10s for faster failure detection.
# With 150s PEER_TIMEOUT, 10s interval = 15 missed heartbeats before DEAD.
# Jan 25, 2026 (later): Reverted to 15s for 40+ node clusters. With 40 nodes,
# 10s creates 160+ heartbeat messages/second, causing congestion and false timeouts.
# At 15s: 40 * (40-1) / 15 = 104 heartbeats/sec, much more manageable.
HEARTBEAT_INTERVAL = int(os.environ.get("RINGRIFT_P2P_HEARTBEAT_INTERVAL", "15") or 15)
# Dec 2025: Originally reduced from 90s to 60s for faster failure detection.
# Dec 30, 2025: Increased back to 90s for coordinator nodes behind NAT.
# Jan 2, 2026: Increased to 120s for NAT-blocked nodes (Lambda GH200, RunPod) that
# rely on relay mode. Relay adds latency and can cause false-positive peer deaths.
# With 15s heartbeats, 8 missed = dead for NAT-blocked, 6 for coordinators, 4 for DC.
# Jan 22, 2026: Increased to 180s to allow more time for gossip state reconciliation.
# Jan 23, 2026: REVERTED to 90s - 180s + ±25% jitter created 70s disagreement window
# causing nodes to disagree on liveness. With 90s unified timeout and ±10% jitter,
# disagreement window is only 18s, much better for gossip convergence.
# Jan 24, 2026: Increased from 90s to 120s for better stability on 40+ node clusters.
# With ±5% jitter on 120s, disagreement window is only 12s (vs 18s at 90s with ±10%).
# Jan 25, 2026: Increased from 120s to 150s for stable 20+ node connectivity.
# Jan 25, 2026 (later): Increased from 150s to 180s to reduce false peer deaths
# during queue backpressure and high CPU load periods.
# Jan 27, 2026: Aligned to 150s to match PEER_DEAD_TIMEOUT and thresholds.py
# Creates sequence: SUSPECT(60s) → PEER_TIMEOUT(150s) → RETIRE(210s).
# With 10s heartbeat, 15 missed = DEAD. ±5% jitter = 142-158s (16s window).
# Jan 28, 2026: Increased from 150s to 180s to handle DERP relay latency.
# Tailscale DERP relays (Helsinki, etc.) add 126ms+ RTT with occasional spikes.
# 180s = 12 missed 15s heartbeats, provides margin for relay congestion.
PEER_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_PEER_TIMEOUT", "180") or 180)
# Jan 23, 2026: UNIFIED - All node types now use the same timeout (90s) for consistency.
# Role-based timeouts caused different nodes to disagree on peer liveness, breaking gossip.
# NAT-blocked nodes compensate via retry/relay, not longer timeouts.
PEER_TIMEOUT_FAST = int(os.environ.get("RINGRIFT_P2P_PEER_TIMEOUT_FAST", "90") or 90)
# Jan 23, 2026: UNIFIED to 90s - was 180s which caused gossip divergence with DC nodes.
PEER_TIMEOUT_NAT_BLOCKED = int(os.environ.get("RINGRIFT_P2P_PEER_TIMEOUT_NAT_BLOCKED", "90") or 90)

# Jan 28, 2026: Unified voter timeout derived from PEER_TIMEOUT.
# Ratio of 0.30 = 45s for 150s PEER_TIMEOUT (3 missed 15s heartbeats).
# Previously voter heartbeat used hardcoded 45s while peer timeout was 150s,
# creating asymmetric liveness views where voters appeared dead to gossip
# before peers appeared dead to the voter heartbeat check.
VOTER_TIMEOUT_RATIO = float(os.environ.get("RINGRIFT_P2P_VOTER_TIMEOUT_RATIO", "0.30") or 0.30)
VOTER_HEARTBEAT_TIMEOUT_UNIFIED = int(PEER_TIMEOUT * VOTER_TIMEOUT_RATIO)


def get_peer_timeout_for_node(is_coordinator: bool = False, nat_blocked: bool = False) -> int:
    """Get peer timeout - now unified for all node types.

    Jan 23, 2026: Changed to return PEER_TIMEOUT (90s) for ALL node types.
    Role-based timeouts caused gossip divergence where different nodes disagreed
    on peer liveness. With unified timeouts, all nodes make the same decision
    within the jitter window (±10% = max 18s disagreement), allowing gossip to converge.

    NAT-blocked and coordinator nodes compensate for latency via:
    - Multiple retry attempts in heartbeat loop
    - Relay/fallback transport mechanisms
    - CPU-adaptive timeout extension (handled separately)

    Args:
        is_coordinator: True if node is a coordinator (ignored for timeout)
        nat_blocked: True if node is NAT-blocked (ignored for timeout)

    Returns:
        Timeout in seconds: Always PEER_TIMEOUT (90s) for consistency
    """
    # Jan 23, 2026: Return unified timeout regardless of role
    # This ensures all nodes agree on peer liveness within the jitter window
    return PEER_TIMEOUT


# Jan 19, 2026: Jitter factor to prevent synchronized death cascades.
# When multiple nodes check peer liveness at exactly the same timeout,
# they all mark the same slow peer dead simultaneously, causing gossip storms.
# Adding randomness desynchronizes these checks.
# Jan 24, 2026: Reduced from ±10% to ±5% for faster gossip convergence.
# Jan 25, 2026: Reduced from ±5% to ±3% to minimize disagreement window.
# Jan 25, 2026 (later): Further reduced from ±3% to ±1% for 40+ node clusters.
# Jan 25, 2026 (fix): INCREASED back from ±1% to ±5% to prevent synchronized deaths.
# CRITICAL: ±1% = 3.6s window caused ALL 40 nodes to mark same peer dead within 1 second,
# creating gossip storms (40 nodes × 3 retries = 120 messages in 1s).
# With ±5% = 18s window, peer deaths spread across 450ms/node, preventing storms.
# Gossip can converge within 18s disagreement window (gossip round = 30s).
PEER_TIMEOUT_JITTER_FACTOR = float(
    os.environ.get("RINGRIFT_P2P_PEER_TIMEOUT_JITTER", "0.05") or 0.05
)


def get_jittered_peer_timeout(base_timeout: int | None = None, node_id: str = "") -> float:
    """Get peer timeout with deterministic jitter to prevent synchronized death cascades.

    Jan 29, 2026: Changed from random to deterministic jitter based on node_id.
    All nodes now calculate the SAME jittered timeout for a given peer, eliminating
    the 18-second disagreement window that caused split-brain.

    When multiple nodes mark the same peer as dead at exactly the same time,
    they all gossip this death simultaneously, causing network storms.
    Deterministic jitter based on node_id ensures all nodes agree on timeouts.

    Args:
        base_timeout: Base timeout in seconds (defaults to PEER_TIMEOUT)
        node_id: Node identifier for deterministic jitter (required for new code)

    Returns:
        Timeout with deterministic ±PEER_TIMEOUT_JITTER_FACTOR jitter

    Note:
        If node_id is empty, falls back to random jitter for backward compatibility.
        New code should always pass node_id for deterministic behavior.
    """
    timeout = float(base_timeout if base_timeout is not None else PEER_TIMEOUT)

    if node_id and _TIMEOUT_PROFILE_AVAILABLE:
        # Use deterministic jitter from TimeoutProfile
        return get_jittered_timeout(timeout, node_id, PEER_TIMEOUT_JITTER_FACTOR)

    # Fallback: deterministic jitter using hashlib directly
    if node_id:
        hash_bytes = hashlib.md5(node_id.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:4], "big")
        normalized = hash_int / 0xFFFFFFFF  # 0.0 to 1.0
        jitter_range = timeout * PEER_TIMEOUT_JITTER_FACTOR * 2
        offset = (normalized * jitter_range) - (jitter_range / 2)
        return timeout + offset

    # Legacy fallback: random jitter (deprecated - pass node_id!)
    import random
    jitter = timeout * PEER_TIMEOUT_JITTER_FACTOR
    return timeout + random.uniform(-jitter, jitter)


# Jan 19, 2026: CPU-adaptive timeouts for busy nodes.
# GPU nodes running selfplay at 100% CPU can't respond to SWIM pings quickly.
# High CPU load -> longer timeouts to avoid false positive deaths.
CPU_LOAD_HIGH_THRESHOLD = float(
    os.environ.get("RINGRIFT_P2P_CPU_LOAD_HIGH", "0.80") or 0.80
)
CPU_LOAD_CRITICAL_THRESHOLD = float(
    os.environ.get("RINGRIFT_P2P_CPU_LOAD_CRITICAL", "0.95") or 0.95
)
# Timeout multipliers for high CPU load
# Jan 23, 2026: Reduced from 1.5x/2.0x to 1.2x/1.4x to keep disagreement windows small.
# With 90s base timeout: High CPU = 108s (+18s), Critical = 126s (+36s).
# This keeps max disagreement under 36s, achievable in 3 gossip rounds.
CPU_HIGH_TIMEOUT_MULTIPLIER = float(
    os.environ.get("RINGRIFT_P2P_CPU_HIGH_TIMEOUT_MULT", "1.2") or 1.2
)
CPU_CRITICAL_TIMEOUT_MULTIPLIER = float(
    os.environ.get("RINGRIFT_P2P_CPU_CRITICAL_TIMEOUT_MULT", "1.4") or 1.4
)


def get_cpu_adaptive_timeout(base_timeout: int | None = None, cpu_load: float | None = None) -> float:
    """Get timeout adjusted for CPU load to avoid false-positive peer deaths.

    Under high CPU load, nodes respond slowly to SWIM probes. Using fixed
    timeouts causes busy nodes to be marked dead incorrectly. This function
    extends timeouts proportionally to CPU load.

    Args:
        base_timeout: Base timeout in seconds (defaults to PEER_TIMEOUT)
        cpu_load: CPU load 0.0-1.0 (if None, queries local CPU)

    Returns:
        Adjusted timeout:
        - base_timeout if CPU < 80%
        - base_timeout * 1.5 if CPU >= 80%
        - base_timeout * 2.0 if CPU >= 95%
    """
    timeout = float(base_timeout if base_timeout is not None else PEER_TIMEOUT)

    if cpu_load is None:
        # Feb 2026: Previously called psutil.cpu_percent(interval=0.1) here which
        # blocked the event loop for 100ms per call. Called per-peer during status
        # iterations, this accumulated to 10+ seconds of blocking, causing 504
        # timeouts and SIGTERM from the supervisor. Just use base timeout when
        # no cpu_load is provided - the caller should supply it from cached data.
        return timeout

    if cpu_load >= CPU_LOAD_CRITICAL_THRESHOLD:
        return timeout * CPU_CRITICAL_TIMEOUT_MULTIPLIER
    if cpu_load >= CPU_LOAD_HIGH_THRESHOLD:
        return timeout * CPU_HIGH_TIMEOUT_MULTIPLIER
    return timeout


# Jan 19, 2026: Rate limiting for peer death detection.
# When 5+ nodes are CPU-saturated, all nodes would mark all of them dead
# simultaneously, causing gossip storms and further instability.
# Jan 20, 2026: DISABLED - rate limiting caused 8-minute inconsistent windows where
# peers were "dead" but not "retired", confusing gossip state. With single-stage
# timeout (dead = retired immediately), rate limiting is no longer needed.
# Jan 22, 2026: Changed from 100 (disabled) to 10 (moderate rate limiting).
# Rate limiting prevents cascade failures: max 10 peers marked dead per 60s cycle.
# This gives busy nodes time to recover before being marked dead by all peers.
# Set to 100 to effectively disable, or 2 for aggressive rate limiting.
PEER_DEATH_RATE_LIMIT = int(
    os.environ.get("RINGRIFT_P2P_PEER_DEATH_RATE_LIMIT", "10") or 10
)


# SUSPECT grace period: nodes transition ALIVE -> SUSPECT -> DEAD
# Dec 29, 2025: Reduced from 60s to 30s - faster suspect detection enables quicker recovery.
# Jan 2026: Reduced from 30s to 15s for aggressive SUSPECT probing (MTTR 90s → 45s)
# Jan 19, 2026: Increased from 15s to 45s - 15s was too aggressive for CPU-saturated nodes.
# Nodes at 100% CPU running selfplay can't respond to SWIM pings in time, causing mass
# SUSPECT->DEAD cascade failures. With 15s heartbeats: 3 missed = suspect, 6 missed = dead.
# Jan 21, 2026: Increased from 45s to 60s for Phase 1 timeout staggering.
# Creates sequence: SUSPECT(60s) → PEER_TIMEOUT(120s) → RETIRE(180s) to prevent race conditions.
# Jan 24, 2026: Increased from 60s to 90s to reduce false-positive SUSPECT states.
# Creates sequence: SUSPECT(90s) → PEER_TIMEOUT(120s) → RETIRE(180s).
# Jan 25, 2026: Decreased from 90s to 60s for earlier suspect detection.
# Jan 25, 2026 (later): Increased back to 90s for 40+ node clusters.
# With 60s SUSPECT, 40 nodes all marking peers SUSPECT simultaneously creates
# gossip storms (40 * 40 = 1600 state updates). At 90s SUSPECT + 15s heartbeat,
# nodes have 6 missed heartbeats before SUSPECT, allowing more time for recovery.
# Sequence: SUSPECT(90s) → DEAD(180s) → RETIRE(240s).
SUSPECT_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_SUSPECT_TIMEOUT", "90") or 90)

# Jan 2, 2026 (Sprint 3.5): Dynamic voter promotion delay
# When enabled via RINGRIFT_P2P_DYNAMIC_VOTER=true, this delay prevents premature promotion
# by waiting for a voter failure to persist before promoting a new voter.
# This avoids voter set thrashing during transient network issues.
# NOTE: Dynamic voter management is disabled by default to avoid cluster instability.
# Enable only in stable clusters with reliable network connectivity.
DYNAMIC_VOTER_PROMOTION_DELAY = int(
    os.environ.get("RINGRIFT_P2P_DYNAMIC_VOTER_PROMOTION_DELAY", "60") or 60
)
# Election timeout configurable for aggressive failover mode
# Dec 29, 2025: Increased from 10 to 30 to reduce leader thrashing (5 changes/6h → 1/6h)
# Dec 30, 2025: Reduced from 30 to 15 - faster failover after mesh stabilization,
# combined with increased PEER_RETIRE_AFTER_SECONDS to prevent false elections during restarts
ELECTION_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_ELECTION_TIMEOUT", "15") or 15)

# Leader lease must be comfortably larger than the heartbeat cadence
# LEARNED LESSONS: Increased from 90s to 180s - network latency between cloud providers
# can cause lease renewal to fail even with Tailscale.
# Jan 2, 2026: Increased from 180s to 300s to reduce election churn during transient
# network issues. 80% of 10-minute monitoring windows showed leaderless states with 180s.
# Jan 24, 2026: Restored to 300s for 40+ node cluster stability.
# 150s lease with 15s heartbeat was too aggressive for cross-cloud networks.
LEADER_LEASE_DURATION = int(os.environ.get("RINGRIFT_P2P_LEADER_LEASE_DURATION", "300") or 300)
LEADER_LEASE_RENEW_INTERVAL = 15  # How often leader renews lease

# Leaderless fallback - trigger local training when no leader for this long
# Jan 2026: Made adaptive based on election timeout + gossip convergence.
# 30s was too short - caused training duplication during normal elections.
# Now: max(45s, ELECTION_TIMEOUT + 15s gossip + 15s buffer)
# NOTE: GOSSIP_INTERVAL (defined later) is typically 15s. We use the literal here
# to avoid forward reference issues during module loading.
LEADERLESS_TRAINING_TIMEOUT_BASE = 45  # Minimum base timeout
_GOSSIP_INTERVAL_DEFAULT = 15  # Matches GOSSIP_INTERVAL default
LEADERLESS_TRAINING_TIMEOUT = max(
    LEADERLESS_TRAINING_TIMEOUT_BASE,
    ELECTION_TIMEOUT + _GOSSIP_INTERVAL_DEFAULT + 15,  # Election + gossip + buffer
)  # Adaptive timeout for leaderless training

# Leader work dispatch timeout - if leader exists but hasn't dispatched work in this long,
# allow nodes to self-assign work. This prevents idle clusters when leader is present
# but not actively dispatching work (e.g., empty queue, populator issues).
# Dec 30, 2025: Added to make network less leader-dependent
LEADER_WORK_DISPATCH_TIMEOUT = int(
    os.environ.get("RINGRIFT_P2P_LEADER_WORK_DISPATCH_TIMEOUT", "120") or 120
)  # 2 minutes - nodes self-assign if leader not dispatching

# ============================================
# Probabilistic Fallback Leadership (Jan 1, 2026)
# ============================================
# When normal elections repeatedly fail (e.g., voter quorum unavailable after partition),
# nodes can claim provisional leadership with increasing probability. This prevents
# indefinite cluster stalls while still preferring proper elections.
#
# Design principles:
# 1. Only activate after significant leaderless period (prevents race with normal elections)
# 2. Use low initial probability that grows over time (exponential backoff in reverse)
# 3. Require quorum acknowledgment OR node_id tiebreaker to finalize leadership
# 4. Use PROVISIONAL_LEADER state before becoming full LEADER
#
# Flow: leaderless N seconds → probabilistic claim → PROVISIONAL_LEADER
#       → quorum ack / tiebreaker win → LEADER (or step down if contested by higher node)

# Minimum time leaderless before fallback leadership kicks in
# Jan 2026: Reduced from 300s to 60s for faster recovery after step-down
# ULSM broadcasts step-down immediately, so 60s is ample time for normal elections
PROVISIONAL_LEADER_MIN_LEADERLESS_TIME = int(
    os.environ.get("RINGRIFT_P2P_PROVISIONAL_MIN_LEADERLESS", "60") or 60
)

# Jan 2026: How many election retry failures before activating provisional fallback
ELECTION_RETRY_COUNT_BEFORE_PROVISIONAL = int(
    os.environ.get("RINGRIFT_P2P_ELECTION_RETRIES_BEFORE_PROVISIONAL", "4") or 4
)

# Jan 2026: Time after which deterministic fallback takes over (highest eligible node wins)
# 3 minutes = 180 seconds - if still leaderless, skip probabilistic and go deterministic
DETERMINISTIC_FALLBACK_TIME = int(
    os.environ.get("RINGRIFT_P2P_DETERMINISTIC_FALLBACK_TIME", "180") or 180
)

# Initial probability of claiming provisional leadership (per check cycle)
# 0.05 = 5% chance per cycle - starts low to prevent race conditions
PROVISIONAL_LEADER_INITIAL_PROBABILITY = float(
    os.environ.get("RINGRIFT_P2P_PROVISIONAL_INITIAL_PROB", "0.05") or 0.05
)

# Maximum probability cap to prevent guaranteed immediate claim
# 0.75 = 75% max - still some randomness even after long leaderless period
PROVISIONAL_LEADER_MAX_PROBABILITY = float(
    os.environ.get("RINGRIFT_P2P_PROVISIONAL_MAX_PROB", "0.75") or 0.75
)

# Probability growth factor per minute of leaderlessness beyond minimum
# 1.3 = 30% increase per minute (compounding)
# After 5 min: 5% → after 6 min: 6.5% → after 7 min: 8.45% → after 10 min: 18.5%
PROVISIONAL_LEADER_PROBABILITY_GROWTH_RATE = float(
    os.environ.get("RINGRIFT_P2P_PROVISIONAL_GROWTH_RATE", "1.3") or 1.3
)

# How long provisional leader waits for quorum acknowledgment before self-promotion
# 60 seconds should be enough for peers to respond if reachable
PROVISIONAL_LEADER_QUORUM_TIMEOUT = int(
    os.environ.get("RINGRIFT_P2P_PROVISIONAL_QUORUM_TIMEOUT", "60") or 60
)

# How often to check for probabilistic leadership opportunity (seconds)
PROVISIONAL_LEADER_CHECK_INTERVAL = int(
    os.environ.get("RINGRIFT_P2P_PROVISIONAL_CHECK_INTERVAL", "30") or 30
)

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
# Jan 26, 2026: HTTP_TOTAL_TIMEOUT increased from 45s to 90s to allow sufficient retries
# before PEER_TIMEOUT (180s). With 15s heartbeat, this allows 6 HTTP attempts before dead.
# Previously 45s caused false timeouts - HTTP would fail before peer was actually dead.
HTTP_CONNECT_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_HTTP_CONNECT_TIMEOUT", "30"))  # Was 15
HTTP_TOTAL_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_HTTP_TOTAL_TIMEOUT", "90"))      # Was 45
# Jan 26, 2026: Increased from 5 to 12 to align with PEER_TIMEOUT/HEARTBEAT_INTERVAL
# Previously 5 failures = 75s but PEER_TIMEOUT is 180s, causing split-brain on peer liveness
MAX_CONSECUTIVE_FAILURES = 12  # Mark node dead after 12 failures (180s / 15s heartbeat)
RETRY_DEAD_NODE_INTERVAL = 120  # Retry dead nodes every 2 minutes (reduced from 5)

# ============================================
# Gossip Protocol
# ============================================

# Gossip fanout - number of peers to forward gossip messages to
# Jan 19, 2026: Increased default from 5 to 8 for larger clusters (20+ nodes)
# to improve peer visibility across NAT boundaries and reduce false disconnections.
# Use get_gossip_fanout() for adaptive fanout based on cluster size.
# Jan 2026: Split into leader/follower fanout for differentiated propagation
# Jan 23, 2026: Increased base fanout from 10 to 12 for 40+ node clusters.
# Higher fanout = faster gossip convergence but more network traffic.
# Jan 25, 2026: Increased base to 14 and leader to 18 for 20+ node stability.
# With 40 nodes and 14 fanout: reach 95% in 3 rounds (36s) vs 6 rounds (72s) with 12.
GOSSIP_FANOUT = int(os.environ.get("RINGRIFT_P2P_GOSSIP_FANOUT", "14") or 14)
# January 2026: Increased fanout for better visibility in 20-40 node clusters
# Jan 23, 2026: Leader: 16 (was 14), Follower: 14 (was 12) for >95% peer visibility
# Jan 25, 2026: Leader: 18 (was 16), Follower: 16 (was 14) for faster convergence
GOSSIP_FANOUT_LEADER = int(os.environ.get("RINGRIFT_P2P_GOSSIP_FANOUT_LEADER", "18") or 18)
GOSSIP_FANOUT_FOLLOWER = int(os.environ.get("RINGRIFT_P2P_GOSSIP_FANOUT_FOLLOWER", "16") or 16)

# Gossip lock timeout - max time to wait for gossip state lock
# Jan 2026: Increased from 2.0s to 3.0s for larger clusters with lock contention
GOSSIP_LOCK_TIMEOUT_BASE = float(os.environ.get("RINGRIFT_P2P_GOSSIP_LOCK_TIMEOUT", "3.0") or 3.0)


def get_gossip_fanout(peer_count: int, is_leader: bool = False, voter_count: int = 0) -> int:
    """Get adaptive gossip fanout with guaranteed voter coverage.

    Larger clusters need higher fanout to ensure gossip messages
    propagate to all nodes within a reasonable number of rounds.
    This is especially important for nodes behind NAT that may
    have limited direct connectivity.

    Leaders use higher fanout to ensure faster propagation of
    authoritative state (work assignments, model updates, etc.).

    Jan 28, 2026: Leaders must reach ALL voters in every gossip round.
    This ensures quorum nodes always receive leader state, preventing
    cluster fragmentation where voters are excluded from gossip.

    Args:
        peer_count: Current number of known peers in the cluster.
        is_leader: Whether this node is the cluster leader.
        voter_count: Number of voter nodes in the cluster (default 0).

    Returns:
        Recommended gossip fanout:
        - Leaders: max(size_based_fanout, voter_count) to guarantee voter coverage
        - Followers: size-based fanout (10-14 depending on cluster size)

    December 30, 2025: Added to fix peer visibility discrepancy where
    mac-studio saw only 11 peers while Nebius nodes saw 21-26.
    January 2026: Added leader/follower distinction for differentiated propagation.
    """
    # Allow environment override for testing/tuning
    env_fanout = os.environ.get("RINGRIFT_P2P_GOSSIP_FANOUT", "").strip()
    if env_fanout:
        return int(env_fanout)

    # Leader gets higher fanout for faster authoritative propagation
    # January 2026: Increased fanout for 20-40 node clusters to achieve >90% visibility
    # Jan 23, 2026: Further increased for >95% visibility in 40+ node clusters
    # Jan 25, 2026: Increased 10-20 range fanout to 14/12 for faster convergence
    if is_leader:
        if peer_count < 10:
            base = 8  # Small cluster leader
        elif peer_count < 20:
            base = 14  # Medium cluster leader
        elif peer_count < 40:
            base = 16  # Large cluster leader
        else:
            base = GOSSIP_FANOUT_LEADER  # Very large cluster leader (18)

        # Jan 28, 2026: Leaders MUST reach all voters to maintain quorum health
        return max(base, voter_count)
    else:
        # Followers use slightly lower fanout
        if peer_count < 10:
            return 6  # Small cluster follower
        elif peer_count < 20:
            return 12  # Medium cluster follower
        elif peer_count < 40:
            return 14  # Large cluster follower
        else:
            return GOSSIP_FANOUT_FOLLOWER  # Very large cluster follower (16)


# Gossip interval - seconds between gossip rounds
# Dec 2025: Reduced from 60s to 15s for faster state convergence (6 gossip rounds per PEER_TIMEOUT)
# Jan 2026: Reduced to 12s for faster convergence at 20+ node clusters
# Jan 25, 2026: Reduced to 10s for 20% faster peer discovery in 10-20 node clusters
GOSSIP_INTERVAL = int(os.environ.get("RINGRIFT_P2P_GOSSIP_INTERVAL", "10") or 10)
# Gossip jitter - randomization factor to prevent thundering herd (±10%)
GOSSIP_JITTER = float(os.environ.get("RINGRIFT_P2P_GOSSIP_JITTER", "0.2") or 0.2)
# Upper bound on peer endpoints included in gossip payloads to limit message size.
# Jan 5, 2026: Increased from 25 to 45 for 41-node cluster to ensure all peers
# are shared in gossip messages. With 25 limit, only ~60% of peers were visible.
# Jan 22, 2026: Increased from 45 to 60 for growing cluster (41+ nodes) to prevent
# gossip messages from randomly excluding peers, which can cause network partitions.
# Jan 23, 2026: Increased from 60 to 80 for reliable 40+ node operation.
# Ensures 100% peer visibility in gossip messages (each endpoint ~30 bytes, total ~2.4KB).
GOSSIP_MAX_PEER_ENDPOINTS = int(
    os.environ.get("RINGRIFT_P2P_GOSSIP_MAX_PEER_ENDPOINTS", "80") or 80
)

# Peer lifecycle
# Dec 30, 2025: Increased from 300 (5 min) to 900 (15 min) for cloud maintenance tolerance.
# Jan 5, 2026: Increased from 900 (15 min) to 1800 (30 min) for rolling restart tolerance.
# Jan 12, 2026: Reduced from 1800 (30 min) to 600 (10 min) to prevent dead node accumulation.
# 80% of cluster nodes were offline because dead nodes weren't being retired fast enough.
# Jan 20, 2026: Merged dead/retired into single-stage timeout by setting this equal to PEER_TIMEOUT.
# CRITICAL FIX: Two-stage death (120s dead, 600s retired) caused 8-minute windows where peers
# were marked "dead" but still counted in quorum calculations by some nodes, causing split-brain.
# Now dead = retired immediately (120s), eliminating the inconsistent window.
# Jan 21, 2026: Increased from 120s to 180s for Phase 1 timeout staggering.
# Creates 60s gap between PEER_TIMEOUT(120s) and RETIRE(180s) to prevent race conditions.
# Rollback: RINGRIFT_USE_LEGACY_TIMEOUTS=true
# Jan 25, 2026: Increased from 180s to 210s to maintain 60s gap with PEER_TIMEOUT=150s.
# Jan 25, 2026 (later): Increased from 210s to 240s to maintain 60s gap with PEER_TIMEOUT=180s.
# Jan 28, 2026: Kept at 240s with PEER_TIMEOUT=180s. 60s gap handles DERP relay reconnects.
# Sequence: SUSPECT(90s) → DEAD(180s) → RETIRE(240s).
PEER_RETIRE_AFTER_SECONDS = int(os.environ.get("RINGRIFT_P2P_PEER_RETIRE_AFTER_SECONDS", "240") or 240)
# Renamed from RETRY_RETIRED_NODE_INTERVAL to PEER_RECOVERY_RETRY_INTERVAL for clarity
PEER_RECOVERY_RETRY_INTERVAL = int(os.environ.get("RINGRIFT_P2P_PEER_RECOVERY_INTERVAL", "120") or 120)
# Backward compat alias (deprecated - use PEER_RECOVERY_RETRY_INTERVAL)
RETRY_RETIRED_NODE_INTERVAL = PEER_RECOVERY_RETRY_INTERVAL
# Jan 12, 2026: Reduced from 21600 (6 hr) to 3600 (1 hr) to purge dead nodes faster.
PEER_PURGE_AFTER_SECONDS = int(os.environ.get("RINGRIFT_P2P_PEER_PURGE_AFTER_SECONDS", "3600") or 3600)

# Jan 19, 2026: Rate limit peer death detection to prevent cascade failures.
# Jan 22, 2026: REMOVED DUPLICATE - PEER_DEATH_RATE_LIMIT is now defined once at line ~179.
# The canonical definition uses default of 10 (moderate rate limiting).
# NOTE: This duplicate was ignored by Python (first definition wins), causing confusion.

# Peer cache / reputation settings
PEER_CACHE_TTL_SECONDS = int(os.environ.get("RINGRIFT_P2P_PEER_CACHE_TTL_SECONDS", "604800") or 604800)
PEER_CACHE_MAX_ENTRIES = int(os.environ.get("RINGRIFT_P2P_PEER_CACHE_MAX_ENTRIES", "200") or 200)
PEER_REPUTATION_ALPHA = float(os.environ.get("RINGRIFT_P2P_PEER_REPUTATION_ALPHA", "0.2") or 0.2)

# ============================================
# NAT/Relay Settings
# ============================================

# Dec 2025: Made configurable for cluster tuning
# Jan 2026: Increased relay intervals to reduce relay saturation with 11+ NAT-blocked nodes
NAT_INBOUND_HEARTBEAT_STALE_SECONDS = int(os.environ.get("RINGRIFT_P2P_NAT_STALE_SECONDS", "180") or 180)
RELAY_HEARTBEAT_INTERVAL = int(os.environ.get("RINGRIFT_P2P_RELAY_HEARTBEAT_INTERVAL", "20") or 20)
RELAY_COMMAND_TTL_SECONDS = 1800
RELAY_COMMAND_MAX_BATCH = int(os.environ.get("RINGRIFT_P2P_RELAY_COMMAND_MAX_BATCH", "32") or 32)
RELAY_COMMAND_MAX_ATTEMPTS = 3
RELAY_MAX_PENDING_START_JOBS = 4

# NAT recovery settings
NAT_BLOCKED_RECOVERY_TIMEOUT = 300
NAT_BLOCKED_PROBE_INTERVAL = 60
NAT_BLOCKED_PROBE_TIMEOUT = 5

# Voter heartbeat settings
VOTER_HEARTBEAT_INTERVAL = 10
# Jan 11, 2026: Phase 4 - Increased from 5s to 10s for CPU-bound voters during elections
# Jan 19, 2026: CRITICAL FIX - Timeout was 10s but HEARTBEAT_INTERVAL is 15s!
# This caused voters to be marked unhealthy before they could send heartbeats.
# Timeout should be >= HEARTBEAT_INTERVAL * 3 to tolerate network jitter and CPU load.
# Jan 28, 2026: Increased from 45s to 60s to handle DERP relay latency to Hetzner voters.
# With 10s voter heartbeat interval, 60s = 6 missed heartbeats before marked unhealthy.
VOTER_HEARTBEAT_TIMEOUT = int(os.environ.get("RINGRIFT_P2P_VOTER_HEARTBEAT_TIMEOUT", "60") or 60)
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

# Dec 30, 2025: Startup grace period before continuous bootstrap loop starts.
# Jan 5, 2026: Increased from 30s to 60s - complex nodes (coordinators, training) need more time.
# 30s was too aggressive, causing false peer retirements during rolling restarts.
# Role-based variants: Coordinators 120s, GPU training 90s, GPU selfplay 60s
# Use RINGRIFT_P2P_STARTUP_GRACE_PERIOD=120 for coordinated restarts if needed.
STARTUP_GRACE_PERIOD = int(os.environ.get("RINGRIFT_P2P_STARTUP_GRACE_PERIOD", "60") or 60)

# Jan 19, 2026: Election participation delay - don't participate in elections until state loaded
# CRITICAL FIX: Nodes were voting at 5s startup but state loads in 30-50s, causing elections
# with incomplete cluster view. This led to split-brain and leader thrashing.
# Default 90s gives time for: state load (30-50s) + gossip convergence (30s) + buffer (10s)
ELECTION_PARTICIPATION_DELAY = int(
    os.environ.get("RINGRIFT_P2P_ELECTION_PARTICIPATION_DELAY", "90") or 90
)

# Jan 13, 2026: Changed default from 3 to 2 for simplified 3-voter setup
# With 3 voters, quorum=2 allows 1 failure (simple majority)
VOTER_MIN_QUORUM = int(os.environ.get("RINGRIFT_P2P_VOTER_MIN_QUORUM", "2") or 2)

# ============================================
# Leader Stickiness (Jan 2, 2026)
# ============================================
# Prefer the current incumbent leader during elections to reduce churn.
# When an election is triggered, if this node was the leader recently,
# it gets a grace period before other nodes can claim leadership.
# This prevents oscillation when the leader has transient connectivity issues.

# Grace period in seconds after stepping down before allowing other leaders
# During this window, the previous leader can reclaim without competition
INCUMBENT_LEADER_GRACE_PERIOD = int(
    os.environ.get("RINGRIFT_P2P_INCUMBENT_GRACE_PERIOD", "45") or 45
)

# Time window to consider a node as "recently was leader"
RECENT_LEADER_WINDOW = int(
    os.environ.get("RINGRIFT_P2P_RECENT_LEADER_WINDOW", "120") or 120
)

# Bootstrap seeds - initial peers to contact for mesh join
# Jan 19, 2026: Added hardcoded essential seeds for reliable bootstrap discovery.
# These are stable, always-online nodes that new peers can connect to.
# Environment variable overrides take precedence if set.
_bootstrap_seeds_env = os.environ.get("RINGRIFT_P2P_BOOTSTRAP_SEEDS", "").strip()
if _bootstrap_seeds_env:
    # Environment variable takes precedence
    BOOTSTRAP_SEEDS: list[str] = [s.strip() for s in _bootstrap_seeds_env.split(",") if s.strip()]
else:
    # Hardcoded essential seeds - stable nodes that are always online
    # Priority: Public IPs first (for non-Tailscale nodes) > Tailscale IPs
    # Jan 25, 2026: Added Lambda public IPs for bootstrap reliability (+40%)
    BOOTSTRAP_SEEDS: list[str] = [
        # Public IPs (reachable from anywhere, including Vast.ai)
        "46.62.147.150:8770",    # hetzner-cpu1 (public IP, always online)
        "135.181.39.239:8770",   # hetzner-cpu2 (public IP, always online)
        "46.62.217.168:8770",    # hetzner-cpu3 (public IP, always online)
        "208.167.249.164:8770",  # vultr-a100-20gb (public IP, stable)
        "89.169.98.165:8770",    # nebius-h100-3 (public IP, H100)
        "192.222.50.174:8770",   # lambda-gh200-8 (public IP, GH200)
        "192.222.51.29:8770",    # lambda-gh200-3 (public IP, GH200)
        # Tailscale IPs (for Tailscale-connected nodes)
        "100.94.174.19:8770",    # hetzner-cpu1 (Tailscale, relay-capable)
        "100.67.131.72:8770",    # hetzner-cpu2 (Tailscale, relay-capable)
        "100.126.21.102:8770",   # hetzner-cpu3 (Tailscale, relay-capable)
        "100.94.201.92:8770",    # vultr-a100-20gb (Tailscale, stable)
        "100.107.168.125:8770",  # mac-studio (coordinator)
        "100.71.89.91:8770",     # lambda-gh200-1 (reachable via relay)
        "100.121.230.110:8770",  # lambda-gh200-8 (direct connection)
    ]

# Minimum number of bootstrap attempts per seed before moving on
MIN_BOOTSTRAP_ATTEMPTS = int(os.environ.get("RINGRIFT_P2P_MIN_BOOTSTRAP_ATTEMPTS", "3") or 3)

# Interval between bootstrap attempts when node is isolated (no connected peers)
# Jan 2026: Reduced from 30s to 15s for faster discovery when isolated
ISOLATED_BOOTSTRAP_INTERVAL = int(os.environ.get("RINGRIFT_P2P_ISOLATED_BOOTSTRAP_INTERVAL", "15") or 15)

# Minimum connected peers to not be considered isolated
# Dec 30, 2025: Increased from 2 to 5 to prevent gossip partitioning at low peer counts
# Jan 20, 2026: Increased from 5 to 8 for 40-node clusters (20% of typical cluster size)
# Jan 25, 2026: Reduced back to 5 for current 11-18 node cluster (prevents false partition detection)
MIN_CONNECTED_PEERS = int(os.environ.get("RINGRIFT_P2P_MIN_CONNECTED_PEERS", "5") or 5)

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

# Jan 29, 2026: Changed default from disabled to enabled for Raft consensus.
# Raft provides stronger consistency guarantees than Bully algorithm.
# Use RINGRIFT_RAFT_ENABLED=false to disable if needed.
RAFT_ENABLED = os.environ.get("RINGRIFT_RAFT_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
RAFT_BIND_PORT = int(os.environ.get("RINGRIFT_RAFT_BIND_PORT", "4321") or 4321)
RAFT_COMPACTION_MIN_ENTRIES = int(os.environ.get("RINGRIFT_RAFT_COMPACTION_MIN_ENTRIES", "1000") or 1000)
RAFT_AUTO_UNLOCK_TIME = float(os.environ.get("RINGRIFT_RAFT_AUTO_UNLOCK_TIME", "300") or 300)

# Jan 24, 2026: Manual tick mode for asyncio-compatible Raft operation.
# When enabled, PySyncObj instances are created with autoTick=False and ticked
# manually by AsyncRaftManager. This prevents the 100% CPU usage caused by
# PySyncObj's internal busy-wait polling threads.
# See: app/p2p/async_raft_wrapper.py for the AsyncRaftManager implementation.
RAFT_USE_MANUAL_TICK = os.environ.get("RINGRIFT_RAFT_USE_MANUAL_TICK", "").lower() in {"1", "true", "yes", "on"}

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
# Jan 5, 2026: SWIM disabled by default because port 7947 is blocked by cloud firewalls
# (Lambda, RunPod, Vast.ai, Nebius). This was causing voter quorum failures.
# Use RINGRIFT_SWIM_ENABLED=true to explicitly enable if port 7947 is open.
if _swim_env in {"1", "true", "yes", "on"}:
    SWIM_ENABLED = _SWIM_PACKAGE_AVAILABLE  # Only enable if explicitly requested AND available
else:
    # Default disabled - port 7947 blocked on most cloud providers
    SWIM_ENABLED = False

SWIM_BIND_PORT = int(os.environ.get("RINGRIFT_SWIM_BIND_PORT", str(SWIM_PORT)) or SWIM_PORT)
# December 29, 2025: Tuned for high-latency cross-cloud networks
# Original values: 5.0s failure, 3.0s suspicion, 3 indirect pings
# Increased for P99 RTT of 2.6s observed between cloud providers
# Jan 2, 2026: Further increased to 15s/10s for NAT-blocked nodes using relay.
# Jan 19, 2026: Increased to 30s/20s - CPU-saturated nodes running selfplay at 100%
# couldn't respond to SWIM pings fast enough, causing cascade SUSPECT->DEAD failures.
# Jan 22, 2026: CRITICAL FIX - Unified SWIM and HTTP timeouts to prevent split-brain membership.
# Problem: SWIM (30s) marked peers dead while HTTP (120s) considered them alive, causing
# conflicting membership views and peer count fluctuations (5-8 instead of stable 20+).
# Solution: SWIM_FAILURE = PEER_TIMEOUT * 0.75 (90s), SWIM_SUSPICION = PEER_TIMEOUT * 0.5 (60s)
# This allows SWIM to detect failures ~25% earlier than HTTP while staying within same window.
_SWIM_FAILURE_DEFAULT = str(PEER_TIMEOUT * 0.75)  # 90s for 120s PEER_TIMEOUT
_SWIM_SUSPICION_DEFAULT = str(PEER_TIMEOUT * 0.5)  # 60s for 120s PEER_TIMEOUT
SWIM_FAILURE_TIMEOUT = float(os.environ.get("RINGRIFT_SWIM_FAILURE_TIMEOUT", _SWIM_FAILURE_DEFAULT) or float(_SWIM_FAILURE_DEFAULT))
SWIM_SUSPICION_TIMEOUT = float(os.environ.get("RINGRIFT_SWIM_SUSPICION_TIMEOUT", _SWIM_SUSPICION_DEFAULT) or float(_SWIM_SUSPICION_DEFAULT))
SWIM_PING_INTERVAL = float(os.environ.get("RINGRIFT_SWIM_PING_INTERVAL", "1.0") or 1.0)
# Increased indirect probes from 3 to 7 per SWIM paper for better success rate
SWIM_INDIRECT_PING_COUNT = int(os.environ.get("RINGRIFT_SWIM_INDIRECT_PING_COUNT", "7") or 7)

# Jan 2, 2026: Tiered SWIM timeouts based on node connectivity type
# Direct nodes (DC, well-connected): Shorter timeouts for faster detection
# Relay nodes (NAT-blocked): Longer timeouts to account for relay latency
# Jan 19, 2026: Increased all timeouts - 15s was too aggressive during high CPU load.
# Jan 22, 2026: Scaled tiered timeouts relative to unified PEER_TIMEOUT (120s) to prevent split-brain.
# Direct nodes: PEER_TIMEOUT * 0.375 failure (45s), PEER_TIMEOUT * 0.25 suspicion (30s)
# Relay nodes: PEER_TIMEOUT * 1.125 failure (135s), PEER_TIMEOUT * 0.75 suspicion (90s)
_SWIM_FAILURE_DIRECT_DEFAULT = str(PEER_TIMEOUT * 0.375)  # 45s for direct nodes
_SWIM_FAILURE_RELAY_DEFAULT = str(PEER_TIMEOUT * 1.125)   # 135s for relay nodes
_SWIM_SUSPICION_DIRECT_DEFAULT = str(PEER_TIMEOUT * 0.25) # 30s for direct nodes
_SWIM_SUSPICION_RELAY_DEFAULT = str(PEER_TIMEOUT * 0.75)  # 90s for relay nodes
SWIM_FAILURE_TIMEOUT_DIRECT = float(
    os.environ.get("RINGRIFT_SWIM_FAILURE_TIMEOUT_DIRECT", _SWIM_FAILURE_DIRECT_DEFAULT) or float(_SWIM_FAILURE_DIRECT_DEFAULT)
)
SWIM_FAILURE_TIMEOUT_RELAY = float(
    os.environ.get("RINGRIFT_SWIM_FAILURE_TIMEOUT_RELAY", _SWIM_FAILURE_RELAY_DEFAULT) or float(_SWIM_FAILURE_RELAY_DEFAULT)
)
SWIM_SUSPICION_TIMEOUT_DIRECT = float(
    os.environ.get("RINGRIFT_SWIM_SUSPICION_TIMEOUT_DIRECT", _SWIM_SUSPICION_DIRECT_DEFAULT) or float(_SWIM_SUSPICION_DIRECT_DEFAULT)
)
SWIM_SUSPICION_TIMEOUT_RELAY = float(
    os.environ.get("RINGRIFT_SWIM_SUSPICION_TIMEOUT_RELAY", _SWIM_SUSPICION_RELAY_DEFAULT) or float(_SWIM_SUSPICION_RELAY_DEFAULT)
)


def get_swim_timeouts_for_node(nat_blocked: bool = False, force_relay: bool = False) -> tuple[float, float]:
    """Get SWIM failure and suspicion timeouts based on node connectivity.

    Jan 2, 2026: Tiered timeouts reduce false positives for relay-dependent nodes
    while maintaining fast detection for directly-connected nodes.

    Args:
        nat_blocked: True if node is NAT-blocked
        force_relay: True if node uses force_relay_mode

    Returns:
        Tuple of (failure_timeout, suspicion_timeout) in seconds
    """
    if nat_blocked or force_relay:
        return SWIM_FAILURE_TIMEOUT_RELAY, SWIM_SUSPICION_TIMEOUT_RELAY
    return SWIM_FAILURE_TIMEOUT_DIRECT, SWIM_SUSPICION_TIMEOUT_DIRECT

# ============================================
# Feature Flags
# ============================================

# December 29, 2025: Changed default from "http" to "hybrid" for faster failure detection
# Hybrid mode uses SWIM when available with HTTP fallback for compatibility
# Jan 20, 2026: Changed back to "http" to resolve timeout conflicts.
# CRITICAL FIX: SWIM uses 30s timeout vs HTTP's 120s timeout. When both are active,
# SWIM marks peers dead at 30s while HTTP still considers them alive, causing
# conflicting gossip states and peer count fluctuations (11-20 instead of stable 20+).
# HTTP-only mode ensures consistent timeout behavior across all nodes.
MEMBERSHIP_MODE = os.environ.get("RINGRIFT_MEMBERSHIP_MODE", "http")
# Jan 29, 2026: Changed default from "bully" to "raft" for stronger consensus guarantees.
# Raft provides quorum-based leader election preventing split-brain.
# Use RINGRIFT_CONSENSUS_MODE=bully for legacy behavior, or =hybrid for migration.
CONSENSUS_MODE = os.environ.get("RINGRIFT_CONSENSUS_MODE", "hybrid")

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


def get_adaptive_peer_timeout(node_id: str = "", role: str = "", nat_blocked: bool = False) -> int:
    """Return adaptive peer timeout based on node characteristics.

    Dec 30, 2025: Added to provide longer timeouts for coordinators and
    NAT-blocked nodes while keeping fast detection for DC nodes.
    Jan 2, 2026: Added nat_blocked parameter for explicit relay mode detection.

    Args:
        node_id: Node identifier (e.g., "local-mac", "nebius-h100-1")
        role: Node role (e.g., "coordinator", "gpu_selfplay")
        nat_blocked: True if node uses relay mode (force_relay_mode: true)

    Returns:
        Peer timeout in seconds:
        - 120s for NAT-blocked nodes (relay mode)
        - 90s for coordinators
        - 60s for DC nodes
    """
    if AGGRESSIVE_FAILOVER_ENABLED:
        return AGGRESSIVE_PEER_TIMEOUT

    # NAT-blocked nodes get longest timeout due to relay latency
    # Detect by explicit flag or node_id patterns (Lambda, RunPod)
    is_nat_blocked = nat_blocked or _is_likely_nat_blocked(node_id)
    if is_nat_blocked:
        return PEER_TIMEOUT_NAT_BLOCKED  # 120s for NAT-blocked

    # Coordinators and local nodes get longer timeout for NAT resilience
    is_coordinator = role in ("coordinator", "leader")
    is_local = node_id.startswith("local-") or node_id.startswith("mac-")

    if is_coordinator or is_local:
        return PEER_TIMEOUT  # 90s for coordinators
    return PEER_TIMEOUT_FAST  # 60s for DC nodes


def _is_likely_nat_blocked(node_id: str) -> bool:
    """Heuristic to detect NAT-blocked nodes by node_id pattern.

    Jan 2, 2026: Lambda GH200 and RunPod nodes are typically NAT-blocked
    and require relay mode for connectivity.

    Jan 20, 2026: Profiled Lambda nodes - only some are truly NAT-blocked.
    Nodes that can receive inbound connections should NOT be marked as blocked.

    Args:
        node_id: Node identifier

    Returns:
        True if node is likely NAT-blocked based on naming pattern
    """
    if not node_id:
        return False
    node_lower = node_id.lower()

    # Lambda nodes that CAN receive inbound connections (verified Jan 20, 2026)
    # These should NOT be marked as NAT-blocked
    lambda_direct_nodes = {
        "lambda-gh200-3",
        "lambda-gh200-4",
        "lambda-gh200-8",
        "lambda-gh200-9",
        "lambda-gh200-10",
    }
    if node_lower in lambda_direct_nodes:
        return False

    # Lambda GH200 nodes that ARE NAT-blocked (verified Jan 20, 2026)
    lambda_nat_blocked = {
        "lambda-gh200-1",
        "lambda-gh200-2",
        "lambda-gh200-5",
        "lambda-gh200-11",
        "lambda-gh200-training",
    }
    if node_lower in lambda_nat_blocked:
        return True

    # RunPod nodes are typically NAT-blocked
    if node_lower.startswith("runpod-"):
        return True

    return False

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
# Jan 26, 2026: Increased from 3 to 12 to align with PEER_TIMEOUT/HEARTBEAT_INTERVAL
# Previously 3 failures = 45s caused voter count to fluctuate - voter marked dead at 45s
# while other nodes still saw it alive (PEER_TIMEOUT=180s). Now 12 failures = 180s aligns.
VOTER_DEMOTION_FAILURES = int(os.environ.get("RINGRIFT_P2P_VOTER_DEMOTION_FAILURES", "12") or 12)
VOTER_HEALTH_THRESHOLD = float(os.environ.get("RINGRIFT_P2P_VOTER_HEALTH_THRESHOLD", "0.8") or 0.8)
VOTER_PROMOTION_UPTIME = int(os.environ.get("RINGRIFT_P2P_VOTER_PROMOTION_UPTIME", "3600") or 3600)  # 1 hour

# ============================================
# Leader Health
# ============================================

LEADER_HEALTH_CHECK_INTERVAL = int(os.environ.get("RINGRIFT_P2P_LEADER_HEALTH_CHECK_INTERVAL", "30") or 30)
LEADER_MIN_RESPONSE_RATE = float(os.environ.get("RINGRIFT_P2P_LEADER_MIN_RESPONSE_RATE", "0.5") or 0.5)
LEADER_DEGRADED_STEPDOWN_DELAY = int(os.environ.get("RINGRIFT_P2P_LEADER_DEGRADED_STEPDOWN_DELAY", "60") or 60)

# ============================================
# Stale Leader Alerting (January 2, 2026)
# ============================================
# Alert when leader lease expires without voluntary stepdown.
# This catches leaders that crash without proper shutdown.

# Grace period after lease expiry before emitting LEADER_LEASE_EXPIRED event
# Gives the leader time to step down gracefully or renew the lease
LEADER_LEASE_EXPIRY_GRACE_SECONDS = int(
    os.environ.get("RINGRIFT_P2P_LEADER_LEASE_EXPIRY_GRACE", "30") or 30
)

# ============================================
# Frozen Leader Detection (January 2, 2026)
# ============================================
# Leaders can heartbeat but have a stuck event loop (frozen).
# These settings control the work acceptance probe that detects this condition.
#
# Problem: A leader may respond to /status (heartbeat) but fail to accept new work
# because its event loop is blocked (long-running sync, deadlock, etc.).
# Solution: Probe /admin/ping_work which requires the event loop to be responsive.

# Timeout for work acceptance probe (should be shorter than heartbeat interval)
FROZEN_LEADER_PROBE_TIMEOUT = float(
    os.environ.get("RINGRIFT_P2P_FROZEN_LEADER_PROBE_TIMEOUT", "5.0") or 5.0
)

# How long a leader can fail work acceptance probes before triggering election
# 300s = 5 minutes of unresponsive event loop
FROZEN_LEADER_TIMEOUT = int(
    os.environ.get("RINGRIFT_P2P_FROZEN_LEADER_TIMEOUT", "300") or 300
)

# Consecutive work acceptance failures before declaring leader frozen
# Requires 3 failures to avoid false positives during transient load spikes
FROZEN_LEADER_CONSECUTIVE_FAILURES = int(
    os.environ.get("RINGRIFT_P2P_FROZEN_LEADER_CONSECUTIVE_FAILURES", "3") or 3
)

# Grace period after leader election before probing for frozen state
# New leaders need time to initialize before being probed
FROZEN_LEADER_GRACE_PERIOD = int(
    os.environ.get("RINGRIFT_P2P_FROZEN_LEADER_GRACE_PERIOD", "60") or 60
)

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
# Jan 21, 2026: Reduced selfplay/gauntlet timeouts to catch zombie processes faster
# GPU selfplay typically completes in 5-20 minutes; processes running 45+ min are likely stuck
MAX_SELFPLAY_RUNTIME = int(os.environ.get("RINGRIFT_P2P_MAX_SELFPLAY_RUNTIME", "2700") or 2700)    # 45 minutes (was 2 hours)
MAX_TRAINING_RUNTIME = int(os.environ.get("RINGRIFT_P2P_MAX_TRAINING_RUNTIME", "86400") or 86400)  # 24 hours
MAX_TOURNAMENT_RUNTIME = int(os.environ.get("RINGRIFT_P2P_MAX_TOURNAMENT_RUNTIME", "7200") or 7200)  # 2 hours (was 4 hours)
MAX_GAUNTLET_RUNTIME = int(os.environ.get("RINGRIFT_P2P_MAX_GAUNTLET_RUNTIME", "3600") or 3600)    # 1 hour (was 2 hours)

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
# Tailscale Peer Discovery (December 30, 2025)
# ============================================
# Enable discovery on ALL nodes (not just leader) with adaptive intervals:
# - Bootstrap mode: 60s interval when < MIN_PEERS connected (aggressive)
# - Maintenance mode: 120s interval when >= MIN_PEERS connected (conservative)
# Jitter prevents discovery storms when multiple nodes probe simultaneously.

TAILSCALE_DISCOVERY_BOOTSTRAP_INTERVAL = int(
    os.environ.get("RINGRIFT_P2P_TAILSCALE_DISCOVERY_BOOTSTRAP_INTERVAL", "60") or 60
)
TAILSCALE_DISCOVERY_MAINTENANCE_INTERVAL = int(
    os.environ.get("RINGRIFT_P2P_TAILSCALE_DISCOVERY_MAINTENANCE_INTERVAL", "120") or 120
)
TAILSCALE_DISCOVERY_MIN_PEERS_FOR_MAINTENANCE = int(
    os.environ.get("RINGRIFT_P2P_TAILSCALE_DISCOVERY_MIN_PEERS", "5") or 5
)
# Jitter factor: actual interval = base_interval * (1 ± jitter)
TAILSCALE_DISCOVERY_JITTER = float(
    os.environ.get("RINGRIFT_P2P_TAILSCALE_DISCOVERY_JITTER", "0.1") or 0.1
)

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

# ============================================
# Job Lifecycle Thresholds (Sprint 9, Jan 2, 2026)
# ============================================
# Centralized job management thresholds used by JobReaperLoop, SpawnVerificationLoop,
# JobReassignmentLoop, and other job lifecycle management components.

# Job Stale Thresholds (seconds) - how long before claimed job is considered stale
# GPU jobs fail fast (expensive), CPU jobs can wait longer (cheap)
JOB_STALE_GPU = int(os.environ.get("RINGRIFT_JOB_STALE_GPU", "600") or 600)  # 10 min
JOB_STALE_GPU_GUMBEL = int(os.environ.get("RINGRIFT_JOB_STALE_GPU_GUMBEL", "600") or 600)  # 10 min
JOB_STALE_TRAINING = int(os.environ.get("RINGRIFT_JOB_STALE_TRAINING", "1800") or 1800)  # 30 min
JOB_STALE_CPU = int(os.environ.get("RINGRIFT_JOB_STALE_CPU", "1800") or 1800)  # 30 min
JOB_STALE_EVALUATION = int(os.environ.get("RINGRIFT_JOB_STALE_EVALUATION", "900") or 900)  # 15 min
JOB_STALE_DEFAULT = int(os.environ.get("RINGRIFT_JOB_STALE_DEFAULT", "1800") or 1800)  # 30 min

# Job Stuck Threshold - how long before running job is considered stuck
JOB_STUCK_THRESHOLD = int(os.environ.get("RINGRIFT_JOB_STUCK_THRESHOLD", "7200") or 7200)  # 2 hours

# Spawn Verification Timeout - how long to wait for job to start after spawn
JOB_SPAWN_VERIFY_TIMEOUT = int(os.environ.get("RINGRIFT_JOB_SPAWN_VERIFY_TIMEOUT", "30") or 30)  # 30 seconds

# Spawn Verification Check Interval - how often to check pending spawns
JOB_SPAWN_VERIFY_INTERVAL = int(os.environ.get("RINGRIFT_JOB_SPAWN_VERIFY_INTERVAL", "5") or 5)  # 5 seconds

# Job Orphan Detection Timeout - how long without heartbeat before job is orphaned
JOB_ORPHAN_TIMEOUT = int(os.environ.get("RINGRIFT_JOB_ORPHAN_TIMEOUT", "300") or 300)  # 5 min

# Job Reaper Check Interval - how often to scan for stale/stuck jobs
JOB_REAPER_INTERVAL = int(os.environ.get("RINGRIFT_JOB_REAPER_INTERVAL", "300") or 300)  # 5 min

# Maximum jobs to reap per cycle (prevents thundering herd)
JOB_REAPER_MAX_PER_CYCLE = int(os.environ.get("RINGRIFT_JOB_REAPER_MAX_PER_CYCLE", "10") or 10)

# Queue Depth Limits
WORK_QUEUE_MAX_DEPTH = int(os.environ.get("RINGRIFT_WORK_QUEUE_MAX_DEPTH", "1080") or 1080)  # Max pending items
WORK_QUEUE_TARGET_DEPTH = int(os.environ.get("RINGRIFT_WORK_QUEUE_TARGET_DEPTH", "50") or 50)  # Target depth

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
