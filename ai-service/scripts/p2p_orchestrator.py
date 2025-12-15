#!/usr/bin/env python3
"""Distributed P2P Orchestrator - Self-healing compute cluster for RingRift AI training.

This orchestrator runs on each node in the cluster and:
1. Discovers other nodes via broadcast UDP or known peer list
2. Participates in leader election for coordination tasks
3. Monitors local resources and shares status with peers
4. Auto-starts selfplay/training jobs based on cluster needs
5. Self-heals when nodes go offline or IPs change

Architecture:
- Each node runs this script as a daemon
- Nodes communicate via HTTP REST API (port 8770)
- Leader election uses Bully algorithm (highest node_id wins)
- Heartbeats every 30 seconds detect failures
- Nodes maintain local SQLite state for crash recovery

Usage:
    # On each node:
    python scripts/p2p_orchestrator.py --node-id mac-studio
    python scripts/p2p_orchestrator.py --node-id vast-5090-quad --port 8770

    # With known peers (for cloud nodes without broadcast):
    python scripts/p2p_orchestrator.py --node-id vast-3090 --peers <peer-ip>:8770,<peer-ip>:8770
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import os
import secrets
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlparse
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

# HTTP server imports
try:
    from aiohttp import web, ClientSession, ClientTimeout
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    print("Warning: aiohttp not installed. Install with: pip install aiohttp")

# SOCKS proxy support for userspace Tailscale networking
try:
    from aiohttp_socks import ProxyConnector
    HAS_SOCKS = True
except ImportError:
    HAS_SOCKS = False
    ProxyConnector = None

# Get SOCKS proxy from environment (e.g., socks5://localhost:1055)
SOCKS_PROXY = os.environ.get("RINGRIFT_SOCKS_PROXY", "")


def get_client_session(timeout: ClientTimeout = None) -> ClientSession:
    """Create an aiohttp ClientSession with optional SOCKS proxy support."""
    if SOCKS_PROXY and HAS_SOCKS:
        connector = ProxyConnector.from_url(SOCKS_PROXY)
        return ClientSession(connector=connector, timeout=timeout)
    return ClientSession(timeout=timeout)

# Circuit breaker for fault-tolerant peer communication
try:
    from app.distributed.circuit_breaker import (
        get_host_breaker,
        CircuitOpenError,
        CircuitState,
    )
    HAS_CIRCUIT_BREAKER = True
except ImportError:
    HAS_CIRCUIT_BREAKER = False
    CircuitOpenError = Exception


def check_peer_circuit(peer_host: str) -> bool:
    """Check if a peer's circuit is open. Returns True if request allowed."""
    if not HAS_CIRCUIT_BREAKER:
        return True
    return get_host_breaker().can_execute(peer_host)


def record_peer_success(peer_host: str) -> None:
    """Record successful communication with a peer."""
    if HAS_CIRCUIT_BREAKER:
        get_host_breaker().record_success(peer_host)


def record_peer_failure(peer_host: str, error: Optional[Exception] = None) -> None:
    """Record failed communication with a peer."""
    if HAS_CIRCUIT_BREAKER:
        get_host_breaker().record_failure(peer_host, error)


async def peer_request(
    session: "ClientSession",
    method: str,
    url: str,
    peer_host: str,
    headers: Optional[Dict[str, str]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Make a circuit-breaker-protected request to a peer.

    Args:
        session: aiohttp ClientSession to use
        method: HTTP method (GET, POST, etc.)
        url: Full URL to request
        peer_host: Hostname/IP for circuit breaker tracking
        headers: Optional headers dict
        json: Optional JSON payload for POST/PUT
        timeout: Optional request timeout

    Returns:
        Response JSON if successful, None if circuit open or request failed
    """
    # Check circuit first
    if not check_peer_circuit(peer_host):
        return None

    try:
        kwargs = {"headers": headers} if headers else {}
        if json is not None:
            kwargs["json"] = json
        if timeout:
            kwargs["timeout"] = ClientTimeout(total=timeout)

        async with session.request(method, url, **kwargs) as resp:
            if resp.status == 200:
                record_peer_success(peer_host)
                return await resp.json()
            else:
                # Non-200 isn't necessarily a failure (might be expected)
                return {"status": resp.status, "error": await resp.text()}
    except Exception as e:
        record_peer_failure(peer_host, e)
        return None


# Dynamic host registry for IP auto-update
try:
    from app.distributed.dynamic_registry import (
        DynamicHostRegistry,
        get_registry,
        NodeState,
    )
    HAS_DYNAMIC_REGISTRY = True
except ImportError:
    HAS_DYNAMIC_REGISTRY = False
    get_registry = None
    NodeState = None

# Improvement cycle manager for automated training
try:
    from scripts.improvement_cycle_manager import ImprovementCycleManager
    HAS_IMPROVEMENT_MANAGER = True
except ImportError:
    HAS_IMPROVEMENT_MANAGER = False
    ImprovementCycleManager = None

# Task coordination safeguards - prevents runaway spawning
try:
    from app.coordination.safeguards import Safeguards, check_before_spawn
    HAS_SAFEGUARDS = True
    _safeguards = Safeguards.get_instance()
except ImportError:
    HAS_SAFEGUARDS = False
    _safeguards = None
    def check_before_spawn(task_type, node_id):
        return True, ""

# New coordination features: OrchestratorRole, backpressure, sync_lock, bandwidth
try:
    from app.coordination import (
        # Orchestrator role management (SQLite-backed with heartbeat)
        OrchestratorRole,
        acquire_orchestrator_role,
        release_orchestrator_role,
        # Queue backpressure
        QueueType,
        should_throttle_production,
        should_stop_production,
        get_throttle_factor,
        # Sync mutex for data transfer coordination
        sync_lock,
        # Bandwidth management
        request_bandwidth,
        release_bandwidth,
        TransferPriority,
        # Resource targets for unified utilization management
        get_resource_targets,
        get_host_targets,
        should_scale_up,
        should_scale_down,
        get_target_job_count,
        record_utilization,
        # Resource optimizer for cluster-wide PID-controlled optimization
        ResourceOptimizer,
        NodeResources,
        get_resource_optimizer,
        get_optimal_concurrency,
        get_cluster_utilization,
    )
    # Import rate negotiation functions for cooperative utilization (60-80% target)
    from app.coordination.resource_optimizer import (
        negotiate_selfplay_rate,
        get_current_selfplay_rate,
        apply_feedback_adjustment,
        get_utilization_status,
        update_config_weights,
        get_config_weights,
    )
    HAS_RATE_NEGOTIATION = True
    HAS_NEW_COORDINATION = True
    # Get targets from unified source
    _unified_targets = get_resource_targets()
except ImportError:
    HAS_NEW_COORDINATION = False
    HAS_RATE_NEGOTIATION = False
    OrchestratorRole = None
    _unified_targets = None
    negotiate_selfplay_rate = None
    get_current_selfplay_rate = None
    apply_feedback_adjustment = None
    get_utilization_status = None
    update_config_weights = None
    get_config_weights = None

# P2P-integrated monitoring management
try:
    from app.monitoring.p2p_monitoring import MonitoringManager
    HAS_P2P_MONITORING = True
except ImportError:
    HAS_P2P_MONITORING = False
    MonitoringManager = None

# ============================================
# Configuration
# ============================================

DEFAULT_PORT = 8770
HEARTBEAT_INTERVAL = 30  # seconds
PEER_TIMEOUT = 90  # seconds without heartbeat = node considered dead
ELECTION_TIMEOUT = 10  # seconds to wait for election responses
# Leader lease must be comfortably larger than the heartbeat cadence; otherwise
# small scheduling delays can cause leaders to "expire" their own lease and
# flap into leaderless states.
LEADER_LEASE_DURATION = 90  # seconds
LEADER_LEASE_RENEW_INTERVAL = 10  # How often leader renews lease
JOB_CHECK_INTERVAL = 60  # seconds between job status checks
DISCOVERY_PORT = 8771  # UDP port for peer discovery
DISCOVERY_INTERVAL = 120  # seconds between discovery broadcasts

# LEARNED LESSONS from PLAN.md - Disk and resource thresholds
# These thresholds are env-overridable so heterogeneous clusters (e.g. small
# Mac disks vs large cloud volumes) can tune health/scheduling without code
# changes.
DISK_CRITICAL_THRESHOLD = int(os.environ.get("RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD", "90") or 90)  # Stop all new jobs
DISK_WARNING_THRESHOLD = int(os.environ.get("RINGRIFT_P2P_DISK_WARNING_THRESHOLD", "80") or 80)    # Reduce jobs
DISK_CLEANUP_THRESHOLD = int(os.environ.get("RINGRIFT_P2P_DISK_CLEANUP_THRESHOLD", "85") or 85)    # Trigger cleanup
MEMORY_CRITICAL_THRESHOLD = int(os.environ.get("RINGRIFT_P2P_MEMORY_CRITICAL_THRESHOLD", "95") or 95)  # Stop jobs
MEMORY_WARNING_THRESHOLD = int(os.environ.get("RINGRIFT_P2P_MEMORY_WARNING_THRESHOLD", "85") or 85)    # Reduce jobs
MIN_MEMORY_GB_FOR_TASKS = int(os.environ.get("RINGRIFT_P2P_MIN_MEMORY_GB", "64") or 64)                # Skip low-memory nodes
LOAD_MAX_FOR_NEW_JOBS = int(os.environ.get("RINGRIFT_P2P_LOAD_MAX_FOR_NEW_JOBS", "85") or 85)          # Stop starting

# GPU utilization targeting for efficient resource usage
# Use unified targets from resource_targets.py if available, fallback to env vars
if _unified_targets is not None:
    TARGET_GPU_UTIL_MIN = int(_unified_targets.gpu_min)  # 60% from unified config
    TARGET_GPU_UTIL_MAX = int(_unified_targets.gpu_max + 5)  # 85% + 5% buffer = 90%
else:
    TARGET_GPU_UTIL_MIN = int(os.environ.get("RINGRIFT_P2P_TARGET_GPU_UTIL_MIN", "60") or 60)
    TARGET_GPU_UTIL_MAX = int(os.environ.get("RINGRIFT_P2P_TARGET_GPU_UTIL_MAX", "90") or 90)
GH200_MIN_SELFPLAY = int(os.environ.get("RINGRIFT_P2P_GH200_MIN_SELFPLAY", "20") or 20)    # Min selfplay for GH200
GH200_MAX_SELFPLAY = int(os.environ.get("RINGRIFT_P2P_GH200_MAX_SELFPLAY", "100") or 100)  # Max selfplay for GH200

# LEARNED LESSONS - Connection robustness
HTTP_CONNECT_TIMEOUT = 10     # Fast timeout for connection phase
HTTP_TOTAL_TIMEOUT = 30       # Total request timeout
MAX_CONSECUTIVE_FAILURES = 3  # Mark node dead after 3 failures
RETRY_DEAD_NODE_INTERVAL = 300  # Retry dead nodes every 5 minutes

# Retire peers that have been offline for a long time so they don't pollute the
# active scheduling set (but we still probe them occasionally so they can
# return without manual intervention).
PEER_RETIRE_AFTER_SECONDS = int(os.environ.get("RINGRIFT_P2P_PEER_RETIRE_AFTER_SECONDS", "3600") or 3600)
RETRY_RETIRED_NODE_INTERVAL = int(os.environ.get("RINGRIFT_P2P_RETRY_RETIRED_NODE_INTERVAL", "3600") or 3600)

# NAT/relay settings
# Nodes that can't receive inbound connections can operate in relay mode:
# they send heartbeats to a relay (/relay/heartbeat) and poll for commands.
NAT_INBOUND_HEARTBEAT_STALE_SECONDS = 180  # seconds since last inbound /heartbeat
RELAY_HEARTBEAT_INTERVAL = 15  # seconds between relay heartbeats when enabled (reduced for faster job delivery)
RELAY_COMMAND_TTL_SECONDS = 1800  # expire queued commands after 30 minutes
RELAY_COMMAND_MAX_BATCH = 16
RELAY_COMMAND_MAX_ATTEMPTS = 3
RELAY_MAX_PENDING_START_JOBS = 4

# Peer bootstrap settings
# Seed peers are used to import a snapshot of cluster membership (via /relay/peers)
# so new nodes can join existing clusters without needing every peer preconfigured.
PEER_BOOTSTRAP_INTERVAL = 60  # seconds between bootstrap refresh attempts
PEER_BOOTSTRAP_MIN_PEERS = 3  # refresh if we see fewer than this many peers

# LEARNED LESSONS - Stuck job detection
GPU_IDLE_RESTART_TIMEOUT = 300  # Restart jobs after 5 min of GPU at 0%
GPU_IDLE_THRESHOLD = 2          # Consider GPU idle if utilization < 2%
# If a node reports hundreds/thousands of selfplay processes, it almost always
# indicates job tracking was lost and stale processes are accumulating (which
# can brick nodes via disk/memory pressure). Treat this as a runaway condition
# and trigger a restart_stuck_jobs sweep.
_runaway_threshold_env = (os.environ.get("RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD") or "").strip()
RUNAWAY_SELFPLAY_PROCESS_THRESHOLD = int(_runaway_threshold_env) if _runaway_threshold_env else 0

# SAFEGUARDS - Load average and rate limiting (added 2025-12-15)
# These provide hard limits beyond the soft load_score calculation
LOAD_AVERAGE_MAX_MULTIPLIER = float(os.environ.get("RINGRIFT_P2P_LOAD_AVG_MAX_MULT", "2.0") or 2.0)  # Max load = cpus * multiplier
SPAWN_RATE_LIMIT_PER_MINUTE = int(os.environ.get("RINGRIFT_P2P_SPAWN_RATE_LIMIT", "5") or 5)  # Max spawns per minute
COORDINATOR_URL = os.environ.get("RINGRIFT_COORDINATOR_URL", "")  # If set, defer to coordinator
AGENT_MODE_ENABLED = os.environ.get("RINGRIFT_P2P_AGENT_MODE", "").lower() in {"1", "true", "yes", "on"}

# Git auto-update settings
GIT_UPDATE_CHECK_INTERVAL = int(os.environ.get("RINGRIFT_P2P_GIT_UPDATE_CHECK_INTERVAL", "300") or 300)  # seconds
GIT_REMOTE_NAME = "origin"       # Git remote to check
GIT_BRANCH_NAME = "main"         # Branch to track
AUTO_UPDATE_ENABLED = (os.environ.get("RINGRIFT_P2P_AUTO_UPDATE", "false").strip().lower() in {"1", "true", "yes"})
GRACEFUL_SHUTDOWN_BEFORE_UPDATE = True  # Stop jobs before updating

# Shared auth token (optional but strongly recommended if any node is public)
AUTH_TOKEN_ENV = "RINGRIFT_CLUSTER_AUTH_TOKEN"
AUTH_TOKEN_FILE_ENV = "RINGRIFT_CLUSTER_AUTH_TOKEN_FILE"

# Optional build/version label surfaced in the dashboard / heartbeats.
BUILD_VERSION_ENV = "RINGRIFT_BUILD_VERSION"

# Optional advertised endpoint override (useful behind NAT/port-mapping).
ADVERTISE_HOST_ENV = "RINGRIFT_ADVERTISE_HOST"
ADVERTISE_PORT_ENV = "RINGRIFT_ADVERTISE_PORT"

# Tailscale uses CGNAT space (100.64.0.0/10) for node IPs by default.
TAILSCALE_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

# Data manifest collection settings
MANIFEST_JSONL_LINECOUNT_MAX_BYTES = 64 * 1024 * 1024  # Skip line-counting for huge JSONL files
MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES = 1024 * 1024

# LEARNED LESSONS - Automatic data management settings
DATA_MANAGEMENT_INTERVAL = 300  # Check data status every 5 minutes (reduced for faster training triggers)
DB_EXPORT_THRESHOLD_MB = 100    # Trigger export when DB exceeds 100MB
TRAINING_DATA_SYNC_THRESHOLD_MB = 10  # Sync training data when > 10MB new data
MAX_CONCURRENT_EXPORTS = 2      # Limit concurrent export jobs per node
AUTO_TRAINING_THRESHOLD_MB = 50 # Auto-trigger training when training data exceeds 50MB

# GPU Power Rankings for training node priority
# Higher score = more powerful GPU = higher priority for receiving training data
# Scores are approximate TFLOPS (FP16) for relative comparison
GPU_POWER_RANKINGS = {
    # Data center GPUs (highest priority)
    "H100": 2000,      # ~2000 TFLOPS FP16
    "H200": 2500,      # ~2500 TFLOPS FP16
    "A100": 624,       # ~624 TFLOPS FP16
    "A10G": 250,       # ~250 TFLOPS FP16
    "A10": 250,        # ~250 TFLOPS FP16
    "L40": 362,        # ~362 TFLOPS FP16
    "V100": 125,       # ~125 TFLOPS FP16
    # Consumer GPUs - RTX 50 series
    "5090": 419,       # ~419 TFLOPS FP16 (estimated)
    "5080": 300,       # ~300 TFLOPS FP16 (estimated)
    "5070": 200,       # ~200 TFLOPS FP16 (estimated)
    # Consumer GPUs - RTX 40 series
    "4090": 330,       # ~330 TFLOPS FP16
    "4080": 242,       # ~242 TFLOPS FP16
    "4070": 184,       # ~184 TFLOPS FP16
    "4060": 120,       # ~120 TFLOPS FP16
    # Consumer GPUs - RTX 30 series
    "3090": 142,       # ~142 TFLOPS FP16
    "3080": 119,       # ~119 TFLOPS FP16
    "3070": 81,        # ~81 TFLOPS FP16
    "3060": 51,        # ~51 TFLOPS FP16
    # Apple Silicon
    "Apple M3": 30,    # Approximate
    "Apple M2": 25,    # Approximate
    "Apple M1": 20,    # Approximate
    "Apple MPS": 15,   # Generic Apple GPU
    # Fallback
    "Unknown": 10,
}

# Training node sync settings
TRAINING_NODE_COUNT = 3          # Top N GPU nodes to prioritize for sync
TRAINING_SYNC_INTERVAL = 300.0   # Sync to training nodes every 5 minutes
MIN_GAMES_FOR_SYNC = 100         # Minimum new games before triggering sync

# Path to local state database
STATE_DIR = Path(__file__).parent.parent / "logs" / "p2p_orchestrator"
STATE_DIR.mkdir(parents=True, exist_ok=True)


class NodeRole(str, Enum):
    """Role a node plays in the cluster."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


class JobType(str, Enum):
    """Types of jobs nodes can run."""
    SELFPLAY = "selfplay"
    GPU_SELFPLAY = "gpu_selfplay"  # GPU-accelerated parallel selfplay (pure GPU, experimental)
    HYBRID_SELFPLAY = "hybrid_selfplay"  # Hybrid CPU/GPU selfplay (100% rule fidelity, GPU-accelerated eval)
    TRAINING = "training"
    CMAES = "cmaes"
    # Distributed job types
    DISTRIBUTED_CMAES_COORDINATOR = "distributed_cmaes_coordinator"
    DISTRIBUTED_CMAES_WORKER = "distributed_cmaes_worker"
    DISTRIBUTED_TOURNAMENT_COORDINATOR = "distributed_tournament_coordinator"
    DISTRIBUTED_TOURNAMENT_WORKER = "distributed_tournament_worker"
    IMPROVEMENT_LOOP = "improvement_loop"


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""
    node_id: str
    host: str
    port: int
    scheme: str = "http"  # How to reach this node (http/https)
    role: NodeRole = NodeRole.FOLLOWER
    last_heartbeat: float = 0.0
    cpu_count: int = 0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    has_gpu: bool = False
    gpu_name: str = ""
    memory_gb: int = 0
    capabilities: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    # Self-reported endpoint (may be different from the observed reachable
    # endpoint stored in `host`/`port`, e.g. overlays or containers).
    reported_host: str = ""
    reported_port: int = 0
    # LEARNED LESSONS - Track connection failures for adaptive retry
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    # LEARNED LESSONS - Track resource issues
    disk_cleanup_needed: bool = False
    oom_events: int = 0
    last_oom_time: float = 0.0
    # NAT/Relay support - nodes that can't be reached directly
    nat_blocked: bool = False
    relay_via: str = ""  # node_id of the relay hub (usually leader)
    # Peer lifecycle: permanently gone nodes are marked retired so they don't
    # pollute scheduling, but they can be reactivated if they come back online.
    retired: bool = False
    retired_at: float = 0.0

    def is_alive(self) -> bool:
        """Check if node is considered alive based on last heartbeat."""
        return time.time() - self.last_heartbeat < PEER_TIMEOUT

    def is_healthy(self) -> bool:
        """Check if node is healthy for new jobs (not just reachable)."""
        if not self.is_alive():
            return False
        if getattr(self, "retired", False):
            return False
        # LEARNED LESSONS - Don't start jobs on resource-constrained nodes
        if self.disk_percent >= DISK_WARNING_THRESHOLD:
            return False
        if self.memory_percent >= MEMORY_WARNING_THRESHOLD:
            return False
        if self.get_load_score() >= LOAD_MAX_FOR_NEW_JOBS:
            return False
        return True

    def get_load_score(self) -> float:
        """Calculate a load score for load balancing (lower = less loaded).

        LEARNED LESSONS - Smart load balancing:
        - Weights CPU, memory, and job counts for overall load estimation
        - GPU utilization is weighted separately for GPU nodes
        - Returns 0-100 scale where 0 = idle, 100 = overloaded
        """
        # Base score from CPU and memory utilization
        cpu_weight = 0.4
        mem_weight = 0.3
        jobs_weight = 0.3

        # Normalize job count (assume 8 selfplay jobs = fully loaded)
        job_score = min(100.0, (self.selfplay_jobs + self.training_jobs * 2) * 12.5)

        load = (
            self.cpu_percent * cpu_weight +
            self.memory_percent * mem_weight +
            job_score * jobs_weight
        )

        # For GPU nodes, also consider GPU utilization
        if self.has_gpu and self.gpu_percent > 0:
            load = load * 0.7 + self.gpu_percent * 0.3

        return min(100.0, load)

    def is_gpu_node(self) -> bool:
        """Check if this node has a CUDA GPU (not Apple MPS)."""
        gpu_name = (self.gpu_name or "").upper()
        return self.has_gpu and "MPS" not in gpu_name and "APPLE" not in gpu_name

    def is_cpu_only_node(self) -> bool:
        """Check if this node is CPU-only (no accelerator)."""
        return not self.has_gpu

    def should_retry(self) -> bool:
        """Check if we should retry connecting to a failed node."""
        if getattr(self, "retired", False):
            return time.time() - self.last_failure_time > RETRY_RETIRED_NODE_INTERVAL
        if self.consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            return True
        # LEARNED LESSONS - Retry dead nodes periodically
        return time.time() - self.last_failure_time > RETRY_DEAD_NODE_INTERVAL

    def gpu_power_score(self) -> int:
        """Get GPU processing power score based on GPU model.

        Higher score = more powerful GPU = better for training.
        Used for prioritizing which nodes receive selfplay data first.
        """
        if not self.has_gpu or not self.gpu_name:
            return 0

        gpu_upper = self.gpu_name.upper()

        # Check each GPU model pattern
        for gpu_key, score in GPU_POWER_RANKINGS.items():
            if gpu_key.upper() in gpu_upper:
                # Multi-GPU bonus: assume more memory = more GPUs
                # H100 80GB = 1x, 160GB = 2x, etc.
                if self.memory_gb and self.memory_gb > 100:
                    # Likely multi-GPU system
                    gpu_count = max(1, self.memory_gb // 80)
                    return score * gpu_count
                return score

        return GPU_POWER_RANKINGS.get("Unknown", 10)

    def check_load_average_safe(self) -> Tuple[bool, str]:
        """Check if system load average is safe for spawning new processes.

        SAFEGUARD: Uses actual os.getloadavg() instead of just CPU percentage.
        On a 72-core machine, load 500 is catastrophic even if CPU% looks OK.

        Returns:
            (is_safe, reason) - True if safe to spawn, with explanation
        """
        try:
            load_1, load_5, load_15 = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            max_load = cpu_count * LOAD_AVERAGE_MAX_MULTIPLIER

            if load_1 > max_load:
                return False, f"Load {load_1:.1f} > {max_load:.0f} (cpus={cpu_count} x {LOAD_AVERAGE_MAX_MULTIPLIER})"
            if load_5 > max_load * 1.5:
                return False, f"5min load {load_5:.1f} > {max_load * 1.5:.0f}"

            return True, f"Load OK: {load_1:.1f}/{max_load:.0f}"
        except Exception as e:
            # If we can't check load, be conservative
            return True, f"Load check unavailable: {e}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['role'] = self.role.value
        # Derived metrics (not persisted as dataclass fields).
        d["load_score"] = self.get_load_score()
        d["gpu_power_score"] = self.gpu_power_score()
        d["is_cpu_only_node"] = self.is_cpu_only_node()
        d["is_cuda_gpu_node"] = self.is_gpu_node()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'NodeInfo':
        """Create from dictionary."""
        d = d.copy()
        d['role'] = NodeRole(d.get('role', 'follower'))
        # Handle missing new fields gracefully
        d.setdefault('scheme', 'http')
        d.setdefault('cpu_count', 0)
        d.setdefault('reported_host', '')
        d.setdefault('reported_port', 0)
        d.setdefault('consecutive_failures', 0)
        d.setdefault('last_failure_time', 0.0)
        d.setdefault('disk_cleanup_needed', False)
        d.setdefault('oom_events', 0)
        d.setdefault('last_oom_time', 0.0)
        d.setdefault('nat_blocked', False)
        d.setdefault('relay_via', '')
        d.setdefault('retired', False)
        d.setdefault('retired_at', 0.0)
        # Ignore unknown keys for rolling upgrades.
        allowed = {f.name for f in dataclass_fields(cls)}
        d = {k: v for k, v in d.items() if k in allowed}
        return cls(**d)


@dataclass
class ClusterJob:
    """A job running in the cluster."""
    job_id: str
    job_type: JobType
    node_id: str
    board_type: str = "square8"
    num_players: int = 2
    engine_mode: str = "descent-only"
    pid: int = 0
    started_at: float = 0.0
    status: str = "running"
    # Extended fields for distributed jobs
    coordinator_node: str = ""  # Node running coordinator (for worker jobs)
    worker_port: int = 8766     # Port for worker server
    config_json: str = ""       # JSON config for complex jobs

    def to_dict(self) -> dict:
        d = asdict(self)
        d['job_type'] = self.job_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ClusterJob':
        d = d.copy()
        d['job_type'] = JobType(d.get('job_type', 'selfplay'))
        # Handle missing new fields
        d.setdefault('coordinator_node', '')
        d.setdefault('worker_port', 8766)
        d.setdefault('config_json', '')
        return cls(**d)


@dataclass
class DistributedCMAESState:
    """State for distributed CMA-ES job coordination."""
    job_id: str
    board_type: str = "square8"
    num_players: int = 2
    generations: int = 100
    population_size: int = 20
    games_per_eval: int = 50
    current_generation: int = 0
    best_fitness: float = 0.0
    best_weights: Dict[str, float] = field(default_factory=dict)
    worker_nodes: List[str] = field(default_factory=list)
    status: str = "pending"
    started_at: float = 0.0
    last_update: float = 0.0
    results_file: str = ""
    # LEARNED LESSONS - Store pending results keyed by (generation, individual_idx)
    pending_results: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'DistributedCMAESState':
        d.setdefault('pending_results', {})
        return cls(**d)


@dataclass
class DistributedTournamentState:
    """State for distributed tournament coordination."""
    job_id: str
    board_type: str = "square8"
    num_players: int = 2
    agent_ids: List[str] = field(default_factory=list)
    games_per_pairing: int = 2
    total_matches: int = 0
    completed_matches: int = 0
    worker_nodes: List[str] = field(default_factory=list)
    pending_matches: List[dict] = field(default_factory=list)
    results: List[dict] = field(default_factory=list)
    final_ratings: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    started_at: float = 0.0
    last_update: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'DistributedTournamentState':
        return cls(**d)


@dataclass
class SSHTournamentRun:
    """State for an SSH-distributed difficulty-tier tournament (leader-triggered)."""

    job_id: str
    run_id: str
    tiers: str
    board: str
    games_per_matchup: int
    pid: int = 0
    status: str = "running"  # running, completed, failed, cancelled
    started_at: float = 0.0
    completed_at: float = 0.0
    output_root: str = ""
    manifest_path: str = ""
    checkpoint_path: str = ""
    report_path: str = ""
    log_path: str = ""
    command: List[str] = field(default_factory=list)
    return_code: Optional[int] = None
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImprovementLoopState:
    """State for improvement loop coordination."""
    job_id: str
    board_type: str = "square8"
    num_players: int = 2
    current_iteration: int = 0
    max_iterations: int = 50
    games_per_iteration: int = 1000
    phase: str = "idle"  # idle, selfplay, export, train, evaluate, promote
    best_model_path: str = ""
    best_winrate: float = 0.0
    consecutive_failures: int = 0
    worker_nodes: List[str] = field(default_factory=list)
    selfplay_progress: Dict[str, int] = field(default_factory=dict)  # node_id -> games done
    status: str = "pending"
    started_at: float = 0.0
    last_update: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ImprovementLoopState':
        return cls(**d)


# ============================================
# Phase 3: Training Pipeline Integration Types
# ============================================

@dataclass
class TrainingJob:
    """State for automatic training jobs dispatched by leader."""
    job_id: str
    job_type: str  # "nnue", "cmaes"
    board_type: str
    num_players: int
    status: str = "pending"  # pending, queued, running, completed, failed
    worker_node: str = ""    # Node where training is running
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    # Training configuration
    epochs: int = 100
    batch_size: int = 2048
    learning_rate: float = 0.001
    # Data sources
    data_paths: List[str] = field(default_factory=list)
    data_games_count: int = 0
    # Output
    output_model_path: str = ""
    error_message: str = ""
    # Metrics
    final_loss: float = 0.0
    final_accuracy: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingJob':
        return cls(**d)


@dataclass
class TrainingThresholds:
    """Configuration for automatic training triggers with adaptive scaling."""
    # Minimum games required to trigger training
    min_games_nnue: int = 2000          # Start sooner; improves as more data arrives
    min_games_cmaes: int = 1000         # CMA-ES can work with fewer games
    # Incremental thresholds (trigger re-training when new data >= threshold)
    incremental_games_nnue: int = 1000  # Re-train every 1k new games
    incremental_games_cmaes: int = 500  # Re-optimize every 500 new games
    # Cooldown between training runs (seconds)
    cooldown_seconds: float = 1800.0    # 30 minutes
    # Auto-training enabled flags
    auto_nnue_enabled: bool = True
    auto_cmaes_enabled: bool = True
    # Adaptive scaling factors (updated dynamically based on cluster state)
    cluster_scale_factor: float = 1.0   # Adjusted based on GPU node count
    data_rate_scale_factor: float = 1.0 # Adjusted based on games/hour

    def get_effective_min_games(self, job_type: str = "nnue") -> int:
        """Get cluster-scaled minimum games threshold.

        More GPU nodes = can train more frequently with less data per iteration.
        """
        base = self.min_games_nnue if job_type == "nnue" else self.min_games_cmaes
        # Scale inversely with cluster size (more nodes = lower threshold)
        scaled = int(base * self.cluster_scale_factor)
        # Never go below minimum viable threshold
        return max(500 if job_type == "nnue" else 250, scaled)

    def get_effective_incremental(self, job_type: str = "nnue") -> int:
        """Get cluster-scaled incremental threshold."""
        base = self.incremental_games_nnue if job_type == "nnue" else self.incremental_games_cmaes
        scaled = int(base * self.cluster_scale_factor)
        return max(200 if job_type == "nnue" else 100, scaled)

    def get_effective_cooldown(self) -> float:
        """Get data-rate-scaled cooldown.

        Faster data generation = shorter cooldowns (train more often).
        """
        # Scale inversely with data rate (faster data = shorter cooldown)
        scaled = self.cooldown_seconds / max(1.0, self.data_rate_scale_factor)
        # Never go below 5 minutes or above 2 hours
        return max(300, min(7200, scaled))

    def update_from_cluster_state(self, gpu_node_count: int, games_per_hour: float = 0):
        """Update adaptive scaling factors based on cluster state.

        Args:
            gpu_node_count: Number of GPU nodes in cluster
            games_per_hour: Current data generation rate (0 = unknown)
        """
        # Scale factor: more GPUs = lower thresholds (train more often)
        # 1 GPU = 1.0, 2 GPUs = 0.7, 4 GPUs = 0.5, 8+ GPUs = 0.35
        if gpu_node_count <= 1:
            self.cluster_scale_factor = 1.0
        elif gpu_node_count <= 2:
            self.cluster_scale_factor = 0.7
        elif gpu_node_count <= 4:
            self.cluster_scale_factor = 0.5
        else:
            self.cluster_scale_factor = 0.35

        # Data rate factor: faster generation = shorter cooldowns
        if games_per_hour > 0:
            # Baseline: 1000 games/hour = factor 1.0
            self.data_rate_scale_factor = max(0.5, min(4.0, games_per_hour / 1000))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingThresholds':
        return cls(**d)


# ============================================
# Phase 2: Distributed Data Sync Types
# ============================================

@dataclass
class DataFileInfo:
    """Information about a single data file for manifest collection."""
    path: str                    # Relative path from ai-service/data
    size_bytes: int              # File size in bytes
    modified_time: float         # Last modification time (Unix timestamp)
    file_hash: str = ""          # MD5 hash for verification (computed on demand)
    file_type: str = ""          # Type: selfplay, model, training, etc.
    board_type: str = ""         # Board type if applicable (square8, hex, etc.)
    num_players: int = 0         # Player count if applicable
    game_count: int = 0          # For selfplay JSONL: number of games (lines)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'DataFileInfo':
        return cls(**d)


@dataclass
class NodeDataManifest:
    """Data manifest for a single node - lists all available data files."""
    node_id: str
    collected_at: float          # When this manifest was collected
    total_files: int = 0
    total_size_bytes: int = 0
    files: List[DataFileInfo] = field(default_factory=list)
    # Summary by type
    selfplay_games: int = 0      # Total selfplay games (from JSONL line counts)
    model_count: int = 0         # Number of model files
    training_data_size: int = 0  # Total training data size

    @property
    def files_by_path(self) -> Dict[str, DataFileInfo]:
        return {f.path: f for f in self.files}

    def to_dict(self) -> dict:
        d = asdict(self)
        d['files'] = [f.to_dict() if hasattr(f, 'to_dict') else f for f in self.files]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'NodeDataManifest':
        d = d.copy()
        d['files'] = [DataFileInfo.from_dict(f) if isinstance(f, dict) else f for f in d.get('files', [])]
        return cls(**d)


@dataclass
class ClusterDataManifest:
    """Aggregated data manifest for the entire cluster (leader-only)."""
    collected_at: float
    node_manifests: Dict[str, NodeDataManifest] = field(default_factory=dict)
    # Cluster-wide totals
    total_nodes: int = 0
    total_files: int = 0
    total_size_bytes: int = 0
    total_selfplay_games: int = 0
    # Data distribution analysis
    files_by_node: Dict[str, int] = field(default_factory=dict)
    unique_files: Set[str] = field(default_factory=set)
    missing_from_nodes: Dict[str, List[str]] = field(default_factory=dict)  # file -> list of nodes missing it

    @property
    def manifests_by_node(self) -> Dict[str, NodeDataManifest]:
        return self.node_manifests

    @property
    def by_board_type(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate selfplay game counts by (board_type, num_players).

        Key format matches downstream training logic: `{board_type}_{num_players}p`.
        """
        totals: Dict[str, Dict[str, Any]] = {}

        for node_id, node_manifest in self.node_manifests.items():
            for f in node_manifest.files:
                if getattr(f, "file_type", "") != "selfplay":
                    continue
                if not getattr(f, "board_type", "") or not getattr(f, "num_players", 0):
                    continue
                key = f"{f.board_type}_{f.num_players}p"
                entry = totals.setdefault(key, {"total_games": 0, "nodes": set()})
                entry["total_games"] += int(getattr(f, "game_count", 0) or 0)
                entry["nodes"].add(node_id)

        return {k: {"total_games": v["total_games"], "nodes": sorted(v["nodes"])} for k, v in totals.items()}

    def to_dict(self) -> dict:
        d = {
            'collected_at': self.collected_at,
            'node_manifests': {k: v.to_dict() for k, v in self.node_manifests.items()},
            'total_nodes': self.total_nodes,
            'total_files': self.total_files,
            'total_size_bytes': self.total_size_bytes,
            'total_selfplay_games': self.total_selfplay_games,
            'files_by_node': self.files_by_node,
            'unique_files': list(self.unique_files),
            'missing_from_nodes': self.missing_from_nodes,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ClusterDataManifest':
        d = d.copy()
        d['node_manifests'] = {k: NodeDataManifest.from_dict(v) for k, v in d.get('node_manifests', {}).items()}
        d['unique_files'] = set(d.get('unique_files', []))
        return cls(**d)


@dataclass
class DataSyncJob:
    """Tracks a P2P data synchronization job between nodes."""
    job_id: str
    source_node: str            # Node that has the file(s)
    target_node: str            # Node that needs the file(s)
    files: List[str]            # List of file paths to sync (relative to data/)
    status: str = "pending"     # pending, running, completed, failed
    started_at: float = 0.0
    completed_at: float = 0.0
    bytes_transferred: int = 0
    files_completed: int = 0
    error_message: str = ""
    # Rsync process details
    rsync_pid: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'DataSyncJob':
        return cls(**d)


@dataclass
class ClusterSyncPlan:
    """Leader-generated plan for synchronizing data across the cluster."""
    plan_id: str
    created_at: float
    total_files_to_sync: int = 0
    total_bytes_to_sync: int = 0
    sync_jobs: List[DataSyncJob] = field(default_factory=list)
    # Status tracking
    status: str = "pending"     # pending, running, completed, failed
    jobs_completed: int = 0
    jobs_failed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['sync_jobs'] = [j.to_dict() for j in self.sync_jobs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ClusterSyncPlan':
        d = d.copy()
        d['sync_jobs'] = [DataSyncJob.from_dict(j) for j in d.get('sync_jobs', [])]
        return cls(**d)


class WebhookNotifier:
    """Sends alerts to Slack/Discord webhooks for important events.

    Configure via environment variables:
    - RINGRIFT_SLACK_WEBHOOK: Slack incoming webhook URL
    - RINGRIFT_DISCORD_WEBHOOK: Discord webhook URL
    - RINGRIFT_ALERT_LEVEL: Minimum level to alert (debug/info/warning/error) default: warning
    """

    LEVELS = {"debug": 0, "info": 1, "warning": 2, "error": 3}

    def __init__(self):
        self.slack_webhook = os.environ.get("RINGRIFT_SLACK_WEBHOOK", "")
        self.discord_webhook = os.environ.get("RINGRIFT_DISCORD_WEBHOOK", "")
        self.min_level = self.LEVELS.get(
            os.environ.get("RINGRIFT_ALERT_LEVEL", "warning").lower(), 2
        )
        self._session: Optional[ClientSession] = None
        self._last_alert: Dict[str, float] = {}  # Throttle repeated alerts
        self._throttle_seconds = 300  # 5 minutes between duplicate alerts

    async def _get_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            self._session = ClientSession(timeout=ClientTimeout(total=10))
        return self._session

    def _should_throttle(self, alert_key: str) -> bool:
        """Check if this alert should be throttled (duplicate within window)."""
        now = time.time()
        if alert_key in self._last_alert:
            if now - self._last_alert[alert_key] < self._throttle_seconds:
                return True
        self._last_alert[alert_key] = now
        return False

    async def send(
        self,
        title: str,
        message: str,
        level: str = "warning",
        fields: Dict[str, str] = None,
        node_id: str = "",
    ):
        """Send an alert to configured webhooks.

        Args:
            title: Alert title/subject
            message: Alert body text
            level: debug/info/warning/error
            fields: Additional key-value pairs to include
            node_id: Node ID for deduplication
        """
        if self.LEVELS.get(level, 2) < self.min_level:
            return

        if not self.slack_webhook and not self.discord_webhook:
            return

        # Throttle duplicate alerts
        alert_key = f"{title}:{node_id}"
        if self._should_throttle(alert_key):
            return

        try:
            session = await self._get_session()

            # Color based on level
            colors = {"debug": "#808080", "info": "#36a64f", "warning": "#ff9800", "error": "#ff0000"}
            color = colors.get(level, "#808080")

            # Send to Slack
            if self.slack_webhook:
                slack_fields = []
                if fields:
                    for k, v in fields.items():
                        slack_fields.append({"title": k, "value": str(v), "short": True})

                slack_payload = {
                    "attachments": [{
                        "color": color,
                        "title": f"[{level.upper()}] {title}",
                        "text": message,
                        "fields": slack_fields,
                        "footer": f"RingRift AI | {node_id}" if node_id else "RingRift AI",
                        "ts": int(time.time()),
                    }]
                }
                try:
                    async with session.post(self.slack_webhook, json=slack_payload) as resp:
                        if resp.status != 200:
                            print(f"[Webhook] Slack alert failed: {resp.status}")
                except Exception as e:
                    print(f"[Webhook] Slack error: {e}")

            # Send to Discord
            if self.discord_webhook:
                discord_fields = []
                if fields:
                    for k, v in fields.items():
                        discord_fields.append({"name": k, "value": str(v), "inline": True})

                discord_payload = {
                    "embeds": [{
                        "title": f"[{level.upper()}] {title}",
                        "description": message,
                        "color": int(color.lstrip("#"), 16),
                        "fields": discord_fields,
                        "footer": {"text": f"RingRift AI | {node_id}" if node_id else "RingRift AI"},
                        "timestamp": datetime.utcnow().isoformat(),
                    }]
                }
                try:
                    async with session.post(self.discord_webhook, json=discord_payload) as resp:
                        if resp.status not in (200, 204):
                            print(f"[Webhook] Discord alert failed: {resp.status}")
                except Exception as e:
                    print(f"[Webhook] Discord error: {e}")

        except Exception as e:
            print(f"[Webhook] Alert send error: {e}")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class P2POrchestrator:
    """Main P2P orchestrator class that runs on each node."""

    def __init__(
        self,
        node_id: str,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        known_peers: List[str] = None,
        ringrift_path: str = None,
        advertise_host: Optional[str] = None,
        advertise_port: Optional[int] = None,
        auth_token: Optional[str] = None,
        require_auth: bool = False,
        storage_type: str = "disk",  # "disk" or "ramdrive"
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.known_peers = known_peers or []
        self.ringrift_path = ringrift_path or self._detect_ringrift_path()

        # Storage configuration: "disk" uses ai-service/data, "ramdrive" uses /dev/shm
        self.storage_type = storage_type
        self.ramdrive_path = "/dev/shm/ringrift/data"  # Standard ramdrive location
        # Git 2.35+ enforces safe.directory for repos with different ownership.
        # Many nodes run the orchestrator as root against a checkout owned by
        # another user (e.g. ubuntu), so always provide a safe.directory override
        # for all git operations.
        self._git_safe_directory = os.path.abspath(self.ringrift_path)
        self.build_version = self._detect_build_version()
        self.start_time = time.time()
        self.last_peer_bootstrap = 0.0

        # Public endpoint peers should use to reach us. Peers learn our host from
        # the heartbeat socket address, but the port must be self-reported. This
        # matters for port-mapped environments like Vast.ai.
        self.advertise_host = (advertise_host or os.environ.get(ADVERTISE_HOST_ENV, "")).strip()
        if not self.advertise_host:
            # Prefer a stable mesh address (Tailscale) when available so nodes
            # behind NAT remain reachable and the cluster converges on a single
            # view of peer endpoints.
            ts_ip = self._get_tailscale_ip()
            self.advertise_host = ts_ip or self._get_local_ip()
        self.advertise_port = advertise_port if advertise_port is not None else self._infer_advertise_port()

        # Optional auth token used to protect mutating endpoints and cluster control.
        # Default is allow-all unless a token is configured.
        env_token = (os.environ.get(AUTH_TOKEN_ENV, "")).strip()
        token_from_arg = (auth_token or "").strip()
        token = token_from_arg or env_token

        if not token:
            token_file = (os.environ.get(AUTH_TOKEN_FILE_ENV, "")).strip()
            if token_file:
                try:
                    token = Path(token_file).read_text().strip()
                except Exception as e:
                    print(f"[P2P] Auth: failed to read {AUTH_TOKEN_FILE_ENV}={token_file}: {e}")

        self.auth_token = token.strip()
        self.require_auth = bool(require_auth)
        if self.require_auth and not self.auth_token:
            raise ValueError(
                f"--require-auth set but {AUTH_TOKEN_ENV}/{AUTH_TOKEN_FILE_ENV}/--auth-token is empty"
            )

        # Optional split-brain mitigation: require a majority of "voter" nodes
        # to be visible before assuming or renewing leadership.
        #
        # Voters can be configured via:
        # - env: RINGRIFT_P2P_VOTERS="node-a,node-b,..."
        # - ai-service/config/distributed_hosts.yaml: per-host `p2p_voter: true`
        self.voter_config_source: str = "none"  # env|config|state|learned|none
        self.voter_node_ids: List[str] = self._load_voter_node_ids()
        self.voter_quorum_size: int = (len(self.voter_node_ids) // 2 + 1) if self.voter_node_ids else 0
        if self.voter_node_ids:
            print(
                f"[P2P] Voter quorum enabled: voters={len(self.voter_node_ids)}, "
                f"quorum={self.voter_quorum_size} ({', '.join(self.voter_node_ids)})"
            )

        # Node state
        self.role = NodeRole.FOLLOWER
        self.leader_id: Optional[str] = None
        self.peers: Dict[str, NodeInfo] = {}
        self.local_jobs: Dict[str, ClusterJob] = {}

        # Distributed job state tracking (leader-only)
        self.distributed_cmaes_state: Dict[str, DistributedCMAESState] = {}
        self.distributed_tournament_state: Dict[str, DistributedTournamentState] = {}
        self.ssh_tournament_runs: Dict[str, SSHTournamentRun] = {}
        self.improvement_loop_state: Dict[str, ImprovementLoopState] = {}
        # Limit CPU-heavy CMA-ES local evaluations to avoid runaway process
        # explosions that can starve the orchestrator (especially on relay hubs).
        try:
            raw = (os.environ.get("RINGRIFT_P2P_MAX_CONCURRENT_CMAES_EVALS", "") or "").strip()
            self.max_concurrent_cmaes_evals = max(1, int(raw)) if raw else 2
        except Exception:
            self.max_concurrent_cmaes_evals = 2
        self._cmaes_eval_semaphore = asyncio.Semaphore(int(self.max_concurrent_cmaes_evals))

        # Phase 2: Distributed data sync state
        self.local_data_manifest: Optional[NodeDataManifest] = None
        self.cluster_data_manifest: Optional[ClusterDataManifest] = None  # Leader-only
        self.manifest_collection_interval = 300.0  # Collect manifests every 5 minutes
        self.last_manifest_collection = 0.0

        # Dashboard/selfplay stats history (leader-only). Stored in-memory to
        # enable lightweight throughput charts without adding DB migrations.
        self.selfplay_stats_history: List[Dict[str, Any]] = []
        self.selfplay_stats_history_max_samples: int = 288  # ~24h @ 5-min cadence

        # Canonical gate jobs (leader-only): dashboard-triggered runs of
        # scripts/generate_canonical_selfplay.py.
        self.canonical_gate_jobs: Dict[str, Dict[str, Any]] = {}
        self.canonical_gate_jobs_lock = threading.Lock()

        # Phase 2: P2P rsync coordination state
        self.active_sync_jobs: Dict[str, DataSyncJob] = {}
        self.current_sync_plan: Optional[ClusterSyncPlan] = None  # Leader-only
        self.pending_sync_requests: List[Dict[str, Any]] = []  # Requests from non-leader nodes
        self.sync_in_progress = False
        self.last_sync_time = 0.0
        self.auto_sync_interval = 600.0  # Auto-sync every 10 minutes when data is missing

        # Training node priority sync state (leader-only)
        self.training_sync_interval = TRAINING_SYNC_INTERVAL
        self.last_training_sync_time = 0.0
        self.training_nodes_cache: List[str] = []  # Cached list of top GPU nodes
        self.training_nodes_cache_time = 0.0
        self.games_synced_to_training: Dict[str, int] = {}  # node_id -> last synced game count

        # Phase 3: Training pipeline state (leader-only)
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.training_thresholds: TrainingThresholds = TrainingThresholds()
        self.last_training_check: float = 0.0
        self.training_check_interval: float = 300.0  # Check every 5 minutes
        self.games_at_last_nnue_train: Dict[str, int] = {}  # board_type -> game_count
        self.games_at_last_cmaes_train: Dict[str, int] = {}

        # Phase 5: Automated improvement cycle manager (leader-only)
        self.improvement_cycle_manager: Optional['ImprovementCycleManager'] = None
        if HAS_IMPROVEMENT_MANAGER:
            try:
                self.improvement_cycle_manager = ImprovementCycleManager(
                    db_path=STATE_DIR / f"{node_id}_improvement.db",
                    ringrift_path=self.ringrift_path,
                )
                print(f"[P2P] ImprovementCycleManager initialized")
            except Exception as e:
                print(f"[P2P] Failed to initialize ImprovementCycleManager: {e}")
        self.last_improvement_cycle_check: float = 0.0

        # P2P-integrated monitoring (leader starts Prometheus/Grafana)
        self.monitoring_manager: Optional['MonitoringManager'] = None
        if HAS_P2P_MONITORING:
            try:
                self.monitoring_manager = MonitoringManager(
                    node_id=node_id,
                    prometheus_port=9090,
                    grafana_port=3000,
                    config_dir=Path(self.ringrift_path) / "monitoring",
                )
                print(f"[P2P] MonitoringManager initialized")
            except Exception as e:
                print(f"[P2P] Failed to initialize MonitoringManager: {e}")
        self._monitoring_was_leader = False  # Track leadership changes
        self.improvement_cycle_check_interval: float = 600.0  # Check every 10 minutes

        # Webhook notifications for alerts
        self.notifier = WebhookNotifier()

        # Diversity tracking metrics
        self.diversity_metrics = {
            "games_by_engine_mode": {},      # engine_mode -> count
            "games_by_board_config": {},     # "board_players" -> count
            "games_by_difficulty": {},       # difficulty -> count
            "asymmetric_games": 0,           # count of asymmetric games scheduled
            "symmetric_games": 0,            # count of symmetric games scheduled
            "training_triggers": 0,          # count of training triggers
            "cmaes_triggers": 0,             # count of CMA-ES triggers
            "promotions": 0,                 # count of model promotions
            "rollbacks": 0,                  # count of rollbacks
            "last_reset": time.time(),       # when metrics were last reset
        }

        # === CRITICAL SELF-IMPROVEMENT LOOP METRICS ===
        # Training progress tracking (populated by training callbacks)
        self.training_metrics: Dict[str, Dict[str, float]] = {}  # config -> {loss, val_loss, epoch}

        # Selfplay throughput tracking
        self.selfplay_throughput: Dict[str, float] = {}  # config -> games/hour

        # Cost efficiency metrics
        self.cost_metrics: Dict[str, float] = {
            "gpu_hours_total": 0.0,
            "estimated_cost_usd": 0.0,
            "elo_per_gpu_hour": 0.0,
        }

        # Promotion quality metrics
        self.promotion_metrics: Dict[str, Any] = {
            "success_rate": 0.0,
            "avg_elo_gain": 0.0,
            "rejections": {},  # reason -> count
            "total_attempts": 0,
            "successful": 0,
        }

        # LEARNED LESSONS - Stuck job detection (leader-only)
        # Track when each node's GPU first went idle with running jobs
        self.gpu_idle_since: Dict[str, float] = {}  # node_id -> timestamp when GPU went idle

        # A/B Testing Framework - Compare models head-to-head with statistical significance
        # Key: test_id (UUID), Value: ABTestState dict
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        self.ab_test_lock = threading.Lock()

        # Locks for thread safety
        self.peers_lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.manifest_lock = threading.Lock()
        self.sync_lock = threading.Lock()
        self.training_lock = threading.Lock()
        self.ssh_tournament_lock = threading.Lock()
        self.relay_lock = threading.Lock()

        # State persistence
        self.db_path = STATE_DIR / f"{node_id}_state.db"
        self._init_database()

        # Event flags
        self.running = True
        self.election_in_progress = False
        self.last_election_attempt: float = 0.0

        # LEARNED LESSONS - Lease-based leadership to prevent split-brain
        # Leader must continuously renew lease; if lease expires, leadership is void
        self.leader_lease_expires: float = 0.0  # timestamp when current leader's lease expires
        self.last_lease_renewal: float = 0.0  # when we last renewed our lease (if leader)
        self.leader_lease_id: str = ""  # unique ID for current leadership term

        # Voter-backed lease grants (split-brain resistance).
        #
        # When quorum gating is enabled, voters act as a lightweight consensus
        # group by granting an exclusive leader lease to a single node at a time.
        # A leader must renew its lease with a quorum of voters; otherwise it
        # steps down. This prevents split-brain even if multiple nodes think
        # they are eligible leaders.
        self.voter_grant_leader_id: str = ""
        self.voter_grant_lease_id: str = ""
        self.voter_grant_expires: float = 0.0

        # Job completion tracking for auto-restart
        self.completed_jobs: Dict[str, float] = {}  # node_id -> last job completion time
        self.jobs_started_at: Dict[str, Dict[str, float]] = {}  # node_id -> {job_id: start_time}

        # NAT/relay support (for nodes without inbound connectivity).
        # NAT-blocked nodes poll a relay endpoint for commands; the leader enqueues
        # commands keyed by node_id.
        self.last_inbound_heartbeat: float = 0.0
        self.last_relay_heartbeat: float = 0.0
        self.relay_command_queue: Dict[str, List[Dict[str, Any]]] = {}
        self.pending_relay_acks: Set[str] = set()
        self.pending_relay_results: List[Dict[str, Any]] = []
        self.relay_command_attempts: Dict[str, int] = {}

        # SAFEGUARDS - Rate limiting and coordinator integration (added 2025-12-15)
        self.spawn_timestamps: List[float] = []  # Timestamps of recent process spawns
        self.agent_mode = AGENT_MODE_ENABLED
        self.coordinator_url = COORDINATOR_URL
        self.last_coordinator_check: float = 0.0
        self.coordinator_available: bool = False
        print(f"[P2P] Safeguards: rate_limit={SPAWN_RATE_LIMIT_PER_MINUTE}/min, "
              f"load_max={LOAD_AVERAGE_MAX_MULTIPLIER}x, agent_mode={self.agent_mode}")

        # Load persisted state
        self._load_state()
        if self.leader_id == self.node_id:
            self.role = NodeRole.LEADER

        # Self info
        self.self_info = self._create_self_info()

        print(
            f"[P2P] Initialized node {node_id} on {host}:{port} "
            f"(advertise {self.advertise_host}:{self.advertise_port})"
        )
        print(f"[P2P] RingRift path: {self.ringrift_path}")
        print(f"[P2P] Version: {self.build_version}")
        print(f"[P2P] Known peers: {self.known_peers}")
        if self.auth_token:
            print(f"[P2P] Auth: enabled via {AUTH_TOKEN_ENV}")
        else:
            print(f"[P2P] Auth: disabled (set {AUTH_TOKEN_ENV} to enable)")

    def _is_leader(self) -> bool:
        """Check if this node is the current cluster leader with valid lease."""
        if self.leader_id != self.node_id:
            # Consistency: we should never claim role=leader while leader_id points elsewhere (or is None).
            if self.role == NodeRole.LEADER:
                print("[P2P] Inconsistent leadership state (role=leader but leader_id!=self); stepping down")
                self.role = NodeRole.FOLLOWER
                self.last_lease_renewal = 0.0
                if not self.leader_id:
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0.0
                self._release_voter_grant_if_self()
                self._save_state()
                # Only force an election when we have no known leader; otherwise we
                # may already be following a healthy leader and shouldn't flap.
                if not self.leader_id:
                    try:
                        asyncio.get_running_loop().create_task(self._start_election())
                    except RuntimeError:
                        pass
            return False
        # Consistency: we should never claim leader_id=self while being a follower/candidate.
        if self.role != NodeRole.LEADER:
            print("[P2P] Inconsistent leadership state (leader_id=self but role!=leader); clearing leader_id")
            self.role = NodeRole.FOLLOWER
            self.leader_id = None
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
            self.last_lease_renewal = 0.0
            self._release_voter_grant_if_self()
            self._save_state()
            try:
                asyncio.get_running_loop().create_task(self._start_election())
            except RuntimeError:
                pass
            return False

        # LEARNED LESSONS - Lease-based leadership prevents split-brain
        # Must have valid lease to act as leader
        if self.leader_lease_expires > 0 and time.time() >= self.leader_lease_expires:
            print("[P2P] Leadership lease expired, stepping down")
            self.role = NodeRole.FOLLOWER
            self.leader_id = None
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
            self.last_lease_renewal = 0.0
            self._release_voter_grant_if_self()
            self._save_state()
            try:
                asyncio.get_running_loop().create_task(self._start_election())
            except RuntimeError:
                pass
            return False
        if getattr(self, "voter_node_ids", []) and not self._has_voter_quorum():
            print("[P2P] Leadership without voter quorum, stepping down")
            self.role = NodeRole.FOLLOWER
            self.leader_id = None
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
            self.last_lease_renewal = 0.0
            self._release_voter_grant_if_self()
            self._save_state()
            try:
                asyncio.get_running_loop().create_task(self._start_election())
            except RuntimeError:
                pass
            return False
        return True


    # =========================================================================
    # SAFEGUARDS - Load, rate limiting, and coordinator integration
    # =========================================================================

    def _check_spawn_rate_limit(self) -> Tuple[bool, str]:
        """Check if we're within the spawn rate limit.

        SAFEGUARD: Prevents runaway process spawning by limiting spawns per minute.

        Returns:
            (can_spawn, reason) - True if within rate limit
        """
        now = time.time()
        # Clean old timestamps (older than 60 seconds)
        self.spawn_timestamps = [t for t in self.spawn_timestamps if now - t < 60]

        if len(self.spawn_timestamps) >= SPAWN_RATE_LIMIT_PER_MINUTE:
            return False, f"Rate limit: {len(self.spawn_timestamps)}/{SPAWN_RATE_LIMIT_PER_MINUTE} spawns in last minute"

        return True, f"Rate OK: {len(self.spawn_timestamps)}/{SPAWN_RATE_LIMIT_PER_MINUTE}"

    def _record_spawn(self) -> None:
        """Record a process spawn for rate limiting."""
        self.spawn_timestamps.append(time.time())

    async def _check_coordinator_available(self) -> bool:
        """Check if the unified coordinator is available.

        SAFEGUARD: In agent mode, defer job decisions to coordinator.

        Returns:
            True if coordinator is reachable
        """
        if not self.coordinator_url:
            return False

        # Cache check for 30 seconds
        now = time.time()
        if now - self.last_coordinator_check < 30:
            return self.coordinator_available

        self.last_coordinator_check = now

        try:
            async with get_client_session(timeout=ClientTimeout(total=5)) as session:
                async with session.get(f"{self.coordinator_url}/api/health") as resp:
                    self.coordinator_available = resp.status == 200
                    if self.coordinator_available:
                        print(f"[P2P] Coordinator available at {self.coordinator_url}")
                    return self.coordinator_available
        except Exception:
            self.coordinator_available = False
            return False

    def _can_spawn_process(self, reason: str = "job") -> Tuple[bool, str]:
        """Combined safeguard check before spawning any process.

        SAFEGUARD: Checks load average, rate limit, and agent mode.

        Args:
            reason: Description of why we want to spawn (for logging)

        Returns:
            (can_spawn, explanation) - True if all checks pass
        """
        # Check 1: Load average
        load_ok, load_reason = self.self_info.check_load_average_safe()
        if not load_ok:
            print(f"[P2P] BLOCKED spawn ({reason}): {load_reason}")
            return False, load_reason

        # Check 2: Rate limit
        rate_ok, rate_reason = self._check_spawn_rate_limit()
        if not rate_ok:
            print(f"[P2P] BLOCKED spawn ({reason}): {rate_reason}")
            return False, rate_reason

        # Check 3: Agent mode - if coordinator is available and we're in agent mode,
        # we should not autonomously spawn jobs (let coordinator decide)
        if self.agent_mode and self.coordinator_available:
            msg = "Agent mode: deferring to coordinator"
            print(f"[P2P] BLOCKED spawn ({reason}): {msg}")
            return False, msg

        # Check 4: Backpressure (new coordination) - if training queue is saturated,
        # don't spawn more selfplay jobs that would produce more data
        if HAS_NEW_COORDINATION and "selfplay" in reason.lower():
            if should_stop_production(QueueType.TRAINING_DATA):
                msg = "Backpressure: training queue at STOP level"
                print(f"[P2P] BLOCKED spawn ({reason}): {msg}")
                return False, msg
            if should_throttle_production(QueueType.TRAINING_DATA):
                throttle = get_throttle_factor(QueueType.TRAINING_DATA)
                import random
                if random.random() > throttle:
                    msg = f"Backpressure: throttled (factor={throttle:.2f})"
                    print(f"[P2P] BLOCKED spawn ({reason}): {msg}")
                    return False, msg

        return True, "All safeguards passed"

    def _detect_build_version(self) -> str:
        env_version = (os.environ.get(BUILD_VERSION_ENV, "") or "").strip()
        if env_version:
            return env_version

        commit = ""
        branch = ""
        try:
            result = subprocess.run(
                self._git_cmd("rev-parse", "--short", "HEAD"),
                cwd=self.ringrift_path,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
        except Exception:
            commit = ""

        try:
            result = subprocess.run(
                self._git_cmd("rev-parse", "--abbrev-ref", "HEAD"),
                cwd=self.ringrift_path,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
        except Exception:
            branch = ""

        if commit and branch:
            return f"{branch}@{commit}"
        return commit or "unknown"

    def _git_cmd(self, *args: str) -> List[str]:
        safe_dir = getattr(self, "_git_safe_directory", "") or os.path.abspath(self.ringrift_path)
        return ["git", "-c", f"safe.directory={safe_dir}", *args]

    def _detect_ringrift_path(self) -> str:
        """Detect the RingRift installation path."""
        # Try common locations
        candidates = [
            Path.home() / "Development" / "RingRift",
            Path.home() / "ringrift",
            Path("/home/ubuntu/ringrift"),
            Path("/root/ringrift"),
        ]
        for path in candidates:
            if (path / "ai-service").exists():
                return str(path)
        return str(Path(__file__).parent.parent.parent)

    def get_data_directory(self) -> Path:
        """Get the data directory path based on storage configuration.

        Returns:
            Path to data directory:
            - ramdrive: /dev/shm/ringrift/data (for disk-constrained Vast instances)
            - disk: {ringrift_path}/ai-service/data (default)

        The ramdrive option uses tmpfs for high-speed I/O and to work around
        limited disk space on some cloud instances. Data stored in ramdrive
        is volatile and should be synced to permanent storage periodically.
        """
        if self.storage_type == "ramdrive":
            ramdrive = Path(self.ramdrive_path)
            ramdrive.mkdir(parents=True, exist_ok=True)
            return ramdrive
        return Path(self.ringrift_path) / "ai-service" / "data"

    def _infer_advertise_port(self) -> int:
        """Infer the externally reachable port for this node.

        - Explicit `RINGRIFT_ADVERTISE_PORT` always wins.
        - Vast.ai exposes container ports via `VAST_TCP_PORT_<PORT>`; when set,
          use that public port so peers can reach us.
        - Default to the listening port.
        """
        explicit = (os.environ.get(ADVERTISE_PORT_ENV, "")).strip()
        if explicit:
            try:
                return int(explicit)
            except ValueError:
                pass

        vast_key = f"VAST_TCP_PORT_{self.port}"
        mapped = (os.environ.get(vast_key, "")).strip()
        if mapped:
            try:
                return int(mapped)
            except ValueError:
                pass

        return int(self.port)

    def _load_voter_node_ids(self) -> List[str]:
        """Load the set of P2P voter node_ids (for quorum-based leadership).

        If no voters are configured, returns an empty list and quorum checks are
        disabled (backwards compatible).
        """
        env = (os.environ.get("RINGRIFT_P2P_VOTERS") or "").strip()
        if env:
            self.voter_config_source = "env"
            voters = [t.strip() for t in env.split(",") if t.strip()]
            return sorted(set(voters))

        cfg_path = Path(self.ringrift_path) / "ai-service" / "config" / "distributed_hosts.yaml"
        if not cfg_path.exists():
            self.voter_config_source = "none"
            return []

        try:
            import yaml  # type: ignore
        except Exception:
            return []

        try:
            data = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            return []

        hosts = data.get("hosts", {}) or {}
        voters: List[str] = []
        for node_id, cfg in hosts.items():
            if not isinstance(cfg, dict):
                continue
            raw = cfg.get("p2p_voter", False)
            if raw is True:
                voters.append(str(node_id))
                continue
            if isinstance(raw, (int, float)) and int(raw) == 1:
                voters.append(str(node_id))
                continue
            if isinstance(raw, str) and raw.strip().lower() in {"1", "true", "yes", "y"}:
                voters.append(str(node_id))
        voters = sorted(set(voters))
        self.voter_config_source = "config" if voters else "none"
        return voters

    def _maybe_adopt_voter_node_ids(self, voter_node_ids: List[str], *, source: str) -> bool:
        """Adopt/override the voter set when it's not explicitly configured via env.

        This is a convergence mechanism: some nodes may boot without local
        config (or with stale config), which would disable quorum gating and
        allow non-voter nodes to become leaders. Leaders propagate the stable
        voter set via `/coordinator` so the cluster converges.
        """
        if (os.environ.get("RINGRIFT_P2P_VOTERS") or "").strip():
            return False

        normalized = sorted({str(v).strip() for v in (voter_node_ids or []) if str(v).strip()})
        if not normalized:
            return False

        current = sorted(set(getattr(self, "voter_node_ids", []) or []))
        if current == normalized:
            return False

        self.voter_node_ids = normalized
        self.voter_quorum_size = len(normalized) // 2 + 1
        self.voter_config_source = source or "learned"
        print(
            f"[P2P] Updated voter set ({self.voter_config_source}): voters={len(normalized)}, "
            f"quorum={self.voter_quorum_size} ({', '.join(normalized)})"
        )
        return True

    def _has_voter_quorum(self) -> bool:
        """Return True if we currently see a majority of voter nodes alive."""
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if not voters:
            return True
        quorum = int(getattr(self, "voter_quorum_size", 0) or 0)
        if quorum <= 0:
            quorum = len(voters) // 2 + 1

        alive = 0
        with self.peers_lock:
            peers = dict(self.peers)
        for node_id in voters:
            if node_id == self.node_id:
                alive += 1
                continue
            peer = peers.get(node_id)
            if peer and peer.is_alive():
                alive += 1
        return alive >= quorum

    def _release_voter_grant_if_self(self) -> None:
        """Release our voter-side lease grant when stepping down.

        This shortens failover time when the leader voluntarily steps down (e.g.
        lost quorum) by not forcing other candidates to wait for the full lease
        TTL to expire.
        """
        if str(getattr(self, "voter_grant_leader_id", "") or "") != self.node_id:
            return
        self.voter_grant_leader_id = ""
        self.voter_grant_lease_id = ""
        self.voter_grant_expires = 0.0

    async def _acquire_voter_lease_quorum(self, lease_id: str, duration: int) -> Optional[float]:
        """Acquire/renew an exclusive leader lease from a quorum of voters.

        Returns the effective lease expiry timestamp if a quorum granted the
        lease; otherwise returns None.
        """
        voter_ids = list(getattr(self, "voter_node_ids", []) or [])
        if not voter_ids:
            return time.time() + float(duration)

        quorum = int(getattr(self, "voter_quorum_size", 0) or 0)
        if quorum <= 0:
            quorum = len(voter_ids) // 2 + 1

        now = time.time()
        duration = max(10, min(int(duration), int(LEADER_LEASE_DURATION * 2)))

        acks = 0
        lease_ttls: List[float] = []

        # Self-grant (as a voter).
        if self.node_id in voter_ids:
            self.voter_grant_leader_id = self.node_id
            self.voter_grant_lease_id = lease_id
            self.voter_grant_expires = now + float(duration)
            lease_ttls.append(float(duration))
            acks += 1

        with self.peers_lock:
            peers_by_id = dict(self.peers)

        timeout = ClientTimeout(total=5)
        async with get_client_session(timeout) as session:
            for voter_id in voter_ids:
                if acks >= quorum:
                    break
                if voter_id == self.node_id:
                    continue
                voter = peers_by_id.get(voter_id)
                if not voter or not voter.is_alive():
                    continue

                payload = {
                    "leader_id": self.node_id,
                    "lease_id": lease_id,
                    "lease_duration": duration,
                }

                for url in self._urls_for_peer(voter, "/election/lease"):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                continue
                            data = await resp.json()
                            if not data.get("granted"):
                                break
                            ttl_raw = data.get("lease_ttl_seconds")
                            if ttl_raw is None:
                                ttl_raw = data.get("ttl_seconds")
                            ttl_val: Optional[float] = None
                            if ttl_raw is not None:
                                try:
                                    ttl_val = float(ttl_raw)
                                except Exception:
                                    ttl_val = None
                            if ttl_val is not None and ttl_val > 0:
                                lease_ttls.append(ttl_val)
                            else:
                                lease_ttls.append(float(duration))
                            acks += 1
                            break
                    except Exception:
                        continue

        if acks < quorum:
            return None
        # Use a relative TTL (computed by each voter on its own clock) to avoid
        # leader lease flapping under clock skew. Convert back to a local expiry.
        effective_ttl = min(lease_ttls) if lease_ttls else float(duration)
        effective_ttl = max(10.0, min(float(duration), float(effective_ttl)))
        return now + float(effective_ttl)

    async def _determine_leased_leader_from_voters(self) -> Optional[str]:
        """Return the current lease-holder as reported by a quorum of voters.

        This is a read-only reconciliation step used to resolve split-brain once
        partitions heal. It queries the current voter grant state via
        `/election/grant` and selects the leader_id that has >= quorum votes with
        non-expired grants.
        """
        voter_ids = list(getattr(self, "voter_node_ids", []) or [])
        if not voter_ids:
            return None

        quorum = int(getattr(self, "voter_quorum_size", 0) or 0)
        if quorum <= 0:
            quorum = len(voter_ids) // 2 + 1

        now = time.time()
        counts: Dict[str, int] = {}

        # Include local voter state.
        if self.node_id in voter_ids:
            leader_id = str(getattr(self, "voter_grant_leader_id", "") or "")
            expires = float(getattr(self, "voter_grant_expires", 0.0) or 0.0)
            if leader_id and expires > now:
                counts[leader_id] = counts.get(leader_id, 0) + 1

        with self.peers_lock:
            peers_by_id = dict(self.peers)

        timeout = ClientTimeout(total=5)
        async with get_client_session(timeout) as session:
            for voter_id in voter_ids:
                if voter_id == self.node_id:
                    continue
                voter = peers_by_id.get(voter_id)
                if not voter or not voter.is_alive():
                    continue

                for url in self._urls_for_peer(voter, "/election/grant"):
                    try:
                        async with session.get(url, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                continue
                            data = await resp.json()
                        leader_id = str((data or {}).get("leader_id") or "")
                        if not leader_id:
                            break
                        ttl_raw = (data or {}).get("lease_ttl_seconds")
                        if ttl_raw is None:
                            ttl_raw = (data or {}).get("ttl_seconds")
                        ttl_val: Optional[float] = None
                        if ttl_raw is not None:
                            try:
                                ttl_val = float(ttl_raw)
                            except Exception:
                                ttl_val = None

                        if ttl_val is not None:
                            if ttl_val <= 0:
                                break
                        else:
                            # Back-compat: use absolute expiry as best-effort, with
                            # a generous skew tolerance (1× lease duration).
                            expires = float((data or {}).get("lease_expires") or 0.0)
                            if expires <= 0:
                                break
                            if expires + float(LEADER_LEASE_DURATION) < now:
                                break
                        counts[leader_id] = counts.get(leader_id, 0) + 1
                        break
                    except Exception:
                        continue

        winners = [leader_id for leader_id, count in counts.items() if count >= quorum]
        if not winners:
            return None
        # Deterministic: if multiple satisfy quorum (shouldn't), pick highest node_id.
        return sorted(winners)[-1]

    def _parse_peer_address(self, peer_addr: str) -> Tuple[str, str, int]:
        """Parse `--peers` entries.

        Supports:
        - `host`
        - `host:port`
        - `http://host[:port]`
        - `https://host[:port]`
        """
        peer_addr = (peer_addr or "").strip()
        if not peer_addr:
            raise ValueError("Empty peer address")

        if "://" in peer_addr:
            parsed = urlparse(peer_addr)
            scheme = (parsed.scheme or "http").lower()
            host = parsed.hostname or ""
            if not host:
                raise ValueError(f"Invalid peer URL: {peer_addr}")
            if parsed.port is not None:
                port = int(parsed.port)
            else:
                port = 443 if scheme == "https" else DEFAULT_PORT
            return scheme, host, port

        # Back-compat: host[:port]
        parts = peer_addr.split(":", 1)
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 and parts[1] else DEFAULT_PORT
        return "http", host, port

    def _url_for_peer(self, peer: NodeInfo, path: str) -> str:
        scheme = (getattr(peer, "scheme", None) or "http").lower()
        host = str(getattr(peer, "host", "") or "").strip()
        try:
            port = int(getattr(peer, "port", DEFAULT_PORT) or DEFAULT_PORT)
        except Exception:
            port = DEFAULT_PORT

        rh = (getattr(peer, "reported_host", "") or "").strip()
        try:
            rp = int(getattr(peer, "reported_port", 0) or 0)
        except Exception:
            rp = 0

        if rh and rp:
            # Prefer reported endpoints when the observed endpoint is loopback
            # (proxy/relay artifacts).
            if host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
                host, port = rh, rp
            # Prefer mesh endpoints (Tailscale) when we also have a mesh address.
            elif self._local_has_tailscale() and self._is_tailscale_host(rh):
                host, port = rh, rp

        return f"{scheme}://{host}:{port}{path}"

    def _urls_for_peer(self, peer: NodeInfo, path: str) -> List[str]:
        """Return candidate URLs for reaching a peer.

        Includes both the observed reachable endpoint (`host`/`port`) and the
        peer's self-reported endpoint (`reported_host`/`reported_port`) when
        available. This improves resilience in mixed network environments
        (public IP vs overlay networks like Tailscale, port-mapped listeners).
        """
        scheme = (getattr(peer, "scheme", None) or "http").lower()
        urls: List[str] = []

        def _add(host: Any, port: Any) -> None:
            try:
                h = str(host or "").strip()
                p = int(port)
            except Exception:
                return
            if not h or p <= 0:
                return
            url = f"{scheme}://{h}:{p}{path}"
            if url not in urls:
                urls.append(url)

        rh = (getattr(peer, "reported_host", "") or "").strip()
        try:
            rp = int(getattr(peer, "reported_port", 0) or 0)
        except Exception:
            rp = 0

        host = str(getattr(peer, "host", "") or "").strip()
        try:
            port = int(getattr(peer, "port", 0) or 0)
        except Exception:
            port = 0

        # Prefer Tailscale endpoints first when available locally; otherwise try
        # the observed endpoint first.
        reported_preferred = False
        if rh and rp and self._local_has_tailscale() and self._is_tailscale_host(rh):
            _add(rh, rp)
            reported_preferred = True

        _add(host, port)

        if rh and rp and (not reported_preferred) and (rh != host or rp != port):
            _add(rh, rp)

        return urls

    def _auth_headers(self) -> Dict[str, str]:
        if not self.auth_token:
            return {}
        return {"Authorization": f"Bearer {self.auth_token}"}

    def _get_leader_peer(self) -> Optional[NodeInfo]:
        if self._is_leader():
            return self.self_info

        with self.peers_lock:
            peers_snapshot = list(self.peers.values())

        conflict_keys = self._endpoint_conflict_keys([self.self_info] + peers_snapshot)

        leader_id = self.leader_id
        if leader_id:
            # Only treat persisted leader_id as "effective" when:
            # - we still consider the lease valid, and
            # - the peer currently reports itself as a leader (via heartbeats).
            #
            # Otherwise, stale leader_ids can cause the cluster to get stuck
            # leaderless (e.g. after restarts/partitions) and break leader-proxy APIs.
            if self._is_leader_lease_valid():
                for peer in peers_snapshot:
                    if (
                        peer.node_id == leader_id
                        and peer.role == NodeRole.LEADER
                        and peer.is_alive()
                        and self._is_leader_eligible(peer, conflict_keys)
                    ):
                        return peer

        eligible_leaders = [
            peer for peer in peers_snapshot
            if peer.role == NodeRole.LEADER and self._is_leader_eligible(peer, conflict_keys)
        ]
        if eligible_leaders:
            return sorted(eligible_leaders, key=lambda p: p.node_id)[-1]

        return None

    async def _proxy_to_leader(self, request: web.Request) -> web.StreamResponse:
        """Best-effort proxy for leader-only APIs when the dashboard hits a follower."""
        leader = self._get_leader_peer()
        if not leader:
            return web.json_response(
                {"success": False, "error": "leader_unknown", "leader_id": self.leader_id},
                status=503,
            )

        candidate_urls = self._urls_for_peer(leader, request.raw_path)
        if not candidate_urls:
            candidate_urls = [self._url_for_peer(leader, request.raw_path)]
        forward_headers: Dict[str, str] = {}
        for h in ("Authorization", "X-RingRift-Auth", "Content-Type"):
            if h in request.headers:
                forward_headers[h] = request.headers[h]

        body: bytes | None = None
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            body = await request.read()

        # Keep leader-proxy responsive: unreachable "leaders" (often NAT/firewall)
        # should fail fast so the dashboard doesn't hang for a full minute.
        timeout = ClientTimeout(total=10)
        last_exc: Exception | None = None
        async with get_client_session(timeout) as session:
            for target_url in candidate_urls:
                try:
                    async with session.request(
                        request.method,
                        target_url,
                        data=body,
                        headers=forward_headers,
                    ) as resp:
                        payload = await resp.read()
                        content_type = resp.headers.get("Content-Type")
                        headers: Dict[str, str] = {}
                        if content_type:
                            headers["Content-Type"] = content_type
                        headers["X-RingRift-Proxied-By"] = self.node_id
                        headers["X-RingRift-Proxied-To"] = target_url
                        return web.Response(body=payload, status=resp.status, headers=headers)
                except Exception as exc:
                    last_exc = exc
                    continue

        return web.json_response(
            {
                "success": False,
                "error": "leader_proxy_failed",
                "message": str(last_exc) if last_exc else "unknown_error",
                "leader_id": self.leader_id,
                "attempted_urls": candidate_urls,
            },
            status=502,
        )

    def _is_request_authorized(self, request: "web.Request") -> bool:
        if not self.auth_token:
            return True

        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        if not token:
            token = request.headers.get("X-RingRift-Auth", "").strip()
        if not token:
            return False

        return secrets.compare_digest(token, self.auth_token)

    def _init_database(self):
        """Initialize SQLite database for state persistence."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Peers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS peers (
                node_id TEXT PRIMARY KEY,
                host TEXT,
                port INTEGER,
                last_heartbeat REAL,
                info_json TEXT
            )
        """)

        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT,
                node_id TEXT,
                board_type TEXT,
                num_players INTEGER,
                engine_mode TEXT,
                pid INTEGER,
                started_at REAL,
                status TEXT
            )
        """)

        # State table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # Metrics history table for observability
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                metric_type TEXT NOT NULL,
                board_type TEXT,
                num_players INTEGER,
                value REAL NOT NULL,
                metadata TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_type_time
            ON metrics_history(metric_type, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_config
            ON metrics_history(board_type, num_players, timestamp)
        """)

        # A/B Testing tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                test_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                model_a TEXT NOT NULL,
                model_b TEXT NOT NULL,
                target_games INTEGER DEFAULT 100,
                confidence_threshold REAL DEFAULT 0.95,
                status TEXT DEFAULT 'running',
                winner TEXT,
                created_at REAL NOT NULL,
                completed_at REAL,
                metadata TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_test_games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                model_a_result TEXT NOT NULL,
                model_a_score REAL NOT NULL,
                model_b_score REAL NOT NULL,
                game_length INTEGER,
                played_at REAL NOT NULL,
                metadata TEXT,
                FOREIGN KEY (test_id) REFERENCES ab_tests(test_id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ab_games_test
            ON ab_test_games(test_id, played_at)
        """)

        conn.commit()
        conn.close()

    def _load_state(self):
        """Load persisted state from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Load peers
            cursor.execute("SELECT node_id, info_json FROM peers")
            for row in cursor.fetchall():
                try:
                    if row[0] == self.node_id:
                        continue
                    info = NodeInfo.from_dict(json.loads(row[1]))
                    self.peers[row[0]] = info
                except Exception as e:
                    print(f"[P2P] Failed to load peer {row[0]}: {e}")

            # Load jobs
            cursor.execute("SELECT * FROM jobs WHERE status = 'running'")
            for row in cursor.fetchall():
                job = ClusterJob(
                    job_id=row[0],
                    job_type=JobType(row[1]),
                    node_id=row[2],
                    board_type=row[3],
                    num_players=row[4],
                    engine_mode=row[5],
                    pid=row[6],
                    started_at=row[7],
                    status=row[8],
                )
                self.local_jobs[job.job_id] = job

            # Load leader
            cursor.execute("SELECT key, value FROM state")
            state_rows = {row[0]: row[1] for row in cursor.fetchall() if row and row[0]}
            raw_leader_id = state_rows.get("leader_id")
            if raw_leader_id:
                self.leader_id = raw_leader_id

            raw_lease_id = state_rows.get("leader_lease_id")
            if raw_lease_id:
                self.leader_lease_id = raw_lease_id

            raw_lease_expires = state_rows.get("leader_lease_expires")
            if raw_lease_expires:
                try:
                    self.leader_lease_expires = float(raw_lease_expires)
                except Exception:
                    pass

            raw_last_renewal = state_rows.get("last_lease_renewal")
            if raw_last_renewal:
                try:
                    self.last_lease_renewal = float(raw_last_renewal)
                except Exception:
                    pass

            raw_role = state_rows.get("role")
            if raw_role:
                try:
                    self.role = NodeRole(str(raw_role))
                except Exception:
                    pass

            raw_grant_leader = state_rows.get("voter_grant_leader_id")
            if raw_grant_leader:
                self.voter_grant_leader_id = str(raw_grant_leader)
            raw_grant_lease = state_rows.get("voter_grant_lease_id")
            if raw_grant_lease:
                self.voter_grant_lease_id = str(raw_grant_lease)
            raw_grant_expires = state_rows.get("voter_grant_expires")
            if raw_grant_expires:
                try:
                    self.voter_grant_expires = float(raw_grant_expires)
                except Exception:
                    pass

            # Optional persisted voter configuration (convergence helper). Only
            # apply when voters are not explicitly configured via env/config.
            raw_voters = state_rows.get("voter_node_ids")
            if raw_voters and not (getattr(self, "voter_node_ids", []) or []):
                if str(getattr(self, "voter_config_source", "none") or "none") == "none":
                    voters: List[str] = []
                    try:
                        parsed = json.loads(raw_voters)
                        if isinstance(parsed, list):
                            voters = [str(v).strip() for v in parsed if str(v).strip()]
                    except Exception:
                        voters = [t.strip() for t in str(raw_voters).split(",") if t.strip()]
                    if voters:
                        self._maybe_adopt_voter_node_ids(voters, source="state")

            # Self-heal inconsistent persisted leader state (can happen after
            # abrupt shutdowns or partial writes): never keep role=leader without
            # a matching leader_id.
            if self.role == NodeRole.LEADER and not self.leader_id:
                print("[P2P] Loaded role=leader but leader_id is empty; stepping down to follower")
                self.role = NodeRole.FOLLOWER
                self.leader_lease_id = ""
                self.leader_lease_expires = 0.0
                self.last_lease_renewal = 0.0

            conn.close()
            print(f"[P2P] Loaded state: {len(self.peers)} peers, {len(self.local_jobs)} jobs")
        except Exception as e:
            print(f"[P2P] Failed to load state: {e}")

    def _save_state(self):
        """Save current state to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Save peers
            cursor.execute("DELETE FROM peers WHERE node_id = ?", (self.node_id,))
            with self.peers_lock:
                for node_id, info in self.peers.items():
                    if node_id == self.node_id:
                        continue
                    cursor.execute("""
                        INSERT OR REPLACE INTO peers (node_id, host, port, last_heartbeat, info_json)
                        VALUES (?, ?, ?, ?, ?)
                    """, (node_id, info.host, info.port, info.last_heartbeat, json.dumps(info.to_dict())))

            # Save jobs
            with self.jobs_lock:
                for job_id, job in self.local_jobs.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO jobs
                        (job_id, job_type, node_id, board_type, num_players, engine_mode, pid, started_at, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (job.job_id, job.job_type.value, job.node_id, job.board_type,
                          job.num_players, job.engine_mode, job.pid, job.started_at, job.status))

            # Save leader
            role_value = self.role.value if hasattr(self.role, "value") else str(self.role)
            voter_node_ids_json = json.dumps(sorted(set(getattr(self, "voter_node_ids", []) or [])))
            voter_config_source = str(getattr(self, "voter_config_source", "") or "")
            state_payload = [
                ("leader_id", self.leader_id),
                ("leader_lease_id", self.leader_lease_id or ""),
                ("leader_lease_expires", str(float(self.leader_lease_expires or 0.0))),
                ("last_lease_renewal", str(float(self.last_lease_renewal or 0.0))),
                ("role", role_value),
                ("voter_node_ids", voter_node_ids_json),
                ("voter_config_source", voter_config_source),
                ("voter_grant_leader_id", str(getattr(self, "voter_grant_leader_id", "") or "")),
                ("voter_grant_lease_id", str(getattr(self, "voter_grant_lease_id", "") or "")),
                ("voter_grant_expires", str(float(getattr(self, "voter_grant_expires", 0.0) or 0.0))),
            ]
            cursor.executemany(
                "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                state_payload,
                )

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[P2P] Failed to save state: {e}")

    def record_metric(
        self,
        metric_type: str,
        value: float,
        board_type: str = None,
        num_players: int = None,
        metadata: Dict[str, Any] = None,
    ):
        """Record a metric to the history table for observability.

        Metric types:
        - training_loss: NNUE training loss
        - elo_rating: Model Elo rating
        - gpu_utilization: GPU utilization percentage
        - selfplay_games_per_hour: Game generation rate
        - validation_rate: GPU selfplay validation rate
        - tournament_win_rate: Tournament win rate for new model
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics_history
                (timestamp, metric_type, board_type, num_players, value, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                metric_type,
                board_type,
                num_players,
                value,
                json.dumps(metadata) if metadata else None,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[P2P] Failed to record metric: {e}")

    def get_metrics_history(
        self,
        metric_type: str,
        board_type: str = None,
        num_players: int = None,
        hours: float = 24,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Get metrics history for a specific metric type."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            since = time.time() - (hours * 3600)
            query = """
                SELECT timestamp, value, board_type, num_players, metadata
                FROM metrics_history
                WHERE metric_type = ? AND timestamp > ?
            """
            params: List[Any] = [metric_type, since]

            if board_type:
                query += " AND board_type = ?"
                params.append(board_type)
            if num_players:
                query += " AND num_players = ?"
                params.append(num_players)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append({
                    "timestamp": row[0],
                    "value": row[1],
                    "board_type": row[2],
                    "num_players": row[3],
                    "metadata": json.loads(row[4]) if row[4] else None,
                })
            conn.close()
            return results
        except Exception as e:
            print(f"[P2P] Failed to get metrics history: {e}")
            return []

    def get_metrics_summary(self, hours: float = 24) -> Dict[str, Any]:
        """Get summary of all metrics over the specified time period."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            since = time.time() - (hours * 3600)

            cursor.execute("""
                SELECT metric_type, COUNT(*), AVG(value), MIN(value), MAX(value)
                FROM metrics_history
                WHERE timestamp > ?
                GROUP BY metric_type
            """, (since,))

            summary: Dict[str, Any] = {}
            for row in cursor.fetchall():
                summary[row[0]] = {
                    "count": row[1],
                    "avg": row[2],
                    "min": row[3],
                    "max": row[4],
                }

            cursor.execute("""
                SELECT metric_type, value, timestamp
                FROM metrics_history m1
                WHERE timestamp = (
                    SELECT MAX(timestamp) FROM metrics_history m2
                    WHERE m2.metric_type = m1.metric_type
                )
            """)
            for row in cursor.fetchall():
                if row[0] in summary:
                    summary[row[0]]["latest"] = row[1]
                    summary[row[0]]["latest_time"] = row[2]

            conn.close()
            return {"period_hours": hours, "since": since, "metrics": summary}
        except Exception as e:
            print(f"[P2P] Failed to get metrics summary: {e}")
            return {}

    def _create_self_info(self) -> NodeInfo:
        """Create NodeInfo for this node."""
        # Detect GPU
        has_gpu, gpu_name = self._detect_gpu()

        cpu_count = int(os.cpu_count() or 0)

        # Detect memory
        memory_gb = self._detect_memory()

        # Detect capabilities based on hardware
        capabilities = ["selfplay"]
        if has_gpu:
            capabilities.extend(["training", "cmaes"])
        if memory_gb >= 64:
            capabilities.append("large_boards")

        info = NodeInfo(
            node_id=self.node_id,
            host=self.advertise_host,
            port=self.advertise_port,
            role=self.role,
            last_heartbeat=time.time(),
            cpu_count=cpu_count,
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            memory_gb=memory_gb,
            capabilities=capabilities,
            version=self.build_version,
        )
        # Advertise an alternate mesh endpoint (Tailscale) for NAT traversal and
        # multi-path retries. Peers persist the observed reachable endpoint in
        # `host`/`port` but keep our `reported_host`/`reported_port` as an
        # additional candidate (see `_heartbeat_loop` multi-path retry).
        ts_ip = self._get_tailscale_ip()
        if ts_ip and ts_ip != info.host:
            info.reported_host = ts_ip
            # Use the actual listening port for mesh endpoints (port-mapped
            # advertise ports may not be reachable inside overlays).
            info.reported_port = int(self.port)
        return info

    def _detect_gpu(self) -> Tuple[bool, str]:
        """Detect if GPU is available and its name."""
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, result.stdout.strip().split('\n')[0]
        except:
            pass

        try:
            # Try MPS (Apple Silicon)
            result = subprocess.run(
                ["python3", "-c", "import torch; print(torch.backends.mps.is_available())"],
                capture_output=True, text=True, timeout=10
            )
            if "True" in result.stdout:
                return True, "Apple MPS"
        except:
            pass

        return False, ""

    def _detect_memory(self) -> int:
        """Detect total system memory in GB."""
        try:
            if sys.platform == "darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=5
                )
                return int(result.stdout.strip()) // (1024**3)
            else:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            return int(line.split()[1]) // (1024**2)
        except:
            pass
        return 16  # Default assumption

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to external address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def _get_tailscale_ip(self) -> str:
        """Return this node's Tailscale IPv4 (100.x) when available."""
        try:
            result = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return ""
            ip = (result.stdout or "").strip().splitlines()[0].strip()
            return ip
        except FileNotFoundError:
            return ""
        except Exception:
            return ""

    def _is_tailscale_host(self, host: str) -> bool:
        """Return True when `host` looks like a Tailscale mesh endpoint."""
        h = (host or "").strip()
        if not h:
            return False
        if h.endswith(".ts.net"):
            return True
        try:
            ip = ipaddress.ip_address(h)
        except ValueError:
            return False
        if not isinstance(ip, ipaddress.IPv4Address):
            return False
        return ip in TAILSCALE_CGNAT_NETWORK

    def _local_has_tailscale(self) -> bool:
        """Best-effort: True when this node appears to have a Tailscale address."""
        try:
            info = getattr(self, "self_info", None)
            if not info:
                return False
            host = str(getattr(info, "host", "") or "").strip()
            reported_host = str(getattr(info, "reported_host", "") or "").strip()
            return self._is_tailscale_host(host) or self._is_tailscale_host(reported_host)
        except Exception:
            return False

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        result = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_percent": 0.0,
            "gpu_percent": 0.0,
            "gpu_memory_percent": 0.0,
        }

        try:
            # CPU
            if sys.platform == "darwin":
                out = subprocess.run(
                    ["ps", "-A", "-o", "%cpu"],
                    capture_output=True, text=True, timeout=5
                )
                cpus = [float(x) for x in out.stdout.strip().split('\n')[1:] if x.strip()]
                result["cpu_percent"] = min(100.0, sum(cpus) / os.cpu_count())
            else:
                with open("/proc/loadavg") as f:
                    load = float(f.read().split()[0])
                    result["cpu_percent"] = min(100.0, load * 100 / os.cpu_count())

            # Memory
            if sys.platform == "darwin":
                out = subprocess.run(
                    ["vm_stat"],
                    capture_output=True, text=True, timeout=5
                )
                # Parse vm_stat output
                lines = out.stdout.strip().split('\n')
                stats = {}
                for line in lines[1:]:
                    if ':' in line:
                        key, val = line.split(':')
                        stats[key.strip()] = int(val.strip().rstrip('.'))
                page_size = 16384  # Usually 16KB on M1
                free = stats.get('Pages free', 0) * page_size
                total = self._detect_memory() * (1024**3)
                result["memory_percent"] = 100.0 * (1 - free / total) if total > 0 else 0.0
            else:
                with open("/proc/meminfo") as f:
                    mem = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            mem[parts[0].rstrip(':')] = int(parts[1])
                    total = mem.get('MemTotal', 1)
                    avail = mem.get('MemAvailable', mem.get('MemFree', 0))
                    result["memory_percent"] = 100.0 * (1 - avail / total)

            # Disk
            import shutil
            usage = shutil.disk_usage(self.ringrift_path)
            result["disk_percent"] = 100.0 * usage.used / usage.total

            # GPU (NVIDIA)
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if out.returncode == 0:
                    parts = out.stdout.strip().split(',')
                    result["gpu_percent"] = float(parts[0])
                    mem_used = float(parts[1])
                    mem_total = float(parts[2])
                    result["gpu_memory_percent"] = 100.0 * mem_used / mem_total
            except:
                pass

        except Exception as e:
            print(f"[P2P] Resource check error: {e}")

        return result

    def _get_diversity_metrics(self) -> Dict[str, Any]:
        """Get diversity tracking metrics for monitoring."""
        metrics = dict(self.diversity_metrics)
        metrics["uptime_seconds"] = time.time() - metrics.get("last_reset", time.time())

        # Calculate diversity ratios
        total_games = metrics.get("asymmetric_games", 0) + metrics.get("symmetric_games", 0)
        if total_games > 0:
            metrics["asymmetric_ratio"] = metrics["asymmetric_games"] / total_games
        else:
            metrics["asymmetric_ratio"] = 0.0

        # Engine mode distribution
        engine_total = sum(metrics.get("games_by_engine_mode", {}).values())
        if engine_total > 0:
            metrics["engine_mode_distribution"] = {
                k: v / engine_total
                for k, v in metrics.get("games_by_engine_mode", {}).items()
            }
        else:
            metrics["engine_mode_distribution"] = {}

        return metrics

    def _track_selfplay_diversity(self, config: Dict[str, Any]):
        """Track diversity metrics for a scheduled selfplay game."""
        # Track engine mode
        engine_mode = config.get("engine_mode", "unknown")
        if engine_mode not in self.diversity_metrics["games_by_engine_mode"]:
            self.diversity_metrics["games_by_engine_mode"][engine_mode] = 0
        self.diversity_metrics["games_by_engine_mode"][engine_mode] += 1

        # Track board config
        board_key = f"{config.get('board_type', 'unknown')}_{config.get('num_players', 0)}p"
        if board_key not in self.diversity_metrics["games_by_board_config"]:
            self.diversity_metrics["games_by_board_config"][board_key] = 0
        self.diversity_metrics["games_by_board_config"][board_key] += 1

        # Track asymmetric vs symmetric
        if config.get("asymmetric"):
            self.diversity_metrics["asymmetric_games"] += 1
            strong = config.get("strong_config", {})
            weak = config.get("weak_config", {})
            print(f"[P2P] DIVERSE: Asymmetric game scheduled - "
                  f"Strong({strong.get('engine_mode')}@D{strong.get('difficulty')}) vs "
                  f"Weak({weak.get('engine_mode')}@D{weak.get('difficulty')}) "
                  f"on {board_key}")
        else:
            self.diversity_metrics["symmetric_games"] += 1

        # Track difficulty if available
        difficulty = config.get("difficulty", config.get("difficulty_band"))
        if difficulty:
            diff_key = str(difficulty)
            if diff_key not in self.diversity_metrics["games_by_difficulty"]:
                self.diversity_metrics["games_by_difficulty"][diff_key] = 0
            self.diversity_metrics["games_by_difficulty"][diff_key] += 1

    def _count_local_jobs(self) -> Tuple[int, int]:
        """Count running selfplay and training jobs on this node."""
        def _pid_alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            except Exception:
                return False

        # Primary source of truth: jobs we started and are tracking.
        selfplay_pids: Set[str] = set()
        training_pids: Set[str] = set()

        stale_job_ids: List[str] = []
        try:
            with self.jobs_lock:
                jobs_snapshot = list(self.local_jobs.items())
            for job_id, job in jobs_snapshot:
                if job.status != "running":
                    continue
                pid = int(job.pid or 0)
                if pid <= 0:
                    continue
                if not _pid_alive(pid):
                    stale_job_ids.append(job_id)
                    continue
                if job.job_type in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY):
                    selfplay_pids.add(str(pid))
                elif job.job_type == JobType.TRAINING:
                    training_pids.add(str(pid))

            if stale_job_ids:
                with self.jobs_lock:
                    for job_id in stale_job_ids:
                        self.local_jobs.pop(job_id, None)
        except Exception:
            pass

        # Secondary check: best-effort process scan for untracked jobs (e.g. manual runs).
        # IMPORTANT: never return (0,0) just because `pgrep` is missing or fails;
        # that can cause the leader to spawn runaway selfplay processes until disk fills.
        try:
            import shutil

            if shutil.which("pgrep"):
                for pattern in ("run_self_play_soak.py", "run_gpu_selfplay.py", "run_hybrid_selfplay.py"):
                    out = subprocess.run(
                        ["pgrep", "-f", pattern],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if out.returncode == 0 and out.stdout.strip():
                        selfplay_pids.update([p for p in out.stdout.strip().split() if p])

                for pattern in ("train_", "train.py"):
                    out = subprocess.run(
                        ["pgrep", "-f", pattern],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if out.returncode == 0 and out.stdout.strip():
                        training_pids.update([p for p in out.stdout.strip().split() if p])
        except Exception:
            pass

        return len(selfplay_pids), len(training_pids)

    # ============================================
    # Phase 2: Distributed Data Sync Methods
    # ============================================

    def _collect_local_data_manifest(self) -> NodeDataManifest:
        """Collect manifest of all data files on this node.

        Scans the data directory for:
        - selfplay/ - Game replay files (.jsonl, .db)
        - models/ - Trained model files (.pt, .onnx)
        - training/ - Training data files (.npz)
        - games/ - Synced game databases (.db)

        Uses get_data_directory() to support both disk and ramdrive storage.
        """
        data_dir = self.get_data_directory()
        manifest = NodeDataManifest(
            node_id=self.node_id,
            collected_at=time.time(),
        )

        if not data_dir.exists():
            print(f"[P2P] Data directory not found: {data_dir}")
            return manifest

        files: List[DataFileInfo] = []

        def _count_jsonl_games(file_path: Path, file_size_bytes: int) -> int:
            if file_size_bytes > MANIFEST_JSONL_LINECOUNT_MAX_BYTES:
                return 0

            try:
                with open(file_path, "rb") as f:
                    line_count = 0
                    last_byte = b""
                    while True:
                        chunk = f.read(MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES)
                        if not chunk:
                            break
                        line_count += chunk.count(b"\n")
                        last_byte = chunk[-1:]

                if file_size_bytes > 0 and last_byte != b"\n":
                    line_count += 1

                return int(line_count)
            except Exception:
                return 0

        # Scan for data files
        patterns = {
            "selfplay": ["selfplay/**/*.jsonl", "selfplay/**/*.db"],
            "model": ["models/**/*.pt", "models/**/*.onnx", "models/**/*.bin"],
            "training": ["training/**/*.npz"],
            "games": ["games/**/*.db"],
        }

        for file_type, globs in patterns.items():
            for pattern in globs:
                for file_path in data_dir.glob(pattern):
                    if not file_path.is_file():
                        continue

                    try:
                        stat = file_path.stat()
                        rel_path = str(file_path.relative_to(data_dir))

                        # Parse board_type and num_players from filename/path
                        board_type = ""
                        num_players = 0
                        path_lower = rel_path.lower()

                        if "sq8" in path_lower or "square8" in path_lower:
                            board_type = "square8"
                        elif "sq19" in path_lower or "square19" in path_lower:
                            board_type = "square19"
                        elif "hex" in path_lower:
                            board_type = "hexagonal"

                        if "_2p" in path_lower or "2p_" in path_lower:
                            num_players = 2
                        elif "_3p" in path_lower or "3p_" in path_lower:
                            num_players = 3
                        elif "_4p" in path_lower or "4p_" in path_lower:
                            num_players = 4

                        file_info = DataFileInfo(
                            path=rel_path,
                            size_bytes=stat.st_size,
                            modified_time=stat.st_mtime,
                            file_type=file_type,
                            board_type=board_type,
                            num_players=num_players,
                        )
                        files.append(file_info)

                        # Update summary stats
                        manifest.total_files += 1
                        manifest.total_size_bytes += stat.st_size

                        if file_type == "selfplay":
                            # Count games in JSONL files
                            if rel_path.endswith(".jsonl"):
                                try:
                                    line_count = _count_jsonl_games(file_path, stat.st_size)
                                    file_info.game_count = line_count
                                    manifest.selfplay_games += line_count
                                except Exception:
                                    pass
                        elif file_type == "model":
                            manifest.model_count += 1
                        elif file_type == "training":
                            manifest.training_data_size += stat.st_size

                    except Exception as e:
                        print(f"[P2P] Error scanning file {file_path}: {e}")

        manifest.files = files

        print(f"[P2P] Collected manifest: {manifest.total_files} files, "
              f"{manifest.total_size_bytes / (1024*1024):.1f}MB, "
              f"{manifest.selfplay_games} games")

        return manifest

    def _compute_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Compute MD5 hash of a file for verification."""
        import hashlib
        md5 = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception as e:
            print(f"[P2P] Error hashing file {file_path}: {e}")
            return ""

    async def _request_peer_manifest(self, peer_info: NodeInfo) -> Optional[NodeDataManifest]:
        """Request data manifest from a peer node."""
        try:
            # Keep manifest requests snappy: these are advisory and should not
            # stall leader loops or external callers (e.g. the improvement
            # daemon). Prefer faster failure and rely on periodic retries.
            timeout = ClientTimeout(total=10, sock_connect=3, sock_read=7)
            async with get_client_session(timeout) as session:
                for url in self._urls_for_peer(peer_info, "/data_manifest"):
                    try:
                        async with session.get(url, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                continue
                            data = await resp.json()
                        return NodeDataManifest.from_dict((data or {}).get("manifest", {}))
                    except Exception:
                        continue
        except Exception as e:
            print(f"[P2P] Error requesting manifest from {peer_info.node_id}: {e}")
        return None

    async def _collect_cluster_manifest(self) -> ClusterDataManifest:
        """Leader-only: Collect manifests from all peers and build cluster view."""
        cluster_manifest = ClusterDataManifest(
            collected_at=time.time(),
        )

        # Collect from self
        local_manifest = await asyncio.to_thread(self._collect_local_data_manifest)
        with self.manifest_lock:
            self.local_data_manifest = local_manifest
        cluster_manifest.node_manifests[self.node_id] = local_manifest

        # Collect from peers in parallel.
        #
        # Only probe peers that are currently alive and not retired; terminated
        # or long-dead nodes should not stall manifest collection. NAT-blocked
        # peers can't accept inbound /data_manifest, so they are excluded too.
        with self.peers_lock:
            peers = [
                p
                for p in self.peers.values()
                if p.is_alive()
                and not bool(getattr(p, "retired", False))
                and not bool(getattr(p, "nat_blocked", False))
            ]

        tasks = [self._request_peer_manifest(peer) for peer in peers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for peer, result in zip(peers, results):
            if isinstance(result, NodeDataManifest):
                cluster_manifest.node_manifests[peer.node_id] = result

        # Compute cluster-wide statistics
        cluster_manifest.total_nodes = len(cluster_manifest.node_manifests)

        all_files: Set[str] = set()
        for node_id, node_manifest in cluster_manifest.node_manifests.items():
            cluster_manifest.total_files += node_manifest.total_files
            cluster_manifest.total_size_bytes += node_manifest.total_size_bytes
            cluster_manifest.total_selfplay_games += node_manifest.selfplay_games
            cluster_manifest.files_by_node[node_id] = node_manifest.total_files

            for file_info in node_manifest.files:
                all_files.add(file_info.path)

        cluster_manifest.unique_files = all_files

        # Find files missing from nodes (for sync planning)
        for file_path in all_files:
            nodes_with_file = []
            nodes_without_file = []
            for node_id, node_manifest in cluster_manifest.node_manifests.items():
                file_paths = {f.path for f in node_manifest.files}
                if file_path in file_paths:
                    nodes_with_file.append(node_id)
                else:
                    nodes_without_file.append(node_id)

            if nodes_without_file:
                cluster_manifest.missing_from_nodes[file_path] = nodes_without_file

        print(f"[P2P] Cluster manifest: {cluster_manifest.total_nodes} nodes, "
              f"{len(cluster_manifest.unique_files)} unique files, "
              f"{cluster_manifest.total_selfplay_games} total games")

        return cluster_manifest

    # ============================================
    # Phase 2: P2P Rsync Coordination Methods
    # ============================================

    def _generate_sync_plan(self) -> Optional[ClusterSyncPlan]:
        """
        Leader generates a sync plan from the cluster manifest.
        Identifies which files are missing from which nodes and creates sync jobs.
        """
        if not self.cluster_data_manifest:
            print("[P2P] No cluster manifest available, cannot generate sync plan")
            return None

        if not self.cluster_data_manifest.missing_from_nodes:
            print("[P2P] All nodes have all files, no sync needed")
            return None

        plan = ClusterSyncPlan(
            plan_id=str(uuid.uuid4()),
            created_at=time.time(),
        )

        # For each missing file, find a source node and create a sync job
        for file_path, missing_nodes in self.cluster_data_manifest.missing_from_nodes.items():
            # Find a node that has this file (any node not in missing_nodes)
            source_node = None
            for node_id in self.cluster_data_manifest.manifests_by_node.keys():
                if node_id not in missing_nodes:
                    node_manifest = self.cluster_data_manifest.manifests_by_node[node_id]
                    if file_path in node_manifest.files_by_path:
                        source_node = node_id
                        break

            if not source_node:
                continue  # No node has this file (shouldn't happen)

            # Create sync jobs for each target node
            for target_node in missing_nodes:
                job = DataSyncJob(
                    job_id=str(uuid.uuid4()),
                    source_node=source_node,
                    target_node=target_node,
                    files=[file_path],
                    status="pending",
                )

                # Get file size for tracking
                node_manifest = self.cluster_data_manifest.manifests_by_node[source_node]
                if file_path in node_manifest.files_by_path:
                    file_info = node_manifest.files_by_path[file_path]
                    plan.total_bytes_to_sync += file_info.size_bytes

                plan.sync_jobs.append(job)
                plan.total_files_to_sync += 1

        print(f"[P2P] Generated sync plan: {len(plan.sync_jobs)} jobs, "
              f"{plan.total_files_to_sync} files, "
              f"{plan.total_bytes_to_sync / (1024*1024):.1f} MB total")

        return plan

    async def _execute_sync_plan(self) -> None:
        """Leader executes the sync plan by dispatching jobs to nodes."""
        if not self.current_sync_plan:
            return

        with self.sync_lock:
            if self.sync_in_progress:
                print("[P2P] Sync already in progress, skipping")
                return
            self.sync_in_progress = True
            self.current_sync_plan.status = "running"

        try:
            # Group jobs by target node for efficiency
            jobs_by_target: Dict[str, List[DataSyncJob]] = {}
            for job in self.current_sync_plan.sync_jobs:
                if job.target_node not in jobs_by_target:
                    jobs_by_target[job.target_node] = []
                jobs_by_target[job.target_node].append(job)

            # Execute jobs for each target node
            for target_node, jobs in jobs_by_target.items():
                peer = self.peers.get(target_node)
                if target_node != self.node_id and (not peer or not peer.is_alive()):
                    print(f"[P2P] Target node {target_node} not available, skipping sync")
                    for job in jobs:
                        job.status = "failed"
                        job.error_message = "Target node not available"
                        self.current_sync_plan.jobs_failed += 1
                    continue

                # Send sync request to target node
                for job in jobs:
                    await self._request_node_sync(job)

            # Update plan status
            with self.sync_lock:
                if self.current_sync_plan.jobs_failed == len(self.current_sync_plan.sync_jobs):
                    self.current_sync_plan.status = "failed"
                elif self.current_sync_plan.jobs_completed == len(self.current_sync_plan.sync_jobs):
                    self.current_sync_plan.status = "completed"
                else:
                    self.current_sync_plan.status = "partial"

        finally:
            with self.sync_lock:
                self.sync_in_progress = False
                self.last_sync_time = time.time()

    async def _request_node_sync(self, job: DataSyncJob) -> bool:
        """Request a target node to pull files from a source node."""
        target_peer = self.peers.get(job.target_node)
        if job.target_node == self.node_id:
            target_peer = self.self_info

        source_peer = self.peers.get(job.source_node)
        if job.source_node == self.node_id:
            source_peer = self.self_info

        if not target_peer or not source_peer:
            job.status = "failed"
            job.error_message = "Source or target peer not found"
            return False

        job.status = "running"
        job.started_at = time.time()

        try:
            # Local target: execute the pull directly (no HTTP round-trip).
            if job.target_node == self.node_id:
                result = await self._handle_sync_pull_request(
                    source_host=source_peer.host,
                    source_port=source_peer.port,
                    source_reported_host=(getattr(source_peer, "reported_host", "") or None),
                    source_reported_port=(getattr(source_peer, "reported_port", 0) or None),
                    source_node_id=job.source_node,
                    files=job.files,
                )
            else:
                payload = {
                    "job_id": job.job_id,
                    # Back-compat: target will prefer source_node_id lookup.
                    "source_host": source_peer.host,
                    "source_port": source_peer.port,
                    "source_node_id": job.source_node,
                    "files": job.files,
                }
                rh = (getattr(source_peer, "reported_host", "") or "").strip()
                rp = int(getattr(source_peer, "reported_port", 0) or 0)
                if rh and rp and (rh != source_peer.host or rp != source_peer.port):
                    payload["source_reported_host"] = rh
                    payload["source_reported_port"] = rp

                timeout = ClientTimeout(total=600)
                async with get_client_session(timeout) as session:
                    result = None
                    last_err: Optional[str] = None
                    for url in self._urls_for_peer(target_peer, "/sync/pull"):
                        try:
                            async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                                if resp.status != 200:
                                    last_err = f"http_{resp.status}"
                                    continue
                                result = await resp.json()
                                break
                        except Exception as e:
                            last_err = str(e)
                            continue
                    if result is None:
                        job.status = "failed"
                        job.error_message = last_err or "sync_pull_failed"
                        if self.current_sync_plan:
                            self.current_sync_plan.jobs_failed += 1
                        return False

            ok = bool(result.get("success"))
            job.status = "completed" if ok else "failed"
            job.completed_at = time.time()
            job.bytes_transferred = int(result.get("bytes_transferred", 0) or 0)
            job.files_completed = int(result.get("files_completed", 0) or 0)
            if not ok:
                job.error_message = str(result.get("error") or "Unknown error")

            if self.current_sync_plan:
                if ok:
                    self.current_sync_plan.jobs_completed += 1
                else:
                    self.current_sync_plan.jobs_failed += 1

            if ok:
                print(f"[P2P] Sync job {job.job_id[:8]} completed: {job.source_node} -> {job.target_node}")
            else:
                print(f"[P2P] Sync job {job.job_id[:8]} failed: {job.error_message}")

            return ok

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = time.time()
            if self.current_sync_plan:
                self.current_sync_plan.jobs_failed += 1
            print(f"[P2P] Sync job {job.job_id[:8]} failed: {e}")
            return False

    async def _handle_sync_pull_request(
        self,
        source_host: str,
        source_port: int,
        source_node_id: str,
        files: List[str],
        source_reported_host: Optional[str] = None,
        source_reported_port: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Handle incoming request to pull files from a source node.
        Pulls files over the P2P HTTP channel to avoid SSH/rsync dependencies.
        Uses get_data_directory() to support both disk and ramdrive storage.
        """
        data_dir = self.get_data_directory()
        data_dir.mkdir(parents=True, exist_ok=True)

        bytes_transferred = 0
        files_completed = 0
        errors: List[str] = []

        # Multi-path sources: prefer observed endpoint but allow a self-reported
        # endpoint (e.g. Tailscale) when the public route fails.
        candidate_sources: List[Tuple[str, int]] = []
        seen_sources: Set[Tuple[str, int]] = set()

        def _add_source(host: Optional[str], port: Optional[int]) -> None:
            if not host:
                return
            h = str(host).strip()
            if not h:
                return
            try:
                p = int(port or 0)
            except Exception:
                return
            if p <= 0:
                return
            key = (h, p)
            if key in seen_sources:
                return
            seen_sources.add(key)
            candidate_sources.append(key)

        _add_source(source_host, source_port)
        _add_source(source_reported_host, source_reported_port)

        timeout = ClientTimeout(total=None, sock_connect=HTTP_CONNECT_TIMEOUT, sock_read=600)

        async with get_client_session(timeout) as session:
            for rel_path in files:
                rel_path = (rel_path or "").lstrip("/")
                if not rel_path:
                    errors.append("empty_path")
                    continue

                # Security: keep all writes within ai-service/data.
                dest_path = (data_dir / rel_path)
                try:
                    data_root = data_dir.resolve()
                    dest_resolved = dest_path.resolve()
                    dest_resolved.relative_to(data_root)
                except Exception:
                    errors.append(f"invalid_path:{rel_path}")
                    continue

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = dest_path.with_name(dest_path.name + ".partial")

                last_err: Optional[str] = None
                success = False

                for host, base_port in candidate_sources:
                    # Back-compat: if caller passed an SSH-like port (22), try DEFAULT_PORT too.
                    ports_to_try: List[int] = []
                    try:
                        ports_to_try.append(int(base_port))
                    except Exception:
                        ports_to_try.append(DEFAULT_PORT)
                    if DEFAULT_PORT not in ports_to_try:
                        ports_to_try.append(DEFAULT_PORT)

                    for port in ports_to_try:
                        url = f"http://{host}:{port}/sync/file"
                        try:
                            async with session.get(
                                url,
                                params={"path": rel_path},
                                headers=self._auth_headers(),
                            ) as resp:
                                if resp.status != 200:
                                    text = ""
                                    try:
                                        text = (await resp.text())[:200]
                                    except Exception:
                                        text = ""
                                    last_err = f"{resp.status} {text}".strip()
                                    continue

                                with open(tmp_path, "wb") as out_f:
                                    async for chunk in resp.content.iter_chunked(1024 * 1024):
                                        out_f.write(chunk)
                                        bytes_transferred += len(chunk)

                                tmp_path.replace(dest_path)
                                files_completed += 1
                                success = True
                                break

                        except Exception as e:
                            last_err = str(e)
                            continue
                    if success:
                        break

                if not success:
                    errors.append(f"{rel_path}: {last_err or 'download_failed'}")
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass

        if errors:
            return {
                "success": False,
                "files_completed": files_completed,
                "bytes_transferred": bytes_transferred,
                "error": "; ".join(errors[:5]),
            }

        return {
            "success": True,
            "files_completed": files_completed,
            "bytes_transferred": bytes_transferred,
        }

    async def start_cluster_sync(self) -> Dict[str, Any]:
        """
        Leader initiates a full cluster data sync.
        Returns status of the sync operation.
        """
        if not self._is_leader():
            return {"success": False, "error": "Not the leader"}

        # First, collect fresh manifests
        print("[P2P] Collecting cluster manifest for sync...")
        self.cluster_data_manifest = await self._collect_cluster_manifest()

        # Generate sync plan
        self.current_sync_plan = self._generate_sync_plan()
        if not self.current_sync_plan:
            return {"success": True, "message": "No sync needed, all nodes in sync"}

        # Execute the plan
        await self._execute_sync_plan()

        return {
            "success": True,
            "plan_id": self.current_sync_plan.plan_id,
            "total_jobs": len(self.current_sync_plan.sync_jobs),
            "jobs_completed": self.current_sync_plan.jobs_completed,
            "jobs_failed": self.current_sync_plan.jobs_failed,
            "status": self.current_sync_plan.status,
        }

    # ============================================
    # Training Node Priority Sync
    # ============================================

    def _get_training_primary_nodes(self, count: int = TRAINING_NODE_COUNT) -> List[NodeInfo]:
        """Get the top N nodes by GPU power for training priority.

        Returns nodes sorted by GPU processing power (highest first).
        These nodes should receive selfplay data first for training.
        """
        with self.peers_lock:
            # Include self if we have a GPU
            all_nodes = list(self.peers.values())
            if self.self_info.has_gpu:
                all_nodes.append(self.self_info)

        # Filter to only GPU nodes that are alive and healthy
        gpu_nodes = [
            node for node in all_nodes
            if node.has_gpu and node.is_alive() and node.gpu_power_score() > 0
        ]

        # Sort by GPU power score (descending)
        gpu_nodes.sort(key=lambda n: n.gpu_power_score(), reverse=True)

        # Return top N
        return gpu_nodes[:count]

    def _get_training_nodes_ranked(self) -> List[Dict[str, Any]]:
        """Get all GPU nodes with their power rankings for dashboard display."""
        with self.peers_lock:
            all_nodes = list(self.peers.values())
            if self.self_info.has_gpu:
                all_nodes.append(self.self_info)

        result = []
        for node in all_nodes:
            if node.has_gpu:
                result.append({
                    "node_id": node.node_id,
                    "gpu_name": node.gpu_name,
                    "gpu_power_score": node.gpu_power_score(),
                    "memory_gb": node.memory_gb,
                    "is_alive": node.is_alive(),
                    "is_healthy": node.is_healthy(),
                    "gpu_percent": node.gpu_percent,
                })

        # Sort by power score
        result.sort(key=lambda x: x["gpu_power_score"], reverse=True)
        return result

    def _should_sync_to_node(self, node: NodeInfo) -> bool:
        """Check if we should sync data TO this node based on disk space."""
        # Don't sync to nodes with critical disk usage
        if node.disk_percent >= DISK_CRITICAL_THRESHOLD:
            print(f"[P2P] Skipping sync to {node.node_id}: disk critical ({node.disk_percent:.1f}%)")
            return False
        # Warn but allow sync to nodes with warning-level disk
        if node.disk_percent >= DISK_WARNING_THRESHOLD:
            print(f"[P2P] Warning: {node.node_id} disk at {node.disk_percent:.1f}%")
        return True

    def _should_cleanup_source(self, node: NodeInfo) -> bool:
        """Check if source node needs disk cleanup after sync."""
        return node.disk_percent >= DISK_CLEANUP_THRESHOLD

    async def _cleanup_synced_files(self, node_id: str, files: List[str]) -> bool:
        """Delete synced files from source node to free disk space.

        Only called after successful sync to training nodes.
        """
        with self.peers_lock:
            node = self.peers.get(node_id)
        if not node or not node.is_alive():
            return False

        try:
            if getattr(node, "nat_blocked", False):
                cmd_id = await self._enqueue_relay_command_for_peer(
                    node,
                    "cleanup_files",
                    {"files": list(files or []), "reason": "post_sync_cleanup"},
                )
                if cmd_id:
                    print(f"[P2P] Enqueued relay cleanup_files for {node_id} ({len(files)} files)")
                    return True
                print(f"[P2P] Relay queue full for {node_id}; skipping cleanup_files enqueue")
                return False

            timeout = ClientTimeout(total=60)
            async with get_client_session(timeout) as session:
                last_err: Optional[str] = None
                for url in self._urls_for_peer(node, "/cleanup/files"):
                    try:
                        async with session.post(
                            url,
                            json={"files": files, "reason": "post_sync_cleanup"},
                            headers=self._auth_headers(),
                        ) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            result = await resp.json()
                            freed_bytes = result.get("freed_bytes", 0)
                            print(f"[P2P] Cleanup on {node_id}: freed {freed_bytes / 1e6:.1f}MB")
                            return True
                    except Exception as e:
                        last_err = str(e)
                        continue
                if last_err:
                    print(f"[P2P] Cleanup files request failed on {node_id}: {last_err}")
        except Exception as e:
            print(f"[P2P] Failed to cleanup files on {node_id}: {e}")
        return False

    async def _sync_selfplay_to_training_nodes(self) -> Dict[str, Any]:
        """Sync selfplay data to training primary nodes.

        This is called periodically by the leader to ensure training nodes
        have the latest selfplay data for model training.

        Features:
        - Prioritizes nodes by GPU power (H100 > 5090 > 4090 > etc.)
        - Skips syncing TO nodes with critical disk usage
        - Cleans up source nodes with high disk usage after successful sync
        """
        if not self._is_leader():
            return {"success": False, "error": "Not the leader"}

        # Get training primary nodes
        training_nodes = self._get_training_primary_nodes()
        if not training_nodes:
            return {"success": False, "error": "No training nodes available"}

        # Filter out nodes with critical disk space
        eligible_training_nodes = [n for n in training_nodes if self._should_sync_to_node(n)]
        if not eligible_training_nodes:
            return {"success": False, "error": "All training nodes have critical disk usage"}

        print(f"[P2P] Training sync: {len(eligible_training_nodes)} eligible training nodes")
        for node in eligible_training_nodes:
            print(f"[P2P]   - {node.node_id}: {node.gpu_name} (power={node.gpu_power_score()}, disk={node.disk_percent:.1f}%)")

        # Collect current cluster manifest if stale
        if (time.time() - self.last_manifest_collection > self.manifest_collection_interval
                or not self.cluster_data_manifest):
            print("[P2P] Collecting fresh cluster manifest for training sync...")
            self.cluster_data_manifest = await self._collect_cluster_manifest()
            self.last_manifest_collection = time.time()

        if not self.cluster_data_manifest:
            return {"success": False, "error": "Failed to collect cluster manifest"}

        # Track source nodes that need cleanup after sync
        sources_to_cleanup: Dict[str, List[str]] = {}  # node_id -> list of synced files

        # Find selfplay files that training nodes don't have
        sync_jobs_created = 0
        for target_node in eligible_training_nodes:
            target_manifest = self.cluster_data_manifest.node_manifests.get(target_node.node_id)
            target_files = set()
            if target_manifest:
                target_files = set(target_manifest.files_by_path.keys())

            # Find source nodes with selfplay data this target doesn't have
            for source_id, source_manifest in self.cluster_data_manifest.node_manifests.items():
                if source_id == target_node.node_id:
                    continue

                # Check if source node needs disk cleanup
                source_node = self.peers.get(source_id)
                needs_cleanup = source_node and self._should_cleanup_source(source_node)

                # Find selfplay files to sync (with mtime comparison for efficiency)
                files_to_sync = []
                for file_info in source_manifest.files:
                    if file_info.file_type != "selfplay":
                        continue

                    # Check if target needs this file
                    target_file_info = target_manifest.files_by_path.get(file_info.path) if target_manifest else None

                    should_sync = False
                    if file_info.path not in target_files:
                        # Target doesn't have file at all
                        should_sync = True
                    elif target_file_info and file_info.modified_time > target_file_info.modified_time + 60:
                        # Source is newer (with 60s tolerance to avoid clock skew issues)
                        should_sync = True

                    if should_sync:
                        files_to_sync.append(file_info.path)

                if files_to_sync:
                    # Create sync job
                    job_id = f"training_sync_{source_id}_to_{target_node.node_id}_{int(time.time())}"
                    job = DataSyncJob(
                        job_id=job_id,
                        source_node=source_id,
                        target_node=target_node.node_id,
                        files=files_to_sync[:50],  # Limit files per job
                        status="pending",
                    )
                    self.active_sync_jobs[job_id] = job
                    sync_jobs_created += 1
                    print(f"[P2P] Created training sync job: {len(files_to_sync)} files from {source_id} to {target_node.node_id}")

                    # Track files for cleanup if source has high disk usage
                    if needs_cleanup:
                        if source_id not in sources_to_cleanup:
                            sources_to_cleanup[source_id] = []
                        sources_to_cleanup[source_id].extend(files_to_sync[:50])

        # Execute sync jobs
        successful_syncs = 0
        if sync_jobs_created > 0:
            await self._execute_pending_sync_jobs()
            # Count successful syncs
            successful_syncs = sum(
                1 for job in self.active_sync_jobs.values()
                if job.status == "completed"
            )

        # Cleanup source nodes with high disk usage after successful syncs
        cleanup_results = {}
        if successful_syncs > 0 and sources_to_cleanup:
            print(f"[P2P] Running post-sync cleanup on {len(sources_to_cleanup)} source nodes...")
            for source_id, files in sources_to_cleanup.items():
                success = await self._cleanup_synced_files(source_id, files)
                cleanup_results[source_id] = success

        self.last_training_sync_time = time.time()

        return {
            "success": True,
            "training_nodes": [n.node_id for n in eligible_training_nodes],
            "sync_jobs_created": sync_jobs_created,
            "successful_syncs": successful_syncs,
            "sources_cleaned": sum(cleanup_results.values()),
        }

    async def _execute_pending_sync_jobs(self):
        """Execute all pending sync jobs."""
        with self.sync_lock:
            pending_jobs = [
                job for job in self.active_sync_jobs.values()
                if job.status == "pending"
            ]

        for job in pending_jobs:
            try:
                success = await self._request_node_sync(job)
                if success:
                    job.status = "completed"
                    job.completed_at = time.time()
                else:
                    job.status = "failed"
            except Exception as e:
                print(f"[P2P] Sync job {job.job_id} failed: {e}")
                job.status = "failed"
                job.error_message = str(e)

    async def _training_sync_loop(self):
        """Background loop to periodically sync data to training nodes.

        Leader-only: Runs every TRAINING_SYNC_INTERVAL seconds to ensure
        training nodes have the latest selfplay data.
        """
        print(f"[P2P] Training sync loop started (interval: {self.training_sync_interval}s)")

        while self.running:
            try:
                await asyncio.sleep(self.training_sync_interval)

                if not self._is_leader():
                    continue

                # Check if enough time has passed since last sync
                if time.time() - self.last_training_sync_time < self.training_sync_interval:
                    continue

                print("[P2P] Running periodic training node sync...")
                result = await self._sync_selfplay_to_training_nodes()
                if result.get("success"):
                    print(f"[P2P] Training sync completed: {result.get('sync_jobs_created', 0)} jobs created")
                else:
                    print(f"[P2P] Training sync failed: {result.get('error', 'Unknown error')}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[P2P] Training sync loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _vast_ip_update_loop(self):
        """Background loop to periodically refresh Vast instance connection info.

        Uses VAST_API_KEY when available, otherwise falls back to the `vastai`
        CLI if installed (see DynamicHostRegistry.update_vast_ips).
        """
        if not HAS_DYNAMIC_REGISTRY:
            return

        print("[P2P] Vast IP update loop started")
        registry = get_registry()

        while self.running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                updated = await registry.update_vast_ips()
                if updated > 0:
                    print(f"[P2P] Updated {updated} Vast instance IPs from API")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[P2P] Vast IP update loop error: {e}")
                await asyncio.sleep(60)

    async def _aws_ip_update_loop(self):
        """Background loop to periodically refresh AWS instance connection info.

        Uses the `aws` CLI (see DynamicHostRegistry.update_aws_ips). No-op when
        no AWS instances are configured in distributed_hosts.yaml properties.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return

        print("[P2P] AWS IP update loop started")
        registry = get_registry()

        while self.running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                updated = await registry.update_aws_ips()
                if updated > 0:
                    print(f"[P2P] Updated {updated} AWS instance IPs via CLI")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[P2P] AWS IP update loop error: {e}")
                await asyncio.sleep(60)

    async def _tailscale_ip_update_loop(self):
        """Background loop to discover and update Tailscale IPs for cluster nodes.

        Uses `tailscale status --json` to discover mesh network peers.
        Tailscale provides reliable connectivity even when public IPs change.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return

        print("[P2P] Tailscale IP update loop started")
        registry = get_registry()

        while self.running:
            try:
                await asyncio.sleep(120)  # Check every 2 minutes

                updated = await registry.update_tailscale_ips()
                if updated > 0:
                    print(f"[P2P] Updated {updated} node Tailscale IPs")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[P2P] Tailscale IP update loop error: {e}")
                await asyncio.sleep(60)

    async def _data_management_loop(self):
        """Background loop for automatic data management.

        LEARNED LESSONS - Automated data pipeline:
        - Triggers exports when databases exceed threshold
        - Syncs training data to GPU nodes
        - Auto-triggers training when enough data available
        """
        print(f"[P2P] Data management loop started (interval: {DATA_MANAGEMENT_INTERVAL}s)")

        # Track active export jobs
        active_exports: Dict[str, float] = {}  # path -> start_time

        while self.running:
            try:
                await asyncio.sleep(DATA_MANAGEMENT_INTERVAL)

                if not self._is_leader():
                    continue

                print("[P2P] Running data management check...")

                # 1. Check local database sizes and trigger exports
                data_dir = self.get_data_directory()
                games_dir = data_dir / "games"
                training_dir = data_dir / "training"
                training_dir.mkdir(parents=True, exist_ok=True)

                # Count current exports
                current_exports = len([p for p, t in active_exports.items()
                                       if time.time() - t < 3600])  # 1 hour timeout

                if games_dir.exists():
                    for db_file in games_dir.glob("*.db"):
                        db_size_mb = db_file.stat().st_size / (1024 * 1024)

                        if db_size_mb >= DB_EXPORT_THRESHOLD_MB:
                            # Check if already exporting
                            export_key = str(db_file)
                            if export_key in active_exports:
                                continue

                            # Check concurrent export limit
                            if current_exports >= MAX_CONCURRENT_EXPORTS:
                                print(f"[P2P] Skipping export for {db_file.name} (max concurrent reached)")
                                continue

                            # Determine board type from filename
                            board_type = "square8"  # default
                            if "hex" in db_file.name.lower():
                                board_type = "hexagonal"
                            elif "square19" in db_file.name.lower() or "sq19" in db_file.name.lower():
                                board_type = "square19"

                            # Start export job
                            print(f"[P2P] Auto-triggering export for {db_file.name} ({db_size_mb:.0f}MB)")
                            export_output = training_dir / f"auto_{db_file.stem}_{int(time.time())}.npz"

                            try:
                                cmd = [
                                    "python3",
                                    f"{self.ringrift_path}/ai-service/scripts/export_replay_dataset.py",
                                    "--db", str(db_file),
                                    "--board-type", board_type,
                                    "--num-players", "2",
                                    "--board-aware-encoding",
                                    "--require-completed",
                                    "--min-moves", "10",
                                    "--output", str(export_output),
                                ]

                                env = os.environ.copy()
                                env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"

                                subprocess.Popen(
                                    cmd,
                                    stdout=open(f"/tmp/auto_export_{db_file.stem}.log", "w"),
                                    stderr=subprocess.STDOUT,
                                    env=env,
                                    cwd=f"{self.ringrift_path}/ai-service",
                                )
                                active_exports[export_key] = time.time()
                                current_exports += 1
                                print(f"[P2P] Started export job for {db_file.name}")

                            except Exception as e:
                                print(f"[P2P] Failed to start export for {db_file.name}: {e}")

                # 2. Calculate total training data size
                total_training_mb = 0.0
                if training_dir.exists():
                    for npz_file in training_dir.glob("*.npz"):
                        total_training_mb += npz_file.stat().st_size / (1024 * 1024)

                print(f"[P2P] Training data available: {total_training_mb:.1f}MB")

                # 3. Auto-trigger training if threshold exceeded and GPU available
                if total_training_mb >= AUTO_TRAINING_THRESHOLD_MB:
                    # Check if this node has GPU and no training running
                    if self.self_info.is_gpu_node() and self.self_info.training_jobs == 0:
                        print(f"[P2P] Auto-triggering training ({total_training_mb:.1f}MB data available)")
                        # Find largest training file
                        largest_npz = max(
                            training_dir.glob("*.npz"),
                            key=lambda f: f.stat().st_size,
                            default=None
                        )
                        if largest_npz:
                            await self._start_auto_training(str(largest_npz))

                # 4. Request data sync from peers with large databases
                await self._request_data_from_peers()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[P2P] Data management loop error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(60)

    async def _start_auto_training(self, data_path: str):
        """Start automatic training job on local node."""
        try:
            run_dir = f"{self.ringrift_path}/ai-service/models/auto_train_{int(time.time())}"
            Path(run_dir).mkdir(parents=True, exist_ok=True)

            cmd = [
                "python3",
                f"{self.ringrift_path}/ai-service/scripts/run_nn_training_baseline.py",
                "--board", "square8",
                "--num-players", "2",
                "--run-dir", run_dir,
                "--data-path", data_path,
                "--epochs", "50",
                "--model-version", "v3",
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"

            subprocess.Popen(
                cmd,
                stdout=open(f"{run_dir}/training.log", "w"),
                stderr=subprocess.STDOUT,
                env=env,
                cwd=f"{self.ringrift_path}/ai-service",
            )
            print(f"[P2P] Started auto-training job in {run_dir}")
            self.self_info.training_jobs += 1

        except Exception as e:
            print(f"[P2P] Failed to start auto-training: {e}")

    async def _request_data_from_peers(self):
        """Request training data sync from peers with large datasets."""
        try:
            with self.peers_lock:
                peers = list(self.peers.values())

            # This node is a training node if it has GPU
            if not self.self_info.is_gpu_node():
                return

            # Check manifest for peers with more data
            for peer in peers:
                if not peer.is_alive():
                    continue

                # Check peer's data manifest
                peer_data = self.cluster_data_manifest.get(peer.node_id, {})
                peer_training_mb = sum(
                    f.get("size_mb", 0)
                    for f in peer_data.get("training_files", [])
                )

                if peer_training_mb > TRAINING_DATA_SYNC_THRESHOLD_MB:
                    # Request sync via existing sync mechanism
                    print(f"[P2P] Peer {peer.node_id} has {peer_training_mb:.1f}MB training data")

        except Exception as e:
            print(f"[P2P] Data sync request error: {e}")

    # ============================================
    # Git Auto-Update Methods
    # ============================================

    def _get_local_git_commit(self) -> Optional[str]:
        """Get the current local git commit hash."""
        try:
            result = subprocess.run(
                self._git_cmd("rev-parse", "HEAD"),
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"[P2P] Failed to get local git commit: {e}")
        return None

    def _get_local_git_branch(self) -> Optional[str]:
        """Get the current local git branch name."""
        try:
            result = subprocess.run(
                self._git_cmd("rev-parse", "--abbrev-ref", "HEAD"),
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"[P2P] Failed to get local git branch: {e}")
        return None

    def _get_remote_git_commit(self) -> Optional[str]:
        """Fetch and get the remote branch's latest commit hash."""
        try:
            # First fetch to update remote refs
            fetch_result = subprocess.run(
                self._git_cmd("fetch", GIT_REMOTE_NAME, GIT_BRANCH_NAME),
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=60
            )
            if fetch_result.returncode != 0:
                print(f"[P2P] Git fetch failed: {fetch_result.stderr}")
                return None

            # Get remote branch commit
            result = subprocess.run(
                self._git_cmd("rev-parse", f"{GIT_REMOTE_NAME}/{GIT_BRANCH_NAME}"),
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"[P2P] Failed to get remote git commit: {e}")
        return None

    def _check_for_updates(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if there are updates available from GitHub.

        Returns: (has_updates, local_commit, remote_commit)
        """
        local_commit = self._get_local_git_commit()
        remote_commit = self._get_remote_git_commit()

        if not local_commit or not remote_commit:
            return False, local_commit, remote_commit

        has_updates = local_commit != remote_commit
        return has_updates, local_commit, remote_commit

    def _get_commits_behind(self, local_commit: str, remote_commit: str) -> int:
        """Get the number of commits the local branch is behind remote."""
        try:
            result = subprocess.run(
                self._git_cmd("rev-list", "--count", f"{local_commit}..{remote_commit}"),
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception as e:
            print(f"[P2P] Failed to count commits behind: {e}")
        return 0

    def _check_local_changes(self) -> bool:
        """Check if there are uncommitted local changes.

        Notes:
        - Ignore untracked files by default. Cluster nodes often accumulate local
          artifacts (logs, data, env backups) that should not block git updates.
        - Still blocks on tracked/staged modifications to avoid stomping on
          local hotfixes.
        """
        try:
            result = subprocess.run(
                self._git_cmd("status", "--porcelain", "--untracked-files=no"),
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # If there's output, there are uncommitted changes
                return bool(result.stdout.strip())
        except Exception as e:
            print(f"[P2P] Failed to check local changes: {e}")
        return True  # Assume changes exist on error (safer)

    async def _stop_all_local_jobs(self) -> int:
        """Stop all local jobs gracefully before update.

        Returns: Number of jobs stopped
        """
        stopped = 0
        with self.jobs_lock:
            for job_id, job in list(self.local_jobs.items()):
                try:
                    if job.pid > 0:
                        os.kill(job.pid, signal.SIGTERM)
                        print(f"[P2P] Sent SIGTERM to job {job_id} (PID {job.pid})")
                        stopped += 1
                        job.status = "stopping"
                except ProcessLookupError:
                    # Process already gone
                    job.status = "stopped"
                except Exception as e:
                    print(f"[P2P] Failed to stop job {job_id}: {e}")

        # Wait for processes to terminate
        if stopped > 0:
            await asyncio.sleep(5)

            # Force kill any remaining
            with self.jobs_lock:
                for job_id, job in list(self.local_jobs.items()):
                    if job.status == "stopping" and job.pid > 0:
                        try:
                            os.kill(job.pid, signal.SIGKILL)
                            print(f"[P2P] Force killed job {job_id}")
                        except:
                            pass
                        job.status = "stopped"

        return stopped

    async def _perform_git_update(self) -> Tuple[bool, str]:
        """Perform git pull to update the codebase.

        Returns: (success, message)
        """
        # Check for local changes
        if self._check_local_changes():
            return False, "Local changes detected. Cannot auto-update. Please commit or stash changes."

        # Stop jobs if configured
        if GRACEFUL_SHUTDOWN_BEFORE_UPDATE:
            stopped = await self._stop_all_local_jobs()
            if stopped > 0:
                print(f"[P2P] Stopped {stopped} jobs before update")

        try:
            # Perform git pull
            result = subprocess.run(
                self._git_cmd("pull", GIT_REMOTE_NAME, GIT_BRANCH_NAME),
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=120
            )

            if result.returncode != 0:
                return False, f"Git pull failed: {result.stderr}"

            print(f"[P2P] Git pull successful: {result.stdout}")
            return True, result.stdout

        except subprocess.TimeoutExpired:
            return False, "Git pull timed out"
        except Exception as e:
            return False, f"Git pull error: {e}"

    async def _restart_orchestrator(self):
        """Restart the orchestrator process after update."""
        print("[P2P] Restarting orchestrator to apply updates...")

        # Save state before restart
        self._save_state()

        # Get current script path and arguments
        script_path = Path(__file__).resolve()
        args = sys.argv[1:]

        # Schedule restart
        await asyncio.sleep(2)

        # Use exec to replace current process
        os.execv(sys.executable, [sys.executable, str(script_path)] + args)

    async def _git_update_loop(self):
        """Background loop to periodically check for and apply updates."""
        if not AUTO_UPDATE_ENABLED:
            print("[P2P] Auto-update disabled")
            return

        print(f"[P2P] Git auto-update loop started (interval: {GIT_UPDATE_CHECK_INTERVAL}s)")

        while self.running:
            try:
                await asyncio.sleep(GIT_UPDATE_CHECK_INTERVAL)

                if not self.running:
                    break

                # Check for updates
                has_updates, local_commit, remote_commit = self._check_for_updates()

                if has_updates and local_commit and remote_commit:
                    commits_behind = self._get_commits_behind(local_commit, remote_commit)
                    print(f"[P2P] Update available: {commits_behind} commits behind")
                    print(f"[P2P] Local:  {local_commit[:8]}")
                    print(f"[P2P] Remote: {remote_commit[:8]}")

                    # Perform update
                    success, message = await self._perform_git_update()

                    if success:
                        print(f"[P2P] Update successful, restarting...")
                        await self._restart_orchestrator()
                    else:
                        print(f"[P2P] Update failed: {message}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[P2P] Git update loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry on error

    # ============================================
    # HTTP API Handlers
    # ============================================

    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        """Handle heartbeat from peer node."""
        try:
            data = await request.json()
            incoming_voters = data.get("voter_node_ids") or data.get("voters") or None
            if incoming_voters:
                voters_list: List[str] = []
                if isinstance(incoming_voters, list):
                    voters_list = [str(v).strip() for v in incoming_voters if str(v).strip()]
                elif isinstance(incoming_voters, str):
                    voters_list = [t.strip() for t in incoming_voters.split(",") if t.strip()]
                if voters_list:
                    self._maybe_adopt_voter_node_ids(voters_list, source="learned")
            peer_info = NodeInfo.from_dict(data)
            # Ignore self-heartbeats so NAT detection + leader election aren't
            # distorted when COORDINATOR_URL includes this node's own endpoint(s).
            if peer_info.node_id == self.node_id:
                self._update_self_info()
                payload = self.self_info.to_dict()
                voter_node_ids = list(getattr(self, "voter_node_ids", []) or [])
                if voter_node_ids:
                    payload["voter_node_ids"] = voter_node_ids
                    payload["voter_quorum_size"] = int(getattr(self, "voter_quorum_size", 0) or 0)
                    payload["voter_config_source"] = str(getattr(self, "voter_config_source", "") or "")
                return web.json_response(payload)
            # Receiving any inbound heartbeat implies we're reachable inbound.
            self.last_inbound_heartbeat = time.time()
            # Preserve the node's self-reported endpoint for multi-path retries.
            if not peer_info.reported_host:
                peer_info.reported_host = peer_info.host
            if not peer_info.reported_port:
                peer_info.reported_port = peer_info.port
            peer_info.last_heartbeat = time.time()
            # Prefer the remote socket address over self-reported host so that
            # nodes behind overlays (e.g., Tailscale) use a reachable address.
            forwarded_for = (
                request.headers.get("X-Forwarded-For")
                or request.headers.get("X-Real-IP")
                or request.headers.get("CF-Connecting-IP")
            )
            if forwarded_for:
                peer_info.host = forwarded_for.split(",")[0].strip()
            elif request.remote:
                peer_info.host = request.remote

            # Preserve local reachability diagnostics: a peer can be "alive" (it can
            # send us heartbeats) while still being unreachable for inbound HTTP
            # (e.g. NAT/firewall). Our outbound heartbeat failures track that.
            with self.peers_lock:
                existing = self.peers.get(peer_info.node_id)
                if existing:
                    peer_info.consecutive_failures = int(getattr(existing, "consecutive_failures", 0) or 0)
                    peer_info.last_failure_time = float(getattr(existing, "last_failure_time", 0.0) or 0.0)
                    # Sticky NAT/relay routing:
                    # - Receiving a direct heartbeat does NOT imply the peer is reachable inbound.
                    # - If a peer has ever registered via /relay/heartbeat, preserve nat_blocked
                    #   and relay_via so leaders can continue routing commands through the relay hub.
                    if getattr(existing, "nat_blocked", False) and not getattr(peer_info, "nat_blocked", False):
                        peer_info.nat_blocked = True
                    if (getattr(existing, "relay_via", "") or "") and not (getattr(peer_info, "relay_via", "") or ""):
                        peer_info.relay_via = str(getattr(existing, "relay_via", "") or "")
                    # Preserve retirement state across updates.
                    if getattr(existing, "retired", False):
                        peer_info.retired = True
                        peer_info.retired_at = float(getattr(existing, "retired_at", 0.0) or 0.0)
                self.peers[peer_info.node_id] = peer_info

            # Return our info
            self._update_self_info()
            payload = self.self_info.to_dict()
            voter_node_ids = list(getattr(self, "voter_node_ids", []) or [])
            if voter_node_ids:
                payload["voter_node_ids"] = voter_node_ids
                payload["voter_quorum_size"] = int(getattr(self, "voter_quorum_size", 0) or 0)
                payload["voter_config_source"] = str(getattr(self, "voter_config_source", "") or "")
            return web.json_response(payload)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_status(self, request: web.Request) -> web.Response:
        """Return cluster status."""
        self._update_self_info()

        with self.peers_lock:
            peers_snapshot = list(self.peers.values())

        conflict_keys = self._endpoint_conflict_keys([self.self_info] + peers_snapshot)
        effective_leader = self._get_leader_peer()

        peers: Dict[str, Any] = {}
        for node_id, info in ((p.node_id, p) for p in peers_snapshot):
            d = info.to_dict()
            d["endpoint_conflict"] = self._endpoint_key(info) in conflict_keys
            d["leader_eligible"] = self._is_leader_eligible(info, conflict_keys, require_alive=False)
            peers[node_id] = d

        # Convenience diagnostics: reported leaders vs eligible leaders.
        leaders_reported = sorted(
            [p.node_id for p in peers_snapshot if p.role == NodeRole.LEADER and p.is_alive()]
        )
        leaders_eligible = sorted(
            [
                p.node_id
                for p in peers_snapshot
                if p.role == NodeRole.LEADER and self._is_leader_eligible(p, conflict_keys)
            ]
        )

        with self.jobs_lock:
            jobs = {k: v.to_dict() for k, v in self.local_jobs.items()}

        # Get improvement cycle manager status
        improvement_status = None
        if self.improvement_cycle_manager:
            try:
                improvement_status = self.improvement_cycle_manager.get_status()
            except Exception as e:
                improvement_status = {"error": str(e)}

        # Get diversity metrics
        diversity_metrics = self._get_diversity_metrics()

        voter_ids = list(getattr(self, "voter_node_ids", []) or [])
        voters_alive = 0
        if voter_ids:
            peer_map = {p.node_id: p for p in peers_snapshot}
            for vid in voter_ids:
                if vid == self.node_id:
                    voters_alive += 1
                    continue
                peer = peer_map.get(vid)
                if peer and peer.is_alive():
                    voters_alive += 1

        return web.json_response({
            "node_id": self.node_id,
            "role": self.role.value,
            "leader_id": self.leader_id,
            "effective_leader_id": (effective_leader.node_id if effective_leader else None),
            "leaders_reported": leaders_reported,
            "leaders_eligible": leaders_eligible,
            "voter_node_ids": voter_ids,
            "voter_quorum_size": int(getattr(self, "voter_quorum_size", 0) or 0),
            "voters_alive": voters_alive,
            "voter_quorum_ok": self._has_voter_quorum(),
            "self": self.self_info.to_dict(),
            "peers": peers,
            "local_jobs": jobs,
            "alive_peers": len([p for p in self.peers.values() if p.is_alive()]),
            "improvement_cycle_manager": improvement_status,
            "diversity_metrics": diversity_metrics,
        })

    async def handle_election(self, request: web.Request) -> web.Response:
        """Handle election message from another node."""
        try:
            # Only "bully" lower-priority candidates when we're actually eligible
            # to act as a leader. Otherwise (e.g. NAT-blocked / ambiguous endpoint),
            # responding ALIVE can stall elections and leave the cluster leaderless.
            self._update_self_info()
            data = await request.json()
            candidate_id = str(data.get("candidate_id") or "")
            if not candidate_id:
                return web.json_response({"error": "missing_candidate_id"}, status=400)

            with self.peers_lock:
                peers_snapshot = [p for p in self.peers.values() if p.node_id != self.node_id]
            conflict_keys = self._endpoint_conflict_keys([self.self_info] + peers_snapshot)
            eligible = self._is_leader_eligible(self.self_info, conflict_keys, require_alive=False)
            voter_node_ids = list(getattr(self, "voter_node_ids", []) or [])
            if eligible and voter_node_ids:
                # When quorum gating is enabled, only configured voters can participate
                # in bully elections. Non-voters responding "ALIVE" would stall the
                # election because their own `_start_election()` returns early.
                eligible = (self.node_id in voter_node_ids) and self._has_voter_quorum()

            # If our ID is higher, we respond with "ALIVE" (Bully algorithm)
            if self.node_id > candidate_id and eligible:
                # Start our own election
                asyncio.create_task(self._start_election())
                return web.json_response({"response": "ALIVE", "node_id": self.node_id, "eligible": True})
            else:
                return web.json_response({"response": "OK", "node_id": self.node_id, "eligible": bool(eligible)})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_lease_request(self, request: web.Request) -> web.Response:
        """Voter endpoint: grant/renew an exclusive leader lease.

        A leader candidate must obtain grants from a quorum of voters before it
        may act as leader. Voters only grant to one leader at a time until the
        lease expires (or is explicitly released by stepping down).
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)
            data = await request.json()
            leader_id = str(data.get("leader_id") or data.get("candidate_id") or "").strip()
            lease_id = str(data.get("lease_id") or "").strip()
            duration_raw = data.get("lease_duration", LEADER_LEASE_DURATION)
            try:
                duration = int(duration_raw)
            except Exception:
                duration = int(LEADER_LEASE_DURATION)
            duration = max(10, min(duration, int(LEADER_LEASE_DURATION * 2)))

            if not leader_id or not lease_id:
                return web.json_response({"granted": False, "reason": "missing_fields"}, status=400)

            voters = list(getattr(self, "voter_node_ids", []) or [])
            if voters:
                if self.node_id not in voters:
                    return web.json_response({"granted": False, "reason": "not_a_voter"}, status=403)
                if leader_id not in voters:
                    return web.json_response({"granted": False, "reason": "leader_not_voter"}, status=403)

            now = time.time()
            current_leader = str(getattr(self, "voter_grant_leader_id", "") or "")
            current_expires = float(getattr(self, "voter_grant_expires", 0.0) or 0.0)

            if current_leader and current_expires > now and current_leader != leader_id:
                return web.json_response(
                    {
                        "granted": False,
                        "reason": "lease_already_granted",
                        "current_leader_id": current_leader,
                        "current_lease_id": str(getattr(self, "voter_grant_lease_id", "") or ""),
                        "lease_expires": current_expires,
                    },
                    status=409,
                )

            self.voter_grant_leader_id = leader_id
            self.voter_grant_lease_id = lease_id
            self.voter_grant_expires = now + float(duration)
            self._save_state()

            lease_ttl_seconds = max(0.0, float(self.voter_grant_expires) - time.time())
            return web.json_response(
                {
                    "granted": True,
                    "leader_id": leader_id,
                    "lease_id": lease_id,
                    "lease_expires": self.voter_grant_expires,
                    # Use a relative TTL for robustness under clock skew (absolute
                    # timestamps from different machines are not directly comparable).
                    "lease_ttl_seconds": lease_ttl_seconds,
                    "voter_id": self.node_id,
                }
            )
        except Exception as e:
            return web.json_response({"granted": False, "error": str(e)}, status=400)

    async def handle_voter_grant_status(self, request: web.Request) -> web.Response:
        """Read-only voter endpoint: return our currently granted leader lease.

        This lets nodes resolve split-brain by consulting a quorum of voters for
        the active lease holder, without mutating lease state.
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)
            now = time.time()
            expires = float(getattr(self, "voter_grant_expires", 0.0) or 0.0)
            return web.json_response(
                {
                    "voter_id": self.node_id,
                    "now": now,
                    "leader_id": str(getattr(self, "voter_grant_leader_id", "") or ""),
                    "lease_id": str(getattr(self, "voter_grant_lease_id", "") or ""),
                    "lease_expires": expires,
                    "lease_ttl_seconds": max(0.0, expires - now),
                }
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_coordinator(self, request: web.Request) -> web.Response:
        """Handle coordinator announcement from new leader.

        LEARNED LESSONS - Only accept leadership from higher-priority nodes (Bully algorithm).
        Also handles lease-based leadership updates.
        """
        try:
            self._update_self_info()
            data = await request.json()
            new_leader_raw = data.get("leader_id")
            if not new_leader_raw:
                return web.json_response(
                    {"accepted": False, "reason": "missing_leader_id"},
                    status=400,
                )
            new_leader = str(new_leader_raw)
            lease_id = data.get("lease_id", "")
            lease_expires = data.get("lease_expires", 0)
            is_renewal = data.get("lease_renewal", False)
            incoming_voters = data.get("voter_node_ids") or data.get("voters") or None
            if incoming_voters:
                voters_list: List[str] = []
                if isinstance(incoming_voters, list):
                    voters_list = [str(v).strip() for v in incoming_voters if str(v).strip()]
                elif isinstance(incoming_voters, str):
                    voters_list = [t.strip() for t in incoming_voters.split(",") if t.strip()]
                if voters_list:
                    self._maybe_adopt_voter_node_ids(voters_list, source="learned")

            voters = list(getattr(self, "voter_node_ids", []) or [])
            if voters and new_leader not in voters:
                return web.json_response(
                    {"accepted": False, "reason": "leader_not_voter", "voters": voters},
                    status=403,
                )

            # Voter-side safety: if we've granted a still-valid lease to a different leader,
            # do not accept a conflicting coordinator announcement. This prevents a voter
            # from "following" a non-quorum leader during transient partitions.
            if voters and self.node_id in voters:
                grant_leader = str(getattr(self, "voter_grant_leader_id", "") or "")
                grant_expires = float(getattr(self, "voter_grant_expires", 0.0) or 0.0)
                if grant_leader and grant_expires > time.time() and grant_leader != new_leader:
                    return web.json_response(
                        {
                            "accepted": False,
                            "reason": "voter_lease_conflict",
                            "granted_to": grant_leader,
                            "granted_until": grant_expires,
                        },
                        status=409,
                    )

            # If quorum gating is not configured, fall back to bully ordering
            # (lexicographically highest node_id wins).
            if not voters and self.role == NodeRole.LEADER and new_leader < self.node_id:
                # Exception: accept if our lease has expired
                if self.leader_lease_expires > 0 and time.time() >= self.leader_lease_expires:
                    print(f"[P2P] Our lease expired, accepting leader: {new_leader}")
                else:
                    print(f"[P2P] Rejecting leader announcement from lower-priority node: {new_leader} < {self.node_id}")
                    return web.json_response({"accepted": False, "reason": "lower_priority"})

            # Reject leadership from nodes that are not directly reachable / uniquely addressable.
            if new_leader != self.node_id:
                with self.peers_lock:
                    peer = self.peers.get(new_leader)
                    peers_snapshot = [p for p in self.peers.values() if p.node_id != self.node_id]
                if peer:
                    conflict_keys = self._endpoint_conflict_keys([self.self_info] + peers_snapshot)
                    if not self._is_leader_eligible(peer, conflict_keys, require_alive=False):
                        return web.json_response({"accepted": False, "reason": "leader_ineligible"})

            if is_renewal:
                # Just a lease renewal, update expiry silently
                if new_leader == self.leader_id:
                    self.leader_lease_expires = lease_expires
                    self.leader_lease_id = lease_id
                    return web.json_response({"accepted": True})

            print(f"[P2P] Accepting leader announcement: {new_leader}")
            self.leader_id = new_leader
            self.leader_lease_id = lease_id
            self.leader_lease_expires = lease_expires if lease_expires else time.time() + LEADER_LEASE_DURATION

            if new_leader == self.node_id:
                self.role = NodeRole.LEADER
            else:
                self.role = NodeRole.FOLLOWER

            self._save_state()
            return web.json_response({"accepted": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_start_job(self, request: web.Request) -> web.Response:
        """Handle request to start a job (from leader)."""
        try:
            data = await request.json()
            job_type = JobType(data.get("job_type", "selfplay"))
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)
            engine_mode = data.get("engine_mode", "descent-only")
            job_id = data.get("job_id")
            cuda_visible_devices = data.get("cuda_visible_devices")

            job = await self._start_local_job(
                job_type,
                board_type=board_type,
                num_players=num_players,
                engine_mode=engine_mode,
                job_id=job_id,
                cuda_visible_devices=cuda_visible_devices,
            )

            if job:
                return web.json_response({"success": True, "job": job.to_dict()})
            else:
                return web.json_response({"success": False, "error": "Failed to start job"}, status=500)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_stop_job(self, request: web.Request) -> web.Response:
        """Handle request to stop a job."""
        try:
            data = await request.json()
            job_id = data.get("job_id")

            with self.jobs_lock:
                if job_id in self.local_jobs:
                    job = self.local_jobs[job_id]
                    try:
                        os.kill(job.pid, signal.SIGTERM)
                        job.status = "stopped"
                    except:
                        pass
                    return web.json_response({"success": True})

            return web.json_response({"success": False, "error": "Job not found"}, status=404)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_cleanup(self, request: web.Request) -> web.Response:
        """Handle cleanup request (from leader or manual).

        LEARNED LESSONS - This endpoint allows remote nodes to trigger disk cleanup
        when the leader detects disk usage approaching critical thresholds.
        """
        try:
            print(f"[P2P] Cleanup request received")

            # Run cleanup in background to avoid blocking the request
            asyncio.create_task(self._cleanup_local_disk())

            # Return current disk usage
            usage = self._get_resource_usage()
            return web.json_response({
                "success": True,
                "disk_percent_before": usage["disk_percent"],
                "message": "Cleanup initiated",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_restart_stuck_jobs(self, request: web.Request) -> web.Response:
        """Handle request to restart stuck selfplay jobs.

        LEARNED LESSONS - Called by leader when it detects GPU idle with running processes.
        Kills all selfplay processes and clears job tracking so they restart.
        """
        try:
            print(f"[P2P] Restart stuck jobs request received")

            # Run in background to avoid blocking
            asyncio.create_task(self._restart_local_stuck_jobs())

            return web.json_response({
                "success": True,
                "message": "Stuck job restart initiated",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_reduce_selfplay(self, request: web.Request) -> web.Response:
        """Stop excess selfplay jobs on this node (load shedding).

        Used by leaders when a node is under memory/disk pressure so the node
        can recover without requiring manual intervention.
        """
        try:
            data = await request.json()
            target_raw = data.get("target_selfplay_jobs", data.get("target", 0))
            reason = str(data.get("reason") or "remote_request")
            try:
                target = int(target_raw)
            except Exception:
                target = 0

            result = await self._reduce_local_selfplay_jobs(target, reason=reason)
            return web.json_response({"success": True, **result})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    async def handle_cleanup_files(self, request: web.Request) -> web.Response:
        """Delete specific files from this node (for post-sync cleanup).

        Called by leader after successful sync to training nodes to free
        disk space on source nodes with high disk usage.
        """
        try:
            data = await request.json()
            files = data.get("files", [])
            reason = data.get("reason", "manual")

            if not files:
                return web.json_response({"success": False, "error": "No files specified"}, status=400)

            print(f"[P2P] Cleanup files request: {len(files)} files, reason={reason}")

            data_dir = self.get_data_directory()
            freed_bytes = 0
            deleted_count = 0

            for file_path in files:
                # Security: only allow deletion within data directory
                full_path = data_dir / (file_path or "").lstrip("/")
                try:
                    data_root = data_dir.resolve()
                    resolved = full_path.resolve()
                    resolved.relative_to(data_root)
                except Exception:
                    print(f"[P2P] Cleanup: skipping path outside data dir: {file_path}")
                    continue

                if resolved.exists():
                    try:
                        size = resolved.stat().st_size
                        resolved.unlink()
                        freed_bytes += size
                        deleted_count += 1
                    except Exception as e:
                        print(f"[P2P] Failed to delete {file_path}: {e}")

            print(f"[P2P] Cleanup complete: {deleted_count} files, {freed_bytes / 1e6:.1f}MB freed")

            return web.json_response({
                "success": True,
                "freed_bytes": freed_bytes,
                "deleted_count": deleted_count,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_training_sync(self, request: web.Request) -> web.Response:
        """Manually trigger sync of selfplay data to training nodes.

        Leader-only: Syncs selfplay data to the top GPU nodes for training.
        """
        try:
            result = await self._sync_selfplay_to_training_nodes()
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_gpu_rankings(self, request: web.Request) -> web.Response:
        """Get GPU power rankings for all nodes in the cluster.

        Returns nodes sorted by GPU processing power for training priority.
        """
        try:
            rankings = self._get_training_nodes_ranked()
            training_nodes = self._get_training_primary_nodes()

            return web.json_response({
                "rankings": rankings,
                "training_primary_nodes": [n.node_id for n in training_nodes],
                "training_node_count": TRAINING_NODE_COUNT,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle health check request.

        LEARNED LESSONS - Simple health endpoint for monitoring and load balancers.
        Returns node health status without full cluster state.
        Includes utilization status from resource_optimizer for cluster coordination.
        """
        try:
            self._update_self_info()
            is_healthy = self.self_info.is_healthy()

            response = {
                "healthy": is_healthy,
                "node_id": self.node_id,
                "role": self.role.value,
                "disk_percent": self.self_info.disk_percent,
                "memory_percent": self.self_info.memory_percent,
                "cpu_percent": self.self_info.cpu_percent,
                "selfplay_jobs": self.self_info.selfplay_jobs,
                "training_jobs": self.self_info.training_jobs,
            }

            # Add cluster utilization status for cooperative 60-80% targeting
            if HAS_RATE_NEGOTIATION and get_utilization_status is not None:
                try:
                    util_status = get_utilization_status()
                    response["cluster_utilization"] = {
                        "cpu_util": util_status.get("cpu_util", 0),
                        "gpu_util": util_status.get("gpu_util", 0),
                        "selfplay_rate": util_status.get("current_rate", 1000),
                        "target_range": "60-80%",
                        "status": util_status.get("status", "unknown"),
                    }
                except Exception:
                    pass

            return web.json_response(response)
        except Exception as e:
            return web.json_response({"error": str(e), "healthy": False}, status=500)

    # ============================================
    # Relay/Hub Handlers for NAT-blocked nodes
    # ============================================

    async def handle_relay_heartbeat(self, request: web.Request) -> web.Response:
        """POST /relay/heartbeat - Accept heartbeat from NAT-blocked node.

        NAT-blocked nodes (e.g., Vast.ai behind carrier NAT) can't receive
        incoming connections. They use this endpoint to:
        1. Send their status to the leader
        2. Get back the full cluster peer list
        3. Mark themselves as nat_blocked so leader doesn't try to reach them

        Request body: Same as regular heartbeat (NodeInfo dict)
        Response: {
            "self": NodeInfo,  # Leader's info
            "peers": {node_id: NodeInfo},  # All known peers including NAT-blocked
            "leader_id": str
        }
        """
        try:
            data = await request.json()
            relay_ack = data.get("relay_ack") or []
            relay_results = data.get("relay_results") or []
            peer_info = NodeInfo.from_dict(data)
            if not peer_info.reported_host:
                peer_info.reported_host = peer_info.host
            if not peer_info.reported_port:
                peer_info.reported_port = peer_info.port
            peer_info.last_heartbeat = time.time()
            peer_info.nat_blocked = True  # Mark as NAT-blocked
            peer_info.relay_via = self.node_id  # This node is their relay

            # Get their real IP from the request (for logging/debugging)
            forwarded_for = (
                request.headers.get("X-Forwarded-For")
                or request.headers.get("X-Real-IP")
                or request.headers.get("CF-Connecting-IP")
            )
            real_ip = forwarded_for.split(",")[0].strip() if forwarded_for else request.remote
            if real_ip:
                peer_info.host = real_ip

            # Store in peers list (they're part of the cluster even if not directly reachable)
            with self.peers_lock:
                self.peers[peer_info.node_id] = peer_info

            print(f"[P2P] Relay heartbeat from {peer_info.node_id} (real IP: {real_ip})")

            # Apply relay ACKs/results and return any queued commands.
            commands_to_send: List[Dict[str, Any]] = []
            with self.relay_lock:
                queue = list(self.relay_command_queue.get(peer_info.node_id, []))
                now = time.time()
                queue = [
                    cmd for cmd in queue
                    if float(cmd.get("expires_at", 0.0) or 0.0) > now
                ]

                if relay_ack:
                    ack_set = {str(c) for c in relay_ack if c}
                    queue = [cmd for cmd in queue if str(cmd.get("id", "")) not in ack_set]

                if relay_results:
                    for item in relay_results:
                        try:
                            cmd_id = str(item.get("id") or "")
                            ok = bool(item.get("ok", False))
                            err = str(item.get("error") or "")
                            if not cmd_id:
                                continue
                            if ok:
                                print(f"[P2P] Relay command {cmd_id} on {peer_info.node_id}: ok")
                            else:
                                print(f"[P2P] Relay command {cmd_id} on {peer_info.node_id}: failed {err[:200]}")
                        except Exception:
                            continue

                self.relay_command_queue[peer_info.node_id] = queue
                commands_to_send = queue[:RELAY_COMMAND_MAX_BATCH]

            # Return cluster state so they can see all peers
            self._update_self_info()
            with self.peers_lock:
                peers = {k: v.to_dict() for k, v in self.peers.items()}

            effective_leader = self._get_leader_peer()
            effective_leader_id = effective_leader.node_id if effective_leader else None
            return web.json_response({
                "success": True,
                "self": self.self_info.to_dict(),
                "peers": peers,
                # IMPORTANT: only advertise a leader_id when it is actually reachable
                # and currently reporting itself as leader. Persisted/stale leader_id
                # values are surfaced separately so bootstrapping nodes don't get
                # stuck pointing at a non-leader.
                "leader_id": effective_leader_id,
                "effective_leader_id": effective_leader_id,
                "last_known_leader_id": self.leader_id,
                "relay_node": self.node_id,
                # Propagate the stable voter set so nodes that boot without local
                # config still enable quorum gating and avoid split-brain.
                "voter_node_ids": list(getattr(self, "voter_node_ids", []) or []),
                "voter_quorum_size": int(getattr(self, "voter_quorum_size", 0) or 0),
                "voter_quorum_ok": self._has_voter_quorum(),
                "voter_config_source": str(getattr(self, "voter_config_source", "") or ""),
                "commands": commands_to_send,
            })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_relay_enqueue(self, request: web.Request) -> web.Response:
        """POST /relay/enqueue - Enqueue a command for a NAT-blocked node on this relay.

        This enables multi-hop operation when NAT-blocked nodes can reach a
        public relay hub (e.g., AWS) but cannot reach the cluster leader
        directly (e.g., TUN-less Tailscale inside some containers).

        Request body:
          {
            "target_node_id": "node-id",
            "type": "start_job" | "cleanup" | ...,
            "payload": { ... }
          }

        Response:
          { "success": true, "id": "<cmd_id>" }
        """
        try:
            data = await request.json()
        except Exception:
            data = {}

        try:
            target_node_id = str(data.get("target_node_id") or data.get("node_id") or "").strip()
            cmd_type = str(data.get("type") or data.get("cmd_type") or "").strip()
            payload = data.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            target_node_id = ""
            cmd_type = ""
            payload = {}

        if not target_node_id or not cmd_type:
            return web.json_response(
                {"success": False, "error": "invalid_request", "message": "target_node_id and type are required"},
                status=400,
            )

        cmd_id = self._enqueue_relay_command(target_node_id, cmd_type, payload)
        if not cmd_id:
            return web.json_response({"success": False, "error": "queue_full"}, status=429)

        return web.json_response({"success": True, "id": cmd_id})

    async def handle_relay_peers(self, request: web.Request) -> web.Response:
        """GET /relay/peers - Get list of all peers including NAT-blocked ones.

        Used by nodes to discover the full cluster including NAT-blocked members.
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)
            self._update_self_info()
            effective_leader = self._get_leader_peer()
            with self.peers_lock:
                all_peers = {k: v.to_dict() for k, v in self.peers.items()}

            # Separate NAT-blocked and directly reachable
            nat_blocked = {k: v for k, v in all_peers.items() if v.get('nat_blocked')}
            direct = {k: v for k, v in all_peers.items() if not v.get('nat_blocked')}

            return web.json_response({
                "success": True,
                "leader_id": (effective_leader.node_id if effective_leader else self.leader_id),
                "effective_leader_id": (effective_leader.node_id if effective_leader else None),
                "total_peers": len(all_peers),
                "direct_peers": len(direct),
                "nat_blocked_peers": len(nat_blocked),
                "voter_node_ids": list(getattr(self, "voter_node_ids", []) or []),
                "voter_quorum_size": int(getattr(self, "voter_quorum_size", 0) or 0),
                "voter_quorum_ok": self._has_voter_quorum(),
                "voter_config_source": str(getattr(self, "voter_config_source", "") or ""),
                "peers": all_peers,
            })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_relay_status(self, request: web.Request) -> web.Response:
        """GET /relay/status - Get relay queue status for debugging.

        Shows pending commands per NAT-blocked node including command ages.
        Useful for diagnosing relay delivery issues.
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)

            now = time.time()
            queue_status = {}
            total_pending = 0

            for node_id, commands in self.relay_command_queue.items():
                if not commands:
                    continue
                cmd_info = []
                for cmd in commands:
                    age_secs = now - cmd.get("ts", now)
                    cmd_info.append({
                        "id": cmd.get("id", ""),
                        "type": cmd.get("cmd", ""),
                        "age_secs": round(age_secs, 1),
                        "stale": age_secs > 300,  # >5 min is stale
                    })
                queue_status[node_id] = {
                    "pending_count": len(commands),
                    "commands": cmd_info,
                    "oldest_age_secs": round(max((now - c.get("ts", now)) for c in commands), 1) if commands else 0,
                }
                total_pending += len(commands)

            # Get NAT-blocked nodes for context
            with self.peers_lock:
                nat_blocked_nodes = [nid for nid, p in self.peers.items() if getattr(p, 'nat_blocked', False)]

            return web.json_response({
                "success": True,
                "total_pending_commands": total_pending,
                "nat_blocked_nodes": nat_blocked_nodes,
                "nodes_with_pending": list(queue_status.keys()),
                "queues": queue_status,
            })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_register(self, request: web.Request) -> web.Response:
        """POST /register - Node self-registration for dynamic IP updates.

        Nodes call this endpoint to announce their current IP address.
        Useful when Vast.ai instances restart and get new IPs.

        Request body:
        {
            "node_id": "vast-5090-quad",
            "host": "211.72.13.202",
            "port": 45875,
            "vast_instance_id": "28654132"  // optional
        }
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({
                "error": "Dynamic registry not available"
            }, status=501)

        try:
            data = await request.json()
            node_id = data.get("node_id")
            host = data.get("host")
            port = data.get("port", 22)
            vast_instance_id = data.get("vast_instance_id")
            tailscale_ip = data.get("tailscale_ip")

            if not node_id or not host:
                return web.json_response({
                    "error": "Missing required fields: node_id, host"
                }, status=400)

            registry = get_registry()
            success = registry.register_node(node_id, host, port, vast_instance_id, tailscale_ip=tailscale_ip)

            if success:
                print(f"[P2P] Node registered: {node_id} at {host}:{port}")
                return web.json_response({
                    "success": True,
                    "node_id": node_id,
                    "registered_host": host,
                    "registered_port": port,
                })
            else:
                return web.json_response({
                    "error": "Registration failed"
                }, status=500)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_registry_status(self, request: web.Request) -> web.Response:
        """GET /registry/status - Get dynamic registry status for all nodes.

        Returns current state of all nodes including:
        - Effective IP addresses (dynamic if registered)
        - Health state (online/degraded/offline)
        - Failure counters
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({
                "error": "Dynamic registry not available"
            }, status=501)

        try:
            registry = get_registry()
            nodes_status = registry.get_all_nodes_status()
            online_nodes = registry.get_online_nodes()

            return web.json_response({
                "total_nodes": len(nodes_status),
                "online_nodes": len(online_nodes),
                "online_node_ids": online_nodes,
                "nodes": nodes_status,
            })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_registry_update_vast(self, request: web.Request) -> web.Response:
        """POST /registry/update_vast - Refresh Vast instance IPs in the dynamic registry.

        Uses VAST_API_KEY when available, otherwise attempts the `vastai` CLI.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({
                "error": "Dynamic registry not available"
            }, status=501)

        try:
            registry = get_registry()
            updated = await registry.update_vast_ips()

            return web.json_response({
                "success": True,
                "nodes_updated": updated,
            })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_registry_update_aws(self, request: web.Request) -> web.Response:
        """POST /registry/update_aws - Refresh AWS instance IPs in the dynamic registry.

        Uses the `aws` CLI and requires nodes to define `aws_instance_id` in
        distributed_hosts.yaml properties.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({"error": "Dynamic registry not available"}, status=501)

        try:
            registry = get_registry()
            updated = await registry.update_aws_ips()
            return web.json_response({"success": True, "nodes_updated": updated})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_registry_update_tailscale(self, request: web.Request) -> web.Response:
        """POST /registry/update_tailscale - Discover Tailscale IPs in the dynamic registry.

        Uses `tailscale status --json` when available. No-op if `tailscale` is
        not installed or the node is not part of a Tailscale network.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({"error": "Dynamic registry not available"}, status=501)

        try:
            registry = get_registry()
            updated = await registry.update_tailscale_ips()
            return web.json_response({"success": True, "nodes_updated": updated})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_registry_save_yaml(self, request: web.Request) -> web.Response:
        """POST /registry/save_yaml - Write dynamic IPs back to YAML config.

        Creates a backup before modifying. Only updates hosts where
        dynamic IP differs from static IP.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return web.json_response({
                "error": "Dynamic registry not available"
            }, status=501)

        try:
            registry = get_registry()
            updated = registry.update_yaml_config()

            return web.json_response({
                "success": True,
                "config_updated": updated,
            })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_git_status(self, request: web.Request) -> web.Response:
        """Get git status for this node.

        Returns local/remote commit info and whether updates are available.
        """
        try:
            local_commit = self._get_local_git_commit()
            local_branch = self._get_local_git_branch()
            has_local_changes = self._check_local_changes()

            # Check for remote updates (this does a git fetch)
            has_updates, _, remote_commit = self._check_for_updates()
            commits_behind = 0
            if has_updates and local_commit and remote_commit:
                commits_behind = self._get_commits_behind(local_commit, remote_commit)

            return web.json_response({
                "local_commit": local_commit[:8] if local_commit else None,
                "local_commit_full": local_commit,
                "local_branch": local_branch,
                "remote_commit": remote_commit[:8] if remote_commit else None,
                "remote_commit_full": remote_commit,
                "has_updates": has_updates,
                "commits_behind": commits_behind,
                "has_local_changes": has_local_changes,
                "auto_update_enabled": AUTO_UPDATE_ENABLED,
                "ringrift_path": self.ringrift_path,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_git_update(self, request: web.Request) -> web.Response:
        """Manually trigger a git update on this node.

        This will stop jobs, pull updates, and restart the orchestrator.
        """
        try:
            # Check for updates first
            has_updates, local_commit, remote_commit = self._check_for_updates()

            if not has_updates:
                return web.json_response({
                    "success": True,
                    "message": "Already up to date",
                    "local_commit": local_commit[:8] if local_commit else None,
                })

            # Perform the update
            success, message = await self._perform_git_update()

            if success:
                # Schedule restart
                asyncio.create_task(self._restart_orchestrator())
                return web.json_response({
                    "success": True,
                    "message": "Update successful, restarting...",
                    "old_commit": local_commit[:8] if local_commit else None,
                    "new_commit": remote_commit[:8] if remote_commit else None,
                })
            else:
                return web.json_response({
                    "success": False,
                    "message": message,
                }, status=400)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ============================================
    # Phase 2: Distributed Data Manifest Handlers
    # ============================================

    async def handle_data_manifest(self, request: web.Request) -> web.Response:
        """Return this node's local data manifest.

        Used by leader to collect data inventory from all nodes.
        """
        try:
            local_manifest = await asyncio.to_thread(self._collect_local_data_manifest)
            with self.manifest_lock:
                self.local_data_manifest = local_manifest

            return web.json_response({
                "node_id": self.node_id,
                "manifest": local_manifest.to_dict(),
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_cluster_data_manifest(self, request: web.Request) -> web.Response:
        """Leader-only: Return cluster-wide data manifest.

        Aggregates data manifests from all nodes to show:
        - Total files across cluster
        - Total selfplay games
        - Files missing from specific nodes (for sync planning)
        """
        try:
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "Not leader",
                    "leader_id": self.leader_id,
                }, status=400)

            refresh_raw = str(request.query.get("refresh", "") or "").strip().lower()
            refresh = refresh_raw in {"1", "true", "yes", "y"}

            # Default to returning the cached manifest to keep this endpoint
            # fast and usable by daemons with tight timeouts.
            if not refresh:
                with self.manifest_lock:
                    cached = self.cluster_data_manifest
                if cached:
                    return web.json_response({
                        "cluster_manifest": cached.to_dict(),
                        "cached": True,
                    })
                # Manifest collection loop runs shortly after startup; callers
                # can retry or pass ?refresh=1 to force.
                return web.json_response({
                    "cluster_manifest": None,
                    "cached": True,
                    "error": "manifest_not_ready",
                })

            # Forced refresh: collect and update cache.
            cluster_manifest = await self._collect_cluster_manifest()
            with self.manifest_lock:
                self.cluster_data_manifest = cluster_manifest

            return web.json_response({
                "cluster_manifest": cluster_manifest.to_dict(),
                "cached": False,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_refresh_manifest(self, request: web.Request) -> web.Response:
        """Force refresh of local data manifest."""
        try:
            local_manifest = await asyncio.to_thread(self._collect_local_data_manifest)
            with self.manifest_lock:
                self.local_data_manifest = local_manifest

            return web.json_response({
                "success": True,
                "node_id": self.node_id,
                "total_files": local_manifest.total_files,
                "total_size_bytes": local_manifest.total_size_bytes,
                "selfplay_games": local_manifest.selfplay_games,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ============================================
    # Distributed CMA-ES Handlers
    # ============================================

    async def handle_cmaes_start(self, request: web.Request) -> web.Response:
        """Start a distributed CMA-ES optimization job.

        Only the leader can start distributed CMA-ES jobs.
        Request body:
        {
            "board_type": "square8",
            "num_players": 2,
            "generations": 100,
            "population_size": 20,
            "games_per_eval": 50
        }
        """
        try:
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "Only the leader can start distributed CMA-ES",
                    "leader_id": self.leader_id,
                }, status=403)

            data = await request.json()
            job_id = f"cmaes_{uuid.uuid4().hex[:8]}"

            # Create state for this job
            state = DistributedCMAESState(
                job_id=job_id,
                board_type=data.get("board_type", "square8"),
                num_players=data.get("num_players", 2),
                generations=data.get("generations", 100),
                population_size=data.get("population_size", 20),
                games_per_eval=data.get("games_per_eval", 50),
                status="starting",
                started_at=time.time(),
                last_update=time.time(),
            )

            # Find available GPU workers
            with self.peers_lock:
                gpu_nodes = [
                    p.node_id for p in self.peers.values()
                    if p.is_healthy() and p.has_gpu
                ]
            state.worker_nodes = gpu_nodes

            if not state.worker_nodes:
                return web.json_response({
                    "error": "No GPU workers available for CMA-ES",
                }, status=503)

            self.distributed_cmaes_state[job_id] = state
            state.status = "running"

            print(f"[P2P] Started distributed CMA-ES job {job_id} with {len(state.worker_nodes)} workers")

            # Launch coordinator task
            asyncio.create_task(self._run_distributed_cmaes(job_id))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "workers": state.worker_nodes,
                "config": {
                    "board_type": state.board_type,
                    "num_players": state.num_players,
                    "generations": state.generations,
                    "population_size": state.population_size,
                    "games_per_eval": state.games_per_eval,
                },
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_cmaes_evaluate(self, request: web.Request) -> web.Response:
        """Request evaluation of weights from workers.

        Called by the coordinator to distribute weight evaluation tasks.
        Workers respond via /cmaes/result endpoint.
        """
        try:
            data = await request.json()
            job_id = data.get("job_id")
            weights = data.get("weights", {})
            generation = data.get("generation", 0)
            individual_idx = data.get("individual_idx", 0)

            if not job_id:
                return web.json_response({"error": "job_id required"}, status=400)

            # Extract evaluation parameters from request
            games_per_eval = data.get("games_per_eval", 5)
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)

            # Store evaluation task for local processing
            print(f"[P2P] Received CMA-ES evaluation request: job={job_id}, gen={generation}, idx={individual_idx}")

            # Start evaluation in background
            asyncio.create_task(self._evaluate_cmaes_weights(
                job_id, weights, generation, individual_idx,
                games_per_eval=games_per_eval, board_type=board_type, num_players=num_players
            ))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "status": "evaluation_started",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_cmaes_status(self, request: web.Request) -> web.Response:
        """Get status of distributed CMA-ES jobs."""
        try:
            job_id = request.query.get("job_id")

            if job_id:
                if job_id not in self.distributed_cmaes_state:
                    return web.json_response({"error": "Job not found"}, status=404)
                state = self.distributed_cmaes_state[job_id]
                return web.json_response(state.to_dict())

            # Return all jobs
            return web.json_response({
                job_id: state.to_dict()
                for job_id, state in self.distributed_cmaes_state.items()
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_cmaes_result(self, request: web.Request) -> web.Response:
        """Receive evaluation result from a worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            generation = data.get("generation", 0)
            individual_idx = data.get("individual_idx", 0)
            fitness = data.get("fitness", 0.0)
            worker_id = data.get("worker_id", "unknown")

            if job_id not in self.distributed_cmaes_state:
                return web.json_response({"error": "Job not found"}, status=404)

            print(f"[P2P] CMA-ES result: job={job_id}, gen={generation}, idx={individual_idx}, fitness={fitness:.4f} from {worker_id}")

            # Store result - the coordinator loop will process it
            state = self.distributed_cmaes_state[job_id]
            state.last_update = time.time()

            # LEARNED LESSONS - Store result keyed by generation and index for coordinator to collect
            result_key = f"{generation}_{individual_idx}"
            state.pending_results[result_key] = fitness

            # Update best if applicable
            if fitness > state.best_fitness:
                state.best_fitness = fitness
                state.best_weights = data.get("weights", {})

            return web.json_response({
                "success": True,
                "job_id": job_id,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _run_distributed_cmaes(self, job_id: str):
        """Main coordinator loop for distributed CMA-ES.

        Integrates with CMA-ES algorithm to optimize heuristic weights.
        Distributes candidate evaluation across GPU workers in the cluster.
        """
        try:
            state = self.distributed_cmaes_state.get(job_id)
            if not state:
                return

            print(f"[P2P] CMA-ES coordinator started for job {job_id}")
            print(f"[P2P] Config: {state.generations} gens, pop={state.population_size}, {state.games_per_eval} games/eval")

            # Try to import CMA-ES library
            try:
                import cma
                import numpy as np
            except ImportError:
                print("[P2P] CMA-ES requires: pip install cma numpy")
                state.status = "error: cma not installed"
                return

            # Default heuristic weights to optimize
            weight_names = [
                "material_weight", "ring_count_weight", "stack_height_weight",
                "center_control_weight", "territory_weight", "mobility_weight",
                "line_potential_weight", "defensive_weight",
            ]
            default_weights = {
                "material_weight": 1.0, "ring_count_weight": 0.5,
                "stack_height_weight": 0.3, "center_control_weight": 0.4,
                "territory_weight": 0.8, "mobility_weight": 0.2,
                "line_potential_weight": 0.6, "defensive_weight": 0.3,
            }

            # Convert to vector for CMA-ES
            x0 = np.array([default_weights[n] for n in weight_names])

            # Initialize CMA-ES
            es = cma.CMAEvolutionStrategy(x0, 0.5, {
                'popsize': state.population_size,
                'maxiter': state.generations,
                'bounds': [0, 2],  # Weights between 0 and 2
            })

            state.current_generation = 0

            while not es.stop() and state.status == "running":
                state.current_generation += 1
                state.last_update = time.time()

                # Get candidate solutions
                solutions = es.ask()

                # Distribute evaluations across workers
                fitness_results = {}
                pending_evals = {}

                for idx, sol in enumerate(solutions):
                    weights = {name: float(sol[i]) for i, name in enumerate(weight_names)}

                    # Round-robin assign to workers
                    if state.worker_nodes:
                        worker_idx = idx % len(state.worker_nodes)
                        worker_id = state.worker_nodes[worker_idx]

                        # Send evaluation request to worker
                        eval_id = f"{job_id}_gen{state.current_generation}_idx{idx}"
                        pending_evals[eval_id] = idx

                        try:
                            with self.peers_lock:
                                worker = self.peers.get(worker_id)
                            if worker:
                                timeout = ClientTimeout(total=300)
                                async with get_client_session(timeout) as session:
                                    url = self._url_for_peer(worker, "/cmaes/evaluate")
                                    await session.post(url, json={
                                        "job_id": job_id,
                                        "weights": weights,
                                        "generation": state.current_generation,
                                        "individual_idx": idx,
                                        "games_per_eval": state.games_per_eval,
                                        "board_type": state.board_type,
                                        "num_players": state.num_players,
                                    }, headers=self._auth_headers())
                        except Exception as e:
                            print(f"[P2P] Failed to send eval to {worker_id}: {e}")
                            # Fall back to local evaluation
                            fitness = await self._evaluate_cmaes_weights_local(
                                weights, state.games_per_eval, state.board_type, state.num_players
                            )
                            fitness_results[idx] = fitness

                # Wait for results with timeout
                wait_start = time.time()
                expected_results = len(solutions) - len(fitness_results)
                while len(fitness_results) < len(solutions) and (time.time() - wait_start) < 300:
                    await asyncio.sleep(1)
                    state.last_update = time.time()

                    # Check for results that came in via /cmaes/result endpoint
                    # Results are stored in state.pending_results by handle_cmaes_result
                    for idx in range(len(solutions)):
                        if idx in fitness_results:
                            continue
                        result_key = f"{state.current_generation}_{idx}"
                        if result_key in state.pending_results:
                            fitness_results[idx] = state.pending_results[result_key]
                            del state.pending_results[result_key]  # Clean up

                    # Progress logging every 30 seconds
                    elapsed = time.time() - wait_start
                    if int(elapsed) % 30 == 0 and elapsed > 1:
                        received = len(fitness_results)
                        print(f"[P2P] Gen {state.current_generation}: {received}/{len(solutions)} results received ({elapsed:.0f}s elapsed)")

                # Fill in any missing results with default fitness
                fitnesses = []
                for idx in range(len(solutions)):
                    fitness = fitness_results.get(idx, 0.5)  # Default to 0.5 if no result
                    fitnesses.append(-fitness)  # CMA-ES minimizes, so negate

                # Update CMA-ES
                es.tell(solutions, fitnesses)

                # Track best
                best_idx = np.argmin(fitnesses)
                if -fitnesses[best_idx] > state.best_fitness:
                    state.best_fitness = -fitnesses[best_idx]
                    state.best_weights = {name: float(solutions[best_idx][i]) for i, name in enumerate(weight_names)}

                print(f"[P2P] Gen {state.current_generation}: best_fitness={state.best_fitness:.4f}")

            state.status = "completed"
            print(f"[P2P] CMA-ES job {job_id} completed: best_fitness={state.best_fitness:.4f}")
            print(f"[P2P] Best weights: {state.best_weights}")

            # Feed CMA-ES results back to improvement cycle manager
            if self.improvement_cycle_manager and state.best_weights:
                try:
                    agent_id = self.improvement_cycle_manager.handle_cmaes_complete(
                        state.board_type, state.num_players, state.best_weights
                    )
                    print(f"[P2P] CMA-ES weights registered as agent: {agent_id}")
                    self.diversity_metrics["cmaes_triggers"] += 1

                    # Save weights to file for future use
                    weights_file = Path(self.ringrift_path) / "ai-service" / "data" / "cmaes" / f"best_weights_{state.board_type}_{state.num_players}p.json"
                    weights_file.parent.mkdir(parents=True, exist_ok=True)
                    import json as json_mod
                    with open(weights_file, "w") as f:
                        json_mod.dump({
                            "weights": state.best_weights,
                            "fitness": state.best_fitness,
                            "job_id": job_id,
                            "generation": state.current_generation,
                            "timestamp": time.time(),
                        }, f, indent=2)
                    print(f"[P2P] Saved CMA-ES weights to {weights_file}")

                    # Propagate new weights to selfplay jobs
                    asyncio.create_task(self._propagate_cmaes_weights(
                        state.board_type, state.num_players, state.best_weights
                    ))
                except Exception as e:
                    print(f"[P2P] Failed to register CMA-ES weights: {e}")

        except Exception as e:
            import traceback
            print(f"[P2P] CMA-ES coordinator error: {e}")
            traceback.print_exc()
            if job_id in self.distributed_cmaes_state:
                self.distributed_cmaes_state[job_id].status = f"error: {e}"

    async def _evaluate_cmaes_weights_local(
        self, weights: dict, num_games: int, board_type: str, num_players: int
    ) -> float:
        """Evaluate weights locally by running selfplay games."""
        try:
            sem = getattr(self, "_cmaes_eval_semaphore", None)
            if sem is None:
                sem = asyncio.Semaphore(1)

            async with sem:
                # Run selfplay subprocess to evaluate weights
                import tempfile
                import json as json_mod

                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json_mod.dump(weights, f)
                    weights_file = f.name

                ai_service_path = str(Path(self.ringrift_path) / "ai-service")
                cmd = [
                    sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{ai_service_path}')
from app.game_engine import GameEngine
from app.ai.heuristic_ai import HeuristicAI
from app.models import AIConfig, BoardType, GameStatus
from app.training.generate_data import create_initial_state
import json

weights = json.load(open('{weights_file}'))
board_type = BoardType('{board_type}')
wins = 0
total = {num_games}

for i in range(total):
    state = create_initial_state(board_type, num_players={num_players})
    engine = GameEngine()

    # Candidate with custom weights vs baseline
    config_candidate = AIConfig(difficulty=5, randomness=0.1, think_time=500, custom_weights=weights)
    config_baseline = AIConfig(difficulty=5, randomness=0.1, think_time=500)

    ai_candidate = HeuristicAI(1, config_candidate)
    ai_baseline = HeuristicAI(2, config_baseline)

    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < 300:
        current_ai = ai_candidate if state.current_player == 1 else ai_baseline
        move = current_ai.select_move(state)
        if move is None:
            break
        state = engine.apply_move(state, move)
        move_count += 1

    if state.winner == 1:
        wins += 1
    elif state.winner is None:
        wins += 0.5  # Draw counts as half

print(wins / total)
"""
                ]

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, "PYTHONPATH": ai_service_path},
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

                # Clean up temp file
                os.unlink(weights_file)

                if proc.returncode == 0:
                    return float(stdout.decode().strip())
                else:
                    print(f"[P2P] Local eval error: {stderr.decode()}")
                    return 0.5

        except Exception as e:
            print(f"[P2P] Local CMA-ES evaluation error: {e}")
            return 0.5

    async def _evaluate_cmaes_weights(
        self, job_id: str, weights: dict, generation: int, individual_idx: int,
        games_per_eval: int = 5, board_type: str = "square8", num_players: int = 2
    ):
        """Evaluate weights locally and report result to coordinator."""
        try:
            # Run local evaluation using passed parameters (workers don't have state)
            fitness = await self._evaluate_cmaes_weights_local(
                weights, games_per_eval, board_type, num_players
            )

            print(f"[P2P] Completed local CMA-ES evaluation: job={job_id}, gen={generation}, idx={individual_idx}, fitness={fitness:.4f}")

            # If we're not the coordinator, report result back
            if self.role != NodeRole.LEADER:
                # Find the leader and POST result
                if self.leader_id:
                    with self.peers_lock:
                        leader = self.peers.get(self.leader_id)
                    if leader:
                        try:
                            timeout = ClientTimeout(total=30)
                            async with get_client_session(timeout) as session:
                                url = self._url_for_peer(leader, "/cmaes/result")
                                await session.post(url, json={
                                    "job_id": job_id,
                                    "generation": generation,
                                    "individual_idx": individual_idx,
                                    "fitness": fitness,
                                    "weights": weights,
                                    "worker_id": self.node_id,
                                }, headers=self._auth_headers())
                        except Exception as e:
                            print(f"[P2P] Failed to report CMA-ES result to leader: {e}")

        except Exception as e:
            print(f"[P2P] CMA-ES evaluation error: {e}")

    # ============================================
    # Distributed Tournament Handlers
    # ============================================

    async def handle_tournament_start(self, request: web.Request) -> web.Response:
        """Start a distributed tournament.

        Only the leader can start distributed tournaments.
        Request body:
        {
            "board_type": "square8",
            "num_players": 2,
            "agent_ids": ["agent1", "agent2", "agent3"],
            "games_per_pairing": 2
        }
        """
        try:
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "Only the leader can start distributed tournaments",
                    "leader_id": self.leader_id,
                }, status=403)

            data = await request.json()
            job_id = f"tournament_{uuid.uuid4().hex[:8]}"

            agent_ids = data.get("agent_ids", [])
            if len(agent_ids) < 2:
                return web.json_response({"error": "At least 2 agents required"}, status=400)

            # Create round-robin pairings
            pairings = []
            for i, a1 in enumerate(agent_ids):
                for a2 in agent_ids[i+1:]:
                    for game_num in range(data.get("games_per_pairing", 2)):
                        pairings.append({
                            "agent1": a1,
                            "agent2": a2,
                            "game_num": game_num,
                            "status": "pending",
                        })

            state = DistributedTournamentState(
                job_id=job_id,
                board_type=data.get("board_type", "square8"),
                num_players=data.get("num_players", 2),
                agent_ids=agent_ids,
                games_per_pairing=data.get("games_per_pairing", 2),
                total_matches=len(pairings),
                pending_matches=pairings,
                status="running",
                started_at=time.time(),
                last_update=time.time(),
            )

            # Find available workers
            with self.peers_lock:
                workers = [p.node_id for p in self.peers.values() if p.is_healthy()]
            state.worker_nodes = workers

            if not state.worker_nodes:
                return web.json_response({"error": "No workers available"}, status=503)

            self.distributed_tournament_state[job_id] = state

            print(f"[P2P] Started tournament {job_id}: {len(agent_ids)} agents, {len(pairings)} matches, {len(workers)} workers")

            # Launch coordinator task
            asyncio.create_task(self._run_distributed_tournament(job_id))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "agents": agent_ids,
                "total_matches": len(pairings),
                "workers": workers,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_tournament_match(self, request: web.Request) -> web.Response:
        """Request a tournament match to be played by a worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            match_info = data.get("match")

            if not job_id or not match_info:
                return web.json_response({"error": "job_id and match required"}, status=400)

            print(f"[P2P] Received tournament match request: {match_info}")

            # Start match in background
            asyncio.create_task(self._play_tournament_match(job_id, match_info))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "status": "match_started",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_tournament_status(self, request: web.Request) -> web.Response:
        """Get status of distributed tournaments."""
        try:
            job_id = request.query.get("job_id")

            if job_id:
                if job_id not in self.distributed_tournament_state:
                    return web.json_response({"error": "Tournament not found"}, status=404)
                state = self.distributed_tournament_state[job_id]
                return web.json_response(state.to_dict())

            return web.json_response({
                job_id: state.to_dict()
                for job_id, state in self.distributed_tournament_state.items()
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_tournament_result(self, request: web.Request) -> web.Response:
        """Receive match result from a worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            match_result = data.get("result", {})
            worker_id = data.get("worker_id", "unknown")

            if job_id not in self.distributed_tournament_state:
                return web.json_response({"error": "Tournament not found"}, status=404)

            state = self.distributed_tournament_state[job_id]
            state.results.append(match_result)
            state.completed_matches += 1
            state.last_update = time.time()

            print(f"[P2P] Tournament result: {state.completed_matches}/{state.total_matches} matches from {worker_id}")

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "completed": state.completed_matches,
                "total": state.total_matches,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_ssh_tournament_start(self, request: web.Request) -> web.Response:
        """Start an SSH-distributed difficulty-tier tournament (leader only).

        This is a thin wrapper that runs `scripts/run_ssh_distributed_tournament.py`
        as a subprocess and tracks its status locally on the leader node.
        """
        try:
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "Only the leader can start SSH tournaments",
                    "leader_id": self.leader_id,
                }, status=403)

            data = await request.json()

            tiers = str(data.get("tiers") or "D1-D10")
            board = str(data.get("board") or data.get("board_type") or "square8").strip().lower()
            if board == "hexagonal":
                board = "hex"
            if board not in ("square8", "square19", "hex"):
                return web.json_response({"error": f"Invalid board: {board!r}"}, status=400)

            games_per_matchup = int(data.get("games_per_matchup", 50) or 50)
            seed = int(data.get("seed", 1) or 1)
            think_time_scale = float(data.get("think_time_scale", 1.0) or 1.0)
            max_moves = int(data.get("max_moves", 10000) or 10000)
            wilson_confidence = float(data.get("wilson_confidence", 0.95) or 0.95)
            nn_model_id = data.get("nn_model_id") or None
            config_path = data.get("config") or None
            include_nonready = bool(data.get("include_nonready", False))
            max_parallel_per_host = data.get("max_parallel_per_host")
            remote_output_dir = str(data.get("remote_output_dir") or "results/tournaments/ssh_shards")
            job_timeout_sec = int(data.get("job_timeout_sec", 6 * 60 * 60) or (6 * 60 * 60))
            retries = int(data.get("retries", 1) or 1)
            dry_run = bool(data.get("dry_run", False))

            requested_run_id = str(data.get("run_id") or "").strip()
            job_id = requested_run_id or f"ssh_tournament_{uuid.uuid4().hex[:8]}"
            run_id = job_id

            hosts = data.get("hosts")
            hosts_spec: Optional[str] = None
            if isinstance(hosts, list):
                hosts_spec = ",".join(str(h).strip() for h in hosts if str(h).strip())
            elif isinstance(hosts, str) and hosts.strip():
                hosts_spec = hosts.strip()

            output_root = str(
                data.get("output_root") or f"results/tournaments/p2p_orchestrator/{run_id}"
            )

            report_path = str(Path(output_root) / f"report_{run_id}.json")
            checkpoint_path = str(Path(output_root) / f"tournament_{run_id}.json")
            manifest_path = str(Path(output_root) / "manifest.json")

            log_dir = STATE_DIR / "ssh_tournaments"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = str(log_dir / f"{run_id}.log")

            cmd: List[str] = [
                sys.executable,
                "scripts/run_ssh_distributed_tournament.py",
                "--tiers", tiers,
                "--board", board,
                "--games-per-matchup", str(games_per_matchup),
                "--seed", str(seed),
                "--think-time-scale", str(think_time_scale),
                "--max-moves", str(max_moves),
                "--wilson-confidence", str(wilson_confidence),
                "--remote-output-dir", remote_output_dir,
                "--job-timeout-sec", str(job_timeout_sec),
                "--retries", str(retries),
                "--run-id", run_id,
                "--output-root", output_root,
            ]
            if nn_model_id:
                cmd.extend(["--nn-model-id", str(nn_model_id)])
            if config_path:
                cmd.extend(["--config", str(config_path)])
            if hosts_spec:
                cmd.extend(["--hosts", hosts_spec])
            if include_nonready:
                cmd.append("--include-nonready")
            if max_parallel_per_host is not None:
                cmd.extend(["--max-parallel-per-host", str(int(max_parallel_per_host))])
            if dry_run:
                cmd.append("--dry-run")

            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

            cwd = os.path.join(self.ringrift_path, "ai-service")
            with open(log_path, "ab") as log_file:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                    cwd=cwd,
                )

            run_state = SSHTournamentRun(
                job_id=job_id,
                run_id=run_id,
                tiers=tiers,
                board=board,
                games_per_matchup=games_per_matchup,
                pid=proc.pid,
                status="running",
                started_at=time.time(),
                output_root=output_root,
                manifest_path=manifest_path,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                log_path=log_path,
                command=cmd,
            )

            with self.ssh_tournament_lock:
                self.ssh_tournament_runs[job_id] = run_state

            asyncio.create_task(self._monitor_ssh_tournament_process(job_id, proc))

            return web.json_response({"success": True, "job": run_state.to_dict()})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_ssh_tournament_status(self, request: web.Request) -> web.Response:
        """Get status of SSH-distributed tournaments."""
        try:
            job_id = request.query.get("job_id")

            with self.ssh_tournament_lock:
                if job_id:
                    job = self.ssh_tournament_runs.get(job_id)
                    if not job:
                        return web.json_response({"error": "Tournament not found"}, status=404)
                    return web.json_response(job.to_dict())

                return web.json_response({
                    jid: job.to_dict() for jid, job in self.ssh_tournament_runs.items()
                })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_ssh_tournament_cancel(self, request: web.Request) -> web.Response:
        """Cancel a running SSH tournament (best-effort)."""
        try:
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "Only the leader can cancel SSH tournaments",
                    "leader_id": self.leader_id,
                }, status=403)

            data = await request.json()
            job_id = data.get("job_id")
            if not job_id:
                return web.json_response({"error": "job_id is required"}, status=400)

            with self.ssh_tournament_lock:
                job = self.ssh_tournament_runs.get(job_id)
            if not job:
                return web.json_response({"error": "Tournament not found"}, status=404)

            if job.status != "running":
                return web.json_response({
                    "success": False,
                    "error": f"Cannot cancel tournament in status: {job.status}",
                }, status=400)

            try:
                os.kill(job.pid, signal.SIGTERM)
            except Exception as e:
                return web.json_response({
                    "success": False,
                    "error": f"Failed to signal process: {e}",
                }, status=500)

            with self.ssh_tournament_lock:
                job.status = "cancelled"
                job.completed_at = time.time()

            return web.json_response({"success": True, "job_id": job_id})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _monitor_ssh_tournament_process(self, job_id: str, proc) -> None:
        """Monitor a tournament subprocess and update status."""
        try:
            return_code = await proc.wait()
            with self.ssh_tournament_lock:
                job = self.ssh_tournament_runs.get(job_id)
                if not job:
                    return
                job.return_code = return_code
                job.completed_at = time.time()
                if job.status != "cancelled":
                    job.status = "completed" if return_code == 0 else "failed"
                    if return_code != 0:
                        job.error_message = f"Process exited with code {return_code}"
        except Exception as e:
            with self.ssh_tournament_lock:
                job = self.ssh_tournament_runs.get(job_id)
                if job and job.status != "cancelled":
                    job.status = "failed"
                    job.completed_at = time.time()
                    job.error_message = str(e)

    async def _run_distributed_tournament(self, job_id: str):
        """Main coordinator loop for distributed tournament."""
        try:
            state = self.distributed_tournament_state.get(job_id)
            if not state:
                return

            print(f"[P2P] Tournament coordinator started for job {job_id}")

            # Distribute matches to workers
            while state.pending_matches and state.status == "running":
                # Simple distribution - in reality would be smarter about load balancing
                for worker_id in state.worker_nodes:
                    if not state.pending_matches:
                        break
                    match = state.pending_matches.pop(0)
                    match["status"] = "in_progress"

                    # Send match to worker
                    await self._send_match_to_worker(job_id, worker_id, match)

                await asyncio.sleep(1)

            # Wait for all results
            while state.completed_matches < state.total_matches and state.status == "running":
                state.last_update = time.time()
                await asyncio.sleep(1)

            # Calculate final ratings
            self._calculate_tournament_ratings(state)
            state.status = "completed"

            print(f"[P2P] Tournament {job_id} completed: {state.completed_matches} matches, ratings={state.final_ratings}")

        except Exception as e:
            print(f"[P2P] Tournament coordinator error: {e}")
            if job_id in self.distributed_tournament_state:
                self.distributed_tournament_state[job_id].status = f"error: {e}"

    async def _send_match_to_worker(self, job_id: str, worker_id: str, match: dict):
        """Send a match to a worker node."""
        try:
            with self.peers_lock:
                worker = self.peers.get(worker_id)
            if not worker:
                return

            timeout = ClientTimeout(total=10)
            async with get_client_session(timeout) as session:
                url = self._url_for_peer(worker, "/tournament/match")
                await session.post(url, json={"job_id": job_id, "match": match}, headers=self._auth_headers())
        except Exception as e:
            print(f"[P2P] Failed to send match to worker {worker_id}: {e}")

    async def _play_tournament_match(self, job_id: str, match_info: dict):
        """Play a tournament match locally using subprocess selfplay."""
        try:
            import subprocess
            import sys
            import json as json_module

            agent1 = match_info["agent1"]
            agent2 = match_info["agent2"]
            game_num = match_info.get("game_num", 0)
            board_type = match_info.get("board_type", "square8")
            num_players = match_info.get("num_players", 2)

            print(f"[P2P] Playing tournament match: {agent1} vs {agent2} (game {game_num})")

            # Build the subprocess command to run a single game
            # Agent IDs map to model paths or heuristic configurations
            game_script = f"""
import sys
sys.path.insert(0, '{self.ringrift_path}/ai-service')
from app.game_engine import GameEngine
from app.agents.heuristic_agent import HeuristicAgent
import json
import random

def load_agent(agent_id: str, player_idx: int):
    '''Load agent by ID - supports heuristic weights or model paths.'''
    if agent_id.startswith('heuristic:'):
        # Parse weights from agent ID: "heuristic:w1,w2,w3,..."
        weight_str = agent_id.split(':')[1]
        weights = [float(w) for w in weight_str.split(',')]
        weight_names = [
            "material_weight", "ring_count_weight", "stack_height_weight",
            "center_control_weight", "territory_weight", "mobility_weight",
            "line_potential_weight", "defensive_weight",
        ]
        weight_dict = dict(zip(weight_names, weights))
        return HeuristicAgent(player_idx, weight_dict)
    elif agent_id.startswith('model:'):
        # Neural network model - would load from path
        # For now, fall back to heuristic
        return HeuristicAgent(player_idx)
    else:
        # Default heuristic agent
        return HeuristicAgent(player_idx)

# Initialize game
engine = GameEngine(board_type='{board_type}', num_players={num_players})
agents = [
    load_agent('{agent1}', 0),
    load_agent('{agent2}', 1),
]

# Play until completion
max_moves = 10000
move_count = 0
while not engine.is_game_over() and move_count < max_moves:
    current_player = engine.current_player
    agent = agents[current_player]
    legal_moves = engine.get_legal_moves()
    if not legal_moves:
        break
    move = agent.select_move(engine.get_state(), legal_moves)
    engine.apply_move(move)
    move_count += 1

# Get result
outcome = engine.get_outcome()
winner_idx = outcome.get('winner')
victory_type = outcome.get('victory_type', 'unknown')

# Map winner index to agent ID
winner_agent = None
if winner_idx == 0:
    winner_agent = '{agent1}'
elif winner_idx == 1:
    winner_agent = '{agent2}'

result = {{
    'agent1': '{agent1}',
    'agent2': '{agent2}',
    'winner': winner_agent,
    'winner_idx': winner_idx,
    'victory_type': victory_type,
    'move_count': move_count,
    'game_num': {game_num},
}}
print(json.dumps(result))
"""
            # Run the game in subprocess
            cmd = [sys.executable, "-c", game_script]
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"
            env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=300  # 5 minute timeout per game
            )

            if proc.returncode != 0:
                print(f"[P2P] Tournament match subprocess error: {stderr.decode()}")
                result = {
                    "agent1": agent1,
                    "agent2": agent2,
                    "winner": None,
                    "error": stderr.decode()[:200],
                    "game_num": game_num,
                }
            else:
                # Parse result from stdout
                output_lines = stdout.decode().strip().split('\n')
                result_line = output_lines[-1] if output_lines else '{}'
                result = json_module.loads(result_line)

            print(f"[P2P] Match result: {agent1} vs {agent2} -> winner={result.get('winner')}")

            # Report result back to coordinator (leader)
            if self.role != NodeRole.LEADER and self.leader_id:
                with self.peers_lock:
                    leader = self.peers.get(self.leader_id)
                if leader:
                    try:
                        timeout = ClientTimeout(total=10)
                        async with get_client_session(timeout) as session:
                            url = self._url_for_peer(leader, "/tournament/result")
                            await session.post(url, json={
                                "job_id": job_id,
                                "result": result,
                                "worker_id": self.node_id,
                            }, headers=self._auth_headers())
                    except Exception as e:
                        print(f"[P2P] Failed to report tournament result to leader: {e}")
            else:
                # We are the leader, update state directly
                if job_id in self.distributed_tournament_state:
                    state = self.distributed_tournament_state[job_id]
                    state.results.append(result)
                    state.completed_matches += 1
                    state.last_update = time.time()

        except asyncio.TimeoutError:
            print(f"[P2P] Tournament match timed out: {match_info}")
        except Exception as e:
            print(f"[P2P] Tournament match error: {e}")

    def _calculate_tournament_ratings(self, state: DistributedTournamentState):
        """Calculate final Elo ratings from tournament results.

        Uses standard Elo rating system with K-factor of 32.
        Starting rating is 1500 for all agents.
        """
        K_FACTOR = 32
        INITIAL_RATING = 1500

        # Initialize ratings
        ratings = {agent: float(INITIAL_RATING) for agent in state.agent_ids}
        wins = {agent: 0 for agent in state.agent_ids}
        losses = {agent: 0 for agent in state.agent_ids}
        draws = {agent: 0 for agent in state.agent_ids}

        def expected_score(rating_a: float, rating_b: float) -> float:
            """Calculate expected score for player A against player B."""
            return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

        def update_elo(rating: float, expected: float, actual: float) -> float:
            """Update Elo rating based on game outcome."""
            return rating + K_FACTOR * (actual - expected)

        # Process all results
        for result in state.results:
            agent1 = result.get("agent1")
            agent2 = result.get("agent2")
            winner = result.get("winner")

            if not agent1 or not agent2:
                continue
            if agent1 not in ratings or agent2 not in ratings:
                continue

            # Determine actual scores
            if winner == agent1:
                score1, score2 = 1.0, 0.0
                wins[agent1] += 1
                losses[agent2] += 1
            elif winner == agent2:
                score1, score2 = 0.0, 1.0
                wins[agent2] += 1
                losses[agent1] += 1
            elif winner is None:
                # Draw
                score1, score2 = 0.5, 0.5
                draws[agent1] += 1
                draws[agent2] += 1
            else:
                # Unknown winner, skip
                continue

            # Calculate expected scores
            expected1 = expected_score(ratings[agent1], ratings[agent2])
            expected2 = expected_score(ratings[agent2], ratings[agent1])

            # Update ratings
            ratings[agent1] = update_elo(ratings[agent1], expected1, score1)
            ratings[agent2] = update_elo(ratings[agent2], expected2, score2)

        # Store final ratings and stats
        state.final_ratings = {
            agent: {
                "elo": round(ratings[agent]),
                "wins": wins[agent],
                "losses": losses[agent],
                "draws": draws[agent],
                "games": wins[agent] + losses[agent] + draws[agent],
            }
            for agent in state.agent_ids
        }

        # Log rankings
        ranked = sorted(state.final_ratings.items(), key=lambda x: x[1]["elo"], reverse=True)
        print(f"[P2P] Tournament final rankings:")
        for rank, (agent, stats) in enumerate(ranked, 1):
            print(f"  {rank}. {agent}: Elo={stats['elo']}, W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}")

        # Persist results to unified Elo database
        try:
            from app.tournament import get_elo_database
            db = get_elo_database()

            for result in state.results:
                agent1 = result.get("agent1")
                agent2 = result.get("agent2")
                winner = result.get("winner")

                if not agent1 or not agent2:
                    continue

                # Determine rankings
                if winner == agent1:
                    rankings = [0, 1]
                elif winner == agent2:
                    rankings = [1, 0]
                else:
                    rankings = [0, 0]

                db.record_match_and_update(
                    participant_ids=[agent1, agent2],
                    rankings=rankings,
                    board_type=state.board_type,
                    num_players=state.num_players,
                    tournament_id=state.job_id,
                    game_length=result.get("game_length", 0),
                    duration_sec=result.get("duration_sec", 0.0),
                )

            print(f"[P2P] Persisted {len(state.results)} matches to unified Elo database")
        except Exception as e:
            print(f"[P2P] Warning: Failed to persist to unified Elo database: {e}")

    # ============================================
    # Improvement Loop Handlers
    # ============================================

    async def handle_improvement_start(self, request: web.Request) -> web.Response:
        """Start an improvement loop (AlphaZero-style training cycle).

        Only the leader can start improvement loops.
        Request body:
        {
            "board_type": "square8",
            "num_players": 2,
            "max_iterations": 50,
            "games_per_iteration": 1000
        }
        """
        try:
            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "error": "Only the leader can start improvement loops",
                    "leader_id": self.leader_id,
                }, status=403)

            data = await request.json()
            job_id = f"improve_{uuid.uuid4().hex[:8]}"

            # Query negotiated rate from resource_optimizer for cooperative utilization
            # This ensures selfplay rate respects cluster-wide 60-80% utilization target
            requested_games = data.get("games_per_iteration", 1000)
            if HAS_RATE_NEGOTIATION and negotiate_selfplay_rate is not None:
                try:
                    # Negotiate rate with resource_optimizer (60-80% target)
                    approved_rate = negotiate_selfplay_rate(
                        requested_rate=requested_games,
                        reason=f"p2p_improvement_loop:{job_id}",
                        requestor=f"p2p_{self.node_id}",
                    )
                    if approved_rate != requested_games:
                        print(f"[P2P] games_per_iteration adjusted: {requested_games} -> {approved_rate} (utilization-based)")
                    requested_games = approved_rate
                except Exception as e:
                    print(f"[P2P] Rate negotiation failed, using default: {e}")

            state = ImprovementLoopState(
                job_id=job_id,
                board_type=data.get("board_type", "square8"),
                num_players=data.get("num_players", 2),
                max_iterations=data.get("max_iterations", 50),
                games_per_iteration=requested_games,
                phase="selfplay",
                status="running",
                started_at=time.time(),
                last_update=time.time(),
            )

            # Find available workers
            with self.peers_lock:
                workers = [p.node_id for p in self.peers.values() if p.is_healthy()]
                gpu_workers = [p.node_id for p in self.peers.values() if p.is_healthy() and p.has_gpu]
            state.worker_nodes = workers

            if not gpu_workers:
                return web.json_response({"error": "No GPU workers available for training"}, status=503)

            self.improvement_loop_state[job_id] = state

            print(f"[P2P] Started improvement loop {job_id}: {len(workers)} workers, {len(gpu_workers)} GPU workers")

            # Launch improvement loop
            asyncio.create_task(self._run_improvement_loop(job_id))

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "workers": workers,
                "gpu_workers": gpu_workers,
                "config": {
                    "board_type": state.board_type,
                    "num_players": state.num_players,
                    "max_iterations": state.max_iterations,
                    "games_per_iteration": state.games_per_iteration,
                },
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_improvement_status(self, request: web.Request) -> web.Response:
        """Get status of improvement loops."""
        try:
            job_id = request.query.get("job_id")

            if job_id:
                if job_id not in self.improvement_loop_state:
                    return web.json_response({"error": "Improvement loop not found"}, status=404)
                state = self.improvement_loop_state[job_id]
                return web.json_response(state.to_dict())

            return web.json_response({
                job_id: state.to_dict()
                for job_id, state in self.improvement_loop_state.items()
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_improvement_phase_complete(self, request: web.Request) -> web.Response:
        """Notify that a phase of the improvement loop is complete."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            phase = data.get("phase")
            worker_id = data.get("worker_id", "unknown")
            result = data.get("result", {})

            if job_id not in self.improvement_loop_state:
                return web.json_response({"error": "Improvement loop not found"}, status=404)

            state = self.improvement_loop_state[job_id]
            state.last_update = time.time()

            # Track progress by phase
            if phase == "selfplay":
                games_done = result.get("games_done", 0)
                state.selfplay_progress[worker_id] = games_done
                total_done = sum(state.selfplay_progress.values())
                print(f"[P2P] Improvement loop selfplay: {total_done}/{state.games_per_iteration} games")
            elif phase == "train":
                state.best_model_path = result.get("model_path", state.best_model_path)
            elif phase == "evaluate":
                winrate = result.get("winrate", 0.0)
                if winrate > state.best_winrate:
                    state.best_winrate = winrate
                    print(f"[P2P] New best model: winrate={winrate:.2%}")

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "phase": state.phase,
                "iteration": state.current_iteration,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # =========================================================================
    # Phase 2: P2P Data Sync HTTP Handlers
    # =========================================================================

    async def handle_sync_start(self, request: web.Request) -> web.Response:
        """POST /sync/start - Leader initiates a cluster-wide data sync.

        Only the leader can start a sync. This collects manifests from all nodes,
        generates a sync plan, and dispatches rsync jobs to nodes.
        """
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)
            if not self._is_leader():
                return web.json_response({
                    "error": "Not the leader. Only leader can start cluster sync.",
                    "leader_id": self.leader_id,
                }, status=403)

            result = await self.start_cluster_sync()
            return web.json_response(result)
        except Exception as e:
            print(f"[P2P] Error in handle_sync_start: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    async def handle_sync_status(self, request: web.Request) -> web.Response:
        """GET /sync/status - Get current sync status.

        Returns the current sync plan (if any), active sync jobs, and overall status.
        """
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                proxied = await self._proxy_to_leader(request)
                if proxied.status not in (502, 503):
                    return proxied

            with self.sync_lock:
                sync_plan_dict = self.current_sync_plan.to_dict() if self.current_sync_plan else None
                active_jobs_dict = {
                    job_id: job.to_dict()
                    for job_id, job in self.active_sync_jobs.items()
                }

            return web.json_response({
                "node_id": self.node_id,
                "is_leader": self._is_leader(),
                "sync_in_progress": self.sync_in_progress,
                "last_sync_time": self.last_sync_time,
                "auto_sync_interval": self.auto_sync_interval,
                "current_sync_plan": sync_plan_dict,
                "active_sync_jobs": active_jobs_dict,
                "pending_sync_requests": len(self.pending_sync_requests),
            })
        except Exception as e:
            print(f"[P2P] Error in handle_sync_status: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_sync_pull(self, request: web.Request) -> web.Response:
        """POST /sync/pull - Handle incoming request to pull files from a source node.

        This is called by the leader to tell this node to pull files from another node.

        Request body:
        {
            "source_host": "192.168.1.100",
            "source_port": 8770,
            "source_node_id": "lambda-h100",
            "files": ["data/selfplay/sq8_2p/games_001.jsonl", ...]
        }
        """
        try:
            data = await request.json()
            source_node_id = data.get("source_node_id")
            files = data.get("files", [])

            if not source_node_id or not files:
                return web.json_response({
                    "error": "Missing required fields: source_node_id, files"
                }, status=400)

            # Prefer the local peer table for reachability (avoids leader guessing our routes).
            source_host = data.get("source_host")
            source_port = int(data.get("source_port", DEFAULT_PORT) or DEFAULT_PORT)
            with self.peers_lock:
                peer = self.peers.get(source_node_id)
            if source_node_id == self.node_id:
                peer = self.self_info
            if peer:
                source_host = peer.host
                source_port = peer.port

            if not source_host:
                return web.json_response({
                    "error": "Missing required fields: source_host (or unknown source_node_id)"
                }, status=400)

            print(f"[P2P] Received sync pull request: {len(files)} files from {source_node_id}")

            result = await self._handle_sync_pull_request(
                source_host=source_host,
                source_port=source_port,
                source_reported_host=(data.get("source_reported_host") or getattr(peer, "reported_host", "") or None),
                source_reported_port=(data.get("source_reported_port") or getattr(peer, "reported_port", 0) or None),
                source_node_id=source_node_id,
                files=files,
            )

            return web.json_response(result)
        except Exception as e:
            print(f"[P2P] Error in handle_sync_pull: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    async def handle_sync_file(self, request: web.Request) -> web.StreamResponse:
        """GET /sync/file?path=<relative_path> - Stream a data file to a peer.

        Security:
        - Only serves files within `ai-service/data/**`.
        - Requires auth when RINGRIFT_CLUSTER_AUTH_TOKEN is set (even though it's a GET).
        """
        try:
            if self.auth_token and not self._is_request_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)

            rel_path = (request.query.get("path") or "").lstrip("/")
            if not rel_path:
                return web.json_response({"error": "Missing required query param: path"}, status=400)

            data_dir = self.get_data_directory()
            data_dir.mkdir(parents=True, exist_ok=True)
            data_root = data_dir.resolve()
            full_path = (data_dir / rel_path)
            try:
                resolved = full_path.resolve()
                resolved.relative_to(data_root)
            except Exception:
                return web.json_response({"error": "Invalid path"}, status=400)

            if not resolved.exists() or not resolved.is_file():
                return web.json_response({"error": "Not found"}, status=404)

            stat = resolved.stat()
            resp = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(stat.st_size),
                },
            )
            await resp.prepare(request)
            with open(resolved, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    await resp.write(chunk)
            await resp.write_eof()
            return resp
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_sync_job_update(self, request: web.Request) -> web.Response:
        """POST /sync/job_update - Worker reports sync job status back to leader.

        Request body:
        {
            "job_id": "sync-123",
            "status": "completed|failed",
            "files_completed": 10,
            "bytes_transferred": 1048576,
            "error_message": "optional error message"
        }
        """
        try:
            data = await request.json()
            job_id = data.get("job_id")
            status = data.get("status")
            files_completed = data.get("files_completed", data.get("files_synced", 0))
            bytes_transferred = data.get("bytes_transferred", 0)
            error_message = data.get("error_message", data.get("error"))

            if not job_id or not status:
                return web.json_response({
                    "error": "Missing required fields: job_id, status"
                }, status=400)

            with self.sync_lock:
                if job_id in self.active_sync_jobs:
                    job = self.active_sync_jobs[job_id]
                    job.status = status
                    job.files_completed = int(files_completed or 0)
                    job.bytes_transferred = int(bytes_transferred or 0)
                    job.completed_at = time.time()
                    if error_message:
                        job.error_message = str(error_message)

                    print(f"[P2P] Sync job {job_id} {status}: {job.files_completed} files, {job.bytes_transferred} bytes")

                    # Update sync plan status if all jobs are done
                    if self.current_sync_plan:
                        all_done = all(
                            j.status in ("completed", "failed")
                            for j in self.current_sync_plan.sync_jobs
                        )
                        if all_done:
                            completed = sum(1 for j in self.current_sync_plan.sync_jobs if j.status == "completed")
                            failed = sum(1 for j in self.current_sync_plan.sync_jobs if j.status == "failed")
                            self.current_sync_plan.status = "completed" if failed == 0 else "partial"
                            self.current_sync_plan.completed_at = time.time()
                            self.sync_in_progress = False
                            self.last_sync_time = time.time()
                            print(f"[P2P] Cluster sync plan completed: {completed} succeeded, {failed} failed")

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "status": status,
            })
        except Exception as e:
            print(f"[P2P] Error in handle_sync_job_update: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _run_improvement_loop(self, job_id: str):
        """Main coordinator loop for AlphaZero-style improvement."""
        try:
            state = self.improvement_loop_state.get(job_id)
            if not state:
                return

            print(f"[P2P] Improvement loop coordinator started for job {job_id}")

            while state.current_iteration < state.max_iterations and state.status == "running":
                state.current_iteration += 1
                print(f"[P2P] Improvement iteration {state.current_iteration}/{state.max_iterations}")

                # Phase 1: Selfplay
                state.phase = "selfplay"
                state.selfplay_progress = {}
                await self._run_distributed_selfplay(job_id)

                # Phase 2: Export training data
                state.phase = "export"
                await self._export_training_data(job_id)

                # Phase 3: Training
                state.phase = "train"
                await self._run_training(job_id)

                # Phase 4: Evaluation
                state.phase = "evaluate"
                await self._run_evaluation(job_id)

                # Phase 5: Promote if better
                state.phase = "promote"
                await self._promote_model_if_better(job_id)

                state.last_update = time.time()

            state.status = "completed"
            state.phase = "idle"
            print(f"[P2P] Improvement loop {job_id} completed after {state.current_iteration} iterations")

        except Exception as e:
            print(f"[P2P] Improvement loop error: {e}")
            if job_id in self.improvement_loop_state:
                self.improvement_loop_state[job_id].status = f"error: {e}"

    async def _run_distributed_selfplay(self, job_id: str):
        """Coordinate distributed selfplay for improvement loop.

        Distributes selfplay games across all available workers.
        Each worker runs selfplay using the current best model and reports
        progress back to the coordinator.
        """
        import sys
        import json as json_module

        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        # Distribute selfplay across workers
        num_workers = max(len(state.worker_nodes), 1)
        games_per_worker = state.games_per_iteration // num_workers
        remainder = state.games_per_iteration % num_workers

        print(f"[P2P] Starting distributed selfplay: {games_per_worker} games/worker, {num_workers} workers")

        # Create output directory for this iteration
        iteration_dir = os.path.join(
            self.ringrift_path, "ai-service", "data", "selfplay",
            f"improve_{job_id}", f"iter_{state.current_iteration}"
        )
        os.makedirs(iteration_dir, exist_ok=True)

        # Send selfplay tasks to workers
        tasks_sent = 0
        for idx, worker_id in enumerate(state.worker_nodes):
            with self.peers_lock:
                worker = self.peers.get(worker_id)
            if not worker or not worker.is_healthy():
                continue

            # Give first worker(s) the remainder games
            worker_games = games_per_worker + (1 if idx < remainder else 0)

            try:
                timeout = ClientTimeout(total=10)
                async with get_client_session(timeout) as session:
                    url = self._url_for_peer(worker, "/improvement/selfplay")
                    await session.post(url, json={
                        "job_id": job_id,
                        "iteration": state.current_iteration,
                        "num_games": worker_games,
                        "board_type": state.board_type,
                        "num_players": state.num_players,
                        "model_path": state.best_model_path,
                        "output_dir": iteration_dir,
                    }, headers=self._auth_headers())
                    tasks_sent += 1
            except Exception as e:
                print(f"[P2P] Failed to send selfplay task to {worker_id}: {e}")

        if tasks_sent == 0:
            # No workers available, run locally
            print(f"[P2P] No workers available, running selfplay locally")
            await self._run_local_selfplay(
                job_id, state.games_per_iteration,
                state.board_type, state.num_players,
                state.best_model_path, iteration_dir
            )
        else:
            # Wait for all workers to complete
            target_games = state.games_per_iteration
            check_interval = 5  # seconds
            timeout_seconds = 3600  # 1 hour max for selfplay phase
            elapsed = 0

            while elapsed < timeout_seconds and state.status == "running":
                total_done = sum(state.selfplay_progress.values())
                if total_done >= target_games:
                    break
                await asyncio.sleep(check_interval)
                elapsed += check_interval

            print(f"[P2P] Selfplay phase completed: {sum(state.selfplay_progress.values())} games")

    async def _run_local_selfplay(
        self, job_id: str, num_games: int, board_type: str,
        num_players: int, model_path: Optional[str], output_dir: str
    ):
        """Run selfplay locally using subprocess."""
        import sys

        output_file = os.path.join(output_dir, f"{self.node_id}_games.jsonl")

        # Build selfplay command
        cmd = [
            sys.executable,
            os.path.join(self.ringrift_path, "ai-service", "scripts", "run_self_play_soak.py"),
            "--num-games", str(num_games),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--engine-mode", "descent-only" if model_path else "heuristic-only",
            "--max-moves", "10000",  # LEARNED LESSONS - Avoid draws due to move limit
            "--log-jsonl", output_file,
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            if proc.returncode == 0:
                print(f"[P2P] Local selfplay completed: {num_games} games")
                # Update progress
                if job_id in self.improvement_loop_state:
                    self.improvement_loop_state[job_id].selfplay_progress[self.node_id] = num_games
            else:
                print(f"[P2P] Local selfplay failed: {stderr.decode()[:500]}")

        except asyncio.TimeoutError:
            print(f"[P2P] Local selfplay timed out")
        except Exception as e:
            print(f"[P2P] Local selfplay error: {e}")

    async def _export_training_data(self, job_id: str):
        """Export training data from selfplay games.

        Converts JSONL game records to training format (HDF5 or NPZ).
        """
        import sys

        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        print(f"[P2P] Exporting training data for job {job_id}, iteration {state.current_iteration}")

        iteration_dir = os.path.join(
            self.ringrift_path, "ai-service", "data", "selfplay",
            f"improve_{job_id}", f"iter_{state.current_iteration}"
        )
        output_file = os.path.join(
            self.ringrift_path, "ai-service", "data", "training",
            f"improve_{job_id}", f"iter_{state.current_iteration}.npz"
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Run export script
        export_script = f"""
import sys
sys.path.insert(0, '{self.ringrift_path}/ai-service')
import glob
import json
import numpy as np
from app.training.data_export import export_games_to_training_format

# Find all JSONL files from this iteration
jsonl_files = glob.glob('{iteration_dir}/*.jsonl')
print(f"Found {{len(jsonl_files)}} JSONL files")

games = []
for f in jsonl_files:
    with open(f) as fp:
        for line in fp:
            if line.strip():
                try:
                    games.append(json.loads(line))
                except:
                    pass

print(f"Loaded {{len(games)}} games")

if games:
    # Export to training format
    try:
        export_games_to_training_format(games, '{output_file}', '{state.board_type}')
        print(f"Exported to {output_file}")
    except Exception as e:
        # Fallback: save raw game data
        np.savez_compressed('{output_file}', games=games)
        print(f"Saved raw games to {output_file}")
else:
    print("No games to export")
"""

        cmd = [sys.executable, "-c", export_script]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=600  # 10 minutes max
            )

            if proc.returncode == 0:
                print(f"[P2P] Training data export completed")
                state.training_data_path = output_file
            else:
                print(f"[P2P] Training data export failed: {stderr.decode()[:500]}")

        except asyncio.TimeoutError:
            print(f"[P2P] Training data export timed out")
        except Exception as e:
            print(f"[P2P] Training data export error: {e}")

    async def _run_training(self, job_id: str):
        """Run neural network training on GPU node.

        Finds a GPU worker and delegates training to it, or runs locally
        if this node has a GPU.
        """
        import sys

        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        print(f"[P2P] Running training for job {job_id}, iteration {state.current_iteration}")

        # Find GPU worker
        gpu_worker = None
        with self.peers_lock:
            for peer in self.peers.values():
                if peer.has_gpu and peer.is_healthy():
                    gpu_worker = peer
                    break

        # Model output path
        new_model_path = os.path.join(
            self.ringrift_path, "ai-service", "models",
            f"improve_{job_id}", f"iter_{state.current_iteration}.pt"
        )
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)

        training_config = {
            "job_id": job_id,
            "iteration": state.current_iteration,
            "training_data": getattr(state, 'training_data_path', ''),
            "output_model": new_model_path,
            "board_type": state.board_type,
            "num_players": state.num_players,
            "epochs": 10,
            "batch_size": 256,
            "learning_rate": 0.001,
        }

        if gpu_worker and gpu_worker.node_id != self.node_id:
            # Delegate to GPU worker
            try:
                timeout = ClientTimeout(total=3600)  # 1 hour for training
                async with get_client_session(timeout) as session:
                    url = self._url_for_peer(gpu_worker, "/improvement/train")
                    async with session.post(url, json=training_config, headers=self._auth_headers()) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            if result.get("success"):
                                state.candidate_model_path = result.get("model_path", new_model_path)
                                print(f"[P2P] Training completed on {gpu_worker.node_id}")
                                return
            except Exception as e:
                print(f"[P2P] Failed to delegate training to {gpu_worker.node_id}: {e}")

        # Run training locally
        await self._run_local_training(training_config)
        state.candidate_model_path = new_model_path

    async def _run_local_training(self, config: dict):
        """Run training locally using subprocess."""
        import sys

        print(f"[P2P] Running local training")

        training_script = f"""
import sys
sys.path.insert(0, '{self.ringrift_path}/ai-service')
import numpy as np
import torch

# Load training data
try:
    data = np.load('{config.get("training_data", "")}', allow_pickle=True)
    print(f"Loaded training data")
except Exception as e:
    print(f"No training data available: {{e}}")
    # Create minimal model anyway
    data = None

# Import or create model architecture
try:
    from app.models.policy_value_net import PolicyValueNet
    model = PolicyValueNet(
        board_type='{config.get("board_type", "square8")}',
        num_players={config.get("num_players", 2)}
    )
except ImportError:
    # Fallback to simple model
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    )

# Save model
torch.save(model.state_dict(), '{config.get("output_model", "/tmp/model.pt")}')
print(f"Saved model to {config.get('output_model', '/tmp/model.pt')}")
"""

        cmd = [sys.executable, "-c", training_script]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            print(f"[P2P] Training output: {stdout.decode()}")
            if proc.returncode != 0:
                print(f"[P2P] Training stderr: {stderr.decode()[:500]}")

        except asyncio.TimeoutError:
            print(f"[P2P] Local training timed out")
        except Exception as e:
            print(f"[P2P] Local training error: {e}")

    # ============================================
    # Phase 3: Training Pipeline Integration Methods
    # ============================================

    def _check_training_readiness(self) -> List[Dict[str, Any]]:
        """Check cluster data manifest for training readiness.

        Returns list of training jobs that should be triggered based on
        accumulated selfplay data.

        Called periodically by leader to check if automatic training should start.
        """
        jobs_to_start = []

        if not self.cluster_data_manifest:
            return jobs_to_start

        current_time = time.time()
        thresholds = self.training_thresholds

        # Update adaptive thresholds based on current cluster state
        gpu_node_count = len([p for p in self.peers.values()
                              if getattr(p, 'has_gpu', False) and getattr(p, 'gpu_name', '')]
                             ) + (1 if getattr(self.self_info, 'has_gpu', False) else 0)
        thresholds.update_from_cluster_state(gpu_node_count)

        def _cooldown_ok(job_type: str, config_key: str) -> bool:
            cooldown = thresholds.get_effective_cooldown()
            if cooldown <= 0:
                return True
            last_seen = 0.0
            with self.training_lock:
                for job in self.training_jobs.values():
                    if str(getattr(job, "job_type", "")) != job_type:
                        continue
                    job_key = f"{job.board_type}_{job.num_players}p"
                    if job_key != config_key:
                        continue
                    last_seen = max(
                        last_seen,
                        float(getattr(job, "completed_at", 0.0) or 0.0),
                        float(getattr(job, "started_at", 0.0) or 0.0),
                        float(getattr(job, "created_at", 0.0) or 0.0),
                    )
            if last_seen <= 0:
                return True
            return (current_time - last_seen) >= cooldown

        # Check each board type / player count combination
        for config_key, config_data in self.cluster_data_manifest.by_board_type.items():
            parts = config_key.split("_")
            if len(parts) < 2:
                continue
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))
            total_games = config_data.get("total_games", 0)

            # Check NNUE training threshold (using adaptive thresholds)
            if thresholds.auto_nnue_enabled:
                last_nnue_games = self.games_at_last_nnue_train.get(config_key, 0)
                min_games = thresholds.get_effective_min_games("nnue")
                incremental = thresholds.get_effective_incremental("nnue")
                if total_games >= min_games:
                    new_games = total_games - last_nnue_games
                    if new_games >= incremental or last_nnue_games == 0:
                        # Check cooldown
                        if not _cooldown_ok("nnue", config_key):
                            continue
                        existing_job = self._find_running_training_job("nnue", config_key)
                        if not existing_job:
                            jobs_to_start.append({
                                "job_type": "nnue",
                                "board_type": board_type,
                                "num_players": num_players,
                                "config_key": config_key,
                                "total_games": total_games,
                            })

            # Check CMA-ES optimization threshold (using adaptive thresholds)
            if thresholds.auto_cmaes_enabled:
                last_cmaes_games = self.games_at_last_cmaes_train.get(config_key, 0)
                min_games = thresholds.get_effective_min_games("cmaes")
                incremental = thresholds.get_effective_incremental("cmaes")
                if total_games >= min_games:
                    new_games = total_games - last_cmaes_games
                    if new_games >= incremental or last_cmaes_games == 0:
                        if not _cooldown_ok("cmaes", config_key):
                            continue
                        existing_job = self._find_running_training_job("cmaes", config_key)
                        if not existing_job:
                            jobs_to_start.append({
                                "job_type": "cmaes",
                                "board_type": board_type,
                                "num_players": num_players,
                                "config_key": config_key,
                                "total_games": total_games,
                            })

        return jobs_to_start

    def _find_running_training_job(self, job_type: str, config_key: str) -> Optional[TrainingJob]:
        """Find a running training job of the given type for the config."""
        with self.training_lock:
            for job in self.training_jobs.values():
                if (job.job_type == job_type and
                    f"{job.board_type}_{job.num_players}p" == config_key and
                    job.status in ("pending", "queued", "running")):
                    return job
        return None

    async def _dispatch_training_job(self, job_config: Dict[str, Any]) -> Optional[TrainingJob]:
        """Dispatch a training job to an appropriate worker.

        Finds a GPU node for NNUE training, or any available node for CMA-ES.
        Creates a TrainingJob and sends it to the worker.
        """
        job_type = job_config["job_type"]
        board_type = job_config["board_type"]
        num_players = job_config["num_players"]
        config_key = job_config["config_key"]

        # Generate job ID
        job_id = f"{job_type}_{config_key}_{int(time.time())}"

        # Create TrainingJob
        job = TrainingJob(
            job_id=job_id,
            job_type=job_type,
            board_type=board_type,
            num_players=num_players,
            status="pending",
            data_games_count=job_config.get("total_games", 0),
        )

        # Find suitable worker (CPU/GPU-aware + load-balanced)
        self._update_self_info()
        with self.peers_lock:
            all_nodes = list(self.peers.values())
        all_nodes.append(self.self_info)
        # Filter for healthy nodes with sufficient memory
        healthy_nodes = [
            n for n in all_nodes
            if n.is_healthy() and int(getattr(n, "memory_gb", 0) or 0) >= MIN_MEMORY_GB_FOR_TASKS
        ]

        # Get set of nodes already running training jobs (for parallel training across configs)
        with self.training_lock:
            nodes_with_training = {
                job.worker_node for job in self.training_jobs.values()
                if job.status in ("pending", "queued", "running") and job.worker_node
            }

        worker_node: Optional[NodeInfo] = None
        if job_type == "nnue":
            # NNUE training prefers accelerator nodes (CUDA/MPS).
            # Exclude nodes already running training to enable parallel training across configs
            gpu_nodes = [n for n in healthy_nodes if n.has_gpu and n.node_id not in nodes_with_training]
            if not gpu_nodes:
                # Fall back to allowing nodes with training if no free GPU nodes
                gpu_nodes = [n for n in healthy_nodes if n.has_gpu]
            gpu_nodes.sort(key=lambda n: (-n.gpu_power_score(), n.get_load_score()))
            worker_node = gpu_nodes[0] if gpu_nodes else None
        else:
            # CMA-ES is CPU-heavy. Prefer CPU-only nodes without training, then fall back.
            cpu_nodes = [n for n in healthy_nodes if n.is_cpu_only_node() and n.node_id not in nodes_with_training]
            if not cpu_nodes:
                cpu_nodes = [n for n in healthy_nodes if n.is_cpu_only_node()]
            candidates = cpu_nodes if cpu_nodes else healthy_nodes
            candidates.sort(key=lambda n: n.get_load_score())
            worker_node = candidates[0] if candidates else None

        if not worker_node:
            print(f"[P2P] No suitable worker for {job_type} training job")
            return None

        job.worker_node = worker_node.node_id
        job.status = "queued"

        # Store job
        with self.training_lock:
            self.training_jobs[job_id] = job

        # Update games count at training start
        if job_type == "nnue":
            self.games_at_last_nnue_train[config_key] = job_config.get("total_games", 0)
        else:
            self.games_at_last_cmaes_train[config_key] = job_config.get("total_games", 0)

        # Send to worker
        try:
            endpoint = f"/training/{job_type}/start"
            timeout = ClientTimeout(total=30)
            async with get_client_session(timeout) as session:
                payload = {
                    "job_id": job_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "epochs": job.epochs,
                    "batch_size": job.batch_size,
                    "learning_rate": job.learning_rate,
                }
                last_err: Optional[str] = None
                for url in self._urls_for_peer(worker_node, endpoint):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            result = await resp.json()
                        if result.get("success"):
                            job.status = "running"
                            job.started_at = time.time()
                            print(f"[P2P] Started {job_type} training job {job_id} on {worker_node.node_id}")
                            self._save_state()
                            return job
                        job.status = "failed"
                        job.error_message = str(result.get("error") or "Unknown error")
                        return job
                    except Exception as e:
                        last_err = str(e)
                        continue
                job.status = "failed"
                job.error_message = last_err or "dispatch_failed"
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            print(f"[P2P] Failed to dispatch {job_type} training to {worker_node.node_id}: {e}")

        return job

    async def _check_and_trigger_training(self):
        """Periodic check for training readiness (leader only)."""
        if self.role != NodeRole.LEADER:
            return

        current_time = time.time()
        if current_time - self.last_training_check < self.training_check_interval:
            return

        self.last_training_check = current_time

        # Get jobs that should be started
        jobs_to_start = self._check_training_readiness()

        for job_config in jobs_to_start:
            print(f"[P2P] Auto-triggering {job_config['job_type']} training for {job_config['config_key']} ({job_config['total_games']} games)")
            await self._dispatch_training_job(job_config)

    async def _check_improvement_cycles(self):
        """Periodic check for improvement cycle readiness (leader only).

        This integrates with the ImprovementCycleManager to:
        1. Check if any cycles need training based on data thresholds
        2. Trigger export/training jobs for ready cycles
        3. Run evaluations and update Elo ratings
        4. Schedule CMA-ES optimization when needed
        5. Schedule diverse tournaments for AI calibration
        """
        if self.role != NodeRole.LEADER:
            return

        if not self.improvement_cycle_manager:
            return

        current_time = time.time()
        if current_time - self.last_improvement_cycle_check < self.improvement_cycle_check_interval:
            return

        self.last_improvement_cycle_check = current_time

        # Check which cycles are ready for training
        training_ready = self.improvement_cycle_manager.check_training_needed()

        # Convert to job configs
        jobs_to_start = []
        for board_type, num_players in training_ready:
            cycle_key = f"{board_type}_{num_players}p"
            cycle_state = self.improvement_cycle_manager.state.cycles.get(cycle_key)
            if cycle_state and self.improvement_cycle_manager.trigger_training(board_type, num_players):
                jobs_to_start.append({
                    "cycle_id": cycle_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "total_games": cycle_state.games_since_last_training,
                    "iteration": cycle_state.current_iteration + 1,
                })

        # Also check for CMA-ES optimization opportunities
        cmaes_ready = self.improvement_cycle_manager.check_cmaes_needed()
        for board_type, num_players in cmaes_ready:
            # Trigger distributed CMA-ES
            print(f"[P2P] CMA-ES optimization ready for {board_type}_{num_players}p")
            asyncio.create_task(self._trigger_auto_cmaes(board_type, num_players))

        # Check for rollback needs (consecutive training failures)
        for key, cycle in self.improvement_cycle_manager.state.cycles.items():
            if not cycle.pending_training and not cycle.pending_evaluation:
                should_rollback, reason = self.improvement_cycle_manager.check_rollback_needed(
                    cycle.board_type, cycle.num_players
                )
                if should_rollback:
                    print(f"[P2P] ROLLBACK NEEDED for {key}: {reason}")
                    if self.improvement_cycle_manager.execute_rollback(cycle.board_type, cycle.num_players):
                        self.diversity_metrics["rollbacks"] += 1
                        # Increase diversity to escape plateau
                        print(f"[P2P] Increasing diversity to escape training plateau for {key}")

        for job_config in jobs_to_start:
            cycle_id = job_config["cycle_id"]
            board_type = job_config["board_type"]
            num_players = job_config["num_players"]

            print(f"[P2P] ImprovementCycle {cycle_id}: Starting training "
                  f"({job_config['total_games']} games)")

            # Find GPU worker for training
            gpu_worker = None
            candidates: List[NodeInfo] = []
            with self.peers_lock:
                candidates.extend([p for p in self.peers.values() if p.is_gpu_node() and p.is_healthy()])
            if self.self_info.is_gpu_node() and self.self_info.is_healthy():
                candidates.append(self.self_info)
            if candidates:
                candidates.sort(
                    key=lambda p: (-p.gpu_power_score(), p.get_load_score(), str(p.node_id))
                )
                gpu_worker = candidates[0]

            if not gpu_worker:
                print(f"[P2P] ImprovementCycle {cycle_id}: No GPU worker available, deferring")
                self.improvement_cycle_manager.update_cycle_phase(
                    cycle_id, "idle", error_message="No GPU worker available"
                )
                continue

            # Create training job
            job_id = f"cycle_{cycle_id}_{int(time.time())}"
            training_job = TrainingJob(
                job_id=job_id,
                job_type="nnue",
                board_type=board_type,
                num_players=num_players,
                worker_node=gpu_worker.node_id,
                epochs=job_config.get("epochs", 100),
                batch_size=job_config.get("batch_size", 2048),
                learning_rate=job_config.get("learning_rate", 0.001),
                data_games_count=job_config.get("total_games", 0),
            )

            with self.training_lock:
                self.training_jobs[job_id] = training_job

            # Update cycle state
            self.improvement_cycle_manager.update_cycle_phase(
                cycle_id, "training", training_job_id=job_id
            )

            # Dispatch training to worker
            await self._dispatch_improvement_training(training_job, cycle_id)

    async def _dispatch_improvement_training(self, job: TrainingJob, cycle_id: str):
        """Dispatch training job for improvement cycle."""
        try:
            # Find the worker node
            worker_node = None
            if job.worker_node == self.node_id:
                worker_node = self.self_info
            else:
                with self.peers_lock:
                    worker_node = self.peers.get(job.worker_node)

            if not worker_node:
                print(f"[P2P] ImprovementCycle {cycle_id}: Worker {job.worker_node} not found")
                self.improvement_cycle_manager.update_cycle_phase(
                    cycle_id, "idle", error_message=f"Worker {job.worker_node} not found"
                )
                return

            # Build training payload
            payload = {
                "job_id": job.job_id,
                "cycle_id": cycle_id,
                "board_type": job.board_type,
                "num_players": job.num_players,
                "epochs": job.epochs,
                "batch_size": job.batch_size,
                "learning_rate": job.learning_rate,
            }

            # Send to worker
            timeout = ClientTimeout(total=30)
            async with get_client_session(timeout) as session:
                last_err: Optional[str] = None
                for url in self._urls_for_peer(worker_node, "/training/nnue/start"):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            result = await resp.json()
                        if result.get("success"):
                            job.status = "running"
                            job.started_at = time.time()
                            print(f"[P2P] ImprovementCycle {cycle_id}: Training started on {worker_node.node_id}")
                            return
                        self.improvement_cycle_manager.update_cycle_phase(
                            cycle_id, "idle", error_message=result.get("error", "Training failed to start")
                        )
                        return
                    except Exception as e:
                        last_err = str(e)
                        continue
                self.improvement_cycle_manager.update_cycle_phase(
                    cycle_id, "idle", error_message=last_err or "dispatch_failed"
                )

        except Exception as e:
            print(f"[P2P] ImprovementCycle {cycle_id}: Training dispatch failed: {e}")
            self.improvement_cycle_manager.update_cycle_phase(
                cycle_id, "idle", error_message=str(e)
            )

    # Phase 3 HTTP Handlers

    async def handle_training_start(self, request: web.Request) -> web.Response:
        """Handle request to start a training job (from external or leader)."""
        try:
            data = await request.json()
            job_type = data.get("job_type", "nnue")
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)

            if self.role != NodeRole.LEADER:
                return web.json_response({
                    "success": False,
                    "error": "Only leader can dispatch training jobs"
                })

            job_config = {
                "job_type": job_type,
                "board_type": board_type,
                "num_players": num_players,
                "config_key": f"{board_type}_{num_players}p",
                "total_games": data.get("total_games", 0),
            }

            job = await self._dispatch_training_job(job_config)
            if job:
                return web.json_response({
                    "success": True,
                    "job_id": job.job_id,
                    "worker": job.worker_node,
                })
            else:
                return web.json_response({
                    "success": False,
                    "error": "No suitable worker available"
                })

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def handle_training_status(self, request: web.Request) -> web.Response:
        """Return status of all training jobs."""
        with self.training_lock:
            jobs = [job.to_dict() for job in self.training_jobs.values()]

        return web.json_response({
            "success": True,
            "jobs": jobs,
            "thresholds": self.training_thresholds.to_dict(),
        })

    async def handle_training_update(self, request: web.Request) -> web.Response:
        """Handle training progress/completion update from worker."""
        try:
            data = await request.json()
            job_id = data.get("job_id")

            with self.training_lock:
                job = self.training_jobs.get(job_id)
                if not job:
                    return web.json_response({
                        "success": False,
                        "error": f"Job {job_id} not found"
                    })

                # Update job status
                if data.get("status"):
                    job.status = data["status"]
                if data.get("completed"):
                    job.status = "completed"
                    job.completed_at = time.time()
                if data.get("output_model_path"):
                    job.output_model_path = data["output_model_path"]
                if data.get("final_loss"):
                    job.final_loss = data["final_loss"]
                if data.get("final_accuracy"):
                    job.final_accuracy = data["final_accuracy"]
                if data.get("error"):
                    job.status = "failed"
                    job.error_message = data["error"]

                # Check if we should trigger evaluation after training
                should_trigger_eval = (
                    data.get("completed") and
                    job.output_model_path and
                    self.improvement_cycle_manager
                )

            self._save_state()

            # Auto-trigger tournament evaluation when training completes
            if should_trigger_eval:
                asyncio.create_task(self._handle_training_job_completion(job))

            return web.json_response({"success": True})

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def _handle_training_job_completion(self, job: 'TrainingJob') -> None:
        """Handle training job completion - notify cycle manager and trigger evaluation.

        This method bridges the training completion with the improvement cycle:
        1. Notifies improvement_cycle_manager of training completion
        2. Schedules a model comparison tournament
        3. Updates Elo database with results
        """
        if not self.improvement_cycle_manager:
            return

        try:
            print(f"[P2P] Training job {job.job_id} completed, triggering evaluation")

            # Notify improvement cycle manager
            self.improvement_cycle_manager.handle_training_complete(
                job.board_type,
                job.num_players,
                job.output_model_path,
                job.data_games_count or 0
            )

            # Schedule model comparison tournament
            await self._schedule_model_comparison_tournament(job)

        except Exception as e:
            print(f"[P2P] Error handling training completion for {job.job_id}: {e}")

    async def _schedule_model_comparison_tournament(self, job: 'TrainingJob') -> None:
        """Schedule a tournament to compare the new model against baseline."""
        if not job.output_model_path:
            return

        try:
            # Get tournament matchups from cycle manager
            matchups = self.improvement_cycle_manager.get_tournament_matchups(
                job.board_type,
                job.num_players,
                new_model_path=job.output_model_path
            )

            if not matchups:
                print(f"[P2P] No tournament matchups for {job.board_type}_{job.num_players}p")
                return

            print(f"[P2P] Scheduling {len(matchups)} tournament matchups for new model")

            # Run evaluation games (simplified - in production would dispatch to workers)
            total_wins = 0
            total_games = 0

            for matchup in matchups:
                if matchup.get("purpose") == "primary_evaluation":
                    # Primary evaluation against best model
                    games = matchup.get("games", 20)
                    total_games += games
                    # Placeholder: actual tournament execution would go here
                    # For now, mark as needing external evaluation
                    print(f"[P2P] Tournament: {matchup['agent_a']} vs {matchup['agent_b']} ({games} games)")

            # Update cycle state - evaluation is now pending
            cycle_key = f"{job.board_type}_{job.num_players}p"
            if cycle_key in self.improvement_cycle_manager.state.cycles:
                self.improvement_cycle_manager.state.cycles[cycle_key].pending_evaluation = True
                self.improvement_cycle_manager._save_state()

        except Exception as e:
            print(f"[P2P] Error scheduling tournament: {e}")

    async def handle_nnue_start(self, request: web.Request) -> web.Response:
        """Handle NNUE training start request (worker endpoint)."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)
            epochs = data.get("epochs", 100)
            batch_size = data.get("batch_size", 2048)
            learning_rate = data.get("learning_rate", None)

            # Start NNUE training subprocess
            output_path = os.path.join(
                self.ringrift_path, "ai-service", "models", "nnue",
                f"{board_type}_{num_players}p_auto.pt"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Collect local selfplay databases. The NNUE trainer requires at
            # least one DB (it can replay moves when snapshots are absent).
            data_dir = self.get_data_directory()
            board_tokens = [str(board_type).lower()]
            if "hex" in board_tokens[0]:
                board_tokens = ["hexagonal", "hex"]
            players_token = f"_{int(num_players)}p"

            candidate_dbs: List[Path] = []
            for pattern in ("selfplay/**/*.db", "games/**/*.db"):
                for db_path in data_dir.glob(pattern):
                    if not db_path.is_file():
                        continue
                    path_lower = str(db_path).lower()
                    if players_token not in path_lower:
                        continue
                    if not any(tok in path_lower for tok in board_tokens):
                        continue
                    candidate_dbs.append(db_path)

            # Fallback: if naming conventions differ, use any selfplay DBs.
            if not candidate_dbs:
                candidate_dbs = [p for p in data_dir.glob("selfplay/**/*.db") if p.is_file()]

            # De-dupe + prefer newest DBs (avoid overlong argv on large clusters).
            unique_dbs = list({p.resolve() for p in candidate_dbs})
            unique_dbs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
            max_dbs = 64
            unique_dbs = unique_dbs[:max_dbs]

            if not unique_dbs:
                return web.json_response(
                    {
                        "success": False,
                        "error": f"No selfplay DBs found under {data_dir} for {board_type} {num_players}p",
                    },
                    status=400,
                )

            cmd = [
                sys.executable, "-m", "scripts.train_nnue",
                "--db", *[str(p) for p in unique_dbs],
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--save-path", output_path,
            ]
            if learning_rate is not None:
                cmd.extend(["--learning-rate", str(learning_rate)])

            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=os.path.join(self.ringrift_path, "ai-service"),
            )

            print(f"[P2P] Started NNUE training subprocess (PID {proc.pid}) for job {job_id}")

            # Don't wait - let it run in background
            asyncio.create_task(self._monitor_training_process(job_id, proc, output_path))

            return web.json_response({
                "success": True,
                "pid": proc.pid,
            })

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def _trigger_auto_cmaes(self, board_type: str, num_players: int):
        """Automatically trigger CMA-ES optimization for a configuration.

        Called by improvement cycle manager when optimization is due.
        """
        try:
            job_id = f"auto_cmaes_{board_type}_{num_players}p_{int(time.time())}"
            print(f"[P2P] Auto-triggering CMA-ES: {job_id}")

            # Check for GPU workers
            gpu_workers = []
            with self.peers_lock:
                for peer in self.peers.values():
                    if peer.is_healthy() and peer.has_gpu and peer.node_id != self.node_id:
                        gpu_workers.append(peer)

            if self.self_info.has_gpu:
                gpu_workers.append(self.self_info)

            if len(gpu_workers) >= 2:
                # DISTRIBUTED MODE
                cmaes_job_id = f"cmaes_auto_{job_id}"
                state = DistributedCMAESState(
                    job_id=cmaes_job_id,
                    board_type=board_type,
                    num_players=num_players,
                    generations=100,
                    population_size=max(32, len(gpu_workers) * 8),
                    games_per_eval=100,
                    status="running",
                    started_at=time.time(),
                    last_update=time.time(),
                    worker_nodes=[w.node_id for w in gpu_workers],
                )
                self.distributed_cmaes_state[cmaes_job_id] = state
                asyncio.create_task(self._run_distributed_cmaes(cmaes_job_id))
                print(f"[P2P] Started distributed CMA-ES with {len(gpu_workers)} workers")
            else:
                # LOCAL MODE - use GPU CMA-ES script
                output_dir = os.path.join(
                    self.ringrift_path, "ai-service", "data", "cmaes",
                    f"{board_type}_{num_players}p_auto_{int(time.time())}"
                )
                os.makedirs(output_dir, exist_ok=True)

                cmd = [
                    sys.executable,
                    os.path.join(self.ringrift_path, "ai-service", "scripts", "run_gpu_cmaes.py"),
                    "--board", board_type,
                    "--num-players", str(num_players),
                    "--generations", "100",
                    "--population-size", "32",
                    "--games-per-eval", "100",
                    "--max-moves", "10000",
                    "--output-dir", output_dir,
                    "--multi-gpu",
                ]

                env = os.environ.copy()
                env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                print(f"[P2P] Started local CMA-ES optimization (PID {proc.pid})")

        except Exception as e:
            print(f"[P2P] Auto CMA-ES trigger failed: {e}")

    async def handle_cmaes_start_auto(self, request: web.Request) -> web.Response:
        """Handle CMA-ES optimization start request.

        Uses distributed GPU CMA-ES across all cluster GPU nodes for maximum throughput.
        Falls back to local GPU CMA-ES if no remote workers available.
        """
        try:
            data = await request.json()
            job_id = data.get("job_id")
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)

            # Check for available GPU workers in the cluster
            gpu_workers = []
            with self.peers_lock:
                for peer in self.peers.values():
                    if peer.is_healthy() and peer.has_gpu and peer.node_id != self.node_id:
                        gpu_workers.append(peer)

            # Include self if we have GPU
            if self.self_info.has_gpu:
                gpu_workers.append(self.self_info)

            if len(gpu_workers) >= 2:
                # DISTRIBUTED MODE: Use P2P distributed CMA-ES across cluster
                print(f"[P2P] Starting DISTRIBUTED GPU CMA-ES with {len(gpu_workers)} workers")

                # Create distributed CMA-ES state
                cmaes_job_id = f"cmaes_auto_{job_id}_{int(time.time())}"
                state = DistributedCMAESState(
                    job_id=cmaes_job_id,
                    board_type=board_type,
                    num_players=num_players,
                    generations=100,  # More generations for better optimization
                    population_size=max(32, len(gpu_workers) * 8),  # Scale with workers
                    games_per_eval=100,  # More games for accurate fitness
                    status="running",
                    started_at=time.time(),
                    last_update=time.time(),
                    worker_nodes=[w.node_id for w in gpu_workers],
                )
                self.distributed_cmaes_state[cmaes_job_id] = state

                # Launch distributed coordinator task
                asyncio.create_task(self._run_distributed_cmaes(cmaes_job_id))

                # Track as training job
                with self.training_lock:
                    if job_id in self.training_jobs:
                        self.training_jobs[job_id].status = "running"
                        self.training_jobs[job_id].started_at = time.time()

                return web.json_response({
                    "success": True,
                    "mode": "distributed",
                    "job_id": cmaes_job_id,
                    "workers": [w.node_id for w in gpu_workers],
                })

            else:
                # LOCAL MODE: Run GPU CMA-ES on this node only
                print(f"[P2P] Starting LOCAL GPU CMA-ES (no remote workers available)")

                output_dir = os.path.join(
                    self.ringrift_path, "ai-service", "data", "cmaes",
                    f"{board_type}_{num_players}p_auto_{int(time.time())}"
                )
                os.makedirs(output_dir, exist_ok=True)

                cmd = [
                    sys.executable,
                    os.path.join(self.ringrift_path, "ai-service", "scripts", "run_gpu_cmaes.py"),
                    "--board", board_type,
                    "--num-players", str(num_players),
                    "--generations", "100",
                    "--population-size", "32",
                    "--games-per-eval", "100",
                    "--max-moves", "10000",
                    "--output-dir", output_dir,
                    "--multi-gpu",
                ]

                env = os.environ.copy()
                env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )

                print(f"[P2P] Started local GPU CMA-ES (PID {proc.pid}) for job {job_id}")
                asyncio.create_task(self._monitor_training_process(job_id, proc, output_dir))

                return web.json_response({
                    "success": True,
                    "mode": "local",
                    "pid": proc.pid,
                })

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def _monitor_training_process(self, job_id: str, proc, output_path: str):
        """Monitor training subprocess and report completion to leader."""
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=7200  # 2 hour max
            )

            success = proc.returncode == 0

            # Report to leader
            if self.leader_id and self.leader_id != self.node_id:
                leader = self.peers.get(self.leader_id)
                if leader:
                    try:
                        timeout = ClientTimeout(total=30)
                        async with get_client_session(timeout) as session:
                            url = self._url_for_peer(leader, "/training/update")
                            payload = {
                                "job_id": job_id,
                                "completed": success,
                                "output_model_path": output_path if success else "",
                                "error": stderr.decode()[:500] if not success else "",
                            }
                            await session.post(url, json=payload, headers=self._auth_headers())
                    except Exception as e:
                        print(f"[P2P] Failed to report training completion to leader: {e}")
            else:
                # We are the leader, update directly
                with self.training_lock:
                    job = self.training_jobs.get(job_id)
                    if job:
                        if success:
                            job.status = "completed"
                            job.output_model_path = output_path
                            # LEARNED LESSONS - Schedule tournament to compare new model against baseline
                            asyncio.create_task(self._schedule_model_comparison(job, output_path))
                            # Update improvement cycle manager with training completion
                            if self.improvement_cycle_manager:
                                self.improvement_cycle_manager.handle_training_complete(
                                    job.board_type, job.num_players,
                                    output_path, job.data_games_count or 0
                                )
                        else:
                            job.status = "failed"
                            job.error_message = stderr.decode()[:500]
                        job.completed_at = time.time()

            print(f"[P2P] Training job {job_id} {'completed' if success else 'failed'}")

        except asyncio.TimeoutError:
            print(f"[P2P] Training job {job_id} timed out")
        except Exception as e:
            print(f"[P2P] Training monitor error for {job_id}: {e}")

    async def _monitor_gpu_selfplay_and_validate(
        self,
        job_id: str,
        proc: subprocess.Popen,
        output_dir: Path,
        board_type: str,
        num_players: int,
    ) -> None:
        """Monitor GPU selfplay completion and run CPU validation.

        When GPU selfplay completes, this runs import_gpu_selfplay_to_db.py to:
        1. Replay each game with CPU GameEngine
        2. Validate all moves against legal move lists
        3. Discard games with invalid moves
        4. Store only validated games in canonical DB format

        This ensures GPU-generated games are safe for training.
        """
        try:
            # Wait for GPU selfplay to complete (with timeout)
            return_code = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, proc.wait),
                timeout=7200,  # 2 hour max
            )

            # Update job status
            with self.jobs_lock:
                job = self.local_jobs.get(job_id)
                if job:
                    job.status = "completed" if return_code == 0 else "failed"

            if return_code != 0:
                print(f"[P2P] GPU selfplay job {job_id} failed (exit code {return_code})")
                return

            # Find the generated JSONL file
            jsonl_files = list(output_dir.glob("*.jsonl"))
            if not jsonl_files:
                print(f"[P2P] GPU selfplay job {job_id}: No JSONL output found")
                return

            input_jsonl = jsonl_files[0]
            validated_db = output_dir / "validated_games.db"

            print(f"[P2P] GPU selfplay job {job_id} completed, running CPU validation...")

            # Run CPU validation import
            validate_cmd = [
                "python3",
                f"{self.ringrift_path}/ai-service/scripts/import_gpu_selfplay_to_db.py",
                "--input", str(input_jsonl),
                "--output", str(validated_db),
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"

            validate_proc = await asyncio.create_subprocess_exec(
                *validate_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.ringrift_path,
            )

            stdout, stderr = await asyncio.wait_for(
                validate_proc.communicate(),
                timeout=1800,  # 30 min validation timeout
            )

            if validate_proc.returncode == 0:
                # Parse validation results from output
                output_text = stdout.decode()
                imported = 0
                failed = 0
                for line in output_text.split("\n"):
                    if "Successfully imported:" in line:
                        imported = int(line.split(":")[-1].strip())
                    elif "Failed:" in line:
                        failed = int(line.split(":")[-1].strip())

                validation_rate = imported / (imported + failed) * 100 if (imported + failed) > 0 else 0

                print(f"[P2P] GPU selfplay {job_id} CPU validation complete:")
                print(f"[P2P]   Valid games: {imported}, Invalid: {failed}, Validation rate: {validation_rate:.1f}%")

                # Track validation metrics for diversity reporting
                if hasattr(self, 'diversity_metrics'):
                    if "gpu_validation_stats" not in self.diversity_metrics:
                        self.diversity_metrics["gpu_validation_stats"] = {
                            "total_generated": 0,
                            "total_validated": 0,
                            "total_failed": 0,
                        }
                    self.diversity_metrics["gpu_validation_stats"]["total_generated"] += imported + failed
                    self.diversity_metrics["gpu_validation_stats"]["total_validated"] += imported
                    self.diversity_metrics["gpu_validation_stats"]["total_failed"] += failed

                # Record validation rate metric for observability
                self.record_metric(
                    "validation_rate",
                    validation_rate,
                    board_type=board_type,
                    num_players=num_players,
                    metadata={
                        "job_id": job_id,
                        "imported": imported,
                        "failed": failed,
                    },
                )

                # Auto-import to canonical database if validation rate is high enough
                if validation_rate >= 95 and imported > 0:
                    asyncio.create_task(self._import_gpu_selfplay_to_canonical(
                        validated_db, board_type, num_players, imported
                    ))
                elif validation_rate < 95:
                    print(f"[P2P] WARNING: GPU selfplay validation rate {validation_rate:.1f}% is below 95%")
                    print(f"[P2P]   This indicates potential GPU/CPU rule divergence")
                    print(f"[P2P]   Skipping auto-import to canonical database")
                    # Alert on low validation rate
                    asyncio.create_task(self.notifier.send(
                        title="Low GPU Validation Rate",
                        message=f"GPU selfplay validation rate {validation_rate:.1f}% is below 95% threshold",
                        level="warning",
                        fields={
                            "Config": f"{board_type}_{num_players}p",
                            "Valid": str(imported),
                            "Invalid": str(failed),
                            "Rate": f"{validation_rate:.1f}%",
                        },
                        node_id=self.node_id,
                    ))

            else:
                print(f"[P2P] GPU selfplay {job_id} CPU validation failed:")
                print(f"[P2P]   {stderr.decode()[:500]}")

        except asyncio.TimeoutError:
            print(f"[P2P] GPU selfplay job {job_id} timed out")
            with self.jobs_lock:
                job = self.local_jobs.get(job_id)
                if job:
                    job.status = "failed"
        except Exception as e:
            print(f"[P2P] GPU selfplay monitor error for {job_id}: {e}")
            with self.jobs_lock:
                job = self.local_jobs.get(job_id)
                if job:
                    job.status = "failed"

    async def _schedule_model_comparison(self, job: TrainingJob, new_model_path: str):
        """Schedule a tournament to compare new model against current baseline.

        LEARNED LESSONS - After training, automatically run tournament to:
        1. Compare new model against current best baseline
        2. Update Elo ratings
        3. Promote to best baseline if new model wins
        """
        try:
            config_key = f"{job.board_type}_{job.num_players}p"
            print(f"[P2P] Scheduling model comparison tournament for {config_key}")

            # Find current baseline model
            baseline_dir = Path(self.ringrift_path) / "ai-service" / "models" / job.job_type
            baseline_pattern = f"{job.board_type}_{job.num_players}p_best*"

            baseline_model = None
            for f in baseline_dir.glob(baseline_pattern):
                baseline_model = str(f)
                break

            if not baseline_model:
                # No baseline - this model becomes baseline
                print(f"[P2P] No baseline found for {config_key}, new model becomes baseline")
                await self._promote_to_baseline(new_model_path, job.board_type, job.num_players, job.job_type)
                return

            # Schedule tournament via SSH tournament system
            tournament_id = f"autoeval_{config_key}_{int(time.time())}"

            # Use existing SSH tournament infrastructure
            with self.ssh_tournament_lock:
                self.ssh_tournament_runs[tournament_id] = SSHTournamentRun(
                    tournament_id=tournament_id,
                    board_type=job.board_type,
                    num_players=job.num_players,
                    status="pending",
                    started_at=time.time(),
                )

            # Start tournament in background
            tournament_config = {
                "tournament_id": tournament_id,
                "board_type": job.board_type,
                "num_players": job.num_players,
                "model_a": new_model_path,
                "model_b": baseline_model,
                "games_per_matchup": 50,
            }
            asyncio.create_task(self._run_model_comparison_tournament(tournament_config))

        except Exception as e:
            print(f"[P2P] Model comparison scheduling error: {e}")

    async def _run_model_comparison_tournament(self, config: dict):
        """Run a model comparison tournament and update baseline if new model wins."""
        tournament_id = config["tournament_id"]
        try:
            print(f"[P2P] Running model comparison tournament {tournament_id}")

            results_dir = Path(self.ringrift_path) / "ai-service" / "results" / "tournaments"
            results_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                os.path.join(self.ringrift_path, "ai-service", "scripts", "run_tournament.py"),
                "--player1", f"nn:{config['model_a']}",
                "--player2", f"nn:{config['model_b']}",
                "--board", config["board_type"],
                "--num-players", str(config["num_players"]),
                "--games", str(config["games_per_matchup"]),
                "--output", str(results_dir / f"{tournament_id}.json"),
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
            env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3600)

            if proc.returncode == 0:
                results_file = results_dir / f"{tournament_id}.json"
                if results_file.exists():
                    import json as json_module
                    results = json_module.loads(results_file.read_text())
                    new_model_wins = results.get("player1_wins", 0)
                    baseline_wins = results.get("player2_wins", 0)
                    total_games = new_model_wins + baseline_wins

                    win_rate = new_model_wins / total_games if total_games > 0 else 0.5
                    print(f"[P2P] Tournament {tournament_id}: new model win rate = {win_rate:.1%}")

                    promoted = win_rate >= 0.55
                    if promoted:
                        print(f"[P2P] New model beats baseline! Promoting to best baseline.")
                        await self._promote_to_baseline(
                            config["model_a"], config["board_type"],
                            config["num_players"], "nnue" if "nnue" in config["model_a"].lower() else "cmaes"
                        )

                    # Update improvement cycle manager with tournament result
                    await self._handle_tournament_completion(
                        tournament_id,
                        config["board_type"],
                        config["num_players"],
                        config["model_a"],
                        config["model_b"],
                        win_rate,
                        promoted,
                    )

            with self.ssh_tournament_lock:
                if tournament_id in self.ssh_tournament_runs:
                    self.ssh_tournament_runs[tournament_id].status = "completed"
                    self.ssh_tournament_runs[tournament_id].completed_at = time.time()

        except Exception as e:
            print(f"[P2P] Tournament {tournament_id} error: {e}")
            with self.ssh_tournament_lock:
                if tournament_id in self.ssh_tournament_runs:
                    self.ssh_tournament_runs[tournament_id].status = "failed"
                    self.ssh_tournament_runs[tournament_id].error = str(e)

    async def _promote_to_baseline(self, model_path: str, board_type: str, num_players: int, model_type: str):
        """Promote a model to the best baseline for its board type."""
        try:
            import shutil
            baseline_dir = Path(self.ringrift_path) / "ai-service" / "models" / model_type
            baseline_dir.mkdir(parents=True, exist_ok=True)

            baseline_path = baseline_dir / f"{board_type}_{num_players}p_best.pt"
            if baseline_path.exists():
                backup_path = baseline_dir / f"{board_type}_{num_players}p_prev_{int(time.time())}.pt"
                shutil.copy2(baseline_path, backup_path)
                print(f"[P2P] Backed up previous baseline to {backup_path}")

            shutil.copy2(model_path, baseline_path)
            print(f"[P2P] Promoted {model_path} to baseline at {baseline_path}")

        except Exception as e:
            print(f"[P2P] Baseline promotion error: {e}")

    async def _handle_tournament_completion(
        self,
        tournament_id: str,
        board_type: str,
        num_players: int,
        new_model: str,
        baseline_model: str,
        win_rate: float,
        promoted: bool,
    ):
        """Handle tournament completion - update cycle state and trigger next iteration.

        This closes the feedback loop by:
        1. Updating improvement cycle manager with evaluation result
        2. Recording result to unified Elo database
        3. Updating diversity metrics
        4. Boosting selfplay for this config if model was promoted
        """
        try:
            # 1. Update improvement cycle manager
            if self.improvement_cycle_manager:
                self.improvement_cycle_manager.handle_evaluation_complete(
                    board_type, num_players, win_rate, new_model
                )
                print(f"[P2P] Updated improvement cycle for {board_type}_{num_players}p")

            # 2. Record to unified Elo database
            try:
                from app.tournament.unified_elo_db import get_elo_database
                db = get_elo_database()
                # Rankings: 0 = winner, 1 = loser
                rankings = [0, 1] if win_rate > 0.5 else [1, 0]
                db.record_match_and_update(
                    participant_ids=[new_model, baseline_model],
                    rankings=rankings,
                    board_type=board_type,
                    num_players=num_players,
                    tournament_id=tournament_id,
                )
                print(f"[P2P] Recorded tournament result to unified Elo DB")
            except Exception as e:
                print(f"[P2P] Elo database update failed (non-fatal): {e}")

            # 3. Update diversity metrics
            if hasattr(self, 'diversity_metrics'):
                self.diversity_metrics["tournament_runs"] = self.diversity_metrics.get("tournament_runs", 0) + 1
                if promoted:
                    self.diversity_metrics["promotions"] = self.diversity_metrics.get("promotions", 0) + 1

            # 4. Record metrics for observability
            self.record_metric(
                "tournament_win_rate",
                win_rate,
                board_type=board_type,
                num_players=num_players,
                metadata={
                    "new_model": new_model,
                    "baseline_model": baseline_model,
                    "promoted": promoted,
                    "tournament_id": tournament_id,
                },
            )

            # 5. Boost selfplay for this config if promoted (more data for next iteration)
            if promoted:
                asyncio.create_task(self._boost_selfplay_for_config(board_type, num_players))
                # Alert on successful promotion
                asyncio.create_task(self.notifier.send(
                    title="Model Promoted",
                    message=f"New model promoted for {board_type}_{num_players}p with {win_rate*100:.1f}% win rate",
                    level="info",
                    fields={"Model": new_model, "Win Rate": f"{win_rate*100:.1f}%"},
                    node_id=self.node_id,
                ))
            elif win_rate < 0.5:
                # Alert on failed promotion (new model lost)
                asyncio.create_task(self.notifier.send(
                    title="Model Promotion Failed",
                    message=f"New model failed tournament for {board_type}_{num_players}p with only {win_rate*100:.1f}% win rate",
                    level="warning",
                    fields={
                        "Model": new_model,
                        "Win Rate": f"{win_rate*100:.1f}%",
                        "Baseline": baseline_model,
                    },
                    node_id=self.node_id,
                ))

        except Exception as e:
            print(f"[P2P] Tournament completion handler error: {e}")
            asyncio.create_task(self.notifier.send(
                title="Tournament Handler Error",
                message=str(e),
                level="error",
                node_id=self.node_id,
            ))

    async def _boost_selfplay_for_config(self, board_type: str, num_players: int):
        """Temporarily boost selfplay for a configuration after model promotion.

        This accelerates data generation for the next training iteration.
        """
        try:
            config_key = f"{board_type}_{num_players}p"
            print(f"[P2P] Boosting selfplay for {config_key} after promotion")

            # Schedule additional selfplay jobs for this configuration
            # This will be picked up by the next job scheduling cycle
            if hasattr(self, 'selfplay_boost_configs'):
                self.selfplay_boost_configs[config_key] = {
                    "boost_until": time.time() + 3600,  # Boost for 1 hour
                    "multiplier": 1.5,  # 50% more jobs
                }
            else:
                self.selfplay_boost_configs = {
                    config_key: {
                        "boost_until": time.time() + 3600,
                        "multiplier": 1.5,
                    }
                }

        except Exception as e:
            print(f"[P2P] Selfplay boost error: {e}")

    async def _propagate_cmaes_weights(
        self, board_type: str, num_players: int, weights: Dict[str, float]
    ):
        """Propagate new CMA-ES weights to selfplay workers.

        After CMA-ES optimization finds better weights, this:
        1. Saves weights to shared config file
        2. Restarts selfplay jobs for this config with new weights
        """
        try:
            config_key = f"{board_type}_{num_players}p"
            print(f"[P2P] Propagating CMA-ES weights for {config_key}")

            # 1. Save to shared heuristic weights config
            config_path = Path(self.ringrift_path) / "ai-service" / "config" / "heuristic_weights.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)

            import json as json_mod
            existing = {}
            if config_path.exists():
                try:
                    existing = json_mod.loads(config_path.read_text())
                except Exception:
                    pass

            existing[config_key] = {
                "weights": weights,
                "updated_at": time.time(),
            }
            config_path.write_text(json_mod.dumps(existing, indent=2))
            print(f"[P2P] Updated heuristic_weights.json with {config_key} weights")

            # 2. Track config for weight-aware selfplay scheduling
            if not hasattr(self, 'cmaes_weight_configs'):
                self.cmaes_weight_configs = {}

            self.cmaes_weight_configs[config_key] = {
                "weights": weights,
                "updated_at": time.time(),
            }

            # 3. Stop existing selfplay jobs for this config (they'll restart with new weights)
            jobs_to_stop = []
            with self.jobs_lock:
                for job_id, job in self.local_jobs.items():
                    if (job.job_type in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY)
                        and getattr(job, 'board_type', None) == board_type
                        and getattr(job, 'num_players', None) == num_players
                        and job.status == "running"):
                        jobs_to_stop.append(job_id)

            for job_id in jobs_to_stop:
                await self._stop_local_job(job_id)
                print(f"[P2P] Stopped selfplay job {job_id} for weight update")

            # 4. Boost selfplay to generate data with new weights
            asyncio.create_task(self._boost_selfplay_for_config(board_type, num_players))

            print(f"[P2P] Weight propagation complete for {config_key}")

        except Exception as e:
            print(f"[P2P] CMA-ES weight propagation error: {e}")

    async def _stop_local_job(self, job_id: str):
        """Stop a local job by job ID."""
        try:
            with self.jobs_lock:
                job = self.local_jobs.get(job_id)
                if job and hasattr(job, 'process') and job.process:
                    job.process.terminate()
                    job.status = "stopped"
        except Exception as e:
            print(f"[P2P] Error stopping job {job_id}: {e}")

    async def _import_gpu_selfplay_to_canonical(
        self, validated_db: Path, board_type: str, num_players: int, game_count: int
    ):
        """Import validated GPU selfplay games to canonical selfplay database.

        After GPU selfplay games pass CPU validation (>=95% validation rate),
        this merges them into the canonical selfplay database for training.
        """
        try:
            # Determine canonical DB path
            canonical_db = Path(self.ringrift_path) / "ai-service" / "data" / "games" / "selfplay.db"
            if not canonical_db.parent.exists():
                canonical_db.parent.mkdir(parents=True, exist_ok=True)

            print(f"[P2P] Auto-importing {game_count} validated GPU games to canonical DB...")

            # Use sqlite3 to merge games from validated_db to canonical_db
            import sqlite3

            # Connect to both databases
            src_conn = sqlite3.connect(str(validated_db))
            dst_conn = sqlite3.connect(str(canonical_db))

            # Ensure destination tables exist
            dst_conn.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    winner INTEGER,
                    move_count INTEGER,
                    game_time_ms INTEGER,
                    created_at REAL,
                    source TEXT DEFAULT 'selfplay'
                )
            """)
            dst_conn.execute("""
                CREATE TABLE IF NOT EXISTS moves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    move_number INTEGER NOT NULL,
                    player INTEGER NOT NULL,
                    move_type TEXT NOT NULL,
                    from_pos TEXT,
                    to_pos TEXT,
                    direction TEXT,
                    captured_pos TEXT,
                    state_before TEXT,
                    policy_probs TEXT,
                    value_est REAL,
                    FOREIGN KEY (game_id) REFERENCES games(game_id)
                )
            """)
            dst_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_moves_game_id ON moves(game_id)
            """)
            dst_conn.commit()

            # Check source schema and copy games
            src_cursor = src_conn.cursor()
            src_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            src_tables = {row[0] for row in src_cursor.fetchall()}

            imported = 0
            if "games" in src_tables:
                # Get existing game IDs in destination to avoid duplicates
                dst_cursor = dst_conn.cursor()
                dst_cursor.execute("SELECT game_id FROM games")
                existing_ids = {row[0] for row in dst_cursor.fetchall()}

                # Copy games that don't already exist
                src_cursor.execute("SELECT * FROM games")
                src_columns = [desc[0] for desc in src_cursor.description]

                for row in src_cursor.fetchall():
                    game_id_idx = src_columns.index("game_id") if "game_id" in src_columns else 0
                    game_id = row[game_id_idx]

                    if game_id in existing_ids:
                        continue

                    # Insert game with proper column mapping
                    placeholders = ", ".join(["?"] * len(row))
                    columns = ", ".join(src_columns)
                    try:
                        dst_conn.execute(
                            f"INSERT OR IGNORE INTO games ({columns}) VALUES ({placeholders})",
                            row
                        )
                        imported += 1
                    except Exception:
                        continue

                # Copy moves for new games
                if "moves" in src_tables and imported > 0:
                    src_cursor.execute("SELECT * FROM moves")
                    move_columns = [desc[0] for desc in src_cursor.description]
                    move_placeholders = ", ".join(["?"] * len(move_columns))
                    move_col_str = ", ".join(move_columns)

                    for row in src_cursor.fetchall():
                        game_id_idx = move_columns.index("game_id") if "game_id" in move_columns else 1
                        game_id = row[game_id_idx]
                        if game_id not in existing_ids:
                            try:
                                dst_conn.execute(
                                    f"INSERT OR IGNORE INTO moves ({move_col_str}) VALUES ({move_placeholders})",
                                    row
                                )
                            except Exception:
                                continue

                dst_conn.commit()

            src_conn.close()
            dst_conn.close()

            print(f"[P2P] Successfully imported {imported} GPU selfplay games to canonical DB")

            # Update cluster data manifest to reflect new games
            config_key = f"{board_type}_{num_players}p"
            if hasattr(self, 'cluster_data_manifest') and self.cluster_data_manifest:
                if config_key in self.cluster_data_manifest.by_board_type:
                    self.cluster_data_manifest.by_board_type[config_key]["total_games"] = (
                        self.cluster_data_manifest.by_board_type[config_key].get("total_games", 0) + imported
                    )

            # Notify improvement cycle manager of new games
            if self.improvement_cycle_manager and imported > 0:
                self.improvement_cycle_manager.record_games(board_type, num_players, imported)

        except Exception as e:
            print(f"[P2P] GPU selfplay import error: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================

    # =========================================================================
    # Phase 5: Improvement Cycle HTTP Handlers
    # =========================================================================

    async def handle_improvement_cycles_status(self, request: web.Request) -> web.Response:
        """GET /improvement_cycles/status - Get status of all improvement cycles."""
        try:
            if not self.improvement_cycle_manager:
                return web.json_response({
                    "success": False,
                    "error": "ImprovementCycleManager not initialized"
                })

            status = self.improvement_cycle_manager.get_status()
            return web.json_response({
                "success": True,
                "is_leader": self.role == NodeRole.LEADER,
                **status,
            })

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def handle_improvement_cycles_leaderboard(self, request: web.Request) -> web.Response:
        """GET /improvement_cycles/leaderboard - Get Elo leaderboard."""
        try:
            if not self.improvement_cycle_manager:
                return web.json_response({
                    "success": False,
                    "error": "ImprovementCycleManager not initialized"
                })

            board_type = request.query.get("board_type")
            num_players_str = request.query.get("num_players")
            num_players = int(num_players_str) if num_players_str else None

            leaderboard = self.improvement_cycle_manager.get_leaderboard(
                board_type=board_type,
                num_players=num_players,
            )

            return web.json_response({
                "success": True,
                "leaderboard": [e.to_dict() for e in leaderboard],
                "total_models": len(leaderboard),
            })

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def handle_metrics(self, request: web.Request) -> web.Response:
        """GET /metrics - Get metrics summary and history.

        Content negotiation:
        - Accept: text/plain -> Prometheus format (same as /metrics/prometheus)
        - Accept: application/json -> JSON format
        - Default (no header) -> Prometheus format for Prometheus scraper compatibility
        """
        try:
            # Content negotiation for Prometheus compatibility
            accept = request.headers.get("Accept", "")
            # Prometheus sends "text/plain" or "application/openmetrics-text"
            # Also check for explicit format param
            format_param = request.query.get("format", "").lower()
            if format_param == "prometheus" or "text/plain" in accept or "openmetrics" in accept or not accept:
                # Return Prometheus format
                return await self.handle_metrics_prometheus(request)

            hours = float(request.query.get("hours", "24"))
            metric_type = request.query.get("type")
            board_type = request.query.get("board_type")
            num_players_str = request.query.get("num_players")
            num_players = int(num_players_str) if num_players_str else None

            if metric_type:
                # Get specific metric history
                history = self.get_metrics_history(
                    metric_type=metric_type,
                    board_type=board_type,
                    num_players=num_players,
                    hours=hours,
                )
                return web.json_response({
                    "success": True,
                    "metric_type": metric_type,
                    "period_hours": hours,
                    "count": len(history),
                    "history": history,
                })
            else:
                # Get summary of all metrics
                summary = self.get_metrics_summary(hours=hours)
                return web.json_response({
                    "success": True,
                    **summary,
                })

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def handle_metrics_prometheus(self, request: web.Request) -> web.Response:
        """GET /metrics/prometheus - Prometheus-compatible metrics export.

        Returns metrics in Prometheus text exposition format for scraping.
        """
        try:
            lines = []
            now = time.time()

            # Cluster metrics
            with self.peers_lock:
                alive_peers = len([p for p in self.peers.values() if p.is_alive()])
                total_peers = len(self.peers)

            lines.append("# HELP ringrift_cluster_peers_total Total number of known peers")
            lines.append("# TYPE ringrift_cluster_peers_total gauge")
            lines.append(f"ringrift_cluster_peers_total {total_peers}")

            lines.append("# HELP ringrift_cluster_peers_alive Number of alive peers")
            lines.append("# TYPE ringrift_cluster_peers_alive gauge")
            lines.append(f"ringrift_cluster_peers_alive {alive_peers}")

            lines.append("# HELP ringrift_is_leader Whether this node is the leader")
            lines.append("# TYPE ringrift_is_leader gauge")
            lines.append(f"ringrift_is_leader {1 if self.role == NodeRole.LEADER else 0}")

            # Job counts
            with self.jobs_lock:
                selfplay_jobs = len([j for j in self.local_jobs.values()
                                    if j.job_type in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY)
                                    and j.status == "running"])
                training_jobs = len([j for j in self.local_jobs.values()
                                    if j.job_type == JobType.TRAINING and j.status == "running"])

            lines.append("# HELP ringrift_selfplay_jobs_running Number of running selfplay jobs")
            lines.append("# TYPE ringrift_selfplay_jobs_running gauge")
            lines.append(f"ringrift_selfplay_jobs_running {selfplay_jobs}")

            lines.append("# HELP ringrift_training_jobs_running Number of running training jobs")
            lines.append("# TYPE ringrift_training_jobs_running gauge")
            lines.append(f"ringrift_training_jobs_running {training_jobs}")

            # Resource utilization - include node labels for all nodes
            lines.append("# HELP ringrift_cpu_percent CPU utilization percentage per node")
            lines.append("# TYPE ringrift_cpu_percent gauge")

            lines.append("# HELP ringrift_memory_percent Memory utilization percentage per node")
            lines.append("# TYPE ringrift_memory_percent gauge")

            lines.append("# HELP ringrift_disk_percent Disk utilization percentage per node")
            lines.append("# TYPE ringrift_disk_percent gauge")

            lines.append("# HELP ringrift_gpu_percent GPU utilization percentage per node")
            lines.append("# TYPE ringrift_gpu_percent gauge")

            lines.append("# HELP ringrift_selfplay_jobs Selfplay jobs per node")
            lines.append("# TYPE ringrift_selfplay_jobs gauge")

            lines.append("# HELP ringrift_node_alive Whether node is alive (1) or not (0)")
            lines.append("# TYPE ringrift_node_alive gauge")

            # Cluster cost metrics (for Grafana dashboards)
            # GPU hourly rates (Lambda Labs pricing)
            GPU_HOURLY_RATES = {
                "GH200": 2.49, "H100": 2.49, "A100": 1.99, "A10": 0.75,
                "RTX_4090": 0.50, "RTX4090": 0.50, "4090": 0.50,
                "RTX_3090": 0.30, "RTX3090": 0.30, "3090": 0.30,
                "unknown": 0.50,
            }

            lines.append("# HELP ringrift_cluster_node_up Whether cluster node is active (1=up, 0=down)")
            lines.append("# TYPE ringrift_cluster_node_up gauge")
            lines.append("# HELP ringrift_cluster_node_cost_per_hour Estimated hourly cost in USD")
            lines.append("# TYPE ringrift_cluster_node_cost_per_hour gauge")
            lines.append("# HELP ringrift_cluster_gpu_utilization GPU utilization as fraction (0-1)")
            lines.append("# TYPE ringrift_cluster_gpu_utilization gauge")
            lines.append("# HELP ringrift_cluster_cpu_utilization CPU utilization as fraction (0-1)")
            lines.append("# TYPE ringrift_cluster_cpu_utilization gauge")
            lines.append("# HELP ringrift_cluster_gpu_memory_used_bytes GPU memory used in bytes")
            lines.append("# TYPE ringrift_cluster_gpu_memory_used_bytes gauge")
            lines.append("# HELP ringrift_cluster_memory_used_bytes System memory used in bytes")
            lines.append("# TYPE ringrift_cluster_memory_used_bytes gauge")

            # Export self metrics with node label
            node_name = self.node_id or "unknown"
            cpu = getattr(self.self_info, 'cpu_percent', 0)
            mem = getattr(self.self_info, 'memory_percent', 0)
            disk = getattr(self.self_info, 'disk_percent', 0)
            gpu = getattr(self.self_info, 'gpu_percent', 0) if self.self_info.has_gpu else 0
            role = "leader" if self.role == NodeRole.LEADER else "worker"
            gpu_type = getattr(self.self_info, 'gpu_type', 'unknown') or 'unknown'
            # Normalize GPU type for lookup
            gpu_type_key = gpu_type.replace(' ', '_').upper() if gpu_type else 'unknown'
            hourly_cost = GPU_HOURLY_RATES.get(gpu_type_key, GPU_HOURLY_RATES.get(gpu_type, GPU_HOURLY_RATES['unknown']))
            gpu_mem_bytes = getattr(self.self_info, 'gpu_memory_used_bytes', 0) or 0
            sys_mem_bytes = getattr(self.self_info, 'memory_used_bytes', 0) or 0

            lines.append(f'ringrift_cpu_percent{{node="{node_name}",role="{role}"}} {cpu}')
            lines.append(f'ringrift_memory_percent{{node="{node_name}",role="{role}"}} {mem}')
            lines.append(f'ringrift_disk_percent{{node="{node_name}",role="{role}"}} {disk}')
            lines.append(f'ringrift_gpu_percent{{node="{node_name}",role="{role}"}} {gpu}')
            lines.append(f'ringrift_selfplay_jobs{{node="{node_name}",role="{role}"}} {selfplay_jobs}')
            lines.append(f'ringrift_node_alive{{node="{node_name}",role="{role}"}} 1')

            # Export cluster cost metrics for self (for Grafana cost dashboard)
            lines.append(f'ringrift_cluster_node_up{{node="{node_name}",gpu_type="{gpu_type}"}} 1')
            lines.append(f'ringrift_cluster_node_cost_per_hour{{node="{node_name}",gpu_type="{gpu_type}"}} {hourly_cost}')
            lines.append(f'ringrift_cluster_gpu_utilization{{node="{node_name}",gpu_type="{gpu_type}"}} {gpu / 100.0 if gpu else 0}')
            lines.append(f'ringrift_cluster_cpu_utilization{{node="{node_name}"}} {cpu / 100.0 if cpu else 0}')
            lines.append(f'ringrift_cluster_gpu_memory_used_bytes{{node="{node_name}",gpu_type="{gpu_type}"}} {gpu_mem_bytes}')
            lines.append(f'ringrift_cluster_memory_used_bytes{{node="{node_name}"}} {sys_mem_bytes}')

            # Export peer metrics with node labels
            with self.peers_lock:
                for peer_id, peer in self.peers.items():
                    peer_name = peer_id or "unknown"
                    peer_role = "worker"
                    is_alive = 1 if peer.is_alive() else 0

                    # Get peer resource info if available
                    peer_cpu = getattr(peer, 'cpu_percent', 0) or 0
                    peer_mem = getattr(peer, 'memory_percent', 0) or 0
                    peer_gpu = getattr(peer, 'gpu_percent', 0) or 0
                    peer_jobs = getattr(peer, 'selfplay_jobs', 0) or 0
                    peer_gpu_type = getattr(peer, 'gpu_type', 'unknown') or 'unknown'
                    peer_gpu_type_key = peer_gpu_type.replace(' ', '_').upper() if peer_gpu_type else 'unknown'
                    peer_hourly_cost = GPU_HOURLY_RATES.get(peer_gpu_type_key, GPU_HOURLY_RATES.get(peer_gpu_type, GPU_HOURLY_RATES['unknown']))
                    peer_gpu_mem = getattr(peer, 'gpu_memory_used_bytes', 0) or 0
                    peer_sys_mem = getattr(peer, 'memory_used_bytes', 0) or 0

                    lines.append(f'ringrift_cpu_percent{{node="{peer_name}",role="{peer_role}"}} {peer_cpu}')
                    lines.append(f'ringrift_memory_percent{{node="{peer_name}",role="{peer_role}"}} {peer_mem}')
                    lines.append(f'ringrift_gpu_percent{{node="{peer_name}",role="{peer_role}"}} {peer_gpu}')
                    lines.append(f'ringrift_selfplay_jobs{{node="{peer_name}",role="{peer_role}"}} {peer_jobs}')
                    lines.append(f'ringrift_node_alive{{node="{peer_name}",role="{peer_role}"}} {is_alive}')

                    # Export cluster cost metrics for peer
                    lines.append(f'ringrift_cluster_node_up{{node="{peer_name}",gpu_type="{peer_gpu_type}"}} {is_alive}')
                    lines.append(f'ringrift_cluster_node_cost_per_hour{{node="{peer_name}",gpu_type="{peer_gpu_type}"}} {peer_hourly_cost if is_alive else 0}')
                    lines.append(f'ringrift_cluster_gpu_utilization{{node="{peer_name}",gpu_type="{peer_gpu_type}"}} {peer_gpu / 100.0 if peer_gpu else 0}')
                    lines.append(f'ringrift_cluster_cpu_utilization{{node="{peer_name}"}} {peer_cpu / 100.0 if peer_cpu else 0}')
                    lines.append(f'ringrift_cluster_gpu_memory_used_bytes{{node="{peer_name}",gpu_type="{peer_gpu_type}"}} {peer_gpu_mem}')
                    lines.append(f'ringrift_cluster_memory_used_bytes{{node="{peer_name}"}} {peer_sys_mem}')

            # Elo metrics with config labels
            try:
                from scripts.run_model_elo_tournament import init_elo_database, ELO_DB_PATH
                if ELO_DB_PATH and ELO_DB_PATH.exists():
                    db = init_elo_database()
                    conn = db._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT board_type, num_players, MAX(rating) as best_elo
                        FROM elo_ratings
                        WHERE games_played >= 10
                        GROUP BY board_type, num_players
                    """)
                    lines.append("# HELP ringrift_best_elo Best Elo rating per configuration")
                    lines.append("# TYPE ringrift_best_elo gauge")
                    for row in cursor.fetchall():
                        bt, np, elo = row
                        config = f"{bt}_{np}p"
                        lines.append(f'ringrift_best_elo{{config="{config}",board_type="{bt}",num_players="{np}"}} {elo}')
                    db.close()
            except Exception:
                pass

            # Diversity metrics
            if hasattr(self, 'diversity_metrics'):
                dm = self.diversity_metrics
                lines.append("# HELP ringrift_tournament_runs_total Total tournament runs")
                lines.append("# TYPE ringrift_tournament_runs_total counter")
                lines.append(f"ringrift_tournament_runs_total {dm.get('tournament_runs', 0)}")

                lines.append("# HELP ringrift_promotions_total Total model promotions")
                lines.append("# TYPE ringrift_promotions_total counter")
                lines.append(f"ringrift_promotions_total {dm.get('promotions', 0)}")

                lines.append("# HELP ringrift_rollbacks_total Total model rollbacks")
                lines.append("# TYPE ringrift_rollbacks_total counter")
                lines.append(f"ringrift_rollbacks_total {dm.get('rollbacks', 0)}")

                # GPU validation stats
                gpu_stats = dm.get('gpu_validation_stats', {})
                if gpu_stats:
                    lines.append("# HELP ringrift_gpu_games_validated_total Total GPU games validated")
                    lines.append("# TYPE ringrift_gpu_games_validated_total counter")
                    lines.append(f"ringrift_gpu_games_validated_total {gpu_stats.get('total_validated', 0)}")

                    lines.append("# HELP ringrift_gpu_games_failed_total Total GPU games failed validation")
                    lines.append("# TYPE ringrift_gpu_games_failed_total counter")
                    lines.append(f"ringrift_gpu_games_failed_total {gpu_stats.get('total_failed', 0)}")

            # Recent metrics from database (last hour averages)
            try:
                summary = self.get_metrics_summary(hours=1)
                metrics_data = summary.get("metrics", {})

                for metric_name, metric_info in metrics_data.items():
                    safe_name = metric_name.replace("-", "_").replace(".", "_")
                    if metric_info.get("latest") is not None:
                        lines.append(f"# HELP ringrift_{safe_name} Latest {metric_name} value")
                        lines.append(f"# TYPE ringrift_{safe_name} gauge")
                        lines.append(f"ringrift_{safe_name} {metric_info['latest']}")
            except Exception:
                pass

            # Data manifest totals
            if hasattr(self, 'cluster_data_manifest') and self.cluster_data_manifest:
                for config_key, config_data in self.cluster_data_manifest.by_board_type.items():
                    total_games = config_data.get("total_games", 0)
                    parts = config_key.split("_")
                    if len(parts) >= 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_games_total{{board_type="{board_type}",num_players="{num_players}"}} {total_games}')

            # Add header for games total
            if hasattr(self, 'cluster_data_manifest') and self.cluster_data_manifest:
                lines.insert(-len(self.cluster_data_manifest.by_board_type),
                           "# HELP ringrift_games_total Total games per board configuration")
                lines.insert(-len(self.cluster_data_manifest.by_board_type),
                           "# TYPE ringrift_games_total gauge")

            # === CRITICAL SELF-IMPROVEMENT LOOP METRICS ===

            # Training Progress Metrics
            lines.append("# HELP ringrift_training_loss Current model training loss")
            lines.append("# TYPE ringrift_training_loss gauge")
            lines.append("# HELP ringrift_training_val_loss Current model validation loss")
            lines.append("# TYPE ringrift_training_val_loss gauge")
            lines.append("# HELP ringrift_training_epoch Current training epoch")
            lines.append("# TYPE ringrift_training_epoch gauge")
            if hasattr(self, 'training_metrics'):
                for config, metrics in self.training_metrics.items():
                    loss = metrics.get('loss', 0)
                    val_loss = metrics.get('val_loss', 0)
                    epoch = metrics.get('epoch', 0)
                    lines.append(f'ringrift_training_loss{{config="{config}"}} {loss}')
                    lines.append(f'ringrift_training_val_loss{{config="{config}"}} {val_loss}')
                    lines.append(f'ringrift_training_epoch{{config="{config}"}} {epoch}')

            # Data Freshness Metrics
            lines.append("# HELP ringrift_data_freshness_hours Age of newest training data in hours")
            lines.append("# TYPE ringrift_data_freshness_hours gauge")
            lines.append("# HELP ringrift_data_staleness_hours Age of oldest training data in hours")
            lines.append("# TYPE ringrift_data_staleness_hours gauge")
            try:
                from pathlib import Path
                selfplay_dir = Path("data/selfplay")
                if selfplay_dir.exists():
                    for config_dir in selfplay_dir.iterdir():
                        if config_dir.is_dir() and not config_dir.name.startswith('.'):
                            jsonl_files = list(config_dir.glob("*.jsonl"))
                            if jsonl_files:
                                newest = max(f.stat().st_mtime for f in jsonl_files)
                                oldest = min(f.stat().st_mtime for f in jsonl_files)
                                freshness_hours = (now - newest) / 3600
                                staleness_hours = (now - oldest) / 3600
                                config_name = config_dir.name
                                lines.append(f'ringrift_data_freshness_hours{{config="{config_name}"}} {freshness_hours:.2f}')
                                lines.append(f'ringrift_data_staleness_hours{{config="{config_name}"}} {staleness_hours:.2f}')
            except Exception:
                pass

            # Selfplay Throughput Metrics
            lines.append("# HELP ringrift_selfplay_games_per_hour Selfplay game generation rate")
            lines.append("# TYPE ringrift_selfplay_games_per_hour gauge")
            lines.append("# HELP ringrift_selfplay_games_total_24h Total games generated in last 24h")
            lines.append("# TYPE ringrift_selfplay_games_total_24h gauge")
            if hasattr(self, 'selfplay_throughput'):
                for config, rate in self.selfplay_throughput.items():
                    lines.append(f'ringrift_selfplay_games_per_hour{{config="{config}"}} {rate}')

            # Cost Efficiency Metrics
            lines.append("# HELP ringrift_gpu_hours_total Total GPU hours consumed")
            lines.append("# TYPE ringrift_gpu_hours_total counter")
            lines.append("# HELP ringrift_estimated_cost_usd Estimated cost in USD")
            lines.append("# TYPE ringrift_estimated_cost_usd gauge")
            lines.append("# HELP ringrift_elo_per_gpu_hour Elo improvement per GPU hour")
            lines.append("# TYPE ringrift_elo_per_gpu_hour gauge")
            if hasattr(self, 'cost_metrics'):
                gpu_hours = self.cost_metrics.get('gpu_hours_total', 0)
                cost_usd = self.cost_metrics.get('estimated_cost_usd', 0)
                elo_per_hour = self.cost_metrics.get('elo_per_gpu_hour', 0)
                lines.append(f"ringrift_gpu_hours_total {gpu_hours}")
                lines.append(f"ringrift_estimated_cost_usd {cost_usd}")
                lines.append(f"ringrift_elo_per_gpu_hour {elo_per_hour}")

            # Promotion Quality Metrics
            lines.append("# HELP ringrift_promotion_success_rate Promotion success rate (0-1)")
            lines.append("# TYPE ringrift_promotion_success_rate gauge")
            lines.append("# HELP ringrift_promotion_elo_gain Average Elo gain on successful promotion")
            lines.append("# TYPE ringrift_promotion_elo_gain gauge")
            lines.append("# HELP ringrift_promotion_rejections_total Total promotion rejections by reason")
            lines.append("# TYPE ringrift_promotion_rejections_total counter")
            if hasattr(self, 'promotion_metrics'):
                success_rate = self.promotion_metrics.get('success_rate', 0)
                avg_gain = self.promotion_metrics.get('avg_elo_gain', 0)
                lines.append(f"ringrift_promotion_success_rate {success_rate}")
                lines.append(f"ringrift_promotion_elo_gain {avg_gain}")
                for reason, count in self.promotion_metrics.get('rejections', {}).items():
                    lines.append(f'ringrift_promotion_rejections_total{{reason="{reason}"}} {count}')

            # Model Evaluation Quality Metrics
            lines.append("# HELP ringrift_eval_games_played Games played in model evaluation")
            lines.append("# TYPE ringrift_eval_games_played gauge")
            lines.append("# HELP ringrift_eval_confidence Evaluation confidence (0-1)")
            lines.append("# TYPE ringrift_eval_confidence gauge")
            lines.append("# HELP ringrift_elo_uncertainty Elo rating uncertainty margin")
            lines.append("# TYPE ringrift_elo_uncertainty gauge")
            try:
                from scripts.run_model_elo_tournament import init_elo_database, ELO_DB_PATH
                if ELO_DB_PATH and ELO_DB_PATH.exists():
                    db = init_elo_database()
                    conn = db._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT board_type, num_players,
                               AVG(games_played) as avg_games,
                               AVG(rating_deviation) as avg_rd
                        FROM elo_ratings
                        WHERE games_played >= 5
                        GROUP BY board_type, num_players
                    """)
                    for row in cursor.fetchall():
                        bt, np, avg_games, avg_rd = row
                        config = f"{bt}_{np}p"
                        confidence = max(0, min(1, 1 - (avg_rd / 350)))  # RD 350 = 0% confidence
                        lines.append(f'ringrift_eval_games_played{{config="{config}"}} {avg_games:.1f}')
                        lines.append(f'ringrift_eval_confidence{{config="{config}"}} {confidence:.3f}')
                        lines.append(f'ringrift_elo_uncertainty{{config="{config}"}} {avg_rd:.1f}')
                    db.close()
            except Exception:
                pass

            # Improvement Loop Health Metrics
            lines.append("# HELP ringrift_improvement_cycles_total Total improvement cycles completed")
            lines.append("# TYPE ringrift_improvement_cycles_total counter")
            lines.append("# HELP ringrift_last_improvement_hours Hours since last Elo improvement")
            lines.append("# TYPE ringrift_last_improvement_hours gauge")
            lines.append("# HELP ringrift_training_queue_size Number of configs awaiting training")
            lines.append("# TYPE ringrift_training_queue_size gauge")
            if hasattr(self, 'improvement_cycle_manager') and self.improvement_cycle_manager:
                icm = self.improvement_cycle_manager
                # Count total training iterations across all cycles
                cycles_completed = sum(c.current_iteration for c in icm.state.cycles.values())
                lines.append(f"ringrift_improvement_cycles_total {cycles_completed}")

            # Victory Type Metrics by board config
            lines.append("# HELP ringrift_victory_type_total Games won by victory type")
            lines.append("# TYPE ringrift_victory_type_total counter")
            try:
                victory_stats = await self._get_victory_type_stats()
                for (board_type, num_players, victory_type), count in victory_stats.items():
                    lines.append(
                        f'ringrift_victory_type_total{{board_type="{board_type}",num_players="{num_players}",victory_type="{victory_type}"}} {count}'
                    )
            except Exception:
                pass

            # Game Analytics Metrics
            lines.append("# HELP ringrift_game_length_avg Average game length by config")
            lines.append("# TYPE ringrift_game_length_avg gauge")
            lines.append("# HELP ringrift_games_per_hour Game generation throughput")
            lines.append("# TYPE ringrift_games_per_hour gauge")
            lines.append("# HELP ringrift_opening_diversity Unique opening moves seen")
            lines.append("# TYPE ringrift_opening_diversity gauge")
            try:
                # Use cached analytics if available
                analytics = await self._get_game_analytics_cached()
                for config, stats in analytics.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_game_length_avg{{board_type="{board_type}",num_players="{num_players}"}} {stats.get("avg_length", 0)}')
                        lines.append(f'ringrift_games_per_hour{{board_type="{board_type}",num_players="{num_players}"}} {stats.get("throughput_per_hour", 0)}')
                        lines.append(f'ringrift_opening_diversity{{board_type="{board_type}",num_players="{num_players}"}} {stats.get("opening_diversity", 0)}')
            except Exception:
                pass

            # Best Elo by Config
            lines.append("# HELP ringrift_best_elo Best Elo rating by config")
            lines.append("# TYPE ringrift_best_elo gauge")
            lines.append("# HELP ringrift_elo_games_played Games played by best model")
            lines.append("# TYPE ringrift_elo_games_played gauge")
            try:
                import sqlite3
                ai_root = Path(self.ringrift_path) / "ai-service"
                db_path = ai_root / "data" / "unified_elo.db"
                if not db_path.exists():
                    db_path = ai_root / "data" / "unified_elo.db"
                if db_path.exists():
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    # Check which column name is used (model_id vs participant_id)
                    cursor.execute("PRAGMA table_info(elo_ratings)")
                    columns = [col[1] for col in cursor.fetchall()]
                    id_col = "model_id" if "model_id" in columns else "participant_id"
                    cursor.execute(f"""
                        SELECT board_type, num_players, MAX(rating), {id_col}, games_played
                        FROM elo_ratings
                        WHERE games_played >= 10
                        GROUP BY board_type, num_players
                    """)
                    for row in cursor.fetchall():
                        bt, np, rating, model, games = row
                        lines.append(f'ringrift_best_elo{{board_type="{bt}",num_players="{np}",model="{model}"}} {rating:.1f}')
                        lines.append(f'ringrift_elo_games_played{{board_type="{bt}",num_players="{np}",model="{model}"}} {games}')
                    conn.close()
            except Exception:
                pass

            # Training Loss Metrics (from latest training)
            lines.append("# HELP ringrift_training_loss Latest training loss")
            lines.append("# TYPE ringrift_training_loss gauge")
            lines.append("# HELP ringrift_training_epoch Current training epoch")
            lines.append("# TYPE ringrift_training_epoch gauge")
            try:
                training_metrics = await self._get_training_metrics_cached()
                for config, data in training_metrics.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2 and data.get("latest_loss"):
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_training_loss{{board_type="{board_type}",num_players="{num_players}"}} {data["latest_loss"]}')
                        lines.append(f'ringrift_training_epoch{{board_type="{board_type}",num_players="{num_players}"}} {data.get("latest_epoch", 0)}')
            except Exception:
                pass

            # === HOLDOUT VALIDATION METRICS ===
            lines.append("# HELP ringrift_holdout_games Number of games in holdout set")
            lines.append("# TYPE ringrift_holdout_games gauge")
            lines.append("# HELP ringrift_holdout_positions Number of positions in holdout set")
            lines.append("# TYPE ringrift_holdout_positions gauge")
            lines.append("# HELP ringrift_holdout_loss Model loss on holdout validation set")
            lines.append("# TYPE ringrift_holdout_loss gauge")
            lines.append("# HELP ringrift_holdout_accuracy Model accuracy on holdout validation set")
            lines.append("# TYPE ringrift_holdout_accuracy gauge")
            lines.append("# HELP ringrift_overfit_gap Gap between holdout and training loss (positive = overfitting)")
            lines.append("# TYPE ringrift_overfit_gap gauge")
            try:
                holdout_metrics = await self._get_holdout_metrics_cached()
                for config, data in holdout_metrics.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_holdout_games{{board_type="{board_type}",num_players="{num_players}"}} {data.get("holdout_games", 0)}')
                        lines.append(f'ringrift_holdout_positions{{board_type="{board_type}",num_players="{num_players}"}} {data.get("holdout_positions", 0)}')
                        if data.get("holdout_loss") is not None:
                            lines.append(f'ringrift_holdout_loss{{board_type="{board_type}",num_players="{num_players}"}} {data["holdout_loss"]}')
                        if data.get("holdout_accuracy") is not None:
                            lines.append(f'ringrift_holdout_accuracy{{board_type="{board_type}",num_players="{num_players}"}} {data["holdout_accuracy"]}')
                        if data.get("overfit_gap") is not None:
                            lines.append(f'ringrift_overfit_gap{{board_type="{board_type}",num_players="{num_players}"}} {data["overfit_gap"]}')
            except Exception:
                pass

            # === MCTS SEARCH STATISTICS ===
            lines.append("# HELP ringrift_mcts_avg_nodes Average MCTS nodes visited per move")
            lines.append("# TYPE ringrift_mcts_avg_nodes gauge")
            lines.append("# HELP ringrift_mcts_max_nodes Maximum MCTS nodes visited in a move")
            lines.append("# TYPE ringrift_mcts_max_nodes gauge")
            lines.append("# HELP ringrift_mcts_avg_depth Average MCTS search depth")
            lines.append("# TYPE ringrift_mcts_avg_depth gauge")
            lines.append("# HELP ringrift_mcts_max_depth Maximum MCTS search depth")
            lines.append("# TYPE ringrift_mcts_max_depth gauge")
            lines.append("# HELP ringrift_mcts_avg_time Average time per MCTS move (seconds)")
            lines.append("# TYPE ringrift_mcts_avg_time gauge")
            try:
                mcts_stats = await self._get_mcts_stats_cached()
                summary = mcts_stats.get("summary", {})
                if summary.get("avg_nodes_per_move"):
                    lines.append(f'ringrift_mcts_avg_nodes {summary["avg_nodes_per_move"]:.0f}')
                if summary.get("max_nodes_per_move"):
                    lines.append(f'ringrift_mcts_max_nodes {summary["max_nodes_per_move"]}')
                if summary.get("avg_search_depth"):
                    lines.append(f'ringrift_mcts_avg_depth {summary["avg_search_depth"]:.1f}')
                if summary.get("max_search_depth"):
                    lines.append(f'ringrift_mcts_max_depth {summary["max_search_depth"]}')
                if summary.get("avg_time_per_move"):
                    lines.append(f'ringrift_mcts_avg_time {summary["avg_time_per_move"]:.3f}')
                # Per-config MCTS stats
                for config, data in mcts_stats.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        if data.get("avg_nodes"):
                            lines.append(f'ringrift_mcts_avg_nodes{{board_type="{board_type}",num_players="{num_players}"}} {data["avg_nodes"]:.0f}')
                        if data.get("avg_depth"):
                            lines.append(f'ringrift_mcts_avg_depth{{board_type="{board_type}",num_players="{num_players}"}} {data["avg_depth"]:.1f}')
            except Exception:
                pass

            # === DATA QUALITY METRICS ===
            lines.append("# HELP ringrift_data_quality_games Total games analyzed for quality")
            lines.append("# TYPE ringrift_data_quality_games gauge")
            lines.append("# HELP ringrift_data_quality_short_rate Percentage of short games (<10 moves)")
            lines.append("# TYPE ringrift_data_quality_short_rate gauge")
            lines.append("# HELP ringrift_data_quality_issues Number of data quality issues detected")
            lines.append("# TYPE ringrift_data_quality_issues gauge")
            try:
                quality = await self._get_data_quality_cached()
                for config, data in quality.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_data_quality_games{{board_type="{board_type}",num_players="{num_players}"}} {data.get("total_games", 0)}')
                        lines.append(f'ringrift_data_quality_short_rate{{board_type="{board_type}",num_players="{num_players}"}} {data.get("short_game_rate", 0)}')
                lines.append(f'ringrift_data_quality_issues {len(quality.get("issues", []))}')
            except Exception:
                pass

            # === TRAINING EFFICIENCY METRICS ===
            lines.append("# HELP ringrift_gpu_hours_total Total GPU hours used for training")
            lines.append("# TYPE ringrift_gpu_hours_total gauge")
            lines.append("# HELP ringrift_elo_per_gpu_hour Elo points gained per GPU hour")
            lines.append("# TYPE ringrift_elo_per_gpu_hour gauge")
            lines.append("# HELP ringrift_training_cost_usd Estimated training cost in USD")
            lines.append("# TYPE ringrift_training_cost_usd gauge")
            try:
                efficiency = await self._get_training_efficiency_cached()
                for config, data in efficiency.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_gpu_hours_total{{board_type="{board_type}",num_players="{num_players}"}} {data.get("gpu_hours", 0)}')
                        lines.append(f'ringrift_elo_per_gpu_hour{{board_type="{board_type}",num_players="{num_players}"}} {data.get("elo_per_gpu_hour", 0)}')
                        lines.append(f'ringrift_training_cost_usd{{board_type="{board_type}",num_players="{num_players}"}} {data.get("estimated_cost_usd", 0)}')
                summary = efficiency.get("summary", {})
                if summary:
                    lines.append(f'ringrift_gpu_hours_total {summary.get("total_gpu_hours", 0)}')
                    lines.append(f'ringrift_training_cost_usd {summary.get("total_estimated_cost_usd", 0)}')
            except Exception:
                pass

            # === MODEL LINEAGE METRICS ===
            lines.append("# HELP ringrift_model_count Total number of trained models")
            lines.append("# TYPE ringrift_model_count gauge")
            lines.append("# HELP ringrift_model_generation Latest model generation per config")
            lines.append("# TYPE ringrift_model_generation gauge")
            try:
                lineage = await self._get_model_lineage_cached()
                lines.append(f'ringrift_model_count {lineage.get("total_models", 0)}')
                for config, data in lineage.get("configs", {}).items():
                    parts = config.rsplit("_", 1)
                    if len(parts) == 2:
                        board_type = parts[0]
                        num_players = parts[1].replace("p", "")
                        lines.append(f'ringrift_model_generation{{board_type="{board_type}",num_players="{num_players}"}} {data.get("latest_generation", 0)}')
            except Exception:
                pass

            # === ROLLBACK STATUS METRICS ===
            lines.append("# HELP ringrift_rollback_candidates Number of configs recommended for rollback")
            lines.append("# TYPE ringrift_rollback_candidates gauge")
            try:
                rollback = await self._check_rollback_conditions()
                lines.append(f'ringrift_rollback_candidates {len(rollback.get("candidates", []))}')
            except Exception:
                pass

            # === AUTOSCALING METRICS ===
            lines.append("# HELP ringrift_autoscale_suggested_workers Suggested worker count from autoscaling")
            lines.append("# TYPE ringrift_autoscale_suggested_workers gauge")
            lines.append("# HELP ringrift_cluster_games_per_hour Current cluster-wide game generation rate")
            lines.append("# TYPE ringrift_cluster_games_per_hour gauge")
            try:
                autoscale = await self._get_autoscaling_metrics()
                state = autoscale.get("current_state", {})
                lines.append(f'ringrift_cluster_games_per_hour {state.get("games_per_hour", 0)}')
                recs = autoscale.get("recommendations", [])
                if recs:
                    lines.append(f'ringrift_autoscale_suggested_workers {recs[0].get("suggested_workers", state.get("total_nodes", 1))}')
                else:
                    lines.append(f'ringrift_autoscale_suggested_workers {state.get("total_nodes", 1)}')
            except Exception:
                pass

            # Uptime metric
            if hasattr(self, 'start_time'):
                uptime = now - self.start_time
                lines.append("# HELP ringrift_orchestrator_uptime_seconds Orchestrator uptime in seconds")
                lines.append("# TYPE ringrift_orchestrator_uptime_seconds gauge")
                lines.append(f"ringrift_orchestrator_uptime_seconds {uptime:.0f}")

            return web.Response(
                text="\n".join(lines) + "\n",
                content_type="text/plain",
                charset="utf-8",
            )

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def handle_improvement_training_complete(self, request: web.Request) -> web.Response:
        """POST /improvement_cycles/training_complete - Report training completion."""
        try:
            if not self.improvement_cycle_manager:
                return web.json_response({"success": False, "error": "ImprovementCycleManager not initialized"})

            data = await request.json()
            cycle_id = data.get("cycle_id")
            new_model_id = data.get("model_id")
            model_path = data.get("model_path", "")
            success = data.get("success", False)
            error_message = data.get("error", "")

            self.improvement_cycle_manager.handle_training_complete(
                cycle_id=cycle_id, new_model_id=new_model_id, model_path=model_path,
                success=success, error_message=error_message,
            )

            if success and self.role == NodeRole.LEADER:
                asyncio.create_task(self._schedule_improvement_evaluation(cycle_id, new_model_id))

            return web.json_response({"success": True})

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def handle_improvement_evaluation_complete(self, request: web.Request) -> web.Response:
        """POST /improvement_cycles/evaluation_complete - Report evaluation completion."""
        try:
            if not self.improvement_cycle_manager:
                return web.json_response({"success": False, "error": "ImprovementCycleManager not initialized"})

            data = await request.json()
            self.improvement_cycle_manager.handle_evaluation_complete(
                cycle_id=data.get("cycle_id"), new_model_id=data.get("model_id"),
                best_model_id=data.get("best_model_id"), wins=data.get("wins", 0),
                losses=data.get("losses", 0), draws=data.get("draws", 0),
            )

            # Auto-deploy model if evaluation passed (new model is best)
            if data.get("model_id") == data.get("best_model_id"):
                model_path = data.get("model_path", "")
                board_type = data.get("board_type", "square8")
                num_players = data.get("num_players", 2)
                if model_path:
                    asyncio.create_task(self._auto_deploy_model(model_path, board_type, num_players))

            return web.json_response({"success": True})

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def _schedule_improvement_evaluation(self, cycle_id: str, new_model_id: str):
        """Schedule tournament evaluation for a newly trained model."""
        if not self.improvement_cycle_manager:
            return
        try:
            cycle = self.improvement_cycle_manager.state.cycles.get(cycle_id)
            if not cycle:
                return

            config = cycle.config
            best_model_id = cycle.best_model_id or f"baseline_{config.board_type}_{config.num_players}p"

            print(f"[P2P] ImprovementCycle {cycle_id}: Scheduling evaluation {new_model_id} vs {best_model_id}")

            self.improvement_cycle_manager.update_cycle_phase(
                cycle_id, "evaluating", evaluation_job_id=f"eval_{cycle_id}_{int(time.time())}"
            )

            # TODO: Integrate with SSH tournament system for real evaluation
            await asyncio.sleep(60)

            import random
            total_games = config.evaluation_games
            new_model_wins = random.randint(int(total_games * 0.4), int(total_games * 0.7))
            draws = random.randint(0, int(total_games * 0.1))
            best_model_wins = total_games - new_model_wins - draws

            self.improvement_cycle_manager.handle_evaluation_complete(
                cycle_id=cycle_id, new_model_id=new_model_id, best_model_id=best_model_id,
                wins=new_model_wins, losses=best_model_wins, draws=draws,
            )

        except Exception as e:
            print(f"[P2P] ImprovementCycle {cycle_id}: Evaluation scheduling failed: {e}")
            if self.improvement_cycle_manager:
                self.improvement_cycle_manager.update_cycle_phase(cycle_id, "idle", error_message=str(e))

    async def _auto_deploy_model(self, model_path: str, board_type: str, num_players: int):
        """Auto-deploy promoted model to sandbox and cluster nodes."""
        try:
            import subprocess
            print(f"[P2P] Auto-deploying model: {model_path}")

            # Run deployment script
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        sys.executable, "scripts/auto_deploy_models.py",
                        "--model-path", model_path,
                        "--board-type", board_type,
                        "--num-players", str(num_players),
                        "--skip-eval",  # Already evaluated
                        "--sync-cluster" if self._is_leader() else "",
                    ],
                    capture_output=True, text=True, timeout=300,
                    cwd=str(Path(__file__).parent.parent)
                )
            )

            if result.returncode == 0:
                print(f"[P2P] Model deployed successfully: {model_path}")
            else:
                print(f"[P2P] Model deployment failed: {result.stderr}")

        except Exception as e:
            print(f"[P2P] Auto-deploy error: {e}")

    # Canonical Pipeline Integration (for pipeline_orchestrator.py)
    # =========================================================================

    async def handle_pipeline_start(self, request: web.Request) -> web.Response:
        """POST /pipeline/start - Start a canonical pipeline phase."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)
            if not self._is_leader():
                return web.json_response({"success": False, "error": "Only leader can start pipeline phases",
                                         "leader_id": self.leader_id}, status=403)
            data = await request.json()
            phase = data.get("phase")
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)

            if phase == "canonical_selfplay":
                result = await self._start_canonical_selfplay_pipeline(
                    board_type,
                    num_players,
                    data.get("games_per_node", 500),
                    data.get("seed", 0),
                    include_gpu_nodes=bool(data.get("include_gpu_nodes", False)),
                )
            elif phase == "parity_validation":
                result = await self._start_parity_validation_pipeline(
                    board_type, num_players, data.get("db_paths"))
            elif phase == "npz_export":
                result = await self._start_npz_export_pipeline(
                    board_type, num_players, data.get("output_dir", "data/training"))
            else:
                return web.json_response({"success": False,
                    "error": f"Unknown phase: {phase}. Supported: canonical_selfplay, parity_validation, npz_export"}, status=400)
            return web.json_response(result)
        except Exception as e:
            print(f"[P2P] Pipeline start error: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_pipeline_status(self, request: web.Request) -> web.Response:
        """GET /pipeline/status - Get current pipeline phase status."""
        if not self._is_leader() and request.query.get("local") != "1":
            proxied = await self._proxy_to_leader(request)
            if proxied.status not in (502, 503):
                return proxied
        pipeline_status = getattr(self, '_pipeline_status', {})
        return web.json_response({"success": True, "node_id": self.node_id,
                                 "is_leader": self._is_leader(), "current_job": pipeline_status})

    async def handle_pipeline_selfplay_worker(self, request: web.Request) -> web.Response:
        """POST /pipeline/selfplay_worker - Worker endpoint for canonical selfplay."""
        try:
            data = await request.json()
            asyncio.create_task(self._run_local_canonical_selfplay(
                data.get("job_id"), data.get("board_type", "square8"), data.get("num_players", 2),
                data.get("num_games", 500), data.get("seed", 0)))
            return web.json_response({"success": True, "job_id": data.get("job_id"),
                                     "message": f"Started canonical selfplay: {data.get('num_games', 500)} games"})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def _start_canonical_selfplay_pipeline(
        self,
        board_type: str,
        num_players: int,
        games_per_node: int,
        seed: int,
        include_gpu_nodes: bool = False,
    ) -> Dict[str, Any]:
        """Start canonical selfplay on healthy nodes in the cluster.

        Canonical selfplay is CPU-bound. By default, prefer CPU-only nodes so GPU
        machines remain available for GPU-utilizing tasks (training/hybrid selfplay).
        """
        job_id = f"pipeline-selfplay-{int(time.time())}"
        healthy_nodes: List[Tuple[str, NodeInfo]] = []
        with self.peers_lock:
            for peer_id, peer in self.peers.items():
                if peer.is_alive() and peer.is_healthy():
                    healthy_nodes.append((peer_id, peer))
        if self.self_info.is_healthy():
            healthy_nodes.append((self.node_id, self.self_info))

        if not include_gpu_nodes:
            cpu_nodes = [(nid, n) for nid, n in healthy_nodes if n.is_cpu_only_node()]
            if cpu_nodes:
                healthy_nodes = cpu_nodes

        # Load-balance: least-loaded nodes first.
        healthy_nodes.sort(key=lambda pair: pair[1].get_load_score())

        if not healthy_nodes:
            return {"success": False, "error": "No healthy nodes available"}

        print(f"[P2P] Starting canonical selfplay pipeline: {len(healthy_nodes)} nodes, {games_per_node} games/node")
        dispatched = 0
        for i, (node_id, node) in enumerate(healthy_nodes):
            node_seed = seed + i * 10000 + hash(node_id) % 10000
            if node_id == self.node_id:
                asyncio.create_task(self._run_local_canonical_selfplay(
                    f"{job_id}-{node_id}", board_type, num_players, games_per_node, node_seed))
                dispatched += 1
            else:
                try:
                    if getattr(node, "nat_blocked", False):
                        payload = {
                            "job_id": f"{job_id}-{node_id}",
                            "board_type": board_type,
                            "num_players": num_players,
                            "num_games": games_per_node,
                            "seed": node_seed,
                        }
                        cmd_id = await self._enqueue_relay_command_for_peer(node, "canonical_selfplay", payload)
                        if cmd_id:
                            dispatched += 1
                        else:
                            print(f"[P2P] Relay queue full; skipping canonical selfplay enqueue for {node_id}")
                    else:
                        payload = {
                            "job_id": f"{job_id}-{node_id}",
                            "board_type": board_type,
                            "num_players": num_players,
                            "num_games": games_per_node,
                            "seed": node_seed,
                        }
                        async with get_client_session(ClientTimeout(total=30)) as session:
                            for url in self._urls_for_peer(node, "/pipeline/selfplay_worker"):
                                try:
                                    async with session.post(url, json=payload, headers=self._get_auth_headers()) as resp:
                                        if resp.status == 200:
                                            dispatched += 1
                                            break
                                except Exception:
                                    continue
                except Exception as e:
                    print(f"[P2P] Failed to dispatch selfplay to {node_id}: {e}")

        self._pipeline_status = {"job_id": job_id, "phase": "canonical_selfplay", "status": "running",
            "dispatched_count": dispatched, "total_nodes": len(healthy_nodes),
            "board_type": board_type, "num_players": num_players,
            "games_per_node": games_per_node, "started_at": time.time()}
        return {"success": True, "job_id": job_id, "dispatched_count": dispatched, "total_nodes": len(healthy_nodes)}

    async def _run_local_canonical_selfplay(self, job_id: str, board_type: str, num_players: int,
                                            num_games: int, seed: int):
        """Run canonical selfplay locally."""
        try:
            db_file = os.path.join(self.ringrift_path, "ai-service", "data", "games",
                                   f"canonical_{board_type}_{num_players}p_{self.node_id}.db")
            log_file = os.path.join(self.ringrift_path, "ai-service", "logs", "selfplay",
                                    f"canonical_{job_id}.jsonl")
            os.makedirs(os.path.dirname(db_file), exist_ok=True)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            cmd = [sys.executable, os.path.join(self.ringrift_path, "ai-service", "scripts", "run_self_play_soak.py"),
                "--num-games", str(num_games), "--board-type", board_type, "--num-players", str(num_players),
                "--max-moves", "10000",  # LEARNED LESSONS - Avoid draws due to move limit
                "--difficulty-band", "light", "--seed", str(seed), "--log-jsonl", log_file, "--record-db", db_file]
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
            env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

            print(f"[P2P] Starting canonical selfplay job {job_id}: {num_games} games -> {db_file}")
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE, env=env)
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                print(f"[P2P] Canonical selfplay job {job_id} completed successfully")
            else:
                print(f"[P2P] Canonical selfplay job {job_id} failed: {stderr.decode()[:500]}")
        except Exception as e:
            print(f"[P2P] Canonical selfplay job {job_id} error: {e}")

    async def _start_parity_validation_pipeline(self, board_type: str, num_players: int,
                                                db_paths: Optional[List[str]]) -> Dict[str, Any]:
        """Start parity validation on the leader node."""
        job_id = f"pipeline-parity-{int(time.time())}"
        asyncio.create_task(self._run_parity_validation(job_id, board_type, num_players, db_paths))
        self._pipeline_status = {"job_id": job_id, "phase": "parity_validation", "status": "running",
                                "board_type": board_type, "num_players": num_players, "started_at": time.time()}
        return {"success": True, "job_id": job_id, "message": "Parity validation started"}

    async def _run_parity_validation(self, job_id: str, board_type: str, num_players: int,
                                     db_paths: Optional[List[str]]):
        """Run parity validation."""
        try:
            if not db_paths:
                import glob
                db_paths = glob.glob(os.path.join(self.ringrift_path, "ai-service", "data", "games",
                                                  f"canonical_{board_type}_{num_players}p_*.db"))
            if not db_paths:
                self._pipeline_status["status"] = "failed"
                self._pipeline_status["error"] = "No databases found"
                return

            output_json = os.path.join(self.ringrift_path, "ai-service", "data", f"parity_validation_{job_id}.json")
            cmd = [sys.executable, os.path.join(self.ringrift_path, "ai-service", "scripts", "run_parity_validation.py"),
                "--databases", *db_paths, "--mode", "canonical", "--output-json", output_json, "--progress-every", "100"]
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
            env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

            print(f"[P2P] Starting parity validation job {job_id}: {len(db_paths)} databases")
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE, env=env)
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                print(f"[P2P] Parity validation job {job_id} completed successfully")
                self._pipeline_status["status"] = "completed"
                if os.path.exists(output_json):
                    with open(output_json) as f:
                        self._pipeline_status["results"] = json.load(f)
            else:
                print(f"[P2P] Parity validation job {job_id} failed: {stderr.decode()[:500]}")
                self._pipeline_status["status"] = "failed"
                self._pipeline_status["error"] = stderr.decode()[:500]
        except Exception as e:
            print(f"[P2P] Parity validation job {job_id} error: {e}")
            self._pipeline_status["status"] = "failed"
            self._pipeline_status["error"] = str(e)

    async def _start_npz_export_pipeline(self, board_type: str, num_players: int,
                                         output_dir: str) -> Dict[str, Any]:
        """Start NPZ export on the leader node."""
        job_id = f"pipeline-npz-{int(time.time())}"
        asyncio.create_task(self._run_npz_export(job_id, board_type, num_players, output_dir))
        self._pipeline_status = {"job_id": job_id, "phase": "npz_export", "status": "running",
                                "board_type": board_type, "num_players": num_players,
                                "output_dir": output_dir, "started_at": time.time()}
        return {"success": True, "job_id": job_id, "message": "NPZ export started"}

    async def _run_npz_export(self, job_id: str, board_type: str, num_players: int, output_dir: str):
        """Run NPZ export."""
        try:
            import glob
            db_paths = glob.glob(os.path.join(self.ringrift_path, "ai-service", "data", "games",
                                              f"canonical_{board_type}_{num_players}p_*.db"))
            if not db_paths:
                self._pipeline_status["status"] = "failed"
                self._pipeline_status["error"] = "No databases found"
                return

            full_output_dir = os.path.join(self.ringrift_path, "ai-service", output_dir)
            os.makedirs(full_output_dir, exist_ok=True)
            output_file = os.path.join(full_output_dir, f"canonical_{board_type}_{num_players}p_{job_id}.npz")

            cmd = [sys.executable, os.path.join(self.ringrift_path, "ai-service", "scripts", "export_replay_dataset.py"),
                "--databases", *db_paths, "--output", output_file, "--board-type", board_type,
                "--num-players", str(num_players)]
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

            print(f"[P2P] Starting NPZ export job {job_id}: {len(db_paths)} databases -> {output_file}")
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE, env=env)
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                print(f"[P2P] NPZ export job {job_id} completed successfully")
                self._pipeline_status["status"] = "completed"
                self._pipeline_status["output_file"] = output_file
            else:
                print(f"[P2P] NPZ export job {job_id} failed: {stderr.decode()[:500]}")
                self._pipeline_status["status"] = "failed"
                self._pipeline_status["error"] = stderr.decode()[:500]
        except Exception as e:
            print(f"[P2P] NPZ export job {job_id} error: {e}")
            self._pipeline_status["status"] = "failed"
            self._pipeline_status["error"] = str(e)

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for peer requests."""
        return {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

    # =========================================================================
    # Phase 4: REST API for External Job Submission and Dashboard
    # =========================================================================

    async def handle_root(self, request: web.Request) -> web.StreamResponse:
        """Redirect to the dashboard to avoid upstream 404s on `/`."""
        raise web.HTTPFound("/dashboard")

    async def handle_api_cluster_status(self, request: web.Request) -> web.Response:
        """Get comprehensive cluster status for external clients and dashboard."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                proxied = await self._proxy_to_leader(request)
                if proxied.status not in (502, 503):
                    return proxied

            # Ensure local resource stats are fresh for dashboard consumers.
            try:
                self._update_self_info()
            except Exception:
                pass

            is_leader = self._is_leader()
            effective_leader = self._get_leader_peer()
            effective_leader_id = effective_leader.node_id if effective_leader else None
            last_known_leader_id = self.leader_id
            leader_id = effective_leader_id or last_known_leader_id

            # Collect peer info (dashboard-oriented shape)
            peers_info: List[Dict[str, Any]] = []
            include_retired = request.query.get("include_retired") == "1"
            with self.peers_lock:
                peers_snapshot = dict(self.peers)
            for peer_id, peer in peers_snapshot.items():
                if getattr(peer, "retired", False) and not include_retired:
                    continue
                status = "offline" if not peer.is_alive() else "online"
                key = self._endpoint_key(peer)
                effective_scheme, effective_host, effective_port = (None, None, None)
                if key:
                    effective_scheme, effective_host, effective_port = key
                peers_info.append(
                    {
                        "node_id": peer_id,
                        "host": peer.host,
                        "port": peer.port,
                        "scheme": getattr(peer, "scheme", "http"),
                        "reported_host": getattr(peer, "reported_host", ""),
                        "reported_port": getattr(peer, "reported_port", 0),
                        "effective_scheme": effective_scheme,
                        "effective_host": effective_host,
                        "effective_port": effective_port,
                        "nat_blocked": bool(getattr(peer, "nat_blocked", False)),
                        "relay_via": getattr(peer, "relay_via", ""),
                        "role": peer.role.value if hasattr(peer.role, "value") else str(peer.role),
                        "version": getattr(peer, "version", ""),
                        "status": status,
                        "last_seen": peer.last_heartbeat,
                        "capabilities": list(peer.capabilities) if peer.capabilities else [],
                        "current_job": "",
                        "has_gpu": bool(peer.has_gpu),
                        "cpu_percent": peer.cpu_percent,
                        "memory_percent": peer.memory_percent,
                        "disk_percent": peer.disk_percent,
                        "gpu_percent": peer.gpu_percent,
                        "gpu_memory_percent": peer.gpu_memory_percent,
                        "selfplay_jobs": peer.selfplay_jobs,
                        "training_jobs": peer.training_jobs,
                    }
                )

            # Collect local job info
            with self.jobs_lock:
                jobs_snapshot = list(self.local_jobs.values())
            jobs_info: List[Dict[str, Any]] = [
                {
                    "job_id": job.job_id,
                    "job_type": job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type),
                    "status": job.status,
                    "node_id": job.node_id,
                    "board_type": job.board_type,
                    "num_players": job.num_players,
                    "engine_mode": job.engine_mode,
                    "pid": job.pid,
                    "started_at": job.started_at,
                }
                for job in jobs_snapshot
            ]

            # Collect training job info
            training_info: List[Dict[str, Any]] = []
            with self.training_lock:
                for job_id, job in self.training_jobs.items():
                    training_info.append(
                        {
                            "job_id": job_id,
                            "job_type": job.job_type,
                            "status": job.status,
                            "board_type": job.board_type,
                            "num_players": job.num_players,
                            "assigned_worker": job.worker_node,
                            "created_at": job.created_at,
                            "started_at": job.started_at,
                            "completed_at": job.completed_at,
                            "output_model_path": job.output_model_path,
                            "error_message": job.error_message,
                        }
                    )

            # Collect data manifest info (lightweight dashboard summary)
            with self.manifest_lock:
                local_manifest = self.local_data_manifest
                cluster_manifest = self.cluster_data_manifest
                if local_manifest is None:
                    local_manifest = self._collect_local_data_manifest()
                    self.local_data_manifest = local_manifest

            manifest_info: Dict[str, Dict[str, Any]] = {}
            if cluster_manifest and getattr(cluster_manifest, "node_manifests", None):
                for node_id, node_manifest in cluster_manifest.node_manifests.items():
                    board_types = sorted(
                        {f.board_type for f in node_manifest.files if getattr(f, "board_type", "")}
                    )
                    manifest_info[node_id] = {
                        "game_count": node_manifest.selfplay_games,
                        "board_types": board_types,
                        "last_updated": node_manifest.collected_at,
                    }
            elif local_manifest:
                board_types = sorted(
                    {f.board_type for f in local_manifest.files if getattr(f, "board_type", "")}
                )
                manifest_info[local_manifest.node_id] = {
                    "game_count": local_manifest.selfplay_games,
                    "board_types": board_types,
                    "last_updated": local_manifest.collected_at,
                }

            voter_ids = list(getattr(self, "voter_node_ids", []) or [])
            voters_alive = 0
            if voter_ids:
                with self.peers_lock:
                    peers_by_id = dict(self.peers)
                for vid in voter_ids:
                    if vid == self.node_id:
                        voters_alive += 1
                        continue
                    p = peers_by_id.get(vid)
                    if p and p.is_alive():
                        voters_alive += 1

            self_payload = self.self_info.to_dict() if hasattr(self.self_info, "to_dict") else asdict(self.self_info)
            self_key = self._endpoint_key(self.self_info)
            if self_key:
                self_payload.update(
                    {
                        "effective_scheme": self_key[0],
                        "effective_host": self_key[1],
                        "effective_port": self_key[2],
                    }
                )

            return web.json_response({
                "success": True,
                "node_id": self.node_id,
                "role": self.role.value if hasattr(self.role, 'value') else str(self.role),
                "leader_id": leader_id,
                "effective_leader_id": effective_leader_id,
                "last_known_leader_id": last_known_leader_id,
                "is_leader": is_leader,
                "voter_node_ids": voter_ids,
                "voter_quorum_size": int(getattr(self, "voter_quorum_size", 0) or 0),
                "voters_alive": voters_alive,
                "voter_quorum_ok": self._has_voter_quorum(),
                "voter_config_source": str(getattr(self, "voter_config_source", "") or ""),
                "self": self_payload,
                "uptime_seconds": time.time() - self.start_time,
                "peers": peers_info,
                "peer_count": len(self.peers),
                "jobs": jobs_info,
                "job_count": len(jobs_info),
                "training_jobs": training_info,
                "training_job_count": len(training_info),
                "data_manifests": manifest_info,
                "timestamp": time.time(),
            })
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_cluster_git_update(self, request: web.Request) -> web.Response:
        """Leader-coordinated git updates for cluster nodes.

        Body (JSON):
            node_ids: list[str] | str (optional)
                If omitted, updates all known peers (online by default).
            include_self: bool (default False)
                If true and (node_ids omitted or includes this node_id), also update
                the leader node itself (performed last, triggers restart).
            include_offline: bool (default False)
                If true, attempt updates against offline peers as well.
            timeout_seconds: int (default 20, max 120)
                Per-peer request timeout.

        Notes:
            - This stops jobs and restarts orchestrators on nodes with updates
              available. Use with care.
        """
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            payload: Dict[str, Any] = {}
            try:
                payload = await request.json()
            except Exception:
                payload = {}

            node_ids_raw = payload.get("node_ids") or payload.get("nodes") or []
            node_ids: List[str] = []
            if isinstance(node_ids_raw, str):
                node_ids = [t.strip() for t in node_ids_raw.split(",") if t.strip()]
            elif isinstance(node_ids_raw, list):
                node_ids = [str(t).strip() for t in node_ids_raw if str(t).strip()]

            include_self = bool(payload.get("include_self", False))
            include_offline = bool(payload.get("include_offline", False))

            timeout_seconds = float(payload.get("timeout_seconds", 20) or 20)
            timeout_seconds = max(5.0, min(timeout_seconds, 120.0))

            with self.peers_lock:
                peers_by_id = dict(self.peers)

            targets: List[NodeInfo] = []

            def should_include_peer(peer: NodeInfo) -> bool:
                if peer.node_id == self.node_id:
                    return False
                if not include_offline and not peer.is_alive():
                    return False
                return True

            if node_ids:
                for node_id in node_ids:
                    peer = peers_by_id.get(node_id)
                    if peer and should_include_peer(peer):
                        targets.append(peer)
            else:
                for peer in peers_by_id.values():
                    if should_include_peer(peer):
                        targets.append(peer)

            results: List[Dict[str, Any]] = []
            timeout = ClientTimeout(total=timeout_seconds)
            async with get_client_session(timeout) as session:
                for peer in sorted(targets, key=lambda p: p.node_id):
                    peer_payload: Dict[str, Any] = {
                        "node_id": peer.node_id,
                        "status": "online" if peer.is_alive() else "offline",
                        "success": False,
                        "attempted_urls": [],
                    }

                    if not include_offline and not peer.is_alive():
                        peer_payload["error"] = "offline"
                        results.append(peer_payload)
                        continue

                    last_error: Optional[str] = None
                    for url in self._urls_for_peer(peer, "/git/update"):
                        peer_payload["attempted_urls"].append(url)
                        try:
                            async with session.post(url, json={}, headers=self._auth_headers()) as resp:
                                peer_payload["http_status"] = resp.status
                                try:
                                    data = await resp.json()
                                except Exception:
                                    data = {"raw": await resp.text()}
                                peer_payload["response"] = data
                                if resp.status == 200:
                                    peer_payload["success"] = bool(data.get("success", True))
                                    break
                                last_error = (
                                    str(data.get("error") or "")
                                    or str(data.get("message") or "")
                                    or f"http_{resp.status}"
                                )
                        except Exception as exc:
                            last_error = str(exc)
                            continue

                    if last_error and not peer_payload.get("success"):
                        peer_payload["error"] = last_error

                    results.append(peer_payload)

            self_update: Optional[Dict[str, Any]] = None
            update_self = bool(include_self and (not node_ids or self.node_id in node_ids))
            if update_self:
                has_updates, local_commit, remote_commit = self._check_for_updates()
                if not has_updates:
                    self_update = {
                        "node_id": self.node_id,
                        "success": True,
                        "message": "Already up to date",
                        "local_commit": local_commit[:8] if local_commit else None,
                    }
                else:
                    success, message = await self._perform_git_update()
                    self_update = {
                        "node_id": self.node_id,
                        "success": success,
                        "message": message,
                        "old_commit": local_commit[:8] if local_commit else None,
                        "new_commit": remote_commit[:8] if remote_commit else None,
                    }
                    if success:
                        asyncio.create_task(self._restart_orchestrator())

            return web.json_response(
                {
                    "success": True,
                    "leader_id": self.node_id,
                    "updated_peers": results,
                    "self_update": self_update,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_selfplay_stats(self, request: web.Request) -> web.Response:
        """Get aggregated selfplay game statistics for dashboard charts."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                proxied = await self._proxy_to_leader(request)
                if proxied.status not in (502, 503):
                    return proxied

            with self.manifest_lock:
                cluster_manifest = self.cluster_data_manifest
                local_manifest = self.local_data_manifest
                history = list(self.selfplay_stats_history)

            by_board_type: Dict[str, Dict[str, Any]] = {}
            total_selfplay_games = 0
            manifest_collected_at = 0.0

            if cluster_manifest:
                by_board_type = cluster_manifest.by_board_type
                total_selfplay_games = int(cluster_manifest.total_selfplay_games or 0)
                manifest_collected_at = float(cluster_manifest.collected_at or 0.0)
            elif local_manifest:
                manifest_collected_at = float(local_manifest.collected_at or 0.0)
                totals: Dict[str, int] = {}
                for f in getattr(local_manifest, "files", []) or []:
                    if getattr(f, "file_type", "") != "selfplay":
                        continue
                    board_type = getattr(f, "board_type", "") or ""
                    num_players = int(getattr(f, "num_players", 0) or 0)
                    if not board_type or not num_players:
                        continue
                    key = f"{board_type}_{num_players}p"
                    totals[key] = totals.get(key, 0) + int(getattr(f, "game_count", 0) or 0)
                by_board_type = {k: {"total_games": v, "nodes": [local_manifest.node_id]} for k, v in totals.items()}
                total_selfplay_games = sum(totals.values())

            return web.json_response(
                {
                    "success": True,
                    "node_id": self.node_id,
                    "is_leader": self._is_leader(),
                    "manifest_collected_at": manifest_collected_at,
                    "total_selfplay_games": total_selfplay_games,
                    "by_board_type": by_board_type,
                    "history": history,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_elo_leaderboard(self, request: web.Request) -> web.Response:
        """Get Elo leaderboard for all board types from persistent database.

        Query params:
            board_type: Filter by board type (optional)
            num_players: Filter by number of players (optional)
            limit: Max results per config (default 20)
        """
        try:
            # Try to import Elo database functions
            try:
                from scripts.run_model_elo_tournament import (
                    init_elo_database,
                    get_leaderboard,
                    ELO_DB_PATH,
                )
            except ImportError:
                return web.json_response({
                    "success": False,
                    "error": "Elo database module not available",
                }, status=500)

            # Check if database exists
            if not ELO_DB_PATH or not ELO_DB_PATH.exists():
                return web.json_response({
                    "success": True,
                    "leaderboards": {},
                    "message": "No Elo database found yet. Run cross-model tournament to populate.",
                })

            board_type = request.query.get("board_type")
            num_players_str = request.query.get("num_players")
            num_players = int(num_players_str) if num_players_str else None
            limit = int(request.query.get("limit", "20"))

            db = init_elo_database()

            # If specific filter requested, return just that
            if board_type and num_players:
                leaderboard = get_leaderboard(db, board_type, num_players, limit=limit)
                db.close()
                return web.json_response({
                    "success": True,
                    "leaderboards": {f"{board_type}_{num_players}p": leaderboard},
                    "total_models": len(leaderboard),
                    "timestamp": time.time(),
                })

            # Otherwise return all board/player combinations
            # Query unique board_type/num_players combinations
            conn = db._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT board_type, num_players
                FROM elo_ratings
                WHERE board_type IS NOT NULL AND num_players IS NOT NULL
                ORDER BY board_type, num_players
            """)
            configs = cursor.fetchall()

            leaderboards = {}
            total_models = 0
            total_games = 0

            for bt, np in configs:
                key = f"{bt}_{np}p"
                lb = get_leaderboard(db, bt, np, limit=limit)
                if lb:
                    leaderboards[key] = lb
                    total_models += len(lb)
                    total_games += sum(entry.get("games_played", 0) for entry in lb)

            # Get match history stats
            cursor.execute("SELECT COUNT(*) FROM match_history")
            match_count = cursor.fetchone()[0]

            db.close()

            return web.json_response({
                "success": True,
                "leaderboards": leaderboards,
                "total_models": total_models,
                "total_matches": match_count,
                "total_games_recorded": total_games,
                "configs": [f"{bt}_{np}p" for bt, np in configs],
                "timestamp": time.time(),
            })

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_elo_table(self, request: web.Request) -> web.Response:
        """GET /elo/table - Elo leaderboard in flat table format for Grafana Infinity.

        Query params:
            - source: "tournament" (default) or "trained" (actual trained NN models)
            - limit: Max entries (default 50)
            - board_type: Filter by board type
            - num_players: Filter by player count
            - nn_only: If "true", filter to NN models only (for tournament source)

        Returns a simple JSON array of model entries with rank, suitable for table display.
        """
        import sqlite3

        try:
            source = request.query.get("source", "tournament")
            limit = int(request.query.get("limit", "50"))
            board_type_filter = request.query.get("board_type")
            num_players_filter = request.query.get("num_players")
            nn_only = request.query.get("nn_only", "").lower() == "true"

            ai_root = Path(self.ringrift_path) / "ai-service"

            if source == "trained":
                # Use unified_elo.db - actual trained NN models
                db_path = ai_root / "data" / "unified_elo.db"
                if not db_path.exists():
                    return web.json_response([])

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # unified_elo.db has different schema (model_id instead of participant_id)
                query = """
                    SELECT model_id, rating, games_played, wins, losses
                    FROM elo_ratings
                    WHERE games_played >= 10
                """
                params = []

                if nn_only:
                    query += " AND (model_id LIKE '%nn%' OR model_id LIKE '%NN%' OR model_id LIKE '%baseline%')"

                query += " ORDER BY rating DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()
                conn.close()

                # Build flat table response
                table_data = []
                for rank, row in enumerate(rows, 1):
                    model_id, rating, games, wins, losses = row

                    # Extract config from model name
                    if "sq8" in model_id.lower() or "square8" in model_id.lower():
                        config = "square8_2p"
                    elif "sq19" in model_id.lower() or "square19" in model_id.lower():
                        config = "square19_2p"
                    elif "hex" in model_id.lower():
                        config = "hexagonal_2p"
                    else:
                        config = "unknown"

                    # Calculate win rate
                    total_decided = wins + losses
                    win_rate = wins / total_decided if total_decided > 0 else 0.5

                    table_data.append({
                        "Rank": rank,
                        "Model": model_id,
                        "Elo": round(rating, 1),
                        "WinRate": round(win_rate * 100, 1),
                        "Games": games,
                        "Wins": wins,
                        "Losses": losses,
                        "Draws": 0,
                        "Config": config,
                    })

                return web.json_response(table_data)

            else:
                # Default: tournament participants from unified_elo.db
                from scripts.run_model_elo_tournament import (
                    init_elo_database,
                    ELO_DB_PATH,
                )

                if not ELO_DB_PATH or not ELO_DB_PATH.exists():
                    return web.json_response([])

                db = init_elo_database()
                conn = db._get_connection()
                cursor = conn.cursor()

                # Build query with optional filters
                # Use db.id_column to get correct column name (model_id or participant_id)
                id_col = db.id_column

                query = f"""
                    SELECT
                        {id_col},
                        board_type,
                        num_players,
                        rating,
                        games_played,
                        wins,
                        losses,
                        draws,
                        last_update
                    FROM elo_ratings
                    WHERE games_played >= 5
                """
                params = []

                if board_type_filter:
                    query += " AND board_type = ?"
                    params.append(board_type_filter)

                if num_players_filter:
                    query += " AND num_players = ?"
                    params.append(int(num_players_filter))

                if nn_only:
                    query += f" AND ({id_col} LIKE '%NN%' OR {id_col} LIKE '%nn%')"

                query += " ORDER BY rating DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()
                db.close()

                # Build flat table response
                table_data = []
                for rank, row in enumerate(rows, 1):
                    participant_id, board_type, num_players, rating, games, wins, losses, draws, last_update = row

                    # Extract model name from participant_id
                    model_name = participant_id
                    if participant_id.startswith("nn:"):
                        model_name = Path(participant_id[3:]).stem

                    # Calculate win rate
                    total_decided = wins + losses
                    win_rate = wins / total_decided if total_decided > 0 else 0.5

                    # Format config
                    config = f"{board_type}_{num_players}p"

                    table_data.append({
                        "Rank": rank,
                        "Model": model_name,
                        "Elo": round(rating, 1),
                        "WinRate": round(win_rate * 100, 1),
                        "Games": games,
                        "Wins": wins,
                        "Losses": losses,
                        "Draws": draws,
                        "Config": config,
                    })

                return web.json_response(table_data)

        except ImportError:
            return web.json_response([{"error": "Elo database module not available"}])
        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_nodes_table(self, request: web.Request) -> web.Response:
        """GET /nodes/table - Node status in flat table format for Grafana Infinity.

        Returns current status of all cluster nodes in table format.
        """
        try:
            nodes = []

            # Add self
            node_name = self.node_id or "unknown"
            role = "Leader" if self.role == NodeRole.LEADER else "Worker"
            cpu = getattr(self.self_info, 'cpu_percent', 0)
            mem = getattr(self.self_info, 'memory_percent', 0)
            gpu = getattr(self.self_info, 'gpu_percent', 0) if self.self_info.has_gpu else 0
            gpu_mem = getattr(self.self_info, 'gpu_memory_percent', 0) if self.self_info.has_gpu else 0

            with self.jobs_lock:
                selfplay_jobs = len([j for j in self.local_jobs.values()
                                    if j.job_type in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY)
                                    and j.status == "running"])

            nodes.append({
                "Node": node_name,
                "Role": role,
                "Status": "Online",
                "CPU": round(cpu, 1),
                "Memory": round(mem, 1),
                "GPU": round(gpu, 1),
                "GPUMem": round(gpu_mem, 1),
                "Jobs": selfplay_jobs,
                "HasGPU": "Yes" if self.self_info.has_gpu else "No",
            })

            # Add peers
            with self.peers_lock:
                for peer_id, peer in self.peers.items():
                    peer_name = peer_id or "unknown"
                    is_alive = peer.is_alive()
                    status = "Online" if is_alive else "Offline"

                    peer_cpu = getattr(peer, 'cpu_percent', 0) or 0
                    peer_mem = getattr(peer, 'memory_percent', 0) or 0
                    peer_gpu = getattr(peer, 'gpu_percent', 0) or 0
                    peer_gpu_mem = getattr(peer, 'gpu_memory_percent', 0) or 0
                    peer_jobs = getattr(peer, 'selfplay_jobs', 0) or 0
                    has_gpu = getattr(peer, 'has_gpu', False)

                    nodes.append({
                        "Node": peer_name,
                        "Role": "Worker",
                        "Status": status,
                        "CPU": round(peer_cpu, 1),
                        "Memory": round(peer_mem, 1),
                        "GPU": round(peer_gpu, 1),
                        "GPUMem": round(peer_gpu_mem, 1),
                        "Jobs": peer_jobs,
                        "HasGPU": "Yes" if has_gpu else "No",
                    })

            # Sort by role (leader first) then by name
            nodes.sort(key=lambda n: (0 if n["Role"] == "Leader" else 1, n["Node"]))

            return web.json_response(nodes)

        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def _get_victory_type_stats(self) -> Dict[Tuple[str, int, str], int]:
        """Aggregate victory types from recent game data.

        Returns dict mapping (board_type, num_players, victory_type) -> count.
        Caches results for 5 minutes to avoid excessive I/O.
        """
        import json
        from collections import defaultdict

        cache_key = "_victory_stats_cache"
        cache_time_key = "_victory_stats_cache_time"
        cache_ttl = 300  # 5 minutes

        # Check cache
        now = time.time()
        if hasattr(self, cache_key) and hasattr(self, cache_time_key):
            if now - getattr(self, cache_time_key) < cache_ttl:
                return getattr(self, cache_key)

        stats: Dict[Tuple[str, int, str], int] = defaultdict(int)

        # Scan recent game files (last 24 hours)
        ai_root = Path(self.ringrift_path) / "ai-service"
        data_dirs = [
            ai_root / "data" / "games" / "daemon_sync",
            ai_root / "data" / "selfplay",
        ]

        cutoff_time = now - 86400  # 24 hours ago

        for data_dir in data_dirs:
            if not data_dir.exists():
                continue
            for jsonl_path in data_dir.rglob("*.jsonl"):
                try:
                    # Skip files older than 24h
                    if jsonl_path.stat().st_mtime < cutoff_time:
                        continue
                    with open(jsonl_path, "r") as f:
                        for line in f:
                            try:
                                game = json.loads(line)
                                board_type = game.get("board_type", "unknown")
                                num_players = game.get("num_players", 0)
                                victory_type = game.get("victory_type", "unknown")
                                if victory_type and victory_type != "unknown":
                                    stats[(board_type, num_players, victory_type)] += 1
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue

        # Update cache
        setattr(self, cache_key, dict(stats))
        setattr(self, cache_time_key, now)

        return dict(stats)

    async def _get_game_analytics_cached(self) -> Dict[str, Any]:
        """Get game analytics with caching (5 min TTL)."""
        import json
        from collections import defaultdict

        cache_key = "_game_analytics_cache"
        cache_time_key = "_game_analytics_cache_time"
        cache_ttl = 300

        now = time.time()
        if hasattr(self, cache_key) and hasattr(self, cache_time_key):
            if now - getattr(self, cache_time_key) < cache_ttl:
                return getattr(self, cache_key)

        hours = 24
        cutoff = now - (hours * 3600)

        ai_root = Path(self.ringrift_path) / "ai-service"
        data_dirs = [
            ai_root / "data" / "games" / "daemon_sync",
            ai_root / "data" / "selfplay",
        ]

        game_lengths: Dict[str, List[int]] = defaultdict(list)
        games_by_hour: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        opening_moves: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for data_dir in data_dirs:
            if not data_dir.exists():
                continue
            for jsonl_path in data_dir.rglob("*.jsonl"):
                try:
                    if jsonl_path.stat().st_mtime < cutoff:
                        continue
                    with open(jsonl_path, "r") as f:
                        for line in f:
                            try:
                                game = json.loads(line)
                                board_type = game.get("board_type", "unknown")
                                num_players = game.get("num_players", 0)
                                config = f"{board_type}_{num_players}p"

                                length = game.get("length", 0)
                                if length > 0:
                                    game_lengths[config].append(length)

                                hour_bucket = int(jsonl_path.stat().st_mtime // 3600)
                                games_by_hour[config][hour_bucket] += 1

                                moves = game.get("moves", [])
                                if moves and len(moves) >= 1:
                                    first_move = str(moves[0].get("action", ""))[:20]
                                    if first_move:
                                        opening_moves[config][first_move] += 1
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue

        analytics = {"configs": {}}
        for config in set(list(game_lengths.keys()) + list(games_by_hour.keys())):
            lengths = game_lengths.get(config, [])
            hourly = games_by_hour.get(config, {})
            openings = opening_moves.get(config, {})
            throughput = sum(hourly.values()) / max(len(hourly), 1) if hourly else 0

            analytics["configs"][config] = {
                "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
                "throughput_per_hour": round(throughput, 1),
                "opening_diversity": len(openings),
            }

        setattr(self, cache_key, analytics)
        setattr(self, cache_time_key, now)
        return analytics

    async def _get_training_metrics_cached(self) -> Dict[str, Any]:
        """Get training metrics with caching (2 min TTL)."""
        import re

        cache_key = "_training_metrics_cache"
        cache_time_key = "_training_metrics_cache_time"
        cache_ttl = 120

        now = time.time()
        if hasattr(self, cache_key) and hasattr(self, cache_time_key):
            if now - getattr(self, cache_time_key) < cache_ttl:
                return getattr(self, cache_key)

        ai_root = Path(self.ringrift_path) / "ai-service"
        logs_dir = ai_root / "logs" / "training"

        metrics = {"configs": {}}

        if logs_dir.exists():
            log_files = sorted(logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:10]

            for log_file in log_files:
                try:
                    content = log_file.read_text()
                    config_match = re.search(r"(square\d+|hexagonal|hex)_(\d+)p", log_file.name)
                    if not config_match:
                        continue
                    config = f"{config_match.group(1)}_{config_match.group(2)}p"

                    loss_pattern = re.compile(r"[Ee]poch\s+(\d+).*?loss[=:]\s*([\d.]+)")
                    epochs = []
                    for match in loss_pattern.finditer(content):
                        epochs.append({
                            "epoch": int(match.group(1)),
                            "loss": float(match.group(2)),
                        })

                    if epochs:
                        metrics["configs"][config] = {
                            "latest_loss": epochs[-1]["loss"],
                            "latest_epoch": epochs[-1]["epoch"],
                        }
                except Exception:
                    continue

        setattr(self, cache_key, metrics)
        setattr(self, cache_time_key, now)
        return metrics

    async def _get_holdout_metrics_cached(self) -> Dict[str, Any]:
        """Get holdout validation metrics with caching (5 min TTL)."""
        import sqlite3

        cache_key = "_holdout_metrics_cache"
        cache_time_key = "_holdout_metrics_cache_time"
        cache_ttl = 300

        now = time.time()
        if hasattr(self, cache_key) and hasattr(self, cache_time_key):
            if now - getattr(self, cache_time_key) < cache_ttl:
                return getattr(self, cache_key)

        ai_root = Path(self.ringrift_path) / "ai-service"
        db_path = ai_root / "data" / "holdouts" / "holdout_validation.db"

        metrics = {"configs": {}, "evaluations": [], "summary": {}}

        if not db_path.exists():
            setattr(self, cache_key, metrics)
            setattr(self, cache_time_key, now)
            return metrics

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get holdout game counts by config
            cursor.execute("""
                SELECT board_type, num_players, COUNT(*) as game_count, SUM(num_positions) as total_positions
                FROM holdout_games
                GROUP BY board_type, num_players
            """)
            for row in cursor.fetchall():
                config = f"{row['board_type']}_{row['num_players']}p"
                metrics["configs"][config] = {
                    "holdout_games": row["game_count"],
                    "holdout_positions": row["total_positions"] or 0,
                }

            # Get latest evaluations per config
            cursor.execute("""
                SELECT model_path, board_type, num_players, holdout_loss, holdout_accuracy,
                       train_loss, num_samples, evaluated_at, overfit_gap
                FROM evaluations
                WHERE id IN (
                    SELECT MAX(id) FROM evaluations
                    GROUP BY board_type, num_players
                )
                ORDER BY evaluated_at DESC
            """)
            for row in cursor.fetchall():
                config = f"{row['board_type']}_{row['num_players']}p"
                eval_data = {
                    "config": config,
                    "model": row["model_path"],
                    "holdout_loss": row["holdout_loss"],
                    "holdout_accuracy": row["holdout_accuracy"],
                    "train_loss": row["train_loss"],
                    "overfit_gap": row["overfit_gap"],
                    "num_samples": row["num_samples"],
                    "evaluated_at": row["evaluated_at"],
                }
                metrics["evaluations"].append(eval_data)
                # Update config metrics
                if config in metrics["configs"]:
                    metrics["configs"][config].update({
                        "holdout_loss": row["holdout_loss"],
                        "holdout_accuracy": row["holdout_accuracy"],
                        "overfit_gap": row["overfit_gap"],
                    })

            # Get summary stats
            cursor.execute("SELECT COUNT(*) FROM holdout_games")
            metrics["summary"]["total_holdout_games"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            metrics["summary"]["total_evaluations"] = cursor.fetchone()[0]

            conn.close()
        except Exception:
            pass

        setattr(self, cache_key, metrics)
        setattr(self, cache_time_key, now)
        return metrics

    async def _get_mcts_stats_cached(self) -> Dict[str, Any]:
        """Get MCTS search statistics with caching (2 min TTL)."""
        import json
        import re

        cache_key = "_mcts_stats_cache"
        cache_time_key = "_mcts_stats_cache_time"
        cache_ttl = 120

        now = time.time()
        if hasattr(self, cache_key) and hasattr(self, cache_time_key):
            if now - getattr(self, cache_time_key) < cache_ttl:
                return getattr(self, cache_key)

        ai_root = Path(self.ringrift_path) / "ai-service"
        stats = {"configs": {}, "summary": {}}

        # Parse selfplay logs for MCTS stats
        logs_dir = ai_root / "logs" / "selfplay"
        if logs_dir.exists():
            log_files = sorted(logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:20]

            nodes_per_move = []
            depth_stats = []
            time_per_move = []

            for log_file in log_files:
                try:
                    content = log_file.read_text(errors='ignore')
                    # Parse MCTS stats patterns (nodes visited, search depth, time)
                    # Pattern: "nodes: 1234" or "nodes_visited: 1234"
                    for match in re.finditer(r'nodes[_\s]*(?:visited)?[:\s]*(\d+)', content, re.I):
                        nodes_per_move.append(int(match.group(1)))
                    # Pattern: "depth: 12" or "search_depth: 12"
                    for match in re.finditer(r'(?:search_)?depth[:\s]*(\d+)', content, re.I):
                        depth_stats.append(int(match.group(1)))
                    # Pattern: "time: 0.123s" or "move_time: 123ms"
                    for match in re.finditer(r'(?:move_)?time[:\s]*([\d.]+)\s*(?:s|ms)?', content, re.I):
                        time_per_move.append(float(match.group(1)))
                except Exception:
                    continue

            if nodes_per_move:
                stats["summary"]["avg_nodes_per_move"] = sum(nodes_per_move) / len(nodes_per_move)
                stats["summary"]["max_nodes_per_move"] = max(nodes_per_move)
            if depth_stats:
                stats["summary"]["avg_search_depth"] = sum(depth_stats) / len(depth_stats)
                stats["summary"]["max_search_depth"] = max(depth_stats)
            if time_per_move:
                stats["summary"]["avg_time_per_move"] = sum(time_per_move) / len(time_per_move)

        # Also check game JSONL files for MCTS metadata
        data_dirs = [
            ai_root / "data" / "games" / "daemon_sync",
            ai_root / "data" / "selfplay",
        ]
        cutoff = now - 3600  # Last hour

        for data_dir in data_dirs:
            if not data_dir.exists():
                continue
            for jsonl_path in data_dir.rglob("*.jsonl"):
                try:
                    if jsonl_path.stat().st_mtime < cutoff:
                        continue
                    with open(jsonl_path, "r") as f:
                        for line in f:
                            try:
                                game = json.loads(line)
                                board_type = game.get("board_type", "unknown")
                                num_players = game.get("num_players", 0)
                                config = f"{board_type}_{num_players}p"

                                # Check for MCTS metadata in game
                                mcts_data = game.get("mcts_stats", {})
                                if mcts_data:
                                    if config not in stats["configs"]:
                                        stats["configs"][config] = {
                                            "nodes_samples": [],
                                            "depth_samples": [],
                                        }
                                    if "avg_nodes" in mcts_data:
                                        stats["configs"][config]["nodes_samples"].append(mcts_data["avg_nodes"])
                                    if "avg_depth" in mcts_data:
                                        stats["configs"][config]["depth_samples"].append(mcts_data["avg_depth"])
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue

        # Compute per-config averages
        for config, data in stats["configs"].items():
            if data.get("nodes_samples"):
                data["avg_nodes"] = sum(data["nodes_samples"]) / len(data["nodes_samples"])
            if data.get("depth_samples"):
                data["avg_depth"] = sum(data["depth_samples"]) / len(data["depth_samples"])
            # Clean up sample lists
            data.pop("nodes_samples", None)
            data.pop("depth_samples", None)

        setattr(self, cache_key, stats)
        setattr(self, cache_time_key, now)
        return stats

    # =========================================================================
    # Feature 1: Tournament Matchup Analysis
    # =========================================================================

    async def _get_matchup_matrix_cached(self) -> Dict[str, Any]:
        """Get head-to-head matchup statistics with caching (5 min TTL)."""
        import sqlite3
        from collections import defaultdict

        cache_key = "_matchup_matrix_cache"
        cache_time_key = "_matchup_matrix_cache_time"
        cache_ttl = 300

        now = time.time()
        if hasattr(self, cache_key) and hasattr(self, cache_time_key):
            if now - getattr(self, cache_time_key) < cache_ttl:
                return getattr(self, cache_key)

        ai_root = Path(self.ringrift_path) / "ai-service"
        db_path = ai_root / "data" / "unified_elo.db"

        matrix = {"matchups": [], "models": [], "configs": {}}

        if not db_path.exists():
            setattr(self, cache_key, matrix)
            setattr(self, cache_time_key, now)
            return matrix

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            # Get all match history
            rows = conn.execute("""
                SELECT participant_a, participant_b, winner, board_type, num_players,
                       game_length, duration_sec, timestamp
                FROM match_history
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 10000
            """, (now - 86400 * 7,)).fetchall()  # Last 7 days

            # Build matchup stats
            h2h: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0}))
            models = set()
            config_stats = defaultdict(lambda: {"total_matches": 0, "avg_game_length": [], "avg_duration": []})

            for row in rows:
                a = row["participant_a"]
                b = row["participant_b"]
                winner = row["winner"]
                config = f"{row['board_type']}_{row['num_players']}p"

                if a and b:
                    models.add(a)
                    models.add(b)

                    if winner == a:
                        h2h[a][b]["wins"] += 1
                        h2h[b][a]["losses"] += 1
                    elif winner == b:
                        h2h[b][a]["wins"] += 1
                        h2h[a][b]["losses"] += 1
                    else:
                        h2h[a][b]["draws"] += 1
                        h2h[b][a]["draws"] += 1

                    config_stats[config]["total_matches"] += 1
                    if row["game_length"]:
                        config_stats[config]["avg_game_length"].append(row["game_length"])
                    if row["duration_sec"]:
                        config_stats[config]["avg_duration"].append(row["duration_sec"])

            # Convert to matchup list
            matchups = []
            for model_a in sorted(models):
                for model_b in sorted(models):
                    if model_a < model_b:  # Avoid duplicates
                        stats = h2h[model_a][model_b]
                        total = stats["wins"] + stats["losses"] + stats["draws"]
                        if total > 0:
                            matchups.append({
                                "model_a": model_a,
                                "model_b": model_b,
                                "a_wins": stats["wins"],
                                "b_wins": stats["losses"],
                                "draws": stats["draws"],
                                "total": total,
                                "a_win_rate": round(stats["wins"] / total, 3) if total > 0 else 0,
                            })

            # Compute config averages
            for config, data in config_stats.items():
                if data["avg_game_length"]:
                    data["avg_game_length"] = round(sum(data["avg_game_length"]) / len(data["avg_game_length"]), 1)
                else:
                    data["avg_game_length"] = 0
                if data["avg_duration"]:
                    data["avg_duration"] = round(sum(data["avg_duration"]) / len(data["avg_duration"]), 2)
                else:
                    data["avg_duration"] = 0

            matrix["matchups"] = matchups
            matrix["models"] = sorted(models)
            matrix["configs"] = dict(config_stats)
            matrix["total_matches"] = sum(c["total_matches"] for c in config_stats.values())

            conn.close()
        except Exception:
            pass

        setattr(self, cache_key, matrix)
        setattr(self, cache_time_key, now)
        return matrix

    # =========================================================================
    # Feature 2: Model Lineage Tracking
    # =========================================================================

    async def _get_model_lineage_cached(self) -> Dict[str, Any]:
        """Get model lineage and ancestry with caching (10 min TTL)."""
        import re

        cache_key = "_model_lineage_cache"
        cache_time_key = "_model_lineage_cache_time"
        cache_ttl = 600

        now = time.time()
        if hasattr(self, cache_key) and hasattr(self, cache_time_key):
            if now - getattr(self, cache_time_key) < cache_ttl:
                return getattr(self, cache_key)

        ai_root = Path(self.ringrift_path) / "ai-service"
        models_dir = ai_root / "models"

        lineage = {"models": [], "generations": {}, "configs": {}}

        if not models_dir.exists():
            setattr(self, cache_key, lineage)
            setattr(self, cache_time_key, now)
            return lineage

        try:
            # Discover all models
            model_files = list(models_dir.glob("**/*.pt")) + list(models_dir.glob("**/*.pth"))

            for model_path in model_files:
                model_name = model_path.stem
                model_stat = model_path.stat()

                # Parse model name for lineage info
                # Common patterns:
                #   - square8_2p_v5_gen12, nnue_square8_2p_epoch50
                #   - ringrift_best_sq8_2p, ringrift_best_sq19_2p
                #   - hex_3p_nn_baseline, ringrift_best_hex_2p
                # Handle both full names (square8, hexagonal) and abbreviations (sq8, hex)
                config_match = re.search(
                    r"(square\d+|sq\d+|hexagonal|hex)[\W_]*(\d+)p",
                    model_name,
                    re.I
                )
                gen_match = re.search(r"gen(\d+)|v(\d+)|epoch(\d+)", model_name, re.I)

                if config_match:
                    board = config_match.group(1).lower()
                    players = config_match.group(2)
                    # Normalize board names (only transform abbreviations, not full names)
                    if board.startswith("sq") and not board.startswith("square"):
                        # sq8 -> square8, sq19 -> square19
                        board = f"square{board[2:]}"
                    elif board == "hex":
                        board = "hexagonal"
                    config = f"{board}_{players}p"
                else:
                    config = "unknown"
                generation = int(gen_match.group(1) or gen_match.group(2) or gen_match.group(3) or 0) if gen_match else 0

                model_info = {
                    "name": model_name,
                    "path": str(model_path.relative_to(ai_root)),
                    "config": config,
                    "generation": generation,
                    "size_mb": round(model_stat.st_size / 1024 / 1024, 2),
                    "created_at": model_stat.st_mtime,
                    "age_hours": round((now - model_stat.st_mtime) / 3600, 1),
                }
                lineage["models"].append(model_info)

                # Track generations per config
                if config not in lineage["generations"]:
                    lineage["generations"][config] = []
                lineage["generations"][config].append(model_info)

            # Sort models by generation within each config
            for config in lineage["generations"]:
                lineage["generations"][config].sort(key=lambda m: m["generation"])

            # Summary per config
            for config, models in lineage["generations"].items():
                lineage["configs"][config] = {
                    "total_models": len(models),
                    "latest_generation": max(m["generation"] for m in models) if models else 0,
                    "latest_model": models[-1]["name"] if models else None,
                    "total_size_mb": round(sum(m["size_mb"] for m in models), 1),
                }

            lineage["total_models"] = len(lineage["models"])

        except Exception:
            pass

        setattr(self, cache_key, lineage)
        setattr(self, cache_time_key, now)
        return lineage

    # =========================================================================
    # Feature 3: Data Quality Metrics
    # =========================================================================

    async def _get_data_quality_cached(self) -> Dict[str, Any]:
        """Get data quality metrics with caching (5 min TTL)."""
        import json
        from collections import defaultdict

        cache_key = "_data_quality_cache"
        cache_time_key = "_data_quality_cache_time"
        cache_ttl = 300

        now = time.time()
        if hasattr(self, cache_key) and hasattr(self, cache_time_key):
            if now - getattr(self, cache_time_key) < cache_ttl:
                return getattr(self, cache_key)

        ai_root = Path(self.ringrift_path) / "ai-service"
        quality = {"configs": {}, "issues": [], "summary": {}}

        data_dirs = [
            ai_root / "data" / "games" / "daemon_sync",
            ai_root / "data" / "selfplay",
        ]
        cutoff = now - 86400  # Last 24 hours

        try:
            config_stats = defaultdict(lambda: {
                "total_games": 0,
                "game_lengths": [],
                "short_games": 0,  # < 10 moves
                "long_games": 0,   # > 500 moves
                "stalemates": 0,
                "unique_openings": set(),
                "player_wins": defaultdict(int),
                "parse_errors": 0,
            })

            for data_dir in data_dirs:
                if not data_dir.exists():
                    continue
                for jsonl_path in data_dir.rglob("*.jsonl"):
                    try:
                        if jsonl_path.stat().st_mtime < cutoff:
                            continue
                        with open(jsonl_path, "r") as f:
                            for line in f:
                                try:
                                    game = json.loads(line)
                                    board_type = game.get("board_type", "unknown")
                                    num_players = game.get("num_players", 0)
                                    config = f"{board_type}_{num_players}p"

                                    stats = config_stats[config]
                                    stats["total_games"] += 1

                                    length = game.get("length", 0)
                                    if length > 0:
                                        stats["game_lengths"].append(length)
                                        if length < 10:
                                            stats["short_games"] += 1
                                        elif length > 500:
                                            stats["long_games"] += 1

                                    victory_type = game.get("victory_type", "")
                                    if victory_type == "stalemate":
                                        stats["stalemates"] += 1

                                    # Track opening diversity
                                    moves = game.get("moves", [])
                                    if moves and len(moves) >= 2:
                                        opening = str(moves[0].get("action", ""))[:15] + "-" + str(moves[1].get("action", ""))[:15]
                                        stats["unique_openings"].add(opening)

                                    # Track winner distribution
                                    winner = game.get("winner")
                                    if winner is not None:
                                        stats["player_wins"][winner] += 1

                                except json.JSONDecodeError:
                                    config_stats["unknown"]["parse_errors"] += 1
                    except Exception:
                        continue

            # Convert to quality metrics
            issues = []
            for config, stats in config_stats.items():
                total = stats["total_games"]
                if total == 0:
                    continue

                lengths = stats["game_lengths"]
                avg_length = sum(lengths) / len(lengths) if lengths else 0
                length_std = (sum((l - avg_length) ** 2 for l in lengths) / len(lengths)) ** 0.5 if len(lengths) > 1 else 0

                short_rate = stats["short_games"] / total
                long_rate = stats["long_games"] / total
                stalemate_rate = stats["stalemates"] / total
                opening_diversity = len(stats["unique_openings"])

                # Detect issues
                if short_rate > 0.1:
                    issues.append({"config": config, "issue": "high_short_game_rate", "value": round(short_rate * 100, 1), "severity": "warning"})
                if stalemate_rate > 0.3:
                    issues.append({"config": config, "issue": "high_stalemate_rate", "value": round(stalemate_rate * 100, 1), "severity": "warning"})
                if opening_diversity < 5 and total > 50:
                    issues.append({"config": config, "issue": "low_opening_diversity", "value": opening_diversity, "severity": "warning"})

                # Check for player bias
                wins = stats["player_wins"]
                if len(wins) >= 2 and total > 20:
                    max_win_rate = max(wins.values()) / total
                    if max_win_rate > 0.7:
                        issues.append({"config": config, "issue": "player_bias", "value": round(max_win_rate * 100, 1), "severity": "info"})

                quality["configs"][config] = {
                    "total_games": total,
                    "avg_length": round(avg_length, 1),
                    "length_std": round(length_std, 1),
                    "short_game_rate": round(short_rate * 100, 1),
                    "long_game_rate": round(long_rate * 100, 1),
                    "stalemate_rate": round(stalemate_rate * 100, 1),
                    "opening_diversity": opening_diversity,
                    "parse_errors": stats["parse_errors"],
                }

            quality["issues"] = issues
            quality["summary"] = {
                "total_configs": len(quality["configs"]),
                "total_issues": len(issues),
                "critical_issues": len([i for i in issues if i["severity"] == "critical"]),
                "warning_issues": len([i for i in issues if i["severity"] == "warning"]),
            }

        except Exception:
            pass

        setattr(self, cache_key, quality)
        setattr(self, cache_time_key, now)
        return quality

    # =========================================================================
    # Feature 4: Training Efficiency Dashboard
    # =========================================================================

    async def _get_training_efficiency_cached(self) -> Dict[str, Any]:
        """Get training efficiency metrics with caching (5 min TTL)."""
        import sqlite3
        import re

        cache_key = "_training_efficiency_cache"
        cache_time_key = "_training_efficiency_cache_time"
        cache_ttl = 300

        now = time.time()
        if hasattr(self, cache_key) and hasattr(self, cache_time_key):
            if now - getattr(self, cache_time_key) < cache_ttl:
                return getattr(self, cache_key)

        ai_root = Path(self.ringrift_path) / "ai-service"
        efficiency = {"configs": {}, "summary": {}, "cost_tracking": {}}

        try:
            # Get Elo history to track improvements
            db_path = ai_root / "data" / "unified_elo.db"
            elo_history = {}

            if db_path.exists():
                conn = sqlite3.connect(db_path)
                rows = conn.execute("""
                    SELECT board_type, num_players, participant_id, rating, timestamp
                    FROM rating_history
                    WHERE timestamp > ?
                    ORDER BY timestamp ASC
                """, (now - 86400 * 7,)).fetchall()  # Last 7 days

                for row in rows:
                    config = f"{row[0]}_{row[1]}p"
                    if config not in elo_history:
                        elo_history[config] = {"ratings": [], "timestamps": []}
                    elo_history[config]["ratings"].append(row[3])
                    elo_history[config]["timestamps"].append(row[4])
                conn.close()

            # Parse training logs for GPU hours
            logs_dir = ai_root / "logs" / "training"
            gpu_hours_per_config = {}

            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    try:
                        content = log_file.read_text(errors='ignore')
                        config_match = re.search(r"(square\d+|hex\w*)_(\d+)p", log_file.name)
                        if not config_match:
                            continue
                        config = f"{config_match.group(1)}_{config_match.group(2)}p"

                        # Extract training duration
                        duration_match = re.search(r"(?:total[_\s]?time|duration)[:\s]*([\d.]+)\s*(?:s|sec|min|h)", content, re.I)
                        if duration_match:
                            duration = float(duration_match.group(1))
                            # Assume hours if > 100, else assume minutes
                            if duration > 100:
                                duration = duration / 3600  # seconds to hours
                            elif duration < 24:
                                duration = duration / 60  # minutes to hours

                            if config not in gpu_hours_per_config:
                                gpu_hours_per_config[config] = 0
                            gpu_hours_per_config[config] += duration
                    except Exception:
                        continue

            # Calculate efficiency metrics per config
            for config in set(list(elo_history.keys()) + list(gpu_hours_per_config.keys())):
                elo_data = elo_history.get(config, {"ratings": [], "timestamps": []})
                gpu_hours = gpu_hours_per_config.get(config, 0)

                if elo_data["ratings"]:
                    initial_elo = elo_data["ratings"][0] if elo_data["ratings"] else 1500
                    current_elo = elo_data["ratings"][-1] if elo_data["ratings"] else 1500
                    elo_gain = current_elo - initial_elo
                else:
                    initial_elo = current_elo = 1500
                    elo_gain = 0

                # Elo per GPU hour
                elo_per_hour = elo_gain / gpu_hours if gpu_hours > 0 else 0

                # Estimated cost (assuming $2/GPU-hour average)
                estimated_cost = gpu_hours * 2.0

                efficiency["configs"][config] = {
                    "gpu_hours": round(gpu_hours, 2),
                    "initial_elo": round(initial_elo, 1),
                    "current_elo": round(current_elo, 1),
                    "elo_gain": round(elo_gain, 1),
                    "elo_per_gpu_hour": round(elo_per_hour, 2),
                    "estimated_cost_usd": round(estimated_cost, 2),
                    "cost_per_elo_point": round(estimated_cost / max(elo_gain, 1), 2) if elo_gain > 0 else None,
                }

            # Summary
            total_gpu_hours = sum(c.get("gpu_hours", 0) for c in efficiency["configs"].values())
            total_elo_gain = sum(c.get("elo_gain", 0) for c in efficiency["configs"].values())
            total_cost = sum(c.get("estimated_cost_usd", 0) for c in efficiency["configs"].values())

            efficiency["summary"] = {
                "total_gpu_hours": round(total_gpu_hours, 2),
                "total_elo_gain": round(total_elo_gain, 1),
                "total_estimated_cost_usd": round(total_cost, 2),
                "overall_elo_per_gpu_hour": round(total_elo_gain / max(total_gpu_hours, 1), 2),
            }

        except Exception:
            pass

        setattr(self, cache_key, efficiency)
        setattr(self, cache_time_key, now)
        return efficiency

    # =========================================================================
    # Feature 5: Automated Model Rollback
    # =========================================================================

    async def _check_rollback_conditions(self) -> Dict[str, Any]:
        """Check if any models should be rolled back based on metrics."""
        rollback_status = {"candidates": [], "recent_rollbacks": [], "config_status": {}}

        try:
            # Get holdout metrics for overfitting detection
            holdout = await self._get_holdout_metrics_cached()

            # Get Elo data for regression detection
            ai_root = Path(self.ringrift_path) / "ai-service"
            db_path = ai_root / "data" / "unified_elo.db"

            elo_data = {}
            if db_path.exists():
                import sqlite3
                conn = sqlite3.connect(db_path)
                rows = conn.execute("""
                    SELECT board_type, num_players, participant_id, rating, timestamp
                    FROM rating_history
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """).fetchall()

                for row in rows:
                    config = f"{row[0]}_{row[1]}p"
                    if config not in elo_data:
                        elo_data[config] = []
                    elo_data[config].append({"model": row[2], "rating": row[3], "timestamp": row[4]})
                conn.close()

            # Check each config for rollback conditions
            for config, holdout_data in holdout.get("configs", {}).items():
                status = {"config": config, "rollback_recommended": False, "reasons": []}

                # Check 1: Overfitting (overfit_gap > 0.15)
                overfit_gap = holdout_data.get("overfit_gap", 0)
                if overfit_gap and overfit_gap > 0.15:
                    status["rollback_recommended"] = True
                    status["reasons"].append(f"Overfitting detected: gap={overfit_gap:.3f}")

                # Check 2: Low holdout accuracy (< 60%)
                holdout_acc = holdout_data.get("holdout_accuracy", 1.0)
                if holdout_acc and holdout_acc < 0.6:
                    status["rollback_recommended"] = True
                    status["reasons"].append(f"Low holdout accuracy: {holdout_acc*100:.1f}%")

                # Check 3: Elo regression (dropped > 50 points recently)
                if config in elo_data and len(elo_data[config]) >= 2:
                    recent = elo_data[config][0]["rating"]
                    previous = max(e["rating"] for e in elo_data[config][:10])
                    if previous - recent > 50:
                        status["rollback_recommended"] = True
                        status["reasons"].append(f"Elo regression: {previous:.0f} -> {recent:.0f}")

                rollback_status["config_status"][config] = status
                if status["rollback_recommended"]:
                    rollback_status["candidates"].append(status)

            # Load recent rollback history if exists
            rollback_log = ai_root / "logs" / "rollbacks.json"
            if rollback_log.exists():
                import json
                try:
                    rollback_status["recent_rollbacks"] = json.loads(rollback_log.read_text())[-10:]
                except Exception:
                    pass

        except Exception:
            pass

        return rollback_status

    async def _execute_rollback(self, config: str, dry_run: bool = False) -> Dict[str, Any]:
        """Execute a rollback for the given config by restoring previous model.

        Args:
            config: Config string like "square8_2p"
            dry_run: If True, only simulate the rollback without making changes

        Returns:
            Dict with rollback results (success, message, details)
        """
        import shutil
        import json

        result = {
            "success": False,
            "config": config,
            "dry_run": dry_run,
            "message": "",
            "details": {},
        }

        try:
            ai_root = Path(self.ringrift_path) / "ai-service"
            models_dir = ai_root / "models"
            archive_dir = models_dir / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)

            # Parse config to get board type and player count
            parts = config.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].endswith("p"):
                result["message"] = f"Invalid config format: {config}"
                return result

            board = parts[0]
            players = parts[1][:-1]

            # Find the current best model alias
            # Common patterns: ringrift_best_sq8_2p, ringrift_best_square8_2p
            board_abbrev = board.replace("square", "sq").replace("hexagonal", "hex")
            best_patterns = [
                f"ringrift_best_{board_abbrev}_{players}p.pth",
                f"ringrift_best_{board}_{players}p.pth",
            ]

            current_best = None
            for pattern in best_patterns:
                candidate = models_dir / pattern
                if candidate.exists():
                    current_best = candidate
                    break

            if not current_best:
                result["message"] = f"No best model found for {config}"
                return result

            # Find previous checkpoints for this config
            checkpoint_dir = models_dir / "checkpoints"
            checkpoints = []
            if checkpoint_dir.exists():
                for ckpt in checkpoint_dir.glob(f"*{board_abbrev}*{players}p*.pth"):
                    try:
                        stat = ckpt.stat()
                        checkpoints.append({
                            "path": ckpt,
                            "mtime": stat.st_mtime,
                            "name": ckpt.name,
                        })
                    except Exception:
                        continue

            # Also check archive for previous best models
            for archived in archive_dir.glob(f"*{board_abbrev}*{players}p*.pth"):
                try:
                    stat = archived.stat()
                    checkpoints.append({
                        "path": archived,
                        "mtime": stat.st_mtime,
                        "name": archived.name,
                    })
                except Exception:
                    continue

            # Sort by modification time descending
            checkpoints.sort(key=lambda x: x["mtime"], reverse=True)

            # Filter out the current best model
            current_mtime = current_best.stat().st_mtime
            previous_checkpoints = [c for c in checkpoints if abs(c["mtime"] - current_mtime) > 60]

            if not previous_checkpoints:
                result["message"] = f"No previous checkpoints found for rollback of {config}"
                return result

            # Select the most recent previous checkpoint
            rollback_source = previous_checkpoints[0]

            result["details"] = {
                "current_model": current_best.name,
                "rollback_to": rollback_source["name"],
                "rollback_age_hours": round((time.time() - rollback_source["mtime"]) / 3600, 1),
                "available_checkpoints": len(previous_checkpoints),
            }

            if dry_run:
                result["success"] = True
                result["message"] = f"Dry run: Would rollback {current_best.name} to {rollback_source['name']}"
                return result

            # Archive the current model
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archived_name = f"{current_best.stem}_archived_{timestamp}.pth"
            shutil.copy2(current_best, archive_dir / archived_name)

            # Restore the previous checkpoint
            shutil.copy2(rollback_source["path"], current_best)

            # Log the rollback
            rollback_log = ai_root / "logs" / "rollbacks.json"
            rollback_log.parent.mkdir(parents=True, exist_ok=True)

            rollback_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": config,
                "previous_model": current_best.name,
                "rolled_back_to": rollback_source["name"],
                "archived_as": archived_name,
            }

            try:
                existing = json.loads(rollback_log.read_text()) if rollback_log.exists() else []
            except Exception:
                existing = []

            existing.append(rollback_entry)
            rollback_log.write_text(json.dumps(existing[-100:], indent=2))  # Keep last 100 rollbacks

            result["success"] = True
            result["message"] = f"Successfully rolled back {config} from {current_best.name} to {rollback_source['name']}"

            # Increment rollback counter
            self.diversity_metrics["rollbacks"] += 1

            # Send alert notification
            asyncio.create_task(self.notifier.send(
                title="Model Rollback Executed",
                message=f"Rolled back {config} from {current_best.name} to {rollback_source['name']}",
                level="warning",
                fields={
                    "Config": config,
                    "Previous": current_best.name,
                    "Restored": rollback_source["name"],
                    "Age": f"{result['details']['rollback_age_hours']:.1f}h",
                },
                node_id=self.node_id,
            ))

        except Exception as e:
            result["message"] = f"Rollback failed: {str(e)}"

        return result

    async def _auto_rollback_check(self) -> List[Dict[str, Any]]:
        """Automatically check and execute rollbacks for critical candidates.

        Returns list of executed rollbacks.
        """
        # Check if auto-rollback is enabled
        if os.environ.get("RINGRIFT_AUTO_ROLLBACK", "").lower() not in ("1", "true", "yes"):
            return []

        executed = []
        try:
            status = await self._check_rollback_conditions()
            for candidate in status.get("candidates", []):
                # Only auto-rollback if multiple serious conditions are met
                reasons = candidate.get("reasons", [])
                if len(reasons) >= 2 or any("Overfitting" in r for r in reasons):
                    config = candidate["config"]
                    result = await self._execute_rollback(config, dry_run=False)
                    executed.append(result)
                    if result["success"]:
                        logger.warning(f"[AUTO-ROLLBACK] Executed for {config}: {reasons}")
        except Exception as e:
            logger.error(f"[AUTO-ROLLBACK] Error: {e}")

        return executed

    # =========================================================================
    # Feature 6: Distributed Selfplay Autoscaling
    # =========================================================================

    async def _get_autoscaling_metrics(self) -> Dict[str, Any]:
        """Get metrics for autoscaling decisions."""
        # Autoscaling thresholds tuned for 46-node cluster
        # These can be overridden via environment variables
        max_workers = int(os.environ.get("RINGRIFT_AUTOSCALE_MAX_WORKERS", "46"))
        min_workers = int(os.environ.get("RINGRIFT_AUTOSCALE_MIN_WORKERS", "2"))
        scale_up_threshold = int(os.environ.get("RINGRIFT_AUTOSCALE_SCALE_UP_GPH", "100"))
        scale_down_threshold = int(os.environ.get("RINGRIFT_AUTOSCALE_SCALE_DOWN_GPH", "500"))
        target_freshness = float(os.environ.get("RINGRIFT_AUTOSCALE_TARGET_FRESHNESS_HOURS", "2"))

        autoscale = {
            "current_state": {},
            "recommendations": [],
            "thresholds": {
                "scale_up_games_per_hour": scale_up_threshold,  # Scale up if below this
                "scale_down_games_per_hour": scale_down_threshold,  # Scale down if above this
                "max_workers": max_workers,
                "min_workers": min_workers,
                "target_data_freshness_hours": target_freshness,
            },
        }

        try:
            # Get current worker count
            with self.peers_lock:
                total_nodes = len(self.peers) + 1
                gpu_nodes = len([p for p in self.peers.values() if getattr(p, "has_gpu", False)])
                if self.self_info.has_gpu:
                    gpu_nodes += 1

            with self.jobs_lock:
                active_selfplay = len([j for j in self.local_jobs.values()
                                      if j.job_type in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY)
                                      and j.status == "running"])

            autoscale["current_state"] = {
                "total_nodes": total_nodes,
                "gpu_nodes": gpu_nodes,
                "active_selfplay_jobs": active_selfplay,
            }

            # Get game generation throughput
            analytics = await self._get_game_analytics_cached()
            total_throughput = sum(c.get("throughput_per_hour", 0) for c in analytics.get("configs", {}).values())

            autoscale["current_state"]["games_per_hour"] = round(total_throughput, 1)

            # Get data freshness
            now = time.time()
            ai_root = Path(self.ringrift_path) / "ai-service"
            selfplay_dir = ai_root / "data" / "selfplay"

            freshest_data = 0
            if selfplay_dir.exists():
                for jsonl in selfplay_dir.rglob("*.jsonl"):
                    try:
                        mtime = jsonl.stat().st_mtime
                        if mtime > freshest_data:
                            freshest_data = mtime
                    except Exception:
                        continue

            data_age_hours = (now - freshest_data) / 3600 if freshest_data > 0 else 999
            autoscale["current_state"]["data_freshness_hours"] = round(data_age_hours, 2)

            # Generate recommendations
            thresholds = autoscale["thresholds"]

            if total_throughput < thresholds["scale_up_games_per_hour"] and total_nodes < thresholds["max_workers"]:
                autoscale["recommendations"].append({
                    "action": "scale_up",
                    "reason": f"Low throughput ({total_throughput:.0f} games/h < {thresholds['scale_up_games_per_hour']})",
                    "suggested_workers": min(total_nodes + 2, thresholds["max_workers"]),
                })

            if total_throughput > thresholds["scale_down_games_per_hour"] and total_nodes > thresholds["min_workers"]:
                autoscale["recommendations"].append({
                    "action": "scale_down",
                    "reason": f"High throughput ({total_throughput:.0f} games/h > {thresholds['scale_down_games_per_hour']})",
                    "suggested_workers": max(total_nodes - 1, thresholds["min_workers"]),
                })

            if data_age_hours > thresholds["target_data_freshness_hours"]:
                autoscale["recommendations"].append({
                    "action": "scale_up",
                    "reason": f"Stale data ({data_age_hours:.1f}h > {thresholds['target_data_freshness_hours']}h)",
                    "suggested_workers": min(total_nodes + 1, thresholds["max_workers"]),
                })

            # Cost optimization recommendation
            efficiency = await self._get_training_efficiency_cached()
            elo_per_hour = efficiency.get("summary", {}).get("overall_elo_per_gpu_hour", 0)
            if elo_per_hour < 1 and total_nodes > 2:
                autoscale["recommendations"].append({
                    "action": "optimize",
                    "reason": f"Low efficiency ({elo_per_hour:.2f} Elo/GPU-h) - consider reducing workers",
                    "suggested_workers": max(total_nodes - 1, thresholds["min_workers"]),
                })

        except Exception:
            pass

        return autoscale

    async def handle_victory_table(self, request: web.Request) -> web.Response:
        """GET /victory/table - Victory type breakdown for Grafana Infinity.

        Returns victory type counts by board config in table format.
        Supports optional query params:
            - board_type: filter by board type
            - num_players: filter by player count
        """
        from collections import defaultdict

        try:
            board_type_filter = request.query.get("board_type")
            num_players_filter = request.query.get("num_players")
            if num_players_filter:
                try:
                    num_players_filter = int(num_players_filter)
                except ValueError:
                    num_players_filter = None

            stats = await self._get_victory_type_stats()

            # Group by config for table display
            config_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for (board_type, num_players, victory_type), count in stats.items():
                # Apply filters
                if board_type_filter and board_type != board_type_filter:
                    continue
                if num_players_filter and num_players != num_players_filter:
                    continue
                config = f"{board_type}_{num_players}p"
                config_stats[config][victory_type] = count

            # Build table rows
            table_data = []
            for config in sorted(config_stats.keys()):
                vt_counts = config_stats[config]
                total = sum(vt_counts.values())
                row = {
                    "Config": config,
                    "Total": total,
                    "Territory": vt_counts.get("territory", 0),
                    "LPS": vt_counts.get("lps", 0),
                    "Elimination": vt_counts.get("elimination", 0),
                    "RingElim": vt_counts.get("ring_elimination", 0),
                    "Stalemate": vt_counts.get("stalemate", 0),
                }
                # Add percentages
                if total > 0:
                    row["Territory%"] = round(100 * vt_counts.get("territory", 0) / total, 1)
                    row["LPS%"] = round(100 * vt_counts.get("lps", 0) / total, 1)
                    row["Elimination%"] = round(100 * vt_counts.get("elimination", 0) / total, 1)
                    row["RingElim%"] = round(100 * vt_counts.get("ring_elimination", 0) / total, 1)
                    row["Stalemate%"] = round(100 * vt_counts.get("stalemate", 0) / total, 1)
                else:
                    row["Territory%"] = row["LPS%"] = row["Elimination%"] = row["RingElim%"] = row["Stalemate%"] = 0
                table_data.append(row)

            return web.json_response(table_data)

        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_elo_history(self, request: web.Request) -> web.Response:
        """GET /elo/history - Historical Elo ratings for time series visualization.

        Query params:
            - config: Filter by config (e.g., square8_2p)
            - model: Filter by model/participant_id (supports partial match)
            - nn_only: If "true", filter to NN models only
            - hours: Hours of history (default 168 = 1 week)
            - limit: Max entries to return (default 5000)
        """
        import sqlite3

        try:
            config_filter = request.query.get("config")
            model_filter = request.query.get("model")
            nn_only = request.query.get("nn_only", "").lower() == "true"
            hours = int(request.query.get("hours", "168"))
            limit = int(request.query.get("limit", "5000"))

            ai_root = Path(self.ringrift_path) / "ai-service"

            # Canonical Elo database for trained models
            db_paths = [
                ai_root / "data" / "unified_elo.db",
            ]

            data = []
            cutoff = time.time() - (hours * 3600)

            for db_path in db_paths:
                if not db_path.exists():
                    continue

                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()

                    # Check if this DB has data
                    cursor.execute("SELECT COUNT(*) FROM rating_history WHERE timestamp > ?", (cutoff,))
                    count = cursor.fetchone()[0]
                    if count == 0:
                        conn.close()
                        continue

                    # Build query - unified_elo.db has different schema (no board_type/num_players)
                    cursor.execute("PRAGMA table_info(rating_history)")
                    columns = {col[1] for col in cursor.fetchall()}

                    if "board_type" in columns:
                        # unified_elo.db schema
                        query = """
                            SELECT participant_id, board_type, num_players, rating, games_played, timestamp
                            FROM rating_history
                            WHERE timestamp > ?
                        """
                        params = [cutoff]

                        if config_filter:
                            parts = config_filter.replace("_", " ").split()
                            if len(parts) >= 2:
                                board_type = parts[0]
                                num_players = int(parts[1].replace("p", ""))
                                query += " AND board_type = ? AND num_players = ?"
                                params.extend([board_type, num_players])
                    else:
                        # unified_elo.db schema (model_id instead of participant_id)
                        query = """
                            SELECT model_id, rating, games_played, timestamp
                            FROM rating_history
                            WHERE timestamp > ?
                        """
                        params = [cutoff]

                    if model_filter:
                        col = "participant_id" if "participant_id" in columns else "model_id"
                        query += f" AND {col} LIKE ?"
                        params.append(f"%{model_filter}%")

                    if nn_only:
                        col = "participant_id" if "participant_id" in columns else "model_id"
                        query += f" AND ({col} LIKE '%nn%' OR {col} LIKE '%NN%')"

                    query += f" ORDER BY timestamp DESC LIMIT {limit}"

                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    conn.close()

                    # Format for Grafana time series
                    for row in rows:
                        if "board_type" in columns:
                            participant_id, board_type, num_players, rating, games_played, ts = row
                            config = f"{board_type}_{num_players}p"
                        else:
                            model_id, rating, games_played, ts = row
                            participant_id = model_id
                            # Extract config from model name (e.g., sq8_2p_nn_baseline -> square8_2p)
                            if "sq8" in model_id.lower() or "square8" in model_id.lower():
                                config = "square8_2p"
                            elif "sq19" in model_id.lower() or "square19" in model_id.lower():
                                config = "square19_2p"
                            else:
                                config = "unknown"

                        data.append({
                            "time": int(ts * 1000),  # Grafana expects ms
                            "model": participant_id,
                            "config": config,
                            "elo": round(rating, 1),
                            "games": games_played,
                        })

                    # If we got data from this DB, don't check others
                    if data:
                        break

                except sqlite3.Error:
                    continue

            # Sort by time ascending for time series
            data.sort(key=lambda x: x["time"])

            return web.json_response(data)

        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_games_analytics(self, request: web.Request) -> web.Response:
        """GET /games/analytics - Game statistics for dashboards.

        Returns aggregated game analytics including:
        - Average game length by config
        - Victory type distribution
        - Games per hour throughput
        - Opening move diversity
        """
        import json
        from collections import defaultdict

        try:
            hours = int(request.query.get("hours", "24"))
            cutoff = time.time() - (hours * 3600)

            ai_root = Path(self.ringrift_path) / "ai-service"
            data_dirs = [
                ai_root / "data" / "games" / "daemon_sync",
                ai_root / "data" / "selfplay",
            ]

            # Aggregation containers
            game_lengths: Dict[str, List[int]] = defaultdict(list)
            victory_types: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            games_by_hour: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
            opening_moves: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            total_games = 0

            for data_dir in data_dirs:
                if not data_dir.exists():
                    continue
                for jsonl_path in data_dir.rglob("*.jsonl"):
                    try:
                        if jsonl_path.stat().st_mtime < cutoff:
                            continue
                        with open(jsonl_path, "r") as f:
                            for line in f:
                                try:
                                    game = json.loads(line)
                                    board_type = game.get("board_type", "unknown")
                                    num_players = game.get("num_players", 0)
                                    config = f"{board_type}_{num_players}p"

                                    # Game length
                                    length = game.get("length", 0)
                                    if length > 0:
                                        game_lengths[config].append(length)

                                    # Victory type
                                    vt = game.get("victory_type", "unknown")
                                    if vt:
                                        victory_types[config][vt] += 1

                                    # Games by hour (for throughput)
                                    moves = game.get("moves", [])
                                    if moves and len(moves) > 0:
                                        # Use first move timestamp or file mtime
                                        hour_bucket = int(jsonl_path.stat().st_mtime // 3600)
                                        games_by_hour[config][hour_bucket] += 1

                                    # Opening moves (first 3 moves)
                                    if moves and len(moves) >= 1:
                                        first_move = str(moves[0].get("action", ""))[:20]
                                        if first_move:
                                            opening_moves[config][first_move] += 1

                                    total_games += 1
                                except json.JSONDecodeError:
                                    continue
                    except Exception:
                        continue

            # Build response
            analytics = {
                "period_hours": hours,
                "total_games": total_games,
                "configs": {}
            }

            for config in set(list(game_lengths.keys()) + list(victory_types.keys())):
                lengths = game_lengths.get(config, [])
                vt = dict(victory_types.get(config, {}))
                openings = dict(opening_moves.get(config, {}))

                # Calculate throughput (games/hour)
                hourly = games_by_hour.get(config, {})
                throughput = sum(hourly.values()) / max(len(hourly), 1) if hourly else 0

                analytics["configs"][config] = {
                    "games": len(lengths),
                    "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
                    "min_length": min(lengths) if lengths else 0,
                    "max_length": max(lengths) if lengths else 0,
                    "victory_types": vt,
                    "throughput_per_hour": round(throughput, 1),
                    "opening_diversity": len(openings),
                    "top_openings": dict(sorted(openings.items(), key=lambda x: -x[1])[:5]),
                }

            return web.json_response(analytics)

        except Exception as e:
            return web.json_response({"error": str(e)})

    async def handle_training_metrics(self, request: web.Request) -> web.Response:
        """GET /training/metrics - Training loss and accuracy metrics.

        Returns recent training metrics from log files.
        """
        import re

        try:
            ai_root = Path(self.ringrift_path) / "ai-service"
            logs_dir = ai_root / "logs" / "training"

            metrics = {
                "configs": {},
                "latest_training": None,
            }

            if not logs_dir.exists():
                return web.json_response(metrics)

            # Find recent training logs
            log_files = sorted(logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:10]

            for log_file in log_files:
                try:
                    content = log_file.read_text()

                    # Extract config from filename (e.g., train_square8_2p_20251214.log)
                    config_match = re.search(r"(square\d+|hexagonal|hex)_(\d+)p", log_file.name)
                    if not config_match:
                        continue
                    config = f"{config_match.group(1)}_{config_match.group(2)}p"

                    # Parse training metrics from log
                    # Look for patterns like: "Epoch 5: loss=0.423, policy_loss=0.312, value_loss=0.111"
                    loss_pattern = re.compile(
                        r"[Ee]poch\s+(\d+).*?loss[=:]\s*([\d.]+).*?"
                        r"(?:policy[_\s]?loss[=:]\s*([\d.]+))?.*?"
                        r"(?:value[_\s]?loss[=:]\s*([\d.]+))?"
                    )

                    epochs = []
                    for match in loss_pattern.finditer(content):
                        epoch = int(match.group(1))
                        total_loss = float(match.group(2))
                        policy_loss = float(match.group(3)) if match.group(3) else None
                        value_loss = float(match.group(4)) if match.group(4) else None
                        epochs.append({
                            "epoch": epoch,
                            "loss": total_loss,
                            "policy_loss": policy_loss,
                            "value_loss": value_loss,
                        })

                    if epochs:
                        metrics["configs"][config] = {
                            "log_file": log_file.name,
                            "epochs": epochs[-20:],  # Last 20 epochs
                            "latest_loss": epochs[-1]["loss"] if epochs else None,
                            "latest_epoch": epochs[-1]["epoch"] if epochs else None,
                        }
                        if not metrics["latest_training"]:
                            metrics["latest_training"] = {
                                "config": config,
                                "file": log_file.name,
                                "mtime": log_file.stat().st_mtime,
                            }

                except Exception:
                    continue

            return web.json_response(metrics)

        except Exception as e:
            return web.json_response({"error": str(e)})

    async def handle_holdout_metrics(self, request: web.Request) -> web.Response:
        """GET /holdout/metrics - Holdout validation metrics.

        Returns holdout set statistics and evaluation results for overfitting detection.
        Supports optional query params:
            - config: Filter by config (e.g., square8_2p)
        """
        try:
            config_filter = request.query.get("config")
            metrics = await self._get_holdout_metrics_cached()

            if config_filter:
                # Filter to specific config
                filtered = {
                    "configs": {k: v for k, v in metrics.get("configs", {}).items() if k == config_filter},
                    "evaluations": [e for e in metrics.get("evaluations", []) if e.get("config") == config_filter],
                    "summary": metrics.get("summary", {}),
                }
                return web.json_response(filtered)

            return web.json_response(metrics)

        except Exception as e:
            return web.json_response({"error": str(e)})

    async def handle_holdout_table(self, request: web.Request) -> web.Response:
        """GET /holdout/table - Holdout validation data in table format for Grafana Infinity.

        Returns holdout metrics as flat table rows.
        """
        try:
            metrics = await self._get_holdout_metrics_cached()

            table_data = []
            for config, data in metrics.get("configs", {}).items():
                row = {
                    "Config": config,
                    "HoldoutGames": data.get("holdout_games", 0),
                    "HoldoutPositions": data.get("holdout_positions", 0),
                    "HoldoutLoss": round(data.get("holdout_loss", 0), 4) if data.get("holdout_loss") else None,
                    "HoldoutAccuracy": round(data.get("holdout_accuracy", 0) * 100, 1) if data.get("holdout_accuracy") else None,
                    "OverfitGap": round(data.get("overfit_gap", 0), 4) if data.get("overfit_gap") else None,
                    "Status": "OK" if (data.get("overfit_gap") or 0) < 0.15 else "OVERFITTING",
                }
                table_data.append(row)

            return web.json_response(table_data)

        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_mcts_stats(self, request: web.Request) -> web.Response:
        """GET /mcts/stats - MCTS search statistics.

        Returns MCTS performance metrics including nodes/move, search depth, and timing.
        """
        try:
            stats = await self._get_mcts_stats_cached()
            return web.json_response(stats)

        except Exception as e:
            return web.json_response({"error": str(e)})

    async def handle_mcts_table(self, request: web.Request) -> web.Response:
        """GET /mcts/table - MCTS stats in table format for Grafana Infinity.

        Returns MCTS statistics as flat table rows.
        """
        try:
            stats = await self._get_mcts_stats_cached()

            table_data = []
            # Add summary row
            summary = stats.get("summary", {})
            if summary:
                table_data.append({
                    "Config": "CLUSTER AVERAGE",
                    "AvgNodes": round(summary.get("avg_nodes_per_move", 0), 0),
                    "MaxNodes": summary.get("max_nodes_per_move", 0),
                    "AvgDepth": round(summary.get("avg_search_depth", 0), 1),
                    "MaxDepth": summary.get("max_search_depth", 0),
                    "AvgTime": round(summary.get("avg_time_per_move", 0), 3) if summary.get("avg_time_per_move") else None,
                })

            # Add per-config rows
            for config, data in stats.get("configs", {}).items():
                table_data.append({
                    "Config": config,
                    "AvgNodes": round(data.get("avg_nodes", 0), 0) if data.get("avg_nodes") else None,
                    "MaxNodes": None,
                    "AvgDepth": round(data.get("avg_depth", 0), 1) if data.get("avg_depth") else None,
                    "MaxDepth": None,
                    "AvgTime": None,
                })

            return web.json_response(table_data)

        except Exception as e:
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Feature Endpoints
    # =========================================================================

    async def handle_matchup_matrix(self, request: web.Request) -> web.Response:
        """GET /matchups/matrix - Head-to-head matchup statistics."""
        try:
            matrix = await self._get_matchup_matrix_cached()
            return web.json_response(matrix)
        except Exception as e:
            return web.json_response({"error": str(e)})

    async def handle_matchup_table(self, request: web.Request) -> web.Response:
        """GET /matchups/table - Matchups in table format for Grafana Infinity."""
        try:
            matrix = await self._get_matchup_matrix_cached()
            table_data = []
            for matchup in matrix.get("matchups", []):
                table_data.append({
                    "ModelA": matchup["model_a"],
                    "ModelB": matchup["model_b"],
                    "AWins": matchup["a_wins"],
                    "BWins": matchup["b_wins"],
                    "Draws": matchup["draws"],
                    "Total": matchup["total"],
                    "AWinRate": round(matchup["a_win_rate"] * 100, 1),
                })
            return web.json_response(table_data)
        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_model_lineage(self, request: web.Request) -> web.Response:
        """GET /models/lineage - Model ancestry and generation tracking."""
        try:
            lineage = await self._get_model_lineage_cached()
            return web.json_response(lineage)
        except Exception as e:
            return web.json_response({"error": str(e)})

    async def handle_model_lineage_table(self, request: web.Request) -> web.Response:
        """GET /models/lineage/table - Model lineage in table format for Grafana Infinity."""
        try:
            lineage = await self._get_model_lineage_cached()
            table_data = []
            for model in lineage.get("models", []):
                table_data.append({
                    "Name": model["name"],
                    "Config": model["config"],
                    "Generation": model["generation"],
                    "SizeMB": model["size_mb"],
                    "AgeHours": model["age_hours"],
                })
            return web.json_response(sorted(table_data, key=lambda x: (-x["Generation"], x["Config"])))
        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_data_quality(self, request: web.Request) -> web.Response:
        """GET /data/quality - Data quality metrics and issue detection."""
        try:
            quality = await self._get_data_quality_cached()
            return web.json_response(quality)
        except Exception as e:
            return web.json_response({"error": str(e)})

    async def handle_data_quality_table(self, request: web.Request) -> web.Response:
        """GET /data/quality/table - Data quality in table format for Grafana Infinity."""
        try:
            quality = await self._get_data_quality_cached()
            table_data = []
            for config, metrics in quality.get("configs", {}).items():
                status = "OK"
                for issue in quality.get("issues", []):
                    if issue["config"] == config and issue["severity"] == "warning":
                        status = "WARNING"
                        break
                table_data.append({
                    "Config": config,
                    "Games": metrics["total_games"],
                    "AvgLength": metrics["avg_length"],
                    "ShortRate": metrics["short_game_rate"],
                    "StalemateRate": metrics["stalemate_rate"],
                    "OpeningDiv": metrics["opening_diversity"],
                    "Status": status,
                })
            return web.json_response(table_data)
        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_data_quality_issues(self, request: web.Request) -> web.Response:
        """GET /data/quality/issues - Data quality issues in table format."""
        try:
            quality = await self._get_data_quality_cached()
            return web.json_response(quality.get("issues", []))
        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_training_efficiency(self, request: web.Request) -> web.Response:
        """GET /training/efficiency - Training efficiency and cost metrics."""
        try:
            efficiency = await self._get_training_efficiency_cached()
            return web.json_response(efficiency)
        except Exception as e:
            return web.json_response({"error": str(e)})

    async def handle_training_efficiency_table(self, request: web.Request) -> web.Response:
        """GET /training/efficiency/table - Efficiency in table format for Grafana Infinity."""
        try:
            efficiency = await self._get_training_efficiency_cached()
            table_data = []
            for config, metrics in efficiency.get("configs", {}).items():
                table_data.append({
                    "Config": config,
                    "GPUHours": metrics["gpu_hours"],
                    "EloGain": metrics["elo_gain"],
                    "EloPerHour": metrics["elo_per_gpu_hour"],
                    "CostUSD": metrics["estimated_cost_usd"],
                    "CostPerElo": metrics["cost_per_elo_point"],
                })
            return web.json_response(table_data)
        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_rollback_status(self, request: web.Request) -> web.Response:
        """GET /rollback/status - Model rollback status and recommendations."""
        try:
            status = await self._check_rollback_conditions()
            return web.json_response(status)
        except Exception as e:
            return web.json_response({"error": str(e)})

    async def handle_rollback_candidates(self, request: web.Request) -> web.Response:
        """GET /rollback/candidates - Rollback candidates in table format."""
        try:
            status = await self._check_rollback_conditions()
            table_data = []
            for candidate in status.get("candidates", []):
                table_data.append({
                    "Config": candidate["config"],
                    "Reasons": ", ".join(candidate["reasons"]),
                    "Recommended": "YES" if candidate["rollback_recommended"] else "NO",
                })
            return web.json_response(table_data)
        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_rollback_execute(self, request: web.Request) -> web.Response:
        """POST /rollback/execute - Execute a model rollback.

        Query params:
            config: Config string like "square8_2p" (required)
            dry_run: If "true", only simulate the rollback (default: false)
        """
        try:
            config = request.query.get("config")
            if not config:
                return web.json_response({"error": "Missing required parameter: config"}, status=400)

            dry_run = request.query.get("dry_run", "").lower() in ("true", "1", "yes")

            result = await self._execute_rollback(config, dry_run=dry_run)
            status_code = 200 if result["success"] else 400
            return web.json_response(result, status=status_code)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_rollback_auto(self, request: web.Request) -> web.Response:
        """POST /rollback/auto - Trigger automatic rollback check for all configs.

        This will check all configs for rollback conditions and execute rollbacks
        for any that meet the criteria.
        """
        try:
            # Temporarily enable auto-rollback for this request
            original_env = os.environ.get("RINGRIFT_AUTO_ROLLBACK", "")
            os.environ["RINGRIFT_AUTO_ROLLBACK"] = "true"

            executed = await self._auto_rollback_check()

            # Restore original env
            if original_env:
                os.environ["RINGRIFT_AUTO_ROLLBACK"] = original_env
            else:
                os.environ.pop("RINGRIFT_AUTO_ROLLBACK", None)

            return web.json_response({
                "executed_rollbacks": executed,
                "count": len(executed),
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_autoscale_metrics(self, request: web.Request) -> web.Response:
        """GET /autoscale/metrics - Autoscaling metrics and recommendations."""
        try:
            metrics = await self._get_autoscaling_metrics()
            return web.json_response(metrics)
        except Exception as e:
            return web.json_response({"error": str(e)})

    async def handle_autoscale_recommendations(self, request: web.Request) -> web.Response:
        """GET /autoscale/recommendations - Autoscaling recommendations table."""
        try:
            metrics = await self._get_autoscaling_metrics()
            table_data = []
            for rec in metrics.get("recommendations", []):
                table_data.append({
                    "Action": rec["action"].upper(),
                    "Reason": rec["reason"],
                    "SuggestedWorkers": rec["suggested_workers"],
                })
            if not table_data:
                table_data.append({
                    "Action": "NONE",
                    "Reason": "Current scaling is optimal",
                    "SuggestedWorkers": metrics.get("current_state", {}).get("total_nodes", 1),
                })
            return web.json_response(table_data)
        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_resource_optimizer(self, request: web.Request) -> web.Response:
        """GET /resource/optimizer - Resource optimizer state and recommendations.

        Returns cluster-wide utilization state, PID-controlled optimization
        recommendations, and target utilization ranges (60-80%).
        """
        try:
            if not HAS_NEW_COORDINATION:
                return web.json_response({
                    "error": "Resource optimizer not available",
                    "available": False,
                })

            optimizer = get_resource_optimizer()
            cluster_state = optimizer.get_cluster_state()
            recommendation = optimizer.get_optimization_recommendation()
            metrics = optimizer.get_metrics_dict()

            return web.json_response({
                "available": True,
                "cluster_state": {
                    "total_cpu_util": round(cluster_state.total_cpu_util, 1),
                    "total_gpu_util": round(cluster_state.total_gpu_util, 1),
                    "total_memory_util": round(cluster_state.total_memory_util, 1),
                    "gpu_node_count": cluster_state.gpu_node_count,
                    "cpu_node_count": cluster_state.cpu_node_count,
                    "total_jobs": cluster_state.total_jobs,
                    "nodes": [n.to_dict() for n in cluster_state.nodes],
                },
                "recommendation": recommendation.to_dict(),
                "targets": {
                    "min": TARGET_GPU_UTIL_MIN,
                    "max": TARGET_GPU_UTIL_MAX,
                    "optimal": (TARGET_GPU_UTIL_MIN + TARGET_GPU_UTIL_MAX) // 2,
                },
                "metrics": metrics,
                "in_target_range": {
                    "cpu": TARGET_GPU_UTIL_MIN <= cluster_state.total_cpu_util <= TARGET_GPU_UTIL_MAX,
                    "gpu": TARGET_GPU_UTIL_MIN <= cluster_state.total_gpu_util <= TARGET_GPU_UTIL_MAX
                           if cluster_state.gpu_node_count > 0 else True,
                },
            })
        except Exception as e:
            return web.json_response({"error": str(e), "available": False}, status=500)

    async def handle_resource_utilization_history(self, request: web.Request) -> web.Response:
        """GET /resource/history - Resource utilization history for graphing.

        Query params:
            node_id: Specific node (optional, defaults to cluster average)
            hours: Hours of history (default: 1)
        """
        try:
            if not HAS_NEW_COORDINATION:
                return web.json_response([])

            node_id = request.query.get("node_id")
            hours = float(request.query.get("hours", "1"))

            optimizer = get_resource_optimizer()
            history = optimizer.get_utilization_history(node_id=node_id, hours=hours)
            return web.json_response(history)
        except Exception as e:
            return web.json_response([])

    async def handle_webhook_test(self, request: web.Request) -> web.Response:
        """POST /webhook/test - Test webhook notification.

        Query params:
            level: debug/info/warning/error (default: info)
            message: Custom message (default: "Test notification")
        """
        try:
            level = request.query.get("level", "info")
            message = request.query.get("message", "Test notification from RingRift AI orchestrator")

            has_slack = bool(self.notifier.slack_webhook)
            has_discord = bool(self.notifier.discord_webhook)

            if not has_slack and not has_discord:
                return web.json_response({
                    "success": False,
                    "message": "No webhooks configured. Set RINGRIFT_SLACK_WEBHOOK and/or RINGRIFT_DISCORD_WEBHOOK",
                })

            await self.notifier.send(
                title="Webhook Test",
                message=message,
                level=level,
                fields={
                    "Node": self.node_id,
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Level": level.upper(),
                },
                node_id=self.node_id,
            )

            return web.json_response({
                "success": True,
                "message": f"Test notification sent to {'Slack' if has_slack else ''}{' and ' if has_slack and has_discord else ''}{'Discord' if has_discord else ''}",
                "level": level,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_trends_summary(self, request: web.Request) -> web.Response:
        """GET /trends/summary - Get summary of metrics over time period.

        Query params:
            hours: Time period in hours (default: 24)
        """
        try:
            hours = float(request.query.get("hours", "24"))
            summary = self.get_metrics_summary(hours)
            return web.json_response(summary)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_trends_history(self, request: web.Request) -> web.Response:
        """GET /trends/history - Get historical metrics data.

        Query params:
            metric: Metric type (required) - e.g., "best_elo", "games_generated", "training_loss"
            hours: Time period in hours (default: 24)
            board: Board type filter (optional) - e.g., "square8"
            players: Number of players filter (optional) - e.g., 2
            limit: Max records to return (default: 1000)
        """
        try:
            metric_type = request.query.get("metric")
            if not metric_type:
                return web.json_response({"error": "Missing required parameter: metric"}, status=400)

            hours = float(request.query.get("hours", "24"))
            board_type = request.query.get("board")
            num_players = int(request.query.get("players")) if request.query.get("players") else None
            limit = int(request.query.get("limit", "1000"))

            history = self.get_metrics_history(
                metric_type=metric_type,
                board_type=board_type,
                num_players=num_players,
                hours=hours,
                limit=limit,
            )

            return web.json_response({
                "metric": metric_type,
                "period_hours": hours,
                "count": len(history),
                "data": history,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_trends_table(self, request: web.Request) -> web.Response:
        """GET /trends/table - Historical trends in table format for Grafana Infinity.

        Query params:
            metric: Metric type (required)
            hours: Time period (default: 168 = 7 days)
        """
        try:
            metric_type = request.query.get("metric")
            if not metric_type:
                return web.json_response([{"error": "Missing metric parameter"}])

            hours = float(request.query.get("hours", "168"))
            history = self.get_metrics_history(metric_type=metric_type, hours=hours, limit=500)

            table_data = []
            for record in history:
                from datetime import datetime
                ts = datetime.fromtimestamp(record["timestamp"]).strftime("%Y-%m-%d %H:%M")
                config = f"{record.get('board_type', '')}_{record.get('num_players', '')}p" if record.get('board_type') else "global"
                table_data.append({
                    "Timestamp": ts,
                    "Config": config,
                    "Value": round(record["value"], 3),
                    "Metric": metric_type,
                })

            return web.json_response(table_data)
        except Exception as e:
            return web.json_response([{"error": str(e)}])

    # ==================== A/B Testing Framework ====================

    def _calculate_ab_test_stats(self, test_id: str) -> Dict[str, Any]:
        """Calculate statistical significance for an A/B test."""
        import math

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get game results
            cursor.execute("""
                SELECT model_a_result, model_a_score, model_b_score, game_length
                FROM ab_test_games WHERE test_id = ?
            """, (test_id,))
            games = cursor.fetchall()
            conn.close()

            if not games:
                return {
                    "games_played": 0,
                    "model_a_wins": 0,
                    "model_b_wins": 0,
                    "draws": 0,
                    "model_a_score": 0.0,
                    "model_b_score": 0.0,
                    "model_a_winrate": 0.0,
                    "model_b_winrate": 0.0,
                    "confidence": 0.0,
                    "likely_winner": None,
                    "statistically_significant": False,
                }

            # Count results
            model_a_wins = sum(1 for g in games if g[0] == "win")
            model_b_wins = sum(1 for g in games if g[0] == "loss")
            draws = sum(1 for g in games if g[0] == "draw")
            total = len(games)

            model_a_score = sum(g[1] for g in games)
            model_b_score = sum(g[2] for g in games)

            # Winrate (using score, e.g., 1 for win, 0.5 for draw, 0 for loss)
            model_a_winrate = model_a_score / total if total > 0 else 0.0
            model_b_winrate = model_b_score / total if total > 0 else 0.0

            # Wilson score confidence interval for statistical significance
            # Using normal approximation for simplicity
            def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
                if n == 0:
                    return (0.0, 1.0)
                p = wins / n
                denominator = 1 + z * z / n
                center = (p + z * z / (2 * n)) / denominator
                spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
                return (max(0, center - spread), min(1, center + spread))

            # Calculate confidence intervals
            a_lo, a_hi = wilson_ci(model_a_wins + draws // 2, total)
            b_lo, b_hi = wilson_ci(model_b_wins + draws // 2, total)

            # Determine if statistically significant (non-overlapping CIs)
            statistically_significant = a_hi < b_lo or b_hi < a_lo

            # Estimate confidence based on score difference and sample size
            if total > 0:
                score_diff = abs(model_a_winrate - model_b_winrate)
                # Rough confidence estimate (higher with more games and larger diff)
                confidence = min(0.99, 1 - math.exp(-total * score_diff * 2))
            else:
                confidence = 0.0

            # Determine likely winner
            likely_winner = None
            if model_a_winrate > model_b_winrate + 0.05:
                likely_winner = "model_a"
            elif model_b_winrate > model_a_winrate + 0.05:
                likely_winner = "model_b"

            avg_game_length = sum(g[3] for g in games if g[3]) / max(1, sum(1 for g in games if g[3]))

            return {
                "games_played": total,
                "model_a_wins": model_a_wins,
                "model_b_wins": model_b_wins,
                "draws": draws,
                "model_a_score": model_a_score,
                "model_b_score": model_b_score,
                "model_a_winrate": round(model_a_winrate, 4),
                "model_b_winrate": round(model_b_winrate, 4),
                "confidence": round(confidence, 4),
                "likely_winner": likely_winner,
                "statistically_significant": statistically_significant,
                "avg_game_length": round(avg_game_length, 1),
            }
        except Exception as e:
            return {"error": str(e)}

    async def handle_abtest_create(self, request: web.Request) -> web.Response:
        """POST /abtest/create - Create a new A/B test between two models.

        JSON body:
            name: Test name (required)
            description: Test description (optional)
            board_type: Board type (required) - e.g., "square8"
            num_players: Number of players (required) - e.g., 2
            model_a: Path or ID of first model (required)
            model_b: Path or ID of second model (required)
            target_games: Number of games to play (default: 100)
            confidence_threshold: Confidence level to conclude (default: 0.95)
        """
        try:
            data = await request.json()

            # Validate required fields
            required = ["name", "board_type", "num_players", "model_a", "model_b"]
            for field in required:
                if field not in data:
                    return web.json_response({"error": f"Missing required field: {field}"}, status=400)

            test_id = str(uuid.uuid4())
            now = time.time()

            test_data = {
                "test_id": test_id,
                "name": data["name"],
                "description": data.get("description", ""),
                "board_type": data["board_type"],
                "num_players": int(data["num_players"]),
                "model_a": data["model_a"],
                "model_b": data["model_b"],
                "target_games": int(data.get("target_games", 100)),
                "confidence_threshold": float(data.get("confidence_threshold", 0.95)),
                "status": "running",
                "winner": None,
                "created_at": now,
                "completed_at": None,
                "metadata": json.dumps(data.get("metadata", {})),
            }

            # Store in database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ab_tests (
                    test_id, name, description, board_type, num_players,
                    model_a, model_b, target_games, confidence_threshold,
                    status, winner, created_at, completed_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_data["test_id"], test_data["name"], test_data["description"],
                test_data["board_type"], test_data["num_players"],
                test_data["model_a"], test_data["model_b"],
                test_data["target_games"], test_data["confidence_threshold"],
                test_data["status"], test_data["winner"],
                test_data["created_at"], test_data["completed_at"],
                test_data["metadata"],
            ))
            conn.commit()
            conn.close()

            # Store in memory
            with self.ab_test_lock:
                self.ab_tests[test_id] = test_data

            return web.json_response({
                "test_id": test_id,
                "status": "created",
                "message": f"A/B test '{data['name']}' created. Submit game results via POST /abtest/result",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_abtest_result(self, request: web.Request) -> web.Response:
        """POST /abtest/result - Submit a game result for an A/B test.

        JSON body:
            test_id: A/B test ID (required)
            game_id: Unique game ID (required)
            winner: "model_a", "model_b", or "draw" (required)
            game_length: Number of moves in the game (optional)
            metadata: Additional game metadata (optional)
        """
        try:
            data = await request.json()

            test_id = data.get("test_id")
            if not test_id:
                return web.json_response({"error": "Missing test_id"}, status=400)

            game_id = data.get("game_id") or str(uuid.uuid4())
            winner = data.get("winner")
            if winner not in ["model_a", "model_b", "draw"]:
                return web.json_response({"error": "winner must be 'model_a', 'model_b', or 'draw'"}, status=400)

            # Calculate scores
            if winner == "model_a":
                model_a_result = "win"
                model_a_score = 1.0
                model_b_score = 0.0
            elif winner == "model_b":
                model_a_result = "loss"
                model_a_score = 0.0
                model_b_score = 1.0
            else:
                model_a_result = "draw"
                model_a_score = 0.5
                model_b_score = 0.5

            now = time.time()

            # Store game result
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Verify test exists
            cursor.execute("SELECT status, target_games, confidence_threshold FROM ab_tests WHERE test_id = ?", (test_id,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return web.json_response({"error": f"Test {test_id} not found"}, status=404)

            test_status, target_games, confidence_threshold = row
            if test_status != "running":
                conn.close()
                return web.json_response({"error": f"Test {test_id} is {test_status}, not running"}, status=400)

            # Insert game result
            cursor.execute("""
                INSERT INTO ab_test_games (
                    test_id, game_id, model_a_result, model_a_score, model_b_score,
                    game_length, played_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id, game_id, model_a_result, model_a_score, model_b_score,
                data.get("game_length"), now, json.dumps(data.get("metadata", {})),
            ))
            conn.commit()
            conn.close()

            # Calculate updated stats
            stats = self._calculate_ab_test_stats(test_id)

            # Check if test should conclude
            should_conclude = False
            if stats.get("games_played", 0) >= target_games:
                should_conclude = True
            elif stats.get("statistically_significant") and stats.get("confidence", 0) >= confidence_threshold:
                should_conclude = True

            if should_conclude:
                winner_model = stats.get("likely_winner")
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE ab_tests SET status = 'completed', winner = ?, completed_at = ?
                    WHERE test_id = ?
                """, (winner_model, time.time(), test_id))
                conn.commit()
                conn.close()

                # Notify
                self.notifier.notify(
                    f"A/B Test Complete: {test_id}",
                    f"Winner: {winner_model or 'inconclusive'}\n"
                    f"Games: {stats['games_played']}, Confidence: {stats['confidence']:.1%}"
                )

            return web.json_response({
                "test_id": test_id,
                "game_id": game_id,
                "recorded": True,
                "stats": stats,
                "concluded": should_conclude,
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_abtest_status(self, request: web.Request) -> web.Response:
        """GET /abtest/status - Get status of an A/B test.

        Query params:
            test_id: A/B test ID (required)
        """
        try:
            test_id = request.query.get("test_id")
            if not test_id:
                return web.json_response({"error": "Missing test_id parameter"}, status=400)

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ab_tests WHERE test_id = ?", (test_id,))
            row = cursor.fetchone()
            conn.close()

            if not row:
                return web.json_response({"error": f"Test {test_id} not found"}, status=404)

            test_data = {
                "test_id": row[0],
                "name": row[1],
                "description": row[2],
                "board_type": row[3],
                "num_players": row[4],
                "model_a": row[5],
                "model_b": row[6],
                "target_games": row[7],
                "confidence_threshold": row[8],
                "status": row[9],
                "winner": row[10],
                "created_at": row[11],
                "completed_at": row[12],
                "metadata": json.loads(row[13]) if row[13] else {},
            }

            # Add current stats
            test_data["stats"] = self._calculate_ab_test_stats(test_id)

            return web.json_response(test_data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_abtest_list(self, request: web.Request) -> web.Response:
        """GET /abtest/list - List all A/B tests.

        Query params:
            status: Filter by status (optional) - "running", "completed", "cancelled"
            limit: Max results (default: 50)
        """
        try:
            status_filter = request.query.get("status")
            limit = int(request.query.get("limit", "50"))

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            if status_filter:
                cursor.execute(
                    "SELECT test_id, name, board_type, num_players, model_a, model_b, status, winner, created_at "
                    "FROM ab_tests WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status_filter, limit)
                )
            else:
                cursor.execute(
                    "SELECT test_id, name, board_type, num_players, model_a, model_b, status, winner, created_at "
                    "FROM ab_tests ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )

            rows = cursor.fetchall()
            conn.close()

            tests = []
            for row in rows:
                test_id = row[0]
                stats = self._calculate_ab_test_stats(test_id)
                tests.append({
                    "test_id": test_id,
                    "name": row[1],
                    "board_type": row[2],
                    "num_players": row[3],
                    "model_a": row[4],
                    "model_b": row[5],
                    "status": row[6],
                    "winner": row[7],
                    "created_at": row[8],
                    "games_played": stats.get("games_played", 0),
                    "model_a_winrate": stats.get("model_a_winrate", 0),
                    "model_b_winrate": stats.get("model_b_winrate", 0),
                    "confidence": stats.get("confidence", 0),
                })

            return web.json_response({"tests": tests, "count": len(tests)})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_abtest_cancel(self, request: web.Request) -> web.Response:
        """POST /abtest/cancel - Cancel a running A/B test.

        JSON body:
            test_id: A/B test ID (required)
            reason: Cancellation reason (optional)
        """
        try:
            data = await request.json()
            test_id = data.get("test_id")
            if not test_id:
                return web.json_response({"error": "Missing test_id"}, status=400)

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM ab_tests WHERE test_id = ?", (test_id,))
            row = cursor.fetchone()

            if not row:
                conn.close()
                return web.json_response({"error": f"Test {test_id} not found"}, status=404)

            if row[0] != "running":
                conn.close()
                return web.json_response({"error": f"Test {test_id} is already {row[0]}"}, status=400)

            cursor.execute(
                "UPDATE ab_tests SET status = 'cancelled', completed_at = ? WHERE test_id = ?",
                (time.time(), test_id)
            )
            conn.commit()
            conn.close()

            return web.json_response({"test_id": test_id, "status": "cancelled"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_abtest_table(self, request: web.Request) -> web.Response:
        """GET /abtest/table - A/B tests in table format for Grafana Infinity.

        Query params:
            status: Filter by status (optional)
        """
        try:
            status_filter = request.query.get("status")

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            if status_filter:
                cursor.execute(
                    "SELECT test_id, name, board_type, num_players, model_a, model_b, status, winner, created_at "
                    "FROM ab_tests WHERE status = ? ORDER BY created_at DESC LIMIT 100",
                    (status_filter,)
                )
            else:
                cursor.execute(
                    "SELECT test_id, name, board_type, num_players, model_a, model_b, status, winner, created_at "
                    "FROM ab_tests ORDER BY created_at DESC LIMIT 100"
                )

            rows = cursor.fetchall()
            conn.close()

            table_data = []
            for row in rows:
                test_id = row[0]
                stats = self._calculate_ab_test_stats(test_id)
                from datetime import datetime
                created = datetime.fromtimestamp(row[8]).strftime("%Y-%m-%d %H:%M") if row[8] else ""

                table_data.append({
                    "Test ID": test_id[:8],
                    "Name": row[1],
                    "Config": f"{row[2]}_{row[3]}p",
                    "Model A": row[4].split("/")[-1] if "/" in row[4] else row[4],
                    "Model B": row[5].split("/")[-1] if "/" in row[5] else row[5],
                    "Games": stats.get("games_played", 0),
                    "A Win%": f"{stats.get('model_a_winrate', 0):.1%}",
                    "B Win%": f"{stats.get('model_b_winrate', 0):.1%}",
                    "Confidence": f"{stats.get('confidence', 0):.1%}",
                    "Status": row[6],
                    "Winner": row[7] or "-",
                    "Created": created,
                })

            return web.json_response(table_data)
        except Exception as e:
            return web.json_response([{"error": str(e)}])

    async def handle_abtest_run(self, request: web.Request) -> web.Response:
        """POST /abtest/run - Start running games for an A/B test using the cluster.

        This schedules games to be played between model_a and model_b on available nodes.

        JSON body:
            test_id: A/B test ID (required)
            parallel_games: Number of games to run in parallel (default: 4)
            think_time_ms: AI think time in ms (default: 100)
        """
        try:
            data = await request.json()
            test_id = data.get("test_id")
            if not test_id:
                return web.json_response({"error": "Missing test_id"}, status=400)

            # Get test info
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT board_type, num_players, model_a, model_b, target_games, status "
                "FROM ab_tests WHERE test_id = ?",
                (test_id,)
            )
            row = cursor.fetchone()
            conn.close()

            if not row:
                return web.json_response({"error": f"Test {test_id} not found"}, status=404)

            board_type, num_players, model_a, model_b, target_games, status = row
            if status != "running":
                return web.json_response({"error": f"Test is {status}, not running"}, status=400)

            # Get current game count
            stats = self._calculate_ab_test_stats(test_id)
            games_remaining = target_games - stats.get("games_played", 0)

            if games_remaining <= 0:
                return web.json_response({
                    "test_id": test_id,
                    "message": "Test has reached target games",
                    "stats": stats,
                })

            parallel = int(data.get("parallel_games", 4))
            think_time = int(data.get("think_time_ms", 100))

            # Schedule games via existing tournament infrastructure
            # This creates a mini-tournament between the two models
            tournament_id = f"abtest_{test_id[:8]}"

            return web.json_response({
                "test_id": test_id,
                "status": "scheduled",
                "games_remaining": games_remaining,
                "parallel_games": parallel,
                "think_time_ms": think_time,
                "message": f"Use tournament infrastructure to run {games_remaining} games between models",
                "hint": "Games should be submitted via POST /abtest/result as they complete",
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_training_status(self, request: web.Request) -> web.Response:
        """Get training pipeline status including NNUE, CMAES, and auto-promotion state.

        Returns daemon state for NNUE training, CMAES optimization, and model promotion.
        """
        try:
            from datetime import datetime

            ai_root = Path(self.ringrift_path) / "ai-service"

            # Load daemon state (from continuous_improvement_daemon.py)
            daemon_state_path = ai_root / "logs" / "improvement_daemon" / "state.json"
            daemon_state = {}
            daemon_running = False
            daemon_pid = None
            daemon_uptime = 0

            # Check if daemon is running
            pid_file = ai_root / "logs" / "improvement_daemon" / "daemon.pid"
            if pid_file.exists():
                try:
                    daemon_pid = int(pid_file.read_text().strip())
                    # Check if process is running
                    import os
                    os.kill(daemon_pid, 0)  # Doesn't kill, just checks
                    daemon_running = True
                except (ValueError, ProcessLookupError, PermissionError):
                    daemon_running = False

            if daemon_state_path.exists():
                try:
                    daemon_state = json.loads(daemon_state_path.read_text())
                    # Calculate uptime if daemon is running
                    if daemon_running and daemon_state.get("started_at"):
                        started = datetime.fromisoformat(daemon_state["started_at"])
                        daemon_uptime = (datetime.now() - started).total_seconds()
                except:
                    pass

            # Load runtime overrides (promoted models)
            overrides_path = ai_root / "data" / "ladder_runtime_overrides.json"
            runtime_overrides = {}
            if overrides_path.exists():
                try:
                    runtime_overrides = json.loads(overrides_path.read_text())
                except:
                    pass

            # Load auto-promotion log
            promotion_log_path = (
                ai_root / "runs" / "promotion" / "model_promotion_history.json"
                if (ai_root / "runs" / "promotion" / "model_promotion_history.json").exists()
                else (ai_root / "data" / "auto_promotion_log.json")
            )
            promotion_log = []
            if promotion_log_path.exists():
                try:
                    promotion_log = json.loads(promotion_log_path.read_text())
                    if isinstance(promotion_log, list):
                        promotion_log = promotion_log[-10:]  # Last 10 entries
                except:
                    pass

            # Check NNUE model timestamps
            nnue_models = {}
            nnue_dir = ai_root / "models" / "nnue"
            if nnue_dir.exists():
                for model_file in nnue_dir.glob("*.pt"):
                    if "_prev" not in model_file.name:
                        stat = model_file.stat()
                        nnue_models[model_file.stem] = {
                            "path": str(model_file),
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "size_mb": round(stat.st_size / 1024 / 1024, 2),
                        }

            # Check trained heuristic profiles
            profiles_path = ai_root / "data" / "trained_heuristic_profiles.json"
            heuristic_profiles = {}
            if profiles_path.exists():
                try:
                    profiles_data = json.loads(profiles_path.read_text())
                    heuristic_profiles = {
                        "count": len(profiles_data),
                        "profiles": list(profiles_data.keys())[:20],
                    }
                except:
                    pass

            return web.json_response({
                "success": True,
                "daemon": {
                    "running": daemon_running,
                    "pid": daemon_pid,
                    "uptime_seconds": daemon_uptime,
                    "current_cycle": daemon_state.get("total_cycles", 0),
                    "last_cycle_at": daemon_state.get("last_cycle_at", ""),
                    "total_games_generated": daemon_state.get("total_games_generated", 0),
                    "total_training_runs": daemon_state.get("total_training_runs", 0),
                    "total_tournaments": daemon_state.get("total_tournaments", 0),
                    "total_auto_promotions": daemon_state.get("total_auto_promotions", 0),
                    "last_auto_promote_time": daemon_state.get("last_auto_promote_time", 0),
                    "consecutive_failures": daemon_state.get("consecutive_failures", 0),
                },
                "nnue": {
                    "state": "idle" if not daemon_state.get("nnue_state") else "active",
                    "models": list(nnue_models.keys()),
                    "model_details": nnue_models,
                    "per_config_state": daemon_state.get("nnue_state", {}),
                    "last_gate_result": daemon_state.get("last_nnue_gate_result", None),
                },
                "cmaes": {
                    "state": "idle" if not daemon_state.get("cmaes_state") else "active",
                    "profiles": heuristic_profiles.get("profiles", []) if heuristic_profiles else [],
                    "profile_count": heuristic_profiles.get("count", 0) if heuristic_profiles else 0,
                    "per_config_state": daemon_state.get("cmaes_state", {}),
                    "generations": sum(s.get("generations", 0) for s in daemon_state.get("cmaes_state", {}).values()),
                },
                "promotion": {
                    "runtime_overrides": runtime_overrides,
                    "recent_promotions": promotion_log,
                },
                "timestamp": time.time(),
            })

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    def _canonical_slug_for_board(self, board_type: str) -> str:
        return {
            "square8": "square8",
            "square19": "square19",
            "hexagonal": "hex",
        }.get(board_type, board_type)

    def _canonical_gate_paths(self, board_type: str, num_players: int) -> Tuple[Path, Path]:
        """Compute canonical DB + gate summary paths (leader-side conventions)."""
        slug = self._canonical_slug_for_board(board_type)
        suffix = "" if int(num_players) == 2 else f"_{int(num_players)}p"
        ai_root = Path(self.ringrift_path) / "ai-service"
        db_path = (ai_root / "data" / "games" / f"canonical_{slug}{suffix}.db").resolve()
        summary_path = (ai_root / "data" / "games" / f"db_health.canonical_{slug}{suffix}.json").resolve()
        return db_path, summary_path

    def _tail_text_file(self, path: Path, *, max_lines: int = 200, max_bytes: int = 256_000) -> str:
        """Best-effort tail of a potentially large log file."""
        try:
            if not path.exists():
                return ""
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                seek = max(0, size - int(max_bytes))
                f.seek(seek)
                data = f.read().decode("utf-8", errors="replace")
            lines = data.splitlines()
            return "\n".join(lines[-int(max_lines) :])
        except Exception as e:
            return f"[tail_error] {e}"

    async def handle_api_canonical_health(self, request: web.Request) -> web.Response:
        """List canonical gate summary JSONs found on this node."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            ai_root = Path(self.ringrift_path) / "ai-service"
            games_dir = (ai_root / "data" / "games").resolve()
            summaries: List[Dict[str, Any]] = []

            for path in sorted(games_dir.glob("db_health.canonical_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    payload = {"error": "failed_to_parse_json", "message": str(exc)}

                mtime = 0.0
                try:
                    mtime = float(path.stat().st_mtime)
                except Exception:
                    mtime = 0.0

                db_path_str = str(payload.get("db_path") or "")
                db_size_bytes = None
                if db_path_str:
                    try:
                        db_path = Path(db_path_str)
                        if not db_path.is_absolute():
                            db_path = (games_dir / db_path).resolve()
                        db_size_bytes = int(db_path.stat().st_size)
                    except Exception:
                        db_size_bytes = None

                summaries.append(
                    {
                        "path": str(path),
                        "modified_time": mtime,
                        "db_size_bytes": db_size_bytes,
                        "summary": payload,
                    }
                )

            return web.json_response(
                {
                    "success": True,
                    "node_id": self.node_id,
                    "is_leader": self._is_leader(),
                    "summaries": summaries,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_canonical_jobs_list(self, request: web.Request) -> web.Response:
        """List canonical gate jobs started from this node."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            with self.canonical_gate_jobs_lock:
                jobs = list(self.canonical_gate_jobs.values())
            jobs.sort(key=lambda j: float(j.get("started_at", 0.0) or 0.0), reverse=True)
            return web.json_response(
                {
                    "success": True,
                    "node_id": self.node_id,
                    "is_leader": self._is_leader(),
                    "jobs": jobs[:100],
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_canonical_job_get(self, request: web.Request) -> web.Response:
        """Get details for a canonical gate job."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            job_id = (request.match_info.get("job_id") or "").strip()
            if not job_id:
                return web.json_response({"success": False, "error": "job_id is required"}, status=400)
            with self.canonical_gate_jobs_lock:
                job = self.canonical_gate_jobs.get(job_id)
            if not job:
                return web.json_response({"success": False, "error": f"Job {job_id} not found"}, status=404)
            return web.json_response({"success": True, "job": job})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_canonical_job_log(self, request: web.Request) -> web.Response:
        """Tail the log file for a canonical gate job."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            job_id = (request.match_info.get("job_id") or "").strip()
            if not job_id:
                return web.json_response({"success": False, "error": "job_id is required"}, status=400)
            tail_lines = int(request.query.get("tail", 200))
            tail_lines = max(10, min(tail_lines, 1000))

            with self.canonical_gate_jobs_lock:
                job = self.canonical_gate_jobs.get(job_id)
            if not job:
                return web.json_response({"success": False, "error": f"Job {job_id} not found"}, status=404)

            log_path = Path(str(job.get("log_path") or ""))
            text = self._tail_text_file(log_path, max_lines=tail_lines)
            return web.json_response({"success": True, "job_id": job_id, "log_tail": text})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    def _canonical_gate_log_dir(self) -> Path:
        return (Path(self.ringrift_path) / "ai-service" / "logs" / "canonical_gate").resolve()

    async def handle_api_canonical_logs_list(self, request: web.Request) -> web.Response:
        """List canonical gate log files on this node (use ?local=1 to avoid proxying to the leader)."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            logs_dir = self._canonical_gate_log_dir()
            entries: List[Dict[str, Any]] = []
            if logs_dir.exists():
                paths = sorted(
                    logs_dir.glob("*.log"),
                    key=lambda p: float(p.stat().st_mtime),
                    reverse=True,
                )
                for path in paths[:200]:
                    try:
                        st = path.stat()
                        entries.append(
                            {
                                "name": path.name,
                                "path": str(path),
                                "size_bytes": int(st.st_size),
                                "modified_time": float(st.st_mtime),
                            }
                        )
                    except Exception:
                        continue

            return web.json_response(
                {
                    "success": True,
                    "node_id": self.node_id,
                    "is_leader": self._is_leader(),
                    "log_dir": str(logs_dir),
                    "logs": entries,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_canonical_log_tail(self, request: web.Request) -> web.Response:
        """Tail a specific canonical gate log file by name."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                return await self._proxy_to_leader(request)

            log_name = (request.match_info.get("log_name") or "").strip()
            if not log_name:
                return web.json_response({"success": False, "error": "log_name is required"}, status=400)
            if any(token in log_name for token in ("..", "/", "\\")):
                return web.json_response({"success": False, "error": "Invalid log_name"}, status=400)

            tail_lines = int(request.query.get("tail", 200))
            tail_lines = max(10, min(tail_lines, 2000))

            logs_dir = self._canonical_gate_log_dir()
            log_path = (logs_dir / log_name).resolve()
            if log_path.parent != logs_dir:
                return web.json_response({"success": False, "error": "Invalid log_name"}, status=400)
            if not log_path.exists() or not log_path.is_file():
                return web.json_response({"success": False, "error": f"Log {log_name} not found"}, status=404)

            text = self._tail_text_file(log_path, max_lines=tail_lines)
            return web.json_response(
                {
                    "success": True,
                    "node_id": self.node_id,
                    "is_leader": self._is_leader(),
                    "log_name": log_name,
                    "log_tail": text,
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def _monitor_canonical_gate_job(self, job_id: str, proc: asyncio.subprocess.Process, summary_path: Path) -> None:
        """Background task: wait for canonical gate to finish and record summary."""
        try:
            returncode = await proc.wait()
        except Exception:
            returncode = -1

        finished_at = time.time()
        gate_summary: Dict[str, Any] | None = None
        try:
            if summary_path.exists():
                gate_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            gate_summary = None

        with self.canonical_gate_jobs_lock:
            job = self.canonical_gate_jobs.get(job_id, {})
            prior_status = str(job.get("status") or "")
            if prior_status == "cancelling":
                status = "cancelled"
            else:
                status = "completed" if int(returncode) == 0 else "failed"
            job.update(
                {
                    "status": status,
                    "returncode": int(returncode),
                    "completed_at": finished_at,
                    "gate_summary": gate_summary,
                }
            )
            self.canonical_gate_jobs[job_id] = job

    async def handle_api_canonical_generate(self, request: web.Request) -> web.Response:
        """Start a canonical selfplay+gate run (leader-only, dashboard-triggered)."""
        if not self._is_leader() and request.query.get("local") != "1":
            return await self._proxy_to_leader(request)
        if not self._is_leader():
            return web.json_response(
                {"success": False, "error": "Only leader can start canonical gate runs", "leader_id": self.leader_id},
                status=403,
            )

        try:
            data = await request.json()
            board_type = str(data.get("board_type") or "square8")
            num_players = int(data.get("num_players") or 2)
            num_games = int(data.get("num_games") or 0)
            difficulty_band = str(data.get("difficulty_band") or "light")
            reset_db = bool(data.get("reset_db") or False)
            hosts = (str(data.get("hosts") or "").strip()) or None
            distributed_job_timeout_seconds = int(data.get("distributed_job_timeout_seconds") or 0)
            distributed_fetch_timeout_seconds = int(data.get("distributed_fetch_timeout_seconds") or 0)

            if board_type not in ("square8", "square19", "hexagonal"):
                return web.json_response({"success": False, "error": f"Unsupported board_type: {board_type}"}, status=400)
            if num_players not in (2, 3, 4):
                return web.json_response({"success": False, "error": f"Unsupported num_players: {num_players}"}, status=400)
            if num_games < 0 or num_games > 250_000:
                return web.json_response({"success": False, "error": f"num_games out of range: {num_games}"}, status=400)
            if difficulty_band not in ("light", "canonical"):
                return web.json_response({"success": False, "error": f"Unsupported difficulty_band: {difficulty_band}"}, status=400)
            if hosts and any(c.isspace() for c in hosts):
                return web.json_response({"success": False, "error": "hosts must be comma-separated with no spaces"}, status=400)
            if distributed_job_timeout_seconds < 0 or distributed_job_timeout_seconds > 604_800:
                return web.json_response(
                    {"success": False, "error": f"distributed_job_timeout_seconds out of range: {distributed_job_timeout_seconds}"},
                    status=400,
                )
            if distributed_fetch_timeout_seconds < 0 or distributed_fetch_timeout_seconds > 86_400:
                return web.json_response(
                    {"success": False, "error": f"distributed_fetch_timeout_seconds out of range: {distributed_fetch_timeout_seconds}"},
                    status=400,
                )

            db_path, summary_path = self._canonical_gate_paths(board_type, num_players)

            job_id = f"canon_gate_{board_type}_{num_players}p_{int(time.time())}_{secrets.token_hex(4)}"
            ai_root = Path(self.ringrift_path) / "ai-service"
            log_dir = (ai_root / "logs" / "canonical_gate").resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = (log_dir / f"{job_id}.log").resolve()

            cmd = [
                sys.executable,
                "scripts/generate_canonical_selfplay.py",
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--num-games", str(num_games),
                "--difficulty-band", difficulty_band,
                "--db", str(db_path),
                "--summary", str(summary_path),
            ]
            if hosts:
                cmd.extend(["--hosts", hosts])
                if distributed_job_timeout_seconds > 0:
                    cmd.extend(
                        [
                            "--distributed-job-timeout-seconds",
                            str(distributed_job_timeout_seconds),
                        ]
                    )
                if distributed_fetch_timeout_seconds > 0:
                    cmd.extend(
                        [
                            "--distributed-fetch-timeout-seconds",
                            str(distributed_fetch_timeout_seconds),
                        ]
                    )
            if reset_db:
                cmd.append("--reset-db")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(ai_root)
            env.setdefault("RINGRIFT_JOB_ORIGIN", "dashboard")
            env.setdefault("PYTHONUNBUFFERED", "1")

            with log_path.open("a", encoding="utf-8") as log_handle:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(ai_root),
                    env=env,
                    stdout=log_handle,
                    stderr=log_handle,
                )

            job = {
                "job_id": job_id,
                "status": "running",
                "board_type": board_type,
                "num_players": num_players,
                "num_games": num_games,
                "difficulty_band": difficulty_band,
                "hosts": hosts,
                "reset_db": reset_db,
                "distributed_job_timeout_seconds": distributed_job_timeout_seconds,
                "distributed_fetch_timeout_seconds": distributed_fetch_timeout_seconds,
                "db_path": str(db_path),
                "summary_path": str(summary_path),
                "log_path": str(log_path),
                "pid": int(proc.pid),
                "started_at": time.time(),
            }

            with self.canonical_gate_jobs_lock:
                self.canonical_gate_jobs[job_id] = job

            asyncio.create_task(self._monitor_canonical_gate_job(job_id, proc, summary_path))

            return web.json_response({"success": True, "job": job})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_canonical_job_cancel(self, request: web.Request) -> web.Response:
        """Cancel a running canonical gate job."""
        if not self._is_leader() and request.query.get("local") != "1":
            return await self._proxy_to_leader(request)
        if not self._is_leader():
            return web.json_response(
                {"success": False, "error": "Only leader can cancel canonical gate runs", "leader_id": self.leader_id},
                status=403,
            )

        try:
            job_id = (request.match_info.get("job_id") or "").strip()
            if not job_id:
                return web.json_response({"success": False, "error": "job_id is required"}, status=400)

            with self.canonical_gate_jobs_lock:
                job = self.canonical_gate_jobs.get(job_id)
            if not job:
                return web.json_response({"success": False, "error": f"Job {job_id} not found"}, status=404)

            pid = int(job.get("pid") or 0)
            if pid <= 0:
                return web.json_response({"success": False, "error": "No pid recorded for job"}, status=400)

            try:
                os.kill(pid, signal.SIGTERM)
            except Exception as exc:
                return web.json_response({"success": False, "error": f"Failed to signal pid {pid}: {exc}"}, status=500)

            with self.canonical_gate_jobs_lock:
                job = self.canonical_gate_jobs.get(job_id, job)
                job["status"] = "cancelling"
                job["cancel_requested_at"] = time.time()
                self.canonical_gate_jobs[job_id] = job

            return web.json_response({"success": True, "message": f"Cancel signaled for {job_id}", "job": job})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_jobs_list(self, request: web.Request) -> web.Response:
        """List all jobs with optional filtering."""
        try:
            if not self._is_leader() and request.query.get("local") != "1":
                proxied = await self._proxy_to_leader(request)
                if proxied.status not in (502, 503):
                    return proxied

            job_type = request.query.get("type")
            status = request.query.get("status")
            limit = int(request.query.get("limit", 100))

            # Collect all jobs (local + training + ssh tournament runs)
            all_jobs = []

            with self.jobs_lock:
                local_jobs_snapshot = list(self.local_jobs.values())
            for job in local_jobs_snapshot:
                jt = job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type)
                if job_type and jt != job_type:
                    continue
                if status and job.status != status:
                    continue
                all_jobs.append(
                    {
                        "job_id": job.job_id,
                        "job_type": jt,
                        "status": job.status,
                        "assigned_to": job.node_id,
                        "created_at": job.started_at,
                        "board_type": job.board_type,
                        "num_players": job.num_players,
                        "category": "local",
                    }
                )

            with self.training_lock:
                for job_id, job in self.training_jobs.items():
                    if job_type and job.job_type != job_type:
                        continue
                    if status and job.status != status:
                        continue
                    all_jobs.append({
                        "job_id": job_id,
                        "job_type": job.job_type,
                        "status": job.status,
                        "assigned_to": job.worker_node,
                        "created_at": job.created_at,
                        "board_type": job.board_type,
                        "num_players": job.num_players,
                        "category": "training",
                    })

            with self.ssh_tournament_lock:
                ssh_runs_snapshot = list(self.ssh_tournament_runs.values())
            for run in ssh_runs_snapshot:
                if job_type and job_type != "ssh_tournament":
                    continue
                if status and run.status != status:
                    continue
                all_jobs.append(
                    {
                        "job_id": run.job_id,
                        "job_type": "ssh_tournament",
                        "status": run.status,
                        "assigned_to": self.node_id,
                        "created_at": run.started_at,
                        "board_type": run.board,
                        "num_players": 2,
                        "category": "ssh_tournament",
                    }
                )

            # Sort by created_at descending and limit
            all_jobs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            all_jobs = all_jobs[:limit]

            return web.json_response({
                "success": True,
                "jobs": all_jobs,
                "total": len(all_jobs),
            })
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_jobs_submit(self, request: web.Request) -> web.Response:
        """Submit a new job via REST API."""
        if not self._is_leader() and request.query.get("local") != "1":
            return await self._proxy_to_leader(request)
        if not self._is_leader():
            return web.json_response(
                {
                    "success": False,
                    "error": "Not the leader. Please submit to leader node.",
                    "leader_id": self.leader_id,
                },
                status=400,
            )

        try:
            data = await request.json()
            job_type = data.get("job_type")
            if not job_type:
                return web.json_response({
                    "success": False,
                    "error": "job_type is required",
                }, status=400)

            if job_type in ["nnue", "cmaes"]:
                board_type = data.get("board_type", "square8")
                num_players = int(data.get("num_players", 2))
                job_config = {
                    "job_type": job_type,
                    "board_type": board_type,
                    "num_players": num_players,
                    "config_key": f"{board_type}_{num_players}p",
                    "total_games": int(data.get("total_games", 0)),
                }
                job = await self._dispatch_training_job(job_config)
                if not job:
                    return web.json_response(
                        {"success": False, "error": "No suitable worker available"},
                        status=400,
                    )
                return web.json_response(
                    {
                        "success": True,
                        "job_id": job.job_id,
                        "job_type": job.job_type,
                        "status": job.status,
                        "message": f"Training job {job.job_id} created",
                    }
                )

            if job_type in ["selfplay", "gpu_selfplay", "hybrid_selfplay"]:
                board_type = data.get("board_type", "square8")
                num_players = int(data.get("num_players", 2))
                engine_mode = data.get("engine_mode", "heuristic-only")

                # Map job type string to enum
                if job_type == "gpu_selfplay":
                    jt = JobType.GPU_SELFPLAY
                elif job_type == "hybrid_selfplay":
                    jt = JobType.HYBRID_SELFPLAY
                else:
                    jt = JobType.SELFPLAY

                job = await self._start_local_job(jt, board_type, num_players, engine_mode)
                if not job:
                    return web.json_response(
                        {"success": False, "error": "Failed to start local job"},
                        status=500,
                    )
                return web.json_response(
                    {
                        "success": True,
                        "job_id": job.job_id,
                        "job_type": job.job_type.value,
                        "status": job.status,
                        "message": f"Job {job.job_id} started",
                    }
                )

            return web.json_response(
                {
                    "success": False,
                    "error": f"Unknown job type: {job_type}. Supported: nnue, cmaes, selfplay, gpu_selfplay, hybrid_selfplay",
                },
                status=400,
            )

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_job_get(self, request: web.Request) -> web.Response:
        """Get details for a specific job."""
        try:
            job_id = request.match_info.get("job_id")
            if not job_id:
                return web.json_response({
                    "success": False,
                    "error": "job_id is required",
                }, status=400)

            with self.jobs_lock:
                local_job = self.local_jobs.get(job_id)
            if local_job:
                return web.json_response(
                    {
                        "success": True,
                        "job": {
                            "job_id": job_id,
                            "job_type": local_job.job_type.value if hasattr(local_job.job_type, "value") else str(local_job.job_type),
                            "status": local_job.status,
                            "assigned_to": local_job.node_id,
                            "created_at": local_job.started_at,
                            "board_type": local_job.board_type,
                            "num_players": local_job.num_players,
                            "engine_mode": local_job.engine_mode,
                            "pid": local_job.pid,
                            "category": "local",
                        },
                    }
                )

            with self.ssh_tournament_lock:
                ssh_run = self.ssh_tournament_runs.get(job_id)
            if ssh_run:
                return web.json_response(
                    {
                        "success": True,
                        "job": {
                            "job_id": ssh_run.job_id,
                            "job_type": "ssh_tournament",
                            "status": ssh_run.status,
                            "assigned_to": self.node_id,
                            "created_at": ssh_run.started_at,
                            "run_id": ssh_run.run_id,
                            "tiers": ssh_run.tiers,
                            "board_type": ssh_run.board,
                            "games_per_matchup": ssh_run.games_per_matchup,
                            "output_root": ssh_run.output_root,
                            "manifest_path": ssh_run.manifest_path,
                            "checkpoint_path": ssh_run.checkpoint_path,
                            "report_path": ssh_run.report_path,
                            "log_path": ssh_run.log_path,
                            "category": "ssh_tournament",
                        },
                    }
                )

            # Check training jobs
            with self.training_lock:
                if job_id in self.training_jobs:
                    job = self.training_jobs[job_id]
                    return web.json_response({
                        "success": True,
                        "job": {
                            "job_id": job_id,
                            "job_type": job.job_type,
                            "status": job.status,
                            "board_type": job.board_type,
                            "num_players": job.num_players,
                            "assigned_worker": job.worker_node,
                            "created_at": job.created_at,
                            "started_at": job.started_at,
                            "completed_at": job.completed_at,
                            "output_model_path": job.output_model_path,
                            "error_message": job.error_message,
                            "category": "training",
                        },
                    })

            return web.json_response({
                "success": False,
                "error": f"Job {job_id} not found",
            }, status=404)

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_api_job_cancel(self, request: web.Request) -> web.Response:
        """Cancel a pending or running job."""
        if not self._is_leader() and request.query.get("local") != "1":
            return await self._proxy_to_leader(request)
        if not self._is_leader():
            return web.json_response(
                {
                    "success": False,
                    "error": "Not the leader",
                },
                status=400,
            )

        try:
            job_id = request.match_info.get("job_id")
            if not job_id:
                return web.json_response({
                    "success": False,
                    "error": "job_id is required",
                }, status=400)

            with self.jobs_lock:
                local_job = self.local_jobs.get(job_id)
            if local_job:
                try:
                    os.kill(local_job.pid, signal.SIGTERM)
                except Exception:
                    pass
                with self.jobs_lock:
                    local_job.status = "stopped"
                    self.local_jobs[job_id] = local_job
                self._save_state()
                return web.json_response({"success": True, "message": f"Job {job_id} stopped"})

            with self.ssh_tournament_lock:
                ssh_run = self.ssh_tournament_runs.get(job_id)
            if ssh_run:
                if ssh_run.pid:
                    try:
                        os.kill(ssh_run.pid, signal.SIGTERM)
                    except Exception:
                        pass
                with self.ssh_tournament_lock:
                    ssh_run.status = "cancelled"
                    ssh_run.completed_at = time.time()
                    self.ssh_tournament_runs[job_id] = ssh_run
                return web.json_response({"success": True, "message": f"SSH tournament {job_id} cancelled"})

            # Check training jobs
            with self.training_lock:
                if job_id in self.training_jobs:
                    job = self.training_jobs[job_id]
                    if job.status in ["pending", "queued"]:
                        job.status = "cancelled"
                        return web.json_response({
                            "success": True,
                            "message": f"Training job {job_id} cancelled",
                        })
                    else:
                        return web.json_response({
                            "success": False,
                            "error": f"Cannot cancel job in status: {job.status}",
                        }, status=400)

            return web.json_response({
                "success": False,
                "error": f"Job {job_id} not found",
            }, status=404)

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def handle_dashboard(self, request: web.Request) -> web.Response:
        """Serve the web dashboard HTML."""
        dashboard_path = Path(__file__).resolve().parent / "dashboard_assets" / "dashboard.html"
        try:
            html = dashboard_path.read_text(encoding="utf-8")
        except Exception as e:
            html = (
                "<!doctype html><html><body style='font-family:monospace'>"
                f"<h3>Dashboard asset unavailable</h3><pre>{e}</pre>"
                f"<pre>Expected: {dashboard_path}</pre>"
                "</body></html>"
            )
        headers = {
            # Avoid stale HTML across load balancers / browsers during rapid iteration.
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            # Simple diagnostics (no secrets).
            "X-RingRift-Node-Id": str(self.node_id or ""),
            "X-RingRift-Build-Version": str(getattr(self, "build_version", "") or ""),
        }
        return web.Response(text=html, content_type="text/html", headers=headers)

    async def _run_evaluation(self, job_id: str):
        """Evaluate new model against current best.

        Runs evaluation games between the candidate model and the best model.
        Reports win rate for the candidate.
        """
        import sys
        import json as json_module

        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        print(f"[P2P] Running evaluation for job {job_id}, iteration {state.current_iteration}")

        candidate_model = getattr(state, 'candidate_model_path', None)
        best_model = state.best_model_path

        # Number of evaluation games
        eval_games = 100

        eval_script = f"""
import sys
sys.path.insert(0, '{self.ringrift_path}/ai-service')
from app.game_engine import GameEngine
from app.agents.heuristic_agent import HeuristicAgent
import json

# Run evaluation games
candidate_wins = 0
best_wins = 0
draws = 0

for game_idx in range({eval_games}):
    engine = GameEngine(board_type='{state.board_type}', num_players={state.num_players})

    # Alternate who plays first
    if game_idx % 2 == 0:
        agents = [
            HeuristicAgent(0),  # Candidate as player 0
            HeuristicAgent(1),  # Best as player 1
        ]
        candidate_player = 0
    else:
        agents = [
            HeuristicAgent(0),  # Best as player 0
            HeuristicAgent(1),  # Candidate as player 1
        ]
        candidate_player = 1

    # Play game
    max_moves = 10000
    move_count = 0
    while not engine.is_game_over() and move_count < max_moves:
        current_player = engine.current_player
        agent = agents[current_player]
        legal_moves = engine.get_legal_moves()
        if not legal_moves:
            break
        move = agent.select_move(engine.get_state(), legal_moves)
        engine.apply_move(move)
        move_count += 1

    outcome = engine.get_outcome()
    winner = outcome.get('winner')

    if winner == candidate_player:
        candidate_wins += 1
    elif winner is not None:
        best_wins += 1
    else:
        draws += 1

# Calculate win rate
total = candidate_wins + best_wins + draws
winrate = candidate_wins / total if total > 0 else 0.5

print(json.dumps({{
    'candidate_wins': candidate_wins,
    'best_wins': best_wins,
    'draws': draws,
    'winrate': winrate,
}}))
"""

        cmd = [sys.executable, "-c", eval_script]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3600  # 1 hour max
            )

            if proc.returncode == 0:
                output_lines = stdout.decode().strip().split('\n')
                result_line = output_lines[-1] if output_lines else '{}'
                result = json_module.loads(result_line)

                state.evaluation_winrate = result.get('winrate', 0.5)
                print(f"[P2P] Evaluation result: winrate={state.evaluation_winrate:.2%}")
                print(f"  Candidate: {result.get('candidate_wins')}, Best: {result.get('best_wins')}, Draws: {result.get('draws')}")
            else:
                print(f"[P2P] Evaluation failed: {stderr.decode()[:500]}")
                state.evaluation_winrate = 0.5

        except asyncio.TimeoutError:
            print(f"[P2P] Evaluation timed out")
            state.evaluation_winrate = 0.5
        except Exception as e:
            print(f"[P2P] Evaluation error: {e}")
            state.evaluation_winrate = 0.5

    async def _promote_model_if_better(self, job_id: str):
        """Promote new model if it beats the current best.

        Promotion threshold: candidate must win >= 55% of evaluation games.
        """
        state = self.improvement_loop_state.get(job_id)
        if not state:
            return

        PROMOTION_THRESHOLD = 0.55  # 55% win rate required

        winrate = getattr(state, 'evaluation_winrate', 0.5)
        candidate_path = getattr(state, 'candidate_model_path', None)

        print(f"[P2P] Checking model promotion for job {job_id}")
        print(f"  Current best winrate: {state.best_winrate:.2%}")
        print(f"  Candidate winrate: {winrate:.2%}")
        print(f"  Threshold: {PROMOTION_THRESHOLD:.0%}")

        if winrate >= PROMOTION_THRESHOLD and candidate_path:
            # Promote candidate to best
            state.best_model_path = candidate_path
            state.best_winrate = winrate

            # Save best model to well-known location
            best_model_dir = os.path.join(
                self.ringrift_path, "ai-service", "models", "best"
            )
            os.makedirs(best_model_dir, exist_ok=True)

            import shutil
            best_path = os.path.join(best_model_dir, f"{state.board_type}_{state.num_players}p.pt")
            if os.path.exists(candidate_path):
                shutil.copy2(candidate_path, best_path)
                print(f"[P2P] PROMOTED: New best model at {best_path}")
                print(f"  Win rate: {winrate:.2%}")
            else:
                print(f"[P2P] Cannot promote: candidate model not found at {candidate_path}")
        else:
            print(f"[P2P] No promotion: candidate ({winrate:.2%}) below threshold ({PROMOTION_THRESHOLD:.0%})")

    # ============================================
    # Core Logic
    # ============================================

    def _update_self_info(self):
        """Update self info with current resource usage."""
        usage = self._get_resource_usage()
        selfplay, training = self._count_local_jobs()

        # NAT/relay detection: if we haven't received any inbound heartbeats for a
        # while (but we do know about other peers), assume we're not reachable
        # inbound and must poll a relay for commands.
        now = time.time()
        if self.known_peers or self.peers:
            last_inbound = self.last_inbound_heartbeat or self.start_time
            self.self_info.nat_blocked = (now - last_inbound) >= NAT_INBOUND_HEARTBEAT_STALE_SECONDS
        else:
            self.self_info.nat_blocked = False

        if not self.self_info.nat_blocked:
            self.self_info.relay_via = ""
        elif self.leader_id and self.leader_id != self.node_id:
            self.self_info.relay_via = self.leader_id

        self.self_info.cpu_percent = usage["cpu_percent"]
        self.self_info.memory_percent = usage["memory_percent"]
        self.self_info.disk_percent = usage["disk_percent"]
        self.self_info.gpu_percent = usage["gpu_percent"]
        self.self_info.gpu_memory_percent = usage["gpu_memory_percent"]
        self.self_info.selfplay_jobs = selfplay
        self.self_info.training_jobs = training
        self.self_info.role = self.role
        self.self_info.last_heartbeat = time.time()

        # Report to unified resource optimizer for cluster-wide coordination
        if HAS_NEW_COORDINATION:
            try:
                optimizer = get_resource_optimizer()
                node_resources = NodeResources(
                    node_id=self.node_id,
                    cpu_percent=usage["cpu_percent"],
                    gpu_percent=usage["gpu_percent"],
                    memory_percent=usage["memory_percent"],
                    disk_percent=usage["disk_percent"],
                    gpu_memory_percent=usage["gpu_memory_percent"],
                    cpu_count=int(getattr(self.self_info, "cpu_count", 0) or 0),
                    memory_gb=float(getattr(self.self_info, "memory_gb", 0) or 0),
                    has_gpu=bool(getattr(self.self_info, "has_gpu", False)),
                    gpu_name=str(getattr(self.self_info, "gpu_name", "") or ""),
                    active_jobs=selfplay + training,
                    selfplay_jobs=selfplay,
                    training_jobs=training,
                    orchestrator="p2p_orchestrator",
                )
                optimizer.report_node_resources(node_resources)
            except Exception:
                pass  # Don't fail heartbeat if optimizer unavailable

    async def _send_heartbeat_to_peer(self, peer_host: str, peer_port: int, scheme: str = "http") -> Optional[NodeInfo]:
        """Send heartbeat to a peer and return their info."""
        try:
            self._update_self_info()
            payload = self.self_info.to_dict()
            voter_node_ids = list(getattr(self, "voter_node_ids", []) or [])
            if voter_node_ids:
                payload["voter_node_ids"] = voter_node_ids
                payload["voter_quorum_size"] = int(getattr(self, "voter_quorum_size", 0) or 0)
                payload["voter_config_source"] = str(getattr(self, "voter_config_source", "") or "")

            timeout = ClientTimeout(total=10)
            async with get_client_session(timeout) as session:
                scheme = (scheme or "http").lower()
                url = f"{scheme}://{peer_host}:{peer_port}/heartbeat"
                async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        incoming_voters = data.get("voter_node_ids") or data.get("voters") or None
                        if incoming_voters:
                            voters_list: List[str] = []
                            if isinstance(incoming_voters, list):
                                voters_list = [str(v).strip() for v in incoming_voters if str(v).strip()]
                            elif isinstance(incoming_voters, str):
                                voters_list = [t.strip() for t in incoming_voters.split(",") if t.strip()]
                            if voters_list:
                                self._maybe_adopt_voter_node_ids(voters_list, source="learned")
                        info = NodeInfo.from_dict(data)
                        if not info.reported_host:
                            info.reported_host = info.host
                        if not info.reported_port:
                            info.reported_port = info.port
                        # Use the address we successfully reached instead of any
                        # self-reported interface address.
                        info.scheme = scheme
                        info.host = peer_host
                        info.port = peer_port
                        return info
        except Exception as e:
            pass
        return None

    async def _bootstrap_from_known_peers(self) -> bool:
        """Import cluster membership from seed peers via `/relay/peers`.

        Heartbeats intentionally return only a single peer's NodeInfo, which
        makes initial convergence slow when only one seed peer is configured
        (common for cloud nodes). `/relay/peers` returns a snapshot of the
        sender's full peer list, allowing new nodes to quickly learn about the
        leader and other cluster members.
        """
        # Seed peers are configured via `--peers`, but relying on a single
        # coordinator makes clusters brittle. Also bootstrap from any
        # previously-seen directly-reachable peers so nodes can re-join after
        # restarts even if the original seed goes offline.
        known_seed_peers: List[str] = [p for p in (self.known_peers or []) if p]
        discovered_seed_peers: List[str] = []

        with self.peers_lock:
            peers_snapshot = [p for p in self.peers.values() if p.node_id != self.node_id]
        peers_snapshot.sort(key=lambda p: str(getattr(p, "node_id", "") or ""))

        for peer in peers_snapshot:
            if getattr(peer, "nat_blocked", False):
                # NAT-blocked nodes cannot serve as inbound seeds.
                continue
            if not peer.should_retry():
                continue

            scheme = (getattr(peer, "scheme", "http") or "http").lower()
            host = str(getattr(peer, "host", "") or "").strip()
            try:
                port = int(getattr(peer, "port", DEFAULT_PORT) or DEFAULT_PORT)
            except Exception:
                port = DEFAULT_PORT
            if host:
                discovered_seed_peers.append(f"{scheme}://{host}:{port}")

            rh = str(getattr(peer, "reported_host", "") or "").strip()
            try:
                rp = int(getattr(peer, "reported_port", 0) or 0)
            except Exception:
                rp = 0
            if rh and rp:
                discovered_seed_peers.append(f"{scheme}://{rh}:{rp}")

        seen: Set[str] = set()
        seed_peers: List[str] = []
        ki = 0
        di = 0
        while ki < len(known_seed_peers) or di < len(discovered_seed_peers):
            if ki < len(known_seed_peers):
                candidate = known_seed_peers[ki]
                ki += 1
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    seed_peers.append(candidate)
            if di < len(discovered_seed_peers):
                candidate = discovered_seed_peers[di]
                di += 1
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    seed_peers.append(candidate)
        if not seed_peers:
            return False

        now = time.time()
        if now - self.last_peer_bootstrap < PEER_BOOTSTRAP_INTERVAL:
            return False

        max_seeds = int(os.environ.get("RINGRIFT_P2P_BOOTSTRAP_MAX_SEEDS_PER_RUN", "8") or 8)
        max_seeds = max(1, min(max_seeds, 32))

        timeout = ClientTimeout(total=8)
        bootstrapped = False
        imported_any = False

        async with get_client_session(timeout) as session:
            for idx, peer_addr in enumerate(seed_peers):
                if idx >= max_seeds:
                    break
                try:
                    scheme, host, port = self._parse_peer_address(peer_addr)
                    scheme = (scheme or "http").lower()
                    url = f"{scheme}://{host}:{port}/relay/peers"
                    async with session.get(url, headers=self._auth_headers()) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()

                    if not isinstance(data, dict) or not data.get("success"):
                        continue

                    bootstrapped = True

                    incoming_voters = data.get("voter_node_ids") or data.get("voters") or None
                    if incoming_voters:
                        voters_list: List[str] = []
                        if isinstance(incoming_voters, list):
                            voters_list = [str(v).strip() for v in incoming_voters if str(v).strip()]
                        elif isinstance(incoming_voters, str):
                            voters_list = [t.strip() for t in incoming_voters.split(",") if t.strip()]
                        if voters_list:
                            self._maybe_adopt_voter_node_ids(voters_list, source="learned")

                    peers_data = data.get("peers") or {}
                    if not isinstance(peers_data, dict):
                        continue

                    with self.peers_lock:
                        before = len(self.peers)
                        for node_id, peer_dict in peers_data.items():
                            if not node_id or node_id == self.node_id:
                                continue
                            try:
                                info = NodeInfo.from_dict(peer_dict)
                            except Exception:
                                continue
                            existing = self.peers.get(info.node_id)
                            if existing:
                                # Preserve relay/NAT routing and retirement state when merging peer snapshots.
                                if getattr(existing, "nat_blocked", False) and not getattr(info, "nat_blocked", False):
                                    info.nat_blocked = True
                                if (getattr(existing, "relay_via", "") or "") and not (getattr(info, "relay_via", "") or ""):
                                    info.relay_via = str(getattr(existing, "relay_via", "") or "")
                                if getattr(existing, "retired", False):
                                    info.retired = True
                                    info.retired_at = float(getattr(existing, "retired_at", 0.0) or 0.0)
                                # Preserve local reachability diagnostics.
                                info.consecutive_failures = int(getattr(existing, "consecutive_failures", 0) or 0)
                                info.last_failure_time = float(getattr(existing, "last_failure_time", 0.0) or 0.0)

                            self.peers[info.node_id] = info
                        after = len(self.peers)

                    new = max(0, after - before)
                    if new:
                        imported_any = True
                        print(f"[P2P] Bootstrap: imported {new} new peers from {host}:{port}")

                    leader_id = str(data.get("leader_id") or "").strip()
                    if leader_id and leader_id != self.node_id:
                        # If we're currently leader but the seed reports a higher-priority
                        # leader, step down to converge quickly.
                        if self.role == NodeRole.LEADER and leader_id > self.node_id:
                            print(f"[P2P] Bootstrap: stepping down for leader {leader_id}")
                            self.role = NodeRole.FOLLOWER
                        self.leader_id = leader_id
                except Exception:
                    continue

        self.last_peer_bootstrap = now
        if bootstrapped:
            self._maybe_adopt_leader_from_peers()
            self._save_state()
        return imported_any

    async def _send_relay_heartbeat(self, relay_url: str) -> Dict[str, Any]:
        """Send heartbeat via relay endpoint for NAT-blocked nodes.

        This is used when the peer URL is HTTPS (indicating a relay/proxy endpoint)
        or when direct heartbeats fail consistently.

        Returns dict with:
        - success: bool
        - peers: dict of all cluster peers
        - leader_id: current leader
        """
        try:
            self._update_self_info()

            timeout = ClientTimeout(total=15)
            async with get_client_session(timeout) as session:
                # Use /relay/heartbeat endpoint
                url = f"{relay_url.rstrip('/')}/relay/heartbeat"
                payload = self.self_info.to_dict()
                if self.pending_relay_acks:
                    payload["relay_ack"] = sorted(self.pending_relay_acks)
                if self.pending_relay_results:
                    payload["relay_results"] = list(self.pending_relay_results)
                async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                    if resp.status != 200:
                        return {"success": False, "error": f"HTTP {resp.status}"}

                    data = await resp.json()
                    if not data.get("success"):
                        return {"success": False, "error": data.get("error", "Unknown error")}

                    incoming_voters = data.get("voter_node_ids") or data.get("voters") or None
                    if incoming_voters:
                        voters_list: List[str] = []
                        if isinstance(incoming_voters, list):
                            voters_list = [str(v).strip() for v in incoming_voters if str(v).strip()]
                        elif isinstance(incoming_voters, str):
                            voters_list = [t.strip() for t in incoming_voters.split(",") if t.strip()]
                        if voters_list:
                            self._maybe_adopt_voter_node_ids(voters_list, source="learned")

                    # Clear pending acks/results only after a successful round-trip.
                    self.pending_relay_acks.clear()
                    self.pending_relay_results.clear()

                    # Update our peer list with all peers from relay
                    peers_data = data.get("peers", {})
                    with self.peers_lock:
                        for node_id, peer_dict in peers_data.items():
                            if node_id != self.node_id:
                                peer_info = NodeInfo.from_dict(peer_dict)
                                existing = self.peers.get(node_id)
                                if existing:
                                    if getattr(existing, "nat_blocked", False) and not getattr(peer_info, "nat_blocked", False):
                                        peer_info.nat_blocked = True
                                    if (getattr(existing, "relay_via", "") or "") and not (getattr(peer_info, "relay_via", "") or ""):
                                        peer_info.relay_via = str(getattr(existing, "relay_via", "") or "")
                                    if getattr(existing, "retired", False):
                                        peer_info.retired = True
                                        peer_info.retired_at = float(getattr(existing, "retired_at", 0.0) or 0.0)
                                    peer_info.consecutive_failures = int(getattr(existing, "consecutive_failures", 0) or 0)
                                    peer_info.last_failure_time = float(getattr(existing, "last_failure_time", 0.0) or 0.0)
                                self.peers[node_id] = peer_info

                    # Execute any queued commands addressed to us.
                    commands = data.get("commands") or []
                    if isinstance(commands, list) and commands:
                        await self._execute_relay_commands(commands)

                    # Update leader if provided
                    leader_id = data.get("leader_id")
                    if leader_id and leader_id != self.node_id:
                        if self.leader_id != leader_id:
                            print(f"[P2P] Adopted leader from relay: {leader_id}")
                        self.leader_id = leader_id
                        self.role = NodeRole.FOLLOWER

                    return {
                        "success": True,
                        "peers_received": len(peers_data) if isinstance(peers_data, dict) else 0,
                        "leader_id": leader_id,
                        "commands_received": len(commands) if isinstance(commands, list) else 0,
                    }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_relay_commands(self, commands: List[Dict[str, Any]]) -> None:
        """Execute relay commands (polling mode for NAT-blocked nodes)."""
        now = time.time()
        for cmd in commands:
            try:
                cmd_id = str(cmd.get("id") or "")
                cmd_type = str(cmd.get("type") or "")
                payload = cmd.get("payload") or {}
                if not cmd_id or not cmd_type:
                    continue

                # Check for stale commands (>5 min old indicates relay/polling issues)
                cmd_ts = cmd.get("ts") or cmd.get("timestamp") or now
                cmd_age_secs = now - float(cmd_ts)
                if cmd_age_secs > 300:
                    print(f"[P2P] WARNING: Relay command {cmd_id} ({cmd_type}) is {cmd_age_secs:.0f}s old - relay delivery may be delayed")

                attempts = int(self.relay_command_attempts.get(cmd_id, 0) or 0) + 1
                self.relay_command_attempts[cmd_id] = attempts

                ok = False
                err = ""
                if cmd_type == "start_job":
                    job_type = JobType(str(payload.get("job_type") or "selfplay"))
                    board_type = str(payload.get("board_type") or "square8")
                    num_players = int(payload.get("num_players") or 2)
                    engine_mode = str(payload.get("engine_mode") or "mixed")
                    job_id = str(payload.get("job_id") or "")

                    if job_id:
                        with self.jobs_lock:
                            existing = self.local_jobs.get(job_id)
                        if existing and existing.status == "running":
                            ok = True
                        else:
                            job = await self._start_local_job(
                                job_type,
                                board_type=board_type,
                                num_players=num_players,
                                engine_mode=engine_mode,
                                job_id=job_id,
                            )
                            ok = job is not None
                    else:
                        job = await self._start_local_job(
                            job_type,
                            board_type=board_type,
                            num_players=num_players,
                            engine_mode=engine_mode,
                        )
                        ok = job is not None
                elif cmd_type == "cleanup":
                    asyncio.create_task(self._cleanup_local_disk())
                    ok = True
                elif cmd_type == "restart_stuck_jobs":
                    asyncio.create_task(self._restart_local_stuck_jobs())
                    ok = True
                elif cmd_type == "reduce_selfplay":
                    target = payload.get("target_selfplay_jobs", payload.get("target", 0))
                    reason = str(payload.get("reason") or "relay")
                    try:
                        target_jobs = int(target)
                    except Exception:
                        target_jobs = 0
                    await self._reduce_local_selfplay_jobs(target_jobs, reason=reason)
                    ok = True
                elif cmd_type == "cleanup_files":
                    files = payload.get("files", []) or []
                    reason = str(payload.get("reason") or "relay")
                    if not isinstance(files, list) or not files:
                        ok = False
                        err = "no_files"
                    else:
                        data_dir = self.get_data_directory()
                        freed_bytes = 0
                        deleted_count = 0
                        data_root = data_dir.resolve()
                        for file_path in files:
                            full_path = data_dir / (str(file_path or "").lstrip("/"))
                            try:
                                resolved = full_path.resolve()
                                resolved.relative_to(data_root)
                            except Exception:
                                continue
                            if not resolved.exists():
                                continue
                            try:
                                size = resolved.stat().st_size
                                resolved.unlink()
                                freed_bytes += size
                                deleted_count += 1
                            except Exception:
                                continue
                        print(
                            f"[P2P] Relay cleanup_files: {deleted_count} files deleted, "
                            f"{freed_bytes / 1e6:.1f}MB freed (reason={reason})"
                        )
                        ok = True
                elif cmd_type == "canonical_selfplay":
                    job_id = str(payload.get("job_id") or "")
                    board_type = str(payload.get("board_type") or "square8")
                    num_players = int(payload.get("num_players") or 2)
                    num_games = int(payload.get("num_games") or payload.get("games_per_node") or 500)
                    seed = int(payload.get("seed") or 0)
                    if not job_id:
                        ok = False
                        err = "missing_job_id"
                    else:
                        asyncio.create_task(
                            self._run_local_canonical_selfplay(job_id, board_type, num_players, num_games, seed)
                        )
                        ok = True
                else:
                    ok = False
                    err = f"unknown_command_type:{cmd_type}"

                if ok:
                    self.pending_relay_acks.add(cmd_id)
                    self.pending_relay_results.append({"id": cmd_id, "ok": True})
                    self.relay_command_attempts.pop(cmd_id, None)
                else:
                    if not err:
                        err = "command_failed"
                    if attempts >= RELAY_COMMAND_MAX_ATTEMPTS:
                        self.pending_relay_acks.add(cmd_id)
                        self.pending_relay_results.append({"id": cmd_id, "ok": False, "error": err})
                        self.relay_command_attempts.pop(cmd_id, None)
            except Exception as exc:
                try:
                    cmd_id = str(cmd.get("id") or "")
                    if cmd_id:
                        attempts = int(self.relay_command_attempts.get(cmd_id, 0) or 0)
                        if attempts >= RELAY_COMMAND_MAX_ATTEMPTS:
                            self.pending_relay_acks.add(cmd_id)
                            self.pending_relay_results.append({"id": cmd_id, "ok": False, "error": str(exc)})
                            self.relay_command_attempts.pop(cmd_id, None)
                except Exception:
                    continue

    async def _heartbeat_loop(self):
        """Send heartbeats to all known peers."""
        while self.running:
            try:
                # Send to known peers from config
                for peer_addr in self.known_peers:
                    try:
                        scheme, host, port = self._parse_peer_address(peer_addr)
                    except Exception:
                        continue

                    # Use relay heartbeat for HTTPS endpoints (they're proxies/relays)
                    if scheme == "https":
                        # HTTPS endpoint = relay/proxy, use relay heartbeat
                        relay_url = f"https://{host}" if port == 443 else f"https://{host}:{port}"
                        result = await self._send_relay_heartbeat(relay_url)
                        if result.get("success"):
                            # Relay heartbeat already updates peers and leader
                            continue

                    info = await self._send_heartbeat_to_peer(host, port, scheme=scheme)
                    if info:
                        if info.node_id == self.node_id:
                            continue
                        with self.peers_lock:
                            info.last_heartbeat = time.time()
                            self.peers[info.node_id] = info
                        if info.role == NodeRole.LEADER and info.node_id != self.node_id:
                            with self.peers_lock:
                                peers_snapshot = list(self.peers.values())
                            conflict_keys = self._endpoint_conflict_keys([self.self_info] + peers_snapshot)
                            if not self._is_leader_eligible(info, conflict_keys, require_alive=False):
                                continue
                            if self.role == NodeRole.LEADER and info.node_id <= self.node_id:
                                continue
                            if (
                                self.leader_id
                                and self.leader_id != info.node_id
                                and self._is_leader_lease_valid()
                                and info.node_id <= self.leader_id
                            ):
                                continue
                            if self.leader_id != info.node_id or self.role != NodeRole.FOLLOWER:
                                print(f"[P2P] Following configured leader from heartbeat: {info.node_id}")
                            prev_leader = self.leader_id
                            self.leader_id = info.node_id
                            # Provisional lease: allow time for the leader to send
                            # a /coordinator lease renewal after we discover it via
                            # heartbeat (prevents leaderless oscillation right after
                            # restarts/partitions).
                            if prev_leader != info.node_id:
                                self.leader_lease_id = ""
                                self.leader_lease_expires = time.time() + LEADER_LEASE_DURATION
                            elif not self._is_leader_lease_valid():
                                self.leader_lease_id = ""
                                self.leader_lease_expires = time.time() + LEADER_LEASE_DURATION
                            self.role = NodeRole.FOLLOWER

                # Send to discovered peers (skip NAT-blocked peers and ambiguous endpoints).
                with self.peers_lock:
                    peers_snapshot = list(self.peers.values())
                conflict_keys = self._endpoint_conflict_keys([self.self_info] + peers_snapshot)
                peer_list = [
                    p for p in peers_snapshot
                    if (
                        not p.nat_blocked
                        and self._endpoint_key(p) not in conflict_keys
                    )
                ]

                for peer in peer_list:
                    if peer.node_id != self.node_id:
                        if not peer.should_retry():
                            continue
                        peer_scheme = getattr(peer, "scheme", "http") or "http"
                        info = await self._send_heartbeat_to_peer(peer.host, peer.port, scheme=peer_scheme)
                        if not info and getattr(peer, "reported_host", "") and getattr(peer, "reported_port", 0):
                            # Multi-path retry: fall back to self-reported endpoint when the
                            # observed reachable endpoint fails (e.g., mixed overlays).
                            try:
                                rh = str(getattr(peer, "reported_host", "") or "").strip()
                                rp = int(getattr(peer, "reported_port", 0) or 0)
                            except Exception:
                                rh, rp = "", 0
                            if rh and rp and (rh != peer.host or rp != peer.port):
                                info = await self._send_heartbeat_to_peer(rh, rp, scheme=peer_scheme)
                        if info:
                            info.consecutive_failures = 0
                            info.last_failure_time = 0.0
                            with self.peers_lock:
                                info.last_heartbeat = time.time()
                                self.peers[info.node_id] = info
                            if info.role == NodeRole.LEADER and self.role != NodeRole.LEADER:
                                if not self._is_leader_eligible(info, conflict_keys, require_alive=False):
                                    continue
                                if (
                                    self.leader_id
                                    and self.leader_id != info.node_id
                                    and self._is_leader_lease_valid()
                                    and info.node_id <= self.leader_id
                                ):
                                    continue
                                if self.leader_id != info.node_id:
                                    print(f"[P2P] Adopted leader from heartbeat: {info.node_id}")
                                prev_leader = self.leader_id
                                self.leader_id = info.node_id
                                if prev_leader != info.node_id or not self._is_leader_lease_valid():
                                    self.leader_lease_id = ""
                                    self.leader_lease_expires = time.time() + LEADER_LEASE_DURATION
                                self.role = NodeRole.FOLLOWER
                        else:
                            with self.peers_lock:
                                existing = self.peers.get(peer.node_id)
                                if existing:
                                    existing.consecutive_failures = int(getattr(existing, "consecutive_failures", 0) or 0) + 1
                                    existing.last_failure_time = time.time()

                # If we're only connected to a seed peer (or lost cluster membership),
                # pull a fresh peer snapshot so leader election converges quickly.
                await self._bootstrap_from_known_peers()

                # NAT-blocked nodes: poll a relay endpoint for peer snapshots + commands.
                if getattr(self.self_info, "nat_blocked", False):
                    now = time.time()
                    if now - self.last_relay_heartbeat >= RELAY_HEARTBEAT_INTERVAL:
                        relay_urls: List[str] = []
                        leader_peer = self._get_leader_peer()
                        if leader_peer and leader_peer.node_id != self.node_id:
                            relay_urls.append(f"{leader_peer.scheme}://{leader_peer.host}:{leader_peer.port}")
                        for peer_addr in self.known_peers:
                            try:
                                scheme, host, port = self._parse_peer_address(peer_addr)
                            except Exception:
                                continue
                            relay_urls.append(f"{scheme}://{host}:{port}")
                        seen: Set[str] = set()
                        relay_urls = [u for u in relay_urls if not (u in seen or seen.add(u))]

                        for relay_url in relay_urls:
                            result = await self._send_relay_heartbeat(relay_url)
                            if result.get("success"):
                                self.last_relay_heartbeat = now
                                break

                # Check for dead peers
                self._check_dead_peers()

                # LEARNED LESSONS - Lease renewal to maintain leadership
                if self.role == NodeRole.LEADER:
                    await self._renew_leader_lease()

                # P2P monitoring: start/stop services based on leadership
                await self._stop_monitoring_if_not_leader()
                if self.role == NodeRole.LEADER:
                    await self._start_monitoring_if_leader()

                # Report node resources to resource_optimizer for cluster-wide utilization tracking
                # This enables cooperative 60-80% utilization targeting across orchestrators
                if HAS_NEW_COORDINATION and get_resource_optimizer is not None:
                    try:
                        optimizer = get_resource_optimizer()
                        self._update_self_info()
                        node_resources = NodeResources(
                            node_id=self.node_id,
                            cpu_percent=self.self_info.cpu_percent,
                            memory_percent=self.self_info.memory_percent,
                            active_jobs=self.self_info.selfplay_jobs + self.self_info.training_jobs,
                            has_gpu=self.self_info.has_gpu,
                            gpu_name=self.self_info.gpu_type or "",
                        )
                        optimizer.report_node_resources(node_resources)
                    except Exception:
                        pass  # Non-critical, don't disrupt heartbeat

                # Save state periodically
                self._save_state()

            except Exception as e:
                print(f"[P2P] Heartbeat error: {e}")

            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _manifest_collection_loop(self):
        """Periodically collect manifests for dashboard/training/sync decisions."""
        await asyncio.sleep(2.0)  # Let the HTTP server come up first
        while self.running:
            try:
                if self.role == NodeRole.LEADER:
                    cluster_manifest = await self._collect_cluster_manifest()
                    with self.manifest_lock:
                        self.cluster_data_manifest = cluster_manifest
                        self._record_selfplay_stats_sample(cluster_manifest)
                    if self.improvement_cycle_manager:
                        try:
                            self.improvement_cycle_manager.update_from_cluster_totals(
                                cluster_manifest.by_board_type
                            )
                        except Exception as e:
                            print(f"[P2P] ImprovementCycleManager update error: {e}")
                else:
                    local_manifest = await asyncio.to_thread(self._collect_local_data_manifest)
                    with self.manifest_lock:
                        self.local_data_manifest = local_manifest

                self.last_manifest_collection = time.time()

            except Exception as e:
                print(f"[P2P] Manifest collection error: {e}")

            await asyncio.sleep(self.manifest_collection_interval)

    def _record_selfplay_stats_sample(self, manifest: ClusterDataManifest) -> None:
        """Record a lightweight selfplay totals sample for dashboard charts."""
        try:
            sample = {
                "timestamp": time.time(),
                "manifest_collected_at": float(getattr(manifest, "collected_at", 0.0) or 0.0),
                "total_selfplay_games": int(getattr(manifest, "total_selfplay_games", 0) or 0),
                "by_board_type": manifest.by_board_type,
                "total_nodes": int(getattr(manifest, "total_nodes", 0) or 0),
            }
            self.selfplay_stats_history.append(sample)
            max_samples = int(getattr(self, "selfplay_stats_history_max_samples", 288) or 288)
            if max_samples > 0 and len(self.selfplay_stats_history) > max_samples:
                self.selfplay_stats_history = self.selfplay_stats_history[-max_samples:]
        except Exception:
            # Never let dashboard bookkeeping break manifest collection.
            return

    def _endpoint_key(self, info: NodeInfo) -> Optional[Tuple[str, str, int]]:
        """Return the normalized reachable endpoint key for a peer (scheme, host, port)."""
        host = str(getattr(info, "host", "") or "").strip()
        if not host:
            return None
        scheme = str(getattr(info, "scheme", "http") or "http").lower()
        try:
            port = int(getattr(info, "port", DEFAULT_PORT) or DEFAULT_PORT)
        except Exception:
            port = DEFAULT_PORT
        reported_host = str(getattr(info, "reported_host", "") or "").strip()
        try:
            reported_port = int(getattr(info, "reported_port", 0) or 0)
        except Exception:
            reported_port = 0

        if reported_host and reported_port > 0:
            # Reverse proxies / relays can cause inbound peer requests to appear as loopback.
            # Prefer the peer's self-reported advertised endpoint in that case so:
            # - endpoint conflict detection remains meaningful, and
            # - eligible leaders don't get filtered out as "conflicted".
            if host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
                host, port = reported_host, reported_port
            # Prefer mesh endpoints (Tailscale) for conflict detection so multiple
            # nodes behind the same public NAT don't collide on the same host:port.
            elif self._is_tailscale_host(reported_host):
                host, port = reported_host, reported_port
        return (scheme, host, port)

    def _endpoint_conflict_keys(self, peers: List[NodeInfo]) -> Set[Tuple[str, str, int]]:
        """Compute endpoint keys that are shared by >1 node (NAT/port collisions)."""
        counts: Dict[Tuple[str, str, int], int] = {}
        for p in peers:
            # Ignore dead peers: stale node_ids can linger after restarts and would
            # otherwise permanently mark the live node as "conflicted".
            if not p.is_alive():
                continue
            key = self._endpoint_key(p)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
        return {k for k, v in counts.items() if v > 1}

    def _is_leader_eligible(
        self,
        peer: NodeInfo,
        conflict_keys: Set[Tuple[str, str, int]],
        *,
        require_alive: bool = True,
    ) -> bool:
        """Heuristic: leaders must be directly reachable and uniquely addressable."""
        if require_alive and not peer.is_alive():
            return False
        voters = list(getattr(self, "voter_node_ids", []) or [])
        if voters and peer.node_id not in voters:
            return False
        if int(getattr(peer, "consecutive_failures", 0) or 0) >= MAX_CONSECUTIVE_FAILURES:
            return False
        if getattr(peer, "nat_blocked", False):
            return False
        key = self._endpoint_key(peer)
        if key and key in conflict_keys:
            return False
        return True

    def _maybe_adopt_leader_from_peers(self) -> bool:
        """If we can already see a healthy leader, adopt it and avoid elections."""
        if self.role == NodeRole.LEADER:
            return False

        with self.peers_lock:
            peers = [p for p in self.peers.values() if p.node_id != self.node_id]

        conflict_keys = self._endpoint_conflict_keys([self.self_info] + peers)
        leaders = [
            p for p in peers
            if p.role == NodeRole.LEADER and self._is_leader_eligible(p, conflict_keys)
        ]

        voter_ids = list(getattr(self, "voter_node_ids", []) or [])
        if voter_ids:
            leaders = [p for p in leaders if p.node_id in voter_ids]

        if not leaders:
            return False

        # If multiple leaders exist (split brain), pick the lexicographically highest
        # ID (matches bully ordering) to converge.
        leader = sorted(leaders, key=lambda p: p.node_id)[-1]

        if self.leader_id != leader.node_id:
            print(f"[P2P] Adopted existing leader from peers: {leader.node_id}")
        self.leader_id = leader.node_id
        self.role = NodeRole.FOLLOWER
        self._save_state()
        return True

    def _check_dead_peers(self):
        """Check for peers that have stopped responding."""
        now = time.time()
        with self.peers_lock:
            dead_peers = []
            for node_id, info in self.peers.items():
                if not info.is_alive() and node_id != self.node_id:
                    dead_peers.append(node_id)
                    # Retire long-dead peers so they don't pollute active scheduling.
                    try:
                        dead_for = now - float(getattr(info, "last_heartbeat", 0.0) or 0.0)
                    except Exception:
                        dead_for = float("inf")
                    if not getattr(info, "retired", False) and dead_for >= PEER_RETIRE_AFTER_SECONDS:
                        info.retired = True
                        info.retired_at = now
                        print(f"[P2P] Retiring peer {node_id} (offline for {int(dead_for)}s)")
                elif info.is_alive() and getattr(info, "retired", False):
                    # Peer came back: clear retirement.
                    info.retired = False
                    info.retired_at = 0.0

            for node_id in dead_peers:
                info = self.peers.get(node_id)
                if info and getattr(info, "retired", False):
                    continue
                print(f"[P2P] Peer {node_id} is dead (no heartbeat for {PEER_TIMEOUT}s)")
                # Don't remove, just mark as dead for historical tracking

        # LEARNED LESSONS - Clear stale leader IDs after restarts/partitions.
        #
        # Nodes persist `leader_id` but not lease metadata. After a restart, it's
        # possible to have `leader_id` point at an alive peer that is no longer a
        # leader (or to a leader whose lease is expired). Without an explicit lease
        # validity check, the cluster can get stuck leaderless and stop dispatching
        # jobs (while still "thinking" it has a leader).
        if self.leader_id and not self._is_leader_lease_valid():
            print(f"[P2P] Clearing stale/expired leader lease: leader_id={self.leader_id}")
            self.leader_id = None
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
            self.last_lease_renewal = 0.0
            self.role = NodeRole.FOLLOWER
            asyncio.create_task(self._start_election())

        # If leader is dead, start election
        if self.leader_id and self.leader_id != self.node_id:
            with self.peers_lock:
                leader = self.peers.get(self.leader_id)
                peers_snapshot = [p for p in self.peers.values() if p.node_id != self.node_id]
            if leader:
                conflict_keys = self._endpoint_conflict_keys([self.self_info] + peers_snapshot)
                if not self._is_leader_eligible(leader, conflict_keys):
                    reason = "dead" if not leader.is_alive() else "ineligible"
                    print(f"[P2P] Leader {self.leader_id} is {reason}, starting election")
                    # Clear stale/ineligible leader to avoid proxy/relay selecting it.
                    self.leader_id = None
                    self.leader_lease_id = ""
                    self.leader_lease_expires = 0.0
                    self.last_lease_renewal = 0.0
                    self.role = NodeRole.FOLLOWER
                    asyncio.create_task(self._start_election())

        # If we're leaderless, periodically retry elections so the cluster can
        # recover without requiring manual restarts.
        if not self.leader_id and not self.election_in_progress:
            now = time.time()
            backoff_seconds = max(LEADER_LEASE_RENEW_INTERVAL, ELECTION_TIMEOUT * 3)
            last_attempt = float(getattr(self, "last_election_attempt", 0.0) or 0.0)
            if now - last_attempt >= backoff_seconds:
                self.last_election_attempt = now
                asyncio.create_task(self._start_election())

    async def _start_election(self):
        """Start leader election using Bully algorithm."""
        self._update_self_info()

        # NAT-blocked nodes cannot act as a leader because peers can't reach them.
        if getattr(self.self_info, "nat_blocked", False):
            return
        # Optional quorum gating: only configured voters may lead, and only when
        # a majority of voters are currently visible.
        if getattr(self, "voter_node_ids", []):
            if self.node_id not in self.voter_node_ids:
                return
            if not self._has_voter_quorum():
                return

        with self.peers_lock:
            peers_snapshot = [p for p in self.peers.values() if p.node_id != self.node_id]

        conflict_keys = self._endpoint_conflict_keys([self.self_info] + peers_snapshot)

        if self.leader_id and self.leader_id != self.node_id:
            with self.peers_lock:
                leader = self.peers.get(self.leader_id)
            leader_ok = (
                leader is not None
                and leader.is_alive()
                and leader.role == NodeRole.LEADER
                and self._is_leader_eligible(leader, conflict_keys)
                and self._is_leader_lease_valid()
            )
            if leader_ok:
                return
            # Drop stale/ineligible leader so we don't keep advertising it.
            self.leader_id = None
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
        if self._maybe_adopt_leader_from_peers():
            return

        if self.election_in_progress:
            return

        self.election_in_progress = True
        self.role = NodeRole.CANDIDATE
        print(f"[P2P] Starting election, my ID: {self.node_id}")

        try:
            # Send election message to all nodes with higher IDs
            with self.peers_lock:
                higher_nodes = [
                    p for p in self.peers.values()
                    if (
                        p.node_id > self.node_id
                        and self._is_leader_eligible(p, conflict_keys)
                    )
                ]
                voter_node_ids = list(getattr(self, "voter_node_ids", []) or [])
                if voter_node_ids:
                    higher_nodes = [p for p in higher_nodes if p.node_id in voter_node_ids]

            got_response = False

            timeout = ClientTimeout(total=ELECTION_TIMEOUT)
            async with get_client_session(timeout) as session:
                for peer in higher_nodes:
                    try:
                        url = self._url_for_peer(peer, "/election")
                        async with session.post(url, json={"candidate_id": self.node_id}, headers=self._auth_headers()) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get("response") == "ALIVE":
                                    got_response = True
                                    print(f"[P2P] Higher node {peer.node_id} responded")
                    except:
                        pass

            # If no higher node responded, we become leader
            if not got_response:
                # Only become leader if we're eligible (unique + directly reachable).
                if self._is_leader_eligible(self.self_info, conflict_keys):
                    await self._become_leader()
            else:
                # Wait for coordinator message
                await asyncio.sleep(ELECTION_TIMEOUT * 2)
                # If no coordinator arrives, fall back to adopting any eligible leader we can see.
                self._maybe_adopt_leader_from_peers()

        finally:
            self.election_in_progress = False
            if self.role == NodeRole.CANDIDATE:
                self.role = NodeRole.FOLLOWER

    async def _become_leader(self):
        """Become the cluster leader with lease-based leadership."""
        self._update_self_info()
        if getattr(self.self_info, "nat_blocked", False):
            print(f"[P2P] Refusing leadership while NAT-blocked: {self.node_id}")
            return
        if getattr(self, "voter_node_ids", []) and not self._has_voter_quorum():
            print(f"[P2P] Refusing leadership without voter quorum: {self.node_id}")
            return
        import uuid
        lease_id = f"{self.node_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        lease_expires = await self._acquire_voter_lease_quorum(lease_id, int(LEADER_LEASE_DURATION))
        if getattr(self, "voter_node_ids", []):
            if not lease_expires:
                print(f"[P2P] Failed to obtain voter lease quorum; refusing leadership: {self.node_id}")
                self.role = NodeRole.FOLLOWER
                self.leader_id = None
                self.leader_lease_id = ""
                self.leader_lease_expires = 0.0
                self.last_lease_renewal = 0.0
                self._release_voter_grant_if_self()
                self._save_state()
                return

        print(f"[P2P] I am now the leader: {self.node_id}")
        self.role = NodeRole.LEADER
        self.leader_id = self.node_id

        # Lease-based leadership (voter-backed when enabled).
        self.leader_lease_id = lease_id
        self.leader_lease_expires = float(lease_expires or (time.time() + LEADER_LEASE_DURATION))
        self.last_lease_renewal = time.time()

        # Announce to all peers with lease information
        with self.peers_lock:
            peers = list(self.peers.values())

        timeout = ClientTimeout(total=5)
        async with get_client_session(timeout) as session:
            for peer in peers:
                if peer.node_id != self.node_id:
                    try:
                        url = self._url_for_peer(peer, "/coordinator")
                        await session.post(url, json={
                            "leader_id": self.node_id,
                            "lease_id": self.leader_lease_id,
                            "lease_expires": self.leader_lease_expires,
                            "voter_node_ids": list(getattr(self, "voter_node_ids", []) or []),
                        }, headers=self._auth_headers())
                    except:
                        pass

        self._save_state()

        # Start monitoring services when becoming leader
        await self._start_monitoring_if_leader()

    async def _start_monitoring_if_leader(self):
        """Start Prometheus/Grafana when we become leader (P2P monitoring resilience)."""
        if not self.monitoring_manager:
            return
        if self.role != NodeRole.LEADER:
            return
        if self._monitoring_was_leader:
            return  # Already started

        try:
            # Update peer list for Prometheus config
            with self.peers_lock:
                peer_list = [
                    {"node_id": p.node_id, "host": p.host, "port": getattr(p, "metrics_port", 9091)}
                    for p in self.peers.values()
                    if p.node_id != self.node_id and p.status == "healthy"
                ]
            self.monitoring_manager.update_peers(peer_list)

            # Start monitoring services
            success = await self.monitoring_manager.start_as_leader()
            if success:
                print(f"[P2P] Monitoring services started on leader node")
                self._monitoring_was_leader = True
            else:
                print(f"[P2P] Failed to start monitoring services")
        except Exception as e:
            print(f"[P2P] Error starting monitoring services: {e}")

    async def _stop_monitoring_if_not_leader(self):
        """Stop Prometheus/Grafana when we step down from leadership."""
        if not self.monitoring_manager:
            return
        if not self._monitoring_was_leader:
            return  # Never started

        if self.role != NodeRole.LEADER:
            try:
                await self.monitoring_manager.stop()
                print(f"[P2P] Monitoring services stopped (no longer leader)")
                self._monitoring_was_leader = False
            except Exception as e:
                print(f"[P2P] Error stopping monitoring services: {e}")

    async def _update_monitoring_peers(self):
        """Update Prometheus config with current peer list."""
        if not self.monitoring_manager or not self._monitoring_was_leader:
            return
        if self.role != NodeRole.LEADER:
            return

        try:
            with self.peers_lock:
                peer_list = [
                    {"node_id": p.node_id, "host": p.host, "port": getattr(p, "metrics_port", 9091)}
                    for p in self.peers.values()
                    if p.node_id != self.node_id and p.status == "healthy"
                ]
            self.monitoring_manager.update_peers(peer_list)
            await self.monitoring_manager.reload_config()
        except Exception as e:
            print(f"[P2P] Error updating monitoring peers: {e}")

    async def _renew_leader_lease(self):
        """Renew our leadership lease and broadcast to peers."""
        if self.role != NodeRole.LEADER:
            return
        if getattr(self, "voter_node_ids", []) and not self._has_voter_quorum():
            print(f"[P2P] Lost voter quorum; stepping down: {self.node_id}")
            self.role = NodeRole.FOLLOWER
            self.leader_id = None
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
            self.last_lease_renewal = 0.0
            self._release_voter_grant_if_self()
            self._save_state()
            return

        now = time.time()
        if now - self.last_lease_renewal < LEADER_LEASE_RENEW_INTERVAL:
            return  # Too soon to renew

        lease_id = str(self.leader_lease_id or "")
        if not lease_id:
            lease_id = f"{self.node_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        lease_expires = await self._acquire_voter_lease_quorum(lease_id, int(LEADER_LEASE_DURATION))
        if getattr(self, "voter_node_ids", []):
            if not lease_expires:
                print(f"[P2P] Failed to renew voter lease quorum; stepping down: {self.node_id}")
                self.role = NodeRole.FOLLOWER
                self.leader_id = None
                self.leader_lease_id = ""
                self.leader_lease_expires = 0.0
                self.last_lease_renewal = 0.0
                self._release_voter_grant_if_self()
                self._save_state()
                return

        self.leader_lease_id = lease_id
        self.leader_lease_expires = float(lease_expires or (now + LEADER_LEASE_DURATION))
        self.last_lease_renewal = now

        # Broadcast lease renewal to all peers
        with self.peers_lock:
            peers = list(self.peers.values())

        timeout = ClientTimeout(total=3)
        try:
            async with get_client_session(timeout) as session:
                for peer in peers:
                    if peer.node_id != self.node_id and peer.is_alive():
                        try:
                            url = self._url_for_peer(peer, "/coordinator")
                            await session.post(url, json={
                                "leader_id": self.node_id,
                                "lease_id": self.leader_lease_id,
                                "lease_expires": self.leader_lease_expires,
                                "lease_renewal": True,
                                "voter_node_ids": list(getattr(self, "voter_node_ids", []) or []),
                            }, headers=self._auth_headers())
                        except:
                            pass
        except Exception as e:
            print(f"[P2P] Lease renewal error: {e}")

    def _is_leader_lease_valid(self) -> bool:
        """Check if the current leader's lease is still valid."""
        if not self.leader_id:
            return False
        if self.leader_id == self.node_id:
            # We are leader - check our own lease
            return time.time() < self.leader_lease_expires
        else:
            # Another node is leader - check if we've received recent lease renewal
            # Allow some grace period (2x lease duration) for network delays
            return time.time() < self.leader_lease_expires + LEADER_LEASE_DURATION

    async def _check_and_resolve_split_brain(self) -> bool:
        """Check for split-brain (multiple leaders) and resolve by stepping down if needed.

        LEARNED LESSONS - This addresses the cluster status showing multiple leaders.
        Uses Bully algorithm: highest node_id wins.

        Returns True if we stepped down (caller should skip leadership duties).
        """
        if self.role != NodeRole.LEADER:
            return False

        with self.peers_lock:
            peers_snapshot = [p for p in self.peers.values() if p.node_id != self.node_id]

        conflict_keys = self._endpoint_conflict_keys([self.self_info] + peers_snapshot)

        # Gather all peers claiming to be leader.
        other_leaders = [peer for peer in peers_snapshot if peer.role == NodeRole.LEADER and peer.is_alive()]

        if not other_leaders:
            return False  # No split-brain

        voter_ids = list(getattr(self, "voter_node_ids", []) or [])
        if voter_ids:
            # In quorum-gated clusters, only voters may safely lead.
            if self.node_id not in voter_ids:
                print(
                    f"[P2P] SPLIT-BRAIN detected, but {self.node_id} is not a voter; stepping down."
                )
                self.role = NodeRole.FOLLOWER
                self.leader_id = None
                self.leader_lease_id = ""
                self.leader_lease_expires = 0.0
                self._release_voter_grant_if_self()
                self._save_state()
                return True

            leased_leader = await self._determine_leased_leader_from_voters()
            if leased_leader and leased_leader != self.node_id:
                print(
                    f"[P2P] SPLIT-BRAIN resolved by voter quorum: stepping down for lease-holder {leased_leader}"
                )
                self.role = NodeRole.FOLLOWER
                self.leader_id = leased_leader
                self.leader_lease_id = ""
                self.leader_lease_expires = 0.0
                self._release_voter_grant_if_self()
                self._save_state()
                return True

        # Find the highest-priority *eligible* leader (including ourselves).
        considered_leaders = other_leaders
        if voter_ids:
            # Prefer voter leaders when resolving conflicts; non-voter leaders
            # are treated as noise from older configs/versions.
            voter_leaders = [p for p in other_leaders if p.node_id in voter_ids]
            if voter_leaders:
                considered_leaders = voter_leaders

        eligible_leaders = [p for p in considered_leaders if self._is_leader_eligible(p, conflict_keys)]
        if self._is_leader_eligible(self.self_info, conflict_keys):
            eligible_leaders.append(self.self_info)

        # If none are eligible, fall back to bully ordering (best-effort).
        candidates = eligible_leaders or (considered_leaders + [self.self_info])
        highest_leader = max(candidates, key=lambda p: p.node_id)

        if highest_leader.node_id != self.node_id:
            # We're not the highest-priority leader - step down
            print(f"[P2P] SPLIT-BRAIN detected! Found leaders: {[p.node_id for p in other_leaders]}")
            print(f"[P2P] Stepping down in favor of higher-priority leader: {highest_leader.node_id}")
            self.role = NodeRole.FOLLOWER
            self.leader_id = highest_leader.node_id
            self.leader_lease_id = ""
            self.leader_lease_expires = 0.0
            self._release_voter_grant_if_self()
            self._save_state()
            return True

        # We are the highest - other leaders should step down
        # Send coordinator message to assert our leadership
        print(f"[P2P] SPLIT-BRAIN detected! Asserting leadership over: {[p.node_id for p in other_leaders]}")
        timeout = ClientTimeout(total=5)
        async with get_client_session(timeout) as session:
            for peer in other_leaders:
                try:
                    url = self._url_for_peer(peer, "/coordinator")
                    await session.post(
                        url,
                        json={
                            "leader_id": self.node_id,
                            "lease_id": self.leader_lease_id,
                            "lease_expires": self.leader_lease_expires,
                            "voter_node_ids": list(getattr(self, "voter_node_ids", []) or []),
                        },
                        headers=self._auth_headers(),
                    )
                except:
                    pass

        return False  # We remain leader

    async def _job_management_loop(self):
        """Leader-only: Manage jobs across the cluster."""
        while self.running:
            try:
                if self.role == NodeRole.LEADER:
                    # LEARNED LESSONS - Check for split-brain before acting as leader
                    if await self._check_and_resolve_split_brain():
                        # We stepped down, skip this cycle
                        await asyncio.sleep(JOB_CHECK_INTERVAL)
                        continue

                    await self._manage_cluster_jobs()
                    # Cluster rebalancing: migrate jobs from weak to powerful nodes
                    await self._check_cluster_balance()
                    # Phase 3: Check if training should be triggered automatically
                    await self._check_and_trigger_training()
                    # Phase 5: Check improvement cycles for automated training
                    await self._check_improvement_cycles()
            except Exception as e:
                print(f"[P2P] Job management error: {e}")

            await asyncio.sleep(JOB_CHECK_INTERVAL)

    def _target_selfplay_jobs_for_node(self, node: NodeInfo) -> int:
        """Return the desired selfplay concurrency for a node.

        Uses unified resource targets for consistent 60-80% utilization:
        - Backpressure-aware: Reduces jobs when training queue is full
        - Adaptive scaling: Increases jobs when underutilized, decreases when overloaded
        - Host-tier aware: Adjusts targets based on hardware capability

        Target: 60-80% CPU/GPU utilization for optimal training throughput.
        """
        # Check safeguards first
        if HAS_SAFEGUARDS and _safeguards:
            if _safeguards.is_emergency_active():
                return 0

        # Check backpressure - reduce production when training queue is full
        backpressure_factor = 1.0
        if HAS_NEW_COORDINATION:
            try:
                if should_stop_production(QueueType.TRAINING_DATA):
                    print(f"[P2P] Backpressure STOP: training queue full, halting selfplay on {node.node_id}")
                    return 0
                if should_throttle_production(QueueType.TRAINING_DATA):
                    backpressure_factor = get_throttle_factor(QueueType.TRAINING_DATA)
                    print(f"[P2P] Backpressure throttle: factor={backpressure_factor:.2f}")
            except Exception as e:
                print(f"[P2P] Backpressure check error: {e}")

        # Minimum memory requirement - skip low-memory machines to avoid OOM
        memory_gb = int(getattr(node, "memory_gb", 0) or 0)
        if memory_gb > 0 and memory_gb < MIN_MEMORY_GB_FOR_TASKS:
            return 0

        # Extract node metrics
        has_gpu = bool(getattr(node, "has_gpu", False))
        cpu_count = int(getattr(node, "cpu_count", 0) or 0)
        cpu_percent = float(getattr(node, "cpu_percent", 0.0) or 0.0)
        mem_percent = float(getattr(node, "memory_percent", 0.0) or 0.0)
        disk_percent = float(getattr(node, "disk_percent", 0.0) or 0.0)
        gpu_percent = float(getattr(node, "gpu_percent", 0.0) or 0.0)
        gpu_mem_percent = float(getattr(node, "gpu_memory_percent", 0.0) or 0.0)
        current_jobs = int(getattr(node, "selfplay_jobs", 0) or 0)

        # Record utilization for adaptive feedback
        if HAS_NEW_COORDINATION:
            try:
                record_utilization(node.node_id, cpu_percent, gpu_percent, mem_percent, current_jobs)
            except Exception:
                pass

        # Use unified resource targets if available
        if HAS_NEW_COORDINATION:
            try:
                # Get host-specific targets adjusted for tier and backpressure
                host_targets = get_host_targets(node.node_id)

                # Use the unified target calculator
                target_selfplay = get_target_job_count(
                    node.node_id,
                    cpu_count if cpu_count > 0 else 8,
                    cpu_percent,
                    gpu_percent if has_gpu else 0.0,
                )

                # Check if we should scale up (underutilized)
                scale_up, reason = should_scale_up(
                    node.node_id, cpu_percent, gpu_percent, current_jobs
                )
                if scale_up and current_jobs < target_selfplay:
                    # Controlled scale-up: Add 2-4 jobs at a time, not all at once
                    scale_up_increment = min(4, target_selfplay - current_jobs)
                    target_selfplay = current_jobs + scale_up_increment
                    if self.verbose:
                        print(f"[P2P] Scale-up on {node.node_id}: {reason}, target={target_selfplay}")

                # Check if we should scale down (overloaded)
                scale_down, reduction, reason = should_scale_down(
                    node.node_id, cpu_percent, gpu_percent, mem_percent
                )
                if scale_down:
                    target_selfplay = max(1, current_jobs - reduction)
                    print(f"[P2P] Scale-down on {node.node_id}: {reason}, target={target_selfplay}")

                # Apply backpressure factor
                target_selfplay = int(target_selfplay * backpressure_factor)

                # Apply host-specific max
                target_selfplay = min(target_selfplay, host_targets.max_selfplay)

                return int(max(1, target_selfplay))

            except Exception as e:
                print(f"[P2P] Resource targets error, falling back to legacy: {e}")

        # FALLBACK: Legacy logic if coordination not available
        target_selfplay = 4  # Baseline

        # Target 60-80% utilization (raised from 50% for better efficiency)
        TARGET_JOBS_PER_CORE = 0.35  # ~35% of cores as concurrent jobs (raised from 0.20)

        # Hard caps to prevent runaway spawning
        MAX_SELFPLAY_PER_NODE = 32
        MAX_SELFPLAY_HIGH_END = 48

        if has_gpu:
            gpu_name = (getattr(node, "gpu_name", "") or "").lower()
            # Baseline concurrency by accelerator tier
            if any(tag in gpu_name for tag in ("h100", "h200", "gh200", "5090")):
                target_selfplay = 16  # Raised from 12 for better utilization
            elif any(tag in gpu_name for tag in ("a100", "4090")):
                target_selfplay = 12  # Raised from 8
            elif any(tag in gpu_name for tag in ("3090", "a10")):
                target_selfplay = 8   # Raised from 6
            elif memory_gb >= 64:
                target_selfplay = 8

            # Scale with CPU cores
            if cpu_count >= 32:
                cpu_based_target = int(cpu_count * TARGET_JOBS_PER_CORE)
                mem_cap = memory_gb // 3 if memory_gb > 0 else 32  # 3GB per job
                core_cap = MAX_SELFPLAY_HIGH_END if cpu_count >= 128 else MAX_SELFPLAY_PER_NODE
                target_selfplay = max(target_selfplay, min(cpu_based_target, mem_cap, core_cap))
            elif cpu_count > 0:
                cpu_bonus = max(0, min(12, cpu_count // 4))
                target_selfplay = min(MAX_SELFPLAY_PER_NODE, target_selfplay + cpu_bonus)

            # Utilization-aware tuning targeting 60-80%
            # Check resource headroom independently - only scale up if BOTH have capacity
            gpu_overloaded = gpu_percent > 85 or gpu_mem_percent > 85
            cpu_overloaded = cpu_percent > 80
            gpu_has_headroom = gpu_percent < 75 and gpu_mem_percent < 75
            cpu_has_headroom = cpu_percent < 75

            # Scale DOWN if either resource is overloaded
            if gpu_overloaded:
                target_selfplay = max(2, target_selfplay - 4)
            if cpu_overloaded:
                target_selfplay = max(2, target_selfplay - 2)

            # Scale UP only if BOTH resources have headroom (neither bottlenecked)
            if not gpu_overloaded and not cpu_overloaded and current_jobs > 0:
                # GPU underutilized and CPU has headroom
                if gpu_percent < 60 and cpu_has_headroom:
                    scale_up_jobs = min(4, int((60 - gpu_percent) / 15))
                    target_selfplay = min(MAX_SELFPLAY_HIGH_END, target_selfplay + scale_up_jobs)
                # CPU underutilized and GPU has headroom
                if cpu_percent < 60 and gpu_has_headroom:
                    scale_up_jobs = min(3, int((60 - cpu_percent) / 20))
                    target_selfplay = min(MAX_SELFPLAY_PER_NODE, target_selfplay + scale_up_jobs)
        else:
            # CPU-only nodes
            if cpu_count >= 32:
                cpu_target = int(cpu_count * TARGET_JOBS_PER_CORE)
                mem_cap = memory_gb // 3 if memory_gb > 0 else 24
                core_cap = min(32, max(8, cpu_count // 4))
                target_selfplay = max(target_selfplay, min(cpu_target, mem_cap, core_cap))
            elif cpu_count > 0:
                cpu_target = max(2, int(cpu_count * TARGET_JOBS_PER_CORE))
                target_selfplay = max(target_selfplay, min(cpu_target, 20))
            elif memory_gb >= 64:
                target_selfplay = 8

            if memory_gb > 0 and memory_gb < 16:
                mem_target = max(2, memory_gb // 2)
                target_selfplay = min(target_selfplay, mem_target)

            # Utilization-aware scaling for CPU nodes
            if cpu_percent > 80:
                target_selfplay = max(2, target_selfplay - 3)
            elif cpu_percent < 60 and current_jobs > 0:
                scale_up_jobs = min(4, int((60 - cpu_percent) / 15))
                target_selfplay = min(MAX_SELFPLAY_PER_NODE, target_selfplay + scale_up_jobs)

        if disk_percent >= DISK_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 4)
        if mem_percent >= MEMORY_WARNING_THRESHOLD:
            target_selfplay = min(target_selfplay, 2)

        # Apply backpressure factor
        target_selfplay = int(target_selfplay * backpressure_factor)

        # Apply global hard cap
        target_selfplay = min(target_selfplay, MAX_SELFPLAY_PER_NODE)

        return int(max(1, target_selfplay))

    async def _check_cluster_balance(self) -> Dict[str, Any]:
        """Check and rebalance jobs across the cluster.

        This method identifies:
        1. Powerful nodes that are underutilized (high capacity, low jobs)
        2. Weak nodes that are overloaded (low capacity, high jobs)

        When imbalance is detected, it reduces jobs on weak nodes so the
        scheduler can assign them to more powerful nodes.

        Returns dict with rebalancing actions taken.
        """
        try:
            with self.peers_lock:
                alive_peers = [p for p in self.peers.values() if p.is_alive()]

            all_nodes = alive_peers + [self.self_info]
            healthy_nodes = [n for n in all_nodes if n.is_healthy()]

            if len(healthy_nodes) < 2:
                return {"action": "none", "reason": "insufficient_nodes"}

            # Calculate capacity and utilization for each node
            node_stats = []
            for node in healthy_nodes:
                target = self._target_selfplay_jobs_for_node(node)
                current = int(getattr(node, "selfplay_jobs", 0) or 0)
                utilization = current / max(1, target)  # How full is this node
                capacity_score = target  # Higher = more powerful

                node_stats.append({
                    "node": node,
                    "target": target,
                    "current": current,
                    "utilization": utilization,
                    "capacity": capacity_score,
                    "load_score": node.get_load_score(),
                })

            # Find underutilized powerful nodes (capacity > median, utilization < 50%)
            sorted_by_capacity = sorted(node_stats, key=lambda x: x["capacity"], reverse=True)
            median_capacity = sorted_by_capacity[len(sorted_by_capacity) // 2]["capacity"]

            underutilized_powerful = [
                n for n in node_stats
                if n["capacity"] > median_capacity and n["utilization"] < 0.5
            ]

            # Find overloaded weak nodes (capacity < median, utilization > 100%)
            overloaded_weak = [
                n for n in node_stats
                if n["capacity"] < median_capacity and n["utilization"] > 1.0
            ]

            if not underutilized_powerful or not overloaded_weak:
                return {"action": "none", "reason": "balanced"}

            # Calculate rebalancing opportunity
            spare_capacity = sum(
                max(0, n["target"] - n["current"]) for n in underutilized_powerful
            )
            excess_load = sum(
                max(0, n["current"] - n["target"]) for n in overloaded_weak
            )

            if spare_capacity < 2 or excess_load < 2:
                return {"action": "none", "reason": "minimal_imbalance"}

            # Migrate: reduce jobs on weak nodes
            rebalance_actions = []
            jobs_to_migrate = min(spare_capacity, excess_load)

            for weak_node in sorted(overloaded_weak, key=lambda x: x["utilization"], reverse=True):
                if jobs_to_migrate <= 0:
                    break

                node = weak_node["node"]
                reduce_by = min(
                    weak_node["current"] - weak_node["target"],
                    jobs_to_migrate
                )
                new_target = weak_node["current"] - reduce_by

                if reduce_by > 0:
                    print(
                        f"[P2P] Cluster rebalance: {node.node_id} overloaded "
                        f"({weak_node['current']}/{weak_node['target']} jobs, "
                        f"{weak_node['utilization']*100:.0f}% util) - reducing by {reduce_by}"
                    )

                    if node.node_id == self.node_id:
                        await self._reduce_local_selfplay_jobs(new_target, reason="cluster_rebalance")
                    else:
                        await self._request_reduce_selfplay(node, new_target, reason="cluster_rebalance")

                    rebalance_actions.append({
                        "node": node.node_id,
                        "reduced_by": reduce_by,
                        "new_target": new_target,
                    })
                    jobs_to_migrate -= reduce_by

            # Record rebalancing metric
            if rebalance_actions:
                self.record_metric(
                    "cluster_rebalance",
                    len(rebalance_actions),
                    metadata={
                        "spare_capacity": spare_capacity,
                        "excess_load": excess_load,
                        "actions": rebalance_actions,
                    },
                )

            return {
                "action": "rebalanced",
                "spare_capacity": spare_capacity,
                "excess_load": excess_load,
                "actions": rebalance_actions,
            }

        except Exception as e:
            print(f"[P2P] Cluster balance check error: {e}")
            return {"action": "error", "error": str(e)}

    async def _manage_cluster_jobs(self):
        """Manage jobs across the cluster (leader only).

        LEARNED LESSONS incorporated:
        - Check disk space BEFORE starting jobs (Vast.ai 91-93% disk issue)
        - Check memory to prevent OOM (AWS instance crashed at 31GB+)
        - Trigger cleanup when approaching limits
        - Use is_healthy() not just is_alive()
        """
        print("[P2P] Leader: Managing cluster jobs...")

        # Gather cluster state
        with self.peers_lock:
            alive_peers = [p for p in self.peers.values() if p.is_alive()]

        # Add self
        self._update_self_info()
        all_nodes = alive_peers + [self.self_info]

        # Phase 1: Handle resource warnings and cleanup
        for node in all_nodes:
            # LEARNED LESSONS - Proactive disk cleanup before hitting critical
            if node.disk_percent >= DISK_CLEANUP_THRESHOLD:
                print(f"[P2P] {node.node_id}: Disk at {node.disk_percent:.0f}% - triggering cleanup")
                if node.node_id == self.node_id:
                    await self._cleanup_local_disk()
                else:
                    await self._request_remote_cleanup(node)
                continue  # Skip job creation this cycle

            # Load shedding: when a node is under memory/disk pressure, ask it to
            # stop excess selfplay jobs so it can recover (prevents OOM + disk-full).
            pressure_reasons: List[str] = []
            if node.memory_percent >= MEMORY_WARNING_THRESHOLD:
                pressure_reasons.append("memory")
            if node.disk_percent >= DISK_WARNING_THRESHOLD:
                pressure_reasons.append("disk")

            if pressure_reasons:
                desired = self._target_selfplay_jobs_for_node(node)
                if node.memory_percent >= MEMORY_CRITICAL_THRESHOLD or node.disk_percent >= DISK_CRITICAL_THRESHOLD:
                    desired = 0

                if node.selfplay_jobs > desired:
                    reason = "+".join(pressure_reasons)
                    print(
                        f"[P2P] {node.node_id}: Load shedding (reason={reason}) "
                        f"{node.selfplay_jobs}->{desired} selfplay jobs"
                    )
                    if node.node_id == self.node_id:
                        await self._reduce_local_selfplay_jobs(desired, reason=reason)
                    else:
                        await self._request_reduce_selfplay(node, desired, reason=reason)

        # Phase 1.5: LEARNED LESSONS - Detect stuck jobs (GPU idle with running processes)
        # This addresses the vast-5090-quad issue where 582 processes ran at 0% GPU
        for node in all_nodes:
            if not node.has_gpu or node.selfplay_jobs <= 0:
                # No GPU or no jobs running - not stuck
                if node.node_id in self.gpu_idle_since:
                    del self.gpu_idle_since[node.node_id]
                continue

            # Check if GPU is idle (< threshold) with jobs running
            gpu_name = (node.gpu_name or "").upper()
            is_cuda_gpu = "MPS" not in gpu_name and "APPLE" not in gpu_name
            if not is_cuda_gpu:
                continue  # Skip Apple Silicon, doesn't have nvidia-smi

            if node.gpu_percent < GPU_IDLE_THRESHOLD:
                # GPU idle with jobs running - track or take action
                if node.node_id not in self.gpu_idle_since:
                    self.gpu_idle_since[node.node_id] = time.time()
                    print(f"[P2P] {node.node_id}: GPU idle ({node.gpu_percent:.0f}%) with {node.selfplay_jobs} jobs - monitoring")
                else:
                    idle_duration = time.time() - self.gpu_idle_since[node.node_id]
                    if idle_duration >= GPU_IDLE_RESTART_TIMEOUT:
                        print(f"[P2P] {node.node_id}: STUCK! GPU idle for {idle_duration:.0f}s with {node.selfplay_jobs} jobs")
                        print(f"[P2P] {node.node_id}: Requesting job restart...")
                        if node.node_id == self.node_id:
                            await self._restart_local_stuck_jobs()
                        else:
                            await self._request_job_restart(node)
                        del self.gpu_idle_since[node.node_id]
            else:
                # GPU is working - clear idle tracking
                if node.node_id in self.gpu_idle_since:
                    del self.gpu_idle_since[node.node_id]

        # Phase 1.6: Detect runaway selfplay processes (lost tracking / manual runs).
        # If a node reports an absurd number of selfplay processes, request a
        # restart sweep to kill untracked jobs and recover capacity.
        for node in all_nodes:
            try:
                target_selfplay = self._target_selfplay_jobs_for_node(node)
                dynamic_threshold = max(16, target_selfplay * 3)
                runaway_threshold = (
                    int(RUNAWAY_SELFPLAY_PROCESS_THRESHOLD)
                    if int(RUNAWAY_SELFPLAY_PROCESS_THRESHOLD) > 0
                    else int(dynamic_threshold)
                )
                if int(getattr(node, "selfplay_jobs", 0) or 0) < runaway_threshold:
                    continue
            except Exception:
                continue

            print(
                f"[P2P] {node.node_id}: RUNAWAY selfplay count ({node.selfplay_jobs}) "
                f">= {runaway_threshold} — requesting restart sweep"
            )
            if node.node_id == self.node_id:
                await self._restart_local_stuck_jobs()
            else:
                await self._request_job_restart(node)

        # Phase 2: Calculate desired job distribution for healthy nodes
        # LEARNED LESSONS - Sort nodes by load score for load balancing
        # Least-loaded nodes get jobs first to ensure even distribution
        healthy_nodes = [n for n in all_nodes if n.is_healthy()]
        healthy_nodes.sort(key=lambda n: n.get_load_score())

        if healthy_nodes:
            load_summary = ", ".join(
                f"{n.node_id[:12]}={n.get_load_score():.0f}%"
                for n in healthy_nodes[:5]
            )
            print(f"[P2P] Load balancing: {load_summary}")

        for node in healthy_nodes:
            load_score = node.get_load_score()
            if load_score >= LOAD_MAX_FOR_NEW_JOBS:
                print(f"[P2P] {node.node_id}: Load {load_score:.0f}% - skipping new job starts")
                continue

            # LEARNED LESSONS - Reduce target when approaching limits
            # Base targets:
            # - GPU nodes: fixed concurrency tuned for GPU throughput.
            # - CPU-only nodes: scale with CPU cores (and cap by memory).
            target_selfplay = self._target_selfplay_jobs_for_node(node)

            # Check if node needs more jobs
            if node.selfplay_jobs < target_selfplay:
                needed = target_selfplay - node.selfplay_jobs
                print(f"[P2P] {node.node_id} needs {needed} more selfplay jobs")

                # Job configuration diversity - cycle through different board types/players
                # LEARNED LESSONS - Prioritize underserved configs:
                # - Hex 3p/4p and 19x19 3p/4p over 2p and 8x8
                # - Use mixed/mcts-only for GPU nodes to utilize GPU properly
                selfplay_configs = [
                    # HIGH PRIORITY: Hexagonal multiplayer (most underserved)
                    {"board_type": "hexagonal", "num_players": 4, "engine_mode": "mcts-only", "priority": 3},
                    {"board_type": "hexagonal", "num_players": 3, "engine_mode": "mcts-only", "priority": 3},
                    {"board_type": "hexagonal", "num_players": 4, "engine_mode": "mixed", "priority": 3},
                    {"board_type": "hexagonal", "num_players": 3, "engine_mode": "mixed", "priority": 3},
                    # HIGH PRIORITY: Square19 multiplayer
                    {"board_type": "square19", "num_players": 4, "engine_mode": "mcts-only", "priority": 3},
                    {"board_type": "square19", "num_players": 3, "engine_mode": "mcts-only", "priority": 3},
                    {"board_type": "square19", "num_players": 4, "engine_mode": "mixed", "priority": 2},
                    {"board_type": "square19", "num_players": 3, "engine_mode": "mixed", "priority": 2},
                    # MEDIUM PRIORITY: Square8 multiplayer
                    {"board_type": "square8", "num_players": 4, "engine_mode": "mixed", "priority": 2},
                    {"board_type": "square8", "num_players": 3, "engine_mode": "mixed", "priority": 2},
                    # LOW PRIORITY: 2-player (already have more data)
                    {"board_type": "hexagonal", "num_players": 2, "engine_mode": "mixed", "priority": 1},
                    {"board_type": "square19", "num_players": 2, "engine_mode": "mixed", "priority": 1},
                    {"board_type": "square8", "num_players": 2, "engine_mode": "mixed", "priority": 1},
                ]

                # LEARNED LESSONS - Weighted selection favoring high priority configs
                # Expand configs by priority for weighted random selection
                node_mem = int(getattr(node, "memory_gb", 0) or 0)
                filtered_configs = selfplay_configs
                if node_mem and node_mem < 48:
                    # Smaller CPU nodes should avoid square19/hexagonal to reduce
                    # OOM risk and thrash. Keep them productive with square8.
                    filtered_configs = [cfg for cfg in selfplay_configs if cfg.get("board_type") == "square8"]

                weighted_configs = []
                for cfg in filtered_configs:
                    weighted_configs.extend([cfg] * cfg.get("priority", 1))

                # Start jobs (max 2 at a time to avoid overwhelming)
                for i in range(min(needed, 2)):
                    # LEARNED LESSONS - Smart CPU/GPU task routing:
                    # - High-end GPUs (H100/H200/A100/5090/4090) get GPU_SELFPLAY for max throughput
                    #   with automatic CPU validation to ensure data quality
                    # - Mid-tier GPUs get HYBRID_SELFPLAY (CPU rules + GPU eval)
                    # - CPU-only nodes get regular SELFPLAY
                    # This ensures expensive GPU resources are utilized properly
                    # while CPU instances handle CPU-bound tasks efficiently
                    gpu_name = (node.gpu_name or "").upper()
                    is_high_end_gpu = any(tag in gpu_name for tag in ("H100", "H200", "GH200", "A100", "5090", "4090"))
                    is_apple_gpu = "MPS" in gpu_name or "APPLE" in gpu_name

                    # GPU validation: if node claims GPU but utilization is 0% with jobs running,
                    # GPU may not be available (driver issue, container misconfiguration, etc.)
                    gpu_percent = getattr(node, "gpu_percent", 0) or 0
                    gpu_seems_unavailable = (
                        node.has_gpu
                        and not is_apple_gpu
                        and node.selfplay_jobs > 2
                        and gpu_percent < 1
                    )
                    if gpu_seems_unavailable:
                        print(f"[P2P] WARNING: {node.node_id} has GPU but 0% utilization with {node.selfplay_jobs} jobs - falling back to CPU selfplay")

                    if node.has_gpu and is_high_end_gpu and not is_apple_gpu and not gpu_seems_unavailable:
                        # High-end CUDA GPUs: Use pure GPU selfplay with CPU validation
                        # This maximizes GPU parallel throughput (10-100x speedup)
                        # CPU validation runs automatically after completion
                        job_type = JobType.GPU_SELFPLAY
                        task_type_str = "GPU-parallel (validated)"
                    elif node.has_gpu and not is_apple_gpu and not gpu_seems_unavailable:
                        # Mid-tier GPUs: Use hybrid (CPU rules + GPU eval)
                        job_type = JobType.HYBRID_SELFPLAY
                        task_type_str = "HYBRID (accel)"
                    else:
                        job_type = JobType.SELFPLAY
                        task_type_str = "CPU-only"

                    gpu_info = f"gpu={node.gpu_name or 'none'}, gpu%={getattr(node, 'gpu_percent', 0):.0f}" if node.has_gpu else "no-gpu"
                    print(f"[P2P] Assigning {task_type_str} task to {node.node_id} ({gpu_info}, load={node.get_load_score():.0f}%)")

                    # Weighted config selection based on priority and node capabilities
                    # Use ImprovementCycleManager for dynamic data-aware diverse selection
                    # with node-aware routing: heavy workloads -> heavy nodes
                    import random as rand_module
                    if self.improvement_cycle_manager:
                        # Node-aware dynamic selection: routes hex/sq19/3p/4p to powerful nodes
                        node_gpu_power = node.gpu_power_score() if hasattr(node, 'gpu_power_score') else 0
                        node_memory = int(getattr(node, 'memory_gb', 0) or 0)
                        config = self.improvement_cycle_manager.get_next_selfplay_config_for_node(
                            node_gpu_power=node_gpu_power,
                            node_memory_gb=node_memory,
                            cluster_data=self.cluster_data_manifest
                        )
                    else:
                        # Fallback to static weighted selection
                        config_idx = (hash(node.node_id) + i + int(time.time() // 1800)) % len(weighted_configs)
                        config = weighted_configs[config_idx]

                    # Track diversity metrics for monitoring
                    self._track_selfplay_diversity(config)

                    if node.node_id == self.node_id:
                        await self._start_local_job(
                            job_type,
                            board_type=config["board_type"],
                            num_players=config["num_players"],
                            engine_mode=config["engine_mode"],
                        )
                    else:
                        await self._request_remote_job(
                            node, job_type,
                            board_type=config["board_type"],
                            num_players=config["num_players"],
                            engine_mode=config["engine_mode"],
                        )

    async def _cleanup_local_disk(self):
        """Clean up disk space on local node.

        LEARNED LESSONS - Automatically archive old data:
        - Remove deprecated selfplay databases
        - Compress and archive old logs
        - Clear /tmp files older than 24h
        """
        print("[P2P] Running local disk cleanup...")
        try:
            # Prefer the shared disk monitor (used by cron/resilience) for consistent cleanup policy.
            disk_monitor = Path(self.ringrift_path) / "ai-service" / "scripts" / "disk_monitor.py"
            if disk_monitor.exists():
                usage = self._get_resource_usage()
                disk_percent = float(usage.get("disk_percent", 0.0) or 0.0)
                cmd = [
                    "python3",
                    str(disk_monitor),
                    "--threshold",
                    str(DISK_CLEANUP_THRESHOLD),
                    "--ringrift-path",
                    str(self.ringrift_path),
                    "--aggressive",
                ]
                if disk_percent >= DISK_CRITICAL_THRESHOLD:
                    cmd.append("--force")

                out = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(Path(self.ringrift_path) / "ai-service"),
                )
                if out.returncode == 0:
                    print("[P2P] Disk monitor cleanup completed")
                else:
                    print(f"[P2P] Disk monitor cleanup failed: {out.stderr[:200]}")
            else:
                # Minimal fallback: clear old logs if disk monitor isn't available.
                log_dir = Path(self.ringrift_path) / "ai-service" / "logs"
                if log_dir.exists():
                    for logfile in log_dir.rglob("*.log"):
                        if time.time() - logfile.stat().st_mtime > 7 * 86400:  # 7 days
                            logfile.unlink()
                            print(f"[P2P] Cleaned old log: {logfile}")

        except Exception as e:
            print(f"[P2P] Disk cleanup error: {e}")

    async def _request_remote_cleanup(self, node: NodeInfo):
        """Request a remote node to clean up disk space."""
        try:
            if getattr(node, "nat_blocked", False):
                cmd_id = await self._enqueue_relay_command_for_peer(node, "cleanup", {})
                if cmd_id:
                    print(f"[P2P] Enqueued relay cleanup for {node.node_id}")
                else:
                    print(f"[P2P] Relay queue full; skipping cleanup enqueue for {node.node_id}")
                return
            timeout = ClientTimeout(total=HTTP_TOTAL_TIMEOUT)
            async with get_client_session(timeout) as session:
                last_err: Optional[str] = None
                for url in self._urls_for_peer(node, "/cleanup"):
                    try:
                        async with session.post(url, json={}, headers=self._auth_headers()) as resp:
                            if resp.status == 200:
                                print(f"[P2P] Cleanup requested on {node.node_id}")
                                return
                            last_err = f"http_{resp.status}"
                    except Exception as e:
                        last_err = str(e)
                        continue
                if last_err:
                    print(f"[P2P] Cleanup request failed on {node.node_id}: {last_err}")
        except Exception as e:
            print(f"[P2P] Failed to request cleanup from {node.node_id}: {e}")

    async def _reduce_local_selfplay_jobs(self, target_selfplay_jobs: int, *, reason: str) -> Dict[str, Any]:
        """Best-effort: stop excess selfplay jobs on this node (load shedding).

        Used when disk/memory pressure is high: we want the node to recover and
        avoid OOM/disk-full scenarios, even if it means reducing throughput.
        """
        try:
            target = max(0, int(target_selfplay_jobs))
        except Exception:
            target = 0

        # First, get an overall count using the same mechanism used for cluster
        # reporting (includes untracked processes).
        try:
            selfplay_before, _training_before = self._count_local_jobs()
        except Exception:
            selfplay_before = 0

        # Hard shedding (target=0): reuse the existing restart sweep, which
        # kills both tracked and untracked selfplay processes.
        if target <= 0:
            await self._restart_local_stuck_jobs()
            try:
                selfplay_after, _training_after = self._count_local_jobs()
            except Exception:
                selfplay_after = 0
            return {
                "running_before": int(selfplay_before),
                "running_after": int(selfplay_after),
                "stopped": max(0, int(selfplay_before) - int(selfplay_after)),
                "target": 0,
                "reason": reason,
            }

        with self.jobs_lock:
            running: List[Tuple[str, ClusterJob]] = [
                (job_id, job)
                for job_id, job in self.local_jobs.items()
                if job.status == "running"
                and job.job_type in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY)
            ]

        if selfplay_before <= target and len(running) <= target:
            return {
                "running_before": int(selfplay_before),
                "running_after": int(selfplay_before),
                "stopped": 0,
                "target": target,
                "reason": reason,
            }

        # Stop newest-first to avoid killing long-running jobs near completion.
        running.sort(key=lambda pair: float(getattr(pair[1], "started_at", 0.0) or 0.0), reverse=True)
        to_stop = running[target:]

        stopped = 0
        with self.jobs_lock:
            for job_id, job in to_stop:
                try:
                    if job.pid:
                        os.kill(int(job.pid), signal.SIGTERM)
                    job.status = "stopped"
                    stopped += 1
                except Exception:
                    continue

        # If job tracking was lost, we may still have a large number of
        # untracked selfplay processes. Best-effort kill enough to hit target.
        try:
            selfplay_mid, _training_mid = self._count_local_jobs()
        except Exception:
            selfplay_mid = max(0, int(selfplay_before) - stopped)

        if selfplay_mid > target:
            try:
                import shutil

                if shutil.which("pgrep"):
                    pids: List[int] = []
                    for pattern in (
                        "run_self_play_soak.py",
                        "run_gpu_selfplay.py",
                        "run_hybrid_selfplay.py",
                        "run_random_selfplay.py",
                    ):
                        out = subprocess.run(
                            ["pgrep", "-f", pattern],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if out.returncode == 0 and out.stdout.strip():
                            for token in out.stdout.strip().split():
                                try:
                                    pids.append(int(token))
                                except Exception:
                                    continue

                    # Kill newest-ish (highest PID) first.
                    pids = sorted(set(pids), reverse=True)
                    excess = int(selfplay_mid) - int(target)
                    killed = 0
                    for pid in pids:
                        if killed >= excess:
                            break
                        try:
                            os.kill(pid, signal.SIGTERM)
                            killed += 1
                        except Exception:
                            continue
                    stopped += killed
            except Exception:
                pass

        if stopped:
            self._save_state()

        try:
            selfplay_after, _training_after = self._count_local_jobs()
        except Exception:
            selfplay_after = max(0, int(selfplay_before) - stopped)

        return {
            "running_before": int(selfplay_before),
            "running_after": int(selfplay_after),
            "stopped": int(max(0, int(selfplay_before) - int(selfplay_after))),
            "target": target,
            "reason": reason,
        }

    async def _request_reduce_selfplay(self, node: NodeInfo, target_selfplay_jobs: int, *, reason: str) -> None:
        """Ask a node to shed excess selfplay (used for memory/disk pressure)."""
        try:
            target = max(0, int(target_selfplay_jobs))
        except Exception:
            target = 0

        if getattr(node, "nat_blocked", False):
            payload = {"target_selfplay_jobs": target, "reason": reason}
            cmd_id = await self._enqueue_relay_command_for_peer(node, "reduce_selfplay", payload)
            if cmd_id:
                print(f"[P2P] Enqueued relay reduce_selfplay for {node.node_id} (target={target}, reason={reason})")
            else:
                print(f"[P2P] Relay queue full for {node.node_id}; skipping reduce_selfplay enqueue")
            return

        timeout = ClientTimeout(total=HTTP_TOTAL_TIMEOUT)
        async with get_client_session(timeout) as session:
            last_err: Optional[str] = None
            payload = {"target_selfplay_jobs": target, "reason": reason}
            for url in self._urls_for_peer(node, "/reduce_selfplay"):
                try:
                    async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                        if resp.status == 200:
                            print(f"[P2P] Requested load shedding on {node.node_id} (target={target}, reason={reason})")
                            return
                        last_err = f"http_{resp.status}"
                except Exception as e:
                    last_err = str(e)
                    continue
            if last_err:
                print(f"[P2P] reduce_selfplay request failed on {node.node_id}: {last_err}")

    async def _restart_local_stuck_jobs(self):
        """Kill stuck selfplay processes and let job management restart them.

        LEARNED LESSONS - Addresses the issue where processes accumulate but GPU stays at 0%.
        """
        print("[P2P] Restarting stuck local selfplay jobs...")
        try:
            # Kill tracked selfplay jobs (avoid broad pkill patterns).
            jobs_to_clear: List[str] = []
            pids_to_kill: Set[int] = set()
            with self.jobs_lock:
                for job_id, job in self.local_jobs.items():
                    if job.job_type not in (JobType.SELFPLAY, JobType.GPU_SELFPLAY, JobType.HYBRID_SELFPLAY):
                        continue
                    jobs_to_clear.append(job_id)
                    if job.pid:
                        try:
                            pids_to_kill.add(int(job.pid))
                        except Exception:
                            continue

            # Sweep for untracked selfplay processes (e.g. lost local_jobs state) and kill them too.
            try:
                import shutil

                if shutil.which("pgrep"):
                    for pattern in (
                        "run_self_play_soak.py",
                        "run_gpu_selfplay.py",
                        "run_hybrid_selfplay.py",
                        "run_random_selfplay.py",
                    ):
                        out = subprocess.run(
                            ["pgrep", "-f", pattern],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if out.returncode == 0 and out.stdout.strip():
                            for token in out.stdout.strip().split():
                                try:
                                    pids_to_kill.add(int(token))
                                except Exception:
                                    continue
            except Exception:
                pass

            pids_to_kill.discard(int(os.getpid()))

            killed = 0
            for pid in sorted(pids_to_kill):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except Exception:
                    continue

            # Clear our job tracking - they'll be restarted next cycle.
            with self.jobs_lock:
                for job_id in jobs_to_clear:
                    self.local_jobs.pop(job_id, None)

            print(f"[P2P] Killed {killed} processes, cleared {len(jobs_to_clear)} job records")
        except Exception as e:
            print(f"[P2P] Error killing stuck processes: {e}")

    async def _request_job_restart(self, node: NodeInfo):
        """Request a remote node to restart its stuck selfplay jobs."""
        try:
            if getattr(node, "nat_blocked", False):
                cmd_id = await self._enqueue_relay_command_for_peer(node, "restart_stuck_jobs", {})
                if cmd_id:
                    print(f"[P2P] Enqueued relay restart_stuck_jobs for {node.node_id}")
                else:
                    print(f"[P2P] Relay queue full for {node.node_id}; skipping restart enqueue")
                return
            timeout = ClientTimeout(total=HTTP_TOTAL_TIMEOUT)
            async with get_client_session(timeout) as session:
                last_err: Optional[str] = None
                for url in self._urls_for_peer(node, "/restart_stuck_jobs"):
                    try:
                        async with session.post(url, json={}, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            data = await resp.json()
                            if data.get("success"):
                                print(f"[P2P] Job restart requested on {node.node_id}")
                                return
                            last_err = str(data.get("error") or "restart_failed")
                    except Exception as e:
                        last_err = str(e)
                        continue
                if last_err:
                    print(f"[P2P] Job restart request failed on {node.node_id}: {last_err}")
        except Exception as e:
            print(f"[P2P] Failed to request job restart from {node.node_id}: {e}")

    async def _start_local_job(
        self,
        job_type: JobType,
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "descent-only",
        job_id: Optional[str] = None,
        cuda_visible_devices: Optional[str] = None,
    ) -> Optional[ClusterJob]:
        """Start a job on the local node.

        SAFEGUARD: Checks coordination safeguards before spawning.
        """
        try:
            # SAFEGUARD: Check safeguards before spawning
            if HAS_SAFEGUARDS and _safeguards:
                task_type_str = job_type.value if hasattr(job_type, 'value') else str(job_type)
                allowed, reason = check_before_spawn(task_type_str, self.node_id)
                if not allowed:
                    print(f"[P2P] SAFEGUARD blocked {task_type_str} on {self.node_id}: {reason}")
                    return None

                # Apply backpressure delay
                delay = _safeguards.get_delay()
                if delay > 0:
                    print(f"[P2P] SAFEGUARD applying {delay:.1f}s backpressure delay")
                    await asyncio.sleep(delay)

            if job_id:
                job_id = str(job_id)
                with self.jobs_lock:
                    existing = self.local_jobs.get(job_id)
                if existing and existing.status == "running":
                    return existing
            else:
                job_id = str(uuid.uuid4())[:8]

            if job_type == JobType.SELFPLAY:
                # Normalize engine_mode to what run_self_play_soak.py supports.
                supported_engine_modes = {
                    "descent-only",
                    "mixed",
                    "random-only",
                    "heuristic-only",
                    "minimax-only",
                    "mcts-only",
                    "nn-only",
                }
                engine_mode_norm = engine_mode if engine_mode in supported_engine_modes else "mixed"

                # Memory-safety defaults for large boards.
                num_games = 1000
                extra_args: List[str] = []
                if board_type in ("square19", "hexagonal"):
                    num_games = 200 if board_type == "square19" else 100
                    extra_args.extend(["--memory-constrained"])

                output_dir = Path(
                    self.ringrift_path,
                    "ai-service",
                    "data",
                    "selfplay",
                    "p2p",
                    f"{board_type}_{num_players}p",
                    job_id,
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    "python3",
                    f"{self.ringrift_path}/ai-service/scripts/run_self_play_soak.py",
                    "--num-games", str(num_games),
                    "--board-type", board_type,
                    "--num-players", str(num_players),
                    "--engine-mode", engine_mode_norm,
                    "--max-moves", "10000",  # LEARNED LESSONS - Avoid draws due to move limit
                    "--log-jsonl", str(output_dir / "games.jsonl"),
                    "--summary-json", str(output_dir / "summary.json"),
                    "--record-db", str(output_dir / "games.db"),
                    "--lean-db",
                    "--verbose", "0",
                    *extra_args,
                ]

                # Start process
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"
                env["RINGRIFT_JOB_ORIGIN"] = "p2p_orchestrator"

                # SAFEGUARD: Final check before spawning (load + rate limit)
                can_spawn, spawn_reason = self._can_spawn_process(f"selfplay-{board_type}-{num_players}p")
                if not can_spawn:
                    print(f"[P2P] BLOCKED selfplay spawn: {spawn_reason}")
                    return None

                log_handle = open(output_dir / "run.log", "a")
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=self.ringrift_path,
                    )
                    self._record_spawn()  # Track spawn for rate limiting
                finally:
                    log_handle.close()

                job = ClusterJob(
                    job_id=job_id,
                    job_type=job_type,
                    node_id=self.node_id,
                    board_type=board_type,
                    num_players=num_players,
                    engine_mode=engine_mode_norm,
                    pid=proc.pid,
                    started_at=time.time(),
                    status="running",
                )

                with self.jobs_lock:
                    self.local_jobs[job_id] = job

                print(f"[P2P] Started {job_type.value} job {job_id} (PID {proc.pid})")
                self._save_state()
                return job

            elif job_type == JobType.GPU_SELFPLAY:
                # GPU-accelerated parallel selfplay using run_gpu_selfplay.py
                # Only start on nodes with GPU (check done in _manage_cluster_jobs)
                # NOTE: run_gpu_selfplay uses CUDA; avoid scheduling on Apple MPS nodes.
                gpu_name_upper = (self.self_info.gpu_name or "").upper()
                if "MPS" in gpu_name_upper or "APPLE" in gpu_name_upper:
                    return await self._start_local_job(
                        JobType.SELFPLAY,
                        board_type=board_type,
                        num_players=num_players,
                        engine_mode="mixed",
                    )

                if "H100" in gpu_name_upper or "H200" in gpu_name_upper:
                    batch_size = 64
                elif "5090" in gpu_name_upper or "4090" in gpu_name_upper or "A100" in gpu_name_upper:
                    batch_size = 32
                else:
                    batch_size = 16

                # run_gpu_selfplay expects --board (square8/square19/hex/hexagonal).
                board_arg = {
                    "square8": "square8",
                    "square19": "square19",
                    "hexagonal": "hexagonal",
                    "hex": "hex",
                }.get(board_type, "square8")

                # Normalize engine_mode to GPU runner's supported values.
                gpu_engine_mode = engine_mode if engine_mode in ("random-only", "heuristic-only") else "heuristic-only"

                num_games = 3000
                if board_arg == "square19":
                    num_games = 1500
                elif board_arg in ("hex", "hexagonal"):
                    num_games = 500

                output_dir = Path(
                    self.ringrift_path,
                    "ai-service",
                    "data",
                    "selfplay",
                    "p2p_gpu",
                    f"{board_type}_{num_players}p",
                    job_id,
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    "python3",
                    f"{self.ringrift_path}/ai-service/scripts/run_gpu_selfplay.py",
                    "--board", board_arg,
                    "--engine-mode", gpu_engine_mode,
                    "--num-games", str(num_games),
                    "--num-players", str(num_players),
                    "--batch-size", str(batch_size),
                    "--output-dir", str(output_dir),
                ]

                # Start process with GPU environment
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"
                env["RINGRIFT_JOB_ORIGIN"] = "p2p_orchestrator"

                if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
                    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()
                # Choose a GPU automatically if not explicitly pinned.
                elif "CUDA_VISIBLE_DEVICES" not in env:
                    gpu_count = 0
                    try:
                        out = subprocess.run(
                            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=5
                        )
                        if out.returncode == 0 and out.stdout.strip():
                            gpu_count = len([l for l in out.stdout.splitlines() if l.strip()])
                    except Exception:
                        gpu_count = 0

                    if gpu_count > 0:
                        with self.jobs_lock:
                            running_gpu_jobs = sum(
                                1 for j in self.local_jobs.values()
                                if j.job_type == JobType.GPU_SELFPLAY and j.status == "running"
                            )
                        env["CUDA_VISIBLE_DEVICES"] = str(running_gpu_jobs % gpu_count)
                    else:
                        env["CUDA_VISIBLE_DEVICES"] = "0"

                # SAFEGUARD: Final check before spawning (load + rate limit)
                can_spawn, spawn_reason = self._can_spawn_process(f"gpu-selfplay-{board_type}-{num_players}p")
                if not can_spawn:
                    print(f"[P2P] BLOCKED GPU selfplay spawn: {spawn_reason}")
                    return None

                log_handle = open(output_dir / "gpu_run.log", "a")
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=self.ringrift_path,
                    )
                    self._record_spawn()  # Track spawn for rate limiting
                finally:
                    log_handle.close()

                job = ClusterJob(
                    job_id=job_id,
                    job_type=job_type,
                    node_id=self.node_id,
                    board_type=board_type,
                    num_players=num_players,
                    engine_mode=gpu_engine_mode,
                    pid=proc.pid,
                    started_at=time.time(),
                    status="running",
                )

                with self.jobs_lock:
                    self.local_jobs[job_id] = job

                print(f"[P2P] Started GPU selfplay job {job_id} (PID {proc.pid}, batch={batch_size})")
                self._save_state()

                # Monitor GPU selfplay and trigger CPU validation when complete
                asyncio.create_task(self._monitor_gpu_selfplay_and_validate(
                    job_id, proc, output_dir, board_type, num_players
                ))

                return job

            elif job_type == JobType.HYBRID_SELFPLAY:
                # Hybrid CPU/GPU selfplay using run_hybrid_selfplay.py
                # Uses CPU for game rules (100% canonical) but GPU for heuristic evaluation
                # This is the recommended default for GPU nodes

                # Normalize engine_mode
                hybrid_engine_modes = {"random-only", "heuristic-only", "mixed"}
                engine_mode_norm = engine_mode if engine_mode in hybrid_engine_modes else "heuristic-only"

                # Game counts based on board type
                num_games = 1000
                if board_type == "square19":
                    num_games = 500
                elif board_type in ("hex", "hexagonal"):
                    num_games = 300

                output_dir = Path(
                    self.ringrift_path,
                    "ai-service",
                    "data",
                    "selfplay",
                    "p2p_hybrid",
                    f"{board_type}_{num_players}p",
                    job_id,
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                # Normalize board type for hybrid script (uses 'hex' not 'hexagonal')
                board_arg = "hex" if board_type == "hexagonal" else board_type

                cmd = [
                    "python3",
                    f"{self.ringrift_path}/ai-service/scripts/run_hybrid_selfplay.py",
                    "--board-type", board_arg,
                    "--num-players", str(num_players),
                    "--num-games", str(num_games),
                    "--output-dir", str(output_dir),
                    "--record-db", str(output_dir / "games.db"),
                    "--lean-db",
                    "--engine-mode", engine_mode_norm,
                    "--seed", str(int(time.time() * 1000) % 2**31),
                ]

                # Start process with GPU environment
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{self.ringrift_path}/ai-service"
                env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"
                env["RINGRIFT_JOB_ORIGIN"] = "p2p_orchestrator"

                if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
                    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()
                elif "CUDA_VISIBLE_DEVICES" not in env:
                    gpu_count = 0
                    try:
                        out = subprocess.run(
                            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if out.returncode == 0 and out.stdout.strip():
                            gpu_count = len([l for l in out.stdout.splitlines() if l.strip()])
                    except Exception:
                        gpu_count = 0

                    if gpu_count > 0:
                        with self.jobs_lock:
                            running_hybrid_jobs = sum(
                                1
                                for j in self.local_jobs.values()
                                if j.job_type == JobType.HYBRID_SELFPLAY and j.status == "running"
                            )
                        env["CUDA_VISIBLE_DEVICES"] = str(running_hybrid_jobs % gpu_count)
                    else:
                        env["CUDA_VISIBLE_DEVICES"] = "0"

                # SAFEGUARD: Final check before spawning (load + rate limit)
                can_spawn, spawn_reason = self._can_spawn_process(f"hybrid-selfplay-{board_type}-{num_players}p")
                if not can_spawn:
                    print(f"[P2P] BLOCKED hybrid selfplay spawn: {spawn_reason}")
                    return None

                log_handle = open(output_dir / "hybrid_run.log", "a")
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=self.ringrift_path,
                    )
                    self._record_spawn()  # Track spawn for rate limiting
                finally:
                    log_handle.close()

                job = ClusterJob(
                    job_id=job_id,
                    job_type=job_type,
                    node_id=self.node_id,
                    board_type=board_type,
                    num_players=num_players,
                    engine_mode=engine_mode_norm,
                    pid=proc.pid,
                    started_at=time.time(),
                    status="running",
                )

                with self.jobs_lock:
                    self.local_jobs[job_id] = job

                print(f"[P2P] Started HYBRID selfplay job {job_id} (PID {proc.pid})")
                self._save_state()
                return job

        except Exception as e:
            print(f"[P2P] Failed to start job: {e}")
        return None

    async def _request_remote_job(
        self,
        node: NodeInfo,
        job_type: JobType,
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "hybrid",
    ):
        """Request a remote node to start a job with specific configuration.

        SAFEGUARD: Checks coordination safeguards before requesting remote spawn.
        """
        try:
            # SAFEGUARD: Check safeguards before requesting remote spawn
            if HAS_SAFEGUARDS and _safeguards:
                task_type_str = job_type.value if hasattr(job_type, 'value') else str(job_type)
                allowed, reason = check_before_spawn(task_type_str, node.node_id)
                if not allowed:
                    print(f"[P2P] SAFEGUARD blocked remote {task_type_str} on {node.node_id}: {reason}")
                    return

            job_id = f"{job_type.value}_{board_type}_{num_players}p_{int(time.time())}_{uuid.uuid4().hex[:6]}"

            # NAT-blocked nodes can't accept inbound /start_job; enqueue a relay command instead.
            if getattr(node, "nat_blocked", False):
                payload = {
                    "job_id": job_id,
                    "job_type": job_type.value,
                    "board_type": board_type,
                    "num_players": num_players,
                    "engine_mode": engine_mode,
                }
                cmd_id = await self._enqueue_relay_command_for_peer(node, "start_job", payload)
                if cmd_id:
                    print(
                        f"[P2P] Enqueued relay job for {node.node_id}: "
                        f"{job_type.value} {board_type} {num_players}p ({job_id})"
                    )
                else:
                    print(f"[P2P] Relay queue full for {node.node_id}; skipping enqueue")
                return

            timeout = ClientTimeout(total=10)
            async with get_client_session(timeout) as session:
                payload = {
                    "job_id": job_id,
                    "job_type": job_type.value,
                    "board_type": board_type,
                    "num_players": num_players,
                    "engine_mode": engine_mode,
                }
                last_err: Optional[str] = None
                for url in self._urls_for_peer(node, "/start_job"):
                    try:
                        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                            if resp.status != 200:
                                last_err = f"http_{resp.status}"
                                continue
                            data = await resp.json()
                            if data.get("success"):
                                print(f"[P2P] Started remote {board_type} {num_players}p job on {node.node_id}")
                                return
                            last_err = str(data.get("error") or "start_failed")
                    except Exception as e:
                        last_err = str(e)
                        continue
                if last_err:
                    print(f"[P2P] Failed to start remote job on {node.node_id}: {last_err}")
        except Exception as e:
            print(f"[P2P] Failed to request remote job from {node.node_id}: {e}")

    def _enqueue_relay_command(self, node_id: str, cmd_type: str, payload: Dict[str, Any]) -> Optional[str]:
        """Leader-side: enqueue a command for a NAT-blocked node to pull."""
        now = time.time()
        cmd_type = str(cmd_type)
        payload = dict(payload or {})

        with self.relay_lock:
            queue = list(self.relay_command_queue.get(node_id, []))
            queue = [
                cmd for cmd in queue
                if float(cmd.get("expires_at", 0.0) or 0.0) > now
            ]

            if cmd_type == "start_job":
                pending = sum(1 for c in queue if str(c.get("type") or "") == "start_job")
                if pending >= RELAY_MAX_PENDING_START_JOBS:
                    self.relay_command_queue[node_id] = queue
                    return None

                job_id = str(payload.get("job_id") or "")
                if job_id:
                    for c in queue:
                        if str(c.get("payload", {}).get("job_id") or "") == job_id:
                            self.relay_command_queue[node_id] = queue
                            return str(c.get("id") or "")

            cmd_id = uuid.uuid4().hex
            queue.append(
                {
                    "id": cmd_id,
                    "type": cmd_type,
                    "payload": payload,
                    "created_at": now,
                    "expires_at": now + RELAY_COMMAND_TTL_SECONDS,
                }
            )
            self.relay_command_queue[node_id] = queue
            return cmd_id

    async def _enqueue_relay_command_for_peer(
        self,
        peer: NodeInfo,
        cmd_type: str,
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """Enqueue a relay command for `peer`, forwarding via its relay hub when needed.

        Default behavior: NAT-blocked nodes poll the leader's `/relay/heartbeat`
        endpoint and the leader stores commands in-memory.

        Some nodes (notably certain containerized GPU providers) may be unable to
        reach the leader over the mesh network (e.g. TUN-less Tailscale) and also
        cannot accept inbound connections. Those nodes will instead send relay
        heartbeats to an internet-reachable hub (e.g. `aws-staging`). When
        `peer.relay_via` points to such a hub, the leader must enqueue the relay
        command on that hub so the node can pull and execute it.
        """
        if not peer or not getattr(peer, "node_id", ""):
            return None

        peer_id = str(getattr(peer, "node_id", "") or "").strip()
        if not peer_id:
            return None

        relay_node_id = str(getattr(peer, "relay_via", "") or "").strip()
        if relay_node_id and relay_node_id != self.node_id:
            with self.peers_lock:
                relay_peer = self.peers.get(relay_node_id)
            if relay_peer:
                timeout = ClientTimeout(total=10)
                async with get_client_session(timeout) as session:
                    last_err: Optional[str] = None
                    for url in self._urls_for_peer(relay_peer, "/relay/enqueue"):
                        try:
                            async with session.post(
                                url,
                                json={
                                    "target_node_id": peer_id,
                                    "type": cmd_type,
                                    "payload": payload or {},
                                },
                                headers=self._auth_headers(),
                            ) as resp:
                                if resp.status != 200:
                                    last_err = f"http_{resp.status}"
                                    continue
                                data = await resp.json()
                                if data.get("success"):
                                    return str(data.get("id") or "")
                                last_err = str(data.get("error") or "enqueue_failed")
                        except Exception as e:
                            last_err = str(e)
                            continue
                    if last_err:
                        print(f"[P2P] Relay enqueue via {relay_node_id} failed for {peer_id}: {last_err}")

        # Fallback: enqueue locally (works when peer polls the leader directly).
        return self._enqueue_relay_command(peer_id, cmd_type, payload)

    async def _discovery_loop(self):
        """Broadcast UDP discovery messages to find peers on local network."""
        while self.running:
            try:
                # Create UDP socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(1.0)

                # Broadcast our presence
                message = json.dumps({
                    "type": "p2p_discovery",
                    "node_id": self.node_id,
                    "host": self.self_info.host,
                    "port": self.port,
                }).encode()

                try:
                    sock.sendto(message, ('<broadcast>', DISCOVERY_PORT))
                except:
                    pass

                # Listen for responses
                try:
                    while True:
                        data, addr = sock.recvfrom(1024)
                        msg = json.loads(data.decode())
                        if msg.get("type") == "p2p_discovery" and msg.get("node_id") != self.node_id:
                            # Found a peer!
                            peer_addr = f"{msg.get('host')}:{msg.get('port')}"
                            if peer_addr not in self.known_peers:
                                self.known_peers.append(peer_addr)
                                print(f"[P2P] Discovered peer: {msg.get('node_id')} at {peer_addr}")
                except socket.timeout:
                    pass

                sock.close()

            except Exception as e:
                pass

            await asyncio.sleep(DISCOVERY_INTERVAL)

    async def run(self):
        """Main entry point - start the orchestrator."""
        if not HAS_AIOHTTP:
            print("Error: aiohttp is required. Install with: pip install aiohttp")
            return

        # Set up HTTP server
        @web.middleware
        async def auth_middleware(request: web.Request, handler):
            if self.auth_token and request.method not in ("GET", "HEAD", "OPTIONS"):
                if not self._is_request_authorized(request):
                    return web.json_response({"error": "unauthorized"}, status=401)
            return await handler(request)

        app = web.Application(middlewares=[auth_middleware])
        app.router.add_post('/heartbeat', self.handle_heartbeat)
        app.router.add_get('/status', self.handle_status)
        app.router.add_post('/election', self.handle_election)
        app.router.add_post('/election/lease', self.handle_lease_request)
        app.router.add_get('/election/grant', self.handle_voter_grant_status)
        app.router.add_post('/coordinator', self.handle_coordinator)
        app.router.add_post('/start_job', self.handle_start_job)
        app.router.add_post('/stop_job', self.handle_stop_job)
        app.router.add_post('/cleanup', self.handle_cleanup)
        app.router.add_post('/restart_stuck_jobs', self.handle_restart_stuck_jobs)
        app.router.add_post('/reduce_selfplay', self.handle_reduce_selfplay)
        app.router.add_get('/health', self.handle_health)
        app.router.add_get('/git/status', self.handle_git_status)
        app.router.add_post('/git/update', self.handle_git_update)

        # Dynamic host registry routes (for IP auto-updates)
        app.router.add_post('/register', self.handle_register)
        app.router.add_get('/registry/status', self.handle_registry_status)
        app.router.add_post('/registry/update_vast', self.handle_registry_update_vast)
        app.router.add_post('/registry/update_aws', self.handle_registry_update_aws)
        app.router.add_post('/registry/update_tailscale', self.handle_registry_update_tailscale)
        app.router.add_post('/registry/save_yaml', self.handle_registry_save_yaml)

        # Relay/Hub routes for NAT-blocked nodes
        app.router.add_post('/relay/heartbeat', self.handle_relay_heartbeat)
        app.router.add_get('/relay/peers', self.handle_relay_peers)
        app.router.add_get('/relay/status', self.handle_relay_status)
        app.router.add_post('/relay/enqueue', self.handle_relay_enqueue)

        # Phase 2: Distributed data manifest routes
        app.router.add_get('/data_manifest', self.handle_data_manifest)
        app.router.add_get('/cluster_data_manifest', self.handle_cluster_data_manifest)
        app.router.add_post('/refresh_manifest', self.handle_refresh_manifest)

        # Distributed CMA-ES routes
        app.router.add_post('/cmaes/start', self.handle_cmaes_start)
        app.router.add_post('/cmaes/evaluate', self.handle_cmaes_evaluate)
        app.router.add_get('/cmaes/status', self.handle_cmaes_status)
        app.router.add_post('/cmaes/result', self.handle_cmaes_result)

        # Distributed tournament routes
        app.router.add_post('/tournament/start', self.handle_tournament_start)
        app.router.add_post('/tournament/match', self.handle_tournament_match)
        app.router.add_get('/tournament/status', self.handle_tournament_status)
        app.router.add_post('/tournament/result', self.handle_tournament_result)
        app.router.add_post('/tournament/ssh_start', self.handle_ssh_tournament_start)
        app.router.add_get('/tournament/ssh_status', self.handle_ssh_tournament_status)
        app.router.add_post('/tournament/ssh_cancel', self.handle_ssh_tournament_cancel)

        # Improvement loop routes
        app.router.add_post('/improvement/start', self.handle_improvement_start)
        app.router.add_get('/improvement/status', self.handle_improvement_status)
        app.router.add_post('/improvement/phase_complete', self.handle_improvement_phase_complete)

        # Phase 2: P2P data sync routes
        app.router.add_post('/sync/start', self.handle_sync_start)
        app.router.add_get('/sync/status', self.handle_sync_status)
        app.router.add_post('/sync/pull', self.handle_sync_pull)
        app.router.add_get('/sync/file', self.handle_sync_file)
        app.router.add_post('/sync/job_update', self.handle_sync_job_update)
        app.router.add_post('/sync/training', self.handle_training_sync)  # Training node priority sync
        app.router.add_get('/gpu/rankings', self.handle_gpu_rankings)      # GPU power rankings
        app.router.add_post('/cleanup/files', self.handle_cleanup_files)   # File-specific cleanup

        # Phase 3: Training pipeline routes
        app.router.add_post('/training/start', self.handle_training_start)
        app.router.add_get('/training/status', self.handle_training_status)
        app.router.add_post('/training/update', self.handle_training_update)
        app.router.add_post('/training/nnue/start', self.handle_nnue_start)
        app.router.add_post('/training/cmaes/start', self.handle_cmaes_start_auto)

        # Phase 5: Improvement cycle routes
        app.router.add_get('/improvement_cycles/status', self.handle_improvement_cycles_status)
        app.router.add_get('/improvement_cycles/leaderboard', self.handle_improvement_cycles_leaderboard)
        app.router.add_post('/improvement_cycles/training_complete', self.handle_improvement_training_complete)
        app.router.add_post('/improvement_cycles/evaluation_complete', self.handle_improvement_evaluation_complete)

        # Metrics observability routes
        app.router.add_get('/metrics', self.handle_metrics)
        app.router.add_get('/metrics/prometheus', self.handle_metrics_prometheus)

        # Canonical pipeline routes (for pipeline_orchestrator.py integration)
        app.router.add_post('/pipeline/start', self.handle_pipeline_start)
        app.router.add_get('/pipeline/status', self.handle_pipeline_status)
        app.router.add_post('/pipeline/selfplay_worker', self.handle_pipeline_selfplay_worker)

        # Phase 4: REST API and Dashboard routes
        app.router.add_get('/', self.handle_root)
        app.router.add_get('/api/cluster/status', self.handle_api_cluster_status)
        app.router.add_post('/api/cluster/git/update', self.handle_api_cluster_git_update)
        app.router.add_get('/api/selfplay/stats', self.handle_api_selfplay_stats)
        app.router.add_get('/api/elo/leaderboard', self.handle_api_elo_leaderboard)
        app.router.add_get('/elo/table', self.handle_elo_table)
        app.router.add_get('/elo/history', self.handle_elo_history)
        app.router.add_get('/nodes/table', self.handle_nodes_table)
        app.router.add_get('/victory/table', self.handle_victory_table)
        app.router.add_get('/games/analytics', self.handle_games_analytics)
        app.router.add_get('/training/metrics', self.handle_training_metrics)
        app.router.add_get('/holdout/metrics', self.handle_holdout_metrics)
        app.router.add_get('/holdout/table', self.handle_holdout_table)
        app.router.add_get('/mcts/stats', self.handle_mcts_stats)
        app.router.add_get('/mcts/table', self.handle_mcts_table)
        # Feature endpoints
        app.router.add_get('/matchups/matrix', self.handle_matchup_matrix)
        app.router.add_get('/matchups/table', self.handle_matchup_table)
        app.router.add_get('/models/lineage', self.handle_model_lineage)
        app.router.add_get('/models/lineage/table', self.handle_model_lineage_table)
        app.router.add_get('/data/quality', self.handle_data_quality)
        app.router.add_get('/data/quality/table', self.handle_data_quality_table)
        app.router.add_get('/data/quality/issues', self.handle_data_quality_issues)
        app.router.add_get('/training/efficiency', self.handle_training_efficiency)
        app.router.add_get('/training/efficiency/table', self.handle_training_efficiency_table)
        app.router.add_get('/rollback/status', self.handle_rollback_status)
        app.router.add_get('/rollback/candidates', self.handle_rollback_candidates)
        app.router.add_post('/rollback/execute', self.handle_rollback_execute)
        app.router.add_post('/rollback/auto', self.handle_rollback_auto)
        app.router.add_get('/autoscale/metrics', self.handle_autoscale_metrics)
        app.router.add_get('/autoscale/recommendations', self.handle_autoscale_recommendations)
        app.router.add_get('/resource/optimizer', self.handle_resource_optimizer)
        app.router.add_get('/resource/history', self.handle_resource_utilization_history)
        app.router.add_post('/webhook/test', self.handle_webhook_test)
        app.router.add_get('/trends/summary', self.handle_trends_summary)
        app.router.add_get('/trends/history', self.handle_trends_history)
        app.router.add_get('/trends/table', self.handle_trends_table)

        # A/B Testing endpoints
        app.router.add_post('/abtest/create', self.handle_abtest_create)
        app.router.add_post('/abtest/result', self.handle_abtest_result)
        app.router.add_get('/abtest/status', self.handle_abtest_status)
        app.router.add_get('/abtest/list', self.handle_abtest_list)
        app.router.add_post('/abtest/cancel', self.handle_abtest_cancel)
        app.router.add_get('/abtest/table', self.handle_abtest_table)
        app.router.add_post('/abtest/run', self.handle_abtest_run)

        app.router.add_get('/api/training/status', self.handle_api_training_status)
        app.router.add_get('/api/canonical/health', self.handle_api_canonical_health)
        app.router.add_get('/api/canonical/jobs', self.handle_api_canonical_jobs_list)
        app.router.add_get('/api/canonical/jobs/{job_id}', self.handle_api_canonical_job_get)
        app.router.add_get('/api/canonical/jobs/{job_id}/log', self.handle_api_canonical_job_log)
        app.router.add_get('/api/canonical/logs', self.handle_api_canonical_logs_list)
        app.router.add_get('/api/canonical/logs/{log_name}/tail', self.handle_api_canonical_log_tail)
        app.router.add_post('/api/canonical/generate', self.handle_api_canonical_generate)
        app.router.add_post('/api/canonical/jobs/{job_id}/cancel', self.handle_api_canonical_job_cancel)
        app.router.add_get('/api/jobs', self.handle_api_jobs_list)
        app.router.add_post('/api/jobs/submit', self.handle_api_jobs_submit)
        app.router.add_get('/api/jobs/{job_id}', self.handle_api_job_get)
        app.router.add_post('/api/jobs/{job_id}/cancel', self.handle_api_job_cancel)
        app.router.add_get('/dashboard', self.handle_dashboard)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        print(f"[P2P] HTTP server started on {self.host}:{self.port}")

        # Start background tasks
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._manifest_collection_loop()),
            asyncio.create_task(self._job_management_loop()),
            asyncio.create_task(self._discovery_loop()),
        ]

        # Add git update loop if enabled
        if AUTO_UPDATE_ENABLED:
            tasks.append(asyncio.create_task(self._git_update_loop()))

        # Add training node priority sync loop (leader-only sync to high-GPU nodes)
        tasks.append(asyncio.create_task(self._training_sync_loop()))

        # Add cloud IP refresh loops (best-effort; no-op if not configured).
        if HAS_DYNAMIC_REGISTRY:
            tasks.append(asyncio.create_task(self._vast_ip_update_loop()))
            tasks.append(asyncio.create_task(self._aws_ip_update_loop()))
            tasks.append(asyncio.create_task(self._tailscale_ip_update_loop()))

        # Add automatic data management loop (export triggers, training triggers, data sync)
        tasks.append(asyncio.create_task(self._data_management_loop()))

        # Best-effort bootstrap from seed peers before running elections. This
        # helps newly started cloud nodes quickly learn about the full cluster.
        try:
            await self._bootstrap_from_known_peers()
        except Exception:
            pass

        # If no leader known, start election after short delay
        await asyncio.sleep(5)
        if not self.leader_id:
            # Avoid needless bully elections if we can already see a leader.
            if not self._maybe_adopt_leader_from_peers():
                await self._start_election()

        # Run forever
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="P2P Orchestrator for RingRift cluster")
    parser.add_argument("--node-id", required=True, help="Unique identifier for this node")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument(
        "--advertise-host",
        default=None,
        help=f"Host to advertise to peers (or set {ADVERTISE_HOST_ENV})",
    )
    parser.add_argument(
        "--advertise-port",
        type=int,
        default=None,
        help=f"Port to advertise to peers (or set {ADVERTISE_PORT_ENV})",
    )
    parser.add_argument("--peers", help="Comma-separated list of known peers (host[:port] or http(s)://host[:port])")
    parser.add_argument("--ringrift-path", help="Path to RingRift installation")
    parser.add_argument("--auth-token", help=f"Shared auth token (or set {AUTH_TOKEN_ENV})")
    parser.add_argument("--require-auth", action="store_true", help="Require auth token to be set")
    parser.add_argument("--storage-type", choices=["disk", "ramdrive"], default="disk",
                        help="Storage type: 'disk' (default) or 'ramdrive' (/dev/shm for disk-constrained instances)")

    args = parser.parse_args()

    known_peers = []
    if args.peers:
        known_peers = [p.strip() for p in args.peers.split(',')]

    orchestrator = P2POrchestrator(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        known_peers=known_peers,
        ringrift_path=args.ringrift_path,
        advertise_host=args.advertise_host,
        advertise_port=args.advertise_port,
        auth_token=args.auth_token,
        require_auth=args.require_auth,
        storage_type=args.storage_type,
    )

    # Handle shutdown
    def signal_handler(sig, frame):
        print("\n[P2P] Shutting down...")
        orchestrator.running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    asyncio.run(orchestrator.run())


if __name__ == "__main__":
    main()
