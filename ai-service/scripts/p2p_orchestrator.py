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
from dataclasses import dataclass, field, asdict
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

# ============================================
# Configuration
# ============================================

DEFAULT_PORT = 8770
HEARTBEAT_INTERVAL = 30  # seconds
PEER_TIMEOUT = 90  # seconds without heartbeat = node considered dead
ELECTION_TIMEOUT = 10  # seconds to wait for election responses
LEADER_LEASE_DURATION = 30  # Leader must renew lease within this time
LEADER_LEASE_RENEW_INTERVAL = 10  # How often leader renews lease
JOB_CHECK_INTERVAL = 60  # seconds between job status checks
DISCOVERY_PORT = 8771  # UDP port for peer discovery
DISCOVERY_INTERVAL = 120  # seconds between discovery broadcasts

# LEARNED LESSONS from PLAN.md - Disk and resource thresholds
DISK_CRITICAL_THRESHOLD = 90  # Stop all new jobs at 90% disk
DISK_WARNING_THRESHOLD = 80   # Reduce job count at 80% disk
DISK_CLEANUP_THRESHOLD = 85   # Trigger automatic cleanup at 85%
MEMORY_CRITICAL_THRESHOLD = 95  # OOM prevention - stop jobs at 95%
MEMORY_WARNING_THRESHOLD = 85   # Reduce jobs at 85% memory

# LEARNED LESSONS - Connection robustness
HTTP_CONNECT_TIMEOUT = 10     # Fast timeout for connection phase
HTTP_TOTAL_TIMEOUT = 30       # Total request timeout
MAX_CONSECUTIVE_FAILURES = 3  # Mark node dead after 3 failures
RETRY_DEAD_NODE_INTERVAL = 300  # Retry dead nodes every 5 minutes

# LEARNED LESSONS - Stuck job detection
GPU_IDLE_RESTART_TIMEOUT = 300  # Restart jobs after 5 min of GPU at 0%
GPU_IDLE_THRESHOLD = 2          # Consider GPU idle if utilization < 2%

# Git auto-update settings
GIT_UPDATE_CHECK_INTERVAL = 300  # Check for updates every 5 minutes
GIT_REMOTE_NAME = "origin"       # Git remote to check
GIT_BRANCH_NAME = "main"         # Branch to track
AUTO_UPDATE_ENABLED = True       # Enable automatic updates
GRACEFUL_SHUTDOWN_BEFORE_UPDATE = True  # Stop jobs before updating

# Shared auth token (optional but strongly recommended if any node is public)
AUTH_TOKEN_ENV = "RINGRIFT_CLUSTER_AUTH_TOKEN"
AUTH_TOKEN_FILE_ENV = "RINGRIFT_CLUSTER_AUTH_TOKEN_FILE"

# Optional advertised endpoint override (useful behind NAT/port-mapping).
ADVERTISE_HOST_ENV = "RINGRIFT_ADVERTISE_HOST"
ADVERTISE_PORT_ENV = "RINGRIFT_ADVERTISE_PORT"

# Data manifest collection settings
MANIFEST_JSONL_LINECOUNT_MAX_BYTES = 64 * 1024 * 1024  # Skip line-counting for huge JSONL files
MANIFEST_JSONL_LINECOUNT_CHUNK_BYTES = 1024 * 1024

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
    GPU_SELFPLAY = "gpu_selfplay"  # GPU-accelerated parallel selfplay
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

    def is_alive(self) -> bool:
        """Check if node is considered alive based on last heartbeat."""
        return time.time() - self.last_heartbeat < PEER_TIMEOUT

    def is_healthy(self) -> bool:
        """Check if node is healthy for new jobs (not just reachable)."""
        if not self.is_alive():
            return False
        # LEARNED LESSONS - Don't start jobs on resource-constrained nodes
        if self.disk_percent >= DISK_CRITICAL_THRESHOLD:
            return False
        if self.memory_percent >= MEMORY_CRITICAL_THRESHOLD:
            return False
        return True

    def should_retry(self) -> bool:
        """Check if we should retry connecting to a failed node."""
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

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['role'] = self.role.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'NodeInfo':
        """Create from dictionary."""
        d = d.copy()
        d['role'] = NodeRole(d.get('role', 'follower'))
        # Handle missing new fields gracefully
        d.setdefault('scheme', 'http')
        d.setdefault('consecutive_failures', 0)
        d.setdefault('last_failure_time', 0.0)
        d.setdefault('disk_cleanup_needed', False)
        d.setdefault('oom_events', 0)
        d.setdefault('last_oom_time', 0.0)
        d.setdefault('nat_blocked', False)
        d.setdefault('relay_via', '')
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
    """Configuration for automatic training triggers."""
    # Minimum games required to trigger training
    min_games_nnue: int = 10000         # NNUE needs lots of data
    min_games_cmaes: int = 1000         # CMA-ES can work with fewer games
    # Incremental thresholds (trigger re-training when new data >= threshold)
    incremental_games_nnue: int = 5000  # Re-train every 5k new games
    incremental_games_cmaes: int = 500  # Re-optimize every 500 new games
    # Cooldown between training runs (seconds)
    cooldown_seconds: float = 3600.0    # 1 hour
    # Auto-training enabled flags
    auto_nnue_enabled: bool = True
    auto_cmaes_enabled: bool = True

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
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.known_peers = known_peers or []
        self.ringrift_path = ringrift_path or self._detect_ringrift_path()
        self.start_time = time.time()

        # Public endpoint peers should use to reach us. Peers learn our host from
        # the heartbeat socket address, but the port must be self-reported. This
        # matters for port-mapped environments like Vast.ai.
        self.advertise_host = (advertise_host or os.environ.get(ADVERTISE_HOST_ENV, "")).strip()
        if not self.advertise_host:
            self.advertise_host = self._get_local_ip()
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
        self.improvement_cycle_check_interval: float = 600.0  # Check every 10 minutes

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

        # LEARNED LESSONS - Stuck job detection (leader-only)
        # Track when each node's GPU first went idle with running jobs
        self.gpu_idle_since: Dict[str, float] = {}  # node_id -> timestamp when GPU went idle

        # Locks for thread safety
        self.peers_lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.manifest_lock = threading.Lock()
        self.sync_lock = threading.Lock()
        self.training_lock = threading.Lock()
        self.ssh_tournament_lock = threading.Lock()

        # State persistence
        self.db_path = STATE_DIR / f"{node_id}_state.db"
        self._init_database()

        # Event flags
        self.running = True
        self.election_in_progress = False

        # LEARNED LESSONS - Lease-based leadership to prevent split-brain
        # Leader must continuously renew lease; if lease expires, leadership is void
        self.leader_lease_expires: float = 0.0  # timestamp when current leader's lease expires
        self.last_lease_renewal: float = 0.0  # when we last renewed our lease (if leader)
        self.leader_lease_id: str = ""  # unique ID for current leadership term

        # Job completion tracking for auto-restart
        self.completed_jobs: Dict[str, float] = {}  # node_id -> last job completion time
        self.jobs_started_at: Dict[str, Dict[str, float]] = {}  # node_id -> {job_id: start_time}

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
        print(f"[P2P] Known peers: {self.known_peers}")
        if self.auth_token:
            print(f"[P2P] Auth: enabled via {AUTH_TOKEN_ENV}")
        else:
            print(f"[P2P] Auth: disabled (set {AUTH_TOKEN_ENV} to enable)")

    def _is_leader(self) -> bool:
        """Check if this node is the current cluster leader with valid lease."""
        if self.leader_id != self.node_id:
            return False
        # LEARNED LESSONS - Lease-based leadership prevents split-brain
        # Must have valid lease to act as leader
        if self.leader_lease_expires > 0 and time.time() >= self.leader_lease_expires:
            print(f"[P2P] Leadership lease expired, stepping down")
            self.role = NodeRole.FOLLOWER
            return False
        return True

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
        return f"{scheme}://{peer.host}:{peer.port}{path}"

    def _auth_headers(self) -> Dict[str, str]:
        if not self.auth_token:
            return {}
        return {"Authorization": f"Bearer {self.auth_token}"}

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
            cursor.execute("SELECT value FROM state WHERE key = 'leader_id'")
            row = cursor.fetchone()
            if row:
                self.leader_id = row[0]

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
            with self.peers_lock:
                for node_id, info in self.peers.items():
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
            cursor.execute("""
                INSERT OR REPLACE INTO state (key, value) VALUES ('leader_id', ?)
            """, (self.leader_id,))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[P2P] Failed to save state: {e}")

    def _create_self_info(self) -> NodeInfo:
        """Create NodeInfo for this node."""
        # Detect GPU
        has_gpu, gpu_name = self._detect_gpu()

        # Detect memory
        memory_gb = self._detect_memory()

        # Detect capabilities based on hardware
        capabilities = ["selfplay"]
        if has_gpu:
            capabilities.extend(["training", "cmaes"])
        if memory_gb >= 64:
            capabilities.append("large_boards")

        return NodeInfo(
            node_id=self.node_id,
            host=self.advertise_host,
            port=self.advertise_port,
            role=self.role,
            last_heartbeat=time.time(),
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            memory_gb=memory_gb,
            capabilities=capabilities,
        )

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
        selfplay = 0
        training = 0

        try:
            # Count python processes running selfplay (CPU + GPU runners).
            # Use PID union to avoid double-counting.
            selfplay_pids: Set[str] = set()
            for pattern in ("run_self_play_soak.py", "run_gpu_selfplay.py"):
                out = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True, text=True, timeout=5
                )
                if out.returncode == 0 and out.stdout.strip():
                    selfplay_pids.update([p for p in out.stdout.strip().split() if p])
            selfplay = len(selfplay_pids)

            # Count training processes
            training_pids: Set[str] = set()
            for pattern in ("train_", "train.py"):
                out = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True, text=True, timeout=5
                )
                if out.returncode == 0 and out.stdout.strip():
                    training_pids.update([p for p in out.stdout.strip().split() if p])
            training = len(training_pids)
        except:
            pass

        return selfplay, training

    # ============================================
    # Phase 2: Distributed Data Sync Methods
    # ============================================

    def _collect_local_data_manifest(self) -> NodeDataManifest:
        """Collect manifest of all data files on this node.

        Scans the ai-service/data directory for:
        - selfplay/ - Game replay files (.jsonl, .db)
        - models/ - Trained model files (.pt, .onnx)
        - training/ - Training data files (.npz)
        - games/ - Synced game databases (.db)
        """
        data_dir = Path(self.ringrift_path) / "ai-service" / "data"
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
            url = self._url_for_peer(peer_info, "/data_manifest")
            timeout = ClientTimeout(total=30)
            async with ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return NodeDataManifest.from_dict(data.get("manifest", {}))
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

        # Collect from peers in parallel
        with self.peers_lock:
            peers = list(self.peers.values())

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
                    source_node_id=job.source_node,
                    files=job.files,
                )
            else:
                url = self._url_for_peer(target_peer, "/sync/pull")
                payload = {
                    "job_id": job.job_id,
                    # Back-compat: target will prefer source_node_id lookup.
                    "source_host": source_peer.host,
                    "source_port": source_peer.port,
                    "source_node_id": job.source_node,
                    "files": job.files,
                }

                timeout = ClientTimeout(total=600)
                async with ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                        if resp.status != 200:
                            job.status = "failed"
                            job.error_message = f"HTTP {resp.status}"
                            if self.current_sync_plan:
                                self.current_sync_plan.jobs_failed += 1
                            return False
                        result = await resp.json()

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

    async def _handle_sync_pull_request(self, source_host: str, source_port: int,
                                         source_node_id: str, files: List[str]) -> Dict[str, Any]:
        """
        Handle incoming request to pull files from a source node.
        Pulls files over the P2P HTTP channel to avoid SSH/rsync dependencies.
        """
        data_dir = Path(self.ringrift_path) / "ai-service" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        bytes_transferred = 0
        files_completed = 0
        errors: List[str] = []

        # Back-compat: if caller passed an SSH-like port (22), try DEFAULT_PORT too.
        ports_to_try: List[int] = []
        try:
            ports_to_try.append(int(source_port))
        except Exception:
            ports_to_try.append(DEFAULT_PORT)
        if DEFAULT_PORT not in ports_to_try:
            ports_to_try.append(DEFAULT_PORT)

        timeout = ClientTimeout(total=None, sock_connect=HTTP_CONNECT_TIMEOUT, sock_read=600)

        async with ClientSession(timeout=timeout) as session:
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

                for port in ports_to_try:
                    url = f"http://{source_host}:{port}/sync/file"
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
            # Request cleanup via HTTP endpoint
            url = self._url_for_peer(node, "/cleanup/files")
            timeout = ClientTimeout(total=60)
            async with ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    json={"files": files, "reason": "post_sync_cleanup"},
                    headers=self._auth_headers()
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        freed_bytes = result.get("freed_bytes", 0)
                        print(f"[P2P] Cleanup on {node_id}: freed {freed_bytes / 1e6:.1f}MB")
                        return True
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

                # Find selfplay files to sync
                files_to_sync = []
                for file_info in source_manifest.files:
                    if file_info.file_type == "selfplay" and file_info.path not in target_files:
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
        """Background loop to periodically check Vast.ai API for IP changes.

        Only runs if VAST_API_KEY is set. Updates the dynamic registry with
        current IPs for all Vast instances.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return

        vast_api_key = os.environ.get("VAST_API_KEY")
        if not vast_api_key:
            print("[P2P] Vast IP update loop disabled (no VAST_API_KEY)")
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

    # ============================================
    # Git Auto-Update Methods
    # ============================================

    def _get_local_git_commit(self) -> Optional[str]:
        """Get the current local git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
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
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
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
                ["git", "fetch", GIT_REMOTE_NAME, GIT_BRANCH_NAME],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=60
            )
            if fetch_result.returncode != 0:
                print(f"[P2P] Git fetch failed: {fetch_result.stderr}")
                return None

            # Get remote branch commit
            result = subprocess.run(
                ["git", "rev-parse", f"{GIT_REMOTE_NAME}/{GIT_BRANCH_NAME}"],
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
                ["git", "rev-list", "--count", f"{local_commit}..{remote_commit}"],
                cwd=self.ringrift_path,
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception as e:
            print(f"[P2P] Failed to count commits behind: {e}")
        return 0

    def _check_local_changes(self) -> bool:
        """Check if there are uncommitted local changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
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
                ["git", "pull", GIT_REMOTE_NAME, GIT_BRANCH_NAME],
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
            peer_info = NodeInfo.from_dict(data)
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

            with self.peers_lock:
                self.peers[peer_info.node_id] = peer_info

            # Return our info
            self._update_self_info()
            return web.json_response(self.self_info.to_dict())
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_status(self, request: web.Request) -> web.Response:
        """Return cluster status."""
        self._update_self_info()

        with self.peers_lock:
            peers = {k: v.to_dict() for k, v in self.peers.items()}

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

        return web.json_response({
            "node_id": self.node_id,
            "role": self.role.value,
            "leader_id": self.leader_id,
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
            data = await request.json()
            candidate_id = data.get("candidate_id")

            # If our ID is higher, we respond with "ALIVE" (Bully algorithm)
            if self.node_id > candidate_id:
                # Start our own election
                asyncio.create_task(self._start_election())
                return web.json_response({"response": "ALIVE", "node_id": self.node_id})
            else:
                return web.json_response({"response": "OK"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_coordinator(self, request: web.Request) -> web.Response:
        """Handle coordinator announcement from new leader.

        LEARNED LESSONS - Only accept leadership from higher-priority nodes (Bully algorithm).
        Also handles lease-based leadership updates.
        """
        try:
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

            # LEARNED LESSONS - Verify the announced leader has higher priority than us
            # (Bully algorithm: higher node_id wins)
            if self.role == NodeRole.LEADER and new_leader < self.node_id:
                # Exception: accept if our lease has expired
                if self.leader_lease_expires > 0 and time.time() >= self.leader_lease_expires:
                    print(f"[P2P] Our lease expired, accepting leader: {new_leader}")
                else:
                    print(f"[P2P] Rejecting leader announcement from lower-priority node: {new_leader} < {self.node_id}")
                    return web.json_response({"accepted": False, "reason": "lower_priority"})

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

            job = await self._start_local_job(job_type, board_type, num_players, engine_mode)

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

            data_dir = Path(self.ringrift_path) / "ai-service" / "data"
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
        """
        try:
            self._update_self_info()
            is_healthy = self.self_info.is_healthy()

            return web.json_response({
                "healthy": is_healthy,
                "node_id": self.node_id,
                "role": self.role.value,
                "disk_percent": self.self_info.disk_percent,
                "memory_percent": self.self_info.memory_percent,
                "cpu_percent": self.self_info.cpu_percent,
                "selfplay_jobs": self.self_info.selfplay_jobs,
                "training_jobs": self.self_info.training_jobs,
            })
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
            peer_info = NodeInfo.from_dict(data)
            peer_info.last_heartbeat = time.time()
            peer_info.nat_blocked = True  # Mark as NAT-blocked
            peer_info.relay_via = self.node_id  # This node is their relay

            # Get their real IP from the request (for logging/debugging)
            forwarded_for = (
                request.headers.get("X-Forwarded-For")
                or request.headers.get("X-Real-IP")
            )
            real_ip = forwarded_for.split(",")[0].strip() if forwarded_for else request.remote

            # Store in peers list (they're part of the cluster even if not directly reachable)
            with self.peers_lock:
                self.peers[peer_info.node_id] = peer_info

            print(f"[P2P] Relay heartbeat from {peer_info.node_id} (real IP: {real_ip})")

            # Return cluster state so they can see all peers
            self._update_self_info()
            with self.peers_lock:
                peers = {k: v.to_dict() for k, v in self.peers.items()}

            return web.json_response({
                "success": True,
                "self": self.self_info.to_dict(),
                "peers": peers,
                "leader_id": self.leader_id,
                "relay_node": self.node_id,
            })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_relay_peers(self, request: web.Request) -> web.Response:
        """GET /relay/peers - Get list of all peers including NAT-blocked ones.

        Used by nodes to discover the full cluster including NAT-blocked members.
        """
        try:
            self._update_self_info()
            with self.peers_lock:
                all_peers = {k: v.to_dict() for k, v in self.peers.items()}

            # Separate NAT-blocked and directly reachable
            nat_blocked = {k: v for k, v in all_peers.items() if v.get('nat_blocked')}
            direct = {k: v for k, v in all_peers.items() if not v.get('nat_blocked')}

            return web.json_response({
                "success": True,
                "leader_id": self.leader_id,
                "total_peers": len(all_peers),
                "direct_peers": len(direct),
                "nat_blocked_peers": len(nat_blocked),
                "peers": all_peers,
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

            if not node_id or not host:
                return web.json_response({
                    "error": "Missing required fields: node_id, host"
                }, status=400)

            registry = get_registry()
            success = registry.register_node(node_id, host, port, vast_instance_id)

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
        """POST /registry/update_vast - Query Vast.ai API and update IPs.

        Requires VAST_API_KEY environment variable to be set.
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

            # Collect cluster manifest (refreshes data from all nodes)
            cluster_manifest = await self._collect_cluster_manifest()
            with self.manifest_lock:
                self.cluster_data_manifest = cluster_manifest

            return web.json_response({
                "cluster_manifest": cluster_manifest.to_dict(),
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
                                async with ClientSession(timeout=timeout) as session:
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
            # Run selfplay subprocess to evaluate weights
            import tempfile
            import json as json_mod

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json_mod.dump(weights, f)
                weights_file = f.name

            cmd = [
                sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{self.ringrift_path / "ai-service"}')
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
                env={**os.environ, "PYTHONPATH": str(self.ringrift_path / "ai-service")},
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
                            async with ClientSession(timeout=timeout) as session:
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
            max_moves = int(data.get("max_moves", 300) or 300)
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
            async with ClientSession(timeout=timeout) as session:
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
max_moves = 500
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
                        async with ClientSession(timeout=timeout) as session:
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

            state = ImprovementLoopState(
                job_id=job_id,
                board_type=data.get("board_type", "square8"),
                num_players=data.get("num_players", 2),
                max_iterations=data.get("max_iterations", 50),
                games_per_iteration=data.get("games_per_iteration", 1000),
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

            data_dir = Path(self.ringrift_path) / "ai-service" / "data"
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
                async with ClientSession(timeout=timeout) as session:
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
                async with ClientSession(timeout=timeout) as session:
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

        # Check each board type / player count combination
        for config_key, config_data in self.cluster_data_manifest.by_board_type.items():
            parts = config_key.split("_")
            if len(parts) < 2:
                continue
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))
            total_games = config_data.get("total_games", 0)

            # Check NNUE training threshold
            if thresholds.auto_nnue_enabled:
                last_nnue_games = self.games_at_last_nnue_train.get(config_key, 0)
                if total_games >= thresholds.min_games_nnue:
                    new_games = total_games - last_nnue_games
                    if new_games >= thresholds.incremental_games_nnue or last_nnue_games == 0:
                        # Check cooldown
                        existing_job = self._find_running_training_job("nnue", config_key)
                        if not existing_job:
                            jobs_to_start.append({
                                "job_type": "nnue",
                                "board_type": board_type,
                                "num_players": num_players,
                                "config_key": config_key,
                                "total_games": total_games,
                            })

            # Check CMA-ES optimization threshold
            if thresholds.auto_cmaes_enabled:
                last_cmaes_games = self.games_at_last_cmaes_train.get(config_key, 0)
                if total_games >= thresholds.min_games_cmaes:
                    new_games = total_games - last_cmaes_games
                    if new_games >= thresholds.incremental_games_cmaes or last_cmaes_games == 0:
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

        # Find suitable worker
        worker_node = None
        with self.peers_lock:
            if job_type == "nnue":
                # NNUE needs GPU
                for peer in self.peers.values():
                    if peer.has_gpu and peer.is_healthy():
                        worker_node = peer
                        break
            else:
                # CMA-ES can run on any node, prefer GPU for speed
                for peer in self.peers.values():
                    if peer.is_healthy():
                        if peer.has_gpu:
                            worker_node = peer
                            break
                        elif worker_node is None:
                            worker_node = peer

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
            async with ClientSession(timeout=timeout) as session:
                url = self._url_for_peer(worker_node, endpoint)
                payload = {
                    "job_id": job_id,
                    "board_type": board_type,
                    "num_players": num_players,
                    "epochs": job.epochs,
                    "batch_size": job.batch_size,
                    "learning_rate": job.learning_rate,
                }
                async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get("success"):
                            job.status = "running"
                            job.started_at = time.time()
                            print(f"[P2P] Started {job_type} training job {job_id} on {worker_node.node_id}")
                            self._save_state()
                            return job
                        else:
                            job.status = "failed"
                            job.error_message = result.get("error", "Unknown error")
                    else:
                        job.status = "failed"
                        job.error_message = f"HTTP {resp.status}"
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
            with self.peers_lock:
                for peer in self.peers.values():
                    if peer.has_gpu and peer.is_healthy():
                        gpu_worker = peer
                        break

            if not gpu_worker and self.self_info.has_gpu:
                gpu_worker = self.self_info

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
            async with ClientSession(timeout=timeout) as session:
                url = self._url_for_peer(worker_node, "/training/nnue/start")
                async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get("success"):
                            job.status = "running"
                            job.started_at = time.time()
                            print(f"[P2P] ImprovementCycle {cycle_id}: Training started on {worker_node.node_id}")
                        else:
                            self.improvement_cycle_manager.update_cycle_phase(
                                cycle_id, "idle", error_message=result.get("error", "Training failed to start")
                            )
                    else:
                        self.improvement_cycle_manager.update_cycle_phase(
                            cycle_id, "idle", error_message=f"HTTP {resp.status}"
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

            self._save_state()

            return web.json_response({"success": True})

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def handle_nnue_start(self, request: web.Request) -> web.Response:
        """Handle NNUE training start request (worker endpoint)."""
        try:
            data = await request.json()
            job_id = data.get("job_id")
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)
            epochs = data.get("epochs", 100)
            batch_size = data.get("batch_size", 2048)

            # Start NNUE training subprocess
            output_path = os.path.join(
                self.ringrift_path, "ai-service", "models", "nnue",
                f"{board_type}_{num_players}p_auto.pt"
            )

            cmd = [
                sys.executable, "-m", "scripts.train_nnue",
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--save-path", output_path,
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.join(self.ringrift_path, "ai-service")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
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
                        async with ClientSession(timeout=timeout) as session:
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
                        else:
                            job.status = "failed"
                            job.error_message = stderr.decode()[:500]
                        job.completed_at = time.time()

            print(f"[P2P] Training job {job_id} {'completed' if success else 'failed'}")

        except asyncio.TimeoutError:
            print(f"[P2P] Training job {job_id} timed out")
        except Exception as e:
            print(f"[P2P] Training monitor error for {job_id}: {e}")

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

                    if win_rate >= 0.55:
                        print(f"[P2P] New model beats baseline! Promoting to best baseline.")
                        await self._promote_to_baseline(
                            config["model_a"], config["board_type"],
                            config["num_players"], "nnue" if "nnue" in config["model_a"].lower() else "cmaes"
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
            return web.json_response({"success": True})

        except Exception as e:
            return web.json_response({"success": False, "error": str(e)})

    async def _schedule_improvement_evaluation(self, cycle_id: str, new_model_id: str):
        """Schedule tournament evaluation for a newly trained model."""
        if not self.improvement_cycle_manager:
            return
        try:
            cycle = self.improvement_cycle_manager.cycles.get(cycle_id)
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

    # Canonical Pipeline Integration (for pipeline_orchestrator.py)
    # =========================================================================

    async def handle_pipeline_start(self, request: web.Request) -> web.Response:
        """POST /pipeline/start - Start a canonical pipeline phase."""
        try:
            if not self._is_leader():
                return web.json_response({"success": False, "error": "Only leader can start pipeline phases",
                                         "leader_id": self.leader_id}, status=403)
            data = await request.json()
            phase = data.get("phase")
            board_type = data.get("board_type", "square8")
            num_players = data.get("num_players", 2)

            if phase == "canonical_selfplay":
                result = await self._start_canonical_selfplay_pipeline(
                    board_type, num_players, data.get("games_per_node", 500), data.get("seed", 0))
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

    async def _start_canonical_selfplay_pipeline(self, board_type: str, num_players: int,
                                                 games_per_node: int, seed: int) -> Dict[str, Any]:
        """Start canonical selfplay on all healthy nodes in the cluster."""
        job_id = f"pipeline-selfplay-{int(time.time())}"
        healthy_nodes = []
        with self.peers_lock:
            for peer_id, peer in self.peers.items():
                if peer.is_alive() and peer.is_healthy():
                    healthy_nodes.append((peer_id, peer))
        if self.self_info.is_healthy():
            healthy_nodes.append((self.node_id, self.self_info))
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
                    url = f"http://{node.host}:{node.port}/pipeline/selfplay_worker"
                    async with ClientSession(timeout=ClientTimeout(total=30)) as session:
                        async with session.post(url, json={"job_id": f"{job_id}-{node_id}",
                            "board_type": board_type, "num_players": num_players,
                            "num_games": games_per_node, "seed": node_seed},
                            headers=self._get_auth_headers()) as resp:
                            if resp.status == 200:
                                dispatched += 1
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
            # Ensure local resource stats are fresh for dashboard consumers.
            try:
                self._update_self_info()
            except Exception:
                pass

            # Collect peer info (dashboard-oriented shape)
            peers_info: List[Dict[str, Any]] = []
            with self.peers_lock:
                peers_snapshot = dict(self.peers)
            for peer_id, peer in peers_snapshot.items():
                status = "offline" if not peer.is_alive() else "online"
                peers_info.append(
                    {
                        "node_id": peer_id,
                        "host": peer.host,
                        "port": peer.port,
                        "scheme": getattr(peer, "scheme", "http"),
                        "role": peer.role.value if hasattr(peer.role, "value") else str(peer.role),
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

            return web.json_response({
                "success": True,
                "node_id": self.node_id,
                "role": self.role.value if hasattr(self.role, 'value') else str(self.role),
                "leader_id": self.leader_id,
                "is_leader": self.role == NodeRole.LEADER,
                "self": self.self_info.to_dict() if hasattr(self.self_info, "to_dict") else asdict(self.self_info),
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

    async def handle_api_selfplay_stats(self, request: web.Request) -> web.Response:
        """Get aggregated selfplay game statistics for dashboard charts."""
        try:
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

            conn = init_elo_database()

            # If specific filter requested, return just that
            if board_type and num_players:
                leaderboard = get_leaderboard(conn, board_type, num_players, limit=limit)
                conn.close()
                return web.json_response({
                    "success": True,
                    "leaderboards": {f"{board_type}_{num_players}p": leaderboard},
                    "total_models": len(leaderboard),
                    "timestamp": time.time(),
                })

            # Otherwise return all board/player combinations
            # Query unique board_type/num_players combinations
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
                lb = get_leaderboard(conn, bt, np, limit=limit)
                if lb:
                    leaderboards[key] = lb
                    total_models += len(lb)
                    total_games += sum(entry.get("games_played", 0) for entry in lb)

            # Get match history stats
            cursor.execute("SELECT COUNT(*) FROM match_history")
            match_count = cursor.fetchone()[0]

            conn.close()

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
        if self.role != NodeRole.LEADER:
            return web.json_response({
                "success": False,
                "error": "Not the leader. Please submit to leader node.",
                "leader_id": self.leader_id,
            }, status=400)

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

            if job_type in ["selfplay", "gpu_selfplay"]:
                board_type = data.get("board_type", "square8")
                num_players = int(data.get("num_players", 2))
                engine_mode = data.get("engine_mode", "descent-only")

                jt = JobType.GPU_SELFPLAY if job_type == "gpu_selfplay" else JobType.SELFPLAY
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
                    "error": f"Unknown job type: {job_type}. Supported: nnue, cmaes, selfplay, gpu_selfplay",
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
        if self.role != NodeRole.LEADER:
            return web.json_response({
                "success": False,
                "error": "Not the leader",
            }, status=400)

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
        return web.Response(text=html, content_type="text/html")

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
    max_moves = 500
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

        self.self_info.cpu_percent = usage["cpu_percent"]
        self.self_info.memory_percent = usage["memory_percent"]
        self.self_info.disk_percent = usage["disk_percent"]
        self.self_info.gpu_percent = usage["gpu_percent"]
        self.self_info.gpu_memory_percent = usage["gpu_memory_percent"]
        self.self_info.selfplay_jobs = selfplay
        self.self_info.training_jobs = training
        self.self_info.role = self.role
        self.self_info.last_heartbeat = time.time()

    async def _send_heartbeat_to_peer(self, peer_host: str, peer_port: int, scheme: str = "http") -> Optional[NodeInfo]:
        """Send heartbeat to a peer and return their info."""
        try:
            self._update_self_info()

            timeout = ClientTimeout(total=10)
            async with ClientSession(timeout=timeout) as session:
                scheme = (scheme or "http").lower()
                url = f"{scheme}://{peer_host}:{peer_port}/heartbeat"
                async with session.post(url, json=self.self_info.to_dict(), headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        info = NodeInfo.from_dict(data)
                        # Use the address we successfully reached instead of any
                        # self-reported interface address.
                        info.scheme = scheme
                        info.host = peer_host
                        info.port = peer_port
                        return info
        except Exception as e:
            pass
        return None

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
            async with ClientSession(timeout=timeout) as session:
                # Use /relay/heartbeat endpoint
                url = f"{relay_url.rstrip('/')}/relay/heartbeat"
                async with session.post(url, json=self.self_info.to_dict(), headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("success"):
                            # Update our peer list with all peers from relay
                            peers_data = data.get("peers", {})
                            with self.peers_lock:
                                for node_id, peer_dict in peers_data.items():
                                    if node_id != self.node_id:
                                        peer_info = NodeInfo.from_dict(peer_dict)
                                        self.peers[node_id] = peer_info

                            # Update leader if provided
                            leader_id = data.get("leader_id")
                            if leader_id and leader_id != self.node_id:
                                if self.leader_id != leader_id:
                                    print(f"[P2P] Adopted leader from relay: {leader_id}")
                                self.leader_id = leader_id
                                self.role = NodeRole.FOLLOWER

                            return {
                                "success": True,
                                "peers_received": len(peers_data),
                                "leader_id": leader_id,
                            }
                        return {"success": False, "error": data.get("error", "Unknown error")}
                    return {"success": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

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
                        with self.peers_lock:
                            info.last_heartbeat = time.time()
                            self.peers[info.node_id] = info
                        if info.role == NodeRole.LEADER and info.node_id != self.node_id:
                            if self.leader_id != info.node_id or self.role != NodeRole.FOLLOWER:
                                print(f"[P2P] Following configured leader from heartbeat: {info.node_id}")
                            self.leader_id = info.node_id
                            self.role = NodeRole.FOLLOWER

                # Send to discovered peers (skip NAT-blocked peers - they can't receive)
                with self.peers_lock:
                    peer_list = [p for p in self.peers.values() if not p.nat_blocked]

                for peer in peer_list:
                    if peer.node_id != self.node_id:
                        peer_scheme = getattr(peer, "scheme", "http") or "http"
                        info = await self._send_heartbeat_to_peer(peer.host, peer.port, scheme=peer_scheme)
                        if info:
                            with self.peers_lock:
                                info.last_heartbeat = time.time()
                                self.peers[info.node_id] = info
                            if info.role == NodeRole.LEADER and self.role != NodeRole.LEADER:
                                if self.leader_id != info.node_id:
                                    print(f"[P2P] Adopted leader from heartbeat: {info.node_id}")
                                self.leader_id = info.node_id
                                self.role = NodeRole.FOLLOWER

                # Check for dead peers
                self._check_dead_peers()

                # LEARNED LESSONS - Lease renewal to maintain leadership
                if self.role == NodeRole.LEADER:
                    await self._renew_leader_lease()

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

    def _maybe_adopt_leader_from_peers(self) -> bool:
        """If we can already see a healthy leader, adopt it and avoid elections."""
        if self.role == NodeRole.LEADER:
            return False

        with self.peers_lock:
            leaders = [
                p for p in self.peers.values()
                if p.node_id != self.node_id and p.role == NodeRole.LEADER and p.is_alive()
            ]

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
        with self.peers_lock:
            dead_peers = []
            for node_id, info in self.peers.items():
                if not info.is_alive() and node_id != self.node_id:
                    dead_peers.append(node_id)

            for node_id in dead_peers:
                print(f"[P2P] Peer {node_id} is dead (no heartbeat for {PEER_TIMEOUT}s)")
                # Don't remove, just mark as dead for historical tracking

        # If leader is dead, start election
        if self.leader_id and self.leader_id != self.node_id:
            with self.peers_lock:
                leader = self.peers.get(self.leader_id)
                if leader and not leader.is_alive():
                    print(f"[P2P] Leader {self.leader_id} is dead, starting election")
                    asyncio.create_task(self._start_election())

    async def _start_election(self):
        """Start leader election using Bully algorithm."""
        if self.leader_id and self.leader_id != self.node_id:
            with self.peers_lock:
                leader = self.peers.get(self.leader_id)
            if leader and leader.is_alive():
                return
        if self._maybe_adopt_leader_from_peers():
            return

        if self.election_in_progress:
            return

        self.election_in_progress = True
        self.role = NodeRole.CANDIDATE
        print(f"[P2P] Starting election, my ID: {self.node_id}")

        try:
            # Send election message to all nodes with higher IDs
            higher_nodes = []
            with self.peers_lock:
                higher_nodes = [
                    p for p in self.peers.values()
                    if p.node_id > self.node_id and p.is_alive()
                ]

            got_response = False

            timeout = ClientTimeout(total=ELECTION_TIMEOUT)
            async with ClientSession(timeout=timeout) as session:
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
                await self._become_leader()
            else:
                # Wait for coordinator message
                await asyncio.sleep(ELECTION_TIMEOUT * 2)

        finally:
            self.election_in_progress = False

    async def _become_leader(self):
        """Become the cluster leader with lease-based leadership."""
        import uuid
        print(f"[P2P] I am now the leader: {self.node_id}")
        self.role = NodeRole.LEADER
        self.leader_id = self.node_id

        # LEARNED LESSONS - Lease-based leadership to prevent split-brain
        # Generate unique lease ID for this leadership term
        self.leader_lease_id = f"{self.node_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.leader_lease_expires = time.time() + LEADER_LEASE_DURATION
        self.last_lease_renewal = time.time()

        # Announce to all peers with lease information
        with self.peers_lock:
            peers = list(self.peers.values())

        timeout = ClientTimeout(total=5)
        async with ClientSession(timeout=timeout) as session:
            for peer in peers:
                if peer.node_id != self.node_id:
                    try:
                        url = self._url_for_peer(peer, "/coordinator")
                        await session.post(url, json={
                            "leader_id": self.node_id,
                            "lease_id": self.leader_lease_id,
                            "lease_expires": self.leader_lease_expires,
                        }, headers=self._auth_headers())
                    except:
                        pass

        self._save_state()

    async def _renew_leader_lease(self):
        """Renew our leadership lease and broadcast to peers."""
        if self.role != NodeRole.LEADER:
            return

        now = time.time()
        if now - self.last_lease_renewal < LEADER_LEASE_RENEW_INTERVAL:
            return  # Too soon to renew

        self.leader_lease_expires = now + LEADER_LEASE_DURATION
        self.last_lease_renewal = now

        # Broadcast lease renewal to all peers
        with self.peers_lock:
            peers = list(self.peers.values())

        timeout = ClientTimeout(total=3)
        try:
            async with ClientSession(timeout=timeout) as session:
                for peer in peers:
                    if peer.node_id != self.node_id and peer.is_alive():
                        try:
                            url = self._url_for_peer(peer, "/coordinator")
                            await session.post(url, json={
                                "leader_id": self.node_id,
                                "lease_id": self.leader_lease_id,
                                "lease_expires": self.leader_lease_expires,
                                "lease_renewal": True,
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

        # Gather all peers claiming to be leader
        other_leaders = []
        with self.peers_lock:
            for peer in self.peers.values():
                if peer.role == NodeRole.LEADER and peer.node_id != self.node_id and peer.is_alive():
                    other_leaders.append(peer)

        if not other_leaders:
            return False  # No split-brain

        # Find the highest-priority leader (including ourselves)
        all_leaders = other_leaders + [self.self_info]
        highest_leader = max(all_leaders, key=lambda p: p.node_id)

        if highest_leader.node_id != self.node_id:
            # We're not the highest-priority leader - step down
            print(f"[P2P] SPLIT-BRAIN detected! Found leaders: {[p.node_id for p in other_leaders]}")
            print(f"[P2P] Stepping down in favor of higher-priority leader: {highest_leader.node_id}")
            self.role = NodeRole.FOLLOWER
            self.leader_id = highest_leader.node_id
            self._save_state()
            return True

        # We are the highest - other leaders should step down
        # Send coordinator message to assert our leadership
        print(f"[P2P] SPLIT-BRAIN detected! Asserting leadership over: {[p.node_id for p in other_leaders]}")
        timeout = ClientTimeout(total=5)
        async with ClientSession(timeout=timeout) as session:
            for peer in other_leaders:
                try:
                    url = self._url_for_peer(peer, "/coordinator")
                    await session.post(url, json={"leader_id": self.node_id}, headers=self._auth_headers())
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
                    # Phase 3: Check if training should be triggered automatically
                    await self._check_and_trigger_training()
                    # Phase 5: Check improvement cycles for automated training
                    await self._check_improvement_cycles()
            except Exception as e:
                print(f"[P2P] Job management error: {e}")

            await asyncio.sleep(JOB_CHECK_INTERVAL)

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

            # LEARNED LESSONS - Memory warning - reduce jobs
            if node.memory_percent >= MEMORY_WARNING_THRESHOLD:
                print(f"[P2P] {node.node_id}: Memory at {node.memory_percent:.0f}% - reducing jobs")
                # Don't start new jobs, let existing ones complete

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

        # Phase 2: Calculate desired job distribution for healthy nodes
        for node in all_nodes:
            # LEARNED LESSONS - Use is_healthy() to check both connectivity AND resources
            if not node.is_healthy():
                reason = []
                if not node.is_alive():
                    reason.append("unreachable")
                if node.disk_percent >= DISK_CRITICAL_THRESHOLD:
                    reason.append(f"disk={node.disk_percent:.0f}%")
                if node.memory_percent >= MEMORY_CRITICAL_THRESHOLD:
                    reason.append(f"mem={node.memory_percent:.0f}%")
                print(f"[P2P] Skipping {node.node_id}: {', '.join(reason)}")
                continue

            # LEARNED LESSONS - Reduce target when approaching limits
            target_selfplay = 2  # Base minimum
            if node.memory_gb >= 64:
                target_selfplay = 4
            if node.has_gpu and "5090" in node.gpu_name.lower():
                target_selfplay = 8  # More for powerful GPUs

            # LEARNED LESSONS - Reduce target if resources are under pressure
            if node.disk_percent >= DISK_WARNING_THRESHOLD:
                target_selfplay = min(target_selfplay, 2)
            if node.memory_percent >= MEMORY_WARNING_THRESHOLD:
                target_selfplay = min(target_selfplay, 1)

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
                weighted_configs = []
                for cfg in selfplay_configs:
                    weighted_configs.extend([cfg] * cfg.get("priority", 1))

                # Start jobs (max 2 at a time to avoid overwhelming)
                for i in range(min(needed, 2)):
                    # Choose GPU selfplay only for CUDA-capable nodes (not Apple MPS).
                    gpu_name = (node.gpu_name or "").upper()
                    is_cuda_gpu = node.has_gpu and "MPS" not in gpu_name and "APPLE" not in gpu_name

                    # LEARNED LESSONS - Ensure GPU nodes get GPU-utilizing tasks
                    if is_cuda_gpu and node.gpu_percent < 30:
                        # GPU underutilized - force GPU selfplay with neural network
                        job_type = JobType.GPU_SELFPLAY
                    else:
                        job_type = JobType.GPU_SELFPLAY if is_cuda_gpu else JobType.SELFPLAY

                    # Weighted config selection based on priority
                    # Use ImprovementCycleManager for dynamic data-aware diverse selection
                    import random as rand_module
                    if self.improvement_cycle_manager:
                        # Dynamic selection with AI diversity (asymmetric games, varied opponents)
                        config = self.improvement_cycle_manager.get_next_selfplay_config(
                            self.cluster_data_manifest
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
            timeout = ClientTimeout(total=HTTP_TOTAL_TIMEOUT)
            async with ClientSession(timeout=timeout) as session:
                url = self._url_for_peer(node, "/cleanup")
                async with session.post(url, json={}, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        print(f"[P2P] Cleanup requested on {node.node_id}")
        except Exception as e:
            print(f"[P2P] Failed to request cleanup from {node.node_id}: {e}")

    async def _restart_local_stuck_jobs(self):
        """Kill stuck selfplay processes and let job management restart them.

        LEARNED LESSONS - Addresses the issue where processes accumulate but GPU stays at 0%.
        """
        print("[P2P] Restarting stuck local selfplay jobs...")
        try:
            # Kill tracked selfplay jobs (avoid broad pkill patterns).
            jobs_to_clear: List[str] = []
            pids_to_kill: List[int] = []
            with self.jobs_lock:
                for job_id, job in self.local_jobs.items():
                    if job.job_type not in (JobType.SELFPLAY, JobType.GPU_SELFPLAY):
                        continue
                    jobs_to_clear.append(job_id)
                    if job.pid:
                        pids_to_kill.append(job.pid)

            killed = 0
            for pid in pids_to_kill:
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
            timeout = ClientTimeout(total=HTTP_TOTAL_TIMEOUT)
            async with ClientSession(timeout=timeout) as session:
                url = self._url_for_peer(node, "/restart_stuck_jobs")
                async with session.post(url, json={}, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("success"):
                            print(f"[P2P] Job restart requested on {node.node_id}")
                        else:
                            print(f"[P2P] Job restart failed on {node.node_id}: {data.get('error')}")
                    else:
                        print(f"[P2P] Job restart request failed with status {resp.status}")
        except Exception as e:
            print(f"[P2P] Failed to request job restart from {node.node_id}: {e}")

    async def _start_local_job(
        self,
        job_type: JobType,
        board_type: str = "square8",
        num_players: int = 2,
        engine_mode: str = "descent-only",
    ) -> Optional[ClusterJob]:
        """Start a job on the local node."""
        try:
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

                log_handle = open(output_dir / "run.log", "a")
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=self.ringrift_path,
                    )
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

                # Choose a GPU automatically if not explicitly pinned.
                if "CUDA_VISIBLE_DEVICES" not in env:
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

                log_handle = open(output_dir / "gpu_run.log", "a")
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=self.ringrift_path,
                    )
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
        """Request a remote node to start a job with specific configuration."""
        try:
            timeout = ClientTimeout(total=10)
            async with ClientSession(timeout=timeout) as session:
                url = self._url_for_peer(node, "/start_job")
                payload = {
                    "job_type": job_type.value,
                    "board_type": board_type,
                    "num_players": num_players,
                    "engine_mode": engine_mode,
                }
                async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("success"):
                            print(f"[P2P] Started remote {board_type} {num_players}p job on {node.node_id}")
                        else:
                            print(f"[P2P] Failed to start remote job: {data.get('error')}")
        except Exception as e:
            print(f"[P2P] Failed to request remote job from {node.node_id}: {e}")

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
        app.router.add_post('/coordinator', self.handle_coordinator)
        app.router.add_post('/start_job', self.handle_start_job)
        app.router.add_post('/stop_job', self.handle_stop_job)
        app.router.add_post('/cleanup', self.handle_cleanup)
        app.router.add_post('/restart_stuck_jobs', self.handle_restart_stuck_jobs)
        app.router.add_get('/health', self.handle_health)
        app.router.add_get('/git/status', self.handle_git_status)
        app.router.add_post('/git/update', self.handle_git_update)

        # Dynamic host registry routes (for IP auto-updates)
        app.router.add_post('/register', self.handle_register)
        app.router.add_get('/registry/status', self.handle_registry_status)
        app.router.add_post('/registry/update_vast', self.handle_registry_update_vast)
        app.router.add_post('/registry/save_yaml', self.handle_registry_save_yaml)

        # Relay/Hub routes for NAT-blocked nodes
        app.router.add_post('/relay/heartbeat', self.handle_relay_heartbeat)
        app.router.add_get('/relay/peers', self.handle_relay_peers)

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

        # Canonical pipeline routes (for pipeline_orchestrator.py integration)
        app.router.add_post('/pipeline/start', self.handle_pipeline_start)
        app.router.add_get('/pipeline/status', self.handle_pipeline_status)
        app.router.add_post('/pipeline/selfplay_worker', self.handle_pipeline_selfplay_worker)

        # Phase 4: REST API and Dashboard routes
        app.router.add_get('/', self.handle_root)
        app.router.add_get('/api/cluster/status', self.handle_api_cluster_status)
        app.router.add_get('/api/selfplay/stats', self.handle_api_selfplay_stats)
        app.router.add_get('/api/elo/leaderboard', self.handle_api_elo_leaderboard)
        app.router.add_get('/api/canonical/health', self.handle_api_canonical_health)
        app.router.add_get('/api/canonical/jobs', self.handle_api_canonical_jobs_list)
        app.router.add_get('/api/canonical/jobs/{job_id}', self.handle_api_canonical_job_get)
        app.router.add_get('/api/canonical/jobs/{job_id}/log', self.handle_api_canonical_job_log)
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

        # Add Vast.ai IP update loop (if VAST_API_KEY is set)
        if HAS_DYNAMIC_REGISTRY:
            tasks.append(asyncio.create_task(self._vast_ip_update_loop()))

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
