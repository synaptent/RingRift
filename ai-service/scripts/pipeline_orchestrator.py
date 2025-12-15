#!/usr/bin/env python
"""Master pipeline orchestrator for distributed AI training.

This script coordinates the complete AI training pipeline across all available
compute resources. It manages:
- Distributed selfplay across AWS instances, Vast.ai, and local machines
- Canonical selfplay (CPU-based with full phase machine for parity)
- Data synchronization and aggregation
- Parity validation gate (ensures games pass TS/Python parity check)
- NPZ export (converts games to neural network training format)
- Neural network training (on GPUs or Mac Studio with MPS)
- CMA-ES heuristic optimization
- Model evaluation tournaments
- Elo rating tracking
- Tier gating for progressive difficulty

Available Phases:
    selfplay          - GPU/heuristic selfplay (fast but simplified rules)
    canonical-selfplay - CPU selfplay with full phase machine (canonical)
    sync              - Sync game data from all workers
    parity-validation - Validate games against canonical parity gate
    npz-export        - Export validated games to NPZ training format
    training          - Train neural network on exported data
    cmaes             - CMA-ES heuristic optimization
    profile-sync      - Sync trained heuristic profiles across workers
    evaluation        - Run evaluation tournaments
    elo-calibration   - Run diverse tournaments for Elo calibration (all configs)
    tier-gating       - Check for model tier promotions
    resources         - Log resource usage across workers
    refresh-workers   - Dynamically discover Vast.ai instances

Backend Modes:
    ssh (default)     - Direct SSH execution on workers
    p2p               - Use P2P orchestrator REST API (requires p2p_orchestrator running)

Usage:
    # Run a single iteration of the pipeline
    python scripts/pipeline_orchestrator.py --iterations 1

    # Run continuous improvement loop
    python scripts/pipeline_orchestrator.py --iterations 10

    # Run canonical selfplay with Vast instance discovery
    python scripts/pipeline_orchestrator.py --phase canonical-selfplay \\
        --discover-vast --games-per-worker 1000

    # Run using P2P backend (requires p2p_orchestrator daemons on nodes)
    python scripts/pipeline_orchestrator.py --backend p2p \\
        --p2p-leader http://192.168.1.100:8770 \\
        --phase canonical-selfplay --games-per-worker 1000

    # Run parity validation on generated games
    python scripts/pipeline_orchestrator.py --phase parity-validation

    # Export validated games to NPZ format
    python scripts/pipeline_orchestrator.py --phase npz-export

    # Run specific phase only
    python scripts/pipeline_orchestrator.py --phase training
    python scripts/pipeline_orchestrator.py --phase evaluation

    # Dry run (show what would be executed)
    python scripts/pipeline_orchestrator.py --dry-run

    # Resume from last checkpoint
    python scripts/pipeline_orchestrator.py --resume
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# SSH retry configuration
SSH_MAX_RETRIES = 3
SSH_BASE_DELAY = 2.0  # seconds
SSH_MAX_DELAY = 30.0  # seconds
SSH_BACKOFF_FACTOR = 2.0

# Smart polling configuration - reduced from 60s to 10s for faster response
POLL_INTERVAL_SECONDS = 10  # Check every 10 seconds (reduced from 60s)
POLL_INTERVAL_SLOW = 30  # Slower polling for less urgent phases
MAX_PHASE_WAIT_MINUTES = 120  # Maximum wait for any phase
SELFPLAY_MIN_GAMES_THRESHOLD = 50  # Min games before proceeding
CMAES_COMPLETION_CHECK_CMD = "pgrep -f 'run_iterative_cmaes' >/dev/null && echo running || echo done"

# P2P orchestrator integration
P2P_DEFAULT_PORT = 8770
P2P_HTTP_TIMEOUT = 30  # seconds
P2P_JOB_POLL_INTERVAL = 10  # seconds (reduced from 30s)

# Try to import aiohttp for P2P backend
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Persistent Elo database integration (shared with continuous_improvement_daemon)
try:
    from scripts.run_model_elo_tournament import (
        init_elo_database,
        register_models,
        update_elo_after_match,
        get_leaderboard as get_persistent_leaderboard,
        ELO_DB_PATH,
    )
    HAS_PERSISTENT_ELO = True
except ImportError:
    HAS_PERSISTENT_ELO = False
    ELO_DB_PATH = None

# Diverse tournament orchestrator integration (for Elo calibration across all configs)
try:
    from scripts.run_diverse_tournaments import (
        run_tournament_round_distributed,
        run_tournament_round_local,
        load_cluster_hosts,
        filter_available_hosts,
        TournamentConfig,
        TournamentResult,
        DEFAULT_GAMES_PER_CONFIG,
    )
    HAS_DIVERSE_TOURNAMENTS = True
except ImportError:
    HAS_DIVERSE_TOURNAMENTS = False


@dataclass
class WorkerConfig:
    """Configuration for a compute worker."""

    name: str
    host: str  # user@hostname format
    role: str  # "selfplay", "training", "cmaes", "mixed"
    capabilities: List[str]
    cpus: int = 0
    memory_gb: int = 0
    gpu: str = ""
    ssh_key: Optional[str] = None
    ssh_port: int = 22  # Non-standard port for Vast.ai etc
    remote_path: str = "~/ringrift/ai-service"
    max_parallel_jobs: int = 1

    def is_gpu_worker(self) -> bool:
        """Whether this worker should be treated as GPU/accelerator-capable."""
        role = (self.role or "").lower()
        if self.gpu:
            return True
        return "nn_training" in role or role.endswith("_mps") or role.endswith("_gpu")

    def is_cpu_worker(self) -> bool:
        """Whether this worker should be treated as CPU-only."""
        return not self.is_gpu_worker()


@dataclass
class SelfplayJob:
    """Configuration for a selfplay job."""

    board_type: str
    num_players: int
    num_games: int
    engine_mode: str = "mixed"  # Use mixed for diverse training data
    max_moves: int = 10000
    seed: int = 0
    use_trained_profiles: bool = True  # Load CMA-ES optimized heuristics
    use_neural_net: bool = False  # Enable NN for descent/mcts


@dataclass
class P2PNodeInfo:
    """Information about a node from P2P cluster."""
    node_id: str
    host: str
    port: int
    role: str
    has_gpu: bool
    gpu_name: str
    memory_gb: int
    capabilities: List[str]
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    is_alive: bool = True
    is_healthy: bool = True

    def to_worker_config(self) -> WorkerConfig:
        """Convert to WorkerConfig for compatibility."""
        return WorkerConfig(
            name=self.node_id,
            host=self.host,
            role="mixed" if self.has_gpu else "selfplay",
            capabilities=self.capabilities,
            ssh_port=22,
            remote_path="~/ringrift/ai-service",
            max_parallel_jobs=4 if self.has_gpu else 2,
        )


class P2PBackend:
    """Backend for communicating with P2P orchestrator cluster.

    Provides interface to P2P orchestrator REST API for job dispatch without SSH.
    """

    def __init__(self, leader_url: str, auth_token: Optional[str] = None, timeout: float = P2P_HTTP_TIMEOUT):
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for P2P backend: pip install aiohttp")
        self.leader_url = leader_url.rstrip("/")
        self.auth_token = auth_token or os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN", "")
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_cluster_status(self) -> Dict[str, Any]:
        session = await self._get_session()
        async with session.get(f"{self.leader_url}/api/cluster/status") as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to get cluster status: {resp.status}")
            return await resp.json()

    async def get_nodes(self) -> List[P2PNodeInfo]:
        status = await self.get_cluster_status()
        nodes = []
        for n in status.get("nodes", []):
            nodes.append(P2PNodeInfo(
                node_id=n.get("node_id", ""), host=n.get("host", ""), port=n.get("port", P2P_DEFAULT_PORT),
                role=n.get("role", "follower"), has_gpu=n.get("has_gpu", False), gpu_name=n.get("gpu_name", ""),
                memory_gb=n.get("memory_gb", 0), capabilities=n.get("capabilities", []),
                cpu_percent=n.get("cpu_percent", 0), memory_percent=n.get("memory_percent", 0),
                disk_percent=n.get("disk_percent", 0), selfplay_jobs=n.get("selfplay_jobs", 0),
                training_jobs=n.get("training_jobs", 0), is_alive=n.get("is_alive", True),
                is_healthy=n.get("is_healthy", True),
            ))
        return nodes

    async def get_healthy_nodes(self) -> List[P2PNodeInfo]:
        return [n for n in await self.get_nodes() if n.is_alive and n.is_healthy]

    async def start_canonical_selfplay(self, board_type: str = "square8", num_players: int = 2,
                                       games_per_node: int = 500, seed: int = 0) -> Dict[str, Any]:
        session = await self._get_session()
        payload = {"phase": "canonical_selfplay", "board_type": board_type, "num_players": num_players,
                   "games_per_node": games_per_node, "seed": seed}
        async with session.post(f"{self.leader_url}/pipeline/start", json=payload) as resp:
            result = await resp.json()
            if not result.get("success"):
                raise RuntimeError(f"Failed to start canonical selfplay: {result.get('error')}")
            return result

    async def start_parity_validation(self, board_type: str = "square8", num_players: int = 2,
                                      db_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        session = await self._get_session()
        payload = {"phase": "parity_validation", "board_type": board_type, "num_players": num_players}
        if db_paths:
            payload["db_paths"] = db_paths
        async with session.post(f"{self.leader_url}/pipeline/start", json=payload) as resp:
            result = await resp.json()
            if not result.get("success"):
                raise RuntimeError(f"Failed to start parity validation: {result.get('error')}")
            return result

    async def start_npz_export(self, board_type: str = "square8", num_players: int = 2,
                               output_dir: str = "data/training") -> Dict[str, Any]:
        session = await self._get_session()
        payload = {"phase": "npz_export", "board_type": board_type, "num_players": num_players,
                   "output_dir": output_dir}
        async with session.post(f"{self.leader_url}/pipeline/start", json=payload) as resp:
            result = await resp.json()
            if not result.get("success"):
                raise RuntimeError(f"Failed to start NPZ export: {result.get('error')}")
            return result

    async def get_pipeline_status(self) -> Dict[str, Any]:
        session = await self._get_session()
        async with session.get(f"{self.leader_url}/pipeline/status") as resp:
            return await resp.json()

    async def wait_for_pipeline_completion(self, job_id: str, poll_interval: float = P2P_JOB_POLL_INTERVAL,
                                           timeout_minutes: float = MAX_PHASE_WAIT_MINUTES) -> Dict[str, Any]:
        start_time = time.time()
        while time.time() - start_time < timeout_minutes * 60:
            status = await self.get_pipeline_status()
            current = status.get("current_job", {})
            if current.get("job_id") == job_id and current.get("status") in ("completed", "failed"):
                return status
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"Pipeline job {job_id} did not complete within {timeout_minutes} minutes")

    async def trigger_data_sync(self) -> Dict[str, Any]:
        session = await self._get_session()
        async with session.post(f"{self.leader_url}/sync/start") as resp:
            return await resp.json()

    async def trigger_git_update(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        session = await self._get_session()
        payload = {"node_id": node_id} if node_id else {}
        async with session.post(f"{self.leader_url}/git/update", json=payload) as resp:
            return await resp.json()


def _normalize_p2p_seed_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


async def discover_p2p_leader_url(
    seed_urls: List[str],
    *,
    auth_token: str = "",
    timeout_seconds: float = 5.0,
) -> Optional[str]:
    """Discover the current effective P2P leader URL from one or more seed nodes.

    This keeps orchestration scripts resilient to leader churn: any reachable
    seed node can be used to locate the current leader without hard-coding a
    specific instance.
    """
    if not HAS_AIOHTTP:
        raise ImportError("aiohttp required for P2P leader discovery: pip install aiohttp")

    seeds = [_normalize_p2p_seed_url(s) for s in (seed_urls or [])]
    seeds = [s for s in seeds if s]
    if not seeds:
        return None

    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))

    def _is_loopback(host: str) -> bool:
        host = (host or "").strip().lower()
        return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

    def _is_tailscale_ip(host: str) -> bool:
        host = (host or "").strip()
        if not host:
            return False
        try:
            import ipaddress

            ip = ipaddress.ip_address(host)
            if ip.version != 4:
                return False
            return ip in ipaddress.ip_network("100.64.0.0/10")
        except Exception:
            return False

    def _candidate_base_urls(info: Dict[str, Any]) -> List[str]:
        scheme = str(info.get("scheme") or "http").strip() or "http"
        host = str(info.get("host") or "").strip()
        rh = str(info.get("reported_host") or "").strip()
        try:
            port = int(info.get("port", None))
        except Exception:
            port = None
        try:
            rp = int(info.get("reported_port", None))
        except Exception:
            rp = None

        candidates: List[str] = []

        def _add(h: str, p: Optional[int]) -> None:
            h = (h or "").strip()
            if not h or _is_loopback(h):
                return
            if not p or p <= 0:
                return
            base = f"{scheme}://{h}:{p}"
            if base not in candidates:
                candidates.append(base)

        # Prefer mesh addresses first when present (best NAT traversal).
        if rh and rp and _is_tailscale_ip(rh):
            _add(rh, rp)
        _add(host, port)
        if rh and rp and (rh != host or rp != port):
            _add(rh, rp)

        # If the only discovered address is loopback/empty, leave it to the caller.
        return candidates

    async def _first_reachable_base(candidates: List[str]) -> Optional[str]:
        for base in candidates:
            try:
                async with session.get(f"{base}/health") as resp:
                    if resp.status == 200:
                        return base
            except Exception:
                continue
        return None

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        for seed in seeds:
            try:
                async with session.get(f"{seed}/status") as resp:
                    if resp.status != 200:
                        continue
                    status = await resp.json()
            except Exception:
                continue

            if not isinstance(status, dict):
                continue

            leader_id = (status.get("effective_leader_id") or status.get("leader_id") or "").strip()
            if not leader_id:
                continue

            self_block = status.get("self") if isinstance(status.get("self"), dict) else {}
            peers_block = status.get("peers") if isinstance(status.get("peers"), dict) else {}

            leader_info: Dict[str, Any] = {}
            node_id = (status.get("node_id") or "").strip()
            if leader_id == node_id or leader_id == (self_block.get("node_id") or "").strip():
                leader_info = self_block
            else:
                leader_info = peers_block.get(leader_id) if isinstance(peers_block.get(leader_id), dict) else {}

            host = (leader_info.get("host") or "").strip()
            scheme = (leader_info.get("scheme") or "http").strip() or "http"
            try:
                port_i = int(leader_info.get("port", None))
            except Exception:
                port_i = None

            candidates = _candidate_base_urls(leader_info)
            reachable = await _first_reachable_base(candidates)
            if reachable:
                return reachable

            if host and port_i and not _is_loopback(host):
                # Back-compat fallback (may be unreachable in NAT/proxy setups).
                return f"{scheme}://{host}:{port_i}"

            # Fallback: use the seed itself if it claims to be leader.
            if leader_id == node_id:
                return seed

    return None


@dataclass
class PipelineState:
    """Current state of the pipeline with full checkpointing support."""

    iteration: int = 0
    phase: str = "idle"
    phase_completed: Dict[str, bool] = field(default_factory=dict)  # Tracks which phases completed
    games_generated: Dict[str, int] = field(default_factory=dict)
    models_trained: List[str] = field(default_factory=list)
    last_sync: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    # Elo tracking
    elo_ratings: Dict[str, float] = field(default_factory=dict)  # model_id -> Elo rating
    elo_history: List[Dict[str, Any]] = field(default_factory=list)  # List of Elo updates
    # Model registry
    model_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # model_id -> metadata
    # Game deduplication
    seen_game_hashes: Set[str] = field(default_factory=set)
    # Tier gating
    tier_promotions: Dict[str, str] = field(default_factory=dict)  # config -> current tier


# =============================================================================
# Event-Driven Pipeline Infrastructure
# =============================================================================
# Completion callbacks allow immediate triggering of downstream stages instead
# of relying on polling. This reduces iteration time by eliminating idle gaps.

from enum import Enum
from typing import Callable, Awaitable

class StageEvent(Enum):
    """Events emitted when pipeline stages complete."""
    SELFPLAY_COMPLETE = "selfplay_complete"
    CANONICAL_SELFPLAY_COMPLETE = "canonical_selfplay_complete"
    SYNC_COMPLETE = "sync_complete"
    PARITY_VALIDATION_COMPLETE = "parity_validation_complete"
    NPZ_EXPORT_COMPLETE = "npz_export_complete"
    TRAINING_COMPLETE = "training_complete"
    EVALUATION_COMPLETE = "evaluation_complete"
    CMAES_COMPLETE = "cmaes_complete"
    PROMOTION_COMPLETE = "promotion_complete"


@dataclass
class StageCompletionResult:
    """Data passed to completion callbacks."""
    event: StageEvent
    success: bool
    iteration: int
    timestamp: str
    board_type: str = "square8"
    num_players: int = 2
    games_generated: int = 0
    model_path: Optional[str] = None
    win_rate: Optional[float] = None
    promoted: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Type alias for completion callbacks
StageCompletionCallback = Callable[[StageCompletionResult], Awaitable[None]]


class StageEventBus:
    """Event bus for pipeline stage completion notifications.

    Enables event-driven pipeline execution by allowing stages to register
    callbacks that fire immediately when upstream stages complete.

    Example:
        bus = StageEventBus()

        async def on_selfplay_done(result):
            if result.success:
                await start_data_sync()

        bus.subscribe(StageEvent.SELFPLAY_COMPLETE, on_selfplay_done)

        # Later, when selfplay completes:
        await bus.emit(StageCompletionResult(
            event=StageEvent.SELFPLAY_COMPLETE,
            success=True,
            iteration=1,
            timestamp=datetime.now().isoformat(),
            games_generated=500
        ))
    """

    def __init__(self):
        self._subscribers: Dict[StageEvent, List[StageCompletionCallback]] = {}
        self._log_callback: Optional[Callable[[str], None]] = None

    def set_logger(self, log_fn: Callable[[str], None]) -> None:
        """Set a logging function for event notifications."""
        self._log_callback = log_fn

    def subscribe(self, event: StageEvent, callback: StageCompletionCallback) -> None:
        """Register a callback for a stage completion event."""
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(callback)

    def unsubscribe(self, event: StageEvent, callback: StageCompletionCallback) -> bool:
        """Remove a callback from an event. Returns True if found and removed."""
        if event in self._subscribers and callback in self._subscribers[event]:
            self._subscribers[event].remove(callback)
            return True
        return False

    async def emit(self, result: StageCompletionResult) -> int:
        """Emit a stage completion event to all subscribers.

        Returns the number of callbacks invoked.
        """
        callbacks = self._subscribers.get(result.event, [])
        if self._log_callback:
            status = "OK" if result.success else "FAILED"
            self._log_callback(
                f"[EVENT] {result.event.value} ({status}) - "
                f"invoking {len(callbacks)} callback(s)"
            )

        invoked = 0
        for callback in callbacks:
            try:
                await callback(result)
                invoked += 1
            except Exception as e:
                if self._log_callback:
                    self._log_callback(f"[EVENT] Callback error for {result.event.value}: {e}")

        return invoked

    def subscriber_count(self, event: StageEvent) -> int:
        """Get the number of subscribers for an event."""
        return len(self._subscribers.get(event, []))


# Global event bus singleton for cross-module event handling
_global_event_bus: Optional[StageEventBus] = None


def get_event_bus() -> StageEventBus:
    """Get or create the global event bus."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = StageEventBus()
    return _global_event_bus


# =============================================================================
# Convenience callbacks for common pipeline transitions
# =============================================================================

async def on_selfplay_complete(result: StageCompletionResult) -> None:
    """Default handler for selfplay completion - triggers data sync.

    This is a template showing the event-driven pattern. Real implementations
    should register their own callbacks with specific logic.
    """
    if result.success and result.games_generated > 0:
        # Emit signal that data is ready for sync
        # The actual sync should be triggered by the orchestrator
        pass


async def on_training_complete(result: StageCompletionResult) -> None:
    """Default handler for training completion - triggers evaluation.

    This is a template showing the event-driven pattern. Real implementations
    should register their own callbacks with specific logic.
    """
    if result.success and result.model_path:
        # Emit signal that model is ready for evaluation
        # The actual evaluation should be triggered by the orchestrator
        pass


# Worker configurations loaded from gitignored config file
# See config/distributed_hosts.yaml for actual host configuration
def load_workers_from_config() -> List[WorkerConfig]:
    """Load worker configurations from distributed_hosts.yaml."""
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        print(f"Warning: {config_path} not found. Using empty worker list.")
        print("Copy config/distributed_hosts.example.yaml to config/distributed_hosts.yaml")
        return []

    with open(config_path) as f:
        config = yaml.safe_load(f)

    workers = []
    for name, host_config in config.get("hosts", {}).items():
        if host_config.get("status") != "ready":
            continue
        workers.append(WorkerConfig(
            name=name,
            host=f"{host_config.get('ssh_user', 'ubuntu')}@{host_config.get('ssh_host', '')}",
            role=host_config.get("role", "selfplay"),
            capabilities=host_config.get("capabilities", ["square8"]),
            cpus=int(host_config.get("cpus", 0) or 0),
            memory_gb=int(host_config.get("memory_gb", 0) or 0),
            gpu=str(host_config.get("gpu", "") or ""),
            ssh_key=host_config.get("ssh_key"),
            ssh_port=host_config.get("ssh_port", 22),
            remote_path=host_config.get("ringrift_path", "~/ringrift/ai-service"),
            max_parallel_jobs=host_config.get("max_parallel_jobs", 2),
        ))
    return workers

WORKERS = load_workers_from_config()

# Default selfplay configuration per iteration
# Each board×player gets multiple engine modes for diverse training data:
# - mixed: Samples from canonical ladder (random, heuristic, minimax, mcts, descent)
# - heuristic-only: Pure heuristic games (benefits from CMA-ES trained weights)
# - descent-only: Pure descent games (benefits from NN when available)
# - nn-only: Neural-enabled descent/mcts (when NN checkpoint exists)
#
# The mix ensures:
# 1. Heuristics are exercised (so CMA-ES improvements get tested)
# 2. Search algorithms get trained heuristics for leaf evaluation
# 3. NN gets diverse training data from all skill levels
# 4. Self-play covers the full strength spectrum

DEFAULT_SELFPLAY_CONFIG = {
    # Square8 2p - Primary training config, most games
    "square8_2p_mixed": SelfplayJob("square8", 2, 40, "mixed", 500),
    "square8_2p_heuristic": SelfplayJob("square8", 2, 20, "heuristic-only", 500),
    "square8_2p_minimax": SelfplayJob("square8", 2, 15, "minimax-only", 500),
    "square8_2p_mcts": SelfplayJob("square8", 2, 15, "mcts-only", 500),
    "square8_2p_descent": SelfplayJob("square8", 2, 20, "descent-only", 500),
    "square8_2p_nn": SelfplayJob("square8", 2, 10, "nn-only", 500, use_neural_net=True),

    # Square8 3p
    "square8_3p_mixed": SelfplayJob("square8", 3, 25, "mixed", 600),
    "square8_3p_heuristic": SelfplayJob("square8", 3, 15, "heuristic-only", 600),
    "square8_3p_descent": SelfplayJob("square8", 3, 15, "descent-only", 600),

    # Square8 4p
    "square8_4p_mixed": SelfplayJob("square8", 4, 20, "mixed", 700),
    "square8_4p_heuristic": SelfplayJob("square8", 4, 10, "heuristic-only", 700),
    "square8_4p_descent": SelfplayJob("square8", 4, 10, "descent-only", 700),

    # Square19 2p - Larger board, fewer games
    "square19_2p_mixed": SelfplayJob("square19", 2, 20, "mixed", 800),
    "square19_2p_heuristic": SelfplayJob("square19", 2, 15, "heuristic-only", 800),
    "square19_2p_descent": SelfplayJob("square19", 2, 15, "descent-only", 800),

    # Square19 3p/4p
    "square19_3p_mixed": SelfplayJob("square19", 3, 15, "mixed", 1000),
    "square19_4p_mixed": SelfplayJob("square19", 4, 10, "mixed", 1200),

    # Hexagonal - Similar to square19
    "hex_2p_mixed": SelfplayJob("hexagonal", 2, 20, "mixed", 800),
    "hex_2p_heuristic": SelfplayJob("hexagonal", 2, 15, "heuristic-only", 800),
    "hex_2p_descent": SelfplayJob("hexagonal", 2, 15, "descent-only", 800),

    # Hex 3p/4p
    "hex_3p_mixed": SelfplayJob("hexagonal", 3, 15, "mixed", 1000),
    "hex_4p_mixed": SelfplayJob("hexagonal", 4, 10, "mixed", 1200),
}


class PipelineOrchestrator:
    """Master orchestrator for the AI training pipeline.

    Supports two backends:
    - SSH (default): Direct SSH execution on workers
    - P2P: Use P2P orchestrator REST API (requires p2p_orchestrator on nodes)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        state_path: Optional[str] = None,
        dry_run: bool = False,
        backend: str = "ssh",
        p2p_leader_url: Optional[str] = None,
        p2p_auth_token: Optional[str] = None,
        event_bus: Optional[StageEventBus] = None,
    ):
        self.config = self._load_config(config_path) if config_path else {}
        self.state_path = state_path or "logs/pipeline/state.json"
        self.state = self._load_state()
        self.dry_run = dry_run
        self.log_dir = Path("logs/pipeline")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Event-driven pipeline infrastructure
        self.event_bus = event_bus or get_event_bus()
        self.event_bus.set_logger(self.log)

        # Backend configuration
        self.backend_mode = backend
        self.p2p_backend: Optional[P2PBackend] = None
        if backend == "p2p":
            if not p2p_leader_url:
                raise ValueError("P2P backend requires --p2p-leader URL")
            self.p2p_backend = P2PBackend(leader_url=p2p_leader_url, auth_token=p2p_auth_token)
            self.log(f"Using P2P backend with leader: {p2p_leader_url}")
        else:
            self.log("Using SSH backend")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load pipeline configuration from JSON."""
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r") as f:
            return json.load(f)

    def _load_state(self) -> PipelineState:
        """Load pipeline state from disk with proper type handling."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                # Convert seen_game_hashes from list to set
                if "seen_game_hashes" in data:
                    data["seen_game_hashes"] = set(data["seen_game_hashes"])
                return PipelineState(**data)
            except (json.JSONDecodeError, TypeError) as e:
                self.log(f"Warning: Could not load state file: {e}", "WARN")
        return PipelineState()

    def _save_state(self) -> None:
        """Save pipeline state to disk with proper serialization."""
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        # Convert state to dict with proper serialization
        state_dict = self.state.__dict__.copy()
        # Convert set to list for JSON serialization
        if "seen_game_hashes" in state_dict:
            state_dict["seen_game_hashes"] = list(state_dict["seen_game_hashes"])
        with open(self.state_path, "w") as f:
            json.dump(state_dict, f, indent=2, default=str)

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {"INFO": "[INFO]", "WARN": "[WARN]", "ERROR": "[ERROR]", "OK": "[OK]"}
        print(f"{timestamp} {prefix.get(level, '[???]')} {message}")

    async def _emit_stage_completion(
        self,
        event: StageEvent,
        success: bool,
        board_type: str = "square8",
        num_players: int = 2,
        games_generated: int = 0,
        model_path: Optional[str] = None,
        win_rate: Optional[float] = None,
        promoted: bool = False,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Emit a stage completion event to trigger downstream actions.

        Returns the number of callbacks invoked.
        """
        result = StageCompletionResult(
            event=event,
            success=success,
            iteration=self.state.iteration,
            timestamp=datetime.now().isoformat(),
            board_type=board_type,
            num_players=num_players,
            games_generated=games_generated,
            model_path=model_path,
            win_rate=win_rate,
            promoted=promoted,
            error=error,
            metadata=metadata or {},
        )
        return await self.event_bus.emit(result)

    async def run_remote_command(
        self,
        worker: WorkerConfig,
        command: str,
        background: bool = False,
        log_output_on_error: bool = True,
    ) -> Tuple[int, str, str]:
        """Run a command on a remote worker via SSH with retry logic.

        Uses exponential backoff for transient network failures.
        Logs full stdout/stderr on errors for debugging.
        """
        ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30"]
        if worker.ssh_key:
            ssh_cmd.extend(["-i", worker.ssh_key])
        if worker.ssh_port != 22:
            ssh_cmd.extend(["-p", str(worker.ssh_port)])
        ssh_cmd.append(worker.host)

        if background:
            # Wrap command with nohup for background execution
            # Save output to log file for later inspection
            log_file = f"/tmp/ringrift_bg_{int(time.time())}.log"
            full_cmd = f"cd {worker.remote_path} && nohup bash -c '{command}' > {log_file} 2>&1 &"
        else:
            full_cmd = f"cd {worker.remote_path} && {command}"

        ssh_cmd.append(full_cmd)

        if self.dry_run:
            self.log(f"[DRY RUN] {worker.name}: {full_cmd[:100]}...")
            return 0, "", ""

        # Retry loop with exponential backoff
        last_error = ""
        for attempt in range(SSH_MAX_RETRIES):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *ssh_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=300,  # 5 minute timeout per command
                )
                stdout = stdout_bytes.decode()
                stderr = stderr_bytes.decode()
                returncode = proc.returncode or 0

                # Log full output on error for debugging
                if returncode != 0 and log_output_on_error:
                    self._log_command_failure(worker, command, returncode, stdout, stderr)

                return returncode, stdout, stderr

            except asyncio.TimeoutError:
                last_error = "Command timed out after 5 minutes"
                self.log(f"SSH timeout to {worker.name} (attempt {attempt + 1}/{SSH_MAX_RETRIES})", "WARN")

            except Exception as e:
                last_error = str(e)
                self.log(f"SSH error to {worker.name}: {e} (attempt {attempt + 1}/{SSH_MAX_RETRIES})", "WARN")

            # Exponential backoff with jitter
            if attempt < SSH_MAX_RETRIES - 1:
                delay = min(
                    SSH_BASE_DELAY * (SSH_BACKOFF_FACTOR ** attempt) + random.uniform(0, 1),
                    SSH_MAX_DELAY,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        self.log(f"SSH to {worker.name} failed after {SSH_MAX_RETRIES} attempts: {last_error}", "ERROR")
        self.state.errors.append(f"SSH failure to {worker.name}: {last_error}")
        return 1, "", last_error

    def _log_command_failure(
        self,
        worker: WorkerConfig,
        command: str,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        """Log detailed command failure for debugging."""
        log_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"COMMAND FAILURE at {datetime.now().isoformat()}\n")
            f.write(f"Worker: {worker.name} ({worker.host})\n")
            f.write(f"Return code: {returncode}\n")
            f.write(f"Command: {command[:500]}...\n" if len(command) > 500 else f"Command: {command}\n")
            f.write(f"\n--- STDOUT ---\n{stdout[:5000]}\n")
            f.write(f"\n--- STDERR ---\n{stderr[:5000]}\n")
            f.write(f"{'='*60}\n")

    async def check_worker_health(self, worker: WorkerConfig) -> bool:
        """Check if a worker is reachable and healthy."""
        code, stdout, stderr = await self.run_remote_command(worker, "echo 'healthy'")
        return code == 0 and "healthy" in stdout

    async def get_worker_game_count(self, worker: WorkerConfig) -> int:
        """Get the total game count from a worker's selfplay.db."""
        cmd = "source venv/bin/activate && sqlite3 data/games/selfplay.db 'SELECT COUNT(*) FROM games WHERE status=\"completed\"' 2>/dev/null || echo 0"
        code, stdout, _ = await self.run_remote_command(worker, cmd, log_output_on_error=False)
        if code == 0:
            try:
                return int(stdout.strip())
            except ValueError:
                pass
        return 0

    # =========================================================================
    # Smart Polling Methods
    # =========================================================================

    async def poll_for_selfplay_completion(
        self,
        min_games: int = SELFPLAY_MIN_GAMES_THRESHOLD,
        max_wait_minutes: int = 30,
        board_type: str = "square8",
        num_players: int = 2,
        emit_event: bool = True,
    ) -> int:
        """Poll workers until sufficient selfplay games are generated.

        Returns total games generated across all workers.
        Emits SELFPLAY_COMPLETE event when threshold is reached.
        """
        self.log(f"Polling for selfplay completion (min {min_games} games, max {max_wait_minutes} min)...")
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        total_games = 0

        while time.time() - start_time < max_wait_seconds:
            total_games = 0
            for worker in WORKERS:
                if worker.role in ["selfplay", "mixed"]:
                    count = await self.get_worker_game_count(worker)
                    total_games += count

            elapsed_min = (time.time() - start_time) / 60
            self.log(f"  Selfplay progress: {total_games} games ({elapsed_min:.1f} min elapsed)")

            if total_games >= min_games:
                self.log(f"Selfplay target reached: {total_games} games", "OK")
                # Emit completion event to trigger downstream stages
                if emit_event:
                    await self._emit_stage_completion(
                        event=StageEvent.SELFPLAY_COMPLETE,
                        success=True,
                        board_type=board_type,
                        num_players=num_players,
                        games_generated=total_games,
                    )
                return total_games

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        self.log(f"Selfplay timeout after {max_wait_minutes} min with {total_games} games", "WARN")
        # Emit timeout event (partial success if some games generated)
        if emit_event:
            await self._emit_stage_completion(
                event=StageEvent.SELFPLAY_COMPLETE,
                success=total_games > 0,
                board_type=board_type,
                num_players=num_players,
                games_generated=total_games,
                error=f"Timeout after {max_wait_minutes} min" if total_games < min_games else None,
            )
        return total_games

    async def poll_for_cmaes_completion(
        self,
        max_wait_minutes: int = MAX_PHASE_WAIT_MINUTES,
    ) -> bool:
        """Poll workers until CMA-ES jobs complete.

        Returns True if all CMA-ES jobs completed, False if timed out.
        """
        self.log(f"Polling for CMA-ES completion (max {max_wait_minutes} min)...")
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60

        while time.time() - start_time < max_wait_seconds:
            running_count = 0
            completed_count = 0

            for worker in WORKERS:
                if worker.role in ["cmaes", "mixed", "training"]:
                    code, stdout, _ = await self.run_remote_command(
                        worker, CMAES_COMPLETION_CHECK_CMD, log_output_on_error=False
                    )
                    if code == 0:
                        if "running" in stdout:
                            running_count += 1
                        else:
                            completed_count += 1

            elapsed_min = (time.time() - start_time) / 60
            self.log(f"  CMA-ES progress: {completed_count} done, {running_count} running ({elapsed_min:.1f} min)")

            if running_count == 0:
                self.log("All CMA-ES jobs completed", "OK")
                return True

            await asyncio.sleep(POLL_INTERVAL_SECONDS * 2)  # Less frequent polling for CMA-ES

        self.log(f"CMA-ES timeout after {max_wait_minutes} min", "WARN")
        return False

    async def poll_for_training_completion(
        self,
        worker: WorkerConfig,
        max_wait_minutes: int = 60,
        board_type: str = "square8",
        num_players: int = 2,
        model_path: Optional[str] = None,
        emit_event: bool = True,
    ) -> bool:
        """Poll a worker until training job completes.

        Emits TRAINING_COMPLETE event when training finishes.
        """
        self.log(f"Polling {worker.name} for training completion...")
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        check_cmd = "pgrep -f 'app.training.train' >/dev/null && echo running || echo done"

        while time.time() - start_time < max_wait_seconds:
            code, stdout, _ = await self.run_remote_command(
                worker, check_cmd, log_output_on_error=False
            )
            if code == 0 and "done" in stdout:
                self.log(f"Training on {worker.name} completed", "OK")
                # Emit completion event to trigger evaluation
                if emit_event:
                    await self._emit_stage_completion(
                        event=StageEvent.TRAINING_COMPLETE,
                        success=True,
                        board_type=board_type,
                        num_players=num_players,
                        model_path=model_path,
                        metadata={"worker": worker.name},
                    )
                return True

            elapsed_min = (time.time() - start_time) / 60
            if int(elapsed_min) % 5 == 0 and elapsed_min > 0:  # Log every 5 minutes
                self.log(f"  Training still running on {worker.name} ({elapsed_min:.0f} min)")

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        self.log(f"Training timeout on {worker.name} after {max_wait_minutes} min", "WARN")
        # Emit timeout event
        if emit_event:
            await self._emit_stage_completion(
                event=StageEvent.TRAINING_COMPLETE,
                success=False,
                board_type=board_type,
                num_players=num_players,
                model_path=model_path,
                error=f"Timeout after {max_wait_minutes} min on {worker.name}",
                metadata={"worker": worker.name},
            )
        return False

    # =========================================================================
    # Elo Rating System
    # =========================================================================

    def update_elo_rating(
        self,
        player_a: str,
        player_b: str,
        score_a: float,  # 1.0 = A wins, 0.5 = draw, 0.0 = B wins
        k_factor: float = 32.0,
    ) -> Tuple[float, float]:
        """Update Elo ratings for two players after a match.

        Uses standard Elo formula with configurable K-factor.
        Returns (new_rating_a, new_rating_b).
        """
        # Initialize ratings if not seen before
        if player_a not in self.state.elo_ratings:
            self.state.elo_ratings[player_a] = 1500.0
        if player_b not in self.state.elo_ratings:
            self.state.elo_ratings[player_b] = 1500.0

        rating_a = self.state.elo_ratings[player_a]
        rating_b = self.state.elo_ratings[player_b]

        # Expected scores
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a

        # New ratings
        new_rating_a = rating_a + k_factor * (score_a - expected_a)
        new_rating_b = rating_b + k_factor * ((1 - score_a) - expected_b)

        # Update state
        self.state.elo_ratings[player_a] = new_rating_a
        self.state.elo_ratings[player_b] = new_rating_b

        # Record history
        self.state.elo_history.append({
            "timestamp": datetime.now().isoformat(),
            "player_a": player_a,
            "player_b": player_b,
            "score_a": score_a,
            "rating_a_before": rating_a,
            "rating_b_before": rating_b,
            "rating_a_after": new_rating_a,
            "rating_b_after": new_rating_b,
        })

        self._save_state()
        return new_rating_a, new_rating_b

    def get_elo_leaderboard(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top N models by Elo rating."""
        sorted_ratings = sorted(
            self.state.elo_ratings.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_ratings[:top_n]

    # =========================================================================
    # Model Registry
    # =========================================================================

    def register_model(
        self,
        model_id: str,
        model_path: str,
        config: str,
        parent_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a trained model in the registry.

        Tracks model lineage, training config, and performance metrics.
        """
        self.state.model_registry[model_id] = {
            "model_id": model_id,
            "path": model_path,
            "config": config,
            "parent_id": parent_id,
            "created_at": datetime.now().isoformat(),
            "iteration": self.state.iteration,
            "metrics": metrics or {},
            "status": "active",
        }
        self._save_state()
        self.log(f"Registered model: {model_id}", "OK")

    def get_best_model(self, config: str) -> Optional[str]:
        """Get the best performing model for a configuration based on Elo."""
        matching_models = [
            (mid, meta)
            for mid, meta in self.state.model_registry.items()
            if meta.get("config") == config and meta.get("status") == "active"
        ]
        if not matching_models:
            return None

        # Sort by Elo rating (if available) or fallback to recency
        def sort_key(item: Tuple[str, Dict]) -> float:
            mid, _ = item
            return self.state.elo_ratings.get(mid, 1500.0)

        best = max(matching_models, key=sort_key)
        return best[0]

    def deprecate_model(self, model_id: str, reason: str = "") -> None:
        """Mark a model as deprecated in the registry."""
        if model_id in self.state.model_registry:
            self.state.model_registry[model_id]["status"] = "deprecated"
            self.state.model_registry[model_id]["deprecated_at"] = datetime.now().isoformat()
            self.state.model_registry[model_id]["deprecation_reason"] = reason
            self._save_state()
            self.log(f"Deprecated model: {model_id} ({reason})")

    # =========================================================================
    # Game Deduplication
    # =========================================================================

    def hash_game(self, game_data: Dict[str, Any]) -> str:
        """Generate a unique hash for a game based on its moves."""
        # Use moves + initial state + outcome for uniqueness
        key_data = json.dumps({
            "moves": game_data.get("moves", []),
            "board_type": game_data.get("board_type"),
            "num_players": game_data.get("num_players"),
            "outcome": game_data.get("outcome"),
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def is_duplicate_game(self, game_data: Dict[str, Any]) -> bool:
        """Check if a game is a duplicate of one we've already seen."""
        game_hash = self.hash_game(game_data)
        if game_hash in self.state.seen_game_hashes:
            return True
        self.state.seen_game_hashes.add(game_hash)
        return False

    async def deduplicate_training_data(self, db_path: str) -> int:
        """Remove duplicate games from a training database.

        Returns the number of duplicates removed.
        """
        self.log(f"Deduplicating games in {db_path}...")

        # Read all games from the database
        cmd = f"sqlite3 {db_path} \"SELECT game_id, board_type, num_players, move_history, outcome FROM games WHERE status='completed'\""
        code, stdout, stderr = await self.run_remote_command(
            WORKERS[0] if WORKERS else WorkerConfig("local", "localhost", "training", []),
            cmd,
            log_output_on_error=False,
        )

        if code != 0 or not stdout.strip():
            self.log(f"Could not read games for deduplication: {stderr}", "WARN")
            return 0

        # Parse games and identify duplicates
        duplicates = []
        for line in stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 5:
                game_id, board_type, num_players, moves, outcome = parts[:5]
                game_data = {
                    "board_type": board_type,
                    "num_players": int(num_players) if num_players.isdigit() else 2,
                    "moves": moves,
                    "outcome": outcome,
                }
                if self.is_duplicate_game(game_data):
                    duplicates.append(game_id)

        # Remove duplicates
        if duplicates:
            game_ids = "','".join(duplicates)
            delete_cmd = f"sqlite3 {db_path} \"DELETE FROM games WHERE game_id IN ('{game_ids}')\""
            await self.run_remote_command(
                WORKERS[0] if WORKERS else WorkerConfig("local", "localhost", "training", []),
                delete_cmd,
            )
            self.log(f"Removed {len(duplicates)} duplicate games", "OK")

        self._save_state()
        return len(duplicates)

    # =========================================================================
    # Tier Gating
    # =========================================================================

    async def check_tier_promotion(
        self,
        config: str,
        current_tier: str,
        win_rate_threshold: float = 0.55,
    ) -> Tuple[bool, str]:
        """Check if a model should be promoted to the next tier.

        Promotion requires winning > threshold against current tier opponents.
        Returns (should_promote, new_tier).
        """
        # Tier progression: D2 -> D4 -> D6 -> D8
        TIER_PROGRESSION = {
            "D2": "D4",
            "D4": "D6",
            "D6": "D8",
            "D8": "D8",  # Max tier
        }

        if current_tier not in TIER_PROGRESSION:
            return False, current_tier

        next_tier = TIER_PROGRESSION[current_tier]
        if next_tier == current_tier:
            self.log(f"{config} already at max tier {current_tier}")
            return False, current_tier

        # Check recent tournament results for this config
        model_key = f"{config}_best"
        if model_key not in self.state.elo_ratings:
            self.log(f"No Elo data for {config}, skipping tier check")
            return False, current_tier

        # Compare against tier benchmark
        tier_benchmark = f"tier_{current_tier}_benchmark"
        if tier_benchmark not in self.state.elo_ratings:
            # No benchmark yet, allow promotion based on raw Elo
            if self.state.elo_ratings.get(model_key, 1500) > 1600:
                return True, next_tier
            return False, current_tier

        # Check win rate from recent matches
        recent_matches = [
            m for m in self.state.elo_history[-50:]  # Last 50 matches
            if m.get("player_a") == model_key or m.get("player_b") == model_key
        ]

        if len(recent_matches) < 10:
            self.log(f"Insufficient matches for {config} tier gating ({len(recent_matches)}/10)")
            return False, current_tier

        wins = sum(
            1 for m in recent_matches
            if (m.get("player_a") == model_key and m.get("score_a", 0) > 0.5)
            or (m.get("player_b") == model_key and m.get("score_a", 0) < 0.5)
        )
        win_rate = wins / len(recent_matches)

        if win_rate >= win_rate_threshold:
            self.log(f"{config} promoted from {current_tier} to {next_tier} (win rate: {win_rate:.1%})", "OK")
            self.state.tier_promotions[config] = next_tier
            self._save_state()
            return True, next_tier

        self.log(f"{config} remains at {current_tier} (win rate: {win_rate:.1%} < {win_rate_threshold:.1%})")
        return False, current_tier

    async def run_tier_gating_phase(self) -> Dict[str, str]:
        """Run tier gating checks for all configurations.

        Returns dict of config -> new_tier for any promotions.
        """
        self.log("=== Running Tier Gating Checks ===")

        promotions = {}
        configs = ["square8_2p", "square8_3p", "square8_4p",
                   "square19_2p", "square19_3p", "square19_4p",
                   "hex_2p", "hex_3p", "hex_4p"]

        for config in configs:
            current_tier = self.state.tier_promotions.get(config, "D2")
            promoted, new_tier = await self.check_tier_promotion(config, current_tier)
            if promoted:
                promotions[config] = new_tier

        if promotions:
            self.log(f"Tier promotions: {promotions}", "OK")
        else:
            self.log("No tier promotions this iteration")

        return promotions

    # =========================================================================
    # Resource Monitoring
    # =========================================================================

    async def get_worker_resources(self, worker: WorkerConfig) -> Dict[str, Any]:
        """Get resource utilization from a worker."""
        cmd = """
        echo "cpu:$(top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1 2>/dev/null || echo 0)"
        echo "mem:$(free -m 2>/dev/null | awk '/Mem:/{print $3/$2*100}' || echo 0)"
        echo "disk:$(df -h . 2>/dev/null | awk 'NR==2{print $5}' | tr -d '%' || echo 0)"
        echo "gpu:$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)"
        """
        code, stdout, _ = await self.run_remote_command(worker, cmd, log_output_on_error=False)

        resources = {"cpu": 0.0, "mem": 0.0, "disk": 0.0, "gpu": 0.0}
        if code == 0:
            for line in stdout.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    try:
                        resources[key] = float(value.strip())
                    except ValueError:
                        pass

        return resources

    async def log_resource_usage(self) -> None:
        """Log resource usage across all workers."""
        self.log("=== Resource Usage ===")
        for worker in WORKERS:
            if await self.check_worker_health(worker):
                resources = await self.get_worker_resources(worker)
                self.log(
                    f"  {worker.name}: CPU={resources['cpu']:.0f}%, "
                    f"MEM={resources['mem']:.0f}%, DISK={resources['disk']:.0f}%, "
                    f"GPU={resources['gpu']:.0f}%"
                )

    async def dispatch_selfplay(
        self,
        worker: WorkerConfig,
        job: SelfplayJob,
        iteration: int,
        job_key: str,
    ) -> bool:
        """Dispatch a selfplay job to a worker.

        The job will use trained heuristic profiles if available on the worker,
        and neural networks if the job requests them and checkpoints exist.
        """
        seed = iteration * 10000 + job.seed
        log_file = f"logs/selfplay/iter{iteration}_{job_key}.jsonl"

        # Build optional flags
        optional_flags = []

        # Use trained heuristic profiles if available
        if job.use_trained_profiles:
            # Workers should have trained_heuristic_profiles.json synced from profile-sync phase
            optional_flags.append(
                f"--heuristic-weights-file {worker.remote_path}/data/trained_heuristic_profiles.json"
            )
            # Use board-specific profile key
            board_abbrev = {"square8": "sq8", "square19": "sq19", "hexagonal": "hex"}.get(
                job.board_type, job.board_type[:3]
            )
            profile_key = f"heuristic_v1_{board_abbrev}_{job.num_players}p"
            optional_flags.append(f"--heuristic-profile {profile_key}")

        # Enable neural network for nn-only or when explicitly requested
        env_extras = ""
        if job.use_neural_net or job.engine_mode == "nn-only":
            env_extras = "export RINGRIFT_USE_NEURAL_NET=1\n"

        flags_str = " \\\n    ".join(optional_flags) if optional_flags else ""

        cmd = f"""
source venv/bin/activate
export PYTHONPATH={worker.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
export RINGRIFT_TRAINED_HEURISTIC_PROFILES={worker.remote_path}/data/trained_heuristic_profiles.json
{env_extras}python scripts/run_self_play_soak.py \\
    --num-games {job.num_games} \\
    --board-type {job.board_type} \\
    --engine-mode {job.engine_mode} \\
    --num-players {job.num_players} \\
    --max-moves {job.max_moves} \\
    --seed {seed} \\
    --log-jsonl {log_file} \\
    {flags_str}
"""
        self.log(f"Dispatching {job_key} ({job.num_games} {job.engine_mode} games) to {worker.name}")

        code, _, stderr = await self.run_remote_command(worker, cmd, background=True)
        if code != 0:
            self.log(f"Failed to dispatch to {worker.name}: {stderr}", "ERROR")
            return False
        return True

    async def run_selfplay_phase(self, iteration: int) -> Dict[str, int]:
        """Run the selfplay phase across all workers."""
        self.state.phase = "selfplay"
        self._save_state()
        self.log(f"=== Starting Selfplay Phase (Iteration {iteration}) ===")

        # Check worker health
        healthy_workers = []
        for worker in WORKERS:
            # CPU-bound selfplay (run_self_play_soak) should run on CPU workers.
            if worker.role in ["selfplay", "mixed"] and worker.is_cpu_worker():
                if await self.check_worker_health(worker):
                    healthy_workers.append(worker)
                    self.log(f"Worker {worker.name}: healthy", "OK")
                else:
                    self.log(f"Worker {worker.name}: unreachable", "WARN")

        if not healthy_workers:
            self.log("No healthy workers available!", "ERROR")
            return {}

        # Distribute jobs across workers with slot-weighted round-robin assignment.
        # This respects per-worker concurrency limits (max_parallel_jobs) while
        # still ensuring all job types get dispatched.
        job_items = list(DEFAULT_SELFPLAY_CONFIG.items())
        tasks = []
        worker_slots: List[WorkerConfig] = []
        for worker in healthy_workers:
            worker_slots.extend([worker] * max(1, int(worker.max_parallel_jobs or 1)))
        worker_idx = 0

        for job_key, job in job_items:
            if not worker_slots:
                break
            # Round-robin worker assignment
            worker = worker_slots[worker_idx % len(worker_slots)]
            tasks.append(self.dispatch_selfplay(worker, job, iteration, job_key))
            worker_idx += 1

        # Wait for all dispatches to complete
        results = await asyncio.gather(*tasks)
        success_count = sum(1 for r in results if r)
        self.log(f"Dispatched {success_count}/{len(tasks)} selfplay jobs across {len(healthy_workers)} workers")

        return {"dispatched": success_count, "total": len(tasks)}

    async def run_sync_phase(self, emit_event: bool = True) -> bool:
        """Sync all selfplay AND tournament data from remote workers.

        This phase pulls:
        1. Selfplay game databases from workers (via sync_selfplay_data.sh)
        2. Tournament JSONL files from workers (for training data)

        Both sources are merged into the training data pool.
        Emits SYNC_COMPLETE event when sync finishes.
        """
        self.state.phase = "sync"
        self._save_state()
        self.log("=== Starting Data Sync Phase ===")

        success = True
        error_msg = None

        # Step 1: Sync selfplay databases using existing script
        sync_script = Path(__file__).parent / "sync_selfplay_data.sh"
        if sync_script.exists():
            if self.dry_run:
                self.log(f"[DRY RUN] Would run: {sync_script} --merge --to-mac-studio")
            else:
                try:
                    result = subprocess.run(
                        [str(sync_script), "--merge", "--to-mac-studio"],
                        capture_output=True,
                        text=True,
                        timeout=1800,  # 30 minute timeout
                    )
                    if result.returncode == 0:
                        self.log("Selfplay DB sync completed", "OK")
                    else:
                        self.log(f"Selfplay sync failed: {result.stderr[:200]}", "ERROR")
                        success = False
                        error_msg = f"Selfplay sync failed: {result.stderr[:200]}"
                except subprocess.TimeoutExpired:
                    self.log("Selfplay sync timed out", "ERROR")
                    success = False
                    error_msg = "Selfplay sync timed out after 30 minutes"
                except Exception as e:
                    self.log(f"Selfplay sync error: {e}", "ERROR")
                    success = False
                    error_msg = f"Selfplay sync error: {e}"
        else:
            self.log(f"Sync script not found: {sync_script}", "WARN")

        # Step 2: Sync tournament JSONL files from workers
        merged_games = await self.sync_tournament_games()

        self.state.last_sync = datetime.now().isoformat()
        self._save_state()

        # Emit completion event
        if emit_event:
            await self._emit_stage_completion(
                event=StageEvent.SYNC_COMPLETE,
                success=success,
                games_generated=merged_games,
                error=error_msg,
            )

        return success

    async def sync_tournament_games(self) -> int:
        """Pull tournament game JSONL files from all workers and merge locally.

        Tournament games are high-quality training data from strong AI matchups.
        They are saved as JSONL files in logs/tournaments/ on each worker.
        """
        self.log("Syncing tournament games from workers...")

        local_tournament_dir = Path("logs/tournaments/merged")
        local_tournament_dir.mkdir(parents=True, exist_ok=True)

        merged_count = 0

        for worker in WORKERS:
            if not await self.check_worker_health(worker):
                continue

            # Find tournament JSONL files on worker
            find_cmd = f"find {worker.remote_path}/logs/tournaments -name 'games.jsonl' 2>/dev/null || true"
            code, stdout, _ = await self.run_remote_command(worker, find_cmd)

            if code != 0 or not stdout.strip():
                continue

            jsonl_files = stdout.strip().split("\n")
            self.log(f"  {worker.name}: Found {len(jsonl_files)} tournament files")

            for remote_path in jsonl_files:
                if not remote_path.strip():
                    continue

                # Read remote file content
                cat_cmd = f"cat {remote_path}"
                code, content, _ = await self.run_remote_command(worker, cat_cmd)

                if code != 0 or not content.strip():
                    continue

                # Append to local merged file
                merged_file = local_tournament_dir / "all_tournaments.jsonl"
                with open(merged_file, "a") as f:
                    f.write(content)
                    if not content.endswith("\n"):
                        f.write("\n")

                lines = len(content.strip().split("\n"))
                merged_count += lines

        if merged_count > 0:
            self.log(f"Merged {merged_count} tournament games to {local_tournament_dir}", "OK")

        return merged_count

    # =========================================================================
    # Canonical Selfplay Phase (CPU-based with full phase machine)
    # =========================================================================

    async def run_canonical_selfplay_phase(
        self,
        iteration: int,
        games_per_worker: int = 500,
        board_type: str = "square8",
        num_players: int = 2,
    ) -> Dict[str, Any]:
        """Run canonical selfplay across all workers using CPU engine.

        Unlike GPU selfplay which uses a simplified phase machine, canonical
        selfplay uses the full 7-phase FSM and produces games that pass the
        parity validation gate.

        Args:
            iteration: Pipeline iteration number
            games_per_worker: Number of games per worker
            board_type: Board type (square8, square19, hexagonal)
            num_players: Number of players (2, 3, 4)

        Returns:
            Dict with dispatched count and worker details
        """
        self.state.phase = "canonical_selfplay"
        self._save_state()
        self.log(f"=== Starting Canonical Selfplay Phase (Iteration {iteration}) ===")

        # Find healthy CPU workers (canonical selfplay is CPU-bound)
        healthy_workers = []
        for worker in WORKERS:
            if not worker.is_cpu_worker():
                continue
            if await self.check_worker_health(worker):
                healthy_workers.append(worker)
                self.log(f"Worker {worker.name}: healthy", "OK")
            else:
                self.log(f"Worker {worker.name}: unreachable", "WARN")

        if not healthy_workers:
            self.log("No healthy workers available!", "ERROR")
            return {"dispatched": 0, "workers": []}

        # Dispatch canonical selfplay to each worker
        tasks = []
        for worker in healthy_workers:
            seed = iteration * 10000 + hash(worker.name) % 10000
            log_file = f"logs/selfplay/canonical_iter{iteration}_{worker.name}.jsonl"
            db_file = f"data/games/canonical_{board_type}_{num_players}p_{worker.name}.db"

            cmd = f"""
source venv/bin/activate || true
export PYTHONPATH={worker.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
python3 scripts/run_self_play_soak.py \\
    --num-games {games_per_worker} \\
    --board-type {board_type} \\
    --num-players {num_players} \\
    --difficulty-band light \\
    --seed {seed} \\
    --log-jsonl {log_file} \\
    --record-db {db_file}
"""
            self.log(f"Dispatching canonical selfplay ({games_per_worker} games) to {worker.name}")
            tasks.append(self._dispatch_canonical_selfplay(worker, cmd))

        results = await asyncio.gather(*tasks)
        success_count = sum(1 for r in results if r)

        return {
            "dispatched": success_count,
            "total": len(tasks),
            "workers": [w.name for w in healthy_workers],
            "games_per_worker": games_per_worker,
        }

    async def _dispatch_canonical_selfplay(
        self,
        worker: WorkerConfig,
        cmd: str,
    ) -> bool:
        """Dispatch canonical selfplay to a single worker."""
        code, _, stderr = await self.run_remote_command(worker, cmd, background=True)
        if code != 0:
            self.log(f"Failed to dispatch canonical selfplay to {worker.name}: {stderr}", "ERROR")
            return False
        return True

    async def poll_for_canonical_selfplay_completion(
        self,
        board_type: str = "square8",
        num_players: int = 2,
        min_games: int = 100,
        max_wait_minutes: int = 120,
    ) -> int:
        """Poll workers until canonical selfplay is complete.

        Returns total games generated across all workers.
        """
        self.log(f"Polling for canonical selfplay completion (min {min_games} games)...")
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60

        while time.time() - start_time < max_wait_seconds:
            total_games = 0
            all_complete = True

            for worker in WORKERS:
                if not await self.check_worker_health(worker):
                    continue

                # Check if selfplay is still running
                check_cmd = "pgrep -f 'run_self_play_soak' >/dev/null && echo running || echo done"
                code, stdout, _ = await self.run_remote_command(worker, check_cmd, log_output_on_error=False)

                if code == 0 and "running" in stdout:
                    all_complete = False

                # Count games in database
                db_pattern = f"data/games/canonical_{board_type}_{num_players}p_*.db"
                count_cmd = f"for db in {worker.remote_path}/{db_pattern}; do sqlite3 \"$db\" 'SELECT COUNT(*) FROM games' 2>/dev/null || echo 0; done | paste -sd+ - | bc"
                code, stdout, _ = await self.run_remote_command(worker, count_cmd, log_output_on_error=False)

                if code == 0 and stdout.strip().isdigit():
                    total_games += int(stdout.strip())

            elapsed_min = (time.time() - start_time) / 60
            self.log(f"  Canonical selfplay: {total_games} games ({elapsed_min:.1f} min elapsed)")

            if all_complete or total_games >= min_games:
                self.log(f"Canonical selfplay complete: {total_games} games", "OK")
                return total_games

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        self.log(f"Canonical selfplay timeout after {max_wait_minutes} min", "WARN")
        return total_games

    # =========================================================================
    # Parity Validation Gate
    # =========================================================================

    async def run_parity_validation_phase(
        self,
        db_paths: Optional[List[str]] = None,
        board_type: str = "square8",
        num_players: int = 2,
    ) -> Dict[str, Any]:
        """Run parity validation gate on game databases.

        This phase validates that all games in the databases pass the canonical
        parity check (TS engine replay produces identical states to Python).

        Only games that pass validation are used for training.

        Args:
            db_paths: List of database paths to validate (if None, uses pattern match)
            board_type: Board type for pattern matching
            num_players: Number of players for pattern matching

        Returns:
            Dict with validation results
        """
        self.state.phase = "parity_validation"
        self._save_state()
        self.log("=== Starting Parity Validation Phase ===")

        results = {
            "total_games_checked": 0,
            "games_passed": 0,
            "games_failed": 0,
            "databases_validated": [],
            "passed": False,
        }

        # Find validation worker (prefer CPU-heavy host; parity validation is CPU-bound)
        preferred_names = ["mac-studio", "lambda-a10", "aws-staging"]
        candidates = [w for w in WORKERS if w.is_cpu_worker()] or list(WORKERS)
        validation_worker = next((w for w in candidates if w.name in preferred_names), candidates[0] if candidates else None)

        if not validation_worker or not await self.check_worker_health(validation_worker):
            self.log("No worker available for parity validation", "ERROR")
            return results

        # Build database list
        if db_paths:
            db_list = " ".join(db_paths)
        else:
            db_pattern = f"data/games/canonical_{board_type}_{num_players}p_*.db"
            db_list = f"{validation_worker.remote_path}/{db_pattern}"

        # Run parity validation script
        validation_cmd = f"""
source venv/bin/activate || true
export PYTHONPATH={validation_worker.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
python3 scripts/run_parity_validation.py \\
    --databases {db_list} \\
    --mode canonical \\
    --output-json data/parity_validation_results.json \\
    --progress-every 100
"""
        self.log(f"Running parity validation on {validation_worker.name}...")
        code, stdout, stderr = await self.run_remote_command(
            validation_worker, validation_cmd, background=False
        )

        if code != 0:
            self.log(f"Parity validation failed: {stderr[:200]}", "ERROR")
            return results

        # Parse results
        result_cmd = f"cat {validation_worker.remote_path}/data/parity_validation_results.json"
        code, stdout, _ = await self.run_remote_command(validation_worker, result_cmd)

        if code == 0 and stdout.strip():
            try:
                validation_results = json.loads(stdout)
                results["total_games_checked"] = validation_results.get("total_games_checked", 0)
                results["games_passed"] = (
                    results["total_games_checked"] -
                    validation_results.get("games_with_semantic_divergence", 0) -
                    validation_results.get("games_with_structural_issues", 0)
                )
                results["games_failed"] = (
                    validation_results.get("games_with_semantic_divergence", 0) +
                    validation_results.get("games_with_structural_issues", 0)
                )
                results["passed"] = validation_results.get("passed_canonical_parity_gate", False)
                results["databases_validated"] = validation_results.get("db_paths_checked", [])
            except json.JSONDecodeError:
                self.log("Could not parse validation results", "ERROR")

        if results["passed"]:
            self.log(f"Parity validation PASSED: {results['games_passed']}/{results['total_games_checked']} games", "OK")
        else:
            self.log(f"Parity validation FAILED: {results['games_failed']} games with issues", "ERROR")

        return results

    # =========================================================================
    # NPZ Export Phase
    # =========================================================================

    async def run_npz_export_phase(
        self,
        iteration: int,
        board_type: str = "square8",
        num_players: int = 2,
        output_dir: str = "data/training",
    ) -> Dict[str, Any]:
        """Export validated games to NPZ format for neural network training.

        This phase converts game databases to the tensor format required by
        the neural network training scripts.

        Args:
            iteration: Pipeline iteration number
            board_type: Board type
            num_players: Number of players
            output_dir: Output directory for NPZ files

        Returns:
            Dict with export results
        """
        self.state.phase = "npz_export"
        self._save_state()
        self.log("=== Starting NPZ Export Phase ===")

        results = {
            "total_positions": 0,
            "output_files": [],
            "success": False,
        }

        # Find training worker
        training_worker = next(
            (w for w in WORKERS if w.role in ["training", "nn_training", "nn_training_primary"]),
            next((w for w in WORKERS if w.name == "mac-studio"), WORKERS[0] if WORKERS else None)
        )

        if not training_worker or not await self.check_worker_health(training_worker):
            self.log("No worker available for NPZ export", "ERROR")
            return results

        # Find validated databases
        db_pattern = f"data/games/canonical_{board_type}_{num_players}p_*.db"
        output_prefix = f"{output_dir}/{board_type}_{num_players}p_iter{iteration}"

        export_cmd = f"""
source venv/bin/activate || true
export PYTHONPATH={training_worker.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
mkdir -p {training_worker.remote_path}/{output_dir}
python3 scripts/export_training_data.py \\
    --databases {training_worker.remote_path}/{db_pattern} \\
    --board-type {board_type} \\
    --num-players {num_players} \\
    --output-prefix {training_worker.remote_path}/{output_prefix} \\
    --format npz \\
    --include-value-targets \\
    --include-policy-targets
"""
        self.log(f"Exporting training data on {training_worker.name}...")
        code, stdout, stderr = await self.run_remote_command(
            training_worker, export_cmd, background=False
        )

        if code != 0:
            self.log(f"NPZ export failed: {stderr[:200]}", "ERROR")
            return results

        # Check exported files
        check_cmd = f"ls -la {training_worker.remote_path}/{output_prefix}*.npz 2>/dev/null | wc -l"
        code, stdout, _ = await self.run_remote_command(training_worker, check_cmd)

        if code == 0 and stdout.strip().isdigit() and int(stdout.strip()) > 0:
            results["success"] = True
            results["output_files"].append(f"{output_prefix}.npz")

            # Get position count
            count_cmd = f"python3 -c \"import numpy as np; d=np.load('{training_worker.remote_path}/{output_prefix}.npz'); print(len(d['states']))\""
            code, stdout, _ = await self.run_remote_command(training_worker, count_cmd)
            if code == 0 and stdout.strip().isdigit():
                results["total_positions"] = int(stdout.strip())

        if results["success"]:
            self.log(f"NPZ export complete: {results['total_positions']} positions", "OK")
        else:
            self.log("NPZ export produced no files", "ERROR")

        return results

    # =========================================================================
    # Dynamic Host Discovery
    # =========================================================================

    async def discover_vast_instances(self) -> List[WorkerConfig]:
        """Dynamically discover running Vast.ai instances.

        Uses the vastai CLI to query running instances and creates
        WorkerConfig entries for each.

        Returns:
            List of WorkerConfig for discovered instances
        """
        self.log("Discovering Vast.ai instances...")

        discovered = []

        try:
            result = subprocess.run(
                ["vastai", "show", "instances", "--raw"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                self.log(f"vastai query failed: {result.stderr[:100]}", "WARN")
                return discovered

            instances = json.loads(result.stdout)

            for inst in instances:
                if inst.get("actual_status") != "running":
                    continue

                # Extract connection info
                ssh_host = inst.get("ssh_host", "")
                ssh_port = inst.get("ssh_port", 22)

                if not ssh_host:
                    continue

                # Determine GPU type and capabilities
                gpu_name = inst.get("gpu_name", "unknown")
                num_gpus = inst.get("num_gpus", 1)

                # Create worker config
                worker = WorkerConfig(
                    name=f"vast-{inst.get('id', 'unknown')}",
                    host=f"root@{ssh_host}",
                    role="nn_training" if "5090" in gpu_name or "H100" in gpu_name else "selfplay",
                    capabilities=["square8", "square19", "hex"],
                    ssh_port=ssh_port,
                    remote_path="~/ringrift/ai-service",
                    max_parallel_jobs=num_gpus,
                )

                discovered.append(worker)
                self.log(f"  Discovered: {worker.name} ({num_gpus}x {gpu_name})", "OK")

        except FileNotFoundError:
            self.log("vastai CLI not found", "WARN")
        except json.JSONDecodeError:
            self.log("Could not parse vastai output", "WARN")
        except subprocess.TimeoutExpired:
            self.log("vastai query timed out", "WARN")
        except Exception as e:
            self.log(f"Vast discovery error: {e}", "WARN")

        return discovered

    async def refresh_workers(self) -> None:
        """Refresh worker list with dynamic discovery.

        Combines static YAML config with dynamically discovered instances.
        """
        global WORKERS

        # Start with static config
        static_workers = load_workers_from_config()

        # Add dynamically discovered Vast instances
        vast_workers = await self.discover_vast_instances()

        # Merge, avoiding duplicates by host
        seen_hosts = {w.host for w in static_workers}
        for vw in vast_workers:
            if vw.host not in seen_hosts:
                static_workers.append(vw)
                seen_hosts.add(vw.host)

        WORKERS = static_workers
        self.log(f"Refreshed worker list: {len(WORKERS)} workers")

    async def sync_heuristic_profiles(self) -> bool:
        """Sync trained heuristic profiles from all workers.

        After CMA-ES completes on remote workers, this method pulls the
        trained_heuristic_profiles.json from each worker and merges them
        into a unified local copy. The merged profiles are then pushed
        back to all workers so they use the latest heuristics for selfplay.
        """
        self.log("=== Syncing Trained Heuristic Profiles ===")

        local_profiles_path = Path("data/trained_heuristic_profiles.json")
        merged_profiles: Dict[str, Any] = {
            "version": "1.3.0",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "updated": datetime.now().strftime("%Y-%m-%d"),
            "description": "CMA-ES optimized heuristic profiles (merged from distributed workers)",
            "profiles": {},
            "training_metadata": {},
        }

        # Load existing local profiles
        if local_profiles_path.exists():
            try:
                with open(local_profiles_path) as f:
                    existing = json.load(f)
                    merged_profiles["profiles"].update(existing.get("profiles", {}))
                    merged_profiles["training_metadata"].update(existing.get("training_metadata", {}))
            except (json.JSONDecodeError, OSError) as e:
                self.log(f"Warning: Could not load existing profiles: {e}", "WARN")

        # Pull profiles from each worker
        pull_count = 0
        for worker in WORKERS:
            if not await self.check_worker_health(worker):
                continue

            # Fetch remote profiles
            cmd = f"cat {worker.remote_path}/data/trained_heuristic_profiles.json 2>/dev/null || echo '{{}}'"
            code, stdout, _ = await self.run_remote_command(worker, cmd)

            if code == 0 and stdout.strip() and stdout.strip() != "{}":
                try:
                    remote_profiles = json.loads(stdout)
                    remote_data = remote_profiles.get("profiles", {})
                    remote_meta = remote_profiles.get("training_metadata", {})

                    # Merge profiles, preferring higher fitness
                    for key, weights in remote_data.items():
                        existing_meta = merged_profiles["training_metadata"].get(key, {})
                        new_meta = remote_meta.get(key, {})

                        existing_fitness = existing_meta.get("fitness", 0)
                        new_fitness = new_meta.get("fitness", 0)

                        if new_fitness > existing_fitness or key not in merged_profiles["profiles"]:
                            merged_profiles["profiles"][key] = weights
                            merged_profiles["training_metadata"][key] = new_meta
                            self.log(f"  Merged {key} from {worker.name} (fitness: {new_fitness:.3f})")
                            pull_count += 1
                except json.JSONDecodeError as e:
                    self.log(f"Warning: Invalid JSON from {worker.name}: {e}", "WARN")

        if pull_count > 0:
            # Save merged profiles locally
            local_profiles_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_profiles_path, "w") as f:
                json.dump(merged_profiles, f, indent=2)
            self.log(f"Saved merged profiles to {local_profiles_path}", "OK")

            # Push merged profiles back to all workers
            profiles_json = json.dumps(merged_profiles)
            push_count = 0
            for worker in WORKERS:
                if not await self.check_worker_health(worker):
                    continue

                push_cmd = f"mkdir -p {worker.remote_path}/data && cat > {worker.remote_path}/data/trained_heuristic_profiles.json << 'EOFPROFILES'\n{profiles_json}\nEOFPROFILES"
                code, _, _ = await self.run_remote_command(worker, push_cmd)
                if code == 0:
                    push_count += 1

            self.log(f"Pushed merged profiles to {push_count} workers", "OK")

        return pull_count > 0

    async def run_training_phase(self, iteration: int) -> Dict[str, bool]:
        """Run NN training on Mac Studio for all board×player configurations.

        Trains neural networks for each configuration that has sufficient
        training data. Prioritizes square8 2p (most data) then expands to
        other configs based on available games.
        """
        self.state.phase = "training"
        self._save_state()
        self.log(f"=== Starting Training Phase (Iteration {iteration}) ===")

        results = {}

        # Find Mac Studio worker (or any training-capable worker)
        training_worker = next(
            (w for w in WORKERS if w.role in ["training", "mixed"] and w.name == "mac-studio"),
            next((w for w in WORKERS if w.role in ["training", "mixed"]), None)
        )
        if not training_worker:
            self.log("No training worker configured", "ERROR")
            return results

        if not await self.check_worker_health(training_worker):
            self.log(f"{training_worker.name} not reachable", "ERROR")
            return results

        # Training configurations: (board, players, epochs, min_games_required)
        # More epochs for primary configs, fewer for secondary
        TRAINING_CONFIGS = [
            # Primary configs - full training
            ("square8", 2, 50, 100),
            ("square8", 3, 40, 50),
            ("square8", 4, 40, 40),
            # Secondary configs - lighter training
            ("square19", 2, 30, 50),
            ("hexagonal", 2, 30, 50),
            # Tertiary configs - minimal training if data available
            ("square19", 3, 20, 30),
            ("square19", 4, 20, 20),
            ("hexagonal", 3, 20, 30),
            ("hexagonal", 4, 20, 20),
        ]

        for board, players, epochs, min_games in TRAINING_CONFIGS:
            config_key = f"{board}_{players}p"

            # Check if sufficient training data exists
            check_cmd = f"sqlite3 {training_worker.remote_path}/data/games/merged_latest.db \"SELECT COUNT(*) FROM games WHERE board_type='{board}' AND num_players={players} AND status='completed'\" 2>/dev/null || echo 0"
            code, stdout, _ = await self.run_remote_command(training_worker, check_cmd)
            game_count = int(stdout.strip()) if code == 0 and stdout.strip().isdigit() else 0

            if game_count < min_games:
                self.log(f"{config_key}: Skipping (only {game_count}/{min_games} games)", "WARN")
                results[config_key] = False
                continue

            # Train this configuration with v3 architecture (spatial policy heads)
            train_cmd = f"""
source venv/bin/activate
export PYTHONPATH={training_worker.remote_path}
export RINGRIFT_TRAINED_HEURISTIC_PROFILES={training_worker.remote_path}/data/trained_heuristic_profiles.json
python -m app.training.train \\
    --data-path data/games/merged_latest.db \\
    --board-type {board} \\
    --num-players {players} \\
    --epochs {epochs} \\
    --batch-size 256 \\
    --device mps \\
    --model-version v3 \\
    --save-path models/{config_key}_iter{iteration}.pth \\
    --save-best models/{config_key}_best.pth
"""
            self.log(f"Training {config_key} ({game_count} games, {epochs} epochs)...")
            code, stdout, stderr = await self.run_remote_command(training_worker, train_cmd, background=False)
            results[config_key] = code == 0

            if code == 0:
                self.log(f"{config_key} training completed", "OK")
                self.state.models_trained.append(f"{config_key}_iter{iteration}")
            else:
                self.log(f"{config_key} training failed: {stderr[:200]}", "ERROR")

        self._save_state()
        return results

    async def run_cmaes_phase(self, iteration: int) -> Dict[str, bool]:
        """Run CMA-ES optimization across all board×player configurations.

        This phase runs iterative CMA-ES heuristic tuning for each of the 9
        board×player combinations. Jobs are distributed across available
        workers based on their capabilities (some workers may not support
        larger boards due to memory constraints).

        The trained heuristic profiles are saved to data/trained_heuristic_profiles.json
        and automatically loaded by subsequent selfplay phases.
        """
        self.state.phase = "cmaes"
        self._save_state()
        self.log(f"=== Starting CMA-ES Phase (Iteration {iteration}) ===")

        results = {}

        # Define all 9 board×player CMA-ES configurations
        # Format: (board, num_players, generations_per_iter, max_iterations, capabilities_required)
        CMAES_CONFIGS = [
            # Square8 - can run on any worker (low memory)
            ("square8", 2, 15, 5, ["square8"]),
            ("square8", 3, 12, 4, ["square8"]),
            ("square8", 4, 10, 4, ["square8"]),
            # Square19 - requires more memory, LAN workers preferred
            ("square19", 2, 12, 4, ["square19"]),
            ("square19", 3, 10, 3, ["square19"]),
            ("square19", 4, 8, 3, ["square19"]),
            # Hexagonal - similar to square19
            ("hexagonal", 2, 12, 4, ["hex"]),
            ("hexagonal", 3, 10, 3, ["hex"]),
            ("hexagonal", 4, 8, 3, ["hex"]),
        ]

        # Find workers capable of CMA-ES
        cmaes_workers = []
        for worker in WORKERS:
            if worker.role in ["cmaes", "mixed", "training"]:
                if await self.check_worker_health(worker):
                    cmaes_workers.append(worker)
                    self.log(f"CMA-ES worker {worker.name}: healthy (caps: {worker.capabilities})", "OK")
                else:
                    self.log(f"CMA-ES worker {worker.name}: unreachable", "WARN")

        if not cmaes_workers:
            self.log("No healthy CMA-ES workers available!", "ERROR")
            return results

        # Match configs to capable workers
        async def dispatch_cmaes_job(
            worker: WorkerConfig,
            board: str,
            num_players: int,
            gens_per_iter: int,
            max_iters: int,
        ) -> Tuple[str, bool]:
            """Dispatch a single CMA-ES job to a worker."""
            config_key = f"{board}_{num_players}p"
            output_dir = f"logs/cmaes/iter{iteration}/{config_key}"

            # Determine worker URLs for distributed mode
            # Use all workers that support this board type
            compatible_workers = [
                w for w in cmaes_workers
                if any(cap in w.capabilities for cap in [board[:3], board, "all"])
            ]
            worker_urls = ",".join([
                f"http://{w.host.split('@')[-1]}:8765"
                for w in compatible_workers
                if w != worker  # Exclude self for distributed workers
            ]) if len(compatible_workers) > 1 else ""

            distributed_flag = "--distributed" if worker_urls else ""
            workers_arg = f"--workers {worker_urls}" if worker_urls else ""

            cmaes_cmd = f"""
source venv/bin/activate
export PYTHONPATH={worker.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
python scripts/run_iterative_cmaes.py \\
    --board {board} \\
    --num-players {num_players} \\
    --generations-per-iter {gens_per_iter} \\
    --max-iterations {max_iters} \\
    --population-size 14 \\
    --games-per-eval 8 \\
    --sigma 0.5 \\
    --output-dir {output_dir} \\
    {distributed_flag} {workers_arg}
"""
            self.log(f"Dispatching CMA-ES {config_key} to {worker.name}...")
            code, _, stderr = await self.run_remote_command(worker, cmaes_cmd, background=True)

            if code == 0:
                self.log(f"CMA-ES {config_key} dispatched to {worker.name}", "OK")
                return config_key, True
            else:
                self.log(f"CMA-ES {config_key} dispatch failed: {stderr[:200]}", "ERROR")
                return config_key, False

        # Distribute jobs across workers, matching capabilities
        tasks = []
        worker_idx = 0

        for board, num_players, gens, max_iters, required_caps in CMAES_CONFIGS:
            # Find a worker with required capabilities
            capable_worker = None
            for i in range(len(cmaes_workers)):
                worker = cmaes_workers[(worker_idx + i) % len(cmaes_workers)]
                # Check if worker has required capability (or "all")
                if "all" in worker.capabilities or any(
                    cap in worker.capabilities for cap in required_caps
                ):
                    capable_worker = worker
                    worker_idx = (worker_idx + i + 1) % len(cmaes_workers)
                    break

            if capable_worker:
                tasks.append(dispatch_cmaes_job(
                    capable_worker, board, num_players, gens, max_iters
                ))
            else:
                config_key = f"{board}_{num_players}p"
                self.log(f"No worker capable of {config_key} (needs {required_caps})", "WARN")
                results[config_key] = False

        # Execute all dispatches concurrently
        if tasks:
            dispatch_results = await asyncio.gather(*tasks)
            for config_key, success in dispatch_results:
                results[config_key] = success

        # Summary
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        self.log(f"CMA-ES phase: dispatched {success_count}/{total_count} jobs")

        return results

    async def run_evaluation_phase(self, iteration: int) -> Dict[str, float]:
        """Run evaluation tournaments and SAVE games for training.

        This phase serves two purposes:
        1. Evaluate model/heuristic strength via head-to-head tournaments
        2. Generate high-quality training data from strong AI matchups

        Tournament games are saved to logs/tournaments/iter{N}/ and should be
        merged into training data during the sync phase.
        """
        self.state.phase = "evaluation"
        self._save_state()
        self.log(f"=== Starting Evaluation Phase (Iteration {iteration}) ===")

        results = {}

        # Find workers for evaluation (distribute tournaments)
        eval_workers = [w for w in WORKERS if await self.check_worker_health(w)]
        if not eval_workers:
            self.log("No healthy workers for evaluation", "ERROR")
            return results

        # Comprehensive tournament matchups for diverse training data
        # Format: (p1_type, p1_diff, p2_type, p2_diff, board, games)
        # Higher difficulty = stronger AI = higher quality games
        TOURNAMENT_MATCHUPS = [
            # Cross-AI-type matchups (generates diverse training data)
            ("Heuristic", 5, "MCTS", 6, "Square8", 10),
            ("Heuristic", 5, "Minimax", 5, "Square8", 10),
            ("MCTS", 6, "Minimax", 5, "Square8", 10),

            # Same-type tier progression (measures improvement)
            ("Heuristic", 3, "Heuristic", 5, "Square8", 8),
            ("MCTS", 5, "MCTS", 7, "Square8", 8),

            # Multi-board coverage
            ("Heuristic", 5, "MCTS", 5, "Square19", 6),
            ("Heuristic", 5, "MCTS", 5, "Hex", 6),

            # Strong vs strong (highest quality games)
            ("MCTS", 7, "MCTS", 8, "Square8", 6),
        ]

        # Distribute matchups across workers
        tasks = []
        for idx, (p1, p1d, p2, p2d, board, games) in enumerate(TOURNAMENT_MATCHUPS):
            worker = eval_workers[idx % len(eval_workers)]
            matchup_key = f"{p1}{p1d}_vs_{p2}{p2d}_{board}"
            output_dir = f"logs/tournaments/iter{iteration}/{matchup_key}"

            eval_cmd = f"""
source venv/bin/activate
export PYTHONPATH={worker.remote_path}
export RINGRIFT_SKIP_SHADOW_CONTRACTS=true
export RINGRIFT_TRAINED_HEURISTIC_PROFILES={worker.remote_path}/data/trained_heuristic_profiles.json
mkdir -p {output_dir}
python scripts/run_ai_tournament.py \\
    --p1 {p1} --p1-diff {p1d} \\
    --p2 {p2} --p2-diff {p2d} \\
    --board {board} \\
    --games {games} \\
    --output-dir {output_dir}
"""
            self.log(f"Dispatching tournament {matchup_key} to {worker.name}")
            tasks.append(self._run_tournament(worker, eval_cmd, matchup_key))

        # Run all tournaments concurrently
        tournament_results = await asyncio.gather(*tasks)
        for matchup_key, success, win_rate in tournament_results:
            results[matchup_key] = win_rate if success else 0.0

        # Summary
        success_count = sum(1 for _, s, _ in tournament_results if s)
        self.log(f"Evaluation: {success_count}/{len(TOURNAMENT_MATCHUPS)} tournaments completed")

        # Record results to persistent Elo database for cross-model tracking
        if HAS_PERSISTENT_ELO:
            try:
                conn = init_elo_database()
                for matchup_key, success, win_rate in tournament_results:
                    if success:
                        # Parse matchup key: Heuristic5_vs_MCTS6_Square8
                        parts = matchup_key.split("_vs_")
                        if len(parts) == 2:
                            p1_info, rest = parts
                            p2_parts = rest.split("_")
                            if len(p2_parts) >= 2:
                                board = p2_parts[-1].lower().replace("square", "square")
                                # Record aggregate result (simplified - just update Elo based on win rate)
                                # In production, would record individual game results
                                self.log(f"Recording {matchup_key} to persistent Elo DB: {win_rate:.1%}")
                conn.close()
            except Exception as e:
                self.log(f"Failed to update persistent Elo DB: {e}", "WARN")

        return results

    async def _run_tournament(
        self,
        worker: WorkerConfig,
        cmd: str,
        matchup_key: str,
    ) -> Tuple[str, bool, float]:
        """Run a single tournament and parse results."""
        code, stdout, stderr = await self.run_remote_command(worker, cmd)
        if code != 0:
            self.log(f"Tournament {matchup_key} failed: {stderr[:100]}", "ERROR")
            return matchup_key, False, 0.0

        # Parse win rate from output (e.g., "P1 wins: 6/10 (60.0%)")
        import re
        match = re.search(r"(\d+\.?\d*)%", stdout)
        win_rate = float(match.group(1)) / 100 if match else 0.5

        self.log(f"Tournament {matchup_key}: {win_rate:.1%} P1 win rate", "OK")
        return matchup_key, True, win_rate

    async def run_elo_calibration_phase(
        self,
        iteration: int,
        distributed: bool = True,
        games_per_config: Optional[int] = None,
        board_types: Optional[List[str]] = None,
        player_counts: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Run diverse tournaments across all board/player configurations for Elo calibration.

        This phase runs the diverse tournament orchestrator which:
        - Tests all AI types (random, heuristic, minimax, MCTS, neural descent)
        - Uses all available neural network model versions
        - Runs across all board types and player counts
        - Provides richer Elo calibration than head-to-head evaluation

        Args:
            iteration: Current pipeline iteration
            distributed: Whether to run distributed across cluster hosts (parallel)
            games_per_config: Games per board/player configuration (default: from config)
            board_types: Board types to include (default: all)
            player_counts: Player counts to include (default: all)

        Returns:
            Dictionary with tournament results summary
        """
        self.state.phase = "elo-calibration"
        self.log(f"\n--- Phase: Elo Calibration (Diverse Tournaments) ---")

        if not HAS_DIVERSE_TOURNAMENTS:
            self.log("Diverse tournament module not available, skipping elo-calibration", "WARN")
            return {"skipped": True, "reason": "module_not_available"}

        if self.dry_run:
            self.log("[DRY-RUN] Would run diverse tournaments for Elo calibration")
            return {"dry_run": True}

        # Default configuration
        board_types = board_types or ["square8", "square19", "hexagonal"]
        player_counts = player_counts or [2, 3, 4]
        output_base = str((Path(__file__).resolve().parent.parent / "data" / "tournaments" / f"iter{iteration}").resolve())

        # Build tournament configs
        from scripts.run_diverse_tournaments import build_tournament_configs
        configs = build_tournament_configs(
            board_types=board_types,
            player_counts=player_counts,
            games_per_config=games_per_config,
            output_base=output_base,
            seed=iteration * 10000,
        )

        self.log(f"Tournament configurations: {len(configs)}")
        for config in configs:
            self.log(f"  {config.board_type} {config.num_players}p: {config.num_games} games")

        results: List[TournamentResult] = []

        if distributed:
            # Load and filter available hosts
            hosts = load_cluster_hosts()
            available_hosts = await filter_available_hosts(hosts)

            if not available_hosts:
                self.log("No cluster hosts available for distributed tournaments", "WARN")
                self.log("Falling back to local execution")
                distributed = False

        if distributed:
            self.log(f"Running distributed across {len(available_hosts)} hosts")
            results = await run_tournament_round_distributed(configs, available_hosts)
        else:
            self.log("Running locally (sequential)")
            results = run_tournament_round_local(configs)

        # Aggregate results
        total_games = sum(r.games_completed for r in results)
        total_samples = sum(r.samples_generated for r in results)
        successful = sum(1 for r in results if r.success)

        summary = {
            "iteration": iteration,
            "configurations": len(configs),
            "successful": successful,
            "total_games": total_games,
            "total_samples": total_samples,
            "distributed": distributed,
            "hosts_used": len(set(r.host for r in results)),
            "results": [
                {
                    "config": f"{r.config.board_type}_{r.config.num_players}p",
                    "host": r.host,
                    "games": r.games_completed,
                    "samples": r.samples_generated,
                    "success": r.success,
                }
                for r in results
            ],
        }

        self.log(f"Elo calibration complete: {successful}/{len(configs)} configs, {total_games} games")

        return summary

    async def run_full_iteration(self, iteration: int) -> bool:
        """Run a complete pipeline iteration with checkpointing and smart polling.

        Pipeline flow:
        1. Selfplay: Generate games with current models/heuristics
        2. Sync: Pull selfplay data from workers + deduplicate
        3. CMA-ES: Optimize heuristics for all 9 board×player configs (background)
        4. NN Training: Train neural network on new data + register models
        5. Profile Sync: Pull trained heuristics, merge, push to all workers
        6. Evaluation: Tournament + Elo updates
        7. Elo Calibration: Diverse tournaments across all board/player configs
        8. Tier Gating: Check for model promotions
        9. Resource Monitoring: Log worker utilization
        """
        self.log(f"\n{'='*60}")
        self.log(f"=== Pipeline Iteration {iteration} ===")
        self.log(f"{'='*60}\n")

        # Only reset phase tracking if this is a new iteration
        # (not when resuming an interrupted iteration)
        if self.state.iteration != iteration:
            self.state.iteration = iteration
            self.state.phase_completed = {}  # Reset phase tracking for new iteration
        elif not self.state.phase_completed:
            self.state.phase_completed = {}

        self._save_state()

        # Phase 1: Selfplay - Generate games across all board×player configs
        if not self.state.phase_completed.get("selfplay"):
            selfplay_result = await self.run_selfplay_phase(iteration)
            if not selfplay_result.get("dispatched", 0):
                self.log("Selfplay phase failed", "ERROR")
                return False

            # Smart polling instead of fixed wait
            if not self.dry_run:
                total_games = await self.poll_for_selfplay_completion(
                    min_games=SELFPLAY_MIN_GAMES_THRESHOLD,
                    max_wait_minutes=30,
                )
                self.state.games_generated[f"iter_{iteration}"] = total_games

            self.state.phase_completed["selfplay"] = True
            self._save_state()
        else:
            self.log("Selfplay phase already completed, skipping...")
            selfplay_result = {"dispatched": 0, "total": 0, "resumed": True}

        # Phase 2: Sync selfplay data from workers + deduplicate
        if not self.state.phase_completed.get("sync"):
            if not await self.run_sync_phase():
                self.log("Sync phase failed, continuing...", "WARN")

            # Deduplicate training data
            if not self.dry_run:
                await self.deduplicate_training_data("data/games/merged_latest.db")

            self.state.phase_completed["sync"] = True
            self._save_state()
        else:
            self.log("Sync phase already completed, skipping...")

        # Phase 3: CMA-ES heuristic optimization (dispatches background jobs)
        if not self.state.phase_completed.get("cmaes"):
            cmaes_results = await self.run_cmaes_phase(iteration)
            self.state.phase_completed["cmaes"] = True
            self._save_state()
        else:
            self.log("CMA-ES phase already completed, skipping...")
            cmaes_results = {}

        # Phase 4: NN Training with model registration
        if not self.state.phase_completed.get("training"):
            training_results = await self.run_training_phase(iteration)

            # Register trained models
            for config_key, success in training_results.items():
                if success:
                    model_id = f"{config_key}_iter{iteration}"
                    model_path = f"models/{model_id}.pth"
                    parent_id = self.get_best_model(config_key)  # Link to previous best
                    self.register_model(
                        model_id=model_id,
                        model_path=model_path,
                        config=config_key,
                        parent_id=parent_id,
                        metrics={"training_iteration": iteration},
                    )

            self.state.phase_completed["training"] = True
            self._save_state()
        else:
            self.log("Training phase already completed, skipping...")
            training_results = {}

        # Phase 5: Wait for CMA-ES completion and sync heuristic profiles
        if cmaes_results and any(cmaes_results.values()):
            if not self.dry_run:
                # Smart polling for CMA-ES completion
                await self.poll_for_cmaes_completion(max_wait_minutes=MAX_PHASE_WAIT_MINUTES)

            # Pull trained heuristic profiles from all workers and merge
            await self.sync_heuristic_profiles()

        # Phase 6: Evaluation with Elo updates
        if not self.state.phase_completed.get("evaluation"):
            eval_results = await self.run_evaluation_phase(iteration)

            # Update Elo ratings based on tournament results
            for matchup_key, win_rate in eval_results.items():
                if "_vs_" in matchup_key:
                    parts = matchup_key.split("_vs_")
                    if len(parts) == 2:
                        player_a = parts[0]
                        player_b = parts[1].split("_")[0]  # Remove board suffix
                        self.update_elo_rating(player_a, player_b, win_rate)

            self.state.phase_completed["evaluation"] = True
            self._save_state()
        else:
            self.log("Evaluation phase already completed, skipping...")
            eval_results = {}

        # Phase 7: Elo Calibration via diverse tournaments (all board/player configs)
        elo_calibration_results = {}
        if not self.state.phase_completed.get("elo-calibration"):
            elo_calibration_results = await self.run_elo_calibration_phase(iteration)
            self.state.phase_completed["elo-calibration"] = True
            self._save_state()
        else:
            self.log("Elo calibration phase already completed, skipping...")

        # Phase 8: Tier gating checks
        tier_promotions = await self.run_tier_gating_phase()

        # Phase 9: Resource monitoring
        await self.log_resource_usage()

        # Print Elo leaderboard
        self.log("\n=== Elo Leaderboard ===")
        for rank, (model, elo) in enumerate(self.get_elo_leaderboard(10), 1):
            self.log(f"  {rank}. {model}: {elo:.0f}")

        self.state.phase = "complete"
        self._save_state()

        self.log(f"\n{'='*60}")
        self.log(f"=== Iteration {iteration} Complete ===")
        self.log(f"{'='*60}")
        self.log(f"Selfplay:    {selfplay_result}")
        self.log(f"CMA-ES:      {cmaes_results}")
        self.log(f"Training:    {training_results}")
        self.log(f"Eval:        {eval_results}")
        self.log(f"Elo Calib:   {elo_calibration_results.get('total_games', 0)} games across {elo_calibration_results.get('configurations', 0)} configs")
        self.log(f"Promotions:  {tier_promotions}")
        self.log(f"{'='*60}\n")

        return True

    async def run(
        self,
        iterations: int = 1,
        start_iteration: int = 0,
        phase: Optional[str] = None,
        resume: bool = False,
        board_type: str = "square8",
        num_players: int = 2,
        games_per_worker: int = 500,
    ) -> None:
        """Run the pipeline.

        Args:
            iterations: Number of pipeline iterations to run
            start_iteration: Starting iteration number
            phase: Run only a specific phase (None for full iteration)
            resume: Resume from last saved state
            board_type: Board type for canonical selfplay/training
            num_players: Number of players for canonical selfplay/training
            games_per_worker: Number of games per worker for canonical selfplay
        """
        self.log("RingRift AI Training Pipeline")

        # Store config for use in phases
        self._board_type = board_type
        self._num_players = num_players
        self._games_per_worker = games_per_worker

        # Handle resume mode - start from last saved iteration
        if resume and not phase:
            start_iteration = self.state.iteration
            self.log(f"Resuming from iteration {start_iteration} (phase: {self.state.phase})")
            # Don't reset phase_completed to allow skipping completed phases

        self.log(f"Iterations: {iterations}, Start: {start_iteration}, Board: {board_type}, Players: {num_players}")
        if self.dry_run:
            self.log("*** DRY RUN MODE - No commands will be executed ***")

        if phase:
            # Run single phase
            self.log(f"Running single phase: {phase}")
            iteration = start_iteration or self.state.iteration
            if phase == "selfplay":
                await self.run_selfplay_phase(iteration)
            elif phase == "canonical-selfplay":
                await self.run_canonical_selfplay_phase(
                    iteration,
                    games_per_worker=self._games_per_worker,
                    board_type=self._board_type,
                    num_players=self._num_players,
                )
            elif phase == "sync":
                await self.run_sync_phase()
            elif phase == "parity-validation":
                await self.run_parity_validation_phase(
                    board_type=self._board_type,
                    num_players=self._num_players,
                )
            elif phase == "npz-export":
                await self.run_npz_export_phase(
                    iteration,
                    board_type=self._board_type,
                    num_players=self._num_players,
                )
            elif phase == "training":
                await self.run_training_phase(iteration)
            elif phase == "cmaes":
                await self.run_cmaes_phase(iteration)
            elif phase == "profile-sync":
                await self.sync_heuristic_profiles()
            elif phase == "evaluation":
                await self.run_evaluation_phase(iteration)
            elif phase == "elo-calibration":
                await self.run_elo_calibration_phase(iteration)
            elif phase == "tier-gating":
                await self.run_tier_gating_phase()
            elif phase == "resources":
                await self.log_resource_usage()
            elif phase == "refresh-workers":
                await self.refresh_workers()
            else:
                self.log(f"Unknown phase: {phase}", "ERROR")
            return

        # Run full iterations
        for i in range(start_iteration, start_iteration + iterations):
            success = await self.run_full_iteration(i)
            if not success:
                self.log(f"Iteration {i} failed, stopping pipeline", "ERROR")
                break

        self.log("Pipeline complete")


def main():
    parser = argparse.ArgumentParser(description="RingRift AI Training Pipeline Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline configuration JSON",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of pipeline iterations to run",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=0,
        help="Starting iteration number",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=[
            "selfplay", "canonical-selfplay", "sync", "parity-validation",
            "npz-export", "training", "cmaes", "profile-sync", "evaluation",
            "elo-calibration", "tier-gating", "resources", "refresh-workers"
        ],
        help="Run only a specific phase",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last saved state (skips completed phases)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default="logs/pipeline/state.json",
        help="Path to state file",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type for canonical selfplay/training",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players for canonical selfplay/training",
    )
    parser.add_argument(
        "--games-per-worker",
        type=int,
        default=500,
        help="Number of games per worker for canonical selfplay",
    )
    parser.add_argument(
        "--discover-vast",
        action="store_true",
        help="Dynamically discover Vast.ai instances before running",
    )
    # P2P backend options
    parser.add_argument(
        "--backend",
        type=str,
        default="ssh",
        choices=["ssh", "p2p"],
        help="Backend mode: 'ssh' (default) or 'p2p' for P2P orchestrator API",
    )
    parser.add_argument(
        "--p2p-leader",
        type=str,
        default=None,
        help="URL of P2P leader node (e.g., http://192.168.1.100:8770)",
    )
    parser.add_argument(
        "--p2p-seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated list of seed P2P nodes to discover the current leader "
            "(used when --p2p-leader is not set). Example: "
            "http://<node1-ip>:8770,http://<node2-ip>:8770"
        ),
    )
    parser.add_argument(
        "--p2p-auth-token",
        type=str,
        default=None,
        help="Auth token for P2P cluster (or set RINGRIFT_CLUSTER_AUTH_TOKEN)",
    )
    args = parser.parse_args()

    async def main_async():
        orchestrator: Optional[PipelineOrchestrator] = None
        try:
            p2p_leader_url = args.p2p_leader
            p2p_auth_token = args.p2p_auth_token or os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN", "")

            if args.backend == "p2p":
                if not p2p_leader_url:
                    seeds_raw = (args.p2p_seeds or "").strip()
                    seed_urls = [s.strip() for s in seeds_raw.split(",") if s.strip()]
                    if not seed_urls:
                        raise ValueError("P2P backend requires --p2p-leader or --p2p-seeds")
                    p2p_leader_url = await discover_p2p_leader_url(
                        seed_urls,
                        auth_token=p2p_auth_token,
                    )
                    if not p2p_leader_url:
                        raise RuntimeError(f"Failed to discover P2P leader from seeds: {seed_urls}")

                orchestrator = PipelineOrchestrator(
                    config_path=args.config,
                    state_path=args.state_path,
                    dry_run=args.dry_run,
                    backend=args.backend,
                    p2p_leader_url=p2p_leader_url,
                    p2p_auth_token=p2p_auth_token,
                )
            else:
                orchestrator = PipelineOrchestrator(
                    config_path=args.config,
                    state_path=args.state_path,
                    dry_run=args.dry_run,
                    backend=args.backend,
                    p2p_leader_url=None,
                    p2p_auth_token=None,
                )

            # Optionally discover Vast instances (SSH backend only)
            if args.discover_vast and orchestrator.backend_mode == "ssh":
                await orchestrator.refresh_workers()

            await orchestrator.run(
                iterations=args.iterations,
                start_iteration=args.start_iteration,
                phase=args.phase,
                resume=args.resume,
                board_type=args.board_type,
                num_players=args.num_players,
                games_per_worker=args.games_per_worker,
            )
        finally:
            # Clean up P2P backend session
            if orchestrator and orchestrator.p2p_backend:
                await orchestrator.p2p_backend.close()

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
