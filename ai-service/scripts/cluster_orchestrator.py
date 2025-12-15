#!/usr/bin/env python3
"""Enhanced Cluster Orchestrator - Manages all compute resources with resource-aware scheduling.

Features:
- Monitors all hosts: 2 Lambda, 2 AWS, 3 Vast.ai, 3 local Macs
- Resource-aware job scheduling (CPU, GPU, RAM, disk)
- Adaptive task allocation based on current utilization
- Automatic job restart on crash
- Consolidated logging and status dashboard

Usage:
    python scripts/cluster_orchestrator.py
    python scripts/cluster_orchestrator.py --dry-run
    python scripts/cluster_orchestrator.py --status-only
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import unified cluster coordination
try:
    from app.coordination import (
        acquire_orchestrator_lock,
        release_orchestrator_lock,
        can_spawn_task,
        register_task,
        check_emergency_halt,
        cleanup_stale_tasks,
        MAX_LOAD_THRESHOLD,
    )
    HAS_COORDINATION = True
except ImportError:
    HAS_COORDINATION = False
    print("[WARN] Cluster coordination module not available, using local lock only")

LOG_DIR = Path(__file__).parent.parent / "logs" / "orchestrator"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = LOG_DIR / "cluster_state.json"
LOCKFILE = Path("/tmp/cluster_orchestrator.lock")

# Mac Studio sync configuration
MAC_STUDIO_HOST = os.environ.get("MAC_STUDIO_HOST", "mac-studio")
MAC_STUDIO_DATA_DIR = "~/Development/RingRift/ai-service/data/games"
SYNC_INTERVAL = 6  # Sync every 6 iterations (30 minutes at 5-min interval)

# Model sync configuration - sync NN/NNUE models across cluster
MODEL_SYNC_INTERVAL = 12  # Sync models every 12 iterations (1 hour at 5-min interval)
MODEL_SYNC_ENABLED = True  # Enable automatic model syncing

# Elo calibration configuration
ELO_CALIBRATION_INTERVAL = 72  # Run Elo tournament every 72 iterations (6 hours at 5-min interval)
ELO_CALIBRATION_GAMES = 50  # Games per config for Elo calibration

# Elo-driven scheduling configuration (curriculum learning)
ELO_LEADERBOARD_DB = Path(__file__).parent.parent / "data" / "unified_elo.db"
ELO_CURRICULUM_ENABLED = True  # Enable Elo-based opponent selection
ELO_MATCH_WINDOW = 200  # Match opponents within this Elo range
ELO_UNDERSERVED_THRESHOLD = 100  # Configs with fewer games are "underserved"

# Auto-scaling configuration
AUTO_SCALE_INTERVAL = 12  # Check for underutilized hosts every 12 iterations (1 hour at 5-min interval)
UNDERUTILIZED_CPU_THRESHOLD = 30  # CPU usage below this % is considered underutilized
UNDERUTILIZED_PYTHON_JOBS = 10  # Fewer than this many python jobs is considered underutilized
SCALE_UP_GAMES_PER_HOST = 50  # Number of games to start on underutilized host

# GPU utilization targeting for better resource efficiency
TARGET_GPU_UTILIZATION_MIN = 60  # Start more jobs if GPU below this %
TARGET_GPU_UTILIZATION_MAX = 90  # Don't start more jobs if GPU above this %
TARGET_CPU_UTILIZATION_MIN = 60  # Start more jobs if CPU below this %
TARGET_CPU_UTILIZATION_MAX = 85  # Don't start more jobs if CPU above this %
GH200_MIN_SELFPLAY_JOBS = 20  # Higher baseline for GH200 hosts
GH200_MAX_SELFPLAY_JOBS = 100  # Upper limit for GH200 hosts
MIN_MEMORY_GB_FOR_TASKS = 64  # Skip nodes with less than 64GB RAM to avoid OOM

# Tournament scheduling configuration
TOURNAMENT_INTERVAL = 6  # Run tournaments every 6 iterations (30 minutes at 5-min interval)
TOURNAMENT_GAMES_PER_MATCHUP = 20  # Games between each pair
TOURNAMENT_CONFIGS = [
    # All board types and player counts for comprehensive coverage
    {"board_type": "square8", "num_players": 2},
    {"board_type": "square8", "num_players": 3},
    {"board_type": "square8", "num_players": 4},
    {"board_type": "square19", "num_players": 2},
    {"board_type": "square19", "num_players": 3},
    {"board_type": "square19", "num_players": 4},
    {"board_type": "hexagonal", "num_players": 2},
    {"board_type": "hexagonal", "num_players": 3},
    {"board_type": "hexagonal", "num_players": 4},
]

# AI types to include in tournaments (baseline calibration)
TOURNAMENT_AI_TYPES = [
    {"ai_type": "random", "difficulty": 1},
    {"ai_type": "heuristic", "difficulty": 5},
    {"ai_type": "mcts", "difficulty": 7, "mcts_simulations": 100},
    {"ai_type": "mcts", "difficulty": 8, "mcts_simulations": 500},
    {"ai_type": "neural_net", "difficulty": 10},  # All active NN models
]


@dataclass
class HostConfig:
    """Configuration for a compute host."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    ssh_key: Optional[str] = None
    ringrift_path: str = "~/ringrift/ai-service"
    venv_activate: str = "source venv/bin/activate"
    memory_gb: int = 16
    cpus: int = 4
    has_gpu: bool = False
    gpu_name: str = ""
    role: str = "selfplay"  # selfplay, training, cmaes, all
    storage_type: str = "disk"  # disk or ram (vast.ai /dev/shm)
    min_selfplay_jobs: int = 2
    enabled: bool = True


@dataclass
class HostStatus:
    """Runtime status of a host."""
    name: str
    reachable: bool = False
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    python_jobs: int = 0
    selfplay_jobs: int = 0
    training_jobs: int = 0
    cmaes_jobs: int = 0
    elo_tournament_jobs: int = 0
    last_check: str = ""
    error: str = ""


@dataclass
class ClusterState:
    """Persistent cluster state for resumability."""
    iteration: int = 0
    total_jobs_started: int = 0
    total_restarts: int = 0
    host_statuses: Dict[str, dict] = field(default_factory=dict)
    last_sync: str = ""
    last_elo_calibration: str = ""
    last_model_sync: str = ""  # Last model sync across cluster
    errors: List[str] = field(default_factory=list)


# ============================================
# Priority-Based Job Scheduling
# ============================================
from enum import IntEnum


class JobPriority(IntEnum):
    """Priority levels for job scheduling.

    Lower values = higher priority.
    Critical jobs (promotion evaluation) always run first.
    """
    CRITICAL = 0   # Promotion evaluation, regression tests
    HIGH = 1       # Shadow tournaments, Elo calibration
    NORMAL = 2     # Regular selfplay, training
    LOW = 3        # Backfill, optional data collection


@dataclass
class ScheduledJob:
    """A job to be scheduled on the cluster."""
    job_type: str  # selfplay, tournament, training, promotion, etc.
    priority: JobPriority
    config: Dict[str, Any] = field(default_factory=dict)
    host_preference: Optional[str] = None  # Preferred host name or None
    requires_gpu: bool = False
    estimated_duration_seconds: int = 3600
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __lt__(self, other):
        """Enable sorting by priority."""
        return self.priority < other.priority


class PriorityJobScheduler:
    """Priority-based job scheduler for the unified AI improvement loop.

    Ensures critical jobs (promotion evaluation, regression tests) run
    first, while regular selfplay jobs fill remaining capacity.

    Usage:
        scheduler = PriorityJobScheduler()

        # Queue jobs with different priorities
        scheduler.schedule(ScheduledJob(
            job_type="promotion_evaluation",
            priority=JobPriority.CRITICAL,
            config={"model_id": "latest"},
            requires_gpu=True,
        ))
        scheduler.schedule(ScheduledJob(
            job_type="selfplay",
            priority=JobPriority.NORMAL,
            config={"board": "square8", "players": 2},
        ))

        # Get next job for a host
        host_status = check_host_status(host)
        next_job = scheduler.next_job([host], [host_status])
    """

    def __init__(self, max_queue_size: int = 1000):
        self._queue: List[ScheduledJob] = []
        self._running: Dict[str, ScheduledJob] = {}  # host_name -> job
        self._max_queue_size = max_queue_size

    def schedule(self, job: ScheduledJob) -> bool:
        """Add a job to the scheduling queue.

        Returns True if job was added, False if queue is full.
        """
        if len(self._queue) >= self._max_queue_size:
            log(f"Job queue full ({self._max_queue_size}), rejecting {job.job_type}", "WARN")
            return False

        self._queue.append(job)
        self._queue.sort()  # Sort by priority
        return True

    def next_job(
        self,
        hosts: List[HostConfig],
        statuses: List[HostStatus],
    ) -> Optional[Tuple[ScheduledJob, HostConfig]]:
        """Get the next job to run and the host to run it on.

        Matches jobs to hosts based on requirements (GPU, capacity).
        Returns (job, host) tuple or None if no suitable match.
        """
        if not self._queue:
            return None

        # Build host availability map
        available_hosts: List[Tuple[HostConfig, HostStatus]] = []
        for host, status in zip(hosts, statuses):
            if not status.reachable:
                continue
            if status.disk_percent > 90:
                continue
            if status.memory_percent > 90:
                continue
            # Skip low-memory hosts to avoid OOM
            if host.memory_gb > 0 and host.memory_gb < MIN_MEMORY_GB_FOR_TASKS:
                continue
            available_hosts.append((host, status))

        if not available_hosts:
            return None

        # Find best match for highest priority job
        for job_idx, job in enumerate(self._queue):
            for host, status in available_hosts:
                # Check GPU requirement
                if job.requires_gpu and not host.has_gpu:
                    continue

                # Check host preference
                if job.host_preference and host.name != job.host_preference:
                    continue

                # Check CPU capacity
                if job.job_type == "selfplay" and status.cpu_percent > TARGET_CPU_UTILIZATION_MAX:
                    continue

                # Found a match
                self._queue.pop(job_idx)
                job.started_at = time.time()
                self._running[host.name] = job
                return (job, host)

        return None

    def complete_job(self, host_name: str) -> Optional[ScheduledJob]:
        """Mark a job as completed for a host."""
        if host_name in self._running:
            job = self._running.pop(host_name)
            job.completed_at = time.time()
            return job
        return None

    def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics about queued jobs by priority."""
        stats = {
            "total": len(self._queue),
            "running": len(self._running),
            "critical": 0,
            "high": 0,
            "normal": 0,
            "low": 0,
        }
        for job in self._queue:
            if job.priority == JobPriority.CRITICAL:
                stats["critical"] += 1
            elif job.priority == JobPriority.HIGH:
                stats["high"] += 1
            elif job.priority == JobPriority.NORMAL:
                stats["normal"] += 1
            else:
                stats["low"] += 1
        return stats

    def has_critical_pending(self) -> bool:
        """Check if any critical priority jobs are pending."""
        return any(j.priority == JobPriority.CRITICAL for j in self._queue)

    def reserve_capacity_for_training(
        self,
        hosts: List[HostConfig],
        statuses: List[HostStatus],
        reserve_percent: float = 20.0,
    ) -> List[str]:
        """Reserve GPU capacity for training on GPU hosts.

        Returns list of host names where capacity was reserved.
        """
        reserved = []
        for host, status in zip(hosts, statuses):
            if not host.has_gpu:
                continue
            if not status.reachable:
                continue

            # Reserve by not scheduling selfplay jobs on this host
            # if GPU utilization is already in acceptable range for training
            if status.gpu_percent < reserve_percent:
                reserved.append(host.name)

        return reserved


# Global scheduler instance
_scheduler: Optional[PriorityJobScheduler] = None


def get_scheduler() -> PriorityJobScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = PriorityJobScheduler()
    return _scheduler


def log(msg: str, level: str = "INFO"):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "orchestrator.log", "a") as f:
        f.write(line + "\n")


def load_hosts_config() -> List[HostConfig]:
    """Load host configurations from YAML files."""
    hosts = []
    config_dir = Path(__file__).parent.parent / "config"

    # Load distributed_hosts.yaml
    distributed_file = config_dir / "distributed_hosts.yaml"
    if distributed_file.exists():
        with open(distributed_file) as f:
            data = yaml.safe_load(f) or {}

        for name, cfg in data.get("hosts", {}).items():
            if cfg.get("status") == "ssh_key_issue":
                continue  # Skip hosts with known issues
            # Ensure path includes ai-service
            ringrift_path = cfg.get("ringrift_path", "~/ringrift/ai-service")
            if not ringrift_path.endswith("ai-service"):
                ringrift_path = ringrift_path + "/ai-service"
            hosts.append(HostConfig(
                name=name,
                ssh_host=cfg.get("ssh_host", ""),
                ssh_user=cfg.get("ssh_user", "ubuntu"),
                ssh_port=cfg.get("ssh_port", 22),
                ssh_key=cfg.get("ssh_key"),
                ringrift_path=ringrift_path,
                venv_activate=cfg.get("venv_activate") or "source venv/bin/activate",
                memory_gb=cfg.get("memory_gb", 16),
                cpus=cfg.get("cpus", 4),
                has_gpu="gpu" in cfg,
                gpu_name=cfg.get("gpu", ""),
                role=cfg.get("role", "selfplay"),
                min_selfplay_jobs=max(1, cfg.get("memory_gb", 16) // 32),  # 1 job per 32GB
            ))

    # Load remote_hosts.yaml for additional config
    remote_file = config_dir / "remote_hosts.yaml"
    if remote_file.exists():
        with open(remote_file) as f:
            data = yaml.safe_load(f) or {}

        # Add standard hosts
        for name, cfg in data.get("standard_hosts", {}).items():
            # Check if already added
            if any(h.name == name for h in hosts):
                continue
            hosts.append(HostConfig(
                name=name,
                ssh_host=cfg.get("ssh_host", ""),
                ssh_key=cfg.get("ssh_key"),
                ringrift_path=cfg.get("remote_path", "~/ringrift/ai-service").replace("/data/games", ""),
                memory_gb=cfg.get("memory_gb", 16),
                cpus=cfg.get("cpus", 4),
                has_gpu=cfg.get("has_gpu", False),
                role=cfg.get("role", "selfplay"),
                min_selfplay_jobs=max(1, cfg.get("memory_gb", 16) // 32),
            ))

        # Add vast hosts
        for name, cfg in data.get("vast_hosts", {}).items():
            hosts.append(HostConfig(
                name=name,
                ssh_host=cfg.get("host", ""),
                ssh_user=cfg.get("user", "root"),
                ssh_port=cfg.get("port", 22),
                ringrift_path="~/ringrift/ai-service",
                memory_gb=cfg.get("memory_gb", 16),
                cpus=cfg.get("cpus", 4),
                has_gpu=cfg.get("has_gpu", False),
                role=cfg.get("role", "selfplay"),
                storage_type=cfg.get("storage_type", "disk"),
                min_selfplay_jobs=max(2, cfg.get("cpus", 4) // 48),  # 1 job per 48 CPUs for vast
            ))

    return hosts


def ssh_cmd(host: HostConfig, cmd: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Execute SSH command on host."""
    ssh_args = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
    ]

    if host.ssh_port != 22:
        ssh_args.extend(["-p", str(host.ssh_port)])

    if host.ssh_key:
        key_path = os.path.expanduser(host.ssh_key)
        if os.path.exists(key_path):
            ssh_args.extend(["-i", key_path])

    ssh_args.append(f"{host.ssh_user}@{host.ssh_host}")
    ssh_args.append(cmd)

    try:
        result = subprocess.run(
            ssh_args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "SSH timeout"
    except Exception as e:
        return 1, "", str(e)


def check_host_status(host: HostConfig) -> HostStatus:
    """Get comprehensive status of a host."""
    status = HostStatus(name=host.name, last_check=datetime.now().isoformat())

    # Check reachability
    code, out, err = ssh_cmd(host, "echo ok", timeout=15)
    if code != 0:
        status.error = err or "Connection failed"
        return status

    status.reachable = True

    # Get resource usage with single SSH command
    metrics_cmd = """
    # CPU usage
    cpu=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' 2>/dev/null || echo "0")

    # Memory usage
    mem=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}' 2>/dev/null || echo "0")

    # Disk usage
    disk=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%' 2>/dev/null || echo "0")

    # GPU usage (if nvidia-smi available)
    if command -v nvidia-smi &> /dev/null; then
        gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
        gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F', ' '{printf "%.1f", $1/$2*100}' || echo "0")
    else
        gpu="0"
        gpu_mem="0"
    fi

    # Job counts
    selfplay=$(pgrep -f "selfplay|run_hybrid" | wc -l)
    training=$(pgrep -f "train_nnue|train.py" | wc -l)
    cmaes=$(pgrep -f "cmaes" | wc -l)
    elo_tournament=$(pgrep -f "run_model_elo_tournament" | wc -l)
    python_total=$(pgrep -f "python" | wc -l)

    echo "$cpu|$mem|$disk|$gpu|$gpu_mem|$selfplay|$training|$cmaes|$elo_tournament|$python_total"
    """

    code, out, err = ssh_cmd(host, metrics_cmd, timeout=30)
    if code == 0 and out:
        try:
            parts = out.strip().split("|")
            if len(parts) >= 10:
                status.cpu_percent = float(parts[0] or 0)
                status.memory_percent = float(parts[1] or 0)
                status.disk_percent = float(parts[2] or 0)
                status.gpu_percent = float(parts[3] or 0)
                status.gpu_memory_percent = float(parts[4] or 0)
                status.selfplay_jobs = int(parts[5] or 0)
                status.training_jobs = int(parts[6] or 0)
                status.cmaes_jobs = int(parts[7] or 0)
                status.elo_tournament_jobs = int(parts[8] or 0)
                status.python_jobs = int(parts[9] or 0)
        except (ValueError, IndexError) as e:
            status.error = f"Parse error: {e}"

    return status


def should_start_selfplay(host: HostConfig, status: HostStatus) -> int:
    """Determine how many selfplay jobs to start based on resources.

    For GH200 hosts, targets 60-90% GPU utilization by scaling up jobs
    when utilization is below target.
    """
    if not status.reachable:
        return 0

    # Check resource constraints
    if status.disk_percent > 90:
        log(f"{host.name}: Disk usage critical ({status.disk_percent}%), skipping selfplay", "WARN")
        return 0

    if status.memory_percent > 85:
        log(f"{host.name}: Memory usage high ({status.memory_percent}%), reducing jobs", "WARN")
        return max(0, host.min_selfplay_jobs - status.selfplay_jobs - 1)

    # Determine if this is a GH200 host
    is_gh200 = "gh200" in host.name.lower() or "GH200" in host.gpu_name

    if is_gh200:
        # Use higher baseline and utilization targeting for GH200s
        min_jobs = GH200_MIN_SELFPLAY_JOBS
        max_jobs = GH200_MAX_SELFPLAY_JOBS

        # Scale based on GPU and CPU utilization
        if status.gpu_percent < TARGET_GPU_UTILIZATION_MIN and status.cpu_percent < TARGET_CPU_UTILIZATION_MAX:
            # GPU underutilized - scale up more aggressively
            utilization_gap = TARGET_GPU_UTILIZATION_MIN - status.gpu_percent
            scale_factor = max(1, int(utilization_gap / 10))  # Add more jobs for bigger gaps
            jobs_needed = min(scale_factor * 5, max_jobs - status.selfplay_jobs)
            if jobs_needed > 0:
                log(f"{host.name}: GPU at {status.gpu_percent:.0f}%, scaling up by {jobs_needed} jobs")
        elif status.gpu_percent >= TARGET_GPU_UTILIZATION_MAX:
            # GPU well utilized - don't add more
            jobs_needed = 0
        else:
            # In target range - just maintain baseline
            jobs_needed = max(0, min_jobs - status.selfplay_jobs)
    else:
        # Original logic for non-GH200 hosts
        jobs_needed = max(0, host.min_selfplay_jobs - status.selfplay_jobs)

    # Adjust based on CPU headroom
    if status.cpu_percent > TARGET_CPU_UTILIZATION_MAX:
        jobs_needed = min(jobs_needed, 1)

    return jobs_needed


# =============================================================================
# Elo-Driven Scheduling (Curriculum Learning)
# =============================================================================

def get_elo_leaderboard() -> Dict[str, Dict[str, Any]]:
    """Load Elo ratings from leaderboard database.

    Returns dict of model_id -> {rating, games_played, board_type, num_players}
    """
    if not ELO_LEADERBOARD_DB.exists():
        return {}

    try:
        conn = sqlite3.connect(str(ELO_LEADERBOARD_DB), timeout=5.0)
        cursor = conn.execute("""
            SELECT model_id, rating, games_played, board_type, num_players
            FROM elo_ratings
            WHERE games_played > 0
            ORDER BY rating DESC
        """)
        leaderboard = {}
        for row in cursor.fetchall():
            leaderboard[row[0]] = {
                "rating": row[1],
                "games_played": row[2],
                "board_type": row[3],
                "num_players": row[4],
            }
        conn.close()
        return leaderboard
    except Exception as e:
        log(f"Failed to load Elo leaderboard: {e}", "WARN")
        return {}


def get_config_game_counts() -> Dict[str, int]:
    """Get game counts per config from match history for curriculum prioritization.

    Returns dict of "board_players" -> game_count
    """
    if not ELO_LEADERBOARD_DB.exists():
        return {}

    try:
        conn = sqlite3.connect(str(ELO_LEADERBOARD_DB), timeout=5.0)
        cursor = conn.execute("""
            SELECT board_type, num_players, COUNT(*) as game_count
            FROM match_history
            GROUP BY board_type, num_players
        """)
        counts = {}
        for row in cursor.fetchall():
            key = f"{row[0]}_{row[1]}p"
            counts[key] = row[2]
        conn.close()
        return counts
    except Exception:
        return {}


def select_curriculum_config(configs: List[Dict], game_counts: Dict[str, int]) -> Dict:
    """Select next config based on curriculum learning (prioritize underserved).

    Configs with fewer games get higher priority to ensure balanced training.
    """
    if not game_counts:
        # No history - use round-robin
        return random.choice(configs)

    # Calculate priority score for each config (lower games = higher priority)
    scored_configs = []
    for cfg in configs:
        key = f"{cfg['board']}_{cfg['players']}p"
        count = game_counts.get(key, 0)
        # Inverse priority: fewer games = higher weight
        priority = 1.0 / (count + 1)
        scored_configs.append((cfg, priority))

    # Weighted random selection based on priority
    total_weight = sum(p for _, p in scored_configs)
    if total_weight <= 0:
        return random.choice(configs)

    r = random.random() * total_weight
    cumulative = 0
    for cfg, priority in scored_configs:
        cumulative += priority
        if r <= cumulative:
            return cfg

    return configs[-1]


def start_selfplay_jobs(host: HostConfig, count: int, dry_run: bool = False) -> int:
    """Start selfplay jobs on a host with Elo-driven curriculum scheduling.

    Uses curriculum learning to prioritize underserved configs,
    ensuring balanced training data across all board types.
    """
    if count <= 0:
        return 0

    configs = [
        {"board": "square8", "players": 2, "games": 1000},
        {"board": "square8", "players": 3, "games": 500},
        {"board": "square8", "players": 4, "games": 300},
        {"board": "hex", "players": 2, "games": 500},
        {"board": "square19", "players": 2, "games": 300},
    ]

    # Load game counts for curriculum-based config selection
    game_counts = get_config_game_counts() if ELO_CURRICULUM_ENABLED else {}

    started = 0
    seed_base = int(time.time())

    for i in range(count):
        # Use curriculum-based selection if enabled, otherwise round-robin
        if ELO_CURRICULUM_ENABLED and game_counts:
            cfg = select_curriculum_config(configs, game_counts)
        else:
            cfg = configs[i % len(configs)]

        seed = seed_base + i

        output_dir = f"data/selfplay/{cfg['board']}_{cfg['players']}p"
        log_file = f"/tmp/selfplay_{cfg['board']}_{cfg['players']}p_{seed}.log"

        cmd = f"""cd {host.ringrift_path} && \\
            mkdir -p {output_dir} && \\
            nohup python3 scripts/run_hybrid_selfplay.py \\
                --num-games {cfg['games']} \\
                --board-type {cfg['board']} \\
                --num-players {cfg['players']} \\
                --output-dir {output_dir} \\
                --seed {seed} \\
                > {log_file} 2>&1 &
        """

        if dry_run:
            log(f"[DRY-RUN] Would start on {host.name}: {cfg['board']} {cfg['players']}p (curriculum)")
            started += 1
        else:
            code, out, err = ssh_cmd(host, cmd, timeout=30)
            if code == 0:
                log(f"{host.name}: Started {cfg['board']} {cfg['players']}p selfplay")
                started += 1
            else:
                log(f"{host.name}: Failed to start selfplay: {err}", "ERROR")

    return started


def check_training_status(host: HostConfig, status: HostStatus) -> Optional[str]:
    """Check if training should be started/continued. Returns action or None."""
    if "training" not in host.role and host.role != "all":
        return None

    if not host.has_gpu:
        return None

    if status.training_jobs > 0:
        return "running"

    # Check if there's a model to continue from
    code, out, err = ssh_cmd(host, f"ls {host.ringrift_path}/models/nnue/*.pt 2>/dev/null | wc -l")
    model_exists = code == 0 and out.strip() != "0"

    if model_exists:
        return "start_improvement"
    else:
        return "start_nnue"


def start_training(host: HostConfig, action: str, dry_run: bool = False) -> bool:
    """Start training job based on action."""
    if action == "running":
        return True

    if action == "start_nnue":
        cmd = f"""cd {host.ringrift_path} && \\
            mkdir -p models/nnue logs/nnue && \\
            nohup python3 scripts/train_nnue.py \\
                --db data/games/*.db \\
                --board-type square8 --num-players 2 \\
                --epochs 50 --batch-size 1024 \\
                --save-path models/nnue/square8_2p_v1.pt \\
                > logs/nnue/train_overnight.log 2>&1 &
        """
    elif action == "start_improvement":
        cmd = f"""cd {host.ringrift_path} && \\
            mkdir -p logs/improvement && \\
            nohup python3 scripts/run_improvement_loop.py \\
                --board square8 --players 2 \\
                --iterations 100 --games-per-iter 50 \\
                --promotion-threshold 0.55 --resume \\
                > logs/improvement/overnight.log 2>&1 &
        """
    else:
        return False

    if dry_run:
        log(f"[DRY-RUN] Would start {action} on {host.name}")
        return True

    code, out, err = ssh_cmd(host, cmd, timeout=30)
    if code == 0:
        log(f"{host.name}: Started {action}")
        return True
    else:
        log(f"{host.name}: Failed to start {action}: {err}", "ERROR")
        return False


def print_status_dashboard(hosts: List[HostConfig], statuses: Dict[str, HostStatus]):
    """Print a nice status dashboard."""
    print("\n" + "=" * 90)
    print(f"CLUSTER STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    print(f"{'Host':<18} {'Status':<8} {'CPU%':<6} {'MEM%':<6} {'Disk%':<6} {'GPU%':<6} {'Jobs':<20}")
    print("-" * 90)

    total_selfplay = 0
    total_training = 0
    total_elo = 0
    reachable_count = 0

    for host in hosts:
        status = statuses.get(host.name)
        if not status:
            print(f"{host.name:<18} {'?':<8}")
            continue

        if status.reachable:
            reachable_count += 1
            total_selfplay += status.selfplay_jobs
            total_training += status.training_jobs
            total_elo += status.elo_tournament_jobs

            jobs_str = f"S:{status.selfplay_jobs} T:{status.training_jobs} C:{status.cmaes_jobs} E:{status.elo_tournament_jobs}"
            print(f"{host.name:<18} {'OK':<8} {status.cpu_percent:<6.1f} {status.memory_percent:<6.1f} "
                  f"{status.disk_percent:<6.1f} {status.gpu_percent:<6.1f} {jobs_str:<20}")
        else:
            print(f"{host.name:<18} {'DOWN':<8} {'-':<6} {'-':<6} {'-':<6} {'-':<6} {status.error[:20]}")

    print("-" * 90)
    print(f"TOTALS: {reachable_count}/{len(hosts)} hosts up | {total_selfplay} selfplay | {total_training} training | {total_elo} elo")
    print("=" * 90 + "\n")


def load_state() -> ClusterState:
    """Load persistent state."""
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            return ClusterState(**data)
        except Exception as e:
            log(f"Failed to load state: {e}", "WARN")
    return ClusterState()


def save_state(state: ClusterState):
    """Save persistent state."""
    STATE_FILE.write_text(json.dumps({
        "iteration": state.iteration,
        "total_jobs_started": state.total_jobs_started,
        "total_restarts": state.total_restarts,
        "host_statuses": state.host_statuses,
        "last_sync": state.last_sync,
        "last_elo_calibration": state.last_elo_calibration,
        "last_model_sync": state.last_model_sync,
        "errors": state.errors[-100:],  # Keep last 100 errors
    }, indent=2))


def acquire_lock() -> bool:
    """Acquire lockfile to prevent multiple instances.

    Uses unified cluster coordination if available, falls back to local lock.
    """
    # Use unified coordination if available
    if HAS_COORDINATION:
        if not acquire_orchestrator_lock("cluster_orchestrator"):
            log("Another orchestrator is already running (via unified lock)", "WARN")
            return False
        log("Acquired unified orchestrator lock")
        return True

    # Fallback to local lockfile
    if LOCKFILE.exists():
        try:
            pid = int(LOCKFILE.read_text().strip())
            # Check if process is still running
            os.kill(pid, 0)
            return False  # Process still running
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # Stale lockfile, proceed

    LOCKFILE.write_text(str(os.getpid()))
    return True


def release_lock():
    """Release lockfile."""
    # Release unified coordination lock
    if HAS_COORDINATION:
        release_orchestrator_lock()

    # Also release local lockfile
    try:
        LOCKFILE.unlink()
    except FileNotFoundError:
        pass


def check_host_can_spawn(host_name: str, ssh_host: str, task_type: str = "selfplay") -> bool:
    """Check if a host can accept new tasks using coordination module."""
    if not HAS_COORDINATION:
        return True  # No coordination, allow spawn

    if check_emergency_halt():
        log(f"Emergency halt active, blocking spawn on {host_name}", "WARN")
        return False

    return can_spawn_task(host=ssh_host, task_type=task_type)


def sync_to_mac_studio(hosts: List[HostConfig], dry_run: bool = False) -> bool:
    """Sync selfplay data from all reachable hosts directly to Mac Studio.

    Uses this machine as a relay: pulls data from cloud hosts, then pushes to Mac Studio.
    This avoids storing large amounts of data on the laptop.
    """
    log("Starting data sync to Mac Studio...")

    # Check Mac Studio reachability
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             MAC_STUDIO_HOST, "echo ok"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            log(f"Mac Studio unreachable: {result.stderr}", "ERROR")
            return False
    except Exception as e:
        log(f"Mac Studio connection failed: {e}", "ERROR")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mac_studio_sync_dir = f"{MAC_STUDIO_DATA_DIR}/synced_{timestamp}"

    if not dry_run:
        # Create sync directory on Mac Studio
        subprocess.run(
            ["ssh", MAC_STUDIO_HOST, f"mkdir -p {mac_studio_sync_dir}"],
            capture_output=True, timeout=30
        )

    synced_count = 0

    # Sync from each cloud host (skip local Macs and Mac Studio itself)
    cloud_hosts = [h for h in hosts if h.enabled and
                   "mac" not in h.name.lower() and
                   h.ssh_host and h.role != "training"]

    for host in cloud_hosts:
        try:
            log(f"Syncing from {host.name} to Mac Studio...")

            # Create temp directory for relay
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Pull from cloud host
                rsync_src = [
                    "rsync", "-avz", "--progress",
                    "-e", f"ssh -p {host.ssh_port} -o ConnectTimeout=15"
                ]

                # Try to sync .db files
                db_cmd = rsync_src + [
                    f"{host.ssh_user}@{host.ssh_host}:{host.ringrift_path}/data/games/*.db",
                    f"{tmp_dir}/"
                ]

                if dry_run:
                    log(f"  [DRY-RUN] Would sync from {host.name}")
                    continue

                result = subprocess.run(db_cmd, capture_output=True, text=True, timeout=300)

                # Check if any files were synced
                synced_files = list(Path(tmp_dir).glob("*.db"))
                if not synced_files:
                    log(f"  {host.name}: No .db files found")
                    continue

                # Push to Mac Studio
                dest_subdir = f"{mac_studio_sync_dir}/{host.name}"
                subprocess.run(
                    ["ssh", MAC_STUDIO_HOST, f"mkdir -p {dest_subdir}"],
                    capture_output=True, timeout=30
                )

                push_cmd = [
                    "rsync", "-avz", "--progress",
                    f"{tmp_dir}/",
                    f"{MAC_STUDIO_HOST}:{dest_subdir}/"
                ]
                result = subprocess.run(push_cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    log(f"  {host.name}: Synced {len(synced_files)} file(s) to Mac Studio")
                    synced_count += 1
                else:
                    log(f"  {host.name}: Failed to push to Mac Studio: {result.stderr}", "WARN")

        except subprocess.TimeoutExpired:
            log(f"  {host.name}: Sync timeout", "WARN")
        except Exception as e:
            log(f"  {host.name}: Sync error: {e}", "WARN")

    if synced_count > 0:
        log(f"Data sync complete: {synced_count} host(s) synced to Mac Studio")
    else:
        log("Data sync: No data synced (hosts may be unreachable or have no data)")

    return synced_count > 0


def sync_models_to_cluster(dry_run: bool = False) -> bool:
    """Sync NN and NNUE models across all cluster hosts (bidirectional).

    Uses the canonical sync_models.py script which:
    1. Collects missing models from remote hosts to local
    2. Distributes local models to all remote hosts
    """
    if not MODEL_SYNC_ENABLED:
        log("Model sync disabled via MODEL_SYNC_ENABLED=False")
        return False

    log("Starting bidirectional model sync across cluster...")

    # Use the canonical sync_models.py script
    script_path = Path(__file__).parent / "sync_models.py"
    if not script_path.exists():
        log(f"Model sync script not found: {script_path}", "ERROR")
        return False

    cmd = ["python3", str(script_path)]
    if dry_run:
        cmd.append("--dry-run")
    else:
        cmd.append("--sync")  # Full bidirectional sync

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes max for full sync
            cwd=Path(__file__).parent.parent,  # ai-service directory
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
        )

        if result.returncode == 0:
            # Extract summary from output
            lines = result.stdout.strip().split("\n")
            for line in lines[-10:]:
                if "synced" in line.lower() or "Total" in line:
                    log(f"Model sync: {line.strip()}")
            return True
        else:
            log(f"Model sync failed: {result.stderr[:500]}", "ERROR")
            return False

    except subprocess.TimeoutExpired:
        log("Model sync timed out after 30 minutes", "ERROR")
        return False
    except Exception as e:
        log(f"Model sync error: {e}", "ERROR")
        return False


def get_cpu_rich_hosts(hosts: List[HostConfig], statuses: Dict[str, HostStatus]) -> List[Tuple[HostConfig, HostStatus]]:
    """Get CPU-rich hosts suitable for tournament workloads.

    Prioritizes hosts with:
    - High CPU count
    - Low current CPU utilization (< 60%)
    - Not GPU-constrained (or no GPU)
    """
    cpu_hosts = []
    for host in hosts:
        if not host.enabled:
            continue
        status = statuses.get(host.name)
        if not status or not status.reachable:
            continue
        # Prefer hosts with available CPU capacity
        if status.cpu_percent < TARGET_CPU_UTILIZATION_MIN:
            cpu_hosts.append((host, status))

    # Sort by CPU count (descending) - bigger hosts first
    cpu_hosts.sort(key=lambda x: x[0].cpus, reverse=True)
    return cpu_hosts


def get_gpu_rich_hosts(hosts: List[HostConfig], statuses: Dict[str, HostStatus]) -> List[Tuple[HostConfig, HostStatus]]:
    """Get GPU-rich hosts suitable for GPU selfplay and training.

    Prioritizes hosts with:
    - GPU available
    - Low GPU utilization (< 60%)
    """
    gpu_hosts = []
    for host in hosts:
        if not host.enabled or not host.has_gpu:
            continue
        status = statuses.get(host.name)
        if not status or not status.reachable:
            continue
        # Prefer hosts with available GPU capacity
        if status.gpu_percent < TARGET_GPU_UTILIZATION_MIN:
            gpu_hosts.append((host, status))

    # Sort by GPU memory descending (bigger GPUs first)
    gpu_hosts.sort(key=lambda x: x[0].memory_gb, reverse=True)
    return gpu_hosts


def trigger_diverse_tournaments(
    hosts: List[HostConfig],
    statuses: Dict[str, HostStatus],
    dry_run: bool = False,
) -> int:
    """Start diverse tournaments on CPU-rich hosts.

    Distributes tournaments across:
    - All board types (square8, square19, hexagonal)
    - All player counts (2, 3, 4)
    - All AI types (random, heuristic, MCTS, neural net)

    Returns number of tournaments started.
    """
    cpu_hosts = get_cpu_rich_hosts(hosts, statuses)
    if not cpu_hosts:
        log("Diverse tournaments: No CPU-rich hosts available")
        return 0

    log(f"Starting diverse tournaments on {len(cpu_hosts)} CPU-rich host(s)...")
    tournaments_started = 0

    # Distribute tournament configs across available hosts
    for idx, (host, status) in enumerate(cpu_hosts):
        # Each host gets a different config from the rotation
        config = TOURNAMENT_CONFIGS[idx % len(TOURNAMENT_CONFIGS)]
        board_type = config["board_type"]
        num_players = config["num_players"]

        log(f"  {host.name}: Assigning {board_type} {num_players}p tournament (CPU {status.cpu_percent:.0f}%)")

        # Build SSH command
        ssh_base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if host.ssh_key:
            ssh_base.extend(["-i", os.path.expanduser(host.ssh_key)])
        if host.ssh_port != 22:
            ssh_base.extend(["-p", str(host.ssh_port)])

        target = f"{host.ssh_user}@{host.ssh_host}"
        ringrift_path = host.ringrift_path
        log_file = f"/tmp/tournament_{board_type}_{num_players}p.log"

        # Run comprehensive Elo tournament with all AI types and cross-inference
        # --both-ai-types ensures games test all 4 AI combinations for robust ratings
        remote_cmd = (
            f"cd {ringrift_path} && "
            f"({host.venv_activate}) 2>/dev/null || true && "
            f"nohup python3 scripts/run_model_elo_tournament.py "
            f"--board {board_type} --players {num_players} "
            f"--games {TOURNAMENT_GAMES_PER_MATCHUP} "
            f"--include-baselines --both-ai-types --run "
            f"> {log_file} 2>&1 & "
            f"echo 'Tournament started'"
        )

        cmd = ssh_base + [target, remote_cmd]

        if dry_run:
            log(f"  [DRY-RUN] Would start {board_type} {num_players}p tournament on {host.name}")
            tournaments_started += 1
        else:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    log(f"  {host.name}: Tournament started ({board_type} {num_players}p)")
                    tournaments_started += 1
                else:
                    log(f"  {host.name}: Tournament start failed: {result.stderr}", "WARN")
            except subprocess.TimeoutExpired:
                log(f"  {host.name}: SSH timeout during tournament start", "WARN")
            except Exception as e:
                log(f"  {host.name}: Tournament error: {e}", "WARN")

    log(f"Diverse tournaments: Started {tournaments_started} tournament(s)")
    return tournaments_started


def start_gpu_selfplay(
    hosts: List[HostConfig],
    statuses: Dict[str, HostStatus],
    dry_run: bool = False,
) -> int:
    """Start GPU-accelerated selfplay on GPU-rich hosts with low utilization.

    Uses hybrid selfplay (CPU MCTS + GPU neural net) for maximum throughput.
    """
    gpu_hosts = get_gpu_rich_hosts(hosts, statuses)
    if not gpu_hosts:
        log("GPU selfplay: No GPU-rich hosts available")
        return 0

    log(f"Starting GPU selfplay on {len(gpu_hosts)} GPU-rich host(s)...")
    jobs_started = 0

    # Distribute board configs across GPU hosts
    board_configs = [
        {"board": "square8", "players": 2, "games": 500},
        {"board": "square8", "players": 3, "games": 300},
        {"board": "square8", "players": 4, "games": 200},
        {"board": "square19", "players": 2, "games": 200},
        {"board": "hexagonal", "players": 2, "games": 300},
    ]

    for idx, (host, status) in enumerate(gpu_hosts):
        # Calculate how many jobs to start based on GPU headroom
        gpu_headroom = TARGET_GPU_UTILIZATION_MIN - status.gpu_percent
        jobs_to_start = max(1, int(gpu_headroom / 15))  # ~15% GPU per job

        # Don't overload
        jobs_to_start = min(jobs_to_start, 5)

        log(f"  {host.name}: GPU {status.gpu_percent:.0f}%, starting {jobs_to_start} GPU selfplay job(s)")

        for job_idx in range(jobs_to_start):
            cfg = board_configs[(idx + job_idx) % len(board_configs)]
            seed = int(time.time()) + job_idx + idx * 100

            ssh_base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
            if host.ssh_key:
                ssh_base.extend(["-i", os.path.expanduser(host.ssh_key)])
            if host.ssh_port != 22:
                ssh_base.extend(["-p", str(host.ssh_port)])

            target = f"{host.ssh_user}@{host.ssh_host}"
            ringrift_path = host.ringrift_path
            log_file = f"/tmp/gpu_selfplay_{cfg['board']}_{cfg['players']}p_{seed}.log"

            # Use hybrid selfplay for GPU-accelerated games
            remote_cmd = (
                f"cd {ringrift_path} && "
                f"({host.venv_activate}) 2>/dev/null || true && "
                f"nohup python3 scripts/run_hybrid_selfplay.py "
                f"--board-type {cfg['board']} --num-players {cfg['players']} "
                f"--num-games {cfg['games']} --seed {seed} "
                f"--use-gpu --difficulty 10 "
                f"> {log_file} 2>&1 & "
            )

            cmd = ssh_base + [target, remote_cmd]

            if dry_run:
                log(f"    [DRY-RUN] Would start GPU selfplay on {host.name}")
                jobs_started += 1
            else:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        jobs_started += 1
                    else:
                        log(f"    {host.name}: GPU selfplay failed: {result.stderr}", "WARN")
                except Exception as e:
                    log(f"    {host.name}: Error: {e}", "WARN")

    log(f"GPU selfplay: Started {jobs_started} job(s)")
    return jobs_started


def scale_to_target_utilization(
    hosts: List[HostConfig],
    statuses: Dict[str, HostStatus],
    dry_run: bool = False,
) -> Dict[str, int]:
    """Scale up jobs to reach 60% target utilization on all hosts.

    Returns dict with counts of jobs started by type.
    """
    results = {"tournaments": 0, "selfplay": 0, "gpu_selfplay": 0, "training": 0}

    for host in hosts:
        if not host.enabled:
            continue
        status = statuses.get(host.name)
        if not status or not status.reachable:
            continue

        # Skip hosts at or above target utilization
        cpu_underutilized = status.cpu_percent < TARGET_CPU_UTILIZATION_MIN
        gpu_underutilized = host.has_gpu and status.gpu_percent < TARGET_GPU_UTILIZATION_MIN

        if not cpu_underutilized and not gpu_underutilized:
            continue

        log(f"{host.name}: Scaling up (CPU {status.cpu_percent:.0f}%, GPU {status.gpu_percent:.0f}%)")

        # For CPU-underutilized hosts: start tournaments or selfplay
        if cpu_underutilized:
            cpu_gap = TARGET_CPU_UTILIZATION_MIN - status.cpu_percent

            # If no tournament running, start one
            if status.elo_tournament_jobs == 0 and cpu_gap > 20:
                # Pick a config based on host name hash for variety
                config_idx = hash(host.name) % len(TOURNAMENT_CONFIGS)
                cfg = TOURNAMENT_CONFIGS[config_idx]

                ssh_args = _build_ssh_args(host)
                target = f"{host.ssh_user}@{host.ssh_host}"
                log_file = f"/tmp/auto_tournament_{cfg['board_type']}_{cfg['num_players']}p.log"

                remote_cmd = (
                    f"cd {host.ringrift_path} && "
                    f"({host.venv_activate}) 2>/dev/null || true && "
                    f"nohup python3 scripts/run_model_elo_tournament.py "
                    f"--board {cfg['board_type']} --players {cfg['num_players']} "
                    f"--games {TOURNAMENT_GAMES_PER_MATCHUP} --include-baselines --both-ai-types --run "
                    f"> {log_file} 2>&1 &"
                )

                if not dry_run:
                    try:
                        result = subprocess.run(
                            ssh_args + [target, remote_cmd],
                            capture_output=True, text=True, timeout=30
                        )
                        if result.returncode == 0:
                            results["tournaments"] += 1
                            log(f"  {host.name}: Started tournament ({cfg['board_type']} {cfg['num_players']}p)")
                    except Exception as e:
                        log(f"  {host.name}: Tournament error: {e}", "WARN")
                else:
                    results["tournaments"] += 1

            # Also start selfplay to fill remaining CPU capacity
            selfplay_jobs = max(1, int(cpu_gap / 15))  # ~15% CPU per job
            started = start_selfplay_jobs(host, selfplay_jobs, dry_run)
            results["selfplay"] += started

        # For GPU-underutilized hosts: start GPU selfplay or training
        if gpu_underutilized:
            gpu_gap = TARGET_GPU_UTILIZATION_MIN - status.gpu_percent

            # Prefer training if no training jobs running
            if status.training_jobs == 0 and "training" in host.role:
                action = check_training_status(host, status)
                if action and action != "running":
                    if start_training(host, action, dry_run):
                        results["training"] += 1
            else:
                # Otherwise start GPU selfplay
                jobs_to_start = max(1, int(gpu_gap / 20))  # ~20% GPU per job
                for _ in range(min(jobs_to_start, 3)):
                    cfg_idx = hash(f"{host.name}_{time.time()}") % len(TOURNAMENT_CONFIGS)
                    cfg = TOURNAMENT_CONFIGS[cfg_idx]

                    ssh_args = _build_ssh_args(host)
                    target = f"{host.ssh_user}@{host.ssh_host}"
                    seed = int(time.time())
                    log_file = f"/tmp/gpu_selfplay_{cfg['board_type']}_{cfg['num_players']}p_{seed}.log"

                    remote_cmd = (
                        f"cd {host.ringrift_path} && "
                        f"({host.venv_activate}) 2>/dev/null || true && "
                        f"nohup python3 scripts/run_hybrid_selfplay.py "
                        f"--board-type {cfg['board_type']} --num-players {cfg['num_players']} "
                        f"--num-games 200 --seed {seed} --use-gpu --difficulty 10 "
                        f"> {log_file} 2>&1 &"
                    )

                    if not dry_run:
                        try:
                            result = subprocess.run(
                                ssh_args + [target, remote_cmd],
                                capture_output=True, text=True, timeout=30
                            )
                            if result.returncode == 0:
                                results["gpu_selfplay"] += 1
                        except Exception as e:
                            log(f"  {host.name}: GPU selfplay error: {e}", "WARN")
                    else:
                        results["gpu_selfplay"] += 1

    return results


def _build_ssh_args(host: HostConfig) -> List[str]:
    """Build SSH command arguments for a host."""
    ssh_args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    if host.ssh_key:
        ssh_args.extend(["-i", os.path.expanduser(host.ssh_key)])
    if host.ssh_port != 22:
        ssh_args.extend(["-p", str(host.ssh_port)])
    return ssh_args


def trigger_elo_calibration(hosts: List[HostConfig], statuses: Dict[str, HostStatus], dry_run: bool = False) -> bool:
    """Trigger Elo calibration tournament on a GPU host.

    Runs run_model_elo_tournament.py with --all-configs on the first available GPU host.
    Returns True if successfully triggered.
    """
    # Find a GPU host that is reachable and not overloaded
    gpu_host = None
    for host in hosts:
        if not host.enabled or not host.has_gpu:
            continue
        status = statuses.get(host.name)
        if not status or not status.reachable:
            continue
        # Prefer hosts with lower CPU usage
        if status.cpu_percent < 80:
            gpu_host = host
            break

    if not gpu_host:
        log("Elo calibration: No available GPU host found", "WARN")
        return False

    log(f"Triggering Elo calibration on {gpu_host.name}...")

    # Build SSH command
    ssh_base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    if gpu_host.ssh_key:
        ssh_base.extend(["-i", os.path.expanduser(gpu_host.ssh_key)])
    if gpu_host.ssh_port != 22:
        ssh_base.extend(["-p", str(gpu_host.ssh_port)])

    target = f"{gpu_host.ssh_user}@{gpu_host.ssh_host}"
    ringrift_path = gpu_host.ringrift_path

    # Run Elo tournament in background with nohup
    # Use --both-ai-types to test all 4 AI type combinations for comprehensive ratings:
    # descent vs descent, mcts vs mcts, mcts vs descent, descent vs mcts
    remote_cmd = (
        f"cd {ringrift_path} && "
        f"source venv/bin/activate && "
        f"nohup python3 scripts/run_model_elo_tournament.py "
        f"--all-configs --games {ELO_CALIBRATION_GAMES} --both-ai-types --run "
        f"> /tmp/elo_calibration_auto.log 2>&1 &"
        f"echo 'Elo calibration started with cross-inference testing'"
    )

    cmd = ssh_base + [target, remote_cmd]

    if dry_run:
        log(f"[DRY-RUN] Would run: {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            log(f"Elo calibration triggered on {gpu_host.name}")
            return True
        else:
            log(f"Elo calibration failed: {result.stderr}", "WARN")
            return False
    except subprocess.TimeoutExpired:
        log("Elo calibration: SSH timeout", "WARN")
        return False
    except Exception as e:
        log(f"Elo calibration error: {e}", "WARN")
        return False


def trigger_auto_scale_selfplay(
    hosts: List[HostConfig],
    statuses: Dict[str, HostStatus],
    dry_run: bool = False,
    include_local_hosts: bool = False,
) -> int:
    """Identify underutilized hosts and start canonical self-play on them.

    A host is considered underutilized if:
    - CPU usage < UNDERUTILIZED_CPU_THRESHOLD
    - Python jobs < UNDERUTILIZED_PYTHON_JOBS
    - Disk usage < 85%
    - Memory usage < 70%

    Returns the number of hosts scaled up.
    """
    scaled_count = 0

    # Board types to cycle through for canonical self-play
    board_configs = [
        {"board_type": "square8", "num_players": 2},
        {"board_type": "square8", "num_players": 3},
        {"board_type": "square8", "num_players": 4},
        {"board_type": "square19", "num_players": 2},
        {"board_type": "hexagonal", "num_players": 2},
    ]

    def _is_local_host(host: HostConfig) -> bool:
        name = host.name.strip().lower()
        return name.startswith("mac") or name.startswith("mbp")

    def _resolve_db_and_summary(board_type: str, num_players: int) -> tuple[str, str]:
        bt = board_type.strip().lower()
        np = int(num_players)
        if bt == "square8" and np == 2:
            return "canonical_square8.db", "data/games/db_health.canonical_square8.json"
        if bt == "square8" and np == 3:
            return "canonical_square8_3p.db", "data/games/db_health.canonical_square8_3p.json"
        if bt == "square8" and np == 4:
            return "canonical_square8_4p.db", "data/games/db_health.canonical_square8_4p.json"
        if bt == "square19" and np == 2:
            return "canonical_square19.db", "data/games/db_health.canonical_square19.json"
        if bt == "hexagonal" and np == 2:
            return "canonical_hex.db", "data/games/db_health.canonical_hex.json"
        raise ValueError(f"Unsupported canonical selfplay config: board={board_type!r} players={num_players!r}")

    underutilized_hosts = []
    for host in hosts:
        if not host.enabled:
            continue
        if not include_local_hosts and _is_local_host(host):
            continue
        status = statuses.get(host.name)
        if not status or not status.reachable:
            continue

        # Check if underutilized
        is_underutilized = (
            status.cpu_percent < UNDERUTILIZED_CPU_THRESHOLD
            and status.python_jobs < UNDERUTILIZED_PYTHON_JOBS
            and status.disk_percent < 85
            and status.memory_percent < 70
        )

        if is_underutilized:
            underutilized_hosts.append((host, status))

    if not underutilized_hosts:
        log("Auto-scale: No underutilized hosts found")
        return 0

    log(f"Auto-scale: Found {len(underutilized_hosts)} underutilized host(s)")

    for idx, (host, status) in enumerate(underutilized_hosts):
        # Cycle through board configs
        cfg = board_configs[idx % len(board_configs)]
        board_type = str(cfg["board_type"])
        num_players = int(cfg["num_players"])
        db_name, summary_path = _resolve_db_and_summary(board_type, num_players)

        log(
            f"  {host.name}: CPU={status.cpu_percent:.0f}%, "
            f"jobs={status.python_jobs}, scaling up with {board_type} {num_players}p"
        )

        # Build SSH command
        ssh_base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        if host.ssh_key:
            ssh_base.extend(["-i", os.path.expanduser(host.ssh_key)])
        if host.ssh_port != 22:
            ssh_base.extend(["-p", str(host.ssh_port)])

        target = f"{host.ssh_user}@{host.ssh_host}"
        ringrift_path = host.ringrift_path

        # Start canonical self-play
        log_file = f"/tmp/autoscale_selfplay_{board_type}_{num_players}p.log"
        venv_activate = host.venv_activate.strip() if host.venv_activate else "source venv/bin/activate"

        remote_cmd = (
            f"cd {ringrift_path} && "
            f"({venv_activate}) 2>/dev/null || true && "
            f"nohup python3 scripts/generate_canonical_selfplay.py "
            f"--board-type {board_type} "
            f"--num-players {num_players} "
            f"--num-games {SCALE_UP_GAMES_PER_HOST} "
            f"--db data/games/{db_name} "
            f"--summary {summary_path} "
            f"> {log_file} 2>&1 & "
            f"echo 'Auto-scale selfplay started'"
        )

        cmd = ssh_base + [target, remote_cmd]

        if dry_run:
            log(f"  [DRY-RUN] Would run: {' '.join(cmd[:6])}...")
            scaled_count += 1
        else:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    log(f"  {host.name}: Auto-scale selfplay started")
                    scaled_count += 1
                else:
                    log(f"  {host.name}: Auto-scale failed: {result.stderr}", "WARN")
            except subprocess.TimeoutExpired:
                log(f"  {host.name}: SSH timeout during auto-scale", "WARN")
            except Exception as e:
                log(f"  {host.name}: Auto-scale error: {e}", "WARN")

    log(f"Auto-scale complete: {scaled_count} host(s) scaled up")
    return scaled_count


def main():
    parser = argparse.ArgumentParser(description="Enhanced Cluster Orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute commands")
    parser.add_argument("--status-only", action="store_true", help="Just show status and exit")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--sync-now", action="store_true", help="Force sync to Mac Studio immediately")
    parser.add_argument("--no-sync", action="store_true", help="Disable periodic sync to Mac Studio")
    parser.add_argument("--elo-calibration-now", action="store_true", help="Trigger Elo calibration immediately")
    parser.add_argument("--no-elo-calibration", action="store_true", help="Disable periodic Elo calibration")
    parser.add_argument("--elo-interval", type=int, default=ELO_CALIBRATION_INTERVAL,
                        help=f"Elo calibration interval in iterations (default: {ELO_CALIBRATION_INTERVAL})")
    parser.add_argument("--auto-scale-now", action="store_true", help="Run auto-scale check immediately")
    parser.add_argument("--no-auto-scale", action="store_true", help="Disable automatic scaling of underutilized hosts")
    parser.add_argument("--auto-scale-interval", type=int, default=AUTO_SCALE_INTERVAL,
                        help=f"Auto-scale check interval in iterations (default: {AUTO_SCALE_INTERVAL})")
    parser.add_argument(
        "--auto-scale-include-local-hosts",
        action="store_true",
        help="Include mac/mbp hosts when auto-scaling (default: skip local hosts).",
    )
    parser.add_argument("--tournament-interval", type=int, default=TOURNAMENT_INTERVAL,
                        help=f"Tournament interval in iterations (default: {TOURNAMENT_INTERVAL})")
    parser.add_argument("--no-tournaments", action="store_true", help="Disable periodic tournaments")
    parser.add_argument("--tournaments-now", action="store_true", help="Start diverse tournaments immediately")
    parser.add_argument("--target-utilization-now", action="store_true",
                        help="Scale up to 60% utilization immediately")
    parser.add_argument("--no-utilization-targeting", action="store_true",
                        help="Disable automatic utilization targeting")
    parser.add_argument("--gpu-selfplay-now", action="store_true", help="Start GPU selfplay immediately")
    args = parser.parse_args()

    # Acquire lockfile to prevent multiple instances
    if not acquire_lock():
        log("Another instance is already running (lockfile exists with active PID)", "ERROR")
        sys.exit(1)

    # Register cleanup handlers
    def cleanup(signum=None, frame=None):
        log("Shutting down...")
        release_lock()
        if signum:
            sys.exit(0)

    atexit.register(release_lock)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    log("=" * 60)
    log("CLUSTER ORCHESTRATOR STARTING")
    log(f"PID: {os.getpid()}")
    log("=" * 60)

    hosts = load_hosts_config()
    log(f"Loaded {len(hosts)} host configurations")

    state = load_state()

    # Force sync if requested
    if args.sync_now:
        log("Forcing sync to Mac Studio...")
        if sync_to_mac_studio(hosts, args.dry_run):
            state.last_sync = datetime.now().isoformat()
            save_state(state)
        return

    # Force Elo calibration if requested
    if args.elo_calibration_now:
        log("Forcing Elo calibration...")
        # Need to collect statuses first
        statuses: Dict[str, HostStatus] = {}
        for host in hosts:
            if host.enabled:
                statuses[host.name] = check_host_status(host)
        if trigger_elo_calibration(hosts, statuses, args.dry_run):
            state.last_elo_calibration = datetime.now().isoformat()
            save_state(state)
        return

    # Force auto-scale if requested
    if args.auto_scale_now:
        log("Forcing auto-scale check...")
        # Need to collect statuses first
        statuses: Dict[str, HostStatus] = {}
        for host in hosts:
            if host.enabled:
                statuses[host.name] = check_host_status(host)
        trigger_auto_scale_selfplay(
            hosts,
            statuses,
            args.dry_run,
            include_local_hosts=bool(args.auto_scale_include_local_hosts),
        )
        return

    # Force tournaments if requested
    if args.tournaments_now:
        log("Starting diverse tournaments immediately...")
        statuses: Dict[str, HostStatus] = {}
        for host in hosts:
            if host.enabled:
                statuses[host.name] = check_host_status(host)
        trigger_diverse_tournaments(hosts, statuses, args.dry_run)
        return

    # Force utilization targeting if requested
    if args.target_utilization_now:
        log("Scaling to 60% utilization target...")
        statuses: Dict[str, HostStatus] = {}
        for host in hosts:
            if host.enabled:
                statuses[host.name] = check_host_status(host)
        results = scale_to_target_utilization(hosts, statuses, args.dry_run)
        log(f"Started: {results['tournaments']} tournaments, {results['selfplay']} selfplay, "
            f"{results['gpu_selfplay']} GPU selfplay, {results['training']} training")
        return

    # Force GPU selfplay if requested
    if args.gpu_selfplay_now:
        log("Starting GPU selfplay immediately...")
        statuses: Dict[str, HostStatus] = {}
        for host in hosts:
            if host.enabled:
                statuses[host.name] = check_host_status(host)
        start_gpu_selfplay(hosts, statuses, args.dry_run)
        return

    while True:
        state.iteration += 1
        log(f"--- Iteration {state.iteration} ---")

        # Collect status from all hosts
        statuses: Dict[str, HostStatus] = {}
        for host in hosts:
            if not host.enabled:
                continue
            status = check_host_status(host)
            statuses[host.name] = status
            state.host_statuses[host.name] = {
                "reachable": status.reachable,
                "cpu_percent": status.cpu_percent,
                "memory_percent": status.memory_percent,
                "disk_percent": status.disk_percent,
                "selfplay_jobs": status.selfplay_jobs,
                "training_jobs": status.training_jobs,
                "last_check": status.last_check,
            }

        # Print dashboard
        print_status_dashboard(hosts, statuses)

        if args.status_only:
            break

        # Manage each host
        for host in hosts:
            if not host.enabled:
                continue

            status = statuses.get(host.name)
            if not status or not status.reachable:
                continue

            # Selfplay management
            jobs_to_start = should_start_selfplay(host, status)
            if jobs_to_start > 0:
                started = start_selfplay_jobs(host, jobs_to_start, args.dry_run)
                state.total_jobs_started += started
                state.total_restarts += started

            # Training management (only on GPU hosts)
            if host.has_gpu:
                action = check_training_status(host, status)
                if action and action != "running":
                    start_training(host, action, args.dry_run)

        # Periodic sync to Mac Studio (every SYNC_INTERVAL iterations)
        if not args.no_sync and state.iteration % SYNC_INTERVAL == 0:
            log(f"Starting periodic sync (every {SYNC_INTERVAL} iterations)...")
            if sync_to_mac_studio(hosts, args.dry_run):
                state.last_sync = datetime.now().isoformat()

        # Periodic model sync across cluster (every MODEL_SYNC_INTERVAL iterations)
        if MODEL_SYNC_ENABLED and state.iteration % MODEL_SYNC_INTERVAL == 0:
            log(f"Starting model sync (every {MODEL_SYNC_INTERVAL} iterations)...")
            if sync_models_to_cluster(args.dry_run):
                state.last_model_sync = datetime.now().isoformat()

        # Periodic Elo calibration (every elo_interval iterations)
        if not args.no_elo_calibration and state.iteration % args.elo_interval == 0:
            log(f"Starting periodic Elo calibration (every {args.elo_interval} iterations)...")
            if trigger_elo_calibration(hosts, statuses, args.dry_run):
                state.last_elo_calibration = datetime.now().isoformat()

        # Periodic auto-scale check (every auto_scale_interval iterations)
        if not args.no_auto_scale and state.iteration % args.auto_scale_interval == 0:
            log(f"Starting auto-scale check (every {args.auto_scale_interval} iterations)...")
            trigger_auto_scale_selfplay(
                hosts,
                statuses,
                args.dry_run,
                include_local_hosts=bool(args.auto_scale_include_local_hosts),
            )

        # Periodic diverse tournaments (every tournament_interval iterations)
        if not args.no_tournaments and state.iteration % args.tournament_interval == 0:
            log(f"Starting periodic diverse tournaments (every {args.tournament_interval} iterations)...")
            trigger_diverse_tournaments(hosts, statuses, args.dry_run)

        # Utilization targeting - scale to 60% on every iteration
        if not args.no_utilization_targeting:
            results = scale_to_target_utilization(hosts, statuses, args.dry_run)
            if any(results.values()):
                log(f"Utilization targeting: tournaments={results['tournaments']} selfplay={results['selfplay']} "
                    f"gpu_selfplay={results['gpu_selfplay']} training={results['training']}")
                state.total_jobs_started += sum(results.values())

        save_state(state)

        if args.once:
            break

        log(f"Sleeping {args.interval}s until next check...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
