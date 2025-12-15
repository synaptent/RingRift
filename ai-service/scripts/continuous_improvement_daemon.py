#!/usr/bin/env python3
"""
DEPRECATED: This module is deprecated. Use scripts/unified_ai_loop.py instead.

The unified_ai_loop.py provides all functionality with:
- Proper cluster coordination and distributed locking
- Event-driven data flow via cross-process events
- Data quality validation and training gates
- Feedback loop integration and curriculum rebalancing
- Health monitoring with Prometheus metrics

To migrate:
    python scripts/unified_ai_loop.py --foreground --verbose

---
Continuous AI Improvement Daemon - Self-healing, cyclical training loop.

This daemon orchestrates the complete AI improvement cycle:
1. Balanced selfplay generation across all board types
2. Automatic training when data thresholds are met
3. Model comparison tournaments after each training
4. Elo rating updates and leaderboard tracking
5. Automatic promotion of improved models
6. Self-healing on failures with exponential backoff

The daemon runs continuously and checkpoints all state for resume capability.

Usage:
    # Start the daemon
    python scripts/continuous_improvement_daemon.py --start

    # Run in foreground with verbose output
    python scripts/continuous_improvement_daemon.py --foreground --verbose

    # Check status
    python scripts/continuous_improvement_daemon.py --status

    # Stop gracefully
    python scripts/continuous_improvement_daemon.py --stop
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow imports from app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.unified_config import get_config

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Import unified cluster coordination
try:
    from app.distributed.cluster_coordinator import (
        ClusterCoordinator,
        TaskRole,
        ProcessLimits,
        check_and_abort_if_role_held,
    )
    HAS_COORDINATION = True
    _coordinator = ClusterCoordinator()
except ImportError:
    HAS_COORDINATION = False
    ClusterCoordinator = None
    TaskRole = None
    _coordinator = None

def check_emergency_halt() -> bool:
    """Check if emergency halt flag is set."""
    halt_file = AI_SERVICE_ROOT / "data" / "coordination" / "EMERGENCY_HALT"
    return halt_file.exists()

# Import persistent Elo database functions (centralized service)
try:
    from app.training.elo_service import (
        get_elo_service,
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
    get_elo_service = None

# Import ImprovementCycleManager for diverse AI scheduling
try:
    from scripts.improvement_cycle_manager import ImprovementCycleManager
    HAS_IMPROVEMENT_MANAGER = True
except ImportError:
    HAS_IMPROVEMENT_MANAGER = False
    ImprovementCycleManager = None

# Import league training system for tiered competition
try:
    from scripts.league_training import (
        init_league_db,
        sync_from_elo_db,
        apply_promotions_demotions,
        get_league_standings,
        LEAGUE_ORDER,
        LEAGUES,
    )
    HAS_LEAGUE_SYSTEM = True
except ImportError:
    HAS_LEAGUE_SYSTEM = False

# TaskCoordinator for global task limits and coordination
try:
    from app.coordination.task_coordinator import (
        TaskCoordinator,
        TaskType,
        TaskLimits,
        CoordinatorState,
    )
    HAS_TASK_COORDINATOR = True
except ImportError:
    HAS_TASK_COORDINATOR = False
    TaskCoordinator = None
    TaskType = None

# New coordination features: OrchestratorRole, backpressure, sync_lock
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
        report_queue_depth,
        # Sync mutex for rsync operations
        sync_lock,
        # Bandwidth management
        request_bandwidth,
        release_bandwidth,
        TransferPriority,
    )
    HAS_NEW_COORDINATION = True
except ImportError:
    HAS_NEW_COORDINATION = False
    OrchestratorRole = None

# Import optimized hyperparameters
try:
    from app.config.hyperparameters import (
        get_hyperparameters,
        get_hyperparameter_info,
        needs_tuning,
    )
    HAS_HYPERPARAMETERS = True
except ImportError:
    HAS_HYPERPARAMETERS = False
    def get_hyperparameters(board_type, num_players):
        return {}
    def get_hyperparameter_info(board_type, num_players):
        return {"optimized": False, "confidence": "none"}
    def needs_tuning(board_type, num_players, min_confidence="medium"):
        return True

# Global improvement cycle manager instance
_improvement_manager = None

def get_improvement_manager():
    """Get or create the global ImprovementCycleManager instance."""
    global _improvement_manager
    if _improvement_manager is None and HAS_IMPROVEMENT_MANAGER:
        try:
            _improvement_manager = ImprovementCycleManager(
                db_path=AI_SERVICE_ROOT / "logs" / "improvement_daemon" / "improvement_manager.db",
                ringrift_path=AI_SERVICE_ROOT.parent,
            )
            print("[Daemon] ImprovementCycleManager initialized")
        except Exception as e:
            print(f"[Daemon] Failed to initialize ImprovementCycleManager: {e}")
    return _improvement_manager

# =============================================================================
# P2P Orchestrator Integration
# =============================================================================
#
# The daemon can optionally coordinate with the P2P orchestrator for:
# - Distributed selfplay across cluster nodes
# - Centralized data manifest tracking
# - Coordinated training schedules
# - Shared Elo leaderboard

P2P_ORCHESTRATOR_URL = os.environ.get("P2P_ORCHESTRATOR_URL", "http://localhost:8770")
P2P_AUTH_TOKEN = (os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN", "") or "").strip()
if not P2P_AUTH_TOKEN:
    token_file = (os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN_FILE", "") or "").strip()
    if token_file:
        try:
            P2P_AUTH_TOKEN = Path(token_file).read_text(encoding="utf-8").strip()
        except Exception:
            P2P_AUTH_TOKEN = ""
USE_P2P_ORCHESTRATOR = os.environ.get("USE_P2P_ORCHESTRATOR", "").lower() in ("1", "true", "yes")
IMPROVEMENT_LEADER_ONLY = os.environ.get("RINGRIFT_IMPROVEMENT_LEADER_ONLY", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
IMPROVEMENT_LEADER_POLL_SECONDS = float(os.environ.get("RINGRIFT_IMPROVEMENT_LEADER_POLL_SECONDS", "15") or 15)

# Optional: sync promoted artifacts to a staging deployment.
SYNC_STAGING = os.environ.get("RINGRIFT_SYNC_STAGING", "").lower() in ("1", "true", "yes", "on")

# Disable local compute tasks (selfplay, training, tournaments) on this machine.
# Use this on low-memory dev machines to avoid OOM while still running coordination.
DISABLE_LOCAL_TASKS = os.environ.get("RINGRIFT_DISABLE_LOCAL_TASKS", "").lower() in ("1", "true", "yes", "on")
SYNC_STAGING_RESTART = os.environ.get("RINGRIFT_SYNC_STAGING_RESTART", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SYNC_STAGING_VALIDATE_HEALTH = os.environ.get("RINGRIFT_SYNC_STAGING_VALIDATE_HEALTH", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SYNC_STAGING_FAIL_ON_MISSING = os.environ.get("RINGRIFT_SYNC_STAGING_FAIL_ON_MISSING", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# S3 Backup Configuration
S3_BACKUP_ENABLED = os.environ.get("RINGRIFT_S3_BACKUP", "").lower() in ("1", "true", "yes", "on")
S3_BACKUP_INTERVAL_HOURS = float(os.environ.get("RINGRIFT_S3_BACKUP_INTERVAL_HOURS", "6"))
_last_s3_backup_time: Optional[float] = None

# Cluster metrics reporting
CLUSTER_METRICS_ENABLED = os.environ.get("RINGRIFT_CLUSTER_METRICS", "1").lower() in ("1", "true", "yes", "on")
_cluster_metrics_imported = False
try:
    from app.metrics import report_cluster_node, GPU_HOURLY_RATES
    _cluster_metrics_imported = True
except ImportError:
    pass


def report_local_cluster_metrics() -> None:
    """Report metrics for the local node to Prometheus."""
    if not CLUSTER_METRICS_ENABLED or not _cluster_metrics_imported:
        return

    try:
        import platform
        import psutil

        # Determine node identifier
        node_id = os.environ.get("RINGRIFT_NODE_ID", platform.node())

        # Determine GPU type
        gpu_type = "unknown"
        gpu_utilization = 0.0
        gpu_memory_bytes = 0

        # Try to get GPU info via nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 3:
                    gpu_name = parts[0].strip().upper()
                    if "GH200" in gpu_name:
                        gpu_type = "GH200"
                    elif "H100" in gpu_name:
                        gpu_type = "H100"
                    elif "A100" in gpu_name:
                        gpu_type = "A100"
                    elif "A10" in gpu_name:
                        gpu_type = "A10"
                    elif "4090" in gpu_name:
                        gpu_type = "RTX_4090"
                    elif "3090" in gpu_name or "3080" in gpu_name:
                        gpu_type = "RTX_4090"  # Similar pricing tier

                    gpu_utilization = float(parts[1].strip()) / 100.0
                    gpu_memory_bytes = int(float(parts[2].strip()) * 1024 * 1024)  # MiB to bytes
        except Exception:
            pass

        # Get CPU and memory info
        cpu_utilization = psutil.cpu_percent() / 100.0
        memory_info = psutil.virtual_memory()
        system_memory_bytes = memory_info.used

        report_cluster_node(
            node=node_id,
            gpu_type=gpu_type,
            is_up=True,
            gpu_utilization=gpu_utilization,
            cpu_utilization=cpu_utilization,
            gpu_memory_bytes=gpu_memory_bytes,
            system_memory_bytes=system_memory_bytes,
        )
    except Exception as e:
        print(f"[Daemon] Cluster metrics reporting failed: {e}")


async def is_local_p2p_leader() -> bool:
    """Return True if the local P2P orchestrator reports this node as the effective leader."""
    if not USE_P2P_ORCHESTRATOR:
        return True

    try:
        import aiohttp
    except Exception:
        # Best-effort: if aiohttp is unavailable, assume we're allowed to run
        # (daemon can still function in non-P2P mode).
        return True

    url = f"{P2P_ORCHESTRATOR_URL.rstrip('/')}/status"
    headers: Dict[str, str] = {}
    if P2P_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {P2P_AUTH_TOKEN}"

    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
    except Exception:
        return False

    node_id = data.get("node_id")
    role = data.get("role")
    effective_leader_id = data.get("effective_leader_id")
    return bool(node_id and role == "leader" and effective_leader_id == node_id)

# =============================================================================
# Curriculum Learning Configuration
# =============================================================================
#
# Curriculum learning progressively increases training difficulty:
# - Stage 0 (Beginner): High exploration, simple positions
# - Stage 1 (Intermediate): Medium exploration, moderate positions
# - Stage 2 (Advanced): Low exploration, complex positions
#
# Each stage has different parameters for temperature, noise, and game complexity

CURRICULUM_STAGES = {
    0: {  # Beginner
        "name": "beginner",
        "temperature": 2.0,           # High exploration
        "noise_scale": 0.5,           # Moderate noise
        "max_moves": 200,             # Shorter games
        "games_to_advance": 2000,     # Games needed to advance
        "description": "High exploration, shorter games",
    },
    1: {  # Intermediate
        "name": "intermediate",
        "temperature": 1.0,           # Balanced exploration
        "noise_scale": 0.25,          # Lower noise
        "max_moves": 300,             # Medium length games
        "games_to_advance": 5000,     # Games needed to advance
        "description": "Balanced exploration, medium games",
    },
    2: {  # Advanced
        "name": "advanced",
        "temperature": 0.5,           # Low exploration (exploit learned knowledge)
        "noise_scale": 0.1,           # Minimal noise
        "max_moves": 400,             # Full length games
        "games_to_advance": None,     # Final stage
        "description": "Low exploration, full games",
    },
}

CURRICULUM_ENABLED = os.environ.get("RINGRIFT_CURRICULUM_LEARNING", "1").lower() in (
    "1", "true", "yes", "on"
)


def get_curriculum_params(state: "DaemonState") -> Dict[str, Any]:
    """Get curriculum parameters for current stage."""
    if not CURRICULUM_ENABLED:
        return CURRICULUM_STAGES[2]  # Use advanced params if curriculum disabled

    stage = state.curriculum_state.get("stage", 0)
    if stage >= len(CURRICULUM_STAGES):
        stage = len(CURRICULUM_STAGES) - 1

    return CURRICULUM_STAGES[stage]


def maybe_advance_curriculum(state: "DaemonState", games_this_cycle: int) -> bool:
    """Check if curriculum should advance to next stage."""
    if not CURRICULUM_ENABLED:
        return False

    stage = state.curriculum_state.get("stage", 0)
    if stage >= len(CURRICULUM_STAGES) - 1:
        return False  # Already at final stage

    params = CURRICULUM_STAGES[stage]
    games_to_advance = params.get("games_to_advance")
    if games_to_advance is None:
        return False

    state.curriculum_state["games_at_stage"] = (
        state.curriculum_state.get("games_at_stage", 0) + games_this_cycle
    )

    if state.curriculum_state["games_at_stage"] >= games_to_advance:
        # Advance to next stage
        old_stage = stage
        new_stage = stage + 1
        state.curriculum_state["stage"] = new_stage
        state.curriculum_state["games_at_stage"] = 0
        state.curriculum_state["stage_transitions"] = state.curriculum_state.get(
            "stage_transitions", []
        ) + [{
            "from": old_stage,
            "to": new_stage,
            "total_games": state.total_games_generated,
            "timestamp": datetime.now().isoformat(),
        }]
        print(f"[Daemon] Curriculum advanced: {CURRICULUM_STAGES[old_stage]['name']} -> "
              f"{CURRICULUM_STAGES[new_stage]['name']}")
        return True

    return False


def run_s3_backup(models_only: bool = True) -> bool:
    """Run S3 backup if enabled and interval has passed."""
    global _last_s3_backup_time

    if not S3_BACKUP_ENABLED:
        return True

    current_time = time.time()
    if _last_s3_backup_time is not None:
        hours_since_backup = (current_time - _last_s3_backup_time) / 3600
        if hours_since_backup < S3_BACKUP_INTERVAL_HOURS:
            return True  # Skip, not yet time

    print(f"[Daemon] Running S3 backup (models_only={models_only})...")
    try:
        cmd = [sys.executable, str(AI_SERVICE_ROOT / "scripts" / "s3_backup.py")]
        if models_only:
            cmd.append("--models-only")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"[Daemon] S3 backup completed successfully")
            _last_s3_backup_time = current_time
            return True
        else:
            print(f"[Daemon] S3 backup failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[Daemon] S3 backup error: {e}")
        return False


async def get_p2p_cluster_status() -> Optional[Dict[str, Any]]:
    """Query P2P orchestrator for cluster status and data manifest."""
    if not USE_P2P_ORCHESTRATOR:
        return None

    try:
        import aiohttp
        headers = {"Authorization": f"Bearer {P2P_AUTH_TOKEN}"} if P2P_AUTH_TOKEN else {}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{P2P_ORCHESTRATOR_URL}/cluster_data_manifest?refresh=0",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as e:
        print(f"[Daemon] P2P orchestrator query failed: {e}")
    return None


async def notify_p2p_training_complete(model_id: str, board_type: str, num_players: int) -> bool:
    """Notify P2P orchestrator that training completed (for coordinated promotion)."""
    if not USE_P2P_ORCHESTRATOR:
        return True

    try:
        import aiohttp
        headers = {"Authorization": f"Bearer {P2P_AUTH_TOKEN}"} if P2P_AUTH_TOKEN else {}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{P2P_ORCHESTRATOR_URL}/improvement/phase_complete",
                headers=headers,
                json={
                    "phase": "training",
                    "model_id": model_id,
                    "board_type": board_type,
                    "num_players": num_players,
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                return resp.status == 200
    except Exception as e:
        print(f"[Daemon] P2P notification failed: {e}")
    return False


def maybe_sync_staging(reason: str) -> bool:
    """Best-effort: push promoted models/weights to staging via SSH."""
    if not SYNC_STAGING:
        return False

    if not os.environ.get("RINGRIFT_STAGING_SSH_HOST") or not os.environ.get("RINGRIFT_STAGING_ROOT"):
        print(
            f"[Daemon] Staging sync requested ({reason}) but missing "
            "RINGRIFT_STAGING_SSH_HOST / RINGRIFT_STAGING_ROOT"
        )
        return False

    cmd = [sys.executable, "scripts/sync_staging_ai_artifacts.py"]
    if SYNC_STAGING_RESTART:
        cmd.append("--restart")
        services = os.environ.get("RINGRIFT_STAGING_RESTART_SERVICES")
        if services:
            cmd.extend(["--restart-services", services])
    if SYNC_STAGING_VALIDATE_HEALTH:
        cmd.append("--validate-health")
    if SYNC_STAGING_FAIL_ON_MISSING:
        cmd.append("--fail-on-missing")

    success, output = run_command(cmd, timeout=900)
    if success:
        print(f"[Daemon] Synced AI artifacts to staging ({reason})")
        return True

    print(f"[Daemon] Staging sync failed ({reason}): {output[:200]}")
    return False


# =============================================================================
# Configuration
# =============================================================================

# Board type configurations for balanced training
BOARD_CONFIGS = [
    {"board": "square8", "players": 2, "priority": 1.0, "min_games": 2000},
    {"board": "square8", "players": 3, "priority": 0.3, "min_games": 1000},
    {"board": "square8", "players": 4, "priority": 0.2, "min_games": 500},
    {"board": "square19", "players": 2, "priority": 0.5, "min_games": 1500},
    {"board": "square19", "players": 3, "priority": 0.2, "min_games": 800},
    {"board": "square19", "players": 4, "priority": 0.1, "min_games": 400},
    {"board": "hexagonal", "players": 2, "priority": 0.4, "min_games": 1500},
    {"board": "hexagonal", "players": 3, "priority": 0.2, "min_games": 800},
    {"board": "hexagonal", "players": 4, "priority": 0.1, "min_games": 400},
]

# Training thresholds - loaded from unified config (single source of truth)
_unified_config = get_config()
MIN_NEW_GAMES_FOR_TRAINING = _unified_config.training.trigger_threshold_games
TRAINING_COOLDOWN_SECONDS = _unified_config.training.min_interval_seconds
TOURNAMENT_GAMES = 50              # Games per model comparison
PROMOTION_THRESHOLD = 0.55         # Win rate needed for promotion

# Adaptive training configuration
# When Elo is improving rapidly, train more frequently
# When Elo is plateauing, train less frequently to save resources
ADAPTIVE_TRAINING_ENABLED = True
ADAPTIVE_MIN_COOLDOWN = 600       # 10 min minimum when improving fast
ADAPTIVE_MAX_COOLDOWN = 1800      # 30 min maximum when plateauing
ADAPTIVE_MIN_GAMES = 300          # Train with fewer games when improving
ADAPTIVE_MAX_GAMES = 800          # Require more games when plateauing
ELO_IMPROVEMENT_THRESHOLD = 20    # Elo gain to consider "improving"
ELO_HISTORY_WINDOW = 5            # Number of recent models to check

# Game count cache for faster threshold checking
GAME_COUNT_CACHE_FILE = AI_SERVICE_ROOT / "logs" / "improvement_daemon" / "game_count_cache.json"
_game_count_cache: Dict[str, Tuple[int, float]] = {}  # path -> (count, mtime)

# Selfplay configuration
SELFPLAY_BATCH_SIZE = 100          # Games per selfplay batch
# Diverse engine modes for richer training data:
# - mixed: Random sampling from D1-D10 ladder (all AI types)
# - descent-only: Pure AlphaZero-style (best for NN training)
# - mcts-only: Monte Carlo tree search (exploration focused)
# - heuristic-only: Pure heuristic (CMA-ES optimization)
# - minimax-only: Traditional search (complements MCTS)
SELFPLAY_ENGINES = ["mixed", "descent-only", "mcts-only", "heuristic-only", "minimax-only"]

# Asymmetric matchup configurations for diverse training data
# These pit different AI types against each other to explore more of the game tree
ASYMMETRIC_MATCHUPS = [
    # Format: (engine1, diff1, engine2, diff2) - creates imbalanced games
    ("heuristic-only", 5, "mcts-only", 6),      # Heuristic vs MCTS
    ("heuristic-only", 4, "descent-only", 7),   # Weak heuristic vs strong descent
    ("minimax-only", 5, "mcts-only", 5),        # Minimax vs MCTS (same level)
    ("random-only", 1, "descent-only", 9),      # Random vs strong (exploration)
    ("mcts-only", 4, "descent-only", 6),        # MCTS vs Descent
]

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5.0
RETRY_MAX_DELAY = 300.0

# Daemon configuration
CYCLE_INTERVAL_SECONDS = 60        # Main loop interval
STATE_FILE = AI_SERVICE_ROOT / "logs" / "improvement_daemon" / "state.json"
PID_FILE = AI_SERVICE_ROOT / "logs" / "improvement_daemon" / "daemon.pid"
LOG_FILE = AI_SERVICE_ROOT / "logs" / "improvement_daemon" / "daemon.log"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelInfo:
    """Information about a trained model."""
    model_id: str
    path: str
    board_type: str
    num_players: int
    iteration: int
    created_at: str
    training_games: int
    elo_rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    is_best: bool = False
    parent_model_id: Optional[str] = None


@dataclass
class BoardTypeState:
    """State for a specific board type configuration."""
    board_type: str
    num_players: int
    total_games: int = 0
    games_since_last_training: int = 0
    last_training_time: float = 0.0
    current_iteration: int = 0
    best_model_id: Optional[str] = None
    models: List[str] = field(default_factory=list)  # List of model IDs


@dataclass
class DaemonState:
    """Complete daemon state for checkpointing."""
    started_at: str = ""
    last_cycle_at: str = ""
    total_cycles: int = 0
    total_games_generated: int = 0
    total_training_runs: int = 0
    total_tournaments: int = 0

    # Per-board-type state
    board_states: Dict[str, BoardTypeState] = field(default_factory=dict)

    # Model registry
    models: Dict[str, ModelInfo] = field(default_factory=dict)

    # Elo ratings (model_id -> rating)
    elo_ratings: Dict[str, float] = field(default_factory=dict)

    # Elo history for graphing
    elo_history: List[Dict[str, Any]] = field(default_factory=list)

    # Error tracking
    consecutive_failures: int = 0
    last_error: str = ""
    last_error_time: str = ""

    # NNUE training tracking (per board_type/num_players config)
    # Maps "square8_2p" -> {"last_train_time": float, "last_train_games": int, "model_path": str}
    nnue_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # CMAES heuristic optimization tracking (per board_type/num_players config)
    # Maps "square8_2p" -> {"last_opt_time": float, "last_opt_games": int, "profile_id": str}
    cmaes_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # NNUE Policy training tracking (per board_type/num_players config)
    # Maps "square8_2p" -> {"last_train_time": float, "last_train_games": int, ...}
    nnue_policy_state: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Curriculum learning state
    # Tracks curriculum stage and parameters for progressive training
    curriculum_state: Dict[str, Any] = field(default_factory=lambda: {
        "stage": 0,           # Current curriculum stage (0=beginner, 1=intermediate, 2=advanced)
        "games_at_stage": 0,  # Games played at current stage
        "total_stages": 3,    # Total number of stages
        "stage_transitions": [],  # History of stage transitions
    })

    # Auto-promotion tracking
    last_auto_promote_time: float = 0.0
    total_auto_promotions: int = 0


# =============================================================================
# State Management
# =============================================================================

def load_state() -> DaemonState:
    """Load daemon state from checkpoint."""
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            state = DaemonState()

            # Load simple fields
            for key in ["started_at", "last_cycle_at", "total_cycles",
                       "total_games_generated", "total_training_runs",
                       "total_tournaments", "consecutive_failures",
                       "last_error", "last_error_time"]:
                if key in data:
                    setattr(state, key, data[key])

            # Load board states
            for key, bs_data in data.get("board_states", {}).items():
                state.board_states[key] = BoardTypeState(**bs_data)

            # Load models
            for model_id, model_data in data.get("models", {}).items():
                state.models[model_id] = ModelInfo(**model_data)

            # Load Elo ratings
            state.elo_ratings = data.get("elo_ratings", {})
            state.elo_history = data.get("elo_history", [])

            # Load NNUE, NNUE Policy, and CMAES state
            state.nnue_state = data.get("nnue_state", {})
            state.nnue_policy_state = data.get("nnue_policy_state", {})
            state.cmaes_state = data.get("cmaes_state", {})
            state.last_auto_promote_time = data.get("last_auto_promote_time", 0.0)
            state.total_auto_promotions = data.get("total_auto_promotions", 0)

            # Load curriculum learning state
            if "curriculum_state" in data:
                state.curriculum_state = data["curriculum_state"]

            return state
        except Exception as e:
            print(f"Warning: Could not load state: {e}")

    return DaemonState(started_at=datetime.now().isoformat())


def save_state(state: DaemonState) -> None:
    """Save daemon state to checkpoint."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable dict
    data = {
        "started_at": state.started_at,
        "last_cycle_at": state.last_cycle_at,
        "total_cycles": state.total_cycles,
        "total_games_generated": state.total_games_generated,
        "total_training_runs": state.total_training_runs,
        "total_tournaments": state.total_tournaments,
        "consecutive_failures": state.consecutive_failures,
        "last_error": state.last_error,
        "last_error_time": state.last_error_time,
        "board_states": {k: asdict(v) for k, v in state.board_states.items()},
        "models": {k: asdict(v) for k, v in state.models.items()},
        "elo_ratings": state.elo_ratings,
        "elo_history": state.elo_history,
        "nnue_state": state.nnue_state,
        "nnue_policy_state": state.nnue_policy_state,
        "cmaes_state": state.cmaes_state,
        "curriculum_state": state.curriculum_state,
        "last_auto_promote_time": state.last_auto_promote_time,
        "total_auto_promotions": state.total_auto_promotions,
    }

    STATE_FILE.write_text(json.dumps(data, indent=2))


# =============================================================================
# Elo Rating System
# =============================================================================

def update_elo(rating_a: float, rating_b: float, score_a: float, k: float = 32.0) -> Tuple[float, float]:
    """Update Elo ratings based on match result.

    Args:
        rating_a: Current rating of player A
        rating_b: Current rating of player B
        score_a: Score for A (1.0 = win, 0.5 = draw, 0.0 = loss)
        k: K-factor (higher = more volatile ratings)

    Returns:
        Tuple of (new_rating_a, new_rating_b)
    """
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a

    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)

    return new_a, new_b


def record_elo_snapshot(state: DaemonState) -> None:
    """Record current Elo ratings for history tracking."""
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "cycle": state.total_cycles,
        "ratings": dict(state.elo_ratings),
    }
    state.elo_history.append(snapshot)

    # Keep last 1000 snapshots
    if len(state.elo_history) > 1000:
        state.elo_history = state.elo_history[-1000:]


# =============================================================================
# Core Operations
# =============================================================================

def get_config_key(board: str, players: int) -> str:
    """Get unique key for board/player config."""
    return f"{board}_{players}p"


def run_command(cmd: List[str], cwd: Path = AI_SERVICE_ROOT, timeout: int = 3600) -> Tuple[bool, str]:
    """Run a command with timeout and capture output."""
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(AI_SERVICE_ROOT)
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


async def run_command_async(cmd: List[str], cwd: Path = AI_SERVICE_ROOT, timeout: int = 3600) -> Tuple[bool, str]:
    """Run a command asynchronously with timeout and capture output.

    Non-blocking version of run_command using asyncio subprocess.
    """
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(AI_SERVICE_ROOT)
        env["RINGRIFT_SKIP_SHADOW_CONTRACTS"] = "true"

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            output = (stdout.decode() if stdout else "") + (stderr.decode() if stderr else "")
            return process.returncode == 0, output
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return False, "Command timed out"
    except Exception as e:
        return False, str(e)


# Semaphore for parallel selfplay (limit concurrent processes)
PARALLEL_SELFPLAY_LIMIT = 3  # Max concurrent selfplay processes
_selfplay_semaphore: Optional[asyncio.Semaphore] = None


def get_selfplay_semaphore() -> asyncio.Semaphore:
    """Get or create the selfplay semaphore."""
    global _selfplay_semaphore
    if _selfplay_semaphore is None:
        _selfplay_semaphore = asyncio.Semaphore(PARALLEL_SELFPLAY_LIMIT)
    return _selfplay_semaphore


# =============================================================================
# Load-based Job Throttling
# =============================================================================
# Prevents spawning new jobs when system is overloaded

# Max load average (relative to CPU count) before refusing new jobs
MAX_LOAD_FACTOR = float(os.environ.get("RINGRIFT_MAX_LOAD_FACTOR", "2.0"))
# Absolute max load average regardless of CPU count
MAX_LOAD_ABSOLUTE = float(os.environ.get("RINGRIFT_MAX_LOAD_ABSOLUTE", "100.0"))
# How long to wait when overloaded before rechecking (seconds)
LOAD_BACKOFF_SECONDS = float(os.environ.get("RINGRIFT_LOAD_BACKOFF_SECONDS", "30.0"))


def get_system_load() -> Tuple[float, float, float]:
    """Get system load averages (1min, 5min, 15min)."""
    try:
        return os.getloadavg()
    except (OSError, AttributeError):
        # Windows or other systems without getloadavg
        return (0.0, 0.0, 0.0)


def get_cpu_count() -> int:
    """Get number of CPU cores."""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def is_system_overloaded(verbose: bool = False) -> bool:
    """Check if system is too overloaded to spawn new jobs.

    Returns True if:
    - Load average exceeds MAX_LOAD_FACTOR * cpu_count, OR
    - Load average exceeds MAX_LOAD_ABSOLUTE

    This prevents runaway job spawning on overloaded systems.
    """
    load_1min, load_5min, _ = get_system_load()
    cpu_count = get_cpu_count()

    relative_threshold = MAX_LOAD_FACTOR * cpu_count

    # Use 5-minute average to avoid reacting to brief spikes
    is_overloaded = load_5min > relative_threshold or load_5min > MAX_LOAD_ABSOLUTE

    if is_overloaded and verbose:
        print(f"[Daemon] System overloaded: load={load_5min:.1f}, "
              f"threshold={min(relative_threshold, MAX_LOAD_ABSOLUTE):.1f} "
              f"(CPUs={cpu_count}, factor={MAX_LOAD_FACTOR})")

    return is_overloaded


async def wait_for_load_decrease(max_wait_seconds: float = 300.0, verbose: bool = True) -> bool:
    """Wait for system load to decrease before proceeding.

    Returns True if load decreased within timeout, False otherwise.
    """
    start_time = time.time()

    while is_system_overloaded(verbose=False):
        elapsed = time.time() - start_time
        if elapsed >= max_wait_seconds:
            if verbose:
                load_1min, load_5min, _ = get_system_load()
                print(f"[Daemon] Load still high after {elapsed:.0f}s wait "
                      f"(load={load_5min:.1f}). Giving up.")
            return False

        if verbose:
            load_1min, load_5min, _ = get_system_load()
            print(f"[Daemon] Waiting for load to decrease "
                  f"(current={load_5min:.1f}, waited={elapsed:.0f}s)...")

        await asyncio.sleep(LOAD_BACKOFF_SECONDS)

    return True


def count_games_in_jsonl(path: Path, use_cache: bool = True) -> int:
    """Count games in a JSONL file with optional caching.

    Uses modification time to invalidate cache entries, avoiding
    re-reading unchanged files on every cycle.
    """
    global _game_count_cache

    if not path.exists():
        return 0

    path_str = str(path)

    try:
        current_mtime = path.stat().st_mtime

        # Check cache if enabled
        if use_cache and path_str in _game_count_cache:
            cached_count, cached_mtime = _game_count_cache[path_str]
            if cached_mtime >= current_mtime:
                return cached_count

        # Count lines (each line is one game)
        with open(path) as f:
            count = sum(1 for _ in f)

        # Update cache
        _game_count_cache[path_str] = (count, current_mtime)

        return count
    except Exception:
        return 0


def load_game_count_cache() -> None:
    """Load game count cache from disk."""
    global _game_count_cache
    try:
        if GAME_COUNT_CACHE_FILE.exists():
            with open(GAME_COUNT_CACHE_FILE) as f:
                data = json.load(f)
                _game_count_cache = {k: tuple(v) for k, v in data.items()}
    except Exception:
        _game_count_cache = {}


def save_game_count_cache() -> None:
    """Save game count cache to disk."""
    try:
        GAME_COUNT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(GAME_COUNT_CACHE_FILE, "w") as f:
            json.dump({k: list(v) for k, v in _game_count_cache.items()}, f)
    except Exception:
        pass


def get_adaptive_training_thresholds(state: DaemonState, config_key: str) -> Tuple[int, int]:
    """Compute adaptive training thresholds based on Elo improvement velocity.

    Returns (min_games_threshold, cooldown_seconds).

    When recent models show Elo improvement:
    - Lower game threshold (train sooner)
    - Shorter cooldown (train more frequently)

    When Elo is plateauing or declining:
    - Higher game threshold (accumulate more data)
    - Longer cooldown (train less frequently)
    """
    if not ADAPTIVE_TRAINING_ENABLED:
        return MIN_NEW_GAMES_FOR_TRAINING, TRAINING_COOLDOWN_SECONDS

    # Get recent models for this config
    recent_models = []
    for model_id, model_info in state.models.items():
        key = get_config_key(model_info.board_type, model_info.num_players)
        if key == config_key:
            recent_models.append(model_info)

    # Sort by creation time (newest first)
    recent_models.sort(key=lambda m: m.created_at, reverse=True)
    recent_models = recent_models[:ELO_HISTORY_WINDOW]

    if len(recent_models) < 2:
        # Not enough history, use defaults
        return MIN_NEW_GAMES_FOR_TRAINING, TRAINING_COOLDOWN_SECONDS

    # Calculate Elo velocity (change in best Elo over recent models)
    elos = [m.elo_rating for m in recent_models]
    if len(elos) >= 2:
        # Compare newest to oldest in window
        elo_change = elos[0] - elos[-1]
    else:
        elo_change = 0

    # Determine if improving, plateauing, or declining
    if elo_change >= ELO_IMPROVEMENT_THRESHOLD:
        # Improving: train more aggressively
        improvement_factor = min(elo_change / ELO_IMPROVEMENT_THRESHOLD, 2.0)  # Cap at 2x
        min_games = int(ADAPTIVE_MIN_GAMES + (MIN_NEW_GAMES_FOR_TRAINING - ADAPTIVE_MIN_GAMES) * (1 - improvement_factor / 2))
        cooldown = int(ADAPTIVE_MIN_COOLDOWN + (TRAINING_COOLDOWN_SECONDS - ADAPTIVE_MIN_COOLDOWN) * (1 - improvement_factor / 2))
        print(f"[Daemon] Adaptive training: IMPROVING (+{elo_change:.0f} Elo), min_games={min_games}, cooldown={cooldown}s")
    elif elo_change <= -ELO_IMPROVEMENT_THRESHOLD:
        # Declining: accumulate more data before training
        min_games = ADAPTIVE_MAX_GAMES
        cooldown = ADAPTIVE_MAX_COOLDOWN
        print(f"[Daemon] Adaptive training: DECLINING ({elo_change:.0f} Elo), min_games={min_games}, cooldown={cooldown}s")
    else:
        # Plateauing: use standard thresholds
        min_games = MIN_NEW_GAMES_FOR_TRAINING
        cooldown = TRAINING_COOLDOWN_SECONDS
        print(f"[Daemon] Adaptive training: STABLE ({elo_change:+.0f} Elo), using defaults")

    return min_games, cooldown


def get_training_data_stats(state: DaemonState) -> Dict[str, Dict[str, int]]:
    """Get training data statistics for each board type."""
    stats = {}

    selfplay_dir = AI_SERVICE_ROOT / "data" / "selfplay"
    training_dir = AI_SERVICE_ROOT / "data" / "training"

    for config in BOARD_CONFIGS:
        key = get_config_key(config["board"], config["players"])

        # Count JSONL games
        jsonl_games = 0
        for jsonl_file in selfplay_dir.rglob(f"*{config['board']}*{config['players']}p*.jsonl"):
            jsonl_games += count_games_in_jsonl(jsonl_file)

        # Check NPZ training files
        npz_size = 0
        for npz_file in training_dir.rglob(f"*{config['board']}*{config['players']}p*.npz"):
            npz_size += npz_file.stat().st_size if npz_file.exists() else 0

        stats[key] = {
            "jsonl_games": jsonl_games,
            "npz_size_mb": npz_size / (1024 * 1024),
            "min_games": config["min_games"],
        }

    return stats


async def run_asymmetric_selfplay(state: DaemonState, board_type: str, num_players: int) -> int:
    """Run asymmetric selfplay games between different AI types.

    This generates diverse training data by pitting different AI algorithms
    against each other, exploring more of the game tree than homogeneous selfplay.

    Note: Currently only supports 2-player games as run_ai_tournament.py only has --p1/--p2.
    """
    total_games = 0
    key = get_config_key(board_type, num_players)

    # Only support 2-player asymmetric games for now
    if num_players != 2:
        return 0

    # Select a random asymmetric matchup
    if not ASYMMETRIC_MATCHUPS:
        return 0

    matchup = random.choice(ASYMMETRIC_MATCHUPS)
    engine1, diff1, engine2, diff2 = matchup

    output_file = AI_SERVICE_ROOT / "data" / "selfplay" / f"daemon_{key}_asymmetric" / f"games_{int(time.time())}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use run_ai_tournament.py for asymmetric matchups since run_self_play_soak.py
    # uses homogeneous engine modes. Map engine modes to AI types.
    ai_type_map = {
        "heuristic-only": "Heuristic",
        "minimax-only": "Minimax",
        "mcts-only": "MCTS",
        "descent-only": "MCTS",  # Descent uses MCTS with neural net
        "random-only": "Random",
    }

    p1_type = ai_type_map.get(engine1, "Heuristic")
    p2_type = ai_type_map.get(engine2, "Heuristic")

    # Run 10 asymmetric games per matchup
    cmd = [
        sys.executable, "scripts/run_ai_tournament.py",
        "--p1", p1_type,
        "--p1-diff", str(diff1),
        "--p2", p2_type,
        "--p2-diff", str(diff2),
        "--board", {"square8": "Square8", "square19": "Square19", "hexagonal": "Hex"}.get(board_type, "Square8"),
        "--games", "10",
        "--output-dir", str(output_file.parent),
    ]

    print(f"[Daemon] Running asymmetric selfplay: {p1_type}(D{diff1}) vs {p2_type}(D{diff2}) on {key}")
    success, output = run_command(cmd, timeout=1800)  # 30 min for 10 games on complex boards

    if success:
        total_games = 10  # Tournament runs specified number of games
        print(f"[Daemon] Asymmetric selfplay completed: {total_games} diverse games")
    else:
        print(f"[Daemon] Asymmetric selfplay failed: {output[:200]}")

    return total_games


async def run_single_selfplay_job(
    board_type: str,
    num_players: int,
    num_games: int,
    engine_mode: str,
    output_file: Path,
    timeout: int,
) -> Tuple[str, int, bool, str]:
    """Run a single selfplay job with semaphore for rate limiting.

    Returns (config_key, games_generated, success, output).
    """
    key = get_config_key(board_type, num_players)

    # Check system load before acquiring semaphore
    if is_system_overloaded(verbose=True):
        # Wait up to 5 minutes for load to decrease
        if not await wait_for_load_decrease(max_wait_seconds=300.0, verbose=True):
            return key, 0, False, "System overloaded - skipping job"

    semaphore = get_selfplay_semaphore()

    async with semaphore:
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Opening randomization: more moves for larger boards and more players
        opening_random = {
            ("square8", 2): 4,
            ("square8", 3): 6,
            ("square8", 4): 8,
            ("square19", 2): 6,
            ("square19", 3): 9,
            ("square19", 4): 12,
            ("hexagonal", 2): 6,
            ("hexagonal", 3): 9,
            ("hexagonal", 4): 12,
        }.get((board_type, num_players), 4)

        cmd = [
            sys.executable, "scripts/run_self_play_soak.py",
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--num-games", str(num_games),
            "--engine-mode", engine_mode,
            "--log-jsonl", str(output_file),
            "--max-moves", "10000",
            "--opening-random-moves", str(opening_random),
        ]
        # Add swap-sides probability for 2-player games to balance P1/P2 win rates
        if num_players == 2:
            cmd.extend(["--swap-sides-probability", "0.5"])

        success, output = await run_command_async(cmd, timeout=timeout)

        if success:
            games_generated = count_games_in_jsonl(output_file)
            return key, games_generated, True, output
        else:
            return key, 0, False, output


async def run_parallel_selfplay(
    configs: List[Dict[str, Any]],
    state: DaemonState,
    duration_minutes: int = 10,
) -> int:
    """Run selfplay for multiple configs in parallel.

    Uses asyncio.gather to run up to PARALLEL_SELFPLAY_LIMIT concurrent jobs.
    """
    if not configs:
        return 0

    timeout = duration_minutes * 60
    tasks = []

    for config in configs:
        board_type = config["board"]
        num_players = config["players"]
        num_games = config.get("games", 40)
        engine_mode = config.get("engine", random.choice(SELFPLAY_ENGINES))
        key = get_config_key(board_type, num_players)

        output_file = (
            AI_SERVICE_ROOT / "data" / "selfplay" / f"daemon_{key}"
            / f"games_{int(time.time())}_{random.randint(1000, 9999)}.jsonl"
        )

        tasks.append(
            run_single_selfplay_job(
                board_type, num_players, num_games, engine_mode, output_file, timeout
            )
        )

    print(f"[Daemon] Running {len(tasks)} selfplay jobs in parallel (limit: {PARALLEL_SELFPLAY_LIMIT})...")

    # Run all jobs (semaphore limits concurrency)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total_games = 0
    for result in results:
        if isinstance(result, Exception):
            print(f"[Daemon] Selfplay job failed with exception: {result}")
            continue

        key, games, success, output = result
        if success and games > 0:
            total_games += games

            # Update state
            board_type, num_players_str = key.rsplit("_", 1)
            num_players = int(num_players_str.rstrip("p"))
            if key not in state.board_states:
                state.board_states[key] = BoardTypeState(board_type, num_players)
            state.board_states[key].total_games += games
            state.board_states[key].games_since_last_training += games

            print(f"[Daemon] Parallel: {key} generated {games} games")
        elif not success:
            print(f"[Daemon] Parallel: {key} failed: {output[:100]}")

    return total_games


async def run_balanced_selfplay(state: DaemonState, duration_minutes: int = 10) -> int:
    """Run selfplay balanced across board types based on priority and need.

    Uses ImprovementCycleManager for diverse AI opponent selection when available.
    Incorporates P2P cluster data manifest if available for smarter balancing.
    Now supports parallel execution for faster data generation.
    """
    # Skip local selfplay if disabled (low-memory machines)
    if DISABLE_LOCAL_TASKS:
        print("[Daemon] Skipping local selfplay (RINGRIFT_DISABLE_LOCAL_TASKS=true)")
        return 0

    # Check backpressure before spawning selfplay (new coordination)
    if HAS_NEW_COORDINATION:
        if should_stop_production(QueueType.TRAINING_DATA):
            print("[Daemon] Skipping selfplay: training queue at STOP backpressure level")
            return 0
        if should_throttle_production(QueueType.TRAINING_DATA):
            throttle_factor = get_throttle_factor(QueueType.TRAINING_DATA)
            print(f"[Daemon] Backpressure active: throttle factor {throttle_factor:.2f}")
            # Skip selfplay probabilistically based on throttle factor
            import random
            if random.random() > throttle_factor:
                print("[Daemon] Skipping selfplay due to backpressure (probabilistic)")
                return 0

    total_games = 0

    # Get improvement manager for diverse config selection
    manager = get_improvement_manager()

    # Query P2P orchestrator for cluster-wide data manifest
    cluster_manifest = await get_p2p_cluster_status()
    cluster_games_by_config = {}
    if cluster_manifest and "by_board_type" in cluster_manifest:
        for board_key, board_data in cluster_manifest["by_board_type"].items():
            cluster_games_by_config[board_key] = board_data.get("total_games", 0)
        print(f"[Daemon] Using cluster manifest: {sum(cluster_games_by_config.values())} total games across cluster")

    # Use ImprovementCycleManager for diverse configs when available
    if manager:
        # Get a batch of diverse selfplay configs
        diverse_configs = manager.get_diverse_selfplay_batch(batch_size=5)
        print(f"[Daemon] Using ImprovementCycleManager: {len(diverse_configs)} diverse configs")

        for config in diverse_configs:
            board_type = config.get("board_type", "square8")
            num_players = config.get("num_players", 2)
            engine_mode = config.get("engine_mode", "mixed")
            key = get_config_key(board_type, num_players)

            batch_games = max(10, int(SELFPLAY_BATCH_SIZE * 0.4))  # ~40 games per config
            output_file = AI_SERVICE_ROOT / "data" / "selfplay" / f"daemon_{key}" / f"games_{int(time.time())}.jsonl"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Handle asymmetric games
            if config.get("asymmetric"):
                strong = config.get("strong_config", {})
                weak = config.get("weak_config", {})
                print(f"[Daemon] DIVERSE: Asymmetric {key} - "
                      f"Strong({strong.get('engine_mode')}@D{strong.get('difficulty')}) vs "
                      f"Weak({weak.get('engine_mode')}@D{weak.get('difficulty')})")

                # Run asymmetric games via run_ai_tournament.py
                games = await run_asymmetric_selfplay(state, board_type, num_players)
                total_games += games

                # Record to manager
                manager.record_games_completed(board_type, num_players, games, engine_mode)
                continue

            cmd = [
                sys.executable, "scripts/run_self_play_soak.py",
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--num-games", str(batch_games),
                "--engine-mode", engine_mode,
                "--log-jsonl", str(output_file),
                "--max-moves", "10000",
            ]

            print(f"[Daemon] DIVERSE: Running {batch_games} {key} games with {engine_mode}...")
            success, output = run_command(cmd, timeout=duration_minutes * 60)

            if success:
                games_generated = count_games_in_jsonl(output_file)
                total_games += games_generated

                # Update daemon state
                if key not in state.board_states:
                    state.board_states[key] = BoardTypeState(board_type, num_players)
                state.board_states[key].total_games += games_generated
                state.board_states[key].games_since_last_training += games_generated

                # Record to improvement manager
                manager.record_games_completed(board_type, num_players, games_generated, engine_mode)

                print(f"[Daemon] Generated {games_generated} {key} games")
            else:
                print(f"[Daemon] Selfplay failed for {key}: {output[:200]}")

    else:
        # Fallback: Calculate weights based on priority and data deficit
        # Now uses PARALLEL execution for faster data generation
        weights = []
        for config in BOARD_CONFIGS:
            key = get_config_key(config["board"], config["players"])
            bs = state.board_states.get(key, BoardTypeState(config["board"], config["players"]))

            # Use cluster-wide game count if available, else local count
            total_games_for_config = cluster_games_by_config.get(key, bs.total_games)

            # Higher weight if we have less data relative to minimum
            deficit_ratio = max(0, 1 - total_games_for_config / config["min_games"])
            weight = config["priority"] * (1 + deficit_ratio)
            weights.append((config, weight))

        # Normalize weights
        total_weight = sum(w for _, w in weights)
        weights = [(c, w / total_weight) for c, w in weights]

        # Build list of configs to run in parallel
        parallel_configs = []
        for config, weight in weights:
            if weight < 0.05:  # Skip very low weight configs
                continue

            batch_games = max(10, int(SELFPLAY_BATCH_SIZE * weight * 2))
            engine = random.choice(SELFPLAY_ENGINES)

            parallel_configs.append({
                "board": config["board"],
                "players": config["players"],
                "games": batch_games,
                "engine": engine,
            })

        # Run all configs in parallel (with semaphore limiting concurrency)
        if parallel_configs:
            parallel_games = await run_parallel_selfplay(parallel_configs, state, duration_minutes)
            total_games += parallel_games

    state.total_games_generated += total_games
    return total_games


# GPU selfplay configuration
GPU_SELFPLAY_BATCH_SIZE = 100
GPU_SELFPLAY_MAX_MOVES = 400


async def run_gpu_policy_selfplay(state: DaemonState, games_per_config: int = 50) -> int:
    """Run GPU-accelerated selfplay with policy-guided move selection.

    Uses curriculum learning parameters to control exploration/exploitation balance.
    """
    total_games = 0

    # Get curriculum parameters for current stage
    curriculum_params = get_curriculum_params(state)
    temperature = curriculum_params.get("temperature", 1.0)
    noise_scale = curriculum_params.get("noise_scale", 0.1)
    curriculum_max_moves = curriculum_params.get("max_moves", GPU_SELFPLAY_MAX_MOVES)
    stage_name = curriculum_params.get("name", "unknown")

    if CURRICULUM_ENABLED:
        print(f"[Daemon] Curriculum stage: {stage_name} (temperature={temperature:.1f}, max_moves={curriculum_max_moves})")

    for config in BOARD_CONFIGS:
        key = get_config_key(config["board"], config["players"])
        policy_model_path = AI_SERVICE_ROOT / "models" / "nnue" / f"nnue_policy_{config['board']}_{config['players']}p.pt"
        if not policy_model_path.exists():
            continue

        output_db = AI_SERVICE_ROOT / "data" / "games" / f"gpu_policy_{key}.db"
        output_dir = AI_SERVICE_ROOT / "data" / "selfplay" / f"daemon_gpu_policy_{key}"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "scripts/run_gpu_selfplay.py",
            "--board", config["board"],
            "--num-players", str(config["players"]),
            "--num-games", str(games_per_config),
            "--batch-size", str(GPU_SELFPLAY_BATCH_SIZE),
            "--max-moves", str(curriculum_max_moves),
            "--use-policy",
            "--policy-model", str(policy_model_path),
            "--output-dir", str(output_dir),
            "--output-db", str(output_db),
            "--temperature", str(temperature),
            "--noise-scale", str(noise_scale),
        ]

        print(f"[Daemon] Running {games_per_config} GPU policy selfplay games for {key}...")
        success, output = run_command(cmd, timeout=1800)

        if success:
            games_generated = games_per_config
            for line in output.split('\n'):
                if 'Total games:' in line:
                    try:
                        games_generated = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                    break

            total_games += games_generated
            if key not in state.board_states:
                state.board_states[key] = BoardTypeState(config["board"], config["players"])
            state.board_states[key].total_games += games_generated
            state.board_states[key].games_since_last_training += games_generated
            print(f"[Daemon] Generated {games_generated} GPU policy games for {key}")
        else:
            print(f"[Daemon] GPU policy selfplay failed for {key}: {output[:200]}")

    state.total_games_generated += total_games

    # Check if curriculum should advance
    if total_games > 0 and CURRICULUM_ENABLED:
        maybe_advance_curriculum(state, total_games)

    return total_games


async def check_and_run_training(state: DaemonState) -> List[str]:
    """Check if any board type needs training and run it."""
    # Skip local training if disabled (low-memory machines)
    if DISABLE_LOCAL_TASKS:
        return []

    # Skip if system is overloaded
    if is_system_overloaded(verbose=True):
        print("[Daemon] Skipping training check - system overloaded")
        return []

    trained_models = []
    current_time = time.time()

    for config in BOARD_CONFIGS:
        key = get_config_key(config["board"], config["players"])
        bs = state.board_states.get(key)

        if not bs:
            continue

        # Get adaptive thresholds based on Elo improvement velocity
        min_games_threshold, cooldown_threshold = get_adaptive_training_thresholds(state, key)

        # Check if training is needed
        needs_training = (
            bs.games_since_last_training >= min_games_threshold and
            (current_time - bs.last_training_time) >= cooldown_threshold and
            bs.total_games >= config["min_games"]
        )

        if not needs_training:
            continue

        print(f"[Daemon] Training needed for {key}: {bs.games_since_last_training} new games (threshold: {min_games_threshold})")

        # Export training data
        # For hex boards, use V3 encoder (16 channels) for compatibility with HexNeuralNet_v3
        # For square boards, use default encoder (works with both V2 and V3 models)
        data_output_path = AI_SERVICE_ROOT / "data" / "training" / f"daemon_{key}.npz"

        if config["board"] == "hexagonal":
            # Use export_replay_dataset.py with V3 encoder for hex boards
            # Find available game databases
            game_dbs = list((AI_SERVICE_ROOT / "data" / "games").glob("*.db"))
            if game_dbs:
                export_cmd = [
                    sys.executable, "scripts/export_replay_dataset.py",
                    "--db", str(game_dbs[0]),  # Use first available DB
                    "--output", str(data_output_path),
                    "--board-type", config["board"],
                    "--num-players", str(config["players"]),
                    "--encoder-version", "v3",
                    "--sample-every", "2",
                ]
            else:
                # Fallback to jsonl if no DBs available
                export_cmd = [
                    sys.executable, "scripts/jsonl_to_npz.py",
                    "--input-dir", str(AI_SERVICE_ROOT / "data" / "selfplay"),
                    "--output", str(data_output_path),
                    "--board-type", config["board"],
                    "--num-players", str(config["players"]),
                ]
        else:
            # Square boards use jsonl_to_npz.py (produces 56 channels, compatible with V3)
            export_cmd = [
                sys.executable, "scripts/jsonl_to_npz.py",
                "--input-dir", str(AI_SERVICE_ROOT / "data" / "selfplay"),
                "--output", str(data_output_path),
                "--board-type", config["board"],
                "--num-players", str(config["players"]),
            ]

        success, output = run_command(export_cmd, timeout=1200)  # Increased timeout for V3 export
        if not success:
            print(f"[Daemon] Export failed for {key}: {output[:200]}")
            continue

        # Train neural model (NN policy/value) from the exported NPZ dataset.
        #
        # NOTE: This daemon previously invoked a legacy `scripts/train_neural_net.py`
        # entrypoint which has been superseded by the unified training stack
        # (`app.training.train`) and the small CLI wrapper
        # `scripts/run_nn_training_baseline.py`.
        iteration = bs.current_iteration + 1
        model_id = f"{key}_iter{iteration}"
        model_path = AI_SERVICE_ROOT / "models" / f"{model_id}.pth"
        # data_output_path defined above during export

        run_dir = (
            AI_SERVICE_ROOT
            / "logs"
            / "nn_training"
            / f"{key}_iter{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        # Use v3 for all board types (hex uses enhanced encoder, square uses spatial policy heads)
        model_version = "v3"

        train_cmd = [
            sys.executable,
            "scripts/run_nn_training_baseline.py",
            "--board",
            config["board"],
            "--num-players",
            str(config["players"]),
            "--run-dir",
            str(run_dir),
            "--model-id",
            model_id,
            "--data-path",
            str(data_output_path),
            "--epochs", "50",
            "--model-version",
            model_version,
        ]

        print(f"[Daemon] Training {model_id}...")
        success, output = run_command(train_cmd, timeout=3600)

        if success and model_path.exists():
            # Register model
            model_info = ModelInfo(
                model_id=model_id,
                path=str(model_path),
                board_type=config["board"],
                num_players=config["players"],
                iteration=iteration,
                created_at=datetime.now().isoformat(),
                training_games=bs.total_games,
                parent_model_id=bs.best_model_id,
            )
            state.models[model_id] = model_info
            state.elo_ratings[model_id] = 1500.0

            # Update board state
            bs.current_iteration = iteration
            bs.last_training_time = current_time
            bs.games_since_last_training = 0
            bs.models.append(model_id)

            trained_models.append(model_id)
            state.total_training_runs += 1

            print(f"[Daemon] Trained {model_id} successfully")
        else:
            print(f"[Daemon] Training failed for {key}: {output[:200]}")

    return trained_models


async def run_model_comparison(state: DaemonState, new_model_id: str) -> Optional[Dict[str, Any]]:
    """Run tournament comparing new model against current best."""
    model_info = state.models.get(new_model_id)
    if not model_info:
        return None

    key = get_config_key(model_info.board_type, model_info.num_players)
    bs = state.board_states.get(key)

    if not bs or not bs.best_model_id:
        # No previous best - this model becomes best by default
        model_info.is_best = True
        bs.best_model_id = new_model_id
        print(f"[Daemon] {new_model_id} is first model for {key}, setting as best")
        return {"new_model": new_model_id, "promoted": True, "reason": "first_model"}

    best_model = state.models.get(bs.best_model_id)
    if not best_model:
        model_info.is_best = True
        bs.best_model_id = new_model_id
        return {"new_model": new_model_id, "promoted": True, "reason": "best_model_missing"}

    print(f"[Daemon] Running tournament: {new_model_id} vs {bs.best_model_id}")

    # Run tournament
    tournament_cmd = [
        sys.executable, "scripts/run_tournament.py",
        "--player1", f"nn:{model_info.path}",
        "--player2", f"nn:{best_model.path}",
        "--board", model_info.board_type,
        "--num-players", str(model_info.num_players),
        "--games", str(TOURNAMENT_GAMES),
        "--output", str(AI_SERVICE_ROOT / "results" / "daemon_tournaments" / f"{new_model_id}_vs_{bs.best_model_id}.json"),
    ]

    success, output = run_command(tournament_cmd, timeout=1800)

    if not success:
        print(f"[Daemon] Tournament failed: {output[:200]}")
        return None

    # Parse results
    try:
        # Look for win rate in output
        import re
        match = re.search(r"P1.*?(\d+)/(\d+)", output)
        if match:
            wins = int(match.group(1))
            total = int(match.group(2))
            win_rate = wins / total if total > 0 else 0.5
        else:
            win_rate = 0.5
    except Exception:
        win_rate = 0.5

    # Update Elo ratings (in-memory)
    score = win_rate
    new_rating, best_rating = update_elo(
        state.elo_ratings.get(new_model_id, 1500),
        state.elo_ratings.get(bs.best_model_id, 1500),
        score
    )
    state.elo_ratings[new_model_id] = new_rating
    state.elo_ratings[bs.best_model_id] = best_rating

    # Also persist to SQLite database for cross-model leaderboard
    if HAS_PERSISTENT_ELO:
        try:
            conn = init_elo_database()
            # Register models if not already registered
            register_models(conn, [
                {
                    "model_id": new_model_id,
                    "model_path": model_info.path,
                    "board_type": model_info.board_type,
                    "num_players": model_info.num_players,
                    "version": "v" + str(model_info.iteration),
                    "created_at": time.time(),
                },
                {
                    "model_id": bs.best_model_id,
                    "model_path": best_model.path,
                    "board_type": best_model.board_type,
                    "num_players": best_model.num_players,
                    "version": "v" + str(best_model.iteration),
                    "created_at": time.time(),
                },
            ])
            # Record match results
            for _ in range(int(win_rate * TOURNAMENT_GAMES)):
                update_elo_after_match(
                    conn, new_model_id, bs.best_model_id, new_model_id,
                    model_info.board_type, model_info.num_players,
                    tournament_id=f"daemon_{state.total_tournaments}"
                )
            for _ in range(int((1 - win_rate) * TOURNAMENT_GAMES)):
                update_elo_after_match(
                    conn, new_model_id, bs.best_model_id, bs.best_model_id,
                    model_info.board_type, model_info.num_players,
                    tournament_id=f"daemon_{state.total_tournaments}"
                )
            conn.close()
        except Exception as e:
            print(f"[Daemon] Warning: Could not update persistent Elo DB: {e}")

    # Update model stats
    model_info.elo_rating = new_rating
    model_info.games_played += TOURNAMENT_GAMES
    model_info.wins += int(win_rate * TOURNAMENT_GAMES)
    model_info.losses += int((1 - win_rate) * TOURNAMENT_GAMES)

    best_model.elo_rating = best_rating
    best_model.games_played += TOURNAMENT_GAMES
    best_model.wins += int((1 - win_rate) * TOURNAMENT_GAMES)
    best_model.losses += int(win_rate * TOURNAMENT_GAMES)

    state.total_tournaments += 1
    record_elo_snapshot(state)

    # Check for promotion
    promoted = win_rate >= PROMOTION_THRESHOLD
    if promoted:
        model_info.is_best = True
        best_model.is_best = False
        bs.best_model_id = new_model_id
        print(f"[Daemon] PROMOTED: {new_model_id} ({win_rate:.1%} win rate, Elo {new_rating:.0f})")
    else:
        print(f"[Daemon] Not promoted: {new_model_id} ({win_rate:.1%} < {PROMOTION_THRESHOLD:.1%})")

    return {
        "new_model": new_model_id,
        "best_model": bs.best_model_id,
        "win_rate": win_rate,
        "new_elo": new_rating,
        "best_elo": best_rating,
        "promoted": promoted,
    }


async def run_cross_model_tournament(state: DaemonState, top_n: int = 10, games_per_matchup: int = 4) -> int:
    """Run a round-robin tournament between top models to keep Elo ratings fresh.

    This ensures all models play against each other, not just new vs best.
    Runs every 10 cycles (configured in daemon_cycle).
    Games are saved to JSONL for training data generation.

    Returns number of games played.
    """
    # Skip local tournament if disabled (low-memory machines)
    if DISABLE_LOCAL_TASKS:
        return 0

    if not HAS_PERSISTENT_ELO:
        print("[Daemon] Cross-model tournament requires persistent Elo database")
        return 0

    # Get top N models from each board config
    conn = init_elo_database()
    total_games = 0

    # Run for ALL board configs (not just first 3) to ensure Elo for all combinations
    for config in BOARD_CONFIGS:
        key = get_config_key(config["board"], config["players"])
        leaderboard = get_persistent_leaderboard(
            conn, config["board"], config["players"], limit=top_n
        )

        if len(leaderboard) < 2:
            continue

        print(f"[Daemon] Running cross-model tournament for {key} with {len(leaderboard)} models")

        # Generate matchups (round-robin but limited)
        matchups = []
        for i, entry_a in enumerate(leaderboard):
            for entry_b in leaderboard[i+1:]:
                matchups.append((entry_a["model_id"], entry_b["model_id"]))

        # Limit matchups to avoid excessive runtime
        max_matchups = 15
        if len(matchups) > max_matchups:
            random.shuffle(matchups)
            matchups = matchups[:max_matchups]

        # Run games using Elo tournament script (saves games to JSONL for training)
        # Use run_model_elo_tournament.py which has canonical JSONL format
        # --both-ai-types enables cross-inference matches for comprehensive ratings
        elo_tournament_cmd = [
            sys.executable, "scripts/run_model_elo_tournament.py",
            "--board", config["board"],
            "--players", str(config["players"]),
            "--games", str(games_per_matchup),
            "--top-n", str(top_n),
            "--both-ai-types",  # Test all MCTS/Descent combinations for robust Elo
            "--run",  # Actually run the tournament
        ]

        print(f"[Daemon] Running Elo tournament for {key}...")
        success, output = run_command(elo_tournament_cmd, timeout=1800)  # 30 min timeout

        if success:
            # Parse game count from output
            import re
            match = re.search(r"Total games.*?(\d+)", output)
            if match:
                games_this_config = int(match.group(1))
                total_games += games_this_config
                print(f"[Daemon]   Completed {games_this_config} games for {key}")
            else:
                # Estimate from matchup count
                total_games += min(len(matchups), 15) * games_per_matchup
                print(f"[Daemon]   Tournament completed for {key}")
        else:
            print(f"[Daemon]   Tournament failed for {key}: {output[:200]}")

    conn.close()
    state.total_tournaments += 1
    return total_games


def print_leaderboard(state: DaemonState) -> None:
    """Print current Elo leaderboard."""
    print("\n" + "=" * 60)
    print("ELO LEADERBOARD")
    print("=" * 60)

    sorted_models = sorted(
        state.elo_ratings.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]

    for i, (model_id, elo) in enumerate(sorted_models, 1):
        model = state.models.get(model_id)
        best_marker = " [BEST]" if model and model.is_best else ""
        games = model.games_played if model else 0
        print(f"{i:2}. {model_id}: {elo:.0f} ({games} games){best_marker}")

    print("=" * 60 + "\n")


def print_status(state: DaemonState) -> None:
    """Print daemon status summary."""
    print("\n" + "=" * 70)
    print("CONTINUOUS IMPROVEMENT DAEMON STATUS")
    print("=" * 70)
    print(f"Started: {state.started_at}")
    print(f"Last cycle: {state.last_cycle_at}")
    print(f"Total cycles: {state.total_cycles}")
    print(f"Total games generated: {state.total_games_generated}")
    print(f"Total training runs: {state.total_training_runs}")
    print(f"Total tournaments: {state.total_tournaments}")
    print(f"Consecutive failures: {state.consecutive_failures}")

    print("\n--- Board Type Status ---")
    for key, bs in state.board_states.items():
        best = bs.best_model_id or "none"
        print(f"  {key}: {bs.total_games} games, iter {bs.current_iteration}, best={best}")

    print("\n--- Training Data ---")
    stats = get_training_data_stats(state)
    for key, s in stats.items():
        pct = 100 * s["jsonl_games"] / s["min_games"] if s["min_games"] > 0 else 0
        print(f"  {key}: {s['jsonl_games']} games ({pct:.0f}% of minimum)")

    print("\n--- NNUE Training Status ---")
    for key, nnue in state.nnue_state.items():
        last_train = datetime.fromtimestamp(nnue.get("last_train_time", 0)).strftime("%Y-%m-%d %H:%M") if nnue.get("last_train_time") else "never"
        games = nnue.get("last_train_games", 0)
        print(f"  {key}: last trained {last_train} at {games} games")

    print("\n--- NNUE Policy Training Status ---")
    for key, policy in state.nnue_policy_state.items():
        last_train = datetime.fromtimestamp(policy.get("last_train_time", 0)).strftime("%Y-%m-%d %H:%M") if policy.get("last_train_time") else "never"
        games = policy.get("last_train_games", 0)
        acc = policy.get("accuracy")
        acc_str = f"{acc:.2%}" if acc is not None else "N/A"
        # Include benchmark results
        top1 = policy.get("benchmark_top1")
        top1_str = f", Top-1={top1:.1%}" if top1 is not None else ""
        mean_rank = policy.get("benchmark_mean_rank")
        rank_str = f", rank={mean_rank:.1f}" if mean_rank is not None else ""
        print(f"  {key}: last trained {last_train} at {games} games, accuracy={acc_str}{top1_str}{rank_str}")

    print("\n--- CMAES Heuristic Status ---")
    for key, cmaes in state.cmaes_state.items():
        last_opt = datetime.fromtimestamp(cmaes.get("last_opt_time", 0)).strftime("%Y-%m-%d %H:%M") if cmaes.get("last_opt_time") else "never"
        games = cmaes.get("last_opt_games", 0)
        profile = cmaes.get("profile_id", "none")
        print(f"  {key}: last optimized {last_opt} at {games} games, profile={profile}")

    print("\n--- Auto-Promotion Status ---")
    last_promote = datetime.fromtimestamp(state.last_auto_promote_time).strftime("%Y-%m-%d %H:%M") if state.last_auto_promote_time else "never"
    print(f"  Last promotion check: {last_promote}")
    print(f"  Total auto-promotions: {state.total_auto_promotions}")

    print_leaderboard(state)


# =============================================================================
# NNUE Auto-Training
# =============================================================================

# Minimum new games before considering NNUE retraining
NNUE_MIN_NEW_GAMES = 10000
# Minimum time between NNUE training runs (seconds) = 4 hours
NNUE_MIN_INTERVAL = 4 * 60 * 60
# NNUE training epochs
NNUE_EPOCHS = 30
# NNUE gating parameters (lightweight, report-driven)
# We gate NNUE promotion on validation loss improvements produced by
# scripts/train_nnue.py. Full head-to-head gating can be layered on later,
# but this keeps the continuous daemon deterministic and fast by default.
NNUE_GATE_MIN_REL_IMPROVEMENT = 0.0  # 0.0 => require strictly lower val loss

# NNUE Model Size Configurations
# Train multiple sizes and select best performing one
NNUE_MODEL_SIZES = {
    "small": {"hidden_dim": 128, "num_hidden_layers": 2, "epochs": 40},
    "medium": {"hidden_dim": 256, "num_hidden_layers": 2, "epochs": 30},
    "large": {"hidden_dim": 512, "num_hidden_layers": 2, "epochs": 25},
}
# Train all sizes every N training cycles (to find optimal size)
NNUE_TRAIN_ALL_SIZES_INTERVAL = 3


def _nnue_model_id(board: str, num_players: int, size: str = "") -> str:
    # Canonical NNUE ids: nnue_<board>_<numPlayers>p[_<size>] (matches ladder_config.py)
    base = f"nnue_{board}_{num_players}p"
    return f"{base}_{size}" if size else base


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp_{int(time.time())}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _gate_nnue_report(
    *,
    candidate_report: Dict[str, Any],
    baseline_best_val_loss: Optional[float],
) -> Dict[str, Any]:
    """Decide whether to promote a candidate NNUE based on training report metrics."""
    cand_loss_raw = candidate_report.get("best_val_loss")
    cand_loss: Optional[float] = None
    try:
        cand_loss = float(cand_loss_raw) if cand_loss_raw is not None else None
    except (TypeError, ValueError):
        cand_loss = None

    # If we have no baseline metric, promote by default (first model or legacy baseline).
    if baseline_best_val_loss is None or cand_loss is None:
        return {
            "promote": True,
            "candidate_best_val_loss": cand_loss,
            "baseline_best_val_loss": baseline_best_val_loss,
            "reason": "no_baseline_metric" if baseline_best_val_loss is None else "missing_candidate_metric",
        }

    # Require strictly lower validation loss (optionally with a relative margin).
    promote = cand_loss < baseline_best_val_loss * (1.0 - float(NNUE_GATE_MIN_REL_IMPROVEMENT))
    return {
        "promote": bool(promote),
        "candidate_best_val_loss": cand_loss,
        "baseline_best_val_loss": baseline_best_val_loss,
        "reason": "improved_val_loss" if promote else "no_val_loss_improvement",
    }


async def _train_nnue_size(
    config: Dict[str, Any],
    size_name: str,
    size_config: Dict[str, Any],
    dbs: List[Path],
    output_dir: Path,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Train a single NNUE model of a specific size.

    Returns (model_path, report) or (None, None) on failure.
    """
    nnue_id = _nnue_model_id(config["board"], config["players"], size_name)
    candidate_path = output_dir / f"{nnue_id}_candidate.pt"

    # Load optimized hyperparameters for this board/player config
    hp = get_hyperparameters(config["board"], config["players"])
    hp_info = get_hyperparameter_info(config["board"], config["players"])

    # Merge: size_config overrides hp for architecture, hp provides training params
    learning_rate = hp.get("learning_rate", 0.0003)
    batch_size = hp.get("batch_size", 256)
    weight_decay = hp.get("weight_decay", 0.0001)
    hidden_dim = size_config.get("hidden_dim", hp.get("hidden_dim", 256))
    num_hidden_layers = size_config.get("num_hidden_layers", hp.get("num_hidden_layers", 2))
    epochs = size_config.get("epochs", hp.get("epochs", NNUE_EPOCHS))

    if hp_info.get("optimized"):
        print(f"[Daemon] Using optimized hyperparameters for {config['board']} {config['players']}p "
              f"(confidence: {hp_info.get('confidence', 'unknown')})")

    nnue_cmd = [
        sys.executable, "scripts/train_nnue.py",
        "--db", *[str(db) for db in dbs[:5]],
        "--board-type", config["board"],
        "--num-players", str(config["players"]),
        "--epochs", str(epochs),
        "--hidden-dim", str(hidden_dim),
        "--num-hidden-layers", str(num_hidden_layers),
        "--learning-rate", str(learning_rate),
        "--batch-size", str(batch_size),
        "--weight-decay", str(weight_decay),
        "--run-dir", str(output_dir / size_name),
        "--model-id", nnue_id,
        "--save-path", str(candidate_path),
    ]

    (output_dir / size_name).mkdir(parents=True, exist_ok=True)
    success, output = run_command(nnue_cmd, timeout=3600)

    if not success:
        print(f"[Daemon] NNUE {size_name} training failed: {output[:200]}")
        return None, None

    report_path = output_dir / size_name / "nnue_training_report.json"
    report: Dict[str, Any] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            pass

    if not candidate_path.exists() or candidate_path.stat().st_size <= 0:
        return None, None

    return candidate_path, report


async def check_and_run_nnue_training(state: DaemonState) -> List[str]:
    """Check if NNUE models need retraining and run training if needed.

    Trains multiple model sizes (small, medium, large) and promotes the best one.
    Returns list of board config keys that were trained.
    """
    # Skip local training if disabled (low-memory machines)
    if DISABLE_LOCAL_TASKS:
        return []

    # Skip if system is overloaded
    if is_system_overloaded(verbose=True):
        print("[Daemon] Skipping NNUE training check - system overloaded")
        return []

    trained = []
    current_time = time.time()

    for config in BOARD_CONFIGS:
        key = get_config_key(config["board"], config["players"])
        bs = state.board_states.get(key)

        if not bs:
            continue

        # Get NNUE state for this config
        nnue = state.nnue_state.get(key, {
            "last_train_time": 0,
            "last_train_games": 0,
            "model_path": None,
            "train_cycle": 0,
        })

        # Check if enough time has passed
        time_since_train = current_time - nnue.get("last_train_time", 0)
        if time_since_train < NNUE_MIN_INTERVAL:
            continue

        # Check if enough new games accumulated
        games_since_train = bs.total_games - nnue.get("last_train_games", 0)
        if games_since_train < NNUE_MIN_NEW_GAMES:
            continue

        print(f"[Daemon] NNUE training triggered for {key}: {games_since_train} new games")

        # Find selfplay databases
        dbs = list(Path(AI_SERVICE_ROOT / "data" / "games").glob("*.db"))

        if not dbs:
            print(f"[Daemon] No selfplay databases found for NNUE training")
            continue

        # Determine which sizes to train
        train_cycle = nnue.get("train_cycle", 0) + 1
        train_all_sizes = (train_cycle % NNUE_TRAIN_ALL_SIZES_INTERVAL) == 0
        current_best_size = nnue.get("best_size", "large")  # Default to large

        if train_all_sizes:
            sizes_to_train = list(NNUE_MODEL_SIZES.keys())
            print(f"[Daemon] Training all NNUE sizes: {sizes_to_train}")
        else:
            # Just train the current best size
            sizes_to_train = [current_best_size]
            print(f"[Daemon] Training NNUE size: {current_best_size}")

        output_dir = AI_SERVICE_ROOT / "logs" / "nnue_auto" / f"{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train each size and collect results
        size_results: Dict[str, Tuple[Path, Dict[str, Any], float]] = {}
        for size_name in sizes_to_train:
            size_config = NNUE_MODEL_SIZES[size_name]
            print(f"[Daemon] Training NNUE {size_name} (hidden_dim={size_config['hidden_dim']})...")

            model_path, report = await _train_nnue_size(
                config, size_name, size_config, dbs, output_dir
            )

            if model_path and report:
                val_loss = report.get("best_val_loss")
                if val_loss is not None:
                    try:
                        size_results[size_name] = (model_path, report, float(val_loss))
                        print(f"[Daemon] NNUE {size_name}: val_loss={val_loss:.4f}")
                    except (TypeError, ValueError):
                        pass

        if not size_results:
            print(f"[Daemon] All NNUE training failed for {key}")
            continue

        # Find the best performing size
        best_size = min(size_results.keys(), key=lambda s: size_results[s][2])
        best_path, best_report, best_val_loss = size_results[best_size]

        print(f"[Daemon] Best NNUE size for {key}: {best_size} (val_loss={best_val_loss:.4f})")

        # Gate against current production model
        baseline_best_val_loss: Optional[float] = None
        baseline_record = state.nnue_state.get(key) or {}
        try:
            raw = baseline_record.get("best_val_loss")
            baseline_best_val_loss = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            baseline_best_val_loss = None

        gate = _gate_nnue_report(candidate_report=best_report, baseline_best_val_loss=baseline_best_val_loss)
        should_promote = bool(gate.get("promote", False))

        # Save size-specific models even if not promoted as primary
        models_dir = AI_SERVICE_ROOT / "models" / "nnue"
        for size_name, (model_path, _, _) in size_results.items():
            size_model_id = _nnue_model_id(config["board"], config["players"], size_name)
            size_stable_path = models_dir / f"{size_model_id}.pt"
            try:
                _atomic_copy(model_path, size_stable_path)
                print(f"[Daemon] Saved NNUE {size_name} model: {size_stable_path}")
            except Exception as e:
                print(f"[Daemon] Failed to save NNUE {size_name}: {e}")

        if should_promote:
            nnue_id = _nnue_model_id(config["board"], config["players"])
            stable_path = models_dir / f"{nnue_id}.pt"
            prev_path = models_dir / f"{nnue_id}_prev.pt"

            # Snapshot prior model
            if stable_path.exists() and stable_path.stat().st_size > 0:
                try:
                    _atomic_copy(stable_path, prev_path)
                except Exception as e:
                    print(f"[Daemon] Warning: failed to backup NNUE baseline: {e}")

            try:
                _atomic_copy(best_path, stable_path)
            except Exception as e:
                print(f"[Daemon] NNUE promotion copy failed for {key}: {e}")
                continue

            # Update NNUE state
            state.nnue_state[key] = {
                "last_train_time": current_time,
                "last_train_games": bs.total_games,
                "model_path": str(stable_path),
                "best_val_loss": best_val_loss,
                "baseline_best_val_loss": baseline_best_val_loss,
                "gate_reason": gate.get("reason"),
                "best_size": best_size,
                "train_cycle": train_cycle,
                "hidden_dim": NNUE_MODEL_SIZES[best_size]["hidden_dim"],
            }
            trained.append(key)
            print(f"[Daemon] NNUE promoted for {key}: {best_size} ({gate.get('reason')})")
        else:
            # Still update training time even if not promoted
            state.nnue_state[key] = {
                **nnue,
                "last_train_time": current_time,
                "last_train_games": bs.total_games,
                "train_cycle": train_cycle,
            }
            print(f"[Daemon] NNUE not promoted for {key} ({gate.get('reason')})")

        # Cleanup candidate files
        for size_name, (model_path, _, _) in size_results.items():
            try:
                if model_path.exists():
                    model_path.unlink()
            except Exception:
                pass

    return trained


# =============================================================================
# NNUE Policy Auto-Training
# =============================================================================

# Minimum new games before considering NNUE Policy retraining
NNUE_POLICY_MIN_NEW_GAMES = 5000
# Minimum time between NNUE Policy training runs (seconds) = 2 hours
NNUE_POLICY_MIN_INTERVAL = 2 * 60 * 60
# NNUE Policy training epochs
NNUE_POLICY_EPOCHS = 20
# Max samples per training run
NNUE_POLICY_MAX_SAMPLES = 50000


def _nnue_policy_model_id(board: str, num_players: int) -> str:
    """Get canonical NNUE policy model ID."""
    return f"nnue_policy_{board}_{num_players}p"


def _gate_nnue_policy_report(
    *,
    candidate_report: Dict[str, Any],
    baseline_accuracy: Optional[float],
) -> Dict[str, Any]:
    """Decide whether to promote a candidate NNUE policy based on accuracy."""
    cand_acc_raw = candidate_report.get("final_val_policy_accuracy")
    cand_acc: Optional[float] = None
    try:
        cand_acc = float(cand_acc_raw) if cand_acc_raw is not None else None
    except (TypeError, ValueError):
        cand_acc = None

    # If we have no baseline metric, promote by default
    if baseline_accuracy is None or cand_acc is None:
        return {
            "promote": True,
            "candidate_accuracy": cand_acc,
            "baseline_accuracy": baseline_accuracy,
            "reason": "no_baseline_metric" if baseline_accuracy is None else "missing_candidate_metric",
        }

    # Require strictly higher accuracy
    promote = cand_acc > baseline_accuracy
    return {
        "promote": bool(promote),
        "candidate_accuracy": cand_acc,
        "baseline_accuracy": baseline_accuracy,
        "reason": "improved_accuracy" if promote else "no_accuracy_improvement",
    }


async def check_and_run_nnue_policy_training(state: DaemonState) -> List[str]:
    """Check if NNUE policy models need retraining and run training if needed.

    Returns list of board config keys that were trained.
    """
    # Skip local training if disabled (low-memory machines)
    if DISABLE_LOCAL_TASKS:
        return []

    trained = []
    current_time = time.time()

    for config in BOARD_CONFIGS:
        key = get_config_key(config["board"], config["players"])
        bs = state.board_states.get(key)

        if not bs:
            continue

        # Get NNUE policy state for this config
        policy_state = state.nnue_policy_state.get(key, {
            "last_train_time": 0,
            "last_train_games": 0,
            "model_path": None,
        })

        # Check if enough time has passed
        time_since_train = current_time - policy_state.get("last_train_time", 0)
        if time_since_train < NNUE_POLICY_MIN_INTERVAL:
            continue

        # Check if enough new games accumulated
        games_since_train = bs.total_games - policy_state.get("last_train_games", 0)
        if games_since_train < NNUE_POLICY_MIN_NEW_GAMES:
            continue

        print(f"[Daemon] NNUE Policy training triggered for {key}: {games_since_train} new games")

        # Find selfplay databases
        dbs = list(Path(AI_SERVICE_ROOT / "data" / "games").glob("*.db"))

        if not dbs:
            print(f"[Daemon] No selfplay databases found for NNUE Policy training")
            continue

        # Run NNUE Policy training
        output_dir = AI_SERVICE_ROOT / "logs" / "nnue_policy_auto" / f"{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        policy_id = _nnue_policy_model_id(config["board"], config["players"])
        stable_path = AI_SERVICE_ROOT / "models" / "nnue" / f"{policy_id}.pt"
        prev_path = AI_SERVICE_ROOT / "models" / "nnue" / f"{policy_id}_prev.pt"
        candidate_path = output_dir / f"{policy_id}_candidate.pt"

        policy_cmd = [
            sys.executable, "scripts/train_nnue_policy.py",
            "--db", *[str(db) for db in dbs[:5]],  # Use up to 5 databases
            "--board-type", config["board"],
            "--num-players", str(config["players"]),
            "--epochs", str(NNUE_POLICY_EPOCHS),
            "--max-samples", str(NNUE_POLICY_MAX_SAMPLES),
            "--run-dir", str(output_dir),
            "--model-id", policy_id,
            "--save-path", str(candidate_path),
        ]

        success, output = run_command(policy_cmd, timeout=3600)  # 1 hour timeout

        if success:
            report_path = output_dir / "nnue_policy_training_report.json"
            report: Dict[str, Any] = {}
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text())
                except Exception:
                    report = {}

            if not candidate_path.exists() or candidate_path.stat().st_size <= 0:
                print(f"[Daemon] NNUE Policy training reported success but missing candidate: {candidate_path}")
                continue

            baseline_accuracy: Optional[float] = None
            baseline_record = state.nnue_policy_state.get(key) or {}
            try:
                raw = baseline_record.get("accuracy")
                baseline_accuracy = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                baseline_accuracy = None

            gate = _gate_nnue_policy_report(candidate_report=report, baseline_accuracy=baseline_accuracy)
            should_promote = bool(gate.get("promote", False))

            if should_promote:
                # Snapshot prior stable model for rollback
                if stable_path.exists() and stable_path.stat().st_size > 0:
                    try:
                        _atomic_copy(stable_path, prev_path)
                    except Exception as e:
                        print(f"[Daemon] Warning: failed to backup NNUE policy baseline: {e}")

                try:
                    _atomic_copy(candidate_path, stable_path)
                except Exception as e:
                    print(f"[Daemon] NNUE Policy promotion copy failed for {key}: {e}")
                    continue

                # Update NNUE Policy state
                state.nnue_policy_state[key] = {
                    "last_train_time": current_time,
                    "last_train_games": bs.total_games,
                    "model_path": str(stable_path),
                    "accuracy": gate.get("candidate_accuracy"),
                    "baseline_accuracy": gate.get("baseline_accuracy"),
                    "gate_reason": gate.get("reason"),
                }
                trained.append(key)
                print(f"[Daemon] NNUE Policy promoted for {key} ({gate.get('reason')})")
            else:
                print(f"[Daemon] NNUE Policy not promoted for {key} ({gate.get('reason')})")
                try:
                    candidate_path.unlink()
                except Exception:
                    pass
        else:
            print(f"[Daemon] NNUE Policy training failed for {key}: {output[:200]}")

    return trained


# =============================================================================
# Policy Benchmarking
# =============================================================================

# Minimum time between policy benchmarks (seconds) = 1 hour
POLICY_BENCHMARK_INTERVAL = 60 * 60
# Number of positions to benchmark
POLICY_BENCHMARK_POSITIONS = 500


async def run_policy_benchmark(state: DaemonState) -> Dict[str, Dict[str, Any]]:
    """Run policy benchmark for all configs with policy models.

    Returns dict of benchmark results per config key.
    """
    results = {}
    current_time = time.time()

    for config in BOARD_CONFIGS:
        key = get_config_key(config["board"], config["players"])

        # Check if policy model exists
        policy_model_path = AI_SERVICE_ROOT / "models" / "nnue" / f"nnue_policy_{config['board']}_{config['players']}p.pt"
        if not policy_model_path.exists():
            continue

        # Check last benchmark time
        policy_state = state.nnue_policy_state.get(key, {})
        last_benchmark = policy_state.get("last_benchmark_time", 0)
        if current_time - last_benchmark < POLICY_BENCHMARK_INTERVAL:
            continue

        # Find databases to benchmark against
        dbs = list(Path(AI_SERVICE_ROOT / "data" / "games").glob("*.db"))
        if not dbs:
            continue

        # Run benchmark
        output_dir = AI_SERVICE_ROOT / "runs" / f"policy_benchmark_{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        benchmark_cmd = [
            sys.executable, "scripts/benchmark_policy.py",
            "--db", *[str(db) for db in dbs[:3]],
            "--board-type", config["board"],
            "--num-players", str(config["players"]),
            "--model", str(policy_model_path),
            "--num-positions", str(POLICY_BENCHMARK_POSITIONS),
            "--output-dir", str(output_dir),
        ]

        print(f"[Daemon] Running policy benchmark for {key}...")
        success, output = run_command(benchmark_cmd, timeout=600)

        if success:
            # Parse benchmark results
            report_path = output_dir / "policy_benchmark_report.json"
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text())
                    top1 = report.get("top_1_accuracy", 0)
                    top3 = report.get("top_3_accuracy", 0)
                    top5 = report.get("top_5_accuracy", 0)
                    mean_rank = report.get("mean_rank", 999)

                    results[key] = {
                        "top_1_accuracy": top1,
                        "top_3_accuracy": top3,
                        "top_5_accuracy": top5,
                        "mean_rank": mean_rank,
                        "timestamp": current_time,
                    }

                    # Update policy state with benchmark results
                    if key in state.nnue_policy_state:
                        state.nnue_policy_state[key]["last_benchmark_time"] = current_time
                        state.nnue_policy_state[key]["benchmark_top1"] = top1
                        state.nnue_policy_state[key]["benchmark_top3"] = top3
                        state.nnue_policy_state[key]["benchmark_mean_rank"] = mean_rank

                    print(f"[Daemon] Policy benchmark {key}: Top-1={top1:.1%}, Top-3={top3:.1%}, Mean rank={mean_rank:.1f}")
                except Exception as e:
                    print(f"[Daemon] Failed to parse benchmark report: {e}")
        else:
            print(f"[Daemon] Policy benchmark failed for {key}: {output[:200]}")

    return results


# =============================================================================
# CMAES Heuristic Auto-Optimization
# =============================================================================

# Minimum new games before considering CMAES optimization
CMAES_MIN_NEW_GAMES = 20000
# Minimum time between CMAES runs (seconds) = 8 hours
CMAES_MIN_INTERVAL = 8 * 60 * 60
# CMAES iterations
CMAES_MAX_ITERATIONS = 3
CMAES_GENERATIONS_PER_ITER = 10


async def check_and_run_cmaes_optimization(state: DaemonState) -> List[str]:
    """Check if heuristic weights need optimization and run CMAES if needed.

    Returns list of board config keys that were optimized.
    """
    optimized = []
    current_time = time.time()

    for config in BOARD_CONFIGS[:3]:  # Focus on main configs (square8 2/3/4p)
        key = get_config_key(config["board"], config["players"])
        bs = state.board_states.get(key)

        if not bs:
            continue

        # Get CMAES state for this config
        cmaes = state.cmaes_state.get(key, {
            "last_opt_time": 0,
            "last_opt_games": 0,
            "profile_id": None,
        })

        # Check if enough time has passed
        time_since_opt = current_time - cmaes.get("last_opt_time", 0)
        if time_since_opt < CMAES_MIN_INTERVAL:
            continue

        # Check if enough new games accumulated
        games_since_opt = bs.total_games - cmaes.get("last_opt_games", 0)
        if games_since_opt < CMAES_MIN_NEW_GAMES:
            continue

        print(f"[Daemon] CMAES optimization triggered for {key}: {games_since_opt} new games")

        # Run CMAES optimization
        output_dir = AI_SERVICE_ROOT / "logs" / "cmaes_auto" / f"{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmaes_cmd = [
            sys.executable, "scripts/run_iterative_cmaes.py",
            "--board", config["board"],
            "--num-players", str(config["players"]),
            "--generations-per-iter", str(CMAES_GENERATIONS_PER_ITER),
            "--max-iterations", str(CMAES_MAX_ITERATIONS),
            "--output-dir", str(output_dir),
            "--games-per-eval", "15",
        ]

        success, output = run_command(cmaes_cmd, timeout=7200)  # 2 hour timeout

        if success:
            # Canonical profile ids use board abbreviations:
            #   heuristic_v1_sq8_2p / heuristic_v1_sq19_3p / heuristic_v1_hex_4p
            board_abbrev = {
                "square8": "sq8",
                "square19": "sq19",
                "hexagonal": "hex",
                "hex": "hex",
            }.get(config["board"], config["board"])
            profile_id = f"heuristic_v1_{board_abbrev}_{config['players']}p"

            # Update CMAES state
            state.cmaes_state[key] = {
                "last_opt_time": current_time,
                "last_opt_games": bs.total_games,
                "profile_id": profile_id,
            }
            optimized.append(key)
            print(f"[Daemon] CMAES optimization completed for {key}")

            # Auto-promote the new heuristic profile to production ladder (D2-D5)
            try:
                from app.config.ladder_config import (
                    update_tier_heuristic_profile,
                    get_heuristic_tiers,
                )
                from app.models import BoardType

                board_type_map = {
                    "square8": BoardType.SQUARE8,
                    "square19": BoardType.SQUARE19,
                    "hexagonal": BoardType.HEXAGONAL,
                }
                board_type_enum = board_type_map.get(config["board"])

                if board_type_enum:
                    heuristic_tiers = get_heuristic_tiers(board_type_enum, config["players"])
                    for difficulty in heuristic_tiers:
                        update_tier_heuristic_profile(
                            difficulty, board_type_enum, config["players"], profile_id
                        )
                    print(f"[Daemon] Auto-promoted heuristic profile {profile_id} to D{heuristic_tiers}")
            except Exception as e:
                print(f"[Daemon] Warning: Could not auto-promote heuristic profile: {e}")
        else:
            print(f"[Daemon] CMAES optimization failed for {key}: {output[:200]}")

    return optimized


# =============================================================================
# Auto-Promotion from Elo Leaderboard
# =============================================================================

# Minimum time between auto-promotion runs (seconds) = 15 minutes (was 1 hour)
# Reduced for faster feedback loop - promotions should happen more frequently
AUTO_PROMOTE_INTERVAL = 15 * 60
# Minimum Elo games required for promotion
AUTO_PROMOTE_MIN_GAMES = 20


async def run_auto_promotion(state: DaemonState) -> int:
    """Run automatic model promotion based on Elo rankings.

    Promotes best Elo models to production ladder tiers.
    Returns number of promotions made.
    """
    current_time = time.time()

    # Check if enough time has passed
    if current_time - state.last_auto_promote_time < AUTO_PROMOTE_INTERVAL:
        return 0

    print("[Daemon] Running auto-promotion from Elo leaderboard...")

    # Publish + sync stable best-model aliases from the Elo leaderboard.
    promote_cmd = [
        sys.executable,
        "scripts/model_promotion_manager.py",
        "--full-pipeline",
        "--min-games",
        str(AUTO_PROMOTE_MIN_GAMES),
    ]

    success, output = run_command(promote_cmd, timeout=300)

    if success:
        # Count promotions from output
        import re
        promotions_match = re.search(r"Promoted\\s+(\\d+)\\s+models", output)
        num_promotions = int(promotions_match.group(1)) if promotions_match else 0

        state.last_auto_promote_time = current_time
        state.total_auto_promotions += num_promotions

        if num_promotions > 0:
            print(f"[Daemon] Auto-promoted {num_promotions} model(s) to production ladder")
        else:
            print("[Daemon] No promotions needed")

        return num_promotions
    else:
        print(f"[Daemon] Auto-promotion failed: {output[:200]}")
        return 0


# =============================================================================
# League Training Integration
# =============================================================================

async def run_league_updates(state: DaemonState) -> Dict[str, int]:
    """Run league system updates - sync models, apply promotions/demotions.

    Returns dict with counts of synced, promoted, and demoted models.
    """
    if not HAS_LEAGUE_SYSTEM:
        return {"synced": 0, "promoted": 0, "demoted": 0}

    result = {"synced": 0, "promoted": 0, "demoted": 0}

    print("[Daemon] Running league system updates...")

    # Initialize league database if needed
    try:
        init_league_db()
    except Exception as e:
        print(f"[Daemon] Failed to initialize league DB: {e}")
        return result

    # Sync models from Elo database to league system
    for config in BOARD_CONFIGS[:3]:  # Focus on main configs
        board_type = config["board"]
        num_players = config["players"]

        try:
            synced = sync_from_elo_db(board_type, num_players)
            result["synced"] += synced
            if synced > 0:
                print(f"[Daemon] Synced {synced} models to league for {board_type}_{num_players}p")
        except Exception as e:
            print(f"[Daemon] Failed to sync {board_type}_{num_players}p: {e}")

        # Apply promotions and demotions
        try:
            pd_result = apply_promotions_demotions(board_type, num_players, dry_run=False)
            result["promoted"] += len(pd_result.get("promotions", []))
            result["demoted"] += len(pd_result.get("demotions", []))

            for p in pd_result.get("promotions", []):
                print(f"[Daemon] League PROMOTION: {p['model_id'][:30]} "
                      f"{p['from_league']} -> {p['to_league']} (Elo: {p['elo']:.0f})")

            for d in pd_result.get("demotions", []):
                print(f"[Daemon] League DEMOTION: {d['model_id'][:30]} "
                      f"{d['from_league']} -> {d['to_league']} (Elo: {d['elo']:.0f})")

        except Exception as e:
            print(f"[Daemon] Failed to apply promotions for {board_type}_{num_players}p: {e}")

    return result


def print_league_status() -> None:
    """Print current league standings summary."""
    if not HAS_LEAGUE_SYSTEM:
        return

    for config in BOARD_CONFIGS[:1]:  # Just show primary config
        board_type = config["board"]
        num_players = config["players"]

        standings = get_league_standings(board_type, num_players)
        if not standings:
            continue

        print(f"\n[League] {board_type}_{num_players}p standings:")
        for league_id in reversed(LEAGUE_ORDER):  # Show highest first
            league_models = [s for s in standings if s["league"] == league_id]
            if league_models:
                config = LEAGUES[league_id]
                top_model = league_models[0]
                print(f"  {config.name}: {len(league_models)} models, "
                      f"top={top_model['model_id'][:25]} (Elo: {top_model['elo']:.0f})")


# =============================================================================
# Main Daemon Loop
# =============================================================================

async def daemon_cycle(state: DaemonState) -> bool:
    """Run one cycle of the improvement daemon."""
    try:
        state.last_cycle_at = datetime.now().isoformat()
        state.total_cycles += 1

        print(f"\n[Daemon] === Cycle {state.total_cycles} at {state.last_cycle_at} ===")

        # Phase 1a: Balanced selfplay (homogeneous engine modes)
        print("[Daemon] Phase 1a: Running balanced selfplay...")
        games = await run_balanced_selfplay(state, duration_minutes=30)  # 30 min for ~40 games per config
        print(f"[Daemon] Generated {games} homogeneous selfplay games")

        # Save game count cache for faster threshold checks
        save_game_count_cache()

        # EVENT-DRIVEN: Check training immediately after selfplay if threshold reached
        # This reduces feedback latency by not waiting for Phase 2
        if games > 0:
            early_trained = await check_and_run_training(state)
            if early_trained:
                print(f"[Daemon] Early training triggered: {early_trained}")
                # Run immediate tournament for early-trained models
                for model_id in early_trained:
                    result = await run_model_comparison(state, model_id)
                    if result:
                        print(f"[Daemon] Early tournament result: {result}")
                # Trigger promotion check immediately after early tournament
                promotions = await run_auto_promotion(state)
                if promotions > 0:
                    print(f"[Daemon] Early promotion: {promotions} model(s)")
                    maybe_sync_staging("early_promotion")

        # Phase 1b: Asymmetric selfplay (every 3rd cycle for diversity)
        if state.total_cycles % 3 == 0:
            print("[Daemon] Phase 1b: Running asymmetric selfplay for diverse training data...")
            for config in BOARD_CONFIGS[:3]:  # Focus on main configs
                asymmetric_games = await run_asymmetric_selfplay(
                    state, config["board"], config["players"]
                )
                games += asymmetric_games
                state.total_games_generated += asymmetric_games
            print(f"[Daemon] Generated {games} total games (including asymmetric)")

        # Phase 1c: GPU policy-guided selfplay (every 4th cycle when policy models exist)
        if state.total_cycles % 4 == 0:
            print("[Daemon] Phase 1c: Running GPU policy-guided selfplay...")
            gpu_policy_games = await run_gpu_policy_selfplay(state, games_per_config=50)
            if gpu_policy_games > 0:
                games += gpu_policy_games
                print(f"[Daemon] Generated {gpu_policy_games} GPU policy-guided games")

        # Phase 2: Check and run training
        print("[Daemon] Phase 2: Checking training thresholds...")
        trained_models = await check_and_run_training(state)

        # Phase 3: Run tournaments for new models
        if trained_models:
            print(f"[Daemon] Phase 3: Running {len(trained_models)} model comparisons...")
            for model_id in trained_models:
                result = await run_model_comparison(state, model_id)
                if result:
                    print(f"[Daemon] Tournament result: {result}")

        # Phase 4: Periodic cross-model tournament (every 5 cycles for faster validation)
        if state.total_cycles % 5 == 0:
            print("[Daemon] Phase 4: Running scheduled cross-model tournament...")
            games = await run_cross_model_tournament(state, top_n=10, games_per_matchup=6)
            print(f"[Daemon] Cross-model tournament completed: {games} games played")

        # Phase 5: Auto-promote best Elo models to production (every cycle, but rate-limited)
        print("[Daemon] Phase 5: Checking auto-promotion...")
        promotions = await run_auto_promotion(state)
        if promotions > 0:
            maybe_sync_staging("auto_promotion")

        # Phase 5b: League system updates (every 3 cycles)
        if state.total_cycles % 3 == 0:
            print("[Daemon] Phase 5b: Running league system updates...")
            league_result = await run_league_updates(state)
            if league_result["synced"] > 0 or league_result["promoted"] > 0 or league_result["demoted"] > 0:
                print(f"[Daemon] League updates: {league_result['synced']} synced, "
                      f"{league_result['promoted']} promoted, {league_result['demoted']} demoted")
            print_league_status()

        # Phase 6a: NNUE (value) retraining (when enough new games accumulated)
        if state.total_cycles % 5 == 0:
            print("[Daemon] Phase 6a: Checking NNUE retraining thresholds...")
            nnue_trained = await check_and_run_nnue_training(state)
            if nnue_trained:
                print(f"[Daemon] NNUE models retrained for: {', '.join(nnue_trained)}")
                maybe_sync_staging("nnue_training")

        # Phase 6b: NNUE Policy retraining (when enough new games accumulated)
        if state.total_cycles % 5 == 0:
            print("[Daemon] Phase 6b: Checking NNUE Policy retraining thresholds...")
            policy_trained = await check_and_run_nnue_policy_training(state)
            if policy_trained:
                print(f"[Daemon] NNUE Policy models retrained for: {', '.join(policy_trained)}")
                maybe_sync_staging("nnue_policy_training")

        # Phase 6c: Policy benchmarking (every 3rd cycle)
        if state.total_cycles % 3 == 0:
            print("[Daemon] Phase 6c: Running policy benchmarks...")
            benchmark_results = await run_policy_benchmark(state)
            if benchmark_results:
                for key, res in benchmark_results.items():
                    print(f"[Daemon] Policy {key}: Top-1={res['top_1_accuracy']:.1%}")

        # Phase 7: CMAES heuristic optimization (when enough new games accumulated)
        if state.total_cycles % 15 == 0:
            print("[Daemon] Phase 7: Checking CMAES optimization thresholds...")
            cmaes_optimized = await check_and_run_cmaes_optimization(state)
            if cmaes_optimized:
                print(f"[Daemon] CMAES heuristics optimized for: {', '.join(cmaes_optimized)}")
                maybe_sync_staging("cmaes_optimization")

        # Phase 8: Print status
        print_leaderboard(state)

        # Reset failure counter on success
        state.consecutive_failures = 0
        save_state(state)

        # Run S3 backup if enabled and interval has passed
        run_s3_backup(models_only=True)

        # Report cluster metrics to Prometheus
        report_local_cluster_metrics()

        return True

    except Exception as e:
        state.consecutive_failures += 1
        state.last_error = str(e)
        state.last_error_time = datetime.now().isoformat()
        save_state(state)

        print(f"[Daemon] Cycle failed: {e}")
        return False


async def run_daemon(foreground: bool = False) -> None:
    """Run the continuous improvement daemon."""
    # Acquire exclusive lock to prevent multiple daemons
    if HAS_COORDINATION and _coordinator is not None:
        if _coordinator.is_role_held(TaskRole.ORCHESTRATOR):
            existing_pid = _coordinator.get_role_holder_pid(TaskRole.ORCHESTRATOR)
            print(f"[Daemon] ERROR: Another orchestrator is already running (PID {existing_pid})")
            print("[Daemon] Kill that process first or wait for it to complete.")
            return
        print("[Daemon] Coordination module active - will acquire exclusive lock")

    # Check system memory - skip on low-memory machines to avoid OOM
    # 64GB minimum required; machines below this threshold should not run local tasks
    MIN_MEMORY_GB = 64
    try:
        import psutil
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        if system_memory_gb < MIN_MEMORY_GB:
            print(f"[Daemon] ERROR: System has only {system_memory_gb:.1f}GB RAM, minimum {MIN_MEMORY_GB}GB required")
            print("[Daemon] Exiting to avoid OOM on low-memory machine")
            return
    except Exception as e:
        print(f"[Daemon] Warning: Could not check system memory: {e}")

    state = load_state()

    if not state.started_at:
        state.started_at = datetime.now().isoformat()

    # Initialize board states
    for config in BOARD_CONFIGS:
        key = get_config_key(config["board"], config["players"])
        if key not in state.board_states:
            state.board_states[key] = BoardTypeState(config["board"], config["players"])

    # Load game count cache for faster threshold checks
    load_game_count_cache()

    print("[Daemon] Starting continuous improvement daemon...")
    print_status(state)

    # Signal handling for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        print("\n[Daemon] Shutdown signal received...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Acquire orchestrator lock
    lock_context = None
    task_id = None
    if HAS_COORDINATION and _coordinator is not None:
        try:
            lock_context = _coordinator.acquire_role(
                TaskRole.ORCHESTRATOR,
                blocking=False,
                description="Continuous improvement daemon"
            )
            task_id = lock_context.__enter__()
            print(f"[Daemon] Acquired orchestrator lock (task_id={task_id})")
        except RuntimeError as e:
            print(f"[Daemon] ERROR: Failed to acquire lock: {e}")
            return
        except Exception as e:
            print(f"[Daemon] Warning: Lock acquisition failed: {e}")

    # New coordination: acquire IMPROVEMENT_DAEMON role (SQLite-backed with heartbeat)
    has_new_role = False
    if HAS_NEW_COORDINATION and OrchestratorRole is not None:
        try:
            if acquire_orchestrator_role(OrchestratorRole.IMPROVEMENT_DAEMON):
                has_new_role = True
                print("[Daemon] Acquired IMPROVEMENT_DAEMON role via new coordination system")
            else:
                print("[Daemon] WARNING: Another improvement daemon already holds the role")
        except Exception as e:
            print(f"[Daemon] Warning: Failed to acquire new orchestrator role: {e}")

    # Task coordination - global limits across all orchestrators
    tc_task_id = None
    task_coordinator = None
    if HAS_TASK_COORDINATOR:
        try:
            task_coordinator = TaskCoordinator.get_instance()
            import socket
            node_id = socket.gethostname()
            allowed, reason = task_coordinator.can_spawn_task(TaskType.IMPROVEMENT_LOOP, node_id)
            if not allowed:
                print(f"[Daemon] WARNING: TaskCoordinator denied spawning: {reason}")
            else:
                tc_task_id = f"daemon_{int(time.time())}"
                task_coordinator.register_task(
                    tc_task_id, TaskType.IMPROVEMENT_LOOP, node_id, os.getpid()
                )
            print("[Daemon] Task coordinator integrated - global limits enforced")
        except Exception as e:
            print(f"[Daemon] Warning: Failed to initialize task coordinator: {e}")

    try:
        # Main loop
        last_leader_ok: Optional[bool] = None
        while not shutdown_event.is_set():
            if USE_P2P_ORCHESTRATOR and IMPROVEMENT_LEADER_ONLY:
                leader_ok = await is_local_p2p_leader()
                if last_leader_ok is None or leader_ok != last_leader_ok:
                    if leader_ok:
                        print("[Daemon] Leadership gate: active (local node is effective cluster leader)")
                    else:
                        print("[Daemon] Leadership gate: paused (local node is not effective cluster leader)")
                    last_leader_ok = leader_ok
                if not leader_ok:
                    try:
                        await asyncio.wait_for(shutdown_event.wait(), timeout=IMPROVEMENT_LEADER_POLL_SECONDS)
                    except asyncio.TimeoutError:
                        pass
                    continue

            success = await daemon_cycle(state)

            if not success:
                # Exponential backoff on failure
                delay = min(
                    RETRY_BASE_DELAY * (2 ** state.consecutive_failures),
                    RETRY_MAX_DELAY
                )
                print(f"[Daemon] Waiting {delay:.0f}s before retry...")
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
            else:
                # Normal cycle interval
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=CYCLE_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    pass

    finally:
        # Unregister from task coordinator
        if task_coordinator and tc_task_id:
            try:
                task_coordinator.unregister_task(tc_task_id)
                print("[Daemon] Unregistered from task coordinator")
            except Exception as e:
                print(f"[Daemon] Warning: Task coordinator unregister failed: {e}")

        # Release new orchestrator role (SQLite-backed)
        if has_new_role and HAS_NEW_COORDINATION:
            try:
                release_orchestrator_role()
                print("[Daemon] Released IMPROVEMENT_DAEMON role")
            except Exception as e:
                print(f"[Daemon] Warning: Failed to release new orchestrator role: {e}")

        # Release lock and save state
        if lock_context is not None:
            try:
                lock_context.__exit__(None, None, None)
                print("[Daemon] Released orchestrator lock")
            except Exception as e:
                print(f"[Daemon] Warning: Lock release failed: {e}")

        print("[Daemon] Shutdown complete")
        save_state(state)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Continuous AI Improvement Daemon")
    parser.add_argument("--start", action="store_true", help="Start daemon in background")
    parser.add_argument("--foreground", action="store_true", help="Run in foreground")
    parser.add_argument("--status", action="store_true", help="Show daemon status")
    parser.add_argument("--stop", action="store_true", help="Stop running daemon")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--sync-staging",
        action="store_true",
        help=(
            "Enable SSH sync of promoted models/weights to staging after promotion/training. "
            "Requires RINGRIFT_STAGING_SSH_HOST and RINGRIFT_STAGING_ROOT."
        ),
    )
    parser.add_argument(
        "--sync-staging-no-restart",
        action="store_true",
        help="When used with --sync-staging, do not restart docker compose services after syncing.",
    )
    parser.add_argument(
        "--sync-staging-validate-health",
        action="store_true",
        help="When used with --sync-staging, validate /internal/ladder/health on staging after sync.",
    )
    parser.add_argument(
        "--sync-staging-fail-on-missing",
        action="store_true",
        help=(
            "When used with --sync-staging, exit non-zero if staging reports missing artifacts "
            "(implies --sync-staging-validate-health)."
        ),
    )

    args = parser.parse_args()

    global SYNC_STAGING
    global SYNC_STAGING_RESTART
    global SYNC_STAGING_VALIDATE_HEALTH
    global SYNC_STAGING_FAIL_ON_MISSING

    if args.sync_staging:
        SYNC_STAGING = True
        if args.sync_staging_no_restart:
            SYNC_STAGING_RESTART = False
        if args.sync_staging_validate_health or args.sync_staging_fail_on_missing:
            SYNC_STAGING_VALIDATE_HEALTH = True
        if args.sync_staging_fail_on_missing:
            SYNC_STAGING_FAIL_ON_MISSING = True

    if args.status:
        state = load_state()
        print_status(state)
        return

    if args.stop:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to daemon (PID {pid})")
            except ProcessLookupError:
                print("Daemon not running")
            PID_FILE.unlink(missing_ok=True)
        else:
            print("No PID file found")
        return

    if args.foreground:
        asyncio.run(run_daemon(foreground=True))
    elif args.start:
        # Daemonize
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        pid = os.fork()
        if pid > 0:
            print(f"Daemon started (PID {pid})")
            PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            PID_FILE.write_text(str(pid))
            sys.exit(0)

        os.setsid()

        # Redirect stdout/stderr to log file with line buffering
        with open(LOG_FILE, "a") as log:
            os.dup2(log.fileno(), sys.stdout.fileno())
            os.dup2(log.fileno(), sys.stderr.fileno())

        # Enable line buffering for real-time log monitoring
        sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, closefd=False)
        sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, closefd=False)

        asyncio.run(run_daemon(foreground=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
