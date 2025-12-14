#!/usr/bin/env python3
"""Continuous AI Improvement Daemon - Self-healing, cyclical training loop.

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

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Import persistent Elo database functions
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

# Import ImprovementCycleManager for diverse AI scheduling
try:
    from scripts.improvement_cycle_manager import ImprovementCycleManager
    HAS_IMPROVEMENT_MANAGER = True
except ImportError:
    HAS_IMPROVEMENT_MANAGER = False
    ImprovementCycleManager = None

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
P2P_AUTH_TOKEN = os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN", "")
USE_P2P_ORCHESTRATOR = os.environ.get("USE_P2P_ORCHESTRATOR", "").lower() in ("1", "true", "yes")

# Optional: sync promoted artifacts to a staging deployment.
SYNC_STAGING = os.environ.get("RINGRIFT_SYNC_STAGING", "").lower() in ("1", "true", "yes", "on")
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


async def get_p2p_cluster_status() -> Optional[Dict[str, Any]]:
    """Query P2P orchestrator for cluster status and data manifest."""
    if not USE_P2P_ORCHESTRATOR:
        return None

    try:
        import aiohttp
        headers = {"Authorization": f"Bearer {P2P_AUTH_TOKEN}"} if P2P_AUTH_TOKEN else {}
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{P2P_ORCHESTRATOR_URL}/cluster_data_manifest",
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
    {"board": "square8", "players": 2, "priority": 1.0, "min_games": 10000},
    {"board": "square8", "players": 3, "priority": 0.3, "min_games": 3000},
    {"board": "square8", "players": 4, "priority": 0.2, "min_games": 2000},
    {"board": "square19", "players": 2, "priority": 0.5, "min_games": 5000},
    {"board": "square19", "players": 3, "priority": 0.2, "min_games": 2000},
    {"board": "square19", "players": 4, "priority": 0.1, "min_games": 1000},
    {"board": "hexagonal", "players": 2, "priority": 0.4, "min_games": 4000},
    {"board": "hexagonal", "players": 3, "priority": 0.2, "min_games": 2000},
    {"board": "hexagonal", "players": 4, "priority": 0.1, "min_games": 1000},
]

# Training thresholds
MIN_NEW_GAMES_FOR_TRAINING = 2000  # Train after this many new games
TRAINING_COOLDOWN_SECONDS = 1800   # 30 min between training runs
TOURNAMENT_GAMES = 50              # Games per model comparison
PROMOTION_THRESHOLD = 0.55         # Win rate needed for promotion

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

            # Load NNUE and CMAES state
            state.nnue_state = data.get("nnue_state", {})
            state.cmaes_state = data.get("cmaes_state", {})
            state.last_auto_promote_time = data.get("last_auto_promote_time", 0.0)
            state.total_auto_promotions = data.get("total_auto_promotions", 0)

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
        "cmaes_state": state.cmaes_state,
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


def count_games_in_jsonl(path: Path) -> int:
    """Count games in a JSONL file."""
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


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
    """
    total_games = 0
    key = get_config_key(board_type, num_players)

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
        "--board", board_type.replace("square", "Square").replace("hex", "Hex"),
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


async def run_balanced_selfplay(state: DaemonState, duration_minutes: int = 10) -> int:
    """Run selfplay balanced across board types based on priority and need.

    Uses ImprovementCycleManager for diverse AI opponent selection when available.
    Incorporates P2P cluster data manifest if available for smarter balancing.
    """
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

        # Run selfplay for each config based on weight
        for config, weight in weights:
            if weight < 0.05:  # Skip very low weight configs
                continue

            batch_games = max(10, int(SELFPLAY_BATCH_SIZE * weight * 2))
            engine = random.choice(SELFPLAY_ENGINES)

            key = get_config_key(config["board"], config["players"])
            output_file = AI_SERVICE_ROOT / "data" / "selfplay" / f"daemon_{key}" / f"games_{int(time.time())}.jsonl"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, "scripts/run_self_play_soak.py",
                "--board-type", config["board"],
                "--num-players", str(config["players"]),
                "--num-games", str(batch_games),
                "--engine-mode", engine,
                "--log-jsonl", str(output_file),
                "--max-moves", "10000",  # Avoid draws due to move limit
            ]

            print(f"[Daemon] Running {batch_games} {key} selfplay games with {engine}...")
            success, output = run_command(cmd, timeout=duration_minutes * 60)

            if success:
                games_generated = count_games_in_jsonl(output_file)
                total_games += games_generated

                # Update state
                if key not in state.board_states:
                    state.board_states[key] = BoardTypeState(config["board"], config["players"])
                state.board_states[key].total_games += games_generated
                state.board_states[key].games_since_last_training += games_generated

                print(f"[Daemon] Generated {games_generated} {key} games")
            else:
                print(f"[Daemon] Selfplay failed for {key}: {output[:200]}")

    state.total_games_generated += total_games
    return total_games


async def check_and_run_training(state: DaemonState) -> List[str]:
    """Check if any board type needs training and run it."""
    trained_models = []
    current_time = time.time()

    for config in BOARD_CONFIGS:
        key = get_config_key(config["board"], config["players"])
        bs = state.board_states.get(key)

        if not bs:
            continue

        # Check if training is needed
        needs_training = (
            bs.games_since_last_training >= MIN_NEW_GAMES_FOR_TRAINING and
            (current_time - bs.last_training_time) >= TRAINING_COOLDOWN_SECONDS and
            bs.total_games >= config["min_games"]
        )

        if not needs_training:
            continue

        print(f"[Daemon] Training needed for {key}: {bs.games_since_last_training} new games")

        # Export training data
        export_cmd = [
            sys.executable, "scripts/jsonl_to_npz.py",
            "--input-dir", str(AI_SERVICE_ROOT / "data" / "selfplay"),
            "--output", str(AI_SERVICE_ROOT / "data" / "training" / f"daemon_{key}.npz"),
            "--board-type", config["board"],
            "--num-players", str(config["players"]),
        ]

        success, output = run_command(export_cmd, timeout=600)
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
        data_path = AI_SERVICE_ROOT / "data" / "training" / f"daemon_{key}.npz"

        run_dir = (
            AI_SERVICE_ROOT
            / "logs"
            / "nn_training"
            / f"{key}_iter{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

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
            str(data_path),
            "--epochs", "50",
            "--model-version",
            "v2",
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
        elo_tournament_cmd = [
            sys.executable, "scripts/run_model_elo_tournament.py",
            "--board", config["board"],
            "--players", str(config["players"]),
            "--games", str(games_per_matchup),
            "--top-n", str(top_n),
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


def _nnue_model_id(board: str, num_players: int) -> str:
    # Canonical NNUE ids: nnue_<board>_<numPlayers>p (matches ladder_config.py)
    return f"nnue_{board}_{num_players}p"


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


async def check_and_run_nnue_training(state: DaemonState) -> List[str]:
    """Check if NNUE models need retraining and run training if needed.

    Returns list of board config keys that were trained.
    """
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
        db_pattern = str(AI_SERVICE_ROOT / "data" / "games" / "*.db")
        dbs = list(Path(AI_SERVICE_ROOT / "data" / "games").glob("*.db"))

        if not dbs:
            print(f"[Daemon] No selfplay databases found for NNUE training")
            continue

        # Run NNUE training
        output_dir = AI_SERVICE_ROOT / "logs" / "nnue_auto" / f"{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        nnue_id = _nnue_model_id(config["board"], config["players"])
        stable_path = AI_SERVICE_ROOT / "models" / "nnue" / f"{nnue_id}.pt"
        prev_path = AI_SERVICE_ROOT / "models" / "nnue" / f"{nnue_id}_prev.pt"
        candidate_path = output_dir / f"{nnue_id}_candidate.pt"

        nnue_cmd = [
            sys.executable, "scripts/train_nnue.py",
            "--db", *[str(db) for db in dbs[:5]],  # Use up to 5 databases
            "--board-type", config["board"],
            "--num-players", str(config["players"]),
            "--epochs", str(NNUE_EPOCHS),
            "--run-dir", str(output_dir),
            "--model-id", nnue_id,
            "--save-path", str(candidate_path),
        ]

        success, output = run_command(nnue_cmd, timeout=3600)  # 1 hour timeout

        if success:
            report_path = output_dir / "nnue_training_report.json"
            report: Dict[str, Any] = {}
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text())
                except Exception:
                    report = {}

            if not candidate_path.exists() or candidate_path.stat().st_size <= 0:
                print(f"[Daemon] NNUE training reported success but missing candidate checkpoint: {candidate_path}")
                continue

            baseline_best_val_loss: Optional[float] = None
            baseline_record = state.nnue_state.get(key) or {}
            try:
                raw = baseline_record.get("best_val_loss")
                baseline_best_val_loss = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                baseline_best_val_loss = None

            gate = _gate_nnue_report(candidate_report=report, baseline_best_val_loss=baseline_best_val_loss)
            should_promote = bool(gate.get("promote", False))

            if should_promote:
                # Snapshot the prior stable model for rollback (best-effort).
                if stable_path.exists() and stable_path.stat().st_size > 0:
                    try:
                        _atomic_copy(stable_path, prev_path)
                    except Exception as e:
                        print(f"[Daemon] Warning: failed to backup NNUE baseline: {e}")

                try:
                    _atomic_copy(candidate_path, stable_path)
                except Exception as e:
                    print(f"[Daemon] NNUE promotion copy failed for {key}: {e}")
                    continue

                # Update NNUE state (record promotion metrics).
                state.nnue_state[key] = {
                    "last_train_time": current_time,
                    "last_train_games": bs.total_games,
                    "model_path": str(stable_path),
                    "best_val_loss": gate.get("candidate_best_val_loss"),
                    "baseline_best_val_loss": gate.get("baseline_best_val_loss"),
                    "gate_reason": gate.get("reason"),
                }
                trained.append(key)
                print(f"[Daemon] NNUE promoted for {key} ({gate.get('reason')})")
            else:
                print(f"[Daemon] NNUE not promoted for {key} ({gate.get('reason')})")
                try:
                    candidate_path.unlink()
                except Exception:
                    pass
        else:
            print(f"[Daemon] NNUE training failed for {key}: {output[:200]}")

    return trained


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

# Minimum time between auto-promotion runs (seconds) = 1 hour
AUTO_PROMOTE_INTERVAL = 60 * 60
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

        # Phase 6: NNUE retraining (when enough new games accumulated)
        if state.total_cycles % 5 == 0:
            print("[Daemon] Phase 6: Checking NNUE retraining thresholds...")
            nnue_trained = await check_and_run_nnue_training(state)
            if nnue_trained:
                print(f"[Daemon] NNUE models retrained for: {', '.join(nnue_trained)}")
                maybe_sync_staging("nnue_training")

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
    state = load_state()

    if not state.started_at:
        state.started_at = datetime.now().isoformat()

    # Initialize board states
    for config in BOARD_CONFIGS:
        key = get_config_key(config["board"], config["players"])
        if key not in state.board_states:
            state.board_states[key] = BoardTypeState(config["board"], config["players"])

    print("[Daemon] Starting continuous improvement daemon...")
    print_status(state)

    # Signal handling for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        print("\n[Daemon] Shutdown signal received...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Main loop
    while not shutdown_event.is_set():
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

    args = parser.parse_args()

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

        # Redirect stdout/stderr to log file
        with open(LOG_FILE, "a") as log:
            os.dup2(log.fileno(), sys.stdout.fileno())
            os.dup2(log.fileno(), sys.stderr.fileno())

        asyncio.run(run_daemon(foreground=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
