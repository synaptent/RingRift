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

# =============================================================================
# Configuration
# =============================================================================

# Board type configurations for balanced training
BOARD_CONFIGS = [
    {"board": "square8", "players": 2, "priority": 1.0, "min_games": 10000},
    {"board": "square8", "players": 3, "priority": 0.3, "min_games": 3000},
    {"board": "square8", "players": 4, "priority": 0.2, "min_games": 2000},
    {"board": "square19", "players": 2, "priority": 0.5, "min_games": 5000},
    {"board": "hexagonal", "players": 2, "priority": 0.4, "min_games": 4000},
    {"board": "hexagonal", "players": 3, "priority": 0.2, "min_games": 2000},
]

# Training thresholds
MIN_NEW_GAMES_FOR_TRAINING = 2000  # Train after this many new games
TRAINING_COOLDOWN_SECONDS = 1800   # 30 min between training runs
TOURNAMENT_GAMES = 50              # Games per model comparison
PROMOTION_THRESHOLD = 0.55         # Win rate needed for promotion

# Selfplay configuration
SELFPLAY_BATCH_SIZE = 100          # Games per selfplay batch
SELFPLAY_ENGINES = ["mixed", "descent-only", "mcts-only"]

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


async def run_balanced_selfplay(state: DaemonState, duration_minutes: int = 10) -> int:
    """Run selfplay balanced across board types based on priority and need."""
    total_games = 0

    # Calculate weights based on priority and data deficit
    weights = []
    for config in BOARD_CONFIGS:
        key = get_config_key(config["board"], config["players"])
        bs = state.board_states.get(key, BoardTypeState(config["board"], config["players"]))

        # Higher weight if we have less data relative to minimum
        deficit_ratio = max(0, 1 - bs.total_games / config["min_games"])
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
            "--max-moves", "500",
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

        # Train model
        iteration = bs.current_iteration + 1
        model_id = f"{key}_iter{iteration}"
        model_path = AI_SERVICE_ROOT / "models" / f"{model_id}.pth"

        train_cmd = [
            sys.executable, "scripts/train_neural_net.py",
            "--data", str(AI_SERVICE_ROOT / "data" / "training" / f"daemon_{key}.npz"),
            "--output", str(model_path),
            "--board-type", config["board"],
            "--epochs", "50",
            "--early-stopping-patience", "10",
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

    # Update Elo ratings
    score = win_rate
    new_rating, best_rating = update_elo(
        state.elo_ratings.get(new_model_id, 1500),
        state.elo_ratings.get(bs.best_model_id, 1500),
        score
    )
    state.elo_ratings[new_model_id] = new_rating
    state.elo_ratings[bs.best_model_id] = best_rating

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

    print_leaderboard(state)


# =============================================================================
# Main Daemon Loop
# =============================================================================

async def daemon_cycle(state: DaemonState) -> bool:
    """Run one cycle of the improvement daemon."""
    try:
        state.last_cycle_at = datetime.now().isoformat()
        state.total_cycles += 1

        print(f"\n[Daemon] === Cycle {state.total_cycles} at {state.last_cycle_at} ===")

        # Phase 1: Balanced selfplay
        print("[Daemon] Phase 1: Running balanced selfplay...")
        games = await run_balanced_selfplay(state, duration_minutes=5)
        print(f"[Daemon] Generated {games} total games this cycle")

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

        # Phase 4: Print status
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
