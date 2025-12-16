#!/usr/bin/env python3
"""Full Automation Loop for RingRift AI Training.

This script provides comprehensive automation for:
1. Monitoring training data across all board types
2. Triggering V3 training when thresholds are met
3. Running evaluation tournaments after training
4. Promoting winning models to production
5. Syncing models across cluster nodes

Usage:
    python scripts/full_automation_loop.py [--dry-run] [--once]
"""

import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup paths
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration
BOARD_CONFIGS = [
    {"board": "square8", "players": 2, "min_games": 500, "encoder": "default"},
    {"board": "square8", "players": 3, "min_games": 300, "encoder": "default"},
    {"board": "square8", "players": 4, "min_games": 300, "encoder": "default"},
    {"board": "square19", "players": 2, "min_games": 200, "encoder": "default"},
    {"board": "hexagonal", "players": 2, "min_games": 200, "encoder": "v3"},
    {"board": "hexagonal", "players": 3, "min_games": 150, "encoder": "v3"},
    {"board": "hexagonal", "players": 4, "min_games": 150, "encoder": "v3"},
]

CLUSTER_NODES = ["lambda-h100", "lambda-a10"]
MODEL_DIR = AI_SERVICE_ROOT / "models"
DATA_DIR = AI_SERVICE_ROOT / "data"
GAMES_DIR = DATA_DIR / "games"
TRAINING_DIR = DATA_DIR / "training"
LOGS_DIR = AI_SERVICE_ROOT / "logs" / "automation"
STATE_FILE = AI_SERVICE_ROOT / "data" / "automation_state.json"

# Thresholds
MIN_GAMES_SINCE_TRAINING = 200
TOURNAMENT_GAMES_PER_PAIR = 30
ELO_IMPROVEMENT_THRESHOLD = 30  # Minimum Elo gain to promote


@dataclass
class BoardState:
    """State tracking for a board configuration."""
    board_type: str
    num_players: int
    encoder_version: str
    last_game_count: int = 0
    last_training_time: Optional[str] = None
    last_tournament_time: Optional[str] = None
    current_best_model: Optional[str] = None
    current_best_elo: float = 1500.0
    training_in_progress: bool = False

    def key(self) -> str:
        return f"{self.board_type}_{self.num_players}p"


@dataclass
class AutomationState:
    """Global automation state."""
    boards: Dict[str, BoardState] = field(default_factory=dict)
    total_cycles: int = 0
    last_cycle_time: Optional[str] = None

    def save(self, path: Path):
        """Save state to JSON file."""
        data = {
            "boards": {
                k: {
                    "board_type": v.board_type,
                    "num_players": v.num_players,
                    "encoder_version": v.encoder_version,
                    "last_game_count": v.last_game_count,
                    "last_training_time": v.last_training_time,
                    "last_tournament_time": v.last_tournament_time,
                    "current_best_model": v.current_best_model,
                    "current_best_elo": v.current_best_elo,
                    "training_in_progress": v.training_in_progress,
                }
                for k, v in self.boards.items()
            },
            "total_cycles": self.total_cycles,
            "last_cycle_time": self.last_cycle_time,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "AutomationState":
        """Load state from JSON file."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            state = cls(
                total_cycles=data.get("total_cycles", 0),
                last_cycle_time=data.get("last_cycle_time"),
            )
            for k, v in data.get("boards", {}).items():
                state.boards[k] = BoardState(**v)
            return state
        except Exception as e:
            logger.warning(f"Failed to load state: {e}, starting fresh")
            return cls()


def run_command(cmd: List[str], timeout: int = 300, cwd: Optional[Path] = None) -> Tuple[bool, str]:
    """Run a command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or AI_SERVICE_ROOT,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def get_game_counts() -> Dict[str, int]:
    """Get game counts from all databases."""
    counts: Dict[str, int] = {}

    for db_path in GAMES_DIR.glob("*.db"):
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute(
                "SELECT board_type, num_players, COUNT(*) FROM games GROUP BY board_type, num_players"
            )
            for row in cursor:
                key = f"{row[0]}_{row[1]}p"
                counts[key] = counts.get(key, 0) + row[2]
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to read {db_path}: {e}")

    return counts


def export_training_data(config: dict, output_path: Path) -> bool:
    """Export training data for a configuration."""
    board = config["board"]
    players = config["players"]
    encoder = config.get("encoder", "default")

    # Find all game databases
    game_dbs = list(GAMES_DIR.glob("*.db"))
    if not game_dbs:
        logger.warning(f"No game databases found for {board} {players}p")
        return False

    # Use the largest database
    largest_db = max(game_dbs, key=lambda p: p.stat().st_size)

    cmd = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "export_replay_dataset.py"),
        "--db", str(largest_db),
        "--output", str(output_path),
        "--board-type", board,
        "--num-players", str(players),
        "--sample-every", "2",
    ]

    if encoder == "v3" and board == "hexagonal":
        cmd.extend(["--encoder-version", "v3"])

    logger.info(f"Exporting {board} {players}p data to {output_path}")
    success, output = run_command(cmd, timeout=1800)

    if not success:
        logger.error(f"Export failed: {output[:500]}")

    return success


def train_model(config: dict, data_path: Path, run_dir: Path) -> Optional[str]:
    """Train a V3 model and return the model path if successful."""
    board = config["board"]
    players = config["players"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"{board}_{players}p_v3_{timestamp}"

    cmd = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "run_nn_training_baseline.py"),
        "--board", board,
        "--num-players", str(players),
        "--data-path", str(data_path),
        "--run-dir", str(run_dir),
        "--model-id", model_id,
        "--model-version", "v3",
        "--epochs", "100",
    ]

    logger.info(f"Training {model_id}...")
    success, output = run_command(cmd, timeout=7200)  # 2 hour timeout

    if not success:
        logger.error(f"Training failed: {output[:500]}")
        return None

    # Find the trained model
    model_files = list(run_dir.glob("*.pth")) + list(MODEL_DIR.glob(f"{model_id}*.pth"))
    if model_files:
        return str(model_files[0])

    return None


def run_tournament(board: str, players: int) -> Dict[str, float]:
    """Run an Elo tournament and return model ratings."""
    cmd = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "run_model_elo_tournament.py"),
        "--board-type", board,
        "--num-players", str(players),
        "--games-per-pair", str(TOURNAMENT_GAMES_PER_PAIR),
        "--db", str(DATA_DIR / "unified_elo.db"),
    ]

    logger.info(f"Running tournament for {board} {players}p")
    success, output = run_command(cmd, timeout=3600)

    if not success:
        logger.error(f"Tournament failed: {output[:500]}")
        return {}

    # Parse ratings from Elo database
    ratings = {}
    try:
        conn = sqlite3.connect(str(DATA_DIR / "unified_elo.db"))
        cursor = conn.execute(
            """
            SELECT model_id, rating FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
            ORDER BY rating DESC
            """,
            (board, players),
        )
        for row in cursor:
            ratings[row[0]] = row[1]
        conn.close()
    except Exception as e:
        logger.error(f"Failed to read Elo ratings: {e}")

    return ratings


def promote_model(model_path: str, board: str, players: int) -> bool:
    """Promote a model to production (copy to standard location)."""
    src = Path(model_path)
    if not src.exists():
        logger.error(f"Model not found: {model_path}")
        return False

    # Create production model name
    prod_name = f"ringrift_best_{board}_{players}p.pth"
    prod_path = MODEL_DIR / prod_name

    # Backup existing model
    if prod_path.exists():
        backup_path = MODEL_DIR / f"{prod_name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(prod_path, backup_path)
        logger.info(f"Backed up existing model to {backup_path}")

    # Copy new model
    shutil.copy(src, prod_path)
    logger.info(f"Promoted {src.name} to {prod_path}")

    # Also create MPS version for Mac compatibility
    mps_path = MODEL_DIR / f"ringrift_best_{board}_{players}p_mps.pth"
    shutil.copy(src, mps_path)

    return True


def sync_models_to_cluster(nodes: List[str]) -> bool:
    """Sync models to all cluster nodes."""
    success = True
    for node in nodes:
        cmd = [
            "rsync", "-avz", "--progress",
            str(MODEL_DIR) + "/",
            f"{node}:~/ringrift/ai-service/models/",
        ]
        ok, output = run_command(cmd, timeout=300)
        if not ok:
            logger.error(f"Failed to sync to {node}: {output[:200]}")
            success = False
        else:
            logger.info(f"Synced models to {node}")
    return success


def run_automation_cycle(state: AutomationState, dry_run: bool = False) -> None:
    """Run one automation cycle."""
    logger.info(f"=== Automation Cycle {state.total_cycles + 1} ===")

    # Get current game counts
    game_counts = get_game_counts()
    logger.info(f"Game counts: {game_counts}")

    for config in BOARD_CONFIGS:
        key = f"{config['board']}_{config['players']}p"

        # Initialize board state if needed
        if key not in state.boards:
            state.boards[key] = BoardState(
                board_type=config["board"],
                num_players=config["players"],
                encoder_version=config.get("encoder", "default"),
            )

        bs = state.boards[key]
        current_count = game_counts.get(key, 0)
        games_since_training = current_count - bs.last_game_count

        logger.info(f"{key}: {current_count} games total, {games_since_training} since last training")

        # Check if training needed
        min_games = config.get("min_games", MIN_GAMES_SINCE_TRAINING)
        if games_since_training < min_games:
            logger.info(f"  Skipping {key}: need {min_games} games, have {games_since_training}")
            continue

        if bs.training_in_progress:
            logger.info(f"  Skipping {key}: training already in progress")
            continue

        logger.info(f"  Training needed for {key}!")

        if dry_run:
            logger.info(f"  [DRY RUN] Would train {key}")
            continue

        # Mark training in progress
        bs.training_in_progress = True
        state.save(STATE_FILE)

        try:
            # Step 1: Export training data
            data_path = TRAINING_DIR / f"auto_{key}.npz"
            if not export_training_data(config, data_path):
                logger.error(f"  Export failed for {key}")
                bs.training_in_progress = False
                continue

            # Step 2: Train model
            run_dir = LOGS_DIR / f"{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir.mkdir(parents=True, exist_ok=True)

            model_path = train_model(config, data_path, run_dir)
            if not model_path:
                logger.error(f"  Training failed for {key}")
                bs.training_in_progress = False
                continue

            logger.info(f"  Trained model: {model_path}")

            # Step 3: Run tournament
            ratings = run_tournament(config["board"], config["players"])
            if not ratings:
                logger.warning(f"  Tournament failed for {key}")
            else:
                logger.info(f"  Top ratings: {dict(list(ratings.items())[:5])}")

                # Find our new model's rating
                model_name = Path(model_path).stem
                new_elo = ratings.get(model_name, 1500.0)

                # Step 4: Check if promotion needed
                elo_gain = new_elo - bs.current_best_elo
                if elo_gain >= ELO_IMPROVEMENT_THRESHOLD:
                    logger.info(f"  Promoting model! Elo gain: {elo_gain:.1f}")
                    if promote_model(model_path, config["board"], config["players"]):
                        bs.current_best_model = model_path
                        bs.current_best_elo = new_elo
                else:
                    logger.info(f"  Not promoting. Elo gain {elo_gain:.1f} < threshold {ELO_IMPROVEMENT_THRESHOLD}")

            # Update state
            bs.last_game_count = current_count
            bs.last_training_time = datetime.now().isoformat()
            bs.last_tournament_time = datetime.now().isoformat()

        finally:
            bs.training_in_progress = False
            state.save(STATE_FILE)

    # Sync models to cluster
    if not dry_run:
        sync_models_to_cluster(CLUSTER_NODES)

    state.total_cycles += 1
    state.last_cycle_time = datetime.now().isoformat()
    state.save(STATE_FILE)

    logger.info(f"=== Cycle {state.total_cycles} complete ===")


def main():
    parser = argparse.ArgumentParser(description="Full automation loop for RingRift AI")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually train, just show what would happen")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=1800, help="Seconds between cycles (default: 1800)")
    args = parser.parse_args()

    # Create directories
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load state
    state = AutomationState.load(STATE_FILE)
    logger.info(f"Loaded state: {state.total_cycles} previous cycles")

    if args.once:
        run_automation_cycle(state, dry_run=args.dry_run)
    else:
        logger.info(f"Starting automation loop (interval: {args.interval}s)")
        while True:
            try:
                run_automation_cycle(state, dry_run=args.dry_run)
            except Exception as e:
                logger.error(f"Cycle failed: {e}", exc_info=True)

            logger.info(f"Sleeping {args.interval}s until next cycle...")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
