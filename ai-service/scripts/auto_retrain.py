#!/usr/bin/env python
"""
Automated retraining script for canonical selfplay data.

Monitors canonical databases and triggers retraining when:
1. New games exceed threshold since last training
2. Minimum cooldown period has elapsed
3. Database passes canonical gate validation

Usage:
  # One-shot check and train if needed
  python scripts/auto_retrain.py --check

  # Daemon mode (continuous monitoring)
  python scripts/auto_retrain.py --daemon --interval 1800

  # Train specific config regardless of thresholds
  python scripts/auto_retrain.py --force --board hex8 --players 4

  # Dry run to see what would be trained
  python scripts/auto_retrain.py --check --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Unified game discovery - finds all game databases across all storage patterns
try:
    from app.utils.game_discovery import GameDiscovery, count_games_for_config
    HAS_GAME_DISCOVERY = True
except ImportError:
    HAS_GAME_DISCOVERY = False
    GameDiscovery = None
    count_games_for_config = None

# All board/player configurations
CONFIGS = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
]

# Default thresholds
DEFAULT_MIN_NEW_GAMES = 50  # Minimum new games to trigger retraining
DEFAULT_MIN_TOTAL_GAMES = 100  # Minimum total games required
DEFAULT_COOLDOWN_HOURS = 4  # Hours between retraining same config
DEFAULT_CHECK_INTERVAL = 1800  # Seconds between checks in daemon mode


@dataclass
class TrainingState:
    """Tracks training state for a single config."""
    board: str
    players: int
    last_trained_at: Optional[datetime] = None
    last_trained_games: int = 0
    last_model_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "board": self.board,
            "players": self.players,
            "last_trained_at": self.last_trained_at.isoformat() if self.last_trained_at else None,
            "last_trained_games": self.last_trained_games,
            "last_model_path": self.last_model_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingState":
        return cls(
            board=data["board"],
            players=data["players"],
            last_trained_at=datetime.fromisoformat(data["last_trained_at"]) if data.get("last_trained_at") else None,
            last_trained_games=data.get("last_trained_games", 0),
            last_model_path=data.get("last_model_path"),
        )


@dataclass
class RetrainConfig:
    """Configuration for auto-retraining."""
    min_new_games: int = DEFAULT_MIN_NEW_GAMES
    min_total_games: int = DEFAULT_MIN_TOTAL_GAMES
    cooldown_hours: float = DEFAULT_COOLDOWN_HOURS
    check_interval: int = DEFAULT_CHECK_INTERVAL
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "games")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "nnue")
    state_file: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / ".auto_retrain_state.json")
    epochs: int = 50
    batch_size: int = 256
    dry_run: bool = False


def get_db_path(config: RetrainConfig, board: str, players: int) -> Path:
    """Get path to canonical database for a config."""
    return config.data_dir / f"canonical_{board}_{players}p.db"


def get_model_path(config: RetrainConfig, board: str, players: int) -> Path:
    """Get path to output model for a config."""
    return config.models_dir / f"nnue_{board}_{players}p_canonical.pth"


def count_games(db_path: Path) -> int:
    """Count games in a database."""
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM games")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.warning(f"Failed to count games in {db_path}: {e}")
        return 0


def count_games_for_config_all_sources(board: str, players: int) -> int:
    """Count games for a config across ALL databases using unified discovery.

    This finds games in central DBs, selfplay DBs, P2P DBs, etc.
    """
    if HAS_GAME_DISCOVERY:
        return count_games_for_config(board, players, PROJECT_ROOT)
    else:
        # Fallback to canonical DB only
        db_path = PROJECT_ROOT / "data" / "games" / f"canonical_{board}_{players}p.db"
        return count_games(db_path)


def check_canonical_gate(db_path: Path) -> bool:
    """Check if database passes canonical gate."""
    if not db_path.exists():
        return False

    # Check for gate summary file
    summary_path = db_path.with_suffix(".gate_summary.json")
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            return summary.get("canonical_ok", False)
        except Exception:
            pass

    # If no summary, check for parity markers in DB
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT COUNT(*) FROM games WHERE game_status = 'completed'"
        )
        completed = cursor.fetchone()[0]
        conn.close()
        # Require at least some completed games
        return completed > 0
    except Exception:
        return False


def load_state(config: RetrainConfig) -> dict[str, TrainingState]:
    """Load training state from disk."""
    states = {}
    if config.state_file.exists():
        try:
            with open(config.state_file) as f:
                data = json.load(f)
            for key, state_dict in data.items():
                states[key] = TrainingState.from_dict(state_dict)
        except Exception as e:
            logger.warning(f"Failed to load state file: {e}")

    # Ensure all configs have state entries
    for board, players in CONFIGS:
        key = f"{board}_{players}p"
        if key not in states:
            states[key] = TrainingState(board=board, players=players)

    return states


def save_state(config: RetrainConfig, states: dict[str, TrainingState]) -> None:
    """Save training state to disk."""
    config.state_file.parent.mkdir(parents=True, exist_ok=True)
    data = {key: state.to_dict() for key, state in states.items()}
    with open(config.state_file, "w") as f:
        json.dump(data, f, indent=2)


def should_train(
    config: RetrainConfig,
    state: TrainingState,
    current_games: int,
) -> tuple[bool, str]:
    """Determine if a config should be retrained."""
    # Check minimum total games
    if current_games < config.min_total_games:
        return False, f"insufficient games ({current_games} < {config.min_total_games})"

    # Check new games since last training
    new_games = current_games - state.last_trained_games
    if new_games < config.min_new_games:
        return False, f"insufficient new games ({new_games} < {config.min_new_games})"

    # Check cooldown
    if state.last_trained_at:
        cooldown_end = state.last_trained_at + timedelta(hours=config.cooldown_hours)
        if datetime.now() < cooldown_end:
            remaining = cooldown_end - datetime.now()
            return False, f"cooldown active ({remaining.seconds // 60}m remaining)"

    return True, f"ready ({new_games} new games)"


def run_training(
    config: RetrainConfig,
    board: str,
    players: int,
    db_path: Path,
) -> bool:
    """Run training for a single config."""
    model_path = get_model_path(config, board, players)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_canonical_training.py"),
        "--db", str(db_path),
        "--board-type", board,
        "--num-players", str(players),
        "--model-version", "v4",
        "--epochs", str(config.epochs),
        "--batch-size", str(config.batch_size),
        "--save-path", str(model_path),
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    if config.dry_run:
        logger.info("[DRY RUN] Would execute training")
        return True

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )

        if result.returncode == 0:
            logger.info(f"Training completed successfully for {board} {players}p")
            return True
        else:
            logger.error(f"Training failed for {board} {players}p")
            logger.error(f"STDERR: {result.stderr[-1000:]}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out for {board} {players}p")
        return False
    except Exception as e:
        logger.error(f"Training error for {board} {players}p: {e}")
        return False


def check_and_train(config: RetrainConfig, force_board: Optional[str] = None, force_players: Optional[int] = None) -> dict:
    """Check all configs and train those that need it."""
    states = load_state(config)
    results = {
        "checked": 0,
        "trained": 0,
        "skipped": 0,
        "failed": 0,
        "details": [],
    }

    configs_to_check = CONFIGS
    if force_board and force_players:
        configs_to_check = [(force_board, force_players)]

    for board, players in configs_to_check:
        key = f"{board}_{players}p"
        state = states[key]
        db_path = get_db_path(config, board, players)

        results["checked"] += 1

        # Check if DB exists and has data
        current_games = count_games(db_path)
        if current_games == 0:
            results["details"].append({
                "config": key,
                "status": "skipped",
                "reason": "no data",
            })
            results["skipped"] += 1
            continue

        # Check canonical gate
        if not check_canonical_gate(db_path):
            results["details"].append({
                "config": key,
                "status": "skipped",
                "reason": "failed canonical gate",
            })
            results["skipped"] += 1
            continue

        # Check if should train (skip check if forcing)
        if force_board and force_players:
            should, reason = True, "forced"
        else:
            should, reason = should_train(config, state, current_games)

        if not should:
            results["details"].append({
                "config": key,
                "status": "skipped",
                "reason": reason,
                "games": current_games,
            })
            results["skipped"] += 1
            continue

        # Run training
        logger.info(f"Training {key}: {reason}")
        success = run_training(config, board, players, db_path)

        if success:
            state.last_trained_at = datetime.now()
            state.last_trained_games = current_games
            state.last_model_path = str(get_model_path(config, board, players))
            results["trained"] += 1
            results["details"].append({
                "config": key,
                "status": "trained",
                "games": current_games,
            })
        else:
            results["failed"] += 1
            results["details"].append({
                "config": key,
                "status": "failed",
                "games": current_games,
            })

    save_state(config, states)
    return results


def daemon_loop(config: RetrainConfig) -> None:
    """Run continuous monitoring loop."""
    logger.info(f"Starting daemon mode (interval: {config.check_interval}s)")

    while True:
        try:
            logger.info("Checking for retraining needs...")
            results = check_and_train(config)

            logger.info(
                f"Check complete: {results['trained']} trained, "
                f"{results['skipped']} skipped, {results['failed']} failed"
            )

            for detail in results["details"]:
                if detail["status"] == "trained":
                    logger.info(f"  ✓ {detail['config']}: trained ({detail['games']} games)")
                elif detail["status"] == "failed":
                    logger.warning(f"  ✗ {detail['config']}: failed")

        except Exception as e:
            logger.error(f"Error in daemon loop: {e}")

        logger.info(f"Sleeping for {config.check_interval}s...")
        time.sleep(config.check_interval)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automated retraining for canonical selfplay data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="One-shot check and train if needed.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in continuous monitoring mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force training regardless of thresholds.",
    )
    parser.add_argument(
        "--board",
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Specific board type to train (with --force).",
    )
    parser.add_argument(
        "--players",
        type=int,
        choices=[2, 3, 4],
        help="Specific player count to train (with --force).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be trained without executing.",
    )
    parser.add_argument(
        "--min-new-games",
        type=int,
        default=DEFAULT_MIN_NEW_GAMES,
        help=f"Minimum new games to trigger retraining (default: {DEFAULT_MIN_NEW_GAMES}).",
    )
    parser.add_argument(
        "--min-total-games",
        type=int,
        default=DEFAULT_MIN_TOTAL_GAMES,
        help=f"Minimum total games required (default: {DEFAULT_MIN_TOTAL_GAMES}).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_HOURS,
        help=f"Hours between retraining same config (default: {DEFAULT_COOLDOWN_HOURS}).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_CHECK_INTERVAL,
        help=f"Seconds between checks in daemon mode (default: {DEFAULT_CHECK_INTERVAL}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (default: 256).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current training state and exit.",
    )

    args = parser.parse_args()

    config = RetrainConfig(
        min_new_games=args.min_new_games,
        min_total_games=args.min_total_games,
        cooldown_hours=args.cooldown,
        check_interval=args.interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    if args.status:
        states = load_state(config)
        print("\n=== Auto-Retrain Status ===\n")
        for key in sorted(states.keys()):
            state = states[key]
            db_path = get_db_path(config, state.board, state.players)
            current_games = count_games(db_path)
            new_games = current_games - state.last_trained_games

            print(f"{key}:")
            print(f"  Current games: {current_games}")
            print(f"  Last trained: {state.last_trained_at or 'never'}")
            print(f"  Games at last train: {state.last_trained_games}")
            print(f"  New games: {new_games}")
            if state.last_model_path:
                print(f"  Model: {state.last_model_path}")
            print()
        return 0

    if args.force:
        if not args.board or not args.players:
            parser.error("--force requires both --board and --players")
        results = check_and_train(config, force_board=args.board, force_players=args.players)
    elif args.daemon:
        daemon_loop(config)
        return 0
    elif args.check:
        results = check_and_train(config)
    else:
        parser.print_help()
        return 1

    # Print results
    print("\n=== Retraining Results ===")
    print(f"Checked: {results['checked']}")
    print(f"Trained: {results['trained']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Failed: {results['failed']}")
    print("\nDetails:")
    for detail in results["details"]:
        status_icon = {"trained": "✓", "skipped": "○", "failed": "✗"}.get(detail["status"], "?")
        games_str = f" ({detail.get('games', 0)} games)" if "games" in detail else ""
        reason_str = f" - {detail.get('reason', '')}" if detail.get("reason") else ""
        print(f"  {status_icon} {detail['config']}: {detail['status']}{games_str}{reason_str}")

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
