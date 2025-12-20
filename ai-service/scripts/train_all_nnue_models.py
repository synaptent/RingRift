#!/usr/bin/env python
"""Train NNUE models for all board type and player count combinations.

This script orchestrates NNUE training for all 9 configurations:
- square8: 2p, 3p, 4p
- square19: 2p, 3p, 4p
- hexagonal: 2p, 3p, 4p

It aggregates training data from all available game databases and trains
models in priority order based on data availability.

Usage:
    # Train all models sequentially
    python scripts/train_all_nnue_models.py

    # Train specific configurations
    python scripts/train_all_nnue_models.py --configs square8:2 square8:3

    # Skip existing models
    python scripts/train_all_nnue_models.py --skip-existing

    # Use specific databases
    python scripts/train_all_nnue_models.py --db data/games/lambda_h100_selfplay.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("train_all_nnue_models")


@dataclass
class TrainingConfig:
    """Configuration for a single NNUE training run."""
    board_type: str
    num_players: int
    min_games: int = 100  # Minimum games required to train
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    early_stopping_patience: int = 20
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    sample_every_n: int = 2
    min_game_length: int = 15


# Default configurations for all board/player combinations
DEFAULT_CONFIGS = [
    # Square8 - most data typically
    TrainingConfig("square8", 2, min_games=50),
    TrainingConfig("square8", 3, min_games=100),
    TrainingConfig("square8", 4, min_games=100),
    # Square19 - larger board, fewer games typically
    TrainingConfig("square19", 2, min_games=50, batch_size=128),
    TrainingConfig("square19", 3, min_games=50, batch_size=128),
    TrainingConfig("square19", 4, min_games=50, batch_size=128),
    # Hexagonal - largest feature dimension
    TrainingConfig("hexagonal", 2, min_games=50, batch_size=128),
    TrainingConfig("hexagonal", 3, min_games=50, batch_size=128),
    TrainingConfig("hexagonal", 4, min_games=50, batch_size=128),
]


def get_game_counts(db_paths: List[str]) -> Dict[Tuple[str, int], int]:
    """Count games per board type and player count across all databases."""
    counts: Dict[Tuple[str, int], int] = {}

    for db_path in db_paths:
        if not os.path.exists(db_path):
            continue
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("""
                SELECT board_type, num_players, COUNT(*) as count
                FROM games
                GROUP BY board_type, num_players
            """)
            for row in cursor:
                key = (row[0], row[1])
                counts[key] = counts.get(key, 0) + row[2]
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to read {db_path}: {e}")

    return counts


def find_databases(data_dir: str = "data/games") -> List[str]:
    """Find all SQLite game databases in the data directory."""
    db_paths = []
    data_path = Path(data_dir)

    if data_path.exists():
        for db_file in data_path.glob("*.db"):
            # Skip temporary and quarantine databases
            if any(x in db_file.name for x in ["tmp_", "quarantine", "holdout"]):
                continue
            db_paths.append(str(db_file))

    return sorted(db_paths)


def get_model_path(board_type: str, num_players: int) -> str:
    """Get the expected model path for a configuration."""
    return f"models/nnue/nnue_{board_type}_{num_players}p.pt"


def train_nnue(
    config: TrainingConfig,
    db_paths: List[str],
    run_dir: str,
    force: bool = False,
) -> bool:
    """Train NNUE model for a specific configuration.

    Returns True if training succeeded, False otherwise.
    """
    model_path = get_model_path(config.board_type, config.num_players)

    if os.path.exists(model_path) and not force:
        logger.info(f"Model {model_path} exists, skipping (use --force to retrain)")
        return True

    logger.info(f"Training NNUE for {config.board_type} {config.num_players}p...")

    # Build command
    cmd = [
        sys.executable, "scripts/train_nnue.py",
        "--db", *db_paths,
        "--board-type", config.board_type,
        "--num-players", str(config.num_players),
        "--epochs", str(config.epochs),
        "--batch-size", str(config.batch_size),
        "--learning-rate", str(config.learning_rate),
        "--early-stopping-patience", str(config.early_stopping_patience),
        "--hidden-dim", str(config.hidden_dim),
        "--num-hidden-layers", str(config.num_hidden_layers),
        "--sample-every-n", str(config.sample_every_n),
        "--min-game-length", str(config.min_game_length),
        "--run-dir", run_dir,
    ]

    logger.info(f"  Command: {' '.join(cmd[:5])}...")

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode == 0:
            logger.info(f"  Training completed successfully")
            return True
        else:
            logger.error(f"  Training failed: {result.stderr[-500:]}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"  Training timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"  Training error: {e}")
        return False


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train NNUE models for all configurations"
    )
    parser.add_argument(
        "--db",
        nargs="+",
        default=None,
        help="Specific database paths to use (default: auto-detect)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Specific configs to train (e.g., 'square8:2' 'hexagonal:3')",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training if model already exists",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if model exists",
    )
    parser.add_argument(
        "--data-dir",
        default="data/games",
        help="Directory containing game databases",
    )
    parser.add_argument(
        "--run-dir-base",
        default="runs/nnue_auto",
        help="Base directory for training runs",
    )

    args = parser.parse_args(args)

    # Find databases
    if args.db:
        db_paths = args.db
    else:
        db_paths = find_databases(args.data_dir)

    if not db_paths:
        logger.error("No game databases found")
        return 1

    logger.info(f"Found {len(db_paths)} databases:")
    for db in db_paths[:5]:
        logger.info(f"  {db}")
    if len(db_paths) > 5:
        logger.info(f"  ... and {len(db_paths) - 5} more")

    # Count games per configuration
    counts = get_game_counts(db_paths)
    logger.info("\nGame counts per configuration:")
    for (bt, np), count in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {bt} {np}p: {count} games")

    # Filter configurations
    if args.configs:
        configs = []
        for spec in args.configs:
            parts = spec.split(":")
            bt = parts[0]
            np = int(parts[1]) if len(parts) > 1 else 2
            for c in DEFAULT_CONFIGS:
                if c.board_type == bt and c.num_players == np:
                    configs.append(c)
                    break
    else:
        configs = DEFAULT_CONFIGS

    # Sort by data availability (most data first)
    configs = sorted(
        configs,
        key=lambda c: counts.get((c.board_type, c.num_players), 0),
        reverse=True,
    )

    # Train each configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    for config in configs:
        key = (config.board_type, config.num_players)
        game_count = counts.get(key, 0)

        if game_count < config.min_games:
            logger.warning(
                f"Skipping {config.board_type} {config.num_players}p: "
                f"only {game_count} games (need {config.min_games})"
            )
            results[key] = "skipped_insufficient_data"
            continue

        if args.skip_existing and os.path.exists(get_model_path(*key)):
            logger.info(f"Skipping {config.board_type} {config.num_players}p: model exists")
            results[key] = "skipped_exists"
            continue

        run_dir = f"{args.run_dir_base}/{config.board_type}_{config.num_players}p_{timestamp}"
        os.makedirs(run_dir, exist_ok=True)

        success = train_nnue(config, db_paths, run_dir, force=args.force)
        results[key] = "success" if success else "failed"

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Training Summary:")
    logger.info("=" * 50)

    for (bt, np), status in sorted(results.items()):
        status_icon = {
            "success": "✓",
            "failed": "✗",
            "skipped_exists": "⊘",
            "skipped_insufficient_data": "○",
        }.get(status, "?")
        logger.info(f"  {status_icon} {bt} {np}p: {status}")

    success_count = sum(1 for s in results.values() if s == "success")
    logger.info(f"\n{success_count}/{len(results)} models trained successfully")

    return 0 if all(s in ("success", "skipped_exists") for s in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
