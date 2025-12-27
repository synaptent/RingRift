#!/usr/bin/env python3
"""Focused training script for square8_2p optimization to reach >2000 Elo.

December 2025 Optimization Campaign
===================================

This script runs a complete training pipeline optimized for square8_2p:
1. Export fresh training data from canonical database
2. Train with optimized hyperparameters (batch_size=256, lr=0.003)
3. Run gauntlet evaluation
4. Auto-promote if passing thresholds

Key optimizations applied:
- batch_size: 64 → 256 (4x larger, better gradient estimates)
- learning_rate: 0.0009 → 0.003 (3x higher, faster convergence)
- early_stopping_patience: 10 → 25 (more time to converge)
- Uses existing 16K+ games from canonical_square8_2p.db

Usage:
    # Full pipeline (export → train → evaluate → promote)
    python scripts/train_square8_2p_optimized.py

    # Skip export (use existing NPZ)
    python scripts/train_square8_2p_optimized.py --skip-export

    # Training only (skip evaluation)
    python scripts/train_square8_2p_optimized.py --train-only

    # Dry run (show what would happen)
    python scripts/train_square8_2p_optimized.py --dry-run
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/train_sq8_2p_opt_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
logger = logging.getLogger(__name__)

# Paths
AI_SERVICE_ROOT = Path(__file__).parent.parent
DATA_DIR = AI_SERVICE_ROOT / "data"
GAMES_DIR = DATA_DIR / "games"
TRAINING_DIR = DATA_DIR / "training"
MODELS_DIR = AI_SERVICE_ROOT / "models"

# Configuration
CONFIG_KEY = "square8_2p"
BOARD_TYPE = "square8"
NUM_PLAYERS = 2

# Optimized hyperparameters (Dec 2025)
BATCH_SIZE = 256
LEARNING_RATE = 0.003
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 25

# Database and NPZ paths
CANONICAL_DB = GAMES_DIR / "canonical_square8_2p.db"
TRAINING_NPZ = TRAINING_DIR / "square8_2p_optimized.npz"
MODEL_OUTPUT = MODELS_DIR / "square8_2p_optimized.pth"


def run_command(cmd: list[str], dry_run: bool = False) -> int:
    """Run a command and return exit code."""
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"Running: {cmd_str}")

    if dry_run:
        logger.info("[DRY RUN] Would execute above command")
        return 0

    result = subprocess.run(cmd, cwd=AI_SERVICE_ROOT)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
    return result.returncode


def export_training_data(dry_run: bool = False) -> bool:
    """Export training data from canonical database."""
    logger.info("=" * 60)
    logger.info("Step 1: Export Training Data")
    logger.info("=" * 60)

    if not CANONICAL_DB.exists():
        logger.error(f"Canonical database not found: {CANONICAL_DB}")
        logger.info("Run selfplay first to generate games")
        return False

    # Ensure output directory exists
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/export_replay_dataset.py",
        "--db", str(CANONICAL_DB),
        "--board-type", BOARD_TYPE,
        "--num-players", str(NUM_PLAYERS),
        "--output", str(TRAINING_NPZ),
        "--sample-rate", "1.0",  # Use all samples
    ]

    exit_code = run_command(cmd, dry_run)
    if exit_code != 0:
        return False

    if not dry_run:
        if TRAINING_NPZ.exists():
            size_mb = TRAINING_NPZ.stat().st_size / (1024 * 1024)
            logger.info(f"Exported training data: {TRAINING_NPZ} ({size_mb:.1f} MB)")
        else:
            logger.error("Export failed - NPZ file not created")
            return False

    return True


def train_model(dry_run: bool = False) -> bool:
    """Train the model with optimized hyperparameters."""
    logger.info("=" * 60)
    logger.info("Step 2: Train Model (Optimized Hyperparameters)")
    logger.info("=" * 60)
    logger.info(f"  batch_size: {BATCH_SIZE}")
    logger.info(f"  learning_rate: {LEARNING_RATE}")
    logger.info(f"  epochs: {EPOCHS}")
    logger.info(f"  early_stopping_patience: {EARLY_STOPPING_PATIENCE}")

    if not dry_run and not TRAINING_NPZ.exists():
        logger.error(f"Training NPZ not found: {TRAINING_NPZ}")
        logger.info("Run with --skip-export=false to generate training data")
        return False

    cmd = [
        sys.executable,
        "-m", "app.training.train",
        "--board-type", BOARD_TYPE,
        "--num-players", str(NUM_PLAYERS),
        "--data-path", str(TRAINING_NPZ),
        "--save-path", str(MODEL_OUTPUT),
        "--model-version", "v2",
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LEARNING_RATE),
        "--epochs", str(EPOCHS),
        "--early-stopping-patience", str(EARLY_STOPPING_PATIENCE),
        "--lr-scheduler", "cosine",
        "--warmup-epochs", "5",
    ]

    exit_code = run_command(cmd, dry_run)
    if exit_code != 0:
        return False

    if not dry_run:
        if MODEL_OUTPUT.exists():
            size_mb = MODEL_OUTPUT.stat().st_size / (1024 * 1024)
            logger.info(f"Model saved: {MODEL_OUTPUT} ({size_mb:.1f} MB)")
        else:
            logger.error("Training failed - model file not created")
            return False

    return True


def run_gauntlet(dry_run: bool = False) -> tuple[bool, float | None]:
    """Run gauntlet evaluation and return (passed, elo)."""
    logger.info("=" * 60)
    logger.info("Step 3: Gauntlet Evaluation")
    logger.info("=" * 60)

    if not dry_run and not MODEL_OUTPUT.exists():
        logger.error(f"Model not found: {MODEL_OUTPUT}")
        return False, None

    cmd = [
        sys.executable,
        "scripts/quick_gauntlet.py",
        "--model", str(MODEL_OUTPUT),
        "--board-type", BOARD_TYPE,
        "--num-players", str(NUM_PLAYERS),
        "--games", "50",
    ]

    exit_code = run_command(cmd, dry_run)

    # For dry run, assume it would pass
    if dry_run:
        return True, 1800.0

    # Parse gauntlet results (exit code 0 = passed)
    passed = exit_code == 0

    # TODO: Parse Elo from gauntlet output
    # For now, we just report pass/fail
    return passed, None


def promote_model(dry_run: bool = False) -> bool:
    """Promote model to canonical if it passes gauntlet."""
    logger.info("=" * 60)
    logger.info("Step 4: Promote Model")
    logger.info("=" * 60)

    canonical_path = MODELS_DIR / f"canonical_{CONFIG_KEY}.pth"

    if dry_run:
        logger.info(f"[DRY RUN] Would copy {MODEL_OUTPUT} to {canonical_path}")
        return True

    if not MODEL_OUTPUT.exists():
        logger.error(f"Model not found: {MODEL_OUTPUT}")
        return False

    # Copy to canonical location
    import shutil
    shutil.copy2(MODEL_OUTPUT, canonical_path)
    logger.info(f"Promoted model to: {canonical_path}")

    # Update symlink
    symlink_path = MODELS_DIR / f"ringrift_best_{CONFIG_KEY}.pth"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(canonical_path.name)
    logger.info(f"Updated symlink: {symlink_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Focused training script for square8_2p optimization"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export step, use existing NPZ",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Training only, skip evaluation and promotion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Square8 2P Optimization Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Target: >2000 Elo")
    logger.info(f"Config: {CONFIG_KEY}")
    logger.info(f"Canonical DB: {CANONICAL_DB}")

    # Ensure logs directory exists
    (AI_SERVICE_ROOT / "logs").mkdir(exist_ok=True)

    # Step 1: Export
    if not args.skip_export:
        if not export_training_data(args.dry_run):
            logger.error("Export failed, aborting")
            return 1
    else:
        logger.info("Skipping export (--skip-export)")

    # Step 2: Train
    if not train_model(args.dry_run):
        logger.error("Training failed, aborting")
        return 1

    if args.train_only:
        logger.info("Training complete (--train-only)")
        return 0

    # Step 3: Gauntlet
    passed, elo = run_gauntlet(args.dry_run)
    if not passed:
        logger.warning("Gauntlet failed - model not promoted")
        logger.info("Model saved but not promoted. Review gauntlet results.")
        return 1

    # Step 4: Promote
    if not promote_model(args.dry_run):
        logger.error("Promotion failed")
        return 1

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    if elo:
        logger.info(f"Estimated Elo: {elo:.0f}")
    logger.info(f"Model promoted to: {MODELS_DIR / f'canonical_{CONFIG_KEY}.pth'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
