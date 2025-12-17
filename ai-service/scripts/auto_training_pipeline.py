#!/usr/bin/env python3
"""Automated NNUE Training Pipeline.

This script orchestrates the full training cycle:
1. Collect game data from Lambda, Vast, and Hetzner nodes
2. Merge into consolidated training database
3. Backfill snapshots for games missing periodic state saves
4. Train NNUE model on the consolidated data
5. Sync trained model back to all nodes

Usage:
    python scripts/auto_training_pipeline.py
    python scripts/auto_training_pipeline.py --dry-run
    python scripts/auto_training_pipeline.py --skip-collect --skip-backfill

Can be run via cron for automated daily/weekly training:
    0 4 * * * cd /path/to/ai-service && python scripts/auto_training_pipeline.py >> logs/auto_train.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = AI_SERVICE_ROOT / "data"
GAMES_DIR = DATA_DIR / "games"
MODELS_DIR = AI_SERVICE_ROOT / "models" / "nnue"
RUNS_DIR = AI_SERVICE_ROOT / "runs"

LOG_DIR = AI_SERVICE_ROOT / "logs"
LOG_FILE = LOG_DIR / "auto_training.log"

# Node configurations
LAMBDA_NODES = [
    "lambda-gh200-a",
    "lambda-gh200-b",
    "lambda-gh200-c",
    "lambda-gh200-d",
    "lambda-gh200-e",
    "lambda-gh200-f",
    "lambda-gh200-g",
    "lambda-gh200-h",
    "lambda-gh200-i",
    # "lambda-gh200-j",  # Often offline
    "lambda-gh200-k",
    "lambda-a10",
]

HETZNER_NODES = [
    ("ringrift-cpu1", "46.62.147.150", "root"),
    ("ringrift-cpu2", "135.181.39.239", "root"),
    ("ringrift-cpu3", "46.62.217.168", "root"),
]

# Default paths on remote nodes
LAMBDA_DB_PATH = "/home/ubuntu/ringrift/ai-service/data/games"
HETZNER_DB_PATH = "/root/ringrift/ai-service/data/games"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_ssh_command(host: str, command: str, user: str = "ubuntu", timeout: int = 30) -> Tuple[bool, str]:
    """Run SSH command and return (success, output)."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=accept-new",
             f"{user}@{host}" if user else host, command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def collect_from_lambda(collect_dir: Path, dry_run: bool = False) -> int:
    """Collect game databases from Lambda nodes."""
    logger.info("Collecting from Lambda nodes...")
    collected = 0

    for node in LAMBDA_NODES:
        # Find all DBs on the node
        success, output = run_ssh_command(
            node,
            f"find {LAMBDA_DB_PATH} -name '*.db' -type f 2>/dev/null | head -20",
            timeout=15,
        )
        if not success:
            logger.warning(f"  {node}: unreachable")
            continue

        db_paths = [p.strip() for p in output.split("\n") if p.strip().endswith(".db")]
        if not db_paths:
            logger.info(f"  {node}: no DBs found")
            continue

        for remote_path in db_paths:
            db_name = Path(remote_path).name
            local_path = collect_dir / f"{node}_{db_name}"

            if dry_run:
                logger.info(f"  {node}: would collect {db_name}")
                collected += 1
                continue

            try:
                result = subprocess.run(
                    ["scp", "-o", "ConnectTimeout=10", f"ubuntu@{node}:{remote_path}", str(local_path)],
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    logger.info(f"  {node}: collected {db_name}")
                    collected += 1
            except Exception as e:
                logger.warning(f"  {node}: failed to collect {db_name}: {e}")

    return collected


def collect_from_hetzner(collect_dir: Path, dry_run: bool = False) -> int:
    """Collect game databases from Hetzner nodes."""
    logger.info("Collecting from Hetzner nodes...")
    collected = 0

    for name, ip, user in HETZNER_NODES:
        # Find all DBs on the node
        success, output = run_ssh_command(
            ip,
            f"find {HETZNER_DB_PATH} -name '*.db' -type f 2>/dev/null | head -20",
            user=user,
            timeout=15,
        )
        if not success:
            logger.warning(f"  {name}: unreachable")
            continue

        db_paths = [p.strip() for p in output.split("\n") if p.strip().endswith(".db")]
        if not db_paths:
            logger.info(f"  {name}: no DBs found")
            continue

        for remote_path in db_paths:
            db_name = Path(remote_path).name
            local_path = collect_dir / f"{name}_{db_name}"

            if dry_run:
                logger.info(f"  {name}: would collect {db_name}")
                collected += 1
                continue

            try:
                result = subprocess.run(
                    ["scp", "-o", "ConnectTimeout=10", f"{user}@{ip}:{remote_path}", str(local_path)],
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    logger.info(f"  {name}: collected {db_name}")
                    collected += 1
            except Exception as e:
                logger.warning(f"  {name}: failed to collect {db_name}: {e}")

    return collected


def collect_from_vast(collect_dir: Path, dry_run: bool = False) -> int:
    """Collect game databases from Vast instances via vast_lifecycle sync."""
    logger.info("Collecting from Vast instances...")

    # Use vast_lifecycle.py --sync to collect data
    if dry_run:
        logger.info("  Would run vast_lifecycle.py --sync")
        return 0

    try:
        result = subprocess.run(
            ["python", str(AI_SERVICE_ROOT / "scripts" / "vast_lifecycle.py"), "--sync"],
            cwd=str(AI_SERVICE_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "PYTHONPATH": str(AI_SERVICE_ROOT)},
        )
        # Count lines mentioning games synced
        import re
        matches = re.findall(r"(\d+) games", result.stdout)
        total = sum(int(m) for m in matches)
        logger.info(f"  Vast sync complete: ~{total} games collected")
        return 1 if total > 0 else 0
    except Exception as e:
        logger.warning(f"  Vast collection failed: {e}")
        return 0


def merge_databases(collect_dir: Path, output_db: Path, dry_run: bool = False) -> bool:
    """Merge all collected databases into a single training database."""
    logger.info(f"Merging databases into {output_db}...")

    dbs = list(collect_dir.glob("*.db"))
    if not dbs:
        logger.warning("No databases to merge")
        return False

    logger.info(f"  Found {len(dbs)} databases to merge")

    if dry_run:
        logger.info("  Would merge databases")
        return True

    # Build merge command
    cmd = [
        "python", str(AI_SERVICE_ROOT / "scripts" / "merge_game_dbs.py"),
        "--output", str(output_db),
        "--dedupe-by-game-id",
    ]
    for db in dbs:
        cmd.extend(["--db", str(db)])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(AI_SERVICE_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONPATH": str(AI_SERVICE_ROOT)},
        )
        if result.returncode == 0:
            logger.info("  Merge complete")
            return True
        else:
            logger.error(f"  Merge failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"  Merge error: {e}")
        return False


def run_backfill(db_path: Path, dry_run: bool = False) -> int:
    """Run snapshot backfill on the database."""
    logger.info(f"Running snapshot backfill on {db_path}...")

    if dry_run:
        logger.info("  Would run backfill")
        return 0

    try:
        result = subprocess.run(
            [
                "python", str(AI_SERVICE_ROOT / "scripts" / "backfill_snapshots.py"),
                "--db", str(db_path),
                "--interval", "20",
            ],
            cwd=str(AI_SERVICE_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # May take a while for large DBs
            env={**os.environ, "PYTHONPATH": str(AI_SERVICE_ROOT)},
        )
        # Parse output for snapshot count
        import re
        match = re.search(r"Snapshots created: (\d+)", result.stdout)
        if match:
            count = int(match.group(1))
            logger.info(f"  Created {count} snapshots")
            return count
        return 0
    except Exception as e:
        logger.error(f"  Backfill error: {e}")
        return 0


def train_nnue(db_path: Path, board_type: str = "square8", num_players: int = 2, dry_run: bool = False) -> Optional[Path]:
    """Train NNUE model on the consolidated database."""
    logger.info(f"Training NNUE model for {board_type}_{num_players}p...")

    if dry_run:
        logger.info("  Would train NNUE model")
        return None

    try:
        result = subprocess.run(
            [
                "python", str(AI_SERVICE_ROOT / "scripts" / "train_nnue.py"),
                "--db", str(db_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--epochs", "50",
            ],
            cwd=str(AI_SERVICE_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,
            env={**os.environ, "PYTHONPATH": str(AI_SERVICE_ROOT)},
        )
        if result.returncode == 0:
            # Find the saved model path
            model_path = MODELS_DIR / f"nnue_{board_type}_{num_players}p.pt"
            if model_path.exists():
                logger.info(f"  Training complete: {model_path}")
                return model_path
        else:
            logger.error(f"  Training failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"  Training error: {e}")
        return None


def train_nn_optimized(
    data_path: Path,
    board_type: str = "square8",
    num_players: int = 2,
    dry_run: bool = False,
    batch_size: int = 256,
    epochs: int = 50,
    sampling_weights: str = "victory_type",
) -> Optional[Path]:
    """Train neural network model with optimized settings.

    Uses the new optimized training pipeline with:
    - Victory-type balanced sampling
    - Higher batch sizes for GPU utilization
    - Warmup epochs and cosine annealing
    - Board-specific hyperparameters from config/hyperparameters.json
    """
    logger.info(f"Training NN model for {board_type}_{num_players}p with optimized settings...")
    logger.info(f"  batch_size={batch_size}, epochs={epochs}, sampling={sampling_weights}")

    if dry_run:
        logger.info("  Would train NN model with optimized settings")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{board_type}_{num_players}p_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_id = f"{board_type}_{num_players}p_v3_{timestamp}"

    try:
        cmd = [
            "python", str(AI_SERVICE_ROOT / "scripts" / "run_nn_training_baseline.py"),
            "--board", board_type,
            "--num-players", str(num_players),
            "--data-path", str(data_path),
            "--run-dir", str(run_dir),
            "--model-id", model_id,
            "--model-version", "v3",
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--sampling-weights", sampling_weights,
            "--use-optimized-hyperparams",
            "--warmup-epochs", "5",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(AI_SERVICE_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout for full training
            env={**os.environ, "PYTHONPATH": str(AI_SERVICE_ROOT)},
        )
        if result.returncode == 0:
            # Find the saved model path
            model_path = AI_SERVICE_ROOT / "models" / f"{model_id}.pth"
            if model_path.exists():
                logger.info(f"  Training complete: {model_path}")
                return model_path
            # Also check alternate location
            alt_path = run_dir / f"{model_id}.pth"
            if alt_path.exists():
                logger.info(f"  Training complete: {alt_path}")
                return alt_path
            logger.warning(f"  Training finished but model not found at expected paths")
            return None
        else:
            logger.error(f"  Training failed: {result.stderr[:500]}")
            return None
    except subprocess.TimeoutExpired:
        logger.error("  Training timed out after 2 hours")
        return None
    except Exception as e:
        logger.error(f"  Training error: {e}")
        return None


def sync_model_to_nodes(model_path: Path, dry_run: bool = False) -> int:
    """Sync trained model to all nodes."""
    logger.info(f"Syncing model {model_path.name} to nodes...")

    if dry_run:
        logger.info("  Would sync to nodes")
        return 0

    synced = 0

    # Sync to Lambda nodes
    for node in LAMBDA_NODES:
        try:
            result = subprocess.run(
                ["scp", "-o", "ConnectTimeout=10", str(model_path),
                 f"ubuntu@{node}:/home/ubuntu/ringrift/ai-service/models/nnue/"],
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 0:
                synced += 1
                logger.info(f"  {node}: synced")
        except Exception as e:
            logger.warning(f"  {node}: sync failed - {e}")

    # Sync to Hetzner nodes
    for name, ip, user in HETZNER_NODES:
        try:
            result = subprocess.run(
                ["scp", "-o", "ConnectTimeout=10", str(model_path),
                 f"{user}@{ip}:/root/ringrift/ai-service/models/nnue/"],
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 0:
                synced += 1
                logger.info(f"  {name}: synced")
        except Exception as e:
            logger.warning(f"  {name}: sync failed - {e}")

    return synced


def run_pipeline(
    skip_collect: bool = False,
    skip_backfill: bool = False,
    skip_train: bool = False,
    skip_sync: bool = False,
    dry_run: bool = False,
    board_type: str = "square8",
    num_players: int = 2,
    use_optimized: bool = True,
    batch_size: int = 256,
    sampling_weights: str = "victory_type",
):
    """Run the full training pipeline."""
    logger.info("=" * 60)
    logger.info("AUTOMATED NNUE TRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN MODE]")

    # Create collection directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collect_dir = DATA_DIR / f"collected_{timestamp}"
    output_db = DATA_DIR / "consolidated_training.db"

    # Step 1: Collect data
    if not skip_collect:
        logger.info("")
        logger.info("STEP 1: Collecting game data from nodes...")
        collect_dir.mkdir(parents=True, exist_ok=True)

        lambda_count = collect_from_lambda(collect_dir, dry_run)
        hetzner_count = collect_from_hetzner(collect_dir, dry_run)
        vast_count = collect_from_vast(collect_dir, dry_run)

        total_collected = lambda_count + hetzner_count + vast_count
        logger.info(f"Total collected: {total_collected} database files")

        if total_collected == 0 and not dry_run:
            logger.warning("No data collected, skipping merge")
        else:
            # Merge databases
            merge_databases(collect_dir, output_db, dry_run)

            # Cleanup collection directory
            if not dry_run and collect_dir.exists():
                shutil.rmtree(collect_dir)
    else:
        logger.info("Skipping data collection")

    # Step 2: Backfill snapshots
    if not skip_backfill:
        logger.info("")
        logger.info("STEP 2: Backfilling snapshots...")
        if output_db.exists():
            run_backfill(output_db, dry_run)
        else:
            logger.warning(f"Training DB not found: {output_db}")
    else:
        logger.info("Skipping backfill")

    # Step 3: Train model
    model_path = None
    if not skip_train:
        logger.info("")
        if use_optimized:
            logger.info("STEP 3: Training NN model with optimized settings...")
            # Export data to NPZ for optimized training
            npz_path = DATA_DIR / "training" / f"{board_type}_{num_players}p_auto.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)

            if output_db.exists() and not npz_path.exists():
                # Export from DB to NPZ
                logger.info(f"  Exporting data to {npz_path}...")
                export_cmd = [
                    "python", str(AI_SERVICE_ROOT / "scripts" / "export_replay_dataset.py"),
                    "--db", str(output_db),
                    "--output", str(npz_path),
                    "--board-type", board_type,
                    "--num-players", str(num_players),
                    "--sample-every", "2",
                    "--require-completed",
                ]
                try:
                    subprocess.run(
                        export_cmd,
                        cwd=str(AI_SERVICE_ROOT),
                        capture_output=True,
                        timeout=1800,
                        env={**os.environ, "PYTHONPATH": str(AI_SERVICE_ROOT)},
                    )
                except Exception as e:
                    logger.error(f"  Export failed: {e}")

            if npz_path.exists():
                model_path = train_nn_optimized(
                    npz_path, board_type, num_players, dry_run,
                    batch_size=batch_size, sampling_weights=sampling_weights,
                )
            else:
                logger.warning(f"NPZ data not found: {npz_path}")
        else:
            logger.info("STEP 3: Training NNUE model...")
            if output_db.exists():
                model_path = train_nnue(output_db, board_type, num_players, dry_run)
            else:
                logger.warning(f"Training DB not found: {output_db}")
    else:
        logger.info("Skipping training")

    # Step 4: Sync model to nodes
    if not skip_sync and model_path:
        logger.info("")
        logger.info("STEP 4: Syncing model to nodes...")
        synced = sync_model_to_nodes(model_path, dry_run)
        logger.info(f"Model synced to {synced} nodes")
    else:
        logger.info("Skipping model sync")

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Finished at: {datetime.now().isoformat()}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Automated Training Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--skip-collect", action="store_true", help="Skip data collection")
    parser.add_argument("--skip-backfill", action="store_true", help="Skip snapshot backfill")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-sync", action="store_true", help="Skip model sync to nodes")
    parser.add_argument("--board-type", default="square8", help="Board type for training")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    # Optimized training settings
    parser.add_argument("--use-optimized", action="store_true", default=True,
                        help="Use optimized NN training (default: True)")
    parser.add_argument("--use-nnue", action="store_true",
                        help="Use legacy NNUE training instead of optimized NN")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for optimized training (default: 256)")
    parser.add_argument("--sampling-weights", type=str, default="victory_type",
                        choices=["uniform", "late_game", "phase_emphasis", "combined", "victory_type"],
                        help="Sampling strategy (default: victory_type)")

    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --use-nnue overrides --use-optimized
    use_optimized = not args.use_nnue

    run_pipeline(
        skip_collect=args.skip_collect,
        skip_backfill=args.skip_backfill,
        skip_train=args.skip_train,
        skip_sync=args.skip_sync,
        dry_run=args.dry_run,
        board_type=args.board_type,
        num_players=args.num_players,
        use_optimized=use_optimized,
        batch_size=args.batch_size,
        sampling_weights=args.sampling_weights,
    )


if __name__ == "__main__":
    main()
