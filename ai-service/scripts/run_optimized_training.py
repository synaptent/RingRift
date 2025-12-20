#!/usr/bin/env python3
"""Optimized Training Pipeline with All Improvements.

This script implements the recommended training loop improvements:
1. Higher batch sizes for better GPU utilization (256)
2. Victory-type balanced sampling for diverse training data
3. Transfer learning from square8 for new board types
4. Automatic Elo calibration after training completes
5. Mixed precision training (already enabled in train.py)

Usage:
    # Train all board types with optimized settings
    python scripts/run_optimized_training.py --all

    # Train specific board type with transfer learning
    python scripts/run_optimized_training.py --board hexagonal --players 2 --transfer-from square8

    # Train with victory-type balanced sampling
    python scripts/run_optimized_training.py --board square19 --players 2 --victory-balanced

    # Queue Elo tournament after training
    python scripts/run_optimized_training.py --board hexagonal --players 2 --run-elo
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Training coordination imports
try:
    from app.config.coordination_defaults import LockDefaults
    from app.coordination.distributed_lock import DistributedLock
    HAS_DISTRIBUTED_LOCK = True
except ImportError:
    HAS_DISTRIBUTED_LOCK = False
    DistributedLock = None
    LockDefaults = None

try:
    from app.training.training_registry import register_trained_model
    HAS_MODEL_REGISTRY = True
except ImportError:
    HAS_MODEL_REGISTRY = False
    register_trained_model = None

LOG_DIR = AI_SERVICE_ROOT / "logs"
MODELS_DIR = AI_SERVICE_ROOT / "models"
DATA_DIR = AI_SERVICE_ROOT / "data"
TRAINING_DIR = DATA_DIR / "training"

LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingJob:
    """Configuration for a training job."""
    board_type: str
    num_players: int
    data_path: str
    batch_size: int = 256
    epochs: int = 50
    sampling_weights: str = "victory_type"
    transfer_from: str | None = None
    run_elo_after: bool = False


# Optimized default configurations
OPTIMIZED_CONFIGS = {
    "hexagonal_2p": TrainingJob(
        board_type="hexagonal",
        num_players=2,
        data_path="data/training/hex_2p_v1.npz",
        batch_size=256,
        epochs=50,
        sampling_weights="victory_type",
        transfer_from="square8",
        run_elo_after=True,
    ),
    "hexagonal_3p": TrainingJob(
        board_type="hexagonal",
        num_players=3,
        data_path="data/training/hex_3p_v1.npz",
        batch_size=256,
        epochs=50,
        sampling_weights="combined",
        run_elo_after=True,
    ),
    "hexagonal_4p": TrainingJob(
        board_type="hexagonal",
        num_players=4,
        data_path="data/training/hex_4p_v1.npz",
        batch_size=256,
        epochs=50,
        sampling_weights="combined",
        run_elo_after=True,
    ),
    "square19_2p": TrainingJob(
        board_type="square19",
        num_players=2,
        data_path="data/training/sq19_2p_v1.npz",
        batch_size=256,
        epochs=50,
        sampling_weights="victory_type",
        transfer_from="square8",
        run_elo_after=True,
    ),
    "square19_3p": TrainingJob(
        board_type="square19",
        num_players=3,
        data_path="data/training/sq19_3p_v1.npz",
        batch_size=256,
        epochs=50,
        sampling_weights="combined",
        run_elo_after=True,
    ),
    "square19_4p": TrainingJob(
        board_type="square19",
        num_players=4,
        data_path="data/training/sq19_4p_v1.npz",
        batch_size=256,
        epochs=50,
        sampling_weights="combined",
        run_elo_after=True,
    ),
    "hex8_2p": TrainingJob(
        board_type="hex8",
        num_players=2,
        data_path="data/training/hex8_combined_v6.npz",
        batch_size=512,  # Smaller board, can use larger batch
        epochs=50,
        sampling_weights="victory_type",
        run_elo_after=True,
    ),
    "square8_2p": TrainingJob(
        board_type="square8",
        num_players=2,
        data_path="data/training/daemon_square8_2p.npz",
        batch_size=512,
        epochs=30,  # Well-trained already
        sampling_weights="victory_type",
        run_elo_after=True,
    ),
}


def find_best_source_model(board_type: str, num_players: int) -> Path | None:
    """Find the best model for transfer learning source."""
    patterns = [
        f"ringrift_best_{board_type}_{num_players}p.pth",
        f"{board_type}_{num_players}p_nn_baseline*.pth",
        f"*{board_type}_{num_players}p*.pth",
    ]

    for pattern in patterns:
        matches = list(MODELS_DIR.glob(pattern))
        if matches:
            # Return most recent
            return max(matches, key=lambda p: p.stat().st_mtime)

    return None


def run_transfer_learning(
    source_board: str,
    target_board: str,
    num_players: int,
) -> Path | None:
    """Run transfer learning to create initial weights for target board."""
    print(f"[Transfer] Looking for {source_board} model to transfer to {target_board}...")

    source_model = find_best_source_model(source_board, 2)  # Usually transfer from 2p
    if not source_model:
        print(f"[Transfer] No source model found for {source_board}, skipping transfer")
        return None

    print(f"[Transfer] Found source: {source_model}")

    # Run transfer learning script
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = MODELS_DIR / "transfer" / f"{target_board}_{num_players}p_from_{source_board}_{timestamp}.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "transfer_learning.py"),
        "--source-model", str(source_model),
        "--source-config", f"{source_board}_2p",
        "--target-config", f"{target_board}_{num_players}p",
        "--output", str(output_path),
    ]

    print(f"[Transfer] Running: {' '.join(cmd[:6])}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and output_path.exists():
            print(f"[Transfer] Success: {output_path}")
            return output_path
        else:
            print(f"[Transfer] Failed: {result.stderr[:500]}")
            return None
    except Exception as e:
        print(f"[Transfer] Error: {e}")
        return None


def run_training_job(job: TrainingJob, initial_model: Path | None = None) -> tuple[bool, str]:
    """Run a single training job."""
    config_key = f"{job.board_type}_{job.num_players}p"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = AI_SERVICE_ROOT / "runs" / f"{config_key}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Acquire distributed lock to prevent concurrent training on same config
    lock = None
    if HAS_DISTRIBUTED_LOCK:
        lock_timeout = LockDefaults.TRAINING_LOCK_TIMEOUT if LockDefaults else 7200
        lock = DistributedLock(f"training:{config_key}", lock_timeout=lock_timeout)
        if not lock.acquire(timeout=60, blocking=True):
            print(f"[Training] Could not acquire lock for {config_key}, another training may be in progress")
            return False, "Lock contention"
        print(f"[Training] Acquired lock for {config_key}")

    try:
        # Check if data exists
        data_path = AI_SERVICE_ROOT / job.data_path
        if not data_path.exists():
            return False, f"Data not found: {data_path}"

        cmd = [
            sys.executable,
            str(AI_SERVICE_ROOT / "scripts" / "run_nn_training_baseline.py"),
            "--board", job.board_type,
            "--num-players", str(job.num_players),
            "--data-path", str(data_path),
            "--run-dir", str(run_dir),
            "--epochs", str(job.epochs),
            "--batch-size", str(job.batch_size),
            "--model-version", "v3",
            "--use-optimized-hyperparams",
            "--sampling-weights", job.sampling_weights,
        ]

        # Add warmup epochs for cosine annealing
        cmd.extend(["--warmup-epochs", "5"])

        print(f"\n{'='*60}")
        print(f"[Training] {job.board_type} {job.num_players}p")
        print(f"{'='*60}")
        print(f"  Data: {job.data_path}")
        print(f"  Batch size: {job.batch_size}")
        print(f"  Epochs: {job.epochs}")
        print(f"  Sampling: {job.sampling_weights}")
        if initial_model:
            print(f"  Initial model: {initial_model}")
        print(f"  Run dir: {run_dir}")
        print()

        log_file = LOG_DIR / f"{job.board_type}_{job.num_players}p_optimized_{timestamp}.log"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(AI_SERVICE_ROOT)

        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=7200,  # 2 hour timeout
            )

        if result.returncode == 0:
            print(f"[Training] Complete. Log: {log_file}")

            # Register trained model in registry
            if HAS_MODEL_REGISTRY:
                model_path = run_dir / "best_model.pt"
                if model_path.exists():
                    try:
                        model_id = register_trained_model(
                            model_path=str(model_path),
                            board_type=job.board_type,
                            num_players=job.num_players,
                            training_config={
                                "batch_size": job.batch_size,
                                "epochs": job.epochs,
                                "sampling": job.sampling_weights,
                                "source": "run_optimized_training",
                            },
                            source_data_paths=[str(data_path)],
                        )
                        if model_id:
                            print(f"[Training] Registered model: {model_id}")
                    except Exception as e:
                        print(f"[Training] Could not register model: {e}")

            return True, str(log_file)
        else:
            print(f"[Training] Failed. Check log: {log_file}")
            return False, str(log_file)

    except subprocess.TimeoutExpired:
        print("[Training] Timeout after 2 hours")
        return False, "Timeout"
    except Exception as e:
        print(f"[Training] Error: {e}")
        return False, str(e)
    finally:
        # Release distributed lock
        if lock is not None:
            lock.release()
            print(f"[Training] Released lock for {config_key}")


def run_elo_tournament(board_type: str, num_players: int, games: int = 30) -> bool:
    """Run Elo tournament for calibration."""
    print(f"\n[Elo] Running tournament for {board_type} {num_players}p...")

    cmd = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "run_model_elo_tournament.py"),
        "--board", board_type,
        "--players", str(num_players),
        "--games", str(games),
        "--quick",
        "--include-baselines",
    ]

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(AI_SERVICE_ROOT)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )

        if result.returncode == 0:
            print("[Elo] Tournament complete")
            return True
        else:
            print(f"[Elo] Tournament failed: {result.stderr[:300]}")
            return False

    except Exception as e:
        print(f"[Elo] Error: {e}")
        return False


def run_all_optimized(configs: list[str], run_elo: bool = True) -> dict[str, bool]:
    """Run all optimized training jobs."""
    results = {}

    for config_name in configs:
        if config_name not in OPTIMIZED_CONFIGS:
            print(f"Unknown config: {config_name}")
            continue

        job = OPTIMIZED_CONFIGS[config_name]

        # Run transfer learning if specified
        initial_model = None
        if job.transfer_from:
            initial_model = run_transfer_learning(
                job.transfer_from,
                job.board_type,
                job.num_players,
            )

        # Run training
        success, _log_path = run_training_job(job, initial_model)
        results[config_name] = success

        # Run Elo tournament if requested and training succeeded
        if success and run_elo and job.run_elo_after:
            run_elo_tournament(job.board_type, job.num_players)

    return results


def main():
    parser = argparse.ArgumentParser(description="Optimized training pipeline")
    parser.add_argument("--all", action="store_true", help="Train all board types")
    parser.add_argument("--board", type=str, help="Specific board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--transfer-from", type=str, help="Source board for transfer learning")
    parser.add_argument("--victory-balanced", action="store_true", help="Use victory-type balanced sampling")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--run-elo", action="store_true", help="Run Elo tournament after training")
    parser.add_argument("--skip-elo", action="store_true", help="Skip Elo tournaments")
    parser.add_argument("--list-configs", action="store_true", help="List available configs")

    args = parser.parse_args()

    if args.list_configs:
        print("Available optimized configurations:")
        for name, job in OPTIMIZED_CONFIGS.items():
            print(f"  {name}: batch={job.batch_size}, epochs={job.epochs}, "
                  f"sampling={job.sampling_weights}, transfer={job.transfer_from or 'none'}")
        return 0

    if args.all:
        configs = list(OPTIMIZED_CONFIGS.keys())
        results = run_all_optimized(configs, run_elo=not args.skip_elo)

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        for config, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {config}: {status}")

        return 0 if all(results.values()) else 1

    if args.board:
        config_key = f"{args.board}_{args.players}p"

        if config_key in OPTIMIZED_CONFIGS:
            job = OPTIMIZED_CONFIGS[config_key]
        else:
            # Create custom job
            data_patterns = [
                f"data/training/{args.board.replace('square', 'sq')}_{args.players}p_v1.npz",
                f"data/training/daemon_{args.board}_{args.players}p.npz",
            ]
            data_path = None
            for pattern in data_patterns:
                if (AI_SERVICE_ROOT / pattern).exists():
                    data_path = pattern
                    break

            if not data_path:
                print(f"No training data found for {config_key}")
                return 1

            job = TrainingJob(
                board_type=args.board,
                num_players=args.players,
                data_path=data_path,
                batch_size=args.batch_size,
                epochs=args.epochs,
                sampling_weights="victory_type" if args.victory_balanced else "uniform",
                transfer_from=args.transfer_from,
                run_elo_after=args.run_elo,
            )

        # Override with CLI args
        if args.batch_size != 256:
            job.batch_size = args.batch_size
        if args.epochs != 50:
            job.epochs = args.epochs
        if args.victory_balanced:
            job.sampling_weights = "victory_type"
        if args.transfer_from:
            job.transfer_from = args.transfer_from

        # Run transfer learning if specified
        initial_model = None
        if job.transfer_from:
            initial_model = run_transfer_learning(
                job.transfer_from,
                job.board_type,
                job.num_players,
            )

        success, _log_path = run_training_job(job, initial_model)

        if success and args.run_elo:
            run_elo_tournament(job.board_type, job.num_players)

        return 0 if success else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
