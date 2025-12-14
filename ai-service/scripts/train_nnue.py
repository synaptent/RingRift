#!/usr/bin/env python
"""Train NNUE (Efficiently Updatable Neural Network) for RingRift Minimax.

This script trains the NNUE evaluation network used by Minimax AI at
difficulty 4+. The NNUE provides fast CPU-based position evaluation
for alpha-beta search.

Training data is extracted from self-play game databases (SQLite), where
positions are labeled with game outcomes (win/loss/draw).

Usage:
    # Train on a single database
    python scripts/train_nnue.py --db data/games/selfplay.db --epochs 50

    # Train on multiple databases
    python scripts/train_nnue.py --db data/games/*.db --epochs 100

    # Train with specific board type
    python scripts/train_nnue.py --db data/games/selfplay.db \\
        --board-type square8 --num-players 2 --epochs 50

    # Demo mode (for testing)
    python scripts/train_nnue.py --demo

Output:
    - Model checkpoint: models/nnue/nnue_{board_type}.pt
    - Training report: {run_dir}/nnue_training_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set up path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from app.ai.nnue import (
    RingRiftNNUE,
    get_feature_dim,
    get_nnue_model_path,
    clear_nnue_cache,
)
from app.models import BoardType
from app.training.nnue_dataset import (
    NNUESQLiteDataset,
    NNUEDatasetConfig,
    count_available_samples,
)
from app.training.seed_utils import seed_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def report_training_metrics(
    board_type: str,
    num_players: int,
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    epoch: int,
    model_path: str = "",
) -> None:
    """Report training metrics to the P2P orchestrator for observability.

    This posts metrics to the orchestrator's /metrics endpoint if available.
    Falls back gracefully if orchestrator is not running.
    """
    try:
        import requests

        # Try common orchestrator ports
        orchestrator_host = os.environ.get("RINGRIFT_ORCHESTRATOR_HOST", "localhost")
        orchestrator_port = int(os.environ.get("RINGRIFT_ORCHESTRATOR_PORT", "8770"))

        metrics = [
            {
                "metric_type": "training_loss",
                "value": train_loss,
                "board_type": board_type,
                "num_players": num_players,
                "metadata": {"epoch": epoch, "model_path": model_path},
            },
            {
                "metric_type": "validation_loss",
                "value": val_loss,
                "board_type": board_type,
                "num_players": num_players,
                "metadata": {"epoch": epoch, "accuracy": val_accuracy},
            },
        ]

        # Try to post to orchestrator
        for metric in metrics:
            try:
                resp = requests.post(
                    f"http://{orchestrator_host}:{orchestrator_port}/metrics/record",
                    json=metric,
                    timeout=2,
                )
                if resp.status_code == 200:
                    logger.debug(f"Reported {metric['metric_type']} to orchestrator")
            except Exception:
                pass  # Orchestrator not available, skip silently

    except ImportError:
        pass  # requests not available
    except Exception as e:
        logger.debug(f"Metrics reporting error (non-fatal): {e}")


def parse_board_type(value: str) -> BoardType:
    """Parse board type string to enum."""
    mapping = {
        "square8": BoardType.SQUARE8,
        "sq8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "sq19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
        "hex": BoardType.HEXAGONAL,
    }
    key = value.lower()
    if key not in mapping:
        raise ValueError(f"Unknown board type: {value}")
    return mapping[key]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NNUE evaluation network for RingRift Minimax AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data sources
    parser.add_argument(
        "--db",
        type=str,
        nargs="+",
        default=[],
        help="Path(s) to SQLite game database(s). Supports glob patterns.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to cache extracted features as NPZ (speeds up repeated runs).",
    )

    # Board configuration
    parser.add_argument(
        "--board-type",
        type=str,
        default="square8",
        help="Board type: square8, square19, or hexagonal (default: square8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )

    # Training parameters - tuned for high-quality training on H100/5090 GPUs
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size (default: 512, use 1024+ for H100)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4, lower for stable convergence)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for L2 regularization (default: 1e-4)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation set fraction (default: 0.1)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (default: 15)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="NNUE hidden layer dimension (default: 256)",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=2,
        help="Number of NNUE hidden layers (default: 2)",
    )

    # Sampling configuration
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=1,
        help="Sample every Nth position from games (default: 1 = all)",
    )
    parser.add_argument(
        "--min-game-length",
        type=int,
        default=10,
        help="Minimum game length to include (default: 10)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of training samples (default: all)",
    )

    # Output
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output directory for reports (default: runs/nnue_{timestamp})",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID for checkpoint (default: auto-generated)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Custom path for model checkpoint (default: models/nnue/nnue_{board}.pt)",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a tiny demo training (for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (default: auto-detect)",
    )

    return parser.parse_args(argv)


def create_demo_dataset(
    board_type: BoardType,
    num_samples: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a synthetic demo dataset for testing."""
    logger.info(f"Creating synthetic demo dataset with {num_samples} samples")

    feature_dim = get_feature_dim(board_type)
    features = np.random.rand(num_samples, feature_dim).astype(np.float32)
    values = np.random.choice([-1.0, 0.0, 1.0], size=num_samples).astype(np.float32)

    return features, values


class NNUETrainer:
    """Trainer for NNUE evaluation network."""

    def __init__(
        self,
        model: RingRiftNNUE,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            # Note: verbose param removed - deprecated in PyTorch 2.3+
        )
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for features, values in dataloader:
            features = features.to(self.device)
            values = values.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, values)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate on held-out data. Returns (loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            for features, values in dataloader:
                features = features.to(self.device)
                values = values.to(self.device)

                predictions = self.model(features)
                loss = self.criterion(predictions, values)
                total_loss += loss.item()
                num_batches += 1

                # Calculate accuracy (correct sign prediction)
                pred_sign = torch.sign(predictions)
                true_sign = torch.sign(values)
                correct += (pred_sign == true_sign).sum().item()
                total += values.numel()

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def update_scheduler(self, val_loss: float) -> None:
        """Update learning rate scheduler based on validation loss."""
        self.scheduler.step(val_loss)


def train_nnue(
    db_paths: List[str],
    board_type: BoardType,
    num_players: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    val_split: float,
    early_stopping_patience: int,
    hidden_dim: int,
    num_hidden_layers: int,
    sample_every_n: int,
    min_game_length: int,
    max_samples: Optional[int],
    save_path: str,
    device: torch.device,
    seed: int,
    cache_path: Optional[str] = None,
    demo: bool = False,
) -> Dict[str, Any]:
    """Train NNUE model and return training report."""
    seed_all(seed)

    # Create dataset
    if demo:
        features, values = create_demo_dataset(board_type, num_samples=1000)
        # Wrap in simple dataset
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(values[:, None]),
        )
    else:
        config = NNUEDatasetConfig(
            board_type=board_type,
            num_players=num_players,
            sample_every_n_moves=sample_every_n,
            min_game_length=min_game_length,
        )
        dataset = NNUESQLiteDataset(
            db_paths=db_paths,
            config=config,
            cache_path=cache_path,
            max_samples=max_samples,
        )

    if len(dataset) == 0:
        logger.error("No training samples found!")
        return {"error": "No training samples"}

    logger.info(f"Dataset size: {len(dataset)} samples")

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # SQLite doesn't like multiprocessing
        pin_memory=True if device.type != "cpu" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type != "cpu" else False,
    )

    # Create model
    model = RingRiftNNUE(
        board_type=board_type,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = NNUETrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_accuracy = trainer.validate(val_loader)
        trainer.update_scheduler(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}"
        )

        # Report metrics to orchestrator for observability
        report_training_metrics(
            board_type=board_type,
            num_players=num_players,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            epoch=epoch + 1,
            model_path=str(output_path) if output_path else "",
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Save best model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "board_type": board_type.value,
                "hidden_dim": hidden_dim,
                "num_hidden_layers": num_hidden_layers,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "architecture_version": model.ARCHITECTURE_VERSION,
            }
            torch.save(checkpoint, save_path)
            logger.info(f"Saved best model to {save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Training report
    report = {
        "board_type": board_type.value,
        "num_players": num_players,
        "dataset_size": len(dataset),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "model_params": sum(p.numel() for p in model.parameters()),
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_accuracy": history["val_accuracy"][-1],
        "save_path": save_path,
        "history": history,
    }

    return report


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Parse board type
    board_type = parse_board_type(args.board_type)

    # Expand glob patterns in database paths
    db_paths: List[str] = []
    for pattern in args.db:
        expanded = glob.glob(pattern)
        if expanded:
            db_paths.extend(expanded)
        elif os.path.exists(pattern):
            db_paths.append(pattern)
        else:
            logger.warning(f"Database not found: {pattern}")

    if not db_paths and not args.demo:
        logger.error("No database paths provided. Use --db or --demo")
        return 1

    # Set up device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Set up output paths
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(PROJECT_ROOT, "runs", f"nnue_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    model_id = args.model_id or f"nnue_{board_type.value}_{args.num_players}p"
    save_path = args.save_path or str(get_nnue_model_path(board_type, args.num_players))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Count available samples (unless demo mode)
    if not args.demo and db_paths:
        config = NNUEDatasetConfig(
            board_type=board_type,
            num_players=args.num_players,
            sample_every_n_moves=args.sample_every_n,
            min_game_length=args.min_game_length,
        )
        sample_counts = count_available_samples(db_paths, config)
        logger.info(f"Available samples: {sample_counts.get('total', 0)}")

    # Train
    logger.info("Starting NNUE training...")
    report = train_nnue(
        db_paths=db_paths,
        board_type=board_type,
        num_players=args.num_players,
        epochs=args.epochs if not args.demo else 5,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        early_stopping_patience=args.early_stopping_patience,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        sample_every_n=args.sample_every_n,
        min_game_length=args.min_game_length,
        max_samples=args.max_samples,
        save_path=save_path,
        device=device,
        seed=args.seed,
        cache_path=args.cache_path,
        demo=args.demo,
    )

    # Add metadata to report
    report["model_id"] = model_id
    report["run_dir"] = run_dir
    report["db_paths"] = db_paths
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["demo_mode"] = args.demo

    # Save report
    report_path = os.path.join(run_dir, "nnue_training_report.json")
    with open(report_path, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        json.dump(report, f, indent=2, default=convert)
    logger.info(f"Saved training report to {report_path}")

    # Clear NNUE cache so new model is loaded
    clear_nnue_cache()

    logger.info("NNUE training complete!")
    logger.info(f"  Model saved to: {save_path}")
    logger.info(f"  Best validation loss: {report.get('best_val_loss', 'N/A'):.4f}")
    logger.info(f"  Final validation accuracy: {report.get('final_val_accuracy', 'N/A'):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
