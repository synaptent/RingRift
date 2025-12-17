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
import yaml

# Ramdrive utilities for high-speed I/O
from app.utils.ramdrive import add_ramdrive_args, get_config_from_args, get_data_directory, RamdriveSyncer

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
    NNUEStreamingDataset,
    NNUEDatasetConfig,
    PrioritizedExperienceSampler,
    count_available_samples,
)
from app.training.seed_utils import seed_all

# Unified resource guard - 80% utilization limits (enforced 2025-12-16)
try:
    from app.utils.resource_guard import (
        check_disk_space,
        check_memory,
        get_degradation_level,
        should_proceed_with_priority,
        OperationPriority,
        get_resource_status,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    check_disk_space = lambda *args, **kwargs: True
    check_memory = lambda *args, **kwargs: True
    get_degradation_level = lambda: 0
    should_proceed_with_priority = lambda p: True
    OperationPriority = type('OperationPriority', (), {'HIGH': 3})()
    get_resource_status = lambda: {'can_proceed': True}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Board-specific hyperparameters config path
HYPERPARAMS_CONFIG = Path(__file__).parent.parent / "config" / "training_hyperparams.yaml"


def load_board_hyperparams(board_type: str, num_players: int = 2) -> Dict[str, Any]:
    """Load board-specific training hyperparameters from config file.

    Args:
        board_type: Board type (e.g., "square8", "square19", "hexagonal")
        num_players: Number of players

    Returns:
        Dict of hyperparameters, or empty dict if config not found
    """
    if not HYPERPARAMS_CONFIG.exists():
        logger.debug(f"Hyperparams config not found: {HYPERPARAMS_CONFIG}")
        return {}

    try:
        with open(HYPERPARAMS_CONFIG) as f:
            config = yaml.safe_load(f)

        # Determine config key based on board type and players
        board_lower = board_type.lower()
        if board_lower in ("square8", "sq8"):
            key = "square8_mp" if num_players > 2 else "square8_2p"
        elif board_lower in ("square19", "sq19"):
            key = "square19"
        elif board_lower in ("hexagonal", "hex"):
            key = "hexagonal"
        else:
            key = "default"

        # Get board-specific config, fall back to default
        params = config.get(key, config.get("default", {}))

        # Merge with mixed precision settings if available
        mp_config = config.get("mixed_precision", {}).get(key, {})
        if mp_config:
            params["mixed_precision_enabled"] = mp_config.get("enabled", False)
            params["mixed_precision_dtype"] = mp_config.get("dtype", "float16")

        logger.info(f"Loaded hyperparams for {key}: {list(params.keys())}")
        return params

    except Exception as e:
        logger.warning(f"Failed to load hyperparams config: {e}")
        return {}


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
        "--adaptive-batch",
        action="store_true",
        help="Enable adaptive batch sizing based on GPU memory",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=32,
        help="Minimum batch size for adaptive mode (default: 32)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4096,
        help="Maximum batch size for adaptive mode (default: 4096)",
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
        "--lr-schedule",
        type=str,
        choices=["plateau", "cosine", "warmup_cosine"],
        default="warmup_cosine",
        help="Learning rate schedule: plateau (reduce on plateau), cosine (cosine annealing), "
             "warmup_cosine (linear warmup + cosine) (default: warmup_cosine)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for warmup_cosine schedule (default: 5)",
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
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Use phase-balanced sampling (25% early, 35% mid, 40% late game)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming DataLoader for memory-efficient training on large datasets",
    )
    parser.add_argument(
        "--streaming-buffer",
        type=int,
        default=10000,
        help="Buffer size for streaming dataset shuffling (default: 10000)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0, use main process)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training (FP16/BF16) for faster training on GPU",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="float16",
        help="Mixed precision dtype: float16 (default) or bfloat16 (better for newer GPUs)",
    )
    parser.add_argument(
        "--per",
        action="store_true",
        help="Enable Prioritized Experience Replay for smarter sample weighting",
    )
    parser.add_argument(
        "--per-alpha",
        type=float,
        default=0.6,
        help="PER prioritization exponent (0=uniform, 1=full priority) (default: 0.6)",
    )
    parser.add_argument(
        "--per-beta",
        type=float,
        default=0.4,
        help="PER importance sampling correction (0=no, 1=full) (default: 0.4)",
    )
    parser.add_argument(
        "--use-board-config",
        action="store_true",
        help="Use board-specific hyperparameters from config/training_hyperparams.yaml",
    )
    parser.add_argument(
        "--early-end",
        type=int,
        default=40,
        help="Move number where early game ends (default: 40)",
    )
    parser.add_argument(
        "--mid-end",
        type=int,
        default=80,
        help="Move number where mid game ends (default: 80)",
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

    # Add ramdrive storage options
    add_ramdrive_args(parser)

    return parser.parse_args(argv)


def find_optimal_batch_size(
    model: nn.Module,
    feature_dim: int,
    device: torch.device,
    min_batch: int = 32,
    max_batch: int = 4096,
    target_batch: int = 512,
) -> int:
    """Find optimal batch size by testing GPU memory capacity.

    Uses binary search to find the largest batch size that fits in memory.

    Args:
        model: The model to test with
        feature_dim: Feature dimension for dummy input
        device: Target device
        min_batch: Minimum batch size to try
        max_batch: Maximum batch size to try
        target_batch: Initial target batch size

    Returns:
        Optimal batch size
    """
    if device.type == "cpu":
        logger.info(f"CPU training: using target batch size {target_batch}")
        return target_batch

    model = model.to(device)
    model.train()

    # Start with target, then binary search if it fails
    batch_sizes_to_try = []

    # First try target, then powers of 2 around it
    current = target_batch
    while current >= min_batch:
        batch_sizes_to_try.append(current)
        current //= 2
    batch_sizes_to_try = sorted(set(batch_sizes_to_try), reverse=True)

    optimal = min_batch
    for batch_size in batch_sizes_to_try:
        if batch_size > max_batch:
            continue

        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Create dummy batch
            dummy_input = torch.randn(batch_size, feature_dim, device=device)
            dummy_target = torch.randn(batch_size, 1, device=device)

            # Forward pass
            output = model(dummy_input)
            loss = nn.MSELoss()(output, dummy_target)

            # Backward pass
            loss.backward()

            # If we get here, this batch size works
            optimal = batch_size
            logger.info(f"Batch size {batch_size} fits in GPU memory")

            # Clean up
            del dummy_input, dummy_target, output, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"Batch size {batch_size} too large, trying smaller")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise

    # Try to go higher if we found a working size below target
    if optimal < target_batch and optimal < max_batch:
        # Try sizes between optimal and target
        for multiplier in [1.5, 1.25]:
            test_size = int(optimal * multiplier)
            if test_size > max_batch or test_size <= optimal:
                continue

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                dummy_input = torch.randn(test_size, feature_dim, device=device)
                dummy_target = torch.randn(test_size, 1, device=device)
                output = model(dummy_input)
                loss = nn.MSELoss()(output, dummy_target)
                loss.backward()

                optimal = test_size
                logger.info(f"Batch size {test_size} also fits in GPU memory")

                del dummy_input, dummy_target, output, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break

    # Reset model gradients
    model.zero_grad()

    logger.info(f"Selected optimal batch size: {optimal}")
    return optimal


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
        lr_schedule: str = "warmup_cosine",
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        use_amp: bool = False,
        amp_dtype: str = "float16",
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_schedule = lr_schedule
        self.current_epoch = 0

        # Mixed precision training setup
        self.use_amp = use_amp and device.type == "cuda"
        if amp_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16

        # GradScaler for mixed precision (only needed for float16)
        self.scaler = None
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled with FP16 + GradScaler")
        elif self.use_amp:
            logger.info(f"Mixed precision training enabled with {amp_dtype}")

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate if lr_schedule != "warmup_cosine" else 1e-7,  # Start low for warmup
            weight_decay=weight_decay,
        )

        # Set up learning rate scheduler based on schedule type
        if lr_schedule == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        elif lr_schedule == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=learning_rate * 0.01,
            )
        elif lr_schedule == "warmup_cosine":
            # Combined warmup + cosine annealing
            # During warmup: linear increase from 1e-7 to learning_rate
            # After warmup: cosine decay from learning_rate to learning_rate * 0.01
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_epochs - warmup_epochs),
                eta_min=learning_rate * 0.01,
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

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

            if self.use_amp:
                # Mixed precision forward pass
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    predictions = self.model(features)
                    loss = self.criterion(predictions, values)

                if self.scaler is not None:
                    # FP16: use scaler for gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # BF16: no scaler needed
                    loss.backward()
                    self.optimizer.step()
            else:
                # Standard precision
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

                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                        predictions = self.model(features)
                        loss = self.criterion(predictions, values)
                else:
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
        self.current_epoch += 1

        if self.lr_schedule == "warmup_cosine":
            if self.current_epoch <= self.warmup_epochs:
                # Linear warmup: increase LR from 1e-7 to learning_rate
                warmup_factor = self.current_epoch / self.warmup_epochs
                new_lr = 1e-7 + (self.learning_rate - 1e-7) * warmup_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
            else:
                # After warmup, use cosine scheduler
                self.scheduler.step()
        elif self.lr_schedule == "plateau":
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


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
    balanced_sampling: bool = False,
    early_end: int = 40,
    mid_end: int = 80,
    lr_schedule: str = "warmup_cosine",
    warmup_epochs: int = 5,
    streaming: bool = False,
    streaming_buffer: int = 10000,
    num_workers: int = 0,
    adaptive_batch: bool = False,
    min_batch_size: int = 32,
    max_batch_size: int = 4096,
    mixed_precision: bool = False,
    amp_dtype: str = "float16",
    per: bool = False,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
) -> Dict[str, Any]:
    """Train NNUE model and return training report."""
    seed_all(seed)

    # Dataset configuration
    config = NNUEDatasetConfig(
        board_type=board_type,
        num_players=num_players,
        sample_every_n_moves=sample_every_n,
        min_game_length=min_game_length,
    )

    # Create datasets based on mode
    streaming_dataset = None
    dataset_size = 0

    if demo:
        features, values = create_demo_dataset(board_type, num_samples=1000)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(values[:, None]),
        )
        dataset_size = len(dataset)
    elif streaming:
        # Streaming mode: use IterableDataset for training
        # For validation, load a small subset into memory
        logger.info("Using streaming DataLoader for memory-efficient training")

        # Split DBs: 90% for training stream, 10% for validation
        np.random.seed(seed)
        shuffled_dbs = db_paths.copy()
        np.random.shuffle(shuffled_dbs)
        val_db_count = max(1, int(len(shuffled_dbs) * val_split))
        val_dbs = shuffled_dbs[:val_db_count]
        train_dbs = shuffled_dbs[val_db_count:]

        if not train_dbs:
            train_dbs = val_dbs  # Fall back if only 1 DB

        streaming_dataset = NNUEStreamingDataset(
            db_paths=train_dbs,
            config=config,
            shuffle_games=True,
            seed=seed,
            buffer_size=streaming_buffer,
        )

        # Load validation set into memory (smaller subset)
        val_dataset = NNUESQLiteDataset(
            db_paths=val_dbs,
            config=config,
            max_samples=max_samples // 10 if max_samples else 10000,
        )
        dataset_size = -1  # Unknown for streaming
        logger.info(f"Streaming from {len(train_dbs)} DBs, validation from {len(val_dbs)} DBs")
    else:
        # Standard mode: load all into memory
        dataset = NNUESQLiteDataset(
            db_paths=db_paths,
            config=config,
            cache_path=cache_path,
            max_samples=max_samples,
        )
        dataset_size = len(dataset)

    if not streaming and dataset_size == 0:
        logger.error("No training samples found!")
        return {"error": "No training samples"}

    if dataset_size > 0:
        logger.info(f"Dataset size: {dataset_size} samples")

    # Split into train/val for non-streaming mode
    if not streaming and not demo:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    elif demo:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

    # Create data loaders
    train_sampler = None
    per_sampler = None

    if per and not streaming and not demo:
        # Prioritized Experience Replay sampler
        logger.info(f"Using PER sampler with alpha={per_alpha}, beta={per_beta}")
        per_sampler = PrioritizedExperienceSampler(
            dataset_size=len(train_dataset),
            alpha=per_alpha,
            beta=per_beta,
            beta_schedule=True,
        )
        train_sampler = per_sampler
    elif not streaming and balanced_sampling and not demo and hasattr(dataset, 'get_balanced_sampler'):
        logger.info("Using phase-balanced sampling for training")
        from torch.utils.data import WeightedRandomSampler
        train_indices = train_dataset.indices
        weights = dataset.compute_phase_balanced_weights(
            early_end=early_end,
            mid_end=mid_end,
            target_balance=(0.25, 0.35, 0.40),
        )
        train_weights = weights[train_indices]
        train_weights = train_weights / train_weights.sum()
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(train_weights).double(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        logger.info(f"Created balanced sampler for {len(train_dataset)} training samples")

    # Create train loader
    if streaming:
        train_loader = DataLoader(
            streaming_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True if device.type != "cpu" else False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
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

    # Adaptive batch sizing - find optimal batch size for GPU
    actual_batch_size = batch_size
    if adaptive_batch and not demo:
        logger.info("Finding optimal batch size for GPU...")
        feature_dim = get_feature_dim(board_type)
        actual_batch_size = find_optimal_batch_size(
            model=model,
            feature_dim=feature_dim,
            device=device,
            min_batch=min_batch_size,
            max_batch=max_batch_size,
            target_batch=batch_size,
        )
        if actual_batch_size != batch_size:
            logger.info(f"Adjusted batch size: {batch_size} -> {actual_batch_size}")
            # Recreate data loaders with new batch size
            if streaming:
                train_loader = DataLoader(
                    streaming_dataset,
                    batch_size=actual_batch_size,
                    num_workers=num_workers,
                    pin_memory=True if device.type != "cpu" else False,
                )
            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=actual_batch_size,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    num_workers=num_workers,
                    pin_memory=True if device.type != "cpu" else False,
                )
            val_loader = DataLoader(
                val_dataset,
                batch_size=actual_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if device.type != "cpu" else False,
            )

    # Create trainer
    trainer = NNUETrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_schedule=lr_schedule,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        use_amp=mixed_precision,
        amp_dtype=amp_dtype,
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    for epoch in range(epochs):
        # Update streaming dataset epoch for proper shuffling
        if streaming and streaming_dataset is not None:
            streaming_dataset.set_epoch(epoch)

        # Update PER sampler epoch for beta annealing
        if per_sampler is not None:
            per_sampler.set_epoch(epoch, epochs)

        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_accuracy = trainer.validate(val_loader)
        trainer.update_scheduler(val_loss)

        # Update PER priorities based on prediction errors (every 5 epochs)
        if per_sampler is not None and (epoch + 1) % 5 == 0:
            logger.info("Updating PER priorities...")
            with torch.no_grad():
                model.eval()
                all_errors = []
                all_indices = list(range(len(train_dataset)))
                eval_batch = actual_batch_size if actual_batch_size else batch_size
                for idx in range(0, len(train_dataset), eval_batch):
                    batch_indices = all_indices[idx:idx + eval_batch]
                    batch_features = []
                    batch_values = []
                    for i in batch_indices:
                        feat, val = train_dataset[i]
                        batch_features.append(feat)
                        batch_values.append(val)
                    features = torch.stack(batch_features).to(device)
                    values = torch.stack(batch_values).to(device)
                    preds = model(features)
                    errors = (preds - values).abs().cpu().numpy().flatten()
                    all_errors.extend(errors)
                per_sampler.update_priorities(all_indices, np.array(all_errors))
                model.train()
                stats = per_sampler.get_stats()
                logger.info(f"PER stats: mean_priority={stats['mean_priority']:.4f}, seen={stats['seen_ratio']:.1%}")

        current_lr = trainer.get_current_lr()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(current_lr)
        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}, "
            f"lr={current_lr:.2e}"
        )

        # Report metrics to orchestrator for observability
        report_training_metrics(
            board_type=board_type,
            num_players=num_players,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            epoch=epoch + 1,
            model_path=str(save_path) if save_path else "",
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
        "dataset_size": dataset_size if dataset_size > 0 else "streaming",
        "train_size": "streaming" if streaming else len(train_dataset),
        "val_size": len(val_dataset),
        "model_params": sum(p.numel() for p in model.parameters()),
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_accuracy": history["val_accuracy"][-1],
        "save_path": save_path,
        "balanced_sampling": balanced_sampling,
        "phase_config": {"early_end": early_end, "mid_end": mid_end} if balanced_sampling else None,
        "lr_schedule": lr_schedule,
        "warmup_epochs": warmup_epochs if lr_schedule == "warmup_cosine" else None,
        "streaming": streaming,
        "streaming_buffer": streaming_buffer if streaming else None,
        "num_workers": num_workers,
        "adaptive_batch": adaptive_batch,
        "requested_batch_size": batch_size,
        "actual_batch_size": actual_batch_size,
        "mixed_precision": mixed_precision,
        "amp_dtype": amp_dtype if mixed_precision else None,
        "per": per,
        "per_alpha": per_alpha if per else None,
        "per_beta": per_beta if per else None,
        "history": history,
    }

    return report


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Resource guard: Training is HIGH priority (3)
    # Check resources before starting memory-intensive training
    # Brief warmup delay to let process launch CPU spike settle
    skip_resource_guard = os.environ.get("RINGRIFT_SKIP_RESOURCE_GUARD", "").lower() in ("1", "true", "yes")
    if HAS_RESOURCE_GUARD and not skip_resource_guard:
        import time
        time.sleep(1)  # Allow transient CPU spike from process launch to settle
        degradation = get_degradation_level()
        if degradation >= 4:  # CRITICAL - resources at/above limits
            logger.error("Resources at critical levels (degradation=4), aborting training")
            return 1
        elif degradation >= 3:  # HEAVY
            if not should_proceed_with_priority(OperationPriority.HIGH):
                logger.error("Heavy resource pressure (degradation=3), training blocked")
                return 1
            logger.warning("Heavy resource pressure, training proceeding with HIGH priority")
        elif degradation >= 2:  # MODERATE
            logger.info(f"Moderate resource pressure (degradation={degradation})")

        # Check specific resources
        if not check_memory(required_gb=2.0):
            logger.warning("Memory constrained, training may be slow")
        if not check_disk_space(required_gb=1.0):
            logger.warning("Disk space low, checkpoint saving may fail")

        status = get_resource_status()
        logger.info(f"Resource check: disk={status['disk']['used_percent']:.1f}%, "
                   f"memory={status['memory']['used_percent']:.1f}%, "
                   f"degradation={degradation}")

    # Parse board type
    board_type = parse_board_type(args.board_type)

    # Apply board-specific hyperparameters if requested
    if getattr(args, 'use_board_config', False):
        board_config = load_board_hyperparams(args.board_type, args.num_players)
        if board_config:
            logger.info(f"Applying board-specific hyperparameters for {args.board_type}")
            # Override args with board-specific values (only if not explicitly set by user)
            for key in ['learning_rate', 'weight_decay', 'batch_size', 'epochs',
                        'hidden_dim', 'num_hidden_layers', 'early_stopping_patience',
                        'lr_schedule', 'warmup_epochs', 'val_split', 'sample_every_n',
                        'min_game_length', 'balanced_sampling']:
                if key in board_config:
                    config_key = key.replace('_', '-')
                    # Check if user explicitly provided this arg (it's at default value)
                    # For simplicity, always apply board config values
                    setattr(args, key.replace('-', '_'), board_config[key])
                    logger.debug(f"  {key} = {board_config[key]}")

            # Apply mixed precision settings from board config
            if board_config.get('mixed_precision_enabled') and not args.mixed_precision:
                args.mixed_precision = True
                args.amp_dtype = board_config.get('mixed_precision_dtype', 'float16')
                logger.info(f"  Mixed precision enabled: {args.amp_dtype}")

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

    # Set up output paths with optional ramdrive support
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    syncer = None

    # Use ramdrive for run_dir if requested (faster training logs/checkpoints)
    if getattr(args, 'ram_storage', False) and not args.run_dir:
        ramdrive_config = get_config_from_args(args)
        ramdrive_config.subdirectory = f"training/nnue_{timestamp}"
        run_dir = str(get_data_directory(prefer_ramdrive=True, config=ramdrive_config, base_name="runs"))
        logger.info(f"Using ramdrive for training output: {run_dir}")

        # Set up periodic sync to persistent storage
        sync_interval = getattr(args, 'sync_interval', 0)
        sync_target = getattr(args, 'sync_target', '')
        if sync_interval > 0 and sync_target:
            syncer = RamdriveSyncer(
                source_dir=Path(run_dir),
                target_dir=Path(sync_target) / f"nnue_{timestamp}",
                interval=sync_interval,
                patterns=["*.json", "*.pt", "*.npz"],
            )
            syncer.start()
            logger.info(f"Started ramdrive sync: {run_dir} -> {sync_target} every {sync_interval}s")
    else:
        run_dir = args.run_dir or os.path.join(PROJECT_ROOT, "runs", f"nnue_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)

    # Use ramdrive for cache_path if ramdrive is enabled and no cache specified
    if getattr(args, 'ram_storage', False) and not args.cache_path:
        ramdrive_config = get_config_from_args(args)
        ramdrive_config.subdirectory = "training/cache"
        cache_dir = get_data_directory(prefer_ramdrive=True, config=ramdrive_config, base_name="nnue_cache")
        args.cache_path = str(cache_dir / f"nnue_{board_type.value}_{args.num_players}p.npz")
        logger.info(f"Using ramdrive for feature cache: {args.cache_path}")

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
    logger.info(f"Starting NNUE training with {args.lr_schedule} LR schedule...")
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
        balanced_sampling=args.balanced_sampling,
        early_end=args.early_end,
        mid_end=args.mid_end,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        streaming=args.streaming,
        streaming_buffer=args.streaming_buffer,
        num_workers=args.num_workers,
        adaptive_batch=args.adaptive_batch,
        min_batch_size=args.min_batch_size,
        max_batch_size=args.max_batch_size,
        mixed_precision=args.mixed_precision,
        amp_dtype=args.amp_dtype,
        per=args.per,
        per_alpha=args.per_alpha,
        per_beta=args.per_beta,
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

    # Stop ramdrive syncer and perform final sync
    if syncer:
        logger.info("Stopping ramdrive syncer and performing final sync...")
        syncer.stop(final_sync=True)
        logger.info(f"Ramdrive sync stats: {syncer.stats}")

    logger.info("NNUE training complete!")
    logger.info(f"  Model saved to: {save_path}")
    best_val = report.get('best_val_loss')
    final_acc = report.get('final_val_accuracy')
    logger.info(f"  Best validation loss: {best_val:.4f}" if isinstance(best_val, (int, float)) else f"  Best validation loss: {best_val}")
    logger.info(f"  Final validation accuracy: {final_acc:.4f}" if isinstance(final_acc, (int, float)) else f"  Final validation accuracy: {final_acc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
