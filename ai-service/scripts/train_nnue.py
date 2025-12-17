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
from app.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    get_device_for_rank,
    wrap_model_ddp,
    get_distributed_sampler,
    scale_learning_rate,
    reduce_tensor,
    synchronize,
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
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker (default: 2)",
    )
    parser.add_argument(
        "--async-pipeline",
        action="store_true",
        help="Enable async overlapped pipeline for data loading and GPU transfer",
    )
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Enable Quantization-Aware Training for better int8 inference accuracy",
    )
    parser.add_argument(
        "--qat-start-epoch",
        type=int,
        default=10,
        help="Epoch to start QAT (default: 10, allows model to converge first)",
    )
    parser.add_argument(
        "--progressive-val",
        action="store_true",
        help="Progressive validation: validate on subset early, full validation later",
    )
    parser.add_argument(
        "--progressive-val-start",
        type=float,
        default=0.2,
        help="Initial validation fraction for progressive validation (default: 0.2)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for memory efficiency (trades compute for memory)",
    )
    parser.add_argument(
        "--async-logging",
        action="store_true",
        help="Enable async metric logging (log metrics in background thread)",
    )
    parser.add_argument(
        "--lr-batch-scale",
        action="store_true",
        help="Scale learning rate with effective batch size (includes accumulation)",
    )
    parser.add_argument(
        "--spectral-norm",
        action="store_true",
        help="Apply spectral normalization to Linear layers for gradient stability",
    )
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        help="Add batch normalization after accumulator layer",
    )
    parser.add_argument(
        "--loss-curriculum",
        action="store_true",
        help="Use phase-based loss curriculum (early/mid/late weighting)",
    )
    parser.add_argument(
        "--loss-curriculum-schedule",
        type=str,
        default="50,35,15->25,35,40",
        help="Loss curriculum schedule: 'early,mid,late->early,mid,late' (default: '50,35,15->25,35,40')",
    )
    parser.add_argument(
        "--progressive-accum",
        action="store_true",
        help="Progressive accumulation unfreezing (higher accum during warmup)",
    )
    parser.add_argument(
        "--progressive-accum-start",
        type=int,
        default=4,
        help="Starting accumulation multiplier for progressive unfreezing (default: 4)",
    )
    parser.add_argument(
        "--cyclic-lr",
        action="store_true",
        help="Use cyclic LR with triangular waves within cosine envelope",
    )
    parser.add_argument(
        "--cyclic-lr-period",
        type=int,
        default=5,
        help="Period of triangular cycles in epochs (default: 5)",
    )
    parser.add_argument(
        "--val-augmentation",
        action="store_true",
        help="Apply data augmentation to validation set for better overfitting detection",
    )
    parser.add_argument(
        "--lars",
        action="store_true",
        help="Use LARS (Layer-wise Adaptive Rate Scaling) optimizer for distributed training",
    )
    parser.add_argument(
        "--lars-trust-coef",
        type=float,
        default=0.001,
        help="LARS trust coefficient (default: 0.001)",
    )
    parser.add_argument(
        "--gradient-profiling",
        action="store_true",
        help="Enable gradient norm tracking for diagnostics",
    )
    parser.add_argument(
        "--gradient-profile-freq",
        type=int,
        default=100,
        help="Log gradient norms every N batches (default: 100)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=1,
        help="Number of heads for multi-head feature projection (default: 1 = single head)",
    )
    parser.add_argument(
        "--knowledge-distill",
        action="store_true",
        help="Enable knowledge distillation from teacher model",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Path to teacher model for knowledge distillation",
    )
    parser.add_argument(
        "--distill-alpha",
        type=float,
        default=0.5,
        help="Knowledge distillation weight (0=pure label, 1=pure teacher) (default: 0.5)",
    )
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=2.0,
        help="Knowledge distillation temperature (default: 2.0)",
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
        "--per-update-freq",
        type=int,
        default=1,
        help="PER priority update frequency in epochs (default: 1 = every epoch)",
    )
    parser.add_argument(
        "--use-board-config",
        action="store_true",
        help="Use board-specific hyperparameters from config/training_hyperparams.yaml",
    )
    parser.add_argument(
        "--gpu-extraction",
        action="store_true",
        help="Use GPU-accelerated feature extraction for faster dataset generation",
    )
    parser.add_argument(
        "--gpu-extraction-batch",
        type=int,
        default=256,
        help="Batch size for GPU feature extraction (default: 256)",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training with DDP (launch with torchrun)",
    )
    parser.add_argument(
        "--lr-scale",
        type=str,
        choices=["linear", "sqrt", "none"],
        default="sqrt",
        help="LR scaling for distributed training: linear, sqrt (default), or none",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1, no accumulation)",
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


def parse_curriculum_schedule(schedule: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Parse loss curriculum schedule string.

    Args:
        schedule: Format 'early,mid,late->early,mid,late' e.g. '50,35,15->25,35,40'

    Returns:
        Tuple of (start_weights, end_weights) each as (early, mid, late) normalized to sum to 1.0
    """
    parts = schedule.split('->')
    start_str, end_str = parts[0], parts[1] if len(parts) > 1 else parts[0]

    def parse_weights(s: str) -> Tuple[float, float, float]:
        vals = [float(x.strip()) for x in s.split(',')]
        total = sum(vals)
        return (vals[0] / total, vals[1] / total, vals[2] / total)

    return parse_weights(start_str), parse_weights(end_str)


def compute_curriculum_weights(
    epoch: int,
    total_epochs: int,
    start_weights: Tuple[float, float, float],
    end_weights: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Compute curriculum weights for current epoch by linear interpolation."""
    progress = epoch / max(total_epochs - 1, 1)
    return tuple(
        s + (e - s) * progress
        for s, e in zip(start_weights, end_weights)
    )


class LARS(optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling optimizer.

    Scales learning rate per-layer based on the ratio of weight norm to gradient norm.
    Particularly effective for large batch distributed training.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        trust_coef: float = 0.001,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            trust_coef=trust_coef, eps=eps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                weight_norm = p.norm(2).item()
                grad_norm = grad.norm(2).item()

                # Compute local learning rate
                if weight_norm > 0 and grad_norm > 0:
                    local_lr = group['trust_coef'] * weight_norm / (
                        grad_norm + group['weight_decay'] * weight_norm + group['eps']
                    )
                else:
                    local_lr = 1.0

                # Apply weight decay
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])

                # Apply momentum
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)

                # Apply update with LARS scaling
                p.add_(buf, alpha=-group['lr'] * local_lr)

        return loss


def get_cyclic_lr_factor(epoch: int, warmup_epochs: int, period: int) -> float:
    """Get triangular cycle factor for cyclic LR within cosine envelope.

    Returns a factor in [0.5, 1.0] that creates triangular waves.
    """
    if epoch < warmup_epochs:
        return 1.0  # No cycling during warmup

    cycle_epoch = (epoch - warmup_epochs) % period
    # Triangular wave: goes from 1.0 -> 0.5 -> 1.0 over the period
    half_period = period / 2
    if cycle_epoch < half_period:
        return 1.0 - 0.5 * (cycle_epoch / half_period)
    else:
        return 0.5 + 0.5 * ((cycle_epoch - half_period) / half_period)


def prepare_model_for_qat(model: nn.Module) -> nn.Module:
    """Prepare model for Quantization-Aware Training.

    Inserts fake quantization observers into the model for simulating
    quantization during training.
    """
    model.cpu()
    model.train()

    # Configure quantization for CPU inference
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    # Prepare for QAT (inserts fake quantization modules)
    torch.quantization.prepare_qat(model, inplace=True)

    logger.info("Model prepared for Quantization-Aware Training")
    return model


def convert_qat_model(model: nn.Module) -> nn.Module:
    """Convert QAT model to actual quantized model for inference."""
    model.cpu()
    model.eval()

    # Convert to quantized model
    quantized_model = torch.quantization.convert(model, inplace=False)

    logger.info("QAT model converted to quantized model")
    return quantized_model


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
        gradient_accumulation: int = 1,
        use_gradient_checkpointing: bool = False,
        async_pipeline: bool = False,
        use_lars: bool = False,
        lars_trust_coef: float = 0.001,
        gradient_profiling: bool = False,
        gradient_profile_freq: int = 100,
        teacher_model: Optional[nn.Module] = None,
        distill_alpha: float = 0.5,
        distill_temperature: float = 2.0,
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

        # Gradient accumulation
        self.gradient_accumulation = gradient_accumulation
        if gradient_accumulation > 1:
            logger.info(f"Gradient accumulation enabled: {gradient_accumulation} steps")

        # Gradient checkpointing for memory efficiency
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if use_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Model does not support gradient checkpointing")

        # Async pipeline: use non-blocking data transfers
        self.async_pipeline = async_pipeline
        if async_pipeline:
            logger.info("Async pipeline: using non-blocking GPU transfers")

        # Gradient profiling
        self.gradient_profiling = gradient_profiling
        self.gradient_profile_freq = gradient_profile_freq
        self.gradient_norms_history = []

        # Knowledge distillation
        self.teacher_model = teacher_model
        self.distill_alpha = distill_alpha
        self.distill_temperature = distill_temperature
        if teacher_model is not None:
            logger.info(f"Knowledge distillation enabled: alpha={distill_alpha}, temp={distill_temperature}")

        # Choose optimizer: LARS for distributed large-batch, AdamW otherwise
        initial_lr = learning_rate if lr_schedule != "warmup_cosine" else 1e-7
        if use_lars:
            self.optimizer = LARS(
                model.parameters(),
                lr=initial_lr,
                momentum=0.9,
                weight_decay=weight_decay,
                trust_coef=lars_trust_coef,
            )
            logger.info(f"Using LARS optimizer with trust_coef={lars_trust_coef}")
        else:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=initial_lr,
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
        """Train for one epoch with gradient accumulation support. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accum_steps = self.gradient_accumulation

        # Use non-blocking transfers for async pipeline
        non_blocking = self.async_pipeline and self.device.type == "cuda"

        self.optimizer.zero_grad()

        for batch_idx, (features, values) in enumerate(dataloader):
            features = features.to(self.device, non_blocking=non_blocking)
            values = values.to(self.device, non_blocking=non_blocking)

            # Scale loss by accumulation steps for proper averaging
            if self.use_amp:
                # Mixed precision forward pass
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    predictions = self.model(features)
                    label_loss = self.criterion(predictions, values)

                    # Knowledge distillation: blend label loss with teacher matching
                    if self.teacher_model is not None:
                        with torch.no_grad():
                            teacher_outputs = self.teacher_model(features)
                        # Temperature-scaled distillation loss
                        temp = self.distill_temperature
                        distill_loss = self.criterion(
                            predictions / temp, teacher_outputs / temp
                        ) * (temp ** 2)
                        # Combined loss
                        loss = (1 - self.distill_alpha) * label_loss + self.distill_alpha * distill_loss
                    else:
                        loss = label_loss

                    if accum_steps > 1:
                        loss = loss / accum_steps

                if self.scaler is not None:
                    # FP16: use scaler for gradient scaling
                    self.scaler.scale(loss).backward()
                else:
                    # BF16: no scaler needed
                    loss.backward()
            else:
                # Standard precision
                predictions = self.model(features)
                label_loss = self.criterion(predictions, values)

                # Knowledge distillation: blend label loss with teacher matching
                if self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(features)
                    # Temperature-scaled distillation loss
                    temp = self.distill_temperature
                    distill_loss = self.criterion(
                        predictions / temp, teacher_outputs / temp
                    ) * (temp ** 2)
                    # Combined loss
                    loss = (1 - self.distill_alpha) * label_loss + self.distill_alpha * distill_loss
                else:
                    loss = label_loss

                if accum_steps > 1:
                    loss = loss / accum_steps
                loss.backward()

            # Gradient profiling before optimizer step
            if self.gradient_profiling and (batch_idx + 1) % self.gradient_profile_freq == 0:
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.gradient_norms_history.append(total_norm)
                if len(self.gradient_norms_history) % 10 == 0:
                    avg_norm = sum(self.gradient_norms_history[-10:]) / 10
                    logger.debug(f"Gradient norm (last 10 avg): {avg_norm:.4f}")

            # Step optimizer every accum_steps batches or at end
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Track unscaled loss for reporting
            total_loss += loss.item() * (accum_steps if accum_steps > 1 else 1)
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self, dataloader: DataLoader, sample_fraction: float = 1.0) -> Tuple[float, float]:
        """Validate on held-out data. Returns (loss, accuracy).

        Args:
            dataloader: Validation data loader
            sample_fraction: Fraction of batches to evaluate (for progressive validation)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        # Calculate number of batches to process
        total_batches = len(dataloader)
        batches_to_process = max(1, int(total_batches * sample_fraction))

        with torch.no_grad():
            for batch_idx, (features, values) in enumerate(dataloader):
                # Skip batches beyond sample_fraction
                if batch_idx >= batches_to_process:
                    break

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
    per_update_freq: int = 1,
    distributed: bool = False,
    lr_scale: str = "sqrt",
    gradient_accumulation: int = 1,
    gpu_extraction: bool = False,
    gpu_extraction_batch: int = 256,
    prefetch_factor: int = 2,
    async_pipeline: bool = False,
    qat: bool = False,
    qat_start_epoch: int = 10,
    progressive_val: bool = False,
    progressive_val_start: float = 0.2,
    gradient_checkpointing: bool = False,
    async_logging: bool = False,
    lr_batch_scale: bool = False,
    spectral_norm: bool = False,
    batch_norm: bool = False,
    loss_curriculum: bool = False,
    loss_curriculum_schedule: str = "50,35,15->25,35,40",
    progressive_accum: bool = False,
    progressive_accum_start: int = 4,
    cyclic_lr: bool = False,
    cyclic_lr_period: int = 5,
    val_augmentation: bool = False,
    lars: bool = False,
    lars_trust_coef: float = 0.001,
    gradient_profiling: bool = False,
    gradient_profile_freq: int = 100,
    num_heads: int = 1,
    knowledge_distill: bool = False,
    teacher_model: Optional[str] = None,
    distill_alpha: float = 0.5,
    distill_temperature: float = 2.0,
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
            use_gpu_extraction=gpu_extraction,
            gpu_batch_size=gpu_extraction_batch,
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
            use_gpu_extraction=gpu_extraction,
            gpu_batch_size=gpu_extraction_batch,
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

    # Configure DataLoader for async pipeline
    use_pin_memory = device.type != "cpu"
    # prefetch_factor is only valid with num_workers > 0
    effective_prefetch = prefetch_factor if num_workers > 0 else None

    # For async pipeline, enable non-blocking transfers and persistent workers
    persistent_workers = async_pipeline and num_workers > 0

    # Create train loader
    if streaming:
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": use_pin_memory,
        }
        if effective_prefetch is not None:
            loader_kwargs["prefetch_factor"] = effective_prefetch
        if persistent_workers:
            loader_kwargs["persistent_workers"] = True

        train_loader = DataLoader(streaming_dataset, **loader_kwargs)
    else:
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": (train_sampler is None),
            "sampler": train_sampler,
            "num_workers": num_workers,
            "pin_memory": use_pin_memory,
        }
        if effective_prefetch is not None:
            loader_kwargs["prefetch_factor"] = effective_prefetch
        if persistent_workers:
            loader_kwargs["persistent_workers"] = True

        train_loader = DataLoader(train_dataset, **loader_kwargs)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    if async_pipeline:
        logger.info(f"Async pipeline enabled: prefetch={effective_prefetch}, "
                   f"pin_memory={use_pin_memory}, persistent_workers={persistent_workers}")

    # Create model
    model = RingRiftNNUE(
        board_type=board_type,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        use_spectral_norm=spectral_norm,
        use_batch_norm=batch_norm,
        num_heads=num_heads,
    )
    if spectral_norm:
        logger.info("Spectral normalization enabled for gradient stability")
    if batch_norm:
        logger.info("Batch normalization enabled after accumulator")
    if num_heads > 1:
        logger.info(f"Multi-head feature projection enabled: {num_heads} heads")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load teacher model for knowledge distillation
    teacher_model_loaded = None
    if knowledge_distill:
        if teacher_model is None:
            logger.warning("Knowledge distillation enabled but no teacher model specified")
        elif not os.path.exists(teacher_model):
            logger.warning(f"Teacher model not found: {teacher_model}")
        else:
            try:
                teacher_checkpoint = torch.load(teacher_model, map_location=device)
                teacher_hidden_dim = teacher_checkpoint.get("hidden_dim", hidden_dim)
                teacher_num_layers = teacher_checkpoint.get("num_hidden_layers", num_hidden_layers)
                teacher_model_loaded = RingRiftNNUE(
                    board_type=board_type,
                    hidden_dim=teacher_hidden_dim,
                    num_hidden_layers=teacher_num_layers,
                )
                teacher_model_loaded.load_state_dict(teacher_checkpoint["model_state_dict"])
                teacher_model_loaded = teacher_model_loaded.to(device)
                teacher_model_loaded.eval()
                for param in teacher_model_loaded.parameters():
                    param.requires_grad = False
                logger.info(f"Loaded teacher model from {teacher_model} "
                           f"(hidden={teacher_hidden_dim}, layers={teacher_num_layers})")
            except Exception as e:
                logger.warning(f"Failed to load teacher model: {e}")

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

    # Distributed training setup
    if distributed:
        if not torch.cuda.is_available():
            logger.error("Distributed training requires CUDA")
            return {"error": "Distributed training requires CUDA"}

        # Initialize distributed process group
        setup_distributed()
        rank = get_rank()
        world_size = get_world_size()
        device = get_device_for_rank()

        logger.info(f"Distributed training: rank {rank}/{world_size}, device {device}")

        # Scale learning rate for distributed training
        original_lr = learning_rate
        learning_rate = scale_learning_rate(learning_rate, world_size, lr_scale)
        if is_main_process():
            logger.info(f"LR scaled for {world_size} GPUs: {original_lr:.2e} -> {learning_rate:.2e} ({lr_scale})")

        # Move model to device and wrap with DDP
        model = model.to(device)
        model = wrap_model_ddp(model, device_ids=[device.index] if device.index is not None else None)

        # Replace sampler with DistributedSampler if not streaming
        if not streaming and train_sampler is None:
            train_sampler = get_distributed_sampler(train_dataset, shuffle=True)
            # Recreate train loader with distributed sampler
            train_loader = DataLoader(
                train_dataset,
                batch_size=actual_batch_size,
                shuffle=False,  # Sampler handles shuffling
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
            if is_main_process():
                logger.info(f"Using DistributedSampler for {len(train_dataset)} samples across {world_size} GPUs")

    # Scale learning rate with effective batch size if requested
    effective_batch = actual_batch_size * gradient_accumulation
    if lr_batch_scale and effective_batch != 512:  # 512 is reference batch size
        base_lr = learning_rate
        # Use sqrt scaling (more stable than linear)
        learning_rate = learning_rate * (effective_batch / 512) ** 0.5
        logger.info(f"LR scaled for batch size {effective_batch}: {base_lr:.2e} -> {learning_rate:.2e}")

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
        gradient_accumulation=gradient_accumulation,
        use_gradient_checkpointing=gradient_checkpointing,
        async_pipeline=async_pipeline,
        use_lars=lars,
        lars_trust_coef=lars_trust_coef,
        gradient_profiling=gradient_profiling,
        gradient_profile_freq=gradient_profile_freq,
        teacher_model=teacher_model_loaded,
        distill_alpha=distill_alpha,
        distill_temperature=distill_temperature,
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    qat_enabled = False
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    if qat:
        logger.info(f"QAT will start at epoch {qat_start_epoch}")

    # Loss curriculum setup
    curriculum_start_weights = None
    curriculum_end_weights = None
    if loss_curriculum:
        curriculum_start_weights, curriculum_end_weights = parse_curriculum_schedule(loss_curriculum_schedule)
        logger.info(f"Loss curriculum enabled: {curriculum_start_weights} -> {curriculum_end_weights}")

    # Progressive accumulation setup
    base_accumulation = gradient_accumulation
    if progressive_accum:
        # Start with higher accumulation, decrease over warmup
        gradient_accumulation = base_accumulation * progressive_accum_start
        trainer.gradient_accumulation = gradient_accumulation
        logger.info(f"Progressive accumulation: {gradient_accumulation} -> {base_accumulation} over {warmup_epochs} epochs")

    # Cyclic LR setup - store initial LR for cycling
    if cyclic_lr:
        for param_group in trainer.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
        logger.info(f"Cyclic LR enabled: period={cyclic_lr_period} epochs")

    # Progressive validation setup
    if progressive_val:
        logger.info(f"Progressive validation enabled: {progressive_val_start:.0%} -> 100%")

    # Async logging setup
    async_log_queue = None
    async_log_thread = None
    if async_logging:
        from queue import Queue
        from threading import Thread

        async_log_queue = Queue()

        def async_logger():
            while True:
                item = async_log_queue.get()
                if item is None:
                    break
                try:
                    report_training_metrics(**item)
                except Exception as e:
                    logger.debug(f"Async logging error: {e}")
                async_log_queue.task_done()

        async_log_thread = Thread(target=async_logger, daemon=True)
        async_log_thread.start()
        logger.info("Async metric logging enabled")

    for epoch in range(epochs):
        # Enable QAT at the specified epoch
        if qat and not qat_enabled and epoch >= qat_start_epoch:
            logger.info(f"Enabling Quantization-Aware Training at epoch {epoch + 1}")
            # Unwrap DDP if needed
            if distributed and hasattr(model, 'module'):
                model.module = prepare_model_for_qat(model.module)
                model.module = model.module.to(device)
            else:
                model = prepare_model_for_qat(model)
                model = model.to(device)
                trainer.model = model
            qat_enabled = True
            logger.info("QAT enabled - continuing training with fake quantization")

        # Update streaming dataset epoch for proper shuffling
        if streaming and streaming_dataset is not None:
            streaming_dataset.set_epoch(epoch)

        # Update distributed sampler epoch for proper shuffling
        if distributed and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # Update PER sampler epoch for beta annealing
        if per_sampler is not None:
            per_sampler.set_epoch(epoch, epochs)

        # Progressive accumulation: decrease accumulation over warmup
        if progressive_accum and epoch < warmup_epochs:
            progress = epoch / max(warmup_epochs - 1, 1)
            # Linear decrease from start multiplier to 1x
            current_mult = progressive_accum_start - (progressive_accum_start - 1) * progress
            trainer.gradient_accumulation = max(base_accumulation, int(base_accumulation * current_mult))
        elif progressive_accum and epoch == warmup_epochs:
            trainer.gradient_accumulation = base_accumulation
            logger.info(f"Progressive accumulation complete: now using {base_accumulation}")

        # Cyclic LR: apply triangular modulation
        if cyclic_lr and epoch >= warmup_epochs:
            cycle_factor = get_cyclic_lr_factor(epoch, warmup_epochs, cyclic_lr_period)
            for param_group in trainer.optimizer.param_groups:
                base_lr = param_group.get('initial_lr', param_group['lr'])
                param_group['lr'] = base_lr * cycle_factor

        train_loss = trainer.train_epoch(train_loader)

        # Reduce train loss across all processes for distributed training
        if distributed:
            train_loss_tensor = torch.tensor(train_loss, device=device)
            train_loss = reduce_tensor(train_loss_tensor).item()

        # Progressive validation: validate on subset early, full validation later
        if progressive_val:
            # Linearly increase validation fraction from start to 1.0
            val_fraction = min(1.0, progressive_val_start + (1.0 - progressive_val_start) * (epoch / max(epochs - 1, 1)))
            val_loss, val_accuracy = trainer.validate(val_loader, sample_fraction=val_fraction)
        else:
            val_loss, val_accuracy = trainer.validate(val_loader)
        trainer.update_scheduler(val_loss)

        # Update PER priorities based on prediction errors
        if per_sampler is not None and (epoch + 1) % per_update_freq == 0:
            logger.info(f"Updating PER priorities (every {per_update_freq} epochs)...")
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
        metric_data = {
            "board_type": board_type,
            "num_players": num_players,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch + 1,
            "model_path": str(save_path) if save_path else "",
        }
        if async_log_queue is not None:
            async_log_queue.put(metric_data)
        else:
            report_training_metrics(**metric_data)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Save best model (only on main process for distributed training)
            if not distributed or is_main_process():
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Get model state dict (unwrap DDP if needed)
                if distributed and hasattr(model, 'module'):
                    model_state = model.module.state_dict()
                    arch_version = model.module.ARCHITECTURE_VERSION
                else:
                    model_state = model.state_dict()
                    arch_version = model.ARCHITECTURE_VERSION

                checkpoint = {
                    "model_state_dict": model_state,
                    "board_type": board_type.value,
                    "hidden_dim": hidden_dim,
                    "num_hidden_layers": num_hidden_layers,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "architecture_version": arch_version,
                }
                torch.save(checkpoint, save_path)
                logger.info(f"Saved best model to {save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Cleanup async logging thread
    if async_log_queue is not None:
        async_log_queue.put(None)  # Signal thread to exit
        if async_log_thread is not None:
            async_log_thread.join(timeout=5.0)

    # Synchronize all processes before cleanup
    if distributed:
        synchronize()

    # Get model param count (unwrap DDP if needed)
    if distributed and hasattr(model, 'module'):
        model_params = sum(p.numel() for p in model.module.parameters())
    else:
        model_params = sum(p.numel() for p in model.parameters())

    # Save QAT-converted quantized model if QAT was used
    qat_model_path = None
    if qat_enabled and (not distributed or is_main_process()):
        try:
            # Get the raw model (unwrap DDP if needed)
            raw_model = model.module if distributed and hasattr(model, 'module') else model
            quantized_model = convert_qat_model(raw_model)

            # Save quantized model
            qat_model_path = save_path.replace('.pt', '_qat_int8.pt').replace('.pth', '_qat_int8.pth')
            if qat_model_path == save_path:
                qat_model_path = save_path + '_qat_int8.pt'

            checkpoint = {
                "model_state_dict": quantized_model.state_dict(),
                "board_type": board_type.value,
                "hidden_dim": hidden_dim,
                "num_hidden_layers": num_hidden_layers,
                "epoch": epoch + 1,
                "val_loss": best_val_loss,
                "architecture_version": raw_model.ARCHITECTURE_VERSION if hasattr(raw_model, 'ARCHITECTURE_VERSION') else "v1.0.0",
                "quantized": True,
                "qat_trained": True,
            }
            torch.save(checkpoint, qat_model_path)
            logger.info(f"Saved QAT-trained quantized model to {qat_model_path}")
        except Exception as e:
            logger.warning(f"Failed to save QAT model: {e}")

    # Training report
    report = {
        "board_type": board_type.value,
        "num_players": num_players,
        "dataset_size": dataset_size if dataset_size > 0 else "streaming",
        "train_size": "streaming" if streaming else len(train_dataset),
        "val_size": len(val_dataset),
        "model_params": model_params,
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
        "distributed": distributed,
        "world_size": get_world_size() if distributed else 1,
        "lr_scale": lr_scale if distributed else None,
        "gradient_accumulation": gradient_accumulation,
        "qat": qat,
        "qat_enabled": qat_enabled,
        "qat_start_epoch": qat_start_epoch if qat else None,
        "qat_model_path": qat_model_path,
        "history": history,
    }

    # Cleanup distributed training
    if distributed:
        cleanup_distributed()

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
    if args.distributed:
        logger.info(f"Distributed training enabled (launch with: torchrun --nproc_per_node=N train_nnue.py ...)")
    if args.gradient_accumulation > 1:
        logger.info(f"Gradient accumulation: {args.gradient_accumulation} steps")

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
        per_update_freq=args.per_update_freq,
        distributed=args.distributed,
        lr_scale=args.lr_scale,
        gradient_accumulation=args.gradient_accumulation,
        gpu_extraction=args.gpu_extraction,
        gpu_extraction_batch=args.gpu_extraction_batch,
        prefetch_factor=args.prefetch_factor,
        async_pipeline=args.async_pipeline,
        qat=args.qat,
        qat_start_epoch=args.qat_start_epoch,
        progressive_val=args.progressive_val,
        progressive_val_start=args.progressive_val_start,
        gradient_checkpointing=args.gradient_checkpointing,
        async_logging=args.async_logging,
        lr_batch_scale=args.lr_batch_scale,
        spectral_norm=args.spectral_norm,
        batch_norm=args.batch_norm,
        loss_curriculum=args.loss_curriculum,
        loss_curriculum_schedule=args.loss_curriculum_schedule,
        progressive_accum=args.progressive_accum,
        progressive_accum_start=args.progressive_accum_start,
        cyclic_lr=args.cyclic_lr,
        cyclic_lr_period=args.cyclic_lr_period,
        val_augmentation=args.val_augmentation,
        lars=args.lars,
        lars_trust_coef=args.lars_trust_coef,
        gradient_profiling=args.gradient_profiling,
        gradient_profile_freq=args.gradient_profile_freq,
        num_heads=args.num_heads,
        knowledge_distill=args.knowledge_distill,
        teacher_model=args.teacher_model,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
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
