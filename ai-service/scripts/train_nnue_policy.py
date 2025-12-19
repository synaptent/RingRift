#!/usr/bin/env python
"""Train NNUE with Policy Head for RingRift.

This script trains the NNUE network with both value and policy heads.
The policy head learns to predict the move that was played from each
position, providing move guidance for search algorithms.

Training data is extracted from self-play game databases (SQLite), where
each position is labeled with:
- Game outcome (for value head)
- The move that was played (for policy head)

Usage:
    # Train on a single database
    python scripts/train_nnue_policy.py --db data/games/selfplay.db --epochs 50

    # Train with custom loss weights
    python scripts/train_nnue_policy.py --db data/games/*.db \\
        --value-weight 1.0 --policy-weight 0.5 --epochs 100

    # Fine-tune from existing value-only model
    python scripts/train_nnue_policy.py --db data/games/selfplay.db \\
        --pretrained models/nnue/nnue_square8_2p.pt --freeze-value

Output:
    - Model checkpoint: models/nnue/nnue_policy_{board_type}.pt
    - Training report: {run_dir}/nnue_policy_training_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Set up path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from app.ai.nnue import RingRiftNNUE, clear_nnue_cache, get_board_size
from app.ai.nnue_policy import (
    RingRiftNNUEWithPolicy,
    NNUEPolicyTrainer,
    NNUEPolicyDataset,
    NNUEPolicyDatasetConfig,
    get_hidden_dim_for_board,
    HexBoardAugmenter,
    LearningRateFinder,
)
from app.models import BoardType
from app.training.seed_utils import seed_all

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("train_nnue_policy")


def parse_board_type(value: str) -> BoardType:
    """Parse board type string to enum."""
    mapping = {
        "square8": BoardType.SQUARE8,
        "sq8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "sq19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
        "hex": BoardType.HEXAGONAL,
        "hex8": BoardType.HEX8,
    }
    key = value.lower()
    if key not in mapping:
        raise ValueError(f"Unknown board type: {value}")
    return mapping[key]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NNUE with policy head for RingRift",
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
        "--jsonl",
        type=str,
        nargs="+",
        default=[],
        help="Path(s) to JSONL game file(s) with MCTS policy data. Supports glob patterns.",
    )

    # Board configuration
    parser.add_argument(
        "--board-type",
        type=str,
        default="square8",
        help="Board type: square8, square19, hexagonal, or hex8 (default: square8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (default: 256)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
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

    # Loss weights
    parser.add_argument(
        "--value-weight",
        type=float,
        default=1.0,
        help="Weight for value loss (default: 1.0)",
    )
    parser.add_argument(
        "--policy-weight",
        type=float,
        default=1.0,
        help="Weight for policy loss (default: 1.0)",
    )

    # Model architecture
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="NNUE hidden layer dimension. If not set, auto-selects based on board type: square8=128, hex8=256, full_hex=1024, square19=512",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=2,
        help="Number of NNUE hidden layers (default: 2)",
    )
    parser.add_argument(
        "--policy-dropout",
        type=float,
        default=0.1,
        help="Dropout rate for policy head (default: 0.1, 0=disabled)",
    )

    # Pre-training / fine-tuning
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained value-only NNUE model for fine-tuning",
    )
    parser.add_argument(
        "--freeze-value",
        action="store_true",
        help="Freeze value weights when fine-tuning from pretrained model",
    )

    # Temperature annealing
    parser.add_argument(
        "--temperature-start",
        type=float,
        default=2.0,
        help="Starting temperature for policy annealing (default: 2.0)",
    )
    parser.add_argument(
        "--temperature-end",
        type=float,
        default=0.5,
        help="Ending temperature for policy annealing (default: 0.5)",
    )
    parser.add_argument(
        "--temperature-schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "exponential"],
        help="Temperature annealing schedule (default: cosine)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for policy loss (default: 0.1)",
    )

    # Sampling configuration
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=2,
        help="Sample every Nth position from games (default: 2)",
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
        "--max-moves-per-position",
        type=int,
        default=None,
        help="Maximum legal moves to encode per position (default: auto based on board type)",
    )

    # Output
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output directory for reports (default: runs/nnue_policy_{timestamp})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Custom path for model checkpoint",
    )

    # Policy distillation from strong games
    parser.add_argument(
        "--distill-from-winners",
        action="store_true",
        help="Only train on positions from winning players (stronger signal)",
    )
    parser.add_argument(
        "--winner-weight-boost",
        type=float,
        default=1.0,
        help="Weight multiplier for winners' moves (1.0=no boost, 2.0=double weight). "
             "Only effective with --distill-from-winners. (default: 1.0)",
    )
    parser.add_argument(
        "--min-winner-margin",
        type=int,
        default=0,
        help="Minimum victory margin (score difference) for distillation. "
             "Filters to decisive wins only. (default: 0 = all wins)",
    )

    # Curriculum learning (move range filter)
    parser.add_argument(
        "--min-move-number",
        type=int,
        default=0,
        help="Only include positions with move_number >= this value. "
             "Used for curriculum learning stages. (default: 0 = all moves)",
    )
    parser.add_argument(
        "--max-move-number",
        type=int,
        default=999999,
        help="Only include positions with move_number <= this value. "
             "Used for curriculum learning stages. (default: 999999 = all moves)",
    )

    # Performance
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for parallel sample extraction. "
             "0 = auto-detect (cpu_count - 2), 1 = sequential (default: 0)",
    )

    # Training stability
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (0 to disable, default: 1.0)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "cosine_warmup"],
        help="Learning rate scheduler: plateau (reduce on plateau), cosine (annealing), "
             "cosine_warmup (warmup + annealing). Default: plateau",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (default: auto-detect)",
    )
    parser.add_argument(
        "--use-kl-loss",
        action="store_true",
        help="Use KL divergence loss with MCTS visit distributions instead of cross-entropy. "
             "Requires training data with mcts_policy field (from MCTS selfplay).",
    )
    parser.add_argument(
        "--auto-kl-loss",
        action="store_true",
        help="Automatically enable KL loss if sufficient MCTS policy data is detected "
             "(>= 50%% coverage and >= 100 samples with MCTS policy). Falls back to "
             "cross-entropy if insufficient data.",
    )
    parser.add_argument(
        "--kl-min-coverage",
        type=float,
        default=0.5,
        help="Minimum MCTS policy coverage (0.0-1.0) for auto KL loss (default: 0.5)",
    )
    parser.add_argument(
        "--kl-min-samples",
        type=int,
        default=100,
        help="Minimum samples with MCTS policy for auto KL loss (default: 100)",
    )
    parser.add_argument(
        "--kl-loss-weight",
        type=float,
        default=1.0,
        help="Mix ratio for KL vs cross-entropy loss when using KL loss. "
             "1.0 = pure KL divergence, 0.5 = 50%% KL + 50%% CE, 0.0 = pure CE. "
             "Mixing can improve stability. (default: 1.0)",
    )

    # Advanced training options
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use mixed precision (FP16) training for faster training on CUDA (default: True)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use Exponential Moving Average for model weights (smoother final model)",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay rate (default: 0.999)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal loss gamma for hard sample mining (0 = disabled, 2.0 = typical, default: 0)",
    )
    parser.add_argument(
        "--label-smoothing-warmup",
        type=int,
        default=0,
        help="Number of epochs to warm up label smoothing from 0 to target (default: 0 = no warmup)",
    )
    parser.add_argument(
        "--save-curves",
        action="store_true",
        help="Save learning curve plots (requires matplotlib)",
    )

    # Distributed training
    parser.add_argument(
        "--use-ddp",
        action="store_true",
        help="Use DistributedDataParallel for multi-GPU training",
    )
    parser.add_argument(
        "--ddp-rank",
        type=int,
        default=0,
        help="Rank for DDP (default: 0)",
    )

    # Stochastic Weight Averaging
    parser.add_argument(
        "--use-swa",
        action="store_true",
        help="Use Stochastic Weight Averaging for better generalization",
    )
    parser.add_argument(
        "--swa-start-epoch",
        type=int,
        default=0,
        help="Epoch to start SWA (0 = 75%% of total epochs)",
    )
    parser.add_argument(
        "--swa-lr",
        type=float,
        default=None,
        help="SWA learning rate (default: 10%% of base LR)",
    )

    # Progressive batch sizing
    parser.add_argument(
        "--progressive-batch",
        action="store_true",
        help="Use progressive batch sizing (start small, grow larger)",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=64,
        help="Minimum batch size for progressive batching (default: 64)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=512,
        help="Maximum batch size for progressive batching (default: 512)",
    )

    # Hex board data augmentation
    parser.add_argument(
        "--hex-augment",
        action="store_true",
        help="Enable D6 symmetry augmentation for hexagonal boards (multiplies training data)",
    )
    parser.add_argument(
        "--hex-augment-count",
        type=int,
        default=6,
        help="Number of augmented copies per sample (1-12 for D6 symmetry, default: 6)",
    )

    # Gradient accumulation
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients (effective batch = batch_size * steps, default: 1)",
    )

    # Learning rate finder
    parser.add_argument(
        "--find-lr",
        action="store_true",
        help="Run learning rate finder before training to suggest optimal LR",
    )
    parser.add_argument(
        "--lr-finder-iterations",
        type=int,
        default=100,
        help="Number of iterations for LR finder sweep (default: 100)",
    )

    return parser.parse_args(argv)


def split_by_game_id(
    dataset: "NNUEPolicyDataset",
    val_split: float,
    seed: int,
) -> tuple:
    """Split dataset into train/val sets at the game level.

    Groups samples by game_id and splits games (not samples) into train/val,
    preventing data leakage from having samples from the same game in both sets.

    Returns:
        Tuple of (train_indices, val_indices)
    """
    import random
    from collections import defaultdict

    # Group sample indices by game_id
    game_to_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        game_to_indices[sample.game_id].append(idx)

    # Get list of game_ids
    game_ids = list(game_to_indices.keys())

    # Shuffle games with seed
    rng = random.Random(seed)
    rng.shuffle(game_ids)

    # Split games
    num_val_games = int(len(game_ids) * val_split)
    val_game_ids = set(game_ids[:num_val_games])
    train_game_ids = set(game_ids[num_val_games:])

    # Collect indices
    train_indices = []
    val_indices = []
    for game_id, indices in game_to_indices.items():
        if game_id in val_game_ids:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return train_indices, val_indices


def collate_policy_batch(batch):
    """Custom collate function for policy dataset batches."""
    features = torch.stack([b[0] for b in batch])
    values = torch.stack([b[1] for b in batch])
    from_indices = torch.stack([b[2] for b in batch])
    to_indices = torch.stack([b[3] for b in batch])
    move_mask = torch.stack([b[4] for b in batch])
    target_idx = torch.stack([b[5] for b in batch])
    sample_weights = torch.stack([b[6] for b in batch])
    mcts_probs = torch.stack([b[7] for b in batch])
    return features, values, from_indices, to_indices, move_mask, target_idx, sample_weights, mcts_probs


def augment_batch_with_hex(
    features: torch.Tensor,
    values: torch.Tensor,
    from_idx: torch.Tensor,
    to_idx: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
    mcts_probs: Optional[torch.Tensor],
    augmenter: "HexBoardAugmenter",
) -> tuple:
    """Apply D6 symmetry augmentation to a batch of hex board samples.

    Expands the batch by applying random transformations to each sample.
    """
    aug_features = []
    aug_values = []
    aug_from_idx = []
    aug_to_idx = []
    aug_mask = []
    aug_target = []
    aug_weights = []
    aug_mcts = [] if mcts_probs is not None else None

    batch_size = features.shape[0]
    for i in range(batch_size):
        # Get numpy arrays for augmentation
        feat_np = features[i].numpy()
        from_np = from_idx[i].numpy()
        to_np = to_idx[i].numpy()
        target_idx_val = target[i].item()

        # Get augmented samples (includes original)
        augmented = augmenter.augment_sample(
            features=feat_np,
            from_indices=from_np,
            to_indices=to_np,
            target_move_idx=target_idx_val,
            include_original=True,
        )

        for aug_feat, aug_from, aug_to, aug_tgt in augmented:
            aug_features.append(torch.from_numpy(aug_feat).float())
            aug_values.append(values[i])
            aug_from_idx.append(torch.from_numpy(aug_from).long())
            aug_to_idx.append(torch.from_numpy(aug_to).long())
            aug_mask.append(mask[i])  # Mask stays the same
            aug_target.append(torch.tensor(aug_tgt, dtype=torch.long))
            aug_weights.append(sample_weights[i])
            if aug_mcts is not None:
                aug_mcts.append(mcts_probs[i])  # MCTS probs stay same (would need transform too)

    return (
        torch.stack(aug_features),
        torch.stack(aug_values),
        torch.stack(aug_from_idx),
        torch.stack(aug_to_idx),
        torch.stack(aug_mask),
        torch.stack(aug_target),
        torch.stack(aug_weights),
        torch.stack(aug_mcts) if aug_mcts is not None else None,
    )


def train_nnue_policy(
    db_paths: List[str],
    board_type: BoardType,
    num_players: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    val_split: float,
    early_stopping_patience: int,
    hidden_dim: Optional[int],
    num_hidden_layers: int,
    value_weight: float,
    policy_weight: float,
    sample_every_n: int,
    min_game_length: int,
    max_samples: Optional[int],
    max_moves_per_position: int,
    save_path: str,
    device: torch.device,
    seed: int,
    pretrained_path: Optional[str] = None,
    freeze_value: bool = False,
    temperature_start: float = 2.0,
    temperature_end: float = 0.5,
    temperature_schedule: str = "cosine",
    label_smoothing: float = 0.1,
    distill_from_winners: bool = False,
    winner_weight_boost: float = 1.0,
    num_workers: int = 0,
    min_move_number: int = 0,
    max_move_number: int = 999999,
    use_kl_loss: bool = False,
    auto_kl_loss: bool = False,
    kl_min_coverage: float = 0.5,
    kl_min_samples: int = 100,
    kl_loss_weight: float = 1.0,
    grad_clip: float = 1.0,
    lr_scheduler: str = "plateau",
    use_amp: bool = True,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    focal_gamma: float = 0.0,
    label_smoothing_warmup: int = 0,
    save_curves: bool = False,
    use_ddp: bool = False,
    ddp_rank: int = 0,
    use_swa: bool = False,
    swa_start_epoch: int = 0,
    swa_lr: Optional[float] = None,
    progressive_batch: bool = False,
    min_batch_size: int = 64,
    max_batch_size: int = 512,
    jsonl_paths: Optional[List[str]] = None,
    hex_augment: bool = False,
    hex_augment_count: int = 6,
    policy_dropout: float = 0.1,
    gradient_accumulation_steps: int = 1,
    find_lr: bool = False,
    lr_finder_iterations: int = 100,
) -> Dict[str, Any]:
    """Train NNUE policy model and return training report."""
    seed_all(seed)

    # Create dataset with distillation options
    config = NNUEPolicyDatasetConfig(
        board_type=board_type,
        num_players=num_players,
        sample_every_n_moves=sample_every_n,
        min_game_length=min_game_length,
        max_moves_per_position=max_moves_per_position,
        distill_from_winners=distill_from_winners,
        winner_weight_boost=winner_weight_boost,
        min_move_number=min_move_number,
        max_move_number=max_move_number,
    )

    if distill_from_winners:
        logger.info(f"Distillation mode: training on winners only (weight boost: {winner_weight_boost}x)")
    if min_move_number > 0 or max_move_number < 999999:
        logger.info(f"Curriculum mode: filtering to moves {min_move_number}-{max_move_number}")
    if num_workers != 1:
        worker_count = num_workers if num_workers > 0 else "auto"
        logger.info(f"Using parallel sample extraction with {worker_count} workers")
    dataset = NNUEPolicyDataset(
        db_paths=db_paths,
        config=config,
        max_samples=max_samples,
        num_workers=num_workers,
        jsonl_paths=jsonl_paths or [],
    )

    if len(dataset) == 0:
        logger.error("No training samples found!")
        return {"error": "No training samples"}

    logger.info(f"Dataset size: {len(dataset)} samples")

    # Auto-detect KL loss if requested
    effective_use_kl_loss = use_kl_loss
    mcts_stats = dataset.get_mcts_policy_stats()
    logger.info(f"MCTS policy stats: {mcts_stats['samples_with_mcts']}/{mcts_stats['total_samples']} "
                f"samples ({mcts_stats['mcts_coverage']:.1%} coverage)")

    if auto_kl_loss and not use_kl_loss:
        if dataset.should_use_kl_loss(min_coverage=kl_min_coverage, min_samples=kl_min_samples):
            effective_use_kl_loss = True
            logger.info(f"Auto-enabled KL loss (coverage {mcts_stats['mcts_coverage']:.1%} >= {kl_min_coverage:.0%}, "
                       f"{mcts_stats['samples_with_mcts']} samples >= {kl_min_samples})")
        else:
            logger.info(f"KL loss not auto-enabled (coverage {mcts_stats['mcts_coverage']:.1%} < {kl_min_coverage:.0%} "
                       f"or {mcts_stats['samples_with_mcts']} samples < {kl_min_samples})")

    # Split into train/val at game level (prevents data leakage)
    train_indices, val_indices = split_by_game_id(dataset, val_split, seed)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Count unique games in each split
    train_game_ids = set(dataset.samples[i].game_id for i in train_indices)
    val_game_ids = set(dataset.samples[i].game_id for i in val_indices)
    logger.info(f"Train size: {len(train_dataset)} samples from {len(train_game_ids)} games")
    logger.info(f"Val size: {len(val_dataset)} samples from {len(val_game_ids)} games")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_policy_batch,
        pin_memory=device.type != "cpu",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_policy_batch,
        pin_memory=device.type != "cpu",
    )

    # Initialize hex augmenter if enabled (only for hex boards)
    hex_augmenter = None
    if hex_augment and board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
        board_size = get_board_size(board_type)
        hex_augmenter = HexBoardAugmenter(board_size=board_size, num_augmentations=hex_augment_count)
        logger.info(f"Enabled D6 symmetry augmentation for hex board (size={board_size}, augmentations={hex_augment_count})")
    elif hex_augment:
        logger.warning(f"Hex augmentation requested but board_type={board_type.value} is not hexagonal, skipping")

    # Resolve hidden_dim (auto-select if not specified)
    actual_hidden_dim = hidden_dim
    if actual_hidden_dim is None:
        board_size = get_board_size(board_type)
        actual_hidden_dim = get_hidden_dim_for_board(board_type, board_size)
        logger.info(f"Auto-selected hidden_dim={actual_hidden_dim} for {board_type.value} (size={board_size})")
    else:
        logger.info(f"Using specified hidden_dim={actual_hidden_dim}")

    # Create model
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained model from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = checkpoint
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            if not isinstance(state_dict, dict):
                raise TypeError(f"Unexpected checkpoint type: {type(state_dict).__name__}")
            inferred_hidden_dim = actual_hidden_dim
            inferred_num_hidden_layers = num_hidden_layers

            try:
                accumulator_weight = state_dict.get("accumulator.weight")
                if accumulator_weight is not None and hasattr(accumulator_weight, "shape"):
                    inferred_hidden_dim = int(accumulator_weight.shape[0])
            except Exception:
                pass

            try:
                import re

                layer_indices = set()
                for key in state_dict:
                    match = re.match(r"hidden\.(\d+)\.weight$", key)
                    if match:
                        layer_indices.add(int(match.group(1)))
                if layer_indices:
                    inferred_num_hidden_layers = len(layer_indices)
            except Exception:
                pass

            value_model = RingRiftNNUE(
                board_type=board_type,
                hidden_dim=inferred_hidden_dim,
                num_hidden_layers=inferred_num_hidden_layers,
            )
            value_model.load_state_dict(state_dict)
            value_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load pretrained model, starting fresh: {e}")
            value_model = None

        if value_model is None:
            model = RingRiftNNUEWithPolicy(
                board_type=board_type,
                hidden_dim=actual_hidden_dim,
                num_hidden_layers=num_hidden_layers,
                policy_dropout=policy_dropout,
            )
        else:
            model = RingRiftNNUEWithPolicy.from_value_only(
                value_model, freeze_value_weights=freeze_value
            )
            logger.info(f"Initialized from pretrained model (freeze_value={freeze_value})")
    else:
        model = RingRiftNNUEWithPolicy(
            board_type=board_type,
            hidden_dim=actual_hidden_dim,
            num_hidden_layers=num_hidden_layers,
            policy_dropout=policy_dropout,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Build progressive batch callback if enabled
    from functools import partial
    progressive_callback = None
    if progressive_batch:
        from app.ai.nnue_policy import progressive_batch_schedule
        progressive_callback = partial(
            progressive_batch_schedule,
            min_batch=min_batch_size,
            max_batch=max_batch_size,
        )

    # Create trainer with temperature annealing and label smoothing
    trainer = NNUEPolicyTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        value_weight=value_weight,
        policy_weight=policy_weight,
        temperature=temperature_start,
        label_smoothing=label_smoothing,
        use_kl_loss=effective_use_kl_loss,
        kl_loss_weight=kl_loss_weight,
        grad_clip=grad_clip,
        lr_scheduler=lr_scheduler,
        total_epochs=epochs,
        use_amp=use_amp,
        use_ema=use_ema,
        ema_decay=ema_decay,
        focal_gamma=focal_gamma,
        label_smoothing_warmup=label_smoothing_warmup,
        use_ddp=use_ddp,
        ddp_rank=ddp_rank,
        use_swa=use_swa,
        swa_start_epoch=swa_start_epoch,
        swa_lr=swa_lr,
        progressive_batch_callback=progressive_callback,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Run learning rate finder if requested
    if find_lr:
        logger.info(f"Running learning rate finder ({lr_finder_iterations} iterations)...")
        lr_finder = LearningRateFinder(model, trainer.optimizer, trainer)
        suggested_lr = lr_finder.find(
            train_loader, device, num_iter=lr_finder_iterations
        )
        logger.info(f"Suggested learning rate: {suggested_lr:.2e}")

        # Save LR finder plot
        lr_plot_path = os.path.join(os.path.dirname(save_path), "lr_finder.png")
        lr_finder.plot(lr_plot_path)
        logger.info(f"LR finder plot saved to: {lr_plot_path}")

        # Update learning rate if found
        if suggested_lr > 0:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = suggested_lr
            logger.info(f"Updated learning rate to: {suggested_lr:.2e}")

    logger.info(f"Temperature annealing: {temperature_start} -> {temperature_end} ({temperature_schedule})")
    logger.info(f"Label smoothing: {label_smoothing}" + (f" (warmup: {label_smoothing_warmup} epochs)" if label_smoothing_warmup > 0 else ""))
    logger.info(f"LR scheduler: {lr_scheduler}, grad_clip: {grad_clip}")
    logger.info(f"AMP: {use_amp}, EMA: {use_ema}" + (f" (decay={ema_decay})" if use_ema else ""))
    if focal_gamma > 0:
        logger.info(f"Focal loss enabled with gamma={focal_gamma}")
    if effective_use_kl_loss:
        if kl_loss_weight < 1.0:
            logger.info(f"KL divergence loss ENABLED with mixed loss: {kl_loss_weight:.0%} KL + {1-kl_loss_weight:.0%} CE")
        else:
            logger.info("KL divergence loss ENABLED (pure KL, using MCTS visit distributions)")
    if use_ddp:
        logger.info(f"DDP enabled (rank={ddp_rank})")
    if use_swa:
        logger.info(f"SWA enabled (start epoch={swa_start_epoch if swa_start_epoch > 0 else 'auto'})")
    if progressive_batch:
        logger.info(f"Progressive batch sizing: {min_batch_size} -> {max_batch_size}")
    if gradient_accumulation_steps > 1:
        effective_batch = batch_size * gradient_accumulation_steps
        logger.info(f"Gradient accumulation: {gradient_accumulation_steps} steps (effective batch={effective_batch})")

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_value_loss": [],
        "train_policy_loss": [],
        "val_loss": [],
        "val_value_loss": [],
        "val_policy_loss": [],
        "val_policy_accuracy": [],
        "temperature": [],
    }

    for epoch in range(epochs):
        # Apply temperature annealing
        current_temp = trainer.anneal_temperature(
            epoch=epoch,
            total_epochs=epochs,
            start_temp=temperature_start,
            end_temp=temperature_end,
            schedule=temperature_schedule,
        )
        history["temperature"].append(current_temp)

        # Training
        model.train()
        train_losses = []
        train_value_losses = []
        train_policy_losses = []

        for batch in train_loader:
            features, values, from_idx, to_idx, mask, target, sample_weights, mcts_probs = batch

            # Apply hex augmentation if enabled (before moving to GPU)
            if hex_augmenter is not None:
                features, values, from_idx, to_idx, mask, target, sample_weights, mcts_probs = (
                    augment_batch_with_hex(
                        features, values, from_idx, to_idx, mask, target, sample_weights,
                        mcts_probs if effective_use_kl_loss else None, hex_augmenter
                    )
                )

            features = features.to(device)
            values = values.to(device)
            from_idx = from_idx.to(device)
            to_idx = to_idx.to(device)
            mask = mask.to(device)
            target = target.to(device)
            sample_weights = sample_weights.to(device)
            mcts_probs = mcts_probs.to(device) if mcts_probs is not None and effective_use_kl_loss else None

            total_loss, value_loss, policy_loss = trainer.train_step(
                features, values, from_idx, to_idx, mask, target, mcts_probs, sample_weights
            )
            train_losses.append(total_loss)
            train_value_losses.append(value_loss)
            train_policy_losses.append(policy_loss)

        avg_train_loss = np.mean(train_losses)
        avg_train_value = np.mean(train_value_losses)
        avg_train_policy = np.mean(train_policy_losses)

        # Validation
        val_losses = []
        val_value_losses = []
        val_policy_losses = []
        val_accuracies = []

        for batch in val_loader:
            features, values, from_idx, to_idx, mask, target, sample_weights, mcts_probs = batch
            features = features.to(device)
            values = values.to(device)
            from_idx = from_idx.to(device)
            to_idx = to_idx.to(device)
            mask = mask.to(device)
            target = target.to(device)
            # sample_weights not used in validation - we want unweighted metrics
            mcts_probs = mcts_probs.to(device) if effective_use_kl_loss else None

            total_loss, value_loss, policy_loss, accuracy = trainer.validate(
                features, values, from_idx, to_idx, mask, target, mcts_probs
            )
            val_losses.append(total_loss)
            val_value_losses.append(value_loss)
            val_policy_losses.append(policy_loss)
            val_accuracies.append(accuracy)

        avg_val_loss = np.mean(val_losses)
        avg_val_value = np.mean(val_value_losses)
        avg_val_policy = np.mean(val_policy_losses)
        avg_val_accuracy = np.mean(val_accuracies)

        trainer.update_scheduler(avg_val_loss)

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["train_value_loss"].append(avg_train_value)
        history["train_policy_loss"].append(avg_train_policy)
        history["val_loss"].append(avg_val_loss)
        history["val_value_loss"].append(avg_val_value)
        history["val_policy_loss"].append(avg_val_policy)
        history["val_policy_accuracy"].append(avg_val_accuracy)

        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"train={avg_train_loss:.4f} (v={avg_train_value:.4f}, p={avg_train_policy:.4f}), "
            f"val={avg_val_loss:.4f} (v={avg_val_value:.4f}, p={avg_val_policy:.4f}), "
            f"policy_acc={avg_val_accuracy:.4f}"
        )

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            # Save best model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "board_type": board_type.value,
                "hidden_dim": actual_hidden_dim,
                "num_hidden_layers": num_hidden_layers,
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "val_policy_accuracy": avg_val_accuracy,
                "architecture_version": model.ARCHITECTURE_VERSION,
                "has_policy_head": True,
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
        "model_params_total": total_params,
        "model_params_trainable": trainable_params,
        "hidden_dim": actual_hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "value_weight": value_weight,
        "policy_weight": policy_weight,
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_policy_accuracy": history["val_policy_accuracy"][-1],
        "save_path": save_path,
        "pretrained_path": pretrained_path,
        "freeze_value": freeze_value,
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

    # Expand glob patterns in JSONL paths
    jsonl_paths: List[str] = []
    for pattern in args.jsonl:
        expanded = glob.glob(pattern)
        if expanded:
            jsonl_paths.extend(expanded)
        elif os.path.exists(pattern):
            jsonl_paths.append(pattern)
        else:
            logger.warning(f"JSONL file not found: {pattern}")

    if not db_paths and not jsonl_paths:
        logger.error("No data sources provided. Use --db and/or --jsonl")
        return 1

    if jsonl_paths:
        logger.info(f"JSONL files: {len(jsonl_paths)}")

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

    # Auto-select max_moves_per_position based on board type
    # Hex boards have more cells and thus more potential legal moves
    max_moves_per_position = args.max_moves_per_position
    if max_moves_per_position is None:
        if board_type in (BoardType.HEXAGONAL,):
            max_moves_per_position = 512  # ~969 cells, many moves possible
        elif board_type in (BoardType.HEX8,):
            max_moves_per_position = 256  # Smaller hex board
        elif board_type in (BoardType.SQUARE19,):
            max_moves_per_position = 384  # 361 cells
        else:  # SQUARE8 and default
            max_moves_per_position = 128  # 64 cells
        logger.info(f"Auto-selected max_moves_per_position={max_moves_per_position} for {board_type.value}")

    # Set up output paths
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(PROJECT_ROOT, "runs", f"nnue_policy_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    save_path = args.save_path or os.path.join(
        PROJECT_ROOT, "models", "nnue", f"nnue_policy_{board_type.value}_{args.num_players}p.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Train
    logger.info("Starting NNUE policy training...")
    report = train_nnue_policy(
        db_paths=db_paths,
        board_type=board_type,
        num_players=args.num_players,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        early_stopping_patience=args.early_stopping_patience,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        value_weight=args.value_weight,
        policy_weight=args.policy_weight,
        sample_every_n=args.sample_every_n,
        min_game_length=args.min_game_length,
        max_samples=args.max_samples,
        max_moves_per_position=max_moves_per_position,
        save_path=save_path,
        device=device,
        seed=args.seed,
        pretrained_path=args.pretrained,
        freeze_value=args.freeze_value,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        temperature_schedule=args.temperature_schedule,
        label_smoothing=args.label_smoothing,
        distill_from_winners=args.distill_from_winners,
        winner_weight_boost=args.winner_weight_boost,
        num_workers=args.num_workers,
        min_move_number=args.min_move_number,
        max_move_number=args.max_move_number,
        use_kl_loss=args.use_kl_loss,
        auto_kl_loss=args.auto_kl_loss,
        kl_min_coverage=args.kl_min_coverage,
        kl_min_samples=args.kl_min_samples,
        kl_loss_weight=args.kl_loss_weight,
        grad_clip=args.grad_clip,
        lr_scheduler=args.lr_scheduler,
        use_amp=args.use_amp and not args.no_amp,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        focal_gamma=args.focal_gamma,
        label_smoothing_warmup=args.label_smoothing_warmup,
        save_curves=args.save_curves,
        use_ddp=args.use_ddp,
        ddp_rank=args.ddp_rank,
        use_swa=args.use_swa,
        swa_start_epoch=args.swa_start_epoch,
        swa_lr=args.swa_lr,
        progressive_batch=args.progressive_batch,
        min_batch_size=args.min_batch_size,
        max_batch_size=args.max_batch_size,
        jsonl_paths=jsonl_paths,
        hex_augment=args.hex_augment,
        hex_augment_count=args.hex_augment_count,
        policy_dropout=args.policy_dropout,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        find_lr=args.find_lr,
        lr_finder_iterations=args.lr_finder_iterations,
    )

    # Add metadata to report
    report["run_dir"] = run_dir
    report["db_paths"] = db_paths
    report["jsonl_paths"] = jsonl_paths
    report["created_at"] = datetime.now(timezone.utc).isoformat()

    # Save report
    report_path = os.path.join(run_dir, "nnue_policy_training_report.json")
    with open(report_path, "w") as f:
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

    logger.info("NNUE policy training complete!")
    logger.info(f"  Model saved to: {save_path}")
    best_loss = report.get('best_val_loss')
    logger.info(f"  Best validation loss: {best_loss:.4f}" if best_loss is not None else "  Best validation loss: N/A")
    final_acc = report.get('final_val_policy_accuracy')
    logger.info(f"  Final policy accuracy: {final_acc:.4f}" if final_acc is not None else "  Final policy accuracy: N/A")

    return 0


if __name__ == "__main__":
    sys.exit(main())
