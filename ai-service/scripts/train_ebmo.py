#!/usr/bin/env python
"""Train EBMO (Energy-Based Move Optimization) network.

This script trains the EBMO energy network that learns to score
(state, action) pairs. The trained model is used by EBMO_AI for
gradient-based move selection at inference time.

Training data sources:
1. Existing self-play NPZ files (converted to EBMO format)
2. Game databases (SQLite) with position/move data

Usage:
    # Train on existing NPZ data
    python scripts/train_ebmo.py --data-dir data/training --epochs 100

    # Train with validation split
    python scripts/train_ebmo.py --data-dir data/training --val-split 0.1

    # Resume training from checkpoint
    python scripts/train_ebmo.py --data-dir data/training --resume models/ebmo/checkpoint.pt

    # Distributed training (multi-GPU)
    torchrun --nproc_per_node=2 scripts/train_ebmo.py --data-dir data/training

Output:
    - Model checkpoint: models/ebmo/ebmo_square8.pt
    - Training logs: logs/ebmo_training.log
    - TensorBoard logs: runs/ebmo_training/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Set up path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Import EBMO components
from app.ai.ebmo_network import (
    EBMOConfig,
    EBMONetwork,
    contrastive_energy_loss,
    outcome_weighted_energy_loss,
)

# Distributed training utilities
try:
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
    )
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_ebmo")


def find_training_data(data_dir: str, pattern: str = "*.npz") -> List[Path]:
    """Find all training data files in directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return []

    files = list(data_path.glob(pattern))
    # Also check subdirectories
    files.extend(data_path.glob(f"**/{pattern}"))

    # Deduplicate
    files = list(set(files))

    # Sort by modification time (newest first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    return files


def load_npz_data(file_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load and validate NPZ data file."""
    try:
        data = np.load(file_path, allow_pickle=True)

        # Check for required keys
        if 'features' not in data or 'values' not in data:
            logger.warning(f"Skipping {file_path}: missing required keys")
            return None

        return dict(data)
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def convert_npz_to_ebmo_format(
    npz_data: Dict[str, np.ndarray],
    board_size: int = 8,
    num_negatives: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert standard NPZ training data to EBMO format.

    Args:
        npz_data: Dictionary with 'features', 'globals', 'values', 'pol_indices', 'pol_values'
        board_size: Board size for action encoding
        num_negatives: Number of negative samples per positive

    Returns:
        Tuple of (board_features, global_features, positive_actions, negative_actions, outcomes)
    """
    features = npz_data['features']  # (N, C, H, W)
    values = npz_data['values']  # (N,)

    # Use globals if available, otherwise create dummy
    if 'globals' in npz_data:
        globals_arr = npz_data['globals']
    else:
        globals_arr = np.zeros((len(features), 20), dtype=np.float32)

    # Get policy data for positive actions
    pol_indices = npz_data.get('pol_indices', None)
    pol_values = npz_data.get('pol_values', None)

    num_samples = len(features)
    action_dim = 14  # EBMO action dimension

    # Initialize outputs
    positive_actions = np.zeros((num_samples, action_dim), dtype=np.float32)
    negative_actions = np.zeros((num_samples, num_negatives, action_dim), dtype=np.float32)
    outcomes = values.astype(np.float32)

    # Convert policy indices to action features
    for i in range(num_samples):
        # Get positive action from policy
        if pol_indices is not None and len(pol_indices[i]) > 0:
            # Use highest probability action as positive
            idx = pol_indices[i]
            if pol_values is not None and len(pol_values[i]) > 0:
                best_idx = idx[np.argmax(pol_values[i])]
            else:
                best_idx = idx[0]

            # Decode index to action features
            positive_actions[i] = decode_action_index(best_idx, board_size, action_dim)
        else:
            # Random action if no policy available
            positive_actions[i] = np.random.randn(action_dim).astype(np.float32)

        # Generate negative actions (random different actions)
        for j in range(num_negatives):
            # Random action that's different from positive
            neg_action = np.random.randn(action_dim).astype(np.float32)
            negative_actions[i, j] = neg_action

    return features, globals_arr, positive_actions, negative_actions, outcomes


def decode_action_index(
    index: int,
    board_size: int,
    action_dim: int,
) -> np.ndarray:
    """Decode a flat action index into EBMO action features.

    This is a simplified decoder - the actual mapping depends on
    the policy encoding used in training data.
    """
    action = np.zeros(action_dim, dtype=np.float32)

    # Simplified: treat index as encoding (from_pos, to_pos, move_type)
    total_squares = board_size * board_size

    # Extract from and to positions
    from_idx = index // total_squares
    to_idx = index % total_squares

    from_x = (from_idx % board_size) / board_size
    from_y = (from_idx // board_size) / board_size
    to_x = (to_idx % board_size) / board_size
    to_y = (to_idx // board_size) / board_size

    # Set action features
    action[0] = from_x
    action[1] = from_y
    action[2] = to_x
    action[3] = to_y

    # Move type (one-hot, slots 4-11)
    move_type = (index // (total_squares * total_squares)) % 8
    action[4 + move_type] = 1.0

    # Direction (slots 12-13)
    action[12] = to_x - from_x
    action[13] = to_y - from_y

    return action


class CombinedEBMODataset(Dataset):
    """Dataset that combines multiple NPZ files into EBMO format."""

    def __init__(
        self,
        npz_files: List[Path],
        board_size: int = 8,
        num_negatives: int = 7,
        max_samples_per_file: Optional[int] = None,
    ):
        self.board_size = board_size
        self.num_negatives = num_negatives

        # Load and combine all files
        self.board_features = []
        self.global_features = []
        self.positive_actions = []
        self.negative_actions = []
        self.outcomes = []

        for file_path in npz_files:
            npz_data = load_npz_data(file_path)
            if npz_data is None:
                continue

            board, globals_arr, pos_act, neg_act, outcomes = convert_npz_to_ebmo_format(
                npz_data,
                board_size=board_size,
                num_negatives=num_negatives,
            )

            # Optionally limit samples per file
            if max_samples_per_file and len(board) > max_samples_per_file:
                indices = np.random.choice(len(board), max_samples_per_file, replace=False)
                board = board[indices]
                globals_arr = globals_arr[indices]
                pos_act = pos_act[indices]
                neg_act = neg_act[indices]
                outcomes = outcomes[indices]

            self.board_features.append(board)
            self.global_features.append(globals_arr)
            self.positive_actions.append(pos_act)
            self.negative_actions.append(neg_act)
            self.outcomes.append(outcomes)

            logger.info(f"Loaded {len(board)} samples from {file_path.name}")

        if not self.board_features:
            raise ValueError("No valid training data found!")

        # Concatenate all data
        self.board_features = np.concatenate(self.board_features, axis=0)
        self.global_features = np.concatenate(self.global_features, axis=0)
        self.positive_actions = np.concatenate(self.positive_actions, axis=0)
        self.negative_actions = np.concatenate(self.negative_actions, axis=0)
        self.outcomes = np.concatenate(self.outcomes, axis=0)

        logger.info(f"Total samples: {len(self.board_features)}")

    def __len__(self) -> int:
        return len(self.board_features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'board_features': torch.from_numpy(self.board_features[idx]),
            'global_features': torch.from_numpy(self.global_features[idx]),
            'positive_action': torch.from_numpy(self.positive_actions[idx]),
            'negative_actions': torch.from_numpy(self.negative_actions[idx]),
            'outcome': torch.tensor(self.outcomes[idx], dtype=torch.float32),
        }


def train_epoch(
    model: EBMONetwork,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: EBMOConfig,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_outcome_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        board_features = batch['board_features'].to(device)
        global_features = batch['global_features'].to(device)
        positive_action = batch['positive_action'].to(device)
        negative_actions = batch['negative_actions'].to(device)
        outcomes = batch['outcome'].to(device)

        optimizer.zero_grad()

        # Forward pass
        # Encode state
        state_embed = model.state_encoder(board_features, global_features)

        # Encode positive and negative actions
        pos_embed = model.action_encoder(positive_action)

        batch_size, num_neg, action_dim = negative_actions.shape
        neg_flat = negative_actions.view(-1, action_dim)
        neg_embed = model.action_encoder(neg_flat)
        neg_embed = neg_embed.view(batch_size, num_neg, -1)

        # Compute energies
        pos_energy = model.energy_head(state_embed, pos_embed)

        neg_energies = []
        for i in range(num_neg):
            neg_e = model.energy_head(state_embed, neg_embed[:, i, :])
            neg_energies.append(neg_e)
        neg_energy = torch.stack(neg_energies, dim=1)

        # Compute losses
        contrastive_loss = contrastive_energy_loss(
            pos_energy,
            neg_energy,
            temperature=config.contrastive_temperature,
        )

        outcome_loss = outcome_weighted_energy_loss(
            pos_energy,
            outcomes,
        )

        # Combined loss
        loss = config.outcome_weight * outcome_loss + (1 - config.outcome_weight) * contrastive_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_contrastive_loss += contrastive_loss.item()
        total_outcome_loss += outcome_loss.item()
        num_batches += 1

        # Log progress
        if batch_idx % 100 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"(contrastive: {contrastive_loss.item():.4f}, outcome: {outcome_loss.item():.4f})"
            )

    return {
        'loss': total_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches,
        'outcome_loss': total_outcome_loss / num_batches,
    }


def validate(
    model: EBMONetwork,
    dataloader: DataLoader,
    device: torch.device,
    config: EBMOConfig,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            board_features = batch['board_features'].to(device)
            global_features = batch['global_features'].to(device)
            positive_action = batch['positive_action'].to(device)
            negative_actions = batch['negative_actions'].to(device)
            outcomes = batch['outcome'].to(device)

            # Forward pass
            state_embed = model.state_encoder(board_features, global_features)
            pos_embed = model.action_encoder(positive_action)

            batch_size, num_neg, action_dim = negative_actions.shape
            neg_flat = negative_actions.view(-1, action_dim)
            neg_embed = model.action_encoder(neg_flat)
            neg_embed = neg_embed.view(batch_size, num_neg, -1)

            # Compute energies
            pos_energy = model.energy_head(state_embed, pos_embed)

            neg_energies = []
            for i in range(num_neg):
                neg_e = model.energy_head(state_embed, neg_embed[:, i, :])
                neg_energies.append(neg_e)
            neg_energy = torch.stack(neg_energies, dim=1)

            # Compute loss
            contrastive_loss = contrastive_energy_loss(
                pos_energy,
                neg_energy,
                temperature=config.contrastive_temperature,
            )
            outcome_loss = outcome_weighted_energy_loss(pos_energy, outcomes)
            loss = config.outcome_weight * outcome_loss + (1 - config.outcome_weight) * contrastive_loss

            total_loss += loss.item() * batch_size

            # Accuracy: positive should have lower energy than negatives
            # Shape: pos_energy (B,), neg_energy (B, num_neg)
            pos_lower = (pos_energy.unsqueeze(1) < neg_energy).all(dim=1)
            total_correct += pos_lower.sum().item()
            total_samples += batch_size

    return {
        'val_loss': total_loss / total_samples,
        'val_accuracy': total_correct / total_samples,
    }


def save_checkpoint(
    model: EBMONetwork,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    output_path: Path,
    config: EBMOConfig,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'state_embed_dim': config.state_embed_dim,
            'action_embed_dim': config.action_embed_dim,
            'energy_hidden_dim': config.energy_hidden_dim,
            'num_energy_layers': config.num_energy_layers,
            'board_size': config.board_size,
        },
        'timestamp': datetime.now().isoformat(),
    }

    torch.save(checkpoint, output_path)
    logger.info(f"Saved checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train EBMO energy network")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/training",
                        help="Directory containing training data")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of data files to use")
    parser.add_argument("--max-samples-per-file", type=int, default=50000,
                        help="Maximum samples per data file")

    # Model arguments
    parser.add_argument("--board-size", type=int, default=8,
                        help="Board size")
    parser.add_argument("--state-embed-dim", type=int, default=256,
                        help="State embedding dimension")
    parser.add_argument("--action-embed-dim", type=int, default=128,
                        help="Action embedding dimension")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--num-negatives", type=int, default=7,
                        help="Number of negative samples")
    parser.add_argument("--outcome-weight", type=float, default=0.5,
                        help="Weight for outcome loss vs contrastive loss")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="models/ebmo",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    # Hardware arguments
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Check for distributed training
    is_dist = HAS_DISTRIBUTED and is_distributed()
    if is_dist:
        setup_distributed()
        device = get_device_for_rank()
        logger.info(f"Distributed training: rank {get_rank()}/{get_world_size()}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find training data
    logger.info(f"Looking for training data in: {args.data_dir}")
    npz_files = find_training_data(args.data_dir)

    if not npz_files:
        logger.error("No training data found!")
        sys.exit(1)

    logger.info(f"Found {len(npz_files)} data files")

    # Limit files if specified
    if args.max_files:
        npz_files = npz_files[:args.max_files]
        logger.info(f"Using {len(npz_files)} files")

    # Create dataset
    logger.info("Loading training data...")
    dataset = CombinedEBMODataset(
        npz_files=npz_files,
        board_size=args.board_size,
        num_negatives=args.num_negatives,
        max_samples_per_file=args.max_samples_per_file,
    )

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create data loaders
    if is_dist:
        train_sampler = get_distributed_sampler(train_dataset)
        val_sampler = get_distributed_sampler(val_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    config = EBMOConfig(
        board_size=args.board_size,
        state_embed_dim=args.state_embed_dim,
        action_embed_dim=args.action_embed_dim,
        num_negatives=args.num_negatives,
        outcome_weight=args.outcome_weight,
    )

    model = EBMONetwork(config)
    model = model.to(device)

    # Wrap for distributed
    if is_dist:
        model = wrap_model_ddp(model)

    # Optimizer
    lr = args.lr
    if is_dist:
        lr = scale_learning_rate(lr, get_world_size())

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=lr * 0.01,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'metrics' in checkpoint and 'val_loss' in checkpoint['metrics']:
            best_val_loss = checkpoint['metrics']['val_loss']

    # Training loop
    logger.info("Starting training...")
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        if is_dist and train_sampler:
            train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
        )

        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            config=config,
        )

        # Update scheduler
        scheduler.step()

        # Log metrics
        epoch_time = time.time() - epoch_start

        if not is_dist or is_main_process():
            logger.info(
                f"Epoch {epoch}/{args.epochs} "
                f"({epoch_time:.1f}s) - "
                f"train_loss: {train_metrics['loss']:.4f}, "
                f"val_loss: {val_metrics['val_loss']:.4f}, "
                f"val_acc: {val_metrics['val_accuracy']:.3f}"
            )

            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_path = output_dir / "ebmo_square8_best.pt"
                save_checkpoint(
                    model=model.module if is_dist else model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    output_path=best_path,
                    config=config,
                )

            # Save periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
                checkpoint_path = output_dir / f"ebmo_square8_epoch{epoch}.pt"
                save_checkpoint(
                    model=model.module if is_dist else model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                    output_path=checkpoint_path,
                    config=config,
                )

    # Final save
    if not is_dist or is_main_process():
        final_path = output_dir / "ebmo_square8.pt"
        save_checkpoint(
            model=model.module if is_dist else model,
            optimizer=optimizer,
            epoch=args.epochs - 1,
            metrics={**train_metrics, **val_metrics},
            output_path=final_path,
            config=config,
        )

        total_time = time.time() - training_start
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Final model saved to: {final_path}")

    # Cleanup distributed
    if is_dist:
        cleanup_distributed()


if __name__ == "__main__":
    main()
