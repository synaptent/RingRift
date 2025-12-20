#!/usr/bin/env python
"""Train EBMO using search-based move quality labels.

This training approach uses move-level quality scores from search evaluation
rather than game outcomes. Each sample has:
- is_best: 1.0 if this is the best move at the position, 0.0 otherwise
- relative_score: Quality score in [0, 1], where 1.0 = best, 0.0 = worst

Training objective:
- Best moves (is_best=1) should have low energy
- Worse moves should have energy proportional to (1 - relative_score)

Usage:
    python scripts/train_ebmo_quality.py --data data/training/ebmo_search_500.npz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.ebmo_network import (
    EBMOConfig,
    EBMONetwork,
    contrastive_energy_loss,
    margin_ranking_loss,
    manifold_boundary_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_ebmo_quality")


class QualityLabeledDataset(Dataset):
    """Dataset with move-level quality labels from search."""

    def __init__(self, data_path: str, device: torch.device):
        """Load search-labeled training data."""
        logger.info(f"Loading data from {data_path}")
        data = np.load(data_path)

        self.boards = torch.from_numpy(data['boards']).float()
        self.globals = torch.from_numpy(data['globals']).float()
        self.actions = torch.from_numpy(data['actions']).float()
        self.is_best = torch.from_numpy(data['is_best']).float()
        self.relative_scores = torch.from_numpy(data['relative_scores']).float()

        # Group samples by position (consecutive samples with same board)
        # For now, we'll use random sampling which is simpler
        self.device = device

        # Separate best moves and non-best moves
        self.best_indices = torch.where(self.is_best > 0.5)[0]
        self.other_indices = torch.where(self.is_best < 0.5)[0]

        logger.info(f"Loaded {len(self.boards)} samples")
        logger.info(f"  Best moves: {len(self.best_indices)}")
        logger.info(f"  Other moves: {len(self.other_indices)}")

    def __len__(self):
        return len(self.best_indices)

    def __getitem__(self, idx):
        """Get a training sample.

        Returns:
            Tuple of (board, globals, positive_action, negative_actions, neg_scores)
        """
        # Get a best move as positive
        pos_idx = self.best_indices[idx]

        # Sample negatives from non-best moves
        num_neg = 8
        neg_indices = self.other_indices[
            torch.randint(0, len(self.other_indices), (num_neg,))
        ]

        return (
            self.boards[pos_idx],
            self.globals[pos_idx],
            self.actions[pos_idx],
            self.actions[neg_indices],
            self.relative_scores[neg_indices],  # Quality scores of negatives
        )


def quality_weighted_loss(
    pos_energy: torch.Tensor,
    neg_energies: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Loss that weights margin by quality difference.

    Better negatives (higher relative_score) should have smaller margin
    since they're closer in quality to the best move.

    Args:
        pos_energy: (batch,) energy of best moves
        neg_energies: (batch, num_neg) energy of worse moves
        neg_scores: (batch, num_neg) relative quality scores of negatives
        margin: Base margin for worst moves (score=0)
    """
    batch_size, num_neg = neg_energies.shape

    # Expand positive energy
    pos_expanded = pos_energy.unsqueeze(1).expand(-1, num_neg)

    # Quality-weighted margin: higher quality negatives have smaller margin
    # This means the model needs to be more confident about truly bad moves
    quality_margin = margin * (1.0 - neg_scores)  # (batch, num_neg)

    # Margin loss: pos_energy + margin < neg_energy
    losses = F.relu(pos_expanded - neg_energies + quality_margin)

    return losses.mean()


def train_epoch(
    model: EBMONetwork,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_boundary_loss: bool = True,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_margin = 0.0
    total_quality = 0.0
    total_boundary = 0.0
    num_batches = 0

    for batch in dataloader:
        boards, globals_vec, pos_actions, neg_actions, neg_scores = batch

        boards = boards.to(device)
        globals_vec = globals_vec.to(device)
        pos_actions = pos_actions.to(device)
        neg_actions = neg_actions.to(device)
        neg_scores = neg_scores.to(device)

        optimizer.zero_grad()

        # Encode state
        state_embed = model.state_encoder(boards, globals_vec)

        # Encode positive action
        pos_embed = model.action_encoder(pos_actions)
        pos_energy = model.energy_head(state_embed, pos_embed)

        # Encode negative actions
        batch_size, num_neg, action_dim = neg_actions.shape
        neg_flat = neg_actions.view(-1, action_dim)
        neg_embed = model.action_encoder(neg_flat).view(batch_size, num_neg, -1)

        # Compute negative energies
        neg_energies = []
        for j in range(num_neg):
            ne = model.energy_head(state_embed, neg_embed[:, j, :])
            neg_energies.append(ne)
        neg_energy = torch.stack(neg_energies, dim=1)

        # Losses
        contrastive = contrastive_energy_loss(pos_energy, neg_energy, temperature=0.1)
        margin = margin_ranking_loss(pos_energy, neg_energy, margin=1.0)
        quality = quality_weighted_loss(pos_energy, neg_energy, neg_scores, margin=2.0)

        # Manifold boundary loss: push random embeddings to high energy
        boundary = torch.tensor(0.0, device=device)
        if use_boundary_loss:
            boundary = manifold_boundary_loss(
                model, state_embed, pos_embed,
                num_random_samples=5, boundary_margin=3.0
            )

        # Combined loss with boundary term to prevent energy landscape escape
        loss = 0.2 * contrastive + 0.25 * margin + 0.4 * quality + 0.15 * boundary

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_contrastive += contrastive.item()
        total_margin += margin.item()
        total_quality += quality.item()
        total_boundary += boundary.item() if use_boundary_loss else 0.0
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'contrastive': total_contrastive / num_batches,
        'margin': total_margin / num_batches,
        'quality': total_quality / num_batches,
        'boundary': total_boundary / num_batches,
    }


def evaluate(
    model: EBMONetwork,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    total = 0
    margin_satisfied = 0

    with torch.no_grad():
        for batch in dataloader:
            boards, globals_vec, pos_actions, neg_actions, neg_scores = batch

            boards = boards.to(device)
            globals_vec = globals_vec.to(device)
            pos_actions = pos_actions.to(device)
            neg_actions = neg_actions.to(device)

            # Encode state
            state_embed = model.state_encoder(boards, globals_vec)

            # Encode positive action
            pos_embed = model.action_encoder(pos_actions)
            pos_energy = model.energy_head(state_embed, pos_embed)

            # Encode negative actions
            batch_size, num_neg, action_dim = neg_actions.shape
            neg_flat = neg_actions.view(-1, action_dim)
            neg_embed = model.action_encoder(neg_flat).view(batch_size, num_neg, -1)

            # Compute negative energies
            for j in range(num_neg):
                neg_e = model.energy_head(state_embed, neg_embed[:, j, :])

                # Accuracy: positive should have lower energy than negative
                correct += (pos_energy < neg_e).sum().item()
                total += len(pos_energy)

                # Margin satisfaction
                margin_satisfied += (pos_energy + 0.5 < neg_e).sum().item()

    return {
        'accuracy': correct / total if total > 0 else 0,
        'margin_sat': margin_satisfied / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train EBMO with quality labels")
    parser.add_argument("--data", type=str, required=True, help="Training data NPZ")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="models/ebmo", help="Output dir")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load dataset
    full_dataset = QualityLabeledDataset(args.data, device)

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    logger.info(f"Train: {train_size}, Val: {val_size}")

    # Detect input channels from data
    sample_board = full_dataset.boards[0]
    num_channels = sample_board.shape[0]
    logger.info(f"Detected {num_channels} input channels from data")

    # Create model with matching config
    config = EBMOConfig(num_input_channels=num_channels)
    model = EBMONetwork(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step()

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"loss={train_metrics['loss']:.4f}, "
                f"quality={train_metrics['quality']:.4f}, "
                f"val_acc={val_metrics['accuracy']:.2%}, "
                f"margin_sat={val_metrics['margin_sat']:.2%}"
            )

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            checkpoint_path = output_dir / "ebmo_quality_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_accuracy': val_metrics['accuracy'],
                'margin_satisfaction': val_metrics['margin_sat'],
            }, checkpoint_path)

        # Save periodic checkpoints
        if (epoch + 1) % 25 == 0:
            checkpoint_path = output_dir / f"ebmo_quality_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
            }, checkpoint_path)

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
