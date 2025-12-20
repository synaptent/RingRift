#!/usr/bin/env python
"""Train EBMO on expert data.

This script trains EBMO directly on the expert data format which already
has action features computed.

Usage:
    python scripts/train_ebmo_expert.py --data data/training/ebmo_expert_heuristic.npz
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.ebmo_network import (
    EBMOConfig,
    EBMONetwork,
    contrastive_energy_loss,
    outcome_weighted_energy_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_ebmo_expert")


class ExpertEBMODataset(Dataset):
    """Dataset for EBMO training from expert data."""

    def __init__(
        self,
        npz_path: str,
        num_negatives: int = 15,
        board_size: int = 8,
    ):
        data = np.load(npz_path, allow_pickle=True)

        self.features = data['features']  # (N, 56, 8, 8)
        self.globals = data['globals']  # (N, 20)
        self.actions = data['actions']  # (N, 14) - positive actions
        self.outcomes = data['values']  # (N,) - +1 win, -1 loss

        self.num_negatives = num_negatives
        self.board_size = board_size
        self.action_dim = 14

        logger.info(f"Loaded {len(self.features)} samples from {npz_path}")

    def __len__(self):
        return len(self.features)

    def _generate_random_action(self) -> np.ndarray:
        """Generate a random action for negative sampling."""
        action = np.zeros(self.action_dim, dtype=np.float32)

        # Random positions
        from_x = np.random.rand()
        from_y = np.random.rand()
        to_x = np.random.rand()
        to_y = np.random.rand()

        action[0] = from_x
        action[1] = from_y
        action[2] = to_x
        action[3] = to_y

        # Random move type (one-hot)
        move_type = np.random.randint(0, 8)
        action[4 + move_type] = 1.0

        # Direction
        action[12] = to_x - from_x
        action[13] = to_y - from_y

        return action

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get positive sample
        board = self.features[idx]
        globals_vec = self.globals[idx]
        pos_action = self.actions[idx]
        outcome = self.outcomes[idx]

        # Generate negative samples (random actions)
        neg_actions = np.stack([
            self._generate_random_action()
            for _ in range(self.num_negatives)
        ])

        return {
            'board_features': torch.from_numpy(board),
            'global_features': torch.from_numpy(globals_vec),
            'positive_action': torch.from_numpy(pos_action),
            'negative_actions': torch.from_numpy(neg_actions),
            'outcome': torch.tensor(outcome, dtype=torch.float32),
        }


def train_epoch(
    model: EBMONetwork,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: EBMOConfig,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_outcome = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        board = batch['board_features'].to(device)
        globals_vec = batch['global_features'].to(device)
        pos_action = batch['positive_action'].to(device)
        neg_actions = batch['negative_actions'].to(device)
        outcomes = batch['outcome'].to(device)

        optimizer.zero_grad()

        # Encode state
        state_embed = model.state_encoder(board, globals_vec)

        # Encode positive action
        pos_embed = model.action_encoder(pos_action)

        # Encode negative actions
        batch_size, num_neg, action_dim = neg_actions.shape
        neg_flat = neg_actions.view(-1, action_dim)
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
            pos_energy, neg_energy, temperature=config.contrastive_temperature
        )
        outcome_loss = outcome_weighted_energy_loss(pos_energy, outcomes)

        # Combined loss
        loss = config.outcome_weight * outcome_loss + (1 - config.outcome_weight) * contrastive_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_contrastive += contrastive_loss.item()
        total_outcome += outcome_loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (c: {contrastive_loss.item():.4f}, o: {outcome_loss.item():.4f})"
            )

    return {
        'loss': total_loss / num_batches,
        'contrastive_loss': total_contrastive / num_batches,
        'outcome_loss': total_outcome / num_batches,
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
            board = batch['board_features'].to(device)
            globals_vec = batch['global_features'].to(device)
            pos_action = batch['positive_action'].to(device)
            neg_actions = batch['negative_actions'].to(device)
            outcomes = batch['outcome'].to(device)

            # Forward pass
            state_embed = model.state_encoder(board, globals_vec)
            pos_embed = model.action_encoder(pos_action)

            batch_size, num_neg, action_dim = neg_actions.shape
            neg_flat = neg_actions.view(-1, action_dim)
            neg_embed = model.action_encoder(neg_flat)
            neg_embed = neg_embed.view(batch_size, num_neg, -1)

            pos_energy = model.energy_head(state_embed, pos_embed)

            neg_energies = []
            for i in range(num_neg):
                neg_e = model.energy_head(state_embed, neg_embed[:, i, :])
                neg_energies.append(neg_e)
            neg_energy = torch.stack(neg_energies, dim=1)

            # Compute loss
            contrastive_loss = contrastive_energy_loss(
                pos_energy, neg_energy, temperature=config.contrastive_temperature
            )
            outcome_loss = outcome_weighted_energy_loss(pos_energy, outcomes)
            loss = config.outcome_weight * outcome_loss + (1 - config.outcome_weight) * contrastive_loss

            total_loss += loss.item() * batch_size

            # Check if positive has lower energy than all negatives
            all_energies = torch.cat([pos_energy.unsqueeze(1), neg_energy], dim=1)
            predictions = all_energies.argmin(dim=1)
            total_correct += (predictions == 0).sum().item()
            total_samples += batch_size

    return {
        'val_loss': total_loss / total_samples,
        'val_accuracy': total_correct / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Train EBMO on expert data")
    parser.add_argument("--data", type=str, required=True, help="Path to expert NPZ file")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-negatives", type=int, default=15, help="Negative samples")
    parser.add_argument("--outcome-weight", type=float, default=0.5, help="Outcome loss weight")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--output-dir", type=str, default="models/ebmo", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, mps, cpu)")

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

    # Create config
    config = EBMOConfig(
        num_negatives=args.num_negatives,
        outcome_weight=args.outcome_weight,
        learning_rate=args.lr,
    )

    # Load dataset
    dataset = ExpertEBMODataset(
        args.data,
        num_negatives=args.num_negatives,
    )

    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info(f"Train size: {train_size}, Val size: {val_size}")

    # Create model
    model = EBMONetwork(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, config, epoch)
        val_metrics = validate(model, val_loader, device, config)
        scheduler.step()

        logger.info(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}, val_acc={val_metrics['val_accuracy']:.4f}"
        )

        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            checkpoint_path = output_dir / "ebmo_expert_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'val_loss': best_val_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"ebmo_expert_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
            }, checkpoint_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
