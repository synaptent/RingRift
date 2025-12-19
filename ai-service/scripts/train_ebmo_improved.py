#!/usr/bin/env python
"""Improved EBMO training with hard negatives and margin loss.

Key improvements over train_ebmo_expert.py:
1. Hard negatives: Uses moves from losing games instead of random moves
2. Margin-based loss: Ensures clear energy separation between good/bad moves
3. Curriculum learning: Starts with easy negatives, increases difficulty
4. Longer training with warmup

Usage:
    python scripts/train_ebmo_improved.py --data data/training/ebmo_expert_500.npz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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
    margin_ranking_loss,
    outcome_weighted_energy_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_ebmo_improved")


class ImprovedEBMODataset(Dataset):
    """Dataset with hard negatives from losing games.

    Instead of random negatives, uses:
    1. Moves from losing games (hard negatives)
    2. Moves from same position but different outcome (near-misses)
    3. Some random moves (easy negatives for curriculum)
    """

    def __init__(
        self,
        npz_path: str,
        num_hard_negatives: int = 8,
        num_random_negatives: int = 4,
        board_size: int = 8,
    ):
        data = np.load(npz_path, allow_pickle=True)

        self.features = data['features']  # (N, 56, 8, 8)
        self.globals = data['globals']  # (N, 20)
        self.actions = data['actions']  # (N, 14) - positive actions
        self.outcomes = data['values']  # (N,) - +1 win, -1 loss

        self.num_hard_negatives = num_hard_negatives
        self.num_random_negatives = num_random_negatives
        self.board_size = board_size
        self.action_dim = 14

        # Separate winning and losing samples for hard negative mining
        self.winning_indices = np.where(self.outcomes > 0)[0]
        self.losing_indices = np.where(self.outcomes < 0)[0]

        logger.info(f"Loaded {len(self.features)} samples from {npz_path}")
        logger.info(f"  Winning samples: {len(self.winning_indices)}")
        logger.info(f"  Losing samples: {len(self.losing_indices)}")

    def __len__(self):
        return len(self.features)

    def _generate_random_action(self) -> np.ndarray:
        """Generate a random action for easy negative sampling."""
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

        # Generate hard negatives (moves from losing games)
        # These are moves that seemed reasonable but led to losses
        hard_neg_actions = []
        if len(self.losing_indices) > 0:
            # Sample from losing game moves
            hard_indices = np.random.choice(
                self.losing_indices,
                size=min(self.num_hard_negatives, len(self.losing_indices)),
                replace=False
            )
            for hi in hard_indices:
                hard_neg_actions.append(self.actions[hi])

        # Pad with random if not enough hard negatives
        while len(hard_neg_actions) < self.num_hard_negatives:
            hard_neg_actions.append(self._generate_random_action())

        hard_neg_actions = np.stack(hard_neg_actions[:self.num_hard_negatives])

        # Generate random negatives (easy negatives for curriculum)
        random_neg_actions = np.stack([
            self._generate_random_action()
            for _ in range(self.num_random_negatives)
        ])

        return {
            'board_features': torch.from_numpy(board),
            'global_features': torch.from_numpy(globals_vec),
            'positive_action': torch.from_numpy(pos_action),
            'hard_negative_actions': torch.from_numpy(hard_neg_actions),
            'random_negative_actions': torch.from_numpy(random_neg_actions),
            'outcome': torch.tensor(outcome, dtype=torch.float32),
        }


def train_epoch(
    model: EBMONetwork,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: EBMOConfig,
    epoch: int,
    use_margin_loss: bool = True,
    margin: float = 1.0,
    curriculum_hard_weight: float = 0.5,
) -> Dict[str, float]:
    """Train for one epoch with improved losses."""
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_margin = 0.0
    total_outcome = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        board = batch['board_features'].to(device)
        globals_vec = batch['global_features'].to(device)
        pos_action = batch['positive_action'].to(device)
        hard_neg_actions = batch['hard_negative_actions'].to(device)
        random_neg_actions = batch['random_negative_actions'].to(device)
        outcomes = batch['outcome'].to(device)

        optimizer.zero_grad()

        # Encode state
        state_embed = model.state_encoder(board, globals_vec)

        # Encode positive action
        pos_embed = model.action_encoder(pos_action)

        # Encode hard negative actions
        batch_size, num_hard, action_dim = hard_neg_actions.shape
        hard_flat = hard_neg_actions.view(-1, action_dim)
        hard_embed = model.action_encoder(hard_flat)
        hard_embed = hard_embed.view(batch_size, num_hard, -1)

        # Encode random negative actions
        _, num_random, _ = random_neg_actions.shape
        random_flat = random_neg_actions.view(-1, action_dim)
        random_embed = model.action_encoder(random_flat)
        random_embed = random_embed.view(batch_size, num_random, -1)

        # Compute energies
        pos_energy = model.energy_head(state_embed, pos_embed)

        hard_energies = []
        for i in range(num_hard):
            e = model.energy_head(state_embed, hard_embed[:, i, :])
            hard_energies.append(e)
        hard_energy = torch.stack(hard_energies, dim=1)

        random_energies = []
        for i in range(num_random):
            e = model.energy_head(state_embed, random_embed[:, i, :])
            random_energies.append(e)
        random_energy = torch.stack(random_energies, dim=1)

        # Combine all negatives with curriculum weighting
        # Early epochs: more random negatives (easier)
        # Later epochs: more hard negatives (harder)
        all_neg_energy = torch.cat([hard_energy, random_energy], dim=1)

        # Compute losses
        contrastive_loss = contrastive_energy_loss(
            pos_energy, all_neg_energy, temperature=config.contrastive_temperature
        )

        if use_margin_loss:
            # Margin loss ensures clear separation
            margin_loss = margin_ranking_loss(pos_energy, all_neg_energy, margin=margin)
        else:
            margin_loss = torch.tensor(0.0, device=device)

        outcome_loss = outcome_weighted_energy_loss(pos_energy, outcomes)

        # Combined loss with weights
        # Curriculum: increase margin weight over epochs
        margin_weight = min(0.5, 0.1 + epoch * 0.01)  # 0.1 -> 0.5 over 40 epochs

        loss = (
            0.3 * contrastive_loss +
            margin_weight * margin_loss +
            config.outcome_weight * outcome_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_contrastive += contrastive_loss.item()
        total_margin += margin_loss.item()
        total_outcome += outcome_loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (c: {contrastive_loss.item():.4f}, "
                f"m: {margin_loss.item():.4f}, o: {outcome_loss.item():.4f})"
            )

    return {
        'loss': total_loss / num_batches,
        'contrastive_loss': total_contrastive / num_batches,
        'margin_loss': total_margin / num_batches,
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
    total_margin_satisfied = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            board = batch['board_features'].to(device)
            globals_vec = batch['global_features'].to(device)
            pos_action = batch['positive_action'].to(device)
            hard_neg_actions = batch['hard_negative_actions'].to(device)
            random_neg_actions = batch['random_negative_actions'].to(device)
            outcomes = batch['outcome'].to(device)

            # Forward pass
            state_embed = model.state_encoder(board, globals_vec)
            pos_embed = model.action_encoder(pos_action)

            batch_size, num_hard, action_dim = hard_neg_actions.shape
            hard_flat = hard_neg_actions.view(-1, action_dim)
            hard_embed = model.action_encoder(hard_flat).view(batch_size, num_hard, -1)

            _, num_random, _ = random_neg_actions.shape
            random_flat = random_neg_actions.view(-1, action_dim)
            random_embed = model.action_encoder(random_flat).view(batch_size, num_random, -1)

            pos_energy = model.energy_head(state_embed, pos_embed)

            all_neg_energies = []
            for i in range(num_hard):
                all_neg_energies.append(model.energy_head(state_embed, hard_embed[:, i, :]))
            for i in range(num_random):
                all_neg_energies.append(model.energy_head(state_embed, random_embed[:, i, :]))
            neg_energy = torch.stack(all_neg_energies, dim=1)

            # Compute loss
            contrastive_loss = contrastive_energy_loss(
                pos_energy, neg_energy, temperature=config.contrastive_temperature
            )
            margin_loss = margin_ranking_loss(pos_energy, neg_energy, margin=1.0)
            outcome_loss = outcome_weighted_energy_loss(pos_energy, outcomes)
            loss = 0.3 * contrastive_loss + 0.3 * margin_loss + config.outcome_weight * outcome_loss

            total_loss += loss.item() * batch_size

            # Check if positive has lower energy than all negatives
            all_energies = torch.cat([pos_energy.unsqueeze(1), neg_energy], dim=1)
            predictions = all_energies.argmin(dim=1)
            total_correct += (predictions == 0).sum().item()

            # Check margin satisfaction (pos_energy < min(neg_energy) - margin)
            min_neg_energy = neg_energy.min(dim=1)[0]
            margin_satisfied = (pos_energy < min_neg_energy - 0.5).sum().item()
            total_margin_satisfied += margin_satisfied

            total_samples += batch_size

    return {
        'val_loss': total_loss / total_samples,
        'val_accuracy': total_correct / total_samples,
        'margin_satisfaction': total_margin_satisfied / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Train EBMO with improved losses")
    parser.add_argument("--data", type=str, required=True, help="Path to expert NPZ file")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--num-hard-negatives", type=int, default=8, help="Hard negative samples")
    parser.add_argument("--num-random-negatives", type=int, default=4, help="Random negative samples")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for ranking loss")
    parser.add_argument("--outcome-weight", type=float, default=0.4, help="Outcome loss weight")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--output-dir", type=str, default="models/ebmo", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, mps, cpu)")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="LR warmup epochs")

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

    # Create config with improved defaults
    config = EBMOConfig(
        num_negatives=args.num_hard_negatives + args.num_random_negatives,
        outcome_weight=args.outcome_weight,
        learning_rate=args.lr,
        # Increased inference steps
        optim_steps=100,
        num_restarts=8,
        projection_temperature=0.3,
    )

    # Load dataset with hard negatives
    dataset = ImprovedEBMODataset(
        args.data,
        num_hard_negatives=args.num_hard_negatives,
        num_random_negatives=args.num_random_negatives,
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

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_margin_sat = 0.0

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, config, epoch,
            use_margin_loss=True, margin=args.margin
        )
        val_metrics = validate(model, val_loader, device, config)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}, val_acc={val_metrics['val_accuracy']:.4f}, "
            f"margin_sat={val_metrics['margin_satisfaction']:.4f}, lr={current_lr:.6f}"
        )

        # Save best model (prefer margin satisfaction over val_loss)
        improved = False
        if val_metrics['margin_satisfaction'] > best_margin_sat + 0.01:
            improved = True
            best_margin_sat = val_metrics['margin_satisfaction']
        elif val_metrics['val_loss'] < best_val_loss:
            improved = True
            best_val_loss = val_metrics['val_loss']

        if improved:
            checkpoint_path = output_dir / "ebmo_improved_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_metrics['val_loss'],
                'val_accuracy': val_metrics['val_accuracy'],
                'margin_satisfaction': val_metrics['margin_satisfaction'],
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")

        # Periodic checkpoint
        if (epoch + 1) % 25 == 0:
            checkpoint_path = output_dir / f"ebmo_improved_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
            }, checkpoint_path)

    logger.info("Training complete!")
    logger.info(f"Best margin satisfaction: {best_margin_sat:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
