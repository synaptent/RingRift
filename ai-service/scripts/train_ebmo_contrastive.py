#!/usr/bin/env python3
"""Train EBMO using contrastive learning from game outcomes.

Trains EBMO to:
1. Assign low energy to winner's moves
2. Assign high energy to loser's moves
3. Learn from Heuristic's winning strategies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from app.ai.ebmo_network import EBMONetwork, EBMOConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("train_ebmo_contrastive")


class ContrastiveDataset(Dataset):
    """Dataset for contrastive EBMO training."""

    def __init__(self, data_path: str):
        data = np.load(data_path, allow_pickle=True)
        self.boards = torch.from_numpy(data['boards'])
        self.actions = torch.from_numpy(data['actions'])
        self.labels = torch.from_numpy(data['labels'])  # 1 = winner, 0 = loser

        # Optional: weight by AI type (learn more from Heuristic)
        if 'ai_types' in data:
            self.ai_types = torch.from_numpy(data['ai_types'])
        else:
            self.ai_types = None

        logger.info(f"Loaded {len(self.boards)} samples")
        logger.info(f"  Winner samples: {self.labels.sum().item():.0f}")
        logger.info(f"  Loser samples: {(1 - self.labels).sum().item():.0f}")

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        item = {
            'board': self.boards[idx],
            'action': self.actions[idx],
            'label': self.labels[idx],
        }
        if self.ai_types is not None:
            item['ai_type'] = self.ai_types[idx]
        return item


class ContrastiveEBMOTrainer:
    """Trainer for EBMO using contrastive outcome-based learning."""

    def __init__(
        self,
        model: EBMONetwork,
        device: torch.device,
        learning_rate: float = 1e-4,
        winner_energy_target: float = -1.0,
        loser_energy_target: float = 1.0,
        margin: float = 0.5,
    ):
        self.model = model.to(device)
        self.device = device
        self.winner_target = winner_energy_target
        self.loser_target = loser_energy_target
        self.margin = margin

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_energy_loss = 0.0
        total_margin_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            boards = batch['board'].to(self.device)
            actions = batch['action'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            # Encode state
            state_embed = self.model.encode_state(boards)

            # Encode action
            action_embed = self.model.encode_action(actions)

            # Compute energy
            energy = self.model.compute_energy(state_embed, action_embed).squeeze()

            # Target energy based on outcome
            target_energy = torch.where(
                labels > 0.5,
                torch.full_like(energy, self.winner_target),
                torch.full_like(energy, self.loser_target),
            )

            # MSE loss to target
            energy_loss = F.mse_loss(energy, target_energy)

            # Margin loss: winners should have lower energy than losers
            winner_mask = labels > 0.5
            loser_mask = labels < 0.5

            if winner_mask.any() and loser_mask.any():
                winner_energy = energy[winner_mask].mean()
                loser_energy = energy[loser_mask].mean()
                margin_loss = F.relu(winner_energy - loser_energy + self.margin)
            else:
                margin_loss = torch.tensor(0.0, device=self.device)

            loss = energy_loss + 0.5 * margin_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_energy_loss += energy_loss.item()
            total_margin_loss += margin_loss.item()
            num_batches += 1

        self.scheduler.step()

        return {
            'loss': total_loss / num_batches,
            'energy_loss': total_energy_loss / num_batches,
            'margin_loss': total_margin_loss / num_batches,
            'lr': self.scheduler.get_last_lr()[0],
        }

    def validate(self, dataloader: DataLoader) -> dict:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        winner_energies = []
        loser_energies = []

        with torch.no_grad():
            for batch in dataloader:
                boards = batch['board'].to(self.device)
                actions = batch['action'].to(self.device)
                labels = batch['label'].to(self.device)

                state_embed = self.model.encode_state(boards)
                action_embed = self.model.encode_action(actions)
                energy = self.model.compute_energy(state_embed, action_embed).squeeze()

                target_energy = torch.where(
                    labels > 0.5,
                    torch.full_like(energy, self.winner_target),
                    torch.full_like(energy, self.loser_target),
                )
                loss = F.mse_loss(energy, target_energy)
                total_loss += loss.item()

                winner_mask = labels > 0.5
                loser_mask = labels < 0.5

                if winner_mask.any():
                    winner_energies.extend(energy[winner_mask].tolist())
                if loser_mask.any():
                    loser_energies.extend(energy[loser_mask].tolist())

        num_batches = len(dataloader)
        avg_winner = np.mean(winner_energies) if winner_energies else 0.0
        avg_loser = np.mean(loser_energies) if loser_energies else 0.0
        separation = avg_loser - avg_winner

        return {
            'val_loss': total_loss / num_batches,
            'winner_energy': avg_winner,
            'loser_energy': avg_loser,
            'separation': separation,
        }


def main():
    parser = argparse.ArgumentParser(description="Train EBMO with contrastive learning")
    parser.add_argument("--data", type=str, required=True, help="Training data path")
    parser.add_argument("--model", type=str,
                        default="models/ebmo_56ch/ebmo_quality_best.pt",
                        help="Base model to fine-tune")
    parser.add_argument("--output", type=str,
                        default="models/ebmo_contrastive/ebmo_best.pt",
                        help="Output model path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load base model
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        if isinstance(cfg, dict):
            config = EBMOConfig(**{k: v for k, v in cfg.items() if hasattr(EBMOConfig, k)})
        else:
            config = cfg
    else:
        config = EBMOConfig()

    model = EBMONetwork(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logger.info(f"Loaded base model from {args.model}")

    # Dataset
    dataset = ContrastiveDataset(args.data)

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Trainer
    trainer = ContrastiveEBMOTrainer(model, device, learning_rate=args.lr)

    # Training loop
    best_separation = -float('inf')
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        logger.info(
            f"Epoch {epoch}/{args.epochs}: "
            f"loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}, "
            f"separation={val_metrics['separation']:.3f} "
            f"(W:{val_metrics['winner_energy']:.2f}, L:{val_metrics['loser_energy']:.2f})"
        )

        # Save best model by separation
        if val_metrics['separation'] > best_separation:
            best_separation = val_metrics['separation']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'separation': best_separation,
            }, args.output)
            logger.info(f"  Saved best model (separation={best_separation:.3f})")

    logger.info(f"Training complete. Best separation: {best_separation:.3f}")


if __name__ == "__main__":
    main()
