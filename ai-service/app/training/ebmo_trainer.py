"""EBMO Training Loop.

Training infrastructure for Energy-Based Move Optimization networks
using contrastive learning on game data.

Features:
- Contrastive energy loss
- Outcome-weighted loss
- Learning rate scheduling
- Checkpointing
- Validation
- Integration with existing training infrastructure

Usage:
    from app.training.ebmo_trainer import EBMOTrainer, EBMOTrainingConfig

    config = EBMOTrainingConfig(
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
    )

    trainer = EBMOTrainer(config)
    trainer.train(
        train_data_dir="data/train",
        val_data_dir="data/val",
        output_dir="models/ebmo",
    )
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..ai.ebmo_network import (
    EBMOConfig,
    EBMONetwork,
    combined_ebmo_loss,
    contrastive_energy_loss,
    save_ebmo_model,
    load_ebmo_model,
)
from .ebmo_dataset import (
    EBMODataset,
    EBMOStreamingDataset,
    create_ebmo_dataloader,
    generate_synthetic_ebmo_data,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EBMOTrainingConfig:
    """Configuration for EBMO training."""

    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    # Loss weights
    contrastive_temperature: float = 0.1
    outcome_weight: float = 0.5

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau", "none"
    lr_warmup_epochs: int = 5
    lr_decay_factor: float = 0.1
    lr_decay_epochs: List[int] = field(default_factory=lambda: [60, 80])

    # Data
    num_negatives: int = 15
    hard_negative_ratio: float = 0.3
    num_workers: int = 4
    preload_data: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints/ebmo"
    checkpoint_every: int = 10
    keep_last_n_checkpoints: int = 5

    # Validation
    val_every: int = 5
    early_stopping_patience: int = 20

    # Hardware
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    mixed_precision: bool = True

    # Network config (optional override)
    network_config: Optional[EBMOConfig] = None

    # Logging
    log_every: int = 50
    wandb_project: Optional[str] = None
    experiment_name: str = "ebmo_training"


# =============================================================================
# Training Metrics
# =============================================================================


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    epoch: int = 0
    train_loss: float = 0.0
    train_contrastive_loss: float = 0.0
    train_outcome_loss: float = 0.0
    val_loss: float = 0.0
    val_contrastive_loss: float = 0.0
    learning_rate: float = 0.0
    samples_per_second: float = 0.0
    epoch_time: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "epoch": self.epoch,
            "train/loss": self.train_loss,
            "train/contrastive": self.train_contrastive_loss,
            "train/outcome": self.train_outcome_loss,
            "val/loss": self.val_loss,
            "val/contrastive": self.val_contrastive_loss,
            "lr": self.learning_rate,
            "throughput": self.samples_per_second,
            "epoch_time": self.epoch_time,
        }


# =============================================================================
# EBMO Trainer
# =============================================================================


class EBMOTrainer:
    """Trainer for EBMO networks.

    Handles:
    - Model initialization
    - Training loop with contrastive loss
    - Learning rate scheduling
    - Checkpointing
    - Validation
    - Early stopping
    """

    def __init__(self, config: Optional[EBMOTrainingConfig] = None):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or EBMOTrainingConfig()

        # Device selection
        self.device = self._select_device()
        logger.info(f"[EBMOTrainer] Using device: {self.device}")

        # Model
        network_config = self.config.network_config or EBMOConfig()
        self.model = EBMONetwork(network_config)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if (
            self.config.mixed_precision and self.device.type == "cuda"
        ) else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history: List[TrainingMetrics] = []

    def _select_device(self) -> torch.device:
        """Select training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.lr_warmup_epochs,
            )
        elif self.config.lr_scheduler == "step":
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.lr_decay_epochs,
                gamma=self.config.lr_decay_factor,
            )
        elif self.config.lr_scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_decay_factor,
                patience=5,
            )
        return None

    def _warmup_lr(self, epoch: int) -> None:
        """Apply learning rate warmup."""
        if epoch < self.config.lr_warmup_epochs:
            warmup_factor = (epoch + 1) / self.config.lr_warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * warmup_factor

    def train(
        self,
        train_data_dir: str = "data/train",
        val_data_dir: Optional[str] = None,
        output_dir: str = "models/ebmo",
        train_data_paths: Optional[List[str]] = None,
        val_data_paths: Optional[List[str]] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run training loop.

        Args:
            train_data_dir: Directory with training data
            val_data_dir: Directory with validation data
            output_dir: Output directory for models
            train_data_paths: Explicit paths to training files
            val_data_paths: Explicit paths to validation files
            resume_from: Checkpoint to resume from

        Returns:
            Training results dict
        """
        # Setup output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(self.config.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)

        # Create data loaders
        train_loader = self._create_dataloader(
            train_data_dir,
            train_data_paths,
            shuffle=True,
        )
        val_loader = None
        if val_data_dir or val_data_paths:
            val_loader = self._create_dataloader(
                val_data_dir,
                val_data_paths,
                shuffle=False,
            )

        logger.info(f"[EBMOTrainer] Starting training for {self.config.epochs} epochs")
        logger.info(f"[EBMOTrainer] Training samples: ~{len(train_loader) * self.config.batch_size}")

        start_time = time.time()

        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch

                # Warmup LR
                self._warmup_lr(epoch)

                # Train epoch
                train_metrics = self._train_epoch(train_loader, epoch)

                # Validation
                val_metrics = {}
                if val_loader and (epoch + 1) % self.config.val_every == 0:
                    val_metrics = self._validate(val_loader)
                    train_metrics.val_loss = val_metrics.get("loss", 0.0)
                    train_metrics.val_contrastive_loss = val_metrics.get("contrastive", 0.0)

                # Update learning rate
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(train_metrics.val_loss or train_metrics.train_loss)
                    elif epoch >= self.config.lr_warmup_epochs:
                        self.scheduler.step()

                train_metrics.learning_rate = self.optimizer.param_groups[0]['lr']

                # Record history
                self.history.append(train_metrics)

                # Checkpointing
                if (epoch + 1) % self.config.checkpoint_every == 0:
                    self._save_checkpoint(checkpoint_path / f"checkpoint_epoch_{epoch+1}.pt")

                # Best model
                current_loss = train_metrics.val_loss or train_metrics.train_loss
                if current_loss < self.best_val_loss:
                    self.best_val_loss = current_loss
                    self.patience_counter = 0
                    save_ebmo_model(
                        self.model,
                        str(output_path / "best_model.pt"),
                        self.optimizer,
                        epoch,
                        best_loss=self.best_val_loss,
                    )
                else:
                    self.patience_counter += 1

                # Log progress
                self._log_epoch(train_metrics)

                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"[EBMOTrainer] Early stopping at epoch {epoch+1}")
                    break

        except KeyboardInterrupt:
            logger.info("[EBMOTrainer] Training interrupted by user")

        # Save final model
        total_time = time.time() - start_time
        save_ebmo_model(
            self.model,
            str(output_path / "final_model.pt"),
            self.optimizer,
            self.current_epoch,
            training_time=total_time,
        )

        # Cleanup old checkpoints
        self._cleanup_checkpoints(checkpoint_path)

        return {
            "epochs_completed": self.current_epoch + 1,
            "best_val_loss": self.best_val_loss,
            "total_time_seconds": total_time,
            "history": [m.to_dict() for m in self.history],
        }

    def _create_dataloader(
        self,
        data_dir: Optional[str],
        data_paths: Optional[List[str]],
        shuffle: bool = True,
    ) -> DataLoader:
        """Create data loader."""
        paths = data_paths
        if paths is None and data_dir:
            paths = [f"{data_dir}/*.npz"]

        # If no real data, use synthetic
        if not paths or not any(Path(p.replace("*", "")).parent.exists() for p in (paths or [])):
            logger.warning("[EBMOTrainer] No data found, using synthetic data")
            samples = generate_synthetic_ebmo_data(
                num_samples=1000,
                board_size=self.model.config.board_size,
                num_negatives=self.config.num_negatives,
            )

            class SyntheticDataset(torch.utils.data.Dataset):
                def __init__(self, samples):
                    self.samples = samples

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    s = self.samples[idx]
                    return (
                        torch.from_numpy(s.board_features),
                        torch.from_numpy(s.global_features),
                        torch.from_numpy(s.positive_action),
                        torch.from_numpy(s.negative_actions),
                        torch.tensor(s.outcome),
                    )

            return DataLoader(
                SyntheticDataset(samples),
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=0,
            )

        return create_ebmo_dataloader(
            data_paths=paths,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=shuffle,
            num_negatives=self.config.num_negatives,
            board_size=self.model.config.board_size,
            preload=self.config.preload_data,
        )

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> TrainingMetrics:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Training metrics for this epoch
        """
        self.model.train()

        total_loss = 0.0
        total_contrastive = 0.0
        total_outcome = 0.0
        num_batches = 0
        num_samples = 0

        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            board_feat, global_feat, pos_action, neg_actions, outcomes = [
                x.to(self.device) for x in batch
            ]

            batch_size = board_feat.shape[0]
            num_samples += batch_size

            # Forward pass
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # Encode state
                state_embed = self.model.encode_state(board_feat, global_feat)

                # Encode positive action
                pos_embed = self.model.encode_action(pos_action)

                # Encode negative actions (reshape batch)
                num_neg = neg_actions.shape[1]
                neg_flat = neg_actions.view(-1, neg_actions.shape[-1])
                neg_embed = self.model.encode_action(neg_flat)
                neg_embed = neg_embed.view(batch_size, num_neg, -1)

                # Compute energies
                pos_energy = self.model.compute_energy(state_embed, pos_embed)

                # Negative energies (expand state for each negative)
                state_expanded = state_embed.unsqueeze(1).expand(-1, num_neg, -1)
                neg_energies = torch.stack([
                    self.model.compute_energy(state_expanded[:, i], neg_embed[:, i])
                    for i in range(num_neg)
                ], dim=1)

                # Compute loss
                loss, loss_dict = combined_ebmo_loss(
                    pos_energy,
                    neg_energies,
                    outcomes,
                    self.config.contrastive_temperature,
                    self.config.outcome_weight,
                )

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
                self.optimizer.step()

            # Track metrics
            total_loss += loss_dict["total"]
            total_contrastive += loss_dict["contrastive"]
            total_outcome += loss_dict["outcome"]
            num_batches += 1
            self.global_step += 1

            # Logging
            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_loss / num_batches
                logger.debug(
                    f"[EBMOTrainer] Epoch {epoch+1}, Batch {batch_idx+1}, "
                    f"Loss: {avg_loss:.4f}"
                )

        epoch_time = time.time() - epoch_start

        return TrainingMetrics(
            epoch=epoch + 1,
            train_loss=total_loss / max(num_batches, 1),
            train_contrastive_loss=total_contrastive / max(num_batches, 1),
            train_outcome_loss=total_outcome / max(num_batches, 1),
            samples_per_second=num_samples / epoch_time,
            epoch_time=epoch_time,
        )

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation.

        Args:
            val_loader: Validation data loader

        Returns:
            Validation metrics dict
        """
        self.model.eval()

        total_loss = 0.0
        total_contrastive = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                board_feat, global_feat, pos_action, neg_actions, outcomes = [
                    x.to(self.device) for x in batch
                ]

                batch_size = board_feat.shape[0]

                # Forward pass
                state_embed = self.model.encode_state(board_feat, global_feat)
                pos_embed = self.model.encode_action(pos_action)

                num_neg = neg_actions.shape[1]
                neg_flat = neg_actions.view(-1, neg_actions.shape[-1])
                neg_embed = self.model.encode_action(neg_flat)
                neg_embed = neg_embed.view(batch_size, num_neg, -1)

                pos_energy = self.model.compute_energy(state_embed, pos_embed)

                state_expanded = state_embed.unsqueeze(1).expand(-1, num_neg, -1)
                neg_energies = torch.stack([
                    self.model.compute_energy(state_expanded[:, i], neg_embed[:, i])
                    for i in range(num_neg)
                ], dim=1)

                loss, loss_dict = combined_ebmo_loss(
                    pos_energy,
                    neg_energies,
                    outcomes,
                    self.config.contrastive_temperature,
                    self.config.outcome_weight,
                )

                total_loss += loss_dict["total"]
                total_contrastive += loss_dict["contrastive"]
                num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "contrastive": total_contrastive / max(num_batches, 1),
        }

    def _save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "model_config": self.model.config,
        }, path)
        logger.info(f"[EBMOTrainer] Saved checkpoint to {path}")

    def _load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(f"[EBMOTrainer] Resumed from {path} (epoch {self.current_epoch})")

    def _cleanup_checkpoints(self, checkpoint_dir: Path) -> None:
        """Remove old checkpoints, keeping only recent ones."""
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old_ckpt in checkpoints[self.config.keep_last_n_checkpoints:]:
            old_ckpt.unlink()
            logger.debug(f"[EBMOTrainer] Removed old checkpoint: {old_ckpt}")

    def _log_epoch(self, metrics: TrainingMetrics) -> None:
        """Log epoch summary."""
        logger.info(
            f"[EBMOTrainer] Epoch {metrics.epoch:3d} | "
            f"Train Loss: {metrics.train_loss:.4f} | "
            f"Val Loss: {metrics.val_loss:.4f} | "
            f"LR: {metrics.learning_rate:.6f} | "
            f"Time: {metrics.epoch_time:.1f}s | "
            f"Throughput: {metrics.samples_per_second:.0f} samples/s"
        )


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Command-line training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train EBMO network")
    parser.add_argument("--train-dir", default="data/train", help="Training data directory")
    parser.add_argument("--val-dir", default=None, help="Validation data directory")
    parser.add_argument("--output-dir", default="models/ebmo", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")

    args = parser.parse_args()

    config = EBMOTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    trainer = EBMOTrainer(config)

    if args.synthetic:
        # Use synthetic data for testing
        trainer.train(
            train_data_dir=None,
            output_dir=args.output_dir,
            resume_from=args.resume,
        )
    else:
        trainer.train(
            train_data_dir=args.train_dir,
            val_data_dir=args.val_dir,
            output_dir=args.output_dir,
            resume_from=args.resume,
        )


if __name__ == "__main__":
    main()


__all__ = [
    "EBMOTrainingConfig",
    "TrainingMetrics",
    "EBMOTrainer",
]
