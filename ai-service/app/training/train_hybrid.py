#!/usr/bin/env python3
"""Train Hybrid CNN-GNN policy network for RingRift.

Combines CNN backbone for local pattern recognition with GNN
refinement for connectivity modeling.

Usage:
    python -m app.training.train_hybrid \
        --data-path data/training/hex8_2p_v6.npz \
        --board-type hex8 \
        --epochs 20
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.ai.neural_net.hybrid_cnn_gnn import HybridPolicyNet, create_hybrid_model


class HybridPolicyDataset(Dataset):
    """Dataset for hybrid CNN-GNN training."""

    def __init__(
        self,
        npz_path: str,
        board_type: str = "hex8",
        action_space_size: int = 6158,
    ):
        data = np.load(npz_path, allow_pickle=True)

        self.features = data["features"]  # (N, C, H, W)
        self.globals = data["globals"]    # (N, G)
        self.values = data["values"]      # (N,)
        self.board_type = board_type

        # Infer board size
        self.board_size = self.features.shape[-1]

        # Convert sparse policy to dense
        policy_indices = data["policy_indices"]
        policy_values = data["policy_values"]

        self.action_space_size = action_space_size
        self.policy_targets = []

        for idx, val in zip(policy_indices, policy_values):
            target = np.zeros(action_space_size, dtype=np.float32)
            idx_arr = np.array(idx).flatten().astype(np.int64) if hasattr(idx, '__len__') else np.array([idx], dtype=np.int64)
            val_arr = np.array(val).flatten().astype(np.float32) if hasattr(val, '__len__') else np.array([val], dtype=np.float32)
            if len(idx_arr) > 0 and idx_arr[0] >= 0:
                target[idx_arr] = val_arr
                if target.sum() > 0:
                    target = target / target.sum()
            self.policy_targets.append(target)

        logger.info(f"Loaded {len(self.features)} samples from {npz_path}")
        logger.info(f"Board type: {board_type}, size: {self.board_size}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.globals[idx], dtype=torch.float32),
            torch.tensor(self.policy_targets[idx], dtype=torch.float32),
            torch.tensor(self.values[idx], dtype=torch.float32),
        )


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch with outcome-weighted policy loss."""
    model.train()
    total_policy_loss = 0
    total_value_loss = 0
    total_correct = 0
    total_samples = 0
    total_weight = 0
    num_batches = len(loader)

    for batch_idx, (features, globals_, policy_target, value_target) in enumerate(loader):
        if batch_idx % 500 == 0:
            logger.info(f"  Batch {batch_idx}/{num_batches} ({100*batch_idx/num_batches:.1f}%)")
        features = features.to(device)
        globals_ = globals_.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)

        optimizer.zero_grad()

        # Forward pass
        policy_logits, value_pred = model(features, globals_)

        # Outcome-weighted policy loss (only learn from winners)
        policy_weights = (value_target + 1.0) / 2.0  # [-1,1] -> [0,1]
        policy_weights = policy_weights.clamp(min=0.0, max=1.0)

        per_sample_ce = -(policy_target * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1)
        weighted_ce = per_sample_ce * policy_weights
        policy_loss = weighted_ce.sum() / (policy_weights.sum() + 1e-8)

        # Value loss
        value_pred_scalar = value_pred[:, 0] if value_pred.dim() > 1 else value_pred
        value_loss = F.mse_loss(value_pred_scalar, value_target)

        loss = policy_loss + value_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_size = len(value_target)
        total_policy_loss += policy_loss.item() * batch_size
        total_value_loss += value_loss.item() * batch_size
        total_weight += policy_weights.sum().item()

        # Accuracy on weighted samples
        pred_actions = policy_logits.argmax(dim=-1)
        target_actions = policy_target.argmax(dim=-1)
        weighted_correct = ((pred_actions == target_actions).float() * policy_weights).sum()
        total_correct += weighted_correct.item()
        total_samples += batch_size

    return {
        "policy_loss": total_policy_loss / total_samples,
        "value_loss": total_value_loss / total_samples,
        "accuracy": total_correct / (total_weight + 1e-8),
    }


def validate(model, loader, device):
    """Validate on all samples (not weighted)."""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, globals_, policy_target, value_target in loader:
            features = features.to(device)
            globals_ = globals_.to(device)
            policy_target = policy_target.to(device)

            policy_logits, _ = model(features, globals_)

            pred_actions = policy_logits.argmax(dim=-1)
            target_actions = policy_target.argmax(dim=-1)
            total_correct += (pred_actions == target_actions).sum().item()
            total_samples += len(value_target)

    return total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to NPZ training data")
    parser.add_argument("--board-type", default="hex8", choices=["square8", "square19", "hex8", "hexagonal"])
    parser.add_argument("--output-dir", default="models/hybrid_hex8_2p")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--cnn-blocks", type=int, default=6)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0=main process)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for regularization (0.0-0.5)")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Early stopping patience (0=disabled)")
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data_path}")

    data = np.load(args.data_path, allow_pickle=True)
    board_size = int(data["board_size"])
    in_channels = data["features"].shape[1]

    # Infer action space size
    max_idx = 0
    for idx_arr in data["policy_indices"]:
        if len(idx_arr) > 0:
            max_idx = max(max_idx, max(idx_arr))
    action_space_size = max_idx + 1
    logger.info(f"Inferred action space size: {action_space_size}")

    dataset = HybridPolicyDataset(
        args.data_path,
        board_type=args.board_type,
        action_space_size=action_space_size,
    )

    # Split train/val
    n_val = min(len(dataset) // 10, 5000)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    is_hex = args.board_type in ("hex8", "hexagonal")
    model = HybridPolicyNet(
        in_channels=in_channels,
        global_features=data["globals"].shape[1],
        hidden_channels=args.hidden_channels,
        cnn_blocks=args.cnn_blocks,
        gnn_layers=args.gnn_layers,
        board_size=board_size,
        action_space_size=action_space_size,
        num_players=2,
        is_hex=is_hex,
        dropout=args.dropout,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, args.device)
        val_acc = validate(model, val_loader, args.device)

        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s): "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"val_acc={val_acc:.4f}, "
            f"policy_loss={train_metrics['policy_loss']:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "action_space_size": action_space_size,
                "board_type": args.board_type,
                "board_size": board_size,
                "hidden_channels": args.hidden_channels,
                "cnn_blocks": args.cnn_blocks,
                "gnn_layers": args.gnn_layers,
                "dropout": args.dropout,
            }, output_dir / "hybrid_policy_best.pt")
            logger.info(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                logger.info(f"Early stopping after {epochs_without_improvement} epochs without improvement")
                break

    logger.info(f"Training complete. Best val_acc: {best_val_acc:.4f}")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_acc": val_acc,
        "action_space_size": action_space_size,
        "board_type": args.board_type,
        "board_size": board_size,
        "hidden_channels": args.hidden_channels,
        "cnn_blocks": args.cnn_blocks,
        "gnn_layers": args.gnn_layers,
    }, output_dir / "hybrid_policy_final.pt")


if __name__ == "__main__":
    main()
