#!/usr/bin/env python3
"""Train a simple CNN policy network.

This is a direct policy prediction approach (vs GMO's indirect optimization).
Uses cross-entropy loss with softmax over all possible actions.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class CNNPolicyNet(nn.Module):
    """CNN policy network with residual blocks.

    Architecture:
    - Input projection: conv 3x3 -> hidden_channels
    - Residual tower: N residual blocks
    - Policy head: conv 1x1 -> flatten -> FC -> action_space
    - Value head: conv 1x1 -> flatten -> FC -> 1
    """

    def __init__(
        self,
        input_channels: int = 56,
        global_features: int = 20,
        hidden_channels: int = 128,
        num_blocks: int = 6,
        board_size: int = 8,
        action_space_size: int = 8192,
    ):
        super().__init__()
        self.board_size = board_size
        self.action_space_size = action_space_size

        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        # Global feature projection
        self.global_fc = nn.Sequential(
            nn.Linear(global_features, hidden_channels),
            nn.ReLU(),
        )

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(32 * board_size * board_size + hidden_channels, action_space_size)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(4 * board_size * board_size + hidden_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, features, globals_):
        # features: (B, C, H, W)
        # globals_: (B, G)

        x = self.input_conv(features)
        for block in self.res_blocks:
            x = block(x)

        # Global features
        g = self.global_fc(globals_)  # (B, hidden)

        # Policy head
        p = self.policy_conv(x)
        p = p.flatten(1)  # (B, 32*H*W)
        p = torch.cat([p, g], dim=1)
        policy_logits = self.policy_fc(p)  # (B, action_space)

        # Value head
        v = self.value_conv(x)
        v = v.flatten(1)
        v = torch.cat([v, g], dim=1)
        value = self.value_fc(v)  # (B, 1)

        return policy_logits, value.squeeze(-1)


class PolicyDataset(Dataset):
    """Dataset for policy training."""

    def __init__(self, npz_path: str, action_space_size: int = 8192):
        data = np.load(npz_path, allow_pickle=True)

        self.features = torch.from_numpy(data["features"]).float()
        self.globals = torch.from_numpy(data["globals"]).float()
        self.values = torch.from_numpy(data["values"]).float()

        # Convert sparse policy to dense
        policy_indices = data["policy_indices"]
        policy_values = data["policy_values"]

        self.action_space_size = action_space_size
        self.policy_targets = []

        for idx, val in zip(policy_indices, policy_values):
            target = np.zeros(action_space_size, dtype=np.float32)
            # Handle object arrays (policy_indices stores arrays inside object array)
            idx_arr = np.array(idx).flatten().astype(np.int64) if hasattr(idx, '__len__') else np.array([idx], dtype=np.int64)
            val_arr = np.array(val).flatten().astype(np.float32) if hasattr(val, '__len__') else np.array([val], dtype=np.float32)
            if len(idx_arr) > 0 and idx_arr[0] >= 0:
                target[idx_arr] = val_arr
                if target.sum() > 0:
                    target = target / target.sum()  # Normalize
            self.policy_targets.append(torch.from_numpy(target))

        logger.info(f"Loaded {len(self.features)} samples from {npz_path}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.globals[idx],
            self.policy_targets[idx],
            self.values[idx],
        )


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_policy_loss = 0
    total_value_loss = 0
    total_correct = 0
    total_samples = 0

    for features, globals_, policy_target, value_target in loader:
        features = features.to(device)
        globals_ = globals_.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)

        optimizer.zero_grad()

        policy_logits, value_pred = model(features, globals_)

        # Policy loss: cross-entropy with soft targets, WEIGHTED BY GAME OUTCOME
        # This is critical - without weighting, the model learns both winner AND loser
        # moves equally, resulting in inverted behavior (predicting loser moves).
        #
        # Weight scheme:
        #   - Winners (value > 0): weight = 1.0 (learn these moves)
        #   - Losers (value < 0): weight = 0.0 (don't learn these moves)
        #   - Draws (value == 0): weight = 0.5 (partial learning)
        policy_weights = (value_target + 1.0) / 2.0  # Maps [-1, 1] to [0, 1]
        policy_weights = policy_weights.clamp(min=0.0, max=1.0)

        # Per-sample cross-entropy
        per_sample_ce = -(policy_target * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1)

        # Weighted mean (only count samples with weight > 0)
        weighted_ce = per_sample_ce * policy_weights
        policy_loss = weighted_ce.sum() / (policy_weights.sum() + 1e-8)

        # Value loss: MSE
        value_loss = F.mse_loss(value_pred, value_target)

        loss = policy_loss + value_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_policy_loss += policy_loss.item() * features.size(0)
        total_value_loss += value_loss.item() * features.size(0)

        # Accuracy: top-1 match with target's argmax
        pred_actions = policy_logits.argmax(dim=-1)
        target_actions = policy_target.argmax(dim=-1)
        total_correct += (pred_actions == target_actions).sum().item()
        total_samples += features.size(0)

    return {
        "policy_loss": total_policy_loss / total_samples,
        "value_loss": total_value_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def validate(model, loader, device):
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
            total_samples += features.size(0)

    return total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to NPZ training data")
    parser.add_argument("--output-dir", default="models/cnn_policy", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--action-space-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data_path}")

    # Get board size and action space from data
    data = np.load(args.data_path, allow_pickle=True)
    board_size = int(data["board_size"])
    input_channels = data["features"].shape[1]
    global_features = data["globals"].shape[1]

    # Infer action space size from policy indices
    max_idx = 0
    for idx_arr in data["policy_indices"]:
        if len(idx_arr) > 0:
            max_idx = max(max_idx, max(idx_arr))
    action_space_size = max_idx + 1
    logger.info(f"Inferred action space size: {action_space_size}")

    dataset = PolicyDataset(args.data_path, action_space_size=action_space_size)

    # Split train/val
    n_val = min(len(dataset) // 10, 10000)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = CNNPolicyNet(
        input_channels=input_channels,
        global_features=global_features,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        board_size=board_size,
        action_space_size=action_space_size,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0

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
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "action_space_size": action_space_size,
                "board_size": board_size,
                "hidden_channels": args.hidden_channels,
                "num_blocks": args.num_blocks,
            }, output_dir / "cnn_policy_best.pt")
            logger.info(f"  -> New best model saved (val_acc={val_acc:.4f})")

    logger.info(f"Training complete. Best val_acc: {best_val_acc:.4f}")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_acc": val_acc,
        "action_space_size": action_space_size,
        "board_size": board_size,
        "hidden_channels": args.hidden_channels,
        "num_blocks": args.num_blocks,
    }, output_dir / "cnn_policy_final.pt")


if __name__ == "__main__":
    main()
