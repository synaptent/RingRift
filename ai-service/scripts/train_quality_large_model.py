#!/usr/bin/env python3
"""
Train Large Model with Quality Data

Combines:
1. Larger model architecture (256 filters, 16 blocks) from hyperparameters.json
2. High-quality harvested training data
3. Extended training (100 epochs)
4. Lower learning rate for stability

Usage:
    python scripts/train_quality_large_model.py \
        --data data/training/harvested_square8_2p_quality.npz \
        --output models/nnue/square8_2p_large_quality.pt
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.nnue import RingRiftNNUE
from app.models import BoardType


def load_hyperparameters(config_key: str = "square8_2p") -> dict:
    """Load hyperparameters from config file."""
    hp_file = Path("config/hyperparameters.json")

    if not hp_file.exists():
        print(f"Warning: {hp_file} not found, using defaults")
        return {}

    with open(hp_file) as f:
        hp = json.load(f)

    defaults = hp.get("defaults", {})
    config = hp.get("configs", {}).get(config_key, {}).get("hyperparameters", {})

    # Merge defaults with config-specific overrides
    merged = {**defaults, **config}
    return merged


def load_npz_data(npz_path: Path) -> tuple:
    """Load training data from NPZ file."""
    print(f"Loading data from {npz_path}...")

    data = np.load(npz_path)
    print(f"NPZ keys: {list(data.keys())}")

    # Handle different NPZ formats
    if "features" in data:
        X = data["features"]
    elif "boards" in data:
        X = data["boards"]
    elif "states" in data:
        X = data["states"]
    else:
        raise ValueError(f"Unknown NPZ format, keys: {list(data.keys())}")

    if "values" in data:
        y = data["values"]
    elif "outcomes" in data:
        y = data["outcomes"]
    else:
        raise ValueError(f"No target values found, keys: {list(data.keys())}")

    print(f"Loaded {len(X)} samples")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hp: dict,
    device: torch.device,
    output_path: Path,
) -> dict:
    """Train the model with the given hyperparameters."""
    epochs = hp.get("epochs", 100)
    lr = hp.get("learning_rate", 0.0001)
    weight_decay = hp.get("weight_decay", 1e-5)
    patience = hp.get("early_stopping_patience", 20)
    warmup_epochs = hp.get("warmup_epochs", 5)

    print(f"\nTraining config:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Warmup epochs: {warmup_epochs}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_count = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    start_time = time.time()

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        scheduler.step()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_count = 0

            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "hyperparameters": hp,
            }, output_path)
        else:
            no_improve_count += 1

        # Progress
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Best: {best_val_loss:.4f} (ep {best_epoch+1}) | "
            f"ETA: {eta/60:.1f}m"
        )

        # Early stopping
        if no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    print(f"Model saved to: {output_path}")

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "total_time": total_time,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train large model with quality data")
    parser.add_argument("--data", required=True, help="Path to NPZ training data")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--config", default="square8_2p", help="Config key from hyperparameters.json")
    parser.add_argument("--board-type", default="square8", help="Board type")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split")
    parser.add_argument("--batch-size", type=int, help="Override batch size")

    args = parser.parse_args()

    # Load hyperparameters
    hp = load_hyperparameters(args.config)
    print(f"Loaded hyperparameters for {args.config}:")
    for k, v in hp.items():
        print(f"  {k}: {v}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    X, y = load_npz_data(Path(args.data))

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # Train/val split
    n_samples = len(X_tensor)
    n_val = int(n_samples * args.val_split)
    indices = torch.randperm(n_samples)

    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")

    # Data loaders
    batch_size = args.batch_size or hp.get("batch_size", 128)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Create model with large architecture
    model = RingRiftNNUE(
        board_type=BoardType(args.board_type),
        hidden_dim=hp.get("hidden_dim", 768),
        num_hidden_layers=hp.get("num_hidden_layers", 2),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Estimated size: {total_params * 4 / 1024 / 1024:.1f} MB")

    model = model.to(device)

    # Train
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = train_model(model, train_loader, val_loader, hp, device, output_path)

    # Save training info
    info_path = output_path.with_suffix(".info.json")
    info = {
        "config": args.config,
        "board_type": args.board_type,
        "data_file": args.data,
        "hyperparameters": hp,
        "training_result": {
            "best_val_loss": result["best_val_loss"],
            "best_epoch": result["best_epoch"],
            "total_epochs": result["total_epochs"],
            "total_time_seconds": result["total_time"],
        },
        "model_params": total_params,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "timestamp": datetime.now().isoformat(),
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nTraining info saved to: {info_path}")


if __name__ == "__main__":
    main()
