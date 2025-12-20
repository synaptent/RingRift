#!/usr/bin/env python3
"""Cross-board transfer learning experiment.

Tests whether pre-training on square8 helps hexagonal model training.
Freezes early convolutional layers and fine-tunes final layers.

Usage:
    # Analyze transfer potential
    python scripts/transfer_learning_experiment.py --source-model models/ringrift_best_sq8_2p.pth --target-board hexagonal --analyze-only

    # Run fine-tuning experiment
    python scripts/transfer_learning_experiment.py --source-model models/ringrift_best_sq8_2p.pth --target-board hexagonal --training-data data/training/hex_2p.npz --epochs 20
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import BoardType
from app.ai.neural_net import (
    get_policy_size_for_board,
    get_spatial_size_for_board,
    HexNeuralNet_v3,
    P_HEX,
    HEX_BOARD_SIZE,
)

logger = logging.getLogger(__name__)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def load_source_model(model_path: str) -> dict:
    """Load source model state dict."""
    checkpoint = torch.load(model_path, map_location="cpu")
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    else:
        return checkpoint


def get_transferable_layers(state_dict: dict) -> dict:
    """Extract layers that can be transferred across board types.

    Early convolutional layers (feature extractors) are typically transferable
    as they learn general patterns. Final layers need reinitialization.
    """
    transferable = {}
    skip_patterns = ["policy", "value_head", "final", "output", "fc"]

    for key, value in state_dict.items():
        # Skip board-specific layers
        should_skip = any(pattern in key.lower() for pattern in skip_patterns)
        if not should_skip:
            transferable[key] = value

    return transferable


def create_transfer_model(
    source_path: str,
    target_board: str,
    freeze_backbone: bool = True,
) -> dict:
    """Create a model for target board type using source weights.

    Args:
        source_path: Path to source model checkpoint
        target_board: Target board type (hexagonal, square19, etc.)
        freeze_backbone: Whether to freeze transferred layers

    Returns:
        Dict with transfer statistics and recommendations
    """
    source_state = load_source_model(source_path)
    transferable = get_transferable_layers(source_state)

    # Get board-specific dimensions
    board_type_map = {
        "hexagonal": BoardType.HEXAGONAL,
        "square19": BoardType.SQUARE19,
        "square8": BoardType.SQUARE8,
        "hex8": BoardType.HEX8,
    }
    target_board_type = board_type_map.get(target_board.lower())

    if target_board_type:
        target_policy_size = get_policy_size_for_board(target_board_type)
        target_spatial_size = get_spatial_size_for_board(target_board_type)
    else:
        target_policy_size = 0
        target_spatial_size = 0

    total_params = sum(p.numel() for p in source_state.values())
    transferred_params = sum(p.numel() for p in transferable.values())
    transfer_ratio = transferred_params / total_params if total_params > 0 else 0

    result = {
        "source_model": source_path,
        "target_board": target_board,
        "target_policy_size": target_policy_size,
        "target_spatial_size": target_spatial_size,
        "total_source_params": total_params,
        "transferred_params": transferred_params,
        "transfer_ratio": transfer_ratio,
        "transferable_layers": list(transferable.keys()),
        "num_transferable_layers": len(transferable),
        "freeze_backbone": freeze_backbone,
        "recommendation": "",
    }

    # Recommendation based on transfer ratio
    if transfer_ratio > 0.8:
        result["recommendation"] = "High transfer potential - most features are transferable"
    elif transfer_ratio > 0.5:
        result["recommendation"] = "Medium transfer potential - backbone can be reused"
    elif transfer_ratio > 0.2:
        result["recommendation"] = "Low transfer potential - only early layers transferable"
    else:
        result["recommendation"] = "Very low transfer potential - consider training from scratch"

    return result


class TransferDataset(Dataset):
    """Simple dataset for transfer learning from NPZ files."""

    def __init__(self, npz_path: str, max_samples: int = 50000):
        """Load training data from NPZ file.

        Args:
            npz_path: Path to NPZ file with features, globals, policy_indices, values
            max_samples: Maximum number of samples to load
        """
        self.npz_path = npz_path
        logger.info(f"Loading dataset from {npz_path}")

        with np.load(npz_path, mmap_mode="r", allow_pickle=True) as data:
            n_samples = min(len(data["features"]), max_samples)
            self.features = data["features"][:n_samples].copy()
            self.globals = data["globals"][:n_samples].copy()

            # Handle policy_indices - may be object array with nested lists
            raw_indices = data["policy_indices"][:n_samples]
            self.policy_indices = []
            for idx in raw_indices:
                if isinstance(idx, np.ndarray):
                    idx = idx.item() if idx.size == 1 else idx[0]
                if isinstance(idx, (list, np.ndarray)) and len(idx) > 0:
                    self.policy_indices.append(int(idx[0]))
                elif isinstance(idx, (int, np.integer)):
                    self.policy_indices.append(int(idx))
                else:
                    self.policy_indices.append(0)  # Fallback
            self.policy_indices = np.array(self.policy_indices, dtype=np.int64)

            # Use values_mp if available (multi-player values)
            if "values_mp" in data:
                self.values = data["values_mp"][:n_samples].copy()
            else:
                self.values = data["values"][:n_samples].copy()

        logger.info(f"Loaded {len(self.features)} samples")
        logger.info(f"  Feature shape: {self.features.shape}")
        logger.info(f"  Globals shape: {self.globals.shape}")
        logger.info(f"  Values shape: {self.values.shape}")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]).float(),
            torch.from_numpy(self.globals[idx]).float(),
            torch.tensor(self.policy_indices[idx], dtype=torch.long),
            torch.from_numpy(self.values[idx]).float(),
        )


def transfer_backbone_weights(
    source_state: dict,
    target_model: nn.Module,
    freeze_backbone: bool = True,
) -> tuple[int, int, list[str]]:
    """Transfer backbone weights from source to target model.

    Args:
        source_state: Source model state dict
        target_model: Target model to initialize
        freeze_backbone: Whether to freeze transferred layers

    Returns:
        Tuple of (transferred_count, skipped_count, transferred_names)
    """
    target_state = target_model.state_dict()
    transferred = []
    skipped = []

    # Layers that are transferable (same architecture across board types)
    transferable_prefixes = ["bn1", "res_blocks"]

    for key in source_state:
        # Check if this layer is transferable
        is_transferable = any(key.startswith(prefix) for prefix in transferable_prefixes)

        if is_transferable and key in target_state:
            source_shape = source_state[key].shape
            target_shape = target_state[key].shape

            if source_shape == target_shape:
                target_state[key] = source_state[key].clone()
                transferred.append(key)
            else:
                skipped.append(f"{key} (shape mismatch: {source_shape} vs {target_shape})")
        else:
            skipped.append(f"{key} (not transferable or not in target)")

    # Load transferred weights
    target_model.load_state_dict(target_state)

    # Freeze backbone if requested
    if freeze_backbone:
        for name, param in target_model.named_parameters():
            if any(name.startswith(prefix) for prefix in transferable_prefixes):
                param.requires_grad = False

    logger.info(f"Transferred {len(transferred)} layers, skipped {len(skipped)}")
    return len(transferred), len(skipped), transferred


def run_transfer_experiment(
    source_path: str,
    target_board: str,
    training_data: str,
    output_dir: str,
    epochs: int = 20,
    learning_rate: float = 0.0001,
    batch_size: int = 128,
    freeze_epochs: int = 5,
    num_players: int = 2,
) -> dict:
    """Run a transfer learning experiment with actual fine-tuning.

    Args:
        source_path: Path to source model checkpoint
        target_board: Target board type (hexagonal, square19, etc.)
        training_data: Path to training NPZ file
        output_dir: Directory to save fine-tuned model
        epochs: Total training epochs
        learning_rate: Base learning rate
        batch_size: Batch size for training
        freeze_epochs: Number of epochs to keep backbone frozen
        num_players: Number of players

    Returns:
        Dict with experiment results and metrics
    """
    # Analysis first
    analysis = create_transfer_model(source_path, target_board)
    logger.info(f"Transfer analysis: {analysis['transfer_ratio']:.1%} transferable")

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load source weights
    source_state = load_source_model(source_path)

    # Create target model
    if target_board.lower() == "hexagonal":
        # Infer input channels and num_players from training data
        with np.load(training_data, allow_pickle=True) as data:
            in_channels = data["features"].shape[1]  # e.g., 56 or 64
            # Infer num_players from values_mp shape if available
            if "values_mp" in data:
                actual_num_players = data["values_mp"].shape[1]
            else:
                actual_num_players = num_players

        logger.info(f"Creating HexNeuralNet_v3 with in_channels={in_channels}, num_players={actual_num_players}")
        target_model = HexNeuralNet_v3(
            in_channels=in_channels,  # Match training data
            global_features=20,
            num_res_blocks=12,
            num_filters=192,
            board_size=HEX_BOARD_SIZE,
            policy_size=P_HEX,
            num_players=actual_num_players,
        )
    else:
        raise ValueError(f"Unsupported target board: {target_board}")

    # Transfer backbone weights (freeze initially)
    transferred, skipped, transferred_names = transfer_backbone_weights(
        source_state, target_model, freeze_backbone=True
    )

    target_model = target_model.to(device)

    # Load training data
    if not training_data or not os.path.exists(training_data):
        logger.warning(f"Training data not found: {training_data}")
        return {
            "analysis": analysis,
            "experiment_status": "no_data",
            "note": "Training data not provided or not found",
            "transferred_layers": transferred,
            "skipped_layers": skipped,
        }

    dataset = TransferDataset(training_data)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Optimizer (only non-frozen params initially)
    trainable_params = [p for p in target_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)

    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "policy_loss": [], "value_loss": []}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = output_path / f"transfer_{target_board}_{timestamp}.pth"

    for epoch in range(epochs):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs:
            logger.info(f"Epoch {epoch}: Unfreezing backbone layers")
            for param in target_model.parameters():
                param.requires_grad = True
            # Rebuild optimizer with all params
            optimizer = optim.AdamW(target_model.parameters(), lr=learning_rate * 0.1, weight_decay=1e-4)

        # Training
        target_model.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_samples = 0

        for batch_idx, (features, globals_vec, policy_idx, values) in enumerate(train_loader):
            features = features.to(device)
            globals_vec = globals_vec.to(device)
            policy_idx = policy_idx.to(device)
            values = values.to(device)

            optimizer.zero_grad()
            pred_value, pred_policy = target_model(features, globals_vec)

            # Policy loss
            p_loss = policy_criterion(pred_policy, policy_idx)

            # Value loss (multi-player)
            if values.dim() == 1:
                # Single value per sample - use first player
                v_loss = value_criterion(pred_value[:, 0], values)
            else:
                v_loss = value_criterion(pred_value, values)

            loss = p_loss + v_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(target_model.parameters(), 1.0)
            optimizer.step()

            train_policy_loss += p_loss.item() * features.size(0)
            train_value_loss += v_loss.item() * features.size(0)
            train_samples += features.size(0)

        # Validation
        target_model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for features, globals_vec, policy_idx, values in val_loader:
                features = features.to(device)
                globals_vec = globals_vec.to(device)
                policy_idx = policy_idx.to(device)
                values = values.to(device)

                pred_value, pred_policy = target_model(features, globals_vec)
                p_loss = policy_criterion(pred_policy, policy_idx)
                if values.dim() == 1:
                    v_loss = value_criterion(pred_value[:, 0], values)
                else:
                    v_loss = value_criterion(pred_value, values)

                val_policy_loss += p_loss.item() * features.size(0)
                val_value_loss += v_loss.item() * features.size(0)
                val_samples += features.size(0)

        # Compute averages
        train_p = train_policy_loss / train_samples
        train_v = train_value_loss / train_samples
        val_p = val_policy_loss / val_samples
        val_v = val_value_loss / val_samples
        train_total = train_p + train_v
        val_total = val_p + val_v

        history["train_loss"].append(train_total)
        history["val_loss"].append(val_total)
        history["policy_loss"].append(val_p)
        history["value_loss"].append(val_v)

        frozen_str = " [frozen]" if epoch < freeze_epochs else " [unfrozen]"
        logger.info(
            f"Epoch {epoch + 1}/{epochs}{frozen_str}: "
            f"train={train_total:.4f} val={val_total:.4f} "
            f"(p={val_p:.4f}, v={val_v:.4f})"
        )

        # Save best model
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save({
                "state_dict": target_model.state_dict(),
                "epoch": epoch,
                "val_loss": val_total,
                "transfer_analysis": analysis,
                "transferred_layers": transferred_names,
            }, model_save_path)
            logger.info(f"  Saved best model (val_loss={val_total:.4f})")

    return {
        "analysis": analysis,
        "experiment_status": "completed",
        "epochs_trained": epochs,
        "best_val_loss": best_val_loss,
        "final_val_loss": val_total,
        "transferred_layers": transferred,
        "model_path": str(model_save_path),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Transfer learning experiment")
    parser.add_argument("--source-model", type=str, required=True,
                       help="Path to source model checkpoint")
    parser.add_argument("--target-board", type=str, default="hexagonal",
                       choices=["hexagonal", "square19", "square8"],
                       help="Target board type")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze transfer potential, don't train")
    parser.add_argument("--training-data", type=str,
                       help="Path to training NPZ file")
    parser.add_argument("--output-dir", type=str, default="models/transfer",
                       help="Output directory for transferred model")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                       help="Learning rate")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                       help="Number of epochs to keep backbone frozen")
    parser.add_argument("--num-players", type=int, default=2,
                       help="Number of players")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.analyze_only:
        result = create_transfer_model(args.source_model, args.target_board)
        print("\n=== Transfer Learning Analysis ===")
        print(f"Source model: {result['source_model']}")
        print(f"Target board: {result['target_board']}")
        print(f"Target policy size: {result['target_policy_size']}")
        print(f"Target spatial size: {result['target_spatial_size']}")
        print(f"Total source params: {result['total_source_params']:,}")
        print(f"Transferred params: {result['transferred_params']:,}")
        print(f"Transfer ratio: {result['transfer_ratio']:.1%}")
        print(f"Transferable layers: {result['num_transferable_layers']}")
        print(f"\nRecommendation: {result['recommendation']}")
        print(f"\nTransferable layer names:")
        for layer in result['transferable_layers'][:10]:
            print(f"  - {layer}")
        if len(result['transferable_layers']) > 10:
            print(f"  ... and {len(result['transferable_layers']) - 10} more")
    else:
        if not args.training_data:
            print("Error: --training-data is required for fine-tuning")
            sys.exit(1)

        result = run_transfer_experiment(
            source_path=args.source_model,
            target_board=args.target_board,
            training_data=args.training_data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            freeze_epochs=args.freeze_epochs,
            num_players=args.num_players,
        )

        print(f"\n=== Transfer Learning Experiment Results ===")
        print(f"Status: {result['experiment_status']}")
        if result['experiment_status'] == 'completed':
            print(f"Epochs trained: {result['epochs_trained']}")
            print(f"Best validation loss: {result['best_val_loss']:.4f}")
            print(f"Final validation loss: {result['final_val_loss']:.4f}")
            print(f"Transferred layers: {result['transferred_layers']}")
            print(f"Model saved to: {result['model_path']}")
        elif 'note' in result:
            print(f"Note: {result['note']}")


if __name__ == "__main__":
    main()
