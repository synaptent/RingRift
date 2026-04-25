#!/usr/bin/env python3
"""Train GNN policy network for RingRift.

This script trains a Graph Neural Network policy/value network using
the same training data format as train_cnn_policy.py but with graph
representation.

Key features:
- Converts spatial features to graph format on-the-fly
- Outcome-weighted policy loss (learns from winners only)
- Supports curriculum learning across board sizes
- Compatible with PyTorch Geometric batching

Usage:
    python -m app.training.train_gnn_policy \
        --data-path data/training/hex8_2p.npz \
        --board-type hex8 \
        --epochs 30
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from app.utils.numpy_utils import safe_load_npz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check PyTorch Geometric
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.error("PyTorch Geometric required. Install with: pip install torch-geometric")

from app.ai.neural_net.gnn_policy import GNNPolicyNet


class GraphPolicyDataset(Dataset):
    """Dataset that converts spatial features to graphs.

    Converts the CNN-format training data (C, H, W) to graph format
    for GNN training.
    """

    def __init__(
        self,
        npz_path: str,
        board_type: str = "square8",
        action_space_size: int = 6158,
    ):
        data = safe_load_npz(npz_path)

        self.features = data["features"]  # (N, C, H, W)
        self.globals = data["globals"]    # (N, G)
        self.values = data["values"]      # (N,)
        self.board_type = board_type

        # Infer board size
        self.board_size = self.features.shape[-1]
        self.is_hex = board_type in ("hex8", "hexagonal")

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

    def _features_to_graph(self, features: np.ndarray) -> Data:
        """Convert spatial features (C, H, W) to graph.

        Uses the first 14 channels (current frame) to extract node features.
        """
        # Take first 14 channels (current state, not history)
        current_features = features[:14]  # (14, H, W)

        if self.is_hex:
            return self._hex_features_to_graph(current_features)
        else:
            return self._square_features_to_graph(current_features)

    def _square_features_to_graph(self, features: np.ndarray) -> Data:
        """Convert square board features to graph."""
        H, W = features.shape[1], features.shape[2]
        num_nodes = H * W

        # Node features: flatten spatial dims, transpose to (nodes, features)
        node_features = features.reshape(14, -1).T  # (H*W, 14)

        # Pad to 32 features
        node_features = np.concatenate([
            node_features,
            np.zeros((num_nodes, 32 - 14), dtype=np.float32)
        ], axis=1)

        # Build 4-connectivity edges
        edges = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y in range(H):
            for x in range(W):
                node_idx = y * W + x
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        neighbor_idx = ny * W + nx
                        edges.append([node_idx, neighbor_idx])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
        )

    def _hex_features_to_graph(self, features: np.ndarray) -> Data:
        """Convert hex board features to graph.

        Only includes valid hex cells (uses channel 12 as validity mask).
        """
        H, W = features.shape[1], features.shape[2]

        # Channel 12 is the valid board mask
        valid_mask = features[12] > 0.5  # (H, W)

        # Build position-to-index mapping for valid cells
        pos_to_idx = {}
        node_features_list = []

        for y in range(H):
            for x in range(W):
                if valid_mask[y, x]:
                    idx = len(pos_to_idx)
                    pos_to_idx[(x, y)] = idx
                    # Extract node features for this cell
                    node_feat = features[:, y, x]  # (14,)
                    # Pad to 32
                    node_feat = np.concatenate([
                        node_feat,
                        np.zeros(32 - 14, dtype=np.float32)
                    ])
                    node_features_list.append(node_feat)

        if not node_features_list:
            # Fallback for empty board
            return Data(
                x=torch.zeros(1, 32),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
            )

        node_features = np.stack(node_features_list)

        # Build 6-connectivity edges for hex
        # Axial coordinate neighbors
        hex_dirs = [
            (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)
        ]

        edges = []
        for (x, y), node_idx in pos_to_idx.items():
            for dx, dy in hex_dirs:
                neighbor_pos = (x + dx, y + dy)
                if neighbor_pos in pos_to_idx:
                    neighbor_idx = pos_to_idx[neighbor_pos]
                    edges.append([node_idx, neighbor_idx])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long)

        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        graph = self._features_to_graph(self.features[idx])
        return (
            graph,
            torch.tensor(self.globals[idx], dtype=torch.float32),
            torch.tensor(self.policy_targets[idx], dtype=torch.float32),
            torch.tensor(self.values[idx], dtype=torch.float32),
        )


def collate_graphs(batch):
    """Collate function for graph batches."""
    graphs, globals_, policies, values = zip(*batch)

    # Batch graphs using PyG
    batched_graph = Batch.from_data_list(graphs)

    return (
        batched_graph,
        torch.stack(globals_),
        torch.stack(policies),
        torch.stack(values),
    )


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch with outcome-weighted policy loss."""
    model.train()
    total_policy_loss = 0
    total_value_loss = 0
    total_correct = 0
    total_samples = 0
    total_weight = 0

    for batched_graph, globals_, policy_target, value_target in loader:
        # Move to device
        batched_graph = batched_graph.to(device)
        globals_ = globals_.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)

        optimizer.zero_grad()

        # Forward pass
        policy_logits, value_pred = model(
            x=batched_graph.x,
            edge_index=batched_graph.edge_index,
            edge_attr=getattr(batched_graph, 'edge_attr', None),
            batch=batched_graph.batch,
            globals_=globals_,
        )

        # Outcome-weighted policy loss (only learn from winners)
        policy_weights = (value_target + 1.0) / 2.0  # [-1,1] -> [0,1]
        policy_weights = policy_weights.clamp(min=0.0, max=1.0)

        per_sample_ce = -(policy_target * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1)
        weighted_ce = per_sample_ce * policy_weights
        policy_loss = weighted_ce.sum() / (policy_weights.sum() + 1e-8)

        # Value loss (MSE for current player)
        # GNN outputs multi-player values, we use first column for 2-player
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
        for batched_graph, globals_, policy_target, value_target in loader:
            batched_graph = batched_graph.to(device)
            globals_ = globals_.to(device)
            policy_target = policy_target.to(device)

            policy_logits, _ = model(
                x=batched_graph.x,
                edge_index=batched_graph.edge_index,
                batch=batched_graph.batch,
                globals_=globals_,
            )

            pred_actions = policy_logits.argmax(dim=-1)
            target_actions = policy_target.argmax(dim=-1)
            total_correct += (pred_actions == target_actions).sum().item()
            total_samples += len(value_target)

    return total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to NPZ training data")
    parser.add_argument("--board-type", default="square8", choices=["square8", "square19", "hex8", "hexagonal"])
    parser.add_argument("--output-dir", default="models/gnn_policy")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)  # Smaller for graphs
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--conv-type", default="sage", choices=["sage", "gat"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not HAS_PYG:
        raise ImportError("PyTorch Geometric required")

    # Load data
    logger.info(f"Loading data from {args.data_path}")

    data = safe_load_npz(args.data_path)
    board_size = int(data["board_size"])

    # Infer action space size
    max_idx = 0
    for idx_arr in data["policy_indices"]:
        if len(idx_arr) > 0:
            max_idx = max(max_idx, max(idx_arr))
    action_space_size = max_idx + 1
    logger.info(f"Inferred action space size: {action_space_size}")

    dataset = GraphPolicyDataset(
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

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=4,
    )
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=4,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = GNNPolicyNet(
        node_feature_dim=32,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        conv_type=args.conv_type,
        action_space_size=action_space_size,
        global_feature_dim=data["globals"].shape[1],
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
                "board_type": args.board_type,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "conv_type": args.conv_type,
            }, output_dir / "gnn_policy_best.pt")
            logger.info(f"  -> New best model saved (val_acc={val_acc:.4f})")

    logger.info(f"Training complete. Best val_acc: {best_val_acc:.4f}")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_acc": val_acc,
        "action_space_size": action_space_size,
        "board_type": args.board_type,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "conv_type": args.conv_type,
    }, output_dir / "gnn_policy_final.pt")


if __name__ == "__main__":
    main()
