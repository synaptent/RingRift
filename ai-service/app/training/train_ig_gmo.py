"""Training script for IG-GMO (Information-Gain Gradient Move Optimization).

IG-GMO uses a Graph Neural Network encoder and MI-based exploration.

Usage:
    python -m app.training.train_ig_gmo --data-path data/training/gmo_square8_2p.jsonl

Architecture:
- GNNStateEncoder (128-dim, 3 GAT layers, 4 heads)
- MoveEncoder (128-dim)
- SoftLegalityPredictor (optional, for soft constraint learning)
- GMOValueNetWithUncertainty (256-dim hidden)
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..ai.ig_gmo import (
    IGGMOConfig,
    GNNStateEncoder,
    SoftLegalityPredictor,
)
from ..ai.gmo_ai import GMOValueNetWithUncertainty, MoveEncoder, nll_loss_with_uncertainty
from ..models import GameState, Move

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class IGGMODataset(Dataset):
    """Dataset for IG-GMO training with GNN state features."""

    def __init__(
        self,
        data_path: Path,
        state_encoder: GNNStateEncoder,
        move_encoder: MoveEncoder,
        max_samples: int | None = None,
    ):
        self.data_path = data_path
        self.state_encoder = state_encoder
        self.move_encoder = move_encoder
        self.samples: list[tuple[torch.Tensor, torch.Tensor, float, bool]] = []

        self._load_data(max_samples)
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def _load_data(self, max_samples: int | None) -> None:
        """Load game data from JSONL file."""
        games_loaded = 0
        device = next(self.state_encoder.parameters()).device

        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    continue

                winner = game.get("winner")
                if winner is None:
                    continue

                moves = game.get("moves", [])
                if not moves:
                    continue

                initial_state_dict = game.get("initial_state")
                if not initial_state_dict:
                    continue

                # Reconstruct game states
                try:
                    game_state = GameState.model_validate(initial_state_dict)
                except Exception:
                    continue

                from ..game_engine import GameEngine

                for i, move_dict in enumerate(moves):
                    try:
                        move = Move.model_validate(move_dict)
                        player = move.player

                        # Outcome: +1 if player won, -1 if lost
                        if winner == player:
                            outcome = 1.0
                        elif winner == 0:  # Draw
                            outcome = 0.0
                        else:
                            outcome = -1.0

                        # Temporal discount
                        move_progress = (i + 1) / len(moves)
                        discount = 0.5 + 0.5 * move_progress
                        outcome *= discount

                        # Check legality for soft legality training
                        legal_moves = GameEngine.get_valid_moves(game_state, player)
                        is_legal = move in legal_moves

                        # Extract GNN node features
                        with torch.no_grad():
                            node_features = self.state_encoder._extract_node_features(
                                game_state
                            )
                            move_embed = self.move_encoder.encode_move(move)

                        self.samples.append((
                            node_features.cpu(),
                            move_embed.cpu(),
                            outcome,
                            is_legal,
                        ))

                        if max_samples and len(self.samples) >= max_samples:
                            return

                        # Apply move
                        game_state = GameEngine.apply_move(game_state, move)

                    except Exception:
                        break

                games_loaded += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_feat, move_embed, outcome, is_legal = self.samples[idx]
        return (
            node_feat,
            move_embed,
            torch.tensor(outcome, dtype=torch.float32),
            torch.tensor(float(is_legal), dtype=torch.float32),
        )


def collate_fn(batch):
    nodes, moves, outcomes, legals = zip(*batch)
    return (
        torch.stack(nodes),
        torch.stack(moves),
        torch.stack(outcomes),
        torch.stack(legals),
    )


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    state_encoder: GNNStateEncoder,
    move_encoder: MoveEncoder,
    value_net: GMOValueNetWithUncertainty,
    legality_net: SoftLegalityPredictor | None,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    legality_weight: float = 0.5,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        (value_loss, legality_loss) tuple
    """
    state_encoder.train()
    move_encoder.train()
    value_net.train()
    if legality_net:
        legality_net.train()

    total_value_loss = 0.0
    total_legality_loss = 0.0
    num_batches = 0

    for node_features, move_embeds, outcomes, legals in tqdm(dataloader, desc="Training"):
        node_features = node_features.to(device)
        move_embeds = move_embeds.to(device)
        outcomes = outcomes.to(device)
        legals = legals.to(device)

        optimizer.zero_grad()

        # Process through GNN encoder (batched)
        batch_size = node_features.size(0)
        adj = state_encoder.adj_template.unsqueeze(0).expand(batch_size, -1, -1)

        # Initial projection
        x = state_encoder.input_proj(node_features)

        # GNN layers
        for gnn_layer, layer_norm in zip(
            state_encoder.gnn_layers, state_encoder.layer_norms, strict=False
        ):
            x_new = gnn_layer(x, adj)
            x = layer_norm(x + x_new)

        # Global pooling
        x = x.mean(dim=1)  # (batch, hidden_dim)
        state_embeds = state_encoder.global_pool(x)

        # Forward through value network
        pred_values, pred_log_vars = value_net(state_embeds, move_embeds)

        # Value loss (NLL with uncertainty)
        value_loss = nll_loss_with_uncertainty(pred_values, pred_log_vars, outcomes)

        # Legality loss (binary cross-entropy)
        if legality_net is not None:
            pred_legality = legality_net(state_embeds, move_embeds)
            legality_loss = nn.functional.binary_cross_entropy(
                pred_legality, legals
            )
            total_loss = value_loss + legality_weight * legality_loss
            total_legality_loss += legality_loss.item()
        else:
            total_loss = value_loss
            legality_loss = torch.tensor(0.0)

        # Backward with gradient clipping
        total_loss.backward()
        params = (
            list(state_encoder.parameters()) +
            list(move_encoder.parameters()) +
            list(value_net.parameters())
        )
        if legality_net:
            params += list(legality_net.parameters())

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        total_value_loss += value_loss.item()
        num_batches += 1

    return (
        total_value_loss / max(num_batches, 1),
        total_legality_loss / max(num_batches, 1),
    )


def evaluate_epoch(
    state_encoder: GNNStateEncoder,
    move_encoder: MoveEncoder,
    value_net: GMOValueNetWithUncertainty,
    legality_net: SoftLegalityPredictor | None,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate on validation set.

    Returns:
        (value_loss, legality_acc, value_acc) tuple
    """
    state_encoder.eval()
    move_encoder.eval()
    value_net.eval()
    if legality_net:
        legality_net.eval()

    total_value_loss = 0.0
    correct_value = 0
    correct_legality = 0
    total = 0

    with torch.no_grad():
        for node_features, move_embeds, outcomes, legals in dataloader:
            node_features = node_features.to(device)
            move_embeds = move_embeds.to(device)
            outcomes = outcomes.to(device)
            legals = legals.to(device)

            # Process through GNN encoder
            batch_size = node_features.size(0)
            adj = state_encoder.adj_template.unsqueeze(0).expand(batch_size, -1, -1)

            x = state_encoder.input_proj(node_features)
            for gnn_layer, layer_norm in zip(
                state_encoder.gnn_layers, state_encoder.layer_norms, strict=False
            ):
                x_new = gnn_layer(x, adj)
                x = layer_norm(x + x_new)

            x = x.mean(dim=1)
            state_embeds = state_encoder.global_pool(x)

            pred_values, pred_log_vars = value_net(state_embeds, move_embeds)
            value_loss = nll_loss_with_uncertainty(pred_values, pred_log_vars, outcomes)

            total_value_loss += value_loss.item() * len(outcomes)
            total += len(outcomes)

            # Value accuracy: correct sign prediction
            pred_signs = (pred_values > 0).float()
            true_signs = (outcomes > 0).float()
            correct_value += (pred_signs == true_signs).sum().item()

            # Legality accuracy
            if legality_net is not None:
                pred_legality = legality_net(state_embeds, move_embeds)
                pred_legal = (pred_legality > 0.5).float()
                correct_legality += (pred_legal == legals).sum().item()

    legality_acc = correct_legality / max(total, 1) if legality_net else 0.0
    return (
        total_value_loss / max(total, 1),
        legality_acc,
        correct_value / max(total, 1),
    )


# =============================================================================
# Main Training
# =============================================================================

def train_ig_gmo(
    data_path: Path,
    output_dir: Path,
    num_epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 0.0003,
    device: str = "cpu",
    use_soft_legality: bool = True,
    legality_weight: float = 0.5,
) -> None:
    """Train IG-GMO model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training IG-GMO on device: {device}")
    logger.info(f"Loading data from {data_path}")

    device = torch.device(device)

    # Create config
    config = IGGMOConfig(
        state_dim=128,
        move_dim=128,
        hidden_dim=256,
        gnn_layers=3,
        gnn_heads=4,
        use_soft_legality=use_soft_legality,
        device=str(device),
    )

    # Create networks
    state_encoder = GNNStateEncoder(
        output_dim=config.state_dim,
        hidden_dim=config.hidden_dim // 2,
        num_layers=config.gnn_layers,
        num_heads=config.gnn_heads,
        dropout=config.dropout_rate,
    ).to(device)

    move_encoder = MoveEncoder(
        embed_dim=config.move_dim,
    ).to(device)

    value_net = GMOValueNetWithUncertainty(
        state_dim=config.state_dim,
        move_dim=config.move_dim,
        hidden_dim=config.hidden_dim,
        dropout_rate=config.dropout_rate,
    ).to(device)

    legality_net = None
    if use_soft_legality:
        legality_net = SoftLegalityPredictor(
            state_dim=config.state_dim,
            move_dim=config.move_dim,
        ).to(device)
        logger.info("Using soft legality predictor")

    # Load dataset
    dataset = IGGMODataset(data_path, state_encoder, move_encoder)
    logger.info(f"Total samples: {len(dataset)}")

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Optimizer with regularization
    params = (
        list(state_encoder.parameters()) +
        list(move_encoder.parameters()) +
        list(value_net.parameters())
    )
    if legality_net:
        params += list(legality_net.parameters())

    optimizer = optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=0.02,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = 12

    for epoch in range(num_epochs):
        train_val_loss, train_leg_loss = train_epoch(
            state_encoder, move_encoder, value_net, legality_net,
            train_loader, optimizer, device, legality_weight
        )

        val_loss, val_leg_acc, val_val_acc = evaluate_epoch(
            state_encoder, move_encoder, value_net, legality_net,
            val_loader, device
        )

        log_msg = (
            f"Epoch {epoch+1}/{num_epochs}: "
            f"train_val_loss={train_val_loss:.4f}, "
            f"train_leg_loss={train_leg_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_val_acc:.2%}"
        )
        if legality_net:
            log_msg += f", leg_acc={val_leg_acc:.2%}"
        logger.info(log_msg)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = output_dir / "ig_gmo_best.pt"
            checkpoint = {
                "state_encoder": state_encoder.state_dict(),
                "move_encoder": move_encoder.state_dict(),
                "value_net": value_net.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
            }
            if legality_net:
                checkpoint["legality_net"] = legality_net.state_dict()
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {early_stopping_patience} epochs)"
                )
                break

        scheduler.step()


# =============================================================================
# CLI
# =============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    parser = argparse.ArgumentParser(description="Train IG-GMO AI")
    parser.add_argument(
        "--data-path", type=Path, required=True,
        help="Path to training data (JSONL)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/ig_gmo"),
        help="Output directory for checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-legality", action="store_true",
                        help="Disable soft legality predictor")
    parser.add_argument("--legality-weight", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    train_ig_gmo(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        use_soft_legality=not args.no_legality,
        legality_weight=args.legality_weight,
    )


if __name__ == "__main__":
    main()
