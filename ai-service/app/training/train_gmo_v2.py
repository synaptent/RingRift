"""Training script for GMO v2 (Attention-based Gradient Move Optimization).

GMO v2 uses an attention-based state encoder for better long-range feature capture.

Usage:
    python -m app.training.train_gmo_v2 --data-path data/training/gmo_square8_2p.jsonl

Architecture differences from GMO v1:
- AttentionStateEncoder (256-dim, transformer) vs MLP encoder (128-dim)
- MoveEncoderV2 (256-dim) vs MoveEncoder (128-dim)
- Ensemble optimization support
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

from ..ai.gmo_v2 import (
    GMOv2AI,
    GMOv2Config,
    AttentionStateEncoder,
    MoveEncoderV2,
)
from ..ai.gmo_ai import GMOValueNetWithUncertainty, nll_loss_with_uncertainty
from ..models import GameState, Move

logger = logging.getLogger(__name__)


# =============================================================================
# GMO v2 Value Network (256-dim compatible)
# =============================================================================

class GMOv2ValueNet(nn.Module):
    """Value network for GMO v2 with larger hidden dimensions."""

    def __init__(
        self,
        state_dim: int = 256,
        move_dim: int = 256,
        hidden_dim: int = 512,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + move_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        self.log_var_head = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self, state_embed: torch.Tensor, move_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([state_embed, move_embed], dim=-1)
        h = self.net(combined)
        value = self.value_head(h).squeeze(-1)
        log_var = self.log_var_head(h).squeeze(-1)
        return value, log_var


# =============================================================================
# Dataset
# =============================================================================

class GMOv2Dataset(Dataset):
    """Dataset for GMO v2 training using AttentionStateEncoder features."""

    def __init__(
        self,
        data_path: Path,
        state_encoder: AttentionStateEncoder,
        move_encoder: MoveEncoderV2,
        max_samples: int | None = None,
    ):
        self.data_path = data_path
        self.state_encoder = state_encoder
        self.move_encoder = move_encoder
        self.samples: list[tuple[torch.Tensor, torch.Tensor, float]] = []

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

                        # Extract features using attention encoder
                        with torch.no_grad():
                            state_features = self.state_encoder._extract_board_features(
                                game_state
                            )
                            move_embed = self.move_encoder(move)

                        self.samples.append((
                            state_features.cpu(),
                            move_embed.cpu(),
                            outcome,
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_feat, move_embed, outcome = self.samples[idx]
        return state_feat, move_embed, torch.tensor(outcome, dtype=torch.float32)


def collate_fn(batch):
    states, moves, outcomes = zip(*batch)
    return (
        torch.stack(states),
        torch.stack(moves),
        torch.stack(outcomes),
    )


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    state_encoder: AttentionStateEncoder,
    move_encoder: MoveEncoderV2,
    value_net: GMOv2ValueNet,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    state_encoder.train()
    move_encoder.train()
    value_net.train()

    total_loss = 0.0
    num_batches = 0

    for state_features, move_embeds, outcomes in tqdm(dataloader, desc="Training"):
        state_features = state_features.to(device)
        move_embeds = move_embeds.to(device)
        outcomes = outcomes.to(device)

        optimizer.zero_grad()

        # Process through attention encoder
        batch_size = state_features.size(0)
        state_embeds_list = []
        for i in range(batch_size):
            # Input projection
            x = state_encoder.input_proj(state_features[i].unsqueeze(0))
            positions = torch.arange(
                state_encoder.num_positions, device=device
            )
            pos_embed = state_encoder.position_embed(positions).unsqueeze(0)
            x = x + pos_embed
            x = state_encoder.transformer(x)
            x = x.transpose(1, 2)
            x = state_encoder.global_pool(x).squeeze(-1)
            x = state_encoder.output_proj(x)
            state_embeds_list.append(x)

        state_embeds = torch.cat(state_embeds_list, dim=0)

        # Forward through value network
        pred_values, pred_log_vars = value_net(state_embeds, move_embeds)

        # Compute loss
        loss = nll_loss_with_uncertainty(pred_values, pred_log_vars, outcomes)

        # Backward with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(state_encoder.parameters()) +
            list(move_encoder.parameters()) +
            list(value_net.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_epoch(
    state_encoder: AttentionStateEncoder,
    move_encoder: MoveEncoderV2,
    value_net: GMOv2ValueNet,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on validation set."""
    state_encoder.eval()
    move_encoder.eval()
    value_net.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for state_features, move_embeds, outcomes in dataloader:
            state_features = state_features.to(device)
            move_embeds = move_embeds.to(device)
            outcomes = outcomes.to(device)

            # Process through attention encoder
            batch_size = state_features.size(0)
            state_embeds_list = []
            for i in range(batch_size):
                x = state_encoder.input_proj(state_features[i].unsqueeze(0))
                positions = torch.arange(
                    state_encoder.num_positions, device=device
                )
                pos_embed = state_encoder.position_embed(positions).unsqueeze(0)
                x = x + pos_embed
                x = state_encoder.transformer(x)
                x = x.transpose(1, 2)
                x = state_encoder.global_pool(x).squeeze(-1)
                x = state_encoder.output_proj(x)
                state_embeds_list.append(x)

            state_embeds = torch.cat(state_embeds_list, dim=0)

            pred_values, pred_log_vars = value_net(state_embeds, move_embeds)
            loss = nll_loss_with_uncertainty(pred_values, pred_log_vars, outcomes)

            total_loss += loss.item() * len(outcomes)
            total += len(outcomes)

            # Accuracy: correct sign prediction
            pred_signs = (pred_values > 0).float()
            true_signs = (outcomes > 0).float()
            correct += (pred_signs == true_signs).sum().item()

    return total_loss / max(total, 1), correct / max(total, 1)


# =============================================================================
# Main Training
# =============================================================================

def train_gmo_v2(
    data_path: Path,
    output_dir: Path,
    num_epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 0.0003,
    device: str = "cpu",
    eval_interval: int = 10,
) -> None:
    """Train GMO v2 model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training GMO v2 on device: {device}")
    logger.info(f"Loading data from {data_path}")

    device = torch.device(device)

    # Create config
    config = GMOv2Config(
        state_dim=256,
        move_dim=256,
        hidden_dim=512,
        device=str(device),
    )

    # Create networks
    state_encoder = AttentionStateEncoder(
        embed_dim=config.state_dim,
        board_size=8,
        num_heads=4,
        num_layers=2,
    ).to(device)

    move_encoder = MoveEncoderV2(
        embed_dim=config.move_dim,
        board_size=8,
    ).to(device)

    value_net = GMOv2ValueNet(
        state_dim=config.state_dim,
        move_dim=config.move_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)

    # Load dataset
    dataset = GMOv2Dataset(data_path, state_encoder, move_encoder)
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
    optimizer = optim.AdamW(
        list(state_encoder.parameters()) +
        list(move_encoder.parameters()) +
        list(value_net.parameters()),
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
    early_stopping_patience = 15

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            state_encoder, move_encoder, value_net,
            train_loader, optimizer, device
        )

        val_loss, val_acc = evaluate_epoch(
            state_encoder, move_encoder, value_net,
            val_loader, device
        )

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.2%}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = output_dir / "gmo_v2_best.pt"
            torch.save({
                "state_encoder": state_encoder.state_dict(),
                "move_encoder": move_encoder.state_dict(),
                "value_net": value_net.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
            }, checkpoint_path)
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

    parser = argparse.ArgumentParser(description="Train GMO v2 AI")
    parser.add_argument(
        "--data-path", type=Path, required=True,
        help="Path to training data (JSONL)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/gmo_v2"),
        help="Output directory for checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    train_gmo_v2(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        eval_interval=args.eval_interval,
    )


if __name__ == "__main__":
    main()
