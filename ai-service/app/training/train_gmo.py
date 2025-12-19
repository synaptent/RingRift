"""Training script for Gradient Move Optimization (GMO) AI.

This script trains the GMO networks (StateEncoder, MoveEncoder, ValueNet)
on existing game data from Gumbel MCTS self-play.

Usage:
    python -m app.training.train_gmo --data-path data/gumbel_selfplay/sq8_gumbel_kl_canonical.jsonl

Training procedure:
1. Load game records with move sequences and outcomes
2. Extract (state, move, outcome) tuples from each game
3. Train networks to predict value and uncertainty
4. Evaluate by playing games vs Random AI
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..ai.gmo_ai import (
    GMOAI,
    GMOConfig,
    GMOValueNetWithUncertainty,
    MoveEncoder,
    StateEncoder,
    gmo_combined_loss,
    nll_loss_with_uncertainty,
)
from ..models import AIConfig, BoardType, GameState, Move

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

class GMODataset(Dataset):
    """Dataset for GMO training.

    Loads game records and extracts (state, move, outcome) tuples.
    """

    def __init__(
        self,
        data_path: Path,
        state_encoder: StateEncoder,
        move_encoder: MoveEncoder,
        max_samples: Optional[int] = None,
    ):
        self.data_path = data_path
        self.state_encoder = state_encoder
        self.move_encoder = move_encoder
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, float]] = []

        self._load_data(max_samples)

    def _load_data(self, max_samples: Optional[int]) -> None:
        """Load and process game records."""
        logger.info(f"Loading data from {self.data_path}")
        games_processed = 0

        with open(self.data_path, "r") as f:
            for line_num, line in enumerate(f):
                if max_samples and len(self.samples) >= max_samples:
                    break

                # Skip comment lines and empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    record = json.loads(line)
                    self._process_record(record)
                    games_processed += 1
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue

        logger.info(f"Processed {games_processed} games, loaded {len(self.samples)} training samples")

    def _process_record(self, record: Dict) -> None:
        """Process a single game record.

        Uses simplified approach: encode initial state once and pair with each move.
        The value is the game outcome from the perspective of the player making the move.
        """
        # Extract winner (1 or 2)
        winner = record.get("winner")
        if winner is None or winner == 0:
            return  # Skip draws/incomplete games

        # Get initial state and moves
        initial_state_dict = record.get("initial_state")
        moves_data = record.get("moves", [])

        if not initial_state_dict or not moves_data:
            return

        try:
            # Parse initial state and encode once
            state = GameState.model_validate(initial_state_dict)
            with torch.no_grad():
                state_features = torch.from_numpy(
                    self.state_encoder.extract_features(state)
                ).float()
        except Exception as e:
            logger.debug(f"Error parsing/encoding initial state: {e}")
            return

        # Process each move (without replaying - simpler approach)
        for i, move_dict in enumerate(moves_data):
            try:
                # Add id if missing (training data may not have it)
                if "id" not in move_dict:
                    move_dict = {**move_dict, "id": f"train_{i}"}

                # Parse move
                move = Move.model_validate(move_dict)
                player = move.player

                # Determine outcome for this player (+1 if winner, -1 if loser)
                # Use temporal discounting: earlier moves have values closer to 0
                move_progress = i / max(len(moves_data) - 1, 1)  # 0 to 1
                discount = 0.5 + 0.5 * move_progress  # 0.5 to 1.0
                player_outcome = discount * (1.0 if player == winner else -1.0)

                # Encode move
                with torch.no_grad():
                    move_embed = self.move_encoder.encode_move(move)

                self.samples.append((state_features, move_embed, player_outcome))

            except Exception as e:
                logger.debug(f"Error processing move {i}: {e}")
                continue  # Skip this move but continue with others

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_features, move_embed, outcome = self.samples[idx]
        return state_features, move_embed, torch.tensor(outcome, dtype=torch.float32)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    states, moves, outcomes = zip(*batch)
    return (
        torch.stack(states),
        torch.stack(moves),
        torch.stack(outcomes),
    )


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    state_encoder: StateEncoder,
    move_encoder: MoveEncoder,
    value_net: GMOValueNetWithUncertainty,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    state_encoder.train()
    value_net.train()
    # Move encoder is not trained (embeddings are fixed during this phase)

    total_loss = 0.0
    num_batches = 0

    for state_features, move_embeds, outcomes in tqdm(dataloader, desc="Training"):
        state_features = state_features.to(device)
        move_embeds = move_embeds.to(device)
        outcomes = outcomes.to(device)

        optimizer.zero_grad()

        # Encode states
        state_embeds = state_encoder.encoder(state_features)

        # Forward through value network
        pred_values, pred_log_vars = value_net(state_embeds, move_embeds)

        # Compute loss
        loss = nll_loss_with_uncertainty(pred_values, pred_log_vars, outcomes)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_epoch(
    state_encoder: StateEncoder,
    move_encoder: MoveEncoder,
    value_net: GMOValueNetWithUncertainty,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate on validation set.

    Returns:
        avg_loss: Average loss
        accuracy: Fraction of correct sign predictions
    """
    state_encoder.eval()
    value_net.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for state_features, move_embeds, outcomes in dataloader:
            state_features = state_features.to(device)
            move_embeds = move_embeds.to(device)
            outcomes = outcomes.to(device)

            # Forward
            state_embeds = state_encoder.encoder(state_features)
            pred_values, pred_log_vars = value_net(state_embeds, move_embeds)

            # Loss
            loss = nll_loss_with_uncertainty(pred_values, pred_log_vars, outcomes)
            total_loss += loss.item()

            # Accuracy (sign prediction)
            pred_sign = (pred_values.squeeze() > 0).float()
            true_sign = (outcomes > 0).float()
            correct += (pred_sign == true_sign).sum().item()
            total += outcomes.shape[0]

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy


# =============================================================================
# Evaluation vs Random AI
# =============================================================================

def evaluate_vs_random(
    gmo_ai: GMOAI,
    num_games: int = 20,
    board_type: BoardType = BoardType.SQUARE8,
) -> Dict[str, float]:
    """Evaluate GMO AI against Random AI.

    Returns:
        Dictionary with win_rate, draw_rate, avg_game_length
    """
    from datetime import datetime
    from ..ai.random_ai import RandomAI
    from ..game_engine import GameEngine
    from ..models import (
        BoardState, GamePhase, GameStatus, Player, TimeControl,
    )
    from ..rules.core import (
        BOARD_CONFIGS,
        get_victory_threshold,
        get_territory_victory_threshold,
    )

    engine = GameEngine()
    wins = 0
    draws = 0
    total_moves = 0

    # Get board configuration
    if board_type in BOARD_CONFIGS:
        config = BOARD_CONFIGS[board_type]
        size = config.size
        rings_per_player = config.rings_per_player
    else:
        size = 8
        rings_per_player = 18

    victory_threshold = get_victory_threshold(board_type, 2)
    territory_threshold = get_territory_victory_threshold(board_type)

    for game_num in range(num_games):
        # Alternate who plays first
        gmo_is_player1 = (game_num % 2 == 0)

        # Create AIs
        if gmo_is_player1:
            player1 = gmo_ai
            player2 = RandomAI(2, AIConfig(difficulty=1))
            gmo_player = 1
        else:
            player1 = RandomAI(1, AIConfig(difficulty=1))
            player2 = gmo_ai
            gmo_player = 2

        # Reset AIs
        gmo_ai.reset_for_new_game(rng_seed=game_num * 1000)
        if hasattr(player1, "reset_for_new_game"):
            player1.reset_for_new_game(rng_seed=game_num * 1000 + 1)
        if hasattr(player2, "reset_for_new_game"):
            player2.reset_for_new_game(rng_seed=game_num * 1000 + 2)

        # Create players
        players = [
            Player(
                id=f"p{idx}",
                username=f"AI {idx}",
                type="ai",
                playerNumber=idx,
                isReady=True,
                timeRemaining=600,
                ringsInHand=rings_per_player,
                eliminatedRings=0,
                territorySpaces=0,
                aiDifficulty=5,
            )
            for idx in range(1, 3)
        ]

        # Create initial game state
        state = GameState(
            id=f"eval_{game_num}",
            boardType=board_type,
            rngSeed=game_num,
            board=BoardState(
                type=board_type,
                size=size,
                stacks={},
                markers={},
                collapsedSpaces={},
                eliminatedRings={},
            ),
            players=players,
            currentPhase=GamePhase.RING_PLACEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(initialTime=600, increment=0, type="blitz"),
            gameStatus=GameStatus.ACTIVE,
            createdAt=datetime.now(),
            lastMoveAt=datetime.now(),
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=rings_per_player * 2,
            totalRingsEliminated=0,
            victoryThreshold=victory_threshold,
            territoryVictoryThreshold=territory_threshold,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
            zobristHash=None,
            lpsRoundIndex=0,
            lpsExclusivePlayerForCompletedRound=None,
        )
        move_count = 0
        max_moves = 500

        while state.winner is None and move_count < max_moves:
            current_player = state.current_player
            ai = player1 if current_player == 1 else player2

            move = ai.select_move(state)
            if move is None:
                break

            state = engine.apply_move(state, move)
            move_count += 1

        total_moves += move_count

        # Record result
        if state.winner == gmo_player:
            wins += 1
        elif state.winner is None or state.winner == 0:
            draws += 1

    return {
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
        "avg_game_length": total_moves / num_games,
    }


# =============================================================================
# Main Training Function
# =============================================================================

def train_gmo(
    data_path: Path,
    output_dir: Path,
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    max_samples: Optional[int] = None,
    eval_interval: int = 5,
    device_str: str = "cpu",
) -> None:
    """Train GMO networks.

    Args:
        data_path: Path to training data (JSONL format)
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        max_samples: Maximum training samples (None for all)
        eval_interval: Epochs between evaluations
        device_str: Device to use ("cpu", "cuda", "mps")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    logger.info(f"Training GMO on device: {device}")

    # Initialize networks
    config = GMOConfig(device=device_str)
    state_encoder = StateEncoder(
        embed_dim=config.state_dim,
        board_size=8,
    ).to(device)

    move_encoder = MoveEncoder(
        embed_dim=config.move_dim,
        board_size=8,
    ).to(device)

    value_net = GMOValueNetWithUncertainty(
        state_dim=config.state_dim,
        move_dim=config.move_dim,
        hidden_dim=config.hidden_dim,
        dropout_rate=config.dropout_rate,
    ).to(device)

    # Create dataset
    dataset = GMODataset(
        data_path=data_path,
        state_encoder=state_encoder,
        move_encoder=move_encoder,
        max_samples=max_samples,
    )

    # Split into train/val
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

    # Optimizer (train encoder and value net, not move embeddings)
    optimizer = optim.Adam(
        list(state_encoder.parameters()) + list(value_net.parameters()),
        lr=learning_rate,
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            state_encoder, move_encoder, value_net,
            train_loader, optimizer, device
        )

        # Validate
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
            checkpoint_path = output_dir / "gmo_best.pt"
            torch.save({
                "state_encoder": state_encoder.state_dict(),
                "move_encoder": move_encoder.state_dict(),
                "value_net": value_net.state_dict(),
                "gmo_config": config,
                "epoch": epoch,
                "val_loss": val_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")

        # Evaluate vs Random AI periodically
        if (epoch + 1) % eval_interval == 0:
            # Create GMO AI with current weights
            gmo_ai = GMOAI(1, AIConfig(difficulty=6), gmo_config=config)
            gmo_ai.state_encoder = state_encoder
            gmo_ai.move_encoder = move_encoder
            gmo_ai.value_net = value_net
            gmo_ai._is_trained = True

            try:
                results = evaluate_vs_random(gmo_ai, num_games=10)
                logger.info(
                    f"Eval vs Random: "
                    f"win_rate={results['win_rate']:.1%}, "
                    f"avg_length={results['avg_game_length']:.1f}"
                )
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

    # Save final model
    final_path = output_dir / "gmo_final.pt"
    torch.save({
        "state_encoder": state_encoder.state_dict(),
        "move_encoder": move_encoder.state_dict(),
        "value_net": value_net.state_dict(),
        "gmo_config": config,
        "epoch": num_epochs,
    }, final_path)

    logger.info(f"Training complete. Final model saved to {final_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GMO AI")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/gumbel_selfplay/sq8_gumbel_kl_canonical.jsonl"),
        help="Path to training data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/gmo"),
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Train
    train_gmo(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        eval_interval=args.eval_interval,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
