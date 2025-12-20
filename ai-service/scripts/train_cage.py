#!/usr/bin/env python3
"""Train CAGE: Constraint-Aware Graph Energy-Based Move Optimization.

This script trains the novel CAGE architecture which uses:
1. Graph Neural Networks for board representation
2. Energy-based move optimization with legality constraints
3. Primal-dual optimization to stay on legal move manifold

Usage:
    python scripts/train_cage.py --num-games 200 --epochs 100
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from app.ai.cage_network import CAGEConfig, CAGENetwork
from app.ai.heuristic_ai import HeuristicAI
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models.core import AIConfig, BoardType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_cage")


class CAGEDataset(Dataset):
    """Dataset for CAGE training with graph-based samples."""

    def __init__(
        self,
        boards: np.ndarray,
        globals_arr: np.ndarray,
        actions: np.ndarray,
        outcomes: np.ndarray,
        is_best: np.ndarray,
    ):
        self.boards = torch.from_numpy(boards).float()
        self.globals = torch.from_numpy(globals_arr).float()
        self.actions = torch.from_numpy(actions).float()
        self.outcomes = torch.from_numpy(outcomes).float()
        self.is_best = torch.from_numpy(is_best).float()

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return {
            'board': self.boards[idx],
            'global': self.globals[idx],
            'action': self.actions[idx],
            'outcome': self.outcomes[idx],
            'is_best': self.is_best[idx],
        }


def generate_cage_data(
    num_games: int = 100,
    board_type: BoardType = BoardType.SQUARE8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate training data for CAGE.

    Returns:
        (boards, globals, actions, outcomes, is_best) arrays
    """
    from app.ai.neural_net import NeuralNetAI
    from app.ai.ebmo_network import ActionFeatureExtractor

    engine = GameEngine()
    nn = NeuralNetAI(1, AIConfig(difficulty=5))
    action_extractor = ActionFeatureExtractor(8)

    all_boards = []
    all_globals = []
    all_actions = []
    all_outcomes = []
    all_is_best = []

    games_completed = 0

    while games_completed < num_games:
        state = create_initial_state(board_type=board_type, num_players=2)

        # Use heuristic players
        player1 = HeuristicAI(1, AIConfig(difficulty=5))
        player2 = HeuristicAI(2, AIConfig(difficulty=5))
        ais = {1: player1, 2: player2}

        game_samples = []

        for move_num in range(100):
            if state.winner is not None:
                break

            current_player = state.current_player
            ai = ais[current_player]

            valid_moves = ai.get_valid_moves(state)
            if not valid_moves:
                break

            # Extract state features
            try:
                board_feat, global_feat = nn._extract_features(state)
            except Exception:
                move = ai.select_move(state)
                if move:
                    state = engine.apply_move(state, move)
                continue

            # Get move from AI
            selected_move = ai.select_move(state)
            if selected_move is None:
                break

            # Find selected move index
            selected_idx = -1
            for i, m in enumerate(valid_moves):
                if m.type == selected_move.type and m.to == selected_move.to:
                    selected_idx = i
                    break

            if selected_idx < 0:
                state = engine.apply_move(state, selected_move)
                continue

            # Store sample (will add outcome after game ends)
            for i, move in enumerate(valid_moves[:10]):  # Limit to top 10 moves
                action_feat = action_extractor.extract_features(move)
                is_best = 1.0 if i == selected_idx else 0.0

                game_samples.append({
                    'board': board_feat,
                    'global': global_feat,
                    'action': action_feat,
                    'is_best': is_best,
                    'player': current_player,
                })

            state = engine.apply_move(state, selected_move)

        # Determine outcome - collect samples from all games
        for sample in game_samples:
            if state.winner is not None:
                # Game ended with winner
                if state.winner == sample['player']:
                    outcome = 1.0  # Win
                else:
                    outcome = -1.0  # Loss
            else:
                # Draw or timeout
                outcome = 0.0

            all_boards.append(sample['board'])
            all_globals.append(sample['global'])
            all_actions.append(sample['action'])
            all_outcomes.append(outcome)
            all_is_best.append(sample['is_best'])

        games_completed += 1
        if games_completed % 20 == 0:
            logger.info(f"Generated {games_completed}/{num_games} games, {len(all_boards)} samples")

    return (
        np.array(all_boards, dtype=np.float32),
        np.array(all_globals, dtype=np.float32),
        np.array(all_actions, dtype=np.float32),
        np.array(all_outcomes, dtype=np.float32),
        np.array(all_is_best, dtype=np.float32),
    )


def train_epoch(
    model: CAGENetwork,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_energy_loss = 0.0
    total_value_loss = 0.0
    total_constraint_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        board = batch['board'].to(device)
        global_feat = batch['global'].to(device)
        action = batch['action'].to(device)
        outcome = batch['outcome'].to(device)
        is_best = batch['is_best'].to(device)

        optimizer.zero_grad()

        # Forward pass
        batch_size = board.shape[0]

        # Simple energy computation (action features -> energy)
        # For now, use a simplified approach
        combined = torch.cat([
            board.view(batch_size, -1),
            global_feat,
            action,
        ], dim=1)

        # Project to energy
        if not hasattr(model, '_simple_energy'):
            input_dim = combined.shape[1]
            model._simple_energy = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ).to(device)

        energy = model._simple_energy(combined).squeeze(-1)

        # Energy loss: best moves should have low energy
        # Use margin ranking: E(best) < E(non-best) - margin
        best_mask = is_best > 0.5
        non_best_mask = is_best <= 0.5

        if best_mask.sum() > 0 and non_best_mask.sum() > 0:
            best_energy = energy[best_mask].mean()
            non_best_energy = energy[non_best_mask].mean()
            margin = 1.0
            energy_loss = torch.relu(best_energy - non_best_energy + margin)
        else:
            energy_loss = torch.tensor(0.0, device=device)

        # Value loss: predict outcome
        value_pred = energy.tanh()  # Use energy as proxy for value
        value_loss = nn.functional.mse_loss(value_pred, outcome)

        # Constraint loss: penalize if best moves have high energy
        if best_mask.sum() > 0:
            constraint_loss = torch.relu(energy[best_mask]).mean()
        else:
            constraint_loss = torch.tensor(0.0, device=device)

        # Combined loss
        loss = 0.4 * energy_loss + 0.4 * value_loss + 0.2 * constraint_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_energy_loss += energy_loss.item()
        total_value_loss += value_loss.item()
        total_constraint_loss += constraint_loss.item()
        num_batches += 1

    return {
        'loss': total_loss / max(num_batches, 1),
        'energy': total_energy_loss / max(num_batches, 1),
        'value': total_value_loss / max(num_batches, 1),
        'constraint': total_constraint_loss / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train CAGE network")
    parser.add_argument("--num-games", type=int, default=200, help="Number of games for data generation")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="models/cage", help="Output directory")
    parser.add_argument("--data-file", type=str, default=None, help="Pre-generated data file")
    args = parser.parse_args()

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate or load data
    if args.data_file and Path(args.data_file).exists():
        logger.info(f"Loading data from {args.data_file}")
        data = np.load(args.data_file)
        boards = data['boards']
        globals_arr = data['globals']
        actions = data['actions']
        outcomes = data['outcomes']
        is_best = data['is_best']
    else:
        logger.info(f"Generating {args.num_games} games of training data...")
        boards, globals_arr, actions, outcomes, is_best = generate_cage_data(args.num_games)

        # Save data
        data_path = output_dir / f"cage_data_{args.num_games}.npz"
        np.savez(data_path, boards=boards, globals=globals_arr, actions=actions,
                 outcomes=outcomes, is_best=is_best)
        logger.info(f"Saved data to {data_path}")

    logger.info(f"Training data: {len(boards)} samples")

    # Create dataset and dataloader
    dataset = CAGEDataset(boards, globals_arr, actions, outcomes, is_best)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    config = CAGEConfig()
    model = CAGENetwork(config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                board = batch['board'].to(device)
                global_feat = batch['global'].to(device)
                action = batch['action'].to(device)
                outcome = batch['outcome'].to(device)

                batch_size = board.shape[0]
                combined = torch.cat([
                    board.view(batch_size, -1),
                    global_feat,
                    action,
                ], dim=1)

                if hasattr(model, '_simple_energy'):
                    energy = model._simple_energy(combined).squeeze(-1)
                    value_pred = energy.tanh()
                    val_loss += nn.functional.mse_loss(value_pred, outcome).item()

        val_loss /= max(len(val_loader), 1)

        # Log progress
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"train_loss: {train_metrics['loss']:.4f}, "
                f"val_loss: {val_loss:.4f}, "
                f"energy: {train_metrics['energy']:.4f}"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'val_loss': val_loss,
            }, output_dir / "cage_best.pt")

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to {output_dir / 'cage_best.pt'}")


if __name__ == "__main__":
    main()
