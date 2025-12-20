#!/usr/bin/env python3
"""Train EBMO using outcome-weighted contrastive loss.

This script trains EBMO to:
1. Assign low energy to winner's moves
2. Assign high energy to loser's moves
3. Learn from game outcomes, not just MCTS labels

Can run on cluster GPU nodes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Check if we can import the app modules
try:
    from app.ai.ebmo_network import EBMONetwork, EBMOConfig, ActionFeatureExtractor
    from app.ai.ebmo_ai import EBMO_AI
    from app.ai.heuristic_ai import HeuristicAI
    from app.ai.random_ai import RandomAI
    from app.ai.neural_net import NeuralNetAI
    from app.game_engine import GameEngine
    from app.training.initial_state import create_initial_state
    from app.models.core import AIConfig, BoardType, GameStatus
    HAS_APP = True
except ImportError as e:
    print(f"Warning: Could not import app modules: {e}")
    print("Running in standalone mode with existing data only")
    HAS_APP = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GameSample:
    """A single (state, action, outcome) sample."""
    board: np.ndarray  # 56-channel board state
    action: np.ndarray  # Action features
    is_winner: bool  # Whether this player won
    player: int  # Which player made this move


class OutcomeDataset(Dataset):
    """Dataset of game samples with outcome labels."""

    def __init__(self, samples: List[GameSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'board': torch.FloatTensor(s.board),
            'action': torch.FloatTensor(s.action),
            'is_winner': torch.FloatTensor([1.0 if s.is_winner else 0.0]),
        }


def generate_games_vs_heuristic(
    num_games: int,
    model_path: Optional[str] = None,
) -> List[GameSample]:
    """Generate training data from EBMO vs Heuristic games."""
    if not HAS_APP:
        logger.error("Cannot generate games without app modules")
        return []

    config = AIConfig(difficulty=5)
    engine = GameEngine()
    nn = NeuralNetAI(1, config)
    action_extractor = ActionFeatureExtractor(8)

    samples = []
    ebmo_wins = 0
    heuristic_wins = 0

    for game_idx in range(num_games):
        # Alternate who plays first
        ebmo_first = game_idx % 2 == 0

        if model_path and Path(model_path).exists():
            ebmo = EBMO_AI(1 if ebmo_first else 2, config, model_path)
        else:
            ebmo = EBMO_AI(1 if ebmo_first else 2, config)

        heuristic = HeuristicAI(2 if ebmo_first else 1, config)

        ais = {ebmo.player_number: ebmo, heuristic.player_number: heuristic}

        state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)

        game_samples = []
        frame_history = []
        move_count = 0
        max_moves = 500

        while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
            current_player = state.current_player
            current_ai = ais[current_player]

            # Extract features
            try:
                board_feat, _ = nn._extract_features(state)
                frame_history.append(board_feat)
                if len(frame_history) > 4:
                    frame_history.pop(0)
            except Exception:
                pass

            # Get move
            move = current_ai.select_move(state)
            if move is None:
                break

            # Stack frames for 56-channel input
            if len(frame_history) >= 4:
                stacked = np.concatenate(frame_history[-4:], axis=0)
            elif len(frame_history) > 0:
                padding = [frame_history[0]] * (4 - len(frame_history))
                stacked = np.concatenate(padding + frame_history, axis=0)
            else:
                stacked = np.zeros((56, 8, 8), dtype=np.float32)

            # Extract action features
            action_features = action_extractor.extract_features(move)

            # Record sample (outcome will be filled after game ends)
            game_samples.append({
                'board': stacked.copy(),
                'action': action_features,
                'player': current_player,
                'ai_type': 'ebmo' if isinstance(current_ai, EBMO_AI) else 'heuristic',
            })

            state = engine.apply_move(state, move)
            move_count += 1

        # Label samples with outcome
        winner = state.winner
        if winner is not None:
            if winner == ebmo.player_number:
                ebmo_wins += 1
            else:
                heuristic_wins += 1

            for sample in game_samples:
                samples.append(GameSample(
                    board=sample['board'],
                    action=sample['action'],
                    is_winner=(sample['player'] == winner),
                    player=sample['player'],
                ))

        if (game_idx + 1) % 10 == 0:
            logger.info(f"Generated {game_idx + 1}/{num_games} games, "
                       f"EBMO: {ebmo_wins}, Heuristic: {heuristic_wins}, "
                       f"Samples: {len(samples)}")

    logger.info(f"Final: EBMO {ebmo_wins} - {heuristic_wins} Heuristic")
    return samples


def generate_games_vs_random(
    num_games: int,
    model_path: Optional[str] = None,
) -> List[GameSample]:
    """Generate training data from EBMO vs Random games."""
    if not HAS_APP:
        logger.error("Cannot generate games without app modules")
        return []

    config = AIConfig(difficulty=5)
    engine = GameEngine()
    nn = NeuralNetAI(1, config)
    action_extractor = ActionFeatureExtractor(8)

    samples = []
    ebmo_wins = 0
    random_wins = 0

    for game_idx in range(num_games):
        ebmo_first = game_idx % 2 == 0

        if model_path and Path(model_path).exists():
            ebmo = EBMO_AI(1 if ebmo_first else 2, config, model_path)
        else:
            ebmo = EBMO_AI(1 if ebmo_first else 2, config)

        random_ai = RandomAI(2 if ebmo_first else 1, config)

        ais = {ebmo.player_number: ebmo, random_ai.player_number: random_ai}

        state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)

        game_samples = []
        frame_history = []
        move_count = 0

        while state.game_status == GameStatus.ACTIVE and move_count < 500:
            current_player = state.current_player
            current_ai = ais[current_player]

            try:
                board_feat, _ = nn._extract_features(state)
                frame_history.append(board_feat)
                if len(frame_history) > 4:
                    frame_history.pop(0)
            except Exception:
                pass

            move = current_ai.select_move(state)
            if move is None:
                break

            if len(frame_history) >= 4:
                stacked = np.concatenate(frame_history[-4:], axis=0)
            elif len(frame_history) > 0:
                padding = [frame_history[0]] * (4 - len(frame_history))
                stacked = np.concatenate(padding + frame_history, axis=0)
            else:
                stacked = np.zeros((56, 8, 8), dtype=np.float32)

            action_features = action_extractor.extract_features(move)

            game_samples.append({
                'board': stacked.copy(),
                'action': action_features,
                'player': current_player,
            })

            state = engine.apply_move(state, move)
            move_count += 1

        winner = state.winner
        if winner is not None:
            if winner == ebmo.player_number:
                ebmo_wins += 1
            else:
                random_wins += 1

            for sample in game_samples:
                samples.append(GameSample(
                    board=sample['board'],
                    action=sample['action'],
                    is_winner=(sample['player'] == winner),
                    player=sample['player'],
                ))

        if (game_idx + 1) % 10 == 0:
            logger.info(f"Generated {game_idx + 1}/{num_games} games, "
                       f"EBMO: {ebmo_wins}, Random: {random_wins}")

    return samples


class OutcomeTrainer:
    """Trainer for outcome-weighted EBMO."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        learning_rate: float = 1e-4,
        winner_target: float = -1.0,
        loser_target: float = 1.0,
    ):
        # Select device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Load or create model
        self.config = EBMOConfig()
        self.network = EBMONetwork(self.config)

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info(f"Loaded model from {model_path}")

        self.network.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # Loss targets
        self.winner_target = winner_target
        self.loser_target = loser_target

        # Action feature extractor
        self.action_extractor = ActionFeatureExtractor(8)

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.network.train()

        total_loss = 0.0
        total_winner_loss = 0.0
        total_loser_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            boards = batch['board'].to(self.device)
            actions = batch['action'].to(self.device)
            is_winner = batch['is_winner'].to(self.device).squeeze()

            # Encode state and action
            state_embed = self.network.encode_state(boards)
            action_embed = self.network.encode_action(actions)

            # Compute energy
            energy = self.network.compute_energy(state_embed, action_embed).squeeze()

            # Outcome-weighted loss
            winner_mask = is_winner > 0.5
            loser_mask = ~winner_mask

            loss = torch.tensor(0.0, device=self.device)

            if winner_mask.sum() > 0:
                winner_loss = F.mse_loss(energy[winner_mask],
                                         torch.full_like(energy[winner_mask], self.winner_target))
                loss = loss + winner_loss
                total_winner_loss += winner_loss.item()

            if loser_mask.sum() > 0:
                loser_loss = F.mse_loss(energy[loser_mask],
                                        torch.full_like(energy[loser_mask], self.loser_target))
                loss = loss + loser_loss
                total_loser_loss += loser_loss.item()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {
            'loss': total_loss / max(num_batches, 1),
            'winner_loss': total_winner_loss / max(num_batches, 1),
            'loser_loss': total_loser_loss / max(num_batches, 1),
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"Saved model to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=100, help='Number of games to generate')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='models/ebmo_56ch/ebmo_quality_best.pt')
    parser.add_argument('--output', type=str, default='models/ebmo_outcome/ebmo_outcome_best.pt')
    parser.add_argument('--opponent', type=str, default='heuristic', choices=['heuristic', 'random'])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--data-only', action='store_true', help='Only generate data, no training')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate training data
    logger.info(f"Generating {args.games} games vs {args.opponent}...")
    if args.opponent == 'heuristic':
        samples = generate_games_vs_heuristic(args.games, args.model)
    else:
        samples = generate_games_vs_random(args.games, args.model)

    if not samples:
        logger.error("No samples generated")
        return

    logger.info(f"Generated {len(samples)} samples")

    # Save data
    data_path = output_dir / f"outcome_data_{args.opponent}.npz"
    np.savez(
        data_path,
        boards=np.array([s.board for s in samples]),
        actions=np.array([s.action for s in samples]),
        is_winner=np.array([s.is_winner for s in samples]),
    )
    logger.info(f"Saved data to {data_path}")

    if args.data_only:
        return

    # Create dataset and dataloader
    dataset = OutcomeDataset(samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create trainer
    trainer = OutcomeTrainer(
        model_path=args.model,
        device=args.device,
        learning_rate=args.lr,
    )

    # Train
    best_loss = float('inf')
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(dataloader)

        logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                   f"loss={metrics['loss']:.4f}, "
                   f"winner={metrics['winner_loss']:.4f}, "
                   f"loser={metrics['loser_loss']:.4f}")

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            trainer.save(args.output)

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
