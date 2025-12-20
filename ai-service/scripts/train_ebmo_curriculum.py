#!/usr/bin/env python
"""Curriculum self-play training for EBMO.

Generates training data against progressively stronger opponents:
1. Start with Random AI (easy)
2. Add Heuristic AI games
3. Add games against itself
4. Add games against stronger AI (Minimax/MCTS)

This gradual difficulty increase helps the model learn fundamentals
before tackling harder opponents.

Usage:
    python scripts/train_ebmo_curriculum.py --output-dir models/ebmo_curriculum
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.factory import AIFactory
from app.ai.ebmo_ai import EBMO_AI
from app.ai.ebmo_network import (
    EBMOConfig,
    EBMONetwork,
    ActionFeatureExtractor,
    contrastive_energy_loss,
    margin_ranking_loss,
    outcome_weighted_energy_loss,
)
from app.ai.neural_net import NeuralNetAI
from app.models.core import AIType, AIConfig, BoardType
from app.game_engine import GameEngine
from app.training.generate_data import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_ebmo_curriculum")


class CurriculumStage:
    """Definition of a curriculum training stage."""

    def __init__(
        self,
        name: str,
        opponent_type: AIType,
        opponent_difficulty: int,
        num_games: int,
        epochs: int,
        use_self_play: bool = False,
    ):
        self.name = name
        self.opponent_type = opponent_type
        self.opponent_difficulty = opponent_difficulty
        self.num_games = num_games
        self.epochs = epochs
        self.use_self_play = use_self_play


# Curriculum stages: easy to hard
CURRICULUM_STAGES = [
    CurriculumStage("random", AIType.RANDOM, 1, num_games=100, epochs=30),
    CurriculumStage("heuristic_easy", AIType.HEURISTIC, 2, num_games=150, epochs=40),
    CurriculumStage("heuristic_medium", AIType.HEURISTIC, 4, num_games=200, epochs=50),
    CurriculumStage("self_play", AIType.HEURISTIC, 5, num_games=150, epochs=40, use_self_play=True),
    CurriculumStage("minimax", AIType.MINIMAX, 3, num_games=100, epochs=30),
]


def play_game(
    ai1,
    ai2,
    board_type: BoardType = BoardType.SQUARE8,
    max_moves: int = 200,
) -> Tuple[int, List[Dict]]:
    """Play a game and record all states/actions.

    Returns:
        (winner, samples) where samples is list of {board, globals, action, player}
    """
    state = create_initial_state(board_type=board_type, num_players=2)
    engine = GameEngine()
    ais = {1: ai1, 2: ai2}
    samples = []

    # For feature extraction
    nn = NeuralNetAI(1, AIConfig(difficulty=5))
    action_extractor = ActionFeatureExtractor(8)

    for move_num in range(max_moves):
        if state.winner is not None:
            break
        if hasattr(state.game_status, 'value') and state.game_status.value != 'active':
            break

        current = state.current_player
        ai = ais[current]

        # Extract features before move
        try:
            board_feat, global_feat = nn._extract_features(state)
        except Exception:
            break

        move = ai.select_move(state)
        if move is None:
            break

        # Extract action features
        action_feat = action_extractor.extract_features(move)

        samples.append({
            'board': board_feat,
            'globals': global_feat,
            'action': action_feat,
            'player': current,
        })

        state = engine.apply_move(state, move)

    return state.winner, samples


def generate_curriculum_data(
    stage: CurriculumStage,
    ebmo_model_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate training data for a curriculum stage.

    Returns:
        (boards, globals, actions, outcomes) arrays
    """
    logger.info(f"Generating {stage.num_games} games for stage '{stage.name}'")

    all_boards = []
    all_globals = []
    all_actions = []
    all_outcomes = []

    for game_idx in range(stage.num_games):
        # Create opponent
        opponent = AIFactory.create(
            stage.opponent_type,
            player_number=2,
            config=AIConfig(difficulty=stage.opponent_difficulty)
        )

        # Create EBMO player (or use model if available)
        if stage.use_self_play and ebmo_model_path and Path(ebmo_model_path).exists():
            ebmo = EBMO_AI(
                player_number=1,
                config=AIConfig(difficulty=5),
                model_path=ebmo_model_path
            )
        else:
            # Use heuristic as proxy for EBMO during initial stages
            ebmo = AIFactory.create(
                AIType.HEURISTIC,
                player_number=1,
                config=AIConfig(difficulty=5)
            )

        # Play game
        winner, samples = play_game(ebmo, opponent)

        if not samples:
            continue

        # Label samples by outcome
        for sample in samples:
            player = sample['player']

            # Determine outcome from player's perspective
            if winner is None:
                outcome = 0  # Draw
            elif winner == player:
                outcome = 1  # Win
            else:
                outcome = -1  # Loss

            all_boards.append(sample['board'])
            all_globals.append(sample['globals'])
            all_actions.append(sample['action'])
            all_outcomes.append(outcome)

        if (game_idx + 1) % 20 == 0:
            logger.info(f"  Completed {game_idx + 1}/{stage.num_games} games, {len(all_boards)} samples")

    return (
        np.array(all_boards),
        np.array(all_globals),
        np.array(all_actions),
        np.array(all_outcomes),
    )


def train_on_data(
    model: EBMONetwork,
    boards: np.ndarray,
    globals_arr: np.ndarray,
    actions: np.ndarray,
    outcomes: np.ndarray,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    batch_size: int = 128,
    num_negatives: int = 8,
) -> float:
    """Train model on curriculum data."""
    model.train()

    # Create tensors
    boards_t = torch.from_numpy(boards).float()
    globals_t = torch.from_numpy(globals_arr).float()
    actions_t = torch.from_numpy(actions).float()
    outcomes_t = torch.from_numpy(outcomes).float()

    # Separate positive (winning) and negative (losing) samples
    winning_mask = outcomes_t > 0
    losing_mask = outcomes_t < 0

    winning_indices = torch.where(winning_mask)[0]
    losing_indices = torch.where(losing_mask)[0]

    logger.info(f"Training on {len(boards)} samples ({len(winning_indices)} wins, {len(losing_indices)} losses)")

    best_loss = float('inf')

    for epoch in range(epochs):
        # Shuffle winning samples
        perm = torch.randperm(len(winning_indices))
        shuffled_win_idx = winning_indices[perm]

        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(shuffled_win_idx), batch_size):
            batch_idx = shuffled_win_idx[i:i+batch_size]

            board = boards_t[batch_idx].to(device)
            globals_vec = globals_t[batch_idx].to(device)
            pos_action = actions_t[batch_idx].to(device)
            batch_outcomes = outcomes_t[batch_idx].to(device)

            # Sample hard negatives from losing moves
            if len(losing_indices) > 0:
                neg_sample_idx = losing_indices[
                    torch.randint(0, len(losing_indices), (len(batch_idx), num_negatives))
                ]
                neg_actions = actions_t[neg_sample_idx].to(device)
            else:
                # Fall back to random negatives
                neg_actions = torch.rand(len(batch_idx), num_negatives, actions_t.shape[1]).to(device)

            optimizer.zero_grad()

            # Forward pass
            state_embed = model.state_encoder(board, globals_vec)
            pos_embed = model.action_encoder(pos_action)

            pos_energy = model.energy_head(state_embed, pos_embed)

            # Encode negatives
            bs, n_neg, act_dim = neg_actions.shape
            neg_flat = neg_actions.view(-1, act_dim)
            neg_embed = model.action_encoder(neg_flat).view(bs, n_neg, -1)

            neg_energies = []
            for j in range(n_neg):
                ne = model.energy_head(state_embed, neg_embed[:, j, :])
                neg_energies.append(ne)
            neg_energy = torch.stack(neg_energies, dim=1)

            # Losses
            contrastive = contrastive_energy_loss(pos_energy, neg_energy, temperature=0.1)
            margin = margin_ranking_loss(pos_energy, neg_energy, margin=1.0)
            outcome = outcome_weighted_energy_loss(pos_energy, batch_outcomes)

            loss = 0.3 * contrastive + 0.3 * margin + 0.4 * outcome

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    return best_loss


def main():
    parser = argparse.ArgumentParser(description="Curriculum self-play training for EBMO")
    parser.add_argument("--output-dir", type=str, default="models/ebmo_curriculum", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, mps, cpu)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--resume-from", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    config = EBMOConfig(
        optim_steps=100,
        num_restarts=8,
        projection_temperature=0.3,
    )

    if args.resume_from and Path(args.resume_from).exists():
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model = EBMONetwork(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Resumed from {args.resume_from}")
    else:
        model = EBMONetwork(config)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Train through curriculum stages
    model_path = None

    for stage_idx, stage in enumerate(CURRICULUM_STAGES):
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE {stage_idx + 1}/{len(CURRICULUM_STAGES)}: {stage.name}")
        logger.info(f"{'='*60}")

        # Generate data for this stage
        boards, globals_arr, actions, outcomes = generate_curriculum_data(
            stage,
            ebmo_model_path=model_path,
            device=device,
        )

        if len(boards) == 0:
            logger.warning(f"No data generated for stage {stage.name}, skipping")
            continue

        # Train on this stage's data
        loss = train_on_data(
            model, boards, globals_arr, actions, outcomes,
            optimizer, device, stage.epochs, args.batch_size
        )

        # Save checkpoint after each stage
        checkpoint_path = output_dir / f"ebmo_curriculum_stage{stage_idx}_{stage.name}.pt"
        torch.save({
            'stage': stage_idx,
            'stage_name': stage.name,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'loss': loss,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        model_path = str(checkpoint_path)

    # Save final model
    final_path = output_dir / "ebmo_curriculum_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)
    logger.info(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
