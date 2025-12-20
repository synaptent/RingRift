"""Train GMO against diverse opponents for better generalization.

Instead of self-play (which produces no training signal), this trains
against a mix of opponents at different skill levels.

Usage:
    python -m app.training.train_gmo_diverse --rounds 5 --games-per-opponent 20
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..ai.factory import AIFactory
from ..ai.gmo_ai import (
    GMOAI,
    GMOConfig,
    GMOValueNetWithUncertainty,
    MoveEncoder,
    StateEncoder,
    nll_loss_with_uncertainty,
)
from ..game_engine import GameEngine
from ..models import (
    AIConfig,
    AIType,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    Move,
    Player,
    TimeControl,
)
from ..rules.core import BOARD_CONFIGS, get_territory_victory_threshold, get_victory_threshold

logger = logging.getLogger(__name__)


class DiverseOpponentDataset(Dataset):
    """Dataset that collects games against diverse opponents."""

    def __init__(
        self,
        state_encoder: StateEncoder,
        move_encoder: MoveEncoder,
    ):
        self.state_encoder = state_encoder
        self.move_encoder = move_encoder
        self.samples: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def add_game(self, states: list[GameState], moves: list[Move], winner: int, gmo_player: int) -> int:
        """Add a game to the dataset.

        Args:
            states: List of game states
            moves: List of moves played
            winner: Winning player number (0 for draw)
            gmo_player: Which player GMO was (1 or 2)

        Returns:
            Number of samples added
        """
        if winner == 0:
            return 0  # Skip draws

        samples_added = 0
        num_moves = len(moves)

        for i, (state, move) in enumerate(zip(states, moves, strict=False)):
            # Only learn from GMO's moves
            if move.player != gmo_player:
                continue

            try:
                # Temporal discounting - later moves are more informative
                progress = i / max(num_moves - 1, 1)
                discount = 0.5 + 0.5 * progress

                # Outcome from GMO's perspective
                if winner == gmo_player:
                    outcome = discount * 1.0  # Win
                else:
                    outcome = discount * -1.0  # Loss

                # Encode state and move
                with torch.no_grad():
                    state_features = torch.tensor(
                        self.state_encoder.extract_features(state),
                        dtype=torch.float32
                    )
                    move_embed = self.move_encoder.encode_move(move)

                self.samples.append((state_features, move_embed, outcome))
                samples_added += 1

            except Exception as e:
                logger.debug(f"Error adding sample: {e}")
                continue

        return samples_added

    def clear(self):
        """Clear all samples."""
        self.samples.clear()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_features, move_embed, outcome = self.samples[idx]
        return state_features, move_embed, torch.tensor(outcome, dtype=torch.float32)


def collate_fn(batch):
    """Custom collate function."""
    states, moves, outcomes = zip(*batch, strict=False)
    return torch.stack(states), torch.stack(moves), torch.stack(outcomes)


def create_initial_state(
    game_id: str,
    board_type: BoardType = BoardType.SQUARE8,
    rng_seed: int = 0,
) -> GameState:
    """Create an initial game state."""
    if board_type in BOARD_CONFIGS:
        config = BOARD_CONFIGS[board_type]
        size = config.size
        rings_per_player = config.rings_per_player
    else:
        size = 8
        rings_per_player = 18

    victory_threshold = get_victory_threshold(board_type, 2)
    territory_threshold = get_territory_victory_threshold(board_type)

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

    return GameState(
        id=game_id,
        boardType=board_type,
        rngSeed=rng_seed,
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


def create_opponent(opponent_type: str, player_number: int):
    """Create an opponent AI."""
    if opponent_type == "random":
        config = AIConfig(difficulty=1)
        return AIFactory.create(AIType.RANDOM, player_number, config)
    elif opponent_type == "heuristic":
        config = AIConfig(difficulty=2)
        return AIFactory.create(AIType.HEURISTIC, player_number, config)
    elif opponent_type == "policy":
        return AIFactory.create_for_tournament("policy_only", player_number, "square8")
    elif opponent_type == "descent":
        config = AIConfig(difficulty=6, think_time=2000, use_neural_net=True)
        return AIFactory.create(AIType.DESCENT, player_number, config)
    elif opponent_type == "mcts":
        config = AIConfig(difficulty=6, think_time=3000, use_neural_net=True)
        return AIFactory.create(AIType.MCTS, player_number, config)
    else:
        raise ValueError(f"Unknown opponent: {opponent_type}")


def play_game(
    gmo_ai: GMOAI,
    opponent,
    gmo_player: int,
    game_id: str,
    max_moves: int = 600,
) -> tuple[int, list[GameState], list[Move]]:
    """Play a game between GMO and an opponent.

    Args:
        gmo_ai: The GMO AI
        opponent: The opponent AI
        gmo_player: Which player GMO is (1 or 2)
        game_id: Game identifier
        max_moves: Maximum moves before stopping

    Returns:
        (winner, states, moves)
    """
    state = create_initial_state(
        game_id=game_id,
        board_type=BoardType.SQUARE8,
        rng_seed=hash(game_id) % (2**31),
    )

    states = [state]
    moves = []

    while state.game_status.value == "active" and len(moves) < max_moves:
        current_player = state.current_player
        legal_moves = GameEngine.get_valid_moves(state, current_player)

        if not legal_moves:
            # Check for bookkeeping moves
            requirement = GameEngine.get_phase_requirement(state, current_player)
            if requirement is not None:
                move = GameEngine.synthesize_bookkeeping_move(requirement, state)
                if move:
                    state = GameEngine.apply_move(state, move)
                    states.append(state)
                    moves.append(move)
                    continue
            break

        # Select AI based on current player
        ai = gmo_ai if current_player == gmo_player else opponent
        move = ai.select_move(state)

        if move is None:
            requirement = GameEngine.get_phase_requirement(state, current_player)
            if requirement is not None:
                move = GameEngine.synthesize_bookkeeping_move(requirement, state)

        if move is None:
            break

        state = GameEngine.apply_move(state, move)
        states.append(state)
        moves.append(move)

    winner = state.winner if state.winner else 0
    return winner, states[:-1], moves


def generate_diverse_games(
    gmo_ai: GMOAI,
    dataset: DiverseOpponentDataset,
    games_per_opponent: int = 20,
    opponent_types: list[str] | None = None,
) -> dict[str, dict]:
    """Generate games against diverse opponents.

    Args:
        gmo_ai: The GMO AI instance
        dataset: Dataset to add samples to
        games_per_opponent: Games per opponent type
        opponent_types: List of opponent types

    Returns:
        Statistics per opponent
    """
    if opponent_types is None:
        opponent_types = ["random", "heuristic", "policy"]

    stats = {}

    for opp_type in opponent_types:
        wins = 0
        losses = 0
        samples = 0

        games_as_p1 = games_per_opponent // 2
        games_as_p2 = games_per_opponent - games_as_p1

        # Play as player 1
        for i in tqdm(range(games_as_p1), desc=f"vs {opp_type} (as P1)"):
            game_id = f"diverse_{opp_type}_p1_{i}"
            gmo_ai.reset_for_new_game()
            opponent = create_opponent(opp_type, player_number=2)

            winner, game_states, game_moves = play_game(
                gmo_ai, opponent, gmo_player=1, game_id=game_id
            )

            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1

            added = dataset.add_game(game_states, game_moves, winner, gmo_player=1)
            samples += added

        # Play as player 2
        for i in tqdm(range(games_as_p2), desc=f"vs {opp_type} (as P2)"):
            game_id = f"diverse_{opp_type}_p2_{i}"
            gmo_ai.reset_for_new_game()
            opponent = create_opponent(opp_type, player_number=1)

            winner, game_states, game_moves = play_game(
                gmo_ai, opponent, gmo_player=2, game_id=game_id
            )

            if winner == 2:
                wins += 1
            elif winner == 1:
                losses += 1

            added = dataset.add_game(game_states, game_moves, winner, gmo_player=2)
            samples += added

        total = games_per_opponent
        win_rate = 100 * wins / total if total > 0 else 0
        stats[opp_type] = {
            "wins": wins,
            "losses": losses,
            "draws": total - wins - losses,
            "win_rate": win_rate,
            "samples": samples,
        }
        logger.info(f"  vs {opp_type}: {win_rate:.1f}% win rate, {samples} samples")

    return stats


def train_on_dataset(
    state_encoder: StateEncoder,
    move_encoder: MoveEncoder,
    value_net: GMOValueNetWithUncertainty,
    dataset: DiverseOpponentDataset,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
) -> float:
    """Train on collected dataset.

    Returns:
        Final training loss
    """
    if len(dataset) == 0:
        logger.warning("Empty dataset, skipping training")
        return 0.0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = optim.Adam(
        list(state_encoder.parameters()) +
        list(move_encoder.parameters()) +
        list(value_net.parameters()),
        lr=lr,
    )

    state_encoder.train()
    move_encoder.train()
    value_net.train()

    total_loss = 0.0
    num_batches = 0

    for _epoch in range(epochs):
        epoch_loss = 0.0
        for state_features, move_embeds, outcomes in dataloader:
            optimizer.zero_grad()

            # Encode states
            state_embeds = state_encoder.encoder(state_features)

            # Get predictions
            pred_values, pred_log_vars = value_net(state_embeds, move_embeds)

            # Compute loss
            loss = nll_loss_with_uncertainty(
                pred_values.squeeze(),
                pred_log_vars.squeeze(),
                outcomes,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        total_loss = epoch_loss / max(len(dataloader), 1)

    return total_loss


def evaluate_model(gmo_ai: GMOAI, num_games: int = 10) -> dict[str, float]:
    """Quick evaluation against random and heuristic."""
    results = {}

    for opp_type in ["random", "heuristic"]:
        wins = 0
        for i in range(num_games):
            game_id = f"eval_{opp_type}_{i}"
            gmo_ai.reset_for_new_game()
            opponent = create_opponent(opp_type, player_number=2)

            winner, _, _ = play_game(gmo_ai, opponent, gmo_player=1, game_id=game_id)
            if winner == 1:
                wins += 1

        results[f"vs_{opp_type}"] = 100 * wins / num_games

    return results


def run_diverse_training(
    checkpoint_path: str | None = None,
    rounds: int = 5,
    games_per_opponent: int = 20,
    epochs_per_round: int = 10,
    opponent_types: list[str] | None = None,
    output_dir: str = "models/gmo/diverse",
):
    """Run training against diverse opponents.

    Args:
        checkpoint_path: Path to starting checkpoint
        rounds: Number of training rounds
        games_per_opponent: Games per opponent per round
        epochs_per_round: Training epochs per round
        opponent_types: Opponent types to train against
        output_dir: Output directory for checkpoints
    """
    if opponent_types is None:
        opponent_types = ["random", "heuristic", "policy"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize GMO
    ai_config = AIConfig(difficulty=6)
    gmo_config = GMOConfig()
    gmo_ai = GMOAI(player_number=1, config=ai_config, gmo_config=gmo_config)

    if checkpoint_path:
        gmo_ai.load_checkpoint(checkpoint_path)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        default_path = "models/gmo/gmo_best.pt"
        if Path(default_path).exists():
            gmo_ai.load_checkpoint(default_path)
            logger.info(f"Loaded checkpoint: {default_path}")

    # Create dataset
    dataset = DiverseOpponentDataset(
        gmo_ai.state_encoder,
        gmo_ai.move_encoder,
    )

    best_score = 0.0

    for round_num in range(1, rounds + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_num}/{rounds}")
        logger.info(f"{'='*60}")

        # Generate games against diverse opponents
        generate_diverse_games(
            gmo_ai,
            dataset,
            games_per_opponent=games_per_opponent,
            opponent_types=opponent_types,
        )

        logger.info(f"Total samples: {len(dataset)}")

        # Train on accumulated data
        if len(dataset) > 0:
            logger.info(f"Training for {epochs_per_round} epochs...")
            loss = train_on_dataset(
                gmo_ai.state_encoder,
                gmo_ai.move_encoder,
                gmo_ai.value_net,
                dataset,
                epochs=epochs_per_round,
            )
            logger.info(f"Training loss: {loss:.4f}")

        # Evaluate
        logger.info("Evaluating...")
        eval_results = evaluate_model(gmo_ai)
        logger.info(f"Results: vs_random={eval_results['vs_random']:.1f}%, "
                   f"vs_heuristic={eval_results['vs_heuristic']:.1f}%")

        # Save checkpoint
        checkpoint_file = output_path / f"gmo_diverse_round{round_num}.pt"
        gmo_ai.save_checkpoint(checkpoint_file)
        logger.info(f"Saved: {checkpoint_file}")

        # Track best
        score = eval_results['vs_random'] + eval_results['vs_heuristic']
        if score > best_score:
            best_score = score
            best_file = output_path / "gmo_diverse_best.pt"
            gmo_ai.save_checkpoint(best_file)
            logger.info(f"New best! Saved: {best_file}")

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Train GMO against diverse opponents")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--games-per-opponent", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--opponents", nargs="+",
                        default=["random", "heuristic", "policy"])
    parser.add_argument("--output-dir", type=str, default="models/gmo/diverse")

    args = parser.parse_args()

    run_diverse_training(
        checkpoint_path=args.checkpoint,
        rounds=args.rounds,
        games_per_opponent=args.games_per_opponent,
        epochs_per_round=args.epochs,
        opponent_types=args.opponents,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
