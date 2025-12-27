"""Self-play training script for GMO AI.

Generates games using GMO self-play and retrains the model on the results.
Uses the optimal hyperparameters from the hyperparameter sweep.

Usage:
    python -m app.training.train_gmo_selfplay --rounds 5 --games-per-round 100
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..ai.gmo_ai import (
    GMOAI,
    GMOConfig,
    GMOValueNetWithUncertainty,
    MoveEncoder,
    StateEncoder,
    nll_loss_with_uncertainty,
)
from ..ai.heuristic_ai import HeuristicAI
from ..ai.random_ai import RandomAI
from ..game_engine import GameEngine
from ..models import (
    AIConfig,
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


class SelfPlayDataset(Dataset):
    """Dataset that collects games from self-play."""

    def __init__(
        self,
        state_encoder: StateEncoder,
        move_encoder: MoveEncoder,
    ):
        self.state_encoder = state_encoder
        self.move_encoder = move_encoder
        self.samples: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def add_game(self, moves: list[dict], winner: int) -> int:
        """Add a game to the dataset.

        Args:
            moves: List of move dicts from the game
            winner: Winning player number

        Returns:
            Number of samples added
        """
        if winner == 0:
            return 0  # Skip draws

        samples_added = 0
        for i, move_dict in enumerate(moves):
            try:
                if "id" not in move_dict:
                    move_dict = {**move_dict, "id": f"sp_{i}"}

                move = Move.model_validate(move_dict)
                player = move.player

                # Temporal discounting
                move_progress = i / max(len(moves) - 1, 1)
                discount = 0.5 + 0.5 * move_progress
                player_outcome = discount * (1.0 if player == winner else -1.0)

                # Encode move (state encoding is simplified - just use move)
                with torch.no_grad():
                    move_embed = self.move_encoder.encode_move(move)
                    # Use zero state features for simplicity (will be learned)
                    state_features = torch.zeros(self.state_encoder.input_dim)

                self.samples.append((state_features, move_embed, player_outcome))
                samples_added += 1

            except Exception as e:
                logger.debug(f"Error adding move {i}: {e}")
                continue

        return samples_added

    def add_full_game(
        self,
        game_states: list[GameState],
        moves: list[Move],
        winner: int,
    ) -> int:
        """Add a game with full state information.

        Args:
            game_states: List of game states
            moves: List of moves played
            winner: Winning player number

        Returns:
            Number of samples added
        """
        if winner == 0:
            return 0

        samples_added = 0
        for i, (state, move) in enumerate(zip(game_states, moves, strict=False)):
            try:
                player = move.player

                # Temporal discounting
                move_progress = i / max(len(moves) - 1, 1)
                discount = 0.5 + 0.5 * move_progress
                player_outcome = discount * (1.0 if player == winner else -1.0)

                # Encode state and move
                with torch.no_grad():
                    state_features = torch.from_numpy(
                        self.state_encoder.extract_features(state)
                    ).float()
                    move_embed = self.move_encoder.encode_move(move)

                self.samples.append((state_features, move_embed, player_outcome))
                samples_added += 1

            except Exception as e:
                logger.debug(f"Error adding position {i}: {e}")
                continue

        return samples_added

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


def play_game(
    player1,
    player2,
    game_id: str,
    board_type: BoardType = BoardType.SQUARE8,
    max_moves: int = 600,
    collect_states: bool = True,
) -> tuple[int, list[GameState], list[Move]]:
    """Play a game between two AIs.

    Args:
        player1: AI for player 1
        player2: AI for player 2
        game_id: Unique game identifier
        board_type: Board type
        max_moves: Maximum moves before draw
        collect_states: Whether to collect intermediate states

    Returns:
        (winner, states, moves) tuple
    """
    state = create_initial_state(game_id, board_type)

    states = [state] if collect_states else []
    moves = []

    for _ in range(max_moves):
        if state.winner is not None:
            break

        current_player = state.current_player
        ai = player1 if current_player == 1 else player2

        move = ai.select_move(state)
        if move is None:
            break

        moves.append(move)
        state = GameEngine.apply_move(state, move)

        if collect_states:
            states.append(state)

    winner = state.winner if state.winner else 0
    return winner, states[:-1] if states else [], moves  # Exclude final state


def generate_selfplay_games(
    gmo_ai: GMOAI,
    num_games: int,
    dataset: SelfPlayDataset,
    opponent_type: str = "self",
    board_type: BoardType = BoardType.SQUARE8,
    jsonl_path: str | Path | None = None,
) -> dict[str, float]:
    """Generate self-play games and add to dataset.

    Args:
        gmo_ai: The GMO AI instance
        num_games: Number of games to generate
        dataset: Dataset to add samples to
        opponent_type: "self", "random", or "heuristic"
        board_type: Board type for games
        jsonl_path: Optional path to write game records as JSONL

    Returns:
        Statistics dict
    """
    wins = 0
    draws = 0
    total_moves = 0
    samples_added = 0

    # Open JSONL file for logging if path provided
    jsonl_file = None
    if jsonl_path:
        jsonl_path = Path(jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(jsonl_path, "a", encoding="utf-8")
        logger.info(f"Logging games to {jsonl_path}")

    try:
        for game_idx in tqdm(range(num_games), desc=f"Self-play vs {opponent_type}"):
            # Alternate who plays first
            gmo_is_player1 = (game_idx % 2 == 0)

            # Create opponent
            if opponent_type == "self":
                opponent = gmo_ai
            elif opponent_type == "random":
                opponent = RandomAI(2 if gmo_is_player1 else 1, AIConfig(difficulty=1))
            elif opponent_type == "heuristic":
                opponent = HeuristicAI(2 if gmo_is_player1 else 1, AIConfig(difficulty=3))
            else:
                opponent = gmo_ai

            # Set up players
            if gmo_is_player1:
                player1 = gmo_ai
                player2 = opponent
                gmo_player = 1
            else:
                player1 = opponent
                player2 = gmo_ai
                gmo_player = 2

            # Reset AIs
            gmo_ai.reset_for_new_game(rng_seed=game_idx * 1000)
            if hasattr(opponent, "reset_for_new_game") and opponent is not gmo_ai:
                opponent.reset_for_new_game(rng_seed=game_idx * 1000 + 1)

            # Play game
            game_id = f"selfplay_{game_idx}"
            winner, states, moves = play_game(
                player1, player2, game_id, board_type,
                collect_states=True,
            )

            # Record result
            if winner == gmo_player:
                wins += 1
            elif winner == 0:
                draws += 1

            total_moves += len(moves)

            # Add to dataset (all moves, not just GMO's)
            if states and moves:
                samples_added += dataset.add_full_game(states, moves, winner)

            # Log game to JSONL
            if jsonl_file and moves:
                game_record = {
                    "id": f"gmo_selfplay_{uuid.uuid4().hex[:8]}",
                    "board_type": board_type.value,
                    "opponent": opponent_type,
                    "gmo_player": gmo_player,
                    "winner": winner,
                    "num_moves": len(moves),
                    "timestamp": datetime.now().isoformat(),
                    "moves": [
                        {
                            "type": m.type.value,
                            "player": m.player,
                            "from": {"x": m.from_pos.x, "y": m.from_pos.y} if m.from_pos else None,
                            "to": {"x": m.to.x, "y": m.to.y} if m.to else None,
                        }
                        for m in moves
                    ],
                }
                jsonl_file.write(json.dumps(game_record) + "\n")
                jsonl_file.flush()
    finally:
        # Close JSONL file (ensures closure even on exception)
        if jsonl_file:
            jsonl_file.close()
            logger.info(f"Saved {num_games} games to JSONL")

    return {
        "games": num_games,
        "wins": wins,
        "draws": draws,
        "win_rate": wins / num_games if num_games > 0 else 0,
        "avg_game_length": total_moves / num_games if num_games > 0 else 0,
        "samples_added": samples_added,
    }


def train_on_dataset(
    state_encoder: StateEncoder,
    value_net: GMOValueNetWithUncertainty,
    dataset: SelfPlayDataset,
    num_epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train networks on collected data.

    Returns:
        Final training loss
    """
    if len(dataset) == 0:
        return 0.0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = optim.Adam(
        list(state_encoder.parameters()) + list(value_net.parameters()),
        lr=learning_rate,
    )

    state_encoder.train()
    value_net.train()

    final_loss = 0.0
    for _epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for state_features, move_embeds, outcomes in dataloader:
            state_features = state_features.to(device)
            move_embeds = move_embeds.to(device)
            outcomes = outcomes.to(device)

            optimizer.zero_grad()

            # Forward pass
            state_embeds = state_encoder.encoder(state_features)
            pred_values, pred_log_vars = value_net(state_embeds, move_embeds)

            # Loss
            loss = nll_loss_with_uncertainty(pred_values, pred_log_vars, outcomes)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        final_loss = total_loss / max(num_batches, 1)

    return final_loss


def evaluate_vs_baseline(
    gmo_ai: GMOAI,
    opponent_type: str,
    num_games: int = 20,
    board_type: BoardType = BoardType.SQUARE8,
) -> float:
    """Evaluate GMO against a baseline opponent.

    Returns:
        Win rate
    """
    wins = 0

    for game_idx in range(num_games):
        gmo_is_player1 = (game_idx % 2 == 0)
        gmo_player = 1 if gmo_is_player1 else 2
        opp_player = 2 if gmo_is_player1 else 1

        if opponent_type == "random":
            opponent = RandomAI(opp_player, AIConfig(difficulty=1))
        elif opponent_type == "heuristic":
            opponent = HeuristicAI(opp_player, AIConfig(difficulty=3))
        else:
            opponent = RandomAI(opp_player, AIConfig(difficulty=1))

        if gmo_is_player1:
            player1 = gmo_ai
            player2 = opponent
        else:
            player1 = opponent
            player2 = gmo_ai

        gmo_ai.reset_for_new_game(rng_seed=game_idx * 1000)
        opponent.reset_for_new_game(rng_seed=game_idx * 1000 + 1)

        winner, _, _ = play_game(
            player1, player2, f"eval_{game_idx}", board_type,
            collect_states=False,
        )

        if winner == gmo_player:
            wins += 1

    return wins / num_games if num_games > 0 else 0


def run_selfplay_training(
    checkpoint_path: Path,
    output_dir: Path,
    num_rounds: int = 5,
    games_per_round: int = 100,
    epochs_per_round: int = 5,
    opponent_mix: str = "mixed",
    device_str: str = "cpu",
    jsonl_output: Path | None = None,
) -> None:
    """Run self-play training loop.

    Args:
        checkpoint_path: Path to initial GMO checkpoint
        output_dir: Directory for output checkpoints
        num_rounds: Number of self-play/train rounds
        games_per_round: Games to generate per round
        epochs_per_round: Training epochs per round
        opponent_mix: "self", "random", "heuristic", or "mixed"
        device_str: Device to use
        jsonl_output: Optional path to write game records as JSONL
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    logger.info(f"Loading GMO checkpoint from {checkpoint_path}")

    # Create GMO AI with optimal config
    config = GMOConfig(device=device_str)
    gmo_ai = GMOAI(1, AIConfig(difficulty=6), gmo_config=config)
    gmo_ai.load_checkpoint(checkpoint_path)

    # Create dataset
    dataset = SelfPlayDataset(
        state_encoder=gmo_ai.state_encoder,
        move_encoder=gmo_ai.move_encoder,
    )

    # Track progress
    best_vs_random = 0.0
    best_vs_heuristic = 0.0

    # Initial evaluation
    logger.info("Initial evaluation...")
    vs_random = evaluate_vs_baseline(gmo_ai, "random", num_games=20)
    vs_heuristic = evaluate_vs_baseline(gmo_ai, "heuristic", num_games=20)
    logger.info(f"Initial: vs_random={vs_random:.1%}, vs_heuristic={vs_heuristic:.1%}")
    best_vs_random = vs_random
    best_vs_heuristic = vs_heuristic

    for round_idx in range(num_rounds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_idx + 1}/{num_rounds}")
        logger.info(f"{'='*60}")

        # Determine opponents for this round
        if opponent_mix == "mixed":
            # Mix of opponents
            opponents = ["self", "random", "heuristic"]
            games_each = games_per_round // len(opponents)
        else:
            opponents = [opponent_mix]
            games_each = games_per_round

        # Generate games
        round_stats = {"total_samples": 0}
        for opponent in opponents:
            # Generate JSONL path for this round/opponent if output requested
            round_jsonl = None
            if jsonl_output:
                round_jsonl = jsonl_output.parent / f"{jsonl_output.stem}_r{round_idx}_{opponent}.jsonl"

            stats = generate_selfplay_games(
                gmo_ai,
                games_each,
                dataset,
                opponent_type=opponent,
                jsonl_path=round_jsonl,
            )
            logger.info(
                f"  vs {opponent}: win_rate={stats['win_rate']:.1%}, "
                f"samples={stats['samples_added']}"
            )
            round_stats["total_samples"] += stats["samples_added"]

        logger.info(f"Total samples in dataset: {len(dataset)}")

        # Train on collected data
        logger.info(f"Training for {epochs_per_round} epochs...")
        loss = train_on_dataset(
            gmo_ai.state_encoder,
            gmo_ai.value_net,
            dataset,
            num_epochs=epochs_per_round,
            device=device,
        )
        logger.info(f"Training loss: {loss:.4f}")

        # Evaluate
        logger.info("Evaluating...")
        vs_random = evaluate_vs_baseline(gmo_ai, "random", num_games=20)
        vs_heuristic = evaluate_vs_baseline(gmo_ai, "heuristic", num_games=20)
        logger.info(f"Results: vs_random={vs_random:.1%}, vs_heuristic={vs_heuristic:.1%}")

        # Save checkpoint if improved
        improved = False
        if vs_random > best_vs_random:
            best_vs_random = vs_random
            improved = True
        if vs_heuristic > best_vs_heuristic:
            best_vs_heuristic = vs_heuristic
            improved = True

        if improved:
            checkpoint_file = output_dir / f"gmo_selfplay_round{round_idx + 1}.pt"
            gmo_ai.save_checkpoint(checkpoint_file)
            logger.info(f"Saved checkpoint: {checkpoint_file}")

            # Also update best checkpoint
            best_file = output_dir / "gmo_selfplay_best.pt"
            gmo_ai.save_checkpoint(best_file)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("Self-play training complete!")
    logger.info(f"Best vs Random: {best_vs_random:.1%}")
    logger.info(f"Best vs Heuristic: {best_vs_heuristic:.1%}")
    logger.info(f"{'='*60}")


def main() -> None:
    """Main entry point for GMO self-play training."""
    parser = argparse.ArgumentParser(description="GMO Self-play Training")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/gmo/gmo_best.pt"),
        help="Path to initial GMO checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/gmo/selfplay"),
        help="Output directory for checkpoints",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of training rounds")
    parser.add_argument("--games-per-round", type=int, default=100, help="Games per round")
    parser.add_argument("--epochs-per-round", type=int, default=5, help="Training epochs per round")
    parser.add_argument(
        "--opponent-mix",
        type=str,
        default="mixed",
        choices=["self", "random", "heuristic", "mixed"],
        help="Opponent type or mix",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=None,
        help="Path to write game records as JSONL (optional)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_selfplay_training(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_rounds=args.rounds,
        games_per_round=args.games_per_round,
        epochs_per_round=args.epochs_per_round,
        opponent_mix=args.opponent_mix,
        device_str=args.device,
        jsonl_output=args.jsonl_output,
    )


if __name__ == "__main__":
    main()
