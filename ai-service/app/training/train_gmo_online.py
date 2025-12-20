"""Train GMO using online/continuous learning.

Unlike batch training, this script updates the model after each game,
allowing the AI to learn continuously during play.

Usage:
    python -m app.training.train_gmo_online --games 100 --opponents random,heuristic
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai.factory import AIFactory
from app.ai.gmo_ai import GMOAI, GMOConfig
from app.game_engine import GameEngine
from app.models import AIConfig, AIType, BoardType
from app.training.train_gmo_selfplay import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def play_game_with_learning(
    gmo_ai: GMOAI,
    opponent,
    game_id: str,
    gmo_player: int = 1,
    max_moves: int = 500,
) -> tuple[int, int]:
    """Play a game with online learning enabled.

    Args:
        gmo_ai: GMO AI with online learning enabled
        opponent: Opponent AI
        game_id: Game identifier
        gmo_player: Which player GMO is (1 or 2)
        max_moves: Maximum moves before timeout

    Returns:
        (winner, move_count)
    """
    state = create_initial_state(
        game_id=game_id,
        board_type=BoardType.SQUARE8,
        rng_seed=hash(game_id) % (2**31),
    )

    move_count = 0
    while state.game_status.value == "active" and move_count < max_moves:
        current_player = state.current_player
        legal_moves = GameEngine.get_valid_moves(state, current_player)

        if not legal_moves:
            requirement = GameEngine.get_phase_requirement(state, current_player)
            if requirement is not None:
                move = GameEngine.synthesize_bookkeeping_move(requirement, state)
                if move:
                    state = GameEngine.apply_move(state, move)
                    move_count += 1
                    continue
            break

        if current_player == gmo_player:
            # Use learning-enabled move selection
            move = gmo_ai.select_move_with_learning(state)
        else:
            move = opponent.select_move(state)

        if move is None:
            requirement = GameEngine.get_phase_requirement(state, current_player)
            if requirement is not None:
                move = GameEngine.synthesize_bookkeeping_move(requirement, state)

        if move is None:
            break

        state = GameEngine.apply_move(state, move)
        move_count += 1

    winner = state.winner if state.winner else 0
    return winner, move_count


def run_online_training(
    checkpoint_path: str | None = None,
    num_games: int = 100,
    opponents: list[str] | None = None,
    lr: float = 0.0001,  # Lowered from 0.001 to prevent catastrophic forgetting
    buffer_size: int = 500,  # Increased from 100 for more diverse replay
    weight_decay: float = 0.01,  # L2 regularization
    max_grad_norm: float = 1.0,  # Gradient clipping threshold
    save_every: int = 20,
    output_dir: str = "models/gmo",
) -> dict:
    """Run online/continuous learning training.

    Uses regularization to prevent catastrophic forgetting:
    - Lower learning rate (0.0001 default)
    - Larger replay buffer (500 default)
    - Weight decay (L2 regularization)
    - Gradient clipping

    Args:
        checkpoint_path: Path to load initial checkpoint
        num_games: Total games to play
        opponents: List of opponent types
        lr: Learning rate for online updates
        buffer_size: Experience replay buffer size
        weight_decay: L2 regularization strength
        max_grad_norm: Maximum gradient norm for clipping
        save_every: Save checkpoint every N games
        output_dir: Output directory for checkpoints

    Returns:
        Training statistics
    """
    if opponents is None:
        opponents = ["random", "heuristic"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create GMO AI with online learning
    ai_config = AIConfig(difficulty=6)
    gmo_config = GMOConfig()
    gmo_ai = GMOAI(player_number=1, config=ai_config, gmo_config=gmo_config)

    # Load checkpoint if exists
    if checkpoint_path:
        gmo_ai.load_checkpoint(checkpoint_path)
    else:
        default_path = output_path / "gmo_best.pt"
        if default_path.exists():
            gmo_ai.load_checkpoint(default_path)
            logger.info(f"Loaded checkpoint from {default_path}")

    # Enable online learning with regularization
    gmo_ai.enable_online_learning(
        lr=lr,
        buffer_size=buffer_size,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
    )

    # Create opponent pool
    opponent_pool = []
    for opp_type in opponents:
        if opp_type == "random":
            opp = AIFactory.create(AIType.RANDOM, player_number=2, config=AIConfig(difficulty=1))
        elif opp_type == "heuristic":
            opp = AIFactory.create(AIType.HEURISTIC, player_number=2, config=AIConfig(difficulty=4))
        elif opp_type == "policy":
            try:
                opp = AIFactory.create_for_tournament("policy_only", player_number=2, board_type="square8")
            except Exception:
                logger.warning("Failed to create policy opponent, using heuristic")
                opp = AIFactory.create(AIType.HEURISTIC, player_number=2, config=AIConfig(difficulty=5))
        else:
            logger.warning(f"Unknown opponent type: {opp_type}")
            continue
        opponent_pool.append((opp_type, opp))

    if not opponent_pool:
        opponent_pool.append(("random", AIFactory.create(
            AIType.RANDOM, player_number=2, config=AIConfig(difficulty=1)
        )))

    logger.info(f"Online training: {num_games} games vs {[o[0] for o in opponent_pool]}")
    logger.info(f"lr={lr}, buffer={buffer_size}, weight_decay={weight_decay}, grad_clip={max_grad_norm}")

    # Training loop
    stats = {
        "games_played": 0,
        "wins": 0,
        "losses": 0,
        "total_loss": 0.0,
        "updates": 0,
        "by_opponent": {name: {"wins": 0, "losses": 0} for name, _ in opponent_pool},
    }

    import random
    for game_num in range(num_games):
        # Select random opponent
        opp_name, opponent = random.choice(opponent_pool)

        # Alternate between playing as P1 and P2
        gmo_player = 1 if game_num % 2 == 0 else 2

        # Create fresh opponent with correct player number
        opp_player = 2 if gmo_player == 1 else 1
        if opp_name == "random":
            opponent = AIFactory.create(AIType.RANDOM, player_number=opp_player, config=AIConfig(difficulty=1))
        elif opp_name == "heuristic":
            opponent = AIFactory.create(AIType.HEURISTIC, player_number=opp_player, config=AIConfig(difficulty=4))
        elif opp_name == "policy":
            try:
                opponent = AIFactory.create_for_tournament("policy_only", player_number=opp_player, board_type="square8")
            except Exception:
                opponent = AIFactory.create(AIType.HEURISTIC, player_number=opp_player, config=AIConfig(difficulty=5))

        game_id = f"online_{game_num}_{opp_name}"

        # Reset GMO for new game
        gmo_ai.reset_for_new_game()
        # Update player number for this game
        gmo_ai.player_number = gmo_player

        # Play game
        winner, _move_count = play_game_with_learning(
            gmo_ai, opponent, game_id, gmo_player=gmo_player
        )

        # Determine outcome from GMO's perspective
        if winner == gmo_player:
            outcome = 1.0
            stats["wins"] += 1
            stats["by_opponent"][opp_name]["wins"] += 1
        elif winner == 0:
            outcome = 0.0  # Shouldn't happen
        else:
            outcome = -1.0
            stats["losses"] += 1
            stats["by_opponent"][opp_name]["losses"] += 1

        # Update model based on game outcome
        loss = gmo_ai.update_on_game_end(outcome)
        stats["total_loss"] += loss
        stats["updates"] += 1
        stats["games_played"] += 1

        # Log progress
        if (game_num + 1) % 10 == 0:
            win_rate = 100 * stats["wins"] / stats["games_played"]
            avg_loss = stats["total_loss"] / stats["updates"]
            learning_stats = gmo_ai.get_learning_stats()
            buffer_fill = learning_stats["buffer_size"] if learning_stats else 0

            logger.info(
                f"Game {game_num + 1}/{num_games}: "
                f"{stats['wins']}W/{stats['losses']}L ({win_rate:.1f}%) | "
                f"loss={avg_loss:.4f} | buffer={buffer_fill}"
            )

        # Save checkpoint periodically
        if (game_num + 1) % save_every == 0:
            checkpoint_file = output_path / f"gmo_online_{game_num + 1}.pt"
            gmo_ai.save_checkpoint(checkpoint_file)
            logger.info(f"Saved checkpoint: {checkpoint_file}")

    # Final save
    final_checkpoint = output_path / "gmo_online_final.pt"
    gmo_ai.save_checkpoint(final_checkpoint)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ONLINE TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Games: {stats['games_played']}")
    logger.info(f"Win rate: {100 * stats['wins'] / stats['games_played']:.1f}%")
    logger.info(f"Average loss: {stats['total_loss'] / max(cast(int, stats['updates']), 1):.4f}")

    logger.info("\nBy opponent:")
    for opp_name, opp_stats in stats["by_opponent"].items():
        total = opp_stats["wins"] + opp_stats["losses"]
        if total > 0:
            wr = 100 * opp_stats["wins"] / total
            logger.info(f"  {opp_name}: {opp_stats['wins']}W/{opp_stats['losses']}L ({wr:.1f}%)")

    logger.info(f"\nFinal checkpoint: {final_checkpoint}")

    # Save stats
    stats_file = output_path / f"online_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Train GMO with online learning")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to initial checkpoint")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games to play")
    parser.add_argument("--opponents", type=str, default="random,heuristic",
                        help="Comma-separated list of opponents")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate (lowered to prevent forgetting)")
    parser.add_argument("--buffer", type=int, default=500,
                        help="Replay buffer size (increased for stability)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="L2 regularization strength")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save checkpoint every N games")
    parser.add_argument("--output-dir", type=str, default="models/gmo",
                        help="Output directory")

    args = parser.parse_args()

    run_online_training(
        checkpoint_path=args.checkpoint,
        num_games=args.games,
        opponents=args.opponents.split(","),
        lr=args.lr,
        buffer_size=args.buffer,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        save_every=args.save_every,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
