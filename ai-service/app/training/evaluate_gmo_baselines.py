"""Evaluate GMO against stronger baselines (MCTS, Gumbel MCTS).

Runs evaluation games to compare GMO performance against stronger AI opponents.

Usage:
    python -m app.training.evaluate_gmo_baselines --games 20
"""

from __future__ import annotations

import argparse
import json
import logging

# Add parent to path for imports
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai.factory import AIFactory
from archive.deprecated_ai.gmo_ai import GMOAI, GMOConfig
from app.game_engine import GameEngine
from app.models import AIConfig, AIType, BoardType
from app.training.train_gmo_selfplay import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def play_game(
    player1,
    player2,
    game_id: str,
    max_moves: int = 600,  # Square8 theoretical max (includes headroom for ~30 turns)
) -> int:
    """Play a game between two AIs.

    Returns:
        Winner player number (1 or 2), or 0 for draw/timeout
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
            # Check for bookkeeping moves
            requirement = GameEngine.get_phase_requirement(state, current_player)
            if requirement is not None:
                move = GameEngine.synthesize_bookkeeping_move(requirement, state)
                if move:
                    state = GameEngine.apply_move(state, move)
                    move_count += 1
                    continue
            break

        ai = player1 if current_player == 1 else player2
        move = ai.select_move(state)

        if move is None:
            requirement = GameEngine.get_phase_requirement(state, current_player)
            if requirement is not None:
                move = GameEngine.synthesize_bookkeeping_move(requirement, state)

        if move is None:
            break

        state = GameEngine.apply_move(state, move)
        move_count += 1

    return state.winner if state.winner else 0


def create_baseline_ai(baseline_type: str, player_number: int):
    """Create a baseline AI with the specified player number."""
    if baseline_type == "mcts":
        config = AIConfig(difficulty=7, think_time=6000, use_neural_net=True)
        return AIFactory.create(AIType.MCTS, player_number=player_number, config=config), "MCTS (D7)"
    elif baseline_type == "gumbel":
        try:
            return AIFactory.create_for_tournament(
                "gumbel_100", player_number=player_number, board_type="square8"
            ), "Gumbel MCTS (100 sims)"
        except Exception as e:
            logger.warning(f"Failed to create Gumbel MCTS: {e}, using MCTS instead")
            config = AIConfig(difficulty=8, think_time=9000, use_neural_net=True)
            return AIFactory.create(AIType.MCTS, player_number=player_number, config=config), "MCTS (D8)"
    elif baseline_type == "policy":
        return AIFactory.create_for_tournament(
            "policy_only", player_number=player_number, board_type="square8"
        ), "Policy Only"
    elif baseline_type == "descent":
        config = AIConfig(difficulty=6, think_time=4500, use_neural_net=True)
        return AIFactory.create(AIType.DESCENT, player_number=player_number, config=config), "Descent (D6)"
    else:
        raise ValueError(f"Unknown baseline: {baseline_type}")


def create_gmo_ai(player_number: int, checkpoint_path: str | None = None) -> GMOAI:
    """Create a GMO AI with the specified player number."""
    ai_config = AIConfig(difficulty=6)
    gmo_config = GMOConfig()
    gmo_ai = GMOAI(player_number=player_number, config=ai_config, gmo_config=gmo_config)

    if checkpoint_path:
        gmo_ai.load_checkpoint(checkpoint_path)
    else:
        default_path = "models/gmo/gmo_best.pt"
        if Path(default_path).exists():
            gmo_ai.load_checkpoint(default_path)

    return gmo_ai


def evaluate_against_baseline(
    checkpoint_path: str | None,
    baseline_type: str,
    num_games: int = 20,
) -> dict:
    """Evaluate GMO against a baseline AI.

    Args:
        checkpoint_path: Path to GMO checkpoint
        baseline_type: Type of baseline ("mcts", "gumbel", "policy", "descent")
        num_games: Number of games to play

    Returns:
        Results dict with win/loss/draw counts
    """
    # Get baseline name
    _, baseline_name = create_baseline_ai(baseline_type, 1)

    logger.info(f"Evaluating GMO vs {baseline_name}: {num_games} games")

    # Play games with GMO as both player 1 and player 2
    wins_as_p1 = 0
    losses_as_p1 = 0
    wins_as_p2 = 0
    losses_as_p2 = 0

    # Half games as player 1 (GMO=P1, baseline=P2)
    games_per_side = num_games // 2
    for i in range(games_per_side):
        game_id = f"gmo_p1_{baseline_type}_{i}"
        # Create fresh AIs with correct player numbers
        gmo_ai = create_gmo_ai(player_number=1, checkpoint_path=checkpoint_path)
        baseline, _ = create_baseline_ai(baseline_type, player_number=2)
        winner = play_game(gmo_ai, baseline, game_id)

        if winner == 1:
            wins_as_p1 += 1
        elif winner == 2:
            losses_as_p1 += 1

        if (i + 1) % 4 == 0:
            logger.info(f"  Game {i+1}/{games_per_side} (as P1): {wins_as_p1}W/{losses_as_p1}L")

    # Half games as player 2 (baseline=P1, GMO=P2)
    for i in range(games_per_side):
        game_id = f"gmo_p2_{baseline_type}_{i}"
        # Create fresh AIs with correct player numbers
        baseline, _ = create_baseline_ai(baseline_type, player_number=1)
        gmo_ai = create_gmo_ai(player_number=2, checkpoint_path=checkpoint_path)
        winner = play_game(baseline, gmo_ai, game_id)

        if winner == 2:
            wins_as_p2 += 1
        elif winner == 1:
            losses_as_p2 += 1

        if (i + 1) % 5 == 0:
            logger.info(f"  Game {i+1}/{games_per_side} (as P2): {wins_as_p2}W/{losses_as_p2}L")

    total_wins = wins_as_p1 + wins_as_p2
    total_losses = losses_as_p1 + losses_as_p2
    total_draws = num_games - total_wins - total_losses
    win_rate = 100 * total_wins / num_games if num_games > 0 else 0

    return {
        "baseline": baseline_name,
        "baseline_type": baseline_type,
        "num_games": num_games,
        "wins": total_wins,
        "losses": total_losses,
        "draws": total_draws,
        "win_rate": win_rate,
        "wins_as_p1": wins_as_p1,
        "losses_as_p1": losses_as_p1,
        "wins_as_p2": wins_as_p2,
        "losses_as_p2": losses_as_p2,
    }


def run_baseline_evaluation(
    checkpoint_path: str | None = None,
    num_games: int = 20,
    baselines: list[str] | None = None,
    output_dir: str = "data/gmo_evaluation"
):
    """Run evaluation against multiple baselines.

    Args:
        checkpoint_path: Path to GMO checkpoint
        num_games: Number of games per baseline
        baselines: List of baseline types to evaluate against
        output_dir: Output directory for results
    """
    if baselines is None:
        baselines = ["policy", "descent", "mcts", "gumbel"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint path
    if checkpoint_path is None:
        default_path = "models/gmo/gmo_best.pt"
        if Path(default_path).exists():
            checkpoint_path = default_path
            logger.info(f"Using default checkpoint: {checkpoint_path}")

    # Evaluate against each baseline
    all_results = []
    for baseline in baselines:
        try:
            result = evaluate_against_baseline(checkpoint_path, baseline, num_games)
            all_results.append(result)
            logger.info(f"\n{result['baseline']}: {result['win_rate']:.1f}% win rate "
                       f"({result['wins']}W/{result['losses']}L/{result['draws']}D)")
        except Exception as e:
            logger.error(f"Failed to evaluate against {baseline}: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for result in all_results:
        logger.info(f"{result['baseline']:20s}: {result['win_rate']:5.1f}% "
                   f"({result['wins']}W/{result['losses']}L/{result['draws']}D)")

    # Save results
    summary = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": checkpoint_path,
        "num_games_per_baseline": num_games,
        "results": all_results,
    }

    results_file = output_path / f"baseline_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    return all_results


def main() -> None:
    """Main entry point for GMO baseline evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate GMO against stronger baselines")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to GMO checkpoint")
    parser.add_argument("--games", type=int, default=20,
                        help="Number of games per baseline")
    parser.add_argument("--baselines", type=str, nargs="+",
                        default=["policy", "descent", "mcts"],
                        help="Baselines to evaluate against")
    parser.add_argument("--output-dir", type=str, default="data/gmo_evaluation",
                        help="Output directory for results")

    args = parser.parse_args()

    run_baseline_evaluation(
        checkpoint_path=args.checkpoint,
        num_games=args.games,
        baselines=args.baselines,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
