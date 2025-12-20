"""Evaluate GMO against stronger baselines (MCTS, Gumbel MCTS).

Runs evaluation games to compare GMO performance against stronger AI opponents.

Usage:
    python -m app.training.evaluate_gmo_baselines --games 20
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Add parent to path for imports
import sys
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


def play_game(
    player1,
    player2,
    game_id: str,
    max_moves: int = 500,
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


def evaluate_against_baseline(
    gmo_ai: GMOAI,
    baseline_type: str,
    num_games: int = 20,
) -> Dict:
    """Evaluate GMO against a baseline AI.

    Args:
        gmo_ai: The GMO AI instance
        baseline_type: Type of baseline ("mcts", "gumbel", "policy", "descent")
        num_games: Number of games to play

    Returns:
        Results dict with win/loss/draw counts
    """
    # Create baseline AI
    if baseline_type == "mcts":
        config = AIConfig(difficulty=7, think_time=6000, use_neural_net=True)
        baseline = AIFactory.create(AIType.MCTS, player_number=2, config=config)
        baseline_name = "MCTS (D7)"
    elif baseline_type == "gumbel":
        try:
            config = AIConfig(difficulty=9, gumbel_simulation_budget=100)
            baseline = AIFactory.create_for_tournament(
                "gumbel_100", player_number=2, board_type="square8"
            )
            baseline_name = "Gumbel MCTS (100 sims)"
        except Exception as e:
            logger.warning(f"Failed to create Gumbel MCTS: {e}, using MCTS instead")
            config = AIConfig(difficulty=8, think_time=9000, use_neural_net=True)
            baseline = AIFactory.create(AIType.MCTS, player_number=2, config=config)
            baseline_name = "MCTS (D8)"
    elif baseline_type == "policy":
        baseline = AIFactory.create_for_tournament(
            "policy_only", player_number=2, board_type="square8"
        )
        baseline_name = "Policy Only"
    elif baseline_type == "descent":
        config = AIConfig(difficulty=6, think_time=4500, use_neural_net=True)
        baseline = AIFactory.create(AIType.DESCENT, player_number=2, config=config)
        baseline_name = "Descent (D6)"
    else:
        raise ValueError(f"Unknown baseline: {baseline_type}")

    logger.info(f"Evaluating GMO vs {baseline_name}: {num_games} games")

    # Play games with GMO as both player 1 and player 2
    wins_as_p1 = 0
    losses_as_p1 = 0
    wins_as_p2 = 0
    losses_as_p2 = 0

    # Half games as player 1
    games_per_side = num_games // 2
    for i in range(games_per_side):
        game_id = f"gmo_p1_{baseline_type}_{i}"
        gmo_ai.reset_for_new_game()
        baseline.reset_for_new_game()
        winner = play_game(gmo_ai, baseline, game_id)

        if winner == 1:
            wins_as_p1 += 1
        elif winner == 2:
            losses_as_p1 += 1

        if (i + 1) % 5 == 0:
            logger.info(f"  Game {i+1}/{games_per_side} (as P1): {wins_as_p1}W/{losses_as_p1}L")

    # Half games as player 2
    for i in range(games_per_side):
        game_id = f"gmo_p2_{baseline_type}_{i}"
        gmo_ai.reset_for_new_game()
        baseline.reset_for_new_game()
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
    checkpoint_path: Optional[str] = None,
    num_games: int = 20,
    baselines: List[str] = None,
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

    # Load GMO AI
    ai_config = AIConfig(difficulty=6)
    gmo_config = GMOConfig()
    gmo_ai = GMOAI(player_number=1, config=ai_config, gmo_config=gmo_config)

    if checkpoint_path:
        gmo_ai.load_checkpoint(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        default_path = "models/gmo/gmo_best.pt"
        if Path(default_path).exists():
            gmo_ai.load_checkpoint(default_path)
            logger.info(f"Loaded checkpoint from {default_path}")

    # Evaluate against each baseline
    all_results = []
    for baseline in baselines:
        try:
            result = evaluate_against_baseline(gmo_ai, baseline, num_games)
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


def main():
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
